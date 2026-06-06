# Bedrock Connectivity & Transient-Failure Debug Guide

Long-running `lits-search` / `lits-chain` experiments (hours to days) make tens
of thousands of Bedrock Converse API calls. Over that many calls, transient
network and AWS-side failures are statistically inevitable. This guide catalogs
the failure modes we have actually hit, how to tell them apart, and how the
client is hardened against them.

It exists because during one KGQA MCTS run (Qwen3-Coder-Next policy, ~69
examples × ~5 MCTS iters × multi-step rollouts) we hit **five distinct error
signatures over two days**, and it was easy to confuse them. Each section
records the exact error string, the verified mechanism, and the fix.

> **Golden rule when triaging:** read the actual logs and process state before
> theorizing. During the original incident, two "obvious" explanations (laptop
> sleep, IPv6 routing) turned out to be wrong when checked against `pmset -g log`
> and `curl -6`. See "Verification commands" at the bottom — run those first.

---

## The Bedrock client config (current hardening)

`lits/lm/bedrock_chat.py::BedrockChatModel.__init__`:

```python
config=BotoConfig(
    connect_timeout=10,   # fail fast on a stalled TCP/TLS connect (SYN_SENT)
    read_timeout=300,     # 5 min — long code generation can exceed 60s default
    retries={"max_attempts": 5, "mode": "adaptive"},
)
```

Each parameter maps to a failure mode below:

| Parameter | Guards against | Section |
|---|---|---|
| `read_timeout=300` | request sent, response never arrives | #1 ReadTimeout |
| `retries` (adaptive) | transient throttling / 5xx | #2 InternalServer |
| `connect_timeout=10` | connection stalls in `SYN_SENT`, holding a signed request | #4 SYN_SENT hang, #5 Signature expired |

`connect_timeout` + adaptive retries together are the key pairing: a stalled
connect now fails in 10s and the retry re-issues the call, which **re-signs the
request with a fresh SigV4 timestamp** rather than letting a stale one expire.

---

## Failure mode catalog

### #1 — `ReadTimeoutError`

```
botocore.exceptions.ReadTimeoutError: Read timeout on endpoint URL:
"https://bedrock-runtime.us-east-1.amazonaws.com/model/<model>/converse"
```

**Mechanism:** The TCP/TLS connection was established and the request was sent,
but Bedrock did not return a response within `read_timeout` (300s). Usually a
genuinely slow generation or a momentary AWS-side delay.

**Layer:** policy/evaluator LLM call (not a tool backend).

**Handled?** Yes — adaptive retries re-issue the call. A *single* ReadTimeout no
longer crashes the run. If it survives all 5 retries, the endpoint is genuinely
unhealthy for an extended window → run aborts; just resume later.

---

### #2 — `InternalServerException (reached max retries: N)`

```
An error occurred (InternalServerException) when calling the Converse operation
(reached max retries: 4): The system encountered an unexpected error during
processing. Try your request again.
```

**Mechanism:** Bedrock's *server* returned HTTP 500 on every retry. `(reached
max retries: 4)` confirms the adaptive-retry machinery fired and exhausted its
attempts — i.e. the retry fix is working, but AWS had a sustained internal
incident during that window.

**Layer:** policy/evaluator LLM call.

**Handled?** Partially — retried 5×, but a multi-minute AWS incident outlasts
that. **Not a code bug.** Wait a few minutes, then resume. More retries would
just burn wall-clock against a down endpoint.

---

### #3 — `ToolServerDownError: ... URLError` (KGQA / SPARQL)

```
lits.tools.utils.ToolServerDownError: Tool 'get_relations' backend appears down:
RuntimeError: SPARQL query failed (URLError). Query: PREFIX rdf: ...
```

**Mechanism:** This is a **tool backend** failure, NOT a Bedrock failure. For
KGQA the SPARQL tool talks to a Freebase Virtuoso server over an SSH tunnel
(`localhost:3001`). `URLError` = transport-layer failure: the connection itself
could not be made, i.e. **the tunnel dropped.** The circuit breaker correctly
classifies this as server-down and aborts after 3 consecutive failures.

**Layer:** tool backend (SPARQL over SSH tunnel).

**Handled?** Working as designed — a real `URLError` *should* trip the breaker.
The fix is operational, not code: keep the tunnel alive (autossh with
keepalive) and check it before resuming. See "Verification commands".

**Distinguish from false positives:** A *reachable* SPARQL server returning a
query error (HTTP 500 with `S0022 Error SQ200`, or HTTP 400 malformed) is NOT
server-down — those are handled in `CIRCUIT_BREAKER.md` Gotchas 1/3. `URLError`
with no HTTP code = genuinely unreachable = correctly trips.

---

### #4 — Process hangs (no error, no progress) — stalled `SYN_SENT`

**Symptom:** The run appears frozen. No new log lines, no crash. CPU ~0%.

**Mechanism:** A Bedrock connection stalled in TCP handshake. `lsof` on the
process shows a socket in `SYN_SENT` (SYN sent, no SYN-ACK received). The OS
keeps retransmitting SYN with exponential backoff (macOS: up to ~75s+ before
giving up), during which the process blocks with no timeout. In the original
incident the run was frozen ~13 minutes before recovering.

**How it "recovers" on its own:** either (a) a later SYN retransmit gets through
when the network recovers, or (b) the OS connect-timeout fires, botocore sees a
connection error and opens a *fresh* connection. Both are normal TCP behavior,
not a revived dead socket.

**Layer:** Bedrock connect phase.

**Handled?** Now yes — `connect_timeout=10` caps the stall at 10s, after which
adaptive retry reconnects. Before this fix there was no connect timeout, only a
read timeout (which does not cover the connect phase).

**How to confirm it's this and not a slow generation:**
```bash
PID=$(pgrep -f lits-search | head -1)
lsof -p "$PID" 2>/dev/null | grep -iE "TCP|SYN_SENT|ESTABLISHED"
```
`SYN_SENT` → stalled connect (this mode). `ESTABLISHED` → connection up, likely
just a slow/long generation (wait, don't kill).

---

### #5 — `InvalidSignatureException: Signature expired`

```
An error occurred (InvalidSignatureException) when calling the Converse
operation: Signature expired: 20260605T155808Z is now earlier than
20260605T155819Z (20260605T160319Z - 5 min.)
```

**Mechanism:** AWS SigV4 signs each request with a timestamp and rejects it if
the signature is older than 5 minutes when it *arrives*. The error string gives
the timeline: signed at `155808Z`, AWS "now" is `160319Z` → the request took
**>5 minutes to travel from signing to arrival.** botocore signs *before* it
connects, so a connection that stalls (mode #4) holds an already-signed request;
if the stall exceeds 5 minutes, the signature is dead on arrival.

`InvalidSignatureException` is a `ClientError` that botocore does **not** retry,
so unlike #1/#2 it crashes immediately.

**Layer:** Bedrock connect/send phase.

**Handled?** Now yes — `connect_timeout=10` prevents the >5min stall; the retry
re-signs with a fresh timestamp. Before the fix, this was the crash that ended
the run silently overnight.

**Two hypotheses we RULED OUT (don't repeat them without checking):**
- *Clock skew on the local machine.* Checked: `date -u` matched AWS `Date`
  header to the second; `sntp` offset was +0.15s. Not skew.
- *Laptop sleep froze the process between signing and sending.* Checked:
  `pmset -g log` showed **no Sleep/Wake transitions** in the crash window. The
  machine did not sleep. (The trigger of the >5min stall was therefore most
  likely a network stall, mode #4 — but this was never positively confirmed.)

---

## Quick triage decision tree

```
Run stopped / misbehaving
├─ No error, no new log lines, CPU ~0%        → #4 SYN_SENT hang
│    └─ lsof shows SYN_SENT? confirm. ESTABLISHED? it's just slow, wait.
├─ Error mentions a TOOL ('get_relations', 'shell', 'sql_query')
│    ├─ URLError / connection refused          → #3 tunnel/backend down (REAL)
│    ├─ HTTP 500 with S0022/Virtuoso/SPARQL    → query error, see CIRCUIT_BREAKER.md Gotcha 3
│    └─ HTTP 404/wget/curl output in 'shell'   → false positive, see CIRCUIT_BREAKER.md Gotcha 4
└─ Error mentions Bedrock Converse
     ├─ ReadTimeoutError                        → #1 (retried; resume if it aborted)
     ├─ InternalServerException (max retries)   → #2 AWS incident (wait, resume)
     └─ InvalidSignatureException: expired      → #5 (connect_timeout fix; check stall)
```

---

## Verification commands (run these FIRST, before theorizing)

```bash
RD=<run_dir>   # e.g. .../qwen.qwen3-coder-next_results/kgqa_mcts/run_v0.3.2

# 1. Is the run actually frozen, or just slow? Compare log mtime to now.
date "+%Y-%m-%d %H:%M:%S"; stat -f "%Sm" "$RD/execution.log"

# 2. Last real LLM activity (authoritative — execution.log mtime can mislead).
tail -1 "$RD/inferencelogger.log" | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print(r['role'], r['timestamp'])"

# 3. What is a hung process waiting on?
PID=$(pgrep -f lits-search | head -1); lsof -p "$PID" 2>/dev/null | grep -iE "TCP|SYN_SENT|ESTABLISHED"

# 4. Local clock vs AWS clock (rule out SigV4 skew).
date -u "+%Y%m%dT%H%M%SZ"; curl -sI --max-time 10 https://bedrock-runtime.us-east-1.amazonaws.com | grep -i "^date:"

# 5. Did the machine sleep? (rule out sleep hypothesis)
pmset -g log 2>/dev/null | grep -E "$(date +%Y-%m-%d)" | grep -iE "Entering Sleep|Wake from|DarkWake"

# 6. Bedrock reachable now? IPv4 and IPv6 separately (rule out IPv6-only breakage).
curl -4 -s -o /dev/null -w "v4 connect=%{time_connect}s code=%{http_code}\n" --max-time 10 https://bedrock-runtime.us-east-1.amazonaws.com
curl -6 -s -o /dev/null -w "v6 connect=%{time_connect}s code=%{http_code}\n" --max-time 10 https://bedrock-runtime.us-east-1.amazonaws.com
# (HTTP 404 is fine here — it means TCP+TLS succeeded, which is all we're testing.)

# 7. (KGQA only) Is the SPARQL tunnel up and the server answering?
lsof -nP -iTCP:3001 -sTCP:LISTEN
curl -s --max-time 8 "http://localhost:3001/sparql?query=SELECT+%3Fs+WHERE+%7B+%3Fs+%3Fp+%3Fo+%7D+LIMIT+1" | head -3
```

---

## Operational mitigations for long runs

Beyond the client config, two operational practices prevent most of the above:

1. **Prevent laptop sleep.** Wrap the run in `caffeinate -i` so idle sleep can't
   freeze the process mid-flight:
   ```bash
   caffeinate -i lits-search ... --output-dir ...
   ```
   (Sleep was ruled out for the original incident, but it remains a real risk
   for unattended overnight runs.)

2. **Keep the SPARQL tunnel resilient** (KGQA). Use autossh with keepalive so a
   dropped tunnel auto-reconnects:
   ```bash
   autossh -M 0 -fN -o "ServerAliveInterval=30" -o "ServerAliveCountMax=3" \
     -o "ExitOnForwardFailure=yes" -L 3001:localhost:3001 \
     -i ~/.ssh/<key>.pem ubuntu@<freebase-host>
   ```
   There is still a ~90s window between drop and reconnect where calls fail; on
   a ~20h run these drops are likely, so expect occasional #3 aborts and resume.

---

## Resuming after any of these

All of the above leave the run resumable:

- **`lits-search`** (MCTS/BFS): run `lits-resume-clean --result-dir <RD>` to
  detect/clean any half-finished example, then re-run the original command with
  the **same `--output-dir`**. See `docs/cli/resume.md`.
- **`lits-chain`** without verifier (DBBench/KGQA): a killed attempt's checkpoint
  is silently treated as complete on rerun — manual `_stale` rename needed. See
  `resume.md` Procedure 2.
- **`lits-chain`** with verifier (Terminal-Bench): just rerun; missing
  `_reward.json` auto-triggers re-run of the incomplete attempt.

Always run the verification commands above before resuming, so you resume into a
healthy environment rather than straight back into the same failure.
