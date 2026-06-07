# Tool Backend Failure Handling

How LiTS handles a tool whose backend misbehaves — a network endpoint that
drops, a query the server rejects, a container that dies. This is the single
canonical reference for tool-backend failures.

LiTS does one **judgment** and then runs two **response mechanisms** gated by it:

- **The gate — classification** (`lits/tools/utils.py::_classify_as_server_down`):
  is this failure a backend-down event, or just an agent/query error to hand back
  to the LLM? This is a predicate, not a handler — but it decides everything
  downstream, because *only* failures judged server-down feed the two mechanisms
  below. A misjudgment here either burns the run on agent bugs (false positive)
  or ignores a real outage (false negative), so most of the subtlety lives in its
  gotchas.
- **Mechanism 1 — retry-with-backoff** (`execute_tool_action`, per call): ride
  out a *transient* drop within one call so it never escalates.
- **Mechanism 2 — circuit breaker** (`lits/components/transition/tool_use.py`,
  per run): after enough *confirmed* failures across the run, abort instead of
  burning LLM budget.

Backend recovery itself (reconnecting an SSH tunnel, preventing host sleep,
re-launching a crashed run) is **not** LiTS's job — it is operational/deployment
work the user owns. LiTS's only stake in it is a **timing constraint**: because
of the retry budget above, your recovery tooling must bring the backend back
within a bounded window or the retry is wasted and the run still aborts. The
last section spells out that constraint and what it means for tunnel keepalive
and resume.

> Scope: this is the **tool backend** layer. Bedrock LLM-call failures
> (ReadTimeout, InternalServerException, SYN_SENT hang, Signature expired) are a
> different layer — see `docs/lm/bedrock_connectivity.md`. A quick tell: a
> tool-backend error names a *tool* (`get_relations`, `sql_query`, `shell`) and
> `lits.tools.*`; a Bedrock error names `bedrock_chat` / Converse.

Throughout, **KGQA over a Freebase SPARQL endpoint** is used as the worked
example, because that is where these failures were first hit. The mechanisms are
general: they apply to any network-backed tool (SQL, web, PDF) and any
connection shim (SSH tunnel, VPN, load balancer).

---

## How the gate and the two mechanisms fit together

```
execute_tool_action
  ├─ classify: server-down?  ──no──▶ return error observation to the agent (counter unchanged)
  │                          ──yes─▶ retry-with-backoff (ride out transient blip)
  │                                     ├─ recovered ▶ return clean observation (counter reset)
  │                                     └─ exhausted ▶ raise ToolServerDownError
  ▼
ToolUseTransition.step
  └─ count consecutive ToolServerDownError across calls; abort run at threshold (default 3)
```

Classification is the **gate**; retry and the breaker are the two **responses**
it feeds. They differ in lifetime and job:

| | Classification (gate) | Retry-with-backoff (mechanism 1) | Circuit breaker (mechanism 2) |
|---|---|---|---|
| Kind | judgment / predicate | response, per call | response, per run |
| Lives in | `_classify_as_server_down` | `execute_tool_action` | `ToolUseTransition.step` |
| Question | is this backend-down? | can I ride out this blip? | should I abort the run? |
| Lifetime | one exception | one tool call | across all examples |
| Effect | route to a mechanism vs the agent | recover silently, or escalate | abort, or keep going |

---

## 1. Classification — what counts as "server-down"

`execute_tool_action` sorts a tool failure into three buckets:

| Bucket | Examples | Behavior |
|---|---|---|
| **Server-down** | `urllib.error.URLError("connection refused")`, `botocore.exceptions.EndpointConnectionError`, `psycopg2.OperationalError("could not connect")` | Wrapped as `ToolServerDownError`. Enters retry, then (if exhausted) the breaker. |
| **Bad action / agent error** | `ValueError("variable must be a relation")`, `sqlite3.OperationalError("no such table: foo")` | Returned to the agent as `Error executing tool. Error report: ...`. Counter unchanged. |
| **HTTP 4xx (client error)** | `urllib.error.HTTPError(code=400)` from a malformed SPARQL query | **Excluded** from server-down (see Gotcha 1). Falls through to bucket 2. |
| **HTTP 5xx (server error)** | `urllib.error.HTTPError(code=503)` | Server-down — *unless* the body carries an application/query-level marker (see Gotcha 3). |

`_classify_as_server_down` walks the `__cause__` / `__context__` chain so an
exception wrapping a network failure underneath (e.g. `BackendError("query
failed") from socket.timeout`) is still caught.

### Gotcha 1: HTTP 4xx looks like a network error if you squint

`urllib.error.HTTPError` subclasses `URLError`. Without a code-range guard, a
Virtuoso `HTTP 400 Bad Request` on a malformed SPARQL query would walk up to
`URLError` in the chain-walker and trip the breaker — even though the server is
healthy and the bug is on the LLM-policy side.

**Symptom (first observed 2026-05-22):** KGQA MCTS Reflection died at example 93
after the agent generated a chained `argmax`-after-`argmax` query producing
invalid SPARQL with two consecutive `ORDER BY DESC(?sk0) LIMIT 1` clauses.
Virtuoso returned HTTP 400; `SPARQLWrapper` raised `QueryBadFormed` chained from
`urllib.error.HTTPError(code=400)`. The classifier saw `URLError` in the chain
and incorrectly tripped after 3 consecutive malformed-query attempts.

**Fix:** `_classify_as_server_down` checks `HTTPError.code` and skips 4xx; 5xx
remains server-down. Tests: `test_tool_server_down_classify.py` cases i (4xx) and
j (5xx).

**Takeaway:** when adding exception types to the classifier, ask "is this
strictly a *transport-layer* failure?" HTTP-protocol errors need code-range
checks; only treat them as server-down when the status indicates real
impairment.

### Gotcha 3: not every HTTP 500 means the server is down

Some backends return HTTP 500 for *application/query-level* failures while the
server is healthy. The canonical case is Virtuoso's SQL planner: a
valid-but-complex SPARQL query can exceed its query-compiler limits and come
back as `HTTP 500` with a body like `Virtuoso S0022 Error SQ200: No column
s_18_9.x`. SPARQLWrapper wraps this as `EndPointInternalError` with the
underlying `urllib.error.HTTPError(code=500)` in the chain.

**Symptom (first observed 2026-06-01):** KGQA MCTS (Qwen3-Coder-Next) died at
example 11. The agent repeatedly issued `get_attributes` on a numeric variable
`#6`; MCTS branches converged on the same query, so the identical Virtuoso 500
fired three times in a row with no intervening success. The breaker counted three
consecutive "server-down" events and aborted — though the endpoint was reachable
and answering other queries fine.

**Fix:** an HTTP 5xx is treated as a *query error* (not server-down) when the
exception chain text matches `_HTTP_5XX_APPLICATION_ERROR_MARKERS` (e.g. `s0022`,
`sq200`, `virtuoso`, `sparql query:`, `syntax error`, `parse error`, `bad
formed`, `malformed`). The scan walks the whole chain because the status code
lives on the inner `HTTPError` while the descriptive body lives on the outer
wrapper. Bare 5xx with no marker (a plain 502/503) remains server-down. Tests:
cases k (Virtuoso 500 → query error) and l (bare 500 → server-down).

**Takeaway:** 5xx is a *hint*, not proof, of impairment. When the body identifies
an application/query failure, hand it back to the agent so it can try a different
query — don't trip the breaker.

### Gotcha 4: a shell tool's output is not a backend-health signal

The string-return path (`_classify_result_as_server_down`) scans a tool's
returned string for network markers + connection vocabulary. Correct for
network-backed tools that *stringify* a connection error instead of raising
(some SQL/SPARQL wrappers). **Wrong** for a tool whose string output is arbitrary
task content — most notably the Terminal-Bench `shell` tool.

**Symptom (first observed 2026-06-04):** an agent ran `wget
http://www.povray.org/.../povray-2.2.tar.gz`; the site returned HTTP 404. The
output contained both `Connecting ... connected` and `ERROR 404: Not Found`. The
classifier matched `"connect"` + `"error"` and raised `ToolServerDownError`,
tripping the breaker — though the Docker backend was healthy and the 404 was
normal task content.

**Fix:** `BaseTool.classify_string_result_as_server_down` (default `True`,
preserving SQL/SPARQL behavior). `ShellTool` sets it `False`, so
`execute_tool_action` skips string-return classification for shell output. A
genuinely dead container still surfaces as a raised exception from `exec_sync()`,
classified normally. Tests: cases m (opt-out respected) and n (same output
without opt-out still trips).

**Takeaway:** the string-return heuristic is a last-resort net for tools that
hide network errors in their return value. Tools whose output is free-form task
content (shells, code interpreters) must opt out — their output will contain
connection/error words as a matter of course.

---

## 2. Retry-with-backoff — ride out a transient drop

### The incident that motivated it

A KGQA MCTS run (`qwen.qwen3-coder-next`, `kgqa_mcts_sibling_aware/run_v0.3.2`)
aborted on example 38. Measured from `execution.log`:

| Event | Timestamp |
|---|---|
| Last healthy tool call | `09:51:35` |
| Server-down attempt 1/3 | `10:11:24.728` |
| Server-down attempt 2/3 | `10:11:32.584` |
| Server-down attempt 3/3 → **TRIPPED** | `10:11:33.303` |

The three failures that tripped the breaker spanned **~8.6 seconds**. The tunnel
had dropped, and the first three tool calls after it dropped all failed back to
back with no successful call in between to reset the counter.

**Key takeaway:** tuning autossh keepalive cannot prevent this. Even an
aggressive `ServerAliveInterval=15, ServerAliveCountMax=2` detects a drop in ~30s
and only *then* starts reconnecting — far slower than the 9s it took to trip. The
calls fail faster than any tunnel can recover. The fix has to absorb the
transient *in code*.

### How it works

When a single tool call is classified as server-down, instead of raising
immediately, `execute_tool_action` re-attempts the *same* call on a backoff
schedule before surfacing the error. The schedule below, `(2, 8, 20)`, is the
value **KGQA's KG tools** choose (see "Opt-in" below); the core framework
defaults to no retry. Using KGQA's schedule as the example:

```
ONE tool call (KGQA's _KGToolBase sets server_down_retry_delays = (2, 8, 20)):
  attempt 1 → URLError → sleep 2s
  attempt 2 → URLError → sleep 8s
  attempt 3 → URLError → sleep 20s
  attempt 4 → URLError → give up → raise ToolServerDownError (breaker += 1)
```

A successful re-attempt returns one clean observation; the transient never
becomes a tree node and never touches the breaker counter.

### Opt-in via a tool attribute, default off

The schedule is a class attribute on `BaseTool`, defined in the **core
framework** (`lits/tools/base.py`) with a default of `()` — no retry:

```python
class BaseTool(ABC):
    # () = no retry (instant fail, same as before). Opt in per tool.
    server_down_retry_delays: tuple[int, ...] = ()
```

- **Default off (`()`) — core default:** every existing tool behaves exactly as
  before — one `URLError` raises instantly, no sleeps. Nothing changes for
  shell/SQL/web/PDF tools unless they opt in. The core framework ships no
  concrete schedule.
- **Opt in — per benchmark/tool:** a tool whose backend may briefly drop and
  recover overrides the attribute. The concrete `(2, 8, 20)` schedule is **not**
  a core value; it lives in the KGQA benchmark
  (`demos/lits_benchmark/kgqa_tools.py::_KGToolBase`), chosen because that
  endpoint is a Freebase SPARQL server behind an SSH tunnel:

  ```python
  # demos/lits_benchmark/kgqa_tools.py — benchmark layer, not core
  class _KGToolBase(BaseTool):
      server_down_retry_delays = (2, 8, 20)   # ride out autossh reconnect (~30s)
  ```

This mirrors the `classify_string_result_as_server_down` opt-out (Gotcha 4): a
per-tool switch so behavior changes only where wanted. The shell tool keeps the
core default — a blocking wait on a Docker hiccup is undesirable there.

### Caveats

- Retry is **off by default** in core, so for the vast majority of tools there
  is no blocking wait at all — a server-down failure raises instantly, exactly
  as before. The caveats below apply *only* to tools that opt in.
- For an opt-in tool, the retry uses a **blocking** `time.sleep`, so the calling
  MCTS branch stalls for that tool's configured schedule (KGQA's `(2, 8, 20)`
  sums to ~30s of sleep worst-case, plus per-attempt failure latency). Acceptable
  for a rare event; it is synchronous. Tune the schedule per tool if a shorter
  stall is wanted.
- Retry trades time-to-abort for resilience: against a *genuinely* dead server,
  an opt-in tool now sleeps its full schedule before each failure escalates, so
  the breaker takes `tool_failure_threshold × schedule` to trip instead of
  failing instantly. KGQA went from a ~9s abort to ~90s+ (3 × ~30s). This is the
  intended cost — we wait longer to be *sure* the server is dead before aborting
  — but it means a real outage burns more wall-clock before the run stops.

---

## 3. Circuit breaker — abort a run against dead infrastructure

`ToolUseTransition` tracks consecutive `ToolServerDownError`s across all examples
in a run. After `tool_failure_threshold` (default 3) consecutive failures, it
re-raises so the CLI top-level handler aborts instead of burning more LLM budget
on dead infrastructure. The counter **resets on any successful tool call** and
**persists across examples** (an intermittently-failing backend still trips).

Because retry (layer 2) now sits below the breaker, a `ToolServerDownError` only
reaches the counter *after* retries have confirmed the backend is unreachable —
so "3 consecutive" means "3 confirmed-dead events", not "3 hiccups in 9 seconds".

### Why retry lives below the breaker, not inside it

A natural question: why not put the retry logic in the breaker? Three reasons:

1. **The breaker doesn't re-issue the call — it only counts.** On a failure,
   `ToolUseTransition.step` increments the counter and returns an observation; it
   never re-invokes the tool. The actual re-attempt of the identical query has to
   happen where the call is made (`execute_tool_action`).
2. **Retrying in the breaker pollutes the search tree.** The breaker writes each
   failed attempt as a step/node — in the incident, the reward model was literally
   invoked to score a `"Tool server unreachable (attempt 1/3)"` trajectory.
   Retrying below the node-creation layer means a transient never becomes a node;
   `step()` returns one clean observation or raises once.
3. **Separation of concerns.** Retry = "recover a single operation"; breaker =
   "give up on the whole run". Different lifetimes (one call vs cross-example) and
   goals. Keeping them separate makes the breaker's threshold mean "confirmed-dead
   events", which is more correct.

A considered alternative — making the breaker *time-aware* (require failures to
span >60s) — would have prevented the specific 9s trip, but it's strictly worse:
it still pollutes the tree with failed nodes, still never obtains a successful
observation, and only delays the abort instead of recovering.

### Diagnosing a trip

When the breaker fires, the trip step is appended with the observation:

```
Tool server unreachable. Final attempt before circuit-breaker abort.
Reason: <original exception>
```

logged at `WARNING` via `log_event`:

```
[TOOL_SERVER_DOWN] example=<idx> TRIPPED after 3 consecutive failures
(cross-example), tool=<tool_name> last_reason=<exc-repr>
```

To read trip causes from `execution.log`:

```bash
grep -E "TRIPPED|Tool server unreachable \(attempt" execution.log | tail -10
```

| `last_reason` pattern | Diagnosis |
|---|---|
| `URLError('connection refused')` / `(URLError)` with no HTTP code | Genuine network/tunnel issue. Check tunnel + SSO. |
| `URLError` wrapping `HTTPError(code=4xx)` | Should NOT trip (Gotcha 1). If seen, the patch may have regressed. |
| `URLError` wrapping `HTTPError(code=5xx)` with no application marker | Server overloaded/restarting. Wait, resume. |
| HTTP 5xx with application marker (`S0022`, `Virtuoso`, ...) | Query-side failure on a healthy server. Should NOT trip (Gotcha 3). |
| `OperationalError` "could not connect" / "server has gone away" | DB endpoint dropped. |

---

## 4. What this means for your operational recovery

Reconnecting the backend is **your** job, not LiTS's — LiTS has no tunnel
supervisor, no auto-reconnect, no host-sleep guard. What LiTS *does* impose is a
**recovery deadline** set by the retry budget of layer 2. This section states
that deadline and what it requires of whatever recovery tooling you run.

### The constraint: how fast recovery must be

Two windows have to line up:

- **Per-call ride-out window** = the opt-in tool's retry budget. For KGQA's
  `(2, 8, 20)` that is ~30s of sleep before a single call gives up.
- **Run-level abort window** = `tool_failure_threshold × ride-out`. For KGQA,
  3 × ~30s ≈ ~90s before the breaker trips and the run aborts.

So the rule of thumb: **your backend must come back within one ride-out window
(~30s for KGQA) for a transient blip to be fully absorbed with zero failed
calls.** If it comes back within the run-level window (~90s) the run survives but
logs some failed attempts. Longer than that and the run aborts — that is by
design (a multi-minute outage is a real outage, resume later).

Tune the trade-off by editing the tool's `server_down_retry_delays` (longer =
more tolerant, but a real outage burns more wall-clock before aborting).

### Meeting the constraint (KGQA SSH-tunnel example)

For KGQA the backend is a Freebase SPARQL server behind an SSH tunnel; autossh
keepalive is how you keep the reconnect inside the ~30s window:

```bash
autossh -M 0 -fN \
  -o "ServerAliveInterval=15" \
  -o "ServerAliveCountMax=2" \
  -o "ExitOnForwardFailure=yes" \
  -o "TCPKeepAlive=yes" \
  -L 3001:localhost:3001 \
  -i ~/.ssh/<key>.pem ubuntu@<freebase-host>
```

`ServerAliveInterval × ServerAliveCountMax = 15 × 2 ≈ 30s` is the worst-case
*detection* window before autossh restarts the tunnel. Add a few seconds for
re-establishment. That total needs to fit inside the run-level abort window
(~90s) for the run to survive a drop — and ideally inside one ride-out window
(~30s) for the blip to be invisible. The defaults above are chosen to do that;
if you widen the keepalive, widen the retry schedule to match.

Verify the tunnel and server before (re)starting a run:

```bash
lsof -nP -iTCP:3001 -sTCP:LISTEN
curl -s --max-time 8 "http://localhost:3001/sparql?query=SELECT+%3Fs+WHERE+%7B+%3Fs+%3Fp+%3Fo+%7D+LIMIT+1" | head -3
```

A `<sparql>...` result (exit 0) means the tunnel and server are healthy.

### If the run aborted anyway (resume)

When recovery is too slow and the breaker trips, the run is resumable —
`lits-search` is incremental and skips examples whose
`terminal_nodes_<idx>.json` already exists. Fix the backend first, then run
`lits-resume-clean --result-dir <RD>` to clean any half-finished example and
re-launch the original command. See `docs/cli/resume.md`.

---

## Code & test references

- `lits/tools/base.py::BaseTool.server_down_retry_delays` — opt-in retry attribute.
- `lits/tools/base.py::BaseTool.classify_string_result_as_server_down` — string-path opt-out.
- `lits/tools/utils.py::execute_tool_action` — retry loop.
- `lits/tools/utils.py::_attempt_tool_call` — single-attempt execute + classify.
- `lits/tools/utils.py::_classify_as_server_down` — the classifier.
- `lits/components/transition/tool_use.py::ToolUseTransition.step` — the breaker.
- `demos/lits_benchmark/kgqa_tools.py::_KGToolBase` — opts in with `(2, 8, 20)`.
- `unit_test/tools/test_server_down_retry.py` — retry behavior (3 cases).
- `unit_test/components/transition/test_tool_server_down_classify.py` — classification (cases a–n).
- `unit_test/components/transition/test_tool_use_circuit_breaker.py` — breaker (counter, trip, reset).
- `.kiro/specs/lits_mem/x-0522-0522-minor-tool-failure-circuit-breaker/` — spec that introduced the breaker.
