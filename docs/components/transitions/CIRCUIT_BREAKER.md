# Tool-Failure Circuit Breaker

The `ToolUseTransition` (`lits/components/transition/tool_use.py`) tracks
consecutive tool-backend failures across all examples in a run. After
`tool_failure_threshold` (default 3) consecutive `ToolServerDownError`s, the
transition re-raises the error so the CLI top-level handler aborts the run
instead of burning more LLM budget on dead infrastructure.

This document covers how exceptions flow into the breaker, what counts as a
"server-down" event, and which gotchas have surfaced in practice.

## Exception Classification (`lits/tools/utils.py::execute_tool_action`)

Exceptions are sorted into three buckets:

| Bucket | Examples | Behavior |
|---|---|---|
| **Server-down** | `urllib.error.URLError("connection refused")`, `botocore.exceptions.EndpointConnectionError`, `psycopg2.OperationalError("could not connect")` | Wrapped as `ToolServerDownError` and propagated. Increments breaker counter. |
| **Bad action / agent error** | `ValueError("variable must be a relation")`, `sqlite3.OperationalError("no such table: foo")` | Returned to the agent as `Error executing tool. Error report: ...`. Counter unchanged. |
| **HTTP 4xx (client error)** | `urllib.error.HTTPError(code=400)` from a malformed SPARQL query | **Excluded** from server-down classification (see "Gotcha 1" below). Falls through to bucket 2. |
| **HTTP 5xx (server error)** | `urllib.error.HTTPError(code=503)` | Classified as server-down. |

The classifier (`_classify_as_server_down`) walks the `__cause__` /
`__context__` chain so that exceptions wrapping a network failure underneath
(e.g., a custom `BackendError("query failed") from socket.timeout`) are still
caught.

## Gotcha 1: HTTP 4xx Looks Like a Network Error If You Squint

`urllib.error.HTTPError` is a subclass of `URLError`. Without a code-range
guard, a Virtuoso `HTTP 400 Bad Request` on a malformed SPARQL query would
walk up to `URLError` in the chain-walker and trip the breaker — even though
the server is healthy and the bug is on the LLM-policy side.

**Symptom (first observed 2026-05-22):** KGQA MCTS Reflection died at
example 93 after the agent generated a chained `argmax`-after-`argmax` query
that produced invalid SPARQL with two consecutive `ORDER BY DESC(?sk0) LIMIT 1`
clauses. Virtuoso returned HTTP 400; `SPARQLWrapper` raised `QueryBadFormed`
chained from `urllib.error.HTTPError(code=400)`. The classifier saw `URLError`
in the chain and incorrectly tripped the circuit breaker after 3 consecutive
malformed-query attempts on the same example.

**Fix:** `_classify_as_server_down` now checks `HTTPError.code` and skips
4xx responses. 5xx remains classified as server-down. Regression tests:
`unit_test/components/transition/test_tool_server_down_classify.py` cases
i (4xx) and j (5xx).

**Takeaway:** When adding new exception types to the classifier, ask "is this
strictly a *transport-layer* failure?" before adding it. HTTP-protocol-level
errors need finer code-range checks; only treat them as server-down when the
status indicates the server is actually impaired.

## Gotcha 2: SSH Tunnels Drop Silently During Sleep / Network Hiccups

The Freebase SPARQL endpoint runs on an EC2 instance reached via SSH tunnel
(school wifi only permits port 22 outbound, so we forward `localhost:3001` to
the EC2 instance's `3001`). When the host Mac sleeps, hibernates, or the
network briefly drops, the tunnel goes down silently and the next SPARQL call
fails with `URLError("connection refused")`. The breaker correctly trips, but
the run aborts even if the underlying server is healthy.

**Symptom (observed 2026-05-22 to 2026-05-23):** Overnight run died around
4-5am when the tunnel dropped. After re-establishing the tunnel manually the
next morning, the agent ran for ~30 more minutes before the tunnel dropped
again and the breaker tripped permanently.

**Recommended fix: `autossh` with SSH keepalives**

Replace the plain `ssh -fN -L ...` tunnel:

```bash
ssh -fN -L 3001:localhost:3001 -i ~/.ssh/race_lits_server_us.pem ubuntu@<ec2-ip>
```

with `autossh`, which auto-reconnects when the tunnel dies:

```bash
brew install autossh   # one-time

autossh -M 0 -fN \
  -o "ServerAliveInterval=30" \
  -o "ServerAliveCountMax=3" \
  -o "ExitOnForwardFailure=yes" \
  -L 3001:localhost:3001 \
  -i ~/.ssh/race_lits_server_us.pem \
  ubuntu@<ec2-ip>
```

What each flag does:

- `-M 0`: disable autossh's own monitor port (modern best practice; uses SSH
  keepalives instead).
- `-fN`: background, no remote command (same as the original `ssh -fN`).
- `ServerAliveInterval=30 + ServerAliveCountMax=3`: SSH sends a heartbeat
  every 30s; after 3 missed heartbeats (~90s), the SSH client exits, which
  triggers autossh to spawn a fresh connection. Without these, the OS-default
  keepalive is hours and the tunnel stays half-broken.
- `ExitOnForwardFailure=yes`: if port 3001 can't be bound (e.g., previous
  tunnel still holding it), exit immediately rather than hanging.

**Verify the setup:**

```bash
# Should show one autossh + one ssh child
ps aux | grep -E "autossh|ssh.*3001" | grep -v grep

# Test resilience: kill the underlying ssh, autossh should respawn it
pkill -f "ssh.*3001:localhost:3001"
sleep 6
ps aux | grep -E "ssh.*3001" | grep -v grep   # new PID should appear
```

**Additional belt-and-suspenders:**

1. **`caffeinate -dimsu &`** before leaving the machine: prevents idle sleep,
   the most common cause of tunnel drops on macOS.
2. **AWS SSO refresh** if your runs use Bedrock: a `crontab` job that runs
   `aws sso login --sso-session <name>` every 8 hours to keep credentials
   fresh (Bedrock calls fail silently if SSO expires mid-run).

A more robust fix would be a tool-side reconnection retry (e.g., on
`URLError`, re-establish the tunnel via `subprocess` and retry once before
raising `ToolServerDownError`). Not implemented yet; tracked as future work.

## Diagnosing a Trip

When the breaker fires, the trip step is appended to the example's state with
the observation:

```
Tool server unreachable. Final attempt before circuit-breaker abort.
Reason: <original exception>
```

This phrase is also logged at `WARNING` level via `log_event`:

```
[TOOL_SERVER_DOWN] example=<idx> TRIPPED after 3 consecutive failures
(cross-example), tool=<tool_name> last_reason=<exc-repr>
```

To distinguish trip causes when reading `execution.log`:

```bash
grep -E "TRIPPED|Tool server unreachable \(attempt" execution.log | tail -10
```

If `last_reason` mentions:

| Pattern | Diagnosis |
|---|---|
| `URLError('connection refused')` or `(URLError)` with no HTTP code | Genuine network/tunnel issue. Check tunnel + SSO. |
| `URLError` wrapping `HTTPError(...,code=4xx,...)` | Should NOT trip after the 2026-05-22 patch. If you see this, the patch may have regressed. |
| `URLError` wrapping `HTTPError(...,code=5xx,...)` | Server overloaded or restarting. Wait, then resume. |
| `OperationalError` with "could not connect" / "server has gone away" | DB endpoint dropped. |

## Resuming After a Trip

`lits-search` is incremental: re-running with the same `--output-dir` skips
examples whose `terminal_nodes_<idx>.json` already exists. After fixing the
underlying issue (reconnect tunnel, refresh SSO, etc.), simply re-launch the
exact same command and it picks up where it stopped.

## Related

- `.kiro/specs/lits_mem/x-0522-0522-minor-tool-failure-circuit-breaker/` —
  spec that introduced the breaker.
- `lits/tools/utils.py::_classify_as_server_down` — classifier source.
- `unit_test/components/transition/test_tool_server_down_classify.py` —
  regression tests (cases a–j).
- `unit_test/components/transition/test_tool_use_circuit_breaker.py` —
  end-to-end breaker tests (counter accumulation, trip behavior, reset rules).
