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
| **HTTP 5xx (server error)** | `urllib.error.HTTPError(code=503)` | Classified as server-down — *unless* the response body carries an application/query-level error marker (see "Gotcha 3"). |

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

## Gotcha 3: Not Every HTTP 500 Means the Server Is Down

Some backends return HTTP 500 for *application/query-level* failures while the
server itself is perfectly healthy. The canonical case is Virtuoso's SQL
planner: a valid-but-complex SPARQL query can exceed its query-compiler limits
and come back as `HTTP 500` with a body like
`Virtuoso S0022 Error SQ200: No column s_18_9.x`. SPARQLWrapper wraps this as
`EndPointInternalError` with the underlying `urllib.error.HTTPError(code=500)`
in the exception chain.

**Symptom (first observed 2026-06-01):** KGQA MCTS (Qwen3-Coder-Next policy)
died at example 11. The agent repeatedly issued `get_attributes` on a numeric
variable `#6`; MCTS branches converged on the same query, so the identical
Virtuoso 500 fired three times in a row with no intervening successful call.
The breaker counted three consecutive "server-down" events and aborted — even
though the endpoint was reachable and answering other queries fine. (The same
500 had appeared once in earlier Sonnet runs but a subsequent successful call
reset the counter before it reached the threshold, masking the issue.)

**Fix:** `_classify_as_server_down` now treats an HTTP 5xx as a *query error*
(not server-down) when the exception chain text matches
`_HTTP_5XX_APPLICATION_ERROR_MARKERS` (e.g. `s0022`, `sq200`, `virtuoso`,
`sparql query:`, `syntax error`, `parse error`, `bad formed`, `malformed`).
The marker scan walks the whole `__cause__`/`__context__` chain because the
status code lives on the inner `HTTPError` while the descriptive body lives on
the outer wrapper. Bare 5xx with no application marker (e.g. a plain 502/503)
remain server-down. Regression tests:
`unit_test/components/transition/test_tool_server_down_classify.py` cases
k (Virtuoso 500 → query error) and l (bare 500 → server-down).

**Takeaway:** 5xx is a *hint*, not proof, that the server is impaired. When the
body identifies an application/query-compilation failure, the right response is
to hand the error back to the agent so it can try a different query — not to
trip the breaker and burn the whole run.

## Gotcha 2: Backend Tunnels Can Drop Silently

When the tool backend lives behind an SSH tunnel or any other long-lived
network shim, the tunnel can go down silently during host sleep/wake, network
hiccups, or peer disconnects. The next call fails with `URLError("connection
refused")`, the breaker correctly classifies it as server-down, and the run
aborts after 3 consecutive trips — even though the underlying server is
healthy and would be reachable if the tunnel were up.

This is operational tooling, not a `lits` library concern. Mitigation is
deployment-specific: configure your tunnel with auto-reconnect (e.g., a
keepalive-driven supervisor) and prevent host sleep during long runs.

A future improvement on the library side could be a tool-level reconnection
retry: on `URLError`, attempt to re-establish the tunnel via `subprocess`
and retry once before raising `ToolServerDownError`. Not implemented; tracked
as future work.

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
| `URLError` wrapping `HTTPError(...,code=5xx,...)` with no application marker | Server overloaded or restarting. Wait, then resume. |
| HTTP 5xx with application marker (`S0022`, `Virtuoso`, `SPARQL query:`, etc.) | Query-side failure on a healthy server. Should NOT trip after the 2026-06-01 patch; surfaced to the agent as an error observation. |
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
