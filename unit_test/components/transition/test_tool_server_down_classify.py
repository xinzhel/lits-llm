"""Unit test for the server-down classification path in
`lits/tools/utils.py::execute_tool_action`.

What this test covers (subtask 1.3 of
`.kiro/specs/lits_mem/0522-minor-tool-failure-circuit-breaker/tasks.md`):

a. A synthetic `urllib.error.URLError("connection refused")` raised by a fake
   tool causes `execute_tool_action` to raise `ToolServerDownError`.
b. A synthetic `ValueError("variable must be a relation of the variable")`
   (a typical KGQA bad-action error) falls through to the legacy
   `PREFIX_FOR_ERROR_OBSERVATION` string return — NOT `ToolServerDownError`.
c. A synthetic `sqlite3.OperationalError("no such table: foo")` falls through
   to the legacy string return (OperationalError without connection-related
   substring).
d. A synthetic `sqlite3.OperationalError("could not connect to server")`
   raises `ToolServerDownError` (OperationalError with connection substring).
e. A tool that itself raises `ToolServerDownError` is propagated unchanged,
   with `tool_name` backfilled.
f. A tool that *returns* (does not raise) a string containing a network
   marker (e.g. ``"... URLError ..."``) causes `execute_tool_action` to
   raise `ToolServerDownError` (string-return path; subtask 1.4).
g. A tool that returns a normal observation string (e.g. ``"Result: 5 rows"``)
   is returned unchanged — no `ToolServerDownError` (subtask 1.4).
h. A tool that returns a string containing the broad word "connect" but no
   error vocabulary (e.g. ``"Step 1: connect to the database"``) is returned
   unchanged — the proximity heuristic prevents false positives (subtask 1.4).
i. An HTTP 4xx error (e.g. SPARQLWrapper's ``QueryBadFormed`` chained from
   ``urllib.error.HTTPError(code=400)``) is **not** classified as server-down.
   These are client errors (malformed query from the LLM), not network failures.
j. An HTTP 5xx error (e.g. ``HTTPError(code=503)``) IS classified as server-down
   — the server is reachable but failing.

Run (no breakpoints, batch-friendly):

    PYTHONBREAKPOINT=0 python -m unit_test.components.transition.test_tool_server_down_classify
"""

import json
import sqlite3
import urllib.error

from lits.tools.utils import execute_tool_action, ToolServerDownError
from lits.utils import PREFIX_FOR_ERROR_OBSERVATION


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeTool:
    """Minimal tool stand-in. `execute_tool_action` looks for `.name` and
    invokes either `_run(**kwargs)` or `__call__(**kwargs)`.
    """

    def __init__(self, name: str, raise_exc: BaseException):
        self.name = name
        self._raise_exc = raise_exc

    def _run(self, **kwargs):
        raise self._raise_exc


class _FakeReturningTool:
    """Tool stand-in that *returns* a fixed string instead of raising. Used to
    exercise the string-return classification path in `execute_tool_action`.
    """

    def __init__(self, name: str, return_value: str):
        self.name = name
        self._return_value = return_value

    def _run(self, **kwargs):
        return self._return_value


def _action(tool_name: str) -> str:
    return json.dumps({"action": tool_name, "action_input": {}})


# ---------------------------------------------------------------------------
# Test cases (sequentially executable, no pytest)
# ---------------------------------------------------------------------------

def case_a_urllib_url_error_is_server_down():
    """URLError → ToolServerDownError."""
    tool = _FakeTool("kgqa_sparql", urllib.error.URLError("connection refused"))
    try:
        execute_tool_action(_action("kgqa_sparql"), [tool])
    except ToolServerDownError as e:
        print("[case a] OK — got ToolServerDownError:", e)
        print("         original_exc =", repr(e.original_exc))
        print("         tool_name    =", e.tool_name)
        return
    print("[case a] FAIL — expected ToolServerDownError, did not get one")
    raise SystemExit(1)


def case_b_value_error_is_legacy_string():
    """LLM-generated bad-action error → legacy observation string."""
    tool = _FakeTool(
        "kgqa_sparql",
        ValueError("variable must be a relation of the variable"),
    )
    out = execute_tool_action(_action("kgqa_sparql"), [tool])
    if isinstance(out, str) and out.startswith(PREFIX_FOR_ERROR_OBSERVATION):
        print("[case b] OK — legacy observation string:", out)
        return
    print("[case b] FAIL — expected legacy error string, got:", repr(out))
    raise SystemExit(1)


def case_c_sqlite_operational_no_substring_is_legacy_string():
    """OperationalError without connection substring → legacy string."""
    tool = _FakeTool(
        "sql_query",
        sqlite3.OperationalError("no such table: foo"),
    )
    out = execute_tool_action(_action("sql_query"), [tool])
    if isinstance(out, str) and out.startswith(PREFIX_FOR_ERROR_OBSERVATION):
        print("[case c] OK — legacy observation string:", out)
        return
    print("[case c] FAIL — expected legacy error string, got:", repr(out))
    raise SystemExit(1)


def case_d_sqlite_operational_with_connect_is_server_down():
    """OperationalError with connection substring → ToolServerDownError."""
    tool = _FakeTool(
        "sql_query",
        sqlite3.OperationalError("could not connect to server"),
    )
    try:
        execute_tool_action(_action("sql_query"), [tool])
    except ToolServerDownError as e:
        print("[case d] OK — got ToolServerDownError:", e)
        print("         original_exc =", repr(e.original_exc))
        return
    print("[case d] FAIL — expected ToolServerDownError, did not get one")
    raise SystemExit(1)


def case_e_tool_raises_server_down_directly():
    """Tool raising ToolServerDownError directly → propagated, tool_name
    backfilled if not set.
    """
    raised = ToolServerDownError("backend gone", original_exc=None, tool_name=None)
    tool = _FakeTool("kgqa_sparql", raised)
    try:
        execute_tool_action(_action("kgqa_sparql"), [tool])
    except ToolServerDownError as e:
        if e is raised and e.tool_name == "kgqa_sparql":
            print("[case e] OK — same instance propagated, tool_name backfilled:", e.tool_name)
            return
        print("[case e] FAIL — expected same instance with backfilled tool_name, got",
              repr(e), "tool_name=", e.tool_name)
        raise SystemExit(1)
    print("[case e] FAIL — expected ToolServerDownError to propagate")
    raise SystemExit(1)


def case_f_string_return_with_network_marker_is_server_down():
    """Tool *returns* a string containing a network marker (no raise) →
    `execute_tool_action` should raise `ToolServerDownError` via the
    string-return path (subtask 1.4).
    """
    return_value = (
        "Error: SPARQL query failed (URLError). Query: SELECT ?x WHERE { ?x ?y ?z }"
    )
    tool = _FakeReturningTool("kgqa_sparql", return_value)
    try:
        execute_tool_action(_action("kgqa_sparql"), [tool])
    except ToolServerDownError as e:
        print("[case f] OK — got ToolServerDownError on string-return marker:", e)
        print("         tool_name =", e.tool_name)
        return
    print("[case f] FAIL — expected ToolServerDownError on string-return path")
    raise SystemExit(1)


def case_g_string_return_no_marker_is_observation():
    """Tool returns a normal observation string → returned unchanged."""
    return_value = "Result: 5 rows"
    tool = _FakeReturningTool("sql_query", return_value)
    out = execute_tool_action(_action("sql_query"), [tool])
    if out == return_value:
        print("[case g] OK — observation returned unchanged:", out)
        return
    print("[case g] FAIL — expected observation unchanged, got:", repr(out))
    raise SystemExit(1)


def case_h_string_return_with_connect_word_but_no_error_vocab_is_observation():
    """Result contains the broad word "connect" but no error vocabulary → must
    be returned unchanged. The proximity heuristic in
    `_classify_result_as_server_down` prevents false-positive on legitimate
    observations like "Step 1: connect to the database".
    """
    return_value = "Step 1: connect to the database"
    tool = _FakeReturningTool("sql_query", return_value)
    out = execute_tool_action(_action("sql_query"), [tool])
    if out == return_value:
        print("[case h] OK — benign 'connect' observation returned unchanged:", out)
        return
    print("[case h] FAIL — expected observation unchanged, got:", repr(out))
    raise SystemExit(1)


def case_i_http_4xx_is_not_server_down():
    """HTTP 4xx (client error like 400 Bad Request from malformed SPARQL) must
    NOT be classified as server-down. The original bug: SPARQLWrapper raises
    `QueryBadFormed` on Virtuoso syntax errors with the underlying
    `urllib.error.HTTPError(code=400)` in the exception chain. Without this
    guard, the chain-walker matched `URLError` (parent of `HTTPError`) and
    incorrectly tripped the circuit breaker on agent-side LLM bugs.
    """
    class _FakeQueryBadFormed(Exception):
        pass

    # Pre-build the exception chain: HTTPError(400) → QueryBadFormed.
    try:
        try:
            raise urllib.error.HTTPError("http://endpoint", 400, "Bad Request", {}, None)
        except urllib.error.HTTPError as he:
            raise _FakeQueryBadFormed(
                "QueryBadFormed: A bad request has been sent to the endpoint"
            ) from he
    except _FakeQueryBadFormed as built:
        chained = built

    tool = _FakeTool("kgqa_sparql", chained)
    out = execute_tool_action(_action("kgqa_sparql"), [tool])

    # Expectation: falls through to legacy string return (not ToolServerDownError),
    # because HTTP 400 is a client error, not a server-down condition.
    if isinstance(out, str) and PREFIX_FOR_ERROR_OBSERVATION in out:
        print("[case i] OK — HTTP 400 fell through to legacy string return:", out[:80])
        return
    print("[case i] FAIL — expected legacy string return, got:", repr(out))
    raise SystemExit(1)


def case_j_http_5xx_is_server_down():
    """HTTP 5xx (server error like 503 Service Unavailable) must still be
    classified as server-down — the server is reachable but failing.
    """
    http_503 = urllib.error.HTTPError("http://endpoint", 503, "Service Unavailable", {}, None)
    tool = _FakeTool("kgqa_sparql", http_503)
    try:
        execute_tool_action(_action("kgqa_sparql"), [tool])
    except ToolServerDownError as e:
        print("[case j] OK — HTTP 503 classified as server-down:", e)
        return
    print("[case j] FAIL — expected ToolServerDownError on HTTP 503")
    raise SystemExit(1)


if __name__ == "__main__":
    case_a_urllib_url_error_is_server_down()
    case_b_value_error_is_legacy_string()
    case_c_sqlite_operational_no_substring_is_legacy_string()
    case_d_sqlite_operational_with_connect_is_server_down()
    case_e_tool_raises_server_down_directly()
    case_f_string_return_with_network_marker_is_server_down()
    case_g_string_return_no_marker_is_observation()
    case_h_string_return_with_connect_word_but_no_error_vocab_is_observation()
    case_i_http_4xx_is_not_server_down()
    case_j_http_5xx_is_server_down()
    print("\nALL CASES PASSED")
