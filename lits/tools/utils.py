import requests
import logging
import socket
import time
import urllib.error
from typing import Optional
from ..utils import parse_json_string, PREFIX_FOR_ERROR_OBSERVATION

logger = logging.getLogger(__name__)


# Server-down classification whitelists.
# `_NETWORK_EXC_TYPES`: an exception of any of these types is treated as
# server-down outright.
# `_OPERATIONAL_ERROR_TYPES`: DB-driver `OperationalError`s are overloaded
# (connection failures, "no such table", locks, etc.), so we treat them as
# server-down only when the message contains a substring from
# `_OPERATIONAL_ERROR_CONNECT_SUBSTRINGS`.
# Optional deps (requests, botocore, psycopg2, pymysql) are imported
# defensively; missing ones simply drop their types from the whitelists.

_NETWORK_EXC_TYPE_LIST: list[type] = [
    urllib.error.URLError,
    ConnectionError,            # builtin; parent of ConnectionRefused/Reset/Aborted
    ConnectionRefusedError,
    ConnectionResetError,
    socket.timeout,
    socket.gaierror,
    TimeoutError,
]

try:
    import requests.exceptions as _requests_exceptions
    _NETWORK_EXC_TYPE_LIST.extend([
        _requests_exceptions.ConnectionError,
        _requests_exceptions.Timeout,
        _requests_exceptions.ConnectTimeout,
    ])
except ImportError:
    pass

try:
    import botocore.exceptions as _botocore_exceptions  # type: ignore
    _NETWORK_EXC_TYPE_LIST.extend([
        _botocore_exceptions.EndpointConnectionError,
        _botocore_exceptions.ConnectTimeoutError,
        _botocore_exceptions.ReadTimeoutError,
    ])
except ImportError:
    pass

_NETWORK_EXC_TYPES: tuple[type, ...] = tuple(_NETWORK_EXC_TYPE_LIST)

_OPERATIONAL_ERROR_TYPE_LIST: list[type] = []

try:
    import sqlite3 as _sqlite3
    _OPERATIONAL_ERROR_TYPE_LIST.append(_sqlite3.OperationalError)
except ImportError:
    pass

try:
    import psycopg2 as _psycopg2  # type: ignore
    _OPERATIONAL_ERROR_TYPE_LIST.append(_psycopg2.OperationalError)
except ImportError:
    pass

try:
    import pymysql.err as _pymysql_err  # type: ignore
    _OPERATIONAL_ERROR_TYPE_LIST.append(_pymysql_err.OperationalError)
except ImportError:
    pass

_OPERATIONAL_ERROR_TYPES: tuple[type, ...] = tuple(_OPERATIONAL_ERROR_TYPE_LIST)

# Case-insensitive substrings signaling a connection-related OperationalError.
_OPERATIONAL_ERROR_CONNECT_SUBSTRINGS: tuple[str, ...] = (
    "connect",
    "server has gone away",
    "lost connection",
    "connection refused",
    "connection reset",
    "could not connect",
    "can't connect",
)

# Case-sensitive class-name markers used to classify a string-returning tool's
# result as server-down (when a tool wrapper catches and stringifies a network
# exception instead of letting it propagate).
_NETWORK_RESULT_MARKERS: tuple[str, ...] = (
    "URLError",
    "ConnectTimeoutError",
    "HTTPConnectionPool",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "EndpointConnectionError",
    "ReadTimeoutError",
)

# Proximity guard: when a result contains a broad operational substring like
# "connect", we additionally require one of these error words to avoid
# misclassifying benign observations such as "Step 1: connect to the database".
_RESULT_ERROR_VOCAB: tuple[str, ...] = (
    "error",
    "fail",
    "refused",
    "timeout",
    "unreachable",
)

# Case-insensitive substrings that, when present in an HTTP 5xx response body,
# indicate an application/query-level failure (the server is reachable and
# healthy, but rejected this specific request) rather than server impairment.
# Example: Virtuoso returns HTTP 500 with "S0022 Error SQ200: No column ..."
# when its SQL planner cannot compile a valid-but-complex SPARQL query. Such a
# failure should be surfaced to the agent as an error observation so it can try
# a different query, NOT counted toward the server-down circuit breaker.
_HTTP_5XX_APPLICATION_ERROR_MARKERS: tuple[str, ...] = (
    "s0022",          # Virtuoso SQL state: compilation error
    "sq200",          # Virtuoso SQL error code
    "virtuoso",       # Virtuoso engine error banner
    "sparql query:",  # endpoint echoes back the offending query
    "syntax error",
    "parse error",
    "bad formed",
    "malformed",
)


def _classify_result_as_server_down(result: str) -> bool:
    """Return True if a string-typed tool result looks like a backend-down message.

    Matches case-sensitive class-name markers (e.g. ``URLError``) or
    case-insensitive operational substrings (e.g. ``"connect"``) when paired
    with error vocabulary (``error``, ``fail``, ``refused``, ``timeout``,
    ``unreachable``). The vocabulary check guards against benign observations
    like ``"Step 1: connect to the database"``.
    """
    if any(marker in result for marker in _NETWORK_RESULT_MARKERS):
        return True
    lower = result.lower()
    if any(vocab in lower for vocab in _RESULT_ERROR_VOCAB):
        if any(s in lower for s in _OPERATIONAL_ERROR_CONNECT_SUBSTRINGS):
            return True
    return False


def _iter_exception_chain(exc: BaseException):
    """Yield ``exc`` and each exception in its ``__cause__``/``__context__`` chain."""
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__


def _is_application_level_5xx(exc: BaseException) -> bool:
    """Return True if any message in ``exc``'s chain carries an HTTP 5xx
    application/query-level error marker.

    The HTTP status code (5xx) is typically on a chained
    ``urllib.error.HTTPError``, while the descriptive body (e.g. Virtuoso's
    ``S0022 Error SQ200: No column ...``) is on an outer wrapper such as
    ``SPARQLWrapper.EndPointInternalError``. We therefore scan the combined
    text of the entire chain rather than a single frame.
    """
    combined = " ".join(str(e) for e in _iter_exception_chain(exc)).lower()
    return any(marker in combined for marker in _HTTP_5XX_APPLICATION_ERROR_MARKERS)


def _classify_as_server_down(exc: BaseException) -> bool:
    """Return True if `exc` (or its `__cause__` / `__context__` chain) looks
    like a backend-unreachable condition.

    Matches known network exception types directly, or DB-driver
    `OperationalError`s whose message contains a connection substring.

    HTTP 4xx errors are explicitly excluded: a 400/404/etc. response means the
    server is reachable and rejected the *client's* request (typically a
    malformed query generated by the LLM). Treating these as server-down
    incorrectly trips the circuit breaker on agent-side bugs. HTTP 5xx (and
    plain `URLError` with no response code, e.g. connection refused) remain
    classified as server-down.

    HTTP 5xx exception: a 5xx whose error text indicates an *application/query*
    failure (e.g. a Virtuoso SQL-planner error `S0022 Error SQ200` returned as
    HTTP 500 for a valid-but-complex SPARQL query) is NOT server-down. The
    server is healthy; it just could not process this particular request. Such
    a 5xx is excluded so the failure is surfaced to the agent as an error
    observation rather than tripping the breaker. Genuine infrastructure 5xx
    (e.g. 502/503 with no application-error marker) remain server-down.
    """
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))

        # HTTP 4xx: client error, not server-down. Skip this frame and
        # continue walking the chain in case a real network error is wrapped
        # underneath.
        if isinstance(cur, urllib.error.HTTPError):
            code = getattr(cur, "code", None)
            if code is not None and 400 <= code < 500:
                cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
                continue
            # HTTP 5xx carrying an application/query-level error marker: the
            # server is reachable but rejected this specific request. Skip this
            # frame so it falls through to a non-server-down classification.
            # Scan from the original exception so an outer wrapper's body (e.g.
            # EndPointInternalError's Virtuoso text) is visible, not just this
            # inner HTTPError frame.
            if code is not None and 500 <= code < 600:
                if _is_application_level_5xx(exc):
                    cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
                    continue

        if _NETWORK_EXC_TYPES and isinstance(cur, _NETWORK_EXC_TYPES):
            return True

        if _OPERATIONAL_ERROR_TYPES and isinstance(cur, _OPERATIONAL_ERROR_TYPES):
            msg = str(cur).lower()
            if any(s in msg for s in _OPERATIONAL_ERROR_CONNECT_SUBSTRINGS):
                return True

        nxt = cur.__cause__ if cur.__cause__ is not None else cur.__context__
        cur = nxt
    return False


class ToolServerDownError(Exception):
    """Raised when a tool's backend appears to be unreachable.

    Distinct from generic tool errors (bad LLM-generated actions, schema
    errors, etc.): this only signals a network / backend-layer failure.
    `ToolUseTransition` counts these to drive the circuit breaker.

    Producers:
        - `execute_tool_action()` translates known network exception types
          and stringified network markers into this exception.
        - Tool authors may also raise it directly from `_run()` for precise
          control.

    Args:
        message: Human-readable description.
        original_exc: The underlying exception, if any. Stored for diagnostics.
        tool_name: Tool whose backend appeared down. Stored for diagnostics.
    """

    def __init__(
        self,
        message: str,
        *,
        original_exc: Optional[BaseException] = None,
        tool_name: Optional[str] = None,
    ):
        super().__init__(message)
        self.original_exc = original_exc
        self.tool_name = tool_name

def execute_tool_action(action_data: str, tools: list):
    """Parse and execute an LLM-generated JSON tool action.

    Server-down handling: if the tool raises a known network exception, or
    returns a string containing a network marker, this function raises
    `ToolServerDownError`. Other tool failures are returned as observation
    strings prefixed with `PREFIX_FOR_ERROR_OBSERVATION`.

    Args:
        action_data: The raw JSON string generated by the model.
        tools: Available tool instances.

    Returns:
        The tool's output (the observation) on success or non-network failure.

    Raises:
        ToolServerDownError: When the tool's backend appears unreachable.
        ValueError: When no tool with the requested name exists.
    """
    # --- 1. Parse JSON safely ---
    logger.debug(f"Raw action data: {action_data}")
    parsed_action, feedback = parse_json_string(action_data)
    logger.debug(f"Parsed action data: {parsed_action}")
    if parsed_action is None:
        assert feedback is not None
        return f"{PREFIX_FOR_ERROR_OBSERVATION}{feedback}"

    # --- 2. Validate required fields ---
    if "action" not in parsed_action or "action_input" not in parsed_action:
        # raise ValueError(f"Invalid tool call format: {parsed_action}")
        return f"{PREFIX_FOR_ERROR_OBSERVATION}Invalid tool call format: missing 'action' or/and 'action_input'."

    tool_name = parsed_action["action"]
    action_input = parsed_action["action_input"]

    # --- 3. Find tool by name ---
    tool = next((t for t in tools if t.name == tool_name), None)
    if tool is None:
        raise ValueError(f"No tool found with name '{tool_name}'")

    # --- 4. Execute tool, with optional retry-with-backoff on server-down ---
    # A single tool call may transiently fail (e.g. an SSH tunnel reconnecting).
    # When the tool opts in via ``server_down_retry_delays``, ride out the blip
    # by re-attempting the *same* call before surfacing ``ToolServerDownError``.
    # Each successful retry returns a clean observation, so the transient never
    # becomes a tree node and never increments the circuit breaker. Only when
    # all retries are exhausted (backend confirmed down) does the error
    # propagate to ``ToolUseTransition`` and count toward the breaker.
    retry_delays = getattr(tool, "server_down_retry_delays", ()) or ()
    attempt = 0
    while True:
        try:
            return _attempt_tool_call(tool, tool_name, action_input)
        except ToolServerDownError as e:
            if attempt < len(retry_delays):
                delay = retry_delays[attempt]
                attempt += 1
                logger.warning(
                    f"Tool '{tool_name}' server-down; retrying in {delay}s "
                    f"(attempt {attempt}/{len(retry_delays)}). Reason: "
                    f"{e.original_exc or e}"
                )
                time.sleep(delay)
                continue
            # No retries configured, or all retries exhausted → real failure.
            raise


def _attempt_tool_call(tool, tool_name: str, action_input):
    """Execute a single tool call and classify the outcome.

    Returns the observation on success or non-network failure; raises
    ``ToolServerDownError`` when the backend appears unreachable. Factored out
    of ``execute_tool_action`` so the retry loop can re-attempt one call without
    re-parsing the action or re-resolving the tool.
    """
    try:
        # Handle empty string or non-dict action_input
        if isinstance(action_input, str) and action_input.strip() == "":
            action_input = {}

        if hasattr(tool, "_run"):
            result = tool._run(**action_input)
        elif hasattr(tool, "__call__"):
            result = tool(**action_input)
        else:
            raise AttributeError(f"Tool {tool_name} has no callable method.")
    except ToolServerDownError as e:
        # Tool raised the typed signal directly; propagate with tool_name backfilled.
        if e.tool_name is None:
            e.tool_name = tool_name
        raise
    except Exception as e:
        if _classify_as_server_down(e):
            logger.info(
                f"Classified tool '{tool_name}' as server-down: "
                f"kind=exception matched_type={type(e).__name__}"
            )
            raise ToolServerDownError(
                f"Tool '{tool_name}' backend appears down: {type(e).__name__}: {e}",
                original_exc=e,
                tool_name=tool_name,
            ) from e
        return f"{PREFIX_FOR_ERROR_OBSERVATION}'{tool_name}': {e}"

    # String-return path: some tool wrappers stringify network exceptions
    # instead of raising. Treat matching results as server-down too — but only
    # for tools that opt in (network-backed tools). Tools whose string output is
    # arbitrary task content (e.g. the shell tool) set
    # ``classify_string_result_as_server_down = False`` to avoid false positives
    # when command output happens to contain words like "connect"/"error".
    if (
        isinstance(result, str)
        and getattr(tool, "classify_string_result_as_server_down", True)
        and _classify_result_as_server_down(result)
    ):
        logger.info(
            f"Classified tool '{tool_name}' as server-down: "
            f"kind=string-marker matched_in_result_preview={result[:120]!r}"
        )
        raise ToolServerDownError(
            f"Tool '{tool_name}' backend appears down (string-return path): {result}",
            tool_name=tool_name,
        )

    return result
def inspect_toolkit(tools):
    for tool in tools:
        inspect_tool(tool)
        
def inspect_tool(tool):
    print("Type: ", type(tool))
    print("Name:", tool.name) # multiply
    print("Description:", tool.description) # Multiply two numbers.
    print("Args:", tool.args)
    print("\n")
    
def test_connection():
    # find_nearby = NearbyPlacesTool(client=client)
    # find_nearby.invoke({"placeId": "0", "type": "accounting" })
    # run the following command to find the Bearer token:
    # curl -X POST http://10.224.245.233:5000/api/login -H "Content-Type: application/json" -d '{"username": "test", "password": "123"}'
    url = "http://10.224.245.233:5000/api"+"/map/nearby"
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InRlc3QiLCJpc3MiOiJtYXBxdWVzdC1hcHAub25yZW5kZXIuY29tIiwiaWF0IjoxNzU5OTk4NzUyfQ.ss23UkIzD73vogcFoRoUG1GrBAfFTOB0_H2CAS6Q-Z0"
    }
            
    params = {
        "location": 0,
        "radius": None,
        "type": "accounting",
        "keyword": None,
        "rankby": "distance"
    }
    response = requests.get(url, headers=headers, params=params)
    print(response)