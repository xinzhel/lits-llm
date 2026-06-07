from pydantic import BaseModel
from typing import Type, Any
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Unified tool abstraction used by LiTS-LLM agents."""
    name: str
    description: str
    args_schema: Type[BaseModel]

    # Whether a string return value from this tool should be scanned for
    # backend-down markers (see ``execute_tool_action`` string-return path).
    # True for tools whose backend is a network service that may stringify a
    # connection error instead of raising (SQL, SPARQL). False for tools whose
    # string output is arbitrary task content — e.g. a shell tool, where command
    # output legitimately contains words like "connect"/"error"/"404" that would
    # otherwise trip the circuit breaker as a false positive.
    classify_string_result_as_server_down: bool = True

    # Retry-with-backoff schedule (in seconds) applied by ``execute_tool_action``
    # when a single tool call is classified as server-down. Each entry is the
    # sleep before the next re-attempt of the *same* call, so e.g. ``(2, 8, 20)``
    # gives up to 3 re-attempts spanning ~30s before the failure is finally
    # raised as ``ToolServerDownError`` (which is what increments the circuit
    # breaker). The default ``()`` means no retry — the call fails instantly,
    # exactly as before. Opt in only for network-backed tools whose backend may
    # briefly disappear and recover (e.g. SPARQL/SQL over an SSH tunnel that
    # autossh reconnects). Leave it off for tools where a 30s blocking wait is
    # undesirable (e.g. a shell tool) or where failures are not transient.
    #
    # Distinction from the circuit breaker: this retries *within one call* to
    # ride out a transient blip so it never becomes a tree node and never counts
    # toward the breaker. The breaker (``tool_failure_threshold`` in
    # ``ToolUseTransition``) counts *across calls* and aborts the run only once
    # the backend is confirmed dead (all retries exhausted).
    server_down_retry_delays: tuple[int, ...] = ()

    def __init__(self, client: Any):
        # 如果子类同时继承了 BaseModel 会报错()，所以使用 object.__setattr__ 避免 Pydantic 拦截，
        object.__setattr__(self, "client", client)

    def pre_step(self, state) -> None:
        """Optional hook called before each tool execution.

        Override to update internal state from the current trajectory.
        For example, KG tools rebuild variable tracking from ToolUseState.

        Args:
            state: Current TrajectoryState (e.g., ToolUseState).
        """
        pass

    @abstractmethod
    def _run(self, **kwargs) -> str:
        raise NotImplementedError
