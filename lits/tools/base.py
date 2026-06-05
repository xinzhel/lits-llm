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
