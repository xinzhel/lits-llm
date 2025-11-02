from pydantic import BaseModel
from typing import Type, Any
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Unified tool abstraction used by LiTS-LLM agents."""
    name: str
    description: str
    args_schema: Type[BaseModel]

    def __init__(self, client: Any):
        # 如果子类同时继承了 BaseModel 会报错，所以使用 object.__setattr__ 避免 Pydantic 拦截，
        object.__setattr__(self, "client", client)


    @abstractmethod
    def _run(self, **kwargs) -> str:
        raise NotImplementedError
