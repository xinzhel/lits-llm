from dataclasses import dataclass
from functools import lru_cache

@dataclass
class ServiceConfig:
    base_url: str
    bearer_token: str
    timeout: int = 30
