"""
Base client abstraction for LiTS-LLM.

Each client (e.g., WebServiceClient, SQLDatabaseClient, MapEvalClient)
inherits from BaseClient to ensure a unified interface for Tool classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseClient(ABC):
    """Abstract base class for all external service clients."""

    def __init__(self, **kwargs):
        """Optionally accept configuration parameters such as credentials or URIs."""
        self.config = kwargs

    @abstractmethod
    def request(self, *args, **kwargs) -> Dict[str, Any]:
        """Send a query or request to the backend service and return parsed data."""
        raise NotImplementedError

    @abstractmethod
    def ping(self) -> bool:
        """Check connectivity to the underlying service (API or DB)."""
        raise NotImplementedError
