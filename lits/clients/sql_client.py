from typing import Any, Dict
from langchain_community.utilities import SQLDatabase
from .base import BaseClient

class SQLDBClient(BaseClient):
    """Unified wrapper for SQL or GeoSQL databases."""

    def __init__(self, uri: str):
        super().__init__(uri=uri)
        self.db = SQLDatabase.from_uri(uri)

    def request(self, query: str, **kwargs) -> Dict[str, Any]:
        """Run SQL query and return results in dict form."""
        result = self.db.run(query)
        return {"result": result}

    def ping(self) -> bool:
        try:
            self.db.run("SELECT 1;")
            return True
        except Exception:
            return False
