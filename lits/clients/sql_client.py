from typing import Any, Dict
from langchain_community.utilities import SQLDatabase
from .base import BaseClient
from sqlalchemy import create_engine

class SQLDBClient(BaseClient):
    """Unified wrapper for SQL or GeoSQL databases."""

    def __init__(self, uri: str, schema: str = None):
        super().__init__(uri=uri)
        
        engine = create_engine(
            uri,
            connect_args={"options": f"-c search_path={schema}"} if schema else {}
        )
        self.db = SQLDatabase(engine)

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
