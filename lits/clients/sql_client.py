from typing import Any, Dict
from langchain_community.utilities import SQLDatabase
from .base import BaseClient
from sqlalchemy import create_engine

class SQLDBClient(BaseClient):
    """Unified wrapper for SQL or GeoSQL databases."""

    def __init__(self, uri: str, schema: str = None, ALLOWED_TABLES: list = None,
                 max_string_length: int = 300):
        super().__init__(uri=uri)

        # search_path connect_args are PostgreSQL-specific; skip for other dialects
        connect_args = {}
        if schema and uri.startswith("postgresql"):
            connect_args = {"options": f"-c search_path={schema}"}

        engine = create_engine(uri, connect_args=connect_args)
        self.db = SQLDatabase(
            engine,
            include_tables=ALLOWED_TABLES,
            max_string_length=max_string_length,
        )

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
