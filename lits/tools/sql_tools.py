"""
The RDB tools are adapters turning LangChain SQLDatabaseTools into LiTS-LLM unified Tool instances.
"""

from typing import Any, List, Type
from pydantic import BaseModel, PrivateAttr
from .base import BaseTool
from ..clients.sql_client import SQLDBClient
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool as LCInfoSQLDatabaseTool,
    ListSQLDatabaseTool as LCListSQLDatabaseTool,
    QuerySQLDatabaseTool as LCQuerySQLDatabaseTool,
)

class QuerySQLDatabaseTool(LCQuerySQLDatabaseTool, BaseTool): # Method Resolution Order: LCQuerySQLDatabaseTool -> Tool -> object
    """Execute SQL queries on the database."""
    
    def __init__(self, client: SQLDBClient):
        LCQuerySQLDatabaseTool.__init__(self, db=client.db)
        BaseTool.__init__(self, client=client)
        # self.name = LCQuerySQLDatabaseTool.name
        self.description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. "
            # "If you encounter an issue with Unknown column "
            # f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
            # "to query the correct table fields."
        )
        # self.args_schema: Type[BaseModel] = LCQuerySQLDatabaseTool.args_schema

    def _run(self, **kwargs) -> str:
        return super()._run(**kwargs)

class InfoSQLDatabaseTool(LCInfoSQLDatabaseTool, BaseTool):
    """Fetch schema and sample rows for specific tables."""
    def __init__(self, client):
        LCInfoSQLDatabaseTool.__init__(self, db=client.db)
        BaseTool.__init__(self, client=client)
        # self.name = LCInfoSQLDatabaseTool.name
        self.description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            # "Be sure that the tables actually exist by calling "
            # f"{list_sql_database_tool.name} first! "
            "Example Input: table1, table2, table3"
        )
        # self.args_schema: Type[BaseModel] = LCInfoSQLDatabaseTool.args_schema

    def _run(self, **kwargs) -> str:
        return super()._run(**kwargs)


class ListSQLDatabaseTool(LCListSQLDatabaseTool, BaseTool):
    """List all available tables in the SQL database."""
    def __init__(self, client):
        LCListSQLDatabaseTool.__init__(self, db=client.db)
        BaseTool.__init__(self, client=client)
        # self.name = LCListSQLDatabaseTool.name
        # self.description = LCListSQLDatabaseTool.description
        # self.args_schema: Type[BaseModel] = LCListSQLDatabaseTool.args_schema

    def _run(self, **kwargs) -> str:
        return super()._run(**kwargs)
