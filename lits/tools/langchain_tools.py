def get_tools(db):
    """Get the tools in the toolkit."""
    from langchain_community.tools.sql_database.tool import (
        InfoSQLDatabaseTool,
        ListSQLDatabaseTool,
        QuerySQLCheckerTool,
        QuerySQLDatabaseTool,
    )
    list_sql_database_tool = ListSQLDatabaseTool(db=db)
    info_sql_database_tool_description = (
        "Input to this tool is a comma-separated list of tables, output is the "
        "schema and sample rows for those tables. "
        "Be sure that the tables actually exist by calling "
        f"{list_sql_database_tool.name} first! "
        "Example Input: table1, table2, table3"
    )
    info_sql_database_tool = InfoSQLDatabaseTool(
        db=db, description=info_sql_database_tool_description
    )
    query_sql_database_tool_description = (
        "Input to this tool is a detailed and correct SQL query, output is a "
        "result from the database. If the query is not correct, an error message "
        "will be returned. If an error is returned, rewrite the query, check the "
        "query, and try again. If you encounter an issue with Unknown column "
        f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
        "to query the correct table fields."
    )
    query_sql_database_tool = QuerySQLDatabaseTool(
        db=db, description=query_sql_database_tool_description
    )
    query_sql_checker_tool_description = (
        "Use this tool to double check if your query is correct before executing "
        "it. Always use this tool before executing a query with "
        f"{query_sql_database_tool.name}!"
    )
    # query_sql_checker_tool = QuerySQLCheckerTool(
    #     db=db, llm=self.llm, description=query_sql_checker_tool_description
    # )
    return [
        query_sql_database_tool,
        info_sql_database_tool,
        list_sql_database_tool,
        # query_sql_checker_tool,
    ]
