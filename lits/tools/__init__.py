from .base import BaseTool

def build_tools(
    client_type: str,
    client_host:str, 
    client_port=5432,  # default port for postgresql
    db_name="clue",
    db_user_name="clueuser",
    db_user_password="cluepassword",
    secret_token=None, 
    db_path: str=None,
    db_dialect = "postgresql",      # e.g., "mysql", "sqlite", "oracle"
    db_driver = "psycopg2",       # e.g., "pymysql", "aiosqlite", None (for default driver)
) -> list[BaseTool]:
    def get_connection():
        driver_part = f"+{db_driver}" if db_driver else "" # 构建 driver 部分（如果 driver 为空，则不加 +driver）
        if db_path is None:
            connection =f"{db_dialect}{driver_part}://{db_user_name}:{db_user_password}@{client_host}:{client_port}/{db_name}"
        else:
            connection = db_path
            assert db_dialect == "sqlite", "db_dialect must be sqlite when db_path is not None"
        return connection
        
    if client_type == "mapeval":
        from datasets import load_dataset
        from ..clients.mapeval_client import MapEvalClient
        from .mapeval_tools import (
            TravelTimeTool,
            PlaceDetailsTool,
            PlaceSearchTool,
            DirectionsTool,
            NearbyPlacesTool,
        )
        client = MapEvalClient(base_url=f"http://{client_host}:{client_port}/api", timeout=30, bearer_token=secret_token)
        return [
            PlaceSearchTool(client=client),
            PlaceDetailsTool(client=client),
            NearbyPlacesTool(client=client),
            TravelTimeTool(client=client),
            DirectionsTool(client=client),
        ]
    elif client_type == "sql":
        from ..clients.sql_client import SQLDBClient
        from .sql_tools import QuerySQLDatabaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool
        connection = get_connection()
        db_client = SQLDBClient(connection)
        return [
            QuerySQLDatabaseTool(client=db_client),
            InfoSQLDatabaseTool(client=db_client),
            ListSQLDatabaseTool(client=db_client),
        ]
    elif client_type == "geosql":
        from ..clients.sql_client import SQLDBClient
        from ..clients.geosql_client import GeoSQLDBClient
        from .geosql_tools import ListSpatialFunctionsTool, InfoSpatialFunctionTool, UniqueValuesTool
        from .sql_tools import QuerySQLDatabaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool
        connection = get_connection()
        db_client = SQLDBClient(connection)
        geosql_db_client = GeoSQLDBClient(connection)
        tools = [
            QuerySQLDatabaseTool(client=db_client),
            InfoSQLDatabaseTool(client=db_client),
            ListSQLDatabaseTool(client=db_client),
        ] + [
            ListSpatialFunctionsTool(geosql_db_client),
            InfoSpatialFunctionTool(geosql_db_client),
            UniqueValuesTool(geosql_db_client),
        ]
        return tools
    else:
        raise ValueError(f"Unsupported client type: {client_type}")