import os
from .base import BaseTool
from .utils import execute_tool_action

def build_tools(
    benchmark_name: str,
    db_host:str=None, 
    db_port=None,  # default port for postgresql
    db_name=None,
    db_user_name=None,
    db_user_password=None,
    secret_token=None, 
    db_path: str=None,
    db_schema: str=None,
    db_dialect = "postgresql",      # e.g., "mysql", "sqlite", "oracle"
    db_driver = "psycopg2",       # e.g., "pymysql", "aiosqlite", None (for default driver)
) -> list[BaseTool]:
    """ Builds a list of tools based on the specified benchmark. 
    The tool calling arguments should be provided via environment variables. 
    But some can also be passed in directly.
    """
    
    db_host= db_host or os.getenv("db_host", "localhost")
    db_port= db_port or int(os.getenv("db_port", 5432))
    db_name= db_name or os.getenv("db_name", "clue")
    db_user_name= db_user_name or os.getenv("db_user_name", "clueuser")
    db_user_password= db_user_password or os.getenv("db_user_password", "cluepass")
    db_schema= db_schema or os.getenv("db_schema", "public")
    
    def get_db_connection():
        driver_part = f"+{db_driver}" if db_driver else "" # 构建 driver 部分（如果 driver 为空，则不加 +driver）
        if db_path is None:
            connection =f"{db_dialect}{driver_part}://{db_user_name}:{db_user_password}@{db_host}:{db_port}/{db_name}"
        else:
            connection = db_path
            assert db_dialect == "sqlite", "db_dialect must be sqlite when db_path is not None"
        return connection
        
    if benchmark_name == "mapeval":
        from datasets import load_dataset
        from ..clients.mapeval_client import MapEvalClient
        from .mapeval_tools import (
            TravelTimeTool,
            PlaceDetailsTool,
            PlaceSearchTool,
            DirectionsTool,
            NearbyPlacesTool,
        )
        assert "http" not in db_host, "Please provide only host address without http:// or https://"
        client = MapEvalClient(base_url=f"http://{db_host}:{db_port}/api", timeout=30, bearer_token=secret_token)
        return [
            PlaceSearchTool(client=client),
            PlaceDetailsTool(client=client),
            NearbyPlacesTool(client=client),
            TravelTimeTool(client=client),
            DirectionsTool(client=client),
        ]
    elif benchmark_name == "sql":
        from ..clients.sql_client import SQLDBClient
        from .sql_tools import QuerySQLDatabaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool
        connection = get_db_connection()
        db_client = SQLDBClient(connection, schema=db_schema)
        return [
            QuerySQLDatabaseTool(client=db_client),
            InfoSQLDatabaseTool(client=db_client),
            ListSQLDatabaseTool(client=db_client),
        ]
    elif benchmark_name == "geosql":
        from ..clients.sql_client import SQLDBClient
        from ..clients.geosql_client import GeoSQLDBClient
        from ..clients.als_client import AmazonLocationClient
        from ..clients.pdf_client import PDFClient
        from .geosql_tools import ListSpatialFunctionsTool, InfoSpatialFunctionTool, UniqueValuesTool
        from .sql_tools import QuerySQLDatabaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool
        from .aws_geocode import AWSGeocodeTool
        from .pdf_tools import PDFQueryTool
        
        connection = get_db_connection()
        db_client = SQLDBClient(connection, schema=db_schema)
        geosql_db_client = GeoSQLDBClient(connection, schema=db_schema)
        geocode_client = AmazonLocationClient()
        client = PDFClient(storage_path="qdrant_pdf/")
        
        tools = [
            QuerySQLDatabaseTool(client=db_client),
            InfoSQLDatabaseTool(client=db_client),
            ListSQLDatabaseTool(client=db_client),
        ] + [
            ListSpatialFunctionsTool(geosql_db_client),
            InfoSpatialFunctionTool(geosql_db_client),
            UniqueValuesTool(geosql_db_client),
        ] + [ 
             AWSGeocodeTool(geocode_client),
             PDFQueryTool(client=client)
        ]
        return tools
    else:
        raise ValueError(f"Unsupported Benchmark: {benchmark_name}")