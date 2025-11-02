from typing import Any, Type, List
from pydantic import BaseModel, Field
from .base import BaseTool

# ===== Input Schemas =====
class InfoSpatialFunctionInput(BaseModel):
    function_names: str = Field(
        ...,
        description=(
            "Comma-separated list of spatial function names to describe. "
            "Example: 'ST_Area, ST_Intersection'"
        ),
    )

class UniqueValuesInput(BaseModel):
    table_column: str = Field(
        ...,
        description="A pair of table and column names, separated by a comma. Example: 'places, region_name'"
    )

# ===== Tool Implementations =====
class ListSpatialFunctionsTool(BaseTool):
    """List all spatial functions available."""
    name: str = "list_spatial_functions"
    description: str = "List the spatial (ST_) functions available in the database."
    args_schema: Type[BaseModel] = None  # no arguments

    def _run(self, **kwargs) -> str:
        funcs = self.client.get_spatial_function_names()
        return ", ".join(funcs)
      

class InfoSpatialFunctionTool(BaseTool):
    """Get signatures or schema info for specified spatial functions."""
    name: str = "info_spatial_functions_sql"
    description: str = (
        "Input: a comma-separated list of spatial function names. "
        "Output: their signature and description. "
        "Use list_spatial_functions first to discover available names."
    )
    args_schema: Type[BaseModel] = InfoSpatialFunctionInput

    def __init__(self, client: Any):
        self.client = client

    def _run(self, function_names: str, **kwargs) -> str:
        names = [n.strip() for n in function_names.split(",")]
        return self.client.get_spatial_function_info(names)


class UniqueValuesTool(BaseTool):
    """Get unique values for a specific column in a spatial table."""
    name: str = "unique_values"
    description: str = (
        "Return a comma-separated list of unique values in a column. "
        "Input format: 'table_name, column_name'. "
        "Useful for identifying value variations when string matching fails."
    )
    args_schema: Type[BaseModel] = UniqueValuesInput

    def __init__(self, client: Any):
        self.client = client

    def _run(self, table_column: str, **kwargs) -> str:
        table, column = [x.strip() for x in table_column.split(",")]
        return self.client.get_column_unique_values(table, column)
