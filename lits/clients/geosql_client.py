from langchain_community.utilities.sql_database import SQLDatabase
from typing import Iterable, List, Type, Optional
import geoalchemy2.functions as geofuncs
from geoalchemy2 import Geometry
from geoalchemy2.shape import to_shape
import inspect
from sqlalchemy import Table, Column, select
from sqlalchemy.orm import Session
import pandas as pd
import geopandas as gpd
from .sql_client import SQLDBClient
from .base import BaseClient

class GeoSQLDBClient(SQLDBClient):
    """SQL database with spatial functions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_spatial_functions()
        self.add_spatial_tables()
    
    def add_spatial_functions(self):
        """Add spatial functions to the database."""
        clsmembers = inspect.getmembers(geofuncs, inspect.isclass)
        infos = []
        for cls in clsmembers:
            name, obj = cls
            if name.startswith("ST_"):
                infos.append({'name':name, 'description':obj.__doc__})
        self._spatial_functions = infos

    def add_spatial_tables(self):
        """record the all the names of spatial tables"""
        tables = []
        for t in self.db._metadata.sorted_tables:
            geom_column = next((column.name for column in t.columns if isinstance(column.type, Geometry)), None)
            if geom_column is not None: #geom_column
                
                pk_column = t.primary_key.columns[0].name if t.primary_key.columns else None
                tables.append({'name':t.name, 'geom_column':geom_column, 'pk_column': pk_column})
        self._spatial_tables = tables
        
    def get_spatial_function_names(self) -> Iterable[str]:
        """Get a list of spatial functions."""
        return sorted([i['name'] for i in self._spatial_functions])
    
    def get_spatial_function_info(self, function_names: Optional[List[str]] = None) -> str:
        """Get information on the spatial functions."""
        functions_info = [f"Function: {i['name']}\nDescription: {i['description']}" for i in self._spatial_functions if i['name'] in function_names]
        return "\n\n".join(functions_info)
    
    def get_column_unique_values(self, table: str, column: str) -> str:
        """Get unique values in a column."""
        tables = [t for t in self.db._metadata.sorted_tables if t.name == table]
        if not tables:
            return f"Table {table} not found."
        t = tables[0]
        c = t.columns[column]
        return ', '.join([r[column] for r in self._get_column_unique_values(c)])
    
    def _get_column_unique_values(self, column: Column) -> List[dict]:
        """Get unique values in a column."""
        query = select(column).distinct()
        result = self.db._execute(query)
        return result

    def _spatial_augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Augment a dataframe with spatial information."""
        # get a list of spatial tables
        spatial_table_pks = [x['pk_column'] for x in self._spatial_tables]
        candidate_columns = [x for x in df.columns if x in spatial_table_pks]
        if candidate_columns:
            join_key = candidate_columns[0]
            tb_idx = spatial_table_pks.index(join_key)
            tb = self._spatial_tables[tb_idx]
            tb_entity = [x for x in self.db._metadata.sorted_tables if x.name == tb['name']][0]
            session = Session(self.db._engine)
            query = session.query(tb_entity).all()
            dicts = [{join_key: getattr(t, join_key), 'geom': getattr(t, tb['geom_column'])} for t in query]
            full_df = pd.DataFrame(dicts)
            final_df = df.merge(full_df, on=join_key, how='left')
            final_df['geom'] = final_df['geom'].apply(lambda x: to_shape(x) if x is not None else None)
            return gpd.GeoDataFrame(final_df, geometry='geom')
        else:
            return df
        
    def fetch_dataframe(self, query: str) -> pd.DataFrame:
        """Fetch a dataframe from the database."""
        df = pd.read_sql(query, self.db._engine)
        return self._spatial_augment_dataframe(df)
