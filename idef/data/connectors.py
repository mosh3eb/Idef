"""
Data Connectors Module for IDEF.

This module provides connectors for various data sources.
"""

import os
import pandas as pd
import numpy as np
import sqlite3
import requests
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Iterator
from sqlalchemy import create_engine
import json

class DataConnector:
    """Base class for data connectors."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def read(self, source: str, **kwargs) -> pd.DataFrame:
        """Read data from source."""
        raise NotImplementedError
    
    def read_batches(self, source: str, batch_size: int = 1000,
                    **kwargs) -> Iterator[pd.DataFrame]:
        """Read data in batches."""
        raise NotImplementedError
    
    def _get_cache_path(self, source: str, params: Dict) -> Path:
        """Get cache file path for source and parameters."""
        if not self.cache_dir:
            return None
            
        # Create hash of source and parameters
        params_str = json.dumps(params, sort_keys=True)
        hash_input = f"{source}_{params_str}".encode()
        file_hash = hashlib.md5(hash_input).hexdigest()
        
        return Path(self.cache_dir) / f"{file_hash}.parquet"
    
    def _read_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Read data from cache if available."""
        if cache_path and cache_path.exists():
            return pd.read_parquet(cache_path)
        return None
    
    def _write_cache(self, data: pd.DataFrame, cache_path: Path):
        """Write data to cache."""
        if cache_path:
            data.to_parquet(cache_path)

class CSVConnector(DataConnector):
    """Connector for CSV files."""
    
    def read(self, source: str, columns: List[str] = None,
            filters: Dict = None, transform: Callable = None,
            use_cache: bool = False, **kwargs) -> pd.DataFrame:
        """Read data from CSV file."""
        if not os.path.exists(source):
            raise FileNotFoundError(f"CSV file not found: {source}")
            
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(source, {
                'columns': columns,
                'filters': filters,
                'kwargs': kwargs
            })
            cached_data = self._read_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        # Read CSV file
        data = pd.read_csv(source, **kwargs)
        
        # Apply column selection
        if columns:
            data = data[columns]
        
        # Apply filters
        if filters:
            for col, value in filters.items():
                data = data[data[col] == value]
        
        # Apply transformation
        if transform:
            data = transform(data)
            if not isinstance(data, pd.DataFrame):
                raise Exception("Transform function must return a DataFrame")
        
        # Cache result if requested
        if use_cache:
            self._write_cache(data, cache_path)
        
        return data
    
    def read_batches(self, source: str, batch_size: int = 1000,
                    **kwargs) -> Iterator[pd.DataFrame]:
        """Read CSV file in batches."""
        for chunk in pd.read_csv(source, chunksize=batch_size, **kwargs):
            yield chunk

class SQLConnector(DataConnector):
    """Connector for SQL databases."""
    
    def read(self, source: str, query: str, params: Dict = None,
            use_cache: bool = False, **kwargs) -> pd.DataFrame:
        """Read data from SQL database."""
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(source, {
                'query': query,
                'params': params,
                'kwargs': kwargs
            })
            cached_data = self._read_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        # Create database engine
        engine = create_engine(source)
        
        try:
            # Execute query
            data = pd.read_sql(query, engine, params=params, **kwargs)
            
            # Cache result if requested
            if use_cache:
                self._write_cache(data, cache_path)
            
            return data
        finally:
            engine.dispose()
    
    def read_batches(self, source: str, query: str,
                    batch_size: int = 1000, **kwargs) -> Iterator[pd.DataFrame]:
        """Read SQL query results in batches."""
        engine = create_engine(source)
        try:
            # Modify query to include pagination
            offset = 0
            while True:
                batch_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
                batch = pd.read_sql(batch_query, engine, **kwargs)
                if len(batch) == 0:
                    break
                yield batch
                offset += batch_size
        finally:
            engine.dispose()

class APIConnector(DataConnector):
    """Connector for REST APIs."""
    
    def read(self, source: str, headers: Dict = None,
            params: Dict = None, use_cache: bool = False,
            **kwargs) -> pd.DataFrame:
        """Read data from REST API."""
        # Check cache first
        if use_cache:
            cache_path = self._get_cache_path(source, {
                'headers': headers,
                'params': params,
                'kwargs': kwargs
            })
            cached_data = self._read_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        # Make API request
        response = requests.get(source, headers=headers, params=params, **kwargs)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}")
        
        # Parse response
        data = response.json()
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            raise Exception("Unexpected API response format")
        
        # Cache result if requested
        if use_cache:
            self._write_cache(df, cache_path)
        
        return df
    
    def read_batches(self, source: str, batch_size: int = 1000,
                    **kwargs) -> Iterator[pd.DataFrame]:
        """Read API data in batches using pagination."""
        page = 1
        while True:
            # Add pagination parameters
            params = kwargs.get('params', {})
            params.update({
                'page': page,
                'per_page': batch_size
            })
            kwargs['params'] = params
            
            try:
                batch = self.read(source, **kwargs)
                if len(batch) == 0:
                    break
                yield batch
                page += 1
            except Exception as e:
                break

class DataConnectorFactory:
    """Factory for creating data connectors."""
    
    _connectors = {
        'csv': CSVConnector,
        'sql': SQLConnector,
        'api': APIConnector
    }
    
    @classmethod
    def create(cls, connector_type: str, **kwargs) -> DataConnector:
        """Create a data connector instance."""
        if connector_type not in cls._connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")
        return cls._connectors[connector_type](**kwargs)
    
    @classmethod
    def available_connectors(cls) -> List[str]:
        """List available connector types."""
        return list(cls._connectors.keys())
