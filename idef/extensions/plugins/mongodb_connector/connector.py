"""
MongoDB Connector Plugin for IDEF.

This plugin provides connectivity to MongoDB databases for data import/export.
"""

from typing import Any, Dict, Optional
import pandas as pd
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from idef.extensions.plugins import Plugin, PluginMetadata

class MongoDBConnector(Plugin):
    """MongoDB data connector plugin."""

    def __init__(self, metadata: PluginMetadata):
        """Initialize MongoDB connector.
        
        Args:
            metadata: Plugin metadata
        """
        super().__init__(metadata)
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._collection: Optional[Collection] = None
        self._config: Dict[str, Any] = {}

    def initialize(self, **kwargs) -> bool:
        """Initialize MongoDB connection.
        
        Args:
            **kwargs: Connection parameters
                - host: MongoDB host (default: localhost)
                - port: MongoDB port (default: 27017)
                - database: Database name
                - collection: Collection name
                - username: Optional username for authentication
                - password: Optional password for authentication
                - auth_source: Optional authentication database
                
        Returns:
            bool: True if initialization successful
        """
        try:
            # Store configuration
            self._config = {
                'host': kwargs.get('host', 'localhost'),
                'port': kwargs.get('port', 27017),
                'database': kwargs['database'],
                'collection': kwargs['collection']
            }

            # Add authentication if provided
            if 'username' in kwargs and 'password' in kwargs:
                self._config.update({
                    'username': kwargs['username'],
                    'password': kwargs['password'],
                    'authSource': kwargs.get('auth_source', 'admin')
                })

            # Create client
            self._client = MongoClient(**self._config)
            self._db = self._client[self._config['database']]
            self._collection = self._db[self._config['collection']]

            # Test connection
            self._client.server_info()
            return True

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize MongoDB connection: {e}")

    def cleanup(self):
        """Clean up MongoDB connection."""
        if self._client:
            self._client.close()
        self._client = None
        self._db = None
        self._collection = None
        self._config = {}

    def connect(self, **kwargs) -> bool:
        """Connect to MongoDB.
        
        Args:
            **kwargs: Connection parameters (same as initialize)
            
        Returns:
            bool: True if connection successful
        """
        if not self._client:
            return self.initialize(**kwargs)
        return True

    def read(self, query: Dict = None, projection: Dict = None, **kwargs) -> pd.DataFrame:
        """Read data from MongoDB.
        
        Args:
            query: MongoDB query filter
            projection: MongoDB projection
            **kwargs: Additional query parameters
                - limit: Maximum number of documents
                - skip: Number of documents to skip
                - sort: Sort specification
                
        Returns:
            pd.DataFrame: Query results as DataFrame
        """
        if not self._collection:
            raise RuntimeError("Not connected to MongoDB")

        try:
            # Build query
            cursor = self._collection.find(
                filter=query or {},
                projection=projection,
                limit=kwargs.get('limit'),
                skip=kwargs.get('skip'),
                sort=kwargs.get('sort')
            )

            # Convert to DataFrame
            data = list(cursor)
            if not data:
                return pd.DataFrame()

            return pd.DataFrame(data)

        except Exception as e:
            raise RuntimeError(f"Failed to read from MongoDB: {e}")

    def write(self, data: pd.DataFrame, **kwargs) -> bool:
        """Write DataFrame to MongoDB.
        
        Args:
            data: DataFrame to write
            **kwargs: Write options
                - mode: Write mode ('insert' or 'replace', default: 'insert')
                
        Returns:
            bool: True if write successful
        """
        if not self._collection:
            raise RuntimeError("Not connected to MongoDB")

        try:
            # Convert DataFrame to records
            records = data.to_dict('records')
            
            # Handle write mode
            mode = kwargs.get('mode', 'insert')
            if mode == 'replace':
                self._collection.delete_many({})
            
            # Insert records
            result = self._collection.insert_many(records)
            return bool(result.inserted_ids)

        except Exception as e:
            raise RuntimeError(f"Failed to write to MongoDB: {e}")

    def get_databases(self) -> list:
        """Get list of available databases.
        
        Returns:
            list: Database names
        """
        if not self._client:
            raise RuntimeError("Not connected to MongoDB")
        return self._client.list_database_names()

    def get_collections(self, database: str = None) -> list:
        """Get list of collections in database.
        
        Args:
            database: Database name (default: current database)
            
        Returns:
            list: Collection names
        """
        if not self._client:
            raise RuntimeError("Not connected to MongoDB")

        db = self._client[database] if database else self._db
        if not db:
            raise ValueError("No database specified")

        return db.list_collection_names()

    @property
    def is_connected(self) -> bool:
        """Check if connected to MongoDB.
        
        Returns:
            bool: True if connected
        """
        return bool(self._client and self._collection) 