"""
Tests for the data connectors module.
"""

import os
import pytest
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch
from idef.data.connectors import (
    DataConnector, CSVConnector, SQLConnector,
    APIConnector, DataConnectorFactory
)

@pytest.fixture
def sample_csv_data(tmp_path):
    """Create a sample CSV file."""
    data = pd.DataFrame({
        'id': range(100),
        'value': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    file_path = tmp_path / 'test.csv'
    data.to_csv(file_path, index=False)
    return str(file_path), data

@pytest.fixture
def sample_db(tmp_path):
    """Create a sample SQLite database."""
    db_path = tmp_path / 'test.db'
    conn = sqlite3.connect(str(db_path))
    
    # Create test table
    data = pd.DataFrame({
        'id': range(100),
        'value': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    data.to_sql('test_table', conn, index=False)
    conn.close()
    
    return str(db_path), data

@pytest.fixture
def mock_api():
    """Create a mock API endpoint."""
    mock_response = Mock()
    mock_response.json.return_value = {
        'data': [
            {'id': i, 'value': np.random.randn(), 'category': np.random.choice(['A', 'B', 'C'])}
            for i in range(100)
        ]
    }
    mock_response.status_code = 200
    return mock_response

def test_csv_connector(sample_csv_data):
    """Test CSV data connector."""
    file_path, expected_data = sample_csv_data
    connector = CSVConnector()
    
    # Test basic read
    data = connector.read(file_path)
    pd.testing.assert_frame_equal(data, expected_data)
    
    # Test with specific columns
    data = connector.read(file_path, columns=['id', 'value'])
    assert list(data.columns) == ['id', 'value']
    
    # Test with filtering
    data = connector.read(file_path, filters={'category': 'A'})
    assert all(data['category'] == 'A')
    
    # Test with invalid file
    with pytest.raises(FileNotFoundError):
        connector.read('nonexistent.csv')

def test_sql_connector(sample_db):
    """Test SQL data connector."""
    db_path, expected_data = sample_db
    connector = SQLConnector()
    
    # Test basic read
    data = connector.read(
        f'sqlite:///{db_path}',
        query='SELECT * FROM test_table'
    )
    pd.testing.assert_frame_equal(data.reset_index(drop=True), expected_data)
    
    # Test with specific columns
    data = connector.read(
        f'sqlite:///{db_path}',
        query='SELECT id, value FROM test_table'
    )
    assert list(data.columns) == ['id', 'value']
    
    # Test with filtering
    data = connector.read(
        f'sqlite:///{db_path}',
        query="SELECT * FROM test_table WHERE category = 'A'"
    )
    assert all(data['category'] == 'A')
    
    # Test with invalid query
    with pytest.raises(Exception):
        connector.read(f'sqlite:///{db_path}', query='INVALID SQL')

def test_api_connector(mock_api):
    """Test API data connector."""
    connector = APIConnector()
    
    # Test successful API call
    with patch('requests.get', return_value=mock_api):
        data = connector.read(
            'https://api.example.com/data',
            headers={'Authorization': 'Bearer token'}
        )
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert list(data.columns) == ['id', 'value', 'category']
    
    # Test failed API call
    mock_api.status_code = 404
    with patch('requests.get', return_value=mock_api):
        with pytest.raises(Exception):
            connector.read('https://api.example.com/data')

def test_connector_factory():
    """Test DataConnectorFactory."""
    factory = DataConnectorFactory()
    
    # Test creation of each connector type
    csv_connector = factory.create('csv')
    assert isinstance(csv_connector, CSVConnector)
    
    sql_connector = factory.create('sql')
    assert isinstance(sql_connector, SQLConnector)
    
    api_connector = factory.create('api')
    assert isinstance(api_connector, APIConnector)
    
    # Test invalid connector type
    with pytest.raises(ValueError):
        factory.create('invalid')

def test_connector_caching(sample_csv_data, tmp_path):
    """Test connector caching functionality."""
    file_path, expected_data = sample_csv_data
    connector = CSVConnector(cache_dir=str(tmp_path))
    
    # First read should cache
    data1 = connector.read(file_path, use_cache=True)
    pd.testing.assert_frame_equal(data1, expected_data)
    
    # Second read should use cache
    data2 = connector.read(file_path, use_cache=True)
    pd.testing.assert_frame_equal(data1, data2)
    
    # Should skip cache
    data3 = connector.read(file_path, use_cache=False)
    pd.testing.assert_frame_equal(data1, data3)

def test_connector_transformations(sample_csv_data):
    """Test data transformations in connectors."""
    file_path, _ = sample_csv_data
    connector = CSVConnector()
    
    # Test with transformation function
    def transform_func(df):
        df['new_col'] = df['value'] * 2
        return df
    
    data = connector.read(file_path, transform=transform_func)
    assert 'new_col' in data.columns
    assert all(data['new_col'] == data['value'] * 2)
    
    # Test with invalid transformation
    with pytest.raises(Exception):
        connector.read(file_path, transform=lambda x: None)

def test_batch_processing(sample_csv_data):
    """Test batch processing in connectors."""
    file_path, _ = sample_csv_data
    connector = CSVConnector()
    
    # Test batch reading
    batches = list(connector.read_batches(file_path, batch_size=10))
    assert len(batches) == 10
    assert all(len(batch) == 10 for batch in batches[:-1])
    
    # Reconstruct full dataset
    full_data = pd.concat(batches).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        full_data,
        connector.read(file_path)
    ) 