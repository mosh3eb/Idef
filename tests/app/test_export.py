"""
Tests for the export module.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from idef.app.export import ExportManager, ExportMetadata

@pytest.fixture
def temp_export_dir(tmp_path):
    """Create a temporary export directory."""
    export_dir = tmp_path / '.idef' / 'exports'
    export_dir.mkdir(parents=True)
    return export_dir

@pytest.fixture
def export_manager(temp_export_dir):
    """Create an ExportManager instance with temporary directory."""
    return ExportManager(str(temp_export_dir))

@pytest.fixture
def sample_data():
    """Create sample data for testing exports."""
    np.random.seed(42)
    return {
        'dataframe': pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.choice(['X', 'Y', 'Z'], 100)
        }),
        'series': pd.Series(np.random.randn(100)),
        'dict': {'key1': 1, 'key2': 'value2', 'key3': [1, 2, 3]},
        'array': np.random.randn(100, 3)
    }

def test_export_data_csv(export_manager, sample_data):
    """Test exporting data to CSV format."""
    # Test DataFrame export
    df_path = export_manager.export_data(
        sample_data['dataframe'],
        'test_df.csv',
        format='csv'
    )
    assert os.path.exists(df_path)
    loaded_df = pd.read_csv(df_path)
    pd.testing.assert_frame_equal(
        loaded_df,
        sample_data['dataframe'],
        check_dtype=False
    )
    
    # Test Series export
    series_path = export_manager.export_data(
        sample_data['series'],
        'test_series.csv',
        format='csv'
    )
    assert os.path.exists(series_path)

def test_export_data_json(export_manager, sample_data):
    """Test exporting data to JSON format."""
    # Test dictionary export
    dict_path = export_manager.export_data(
        sample_data['dict'],
        'test_dict.json',
        format='json'
    )
    assert os.path.exists(dict_path)
    with open(dict_path) as f:
        loaded_dict = json.load(f)
    assert loaded_dict == sample_data['dict']
    
    # Test DataFrame export
    df_path = export_manager.export_data(
        sample_data['dataframe'],
        'test_df.json',
        format='json'
    )
    assert os.path.exists(df_path)
    loaded_df = pd.read_json(df_path)
    pd.testing.assert_frame_equal(
        loaded_df,
        sample_data['dataframe']
    )

def test_export_data_pickle(export_manager, sample_data):
    """Test exporting data to pickle format."""
    import pickle
    
    # Test array export
    array_path = export_manager.export_data(
        sample_data['array'],
        'test_array.pkl',
        format='pickle'
    )
    assert os.path.exists(array_path)
    with open(array_path, 'rb') as f:
        loaded_array = pickle.load(f)
    np.testing.assert_array_equal(loaded_array, sample_data['array'])

def test_export_bundle(export_manager, sample_data):
    """Test creating and loading export bundles."""
    items = [
        {
            'content': sample_data['dataframe'],
            'filename': 'data.csv',
            'type': 'data',
            'format': 'csv',
            'description': 'Sample DataFrame',
            'tags': ['test', 'sample']
        },
        {
            'content': sample_data['dict'],
            'filename': 'config.json',
            'type': 'data',
            'format': 'json',
            'description': 'Sample configuration',
            'tags': ['config']
        }
    ]
    
    # Create bundle
    bundle_path = export_manager.create_export_bundle(items, 'test_bundle.zip')
    assert os.path.exists(bundle_path)
    
    # Load bundle
    loaded_items = export_manager.load_export_bundle(bundle_path)
    assert len(loaded_items) == 2
    assert 'data.csv' in loaded_items
    assert 'config.json' in loaded_items
    
    # Check metadata
    data_metadata = loaded_items['data.csv']['metadata']
    assert data_metadata['content_type'] == 'data'
    assert data_metadata['format'] == 'csv'
    assert 'test' in data_metadata['tags']

def test_invalid_format(export_manager, sample_data):
    """Test handling of invalid export formats."""
    with pytest.raises(ValueError):
        export_manager.export_data(
            sample_data['dataframe'],
            'test.invalid',
            format='invalid'
        )

def test_metadata_creation(export_manager):
    """Test creation of export metadata."""
    metadata = export_manager._create_metadata(
        content_type='data',
        format='csv',
        description='Test data',
        tags=['test', 'sample']
    )
    
    assert isinstance(metadata, ExportMetadata)
    assert metadata.content_type == 'data'
    assert metadata.format == 'csv'
    assert metadata.description == 'Test data'
    assert metadata.tags == ['test', 'sample']
    assert metadata.version == '1.0' 