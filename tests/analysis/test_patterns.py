"""
Tests for the pattern detection and clustering module.
"""

import numpy as np
import pandas as pd
import pytest
from idef.analysis.patterns import (
    PatternResult, ClusteringFunction, KMeansClustering,
    DBSCANClustering, TimeSeriesPatterns, PatternFunctionFactory
)

@pytest.fixture
def sample_data():
    """Create sample data for testing patterns."""
    np.random.seed(42)
    # Create clusterable data with 3 clear clusters
    cluster1 = np.random.normal(0, 1, (100, 2))
    cluster2 = np.random.normal(5, 1, (100, 2))
    cluster3 = np.random.normal(-5, 1, (100, 2))
    return np.vstack([cluster1, cluster2, cluster3])

@pytest.fixture
def time_series_data():
    """Create sample time series data."""
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    # Create a signal with clear seasonal and trend patterns
    seasonal = 2 * np.sin(2 * np.pi * t / 20)
    trend = 0.05 * t
    noise = np.random.normal(0, 0.5, len(t))
    return pd.Series(seasonal + trend + noise, index=t)

def test_pattern_result():
    """Test PatternResult dataclass."""
    result = PatternResult(
        name='test_pattern',
        patterns={'cluster_1': [1, 2, 3]},
        metadata={'algorithm': 'kmeans'},
        visualization_hints={'plot_type': 'scatter'}
    )
    assert result.name == 'test_pattern'
    assert result.patterns == {'cluster_1': [1, 2, 3]}
    assert result.metadata == {'algorithm': 'kmeans'}
    assert result.visualization_hints == {'plot_type': 'scatter'}

def test_kmeans_clustering(sample_data):
    """Test KMeans clustering implementation."""
    kmeans = KMeansClustering('test_kmeans')
    result = kmeans.compute(sample_data, n_clusters=3)
    
    assert isinstance(result, PatternResult)
    assert result.name == 'kmeans_clusters'
    assert 'labels' in result.patterns
    assert len(np.unique(result.patterns['labels'])) == 3
    assert 'centroids' in result.patterns
    assert len(result.patterns['centroids']) == 3
    
    # Test with invalid number of clusters
    with pytest.raises(ValueError):
        kmeans.compute(sample_data, n_clusters=0)

def test_dbscan_clustering(sample_data):
    """Test DBSCAN clustering implementation."""
    dbscan = DBSCANClustering('test_dbscan')
    result = dbscan.compute(sample_data, eps=1.0, min_samples=5)
    
    assert isinstance(result, PatternResult)
    assert result.name == 'dbscan_clusters'
    assert 'labels' in result.patterns
    assert 'core_samples' in result.patterns
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        dbscan.compute(sample_data, eps=-1.0)

def test_time_series_patterns(time_series_data):
    """Test time series pattern detection."""
    ts_patterns = TimeSeriesPatterns('test_ts')
    result = ts_patterns.compute(time_series_data)
    
    assert isinstance(result, PatternResult)
    assert result.name == 'time_series_patterns'
    assert 'seasonality' in result.patterns
    assert 'trend' in result.patterns
    assert 'changepoints' in result.patterns
    
    # Test seasonality detection
    assert result.patterns['seasonality']['period'] == pytest.approx(20, rel=0.1)
    
    # Test trend detection
    assert result.patterns['trend']['slope'] == pytest.approx(0.05, rel=0.1)

def test_pattern_function_factory():
    """Test PatternFunctionFactory."""
    # Test creation of each function type
    kmeans = PatternFunctionFactory.create('kmeans')
    assert isinstance(kmeans, KMeansClustering)
    
    dbscan = PatternFunctionFactory.create('dbscan')
    assert isinstance(dbscan, DBSCANClustering)
    
    ts_patterns = PatternFunctionFactory.create('time_series')
    assert isinstance(ts_patterns, TimeSeriesPatterns)
    
    # Test invalid function type
    with pytest.raises(ValueError):
        PatternFunctionFactory.create('invalid')
    
    # Test available functions
    available = PatternFunctionFactory.available_functions()
    assert set(available) == {'kmeans', 'dbscan', 'time_series'}

def test_clustering_with_preprocessing(sample_data):
    """Test clustering with data preprocessing."""
    kmeans = KMeansClustering('test_kmeans')
    
    # Test with scaling
    result = kmeans.compute(
        sample_data,
        n_clusters=3,
        preprocess={'scale': True}
    )
    assert 'preprocessing' in result.metadata
    assert result.metadata['preprocessing']['scale'] == True
    
    # Test with dimensionality reduction
    result = kmeans.compute(
        sample_data,
        n_clusters=3,
        preprocess={'pca_components': 2}
    )
    assert 'pca_components' in result.metadata['preprocessing']

def test_pattern_persistence(sample_data, tmp_path):
    """Test saving and loading pattern results."""
    kmeans = KMeansClustering('test_kmeans')
    result = kmeans.compute(sample_data, n_clusters=3)
    
    # Save result
    save_path = tmp_path / 'pattern_result.json'
    result.save(save_path)
    
    # Load result
    loaded_result = PatternResult.load(save_path)
    assert loaded_result.name == result.name
    assert loaded_result.patterns.keys() == result.patterns.keys()
    np.testing.assert_array_equal(
        loaded_result.patterns['labels'],
        result.patterns['labels']
    ) 