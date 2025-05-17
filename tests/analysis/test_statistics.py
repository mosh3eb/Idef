"""
Tests for the statistics module.
"""

import numpy as np
import pytest
from idef.analysis.statistics import (
    StatResult, StatFunction, DescriptiveStats,
    Correlation, Histogram, StatFunctionFactory
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return np.random.randn(100, 3)

def test_stat_result():
    """Test StatResult dataclass."""
    result = StatResult(
        name='test',
        value=1.0,
        metadata={'key': 'value'},
        visualization_hints={'plot': 'line'}
    )
    assert result.name == 'test'
    assert result.value == 1.0
    assert result.metadata == {'key': 'value'}
    assert result.visualization_hints == {'plot': 'line'}

def test_descriptive_stats(sample_data):
    """Test DescriptiveStats computation."""
    stats = DescriptiveStats('test')
    result = stats.compute(sample_data)
    
    assert isinstance(result, StatResult)
    assert result.name == 'descriptive_stats'
    assert isinstance(result.value, dict)
    assert 'mean' in result.value
    assert 'median' in result.value
    assert 'std' in result.value
    assert 'min' in result.value
    assert 'max' in result.value
    assert 'q1' in result.value
    assert 'q3' in result.value
    
    # Test actual computations
    np_mean = np.mean(sample_data)
    assert np.allclose(result.value['mean'], np_mean)

def test_correlation(sample_data):
    """Test Correlation computation."""
    corr = Correlation('test')
    
    # Test Pearson correlation
    result = corr.compute(sample_data, method='pearson')
    assert isinstance(result, StatResult)
    assert result.name == 'pearson_correlation'
    assert isinstance(result.value, np.ndarray)
    assert result.value.shape == (3, 3)  # 3x3 correlation matrix
    
    # Test Spearman correlation
    result = corr.compute(sample_data, method='spearman')
    assert result.name == 'spearman_correlation'
    
    # Test invalid method
    with pytest.raises(ValueError):
        corr.compute(sample_data, method='invalid')

def test_histogram(sample_data):
    """Test Histogram computation."""
    hist = Histogram('test')
    result = hist.compute(sample_data[:, 0])  # Test on 1D data
    
    assert isinstance(result, StatResult)
    assert result.name == 'histogram'
    assert isinstance(result.value, np.ndarray)
    assert 'bin_edges' in result.visualization_hints

def test_stat_function_factory():
    """Test StatFunctionFactory."""
    # Test creation of each function type
    descriptive = StatFunctionFactory.create('descriptive')
    assert isinstance(descriptive, DescriptiveStats)
    
    correlation = StatFunctionFactory.create('correlation')
    assert isinstance(correlation, Correlation)
    
    histogram = StatFunctionFactory.create('histogram')
    assert isinstance(histogram, Histogram)
    
    # Test invalid function type
    with pytest.raises(ValueError):
        StatFunctionFactory.create('invalid')
    
    # Test available functions
    available = StatFunctionFactory.available_functions()
    assert set(available) == {'descriptive', 'correlation', 'histogram'} 