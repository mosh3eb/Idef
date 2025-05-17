"""
Tests for the feature extraction and selection module.
"""

import numpy as np
import pandas as pd
import pytest
from idef.analysis.features import (
    FeatureResult, FeatureExtractor, StatisticalFeatures,
    FrequencyFeatures, TextFeatures, FeatureSelector,
    FeatureExtractorFactory
)

@pytest.fixture
def sample_numerical_data():
    """Create sample numerical data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'A': np.random.normal(0, 1, 1000),
        'B': np.random.exponential(2, 1000),
        'C': np.random.uniform(-1, 1, 1000),
        'target': np.random.choice([0, 1], 1000)
    })

@pytest.fixture
def sample_time_series():
    """Create sample time series data."""
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    signal = (
        2 * np.sin(2 * np.pi * t / 20) +  # Main frequency
        0.5 * np.sin(2 * np.pi * t / 5) +  # Higher frequency
        0.05 * t +  # Trend
        np.random.normal(0, 0.5, len(t))  # Noise
    )
    return pd.Series(signal, index=t)

@pytest.fixture
def sample_text_data():
    """Create sample text data."""
    return pd.Series([
        "This is a sample text document about data science",
        "Another document discussing machine learning",
        "Text analysis and natural language processing",
        "Feature extraction from text data"
    ])

def test_feature_result():
    """Test FeatureResult dataclass."""
    result = FeatureResult(
        name='test_features',
        features=pd.DataFrame({'feat1': [1, 2, 3]}),
        metadata={'method': 'statistical'},
        feature_importance={'feat1': 0.8}
    )
    assert result.name == 'test_features'
    assert isinstance(result.features, pd.DataFrame)
    assert result.metadata == {'method': 'statistical'}
    assert result.feature_importance == {'feat1': 0.8}

def test_statistical_features(sample_numerical_data):
    """Test statistical feature extraction."""
    extractor = StatisticalFeatures('test_stats')
    result = extractor.compute(sample_numerical_data)
    
    assert isinstance(result, FeatureResult)
    assert result.name == 'statistical_features'
    assert isinstance(result.features, pd.DataFrame)
    
    # Check computed features
    features = result.features
    assert 'A_mean' in features.columns
    assert 'A_std' in features.columns
    assert 'A_skew' in features.columns
    assert 'B_kurtosis' in features.columns
    
    # Test with invalid input
    with pytest.raises(ValueError):
        extractor.compute(pd.DataFrame({'A': ['a', 'b', 'c']}))

def test_frequency_features(sample_time_series):
    """Test frequency-based feature extraction."""
    extractor = FrequencyFeatures('test_freq')
    result = extractor.compute(sample_time_series)
    
    assert isinstance(result, FeatureResult)
    assert result.name == 'frequency_features'
    
    # Check computed features
    features = result.features
    assert 'main_frequency' in features.columns
    assert 'power_spectrum' in features.columns
    assert 'frequency_ratio' in features.columns
    
    # Verify detected frequencies
    main_freq = features['main_frequency'].iloc[0]
    assert pytest.approx(1/main_freq, rel=0.1) == 20  # Main period

def test_text_features(sample_text_data):
    """Test text feature extraction."""
    extractor = TextFeatures('test_text')
    result = extractor.compute(sample_text_data)
    
    assert isinstance(result, FeatureResult)
    assert result.name == 'text_features'
    
    # Check computed features
    features = result.features
    assert 'word_count' in features.columns
    assert 'unique_words' in features.columns
    assert 'tfidf' in result.metadata
    
    # Test with different parameters
    result = extractor.compute(
        sample_text_data,
        vectorizer='count',
        remove_stopwords=True
    )
    assert 'bow' in result.metadata

def test_feature_selector(sample_numerical_data):
    """Test feature selection functionality."""
    selector = FeatureSelector()
    
    # Test correlation-based selection
    selected = selector.select_by_correlation(
        sample_numerical_data.drop('target', axis=1),
        threshold=0.5
    )
    assert isinstance(selected, list)
    assert len(selected) <= sample_numerical_data.shape[1]
    
    # Test importance-based selection
    selected = selector.select_by_importance(
        sample_numerical_data.drop('target', axis=1),
        sample_numerical_data['target'],
        method='mutual_info',
        k=2
    )
    assert len(selected) == 2

def test_feature_extractor_factory():
    """Test FeatureExtractorFactory."""
    # Test creation of each extractor type
    stats = FeatureExtractorFactory.create('statistical')
    assert isinstance(stats, StatisticalFeatures)
    
    freq = FeatureExtractorFactory.create('frequency')
    assert isinstance(freq, FrequencyFeatures)
    
    text = FeatureExtractorFactory.create('text')
    assert isinstance(text, TextFeatures)
    
    # Test invalid extractor type
    with pytest.raises(ValueError):
        FeatureExtractorFactory.create('invalid')
    
    # Test available extractors
    available = FeatureExtractorFactory.available_extractors()
    assert set(available) == {'statistical', 'frequency', 'text'}

def test_feature_persistence(sample_numerical_data, tmp_path):
    """Test saving and loading feature results."""
    extractor = StatisticalFeatures('test_stats')
    result = extractor.compute(sample_numerical_data)
    
    # Save result
    save_path = tmp_path / 'feature_result.pkl'
    result.save(save_path)
    
    # Load result
    loaded_result = FeatureResult.load(save_path)
    assert loaded_result.name == result.name
    pd.testing.assert_frame_equal(
        loaded_result.features,
        result.features
    ) 