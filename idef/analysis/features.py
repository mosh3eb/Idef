"""
Feature Extraction Module for IDEF.

This module provides feature extraction capabilities for complex datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
from pathlib import Path

@dataclass
class FeatureResult:
    """Container for feature extraction results."""
    name: str
    features: pd.DataFrame
    metadata: Dict
    feature_importance: Dict[str, float]
    
    def save(self, path: Union[str, Path]):
        """Save feature result to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FeatureResult':
        """Load feature result from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)

class FeatureExtractor:
    """Base class for feature extraction."""
    
    def __init__(self, name: str):
        self.name = name
        
    def compute(self, data: Union[pd.DataFrame, pd.Series]) -> FeatureResult:
        """Compute features from the data."""
        raise NotImplementedError

class StatisticalFeatures(FeatureExtractor):
    """Extracts statistical features from numerical data."""
    
    def compute(self, data: pd.DataFrame) -> FeatureResult:
        if not all(data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("All columns must be numeric")
        
        features = pd.DataFrame()
        for col in data.columns:
            if col != 'target':
                features[f'{col}_mean'] = [data[col].mean()]
                features[f'{col}_std'] = [data[col].std()]
                features[f'{col}_skew'] = [data[col].skew()]
                features[f'{col}_kurtosis'] = [data[col].kurtosis()]
        
        importance = {col: 1.0 for col in features.columns}
        
        return FeatureResult(
            name='statistical_features',
            features=features,
            metadata={'method': 'statistical'},
            feature_importance=importance
        )

class FrequencyFeatures(FeatureExtractor):
    """Extracts frequency-based features from time series data."""
    
    def compute(self, data: pd.Series) -> FeatureResult:
        # Compute FFT
        fft = np.fft.fft(data.values)
        freqs = np.fft.fftfreq(len(data), d=1)
        power = np.abs(fft)
        
        # Find main frequency
        main_freq_idx = np.argmax(power[1:]) + 1
        main_freq = freqs[main_freq_idx]
        
        features = pd.DataFrame({
            'main_frequency': [main_freq],
            'power_spectrum': [np.sum(power)],
            'frequency_ratio': [power[main_freq_idx] / np.sum(power)]
        })
        
        importance = {col: 1.0 for col in features.columns}
        
        return FeatureResult(
            name='frequency_features',
            features=features,
            metadata={'sampling_rate': 1/np.mean(np.diff(data.index))},
            feature_importance=importance
        )

class TextFeatures(FeatureExtractor):
    """Extracts features from text data."""
    
    def compute(self, data: pd.Series, vectorizer: str = 'tfidf',
               remove_stopwords: bool = False) -> FeatureResult:
        # Basic features
        features = pd.DataFrame({
            'word_count': data.str.split().str.len(),
            'unique_words': data.str.split().apply(lambda x: len(set(x)))
        })
        
        # Vectorization
        if vectorizer == 'tfidf':
            vec = TfidfVectorizer(stop_words='english' if remove_stopwords else None)
            matrix = vec.fit_transform(data)
            metadata = {'tfidf': True, 'vocabulary': vec.vocabulary_}
        else:  # count vectorizer
            vec = CountVectorizer(stop_words='english' if remove_stopwords else None)
            matrix = vec.fit_transform(data)
            metadata = {'bow': True, 'vocabulary': vec.vocabulary_}
        
        importance = {col: 1.0 for col in features.columns}
        
        return FeatureResult(
            name='text_features',
            features=features,
            metadata=metadata,
            feature_importance=importance
        )

class FeatureSelector:
    """Handles feature selection tasks."""
    
    def select_by_correlation(self, data: pd.DataFrame,
                            threshold: float = 0.5) -> List[str]:
        """Select features based on correlation threshold."""
        corr = data.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns 
                  if any(upper[column] > threshold)]
        return [col for col in data.columns if col not in to_drop]
    
    def select_by_importance(self, X: pd.DataFrame, y: pd.Series,
                           method: str = 'mutual_info',
                           k: int = 10) -> List[str]:
        """Select features based on importance scores."""
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        else:
            selector = SelectKBest(f_classif, k=k)
            
        selector.fit(X, y)
        mask = selector.get_support()
        return list(X.columns[mask])

class FeatureExtractorFactory:
    """Factory for creating feature extractors."""
    
    _extractors = {
        'statistical': StatisticalFeatures,
        'frequency': FrequencyFeatures,
        'text': TextFeatures
    }
    
    @classmethod
    def create(cls, extractor_type: str) -> FeatureExtractor:
        """Create a feature extractor instance."""
        if extractor_type not in cls._extractors:
            raise ValueError(f"Unknown feature extractor: {extractor_type}")
        return cls._extractors[extractor_type](extractor_type)
    
    @classmethod
    def available_extractors(cls) -> List[str]:
        """List available feature extractors."""
        return list(cls._extractors.keys())
