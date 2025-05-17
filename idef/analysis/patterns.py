"""
Pattern Detection Module for IDEF.

This module provides pattern detection and clustering capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
from pathlib import Path

@dataclass
class PatternResult:
    """Container for pattern detection results."""
    name: str
    patterns: Dict[str, Any]
    metadata: Dict
    visualization_hints: Dict
    
    def save(self, path: Union[str, Path]):
        """Save pattern result to disk."""
        data = {
            'name': self.name,
            'patterns': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.patterns.items()
            },
            'metadata': self.metadata,
            'visualization_hints': self.visualization_hints
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PatternResult':
        """Load pattern result from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
            # Convert lists back to numpy arrays where appropriate
            patterns = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in data['patterns'].items()
            }
            return cls(
                name=data['name'],
                patterns=patterns,
                metadata=data['metadata'],
                visualization_hints=data['visualization_hints']
            )

class ClusteringFunction:
    """Base class for clustering functions."""
    
    def __init__(self, name: str):
        self.name = name
    
    def compute(self, data: np.ndarray, **kwargs) -> PatternResult:
        """Compute patterns in the data."""
        raise NotImplementedError
    
    def _preprocess_data(self, data: np.ndarray,
                        preprocess: Optional[Dict] = None) -> np.ndarray:
        """Preprocess data before pattern detection."""
        if not preprocess:
            return data
            
        processed = data.copy()
        preprocessing_info = {}
        
        # Apply scaling if requested
        if preprocess.get('scale', False):
            scaler = StandardScaler()
            processed = scaler.fit_transform(processed)
            preprocessing_info['scale'] = True
        
        # Apply dimensionality reduction if requested
        if 'pca_components' in preprocess:
            n_components = preprocess['pca_components']
            pca = PCA(n_components=n_components)
            processed = pca.fit_transform(processed)
            preprocessing_info['pca_components'] = n_components
        
        return processed, preprocessing_info

class KMeansClustering(ClusteringFunction):
    """K-means clustering implementation."""
    
    def compute(self, data: np.ndarray, n_clusters: int = 3,
               preprocess: Optional[Dict] = None, **kwargs) -> PatternResult:
        if n_clusters <= 0:
            raise ValueError("Number of clusters must be positive")
        
        # Preprocess data if requested
        if preprocess:
            processed_data, preprocessing_info = self._preprocess_data(data, preprocess)
        else:
            processed_data = data
            preprocessing_info = {}
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, **kwargs)
        labels = kmeans.fit_predict(processed_data)
        
        patterns = {
            'labels': labels,
            'centroids': kmeans.cluster_centers_
        }
        
        metadata = {
            'algorithm': 'kmeans',
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_,
            'preprocessing': preprocessing_info
        }
        
        viz_hints = {
            'plot_type': 'scatter',
            'color_by': 'labels',
            'show_centroids': True
        }
        
        return PatternResult(
            name='kmeans_clusters',
            patterns=patterns,
            metadata=metadata,
            visualization_hints=viz_hints
        )

class DBSCANClustering(ClusteringFunction):
    """DBSCAN clustering implementation."""
    
    def compute(self, data: np.ndarray, eps: float = 0.5,
               min_samples: int = 5, preprocess: Optional[Dict] = None,
               **kwargs) -> PatternResult:
        if eps <= 0:
            raise ValueError("eps must be positive")
        
        # Preprocess data if requested
        if preprocess:
            processed_data, preprocessing_info = self._preprocess_data(data, preprocess)
        else:
            processed_data = data
            preprocessing_info = {}
        
        # Perform clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = dbscan.fit_predict(processed_data)
        
        patterns = {
            'labels': labels,
            'core_samples': dbscan.core_sample_indices_
        }
        
        metadata = {
            'algorithm': 'dbscan',
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'preprocessing': preprocessing_info
        }
        
        viz_hints = {
            'plot_type': 'scatter',
            'color_by': 'labels',
            'highlight_core_samples': True
        }
        
        return PatternResult(
            name='dbscan_clusters',
            patterns=patterns,
            metadata=metadata,
            visualization_hints=viz_hints
        )

class TimeSeriesPatterns(ClusteringFunction):
    """Time series pattern detection."""
    
    def compute(self, data: pd.Series, **kwargs) -> PatternResult:
        # Detect seasonality
        seasonality = self._detect_seasonality(data)
        
        # Detect trend
        trend = self._detect_trend(data)
        
        # Detect change points
        changepoints = self._detect_changepoints(data)
        
        patterns = {
            'seasonality': seasonality,
            'trend': trend,
            'changepoints': changepoints
        }
        
        metadata = {
            'algorithm': 'time_series_patterns',
            'data_length': len(data),
            'sampling_rate': 1/np.mean(np.diff(data.index))
        }
        
        viz_hints = {
            'plot_type': 'line',
            'components': ['original', 'trend', 'seasonal'],
            'mark_changepoints': True
        }
        
        return PatternResult(
            name='time_series_patterns',
            patterns=patterns,
            metadata=metadata,
            visualization_hints=viz_hints
        )
    
    def _detect_seasonality(self, data: pd.Series) -> Dict:
        """Detect seasonal patterns in time series."""
        # Compute FFT
        fft = np.fft.fft(data.values)
        freqs = np.fft.fftfreq(len(data), d=1)
        power = np.abs(fft)
        
        # Find main frequency
        main_freq_idx = np.argmax(power[1:]) + 1
        period = 1/freqs[main_freq_idx]
        
        return {
            'period': abs(period),
            'strength': power[main_freq_idx] / np.sum(power)
        }
    
    def _detect_trend(self, data: pd.Series) -> Dict:
        """Detect trend in time series."""
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data.values, 1)
        
        return {
            'slope': coeffs[0],
            'intercept': coeffs[1]
        }
    
    def _detect_changepoints(self, data: pd.Series) -> List[int]:
        """Detect change points in time series."""
        # Simple implementation using rolling statistics
        window = len(data) // 10
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        
        # Detect points where there are significant changes
        threshold = 2 * rolling_std.mean()
        changes = np.where(np.abs(np.diff(rolling_mean)) > threshold)[0]
        
        return changes.tolist()

class PatternFunctionFactory:
    """Factory for creating pattern detection functions."""
    
    _functions = {
        'kmeans': KMeansClustering,
        'dbscan': DBSCANClustering,
        'time_series': TimeSeriesPatterns
    }
    
    @classmethod
    def create(cls, function_type: str) -> ClusteringFunction:
        """Create a pattern detection function."""
        if function_type not in cls._functions:
            raise ValueError(f"Unknown pattern function: {function_type}")
        return cls._functions[function_type](function_type)
    
    @classmethod
    def available_functions(cls) -> List[str]:
        """List available pattern functions."""
        return list(cls._functions.keys())
