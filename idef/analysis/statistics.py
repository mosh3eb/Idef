"""
Statistical Functions Module for IDEF.

This module provides common statistical operations on datasets with visualization integration.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class StatResult:
    """Container for statistical results."""
    name: str
    value: Union[float, np.ndarray]
    metadata: Dict = None
    visualization_hints: Dict = None

class StatFunction:
    """Base interface for statistical functions."""
    
    def __init__(self, name: str):
        self.name = name
        
    def compute(self, data: np.ndarray, **kwargs) -> StatResult:
        """Compute the statistical function on the data."""
        raise NotImplementedError

class DescriptiveStats(StatFunction):
    """Computes basic descriptive statistics."""
    
    def compute(self, data: np.ndarray, **kwargs) -> StatResult:
        stats = {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75)
        }
        
        viz_hints = {
            'suggested_plot': 'box_plot',
            'summary_stats': list(stats.keys())
        }
        
        return StatResult(
            name='descriptive_stats',
            value=stats,
            metadata={'n_samples': len(data)},
            visualization_hints=viz_hints
        )

class Correlation(StatFunction):
    """Computes correlation between variables."""
    
    def compute(self, data: np.ndarray, method: str = 'pearson', **kwargs) -> StatResult:
        if method not in ['pearson', 'spearman']:
            raise ValueError(f"Unknown correlation method: {method}")
            
        if method == 'pearson':
            corr_matrix = np.corrcoef(data.T)
        else:
            from scipy.stats import spearmanr
            corr_matrix, _ = spearmanr(data)
            
        viz_hints = {
            'suggested_plot': 'heatmap',
            'colormap': 'RdBu_r',
            'symmetric': True
        }
        
        return StatResult(
            name=f'{method}_correlation',
            value=corr_matrix,
            metadata={'method': method},
            visualization_hints=viz_hints
        )

class Histogram(StatFunction):
    """Computes histogram statistics."""
    
    def compute(self, data: np.ndarray, bins: int = 'auto', **kwargs) -> StatResult:
        hist, bin_edges = np.histogram(data, bins=bins)
        
        viz_hints = {
            'suggested_plot': 'histogram',
            'bin_edges': bin_edges
        }
        
        return StatResult(
            name='histogram',
            value=hist,
            metadata={'bins': bins},
            visualization_hints=viz_hints
        )

# Factory for creating statistical functions
class StatFunctionFactory:
    """Factory for creating statistical function instances."""
    
    _functions = {
        'descriptive': DescriptiveStats,
        'correlation': Correlation,
        'histogram': Histogram
    }
    
    @classmethod
    def create(cls, func_type: str) -> StatFunction:
        """Create a statistical function instance."""
        if func_type not in cls._functions:
            raise ValueError(f"Unknown statistical function: {func_type}")
        return cls._functions[func_type](func_type)
    
    @classmethod
    def available_functions(cls) -> List[str]:
        """List available statistical functions."""
        return list(cls._functions.keys())
