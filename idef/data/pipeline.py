"""
Data transformation pipeline module for the Interactive Data Exploration Framework.
Provides functionality for processing and transforming datasets.
"""

from typing import Dict, List, Union, Optional, Any, Callable, TypeVar
import xarray as xr
import numpy as np
import pandas as pd

from .model import Dataset

T = TypeVar('T')


class Transformer:
    """Base class for data transformations."""
    
    def __init__(self, name: str):
        """
        Initialize a transformer.
        
        Args:
            name: Name of the transformer
        """
        self.name = name
    
    def transform(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Apply the transformation to a dataset.
        
        Args:
            dataset: The dataset to transform
            **kwargs: Additional arguments for the transformation
            
        Returns:
            Dataset: The transformed dataset
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement transform()")


class Pipeline:
    """
    A pipeline for applying a sequence of transformations to a dataset.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a pipeline.
        
        Args:
            name: Optional name for the pipeline
        """
        self.name = name or "Data Pipeline"
        self.steps: List[Dict[str, Any]] = []
    
    def add_step(self, transformer: Union[Transformer, Callable], **kwargs) -> 'Pipeline':
        """
        Add a transformation step to the pipeline.
        
        Args:
            transformer: The transformer to apply
            **kwargs: Additional arguments for the transformer
            
        Returns:
            Pipeline: The pipeline instance (for method chaining)
        """
        if isinstance(transformer, Transformer):
            self.steps.append({
                "transformer": transformer,
                "kwargs": kwargs
            })
        elif callable(transformer):
            self.steps.append({
                "transformer": FunctionTransformer(transformer),
                "kwargs": kwargs
            })
        else:
            raise TypeError(f"Expected Transformer or callable, got {type(transformer)}")
        
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
        """
        Apply all transformation steps to a dataset.
        
        Args:
            dataset: The dataset to transform
            
        Returns:
            Dataset: The transformed dataset
        """
        result = dataset
        for step in self.steps:
            transformer = step["transformer"]
            kwargs = step["kwargs"]
            result = transformer.transform(result, **kwargs)
        
        return result
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        steps_str = "\n".join([f"  - {step['transformer'].name}" for step in self.steps])
        return f"Pipeline('{self.name}', steps=[\n{steps_str}\n])"


class FunctionTransformer(Transformer):
    """Transformer that applies a custom function."""
    
    def __init__(self, func: Callable[[Dataset, ...], Dataset], name: Optional[str] = None):
        """
        Initialize a function transformer.
        
        Args:
            func: The function to apply
            name: Optional name for the transformer
        """
        super().__init__(name or func.__name__)
        self.func = func
    
    def transform(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Apply the function to a dataset.
        
        Args:
            dataset: The dataset to transform
            **kwargs: Additional arguments for the function
            
        Returns:
            Dataset: The transformed dataset
        """
        return self.func(dataset, **kwargs)


# Common transformations

class SelectTransformer(Transformer):
    """Transformer that selects a subset of a dataset."""
    
    def __init__(self):
        """Initialize a select transformer."""
        super().__init__("Select")
    
    def transform(self, dataset: Dataset, **kwargs) -> Dataset:
        """
        Select a subset of a dataset.
        
        Args:
            dataset: The dataset to transform
            **kwargs: Selection criteria for dimensions
            
        Returns:
            Dataset: The selected dataset
        """
        return dataset.select(**kwargs)


class RenameTransformer(Transformer):
    """Transformer that renames variables in a dataset."""
    
    def __init__(self):
        """Initialize a rename transformer."""
        super().__init__("Rename")
    
    def transform(self, dataset: Dataset, name_mapping: Dict[str, str]) -> Dataset:
        """
        Rename variables in a dataset.
        
        Args:
            dataset: The dataset to transform
            name_mapping: Mapping from old names to new names
            
        Returns:
            Dataset: The dataset with renamed variables
        """
        new_data = dataset.data.rename(name_mapping)
        result = Dataset(new_data, name=dataset.name)
        result._metadata = dataset._metadata.copy()
        result._visualization_hints = dataset._visualization_hints.copy()
        return result


class AggregateTransformer(Transformer):
    """Transformer that aggregates data along dimensions."""
    
    def __init__(self):
        """Initialize an aggregate transformer."""
        super().__init__("Aggregate")
    
    def transform(self, dataset: Dataset, 
                  dimensions: Union[str, List[str]], 
                  method: str = "mean") -> Dataset:
        """
        Aggregate data along dimensions.
        
        Args:
            dataset: The dataset to transform
            dimensions: Dimension(s) to aggregate along
            method: Aggregation method ('mean', 'sum', 'min', 'max', etc.)
            
        Returns:
            Dataset: The aggregated dataset
            
        Raises:
            ValueError: If the method is not supported
        """
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        
        if method == "mean":
            new_data = dataset.data.mean(dim=dimensions)
        elif method == "sum":
            new_data = dataset.data.sum(dim=dimensions)
        elif method == "min":
            new_data = dataset.data.min(dim=dimensions)
        elif method == "max":
            new_data = dataset.data.max(dim=dimensions)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        
        result = Dataset(new_data, name=f"{dataset.name} ({method} over {', '.join(dimensions)})")
        result._metadata = dataset._metadata.copy()
        result._visualization_hints = dataset._visualization_hints.copy()
        return result


class ResampleTransformer(Transformer):
    """Transformer that resamples time series data."""
    
    def __init__(self):
        """Initialize a resample transformer."""
        super().__init__("Resample")
    
    def transform(self, dataset: Dataset, 
                  time_dim: str, 
                  freq: str, 
                  method: str = "mean") -> Dataset:
        """
        Resample time series data.
        
        Args:
            dataset: The dataset to transform
            time_dim: Name of the time dimension
            freq: Resampling frequency (e.g., '1D', '1H', '1M')
            method: Aggregation method ('mean', 'sum', 'min', 'max', etc.)
            
        Returns:
            Dataset: The resampled dataset
            
        Raises:
            ValueError: If the time dimension does not exist
        """
        if time_dim not in dataset.dimensions:
            raise ValueError(f"Time dimension '{time_dim}' not found in dataset")
        
        resampler = dataset.data.resample({time_dim: freq})
        
        if method == "mean":
            new_data = resampler.mean()
        elif method == "sum":
            new_data = resampler.sum()
        elif method == "min":
            new_data = resampler.min()
        elif method == "max":
            new_data = resampler.max()
        else:
            raise ValueError(f"Unsupported resampling method: {method}")
        
        result = Dataset(new_data, name=f"{dataset.name} (Resampled to {freq})")
        result._metadata = dataset._metadata.copy()
        result._visualization_hints = dataset._visualization_hints.copy()
        return result


class NormalizeTransformer(Transformer):
    """Transformer that normalizes data."""
    
    def __init__(self):
        """Initialize a normalize transformer."""
        super().__init__("Normalize")
    
    def transform(self, dataset: Dataset, 
                  variables: Optional[List[str]] = None, 
                  method: str = "minmax") -> Dataset:
        """
        Normalize data variables.
        
        Args:
            dataset: The dataset to transform
            variables: Variables to normalize (if None, normalize all)
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            Dataset: The normalized dataset
        """
        if variables is None:
            variables = dataset.variables
        
        new_data = dataset.data.copy()
        
        for var in variables:
            if var not in dataset.variables:
                continue
                
            data_array = dataset.data[var]
            
            if method == "minmax":
                # Min-max normalization to [0, 1]
                min_val = data_array.min().item()
                max_val = data_array.max().item()
                if max_val > min_val:
                    new_data[var] = (data_array - min_val) / (max_val - min_val)
            
            elif method == "zscore":
                # Z-score normalization
                mean_val = data_array.mean().item()
                std_val = data_array.std().item()
                if std_val > 0:
                    new_data[var] = (data_array - mean_val) / std_val
            
            elif method == "robust":
                # Robust normalization using percentiles
                q25 = data_array.quantile(0.25).item()
                q75 = data_array.quantile(0.75).item()
                iqr = q75 - q25
                if iqr > 0:
                    new_data[var] = (data_array - q25) / iqr
            
            else:
                raise ValueError(f"Unsupported normalization method: {method}")
        
        result = Dataset(new_data, name=f"{dataset.name} (Normalized)")
        result._metadata = dataset._metadata.copy()
        result._visualization_hints = dataset._visualization_hints.copy()
        return result


# Registry of available transformers
_transformers: Dict[str, Transformer] = {}


def register_transformer(transformer: Transformer):
    """
    Register a transformer.
    
    Args:
        transformer: The transformer to register
    """
    _transformers[transformer.name] = transformer


def get_transformer(name: str) -> Transformer:
    """
    Get a registered transformer by name.
    
    Args:
        name: Name of the transformer
        
    Returns:
        Transformer: The requested transformer
        
    Raises:
        KeyError: If the transformer is not registered
    """
    if name not in _transformers:
        raise KeyError(f"Transformer '{name}' not registered")
    return _transformers[name]


# Register built-in transformers
register_transformer(SelectTransformer())
register_transformer(RenameTransformer())
register_transformer(AggregateTransformer())
register_transformer(ResampleTransformer())
register_transformer(NormalizeTransformer())
