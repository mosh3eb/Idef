"""
Python API module for the Interactive Data Exploration Framework.
Provides a programmatic interface for data exploration.
"""

from typing import Dict, List, Union, Optional, Any
import pandas as pd
import xarray as xr
import numpy as np
import plotly.graph_objects as go
import holoviews as hv

from ..app.session import Explorer
from ..data.model import Dataset
from ..visualization.components import Visualization


class IDEF:
    """
    High-level API for the Interactive Data Exploration Framework.
    Provides a simplified interface for common data exploration tasks.
    """
    
    def __init__(self):
        """Initialize the IDEF API."""
        self._explorer = Explorer()
    
    def load_data(self, source: Any, **kwargs) -> Dataset:
        """
        Load data from a source.
        
        Args:
            source: Data source (file path, DataFrame, xarray Dataset, or numpy array)
            **kwargs: Additional arguments for data loading
            
        Returns:
            Dataset: The loaded dataset
        """
        return self._explorer.load_dataset(source, **kwargs)
    
    def visualize(self, data: Union[Dataset, str, pd.DataFrame, xr.Dataset, np.ndarray], 
                 type: str, **kwargs) -> Visualization:
        """
        Create a visualization.
        
        Args:
            data: Dataset, dataset name, or raw data
            type: Visualization type ('scatter', 'line', 'heatmap', 'scatter3d', 'parallel')
            **kwargs: Additional arguments for the visualization
            
        Returns:
            Visualization: The created visualization
        """
        # If data is not a Dataset or dataset name, load it first
        if not isinstance(data, (Dataset, str)) or (
            isinstance(data, str) and data not in self._explorer.list_datasets()
        ):
            data = self._explorer.load_dataset(data)
        
        return self._explorer.visualize(data, type, **kwargs)
    
    def show(self, visualization: Union[Visualization, str]) -> Any:
        """
        Show a visualization.
        
        Args:
            visualization: Visualization or name of a visualization
            
        Returns:
            The rendered visualization
        """
        # Get visualization if name is provided
        if isinstance(visualization, str):
            visualization = self._explorer.get_visualization(visualization)
        
        return visualization.render()
    
    def dashboard(self) -> Any:
        """
        Create and show an interactive dashboard.
        
        Returns:
            The dashboard object
        """
        dashboard = self._explorer.create_dashboard()
        return dashboard.show()
    
    def suggest_visualizations(self, data: Union[Dataset, str, pd.DataFrame, xr.Dataset, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Suggest appropriate visualizations for data.
        
        Args:
            data: Dataset, dataset name, or raw data
            
        Returns:
            List[Dict[str, Any]]: List of visualization suggestions
        """
        # If data is not a Dataset or dataset name, load it first
        if not isinstance(data, (Dataset, str)) or (
            isinstance(data, str) and data not in self._explorer.list_datasets()
        ):
            data = self._explorer.load_dataset(data)
        
        return self._explorer.suggest_visualizations(data)
    
    def export(self, visualization: Union[Visualization, str], filename: str):
        """
        Export a visualization to a file.
        
        Args:
            visualization: Visualization or name of a visualization
            filename: Output filename
        """
        self._explorer.export(filename, visualization)
    
    @property
    def explorer(self) -> Explorer:
        """
        Get the underlying Explorer instance.
        
        Returns:
            Explorer: The Explorer instance
        """
        return self._explorer


# Create a default instance for easy import
idef = IDEF()


# Define convenience functions that use the default instance

def load_data(source: Any, **kwargs) -> Dataset:
    """
    Load data from a source.
    
    Args:
        source: Data source (file path, DataFrame, xarray Dataset, or numpy array)
        **kwargs: Additional arguments for data loading
        
    Returns:
        Dataset: The loaded dataset
    """
    return idef.load_data(source, **kwargs)


def visualize(data: Union[Dataset, str, pd.DataFrame, xr.Dataset, np.ndarray], 
             type: str, **kwargs) -> Visualization:
    """
    Create a visualization.
    
    Args:
        data: Dataset, dataset name, or raw data
        type: Visualization type ('scatter', 'line', 'heatmap', 'scatter3d', 'parallel')
        **kwargs: Additional arguments for the visualization
        
    Returns:
        Visualization: The created visualization
    """
    return idef.visualize(data, type, **kwargs)


def show(visualization: Union[Visualization, str]) -> Any:
    """
    Show a visualization.
    
    Args:
        visualization: Visualization or name of a visualization
        
    Returns:
        The rendered visualization
    """
    return idef.show(visualization)


def dashboard() -> Any:
    """
    Create and show an interactive dashboard.
    
    Returns:
        The dashboard object
    """
    return idef.dashboard()


def suggest_visualizations(data: Union[Dataset, str, pd.DataFrame, xr.Dataset, np.ndarray]) -> List[Dict[str, Any]]:
    """
    Suggest appropriate visualizations for data.
    
    Args:
        data: Dataset, dataset name, or raw data
        
    Returns:
        List[Dict[str, Any]]: List of visualization suggestions
    """
    return idef.suggest_visualizations(data)


def export(visualization: Union[Visualization, str], filename: str):
    """
    Export a visualization to a file.
    
    Args:
        visualization: Visualization or name of a visualization
        filename: Output filename
    """
    idef.export(visualization, filename)
