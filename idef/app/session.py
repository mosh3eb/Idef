"""
Application session module for the Interactive Data Exploration Framework.
Provides the main Explorer class for interactive data exploration.
"""

from typing import Dict, List, Union, Optional, Any
import os
import uuid

from ..data.model import Dataset
from ..data.connectors import load_dataset
from ..visualization.components import visualize, Visualization


class Explorer:
    """
    Main class for interactive data exploration.
    Provides a high-level interface for loading data, creating visualizations,
    and managing exploration sessions.
    """
    
    def __init__(self):
        """Initialize an Explorer instance."""
        self.datasets: Dict[str, Dataset] = {}
        self.visualizations: Dict[str, Visualization] = {}
        self.session_id = str(uuid.uuid4())
        self.dashboard = None  # Will be initialized when needed
    
    def load_dataset(self, source: Any, format: Optional[str] = None, 
                    name: Optional[str] = None, **kwargs) -> Dataset:
        """
        Load a dataset from a source.
        
        Args:
            source: Data source (file path, DataFrame, xarray Dataset, or numpy array)
            format: Optional format specification for file sources
            name: Optional name for the dataset
            **kwargs: Additional arguments passed to the underlying loader
            
        Returns:
            Dataset: The loaded dataset
        """
        dataset = load_dataset(source, format=format, name=name, **kwargs)
        
        # Generate a unique ID if name is not provided
        if name is None:
            if isinstance(source, str) and os.path.exists(source):
                name = os.path.splitext(os.path.basename(source))[0]
            else:
                name = f"Dataset_{len(self.datasets) + 1}"
        
        # Ensure unique name
        base_name = name
        counter = 1
        while name in self.datasets:
            name = f"{base_name}_{counter}"
            counter += 1
        
        # Store dataset
        self.datasets[name] = dataset
        return dataset
    
    def get_dataset(self, name: str) -> Dataset:
        """
        Get a dataset by name.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Dataset: The requested dataset
            
        Raises:
            KeyError: If the dataset does not exist
        """
        if name not in self.datasets:
            raise KeyError(f"Dataset '{name}' not found")
        return self.datasets[name]
    
    def list_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
            List[str]: Names of available datasets
        """
        return list(self.datasets.keys())
    
    def visualize(self, dataset: Union[Dataset, str], type: str, **kwargs) -> Visualization:
        """
        Create a visualization for a dataset.
        
        Args:
            dataset: Dataset or name of a dataset
            type: Type of visualization to create
            **kwargs: Additional arguments for the visualization
            
        Returns:
            Visualization: The created visualization
        """
        # Get dataset if name is provided
        if isinstance(dataset, str):
            dataset = self.get_dataset(dataset)
        
        # Create visualization
        viz = visualize(dataset, type, **kwargs)
        
        # Generate a unique ID
        name = kwargs.get('name')
        if name is None:
            name = f"{type.capitalize()}_{len(self.visualizations) + 1}"
        
        # Ensure unique name
        base_name = name
        counter = 1
        while name in self.visualizations:
            name = f"{base_name}_{counter}"
            counter += 1
        
        # Store visualization
        self.visualizations[name] = viz
        return viz
    
    def get_visualization(self, name: str) -> Visualization:
        """
        Get a visualization by name.
        
        Args:
            name: Name of the visualization
            
        Returns:
            Visualization: The requested visualization
            
        Raises:
            KeyError: If the visualization does not exist
        """
        if name not in self.visualizations:
            raise KeyError(f"Visualization '{name}' not found")
        return self.visualizations[name]
    
    def list_visualizations(self) -> List[str]:
        """
        List all available visualizations.
        
        Returns:
            List[str]: Names of available visualizations
        """
        return list(self.visualizations.keys())
    
    def suggest_visualizations(self, dataset: Union[Dataset, str]) -> List[Dict[str, Any]]:
        """
        Suggest appropriate visualizations for a dataset.
        
        Args:
            dataset: Dataset or name of a dataset
            
        Returns:
            List[Dict[str, Any]]: List of visualization suggestions
        """
        # Get dataset if name is provided
        if isinstance(dataset, str):
            dataset = self.get_dataset(dataset)
        
        # Get suggestions from visualization factory
        from ..visualization.components import VisualizationFactory
        return VisualizationFactory.suggest_visualizations(dataset)
    
    def create_dashboard(self):
        """
        Create a dashboard for interactive exploration.
        
        Returns:
            Dashboard: The created dashboard
        """
        # Import here to avoid circular imports
        from ..ui.dashboard import Dashboard
        self.dashboard = Dashboard(self)
        return self.dashboard
    
    def export(self, filename: str, visualization: Optional[Union[Visualization, str]] = None):
        """
        Export a visualization or dashboard to a file.
        
        Args:
            filename: Output filename
            visualization: Optional visualization or name of a visualization to export
                          If None, export the dashboard
        """
        # Get visualization if name is provided
        if isinstance(visualization, str):
            visualization = self.get_visualization(visualization)
        
        # Determine file extension
        _, ext = os.path.splitext(filename)
        if not ext:
            ext = '.html'
            filename = f"{filename}{ext}"
        
        # Export based on type
        if visualization is None:
            # Export dashboard
            if self.dashboard is None:
                self.create_dashboard()
            self.dashboard.export(filename)
        else:
            # Export single visualization
            if ext.lower() == '.html':
                # Export as HTML
                import plotly.io as pio
                fig = visualization.render()
                pio.write_html(fig, file=filename)
            elif ext.lower() in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']:
                # Export as image
                import plotly.io as pio
                fig = visualization.render()
                pio.write_image(fig, file=filename)
            else:
                raise ValueError(f"Unsupported export format: {ext}")
    
    def save_session(self, filename: str):
        """
        Save the current session to a file.
        
        Args:
            filename: Output filename
        """
        # TODO: Implement session serialization
        raise NotImplementedError("Session saving not yet implemented")
    
    def load_session(self, filename: str):
        """
        Load a session from a file.
        
        Args:
            filename: Input filename
        """
        # TODO: Implement session deserialization
        raise NotImplementedError("Session loading not yet implemented")
