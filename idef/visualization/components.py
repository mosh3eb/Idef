"""
Visualization components module for the Interactive Data Exploration Framework.
Provides visualization types and factories for multi-dimensional data.
"""

from typing import Dict, List, Union, Optional, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
import holoviews as hv
import pandas as pd
import numpy as np

from ..data.model import Dataset

# Initialize HoloViews extension
hv.extension('plotly')


class Visualization:
    """Base class for all visualizations."""
    
    def __init__(self, dataset: Dataset, name: Optional[str] = None):
        """
        Initialize a visualization.
        
        Args:
            dataset: The dataset to visualize
            name: Optional name for the visualization
        """
        self.dataset = dataset
        self.name = name or f"{dataset.name} Visualization"
        self.options = {}
    
    def render(self) -> Any:
        """
        Render the visualization.
        
        Returns:
            The rendered visualization object
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement render()")
    
    def set_option(self, key: str, value: Any) -> 'Visualization':
        """
        Set a visualization option.
        
        Args:
            key: Option key
            value: Option value
            
        Returns:
            Visualization: The visualization instance (for method chaining)
        """
        self.options[key] = value
        return self
    
    def get_option(self, key: str, default: Any = None) -> Any:
        """
        Get a visualization option.
        
        Args:
            key: Option key
            default: Default value if key does not exist
            
        Returns:
            The option value or default
        """
        return self.options.get(key, default)
    
    @property
    def supported_dimensions(self) -> int:
        """
        Get the number of dimensions supported by this visualization.
        
        Returns:
            int: The number of supported dimensions
        """
        raise NotImplementedError("Subclasses must implement supported_dimensions")


class ScatterPlot(Visualization):
    """2D scatter plot visualization."""
    
    def __init__(self, dataset: Dataset, x: str, y: str, name: Optional[str] = None):
        """
        Initialize a scatter plot.
        
        Args:
            dataset: The dataset to visualize
            x: Variable for x-axis
            y: Variable for y-axis
            name: Optional name for the visualization
        """
        super().__init__(dataset, name)
        self.x = x
        self.y = y
        self.color = None
        self.size = None
        self.hover_data = []
    
    def set_color(self, variable: str) -> 'ScatterPlot':
        """
        Set the color variable.
        
        Args:
            variable: Variable for color mapping
            
        Returns:
            ScatterPlot: The scatter plot instance
        """
        self.color = variable
        return self
    
    def set_size(self, variable: str) -> 'ScatterPlot':
        """
        Set the size variable.
        
        Args:
            variable: Variable for size mapping
            
        Returns:
            ScatterPlot: The scatter plot instance
        """
        self.size = variable
        return self
    
    def add_hover_data(self, variables: List[str]) -> 'ScatterPlot':
        """
        Add variables to hover data.
        
        Args:
            variables: Variables to include in hover data
            
        Returns:
            ScatterPlot: The scatter plot instance
        """
        self.hover_data.extend(variables)
        return self
    
    def render(self) -> go.Figure:
        """
        Render the scatter plot.
        
        Returns:
            plotly.graph_objects.Figure: The rendered scatter plot
        """
        # Convert dataset to pandas DataFrame for plotting
        df = self.dataset.data.to_dataframe().reset_index()
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x=self.x,
            y=self.y,
            color=self.color,
            size=self.size,
            hover_data=self.hover_data,
            title=self.name,
            labels={
                self.x: self.get_option('x_label', self.x),
                self.y: self.get_option('y_label', self.y)
            }
        )
        
        # Apply additional options
        if 'width' in self.options and 'height' in self.options:
            fig.update_layout(
                width=self.options['width'],
                height=self.options['height']
            )
        
        if 'colorscale' in self.options and self.color is not None:
            fig.update_traces(marker=dict(colorscale=self.options['colorscale']))
        
        return fig
    
    @property
    def supported_dimensions(self) -> int:
        """Get the number of dimensions supported by this visualization."""
        return 2


class LinePlot(Visualization):
    """Line plot visualization."""
    
    def __init__(self, dataset: Dataset, x: str, y: Union[str, List[str]], name: Optional[str] = None):
        """
        Initialize a line plot.
        
        Args:
            dataset: The dataset to visualize
            x: Variable for x-axis
            y: Variable(s) for y-axis
            name: Optional name for the visualization
        """
        super().__init__(dataset, name)
        self.x = x
        self.y = y if isinstance(y, list) else [y]
        self.color = None
        self.line_dash = None
    
    def set_color(self, variable: str) -> 'LinePlot':
        """
        Set the color variable.
        
        Args:
            variable: Variable for color mapping
            
        Returns:
            LinePlot: The line plot instance
        """
        self.color = variable
        return self
    
    def set_line_dash(self, variable: str) -> 'LinePlot':
        """
        Set the line dash variable.
        
        Args:
            variable: Variable for line dash mapping
            
        Returns:
            LinePlot: The line plot instance
        """
        self.line_dash = variable
        return self
    
    def render(self) -> go.Figure:
        """
        Render the line plot.
        
        Returns:
            plotly.graph_objects.Figure: The rendered line plot
        """
        # Convert dataset to pandas DataFrame for plotting
        df = self.dataset.data.to_dataframe().reset_index()
        
        # Create line plot
        fig = px.line(
            df,
            x=self.x,
            y=self.y,
            color=self.color,
            line_dash=self.line_dash,
            title=self.name,
            labels={
                self.x: self.get_option('x_label', self.x),
                **{y_var: self.get_option(f'y_label_{y_var}', y_var) for y_var in self.y}
            }
        )
        
        # Apply additional options
        if 'width' in self.options and 'height' in self.options:
            fig.update_layout(
                width=self.options['width'],
                height=self.options['height']
            )
        
        return fig
    
    @property
    def supported_dimensions(self) -> int:
        """Get the number of dimensions supported by this visualization."""
        return 2


class Heatmap(Visualization):
    """Heatmap visualization."""
    
    def __init__(self, dataset: Dataset, x: str, y: str, z: str, name: Optional[str] = None):
        """
        Initialize a heatmap.
        
        Args:
            dataset: The dataset to visualize
            x: Variable for x-axis
            y: Variable for y-axis
            z: Variable for color mapping
            name: Optional name for the visualization
        """
        super().__init__(dataset, name)
        self.x = x
        self.y = y
        self.z = z
    
    def render(self) -> go.Figure:
        """
        Render the heatmap.
        
        Returns:
            plotly.graph_objects.Figure: The rendered heatmap
        """
        # Extract data for heatmap
        if self.x in self.dataset.dimensions and self.y in self.dataset.dimensions:
            # If x and y are dimensions, we can use the data directly
            z_data = self.dataset.get_variable(self.z).values
            x_values = self.dataset.data[self.x].values
            y_values = self.dataset.data[self.y].values
        else:
            # Otherwise, pivot the data
            df = self.dataset.data.to_dataframe().reset_index()
            pivot_table = df.pivot_table(
                values=self.z,
                index=self.y,
                columns=self.x,
                aggfunc='mean'
            )
            z_data = pivot_table.values
            x_values = pivot_table.columns.tolist()
            y_values = pivot_table.index.tolist()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=x_values,
            y=y_values,
            colorscale=self.get_option('colorscale', 'Viridis'),
            colorbar=dict(title=self.z)
        ))
        
        # Set layout
        fig.update_layout(
            title=self.name,
            xaxis_title=self.get_option('x_label', self.x),
            yaxis_title=self.get_option('y_label', self.y),
            width=self.get_option('width', None),
            height=self.get_option('height', None)
        )
        
        return fig
    
    @property
    def supported_dimensions(self) -> int:
        """Get the number of dimensions supported by this visualization."""
        return 3


class Scatter3D(Visualization):
    """3D scatter plot visualization."""
    
    def __init__(self, dataset: Dataset, x: str, y: str, z: str, name: Optional[str] = None):
        """
        Initialize a 3D scatter plot.
        
        Args:
            dataset: The dataset to visualize
            x: Variable for x-axis
            y: Variable for y-axis
            z: Variable for z-axis
            name: Optional name for the visualization
        """
        super().__init__(dataset, name)
        self.x = x
        self.y = y
        self.z = z
        self.color = None
        self.size = None
        self.hover_data = []
    
    def set_color(self, variable: str) -> 'Scatter3D':
        """
        Set the color variable.
        
        Args:
            variable: Variable for color mapping
            
        Returns:
            Scatter3D: The 3D scatter plot instance
        """
        self.color = variable
        return self
    
    def set_size(self, variable: str) -> 'Scatter3D':
        """
        Set the size variable.
        
        Args:
            variable: Variable for size mapping
            
        Returns:
            Scatter3D: The 3D scatter plot instance
        """
        self.size = variable
        return self
    
    def add_hover_data(self, variables: List[str]) -> 'Scatter3D':
        """
        Add variables to hover data.
        
        Args:
            variables: Variables to include in hover data
            
        Returns:
            Scatter3D: The 3D scatter plot instance
        """
        self.hover_data.extend(variables)
        return self
    
    def render(self) -> go.Figure:
        """
        Render the 3D scatter plot.
        
        Returns:
            plotly.graph_objects.Figure: The rendered 3D scatter plot
        """
        # Convert dataset to pandas DataFrame for plotting
        df = self.dataset.data.to_dataframe().reset_index()
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df,
            x=self.x,
            y=self.y,
            z=self.z,
            color=self.color,
            size=self.size,
            hover_data=self.hover_data,
            title=self.name,
            labels={
                self.x: self.get_option('x_label', self.x),
                self.y: self.get_option('y_label', self.y),
                self.z: self.get_option('z_label', self.z)
            }
        )
        
        # Apply additional options
        if 'width' in self.options and 'height' in self.options:
            fig.update_layout(
                width=self.options['width'],
                height=self.options['height']
            )
        
        if 'colorscale' in self.options and self.color is not None:
            fig.update_traces(marker=dict(colorscale=self.options['colorscale']))
        
        return fig
    
    @property
    def supported_dimensions(self) -> int:
        """Get the number of dimensions supported by this visualization."""
        return 3


class ParallelCoordinates(Visualization):
    """Parallel coordinates visualization for multi-dimensional data."""
    
    def __init__(self, dataset: Dataset, dimensions: List[str], name: Optional[str] = None):
        """
        Initialize a parallel coordinates plot.
        
        Args:
            dataset: The dataset to visualize
            dimensions: Variables to include as dimensions
            name: Optional name for the visualization
        """
        super().__init__(dataset, name)
        self.dimensions = dimensions
        self.color = None
    
    def set_color(self, variable: str) -> 'ParallelCoordinates':
        """
        Set the color variable.
        
        Args:
            variable: Variable for color mapping
            
        Returns:
            ParallelCoordinates: The parallel coordinates plot instance
        """
        self.color = variable
        return self
    
    def render(self) -> go.Figure:
        """
        Render the parallel coordinates plot.
        
        Returns:
            plotly.graph_objects.Figure: The rendered parallel coordinates plot
        """
        # Convert dataset to pandas DataFrame for plotting
        df = self.dataset.data.to_dataframe().reset_index()
        
        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            df,
            dimensions=self.dimensions,
            color=self.color,
            title=self.name,
            labels={dim: self.get_option(f'label_{dim}', dim) for dim in self.dimensions}
        )
        
        # Apply additional options
        if 'width' in self.options and 'height' in self.options:
            fig.update_layout(
                width=self.options['width'],
                height=self.options['height']
            )
        
        if 'colorscale' in self.options and self.color is not None:
            fig.update_layout(coloraxis_colorscale=self.options['colorscale'])
        
        return fig
    
    @property
    def supported_dimensions(self) -> int:
        """Get the number of dimensions supported by this visualization."""
        return len(self.dimensions)


class VisualizationFactory:
    """Factory for creating visualizations based on data characteristics."""
    
    @staticmethod
    def create_visualization(dataset: Dataset, 
                            viz_type: str, 
                            **kwargs) -> Visualization:
        """
        Create a visualization of the specified type.
        
        Args:
            dataset: The dataset to visualize
            viz_type: Type of visualization to create
            **kwargs: Additional arguments for the visualization
            
        Returns:
            Visualization: The created visualization
            
        Raises:
            ValueError: If the visualization type is not supported
        """
        if viz_type == 'scatter':
            if 'x' not in kwargs or 'y' not in kwargs:
                raise ValueError("Scatter plot requires 'x' and 'y' parameters")
            return ScatterPlot(dataset, kwargs['x'], kwargs['y'], kwargs.get('name'))
        elif viz_type == 'line':
            if 'x' not in kwargs or 'y' not in kwargs:
                raise ValueError("Line plot requires 'x' and 'y' parameters")
            return LinePlot(dataset, kwargs['x'], kwargs['y'], kwargs.get('name'))
        elif viz_type == 'heatmap':
            if 'x' not in kwargs or 'y' not in kwargs or 'z' not in kwargs:
                raise ValueError("Heatmap requires 'x', 'y', and 'z' parameters")
            return Heatmap(dataset, kwargs['x'], kwargs['y'], kwargs['z'], kwargs.get('name'))
        elif viz_type == 'scatter3d':
            if 'x' not in kwargs or 'y' not in kwargs or 'z' not in kwargs:
                raise ValueError("3D scatter plot requires 'x', 'y', and 'z' parameters")
            return Scatter3D(dataset, kwargs['x'], kwargs['y'], kwargs['z'], kwargs.get('name'))
        elif viz_type == 'parallel':
            if 'dimensions' not in kwargs:
                raise ValueError("Parallel coordinates plot requires 'dimensions' parameter")
            return ParallelCoordinates(dataset, kwargs['dimensions'], kwargs.get('name'))
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")