"""
Rendering Module for IDEF.

This module provides rendering capabilities for different visualization backends.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

@dataclass
class RenderingContext:
    """Context for rendering operations."""
    width: int = 800
    height: int = 600
    theme: str = 'default'
    colormap: str = 'viridis'
    title: str = None
    xlabel: str = None
    ylabel: str = None
    interactive: bool = True
    backend: str = 'plotly'
    additional_params: Dict = None

class Renderer(ABC):
    """Abstract base class for renderers."""
    
    def __init__(self, context: RenderingContext = None):
        self.context = context or RenderingContext()
        
    @abstractmethod
    def render(self, data: Any, viz_type: str, **kwargs) -> Any:
        """Render visualization."""
        pass
        
    @abstractmethod
    def update(self, figure: Any, data: Any, **kwargs) -> Any:
        """Update existing visualization."""
        pass
        
    @abstractmethod
    def save(self, figure: Any, filename: str, **kwargs) -> bool:
        """Save visualization to file."""
        pass

class PlotlyRenderer(Renderer):
    """Renderer implementation using Plotly."""
    
    def render(self, data: Any, viz_type: str, **kwargs) -> go.Figure:
        if viz_type == 'scatter':
            return self._render_scatter(data, **kwargs)
        elif viz_type == 'line':
            return self._render_line(data, **kwargs)
        elif viz_type == 'histogram':
            return self._render_histogram(data, **kwargs)
        elif viz_type == 'heatmap':
            return self._render_heatmap(data, **kwargs)
        elif viz_type == 'box':
            return self._render_box(data, **kwargs)
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
            
    def update(self, figure: go.Figure, data: Any, **kwargs) -> go.Figure:
        # Update data
        if 'data' in kwargs:
            for i, new_data in enumerate(kwargs['data']):
                figure.data[i].update(new_data)
                
        # Update layout
        if 'layout' in kwargs:
            figure.update_layout(**kwargs['layout'])
            
        return figure
        
    def save(self, figure: go.Figure, filename: str, **kwargs) -> bool:
        try:
            figure.write_html(filename)
            return True
        except Exception as e:
            print(f"Error saving figure: {e}")
            return False
            
    def _render_scatter(self, data: np.ndarray, **kwargs) -> go.Figure:
        """Render scatter plot."""
        x = kwargs.get('x', data[:, 0])
        y = kwargs.get('y', data[:, 1])
        
        fig = go.Figure(data=go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                color=kwargs.get('color'),
                size=kwargs.get('size', 8),
                colorscale=self.context.colormap
            ),
            text=kwargs.get('labels')
        ))
        
        self._update_layout(fig)
        return fig
        
    def _render_line(self, data: np.ndarray, **kwargs) -> go.Figure:
        """Render line plot."""
        x = kwargs.get('x', np.arange(len(data)))
        y = kwargs.get('y', data)
        
        fig = go.Figure(data=go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(
                color=kwargs.get('color'),
                width=kwargs.get('width', 2)
            )
        ))
        
        self._update_layout(fig)
        return fig
        
    def _render_histogram(self, data: np.ndarray, **kwargs) -> go.Figure:
        """Render histogram."""
        fig = go.Figure(data=go.Histogram(
            x=data,
            nbinsx=kwargs.get('bins', 30),
            histnorm=kwargs.get('normalize', None)
        ))
        
        self._update_layout(fig)
        return fig
        
    def _render_heatmap(self, data: np.ndarray, **kwargs) -> go.Figure:
        """Render heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=data,
            colorscale=self.context.colormap,
            zmin=kwargs.get('vmin'),
            zmax=kwargs.get('vmax')
        ))
        
        self._update_layout(fig)
        return fig
        
    def _render_box(self, data: np.ndarray, **kwargs) -> go.Figure:
        """Render box plot."""
        fig = go.Figure(data=go.Box(
            y=data,
            name=kwargs.get('name', ''),
            boxpoints=kwargs.get('points', 'outliers')
        ))
        
        self._update_layout(fig)
        return fig
        
    def _update_layout(self, fig: go.Figure):
        """Update figure layout based on context."""
        fig.update_layout(
            width=self.context.width,
            height=self.context.height,
            title=self.context.title,
            xaxis_title=self.context.xlabel,
            yaxis_title=self.context.ylabel,
            template=self.context.theme
        )

class MatplotlibRenderer(Renderer):
    """Renderer implementation using Matplotlib."""
    
    def render(self, data: Any, viz_type: str, **kwargs) -> Figure:
        fig, ax = plt.subplots(figsize=(
            self.context.width/100,
            self.context.height/100
        ))
        
        if viz_type == 'scatter':
            self._render_scatter(ax, data, **kwargs)
        elif viz_type == 'line':
            self._render_line(ax, data, **kwargs)
        elif viz_type == 'histogram':
            self._render_histogram(ax, data, **kwargs)
        elif viz_type == 'heatmap':
            self._render_heatmap(ax, data, **kwargs)
        elif viz_type == 'box':
            self._render_box(ax, data, **kwargs)
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
            
        self._update_layout(fig, ax)
        return fig
        
    def update(self, figure: Figure, data: Any, **kwargs) -> Figure:
        # Clear the figure and redraw
        figure.clear()
        ax = figure.add_subplot(111)
        
        viz_type = kwargs.pop('viz_type', 'scatter')
        if viz_type == 'scatter':
            self._render_scatter(ax, data, **kwargs)
        elif viz_type == 'line':
            self._render_line(ax, data, **kwargs)
        # ... implement other visualization types
        
        self._update_layout(figure, ax)
        return figure
        
    def save(self, figure: Figure, filename: str, **kwargs) -> bool:
        try:
            figure.savefig(filename, **kwargs)
            return True
        except Exception as e:
            print(f"Error saving figure: {e}")
            return False
            
    def _render_scatter(self, ax: plt.Axes, data: np.ndarray, **kwargs):
        """Render scatter plot."""
        x = kwargs.get('x', data[:, 0])
        y = kwargs.get('y', data[:, 1])
        
        scatter = ax.scatter(
            x, y,
            c=kwargs.get('color'),
            s=kwargs.get('size', 50),
            cmap=self.context.colormap,
            alpha=kwargs.get('alpha', 0.6)
        )
        
        if 'labels' in kwargs:
            for i, label in enumerate(kwargs['labels']):
                ax.annotate(label, (x[i], y[i]))
                
    def _render_line(self, ax: plt.Axes, data: np.ndarray, **kwargs):
        """Render line plot."""
        x = kwargs.get('x', np.arange(len(data)))
        y = kwargs.get('y', data)
        
        ax.plot(
            x, y,
            color=kwargs.get('color'),
            linewidth=kwargs.get('width', 2),
            linestyle=kwargs.get('style', '-')
        )
        
    def _render_histogram(self, ax: plt.Axes, data: np.ndarray, **kwargs):
        """Render histogram."""
        ax.hist(
            data,
            bins=kwargs.get('bins', 30),
            density=kwargs.get('normalize', False),
            alpha=kwargs.get('alpha', 0.7),
            color=kwargs.get('color')
        )
        
    def _render_heatmap(self, ax: plt.Axes, data: np.ndarray, **kwargs):
        """Render heatmap."""
        sns.heatmap(
            data,
            ax=ax,
            cmap=self.context.colormap,
            vmin=kwargs.get('vmin'),
            vmax=kwargs.get('vmax'),
            cbar=kwargs.get('colorbar', True)
        )
        
    def _render_box(self, ax: plt.Axes, data: np.ndarray, **kwargs):
        """Render box plot."""
        ax.boxplot(
            data,
            labels=[kwargs.get('name', '')],
            showfliers=kwargs.get('show_outliers', True)
        )
        
    def _update_layout(self, fig: Figure, ax: plt.Axes):
        """Update figure layout based on context."""
        if self.context.title:
            ax.set_title(self.context.title)
        if self.context.xlabel:
            ax.set_xlabel(self.context.xlabel)
        if self.context.ylabel:
            ax.set_ylabel(self.context.ylabel)
            
        plt.style.use(self.context.theme)
        fig.tight_layout()

class BokehRenderer(Renderer):
    """Renderer for Bokeh visualizations."""
    
    def render(self, data: pd.DataFrame, spec: Dict) -> Any:
        """Render Bokeh visualization."""
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource
        
        plot_type = spec['type']
        if plot_type not in self.supported_types:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        # Create data source
        source = ColumnDataSource(data)
        
        # Create figure
        layout = spec.get('layout', {})
        p = figure(
            width=layout.get('width', 800),
            height=layout.get('height', 600),
            title=layout.get('title', '')
        )
        
        # Extract data mappings
        x_field = spec['data']['x']
        y_field = spec['data']['y']
        color_field = spec['data'].get('color')
        
        # Create plot
        plot_args = {
            'x': x_field,
            'y': y_field,
            'source': source
        }
        
        if color_field:
            plot_args['color'] = color_field
        
        # Apply custom styling
        if 'style' in spec:
            if 'marker' in spec['style']:
                plot_args.update(spec['style']['marker'])
            if 'line' in spec['style']:
                plot_args.update(spec['style']['line'])
        
        if plot_type == 'scatter':
            p.scatter(**plot_args)
        elif plot_type == 'line':
            p.line(**plot_args)
        else:  # bar
            p.vbar(
                x=x_field,
                top=y_field,
                width=0.8,
                source=source,
                **plot_args
            )
        
        return p

# Factory for creating renderers
class RendererFactory:
    """Factory for creating renderer instances."""
    
    _renderers = {
        'plotly': PlotlyRenderer,
        'matplotlib': MatplotlibRenderer,
        'bokeh': BokehRenderer
    }
    
    @classmethod
    def create_renderer(cls, backend: str = 'plotly',
                       context: RenderingContext = None) -> Renderer:
        """Create a renderer instance."""
        if backend not in cls._renderers:
            raise ValueError(f"Unknown rendering backend: {backend}")
        return cls._renderers[backend](context)
    
    @classmethod
    def available_backends(cls) -> List[str]:
        """List available rendering backends."""
        return list(cls._renderers.keys())
