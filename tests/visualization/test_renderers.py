"""
Tests for the visualization renderers module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from idef.visualization.renderers import (
    Renderer, PlotlyRenderer, MatplotlibRenderer,
    BokehRenderer, RendererRegistry
)

@pytest.fixture
def sample_data():
    """Create sample data for renderer testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.linspace(0, 10, 100),
        'y1': np.sin(np.linspace(0, 10, 100)),
        'y2': np.cos(np.linspace(0, 10, 100)),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })

@pytest.fixture
def plot_spec():
    """Create a sample plot specification."""
    return {
        'type': 'scatter',
        'data': {
            'x': 'x',
            'y': 'y1',
            'color': 'category'
        },
        'layout': {
            'title': 'Test Plot',
            'width': 800,
            'height': 600
        }
    }

def test_renderer_base_class():
    """Test base renderer functionality."""
    class TestRenderer(Renderer):
        def render(self, data, spec):
            return {'rendered': True}
    
    renderer = TestRenderer()
    assert renderer.name == 'test'
    assert renderer.supported_types == set()
    
    # Test abstract base class
    with pytest.raises(TypeError):
        Renderer()

def test_plotly_renderer(sample_data, plot_spec):
    """Test Plotly renderer functionality."""
    renderer = PlotlyRenderer()
    
    # Test basic rendering
    result = renderer.render(sample_data, plot_spec)
    assert 'data' in result
    assert 'layout' in result
    
    # Test data transformation
    trace = result['data'][0]
    assert len(trace['x']) == len(sample_data)
    assert len(trace['y']) == len(sample_data)
    
    # Test layout application
    layout = result['layout']
    assert layout['width'] == 800
    assert layout['height'] == 600
    assert layout['title']['text'] == 'Test Plot'

def test_matplotlib_renderer(sample_data, plot_spec):
    """Test Matplotlib renderer functionality."""
    renderer = MatplotlibRenderer()
    
    # Test basic rendering
    with patch('matplotlib.pyplot.Figure') as mock_figure:
        result = renderer.render(sample_data, plot_spec)
        assert mock_figure.called
    
    # Test figure properties
    assert hasattr(result, 'savefig')
    
    # Test with different plot types
    bar_spec = {**plot_spec, 'type': 'bar'}
    with patch('matplotlib.pyplot.Figure'):
        result = renderer.render(sample_data, bar_spec)
        assert result is not None

def test_bokeh_renderer(sample_data, plot_spec):
    """Test Bokeh renderer functionality."""
    renderer = BokehRenderer()
    
    # Test basic rendering
    result = renderer.render(sample_data, plot_spec)
    assert hasattr(result, 'to_html')
    
    # Test with different plot types
    line_spec = {**plot_spec, 'type': 'line'}
    result = renderer.render(sample_data, line_spec)
    assert result is not None

def test_renderer_registry():
    """Test renderer registry functionality."""
    registry = RendererRegistry()
    
    # Register renderers
    registry.register('plotly', PlotlyRenderer())
    registry.register('matplotlib', MatplotlibRenderer())
    
    # Test retrieval
    plotly = registry.get('plotly')
    assert isinstance(plotly, PlotlyRenderer)
    
    matplotlib = registry.get('matplotlib')
    assert isinstance(matplotlib, MatplotlibRenderer)
    
    # Test default renderer
    registry.set_default('plotly')
    default = registry.get_default()
    assert isinstance(default, PlotlyRenderer)
    
    # Test invalid renderer
    with pytest.raises(KeyError):
        registry.get('invalid')

def test_renderer_plot_types():
    """Test supported plot types for each renderer."""
    plotly = PlotlyRenderer()
    matplotlib = MatplotlibRenderer()
    bokeh = BokehRenderer()
    
    # Test Plotly supported types
    assert 'scatter' in plotly.supported_types
    assert 'bar' in plotly.supported_types
    assert 'line' in plotly.supported_types
    
    # Test Matplotlib supported types
    assert 'scatter' in matplotlib.supported_types
    assert 'bar' in matplotlib.supported_types
    assert 'line' in matplotlib.supported_types
    
    # Test Bokeh supported types
    assert 'scatter' in bokeh.supported_types
    assert 'bar' in bokeh.supported_types
    assert 'line' in bokeh.supported_types

def test_renderer_styling(sample_data):
    """Test renderer styling capabilities."""
    renderer = PlotlyRenderer()
    
    # Test with custom styles
    spec = {
        'type': 'scatter',
        'data': {
            'x': 'x',
            'y': 'y1'
        },
        'style': {
            'marker': {'size': 10, 'color': 'red'},
            'line': {'width': 2, 'dash': 'dot'}
        }
    }
    
    result = renderer.render(sample_data, spec)
    trace = result['data'][0]
    assert trace['marker']['size'] == 10
    assert trace['marker']['color'] == 'red'
    assert trace['line']['width'] == 2
    assert trace['line']['dash'] == 'dot'

def test_renderer_animations(sample_data):
    """Test renderer animation capabilities."""
    renderer = PlotlyRenderer()
    
    # Test with animation frames
    spec = {
        'type': 'scatter',
        'data': {
            'x': 'x',
            'y': 'y1'
        },
        'animation': {
            'frame_data': [
                {'y': 'y1'},
                {'y': 'y2'}
            ],
            'transition': {'duration': 500}
        }
    }
    
    result = renderer.render(sample_data, spec)
    assert 'frames' in result
    assert len(result['frames']) == 2
    assert 'transition' in result['layout']

def test_renderer_events(sample_data, plot_spec):
    """Test renderer event handling."""
    renderer = PlotlyRenderer()
    
    # Add event callbacks
    events = []
    spec = {
        **plot_spec,
        'events': {
            'click': lambda e: events.append(e),
            'hover': lambda e: events.append(e)
        }
    }
    
    result = renderer.render(sample_data, spec)
    assert 'config' in result
    assert 'events' in result['config']

def test_renderer_export(sample_data, plot_spec, tmp_path):
    """Test renderer export capabilities."""
    renderer = PlotlyRenderer()
    
    # Test HTML export
    result = renderer.render(sample_data, plot_spec)
    html_path = tmp_path / 'plot.html'
    renderer.export(result, html_path, format='html')
    assert html_path.exists()
    
    # Test image export
    png_path = tmp_path / 'plot.png'
    renderer.export(result, png_path, format='png')
    assert png_path.exists()
    
    # Test invalid format
    with pytest.raises(ValueError):
        renderer.export(result, tmp_path / 'invalid', format='invalid') 