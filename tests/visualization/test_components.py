"""
Tests for the visualization components module.
"""

import pytest
import numpy as np
import pandas as pd
from idef.visualization.components import (
    Figure, Axis, Legend, ColorMap,
    PlotComponent, ComponentRegistry
)

@pytest.fixture
def sample_data():
    """Create sample data for visualization testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.linspace(0, 10, 100),
        'y1': np.sin(np.linspace(0, 10, 100)),
        'y2': np.cos(np.linspace(0, 10, 100)),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })

@pytest.fixture
def sample_figure():
    """Create a sample figure component."""
    return Figure(
        width=800,
        height=600,
        title='Test Figure',
        theme='default'
    )

def test_figure_creation():
    """Test figure component creation."""
    figure = Figure(width=800, height=600)
    
    # Test basic properties
    assert figure.width == 800
    assert figure.height == 600
    assert figure.theme == 'default'
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        Figure(width=-100, height=600)
    
    # Test with invalid theme
    with pytest.raises(ValueError):
        Figure(width=800, height=600, theme='invalid')

def test_axis_component():
    """Test axis component functionality."""
    axis = Axis(
        title='X Axis',
        scale='linear',
        grid=True,
        position='bottom'
    )
    
    # Test properties
    assert axis.title == 'X Axis'
    assert axis.scale == 'linear'
    assert axis.grid
    assert axis.position == 'bottom'
    
    # Test scale validation
    with pytest.raises(ValueError):
        Axis(title='Invalid', scale='invalid')
    
    # Test position validation
    with pytest.raises(ValueError):
        Axis(title='Invalid', position='invalid')

def test_legend_component():
    """Test legend component functionality."""
    legend = Legend(
        title='Categories',
        position='right',
        orientation='vertical'
    )
    
    # Test properties
    assert legend.title == 'Categories'
    assert legend.position == 'right'
    assert legend.orientation == 'vertical'
    
    # Test position validation
    with pytest.raises(ValueError):
        Legend(title='Invalid', position='invalid')
    
    # Test orientation validation
    with pytest.raises(ValueError):
        Legend(title='Invalid', orientation='invalid')

def test_color_map():
    """Test color map functionality."""
    color_map = ColorMap(
        name='custom',
        colors=['#FF0000', '#00FF00', '#0000FF'],
        domain=['A', 'B', 'C']
    )
    
    # Test color mapping
    assert color_map.get_color('A') == '#FF0000'
    assert color_map.get_color('B') == '#00FF00'
    assert color_map.get_color('C') == '#0000FF'
    
    # Test continuous mapping
    continuous_map = ColorMap(
        name='continuous',
        colors=['#000000', '#FFFFFF'],
        continuous=True
    )
    assert continuous_map.get_color(0.5) is not None

def test_plot_component(sample_data):
    """Test plot component functionality."""
    class TestPlot(PlotComponent):
        def render(self, data):
            return {'type': 'scatter', 'x': data['x'], 'y': data['y1']}
    
    plot = TestPlot(
        name='test_plot',
        data_keys=['x', 'y1']
    )
    
    # Test rendering
    result = plot.render(sample_data)
    assert result['type'] == 'scatter'
    assert 'x' in result
    assert 'y' in result
    
    # Test data validation
    with pytest.raises(ValueError):
        plot.render(sample_data[['x']])  # Missing y1

def test_component_registry():
    """Test component registry functionality."""
    registry = ComponentRegistry()
    
    # Create test components
    class TestComponent1(PlotComponent):
        def render(self, data): pass
    
    class TestComponent2(PlotComponent):
        def render(self, data): pass
    
    # Register components
    registry.register('test1', TestComponent1)
    registry.register('test2', TestComponent2)
    
    # Test retrieval
    assert registry.get('test1') == TestComponent1
    assert registry.get('test2') == TestComponent2
    
    # Test listing
    assert set(registry.list_components()) == {'test1', 'test2'}
    
    # Test duplicate registration
    with pytest.raises(ValueError):
        registry.register('test1', TestComponent1)

def test_component_composition(sample_figure, sample_data):
    """Test component composition."""
    # Add components to figure
    sample_figure.add_axis(
        Axis(title='X Axis', position='bottom')
    )
    sample_figure.add_axis(
        Axis(title='Y Axis', position='left')
    )
    sample_figure.add_legend(
        Legend(title='Series')
    )
    
    # Test component access
    assert len(sample_figure.axes) == 2
    assert sample_figure.legend is not None
    
    # Test rendering
    result = sample_figure.render(sample_data)
    assert 'layout' in result
    assert 'xaxis' in result['layout']
    assert 'yaxis' in result['layout']
    assert 'showlegend' in result['layout']

def test_component_styling():
    """Test component styling functionality."""
    figure = Figure(
        width=800,
        height=600,
        theme='default',
        styles={
            'font_family': 'Arial',
            'background_color': '#FFFFFF',
            'grid_color': '#EEEEEE'
        }
    )
    
    # Test style application
    styles = figure.get_styles()
    assert styles['font_family'] == 'Arial'
    assert styles['background_color'] == '#FFFFFF'
    
    # Test style inheritance
    axis = Axis(
        title='Test',
        styles={'grid_color': '#CCCCCC'}
    )
    figure.add_axis(axis)
    
    axis_styles = axis.get_styles()
    assert axis_styles['font_family'] == 'Arial'  # Inherited
    assert axis_styles['grid_color'] == '#CCCCCC'  # Overridden

def test_component_events():
    """Test component event handling."""
    figure = Figure(width=800, height=600)
    
    # Test event registration
    events = []
    figure.on('click', lambda e: events.append(e))
    figure.on('hover', lambda e: events.append(e))
    
    # Simulate events
    figure.trigger_event('click', {'x': 100, 'y': 100})
    figure.trigger_event('hover', {'x': 200, 'y': 200})
    
    assert len(events) == 2
    assert events[0]['type'] == 'click'
    assert events[1]['type'] == 'hover'

def test_component_serialization(sample_figure, tmp_path):
    """Test component serialization."""
    # Add components
    sample_figure.add_axis(Axis(title='X Axis'))
    sample_figure.add_legend(Legend(title='Legend'))
    
    # Save figure
    save_path = tmp_path / 'figure.json'
    sample_figure.save(save_path)
    
    # Load figure
    loaded_figure = Figure.load(save_path)
    assert loaded_figure.width == sample_figure.width
    assert loaded_figure.height == sample_figure.height
    assert len(loaded_figure.axes) == len(sample_figure.axes)
    assert loaded_figure.legend is not None 