"""
Tests for the visualization interactions module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from idef.visualization.interactions import (
    InteractionHandler, Event, EventDispatcher,
    ZoomHandler, PanHandler, SelectionHandler,
    InteractionState
)

@pytest.fixture
def sample_data():
    """Create sample data for interaction testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'x': np.linspace(0, 10, 100),
        'y': np.sin(np.linspace(0, 10, 100)),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })

@pytest.fixture
def interaction_handler():
    """Create a sample interaction handler."""
    return InteractionHandler(
        supported_events=['click', 'hover', 'zoom', 'pan', 'select']
    )

def test_event_creation():
    """Test event creation and properties."""
    event = Event(
        type='click',
        x=100,
        y=200,
        data={'point_index': 42},
        modifiers={'shift': True}
    )
    
    # Test basic properties
    assert event.type == 'click'
    assert event.x == 100
    assert event.y == 200
    assert event.data['point_index'] == 42
    assert event.modifiers['shift']
    
    # Test invalid event type
    with pytest.raises(ValueError):
        Event(type='invalid', x=0, y=0)

def test_event_dispatcher():
    """Test event dispatcher functionality."""
    dispatcher = EventDispatcher()
    
    # Test event registration
    mock_handler = Mock()
    dispatcher.register('click', mock_handler)
    
    # Test event dispatch
    event = Event('click', x=100, y=200)
    dispatcher.dispatch(event)
    mock_handler.assert_called_once_with(event)
    
    # Test multiple handlers
    mock_handler2 = Mock()
    dispatcher.register('click', mock_handler2)
    dispatcher.dispatch(event)
    mock_handler2.assert_called_once_with(event)
    
    # Test handler removal
    dispatcher.unregister('click', mock_handler)
    dispatcher.dispatch(event)
    assert mock_handler.call_count == 1  # No additional calls

def test_zoom_handler(sample_data):
    """Test zoom interaction handler."""
    handler = ZoomHandler()
    
    # Test zoom in
    state = handler.handle_event(Event(
        'zoom',
        x=100,
        y=100,
        data={'scale': 2.0, 'center': (100, 100)}
    ))
    assert state.scale == 2.0
    assert state.center == (100, 100)
    
    # Test zoom out
    state = handler.handle_event(Event(
        'zoom',
        x=100,
        y=100,
        data={'scale': 0.5, 'center': (100, 100)}
    ))
    assert state.scale == 0.5
    
    # Test zoom limits
    state = handler.handle_event(Event(
        'zoom',
        x=100,
        y=100,
        data={'scale': 100.0, 'center': (100, 100)}
    ))
    assert state.scale <= handler.max_scale

def test_pan_handler(sample_data):
    """Test pan interaction handler."""
    handler = PanHandler()
    
    # Test pan movement
    state = handler.handle_event(Event(
        'pan',
        x=100,
        y=100,
        data={'dx': 50, 'dy': 30}
    ))
    assert state.offset == (50, 30)
    
    # Test cumulative pan
    state = handler.handle_event(Event(
        'pan',
        x=150,
        y=130,
        data={'dx': 25, 'dy': 15}
    ))
    assert state.offset == (75, 45)
    
    # Test pan bounds
    state = handler.handle_event(Event(
        'pan',
        x=1000,
        y=1000,
        data={'dx': 1000, 'dy': 1000}
    ))
    assert max(state.offset) <= handler.max_offset

def test_selection_handler(sample_data):
    """Test selection interaction handler."""
    handler = SelectionHandler()
    
    # Test point selection
    state = handler.handle_event(Event(
        'select',
        x=100,
        y=100,
        data={'indices': [0, 1, 2]}
    ))
    assert state.selected_indices == {0, 1, 2}
    
    # Test selection addition with shift
    state = handler.handle_event(Event(
        'select',
        x=200,
        y=200,
        data={'indices': [3, 4, 5]},
        modifiers={'shift': True}
    ))
    assert state.selected_indices == {0, 1, 2, 3, 4, 5}
    
    # Test selection clearing
    state = handler.handle_event(Event(
        'select',
        x=300,
        y=300,
        data={'indices': [6, 7, 8]},
        modifiers={'shift': False}
    ))
    assert state.selected_indices == {6, 7, 8}

def test_interaction_state():
    """Test interaction state management."""
    state = InteractionState()
    
    # Test state updates
    state.update(scale=2.0, center=(100, 100))
    assert state.scale == 2.0
    assert state.center == (100, 100)
    
    # Test state history
    state.update(scale=1.5)
    assert len(state.history) == 2
    
    # Test state rollback
    state.rollback()
    assert state.scale == 2.0
    
    # Test state reset
    state.reset()
    assert state.scale == 1.0
    assert state.center == (0, 0)
    assert not state.history

def test_interaction_handler_integration(interaction_handler, sample_data):
    """Test interaction handler integration."""
    # Test event handling
    mock_callback = Mock()
    interaction_handler.on('click', mock_callback)
    
    event = Event('click', x=100, y=100)
    interaction_handler.handle_event(event)
    mock_callback.assert_called_once_with(event)
    
    # Test state management
    interaction_handler.handle_event(Event(
        'zoom',
        x=100,
        y=100,
        data={'scale': 2.0, 'center': (100, 100)}
    ))
    assert interaction_handler.state.scale == 2.0
    
    # Test compound interactions
    interaction_handler.handle_event(Event(
        'pan',
        x=150,
        y=150,
        data={'dx': 50, 'dy': 30}
    ))
    assert interaction_handler.state.scale == 2.0  # Zoom preserved
    assert interaction_handler.state.offset == (50, 30)  # Pan applied

def test_interaction_constraints():
    """Test interaction constraints and validation."""
    handler = InteractionHandler()
    
    # Test coordinate bounds
    with pytest.raises(ValueError):
        handler.handle_event(Event('click', x=-1, y=-1))
    
    # Test invalid state transitions
    handler.handle_event(Event(
        'zoom',
        x=100,
        y=100,
        data={'scale': 0.1, 'center': (100, 100)}
    ))
    assert handler.state.scale >= handler.min_scale
    
    # Test interaction mode restrictions
    handler.mode = 'view_only'
    with pytest.raises(ValueError):
        handler.handle_event(Event('select', x=100, y=100))

def test_interaction_serialization(interaction_handler, tmp_path):
    """Test interaction state serialization."""
    # Create some interaction state
    interaction_handler.handle_event(Event(
        'zoom',
        x=100,
        y=100,
        data={'scale': 2.0, 'center': (100, 100)}
    ))
    
    # Save state
    save_path = tmp_path / 'interaction_state.json'
    interaction_handler.save_state(save_path)
    
    # Load state
    new_handler = InteractionHandler()
    new_handler.load_state(save_path)
    
    # Verify state
    assert new_handler.state.scale == interaction_handler.state.scale
    assert new_handler.state.center == interaction_handler.state.center 