"""
Visualization Interactions Module for IDEF.

This module provides interaction handling for visualizations.
"""

from typing import Dict, List, Set, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

VALID_EVENT_TYPES = {'click', 'hover', 'zoom', 'pan', 'select'}

@dataclass
class Event:
    """Representation of an interaction event."""
    type: str
    x: float
    y: float
    data: Dict[str, Any] = field(default_factory=dict)
    modifiers: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.type not in VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event type: {self.type}")

@dataclass
class InteractionState:
    """State container for interactions."""
    scale: float = 1.0
    center: Tuple[float, float] = (0, 0)
    offset: Tuple[float, float] = (0, 0)
    selected_indices: Set[int] = field(default_factory=set)
    history: List[Dict] = field(default_factory=list)
    
    def update(self, **kwargs):
        """Update state with new values."""
        # Save current state to history
        self.history.append({
            'scale': self.scale,
            'center': self.center,
            'offset': self.offset,
            'selected_indices': set(self.selected_indices)
        })
        
        # Update state
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def rollback(self):
        """Rollback to previous state."""
        if self.history:
            previous = self.history.pop()
            self.scale = previous['scale']
            self.center = previous['center']
            self.offset = previous['offset']
            self.selected_indices = previous['selected_indices']
    
    def reset(self):
        """Reset state to defaults."""
        self.scale = 1.0
        self.center = (0, 0)
        self.offset = (0, 0)
        self.selected_indices.clear()
        self.history.clear()

class EventDispatcher:
    """Handles event registration and dispatch."""
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
    
    def register(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def unregister(self, event_type: str, handler: Callable):
        """Unregister an event handler."""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)
            if not self._handlers[event_type]:
                del self._handlers[event_type]
    
    def dispatch(self, event: Event):
        """Dispatch an event to registered handlers."""
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                handler(event)

class InteractionHandler:
    """Base class for interaction handlers."""
    
    def __init__(self, supported_events: List[str]):
        self.supported_events = set(supported_events)
        self.dispatcher = EventDispatcher()
        self.state = InteractionState()
    
    def on(self, event_type: str, callback: Callable):
        """Register an event callback."""
        if event_type not in self.supported_events:
            raise ValueError(f"Unsupported event type: {event_type}")
        self.dispatcher.register(event_type, callback)
    
    def off(self, event_type: str, callback: Callable):
        """Unregister an event callback."""
        self.dispatcher.unregister(event_type, callback)
    
    def handle_event(self, event: Event) -> InteractionState:
        """Handle an interaction event."""
        if event.type not in self.supported_events:
            raise ValueError(f"Unsupported event type: {event.type}")
        self.dispatcher.dispatch(event)
        return self.state
    
    def save_state(self, path: Path):
        """Save interaction state."""
        state_dict = {
            'scale': self.state.scale,
            'center': self.state.center,
            'offset': self.state.offset,
            'selected_indices': list(self.state.selected_indices)
        }
        with open(path, 'w') as f:
            json.dump(state_dict, f)
    
    def load_state(self, path: Path):
        """Load interaction state."""
        with open(path, 'r') as f:
            state_dict = json.load(f)
            self.state.scale = state_dict['scale']
            self.state.center = tuple(state_dict['center'])
            self.state.offset = tuple(state_dict['offset'])
            self.state.selected_indices = set(state_dict['selected_indices'])

class ZoomHandler(InteractionHandler):
    """Handles zoom interactions."""
    
    def __init__(self, min_scale: float = 0.1, max_scale: float = 10.0):
        super().__init__(['zoom'])
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def handle_event(self, event: Event) -> InteractionState:
        """Handle zoom event."""
        if event.type != 'zoom':
            return self.state
            
        scale = event.data.get('scale', 1.0)
        center = event.data.get('center', (0, 0))
        
        # Constrain scale
        new_scale = max(min(self.state.scale * scale, self.max_scale), self.min_scale)
        
        self.state.update(
            scale=new_scale,
            center=center
        )
        
        return self.state

class PanHandler(InteractionHandler):
    """Handles pan interactions."""
    
    def __init__(self, max_offset: float = 1000.0):
        super().__init__(['pan'])
        self.max_offset = max_offset
    
    def handle_event(self, event: Event) -> InteractionState:
        """Handle pan event."""
        if event.type != 'pan':
            return self.state
            
        dx = event.data.get('dx', 0)
        dy = event.data.get('dy', 0)
        
        # Update offset with constraints
        current_x, current_y = self.state.offset
        new_x = max(min(current_x + dx, self.max_offset), -self.max_offset)
        new_y = max(min(current_y + dy, self.max_offset), -self.max_offset)
        
        self.state.update(offset=(new_x, new_y))
        
        return self.state

class SelectionHandler(InteractionHandler):
    """Handles selection interactions."""
    
    def __init__(self):
        super().__init__(['select'])
    
    def handle_event(self, event: Event) -> InteractionState:
        """Handle selection event."""
        if event.type != 'select':
            return self.state
            
        indices = set(event.data.get('indices', []))
        
        if event.modifiers.get('shift', False):
            # Add to existing selection
            self.state.update(
                selected_indices=self.state.selected_indices | indices
            )
        else:
            # Replace selection
            self.state.update(selected_indices=indices)
        
        return self.state
