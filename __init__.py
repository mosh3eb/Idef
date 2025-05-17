"""
Interactive Data Exploration Framework (IDEF)
Main package initialization
"""

__version__ = "0.1.0"

from .data.model import Dataset
from .data.connectors import load_dataset
from .visualization.components import visualize
from .app.session import Explorer

# Convenience function to create an explorer instance
def create_explorer():
    """
    Create a new Explorer instance for interactive data exploration.
    
    Returns:
        Explorer: A new explorer instance
    """
    return Explorer()
