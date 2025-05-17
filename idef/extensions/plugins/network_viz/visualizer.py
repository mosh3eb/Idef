"""
Network Graph Visualization Plugin for IDEF.
"""

from typing import Any, Dict, List
import networkx as nx
from pyvis.network import Network
import pandas as pd
from pathlib import Path

from ...plugins import VisualizationPlugin, PluginMetadata

class NetworkVisualizer(VisualizationPlugin):
    """Network graph visualization plugin."""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self._config = {}
        self._supported_types = ['network', 'graph', 'directed_graph']
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the visualizer."""
        try:
            # Store configuration
            self._config = {
                'height': kwargs.get('height', '500px'),
                'width': kwargs.get('width', '100%'),
                'bgcolor': kwargs.get('bgcolor', '#ffffff'),
                'font_color': kwargs.get('font_color', '#000000')
            }
            return True
        except Exception as e:
            return False
    
    def cleanup(self):
        """Clean up resources."""
        self._config = {}
    
    def create_visualization(self, data: Any, **kwargs) -> str:
        """Create network visualization."""
        # Create network
        net = Network(
            height=self._config['height'],
            width=self._config['width'],
            bgcolor=self._config['bgcolor'],
            font_color=self._config['font_color']
        )
        
        if isinstance(data, nx.Graph):
            # Data is already a NetworkX graph
            graph = data
        elif isinstance(data, pd.DataFrame):
            # Convert DataFrame to graph
            # Assuming DataFrame has 'source' and 'target' columns
            graph = nx.from_pandas_edgelist(
                data,
                source=kwargs.get('source_col', 'source'),
                target=kwargs.get('target_col', 'target'),
                edge_attr=kwargs.get('edge_attrs')
            )
        else:
            raise ValueError("Unsupported data type for network visualization")
        
        # Add nodes and edges to network
        net.from_nx(graph)
        
        # Configure physics
        net.toggle_physics(kwargs.get('physics', True))
        
        # Set other options
        options = kwargs.get('options', {})
        if options:
            net.set_options(options)
        
        # Generate HTML
        output_path = kwargs.get('output_path')
        if output_path:
            output_path = Path(output_path)
            net.save_graph(str(output_path))
            return str(output_path)
        else:
            # Return HTML string
            return net.generate_html()
    
    def get_supported_types(self) -> List[str]:
        """Get supported visualization types."""
        return self._supported_types 