"""
Plugin System Module for IDEF.

This module provides the core plugin system functionality, allowing users to extend
IDEF with custom components, analysis methods, and visualizations.
"""

import importlib
import inspect
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Callable
import yaml

# Configure logging
logger = logging.getLogger(__name__)

class PluginType(Enum):
    """Types of plugins supported by IDEF."""
    DATA_CONNECTOR = auto()
    ANALYSIS_METHOD = auto()
    VISUALIZATION = auto()
    UI_COMPONENT = auto()
    EXPORT_FORMAT = auto()

@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    plugin_type: PluginType
    entry_point: str
    config_schema: Optional[Dict] = None

class Plugin(ABC):
    """Base class for all plugins."""
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self._is_initialized = False
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up plugin resources."""
        pass
    
    def __repr__(self) -> str:
        return f"Plugin({self.metadata.name} v{self.metadata.version})"

class PluginManager:
    """Manages plugin discovery, loading, and lifecycle."""
    
    def __init__(self, plugin_dirs: List[str] = None):
        self.plugin_dirs = plugin_dirs or []
        self._plugins: Dict[str, Plugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        
        # Add default plugin directory
        default_dir = Path(__file__).parent / 'plugins'
        if default_dir.exists():
            self.plugin_dirs.append(str(default_dir))
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover available plugins in plugin directories."""
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
                
            # Look for plugin.yaml files
            for root, _, files in os.walk(plugin_dir):
                for file in files:
                    if file == 'plugin.yaml':
                        try:
                            metadata = self._load_plugin_metadata(
                                Path(root) / file
                            )
                            discovered.append(metadata)
                        except Exception as e:
                            logger.error(f"Error loading plugin metadata: {e}")
        
        return discovered
    
    def load_plugin(self, metadata: PluginMetadata) -> Optional[Plugin]:
        """Load a plugin from its metadata."""
        try:
            # Import plugin module
            module_path = metadata.entry_point
            if module_path.endswith('.py'):
                module_path = module_path[:-3]
            module = importlib.import_module(module_path)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and issubclass(obj, Plugin) 
                    and obj != Plugin):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                raise ValueError(f"No plugin class found in {module_path}")
            
            # Create plugin instance
            plugin = plugin_class(metadata)
            self._plugins[metadata.name] = plugin
            
            return plugin
            
        except Exception as e:
            logger.error(f"Error loading plugin {metadata.name}: {e}")
            return None
    
    def initialize_plugin(self, name: str, **kwargs) -> bool:
        """Initialize a loaded plugin."""
        if name not in self._plugins:
            return False
            
        plugin = self._plugins[name]
        if plugin._is_initialized:
            return True
            
        try:
            success = plugin.initialize(**kwargs)
            if success:
                plugin._is_initialized = True
                self._call_hook('post_plugin_init', plugin)
            return success
        except Exception as e:
            logger.error(f"Error initializing plugin {name}: {e}")
            return False
    
    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        if name not in self._plugins:
            return False
            
        plugin = self._plugins[name]
        try:
            self._call_hook('pre_plugin_unload', plugin)
            plugin.cleanup()
            del self._plugins[name]
            return True
        except Exception as e:
            logger.error(f"Error unloading plugin {name}: {e}")
            return False
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a loaded plugin by name."""
        return self._plugins.get(name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """Get all loaded plugins of a specific type."""
        return [p for p in self._plugins.values() 
                if p.metadata.plugin_type == plugin_type]
    
    def register_hook(self, hook_name: str, callback: Callable):
        """Register a hook callback."""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
    
    def _call_hook(self, hook_name: str, *args, **kwargs):
        """Call all registered callbacks for a hook."""
        if hook_name in self._hooks:
            for callback in self._hooks[hook_name]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in hook {hook_name}: {e}")
    
    def _load_plugin_metadata(self, yaml_path: Path) -> PluginMetadata:
        """Load plugin metadata from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        required_fields = {'name', 'version', 'description', 'author',
                         'plugin_type', 'entry_point'}
        missing = required_fields - set(data.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
            
        return PluginMetadata(
            name=data['name'],
            version=data['version'],
            description=data['description'],
            author=data['author'],
            dependencies=data.get('dependencies', []),
            plugin_type=PluginType[data['plugin_type']],
            entry_point=data['entry_point'],
            config_schema=data.get('config_schema')
        )

class DataConnectorPlugin(Plugin):
    """Base class for data connector plugins."""
    
    @abstractmethod
    def connect(self, **kwargs) -> Any:
        """Connect to the data source."""
        pass
    
    @abstractmethod
    def read(self, **kwargs) -> Any:
        """Read data from the source."""
        pass
    
    @abstractmethod
    def write(self, data: Any, **kwargs) -> bool:
        """Write data to the source."""
        pass

class AnalysisMethodPlugin(Plugin):
    """Base class for analysis method plugins."""
    
    @abstractmethod
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Analyze the provided data."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the analysis method parameters."""
        pass

class VisualizationPlugin(Plugin):
    """Base class for visualization plugins."""
    
    @abstractmethod
    def create_visualization(self, data: Any, **kwargs) -> Any:
        """Create a visualization from the data."""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get supported visualization types."""
        pass

class UIComponentPlugin(Plugin):
    """Base class for UI component plugins."""
    
    @abstractmethod
    def create_component(self, **kwargs) -> Any:
        """Create a UI component."""
        pass
    
    @abstractmethod
    def get_component_type(self) -> str:
        """Get the UI component type."""
        pass

class ExportFormatPlugin(Plugin):
    """Base class for export format plugins."""
    
    @abstractmethod
    def export(self, data: Any, path: Union[str, Path], **kwargs) -> bool:
        """Export data to the specified format."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported export formats."""
        pass
