# IDEF Plugin System API Reference

This document provides detailed API documentation for the IDEF plugin system.

## Core Classes

### Plugin

Base class for all plugins.

```python
class Plugin(ABC):
    def __init__(self, metadata: PluginMetadata):
        """Initialize plugin with metadata."""
        
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize plugin resources."""
        
    @abstractmethod
    def cleanup(self):
        """Clean up plugin resources."""
```

### PluginMetadata

Container for plugin metadata.

```python
@dataclass
class PluginMetadata:
    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    entry_point: str
    dependencies: List[str] = None
```

## Plugin Types

### DataConnectorPlugin

Interface for data source connectors.

```python
class DataConnectorPlugin(Plugin):
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """Connect to data source."""
        
    @abstractmethod
    def read(self, **kwargs) -> pd.DataFrame:
        """Read data from source."""
        
    @abstractmethod
    def write(self, data: pd.DataFrame, **kwargs) -> bool:
        """Write data to source."""
```

### AnalysisMethodPlugin

Interface for analysis methods.

```python
class AnalysisMethodPlugin(Plugin):
    @abstractmethod
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Perform analysis on data."""
        
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get analysis parameters."""
```

### VisualizationPlugin

Interface for visualizations.

```python
class VisualizationPlugin(Plugin):
    @abstractmethod
    def create_visualization(self, data: Any, **kwargs) -> str:
        """Create visualization from data."""
        
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get supported data types."""
```

## Configuration

### ConfigManager

Manages plugin configurations.

```python
class ConfigManager:
    def __init__(self, config_dir: Union[str, Path]):
        """Initialize with configuration directory."""
        
    def load_configs(self):
        """Load all plugin configurations."""
        
    def save_configs(self):
        """Save all plugin configurations."""
        
    def register_plugin(self, name: str, schema: Dict):
        """Register plugin with configuration schema."""
        
    def get_config(self, name: str) -> Optional[PluginConfig]:
        """Get plugin configuration."""
        
    def update_config(self, name: str, settings: Dict[str, Any]) -> bool:
        """Update plugin configuration."""
```

### ConfigValidator

Validates plugin configurations.

```python
class ConfigValidator:
    @staticmethod
    def validate_config(config: Dict, schema: Dict) -> List[str]:
        """Validate configuration against schema."""
        
    @staticmethod
    def validate_value(value: Any, property_schema: Dict) -> List[str]:
        """Validate a single configuration value."""
```

## Dependency Management

### DependencyResolver

Handles plugin dependency resolution.

```python
class DependencyResolver:
    def add_plugin(self, name: str, version: str, dependencies: List[str]):
        """Add plugin to dependency graph."""
        
    def resolve(self) -> List[str]:
        """Resolve dependencies and return load order."""
```

### VersionManager

Handles version compatibility checks.

```python
class VersionManager:
    @staticmethod
    def check_compatibility(plugin_name: str, required_version: str,
                          installed_version: str) -> bool:
        """Check version compatibility."""
        
    @staticmethod
    def check_python_dependencies(dependencies: List[str]) -> Tuple[bool, List[str]]:
        """Check Python package dependencies."""
```

## Testing

### PluginTestHarness

Test harness for plugin testing.

```python
class PluginTestHarness:
    def __init__(self, plugin_class: Type[Plugin], metadata: Dict):
        """Initialize test harness."""
        
    def setup(self):
        """Set up test environment."""
        
    def teardown(self):
        """Clean up test environment."""
```

### PluginTestCase

Base class for plugin test cases.

```python
class PluginTestCase:
    @pytest.fixture
    def plugin_metadata(self) -> Dict:
        """Provide plugin metadata for testing."""
        
    @pytest.fixture
    def plugin_config(self) -> Dict:
        """Provide plugin configuration for testing."""
        
    @contextmanager
    def plugin_context(self, plugin_class: Type[Plugin],
                      metadata: Dict, config: Dict = None):
        """Context manager for plugin testing."""
```

### Test Decorators

```python
def test_plugin_metadata(metadata: Dict):
    """Test decorator for validating plugin metadata."""
    
def test_plugin_config(config: Dict):
    """Test decorator for validating plugin configuration."""
```

## Plugin Development

### Directory Structure

```
my_plugin/
├── plugin.yaml          # Plugin metadata
├── __init__.py         # Package initialization
├── plugin.py           # Plugin implementation
└── tests/
    └── test_plugin.py  # Plugin tests
```

### Metadata Schema

```yaml
name: str               # Plugin name
version: str           # Semantic version
description: str       # Plugin description
author: str           # Plugin author
plugin_type: str      # Plugin type
entry_point: str      # Plugin entry point
dependencies: list    # Package dependencies
config_schema: dict   # Configuration schema
```

### Configuration Schema

```yaml
type: object
properties:
  property_name:
    type: string|number|boolean|array|object
    description: str
    default: any
    enum: list        # Optional value choices
required: list        # Required properties
```

## Error Handling

Plugins should handle the following error cases:

1. Initialization Errors
```python
def initialize(self, **kwargs) -> bool:
    try:
        # Initialize resources
        return True
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return False
```

2. Resource Cleanup
```python
def cleanup(self):
    try:
        # Clean up resources
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
```

3. Operation Errors
```python
def operation(self, **kwargs):
    try:
        # Perform operation
    except ValueError as e:
        raise PluginError(f"Invalid input: {e}")
    except Exception as e:
        raise PluginError(f"Operation failed: {e}")
```

## Best Practices

1. Type Hints
```python
from typing import Any, Dict, List, Optional

def method(self, param: str) -> Optional[Dict[str, Any]]:
    pass
```

2. Documentation
```python
def method(self, param: str) -> bool:
    """
    Method description.
    
    Args:
        param: Parameter description
        
    Returns:
        bool: Return value description
        
    Raises:
        ValueError: Error description
    """
    pass
```

3. Logging
```python
import logging

logger = logging.getLogger(__name__)

def method(self):
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
```

## Plugin Lifecycle

1. Registration
```python
plugin_manager.register_plugin(metadata)
```

2. Configuration
```python
config_manager.register_plugin(name, schema)
config_manager.update_config(name, settings)
```

3. Initialization
```python
plugin.initialize(**settings)
```

4. Usage
```python
plugin.method(**params)
```

5. Cleanup
```python
plugin.cleanup()
``` 