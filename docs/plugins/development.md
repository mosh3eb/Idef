# IDEF Plugin Development Guide

This guide walks you through the process of developing plugins for the Interactive Data Exploration Framework (IDEF).

## Getting Started

### Prerequisites

1. Python 3.8 or higher
2. IDEF installed in development mode:
```bash
git clone https://github.com/your-org/idef.git
cd idef
pip install -e .
```

### Plugin Development Environment

1. Create a new plugin directory:
```bash
mkdir my_plugin
cd my_plugin
```

2. Set up virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install idef[dev]
```

3. Initialize plugin structure:
```
my_plugin/
├── plugin.yaml
├── __init__.py
├── plugin.py
└── tests/
    └── test_plugin.py
```

## Plugin Development Process

### 1. Define Plugin Metadata

Create `plugin.yaml`:
```yaml
name: my_plugin
version: "1.0.0"
description: "My custom IDEF plugin"
author: "Your Name"
plugin_type: DATA_CONNECTOR  # or other type
entry_point: my_plugin.plugin
dependencies:
  - pandas>=1.3.0
  - numpy>=1.20.0
config_schema:
  type: object
  properties:
    param1:
      type: string
      description: "Parameter description"
      default: "default value"
  required:
    - param1
```

### 2. Implement Plugin Interface

Create `plugin.py`:
```python
from typing import Any, Dict
import pandas as pd
from idef.extensions.plugins import DataConnectorPlugin, PluginMetadata

class MyPlugin(DataConnectorPlugin):
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self._initialized = False
    
    def initialize(self, **kwargs) -> bool:
        try:
            # Initialize resources
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def cleanup(self):
        try:
            # Clean up resources
            self._initialized = False
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def connect(self, **kwargs) -> bool:
        if not self._initialized:
            return False
        try:
            # Connect to data source
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def read(self, **kwargs) -> pd.DataFrame:
        if not self._initialized:
            raise RuntimeError("Plugin not initialized")
        try:
            # Read data
            return pd.DataFrame()
        except Exception as e:
            raise RuntimeError(f"Read failed: {e}")
```

### 3. Write Tests

Create `tests/test_plugin.py`:
```python
import pytest
from idef.extensions.testing import PluginTestCase, test_plugin_metadata
from ..plugin import MyPlugin

class TestMyPlugin(PluginTestCase):
    @pytest.fixture
    def plugin_metadata(self):
        return {
            'name': 'test_plugin',
            'version': '1.0.0',
            'description': 'Test plugin',
            'author': 'Test Author',
            'plugin_type': 'DATA_CONNECTOR',
            'entry_point': 'test.plugin'
        }
    
    @test_plugin_metadata
    def test_initialization(self):
        with self.plugin_context(MyPlugin, self.plugin_metadata) as harness:
            assert harness.plugin.initialize()
    
    @test_plugin_metadata
    def test_connect(self):
        with self.plugin_context(MyPlugin, self.plugin_metadata) as harness:
            harness.plugin.initialize()
            assert harness.plugin.connect()
    
    @test_plugin_metadata
    def test_read(self):
        with self.plugin_context(MyPlugin, self.plugin_metadata) as harness:
            harness.plugin.initialize()
            harness.plugin.connect()
            df = harness.plugin.read()
            assert isinstance(df, pd.DataFrame)
```

### 4. Document Your Plugin

Create `README.md`:
```markdown
# My IDEF Plugin

Description of your plugin.

## Installation

```bash
pip install my-plugin
```

## Usage

```python
from idef import PluginManager

# Load plugin
manager = PluginManager()
plugin = manager.load_plugin('my_plugin')

# Configure plugin
plugin.initialize(param1='value1')

# Use plugin
result = plugin.method()
```

## Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| param1 | string | Description | default |

## Examples

```python
# Example code
```
```

## Development Best Practices

### 1. Code Organization

- Keep plugin code modular
- Use clear class and method names
- Follow Python style guidelines (PEP 8)
- Use type hints consistently

### 2. Error Handling

- Handle all expected exceptions
- Provide meaningful error messages
- Log errors appropriately
- Clean up resources in case of errors

### 3. Testing

- Write unit tests for all functionality
- Use test fixtures for common setup
- Mock external dependencies
- Test error cases
- Test configuration validation

### 4. Documentation

- Document all public methods
- Provide usage examples
- Document configuration options
- Keep README up to date

### 5. Performance

- Cache expensive operations
- Use efficient data structures
- Profile code for bottlenecks
- Handle large datasets efficiently

## Plugin Types

### Data Connector

For connecting to data sources:
```python
class MyDataConnector(DataConnectorPlugin):
    def connect(self, **kwargs) -> bool:
        # Connect to data source
        pass
    
    def read(self, **kwargs) -> pd.DataFrame:
        # Read data
        pass
    
    def write(self, data: pd.DataFrame, **kwargs) -> bool:
        # Write data
        pass
```

### Analysis Method

For implementing analysis algorithms:
```python
class MyAnalysis(AnalysisMethodPlugin):
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        # Perform analysis
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        # Define parameters
        pass
```

### Visualization

For creating visualizations:
```python
class MyVisualization(VisualizationPlugin):
    def create_visualization(self, data: Any, **kwargs) -> str:
        # Create visualization
        pass
    
    def get_supported_types(self) -> List[str]:
        # Define supported types
        pass
```

## Configuration Management

### 1. Schema Definition

```yaml
config_schema:
  type: object
  properties:
    string_param:
      type: string
      description: "String parameter"
      default: "default"
    number_param:
      type: number
      description: "Number parameter"
      minimum: 0
      maximum: 100
    enum_param:
      type: string
      description: "Enumerated parameter"
      enum: ["option1", "option2"]
  required:
    - string_param
```

### 2. Configuration Validation

```python
def initialize(self, **kwargs) -> bool:
    # Validate configuration
    validator = ConfigValidator()
    errors = validator.validate_config(kwargs, self.metadata.config_schema)
    if errors:
        logger.error(f"Configuration errors: {errors}")
        return False
    
    # Store configuration
    self._config = kwargs
    return True
```

### 3. Configuration Updates

```python
def update_config(self, **kwargs) -> bool:
    # Validate and update configuration
    validator = ConfigValidator()
    errors = validator.validate_config(kwargs, self.metadata.config_schema)
    if errors:
        return False
    
    self._config.update(kwargs)
    return True
```

## Dependency Management

### 1. Specify Dependencies

```yaml
dependencies:
  - package1>=1.0.0
  - package2>=2.0.0,<3.0.0
```

### 2. Check Dependencies

```python
def initialize(self, **kwargs) -> bool:
    # Check dependencies
    version_manager = VersionManager()
    satisfied, missing = version_manager.check_python_dependencies(
        self.metadata.dependencies
    )
    if not satisfied:
        logger.error(f"Missing dependencies: {missing}")
        return False
    return True
```

## Testing Strategies

### 1. Unit Tests

```python
def test_method(self):
    """Test specific method functionality."""
    with self.plugin_context(MyPlugin, self.plugin_metadata) as harness:
        result = harness.plugin.method()
        assert result == expected
```

### 2. Integration Tests

```python
def test_workflow(self):
    """Test complete workflow."""
    with self.plugin_context(MyPlugin, self.plugin_metadata) as harness:
        harness.plugin.initialize()
        harness.plugin.connect()
        result = harness.plugin.process()
        harness.plugin.cleanup()
        assert result == expected
```

### 3. Error Tests

```python
def test_error_handling(self):
    """Test error conditions."""
    with self.plugin_context(MyPlugin, self.plugin_metadata) as harness:
        with pytest.raises(ValueError):
            harness.plugin.method(invalid_input)
```

## Debugging Tips

1. Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. Use Plugin Test Harness
```python
with PluginTestHarness(MyPlugin, metadata) as harness:
    # Set breakpoints
    harness.plugin.method()
```

3. Profile Performance
```python
import cProfile
cProfile.runctx('plugin.method()', globals(), locals())
```

## Distribution

1. Create `setup.py`:
```python
from setuptools import setup

setup(
    name='my-plugin',
    version='1.0.0',
    packages=['my_plugin'],
    install_requires=[
        'idef>=1.0.0',
        'pandas>=1.3.0'
    ],
    entry_points={
        'idef.plugins': [
            'my_plugin = my_plugin.plugin:MyPlugin'
        ]
    }
)
```

2. Build and Distribute:
```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```

## Support and Resources

- [IDEF Documentation](https://idef.readthedocs.io)
- [Plugin API Reference](./api.md)
- [Example Plugins](./examples.md)
- [GitHub Issues](https://github.com/your-org/idef/issues) 