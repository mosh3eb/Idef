# IDEF Plugin System

The Interactive Data Exploration Framework (IDEF) provides a powerful plugin system that allows you to extend its functionality with custom components, analysis methods, and visualizations.

## Table of Contents

1. [Plugin Types](#plugin-types)
2. [Creating Plugins](#creating-plugins)
3. [Plugin Configuration](#plugin-configuration)
4. [Testing Plugins](#testing-plugins)
5. [Plugin Examples](#plugin-examples)

## Plugin Types

IDEF supports several types of plugins:

- **Data Connectors**: Connect to various data sources
- **Analysis Methods**: Implement custom data analysis algorithms
- **Visualizations**: Create custom visualization types
- **UI Components**: Add new interface elements
- **Export Formats**: Support additional export formats

## Creating Plugins

To create a plugin, you need:

1. A plugin metadata file (`plugin.yaml`):
```yaml
name: my_plugin
version: "1.0.0"
description: "My custom plugin"
author: "Your Name"
plugin_type: DATA_CONNECTOR  # or other type
entry_point: my_package.my_module
dependencies:
  - package>=1.0.0
config_schema:
  type: object
  properties:
    option1:
      type: string
      description: "Option description"
```

2. A Python module implementing the plugin:
```python
from idef.extensions.plugins import DataConnectorPlugin, PluginMetadata

class MyPlugin(DataConnectorPlugin):
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
    
    def initialize(self, **kwargs) -> bool:
        # Initialize your plugin
        return True
    
    def cleanup(self):
        # Clean up resources
        pass
```

## Plugin Configuration

Plugins can be configured using:

1. Configuration Schema:
```yaml
config_schema:
  type: object
  properties:
    host:
      type: string
      description: "Server hostname"
    port:
      type: integer
      description: "Server port"
      default: 8080
```

2. Runtime Configuration:
```python
config_manager.update_config('my_plugin', {
    'host': 'localhost',
    'port': 8080
})
```

## Testing Plugins

IDEF provides a testing framework for plugins:

```python
from idef.extensions.testing import PluginTestCase, test_plugin_metadata

class TestMyPlugin(PluginTestCase):
    @test_plugin_metadata({
        'name': 'test_plugin',
        'version': '1.0.0',
        # ... other metadata
    })
    def test_initialization(self):
        with self.plugin_context(MyPlugin, self.plugin_metadata) as harness:
            assert harness.plugin.initialize()
```

## Plugin Examples

### Data Connector Plugin

```python
class MongoDBConnector(DataConnectorPlugin):
    def connect(self, **kwargs) -> bool:
        host = kwargs.get('host', 'localhost')
        port = kwargs.get('port', 27017)
        # Connect to MongoDB
        return True
    
    def read(self, **kwargs) -> pd.DataFrame:
        # Read data
        pass
```

### Visualization Plugin

```python
class NetworkVisualizer(VisualizationPlugin):
    def create_visualization(self, data: Any, **kwargs) -> str:
        # Create network visualization
        pass
    
    def get_supported_types(self) -> List[str]:
        return ['network', 'graph']
```

### Analysis Method Plugin

```python
class TimeSeriesAnalyzer(AnalysisMethodPlugin):
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        # Perform analysis
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'window_size': {
                'type': 'integer',
                'default': 10
            }
        }
```

## Best Practices

1. **Metadata**:
   - Use semantic versioning
   - Provide clear descriptions
   - Specify all dependencies

2. **Configuration**:
   - Use schema validation
   - Provide sensible defaults
   - Document all options

3. **Implementation**:
   - Handle errors gracefully
   - Clean up resources
   - Follow type hints

4. **Testing**:
   - Write comprehensive tests
   - Use test fixtures
   - Mock external dependencies

## Plugin Development Workflow

1. Create plugin structure:
```
my_plugin/
├── plugin.yaml
├── __init__.py
├── plugin.py
└── tests/
    └── test_plugin.py
```

2. Implement plugin class
3. Write tests
4. Create documentation
5. Test installation and usage
6. Submit for review

## Troubleshooting

Common issues and solutions:

1. **Plugin not found**:
   - Check plugin directory
   - Verify entry_point

2. **Initialization failed**:
   - Check dependencies
   - Verify configuration

3. **Type errors**:
   - Follow interface
   - Use type hints

## Contributing

To contribute a plugin:

1. Fork the repository
2. Create feature branch
3. Implement plugin
4. Add tests
5. Create pull request

## Resources

- [API Reference](./api.md)
- [Development Guide](./development.md)
- [Example Plugins](./examples.md) 