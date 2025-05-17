# IDEF Example Plugins

This document provides examples of different types of IDEF plugins.

## Table of Contents

1. [Data Connector Examples](#data-connector-examples)
2. [Analysis Method Examples](#analysis-method-examples)
3. [Visualization Examples](#visualization-examples)

## Data Connector Examples

### MongoDB Connector

A plugin for connecting to MongoDB databases.

#### Metadata (`plugin.yaml`):
```yaml
name: mongodb_connector
version: "1.0.0"
description: "MongoDB data connector for IDEF"
author: "IDEF Team"
plugin_type: DATA_CONNECTOR
entry_point: idef.extensions.plugins.mongodb_connector.connector
dependencies:
  - pymongo>=4.0.0
config_schema:
  type: object
  properties:
    host:
      type: string
      description: "MongoDB host"
    port:
      type: integer
      description: "MongoDB port"
    database:
      type: string
      description: "Database name"
    collection:
      type: string
      description: "Collection name"
  required:
    - host
    - database
    - collection
```

#### Implementation:
```python
from typing import Any, Dict, Optional
import pandas as pd
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from idef.extensions.plugins import DataConnectorPlugin, PluginMetadata

class MongoDBConnector(DataConnectorPlugin):
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._collection: Optional[Collection] = None
    
    def initialize(self, **kwargs) -> bool:
        try:
            # Get configuration
            host = kwargs.get('host', 'localhost')
            port = kwargs.get('port', 27017)
            database = kwargs['database']
            collection = kwargs['collection']
            
            # Create client
            self._client = MongoClient(host, port)
            self._db = self._client[database]
            self._collection = self._db[collection]
            
            return True
        except Exception as e:
            self._client = None
            self._db = None
            self._collection = None
            return False
    
    def cleanup(self):
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._collection = None
    
    def connect(self, **kwargs) -> bool:
        if not self._client:
            return self.initialize(**kwargs)
        return True
    
    def read(self, query: Dict = None, **kwargs) -> pd.DataFrame:
        if not self._collection:
            raise RuntimeError("Not connected to MongoDB")
        
        # Execute query
        cursor = self._collection.find(query or {})
        
        # Convert to DataFrame
        data = list(cursor)
        if not data:
            return pd.DataFrame()
            
        return pd.DataFrame(data)
    
    def write(self, data: pd.DataFrame, **kwargs) -> bool:
        if not self._collection:
            raise RuntimeError("Not connected to MongoDB")
            
        try:
            # Convert DataFrame to records
            records = data.to_dict('records')
            
            # Insert records
            result = self._collection.insert_many(records)
            
            return bool(result.inserted_ids)
        except Exception as e:
            return False
```

#### Usage:
```python
from idef import PluginManager

# Load plugin
manager = PluginManager()
plugin = manager.load_plugin('mongodb_connector')

# Configure and connect
plugin.initialize(
    host='localhost',
    port=27017,
    database='mydb',
    collection='mycollection'
)

# Read data
df = plugin.read({'status': 'active'})

# Write data
plugin.write(df)

# Clean up
plugin.cleanup()
```

## Analysis Method Examples

### Prophet Forecasting

A plugin for time series forecasting using Facebook Prophet.

#### Metadata (`plugin.yaml`):
```yaml
name: prophet_forecast
version: "1.0.0"
description: "Time series forecasting plugin using Facebook Prophet"
author: "IDEF Team"
plugin_type: ANALYSIS_METHOD
entry_point: idef.extensions.plugins.prophet_forecast.forecaster
dependencies:
  - prophet>=1.1.0
  - pandas>=1.3.0
config_schema:
  type: object
  properties:
    periods:
      type: integer
      description: "Number of periods to forecast"
      default: 30
    freq:
      type: string
      description: "Frequency of the time series"
      default: "D"
    seasonality_mode:
      type: string
      description: "Seasonality mode (additive or multiplicative)"
      enum: ["additive", "multiplicative"]
      default: "additive"
    changepoint_prior_scale:
      type: number
      description: "Flexibility of the trend"
      default: 0.05
```

#### Implementation:
```python
from typing import Any, Dict
import pandas as pd
from prophet import Prophet

from idef.extensions.plugins import AnalysisMethodPlugin, PluginMetadata

class ProphetForecaster(AnalysisMethodPlugin):
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self._model = None
        self._params = {}
    
    def initialize(self, **kwargs) -> bool:
        try:
            # Store parameters
            self._params = {
                'periods': kwargs.get('periods', 30),
                'freq': kwargs.get('freq', 'D'),
                'seasonality_mode': kwargs.get('seasonality_mode', 'additive'),
                'changepoint_prior_scale': kwargs.get('changepoint_prior_scale', 0.05)
            }
            return True
        except Exception as e:
            return False
    
    def cleanup(self):
        self._model = None
        self._params = {}
    
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        # Validate input data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        # Get column names
        ds_col = kwargs.get('date_col', 'ds')
        y_col = kwargs.get('value_col', 'y')
        
        if ds_col not in data.columns or y_col not in data.columns:
            raise ValueError(f"Data must contain '{ds_col}' and '{y_col}' columns")
        
        # Prepare data for Prophet
        df = data.rename(columns={ds_col: 'ds', y_col: 'y'})
        
        # Create and fit model
        self._model = Prophet(
            seasonality_mode=self._params['seasonality_mode'],
            changepoint_prior_scale=self._params['changepoint_prior_scale']
        )
        
        # Add additional regressors if specified
        regressors = kwargs.get('regressors', [])
        for regressor in regressors:
            if regressor in data.columns:
                self._model.add_regressor(regressor)
        
        self._model.fit(df)
        
        # Create future dataframe
        future = self._model.make_future_dataframe(
            periods=self._params['periods'],
            freq=self._params['freq']
        )
        
        # Make forecast
        forecast = self._model.predict(future)
        
        # Prepare results
        results = {
            'forecast': forecast,
            'model': self._model,
            'metrics': {
                'mse': ((df['y'] - forecast['yhat'][:len(df)])** 2).mean(),
                'rmse': ((df['y'] - forecast['yhat'][:len(df)])** 2).mean() ** 0.5
            }
        }
        
        return results
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'periods': {
                'type': 'integer',
                'description': 'Number of periods to forecast',
                'default': 30
            },
            'freq': {
                'type': 'string',
                'description': 'Frequency of the time series',
                'default': 'D'
            },
            'seasonality_mode': {
                'type': 'string',
                'description': 'Seasonality mode',
                'options': ['additive', 'multiplicative'],
                'default': 'additive'
            }
        }
```

#### Usage:
```python
from idef import PluginManager
import pandas as pd

# Load plugin
manager = PluginManager()
plugin = manager.load_plugin('prophet_forecast')

# Configure
plugin.initialize(
    periods=30,
    freq='D',
    seasonality_mode='additive'
)

# Prepare data
data = pd.DataFrame({
    'ds': pd.date_range('2023-01-01', '2023-12-31'),
    'y': np.random.randn(365).cumsum()
})

# Analyze
results = plugin.analyze(data)

# Access results
forecast = results['forecast']
metrics = results['metrics']
```

## Visualization Examples

### Network Graph

A plugin for creating network visualizations.

#### Metadata (`plugin.yaml`):
```yaml
name: network_viz
version: "1.0.0"
description: "Network graph visualization plugin for IDEF"
author: "IDEF Team"
plugin_type: VISUALIZATION
entry_point: idef.extensions.plugins.network_viz.visualizer
dependencies:
  - networkx>=2.6.0
  - pyvis>=0.2.0
config_schema:
  type: object
  properties:
    height:
      type: string
      description: "Height of the network visualization"
      default: "500px"
    width:
      type: string
      description: "Width of the network visualization"
      default: "100%"
    bgcolor:
      type: string
      description: "Background color"
      default: "#ffffff"
    font_color:
      type: string
      description: "Font color for node labels"
      default: "#000000"
```

#### Implementation:
```python
from typing import Any, Dict, List
import networkx as nx
from pyvis.network import Network
import pandas as pd
from pathlib import Path

from idef.extensions.plugins import VisualizationPlugin, PluginMetadata

class NetworkVisualizer(VisualizationPlugin):
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self._config = {}
        self._supported_types = ['network', 'graph', 'directed_graph']
    
    def initialize(self, **kwargs) -> bool:
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
        self._config = {}
    
    def create_visualization(self, data: Any, **kwargs) -> str:
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
        return self._supported_types
```

#### Usage:
```python
from idef import PluginManager
import networkx as nx
import pandas as pd

# Load plugin
manager = PluginManager()
plugin = manager.load_plugin('network_viz')

# Configure
plugin.initialize(
    height='600px',
    width='100%',
    bgcolor='#f0f0f0'
)

# Create sample graph
G = nx.random_geometric_graph(20, 0.2)

# Create visualization
html = plugin.create_visualization(G, physics=True)

# Or use DataFrame
edges = pd.DataFrame({
    'source': [0, 1, 2],
    'target': [1, 2, 0],
    'weight': [1, 2, 3]
})

html = plugin.create_visualization(
    edges,
    source_col='source',
    target_col='target',
    edge_attrs=['weight']
)
```

## Contributing Examples

To contribute a new example plugin:

1. Create a new directory in `examples/`
2. Add the following files:
   - `plugin.yaml`: Plugin metadata
   - `plugin.py`: Plugin implementation
   - `README.md`: Plugin documentation
   - `tests/`: Test files
3. Submit a pull request

## Best Practices

When creating example plugins:

1. **Documentation**:
   - Provide clear descriptions
   - Include usage examples
   - Document all parameters

2. **Code Quality**:
   - Follow PEP 8 style
   - Add type hints
   - Include docstrings

3. **Testing**:
   - Write comprehensive tests
   - Include edge cases
   - Test error handling

4. **Dependencies**:
   - Minimize dependencies
   - Use stable versions
   - Document requirements 