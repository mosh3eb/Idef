import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
from pymongo import MongoClient
from prophet import Prophet
from pyvis.network import Network

from idef.extensions.testing import PluginTestCase, test_plugin_metadata
from idef.extensions.plugins import Plugin, PluginMetadata
from idef.extensions.plugins.mongodb_connector.connector import MongoDBConnector
from idef.extensions.plugins.prophet_forecast.forecaster import ProphetForecaster
from idef.extensions.plugins.network_viz.visualizer import NetworkVisualizer

# MongoDB Connector Tests
class TestMongoDBConnector(PluginTestCase):
    @pytest.fixture
    def mongo_metadata(self):
        return {
            'name': 'mongodb_connector',
            'version': '1.0.0',
            'description': 'MongoDB data connector',
            'author': 'Test Author',
            'plugin_type': 'DATA_CONNECTOR',
            'entry_point': 'idef.extensions.plugins.mongodb_connector.connector'
        }

    @pytest.fixture
    def mock_mongo_client(self, monkeypatch):
        """Mock MongoDB client."""
        class MockCollection:
            def find(self, query=None):
                return [{'id': 1, 'value': 'test'}]
            
            def insert_many(self, records):
                class MockResult:
                    inserted_ids = [1, 2, 3]
                return MockResult()

        class MockDB:
            def __getitem__(self, name):
                return MockCollection()

        class MockClient:
            def __init__(self, *args, **kwargs):
                pass
            
            def __getitem__(self, name):
                return MockDB()
            
            def close(self):
                pass

        monkeypatch.setattr("pymongo.MongoClient", MockClient)
        return MockClient

    @test_plugin_metadata
    def test_mongodb_initialization(self, mongo_metadata, mock_mongo_client):
        """Test MongoDB connector initialization."""
        with self.plugin_context(MongoDBConnector, mongo_metadata) as harness:
            success = harness.plugin.initialize(
                host='localhost',
                database='test_db',
                collection='test_collection'
            )
            assert success
            assert harness.plugin._client is not None
            assert harness.plugin._collection is not None

    @test_plugin_metadata
    def test_mongodb_read(self, mongo_metadata, mock_mongo_client):
        """Test MongoDB connector read operation."""
        with self.plugin_context(MongoDBConnector, mongo_metadata) as harness:
            harness.plugin.initialize(
                host='localhost',
                database='test_db',
                collection='test_collection'
            )
            df = harness.plugin.read({'status': 'active'})
            assert isinstance(df, pd.DataFrame)
            assert not df.empty

    @test_plugin_metadata
    def test_mongodb_write(self, mongo_metadata, mock_mongo_client):
        """Test MongoDB connector write operation."""
        with self.plugin_context(MongoDBConnector, mongo_metadata) as harness:
            harness.plugin.initialize(
                host='localhost',
                database='test_db',
                collection='test_collection'
            )
            df = pd.DataFrame({'id': [1, 2], 'value': ['a', 'b']})
            success = harness.plugin.write(df)
            assert success

# Prophet Forecasting Tests
class TestProphetForecaster(PluginTestCase):
    @pytest.fixture
    def prophet_metadata(self):
        return {
            'name': 'prophet_forecast',
            'version': '1.0.0',
            'description': 'Prophet forecasting plugin',
            'author': 'Test Author',
            'plugin_type': 'ANALYSIS_METHOD',
            'entry_point': 'idef.extensions.plugins.prophet_forecast.forecaster'
        }

    @pytest.fixture
    def sample_timeseries(self):
        """Create sample time series data."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        values = np.random.randn(len(dates)).cumsum()
        return pd.DataFrame({
            'ds': dates,
            'y': values
        })

    @test_plugin_metadata
    def test_prophet_initialization(self, prophet_metadata):
        """Test Prophet forecaster initialization."""
        with self.plugin_context(ProphetForecaster, prophet_metadata) as harness:
            success = harness.plugin.initialize(
                periods=30,
                freq='D',
                seasonality_mode='additive'
            )
            assert success
            assert harness.plugin._params['periods'] == 30
            assert harness.plugin._params['freq'] == 'D'

    @test_plugin_metadata
    def test_prophet_analysis(self, prophet_metadata, sample_timeseries):
        """Test Prophet forecaster analysis."""
        with self.plugin_context(ProphetForecaster, prophet_metadata) as harness:
            harness.plugin.initialize(periods=30, freq='D')
            results = harness.plugin.analyze(sample_timeseries)
            
            assert 'forecast' in results
            assert 'model' in results
            assert 'metrics' in results
            assert isinstance(results['forecast'], pd.DataFrame)
            assert isinstance(results['model'], Prophet)
            assert 'mse' in results['metrics']
            assert 'rmse' in results['metrics']

    @test_plugin_metadata
    def test_prophet_parameters(self, prophet_metadata):
        """Test Prophet forecaster parameters."""
        with self.plugin_context(ProphetForecaster, prophet_metadata) as harness:
            params = harness.plugin.get_parameters()
            assert 'periods' in params
            assert 'freq' in params
            assert 'seasonality_mode' in params

# Network Visualization Tests
class TestNetworkVisualizer(PluginTestCase):
    @pytest.fixture
    def network_metadata(self):
        return {
            'name': 'network_viz',
            'version': '1.0.0',
            'description': 'Network visualization plugin',
            'author': 'Test Author',
            'plugin_type': 'VISUALIZATION',
            'entry_point': 'idef.extensions.plugins.network_viz.visualizer'
        }

    @pytest.fixture
    def sample_graph(self):
        """Create sample network graph."""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        return G

    @pytest.fixture
    def sample_edges(self):
        """Create sample edge DataFrame."""
        return pd.DataFrame({
            'source': [1, 2, 3],
            'target': [2, 3, 1],
            'weight': [1.0, 2.0, 3.0]
        })

    @test_plugin_metadata
    def test_network_initialization(self, network_metadata):
        """Test network visualizer initialization."""
        with self.plugin_context(NetworkVisualizer, network_metadata) as harness:
            success = harness.plugin.initialize(
                height='500px',
                width='100%',
                bgcolor='#ffffff'
            )
            assert success
            assert harness.plugin._config['height'] == '500px'
            assert harness.plugin._config['width'] == '100%'

    @test_plugin_metadata
    def test_network_visualization_from_graph(self, network_metadata, sample_graph):
        """Test creating visualization from NetworkX graph."""
        with self.plugin_context(NetworkVisualizer, network_metadata) as harness:
            harness.plugin.initialize()
            html = harness.plugin.create_visualization(sample_graph)
            assert isinstance(html, str)
            assert '<html>' in html
            assert 'vis-network' in html

    @test_plugin_metadata
    def test_network_visualization_from_dataframe(self, network_metadata, sample_edges):
        """Test creating visualization from DataFrame."""
        with self.plugin_context(NetworkVisualizer, network_metadata) as harness:
            harness.plugin.initialize()
            html = harness.plugin.create_visualization(
                sample_edges,
                source_col='source',
                target_col='target',
                edge_attrs=['weight']
            )
            assert isinstance(html, str)
            assert '<html>' in html
            assert 'vis-network' in html

    @test_plugin_metadata
    def test_network_supported_types(self, network_metadata):
        """Test supported visualization types."""
        with self.plugin_context(NetworkVisualizer, network_metadata) as harness:
            types = harness.plugin.get_supported_types()
            assert 'network' in types
            assert 'graph' in types
            assert 'directed_graph' in types 