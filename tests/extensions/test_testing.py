import pytest
from pathlib import Path
import tempfile
import yaml
from unittest.mock import MagicMock, patch
import pandas as pd
import networkx as nx

from idef.extensions.testing import (
    PluginTestHarness,
    PluginTestCase,
    MockPluginEnvironment,
    PluginTestUtils,
    test_plugin_metadata,
    test_plugin_config,
    PluginTestRunner,
    PluginTestReport
)
from idef.extensions.plugins import Plugin, PluginMetadata

# Mock Plugin for Testing
class MockPlugin(Plugin):
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.initialized = False
        self.cleaned_up = False

    def initialize(self, **kwargs) -> bool:
        self.initialized = True
        return True

    def cleanup(self):
        self.cleaned_up = True

# PluginTestHarness Tests
class TestPluginTestHarness:
    @pytest.fixture
    def metadata(self):
        return {
            'name': 'mock_plugin',
            'version': '1.0.0',
            'description': 'Mock plugin for testing',
            'author': 'Test Author',
            'plugin_type': 'TEST',
            'entry_point': 'test.plugin'
        }

    def test_harness_setup(self, metadata):
        """Test harness setup."""
        with PluginTestHarness(MockPlugin, metadata) as harness:
            assert harness.plugin is not None
            assert isinstance(harness.plugin, MockPlugin)
            assert harness.temp_dir is not None
            assert Path(harness.temp_dir).exists()

    def test_harness_cleanup(self, metadata):
        """Test harness cleanup."""
        temp_dir = None
        with PluginTestHarness(MockPlugin, metadata) as harness:
            temp_dir = harness.temp_dir
            assert Path(temp_dir).exists()
        assert not Path(temp_dir).exists()

    def test_harness_plugin_lifecycle(self, metadata):
        """Test plugin lifecycle in harness."""
        with PluginTestHarness(MockPlugin, metadata) as harness:
            assert not harness.plugin.initialized
            assert not harness.plugin.cleaned_up
            harness.plugin.initialize()
            assert harness.plugin.initialized
        assert harness.plugin.cleaned_up

# PluginTestCase Tests
class TestPluginTestCase(PluginTestCase):
    def test_plugin_metadata_fixture(self):
        """Test plugin metadata fixture."""
        metadata = self.plugin_metadata()
        assert isinstance(metadata, dict)
        assert 'name' in metadata
        assert 'version' in metadata

    def test_plugin_config_fixture(self):
        """Test plugin config fixture."""
        config = self.plugin_config()
        assert isinstance(config, dict)

    def test_plugin_context(self):
        """Test plugin context manager."""
        metadata = self.plugin_metadata()
        config = {'param1': 'value1'}
        with self.plugin_context(MockPlugin, metadata, config) as harness:
            assert harness.plugin is not None
            assert isinstance(harness.plugin, MockPlugin)

# MockPluginEnvironment Tests
class TestMockPluginEnvironment:
    def test_mock_environment_setup(self):
        """Test mock environment setup."""
        with MockPluginEnvironment() as env:
            assert env.plugin_manager is not None
            assert env.config_manager is not None
            assert Path(env.temp_dir).exists()
            assert Path(env.temp_dir, 'plugins').exists()

    def test_mock_environment_cleanup(self):
        """Test mock environment cleanup."""
        temp_dir = None
        with MockPluginEnvironment() as env:
            temp_dir = env.temp_dir
            assert Path(temp_dir).exists()
        assert not Path(temp_dir).exists()

    def test_mock_plugin_manager(self):
        """Test mock plugin manager."""
        with MockPluginEnvironment() as env:
            assert env.plugin_manager.get_plugin.return_value is None
            env.plugin_manager.get_plugin.return_value = "test"
            assert env.plugin_manager.get_plugin("any") == "test"

# PluginTestUtils Tests
class TestPluginTestUtils:
    @pytest.fixture
    def temp_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_create_test_plugin_yaml(self, temp_path):
        """Test creating plugin.yaml file."""
        metadata = {
            'name': 'test_plugin',
            'version': '1.0.0'
        }
        PluginTestUtils.create_test_plugin_yaml(temp_path, metadata)
        yaml_file = temp_path / 'plugin.yaml'
        assert yaml_file.exists()
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
            assert data == metadata

    def test_validate_plugin_interface(self):
        """Test plugin interface validation."""
        plugin = MockPlugin(PluginMetadata(
            name='test',
            version='1.0.0',
            description='test',
            author='test',
            plugin_type='TEST',
            entry_point='test'
        ))
        missing = PluginTestUtils.validate_plugin_interface(
            plugin,
            ['initialize', 'cleanup', 'nonexistent_method']
        )
        assert 'nonexistent_method' in missing
        assert 'initialize' not in missing
        assert 'cleanup' not in missing

    def test_create_mock_data(self):
        """Test creating mock data."""
        df = PluginTestUtils.create_mock_data('dataframe')
        assert isinstance(df, pd.DataFrame)

        graph = PluginTestUtils.create_mock_data('graph')
        assert isinstance(graph, nx.Graph)

        array = PluginTestUtils.create_mock_data('array')
        assert array is not None

        none = PluginTestUtils.create_mock_data('invalid')
        assert none is None

# Test Decorators Tests
class TestDecorators:
    @test_plugin_metadata({
        'name': 'test',
        'version': '1.0.0',
        'description': 'test',
        'author': 'test',
        'plugin_type': 'TEST',
        'entry_point': 'test'
    })
    def test_metadata_decorator(self):
        """Test plugin metadata decorator."""
        assert True  # If we get here, validation passed

    @test_plugin_config({
        'type': 'object',
        'properties': {
            'param1': {
                'type': 'string',
                'description': 'test'
            }
        }
    })
    def test_config_decorator(self):
        """Test plugin config decorator."""
        assert True  # If we get here, validation passed

# PluginTestRunner Tests
class TestPluginTestRunner:
    @pytest.fixture
    def plugin_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            # Create test files
            (path / 'test_plugin1.py').touch()
            (path / 'test_plugin2.py').touch()
            (path / 'not_a_test.py').touch()
            yield path

    def test_discover_tests(self, plugin_dir):
        """Test test file discovery."""
        runner = PluginTestRunner(plugin_dir)
        test_files = runner.discover_tests()
        assert len(test_files) == 2
        assert all('test_' in file for file in test_files)

    @patch('pytest.main')
    def test_run_tests(self, mock_pytest_main, plugin_dir):
        """Test running tests."""
        runner = PluginTestRunner(plugin_dir)
        mock_pytest_main.return_value = 0
        runner.run_tests()
        assert len(runner.get_results()) == 2
        assert all(result == 0 for result in runner.get_results().values())

# PluginTestReport Tests
class TestPluginTestReport:
    def test_generate_summary(self):
        """Test generating test summary."""
        results = {
            'test1.py': 0,  # passed
            'test2.py': 1   # failed
        }
        report = PluginTestReport(results)
        summary = report.generate_summary()
        assert 'Total tests: 2' in summary
        assert 'Passed: 1' in summary
        assert 'Failed: 1' in summary
        assert 'test1.py: PASSED' in summary
        assert 'test2.py: FAILED' in summary

    def test_save_report(self, tmp_path):
        """Test saving test report."""
        results = {
            'test1.py': 0,
            'test2.py': 1
        }
        report = PluginTestReport(results)
        report_file = tmp_path / 'report.txt'
        report.save_report(report_file)
        assert report_file.exists()
        content = report_file.read_text()
        assert 'Total tests: 2' in content 