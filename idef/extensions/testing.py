"""
Plugin Testing Framework for IDEF.

This module provides testing utilities for plugin development and validation.
"""

import pytest
from typing import Any, Dict, List, Type, Optional
from pathlib import Path
import yaml
import tempfile
import shutil
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from .plugins import Plugin, PluginMetadata, PluginManager
from .dependency import PluginValidator
from .config import ConfigManager, ConfigValidator

class PluginTestHarness:
    """Test harness for plugin testing."""
    
    def __init__(self, plugin_class: Type[Plugin], metadata: Dict):
        self.plugin_class = plugin_class
        self.metadata = PluginMetadata(**metadata)
        self.temp_dir = None
        self.config_manager = None
        self.plugin = None
    
    def setup(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create config manager
        self.config_manager = ConfigManager(self.temp_dir)
        
        # Create plugin instance
        self.plugin = self.plugin_class(self.metadata)
    
    def teardown(self):
        """Clean up test environment."""
        if self.plugin:
            self.plugin.cleanup()
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)
    
    def __enter__(self):
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()

class PluginTestCase:
    """Base class for plugin test cases."""
    
    @pytest.fixture
    def plugin_metadata(self) -> Dict:
        """Provide plugin metadata for testing."""
        return {
            'name': 'test_plugin',
            'version': '1.0.0',
            'description': 'Test plugin',
            'author': 'Test Author',
            'plugin_type': 'TEST',
            'entry_point': 'test.plugin',
            'dependencies': []
        }
    
    @pytest.fixture
    def plugin_config(self) -> Dict:
        """Provide plugin configuration for testing."""
        return {}
    
    @contextmanager
    def plugin_context(self, plugin_class: Type[Plugin],
                      metadata: Dict, config: Dict = None):
        """Context manager for plugin testing."""
        with PluginTestHarness(plugin_class, metadata) as harness:
            if config:
                harness.config_manager.register_plugin(
                    metadata['name'],
                    {'type': 'object', 'properties': config}
                )
            yield harness

class MockPluginEnvironment:
    """Mock environment for plugin testing."""
    
    def __init__(self):
        self.plugin_manager = MagicMock(spec=PluginManager)
        self.config_manager = MagicMock(spec=ConfigManager)
        self.temp_dir = tempfile.mkdtemp()
    
    def setup(self):
        """Set up mock environment."""
        # Create temporary plugin directory
        plugin_dir = Path(self.temp_dir) / 'plugins'
        plugin_dir.mkdir(parents=True)
        
        # Configure plugin manager mock
        self.plugin_manager.plugin_dirs = [str(plugin_dir)]
        self.plugin_manager.get_plugin.return_value = None
        
        # Configure config manager mock
        self.config_manager.get_config.return_value = None
    
    def teardown(self):
        """Clean up mock environment."""
        shutil.rmtree(self.temp_dir)
    
    def __enter__(self):
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()

class PluginTestUtils:
    """Utility functions for plugin testing."""
    
    @staticmethod
    def create_test_plugin_yaml(path: Path, metadata: Dict):
        """Create a plugin.yaml file for testing."""
        with open(path / 'plugin.yaml', 'w') as f:
            yaml.dump(metadata, f)
    
    @staticmethod
    def validate_plugin_interface(plugin: Plugin,
                                required_methods: List[str]) -> List[str]:
        """Validate that a plugin implements required methods."""
        missing = []
        for method in required_methods:
            if not hasattr(plugin, method):
                missing.append(method)
            elif not callable(getattr(plugin, method)):
                missing.append(method)
        return missing
    
    @staticmethod
    def create_mock_data(data_type: str) -> Any:
        """Create mock data for testing."""
        if data_type == 'dataframe':
            import pandas as pd
            return pd.DataFrame({'test': [1, 2, 3]})
        elif data_type == 'graph':
            import networkx as nx
            return nx.Graph()
        elif data_type == 'array':
            import numpy as np
            return np.array([1, 2, 3])
        else:
            return None

def test_plugin_metadata(metadata: Dict):
    """Test decorator for validating plugin metadata."""
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            # Validate metadata
            valid, errors = PluginValidator.validate_metadata(metadata)
            if not valid:
                pytest.fail(f"Invalid plugin metadata: {errors}")
            return test_func(*args, **kwargs)
        return wrapper
    return decorator

def test_plugin_config(config: Dict):
    """Test decorator for validating plugin configuration."""
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            # Validate configuration schema
            valid, errors = PluginValidator.validate_config_schema(config)
            if not valid:
                pytest.fail(f"Invalid configuration schema: {errors}")
            return test_func(*args, **kwargs)
        return wrapper
    return decorator

class PluginTestRunner:
    """Runner for plugin tests."""
    
    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self.results = {}
    
    def discover_tests(self) -> List[str]:
        """Discover plugin test files."""
        test_files = []
        for path in self.plugin_dir.rglob('test_*.py'):
            if path.is_file():
                test_files.append(str(path))
        return test_files
    
    def run_tests(self, test_files: List[str] = None):
        """Run plugin tests."""
        if test_files is None:
            test_files = self.discover_tests()
            
        for test_file in test_files:
            # Run pytest on each test file
            result = pytest.main(['-v', test_file])
            self.results[test_file] = result
    
    def get_results(self) -> Dict[str, int]:
        """Get test results."""
        return self.results

class PluginTestReport:
    """Test report generator for plugins."""
    
    def __init__(self, results: Dict[str, int]):
        self.results = results
        
    def generate_summary(self) -> str:
        """Generate test summary."""
        total = len(self.results)
        passed = sum(1 for result in self.results.values() if result == 0)
        failed = total - passed
        
        summary = [
            "Plugin Test Summary",
            "==================",
            f"Total tests: {total}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            "",
            "Detailed Results:",
            "----------------"
        ]
        
        for test_file, result in self.results.items():
            status = "PASSED" if result == 0 else "FAILED"
            summary.append(f"{test_file}: {status}")
        
        return "\n".join(summary)
    
    def save_report(self, path: Path):
        """Save test report to file."""
        with open(path, 'w') as f:
            f.write(self.generate_summary()) 