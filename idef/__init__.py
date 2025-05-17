"""
Interactive Data Exploration Framework (IDEF)

A flexible framework for interactive data exploration and visualization.
"""

from .extensions import (
    # Plugin Base
    Plugin,
    PluginMetadata,
    
    # Dependency Management
    DependencyNode,
    DependencyResolver,
    VersionManager,
    PluginValidator,
    
    # Configuration Management
    PluginConfig,
    ConfigManager,
    ConfigValidator,
    ConfigMigrator,
    
    # Testing Framework
    PluginTestHarness,
    PluginTestCase,
    MockPluginEnvironment,
    PluginTestUtils,
    test_plugin_metadata,
    test_plugin_config,
    PluginTestRunner,
    PluginTestReport
)

from .extensions.plugins import (
    MongoDBConnector,
    ProphetForecaster,
    NetworkVisualizer
)

__version__ = '1.0.0'

__all__ = [
    # Plugin Base
    'Plugin',
    'PluginMetadata',
    
    # Dependency Management
    'DependencyNode',
    'DependencyResolver',
    'VersionManager',
    'PluginValidator',
    
    # Configuration Management
    'PluginConfig',
    'ConfigManager',
    'ConfigValidator',
    'ConfigMigrator',
    
    # Testing Framework
    'PluginTestHarness',
    'PluginTestCase',
    'MockPluginEnvironment',
    'PluginTestUtils',
    'test_plugin_metadata',
    'test_plugin_config',
    'PluginTestRunner',
    'PluginTestReport',
    
    # Example Plugins
    'MongoDBConnector',
    'ProphetForecaster',
    'NetworkVisualizer'
]

