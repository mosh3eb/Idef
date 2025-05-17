"""
IDEF Plugin System Extensions Package.
"""

from .dependency import (
    DependencyNode,
    DependencyResolver,
    VersionManager,
    PluginValidator
)
from .config import (
    PluginConfig,
    ConfigManager,
    ConfigValidator,
    ConfigMigrator
)
from .testing import (
    PluginTestHarness,
    PluginTestCase,
    MockPluginEnvironment,
    PluginTestUtils,
    test_plugin_metadata,
    test_plugin_config,
    PluginTestRunner,
    PluginTestReport
)
from .plugins import Plugin, PluginMetadata

__all__ = [
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
    
    # Plugin Base
    'Plugin',
    'PluginMetadata'
]

