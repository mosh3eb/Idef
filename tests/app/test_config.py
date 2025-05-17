"""
Tests for the configuration management module.
"""

import os
import pytest
import yaml
from pathlib import Path
from idef.app.config import (
    ConfigManager, AppConfig, VisualizationConfig,
    CacheConfig, AnalysisConfig
)

@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / '.idef'
    config_dir.mkdir()
    return config_dir

@pytest.fixture
def config_manager(temp_config_dir):
    """Create a ConfigManager instance with temporary directory."""
    config_path = str(temp_config_dir / 'config.yaml')
    return ConfigManager(config_path)

def test_default_config_creation(config_manager):
    """Test creation of default configuration."""
    config = config_manager.config
    
    assert isinstance(config, AppConfig)
    assert isinstance(config.visualization, VisualizationConfig)
    assert isinstance(config.cache, CacheConfig)
    assert isinstance(config.analysis, AnalysisConfig)
    
    # Check default values
    assert config.visualization.default_backend == 'plotly'
    assert config.cache.cache_type == 'memory'
    assert config.analysis.n_jobs == -1
    assert config.log_level == 'INFO'
    assert not config.debug_mode

def test_config_save_load(config_manager, temp_config_dir):
    """Test saving and loading configuration."""
    # Modify some settings
    config_manager.update_config('visualization', default_backend='matplotlib')
    config_manager.update_config('cache', cache_type='disk')
    config_manager.update_config('app', debug_mode=True)
    
    # Save config
    config_manager.save_config()
    
    # Create new manager to load saved config
    config_path = str(temp_config_dir / 'config.yaml')
    new_manager = ConfigManager(config_path)
    loaded_config = new_manager.config
    
    # Verify loaded values
    assert loaded_config.visualization.default_backend == 'matplotlib'
    assert loaded_config.cache.cache_type == 'disk'
    assert loaded_config.debug_mode == True

def test_config_update(config_manager):
    """Test configuration updates."""
    # Update visualization settings
    config_manager.update_config('visualization',
                               default_backend='matplotlib',
                               width=1024,
                               height=768)
    
    viz_config = config_manager.get_config('visualization')
    assert viz_config.default_backend == 'matplotlib'
    assert viz_config.width == 1024
    assert viz_config.height == 768
    
    # Update cache settings
    config_manager.update_config('cache',
                               cache_type='disk',
                               max_memory_size=1024*1024*1024)
    
    cache_config = config_manager.get_config('cache')
    assert cache_config.cache_type == 'disk'
    assert cache_config.max_memory_size == 1024*1024*1024

def test_invalid_config_section(config_manager):
    """Test handling of invalid configuration sections."""
    with pytest.raises(ValueError):
        config_manager.update_config('invalid_section', some_setting=True)
    
    with pytest.raises(ValueError):
        config_manager.get_config('invalid_section')

def test_config_file_structure(config_manager, temp_config_dir):
    """Test the structure of the saved config file."""
    config_manager.save_config()
    config_path = temp_config_dir / 'config.yaml'
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Check main sections
    assert 'visualization' in config_dict
    assert 'cache' in config_dict
    assert 'analysis' in config_dict
    assert 'log_level' in config_dict
    assert 'debug_mode' in config_dict
    
    # Check nested settings
    viz_config = config_dict['visualization']
    assert 'default_backend' in viz_config
    assert 'default_theme' in viz_config
    
    cache_config = config_dict['cache']
    assert 'cache_dir' in cache_config
    assert 'max_memory_size' in cache_config 