"""
Configuration Management Module for IDEF.

This module handles application configuration and settings.
"""

import os
import json
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict, field
import yaml

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    default_backend: str = 'plotly'
    default_theme: str = 'default'
    default_colormap: str = 'viridis'
    width: int = 800
    height: int = 600
    interactive: bool = True

@dataclass
class CacheConfig:
    """Configuration for caching settings."""
    cache_dir: str = '.idef/cache'
    max_memory_size: int = 512 * 1024 * 1024  # 512MB
    max_disk_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    default_ttl: int = 3600  # 1 hour
    cache_type: str = 'memory'

@dataclass
class AnalysisConfig:
    """Configuration for analysis settings."""
    default_random_state: int = 42
    n_jobs: int = -1  # Use all available cores
    default_feature_importance_method: str = 'auto'
    default_clustering_method: str = 'kmeans'
    default_dimension_reduction: str = 'pca'

@dataclass
class AppConfig:
    """Main application configuration."""
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    log_level: str = 'INFO'
    debug_mode: bool = False

class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        home = Path.home()
        return str(home / '.idef' / 'config.yaml')
        
    def _load_config(self) -> AppConfig:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                return self._dict_to_config(config_dict)
        return AppConfig()
        
    def _dict_to_config(self, config_dict: Dict) -> AppConfig:
        """Convert dictionary to AppConfig."""
        viz_config = VisualizationConfig(
            **config_dict.get('visualization', {})
        )
        cache_config = CacheConfig(
            **config_dict.get('cache', {})
        )
        analysis_config = AnalysisConfig(
            **config_dict.get('analysis', {})
        )
        
        return AppConfig(
            visualization=viz_config,
            cache=cache_config,
            analysis=analysis_config,
            log_level=config_dict.get('log_level', 'INFO'),
            debug_mode=config_dict.get('debug_mode', False)
        )
        
    def save_config(self):
        """Save current configuration to file."""
        config_dict = {
            'visualization': asdict(self.config.visualization),
            'cache': asdict(self.config.cache),
            'analysis': asdict(self.config.analysis),
            'log_level': self.config.log_level,
            'debug_mode': self.config.debug_mode
        }
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    def update_config(self, section: str, **kwargs):
        """Update configuration settings."""
        if section == 'visualization':
            for key, value in kwargs.items():
                setattr(self.config.visualization, key, value)
        elif section == 'cache':
            for key, value in kwargs.items():
                setattr(self.config.cache, key, value)
        elif section == 'analysis':
            for key, value in kwargs.items():
                setattr(self.config.analysis, key, value)
        elif section == 'app':
            for key, value in kwargs.items():
                setattr(self.config, key, value)
        else:
            raise ValueError(f"Unknown configuration section: {section}")
            
        self.save_config()
        
    def get_config(self, section: Optional[str] = None) -> Any:
        """Get configuration settings."""
        if section is None:
            return self.config
        elif section == 'visualization':
            return self.config.visualization
        elif section == 'cache':
            return self.config.cache
        elif section == 'analysis':
            return self.config.analysis
        elif section == 'app':
            return self.config
        else:
            raise ValueError(f"Unknown configuration section: {section}")

# Global configuration instance
config_manager = ConfigManager()
