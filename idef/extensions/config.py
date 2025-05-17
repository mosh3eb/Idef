"""
Plugin Configuration Management Module for IDEF.

This module handles plugin configuration loading, validation, and persistence.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import jsonschema
from datetime import datetime
@dataclass
class PluginConfig:
    """Container for plugin configuration."""
    name: str
    enabled: bool = True
    settings: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PluginConfig':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            enabled=data.get('enabled', True),
            settings=data.get('settings', {})
        )

class ConfigManager:
    """Manages plugin configurations."""
    
    def __init__(self, config_dir: Union[str, Path]):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._configs: Dict[str, PluginConfig] = {}
        self._schemas: Dict[str, Dict] = {}
        
    def load_configs(self):
        """Load all plugin configurations."""
        config_file = self.config_dir / 'plugin_configs.yaml'
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
                if data and isinstance(data, dict):
                    for name, config_data in data.items():
                        self._configs[name] = PluginConfig.from_dict({
                            'name': name,
                            **config_data
                        })
    
    def save_configs(self):
        """Save all plugin configurations."""
        config_file = self.config_dir / 'plugin_configs.yaml'
        data = {
            name: {
                'enabled': config.enabled,
                'settings': config.settings
            }
            for name, config in self._configs.items()
        }
        with open(config_file, 'w') as f:
            yaml.dump(data, f)
    
    def register_plugin(self, name: str, schema: Dict):
        """Register a plugin with its configuration schema."""
        try:
            # Validate schema format
            jsonschema.Draft7Validator.check_schema(schema)
            self._schemas[name] = schema
            
            # Create default config if not exists
            if name not in self._configs:
                self._configs[name] = PluginConfig(
                    name=name,
                    settings=self._get_default_settings(schema)
                )
            
        except Exception as e:
            raise ValueError(f"Invalid configuration schema: {e}")
    
    def get_config(self, name: str) -> Optional[PluginConfig]:
        """Get plugin configuration."""
        return self._configs.get(name)
    
    def update_config(self, name: str, settings: Dict[str, Any]) -> bool:
        """Update plugin configuration."""
        if name not in self._configs:
            return False
            
        # Validate settings against schema
        if name in self._schemas:
            try:
                jsonschema.validate(settings, self._schemas[name])
            except jsonschema.ValidationError as e:
                raise ValueError(f"Invalid configuration: {e}")
        
        # Update settings
        config = self._configs[name]
        config.settings.update(settings)
        self.save_configs()
        return True
    
    def enable_plugin(self, name: str, enabled: bool = True):
        """Enable or disable a plugin."""
        if name in self._configs:
            self._configs[name].enabled = enabled
            self.save_configs()
    
    def is_plugin_enabled(self, name: str) -> bool:
        """Check if a plugin is enabled."""
        config = self._configs.get(name)
        return config.enabled if config else False
    
    def _get_default_settings(self, schema: Dict) -> Dict[str, Any]:
        """Get default settings from schema."""
        defaults = {}
        if 'properties' in schema:
            for name, prop in schema['properties'].items():
                if 'default' in prop:
                    defaults[name] = prop['default']
        return defaults

class ConfigValidator:
    """Validates plugin configurations."""
    
    @staticmethod
    def validate_config(config: Dict, schema: Dict) -> List[str]:
        """Validate configuration against schema."""
        errors = []
        try:
            jsonschema.validate(config, schema)
        except jsonschema.ValidationError as e:
            path = ' -> '.join(str(p) for p in e.path)
            errors.append(f"Validation error at {path}: {e.message}")
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e}")
        return errors
    
    @staticmethod
    def validate_value(value: Any, property_schema: Dict) -> List[str]:
        """Validate a single configuration value."""
        errors = []
        try:
            jsonschema.validate(value, property_schema)
        except jsonschema.ValidationError as e:
            errors.append(e.message)
        return errors

class ConfigMigrator:
    """Handles plugin configuration migrations."""
    
    def __init__(self, config_dir: Union[str, Path]):
        self.config_dir = Path(config_dir)
        self.backup_dir = self.config_dir / 'backups'
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def backup_config(self, name: str, config: PluginConfig):
        """Create a backup of plugin configuration."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_dir / f"{name}_{timestamp}.yaml"
        
        with open(backup_file, 'w') as f:
            yaml.dump(config.to_dict(), f)
    
    def migrate_config(self, name: str, old_config: Dict,
                      new_schema: Dict) -> Dict:
        """Migrate configuration to new schema."""
        # Create backup
        self.backup_config(name, PluginConfig(name=name, settings=old_config))
        
        # Get default values for new properties
        new_config = ConfigManager._get_default_settings(new_schema)
        
        # Copy over existing values that are still valid
        for key, value in old_config.items():
            if key in new_schema.get('properties', {}):
                try:
                    jsonschema.validate(
                        {key: value},
                        {'properties': {key: new_schema['properties'][key]}}
                    )
                    new_config[key] = value
                except jsonschema.ValidationError:
                    pass  # Use default value instead
        
        return new_config 