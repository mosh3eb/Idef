import pytest
import tempfile
from pathlib import Path
import yaml
import json
from idef.extensions.config import (
    PluginConfig,
    ConfigManager,
    ConfigValidator,
    ConfigMigrator
)

# PluginConfig Tests
class TestPluginConfig:
    def test_plugin_config_creation(self):
        """Test PluginConfig initialization."""
        config = PluginConfig("test_plugin")
        assert config.name == "test_plugin"
        assert config.enabled is True
        assert config.settings is None

    def test_plugin_config_with_settings(self):
        """Test PluginConfig with custom settings."""
        settings = {"param1": "value1"}
        config = PluginConfig("test_plugin", enabled=False, settings=settings)
        assert config.enabled is False
        assert config.settings == settings

    def test_plugin_config_to_dict(self):
        """Test converting PluginConfig to dictionary."""
        settings = {"param1": "value1"}
        config = PluginConfig("test_plugin", enabled=False, settings=settings)
        config_dict = config.to_dict()
        assert config_dict["name"] == "test_plugin"
        assert config_dict["enabled"] is False
        assert config_dict["settings"] == settings

    def test_plugin_config_from_dict(self):
        """Test creating PluginConfig from dictionary."""
        data = {
            "name": "test_plugin",
            "enabled": False,
            "settings": {"param1": "value1"}
        }
        config = PluginConfig.from_dict(data)
        assert config.name == data["name"]
        assert config.enabled == data["enabled"]
        assert config.settings == data["settings"]

# ConfigManager Tests
class TestConfigManager:
    @pytest.fixture
    def config_dir(self):
        """Create temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def manager(self, config_dir):
        """Create ConfigManager instance."""
        return ConfigManager(config_dir)

    @pytest.fixture
    def test_schema(self):
        """Create test configuration schema."""
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Test parameter",
                    "default": "default_value"
                }
            }
        }

    def test_load_configs_empty(self, manager):
        """Test loading configs when no config file exists."""
        manager.load_configs()
        assert len(manager._configs) == 0

    def test_load_configs_existing(self, manager, config_dir):
        """Test loading existing config file."""
        config_data = {
            "test_plugin": {
                "enabled": True,
                "settings": {"param1": "value1"}
            }
        }
        config_file = config_dir / "plugin_configs.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        manager.load_configs()
        assert "test_plugin" in manager._configs
        assert manager._configs["test_plugin"].enabled is True
        assert manager._configs["test_plugin"].settings == {"param1": "value1"}

    def test_save_configs(self, manager, config_dir):
        """Test saving configurations."""
        config = PluginConfig(
            "test_plugin",
            enabled=True,
            settings={"param1": "value1"}
        )
        manager._configs["test_plugin"] = config
        manager.save_configs()

        config_file = config_dir / "plugin_configs.yaml"
        assert config_file.exists()
        with open(config_file) as f:
            data = yaml.safe_load(f)
            assert "test_plugin" in data
            assert data["test_plugin"]["enabled"] is True
            assert data["test_plugin"]["settings"] == {"param1": "value1"}

    def test_register_plugin(self, manager, test_schema):
        """Test registering a plugin with schema."""
        manager.register_plugin("test_plugin", test_schema)
        assert "test_plugin" in manager._schemas
        assert "test_plugin" in manager._configs
        assert manager._configs["test_plugin"].settings == {"param1": "default_value"}

    def test_register_plugin_invalid_schema(self, manager):
        """Test registering plugin with invalid schema."""
        invalid_schema = {"invalid": "schema"}
        with pytest.raises(ValueError, match="Invalid configuration schema"):
            manager.register_plugin("test_plugin", invalid_schema)

    def test_get_config(self, manager):
        """Test getting plugin configuration."""
        config = PluginConfig("test_plugin")
        manager._configs["test_plugin"] = config
        retrieved = manager.get_config("test_plugin")
        assert retrieved == config
        assert manager.get_config("nonexistent") is None

    def test_update_config(self, manager, test_schema):
        """Test updating plugin configuration."""
        manager.register_plugin("test_plugin", test_schema)
        success = manager.update_config("test_plugin", {"param1": "new_value"})
        assert success
        assert manager._configs["test_plugin"].settings["param1"] == "new_value"

    def test_update_config_invalid(self, manager, test_schema):
        """Test updating config with invalid values."""
        manager.register_plugin("test_plugin", test_schema)
        with pytest.raises(ValueError, match="Invalid configuration"):
            manager.update_config("test_plugin", {"param1": 123})  # should be string

    def test_enable_plugin(self, manager):
        """Test enabling/disabling plugin."""
        config = PluginConfig("test_plugin")
        manager._configs["test_plugin"] = config
        manager.enable_plugin("test_plugin", False)
        assert not manager._configs["test_plugin"].enabled
        manager.enable_plugin("test_plugin", True)
        assert manager._configs["test_plugin"].enabled

# ConfigValidator Tests
class TestConfigValidator:
    def test_validate_config_valid(self):
        """Test validating valid configuration."""
        schema = {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            }
        }
        config = {"param1": "value1"}
        errors = ConfigValidator.validate_config(config, schema)
        assert not errors

    def test_validate_config_invalid(self):
        """Test validating invalid configuration."""
        schema = {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            }
        }
        config = {"param1": 123}  # should be string
        errors = ConfigValidator.validate_config(config, schema)
        assert len(errors) == 1
        assert "123 is not of type 'string'" in errors[0]

    def test_validate_value_valid(self):
        """Test validating valid value."""
        schema = {"type": "string"}
        errors = ConfigValidator.validate_value("test", schema)
        assert not errors

    def test_validate_value_invalid(self):
        """Test validating invalid value."""
        schema = {"type": "string"}
        errors = ConfigValidator.validate_value(123, schema)
        assert len(errors) == 1
        assert "123 is not of type 'string'" in errors[0]

# ConfigMigrator Tests
class TestConfigMigrator:
    @pytest.fixture
    def migrator(self):
        """Create ConfigMigrator instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigMigrator(temp_dir)

    def test_backup_config(self, migrator):
        """Test creating config backup."""
        config = PluginConfig(
            "test_plugin",
            settings={"param1": "value1"}
        )
        migrator.backup_config("test_plugin", config)
        backup_files = list(migrator.backup_dir.glob("test_plugin_*.yaml"))
        assert len(backup_files) == 1
        with open(backup_files[0]) as f:
            data = yaml.safe_load(f)
            assert data["name"] == "test_plugin"
            assert data["settings"] == {"param1": "value1"}

    def test_migrate_config(self, migrator):
        """Test migrating configuration to new schema."""
        old_config = {"old_param": "old_value"}
        new_schema = {
            "type": "object",
            "properties": {
                "old_param": {
                    "type": "string"
                },
                "new_param": {
                    "type": "string",
                    "default": "default_value"
                }
            }
        }
        new_config = migrator.migrate_config("test_plugin", old_config, new_schema)
        assert new_config["old_param"] == "old_value"
        assert new_config["new_param"] == "default_value" 