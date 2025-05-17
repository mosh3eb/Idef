import pytest
from packaging.version import Version
from idef.extensions.dependency import (
    DependencyNode,
    DependencyResolver,
    VersionManager,
    PluginValidator
)

# DependencyNode Tests
def test_dependency_node_creation():
    """Test DependencyNode initialization."""
    node = DependencyNode("test_plugin", "1.0.0", ["dep1", "dep2"])
    assert node.name == "test_plugin"
    assert node.version == "1.0.0"
    assert node.dependencies == ["dep1", "dep2"]
    assert node.required_by == set()

def test_dependency_node_required_by():
    """Test DependencyNode required_by set."""
    node = DependencyNode("test_plugin", "1.0.0", [], {"plugin1", "plugin2"})
    assert node.required_by == {"plugin1", "plugin2"}

# DependencyResolver Tests
class TestDependencyResolver:
    @pytest.fixture
    def resolver(self):
        return DependencyResolver()

    def test_add_plugin(self, resolver):
        """Test adding a plugin to the resolver."""
        resolver.add_plugin("plugin1", "1.0.0", ["dep1"])
        assert "plugin1" in resolver._nodes
        assert "dep1" in resolver._nodes
        assert "plugin1" in resolver._nodes["dep1"].required_by

    def test_add_duplicate_plugin(self, resolver):
        """Test adding a duplicate plugin raises error."""
        resolver.add_plugin("plugin1", "1.0.0", [])
        with pytest.raises(ValueError, match="Plugin plugin1 already exists"):
            resolver.add_plugin("plugin1", "2.0.0", [])

    def test_resolve_simple(self, resolver):
        """Test resolving simple dependencies."""
        resolver.add_plugin("plugin1", "1.0.0", ["dep1"])
        resolver.add_plugin("dep1", "1.0.0", [])
        order = resolver.resolve()
        assert order == ["dep1", "plugin1"]

    def test_resolve_complex(self, resolver):
        """Test resolving complex dependencies."""
        resolver.add_plugin("plugin1", "1.0.0", ["dep1", "dep2"])
        resolver.add_plugin("dep1", "1.0.0", ["dep3"])
        resolver.add_plugin("dep2", "1.0.0", ["dep3"])
        resolver.add_plugin("dep3", "1.0.0", [])
        order = resolver.resolve()
        assert order[0] == "dep3"  # dep3 should be first
        assert order[-1] == "plugin1"  # plugin1 should be last

    def test_resolve_circular(self, resolver):
        """Test detecting circular dependencies."""
        resolver.add_plugin("plugin1", "1.0.0", ["plugin2"])
        resolver.add_plugin("plugin2", "1.0.0", ["plugin1"])
        with pytest.raises(ValueError, match="Circular dependency detected"):
            resolver.resolve()

    def test_resolve_missing_dependency(self, resolver):
        """Test resolving with missing dependency."""
        resolver.add_plugin("plugin1", "1.0.0", ["missing_dep"])
        with pytest.raises(ValueError, match="Missing dependency: missing_dep"):
            resolver.resolve()

# VersionManager Tests
class TestVersionManager:
    def test_check_compatibility_exact(self):
        """Test exact version compatibility."""
        assert VersionManager.check_compatibility("pkg", "==1.0.0", "1.0.0")
        assert not VersionManager.check_compatibility("pkg", "==1.0.0", "1.0.1")

    def test_check_compatibility_range(self):
        """Test version range compatibility."""
        assert VersionManager.check_compatibility("pkg", ">=1.0.0,<2.0.0", "1.5.0")
        assert not VersionManager.check_compatibility("pkg", ">=1.0.0,<2.0.0", "2.0.0")

    def test_check_compatibility_invalid_version(self):
        """Test compatibility with invalid version."""
        assert not VersionManager.check_compatibility("pkg", "==1.0.0", "invalid")

    def test_check_python_dependencies_satisfied(self):
        """Test checking satisfied Python dependencies."""
        deps = ["pytest>=1.0.0"]  # pytest should be installed in test environment
        satisfied, missing = VersionManager.check_python_dependencies(deps)
        assert satisfied
        assert not missing

    def test_check_python_dependencies_missing(self):
        """Test checking missing Python dependencies."""
        deps = ["nonexistent-package>=1.0.0"]
        satisfied, missing = VersionManager.check_python_dependencies(deps)
        assert not satisfied
        assert len(missing) == 1
        assert "nonexistent-package" in missing[0]

# PluginValidator Tests
class TestPluginValidator:
    @pytest.fixture
    def valid_metadata(self):
        return {
            'name': 'test_plugin',
            'version': '1.0.0',
            'description': 'Test plugin',
            'author': 'Test Author',
            'plugin_type': 'TEST',
            'entry_point': 'test.plugin'
        }

    @pytest.fixture
    def valid_schema(self):
        return {
            'type': 'object',
            'properties': {
                'param1': {
                    'type': 'string',
                    'description': 'Test parameter'
                }
            }
        }

    def test_validate_metadata_valid(self, valid_metadata):
        """Test validating valid metadata."""
        valid, errors = PluginValidator.validate_metadata(valid_metadata)
        assert valid
        assert not errors

    def test_validate_metadata_missing_fields(self, valid_metadata):
        """Test validating metadata with missing fields."""
        del valid_metadata['version']
        valid, errors = PluginValidator.validate_metadata(valid_metadata)
        assert not valid
        assert len(errors) == 1
        assert "Missing required fields" in errors[0]

    def test_validate_metadata_invalid_version(self, valid_metadata):
        """Test validating metadata with invalid version."""
        valid_metadata['version'] = 'invalid'
        valid, errors = PluginValidator.validate_metadata(valid_metadata)
        assert not valid
        assert "Invalid version format" in errors[0]

    def test_validate_metadata_invalid_dependencies(self, valid_metadata):
        """Test validating metadata with invalid dependencies."""
        valid_metadata['dependencies'] = ['invalid==1.0']
        valid, errors = PluginValidator.validate_metadata(valid_metadata)
        assert not valid
        assert "Invalid dependency format" in errors[0]

    def test_validate_config_schema_valid(self, valid_schema):
        """Test validating valid config schema."""
        valid, errors = PluginValidator.validate_config_schema(valid_schema)
        assert valid
        assert not errors

    def test_validate_config_schema_missing_fields(self, valid_schema):
        """Test validating schema with missing fields."""
        del valid_schema['type']
        valid, errors = PluginValidator.validate_config_schema(valid_schema)
        assert not valid
        assert "Missing required fields" in errors[0]

    def test_validate_config_schema_invalid_properties(self, valid_schema):
        """Test validating schema with invalid properties."""
        valid_schema['properties'] = "invalid"
        valid, errors = PluginValidator.validate_config_schema(valid_schema)
        assert not valid
        assert "Properties must be a dictionary" in errors[0] 