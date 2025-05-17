"""
Plugin Dependency Resolution Module for IDEF.

This module handles plugin dependency resolution and version compatibility checks.
"""

import pkg_resources # type: ignore
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from packaging.version import Version, parse
from packaging.requirements import Requirement
import importlib.metadata

@dataclass
class DependencyNode:
    """Node representing a plugin in the dependency graph."""
    name: str
    version: str
    dependencies: List[str]
    required_by: Set[str] = None
    
    def __post_init__(self):
        if self.required_by is None:
            self.required_by = set()

class DependencyResolver:
    """Handles plugin dependency resolution."""
    
    def __init__(self):
        self._nodes: Dict[str, DependencyNode] = {}
        self._sorted_plugins: List[str] = []
    
    def add_plugin(self, name: str, version: str, dependencies: List[str]):
        """Add a plugin to the dependency graph."""
        if name in self._nodes:
            raise ValueError(f"Plugin {name} already exists")
            
        node = DependencyNode(name, version, dependencies)
        self._nodes[name] = node
        
        # Update required_by for dependencies
        for dep in dependencies:
            if dep not in self._nodes:
                # Add placeholder node for missing dependency
                self._nodes[dep] = DependencyNode(dep, "", [])
            self._nodes[dep].required_by.add(name)
    
    def resolve(self) -> List[str]:
        """Resolve dependencies and return plugins in load order."""
        if not self._sorted_plugins:
            self._topological_sort()
        return self._sorted_plugins
    
    def _topological_sort(self):
        """Sort plugins based on dependencies."""
        visited = set()
        temp = set()
        
        def visit(name: str):
            if name in temp:
                cycle = " -> ".join(temp) + " -> " + name
                raise ValueError(f"Circular dependency detected: {cycle}")
            if name in visited:
                return
                
            temp.add(name)
            node = self._nodes[name]
            
            for dep in node.dependencies:
                if dep not in self._nodes:
                    raise ValueError(f"Missing dependency: {dep}")
                visit(dep)
                
            temp.remove(name)
            visited.add(name)
            self._sorted_plugins.insert(0, name)
        
        for name in self._nodes:
            if name not in visited:
                visit(name)

class VersionManager:
    """Handles version compatibility checks."""
    
    @staticmethod
    def check_compatibility(plugin_name: str, required_version: str,
                          installed_version: str) -> bool:
        """Check if installed version is compatible with required version."""
        try:
            req = Requirement(f"{plugin_name}{required_version}")
            return parse(installed_version) in req.specifier
        except Exception:
            return False
    
    @staticmethod
    def check_python_dependencies(dependencies: List[str]) -> Tuple[bool, List[str]]:
        """Check if Python package dependencies are satisfied."""
        missing = []
        satisfied = True
        
        for dep in dependencies:
            try:
                req = Requirement(dep)
                pkg_name = req.name
                
                try:
                    installed_version = importlib.metadata.version(pkg_name)
                    if not req.specifier.contains(installed_version):
                        missing.append(f"{pkg_name} (required: {req.specifier}, "
                                    f"installed: {installed_version})")
                        satisfied = False
                except importlib.metadata.PackageNotFoundError:
                    missing.append(f"{pkg_name} (not installed)")
                    satisfied = False
                    
            except Exception as e:
                missing.append(f"{dep} (invalid requirement)")
                satisfied = False
        
        return satisfied, missing

class PluginValidator:
    """Validates plugin metadata and configuration."""
    
    @staticmethod
    def validate_metadata(metadata: Dict) -> Tuple[bool, List[str]]:
        """Validate plugin metadata."""
        errors = []
        required_fields = {'name', 'version', 'description', 'author',
                         'plugin_type', 'entry_point'}
        
        # Check required fields
        missing = required_fields - set(metadata.keys())
        if missing:
            errors.append(f"Missing required fields: {missing}")
        
        # Validate version format
        try:
            parse(metadata.get('version', ''))
        except Exception:
            errors.append("Invalid version format")
        
        # Validate dependencies
        if 'dependencies' in metadata:
            if not isinstance(metadata['dependencies'], list):
                errors.append("Dependencies must be a list")
            else:
                for dep in metadata['dependencies']:
                    try:
                        Requirement(dep)
                    except Exception:
                        errors.append(f"Invalid dependency format: {dep}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_config_schema(schema: Dict) -> Tuple[bool, List[str]]:
        """Validate configuration schema."""
        errors = []
        required_fields = {'type', 'properties'}
        
        # Check required fields
        missing = required_fields - set(schema.keys())
        if missing:
            errors.append(f"Missing required fields in schema: {missing}")
        
        # Validate properties
        if 'properties' in schema:
            if not isinstance(schema['properties'], dict):
                errors.append("Properties must be a dictionary")
            else:
                for prop_name, prop in schema['properties'].items():
                    if 'type' not in prop:
                        errors.append(f"Missing type for property: {prop_name}")
                    if 'description' not in prop:
                        errors.append(f"Missing description for property: {prop_name}")
        
        return len(errors) == 0, errors 