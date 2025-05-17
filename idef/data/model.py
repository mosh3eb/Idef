"""
Data Model Module for IDEF.

This module provides data modeling and validation capabilities.
"""

import json
import xarray as xr
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Type, Set, Union, Optional, Tuple
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path

@dataclass
class Field:
    """Representation of a data field."""
    name: str
    type: Type
    required: bool = True
    constraints: Dict[str, Any] = None
    
    def validate(self, value: Any) -> bool:
        """Validate a value against this field's specifications."""
        if value is None:
            return not self.required
            
        # Check type
        try:
            if isinstance(value, self.type):
                pass
            else:
                value = self.type(value)
        except:
            return False
        
        # Check constraints
        if self.constraints:
            if 'min' in self.constraints and value < self.constraints['min']:
                return False
            if 'max' in self.constraints and value > self.constraints['max']:
                return False
        
        return True

class Schema:
    """Base class for data schemas."""
    
    def __init__(self):
        self._fields = {
            field.name: Field(
                name=field.name,
                type=field.type,
                required=True
            )
            for field in fields(self.__class__)
        }
    
    def get_field_type(self, field_name: str) -> Type:
        """Get the type of a field."""
        return self._fields[field_name].type
    
    def get_required_fields(self) -> Set[str]:
        """Get names of required fields."""
        return {name for name, field in self._fields.items() 
                if field.required}
    
    def get_optional_fields(self) -> Set[str]:
        """Get names of optional fields."""
        return {name for name, field in self._fields.items() 
                if not field.required}
    
    def validate_field(self, field_name: str, value: Any) -> bool:
        """Validate a single field value."""
        if field_name not in self._fields:
            return False
        return self._fields[field_name].validate(value)

class Validator:
    """Collection of validation functions."""
    
    @staticmethod
    def range(field: str, min_val: float, max_val: float):
        """Create a range validator."""
        def validate(data: pd.DataFrame) -> bool:
            values = data[field]
            if not all(min_val <= v <= max_val for v in values):
                raise ValueError(f"Values in {field} must be between {min_val} and {max_val}")
            return True
        return validate
    
    @staticmethod
    def not_null(fields: List[str]):
        """Create a not-null validator."""
        def validate(data: pd.DataFrame) -> bool:
            for field in fields:
                if data[field].isnull().any():
                    raise ValueError(f"Field {field} cannot contain null values")
            return True
        return validate
    
    @staticmethod
    def unique(field: str):
        """Create a uniqueness validator."""
        def validate(data: pd.DataFrame) -> bool:
            if not data[field].is_unique:
                raise ValueError(f"Field {field} must contain unique values")
            return True
        return validate

class DataTransformer:
    """Collection of data transformation functions."""
    
    @staticmethod
    def apply(field: str, func: callable):
        """Create a field transformation."""
        def transform(data: pd.DataFrame) -> pd.DataFrame:
            result = data.copy()
            result[field] = result[field].apply(func)
            return result
        return transform
    
    @staticmethod
    def rename(mapping: Dict[str, str]):
        """Create a column renaming transformation."""
        def transform(data: pd.DataFrame) -> pd.DataFrame:
            return data.rename(columns=mapping)
        return transform

class DataModel:
    """Main data model class."""
    
    def __init__(self, name: str, schema: Type[Schema],
                validators: List[callable] = None):
        self.name = name
        self.schema = schema()
        self.validators = validators or []
        self.transformers = []
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate data against the model."""
        # Check schema fields
        for field_name in self.schema.get_required_fields():
            if field_name not in data.columns:
                raise ValueError(f"Required field {field_name} missing")
            
            if not all(self.schema.validate_field(field_name, value)
                      for value in data[field_name]):
                raise ValueError(f"Invalid values in field {field_name}")
        
        # Run validators
        for validator in self.validators:
            validator(data)
        
        return True
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to data."""
        result = data.copy()
        for transformer in self.transformers:
            result = transformer(result)
        return result
    
    def add_transformer(self, transformer: callable):
        """Add a data transformer."""
        self.transformers.append(transformer)
    
    def save(self, path: Union[str, Path]):
        """Save model to file."""
        model_data = {
            'name': self.name,
            'schema': self.schema.__class__.__name__,
            'required_fields': list(self.schema.get_required_fields()),
            'optional_fields': list(self.schema.get_optional_fields())
        }
        with open(path, 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DataModel':
        """Load model from file."""
        with open(path, 'r') as f:
            model_data = json.load(f)
        
        # Create a simple schema class dynamically
        schema_fields = {
            field: Any for field in 
            model_data['required_fields'] + model_data['optional_fields']
        }
        schema_cls = type(
            model_data['schema'],
            (Schema,),
            schema_fields
        )
        
        return cls(model_data['name'], schema_cls)
    
    @classmethod
    def compose(cls, name: str, models: List['DataModel']) -> 'DataModel':
        """Compose multiple models into one."""
        # Combine validators
        validators = []
        for model in models:
            validators.extend(model.validators)
        
        # Create a combined schema
        schema_fields = {}
        for model in models:
            schema_fields.update({
                field: Any for field in
                model.schema.get_required_fields() |
                model.schema.get_optional_fields()
            })
        
        schema_cls = type(
            f"Composite{name}Schema",
            (Schema,),
            schema_fields
        )
        
        return cls(name, schema_cls, validators)

class ModelRegistry:
    """Registry for data models."""
    
    def __init__(self):
        self._models: Dict[str, DataModel] = {}
    
    def register(self, model: DataModel):
        """Register a model."""
        if model.name in self._models:
            raise ValueError(f"Model {model.name} already registered")
        self._models[model.name] = model
    
    def get(self, name: str) -> DataModel:
        """Get a registered model."""
        if name not in self._models:
            raise KeyError(f"Model {name} not found")
        return self._models[name]
    
    def list_models(self) -> List[str]:
        """List registered models."""
        return list(self._models.keys())

class Dataset:
    """Core data structure for multi-dimensional scientific datasets."""
    
    def __init__(self, data: Union[xr.Dataset, pd.DataFrame, np.ndarray, str], 
                 name: Optional[str] = None):
        """Initialize a Dataset object."""
        self._data = self._initialize_data(data)
        self._name = name or "Unnamed Dataset"
        self._metadata = {}
        self._visualization_hints = {}
    
    def _initialize_data(self, data: Union[xr.Dataset, pd.DataFrame, np.ndarray, str]) -> xr.Dataset:
        """Convert input data to xarray.Dataset format."""
        if isinstance(data, xr.Dataset):
            return data
        elif isinstance(data, pd.DataFrame):
            return xr.Dataset.from_dataframe(data)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return xr.Dataset({"values": ("x", data)})
            elif data.ndim == 2:
                return xr.Dataset({"values": (("y", "x"), data)})
            else:
                dims = [f"dim_{i}" for i in range(data.ndim)]
                return xr.Dataset({"values": (dims, data)})
        elif isinstance(data, str):
            try:
                return xr.open_dataset(data)
            except Exception as e:
                raise ValueError(f"Could not open file {data}: {str(e)}")
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    @property
    def data(self) -> xr.Dataset:
        """Get the underlying xarray Dataset."""
        return self._data
    
    @property
    def name(self) -> str:
        """Get the dataset name."""
        return self._name
    
    @name.setter
    def name(self, value: str):
        """Set the dataset name."""
        self._name = value
    
    @property
    def dimensions(self) -> List[str]:
        """Get the dimensions of the dataset."""
        return list(self._data.dims)
    
    @property
    def variables(self) -> List[str]:
        """Get the variables in the dataset."""
        return list(self._data.data_vars)
    
    @property
    def coordinates(self) -> List[str]:
        """Get the coordinates of the dataset."""
        return list(self._data.coords)
    
    @property
    def shape(self) -> Dict[str, int]:
        """Get the shape of each dimension."""
        return {dim: size for dim, size in self._data.sizes.items()}
    
    def get_variable(self, name: str) -> xr.DataArray:
        """Get a variable from the dataset."""
        if name not in self._data.data_vars:
            raise KeyError(f"Variable '{name}' not found in dataset")
        return self._data[name]
    
    def set_metadata(self, key: str, value: Any):
        """Set metadata for the dataset."""
        self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from the dataset."""
        return self._metadata.get(key, default)
    
    def set_visualization_hint(self, key: str, value: Any):
        """Set a visualization hint for the dataset."""
        self._visualization_hints[key] = value
    
    def get_visualization_hint(self, key: str, default: Any = None) -> Any:
        """Get a visualization hint for the dataset."""
        return self._visualization_hints.get(key, default)
    
    def select(self, **kwargs) -> 'Dataset':
        """Select a subset of the dataset."""
        selected_data = self._data.sel(**kwargs)
        result = Dataset(selected_data, name=f"{self._name} (Selection)")
        result._metadata = self._metadata.copy()
        result._visualization_hints = self._visualization_hints.copy()
        return result
    
    def transform(self, func, *args, **kwargs) -> 'Dataset':
        """Apply a transformation function to the dataset."""
        transformed_data = func(self._data, *args, **kwargs)
        if not isinstance(transformed_data, xr.Dataset):
            raise TypeError(f"Transformation function must return an xarray.Dataset, got {type(transformed_data)}")
        
        result = Dataset(transformed_data, name=f"{self._name} (Transformed)")
        result._metadata = self._metadata.copy()
        result._visualization_hints = self._visualization_hints.copy()
        return result
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        dims_str = ", ".join([f"{dim}: {size}" for dim, size in self.shape.items()])
        vars_str = ", ".join(self.variables)
        return f"Dataset('{self._name}', dimensions=({dims_str}), variables=[{vars_str}])"
