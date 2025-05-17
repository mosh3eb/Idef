"""
Tests for the data model module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass
from idef.data.model import (
    DataModel, Schema, Field, Validator,
    DataTransformer, ModelRegistry
)

@dataclass
class SampleSchema(Schema):
    """Sample schema for testing."""
    id: int
    name: str
    value: float
    timestamp: datetime
    tags: List[str]
    metadata: Dict[str, Any]

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'id': range(100),
        'name': [f'item_{i}' for i in range(100)],
        'value': np.random.randn(100),
        'timestamp': [datetime.now() for _ in range(100)],
        'tags': [['tag1', 'tag2'] for _ in range(100)],
        'metadata': [{'key': 'value'} for _ in range(100)]
    })

@pytest.fixture
def sample_model():
    """Create a sample data model."""
    return DataModel(
        name='test_model',
        schema=SampleSchema,
        validators=[
            Validator.range('value', -10, 10),
            Validator.not_null(['id', 'name']),
            Validator.unique('id')
        ]
    )

def test_schema_validation():
    """Test schema validation."""
    schema = SampleSchema()
    
    # Test field types
    assert schema.get_field_type('id') == int
    assert schema.get_field_type('tags') == List[str]
    
    # Test required fields
    assert schema.get_required_fields() == {'id', 'name', 'value', 'timestamp'}
    
    # Test optional fields
    assert schema.get_optional_fields() == {'tags', 'metadata'}
    
    # Test field validation
    assert schema.validate_field('id', 1)
    assert not schema.validate_field('id', 'invalid')
    assert schema.validate_field('tags', ['tag1', 'tag2'])
    assert not schema.validate_field('tags', 'invalid')

def test_data_model_validation(sample_model, sample_data):
    """Test data model validation."""
    # Test valid data
    assert sample_model.validate(sample_data)
    
    # Test invalid data types
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'id'] = 'invalid'
    with pytest.raises(ValueError):
        sample_model.validate(invalid_data)
    
    # Test range validation
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'value'] = 100
    with pytest.raises(ValueError):
        sample_model.validate(invalid_data)
    
    # Test null validation
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'name'] = None
    with pytest.raises(ValueError):
        sample_model.validate(invalid_data)
    
    # Test uniqueness validation
    invalid_data = sample_data.copy()
    invalid_data.loc[1, 'id'] = invalid_data.loc[0, 'id']
    with pytest.raises(ValueError):
        sample_model.validate(invalid_data)

def test_data_transformation(sample_model, sample_data):
    """Test data transformations."""
    # Add transformations to model
    sample_model.add_transformer(
        DataTransformer.apply('value', lambda x: x * 2)
    )
    sample_model.add_transformer(
        DataTransformer.rename({'name': 'item_name'})
    )
    
    # Apply transformations
    transformed_data = sample_model.transform(sample_data)
    
    # Check transformations
    assert all(transformed_data['value'] == sample_data['value'] * 2)
    assert 'item_name' in transformed_data.columns
    assert 'name' not in transformed_data.columns

def test_model_registry():
    """Test model registry functionality."""
    registry = ModelRegistry()
    
    # Register models
    model1 = DataModel('model1', SampleSchema)
    model2 = DataModel('model2', SampleSchema)
    
    registry.register(model1)
    registry.register(model2)
    
    # Test retrieval
    assert registry.get('model1') == model1
    assert registry.get('model2') == model2
    
    # Test listing
    assert set(registry.list_models()) == {'model1', 'model2'}
    
    # Test duplicate registration
    with pytest.raises(ValueError):
        registry.register(model1)
    
    # Test invalid model retrieval
    with pytest.raises(KeyError):
        registry.get('invalid')

def test_field_operations():
    """Test field operations."""
    field = Field('test_field', int, required=True)
    
    # Test basic properties
    assert field.name == 'test_field'
    assert field.type == int
    assert field.required
    
    # Test validation
    assert field.validate(42)
    assert not field.validate('invalid')
    assert not field.validate(None)
    
    # Test with constraints
    field_with_constraints = Field(
        'value',
        float,
        required=True,
        constraints={'min': 0, 'max': 1}
    )
    assert field_with_constraints.validate(0.5)
    assert not field_with_constraints.validate(2)

def test_model_serialization(sample_model, sample_data, tmp_path):
    """Test model serialization."""
    # Save model
    model_path = tmp_path / 'model.json'
    sample_model.save(model_path)
    
    # Load model
    loaded_model = DataModel.load(model_path)
    assert loaded_model.name == sample_model.name
    assert loaded_model.schema == sample_model.schema
    
    # Verify functionality
    assert loaded_model.validate(sample_data)

def test_model_composition(sample_data):
    """Test model composition."""
    # Create component models
    id_model = DataModel(
        'id_model',
        Schema(['id'], [Validator.unique('id')])
    )
    value_model = DataModel(
        'value_model',
        Schema(['value'], [Validator.range('value', -10, 10)])
    )
    
    # Compose models
    composite_model = DataModel.compose(
        'composite',
        [id_model, value_model]
    )
    
    # Test validation
    assert composite_model.validate(sample_data[['id', 'value']])
    
    # Test with invalid data
    invalid_data = sample_data[['id', 'value']].copy()
    invalid_data.loc[0, 'value'] = 100
    with pytest.raises(ValueError):
        composite_model.validate(invalid_data)

def test_model_versioning(sample_model, sample_data):
    """Test model versioning."""
    # Create new version
    v2_model = sample_model.create_version(
        '2.0',
        transformers=[
            DataTransformer.apply('value', lambda x: x + 1)
        ]
    )
    
    # Test version tracking
    assert v2_model.version == '2.0'
    assert v2_model.previous_version == sample_model.version
    
    # Test data compatibility
    v1_data = sample_model.transform(sample_data)
    v2_data = v2_model.transform(sample_data)
    assert all(v2_data['value'] == v1_data['value'] + 1) 