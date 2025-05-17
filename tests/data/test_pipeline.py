"""
Tests for the data pipeline module.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict
from idef.data.pipeline import (
    Pipeline, PipelineStage, DataTransformation,
    Aggregation, Filter, Join, PipelineExecutor
)

@pytest.fixture
def sample_data():
    """Create sample data for pipeline testing."""
    np.random.seed(42)
    return {
        'main': pd.DataFrame({
            'id': range(100),
            'value': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        }),
        'lookup': pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'weight': [1.0, 2.0, 3.0]
        })
    }

@pytest.fixture
def sample_pipeline():
    """Create a sample data pipeline."""
    return Pipeline(
        name='test_pipeline',
        stages=[
            Filter('filter_positive', lambda df: df['value'] > 0),
            DataTransformation('double_value', lambda df: df.assign(value=df['value'] * 2)),
            Aggregation('group_by_category', ['category'], {'value': 'mean'})
        ]
    )

def test_pipeline_stage():
    """Test pipeline stage functionality."""
    # Test transformation stage
    transform_stage = DataTransformation(
        'test_transform',
        lambda df: df.assign(new_col=df['value'] + 1)
    )
    assert transform_stage.name == 'test_transform'
    assert callable(transform_stage.operation)
    
    # Test with invalid operation
    with pytest.raises(ValueError):
        DataTransformation('invalid', None)

def test_filter_stage(sample_data):
    """Test filter stage functionality."""
    filter_stage = Filter(
        'positive_values',
        lambda df: df['value'] > 0
    )
    
    # Apply filter
    result = filter_stage.execute(sample_data['main'])
    assert all(result['value'] > 0)
    
    # Test with invalid condition
    invalid_filter = Filter('invalid', lambda df: 'invalid')
    with pytest.raises(Exception):
        invalid_filter.execute(sample_data['main'])

def test_transformation_stage(sample_data):
    """Test transformation stage functionality."""
    transform = DataTransformation(
        'scale_values',
        lambda df: df.assign(scaled=df['value'] / df['value'].max())
    )
    
    # Apply transformation
    result = transform.execute(sample_data['main'])
    assert 'scaled' in result.columns
    assert all(result['scaled'] <= 1)
    
    # Test with invalid transformation
    invalid_transform = DataTransformation('invalid', lambda df: None)
    with pytest.raises(Exception):
        invalid_transform.execute(sample_data['main'])

def test_aggregation_stage(sample_data):
    """Test aggregation stage functionality."""
    agg = Aggregation(
        'category_stats',
        groupby=['category'],
        aggs={'value': ['mean', 'std']}
    )
    
    # Apply aggregation
    result = agg.execute(sample_data['main'])
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == 'category'
    assert 'value_mean' in result.columns
    assert 'value_std' in result.columns

def test_join_stage(sample_data):
    """Test join stage functionality."""
    join = Join(
        'add_weights',
        right_df=sample_data['lookup'],
        on='category',
        how='left'
    )
    
    # Apply join
    result = join.execute(sample_data['main'])
    assert 'weight' in result.columns
    assert len(result) == len(sample_data['main'])
    
    # Test with invalid join
    invalid_join = Join('invalid', pd.DataFrame(), on='invalid')
    with pytest.raises(Exception):
        invalid_join.execute(sample_data['main'])

def test_pipeline_execution(sample_pipeline, sample_data):
    """Test pipeline execution."""
    # Execute pipeline
    result = sample_pipeline.execute(sample_data['main'])
    
    # Verify results
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == 'category'
    assert 'value' in result.columns
    assert all(result['value'] >= 0)  # Due to filter stage
    
    # Test execution with invalid data
    with pytest.raises(Exception):
        sample_pipeline.execute(None)

def test_pipeline_executor(sample_data):
    """Test pipeline executor functionality."""
    executor = PipelineExecutor()
    
    # Create multiple pipelines
    pipeline1 = Pipeline('pipeline1', [
        Filter('filter_a', lambda df: df['category'] == 'A')
    ])
    pipeline2 = Pipeline('pipeline2', [
        Filter('filter_b', lambda df: df['category'] == 'B')
    ])
    
    # Register pipelines
    executor.register_pipeline(pipeline1)
    executor.register_pipeline(pipeline2)
    
    # Execute specific pipeline
    result1 = executor.execute('pipeline1', sample_data['main'])
    assert all(result1['category'] == 'A')
    
    # Execute all pipelines
    results = executor.execute_all(sample_data['main'])
    assert 'pipeline1' in results
    assert 'pipeline2' in results

def test_pipeline_validation():
    """Test pipeline validation."""
    # Test with invalid stage
    with pytest.raises(ValueError):
        Pipeline('invalid', stages=[None])
    
    # Test with duplicate stage names
    with pytest.raises(ValueError):
        Pipeline('duplicate', stages=[
            Filter('stage1', lambda df: df['value'] > 0),
            Filter('stage1', lambda df: df['value'] < 0)
        ])

def test_pipeline_monitoring(sample_pipeline, sample_data):
    """Test pipeline monitoring functionality."""
    # Add monitoring
    sample_pipeline.enable_monitoring()
    
    # Execute pipeline
    result = sample_pipeline.execute(sample_data['main'])
    
    # Check metrics
    metrics = sample_pipeline.get_metrics()
    assert 'execution_time' in metrics
    assert 'rows_processed' in metrics
    assert 'memory_usage' in metrics
    
    # Check stage metrics
    stage_metrics = sample_pipeline.get_stage_metrics()
    assert len(stage_metrics) == len(sample_pipeline.stages)
    for stage in sample_pipeline.stages:
        assert stage.name in stage_metrics

def test_pipeline_error_handling(sample_data):
    """Test pipeline error handling."""
    def failing_operation(df):
        raise ValueError("Intentional failure")
    
    pipeline = Pipeline('error_test', stages=[
        DataTransformation('will_fail', failing_operation)
    ])
    
    # Test with error handling disabled
    with pytest.raises(ValueError):
        pipeline.execute(sample_data['main'])
    
    # Test with error handling enabled
    pipeline.enable_error_handling(
        on_error='continue',
        error_handler=lambda e: pd.DataFrame()
    )
    result = pipeline.execute(sample_data['main'])
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_pipeline_branching(sample_data):
    """Test pipeline branching functionality."""
    # Create branching pipeline
    pipeline = Pipeline('branching_test', stages=[
        DataTransformation('common', lambda df: df.assign(common=1)),
        {
            'branch1': [
                Filter('filter_a', lambda df: df['category'] == 'A')
            ],
            'branch2': [
                Filter('filter_b', lambda df: df['category'] == 'B')
            ]
        }
    ])
    
    # Execute pipeline
    results = pipeline.execute(sample_data['main'])
    
    # Check results
    assert 'branch1' in results
    assert 'branch2' in results
    assert all(results['branch1']['category'] == 'A')
    assert all(results['branch2']['category'] == 'B')
    assert 'common' in results['branch1'].columns
    assert 'common' in results['branch2'].columns 