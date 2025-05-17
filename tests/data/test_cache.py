"""
Tests for the caching module.
"""

import os
import time
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from idef.data.cache import (
    CacheResult, MemoryCache, DiskCache,
    CacheManager, CachePolicy
)

@pytest.fixture
def sample_data():
    """Create sample data for caching."""
    np.random.seed(42)
    return {
        'small': pd.DataFrame(np.random.randn(100, 3)),
        'medium': pd.DataFrame(np.random.randn(10000, 10)),
        'large': pd.DataFrame(np.random.randn(100000, 5))
    }

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / '.idef' / 'cache'
    cache_dir.mkdir(parents=True)
    return cache_dir

@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create a CacheManager instance."""
    return CacheManager(
        cache_dir=str(temp_cache_dir),
        max_memory_size=1024*1024*10,  # 10MB
        max_disk_size=1024*1024*100    # 100MB
    )

def test_cache_result():
    """Test CacheResult dataclass."""
    result = CacheResult(
        key='test_key',
        data=pd.DataFrame({'A': [1, 2, 3]}),
        metadata={'size': 1000, 'type': 'pandas'},
        created_at=time.time(),
        ttl=3600
    )
    assert result.key == 'test_key'
    assert isinstance(result.data, pd.DataFrame)
    assert result.metadata['size'] == 1000
    assert result.ttl == 3600

def test_memory_cache(sample_data):
    """Test memory-based caching."""
    cache = MemoryCache(max_size=1024*1024*10)  # 10MB
    
    # Test storing and retrieving
    cache.put('small', sample_data['small'], ttl=60)
    result = cache.get('small')
    assert result is not None
    pd.testing.assert_frame_equal(result.data, sample_data['small'])
    
    # Test size limit
    cache.put('large', sample_data['large'])
    assert cache.get('small') is None  # Should be evicted
    
    # Test TTL expiration
    cache.put('medium', sample_data['medium'], ttl=1)
    time.sleep(1.1)
    assert cache.get('medium') is None
    
    # Test explicit removal
    cache.put('small', sample_data['small'])
    cache.remove('small')
    assert cache.get('small') is None

def test_disk_cache(sample_data, temp_cache_dir):
    """Test disk-based caching."""
    cache = DiskCache(
        cache_dir=str(temp_cache_dir),
        max_size=1024*1024*100  # 100MB
    )
    
    # Test storing and retrieving
    cache.put('medium', sample_data['medium'], ttl=60)
    result = cache.get('medium')
    assert result is not None
    pd.testing.assert_frame_equal(result.data, sample_data['medium'])
    
    # Test file creation
    cache_file = temp_cache_dir / 'medium.cache'
    assert cache_file.exists()
    
    # Test size limit
    for i in range(5):
        cache.put(f'large_{i}', sample_data['large'])
    assert cache.get('medium') is None  # Should be evicted
    
    # Test TTL expiration
    cache.put('small', sample_data['small'], ttl=1)
    time.sleep(1.1)
    assert cache.get('small') is None
    
    # Test cache persistence
    cache.put('persist_test', sample_data['small'])
    new_cache = DiskCache(str(temp_cache_dir), max_size=1024*1024*100)
    result = new_cache.get('persist_test')
    assert result is not None

def test_cache_manager(cache_manager, sample_data):
    """Test CacheManager functionality."""
    # Test automatic cache type selection
    cache_manager.put('small', sample_data['small'])  # Should use memory
    assert isinstance(cache_manager.get('small'), CacheResult)
    
    cache_manager.put('large', sample_data['large'])  # Should use disk
    assert isinstance(cache_manager.get('large'), CacheResult)
    
    # Test cache policy
    cache_manager.set_policy(CachePolicy.MEMORY_ONLY)
    with pytest.raises(ValueError):
        cache_manager.put('large', sample_data['large'])
    
    cache_manager.set_policy(CachePolicy.DISK_ONLY)
    cache_manager.put('small', sample_data['small'])
    assert cache_manager.get('small').metadata['storage'] == 'disk'

def test_cache_cleanup(cache_manager, sample_data):
    """Test cache cleanup operations."""
    # Fill cache
    for i in range(5):
        cache_manager.put(f'data_{i}', sample_data['medium'])
    
    # Test size-based cleanup
    cache_manager.cleanup(max_age=None)
    assert len(cache_manager.list_keys()) < 5
    
    # Test age-based cleanup
    cache_manager.put('old_data', sample_data['small'], ttl=1)
    time.sleep(1.1)
    cache_manager.cleanup(max_age=1)
    assert 'old_data' not in cache_manager.list_keys()

def test_cache_statistics(cache_manager, sample_data):
    """Test cache statistics tracking."""
    # Generate some cache activity
    cache_manager.put('test1', sample_data['small'])
    cache_manager.get('test1')
    cache_manager.put('test2', sample_data['medium'])
    cache_manager.get('nonexistent')
    
    stats = cache_manager.get_statistics()
    assert stats['hits'] == 1
    assert stats['misses'] == 1
    assert stats['memory_usage'] > 0
    assert stats['disk_usage'] > 0

def test_cache_serialization(cache_manager, sample_data):
    """Test cache serialization methods."""
    # Test different data types
    test_data = {
        'df': sample_data['small'],
        'series': pd.Series(np.random.randn(100)),
        'array': np.random.randn(100, 3),
        'dict': {'a': 1, 'b': [1, 2, 3]},
        'list': list(range(100))
    }
    
    for key, data in test_data.items():
        cache_manager.put(key, data)
        result = cache_manager.get(key)
        if isinstance(data, (pd.DataFrame, pd.Series)):
            pd.testing.assert_equal(result.data, data)
        elif isinstance(data, np.ndarray):
            np.testing.assert_array_equal(result.data, data)
        else:
            assert result.data == data 