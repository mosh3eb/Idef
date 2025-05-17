"""
Cache Module for IDEF.

This module provides caching capabilities for optimizing data access and storage.
"""

import os
import time
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from dataclasses import dataclass
from enum import Enum, auto

@dataclass
class CacheResult:
    """Container for cached data."""
    key: str
    data: Any
    metadata: Dict
    created_at: float
    ttl: Optional[int] = None

class CachePolicy(Enum):
    """Cache storage policies."""
    MEMORY_ONLY = auto()
    DISK_ONLY = auto()
    MEMORY_AND_DISK = auto()

class MemoryCache:
    """Memory-based cache implementation."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._cache: Dict[str, CacheResult] = {}
        self._size = 0
    
    def get(self, key: str) -> Optional[CacheResult]:
        """Get item from cache."""
        if key not in self._cache:
            return None
            
        result = self._cache[key]
        
        # Check TTL
        if result.ttl is not None:
            if time.time() - result.created_at > result.ttl:
                self.remove(key)
                return None
        
        return result
    
    def put(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Store item in cache."""
        # Calculate size
        size = self._calculate_size(data)
        
        # Check if item would exceed max size
        if size > self.max_size:
            return False
        
        # Make space if needed
        while self._size + size > self.max_size and self._cache:
            self._evict_oldest()
        
        # Store item
        result = CacheResult(
            key=key,
            data=data,
            metadata={'size': size},
            created_at=time.time(),
            ttl=ttl
        )
        
        self._cache[key] = result
        self._size += size
        return True
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        if key in self._cache:
            self._size -= self._cache[key].metadata['size']
            del self._cache[key]
            return True
        return False
    
    def clear(self):
        """Clear all items from cache."""
        self._cache.clear()
        self._size = 0
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate size of data in bytes."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.memory_usage(deep=True).sum()
        elif isinstance(data, np.ndarray):
            return data.nbytes
        else:
            return len(pickle.dumps(data))
    
    def _evict_oldest(self):
        """Evict the oldest item from cache."""
        if not self._cache:
            return
            
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        self.remove(oldest_key)

class DiskCache:
    """Disk-based cache implementation."""
    
    def __init__(self, cache_dir: str, max_size: int):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.cache_dir / 'metadata.json'
        self._load_metadata()
    
    def get(self, key: str) -> Optional[CacheResult]:
        """Get item from cache."""
        if key not in self._metadata:
            return None
            
        metadata = self._metadata[key]
        
        # Check TTL
        if metadata['ttl'] is not None:
            if time.time() - metadata['created_at'] > metadata['ttl']:
                self.remove(key)
                return None
        
        # Load data
        try:
            with open(self.cache_dir / f"{key}.cache", 'rb') as f:
                data = pickle.load(f)
            
            return CacheResult(
                key=key,
                data=data,
                metadata=metadata,
                created_at=metadata['created_at'],
                ttl=metadata['ttl']
            )
        except:
            self.remove(key)
            return None
    
    def put(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Store item in cache."""
        # Calculate size
        size = self._calculate_size(data)
        
        # Check if item would exceed max size
        if size > self.max_size:
            return False
        
        # Make space if needed
        while self._get_total_size() + size > self.max_size and self._metadata:
            self._evict_oldest()
        
        # Store data
        try:
            with open(self.cache_dir / f"{key}.cache", 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            self._metadata[key] = {
                'size': size,
                'created_at': time.time(),
                'ttl': ttl,
                'storage': 'disk'
            }
            self._save_metadata()
            return True
        except:
            return False
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        if key in self._metadata:
            try:
                (self.cache_dir / f"{key}.cache").unlink()
                del self._metadata[key]
                self._save_metadata()
                return True
            except:
                pass
        return False
    
    def clear(self):
        """Clear all items from cache."""
        for key in list(self._metadata.keys()):
            self.remove(key)
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate size of data in bytes."""
        return len(pickle.dumps(data))
    
    def _get_total_size(self) -> int:
        """Get total size of cached items."""
        return sum(meta['size'] for meta in self._metadata.values())
    
    def _evict_oldest(self):
        """Evict the oldest item from cache."""
        if not self._metadata:
            return
            
        oldest_key = min(
            self._metadata.keys(),
            key=lambda k: self._metadata[k]['created_at']
        )
        self.remove(oldest_key)
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self._metadata_file.exists():
            with open(self._metadata_file, 'r') as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self._metadata_file, 'w') as f:
            json.dump(self._metadata, f)

class CacheManager:
    """High-level cache management."""
    
    def __init__(self, cache_dir: str,
                max_memory_size: int,
                max_disk_size: int):
        self.memory_cache = MemoryCache(max_memory_size)
        self.disk_cache = DiskCache(cache_dir, max_disk_size)
        self.policy = CachePolicy.MEMORY_AND_DISK
        self._stats = {'hits': 0, 'misses': 0}
    
    def get(self, key: str) -> Optional[CacheResult]:
        """Get item from cache."""
        # Try memory first
        if self.policy != CachePolicy.DISK_ONLY:
            result = self.memory_cache.get(key)
            if result is not None:
                self._stats['hits'] += 1
                return result
        
        # Try disk
        if self.policy != CachePolicy.MEMORY_ONLY:
            result = self.disk_cache.get(key)
            if result is not None:
                self._stats['hits'] += 1
                # Cache in memory if policy allows
                if self.policy == CachePolicy.MEMORY_AND_DISK:
                    self.memory_cache.put(key, result.data, result.ttl)
                return result
        
        self._stats['misses'] += 1
        return None
    
    def put(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Store item in cache."""
        size = len(pickle.dumps(data))
        
        if self.policy == CachePolicy.MEMORY_ONLY:
            return self.memory_cache.put(key, data, ttl)
        elif self.policy == CachePolicy.DISK_ONLY:
            return self.disk_cache.put(key, data, ttl)
        else:  # MEMORY_AND_DISK
            # Try memory first
            memory_success = self.memory_cache.put(key, data, ttl)
            # Always try disk
            disk_success = self.disk_cache.put(key, data, ttl)
            return memory_success or disk_success
    
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        memory_success = self.memory_cache.remove(key)
        disk_success = self.disk_cache.remove(key)
        return memory_success or disk_success
    
    def clear(self):
        """Clear all caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()
    
    def set_policy(self, policy: CachePolicy):
        """Set cache storage policy."""
        self.policy = policy
    
    def cleanup(self, max_age: Optional[float] = None):
        """Clean up old or oversized cache entries."""
        if max_age is not None:
            cutoff = time.time() - max_age
            for key in list(self.list_keys()):
                result = self.get(key)
                if result and result.created_at < cutoff:
                    self.remove(key)
    
    def list_keys(self) -> List[str]:
        """List all cached keys."""
        memory_keys = set(self.memory_cache._cache.keys())
        disk_keys = set(self.disk_cache._metadata.keys())
        return list(memory_keys | disk_keys)
    
    def get_statistics(self) -> Dict:
        """Get cache statistics."""
        return {
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'memory_usage': self.memory_cache._size,
            'disk_usage': self.disk_cache._get_total_size()
        }
