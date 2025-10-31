"""
CineBrain Cache Manager
Intelligent caching for details module
"""

import json
import logging
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

from .errors import CacheError

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching for the details module"""
    
    def __init__(self, cache_backend=None):
        self.cache = cache_backend
        self.default_timeout = 1800  # 30 minutes
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        logger.info(f"Cache Manager initialized with backend: {type(cache_backend).__name__ if cache_backend else 'None'}")
    
    def is_available(self) -> bool:
        """Check if cache is available"""
        return self.cache is not None
    
    def generate_details_cache_key(self, slug: str, user_id: Optional[int] = None) -> str:
        """Generate cache key for content details"""
        try:
            base_key = f"details:content:{slug}"
            if user_id:
                base_key += f":user:{user_id}"
            return base_key
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"details:content:{slug}"
    
    def generate_person_cache_key(self, slug: str) -> str:
        """Generate cache key for person details"""
        return f"details:person:{slug}"
    
    def generate_trailer_cache_key(self, content_id: int) -> str:
        """Generate cache key for trailer data"""
        return f"details:trailer:{content_id}"
    
    def generate_reviews_cache_key(self, slug: str, page: int, limit: int, sort_by: str) -> str:
        """Generate cache key for reviews"""
        params_hash = hashlib.md5(f"{page}:{limit}:{sort_by}".encode()).hexdigest()[:8]
        return f"details:reviews:{slug}:{params_hash}"
    
    def generate_similar_cache_key(self, content_id: int, algorithm: str, limit: int) -> str:
        """Generate cache key for similar content"""
        return f"details:similar:{content_id}:{algorithm}:{limit}"
    
    def get_cached_details(self, cache_key: str) -> Optional[Dict]:
        """Get cached content details"""
        try:
            if not self.cache:
                return None
            
            cached_data = self.cache.get(cache_key)
            if cached_data:
                self.cache_stats['hits'] += 1
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_data
            else:
                self.cache_stats['misses'] += 1
                logger.debug(f"Cache miss for key: {cache_key}")
                return None
                
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"Cache get error for key {cache_key}: {e}")
            return None
    
    def cache_details(self, cache_key: str, data: Dict, timeout: Optional[int] = None) -> bool:
        """Cache content details"""
        try:
            if not self.cache or not data:
                return False
            
            cache_timeout = timeout or self.default_timeout
            
            # Add metadata
            cached_data = {
                'data': data,
                'cached_at': datetime.utcnow().isoformat(),
                'cache_version': '2.0'
            }
            
            success = self.cache.set(cache_key, cached_data, timeout=cache_timeout)
            if success:
                self.cache_stats['sets'] += 1
                logger.debug(f"Cached data for key: {cache_key}")
            
            return success
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"Cache set error for key {cache_key}: {e}")
            return False
    
    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get any cached data"""
        try:
            if not self.cache:
                return None
            
            data = self.cache.get(cache_key)
            if data:
                self.cache_stats['hits'] += 1
                # If it's our wrapped format, extract the data
                if isinstance(data, dict) and 'data' in data and 'cached_at' in data:
                    return data['data']
                return data
            else:
                self.cache_stats['misses'] += 1
                return None
                
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"Cache get error: {e}")
            return None
    
    def cache_data(self, cache_key: str, data: Any, timeout: Optional[int] = None) -> bool:
        """Cache any data"""
        try:
            if not self.cache:
                return False
            
            cache_timeout = timeout or self.default_timeout
            
            wrapped_data = {
                'data': data,
                'cached_at': datetime.utcnow().isoformat(),
                'cache_version': '2.0'
            }
            
            success = self.cache.set(cache_key, wrapped_data, timeout=cache_timeout)
            if success:
                self.cache_stats['sets'] += 1
            
            return success
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete_cached_data(self, cache_key: str) -> bool:
        """Delete cached data"""
        try:
            if not self.cache:
                return False
            
            success = self.cache.delete(cache_key)
            if success:
                self.cache_stats['deletes'] += 1
                logger.debug(f"Deleted cache key: {cache_key}")
            
            return success
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear_content_cache(self, slug: str) -> int:
        """Clear all cache for a specific content"""
        try:
            if not self.cache:
                return 0
            
            patterns = [
                f"details:content:{slug}*",
                f"details:reviews:{slug}*",
                f"details:trailer:*"  # May need content ID
            ]
            
            cleared_count = 0
            for pattern in patterns:
                try:
                    # This depends on cache backend supporting pattern deletion
                    if hasattr(self.cache, 'delete_pattern'):
                        cleared_count += self.cache.delete_pattern(pattern)
                    elif hasattr(self.cache, 'delete_many'):
                        # Alternative implementation
                        keys = self._get_keys_by_pattern(pattern)
                        cleared_count += self.cache.delete_many(keys)
                except Exception as e:
                    logger.warning(f"Error clearing cache pattern {pattern}: {e}")
            
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing content cache: {e}")
            return 0
    
    def clear_all_cache(self) -> int:
        """Clear all details cache"""
        try:
            if not self.cache:
                return 0
            
            # Get all details-related keys
            patterns = [
                "details:*"
            ]
            
            cleared_count = 0
            for pattern in patterns:
                try:
                    if hasattr(self.cache, 'delete_pattern'):
                        cleared_count += self.cache.delete_pattern(pattern)
                    elif hasattr(self.cache, 'clear'):
                        # Fallback: clear entire cache (not ideal)
                        self.cache.clear()
                        cleared_count = 1
                        break
                except Exception as e:
                    logger.warning(f"Error clearing cache pattern {pattern}: {e}")
            
            return cleared_count
            
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
            return 0
    
    def warm_cache(self, content_slugs: List[str]) -> Dict[str, bool]:
        """Pre-warm cache for frequently accessed content"""
        try:
            results = {}
            
            for slug in content_slugs:
                try:
                    # This would need to be implemented with the main details service
                    # For now, just mark as placeholder
                    results[slug] = False
                    logger.debug(f"Cache warming placeholder for: {slug}")
                except Exception as e:
                    logger.error(f"Error warming cache for {slug}: {e}")
                    results[slug] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cache warming: {e}")
            return {}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            stats = {
                'hits': self.cache_stats['hits'],
                'misses': self.cache_stats['misses'],
                'hit_rate': round(hit_rate, 2),
                'sets': self.cache_stats['sets'],
                'deletes': self.cache_stats['deletes'],
                'errors': self.cache_stats['errors'],
                'total_requests': total_requests,
                'backend_available': self.is_available()
            }
            
            # Add backend-specific stats if available
            if self.cache and hasattr(self.cache, 'get_stats'):
                try:
                    backend_stats = self.cache.get_stats()
                    stats['backend_stats'] = backend_stats
                except Exception as e:
                    logger.warning(f"Error getting backend stats: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    def reset_stats(self):
        """Reset cache statistics"""
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        logger.info("Cache statistics reset")
    
    def _get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Get cache keys matching pattern (backend-specific implementation needed)"""
        # This would need to be implemented based on the specific cache backend
        # For now, return empty list
        return []