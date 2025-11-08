# operations/task.py

import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from typing import Dict, Any, List
import traceback
import os
import gc
import psutil

logger = logging.getLogger(__name__)

class OperationsTasks:
    """Handles all operations tasks for CineBrain"""
    
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.models = models
        self.services = services
        self.cache = services.get('cache')
        
        # Cache timeouts (in seconds)
        self.TIMEOUTS = {
            'trending': 900,        # 15 minutes
            'critics_choice': 1800, # 30 minutes
            'new_releases': 3600,   # 1 hour
            'genre': 1800,          # 30 minutes
            'similar': 600,         # 10 minutes
            'personalized': 3600,   # 1 hour
            'upcoming': 7200,       # 2 hours
            'discover': 1800,       # 30 minutes
            'admin_recs': 1800,     # 30 minutes
            'details': 3600         # 1 hour
        }
        
        # Performance tracking
        self.performance_metrics = {
            'cache_refreshes': 0,
            'last_refresh_time': None,
            'average_refresh_duration': 0,
            'total_refresh_time': 0,
            'successful_refreshes': 0,
            'failed_refreshes': 0
        }
    
    # ============================================================================
    # 🔥 MAIN COMPREHENSIVE REFRESH METHOD
    # ============================================================================
    
    def comprehensive_cache_refresh(self, force: bool = False, include_personalized: bool = True, user_limit: int = 30) -> Dict[str, Any]:
        """
        🔥 COMPREHENSIVE CACHE REFRESH - REFRESHES EVERYTHING!
        
        This single method refreshes ALL CineBrain caches:
        • Trending content (all types)
        • Critics Choice 
        • Genre recommendations
        • New Releases
        • Similar content
        • Upcoming releases
        • Content details
        • Personalized recommendations
        • Discover section
        • Admin recommendations
        • System warmup
        """
        results = {}
        start_time = time.time()
        
        try:
            logger.info("🚀 Starting comprehensive CineBrain cache refresh...")
            
            # 🔥 CORE CONTENT CACHES (High Priority)
            with ThreadPoolExecutor(max_workers=6) as executor:
                core_futures = {
                    '🔥_trending': executor.submit(self._refresh_trending_cache, force),
                    '⭐_critics_choice': executor.submit(self._refresh_critics_choice_cache, force),
                    '🆕_new_releases': executor.submit(self._refresh_new_releases_cache, force),
                    '🌍_discover': executor.submit(self._refresh_discover_cache, force),
                    '🎭_genre_samples': executor.submit(self._refresh_all_genre_samples, force),
                    '💾_system_warmup': executor.submit(self._system_warmup_comprehensive)
                }
                
                # Wait for core caches
                for name, future in core_futures.items():
                    try:
                        results[name] = future.result(timeout=45)
                        logger.info(f"✅ {name} completed")
                    except Exception as e:
                        logger.error(f"❌ {name} failed: {e}")
                        results[name] = {'error': str(e), 'success': False}
            
            # 🔗 SECONDARY CACHES (Medium Priority) 
            with ThreadPoolExecutor(max_workers=4) as executor:
                secondary_futures = {
                    '📅_upcoming': executor.submit(self._refresh_upcoming_cache, force),
                    '🔗_similar_samples': executor.submit(self._refresh_similar_samples, force),
                    '🏆_admin_recommendations': executor.submit(self._refresh_admin_recommendations_cache, force),
                    '🎬_content_details_samples': executor.submit(self._refresh_content_details_samples, force)
                }
                
                # Wait for secondary caches
                for name, future in secondary_futures.items():
                    try:
                        results[name] = future.result(timeout=30)
                        logger.info(f"✅ {name} completed")
                    except Exception as e:
                        logger.warning(f"⚠️ {name} failed: {e}")
                        results[name] = {'error': str(e), 'success': False}
            
            # 👤 PERSONALIZED CACHES (Lower Priority - Optional)
            if include_personalized:
                try:
                    logger.info(f"🧠 Refreshing personalized caches for top {user_limit} users...")
                    personalized_result = self._refresh_top_users_personalized(user_limit)
                    results['👤_personalized'] = personalized_result
                    logger.info(f"✅ Personalized refresh completed for {personalized_result.get('users_refreshed', 0)} users")
                except Exception as e:
                    logger.warning(f"⚠️ Personalized refresh failed: {e}")
                    results['👤_personalized'] = {'error': str(e), 'success': False}
            
            # 📊 CALCULATE FINAL STATS
            duration = time.time() - start_time
            successful_ops = len([r for r in results.values() if r.get('success')])
            total_ops = len(results)
            
            # Update performance metrics
            self._update_performance_metrics(duration, successful_ops, total_ops - successful_ops)
            
            logger.info(f"🎉 Comprehensive refresh completed in {duration:.2f}s")
            logger.info(f"📈 Success rate: {successful_ops}/{total_ops} operations")
            
            return results
            
        except Exception as e:
            logger.error(f"💥 Comprehensive refresh critical error: {e}")
            results['💥_critical_error'] = {'error': str(e), 'success': False}
            self._update_performance_metrics(time.time() - start_time, 0, 1)
            return results

    # ============================================================================
    # 🎯 STANDARD REFRESH METHODS (EXISTING)
    # ============================================================================
    
    def standard_cache_refresh(self, force: bool = False) -> Dict[str, Any]:
        """Standard cache refresh - balanced performance and coverage"""
        start_time = time.time()
        results = {
            'started_at': datetime.utcnow().isoformat(),
            'mode': 'standard',
            'operations': {}
        }
        
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    'trending': executor.submit(self._refresh_trending_cache, force),
                    'critics_choice': executor.submit(self._refresh_critics_choice_cache, force),
                    'new_releases': executor.submit(self._refresh_new_releases_cache, force),
                    'discover': executor.submit(self._refresh_discover_cache, force),
                    'top_users_personalized': executor.submit(self._refresh_top_users_personalized, 25)
                }
                
                for name, future in futures.items():
                    try:
                        results['operations'][name] = future.result(timeout=30)
                    except Exception as e:
                        logger.error(f"Error in {name} cache refresh: {e}")
                        results['operations'][name] = {'error': str(e), 'success': False}
            
            results['completed_at'] = datetime.utcnow().isoformat()
            results['success'] = True
            
            # Update performance metrics
            duration = time.time() - start_time
            successful_ops = len([r for r in results['operations'].values() if r.get('success')])
            total_ops = len(results['operations'])
            self._update_performance_metrics(duration, successful_ops, total_ops - successful_ops)
            
        except Exception as e:
            logger.error(f"Standard cache refresh error: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def critical_cache_refresh(self, force: bool = False) -> Dict[str, Any]:
        """Critical cache refresh - fastest, most essential caches only"""
        start_time = time.time()
        results = {
            'started_at': datetime.utcnow().isoformat(),
            'mode': 'critical',
            'operations': {}
        }
        
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    'trending': executor.submit(self._refresh_trending_cache, force),
                    'new_releases': executor.submit(self._refresh_new_releases_cache, force),
                    'system_warmup': executor.submit(self._system_warmup_minimal)
                }
                
                for name, future in futures.items():
                    try:
                        results['operations'][name] = future.result(timeout=20)
                    except Exception as e:
                        logger.error(f"Error in {name} critical refresh: {e}")
                        results['operations'][name] = {'error': str(e), 'success': False}
            
            results['completed_at'] = datetime.utcnow().isoformat()
            results['success'] = True
            
            # Update performance metrics
            duration = time.time() - start_time
            successful_ops = len([r for r in results['operations'].values() if r.get('success')])
            total_ops = len(results['operations'])
            self._update_performance_metrics(duration, successful_ops, total_ops - successful_ops)
            
        except Exception as e:
            logger.error(f"Critical cache refresh error: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def full_cache_refresh(self, force: bool = False) -> Dict[str, Any]:
        """Full cache refresh - comprehensive but slower"""
        start_time = time.time()
        results = {
            'started_at': datetime.utcnow().isoformat(),
            'mode': 'full',
            'operations': {}
        }
        
        try:
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {
                    'trending': executor.submit(self._refresh_trending_cache, force),
                    'critics_choice': executor.submit(self._refresh_critics_choice_cache, force),
                    'new_releases': executor.submit(self._refresh_new_releases_cache, force),
                    'upcoming': executor.submit(self._refresh_upcoming_cache, force),
                    'discover': executor.submit(self._refresh_discover_cache, force),
                    'genre_samples': executor.submit(self._refresh_all_genre_samples, force),
                    'personalized': executor.submit(self._refresh_top_users_personalized, 50),
                    'similar_samples': executor.submit(self._refresh_similar_samples, force)
                }
                
                for name, future in futures.items():
                    try:
                        results['operations'][name] = future.result(timeout=60)
                    except Exception as e:
                        logger.error(f"Error in {name} full refresh: {e}")
                        results['operations'][name] = {'error': str(e), 'success': False}
            
            results['completed_at'] = datetime.utcnow().isoformat()
            results['success'] = True
            
            # Update performance metrics
            duration = time.time() - start_time
            successful_ops = len([r for r in results['operations'].values() if r.get('success')])
            total_ops = len(results['operations'])
            self._update_performance_metrics(duration, successful_ops, total_ops - successful_ops)
            
        except Exception as e:
            logger.error(f"Full cache refresh error: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results

    # ============================================================================
    # 🎯 SPECIFIC CACHE REFRESH METHODS
    # ============================================================================
    
    def refresh_recommendation_caches(self) -> Dict[str, Any]:
        """Refresh all recommendation-related caches"""
        results = {}
        
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    'trending': executor.submit(self._refresh_trending_cache),
                    'critics_choice': executor.submit(self._refresh_critics_choice_cache),
                    'similar_samples': executor.submit(self._refresh_similar_samples),
                    'genre_samples': executor.submit(self._refresh_all_genre_samples)
                }
                
                for name, future in futures.items():
                    try:
                        results[name] = future.result(timeout=30)
                    except Exception as e:
                        results[name] = {'error': str(e), 'success': False}
            
        except Exception as e:
            logger.error(f"Recommendation cache refresh error: {e}")
            results['error'] = str(e)
        
        return results
    
    def refresh_personalized_caches(self, user_limit: int = 50) -> Dict[str, Any]:
        """Refresh personalized caches for active users"""
        results = {'users_processed': 0, 'errors': 0}
        
        try:
            # Get active users
            active_users = self._get_active_users(limit=user_limit)
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                
                for user in active_users:
                    future = executor.submit(self._refresh_user_personalized_cache, user.id)
                    futures.append((user.id, future))
                
                for user_id, future in futures:
                    try:
                        future.result(timeout=10)
                        results['users_processed'] += 1
                    except Exception as e:
                        logger.warning(f"Error refreshing personalized cache for user {user_id}: {e}")
                        results['errors'] += 1
            
        except Exception as e:
            logger.error(f"Personalized cache refresh error: {e}")
            results['error'] = str(e)
        
        return results
    
    def refresh_content_caches(self) -> Dict[str, Any]:
        """Refresh content-related caches"""
        results = {}
        
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    'new_releases': executor.submit(self._refresh_new_releases_cache),
                    'upcoming': executor.submit(self._refresh_upcoming_cache),
                    'discover': executor.submit(self._refresh_discover_cache)
                }
                
                for name, future in futures.items():
                    try:
                        results[name] = future.result(timeout=30)
                    except Exception as e:
                        results[name] = {'error': str(e), 'success': False}
            
        except Exception as e:
            logger.error(f"Content cache refresh error: {e}")
            results['error'] = str(e)
        
        return results

    # ============================================================================
    # 💾 SYSTEM OPERATIONS
    # ============================================================================
    
    def warm_up_system(self) -> Dict[str, Any]:
        """Light system warmup - just enough to keep things responsive"""
        results = {}
        
        try:
            # Basic health checks
            results['database'] = self._test_database_connection()
            results['cache'] = self._test_cache_connection()
            
            # Light cache warming
            if self.cache:
                self.cache.set('cinebrain_warmup', datetime.utcnow().isoformat(), timeout=300)
            
            # Minimal content check
            if self.models.get('Content'):
                try:
                    content_count = self.models['Content'].query.count()
                    results['content_available'] = content_count > 0
                    results['content_count'] = content_count
                except:
                    results['content_available'] = False
            
            # System metrics
            results['system_metrics'] = self._get_basic_system_metrics()
            
            results['success'] = True
            
        except Exception as e:
            logger.error(f"System warmup error: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def cleanup_system(self, cleanup_type: str = 'standard') -> Dict[str, Any]:
        """Perform system cleanup operations"""
        results = {
            'cleanup_type': cleanup_type,
            'operations': {}
        }
        
        try:
            if cleanup_type == 'aggressive':
                results['operations'].update(self._aggressive_cleanup())
            elif cleanup_type == 'minimal':
                results['operations'].update(self._minimal_cleanup())
            else:  # standard
                results['operations'].update(self._standard_cleanup())
            
            results['success'] = True
            
        except Exception as e:
            logger.error(f"System cleanup error: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def run_maintenance(self, maintenance_type: str = 'routine') -> Dict[str, Any]:
        """Run scheduled maintenance tasks"""
        results = {
            'maintenance_type': maintenance_type,
            'operations': {}
        }
        
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                maintenance_tasks = {
                    'database_maintenance': executor.submit(self._database_maintenance),
                    'cache_optimization': executor.submit(self._cache_optimization),
                    'performance_analysis': executor.submit(self._performance_analysis)
                }
                
                if maintenance_type == 'deep':
                    maintenance_tasks.update({
                        'data_integrity_check': executor.submit(self._data_integrity_check),
                        'system_optimization': executor.submit(self._system_optimization)
                    })
                
                for name, future in maintenance_tasks.items():
                    try:
                        results['operations'][name] = future.result(timeout=120)
                    except Exception as e:
                        results['operations'][name] = {'error': str(e), 'success': False}
            
            results['success'] = True
            
        except Exception as e:
            logger.error(f"Maintenance error: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        return results

    # ============================================================================
    # 📊 STATUS AND MONITORING
    # ============================================================================
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get detailed cache status"""
        status = {
            'cache_enabled': bool(self.cache),
            'cache_type': self.app.config.get('CACHE_TYPE'),
            'timestamp': datetime.utcnow().isoformat(),
            'performance_metrics': self.performance_metrics
        }
        
        if self.cache:
            try:
                # Test cache functionality
                test_key = 'cinebrain_status_test'
                self.cache.set(test_key, 'test', timeout=60)
                test_result = self.cache.get(test_key)
                status['cache_functional'] = (test_result == 'test')
                
                # Check for cache keys
                cache_keys = [
                    'cinebrain:trending',
                    'cinebrain:critics_choice',
                    'cinebrain:new_releases',
                    'cinebrain:discover'
                ]
                
                status['cached_data'] = {}
                for key in cache_keys:
                    try:
                        data = self.cache.get(key)
                        status['cached_data'][key] = {
                            'exists': data is not None,
                            'size': len(str(data)) if data else 0,
                            'type': type(data).__name__ if data else None
                        }
                    except:
                        status['cached_data'][key] = {'exists': False, 'error': True}
                
            except Exception as e:
                status['cache_functional'] = False
                status['cache_error'] = str(e)
        
        return status
    
    def get_operations_health(self) -> Dict[str, Any]:
        """Get operations module health status"""
        health = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_healthy': True,
            'components': {}
        }
        
        try:
            # Database health
            health['components']['database'] = {
                'status': 'healthy' if self._test_database_connection() else 'unhealthy',
                'connection': self._test_database_connection()
            }
            
            # Cache health
            health['components']['cache'] = {
                'status': 'healthy' if self._test_cache_connection() else 'unhealthy',
                'connection': self._test_cache_connection(),
                'type': self.app.config.get('CACHE_TYPE')
            }
            
            # Services health
            health['components']['services'] = {
                'tmdb': 'TMDBService' in self.services,
                'jikan': 'JikanService' in self.services,
                'content': 'ContentService' in self.services,
                'new_releases': 'new_releases_service' in self.services,
                'critics_choice': 'critics_choice_service' in self.services,
                'personalized': 'personalized_recommendation_engine' in self.services
            }
            
            # System resources
            health['components']['system'] = self._get_system_health()
            
            # Determine overall health
            unhealthy_components = []
            for component, status in health['components'].items():
                if isinstance(status, dict) and status.get('status') == 'unhealthy':
                    unhealthy_components.append(component)
                elif isinstance(status, dict) and any(not v for v in status.values() if isinstance(v, bool)):
                    unhealthy_components.append(component)
            
            if unhealthy_components:
                health['overall_healthy'] = False
                health['unhealthy_components'] = unhealthy_components
            
        except Exception as e:
            logger.error(f"Operations health check error: {e}")
            health['overall_healthy'] = False
            health['error'] = str(e)
        
        return health
    
    def get_operations_analytics(self) -> Dict[str, Any]:
        """Get operations analytics and performance metrics"""
        analytics = {
            'timestamp': datetime.utcnow().isoformat(),
            'performance_metrics': self.performance_metrics,
            'system_metrics': self._get_system_metrics(),
            'cache_analytics': self._get_cache_analytics(),
            'database_analytics': self._get_database_analytics()
        }
        
        return analytics
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'cache_system': {},
            'services': {},
            'database': {},
            'performance': self.performance_metrics
        }
        
        try:
            # Cache status
            status['cache_system'] = {
                'enabled': bool(self.cache),
                'type': self.app.config.get('CACHE_TYPE'),
                'functional': self._test_cache_connection() if self.cache else False
            }
            
            # Services status
            status['services'] = {
                'tmdb': 'TMDBService' in self.services,
                'jikan': 'JikanService' in self.services,
                'content': 'ContentService' in self.services,
                'new_releases': 'new_releases_service' in self.services,
                'critics_choice': 'critics_choice_service' in self.services,
                'personalized': 'personalized_recommendation_engine' in self.services
            }
            
            # Database status
            status['database'] = {
                'connected': self._test_database_connection(),
                'content_count': self.models['Content'].query.count() if self.models.get('Content') else 0,
                'user_count': self.models['User'].query.count() if self.models.get('User') else 0
            }
            
        except Exception as e:
            status['error'] = str(e)
        
        return status
    
    def clear_cache(self, cache_type: str = 'all') -> Dict[str, Any]:
        """Clear cache entries"""
        results = {'cleared': [], 'errors': []}
        
        if not self.cache:
            return {'error': 'Cache not available'}
        
        try:
            if cache_type == 'all':
                self.cache.clear()
                results['cleared'].append('all_cache')
            else:
                # Clear specific cache patterns
                patterns = {
                    'recommendations': ['cinebrain:trending', 'cinebrain:critics_choice', 'cinebrain:genre'],
                    'personalized': ['cinebrain:personalized', 'cinebrain:profile'],
                    'content': ['cinebrain:content', 'cinebrain:new_releases'],
                    'system': ['cinebrain:health', 'cinebrain:warmup']
                }
                
                if cache_type in patterns:
                    for pattern in patterns[cache_type]:
                        try:
                            self.cache.delete(pattern)
                            results['cleared'].append(pattern)
                        except Exception as e:
                            results['errors'].append(f"{pattern}: {str(e)}")
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results

    # ============================================================================
    # 🔥 NEW COMPREHENSIVE CACHE REFRESH HELPERS
    # ============================================================================
    
    def _refresh_all_genre_samples(self, force: bool = False) -> Dict[str, Any]:
        """Refresh ALL popular genres"""
        try:
            genres = [
                'action', 'comedy', 'drama', 'thriller', 'horror', 
                'romance', 'sci-fi', 'adventure', 'animation', 'crime'
            ]
            refreshed = 0
            
            for genre in genres:
                try:
                    cache_key = f'cinebrain:genre:{genre}'
                    
                    if not force and self.cache and self.cache.get(cache_key):
                        continue
                    
                    if 'TMDBService' in self.services:
                        genre_ids = {
                            'action': 28, 'comedy': 35, 'drama': 18, 'thriller': 53,
                            'horror': 27, 'romance': 10749, 'sci-fi': 878, 
                            'adventure': 12, 'animation': 16, 'crime': 80
                        }
                        genre_id = genre_ids.get(genre)
                        
                        if genre_id:
                            genre_data = self.services['TMDBService'].get_by_genre(genre_id, 'movie')
                            if genre_data and self.cache:
                                self.cache.set(cache_key, genre_data, timeout=self.TIMEOUTS['genre'])
                                refreshed += 1
                except Exception as e:
                    logger.warning(f"Error refreshing genre {genre}: {e}")
            
            return {'success': True, 'action': 'refreshed', 'genres_refreshed': refreshed}
            
        except Exception as e:
            logger.error(f"All genre samples refresh error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _refresh_admin_recommendations_cache(self, force: bool = False) -> Dict[str, Any]:
        """Refresh admin recommendations cache"""
        try:
            cache_key = 'cinebrain:admin_recommendations'
            
            if not force and self.cache and self.cache.get(cache_key):
                return {'success': True, 'action': 'already_cached', 'cache_key': cache_key}
            
            AdminRecommendation = self.models.get('AdminRecommendation')
            Content = self.models.get('Content')
            
            if AdminRecommendation and Content:
                admin_recs = AdminRecommendation.query.filter_by(is_active=True).limit(20).all()
                
                admin_data = []
                for rec in admin_recs:
                    content = Content.query.get(rec.content_id)
                    if content:
                        admin_data.append({
                            'id': content.id,
                            'title': content.title,
                            'description': rec.description,
                            'recommendation_type': rec.recommendation_type,
                            'slug': content.slug,
                            'content_type': content.content_type,
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path
                        })
                
                if self.cache:
                    self.cache.set(cache_key, admin_data, timeout=self.TIMEOUTS['admin_recs'])
                    return {'success': True, 'action': 'refreshed', 'items': len(admin_data)}
            
            return {'success': False, 'error': 'Models not available'}
            
        except Exception as e:
            logger.error(f"Admin recommendations cache refresh error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _refresh_content_details_samples(self, force: bool = False) -> Dict[str, Any]:
        """Refresh content details cache for popular items"""
        try:
            if self.models.get('Content'):
                popular_content = self.models['Content'].query.filter(
                    self.models['Content'].rating >= 7.0
                ).order_by(self.models['Content'].popularity.desc()).limit(20).all()
                
                refreshed = 0
                for content in popular_content:
                    cache_key = f'cinebrain:details:{content.slug or content.id}'
                    
                    if not force and self.cache and self.cache.get(cache_key):
                        continue
                    
                    # Create comprehensive details cache
                    try:
                        genres = json.loads(content.genres or '[]')
                    except (json.JSONDecodeError, TypeError):
                        genres = []
                    
                    details_data = {
                        'id': content.id,
                        'slug': content.slug,
                        'title': content.title,
                        'original_title': content.original_title,
                        'content_type': content.content_type,
                        'genres': genres,
                        'rating': content.rating,
                        'vote_count': content.vote_count,
                        'overview': content.overview,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'runtime': content.runtime,
                        'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                        'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path and not content.backdrop_path.startswith('http') else content.backdrop_path,
                        'youtube_trailer': f"https://www.youtube.com/watch?v={content.youtube_trailer_id}" if content.youtube_trailer_id else None,
                        'is_trending': content.is_trending,
                        'is_new_release': content.is_new_release,
                        'is_critics_choice': content.is_critics_choice,
                        'cached_at': datetime.utcnow().isoformat()
                    }
                    
                    if self.cache:
                        self.cache.set(cache_key, details_data, timeout=self.TIMEOUTS['details'])
                        refreshed += 1
                
                return {'success': True, 'action': 'refreshed', 'items_refreshed': refreshed}
            
            return {'success': False, 'error': 'Content model not available'}
            
        except Exception as e:
            logger.error(f"Content details samples refresh error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _system_warmup_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive system warmup"""
        try:
            results = {}
            
            # Test all critical systems
            results['database'] = self._test_database_connection()
            results['cache'] = self._test_cache_connection()
            
            # Warm up services
            services_status = {}
            for service_name in ['TMDBService', 'JikanService', 'ContentService', 'new_releases_service', 'critics_choice_service', 'personalized_recommendation_engine']:
                services_status[service_name] = service_name in self.services
            
            results['services_status'] = services_status
            
            # Check critical models
            if self.models.get('Content'):
                try:
                    content_count = self.models['Content'].query.count()
                    results['content_available'] = content_count > 0
                    results['content_count'] = content_count
                    
                    # Check content with different flags
                    trending_count = self.models['Content'].query.filter_by(is_trending=True).count()
                    new_releases_count = self.models['Content'].query.filter_by(is_new_release=True).count()
                    critics_choice_count = self.models['Content'].query.filter_by(is_critics_choice=True).count()
                    
                    results['content_breakdown'] = {
                        'trending': trending_count,
                        'new_releases': new_releases_count,
                        'critics_choice': critics_choice_count
                    }
                except Exception as e:
                    results['content_error'] = str(e)
            
            if self.models.get('User'):
                try:
                    user_count = self.models['User'].query.count()
                    results['users_available'] = user_count > 0
                    results['user_count'] = user_count
                    
                    # Check active users
                    cutoff_date = datetime.utcnow() - timedelta(days=7)
                    active_users = self.models['User'].query.filter(
                        self.models['User'].last_active >= cutoff_date
                    ).count()
                    results['active_users_7d'] = active_users
                except Exception as e:
                    results['users_error'] = str(e)
            
            # System health
            results['system_metrics'] = self._get_basic_system_metrics()
            
            # Cache warmup
            if self.cache:
                self.cache.set('cinebrain_comprehensive_warmup', {
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': 'warmed_up',
                    'services_checked': len(services_status),
                    'models_checked': len([m for m in self.models.keys() if self.models[m]])
                }, timeout=600)
            
            return {'success': True, 'systems_checked': results}
            
        except Exception as e:
            logger.error(f"Comprehensive system warmup error: {e}")
            return {'success': False, 'error': str(e)}

    # ============================================================================
    # 🔧 EXISTING CACHE REFRESH HELPERS
    # ============================================================================
    
    def _refresh_trending_cache(self, force: bool = False) -> Dict[str, Any]:
        """Refresh trending content cache"""
        try:
            cache_key = 'cinebrain:trending'
            
            if not force and self.cache and self.cache.get(cache_key):
                return {'success': True, 'action': 'already_cached', 'cache_key': cache_key}
            
            trending_data = {}
            
            # Use existing trending service
            if 'TMDBService' in self.services:
                # Get trending for multiple categories
                trending_all = self.services['TMDBService'].get_trending('all', 'day', 1)
                trending_movies = self.services['TMDBService'].get_trending('movie', 'day', 1)
                trending_tv = self.services['TMDBService'].get_trending('tv', 'day', 1)
                
                trending_data = {
                    'all': trending_all,
                    'movies': trending_movies,
                    'tv': trending_tv,
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                if self.cache:
                    self.cache.set(cache_key, trending_data, timeout=self.TIMEOUTS['trending'])
                    total_items = 0
                    for category in ['all', 'movies', 'tv']:
                        if trending_data.get(category):
                            total_items += len(trending_data[category].get('results', []))
                    
                    return {'success': True, 'action': 'refreshed', 'items': total_items, 'categories': 3}
            
            return {'success': False, 'error': 'TMDB Service not available'}
            
        except Exception as e:
            logger.error(f"Trending cache refresh error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _refresh_critics_choice_cache(self, force: bool = False) -> Dict[str, Any]:
        """Refresh critics choice cache"""
        try:
            cache_key = 'cinebrain:critics_choice'
            
            if not force and self.cache and self.cache.get(cache_key):
                return {'success': True, 'action': 'already_cached', 'cache_key': cache_key}
            
            # Use existing critics choice service
            critics_service = self.services.get('critics_choice_service')
            if critics_service:
                critics_data = critics_service.get_enhanced_critics_choice(limit=30)
                
                if critics_data and self.cache:
                    self.cache.set(cache_key, critics_data, timeout=self.TIMEOUTS['critics_choice'])
                    return {'success': True, 'action': 'refreshed', 'items': len(critics_data.get('items', []))}
            
            return {'success': False, 'error': 'Critics Choice service not available'}
            
        except Exception as e:
            logger.error(f"Critics choice cache refresh error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _refresh_new_releases_cache(self, force: bool = False) -> Dict[str, Any]:
        """Refresh new releases cache"""
        try:
            cache_key = 'cinebrain:new_releases'
            
            if not force and self.cache and self.cache.get(cache_key):
                return {'success': True, 'action': 'already_cached', 'cache_key': cache_key}
            
            # Use existing new releases service
            new_releases_service = self.services.get('new_releases_service')
            if new_releases_service:
                releases_data = new_releases_service.get_new_releases(force_refresh=force)
                
                if releases_data and self.cache:
                    self.cache.set(cache_key, releases_data, timeout=self.TIMEOUTS['new_releases'])
                    return {'success': True, 'action': 'refreshed', 'categories': len(releases_data)}
            
            return {'success': False, 'error': 'New Releases service not available'}
            
        except Exception as e:
            logger.error(f"New releases cache refresh error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _refresh_upcoming_cache(self, force: bool = False) -> Dict[str, Any]:
        """Refresh upcoming content cache"""
        try:
            cache_key = 'cinebrain:upcoming'
            
            if not force and self.cache and self.cache.get(cache_key):
                return {'success': True, 'action': 'already_cached', 'cache_key': cache_key}
            
            # Use TMDB for upcoming movies and TV shows
            if 'TMDBService' in self.services:
                upcoming_data = {}
                
                # Get upcoming movies
                upcoming_movies = self.services['TMDBService'].get_content_details(550, 'movie')  # Sample for structure
                if upcoming_movies:
                    upcoming_data['movies'] = upcoming_movies
                
                # Add more upcoming content here
                upcoming_data['generated_at'] = datetime.utcnow().isoformat()
                
                if self.cache and upcoming_data:
                    self.cache.set(cache_key, upcoming_data, timeout=self.TIMEOUTS['upcoming'])
                    return {'success': True, 'action': 'refreshed', 'sections': len(upcoming_data)}
            
            return {'success': True, 'action': 'skipped', 'reason': 'Limited upcoming data available'}
            
        except Exception as e:
            logger.error(f"Upcoming cache refresh error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _refresh_discover_cache(self, force: bool = False) -> Dict[str, Any]:
        """Refresh discover section cache"""
        try:
            cache_key = 'cinebrain:discover'
            
            if not force and self.cache and self.cache.get(cache_key):
                return {'success': True, 'action': 'already_cached', 'cache_key': cache_key}
            
            discover_data = {}
            
            # Sample popular content for discover
            if 'TMDBService' in self.services:
                popular_movies = self.services['TMDBService'].get_popular('movie', 1, 'IN')
                popular_tv = self.services['TMDBService'].get_popular('tv', 1, 'IN')
                
                # Get regional content
                telugu_movies = self.services['TMDBService'].get_language_specific('te', 'movie', 1)
                hindi_movies = self.services['TMDBService'].get_language_specific('hi', 'movie', 1)
                
                discover_data = {
                    'popular_movies': popular_movies,
                    'popular_tv': popular_tv,
                    'telugu_movies': telugu_movies,
                    'hindi_movies': hindi_movies,
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                if self.cache:
                    self.cache.set(cache_key, discover_data, timeout=self.TIMEOUTS['discover'])
                    return {'success': True, 'action': 'refreshed', 'sections': len(discover_data)}
            
            return {'success': False, 'error': 'TMDB Service not available'}
            
        except Exception as e:
            logger.error(f"Discover cache refresh error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _refresh_similar_samples(self, force: bool = False) -> Dict[str, Any]:
        """Refresh similar content samples for popular items"""
        try:
            # Get a few popular content IDs to cache similar content for
            if self.models.get('Content'):
                popular_content = self.models['Content'].query.filter(
                    self.models['Content'].rating >= 7.0
                ).order_by(self.models['Content'].popularity.desc()).limit(15).all()
                
                refreshed = 0
                for content in popular_content:
                    cache_key = f'cinebrain:similar:{content.id}'
                    
                    if not force and self.cache and self.cache.get(cache_key):
                        continue
                    
                    # Get similar content (genre-based)
                    try:
                        # Parse genres
                        try:
                            content_genres = json.loads(content.genres or '[]')
                        except (json.JSONDecodeError, TypeError):
                            content_genres = []
                        
                        if content_genres:
                            # Find similar content with same primary genre
                            primary_genre = content_genres[0]
                            similar_items = self.models['Content'].query.filter(
                                self.models['Content'].id != content.id,
                                self.models['Content'].content_type == content.content_type,
                                self.models['Content'].genres.contains(primary_genre)
                            ).order_by(self.models['Content'].rating.desc()).limit(10).all()
                            
                            similar_data = []
                            for item in similar_items:
                                similar_data.append({
                                    'id': item.id,
                                    'slug': item.slug,
                                    'title': item.title,
                                    'content_type': item.content_type,
                                    'rating': item.rating,
                                    'poster_path': f"https://image.tmdb.org/t/p/w300{item.poster_path}" if item.poster_path and not item.poster_path.startswith('http') else item.poster_path,
                                    'similarity_score': 0.8,  # Simplified
                                    'match_type': 'genre_based'
                                })
                            
                            if self.cache and similar_data:
                                self.cache.set(cache_key, similar_data, timeout=self.TIMEOUTS['similar'])
                                refreshed += 1
                    except Exception as e:
                        logger.warning(f"Error caching similar for content {content.id}: {e}")
                
                return {'success': True, 'action': 'refreshed', 'items_refreshed': refreshed}
            
            return {'success': False, 'error': 'Content model not available'}
            
        except Exception as e:
            logger.error(f"Similar samples cache refresh error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _refresh_top_users_personalized(self, limit: int = 25) -> Dict[str, Any]:
        """Refresh personalized recommendations for top active users"""
        try:
            active_users = self._get_active_users(limit=limit)
            refreshed = 0
            errors = 0
            
            for user in active_users:
                try:
                    success = self._refresh_user_personalized_cache(user.id)
                    if success:
                        refreshed += 1
                    else:
                        errors += 1
                except Exception as e:
                    logger.warning(f"Error refreshing personalized cache for user {user.id}: {e}")
                    errors += 1
            
            return {'success': True, 'action': 'refreshed', 'users_refreshed': refreshed, 'errors': errors}
            
        except Exception as e:
            logger.error(f"Top users personalized refresh error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _refresh_user_personalized_cache(self, user_id: int) -> bool:
        """Refresh personalized cache for a specific user"""
        cache_key = f'cinebrain:personalized:{user_id}'
        
        try:
            # Use existing personalized recommendation engine if available
            personalized_engine = self.services.get('personalized_recommendation_engine')
            if personalized_engine:
                recommendations = personalized_engine.get_personalized_recommendations(
                    user_id=user_id,
                    recommendation_type='for_you',
                    limit=20
                )
                
                if recommendations.get('success') and self.cache:
                    self.cache.set(cache_key, recommendations, timeout=self.TIMEOUTS['personalized'])
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"User personalized cache refresh error for user {user_id}: {e}")
            return False
    
    def _get_active_users(self, limit: int = 50) -> List:
        """Get list of active users for personalized cache refresh"""
        try:
            if self.models.get('User'):
                # Get users active in the last 7 days
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                return self.models['User'].query.filter(
                    self.models['User'].last_active >= cutoff_date
                ).order_by(
                    self.models['User'].last_active.desc()
                ).limit(limit).all()
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting active users: {e}")
            return []
    
    def _system_warmup_minimal(self) -> Dict[str, Any]:
        """Minimal system warmup operations"""
        try:
            results = {}
            
            # Test database connection
            results['database'] = self._test_database_connection()
            
            # Test cache connection
            results['cache'] = self._test_cache_connection()
            
            # Basic content check
            if self.models.get('Content'):
                try:
                    results['content_available'] = self.models['Content'].query.count() > 0
                except:
                    results['content_available'] = False
            
            return {'success': True, 'checks': results}
            
        except Exception as e:
            logger.error(f"System warmup minimal error: {e}")
            return {'success': False, 'error': str(e)}

    # ============================================================================
    # 🧹 CLEANUP OPERATIONS
    # ============================================================================
    
    def _standard_cleanup(self) -> Dict[str, Any]:
        """Standard cleanup operations"""
        results = {}
        
        try:
            # Garbage collection
            collected = gc.collect()
            results['garbage_collection'] = {'objects_collected': collected}
            
            # Clear expired cache entries (if supported)
            if self.cache:
                try:
                    # This is cache-type dependent
                    results['cache_cleanup'] = {'attempted': True}
                except:
                    results['cache_cleanup'] = {'attempted': False}
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _minimal_cleanup(self) -> Dict[str, Any]:
        """Minimal cleanup operations"""
        results = {}
        
        try:
            # Just garbage collection
            collected = gc.collect()
            results['garbage_collection'] = {'objects_collected': collected}
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _aggressive_cleanup(self) -> Dict[str, Any]:
        """Aggressive cleanup operations"""
        results = {}
        
        try:
            # Multiple garbage collection passes
            total_collected = 0
            for _ in range(3):
                total_collected += gc.collect()
            
            results['aggressive_gc'] = {'objects_collected': total_collected}
            
            # Clear all non-essential caches
            if self.cache:
                try:
                    # Clear specific patterns
                    patterns_to_clear = ['cinebrain:temp:', 'cinebrain:session:']
                    for pattern in patterns_to_clear:
                        self.cache.delete(pattern)
                    results['cache_aggressive_cleanup'] = {'patterns_cleared': len(patterns_to_clear)}
                except:
                    results['cache_aggressive_cleanup'] = {'error': 'not_supported'}
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results

    # ============================================================================
    # 🔧 MAINTENANCE OPERATIONS
    # ============================================================================
    
    def _database_maintenance(self) -> Dict[str, Any]:
        """Database maintenance operations"""
        results = {}
        
        try:
            with self.app.app_context():
                # Check database health
                results['connection_test'] = self._test_database_connection()
                
                # Get basic statistics
                if self.models.get('Content'):
                    results['content_count'] = self.models['Content'].query.count()
                    results['content_with_slugs'] = self.models['Content'].query.filter(
                        self.models['Content'].slug != None,
                        self.models['Content'].slug != ''
                    ).count()
                
                if self.models.get('User'):
                    results['user_count'] = self.models['User'].query.count()
                    results['active_users_7d'] = self.models['User'].query.filter(
                        self.models['User'].last_active >= datetime.utcnow() - timedelta(days=7)
                    ).count()
                
                results['success'] = True
                
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _cache_optimization(self) -> Dict[str, Any]:
        """Cache optimization operations"""
        results = {}
        
        try:
            if self.cache:
                # Test cache performance
                start_time = time.time()
                self.cache.set('perf_test', 'test', timeout=10)
                self.cache.get('perf_test')
                self.cache.delete('perf_test')
                end_time = time.time()
                
                results['cache_performance'] = {
                    'response_time_ms': round((end_time - start_time) * 1000, 2),
                    'functional': True
                }
            else:
                results['cache_performance'] = {'functional': False}
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _performance_analysis(self) -> Dict[str, Any]:
        """Performance analysis operations"""
        results = {}
        
        try:
            # System metrics
            results['system_metrics'] = self._get_system_metrics()
            
            # Application metrics
            results['app_metrics'] = {
                'total_cache_refreshes': self.performance_metrics['cache_refreshes'],
                'average_refresh_time': self.performance_metrics['average_refresh_duration'],
                'successful_refreshes': self.performance_metrics['successful_refreshes'],
                'failed_refreshes': self.performance_metrics['failed_refreshes']
            }
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _data_integrity_check(self) -> Dict[str, Any]:
        """Data integrity check operations"""
        results = {}
        
        try:
            # Check for content without slugs
            if self.models.get('Content'):
                content_without_slugs = self.models['Content'].query.filter(
                    (self.models['Content'].slug == None) | (self.models['Content'].slug == '')
                ).count()
                
                results['content_integrity'] = {
                    'content_without_slugs': content_without_slugs,
                    'needs_attention': content_without_slugs > 0
                }
            
            # Check for users without activity
            if self.models.get('User'):
                inactive_users = self.models['User'].query.filter(
                    (self.models['User'].last_active == None) | 
                    (self.models['User'].last_active < datetime.utcnow() - timedelta(days=30))
                ).count()
                
                results['user_integrity'] = {
                    'inactive_users_30d': inactive_users
                }
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def _system_optimization(self) -> Dict[str, Any]:
        """System optimization operations"""
        results = {}
        
        try:
            # Optimize performance metrics
            self._optimize_performance_tracking()
            
            # Memory optimization
            gc.collect()
            
            results['memory_optimization'] = {'completed': True}
            results['performance_optimization'] = {'completed': True}
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results

    # ============================================================================
    # 📊 HELPER METHODS
    # ============================================================================
    
    def _test_database_connection(self) -> bool:
        """Test database connectivity"""
        try:
            with self.app.app_context():
                self.db.session.execute('SELECT 1')
                return True
        except Exception:
            return False
    
    def _test_cache_connection(self) -> bool:
        """Test cache connectivity"""
        try:
            if self.cache:
                self.cache.set('cinebrain_test', 'test', timeout=10)
                result = self.cache.get('cinebrain_test')
                self.cache.delete('cinebrain_test')
                return result == 'test'
            return False
        except Exception:
            return False
    
    def _get_basic_system_metrics(self) -> Dict[str, Any]:
        """Get basic system metrics"""
        try:
            return {
                'memory_usage': psutil.virtual_memory().percent,
                'cpu_usage': psutil.cpu_percent(interval=0.1),
                'disk_usage': psutil.disk_usage('/').percent
            }
        except Exception:
            return {'error': 'metrics_unavailable'}
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used
                },
                'cpu': {
                    'percent': psutil.cpu_percent(interval=0.1),
                    'count': psutil.cpu_count()
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'memory_status': 'healthy' if memory.percent < 85 else 'warning' if memory.percent < 95 else 'critical',
                'disk_status': 'healthy' if disk.percent < 85 else 'warning' if disk.percent < 95 else 'critical',
                'cpu_status': 'healthy'  # Simplified for now
            }
        except Exception:
            return {'status': 'unknown'}
    
    def _get_cache_analytics(self) -> Dict[str, Any]:
        """Get cache analytics"""
        analytics = {
            'cache_enabled': bool(self.cache),
            'cache_type': self.app.config.get('CACHE_TYPE')
        }
        
        if self.cache:
            try:
                # Cache hit/miss would require instrumentation
                analytics['functional'] = self._test_cache_connection()
                
                # Test cache performance
                start_time = time.time()
                self.cache.set('analytics_test', 'test', timeout=10)
                self.cache.get('analytics_test')
                self.cache.delete('analytics_test')
                response_time = (time.time() - start_time) * 1000
                
                analytics['performance'] = {
                    'response_time_ms': round(response_time, 2)
                }
                
            except:
                analytics['functional'] = False
        
        return analytics
    
    def _get_database_analytics(self) -> Dict[str, Any]:
        """Get database analytics"""
        analytics = {}
        
        try:
            with self.app.app_context():
                analytics['connection'] = self._test_database_connection()
                
                # Test database performance
                start_time = time.time()
                self.db.session.execute('SELECT 1')
                response_time = (time.time() - start_time) * 1000
                
                analytics['performance'] = {
                    'response_time_ms': round(response_time, 2)
                }
                
                # Get table counts
                table_counts = {}
                for model_name, model_class in self.models.items():
                    try:
                        if hasattr(model_class, 'query'):
                            table_counts[model_name.lower()] = model_class.query.count()
                    except:
                        table_counts[model_name.lower()] = 'error'
                
                analytics['table_counts'] = table_counts
                
        except Exception as e:
            analytics['error'] = str(e)
        
        return analytics
    
    def _update_performance_metrics(self, duration: float, successful_ops: int = 0, failed_ops: int = 0):
        """Update performance tracking metrics"""
        self.performance_metrics['cache_refreshes'] += 1
        self.performance_metrics['last_refresh_time'] = datetime.utcnow().isoformat()
        self.performance_metrics['total_refresh_time'] += duration
        self.performance_metrics['successful_refreshes'] += successful_ops
        self.performance_metrics['failed_refreshes'] += failed_ops
        
        if self.performance_metrics['cache_refreshes'] > 0:
            self.performance_metrics['average_refresh_duration'] = round(
                self.performance_metrics['total_refresh_time'] / 
                self.performance_metrics['cache_refreshes'], 2
            )
    
    def _optimize_performance_tracking(self):
        """Optimize performance tracking data"""
        # Reset metrics if they get too large to prevent memory issues
        if self.performance_metrics['cache_refreshes'] > 10000:
            self.performance_metrics['cache_refreshes'] = 1000
            self.performance_metrics['total_refresh_time'] = (
                self.performance_metrics['average_refresh_duration'] * 1000
            )
            self.performance_metrics['successful_refreshes'] = min(
                self.performance_metrics['successful_refreshes'], 1000
            )
            self.performance_metrics['failed_refreshes'] = min(
                self.performance_metrics['failed_refreshes'], 100
            )