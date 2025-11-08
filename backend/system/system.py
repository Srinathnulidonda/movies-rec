# System/system.py

import os
import psutil
import platform
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
from sqlalchemy import text, func
from sqlalchemy.exc import OperationalError
import redis
import json

logger = logging.getLogger(__name__)

class SystemService:
    """CineBrain System Service for health and performance monitoring"""
    
    db = None
    app = None
    cache = None
    models = None
    services = None
    
    @classmethod
    def init(cls, flask_app, database, app_models, app_services):
        """Initialize the system service"""
        cls.app = flask_app
        cls.db = database
        cls.cache = app_services.get('cache')
        cls.models = app_models
        cls.services = app_services
    
    @classmethod
    def get_health_status(cls) -> Dict[str, Any]:
        """Get overall system health status"""
        health_info = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '5.5.0',
            'python_version': platform.python_version(),
            'cinebrain_brand': 'CineBrain Entertainment Platform',
            'service': 'cinebrain_system'
        }
        
        # Database health
        try:
            cls.db.session.execute(text('SELECT 1'))
            health_info['database'] = 'connected'
        except Exception as e:
            health_info['database'] = f'disconnected: {str(e)}'
            health_info['status'] = 'degraded'
        
        # Cache health
        try:
            if cls.cache:
                cls.cache.set('cinebrain_health_check', 'ok', timeout=10)
                if cls.cache.get('cinebrain_health_check') == 'ok':
                    health_info['cache'] = 'connected'
                else:
                    health_info['cache'] = 'error'
                    health_info['status'] = 'degraded'
            else:
                health_info['cache'] = 'not_configured'
        except Exception as e:
            health_info['cache'] = f'disconnected: {str(e)}'
            health_info['status'] = 'degraded'
        
        # API Keys health
        health_info['api_keys'] = {
            'tmdb': bool(os.environ.get('TMDB_API_KEY')),
            'youtube': bool(os.environ.get('YOUTUBE_API_KEY')),
            'cloudinary': all([
                os.environ.get('CLOUDINARY_CLOUD_NAME'),
                os.environ.get('CLOUDINARY_API_KEY'),
                os.environ.get('CLOUDINARY_API_SECRET')
            ])
        }
        
        # Services health
        health_info['services'] = cls._get_services_health()
        
        # System resources
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            health_info['resources'] = {
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'cpu_usage': psutil.cpu_percent(interval=1)
            }
            
            # Alert on high resource usage
            if memory.percent > 90 or disk.percent > 90:
                health_info['status'] = 'degraded'
                health_info['alerts'] = ['High resource usage detected']
                
        except Exception as e:
            logger.warning(f"Could not get system resources: {e}")
        
        return health_info
    
    @classmethod
    def get_detailed_health_status(cls) -> Dict[str, Any]:
        """Get detailed health status with all components"""
        basic_health = cls.get_health_status()
        
        detailed = {
            **basic_health,
            'components': {}
        }
        
        # Database detailed health
        detailed['components']['database'] = cls._get_database_health()
        
        # Cache detailed health
        detailed['components']['cache'] = cls._get_cache_health()
        
        # Services detailed health
        detailed['components']['services'] = cls._get_detailed_services_health()
        
        # External APIs health
        detailed['components']['external_apis'] = cls._get_external_apis_health()
        
        # Background processes health
        detailed['components']['background_processes'] = cls._get_background_processes_health()
        
        return detailed
    
    @classmethod
    def get_performance_metrics(cls) -> Dict[str, Any]:
        """Get system performance metrics"""
        performance_data = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'cinebrain_brand': 'CineBrain Entertainment Platform',
            'cinebrain_service': 'performance'
        }
        
        # Database performance
        try:
            Content = cls.models.get('Content')
            if Content:
                total_content = Content.query.count()
                content_with_slugs = Content.query.filter(
                    Content.slug != None, Content.slug != ''
                ).count()
                
                performance_data['database'] = {
                    'total_content': total_content,
                    'content_with_slugs': content_with_slugs,
                    'content_without_slugs': total_content - content_with_slugs,
                    'slug_coverage': round((content_with_slugs / total_content * 100), 2) if total_content > 0 else 0
                }
        except Exception as e:
            performance_data['database'] = {'error': str(e)}
        
        # Cache performance
        performance_data['cache'] = {
            'type': cls.app.config.get('CACHE_TYPE', 'unknown'),
            'status': 'enabled' if cls.cache else 'disabled'
        }
        
        # Service performance
        performance_data['cinebrain_services'] = cls._get_services_performance()
        
        # Optimization status
        performance_data['cinebrain_performance'] = {
            'optimizations_applied': [
                'cinebrain_python_3.13_compatibility',
                'cinebrain_reduced_api_timeouts', 
                'cinebrain_optimized_thread_pools',
                'cinebrain_enhanced_caching',
                'cinebrain_error_handling_improvements',
                'cinebrain_cast_crew_optimization',
                'cinebrain_support_service_integration',
                'cinebrain_admin_notification_system',
                'cinebrain_real_time_monitoring',
                'cinebrain_auth_service_enhanced',
                'cinebrain_user_service_modular',
                'cinebrain_new_releases_service',
                'cinebrain_enhanced_critics_choice_service',
                'cinebrain_recommendation_service_modular',
                'cinebrain_advanced_personalized_system'
            ],
            'memory_optimizations': 'cinebrain_enabled',
            'unicode_fixes': 'cinebrain_applied',
            'monitoring': 'cinebrain_background_threads_active'
        }
        
        return performance_data
    
    @classmethod
    def get_detailed_performance_metrics(cls) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        basic_performance = cls.get_performance_metrics()
        
        detailed = {
            **basic_performance,
            'system_metrics': cls._get_system_metrics(),
            'application_metrics': cls._get_application_metrics(),
            'database_metrics': cls._get_database_metrics(),
            'cache_metrics': cls._get_cache_metrics()
        }
        
        return detailed
    
    @classmethod
    def get_system_statistics(cls) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'cinebrain_brand': 'CineBrain Entertainment Platform'
        }
        
        try:
            # Content statistics
            Content = cls.models.get('Content')
            User = cls.models.get('User')
            UserInteraction = cls.models.get('UserInteraction')
            Review = cls.models.get('Review')
            
            if Content:
                stats['content'] = {
                    'total': Content.query.count(),
                    'movies': Content.query.filter_by(content_type='movie').count(),
                    'tv_shows': Content.query.filter_by(content_type='tv').count(),
                    'anime': Content.query.filter_by(content_type='anime').count(),
                    'trending': Content.query.filter_by(is_trending=True).count(),
                    'new_releases': Content.query.filter_by(is_new_release=True).count(),
                    'critics_choice': Content.query.filter_by(is_critics_choice=True).count()
                }
            
            if User:
                total_users = User.query.count()
                active_users = User.query.filter(
                    User.last_active >= datetime.utcnow() - timedelta(days=7)
                ).count()
                
                stats['users'] = {
                    'total': total_users,
                    'active_7d': active_users,
                    'admins': User.query.filter_by(is_admin=True).count(),
                    'activity_rate': round((active_users / total_users * 100), 1) if total_users > 0 else 0
                }
            
            if UserInteraction:
                stats['interactions'] = {
                    'total': UserInteraction.query.count(),
                    'views': UserInteraction.query.filter_by(interaction_type='view').count(),
                    'likes': UserInteraction.query.filter_by(interaction_type='like').count(),
                    'favorites': UserInteraction.query.filter_by(interaction_type='favorite').count(),
                    'ratings': UserInteraction.query.filter_by(interaction_type='rating').count(),
                    'watchlist': UserInteraction.query.filter_by(interaction_type='watchlist').count()
                }
            
            if Review:
                stats['reviews'] = {
                    'total': Review.query.count(),
                    'approved': Review.query.filter_by(is_approved=True).count(),
                    'with_spoilers': Review.query.filter_by(has_spoilers=True).count()
                }
                
        except Exception as e:
            logger.error(f"Error getting system statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    @classmethod
    def get_database_statistics(cls) -> Dict[str, Any]:
        """Get detailed database statistics"""
        db_stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'connection_status': 'unknown'
        }
        
        try:
            # Test connection
            cls.db.session.execute(text('SELECT 1'))
            db_stats['connection_status'] = 'connected'
            
            # Get database size information
            if cls.app.config.get('SQLALCHEMY_DATABASE_URI', '').startswith('postgresql'):
                try:
                    result = cls.db.session.execute(text(
                        "SELECT pg_size_pretty(pg_database_size(current_database())) as size"
                    )).fetchone()
                    db_stats['database_size'] = result[0] if result else 'unknown'
                except Exception:
                    db_stats['database_size'] = 'unknown'
            
            # Table statistics
            tables_stats = {}
            for model_name, model_class in cls.models.items():
                try:
                    if hasattr(model_class, 'query'):
                        tables_stats[model_name.lower()] = {
                            'count': model_class.query.count(),
                            'table_name': model_class.__tablename__ if hasattr(model_class, '__tablename__') else model_name.lower()
                        }
                except Exception as e:
                    tables_stats[model_name.lower()] = {'error': str(e)}
            
            db_stats['tables'] = tables_stats
            
        except Exception as e:
            db_stats['connection_status'] = 'error'
            db_stats['error'] = str(e)
        
        return db_stats
    
    @classmethod
    def get_services_status(cls) -> Dict[str, Any]:
        """Get status of all CineBrain services"""
        services_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'cinebrain_brand': 'CineBrain Entertainment Platform'
        }
        
        # Core services
        services_status['core_services'] = {
            'tmdb_api': bool(os.environ.get('TMDB_API_KEY')),
            'youtube_api': bool(os.environ.get('YOUTUBE_API_KEY')),
            'database': cls._test_database_connection(),
            'cache': cls._test_cache_connection()
        }
        
        # Application services
        services_status['application_services'] = {
            'details_service': 'details_service' in cls.services,
            'content_service': 'ContentService' in cls.services,
            'new_releases_service': 'new_releases_service' in cls.services,
            'critics_choice_service': 'critics_choice_service' in cls.services,
            'support_service': 'support_bp' in cls.app.blueprints,
            'admin_service': 'admin_bp' in cls.app.blueprints,
            'auth_service': 'auth_bp' in cls.app.blueprints,
            'user_service': 'user_bp' in cls.app.blueprints,
            'recommendation_service': 'recommendations' in cls.app.blueprints,
            'personalized_service': 'personalized' in cls.app.blueprints,
            'system_service': True  # This service
        }
        
        # Advanced services
        services_status['advanced_services'] = {
            'profile_analyzer': 'profile_analyzer' in cls.services,
            'personalized_recommendation_engine': 'personalized_recommendation_engine' in cls.services,
            'recommendation_orchestrator': 'recommendation_orchestrator' in cls.services,
            'ultra_powerful_similarity_engine': True,
            'cinematic_dna_analysis': 'profile_analyzer' in cls.services,
            'real_time_learning': 'profile_analyzer' in cls.services
        }
        
        # External services
        services_status['external_services'] = {
            'cloudinary': all([
                os.environ.get('CLOUDINARY_CLOUD_NAME'),
                os.environ.get('CLOUDINARY_API_KEY'),
                os.environ.get('CLOUDINARY_API_SECRET')
            ]),
            'redis': cls._test_redis_connection(),
            'tmdb': cls._test_tmdb_connection(),
            'jikan': True,  # No API key required
            'youtube': bool(os.environ.get('YOUTUBE_API_KEY'))
        }
        
        return services_status
    
    @classmethod
    def get_cache_statistics(cls) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'cache_enabled': bool(cls.cache),
            'cache_type': cls.app.config.get('CACHE_TYPE', 'not_configured')
        }
        
        if cls.cache:
            try:
                # Test cache functionality
                test_key = 'cinebrain_cache_test'
                test_value = 'test_value'
                
                cls.cache.set(test_key, test_value, timeout=60)
                retrieved_value = cls.cache.get(test_key)
                
                cache_stats['cache_working'] = (retrieved_value == test_value)
                
                # Clean up test key
                cls.cache.delete(test_key)
                
            except Exception as e:
                cache_stats['cache_working'] = False
                cache_stats['cache_error'] = str(e)
        else:
            cache_stats['cache_working'] = False
            cache_stats['message'] = 'Cache not configured'
        
        return cache_stats
    
    @classmethod
    def get_monitoring_alerts(cls) -> Dict[str, Any]:
        """Get system monitoring alerts"""
        alerts = {
            'timestamp': datetime.utcnow().isoformat(),
            'alerts': []
        }
        
        try:
            # Check system resources
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu = psutil.cpu_percent(interval=1)
            
            if memory.percent > 90:
                alerts['alerts'].append({
                    'level': 'critical',
                    'component': 'memory',
                    'message': f'Memory usage is {memory.percent}%',
                    'threshold': '90%'
                })
            
            if disk.percent > 90:
                alerts['alerts'].append({
                    'level': 'critical',
                    'component': 'disk',
                    'message': f'Disk usage is {disk.percent}%',
                    'threshold': '90%'
                })
            
            if cpu > 80:
                alerts['alerts'].append({
                    'level': 'warning',
                    'component': 'cpu',
                    'message': f'CPU usage is {cpu}%',
                    'threshold': '80%'
                })
            
            # Check database connection
            if not cls._test_database_connection():
                alerts['alerts'].append({
                    'level': 'critical',
                    'component': 'database',
                    'message': 'Database connection failed',
                    'action': 'Check database connectivity'
                })
            
            # Check cache connection
            if cls.cache and not cls._test_cache_connection():
                alerts['alerts'].append({
                    'level': 'warning',
                    'component': 'cache',
                    'message': 'Cache connection failed',
                    'action': 'Check cache service'
                })
            
        except Exception as e:
            alerts['alerts'].append({
                'level': 'error',
                'component': 'monitoring',
                'message': f'Error generating alerts: {str(e)}',
                'action': 'Check monitoring system'
            })
        
        alerts['total_alerts'] = len(alerts['alerts'])
        alerts['critical_count'] = len([a for a in alerts['alerts'] if a.get('level') == 'critical'])
        alerts['warning_count'] = len([a for a in alerts['alerts'] if a.get('level') == 'warning'])
        
        return alerts
    
    @classmethod
    def get_real_time_metrics(cls) -> Dict[str, Any]:
        """Get real-time system metrics"""
        metrics = {
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # System metrics
            metrics['system'] = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
            
            # Database metrics
            metrics['database'] = {
                'connection_status': cls._test_database_connection(),
                'active_connections': cls._get_active_db_connections()
            }
            
            # Cache metrics
            if cls.cache:
                metrics['cache'] = {
                    'connection_status': cls._test_cache_connection(),
                    'type': cls.app.config.get('CACHE_TYPE')
                }
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    @classmethod
    def get_cli_operations_status(cls) -> Dict[str, Any]:
        """Get status of CLI operations"""
        cli_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'available_commands': [
                'generate-slugs',
                'populate-cast-crew',
                'cinebrain-new-releases-refresh',
                'analyze-user-profiles',
                'test-personalized-recommendations'
            ]
        }
        
        # Check if required services are available for each command
        cli_status['command_status'] = {
            'generate-slugs': {
                'available': 'details_service' in cls.services,
                'requirements': ['details_service', 'database']
            },
            'populate-cast-crew': {
                'available': all([
                    'details_service' in cls.services,
                    bool(os.environ.get('TMDB_API_KEY'))
                ]),
                'requirements': ['details_service', 'tmdb_api', 'database']
            },
            'cinebrain-new-releases-refresh': {
                'available': 'new_releases_service' in cls.services,
                'requirements': ['new_releases_service', 'tmdb_api', 'database']
            },
            'analyze-user-profiles': {
                'available': 'profile_analyzer' in cls.services,
                'requirements': ['profile_analyzer', 'database']
            },
            'test-personalized-recommendations': {
                'available': 'personalized_recommendation_engine' in cls.services,
                'requirements': ['personalized_recommendation_engine', 'database']
            }
        }
        
        return cli_status
    
    @classmethod
    def get_system_info(cls) -> Dict[str, Any]:
        """Get comprehensive system information"""
        sys_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'cinebrain_brand': 'CineBrain Entertainment Platform'
        }
        
        # Platform information
        sys_info['platform'] = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation()
        }
        
        # Environment information
        sys_info['environment'] = {
            'flask_env': os.environ.get('FLASK_ENV', 'production'),
            'debug_mode': cls.app.debug,
            'testing_mode': cls.app.testing,
            'database_url_configured': bool(os.environ.get('DATABASE_URL')),
            'redis_url_configured': bool(os.environ.get('REDIS_URL')),
            'secret_key_configured': bool(cls.app.secret_key)
        }
        
        # Application information
        sys_info['application'] = {
            'name': 'CineBrain',
            'version': '5.5.0',
            'blueprints': list(cls.app.blueprints.keys()),
            'url_rules_count': len(list(cls.app.url_map.iter_rules())),
            'registered_services': list(cls.services.keys()) if cls.services else []
        }
        
        return sys_info
    
    @classmethod
    def get_version_info(cls) -> Dict[str, Any]:
        """Get version and build information"""
        return {
            'cinebrain': {
                'version': '5.5.0',
                'build_date': datetime.utcnow().isoformat(),
                'python_version': platform.python_version(),
                'platform': platform.system()
            },
            'features': {
                'advanced_personalized_recommendations': True,
                'cinematic_dna_analysis': True,
                'real_time_learning': True,
                'telugu_cultural_priority': True,
                'modular_architecture': True,
                'comprehensive_monitoring': True
            },
            'api_version': '3.0',
            'database_schema_version': '1.0'
        }
    
    # Helper methods
    @classmethod
    def _get_services_health(cls) -> Dict[str, str]:
        """Get basic services health status"""
        return {
            'tmdb': 'enabled' if os.environ.get('TMDB_API_KEY') else 'disabled',
            'youtube': 'enabled' if os.environ.get('YOUTUBE_API_KEY') else 'disabled',
            'details_service': 'enabled' if 'details_service' in cls.services else 'disabled',
            'content_service': 'enabled' if 'ContentService' in cls.services else 'disabled',
            'new_releases_service': 'enabled' if 'new_releases_service' in cls.services else 'disabled',
            'personalized_service': 'enabled' if 'personalized_recommendation_engine' in cls.services else 'disabled'
        }
    
    @classmethod
    def _get_database_health(cls) -> Dict[str, Any]:
        """Get detailed database health"""
        health = {'status': 'unknown'}
        
        try:
            cls.db.session.execute(text('SELECT 1'))
            health['status'] = 'healthy'
            health['connection'] = 'active'
            
            # Check table existence
            tables_exist = {}
            for model_name, model_class in cls.models.items():
                try:
                    if hasattr(model_class, 'query'):
                        model_class.query.limit(1).first()
                        tables_exist[model_name.lower()] = True
                except Exception:
                    tables_exist[model_name.lower()] = False
            
            health['tables'] = tables_exist
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health
    
    @classmethod
    def _get_cache_health(cls) -> Dict[str, Any]:
        """Get detailed cache health"""
        health = {
            'configured': bool(cls.cache),
            'type': cls.app.config.get('CACHE_TYPE', 'not_configured')
        }
        
        if cls.cache:
            try:
                test_key = 'health_check_test'
                cls.cache.set(test_key, 'test', timeout=10)
                result = cls.cache.get(test_key)
                cls.cache.delete(test_key)
                
                health['status'] = 'healthy' if result == 'test' else 'unhealthy'
                health['functional'] = result == 'test'
                
            except Exception as e:
                health['status'] = 'unhealthy'
                health['error'] = str(e)
                health['functional'] = False
        else:
            health['status'] = 'not_configured'
            health['functional'] = False
        
        return health
    
    @classmethod
    def _get_detailed_services_health(cls) -> Dict[str, Any]:
        """Get detailed services health"""
        return {
            'recommendation_engine': {
                'legacy': 'recommendation_engine' in cls.services,
                'advanced': 'personalized_recommendation_engine' in cls.services,
                'profile_analyzer': 'profile_analyzer' in cls.services
            },
            'content_services': {
                'details_service': 'details_service' in cls.services,
                'content_service': 'ContentService' in cls.services,
                'new_releases': 'new_releases_service' in cls.services,
                'critics_choice': 'critics_choice_service' in cls.services
            },
            'user_services': {
                'auth': 'auth_bp' in cls.app.blueprints,
                'user_management': 'user_bp' in cls.app.blueprints,
                'admin': 'admin_bp' in cls.app.blueprints,
                'support': 'support_bp' in cls.app.blueprints
            }
        }
    
    @classmethod
    def _get_external_apis_health(cls) -> Dict[str, Any]:
        """Get external APIs health"""
        return {
            'tmdb': {
                'configured': bool(os.environ.get('TMDB_API_KEY')),
                'functional': cls._test_tmdb_connection()
            },
            'youtube': {
                'configured': bool(os.environ.get('YOUTUBE_API_KEY')),
                'functional': bool(os.environ.get('YOUTUBE_API_KEY'))  # Can't easily test without making requests
            },
            'cloudinary': {
                'configured': all([
                    os.environ.get('CLOUDINARY_CLOUD_NAME'),
                    os.environ.get('CLOUDINARY_API_KEY'),
                    os.environ.get('CLOUDINARY_API_SECRET')
                ]),
                'functional': True  # Assume functional if configured
            },
            'jikan': {
                'configured': True,  # No API key required
                'functional': True   # Assume functional
            }
        }
    
    @classmethod
    def _get_background_processes_health(cls) -> Dict[str, Any]:
        """Get background processes health"""
        return {
            'support_monitoring': True,  # Always active in app.py
            'recommendation_updates': 'personalized_recommendation_engine' in cls.services,
            'cache_management': bool(cls.cache),
            'database_connections': cls._test_database_connection()
        }
    
    @classmethod
    def _get_services_performance(cls) -> Dict[str, str]:
        """Get services performance status"""
        return {
            'new_releases_service': 'enabled' if 'new_releases_service' in cls.services else 'disabled',
            'critics_choice_service': 'enabled' if 'critics_choice_service' in cls.services else 'disabled',
            'algorithms': 'cinebrain_optimized_enabled',
            'slug_support': 'cinebrain_comprehensive_enabled',
            'details_service': 'enabled' if 'details_service' in cls.services else 'disabled',
            'content_service': 'enabled' if 'ContentService' in cls.services else 'disabled',
            'cast_crew': 'cinebrain_fully_enabled',
            'support_service': 'enabled' if 'support_bp' in cls.app.blueprints else 'disabled',
            'admin_notifications': 'cinebrain_enabled',
            'monitoring': 'cinebrain_active',
            'auth_service': 'enabled' if 'auth_bp' in cls.app.blueprints else 'disabled',
            'user_service': 'enabled' if 'user_bp' in cls.app.blueprints else 'disabled',
            'recommendation_service': 'enabled' if 'recommendations' in cls.app.blueprints else 'disabled',
            'personalized_service': 'enabled' if 'personalized' in cls.app.blueprints else 'disabled'
        }
    
    @classmethod
    def _get_system_metrics(cls) -> Dict[str, Any]:
        """Get detailed system metrics"""
        try:
            return {
                'cpu': {
                    'percent': psutil.cpu_percent(interval=1),
                    'count': psutil.cpu_count(),
                    'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
                },
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'percent': psutil.virtual_memory().percent,
                    'used': psutil.virtual_memory().used
                },
                'disk': {
                    'total': psutil.disk_usage('/').total,
                    'used': psutil.disk_usage('/').used,
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    @classmethod
    def _get_application_metrics(cls) -> Dict[str, Any]:
        """Get application-specific metrics"""
        return {
            'blueprints_count': len(cls.app.blueprints),
            'url_rules_count': len(list(cls.app.url_map.iter_rules())),
            'registered_services_count': len(cls.services) if cls.services else 0,
            'debug_mode': cls.app.debug,
            'testing_mode': cls.app.testing
        }
    
    @classmethod
    def _get_database_metrics(cls) -> Dict[str, Any]:
        """Get database performance metrics"""
        metrics = {}
        
        try:
            # Connection test
            start_time = datetime.utcnow()
            cls.db.session.execute(text('SELECT 1'))
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            metrics['response_time_ms'] = response_time
            metrics['connection_status'] = 'healthy'
            
        except Exception as e:
            metrics['connection_status'] = 'unhealthy'
            metrics['error'] = str(e)
        
        return metrics
    
    @classmethod
    def _get_cache_metrics(cls) -> Dict[str, Any]:
        """Get cache performance metrics"""
        metrics = {
            'configured': bool(cls.cache),
            'type': cls.app.config.get('CACHE_TYPE')
        }
        
        if cls.cache:
            try:
                # Test cache performance
                start_time = datetime.utcnow()
                cls.cache.set('perf_test', 'test', timeout=10)
                cls.cache.get('perf_test')
                cls.cache.delete('perf_test')
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                metrics['response_time_ms'] = response_time
                metrics['functional'] = True
                
            except Exception as e:
                metrics['functional'] = False
                metrics['error'] = str(e)
        
        return metrics
    
    @classmethod
    def _test_database_connection(cls) -> bool:
        """Test database connection"""
        try:
            cls.db.session.execute(text('SELECT 1'))
            return True
        except Exception:
            return False
    
    @classmethod
    def _test_cache_connection(cls) -> bool:
        """Test cache connection"""
        if not cls.cache:
            return False
        
        try:
            cls.cache.set('test_connection', 'test', timeout=10)
            result = cls.cache.get('test_connection')
            cls.cache.delete('test_connection')
            return result == 'test'
        except Exception:
            return False
    
    @classmethod
    def _test_redis_connection(cls) -> bool:
        """Test Redis connection"""
        redis_url = os.environ.get('REDIS_URL')
        if not redis_url:
            return False
        
        try:
            import redis
            r = redis.from_url(redis_url)
            r.ping()
            return True
        except Exception:
            return False
    
    @classmethod
    def _test_tmdb_connection(cls) -> bool:
        """Test TMDB API connection"""
        return bool(os.environ.get('TMDB_API_KEY'))  # Basic check
    
    @classmethod
    def _get_active_db_connections(cls) -> int:
        """Get active database connections count"""
        try:
            if cls.app.config.get('SQLALCHEMY_DATABASE_URI', '').startswith('postgresql'):
                result = cls.db.session.execute(text(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                )).fetchone()
                return result[0] if result else 0
        except Exception:
            pass
        return 0