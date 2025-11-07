# backend/system/restart_service.py

"""
CineBrain Advanced Restart Service
Optimized for Render Free Tier - Automatic Restart & Health Management System
"""

import os
import sys
import time
import signal
import threading
import logging
import traceback
import atexit
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import json
import psutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import gc
from sqlalchemy import text

logger = logging.getLogger('cinebrain.restart')

class CineBrainAdvancedRestartService:
    """
    Advanced Restart & Health Management Service for CineBrain
    Ensures optimal performance through intelligent restarts and comprehensive monitoring
    """
    
    def __init__(self, app, db, cache, memory_optimizer, config=None):
        self.app = app
        self.db = db
        self.cache = cache
        self.memory_optimizer = memory_optimizer
        self.config = config or {}
        
        # Configuration
        self.restart_interval = self.config.get('restart_interval', 600)  # 10 minutes
        self.memory_threshold = self.config.get('memory_threshold', 200)  # 200 MB
        self.health_check_interval = self.config.get('health_check_interval', 60)  # 1 minute
        self.max_restart_attempts = self.config.get('max_restart_attempts', 5)
        self.force_restart_on_memory = self.config.get('force_restart_on_memory', True)
        self.aggressive_cleanup = self.config.get('aggressive_cleanup', True)
        
        # Service state
        self.restart_count = 0
        self.start_time = datetime.utcnow()
        self.last_health_check = None
        self.last_restart_time = None
        self.is_running = False
        self.health_status = {}
        self.error_count = 0
        self.last_error = None
        
        # Threading
        self.restart_thread = None
        self.monitor_thread = None
        self.cleanup_handlers = []
        self.performance_history = []
        
        # HTTP session for health checks
        self.session = self._create_http_session()
        self.restart_logger = logging.getLogger('cinebrain.restart')
        
        self.restart_logger.info("🚀 CineBrain Advanced Restart Service initialized")
    
    def _create_http_session(self):
        """Create optimized HTTP session for health checks"""
        session = requests.Session()
        retry = Retry(
            total=2,
            read=1,
            connect=1,
            backoff_factor=0.3,
            status_forcelist=(500, 502, 504)
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=2, pool_maxsize=2)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def is_render_deployment(self) -> bool:
        """Check if running on Render platform"""
        return (
            os.environ.get('PORT') is not None and 
            os.environ.get('FLASK_ENV') != 'development' and
            (os.environ.get('RENDER') is not None or 
             'render' in os.environ.get('HOSTNAME', '').lower() or
             'render.com' in os.environ.get('EXTERNAL_URL', ''))
        )
    
    def get_system_metrics(self):
        """Get comprehensive system metrics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
            
            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'service_uptime': {
                    'seconds': uptime_seconds,
                    'minutes': round(uptime_seconds / 60, 2),
                    'hours': round(uptime_seconds / 3600, 2)
                },
                'memory': {
                    'usage_mb': round(memory_info.rss / 1024 / 1024, 2),
                    'usage_percent': round(process.memory_percent(), 2),
                    'virtual_mb': round(memory_info.vms / 1024 / 1024, 2),
                    'threshold_mb': self.memory_threshold,
                    'exceeds_threshold': memory_info.rss / 1024 / 1024 > self.memory_threshold
                },
                'cpu': {
                    'percent': round(process.cpu_percent(interval=0.1), 2),
                    'threads': process.num_threads()
                },
                'process': {
                    'pid': process.pid,
                    'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
                },
                'restart_info': {
                    'count': self.restart_count,
                    'last_restart': self.last_restart_time.isoformat() if self.last_restart_time else None,
                    'next_restart_in_seconds': max(0, self.restart_interval - (uptime_seconds % self.restart_interval)),
                    'error_count': self.error_count,
                    'last_error': self.last_error
                },
                'deployment': {
                    'is_render': self.is_render_deployment(),
                    'environment': os.environ.get('FLASK_ENV', 'production'),
                    'port': os.environ.get('PORT'),
                    'hostname': os.environ.get('HOSTNAME', 'unknown')
                }
            }
            
            # Database health
            if self.db:
                try:
                    start_time = time.time()
                    self.db.session.execute(text('SELECT 1')).fetchone()
                    db_time = round((time.time() - start_time) * 1000, 2)
                    metrics['database'] = {
                        'status': 'connected',
                        'response_time_ms': db_time
                    }
                except Exception as e:
                    metrics['database'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Cache health
            if self.cache:
                try:
                    start_time = time.time()
                    test_key = f'restart_health_{int(time.time())}'
                    self.cache.set(test_key, 'test', timeout=5)
                    result = self.cache.get(test_key)
                    cache_time = round((time.time() - start_time) * 1000, 2)
                    
                    metrics['cache'] = {
                        'status': 'connected' if result == 'test' else 'degraded',
                        'response_time_ms': cache_time
                    }
                    
                    try:
                        self.cache.delete(test_key)
                    except:
                        pass
                        
                except Exception as e:
                    metrics['cache'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Memory optimizer status
            if self.memory_optimizer:
                try:
                    optimizer_status = self.memory_optimizer.get_status()
                    metrics['memory_optimizer'] = {
                        'is_running': optimizer_status['service_info']['is_running'],
                        'optimization_count': optimizer_status['service_info']['optimization_count'],
                        'total_freed_mb': optimizer_status['service_info']['total_memory_freed_mb'],
                        'efficiency_score': optimizer_status['performance']['memory_efficiency_score']
                    }
                except Exception as e:
                    metrics['memory_optimizer'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            return metrics
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            self.restart_logger.error(f"❌ Error getting system metrics: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'restart_count': self.restart_count
            }
    
    def perform_health_check(self):
        """Perform comprehensive health check with scoring"""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'score': 100,
            'checks': {},
            'warnings': [],
            'critical_issues': []
        }
        
        try:
            score = 100
            
            # App endpoint health check
            if self.app:
                port = os.environ.get('PORT', 5000)
                try:
                    response = self.session.get(
                        f"http://localhost:{port}/api/health",
                        timeout=3
                    )
                    response_time = round(response.elapsed.total_seconds() * 1000, 2)
                    
                    if response.status_code == 200:
                        health_status['checks']['app_endpoint'] = {
                            'status': 'healthy',
                            'response_code': response.status_code,
                            'response_time_ms': response_time
                        }
                        if response_time > 3000:
                            health_status['warnings'].append('App response time over 3 seconds')
                            score -= 10
                    else:
                        health_status['checks']['app_endpoint'] = {
                            'status': 'unhealthy',
                            'response_code': response.status_code,
                            'response_time_ms': response_time
                        }
                        health_status['critical_issues'].append('App endpoint unhealthy')
                        score -= 30
                        
                except Exception as e:
                    health_status['checks']['app_endpoint'] = {
                        'status': 'critical',
                        'error': str(e)
                    }
                    health_status['critical_issues'].append('App endpoint unreachable')
                    score -= 40
            
            # Database health check
            if self.db:
                try:
                    start_time = time.time()
                    result = self.db.session.execute(text('SELECT COUNT(*) as count FROM "user"')).fetchone()
                    query_time = round((time.time() - start_time) * 1000, 2)
                    
                    health_status['checks']['database'] = {
                        'status': 'healthy',
                        'query_time_ms': query_time,
                        'user_count': result[0] if result else 0
                    }
                    
                    if query_time > 1000:
                        health_status['warnings'].append('Database response time over 1 second')
                        score -= 5
                        
                except Exception as e:
                    health_status['checks']['database'] = {
                        'status': 'critical',
                        'error': str(e)
                    }
                    health_status['critical_issues'].append('Database connection failed')
                    score -= 50
            
            # Memory health check
            metrics = self.get_system_metrics()
            memory_mb = metrics.get('memory', {}).get('usage_mb', 0)
            
            if memory_mb > self.memory_threshold:
                health_status['checks']['memory'] = {
                    'status': 'critical',
                    'usage_mb': memory_mb,
                    'threshold_mb': self.memory_threshold
                }
                health_status['critical_issues'].append(f'Memory usage ({memory_mb} MB) exceeds threshold')
                score -= 25
            elif memory_mb > self.memory_threshold * 0.8:
                health_status['checks']['memory'] = {
                    'status': 'warning',
                    'usage_mb': memory_mb,
                    'threshold_mb': self.memory_threshold
                }
                health_status['warnings'].append(f'Memory usage ({memory_mb} MB) approaching threshold')
                score -= 10
            else:
                health_status['checks']['memory'] = {
                    'status': 'healthy',
                    'usage_mb': memory_mb,
                    'threshold_mb': self.memory_threshold
                }
            
            # Uptime check
            uptime_minutes = metrics.get('service_uptime', {}).get('minutes', 0)
            if uptime_minutes > 9:
                health_status['checks']['uptime'] = {
                    'status': 'restart_due',
                    'uptime_minutes': uptime_minutes,
                    'restart_interval_minutes': self.restart_interval / 60
                }
                health_status['warnings'].append('Restart due soon')
                score -= 5
            else:
                health_status['checks']['uptime'] = {
                    'status': 'healthy',
                    'uptime_minutes': uptime_minutes
                }
            
            # Memory optimizer check
            if self.memory_optimizer:
                try:
                    optimizer_status = self.memory_optimizer.get_status()
                    if optimizer_status['service_info']['is_running']:
                        health_status['checks']['memory_optimizer'] = {
                            'status': 'healthy',
                            'optimization_count': optimizer_status['service_info']['optimization_count'],
                            'efficiency_score': optimizer_status['performance']['memory_efficiency_score']
                        }
                    else:
                        health_status['checks']['memory_optimizer'] = {
                            'status': 'warning',
                            'message': 'Memory optimizer not running'
                        }
                        health_status['warnings'].append('Memory optimizer inactive')
                        score -= 15
                except Exception as e:
                    health_status['checks']['memory_optimizer'] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    score -= 10
            
            # Calculate overall status
            health_status['score'] = max(0, score)
            
            if score >= 90:
                health_status['overall_status'] = 'excellent'
            elif score >= 70:
                health_status['overall_status'] = 'healthy'
            elif score >= 50:
                health_status['overall_status'] = 'degraded'
            elif score >= 30:
                health_status['overall_status'] = 'unhealthy'
            else:
                health_status['overall_status'] = 'critical'
            
            health_status['system_metrics'] = metrics
            
            # Update service state
            self.last_health_check = datetime.utcnow()
            self.health_status = health_status
            
            # Add to performance history
            if len(self.performance_history) > 20:
                self.performance_history.pop(0)
            self.performance_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'score': score,
                'status': health_status['overall_status'],
                'memory_mb': memory_mb
            })
            
            return health_status
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            self.restart_logger.error(f"❌ Health check error: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_status': 'error',
                'error': str(e),
                'score': 0
            }
    
    def aggressive_resource_cleanup(self):
        """Perform comprehensive resource cleanup before restart"""
        self.restart_logger.info("🧹 CineBrain performing aggressive resource cleanup...")
        
        cleanup_results = {
            'successful': [],
            'failed': [],
            'memory_before_mb': 0,
            'memory_after_mb': 0,
            'cleanup_time_ms': 0
        }
        
        start_time = time.time()
        
        try:
            cleanup_results['memory_before_mb'] = round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
        except:
            pass
        
        # Memory optimizer cleanup
        if self.memory_optimizer:
            try:
                self.memory_optimizer.aggressive_memory_cleanup()
                cleanup_results['successful'].append('memory_optimizer_cleanup')
            except Exception as e:
                cleanup_results['failed'].append(f'memory_optimizer: {str(e)}')
        
        # Database cleanup
        if self.db:
            try:
                self.db.session.remove()
                self.db.session.close()
                if hasattr(self.db, 'engine') and hasattr(self.db.engine, 'dispose'):
                    self.db.engine.dispose()
                cleanup_results['successful'].append('database_connections_closed')
            except Exception as e:
                cleanup_results['failed'].append(f'database: {str(e)}')
        
        # Cache cleanup
        if self.cache and self.aggressive_cleanup:
            try:
                if hasattr(self.cache, 'clear'):
                    self.cache.clear()
                    cleanup_results['successful'].append('cache_completely_cleared')
            except Exception as e:
                cleanup_results['failed'].append(f'cache: {str(e)}')
        
        # HTTP session cleanup
        try:
            if hasattr(self, 'session') and self.session:
                self.session.close()
            cleanup_results['successful'].append('http_session_closed')
        except Exception as e:
            cleanup_results['failed'].append(f'http_session: {str(e)}')
        
        # Garbage collection
        try:
            for _ in range(3):
                collected = gc.collect()
            cleanup_results['successful'].append('garbage_collection_completed')
        except Exception as e:
            cleanup_results['failed'].append(f'garbage_collection: {str(e)}')
        
        # Custom cleanup handlers
        for handler in self.cleanup_handlers:
            try:
                handler()
                cleanup_results['successful'].append('custom_handler')
            except Exception as e:
                cleanup_results['failed'].append(f'custom_handler: {str(e)}')
        
        # Final measurements
        try:
            cleanup_results['memory_after_mb'] = round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
            memory_freed = cleanup_results['memory_before_mb'] - cleanup_results['memory_after_mb']
            cleanup_results['memory_freed_mb'] = memory_freed
        except:
            pass
        
        cleanup_results['cleanup_time_ms'] = round((time.time() - start_time) * 1000, 2)
        
        self.restart_logger.info(f"🎯 Cleanup completed in {cleanup_results['cleanup_time_ms']}ms")
        self.restart_logger.info(f"✅ Successful: {len(cleanup_results['successful'])} | ❌ Failed: {len(cleanup_results['failed'])}")
        
        return cleanup_results
    
    def intelligent_force_restart(self, reason="scheduled", health_data=None):
        """Force intelligent restart with comprehensive logging"""
        self.restart_count += 1
        self.last_restart_time = datetime.utcnow()
        
        restart_info = {
            'restart_number': self.restart_count,
            'reason': reason,
            'timestamp': self.last_restart_time.isoformat(),
            'uptime_minutes': round((self.last_restart_time - self.start_time).total_seconds() / 60, 2),
            'memory_usage_mb': round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
            'health_score': health_data.get('score', 0) if health_data else 0,
            'critical_issues': health_data.get('critical_issues', []) if health_data else []
        }
        
        self.restart_logger.info(f"🔄 CineBrain Intelligent Force Restart #{self.restart_count}")
        self.restart_logger.info(f"📊 Restart Details: {json.dumps(restart_info, indent=2)}")
        
        # Perform cleanup
        cleanup_results = self.aggressive_resource_cleanup()
        
        self.restart_logger.info(f"💫 CineBrain restarting due to: {reason}")
        self.restart_logger.info(f"⏱️ Uptime: {restart_info['uptime_minutes']} minutes")
        self.restart_logger.info(f"💾 Memory usage: {restart_info['memory_usage_mb']} MB")
        self.restart_logger.info(f"🏥 Health score: {restart_info['health_score']}/100")
        
        if cleanup_results.get('memory_freed_mb', 0) > 0:
            self.restart_logger.info(f"🆓 Memory freed: {cleanup_results['memory_freed_mb']} MB")
        
        # Give time for logs to flush
        time.sleep(2)
        
        self.restart_logger.info("🚀 CineBrain initiating intelligent restart now...")
        os._exit(0)
    
    def auto_restart_worker(self):
        """Main auto-restart worker thread"""
        self.restart_logger.info(f"🔄 Auto-restart worker started (interval: {self.restart_interval}s)")
        
        consecutive_health_failures = 0
        
        while self.is_running:
            try:
                time.sleep(60)  # Check every minute
                
                if not self.is_render_deployment():
                    continue
                
                uptime = (datetime.utcnow() - self.start_time).total_seconds()
                should_restart = False
                restart_reason = "scheduled_10min_cycle"
                
                # Check scheduled restart
                if uptime >= self.restart_interval:
                    should_restart = True
                    restart_reason = "scheduled_10min_cycle"
                
                # Check memory threshold
                try:
                    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    if memory_mb > self.memory_threshold and self.force_restart_on_memory:
                        should_restart = True
                        restart_reason = f"memory_threshold_exceeded_{memory_mb:.1f}MB"
                except Exception as e:
                    self.restart_logger.warning(f"⚠️ Memory check error: {e}")
                
                # Check health status
                if uptime > 60:
                    try:
                        health = self.perform_health_check()
                        
                        if health.get('overall_status') in ['critical', 'unhealthy']:
                            consecutive_health_failures += 1
                            if consecutive_health_failures >= 2:
                                should_restart = True
                                restart_reason = f"health_critical_score_{health.get('score', 0)}"
                        else:
                            consecutive_health_failures = 0
                        
                        if should_restart:
                            self.restart_logger.info(f"🔄 Auto-restart triggered - Reason: {restart_reason}")
                            self.intelligent_force_restart(restart_reason, health)
                            
                    except Exception as e:
                        self.restart_logger.error(f"❌ Health check error: {e}")
                        consecutive_health_failures += 1
                
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                self.restart_logger.error(f"❌ Auto-restart worker error: {e}")
                time.sleep(60)
    
    def health_monitor_worker(self):
        """Health monitoring worker thread"""
        self.restart_logger.info(f"📊 Health monitor started (interval: {self.health_check_interval}s)")
        
        alert_cooldown = {}
        
        while self.is_running:
            try:
                time.sleep(self.health_check_interval)
                
                health = self.perform_health_check()
                current_time = time.time()
                
                # Alert management
                if health.get('overall_status') == 'critical':
                    if 'critical' not in alert_cooldown or current_time - alert_cooldown['critical'] > 300:
                        self.restart_logger.error(f"🚨 CRITICAL HEALTH ALERT: Score {health.get('score', 0)}/100")
                        alert_cooldown['critical'] = current_time
                
                if health.get('overall_status') == 'unhealthy':
                    if 'unhealthy' not in alert_cooldown or current_time - alert_cooldown['unhealthy'] > 180:
                        self.restart_logger.warning(f"⚠️ UNHEALTHY STATUS: Score {health.get('score', 0)}/100")
                        alert_cooldown['unhealthy'] = current_time
                
                if len(health.get('warnings', [])) > 0:
                    if 'warnings' not in alert_cooldown or current_time - alert_cooldown['warnings'] > 120:
                        self.restart_logger.info(f"💡 Health warnings: {health.get('warnings', [])}")
                        alert_cooldown['warnings'] = current_time
                
                # Periodic health logging
                score = health.get('score', 0)
                if self.restart_count % 5 == 0:
                    if score >= 70:
                        self.restart_logger.info(f"💓 Health: {health['overall_status']} (Score: {score}/100)")
                    else:
                        self.restart_logger.error(f"💓 Health: {health['overall_status']} (Score: {score}/100)")
                
            except Exception as e:
                self.restart_logger.error(f"❌ Health monitor error: {e}")
                time.sleep(self.health_check_interval)
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            self.restart_logger.info(f"🔄 CineBrain received signal {signum} - graceful shutdown...")
            self.stop()
            self.aggressive_resource_cleanup()
            time.sleep(1)
            os._exit(0)
        
        # Register signal handlers
        for sig_name in ['SIGTERM', 'SIGINT', 'SIGHUP']:
            if hasattr(signal, sig_name):
                try:
                    signal.signal(getattr(signal, sig_name), signal_handler)
                except Exception as e:
                    self.restart_logger.warning(f"Failed to register {sig_name}: {e}")
        
        # Register exit handler
        atexit.register(self.aggressive_resource_cleanup)
        
        self.restart_logger.info("🛡️ Comprehensive signal handlers registered")
    
    def add_cleanup_handler(self, handler: Callable):
        """Add custom cleanup handler"""
        self.cleanup_handlers.append(handler)
        self.restart_logger.info(f"🔧 Custom cleanup handler added (total: {len(self.cleanup_handlers)})")
    
    def start(self):
        """Start the restart service"""
        if self.is_running:
            self.restart_logger.warning("⚠️ Restart service already running")
            return
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Only start on Render deployment
        if self.is_render_deployment():
            # Start auto-restart thread
            self.restart_thread = threading.Thread(
                target=self.auto_restart_worker, 
                daemon=True,
                name="CineBrainAdvancedAutoRestart"
            )
            self.restart_thread.start()
            
            # Start health monitor thread
            self.monitor_thread = threading.Thread(
                target=self.health_monitor_worker,
                daemon=True,
                name="CineBrainContinuousHealthMonitor"
            )
            self.monitor_thread.start()
            
            self.restart_logger.info("🚀 CineBrain Advanced Restart Service started for Render deployment")
            self.restart_logger.info(f"⏰ Auto-restart interval: {self.restart_interval} seconds ({self.restart_interval/60} minutes)")
            self.restart_logger.info(f"💾 Memory threshold: {self.memory_threshold} MB")
            self.restart_logger.info(f"🔄 Aggressive cleanup: {'enabled' if self.aggressive_cleanup else 'disabled'}")
        else:
            self.restart_logger.info("🏠 Development mode detected - restart service disabled")
    
    def stop(self):
        """Stop the restart service"""
        self.is_running = False
        if self.memory_optimizer:
            self.memory_optimizer.stop()
        self.restart_logger.info("🛑 CineBrain Advanced Restart Service stopped")
    
    def get_status(self):
        """Get comprehensive service status"""
        status = {
            'service_info': {
                'name': 'CineBrain Advanced Restart Service',
                'version': '2.0.0',
                'is_running': self.is_running,
                'is_render_deployment': self.is_render_deployment(),
                'start_time': self.start_time.isoformat(),
                'uptime_hours': round((datetime.utcnow() - self.start_time).total_seconds() / 3600, 2)
            },
            'restart_info': {
                'total_restarts': self.restart_count,
                'last_restart': self.last_restart_time.isoformat() if self.last_restart_time else None,
                'error_count': self.error_count,
                'last_error': self.last_error
            },
            'configuration': {
                'restart_interval_seconds': self.restart_interval,
                'restart_interval_minutes': self.restart_interval / 60,
                'memory_threshold_mb': self.memory_threshold,
                'health_check_interval_seconds': self.health_check_interval,
                'force_restart_on_memory': self.force_restart_on_memory,
                'aggressive_cleanup': self.aggressive_cleanup
            },
            'thread_status': {
                'auto_restart_thread': self.restart_thread.is_alive() if self.restart_thread else False,
                'health_monitor_thread': self.monitor_thread.is_alive() if self.monitor_thread else False,
                'cleanup_handlers_count': len(self.cleanup_handlers)
            },
            'health_summary': {
                'last_check': self.last_health_check.isoformat() if self.last_health_check else None,
                'current_status': self.health_status.get('overall_status', 'unknown'),
                'current_score': self.health_status.get('score', 0)
            },
            'system_metrics': self.get_system_metrics(),
            'performance_trends': self.performance_history[-5:] if len(self.performance_history) >= 5 else self.performance_history,
            'memory_optimizer': self.memory_optimizer.get_status() if self.memory_optimizer else None
        }
        
        return status


def init_restart_service(app, db, cache, memory_optimizer, config=None):
    """Initialize the CineBrain Advanced Restart Service"""
    try:
        # Default configuration
        default_config = {
            'restart_interval': 600,  # 10 minutes
            'memory_threshold': 200,  # 200 MB for Render free tier
            'health_check_interval': 60,  # 1 minute
            'max_restart_attempts': 5,
            'force_restart_on_memory': True,
            'aggressive_cleanup': True
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        restart_service = CineBrainAdvancedRestartService(
            app=app,
            db=db,
            cache=cache,
            memory_optimizer=memory_optimizer,
            config=default_config
        )
        
        restart_service.start()
        
        logger.info("✅ CineBrain Advanced Restart Service initialized successfully")
        return restart_service
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize restart service: {e}")
        logger.error(traceback.format_exc())
        return None