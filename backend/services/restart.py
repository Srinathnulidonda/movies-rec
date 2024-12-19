# backend/services/restart.py

"""
CineBrain Advanced Restart & Keep-Alive Service
Optimized for Render Free Tier Deployment
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
import sqlite3

logger = logging.getLogger('cinebrain.restart')

class CineBrainRestartService:
    def __init__(self, app=None, db=None, cache=None, config=None):
        self.app = app
        self.db = db
        self.cache = cache
        self.config = config or {}
        
        self.restart_interval = self.config.get('restart_interval', 600)
        self.memory_threshold = self.config.get('memory_threshold', 300)
        self.health_check_interval = self.config.get('health_check_interval', 60)
        self.max_restart_attempts = self.config.get('max_restart_attempts', 3)
        self.force_restart_on_memory = self.config.get('force_restart_on_memory', True)
        self.aggressive_cleanup = self.config.get('aggressive_cleanup', True)
        
        self.restart_count = 0
        self.start_time = datetime.utcnow()
        self.last_health_check = None
        self.last_restart_time = None
        self.is_running = False
        self.health_status = {}
        self.performance_metrics = {}
        self.error_count = 0
        self.last_error = None
        
        self.restart_thread = None
        self.monitor_thread = None
        self.cleanup_handlers = []
        self.performance_history = []
        
        self.session = self._create_http_session()
        
        logger.info("🚀 CineBrain Advanced Restart Service initialized")
    
    def _create_http_session(self):
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
        return (
            os.environ.get('PORT') is not None and 
            os.environ.get('FLASK_ENV') != 'development' and
            (os.environ.get('RENDER') is not None or 
             'render' in os.environ.get('HOSTNAME', '').lower() or
             'render.com' in os.environ.get('EXTERNAL_URL', ''))
        )
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_times = process.cpu_times()
            
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
                    'user_time': cpu_times.user,
                    'system_time': cpu_times.system
                },
                'process': {
                    'pid': process.pid,
                    'threads': process.num_threads(),
                    'connections': len(process.connections()) if hasattr(process, 'connections') else 0,
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
            
            if self.db:
                try:
                    start_time = time.time()
                    self.db.session.execute('SELECT 1').fetchone()
                    db_time = round((time.time() - start_time) * 1000, 2)
                    metrics['database'] = {
                        'status': 'connected',
                        'response_time_ms': db_time,
                        'pool_size': getattr(self.db.engine.pool, 'size', 'unknown'),
                        'checked_out': getattr(self.db.engine.pool, 'checkedout', 'unknown')
                    }
                except Exception as e:
                    metrics['database'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            if self.cache:
                try:
                    start_time = time.time()
                    test_key = f'restart_health_{int(time.time())}'
                    self.cache.set(test_key, 'test', timeout=5)
                    result = self.cache.get(test_key)
                    cache_time = round((time.time() - start_time) * 1000, 2)
                    
                    metrics['cache'] = {
                        'status': 'connected' if result == 'test' else 'degraded',
                        'response_time_ms': cache_time,
                        'type': self.app.config.get('CACHE_TYPE') if self.app else 'unknown'
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
            
            try:
                disk_usage = psutil.disk_usage('/')
                metrics['disk'] = {
                    'total_gb': round(disk_usage.total / 1024 / 1024 / 1024, 2),
                    'used_gb': round(disk_usage.used / 1024 / 1024 / 1024, 2),
                    'free_gb': round(disk_usage.free / 1024 / 1024 / 1024, 2),
                    'percent_used': round((disk_usage.used / disk_usage.total) * 100, 2)
                }
            except:
                metrics['disk'] = {'status': 'unavailable'}
            
            return metrics
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"❌ Error getting comprehensive metrics: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'restart_count': self.restart_count
            }
    
    def perform_intensive_health_check(self) -> Dict[str, Any]:
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
            
            if self.app:
                port = os.environ.get('PORT', 5000)
                try:
                    response = self.session.get(
                        f"http://localhost:{port}/api/health",
                        timeout=5
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
            
            if self.db:
                try:
                    start_time = time.time()
                    result = self.db.session.execute('SELECT COUNT(*) as count FROM user').fetchone()
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
            
            if self.cache:
                try:
                    start_time = time.time()
                    test_key = f'health_intensive_{int(time.time())}'
                    self.cache.set(test_key, 'intensive_test', timeout=10)
                    result = self.cache.get(test_key)
                    cache_time = round((time.time() - start_time) * 1000, 2)
                    
                    if result == 'intensive_test':
                        health_status['checks']['cache'] = {
                            'status': 'healthy',
                            'operation_time_ms': cache_time
                        }
                    else:
                        health_status['checks']['cache'] = {
                            'status': 'degraded',
                            'operation_time_ms': cache_time,
                            'issue': 'cache_read_failed'
                        }
                        health_status['warnings'].append('Cache read/write issues')
                        score -= 15
                        
                    try:
                        self.cache.delete(test_key)
                    except:
                        pass
                        
                except Exception as e:
                    health_status['checks']['cache'] = {
                        'status': 'degraded',
                        'error': str(e)
                    }
                    health_status['warnings'].append('Cache connection issues')
                    score -= 10
            
            metrics = self.get_comprehensive_metrics()
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
            
            health_status['comprehensive_metrics'] = metrics
            
            self.last_health_check = datetime.utcnow()
            self.health_status = health_status
            
            if len(self.performance_history) > 50:
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
            logger.error(f"❌ Intensive health check error: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_status': 'error',
                'error': str(e),
                'score': 0
            }
    
    def aggressive_resource_cleanup(self):
        logger.info("🧹 CineBrain performing aggressive resource cleanup...")
        
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
        
        if self.db:
            try:
                self.db.session.remove()
                self.db.session.close()
                
                if hasattr(self.db, 'engine') and hasattr(self.db.engine, 'dispose'):
                    self.db.engine.dispose()
                
                cleanup_results['successful'].append('database_connections_closed')
                logger.info("✅ Database connections aggressively closed")
            except Exception as e:
                cleanup_results['failed'].append(f'database: {str(e)}')
                logger.warning(f"⚠️ Database cleanup error: {e}")
        
        if self.cache and self.aggressive_cleanup:
            try:
                cache_keys_to_clear = [
                    'cinebrain:health_check:*',
                    'cinebrain:temp:*',
                    'cinebrain:session:*',
                    'cinebrain:search:*',
                    'cinebrain:recommendations:temp:*'
                ]
                
                if hasattr(self.cache, 'clear'):
                    try:
                        self.cache.clear()
                        cleanup_results['successful'].append('cache_completely_cleared')
                    except:
                        cleanup_results['successful'].append('cache_partial_cleanup')
                
                cleanup_results['successful'].append('cache_cleanup')
                logger.info("✅ Cache aggressively cleaned")
            except Exception as e:
                cleanup_results['failed'].append(f'cache: {str(e)}')
                logger.warning(f"⚠️ Cache cleanup error: {e}")
        
        try:
            if hasattr(self, 'session') and self.session:
                self.session.close()
            cleanup_results['successful'].append('http_session_closed')
        except Exception as e:
            cleanup_results['failed'].append(f'http_session: {str(e)}')
        
        try:
            collected = gc.collect()
            cleanup_results['successful'].append(f'garbage_collection_{collected}_objects')
            logger.info(f"✅ Aggressive garbage collection: {collected} objects collected")
        except Exception as e:
            cleanup_results['failed'].append(f'garbage_collection: {str(e)}')
        
        try:
            import threading
            active_threads = threading.active_count()
            cleanup_results['successful'].append(f'active_threads_{active_threads}')
        except Exception as e:
            cleanup_results['failed'].append(f'thread_check: {str(e)}')
        
        for handler in self.cleanup_handlers:
            try:
                handler()
                cleanup_results['successful'].append('custom_handler')
            except Exception as e:
                cleanup_results['failed'].append(f'custom_handler: {str(e)}')
                logger.warning(f"⚠️ Custom cleanup handler error: {e}")
        
        try:
            cleanup_results['memory_after_mb'] = round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
            memory_freed = cleanup_results['memory_before_mb'] - cleanup_results['memory_after_mb']
            cleanup_results['memory_freed_mb'] = memory_freed
        except:
            pass
        
        cleanup_results['cleanup_time_ms'] = round((time.time() - start_time) * 1000, 2)
        
        logger.info(f"🎯 Aggressive cleanup completed in {cleanup_results['cleanup_time_ms']}ms")
        logger.info(f"📊 Memory freed: {cleanup_results.get('memory_freed_mb', 0)} MB")
        logger.info(f"✅ Successful: {len(cleanup_results['successful'])} | ❌ Failed: {len(cleanup_results['failed'])}")
        
        return cleanup_results
    
    def intelligent_force_restart(self, reason="scheduled", health_data=None):
        self.restart_count += 1
        self.last_restart_time = datetime.utcnow()
        
        restart_info = {
            'restart_number': self.restart_count,
            'reason': reason,
            'timestamp': self.last_restart_time.isoformat(),
            'uptime_minutes': round((self.last_restart_time - self.start_time).total_seconds() / 60, 2),
            'memory_usage_mb': round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
            'health_score': health_data.get('score', 0) if health_data else 0,
            'critical_issues': health_data.get('critical_issues', []) if health_data else [],
            'warnings': health_data.get('warnings', []) if health_data else []
        }
        
        logger.info(f"🔄 CineBrain Intelligent Force Restart #{self.restart_count}")
        logger.info(f"📊 Restart Details: {json.dumps(restart_info, indent=2)}")
        
        cleanup_results = self.aggressive_resource_cleanup()
        
        logger.info(f"💫 CineBrain restarting due to: {reason}")
        logger.info(f"⏱️ Uptime: {restart_info['uptime_minutes']} minutes")
        logger.info(f"💾 Memory usage: {restart_info['memory_usage_mb']} MB")
        logger.info(f"🏥 Health score: {restart_info['health_score']}/100")
        
        if cleanup_results.get('memory_freed_mb', 0) > 0:
            logger.info(f"🆓 Memory freed: {cleanup_results['memory_freed_mb']} MB")
        
        time.sleep(3)
        
        logger.info("🚀 CineBrain initiating intelligent restart now...")
        os._exit(0)
    
    def advanced_restart_worker(self):
        logger.info(f"🔄 Advanced auto-restart worker started (interval: {self.restart_interval}s)")
        
        consecutive_health_failures = 0
        last_memory_check = time.time()
        
        while self.is_running:
            try:
                time.sleep(min(60, self.restart_interval // 10))
                
                if not self.is_render_deployment():
                    logger.debug("🏠 Development mode - skipping auto-restart")
                    continue
                
                current_time = datetime.utcnow()
                uptime = (current_time - self.start_time).total_seconds()
                
                should_restart = False
                restart_reason = "scheduled_10min_cycle"
                
                if uptime >= self.restart_interval:
                    should_restart = True
                    restart_reason = "scheduled_10min_cycle"
                
                if time.time() - last_memory_check > 30:
                    try:
                        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                        if memory_mb > self.memory_threshold and self.force_restart_on_memory:
                            should_restart = True
                            restart_reason = f"memory_threshold_exceeded_{memory_mb:.1f}MB"
                        last_memory_check = time.time()
                    except Exception as e:
                        logger.warning(f"⚠️ Memory check error: {e}")
                
                if uptime > 60:
                    try:
                        health = self.perform_intensive_health_check()
                        
                        if health.get('overall_status') in ['critical', 'unhealthy']:
                            consecutive_health_failures += 1
                            if consecutive_health_failures >= 2:
                                should_restart = True
                                restart_reason = f"health_critical_score_{health.get('score', 0)}"
                        else:
                            consecutive_health_failures = 0
                        
                        if should_restart:
                            logger.info(f"🔄 Auto-restart triggered - Reason: {restart_reason}")
                            self.intelligent_force_restart(restart_reason, health)
                            
                    except Exception as e:
                        logger.error(f"❌ Health check in restart worker failed: {e}")
                        consecutive_health_failures += 1
                
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                logger.error(f"❌ Advanced restart worker error: {e}")
                logger.error(traceback.format_exc())
                time.sleep(60)
    
    def continuous_health_monitor_worker(self):
        logger.info(f"📊 Continuous health monitor started (interval: {self.health_check_interval}s)")
        
        health_history = []
        alert_cooldown = {}
        
        while self.is_running:
            try:
                time.sleep(self.health_check_interval)
                
                health = self.perform_intensive_health_check()
                health_history.append(health)
                
                if len(health_history) > 20:
                    health_history.pop(0)
                
                current_time = time.time()
                
                if health.get('overall_status') == 'critical':
                    if 'critical' not in alert_cooldown or current_time - alert_cooldown['critical'] > 300:
                        logger.error(f"🚨 CRITICAL HEALTH ALERT: {health.get('critical_issues', [])}")
                        alert_cooldown['critical'] = current_time
                
                if health.get('overall_status') == 'unhealthy':
                    if 'unhealthy' not in alert_cooldown or current_time - alert_cooldown['unhealthy'] > 180:
                        logger.warning(f"⚠️ UNHEALTHY STATUS: Score {health.get('score', 0)}/100")
                        alert_cooldown['unhealthy'] = current_time
                
                if len(health.get('warnings', [])) > 0:
                    if 'warnings' not in alert_cooldown or current_time - alert_cooldown['warnings'] > 120:
                        logger.info(f"💡 Health warnings: {health.get('warnings', [])}")
                        alert_cooldown['warnings'] = current_time
                
                score = health.get('score', 0)
                if score >= 90:
                    log_level = "DEBUG"
                elif score >= 70:
                    log_level = "INFO"
                elif score >= 50:
                    log_level = "WARNING"
                else:
                    log_level = "ERROR"
                
                if self.restart_count % 5 == 0:
                    getattr(logger, log_level.lower())(f"💓 Health: {health['overall_status']} (Score: {score}/100)")
                
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                logger.error(f"❌ Health monitor error: {e}")
                time.sleep(self.health_check_interval)
    
    def setup_comprehensive_signal_handlers(self):
        def signal_handler(signum, frame):
            logger.info(f"🔄 CineBrain received signal {signum} - preparing for graceful shutdown...")
            self.stop()
            self.aggressive_resource_cleanup()
            time.sleep(1)
            os._exit(0)
        
        for sig_name in ['SIGTERM', 'SIGINT', 'SIGHUP', 'SIGUSR1', 'SIGUSR2']:
            if hasattr(signal, sig_name):
                signal.signal(getattr(signal, sig_name), signal_handler)
        
        atexit.register(self.aggressive_resource_cleanup)
        
        logger.info("🛡️ Comprehensive signal handlers registered")
    
    def add_cleanup_handler(self, handler: Callable):
        self.cleanup_handlers.append(handler)
        logger.info(f"🔧 Custom cleanup handler added (total: {len(self.cleanup_handlers)})")
    
    def start(self):
        if self.is_running:
            logger.warning("⚠️ Restart service already running")
            return
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        self.setup_comprehensive_signal_handlers()
        
        if self.is_render_deployment():
            self.restart_thread = threading.Thread(
                target=self.advanced_restart_worker, 
                daemon=True,
                name="CineBrainAdvancedAutoRestart"
            )
            self.restart_thread.start()
            
            self.monitor_thread = threading.Thread(
                target=self.continuous_health_monitor_worker,
                daemon=True,
                name="CineBrainContinuousHealthMonitor"
            )
            self.monitor_thread.start()
            
            logger.info("🚀 CineBrain Advanced Restart Service started for Render deployment")
            logger.info(f"⏰ Auto-restart interval: {self.restart_interval} seconds ({self.restart_interval/60} minutes)")
            logger.info(f"💾 Memory threshold: {self.memory_threshold} MB")
            logger.info(f"🔄 Aggressive cleanup: {'enabled' if self.aggressive_cleanup else 'disabled'}")
        else:
            logger.info("🏠 Development mode detected - restart service disabled")
    
    def stop(self):
        self.is_running = False
        logger.info("🛑 CineBrain Advanced Restart Service stopped")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
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
                'aggressive_cleanup': self.aggressive_cleanup,
                'max_restart_attempts': self.max_restart_attempts
            },
            'thread_status': {
                'auto_restart_thread': self.restart_thread.is_alive() if self.restart_thread else False,
                'health_monitor_thread': self.monitor_thread.is_alive() if self.monitor_thread else False,
                'cleanup_handlers_count': len(self.cleanup_handlers)
            },
            'health_summary': {
                'last_check': self.last_health_check.isoformat() if self.last_health_check else None,
                'current_status': self.health_status.get('overall_status', 'unknown'),
                'current_score': self.health_status.get('score', 0),
                'performance_history_length': len(self.performance_history)
            },
            'system_metrics': self.get_comprehensive_metrics(),
            'performance_trends': self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        }
        
        return status


_restart_service = None

def init_restart_service(app, db=None, cache=None, config=None):
    global _restart_service
    
    if _restart_service is not None:
        logger.warning("⚠️ Advanced restart service already initialized")
        return _restart_service
    
    try:
        default_config = {
            'restart_interval': 600,
            'memory_threshold': 280,
            'health_check_interval': 45,
            'max_restart_attempts': 5,
            'force_restart_on_memory': True,
            'aggressive_cleanup': True
        }
        
        if config:
            default_config.update(config)
        
        _restart_service = CineBrainRestartService(
            app=app,
            db=db,
            cache=cache,
            config=default_config
        )
        
        @app.route('/api/restart/status', methods=['GET'])
        def restart_service_status():
            try:
                status = _restart_service.get_comprehensive_status()
                return status, 200
            except Exception as e:
                return {
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat(),
                    'service': 'restart_status'
                }, 500
        
        @app.route('/api/restart/health', methods=['GET'])
        def restart_service_health():
            try:
                health = _restart_service.perform_intensive_health_check()
                return health, 200
            except Exception as e:
                return {
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat(),
                    'service': 'restart_health'
                }, 500
        
        @app.route('/api/restart/metrics', methods=['GET'])
        def restart_service_metrics():
            try:
                metrics = _restart_service.get_comprehensive_metrics()
                return metrics, 200
            except Exception as e:
                return {
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat(),
                    'service': 'restart_metrics'
                }, 500
        
        @app.route('/api/restart/force', methods=['POST'])
        def force_restart_endpoint():
            try:
                data = request.get_json() or {}
                reason = data.get('reason', 'manual_admin_trigger')
                
                health = _restart_service.perform_intensive_health_check()
                _restart_service.intelligent_force_restart(reason, health)
                
                return {'message': 'Intelligent restart initiated', 'reason': reason}, 200
            except Exception as e:
                return {'error': str(e), 'service': 'force_restart'}, 500
        
        @app.route('/api/restart/cleanup', methods=['POST'])
        def force_cleanup_endpoint():
            try:
                cleanup_results = _restart_service.aggressive_resource_cleanup()
                return {
                    'message': 'Aggressive cleanup completed',
                    'results': cleanup_results
                }, 200
            except Exception as e:
                return {'error': str(e), 'service': 'force_cleanup'}, 500
        
        _restart_service.start()
        
        logger.info("✅ CineBrain Advanced Restart Service initialized successfully")
        return _restart_service
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize advanced restart service: {e}")
        logger.error(traceback.format_exc())
        return None

def get_restart_service():
    return _restart_service

def cleanup_restart_service():
    global _restart_service
    if _restart_service:
        _restart_service.stop()
        _restart_service.aggressive_resource_cleanup()
        _restart_service = None
        logger.info("🧹 Advanced restart service cleaned up")