# backend/system/memory_optimizer.py

import gc
import logging
import threading
import time
import psutil
from datetime import datetime

logger = logging.getLogger('cinebrain.memory_optimizer')

class CineBrainMemoryOptimizer:
    """
    Advanced Memory Optimizer for CineBrain
    Ensures optimal memory usage on Render Free Tier
    """
    
    def __init__(self, app, db, cache, target_memory_mb=180):
        self.app = app
        self.db = db
        self.cache = cache
        self.target_memory_mb = target_memory_mb
        self.is_running = False
        self.optimization_count = 0
        self.last_optimization = None
        self.total_memory_freed = 0
        
        self.memory_logger = logging.getLogger('cinebrain.memory_optimizer')
        self.memory_logger.info(f"🧠 CineBrain Memory Optimizer initialized (target: {target_memory_mb}MB)")
        
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except Exception as e:
            self.memory_logger.error(f"Error getting memory usage: {e}")
            return 0
    
    def get_detailed_memory_info(self):
        """Get detailed memory information"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
                'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
                'percent': round(process.memory_percent(), 2),
                'available_mb': round(psutil.virtual_memory().available / 1024 / 1024, 2),
                'target_mb': self.target_memory_mb,
                'exceeds_target': memory_info.rss / 1024 / 1024 > self.target_memory_mb
            }
        except Exception as e:
            self.memory_logger.error(f"Error getting detailed memory info: {e}")
            return {}
    
    def database_cleanup(self):
        """Aggressive database connection cleanup"""
        success = False
        try:
            if self.db:
                # Close all sessions
                self.db.session.remove()
                self.db.session.close()
                
                # Dispose engine connections
                if hasattr(self.db, 'engine') and hasattr(self.db.engine, 'dispose'):
                    self.db.engine.dispose()
                
                # Force garbage collection of SQLAlchemy objects
                gc.collect()
                
                success = True
                self.memory_logger.debug("✅ Database connections cleaned")
        except Exception as e:
            self.memory_logger.warning(f"Database cleanup error: {e}")
        
        return success
    
    def cache_cleanup(self):
        """Intelligent cache cleanup"""
        success = False
        try:
            if self.cache:
                # Clear temporary and search-related cache
                temp_key_patterns = [
                    'cinebrain:temp:*',
                    'cinebrain:search:*',
                    'cinebrain:health_check:*',
                    'cinebrain:session:*'
                ]
                
                for pattern in temp_key_patterns:
                    try:
                        if hasattr(self.cache, 'delete_many'):
                            self.cache.delete_many(pattern)
                        elif hasattr(self.cache, 'clear'):
                            # Fallback to full clear if pattern deletion not available
                            self.cache.clear()
                            break
                    except Exception as e:
                        self.memory_logger.debug(f"Cache pattern {pattern} cleanup failed: {e}")
                
                success = True
                self.memory_logger.debug("✅ Cache cleaned")
        except Exception as e:
            self.memory_logger.warning(f"Cache cleanup error: {e}")
        
        return success
    
    def garbage_collection_cleanup(self):
        """Aggressive garbage collection"""
        total_collected = 0
        try:
            # Multiple rounds of garbage collection
            for generation in range(3):
                collected = gc.collect(generation)
                total_collected += collected
            
            # Force collection of all generations
            for _ in range(2):
                collected = gc.collect()
                total_collected += collected
            
            self.memory_logger.debug(f"✅ Garbage collection: {total_collected} objects collected")
            return total_collected
        except Exception as e:
            self.memory_logger.warning(f"Garbage collection error: {e}")
            return 0
    
    def system_cleanup(self):
        """System-level cleanup operations"""
        try:
            # Clear Python's internal caches
            if hasattr(gc, 'set_threshold'):
                # Temporarily lower GC thresholds for aggressive collection
                old_thresholds = gc.get_threshold()
                gc.set_threshold(50, 5, 5)
                
                # Perform collection
                gc.collect()
                
                # Restore original thresholds
                gc.set_threshold(*old_thresholds)
            
            # Clear module-level caches if available
            if hasattr(self.app, 'jinja_env'):
                try:
                    self.app.jinja_env.cache.clear()
                except:
                    pass
            
            self.memory_logger.debug("✅ System cleanup completed")
            return True
        except Exception as e:
            self.memory_logger.warning(f"System cleanup error: {e}")
            return False
    
    def aggressive_memory_cleanup(self):
        """Perform comprehensive memory cleanup"""
        memory_before = self.get_memory_usage()
        cleanup_start_time = time.time()
        
        cleanup_results = {
            'database_cleanup': False,
            'cache_cleanup': False,
            'garbage_collected': 0,
            'system_cleanup': False
        }
        
        try:
            # 1. Database cleanup
            cleanup_results['database_cleanup'] = self.database_cleanup()
            
            # 2. Cache cleanup
            cleanup_results['cache_cleanup'] = self.cache_cleanup()
            
            # 3. Aggressive garbage collection
            cleanup_results['garbage_collected'] = self.garbage_collection_cleanup()
            
            # 4. System cleanup
            cleanup_results['system_cleanup'] = self.system_cleanup()
            
            # 5. Final memory measurement
            memory_after = self.get_memory_usage()
            memory_freed = memory_before - memory_after
            cleanup_time = round((time.time() - cleanup_start_time) * 1000, 2)
            
            # Update statistics
            self.optimization_count += 1
            self.last_optimization = datetime.utcnow()
            self.total_memory_freed += max(0, memory_freed)
            
            # Log results
            self.memory_logger.info(
                f"🧹 Memory optimization #{self.optimization_count}: "
                f"Freed {memory_freed:.1f}MB in {cleanup_time}ms "
                f"(Before: {memory_before:.1f}MB → After: {memory_after:.1f}MB)"
            )
            
            if memory_freed > 10:
                self.memory_logger.info(f"✅ Significant memory freed: {memory_freed:.1f}MB")
            elif memory_freed < 0:
                self.memory_logger.warning(f"⚠️ Memory usage increased by {abs(memory_freed):.1f}MB")
            
            return memory_freed
            
        except Exception as e:
            self.memory_logger.error(f"❌ Memory cleanup error: {e}")
            return 0
    
    def memory_monitor_worker(self):
        """Background memory monitoring worker"""
        self.memory_logger.info(f"🔍 Memory monitor started (target: {self.target_memory_mb}MB, check interval: 2min)")
        
        consecutive_high_memory = 0
        
        while self.is_running:
            try:
                current_memory = self.get_memory_usage()
                
                # Check if memory exceeds target
                if current_memory > self.target_memory_mb:
                    consecutive_high_memory += 1
                    
                    self.memory_logger.warning(
                        f"💾 Memory usage high: {current_memory:.1f}MB > {self.target_memory_mb}MB "
                        f"(consecutive: {consecutive_high_memory})"
                    )
                    
                    # Perform cleanup
                    freed = self.aggressive_memory_cleanup()
                    
                    if freed > 5:
                        self.memory_logger.info(f"✅ Memory optimization successful: {freed:.1f}MB freed")
                        consecutive_high_memory = 0
                    elif consecutive_high_memory >= 3:
                        self.memory_logger.error(
                            f"🚨 PERSISTENT HIGH MEMORY: {current_memory:.1f}MB "
                            f"(failed to optimize {consecutive_high_memory} times)"
                        )
                        # Could trigger restart service here if integrated
                else:
                    consecutive_high_memory = 0
                    if self.optimization_count % 10 == 0:  # Log every 10th check when healthy
                        self.memory_logger.debug(f"💚 Memory healthy: {current_memory:.1f}MB")
                
                # Sleep for 2 minutes
                time.sleep(120)
                
            except Exception as e:
                self.memory_logger.error(f"❌ Memory monitor error: {e}")
                time.sleep(120)  # Continue monitoring even on error
    
    def start_monitoring(self):
        """Start background memory monitoring"""
        if self.is_running:
            self.memory_logger.warning("⚠️ Memory optimizer already running")
            return
        
        self.is_running = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self.memory_monitor_worker,
            daemon=True,
            name="CineBrainMemoryMonitor"
        )
        monitor_thread.start()
        
        self.memory_logger.info(f"🚀 Memory optimizer started (target: {self.target_memory_mb}MB)")
    
    def stop(self):
        """Stop memory monitoring"""
        self.is_running = False
        self.memory_logger.info("🛑 Memory optimizer stopped")
    
    def get_status(self):
        """Get comprehensive memory optimizer status"""
        memory_info = self.get_detailed_memory_info()
        
        status = {
            'service_info': {
                'is_running': self.is_running,
                'target_memory_mb': self.target_memory_mb,
                'optimization_count': self.optimization_count,
                'total_memory_freed_mb': round(self.total_memory_freed, 2),
                'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None
            },
            'current_memory': memory_info,
            'performance': {
                'average_memory_freed_per_optimization': round(
                    self.total_memory_freed / max(1, self.optimization_count), 2
                ),
                'memory_efficiency_score': min(100, max(0, 
                    100 - ((memory_info.get('rss_mb', 0) - self.target_memory_mb) / self.target_memory_mb * 100)
                )) if memory_info.get('rss_mb', 0) > 0 else 0
            },
            'health_status': {
                'status': 'healthy' if memory_info.get('rss_mb', 0) <= self.target_memory_mb else 'high_memory',
                'needs_optimization': memory_info.get('exceeds_target', False),
                'memory_pressure': 'low' if memory_info.get('rss_mb', 0) <= self.target_memory_mb * 0.8 else 
                                 'medium' if memory_info.get('rss_mb', 0) <= self.target_memory_mb else 'high'
            }
        }
        
        return status
    
    def force_optimization(self):
        """Force immediate memory optimization"""
        self.memory_logger.info("🔧 Force memory optimization triggered")
        return self.aggressive_memory_cleanup()


def init_memory_optimizer(app, db, cache, target_memory_mb=180):
    """Initialize and start the CineBrain Memory Optimizer"""
    try:
        optimizer = CineBrainMemoryOptimizer(
            app=app,
            db=db,
            cache=cache,
            target_memory_mb=target_memory_mb
        )
        
        optimizer.start_monitoring()
        
        logger.info(f"✅ CineBrain Memory Optimizer initialized successfully (target: {target_memory_mb}MB)")
        return optimizer
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize memory optimizer: {e}")
        return None