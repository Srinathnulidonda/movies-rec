# system/routes.py

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import logging
import os
import psutil
from sqlalchemy import text

from .system import SystemService
from .admin_monitor import AdminMonitoringService

# Create the system blueprint
system_bp = Blueprint('system', __name__)

logger = logging.getLogger(__name__)

# Global variables (will be set by init function)
db = None
app = None
cache = None
models = None
services = None

def init_system_routes(flask_app, database, app_models, app_services):
    """Initialize the system routes with dependencies"""
    global db, app, cache, models, services
    
    app = flask_app
    db = database
    cache = app_services.get('cache')
    models = app_models
    services = app_services
    
    # Initialize SystemService and AdminMonitoringService
    SystemService.init(flask_app, database, app_models, app_services)
    AdminMonitoringService.init(flask_app, database, app_models, app_services)
    
    logger.info("✅ CineBrain system routes initialized successfully")

# ============================================================================
# HEALTH CHECK ROUTES
# ============================================================================

@system_bp.route('/api/health', methods=['GET'])
def health_check():
    """Comprehensive system health check"""
    try:
        health_info = SystemService.get_health_status()
        
        status_code = 200
        if health_info['status'] == 'degraded':
            status_code = 206
        elif health_info['status'] == 'unhealthy':
            status_code = 503
            
        return jsonify(health_info), status_code
        
    except Exception as e:
        logger.error(f"CineBrain health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'service': 'cinebrain_system',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/health/detailed', methods=['GET'])
def detailed_health_check():
    """Detailed health check with all system components"""
    try:
        detailed_health = SystemService.get_detailed_health_status()
        return jsonify(detailed_health), 200
        
    except Exception as e:
        logger.error(f"CineBrain detailed health check error: {e}")
        return jsonify({
            'error': 'Failed to get detailed health status',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/health/admin', methods=['GET'])
def admin_health_check():
    """Admin-specific health check"""
    try:
        admin_health = AdminMonitoringService.get_admin_health_status()
        return jsonify(admin_health), 200
        
    except Exception as e:
        logger.error(f"CineBrain admin health check error: {e}")
        return jsonify({
            'error': 'Failed to get admin health status',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# ============================================================================
# ADMIN MONITORING ROUTES
# ============================================================================

@system_bp.route('/api/admin/monitoring/overview', methods=['GET'])
def admin_monitoring_overview():
    """Get admin monitoring overview"""
    try:
        overview = AdminMonitoringService.get_monitoring_overview()
        return jsonify(overview), 200
        
    except Exception as e:
        logger.error(f"Admin monitoring overview error: {e}")
        return jsonify({
            'error': 'Failed to get admin monitoring overview',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/admin/monitoring/activity', methods=['GET'])
def admin_activity_monitoring():
    """Get admin activity monitoring data"""
    try:
        activity = AdminMonitoringService.get_admin_activity()
        return jsonify(activity), 200
        
    except Exception as e:
        logger.error(f"Admin activity monitoring error: {e}")
        return jsonify({
            'error': 'Failed to get admin activity data',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/admin/monitoring/performance', methods=['GET'])
def admin_performance_monitoring():
    """Get admin performance metrics"""
    try:
        performance = AdminMonitoringService.get_admin_performance()
        return jsonify(performance), 200
        
    except Exception as e:
        logger.error(f"Admin performance monitoring error: {e}")
        return jsonify({
            'error': 'Failed to get admin performance metrics',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/admin/monitoring/security', methods=['GET'])
def admin_security_monitoring():
    """Get admin security monitoring data"""
    try:
        security = AdminMonitoringService.get_security_monitoring()
        return jsonify(security), 200
        
    except Exception as e:
        logger.error(f"Admin security monitoring error: {e}")
        return jsonify({
            'error': 'Failed to get admin security data',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/admin/monitoring/notifications', methods=['GET'])
def admin_notifications_monitoring():
    """Get admin notifications system health"""
    try:
        notifications = AdminMonitoringService.get_notification_system_health()
        return jsonify(notifications), 200
        
    except Exception as e:
        logger.error(f"Admin notifications monitoring error: {e}")
        return jsonify({
            'error': 'Failed to get admin notifications data',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/admin/monitoring/alerts', methods=['GET'])
def admin_monitoring_alerts():
    """Get admin-specific monitoring alerts"""
    try:
        alerts = AdminMonitoringService.get_admin_alerts()
        return jsonify(alerts), 200
        
    except Exception as e:
        logger.error(f"Admin alerts monitoring error: {e}")
        return jsonify({
            'error': 'Failed to get admin alerts',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# ============================================================================
# PERFORMANCE MONITORING ROUTES
# ============================================================================

@system_bp.route('/api/performance', methods=['GET'])
def performance_check():
    """System performance metrics"""
    try:
        performance_data = SystemService.get_performance_metrics()
        return jsonify(performance_data), 200
        
    except Exception as e:
        logger.error(f"CineBrain performance check error: {e}")
        return jsonify({
            'error': 'Failed to get performance metrics',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/performance/detailed', methods=['GET'])
def detailed_performance():
    """Detailed performance metrics"""
    try:
        detailed_performance = SystemService.get_detailed_performance_metrics()
        return jsonify(detailed_performance), 200
        
    except Exception as e:
        logger.error(f"CineBrain detailed performance error: {e}")
        return jsonify({
            'error': 'Failed to get detailed performance metrics',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/performance/admin', methods=['GET'])
def admin_performance():
    """Admin-specific performance metrics"""
    try:
        admin_perf = AdminMonitoringService.get_detailed_admin_performance()
        return jsonify(admin_perf), 200
        
    except Exception as e:
        logger.error(f"Admin performance error: {e}")
        return jsonify({
            'error': 'Failed to get admin performance metrics',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# ============================================================================
# SYSTEM STATISTICS ROUTES
# ============================================================================

@system_bp.route('/api/system/stats', methods=['GET'])
def system_stats():
    """Get comprehensive system statistics"""
    try:
        stats = SystemService.get_system_statistics()
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"CineBrain system stats error: {e}")
        return jsonify({
            'error': 'Failed to get system statistics',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/system/database', methods=['GET'])
def database_stats():
    """Get database statistics and health"""
    try:
        db_stats = SystemService.get_database_statistics()
        return jsonify(db_stats), 200
        
    except Exception as e:
        logger.error(f"CineBrain database stats error: {e}")
        return jsonify({
            'error': 'Failed to get database statistics',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/system/services', methods=['GET'])
def services_status():
    """Get status of all CineBrain services"""
    try:
        services_status = SystemService.get_services_status()
        return jsonify(services_status), 200
        
    except Exception as e:
        logger.error(f"CineBrain services status error: {e}")
        return jsonify({
            'error': 'Failed to get services status',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# ============================================================================
# SYSTEM MAINTENANCE ROUTES
# ============================================================================

@system_bp.route('/api/system/cache/clear', methods=['POST'])
def clear_cache():
    """Clear system cache"""
    try:
        if cache:
            cache.clear()
            return jsonify({
                'success': True,
                'message': 'CineBrain cache cleared successfully',
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'CineBrain cache not configured',
                'timestamp': datetime.utcnow().isoformat()
            }), 400
            
    except Exception as e:
        logger.error(f"CineBrain cache clear error: {e}")
        return jsonify({
            'error': 'Failed to clear cache',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/system/cache/stats', methods=['GET'])
def cache_stats():
    """Get cache statistics"""
    try:
        cache_data = SystemService.get_cache_statistics()
        return jsonify(cache_data), 200
        
    except Exception as e:
        logger.error(f"CineBrain cache stats error: {e}")
        return jsonify({
            'error': 'Failed to get cache statistics',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# ============================================================================
# MONITORING ROUTES
# ============================================================================

@system_bp.route('/api/system/monitoring/alerts', methods=['GET'])
def monitoring_alerts():
    """Get system monitoring alerts"""
    try:
        alerts = SystemService.get_monitoring_alerts()
        return jsonify(alerts), 200
        
    except Exception as e:
        logger.error(f"CineBrain monitoring alerts error: {e}")
        return jsonify({
            'error': 'Failed to get monitoring alerts',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/system/monitoring/metrics', methods=['GET'])
def monitoring_metrics():
    """Get real-time monitoring metrics"""
    try:
        metrics = SystemService.get_real_time_metrics()
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"CineBrain monitoring metrics error: {e}")
        return jsonify({
            'error': 'Failed to get monitoring metrics',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# ============================================================================
# CLI OPERATIONS ROUTES
# ============================================================================

@system_bp.route('/api/system/cli/status', methods=['GET'])
def cli_operations_status():
    """Get status of CLI operations"""
    try:
        cli_status = SystemService.get_cli_operations_status()
        return jsonify(cli_status), 200
        
    except Exception as e:
        logger.error(f"CineBrain CLI status error: {e}")
        return jsonify({
            'error': 'Failed to get CLI operations status',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# ============================================================================
# SYSTEM INFORMATION ROUTES
# ============================================================================

@system_bp.route('/api/system/info', methods=['GET'])
def system_info():
    """Get comprehensive system information"""
    try:
        sys_info = SystemService.get_system_info()
        return jsonify(sys_info), 200
        
    except Exception as e:
        logger.error(f"CineBrain system info error: {e}")
        return jsonify({
            'error': 'Failed to get system information',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@system_bp.route('/api/system/version', methods=['GET'])
def version_info():
    """Get CineBrain version and build information"""
    try:
        version_data = SystemService.get_version_info()
        return jsonify(version_data), 200
        
    except Exception as e:
        logger.error(f"CineBrain version info error: {e}")
        return jsonify({
            'error': 'Failed to get version information',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Export the initialization function
__all__ = ['system_bp', 'init_system_routes']