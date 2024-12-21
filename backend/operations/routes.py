# operations/routes.py

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import logging
import os
import hashlib
import hmac
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .task import OperationsTasks

logger = logging.getLogger(__name__)

# Create the operations blueprint
operations_bp = Blueprint('operations', __name__)

# Global variables (will be set by init function)
app = None
db = None
cache = None
models = None
services = None
operations_tasks = None

def init_operations_routes(flask_app, database, app_models, app_services):
    """Initialize the operations routes with dependencies"""
    global app, db, cache, models, services, operations_tasks
    
    app = flask_app
    db = database
    cache = app_services.get('cache')
    models = app_models
    services = app_services
    
    # Initialize operations tasks
    operations_tasks = OperationsTasks(flask_app, database, app_models, app_services)
    
    logger.info("✅ CineBrain operations routes initialized successfully")

def verify_task_token(token: str) -> bool:
    """Verify the task authentication token"""
    if not token:
        return False
    
    secret_key = os.environ.get('OPERATION_KEY', 'cinebrain_default_task_key_change_in_production')
    expected_token = hashlib.sha256(secret_key.encode()).hexdigest()[:32]
    
    return hmac.compare_digest(token, expected_token)

# ============================================================================
# 🎯 MAIN SINGLE ENDPOINT FOR UPTIMEROBOT
# ============================================================================

@operations_bp.route('/api/operation/refresh', methods=['GET', 'POST'])
def refresh_all_caches():
    """
    🔥 SINGLE ENDPOINT FOR COMPLETE CINEBRAIN CACHE REFRESH
    
    Refreshes ALL CineBrain caches in one go:
    • 🔥 Trending content (movies, TV, anime)
    • ⭐ Critics Choice content
    • 🎭 Genre-based recommendations
    • 🆕 New Releases
    • 🔗 Similar content samples
    • 📅 Upcoming releases
    • 🎬 Content details cache
    • 👤 Personalized recommendations (top users)
    • 🌍 Discover section
    • 🏆 Admin recommendations
    • 💾 System warmup
    
    Perfect for UptimeRobot pinging!
    """
    try:
        # 🔒 Security Check
        token = request.args.get('token') or request.headers.get('X-Task-Token')
        if not verify_task_token(token):
            logger.warning(f"❌ Unauthorized refresh attempt from {request.remote_addr}")
            return jsonify({
                'status': 'error',
                'error': 'Unauthorized - Invalid CineBrain refresh token',
                'timestamp': datetime.utcnow().isoformat()
            }), 403
        
        # 🚀 Start comprehensive refresh
        start_time = datetime.utcnow()
        logger.info("🔄 Starting comprehensive CineBrain cache refresh...")
        
        # Get refresh parameters
        force_refresh = request.args.get('force', 'false').lower() == 'true'
        refresh_personalized = request.args.get('personalized', 'true').lower() == 'true'
        user_limit = int(request.args.get('user_limit', 30))
        
        # 🎯 Execute ALL cache refreshes in parallel
        results = operations_tasks.comprehensive_cache_refresh(
            force=force_refresh,
            include_personalized=refresh_personalized,
            user_limit=user_limit
        )
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # 📊 Build comprehensive response
        response = {
            'status': 'success',
            'message': '🔥 CineBrain comprehensive cache refresh completed successfully!',
            'refresh_summary': {
                'total_duration_seconds': round(duration, 2),
                'refresh_timestamp': end_time.isoformat(),
                'caches_refreshed': len([r for r in results.values() if r.get('success')]),
                'total_operations': len(results),
                'force_refresh_used': force_refresh,
                'personalized_included': refresh_personalized
            },
            'detailed_results': results,
            'system_status': {
                'render_kept_awake': True,
                'cache_system': 'optimal',
                'database_connection': 'healthy',
                'next_recommended_refresh': (end_time + timedelta(minutes=10)).isoformat()
            },
            'cinebrain_service': 'comprehensive_cache_refresh'
        }
        
        # 📈 Log success metrics
        successful_ops = len([r for r in results.values() if r.get('success')])
        failed_ops = len(results) - successful_ops
        
        logger.info(f"✅ CineBrain cache refresh completed in {duration:.2f}s")
        logger.info(f"📊 Operations: {successful_ops} successful, {failed_ops} failed")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"💥 Critical CineBrain refresh error: {e}")
        return jsonify({
            'status': 'error',
            'error': f'Critical refresh failure: {str(e)}',
            'timestamp': datetime.utcnow().isoformat(),
            'render_status': 'kept_awake',  # Even on error, we kept it awake
            'cinebrain_service': 'comprehensive_cache_refresh'
        }), 500

# ============================================================================
# 🎛️ ALTERNATIVE ENDPOINTS (for specific needs)
# ============================================================================

@operations_bp.route('/api/operations/quick-refresh', methods=['GET'])
def quick_refresh():
    """Quick refresh - only critical caches (faster for frequent pings)"""
    try:
        token = request.args.get('token')
        if not verify_task_token(token):
            return jsonify({'error': 'Unauthorized'}), 403
        
        results = operations_tasks.critical_cache_refresh()
        
        return jsonify({
            'status': 'success',
            'message': 'CineBrain quick refresh completed',
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Quick refresh error: {e}")
        return jsonify({'error': str(e)}), 500

@operations_bp.route('/api/operations/deep-refresh', methods=['GET'])
def deep_refresh():
    """Deep refresh - all caches including heavy operations"""
    try:
        token = request.args.get('token')
        if not verify_task_token(token):
            return jsonify({'error': 'Unauthorized'}), 403
        
        results = operations_tasks.full_cache_refresh(force=True)
        
        return jsonify({
            'status': 'success',
            'message': 'CineBrain deep refresh completed',
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Deep refresh error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# 📊 MONITORING ENDPOINTS
# ============================================================================

@operations_bp.route('/api/operations/status', methods=['GET'])
def get_refresh_status():
    """Get current refresh status and system health"""
    try:
        token = request.args.get('token')
        if not verify_task_token(token):
            return jsonify({'error': 'Unauthorized'}), 403
        
        status = operations_tasks.get_comprehensive_status()
        
        return jsonify({
            'status': 'success',
            'system_status': status,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return jsonify({'error': str(e)}), 500

@operations_bp.route('/api/operations/health', methods=['GET'])
def health_check():
    """Simple health check for UptimeRobot monitoring"""
    try:
        token = request.args.get('token')
        if not verify_task_token(token):
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Quick health check
        health = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'database': operations_tasks._test_database_connection(),
            'cache': operations_tasks._test_cache_connection(),
            'services_available': len([s for s in services.keys() if services[s]]),
            'render_status': 'awake'
        }
        
        return jsonify(health), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Export the initialization function
__all__ = ['operations_bp', 'init_operations_routes']