# admin/routes.py

from flask import Blueprint, request, jsonify
from datetime import datetime
import json
import logging
import jwt
from functools import wraps

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__)

# Global services - will be initialized by init_admin_routes
admin_service = None
dashboard_service = None
telegram_service = None
app = None
db = None
User = None
Content = None
UserInteraction = None
AdminRecommendation = None
cache = None

def get_user_from_token():
    """Extract user from JWT token"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    try:
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        return User.query.get(payload.get('user_id'))
    except:
        return None

def require_admin(f):
    """Decorator for admin-only endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_user_from_token()
        if not user or not getattr(user, 'is_admin', False):
            return jsonify({'error': 'Admin access required'}), 403
        return f(user, *args, **kwargs)
    return decorated_function

# Content Management Routes
@admin_bp.route('/api/admin/search', methods=['GET'])
@require_admin
def admin_search(current_user):
    """Search for content in external APIs"""
    try:
        query = request.args.get('query', '')
        source = request.args.get('source', 'tmdb')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        results = admin_service.search_external_content(query, source, page)
        return jsonify({'results': results}), 200
        
    except Exception as e:
        logger.error(f"Admin search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@admin_bp.route('/api/admin/content', methods=['POST'])
@require_admin
def save_external_content(current_user):
    """Save external content to database"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No content data provided'}), 400
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.save_external_content(data)
        return jsonify(result), 201 if result.get('created') else 200
        
    except Exception as e:
        logger.error(f"Save content error: {e}")
        return jsonify({'error': 'Failed to process content'}), 500

# Admin Recommendations
@admin_bp.route('/api/admin/recommendations', methods=['POST'])
@require_admin
def create_admin_recommendation(current_user):
    """Create admin recommendation"""
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'recommendation_type', 'description']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.create_recommendation(
            current_user, 
            data['content_id'],
            data['recommendation_type'],
            data['description']
        )
        
        return jsonify(result), 201
        
    except Exception as e:
        logger.error(f"Admin recommendation error: {e}")
        return jsonify({'error': 'Failed to create recommendation'}), 500

@admin_bp.route('/api/admin/recommendations', methods=['GET'])
@require_admin
def get_admin_recommendations(current_user):
    """Get admin recommendations"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.get_recommendations(page, per_page)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Get admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@admin_bp.route('/api/admin/recommendations/<int:recommendation_id>/publish', methods=['POST'])
@require_admin
def publish_recommendation(current_user, recommendation_id):
    """Publish a draft recommendation with comprehensive tracking"""
    try:
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        data = request.get_json() or {}
        
        # Optional parameters for enhanced publishing
        options = {
            'notify_users': data.get('notify_users', True),
            'schedule_time': data.get('schedule_time'),  # ISO format datetime
            'priority': data.get('priority', 'normal'),  # normal, high, urgent
            'tags': data.get('tags', []),
            'target_audience': data.get('target_audience', 'all'),  # all, regional, genre-specific
            'publish_to': data.get('publish_to', ['telegram', 'website'])  # channels to publish
        }
        
        result = admin_service.publish_recommendation(
            recommendation_id, 
            current_user,
            options
        )
        
        if result.get('error'):
            return jsonify(result), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Publish recommendation error: {e}")
        return jsonify({
            'error': 'Failed to publish recommendation',
            'details': str(e) if app.debug else None
        }), 500

@admin_bp.route('/api/admin/recommendations/publish-batch', methods=['POST'])
@require_admin
def publish_recommendations_batch(current_user):
    """Batch publish multiple recommendations"""
    try:
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        data = request.get_json()
        if not data or 'recommendation_ids' not in data:
            return jsonify({'error': 'recommendation_ids required'}), 400
        
        recommendation_ids = data['recommendation_ids']
        if not isinstance(recommendation_ids, list) or not recommendation_ids:
            return jsonify({'error': 'recommendation_ids must be a non-empty list'}), 400
        
        options = {
            'notify_users': data.get('notify_users', True),
            'stagger_delay': data.get('stagger_delay', 60),  # seconds between posts
            'priority': data.get('priority', 'normal'),
            'publish_to': data.get('publish_to', ['telegram', 'website'])
        }
        
        result = admin_service.publish_recommendations_batch(
            recommendation_ids,
            current_user,
            options
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Batch publish error: {e}")
        return jsonify({'error': 'Failed to batch publish recommendations'}), 500

# Dashboard and Analytics Routes
@admin_bp.route('/api/admin/dashboard', methods=['GET'])
@require_admin
def get_admin_dashboard(current_user):
    """Get admin dashboard data"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503
        
        dashboard_data = dashboard_service.get_overview()
        return jsonify(dashboard_data), 200
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return jsonify({'error': 'Failed to load dashboard'}), 500

@admin_bp.route('/api/admin/analytics', methods=['GET'])
@require_admin
def get_analytics(current_user):
    """Get detailed analytics"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503
        
        analytics_data = dashboard_service.get_analytics()
        return jsonify(analytics_data), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

@admin_bp.route('/api/admin/system-health', methods=['GET'])
@require_admin
def get_system_health(current_user):
    """Get system health status"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503
        
        health_data = dashboard_service.get_system_health()
        return jsonify(health_data), 200
        
    except Exception as e:
        logger.error(f"System health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Support Management Routes
@admin_bp.route('/api/admin/support/dashboard', methods=['GET'])
@require_admin
def get_support_dashboard(current_user):
    """Get support dashboard data"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503
        
        support_data = dashboard_service.get_support_dashboard()
        return jsonify(support_data), 200
        
    except Exception as e:
        logger.error(f"Support dashboard error: {e}")
        return jsonify({'error': 'Failed to load support dashboard'}), 500

@admin_bp.route('/api/admin/support/tickets', methods=['GET'])
@require_admin
def get_support_tickets(current_user):
    """Get support tickets list"""
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        status = request.args.get('status')
        priority = request.args.get('priority')
        category_id = request.args.get('category_id', type=int)
        search = request.args.get('search', '').strip()
        
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503
        
        tickets_data = dashboard_service.get_support_tickets(
            page, per_page, status, priority, category_id, search
        )
        return jsonify(tickets_data), 200
        
    except Exception as e:
        logger.error(f"Get support tickets error: {e}")
        return jsonify({'error': 'Failed to get support tickets'}), 500

@admin_bp.route('/api/admin/support/feedback', methods=['GET'])
@require_admin
def get_feedback_list(current_user):
    """Get feedback list"""
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        feedback_type = request.args.get('type')
        is_read = request.args.get('is_read')
        search = request.args.get('search', '').strip()
        
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503
        
        feedback_data = dashboard_service.get_feedback_list(
            page, per_page, feedback_type, is_read, search
        )
        return jsonify(feedback_data), 200
        
    except Exception as e:
        logger.error(f"Get feedback list error: {e}")
        return jsonify({'error': 'Failed to get feedback list'}), 500

# User Management Routes
@admin_bp.route('/api/admin/users', methods=['GET'])
@require_admin
def get_users_management(current_user):
    """Get users management data"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        search = request.args.get('search', '')
        
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503
        
        users_data = dashboard_service.get_users_management(page, per_page, search)
        return jsonify(users_data), 200
        
    except Exception as e:
        logger.error(f"Users management error: {e}")
        return jsonify({'error': 'Failed to get users'}), 500

# Content Management Routes
@admin_bp.route('/api/admin/content/manage', methods=['GET'])
@require_admin
def get_content_management(current_user):
    """Get content management data"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        content_type = request.args.get('type', 'all')
        search = request.args.get('search', '')
        
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503
        
        content_data = dashboard_service.get_content_management(
            page, per_page, content_type, search
        )
        return jsonify(content_data), 200
        
    except Exception as e:
        logger.error(f"Content management error: {e}")
        return jsonify({'error': 'Failed to get content'}), 500

# Notification Routes
@admin_bp.route('/api/admin/notifications', methods=['GET'])
@require_admin
def get_admin_notifications(current_user):
    """Get admin notifications"""
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        unread_only = request.args.get('unread_only', 'false').lower() == 'true'
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        notifications_data = admin_service.get_notifications(page, per_page, unread_only)
        return jsonify(notifications_data), 200
        
    except Exception as e:
        logger.error(f"Get admin notifications error: {e}")
        return jsonify({'error': 'Failed to get notifications'}), 500

@admin_bp.route('/api/admin/notifications/mark-all-read', methods=['PUT'])
@require_admin
def mark_all_notifications_read(current_user):
    """Mark all notifications as read"""
    try:
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.mark_all_notifications_read()
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Mark all notifications read error: {e}")
        return jsonify({'error': 'Failed to mark all notifications as read'}), 500

# Cache Management Routes
@admin_bp.route('/api/admin/cache/clear', methods=['POST'])
@require_admin
def clear_cache(current_user):
    """Clear application cache"""
    try:
        cache_type = request.args.get('type', 'all')
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.clear_cache(cache_type)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return jsonify({'error': 'Failed to clear cache'}), 500

@admin_bp.route('/api/admin/cache/stats', methods=['GET'])
@require_admin
def get_cache_stats(current_user):
    """Get cache statistics"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503
        
        cache_stats = dashboard_service.get_cache_stats()
        return jsonify(cache_stats), 200
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return jsonify({'error': 'Failed to get cache stats'}), 500

# Slug Management Routes (moved from app.py)
@admin_bp.route('/api/admin/slugs/migrate', methods=['POST'])
@require_admin
def migrate_all_slugs(current_user):
    """Migrate all content slugs"""
    try:
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        data = request.get_json() or {}
        batch_size = int(data.get('batch_size', 50))
        
        result = admin_service.migrate_all_slugs(batch_size)
        
        return jsonify({
            'success': True,
            'migration_stats': result,
            'cinebrain_service': 'slug_migration'
        }), 200
        
    except Exception as e:
        logger.error(f"Error migrating slugs: {e}")
        return jsonify({'error': 'Failed to migrate slugs'}), 500

@admin_bp.route('/api/admin/content/<int:content_id>/slug', methods=['PUT'])
@require_admin
def update_content_slug(current_user, content_id):
    """Update content slug"""
    try:
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        data = request.get_json() or {}
        force_update = data.get('force_update', False)
        result = admin_service.update_content_slug(content_id, force_update)
        
        if result:
            return jsonify({
                'success': True,
                'new_slug': result,
                'cinebrain_service': 'slug_update'
            }), 200
        else:
            return jsonify({'error': 'Content not found or update failed'}), 404
            
    except Exception as e:
        logger.error(f"Error updating content slug: {e}")
        return jsonify({'error': 'Failed to update slug'}), 500

@admin_bp.route('/api/admin/populate-cast-crew', methods=['POST'])
@require_admin
def populate_all_cast_crew(current_user):
    """Populate cast and crew data"""
    try:
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        data = request.get_json() or {}
        batch_size = int(data.get('batch_size', 10))
        result = admin_service.populate_cast_crew(batch_size)
        
        return jsonify({
            'success': True,
            'processed': result.get('processed', 0),
            'errors': result.get('errors', 0),
            'message': f"Successfully populated cast/crew for {result.get('processed', 0)} content items",
            'cinebrain_service': 'cast_crew_population'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in bulk cast/crew population: {e}")
        return jsonify({'error': 'Failed to populate cast/crew'}), 500

# Error Handlers
@admin_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Admin endpoint not found'}), 404

@admin_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error in admin service'}), 500

# CORS Headers
@admin_bp.after_request
def after_request(response):
    """Add CORS headers"""
    origin = request.headers.get('Origin')
    allowed_origins = [
        'https://cinebrain.vercel.app',
        'http://127.0.0.1:5500', 
        'http://127.0.0.1:5501',
        'http://localhost:3000',
        'http://localhost:5173'
    ]
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response

# Initialization Function
def init_admin_routes(flask_app, database, models, services):
    """Initialize admin routes with dependencies"""
    global admin_service, dashboard_service, telegram_service
    global app, db, User, Content, UserInteraction, AdminRecommendation, cache
    
    app = flask_app
    db = database
    User = models.get('User')
    Content = models.get('Content')
    UserInteraction = models.get('UserInteraction')
    AdminRecommendation = models.get('AdminRecommendation')
    cache = services.get('cache')
    
    # Initialize individual services
    try:
        from .service import init_admin_service
        from .dashboard import init_dashboard_service
        from .telegram import init_telegram_service
        
        admin_service = init_admin_service(app, db, models, services)
        dashboard_service = init_dashboard_service(app, db, models, services)
        telegram_service = init_telegram_service(app, db, models, services)
        
        logger.info("✅ Admin routes initialized successfully")
        logger.info(f"   - Admin service: {'✓' if admin_service else '✗'}")
        logger.info(f"   - Dashboard service: {'✓' if dashboard_service else '✗'}")
        logger.info(f"   - Telegram service: {'✓' if telegram_service else '✗'}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize admin routes: {e}")
        raise e