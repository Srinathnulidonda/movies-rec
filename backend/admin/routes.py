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
AdminEmailPreferences = None
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

# Template Management Routes (NEW)
@admin_bp.route('/api/admin/templates', methods=['GET'])
@require_admin
def get_available_templates(current_user):
    """Get available Telegram templates"""
    try:
        templates = {
            'standard': {
                'name': 'Standard Recommendation',
                'description': 'Classic CineBrain recommendation format',
                'icon': '📽️',
                'supports': ['movie', 'tv', 'anime'],
                'parameters': ['admin_name', 'description']
            },
            'mind_bending': {
                'name': 'Mind-Bending',
                'description': 'For reality-warping, twist-heavy content',
                'icon': '🔥',
                'supports': ['movie', 'tv', 'anime'],
                'parameters': ['if_you_like'],
                'example_if_you_like': 'Inception, Primer, Dark'
            },
            'hidden_gem': {
                'name': 'Hidden Gem',
                'description': 'For lesser-known masterpieces',
                'icon': '💎',
                'supports': ['movie', 'tv', 'anime'],
                'parameters': ['hook', 'if_you_like'],
                'example_hook': 'A masterpiece that flew under the radar',
                'example_if_you_like': 'A24 films, indie cinema'
            },
            'anime_gem': {
                'name': 'Anime Gem',
                'description': 'For emotionally impactful anime',
                'icon': '🎐',
                'supports': ['anime'],
                'parameters': ['emotion_hook'],
                'example_emotion_hook': 'Prepare for tears and existential questions'
            }
        }
        
        return jsonify({
            'templates': templates,
            'default': 'standard'
        }), 200
        
    except Exception as e:
        logger.error(f"Get templates error: {e}")
        return jsonify({'error': 'Failed to get templates'}), 500

@admin_bp.route('/api/admin/templates/suggest', methods=['POST'])
@require_admin
def suggest_template(current_user):
    """Suggest best template for content"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        
        if not content_id:
            return jsonify({'error': 'content_id required'}), 400
        
        content = Content.query.get(content_id)
        if not content:
            return jsonify({'error': 'Content not found'}), 404
        
        # Smart template suggestion logic
        suggested = 'standard'
        reason = 'Default template'
        
        # Check for mind-bending characteristics
        mind_bending_genres = {'Sci-Fi', 'Thriller', 'Mystery', 'Psychological'}
        mind_bending_keywords = ['time', 'loop', 'parallel', 'reality', 'mind', 'twist', 'consciousness']
        
        if content.genres:
            genres = json.loads(content.genres) if isinstance(content.genres, str) else content.genres
            if any(genre in mind_bending_genres for genre in genres):
                overview_lower = content.overview.lower() if content.overview else ''
                if any(keyword in overview_lower for keyword in mind_bending_keywords):
                    suggested = 'mind_bending'
                    reason = 'Content has mind-bending themes and genres'
        
        # Check for hidden gem (low popularity, high rating)
        if hasattr(content, 'popularity') and hasattr(content, 'rating'):
            if content.popularity < 50 and content.rating > 7.5:
                suggested = 'hidden_gem'
                reason = 'Low popularity but high rating indicates hidden gem'
        
        # Check for anime gem
        if content.content_type == 'anime' and content.rating and content.rating > 8.0:
            suggested = 'anime_gem'
            reason = 'High-rated anime deserves special treatment'
        
        return jsonify({
            'suggested_template': suggested,
            'reason': reason,
            'content_type': content.content_type
        }), 200
        
    except Exception as e:
        logger.error(f"Template suggestion error: {e}")
        return jsonify({'error': 'Failed to suggest template'}), 500

# Admin Recommendations Routes
@admin_bp.route('/api/admin/recommendations', methods=['POST'])
@require_admin
def create_admin_recommendation(current_user):
    """Create admin recommendation with template support"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        # Extract template information
        template_type = data.get('template_type', 'standard')
        template_params = data.get('template_params', {})
        
        # Handle both content_data and content_id formats
        if 'content_data' in data:
            # New format: save content first, then create recommendation
            result = admin_service.create_recommendation_from_external_content(
                current_user, 
                data['content_data'],
                data.get('recommendation_type'),
                data.get('description'),
                data.get('status', 'draft'),
                data.get('publish_to_telegram', False),
                template_type=template_type,
                template_params=template_params
            )
        else:
            # Legacy format: content already exists
            required_fields = ['content_id', 'recommendation_type', 'description']
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400
            
            result = admin_service.create_recommendation(
                current_user, 
                data['content_id'],
                data['recommendation_type'],
                data['description'],
                template_type=template_type,
                template_params=template_params
            )
        
        return jsonify(result), 201
        
    except Exception as e:
        logger.error(f"Admin recommendation error: {e}")
        return jsonify({'error': 'Failed to create recommendation'}), 500

@admin_bp.route('/api/admin/recommendations', methods=['GET'])
@require_admin
def get_admin_recommendations(current_user):
    """Get admin recommendations with better error handling"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        filter_type = request.args.get('filter', 'all')
        status = request.args.get('status')  # 'draft', 'active', etc.
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.get_recommendations(page, per_page, filter_type, status)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Get admin recommendations error: {e}")
        try:
            db.session.rollback()
        except:
            pass
        return jsonify({'error': 'Failed to get recommendations'}), 500

@admin_bp.route('/api/admin/recommendations/<int:recommendation_id>', methods=['GET'])
@require_admin
def get_recommendation_details(current_user, recommendation_id):
    """Get specific recommendation details"""
    try:
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.get_recommendation_details(recommendation_id)
        if not result:
            return jsonify({'error': 'Recommendation not found'}), 404
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Get recommendation details error: {e}")
        return jsonify({'error': 'Failed to get recommendation details'}), 500

@admin_bp.route('/api/admin/recommendations/<int:recommendation_id>', methods=['PUT'])
@require_admin
def update_recommendation(current_user, recommendation_id):
    """Update existing recommendation with template support"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        # Add template information if provided
        if 'template_type' in data or 'template_params' in data:
            data['template_data'] = {
                'type': data.get('template_type', 'standard'),
                'params': data.get('template_params', {})
            }
        
        result = admin_service.update_recommendation(current_user, recommendation_id, data)
        if not result:
            return jsonify({'error': 'Recommendation not found'}), 404
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Update recommendation error: {e}")
        return jsonify({'error': 'Failed to update recommendation'}), 500

@admin_bp.route('/api/admin/recommendations/<int:recommendation_id>', methods=['DELETE'])
@require_admin
def delete_recommendation(current_user, recommendation_id):
    """Delete recommendation"""
    try:
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.delete_recommendation(current_user, recommendation_id)
        if not result:
            return jsonify({'error': 'Recommendation not found'}), 404
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Delete recommendation error: {e}")
        return jsonify({'error': 'Failed to delete recommendation'}), 500

@admin_bp.route('/api/admin/recommendations/<int:recommendation_id>/publish', methods=['POST'])
@require_admin
def publish_recommendation(current_user, recommendation_id):
    """Publish upcoming recommendation"""
    try:
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.publish_recommendation(current_user, recommendation_id)
        if not result:
            return jsonify({'error': 'Recommendation not found'}), 404
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Publish recommendation error: {e}")
        return jsonify({'error': 'Failed to publish recommendation'}), 500

@admin_bp.route('/api/admin/recommendations/<int:recommendation_id>/send', methods=['POST'])
@require_admin
def send_recommendation_to_telegram(current_user, recommendation_id):
    """Send recommendation to Telegram with template support"""
    try:
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        # Get optional template override from request
        data = request.get_json() or {}
        template_override = data.get('template_type')
        template_params = data.get('template_params', {})
        
        result = admin_service.send_recommendation_to_telegram(
            current_user, 
            recommendation_id,
            template_override=template_override,
            template_params=template_params
        )
        
        if not result:
            return jsonify({'error': 'Recommendation not found'}), 404
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Send to Telegram error: {e}")
        return jsonify({'error': 'Failed to send to Telegram'}), 500

@admin_bp.route('/api/admin/recommendations/<int:recommendation_id>/preview', methods=['POST'])
@require_admin
def preview_telegram_message(current_user, recommendation_id):
    """Preview Telegram message with specific template"""
    try:
        if not admin_service or not telegram_service:
            return jsonify({'error': 'Service not available'}), 503
        
        data = request.get_json() or {}
        template_type = data.get('template_type', 'standard')
        template_params = data.get('template_params', {})
        
        # Get recommendation and content
        recommendation = AdminRecommendation.query.get(recommendation_id)
        if not recommendation:
            return jsonify({'error': 'Recommendation not found'}), 404
        
        content = Content.query.get(recommendation.content_id)
        if not content:
            return jsonify({'error': 'Content not found'}), 404
        
        # Generate preview based on template
        from .telegram import TelegramTemplates
        
        if template_type == 'mind_bending':
            message = TelegramTemplates.mind_bending_template(
                content,
                template_params.get('if_you_like')
            )
        elif template_type == 'hidden_gem':
            message = TelegramTemplates.hidden_gem_template(
                content,
                template_params.get('hook', 'A masterpiece waiting to be discovered'),
                template_params.get('if_you_like')
            )
        elif template_type == 'anime_gem':
            message = TelegramTemplates.anime_gem_template(
                content,
                template_params.get('emotion_hook', 'An emotional journey')
            )
        else:
            # Standard templates
            if content.content_type == 'anime':
                message = TelegramTemplates.anime_recommendation_template(
                    content, current_user.username, recommendation.description
                )
            elif content.content_type in ['tv', 'series']:
                message = TelegramTemplates.tv_show_recommendation_template(
                    content, current_user.username, recommendation.description
                )
            else:
                message = TelegramTemplates.movie_recommendation_template(
                    content, current_user.username, recommendation.description
                )
        
        return jsonify({
            'preview': message,
            'template_type': template_type,
            'has_poster': bool(content.poster_path)
        }), 200
        
    except Exception as e:
        logger.error(f"Preview telegram message error: {e}")
        return jsonify({'error': 'Failed to generate preview'}), 500

# Send curated lists
@admin_bp.route('/api/admin/telegram/send-list', methods=['POST'])
@require_admin
def send_telegram_list(current_user):
    """Send curated list to Telegram"""
    try:
        if not telegram_service:
            return jsonify({'error': 'Telegram service not available'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        list_title = data.get('title')
        items = data.get('items', [])  # List of (title, year, hook) tuples
        content_type = data.get('content_type', 'Movie')
        
        if not list_title or not items:
            return jsonify({'error': 'Title and items required'}), 400
        
        # Convert items to proper format
        formatted_items = []
        for item in items:
            if isinstance(item, dict):
                formatted_items.append((
                    item.get('title', ''),
                    item.get('year', ''),
                    item.get('hook', '')
                ))
            else:
                formatted_items.append(item)
        
        from .telegram import TelegramService
        success = TelegramService.send_top_list(list_title, formatted_items, content_type)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'List sent to Telegram successfully'
            }), 200
        else:
            return jsonify({'error': 'Failed to send list to Telegram'}), 500
        
    except Exception as e:
        logger.error(f"Send Telegram list error: {e}")
        return jsonify({'error': 'Failed to send list'}), 500

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
        try:
            db.session.rollback()
        except:
            pass
        return jsonify({'error': 'Failed to load dashboard'}), 500

@admin_bp.route('/api/admin/dashboard/stats', methods=['GET'])
@require_admin
def get_dashboard_stats(current_user):
    """Get dashboard statistics - separate endpoint for stats"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503
        
        # Get basic stats only to avoid complex queries
        stats = {}
        
        try:
            stats['total_users'] = User.query.count() if User else 0
        except Exception as e:
            logger.error(f"Error getting user count: {e}")
            stats['total_users'] = 0
        
        try:
            stats['total_content'] = Content.query.count() if Content else 0
        except Exception as e:
            logger.error(f"Error getting content count: {e}")
            stats['total_content'] = 0
        
        try:
            stats['total_interactions'] = UserInteraction.query.count() if UserInteraction else 0
        except Exception as e:
            logger.error(f"Error getting interaction count: {e}")
            stats['total_interactions'] = 0
        
        try:
            stats['active_recommendations'] = AdminRecommendation.query.filter_by(is_active=True).count() if AdminRecommendation else 0
        except Exception as e:
            logger.error(f"Error getting recommendation count: {e}")
            stats['active_recommendations'] = 0
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        return jsonify({'error': 'Failed to load dashboard stats'}), 500

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

# Email Preferences Routes (NEW)
@admin_bp.route('/api/admin/email-preferences', methods=['GET'])
@require_admin
def get_email_preferences(current_user):
    """Get admin email preferences"""
    try:
        if not AdminEmailPreferences:
            return jsonify({'error': 'Email preferences not available'}), 503
        
        preferences = AdminEmailPreferences.query.filter_by(admin_id=current_user.id).first()
        if not preferences:
            # Create default preferences
            preferences = AdminEmailPreferences(admin_id=current_user.id)
            db.session.add(preferences)
            db.session.commit()
        
        return jsonify({
            'preferences': {
                'critical_alerts': {
                    'urgent_tickets': preferences.urgent_tickets,
                    'sla_breaches': preferences.sla_breaches,
                    'system_alerts': preferences.system_alerts
                },
                'content_management': {
                    'content_added': preferences.content_added,
                    'recommendation_created': preferences.recommendation_created,
                    'recommendation_updated': preferences.recommendation_updated,
                    'recommendation_deleted': preferences.recommendation_deleted,
                    'recommendation_published': preferences.recommendation_published
                },
                'user_activity': {
                    'user_feedback': preferences.user_feedback,
                    'regular_tickets': preferences.regular_tickets
                },
                'system_operations': {
                    'cache_operations': preferences.cache_operations,
                    'bulk_operations': preferences.bulk_operations,
                    'slug_updates': preferences.slug_updates
                }
            },
            'updated_at': preferences.updated_at.isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Get email preferences error: {e}")
        return jsonify({'error': 'Failed to get email preferences'}), 500

@admin_bp.route('/api/admin/email-preferences', methods=['PUT'])
@require_admin
def update_email_preferences(current_user):
    """Update admin email preferences"""
    try:
        if not AdminEmailPreferences:
            return jsonify({'error': 'Email preferences not available'}), 503
        
        data = request.get_json()
        if not data or 'preferences' not in data:
            return jsonify({'error': 'Invalid request data'}), 400
        
        preferences = AdminEmailPreferences.query.filter_by(admin_id=current_user.id).first()
        if not preferences:
            preferences = AdminEmailPreferences(admin_id=current_user.id)
            db.session.add(preferences)
        
        prefs = data['preferences']
        
        # Update critical alerts
        if 'critical_alerts' in prefs:
            critical = prefs['critical_alerts']
            preferences.urgent_tickets = critical.get('urgent_tickets', preferences.urgent_tickets)
            preferences.sla_breaches = critical.get('sla_breaches', preferences.sla_breaches)
            preferences.system_alerts = critical.get('system_alerts', preferences.system_alerts)
        
        # Update content management
        if 'content_management' in prefs:
            content = prefs['content_management']
            preferences.content_added = content.get('content_added', preferences.content_added)
            preferences.recommendation_created = content.get('recommendation_created', preferences.recommendation_created)
            preferences.recommendation_updated = content.get('recommendation_updated', preferences.recommendation_updated)
            preferences.recommendation_deleted = content.get('recommendation_deleted', preferences.recommendation_deleted)
            preferences.recommendation_published = content.get('recommendation_published', preferences.recommendation_published)
        
        # Update user activity
        if 'user_activity' in prefs:
            user = prefs['user_activity']
            preferences.user_feedback = user.get('user_feedback', preferences.user_feedback)
            preferences.regular_tickets = user.get('regular_tickets', preferences.regular_tickets)
        
        # Update system operations
        if 'system_operations' in prefs:
            system = prefs['system_operations']
            preferences.cache_operations = system.get('cache_operations', preferences.cache_operations)
            preferences.bulk_operations = system.get('bulk_operations', preferences.bulk_operations)
            preferences.slug_updates = system.get('slug_updates', preferences.slug_updates)
        
        preferences.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"✅ Email preferences updated for admin {current_user.username}")
        return jsonify({
            'success': True,
            'message': 'Email preferences updated successfully',
            'updated_at': preferences.updated_at.isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Update email preferences error: {e}")
        try:
            db.session.rollback()
        except:
            pass
        return jsonify({'error': 'Failed to update email preferences'}), 500

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

# Health Check Routes (NEW)
@admin_bp.route('/api/admin/health', methods=['GET'])
def admin_health_check():
    """Simple health check for admin service"""
    return jsonify({
        'status': 'healthy',
        'service': 'CineBrain Admin',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '4.0'
    }), 200

@admin_bp.route('/api/admin/ping', methods=['GET'])
def admin_ping():
    """Simple ping endpoint"""
    return jsonify({
        'message': 'pong',
        'service': 'admin',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

# Service Status Routes (NEW)
@admin_bp.route('/api/admin/services/status', methods=['GET'])
@require_admin
def get_services_status(current_user):
    """Get status of all admin services"""
    try:
        status = {
            'admin_service': 'available' if admin_service else 'unavailable',
            'dashboard_service': 'available' if dashboard_service else 'unavailable',
            'telegram_service': 'available' if telegram_service else 'unavailable',
            'database': 'connected' if db else 'disconnected',
            'cache': 'available' if cache else 'unavailable',
            'email_preferences': 'available' if AdminEmailPreferences else 'unavailable'
        }
        
        return jsonify({
            'status': status,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Services status error: {e}")
        return jsonify({'error': 'Failed to get services status'}), 500

# Error Handlers
@admin_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Admin endpoint not found'}), 404

@admin_bp.errorhandler(500)
def internal_error(error):
    try:
        db.session.rollback()
    except:
        pass
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
    global app, db, User, Content, UserInteraction, AdminRecommendation, AdminEmailPreferences, cache
    
    app = flask_app
    db = database
    User = models.get('User')
    Content = models.get('Content')
    UserInteraction = models.get('UserInteraction')
    AdminRecommendation = models.get('AdminRecommendation')
    AdminEmailPreferences = models.get('AdminEmailPreferences')  # NEW
    cache = services.get('cache')
    
    # Initialize individual services
    try:
        from .service import init_admin_service
        from .dashboard import init_dashboard_service
        from .telegram import init_telegram_service
        
        admin_service = init_admin_service(app, db, models, services)
        dashboard_service = init_dashboard_service(app, db, models, services)
        telegram_results = init_telegram_service(app, db, models, services)
        
        # Extract telegram_service from results
        if telegram_results:
            telegram_service = telegram_results.get('telegram_service')
        
        logger.info("✅ Admin routes initialized successfully")
        logger.info(f"   - Admin service: {'✓' if admin_service else '✗'}")
        logger.info(f"   - Dashboard service: {'✓' if dashboard_service else '✗'}")
        logger.info(f"   - Telegram service: {'✓' if telegram_service else '✗'}")
        logger.info(f"   - Email preferences: {'✓' if AdminEmailPreferences else '✗'}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize admin routes: {e}")
        raise e