# admin/routes.py

from flask import Blueprint, request, jsonify, Response
from datetime import datetime, timedelta
import json
import logging
import jwt
from functools import wraps

# Import the user service
from .users import init_user_service

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__)

# Global services - initialized by init_admin_routes
admin_service = None
dashboard_service = None
telegram_service = None
user_service = None
app = None
db = None
User = None
Content = None
UserInteraction = None
AdminRecommendation = None
AdminEmailPreferences = None
SupportTicket = None
ContactMessage = None
IssueReport = None
SupportCategory = None
TicketActivity = None
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

# =============================================================================
# REAL-TIME MONITORING ENDPOINTS (4 Main Endpoints)
# =============================================================================

@admin_bp.route('/api/system-monitoring', methods=['GET'])
@require_admin
def get_system_monitoring(current_user):
    """Real-time system monitoring - /api/system-monitoring/"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503

        monitoring_data = dashboard_service.get_system_monitoring()
        
        return jsonify({
            'success': True,
            'data': monitoring_data,
            'timestamp': datetime.utcnow().isoformat(),
            'refresh_interval': 30
        }), 200

    except Exception as e:
        logger.error(f"System monitoring error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get system monitoring data'
        }), 500

@admin_bp.route('/api/overview', methods=['GET'])
@require_admin
def get_overview_stats(current_user):
    """Real-time overview statistics - /api/overview/"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503

        overview_data = dashboard_service.get_overview_stats()
        
        return jsonify({
            'success': True,
            'data': overview_data,
            'timestamp': datetime.utcnow().isoformat(),
            'refresh_interval': 60
        }), 200

    except Exception as e:
        logger.error(f"Overview stats error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get overview statistics'
        }), 500

@admin_bp.route('/api/service-status', methods=['GET'])
@require_admin
def get_service_status(current_user):
    """Real-time service status - /api/service-status/"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503

        service_status = dashboard_service.get_service_status()
        
        return jsonify({
            'success': True,
            'data': service_status,
            'timestamp': datetime.utcnow().isoformat(),
            'refresh_interval': 15
        }), 200

    except Exception as e:
        logger.error(f"Service status error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get service status'
        }), 500

@admin_bp.route('/api/admin-activity', methods=['GET'])
@require_admin
def get_admin_activity(current_user):
    """Real-time admin activity - /api/admin-activity/"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503

        admin_activity = dashboard_service.get_admin_activity()
        
        return jsonify({
            'success': True,
            'data': admin_activity,
            'timestamp': datetime.utcnow().isoformat(),
            'refresh_interval': 45
        }), 200

    except Exception as e:
        logger.error(f"Admin activity error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get admin activity data'
        }), 500

# =============================================================================
# CONTENT MANAGEMENT ROUTES
# =============================================================================

@admin_bp.route('/api/admin/search', methods=['GET'])
@require_admin
def admin_search(current_user):
    """Search external content from TMDB/Jikan"""
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
                'service': 'slug_update'
            }), 200
        else:
            return jsonify({'error': 'Content not found or update failed'}), 404
            
    except Exception as e:
        logger.error(f"Error updating content slug: {e}")
        return jsonify({'error': 'Failed to update slug'}), 500

# =============================================================================
# RECOMMENDATION MANAGEMENT ROUTES
# =============================================================================

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
        
        template_type = data.get('template_type', 'auto')
        template_params = data.get('template_params', {})
        
        if 'content_data' in data:
            result = admin_service.create_recommendation_from_external_content(
                current_user, 
                data['content_data'],
                data.get('recommendation_type'),
                data.get('description'),
                data.get('status', 'draft'),
                data.get('publish_to_telegram', False),
                template_type,
                template_params
            )
        else:
            required_fields = ['content_id', 'recommendation_type', 'description']
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400
            
            result = admin_service.create_recommendation(
                current_user, 
                data['content_id'],
                data['recommendation_type'],
                data['description'],
                template_type,
                template_params
            )
        
        return jsonify(result), 201
        
    except Exception as e:
        logger.error(f"Admin recommendation error: {e}")
        return jsonify({'error': 'Failed to create recommendation'}), 500

@admin_bp.route('/api/admin/recommendations', methods=['GET'])
@require_admin
def get_admin_recommendations(current_user):
    """Get admin recommendations list"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        filter_type = request.args.get('filter', 'all')
        status = request.args.get('status')
        
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
    """Get recommendation details"""
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
    """Update recommendation"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
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
    """Publish recommendation"""
    try:
        data = request.get_json() or {}
        template_type = data.get('template_type', 'auto')
        template_params = data.get('template_params', {})
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.publish_recommendation(current_user, recommendation_id, template_type, template_params)
        if not result:
            return jsonify({'error': 'Recommendation not found'}), 404
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Publish recommendation error: {e}")
        return jsonify({'error': 'Failed to publish recommendation'}), 500

@admin_bp.route('/api/admin/recommendations/<int:recommendation_id>/send', methods=['POST'])
@require_admin
def send_recommendation_to_telegram(current_user, recommendation_id):
    """Send recommendation to Telegram"""
    try:
        data = request.get_json() or {}
        template_type = data.get('template_type', 'auto')
        template_params = data.get('template_params', {})
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.send_recommendation_to_telegram(current_user, recommendation_id, template_type, template_params)
        if not result:
            return jsonify({'error': 'Recommendation not found'}), 404
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Send to Telegram error: {e}")
        return jsonify({'error': 'Failed to send to Telegram'}), 500

@admin_bp.route('/api/admin/recommendations/create-with-template', methods=['POST'])
@require_admin
def create_recommendation_with_template(current_user):
    """Create recommendation with specific template"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        template_type = data.get('template_type', 'auto')
        template_params = data.get('template_params', {})
        
        if 'content_data' in data:
            result = admin_service.create_recommendation_from_external_content(
                current_user, 
                data['content_data'],
                data.get('recommendation_type'),
                data.get('description'),
                data.get('status', 'draft'),
                data.get('publish_to_telegram', False),
                template_type,
                template_params
            )
        else:
            return jsonify({'error': 'Content data required for template-based recommendations'}), 400
        
        return jsonify(result), 201
        
    except Exception as e:
        logger.error(f"Create recommendation with template error: {e}")
        return jsonify({'error': 'Failed to create recommendation with template'}), 500

@admin_bp.route('/api/admin/recommendations/<int:recommendation_id>/send-custom', methods=['POST'])
@require_admin
def send_recommendation_with_template(current_user, recommendation_id):
    """Send recommendation with custom template"""
    try:
        data = request.get_json()
        template_type = data.get('template_type', 'auto')
        template_params = data.get('template_params', {})
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.send_recommendation_to_telegram(
            current_user, recommendation_id, template_type, template_params
        )
        if not result:
            return jsonify({'error': 'Recommendation not found'}), 404
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Send with template error: {e}")
        return jsonify({'error': 'Failed to send with custom template'}), 500

# =============================================================================
# TELEGRAM INTEGRATION ROUTES
# =============================================================================

@admin_bp.route('/api/admin/telegram/templates', methods=['GET'])
@require_admin
def get_telegram_templates(current_user):
    """Get available Telegram templates"""
    try:
        from admin.telegram import TelegramTemplates
        templates = TelegramTemplates.get_available_templates()
        return jsonify({
            'templates': templates,
            'default': 'auto'
        }), 200
    except Exception as e:
        logger.error(f"Get templates error: {e}")
        return jsonify({'error': 'Failed to get templates'}), 500

@admin_bp.route('/api/admin/telegram/templates/prompts', methods=['GET'])
@require_admin
def get_telegram_template_prompts(current_user):
    """Get template prompts for AI generation"""
    try:
        from admin.telegram import TelegramTemplates
        
        template_type = request.args.get('template')
        
        if template_type:
            prompts = TelegramTemplates.get_template_prompts()
            fields = TelegramTemplates.get_template_fields(template_type)
            
            if template_type in prompts:
                return jsonify({
                    'template': template_type,
                    'prompt_info': prompts[template_type],
                    'field_specs': fields
                }), 200
            else:
                return jsonify({'error': 'Template not found'}), 404
        else:
            return jsonify({
                'prompts': TelegramTemplates.get_template_prompts(),
                'available_templates': TelegramTemplates.get_available_templates()
            }), 200
            
    except Exception as e:
        logger.error(f"Get template prompts error: {e}")
        return jsonify({'error': 'Failed to get template prompts'}), 500

@admin_bp.route('/api/admin/telegram/custom-message', methods=['POST'])
@require_admin
def send_custom_telegram_message(current_user):
    """Send custom Telegram message"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        template_type = data.get('template_type')
        template_params = data.get('template_params', {})
        
        if not template_type:
            return jsonify({'error': 'Template type required'}), 400
        
        if not admin_service:
            return jsonify({'error': 'Admin service not available'}), 503
        
        result = admin_service.send_custom_telegram_message(current_user, template_type, template_params)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Send custom message error: {e}")
        return jsonify({'error': 'Failed to send custom message'}), 500

# =============================================================================
# DASHBOARD ROUTES
# =============================================================================

@admin_bp.route('/api/admin/dashboard', methods=['GET'])
@require_admin
def get_admin_dashboard(current_user):
    """Get comprehensive admin dashboard"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503
        
        include_support = request.args.get('include_support', 'true').lower() == 'true'
        
        dashboard_data = dashboard_service.get_overview()
        
        if include_support:
            try:
                since = request.args.get('since')
                real_time_support = dashboard_service.get_real_time_support_data(since)
                dashboard_data['real_time_support'] = real_time_support
                
                summary_stats = dashboard_service.get_support_summary_stats()
                dashboard_data['support_summary'] = summary_stats
                
            except Exception as e:
                logger.warning(f"Failed to get real-time support data: {e}")
                dashboard_data['real_time_support'] = {'error': str(e)}

        return jsonify({
            'success': True,
            'dashboard': dashboard_data,
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Enhanced dashboard error: {e}")
        try:
            db.session.rollback()
        except:
            pass
        return jsonify({
            'success': False,
            'error': 'Failed to load dashboard'
        }), 500

@admin_bp.route('/api/admin/dashboard/stats', methods=['GET'])
@require_admin
def get_dashboard_stats(current_user):
    """Get basic dashboard statistics"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503
        
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
    """Get comprehensive analytics data"""
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

# =============================================================================
# SUPPORT INTEGRATION ROUTES
# =============================================================================

@admin_bp.route('/api/admin/support/real-time', methods=['GET'])
@require_admin
def get_real_time_support_data(current_user):
    """Get real-time support data"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503

        since = request.args.get('since')
        include_stats = request.args.get('include_stats', 'true').lower() == 'true'

        real_time_data = dashboard_service.get_real_time_support_data(since)
        
        if include_stats:
            summary_stats = dashboard_service.get_support_summary_stats()
            real_time_data['summary_stats'] = summary_stats

        return jsonify({
            'success': True,
            'data': real_time_data,
            'server_time': datetime.utcnow().isoformat(),
            'has_new_items': real_time_data['new_items_count'] > 0
        }), 200

    except Exception as e:
        logger.error(f"Real-time support data error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get real-time support data',
            'server_time': datetime.utcnow().isoformat()
        }), 500

@admin_bp.route('/api/admin/support/summary-stats', methods=['GET'])
@require_admin  
def get_support_summary_stats(current_user):
    """Get support summary statistics"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503

        stats = dashboard_service.get_support_summary_stats()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Support summary stats error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get support summary statistics'
        }), 500

@admin_bp.route('/api/admin/support/alerts', methods=['GET'])
@require_admin
def get_support_alerts(current_user):
    """Get support alerts"""
    try:
        if not dashboard_service:
            return jsonify({'error': 'Dashboard service not available'}), 503

        since = (datetime.utcnow() - timedelta(hours=4)).isoformat()
        real_time_data = dashboard_service.get_real_time_support_data(since)
        
        alerts = real_time_data.get('urgent_alerts', [])
        
        try:
            health_data = dashboard_service.get_system_health()
            if health_data.get('status') != 'healthy':
                alerts.append({
                    'type': 'system_health',
                    'message': 'System health issues detected',
                    'url': '/admin/system-health',
                    'created_at': datetime.utcnow().isoformat()
                })
        except Exception as e:
            logger.warning(f"Health check failed: {e}")

        return jsonify({
            'success': True,
            'alerts': alerts[:10],
            'alert_count': len(alerts),
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Support alerts error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get support alerts'
        }), 500

@admin_bp.route('/api/admin/support/mark-seen', methods=['POST'])
@require_admin
def mark_support_items_seen(current_user):
    """Mark support items as seen by admin"""
    try:
        data = request.get_json()
        item_type = data.get('type')
        item_ids = data.get('ids', [])
        
        if not item_type or not item_ids:
            return jsonify({'error': 'Type and IDs required'}), 400
        
        marked_count = 0
        current_time = datetime.utcnow()
        
        if item_type == 'contact' and ContactMessage:
            for contact_id in item_ids:
                try:
                    contact = ContactMessage.query.get(contact_id)
                    if contact:
                        contact.is_read = True
                        contact.admin_viewed = True
                        contact.admin_viewed_at = current_time
                        contact.admin_viewed_by = current_user.id
                        marked_count += 1
                except Exception as e:
                    logger.error(f"Error marking contact {contact_id}: {e}")
                    continue
        
        elif item_type == 'issue' and IssueReport:
            for issue_id in item_ids:
                try:
                    issue = IssueReport.query.get(issue_id)
                    if issue:
                        issue.admin_viewed = True
                        issue.admin_viewed_at = current_time
                        issue.admin_viewed_by = current_user.id
                        marked_count += 1
                except Exception as e:
                    logger.error(f"Error marking issue {issue_id}: {e}")
                    continue
        
        elif item_type == 'ticket' and SupportTicket:
            for ticket_id in item_ids:
                try:
                    ticket = SupportTicket.query.get(ticket_id)
                    if ticket:
                        ticket.admin_viewed = True
                        ticket.admin_viewed_at = current_time
                        ticket.admin_viewed_by = current_user.id
                        marked_count += 1
                        
                        if TicketActivity:
                            activity = TicketActivity(
                                ticket_id=ticket.id,
                                action='admin_viewed',
                                description=f'Viewed by admin {current_user.username}',
                                actor_type='admin',
                                actor_id=current_user.id,
                                actor_name=current_user.username
                            )
                            db.session.add(activity)
                except Exception as e:
                    logger.error(f"Error marking ticket {ticket_id}: {e}")
                    continue
        
        if marked_count > 0:
            db.session.commit()
            logger.info(f"✅ Admin {current_user.username} marked {marked_count} {item_type}s as seen")
        
        return jsonify({
            'success': True,
            'marked_count': marked_count,
            'message': f'Marked {marked_count} {item_type}s as seen'
        }), 200
        
    except Exception as e:
        logger.error(f"Mark items seen error: {e}")
        try:
            db.session.rollback()
        except:
            pass
        return jsonify({'error': 'Failed to mark items as seen'}), 500

@admin_bp.route('/api/admin/support/ticket/<int:ticket_id>/quick-update', methods=['POST'])
@require_admin
def quick_update_ticket(current_user, ticket_id):
    """Quick update ticket status/priority"""
    try:
        data = request.get_json()
        action = data.get('action')
        value = data.get('value')
        
        if not action or not value:
            return jsonify({'error': 'Action and value required'}), 400
        
        if not SupportTicket:
            return jsonify({'error': 'Support system not available'}), 503
        
        ticket = SupportTicket.query.get_or_404(ticket_id)
        old_value = None
        
        if action == 'status':
            old_value = ticket.status
            ticket.status = value
            if value == 'resolved' and not ticket.resolved_at:
                ticket.resolved_at = datetime.utcnow()
            elif value == 'closed' and not ticket.closed_at:
                ticket.closed_at = datetime.utcnow()
        
        elif action == 'priority':
            old_value = ticket.priority
            ticket.priority = value
        
        elif action == 'assign':
            old_value = ticket.assigned_to
            if value == 'self':
                ticket.assigned_to = current_user.id
                value = current_user.username
            elif value == 'unassign':
                ticket.assigned_to = None
                value = None
            else:
                try:
                    user_id = int(value)
                    user = User.query.get(user_id)
                    if user:
                        ticket.assigned_to = user_id
                        value = user.username
                    else:
                        return jsonify({'error': 'User not found'}), 404
                except ValueError:
                    return jsonify({'error': 'Invalid user ID'}), 400
        
        if TicketActivity:
            activity = TicketActivity(
                ticket_id=ticket.id,
                action=f'{action}_updated',
                description=f'{action.title()} changed from {old_value} to {value}',
                old_value=str(old_value) if old_value else None,
                new_value=str(value) if value else None,
                actor_type='admin',
                actor_id=current_user.id,
                actor_name=current_user.username
            )
            db.session.add(activity)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Ticket {action} updated successfully',
            'ticket': {
                'id': ticket.id,
                'ticket_number': ticket.ticket_number,
                'status': ticket.status,
                'priority': ticket.priority,
                'assigned_to': ticket.assigned_to,
                'updated_at': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Quick update ticket error: {e}")
        try:
            db.session.rollback()
        except:
            pass
        return jsonify({'error': 'Failed to update ticket'}), 500

@admin_bp.route('/api/admin/support/contact/<int:contact_id>/quick-reply', methods=['POST'])
@require_admin
def quick_reply_contact(current_user, contact_id):
    """Quick reply to contact message"""
    try:
        data = request.get_json()
        reply_message = data.get('message')
        
        if not reply_message:
            return jsonify({'error': 'Reply message required'}), 400
        
        if not ContactMessage:
            return jsonify({'error': 'Support system not available'}), 503
        
        contact = ContactMessage.query.get_or_404(contact_id)
        
        contact.is_read = True
        contact.admin_viewed = True
        contact.admin_viewed_at = datetime.utcnow()
        contact.admin_viewed_by = current_user.id
        
        try:
            from auth.service import email_service
            from auth.support_mail_templates import get_support_template
            
            if email_service:
                html, text = get_support_template(
                    'admin_reply',
                    user_name=contact.name,
                    admin_name=current_user.username,
                    original_subject=contact.subject,
                    reply_message=reply_message,
                    contact_id=contact.id
                )
                
                email_service.queue_email(
                    to=contact.email,
                    subject=f"Re: {contact.subject} - CineBrain Support",
                    html=html,
                    text=text,
                    priority='high',
                    to_name=contact.name
                )
                
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'message': 'Reply sent successfully',
                    'contact_id': contact.id
                }), 200
            else:
                return jsonify({'error': 'Email service not available'}), 503
                
        except Exception as e:
            logger.error(f"Error sending reply email: {e}")
            return jsonify({'error': 'Failed to send reply'}), 500
        
    except Exception as e:
        logger.error(f"Quick reply error: {e}")
        try:
            db.session.rollback()
        except:
            pass
        return jsonify({'error': 'Failed to send reply'}), 500

# =============================================================================
# USER MANAGEMENT ROUTES (EXISTING + NEW)
# =============================================================================

@admin_bp.route('/api/admin/users', methods=['GET'])
@require_admin
def get_users_list(current_user):
    """Get paginated, filtered, and sorted user list"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 25))
        sort_by = request.args.get('sort_by', 'created_at')
        sort_direction = request.args.get('sort_direction', 'desc')
        
        # Get filters
        filters = {
            'status': request.args.get('status', ''),
            'role': request.args.get('role', ''),
            'registration': request.args.get('registration', ''),
            'search': request.args.get('search', '')
        }
        
        # Remove empty filters
        filters = {k: v for k, v in filters.items() if v}
        
        result = user_service.get_users_list(
            page=page,
            per_page=per_page,
            sort_by=sort_by,
            sort_direction=sort_direction,
            filters=filters
        )
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Get users list error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get users list'}), 500

@admin_bp.route('/api/admin/users/statistics', methods=['GET'])
@require_admin
def get_user_statistics(current_user):
    """Get comprehensive user statistics"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        timeframe = request.args.get('timeframe', 'week')
        result = user_service.get_user_statistics(timeframe)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Get user statistics error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get user statistics'}), 500

@admin_bp.route('/api/admin/users/analytics', methods=['GET'])
@require_admin
def get_user_analytics(current_user):
    """Get detailed user analytics with activity trends"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        period = request.args.get('period', '30d')
        result = user_service.get_user_analytics(period)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Get user analytics error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get user analytics'}), 500

@admin_bp.route('/api/admin/users/<int:user_id>', methods=['GET'])
@require_admin
def get_user_details(current_user, user_id):
    """Get comprehensive details for a specific user"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        result = user_service.get_user_details(user_id)
        
        return jsonify(result), 200 if result['success'] else 404
        
    except Exception as e:
        logger.error(f"Get user details error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get user details'}), 500

@admin_bp.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@require_admin
def update_user(current_user, user_id):
    """Update user information"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        result = user_service.update_user(user_id, current_user, data)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Update user error: {e}")
        return jsonify({'success': False, 'error': 'Failed to update user'}), 500

@admin_bp.route('/api/admin/users/<int:user_id>/status', methods=['PUT'])
@require_admin
def toggle_user_status(current_user, user_id):
    """Toggle user status (suspend/activate)"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        data = request.get_json()
        action = data.get('action') if data else None
        
        if not action or action not in ['suspend', 'activate']:
            return jsonify({'success': False, 'error': 'Invalid action'}), 400
        
        result = user_service.toggle_user_status(user_id, current_user, action)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Toggle user status error: {e}")
        return jsonify({'success': False, 'error': 'Failed to toggle user status'}), 500

@admin_bp.route('/api/admin/users/bulk', methods=['POST'])
@require_admin
def bulk_user_operation(current_user):
    """Perform bulk operations on multiple users"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        action = data.get('action')
        user_ids = data.get('user_ids', [])
        
        if not action or action not in ['suspend', 'activate', 'delete']:
            return jsonify({'success': False, 'error': 'Invalid action'}), 400
        
        if not user_ids or not isinstance(user_ids, list):
            return jsonify({'success': False, 'error': 'Invalid user IDs'}), 400
        
        result = user_service.bulk_user_operation(current_user, action, user_ids)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Bulk user operation error: {e}")
        return jsonify({'success': False, 'error': 'Failed to perform bulk operation'}), 500

@admin_bp.route('/api/admin/users/export', methods=['GET'])
@require_admin
def export_users(current_user):
    """Export users data in specified format"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        # Get export parameters
        format = request.args.get('format', 'csv')
        
        # Get filters
        filters = {
            'status': request.args.get('status', ''),
            'role': request.args.get('role', ''),
            'registration': request.args.get('registration', ''),
            'search': request.args.get('search', '')
        }
        
        # Remove empty filters
        filters = {k: v for k, v in filters.items() if v}
        
        # Export data
        file_data, filename, mimetype = user_service.export_users(filters, format)
        
        # Create response
        if format == 'csv':
            response = Response(
                file_data.getvalue(),
                mimetype=mimetype,
                headers={
                    'Content-Disposition': f'attachment; filename={filename}'
                }
            )
        else:
            response = Response(
                file_data,
                mimetype=mimetype,
                headers={
                    'Content-Disposition': f'attachment; filename={filename}'
                }
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Export users error: {e}")
        return jsonify({'success': False, 'error': 'Failed to export users'}), 500

# ========== NEW USER MANAGEMENT ROUTES ==========

@admin_bp.route('/api/admin/users/segmentation', methods=['GET'])
@require_admin
def get_user_segmentation_route(current_user):
    """Get user segmentation analysis"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        result = user_service.get_user_segmentation()
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Get user segmentation error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get user segmentation'}), 500

@admin_bp.route('/api/admin/users/lifecycle-analysis', methods=['GET'])
@require_admin
def get_user_lifecycle_analysis_route(current_user):
    """Get user lifecycle analysis"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        period_days = int(request.args.get('period_days', 90))
        result = user_service.get_user_lifecycle_analysis(period_days)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Get user lifecycle analysis error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get lifecycle analysis'}), 500

@admin_bp.route('/api/admin/users/<int:user_id>/behavior-intelligence', methods=['GET'])
@require_admin
def get_user_behavior_intelligence_route(current_user, user_id):
    """Get user behavior intelligence"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        result = user_service.get_user_behavior_intelligence(user_id)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Get user behavior intelligence error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get behavior intelligence'}), 500

@admin_bp.route('/api/admin/users/advanced-search', methods=['POST'])
@require_admin
def advanced_user_search_route(current_user):
    """Advanced user search with multiple criteria"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        search_params = request.get_json() or {}
        result = user_service.advanced_user_search(search_params)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Advanced user search error: {e}")
        return jsonify({'success': False, 'error': 'Failed to perform advanced search'}), 500

@admin_bp.route('/api/admin/users/targeted-communication', methods=['POST'])
@require_admin
def send_targeted_communication_route(current_user):
    """Send targeted communication to users"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        recipient_params = data.get('recipients', {})
        message_data = data.get('message', {})
        
        if not recipient_params or not message_data:
            return jsonify({'error': 'Recipients and message data required'}), 400
        
        result = user_service.send_targeted_communication(recipient_params, message_data)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Send targeted communication error: {e}")
        return jsonify({'success': False, 'error': 'Failed to send targeted communication'}), 500

@admin_bp.route('/api/admin/users/value-scores', methods=['GET'])
@require_admin
def get_user_value_scores_route(current_user):
    """Calculate user value scores"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        result = user_service.calculate_user_value_scores()
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Calculate user value scores error: {e}")
        return jsonify({'success': False, 'error': 'Failed to calculate user value scores'}), 500

@admin_bp.route('/api/admin/users/anomalies', methods=['GET'])
@require_admin
def detect_user_anomalies_route(current_user):
    """Detect user anomalies"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        period_days = int(request.args.get('period_days', 30))
        result = user_service.detect_user_anomalies(period_days)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Detect user anomalies error: {e}")
        return jsonify({'success': False, 'error': 'Failed to detect user anomalies'}), 500

@admin_bp.route('/api/admin/users/cohort-analysis', methods=['GET'])
@require_admin
def analyze_user_cohorts_route(current_user):
    """Analyze user cohorts"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        cohort_period = request.args.get('cohort_period', 'monthly')
        result = user_service.analyze_user_cohorts(cohort_period)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Analyze user cohorts error: {e}")
        return jsonify({'success': False, 'error': 'Failed to analyze user cohorts'}), 500

@admin_bp.route('/api/admin/users/<int:user_id>/support-profile', methods=['GET'])
@require_admin
def get_user_support_profile_route(current_user, user_id):
    """Get user support profile"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        result = user_service.get_user_support_profile(user_id)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Get user support profile error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get user support profile'}), 500

@admin_bp.route('/api/admin/users/compare', methods=['POST'])
@require_admin
def compare_users_route(current_user):
    """Compare multiple users"""
    try:
        if not user_service:
            return jsonify({'error': 'User management service not available'}), 503
        
        data = request.get_json()
        user_ids = data.get('user_ids', []) if data else []
        
        if not user_ids or not isinstance(user_ids, list):
            return jsonify({'error': 'List of user IDs required'}), 400
        
        result = user_service.compare_users(user_ids)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Compare users error: {e}")
        return jsonify({'success': False, 'error': 'Failed to compare users'}), 500

# =============================================================================
# EMAIL PREFERENCES ROUTES
# =============================================================================

@admin_bp.route('/api/admin/email-preferences', methods=['GET'])
@require_admin
def get_email_preferences(current_user):
    """Get admin email preferences"""
    try:
        if not AdminEmailPreferences:
            return jsonify({'error': 'Email preferences not available'}), 503
        
        preferences = AdminEmailPreferences.query.filter_by(admin_id=current_user.id).first()
        if not preferences:
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
        
        if 'critical_alerts' in prefs:
            critical = prefs['critical_alerts']
            preferences.urgent_tickets = critical.get('urgent_tickets', preferences.urgent_tickets)
            preferences.sla_breaches = critical.get('sla_breaches', preferences.sla_breaches)
            preferences.system_alerts = critical.get('system_alerts', preferences.system_alerts)
        
        if 'content_management' in prefs:
            content = prefs['content_management']
            preferences.content_added = content.get('content_added', preferences.content_added)
            preferences.recommendation_created = content.get('recommendation_created', preferences.recommendation_created)
            preferences.recommendation_updated = content.get('recommendation_updated', preferences.recommendation_updated)
            preferences.recommendation_deleted = content.get('recommendation_deleted', preferences.recommendation_deleted)
            preferences.recommendation_published = content.get('recommendation_published', preferences.recommendation_published)
        
        if 'user_activity' in prefs:
            user = prefs['user_activity']
            preferences.user_feedback = user.get('user_feedback', preferences.user_feedback)
            preferences.regular_tickets = user.get('regular_tickets', preferences.regular_tickets)
        
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

# =============================================================================
# NOTIFICATIONS ROUTES
# =============================================================================

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

# =============================================================================
# CACHE MANAGEMENT ROUTES
# =============================================================================

@admin_bp.route('/api/admin/cache/clear', methods=['POST'])
@require_admin
def clear_cache(current_user):
    """Clear cache"""
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

# =============================================================================
# SYSTEM OPERATIONS ROUTES
# =============================================================================

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
            'service': 'slug_migration'
        }), 200
        
    except Exception as e:
        logger.error(f"Error migrating slugs: {e}")
        return jsonify({'error': 'Failed to migrate slugs'}), 500

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
            'service': 'cast_crew_population'
        }), 200
        
    except Exception as e:
        logger.error(f"Error in bulk cast/crew population: {e}")
        return jsonify({'error': 'Failed to populate cast/crew'}), 500

# =============================================================================
# UTILITY ROUTES
# =============================================================================

@admin_bp.route('/api/admin/health', methods=['GET'])
def admin_health_check():
    """Admin service health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'CineBrain Admin',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '4.0'
    }), 200

@admin_bp.route('/api/admin/ping', methods=['GET'])
def admin_ping():
    """Admin service ping"""
    return jsonify({
        'message': 'pong',
        'service': 'admin',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@admin_bp.route('/api/admin/services/status', methods=['GET'])
@require_admin
def get_services_status(current_user):
    """Get services status"""
    try:
        status = {
            'admin_service': 'available' if admin_service else 'unavailable',
            'dashboard_service': 'available' if dashboard_service else 'unavailable',
            'telegram_service': 'available' if telegram_service else 'unavailable',
            'user_service': 'available' if user_service else 'unavailable',
            'database': 'connected' if db else 'disconnected',
            'cache': 'available' if cache else 'unavailable',
            'email_preferences': 'available' if AdminEmailPreferences else 'unavailable',
            'support_models': {
                'tickets': 'available' if SupportTicket else 'unavailable',
                'contacts': 'available' if ContactMessage else 'unavailable',
                'issues': 'available' if IssueReport else 'unavailable'
            }
        }
        
        return jsonify({
            'status': status,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Services status error: {e}")
        return jsonify({'error': 'Failed to get services status'}), 500

# =============================================================================
# OPTIONS SUPPORT FOR ALL ROUTES
# =============================================================================

# OPTIONS support for new user routes
@admin_bp.route('/api/admin/users/segmentation', methods=['OPTIONS'])
@admin_bp.route('/api/admin/users/lifecycle-analysis', methods=['OPTIONS'])
@admin_bp.route('/api/admin/users/<int:user_id>/behavior-intelligence', methods=['OPTIONS'])
@admin_bp.route('/api/admin/users/advanced-search', methods=['OPTIONS'])
@admin_bp.route('/api/admin/users/targeted-communication', methods=['OPTIONS'])
@admin_bp.route('/api/admin/users/value-scores', methods=['OPTIONS'])
@admin_bp.route('/api/admin/users/anomalies', methods=['OPTIONS'])
@admin_bp.route('/api/admin/users/cohort-analysis', methods=['OPTIONS'])
@admin_bp.route('/api/admin/users/<int:user_id>/support-profile', methods=['OPTIONS'])
@admin_bp.route('/api/admin/users/compare', methods=['OPTIONS'])
def handle_user_options(*args, **kwargs):
    """Handle OPTIONS requests for new user endpoints"""
    return '', 200

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@admin_bp.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return jsonify({'error': 'Admin endpoint not found'}), 404

@admin_bp.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    try:
        db.session.rollback()
    except:
        pass
    return jsonify({'error': 'Internal server error in admin service'}), 500

@admin_bp.after_request
def after_request(response):
    """Add CORS headers"""
    origin = request.headers.get('Origin')
    allowed_origins = [
        'https://cinebrain.vercel.app',
        'http://127.0.0.1:5500', 
        'http://127.0.0.1:5501'
    ]
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response

# =============================================================================
# INITIALIZATION FUNCTION
# =============================================================================

def init_admin_routes(flask_app, database, models, services):
    """Initialize admin routes with dependencies"""
    global admin_service, dashboard_service, telegram_service, user_service
    global app, db, User, Content, UserInteraction, AdminRecommendation, AdminEmailPreferences
    global SupportTicket, ContactMessage, IssueReport, SupportCategory, TicketActivity, cache
    
    app = flask_app
    db = database
    User = models.get('User')
    Content = models.get('Content')
    UserInteraction = models.get('UserInteraction')
    AdminRecommendation = models.get('AdminRecommendation')
    AdminEmailPreferences = models.get('AdminEmailPreferences')
    
    SupportTicket = models.get('SupportTicket')
    ContactMessage = models.get('ContactMessage')
    IssueReport = models.get('IssueReport')
    SupportCategory = models.get('SupportCategory')
    TicketActivity = models.get('TicketActivity')
    
    cache = services.get('cache')
    
    try:
        from .service import init_admin_service
        from .dashboard import init_dashboard_service
        from .telegram import init_telegram_service
        
        admin_service = init_admin_service(app, db, models, services)
        dashboard_service = init_dashboard_service(app, db, models, services)
        telegram_service = init_telegram_service(app, db, models, services)
        
        # Initialize user management service
        try:
            user_service = init_user_service(app, db, models, services)
            logger.info(f"   - User management service: {'✓' if user_service else '✗'}")
        except Exception as e:
            logger.error(f"Failed to initialize user management service: {e}")
            user_service = None
        
        logger.info("✅ Admin routes initialized successfully")
        logger.info(f"   - Admin service: {'✓' if admin_service else '✗'}")
        logger.info(f"   - Dashboard service: {'✓' if dashboard_service else '✗'}")
        logger.info(f"   - Telegram service: {'✓' if telegram_service else '✗'}")
        logger.info(f"   - User management routes: ✓")
        logger.info(f"   - Email preferences: {'✓' if AdminEmailPreferences else '✗'}")
        logger.info(f"   - Support integration: {'✓' if SupportTicket and ContactMessage and IssueReport else '✗'}")
        logger.info(f"   - 4 Real-time monitoring endpoints: ✓")
        logger.info(f"   - Content management: ✓")
        logger.info(f"   - Recommendation system: ✓")
        logger.info(f"   - Telegram integration: ✓")
        logger.info(f"   - 10 NEW user management endpoints: ✓")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize admin routes: {e}")
        raise e