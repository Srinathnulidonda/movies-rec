# support/routes.py

from flask import Blueprint, request, jsonify
from datetime import datetime
import logging
import jwt
import os
from functools import wraps

logger = logging.getLogger(__name__)

support_bp = Blueprint('support', __name__)

# Global services - will be initialized by init_support_routes
ticket_service = None
contact_service = None
issue_service = None
app = None
db = None
User = None
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

def admin_required(f):
    """Decorator for admin-only endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_user_from_token()
        if not user or not getattr(user, 'is_admin', False):
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

# Health and Info Routes
@support_bp.route('/api/support/health', methods=['GET'])
def support_health():
    """Support service health check"""
    try:
        # Test database connection
        if db and User:
            User.query.limit(1).first()
        
        # Check service availability
        services_status = {
            'tickets': ticket_service is not None,
            'contact': contact_service is not None,
            'issues': issue_service is not None
        }
        
        # Get environment status
        env_status = {
            'frontend_url': bool(os.environ.get('FRONTEND_URL')),
            'admin_email': bool(os.environ.get('ADMIN_EMAIL')),
            'cloudinary_configured': all([
                os.environ.get('CLOUDINARY_CLOUD_NAME'),
                os.environ.get('CLOUDINARY_API_KEY'),
                os.environ.get('CLOUDINARY_API_SECRET')
            ])
        }
        
        # Get database stats
        db_stats = {}
        try:
            if 'SupportTicket' in globals():
                SupportTicket = globals()['SupportTicket']
                db_stats = {
                    'total_tickets': SupportTicket.query.count(),
                    'open_tickets': SupportTicket.query.filter_by(status='open').count(),
                    'resolved_tickets': SupportTicket.query.filter_by(status='resolved').count()
                }
        except:
            db_stats = {'error': 'Could not fetch database stats'}
        
        return jsonify({
            'status': 'healthy',
            'service': 'support',
            'version': '2.0.0',
            'services': services_status,
            'environment': env_status,
            'database': db_stats,
            'features': {
                'ticket_management': True,
                'contact_forms': True,
                'issue_reporting': True,
                'file_uploads': True,
                'email_notifications': True,
                'admin_integration': True,
                'rate_limiting': True
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Support health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'service': 'support',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@support_bp.route('/api/support/stats', methods=['GET'])
@admin_required
def get_support_stats():
    """Get support statistics for admin dashboard"""
    try:
        stats = {}
        
        # Get ticket stats
        if ticket_service and 'SupportTicket' in globals():
            SupportTicket = globals()['SupportTicket']
            
            # Basic counts
            stats['tickets'] = {
                'total': SupportTicket.query.count(),
                'open': SupportTicket.query.filter_by(status='open').count(),
                'in_progress': SupportTicket.query.filter_by(status='in_progress').count(),
                'resolved': SupportTicket.query.filter_by(status='resolved').count(),
                'closed': SupportTicket.query.filter_by(status='closed').count()
            }
            
            # Priority breakdown
            stats['priority'] = {
                'urgent': SupportTicket.query.filter_by(priority='urgent').count(),
                'high': SupportTicket.query.filter_by(priority='high').count(),
                'normal': SupportTicket.query.filter_by(priority='normal').count(),
                'low': SupportTicket.query.filter_by(priority='low').count()
            }
            
            # Recent tickets (last 7 days)
            from datetime import timedelta
            week_ago = datetime.utcnow() - timedelta(days=7)
            stats['recent'] = {
                'this_week': SupportTicket.query.filter(
                    SupportTicket.created_at >= week_ago
                ).count(),
                'overdue': SupportTicket.query.filter(
                    SupportTicket.sla_deadline < datetime.utcnow(),
                    SupportTicket.status.in_(['open', 'in_progress'])
                ).count()
            }
        
        # Get contact message stats
        if contact_service and 'ContactMessage' in globals():
            ContactMessage = globals()['ContactMessage']
            stats['contact'] = {
                'total': ContactMessage.query.count(),
                'unread': ContactMessage.query.filter_by(is_read=False).count()
            }
        
        # Get issue report stats
        if issue_service and 'IssueReport' in globals():
            IssueReport = globals()['IssueReport']
            stats['issues'] = {
                'total': IssueReport.query.count(),
                'unresolved': IssueReport.query.filter_by(is_resolved=False).count(),
                'critical': IssueReport.query.filter_by(severity='critical').count()
            }
        
        return jsonify({
            'success': True,
            'stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching support stats: {e}")
        return jsonify({'error': 'Failed to fetch support statistics'}), 500

# Ticket Routes
@support_bp.route('/api/support/tickets', methods=['POST', 'OPTIONS'])
def create_ticket():
    """Create a new support ticket"""
    if request.method == 'OPTIONS':
        return '', 200
    
    if not ticket_service:
        return jsonify({'error': 'Ticket service not available'}), 503
    
    return ticket_service.create_ticket()

@support_bp.route('/api/support/tickets/<ticket_number>', methods=['GET'])
def get_ticket(ticket_number):
    """Get ticket by number"""
    if not ticket_service:
        return jsonify({'error': 'Ticket service not available'}), 503
    
    return ticket_service.get_ticket(ticket_number)

@support_bp.route('/api/support/tickets', methods=['GET'])
@admin_required
def list_tickets():
    """List all tickets for admin"""
    try:
        if not ticket_service or 'SupportTicket' not in globals():
            return jsonify({'error': 'Ticket service not available'}), 503
        
        SupportTicket = globals()['SupportTicket']
        SupportCategory = globals()['SupportCategory']
        
        # Get query parameters
        status = request.args.get('status')
        priority = request.args.get('priority')
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 25)), 100)
        search = request.args.get('search', '').strip()
        
        # Build query
        query = SupportTicket.query
        
        if status:
            query = query.filter_by(status=status)
        
        if priority:
            query = query.filter_by(priority=priority)
        
        if search:
            query = query.filter(
                db.or_(
                    SupportTicket.ticket_number.contains(search),
                    SupportTicket.subject.contains(search),
                    SupportTicket.user_email.contains(search),
                    SupportTicket.user_name.contains(search)
                )
            )
        
        # Order by created_at desc
        query = query.order_by(SupportTicket.created_at.desc())
        
        # Paginate
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        tickets = []
        for ticket in pagination.items:
            category = SupportCategory.query.get(ticket.category_id)
            tickets.append({
                'id': ticket.id,
                'ticket_number': ticket.ticket_number,
                'subject': ticket.subject,
                'status': ticket.status,
                'priority': ticket.priority,
                'ticket_type': ticket.ticket_type,
                'user_name': ticket.user_name,
                'user_email': ticket.user_email,
                'category': {
                    'id': category.id,
                    'name': category.name,
                    'icon': category.icon
                } if category else None,
                'created_at': ticket.created_at.isoformat(),
                'sla_deadline': ticket.sla_deadline.isoformat() if ticket.sla_deadline else None,
                'sla_breached': ticket.sla_breached,
                'first_response_at': ticket.first_response_at.isoformat() if ticket.first_response_at else None
            })
        
        return jsonify({
            'success': True,
            'tickets': tickets,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing tickets: {e}")
        return jsonify({'error': 'Failed to fetch tickets'}), 500

@support_bp.route('/api/support/tickets/<int:ticket_id>/status', methods=['PUT'])
@admin_required
def update_ticket_status(ticket_id):
    """Update ticket status"""
    try:
        if not ticket_service or 'SupportTicket' not in globals():
            return jsonify({'error': 'Ticket service not available'}), 503
        
        data = request.get_json()
        new_status = data.get('status')
        update_message = data.get('message', '')
        
        if not new_status:
            return jsonify({'error': 'Status is required'}), 400
        
        SupportTicket = globals()['SupportTicket']
        TicketActivity = globals()['TicketActivity']
        
        ticket = SupportTicket.query.get_or_404(ticket_id)
        old_status = ticket.status
        
        # Update status
        ticket.status = new_status
        
        # Update timestamps based on status
        if new_status == 'resolved' and not ticket.resolved_at:
            ticket.resolved_at = datetime.utcnow()
        elif new_status == 'closed' and not ticket.closed_at:
            ticket.closed_at = datetime.utcnow()
        
        # Add activity
        user = get_user_from_token()
        activity = TicketActivity(
            ticket_id=ticket.id,
            action='status_updated',
            description=f'Status changed from {old_status} to {new_status}',
            old_value=old_status,
            new_value=new_status,
            actor_type='admin',
            actor_id=user.id,
            actor_name=user.username
        )
        
        db.session.add(activity)
        db.session.commit()
        
        # Send status update email to user
        try:
            from auth.service import email_service
            from auth.support_mail_templates import get_support_template
            
            if email_service:
                html, text = get_support_template(
                    'ticket_status_updated',
                    ticket_number=ticket.ticket_number,
                    user_name=ticket.user_name,
                    old_status=old_status,
                    new_status=new_status,
                    update_message=update_message,
                    staff_name=user.username
                )
                
                email_service.queue_email(
                    to=ticket.user_email,
                    subject=f"Ticket Status Updated #{ticket.ticket_number} - CineBrain",
                    html=html,
                    text=text,
                    priority='high',
                    to_name=ticket.user_name
                )
        except Exception as e:
            logger.warning(f"Failed to send status update email: {e}")
        
        return jsonify({
            'success': True,
            'message': 'Ticket status updated successfully',
            'ticket': {
                'id': ticket.id,
                'ticket_number': ticket.ticket_number,
                'status': ticket.status,
                'updated_at': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating ticket status: {e}")
        return jsonify({'error': 'Failed to update ticket status'}), 500

# Contact Routes
@support_bp.route('/api/support/contact', methods=['POST', 'OPTIONS'])
def submit_contact():
    """Submit contact form"""
    if request.method == 'OPTIONS':
        return '', 200
    
    if not contact_service:
        return jsonify({'error': 'Contact service not available'}), 503
    
    return contact_service.submit_contact()

@support_bp.route('/api/support/contact/messages', methods=['GET'])
@admin_required
def list_contact_messages():
    """List contact messages for admin"""
    try:
        if not contact_service or 'ContactMessage' not in globals():
            return jsonify({'error': 'Contact service not available'}), 503
        
        ContactMessage = globals()['ContactMessage']
        
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 25)), 100)
        unread_only = request.args.get('unread_only', 'false').lower() == 'true'
        search = request.args.get('search', '').strip()
        
        # Build query
        query = ContactMessage.query
        
        if unread_only:
            query = query.filter_by(is_read=False)
        
        if search:
            query = query.filter(
                db.or_(
                    ContactMessage.name.contains(search),
                    ContactMessage.email.contains(search),
                    ContactMessage.subject.contains(search)
                )
            )
        
        # Order by created_at desc
        query = query.order_by(ContactMessage.created_at.desc())
        
        # Paginate
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        messages = []
        for message in pagination.items:
            messages.append({
                'id': message.id,
                'name': message.name,
                'email': message.email,
                'subject': message.subject,
                'message': message.message,
                'phone': message.phone,
                'company': message.company,
                'is_read': message.is_read,
                'admin_notes': message.admin_notes,
                'created_at': message.created_at.isoformat(),
                'ip_address': message.ip_address
            })
        
        return jsonify({
            'success': True,
            'messages': messages,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing contact messages: {e}")
        return jsonify({'error': 'Failed to fetch contact messages'}), 500

@support_bp.route('/api/support/contact/messages/<int:message_id>/read', methods=['PUT'])
@admin_required
def mark_message_read(message_id):
    """Mark contact message as read"""
    try:
        if 'ContactMessage' not in globals():
            return jsonify({'error': 'Contact service not available'}), 503
        
        ContactMessage = globals()['ContactMessage']
        message = ContactMessage.query.get_or_404(message_id)
        
        message.is_read = True
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Message marked as read'
        }), 200
        
    except Exception as e:
        logger.error(f"Error marking message as read: {e}")
        return jsonify({'error': 'Failed to update message status'}), 500

# Issue Report Routes
@support_bp.route('/api/support/report-issue', methods=['POST', 'OPTIONS'])
def report_issue():
    """Report an issue with file uploads"""
    if request.method == 'OPTIONS':
        return '', 200
    
    if not issue_service:
        return jsonify({'error': 'Issue service not available'}), 503
    
    return issue_service.report_issue()

@support_bp.route('/api/support/issues', methods=['GET'])
@admin_required
def list_issues():
    """List issue reports for admin"""
    try:
        if not issue_service or 'IssueReport' not in globals():
            return jsonify({'error': 'Issue service not available'}), 503
        
        IssueReport = globals()['IssueReport']
        
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 25)), 100)
        severity = request.args.get('severity')
        unresolved_only = request.args.get('unresolved_only', 'false').lower() == 'true'
        search = request.args.get('search', '').strip()
        
        # Build query
        query = IssueReport.query
        
        if severity:
            query = query.filter_by(severity=severity)
        
        if unresolved_only:
            query = query.filter_by(is_resolved=False)
        
        if search:
            query = query.filter(
                db.or_(
                    IssueReport.issue_id.contains(search),
                    IssueReport.issue_title.contains(search),
                    IssueReport.email.contains(search),
                    IssueReport.name.contains(search)
                )
            )
        
        # Order by severity (critical first) then created_at desc
        severity_order = {
            'critical': 0,
            'high': 1,
            'normal': 2,
            'low': 3
        }
        
        query = query.order_by(
            db.case(severity_order, value=IssueReport.severity),
            IssueReport.created_at.desc()
        )
        
        # Paginate
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        issues = []
        for issue in pagination.items:
            issues.append({
                'id': issue.id,
                'issue_id': issue.issue_id,
                'name': issue.name,
                'email': issue.email,
                'issue_type': issue.issue_type,
                'severity': issue.severity,
                'issue_title': issue.issue_title,
                'description': issue.description[:200] + '...' if len(issue.description) > 200 else issue.description,
                'browser_version': issue.browser_version,
                'device_os': issue.device_os,
                'page_url_reported': issue.page_url_reported,
                'screenshots': issue.screenshots,
                'ticket_id': issue.ticket_id,
                'is_resolved': issue.is_resolved,
                'created_at': issue.created_at.isoformat(),
                'resolved_at': issue.resolved_at.isoformat() if issue.resolved_at else None
            })
        
        return jsonify({
            'success': True,
            'issues': issues,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing issues: {e}")
        return jsonify({'error': 'Failed to fetch issues'}), 500

@support_bp.route('/api/support/issues/<int:issue_id>/resolve', methods=['PUT'])
@admin_required
def resolve_issue(issue_id):
    """Mark issue as resolved"""
    try:
        if 'IssueReport' not in globals():
            return jsonify({'error': 'Issue service not available'}), 503
        
        data = request.get_json()
        admin_notes = data.get('admin_notes', '')
        
        IssueReport = globals()['IssueReport']
        issue = IssueReport.query.get_or_404(issue_id)
        
        issue.is_resolved = True
        issue.resolved_at = datetime.utcnow()
        issue.admin_notes = admin_notes
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Issue marked as resolved'
        }), 200
        
    except Exception as e:
        logger.error(f"Error resolving issue: {e}")
        return jsonify({'error': 'Failed to resolve issue'}), 500

# Support Categories Routes
@support_bp.route('/api/support/categories', methods=['GET'])
def get_support_categories():
    """Get support categories"""
    try:
        if 'SupportCategory' not in globals():
            # Return default categories if not in database
            default_categories = [
                {'id': 1, 'name': 'Account & Login', 'description': 'Issues with account creation, login, password reset', 'icon': '👤'},
                {'id': 2, 'name': 'Technical Issues', 'description': 'App crashes, loading issues, performance problems', 'icon': '🔧'},
                {'id': 3, 'name': 'Features & Functions', 'description': 'How to use features, feature requests', 'icon': '⚡'},
                {'id': 4, 'name': 'Content & Recommendations', 'description': 'Issues with movies, shows, recommendations', 'icon': '🎬'},
                {'id': 5, 'name': 'General Support', 'description': 'Other questions and general inquiries', 'icon': '❓'}
            ]
            return jsonify({'categories': default_categories}), 200
        
        SupportCategory = globals()['SupportCategory']
        categories = SupportCategory.query.filter_by(is_active=True).order_by(SupportCategory.sort_order).all()
        
        return jsonify({
            'categories': [
                {
                    'id': cat.id,
                    'name': cat.name,
                    'description': cat.description,
                    'icon': cat.icon
                }
                for cat in categories
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        return jsonify({'error': 'Failed to retrieve categories'}), 500

# Admin Management Routes
@support_bp.route('/api/support/dashboard', methods=['GET'])
@admin_required
def get_admin_dashboard():
    """Get admin dashboard data"""
    try:
        dashboard_data = {}
        
        # Recent tickets
        if 'SupportTicket' in globals():
            SupportTicket = globals()['SupportTicket']
            SupportCategory = globals()['SupportCategory']
            
            recent_tickets = SupportTicket.query.order_by(
                SupportTicket.created_at.desc()
            ).limit(10).all()
            
            dashboard_data['recent_tickets'] = []
            for ticket in recent_tickets:
                category = SupportCategory.query.get(ticket.category_id) if SupportCategory else None
                dashboard_data['recent_tickets'].append({
                    'id': ticket.id,
                    'ticket_number': ticket.ticket_number,
                    'subject': ticket.subject,
                    'status': ticket.status,
                    'priority': ticket.priority,
                    'user_name': ticket.user_name,
                    'category': category.name if category else 'General',
                    'created_at': ticket.created_at.isoformat(),
                    'sla_breached': ticket.sla_breached
                })
        
        # Recent contact messages
        if 'ContactMessage' in globals():
            ContactMessage = globals()['ContactMessage']
            recent_messages = ContactMessage.query.filter_by(
                is_read=False
            ).order_by(ContactMessage.created_at.desc()).limit(5).all()
            
            dashboard_data['unread_messages'] = [
                {
                    'id': msg.id,
                    'name': msg.name,
                    'email': msg.email,
                    'subject': msg.subject,
                    'created_at': msg.created_at.isoformat()
                }
                for msg in recent_messages
            ]
        
        # Critical issues
        if 'IssueReport' in globals():
            IssueReport = globals()['IssueReport']
            critical_issues = IssueReport.query.filter(
                IssueReport.severity.in_(['critical', 'high']),
                IssueReport.is_resolved == False
            ).order_by(IssueReport.created_at.desc()).limit(5).all()
            
            dashboard_data['critical_issues'] = [
                {
                    'id': issue.id,
                    'issue_id': issue.issue_id,
                    'issue_title': issue.issue_title,
                    'severity': issue.severity,
                    'name': issue.name,
                    'created_at': issue.created_at.isoformat()
                }
                for issue in critical_issues
            ]
        
        return jsonify({
            'success': True,
            'dashboard': dashboard_data,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {e}")
        return jsonify({'error': 'Failed to fetch dashboard data'}), 500

# Webhook Routes - UPDATED to use admin notification service through existing services
@support_bp.route('/api/webhooks/support/ticket-created', methods=['POST'])
def webhook_ticket_created():
    """Handle ticket creation webhook"""
    try:
        data = request.get_json()
        ticket_id = data.get('ticket_id')
        
        if ticket_id and 'SupportTicket' in globals():
            SupportTicket = globals()['SupportTicket']
            ticket = SupportTicket.query.get(ticket_id)
            if ticket and ticket_service:
                # Use the ticket service's admin notification service
                try:
                    if hasattr(ticket_service, 'admin_notification_service') and ticket_service.admin_notification_service:
                        ticket_service.admin_notification_service.notify_new_ticket(ticket)
                    else:
                        logger.warning("Admin notification service not available for webhook")
                except Exception as e:
                    logger.error(f"CineBrain error handling new ticket notification: {e}")
        
        return jsonify({'success': True, 'cinebrain_service': 'support_webhook'}), 200
    except Exception as e:
        logger.error(f"CineBrain webhook error: {e}")
        return jsonify({'error': 'CineBrain webhook processing failed'}), 500

@support_bp.route('/api/webhooks/support/feedback-created', methods=['POST'])
def webhook_feedback_created():
    """Handle feedback creation webhook"""
    try:
        data = request.get_json()
        feedback_id = data.get('feedback_id')
        
        if feedback_id and 'ContactMessage' in globals():
            ContactMessage = globals()['ContactMessage']
            feedback = ContactMessage.query.get(feedback_id)
            if feedback and contact_service:
                # Use the contact service's admin notification service
                try:
                    if hasattr(contact_service, 'admin_notification_service') and contact_service.admin_notification_service:
                        contact_service.admin_notification_service.notify_feedback_received(feedback)
                    else:
                        logger.warning("Admin notification service not available for webhook")
                except Exception as e:
                    logger.error(f"CineBrain error handling new feedback notification: {e}")
        
        return jsonify({'success': True, 'cinebrain_service': 'support_webhook'}), 200
    except Exception as e:
        logger.error(f"CineBrain feedback webhook error: {e}")
        return jsonify({'error': 'CineBrain webhook processing failed'}), 500

# Error Handlers
@support_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Support endpoint not found'}), 404

@support_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error in support service'}), 500

# CORS Headers
@support_bp.after_request
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
def init_support_routes(flask_app, database, models, services):
    """Initialize support routes with dependencies"""
    global ticket_service, contact_service, issue_service, app, db, User, cache
    
    app = flask_app
    db = database
    User = models.get('User')
    cache = services.get('cache')
    
    # Make models available globally for the routes
    for model_name, model_class in models.items():
        if model_name.startswith('Support') or model_name in ['ContactMessage', 'IssueReport', 'TicketActivity']:
            globals()[model_name] = model_class
    
    # Initialize individual services
    try:
        from .tickets import init_ticket_service
        from .contact import init_contact_service
        from .report_issues import init_issue_service
        
        ticket_service = init_ticket_service(app, db, models, services)
        contact_service = init_contact_service(app, db, models, services)
        issue_service = init_issue_service(app, db, models, services)
        
        logger.info("✅ Support routes initialized successfully")
        logger.info(f"   - Ticket service: {'✓' if ticket_service else '✗'}")
        logger.info(f"   - Contact service: {'✓' if contact_service else '✗'}")
        logger.info(f"   - Issue service: {'✓' if issue_service else '✗'}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize support routes: {e}")
        raise e