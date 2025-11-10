# support/tickets.py

from flask import request, jsonify
from datetime import datetime, timedelta
from sqlalchemy import func, desc, and_, or_
import jwt
import logging
import json
import enum
import os
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Keep enum definitions for type checking and validation
class TicketStatus(enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_USER = "waiting_for_user"
    RESOLVED = "resolved"
    CLOSED = "closed"

class TicketPriority(enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class TicketType(enum.Enum):
    GENERAL = "general"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    BILLING = "billing"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"

class TicketService:
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.User = models['User']
        self.SupportTicket = models['SupportTicket']
        self.SupportCategory = models['SupportCategory']
        self.TicketActivity = models['TicketActivity']
        
        # Enhanced email service initialization
        self.email_service = self._initialize_email_service(services)
        self.redis_client = services.get('redis_client')
        
        logger.info("✅ TicketService initialized successfully")
    
    def _initialize_email_service(self, services):
        """Initialize email service with fallbacks"""
        email_service = services.get('email_service')
        if email_service:
            return email_service
        
        # Try to get from auth module
        try:
            from auth.service import email_service as auth_email_service
            if auth_email_service and hasattr(auth_email_service, 'queue_email'):
                logger.info("✅ Email service loaded from auth module")
                return auth_email_service
        except Exception as e:
            logger.warning(f"Could not load auth email service: {e}")
        
        logger.warning("⚠️ No email service available for tickets")
        return None
        
    def generate_ticket_number(self) -> str:
        """Generate unique ticket number"""
        import random
        import string
        
        date_str = datetime.now().strftime('%Y%m%d')
        random_str = ''.join(random.choices(string.digits, k=4))
        ticket_number = f"CB-{date_str}-{random_str}"
        
        # Ensure uniqueness
        while self.SupportTicket.query.filter_by(ticket_number=ticket_number).first():
            random_str = ''.join(random.choices(string.digits, k=4))
            ticket_number = f"CB-{date_str}-{random_str}"
        
        return ticket_number
    
    def calculate_sla_deadline(self, priority: TicketPriority) -> datetime:
        """Calculate SLA deadline based on priority enum"""
        now = datetime.utcnow()
        
        sla_hours = {
            TicketPriority.URGENT: 4,
            TicketPriority.HIGH: 24,
            TicketPriority.NORMAL: 48,
            TicketPriority.LOW: 72
        }
        
        hours = sla_hours.get(priority, 48)
        return now + timedelta(hours=hours)
    
    def calculate_sla_deadline_string(self, priority_str: str) -> datetime:
        """Calculate SLA deadline based on priority string"""
        now = datetime.utcnow()
        
        sla_hours = {
            'urgent': 4,
            'high': 24,
            'normal': 48,
            'low': 72
        }
        
        hours = sla_hours.get(priority_str, 48)
        return now + timedelta(hours=hours)
    
    def get_user_from_token(self):
        """Extract user from JWT token"""
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
        
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, self.app.secret_key, algorithms=['HS256'])
            return self.User.query.get(payload.get('user_id'))
        except:
            return None
    
    def get_request_info(self):
        """Extract request information"""
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
        if ip_address:
            ip_address = ip_address.split(',')[0].strip()
        
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        return {
            'ip_address': ip_address,
            'user_agent': user_agent,
            'page_url': request.headers.get('Referer', ''),
            'browser_info': self._parse_user_agent(user_agent)
        }
    
    def _parse_user_agent(self, user_agent: str) -> str:
        """Parse user agent into readable format"""
        if not user_agent or user_agent == 'Unknown':
            return 'Unknown Browser'
        
        browsers = {
            'Chrome': 'Google Chrome',
            'Firefox': 'Mozilla Firefox',
            'Safari': 'Apple Safari',
            'Edge': 'Microsoft Edge',
            'Opera': 'Opera'
        }
        
        for key, value in browsers.items():
            if key in user_agent:
                return value
        
        return 'Unknown Browser'
    
    def create_ticket(self):
        """Create a new support ticket with enhanced error handling"""
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['subject', 'message', 'name', 'email', 'category_id']
            for field in required_fields:
                if not data.get(field):
                    return jsonify({'error': f'{field.replace("_", " ").title()} is required'}), 400
            
            # Rate limiting check
            if not self._check_rate_limit(data['email']):
                return jsonify({'error': 'Too many requests. Please try again later.'}), 429
            
            user = self.get_user_from_token()
            request_info = self.get_request_info()
            
            # Generate ticket number
            ticket_number = self.generate_ticket_number()
            
            # Use string values directly to avoid enum issues
            priority_value = data.get('priority', 'normal')
            ticket_type_value = data.get('ticket_type', 'general')
            
            # Validate priority and type values
            valid_priorities = ['low', 'normal', 'high', 'urgent']
            valid_types = ['general', 'technical', 'account', 'billing', 'feature_request', 'bug_report']
            
            if priority_value not in valid_priorities:
                priority_value = 'normal'
            if ticket_type_value not in valid_types:
                ticket_type_value = 'general'
            
            # Create ticket with string values
            ticket = self.SupportTicket(
                ticket_number=ticket_number,
                subject=data['subject'],
                description=data['message'],
                user_id=user.id if user else None,
                user_email=data['email'],
                user_name=data['name'],
                category_id=data['category_id'],
                ticket_type=ticket_type_value,
                priority=priority_value,
                status='open',
                sla_deadline=self.calculate_sla_deadline_string(priority_value),
                **request_info
            )
            
            self.db.session.add(ticket)
            self.db.session.flush()
            
            # Add activity log
            activity = self.TicketActivity(
                ticket_id=ticket.id,
                action='created',
                description=f'Ticket created by {data["name"]}',
                actor_type='user',
                actor_id=user.id if user else None,
                actor_name=data['name']
            )
            self.db.session.add(activity)
            self.db.session.commit()
            
            # Send notifications
            self._send_user_confirmation(ticket, data)
            self._send_admin_notification(ticket, data)
            
            logger.info(f"Support ticket {ticket_number} created for {data['email']}")
            
            return jsonify({
                'success': True,
                'message': 'Support ticket created successfully. You will receive a confirmation email shortly.',
                'ticket_number': ticket_number,
                'ticket_id': ticket.id,
                'sla_deadline': ticket.sla_deadline.isoformat() if ticket.sla_deadline else None
            }), 201
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error creating support ticket: {e}")
            return jsonify({'error': 'Failed to create support ticket'}), 500
    
    def get_ticket(self, ticket_number):
        """Get ticket details by number"""
        try:
            user = self.get_user_from_token()
            
            # Build query
            query = self.SupportTicket.query.filter_by(ticket_number=ticket_number)
            
            # If not admin, only show user's own tickets
            if user and not getattr(user, 'is_admin', False):
                query = query.filter_by(user_id=user.id)
            elif not user:
                # For anonymous users, they need to provide email
                email = request.args.get('email')
                if not email:
                    return jsonify({'error': 'Email required for ticket access'}), 400
                query = query.filter_by(user_email=email)
            
            ticket = query.first()
            if not ticket:
                return jsonify({'error': 'Ticket not found'}), 404
            
            # Get category info
            category = self.SupportCategory.query.get(ticket.category_id)
            
            # Get activities
            activities = self.TicketActivity.query.filter_by(
                ticket_id=ticket.id
            ).order_by(self.TicketActivity.created_at.desc()).all()
            
            return jsonify({
                'ticket': {
                    'id': ticket.id,
                    'ticket_number': ticket.ticket_number,
                    'subject': ticket.subject,
                    'description': ticket.description,
                    'status': ticket.status,
                    'priority': ticket.priority,
                    'ticket_type': ticket.ticket_type,
                    'category': {
                        'id': category.id,
                        'name': category.name,
                        'icon': category.icon
                    } if category else None,
                    'user_name': ticket.user_name,
                    'user_email': ticket.user_email,
                    'created_at': ticket.created_at.isoformat(),
                    'sla_deadline': ticket.sla_deadline.isoformat() if ticket.sla_deadline else None,
                    'sla_breached': ticket.sla_breached,
                    'first_response_at': ticket.first_response_at.isoformat() if ticket.first_response_at else None,
                    'resolved_at': ticket.resolved_at.isoformat() if ticket.resolved_at else None,
                    'browser_info': ticket.browser_info,
                    'page_url': ticket.page_url
                },
                'activities': [
                    {
                        'id': activity.id,
                        'action': activity.action,
                        'description': activity.description,
                        'actor_type': activity.actor_type,
                        'actor_name': activity.actor_name,
                        'created_at': activity.created_at.isoformat()
                    }
                    for activity in activities
                ]
            }), 200
            
        except Exception as e:
            logger.error(f"Error fetching ticket {ticket_number}: {e}")
            return jsonify({'error': 'Failed to fetch ticket'}), 500
    
    def _check_rate_limit(self, email: str) -> bool:
        """Check rate limit for ticket creation"""
        if not self.redis_client:
            return True
        
        try:
            key = f"ticket_rate_limit:{email}"
            current = self.redis_client.get(key)
            
            if current is None:
                self.redis_client.setex(key, 900, 1)  # 15 minutes
                return True
            
            if int(current) >= 3:  # Max 3 tickets per 15 minutes
                return False
            
            self.redis_client.incr(key)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True
    
    def _send_user_confirmation(self, ticket, data):
        """Send confirmation email to user"""
        try:
            if not self.email_service:
                logger.warning("Email service not available - cannot send user confirmation")
                return
            
            from auth.support_mail_templates import get_support_template
            
            category = self.SupportCategory.query.get(ticket.category_id)
            
            html, text = get_support_template(
                'ticket_created',
                ticket_number=ticket.ticket_number,
                user_name=data['name'],
                subject=ticket.subject,
                priority=ticket.priority,
                category=category.name if category else 'General Support'
            )
            
            self.email_service.queue_email(
                to=data['email'],
                subject=f"Support Ticket Created #{ticket.ticket_number} - CineBrain",
                html=html,
                text=text,
                priority='high',
                to_name=data['name']
            )
            
            logger.info(f"✅ User confirmation email queued for {data['email']}")
            
        except Exception as e:
            logger.error(f"❌ Error sending user confirmation: {e}")
    
    def _send_admin_notification(self, ticket, data):
        """Send admin notification email"""
        try:
            if not self.email_service:
                logger.warning("Email service not available - cannot send admin notification")
                return
            
            admin_email = os.environ.get('ADMIN_EMAIL', 'srinathnulidonda.dev@gmail.com')
            
            from auth.support_mail_templates import get_support_template
            
            category = self.SupportCategory.query.get(ticket.category_id)
            
            # Add link to admin dashboard
            admin_link = f"{os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')}/admin/support/ticket/{ticket.ticket_number}"
            
            html, text = get_support_template(
                'admin_notification',
                notification_type='new_ticket',
                title=f"New Support Ticket #{ticket.ticket_number}",
                message=f"""
                <p><strong>New support ticket received:</strong></p>
                <ul>
                    <li><strong>Ticket:</strong> #{ticket.ticket_number}</li>
                    <li><strong>Subject:</strong> {ticket.subject}</li>
                    <li><strong>From:</strong> {data['name']} ({data['email']})</li>
                    <li><strong>Category:</strong> {category.name if category else 'General'}</li>
                    <li><strong>Priority:</strong> {ticket.priority.upper()}</li>
                    <li><strong>Status:</strong> {ticket.status.upper()}</li>
                </ul>
                <p><strong>Message:</strong></p>
                <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin: 15px 0;">
                    {ticket.description}
                </div>
                <p style="margin-top: 20px;">
                    <a href="{admin_link}" style="background: #113CCF; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">
                        View in Admin Dashboard
                    </a>
                </p>
                """,
                ticket_number=ticket.ticket_number,
                user_email=data['email']
            )
            
            self.email_service.queue_email(
                to=admin_email,
                subject=f"🎫 New Support Ticket #{ticket.ticket_number} - CineBrain Admin",
                html=html,
                text=text,
                priority='urgent' if ticket.priority == 'urgent' else 'high',
                to_name='CineBrain Admin'
            )
            
            logger.info(f"✅ Admin notification email queued for ticket {ticket.ticket_number}")
            
        except Exception as e:
            logger.error(f"❌ Error sending admin notification: {e}")

def init_ticket_service(app, db, models, services):
    """Initialize ticket service"""
    return TicketService(app, db, models, services)