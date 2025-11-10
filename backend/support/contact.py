# support/contact.py

from flask import request, jsonify
from datetime import datetime
import logging
import re
import jwt

logger = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

class ContactService:
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.User = models['User']
        self.ContactMessage = models.get('ContactMessage')  # Will create this model
        self.email_service = services.get('email_service')  # Brevo from auth
        self.redis_client = services.get('redis_client')
    
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
        
        return {
            'ip_address': ip_address,
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'page_url': request.headers.get('Referer', '')
        }
    
    def submit_contact(self):
        """Handle contact form submission"""
        try:
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['name', 'email', 'subject', 'message']
            for field in required_fields:
                if not data.get(field):
                    return jsonify({'error': f'{field.replace("_", " ").title()} is required'}), 400
            
            # Validate email format
            if not EMAIL_REGEX.match(data['email']):
                return jsonify({'error': 'Please provide a valid email address'}), 400
            
            # Rate limiting
            if not self._check_rate_limit(data['email']):
                return jsonify({'error': 'Too many contact form submissions. Please try again in 15 minutes.'}), 429
            
            user = self.get_user_from_token()
            request_info = self.get_request_info()
            
            # Create contact message record
            contact_message = self.ContactMessage(
                name=data['name'],
                email=data['email'],
                subject=data['subject'],
                message=data['message'],
                user_id=user.id if user else None,
                phone=data.get('phone'),
                company=data.get('company'),
                **request_info
            )
            
            self.db.session.add(contact_message)
            self.db.session.commit()
            
            # Send user confirmation
            self._send_user_confirmation(data)
            
            # Send admin notification
            self._send_admin_notification(contact_message, data)
            
            logger.info(f"Contact form submitted by {data['email']}")
            
            return jsonify({
                'success': True,
                'message': 'Thank you for your message! We will get back to you soon.',
                'contact_id': contact_message.id
            }), 201
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Error processing contact form: {e}")
            return jsonify({'error': 'Failed to send your message. Please try again.'}), 500
    
    def _check_rate_limit(self, email: str) -> bool:
        """Check rate limit for contact form"""
        if not self.redis_client:
            return True
        
        try:
            key = f"contact_rate_limit:{email}"
            current = self.redis_client.get(key)
            
            if current is None:
                self.redis_client.setex(key, 900, 1)  # 15 minutes
                return True
            
            if int(current) >= 5:  # Max 5 messages per 15 minutes
                return False
            
            self.redis_client.incr(key)
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True
    
    def _send_user_confirmation(self, data):
        """Send confirmation email to user"""
        try:
            if not self.email_service:
                return
            
            from auth.support_mail_templates import get_support_template
            
            html, text = get_support_template(
                'contact_received',
                user_name=data['name'],
                subject=data['subject']
            )
            
            self.email_service.queue_email(
                to=data['email'],
                subject="Message Received - CineBrain Support",
                html=html,
                text=text,
                priority='normal',
                to_name=data['name']
            )
            
            logger.info(f"Contact confirmation email queued for {data['email']}")
            
        except Exception as e:
            logger.error(f"Error sending contact confirmation: {e}")
    
    def _send_admin_notification(self, contact_message, data):
        """Send admin notification email"""
        try:
            if not self.email_service:
                return
            
            import os
            admin_email = os.environ.get('ADMIN_EMAIL', 'srinathnulidonda.dev@gmail.com')
            
            from auth.support_mail_templates import get_support_template
            
            html, text = get_support_template(
                'admin_notification',
                notification_type='contact',
                title=f"New Contact Message: {data['subject']}",
                message=f"""
                <p><strong>New contact message received:</strong></p>
                <ul>
                    <li><strong>From:</strong> {data['name']} ({data['email']})</li>
                    <li><strong>Subject:</strong> {data['subject']}</li>
                    {f"<li><strong>Phone:</strong> {data.get('phone')}</li>" if data.get('phone') else ''}
                    {f"<li><strong>Company:</strong> {data.get('company')}</li>" if data.get('company') else ''}
                    <li><strong>IP Address:</strong> {contact_message.ip_address}</li>
                    <li><strong>User Agent:</strong> {contact_message.user_agent}</li>
                </ul>
                <p><strong>Message:</strong></p>
                <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #28a745; margin: 15px 0;">
                    {data['message']}
                </div>
                """,
                user_email=data['email']
            )
            
            self.email_service.queue_email(
                to=admin_email,
                subject=f"📧 New Contact: {data['subject']} - CineBrain Admin",
                html=html,
                text=text,
                priority='high',
                to_name='CineBrain Admin'
            )
            
            logger.info(f"Admin notification email queued for contact from {data['email']}")
            
        except Exception as e:
            logger.error(f"Error sending admin notification: {e}")

def init_contact_service(app, db, models, services):
    """Initialize contact service"""
    return ContactService(app, db, models, services)