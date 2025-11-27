# support/contact.py

from flask import request, jsonify
from datetime import datetime
import logging
import re
import jwt
import os

logger = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

class ContactService:
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.User = models['User']
        self.ContactMessage = models.get('ContactMessage')
        
        # Enhanced email service initialization
        self.email_service = self._initialize_email_service(services)
        self.redis_client = services.get('redis_client')
        
        # Get admin notification service if available
        self.admin_notification_service = services.get('admin_notification_service')
        
        logger.info("✅ ContactService initialized successfully")
    
    def _initialize_email_service(self, services):
        """Initialize email service with fallbacks"""
        email_service = services.get('email_service')
        if email_service:
            return email_service
        
        try:
            from auth.service import email_service as auth_email_service
            if auth_email_service and hasattr(auth_email_service, 'queue_email'):
                logger.info("✅ Email service loaded from auth module for contact")
                return auth_email_service
        except Exception as e:
            logger.warning(f"Could not load auth email service for contact: {e}")
        
        logger.warning("⚠️ No email service available for contact")
        return None
    
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
        """Handle contact form submission with enhanced notifications - FIXED"""
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
            
            # Get the timestamp for response
            submitted_at = contact_message.created_at
            formatted_time = submitted_at.strftime('%Y-%m-%d at %H:%M UTC')
            
            # Send notifications
            self._send_user_confirmation(data)
            self._send_admin_notification_enhanced(contact_message, data)
            
            logger.info(f"✅ Contact form submitted by {data['email']} at {formatted_time}")
            
            return jsonify({
                'success': True,
                'message': 'Thank you for your message! We will get back to you soon.',
                'contact_id': contact_message.id,
                'submitted_at': submitted_at.isoformat(),
                'submitted_time': formatted_time,
                'reference_number': f"CB-CONTACT-{contact_message.id:06d}"
            }), 201
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"❌ Error processing contact form: {e}")
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
                logger.warning("Email service not available - cannot send user confirmation")
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
            
            logger.info(f"✅ Contact confirmation email queued for {data['email']}")
            
        except Exception as e:
            logger.error(f"❌ Error sending contact confirmation: {e}")
    
    def _send_admin_notification(self, contact_message, data):
        """Send admin notification email - DEPRECATED, use enhanced version"""
        self._send_admin_notification_enhanced(contact_message, data)
    
    def _send_admin_notification_enhanced(self, contact_message, data):
        """Send enhanced admin notification email - FIXED"""
        try:
            # Check if the message is from a potential partner/business inquiry
            is_business_inquiry = any([
                data.get('company'),
                'partnership' in data.get('subject', '').lower(),
                'business' in data.get('subject', '').lower(),
                'collaborate' in data.get('message', '').lower()
            ])
            
            # Try using admin notification service first for general feedback
            if self.admin_notification_service and hasattr(self.admin_notification_service, 'notify_feedback_received'):
                try:
                    # Create a feedback-like object for the notification service
                    class ContactAsFeedback:
                        def __init__(self, contact, data):
                            self.id = contact.id
                            self.user_name = data['name']
                            self.user_email = data['email']
                            self.subject = data['subject']
                            self.feedback_type = 'business_inquiry' if is_business_inquiry else 'contact'
                            self.rating = None
                    
                    feedback_obj = ContactAsFeedback(contact_message, data)
                    self.admin_notification_service.notify_feedback_received(feedback_obj)
                    logger.info(f"✅ Admin notification sent via notification service for contact from {data['email']}")
                except Exception as e:
                    logger.warning(f"Admin notification service failed, using direct email: {e}")
            
            # Always send direct email for contact forms (in addition to notification service)
            if not self.email_service:
                logger.warning("Email service not available - cannot send admin notification")
                return
            
            # Get admin email(s)
            admin_emails = []
            
            # Add environment admin email
            env_admin_email = os.environ.get('ADMIN_EMAIL', 'srinathnulidonda.dev@gmail.com')
            if env_admin_email:
                admin_emails.append(env_admin_email)
            
            # Add database admin emails
            admin_users = self.User.query.filter_by(is_admin=True).all()
            for admin_user in admin_users:
                if admin_user.email and admin_user.email not in admin_emails:
                    admin_emails.append(admin_user.email)
            
            if not admin_emails:
                logger.warning("No admin emails configured for notifications")
                return
            
            admin_link = f"{os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')}/admin/support/contact/{contact_message.id}"
            
            from auth.support_mail_templates import get_support_template
            
            # Determine priority based on content
            priority = 'urgent' if is_business_inquiry else 'high'
            
            html, text = get_support_template(
                'admin_notification',
                notification_type='contact',
                title=f"{'🤝 Business Inquiry' if is_business_inquiry else '📧 New Contact'}: {data['subject']}",
                message=f"""
                <p><strong>{'Business inquiry' if is_business_inquiry else 'New contact message'} received:</strong></p>
                <ul>
                    <li><strong>From:</strong> {data['name']} ({data['email']})</li>
                    <li><strong>Subject:</strong> {data['subject']}</li>
                    {f"<li><strong>Phone:</strong> {data.get('phone')}</li>" if data.get('phone') else ''}
                    {f"<li><strong>Company:</strong> {data.get('company')}</li>" if data.get('company') else ''}
                    <li><strong>IP Address:</strong> {contact_message.ip_address}</li>
                    <li><strong>User Agent:</strong> {contact_message.user_agent}</li>
                    <li><strong>Submitted:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</li>
                </ul>
                <p><strong>Message:</strong></p>
                <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid {'#10b981' if is_business_inquiry else '#28a745'}; margin: 15px 0;">
                    {data['message'].replace(chr(10), '<br>')}
                </div>
                <p style="margin-top: 20px;">
                    <a href="{admin_link}" style="background: #113CCF; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">
                        View in Admin Dashboard
                    </a>
                </p>
                """,
                user_email=data['email']
            )
            
            # Send to all admin emails
            for admin_email in admin_emails:
                try:
                    self.email_service.queue_email(
                        to=admin_email,
                        subject=f"{'🤝' if is_business_inquiry else '📧'} New Contact: {data['subject']} - CineBrain Admin",
                        html=html,
                        text=text,
                        priority=priority,
                        to_name='CineBrain Admin'
                    )
                    logger.info(f"✅ Admin notification email queued for {admin_email}")
                except Exception as e:
                    logger.error(f"Failed to queue email for admin {admin_email}: {e}")
            
            logger.info(f"✅ Admin notifications sent for contact from {data['email']}")
            
        except Exception as e:
            logger.error(f"❌ Error sending admin notification: {e}")

def init_contact_service(app, db, models, services):
    """Initialize contact service"""
    return ContactService(app, db, models, services)