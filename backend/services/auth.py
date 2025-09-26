from flask import Blueprint, request, jsonify
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate
import jwt
import os
import logging
from functools import wraps
import re
import threading
import uuid
import time
from typing import Dict, Optional
import hashlib
import json
import redis
from urllib.parse import urlparse
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)

FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')
BACKEND_URL = os.environ.get('BACKEND_URL', 'https://backend-app-970m.onrender.com')
REDIS_URL = os.environ.get('REDIS_URL', 'redis://red-d2qlbuje5dus73c71qog:xp7inVzgblGCbo9I4taSGLdKUg0xY91I@red-d2qlbuje5dus73c71qog:6379')

# Email service configuration
BREVO_API_KEY = os.environ.get('BREVO_API_KEY', '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')  # Add your Brevo API key here
SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY', '')  # Optional: SendGrid as backup
EMAIL_SERVICE = os.environ.get('EMAIL_SERVICE', 'brevo')  # 'brevo' or 'sendgrid'

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

app = None
db = None
User = None
serializer = None
redis_client = None

PASSWORD_RESET_SALT = 'password-reset-salt-cinebrain-2025'

def init_redis():
    global redis_client
    try:
        url = urlparse(REDIS_URL)
        redis_client = redis.StrictRedis(
            host=url.hostname,
            port=url.port,
            password=url.password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        redis_client.ping()
        logger.info("Redis connected successfully")
        return redis_client
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return None

class ProductionEmailService:
    def __init__(self):
        self.from_email = "noreply@cinebrain.com"
        self.from_name = "CineBrain"
        self.redis_client = redis_client
        self.brevo_api_key = BREVO_API_KEY
        self.sendgrid_api_key = SENDGRID_API_KEY
        self.email_service = EMAIL_SERVICE
        
        # Test email service on initialization
        self.email_enabled = self._test_email_service()
        
        if self.email_enabled:
            self.start_email_worker()
            logger.info(f"✅ Email service initialized with {self.email_service}")
        else:
            logger.warning("⚠️ Email service disabled - using fallback mode")
    
    def _test_email_service(self):
        """Test if email service is configured and working"""
        if self.email_service == 'brevo' and self.brevo_api_key:
            try:
                headers = {
                    'accept': 'application/json',
                    'api-key': self.brevo_api_key
                }
                response = requests.get(
                    'https://api.brevo.com/v3/account',
                    headers=headers,
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info("✅ Brevo API connected successfully")
                    return True
                else:
                    logger.error(f"Brevo API error: {response.status_code}")
            except Exception as e:
                logger.error(f"Brevo connection test failed: {e}")
        
        elif self.email_service == 'sendgrid' and self.sendgrid_api_key:
            try:
                headers = {
                    'Authorization': f'Bearer {self.sendgrid_api_key}',
                    'Content-Type': 'application/json'
                }
                response = requests.get(
                    'https://api.sendgrid.com/v3/user/profile',
                    headers=headers,
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info("✅ SendGrid API connected successfully")
                    return True
                else:
                    logger.error(f"SendGrid API error: {response.status_code}")
            except Exception as e:
                logger.error(f"SendGrid connection test failed: {e}")
        
        logger.warning("No email service configured or API key missing")
        return False
    
    def start_email_worker(self):
        """Start background worker for processing email queue"""
        def worker():
            while True:
                try:
                    if self.redis_client and self.email_enabled:
                        email_json = self.redis_client.lpop('email_queue')
                        if email_json:
                            email_data = json.loads(email_json)
                            self._send_email_api(email_data)
                        else:
                            time.sleep(1)
                    else:
                        time.sleep(5)
                except Exception as e:
                    logger.error(f"Email worker error: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=worker, daemon=True, name="EmailWorker")
        thread.start()
        logger.info("Started email worker thread")
    
    def _send_email_api(self, email_data: Dict):
        """Send email using API service (Brevo or SendGrid)"""
        max_retries = 3
        retry_count = email_data.get('retry_count', 0)
        
        try:
            if self.email_service == 'brevo':
                success = self._send_via_brevo(email_data)
            elif self.email_service == 'sendgrid':
                success = self._send_via_sendgrid(email_data)
            else:
                success = False
            
            if success:
                logger.info(f"✅ Email sent successfully to {email_data['to']} via {self.email_service}")
                
                if self.redis_client:
                    self.redis_client.setex(
                        f"email_sent:{email_data.get('id', 'unknown')}",
                        86400,
                        json.dumps({
                            'status': 'sent',
                            'timestamp': datetime.utcnow().isoformat(),
                            'to': email_data['to'],
                            'service': self.email_service
                        })
                    )
            else:
                raise Exception(f"Failed to send email via {self.email_service}")
                
        except Exception as e:
            logger.error(f"❌ Email API Error: {e}")
            
            if retry_count < max_retries:
                retry_count += 1
                email_data['retry_count'] = retry_count
                retry_delay = 5 * (2 ** retry_count)
                
                logger.info(f"🔄 Retrying email in {retry_delay} seconds (attempt {retry_count}/{max_retries})")
                
                if self.redis_client:
                    threading.Timer(
                        retry_delay,
                        lambda: self.redis_client.rpush('email_queue', json.dumps(email_data))
                    ).start()
            else:
                logger.error(f"❌ Failed after {max_retries} attempts - storing in fallback")
                self._store_fallback_email(email_data)
    
    def _send_via_brevo(self, email_data: Dict) -> bool:
        """Send email using Brevo API"""
        try:
            url = "https://api.brevo.com/v3/smtp/email"
            
            headers = {
                'accept': 'application/json',
                'api-key': self.brevo_api_key,
                'content-type': 'application/json'
            }
            
            payload = {
                "sender": {
                    "name": self.from_name,
                    "email": "noreply@cinebrain.com"
                },
                "to": [
                    {
                        "email": email_data['to'],
                        "name": email_data.get('user_name', 'User')
                    }
                ],
                "subject": email_data['subject'],
                "htmlContent": email_data['html'],
                "textContent": email_data.get('text', ''),
                "headers": {
                    "X-Mailin-custom": f"email_id:{email_data.get('id', 'unknown')}"
                }
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code in [200, 201, 202]:
                return True
            else:
                logger.error(f"Brevo API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Brevo send error: {e}")
            return False
    
    def _send_via_sendgrid(self, email_data: Dict) -> bool:
        """Send email using SendGrid API"""
        try:
            url = "https://api.sendgrid.com/v3/mail/send"
            
            headers = {
                'Authorization': f'Bearer {self.sendgrid_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                "personalizations": [
                    {
                        "to": [{"email": email_data['to']}],
                        "subject": email_data['subject']
                    }
                ],
                "from": {
                    "email": "noreply@cinebrain.com",
                    "name": self.from_name
                },
                "content": [
                    {
                        "type": "text/plain",
                        "value": email_data.get('text', '')
                    },
                    {
                        "type": "text/html",
                        "value": email_data['html']
                    }
                ]
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code in [200, 201, 202]:
                return True
            else:
                logger.error(f"SendGrid API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"SendGrid send error: {e}")
            return False
    
    def _store_fallback_email(self, email_data: Dict):
        """Store email for manual retrieval when all methods fail"""
        try:
            if self.redis_client:
                fallback_key = f"email_fallback:{email_data.get('id', uuid.uuid4())}"
                self.redis_client.setex(
                    fallback_key,
                    604800,  # 7 days
                    json.dumps({
                        'to': email_data['to'],
                        'subject': email_data['subject'],
                        'timestamp': datetime.utcnow().isoformat(),
                        'reset_token': email_data.get('reset_token'),
                        'reset_url': email_data.get('reset_url'),
                        'fallback_reason': 'All email services failed'
                    })
                )
                
                self.redis_client.rpush('email_fallback_queue', fallback_key)
                logger.info(f"📥 Email stored in fallback queue: {fallback_key}")
                
                # Log the reset URL for immediate access
                if email_data.get('reset_url'):
                    logger.info(f"🔗 Reset link for {email_data['to']}: {email_data['reset_url']}")
                    
        except Exception as e:
            logger.error(f"Failed to store fallback email: {e}")
    
    def queue_email(self, to: str, subject: str, html: str, text: str, 
                   priority: str = 'normal', reset_token: str = None, 
                   reset_url: str = None, user_name: str = None):
        """Queue email for sending"""
        email_id = str(uuid.uuid4())
        email_data = {
            'id': email_id,
            'to': to,
            'subject': subject,
            'html': html,
            'text': text,
            'priority': priority,
            'timestamp': datetime.utcnow().isoformat(),
            'retry_count': 0,
            'reset_token': reset_token,
            'reset_url': reset_url,
            'user_name': user_name
        }
        
        try:
            if not self.email_enabled:
                logger.warning(f"Email service disabled - storing in fallback for {to}")
                self._store_fallback_email(email_data)
                return True
            
            if self.redis_client:
                if priority == 'high':
                    self.redis_client.lpush('email_queue', json.dumps(email_data))
                else:
                    self.redis_client.rpush('email_queue', json.dumps(email_data))
                
                logger.info(f"📧 Email queued for {to} - ID: {email_id}")
            else:
                # Try to send directly if Redis is not available
                threading.Thread(
                    target=self._send_email_api,
                    args=(email_data,),
                    daemon=True
                ).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue email: {e}")
            self._store_fallback_email(email_data)
            return True
    
    def get_professional_template(self, content_type: str, **kwargs) -> tuple:
        """Generate professional email templates"""
        if content_type == 'password_reset':
            return self._get_password_reset_template(**kwargs)
        elif content_type == 'password_changed':
            return self._get_password_changed_template(**kwargs)
        else:
            return self._get_generic_template(**kwargs)
    
    def _get_password_reset_template(self, **kwargs) -> tuple:
        reset_url = kwargs.get('reset_url', '')
        user_name = kwargs.get('user_name', 'there')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #113CCF 0%, #1E4FE5 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .content {{
                    background: white;
                    padding: 30px;
                    border: 1px solid #e0e0e0;
                    border-radius: 0 0 10px 10px;
                }}
                .button {{
                    display: inline-block;
                    padding: 12px 30px;
                    background: linear-gradient(135deg, #113CCF 0%, #1E4FE5 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 25px;
                    margin: 20px 0;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    color: #888;
                    font-size: 12px;
                }}
                .warning {{
                    background: #fff3cd;
                    border: 1px solid #ffc107;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1 style="margin: 0; font-size: 28px;">CineBrain</h1>
                <p style="margin: 5px 0; opacity: 0.9;">The Mind Behind Your Next Favorite</p>
            </div>
            <div class="content">
                <h2>Reset Your Password</h2>
                <p>Hi {user_name},</p>
                <p>We received a request to reset your password. Click the button below to create a new password:</p>
                <center>
                    <a href="{reset_url}" class="button">Reset Password</a>
                </center>
                <div class="warning">
                    <strong>⏰ This link expires in 1 hour</strong><br>
                    For security reasons, this password reset link will expire soon.
                </div>
                <p style="font-size: 12px; color: #666;">
                    If the button doesn't work, copy and paste this link into your browser:<br>
                    <span style="color: #113CCF; word-break: break-all;">{reset_url}</span>
                </p>
            </div>
            <div class="footer">
                <p>If you didn't request this, you can safely ignore this email.</p>
                <p>© {datetime.now().year} CineBrain. All rights reserved.</p>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Reset Your Password - CineBrain

Hi {user_name},

We received a request to reset your password.

To reset your password, click this link:
{reset_url}

This link expires in 1 hour.

If you didn't request this, you can safely ignore this email.

Best regards,
The CineBrain Team

© {datetime.now().year} CineBrain. All rights reserved.
        """
        
        return html, text
    
    def _get_password_changed_template(self, **kwargs) -> tuple:
        user_name = kwargs.get('user_name', 'there')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 10px 10px 0 0;
                }}
                .content {{
                    background: white;
                    padding: 30px;
                    border: 1px solid #e0e0e0;
                    border-radius: 0 0 10px 10px;
                }}
                .success {{
                    background: #d4edda;
                    border: 1px solid #28a745;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1 style="margin: 0;">✅ Password Changed</h1>
            </div>
            <div class="content">
                <p>Hi {user_name},</p>
                <div class="success">
                    <strong>Your password has been successfully changed!</strong>
                </div>
                <p>You can now sign in with your new password.</p>
                <p>If you didn't make this change, please contact our support team immediately.</p>
            </div>
            <div style="text-align: center; margin-top: 30px; color: #888; font-size: 12px;">
                <p>© {datetime.now().year} CineBrain. All rights reserved.</p>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Password Changed Successfully - CineBrain

Hi {user_name},

Your password has been successfully changed!

You can now sign in with your new password.

If you didn't make this change, please contact our support team immediately.

© {datetime.now().year} CineBrain. All rights reserved.
        """
        
        return html, text
    
    def _get_generic_template(self, **kwargs) -> tuple:
        subject = kwargs.get('subject', 'CineBrain Notification')
        content = kwargs.get('content', '')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>{subject}</h2>
                <div>{content}</div>
                <hr>
                <p style="color: #888; font-size: 12px;">© {datetime.now().year} CineBrain</p>
            </div>
        </body>
        </html>
        """
        
        text = f"{subject}\n\n{content}\n\n© {datetime.now().year} CineBrain"
        
        return html, text

email_service = None

def init_auth(flask_app, database, user_model):
    global app, db, User, serializer, email_service, redis_client
    
    app = flask_app
    db = database
    User = user_model
    
    redis_client = init_redis()
    
    email_service = ProductionEmailService()
    
    serializer = URLSafeTimedSerializer(app.secret_key)
    
    if email_service.email_enabled:
        logger.info(f"✅ Auth module initialized with {email_service.email_service} email service")
    else:
        logger.warning("⚠️ Auth module initialized in fallback mode (no email service configured)")

def check_rate_limit(identifier: str, max_requests: int = 5, window: int = 300) -> bool:
    if not redis_client:
        return True
    
    try:
        key = f"rate_limit:{identifier}"
        
        pipe = redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        results = pipe.execute()
        
        current_count = results[0]
        
        if current_count > max_requests:
            logger.warning(f"Rate limit exceeded for {identifier}: {current_count}/{max_requests}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Rate limit check error: {e}")
        return True

def generate_reset_token(email):
    token = serializer.dumps(email, salt=PASSWORD_RESET_SALT)
    
    if redis_client:
        try:
            redis_client.setex(
                f"reset_token:{token[:20]}",
                3600,
                email
            )
        except Exception as e:
            logger.error(f"Failed to cache token in Redis: {e}")
    
    return token

def verify_reset_token(token, expiration=3600):
    if redis_client:
        try:
            cached_email = redis_client.get(f"reset_token:{token[:20]}")
            if cached_email:
                email = serializer.loads(token, salt=PASSWORD_RESET_SALT, max_age=expiration)
                if email == cached_email:
                    return email
        except Exception as e:
            logger.error(f"Redis token verification error: {e}")
    
    try:
        email = serializer.loads(token, salt=PASSWORD_RESET_SALT, max_age=expiration)
        return email
    except SignatureExpired:
        return None
    except BadTimeSignature:
        return None

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, "Valid password"

def get_request_info(request):
    ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
    if ip_address:
        ip_address = ip_address.split(',')[0].strip()
    
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    device = "Unknown device"
    if 'Mobile' in user_agent or 'Android' in user_agent:
        device = "Mobile device"
    elif 'iPad' in user_agent or 'Tablet' in user_agent:
        device = "Tablet"
    elif 'Windows' in user_agent:
        device = "Windows PC"
    elif 'Macintosh' in user_agent:
        device = "Mac"
    elif 'Linux' in user_agent:
        device = "Linux PC"
    
    return ip_address, "Unknown location", device

@auth_bp.route('/api/auth/forgot-password', methods=['POST', 'OPTIONS'])
def forgot_password():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        if not email or not EMAIL_REGEX.match(email):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        if not check_rate_limit(f"forgot_password:{email}", max_requests=3, window=600):
            return jsonify({
                'error': 'Too many password reset requests. Please try again in 10 minutes.'
            }), 429
        
        user = User.query.filter_by(email=email).first()
        
        response_data = {
            'success': True,
            'message': 'If an account exists with this email, you will receive password reset instructions shortly.'
        }
        
        if user:
            token = generate_reset_token(email)
            reset_url = f"{FRONTEND_URL}/auth/reset-password.html?token={token}"
            
            html_content, text_content = email_service.get_professional_template(
                'password_reset',
                reset_url=reset_url,
                user_name=user.username,
                user_email=email
            )
            
            email_service.queue_email(
                to=email,
                subject="Reset your password - CineBrain",
                html=html_content,
                text=text_content,
                priority='high',
                reset_token=token,
                reset_url=reset_url,
                user_name=user.username
            )
            
            logger.info(f"Password reset requested for {email}")
            
            # Only include reset_url in development or when email service is disabled
            if not email_service.email_enabled or app.debug:
                response_data['fallback_mode'] = True
                response_data['reset_url'] = reset_url
                logger.info(f"🔗 Development/Fallback - Reset URL: {reset_url}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({'error': 'Failed to process password reset request'}), 500

@auth_bp.route('/api/auth/reset-password', methods=['POST', 'OPTIONS'])
def reset_password():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        token = data.get('token', '').strip()
        new_password = data.get('password', '')
        confirm_password = data.get('confirmPassword', '')
        
        if not token:
            return jsonify({'error': 'Reset token is required'}), 400
        
        if not new_password or not confirm_password:
            return jsonify({'error': 'Password and confirmation are required'}), 400
        
        if new_password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400
        
        is_valid, message = validate_password(new_password)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        email = verify_reset_token(token)
        if not email:
            return jsonify({'error': 'Invalid or expired reset token'}), 400
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        user.password_hash = generate_password_hash(new_password)
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        if redis_client:
            try:
                redis_client.delete(f"reset_token:{token[:20]}")
            except:
                pass
        
        ip_address, location, device = get_request_info(request)
        
        html_content, text_content = email_service.get_professional_template(
            'password_changed',
            user_name=user.username,
            user_email=email,
            ip_address=ip_address,
            location=location,
            device=device
        )
        
        email_service.queue_email(
            to=email,
            subject="Your password was changed - CineBrain",
            html=html_content,
            text=text_content,
            priority='high',
            user_name=user.username
        )
        
        auth_token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30),
            'iat': datetime.utcnow()
        }, app.secret_key, algorithm='HS256')
        
        logger.info(f"Password reset successful for {email}")
        
        return jsonify({
            'success': True,
            'message': 'Password has been reset successfully',
            'token': auth_token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        return jsonify({'error': 'Failed to reset password'}), 500

@auth_bp.route('/api/auth/verify-reset-token', methods=['POST', 'OPTIONS'])
def verify_token():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        token = data.get('token', '').strip()
        
        if not token:
            return jsonify({'valid': False, 'error': 'No token provided'}), 400
        
        email = verify_reset_token(token)
        if email:
            user = User.query.filter_by(email=email).first()
            if user:
                return jsonify({
                    'valid': True,
                    'email': email,
                    'masked_email': email[:3] + '***' + email[email.index('@'):]
                }), 200
            else:
                return jsonify({'valid': False, 'error': 'User not found'}), 400
        else:
            return jsonify({'valid': False, 'error': 'Invalid or expired token'}), 400
            
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return jsonify({'valid': False, 'error': 'Failed to verify token'}), 500

@auth_bp.route('/api/auth/health', methods=['GET'])
def auth_health():
    try:
        health_info = {
            'status': 'healthy',
            'service': 'authentication',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check database
        try:
            if User:
                User.query.limit(1).first()
            health_info['database'] = 'connected'
        except:
            health_info['database'] = 'disconnected'
            health_info['status'] = 'degraded'
        
        # Check Redis
        redis_status = 'not_configured'
        if redis_client:
            try:
                redis_client.ping()
                redis_status = 'connected'
            except:
                redis_status = 'disconnected'
        health_info['redis_status'] = redis_status
        
        # Check email service
        if email_service:
            health_info['email_service'] = {
                'enabled': email_service.email_enabled,
                'provider': email_service.email_service if email_service.email_enabled else 'none',
                'status': 'operational' if email_service.email_enabled else 'fallback_mode'
            }
            
            if redis_client:
                try:
                    queue_size = redis_client.llen('email_queue')
                    fallback_size = redis_client.llen('email_fallback_queue')
                    health_info['email_queue'] = {
                        'pending': queue_size,
                        'fallback': fallback_size
                    }
                except:
                    pass
        
        return jsonify(health_info), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'authentication',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@auth_bp.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    allowed_origins = [
        FRONTEND_URL,
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

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return '', 200
            
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
            
            current_user.last_active = datetime.utcnow()
            db.session.commit()
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

class EnhancedUserAnalytics:
    @staticmethod
    def get_comprehensive_user_stats(user_id):
        try:
            from app import UserInteraction, Content
            
            interactions = UserInteraction.query.filter_by(user_id=user_id).all() if UserInteraction else []
            
            stats = {
                'total_interactions': len(interactions),
                'content_watched': len([i for i in interactions if i.interaction_type == 'view']),
                'favorites': len([i for i in interactions if i.interaction_type == 'favorite']),
                'watchlist_items': len([i for i in interactions if i.interaction_type == 'watchlist']),
                'ratings_given': len([i for i in interactions if i.interaction_type == 'rating']),
                'likes_given': len([i for i in interactions if i.interaction_type == 'like']),
                'searches_made': len([i for i in interactions if i.interaction_type == 'search'])
            }
            
            ratings = [i.rating for i in interactions if i.rating is not None]
            stats['average_rating'] = round(sum(ratings) / len(ratings), 1) if ratings else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating enhanced stats: {e}")
            return {}

__all__ = [
    'auth_bp',
    'init_auth',
    'require_auth',
    'generate_reset_token',
    'verify_reset_token',
    'validate_password',
    'EnhancedUserAnalytics'
]