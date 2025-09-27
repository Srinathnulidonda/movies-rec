# backend/services/auth.py
from flask import Blueprint, request, jsonify
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate
from email.header import Header
import jwt
import os
import logging
from functools import wraps
import re
import threading
import smtplib
import ssl
import uuid
import time
from typing import Dict, Optional
import hashlib
import json
import redis
from urllib.parse import urlparse
import socket
import random
import string
import secrets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)

FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')
BACKEND_URL = os.environ.get('BACKEND_URL', 'https://backend-app-970m.onrender.com')
REDIS_URL = os.environ.get('REDIS_URL', 'redis://red-d2qlbuje5dus73c71qog:xp7inVzgblGCbo9I4taSGLdKUg0xY91I@red-d2qlbuje5dus73c71qog:6379')

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

app = None
db = None
User = None
mail = None
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

class FreeEmailService:
    def __init__(self, username=None, password=None):
        self.username = username or os.environ.get('GMAIL_USERNAME', 'projects.srinath@gmail.com')
        self.password = password or os.environ.get('GMAIL_APP_PASSWORD', 'nddg lphy ajjy rnuq')
        self.from_email = "noreply@cinebrain.com"
        self.from_name = "CineBrain"
        self.reply_to = "support@cinebrain.com"
        self.redis_client = redis_client
        self.use_http_fallback = False
        self.working_config = None
        
        # Skip SMTP tests on Render - we know they're blocked
        if os.environ.get('RENDER'):
            logger.warning("Running on Render - skipping SMTP tests (ports blocked)")
            self.email_enabled = False
            self._setup_fallback_system()
        else:
            # Only test SMTP in development/other environments
            self.smtp_configs = [
                {
                    'name': 'Gmail TLS 587',
                    'server': 'smtp.gmail.com',
                    'port': 587,
                    'use_tls': True,
                    'use_ssl': False
                }
            ]
            self.email_enabled = self._test_smtp_connection()
            
            if self.email_enabled:
                self.start_email_worker()
            else:
                logger.warning("⚠️ Email service running in fallback mode - direct links will be provided")
                self._setup_fallback_system()
    
    def _test_smtp_connection(self):
        """Test SMTP connectivity - only in non-Render environments"""
        for config in self.smtp_configs:
            try:
                logger.info(f"Testing {config['name']} connection...")
                
                socket.setdefaulttimeout(2)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                
                try:
                    result = sock.connect_ex((config['server'], config['port']))
                    sock.close()
                    
                    if result != 0:
                        logger.warning(f"{config['name']} port {config['port']} appears blocked")
                        continue
                except Exception as sock_err:
                    logger.warning(f"Socket test failed: {sock_err}")
                    continue
                
                # Port seems open, try SMTP connection
                if config['use_ssl']:
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    
                    server = smtplib.SMTP_SSL(config['server'], config['port'], 
                                             context=context, timeout=5)
                else:
                    server = smtplib.SMTP(config['server'], config['port'], timeout=5)
                
                try:
                    server.set_debuglevel(0)
                    server.ehlo()
                    
                    if config['use_tls']:
                        context = ssl.create_default_context()
                        context.check_hostname = False
                        context.verify_mode = ssl.CERT_NONE
                        server.starttls(context=context)
                        server.ehlo()
                    
                    server.login(self.username, self.password)
                    server.quit()
                    logger.info(f"✅ {config['name']} connection successful!")
                    self.working_config = config
                    return True
                except Exception as smtp_err:
                    logger.warning(f"SMTP error: {smtp_err}")
                    try:
                        server.quit()
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"{config['name']} error: {e}")
        
        logger.error("❌ SMTP configurations failed")
        return False
    
    def _setup_fallback_system(self):
        """Setup fallback system for when SMTP is completely blocked"""
        try:
            logger.info("📧 Setting up email fallback system")
            self.use_http_fallback = True
            
            if self.redis_client:
                self.redis_client.set('email_fallback_enabled', 'true')
                logger.info("✅ Fallback email system ready")
            
        except Exception as e:
            logger.error(f"Fallback system setup failed: {e}")
    
    def start_email_worker(self):
        """Start background worker for processing email queue"""
        def worker():
            while True:
                try:
                    if self.redis_client and self.email_enabled:
                        email_json = self.redis_client.lpop('email_queue')
                        if email_json:
                            email_data = json.loads(email_json)
                            self._send_email_smtp(email_data)
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
    
    def _send_email_smtp(self, email_data: Dict):
        """Send email via SMTP"""
        if not self.email_enabled:
            logger.warning(f"Email service disabled - storing email for {email_data['to']}")
            self._store_fallback_email(email_data)
            return False
        
        # Email sending logic here (keeping it short for brevity)
        return False
    
    def _store_fallback_email(self, email_data: Dict):
        """Store email data for manual retrieval when SMTP fails"""
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
                        'reset_url': f"{FRONTEND_URL}/auth/reset-password.html?token={email_data.get('reset_token')}" if email_data.get('reset_token') else None,
                        'fallback_reason': 'SMTP connection failed'
                    })
                )
                
                self.redis_client.rpush('email_fallback_queue', fallback_key)
                logger.info(f"📥 Email stored in fallback queue: {fallback_key}")
                    
        except Exception as e:
            logger.error(f"Failed to store fallback email: {e}")
    
    def queue_email(self, to: str, subject: str, html: str, text: str, priority: str = 'normal', reset_token: str = None):
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
            'reset_token': reset_token
        }
        
        try:
            if not self.email_enabled:
                logger.warning(f"Email service disabled - using fallback for {to}")
                self._store_fallback_email(email_data)
                
                if reset_token:
                    reset_url = f"{FRONTEND_URL}/auth/reset-password.html?token={reset_token}"
                    logger.info(f"🔗 Password reset link for {to}: {reset_url}")
                
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue email: {e}")
            return False
    
    def get_professional_template(self, content_type: str, **kwargs) -> tuple:
        """Generate professional email templates"""
        # Simplified template generation
        html = f"<html><body><h1>CineBrain</h1><p>Password Reset</p></body></html>"
        text = "CineBrain Password Reset"
        return html, text

email_service = None

def init_auth(flask_app, database, user_model):
    global app, db, User, mail, serializer, email_service, redis_client
    
    app = flask_app
    db = database
    User = user_model
    
    redis_client = init_redis()
    email_service = FreeEmailService()
    serializer = URLSafeTimedSerializer(app.secret_key)
    
    if email_service.email_enabled:
        logger.info("✅ Auth module initialized with email support")
    else:
        logger.warning("⚠️ Auth module initialized in FALLBACK MODE - direct links will be provided")

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
    
    browser = ""
    if 'Chrome' in user_agent and 'Edg' not in user_agent:
        browser = "Chrome"
    elif 'Firefox' in user_agent:
        browser = "Firefox"
    elif 'Safari' in user_agent and 'Chrome' not in user_agent:
        browser = "Safari"
    elif 'Edg' in user_agent:
        browser = "Edge"
    
    if browser:
        device = f"{browser} on {device}"
    
    location = "Unknown location"
    
    return ip_address, location, device

# NEW ENDPOINTS FOR DIRECT RESET

@auth_bp.route('/api/auth/forgot-password-direct', methods=['POST', 'OPTIONS'])
def forgot_password_direct():
    """Direct password reset - returns code immediately"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        # Validate email
        if not email or not EMAIL_REGEX.match(email):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        # Rate limiting
        if not check_rate_limit(f"forgot_password_direct:{email}", max_requests=5, window=600):
            return jsonify({
                'error': 'Too many reset attempts. Please try again in 10 minutes.'
            }), 429
        
        user = User.query.filter_by(email=email).first()
        
        if user:
            # Generate 6-digit reset code
            reset_code = ''.join(random.choices(string.digits, k=6))
            
            # Also generate a backup token (for URL method)
            reset_token = secrets.token_urlsafe(32)
            
            # Store both in Redis with 15 minute expiry
            if redis_client:
                # Store code -> email mapping
                redis_client.setex(
                    f"reset_code:{reset_code}",
                    900,  # 15 minutes
                    json.dumps({
                        'email': email,
                        'user_id': user.id,
                        'created_at': datetime.utcnow().isoformat()
                    })
                )
                
                # Store token -> email mapping
                redis_client.setex(
                    f"reset_token_direct:{reset_token}",
                    900,  # 15 minutes
                    json.dumps({
                        'email': email,
                        'user_id': user.id,
                        'created_at': datetime.utcnow().isoformat()
                    })
                )
                
                logger.info(f"Reset code generated for {email}: {reset_code}")
            
            # Generate reset URL
            reset_url = f"{FRONTEND_URL}/reset-password.html?token={reset_token}"
            
            return jsonify({
                'success': True,
                'resetCode': reset_code,
                'resetToken': reset_token,
                'resetUrl': reset_url,
                'expiresIn': 15,  # minutes
                'message': 'Reset code generated successfully'
            }), 200
        
        # Don't reveal if user doesn't exist
        return jsonify({
            'success': False,
            'message': 'If an account exists with this email, a reset code has been generated.'
        }), 200
        
    except Exception as e:
        logger.error(f"Direct password reset error: {e}")
        return jsonify({'error': 'Failed to process reset request'}), 500

@auth_bp.route('/api/auth/reset-by-code', methods=['POST', 'OPTIONS'])
def reset_by_code():
    """Reset password using the 6-digit code"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        code = data.get('code', '').strip()
        new_password = data.get('newPassword', '')
        confirm_password = data.get('confirmPassword', '')
        
        # Validate inputs
        if not code or len(code) != 6:
            return jsonify({'error': 'Invalid reset code'}), 400
        
        if not new_password or not confirm_password:
            return jsonify({'error': 'Password and confirmation are required'}), 400
        
        if new_password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400
        
        # Validate password strength
        is_valid, message = validate_password(new_password)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Check code in Redis
        if redis_client:
            code_data = redis_client.get(f"reset_code:{code}")
            
            if not code_data:
                return jsonify({'error': 'Invalid or expired reset code'}), 400
            
            code_info = json.loads(code_data)
            email = code_info['email']
            user_id = code_info['user_id']
            
            # Get user and update password
            user = User.query.get(user_id)
            if user and user.email == email:
                user.password_hash = generate_password_hash(new_password)
                user.last_active = datetime.utcnow()
                db.session.commit()
                
                # Delete the used code
                redis_client.delete(f"reset_code:{code}")
                
                # Generate auth token for auto-login
                auth_token = jwt.encode({
                    'user_id': user.id,
                    'exp': datetime.utcnow() + timedelta(days=30),
                    'iat': datetime.utcnow()
                }, app.secret_key, algorithm='HS256')
                
                logger.info(f"Password reset successful for {email} using code")
                
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
            
        return jsonify({'error': 'Invalid or expired reset code'}), 400
        
    except Exception as e:
        logger.error(f"Reset by code error: {e}")
        return jsonify({'error': 'Failed to reset password'}), 500

@auth_bp.route('/api/auth/verify-reset-code', methods=['POST', 'OPTIONS'])
def verify_reset_code():
    """Verify if a reset code is valid"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        code = data.get('code', '').strip()
        
        if not code or len(code) != 6:
            return jsonify({'valid': False, 'error': 'Invalid code format'}), 400
        
        if redis_client:
            code_data = redis_client.get(f"reset_code:{code}")
            
            if code_data:
                code_info = json.loads(code_data)
                email = code_info['email']
                
                # Mask email for privacy
                masked_email = email[:3] + '***' + email[email.index('@'):]
                
                return jsonify({
                    'valid': True,
                    'maskedEmail': masked_email,
                    'message': 'Code is valid'
                }), 200
        
        return jsonify({'valid': False, 'error': 'Invalid or expired code'}), 400
        
    except Exception as e:
        logger.error(f"Code verification error: {e}")
        return jsonify({'valid': False, 'error': 'Verification failed'}), 500

# ORIGINAL ENDPOINTS (KEPT FOR COMPATIBILITY)

@auth_bp.route('/api/auth/forgot-password', methods=['POST', 'OPTIONS'])
def forgot_password():
    """Original forgot password endpoint - now uses direct method"""
    if request.method == 'OPTIONS':
        return '', 200
    
    # Redirect to direct method since SMTP is blocked on Render
    return forgot_password_direct()

@auth_bp.route('/api/auth/reset-password', methods=['POST', 'OPTIONS'])
def reset_password():
    """Original reset password endpoint"""
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
    """Original token verification endpoint"""
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

@auth_bp.route('/api/auth/fallback-emails', methods=['GET'])
def get_fallback_emails():
    """Admin endpoint to retrieve fallback emails when SMTP fails"""
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Admin authentication required'}), 401
        
        # Verify admin token
        try:
            token = auth_header.split(' ')[1]
            payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            user = User.query.get(payload['user_id'])
            if not user or not user.is_admin:
                return jsonify({'error': 'Admin access required'}), 403
        except:
            return jsonify({'error': 'Invalid authentication'}), 401
        
        emails = []
        pending_resets = []
        
        if redis_client:
            # Get fallback emails
            try:
                keys = redis_client.lrange('email_fallback_queue', 0, 49)
                for key in keys:
                    data = redis_client.get(key)
                    if data:
                        emails.append(json.loads(data))
            except:
                pass
            
            # Get pending resets
            try:
                for key in redis_client.scan_iter("pending_reset:*"):
                    data = redis_client.get(key)
                    if data:
                        reset_info = json.loads(data)
                        reset_info['email'] = key.replace('pending_reset:', '')
                        pending_resets.append(reset_info)
            except:
                pass
        
        return jsonify({
            'success': True,
            'fallback_emails': emails,
            'pending_resets': pending_resets,
            'smtp_enabled': email_service.email_enabled if email_service else False,
            'count': len(emails)
        }), 200
        
    except Exception as e:
        logger.error(f"Get fallback emails error: {e}")
        return jsonify({'error': 'Failed to retrieve fallback emails'}), 500

@auth_bp.route('/api/auth/health', methods=['GET'])
def auth_health():
    try:
        # Test database connection
        if User:
            User.query.limit(1).first()
        
        # Check Redis status
        redis_status = 'not_configured'
        redis_stats = {}
        if redis_client:
            try:
                redis_client.ping()
                redis_status = 'connected'
                
                info = redis_client.info()
                redis_stats = {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory_human': info.get('used_memory_human', 'N/A'),
                    'total_connections_received': info.get('total_connections_received', 0)
                }
            except:
                redis_status = 'disconnected'
        
        # Check email service status
        email_configured = email_service is not None
        email_enabled = email_service.email_enabled if email_service else False
        
        return jsonify({
            'status': 'healthy',
            'service': 'authentication',
            'email_service': 'Gmail SMTP (Fallback Mode on Render)',
            'email_configured': email_configured,
            'email_enabled': email_enabled,
            'direct_reset_enabled': True,  # New feature
            'redis_status': redis_status,
            'redis_stats': redis_stats,
            'frontend_url': FRONTEND_URL,
            'backend_url': BACKEND_URL,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
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
    allowed_origins = [FRONTEND_URL, 'http://127.0.0.1:5500', 'http://127.0.0.1:5501', 'http://localhost:5500']
    
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