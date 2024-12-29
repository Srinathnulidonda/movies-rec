# auth/service.py

from flask import request, jsonify
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from datetime import datetime, timedelta
import jwt
import os
import logging
from functools import wraps
import re
import threading
import uuid
import time
from typing import Dict, Optional
import json
import redis
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket

# Load environment variables
load_dotenv()

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
app = None
db = None
User = None
serializer = None
redis_client = None

# Configuration
FRONTEND_URL = os.getenv('FRONTEND_URL')
BACKEND_URL = os.getenv('BACKEND_URL')
REDIS_URL = os.getenv('REDIS_URL')
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# Brevo Configuration
BREVO_API_KEY = os.getenv('BREVO_API_KEY')
BREVO_SMTP_SERVER = os.getenv('BREVO_SMTP_SERVER', 'smtp-relay.brevo.com')
BREVO_SMTP_PORT = int(os.getenv('BREVO_SMTP_PORT', '587'))
BREVO_SMTP_USERNAME = os.getenv('BREVO_SMTP_USERNAME')
BREVO_SMTP_PASSWORD = os.getenv('BREVO_SMTP_PASSWORD')
BREVO_SENDER_EMAIL = os.getenv('BREVO_SENDER_EMAIL', 'projects.srinath@gmail.com')
BREVO_SENDER_NAME = os.getenv('BREVO_SENDER_NAME', 'CineBrain')
REPLY_TO_EMAIL = os.getenv('REPLY_TO_EMAIL', BREVO_SENDER_EMAIL)

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PASSWORD_RESET_SALT = 'password-reset-salt-cinebrain-2025'


def init_redis():
    """Initialize Redis connection"""
    global redis_client
    try:
        if not REDIS_URL:
            logger.warning("Redis URL not configured for auth service")
            return None

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
        logger.info("✅ Auth Redis connected successfully")
        return redis_client
    except Exception as e:
        logger.error(f"❌ Auth Redis connection failed: {e}")
        return None


class BrevoEmailService:
    """Email service using Brevo SMTP and API"""

    def __init__(self, api_key=None):
        self.api_key = api_key or BREVO_API_KEY
        
        # SMTP Configuration
        self.smtp_server = BREVO_SMTP_SERVER
        self.smtp_port = BREVO_SMTP_PORT
        self.smtp_username = BREVO_SMTP_USERNAME
        self.smtp_password = BREVO_SMTP_PASSWORD
        
        # Email Configuration
        self.sender_email = BREVO_SENDER_EMAIL
        self.sender_name = BREVO_SENDER_NAME
        self.reply_to_email = REPLY_TO_EMAIL
        
        # API Configuration
        self.base_url = "https://api.brevo.com/v3"
        
        # Service Configuration
        self.redis_client = redis_client
        self.email_enabled = False
        self.smtp_enabled = False
        self.api_enabled = False
        self.is_configured = False
        
        # Initialize email service based on environment
        self._initialize_email_service()
        
        if self.email_enabled:
            self.start_email_worker()
            logger.info("✅ Brevo email worker initialized successfully")
        else:
            logger.warning("⚠️ Email service disabled - no valid configuration found")

    def _initialize_email_service(self):
        """Initialize email service - API first in production, SMTP first in development"""
        
        # In production, prefer API over SMTP (many hosts block SMTP)
        if ENVIRONMENT == 'production':
            # Try API first in production
            if self.api_key and self.sender_email:
                self.api_enabled = self._test_api_connection()
                if self.api_enabled:
                    logger.info("✅ Using Brevo API for email delivery (Production)")
                    self.email_enabled = True
                    self.is_configured = True
                    return
            
            # Try SMTP as fallback (might be blocked)
            if self.smtp_username and self.smtp_password:
                self.smtp_enabled = self._test_smtp_connection_safe()
                if self.smtp_enabled:
                    logger.info("✅ Using Brevo SMTP for email delivery (Production)")
                    self.email_enabled = True
                    self.is_configured = True
                    return
        else:
            # Try API first since SMTP seems blocked
            if self.api_key and self.sender_email:
                self.api_enabled = self._test_api_connection()
                if self.api_enabled:
                    logger.info("✅ Using Brevo API for email delivery (Development)")
                    self.email_enabled = True
                    self.is_configured = True
                    return
            
            # Try SMTP as fallback
            if self.smtp_username and self.smtp_password:
                self.smtp_enabled = self._test_smtp_connection_safe()
                if self.smtp_enabled:
                    logger.info("✅ Using Brevo SMTP for email delivery (Development)")
                    self.email_enabled = True
                    self.is_configured = True
                    return
        
        # No valid configuration
        if not self.smtp_username and not self.api_key:
            logger.warning("⚠️ Neither SMTP nor API credentials configured")
        elif ENVIRONMENT == 'production' and not self.api_key:
            logger.warning("⚠️ In production environment - Brevo API key recommended (SMTP might be blocked)")
        
    def _test_smtp_connection_safe(self):
        """Test SMTP connectivity with timeout protection"""
        try:
            logger.info(f"Testing SMTP connection to {self.smtp_server}:{self.smtp_port}")
            
            # Set a short timeout for production environments
            timeout = 5 if ENVIRONMENT == 'production' else 10
            
            # Create socket with timeout
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            # Test if port is reachable
            result = sock.connect_ex((self.smtp_server, self.smtp_port))
            sock.close()
            
            if result != 0:
                logger.warning(f"⚠️ SMTP port {self.smtp_port} appears to be blocked or unreachable")
                return False
            
            # Now try actual SMTP connection
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=timeout)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.quit()
            
            logger.info("✅ Brevo SMTP connection successful")
            return True
            
        except socket.timeout:
            logger.warning(f"⚠️ SMTP connection timed out - port {self.smtp_port} might be blocked")
            return False
        except socket.error as e:
            logger.warning(f"⚠️ SMTP socket error - {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Brevo SMTP connection failed: {e}")
            return False

    def _test_api_connection(self):
        """Test Brevo API connectivity"""
        if not self.api_key:
            logger.warning("⚠️ No Brevo API key configured")
            return False
            
        try:
            headers = {
                'api-key': self.api_key,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Test with a simple endpoint
            response = requests.get(f"{self.base_url}/account", headers=headers, timeout=10)
            
            if response.status_code == 200:
                try:
                    account_data = response.json()
                    
                    # Handle both dict and list responses
                    if isinstance(account_data, dict):
                        account_email = account_data.get('email', 'Unknown')
                        plan_info = account_data.get('plan', {})
                        plan = plan_info.get('type', 'Unknown') if isinstance(plan_info, dict) else 'Unknown'
                    else:
                        # If response is not what we expect, but status is 200, API is working
                        account_email = 'Connected'
                        plan = 'Active'
                    
                    logger.info(f"✅ Brevo API connected successfully")
                    logger.info(f"   Account: {account_email}")
                    logger.info(f"   Status: Active")
                    return True
                except Exception as e:
                    # Even if parsing fails, if we got 200, API is working
                    logger.info(f"✅ Brevo API connected (response parsing issue: {e})")
                    return True
            else:
                logger.warning(f"⚠️ Brevo API test failed with status: {response.status_code}")
                if response.status_code == 401:
                    logger.error("❌ Invalid API key - please check your BREVO_API_KEY")
                return False
                
        except requests.exceptions.Timeout:
            logger.error(f"❌ Brevo API connection timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Brevo API connection test failed: {str(e)}")
            return False

    def start_email_worker(self):
        """Background worker to process queued emails"""

        def worker():
            while True:
                try:
                    if self.redis_client and self.email_enabled:
                        email_json = self.redis_client.lpop('email_queue')
                        if email_json:
                            email_data = json.loads(email_json)
                            self._send_email(email_data)
                        else:
                            time.sleep(1)
                    else:
                        time.sleep(5)
                except Exception as e:
                    logger.error(f"Email worker error: {e}")
                    time.sleep(5)

        threading.Thread(target=worker, daemon=True, name="BrevoEmailWorker").start()

    def _send_email(self, email_data: Dict, retry_count: int = 0):
        """Send email using available method"""
        if not self.email_enabled:
            self._store_fallback_email(email_data)
            return
        
        # Use whichever method is enabled
        if self.api_enabled:
            success = self._send_email_api(email_data, retry_count)
            if success:
                return
        
        if self.smtp_enabled:
            success = self._send_email_smtp(email_data, retry_count)
            if success:
                return
        
        # Store as fallback if both fail
        self._store_fallback_email(email_data)

    def _send_email_smtp(self, email_data: Dict, retry_count: int = 0) -> bool:
        """Send email using SMTP"""
        try:
            # In production, use shorter timeout
            timeout = 10 if ENVIRONMENT == 'production' else 30
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = email_data['subject']
            message["From"] = f"{self.sender_name} <{self.sender_email}>"
            message["To"] = email_data['to']
            
            if self.reply_to_email:
                message["Reply-To"] = self.reply_to_email
            
            # Add text and HTML parts
            if email_data.get('text'):
                part1 = MIMEText(email_data['text'], "plain")
                message.attach(part1)
            
            if email_data.get('html'):
                part2 = MIMEText(email_data['html'], "html")
                message.attach(part2)
            
            # Send email with timeout
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=timeout)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(message)
            server.quit()
            
            logger.info(f"✅ Email sent successfully via SMTP to {email_data['to']}")
            
            # Store success status
            if self.redis_client:
                self.redis_client.setex(f"email_sent:{email_data['id']}", 86400, json.dumps({
                    'status': 'sent',
                    'timestamp': datetime.utcnow().isoformat(),
                    'to': email_data['to'],
                    'service': 'brevo_smtp',
                    'method': 'smtp'
                }))
            
            return True
            
        except (socket.timeout, socket.error) as e:
            logger.warning(f"⚠️ SMTP network error: {e}")
            # Don't retry network errors in production
            if ENVIRONMENT == 'production':
                return False
            
            if retry_count < 1:
                time.sleep(2)
                return self._send_email_smtp(email_data, retry_count + 1)
            return False
            
        except Exception as e:
            error_msg = f"SMTP send failed: {e}"
            
            # Limited retry in production
            max_retries = 1 if ENVIRONMENT == 'production' else 2
            if retry_count < max_retries:
                logger.warning(f"⚠️ SMTP send failed, retrying... (attempt {retry_count + 1}/{max_retries})")
                time.sleep(2 ** retry_count)
                return self._send_email_smtp(email_data, retry_count + 1)
            else:
                logger.error(f"❌ SMTP send failed after retries to {email_data['to']}: {error_msg}")
                return False

    def _send_email_api(self, email_data: Dict, retry_count: int = 0) -> bool:
        """Send email using Brevo API"""
        if not self.api_key:
            return False
            
        headers = {
            'api-key': self.api_key,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        payload = {
            'sender': {
                'email': self.sender_email,
                'name': self.sender_name
            },
            'to': [
                {
                    'email': email_data['to'],
                    'name': email_data.get('to_name', '')
                }
            ],
            'subject': email_data['subject'],
            'htmlContent': email_data.get('html', '<p>No HTML content</p>'),
            'textContent': email_data.get('text', 'No text content'),
        }
        
        if self.reply_to_email:
            payload['replyTo'] = {
                'email': self.reply_to_email,
                'name': self.sender_name
            }

        try:
            response = requests.post(
                f"{self.base_url}/smtp/email", 
                headers=headers, 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 201:
                try:
                    res = response.json()
                    message_id = res.get('messageId', 'unknown') if isinstance(res, dict) else 'sent'
                except:
                    message_id = 'sent'
                    
                logger.info(f"✅ Email sent successfully via API to {email_data['to']} (ID: {message_id})")
                
                # Store success status
                if self.redis_client:
                    self.redis_client.setex(f"email_sent:{email_data['id']}", 86400, json.dumps({
                        'status': 'sent',
                        'timestamp': datetime.utcnow().isoformat(),
                        'to': email_data['to'],
                        'brevo_message_id': message_id,
                        'service': 'brevo_api',
                        'method': 'api'
                    }))
                
                return True
            else:
                error_msg = f"API returned {response.status_code}: {response.text}"
                
                # Retry logic
                if retry_count < 2:
                    logger.warning(f"⚠️ API send failed, retrying... (attempt {retry_count + 1}/2)")
                    time.sleep(2 ** retry_count)
                    return self._send_email_api(email_data, retry_count + 1)
                else:
                    logger.error(f"❌ API send failed after retries to {email_data['to']}: {error_msg}")
                    return False
                    
        except Exception as e:
            error_msg = f"API request failed: {e}"
            
            # Retry logic
            if retry_count < 2:
                logger.warning(f"⚠️ API send failed, retrying... (attempt {retry_count + 1}/2)")
                time.sleep(2 ** retry_count)
                return self._send_email_api(email_data, retry_count + 1)
            else:
                logger.error(f"❌ API send failed after retries to {email_data['to']}: {error_msg}")
                return False

    def _store_fallback_email(self, email_data: Dict):
        """Store unsent email in fallback queue"""
        if not self.redis_client:
            logger.warning(f"📧 Fallback email for {email_data['to']}: {email_data.get('reset_token', 'No token')}")
            if email_data.get('reset_token'):
                reset_url = f"{FRONTEND_URL}/auth/reset-password.html?token={email_data['reset_token']}"
                logger.info(f"🔗 Password reset link: {reset_url}")
            return
            
        email_data['failed_at'] = datetime.utcnow().isoformat()
        email_data['service'] = 'brevo'
        email_data['environment'] = ENVIRONMENT
        
        key = f"email_fallback:{uuid.uuid4()}"
        self.redis_client.setex(key, 604800, json.dumps(email_data))  # 7 days
        self.redis_client.rpush('email_fallback_queue', key)
        logger.info(f"📥 Stored unsent email for {email_data['to']} in fallback queue")

    def queue_email(self, to: str, subject: str, html: str, text: str = "", priority: str = 'normal', reset_token: str = None, to_name: str = ""):
        """Queue email to be sent asynchronously"""
        email_id = str(uuid.uuid4())
        email_data = {
            'id': email_id,
            'to': to,
            'to_name': to_name,
            'subject': subject,
            'html': html,
            'text': text,
            'timestamp': datetime.utcnow().isoformat(),
            'reset_token': reset_token,
            'service': 'brevo',
            'environment': ENVIRONMENT
        }

        if not self.email_enabled:
            logger.warning(f"Email service disabled - providing fallback for {to}")
            self._store_fallback_email(email_data)
            
            if reset_token:
                reset_url = f"{FRONTEND_URL}/auth/reset-password.html?token={reset_token}"
                logger.info(f"🔗 Password reset link for {to}: {reset_url}")
            
            return True

        if self.redis_client:
            if priority == 'high':
                self.redis_client.lpush('email_queue', json.dumps(email_data))
            else:
                self.redis_client.rpush('email_queue', json.dumps(email_data))
            logger.info(f"📧 Queued email for {to} (ID: {email_id})")
        else:
            # If no Redis, send directly in background thread
            threading.Thread(target=self._send_email, args=(email_data,), daemon=True).start()
        
        return True

    def get_email_status(self, email_id: str) -> Dict:
        """Get email delivery status"""
        if not self.redis_client:
            return {'status': 'unknown', 'id': email_id, 'service': 'brevo'}
        
        try:
            # Check sent emails
            sent_key = f"email_sent:{email_id}"
            sent_data = self.redis_client.get(sent_key)
            if sent_data:
                return json.loads(sent_data)
            
            # Check failed emails
            failed_key = f"email_failed:{email_id}"
            failed_data = self.redis_client.get(failed_key)
            if failed_data:
                return json.loads(failed_data)
            
            # Check fallback queue
            fallback_key = f"email_fallback:{email_id}"
            fallback_data = self.redis_client.get(fallback_key)
            if fallback_data:
                data = json.loads(fallback_data)
                data['status'] = 'fallback'
                return data
            
            return {'status': 'not_found', 'id': email_id, 'service': 'brevo'}
            
        except Exception as e:
            logger.error(f"Error getting email status: {e}")
            return {'status': 'error', 'id': email_id, 'service': 'brevo'}
    
    def get_fallback_emails(self, limit: int = 10) -> list:
        """Get recent fallback emails for manual processing"""
        if not self.redis_client:
            return []
        
        try:
            keys = self.redis_client.lrange('email_fallback_queue', 0, limit - 1)
            emails = []
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    email_data = json.loads(data)
                    email_data['fallback_key'] = key
                    emails.append(email_data)
            return emails
        except Exception as e:
            logger.error(f"Error getting fallback emails: {e}")
            return []

    def get_queue_stats(self) -> Dict:
        """Get email queue statistics"""
        if not self.redis_client:
            return {
                'queue_size': 0, 
                'fallback_queue_size': 0,
                'smtp_enabled': self.smtp_enabled,
                'api_enabled': self.api_enabled,
                'environment': ENVIRONMENT
            }
            
        try:
            queue_size = self.redis_client.llen('email_queue')
            fallback_queue_size = self.redis_client.llen('email_fallback_queue')
            
            return {
                'queue_size': queue_size,
                'fallback_queue_size': fallback_queue_size,
                'service': 'brevo',
                'smtp_enabled': self.smtp_enabled,
                'api_enabled': self.api_enabled,
                'method': 'smtp' if self.smtp_enabled else 'api' if self.api_enabled else 'none',
                'environment': ENVIRONMENT
            }
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {
                'queue_size': 0, 
                'fallback_queue_size': 0, 
                'error': str(e),
                'smtp_enabled': self.smtp_enabled,
                'api_enabled': self.api_enabled,
                'environment': ENVIRONMENT
            }


# Global email service instance
email_service = None


def init_auth(flask_app, database, user_model):
    """Initialize authentication service"""
    global app, db, User, serializer, email_service, redis_client

    app = flask_app
    db = database
    User = user_model
    redis_client = init_redis()
    email_service = BrevoEmailService()
    serializer = URLSafeTimedSerializer(app.secret_key)

    if email_service.email_enabled:
        method = "SMTP" if email_service.smtp_enabled else "API"
        logger.info(f"✅ Auth module initialized with Brevo {method} support [{ENVIRONMENT}]")
    else:
        logger.warning(f"⚠️ Auth module initialized WITHOUT email - using fallback mode [{ENVIRONMENT}]")
        if BREVO_API_KEY:
            logger.info(f"   API Key present: {BREVO_API_KEY[:20]}...")
        else:
            logger.warning("   No BREVO_API_KEY found in environment variables")


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
            redis_client.setex(f"reset_token:{token[:20]}", 3600, email)
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
    except Exception:
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
    if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, "Valid password"


def get_request_info(request):
    """Extract request information for logging"""
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
        except Exception:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function


class EnhancedUserAnalytics:
    """Enhanced user analytics for authentication"""
    
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
    'init_auth',
    'require_auth',
    'generate_reset_token',
    'verify_reset_token',
    'validate_password',
    'EnhancedUserAnalytics'
]