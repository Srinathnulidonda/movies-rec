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
RESEND_API_KEY = os.getenv('RESEND_API_KEY')
MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER')
REPLY_TO_EMAIL = os.getenv('REPLY_TO_EMAIL')

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


class ResendEmailService:
    """Email service using Resend API (supports free sandbox mode)"""

    def __init__(self, api_key=None):
        self.api_key = api_key or RESEND_API_KEY
        self.from_email = MAIL_DEFAULT_SENDER.split('<')[-1].replace('>', '').strip()
        self.from_name = MAIL_DEFAULT_SENDER.split('<')[0].strip()
        self.reply_to_email = REPLY_TO_EMAIL
        self.base_url = "https://api.resend.com"
        self.redis_client = redis_client
        self.email_enabled = False
        self.is_configured = bool(self.api_key)

        if not self.api_key:
            logger.warning("⚠️ RESEND_API_KEY not set — email service disabled")
            return

        # Check sandbox or production
        if "resend.dev" in self.from_email:
            logger.info("📦 Using Resend Sandbox mode (free tier)")
            self.email_enabled = True
        else:
            self.email_enabled = self._test_resend_connection()

        if self.email_enabled:
            self.start_email_worker()
            logger.info("✅ Resend email worker initialized successfully")
        else:
            logger.warning("⚠️ Email service disabled - Resend API connection failed or not configured")

    def _test_resend_connection(self):
        """Basic Resend API connectivity test"""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
            response = requests.get(f"{self.base_url}/api/v1", headers=headers, timeout=10)
            if response.status_code in [200, 404]:
                logger.info("✅ Resend API reachable (production mode)")
                return True
            logger.warning(f"⚠️ Resend test returned status: {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"❌ Resend API connection test failed: {e}")
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
                            self._send_email_resend(email_data)
                        else:
                            time.sleep(1)
                    else:
                        time.sleep(5)
                except Exception as e:
                    logger.error(f"Email worker error: {e}")
                    time.sleep(5)

        threading.Thread(target=worker, daemon=True, name="ResendEmailWorker").start()

    def _send_email_resend(self, email_data: Dict):
        """Send email using Resend API"""
        if not self.email_enabled:
            self._store_fallback_email(email_data)
            return

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'from': f"{self.from_name} <{self.from_email}>",
            'to': [email_data['to']],
            'subject': email_data['subject'],
            'html': email_data['html'],
            'text': email_data.get('text', ''),
            'reply_to': self.reply_to_email
        }

        try:
            response = requests.post(f"{self.base_url}/emails", headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                res = response.json()
                logger.info(f"✅ Email sent successfully to {email_data['to']} (ID: {res.get('id')})")
                if self.redis_client:
                    self.redis_client.setex(f"email_sent:{res.get('id')}", 86400, json.dumps({
                        'status': 'sent',
                        'timestamp': datetime.utcnow().isoformat(),
                        'to': email_data['to']
                    }))
            else:
                raise Exception(f"Resend returned {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"❌ Email send failed to {email_data['to']}: {e}")
            self._store_fallback_email(email_data)

    def _store_fallback_email(self, email_data: Dict):
        """Store unsent email in fallback queue"""
        if not self.redis_client:
            return
        key = f"email_fallback:{uuid.uuid4()}"
        self.redis_client.setex(key, 604800, json.dumps(email_data))
        self.redis_client.rpush('email_fallback_queue', key)
        logger.info(f"📥 Stored unsent email for {email_data['to']} in fallback queue")

    def queue_email(self, to: str, subject: str, html: str, text: str = "", priority: str = 'normal', reset_token: str = None):
        """Queue email to be sent asynchronously"""
        email_id = str(uuid.uuid4())
        email_data = {
            'id': email_id,
            'to': to,
            'subject': subject,
            'html': html,
            'text': text,
            'timestamp': datetime.utcnow().isoformat(),
            'reset_token': reset_token
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
            threading.Thread(target=self._send_email_resend, args=(email_data,), daemon=True).start()
        
        return True

    def get_email_status(self, email_id: str) -> Dict:
        """Get email delivery status"""
        if not self.redis_client:
            return {'status': 'unknown', 'id': email_id}
        
        try:
            for status_type in ['sent', 'failed', 'queued']:
                key = f"email_{status_type}:{email_id}"
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            
            fallback_key = f"email_fallback:{email_id}"
            fallback_data = self.redis_client.get(fallback_key)
            if fallback_data:
                return json.loads(fallback_data)
            
            return {'status': 'not_found', 'id': email_id}
        except Exception as e:
            logger.error(f"Error getting email status: {e}")
            return {'status': 'error', 'id': email_id}
    
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
                    emails.append(json.loads(data))
            return emails
        except Exception as e:
            logger.error(f"Error getting fallback emails: {e}")
            return []


# Global email service instance
email_service = None


def init_auth(flask_app, database, user_model):
    """Initialize authentication service"""
    global app, db, User, serializer, email_service, redis_client

    app = flask_app
    db = database
    User = user_model
    redis_client = init_redis()
    email_service = ResendEmailService()
    serializer = URLSafeTimedSerializer(app.secret_key)

    if email_service.email_enabled:
        logger.info("✅ Auth module initialized with Resend email support")
    else:
        logger.warning("⚠️ Auth module initialized WITHOUT email - using fallback mode")


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