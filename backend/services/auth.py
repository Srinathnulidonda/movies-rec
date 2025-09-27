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
        
        # Extended SMTP configurations including alternative ports and servers
        self.smtp_configs = [
            {
                'name': 'Gmail TLS 587',
                'server': 'smtp.gmail.com',
                'port': 587,
                'use_tls': True,
                'use_ssl': False
            },
            {
                'name': 'Gmail Alternative 2525',
                'server': 'smtp.gmail.com',
                'port': 2525,  # Alternative port sometimes not blocked
                'use_tls': True,
                'use_ssl': False
            },
            {
                'name': 'Gmail Alternative 2587',
                'server': 'smtp.gmail.com',
                'port': 2587,  # Another alternative port
                'use_tls': True,
                'use_ssl': False
            },
            {
                'name': 'Gmail Relay 587',
                'server': 'smtp-relay.gmail.com',  # Gmail relay server
                'port': 587,
                'use_tls': True,
                'use_ssl': False
            },
            {
                'name': 'Gmail SSL 465',
                'server': 'smtp.gmail.com',
                'port': 465,
                'use_tls': False,
                'use_ssl': True
            },
            {
                'name': 'Gmail Alternative SSL 8465',
                'server': 'smtp.gmail.com',
                'port': 8465,  # Alternative SSL port
                'use_tls': False,
                'use_ssl': True
            },
            {
                'name': 'Gmail Submission 25',
                'server': 'smtp.gmail.com',
                'port': 25,
                'use_tls': True,
                'use_ssl': False
            },
            {
                'name': 'Gmail HTTP Tunnel 8025',
                'server': 'smtp.gmail.com',
                'port': 8025,  # Sometimes works on cloud providers
                'use_tls': True,
                'use_ssl': False
            }
        ]
        
        # Try to establish SMTP connection
        self.email_enabled = self._test_smtp_connection()
        
        # If direct SMTP fails, try alternative methods
        if not self.email_enabled:
            logger.warning("Direct SMTP failed, trying alternative methods...")
            self.email_enabled = self._try_alternative_smtp()
        
        if self.email_enabled:
            self.start_email_worker()
        else:
            logger.warning("⚠️ Email service running in fallback mode - direct links will be provided")
            self._setup_fallback_system()
    
    def _test_smtp_connection(self):
        """Test SMTP connectivity with extended timeout and retry logic"""
        for config in self.smtp_configs:
            for attempt in range(2):  # Try each config twice
                try:
                    logger.info(f"Testing {config['name']} connection (attempt {attempt + 1})...")
                    
                    # First test if port is reachable using basic socket
                    socket.setdefaulttimeout(10)
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(10)
                    
                    try:
                        # Try connecting to the server
                        result = sock.connect_ex((config['server'], config['port']))
                        sock.close()
                        
                        if result != 0:
                            logger.warning(f"{config['name']} port {config['port']} appears blocked (error code: {result})")
                            continue
                    except Exception as sock_err:
                        logger.warning(f"Socket test failed for {config['name']}: {sock_err}")
                        continue
                    
                    # Port seems open, try SMTP connection
                    if config['use_ssl']:
                        context = ssl.create_default_context()
                        # For testing - in production, use proper certificate validation
                        context.check_hostname = False
                        context.verify_mode = ssl.CERT_NONE
                        
                        server = smtplib.SMTP_SSL(config['server'], config['port'], 
                                                 context=context, timeout=15)
                        try:
                            server.set_debuglevel(1)  # Enable debug output
                            server.login(self.username, self.password)
                            server.quit()
                            logger.info(f"✅ {config['name']} connection successful!")
                            self.working_config = config
                            return True
                        except Exception as smtp_err:
                            logger.warning(f"SMTP SSL error for {config['name']}: {smtp_err}")
                            server.quit()
                    else:
                        server = smtplib.SMTP(config['server'], config['port'], timeout=15)
                        try:
                            server.set_debuglevel(1)
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
                            logger.warning(f"SMTP error for {config['name']}: {smtp_err}")
                            try:
                                server.quit()
                            except:
                                pass
                            
                except socket.timeout:
                    logger.warning(f"{config['name']} connection timed out (attempt {attempt + 1})")
                except socket.gaierror as e:
                    logger.warning(f"{config['name']} DNS resolution failed: {e}")
                except smtplib.SMTPAuthenticationError as e:
                    logger.error(f"{config['name']} authentication failed: {e}")
                    if "Application-specific password required" in str(e):
                        logger.error("⚠️ Gmail requires an App Password, not regular password!")
                        logger.error("Go to: https://myaccount.google.com/apppasswords to generate one")
                except Exception as e:
                    logger.warning(f"{config['name']} unexpected error: {type(e).__name__}: {e}")
                
                time.sleep(1)  # Small delay between attempts
        
        logger.error("❌ All standard SMTP configurations failed")
        return False
    
    def _try_alternative_smtp(self):
        """Try alternative SMTP methods including proxy and relay services"""
        try:
            # Try using environment variable for custom SMTP server if provided
            custom_smtp = os.environ.get('CUSTOM_SMTP_SERVER')
            custom_port = int(os.environ.get('CUSTOM_SMTP_PORT', 587))
            
            if custom_smtp:
                logger.info(f"Trying custom SMTP server: {custom_smtp}:{custom_port}")
                try:
                    server = smtplib.SMTP(custom_smtp, custom_port, timeout=15)
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(self.username, self.password)
                    server.quit()
                    
                    self.working_config = {
                        'name': 'Custom SMTP',
                        'server': custom_smtp,
                        'port': custom_port,
                        'use_tls': True,
                        'use_ssl': False
                    }
                    logger.info("✅ Custom SMTP server connected successfully!")
                    return True
                except Exception as e:
                    logger.error(f"Custom SMTP failed: {e}")
            
            # If no custom SMTP or it failed, return False
            return False
            
        except Exception as e:
            logger.error(f"Alternative SMTP methods failed: {e}")
            return False
    
    def _setup_fallback_system(self):
        """Setup fallback system for when SMTP is completely blocked"""
        try:
            logger.info("📧 Setting up email fallback system")
            self.use_http_fallback = True
            
            # Create a simple queue system in Redis for storing emails
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
        """Send email via SMTP with retry logic"""
        if not self.email_enabled:
            logger.warning(f"Email service disabled - storing email for {email_data['to']}")
            self._store_fallback_email(email_data)
            return False
        
        max_retries = 3
        retry_count = email_data.get('retry_count', 0)
        
        for attempt in range(max_retries - retry_count):
            try:
                msg = MIMEMultipart('alternative')
                msg['From'] = formataddr((self.from_name, self.username))
                msg['To'] = email_data['to']
                msg['Subject'] = email_data['subject']
                msg['Reply-To'] = self.reply_to
                msg['Date'] = formatdate(localtime=True)
                msg['Message-ID'] = f"<{email_data.get('id', uuid.uuid4())}@cinebrain.com>"
                
                text_part = MIMEText(email_data['text'], 'plain', 'utf-8')
                html_part = MIMEText(email_data['html'], 'html', 'utf-8')
                
                msg.attach(text_part)
                msg.attach(html_part)
                
                config = self.working_config
                
                # Send email using the working configuration
                if config['use_ssl']:
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    
                    with smtplib.SMTP_SSL(config['server'], config['port'], 
                                         context=context, timeout=30) as server:
                        server.login(self.username, self.password)
                        server.send_message(msg)
                else:
                    with smtplib.SMTP(config['server'], config['port'], timeout=30) as server:
                        server.ehlo()
                        if config['use_tls']:
                            context = ssl.create_default_context()
                            context.check_hostname = False
                            context.verify_mode = ssl.CERT_NONE
                            server.starttls(context=context)
                            server.ehlo()
                        server.login(self.username, self.password)
                        server.send_message(msg)
                
                logger.info(f"✅ Email sent successfully to {email_data['to']}")
                
                # Log success in Redis
                if self.redis_client:
                    self.redis_client.setex(
                        f"email_sent:{email_data.get('id', 'unknown')}",
                        86400,
                        json.dumps({
                            'status': 'sent',
                            'timestamp': datetime.utcnow().isoformat(),
                            'to': email_data['to'],
                            'method': config['name']
                        })
                    )
                return True
                
            except smtplib.SMTPServerDisconnected:
                logger.error(f"SMTP server disconnected, trying to reconnect...")
                self.email_enabled = self._test_smtp_connection()
                if not self.email_enabled:
                    self._store_fallback_email(email_data)
                    return False
                    
            except Exception as e:
                logger.error(f"SMTP Error: {e} (attempt {attempt + 1}/{max_retries})")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self._store_fallback_email(email_data)
                    return False
        
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
                
                # If it's a password reset, also store direct access
                if email_data.get('reset_token'):
                    reset_url = f"{FRONTEND_URL}/auth/reset-password.html?token={email_data.get('reset_token')}"
                    self.redis_client.setex(
                        f"direct_reset:{email_data['to']}",
                        3600,  # 1 hour
                        json.dumps({
                            'url': reset_url,
                            'token': email_data.get('reset_token'),
                            'created_at': datetime.utcnow().isoformat()
                        })
                    )
                    logger.info(f"🔗 Direct reset link stored for {email_data['to']}: {reset_url}")
                    
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
            # If email is disabled and this is critical (password reset), handle specially
            if not self.email_enabled:
                logger.warning(f"Email service disabled - using fallback for {to}")
                self._store_fallback_email(email_data)
                
                if reset_token:
                    reset_url = f"{FRONTEND_URL}/auth/reset-password.html?token={reset_token}"
                    logger.info(f"🔗 Password reset link for {to}: {reset_url}")
                    
                    # Store for API retrieval
                    if self.redis_client:
                        self.redis_client.setex(
                            f"pending_reset:{to}",
                            3600,
                            json.dumps({
                                'token': reset_token,
                                'url': reset_url,
                                'email_id': email_id,
                                'created_at': datetime.utcnow().isoformat()
                            })
                        )
                
                return True
            
            # Normal email queueing when service is enabled
            if self.redis_client:
                if priority == 'high':
                    self.redis_client.lpush('email_queue', json.dumps(email_data))
                else:
                    self.redis_client.rpush('email_queue', json.dumps(email_data))
                
                logger.info(f"📧 Email queued for {to} - ID: {email_id}")
            else:
                # Direct send if no Redis
                logger.warning("Redis not available, attempting direct send")
                threading.Thread(
                    target=self._send_email_smtp,
                    args=(email_data,),
                    daemon=True
                ).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue email: {e}")
            
            # Last resort - log the reset link
            if reset_token:
                reset_url = f"{FRONTEND_URL}/auth/reset-password.html?token={reset_token}"
                logger.critical(f"EMERGENCY: Reset link for {to}: {reset_url}")
            
            return False
    
    def get_email_status(self, email_id: str) -> Dict:
        """Get status of a queued email"""
        if not self.redis_client:
            return {'status': 'unknown', 'id': email_id}
        
        try:
            # Check various status keys
            for status_type in ['sent', 'failed', 'queued']:
                key = f"email_{status_type}:{email_id}"
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            
            # Check fallback queue
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
    
    def get_professional_template(self, content_type: str, **kwargs) -> tuple:
        """Generate professional email templates"""
        base_css = """
        <style type="text/css">
            @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Inter:wght@300;400;500;600;700&display=swap');
            
            body {
                margin: 0;
                padding: 0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                font-size: 16px;
                line-height: 1.6;
                color: #1a1a1a;
                background: #f8f9fa;
            }
            
            .email-wrapper {
                width: 100%;
                background: #f8f9fa;
                padding: 32px 16px;
            }
            
            .email-container {
                max-width: 600px;
                margin: 0 auto;
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(17,60,207,0.2);
                overflow: hidden;
                border: 1px solid #e8eaed;
            }
            
            .header {
                background: linear-gradient(135deg, #113CCF 0%, #1E4FE5 50%, #1E4FE5 100%);
                padding: 48px 32px;
                text-align: center;
            }
            
            .brand-logo {
                font-family: 'Bangers', cursive;
                font-size: 42px;
                font-weight: 400;
                letter-spacing: 1px;
                color: #ffffff;
                margin: 0;
            }
            
            .brand-tagline {
                font-size: 14px;
                color: rgba(255,255,255,0.95);
                margin: 8px 0 0;
            }
            
            .content {
                padding: 48px 32px;
                background: #ffffff;
            }
            
            .content-title {
                font-size: 32px;
                font-weight: 600;
                color: #1a1a1a;
                margin: 0 0 16px;
                text-align: center;
            }
            
            .content-body {
                font-size: 16px;
                line-height: 1.7;
                color: #1a1a1a;
                margin-bottom: 24px;
            }
            
            .btn {
                display: inline-block;
                font-size: 16px;
                font-weight: 600;
                text-decoration: none;
                text-align: center;
                padding: 16px 32px;
                border-radius: 50px;
                background: linear-gradient(135deg, #113CCF 0%, #1E4FE5 100%);
                color: #ffffff !important;
                min-width: 200px;
            }
            
            .btn-container {
                text-align: center;
                margin: 32px 0;
            }
            
            .alert {
                padding: 16px 24px;
                border-radius: 12px;
                margin: 24px 0;
                background: rgba(245,158,11,0.1);
                border-left: 4px solid #f59e0b;
                color: #d97706;
            }
            
            .footer {
                background: #f8f9fa;
                padding: 32px;
                text-align: center;
                border-top: 1px solid #e8eaed;
            }
            
            .footer-text {
                font-size: 12px;
                color: #999999;
                margin: 8px 0;
            }
        </style>
        """
        
        if content_type == 'password_reset':
            return self._get_password_reset_template(base_css, **kwargs)
        elif content_type == 'password_changed':
            return self._get_password_changed_template(base_css, **kwargs)
        else:
            return self._get_generic_template(base_css, **kwargs)
    
    def _get_password_reset_template(self, base_css: str, **kwargs) -> tuple:
        reset_url = kwargs.get('reset_url', '')
        user_name = kwargs.get('user_name', 'there')
        user_email = kwargs.get('user_email', '')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reset your password - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header">
                        <div class="brand-logo">CineBrain</div>
                        <div class="brand-tagline">The Mind Behind Your Next Favorite</div>
                    </div>
                    
                    <div class="content">
                        <h1 class="content-title">Reset your password</h1>
                        
                        <div class="content-body">
                            <p>Hi {user_name},</p>
                            <p>We received a request to reset your CineBrain account password. Click the button below to create a new password:</p>
                        </div>
                        
                        <div class="btn-container">
                            <a href="{reset_url}" class="btn">Reset Password</a>
                        </div>
                        
                        <div class="alert">
                            <strong>⏰ This link expires in 1 hour</strong><br>
                            For security reasons, this password reset link will expire soon.
                        </div>
                        
                        <div style="margin-top: 24px; padding: 16px; background: #f8f9fa; border-radius: 8px;">
                            <p style="margin: 0; font-size: 13px; color: #666;">
                                Can't click the button? Copy this link:<br>
                                <code style="word-break: break-all;">{reset_url}</code>
                            </p>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p class="footer-text">
                            If you didn't request this, you can safely ignore this email.
                        </p>
                        <p class="footer-text">
                            © {datetime.now().year} CineBrain, Inc.
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Reset your password - CineBrain

Hi {user_name},

We received a request to reset your CineBrain account password.

To reset your password, visit:
{reset_url}

This link expires in 1 hour.

If you didn't request this, you can safely ignore this email.

Best regards,
The CineBrain Team

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text
    
    def _get_password_changed_template(self, base_css: str, **kwargs) -> tuple:
        user_name = kwargs.get('user_name', 'there')
        user_email = kwargs.get('user_email', '')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Password changed - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header" style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);">
                        <div class="brand-logo">✅ Password Changed</div>
                        <div class="brand-tagline">Your account is now secured</div>
                    </div>
                    
                    <div class="content">
                        <h1 class="content-title">Password successfully changed</h1>
                        
                        <div class="content-body">
                            <p>Hi {user_name},</p>
                            <p>Your CineBrain account password was successfully changed.</p>
                        </div>
                        
                        <div class="btn-container">
                            <a href="{FRONTEND_URL}/login" class="btn">Sign in to CineBrain</a>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p class="footer-text">
                            If you didn't make this change, please contact support immediately.
                        </p>
                        <p class="footer-text">
                            © {datetime.now().year} CineBrain, Inc.
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Password Changed Successfully - CineBrain

Hi {user_name},

Your CineBrain account password was successfully changed.

If you didn't make this change, please contact support immediately.

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text
    
    def _get_generic_template(self, base_css: str, **kwargs) -> tuple:
        subject = kwargs.get('subject', 'CineBrain')
        content = kwargs.get('content', '')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{subject}</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header">
                        <div class="brand-logo">CineBrain</div>
                    </div>
                    <div class="content">
                        <div class="content-body">{content}</div>
                    </div>
                    <div class="footer">
                        <p class="footer-text">© {datetime.now().year} CineBrain, Inc.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"{subject}\n\n{content}\n\n© {datetime.now().year} CineBrain, Inc."
        
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
        
        if user:
            token = generate_reset_token(email)
            reset_url = f"{FRONTEND_URL}/auth/reset-password.html?token={token}"
            
            # Try to send email
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
                reset_token=token
            )
            
            logger.info(f"Password reset requested for {email}")
            
            # Check if email service is disabled and provide fallback
            if not email_service.email_enabled or email_service.use_http_fallback:
                # For testing/development - remove in production
                return jsonify({
                    'success': True,
                    'message': 'Password reset link generated',
                    'fallback_mode': True,
                    'reset_url': reset_url,  # REMOVE IN PRODUCTION
                    'note': 'Email service is currently unavailable. Use the link above to reset your password.'
                }), 200
        
        # Always return success to prevent email enumeration
        return jsonify({
            'success': True,
            'message': 'If an account exists with this email, you will receive password reset instructions shortly.'
        }), 200
        
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
        
        # Clear the reset token from Redis
        if redis_client:
            try:
                redis_client.delete(f"reset_token:{token[:20]}")
                redis_client.delete(f"pending_reset:{email}")
                redis_client.delete(f"direct_reset:{email}")
            except:
                pass
        
        ip_address, location, device = get_request_info(request)
        
        # Send confirmation email
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
            priority='high'
        )
        
        # Generate new auth token
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
        
        emails = email_service.get_fallback_emails(limit=50)
        
        # Also get any pending reset links
        pending_resets = []
        if redis_client:
            for key in redis_client.scan_iter("pending_reset:*"):
                data = redis_client.get(key)
                if data:
                    reset_info = json.loads(data)
                    reset_info['email'] = key.replace('pending_reset:', '')
                    pending_resets.append(reset_info)
        
        return jsonify({
            'success': True,
            'fallback_emails': emails,
            'pending_resets': pending_resets,
            'smtp_enabled': email_service.email_enabled,
            'smtp_config': email_service.working_config['name'] if email_service.working_config else None,
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
        
        queue_size = 0
        fallback_queue_size = 0
        if redis_client:
            try:
                queue_size = redis_client.llen('email_queue')
                fallback_queue_size = redis_client.llen('email_fallback_queue')
            except:
                pass
        
        smtp_config = None
        if email_service and email_service.working_config:
            smtp_config = email_service.working_config.get('name', 'Unknown')
        
        return jsonify({
            'status': 'healthy',
            'service': 'authentication',
            'email_service': 'Gmail SMTP',
            'email_configured': email_configured,
            'email_enabled': email_enabled,
            'smtp_config': smtp_config,
            'email_queue_size': queue_size,
            'fallback_queue_size': fallback_queue_size,
            'fallback_mode': email_service.use_http_fallback if email_service else True,
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