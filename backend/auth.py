# backend/auth.py
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__)

# Frontend URL configuration
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')
BACKEND_URL = os.environ.get('BACKEND_URL', 'https://backend-app-970m.onrender.com')

# Redis configuration
REDIS_URL = os.environ.get('REDIS_URL', 'redis://red-d2qlbuje5dus73c71qog:xp7inVzgblGCbo9I4taSGLdKUg0xY91I@red-d2qlbuje5dus73c71qog:6379')

# Email validation regex
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# These will be initialized by init_auth function
app = None
db = None
User = None
mail = None
serializer = None
redis_client = None

# Password reset token salt
PASSWORD_RESET_SALT = 'password-reset-salt-cinebrain-2025'

def init_redis():
    """Initialize Redis connection"""
    global redis_client
    try:
        # Parse Redis URL
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
        
        # Test connection
        redis_client.ping()
        logger.info("Redis connected successfully")
        return redis_client
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        # Fall back to in-memory storage if Redis fails
        return None

class ProfessionalEmailService:
    """Professional email service using Gmail SMTP with Redis queue"""
    
    def __init__(self, username, password):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.username = username
        self.password = password
        self.from_email = "noreply@cinebrain.com"  # Display email
        self.from_name = "CineBrain"
        self.reply_to = "support@cinebrain.com"
        
        # Initialize Redis
        self.redis_client = redis_client
        
        # Start email worker
        self.start_email_worker()
    
    def start_email_worker(self):
        """Start background thread for sending emails from Redis queue"""
        def worker():
            while True:
                try:
                    # Try to get email from Redis queue
                    if self.redis_client:
                        email_json = self.redis_client.lpop('email_queue')
                        if email_json:
                            email_data = json.loads(email_json)
                            self._send_email_smtp(email_data)
                        else:
                            time.sleep(1)  # Wait if queue is empty
                    else:
                        time.sleep(5)  # Wait longer if Redis is not available
                except Exception as e:
                    logger.error(f"Email worker error: {e}")
                    time.sleep(5)
        
        # Start multiple worker threads for better performance
        for i in range(3):
            thread = threading.Thread(target=worker, daemon=True, name=f"EmailWorker-{i}")
            thread.start()
            logger.info(f"Started email worker thread {i}")
    
    def _send_email_smtp(self, email_data: Dict):
        """Send email using Gmail SMTP with retry logic"""
        max_retries = 3
        retry_count = email_data.get('retry_count', 0)
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            
            # Professional headers for better deliverability
            msg['From'] = formataddr((self.from_name, self.username))
            msg['To'] = email_data['to']
            msg['Subject'] = email_data['subject']
            msg['Reply-To'] = self.reply_to
            msg['Date'] = formatdate(localtime=True)
            msg['Message-ID'] = f"<{email_data.get('id', uuid.uuid4())}@cinebrain.com>"
            
            # Anti-spam headers
            msg['X-Priority'] = '1' if email_data.get('priority') == 'high' else '3'
            msg['X-Mailer'] = 'CineBrain-Mailer/2.0'
            msg['X-Entity-Ref-ID'] = str(uuid.uuid4())
            msg['List-Unsubscribe'] = f'<mailto:unsubscribe@cinebrain.com?subject=Unsubscribe>'
            msg['List-Unsubscribe-Post'] = 'List-Unsubscribe=One-Click'
            msg['Precedence'] = 'bulk'
            msg['Auto-Submitted'] = 'auto-generated'
            msg['X-Auto-Response-Suppress'] = 'All'
            msg['X-Campaign-Id'] = 'password-reset' if 'reset' in email_data['subject'].lower() else 'transactional'
            
            # Add both plain text and HTML parts
            text_part = MIMEText(email_data['text'], 'plain', 'utf-8')
            html_part = MIMEText(email_data['html'], 'html', 'utf-8')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send email with SSL
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"✅ Email sent successfully to {email_data['to']} - Subject: {email_data['subject']}")
            
            # Store success in Redis
            if self.redis_client:
                self.redis_client.setex(
                    f"email_sent:{email_data.get('id', 'unknown')}",
                    86400,  # 24 hours TTL
                    json.dumps({
                        'status': 'sent',
                        'timestamp': datetime.utcnow().isoformat(),
                        'to': email_data['to']
                    })
                )
            
        except Exception as e:
            logger.error(f"❌ SMTP Error sending to {email_data['to']}: {e}")
            
            # Retry logic with exponential backoff
            if retry_count < max_retries:
                retry_count += 1
                email_data['retry_count'] = retry_count
                retry_delay = 5 * (2 ** retry_count)  # 5, 10, 20 seconds
                
                logger.info(f"🔄 Retrying email to {email_data['to']} in {retry_delay} seconds (attempt {retry_count}/{max_retries})")
                
                # Re-queue with delay
                if self.redis_client:
                    threading.Timer(
                        retry_delay,
                        lambda: self.redis_client.rpush('email_queue', json.dumps(email_data))
                    ).start()
            else:
                logger.error(f"❌ Failed to send email after {max_retries} attempts to {email_data['to']}")
                
                # Store failure in Redis
                if self.redis_client:
                    self.redis_client.setex(
                        f"email_failed:{email_data.get('id', 'unknown')}",
                        86400,  # 24 hours TTL
                        json.dumps({
                            'status': 'failed',
                            'error': str(e),
                            'timestamp': datetime.utcnow().isoformat(),
                            'to': email_data['to']
                        })
                    )
    
    def queue_email(self, to: str, subject: str, html: str, text: str, priority: str = 'normal'):
        """Add email to Redis queue for async sending"""
        email_id = str(uuid.uuid4())
        email_data = {
            'id': email_id,
            'to': to,
            'subject': subject,
            'html': html,
            'text': text,
            'priority': priority,
            'timestamp': datetime.utcnow().isoformat(),
            'retry_count': 0
        }
        
        try:
            if self.redis_client:
                # Use Redis queue
                if priority == 'high':
                    # Add to front of queue for high priority
                    self.redis_client.lpush('email_queue', json.dumps(email_data))
                else:
                    # Add to back of queue for normal priority
                    self.redis_client.rpush('email_queue', json.dumps(email_data))
                
                # Track email status
                self.redis_client.setex(
                    f"email_queued:{email_id}",
                    3600,  # 1 hour TTL
                    json.dumps({
                        'status': 'queued',
                        'timestamp': datetime.utcnow().isoformat(),
                        'to': to,
                        'subject': subject
                    })
                )
                
                logger.info(f"📧 Email queued (Redis) for {to} - ID: {email_id}")
            else:
                # Fallback: send directly if Redis is not available
                logger.warning("Redis not available, sending email directly")
                threading.Thread(
                    target=self._send_email_smtp,
                    args=(email_data,),
                    daemon=True
                ).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue email: {e}")
            # Try to send directly as last resort
            threading.Thread(
                target=self._send_email_smtp,
                args=(email_data,),
                daemon=True
            ).start()
            return True
    
    def get_email_status(self, email_id: str) -> Dict:
        """Get email status from Redis"""
        if not self.redis_client:
            return {'status': 'unknown', 'id': email_id}
        
        try:
            # Check different status keys
            for status_type in ['sent', 'failed', 'queued']:
                key = f"email_{status_type}:{email_id}"
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            
            return {'status': 'not_found', 'id': email_id}
        except Exception as e:
            logger.error(f"Error getting email status: {e}")
            return {'status': 'error', 'id': email_id}
    
    def get_professional_template(self, content_type: str, **kwargs) -> tuple:
        """Get professional HTML and text email templates"""
        
        # Base CSS for all emails (Google/Microsoft/Spotify style)
        base_css = """
        <style type="text/css">
            /* Reset styles */
            body, table, td, a { -webkit-text-size-adjust: 100%; -ms-text-size-adjust: 100%; }
            table, td { mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
            img { -ms-interpolation-mode: bicubic; border: 0; outline: none; text-decoration: none; }
            
            /* Base styles */
            body {
                margin: 0 !important;
                padding: 0 !important;
                width: 100% !important;
                min-width: 100% !important;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
                font-size: 14px;
                line-height: 1.6;
                color: #202124;
                background-color: #f8f9fa;
            }
            
            /* Container styles */
            .email-wrapper {
                width: 100%;
                background-color: #f8f9fa;
                padding: 40px 20px;
            }
            
            .email-container {
                max-width: 600px;
                margin: 0 auto;
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
                overflow: hidden;
            }
            
            /* Header styles - Spotify inspired gradient */
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 40px 48px;
                text-align: center;
            }
            
            .header-logo {
                font-size: 32px;
                font-weight: 700;
                color: #ffffff;
                letter-spacing: -0.5px;
                margin: 0;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .header-tagline {
                font-size: 14px;
                color: rgba(255,255,255,0.95);
                margin: 8px 0 0 0;
                font-weight: 400;
            }
            
            /* Content styles */
            .content {
                padding: 48px;
                background-color: #ffffff;
            }
            
            h1 {
                font-size: 24px;
                font-weight: 400;
                color: #202124;
                margin: 0 0 24px;
                line-height: 1.3;
            }
            
            h2 {
                font-size: 18px;
                font-weight: 500;
                color: #202124;
                margin: 24px 0 12px;
            }
            
            p {
                margin: 0 0 16px;
                color: #5f6368;
                font-size: 14px;
                line-height: 1.6;
            }
            
            /* Button styles - Google Material Design */
            .btn {
                display: inline-block;
                padding: 12px 32px;
                font-size: 14px;
                font-weight: 500;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                text-decoration: none !important;
                text-align: center;
                border-radius: 24px;
                transition: all 0.3s;
                cursor: pointer;
                margin: 24px 0;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #ffffff !important;
                box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4);
            }
            
            .btn-primary:hover {
                box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
                transform: translateY(-1px);
            }
            
            /* Alert boxes */
            .alert {
                padding: 16px;
                border-radius: 8px;
                margin: 24px 0;
                font-size: 14px;
            }
            
            .alert-info {
                background-color: #e8f0fe;
                border-left: 4px solid #1a73e8;
                color: #1967d2;
            }
            
            .alert-success {
                background-color: #e6f4ea;
                border-left: 4px solid #34a853;
                color: #188038;
            }
            
            .alert-warning {
                background-color: #fef7e0;
                border-left: 4px solid #fbbc04;
                color: #ea8600;
            }
            
            .alert-error {
                background-color: #fce8e6;
                border-left: 4px solid #ea4335;
                color: #d33b27;
            }
            
            /* Code block */
            .code-block {
                background-color: #f8f9fa;
                border: 1px solid #dadce0;
                border-radius: 8px;
                padding: 16px;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Courier New', monospace;
                font-size: 13px;
                color: #202124;
                word-break: break-all;
                margin: 16px 0;
            }
            
            /* Info box */
            .info-box {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 24px;
                margin: 24px 0;
                border: 1px solid #e8eaed;
            }
            
            /* Footer */
            .footer {
                background-color: #f8f9fa;
                padding: 32px 48px;
                text-align: center;
                border-top: 1px solid #e8eaed;
            }
            
            .footer-text {
                font-size: 12px;
                color: #80868b;
                margin: 0 0 8px;
                line-height: 1.5;
            }
            
            .footer-links {
                margin: 16px 0;
            }
            
            .footer-link {
                color: #1a73e8 !important;
                text-decoration: none;
                font-size: 12px;
                margin: 0 12px;
            }
            
            .footer-link:hover {
                text-decoration: underline;
            }
            
            .social-links {
                margin: 20px 0;
            }
            
            .social-link {
                display: inline-block;
                margin: 0 8px;
                text-decoration: none;
            }
            
            /* Divider */
            .divider {
                height: 1px;
                background-color: #e8eaed;
                margin: 32px 0;
            }
            
            /* Responsive */
            @media screen and (max-width: 600px) {
                .email-wrapper {
                    padding: 0 !important;
                }
                .email-container {
                    width: 100% !important;
                    border-radius: 0 !important;
                }
                .content, .footer {
                    padding: 32px 24px !important;
                }
                .header {
                    padding: 32px 24px !important;
                }
                h1 {
                    font-size: 20px !important;
                }
                .btn {
                    display: block !important;
                    width: 100% !important;
                }
            }
            
            /* Dark mode support */
            @media (prefers-color-scheme: dark) {
                body { background-color: #202124 !important; }
                .email-wrapper { background-color: #202124 !important; }
                .email-container { 
                    background-color: #292a2d !important;
                    box-shadow: 0 1px 2px 0 rgba(0,0,0,0.6), 0 2px 6px 2px rgba(0,0,0,0.3) !important;
                }
                .content { background-color: #292a2d !important; }
                .footer { background-color: #202124 !important; }
                h1, h2 { color: #e8eaed !important; }
                p { color: #9aa0a6 !important; }
                .footer-text { color: #9aa0a6 !important; }
                .info-box, .code-block { 
                    background-color: #35363a !important;
                    border-color: #5f6368 !important;
                }
                .divider { background-color: #5f6368 !important; }
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
        """Password reset email template"""
        reset_url = kwargs.get('reset_url', '')
        user_name = kwargs.get('user_name', 'there')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <title>Reset your password - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                    <tr>
                        <td>
                            <div class="email-container">
                                <!-- Header -->
                                <div class="header">
                                    <h1 class="header-logo">🎬 CineBrain</h1>
                                    <p class="header-tagline">Your AI-Powered Entertainment Companion</p>
                                </div>
                                
                                <!-- Content -->
                                <div class="content">
                                    <h1>Reset your password</h1>
                                    
                                    <p>Hi {user_name},</p>
                                    
                                    <p>We received a request to reset your CineBrain account password. Click the button below to create a new password:</p>
                                    
                                    <center>
                                        <a href="{reset_url}" class="btn btn-primary">Reset Password</a>
                                    </center>
                                    
                                    <div class="info-box">
                                        <p style="margin: 0; font-size: 13px; color: #5f6368;">
                                            <strong>Can't click the button?</strong><br>
                                            Copy and paste this link into your browser:
                                        </p>
                                        <div class="code-block">
                                            {reset_url}
                                        </div>
                                    </div>
                                    
                                    <div class="alert alert-warning">
                                        <strong>⏰ This link expires in 1 hour</strong><br>
                                        For security reasons, this password reset link will expire soon.
                                    </div>
                                    
                                    <div class="divider"></div>
                                    
                                    <p style="font-size: 13px; color: #5f6368;">
                                        <strong>Didn't request this?</strong><br>
                                        If you didn't request a password reset, you can safely ignore this email. Your password won't be changed.
                                    </p>
                                </div>
                                
                                <!-- Footer -->
                                <div class="footer">
                                    <div class="footer-links">
                                        <a href="{FRONTEND_URL}/privacy" class="footer-link">Privacy</a>
                                        <a href="{FRONTEND_URL}/terms" class="footer-link">Terms</a>
                                        <a href="{FRONTEND_URL}/help" class="footer-link">Help</a>
                                    </div>
                                    
                                    <p class="footer-text">
                                        © {datetime.now().year} CineBrain, Inc. All rights reserved.<br>
                                        This email was sent to {kwargs.get('user_email', '')}
                                    </p>
                                </div>
                            </div>
                        </td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Reset your password

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
        """Password changed confirmation template"""
        user_name = kwargs.get('user_name', 'there')
        user_email = kwargs.get('user_email', '')
        change_time = datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Password changed - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                    <tr>
                        <td>
                            <div class="email-container">
                                <!-- Header with success theme -->
                                <div class="header" style="background: linear-gradient(135deg, #34a853 0%, #0d8043 100%);">
                                    <h1 class="header-logo">✅ Password Changed</h1>
                                    <p class="header-tagline">Your account is now secured</p>
                                </div>
                                
                                <!-- Content -->
                                <div class="content">
                                    <h1>Password successfully changed</h1>
                                    
                                    <p>Hi {user_name},</p>
                                    
                                    <p>Your CineBrain account password was successfully changed.</p>
                                    
                                    <div class="alert alert-success">
                                        <strong>✓ Your account is secured</strong><br>
                                        You can now sign in with your new password.
                                    </div>
                                    
                                    <center>
                                        <a href="{FRONTEND_URL}/login" class="btn btn-primary">Sign in to CineBrain</a>
                                    </center>
                                    
                                    <div class="alert alert-error">
                                        <strong>⚠️ Didn't make this change?</strong><br>
                                        If you didn't change your password, 
                                        <a href="{FRONTEND_URL}/security/recover" style="color: #ea4335; font-weight: bold;">secure your account immediately</a>
                                    </div>
                                    
                                    <p style="font-size: 12px; color: #5f6368; margin-top: 24px;">
                                        <strong>Change details:</strong><br>
                                        Time: {change_time}<br>
                                        IP: {kwargs.get('ip_address', 'Unknown')}<br>
                                        Location: {kwargs.get('location', 'Unknown')}<br>
                                        Device: {kwargs.get('device', 'Unknown')}
                                    </p>
                                </div>
                                
                                <!-- Footer -->
                                <div class="footer">
                                    <p class="footer-text">
                                        This is a security notification for {user_email}
                                    </p>
                                    <p class="footer-text">
                                        © {datetime.now().year} CineBrain, Inc.
                                    </p>
                                </div>
                            </div>
                        </td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Password Changed Successfully

Hi {user_name},

Your CineBrain account password was successfully changed.

Change details:
- Time: {change_time}
- IP: {kwargs.get('ip_address', 'Unknown')}
- Location: {kwargs.get('location', 'Unknown')}
- Device: {kwargs.get('device', 'Unknown')}

If you didn't make this change, secure your account immediately:
{FRONTEND_URL}/security/recover

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text
    
    def _get_generic_template(self, base_css: str, **kwargs) -> tuple:
        """Generic email template"""
        subject = kwargs.get('subject', 'CineBrain')
        content = kwargs.get('content', '')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>{subject}</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header">
                        <h1 class="header-logo">🎬 CineBrain</h1>
                    </div>
                    <div class="content">{content}</div>
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

# Initialize email service (will be done in init_auth)
email_service = None

def init_auth(flask_app, database, user_model):
    """Initialize auth module with Flask app and models"""
    global app, db, User, mail, serializer, email_service, redis_client
    
    app = flask_app
    db = database
    User = user_model
    
    # Initialize Redis
    redis_client = init_redis()
    
    # Initialize professional email service with Gmail SMTP
    gmail_username = os.environ.get('GMAIL_USERNAME', 'projects.srinath@gmail.com')
    gmail_password = os.environ.get('GMAIL_APP_PASSWORD', 'wuus nsow nbee xewv')
    
    email_service = ProfessionalEmailService(gmail_username, gmail_password)
    
    # Initialize token serializer
    serializer = URLSafeTimedSerializer(app.secret_key)
    
    logger.info("✅ Auth module initialized with Gmail SMTP and Redis")

# Rate limiting with Redis
def check_rate_limit(identifier: str, max_requests: int = 5, window: int = 300) -> bool:
    """Check if rate limit is exceeded using Redis"""
    if not redis_client:
        # Fallback to always allow if Redis is not available
        return True
    
    try:
        key = f"rate_limit:{identifier}"
        
        # Use Redis pipeline for atomic operations
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
        # Allow request if Redis fails
        return True

# Token generation and verification with Redis caching
def generate_reset_token(email):
    """Generate a secure password reset token and cache in Redis"""
    token = serializer.dumps(email, salt=PASSWORD_RESET_SALT)
    
    # Cache token in Redis for quick validation
    if redis_client:
        try:
            redis_client.setex(
                f"reset_token:{token[:20]}",  # Use first 20 chars as key
                3600,  # 1 hour TTL
                email
            )
        except Exception as e:
            logger.error(f"Failed to cache token in Redis: {e}")
    
    return token

def verify_reset_token(token, expiration=3600):
    """Verify password reset token with Redis cache"""
    # Quick check in Redis first
    if redis_client:
        try:
            cached_email = redis_client.get(f"reset_token:{token[:20]}")
            if cached_email:
                # Still verify the actual token
                email = serializer.loads(token, salt=PASSWORD_RESET_SALT, max_age=expiration)
                if email == cached_email:
                    return email
        except Exception as e:
            logger.error(f"Redis token verification error: {e}")
    
    # Fallback to regular verification
    try:
        email = serializer.loads(token, salt=PASSWORD_RESET_SALT, max_age=expiration)
        return email
    except SignatureExpired:
        return None
    except BadTimeSignature:
        return None

def validate_password(password):
    """Validate password strength"""
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
    """Get request information for security emails"""
    ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
    if ip_address:
        ip_address = ip_address.split(',')[0].strip()
    
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    # Simple device detection
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
    
    # Simple browser detection
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
    
    # For location, we'll use a placeholder (you'd need GeoIP service for real data)
    location = "Unknown location"
    
    return ip_address, location, device

# Authentication routes
@auth_bp.route('/api/auth/forgot-password', methods=['POST', 'OPTIONS'])
def forgot_password():
    """Request password reset with professional email"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        # Validate email format
        if not email or not EMAIL_REGEX.match(email):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        # Check rate limiting with Redis
        if not check_rate_limit(f"forgot_password:{email}", max_requests=3, window=600):
            return jsonify({
                'error': 'Too many password reset requests. Please try again in 10 minutes.'
            }), 429
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        
        # Always return success to prevent email enumeration
        if user:
            # Generate reset token
            token = generate_reset_token(email)
            reset_url = f"{FRONTEND_URL}/auth/reset-password.html?token={token}"
            
            # Get professional email template
            html_content, text_content = email_service.get_professional_template(
                'password_reset',
                reset_url=reset_url,
                user_name=user.username,
                user_email=email
            )
            
            # Queue email for sending
            email_service.queue_email(
                to=email,
                subject="Reset your password - CineBrain",
                html=html_content,
                text=text_content,
                priority='high'
            )
            
            logger.info(f"Password reset requested for {email}")
        
        return jsonify({
            'success': True,
            'message': 'If an account exists with this email, you will receive password reset instructions shortly.'
        }), 200
        
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({'error': 'Failed to process password reset request'}), 500

@auth_bp.route('/api/auth/reset-password', methods=['POST', 'OPTIONS'])
def reset_password():
    """Reset password with token"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        token = data.get('token', '').strip()
        new_password = data.get('password', '')
        confirm_password = data.get('confirmPassword', '')
        
        # Validate input
        if not token:
            return jsonify({'error': 'Reset token is required'}), 400
        
        if not new_password or not confirm_password:
            return jsonify({'error': 'Password and confirmation are required'}), 400
        
        if new_password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400
        
        # Validate password strength
        is_valid, message = validate_password(new_password)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Verify token
        email = verify_reset_token(token)
        if not email:
            return jsonify({'error': 'Invalid or expired reset token'}), 400
        
        # Find user
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Update password
        user.password_hash = generate_password_hash(new_password)
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        # Clear token from Redis cache
        if redis_client:
            try:
                redis_client.delete(f"reset_token:{token[:20]}")
            except:
                pass
        
        # Get request info for security email
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
        
        # Generate a new auth token for immediate login
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
    """Verify if a reset token is valid"""
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
    """Check auth service health including Redis"""
    try:
        # Test database connection
        if User:
            User.query.limit(1).first()
        
        # Test Redis connection
        redis_status = 'not_configured'
        if redis_client:
            try:
                redis_client.ping()
                redis_status = 'connected'
                
                # Get Redis stats
                info = redis_client.info()
                redis_stats = {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory_human': info.get('used_memory_human', 'N/A'),
                    'total_connections_received': info.get('total_connections_received', 0)
                }
            except:
                redis_status = 'disconnected'
                redis_stats = {}
        else:
            redis_stats = {}
        
        # Check email service
        email_configured = email_service is not None
        
        # Get email queue size from Redis
        queue_size = 0
        if redis_client:
            try:
                queue_size = redis_client.llen('email_queue')
            except:
                pass
        
        return jsonify({
            'status': 'healthy',
            'service': 'authentication',
            'email_service': 'Gmail SMTP',
            'email_configured': email_configured,
            'email_queue_size': queue_size,
            'redis_status': redis_status,
            'redis_stats': redis_stats,
            'frontend_url': FRONTEND_URL,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'authentication',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# CORS headers for all responses
@auth_bp.after_request
def after_request(response):
    """Add CORS headers to responses"""
    origin = request.headers.get('Origin')
    allowed_origins = [FRONTEND_URL, 'http://localhost:3000', 'http://localhost:5173']
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response

# Authentication decorator
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
            
            # Update last active
            current_user.last_active = datetime.utcnow()
            db.session.commit()
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

# Export everything needed
__all__ = [
    'auth_bp',
    'init_auth',
    'require_auth',
    'generate_reset_token',
    'verify_reset_token',
    'validate_password'
]