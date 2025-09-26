#backend/services/auth.py
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

class ProfessionalEmailService:
    def __init__(self, username, password):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.username = username
        self.password = password
        self.from_email = "noreply@cinebrain.com"
        self.from_name = "CineBrain"
        self.reply_to = "support@cinebrain.com"
        self.redis_client = redis_client
        self.start_email_worker()
    
    def start_email_worker(self):
        def worker():
            while True:
                try:
                    if self.redis_client:
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
        
        for i in range(3):
            thread = threading.Thread(target=worker, daemon=True, name=f"EmailWorker-{i}")
            thread.start()
            logger.info(f"Started email worker thread {i}")
    
    def _send_email_smtp(self, email_data: Dict):
        max_retries = 3
        retry_count = email_data.get('retry_count', 0)
        
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = formataddr((self.from_name, self.username))
            msg['To'] = email_data['to']
            msg['Subject'] = email_data['subject']
            msg['Reply-To'] = self.reply_to
            msg['Date'] = formatdate(localtime=True)
            msg['Message-ID'] = f"<{email_data.get('id', uuid.uuid4())}@cinebrain.com>"
            msg['X-Priority'] = '1' if email_data.get('priority') == 'high' else '3'
            msg['X-Mailer'] = 'CineBrain-Mailer/3.0'
            msg['X-Entity-Ref-ID'] = str(uuid.uuid4())
            msg['List-Unsubscribe'] = f'<mailto:unsubscribe@cinebrain.com?subject=Unsubscribe>'
            msg['List-Unsubscribe-Post'] = 'List-Unsubscribe=One-Click'
            msg['Precedence'] = 'bulk'
            msg['Auto-Submitted'] = 'auto-generated'
            msg['X-Auto-Response-Suppress'] = 'All'
            msg['X-Campaign-Id'] = 'password-reset' if 'reset' in email_data['subject'].lower() else 'transactional'
            
            text_part = MIMEText(email_data['text'], 'plain', 'utf-8')
            html_part = MIMEText(email_data['html'], 'html', 'utf-8')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"✅ Email sent successfully to {email_data['to']} - Subject: {email_data['subject']}")
            
            if self.redis_client:
                self.redis_client.setex(
                    f"email_sent:{email_data.get('id', 'unknown')}",
                    86400,
                    json.dumps({
                        'status': 'sent',
                        'timestamp': datetime.utcnow().isoformat(),
                        'to': email_data['to']
                    })
                )
            
        except Exception as e:
            logger.error(f"❌ SMTP Error sending to {email_data['to']}: {e}")
            
            if retry_count < max_retries:
                retry_count += 1
                email_data['retry_count'] = retry_count
                retry_delay = 5 * (2 ** retry_count)
                
                logger.info(f"🔄 Retrying email to {email_data['to']} in {retry_delay} seconds (attempt {retry_count}/{max_retries})")
                
                if self.redis_client:
                    threading.Timer(
                        retry_delay,
                        lambda: self.redis_client.rpush('email_queue', json.dumps(email_data))
                    ).start()
            else:
                logger.error(f"❌ Failed to send email after {max_retries} attempts to {email_data['to']}")
                
                if self.redis_client:
                    self.redis_client.setex(
                        f"email_failed:{email_data.get('id', 'unknown')}",
                        86400,
                        json.dumps({
                            'status': 'failed',
                            'error': str(e),
                            'timestamp': datetime.utcnow().isoformat(),
                            'to': email_data['to']
                        })
                    )
    
    def queue_email(self, to: str, subject: str, html: str, text: str, priority: str = 'normal'):
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
                if priority == 'high':
                    self.redis_client.lpush('email_queue', json.dumps(email_data))
                else:
                    self.redis_client.rpush('email_queue', json.dumps(email_data))
                
                self.redis_client.setex(
                    f"email_queued:{email_id}",
                    3600,
                    json.dumps({
                        'status': 'queued',
                        'timestamp': datetime.utcnow().isoformat(),
                        'to': to,
                        'subject': subject
                    })
                )
                
                logger.info(f"📧 Email queued (Redis) for {to} - ID: {email_id}")
            else:
                logger.warning("Redis not available, sending email directly")
                threading.Thread(
                    target=self._send_email_smtp,
                    args=(email_data,),
                    daemon=True
                ).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue email: {e}")
            threading.Thread(
                target=self._send_email_smtp,
                args=(email_data,),
                daemon=True
            ).start()
            return True
    
    def get_email_status(self, email_id: str) -> Dict:
        if not self.redis_client:
            return {'status': 'unknown', 'id': email_id}
        
        try:
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
        base_css = """
        <style type="text/css">
            @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Inter:wght@300;400;500;600;700&display=swap');
            
            :root {
                --cinebrain-primary: #113CCF;
                --cinebrain-primary-light: #1E4FE5;
                --cinebrain-accent: #1E4FE5;
                --cinebrain-gradient: linear-gradient(135deg, #113CCF 0%, #1E4FE5 50%, #1E4FE5 100%);
                --text-primary: #1a1a1a;
                --text-secondary: #666666;
                --text-muted: #999999;
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --bg-accent: #113CCF;
                --border-light: #e8eaed;
                --shadow-light: 0 1px 3px rgba(0,0,0,0.1);
                --shadow-medium: 0 4px 12px rgba(17,60,207,0.15);
                --shadow-heavy: 0 8px 32px rgba(17,60,207,0.2);
                --radius-small: 8px;
                --radius-medium: 12px;
                --radius-large: 16px;
                --spacing-xs: 4px;
                --spacing-sm: 8px;
                --spacing-md: 16px;
                --spacing-lg: 24px;
                --spacing-xl: 32px;
                --spacing-xxl: 48px;
            }
            
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            body, table, td, a {
                -webkit-text-size-adjust: 100%;
                -ms-text-size-adjust: 100%;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            table, td {
                mso-table-lspace: 0pt;
                mso-table-rspace: 0pt;
                border-collapse: collapse;
            }
            
            img {
                -ms-interpolation-mode: bicubic;
                border: 0;
                outline: none;
                text-decoration: none;
                max-width: 100%;
                height: auto;
            }
            
            body {
                margin: 0 !important;
                padding: 0 !important;
                width: 100% !important;
                min-width: 100% !important;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                font-size: 16px;
                line-height: 1.6;
                color: var(--text-primary);
                background: var(--bg-secondary);
            }
            
            .email-wrapper {
                width: 100%;
                background: var(--bg-secondary);
                padding: var(--spacing-xl) var(--spacing-md);
                min-height: 100vh;
            }
            
            .email-container {
                max-width: 600px;
                margin: 0 auto;
                background: var(--bg-primary);
                border-radius: var(--radius-large);
                box-shadow: var(--shadow-heavy);
                overflow: hidden;
                border: 1px solid var(--border-light);
            }
            
            .header {
                background: var(--cinebrain-gradient);
                padding: var(--spacing-xxl) var(--spacing-xl);
                text-align: center;
                position: relative;
                overflow: hidden;
            }
            
            .header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.05)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
                opacity: 0.6;
            }
            
            .brand-container {
                position: relative;
                z-index: 2;
            }
            
            .brand-logo {
                font-family: 'Bangers', cursive;
                font-size: clamp(28px, 6vw, 42px);
                font-weight: 400;
                letter-spacing: 1px;
                color: #ffffff;
                text-shadow: 0 2px 10px rgba(0,0,0,0.3);
                margin: 0;
                line-height: 1;
                display: inline-block;
                transform: perspective(500px) rotateX(5deg);
            }
            
            .brand-tagline {
                font-family: 'Inter', sans-serif;
                font-size: clamp(11px, 2vw, 14px);
                font-weight: 500;
                letter-spacing: 0.5px;
                color: rgba(255,255,255,0.95);
                margin: var(--spacing-sm) 0 0;
                line-height: 1.2;
                opacity: 0.9;
            }
            
            .content {
                padding: var(--spacing-xxl) var(--spacing-xl);
                background: var(--bg-primary);
                position: relative;
            }
            
            .content-header {
                text-align: center;
                margin-bottom: var(--spacing-xl);
            }
            
            .content-title {
                font-family: 'Inter', sans-serif;
                font-size: clamp(24px, 5vw, 32px);
                font-weight: 600;
                color: var(--text-primary);
                margin: 0 0 var(--spacing-md);
                line-height: 1.2;
            }
            
            .content-subtitle {
                font-size: clamp(14px, 3vw, 18px);
                font-weight: 400;
                color: var(--text-secondary);
                margin: 0;
                line-height: 1.4;
            }
            
            .content-body {
                font-size: clamp(14px, 3vw, 16px);
                line-height: 1.7;
                color: var(--text-primary);
                margin-bottom: var(--spacing-lg);
            }
            
            .content-body p {
                margin: 0 0 var(--spacing-md);
                color: var(--text-primary);
            }
            
            .content-body p:last-child {
                margin-bottom: 0;
            }
            
            .btn-container {
                text-align: center;
                margin: var(--spacing-xl) 0;
            }
            
            .btn {
                display: inline-block;
                font-family: 'Inter', sans-serif;
                font-size: clamp(14px, 3vw, 16px);
                font-weight: 600;
                text-decoration: none !important;
                text-align: center;
                padding: var(--spacing-md) var(--spacing-xl);
                border-radius: 50px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
                border: none;
                outline: none;
                letter-spacing: 0.3px;
                position: relative;
                overflow: hidden;
                min-width: 200px;
            }
            
            .btn-primary {
                background: var(--cinebrain-gradient);
                color: #ffffff !important;
                box-shadow: var(--shadow-medium);
                transform: translateY(0);
            }
            
            .btn-primary:hover {
                box-shadow: var(--shadow-heavy);
                transform: translateY(-2px);
            }
            
            .btn-primary::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }
            
            .btn-primary:hover::before {
                left: 100%;
            }
            
            .alert {
                padding: var(--spacing-md) var(--spacing-lg);
                border-radius: var(--radius-medium);
                margin: var(--spacing-lg) 0;
                font-size: clamp(13px, 3vw, 14px);
                line-height: 1.5;
                border-left: 4px solid;
                position: relative;
                background: rgba(255,255,255,0.5);
                backdrop-filter: blur(10px);
            }
            
            .alert-info {
                background: linear-gradient(135deg, rgba(17,60,207,0.1) 0%, rgba(30,79,229,0.05) 100%);
                border-left-color: var(--cinebrain-primary);
                color: var(--cinebrain-primary);
            }
            
            .alert-success {
                background: linear-gradient(135deg, rgba(34,197,94,0.1) 0%, rgba(21,128,61,0.05) 100%);
                border-left-color: #22c55e;
                color: #15803d;
            }
            
            .alert-warning {
                background: linear-gradient(135deg, rgba(245,158,11,0.1) 0%, rgba(217,119,6,0.05) 100%);
                border-left-color: #f59e0b;
                color: #d97706;
            }
            
            .alert-error {
                background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(185,28,28,0.05) 100%);
                border-left-color: #ef4444;
                color: #b91c1c;
            }
            
            .info-box {
                background: linear-gradient(135deg, var(--bg-secondary) 0%, rgba(248,249,250,0.8) 100%);
                border: 1px solid var(--border-light);
                border-radius: var(--radius-medium);
                padding: var(--spacing-lg);
                margin: var(--spacing-lg) 0;
                backdrop-filter: blur(5px);
            }
            
            .info-box-title {
                font-weight: 600;
                color: var(--text-primary);
                margin: 0 0 var(--spacing-sm);
                font-size: clamp(13px, 3vw, 14px);
            }
            
            .code-block {
                background: linear-gradient(135deg, #f1f3f4 0%, #e8eaed 100%);
                border: 1px solid var(--border-light);
                border-radius: var(--radius-small);
                padding: var(--spacing-md);
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Courier New', monospace;
                font-size: clamp(11px, 2.5vw, 13px);
                color: var(--text-primary);
                word-break: break-all;
                margin: var(--spacing-sm) 0;
                line-height: 1.4;
                overflow-x: auto;
            }
            
            .divider {
                height: 1px;
                background: linear-gradient(90deg, transparent, var(--border-light), transparent);
                margin: var(--spacing-xl) 0;
                border: none;
            }
            
            .footer {
                background: linear-gradient(135deg, var(--bg-secondary) 0%, #f1f3f4 100%);
                padding: var(--spacing-xl);
                text-align: center;
                border-top: 1px solid var(--border-light);
                position: relative;
            }
            
            .footer-content {
                max-width: 480px;
                margin: 0 auto;
            }
            
            .footer-links {
                margin: var(--spacing-md) 0 var(--spacing-lg);
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: var(--spacing-lg);
            }
            
            .footer-link {
                color: var(--cinebrain-primary) !important;
                text-decoration: none;
                font-size: clamp(12px, 2.5vw, 14px);
                font-weight: 500;
                transition: all 0.3s ease;
                padding: var(--spacing-xs) var(--spacing-sm);
                border-radius: var(--radius-small);
            }
            
            .footer-link:hover {
                background: rgba(17,60,207,0.1);
                transform: translateY(-1px);
            }
            
            .footer-text {
                font-size: clamp(11px, 2vw, 12px);
                color: var(--text-muted);
                margin: var(--spacing-sm) 0;
                line-height: 1.5;
            }
            
            .security-info {
                background: linear-gradient(135deg, rgba(17,60,207,0.05) 0%, rgba(30,79,229,0.02) 100%);
                border: 1px solid rgba(17,60,207,0.2);
                border-radius: var(--radius-medium);
                padding: var(--spacing-md);
                margin: var(--spacing-lg) 0;
                font-size: clamp(11px, 2.5vw, 12px);
                color: var(--text-secondary);
                line-height: 1.4;
            }
            
            .security-info-title {
                font-weight: 600;
                color: var(--cinebrain-primary);
                margin-bottom: var(--spacing-xs);
            }
            
            @media screen and (max-width: 640px) {
                .email-wrapper {
                    padding: var(--spacing-md) var(--spacing-sm) !important;
                }
                
                .email-container {
                    border-radius: var(--radius-medium) !important;
                    margin: 0 !important;
                }
                
                .header {
                    padding: var(--spacing-xl) var(--spacing-lg) !important;
                }
                
                .content {
                    padding: var(--spacing-xl) var(--spacing-lg) !important;
                }
                
                .footer {
                    padding: var(--spacing-lg) !important;
                }
                
                .footer-links {
                    flex-direction: column;
                    gap: var(--spacing-sm) !important;
                }
                
                .btn {
                    width: 100% !important;
                    min-width: auto !important;
                }
                
                .brand-logo {
                    font-size: 32px !important;
                }
                
                .brand-tagline {
                    font-size: 12px !important;
                }
                
                .content-title {
                    font-size: 24px !important;
                }
                
                .content-subtitle {
                    font-size: 16px !important;
                }
                
                .alert, .info-box {
                    margin: var(--spacing-md) 0 !important;
                    padding: var(--spacing-md) !important;
                }
            }
            
            @media screen and (max-width: 480px) {
                .email-wrapper {
                    padding: var(--spacing-sm) !important;
                }
                
                .header {
                    padding: var(--spacing-lg) var(--spacing-md) !important;
                }
                
                .content {
                    padding: var(--spacing-lg) var(--spacing-md) !important;
                }
                
                .footer {
                    padding: var(--spacing-md) !important;
                }
                
                .brand-logo {
                    font-size: 28px !important;
                }
                
                .brand-tagline {
                    font-size: 11px !important;
                }
                
                .content-title {
                    font-size: 20px !important;
                }
            }
            
            @media (prefers-color-scheme: dark) {
                :root {
                    --text-primary: #e8eaed;
                    --text-secondary: #9aa0a6;
                    --text-muted: #80868b;
                    --bg-primary: #1f2937;
                    --bg-secondary: #111827;
                    --border-light: #374151;
                }
                
                body {
                    background: var(--bg-secondary) !important;
                }
                
                .email-container {
                    background: var(--bg-primary) !important;
                    border-color: var(--border-light) !important;
                }
                
                .content {
                    background: var(--bg-primary) !important;
                }
                
                .footer {
                    background: var(--bg-secondary) !important;
                    border-color: var(--border-light) !important;
                }
                
                .info-box {
                    background: var(--bg-secondary) !important;
                    border-color: var(--border-light) !important;
                }
                
                .code-block {
                    background: var(--bg-secondary) !important;
                    border-color: var(--border-light) !important;
                }
                
                .security-info {
                    background: rgba(17,60,207,0.1) !important;
                    border-color: rgba(17,60,207,0.3) !important;
                }
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
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="format-detection" content="telephone=no">
            <meta name="format-detection" content="date=no">
            <meta name="format-detection" content="address=no">
            <meta name="format-detection" content="email=no">
            <title>Reset your password - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                    <tr>
                        <td align="center">
                            <div class="email-container">
                                <div class="header">
                                    <div class="brand-container">
                                        <div class="brand-logo">CineBrain</div>
                                        <div class="brand-tagline">The Mind Behind Your Next Favorite</div>
                                    </div>
                                </div>
                                
                                <div class="content">
                                    <div class="content-header">
                                        <h1 class="content-title">Reset your password</h1>
                                        <p class="content-subtitle">Secure your account with a new password</p>
                                    </div>
                                    
                                    <div class="content-body">
                                        <p>Hi {user_name},</p>
                                        
                                        <p>We received a request to reset your CineBrain account password. Click the button below to create a new password and get back to discovering amazing content:</p>
                                    </div>
                                    
                                    <div class="btn-container">
                                        <a href="{reset_url}" class="btn btn-primary">Reset Password</a>
                                    </div>
                                    
                                    <div class="info-box">
                                        <div class="info-box-title">Can't click the button?</div>
                                        <p style="margin: 0; font-size: 13px; color: var(--text-secondary);">
                                            Copy and paste this link into your browser:
                                        </p>
                                        <div class="code-block">{reset_url}</div>
                                    </div>
                                    
                                    <div class="alert alert-warning">
                                        <strong>⏰ This link expires in 1 hour</strong><br>
                                        For security reasons, this password reset link will expire soon.
                                    </div>
                                    
                                    <hr class="divider">
                                    
                                    <div class="security-info">
                                        <div class="security-info-title">Security Notice</div>
                                        If you didn't request this password reset, you can safely ignore this email. Your password won't be changed unless you click the link above.
                                    </div>
                                </div>
                                
                                <div class="footer">
                                    <div class="footer-content">
                                        <div class="footer-links">
                                            <a href="{FRONTEND_URL}/privacy" class="footer-link">Privacy Policy</a>
                                            <a href="{FRONTEND_URL}/terms" class="footer-link">Terms of Service</a>
                                            <a href="{FRONTEND_URL}/help" class="footer-link">Help Center</a>
                                            <a href="{FRONTEND_URL}/contact" class="footer-link">Contact Us</a>
                                        </div>
                                        
                                        <p class="footer-text">
                                            © {datetime.now().year} CineBrain, Inc. All rights reserved.
                                        </p>
                                        <p class="footer-text">
                                            This email was sent to <strong>{user_email}</strong>
                                        </p>
                                    </div>
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
        change_time = datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')
        ip_address = kwargs.get('ip_address', 'Unknown')
        location = kwargs.get('location', 'Unknown')
        device = kwargs.get('device', 'Unknown')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <title>Password changed - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                    <tr>
                        <td align="center">
                            <div class="email-container">
                                <div class="header" style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 50%, #15803d 100%);">
                                    <div class="brand-container">
                                        <div class="brand-logo">✅ Password Changed</div>
                                        <div class="brand-tagline">Your account is now secured</div>
                                    </div>
                                </div>
                                
                                <div class="content">
                                    <div class="content-header">
                                        <h1 class="content-title">Password successfully changed</h1>
                                        <p class="content-subtitle">Your CineBrain account is now secured</p>
                                    </div>
                                    
                                    <div class="content-body">
                                        <p>Hi {user_name},</p>
                                        
                                        <p>Your CineBrain account password was successfully changed. You can now sign in with your new password and continue enjoying personalized content recommendations.</p>
                                    </div>
                                    
                                    <div class="alert alert-success">
                                        <strong>✓ Your account is secured</strong><br>
                                        You can now sign in with your new password.
                                    </div>
                                    
                                    <div class="btn-container">
                                        <a href="{FRONTEND_URL}/login" class="btn btn-primary" style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);">Sign in to CineBrain</a>
                                    </div>
                                    
                                    <div class="alert alert-error">
                                        <strong>⚠️ Didn't make this change?</strong><br>
                                        If you didn't change your password, 
                                        <a href="{FRONTEND_URL}/security/recover" style="color: #ef4444; font-weight: bold;">secure your account immediately</a>
                                    </div>
                                    
                                    <hr class="divider">
                                    
                                    <div class="security-info">
                                        <div class="security-info-title">Change Details</div>
                                        <strong>Time:</strong> {change_time}<br>
                                        <strong>IP Address:</strong> {ip_address}<br>
                                        <strong>Location:</strong> {location}<br>
                                        <strong>Device:</strong> {device}
                                    </div>
                                </div>
                                
                                <div class="footer">
                                    <div class="footer-content">
                                        <div class="footer-links">
                                            <a href="{FRONTEND_URL}/security" class="footer-link">Security Settings</a>
                                            <a href="{FRONTEND_URL}/help" class="footer-link">Help Center</a>
                                            <a href="{FRONTEND_URL}/contact" class="footer-link">Contact Support</a>
                                        </div>
                                        
                                        <p class="footer-text">
                                            This is a security notification for <strong>{user_email}</strong>
                                        </p>
                                        <p class="footer-text">
                                            © {datetime.now().year} CineBrain, Inc.
                                        </p>
                                    </div>
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
Password Changed Successfully - CineBrain

Hi {user_name},

Your CineBrain account password was successfully changed.

Change details:
- Time: {change_time}
- IP: {ip_address}
- Location: {location}
- Device: {device}

If you didn't make this change, secure your account immediately:
{FRONTEND_URL}/security/recover

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text
    
    def _get_generic_template(self, base_css: str, **kwargs) -> tuple:
        subject = kwargs.get('subject', 'CineBrain')
        content = kwargs.get('content', '')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{subject}</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                    <tr>
                        <td align="center">
                            <div class="email-container">
                                <div class="header">
                                    <div class="brand-container">
                                        <div class="brand-logo">CineBrain</div>
                                        <div class="brand-tagline">The Mind Behind Your Next Favorite</div>
                                    </div>
                                </div>
                                <div class="content">
                                    <div class="content-body">{content}</div>
                                </div>
                                <div class="footer">
                                    <div class="footer-content">
                                        <p class="footer-text">© {datetime.now().year} CineBrain, Inc.</p>
                                    </div>
                                </div>
                            </div>
                        </td>
                    </tr>
                </table>
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
    
    gmail_username = os.environ.get('GMAIL_USERNAME', 'projects.srinath@gmail.com')
    gmail_password = os.environ.get('GMAIL_APP_PASSWORD', 'wuus nsow nbee xewv')
    
    email_service = ProfessionalEmailService(gmail_username, gmail_password)
    
    serializer = URLSafeTimedSerializer(app.secret_key)
    
    logger.info("✅ Auth module initialized with Gmail SMTP and Redis")

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
            priority='high'
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
        if User:
            User.query.limit(1).first()
        
        redis_status = 'not_configured'
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
                redis_stats = {}
        else:
            redis_stats = {}
        
        email_configured = email_service is not None
        
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

@auth_bp.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    allowed_origins = [FRONTEND_URL, 'http://127.0.0.1:5500', 'http://127.0.0.1:5501']
    
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