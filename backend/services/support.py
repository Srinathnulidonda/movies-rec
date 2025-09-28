#backend/services/support.py
from flask import Blueprint, request, jsonify
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
import smtplib
import ssl
import uuid
import time
from typing import Dict, Optional, List
import hashlib
import json
import redis
from urllib.parse import urlparse
from sqlalchemy import func, desc, and_, or_
from collections import defaultdict
import enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

support_bp = Blueprint('support', __name__)

FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')
BACKEND_URL = os.environ.get('BACKEND_URL', 'https://cinebrain.onrender.com')
REDIS_URL = os.environ.get('REDIS_URL', 'redis://red-d3cdplidbo4c73e352eg:Fin34Hk4Hq42PYejhV4Tufncmi4Ym4H6@red-d3cdplidbo4c73e352eg:6379')

EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

app = None
db = None
User = None
cache = None
redis_client = None

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
    CONTACT = "contact"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    ACCOUNT_ISSUE = "account_issue"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_INQUIRY = "general_inquiry"

class FeedbackType(enum.Enum):
    GENERAL = "general"
    FEATURE_REQUEST = "feature_request"
    UI_UX = "ui_ux"
    PERFORMANCE = "performance"
    CONTENT_SUGGESTION = "content_suggestion"

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
        logger.info("Support service Redis connected successfully")
        return redis_client
    except Exception as e:
        logger.error(f"Support service Redis connection failed: {e}")
        return None

class SupportEmailService:
    def __init__(self, username, password):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.username = username
        self.password = password
        self.from_email = "support@cinebrain.com"
        self.from_name = "CineBrain Support"
        self.reply_to = "support@cinebrain.com"
        
        self.redis_client = redis_client
        self.start_email_worker()
    
    def start_email_worker(self):
        def worker():
            while True:
                try:
                    if self.redis_client:
                        email_json = self.redis_client.lpop('support_email_queue')
                        if email_json:
                            email_data = json.loads(email_json)
                            self._send_email_smtp(email_data)
                        else:
                            time.sleep(1)
                    else:
                        time.sleep(5)
                except Exception as e:
                    logger.error(f"Support email worker error: {e}")
                    time.sleep(5)
        
        for i in range(2):
            thread = threading.Thread(target=worker, daemon=True, name=f"SupportEmailWorker-{i}")
            thread.start()
            logger.info(f"Started support email worker thread {i}")
    
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
            msg['Message-ID'] = f"<{email_data.get('id', uuid.uuid4())}@support.cinebrain.com>"
            
            msg['X-Support-Category'] = email_data.get('category', 'general')
            msg['X-Ticket-Number'] = email_data.get('ticket_number', '')
            msg['X-Auto-Response-Suppress'] = 'All'
            msg['X-Priority'] = '1' if email_data.get('priority') == 'urgent' else '3'
            
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
            
            logger.info(f"✅ Support email sent to {email_data['to']} - {email_data['subject']}")
            
        except Exception as e:
            logger.error(f"❌ Support email error: {e}")
            if retry_count < max_retries:
                retry_count += 1
                email_data['retry_count'] = retry_count
                retry_delay = 5 * (2 ** retry_count)
                
                if self.redis_client:
                    threading.Timer(
                        retry_delay,
                        lambda: self.redis_client.rpush('support_email_queue', json.dumps(email_data))
                    ).start()
    
    def queue_email(self, to: str, subject: str, html: str, text: str, **kwargs):
        email_id = str(uuid.uuid4())
        email_data = {
            'id': email_id,
            'to': to,
            'subject': subject,
            'html': html,
            'text': text,
            'timestamp': datetime.utcnow().isoformat(),
            'retry_count': 0,
            **kwargs
        }
        
        try:
            if self.redis_client:
                if kwargs.get('priority') == 'urgent':
                    self.redis_client.lpush('support_email_queue', json.dumps(email_data))
                else:
                    self.redis_client.rpush('support_email_queue', json.dumps(email_data))
            else:
                threading.Thread(target=self._send_email_smtp, args=(email_data,), daemon=True).start()
            
            return True
        except Exception as e:
            logger.error(f"Failed to queue support email: {e}")
            return False
    
    def get_template(self, template_type: str, **kwargs) -> tuple:
        base_css = """
        <style type="text/css">
            body, table, td, a { -webkit-text-size-adjust: 100%; -ms-text-size-adjust: 100%; }
            table, td { mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
            img { -ms-interpolation-mode: bicubic; border: 0; outline: none; text-decoration: none; }
            
            body {
                margin: 0 !important; padding: 0 !important; width: 100% !important;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                -webkit-font-smoothing: antialiased; font-size: 14px; line-height: 1.6;
                color: #202124; background-color: #f8f9fa;
            }
            
            .email-wrapper { width: 100%; background-color: #f8f9fa; padding: 40px 20px; }
            .email-container { max-width: 600px; margin: 0 auto; background-color: #ffffff;
                border-radius: 8px; box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
                overflow: hidden; }
            
            .header { background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%); padding: 40px 48px; text-align: center; }
            .header-logo { font-size: 32px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;
                margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .header-tagline { font-size: 14px; color: rgba(255,255,255,0.95); margin: 8px 0 0 0; font-weight: 400; }
            
            .content { padding: 48px; background-color: #ffffff; }
            h1 { font-size: 24px; font-weight: 400; color: #202124; margin: 0 0 24px; line-height: 1.3; }
            h2 { font-size: 18px; font-weight: 500; color: #202124; margin: 24px 0 12px; }
            p { margin: 0 0 16px; color: #5f6368; font-size: 14px; line-height: 1.6; }
            
            .btn { display: inline-block; padding: 12px 32px; font-size: 14px; font-weight: 500;
                text-decoration: none !important; text-align: center; border-radius: 24px;
                margin: 24px 0; cursor: pointer; }
            .btn-primary { background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%);
                color: #ffffff !important; box-shadow: 0 4px 15px 0 rgba(26, 115, 232, 0.4); }
            
            .alert { padding: 16px; border-radius: 8px; margin: 24px 0; font-size: 14px; }
            .alert-info { background-color: #e8f0fe; border-left: 4px solid #1a73e8; color: #1967d2; }
            .alert-success { background-color: #e6f4ea; border-left: 4px solid #34a853; color: #188038; }
            .alert-warning { background-color: #fef7e0; border-left: 4px solid #fbbc04; color: #ea8600; }
            
            .ticket-info { background-color: #f8f9fa; border-radius: 8px; padding: 24px; margin: 24px 0;
                border: 1px solid #e8eaed; }
            .ticket-number { font-family: 'Monaco', 'Menlo', monospace; font-size: 16px; font-weight: bold;
                color: #1a73e8; background: #e8f0fe; padding: 8px 12px; border-radius: 4px; display: inline-block; }
            
            .footer { background-color: #f8f9fa; padding: 32px 48px; text-align: center; border-top: 1px solid #e8eaed; }
            .footer-text { font-size: 12px; color: #80868b; margin: 0 0 8px; line-height: 1.5; }
            .footer-link { color: #1a73e8 !important; text-decoration: none; font-size: 12px; margin: 0 12px; }
            
            @media screen and (max-width: 600px) {
                .email-wrapper { padding: 0 !important; }
                .email-container { width: 100% !important; border-radius: 0 !important; }
                .content, .footer { padding: 32px 24px !important; }
                .header { padding: 32px 24px !important; }
                h1 { font-size: 20px !important; }
                .btn { display: block !important; width: 100% !important; }
            }
        </style>
        """
        
        if template_type == 'ticket_created':
            return self._get_ticket_created_template(base_css, **kwargs)
        elif template_type == 'feedback_received':
            return self._get_feedback_received_template(base_css, **kwargs)
        elif template_type == 'ticket_response':
            return self._get_ticket_response_template(base_css, **kwargs)
        elif template_type == 'issue_report':
            return self._get_issue_report_template(base_css, **kwargs)
        else:
            return self._get_generic_template(base_css, **kwargs)
    
    def _get_ticket_created_template(self, base_css: str, **kwargs) -> tuple:
        ticket_number = kwargs.get('ticket_number', '')
        user_name = kwargs.get('user_name', 'there')
        subject = kwargs.get('subject', '')
        priority = kwargs.get('priority', 'normal')
        category = kwargs.get('category', 'General')
        
        priority_colors = {
            'low': '#34a853',
            'normal': '#1a73e8', 
            'high': '#fbbc04',
            'urgent': '#ea4335'
        }
        
        priority_color = priority_colors.get(priority, '#1a73e8')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Support Ticket Created - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                    <tr>
                        <td>
                            <div class="email-container">
                                <div class="header">
                                    <h1 class="header-logo">🎬 CineBrain Support</h1>
                                    <p class="header-tagline">We're here to help you</p>
                                </div>
                                
                                <div class="content">
                                    <h1>Your support ticket has been created</h1>
                                    
                                    <p>Hi {user_name},</p>
                                    
                                    <p>Thank you for contacting CineBrain support. We've received your request and created a support ticket for you.</p>
                                    
                                    <div class="ticket-info">
                                        <h2 style="margin-top: 0;">Ticket Details</h2>
                                        <p><strong>Ticket Number:</strong> <span class="ticket-number">#{ticket_number}</span></p>
                                        <p><strong>Subject:</strong> {subject}</p>
                                        <p><strong>Category:</strong> {category}</p>
                                        <p><strong>Priority:</strong> 
                                            <span style="color: {priority_color}; font-weight: bold; text-transform: uppercase;">{priority}</span>
                                        </p>
                                        <p><strong>Status:</strong> Open</p>
                                    </div>
                                    
                                    <div class="alert alert-info">
                                        <strong>📞 What happens next?</strong><br>
                                        Our support team will review your ticket and respond within 24 hours for normal priority tickets.
                                        You'll receive an email notification when we respond.
                                    </div>
                                    
                                    <center>
                                        <a href="{FRONTEND_URL}/support/ticket/{ticket_number}" class="btn btn-primary">
                                            View Ticket Status
                                        </a>
                                    </center>
                                    
                                    <div class="alert alert-warning">
                                        <strong>💡 Quick Tips</strong><br>
                                        • Reply to this email to add more information to your ticket<br>
                                        • Include your ticket number #{ticket_number} in any communications<br>
                                        • Check our <a href="{FRONTEND_URL}/support/faq" style="color: #ea8600;">FAQ section</a> for instant answers
                                    </div>
                                </div>
                                
                                <div class="footer">
                                    <div class="footer-links">
                                        <a href="{FRONTEND_URL}/support" class="footer-link">Support Center</a>
                                        <a href="{FRONTEND_URL}/support/faq" class="footer-link">FAQ</a>
                                        <a href="{FRONTEND_URL}/contact" class="footer-link">Contact Us</a>
                                    </div>
                                    <p class="footer-text">
                                        © {datetime.now().year} CineBrain, Inc. All rights reserved.<br>
                                        Ticket #{ticket_number} • {datetime.now().strftime('%B %d, %Y')}
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
Support Ticket Created - #{ticket_number}

Hi {user_name},

Thank you for contacting CineBrain support. We've created ticket #{ticket_number} for your request.

Ticket Details:
- Subject: {subject}
- Category: {category}
- Priority: {priority}
- Status: Open

Our team will respond within 24 hours. You can track your ticket at:
{FRONTEND_URL}/support/ticket/{ticket_number}

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text
    
    def _get_feedback_received_template(self, base_css: str, **kwargs) -> tuple:
        user_name = kwargs.get('user_name', 'there')
        feedback_type = kwargs.get('feedback_type', 'general')
        subject = kwargs.get('subject', '')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Feedback Received - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header" style="background: linear-gradient(135deg, #34a853 0%, #0d8043 100%);">
                        <h1 class="header-logo">💝 Thank You!</h1>
                        <p class="header-tagline">Your feedback helps us improve</p>
                    </div>
                    
                    <div class="content">
                        <h1>We received your feedback</h1>
                        
                        <p>Hi {user_name},</p>
                        
                        <p>Thank you for taking the time to share your feedback with us. Your input is incredibly valuable and helps us make CineBrain better for everyone.</p>
                        
                        <div class="alert alert-success">
                            <strong>✅ Feedback received</strong><br>
                            <strong>Type:</strong> {feedback_type.replace('_', ' ').title()}<br>
                            <strong>Subject:</strong> {subject}
                        </div>
                        
                        <p>Our product team reviews all feedback carefully. While we can't respond to every submission individually, your suggestions directly influence our development roadmap.</p>
                        
                        <center>
                            <a href="{FRONTEND_URL}/support/feedback" class="btn btn-primary">
                                Share More Feedback
                            </a>
                        </center>
                        
                        <div class="alert alert-info">
                            <strong>🚀 Stay updated</strong><br>
                            Follow our <a href="{FRONTEND_URL}/changelog" style="color: #1967d2;">changelog</a> 
                            to see when your suggestions are implemented!
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p class="footer-text">
                            © {datetime.now().year} CineBrain, Inc. • Thank you for helping us improve!
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Feedback Received - Thank You!

Hi {user_name},

Thank you for your feedback about CineBrain. Your input helps us improve!

Feedback Details:
- Type: {feedback_type.replace('_', ' ').title()}
- Subject: {subject}

Our team reviews all feedback carefully. Check our changelog for updates!

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text
    
    def _get_ticket_response_template(self, base_css: str, **kwargs) -> tuple:
        ticket_number = kwargs.get('ticket_number', '')
        user_name = kwargs.get('user_name', 'there')
        staff_name = kwargs.get('staff_name', 'Support Team')
        response_message = kwargs.get('response_message', '')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>New Response - Ticket #{ticket_number}</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header">
                        <h1 class="header-logo">💬 New Response</h1>
                        <p class="header-tagline">Ticket #{ticket_number}</p>
                    </div>
                    
                    <div class="content">
                        <h1>New response to your support ticket</h1>
                        
                        <p>Hi {user_name},</p>
                        
                        <p><strong>{staff_name}</strong> has responded to your support ticket:</p>
                        
                        <div class="ticket-info">
                            <h2 style="margin-top: 0;">Response from {staff_name}</h2>
                            <div style="background: white; padding: 20px; border-left: 4px solid #1a73e8; margin: 16px 0;">
                                {response_message}
                            </div>
                        </div>
                        
                        <center>
                            <a href="{FRONTEND_URL}/support/ticket/{ticket_number}" class="btn btn-primary">
                                View Full Conversation
                            </a>
                        </center>
                        
                        <div class="alert alert-info">
                            <strong>📧 Reply directly</strong><br>
                            You can reply to this email to continue the conversation, or visit the support portal to view your ticket.
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p class="footer-text">
                            Ticket #{ticket_number} • © {datetime.now().year} CineBrain Support
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""
New Response - Ticket #{ticket_number}

Hi {user_name},

{staff_name} has responded to your support ticket:

---
{response_message}
---

View full conversation: {FRONTEND_URL}/support/ticket/{ticket_number}

You can reply to this email to continue the conversation.

© {datetime.now().year} CineBrain Support
        """
        
        return html, text
    
    def _get_issue_report_template(self, base_css: str, **kwargs) -> tuple:
        user_name = kwargs.get('user_name', 'there')
        issue_type = kwargs.get('issue_type', 'bug report')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Issue Report Received - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header" style="background: linear-gradient(135deg, #ea4335 0%, #d33b27 100%);">
                        <h1 class="header-logo">🐛 Issue Reported</h1>
                        <p class="header-tagline">Thank you for helping us improve</p>
                    </div>
                    
                    <div class="content">
                        <h1>Issue report received</h1>
                        
                        <p>Hi {user_name},</p>
                        
                        <p>Thank you for reporting this {issue_type}. Your report helps us identify and fix issues to improve CineBrain for everyone.</p>
                        
                        <div class="alert alert-success">
                            <strong>✅ Report submitted successfully</strong><br>
                            Our development team has been notified and will investigate the issue.
                        </div>
                        
                        <div class="alert alert-info">
                            <strong>🔍 What happens next?</strong><br>
                            • Our team will reproduce and investigate the issue<br>
                            • Critical bugs are prioritized for immediate fixes<br>
                            • You'll be notified when the issue is resolved
                        </div>
                        
                        <center>
                            <a href="{FRONTEND_URL}/support/report-issue" class="btn btn-primary">
                                Report Another Issue
                            </a>
                        </center>
                    </div>
                    
                    <div class="footer">
                        <p class="footer-text">
                            © {datetime.now().year} CineBrain, Inc. • Thank you for helping us improve!
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Issue Report Received

Hi {user_name},

Thank you for reporting this {issue_type}. Our development team has been notified and will investigate.

What happens next:
- Our team will reproduce and investigate the issue
- Critical bugs are prioritized for immediate fixes  
- You'll be notified when the issue is resolved

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text
    
    def _get_generic_template(self, base_css: str, **kwargs) -> tuple:
        subject = kwargs.get('subject', 'CineBrain Support')
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
                        <h1 class="header-logo">🎬 CineBrain Support</h1>
                    </div>
                    <div class="content">{content}</div>
                    <div class="footer">
                        <p class="footer-text">© {datetime.now().year} CineBrain Support</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"{subject}\n\n{content}\n\n© {datetime.now().year} CineBrain Support"
        
        return html, text

class SupportService:
    def __init__(self, email_service):
        self.email_service = email_service
    
    def generate_ticket_number(self) -> str:
        import random
        import string
        
        date_str = datetime.now().strftime('%Y%m%d')
        random_str = ''.join(random.choices(string.digits, k=4))
        
        ticket_number = f"CB-{date_str}-{random_str}"
        
        while SupportTicket.query.filter_by(ticket_number=ticket_number).first():
            random_str = ''.join(random.choices(string.digits, k=4))
            ticket_number = f"CB-{date_str}-{random_str}"
        
        return ticket_number
    
    def calculate_sla_deadline(self, priority: TicketPriority) -> datetime:
        now = datetime.utcnow()
        
        sla_hours = {
            TicketPriority.URGENT: 4,
            TicketPriority.HIGH: 24,
            TicketPriority.NORMAL: 48,
            TicketPriority.LOW: 72
        }
        
        hours = sla_hours.get(priority, 48)
        return now + timedelta(hours=hours)
    
    def create_ticket(self, data: Dict) -> Dict:
        try:
            ticket_number = self.generate_ticket_number()
            
            priority = TicketPriority(data.get('priority', 'normal'))
            
            ticket = SupportTicket(
                ticket_number=ticket_number,
                subject=data['subject'],
                description=data['description'],
                user_id=data.get('user_id'),
                user_email=data['user_email'],
                user_name=data['user_name'],
                category_id=data['category_id'],
                ticket_type=TicketType(data.get('ticket_type', 'contact')),
                priority=priority,
                browser_info=data.get('browser_info'),
                device_info=data.get('device_info'),
                ip_address=data.get('ip_address'),
                user_agent=data.get('user_agent'),
                page_url=data.get('page_url'),
                sla_deadline=self.calculate_sla_deadline(priority)
            )
            
            db.session.add(ticket)
            db.session.flush()
            
            activity = TicketActivity(
                ticket_id=ticket.id,
                action='created',
                description=f'Ticket created by {data["user_name"]}',
                actor_type='user',
                actor_id=data.get('user_id'),
                actor_name=data['user_name']
            )
            db.session.add(activity)
            
            db.session.commit()
            
            category = SupportCategory.query.get(data['category_id'])
            html, text = self.email_service.get_template(
                'ticket_created',
                ticket_number=ticket_number,
                user_name=data['user_name'],
                subject=data['subject'],
                priority=priority.value,
                category=category.name if category else 'General'
            )
            
            self.email_service.queue_email(
                to=data['user_email'],
                subject=f"Support Ticket Created - #{ticket_number}",
                html=html,
                text=text,
                ticket_number=ticket_number,
                category='support',
                priority=priority.value
            )
            
            logger.info(f"Support ticket {ticket_number} created for {data['user_email']}")
            
            return {
                'success': True,
                'ticket_id': ticket.id,
                'ticket_number': ticket_number,
                'sla_deadline': ticket.sla_deadline.isoformat()
            }
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating support ticket: {e}")
            return {'success': False, 'error': str(e)}
    
    def submit_feedback(self, data: Dict) -> Dict:
        try:
            feedback = Feedback(
                subject=data['subject'],
                message=data['message'],
                user_id=data.get('user_id'),
                user_email=data['user_email'],
                user_name=data['user_name'],
                feedback_type=FeedbackType(data.get('feedback_type', 'general')),
                rating=data.get('rating'),
                page_url=data.get('page_url'),
                browser_info=data.get('browser_info'),
                ip_address=data.get('ip_address')
            )
            
            db.session.add(feedback)
            db.session.commit()
            
            html, text = self.email_service.get_template(
                'feedback_received',
                user_name=data['user_name'],
                feedback_type=data.get('feedback_type', 'general'),
                subject=data['subject']
            )
            
            self.email_service.queue_email(
                to=data['user_email'],
                subject="Feedback Received - Thank You!",
                html=html,
                text=text,
                category='feedback'
            )
            
            logger.info(f"Feedback submitted by {data['user_email']}")
            
            return {'success': True, 'feedback_id': feedback.id}
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error submitting feedback: {e}")
            return {'success': False, 'error': str(e)}

class RateLimitService:
    @staticmethod
    def check_rate_limit(identifier: str, max_requests: int = 5, window: int = 300) -> bool:
        if not redis_client:
            return True
        
        try:
            key = f"support_rate_limit:{identifier}"
            pipe = redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            results = pipe.execute()
            
            current_count = results[0]
            return current_count <= max_requests
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True

class CacheService:
    @staticmethod
    def get_faqs(category_id: int = None, featured_only: bool = False) -> List[Dict]:
        cache_key = f"faqs:{category_id}:{featured_only}"
        
        if cache:
            cached_data = cache.get(cache_key)
            if cached_data:
                return cached_data
        
        query = FAQ.query.filter_by(is_published=True)
        
        if category_id:
            query = query.filter_by(category_id=category_id)
        
        if featured_only:
            query = query.filter_by(is_featured=True)
        
        faqs = query.order_by(FAQ.sort_order, FAQ.created_at.desc()).all()
        
        result = []
        for faq in faqs:
            category = SupportCategory.query.get(faq.category_id)
            result.append({
                'id': faq.id,
                'question': faq.question,
                'answer': faq.answer,
                'category': {
                    'id': category.id,
                    'name': category.name,
                    'icon': category.icon
                } if category else None,
                'tags': json.loads(faq.tags or '[]'),
                'view_count': faq.view_count,
                'helpful_count': faq.helpful_count,
                'is_featured': faq.is_featured
            })
        
        if cache:
            cache.set(cache_key, result, timeout=1800)
        
        return result

email_service = None
support_service = None
SupportCategory = None
SupportTicket = None
SupportResponse = None
FAQ = None
Feedback = None
TicketActivity = None

def init_support(flask_app, database, models, services):
    global app, db, User, cache, email_service, support_service, redis_client
    global SupportCategory, SupportTicket, SupportResponse, FAQ, Feedback, TicketActivity
    
    app = flask_app
    db = database
    User = models['User']
    cache = services.get('cache')
    
    redis_client = init_redis()
    
    gmail_username = os.environ.get('GMAIL_USERNAME', 'projects.srinath@gmail.com')
    gmail_password = os.environ.get('GMAIL_APP_PASSWORD', 'wuus nsow nbee xewv')
    
    email_service = SupportEmailService(gmail_username, gmail_password)
    support_service = SupportService(email_service)
    
    class SupportCategory(database.Model):
        __tablename__ = 'support_categories'
        
        id = database.Column(database.Integer, primary_key=True)
        name = database.Column(database.String(100), nullable=False, unique=True)
        description = database.Column(database.Text)
        icon = database.Column(database.String(50))
        sort_order = database.Column(database.Integer, default=0)
        is_active = database.Column(database.Boolean, default=True)
        created_at = database.Column(database.DateTime, default=datetime.utcnow)
        
        tickets = database.relationship('SupportTicket', backref='category', lazy='dynamic')
        faqs = database.relationship('FAQ', backref='category', lazy='dynamic')

    class SupportTicket(database.Model):
        __tablename__ = 'support_tickets'
        
        id = database.Column(database.Integer, primary_key=True)
        ticket_number = database.Column(database.String(20), unique=True, nullable=False)
        subject = database.Column(database.String(255), nullable=False)
        description = database.Column(database.Text, nullable=False)
        
        user_id = database.Column(database.Integer, database.ForeignKey('user.id'), nullable=True)
        user_email = database.Column(database.String(255), nullable=False)
        user_name = database.Column(database.String(255), nullable=False)
        
        category_id = database.Column(database.Integer, database.ForeignKey('support_categories.id'), nullable=False)
        ticket_type = database.Column(database.Enum(TicketType), nullable=False)
        priority = database.Column(database.Enum(TicketPriority), default=TicketPriority.NORMAL)
        status = database.Column(database.Enum(TicketStatus), default=TicketStatus.OPEN)
        
        browser_info = database.Column(database.Text)
        device_info = database.Column(database.Text)
        ip_address = database.Column(database.String(45))
        user_agent = database.Column(database.Text)
        page_url = database.Column(database.String(500))
        
        created_at = database.Column(database.DateTime, default=datetime.utcnow)
        first_response_at = database.Column(database.DateTime)
        resolved_at = database.Column(database.DateTime)
        closed_at = database.Column(database.DateTime)
        sla_deadline = database.Column(database.DateTime)
        sla_breached = database.Column(database.Boolean, default=False)
        
        assigned_to = database.Column(database.Integer, database.ForeignKey('user.id'), nullable=True)
        
        responses = database.relationship('SupportResponse', backref='ticket', lazy='dynamic', cascade='all, delete-orphan')
        activities = database.relationship('TicketActivity', backref='ticket', lazy='dynamic', cascade='all, delete-orphan')

    class SupportResponse(database.Model):
        __tablename__ = 'support_responses'
        
        id = database.Column(database.Integer, primary_key=True)
        ticket_id = database.Column(database.Integer, database.ForeignKey('support_tickets.id'), nullable=False)
        message = database.Column(database.Text, nullable=False)
        
        is_from_staff = database.Column(database.Boolean, default=False)
        staff_id = database.Column(database.Integer, database.ForeignKey('user.id'), nullable=True)
        staff_name = database.Column(database.String(255))
        
        created_at = database.Column(database.DateTime, default=datetime.utcnow)
        
        email_sent = database.Column(database.Boolean, default=False)
        email_sent_at = database.Column(database.DateTime)

    class FAQ(database.Model):
        __tablename__ = 'faqs'
        
        id = database.Column(database.Integer, primary_key=True)
        question = database.Column(database.String(500), nullable=False)
        answer = database.Column(database.Text, nullable=False)
        
        category_id = database.Column(database.Integer, database.ForeignKey('support_categories.id'), nullable=False)
        tags = database.Column(database.Text)
        
        sort_order = database.Column(database.Integer, default=0)
        is_featured = database.Column(database.Boolean, default=False)
        is_published = database.Column(database.Boolean, default=True)
        
        view_count = database.Column(database.Integer, default=0)
        helpful_count = database.Column(database.Integer, default=0)
        not_helpful_count = database.Column(database.Integer, default=0)
        
        created_at = database.Column(database.DateTime, default=datetime.utcnow)
        updated_at = database.Column(database.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    class Feedback(database.Model):
        __tablename__ = 'feedback'
        
        id = database.Column(database.Integer, primary_key=True)
        subject = database.Column(database.String(255), nullable=False)
        message = database.Column(database.Text, nullable=False)
        
        user_id = database.Column(database.Integer, database.ForeignKey('user.id'), nullable=True)
        user_email = database.Column(database.String(255), nullable=False)
        user_name = database.Column(database.String(255), nullable=False)
        
        feedback_type = database.Column(database.Enum(FeedbackType), default=FeedbackType.GENERAL)
        rating = database.Column(database.Integer)
        
        page_url = database.Column(database.String(500))
        browser_info = database.Column(database.Text)
        ip_address = database.Column(database.String(45))
        
        is_read = database.Column(database.Boolean, default=False)
        admin_notes = database.Column(database.Text)
        
        created_at = database.Column(database.DateTime, default=datetime.utcnow)

    class TicketActivity(database.Model):
        __tablename__ = 'ticket_activities'
        
        id = database.Column(database.Integer, primary_key=True)
        ticket_id = database.Column(database.Integer, database.ForeignKey('support_tickets.id'), nullable=False)
        
        action = database.Column(database.String(100), nullable=False)
        description = database.Column(database.Text)
        old_value = database.Column(database.Text)
        new_value = database.Column(database.Text)
        
        actor_type = database.Column(database.String(20), nullable=False)
        actor_id = database.Column(database.Integer, nullable=True)
        actor_name = database.Column(database.String(255))
        
        created_at = database.Column(database.DateTime, default=datetime.utcnow)

    globals().update({
        'SupportCategory': SupportCategory,
        'SupportTicket': SupportTicket,
        'SupportResponse': SupportResponse,
        'FAQ': FAQ,
        'Feedback': Feedback,
        'TicketActivity': TicketActivity
    })
    
    models.update({
        'SupportCategory': SupportCategory,
        'SupportTicket': SupportTicket,
        'SupportResponse': SupportResponse,
        'FAQ': FAQ,
        'Feedback': Feedback,
        'TicketActivity': TicketActivity
    })
    
    try:
        with app.app_context():
            db.create_all()
            from threading import Timer
            Timer(3.0, create_default_data).start()
    except Exception as e:
        logger.error(f"Error creating support tables: {e}")
    
    logger.info("✅ Support service initialized successfully")

def create_default_data():
    try:
        if not app:
            logger.warning("App not initialized, skipping default data creation")
            return
            
        with app.app_context():
            if SupportCategory.query.first():
                logger.info("Support categories already exist, skipping default data creation")
                return
                
            categories = [
                {'name': 'Account & Login', 'description': 'Issues with account creation, login, password reset', 'icon': '👤', 'sort_order': 1},
                {'name': 'Technical Issues', 'description': 'App crashes, loading issues, performance problems', 'icon': '🔧', 'sort_order': 2},
                {'name': 'Features & Functions', 'description': 'How to use features, feature requests', 'icon': '⚡', 'sort_order': 3},
                {'name': 'Content & Recommendations', 'description': 'Issues with movies, shows, recommendations', 'icon': '🎬', 'sort_order': 4},
                {'name': 'Billing & Subscription', 'description': 'Payment issues, subscription questions', 'icon': '💳', 'sort_order': 5},
                {'name': 'General Support', 'description': 'Other questions and general inquiries', 'icon': '❓', 'sort_order': 6}
            ]
            
            for cat_data in categories:
                if not SupportCategory.query.filter_by(name=cat_data['name']).first():
                    category = SupportCategory(**cat_data)
                    db.session.add(category)
            
            db.session.commit()
            
            account_category = SupportCategory.query.filter_by(name='Account & Login').first()
            tech_category = SupportCategory.query.filter_by(name='Technical Issues').first()
            features_category = SupportCategory.query.filter_by(name='Features & Functions').first()
            content_category = SupportCategory.query.filter_by(name='Content & Recommendations').first()
            
            if account_category and tech_category and features_category and content_category:
                faqs = [
                    {
                        'category_id': account_category.id,
                        'question': 'How do I reset my password?',
                        'answer': '''<p>To reset your password:</p>
                        <ol>
                            <li>Go to the login page and click "Forgot Password"</li>
                            <li>Enter your email address</li>
                            <li>Check your email for reset instructions</li>
                            <li>Click the reset link and create a new password</li>
                        </ol>
                        <p>If you don't receive the email, check your spam folder or contact us for assistance.</p>''',
                        'tags': '["password", "reset", "login", "account"]',
                        'is_featured': True,
                        'sort_order': 1
                    },
                    {
                        'category_id': tech_category.id,
                        'question': 'The app is loading slowly. What can I do?',
                        'answer': '''<p>Try these troubleshooting steps:</p>
                        <ol>
                            <li><strong>Check your internet connection</strong> - Ensure you have a stable connection</li>
                            <li><strong>Clear browser cache</strong> - Clear cookies and cached data</li>
                            <li><strong>Disable browser extensions</strong> - Some extensions can slow down performance</li>
                            <li><strong>Try incognito/private mode</strong> - This helps identify extension issues</li>
                            <li><strong>Update your browser</strong> - Use the latest version for best performance</li>
                        </ol>
                        <p>If issues persist, please report the problem with your browser and device details.</p>''',
                        'tags': '["performance", "loading", "slow", "browser"]',
                        'is_featured': True,
                        'sort_order': 1
                    }
                ]
                
                for faq_data in faqs:
                    if not FAQ.query.filter_by(question=faq_data['question']).first():
                        faq = FAQ(**faq_data)
                        db.session.add(faq)
                
                db.session.commit()
            
            logger.info("Default support data created successfully")
            
    except Exception as e:
        logger.error(f"Error creating default support data: {e}")
        if db:
            db.session.rollback()

def get_user_from_token():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    try:
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        return User.query.get(payload.get('user_id'))
    except:
        return None

def get_request_info():
    ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
    if ip_address:
        ip_address = ip_address.split(',')[0].strip()
    
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    return {
        'ip_address': ip_address,
        'user_agent': user_agent,
        'browser_info': user_agent,
        'page_url': request.headers.get('Referer', '')
    }

@support_bp.route('/api/support/contact', methods=['POST', 'OPTIONS'])
def submit_contact_form():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        required_fields = ['name', 'email', 'subject', 'message', 'category_id']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field.replace("_", " ").title()} is required'}), 400
        
        if not EMAIL_REGEX.match(data['email']):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        if not RateLimitService.check_rate_limit(f"contact:{data['email']}", max_requests=3, window=900):
            return jsonify({'error': 'Too many requests. Please try again in 15 minutes.'}), 429
        
        user = get_user_from_token()
        request_info = get_request_info()
        
        ticket_data = {
            'subject': data['subject'],
            'description': data['message'],
            'user_id': user.id if user else None,
            'user_email': data['email'],
            'user_name': data['name'],
            'category_id': data['category_id'],
            'ticket_type': 'contact',
            'priority': data.get('priority', 'normal'),
            **request_info
        }
        
        result = support_service.create_ticket(ticket_data)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': 'Your message has been sent successfully. We\'ll get back to you soon!',
                'ticket_number': result['ticket_number']
            }), 201
        else:
            return jsonify({'error': 'Failed to submit contact form'}), 500
        
    except Exception as e:
        logger.error(f"Contact form error: {e}")
        return jsonify({'error': 'Failed to process your request'}), 500

@support_bp.route('/api/support/feedback', methods=['POST', 'OPTIONS'])
def submit_feedback():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        required_fields = ['name', 'email', 'subject', 'message']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field.replace("_", " ").title()} is required'}), 400
        
        if not EMAIL_REGEX.match(data['email']):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        if not RateLimitService.check_rate_limit(f"feedback:{data['email']}", max_requests=5, window=900):
            return jsonify({'error': 'Too many feedback submissions. Please try again in 15 minutes.'}), 429
        
        user = get_user_from_token()
        request_info = get_request_info()
        
        feedback_data = {
            'subject': data['subject'],
            'message': data['message'],
            'user_id': user.id if user else None,
            'user_email': data['email'],
            'user_name': data['name'],
            'feedback_type': data.get('feedback_type', 'general'),
            'rating': data.get('rating'),
            **request_info
        }
        
        result = support_service.submit_feedback(feedback_data)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': 'Thank you for your feedback! We appreciate your input.'
            }), 201
        else:
            return jsonify({'error': 'Failed to submit feedback'}), 500
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        return jsonify({'error': 'Failed to process your feedback'}), 500

@support_bp.route('/api/support/report-issue', methods=['POST', 'OPTIONS'])
def report_issue():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        required_fields = ['name', 'email', 'subject', 'description']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field.replace("_", " ").title()} is required'}), 400
        
        if not EMAIL_REGEX.match(data['email']):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        if not RateLimitService.check_rate_limit(f"issue:{data['email']}", max_requests=5, window=900):
            return jsonify({'error': 'Too many issue reports. Please try again in 15 minutes.'}), 429
        
        user = get_user_from_token()
        request_info = get_request_info()
        
        tech_category = SupportCategory.query.filter_by(name='Technical Issues').first()
        if not tech_category:
            tech_category = SupportCategory.query.first()
        
        ticket_data = {
            'subject': f"[BUG REPORT] {data['subject']}",
            'description': f"Steps to reproduce:\n{data.get('steps', 'Not provided')}\n\nExpected behavior:\n{data.get('expected', 'Not provided')}\n\nActual behavior:\n{data['description']}\n\nAdditional info:\n{data.get('additional_info', 'None')}",
            'user_id': user.id if user else None,
            'user_email': data['email'],
            'user_name': data['name'],
            'category_id': tech_category.id,
            'ticket_type': 'bug_report',
            'priority': data.get('severity', 'normal'),
            **request_info
        }
        
        result = support_service.create_ticket(ticket_data)
        
        if result['success']:
            html, text = email_service.get_template(
                'issue_report',
                user_name=data['name'],
                issue_type='bug report'
            )
            
            email_service.queue_email(
                to=data['email'],
                subject="Issue Report Received - Thank You!",
                html=html,
                text=text,
                category='issue_report'
            )
            
            return jsonify({
                'success': True,
                'message': 'Issue report submitted successfully. Our team will investigate.',
                'ticket_number': result['ticket_number']
            }), 201
        else:
            return jsonify({'error': 'Failed to submit issue report'}), 500
        
    except Exception as e:
        logger.error(f"Issue report error: {e}")
        return jsonify({'error': 'Failed to process your issue report'}), 500

@support_bp.route('/api/support/faq', methods=['GET'])
def get_faqs():
    try:
        category_id = request.args.get('category_id', type=int)
        featured_only = request.args.get('featured', 'false').lower() == 'true'
        search = request.args.get('search', '').strip()
        
        faqs = CacheService.get_faqs(category_id, featured_only)
        
        if search:
            search_lower = search.lower()
            faqs = [
                faq for faq in faqs 
                if search_lower in faq['question'].lower() or search_lower in faq['answer'].lower()
            ]
        
        categories = SupportCategory.query.filter_by(is_active=True).order_by(SupportCategory.sort_order).all()
        
        return jsonify({
            'faqs': faqs,
            'categories': [
                {
                    'id': cat.id,
                    'name': cat.name,
                    'description': cat.description,
                    'icon': cat.icon,
                    'faq_count': FAQ.query.filter_by(category_id=cat.id, is_published=True).count()
                }
                for cat in categories
            ],
            'total_results': len(faqs)
        }), 200
        
    except Exception as e:
        logger.error(f"FAQ retrieval error: {e}")
        return jsonify({'error': 'Failed to retrieve FAQs'}), 500

@support_bp.route('/api/support/faq/<int:faq_id>/helpful', methods=['POST'])
def mark_faq_helpful(faq_id):
    try:
        data = request.get_json()
        is_helpful = data.get('helpful', True)
        
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
        if not RateLimitService.check_rate_limit(f"faq_vote:{ip_address}:{faq_id}", max_requests=1, window=3600):
            return jsonify({'error': 'You can only vote once per hour per FAQ'}), 429
        
        faq = FAQ.query.get_or_404(faq_id)
        
        if is_helpful:
            faq.helpful_count += 1
        else:
            faq.not_helpful_count += 1
        
        faq.view_count += 1
        db.session.commit()
        
        return jsonify({
            'success': True,
            'helpful_count': faq.helpful_count,
            'not_helpful_count': faq.not_helpful_count
        }), 200
        
    except Exception as e:
        logger.error(f"FAQ voting error: {e}")
        return jsonify({'error': 'Failed to record your vote'}), 500

@support_bp.route('/api/support/help-center', methods=['GET'])
def get_help_center_data():
    try:
        categories = SupportCategory.query.filter_by(is_active=True).order_by(SupportCategory.sort_order).all()
        
        featured_faqs = CacheService.get_faqs(featured_only=True)
        
        user = get_user_from_token()
        recent_tickets_count = 0
        if user:
            recent_tickets_count = SupportTicket.query.filter(
                SupportTicket.user_id == user.id,
                SupportTicket.created_at >= datetime.utcnow() - timedelta(days=30)
            ).count()
        
        quick_links = [
            {
                'title': 'Contact Support',
                'description': 'Get personalized help from our team',
                'url': '/support/contact',
                'icon': '💬'
            },
            {
                'title': 'Report an Issue',
                'description': 'Found a bug? Let us know!',
                'url': '/support/report-issue',
                'icon': '🐛'
            },
            {
                'title': 'Send Feedback',
                'description': 'Share your thoughts and suggestions',
                'url': '/support/feedback',
                'icon': '💝'
            },
            {
                'title': 'Browse FAQ',
                'description': 'Find answers to common questions',
                'url': '/support/faq',
                'icon': '❓'
            }
        ]
        
        return jsonify({
            'categories': [
                {
                    'id': cat.id,
                    'name': cat.name,
                    'description': cat.description,
                    'icon': cat.icon,
                    'faq_count': FAQ.query.filter_by(category_id=cat.id, is_published=True).count()
                }
                for cat in categories
            ],
            'featured_faqs': featured_faqs[:6],
            'quick_links': quick_links,
            'stats': {
                'total_faqs': FAQ.query.filter_by(is_published=True).count(),
                'total_categories': len(categories),
                'user_recent_tickets': recent_tickets_count
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Help center data error: {e}")
        return jsonify({'error': 'Failed to load help center data'}), 500

@support_bp.route('/api/support/categories', methods=['GET'])
def get_support_categories():
    try:
        categories = SupportCategory.query.filter_by(is_active=True).order_by(SupportCategory.sort_order).all()
        
        return jsonify({
            'categories': [
                {
                    'id': cat.id,
                    'name': cat.name,
                    'description': cat.description,
                    'icon': cat.icon
                }
                for cat in categories
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Categories retrieval error: {e}")
        return jsonify({'error': 'Failed to retrieve categories'}), 500

@support_bp.route('/api/support/health', methods=['GET'])
def support_health():
    try:
        categories_count = SupportCategory.query.count()
        faqs_count = FAQ.query.count()
        tickets_count = SupportTicket.query.count()
        
        redis_status = 'not_configured'
        email_queue_size = 0
        if redis_client:
            try:
                redis_client.ping()
                redis_status = 'connected'
                email_queue_size = redis_client.llen('support_email_queue')
            except:
                redis_status = 'disconnected'
        
        email_status = 'configured' if email_service else 'not_configured'
        
        return jsonify({
            'status': 'healthy',
            'service': 'support',
            'database': {
                'categories': categories_count,
                'faqs': faqs_count,
                'tickets': tickets_count
            },
            'redis_status': redis_status,
            'email_status': email_status,
            'email_queue_size': email_queue_size,
            'features': {
                'contact_forms': True,
                'feedback_collection': True,
                'issue_reporting': True,
                'faq_system': True,
                'rate_limiting': True,
                'email_notifications': True
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Support health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'service': 'support',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@support_bp.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    allowed_origins = [FRONTEND_URL, 'http://127.0.0.1:5500', 'http://127.0.0.1:5501']
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response