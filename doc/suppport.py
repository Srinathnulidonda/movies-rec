# backend/services/support.py
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash
import jwt
import os
import logging
import json
import uuid
import re
import threading
import time
from functools import wraps
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import redis
from sqlalchemy import text, desc, and_, or_
from sqlalchemy.dialects.postgresql import JSON

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
support_bp = Blueprint('support', __name__)

# Frontend URL configuration
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://cinebrain.vercel.app')
BACKEND_URL = os.environ.get('BACKEND_URL', 'https://backend-app-970m.onrender.com')

# Redis configuration
REDIS_URL = os.environ.get('REDIS_URL', 'redis://red-d2qlbuje5dus73c71qog:xp7inVzgblGCbo9I4taSGLdKUg0xY91I@red-d2qlbuje5dus73c71qog:6379')

# Email validation regex
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# These will be initialized by init_support function
app = None
db = None
User = None
redis_client = None
support_email_service = None

# Database Models for Support System
class SupportTicket(db.Model):
    """Support ticket model"""
    __tablename__ = 'support_tickets'
    
    id = db.Column(db.Integer, primary_key=True)
    ticket_id = db.Column(db.String(20), unique=True, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    email = db.Column(db.String(255), nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    subject = db.Column(db.String(500), nullable=False)
    message = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), nullable=False, index=True)
    priority = db.Column(db.String(20), default='medium', index=True)  # low, medium, high, urgent
    status = db.Column(db.String(20), default='open', index=True)  # open, in_progress, resolved, closed
    assigned_to = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    metadata = db.Column(JSON)
    user_agent = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    user = db.relationship('User', foreign_keys=[user_id], backref='support_tickets')
    assigned_user = db.relationship('User', foreign_keys=[assigned_to])
    responses = db.relationship('SupportResponse', backref='ticket', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<SupportTicket {self.ticket_id}: {self.subject[:50]}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'ticket_id': self.ticket_id,
            'email': self.email,
            'name': self.name,
            'subject': self.subject,
            'message': self.message,
            'category': self.category,
            'priority': self.priority,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'user_id': self.user_id,
            'assigned_to': self.assigned_to
        }

class SupportResponse(db.Model):
    """Support ticket responses"""
    __tablename__ = 'support_responses'
    
    id = db.Column(db.Integer, primary_key=True)
    ticket_id = db.Column(db.Integer, db.ForeignKey('support_tickets.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    email = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_staff_response = db.Column(db.Boolean, default=False)
    is_internal = db.Column(db.Boolean, default=False)
    metadata = db.Column(JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='support_responses')
    
    def to_dict(self):
        return {
            'id': self.id,
            'ticket_id': self.ticket_id,
            'email': self.email,
            'name': self.name,
            'message': self.message,
            'is_staff_response': self.is_staff_response,
            'is_internal': self.is_internal,
            'created_at': self.created_at.isoformat(),
            'user_id': self.user_id
        }

class FAQ(db.Model):
    """FAQ model"""
    __tablename__ = 'faqs'
    
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(100), nullable=False, index=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    order = db.Column(db.Integer, default=0, index=True)
    is_featured = db.Column(db.Boolean, default=False, index=True)
    is_active = db.Column(db.Boolean, default=True, index=True)
    view_count = db.Column(db.Integer, default=0)
    helpful_count = db.Column(db.Integer, default=0)
    not_helpful_count = db.Column(db.Integer, default=0)
    tags = db.Column(db.Text)  # JSON string of tags
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    creator = db.relationship('User', backref='created_faqs')
    
    def to_dict(self):
        return {
            'id': self.id,
            'category': self.category,
            'question': self.question,
            'answer': self.answer,
            'order': self.order,
            'is_featured': self.is_featured,
            'is_active': self.is_active,
            'view_count': self.view_count,
            'helpful_count': self.helpful_count,
            'not_helpful_count': self.not_helpful_count,
            'tags': json.loads(self.tags) if self.tags else [],
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Feedback(db.Model):
    """User feedback model"""
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    email = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    feedback_type = db.Column(db.String(50), nullable=False, index=True)  # suggestion, bug, feature, general
    subject = db.Column(db.String(500), nullable=False)
    message = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer, nullable=True)  # 1-5 star rating
    page_url = db.Column(db.String(1000), nullable=True)
    browser_info = db.Column(db.Text)
    screenshot_url = db.Column(db.String(500), nullable=True)
    is_reviewed = db.Column(db.Boolean, default=False, index=True)
    is_implemented = db.Column(db.Boolean, default=False)
    admin_notes = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    reviewed_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    user = db.relationship('User', foreign_keys=[user_id], backref='feedback_submissions')
    reviewer = db.relationship('User', foreign_keys=[reviewed_by])
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'feedback_type': self.feedback_type,
            'subject': self.subject,
            'message': self.message,
            'rating': self.rating,
            'page_url': self.page_url,
            'is_reviewed': self.is_reviewed,
            'is_implemented': self.is_implemented,
            'created_at': self.created_at.isoformat(),
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'user_id': self.user_id
        }

class SupportCategory(db.Model):
    """Support categories for organization"""
    __tablename__ = 'support_categories'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    slug = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    icon = db.Column(db.String(50))  # Icon name/class
    color = db.Column(db.String(7))  # Hex color
    order = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'slug': self.slug,
            'description': self.description,
            'icon': self.icon,
            'color': self.color,
            'order': self.order,
            'is_active': self.is_active
        }

def init_redis():
    """Initialize Redis connection"""
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
        logger.info("Redis connected successfully for support service")
        return redis_client
    except Exception as e:
        logger.error(f"Redis connection failed for support: {e}")
        return None

class SupportEmailService:
    """Professional email service for support system"""
    
    def __init__(self, username, password):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.username = username
        self.password = password
        self.from_email = "support@cinebrain.com"
        self.from_name = "CineBrain Support"
        self.redis_client = redis_client
        
        # Import the email service from auth
        try:
            from .auth import ProfessionalEmailService
            self.email_service = ProfessionalEmailService(username, password)
        except ImportError:
            logger.error("Could not import ProfessionalEmailService from auth module")
            self.email_service = None
    
    def send_support_email(self, template_type: str, to_email: str, **kwargs):
        """Send support-related emails using auth email service"""
        if not self.email_service:
            logger.error("Email service not available")
            return False
        
        try:
            html_content, text_content = self.get_support_template(template_type, **kwargs)
            
            subject_map = {
                'ticket_created': f"Support Ticket Created - {kwargs.get('ticket_id', 'N/A')} - CineBrain",
                'ticket_response': f"Support Ticket Updated - {kwargs.get('ticket_id', 'N/A')} - CineBrain",
                'feedback_received': "Thank you for your feedback - CineBrain",
                'contact_confirmation': "We've received your message - CineBrain",
                'issue_report': f"Issue Report Received - {kwargs.get('report_id', 'N/A')} - CineBrain"
            }
            
            subject = subject_map.get(template_type, "CineBrain Support")
            
            return self.email_service.queue_email(
                to=to_email,
                subject=subject,
                html=html_content,
                text=text_content,
                priority='normal'
            )
            
        except Exception as e:
            logger.error(f"Failed to send support email: {e}")
            return False
    
    def get_support_template(self, template_type: str, **kwargs) -> tuple:
        """Get support-specific email templates"""
        
        # Base CSS from auth service
        base_css = """
        <style type="text/css">
            body, table, td, a { -webkit-text-size-adjust: 100%; -ms-text-size-adjust: 100%; }
            table, td { mso-table-lspace: 0pt; mso-table-rspace: 0pt; }
            img { -ms-interpolation-mode: bicubic; border: 0; outline: none; text-decoration: none; }
            
            body {
                margin: 0 !important; padding: 0 !important; width: 100% !important; min-width: 100% !important;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
                font-size: 14px; line-height: 1.6; color: #202124; background-color: #f8f9fa;
            }
            
            .email-wrapper { width: 100%; background-color: #f8f9fa; padding: 40px 20px; }
            .email-container { max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; 
                box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15); overflow: hidden; }
            
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 48px; text-align: center; }
            .header-logo { font-size: 32px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; margin: 0; 
                text-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .header-tagline { font-size: 14px; color: rgba(255,255,255,0.95); margin: 8px 0 0 0; font-weight: 400; }
            
            .content { padding: 48px; background-color: #ffffff; }
            h1 { font-size: 24px; font-weight: 400; color: #202124; margin: 0 0 24px; line-height: 1.3; }
            h2 { font-size: 18px; font-weight: 500; color: #202124; margin: 24px 0 12px; }
            p { margin: 0 0 16px; color: #5f6368; font-size: 14px; line-height: 1.6; }
            
            .btn { display: inline-block; padding: 12px 32px; font-size: 14px; font-weight: 500;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                text-decoration: none !important; text-align: center; border-radius: 24px; transition: all 0.3s;
                cursor: pointer; margin: 24px 0; }
            
            .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff !important;
                box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4); }
            
            .alert { padding: 16px; border-radius: 8px; margin: 24px 0; font-size: 14px; }
            .alert-info { background-color: #e8f0fe; border-left: 4px solid #1a73e8; color: #1967d2; }
            .alert-success { background-color: #e6f4ea; border-left: 4px solid #34a853; color: #188038; }
            .alert-warning { background-color: #fef7e0; border-left: 4px solid #fbbc04; color: #ea8600; }
            
            .info-box { background-color: #f8f9fa; border-radius: 8px; padding: 24px; margin: 24px 0; border: 1px solid #e8eaed; }
            .ticket-details { background-color: #f8f9fa; border-radius: 8px; padding: 20px; margin: 20px 0; 
                border-left: 4px solid #667eea; }
            
            .footer { background-color: #f8f9fa; padding: 32px 48px; text-align: center; border-top: 1px solid #e8eaed; }
            .footer-text { font-size: 12px; color: #80868b; margin: 0 0 8px; line-height: 1.5; }
            .footer-links { margin: 16px 0; }
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
        elif template_type == 'ticket_response':
            return self._get_ticket_response_template(base_css, **kwargs)
        elif template_type == 'feedback_received':
            return self._get_feedback_template(base_css, **kwargs)
        elif template_type == 'contact_confirmation':
            return self._get_contact_template(base_css, **kwargs)
        elif template_type == 'issue_report':
            return self._get_issue_report_template(base_css, **kwargs)
        else:
            return self._get_generic_support_template(base_css, **kwargs)
    
    def _get_ticket_created_template(self, base_css: str, **kwargs) -> tuple:
        """Support ticket created email template"""
        ticket_id = kwargs.get('ticket_id', 'N/A')
        name = kwargs.get('name', 'Customer')
        subject = kwargs.get('subject', 'Support Request')
        category = kwargs.get('category', 'General')
        priority = kwargs.get('priority', 'Medium')
        
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
                <div class="email-container">
                    <div class="header">
                        <h1 class="header-logo">🎬 CineBrain Support</h1>
                        <p class="header-tagline">We're here to help you</p>
                    </div>
                    
                    <div class="content">
                        <h1>Support Ticket Created</h1>
                        
                        <p>Hi {name},</p>
                        
                        <p>Thank you for contacting CineBrain Support. We've received your support request and created a ticket for you.</p>
                        
                        <div class="ticket-details">
                            <h3 style="margin: 0 0 12px; color: #202124;">Ticket Details</h3>
                            <p style="margin: 4px 0;"><strong>Ticket ID:</strong> {ticket_id}</p>
                            <p style="margin: 4px 0;"><strong>Subject:</strong> {subject}</p>
                            <p style="margin: 4px 0;"><strong>Category:</strong> {category}</p>
                            <p style="margin: 4px 0;"><strong>Priority:</strong> {priority}</p>
                            <p style="margin: 4px 0;"><strong>Created:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')}</p>
                        </div>
                        
                        <div class="alert alert-info">
                            <strong>📧 Keep this ticket ID for reference</strong><br>
                            Please save the ticket ID <strong>{ticket_id}</strong> for future correspondence about this issue.
                        </div>
                        
                        <h2>What happens next?</h2>
                        <p>Our support team will review your request and respond within:</p>
                        <ul style="color: #5f6368; padding-left: 20px;">
                            <li><strong>Urgent issues:</strong> 2-4 hours</li>
                            <li><strong>High priority:</strong> 4-8 hours</li>
                            <li><strong>Normal requests:</strong> 12-24 hours</li>
                            <li><strong>Low priority:</strong> 24-48 hours</li>
                        </ul>
                        
                        <center>
                            <a href="{FRONTEND_URL}/support/ticket?id={ticket_id}" class="btn btn-primary">View Ticket Status</a>
                        </center>
                        
                        <div class="info-box">
                            <p style="margin: 0; font-size: 13px; color: #5f6368;">
                                <strong>Need immediate help?</strong><br>
                                Check our <a href="{FRONTEND_URL}/support/faq.html" style="color: #1a73e8;">FAQ section</a> 
                                or browse our <a href="{FRONTEND_URL}/support/help-center.html" style="color: #1a73e8;">Help Center</a> 
                                for instant answers to common questions.
                            </p>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <div class="footer-links">
                            <a href="{FRONTEND_URL}/support" class="footer-link">Support Center</a>
                            <a href="{FRONTEND_URL}/support/faq.html" class="footer-link">FAQ</a>
                            <a href="{FRONTEND_URL}/contact" class="footer-link">Contact</a>
                        </div>
                        <p class="footer-text">
                            © {datetime.now().year} CineBrain, Inc. All rights reserved.<br>
                            This email was sent regarding your support request.
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Support Ticket Created - CineBrain

Hi {name},

Thank you for contacting CineBrain Support. We've received your support request.

Ticket Details:
- Ticket ID: {ticket_id}
- Subject: {subject}
- Category: {category}
- Priority: {priority}
- Created: {datetime.now().strftime('%B %d, %Y at %I:%M %p UTC')}

Our support team will respond within:
- Urgent issues: 2-4 hours
- High priority: 4-8 hours  
- Normal requests: 12-24 hours
- Low priority: 24-48 hours

View your ticket: {FRONTEND_URL}/support/ticket?id={ticket_id}

Need immediate help? Visit our Help Center: {FRONTEND_URL}/support/help-center.html

Best regards,
CineBrain Support Team

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text
    
    def _get_feedback_template(self, base_css: str, **kwargs) -> tuple:
        """Feedback received confirmation template"""
        name = kwargs.get('name', 'User')
        feedback_type = kwargs.get('feedback_type', 'feedback')
        subject = kwargs.get('subject', 'User Feedback')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Thank you for your feedback - CineBrain</title>
            {base_css}
        </head>
        <body>
            <div class="email-wrapper">
                <div class="email-container">
                    <div class="header" style="background: linear-gradient(135deg, #34a853 0%, #0d8043 100%);">
                        <h1 class="header-logo">💚 Thank You!</h1>
                        <p class="header-tagline">Your feedback helps us improve</p>
                    </div>
                    
                    <div class="content">
                        <h1>Feedback Received</h1>
                        
                        <p>Hi {name},</p>
                        
                        <p>Thank you for taking the time to share your {feedback_type} with CineBrain. Your input is incredibly valuable to us and helps make our platform better for everyone.</p>
                        
                        <div class="alert alert-success">
                            <strong>✅ Your {feedback_type} has been recorded</strong><br>
                            Subject: "{subject}"
                        </div>
                        
                        <h2>What happens next?</h2>
                        <ul style="color: #5f6368; padding-left: 20px;">
                            <li>Our team will review your feedback within 48 hours</li>
                            <li>If it's a feature request, we'll consider it for future updates</li>
                            <li>For bug reports, we'll investigate and work on a fix</li>
                            <li>We may reach out if we need more information</li>
                        </ul>
                        
                        <center>
                            <a href="{FRONTEND_URL}/support/feedback.html" class="btn btn-primary">Submit More Feedback</a>
                        </center>
                        
                        <div class="info-box">
                            <h3 style="margin: 0 0 12px; color: #202124;">Help us improve further</h3>
                            <p style="margin: 0; font-size: 13px; color: #5f6368;">
                                Love CineBrain? Consider <a href="{FRONTEND_URL}/review" style="color: #1a73e8;">leaving us a review</a> 
                                or <a href="{FRONTEND_URL}/share" style="color: #1a73e8;">sharing with friends</a>!
                            </p>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p class="footer-text">
                            Keep the feedback coming! We read every single submission.<br>
                            © {datetime.now().year} CineBrain, Inc.
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""
Thank you for your feedback - CineBrain

Hi {name},

Thank you for sharing your {feedback_type} with CineBrain. Your input helps us improve!

Subject: "{subject}"

What happens next:
- Our team will review your feedback within 48 hours
- We'll consider feature requests for future updates
- Bug reports will be investigated and fixed
- We may reach out if we need more information

Submit more feedback: {FRONTEND_URL}/support/feedback.html

Best regards,
CineBrain Team

© {datetime.now().year} CineBrain, Inc.
        """
        
        return html, text

def init_support(flask_app, database, models, services):
    """Initialize support module"""
    global app, db, User, redis_client, support_email_service
    
    app = flask_app
    db = database
    User = models.get('User')
    
    # Initialize Redis
    redis_client = init_redis()
    
    # Initialize support email service
    gmail_username = os.environ.get('GMAIL_USERNAME', 'projects.srinath@gmail.com')
    gmail_password = os.environ.get('GMAIL_APP_PASSWORD', 'wuus nsow nbee xewv')
    
    support_email_service = SupportEmailService(gmail_username, gmail_password)
    
    # Create support database tables
    with app.app_context():
        try:
            # Create tables if they don't exist
            db.create_all()
            
            # Initialize default support categories
            init_default_categories()
            
            # Initialize default FAQs
            init_default_faqs()
            
            logger.info("✅ Support module initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize support module: {e}")

def init_default_categories():
    """Initialize default support categories"""
    try:
        if SupportCategory.query.count() == 0:
            default_categories = [
                {
                    'name': 'Account & Login',
                    'slug': 'account-login',
                    'description': 'Issues with account creation, login, password reset',
                    'icon': 'user',
                    'color': '#667eea',
                    'order': 1
                },
                {
                    'name': 'Movie & TV Recommendations',
                    'slug': 'recommendations',
                    'description': 'Questions about our AI recommendations and content discovery',
                    'icon': 'film',
                    'color': '#764ba2',
                    'order': 2
                },
                {
                    'name': 'Technical Issues',
                    'slug': 'technical',
                    'description': 'Website bugs, loading issues, performance problems',
                    'icon': 'settings',
                    'color': '#ea4335',
                    'order': 3
                },
                {
                    'name': 'Feature Requests',
                    'slug': 'features',
                    'description': 'Suggestions for new features and improvements',
                    'icon': 'plus',
                    'color': '#34a853',
                    'order': 4
                },
                {
                    'name': 'Content Issues',
                    'slug': 'content',
                    'description': 'Missing movies/shows, incorrect information, content updates',
                    'icon': 'database',
                    'color': '#fbbc04',
                    'order': 5
                },
                {
                    'name': 'General Support',
                    'slug': 'general',
                    'description': 'Other questions and general inquiries',
                    'icon': 'help-circle',
                    'color': '#80868b',
                    'order': 6
                }
            ]
            
            for cat_data in default_categories:
                category = SupportCategory(**cat_data)
                db.session.add(category)
            
            db.session.commit()
            logger.info("Default support categories created")
            
    except Exception as e:
        logger.error(f"Failed to create default categories: {e}")
        db.session.rollback()

def init_default_faqs():
    """Initialize default FAQ entries"""
    try:
        if FAQ.query.count() == 0:
            # Get admin user for FAQ creation
            admin_user = User.query.filter_by(is_admin=True).first()
            if not admin_user:
                logger.warning("No admin user found, skipping default FAQ creation")
                return
            
            default_faqs = [
                {
                    'category': 'Account & Login',
                    'question': 'How do I create a CineBrain account?',
                    'answer': 'Creating an account is simple! Click the "Sign Up" button in the top-right corner, fill in your email and create a password. You can also sign up using your Google account for faster registration.',
                    'is_featured': True,
                    'order': 1,
                    'tags': json.dumps(['account', 'registration', 'signup'])
                },
                {
                    'category': 'Account & Login',
                    'question': 'I forgot my password. How can I reset it?',
                    'answer': 'No worries! Click "Forgot Password" on the login page, enter your email address, and we\'ll send you a secure reset link. The link expires in 1 hour for security.',
                    'is_featured': True,
                    'order': 2,
                    'tags': json.dumps(['password', 'reset', 'login'])
                },
                {
                    'category': 'Movie & TV Recommendations',
                    'question': 'How does CineBrain\'s AI recommendation system work?',
                    'answer': 'Our AI analyzes your viewing preferences, ratings, and behavior patterns to suggest movies, TV shows, and anime you\'ll love. The more you interact with content (like, rate, add to wishlist), the better our recommendations become!',
                    'is_featured': True,
                    'order': 1,
                    'tags': json.dumps(['ai', 'recommendations', 'algorithm'])
                },
                {
                    'category': 'Movie & TV Recommendations',
                    'question': 'Can I get recommendations without creating an account?',
                    'answer': 'Yes! Our system provides great recommendations for anonymous users based on trending content, popular picks, and regional preferences. However, creating an account gives you personalized recommendations tailored to your taste.',
                    'is_featured': False,
                    'order': 2,
                    'tags': json.dumps(['anonymous', 'guest', 'recommendations'])
                },
                {
                    'category': 'Technical Issues',
                    'question': 'The website is loading slowly. What should I do?',
                    'answer': 'Try refreshing the page, clearing your browser cache, or checking your internet connection. If the issue persists, try using a different browser or device. You can also report the issue to our support team.',
                    'is_featured': False,
                    'order': 1,
                    'tags': json.dumps(['performance', 'loading', 'slow'])
                },
                {
                    'category': 'Content Issues',
                    'question': 'I can\'t find a specific movie or TV show. Why?',
                    'answer': 'CineBrain sources content from multiple databases. If you can\'t find something, it might not be in our system yet. Use the search function and submit a content request through our feedback form.',
                    'is_featured': False,
                    'order': 1,
                    'tags': json.dumps(['missing', 'content', 'search'])
                },
                {
                    'category': 'General Support',
                    'question': 'Is CineBrain free to use?',
                    'answer': 'Yes! CineBrain is completely free to use. You get unlimited AI-powered recommendations, can create wishlists, rate content, and access all features without any subscription fees.',
                    'is_featured': True,
                    'order': 1,
                    'tags': json.dumps(['free', 'pricing', 'cost'])
                },
                {
                    'category': 'General Support',
                    'question': 'How can I contact CineBrain support?',
                    'answer': 'You can reach us through multiple channels: use the contact form on our website, send email to support@cinebrain.com, or submit a support ticket. We typically respond within 24 hours.',
                    'is_featured': False,
                    'order': 2,
                    'tags': json.dumps(['contact', 'support', 'help'])
                }
            ]
            
            for faq_data in default_faqs:
                faq_data['created_by'] = admin_user.id
                faq = FAQ(**faq_data)
                db.session.add(faq)
            
            db.session.commit()
            logger.info("Default FAQs created")
            
    except Exception as e:
        logger.error(f"Failed to create default FAQs: {e}")
        db.session.rollback()

# Rate limiting function
def check_rate_limit(identifier: str, max_requests: int = 5, window: int = 300) -> bool:
    """Check if rate limit is exceeded using Redis"""
    if not redis_client:
        return True
    
    try:
        key = f"support_rate_limit:{identifier}"
        pipe = redis_client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        results = pipe.execute()
        
        current_count = results[0]
        
        if current_count > max_requests:
            logger.warning(f"Support rate limit exceeded for {identifier}: {current_count}/{max_requests}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Support rate limit check error: {e}")
        return True

def generate_ticket_id() -> str:
    """Generate unique ticket ID"""
    import random
    import string
    
    # Format: CB-YYYYMMDD-XXXX (CB = CineBrain)
    date_str = datetime.now().strftime('%Y%m%d')
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"CB-{date_str}-{random_str}"

def get_request_info(request):
    """Get request information for logging"""
    ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
    if ip_address:
        ip_address = ip_address.split(',')[0].strip()
    
    user_agent = request.headers.get('User-Agent', 'Unknown')
    return ip_address, user_agent

# Authentication decorator
def require_auth(f):
    """Require authentication"""
    from functools import wraps
    
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
            
            return f(current_user, *args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except:
            return jsonify({'error': 'Invalid token'}), 401
    
    return decorated_function

def require_admin(f):
    """Require admin authentication"""
    from functools import wraps
    
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
            if not current_user or not current_user.is_admin:
                return jsonify({'error': 'Admin access required'}), 403
            
            return f(current_user, *args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except:
            return jsonify({'error': 'Invalid token'}), 401
    
    return decorated_function

# === SUPPORT API ENDPOINTS ===

@support_bp.route('/api/support/contact', methods=['POST', 'OPTIONS'])
def contact_us():
    """Handle contact form submissions"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'subject', 'message', 'category']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field.title()} is required'}), 400
        
        name = data['name'].strip()
        email = data['email'].strip().lower()
        subject = data['subject'].strip()
        message = data['message'].strip()
        category = data['category'].strip()
        
        # Validate email format
        if not EMAIL_REGEX.match(email):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        # Check rate limiting
        if not check_rate_limit(f"contact:{email}", max_requests=3, window=3600):
            return jsonify({
                'error': 'Too many contact requests. Please try again in 1 hour.'
            }), 429
        
        # Get request info
        ip_address, user_agent = get_request_info(request)
        
        # Check if user is authenticated
        user_id = None
        try:
            token = request.headers.get('Authorization')
            if token:
                token = token.replace('Bearer ', '')
                data_decoded = jwt.decode(token, app.secret_key, algorithms=['HS256'])
                user_id = data_decoded['user_id']
        except:
            pass
        
        # Create support ticket
        ticket_id = generate_ticket_id()
        
        ticket = SupportTicket(
            ticket_id=ticket_id,
            user_id=user_id,
            email=email,
            name=name,
            subject=subject,
            message=message,
            category=category,
            priority='medium',
            status='open',
            metadata={
                'source': 'contact_form',
                'page_url': data.get('page_url'),
                'browser_info': user_agent,
                'form_data': {
                    'company': data.get('company'),
                    'phone': data.get('phone')
                }
            },
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        db.session.add(ticket)
        db.session.commit()
        
        # Send confirmation email to user
        if support_email_service:
            support_email_service.send_support_email(
                'ticket_created',
                email,
                ticket_id=ticket_id,
                name=name,
                subject=subject,
                category=category,
                priority='medium'
            )
        
        logger.info(f"Contact form submitted: {ticket_id} from {email}")
        
        return jsonify({
            'success': True,
            'message': 'Thank you for contacting us! We\'ll respond within 24 hours.',
            'ticket_id': ticket_id
        }), 200
        
    except Exception as e:
        logger.error(f"Contact form error: {e}")
        return jsonify({'error': 'Failed to submit contact form'}), 500

@support_bp.route('/api/support/feedback', methods=['POST', 'OPTIONS'])
def submit_feedback():
    """Handle feedback submissions"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'feedback_type', 'subject', 'message']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field.replace("_", " ").title()} is required'}), 400
        
        name = data['name'].strip()
        email = data['email'].strip().lower()
        feedback_type = data['feedback_type'].strip()
        subject = data['subject'].strip()
        message = data['message'].strip()
        
        # Validate email format
        if not EMAIL_REGEX.match(email):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        # Validate feedback type
        valid_types = ['suggestion', 'bug', 'feature', 'general', 'complaint', 'compliment']
        if feedback_type not in valid_types:
            return jsonify({'error': 'Invalid feedback type'}), 400
        
        # Check rate limiting
        if not check_rate_limit(f"feedback:{email}", max_requests=5, window=3600):
            return jsonify({
                'error': 'Too many feedback submissions. Please try again in 1 hour.'
            }), 429
        
        # Get request info
        ip_address, user_agent = get_request_info(request)
        
        # Check if user is authenticated
        user_id = None
        try:
            token = request.headers.get('Authorization')
            if token:
                token = token.replace('Bearer ', '')
                data_decoded = jwt.decode(token, app.secret_key, algorithms=['HS256'])
                user_id = data_decoded['user_id']
        except:
            pass
        
        # Create feedback entry
        feedback = Feedback(
            user_id=user_id,
            email=email,
            name=name,
            feedback_type=feedback_type,
            subject=subject,
            message=message,
            rating=data.get('rating'),
            page_url=data.get('page_url'),
            browser_info=user_agent,
            screenshot_url=data.get('screenshot_url'),
            ip_address=ip_address
        )
        
        db.session.add(feedback)
        db.session.commit()
        
        # Send confirmation email
        if support_email_service:
            support_email_service.send_support_email(
                'feedback_received',
                email,
                name=name,
                feedback_type=feedback_type,
                subject=subject
            )
        
        logger.info(f"Feedback submitted: {feedback_type} from {email}")
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your feedback! We appreciate your input.',
            'feedback_id': feedback.id
        }), 200
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        return jsonify({'error': 'Failed to submit feedback'}), 500

@support_bp.route('/api/support/report-issue', methods=['POST', 'OPTIONS'])
def report_issue():
    """Handle issue/bug reports"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'issue_type', 'title', 'description']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field.replace("_", " ").title()} is required'}), 400
        
        name = data['name'].strip()
        email = data['email'].strip().lower()
        issue_type = data['issue_type'].strip()
        title = data['title'].strip()
        description = data['description'].strip()
        
        # Validate email format
        if not EMAIL_REGEX.match(email):
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        # Check rate limiting
        if not check_rate_limit(f"issue_report:{email}", max_requests=5, window=3600):
            return jsonify({
                'error': 'Too many issue reports. Please try again in 1 hour.'
            }), 429
        
        # Get request info
        ip_address, user_agent = get_request_info(request)
        
        # Check if user is authenticated
        user_id = None
        try:
            token = request.headers.get('Authorization')
            if token:
                token = token.replace('Bearer ', '')
                data_decoded = jwt.decode(token, app.secret_key, algorithms=['HS256'])
                user_id = data_decoded['user_id']
        except:
            pass
        
        # Determine priority based on issue type
        priority_map = {
            'critical': 'urgent',
            'security': 'urgent',
            'performance': 'high',
            'functionality': 'high',
            'ui_ux': 'medium',
            'content': 'medium',
            'feature_request': 'low',
            'other': 'medium'
        }
        priority = priority_map.get(issue_type, 'medium')
        
        # Create support ticket for issue
        ticket_id = generate_ticket_id()
        
        ticket = SupportTicket(
            ticket_id=ticket_id,
            user_id=user_id,
            email=email,
            name=name,
            subject=f"[ISSUE] {title}",
            message=description,
            category='Technical Issues',
            priority=priority,
            status='open',
            metadata={
                'source': 'issue_report',
                'issue_type': issue_type,
                'page_url': data.get('page_url'),
                'browser_info': user_agent,
                'steps_to_reproduce': data.get('steps_to_reproduce'),
                'expected_behavior': data.get('expected_behavior'),
                'actual_behavior': data.get('actual_behavior'),
                'device_info': data.get('device_info'),
                'screenshot_urls': data.get('screenshot_urls', [])
            },
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        db.session.add(ticket)
        db.session.commit()
        
        # Send confirmation email
        if support_email_service:
            support_email_service.send_support_email(
                'issue_report',
                email,
                ticket_id=ticket_id,
                name=name,
                report_id=ticket_id,
                issue_type=issue_type,
                title=title,
                priority=priority
            )
        
        logger.info(f"Issue reported: {ticket_id} - {issue_type} from {email}")
        
        return jsonify({
            'success': True,
            'message': 'Thank you for reporting this issue! We\'ll investigate and fix it.',
            'ticket_id': ticket_id,
            'priority': priority
        }), 200
        
    except Exception as e:
        logger.error(f"Issue report error: {e}")
        return jsonify({'error': 'Failed to submit issue report'}), 500

@support_bp.route('/api/support/faq', methods=['GET'])
def get_faqs():
    """Get FAQ entries with optional filtering"""
    try:
        category = request.args.get('category')
        featured_only = request.args.get('featured', 'false').lower() == 'true'
        search_query = request.args.get('q', '').strip()
        
        # Build query
        query = FAQ.query.filter_by(is_active=True)
        
        if category:
            query = query.filter(FAQ.category == category)
        
        if featured_only:
            query = query.filter_by(is_featured=True)
        
        if search_query:
            search_pattern = f"%{search_query}%"
            query = query.filter(
                or_(
                    FAQ.question.ilike(search_pattern),
                    FAQ.answer.ilike(search_pattern),
                    FAQ.tags.ilike(search_pattern)
                )
            )
        
        # Order by featured first, then by order, then by view count
        faqs = query.order_by(
            FAQ.is_featured.desc(),
            FAQ.order.asc(),
            FAQ.view_count.desc()
        ).all()
        
        # Group by category
        faq_categories = {}
        for faq in faqs:
            if faq.category not in faq_categories:
                faq_categories[faq.category] = []
            faq_categories[faq.category].append(faq.to_dict())
        
        # Get FAQ statistics
        total_faqs = FAQ.query.filter_by(is_active=True).count()
        featured_faqs = FAQ.query.filter_by(is_active=True, is_featured=True).count()
        
        return jsonify({
            'success': True,
            'faqs': faq_categories,
            'statistics': {
                'total_faqs': total_faqs,
                'featured_faqs': featured_faqs,
                'categories': len(faq_categories)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"FAQ retrieval error: {e}")
        return jsonify({'error': 'Failed to retrieve FAQs'}), 500

@support_bp.route('/api/support/faq/<int:faq_id>/helpful', methods=['POST', 'OPTIONS'])
def rate_faq_helpful(faq_id):
    """Rate FAQ as helpful or not helpful"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        is_helpful = data.get('is_helpful', True)
        
        faq = FAQ.query.get_or_404(faq_id)
        
        # Update view count and helpful count
        faq.view_count += 1
        
        if is_helpful:
            faq.helpful_count += 1
        else:
            faq.not_helpful_count += 1
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your feedback!',
            'helpful_count': faq.helpful_count,
            'not_helpful_count': faq.not_helpful_count
        }), 200
        
    except Exception as e:
        logger.error(f"FAQ rating error: {e}")
        return jsonify({'error': 'Failed to rate FAQ'}), 500

@support_bp.route('/api/support/categories', methods=['GET'])
def get_support_categories():
    """Get support categories"""
    try:
        categories = SupportCategory.query.filter_by(is_active=True).order_by(
            SupportCategory.order.asc()
        ).all()
        
        return jsonify({
            'success': True,
            'categories': [cat.to_dict() for cat in categories]
        }), 200
        
    except Exception as e:
        logger.error(f"Support categories error: {e}")
        return jsonify({'error': 'Failed to retrieve support categories'}), 500

@support_bp.route('/api/support/help-center', methods=['GET'])
def get_help_center():
    """Get help center data including FAQs, categories, and quick links"""
    try:
        # Get featured FAQs
        featured_faqs = FAQ.query.filter_by(
            is_active=True, 
            is_featured=True
        ).order_by(FAQ.order.asc()).limit(6).all()
        
        # Get support categories
        categories = SupportCategory.query.filter_by(
            is_active=True
        ).order_by(SupportCategory.order.asc()).all()
        
        # Get FAQ categories with counts
        faq_categories = db.session.query(
            FAQ.category,
            db.func.count(FAQ.id).label('count')
        ).filter_by(is_active=True).group_by(FAQ.category).all()
        
        # Get recent helpful FAQs
        popular_faqs = FAQ.query.filter_by(is_active=True).order_by(
            FAQ.helpful_count.desc()
        ).limit(5).all()
        
        return jsonify({
            'success': True,
            'help_center': {
                'featured_faqs': [faq.to_dict() for faq in featured_faqs],
                'categories': [cat.to_dict() for cat in categories],
                'faq_categories': [
                    {'name': cat[0], 'count': cat[1]} for cat in faq_categories
                ],
                'popular_faqs': [faq.to_dict() for faq in popular_faqs],
                'quick_links': [
                    {
                        'title': 'Contact Support',
                        'description': 'Get help from our support team',
                        'url': f'{FRONTEND_URL}/support/contact.html',
                        'icon': 'mail'
                    },
                    {
                        'title': 'Report a Bug',
                        'description': 'Found something that\'s not working?',
                        'url': f'{FRONTEND_URL}/support/report-issue.html',
                        'icon': 'bug'
                    },
                    {
                        'title': 'Feature Request',
                        'description': 'Suggest new features',
                        'url': f'{FRONTEND_URL}/support/feedback.html',
                        'icon': 'lightbulb'
                    },
                    {
                        'title': 'FAQ',
                        'description': 'Find answers to common questions',
                        'url': f'{FRONTEND_URL}/support/faq.html',
                        'icon': 'help-circle'
                    }
                ]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Help center error: {e}")
        return jsonify({'error': 'Failed to retrieve help center data'}), 500

@support_bp.route('/api/support/ticket/<ticket_id>', methods=['GET'])
def get_ticket_status(ticket_id):
    """Get support ticket status and details"""
    try:
        ticket = SupportTicket.query.filter_by(ticket_id=ticket_id).first_or_404()
        
        # Get ticket responses
        responses = SupportResponse.query.filter_by(
            ticket_id=ticket.id
        ).order_by(SupportResponse.created_at.asc()).all()
        
        ticket_data = ticket.to_dict()
        ticket_data['responses'] = [response.to_dict() for response in responses]
        
        return jsonify({
            'success': True,
            'ticket': ticket_data
        }), 200
        
    except Exception as e:
        logger.error(f"Ticket status error: {e}")
        return jsonify({'error': 'Ticket not found'}), 404

# === ADMIN SUPPORT ENDPOINTS ===

@support_bp.route('/api/admin/support/tickets', methods=['GET'])
@require_admin
def get_all_tickets(current_user):
    """Get all support tickets for admin"""
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        status = request.args.get('status')
        category = request.args.get('category')
        priority = request.args.get('priority')
        
        # Build query
        query = SupportTicket.query
        
        if status:
            query = query.filter(SupportTicket.status == status)
        if category:
            query = query.filter(SupportTicket.category == category)
        if priority:
            query = query.filter(SupportTicket.priority == priority)
        
        # Paginate results
        tickets = query.order_by(
            SupportTicket.created_at.desc()
        ).paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        # Get statistics
        stats = {
            'total': SupportTicket.query.count(),
            'open': SupportTicket.query.filter_by(status='open').count(),
            'in_progress': SupportTicket.query.filter_by(status='in_progress').count(),
            'resolved': SupportTicket.query.filter_by(status='resolved').count(),
            'closed': SupportTicket.query.filter_by(status='closed').count()
        }
        
        return jsonify({
            'success': True,
            'tickets': [ticket.to_dict() for ticket in tickets.items],
            'pagination': {
                'page': page,
                'pages': tickets.pages,
                'per_page': per_page,
                'total': tickets.total,
                'has_next': tickets.has_next,
                'has_prev': tickets.has_prev
            },
            'statistics': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Admin tickets error: {e}")
        return jsonify({'error': 'Failed to retrieve tickets'}), 500

@support_bp.route('/api/admin/support/ticket/<ticket_id>/respond', methods=['POST'])
@require_admin
def respond_to_ticket(current_user, ticket_id):
    """Respond to a support ticket"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        new_status = data.get('status')
        is_internal = data.get('is_internal', False)
        
        if not message:
            return jsonify({'error': 'Response message is required'}), 400
        
        ticket = SupportTicket.query.filter_by(ticket_id=ticket_id).first_or_404()
        
        # Create response
        response = SupportResponse(
            ticket_id=ticket.id,
            user_id=current_user.id,
            email=current_user.email,
            name=current_user.username,
            message=message,
            is_staff_response=True,
            is_internal=is_internal
        )
        
        db.session.add(response)
        
        # Update ticket status if provided
        if new_status and new_status in ['open', 'in_progress', 'resolved', 'closed']:
            ticket.status = new_status
            ticket.assigned_to = current_user.id
            
            if new_status == 'resolved':
                ticket.resolved_at = datetime.utcnow()
        
        ticket.updated_at = datetime.utcnow()
        db.session.commit()
        
        # Send email to customer if not internal
        if not is_internal and support_email_service:
            support_email_service.send_support_email(
                'ticket_response',
                ticket.email,
                ticket_id=ticket.ticket_id,
                name=ticket.name,
                response_message=message,
                staff_name=current_user.username,
                status=ticket.status
            )
        
        logger.info(f"Admin {current_user.username} responded to ticket {ticket_id}")
        
        return jsonify({
            'success': True,
            'message': 'Response added successfully',
            'response': response.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Ticket response error: {e}")
        return jsonify({'error': 'Failed to add response'}), 500

@support_bp.route('/api/admin/support/feedback', methods=['GET'])
@require_admin
def get_all_feedback(current_user):
    """Get all feedback for admin"""
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        feedback_type = request.args.get('type')
        reviewed_only = request.args.get('reviewed') == 'true'
        
        # Build query
        query = Feedback.query
        
        if feedback_type:
            query = query.filter(Feedback.feedback_type == feedback_type)
        if reviewed_only:
            query = query.filter_by(is_reviewed=True)
        
        # Paginate results
        feedback_list = query.order_by(
            Feedback.created_at.desc()
        ).paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        return jsonify({
            'success': True,
            'feedback': [fb.to_dict() for fb in feedback_list.items],
            'pagination': {
                'page': page,
                'pages': feedback_list.pages,
                'per_page': per_page,
                'total': feedback_list.total
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Admin feedback error: {e}")
        return jsonify({'error': 'Failed to retrieve feedback'}), 500

@support_bp.route('/api/support/health', methods=['GET'])
def support_health():
    """Check support service health"""
    try:
        # Test database connection
        SupportTicket.query.limit(1).first()
        
        # Test Redis connection
        redis_status = 'not_configured'
        if redis_client:
            try:
                redis_client.ping()
                redis_status = 'connected'
            except:
                redis_status = 'disconnected'
        
        # Check email service
        email_configured = support_email_service is not None
        
        # Get statistics
        stats = {
            'total_tickets': SupportTicket.query.count(),
            'open_tickets': SupportTicket.query.filter_by(status='open').count(),
            'total_feedback': Feedback.query.count(),
            'total_faqs': FAQ.query.filter_by(is_active=True).count()
        }
        
        return jsonify({
            'status': 'healthy',
            'service': 'support',
            'email_service': 'Gmail SMTP',
            'email_configured': email_configured,
            'redis_status': redis_status,
            'statistics': stats,
            'features': [
                'contact_form',
                'feedback_system',
                'issue_reporting',
                'faq_management',
                'ticket_system',
                'email_notifications',
                'admin_dashboard',
                'rate_limiting'
            ],
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'support',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# CORS headers for all responses
@support_bp.after_request
def after_request(response):
    """Add CORS headers to responses"""
    origin = request.headers.get('Origin')
    allowed_origins = [FRONTEND_URL, 'http://127.0.0.1:5500', 'http://127.0.0.1:5501']
    
    if origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    
    return response

# Export everything needed
__all__ = [
    'support_bp',
    'init_support',
    'SupportTicket',
    'SupportResponse',
    'FAQ',
    'Feedback',
    'SupportCategory'
]