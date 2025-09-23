# backend/services/admin.py
from flask import Blueprint, request, jsonify
from flask_caching import Cache
from datetime import datetime, timedelta
import json
import logging
import time
import telebot
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate
from sqlalchemy import func, desc, and_, or_, text
from collections import defaultdict
from functools import wraps
import jwt
import os
import uuid
import redis
from urllib.parse import urlparse
import enum
from typing import Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

admin_bp = Blueprint('admin', __name__)

logger = logging.getLogger(__name__)

cache = Cache()

# Configuration
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '-1002850793757')
ADMIN_TELEGRAM_CHAT_ID = os.environ.get('ADMIN_TELEGRAM_CHAT_ID', '-1002850793757')
REDIS_URL = os.environ.get('REDIS_URL', 'redis://red-d2qlbuje5dus73c71qog:xp7inVzgblGCbo9I4taSGLdKUg0xY91I@red-d2qlbuje5dus73c71qog:6379')

# Email Configuration
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'srinathnulidonda.dev@gmail.com')
SUPPORT_EMAIL = os.environ.get('SUPPORT_EMAIL', 'support@cinebrain.com')
GMAIL_USERNAME = os.environ.get('GMAIL_USERNAME', 'projects.srinath@gmail.com')
GMAIL_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', 'wuus nsow nbee xewv')

# Initialize services
if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != 'your_telegram_bot_token':
    try:
        bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
    except:
        bot = None
        logging.warning("Failed to initialize Telegram bot")
else:
    bot = None

# Redis client for real-time notifications
redis_client = None
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
    logger.info("Admin Redis connected successfully")
except Exception as e:
    logger.error(f"Admin Redis connection failed: {e}")
    redis_client = None

# Global variables for dependency injection
db = None
User = None
Content = None
UserInteraction = None
AdminRecommendation = None
TMDBService = None
JikanService = None
ContentService = None
http_session = None
cache = None
app = None

# Support Models (imported from support service)
SupportCategory = None
SupportTicket = None
SupportResponse = None
FAQ = None
Feedback = None
TicketActivity = None

# Enums
class NotificationPriority(enum.Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class AdminRole(enum.Enum):
    SUPER_ADMIN = "super_admin"
    SUPPORT_MANAGER = "support_manager"
    SUPPORT_AGENT = "support_agent"
    CONTENT_MODERATOR = "content_moderator"

def init_admin(flask_app, database, models, services):
    global db, User, Content, UserInteraction, AdminRecommendation
    global TMDBService, JikanService, ContentService
    global http_session, cache, app
    global SupportCategory, SupportTicket, SupportResponse, FAQ, Feedback, TicketActivity
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    AdminRecommendation = models['AdminRecommendation']
    
    # Support models
    SupportCategory = models.get('SupportCategory')
    SupportTicket = models.get('SupportTicket')
    SupportResponse = models.get('SupportResponse')
    FAQ = models.get('FAQ')
    Feedback = models.get('Feedback')
    TicketActivity = models.get('TicketActivity')
    
    TMDBService = services['TMDBService']
    JikanService = services['JikanService']
    ContentService = services['ContentService']
    http_session = services['http_session']
    cache = services['cache']
    
    # Initialize notification services
    AdminNotificationService.initialize()
    AdminEmailService.initialize()
    
    logger.info("✅ Enhanced Admin service with support integration initialized")

def require_admin(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user or not current_user.is_admin:
                return jsonify({'error': 'Admin access required'}), 403
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

def require_support_access(f):
    @wraps(f)
    def decorated_function(current_user, *args, **kwargs):
        # For now, all admins have support access
        # In future, implement role-based access control
        return f(current_user, *args, **kwargs)
    return decorated_function

class AdminNotificationService:
    @staticmethod
    def initialize():
        """Initialize notification workers"""
        if redis_client:
            threading.Thread(target=AdminNotificationService._notification_worker, daemon=True).start()
            logger.info("Admin notification worker started")
    
    @staticmethod
    def _notification_worker():
        """Background worker for processing notifications"""
        while True:
            try:
                if redis_client:
                    notification_data = redis_client.blpop('admin_notifications', timeout=1)
                    if notification_data:
                        notification = json.loads(notification_data[1])
                        AdminNotificationService._process_notification(notification)
                else:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Notification worker error: {e}")
                time.sleep(5)
    
    @staticmethod
    def _process_notification(notification: Dict):
        """Process individual notification"""
        try:
            # Send Telegram notification
            if bot and notification.get('telegram', True):
                AdminNotificationService._send_telegram_notification(notification)
            
            # Send email notification
            if notification.get('email', True):
                AdminEmailService.send_admin_notification(notification)
            
            # Store in database for dashboard
            AdminNotificationService._store_dashboard_notification(notification)
            
        except Exception as e:
            logger.error(f"Error processing notification: {e}")
    
    @staticmethod
    def _send_telegram_notification(notification: Dict):
        """Send Telegram notification to admin"""
        try:
            if not bot or not ADMIN_TELEGRAM_CHAT_ID:
                return
            
            priority_emoji = {
                'low': '🔵',
                'normal': '🟡',
                'high': '🟠',
                'urgent': '🔴'
            }
            
            emoji = priority_emoji.get(notification.get('priority', 'normal'), '🔵')
            
            message = f"""{emoji} **CineBrain Admin Alert**

**{notification['title']}**

{notification['message']}

**Type:** {notification.get('type', 'general').title()}
**Priority:** {notification.get('priority', 'normal').title()}
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#{notification.get('type', 'general').replace('_', '')}"""
            
            bot.send_message(ADMIN_TELEGRAM_CHAT_ID, message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Telegram notification error: {e}")
    
    @staticmethod
    def _store_dashboard_notification(notification: Dict):
        """Store notification for admin dashboard"""
        try:
            if redis_client:
                notification['id'] = str(uuid.uuid4())
                notification['timestamp'] = datetime.utcnow().isoformat()
                notification['read'] = False
                
                redis_client.lpush('dashboard_notifications', json.dumps(notification))
                redis_client.ltrim('dashboard_notifications', 0, 99)  # Keep last 100
                
        except Exception as e:
            logger.error(f"Dashboard notification storage error: {e}")
    
    @staticmethod
    def send_notification(title: str, message: str, notification_type: str = 'general', 
                         priority: str = 'normal', **kwargs):
        """Send notification to admins"""
        try:
            notification = {
                'title': title,
                'message': message,
                'type': notification_type,
                'priority': priority,
                'created_at': datetime.utcnow().isoformat(),
                **kwargs
            }
            
            if redis_client:
                redis_client.rpush('admin_notifications', json.dumps(notification))
            else:
                # Direct processing if Redis unavailable
                AdminNotificationService._process_notification(notification)
                
        except Exception as e:
            logger.error(f"Send notification error: {e}")

class AdminEmailService:
    smtp_server = None
    
    @staticmethod
    def initialize():
        """Initialize email service"""
        threading.Thread(target=AdminEmailService._email_worker, daemon=True).start()
        logger.info("Admin email worker started")
    
    @staticmethod
    def _email_worker():
        """Background email worker"""
        while True:
            try:
                if redis_client:
                    email_data = redis_client.blpop('admin_emails', timeout=1)
                    if email_data:
                        email = json.loads(email_data[1])
                        AdminEmailService._send_email_smtp(email)
                else:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Admin email worker error: {e}")
                time.sleep(5)
    
    @staticmethod
    def _send_email_smtp(email_data: Dict):
        """Send email via SMTP"""
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = formataddr(('CineBrain Admin', GMAIL_USERNAME))
            msg['To'] = email_data['to']
            msg['Subject'] = email_data['subject']
            msg['Date'] = formatdate(localtime=True)
            
            if email_data.get('html'):
                html_part = MIMEText(email_data['html'], 'html', 'utf-8')
                msg.attach(html_part)
            
            if email_data.get('text'):
                text_part = MIMEText(email_data['text'], 'plain', 'utf-8')
                msg.attach(text_part)
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(GMAIL_USERNAME, GMAIL_PASSWORD)
                server.send_message(msg)
            
            logger.info(f"Admin email sent to {email_data['to']}")
            
        except Exception as e:
            logger.error(f"Admin email send error: {e}")
    
    @staticmethod
    def send_admin_notification(notification: Dict):
        """Send email notification to admin"""
        try:
            subject = f"[CineBrain Admin] {notification['title']}"
            
            html = f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #1a73e8;">{notification['title']}</h2>
                <p><strong>Priority:</strong> <span style="color: {'#ea4335' if notification.get('priority') == 'urgent' else '#34a853'};">{notification.get('priority', 'normal').title()}</span></p>
                <p><strong>Type:</strong> {notification.get('type', 'general').title()}</p>
                <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #1a73e8; margin: 15px 0;">
                    {notification['message']}
                </div>
                <p><small>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            </div>
            """
            
            text = f"""
{notification['title']}

Priority: {notification.get('priority', 'normal').title()}
Type: {notification.get('type', 'general').title()}

{notification['message']}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            email_data = {
                'to': ADMIN_EMAIL,
                'subject': subject,
                'html': html,
                'text': text
            }
            
            if redis_client:
                redis_client.rpush('admin_emails', json.dumps(email_data))
            else:
                AdminEmailService._send_email_smtp(email_data)
                
        except Exception as e:
            logger.error(f"Admin notification email error: {e}")

class SupportAnalytics:
    @staticmethod
    def get_support_metrics():
        """Get comprehensive support metrics"""
        try:
            # Ticket metrics
            total_tickets = SupportTicket.query.count()
            open_tickets = SupportTicket.query.filter_by(status='open').count()
            pending_tickets = SupportTicket.query.filter_by(status='in_progress').count()
            resolved_tickets = SupportTicket.query.filter_by(status='resolved').count()
            
            # Today's metrics
            today = datetime.utcnow().date()
            tickets_today = SupportTicket.query.filter(
                func.date(SupportTicket.created_at) == today
            ).count()
            
            # SLA metrics
            sla_breached = SupportTicket.query.filter_by(sla_breached=True).count()
            overdue_tickets = SupportTicket.query.filter(
                and_(
                    SupportTicket.sla_deadline < datetime.utcnow(),
                    SupportTicket.status.in_(['open', 'in_progress'])
                )
            ).count()
            
            # Response time metrics
            avg_response_time = db.session.query(
                func.avg(
                    func.extract('epoch', SupportTicket.first_response_at) - 
                    func.extract('epoch', SupportTicket.created_at)
                )
            ).filter(SupportTicket.first_response_at.isnot(None)).scalar()
            
            # Category distribution
            category_stats = db.session.query(
                SupportCategory.name,
                func.count(SupportTicket.id).label('count')
            ).join(SupportTicket).group_by(SupportCategory.name).all()
            
            # Priority distribution
            priority_stats = db.session.query(
                SupportTicket.priority,
                func.count(SupportTicket.id).label('count')
            ).group_by(SupportTicket.priority).all()
            
            # Feedback metrics
            total_feedback = Feedback.query.count()
            unread_feedback = Feedback.query.filter_by(is_read=False).count()
            
            # FAQ metrics
            total_faqs = FAQ.query.filter_by(is_published=True).count()
            most_viewed_faqs = FAQ.query.order_by(desc(FAQ.view_count)).limit(5).all()
            
            return {
                'tickets': {
                    'total': total_tickets,
                    'open': open_tickets,
                    'in_progress': pending_tickets,
                    'resolved': resolved_tickets,
                    'today': tickets_today,
                    'sla_breached': sla_breached,
                    'overdue': overdue_tickets
                },
                'performance': {
                    'avg_response_time_hours': round(avg_response_time / 3600, 2) if avg_response_time else 0,
                    'sla_compliance_rate': round((1 - sla_breached / total_tickets) * 100, 2) if total_tickets > 0 else 100
                },
                'distribution': {
                    'by_category': [{'name': cat.name, 'count': cat.count} for cat in category_stats],
                    'by_priority': [{'priority': str(pri.priority), 'count': pri.count} for pri in priority_stats]
                },
                'feedback': {
                    'total': total_feedback,
                    'unread': unread_feedback
                },
                'faq': {
                    'total': total_faqs,
                    'most_viewed': [
                        {'question': faq.question, 'views': faq.view_count}
                        for faq in most_viewed_faqs
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Support analytics error: {e}")
            return {}

class TelegramService:
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

#AdminChoice #MovieRecommendation #CineBrain"""
            
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='Markdown'
                    )
                except Exception as photo_error:
                    logger.error(f"Failed to send photo, sending text only: {photo_error}")
                    bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            else:
                bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

# ========== EXISTING ENDPOINTS ==========

@admin_bp.route('/api/admin/search', methods=['GET'])
@require_admin
def admin_search(current_user):
    try:
        query = request.args.get('query', '')
        source = request.args.get('source', 'tmdb')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        results = []
        
        if source == 'tmdb':
            tmdb_results = TMDBService.search_content(query, page=page)
            if tmdb_results:
                for item in tmdb_results.get('results', []):
                    results.append({
                        'id': item['id'],
                        'title': item.get('title') or item.get('name'),
                        'content_type': 'movie' if 'title' in item else 'tv',
                        'release_date': item.get('release_date') or item.get('first_air_date'),
                        'poster_path': f"https://image.tmdb.org/t/p/w300{item['poster_path']}" if item.get('poster_path') else None,
                        'overview': item.get('overview'),
                        'rating': item.get('vote_average'),
                        'source': 'tmdb'
                    })
        
        elif source == 'anime':
            anime_results = JikanService.search_anime(query, page=page)
            if anime_results:
                for anime in anime_results.get('data', []):
                    results.append({
                        'id': anime['mal_id'],
                        'title': anime.get('title'),
                        'content_type': 'anime',
                        'release_date': anime.get('aired', {}).get('from'),
                        'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                        'overview': anime.get('synopsis'),
                        'rating': anime.get('score'),
                        'source': 'anime'
                    })
        
        return jsonify({'results': results}), 200
        
    except Exception as e:
        logger.error(f"Admin search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@admin_bp.route('/api/admin/content', methods=['POST'])
@require_admin
def save_external_content(current_user):
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No content data provided'}), 400
        
        existing_content = None
        if data.get('source') == 'anime' and data.get('id'):
            existing_content = Content.query.filter_by(mal_id=data['id']).first()
        elif data.get('id'):
            existing_content = Content.query.filter_by(tmdb_id=data['id']).first()
        
        if existing_content:
            return jsonify({
                'message': 'Content already exists',
                'content_id': existing_content.id
            }), 200
        
        try:
            release_date = None
            if data.get('release_date'):
                try:
                    release_date = datetime.strptime(data['release_date'][:10], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            youtube_trailer_id = ContentService.get_youtube_trailer(data.get('title'), data.get('content_type')) if ContentService else None
            
            if data.get('source') == 'anime':
                content = Content(
                    mal_id=data.get('id'),
                    title=data.get('title'),
                    original_title=data.get('original_title'),
                    content_type='anime',
                    genres=json.dumps(data.get('genres', [])),
                    anime_genres=json.dumps([]),
                    languages=json.dumps(['japanese']),
                    release_date=release_date,
                    rating=data.get('rating'),
                    overview=data.get('overview'),
                    poster_path=data.get('poster_path'),
                    youtube_trailer_id=youtube_trailer_id
                )
            else:
                content = Content(
                    tmdb_id=data.get('id'),
                    title=data.get('title'),
                    original_title=data.get('original_title'),
                    content_type=data.get('content_type', 'movie'),
                    genres=json.dumps(data.get('genres', [])),
                    languages=json.dumps(data.get('languages', ['en'])),
                    release_date=release_date,
                    runtime=data.get('runtime'),
                    rating=data.get('rating'),
                    vote_count=data.get('vote_count'),
                    popularity=data.get('popularity'),
                    overview=data.get('overview'),
                    poster_path=data.get('poster_path'),
                    backdrop_path=data.get('backdrop_path'),
                    youtube_trailer_id=youtube_trailer_id
                )
            
            content.ensure_slug()
            
            db.session.add(content)
            db.session.commit()
            
            # Send notification
            AdminNotificationService.send_notification(
                title="New Content Added",
                message=f"Admin {current_user.username} added new content: {content.title}",
                notification_type="content_added",
                priority="normal",
                content_id=content.id,
                admin_id=current_user.id
            )
            
            return jsonify({
                'message': 'Content saved successfully',
                'content_id': content.id
            }), 201
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving content: {e}")
            return jsonify({'error': 'Failed to save content to database'}), 500
        
    except Exception as e:
        logger.error(f"Save content error: {e}")
        return jsonify({'error': 'Failed to process content'}), 500

@admin_bp.route('/api/admin/recommendations', methods=['POST'])
@require_admin
def create_admin_recommendation(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'recommendation_type', 'description']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        content = Content.query.get(data['content_id'])
        if not content:
            content = Content.query.filter_by(tmdb_id=data['content_id']).first()
        
        if not content:
            return jsonify({'error': 'Content not found. Please save content first.'}), 404
        
        admin_rec = AdminRecommendation(
            content_id=content.id,
            admin_id=current_user.id,
            recommendation_type=data['recommendation_type'],
            description=data['description']
        )
        
        db.session.add(admin_rec)
        db.session.commit()
        
        telegram_success = TelegramService.send_admin_recommendation(content, current_user.username, data['description'])
        
        # Send notification
        AdminNotificationService.send_notification(
            title="New Admin Recommendation",
            message=f"Admin {current_user.username} recommended: {content.title}",
            notification_type="recommendation_created",
            priority="normal",
            content_id=content.id,
            admin_id=current_user.id
        )
        
        return jsonify({
            'message': 'Admin recommendation created successfully',
            'telegram_sent': telegram_success
        }), 201
        
    except Exception as e:
        logger.error(f"Admin recommendation error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create recommendation'}), 500

# ========== SUPPORT MANAGEMENT ENDPOINTS ==========

@admin_bp.route('/api/admin/support/dashboard', methods=['GET'])
@require_admin
@require_support_access
def get_support_dashboard(current_user):
    """Comprehensive support dashboard with real-time metrics"""
    try:
        metrics = SupportAnalytics.get_support_metrics()
        
        # Recent tickets requiring attention
        urgent_tickets = SupportTicket.query.filter(
            or_(
                SupportTicket.priority == 'urgent',
                SupportTicket.sla_deadline < datetime.utcnow()
            )
        ).order_by(SupportTicket.created_at.desc()).limit(10).all()
        
        # Unassigned tickets
        unassigned_tickets = SupportTicket.query.filter(
            and_(
                SupportTicket.assigned_to.is_(None),
                SupportTicket.status.in_(['open', 'in_progress'])
            )
        ).count()
        
        # Recent feedback
        recent_feedback = Feedback.query.filter_by(is_read=False)\
            .order_by(Feedback.created_at.desc()).limit(5).all()
        
        return jsonify({
            'metrics': metrics,
            'urgent_tickets': [
                {
                    'id': ticket.id,
                    'ticket_number': ticket.ticket_number,
                    'subject': ticket.subject,
                    'priority': ticket.priority.value,
                    'status': ticket.status.value,
                    'created_at': ticket.created_at.isoformat(),
                    'sla_deadline': ticket.sla_deadline.isoformat() if ticket.sla_deadline else None,
                    'user_name': ticket.user_name,
                    'category': ticket.category.name if ticket.category else None
                }
                for ticket in urgent_tickets
            ],
            'unassigned_tickets': unassigned_tickets,
            'recent_feedback': [
                {
                    'id': feedback.id,
                    'subject': feedback.subject,
                    'user_name': feedback.user_name,
                    'feedback_type': feedback.feedback_type.value,
                    'created_at': feedback.created_at.isoformat()
                }
                for feedback in recent_feedback
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Support dashboard error: {e}")
        return jsonify({'error': 'Failed to load support dashboard'}), 500

@admin_bp.route('/api/admin/support/tickets', methods=['GET'])
@require_admin
@require_support_access
def get_support_tickets(current_user):
    """Get paginated support tickets with advanced filtering"""
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        status = request.args.get('status')
        priority = request.args.get('priority')
        category_id = request.args.get('category_id', type=int)
        assigned_to = request.args.get('assigned_to', type=int)
        search = request.args.get('search', '').strip()
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')
        
        query = SupportTicket.query
        
        # Apply filters
        if status:
            query = query.filter(SupportTicket.status == status)
        
        if priority:
            query = query.filter(SupportTicket.priority == priority)
        
        if category_id:
            query = query.filter(SupportTicket.category_id == category_id)
        
        if assigned_to:
            query = query.filter(SupportTicket.assigned_to == assigned_to)
        
        if search:
            query = query.filter(
                or_(
                    SupportTicket.ticket_number.contains(search),
                    SupportTicket.subject.contains(search),
                    SupportTicket.user_name.contains(search),
                    SupportTicket.user_email.contains(search)
                )
            )
        
        # Apply sorting
        if sort_order == 'desc':
            query = query.order_by(desc(getattr(SupportTicket, sort_by)))
        else:
            query = query.order_by(getattr(SupportTicket, sort_by))
        
        tickets = query.paginate(page=page, per_page=per_page, error_out=False)
        
        result = []
        for ticket in tickets.items:
            assigned_admin = User.query.get(ticket.assigned_to) if ticket.assigned_to else None
            response_count = SupportResponse.query.filter_by(ticket_id=ticket.id).count()
            
            result.append({
                'id': ticket.id,
                'ticket_number': ticket.ticket_number,
                'subject': ticket.subject,
                'description': ticket.description,
                'user_name': ticket.user_name,
                'user_email': ticket.user_email,
                'status': ticket.status.value,
                'priority': ticket.priority.value,
                'ticket_type': ticket.ticket_type.value,
                'category': {
                    'id': ticket.category.id,
                    'name': ticket.category.name,
                    'icon': ticket.category.icon
                } if ticket.category else None,
                'assigned_to': {
                    'id': assigned_admin.id,
                    'username': assigned_admin.username
                } if assigned_admin else None,
                'created_at': ticket.created_at.isoformat(),
                'first_response_at': ticket.first_response_at.isoformat() if ticket.first_response_at else None,
                'resolved_at': ticket.resolved_at.isoformat() if ticket.resolved_at else None,
                'sla_deadline': ticket.sla_deadline.isoformat() if ticket.sla_deadline else None,
                'sla_breached': ticket.sla_breached,
                'response_count': response_count,
                'browser_info': ticket.browser_info,
                'page_url': ticket.page_url
            })
        
        return jsonify({
            'tickets': result,
            'total': tickets.total,
            'pages': tickets.pages,
            'current_page': page,
            'per_page': per_page
        }), 200
        
    except Exception as e:
        logger.error(f"Get support tickets error: {e}")
        return jsonify({'error': 'Failed to get support tickets'}), 500

@admin_bp.route('/api/admin/support/tickets/<int:ticket_id>', methods=['GET'])
@require_admin
@require_support_access
def get_ticket_details(current_user, ticket_id):
    """Get detailed ticket information with full conversation"""
    try:
        ticket = SupportTicket.query.get_or_404(ticket_id)
        
        # Get all responses
        responses = SupportResponse.query.filter_by(ticket_id=ticket_id)\
            .order_by(SupportResponse.created_at.asc()).all()
        
        # Get ticket activities
        activities = TicketActivity.query.filter_by(ticket_id=ticket_id)\
            .order_by(TicketActivity.created_at.desc()).all()
        
        assigned_admin = User.query.get(ticket.assigned_to) if ticket.assigned_to else None
        
        return jsonify({
            'ticket': {
                'id': ticket.id,
                'ticket_number': ticket.ticket_number,
                'subject': ticket.subject,
                'description': ticket.description,
                'user_name': ticket.user_name,
                'user_email': ticket.user_email,
                'status': ticket.status.value,
                'priority': ticket.priority.value,
                'ticket_type': ticket.ticket_type.value,
                'category': {
                    'id': ticket.category.id,
                    'name': ticket.category.name,
                    'icon': ticket.category.icon
                } if ticket.category else None,
                'assigned_to': {
                    'id': assigned_admin.id,
                    'username': assigned_admin.username
                } if assigned_admin else None,
                'created_at': ticket.created_at.isoformat(),
                'first_response_at': ticket.first_response_at.isoformat() if ticket.first_response_at else None,
                'resolved_at': ticket.resolved_at.isoformat() if ticket.resolved_at else None,
                'sla_deadline': ticket.sla_deadline.isoformat() if ticket.sla_deadline else None,
                'sla_breached': ticket.sla_breached,
                'browser_info': ticket.browser_info,
                'device_info': ticket.device_info,
                'page_url': ticket.page_url,
                'ip_address': ticket.ip_address
            },
            'responses': [
                {
                    'id': response.id,
                    'message': response.message,
                    'is_from_staff': response.is_from_staff,
                    'staff_name': response.staff_name,
                    'created_at': response.created_at.isoformat()
                }
                for response in responses
            ],
            'activities': [
                {
                    'id': activity.id,
                    'action': activity.action,
                    'description': activity.description,
                    'actor_name': activity.actor_name,
                    'created_at': activity.created_at.isoformat()
                }
                for activity in activities
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Get ticket details error: {e}")
        return jsonify({'error': 'Failed to get ticket details'}), 500

@admin_bp.route('/api/admin/support/tickets/<int:ticket_id>/respond', methods=['POST'])
@require_admin
@require_support_access
def respond_to_ticket(current_user, ticket_id):
    """Add response to support ticket"""
    try:
        ticket = SupportTicket.query.get_or_404(ticket_id)
        data = request.get_json()
        
        if not data.get('message'):
            return jsonify({'error': 'Message is required'}), 400
        
        # Create response
        response = SupportResponse(
            ticket_id=ticket_id,
            message=data['message'],
            is_from_staff=True,
            staff_id=current_user.id,
            staff_name=current_user.username
        )
        
        # Update ticket
        if not ticket.first_response_at:
            ticket.first_response_at = datetime.utcnow()
        
        if not ticket.assigned_to:
            ticket.assigned_to = current_user.id
        
        # Update status if specified
        new_status = data.get('status')
        if new_status and new_status != ticket.status.value:
            old_status = ticket.status.value
            ticket.status = new_status
            
            if new_status == 'resolved':
                ticket.resolved_at = datetime.utcnow()
            elif new_status == 'closed':
                ticket.closed_at = datetime.utcnow()
            
            # Log status change
            activity = TicketActivity(
                ticket_id=ticket_id,
                action='status_changed',
                description=f'Status changed from {old_status} to {new_status}',
                old_value=old_status,
                new_value=new_status,
                actor_type='staff',
                actor_id=current_user.id,
                actor_name=current_user.username
            )
            db.session.add(activity)
        
        # Log response activity
        response_activity = TicketActivity(
            ticket_id=ticket_id,
            action='response_added',
            description=f'Response added by {current_user.username}',
            actor_type='staff',
            actor_id=current_user.id,
            actor_name=current_user.username
        )
        
        db.session.add(response)
        db.session.add(response_activity)
        db.session.commit()
        
        # Send email notification to user
        # This would integrate with the support email service
        
        # Send admin notification
        AdminNotificationService.send_notification(
            title="Ticket Response Added",
            message=f"Admin {current_user.username} responded to ticket #{ticket.ticket_number}",
            notification_type="ticket_response",
            priority="normal",
            ticket_id=ticket.id,
            admin_id=current_user.id
        )
        
        return jsonify({
            'message': 'Response added successfully',
            'response_id': response.id
        }), 201
        
    except Exception as e:
        logger.error(f"Respond to ticket error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to add response'}), 500

@admin_bp.route('/api/admin/support/tickets/<int:ticket_id>/assign', methods=['POST'])
@require_admin
@require_support_access
def assign_ticket(current_user, ticket_id):
    """Assign ticket to admin"""
    try:
        ticket = SupportTicket.query.get_or_404(ticket_id)
        data = request.get_json()
        
        admin_id = data.get('admin_id')
        if admin_id:
            admin = User.query.filter_by(id=admin_id, is_admin=True).first()
            if not admin:
                return jsonify({'error': 'Admin not found'}), 404
        else:
            admin = None
            admin_id = None
        
        old_assigned = User.query.get(ticket.assigned_to) if ticket.assigned_to else None
        ticket.assigned_to = admin_id
        
        # Log assignment activity
        activity = TicketActivity(
            ticket_id=ticket_id,
            action='assigned',
            description=f'Ticket assigned to {admin.username if admin else "unassigned"}',
            old_value=old_assigned.username if old_assigned else None,
            new_value=admin.username if admin else None,
            actor_type='staff',
            actor_id=current_user.id,
            actor_name=current_user.username
        )
        
        db.session.add(activity)
        db.session.commit()
        
        # Send notification
        AdminNotificationService.send_notification(
            title="Ticket Assigned",
            message=f"Ticket #{ticket.ticket_number} assigned to {admin.username if admin else 'unassigned'}",
            notification_type="ticket_assigned",
            priority="normal",
            ticket_id=ticket.id,
            admin_id=current_user.id
        )
        
        return jsonify({'message': 'Ticket assigned successfully'}), 200
        
    except Exception as e:
        logger.error(f"Assign ticket error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to assign ticket'}), 500

@admin_bp.route('/api/admin/support/tickets/bulk', methods=['POST'])
@require_admin
@require_support_access
def bulk_ticket_operations(current_user):
    """Perform bulk operations on tickets"""
    try:
        data = request.get_json()
        ticket_ids = data.get('ticket_ids', [])
        operation = data.get('operation')
        
        if not ticket_ids or not operation:
            return jsonify({'error': 'Ticket IDs and operation are required'}), 400
        
        tickets = SupportTicket.query.filter(SupportTicket.id.in_(ticket_ids)).all()
        
        if not tickets:
            return jsonify({'error': 'No tickets found'}), 404
        
        success_count = 0
        
        for ticket in tickets:
            try:
                if operation == 'assign':
                    admin_id = data.get('admin_id')
                    ticket.assigned_to = admin_id
                    
                elif operation == 'change_status':
                    new_status = data.get('status')
                    ticket.status = new_status
                    
                    if new_status == 'resolved':
                        ticket.resolved_at = datetime.utcnow()
                    elif new_status == 'closed':
                        ticket.closed_at = datetime.utcnow()
                
                elif operation == 'change_priority':
                    new_priority = data.get('priority')
                    ticket.priority = new_priority
                
                # Log activity
                activity = TicketActivity(
                    ticket_id=ticket.id,
                    action=f'bulk_{operation}',
                    description=f'Bulk operation: {operation}',
                    actor_type='staff',
                    actor_id=current_user.id,
                    actor_name=current_user.username
                )
                db.session.add(activity)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Bulk operation error for ticket {ticket.id}: {e}")
                continue
        
        db.session.commit()
        
        # Send notification
        AdminNotificationService.send_notification(
            title="Bulk Operation Completed",
            message=f"Admin {current_user.username} performed bulk {operation} on {success_count} tickets",
            notification_type="bulk_operation",
            priority="normal",
            admin_id=current_user.id
        )
        
        return jsonify({
            'message': f'Bulk operation completed successfully on {success_count} tickets',
            'success_count': success_count
        }), 200
        
    except Exception as e:
        logger.error(f"Bulk ticket operations error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to perform bulk operations'}), 500

@admin_bp.route('/api/admin/support/faq', methods=['GET'])
@require_admin
@require_support_access
def get_admin_faqs(current_user):
    """Get all FAQs for admin management"""
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        category_id = request.args.get('category_id', type=int)
        search = request.args.get('search', '').strip()
        published_only = request.args.get('published_only', 'false').lower() == 'true'
        
        query = FAQ.query
        
        if category_id:
            query = query.filter(FAQ.category_id == category_id)
        
        if search:
            query = query.filter(
                or_(
                    FAQ.question.contains(search),
                    FAQ.answer.contains(search)
                )
            )
        
        if published_only:
            query = query.filter(FAQ.is_published == True)
        
        faqs = query.order_by(FAQ.sort_order, FAQ.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        result = []
        for faq in faqs.items:
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
                'sort_order': faq.sort_order,
                'is_featured': faq.is_featured,
                'is_published': faq.is_published,
                'view_count': faq.view_count,
                'helpful_count': faq.helpful_count,
                'not_helpful_count': faq.not_helpful_count,
                'created_at': faq.created_at.isoformat(),
                'updated_at': faq.updated_at.isoformat() if faq.updated_at else None
            })
        
        return jsonify({
            'faqs': result,
            'total': faqs.total,
            'pages': faqs.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Get admin FAQs error: {e}")
        return jsonify({'error': 'Failed to get FAQs'}), 500

@admin_bp.route('/api/admin/support/faq', methods=['POST'])
@require_admin
@require_support_access
def create_faq(current_user):
    """Create new FAQ"""
    try:
        data = request.get_json()
        
        required_fields = ['question', 'answer', 'category_id']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        faq = FAQ(
            question=data['question'],
            answer=data['answer'],
            category_id=data['category_id'],
            tags=json.dumps(data.get('tags', [])),
            sort_order=data.get('sort_order', 0),
            is_featured=data.get('is_featured', False),
            is_published=data.get('is_published', True)
        )
        
        db.session.add(faq)
        db.session.commit()
        
        # Send notification
        AdminNotificationService.send_notification(
            title="New FAQ Created",
            message=f"Admin {current_user.username} created a new FAQ: {faq.question[:50]}...",
            notification_type="faq_created",
            priority="normal",
            faq_id=faq.id,
            admin_id=current_user.id
        )
        
        return jsonify({
            'message': 'FAQ created successfully',
            'faq_id': faq.id
        }), 201
        
    except Exception as e:
        logger.error(f"Create FAQ error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create FAQ'}), 500

@admin_bp.route('/api/admin/support/faq/<int:faq_id>', methods=['PUT'])
@require_admin
@require_support_access
def update_faq(current_user, faq_id):
    """Update existing FAQ"""
    try:
        faq = FAQ.query.get_or_404(faq_id)
        data = request.get_json()
        
        faq.question = data.get('question', faq.question)
        faq.answer = data.get('answer', faq.answer)
        faq.category_id = data.get('category_id', faq.category_id)
        faq.tags = json.dumps(data.get('tags', json.loads(faq.tags or '[]')))
        faq.sort_order = data.get('sort_order', faq.sort_order)
        faq.is_featured = data.get('is_featured', faq.is_featured)
        faq.is_published = data.get('is_published', faq.is_published)
        faq.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        # Send notification
        AdminNotificationService.send_notification(
            title="FAQ Updated",
            message=f"Admin {current_user.username} updated FAQ: {faq.question[:50]}...",
            notification_type="faq_updated",
            priority="normal",
            faq_id=faq.id,
            admin_id=current_user.id
        )
        
        return jsonify({'message': 'FAQ updated successfully'}), 200
        
    except Exception as e:
        logger.error(f"Update FAQ error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to update FAQ'}), 500

@admin_bp.route('/api/admin/support/faq/<int:faq_id>', methods=['DELETE'])
@require_admin
@require_support_access
def delete_faq(current_user, faq_id):
    """Delete FAQ"""
    try:
        faq = FAQ.query.get_or_404(faq_id)
        
        db.session.delete(faq)
        db.session.commit()
        
        # Send notification
        AdminNotificationService.send_notification(
            title="FAQ Deleted",
            message=f"Admin {current_user.username} deleted FAQ: {faq.question[:50]}...",
            notification_type="faq_deleted",
            priority="normal",
            admin_id=current_user.id
        )
        
        return jsonify({'message': 'FAQ deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Delete FAQ error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to delete FAQ'}), 500

@admin_bp.route('/api/admin/support/feedback', methods=['GET'])
@require_admin
@require_support_access
def get_admin_feedback(current_user):
    """Get all feedback for admin review"""
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        feedback_type = request.args.get('feedback_type')
        is_read = request.args.get('is_read')
        search = request.args.get('search', '').strip()
        
        query = Feedback.query
        
        if feedback_type:
            query = query.filter(Feedback.feedback_type == feedback_type)
        
        if is_read is not None:
            query = query.filter(Feedback.is_read == (is_read.lower() == 'true'))
        
        if search:
            query = query.filter(
                or_(
                    Feedback.subject.contains(search),
                    Feedback.message.contains(search),
                    Feedback.user_name.contains(search),
                    Feedback.user_email.contains(search)
                )
            )
        
        feedback_items = query.order_by(Feedback.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        result = []
        for feedback in feedback_items.items:
            result.append({
                'id': feedback.id,
                'subject': feedback.subject,
                'message': feedback.message,
                'user_name': feedback.user_name,
                'user_email': feedback.user_email,
                'feedback_type': feedback.feedback_type.value,
                'rating': feedback.rating,
                'page_url': feedback.page_url,
                'is_read': feedback.is_read,
                'admin_notes': feedback.admin_notes,
                'created_at': feedback.created_at.isoformat()
            })
        
        return jsonify({
            'feedback': result,
            'total': feedback_items.total,
            'pages': feedback_items.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Get admin feedback error: {e}")
        return jsonify({'error': 'Failed to get feedback'}), 500

@admin_bp.route('/api/admin/support/feedback/<int:feedback_id>/mark-read', methods=['POST'])
@require_admin
@require_support_access
def mark_feedback_read(current_user, feedback_id):
    """Mark feedback as read and add admin notes"""
    try:
        feedback = Feedback.query.get_or_404(feedback_id)
        data = request.get_json()
        
        feedback.is_read = True
        feedback.admin_notes = data.get('admin_notes', feedback.admin_notes)
        
        db.session.commit()
        
        return jsonify({'message': 'Feedback marked as read'}), 200
        
    except Exception as e:
        logger.error(f"Mark feedback read error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to mark feedback as read'}), 500

@admin_bp.route('/api/admin/support/categories', methods=['GET'])
@require_admin
@require_support_access
def get_support_categories(current_user):
    """Get all support categories"""
    try:
        categories = SupportCategory.query.order_by(SupportCategory.sort_order).all()
        
        result = []
        for category in categories:
            ticket_count = SupportTicket.query.filter_by(category_id=category.id).count()
            faq_count = FAQ.query.filter_by(category_id=category.id, is_published=True).count()
            
            result.append({
                'id': category.id,
                'name': category.name,
                'description': category.description,
                'icon': category.icon,
                'sort_order': category.sort_order,
                'is_active': category.is_active,
                'ticket_count': ticket_count,
                'faq_count': faq_count,
                'created_at': category.created_at.isoformat()
            })
        
        return jsonify({'categories': result}), 200
        
    except Exception as e:
        logger.error(f"Get support categories error: {e}")
        return jsonify({'error': 'Failed to get categories'}), 500

@admin_bp.route('/api/admin/notifications', methods=['GET'])
@require_admin
def get_admin_notifications(current_user):
    """Get real-time admin notifications"""
    try:
        limit = min(int(request.args.get('limit', 50)), 100)
        unread_only = request.args.get('unread_only', 'false').lower() == 'true'
        
        notifications = []
        
        if redis_client:
            try:
                notification_data = redis_client.lrange('dashboard_notifications', 0, limit - 1)
                for data in notification_data:
                    notification = json.loads(data)
                    if not unread_only or not notification.get('read', False):
                        notifications.append(notification)
            except Exception as e:
                logger.error(f"Redis notification retrieval error: {e}")
        
        return jsonify({
            'notifications': notifications,
            'total': len(notifications)
        }), 200
        
    except Exception as e:
        logger.error(f"Get admin notifications error: {e}")
        return jsonify({'error': 'Failed to get notifications'}), 500

@admin_bp.route('/api/admin/notifications/<notification_id>/mark-read', methods=['POST'])
@require_admin
def mark_notification_read(current_user, notification_id):
    """Mark notification as read"""
    try:
        if redis_client:
            # This is a simplified implementation
            # In production, you'd want a more robust notification system
            pass
        
        return jsonify({'message': 'Notification marked as read'}), 200
        
    except Exception as e:
        logger.error(f"Mark notification read error: {e}")
        return jsonify({'error': 'Failed to mark notification as read'}), 500

# ========== EXISTING ENDPOINTS (UPDATED) ==========

@admin_bp.route('/api/admin/analytics', methods=['GET'])
@require_admin
def get_analytics(current_user):
    try:
        # Existing analytics
        total_users = User.query.count()
        total_content = Content.query.count()
        total_interactions = UserInteraction.query.count()
        active_users_last_week = User.query.filter(
            User.last_active >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        popular_content = db.session.query(
            Content.id, Content.title, func.count(UserInteraction.id).label('interaction_count')
        ).join(UserInteraction).group_by(Content.id, Content.title)\
         .order_by(desc('interaction_count')).limit(10).all()
        
        all_interactions = UserInteraction.query.join(Content).all()
        genre_counts = defaultdict(int)
        for interaction in all_interactions:
            content = Content.query.get(interaction.content_id)
            if content and content.genres:
                try:
                    genres = json.loads(content.genres)
                    for genre in genres:
                        genre_counts[genre] += 1
                except:
                    pass
        
        popular_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        user_interactions_by_type = db.session.query(
            UserInteraction.interaction_type,
            func.count(UserInteraction.id).label('count')
        ).group_by(UserInteraction.interaction_type).all()
        
        content_by_type = db.session.query(
            Content.content_type,
            func.count(Content.id).label('count')
        ).group_by(Content.content_type).all()
        
        recent_users = User.query.filter(
            User.created_at >= datetime.utcnow() - timedelta(days=30)
        ).count()
        
        # Support analytics
        support_metrics = SupportAnalytics.get_support_metrics()
        
        return jsonify({
            'total_users': total_users,
            'total_content': total_content,
            'total_interactions': total_interactions,
            'active_users_last_week': active_users_last_week,
            'new_users_last_month': recent_users,
            'popular_content': [
                {'title': item.title, 'interactions': item.interaction_count}
                for item in popular_content
            ],
            'popular_genres': [
                {'genre': genre, 'count': count}
                for genre, count in popular_genres
            ],
            'interactions_by_type': [
                {'type': item.interaction_type, 'count': item.count}
                for item in user_interactions_by_type
            ],
            'content_by_type': [
                {'type': item.content_type, 'count': item.count}
                for item in content_by_type
            ],
            'support_metrics': support_metrics
        }), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

@admin_bp.route('/api/admin/users', methods=['GET'])
@require_admin
def get_users_management(current_user):
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        search = request.args.get('search', '')
        
        query = User.query
        
        if search:
            query = query.filter(
                User.username.contains(search) | 
                User.email.contains(search)
            )
        
        users = query.order_by(User.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        result = []
        for user in users.items:
            user_interactions = UserInteraction.query.filter_by(user_id=user.id).count()
            user_tickets = SupportTicket.query.filter_by(user_id=user.id).count() if SupportTicket else 0
            
            result.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'created_at': user.created_at.isoformat(),
                'last_active': user.last_active.isoformat() if user.last_active else None,
                'total_interactions': user_interactions,
                'total_support_tickets': user_tickets,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]')
            })
        
        return jsonify({
            'users': result,
            'total': users.total,
            'pages': users.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Users management error: {e}")
        return jsonify({'error': 'Failed to get users'}), 500

@admin_bp.route('/api/admin/content/manage', methods=['GET'])
@require_admin
def get_content_management(current_user):
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        content_type = request.args.get('type', 'all')
        search = request.args.get('search', '')
        
        query = Content.query
        
        if content_type != 'all':
            query = query.filter_by(content_type=content_type)
        
        if search:
            query = query.filter(Content.title.contains(search))
        
        contents = query.order_by(Content.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        result = []
        for content in contents.items:
            interaction_count = UserInteraction.query.filter_by(content_id=content.id).count()
            
            result.append({
                'id': content.id,
                'slug': content.slug,
                'title': content.title,
                'content_type': content.content_type,
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'created_at': content.created_at.isoformat(),
                'interaction_count': interaction_count,
                'genres': json.loads(content.genres or '[]'),
                'tmdb_id': content.tmdb_id,
                'mal_id': content.mal_id,
                'poster_path': content.poster_path
            })
        
        return jsonify({
            'content': result,
            'total': contents.total,
            'pages': contents.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Content management error: {e}")
        return jsonify({'error': 'Failed to get content'}), 500

@admin_bp.route('/api/admin/recommendations', methods=['GET'])
@require_admin
def get_admin_recommendations(current_user):
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        admin_recs = AdminRecommendation.query.filter_by(is_active=True)\
            .order_by(AdminRecommendation.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        result = []
        for rec in admin_recs.items:
            content = Content.query.get(rec.content_id)
            admin = User.query.get(rec.admin_id)
            
            result.append({
                'id': rec.id,
                'recommendation_type': rec.recommendation_type,
                'description': rec.description,
                'created_at': rec.created_at.isoformat(),
                'admin_name': admin.username if admin else 'Unknown',
                'content': {
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path
                }
            })
        
        return jsonify({
            'recommendations': result,
            'total': admin_recs.total,
            'pages': admin_recs.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Get admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@admin_bp.route('/api/admin/cache/clear', methods=['POST'])
@require_admin
def clear_cache(current_user):
    try:
        cache_type = request.args.get('type', 'all')
        
        if cache_type == 'all':
            cache.clear()
            message = 'All cache cleared'
        elif cache_type == 'search':
            cache.delete_memoized(TMDBService.search_content)
            cache.delete_memoized(JikanService.search_anime)
            message = 'Search cache cleared'
        elif cache_type == 'recommendations':
            message = 'Recommendations cache cleared'
        elif cache_type == 'support':
            # Clear support-related cache
            if redis_client:
                keys = redis_client.keys('support_*')
                if keys:
                    redis_client.delete(*keys)
            message = 'Support cache cleared'
        else:
            return jsonify({'error': 'Invalid cache type'}), 400
        
        # Send notification
        AdminNotificationService.send_notification(
            title="Cache Cleared",
            message=f"Admin {current_user.username} cleared {cache_type} cache",
            notification_type="cache_cleared",
            priority="normal",
            admin_id=current_user.id
        )
        
        return jsonify({
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return jsonify({'error': 'Failed to clear cache'}), 500

@admin_bp.route('/api/admin/cache/stats', methods=['GET'])
@require_admin
def get_cache_stats(current_user):
    try:
        cache_info = {
            'type': app.config.get('CACHE_TYPE', 'unknown'),
            'default_timeout': app.config.get('CACHE_DEFAULT_TIMEOUT', 0),
        }
        
        if app.config.get('CACHE_TYPE') == 'redis':
            try:
                import redis
                REDIS_URL = app.config.get('CACHE_REDIS_URL')
                if REDIS_URL:
                    r = redis.from_url(REDIS_URL)
                    redis_info = r.info()
                    cache_info['redis'] = {
                        'used_memory': redis_info.get('used_memory_human', 'N/A'),
                        'connected_clients': redis_info.get('connected_clients', 0),
                        'total_commands_processed': redis_info.get('total_commands_processed', 0),
                        'uptime_in_seconds': redis_info.get('uptime_in_seconds', 0)
                    }
            except:
                cache_info['redis'] = {'status': 'Unable to connect'}
        
        # Add support system cache stats
        if redis_client:
            try:
                cache_info['support_queues'] = {
                    'admin_notifications': redis_client.llen('admin_notifications'),
                    'admin_emails': redis_client.llen('admin_emails'),
                    'dashboard_notifications': redis_client.llen('dashboard_notifications')
                }
            except:
                cache_info['support_queues'] = {'status': 'Unable to connect'}
        
        return jsonify(cache_info), 200
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return jsonify({'error': 'Failed to get cache stats'}), 500

@admin_bp.route('/api/admin/system-health', methods=['GET'])
@require_admin
def get_system_health(current_user):
    try:
        health_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy',
            'components': {}
        }
        
        # Database health
        try:
            total_users = User.query.count()
            total_content = Content.query.count()
            total_interactions = UserInteraction.query.count()
            
            health_data['components']['database'] = {
                'status': 'healthy',
                'total_users': total_users,
                'total_content': total_content,
                'total_interactions': total_interactions
            }
            
            # Support system health
            if SupportTicket:
                total_tickets = SupportTicket.query.count()
                open_tickets = SupportTicket.query.filter_by(status='open').count()
                overdue_tickets = SupportTicket.query.filter(
                    and_(
                        SupportTicket.sla_deadline < datetime.utcnow(),
                        SupportTicket.status.in_(['open', 'in_progress'])
                    )
                ).count()
                
                health_data['components']['support_system'] = {
                    'status': 'healthy',
                    'total_tickets': total_tickets,
                    'open_tickets': open_tickets,
                    'overdue_tickets': overdue_tickets
                }
                
        except Exception as e:
            health_data['components']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_data['status'] = 'degraded'
        
        # Cache health
        try:
            cache.set('health_check', 'ok', timeout=10)
            if cache.get('health_check') == 'ok':
                health_data['components']['cache'] = {'status': 'healthy'}
            else:
                health_data['components']['cache'] = {'status': 'degraded'}
                health_data['status'] = 'degraded'
        except:
            health_data['components']['cache'] = {'status': 'unhealthy'}
            health_data['status'] = 'degraded'
        
        # Redis health
        if redis_client:
            try:
                redis_client.ping()
                health_data['components']['redis'] = {'status': 'healthy'}
            except:
                health_data['components']['redis'] = {'status': 'unhealthy'}
                health_data['status'] = 'degraded'
        else:
            health_data['components']['redis'] = {'status': 'not_configured'}
        
        # External APIs health
        health_data['components']['external_apis'] = {
            'tmdb': 'configured' if TMDBService else 'not_configured',
            'jikan': 'configured' if JikanService else 'not_configured',
            'telegram': 'configured' if bot else 'not_configured'
        }
        
        # Notification services health
        health_data['components']['notification_services'] = {
            'telegram_notifications': 'enabled' if bot else 'disabled',
            'email_notifications': 'enabled' if GMAIL_USERNAME and GMAIL_PASSWORD else 'disabled',
            'dashboard_notifications': 'enabled' if redis_client else 'disabled'
        }
        
        return jsonify(health_data), 200
        
    except Exception as e:
        logger.error(f"System health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Auto-notification for new tickets (called from support service)
def notify_new_ticket(ticket):
    """Called when a new support ticket is created"""
    try:
        priority_level = "urgent" if ticket.priority.value in ['urgent', 'high'] else "normal"
        
        AdminNotificationService.send_notification(
            title="New Support Ticket",
            message=f"New {ticket.priority.value} priority ticket #{ticket.ticket_number} from {ticket.user_name}",
            notification_type="new_ticket",
            priority=priority_level,
            ticket_id=ticket.id,
            ticket_number=ticket.ticket_number,
            user_email=ticket.user_email
        )
        
    except Exception as e:
        logger.error(f"New ticket notification error: {e}")

# Auto-notification for SLA breaches
def notify_sla_breach(ticket):
    """Called when a ticket breaches SLA"""
    try:
        AdminNotificationService.send_notification(
            title="SLA BREACH ALERT",
            message=f"Ticket #{ticket.ticket_number} has breached SLA deadline. Immediate attention required!",
            notification_type="sla_breach",
            priority="urgent",
            ticket_id=ticket.id,
            ticket_number=ticket.ticket_number
        )
        
    except Exception as e:
        logger.error(f"SLA breach notification error: {e}")

__all__ = [
    'admin_bp',
    'init_admin',
    'require_admin',
    'require_support_access',
    'AdminNotificationService',
    'AdminEmailService',
    'SupportAnalytics',
    'TelegramService',
    'notify_new_ticket',
    'notify_sla_breach'
]