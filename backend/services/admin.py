from flask import Blueprint, request, jsonify
from flask_caching import Cache
from datetime import datetime, timedelta
import json
import logging
import time
import telebot
from sqlalchemy import func, desc, and_, or_, text
from collections import defaultdict
from functools import wraps
import jwt
import os
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import ssl
import uuid
import redis
from typing import Dict, List, Optional
import enum
from urllib.parse import urlparse

admin_bp = Blueprint('admin', __name__)

logger = logging.getLogger(__name__)

cache = Cache()

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '-1002850793757')
TELEGRAM_ADMIN_CHAT_ID = os.environ.get('TELEGRAM_ADMIN_CHAT_ID', '-1002850793757')
GMAIL_USERNAME = os.environ.get('GMAIL_USERNAME', 'projects.srinath@gmail.com')
GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', 'wuus nsow nbee xewv')
REDIS_URL = os.environ.get('REDIS_URL', 'redis://red-d2qlbuje5dus73c71qog:xp7inVzgblGCbo9I4taSGLdKUg0xY91I@red-d2qlbuje5dus73c71qog:6379')

if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != 'your_telegram_bot_token':
    try:
        bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
    except:
        bot = None
        logging.warning("Failed to initialize Telegram bot")
else:
    bot = None

db = None
User = None
Content = None
UserInteraction = None
AdminRecommendation = None
SupportCategory = None
SupportTicket = None
SupportResponse = None
FAQ = None
Feedback = None
TicketActivity = None
TMDBService = None
JikanService = None
ContentService = None
http_session = None
cache = None
app = None
redis_client = None

TicketStatus = None
TicketPriority = None
TicketType = None
FeedbackType = None

class NotificationType(enum.Enum):
    NEW_TICKET = "new_ticket"
    URGENT_TICKET = "urgent_ticket"
    TICKET_ESCALATION = "ticket_escalation"
    SLA_BREACH = "sla_breach"
    FEEDBACK_RECEIVED = "feedback_received"
    SYSTEM_ALERT = "system_alert"

def init_redis_admin():
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
        logger.info("Admin Redis connected successfully")
        return redis_client
    except Exception as e:
        logger.error(f"Admin Redis connection failed: {e}")
        return None

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

class AdminEmailService:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.username = GMAIL_USERNAME
        self.password = GMAIL_APP_PASSWORD
        self.from_email = "admin@cinebrain.com"
        self.from_name = "CineBrain Admin"
    
    def send_admin_notification(self, subject: str, content: str, admin_emails: List[str]):
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = f"{self.from_name} <{self.username}>"
            msg['To'] = ', '.join(admin_emails)
            msg['Subject'] = f"[CineBrain Admin] {subject}"
            
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5;">
                <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%); padding: 30px; text-align: center;">
                        <h1 style="color: white; margin: 0; font-size: 24px;">🎬 CineBrain Admin Alert</h1>
                    </div>
                    <div style="padding: 30px;">
                        <h2 style="color: #333; margin-top: 0;">{subject}</h2>
                        <div style="color: #666; line-height: 1.6;">
                            {content}
                        </div>
                        <div style="margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 8px;">
                            <p style="margin: 0; color: #666; font-size: 14px;">
                                <strong>Timestamp:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}<br>
                                <strong>System:</strong> CineBrain Admin Dashboard
                            </p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            text_content = f"{subject}\n\n{content}\n\nTimestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))
            
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"✅ Admin notification email sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Admin email error: {e}")
            return False

class TelegramAdminService:
    @staticmethod
    def send_admin_notification(notification_type: str, message: str, is_urgent: bool = False):
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                logger.warning("Telegram bot or admin chat ID not configured")
                return False
            
            icon_map = {
                'new_ticket': '🎫',
                'urgent_ticket': '🚨',
                'sla_breach': '⚠️',
                'feedback': '📝',
                'system_alert': '🔔',
                'recommendation': '🎬'
            }
            
            icon = icon_map.get(notification_type, '📢')
            priority = "🚨 URGENT" if is_urgent else "📋 NORMAL"
            
            formatted_message = f"""
{icon} **CineBrain Admin Alert**

**Priority:** {priority}
**Type:** {notification_type.replace('_', ' ').title()}
**Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

**Message:**
{message}

**Dashboard:** [View Admin Panel](https://cinebrain.vercel.app/admin)

#AdminAlert #CineBrain #{notification_type}
            """
            
            bot.send_message(
                chat_id=TELEGRAM_ADMIN_CHAT_ID,
                text=formatted_message,
                parse_mode='Markdown'
            )
            
            return True
        except Exception as e:
            logger.error(f"Telegram admin notification error: {e}")
            return False
    
    @staticmethod
    def send_support_summary():
        try:
            if not SupportTicket:
                return False
            
            today = datetime.utcnow().date()
            yesterday = today - timedelta(days=1)
            
            new_tickets = SupportTicket.query.filter(
                func.date(SupportTicket.created_at) == today
            ).count()
            
            resolved_tickets = SupportTicket.query.filter(
                func.date(SupportTicket.resolved_at) == today
            ).count()
            
            open_tickets = SupportTicket.query.filter(
                SupportTicket.status.in_(['open', 'in_progress'])
            ).count()
            
            urgent_tickets = SupportTicket.query.filter(
                and_(
                    SupportTicket.priority == 'urgent',
                    SupportTicket.status.in_(['open', 'in_progress'])
                )
            ).count()
            
            new_feedback = Feedback.query.filter(
                func.date(Feedback.created_at) == today
            ).count() if Feedback else 0
            
            message = f"""📊 **Daily Support Summary**

**Today's Activity:**
🎫 New Tickets: {new_tickets}
✅ Resolved: {resolved_tickets}
📋 Open Tickets: {open_tickets}
🚨 Urgent: {urgent_tickets}
💬 New Feedback: {new_feedback}

**Status:** {'🟢 Good' if urgent_tickets == 0 else '🟡 Attention Needed' if urgent_tickets < 5 else '🔴 Critical'}

#DailySummary #Support"""
            
            TelegramAdminService.send_admin_notification('daily_summary', message)
            return True
            
        except Exception as e:
            logger.error(f"Support summary error: {e}")
            return False

class AdminNotificationService:
    email_service = None
    
    @classmethod
    def initialize(cls):
        cls.email_service = AdminEmailService()
        
        def schedule_daily_summary():
            import threading
            import time
            
            def daily_summary_worker():
                while True:
                    try:
                        now = datetime.utcnow()
                        if now.hour == 9 and now.minute == 0:
                            TelegramAdminService.send_support_summary()
                        time.sleep(60)
                    except Exception as e:
                        logger.error(f"Daily summary scheduler error: {e}")
                        time.sleep(300)
            
            thread = threading.Thread(target=daily_summary_worker, daemon=True)
            thread.start()
        
        schedule_daily_summary()
    
    @classmethod
    def create_notification(cls, notification_type: NotificationType, title: str, message: str, 
                          admin_id: int = None, related_ticket_id: int = None, 
                          related_content_id: int = None, is_urgent: bool = False,
                          action_required: bool = False, action_url: str = None,
                          metadata: dict = None):
        try:
            if not globals().get('AdminNotification'):
                logger.warning("AdminNotification model not available")
                return None
            
            notification = globals()['AdminNotification'](
                notification_type=notification_type,
                title=title,
                message=message,
                admin_id=admin_id,
                related_ticket_id=related_ticket_id,
                related_content_id=related_content_id,
                is_urgent=is_urgent,
                action_required=action_required,
                action_url=action_url,
                notification_metadata=metadata or {}
            )
            
            db.session.add(notification)
            db.session.commit()
            
            TelegramAdminService.send_admin_notification(
                notification_type.value, 
                f"{title}\n\n{message}", 
                is_urgent
            )
            
            if is_urgent:
                admin_emails = [user.email for user in User.query.filter_by(is_admin=True).all()]
                if admin_emails:
                    cls.email_service.send_admin_notification(title, message, admin_emails)
            
            if redis_client:
                try:
                    notification_data = {
                        'id': notification.id,
                        'type': notification_type.value,
                        'title': title,
                        'message': message,
                        'is_urgent': is_urgent,
                        'action_required': action_required,
                        'action_url': action_url,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    redis_client.lpush('admin_notifications', json.dumps(notification_data))
                    redis_client.ltrim('admin_notifications', 0, 99)
                    redis_client.expire('admin_notifications', 86400)
                except Exception as e:
                    logger.error(f"Redis notification error: {e}")
            
            logger.info(f"Admin notification created: {title}")
            return notification
            
        except Exception as e:
            logger.error(f"Error creating admin notification: {e}")
            if db:
                db.session.rollback()
            return None

    @classmethod
    def notify_new_ticket(cls, ticket):
        try:
            is_urgent = hasattr(ticket, 'priority') and ticket.priority and ticket.priority.value == 'urgent'
            
            cls.create_notification(
                NotificationType.NEW_TICKET,
                f"New {'Urgent ' if is_urgent else ''}Support Ticket",
                f"Ticket #{ticket.ticket_number} created by {ticket.user_name}\n"
                f"Subject: {ticket.subject}\n"
                f"Priority: {ticket.priority.value.upper() if hasattr(ticket, 'priority') and ticket.priority else 'NORMAL'}\n"
                f"Category: {ticket.category.name if hasattr(ticket, 'category') and ticket.category else 'Unknown'}",
                related_ticket_id=ticket.id,
                is_urgent=is_urgent,
                action_required=True,
                action_url=f"/admin/support/tickets/{ticket.id}",
                metadata={
                    'ticket_number': ticket.ticket_number,
                    'priority': ticket.priority.value if hasattr(ticket, 'priority') and ticket.priority else 'normal',
                    'user_email': ticket.user_email
                }
            )
        except Exception as e:
            logger.error(f"Error notifying new ticket: {e}")
    
    @classmethod
    def notify_sla_breach(cls, ticket):
        try:
            cls.create_notification(
                NotificationType.SLA_BREACH,
                f"SLA Breach - Ticket #{ticket.ticket_number}",
                f"Ticket #{ticket.ticket_number} has exceeded its SLA deadline\n"
                f"Created: {ticket.created_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"Deadline: {ticket.sla_deadline.strftime('%Y-%m-%d %H:%M UTC') if ticket.sla_deadline else 'N/A'}\n"
                f"Priority: {ticket.priority.value.upper() if hasattr(ticket, 'priority') and ticket.priority else 'NORMAL'}",
                related_ticket_id=ticket.id,
                is_urgent=True,
                action_required=True,
                action_url=f"/admin/support/tickets/{ticket.id}"
            )
        except Exception as e:
            logger.error(f"Error notifying SLA breach: {e}")
    
    @classmethod
    def notify_feedback_received(cls, feedback):
        try:
            cls.create_notification(
                NotificationType.FEEDBACK_RECEIVED,
                "New User Feedback Received",
                f"Feedback from {feedback.user_name}\n"
                f"Type: {feedback.feedback_type.value.replace('_', ' ').title() if hasattr(feedback, 'feedback_type') and feedback.feedback_type else 'General'}\n"
                f"Subject: {feedback.subject}\n"
                f"Rating: {'⭐' * (feedback.rating or 0) if feedback.rating else 'No rating'}",
                action_required=False,
                action_url=f"/admin/support/feedback/{feedback.id}",
                metadata={
                    'feedback_type': feedback.feedback_type.value if hasattr(feedback, 'feedback_type') and feedback.feedback_type else 'general',
                    'user_email': feedback.user_email,
                    'rating': feedback.rating
                }
            )
        except Exception as e:
            logger.error(f"Error notifying feedback: {e}")

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

#AdminChoice #MovieRecommendation #CineScope"""
            
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

def init_admin(flask_app, database, models, services):
    global db, User, Content, UserInteraction, AdminRecommendation
    global SupportCategory, SupportTicket, SupportResponse, FAQ, Feedback, TicketActivity
    global TMDBService, JikanService, ContentService
    global http_session, cache, app, redis_client
    global TicketStatus, TicketPriority, TicketType, FeedbackType
    
    app = flask_app
    db = database
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    AdminRecommendation = models['AdminRecommendation']
    
    SupportCategory = models.get('SupportCategory')
    SupportTicket = models.get('SupportTicket')
    SupportResponse = models.get('SupportResponse')
    FAQ = models.get('FAQ')
    Feedback = models.get('Feedback')
    TicketActivity = models.get('TicketActivity')
    
    try:
        from services.support import TicketStatus as TS, TicketPriority as TP, TicketType as TT, FeedbackType as FT
        TicketStatus = TS
        TicketPriority = TP
        TicketType = TT
        FeedbackType = FT
    except ImportError:
        logger.warning("Support enums not available")
    
    TMDBService = services['TMDBService']
    JikanService = services['JikanService']
    ContentService = services['ContentService']
    http_session = services['http_session']
    cache = services['cache']
    
    redis_client = init_redis_admin()
    
    create_admin_models(database)
    AdminNotificationService.initialize()
    
    logger.info("✅ Admin service with support management initialized successfully")

def create_admin_models(database):
    class AdminNotification(database.Model):
        __tablename__ = 'admin_notifications'
        
        id = database.Column(database.Integer, primary_key=True)
        notification_type = database.Column(database.Enum(NotificationType), nullable=False)
        title = database.Column(database.String(255), nullable=False)
        message = database.Column(database.Text, nullable=False)
        
        admin_id = database.Column(database.Integer, database.ForeignKey('user.id'), nullable=True)
        related_ticket_id = database.Column(database.Integer, nullable=True)
        related_content_id = database.Column(database.Integer, database.ForeignKey('content.id'), nullable=True)
        
        is_read = database.Column(database.Boolean, default=False)
        is_urgent = database.Column(database.Boolean, default=False)
        action_required = database.Column(database.Boolean, default=False)
        action_url = database.Column(database.String(500))
        
        notification_metadata = database.Column(database.JSON)
        
        created_at = database.Column(database.DateTime, default=datetime.utcnow)
        read_at = database.Column(database.DateTime)
    
    class CannedResponse(database.Model):
        __tablename__ = 'canned_responses'
        
        id = database.Column(database.Integer, primary_key=True)
        title = database.Column(database.String(255), nullable=False)
        content = database.Column(database.Text, nullable=False)
        
        category_id = database.Column(database.Integer, nullable=True)
        tags = database.Column(database.JSON)
        
        is_active = database.Column(database.Boolean, default=True)
        usage_count = database.Column(database.Integer, default=0)
        
        created_by = database.Column(database.Integer, database.ForeignKey('user.id'), nullable=False)
        created_at = database.Column(database.DateTime, default=datetime.utcnow)
        updated_at = database.Column(database.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    class SupportMetrics(database.Model):
        __tablename__ = 'support_metrics'
        
        id = database.Column(database.Integer, primary_key=True)
        date = database.Column(database.Date, nullable=False)
        
        tickets_created = database.Column(database.Integer, default=0)
        tickets_resolved = database.Column(database.Integer, default=0)
        tickets_closed = database.Column(database.Integer, default=0)
        
        avg_first_response_time = database.Column(database.Float)
        avg_resolution_time = database.Column(database.Float)
        
        sla_breaches = database.Column(database.Integer, default=0)
        escalations = database.Column(database.Integer, default=0)
        
        customer_satisfaction = database.Column(database.Float)
        feedback_count = database.Column(database.Integer, default=0)
        
        created_at = database.Column(database.DateTime, default=datetime.utcnow)
    
    globals()['AdminNotification'] = AdminNotification
    globals()['CannedResponse'] = CannedResponse
    globals()['SupportMetrics'] = SupportMetrics
    
    try:
        with app.app_context():
            database.create_all()
            logger.info("Admin models created successfully")
    except Exception as e:
        logger.error(f"Error creating admin models: {e}")

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
        
        AdminNotificationService.create_notification(
            NotificationType.NEW_TICKET,
            f"New Admin Recommendation Created",
            f"Admin {current_user.username} recommended '{content.title}'\n"
            f"Type: {data['recommendation_type']}\n"
            f"Description: {data['description'][:100]}...",
            admin_id=current_user.id,
            related_content_id=content.id
        )
        
        return jsonify({
            'message': 'Admin recommendation created successfully',
            'telegram_sent': telegram_success
        }), 201
        
    except Exception as e:
        logger.error(f"Admin recommendation error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create recommendation'}), 500

@admin_bp.route('/api/admin/support/dashboard', methods=['GET'])
@require_admin
def get_support_dashboard(current_user):
    try:
        if not SupportTicket:
            return jsonify({'error': 'Support system not available'}), 503
        
        today = datetime.utcnow().date()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        total_tickets = SupportTicket.query.count()
        
        if TicketStatus:
            open_tickets = SupportTicket.query.filter(
                SupportTicket.status.in_([TicketStatus.OPEN, TicketStatus.IN_PROGRESS, TicketStatus.WAITING_FOR_USER])
            ).count()
            
            urgent_tickets = SupportTicket.query.filter(
                and_(
                    SupportTicket.priority == TicketPriority.URGENT,
                    SupportTicket.status.in_([TicketStatus.OPEN, TicketStatus.IN_PROGRESS])
                )
            ).count() if TicketPriority else 0
            
            sla_breached = SupportTicket.query.filter(
                and_(
                    SupportTicket.sla_breached == True,
                    SupportTicket.status.in_([TicketStatus.OPEN, TicketStatus.IN_PROGRESS])
                )
            ).count()
        else:
            open_tickets = 0
            urgent_tickets = 0
            sla_breached = 0
        
        today_tickets = SupportTicket.query.filter(
            func.date(SupportTicket.created_at) == today
        ).count()
        
        today_resolved = SupportTicket.query.filter(
            func.date(SupportTicket.resolved_at) == today
        ).count()
        
        avg_response_time = db.session.query(
            func.avg(
                func.extract('epoch', SupportTicket.first_response_at - SupportTicket.created_at) / 3600
            )
        ).filter(SupportTicket.first_response_at.isnot(None)).scalar() or 0
        
        category_stats = db.session.query(
            SupportCategory.name,
            func.count(SupportTicket.id).label('count')
        ).join(SupportTicket).group_by(SupportCategory.name).all() if SupportCategory else []
        
        if TicketStatus:
            priority_stats = db.session.query(
                SupportTicket.priority,
                func.count(SupportTicket.id).label('count')
            ).filter(
                SupportTicket.status.in_([TicketStatus.OPEN, TicketStatus.IN_PROGRESS])
            ).group_by(SupportTicket.priority).all()
        else:
            priority_stats = []
        
        recent_tickets = SupportTicket.query.order_by(
            SupportTicket.created_at.desc()
        ).limit(10).all()
        
        recent_tickets_data = []
        for ticket in recent_tickets:
            recent_tickets_data.append({
                'id': ticket.id,
                'ticket_number': ticket.ticket_number,
                'subject': ticket.subject,
                'user_name': ticket.user_name,
                'priority': ticket.priority.value if hasattr(ticket, 'priority') and ticket.priority else 'normal',
                'status': ticket.status.value if hasattr(ticket, 'status') and ticket.status else 'open',
                'category': ticket.category.name if hasattr(ticket, 'category') and ticket.category else 'Unknown',
                'created_at': ticket.created_at.isoformat(),
                'is_sla_breached': ticket.sla_breached
            })
        
        total_feedback = Feedback.query.count() if Feedback else 0
        unread_feedback = Feedback.query.filter_by(is_read=False).count() if Feedback else 0
        
        recent_feedback = []
        if Feedback:
            feedback_items = Feedback.query.order_by(Feedback.created_at.desc()).limit(5).all()
            for feedback in feedback_items:
                recent_feedback.append({
                    'id': feedback.id,
                    'subject': feedback.subject,
                    'user_name': feedback.user_name,
                    'feedback_type': feedback.feedback_type.value if hasattr(feedback, 'feedback_type') and feedback.feedback_type else 'general',
                    'rating': feedback.rating,
                    'is_read': feedback.is_read,
                    'created_at': feedback.created_at.isoformat()
                })
        
        return jsonify({
            'ticket_stats': {
                'total': total_tickets,
                'open': open_tickets,
                'urgent': urgent_tickets,
                'sla_breached': sla_breached,
                'today_created': today_tickets,
                'today_resolved': today_resolved
            },
            'metrics': {
                'avg_response_time_hours': round(avg_response_time, 2),
                'resolution_rate': round((today_resolved / max(today_tickets, 1)) * 100, 1)
            },
            'category_breakdown': [
                {'category': stat[0], 'count': stat[1]} 
                for stat in category_stats
            ],
            'priority_breakdown': [
                {'priority': stat[0].value if hasattr(stat[0], 'value') else str(stat[0]), 'count': stat[1]} 
                for stat in priority_stats
            ],
            'recent_tickets': recent_tickets_data,
            'feedback_stats': {
                'total': total_feedback,
                'unread': unread_feedback
            },
            'recent_feedback': recent_feedback,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Support dashboard error: {e}")
        return jsonify({'error': 'Failed to load support dashboard'}), 500

@admin_bp.route('/api/admin/support/tickets', methods=['GET'])
@require_admin
def get_support_tickets(current_user):
    try:
        if not SupportTicket:
            return jsonify({'error': 'Support system not available'}), 503
        
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        status = request.args.get('status')
        priority = request.args.get('priority')
        category_id = request.args.get('category_id', type=int)
        search = request.args.get('search', '').strip()
        
        query = SupportTicket.query
        
        if status and status != 'all' and TicketStatus:
            try:
                status_enum = TicketStatus(status)
                query = query.filter(SupportTicket.status == status_enum)
            except ValueError:
                pass
        
        if priority and priority != 'all' and TicketPriority:
            try:
                priority_enum = TicketPriority(priority)
                query = query.filter(SupportTicket.priority == priority_enum)
            except ValueError:
                pass
        
        if category_id:
            query = query.filter(SupportTicket.category_id == category_id)
        
        if search:
            query = query.filter(
                or_(
                    SupportTicket.ticket_number.contains(search),
                    SupportTicket.subject.contains(search),
                    SupportTicket.user_name.contains(search),
                    SupportTicket.user_email.contains(search)
                )
            )
        
        query = query.order_by(
            SupportTicket.priority.desc(),
            SupportTicket.created_at.desc()
        )
        
        tickets = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        result = []
        for ticket in tickets.items:
            result.append({
                'id': ticket.id,
                'ticket_number': ticket.ticket_number,
                'subject': ticket.subject,
                'description': ticket.description[:200] + '...' if len(ticket.description) > 200 else ticket.description,
                'user_name': ticket.user_name,
                'user_email': ticket.user_email,
                'user_id': ticket.user_id,
                'category': {
                    'id': ticket.category.id,
                    'name': ticket.category.name,
                    'icon': ticket.category.icon
                } if hasattr(ticket, 'category') and ticket.category else None,
                'ticket_type': ticket.ticket_type.value if hasattr(ticket, 'ticket_type') and ticket.ticket_type else 'contact',
                'priority': ticket.priority.value if hasattr(ticket, 'priority') and ticket.priority else 'normal',
                'status': ticket.status.value if hasattr(ticket, 'status') and ticket.status else 'open',
                'created_at': ticket.created_at.isoformat(),
                'first_response_at': ticket.first_response_at.isoformat() if ticket.first_response_at else None,
                'resolved_at': ticket.resolved_at.isoformat() if ticket.resolved_at else None,
                'sla_deadline': ticket.sla_deadline.isoformat() if ticket.sla_deadline else None,
                'sla_breached': ticket.sla_breached,
                'response_count': ticket.responses.count() if hasattr(ticket, 'responses') else 0,
                'last_activity': ticket.activities.order_by(TicketActivity.created_at.desc()).first().created_at.isoformat() if hasattr(ticket, 'activities') and ticket.activities.first() else ticket.created_at.isoformat()
            })
        
        return jsonify({
            'tickets': result,
            'pagination': {
                'current_page': page,
                'per_page': per_page,
                'total': tickets.total,
                'pages': tickets.pages,
                'has_prev': tickets.has_prev,
                'has_next': tickets.has_next
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Get support tickets error: {e}")
        return jsonify({'error': 'Failed to get support tickets'}), 500

@admin_bp.route('/api/admin/analytics', methods=['GET'])
@require_admin
def get_analytics(current_user):
    try:
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
        
        support_stats = {}
        if SupportTicket:
            try:
                support_stats = {
                    'total_tickets': SupportTicket.query.count(),
                    'open_tickets': SupportTicket.query.filter(
                        SupportTicket.status.in_(['open', 'in_progress'])
                    ).count(),
                    'resolved_today': SupportTicket.query.filter(
                        func.date(SupportTicket.resolved_at) == datetime.utcnow().date()
                    ).count(),
                    'total_feedback': Feedback.query.count() if Feedback else 0
                }
            except Exception as e:
                logger.error(f"Support stats error: {e}")
                support_stats = {
                    'total_tickets': 0,
                    'open_tickets': 0,
                    'resolved_today': 0,
                    'total_feedback': 0
                }
        
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
            'support_stats': support_stats
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
            
            result.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'created_at': user.created_at.isoformat(),
                'last_active': user.last_active.isoformat() if user.last_active else None,
                'total_interactions': user_interactions,
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
        else:
            return jsonify({'error': 'Invalid cache type'}), 400
        
        return jsonify({
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return jsonify({'error': 'Failed to clear cache'}), 500

@admin_bp.route('/api/admin/system-health', methods=['GET'])
@require_admin
def get_system_health(current_user):
    try:
        health_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy',
            'components': {}
        }
        
        health_data['components']['database'] = {
            'status': 'healthy',
            'total_users': User.query.count(),
            'total_content': Content.query.count(),
            'total_interactions': UserInteraction.query.count()
        }
        
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
        
        health_data['components']['external_apis'] = {
            'tmdb': 'configured' if TMDBService else 'not_configured',
            'jikan': 'configured' if JikanService else 'not_configured',
            'telegram': 'configured' if bot else 'not_configured'
        }
        
        try:
            if SupportTicket and TicketStatus:
                health_data['components']['support_system'] = {
                    'status': 'healthy',
                    'total_tickets': SupportTicket.query.count(),
                    'open_tickets': SupportTicket.query.filter(
                        SupportTicket.status.in_([TicketStatus.OPEN, TicketStatus.IN_PROGRESS])
                    ).count(),
                    'total_feedback': Feedback.query.count() if Feedback else 0
                }
            else:
                health_data['components']['support_system'] = {
                    'status': 'not_available',
                    'total_tickets': 0,
                    'open_tickets': 0,
                    'total_feedback': 0
                }
        except Exception as e:
            logger.error(f"Support system health check error: {e}")
            health_data['components']['support_system'] = {
                'status': 'error',
                'error': str(e),
                'total_tickets': 0,
                'open_tickets': 0,
                'total_feedback': 0
            }
        
        return jsonify(health_data), 200
        
    except Exception as e:
        logger.error(f"System health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@admin_bp.route('/api/admin/notifications', methods=['GET'])
@require_admin
def get_admin_notifications(current_user):
    try:
        page = int(request.args.get('page', 1))
        per_page = min(int(request.args.get('per_page', 20)), 100)
        unread_only = request.args.get('unread_only', 'false').lower() == 'true'
        
        recent_notifications = []
        if redis_client:
            try:
                notifications_json = redis_client.lrange('admin_notifications', 0, 19)
                for notif_json in notifications_json:
                    recent_notifications.append(json.loads(notif_json))
            except Exception as e:
                logger.error(f"Redis notification retrieval error: {e}")
        
        db_notifications = []
        if globals().get('AdminNotification'):
            query = globals()['AdminNotification'].query
            
            if unread_only:
                query = query.filter_by(is_read=False)
            
            notifications = query.order_by(
                globals()['AdminNotification'].created_at.desc()
            ).paginate(page=page, per_page=per_page, error_out=False)
            
            for notif in notifications.items:
                related_ticket = None
                if notif.related_ticket_id and SupportTicket:
                    ticket = SupportTicket.query.get(notif.related_ticket_id)
                    if ticket:
                        related_ticket = {
                            'id': ticket.id,
                            'ticket_number': ticket.ticket_number,
                            'subject': ticket.subject
                        }
                
                db_notifications.append({
                    'id': notif.id,
                    'type': notif.notification_type.value,
                    'title': notif.title,
                    'message': notif.message,
                    'is_read': notif.is_read,
                    'is_urgent': notif.is_urgent,
                    'action_required': notif.action_required,
                    'action_url': notif.action_url,
                    'related_ticket': related_ticket,
                    'metadata': notif.notification_metadata,
                    'created_at': notif.created_at.isoformat(),
                    'read_at': notif.read_at.isoformat() if notif.read_at else None
                })
        
        return jsonify({
            'recent_notifications': recent_notifications,
            'notifications': db_notifications,
            'pagination': {
                'current_page': page,
                'per_page': per_page,
                'total': notifications.total if 'notifications' in locals() else 0,
                'pages': notifications.pages if 'notifications' in locals() else 0,
                'has_prev': notifications.has_prev if 'notifications' in locals() else False,
                'has_next': notifications.has_next if 'notifications' in locals() else False
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Get admin notifications error: {e}")
        return jsonify({'error': 'Failed to get notifications'}), 500