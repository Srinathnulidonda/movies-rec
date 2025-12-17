# admin/service.py

import os
import json
import logging
import redis
import uuid
from datetime import datetime, timedelta
from urllib.parse import urlparse
from sqlalchemy import func, desc, and_, or_
from collections import defaultdict

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get('REDIS_URL')

class NotificationType:
    NEW_TICKET = "NEW_TICKET"
    URGENT_TICKET = "URGENT_TICKET"
    TICKET_ESCALATION = "TICKET_ESCALATION"
    SLA_BREACH = "SLA_BREACH"
    FEEDBACK_RECEIVED = "FEEDBACK_RECEIVED"
    SYSTEM_ALERT = "SYSTEM_ALERT"
    USER_ACTIVITY = "user_activity"
    CONTENT_ADDED = "content_added"

class AdminEmailPreferences:
    def __init__(self, db):
        self.db = db
        
        class AdminEmailPreferencesModel(db.Model):
            __tablename__ = 'admin_email_preferences'
            
            id = db.Column(db.Integer, primary_key=True)
            admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
            
            urgent_tickets = db.Column(db.Boolean, default=True)
            sla_breaches = db.Column(db.Boolean, default=True)
            system_alerts = db.Column(db.Boolean, default=True)
            
            content_added = db.Column(db.Boolean, default=True)
            recommendation_created = db.Column(db.Boolean, default=True)
            recommendation_updated = db.Column(db.Boolean, default=False)
            recommendation_deleted = db.Column(db.Boolean, default=False)
            recommendation_published = db.Column(db.Boolean, default=True)
            
            user_feedback = db.Column(db.Boolean, default=True)
            regular_tickets = db.Column(db.Boolean, default=False)
            
            cache_operations = db.Column(db.Boolean, default=False)
            bulk_operations = db.Column(db.Boolean, default=False)
            slug_updates = db.Column(db.Boolean, default=False)
            
            created_at = db.Column(db.DateTime, default=datetime.utcnow)
            updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
            
            __table_args__ = (db.UniqueConstraint('admin_id'),)
        
        self.Model = AdminEmailPreferencesModel

class AdminEmailService:
    def __init__(self, services):
        self.email_service = services.get('email_service')
        
        if not self.email_service:
            try:
                from auth.service import email_service as auth_email_service
                self.email_service = auth_email_service
                logger.info("✅ Email service loaded from auth module for admin")
            except Exception as e:
                logger.warning(f"Could not load auth email service for admin: {e}")
        
        self.is_configured = (
            self.email_service is not None and 
            hasattr(self.email_service, 'email_enabled') and 
            self.email_service.email_enabled and
            hasattr(self.email_service, 'queue_email')
        )
        
        if self.is_configured:
            logger.info("✅ Admin email service configured successfully with Brevo")
        else:
            logger.warning("⚠️ Admin email service not configured - using fallback mode")
    
    def send_admin_notification(self, subject: str, content: str, admin_emails: list, is_urgent: bool = False):
        try:
            if not self.is_configured:
                logger.warning("Email service not configured, skipping admin email notification")
                return False
            
            if not admin_emails:
                logger.warning("No admin emails provided")
                return False
            
            from auth.admin_mail_templates import get_admin_template
            
            html, text = get_admin_template(
                'admin_notification',
                subject=subject,
                content=content,
                is_urgent=is_urgent,
                timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            )
            
            success_count = 0
            for email in admin_emails:
                try:
                    admin_name = email.split('@')[0].replace('.', ' ').title()
                    
                    self.email_service.queue_email(
                        to=email,
                        subject=f"[CineBrain Admin] {subject}",
                        html=html,
                        text=text,
                        priority='urgent' if is_urgent else 'high',
                        to_name=admin_name
                    )
                    success_count += 1
                    logger.info(f"✅ Admin notification email queued for {email}: {subject}")
                except Exception as e:
                    logger.error(f"❌ Failed to queue email for {email}: {e}")
            
            if success_count > 0:
                logger.info(f"✅ Admin notification emails queued successfully: {subject} ({success_count}/{len(admin_emails)})")
                return True
            else:
                logger.error(f"❌ Failed to queue any admin notification emails")
                return False
            
        except Exception as e:
            logger.error(f"❌ Admin email error: {e}")
            return False

class AdminNotificationService:
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.User = models['User']
        self.email_service = AdminEmailService(services)
        self.redis_client = self._init_redis()
        
        self.AdminNotification = models.get('AdminNotification')
        self.CannedResponse = models.get('CannedResponse')
        self.SupportMetrics = models.get('SupportMetrics')
        self.AdminEmailPreferences = models.get('AdminEmailPreferences')
        
        if not self.AdminEmailPreferences:
            try:
                email_prefs = AdminEmailPreferences(db)
                self.AdminEmailPreferences = email_prefs.Model
                logger.info("✅ AdminEmailPreferences model created dynamically")
            except Exception as e:
                logger.error(f"❌ Failed to create AdminEmailPreferences model: {e}")
                self.AdminEmailPreferences = None
        
        if not self.AdminNotification:
            logger.warning("⚠️ AdminNotification model not available")
        if not self.CannedResponse:
            logger.warning("⚠️ CannedResponse model not available")
        if not self.SupportMetrics:
            logger.warning("⚠️ SupportMetrics model not available")
        if not self.AdminEmailPreferences:
            logger.warning("⚠️ AdminEmailPreferences model not available")
        
        logger.info("✅ Admin notification service initialized (EMAIL ONLY)")
    
    def _init_redis(self):
        try:
            if not REDIS_URL:
                logger.warning("Redis URL not configured for admin notifications")
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
            logger.info("✅ Admin Redis connected successfully")
            return redis_client
            
        except Exception as e:
            logger.error(f"❌ Admin Redis connection failed: {e}")
            return None
    
    def _should_send_email(self, admin_id: int, alert_type: str) -> bool:
        try:
            if not self.AdminEmailPreferences:
                return True
            
            preferences = self.AdminEmailPreferences.query.filter_by(admin_id=admin_id).first()
            if not preferences:
                preferences = self._create_default_preferences(admin_id)
                if not preferences:
                    return True
            
            alert_mapping = {
                'urgent_ticket': preferences.urgent_tickets,
                'new_ticket': preferences.regular_tickets,
                'sla_breach': preferences.sla_breaches,
                'system_alert': preferences.system_alerts,
                'content_added': preferences.content_added,
                'recommendation_created': preferences.recommendation_created,
                'recommendation_updated': preferences.recommendation_updated,
                'recommendation_deleted': preferences.recommendation_deleted,
                'recommendation_published': preferences.recommendation_published,
                'feedback_received': preferences.user_feedback,
                'cache_operation': preferences.cache_operations,
                'bulk_operation': preferences.bulk_operations,
                'slug_update': preferences.slug_updates
            }
            
            return alert_mapping.get(alert_type, True)
            
        except Exception as e:
            logger.error(f"Error checking email preferences: {e}")
            return True
    
    def _create_default_preferences(self, admin_id: int):
        try:
            preferences = self.AdminEmailPreferences(admin_id=admin_id)
            self.db.session.add(preferences)
            self.db.session.commit()
            logger.info(f"✅ Default email preferences created for admin {admin_id}")
            return preferences
        except Exception as e:
            logger.error(f"Error creating default preferences for admin {admin_id}: {e}")
            try:
                self.db.session.rollback()
            except:
                pass
            return None
    
    def create_notification(self, notification_type: str, title: str, message: str, 
                          admin_id: int = None, related_ticket_id: int = None, 
                          related_content_id: int = None, is_urgent: bool = False,
                          action_required: bool = False, action_url: str = None,
                          metadata: dict = None, alert_type: str = None):
        try:
            notification = None
            if not self.AdminNotification:
                logger.warning("AdminNotification model not available, skipping database notification")
            else:
                try:
                    notification = self.AdminNotification(
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
                    
                    self.db.session.add(notification)
                    self.db.session.commit()
                    logger.info(f"✅ Database notification created: {title}")
                except Exception as e:
                    logger.error(f"Database notification creation failed: {e}")
                    try:
                        self.db.session.rollback()
                    except:
                        pass
            
            if self.email_service and self.email_service.is_configured:
                try:
                    admin_users = self.User.query.filter_by(is_admin=True).all()
                    
                    admins_to_notify = []
                    for admin in admin_users:
                        if admin.email and self._should_send_email(admin.id, alert_type or notification_type.lower()):
                            admins_to_notify.append(admin.email)
                    
                    env_admin_email = os.environ.get('ADMIN_EMAIL')
                    if env_admin_email and env_admin_email not in admins_to_notify:
                        env_admin = self.User.query.filter_by(email=env_admin_email).first()
                        if env_admin and self._should_send_email(env_admin.id, alert_type or notification_type.lower()):
                            admins_to_notify.append(env_admin_email)
                        elif not env_admin:
                            critical_alerts = ['urgent_ticket', 'sla_breach', 'system_alert']
                            if (alert_type or notification_type.lower()) in critical_alerts:
                                admins_to_notify.append(env_admin_email)
                    
                    if admins_to_notify:
                        self.email_service.send_admin_notification(title, message, admins_to_notify, is_urgent)
                        logger.info(f"✅ Email notifications sent to {len(admins_to_notify)} admins: {title}")
                    else:
                        logger.info(f"📧 No admins configured to receive '{alert_type or notification_type}' emails")
                        
                except Exception as e:
                    logger.warning(f"Admin email notification failed: {e}")
            
            if self.redis_client:
                try:
                    notification_data = {
                        'id': notification.id if notification else f"temp_{int(datetime.utcnow().timestamp())}",
                        'type': notification_type,
                        'title': title,
                        'message': message,
                        'is_urgent': is_urgent,
                        'action_required': action_required,
                        'action_url': action_url,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    self.redis_client.lpush('admin_notifications', json.dumps(notification_data))
                    self.redis_client.ltrim('admin_notifications', 0, 99)
                    self.redis_client.expire('admin_notifications', 86400)
                    logger.info(f"✅ Redis notification stored: {title}")
                except Exception as e:
                    logger.error(f"Redis notification error: {e}")
            
            logger.info(f"✅ Admin notification created successfully (EMAIL ONLY): {title}")
            return notification if notification else True
            
        except Exception as e:
            logger.error(f"❌ Error creating admin notification: {e}")
            try:
                self.db.session.rollback()
            except:
                pass
            return None
    
    def notify_new_ticket(self, ticket):
        try:
            is_urgent = ticket.priority == 'urgent'
            alert_type = 'urgent_ticket' if is_urgent else 'new_ticket'
            
            self.create_notification(
                NotificationType.NEW_TICKET if not is_urgent else NotificationType.URGENT_TICKET,
                f"New {'Urgent ' if is_urgent else ''}Support Ticket",
                f"Ticket #{ticket.ticket_number} created by {ticket.user_name}\n"
                f"Subject: {ticket.subject}\n"
                f"Priority: {ticket.priority.upper()}\n"
                f"Category: {ticket.category.name if hasattr(ticket, 'category') and ticket.category else 'Unknown'}",
                related_ticket_id=ticket.id,
                is_urgent=is_urgent,
                action_required=True,
                action_url=f"/admin/support/tickets/{ticket.id}",
                metadata={
                    'ticket_number': ticket.ticket_number,
                    'priority': ticket.priority,
                    'user_email': ticket.user_email
                },
                alert_type=alert_type
            )
            logger.info(f"✅ New ticket notification sent (EMAIL): #{ticket.ticket_number}")
        except Exception as e:
            logger.error(f"❌ Error notifying new ticket: {e}")
    
    def notify_sla_breach(self, ticket):
        try:
            self.create_notification(
                NotificationType.SLA_BREACH,
                f"SLA Breach - Ticket #{ticket.ticket_number}",
                f"Ticket #{ticket.ticket_number} has exceeded its SLA deadline\n"
                f"Created: {ticket.created_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"Deadline: {ticket.sla_deadline.strftime('%Y-%m-%d %H:%M UTC') if ticket.sla_deadline else 'N/A'}\n"
                f"Priority: {ticket.priority.upper()}",
                related_ticket_id=ticket.id,
                is_urgent=True,
                action_required=True,
                action_url=f"/admin/support/tickets/{ticket.id}",
                metadata={
                    'ticket_number': ticket.ticket_number,
                    'sla_deadline': ticket.sla_deadline.isoformat() if ticket.sla_deadline else None
                },
                alert_type='sla_breach'
            )
            logger.info(f"✅ SLA breach notification sent (EMAIL): #{ticket.ticket_number}")
        except Exception as e:
            logger.error(f"❌ Error notifying SLA breach: {e}")
    
    def notify_feedback_received(self, feedback):
        try:
            feedback_type = getattr(feedback, 'feedback_type', 'general')
            rating = getattr(feedback, 'rating', 0)
            
            self.create_notification(
                NotificationType.FEEDBACK_RECEIVED,
                "New User Feedback Received",
                f"Feedback from {feedback.user_name}\n"
                f"Type: {feedback_type.replace('_', ' ').title() if isinstance(feedback_type, str) else 'General'}\n"
                f"Subject: {feedback.subject}\n"
                f"Rating: {'⭐' * rating if rating else 'No rating'}",
                action_required=False,
                action_url=f"/admin/support/feedback/{feedback.id}",
                metadata={
                    'feedback_type': feedback_type if isinstance(feedback_type, str) else 'general',
                    'user_email': feedback.user_email,
                    'rating': rating
                },
                alert_type='feedback_received'
            )
            logger.info(f"✅ Feedback notification sent (EMAIL): feedback #{feedback.id}")
        except Exception as e:
            logger.error(f"❌ Error notifying feedback: {e}")
    
    def notify_system_alert(self, alert_type_param: str, title: str, details: str, is_urgent: bool = False):
        try:
            self.create_notification(
                NotificationType.SYSTEM_ALERT,
                f"System Alert: {title}",
                f"Alert Type: {alert_type_param}\n"
                f"Details: {details}\n"
                f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                is_urgent=is_urgent,
                action_required=True,
                action_url="/admin/system-health",
                metadata={
                    'alert_type': alert_type_param,
                    'timestamp': datetime.utcnow().isoformat()
                },
                alert_type='system_alert'
            )
            logger.info(f"✅ System alert notification sent (EMAIL): {title}")
        except Exception as e:
            logger.error(f"❌ Error notifying system alert: {e}")

class AdminService:
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.AdminRecommendation = models['AdminRecommendation']
        
        self.AdminNotification = models.get('AdminNotification')
        self.CannedResponse = models.get('CannedResponse')
        self.SupportMetrics = models.get('SupportMetrics')
        self.AdminEmailPreferences = models.get('AdminEmailPreferences')
        
        self.TMDBService = services.get('TMDBService')
        self.JikanService = services.get('JikanService')
        self.ContentService = services.get('ContentService')
        self.cache = services.get('cache')
        
        try:
            from admin.telegram import TelegramService
            self.telegram_service = TelegramService
        except ImportError:
            self.telegram_service = None
        
        self.notification_service = AdminNotificationService(app, db, models, services)
        
        logger.info("✅ Admin service initialized")
    
    def _map_genre_ids_to_names(self, genre_ids):
        genre_map = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
            99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
            27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
        }
        
        genre_names = []
        for genre_id in genre_ids:
            if isinstance(genre_id, int) and genre_id in genre_map:
                genre_names.append(genre_map[genre_id])
            elif isinstance(genre_id, str):
                genre_names.append(genre_id)
        
        return genre_names or ["Drama"]
    
    def search_external_content(self, query: str, source: str, page: int = 1):
        try:
            results = []
            
            if source == 'tmdb':
                tmdb_results = self.TMDBService.search_content(query, page=page)
                if tmdb_results:
                    for item in tmdb_results.get('results', []):
                        content_item = {
                            'id': item.get('id'),
                            'title': item.get('title') or item.get('name') or 'Unknown Title',
                            'content_type': 'movie' if 'title' in item else 'tv',
                            'release_date': item.get('release_date') or item.get('first_air_date'),
                            'poster_path': f"https://image.tmdb.org/t/p/w300{item['poster_path']}" if item.get('poster_path') else None,
                            'overview': item.get('overview') or '',
                            'rating': item.get('vote_average') or 0,
                            'vote_average': item.get('vote_average') or 0,
                            'vote_count': item.get('vote_count') or 0,
                            'popularity': item.get('popularity') or 0,
                            'genre_ids': item.get('genre_ids', []),
                            'original_language': item.get('original_language') or 'en',
                            'backdrop_path': item.get('backdrop_path'),
                            'source': 'tmdb'
                        }
                        results.append(content_item)
            
            elif source == 'anime':
                anime_results = self.JikanService.search_anime(query, page=page)
                if anime_results:
                    for anime in anime_results.get('data', []):
                        content_item = {
                            'id': anime.get('mal_id'),
                            'title': anime.get('title') or 'Unknown Title',
                            'content_type': 'anime',
                            'release_date': anime.get('aired', {}).get('from') if anime.get('aired') else None,
                            'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url') if anime.get('images') else None,
                            'overview': anime.get('synopsis') or '',
                            'rating': anime.get('score') or 0,
                            'vote_average': anime.get('score') or 0,
                            'vote_count': anime.get('scored_by') or 0,
                            'popularity': anime.get('popularity') or 0,
                            'genres': [genre.get('name', 'Unknown') for genre in anime.get('genres', [])],
                            'source': 'anime'
                        }
                        results.append(content_item)
            
            logger.info(f"✅ External content search completed: {len(results)} results for '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"External search error: {e}")
            return []
    
    def save_external_content(self, data):
        try:
            existing_content = None
            
            if not data or not data.get('id'):
                logger.error("Invalid content data: missing ID")
                return {'error': 'Invalid content data: missing ID'}
            
            if data.get('source') == 'anime' and data.get('id'):
                mal_id = int(data['id']) if isinstance(data['id'], str) and data['id'].isdigit() else data['id']
                existing_content = self.Content.query.filter_by(mal_id=mal_id).first()
            elif data.get('id'):
                tmdb_id = int(data['id']) if isinstance(data['id'], str) and data['id'].isdigit() else data['id']
                existing_content = self.Content.query.filter_by(tmdb_id=tmdb_id).first()
            
            if existing_content:
                logger.info(f"Content already exists: {existing_content.title}")
                return {
                    'message': 'Content already exists',
                    'content_id': existing_content.id,
                    'created': False
                }
            
            release_date = None
            if data.get('release_date'):
                try:
                    from datetime import datetime
                    release_date = datetime.strptime(data['release_date'][:10], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            youtube_trailer_id = None
            try:
                if self.ContentService and hasattr(self.ContentService, 'get_youtube_trailer'):
                    youtube_trailer_id = self.ContentService.get_youtube_trailer(
                        data.get('title'), 
                        data.get('content_type')
                    )
                else:
                    youtube_trailer_id = data.get('youtube_trailer_id') or None
                    logger.info(f"YouTube trailer lookup skipped - ContentService not available or method missing")
            except Exception as e:
                logger.warning(f"YouTube trailer lookup failed for '{data.get('title')}': {e}")
                youtube_trailer_id = None
            
            content_title = data.get('title') or data.get('name') or 'Unknown Title'
            
            if data.get('source') == 'anime':
                genres = data.get('genres', [])
                anime_genres = data.get('anime_genres', [])
                mal_id = int(data['id']) if isinstance(data['id'], str) and data['id'].isdigit() else data['id']
                content = self.Content(
                    mal_id=mal_id,
                    title=content_title,
                    original_title=data.get('original_title') or content_title,
                    content_type='anime',
                    genres=json.dumps(genres),
                    anime_genres=json.dumps(anime_genres),
                    languages=json.dumps(['japanese']),
                    release_date=release_date,
                    rating=data.get('rating') or 0,
                    overview=data.get('overview') or '',
                    poster_path=data.get('poster_path'),
                    youtube_trailer_id=youtube_trailer_id
                )
            else:
                raw_genres = data.get('genres', [])
                genre_ids = data.get('genre_ids', [])
                
                if raw_genres and all(isinstance(g, str) for g in raw_genres):
                    genres = raw_genres
                elif genre_ids:
                    genres = self._map_genre_ids_to_names(genre_ids)
                else:
                    genres = []
                
                tmdb_id = int(data['id']) if isinstance(data['id'], str) and data['id'].isdigit() else data['id']
                content = self.Content(
                    tmdb_id=tmdb_id,
                    title=content_title,
                    original_title=data.get('original_title') or content_title,
                    content_type=data.get('content_type', 'movie'),
                    genres=json.dumps(genres),
                    languages=json.dumps(data.get('languages', ['en'])),
                    release_date=release_date,
                    runtime=data.get('runtime'),
                    rating=data.get('rating') or data.get('vote_average') or 0,
                    vote_count=data.get('vote_count') or 0,
                    popularity=data.get('popularity') or 0,
                    overview=data.get('overview') or '',
                    poster_path=data.get('poster_path'),
                    backdrop_path=data.get('backdrop_path'),
                    youtube_trailer_id=youtube_trailer_id
                )
            
            try:
                if hasattr(content, 'ensure_slug'):
                    content.ensure_slug()
                else:
                    if content.title:
                        import re
                        slug = re.sub(r'[^\w\s-]', '', content.title.lower())
                        slug = re.sub(r'[-\s]+', '-', slug).strip('-')
                        content.slug = slug[:150] if slug else f"content-{content.tmdb_id or content.mal_id or 'unknown'}"
                    logger.info(f"Generated fallback slug: {content.slug}")
            except Exception as e:
                logger.warning(f"Slug generation failed for '{content.title}': {e}")
                content.slug = f"content-{content.tmdb_id or content.mal_id or int(datetime.utcnow().timestamp())}"
            
            self.db.session.add(content)
            self.db.session.commit()
            
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.CONTENT_ADDED,
                        f"New {content.content_type.title()} Added",
                        f"'{content.title}' has been added to CineBrain\n"
                        f"Rating: {content.rating or 'N/A'}/10\n"
                        f"Type: {content.content_type}",
                        related_content_id=content.id,
                        action_url=f"/admin/content/{content.id}",
                        alert_type='content_added'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create content notification: {e}")
            
            logger.info(f"✅ External content saved: {content.title}")
            return {
                'message': 'Content saved successfully',
                'content_id': content.id,
                'created': True
            }
            
        except Exception as e:
            try:
                self.db.session.rollback()
            except:
                pass
            logger.error(f"Error saving content: {e}")
            raise e
    
    def create_recommendation_from_external_content(self, admin_user, content_data, recommendation_type, description, status='draft', publish_to_telegram=False, template_type='auto', template_params=None):
        try:
            content = None
            
            if not content_data or not content_data.get('id'):
                logger.error("Invalid content data for recommendation creation")
                return {'error': 'Invalid content data'}
            
            if content_data.get('source') == 'anime' and content_data.get('id'):
                mal_id = content_data['id']
                content = self.Content.query.filter_by(mal_id=mal_id).first()
            elif content_data.get('id'):
                tmdb_id = content_data['id']
                content = self.Content.query.filter_by(tmdb_id=tmdb_id).first()
            
            if not content:
                content_result = self.save_external_content(content_data)
                if content_result.get('error'):
                    return content_result
                
                content_id = content_result.get('content_id')
                content = self.Content.query.get(content_id)
            
            if not content:
                logger.error("Failed to create or find content for recommendation")
                return {'error': 'Failed to create content'}
            
            recommendation_type = recommendation_type or 'general'
            description = description or f"Recommended: {content.title}"
            template_params = template_params or {}

            admin_rec = self.AdminRecommendation(
                content_id=content.id,
                admin_id=admin_user.id,
                recommendation_type=recommendation_type,
                description=description,
                is_active=(status == 'active')
            )
            
            if hasattr(admin_rec, 'template_type'):
                admin_rec.template_type = template_type
            if hasattr(admin_rec, 'template_data'):
                admin_rec.template_data = template_params
            
            if hasattr(admin_rec, 'hook_text') and template_params.get('hook'):
                admin_rec.hook_text = template_params.get('hook', '')[:500] if template_params.get('hook') else None
            if hasattr(admin_rec, 'if_you_like') and template_params.get('if_you_like'):
                admin_rec.if_you_like = template_params.get('if_you_like', '')[:500] if template_params.get('if_you_like') else None
            if hasattr(admin_rec, 'custom_overview') and template_params.get('overview'):
                admin_rec.custom_overview = template_params.get('overview', '')[:1000] if template_params.get('overview') else None
            if hasattr(admin_rec, 'emotion_hook') and template_params.get('emotion_hook'):
                admin_rec.emotion_hook = template_params.get('emotion_hook', '')[:200] if template_params.get('emotion_hook') else None
            if hasattr(admin_rec, 'scene_caption') and template_params.get('caption'):
                admin_rec.scene_caption = template_params.get('caption', '')[:200] if template_params.get('caption') else None
            if hasattr(admin_rec, 'list_title') and template_params.get('list_title'):
                admin_rec.list_title = template_params.get('list_title', '')[:200] if template_params.get('list_title') else None
            if hasattr(admin_rec, 'list_items') and template_params.get('items'):
                admin_rec.list_items = template_params.get('items', []) if template_params.get('items') else None
            if hasattr(admin_rec, 'hashtags') and template_params.get('hashtags'):
                admin_rec.hashtags = template_params.get('hashtags', '')[:500] if template_params.get('hashtags') else None
            
            if hasattr(admin_rec, 'last_template_edit'):
                admin_rec.last_template_edit = datetime.utcnow()
            
            self.db.session.add(admin_rec)
            self.db.session.commit()

            telegram_sent = False
            if publish_to_telegram and status == 'active':
                if self.telegram_service:
                    try:
                        template_params = template_params or {}
                        telegram_sent = self.telegram_service.send_admin_recommendation(
                            content, admin_user.username, description, template_type, template_params
                        )
                        logger.info(f"✅ Telegram recommendation sent with {template_type} template: {content.title}")
                    except Exception as e:
                        logger.warning(f"Telegram send failed: {e}")
            
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.CONTENT_ADDED,
                        f"New {'Published' if status == 'active' else 'Draft'} Recommendation Created",
                        f"Admin {admin_user.username} created a recommendation for '{content.title}'\n"
                        f"Type: {recommendation_type}\n"
                        f"Status: {status}\n"
                        f"Template: {template_type}\n"
                        f"Description: {description[:100]}...",
                        admin_id=admin_user.id,
                        related_content_id=content.id,
                        action_url=f"/admin/recommendations/{admin_rec.id}",
                        metadata={
                            'template_type': template_type,
                            'template_params': template_params
                        },
                        alert_type='recommendation_created'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create recommendation notification: {e}")
            
            logger.info(f"✅ Recommendation created with template fields saved: {content.title} by {admin_user.username}")
            return {
                'success': True,
                'message': f'Recommendation {"published" if status == "active" else "saved as draft"}',
                'template_fields_saved': len(template_params),
                'telegram_sent': telegram_sent,
                'recommendation_id': admin_rec.id,
                'content_id': content.id,
                'status': status,
                'template_type': template_type
            }
            
        except Exception as e:
            try:
                self.db.session.rollback()
            except:
                pass
            logger.error(f"Create recommendation from external content error: {e}")
            raise e
    
    def create_recommendation(self, admin_user, content_id, recommendation_type, description, template_type='auto', template_params=None):
        try:
            content = self.Content.query.get(content_id)
            if not content:
                content = self.Content.query.filter_by(tmdb_id=content_id).first()
            
            if not content:
                logger.warning(f"Content not found for ID: {content_id}")
                return {'error': 'Content not found. Please save content first.'}
            
            recommendation_type = recommendation_type or 'general'
            description = description or f"Recommended: {content.title}"
            
            admin_rec = self.AdminRecommendation(
                content_id=content.id,
                admin_id=admin_user.id,
                recommendation_type=recommendation_type,
                description=description
            )
            
            self.db.session.add(admin_rec)
            self.db.session.commit()
            
            telegram_success = False
            if self.telegram_service:
                try:
                    template_params = template_params or {}
                    telegram_success = self.telegram_service.send_admin_recommendation(
                        content, admin_user.username, description, template_type, template_params
                    )
                    logger.info(f"✅ Telegram recommendation sent with {template_type} template: {content.title}")
                except Exception as e:
                    logger.warning(f"Telegram send failed: {e}")
            
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.CONTENT_ADDED,
                        f"New Admin Recommendation Created",
                        f"Admin {admin_user.username} recommended '{content.title}'\n"
                        f"Type: {recommendation_type}\n"
                        f"Template: {template_type}\n"
                        f"Description: {description[:100]}...",
                        admin_id=admin_user.id,
                        related_content_id=content.id,
                        action_url=f"/admin/recommendations/{admin_rec.id}",
                        metadata={
                            'template_type': template_type,
                            'template_params': template_params
                        },
                        alert_type='recommendation_created'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create recommendation notification: {e}")
            
            logger.info(f"✅ Admin recommendation created with {template_type} template: {content.title} by {admin_user.username}")
            return {
                'message': 'Admin recommendation created successfully',
                'telegram_sent': telegram_success,
                'recommendation_id': admin_rec.id,
                'template_type': template_type
            }
            
        except Exception as e:
            try:
                self.db.session.rollback()
            except:
                pass
            logger.error(f"Admin recommendation error: {e}")
            raise e
    
    def get_recommendations(self, page=1, per_page=20, filter_type='all', status=None):
        try:
            query = self.AdminRecommendation.query
            
            if status == 'draft':
                query = query.filter_by(is_active=False)
            elif status == 'active':
                query = query.filter_by(is_active=True)
            elif filter_type == 'active':
                query = query.filter_by(is_active=True)
            elif filter_type == 'inactive':
                query = query.filter_by(is_active=False)
            
            admin_recs = query.order_by(
                self.AdminRecommendation.created_at.desc()
            ).paginate(page=page, per_page=per_page, error_out=False)
            
            result = []
            for rec in admin_recs.items:
                try:
                    content = self.Content.query.get(rec.content_id)
                    admin = self.User.query.get(rec.admin_id)
                    
                    if content and admin:
                        updated_at = getattr(rec, 'updated_at', None)
                        if updated_at is None:
                            updated_at = rec.created_at
                        
                        recommendation_data = {
                            'id': rec.id,
                            'recommendation_type': rec.recommendation_type or 'general',
                            'description': rec.description or '',
                            'is_active': rec.is_active if rec.is_active is not None else False,
                            'created_at': rec.created_at.isoformat(),
                            'updated_at': updated_at.isoformat(),
                            'admin_name': admin.username or 'Unknown Admin',
                            'admin_id': admin.id,
                            'content': {
                                'id': content.id,
                                'title': content.title or 'Unknown Title',
                                'content_type': content.content_type or 'movie',
                                'rating': content.rating or 0,
                                'release_date': content.release_date.isoformat() if content.release_date else None,
                                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                                'overview': content.overview or ''
                            }
                        }
                        result.append(recommendation_data)
                except Exception as e:
                    logger.error(f"Error processing recommendation {rec.id}: {e}")
                    continue
            
            logger.info(f"✅ Retrieved {len(result)} admin recommendations")
            return {
                'recommendations': result,
                'total': admin_recs.total,
                'pages': admin_recs.pages,
                'current_page': page,
                'per_page': per_page,
                'has_prev': admin_recs.has_prev,
                'has_next': admin_recs.has_next
            }
            
        except Exception as e:
            logger.error(f"Get admin recommendations error: {e}")
            try:
                self.db.session.rollback()
            except:
                pass
            return {
                'recommendations': [],
                'total': 0,
                'pages': 0,
                'current_page': page,
                'per_page': per_page,
                'has_prev': False,
                'has_next': False,
                'error': 'Failed to get recommendations'
            }
    
    def get_recommendation_details(self, recommendation_id):
        try:
            recommendation = self.AdminRecommendation.query.get(recommendation_id)
            if not recommendation:
                logger.warning(f"Recommendation not found: {recommendation_id}")
                return None

            content = self.Content.query.get(recommendation.content_id)
            admin = self.User.query.get(recommendation.admin_id)

            if not content or not admin:
                logger.warning(f"Missing content or admin for recommendation {recommendation_id}")
                return None

            updated_at = getattr(recommendation, 'updated_at', None)
            if updated_at is None:
                updated_at = recommendation.created_at

            template_fields = {}
            
            if hasattr(recommendation, 'hook_text') and recommendation.hook_text:
                template_fields['hook'] = recommendation.hook_text
            if hasattr(recommendation, 'if_you_like') and recommendation.if_you_like:
                template_fields['if_you_like'] = recommendation.if_you_like
            if hasattr(recommendation, 'custom_overview') and recommendation.custom_overview:
                template_fields['overview'] = recommendation.custom_overview
            if hasattr(recommendation, 'emotion_hook') and recommendation.emotion_hook:
                template_fields['emotion_hook'] = recommendation.emotion_hook
            if hasattr(recommendation, 'scene_caption') and recommendation.scene_caption:
                template_fields['caption'] = recommendation.scene_caption
            if hasattr(recommendation, 'list_title') and recommendation.list_title:
                template_fields['list_title'] = recommendation.list_title
            if hasattr(recommendation, 'list_items') and recommendation.list_items:
                template_fields['items'] = recommendation.list_items
            if hasattr(recommendation, 'hashtags') and recommendation.hashtags:
                template_fields['hashtags'] = recommendation.hashtags

            if hasattr(recommendation, 'template_data') and recommendation.template_data:
                template_fields.update(recommendation.template_data)

            return {
                'id': recommendation.id,
                'recommendation_type': recommendation.recommendation_type or 'general',
                'description': recommendation.description or '',
                'is_active': recommendation.is_active if recommendation.is_active is not None else False,
                'created_at': recommendation.created_at.isoformat(),
                'updated_at': updated_at.isoformat(),
                'admin_name': admin.username or 'Unknown Admin',
                'admin_id': admin.id,
                
                'template_type': getattr(recommendation, 'template_type', None),
                'template_fields': template_fields,
                'template_data': {
                    'selected_template': getattr(recommendation, 'template_type', None),
                    'template_fields': template_fields,
                    'last_edited': getattr(recommendation, 'last_template_edit', recommendation.updated_at).isoformat() if hasattr(recommendation, 'last_template_edit') else updated_at.isoformat()
                },
                
                'content': {
                    'id': content.id,
                    'title': content.title or 'Unknown Title',
                    'content_type': content.content_type or 'movie',
                    'rating': content.rating or 0,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'overview': content.overview or ''
                }
            }
            
        except Exception as e:
            logger.error(f"Get recommendation details error: {e}")
            return None
    
    def update_recommendation(self, admin_user, recommendation_id, data):
        try:
            recommendation = self.AdminRecommendation.query.get(recommendation_id)
            if not recommendation:
                logger.warning(f"Recommendation not found for update: {recommendation_id}")
                return None

            if 'recommendation_type' in data:
                recommendation.recommendation_type = data['recommendation_type'] or 'general'
            if 'description' in data:
                recommendation.description = data['description'] or ''
            if 'is_active' in data:
                recommendation.is_active = data['is_active'] if data['is_active'] is not None else False

            if 'template_data' in data and data['template_data']:
                template_data = data['template_data']
                
                if 'selected_template' in template_data:
                    if hasattr(recommendation, 'template_type'):
                        recommendation.template_type = template_data['selected_template']
                if 'template_fields' in template_data:
                    if hasattr(recommendation, 'template_data'):
                        recommendation.template_data = template_data['template_fields']
                    
                    template_fields = template_data['template_fields']
                    
                    if 'hook' in template_fields and hasattr(recommendation, 'hook_text'):
                        recommendation.hook_text = template_fields['hook'][:500] if template_fields['hook'] else None
                    if 'if_you_like' in template_fields and hasattr(recommendation, 'if_you_like'):
                        recommendation.if_you_like = template_fields['if_you_like'][:500] if template_fields['if_you_like'] else None
                    if 'overview' in template_fields and hasattr(recommendation, 'custom_overview'):
                        recommendation.custom_overview = template_fields['overview'][:1000] if template_fields['overview'] else None
                    if 'emotion_hook' in template_fields and hasattr(recommendation, 'emotion_hook'):
                        recommendation.emotion_hook = template_fields['emotion_hook'][:200] if template_fields['emotion_hook'] else None
                    if 'caption' in template_fields and hasattr(recommendation, 'scene_caption'):
                        recommendation.scene_caption = template_fields['caption'][:200] if template_fields['caption'] else None
                    if 'list_title' in template_fields and hasattr(recommendation, 'list_title'):
                        recommendation.list_title = template_fields['list_title'][:200] if template_fields['list_title'] else None
                    if 'items' in template_fields and hasattr(recommendation, 'list_items'):
                        recommendation.list_items = template_fields['items'] if template_fields['items'] else None
                    if 'hashtags' in template_fields and hasattr(recommendation, 'hashtags'):
                        recommendation.hashtags = template_fields['hashtags'][:500] if template_fields['hashtags'] else None
                    
                    if hasattr(recommendation, 'last_template_edit'):
                        recommendation.last_template_edit = datetime.utcnow()

            if hasattr(recommendation, 'updated_at'):
                recommendation.updated_at = datetime.utcnow()

            self.db.session.commit()

            if self.notification_service:
                try:
                    content = self.Content.query.get(recommendation.content_id)
                    content_title = content.title if content else 'Unknown'
                    self.notification_service.create_notification(
                        NotificationType.CONTENT_ADDED,
                        f"Recommendation Updated",
                        f"Admin {admin_user.username} updated recommendation for '{content_title}'\nTemplate fields updated: {len(data.get('template_data', {}).get('template_fields', {}))} fields",
                        admin_id=admin_user.id,
                        related_content_id=recommendation.content_id,
                        action_url=f"/admin/recommendations/{recommendation.id}",
                        alert_type='recommendation_updated'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create update notification: {e}")

            logger.info(f"✅ Recommendation updated with template fields: {recommendation.id} by {admin_user.username}")
            return {
                'success': True,
                'message': 'Recommendation updated successfully',
                'recommendation_id': recommendation.id,
                'template_fields_updated': len(data.get('template_data', {}).get('template_fields', {}))
            }
            
        except Exception as e:
            try:
                self.db.session.rollback()
            except:
                pass
            logger.error(f"Update recommendation error: {e}")
            raise e
    
    def delete_recommendation(self, admin_user, recommendation_id):
        try:
            recommendation = self.AdminRecommendation.query.get(recommendation_id)
            if not recommendation:
                logger.warning(f"Recommendation not found for deletion: {recommendation_id}")
                return None
            
            content = self.Content.query.get(recommendation.content_id)
            content_title = content.title if content else 'Unknown'
            
            self.db.session.delete(recommendation)
            self.db.session.commit()
            
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.CONTENT_ADDED,
                        f"Recommendation Deleted",
                        f"Admin {admin_user.username} deleted recommendation for '{content_title}'",
                        admin_id=admin_user.id,
                        related_content_id=recommendation.content_id if content else None,
                        alert_type='recommendation_deleted'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create delete notification: {e}")
            
            logger.info(f"✅ Recommendation deleted: {recommendation_id} by {admin_user.username}")
            return {
                'success': True,
                'message': 'Recommendation deleted successfully'
            }
            
        except Exception as e:
            try:
                self.db.session.rollback()
            except:
                pass
            logger.error(f"Delete recommendation error: {e}")
            raise e
    
    def publish_recommendation(self, admin_user, recommendation_id, template_type='auto', template_params=None):
        try:
            recommendation = self.AdminRecommendation.query.get(recommendation_id)
            if not recommendation:
                logger.warning(f"Recommendation not found for publish: {recommendation_id}")
                return None
            
            content = self.Content.query.get(recommendation.content_id)
            if not content:
                logger.error(f"Associated content not found for recommendation: {recommendation_id}")
                return {'error': 'Associated content not found'}
            
            recommendation.is_active = True
            if hasattr(recommendation, 'updated_at'):
                recommendation.updated_at = datetime.utcnow()
            
            self.db.session.commit()
            
            telegram_sent = False
            if self.telegram_service:
                try:
                    template_params = template_params or {}
                    telegram_sent = self.telegram_service.send_admin_recommendation(
                        content, admin_user.username, recommendation.description, template_type, template_params
                    )
                    logger.info(f"✅ Telegram recommendation sent with {template_type} template: {content.title}")
                except Exception as e:
                    logger.warning(f"Telegram send failed: {e}")
            
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.CONTENT_ADDED,
                        f"Recommendation Published",
                        f"Admin {admin_user.username} published recommendation for '{content.title}'\n"
                        f"Type: {recommendation.recommendation_type}\n"
                        f"Template: {template_type}\n"
                        f"Telegram sent: {'Yes' if telegram_sent else 'No'}",
                        admin_id=admin_user.id,
                        related_content_id=content.id,
                        action_url=f"/admin/recommendations/{recommendation.id}",
                        metadata={
                            'template_type': template_type,
                            'template_params': template_params
                        },
                        alert_type='recommendation_published'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create publish notification: {e}")
            
            logger.info(f"✅ Recommendation published with {template_type} template: {content.title} by {admin_user.username}")
            return {
                'success': True,
                'message': 'Recommendation published successfully',
                'telegram_sent': telegram_sent,
                'recommendation_id': recommendation.id,
                'template_type': template_type
            }
            
        except Exception as e:
            try:
                self.db.session.rollback()
            except:
                pass
            logger.error(f"Publish recommendation error: {e}")
            raise e
    
    def send_recommendation_to_telegram(self, admin_user, recommendation_id, template_type='auto', template_params=None):
        try:
            recommendation = self.AdminRecommendation.query.get(recommendation_id)
            if not recommendation:
                logger.warning(f"Recommendation not found for Telegram send: {recommendation_id}")
                return None
            
            content = self.Content.query.get(recommendation.content_id)
            if not content:
                logger.error(f"Associated content not found for recommendation: {recommendation_id}")
                return {'error': 'Associated content not found'}
            
            telegram_sent = False
            if self.telegram_service:
                try:
                    template_params = template_params or {}
                    telegram_sent = self.telegram_service.send_admin_recommendation(
                        content, admin_user.username, recommendation.description, template_type, template_params
                    )
                    logger.info(f"✅ Telegram recommendation sent with {template_type} template: {content.title}")
                except Exception as e:
                    logger.warning(f"Telegram send failed: {e}")
                    return {'error': f'Failed to send to Telegram: {str(e)}'}
            else:
                return {'error': 'Telegram service not configured'}
            
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.CONTENT_ADDED,
                        f"Recommendation Sent to Telegram",
                        f"Admin {admin_user.username} sent recommendation for '{content.title}' to Telegram\n"
                        f"Template: {template_type}",
                        admin_id=admin_user.id,
                        related_content_id=content.id,
                        action_url=f"/admin/recommendations/{recommendation.id}",
                        metadata={
                            'template_type': template_type,
                            'template_params': template_params
                        },
                        alert_type='recommendation_published'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create send notification: {e}")
            
            logger.info(f"✅ Recommendation sent to Telegram with {template_type} template: {content.title} by {admin_user.username}")
            return {
                'success': True,
                'message': 'Recommendation sent to Telegram successfully',
                'telegram_sent': telegram_sent,
                'template_type': template_type
            }
            
        except Exception as e:
            logger.error(f"Send recommendation to Telegram error: {e}")
            raise e
    
    def send_custom_telegram_message(self, admin_user, template_type, template_params):
        try:
            if not self.telegram_service:
                return {'error': 'Telegram service not configured'}
            
            if not template_params:
                return {'error': 'Template parameters are required'}
            
            if template_type == 'top_list':
                telegram_sent = self.telegram_service.send_admin_recommendation(
                    None, admin_user.username, '', template_type, template_params
                )
            else:
                return {'error': 'Invalid template type for custom message'}
            
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.CONTENT_ADDED,
                        f"Custom {template_type.replace('_', ' ').title()} Sent",
                        f"Admin {admin_user.username} sent custom {template_type} message to Telegram\n"
                        f"Title: {template_params.get('list_title', 'Custom List')}",
                        admin_id=admin_user.id,
                        action_url="/admin/telegram/templates",
                        metadata={
                            'template_type': template_type,
                            'template_params': template_params
                        },
                        alert_type='recommendation_published'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create custom message notification: {e}")
            
            logger.info(f"✅ Custom {template_type} message sent by {admin_user.username}")
            return {
                'success': True,
                'message': f'Custom {template_type.replace("_", " ").title()} sent successfully',
                'telegram_sent': telegram_sent,
                'template_type': template_type
            }
            
        except Exception as e:
            logger.error(f"Send custom telegram message error: {e}")
            return {'error': 'Failed to send custom message'}
    
    def migrate_all_slugs(self, batch_size=50):
        try:
            from services.details import SlugManager
            
            stats = {
                'content_updated': 0,
                'persons_updated': 0,
                'total_processed': 0,
                'errors': 0
            }
            
            content_items = self.Content.query.filter(
                or_(self.Content.slug == None, self.Content.slug == '')
            ).limit(batch_size).all()
            
            for content in content_items:
                try:
                    if hasattr(content, 'ensure_slug'):
                        content.ensure_slug()
                        stats['content_updated'] += 1
                    else:
                        if content.title:
                            import re
                            slug = re.sub(r'[^\w\s-]', '', content.title.lower())
                            slug = re.sub(r'[-\s]+', '-', slug)
                            content.slug = slug[:150]
                            stats['content_updated'] += 1
                except Exception as e:
                    logger.error(f"Error updating slug for content {content.id}: {e}")
                    stats['errors'] += 1
            
            self.db.session.commit()
            stats['total_processed'] = stats['content_updated'] + stats['persons_updated']
            
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.SYSTEM_ALERT,
                        f"Slug Migration Completed",
                        f"Migrated {stats['content_updated']} content slugs\n"
                        f"Errors: {stats['errors']}\n"
                        f"Total processed: {stats['total_processed']}",
                        action_url="/admin/content/manage",
                        alert_type='bulk_operation'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create slug migration notification: {e}")
            
            logger.info(f"✅ Slug migration completed: {stats['content_updated']} content items updated")
            return stats
            
        except Exception as e:
            logger.error(f"Slug migration error: {e}")
            try:
                self.db.session.rollback()
            except:
                pass
            raise e
    
    def update_content_slug(self, content_id, force_update=False):
        try:
            content = self.Content.query.get(content_id)
            if not content:
                logger.warning(f"Content not found for slug update: {content_id}")
                return None
            
            if content.slug and not force_update:
                return content.slug
            
            if hasattr(content, 'ensure_slug'):
                content.ensure_slug()
            else:
                if content.title:
                    import re
                    slug = re.sub(r'[^\w\s-]', '', content.title.lower())
                    slug = re.sub(r'[-\s]+', '-', slug)
                    content.slug = slug[:150]
            
            self.db.session.commit()
            
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.SYSTEM_ALERT,
                        f"Content Slug Updated",
                        f"Updated slug for '{content.title}'\n"
                        f"New slug: {content.slug}",
                        related_content_id=content.id,
                        action_url=f"/admin/content/{content.id}",
                        alert_type='slug_update'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create slug update notification: {e}")
            
            logger.info(f"✅ Content slug updated: {content.title} -> {content.slug}")
            return content.slug
            
        except Exception as e:
            logger.error(f"Error updating content slug: {e}")
            try:
                self.db.session.rollback()
            except:
                pass
            return None
    
    def populate_cast_crew(self, batch_size=10):
        try:
            result = {
                'processed': 0,
                'errors': 0
            }
            content_items = self.Content.query.filter(
                self.Content.tmdb_id.isnot(None)
            ).limit(batch_size).all()
            
            for content in content_items:
                try:
                    result['processed'] += 1
                    logger.info(f"Would populate cast/crew for {content.title}")
                except Exception as e:
                    logger.error(f"Error processing cast/crew for {content.title}: {e}")
                    result['errors'] += 1
            
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.SYSTEM_ALERT,
                        f"Cast/Crew Population Completed",
                        f"Processed {result['processed']} content items\n"
                        f"Errors: {result['errors']}",
                        action_url="/admin/content/manage",
                        alert_type='bulk_operation'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create cast/crew notification: {e}")
            
            logger.info(f"✅ Cast/crew population completed: {result['processed']} items processed")
            return result
            
        except Exception as e:
            logger.error(f"Cast/crew population error: {e}")
            raise e
    
    def get_notifications(self, page=1, per_page=20, unread_only=False):
        try:
            recent_notifications = []
            if self.notification_service.redis_client:
                try:
                    notifications_json = self.notification_service.redis_client.lrange('admin_notifications', 0, 19)
                    for notif_json in notifications_json:
                        recent_notifications.append(json.loads(notif_json))
                    logger.info(f"✅ Retrieved {len(recent_notifications)} recent notifications from Redis")
                except Exception as e:
                    logger.error(f"Redis notification retrieval error: {e}")
            
            db_notifications = []
            pagination_info = {
                'current_page': 1,
                'per_page': per_page,
                'total': 0,
                'pages': 0,
                'has_prev': False,
                'has_next': False
            }
            
            if self.AdminNotification:
                try:
                    query = self.AdminNotification.query
                    
                    if unread_only:
                        query = query.filter_by(is_read=False)
                    
                    notifications = query.order_by(
                        self.AdminNotification.created_at.desc()
                    ).paginate(page=page, per_page=per_page, error_out=False)
                    
                    for notif in notifications.items:
                        db_notifications.append({
                            'id': notif.id,
                            'type': notif.notification_type,
                            'title': notif.title,
                            'message': notif.message,
                            'is_read': notif.is_read,
                            'is_urgent': notif.is_urgent,
                            'action_required': notif.action_required,
                            'action_url': notif.action_url,
                            'metadata': notif.notification_metadata,
                            'created_at': notif.created_at.isoformat(),
                            'read_at': notif.read_at.isoformat() if notif.read_at else None
                        })
                    
                    pagination_info = {
                        'current_page': page,
                        'per_page': per_page,
                        'total': notifications.total,
                        'pages': notifications.pages,
                        'has_prev': notifications.has_prev,
                        'has_next': notifications.has_next
                    }
                    
                    logger.info(f"✅ Retrieved {len(db_notifications)} database notifications")
                    
                except Exception as e:
                    logger.error(f"Database notification retrieval error: {e}")
                    try:
                        self.db.session.rollback()
                    except:
                        pass
            
            return {
                'recent_notifications': recent_notifications,
                'notifications': db_notifications,
                'pagination': pagination_info
            }
            
        except Exception as e:
            logger.error(f"Get admin notifications error: {e}")
            return {'error': 'Failed to get notifications'}
    
    def mark_all_notifications_read(self):
        try:
            if not self.AdminNotification:
                logger.warning("AdminNotification model not available")
                return {'error': 'Notification system not available'}
            
            updated_count = self.AdminNotification.query.filter_by(is_read=False).update({
                'is_read': True,
                'read_at': datetime.utcnow()
            })
            
            self.db.session.commit()
            
            logger.info(f"✅ Marked {updated_count} notifications as read")
            return {
                'success': True,
                'message': f'Marked {updated_count} notifications as read'
            }
            
        except Exception as e:
            logger.error(f"Mark all notifications read error: {e}")
            try:
                self.db.session.rollback()
            except:
                pass
            return {'error': 'Failed to mark all notifications as read'}
    
    def clear_cache(self, cache_type='all'):
        try:
            if cache_type == 'all':
                self.cache.clear()
                message = 'All cache cleared'
            elif cache_type == 'search':
                self.cache.delete_memoized(self.TMDBService.search_content)
                self.cache.delete_memoized(self.JikanService.search_anime)
                message = 'Search cache cleared'
            elif cache_type == 'recommendations':
                message = 'Recommendations cache cleared'
            else:
                return {'error': 'Invalid cache type'}
            
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.SYSTEM_ALERT,
                        f"Cache Cleared",
                        f"Cache type '{cache_type}' has been cleared\n"
                        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                        action_url="/admin/cache/stats",
                        alert_type='cache_operation'
                    )
                except Exception as e:
                    logger.warning(f"Failed to create cache clear notification: {e}")
            
            logger.info(f"✅ Cache cleared: {cache_type}")
            return {
                'success': True,
                'message': message,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return {'error': 'Failed to clear cache'}
    
    def get_canned_responses(self, category_id=None):
        try:
            if not self.CannedResponse:
                logger.warning("CannedResponse model not available")
                return []
            
            query = self.CannedResponse.query.filter_by(is_active=True)
            
            if category_id:
                query = query.filter_by(category_id=category_id)
            
            responses = query.order_by(self.CannedResponse.usage_count.desc()).all()
            
            result = []
            for response in responses:
                result.append({
                    'id': response.id,
                    'title': response.title,
                    'content': response.content,
                    'tags': response.tags or [],
                    'usage_count': response.usage_count,
                    'created_at': response.created_at.isoformat()
                })
            
            logger.info(f"✅ Retrieved {len(result)} canned responses")
            return result
            
        except Exception as e:
            logger.error(f"Get canned responses error: {e}")
            return []
    
    def create_canned_response(self, admin_user, title, content, category_id=None, tags=None):
        try:
            if not self.CannedResponse:
                return {'error': 'Canned response system not available'}
            
            response = self.CannedResponse(
                title=title,
                content=content,
                category_id=category_id,
                tags=tags or [],
                created_by=admin_user.id
            )
            
            self.db.session.add(response)
            self.db.session.commit()
            
            logger.info(f"✅ Canned response created: {title}")
            return {
                'success': True,
                'response_id': response.id,
                'message': 'Canned response created successfully'
            }
            
        except Exception as e:
            logger.error(f"Create canned response error: {e}")
            try:
                self.db.session.rollback()
            except:
                pass
            return {'error': 'Failed to create canned response'}

def init_admin_service(app, db, models, services):
    try:
        admin_service = AdminService(app, db, models, services)
        logger.info("✅ Admin service initialized successfully")
        return admin_service
    except Exception as e:
        logger.error(f"❌ Failed to initialize admin service: {e}")
        return None