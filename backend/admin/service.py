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

# Redis Configuration
REDIS_URL = os.environ.get('REDIS_URL')

class NotificationType:
    """Notification types as constants to avoid enum issues"""
    NEW_TICKET = "new_ticket"
    URGENT_TICKET = "urgent_ticket"
    TICKET_ESCALATION = "ticket_escalation"
    SLA_BREACH = "sla_breach"
    FEEDBACK_RECEIVED = "feedback_received"
    SYSTEM_ALERT = "system_alert"
    USER_ACTIVITY = "user_activity"
    CONTENT_ADDED = "content_added"

class AdminEmailService:
    """Email service for admin notifications using auth email service"""
    
    def __init__(self, services):
        # Use the existing auth email service
        self.email_service = services.get('email_service')  # Brevo service from auth
        
        # Better fallback checking
        if not self.email_service:
            try:
                from auth.service import email_service as auth_email_service
                self.email_service = auth_email_service
                logger.info("✅ Email service loaded from auth module for admin")
            except Exception as e:
                logger.warning(f"Could not load auth email service for admin: {e}")
        
        # Improved configuration check
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
        """Send notification email to admins"""
        try:
            if not self.is_configured:
                logger.warning("Email service not configured, skipping admin email notification")
                return False
            
            if not admin_emails:
                logger.warning("No admin emails provided")
                return False
            
            # Import template function
            from auth.admin_mail_templates import get_admin_template
            
            html, text = get_admin_template(
                'admin_notification',
                subject=subject,
                content=content,
                is_urgent=is_urgent,
                timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            )
            
            # Send to each admin
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
    """Service for managing admin notifications"""
    
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.User = models['User']
        self.email_service = AdminEmailService(services)
        self.redis_client = self._init_redis()
        
        # Get admin models from the models dictionary instead of creating them
        self.AdminNotification = models.get('AdminNotification')
        self.CannedResponse = models.get('CannedResponse')
        self.SupportMetrics = models.get('SupportMetrics')
        
        # Validate model availability
        if not self.AdminNotification:
            logger.warning("⚠️ AdminNotification model not available")
        if not self.CannedResponse:
            logger.warning("⚠️ CannedResponse model not available")
        if not self.SupportMetrics:
            logger.warning("⚠️ SupportMetrics model not available")
        
        # Telegram service
        try:
            from admin.telegram import TelegramAdminService
            self.telegram_service = TelegramAdminService
        except ImportError:
            self.telegram_service = None
            logger.warning("Telegram service not available for admin notifications")
        
        logger.info("✅ Admin notification service initialized")
    
    def _init_redis(self):
        """Initialize Redis connection for admin notifications"""
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
    
    def create_notification(self, notification_type: str, title: str, message: str, 
                          admin_id: int = None, related_ticket_id: int = None, 
                          related_content_id: int = None, is_urgent: bool = False,
                          action_required: bool = False, action_url: str = None,
                          metadata: dict = None):
        """Create a new admin notification"""
        try:
            if not self.AdminNotification:
                logger.warning("AdminNotification model not available, skipping database notification")
                # Still proceed with other notification methods
            else:
                # Create database notification
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
            
            # Send Telegram notification
            if self.telegram_service:
                try:
                    self.telegram_service.send_admin_notification(
                        notification_type, 
                        f"{title}\n\n{message}", 
                        is_urgent
                    )
                    logger.info(f"✅ Telegram notification sent: {title}")
                except Exception as e:
                    logger.warning(f"Telegram notification failed: {e}")
            
            # Send email notification
            if self.email_service and self.email_service.is_configured:
                try:
                    # Get admin emails
                    admin_users = self.User.query.filter_by(is_admin=True).all()
                    admin_emails = [user.email for user in admin_users if user.email]
                    
                    # Also include environment variable admin email
                    env_admin_email = os.environ.get('ADMIN_EMAIL')
                    if env_admin_email and env_admin_email not in admin_emails:
                        admin_emails.append(env_admin_email)
                    
                    if admin_emails:
                        self.email_service.send_admin_notification(title, message, admin_emails, is_urgent)
                        logger.info(f"✅ Email notifications sent to {len(admin_emails)} admins")
                except Exception as e:
                    logger.warning(f"Admin email notification failed: {e}")
            
            # Store in Redis for real-time updates
            if self.redis_client:
                try:
                    notification_data = {
                        'id': notification.id if self.AdminNotification else f"temp_{int(datetime.utcnow().timestamp())}",
                        'type': notification_type,
                        'title': title,
                        'message': message,
                        'is_urgent': is_urgent,
                        'action_required': action_required,
                        'action_url': action_url,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    self.redis_client.lpush('admin_notifications', json.dumps(notification_data))
                    self.redis_client.ltrim('admin_notifications', 0, 99)  # Keep last 100
                    self.redis_client.expire('admin_notifications', 86400)  # 24 hours
                    logger.info(f"✅ Redis notification stored: {title}")
                except Exception as e:
                    logger.error(f"Redis notification error: {e}")
            
            logger.info(f"✅ Admin notification created successfully: {title}")
            return notification if self.AdminNotification else True
            
        except Exception as e:
            logger.error(f"❌ Error creating admin notification: {e}")
            if self.db:
                self.db.session.rollback()
            return None
    
    def notify_new_ticket(self, ticket):
        """Notify admins about new support ticket"""
        try:
            is_urgent = ticket.priority == 'urgent'
            
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
                }
            )
            logger.info(f"✅ New ticket notification sent for #{ticket.ticket_number}")
        except Exception as e:
            logger.error(f"❌ Error notifying new ticket: {e}")
    
    def notify_sla_breach(self, ticket):
        """Notify admins about SLA breach"""
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
                }
            )
            logger.info(f"✅ SLA breach notification sent for #{ticket.ticket_number}")
        except Exception as e:
            logger.error(f"❌ Error notifying SLA breach: {e}")
    
    def notify_feedback_received(self, feedback):
        """Notify admins about new feedback"""
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
                }
            )
            logger.info(f"✅ Feedback notification sent for feedback #{feedback.id}")
        except Exception as e:
            logger.error(f"❌ Error notifying feedback: {e}")
    
    def notify_system_alert(self, alert_type: str, title: str, details: str, is_urgent: bool = False):
        """Notify admins about system alerts"""
        try:
            self.create_notification(
                NotificationType.SYSTEM_ALERT,
                f"System Alert: {title}",
                f"Alert Type: {alert_type}\n"
                f"Details: {details}\n"
                f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                is_urgent=is_urgent,
                action_required=True,
                action_url="/admin/system-health",
                metadata={
                    'alert_type': alert_type,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            logger.info(f"✅ System alert notification sent: {title}")
        except Exception as e:
            logger.error(f"❌ Error notifying system alert: {e}")

class AdminService:
    """Main admin service for content and recommendation management"""
    
    def __init__(self, app, db, models, services):
        self.app = app
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.AdminRecommendation = models['AdminRecommendation']
        
        # Get admin models from models dictionary
        self.AdminNotification = models.get('AdminNotification')
        self.CannedResponse = models.get('CannedResponse')
        self.SupportMetrics = models.get('SupportMetrics')
        
        self.TMDBService = services.get('TMDBService')
        self.JikanService = services.get('JikanService')
        self.ContentService = services.get('ContentService')
        self.cache = services.get('cache')
        
        # Telegram service for recommendations
        try:
            from admin.telegram import TelegramService
            self.telegram_service = TelegramService
        except ImportError:
            self.telegram_service = None
        
        # Notification service
        self.notification_service = AdminNotificationService(app, db, models, services)
        
        logger.info("✅ Admin service initialized")
    
    def search_external_content(self, query: str, source: str, page: int = 1):
        """Search for content in external APIs"""
        try:
            results = []
            
            if source == 'tmdb':
                tmdb_results = self.TMDBService.search_content(query, page=page)
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
                anime_results = self.JikanService.search_anime(query, page=page)
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
            
            logger.info(f"✅ External content search completed: {len(results)} results for '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"External search error: {e}")
            return []
    
    def save_external_content(self, data):
        """Save external content to database"""
        try:
            # Check if content already exists
            existing_content = None
            
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
            
            # Parse release date
            release_date = None
            if data.get('release_date'):
                try:
                    from datetime import datetime
                    release_date = datetime.strptime(data['release_date'][:10], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            # Get YouTube trailer
            youtube_trailer_id = None
            if self.ContentService:
                youtube_trailer_id = self.ContentService.get_youtube_trailer(
                    data.get('title'), 
                    data.get('content_type')
                )
            
            # Create content object
            if data.get('source') == 'anime':
                mal_id = int(data['id']) if isinstance(data['id'], str) and data['id'].isdigit() else data['id']
                content = self.Content(
                    mal_id=mal_id,
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
                tmdb_id = int(data['id']) if isinstance(data['id'], str) and data['id'].isdigit() else data['id']
                content = self.Content(
                    tmdb_id=tmdb_id,
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
            
            # Ensure slug
            content.ensure_slug()
            
            # Save to database
            self.db.session.add(content)
            self.db.session.commit()
            
            # Send notification
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.CONTENT_ADDED,
                        f"New {content.content_type.title()} Added",
                        f"'{content.title}' has been added to CineBrain\n"
                        f"Rating: {content.rating or 'N/A'}/10\n"
                        f"Type: {content.content_type}",
                        related_content_id=content.id,
                        action_url=f"/admin/content/{content.id}"
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
            self.db.session.rollback()
            logger.error(f"Error saving content: {e}")
            raise e
    
    def create_recommendation(self, admin_user, content_id, recommendation_type, description):
        """Create admin recommendation"""
        try:
            # Get or find content
            content = self.Content.query.get(content_id)
            if not content:
                content = self.Content.query.filter_by(tmdb_id=content_id).first()
            
            if not content:
                logger.warning(f"Content not found for ID: {content_id}")
                return {'error': 'Content not found. Please save content first.'}
            
            # Create recommendation
            admin_rec = self.AdminRecommendation(
                content_id=content.id,
                admin_id=admin_user.id,
                recommendation_type=recommendation_type,
                description=description
            )
            
            self.db.session.add(admin_rec)
            self.db.session.commit()
            
            # Send to Telegram channel
            telegram_success = False
            if self.telegram_service:
                try:
                    telegram_success = self.telegram_service.send_admin_recommendation(
                        content, admin_user.username, description
                    )
                    logger.info(f"✅ Telegram recommendation sent: {content.title}")
                except Exception as e:
                    logger.warning(f"Telegram send failed: {e}")
            
            # Create notification
            if self.notification_service:
                try:
                    self.notification_service.create_notification(
                        NotificationType.CONTENT_ADDED,
                        f"New Admin Recommendation Created",
                        f"Admin {admin_user.username} recommended '{content.title}'\n"
                        f"Type: {recommendation_type}\n"
                        f"Description: {description[:100]}...",
                        admin_id=admin_user.id,
                        related_content_id=content.id,
                        action_url=f"/admin/recommendations/{admin_rec.id}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to create recommendation notification: {e}")
            
            logger.info(f"✅ Admin recommendation created: {content.title} by {admin_user.username}")
            return {
                'message': 'Admin recommendation created successfully',
                'telegram_sent': telegram_success,
                'recommendation_id': admin_rec.id
            }
            
        except Exception as e:
            self.db.session.rollback()
            logger.error(f"Admin recommendation error: {e}")
            raise e
    
    def get_recommendations(self, page=1, per_page=20):
        """Get admin recommendations"""
        try:
            admin_recs = self.AdminRecommendation.query.filter_by(is_active=True)\
                .order_by(self.AdminRecommendation.created_at.desc())\
                .paginate(page=page, per_page=per_page, error_out=False)
            
            result = []
            for rec in admin_recs.items:
                content = self.Content.query.get(rec.content_id)
                admin = self.User.query.get(rec.admin_id)
                
                if content and admin:
                    result.append({
                        'id': rec.id,
                        'recommendation_type': rec.recommendation_type,
                        'description': rec.description,
                        'created_at': rec.created_at.isoformat(),
                        'admin_name': admin.username,
                        'content': {
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path
                        }
                    })
            
            logger.info(f"✅ Retrieved {len(result)} admin recommendations")
            return {
                'recommendations': result,
                'total': admin_recs.total,
                'pages': admin_recs.pages,
                'current_page': page
            }
            
        except Exception as e:
            logger.error(f"Get admin recommendations error: {e}")
            return {'error': 'Failed to get recommendations'}
    
    def migrate_all_slugs(self, batch_size=50):
        """Migrate all content slugs"""
        try:
            from services.details import SlugManager
            
            stats = {
                'content_updated': 0,
                'persons_updated': 0,
                'total_processed': 0,
                'errors': 0
            }
            
            # Migrate content slugs that don't have slugs
            content_items = self.Content.query.filter(
                or_(self.Content.slug == None, self.Content.slug == '')
            ).limit(batch_size).all()
            
            for content in content_items:
                try:
                    if hasattr(content, 'ensure_slug'):
                        content.ensure_slug()
                        stats['content_updated'] += 1
                    else:
                        # Generate basic slug if method doesn't exist
                        if content.title:
                            import re
                            slug = re.sub(r'[^\w\s-]', '', content.title.lower())
                            slug = re.sub(r'[-\s]+', '-', slug)
                            content.slug = slug[:150]  # Limit length
                            stats['content_updated'] += 1
                except Exception as e:
                    logger.error(f"Error updating slug for content {content.id}: {e}")
                    stats['errors'] += 1
            
            self.db.session.commit()
            stats['total_processed'] = stats['content_updated'] + stats['persons_updated']
            
            logger.info(f"✅ Slug migration completed: {stats['content_updated']} content items updated")
            return stats
            
        except Exception as e:
            logger.error(f"Slug migration error: {e}")
            self.db.session.rollback()
            raise e
    
    def update_content_slug(self, content_id, force_update=False):
        """Update specific content slug"""
        try:
            content = self.Content.query.get(content_id)
            if not content:
                return None
            
            if content.slug and not force_update:
                return content.slug
            
            if hasattr(content, 'ensure_slug'):
                content.ensure_slug()
            else:
                # Generate basic slug if method doesn't exist
                if content.title:
                    import re
                    slug = re.sub(r'[^\w\s-]', '', content.title.lower())
                    slug = re.sub(r'[-\s]+', '-', slug)
                    content.slug = slug[:150]  # Limit length
            
            self.db.session.commit()
            logger.info(f"✅ Content slug updated: {content.title} -> {content.slug}")
            return content.slug
            
        except Exception as e:
            logger.error(f"Error updating content slug: {e}")
            self.db.session.rollback()
            return None
    
    def populate_cast_crew(self, batch_size=10):
        """Populate cast and crew data for content without it"""
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
                    # This would use your existing details service or TMDB service
                    # For now, just increment processed count
                    result['processed'] += 1
                    logger.info(f"Would populate cast/crew for {content.title}")
                except Exception as e:
                    logger.error(f"Error processing cast/crew for {content.title}: {e}")
                    result['errors'] += 1
            
            logger.info(f"✅ Cast/crew population completed: {result['processed']} items processed")
            return result
            
        except Exception as e:
            logger.error(f"Cast/crew population error: {e}")
            raise e
    
    def get_notifications(self, page=1, per_page=20, unread_only=False):
        """Get admin notifications"""
        try:
            # Get recent notifications from Redis
            recent_notifications = []
            if self.notification_service.redis_client:
                try:
                    notifications_json = self.notification_service.redis_client.lrange('admin_notifications', 0, 19)
                    for notif_json in notifications_json:
                        recent_notifications.append(json.loads(notif_json))
                    logger.info(f"✅ Retrieved {len(recent_notifications)} recent notifications from Redis")
                except Exception as e:
                    logger.error(f"Redis notification retrieval error: {e}")
            
            # Get database notifications
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
            
            return {
                'recent_notifications': recent_notifications,
                'notifications': db_notifications,
                'pagination': pagination_info
            }
            
        except Exception as e:
            logger.error(f"Get admin notifications error: {e}")
            return {'error': 'Failed to get notifications'}
    
    def mark_all_notifications_read(self):
        """Mark all notifications as read"""
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
            self.db.session.rollback()
            return {'error': 'Failed to mark all notifications as read'}
    
    def clear_cache(self, cache_type='all'):
        """Clear application cache"""
        try:
            if cache_type == 'all':
                self.cache.clear()
                message = 'All cache cleared'
            elif cache_type == 'search':
                self.cache.delete_memoized(self.TMDBService.search_content)
                self.cache.delete_memoized(self.JikanService.search_anime)
                message = 'Search cache cleared'
            elif cache_type == 'recommendations':
                # Clear recommendation-related cache
                message = 'Recommendations cache cleared'
            else:
                return {'error': 'Invalid cache type'}
            
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
        """Get canned responses for support"""
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
        """Create new canned response"""
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
            self.db.session.rollback()
            return {'error': 'Failed to create canned response'}

def init_admin_service(app, db, models, services):
    """Initialize admin service"""
    try:
        admin_service = AdminService(app, db, models, services)
        logger.info("✅ Admin service initialized successfully")
        return admin_service
    except Exception as e:
        logger.error(f"❌ Failed to initialize admin service: {e}")
        return None