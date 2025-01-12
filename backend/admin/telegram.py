# admin/telegram.py

import os
import json
import logging
import threading
import time
import telebot
from datetime import datetime, timedelta
from sqlalchemy import func, and_
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID')
TELEGRAM_ADMIN_CHAT_ID = os.environ.get('TELEGRAM_ADMIN_CHAT_ID')

# Initialize bot
bot = None
if TELEGRAM_BOT_TOKEN:
    try:
        bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
        logger.info("✅ Telegram bot initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Telegram bot: {e}")
        bot = None
else:
    logger.warning("TELEGRAM_BOT_TOKEN not set - Telegram notifications disabled")

class TelegramAdminService:
    """Service for sending admin notifications via Telegram"""
    
    @staticmethod
    def send_admin_notification(notification_type: str, message: str, is_urgent: bool = False):
        """Send notification to admin chat"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                logger.warning("Telegram admin notification skipped - not configured")
                return False
            
            icon_map = {
                'new_ticket': '🎫',
                'urgent_ticket': '🚨',
                'sla_breach': '⚠️',
                'feedback': '📝',
                'system_alert': '🔔',
                'recommendation': '🎬',
                'user_activity': '👤',
                'content_added': '🎭',
                'daily_summary': '📊'
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
            
            logger.info(f"✅ Telegram admin notification sent: {notification_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram admin notification error: {e}")
            return False
    
    @staticmethod
    def send_support_summary(support_ticket_model, feedback_model):
        """Send daily support summary"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                logger.warning("Support summary skipped - Telegram not configured")
                return False
            
            if not support_ticket_model:
                logger.warning("Support summary skipped - support system not available")
                return False
            
            today = datetime.utcnow().date()
            
            # Get ticket statistics
            new_tickets = support_ticket_model.query.filter(
                func.date(support_ticket_model.created_at) == today
            ).count()
            
            resolved_tickets = support_ticket_model.query.filter(
                func.date(support_ticket_model.resolved_at) == today
            ).count()
            
            try:
                from support.tickets import TicketStatus, TicketPriority
                
                open_tickets = support_ticket_model.query.filter(
                    support_ticket_model.status.in_([TicketStatus.OPEN, TicketStatus.IN_PROGRESS])
                ).count()
                
                urgent_tickets = support_ticket_model.query.filter(
                    and_(
                        support_ticket_model.priority == TicketPriority.URGENT,
                        support_ticket_model.status.in_([TicketStatus.OPEN, TicketStatus.IN_PROGRESS])
                    )
                ).count()
                
            except ImportError:
                open_tickets = 0
                urgent_tickets = 0
            
            # Get feedback count
            new_feedback = 0
            if feedback_model:
                new_feedback = feedback_model.query.filter(
                    func.date(feedback_model.created_at) == today
                ).count()
            
            message = f"""📊 **Daily Support Summary**

**Today's Activity:**
🎫 New Tickets: {new_tickets}
✅ Resolved: {resolved_tickets}
📋 Open Tickets: {open_tickets}
🚨 Urgent: {urgent_tickets}
💬 New Feedback: {new_feedback}

**Status:** {'🟢 Good' if urgent_tickets == 0 else '🟡 Attention Needed' if urgent_tickets < 5 else '🔴 Critical'}

#DailySummary #Support #CineBrain"""
            
            return TelegramAdminService.send_admin_notification('daily_summary', message)
            
        except Exception as e:
            logger.error(f"❌ Support summary error: {e}")
            return False
    
    @staticmethod
    def send_system_alert(alert_type: str, title: str, details: str):
        """Send system alert notification"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                logger.warning("System alert skipped - Telegram not configured")
                return False
            
            alert_icons = {
                'database_error': '🔴',
                'cache_error': '🟡',
                'api_error': '🔌',
                'high_load': '📈',
                'security_alert': '🔒',
                'backup_failed': '💾'
            }
            
            icon = alert_icons.get(alert_type, '⚠️')
            
            message = f"""
{icon} **System Alert**

**Alert Type:** {title}
**Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

**Details:**
{details}

**Action Required:** Please check the admin dashboard for more details.

#SystemAlert #CineBrain #{alert_type}
            """
            
            return TelegramAdminService.send_admin_notification('system_alert', message, is_urgent=True)
            
        except Exception as e:
            logger.error(f"❌ System alert error: {e}")
            return False

class TelegramService:
    """Service for sending public content recommendations to Telegram channel"""
    
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        """Send admin recommendation to public channel"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram recommendation skipped - channel not configured")
                return False
            
            # Parse genres
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            # Handle poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Format message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

#AdminChoice #MovieRecommendation #CineBrain"""
            
            # Send with photo if available
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
            
            logger.info(f"✅ Admin recommendation sent to Telegram: {content.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False
    
    @staticmethod
    def send_new_content_alert(content, content_count):
        """Send alert about new content added"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            if content_count > 1:
                message = f"""🎬 **New Content Added!**

Added {content_count} new {content.content_type}s to CineBrain!

Latest addition: **{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10

Check out all the new content on CineBrain! 🍿

#NewContent #CineBrain"""
            else:
                message = f"""🎬 **New {content.content_type.title()} Added!**

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}

Now available on CineBrain! 🍿

#NewContent #CineBrain"""
            
            bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            logger.info(f"✅ New content alert sent: {content.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ New content alert error: {e}")
            return False

class TelegramScheduler:
    """Background scheduler for Telegram notifications"""
    
    def __init__(self, support_models=None):
        self.support_ticket_model = support_models.get('SupportTicket') if support_models else None
        self.feedback_model = support_models.get('Feedback') if support_models else None
        self.running = False
    
    def start_scheduler(self):
        """Start the background scheduler"""
        if self.running:
            return
        
        self.running = True
        
        def scheduler_worker():
            while self.running:
                try:
                    now = datetime.utcnow()
                    
                    # Send daily summary at 9:00 UTC
                    if now.hour == 9 and now.minute == 0:
                        if self.support_ticket_model:
                            TelegramAdminService.send_support_summary(
                                self.support_ticket_model, 
                                self.feedback_model
                            )
                    
                    # Check for urgent tickets every 30 minutes during work hours
                    if 8 <= now.hour <= 20 and now.minute % 30 == 0:
                        self._check_urgent_tickets()
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Telegram scheduler error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        thread = threading.Thread(target=scheduler_worker, daemon=True, name="TelegramScheduler")
        thread.start()
        logger.info("✅ Telegram scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("🛑 Telegram scheduler stopped")
    
    def _check_urgent_tickets(self):
        """Check for urgent tickets and send alerts"""
        try:
            if not self.support_ticket_model:
                return
            
            from support.tickets import TicketStatus, TicketPriority
            
            # Check for urgent tickets without response in last hour
            hour_ago = datetime.utcnow() - timedelta(hours=1)
            urgent_tickets = self.support_ticket_model.query.filter(
                and_(
                    self.support_ticket_model.priority == TicketPriority.URGENT,
                    self.support_ticket_model.status.in_([TicketStatus.OPEN, TicketStatus.IN_PROGRESS]),
                    self.support_ticket_model.first_response_at.is_(None),
                    self.support_ticket_model.created_at >= hour_ago
                )
            ).all()
            
            for ticket in urgent_tickets:
                message = f"""Urgent ticket #{ticket.ticket_number} needs immediate attention!

Subject: {ticket.subject}
From: {ticket.user_name}
Created: {ticket.created_at.strftime('%H:%M UTC')}

No response yet - SLA deadline approaching!"""
                
                TelegramAdminService.send_admin_notification(
                    'urgent_ticket', 
                    message, 
                    is_urgent=True
                )
            
        except ImportError:
            pass  # Support system not available
        except Exception as e:
            logger.error(f"Error checking urgent tickets: {e}")

# Global scheduler instance
telegram_scheduler = None

def init_telegram_service(app, db, models, services):
    """Initialize Telegram service"""
    global telegram_scheduler
    
    try:
        # Initialize scheduler with support models
        support_models = {
            'SupportTicket': models.get('SupportTicket'),
            'Feedback': models.get('Feedback')
        }
        
        telegram_scheduler = TelegramScheduler(support_models)
        
        # Start scheduler if bot is configured
        if bot:
            telegram_scheduler.start_scheduler()
            logger.info("✅ Telegram service initialized successfully")
        else:
            logger.warning("⚠️ Telegram service initialized but bot not configured")
        
        return {
            'telegram_admin_service': TelegramAdminService,
            'telegram_service': TelegramService,
            'telegram_scheduler': telegram_scheduler
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Telegram service: {e}")
        return None

def cleanup_telegram_service():
    """Cleanup Telegram service"""
    global telegram_scheduler
    if telegram_scheduler:
        telegram_scheduler.stop_scheduler()