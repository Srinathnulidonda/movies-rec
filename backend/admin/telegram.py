# admin/telegram.py

import os
import json
import logging
import threading
import time
import telebot
from datetime import datetime, timedelta
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

class TelegramService:
    """Service for sending PUBLIC content recommendations to Telegram channel - CONTENT ONLY"""
    
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
    def send_new_content_alert(content, content_count=1):
        """Send alert about new content added to the platform"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("New content alert skipped - channel not configured")
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

class TelegramAdminService:
    """Service for admin-related Telegram notifications - CONTENT MANAGEMENT ONLY"""
    
    @staticmethod
    def send_content_notification(content_title, admin_name, action_type="added"):
        """Send notification about content management actions to admin chat"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                logger.warning("Telegram content notification skipped - admin chat not configured")
                return False
            
            message = f"""
🎬 **Content {action_type.title()}**

**{content_title}**
**Admin:** {admin_name}
**Action:** {action_type}
**Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

#ContentManagement #CineBrain
            """
            
            bot.send_message(
                chat_id=TELEGRAM_ADMIN_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
            
            logger.info(f"✅ Telegram content notification sent: {action_type} - {content_title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram content notification error: {e}")
            return False
    
    @staticmethod
    def send_content_milestone(milestone_type, count, details=""):
        """Send milestone notifications for content achievements"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                return False
            
            milestone_icons = {
                'content_count': '🎬',
                'user_milestone': '👥',
                'recommendation_milestone': '⭐',
                'review_milestone': '📝'
            }
            
            icon = milestone_icons.get(milestone_type, '🎉')
            
            message = f"""
{icon} **Milestone Achieved!**

**Type:** {milestone_type.replace('_', ' ').title()}
**Count:** {count}
**Details:** {details}
**Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

#Milestone #CineBrain #{milestone_type}
            """
            
            bot.send_message(
                chat_id=TELEGRAM_ADMIN_CHAT_ID,
                text=message,
                parse_mode='Markdown'
            )
            
            logger.info(f"✅ Milestone notification sent: {milestone_type} - {count}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Milestone notification error: {e}")
            return False

class TelegramScheduler:
    """Background scheduler for Telegram notifications - CONTENT ONLY"""
    
    def __init__(self, app=None):
        self.app = app
        self.running = False
        # REMOVED: Support-related monitoring
    
    def start_scheduler(self):
        """Start the background scheduler - CONTENT RECOMMENDATIONS ONLY"""
        if self.running:
            return
        
        self.running = True
        
        def scheduler_worker():
            while self.running:
                try:
                    now = datetime.utcnow()
                    
                    # Only content-related periodic tasks
                    # Example: Daily content statistics (if needed)
                    if now.hour == 9 and now.minute == 0:  # 9 AM UTC
                        try:
                            if self.app:
                                with self.app.app_context():
                                    self._send_daily_content_summary()
                        except Exception as e:
                            logger.error(f"Daily content summary error: {e}")
                    
                    # Content milestone checks every 6 hours
                    if now.hour % 6 == 0 and now.minute == 0:
                        try:
                            if self.app:
                                with self.app.app_context():
                                    self._check_content_milestones()
                        except Exception as e:
                            logger.error(f"Content milestone check error: {e}")
                    
                    # Sleep for 1 hour
                    time.sleep(3600)
                    
                except Exception as e:
                    logger.error(f"Telegram content scheduler error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        thread = threading.Thread(target=scheduler_worker, daemon=True, name="TelegramContentScheduler")
        thread.start()
        logger.info("✅ Telegram content scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("🛑 Telegram content scheduler stopped")
    
    def _send_daily_content_summary(self):
        """Send daily content statistics summary"""
        try:
            # This would query your content database for daily stats
            # For now, just a placeholder
            TelegramAdminService.send_content_notification(
                "Daily Summary", 
                "System", 
                "generated daily content report"
            )
            logger.info("✅ Daily content summary sent")
        except Exception as e:
            logger.error(f"Daily content summary error: {e}")
    
    def _check_content_milestones(self):
        """Check for content-related milestones"""
        try:
            # This would check for milestones like:
            # - 1000th movie added
            # - 100th admin recommendation
            # - etc.
            logger.debug("Content milestone check completed")
        except Exception as e:
            logger.error(f"Content milestone check error: {e}")

# Global scheduler instance
telegram_scheduler = None

def init_telegram_service(app, db, models, services):
    """Initialize Telegram service - CONTENT ONLY"""
    global telegram_scheduler
    
    try:
        # Initialize scheduler for content only
        telegram_scheduler = TelegramScheduler(app)
        
        # Start scheduler if bot is configured
        if bot:
            telegram_scheduler.start_scheduler()
            logger.info("✅ Telegram content service initialized successfully")
            logger.info("   - Content recommendations: Active")
            logger.info("   - New content alerts: Active")
            logger.info("   - Admin content notifications: Active")
            logger.info("   - Support notifications: DISABLED (Email Only)")
        else:
            logger.warning("⚠️ Telegram service initialized but bot not configured")
        
        return {
            'telegram_service': TelegramService,  # For content recommendations
            'telegram_admin_service': TelegramAdminService,  # For admin content notifications
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
        logger.info("✅ Telegram content service cleaned up")