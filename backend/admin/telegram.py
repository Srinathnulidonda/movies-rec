# admin/telegram.py

import os
import json
import logging
import threading
import time
import telebot
from datetime import datetime, timedelta
from dotenv import load_dotenv
from telebot import types

load_dotenv()

logger = logging.getLogger(__name__)

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID')  # For public recommendations
TELEGRAM_ADMIN_CHAT_ID = os.environ.get('TELEGRAM_ADMIN_CHAT_ID')  # For admin notifications

# Initialize bot
bot = None
if TELEGRAM_BOT_TOKEN:
    try:
        bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode='HTML')
        logger.info("✅ Telegram bot initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Telegram bot: {e}")
        bot = None
else:
    logger.warning("TELEGRAM_BOT_TOKEN not set - Telegram notifications disabled")

class TelegramTemplates:
    """Beautiful templates for different content types"""
    
    @staticmethod
    def get_emoji_for_rating(rating):
        """Get emoji based on rating"""
        if not rating:
            return "⭐"
        elif rating >= 9.0:
            return "🌟"
        elif rating >= 8.0:
            return "⭐"
        elif rating >= 7.0:
            return "✨"
        elif rating >= 6.0:
            return "💫"
        else:
            return "⚡"
    
    @staticmethod
    def get_content_type_emoji(content_type):
        """Get emoji for content type"""
        emoji_map = {
            'movie': '🎬',
            'tv': '📺',
            'anime': '🎌',
            'series': '📺',
            'show': '🎭'
        }
        return emoji_map.get(content_type.lower(), '🎬')
    
    @staticmethod
    def format_runtime(runtime):
        """Format runtime in hours and minutes"""
        if not runtime:
            return "N/A"
        hours = runtime // 60
        minutes = runtime % 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    
    @staticmethod
    def format_genres(genres_list, max_genres=3):
        """Format genres with hashtags"""
        if not genres_list:
            return ""
        
        hashtags = []
        for genre in genres_list[:max_genres]:
            # Clean genre name for hashtag
            clean_genre = genre.replace(' ', '').replace('-', '')
            hashtags.append(f"#{clean_genre}")
        
        return " ".join(hashtags)
    
    @staticmethod
    def movie_recommendation_template(content, admin_name, description, genres_list=None):
        """Template for movie recommendations"""
        
        rating_emoji = TelegramTemplates.get_emoji_for_rating(content.rating)
        content_emoji = TelegramTemplates.get_content_type_emoji(content.content_type)
        
        # Parse genres
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        genres_text = ', '.join(genres_list[:3]) if genres_list else 'N/A'
        genre_hashtags = TelegramTemplates.format_genres(genres_list)
        
        # Format release year
        release_year = ""
        if content.release_date:
            try:
                release_year = content.release_date.strftime('%Y')
            except:
                release_year = str(content.release_date)[:4] if content.release_date else ""
        
        # Build message
        message = f"""╔══════════════════════════╗
{content_emoji} <b>ADMIN'S PICK</b> {content_emoji}
╚══════════════════════════╝

<b>🎬 {content.title}</b>"""
        
        if release_year:
            message += f" <i>({release_year})</i>"
        
        message += f"""

{rating_emoji} <b>Rating:</b> {content.rating or 'N/A'}/10
"""
        
        if content.vote_count:
            message += f"📊 <b>Votes:</b> {content.vote_count:,}\n"
        
        if content.runtime:
            message += f"⏱ <b>Runtime:</b> {TelegramTemplates.format_runtime(content.runtime)}\n"
        
        message += f"🎭 <b>Genres:</b> {genres_text}\n"
        message += f"📱 <b>Type:</b> {content.content_type.upper()}\n"
        
        message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━

💭 <b>Admin's Note</b> by <i>{admin_name}</i>:
<i>"{description}"</i>

━━━━━━━━━━━━━━━━━━━━━━━━

📖 <b>Synopsis:</b>
{(content.overview[:250] + '...') if content.overview and len(content.overview) > 250 else (content.overview or 'No synopsis available.')}

━━━━━━━━━━━━━━━━━━━━━━━━

🔗 <b>Watch on CineBrain</b>
👉 <a href="https://cinebrain.vercel.app/details/{content.slug}">View Details</a>

{genre_hashtags} #AdminPick #CineBrain #MovieRecommendation"""
        
        return message
    
    @staticmethod
    def anime_recommendation_template(content, admin_name, description, genres_list=None, anime_genres_list=None):
        """Template for anime recommendations"""
        
        rating_emoji = TelegramTemplates.get_emoji_for_rating(content.rating)
        
        # Parse genres
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        if not anime_genres_list and content.anime_genres:
            try:
                anime_genres_list = json.loads(content.anime_genres)
            except:
                anime_genres_list = []
        
        genres_text = ', '.join(genres_list[:3]) if genres_list else 'N/A'
        anime_genres_text = ', '.join(anime_genres_list[:3]) if anime_genres_list else ''
        
        all_genres = genres_list + anime_genres_list if anime_genres_list else genres_list
        genre_hashtags = TelegramTemplates.format_genres(all_genres)
        
        # Format release year
        release_year = ""
        if content.release_date:
            try:
                release_year = content.release_date.strftime('%Y')
            except:
                release_year = str(content.release_date)[:4] if content.release_date else ""
        
        message = f"""╔══════════════════════════╗
🎌 <b>ANIME PICK</b> 🎌
╚══════════════════════════╝

<b>⚡ {content.title}</b>"""
        
        if content.original_title and content.original_title != content.title:
            message += f"\n<i>({content.original_title})</i>"
        
        if release_year:
            message += f" • <i>{release_year}</i>"
        
        message += f"""

{rating_emoji} <b>Score:</b> {content.rating or 'N/A'}/10
🎭 <b>Genres:</b> {genres_text}
"""
        
        if anime_genres_text:
            message += f"🏷 <b>Tags:</b> {anime_genres_text}\n"
        
        message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━

💭 <b>Admin's Recommendation</b> by <i>{admin_name}</i>:
<i>"{description}"</i>

━━━━━━━━━━━━━━━━━━━━━━━━

📖 <b>Synopsis:</b>
{(content.overview[:250] + '...') if content.overview and len(content.overview) > 250 else (content.overview or 'No synopsis available.')}

━━━━━━━━━━━━━━━━━━━━━━━━

🔗 <b>Watch on CineBrain</b>
👉 <a href="https://cinebrain.vercel.app/details/{content.slug}">View Details</a>

{genre_hashtags} #AnimeRecommendation #AdminPick #CineBrain"""
        
        return message
    
    @staticmethod
    def tv_show_recommendation_template(content, admin_name, description, genres_list=None):
        """Template for TV show recommendations"""
        
        rating_emoji = TelegramTemplates.get_emoji_for_rating(content.rating)
        
        # Parse genres
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        genres_text = ', '.join(genres_list[:3]) if genres_list else 'N/A'
        genre_hashtags = TelegramTemplates.format_genres(genres_list)
        
        # Format release year
        release_year = ""
        if content.release_date:
            try:
                release_year = content.release_date.strftime('%Y')
            except:
                release_year = str(content.release_date)[:4] if content.release_date else ""
        
        message = f"""╔══════════════════════════╗
📺 <b>SERIES PICK</b> 📺
╚══════════════════════════╝

<b>🎭 {content.title}</b>"""
        
        if release_year:
            message += f" <i>({release_year})</i>"
        
        message += f"""

{rating_emoji} <b>Rating:</b> {content.rating or 'N/A'}/10
"""
        
        if content.vote_count:
            message += f"📊 <b>Votes:</b> {content.vote_count:,}\n"
        
        message += f"🎭 <b>Genres:</b> {genres_text}\n"
        message += f"📱 <b>Type:</b> TV Series\n"
        
        message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━

💭 <b>Why You Should Watch</b> - <i>{admin_name}</i>:
<i>"{description}"</i>

━━━━━━━━━━━━━━━━━━━━━━━━

📖 <b>About the Series:</b>
{(content.overview[:250] + '...') if content.overview and len(content.overview) > 250 else (content.overview or 'No synopsis available.')}

━━━━━━━━━━━━━━━━━━━━━━━━

🔗 <b>Stream on CineBrain</b>
👉 <a href="https://cinebrain.vercel.app/details/{content.slug}">View Details</a>

{genre_hashtags} #TVSeries #AdminPick #CineBrain #Binge"""
        
        return message
    
    @staticmethod
    def weekly_digest_template(recommendations_list, week_number):
        """Template for weekly digest"""
        
        message = f"""╔══════════════════════════╗
🌟 <b>WEEK {week_number} HIGHLIGHTS</b> 🌟
╚══════════════════════════╝

<b>Admin's Top Picks This Week</b>

━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        for i, rec in enumerate(recommendations_list, 1):
            emoji = TelegramTemplates.get_content_type_emoji(rec['content_type'])
            message += f"{i}. {emoji} <b>{rec['title']}</b>\n"
            message += f"   ⭐ {rec['rating']}/10 • {rec['content_type'].upper()}\n"
            message += f"   💭 <i>{rec['description'][:50]}...</i>\n\n"
        
        message += f"""━━━━━━━━━━━━━━━━━━━━━━━━

🔗 <b>Explore More on CineBrain</b>
👉 <a href="https://cinebrain.vercel.app">Visit Now</a>

#WeeklyPicks #CineBrain #MovieRecommendations"""
        
        return message
    
    @staticmethod
    def trending_alert_template(content, trend_type="rising"):
        """Template for trending content alerts"""
        
        emoji_map = {
            'rising': '📈',
            'hot': '🔥',
            'viral': '💥',
            'popular': '⭐'
        }
        
        trend_emoji = emoji_map.get(trend_type, '📈')
        content_emoji = TelegramTemplates.get_content_type_emoji(content.content_type)
        
        # Parse genres
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                pass
        
        genres_text = ', '.join(genres_list[:3]) if genres_list else 'N/A'
        
        message = f"""╔══════════════════════════╗
{trend_emoji} <b>{trend_type.upper()} NOW</b> {trend_emoji}
╚══════════════════════════╝

{content_emoji} <b>{content.title}</b>

⭐ <b>Rating:</b> {content.rating or 'N/A'}/10
🎭 <b>Genres:</b> {genres_text}

━━━━━━━━━━━━━━━━━━━━━━━━

📖 <b>What is it about:</b>
{(content.overview[:200] + '...') if content.overview else 'Discover more on CineBrain!'}

━━━━━━━━━━━━━━━━━━━━━━━━

🔗 <b>Check it out on CineBrain</b>
👉 <a href="https://cinebrain.vercel.app/details/{content.slug}">View Details</a>

#Trending #CineBrain #{trend_type}"""
        
        return message

class TelegramService:
    """Service for sending content recommendations to Telegram channel"""
    
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        """Send admin recommendation to public channel with beautiful template"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram recommendation skipped - channel not configured")
                return False
            
            # Choose template based on content type
            if content.content_type == 'anime':
                message = TelegramTemplates.anime_recommendation_template(
                    content, admin_name, description
                )
            elif content.content_type in ['tv', 'series']:
                message = TelegramTemplates.tv_show_recommendation_template(
                    content, admin_name, description
                )
            else:  # movie
                message = TelegramTemplates.movie_recommendation_template(
                    content, admin_name, description
                )
            
            # Get poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create inline keyboard for interaction
            keyboard = types.InlineKeyboardMarkup()
            
            # Add buttons
            view_button = types.InlineKeyboardButton(
                text="🎬 View on CineBrain",
                url=f"https://cinebrain.vercel.app/details/{content.slug}"
            )
            keyboard.add(view_button)
            
            if content.youtube_trailer_id:
                trailer_button = types.InlineKeyboardButton(
                    text="🎥 Watch Trailer",
                    url=f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                )
                keyboard.add(trailer_button)
            
            # Add more info button
            more_button = types.InlineKeyboardButton(
                text="ℹ️ More Info",
                url=f"https://cinebrain.vercel.app/details/{content.slug}"
            )
            keyboard.add(more_button)
            
            # Send with photo if available
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                    logger.info(f"✅ Admin recommendation sent to Telegram with photo: {content.title}")
                except Exception as photo_error:
                    logger.error(f"Failed to send photo: {photo_error}")
                    # Try sending without photo
                    bot.send_message(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        text=message,
                        parse_mode='HTML',
                        reply_markup=keyboard,
                        disable_web_page_preview=False
                    )
                    logger.info(f"✅ Admin recommendation sent to Telegram (text only): {content.title}")
            else:
                bot.send_message(
                    chat_id=TELEGRAM_CHANNEL_ID,
                    text=message,
                    parse_mode='HTML',
                    reply_markup=keyboard,
                    disable_web_page_preview=False
                )
                logger.info(f"✅ Admin recommendation sent to Telegram: {content.title}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False
    
    @staticmethod
    def send_trending_alert(content, trend_type="rising"):
        """Send trending content alert"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.trending_alert_template(content, trend_type)
            
            # Get poster
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create keyboard
            keyboard = types.InlineKeyboardMarkup()
            view_button = types.InlineKeyboardButton(
                text="🔥 Check it out",
                url=f"https://cinebrain.vercel.app/details/{content.slug}"
            )
            keyboard.add(view_button)
            
            # Send
            if poster_url:
                bot.send_photo(
                    chat_id=TELEGRAM_CHANNEL_ID,
                    photo=poster_url,
                    caption=message,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
            else:
                bot.send_message(
                    chat_id=TELEGRAM_CHANNEL_ID,
                    text=message,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
            
            logger.info(f"✅ Trending alert sent: {content.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trending alert error: {e}")
            return False
    
    @staticmethod
    def send_weekly_digest(recommendations_list, week_number):
        """Send weekly digest of recommendations"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.weekly_digest_template(recommendations_list, week_number)
            
            keyboard = types.InlineKeyboardMarkup()
            explore_button = types.InlineKeyboardButton(
                text="🌟 Explore All Picks",
                url="https://cinebrain.vercel.app"
            )
            keyboard.add(explore_button)
            
            bot.send_message(
                chat_id=TELEGRAM_CHANNEL_ID,
                text=message,
                parse_mode='HTML',
                reply_markup=keyboard
            )
            
            logger.info(f"✅ Weekly digest sent for week {week_number}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Weekly digest error: {e}")
            return False
    
    @staticmethod
    def send_new_content_batch_alert(content_list, batch_type="movies"):
        """Send alert when multiple content items are added"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            emoji_map = {
                'movies': '🎬',
                'shows': '📺',
                'anime': '🎌'
            }
            
            emoji = emoji_map.get(batch_type, '🎬')
            count = len(content_list)
            
            message = f"""╔══════════════════════════╗
{emoji} <b>NEW CONTENT ADDED</b> {emoji}
╚══════════════════════════╝

<b>{count} New {batch_type.title()} just landed on CineBrain!</b>

━━━━━━━━━━━━━━━━━━━━━━━━

<b>Latest Additions:</b>

"""
            
            for i, content in enumerate(content_list[:5], 1):
                message += f"{i}. <b>{content.title}</b>\n"
                message += f"   ⭐ {content.rating or 'N/A'}/10\n\n"
            
            if count > 5:
                message += f"<i>...and {count - 5} more!</i>\n\n"
            
            message += f"""━━━━━━━━━━━━━━━━━━━━━━━━

🍿 <b>Start Watching Now</b>
👉 <a href="https://cinebrain.vercel.app">Browse All Content</a>

#NewContent #CineBrain #{batch_type}"""
            
            keyboard = types.InlineKeyboardMarkup()
            browse_button = types.InlineKeyboardButton(
                text=f"Browse {batch_type.title()}",
                url="https://cinebrain.vercel.app"
            )
            keyboard.add(browse_button)
            
            bot.send_message(
                chat_id=TELEGRAM_CHANNEL_ID,
                text=message,
                parse_mode='HTML',
                reply_markup=keyboard
            )
            
            logger.info(f"✅ Batch content alert sent: {count} {batch_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch alert error: {e}")
            return False

class TelegramAdminService:
    """Service for admin-related Telegram notifications"""
    
    @staticmethod
    def send_content_notification(content_title, admin_name, action_type="added"):
        """Send notification about content management to admin chat"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                logger.warning("Telegram admin notification skipped - admin chat not configured")
                return False
            
            action_emoji = {
                'added': '➕',
                'updated': '✏️',
                'deleted': '🗑',
                'recommended': '⭐'
            }
            
            emoji = action_emoji.get(action_type, '📝')
            
            message = f"""
{emoji} <b>Content {action_type.title()}</b>

<b>Title:</b> {content_title}
<b>Admin:</b> {admin_name}
<b>Action:</b> {action_type.upper()}
<b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

#ContentManagement #CineBrain
            """
            
            bot.send_message(
                chat_id=TELEGRAM_ADMIN_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            
            logger.info(f"✅ Admin notification sent: {action_type} - {content_title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Admin notification error: {e}")
            return False
    
    @staticmethod
    def send_recommendation_stats(stats_data):
        """Send recommendation statistics to admin"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                return False
            
            message = f"""
📊 <b>Recommendation Stats</b>

<b>Total Recommendations:</b> {stats_data.get('total', 0)}
<b>This Week:</b> {stats_data.get('this_week', 0)}
<b>Most Active Admin:</b> {stats_data.get('top_admin', 'N/A')}
<b>Most Recommended Genre:</b> {stats_data.get('top_genre', 'N/A')}

<b>Engagement:</b>
• Views: {stats_data.get('views', 0):,}
• Clicks: {stats_data.get('clicks', 0):,}
• CTR: {stats_data.get('ctr', 0):.2f}%

<b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

#Stats #CineBrain
            """
            
            bot.send_message(
                chat_id=TELEGRAM_ADMIN_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            
            logger.info("✅ Recommendation stats sent to admin")
            return True
            
        except Exception as e:
            logger.error(f"❌ Stats notification error: {e}")
            return False

class TelegramScheduler:
    """Background scheduler for Telegram notifications"""
    
    def __init__(self, app=None):
        self.app = app
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
                    
                    # Weekly digest - Every Monday at 9 AM UTC
                    if now.weekday() == 0 and now.hour == 9 and now.minute == 0:
                        try:
                            if self.app:
                                with self.app.app_context():
                                    self._send_weekly_digest()
                        except Exception as e:
                            logger.error(f"Weekly digest error: {e}")
                    
                    # Sleep for 1 hour
                    time.sleep(3600)
                    
                except Exception as e:
                    logger.error(f"Telegram scheduler error: {e}")
                    time.sleep(300)
        
        thread = threading.Thread(target=scheduler_worker, daemon=True, name="TelegramScheduler")
        thread.start()
        logger.info("✅ Telegram scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("🛑 Telegram scheduler stopped")
    
    def _send_weekly_digest(self):
        """Send weekly digest of recommendations"""
        try:
            # Get week number
            week_number = datetime.utcnow().isocalendar()[1]
            
            # This would fetch recommendations from database
            # For now, placeholder
            logger.info(f"Would send weekly digest for week {week_number}")
            
        except Exception as e:
            logger.error(f"Weekly digest error: {e}")

# Global scheduler instance
telegram_scheduler = None

def init_telegram_service(app, db, models, services):
    """Initialize Telegram service"""
    global telegram_scheduler
    
    try:
        # Initialize scheduler
        telegram_scheduler = TelegramScheduler(app)
        
        # Start scheduler if bot is configured
        if bot:
            telegram_scheduler.start_scheduler()
            logger.info("✅ Telegram service initialized successfully")
            logger.info("   - Content recommendations: ✓ Active")
            logger.info("   - Trending alerts: ✓ Active")
            logger.info("   - Admin notifications: ✓ Active")
            logger.info("   - Weekly digest: ✓ Scheduled")
        else:
            logger.warning("⚠️ Telegram service initialized but bot not configured")
        
        return {
            'telegram_service': TelegramService,
            'telegram_admin_service': TelegramAdminService,
            'telegram_scheduler': telegram_scheduler,
            'telegram_templates': TelegramTemplates
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Telegram service: {e}")
        return None

def cleanup_telegram_service():
    """Cleanup Telegram service"""
    global telegram_scheduler
    if telegram_scheduler:
        telegram_scheduler.stop_scheduler()
        logger.info("✅ Telegram service cleaned up")
