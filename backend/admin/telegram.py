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
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID')
TELEGRAM_ADMIN_CHAT_ID = os.environ.get('TELEGRAM_ADMIN_CHAT_ID')

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
    """Modern, compact templates for CineBrain"""
    
    @staticmethod
    def get_rating_display(rating):
        """Get clean rating display"""
        if not rating:
            return "N/A"
        return f"{rating}/10"
    
    @staticmethod
    def format_runtime(runtime):
        """Format runtime compactly"""
        if not runtime:
            return None
        hours = runtime // 60
        minutes = runtime % 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    
    @staticmethod
    def format_genres(genres_list, limit=3):
        """Format genres with bullet separator"""
        if not genres_list:
            return "Drama"
        return " • ".join(genres_list[:limit])
    
    @staticmethod
    def format_info_line(rating, runtime, genres_list):
        """Create aligned info line"""
        parts = []
        if rating:
            parts.append(f"⭐ <b>{rating}/10</b>")
        if runtime:
            runtime_str = TelegramTemplates.format_runtime(runtime)
            if runtime_str:
                parts.append(f"⏱ <b>{runtime_str}</b>")
        if genres_list:
            genres_str = TelegramTemplates.format_genres(genres_list)
            parts.append(f"🎭 <b>{genres_str}</b>")
        
        return " | ".join(parts) if parts else "🎭 <b>Drama</b>"
    
    @staticmethod
    def format_year(release_date):
        """Extract year from date"""
        if not release_date:
            return ""
        try:
            if hasattr(release_date, 'year'):
                return f"({release_date.year})"
            return f"({str(release_date)[:4]})"
        except:
            return ""
    
    @staticmethod
    def truncate_overview(text, limit=180):
        """Truncate text elegantly"""
        if not text:
            return "A cinematic experience awaits on CineBrain."
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(' ', 1)[0] + "..."
    
    @staticmethod
    def get_hashtags(content_type, genres_list=None):
        """Generate relevant hashtags"""
        tags = ["#CineBrain"]
        
        if content_type == "movie":
            tags.append("#MovieRecommendation")
        elif content_type == "tv":
            tags.append("#TVSeries")
        elif content_type == "anime":
            tags.append("#AnimeRecommendation")
        
        if genres_list and len(genres_list) > 0:
            clean_genre = genres_list[0].replace(" ", "").replace("-", "")
            tags.append(f"#{clean_genre}")
        
        tags.append("#NowTrending")
        return " ".join(tags)
    
    @staticmethod
    def get_cinebrain_url(slug):
        """Generate CineBrain URL"""
        return f"https://cinebrain.vercel.app/explore/details.html?{slug}"
    
    @staticmethod
    def movie_recommendation_template(content, admin_name, description, genres_list=None):
        """Compact movie recommendation template"""
        
        # Parse data
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        year = TelegramTemplates.format_year(content.release_date)
        info_line = TelegramTemplates.format_info_line(
            content.rating, content.runtime, genres_list
        )
        overview = TelegramTemplates.truncate_overview(content.overview)
        hashtags = TelegramTemplates.get_hashtags("movie", genres_list)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        message = f"""🎬 <b>{content.title.upper()}</b> <i>{year}</i>

{info_line}

💡 <b>CineBrain Insight:</b>
<i>{description}</i>

━━━━━━━━━━━━━━━━━━━

<i>{overview}</i>

👉 <a href="{url}"><b>Watch on CineBrain</b></a>

{hashtags}

<i>Curated with 💙 by CineBrain</i>"""
        
        return message
    
    @staticmethod
    def tv_show_recommendation_template(content, admin_name, description, genres_list=None):
        """Compact TV series template"""
        
        # Parse data
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        year = TelegramTemplates.format_year(content.release_date)
        # For TV shows, we might not have runtime
        info_parts = []
        if content.rating:
            info_parts.append(f"⭐ <b>{content.rating}/10</b>")
        info_parts.append(f"📺 <b>TV Series</b>")
        if genres_list:
            info_parts.append(f"🎭 <b>{TelegramTemplates.format_genres(genres_list)}</b>")
        
        info_line = " | ".join(info_parts)
        overview = TelegramTemplates.truncate_overview(content.overview)
        hashtags = TelegramTemplates.get_hashtags("tv", genres_list)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        message = f"""📺 <b>{content.title.upper()}</b> <i>{year}</i>

{info_line}

🌟 <b>Editor's Note:</b>
<i>{description}</i>

━━━━━━━━━━━━━━━━━━━

<i>{overview}</i>

👉 <a href="{url}"><b>Binge on CineBrain</b></a>

{hashtags} #BingeWorthy

<i>Curated with 💙 by CineBrain</i>"""
        
        return message
    
    @staticmethod
    def anime_recommendation_template(content, admin_name, description, genres_list=None, anime_genres_list=None):
        """Compact anime template"""
        
        # Parse genres
        all_genres = []
        if genres_list:
            all_genres.extend(genres_list)
        elif content.genres:
            try:
                all_genres.extend(json.loads(content.genres))
            except:
                pass
        
        if anime_genres_list:
            all_genres.extend(anime_genres_list)
        elif content.anime_genres:
            try:
                all_genres.extend(json.loads(content.anime_genres))
            except:
                pass
        
        year = TelegramTemplates.format_year(content.release_date)
        info_parts = []
        if content.rating:
            info_parts.append(f"⭐ <b>{content.rating}/10</b>")
        info_parts.append(f"🎌 <b>Anime</b>")
        if all_genres:
            info_parts.append(f"🎭 <b>{TelegramTemplates.format_genres(all_genres)}</b>")
        
        info_line = " | ".join(info_parts)
        overview = TelegramTemplates.truncate_overview(content.overview)
        hashtags = TelegramTemplates.get_hashtags("anime", all_genres)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        message = f"""🎌 <b>{content.title.upper()}</b> <i>{year}</i>

{info_line}

🎯 <b>Spotlight Highlight:</b>
<i>{description}</i>

━━━━━━━━━━━━━━━━━━━

<i>{overview}</i>

👉 <a href="{url}"><b>Stream on CineBrain</b></a>

{hashtags}

<i>Curated with 💙 by CineBrain</i>"""
        
        return message
    
    @staticmethod
    def new_content_alert_template(content):
        """Compact new arrival template"""
        
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                pass
        
        year = TelegramTemplates.format_year(content.release_date)
        info_line = TelegramTemplates.format_info_line(
            content.rating, content.runtime, genres_list
        )
        overview = TelegramTemplates.truncate_overview(content.overview, 150)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        content_emoji = "🎬"
        if content.content_type == "tv":
            content_emoji = "📺"
        elif content.content_type == "anime":
            content_emoji = "🎌"
        
        message = f"""🆕 <b>{content.title.upper()}</b> <i>just added to CineBrain!</i>

{info_line}

━━━━━━━━━━━━━━━━━━━

<i>{overview}</i>

👉 <a href="{url}"><b>View Full Details</b></a>

#CineBrain #NewArrival #NowTrending #MustWatch

<i>Curated with 💙 by CineBrain</i>"""
        
        return message
    
    @staticmethod
    def trending_alert_template(content, trend_type="trending"):
        """Compact trending alert"""
        
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                pass
        
        trend_emoji = {
            'rising': '📈',
            'hot': '🔥',
            'viral': '💥',
            'popular': '🌟',
            'trending': '⚡'
        }.get(trend_type, '⚡')
        
        year = TelegramTemplates.format_year(content.release_date)
        info_line = TelegramTemplates.format_info_line(
            content.rating, content.runtime, genres_list
        )
        overview = TelegramTemplates.truncate_overview(content.overview, 150)
        hashtags = TelegramTemplates.get_hashtags(content.content_type, genres_list)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        message = f"""{trend_emoji} <b>{content.title.upper()}</b> <i>is {trend_type} now!</i>

{info_line}

━━━━━━━━━━━━━━━━━━━

<i>{overview}</i>

👉 <a href="{url}"><b>Join the hype on CineBrain</b></a>

{hashtags} #{trend_type.title()}

<i>Curated with 💙 by CineBrain</i>"""
        
        return message
    
    @staticmethod
    def weekly_digest_template(recommendations_list, week_number):
        """Compact weekly digest"""
        
        message = f"""🌟 <b>CINEBRAIN WEEKLY PICKS</b> <i>Week {week_number}</i>

━━━━━━━━━━━━━━━━━━━

"""
        
        for i, rec in enumerate(recommendations_list[:5], 1):
            emoji = {'movie': '🎬', 'tv': '📺', 'anime': '🎌'}.get(rec['content_type'], '🎬')
            rating = rec.get('rating', 'N/A')
            
            message += f"""<b>{i}.</b> {emoji} <b>{rec['title']}</b>
   ⭐ {rating}/10 | <i>{rec['description'][:60]}...</i>

"""
        
        message += f"""━━━━━━━━━━━━━━━━━━━

🎯 <a href="https://cinebrain.vercel.app"><b>Explore all picks on CineBrain</b></a>

#CineBrain #WeeklyPicks #PersonalizedRecommendations

<i>Curated with 💙 by CineBrain</i>"""
        
        return message
    
    @staticmethod
    def batch_content_template(content_list, batch_type="content"):
        """Compact batch alert"""
        
        emoji = {'movies': '🎬', 'shows': '📺', 'anime': '🎌'}.get(batch_type, '🎬')
        count = len(content_list)
        
        message = f"""{emoji} <b>{count} NEW {batch_type.upper()} ADDED!</b>

━━━━━━━━━━━━━━━━━━━

"""
        
        for i, content in enumerate(content_list[:5], 1):
            rating = content.rating or 'N/A'
            message += f"<b>{i}.</b> {emoji} <b>{content.title}</b> <i>(⭐ {rating}/10)</i>\n"
        
        if count > 5:
            message += f"\n<i>...and {count - 5} more!</i>\n"
        
        message += f"""
━━━━━━━━━━━━━━━━━━━

🎯 <a href="https://cinebrain.vercel.app"><b>Browse all on CineBrain</b></a>

#CineBrain #NewContent #{batch_type.title()}

<i>Curated with 💙 by CineBrain</i>"""
        
        return message

class TelegramService:
    """Service for sending Telegram notifications"""
    
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        """Send admin recommendation"""
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
            else:
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
            
            # Create inline keyboard
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            
            detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
            
            # Primary buttons
            watch_btn = types.InlineKeyboardButton(
                text="🎬 Watch Now",
                url=detail_url
            )
            details_btn = types.InlineKeyboardButton(
                text="📖 Full Details",
                url=detail_url
            )
            keyboard.add(watch_btn, details_btn)
            
            # Trailer button if available
            if content.youtube_trailer_id:
                trailer_btn = types.InlineKeyboardButton(
                    text="🎥 Watch Trailer",
                    url=f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                )
                keyboard.add(trailer_btn)
            
            # CineBrain button
            explore_btn = types.InlineKeyboardButton(
                text="🌟 Explore CineBrain",
                url="https://cinebrain.vercel.app"
            )
            keyboard.add(explore_btn)
            
            # Send message
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                except Exception as e:
                    logger.error(f"Photo send failed: {e}, sending text only")
                    bot.send_message(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        text=message,
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
            
            logger.info(f"✅ Recommendation sent: {content.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False
    
    @staticmethod
    def send_trending_alert(content, trend_type="trending"):
        """Send trending alert"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.trending_alert_template(content, trend_type)
            
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            keyboard = types.InlineKeyboardMarkup()
            detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
            watch_btn = types.InlineKeyboardButton(
                text="🔥 Watch Now",
                url=detail_url
            )
            keyboard.add(watch_btn)
            
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                except:
                    bot.send_message(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        text=message,
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
    def send_new_content_alert(content):
        """Send new content alert"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.new_content_alert_template(content)
            
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            keyboard = types.InlineKeyboardMarkup()
            detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
            discover_btn = types.InlineKeyboardButton(
                text="✨ Discover Now",
                url=detail_url
            )
            keyboard.add(discover_btn)
            
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                except:
                    bot.send_message(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        text=message,
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
            
            logger.info(f"✅ New content alert sent: {content.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ New content alert error: {e}")
            return False
    
    @staticmethod
    def send_weekly_digest(recommendations_list, week_number):
        """Send weekly digest"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.weekly_digest_template(recommendations_list, week_number)
            
            keyboard = types.InlineKeyboardMarkup()
            explore_btn = types.InlineKeyboardButton(
                text="🌟 View All Picks",
                url="https://cinebrain.vercel.app"
            )
            keyboard.add(explore_btn)
            
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
    def send_new_content_batch_alert(content_list, batch_type="content"):
        """Send batch content alert"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.batch_content_template(content_list, batch_type)
            
            keyboard = types.InlineKeyboardMarkup()
            browse_btn = types.InlineKeyboardButton(
                text=f"🎬 Browse All {batch_type.title()}",
                url="https://cinebrain.vercel.app"
            )
            keyboard.add(browse_btn)
            
            bot.send_message(
                chat_id=TELEGRAM_CHANNEL_ID,
                text=message,
                parse_mode='HTML',
                reply_markup=keyboard
            )
            
            logger.info(f"✅ Batch alert sent: {len(content_list)} {batch_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch alert error: {e}")
            return False

class TelegramAdminService:
    """Admin notification service"""
    
    @staticmethod
    def send_content_notification(content_title, admin_name, action_type="added"):
        """Send admin notification"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                return False
            
            action_emoji = {
                'added': '➕',
                'updated': '✏️',
                'deleted': '🗑️',
                'recommended': '⭐'
            }
            
            emoji = action_emoji.get(action_type, '📝')
            
            message = f"""
{emoji} <b>CONTENT {action_type.upper()}</b>

<b>📌 Title:</b> {content_title}
<b>👤 Admin:</b> {admin_name}
<b>⚡ Action:</b> {action_type}
<b>🕐 Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

#AdminAction #CineBrain
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
        """Send recommendation statistics"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                return False
            
            message = f"""
📊 <b>RECOMMENDATION STATS</b>

<b>📈 Overview:</b>
• Total Recommendations: <b>{stats_data.get('total', 0)}</b>
• This Week: <b>{stats_data.get('this_week', 0)}</b>
• Top Admin: <b>{stats_data.get('top_admin', 'N/A')}</b>
• Top Genre: <b>{stats_data.get('top_genre', 'N/A')}</b>

<b>🎯 Engagement:</b>
• Views: <b>{stats_data.get('views', 0):,}</b>
• Clicks: <b>{stats_data.get('clicks', 0):,}</b>
• CTR: <b>{stats_data.get('ctr', 0):.2f}%</b>

<b>🕐 Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

#Stats #CineBrain
"""
            
            bot.send_message(
                chat_id=TELEGRAM_ADMIN_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            
            logger.info("✅ Stats sent to admin")
            return True
            
        except Exception as e:
            logger.error(f"❌ Stats notification error: {e}")
            return False

class TelegramScheduler:
    """Background scheduler for automated tasks"""
    
    def __init__(self, app=None):
        self.app = app
        self.running = False
    
    def start_scheduler(self):
        """Start the scheduler"""
        if self.running:
            return
        
        self.running = True
        
        def scheduler_worker():
            while self.running:
                try:
                    now = datetime.utcnow()
                    
                    # Weekly digest - Monday 9 AM UTC
                    if now.weekday() == 0 and now.hour == 9 and now.minute == 0:
                        if self.app:
                            with self.app.app_context():
                                self._send_weekly_digest()
                    
                    time.sleep(3600)  # Check every hour
                    
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        thread = threading.Thread(target=scheduler_worker, daemon=True)
        thread.start()
        logger.info("✅ Telegram scheduler started")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("🛑 Scheduler stopped")
    
    def _send_weekly_digest(self):
        """Send weekly digest (placeholder)"""
        try:
            week_number = datetime.utcnow().isocalendar()[1]
            logger.info(f"Would send weekly digest for week {week_number}")
            # Implementation would query recent recommendations and send digest
        except Exception as e:
            logger.error(f"Weekly digest error: {e}")

telegram_scheduler = None

def init_telegram_service(app, db, models, services):
    """Initialize Telegram service"""
    global telegram_scheduler
    
    try:
        telegram_scheduler = TelegramScheduler(app)
        
        if bot:
            telegram_scheduler.start_scheduler()
            logger.info("✅ Telegram service initialized")
            logger.info("   - Modern compact templates: ✓")
            logger.info("   - Content recommendations: ✓")
            logger.info("   - Trending alerts: ✓")
            logger.info("   - Admin notifications: ✓")
            logger.info("   - CineBrain branding: ✓")
        else:
            logger.warning("⚠️ Telegram bot not configured")
        
        return {
            'telegram_service': TelegramService,
            'telegram_admin_service': TelegramAdminService,
            'telegram_scheduler': telegram_scheduler,
            'telegram_templates': TelegramTemplates
        }
        
    except Exception as e:
        logger.error(f"❌ Telegram initialization failed: {e}")
        return None

def cleanup_telegram_service():
    """Cleanup Telegram service"""
    global telegram_scheduler
    if telegram_scheduler:
        telegram_scheduler.stop_scheduler()
        logger.info("✅ Telegram service cleaned up")