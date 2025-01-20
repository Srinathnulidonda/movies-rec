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
    """Ultra beautiful and compact templates"""
    
    @staticmethod
    def get_rating_display(rating):
        """Get beautiful rating display with stars"""
        if not rating:
            return "⭐ N/A"
        
        full_stars = int(rating)
        half_star = (rating - full_stars) >= 0.5
        
        stars = "⭐" * full_stars
        if half_star and full_stars < 10:
            stars += "✨"
        
        if rating >= 9.0:
            return f"🏆 {stars} <b>{rating}/10</b>"
        elif rating >= 8.0:
            return f"💎 {stars} <b>{rating}/10</b>"
        elif rating >= 7.0:
            return f"⭐ {stars} <b>{rating}/10</b>"
        else:
            return f"✨ {stars} <b>{rating}/10</b>"
    
    @staticmethod
    def get_content_type_badge(content_type):
        """Get beautiful badge for content type"""
        badges = {
            'movie': '🎬 𝗠𝗢𝗩𝗜𝗘',
            'tv': '📺 𝗦𝗘𝗥𝗜𝗘𝗦',
            'anime': '🎌 𝗔𝗡𝗜𝗠𝗘',
            'series': '🎭 𝗦𝗛𝗢𝗪',
        }
        return badges.get(content_type.lower(), '🎬 𝗠𝗢𝗩𝗜𝗘')
    
    @staticmethod
    def format_runtime(runtime):
        """Format runtime beautifully"""
        if not runtime:
            return "⏱ N/A"
        hours = runtime // 60
        minutes = runtime % 60
        if hours > 0:
            return f"⏱ <b>{hours}h {minutes}m</b>"
        return f"⏱ <b>{minutes}m</b>"
    
    @staticmethod
    def format_genres_compact(genres_list):
        """Format genres compactly with emojis"""
        genre_emojis = {
            'Action': '💥', 'Adventure': '🗺', 'Animation': '🎨', 'Comedy': '😂',
            'Crime': '🔫', 'Documentary': '📹', 'Drama': '🎭', 'Fantasy': '🧙',
            'Horror': '👻', 'Mystery': '🔍', 'Romance': '💕', 'Sci-Fi': '🚀',
            'Science Fiction': '🚀', 'Thriller': '😱', 'War': '⚔️', 'Western': '🤠',
            'Music': '🎵', 'Family': '👨‍👩‍👧‍👦', 'History': '📜'
        }
        
        if not genres_list:
            return "🎬 General"
        
        formatted = []
        for genre in genres_list[:3]:
            emoji = genre_emojis.get(genre, '🎬')
            formatted.append(f"{emoji} {genre}")
        
        return " • ".join(formatted)
    
    @staticmethod
    def format_hashtags(genres_list):
        """Format hashtags"""
        if not genres_list:
            return ""
        
        hashtags = []
        for genre in genres_list[:3]:
            clean = genre.replace(' ', '').replace('-', '')
            hashtags.append(f"#{clean}")
        
        return " ".join(hashtags)
    
    @staticmethod
    def get_cinebrain_url(slug):
        """Generate CineBrain URL"""
        return f"https://cinebrain.vercel.app/explore/details.html?{slug}"
    
    @staticmethod
    def movie_recommendation_template(content, admin_name, description, genres_list=None):
        """Compact and attractive movie template"""
        
        # Parse genres
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        # Format year
        release_year = ""
        if content.release_date:
            try:
                release_year = content.release_date.strftime('%Y')
            except:
                release_year = str(content.release_date)[:4] if content.release_date else ""
        
        # Get components
        rating_display = TelegramTemplates.get_rating_display(content.rating)
        content_badge = TelegramTemplates.get_content_type_badge(content.content_type)
        genres_compact = TelegramTemplates.format_genres_compact(genres_list)
        runtime_display = TelegramTemplates.format_runtime(content.runtime)
        hashtags = TelegramTemplates.format_hashtags(genres_list)
        detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        # Shorten text
        short_description = description[:180] + "..." if len(description) > 180 else description
        short_overview = (content.overview[:220] + '...') if content.overview and len(content.overview) > 220 else (content.overview or '✨ Discover the full story on CineBrain!')
        
        message = f"""╔═══════════════════════════════╗
║  🌟 <b>ADMIN'S PICK</b> 🌟  ║
╚═══════════════════════════════╝

<b>🎬 {content.title.upper()}</b>"""
        
        if release_year:
            message += f" <i>({release_year})</i>"
        
        message += f"""

{rating_display}
"""
        
        if content.vote_count:
            message += f"👥 {content.vote_count:,} votes • "
        
        message += f"""{runtime_display}

🎭 {genres_compact}

━━━━━━━━━━━━━━━━━━━━━━━━

💬 <b>ADMIN'S NOTE:</b>
<i>"{short_description}"</i>

━━━━━━━━━━━━━━━━━━━━━━━━

<b>📖 SYNOPSIS</b>

<i>{short_overview}</i>

━━━━━━━━━━━━━━━━━━━━━━━━

🎯 <a href="{detail_url}"><b>WATCH ON CINEBRAIN</b></a>

{hashtags} #CineBrain #MustWatch

{content_badge}"""
        
        return message
    
    @staticmethod
    def anime_recommendation_template(content, admin_name, description, genres_list=None, anime_genres_list=None):
        """Compact and attractive anime template"""
        
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
        
        all_genres = genres_list + (anime_genres_list if anime_genres_list else [])
        
        # Format year
        release_year = ""
        if content.release_date:
            try:
                release_year = content.release_date.strftime('%Y')
            except:
                release_year = str(content.release_date)[:4] if content.release_date else ""
        
        # Get components
        rating_display = TelegramTemplates.get_rating_display(content.rating)
        genres_compact = TelegramTemplates.format_genres_compact(all_genres)
        hashtags = TelegramTemplates.format_hashtags(all_genres)
        detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        # Shorten text
        short_description = description[:180] + "..." if len(description) > 180 else description
        short_overview = (content.overview[:220] + '...') if content.overview and len(content.overview) > 220 else (content.overview or '✨ Epic anime adventure awaits!')
        
        message = f"""╔═══════════════════════════════╗
║   🎌 <b>ANIME PICK</b> 🎌   ║
╚═══════════════════════════════╝

<b>⚡ {content.title.upper()}</b>"""
        
        if content.original_title and content.original_title != content.title:
            message += f"\n<i>({content.original_title})</i>"
        
        if release_year:
            message += f" • <b>{release_year}</b>"
        
        message += f"""

{rating_display}

🎭 {genres_compact}

━━━━━━━━━━━━━━━━━━━━━━━━

💬 <b>ADMIN'S NOTE:</b>
<i>"{short_description}"</i>

━━━━━━━━━━━━━━━━━━━━━━━━

<b>📖 SYNOPSIS</b>

<i>{short_overview}</i>

━━━━━━━━━━━━━━━━━━━━━━━━

🎯 <a href="{detail_url}"><b>WATCH ON CINEBRAIN</b></a>

{hashtags} #Anime #CineBrain

🎌 𝗔𝗡𝗜𝗠𝗘"""
        
        return message
    
    @staticmethod
    def tv_show_recommendation_template(content, admin_name, description, genres_list=None):
        """Compact and attractive TV show template"""
        
        # Parse genres
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        # Format year
        release_year = ""
        if content.release_date:
            try:
                release_year = content.release_date.strftime('%Y')
            except:
                release_year = str(content.release_date)[:4] if content.release_date else ""
        
        # Get components
        rating_display = TelegramTemplates.get_rating_display(content.rating)
        genres_compact = TelegramTemplates.format_genres_compact(genres_list)
        hashtags = TelegramTemplates.format_hashtags(genres_list)
        detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        # Shorten text
        short_description = description[:180] + "..." if len(description) > 180 else description
        short_overview = (content.overview[:220] + '...') if content.overview and len(content.overview) > 220 else (content.overview or '✨ Addictive series awaits!')
        
        message = f"""╔═══════════════════════════════╗
║  📺 <b>SERIES PICK</b> 📺  ║
╚═══════════════════════════════╝

<b>🎭 {content.title.upper()}</b>"""
        
        if release_year:
            message += f" <i>({release_year})</i>"
        
        message += f"""

{rating_display}
"""
        
        if content.vote_count:
            message += f"👥 {content.vote_count:,} viewers\n"
        
        message += f"""
🎭 {genres_compact}

━━━━━━━━━━━━━━━━━━━━━━━━

💬 <b>ADMIN'S NOTE:</b>
<i>"{short_description}"</i>

━━━━━━━━━━━━━━━━━━━━━━━━

<b>📖 SYNOPSIS</b>

<i>{short_overview}</i>

━━━━━━━━━━━━━━━━━━━━━━━━

🎯 <a href="{detail_url}"><b>STREAM ON CINEBRAIN</b></a>

{hashtags} #TVSeries #CineBrain

📺 𝗦𝗘𝗥𝗜𝗘𝗦"""
        
        return message
    
    @staticmethod
    def weekly_digest_template(recommendations_list, week_number):
        """Compact weekly digest"""
        
        message = f"""╔═══════════════════════════════╗
║   🌟 <b>WEEK {week_number}</b> 🌟   ║
╚═══════════════════════════════╝

<b>🎬 TOP PICKS THIS WEEK</b>

━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        for i, rec in enumerate(recommendations_list[:5], 1):
            emoji_map = {'movie': '🎬', 'tv': '📺', 'anime': '🎌'}
            emoji = emoji_map.get(rec['content_type'], '🎬')
            rating = rec.get('rating', 0)
            stars = "⭐" * int(rating) if rating else "✨"
            
            message += f"<b>{i}.</b> {emoji} <b>{rec['title']}</b>\n"
            message += f"{stars} <b>{rating or 'N/A'}/10</b>\n\n"
        
        message += f"""━━━━━━━━━━━━━━━━━━━━━━━━

🎯 <a href="https://cinebrain.vercel.app"><b>EXPLORE ALL</b></a>

#WeeklyPicks #CineBrain"""
        
        return message
    
    @staticmethod
    def trending_alert_template(content, trend_type="rising"):
        """Compact trending alert"""
        
        emoji_map = {
            'rising': '📈', 'hot': '🔥', 'viral': '💥',
            'popular': '🌟', 'trending': '⚡'
        }
        
        trend_emoji = emoji_map.get(trend_type, '📈')
        
        # Parse genres
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                pass
        
        rating_display = TelegramTemplates.get_rating_display(content.rating)
        genres_compact = TelegramTemplates.format_genres_compact(genres_list)
        hashtags = TelegramTemplates.format_hashtags(genres_list)
        detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        short_overview = (content.overview[:200] + '...') if content.overview and len(content.overview) > 200 else (content.overview or '🔥 See why everyone is watching!')
        
        message = f"""╔═══════════════════════════════╗
║  {trend_emoji} <b>{trend_type.upper()}</b> {trend_emoji}  ║
╚═══════════════════════════════╝

<b>🔥 {content.title.upper()}</b>

{rating_display}

🎭 {genres_compact}

━━━━━━━━━━━━━━━━━━━━━━━━

<b>📖 SYNOPSIS</b>

<i>{short_overview}</i>

━━━━━━━━━━━━━━━━━━━━━━━━

🎯 <a href="{detail_url}"><b>WATCH NOW</b></a>

{hashtags} #Trending #CineBrain"""
        
        return message
    
    @staticmethod
    def new_content_alert_template(content):
        """Compact new content alert"""
        
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                pass
        
        rating_display = TelegramTemplates.get_rating_display(content.rating)
        genres_compact = TelegramTemplates.format_genres_compact(genres_list)
        content_badge = TelegramTemplates.get_content_type_badge(content.content_type)
        detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        message = f"""╔═══════════════════════════════╗
║    ✨ <b>NEW!</b> ✨    ║
╚═══════════════════════════════╝

<b>🎉 {content.title.upper()}</b>

{rating_display}

🎭 {genres_compact}

━━━━━━━━━━━━━━━━━━━━━━━━

🎯 <a href="{detail_url}"><b>DISCOVER NOW</b></a>

#NewRelease #CineBrain

{content_badge}"""
        
        return message
    
    @staticmethod
    def batch_content_template(content_list, batch_type="movies"):
        """Compact batch alert"""
        
        emoji_map = {'movies': '🎬', 'shows': '📺', 'anime': '🎌'}
        emoji = emoji_map.get(batch_type, '🎬')
        count = len(content_list)
        
        message = f"""╔═══════════════════════════════╗
║   {emoji} <b>{count} NEW!</b> {emoji}   ║
╚═══════════════════════════════╝

<b>🎉 FRESH {batch_type.upper()}</b>

━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        for i, content in enumerate(content_list[:5], 1):
            rating = content.rating or 0
            stars = "⭐" * int(rating) if rating else "✨"
            message += f"<b>{i}.</b> {content.title}\n{stars} {rating or 'N/A'}/10\n\n"
        
        if count > 5:
            message += f"<i>+{count - 5} more...</i>\n\n"
        
        message += f"""━━━━━━━━━━━━━━━━━━━━━━━━

🎯 <a href="https://cinebrain.vercel.app"><b>BROWSE ALL</b></a>

#NewContent #CineBrain"""
        
        return message

class TelegramService:
    """Service for sending ultra-attractive Telegram messages"""
    
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        """Send stunning admin recommendation"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram recommendation skipped - channel not configured")
                return False
            
            # Choose template
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
            
            # Telegram caption limit
            MAX_CAPTION_LENGTH = 1024
            
            # Get poster
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create keyboard
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            
            detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
            
            watch_btn = types.InlineKeyboardButton(
                text="🎬 Watch Now",
                url=detail_url
            )
            
            details_btn = types.InlineKeyboardButton(
                text="📖 Details",
                url=detail_url
            )
            
            keyboard.add(watch_btn, details_btn)
            
            if content.youtube_trailer_id:
                trailer_btn = types.InlineKeyboardButton(
                    text="🎥 Trailer",
                    url=f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                )
                keyboard.add(trailer_btn)
            
            explore_btn = types.InlineKeyboardButton(
                text="🌟 Explore CineBrain",
                url="https://cinebrain.vercel.app"
            )
            keyboard.add(explore_btn)
            
            # Send based on length
            if poster_url and len(message) <= MAX_CAPTION_LENGTH:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                    logger.info(f"✅ Recommendation sent with photo: {content.title}")
                except Exception as e:
                    logger.error(f"Photo send failed: {e}")
                    # Send separately
                    try:
                        bot.send_photo(chat_id=TELEGRAM_CHANNEL_ID, photo=poster_url)
                        bot.send_message(
                            chat_id=TELEGRAM_CHANNEL_ID,
                            text=message,
                            parse_mode='HTML',
                            reply_markup=keyboard
                        )
                        logger.info(f"✅ Sent as separate messages: {content.title}")
                    except:
                        bot.send_message(
                            chat_id=TELEGRAM_CHANNEL_ID,
                            text=message,
                            parse_mode='HTML',
                            reply_markup=keyboard
                        )
            else:
                # Too long - send separately
                if poster_url:
                    try:
                        bot.send_photo(chat_id=TELEGRAM_CHANNEL_ID, photo=poster_url)
                    except:
                        pass
                
                bot.send_message(
                    chat_id=TELEGRAM_CHANNEL_ID,
                    text=message,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
                logger.info(f"✅ Sent as text: {content.title}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False
    
    @staticmethod
    def send_trending_alert(content, trend_type="rising"):
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
            watch_btn = types.InlineKeyboardButton(text="🔥 Watch Now!", url=detail_url)
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
            discover_btn = types.InlineKeyboardButton(text="✨ Discover", url=detail_url)
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
                text="🌟 View All",
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
    def send_new_content_batch_alert(content_list, batch_type="movies"):
        """Send batch content alert"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.batch_content_template(content_list, batch_type)
            
            keyboard = types.InlineKeyboardMarkup()
            browse_btn = types.InlineKeyboardButton(
                text=f"🎬 Browse {batch_type.title()}",
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
    """Admin notifications"""
    
    @staticmethod
    def send_content_notification(content_title, admin_name, action_type="added"):
        """Send admin notification"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                return False
            
            action_emoji = {
                'added': '➕', 'updated': '✏️',
                'deleted': '🗑️', 'recommended': '⭐'
            }
            
            emoji = action_emoji.get(action_type, '📝')
            
            message = f"""{emoji} <b>{action_type.upper()}</b>

<b>Title:</b> {content_title}
<b>Admin:</b> {admin_name}
<b>Time:</b> {datetime.utcnow().strftime('%H:%M UTC')}

#CineBrain"""
            
            bot.send_message(
                chat_id=TELEGRAM_ADMIN_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            
            logger.info(f"✅ Admin notification: {action_type} - {content_title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Admin notification error: {e}")
            return False

class TelegramScheduler:
    """Background scheduler"""
    
    def __init__(self, app=None):
        self.app = app
        self.running = False
    
    def start_scheduler(self):
        """Start scheduler"""
        if self.running:
            return
        
        self.running = True
        
        def scheduler_worker():
            while self.running:
                try:
                    now = datetime.utcnow()
                    
                    if now.weekday() == 0 and now.hour == 9 and now.minute == 0:
                        if self.app:
                            with self.app.app_context():
                                self._send_weekly_digest()
                    
                    time.sleep(3600)
                    
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    time.sleep(300)
        
        thread = threading.Thread(target=scheduler_worker, daemon=True)
        thread.start()
        logger.info("✅ Telegram scheduler started")
    
    def stop_scheduler(self):
        """Stop scheduler"""
        self.running = False
    
    def _send_weekly_digest(self):
        """Send weekly digest"""
        try:
            week_number = datetime.utcnow().isocalendar()[1]
            logger.info(f"Weekly digest for week {week_number}")
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
        else:
            logger.warning("⚠️ Telegram not configured")
        
        return {
            'telegram_service': TelegramService,
            'telegram_admin_service': TelegramAdminService,
            'telegram_scheduler': telegram_scheduler,
            'telegram_templates': TelegramTemplates
        }
        
    except Exception as e:
        logger.error(f"❌ Init failed: {e}")
        return None

def cleanup_telegram_service():
    """Cleanup"""
    global telegram_scheduler
    if telegram_scheduler:
        telegram_scheduler.stop_scheduler()