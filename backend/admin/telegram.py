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
from typing import Optional, List, Dict, Any

load_dotenv()

logger = logging.getLogger(__name__)

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID')
TELEGRAM_ADMIN_CHAT_ID = os.environ.get('TELEGRAM_ADMIN_CHAT_ID')

# Visual Constants
DIVIDER = "━━━━━━━━━━━━━━━━━━━"
CINEBRAIN_FOOTER = "<b><i>🎥 Recommended by CineBrain</i></b>"

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


def cinebrain_tracking_url(slug: str, campaign: str, content: Optional[str] = None) -> str:
    """
    Generate CineBrain URL with UTM tracking parameters for Google Analytics
    
    @param slug: Content slug for the detail page
    @param campaign: Campaign name (e.g., "movie_recommendation", "anime_recommendation")
    @param content: Optional content identifier for more detailed tracking
    @return: Full URL with tracking parameters
    """
    base = f"https://cinebrain.vercel.app/explore/details.html?{slug}"
    utm = {
        "utm_source": "telegram",
        "utm_medium": "bot",
        "utm_campaign": campaign,
    }
    if content:
        utm["utm_content"] = content

    params = "&".join([f"{k}={v}" for k, v in utm.items()])
    return f"{base}&{params}"


class TelegramTemplates:
    """
    Premium cinematic templates for CineBrain's Telegram channel
    Every message is a mini movie poster in text form
    """
    
    @staticmethod
    def get_rating_display(rating: Optional[float]) -> str:
        """Format rating display"""
        if not rating:
            return "N/A"
        return f"{rating}/10"
    
    @staticmethod
    def format_runtime(runtime: Optional[int]) -> Optional[str]:
        """Format runtime into human-readable format"""
        if not runtime:
            return None
        hours = runtime // 60
        minutes = runtime % 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    
    @staticmethod
    def format_genres(genres_list: Optional[List[str]], limit: int = 3) -> str:
        """Format genres with bullet separator"""
        if not genres_list:
            return "Drama"
        return " • ".join(genres_list[:limit])
    
    @staticmethod
    def format_year(release_date: Any) -> str:
        """Extract and format year from release date"""
        if not release_date:
            return ""
        try:
            if hasattr(release_date, 'year'):
                return f" ({release_date.year})"
            return f" ({str(release_date)[:4]})"
        except:
            return ""
    
    @staticmethod
    def truncate_synopsis(text: Optional[str], limit: int = 150) -> str:
        """Elegantly truncate synopsis at word boundary"""
        if not text:
            return "A cinematic experience awaits your discovery on CineBrain."
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(' ', 1)[0] + "..."
    
    @staticmethod
    def get_content_type_label(content: Any) -> str:
        """Get proper content type label"""
        if hasattr(content, 'content_type'):
            if content.content_type == 'anime':
                return "🎞️ Anime"
            elif content.content_type in ['tv', 'series']:
                return "🎞️ TV-Show/Web Series"
            else:
                return "🎞️ Movie"
        return "🎞️ Movie"
    
    @staticmethod
    def get_cinebrain_url(slug: str) -> str:
        """
        Generate CineBrain detail page URL
        @deprecated Use cinebrain_tracking_url() for tracked URLs
        """
        return f"https://cinebrain.vercel.app/explore/details.html?{slug}"
    
    @staticmethod
    def mind_bending_template(content: Any, if_you_like: Optional[str] = None, genres_list: Optional[List[str]] = None) -> str:
        """
        Mind-bending movie template with maximum impact
        Perfect for reality-warping, twist-heavy films
        """
        # Parse genres if needed
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        # Format components
        content_type = TelegramTemplates.get_content_type_label(content)
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        runtime = TelegramTemplates.format_runtime(content.runtime)
        genres = TelegramTemplates.format_genres(genres_list)
        overview = TelegramTemplates.truncate_synopsis(content.overview, 200)
        
        # Default if_you_like suggestions based on content type
        if not if_you_like:
            if content.content_type == 'anime':
                if_you_like = "Steins;Gate, Serial Experiments Lain, Perfect Blue"
            elif content.content_type in ['tv', 'series']:
                if_you_like = "Dark, Westworld, The OA"
            else:
                if_you_like = "Inception, Primer, Coherence"
        
        message = f"""🔥 THIS WILL MELT YOUR BRAIN
<b>{content_type}: {content.title}{year}</b>
<i>{genres} • ⭐ {rating} • ⏱ {runtime}</i>
━━━━━━━━━━━━━━━━━━━

<b>Why this will break your reality:</b>
<blockquote><i>{overview}</i></blockquote>
• A concept that bends reality  
• A twist that rewrites the whole story  
• A low-budget masterpiece with maximum impact  

━━━━━━━━━━━━━━━━━━━
<b>If you like:</b> {if_you_like}

🔎 <i>More hidden gems — @cinebrain</i>
<i>🧠 CineBrain — Hidden Gems • Mind-Bending Sci-Fi • Rare Anime</i>"""
        
        return message.strip()
    
    @staticmethod
    def hidden_gem_template(content: Any, hook: str, if_you_like: Optional[str] = None, genres_list: Optional[List[str]] = None) -> str:
        """
        Hidden gem template for lesser-known masterpieces
        """
        # Parse genres if needed
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        # Format components
        content_type = TelegramTemplates.get_content_type_label(content)
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        genres = TelegramTemplates.format_genres(genres_list)
        
        if_you_like_line = f"\n\n<b>If you like:</b> {if_you_like}" if if_you_like else ""
        
        message = f"""💎 Hidden Gem — <b>{content_type}: {content.title}{year}</b>
<i>{genres} • ⭐ {rating}</i>

{hook}
{if_you_like_line}

🔎 <i>More hidden gems — @cinebrain</i>
<i>🧠 CineBrain — Hidden Gems • Mind-Bending Sci-Fi • Rare Anime</i>"""
        
        return message.strip()
    
    @staticmethod
    def anime_gem_template(content: Any, emotion_hook: str, genres_list: Optional[List[str]] = None) -> str:
        """
        Anime gem template with emotional impact focus
        """
        # Combine all genres
        all_genres = []
        if genres_list:
            all_genres.extend(genres_list)
        elif content.genres:
            try:
                all_genres.extend(json.loads(content.genres))
            except:
                pass
        
        if hasattr(content, 'anime_genres') and content.anime_genres:
            try:
                all_genres.extend(json.loads(content.anime_genres))
            except:
                pass
        
        # Format components
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        genres = TelegramTemplates.format_genres(all_genres)
        overview = TelegramTemplates.truncate_synopsis(content.overview, 200)
        
        # Status
        status = "Ongoing"
        if hasattr(content, 'status') and content.status:
            status = content.status
        
        message = f"""🔥 THIS ANIME WILL BLOW YOUR MIND
<b>🎐 Anime: {content.title}{year}</b>
<i>{genres} • {status} • ⭐ {rating}</i>
━━━━━━━━━━━━━━━━━━━

<b>Why this hits hard:</b>
<blockquote><i>{overview}</i></blockquote>
• {emotion_hook}

━━━━━━━━━━━━━━━━━━━
🔎 <i>More rare anime — @cinebrain</i>
<i>🧠 CineBrain — Hidden Gems • Mind-Bending Sci-Fi • Rare Anime</i>"""
        
        return message.strip()
    
    @staticmethod
    def top_list_template(list_title: str, items: List[tuple], content_type: str = "Movie") -> str:
        """
        Top list template for curated collections
        
        @param list_title: Title of the list
        @param items: List of tuples (title, year, hook)
        @param content_type: Type of content in the list (Movie, Anime, TV-Show)
        """
        body = "\n".join(
            [f"{i+1}. <b>{title}</b> ({year}) — {hook}" 
             for i, (title, year, hook) in enumerate(items)]
        )
        
        message = f"""🧠 <b>{list_title}</b>
<i>{content_type} Collection</i>

{body}

━━━━━━━━━━━━━━━━━━━
🔎 <i>Save this list — More hidden gems @cinebrain</i>
<i>🧠 CineBrain — Hidden Gems • Mind-Bending Sci-Fi • Rare Anime</i>"""
        
        return message.strip()
    
    @staticmethod
    def movie_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        """
        Premium movie recommendation with minimalist design
        """
        
        # Parse genres if needed
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        # Format components
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        runtime = TelegramTemplates.format_runtime(content.runtime)
        genres = TelegramTemplates.format_genres(genres_list)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview)
        
        # Build runtime display
        runtime_str = f" | ⏱ {runtime}" if runtime else ""
        
        message = f"""<b>🎞️ Movie: {content.title}{year}</b>
<b>✨ Ratings:</b> {rating}{runtime_str}
<b>🎭 Genre:</b> {genres}
{DIVIDER}
💬 <b>Synopsis</b>
<blockquote><i>{synopsis}</i></blockquote>
{DIVIDER}
<i>🍿 Smart recommendations • Upcoming updates • Latest updates • New releases • Trending updates — visit <a href="https://cinebrain.vercel.app/">CineBrain</a></i>

{CINEBRAIN_FOOTER}"""
        
        return message
    
    @staticmethod
    def tv_show_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        """
        Premium TV series template with minimalist design
        """
        
        # Parse genres if needed
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        # Format components
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        genres = TelegramTemplates.format_genres(genres_list)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview)
        
        # For TV shows, show seasons if available
        runtime_str = ""
        if hasattr(content, 'seasons') and content.seasons:
            runtime_str = f" | ⏱ {content.seasons} Seasons"
        
        message = f"""<b>🎞️ TV-Show/Web Series: {content.title}{year}</b>
<b>✨ Ratings:</b> {rating}{runtime_str}
<b>🎭 Genre:</b> {genres}
{DIVIDER}
💬 <b>Synopsis</b>
<blockquote><i>{synopsis}</i></blockquote>
{DIVIDER}
<i>🍿 Smart recommendations • Upcoming updates • Latest updates • New releases • Trending updates — visit <a href="https://cinebrain.vercel.app/">CineBrain</a></i>

{CINEBRAIN_FOOTER}"""
        
        return message
    
    @staticmethod
    def anime_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None, anime_genres_list: Optional[List[str]] = None) -> str:
        """
        Premium anime template with minimalist design
        """
        
        # Combine all genres
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
        elif hasattr(content, 'anime_genres') and content.anime_genres:
            try:
                all_genres.extend(json.loads(content.anime_genres))
            except:
                pass
        
        # Format components
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        genres = TelegramTemplates.format_genres(all_genres)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview)
        
        # For anime, show status if available
        runtime_str = ""
        if hasattr(content, 'status') and content.status:
            runtime_str = f" | ⏱ {content.status}"
        else:
            runtime_str = " | ⏱ Ongoing"
        
        message = f"""<b>🎞️ Anime: {content.title}{year}</b>
<b>✨ Ratings:</b> {rating}{runtime_str}
<b>🎭 Genre:</b> {genres}
{DIVIDER}
💬 <b>Synopsis</b>
<blockquote><i>{synopsis}</i></blockquote>
{DIVIDER}
<i>🍿 Smart recommendations • Upcoming updates • Latest updates • New releases • Trending updates — visit <a href="https://cinebrain.vercel.app/">CineBrain</a></i>

{CINEBRAIN_FOOTER}"""
        
        return message


class TelegramService:
    """
    Service for sending beautifully formatted Telegram notifications
    Handles all public channel communications
    """
    
    @staticmethod
    def send_admin_recommendation(content: Any, admin_name: str, description: str) -> bool:
        """
        Send admin-curated recommendation with premium formatting
        
        @param content: Content object with movie/show details
        @param admin_name: Name of the admin making recommendation
        @param description: CineBrain Insight text
        @return: Success status
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram recommendation skipped - channel not configured")
                return False
            
            # Select appropriate template based on content type
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
            
            # Create inline keyboard with two buttons
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            
            # Generate tracking URLs with appropriate campaign names
            campaign_type = f"{content.content_type}_recommendation"
            content_identifier = content.slug.replace('-', '_')
            
            detail_url = cinebrain_tracking_url(
                content.slug, 
                campaign_type, 
                content_identifier
            )
            
            # Two action buttons
            explore_btn = types.InlineKeyboardButton(
                text="Explore More",
                url=f"https://cinebrain.vercel.app/?utm_source=telegram&utm_medium=bot&utm_campaign={campaign_type}&utm_content=explore_more"
            )
            details_btn = types.InlineKeyboardButton(
                text="Full Details",
                url=detail_url
            )
            keyboard.add(explore_btn, details_btn)
            
            # Send message with or without poster
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                    logger.info(f"✅ Premium recommendation with poster sent: {content.title}")
                except Exception as e:
                    logger.error(f"Photo send failed: {e}, sending text only")
                    bot.send_message(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        text=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                    logger.info(f"✅ Premium recommendation sent (text only): {content.title}")
            else:
                bot.send_message(
                    chat_id=TELEGRAM_CHANNEL_ID,
                    text=message,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
                logger.info(f"✅ Premium recommendation sent: {content.title}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False
    
    @staticmethod
    def send_mind_bending_recommendation(content: Any, if_you_like: Optional[str] = None) -> bool:
        """
        Send mind-bending movie recommendation
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.mind_bending_template(content, if_you_like)
            
            # Get poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create inline keyboard
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            detail_url = cinebrain_tracking_url(content.slug, "mind_bending", content.slug.replace('-', '_'))
            
            watch_btn = types.InlineKeyboardButton(
                text="🧠 Bend My Mind",
                url=detail_url
            )
            more_btn = types.InlineKeyboardButton(
                text="🔍 More Gems",
                url="https://cinebrain.vercel.app/?utm_source=telegram&utm_medium=bot&utm_campaign=mind_bending&utm_content=more_gems"
            )
            keyboard.add(watch_btn, more_btn)
            
            # Send with poster
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
            
            logger.info(f"✅ Mind-bending recommendation sent: {content.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Mind-bending send error: {e}")
            return False
    
    @staticmethod
    def send_hidden_gem(content: Any, hook: str, if_you_like: Optional[str] = None) -> bool:
        """
        Send hidden gem recommendation
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.hidden_gem_template(content, hook, if_you_like)
            
            # Get poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create inline keyboard
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            detail_url = cinebrain_tracking_url(content.slug, "hidden_gem", content.slug.replace('-', '_'))
            
            discover_btn = types.InlineKeyboardButton(
                text="💎 Discover This Gem",
                url=detail_url
            )
            keyboard.add(discover_btn)
            
            # Send with poster
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
            
            logger.info(f"✅ Hidden gem sent: {content.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Hidden gem send error: {e}")
            return False
    
    @staticmethod
    def send_anime_gem(content: Any, emotion_hook: str) -> bool:
        """
        Send anime gem recommendation
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.anime_gem_template(content, emotion_hook)
            
            # Get poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create inline keyboard
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            detail_url = cinebrain_tracking_url(content.slug, "anime_gem", content.slug.replace('-', '_'))
            
            watch_btn = types.InlineKeyboardButton(
                text="🎐 Watch Now",
                url=detail_url
            )
            more_anime_btn = types.InlineKeyboardButton(
                text="🔥 More Anime",
                url="https://cinebrain.vercel.app/?utm_source=telegram&utm_medium=bot&utm_campaign=anime_gem&utm_content=more_anime"
            )
            keyboard.add(watch_btn, more_anime_btn)
            
            # Send with poster
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
            
            logger.info(f"✅ Anime gem sent: {content.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Anime gem send error: {e}")
            return False
    
    @staticmethod
    def send_top_list(list_title: str, items: List[tuple], content_type: str = "Movie") -> bool:
        """
        Send curated top list
        
        @param list_title: Title of the list
        @param items: List of tuples (title, year, hook)
        @param content_type: Type of content in the list
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.top_list_template(list_title, items, content_type)
            
            # Create inline keyboard
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            explore_btn = types.InlineKeyboardButton(
                text="🧠 Explore Full List",
                url="https://cinebrain.vercel.app/?utm_source=telegram&utm_medium=bot&utm_campaign=top_list&utm_content=explore_list"
            )
            keyboard.add(explore_btn)
            
            bot.send_message(
                chat_id=TELEGRAM_CHANNEL_ID,
                text=message,
                parse_mode='HTML',
                reply_markup=keyboard
            )
            
            logger.info(f"✅ Top list sent: {list_title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Top list send error: {e}")
            return False


class TelegramAdminService:
    """
    Admin notification service for internal updates
    Handles admin-only communications
    """
    
    @staticmethod
    def send_content_notification(content_title: str, admin_name: str, action_type: str = "added") -> bool:
        """
        Send admin action notification to admin chat
        
        @param content_title: Title of the content
        @param admin_name: Admin who performed action
        @param action_type: Type of action performed
        @return: Success status
        """
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                return False
            
            action_emoji = {
                'added': '➕',
                'updated': '✏️',
                'deleted': '🗑️',
                'recommended': '⭐'
            }.get(action_type, '📝')
            
            message = f"""{action_emoji} <b>Admin Action</b>
{DIVIDER}

<b>Content:</b> {content_title}
<b>Admin:</b> {admin_name}
<b>Action:</b> {action_type.upper()}
<b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

{DIVIDER}
#AdminAction #CineBrain"""
            
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
    def send_recommendation_stats(stats_data: Dict[str, Any]) -> bool:
        """
        Send recommendation statistics to admin chat
        
        @param stats_data: Dictionary containing stats
        @return: Success status
        """
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                return False
            
            message = f"""📊 <b>CineBrain Analytics</b>
{DIVIDER}

<b>📈 Recommendation Overview</b>
• Total Recommendations: <b>{stats_data.get('total', 0):,}</b>
• This Week: <b>{stats_data.get('this_week', 0):,}</b>
• Top Admin: <b>{stats_data.get('top_admin', 'N/A')}</b>
• Top Genre: <b>{stats_data.get('top_genre', 'N/A')}</b>

<b>🎯 Engagement Metrics</b>
• Total Views: <b>{stats_data.get('views', 0):,}</b>
• Total Clicks: <b>{stats_data.get('clicks', 0):,}</b>
• Click-Through Rate: <b>{stats_data.get('ctr', 0):.2f}%</b>
• Avg. Engagement: <b>{stats_data.get('avg_engagement', 0):.1f}%</b>

{DIVIDER}
<i>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</i>

#Analytics #CineBrain"""
            
            bot.send_message(
                chat_id=TELEGRAM_ADMIN_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            
            logger.info("✅ Analytics stats sent to admin")
            return True
            
        except Exception as e:
            logger.error(f"❌ Stats notification error: {e}")
            return False


def init_telegram_service(app, db, models, services) -> Optional[Dict[str, Any]]:
    """
    Initialize Telegram service with all components
    
    @param app: Flask application instance
    @param db: Database instance
    @param models: Database models
    @param services: Service dependencies
    @return: Dictionary of initialized services or None
    """
    try:
        if bot:
            logger.info("✅ CineBrain Telegram service initialized successfully")
            logger.info("   ├─ Minimalist cinematic templates: ✓")
            logger.info("   ├─ Mobile-optimized layouts: ✓")
            logger.info("   ├─ Google Analytics tracking: ✓")
            logger.info("   ├─ Content recommendations: ✓")
            logger.info("   └─ Admin notifications: ✓")
        else:
            logger.warning("⚠️ Telegram bot not configured - service disabled")
            logger.warning("   Set TELEGRAM_BOT_TOKEN to enable Telegram features")
        
        return {
            'telegram_service': TelegramService,
            'telegram_admin_service': TelegramAdminService,
            'telegram_templates': TelegramTemplates,
            'cinebrain_tracking_url': cinebrain_tracking_url
        }
        
    except Exception as e:
        logger.error(f"❌ Telegram initialization failed: {e}")
        return None


# Export public API
__all__ = [
    'TelegramTemplates',
    'TelegramService',
    'TelegramAdminService',
    'cinebrain_tracking_url',
    'init_telegram_service'
]