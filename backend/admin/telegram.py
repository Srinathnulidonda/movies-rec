# admin/telegram.py

import os
import json
import logging
import threading
import time
import telebot
import random
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
CINEBRAIN_FOOTER_NEW = '<i>🧠 <b>CineBrain</b> — Hidden Gems • Mind-Bending Sci-Fi • Rare Anime</i>'
CTA_FOLLOW = "🔎 <i>More hidden gems daily — <b>@cinebrain</b></i>"

# Hook Pool for Viral Templates
HOOK_POOL = [
    "🔥 THIS MOVIE WILL MELT YOUR BRAIN",
    "⚠️ WARNING: This hidden gem will BREAK your reality",
    "💥 99% of viewers MISSED this masterpiece",
    "🧠 Only for smart viewers — proceed if you think you can handle it",
    "🚨 INSANE hidden gem nobody talks about",
    "🔥 This sci-fi twist will RIP the rules of reality",
    "⚡ If you like puzzles & mind-benders — stop scrolling",
    "🎯 This one rewrites everything you know about storytelling",
    "🔪 A psychological cut so sharp it leaves a scar",
    "🌌 This film will change how you see 'choice' and 'fate'"
]

# Template Types
class TemplateType:
    CLASSIC = "classic"
    MIND_BENDING = "mind_bending"
    HIDDEN_GEM = "hidden_gem"
    ANIME_GEM = "anime_gem"
    SCENE_CLIP = "scene_clip"

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
    
    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def pick_random_hook():
        return random.choice(HOOK_POOL)
    
    @staticmethod
    def safe_escape(text: str) -> str:
        if text is None:
            return ""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
    
    @staticmethod
    def get_content_type_label(content_type: str) -> str:
        """Get proper label for content type"""
        if content_type == 'anime':
            return "Anime"
        elif content_type in ['tv', 'series']:
            return "TV Show/Series"
        else:
            return "Movie"
    
    @staticmethod
    def get_rating_display(rating: Optional[float]) -> str:
        """Format rating display"""
        if not rating:
            return "N/A"
        try:
            return f"{round(float(rating), 1)}/10"
        except:
            return "N/A"
    
    @staticmethod
    def format_runtime(runtime: Optional[int]) -> Optional[str]:
        """Format runtime into human-readable format"""
        if not runtime:
            return None
        try:
            hours = runtime // 60
            minutes = runtime % 60
            if hours > 0:
                return f"{hours}h {minutes}m"
            return f"{minutes}m"
        except:
            return None
    
    @staticmethod
    def format_genres(genres, limit: int = 3) -> str:
        """Format genres with bullet separator"""
        if not genres:
            return "Unknown"
        
        if isinstance(genres, list):
            return " • ".join(genres[:limit])
        
        if isinstance(genres, str):
            try:
                arr = json.loads(genres)
                if isinstance(arr, list):
                    return " • ".join(arr[:limit])
            except:
                parts = [p.strip() for p in genres.split(",") if p.strip()]
                return " • ".join(parts[:limit]) if parts else "Unknown"
        
        return "Unknown"
    
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
        text = text.strip()
        if len(text) <= limit:
            return TelegramTemplates.safe_escape(text)
        return TelegramTemplates.safe_escape(text[:limit].rsplit(' ', 1)[0] + "...")
    
    @staticmethod
    def get_cinebrain_url(slug: str) -> str:
        """
        Generate CineBrain detail page URL
        @deprecated Use cinebrain_tracking_url() for tracked URLs
        """
        return f"https://cinebrain.vercel.app/explore/details.html?{slug}"
    
    # ==========================================================
    # CLASSIC TEMPLATES (Original Templates)
    # ==========================================================
    @staticmethod
    def classic_movie_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        """
        Classic premium movie recommendation with minimalist design
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
    def classic_tv_show_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        """
        Classic premium TV series template with minimalist design
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
        
        message = f"""<b>🎞️ TV Show/Series: {content.title}{year}</b>
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
    def classic_anime_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None, anime_genres_list: Optional[List[str]] = None) -> str:
        """
        Classic premium anime template with minimalist design
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
    
    # ==========================================================
    # 🔥 NEW VIRAL TEMPLATES
    # ==========================================================
    @staticmethod
    def mind_bending_template(content: Any, description: str = None) -> str:
        """Mind-bending template for reality-breaking content"""
        hook = TelegramTemplates.pick_random_hook()
        title = TelegramTemplates.safe_escape(content.title)
        content_type = TelegramTemplates.get_content_type_label(content.content_type)
        year = TelegramTemplates.format_year(content.release_date)
        genres = TelegramTemplates.format_genres(content.genres)
        rating = TelegramTemplates.get_rating_display(content.rating)
        runtime = TelegramTemplates.format_runtime(getattr(content, 'runtime', None))
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, limit=220)
        
        if_you_like = TelegramTemplates.safe_escape(
            getattr(content, 'if_you_like', 'Inception, Dark, Tenet')
        )
        
        reasons = [
            "A concept that bends reality.",
            "A twist that rewrites the entire story.",
            "A low-budget masterpiece with maximum impact."
        ]
        
        runtime_block = f" | ⏱ {runtime}" if runtime else ""
        
        return f"""<b>{hook}</b>
<b>{content_type}: {title}{year}</b>
<i>{genres}{runtime_block} • ⭐ {rating}</i>
{DIVIDER}

<b>Why this will melt your brain</b>
<blockquote><i>{synopsis}</i></blockquote>
• {reasons[0]}
• {reasons[1]}
• {reasons[2]}

{DIVIDER}
<b>If you like:</b> {if_you_like}

{CTA_FOLLOW}
{CINEBRAIN_FOOTER_NEW}"""
    
    @staticmethod
    def hidden_gem_template(content: Any, description: str = None) -> str:
        """Short & punchy hidden gem template"""
        title = TelegramTemplates.safe_escape(content.title)
        content_type = TelegramTemplates.get_content_type_label(content.content_type)
        year = TelegramTemplates.format_year(content.release_date)
        genres = TelegramTemplates.format_genres(content.genres)
        rating = TelegramTemplates.get_rating_display(content.rating)
        
        short_hook = TelegramTemplates.safe_escape(
            description or getattr(content, 'short_hook', 'Criminally underrated.')
        )
        if_you_like = TelegramTemplates.safe_escape(getattr(content, 'if_you_like', ''))
        
        return f"""<b>💎 Hidden Gem {content_type} — {title}{year}</b>
<i>{genres} • ⭐ {rating}</i>

{short_hook}

{f"<b>If you like:</b> {if_you_like}" if if_you_like else ""}

{CTA_FOLLOW}
{CINEBRAIN_FOOTER_NEW}"""
    
    @staticmethod
    def anime_gem_template(content: Any, description: str = None) -> str:
        """Emotional & viral anime template"""
        title = TelegramTemplates.safe_escape(content.title)
        year = TelegramTemplates.format_year(content.release_date)
        
        # Merge genres & anime_genres
        all_genres = []
        if content.genres:
            try:
                if isinstance(content.genres, list):
                    all_genres.extend(content.genres)
                else:
                    all_genres.extend(json.loads(content.genres))
            except:
                pass
        
        if hasattr(content, 'anime_genres') and content.anime_genres:
            try:
                if isinstance(content.anime_genres, list):
                    all_genres.extend(content.anime_genres)
                else:
                    all_genres.extend(json.loads(content.anime_genres))
            except:
                pass
        
        genres = TelegramTemplates.format_genres(all_genres)
        status = TelegramTemplates.safe_escape(getattr(content, 'status', 'Completed'))
        rating = TelegramTemplates.get_rating_display(content.rating)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, limit=200)
        emotion_hook = TelegramTemplates.safe_escape(
            description or getattr(content, 'emotion_hook', 'Deep, haunting, unforgettable.')
        )
        
        hook = TelegramTemplates.pick_random_hook()
        
        return f"""<b>{hook}</b>
<b>🎐 Anime: {title}{year}</b>
<i>{genres} • {status} • ⭐ {rating}</i>
{DIVIDER}

<b>Why this hits hard</b>
<blockquote><i>{synopsis}</i></blockquote>
• {emotion_hook}

{DIVIDER}
{CTA_FOLLOW}
{CINEBRAIN_FOOTER_NEW}"""
    
    @staticmethod
    def scene_clip_template(content: Any, caption: str = None) -> str:
        """Template for video clips"""
        cap = TelegramTemplates.safe_escape(caption or "This 10-second scene will hook you")
        title = TelegramTemplates.safe_escape(content.title)
        content_type = TelegramTemplates.get_content_type_label(content.content_type)
        year = TelegramTemplates.format_year(content.release_date)
        genres = TelegramTemplates.format_genres(content.genres)
        
        return f"""<b>{cap}</b>
<b>🎥 {content_type}: {title}{year}</b>
<i>{genres}</i>
{DIVIDER}

🔎 Watch the clip below. If this hooks you, the full {content_type.lower()} will blow your mind.

{CTA_FOLLOW}
{CINEBRAIN_FOOTER_NEW}"""
    
    @staticmethod
    def get_template(template_type: str, content: Any, admin_name: str = None, description: str = None) -> str:
        """Get template by type"""
        # Classic templates for backward compatibility
        if template_type == TemplateType.CLASSIC:
            if content.content_type == 'anime':
                return TelegramTemplates.classic_anime_recommendation_template(content, admin_name, description)
            elif content.content_type in ['tv', 'series']:
                return TelegramTemplates.classic_tv_show_recommendation_template(content, admin_name, description)
            else:
                return TelegramTemplates.classic_movie_recommendation_template(content, admin_name, description)
        
        # New viral templates
        elif template_type == TemplateType.MIND_BENDING:
            return TelegramTemplates.mind_bending_template(content, description)
        elif template_type == TemplateType.HIDDEN_GEM:
            return TelegramTemplates.hidden_gem_template(content, description)
        elif template_type == TemplateType.ANIME_GEM:
            return TelegramTemplates.anime_gem_template(content, description)
        elif template_type == TemplateType.SCENE_CLIP:
            return TelegramTemplates.scene_clip_template(content, description)
        
        # Default to classic based on content type
        else:
            if content.content_type == 'anime':
                return TelegramTemplates.classic_anime_recommendation_template(content, admin_name, description)
            elif content.content_type in ['tv', 'series']:
                return TelegramTemplates.classic_tv_show_recommendation_template(content, admin_name, description)
            else:
                return TelegramTemplates.classic_movie_recommendation_template(content, admin_name, description)


class TelegramService:
    """
    Service for sending beautifully formatted Telegram notifications
    Handles all public channel communications
    """
    
    @staticmethod
    def send_admin_recommendation(content: Any, admin_name: str, description: str, template_type: str = TemplateType.CLASSIC) -> bool:
        """
        Send admin-curated recommendation with premium formatting
        
        @param content: Content object with movie/show details
        @param admin_name: Name of the admin making recommendation
        @param description: CineBrain Insight text
        @param template_type: Type of template to use (classic, mind_bending, hidden_gem, etc.)
        @return: Success status
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram recommendation skipped - channel not configured")
                return False
            
            # Get the appropriate template
            message = TelegramTemplates.get_template(template_type, content, admin_name, description)
            
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
            campaign_type = f"{content.content_type}_recommendation_{template_type}"
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
            
            # Send message with poster
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                    logger.info(f"✅ Premium recommendation with poster sent ({template_type}): {content.title}")
                except Exception as e:
                    logger.error(f"Photo send failed: {e}, sending text only")
                    bot.send_message(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        text=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                    logger.info(f"✅ Premium recommendation sent (text only, {template_type}): {content.title}")
            else:
                bot.send_message(
                    chat_id=TELEGRAM_CHANNEL_ID,
                    text=message,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
                logger.info(f"✅ Premium recommendation sent ({template_type}): {content.title}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
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
            logger.info("   ├─ Classic cinematic templates: ✓")
            logger.info("   ├─ Viral marketing templates: ✓")
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
            'template_types': TemplateType,
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
    'TemplateType',
    'cinebrain_tracking_url',   
    'init_telegram_service'
]