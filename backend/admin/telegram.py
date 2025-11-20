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

# NEW: Viral-style Constants
CINEBRAIN_FOOTER_VIRAL = '<i>🧠 <b>CineBrain</b> — Hidden Gems • Mind-Bending Sci-Fi • Rare Anime</i>'
CTA_FOLLOW = "🔎 <i>More hidden gems daily — <b>@cinebrain</b></i>"

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


def get_poster_url(content: Any) -> Optional[str]:
    """
    Get formatted poster URL for content
    
    @param content: Content object
    @return: Full poster URL or None
    """
    if not content or not hasattr(content, 'poster_path') or not content.poster_path:
        return None
    
    if content.poster_path.startswith('http'):
        return content.poster_path
    else:
        return f"https://image.tmdb.org/t/p/w500{content.poster_path}"


def get_content_type_label(content_type: str) -> str:
    """
    Get proper content type label for templates
    
    @param content_type: Content type string
    @return: Formatted label
    """
    content_type = content_type.lower() if content_type else 'movie'
    
    type_labels = {
        'movie': 'Movie Name',
        'tv': 'TV Show Name',
        'series': 'Series Name',
        'web_series': 'Web Series Name', 
        'tv_series': 'TV Series Name',
        'anime': 'Anime Name'
    }
    
    return type_labels.get(content_type, 'Movie Name')


class TelegramTemplates:
    """
    Premium cinematic templates for CineBrain's Telegram channel
    Now includes both Classic and Viral template styles with poster support
    """
    
    # ===========================================
    # SHARED UTILITY METHODS
    # ===========================================
    
    @staticmethod
    def pick_random_hook():
        """Get a random viral hook from the pool"""
        return random.choice(HOOK_POOL)
    
    @staticmethod
    def safe_escape(text: str) -> str:
        """Safely escape HTML characters"""
        if text is None:
            return ""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
    
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
            return ""
    
    @staticmethod
    def format_genres(val, limit: int = 3) -> str:
        """Format genres with bullet separator - handles multiple input types"""
        if not val:
            return "Unknown"

        if isinstance(val, list):
            return " • ".join(val[:limit])

        if isinstance(val, str):
            try:
                arr = json.loads(val)
                if isinstance(arr, list):
                    return " • ".join(arr[:limit])
            except:
                parts = [p.strip() for p in val.split(",") if p.strip()]
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
        t = text.strip()
        if len(t) <= limit:
            return TelegramTemplates.safe_escape(t)
        return TelegramTemplates.safe_escape(t[:limit].rsplit(' ', 1)[0] + "...")
    
    @staticmethod
    def get_cinebrain_url(slug: str) -> str:
        """
        Generate CineBrain detail page URL
        @deprecated Use cinebrain_tracking_url() for tracked URLs
        """
        return f"https://cinebrain.vercel.app/explore/details.html?{slug}"
    
    # ===========================================
    # CLASSIC TEMPLATES (Original Design)
    # ===========================================
    
    @staticmethod
    def movie_recommendation_classic(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        """
        Classic movie recommendation with minimalist design
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
        content_label = get_content_type_label(content.content_type)
        
        # Build runtime display
        runtime_str = f" | ⏱ {runtime}" if runtime else ""
        
        message = f"""<b>🎞️ {content_label}: {content.title}{year}</b>
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
    def tv_show_recommendation_classic(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        """
        Classic TV series template with minimalist design
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
        content_label = get_content_type_label(content.content_type)
        
        # For TV shows, show seasons if available
        runtime_str = ""
        if hasattr(content, 'seasons') and content.seasons:
            runtime_str = f" | ⏱ {content.seasons} Seasons"
        elif hasattr(content, 'number_of_seasons') and content.number_of_seasons:
            runtime_str = f" | ⏱ {content.number_of_seasons} Seasons"
        
        message = f"""<b>🎞️ {content_label}: {content.title}{year}</b>
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
    def anime_recommendation_classic(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None, anime_genres_list: Optional[List[str]] = None) -> str:
        """
        Classic anime template with minimalist design
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
        
        message = f"""<b>🎞️ Anime Name: {content.title}{year}</b>
<b>✨ Ratings:</b> {rating}{runtime_str}
<b>🎭 Genre:</b> {genres}
{DIVIDER}
💬 <b>Synopsis</b>
<blockquote><i>{synopsis}</i></blockquote>
{DIVIDER}
<i>🍿 Smart recommendations • Upcoming updates • Latest updates • New releases • Trending updates — visit <a href="https://cinebrain.vercel.app/">CineBrain</a></i>

{CINEBRAIN_FOOTER}"""
        
        return message
    
    # ===========================================
    # VIRAL TEMPLATES (New High-Engagement Style)
    # ===========================================
    
    @staticmethod
    def movie_recommendation_viral(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        """
        🔥 MIND-BENDING viral movie template for maximum engagement
        """
        hook = TelegramTemplates.pick_random_hook()
        title = TelegramTemplates.safe_escape(getattr(content, "title", "Unknown"))
        year = TelegramTemplates.format_year(getattr(content, "release_date", None))
        genres = TelegramTemplates.format_genres(getattr(content, "genres", None))
        rating = TelegramTemplates.get_rating_display(getattr(content, "rating", None))
        runtime = TelegramTemplates.format_runtime(getattr(content, "runtime", None))
        synopsis = TelegramTemplates.truncate_synopsis(getattr(content, "overview", ""), limit=220)
        content_label = get_content_type_label(getattr(content, "content_type", "movie"))

        if_you_like = TelegramTemplates.safe_escape(
            description or "Inception, Dark, Tenet"
        )

        reasons = [
            "A concept that bends reality.",
            "A twist that rewrites the entire movie.",
            "A low-budget masterpiece with maximum impact."
        ]

        runtime_block = f" | ⏱ {runtime}" if runtime else ""

        return f"""<b>{hook}</b>

<b>🎞️ {content_label}: {title}{year}</b>
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
{CINEBRAIN_FOOTER_VIRAL}"""
    
    @staticmethod
    def tv_show_recommendation_viral(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        """
        💎 HIDDEN GEM viral TV show template (short & punchy)
        """
        title = TelegramTemplates.safe_escape(getattr(content, "title", "Unknown"))
        year = TelegramTemplates.format_year(getattr(content, "release_date", None))
        genres = TelegramTemplates.format_genres(getattr(content, "genres", None))
        rating = TelegramTemplates.get_rating_display(getattr(content, "rating", None))
        content_label = get_content_type_label(getattr(content, "content_type", "tv"))

        short_hook = TelegramTemplates.safe_escape(
            description or "Criminally underrated series."
        )
        if_you_like = TelegramTemplates.safe_escape("Breaking Bad, Dark, Westworld")

        return f"""<b>💎 Hidden Gem Alert!</b>

<b>🎞️ {content_label}: {title}{year}</b>
<i>{genres} • ⭐ {rating}</i>

{short_hook}

<b>If you like:</b> {if_you_like}

{CTA_FOLLOW}
{CINEBRAIN_FOOTER_VIRAL}"""
    
    @staticmethod
    def anime_recommendation_viral(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None, anime_genres_list: Optional[List[str]] = None) -> str:
        """
        🎐 ANIME GEM viral template (emotional & viral)
        """
        title = TelegramTemplates.safe_escape(getattr(content, "title", "Unknown"))
        year = TelegramTemplates.format_year(getattr(content, "release_date", None))

        # merge genres & anime_genres
        raw = []
        g1 = getattr(content, "genres", None)
        g2 = getattr(content, "anime_genres", None)

        try:
            if isinstance(g1, list): 
                raw.extend(g1)
            elif isinstance(g1, str):
                raw.extend(json.loads(g1))
        except: 
            pass

        try:
            if isinstance(g2, list): 
                raw.extend(g2)
            elif isinstance(g2, str):
                raw.extend(json.loads(g2))
        except: 
            pass

        genres = " • ".join(raw[:3]) if raw else "Anime"

        status = TelegramTemplates.safe_escape(getattr(content, "status", "Completed"))
        rating = TelegramTemplates.get_rating_display(getattr(content, "rating", None))
        synopsis = TelegramTemplates.truncate_synopsis(getattr(content, "overview", ""), limit=200)
        emotion_hook = TelegramTemplates.safe_escape(
            description or "Deep, haunting, unforgettable."
        )

        hook = TelegramTemplates.pick_random_hook()

        return f"""<b>{hook}</b>

<b>🎐 Anime Name: {title}{year}</b>
<i>{genres} • {status} • ⭐ {rating}</i>
{DIVIDER}

<b>Why this hits hard</b>
<blockquote><i>{synopsis}</i></blockquote>
• {emotion_hook}

{DIVIDER}
{CTA_FOLLOW}
{CINEBRAIN_FOOTER_VIRAL}"""
    
    # ===========================================
    # SPECIAL VIRAL TEMPLATES
    # ===========================================
    
    @staticmethod
    def top_list_viral(title: str, items: List[Dict[str, Any]]) -> str:
        """
        📌 TOP LIST viral template (most viral format)
        """
        t = TelegramTemplates.safe_escape(title)
        lines = [f"<b>🧠 {t}</b>", ""]

        for idx, item in enumerate(items[:10], start=1):
            movie = TelegramTemplates.safe_escape(item.get("title", "Unknown"))
            year = f" ({item.get('year')})" if item.get("year") else ""
            hook = TelegramTemplates.safe_escape(item.get("hook", ""))
            content_label = get_content_type_label(item.get("content_type", "movie"))
            lines.append(f"{idx}. <b>{content_label}: {movie}</b>{year} — {hook}")

        lines.append("")
        lines.append(DIVIDER)
        lines.append(CTA_FOLLOW)
        lines.append(CINEBRAIN_FOOTER_VIRAL)

        return "\n".join(lines)
    
    @staticmethod
    def scene_clip_viral(content: Any, caption: Optional[str] = None) -> str:
        """
        🎥 SCENE CLIP viral template (for video clips)
        """
        cap = TelegramTemplates.safe_escape(caption or "This 10-second scene will hook you")
        title = TelegramTemplates.safe_escape(getattr(content, "title", "Unknown"))
        year = TelegramTemplates.format_year(getattr(content, "release_date", None))
        genres = TelegramTemplates.format_genres(
            getattr(content, "genres_list", None) or getattr(content, "genres", None)
        )
        content_label = get_content_type_label(getattr(content, "content_type", "movie"))

        return f"""<b>{cap}</b>

<b>🎥 {content_label}: {title}{year}</b>
<i>{genres}</i>
{DIVIDER}

🔎 Watch the clip below. If this hooks you, the full {content_label.lower()} will blow your mind.

{CTA_FOLLOW}
{CINEBRAIN_FOOTER_VIRAL}"""
    
    # ===========================================
    # TEMPLATE SELECTORS (For Easy Switching)
    # ===========================================
    
    @staticmethod
    def movie_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None, style: str = "classic") -> str:
        """
        Movie recommendation template with style selector
        @param style: "classic" or "viral"
        """
        if style == "viral":
            return TelegramTemplates.movie_recommendation_viral(content, admin_name, description, genres_list)
        return TelegramTemplates.movie_recommendation_classic(content, admin_name, description, genres_list)
    
    @staticmethod
    def tv_show_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None, style: str = "classic") -> str:
        """
        TV show recommendation template with style selector
        @param style: "classic" or "viral"
        """
        if style == "viral":
            return TelegramTemplates.tv_show_recommendation_viral(content, admin_name, description, genres_list)
        return TelegramTemplates.tv_show_recommendation_classic(content, admin_name, description, genres_list)
    
    @staticmethod
    def anime_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None, anime_genres_list: Optional[List[str]] = None, style: str = "classic") -> str:
        """
        Anime recommendation template with style selector
        @param style: "classic" or "viral"
        """
        if style == "viral":
            return TelegramTemplates.anime_recommendation_viral(content, admin_name, description, genres_list, anime_genres_list)
        return TelegramTemplates.anime_recommendation_classic(content, admin_name, description, genres_list, anime_genres_list)


class TelegramService:
    """
    Service for sending beautifully formatted Telegram notifications
    Handles all public channel communications with poster support
    """
    
    @staticmethod
    def send_admin_recommendation(content: Any, admin_name: str, description: str, style: str = "classic") -> bool:
        """
        Send admin-curated recommendation with premium formatting and poster
        
        @param content: Content object with movie/show details
        @param admin_name: Name of the admin making recommendation
        @param description: CineBrain Insight text
        @param style: Template style - "classic" or "viral"
        @return: Success status
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram recommendation skipped - channel not configured")
                return False
            
            # Select appropriate template based on content type and style
            if content.content_type == 'anime':
                message = TelegramTemplates.anime_recommendation_template(
                    content, admin_name, description, style=style
                )
            elif content.content_type in ['tv', 'series']:
                message = TelegramTemplates.tv_show_recommendation_template(
                    content, admin_name, description, style=style
                )
            else:
                message = TelegramTemplates.movie_recommendation_template(
                    content, admin_name, description, style=style
                )
            
            # Get poster URL - ALWAYS try to get poster
            poster_url = get_poster_url(content)
            
            # Create inline keyboard with two buttons
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            
            # Generate tracking URLs with appropriate campaign names
            campaign_type = f"{content.content_type}_recommendation_{style}"
            content_identifier = content.slug.replace('-', '_')
            
            detail_url = cinebrain_tracking_url(
                content.slug, 
                campaign_type, 
                content_identifier
            )
            
            # Two action buttons
            explore_btn = types.InlineKeyboardButton(
                text="🔍 Explore More",
                url=f"https://cinebrain.vercel.app/?utm_source=telegram&utm_medium=bot&utm_campaign={campaign_type}&utm_content=explore_more"
            )
            details_btn = types.InlineKeyboardButton(
                text="📖 Full Details",
                url=detail_url
            )
            keyboard.add(explore_btn, details_btn)
            
            # ALWAYS try to send with poster first, fallback to text if needed
            success = False
            
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                    logger.info(f"✅ Premium {style} recommendation with poster sent: {content.title}")
                    success = True
                except Exception as e:
                    logger.warning(f"Photo send failed: {e}, trying text only")
                    success = False
            
            # If poster failed or no poster URL, send as text message
            if not success:
                try:
                    bot.send_message(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        text=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                    logger.info(f"✅ Premium {style} recommendation sent (text only): {content.title}")
                    success = True
                except Exception as e:
                    logger.error(f"Text message also failed: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False
    
    @staticmethod
    def send_viral_top_list(title: str, items: List[Dict[str, Any]], poster_url: Optional[str] = None) -> bool:
        """
        Send viral top list to Telegram channel with optional poster
        
        @param title: List title
        @param items: List of items with title, year, hook, content_type
        @param poster_url: Optional poster URL for the list
        @return: Success status
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram top list skipped - channel not configured")
                return False
            
            message = TelegramTemplates.top_list_viral(title, items)
            
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            explore_btn = types.InlineKeyboardButton(
                text="🧠 Discover More Hidden Gems",
                url="https://cinebrain.vercel.app/?utm_source=telegram&utm_medium=bot&utm_campaign=top_list_viral&utm_content=discover_more"
            )
            keyboard.add(explore_btn)
            
            success = False
            
            # Try to send with poster if provided
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                    logger.info(f"✅ Viral top list with poster sent: {title}")
                    success = True
                except Exception as e:
                    logger.warning(f"Top list photo send failed: {e}, trying text only")
                    success = False
            
            # Fallback to text if poster failed or not provided
            if not success:
                bot.send_message(
                    chat_id=TELEGRAM_CHANNEL_ID,
                    text=message,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
                logger.info(f"✅ Viral top list sent (text only): {title}")
                success = True
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Viral top list error: {e}")
            return False
    
    @staticmethod
    def send_scene_clip(content: Any, video_url: str, caption: Optional[str] = None) -> bool:
        """
        Send scene clip with video and caption
        
        @param content: Content object
        @param video_url: URL of the video clip
        @param caption: Optional caption override
        @return: Success status
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram scene clip skipped - channel not configured")
                return False
            
            message = TelegramTemplates.scene_clip_viral(content, caption)
            
            # Create inline keyboard
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            
            detail_url = cinebrain_tracking_url(
                content.slug, 
                "scene_clip_viral", 
                content.slug.replace('-', '_')
            )
            
            watch_btn = types.InlineKeyboardButton(
                text="🎬 Watch Full Movie",
                url=detail_url
            )
            explore_btn = types.InlineKeyboardButton(
                text="🔍 More Clips",
                url="https://cinebrain.vercel.app/?utm_source=telegram&utm_medium=bot&utm_campaign=scene_clip&utm_content=more_clips"
            )
            keyboard.add(watch_btn, explore_btn)
            
            # Send video with caption
            try:
                bot.send_video(
                    chat_id=TELEGRAM_CHANNEL_ID,
                    video=video_url,
                    caption=message,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
                logger.info(f"✅ Scene clip sent: {content.title}")
                return True
            except Exception as e:
                logger.error(f"Scene clip send failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Scene clip error: {e}")
            return False


class TelegramAdminService:
    """
    Admin notification service for internal updates
    Handles admin-only communications
    """
    
    @staticmethod
    def send_content_notification(content_title: str, admin_name: str, action_type: str = "added", poster_url: Optional[str] = None) -> bool:
        """
        Send admin action notification to admin chat with optional poster
        
        @param content_title: Title of the content
        @param admin_name: Admin who performed action
        @param action_type: Type of action performed
        @param poster_url: Optional poster URL
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
            
            success = False
            
            # Try with poster first if provided
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_ADMIN_CHAT_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML'
                    )
                    success = True
                except Exception as e:
                    logger.warning(f"Admin notification photo failed: {e}, sending text")
                    success = False
            
            # Fallback to text
            if not success:
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
            logger.info("   ├─ Classic cinematic templates with posters: ✓")
            logger.info("   ├─ Viral engagement templates with posters: ✓")
            logger.info("   ├─ Content type labels (Movie Name, TV Show Name, Anime Name): ✓")
            logger.info("   ├─ Mobile-optimized layouts: ✓")
            logger.info("   ├─ Google Analytics tracking: ✓")
            logger.info("   ├─ Content recommendations: ✓")
            logger.info("   ├─ Top list viral formats: ✓")
            logger.info("   ├─ Scene clip support: ✓")
            logger.info("   └─ Admin notifications with posters: ✓")
        else:
            logger.warning("⚠️ Telegram bot not configured - service disabled")
            logger.warning("   Set TELEGRAM_BOT_TOKEN to enable Telegram features")
        
        return {
            'telegram_service': TelegramService,
            'telegram_admin_service': TelegramAdminService,
            'telegram_templates': TelegramTemplates,
            'cinebrain_tracking_url': cinebrain_tracking_url,
            'get_poster_url': get_poster_url
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
    'get_poster_url',
    'get_content_type_label',
    'init_telegram_service'
]