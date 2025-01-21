# admin/telegram.py

"""
CineBrain Telegram Integration
Premium cinematic messaging for intelligent movie discovery
"""

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
DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━"  # Exactly 22 chars
PIPE_SEP = "  |  "  # 2 spaces around pipe
SIGNATURE = "╰┈➤ Personalized for you by <b>CineBrain</b> 💙"
TAGLINE = "🍿 <i>Smart recommendations • Upcoming releases • Latest updates</i>"

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
    """
    Premium cinematic templates for CineBrain's Telegram channel
    Every message is a mini movie poster in text form
    """
    
    @staticmethod
    def get_rating_display(rating: Optional[float]) -> str:
        """Format rating with trophy emoji for excellence"""
        if not rating:
            return "🏆 <b>N/A</b>"
        if rating >= 9.0:
            return f"🏆 <b>{rating}/10</b>"
        elif rating >= 8.0:
            return f"⭐ <b>{rating}/10</b>"
        else:
            return f"⭐ <b>{rating}/10</b>"
    
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
                return f"({release_date.year})"
            return f"({str(release_date)[:4]})"
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
    def get_cinebrain_url(slug: str) -> str:
        """Generate CineBrain detail page URL"""
        return f"https://cinebrain.vercel.app/explore/details.html?{slug}"
    
    @staticmethod
    def get_hashtags(content_type: str, is_trending: bool = False, is_new: bool = False) -> str:
        """Generate relevant hashtags for the content"""
        tags = ["#CineBrain", "#Curated"]
        
        if is_trending:
            tags.append("#Trending")
        elif is_new:
            tags.append("#JustAdded")
        else:
            tags.append("#NowWatching")
        
        return " ".join(tags)
    
    @staticmethod
    def format_info_line(rating: Optional[float], runtime: Optional[int], genres_list: Optional[List[str]]) -> str:
        """Create perfectly aligned info line with consistent spacing"""
        parts = []
        
        # Rating
        rating_str = TelegramTemplates.get_rating_display(rating)
        parts.append(rating_str)
        
        # Runtime
        if runtime:
            runtime_str = TelegramTemplates.format_runtime(runtime)
            if runtime_str:
                parts.append(f"⏱ <b>{runtime_str}</b>")
        
        # Genres
        if genres_list:
            genres_str = TelegramTemplates.format_genres(genres_list)
            parts.append(f"🎭 {genres_str}")
        
        return PIPE_SEP.join(parts)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STANDARD RECOMMENDATION TEMPLATE
    # The flagship template for all movie, TV, and anime recommendations
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def movie_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        """
        Premium movie recommendation with CineBrain Insight
        
        Example output:
        🎬 CineBrain Spotlight
        ━━━━━━━━━━━━━━━━━━━━━━
        
        🎥 DUNE: PART TWO (2024)
        
        🏆 9.1/10  |  ⏱ 2h 45m  |  🎭 Sci-Fi • Action • Drama
        
        ━━━━━━━━━━━━━━━━━━━━━━
        💡 CineBrain Insight
        "A breathtaking continuation of the Atreides legacy..."
        """
        
        # Parse genres if needed
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        # Format components
        year = TelegramTemplates.format_year(content.release_date)
        info_line = TelegramTemplates.format_info_line(content.rating, content.runtime, genres_list)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, 150)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        hashtags = TelegramTemplates.get_hashtags("movie")
        
        message = f"""🎬 <b>CineBrain Spotlight</b>
{DIVIDER}

<b>🎥 {content.title.upper()}</b> <i>{year}</i>

{info_line}

{DIVIDER}
💡 <b>CineBrain Insight</b>
<i>"{description}"</i>

{DIVIDER}
📖 <b>Synopsis</b>
<i>{synopsis}</i>

{DIVIDER}
🎯 <b>Experience on CineBrain</b>
👉 <a href="{url}"><b>EXPLORE FULL DETAILS</b></a>

{TAGLINE}
{hashtags}

{SIGNATURE}"""
        
        return message
    
    @staticmethod
    def tv_show_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        """Premium TV series template optimized for binge-watching appeal"""
        
        # Parse genres if needed
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        # Format components
        year = TelegramTemplates.format_year(content.release_date)
        info_line = TelegramTemplates.format_info_line(content.rating, None, genres_list)  # TV shows often lack runtime
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, 150)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        hashtags = TelegramTemplates.get_hashtags("tv")
        
        message = f"""🎬 <b>CineBrain Spotlight</b>
{DIVIDER}

<b>📺 {content.title.upper()}</b> <i>{year}</i>

{info_line}

{DIVIDER}
💡 <b>CineBrain Insight</b>
<i>"{description}"</i>

{DIVIDER}
📖 <b>Synopsis</b>
<i>{synopsis}</i>

{DIVIDER}
🎯 <b>Binge on CineBrain</b>
👉 <a href="{url}"><b>START WATCHING NOW</b></a>

{TAGLINE}
{hashtags}

{SIGNATURE}"""
        
        return message
    
    @staticmethod
    def anime_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None, anime_genres_list: Optional[List[str]] = None) -> str:
        """Premium anime template with cultural authenticity"""
        
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
        info_line = TelegramTemplates.format_info_line(content.rating, None, all_genres)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, 150)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        hashtags = TelegramTemplates.get_hashtags("anime")
        
        message = f"""🎬 <b>CineBrain Spotlight</b>
{DIVIDER}

<b>🎌 {content.title.upper()}</b> <i>{year}</i>

{info_line}

{DIVIDER}
💡 <b>CineBrain Insight</b>
<i>"{description}"</i>

{DIVIDER}
📖 <b>Synopsis</b>
<i>{synopsis}</i>

{DIVIDER}
🎯 <b>Stream on CineBrain</b>
👉 <a href="{url}"><b>WATCH WITH SUBTITLES</b></a>

{TAGLINE}
{hashtags}

{SIGNATURE}"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UPCOMING RELEASE TEMPLATE
    # Build anticipation for future releases
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def format_upcoming_release(content: Any, release_date: str, anticipation_reason: str) -> str:
        """
        Format upcoming release with anticipation building
        
        @param content: Content object with movie/show details
        @param release_date: Formatted release date string
        @param anticipation_reason: Why audiences are excited
        """
        
        # Parse genres
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                pass
        
        # Format components
        year = TelegramTemplates.format_year(content.release_date)
        info_line = TelegramTemplates.format_info_line(None, content.runtime, genres_list)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, 150)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        message = f"""🎯 <b>Coming Soon to CineBrain</b>
{DIVIDER}

<b>🎬 {content.title.upper()}</b> <i>{year}</i>

📅 <b>Release:</b> {release_date}{PIPE_SEP}🎭 {TelegramTemplates.format_genres(genres_list)}

{DIVIDER}
💡 <b>CineBrain Insight</b>
<i>"{anticipation_reason}"</i>

{DIVIDER}
📖 <b>Synopsis</b>
<i>{synopsis}</i>

{DIVIDER}
🎯 <b>Set Your Reminder</b>
👉 <a href="{url}"><b>GET NOTIFIED ON RELEASE</b></a>

🍿 <i>Be first to watch • Exclusive early access • Personalized alerts</i>
#CineBrain #ComingSoon #Anticipated

{SIGNATURE}"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRENDING ALERT TEMPLATE
    # Capture the viral moment
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def trending_alert_template(content: Any, trend_type: str = "trending", trending_reason: str = None) -> str:
        """
        Trending content with viral urgency
        
        @param content: Content object
        @param trend_type: Type of trend (viral, rising, hot, popular)
        @param trending_reason: Why it's trending (optional custom reason)
        """
        
        # Parse genres
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                pass
        
        # Format components
        trend_emoji = {
            'rising': '📈',
            'hot': '🔥',
            'viral': '💥',
            'popular': '🌟',
            'trending': '⚡'
        }.get(trend_type, '⚡')
        
        year = TelegramTemplates.format_year(content.release_date)
        info_line = TelegramTemplates.format_info_line(content.rating, content.runtime, genres_list)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, 150)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        hashtags = TelegramTemplates.get_hashtags(content.content_type, is_trending=True)
        
        # Default trending reason if not provided
        if not trending_reason:
            trending_reason = f"Everyone's talking about this {content.content_type} — join millions discovering why it's breaking the internet right now."
        
        message = f"""{trend_emoji} <b>Trending Now on CineBrain</b>
{DIVIDER}

<b>🎥 {content.title.upper()}</b> <i>{year}</i>

{info_line}

{DIVIDER}
💡 <b>CineBrain Insight</b>
<i>"{trending_reason}"</i>

{DIVIDER}
📖 <b>Synopsis</b>
<i>{synopsis}</i>

{DIVIDER}
🎯 <b>Join the Conversation</b>
👉 <a href="{url}"><b>WATCH NOW WHILE TRENDING</b></a>

🍿 <i>Trending globally • Social buzz • Must-watch moment</i>
{hashtags}

{SIGNATURE}"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEW ARRIVAL TEMPLATE
    # Fresh content just added
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def new_content_alert_template(content: Any, added_date: Optional[str] = None) -> str:
        """
        New arrival notification with freshness appeal
        
        @param content: Content object
        @param added_date: When it was added (defaults to "Today")
        """
        
        # Parse genres
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                pass
        
        # Format components
        year = TelegramTemplates.format_year(content.release_date)
        info_line = TelegramTemplates.format_info_line(content.rating, content.runtime, genres_list)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, 150)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        hashtags = TelegramTemplates.get_hashtags(content.content_type, is_new=True)
        
        # Content type emoji
        content_emoji = {
            'movie': '🎥',
            'tv': '📺',
            'anime': '🎌'
        }.get(content.content_type, '🎬')
        
        # Added date
        if not added_date:
            added_date = "Today"
        
        # Create insight for new content
        new_insight = f"Just landed on CineBrain and already making waves — be among the first to experience this {content.content_type} gem before everyone's talking about it."
        
        message = f"""✨ <b>Fresh on CineBrain</b>
{DIVIDER}

<b>{content_emoji} {content.title.upper()}</b> <i>{year}</i>

🆕 <b>Added:</b> {added_date}{PIPE_SEP}{info_line.split(PIPE_SEP, 1)[0] if PIPE_SEP in info_line else info_line}

{DIVIDER}
💡 <b>CineBrain Insight</b>
<i>"{new_insight}"</i>

{DIVIDER}
📖 <b>Synopsis</b>
<i>{synopsis}</i>

{DIVIDER}
🎯 <b>Watch It First</b>
👉 <a href="{url}"><b>STREAM NOW ON CINEBRAIN</b></a>

🍿 <i>Just added • First access • Latest releases</i>
{hashtags}

{SIGNATURE}"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WEEKLY DIGEST TEMPLATE
    # Curated weekly highlights
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def weekly_digest_template(recommendations_list: List[Dict], week_number: int) -> str:
        """
        Weekly digest with compact, scannable entries
        
        @param recommendations_list: List of recommendation dicts
        @param week_number: Week number of the year
        """
        
        message = f"""📬 <b>Your Weekly CineBrain Digest</b>
{DIVIDER}

<i>Week {week_number} • Hand-picked for your taste</i>

"""
        
        for i, rec in enumerate(recommendations_list[:5], 1):
            # Extract data safely
            title = rec.get('title', 'Unknown Title')
            rating = rec.get('rating', 'N/A')
            runtime = rec.get('runtime')
            year = rec.get('year', '')
            insight = rec.get('description', '')[:80]
            
            # Format rating
            if rating != 'N/A':
                rating_str = f"⭐ {rating}/10"
            else:
                rating_str = "⭐ N/A"
            
            # Format runtime
            runtime_str = ""
            if runtime:
                runtime_str = f"  |  ⏱ {runtime}"
            
            # Add entry
            message += f"""<b>{i}. {title}</b> {year}
{rating_str}{runtime_str}
<i>{insight}...</i>

"""
        
        message += f"""{DIVIDER}
🎯 <b>Explore Your Collection</b>
👉 <a href="https://cinebrain.vercel.app"><b>VIEW ALL RECOMMENDATIONS</b></a>

{TAGLINE}
#CineBrain #WeeklyDigest #Personalized

{SIGNATURE}"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BATCH RECOMMENDATIONS TEMPLATE
    # Themed collections
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def batch_content_template(content_list: List[Any], batch_type: str = "content", theme: Optional[str] = None) -> str:
        """
        Batch content notification for themed collections
        
        @param content_list: List of content objects
        @param batch_type: Type of batch (movies, shows, anime)
        @param theme: Optional theme (e.g., "Sci-Fi Masterpieces")
        """
        
        emoji = {'movies': '🎬', 'shows': '📺', 'anime': '🎌'}.get(batch_type, '🎬')
        count = len(content_list)
        
        # Header with theme
        if theme:
            header = f"{emoji} <b>Curated Collection: {theme}</b>"
        else:
            header = f"{emoji} <b>{count} New {batch_type.title()} Added</b>"
        
        message = f"""{header}
{DIVIDER}

<b>FEATURED SELECTIONS</b>

"""
        
        for i, content in enumerate(content_list[:5], 1):
            rating = content.rating if content.rating else 'N/A'
            if rating != 'N/A':
                rating_str = f"⭐ {rating}/10"
            else:
                rating_str = "⭐ N/A"
            
            year = TelegramTemplates.format_year(content.release_date)
            
            message += f"""<b>{i}.</b> {emoji} <b>{content.title}</b> {year}
    {rating_str}
"""
        
        if count > 5:
            message += f"""
<i>...plus {count - 5} more incredible {batch_type}!</i>
"""
        
        message += f"""
{DIVIDER}
🎯 <b>Browse Full Collection</b>
👉 <a href="https://cinebrain.vercel.app"><b>EXPLORE ALL {batch_type.upper()}</b></a>

🍿 <i>Curated collections • Themed recommendations • Discover more</i>
#CineBrain #Collection #{batch_type.title()}

{SIGNATURE}"""
        
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
            
            # Create premium inline keyboard
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            
            detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
            
            # Primary action buttons
            watch_btn = types.InlineKeyboardButton(
                text="🎬 Watch Now",
                url=detail_url
            )
            details_btn = types.InlineKeyboardButton(
                text="📖 Full Details",
                url=detail_url
            )
            keyboard.add(watch_btn, details_btn)
            
            # Add trailer button if available
            if hasattr(content, 'youtube_trailer_id') and content.youtube_trailer_id:
                trailer_btn = types.InlineKeyboardButton(
                    text="🎞️ Trailer",
                    url=f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                )
                share_btn = types.InlineKeyboardButton(
                    text="📤 Share",
                    url=f"https://t.me/share/url?url={detail_url}&text=Check%20out%20{content.title}%20on%20CineBrain!"
                )
                keyboard.add(trailer_btn, share_btn)
            
            # CineBrain exploration button
            explore_btn = types.InlineKeyboardButton(
                text="🔍 Explore More",
                url="https://cinebrain.vercel.app"
            )
            keyboard.add(explore_btn)
            
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
    def send_trending_alert(content: Any, trend_type: str = "trending", trending_reason: str = None) -> bool:
        """
        Send trending content alert with viral appeal
        
        @param content: Content object
        @param trend_type: Type of trend
        @param trending_reason: Custom reason for trending
        @return: Success status
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.trending_alert_template(content, trend_type, trending_reason)
            
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create keyboard with trending-specific buttons
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
            
            watch_btn = types.InlineKeyboardButton(
                text="🔥 Watch Now",
                url=detail_url
            )
            share_btn = types.InlineKeyboardButton(
                text="📤 Share Trend",
                url=f"https://t.me/share/url?url={detail_url}&text=🔥%20{content.title}%20is%20trending%20on%20CineBrain!"
            )
            keyboard.add(watch_btn, share_btn)
            
            explore_btn = types.InlineKeyboardButton(
                text="⚡ More Trending",
                url="https://cinebrain.vercel.app/trending"
            )
            keyboard.add(explore_btn)
            
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
    def send_new_content_alert(content: Any, added_date: Optional[str] = None) -> bool:
        """
        Send new content arrival notification
        
        @param content: Content object
        @param added_date: When content was added
        @return: Success status
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.new_content_alert_template(content, added_date)
            
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
            
            discover_btn = types.InlineKeyboardButton(
                text="✨ Discover Now",
                url=detail_url
            )
            browse_btn = types.InlineKeyboardButton(
                text="🆕 More New",
                url="https://cinebrain.vercel.app/new"
            )
            keyboard.add(discover_btn, browse_btn)
            
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
    def send_upcoming_release(content: Any, release_date: str, anticipation_reason: str) -> bool:
        """
        Send upcoming release notification
        
        @param content: Content object
        @param release_date: Formatted release date
        @param anticipation_reason: Why it's anticipated
        @return: Success status
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.format_upcoming_release(content, release_date, anticipation_reason)
            
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
            
            notify_btn = types.InlineKeyboardButton(
                text="🔔 Get Notified",
                url=detail_url
            )
            upcoming_btn = types.InlineKeyboardButton(
                text="📅 More Upcoming",
                url="https://cinebrain.vercel.app/upcoming"
            )
            keyboard.add(notify_btn, upcoming_btn)
            
            if hasattr(content, 'youtube_trailer_id') and content.youtube_trailer_id:
                trailer_btn = types.InlineKeyboardButton(
                    text="🎞️ Watch Trailer",
                    url=f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                )
                keyboard.add(trailer_btn)
            
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
            
            logger.info(f"✅ Upcoming release alert sent: {content.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Upcoming release error: {e}")
            return False
    
    @staticmethod
    def send_weekly_digest(recommendations_list: List[Dict], week_number: int) -> bool:
        """
        Send weekly curated digest
        
        @param recommendations_list: List of recommendations
        @param week_number: Week number of the year
        @return: Success status
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.weekly_digest_template(recommendations_list, week_number)
            
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            
            explore_btn = types.InlineKeyboardButton(
                text="🌟 View All Recommendations",
                url="https://cinebrain.vercel.app"
            )
            subscribe_btn = types.InlineKeyboardButton(
                text="🔔 Get Daily Updates",
                url=f"https://t.me/{TELEGRAM_CHANNEL_ID.replace('@', '')}"
            )
            keyboard.add(explore_btn, subscribe_btn)
            
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
    def send_new_content_batch_alert(content_list: List[Any], batch_type: str = "content", theme: Optional[str] = None) -> bool:
        """
        Send batch content notification for collections
        
        @param content_list: List of content objects
        @param batch_type: Type of batch
        @param theme: Optional theme name
        @return: Success status
        """
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.batch_content_template(content_list, batch_type, theme)
            
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            
            browse_btn = types.InlineKeyboardButton(
                text=f"🎬 Browse Full Collection",
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


class TelegramScheduler:
    """
    Background scheduler for automated Telegram tasks
    Handles weekly digests and scheduled notifications
    """
    
    def __init__(self, app=None):
        self.app = app
        self.running = False
    
    def start_scheduler(self):
        """Start the background scheduler thread"""
        if self.running:
            return
        
        self.running = True
        
        def scheduler_worker():
            """Worker thread for scheduled tasks"""
            while self.running:
                try:
                    now = datetime.utcnow()
                    
                    # Weekly digest - Monday 9 AM UTC
                    if now.weekday() == 0 and now.hour == 9 and now.minute == 0:
                        if self.app:
                            with self.app.app_context():
                                self._send_weekly_digest()
                    
                    # Daily trending check - Every day at 6 PM UTC
                    if now.hour == 18 and now.minute == 0:
                        if self.app:
                            with self.app.app_context():
                                self._check_trending_content()
                    
                    time.sleep(3600)  # Check every hour
                    
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        thread = threading.Thread(target=scheduler_worker, daemon=True)
        thread.start()
        logger.info("✅ Telegram scheduler started")
    
    def stop_scheduler(self):
        """Stop the background scheduler"""
        self.running = False
        logger.info("🛑 Scheduler stopped")
    
    def _send_weekly_digest(self):
        """Internal method to send weekly digest"""
        try:
            week_number = datetime.utcnow().isocalendar()[1]
            logger.info(f"Processing weekly digest for week {week_number}")
            
            # Implementation would query recent recommendations from database
            # This is a placeholder - actual implementation would fetch from DB
            # and call TelegramService.send_weekly_digest()
            
        except Exception as e:
            logger.error(f"Weekly digest processing error: {e}")
    
    def _check_trending_content(self):
        """Internal method to check and alert trending content"""
        try:
            logger.info("Checking for trending content")
            
            # Implementation would query trending content from database
            # This is a placeholder - actual implementation would fetch from DB
            # and call TelegramService.send_trending_alert()
            
        except Exception as e:
            logger.error(f"Trending check error: {e}")


# Module-level scheduler instance
telegram_scheduler = None


def init_telegram_service(app, db, models, services) -> Optional[Dict[str, Any]]:
    """
    Initialize Telegram service with all components
    
    @param app: Flask application instance
    @param db: Database instance
    @param models: Database models
    @param services: Service dependencies
    @return: Dictionary of initialized services or None
    """
    global telegram_scheduler
    
    try:
        telegram_scheduler = TelegramScheduler(app)
        
        if bot:
            telegram_scheduler.start_scheduler()
            logger.info("✅ CineBrain Telegram service initialized successfully")
            logger.info("   ├─ Premium cinematic templates: ✓")
            logger.info("   ├─ CineBrain Insight branding: ✓")
            logger.info("   ├─ Mobile-optimized layouts: ✓")
            logger.info("   ├─ Content recommendations: ✓")
            logger.info("   ├─ Trending & upcoming alerts: ✓")
            logger.info("   ├─ Admin notifications: ✓")
            logger.info("   └─ Automated scheduler: ✓")
        else:
            logger.warning("⚠️ Telegram bot not configured - service disabled")
            logger.warning("   Set TELEGRAM_BOT_TOKEN to enable Telegram features")
        
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
    """
    Cleanup Telegram service resources
    Called during application shutdown
    """
    global telegram_scheduler
    if telegram_scheduler:
        telegram_scheduler.stop_scheduler()
        logger.info("✅ Telegram service cleaned up successfully")


# Export public API
__all__ = [
    'TelegramTemplates',
    'TelegramService',
    'TelegramAdminService',
    'TelegramScheduler',
    'init_telegram_service',
    'cleanup_telegram_service'
]