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

# Visual Constants - Compact for mobile
DIVIDER_SHORT = "━━━━━━━━━━━━━"  # 13 chars for mobile
DIVIDER = "━━━━━━━━━━━━━━━━━━━"  # 19 chars standard
SIGNATURE = "💙 <b>CineBrain</b>"
COMPACT_TAGLINE = "🍿 Smart picks • Latest releases"

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
    Mobile-optimized templates for CineBrain
    Compact, readable, and beautiful on small screens
    """
    
    @staticmethod
    def get_rating_display(rating: Optional[float]) -> str:
        """Compact rating display"""
        if not rating:
            return "⭐"
        if rating >= 9.0:
            return f"🏆 {rating}"
        elif rating >= 8.0:
            return f"⭐ {rating}"
        else:
            return f"⭐ {rating}"
    
    @staticmethod
    def format_runtime(runtime: Optional[int]) -> Optional[str]:
        """Compact runtime format"""
        if not runtime:
            return None
        hours = runtime // 60
        minutes = runtime % 60
        if hours > 0:
            return f"{hours}h{minutes}m"
        return f"{minutes}m"
    
    @staticmethod
    def format_genres(genres_list: Optional[List[str]], limit: int = 2) -> str:
        """Format genres compactly"""
        if not genres_list:
            return ""
        return "•".join(genres_list[:limit])
    
    @staticmethod
    def format_year(release_date: Any) -> str:
        """Extract year only"""
        if not release_date:
            return ""
        try:
            if hasattr(release_date, 'year'):
                return str(release_date.year)
            return str(release_date)[:4]
        except:
            return ""
    
    @staticmethod
    def truncate_synopsis(text: Optional[str], limit: int = 100) -> str:
        """Ultra-compact synopsis for mobile"""
        if not text:
            return "Discover this gem on CineBrain."
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(' ', 1)[0] + "..."
    
    @staticmethod
    def get_cinebrain_url(slug: str) -> str:
        """Generate CineBrain URL"""
        return f"https://cinebrain.vercel.app/explore/details.html?{slug}"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMPACT MOVIE RECOMMENDATION
    # Mobile-first design with minimal height
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def movie_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        """
        Ultra-compact movie recommendation for mobile
        Total lines: ~15-18 (from 35+)
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
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, 100)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        # Compact insight
        if len(description) > 60:
            description = description[:60] + "..."
        
        # Build compact info line
        info_parts = []
        if rating:
            info_parts.append(rating)
        if runtime:
            info_parts.append(runtime)
        if genres:
            info_parts.append(genres)
        info_line = " • ".join(info_parts)
        
        message = f"""🎬 <b>{content.title.upper()}</b> ({year})
{info_line}

💡 <i>{description}</i>

📖 {synopsis}

👉 <a href="{url}"><b>WATCH ON CINEBRAIN</b></a>

{COMPACT_TAGLINE}
#CineBrain #NowWatching

{SIGNATURE}"""
        
        return message
    
    @staticmethod
    def tv_show_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        """Compact TV series template"""
        
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        genres = TelegramTemplates.format_genres(genres_list)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, 100)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        if len(description) > 60:
            description = description[:60] + "..."
        
        info_parts = []
        if rating:
            info_parts.append(rating)
        info_parts.append("📺 Series")
        if genres:
            info_parts.append(genres)
        info_line = " • ".join(info_parts)
        
        message = f"""📺 <b>{content.title.upper()}</b> ({year})
{info_line}

💡 <i>{description}</i>

📖 {synopsis}

👉 <a href="{url}"><b>BINGE ON CINEBRAIN</b></a>

{COMPACT_TAGLINE}
#CineBrain #BingeWorthy

{SIGNATURE}"""
        
        return message
    
    @staticmethod
    def anime_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None, anime_genres_list: Optional[List[str]] = None) -> str:
        """Compact anime template"""
        
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
        
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        genres = TelegramTemplates.format_genres(all_genres)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, 100)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        if len(description) > 60:
            description = description[:60] + "..."
        
        info_parts = []
        if rating:
            info_parts.append(rating)
        info_parts.append("🎌 Anime")
        if genres:
            info_parts.append(genres)
        info_line = " • ".join(info_parts)
        
        message = f"""🎌 <b>{content.title.upper()}</b> ({year})
{info_line}

💡 <i>{description}</i>

📖 {synopsis}

👉 <a href="{url}"><b>STREAM ON CINEBRAIN</b></a>

{COMPACT_TAGLINE}
#CineBrain #Anime

{SIGNATURE}"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UPCOMING RELEASE - COMPACT
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def format_upcoming_release(content: Any, release_date: str, anticipation_reason: str) -> str:
        """Compact upcoming release notification"""
        
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                pass
        
        year = TelegramTemplates.format_year(content.release_date)
        genres = TelegramTemplates.format_genres(genres_list)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, 80)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        if len(anticipation_reason) > 60:
            anticipation_reason = anticipation_reason[:60] + "..."
        
        message = f"""🎯 <b>COMING SOON</b>

🎬 <b>{content.title.upper()}</b>
📅 {release_date} • {genres}

💡 <i>{anticipation_reason}</i>

📖 {synopsis}

👉 <a href="{url}"><b>SET REMINDER</b></a>

#CineBrain #ComingSoon

{SIGNATURE}"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRENDING ALERT - COMPACT
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def trending_alert_template(content: Any, trend_type: str = "trending", trending_reason: str = None) -> str:
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
            'trending': '⚡'
        }.get(trend_type, '⚡')
        
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        runtime = TelegramTemplates.format_runtime(content.runtime)
        genres = TelegramTemplates.format_genres(genres_list)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, 80)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        if not trending_reason:
            trending_reason = "Breaking the internet right now!"
        elif len(trending_reason) > 60:
            trending_reason = trending_reason[:60] + "..."
        
        info_parts = []
        if rating:
            info_parts.append(rating)
        if runtime:
            info_parts.append(runtime)
        if genres:
            info_parts.append(genres)
        info_line = " • ".join(info_parts)
        
        message = f"""{trend_emoji} <b>TRENDING NOW</b>

🎥 <b>{content.title.upper()}</b> ({year})
{info_line}

💡 <i>{trending_reason}</i>

👉 <a href="{url}"><b>WATCH NOW</b></a>

#CineBrain #Trending

{SIGNATURE}"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEW ARRIVAL - COMPACT
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def new_content_alert_template(content: Any, added_date: Optional[str] = None) -> str:
        """Compact new arrival notification"""
        
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                pass
        
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        runtime = TelegramTemplates.format_runtime(content.runtime)
        genres = TelegramTemplates.format_genres(genres_list)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview, 80)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        content_emoji = {
            'movie': '🎥',
            'tv': '📺',
            'anime': '🎌'
        }.get(content.content_type, '🎬')
        
        if not added_date:
            added_date = "Today"
        
        info_parts = []
        if rating:
            info_parts.append(rating)
        if runtime:
            info_parts.append(runtime)
        if genres:
            info_parts.append(genres)
        info_line = " • ".join(info_parts)
        
        message = f"""✨ <b>JUST ADDED</b> • {added_date}

{content_emoji} <b>{content.title.upper()}</b> ({year})
{info_line}

📖 {synopsis}

👉 <a href="{url}"><b>WATCH NOW</b></a>

#CineBrain #NewArrival

{SIGNATURE}"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WEEKLY DIGEST - ULTRA COMPACT
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def weekly_digest_template(recommendations_list: List[Dict], week_number: int) -> str:
        """Ultra-compact weekly digest"""
        
        message = f"""📬 <b>WEEKLY PICKS</b> • Week {week_number}

"""
        
        for i, rec in enumerate(recommendations_list[:4], 1):  # Reduced to 4 items
            title = rec.get('title', 'Unknown')
            rating = rec.get('rating', 'N/A')
            
            if rating != 'N/A':
                rating_str = f"⭐{rating}"
            else:
                rating_str = ""
            
            # Super compact: just number, title, and rating
            message += f"{i}. <b>{title}</b> {rating_str}\n"
        
        message += f"""
👉 <a href="https://cinebrain.vercel.app"><b>SEE ALL PICKS</b></a>

#CineBrain #WeeklyDigest

{SIGNATURE}"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BATCH CONTENT - COMPACT LIST
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def batch_content_template(content_list: List[Any], batch_type: str = "content", theme: Optional[str] = None) -> str:
        """Compact batch notification"""
        
        emoji = {'movies': '🎬', 'shows': '📺', 'anime': '🎌'}.get(batch_type, '🎬')
        count = len(content_list)
        
        if theme:
            header = f"{emoji} <b>{theme}</b>"
        else:
            header = f"{emoji} <b>{count} NEW {batch_type.upper()}</b>"
        
        message = f"""{header}

"""
        
        for i, content in enumerate(content_list[:3], 1):  # Show only 3
            rating = content.rating if content.rating else ''
            if rating:
                rating_str = f" ⭐{rating}"
            else:
                rating_str = ""
            
            message += f"{i}. <b>{content.title}</b>{rating_str}\n"
        
        if count > 3:
            message += f"<i>+{count - 3} more</i>\n"
        
        message += f"""
👉 <a href="https://cinebrain.vercel.app"><b>BROWSE ALL</b></a>

#CineBrain #{batch_type.title()}

{SIGNATURE}"""
        
        return message


class TelegramService:
    """
    Mobile-optimized Telegram notification service
    Handles all public channel communications
    """
    
    @staticmethod
    def _get_mobile_optimized_poster_url(poster_path: str) -> Optional[str]:
        """
        Get mobile-optimized poster URL
        Using w342 size for better mobile loading (instead of w500)
        """
        if not poster_path:
            return None
        
        if poster_path.startswith('http'):
            return poster_path
        else:
            # Use smaller size for mobile
            return f"https://image.tmdb.org/t/p/w342{poster_path}"
    
    @staticmethod
    def send_admin_recommendation(content: Any, admin_name: str, description: str) -> bool:
        """Send mobile-optimized admin recommendation"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram recommendation skipped - channel not configured")
                return False
            
            # Select appropriate template
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
            
            # Get mobile-optimized poster
            poster_url = TelegramService._get_mobile_optimized_poster_url(content.poster_path)
            
            # Simplified mobile keyboard - single column for better mobile UX
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            
            detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
            
            # Single prominent button
            watch_btn = types.InlineKeyboardButton(
                text="🎬 Watch Now",
                url=detail_url
            )
            keyboard.add(watch_btn)
            
            # Add trailer if available
            if hasattr(content, 'youtube_trailer_id') and content.youtube_trailer_id:
                trailer_btn = types.InlineKeyboardButton(
                    text="🎞️ Trailer",
                    url=f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                )
                keyboard.add(trailer_btn)
            
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
                    logger.info(f"✅ Mobile-optimized recommendation sent: {content.title}")
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
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False
    
    @staticmethod
    def send_trending_alert(content: Any, trend_type: str = "trending", trending_reason: str = None) -> bool:
        """Send mobile-optimized trending alert"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.trending_alert_template(content, trend_type, trending_reason)
            
            poster_url = TelegramService._get_mobile_optimized_poster_url(content.poster_path)
            
            # Single column keyboard for mobile
            keyboard = types.InlineKeyboardMarkup(row_width=1)
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
    def send_new_content_alert(content: Any, added_date: Optional[str] = None) -> bool:
        """Send mobile-optimized new content alert"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.new_content_alert_template(content, added_date)
            
            poster_url = TelegramService._get_mobile_optimized_poster_url(content.poster_path)
            
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
            
            discover_btn = types.InlineKeyboardButton(
                text="✨ Watch Now",
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
    def send_upcoming_release(content: Any, release_date: str, anticipation_reason: str) -> bool:
        """Send mobile-optimized upcoming release notification"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.format_upcoming_release(content, release_date, anticipation_reason)
            
            poster_url = TelegramService._get_mobile_optimized_poster_url(content.poster_path)
            
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            detail_url = TelegramTemplates.get_cinebrain_url(content.slug)
            
            notify_btn = types.InlineKeyboardButton(
                text="🔔 Set Reminder",
                url=detail_url
            )
            keyboard.add(notify_btn)
            
            if hasattr(content, 'youtube_trailer_id') and content.youtube_trailer_id:
                trailer_btn = types.InlineKeyboardButton(
                    text="🎞️ Trailer",
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
        """Send mobile-optimized weekly digest"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.weekly_digest_template(recommendations_list, week_number)
            
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            
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
    def send_new_content_batch_alert(content_list: List[Any], batch_type: str = "content", theme: Optional[str] = None) -> bool:
        """Send mobile-optimized batch notification"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.batch_content_template(content_list, batch_type, theme)
            
            keyboard = types.InlineKeyboardMarkup(row_width=1)
            
            browse_btn = types.InlineKeyboardButton(
                text="🎬 Browse All",
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
    Admin notification service - Compact for mobile
    """
    
    @staticmethod
    def send_content_notification(content_title: str, admin_name: str, action_type: str = "added") -> bool:
        """Send compact admin notification"""
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

<b>Content:</b> {content_title}
<b>Admin:</b> {admin_name}
<b>Action:</b> {action_type.upper()}
<b>Time:</b> {datetime.utcnow().strftime('%H:%M UTC')}

#CineBrain"""
            
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
        """Send compact stats to admin"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                return False
            
            message = f"""📊 <b>Quick Stats</b>

<b>Recommendations:</b> {stats_data.get('total', 0):,}
<b>This Week:</b> {stats_data.get('this_week', 0)}
<b>Views:</b> {stats_data.get('views', 0):,}
<b>CTR:</b> {stats_data.get('ctr', 0):.1f}%

#CineBrain"""
            
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
    """
    Background scheduler for automated tasks
    """
    
    def __init__(self, app=None):
        self.app = app
        self.running = False
    
    def start_scheduler(self):
        """Start the background scheduler"""
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
                    
                    # Daily trending check - 6 PM UTC
                    if now.hour == 18 and now.minute == 0:
                        if self.app:
                            with self.app.app_context():
                                self._check_trending_content()
                    
                    time.sleep(3600)  # Check every hour
                    
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    time.sleep(300)
        
        thread = threading.Thread(target=scheduler_worker, daemon=True)
        thread.start()
        logger.info("✅ Telegram scheduler started")
    
    def stop_scheduler(self):
        """Stop the background scheduler"""
        self.running = False
        logger.info("🛑 Scheduler stopped")
    
    def _send_weekly_digest(self):
        """Send weekly digest"""
        try:
            week_number = datetime.utcnow().isocalendar()[1]
            logger.info(f"Processing weekly digest for week {week_number}")
        except Exception as e:
            logger.error(f"Weekly digest error: {e}")
    
    def _check_trending_content(self):
        """Check trending content"""
        try:
            logger.info("Checking for trending content")
        except Exception as e:
            logger.error(f"Trending check error: {e}")


# Module-level scheduler
telegram_scheduler = None


def init_telegram_service(app, db, models, services) -> Optional[Dict[str, Any]]:
    """Initialize mobile-optimized Telegram service"""
    global telegram_scheduler
    
    try:
        telegram_scheduler = TelegramScheduler(app)
        
        if bot:
            telegram_scheduler.start_scheduler()
            logger.info("✅ CineBrain Mobile Telegram service initialized")
            logger.info("   ├─ Compact mobile templates: ✓")
            logger.info("   ├─ Optimized image sizes: ✓")
            logger.info("   ├─ Single-column buttons: ✓")
            logger.info("   └─ Reduced message height: ✓")
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


# Export public API
__all__ = [
    'TelegramTemplates',
    'TelegramService', 
    'TelegramAdminService',
    'TelegramScheduler',
    'init_telegram_service',
    'cleanup_telegram_service'
]