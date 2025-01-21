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
    """Premium cinematic templates for CineBrain's Telegram channel"""
    
    @staticmethod
    def get_rating_display(rating):
        """Format rating with trophy emoji for high scores"""
        if not rating:
            return "🏆 <b>N/A</b>"
        if rating >= 9.0:
            return f"🏆 <b>{rating}/10</b>"
        elif rating >= 8.0:
            return f"⭐ <b>{rating}/10</b>"
        else:
            return f"⭐ <b>{rating}/10</b>"
    
    @staticmethod
    def format_runtime(runtime):
        """Format runtime elegantly"""
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
    def format_year(release_date):
        """Extract and format year"""
        if not release_date:
            return ""
        try:
            if hasattr(release_date, 'year'):
                return f"({release_date.year})"
            return f"({str(release_date)[:4]})"
        except:
            return ""
    
    @staticmethod
    def truncate_text(text, limit=150):
        """Elegantly truncate text at word boundary"""
        if not text:
            return "A cinematic masterpiece awaits your discovery on CineBrain."
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(' ', 1)[0] + "..."
    
    @staticmethod
    def get_cinebrain_url(slug):
        """Generate CineBrain detail page URL"""
        return f"https://cinebrain.vercel.app/explore/details.html?{slug}"
    
    @staticmethod
    def get_content_emoji(content_type):
        """Get appropriate emoji for content type"""
        return {
            'movie': '🎥',
            'tv': '📺',
            'anime': '🎌',
            'series': '📺'
        }.get(content_type, '🎬')
    
    # ═══════════════════════════════════════════════════════════════
    # MOVIE RECOMMENDATION TEMPLATE
    # Premium cinematic layout with CineBrain Insight branding
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def movie_recommendation_template(content, admin_name, description, genres_list=None):
        """Ultra-premium movie recommendation with cinematic layout"""
        
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
        overview = TelegramTemplates.truncate_text(content.overview, 150)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        # Build info line with proper spacing
        info_parts = [rating]
        if runtime:
            info_parts.append(f"⏱ <b>{runtime}</b>")
        info_parts.append(f"🎭 {genres}")
        info_line = " | ".join(info_parts)
        
        message = f"""🎬 <b>CineBrain Spotlight</b>
━━━━━━━━━━━━━━━━━━━

<b>🎥 {content.title.upper()}</b> <i>{year}</i>

{info_line}

━━━━━━━━━━━━━━━━━━━
💡 <b>CineBrain Insight</b>
<i>"{description}"</i>

━━━━━━━━━━━━━━━━━━━
📖 <b>Synopsis</b>
<i>{overview}</i>

━━━━━━━━━━━━━━━━━━━
🎯 <b>Watch on CineBrain</b>
👉 <a href="{url}"><b>TAP HERE FOR FULL DETAILS</b></a>

🍿 <i>Personalized recommendations & latest updates only on CineBrain</i>

#CineBrain #SmartRecommendations #NowTrending

╰┈➤ Curated with 💙 by <b>CineBrain</b>"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════
    # TV SERIES RECOMMENDATION TEMPLATE
    # Binge-worthy series with editorial insights
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def tv_show_recommendation_template(content, admin_name, description, genres_list=None):
        """Premium TV series template with binge-worthy appeal"""
        
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
        overview = TelegramTemplates.truncate_text(content.overview, 150)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        # Build info line
        info_parts = [rating, f"📺 <b>Series</b>", f"🎭 {genres}"]
        info_line = " | ".join(info_parts)
        
        message = f"""📺 <b>CineBrain Spotlight</b>
━━━━━━━━━━━━━━━━━━━

<b>📺 {content.title.upper()}</b> <i>{year}</i>

{info_line}

━━━━━━━━━━━━━━━━━━━
💡 <b>CineBrain Insight</b>
<i>"{description}"</i>

━━━━━━━━━━━━━━━━━━━
📖 <b>Synopsis</b>
<i>{overview}</i>

━━━━━━━━━━━━━━━━━━━
🎯 <b>Binge on CineBrain</b>
👉 <a href="{url}"><b>START WATCHING NOW</b></a>

🍿 <i>Complete seasons & exclusive content on CineBrain</i>

#CineBrain #BingeWorthy #TVSeries

╰┈➤ Curated with 💙 by <b>CineBrain</b>"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════
    # ANIME RECOMMENDATION TEMPLATE
    # Japanese animation with otaku appeal
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def anime_recommendation_template(content, admin_name, description, genres_list=None, anime_genres_list=None):
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
        elif content.anime_genres:
            try:
                all_genres.extend(json.loads(content.anime_genres))
            except:
                pass
        
        # Format components
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        genres = TelegramTemplates.format_genres(all_genres)
        overview = TelegramTemplates.truncate_text(content.overview, 150)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        
        # Build info line
        info_parts = [rating, f"🎌 <b>Anime</b>", f"🎭 {genres}"]
        info_line = " | ".join(info_parts)
        
        # Add original title if different
        title_display = f"<b>🎌 {content.title.upper()}</b>"
        if content.original_title and content.original_title != content.title:
            title_display += f"\n<i>{content.original_title}</i>"
        
        message = f"""🎌 <b>CineBrain Spotlight</b>
━━━━━━━━━━━━━━━━━━━

{title_display} <i>{year}</i>

{info_line}

━━━━━━━━━━━━━━━━━━━
💡 <b>CineBrain Insight</b>
<i>"{description}"</i>

━━━━━━━━━━━━━━━━━━━
📖 <b>Synopsis</b>
<i>{overview}</i>

━━━━━━━━━━━━━━━━━━━
🎯 <b>Stream on CineBrain</b>
👉 <a href="{url}"><b>WATCH WITH SUBTITLES</b></a>

🍜 <i>Latest anime & classics streaming on CineBrain</i>

#CineBrain #Anime #MustWatch

╰┈➤ Curated with 💙 by <b>CineBrain</b>"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════
    # NEW CONTENT ALERT TEMPLATE
    # Fresh arrivals with immediate appeal
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def new_content_alert_template(content):
        """Clean new arrival notification"""
        
        # Parse genres
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                pass
        
        # Format components
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        runtime = TelegramTemplates.format_runtime(content.runtime)
        genres = TelegramTemplates.format_genres(genres_list)
        overview = TelegramTemplates.truncate_text(content.overview, 120)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        emoji = TelegramTemplates.get_content_emoji(content.content_type)
        
        # Build info line
        info_parts = [rating]
        if runtime:
            info_parts.append(f"⏱ <b>{runtime}</b>")
        info_parts.append(f"🎭 {genres}")
        info_line = " | ".join(info_parts)
        
        message = f"""🆕 <b>Just Added to CineBrain</b>
━━━━━━━━━━━━━━━━━━━

<b>{emoji} {content.title.upper()}</b> <i>{year}</i>

{info_line}

━━━━━━━━━━━━━━━━━━━
📖 <b>Synopsis</b>
<i>{overview}</i>

━━━━━━━━━━━━━━━━━━━
🎯 <b>Available Now</b>
👉 <a href="{url}"><b>WATCH ON CINEBRAIN</b></a>

#CineBrain #NewArrival #JustAdded

╰┈➤ Fresh content daily on <b>CineBrain</b>"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════
    # TRENDING ALERT TEMPLATE
    # Viral content with urgency
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def trending_alert_template(content, trend_type="trending"):
        """Trending content with viral appeal"""
        
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
        rating = TelegramTemplates.get_rating_display(content.rating)
        runtime = TelegramTemplates.format_runtime(content.runtime)
        genres = TelegramTemplates.format_genres(genres_list)
        overview = TelegramTemplates.truncate_text(content.overview, 120)
        url = TelegramTemplates.get_cinebrain_url(content.slug)
        emoji = TelegramTemplates.get_content_emoji(content.content_type)
        
        # Build info line
        info_parts = [rating]
        if runtime:
            info_parts.append(f"⏱ <b>{runtime}</b>")
        info_parts.append(f"🎭 {genres}")
        info_line = " | ".join(info_parts)
        
        message = f"""{trend_emoji} <b>{trend_type.upper()} NOW</b>
━━━━━━━━━━━━━━━━━━━

<b>{emoji} {content.title.upper()}</b> <i>{year}</i>

{info_line}

━━━━━━━━━━━━━━━━━━━
📖 <b>What's the Buzz?</b>
<i>{overview}</i>

━━━━━━━━━━━━━━━━━━━
🎯 <b>Join the Hype</b>
👉 <a href="{url}"><b>WATCH NOW ON CINEBRAIN</b></a>

#CineBrain #{trend_type.title()} #MustWatch

╰┈➤ Trending content on <b>CineBrain</b>"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════
    # WEEKLY DIGEST TEMPLATE
    # Curated weekly highlights
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def weekly_digest_template(recommendations_list, week_number):
        """Weekly curated picks in compact format"""
        
        message = f"""🌟 <b>CineBrain Weekly</b> • <i>Week {week_number}</i>
━━━━━━━━━━━━━━━━━━━

<b>TOP PICKS THIS WEEK</b>

"""
        
        for i, rec in enumerate(recommendations_list[:5], 1):
            emoji = TelegramTemplates.get_content_emoji(rec.get('content_type', 'movie'))
            rating = rec.get('rating', 'N/A')
            if rating != 'N/A':
                rating = f"{rating}/10"
            
            # Truncate description to keep it compact
            desc = rec.get('description', '')[:50]
            if len(rec.get('description', '')) > 50:
                desc += "..."
            
            message += f"""<b>{i}.</b> {emoji} <b>{rec['title']}</b>
   ⭐ {rating} | <i>{desc}</i>

"""
        
        message += f"""━━━━━━━━━━━━━━━━━━━
🎯 <b>Explore All Picks</b>
👉 <a href="https://cinebrain.vercel.app"><b>VISIT CINEBRAIN</b></a>

#CineBrain #WeeklyPicks #Curated

╰┈➤ Your weekly dose from <b>CineBrain</b>"""
        
        return message
    
    # ═══════════════════════════════════════════════════════════════
    # BATCH CONTENT TEMPLATE
    # Multiple new additions announcement
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def batch_content_template(content_list, batch_type="content"):
        """Batch content drop notification"""
        
        emoji = {'movies': '🎬', 'shows': '📺', 'anime': '🎌'}.get(batch_type, '🎬')
        count = len(content_list)
        
        message = f"""{emoji} <b>{count} NEW {batch_type.upper()} ADDED</b>
━━━━━━━━━━━━━━━━━━━

<b>LATEST ADDITIONS</b>

"""
        
        for i, content in enumerate(content_list[:5], 1):
            rating = content.rating if content.rating else 'N/A'
            if rating != 'N/A':
                rating = f"{rating}/10"
            
            message += f"<b>{i}.</b> {emoji} <b>{content.title}</b> • ⭐ {rating}\n"
        
        if count > 5:
            message += f"\n<i>...plus {count - 5} more titles!</i>\n"
        
        message += f"""
━━━━━━━━━━━━━━━━━━━
🎯 <b>Browse Collection</b>
👉 <a href="https://cinebrain.vercel.app"><b>EXPLORE ALL ON CINEBRAIN</b></a>

#CineBrain #NewContent #{batch_type.title()}

╰┈➤ Fresh content daily on <b>CineBrain</b>"""
        
        return message

class TelegramService:
    """Service for sending beautifully formatted Telegram notifications"""
    
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        """Send admin-curated recommendation with premium formatting"""
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
            if content.youtube_trailer_id:
                trailer_btn = types.InlineKeyboardButton(
                    text="🎥 Watch Trailer",
                    url=f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                )
                keyboard.add(trailer_btn)
            
            # CineBrain exploration button
            explore_btn = types.InlineKeyboardButton(
                text="🌐 Explore CineBrain",
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
            
            logger.info(f"✅ Premium recommendation sent: {content.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False
    
    @staticmethod
    def send_trending_alert(content, trend_type="trending"):
        """Send trending content alert with viral appeal"""
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
            explore_btn = types.InlineKeyboardButton(
                text="🌐 More Trending",
                url="https://cinebrain.vercel.app"
            )
            keyboard.add(watch_btn)
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
    def send_new_content_alert(content):
        """Send new content arrival notification"""
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
            browse_btn = types.InlineKeyboardButton(
                text="🆕 More New Arrivals",
                url="https://cinebrain.vercel.app"
            )
            keyboard.add(discover_btn)
            keyboard.add(browse_btn)
            
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
        """Send weekly curated digest"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = TelegramTemplates.weekly_digest_template(recommendations_list, week_number)
            
            keyboard = types.InlineKeyboardMarkup()
            explore_btn = types.InlineKeyboardButton(
                text="🌟 View All Picks",
                url="https://cinebrain.vercel.app"
            )
            subscribe_btn = types.InlineKeyboardButton(
                text="🔔 Get Updates",
                url="https://t.me/" + TELEGRAM_CHANNEL_ID.replace('@', '')
            )
            keyboard.add(explore_btn)
            keyboard.add(subscribe_btn)
            
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
        """Send batch content notification"""
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
    """Admin notification service for internal updates"""
    
    @staticmethod
    def send_content_notification(content_title, admin_name, action_type="added"):
        """Send admin action notification"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                return False
            
            action_emoji = {
                'added': '➕',
                'updated': '✏️',
                'deleted': '🗑️',
                'recommended': '⭐'
            }.get(action_type, '📝')
            
            message = f"""{action_emoji} <b>ADMIN ACTION</b>
━━━━━━━━━━━━━━━━━━━

<b>Content:</b> {content_title}
<b>Admin:</b> {admin_name}
<b>Action:</b> {action_type.upper()}
<b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

━━━━━━━━━━━━━━━━━━━
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
    def send_recommendation_stats(stats_data):
        """Send recommendation statistics to admin"""
        try:
            if not bot or not TELEGRAM_ADMIN_CHAT_ID:
                return False
            
            message = f"""📊 <b>RECOMMENDATION STATS</b>
━━━━━━━━━━━━━━━━━━━

<b>📈 Overview</b>
• Total Recommendations: <b>{stats_data.get('total', 0)}</b>
• This Week: <b>{stats_data.get('this_week', 0)}</b>
• Top Admin: <b>{stats_data.get('top_admin', 'N/A')}</b>
• Top Genre: <b>{stats_data.get('top_genre', 'N/A')}</b>

<b>🎯 Engagement</b>
• Views: <b>{stats_data.get('views', 0):,}</b>
• Clicks: <b>{stats_data.get('clicks', 0):,}</b>
• CTR: <b>{stats_data.get('ctr', 0):.2f}%</b>

━━━━━━━━━━━━━━━━━━━
<i>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</i>

#Stats #CineBrain"""
            
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
    """Background scheduler for automated Telegram tasks"""
    
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
        """Stop the background scheduler"""
        self.running = False
        logger.info("🛑 Scheduler stopped")
    
    def _send_weekly_digest(self):
        """Internal method to send weekly digest"""
        try:
            week_number = datetime.utcnow().isocalendar()[1]
            logger.info(f"Would send weekly digest for week {week_number}")
            # Implementation would query recent recommendations from database
            # and call TelegramService.send_weekly_digest()
        except Exception as e:
            logger.error(f"Weekly digest error: {e}")

telegram_scheduler = None

def init_telegram_service(app, db, models, services):
    """Initialize Telegram service with all components"""
    global telegram_scheduler
    
    try:
        telegram_scheduler = TelegramScheduler(app)
        
        if bot:
            telegram_scheduler.start_scheduler()
            logger.info("✅ Telegram service initialized with CineBrain branding")
            logger.info("   - Premium cinematic templates: ✓")
            logger.info("   - CineBrain Insight branding: ✓")
            logger.info("   - Content recommendations: ✓")
            logger.info("   - Trending alerts: ✓")
            logger.info("   - Admin notifications: ✓")
            logger.info("   - Weekly digest scheduler: ✓")
        else:
            logger.warning("⚠️ Telegram bot not configured - service disabled")
        
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
    """Cleanup Telegram service resources"""
    global telegram_scheduler
    if telegram_scheduler:
        telegram_scheduler.stop_scheduler()
        logger.info("✅ Telegram service cleaned up")