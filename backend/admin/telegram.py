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

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID')
TELEGRAM_ADMIN_CHAT_ID = os.environ.get('TELEGRAM_ADMIN_CHAT_ID')

DIVIDER = "━━━━━━━━━━━━━━━━━━━"
CINEBRAIN_FOOTER = "<b><i>🎥 Recommended by CineBrain</i></b>"

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
    
    @staticmethod
    def get_rating_display(rating: Optional[float]) -> str:
        if not rating:
            return "N/A"
        return f"{rating}/10"
    
    @staticmethod
    def format_runtime(runtime: Optional[int]) -> Optional[str]:
        if not runtime:
            return None
        hours = runtime // 60
        minutes = runtime % 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"
    
    @staticmethod
    def format_genres(genres_list: Optional[List[str]], limit: int = 3) -> str:
        if not genres_list:
            return "Drama"
        
        safe_genres = []
        for item in genres_list[:limit]:
            str_item = str(item)
            if str_item.isdigit():
                continue
            safe_genres.append(str_item)
        
        if not safe_genres:
            return "Drama"
        
        return " • ".join(safe_genres)
    
    @staticmethod
    def format_year(release_date: Any) -> str:
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
        if not text:
            return "A cinematic experience awaits your discovery on CineBrain."
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(' ', 1)[0] + "..."
    
    @staticmethod
    def get_content_type_prefix(content_type: str) -> str:
        content_type_lower = content_type.lower()
        
        if content_type_lower == 'anime':
            return "Anime:"
        elif content_type_lower in ['tv', 'series', 'tv_show', 'tv-show']:
            return "TV Show/Series:"
        else:
            return "Movie:"
    
    @staticmethod
    def get_cinebrain_url(slug: str) -> str:
        return f"https://cinebrain.vercel.app/explore/details.html?{slug}"
    
    @staticmethod
    def movie_recommendation_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None) -> str:
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        content_prefix = TelegramTemplates.get_content_type_prefix(content.content_type)
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        runtime = TelegramTemplates.format_runtime(content.runtime)
        genres = TelegramTemplates.format_genres(genres_list)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview)
        
        runtime_str = f" | ⏱ {runtime}" if runtime else ""
        
        message = f"""<b>🎞️ {content_prefix} {content.title}{year}</b>
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
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        content_prefix = TelegramTemplates.get_content_type_prefix(content.content_type)
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        genres = TelegramTemplates.format_genres(genres_list)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview)
        
        runtime_str = ""
        if hasattr(content, 'seasons') and content.seasons:
            runtime_str = f" | ⏱ {content.seasons} Seasons"
        
        message = f"""<b>🎞️ {content_prefix} {content.title}{year}</b>
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
        
        content_prefix = TelegramTemplates.get_content_type_prefix(content.content_type)
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        genres = TelegramTemplates.format_genres(all_genres)
        synopsis = TelegramTemplates.truncate_synopsis(content.overview)
        
        runtime_str = ""
        if hasattr(content, 'status') and content.status:
            runtime_str = f" | ⏱ {content.status}"
        else:
            runtime_str = " | ⏱ Ongoing"
        
        message = f"""<b>🎞️ {content_prefix} {content.title}{year}</b>
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
    def mind_bending_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None, overview: Optional[str] = None, if_you_like: Optional[str] = None) -> str:
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        content_prefix = TelegramTemplates.get_content_type_prefix(content.content_type)
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        runtime = TelegramTemplates.format_runtime(content.runtime)
        genres = TelegramTemplates.format_genres(genres_list, limit=3)
        
        overview_text = overview or TelegramTemplates.truncate_synopsis(content.overview, 200)
        
        runtime_str = f" • ⏱ {runtime}" if runtime else ""
        
        message = f"""🔥 <b>THIS MOVIE WILL MELT YOUR BRAIN</b>
<b>{content_prefix} {content.title}{year}</b>
<i>{genres} • ⭐ {rating}{runtime_str}</i>
{DIVIDER}
<b>Why this will break your reality:</b>
<blockquote><i>{overview_text}</i></blockquote>
• A concept that bends reality
• A twist that rewrites the whole story
{DIVIDER}"""
        
        if if_you_like:
            message += f"\n<b>If you like:</b> {if_you_like}\n"
        
        message += f"""
🔎 <i>More hidden gems — @cinebrain</i>
<i>🧠 CineBrain — Hidden Gems • Mind-Bending Sci-Fi • Rare Anime</i>

{CINEBRAIN_FOOTER}"""
        
        return message
    
    @staticmethod
    def hidden_gem_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None, hook: Optional[str] = None, if_you_like: Optional[str] = None) -> str:
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        content_prefix = TelegramTemplates.get_content_type_prefix(content.content_type)
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        genres = TelegramTemplates.format_genres(genres_list, limit=3)
        
        hook_text = hook or "A beautifully crafted gem buried under the algorithm."
        
        message = f"""💎 <b>Hidden Gem — {content_prefix} {content.title}{year}</b>
<i>{genres} • ⭐ {rating}</i>

{hook_text}"""
        
        if if_you_like:
            message += f"\n\n<b>If you like:</b> {if_you_like}"
        
        message += f"""

🔎 <i>More hidden gems — @cinebrain</i>
<i>🧠 CineBrain — Hidden Gems • Mind-Bending Sci-Fi • Rare Anime</i>

{CINEBRAIN_FOOTER}"""
        
        return message
    
    @staticmethod
    def anime_gem_template(content: Any, admin_name: str, description: str, genres_list: Optional[List[str]] = None, anime_genres_list: Optional[List[str]] = None, overview: Optional[str] = None, emotion_hook: Optional[str] = None) -> str:
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
        
        content_prefix = TelegramTemplates.get_content_type_prefix(content.content_type)
        year = TelegramTemplates.format_year(content.release_date)
        rating = TelegramTemplates.get_rating_display(content.rating)
        genres = TelegramTemplates.format_genres(all_genres, limit=4)
        
        status = "Completed"
        if hasattr(content, 'status') and content.status:
            status = content.status
        elif hasattr(content, 'release_date') and content.release_date:
            from datetime import datetime, timedelta
            if content.release_date > datetime.now().date() - timedelta(days=365):
                status = "Ongoing"
        
        overview_text = overview or TelegramTemplates.truncate_synopsis(content.overview, 180)
        
        emotion_hook_text = emotion_hook or "A time-loop tragedy that hits harder the more you think about it."
        
        message = f"""🔥 <b>THIS ANIME WILL BLOW YOUR MIND</b>
🎐 <b>Anime Gem — {content_prefix} {content.title}{year}</b>
<i>{genres} • {status} • ⭐ {rating}</i>
{DIVIDER}
<b>Why this hits hard:</b>
<blockquote><i>{overview_text}</i></blockquote>
• {emotion_hook_text}
{DIVIDER}
🔎 <i>More rare anime — @cinebrain</i>
<i>🧠 CineBrain — Hidden Gems • Mind-Bending Sci-Fi • Rare Anime</i>

{CINEBRAIN_FOOTER}"""
        
        return message
    
    @staticmethod
    def top_list_template(list_title: str, items: List[tuple], admin_name: str = "", description: str = "", poster_url: Optional[str] = None) -> str:
        limited_items = items[:10]
        
        body = "\n".join([
            f"{i+1}. <b>{title}</b> ({year}) — {hook}"
            for i, (title, year, hook) in enumerate(limited_items)
        ])
        
        message = f"""🧠 <b>{list_title}</b>

{body}
{DIVIDER}
📌 <i>Save this list — @cinebrain</i>
<i>🧠 CineBrain — Hidden Gems • Mind-Bending Sci-Fi • Rare Anime</i>

{CINEBRAIN_FOOTER}"""
        
        return message
    
    @staticmethod
    def scene_clip_template(content: Any, admin_name: str, description: str, caption: str, genres_list: Optional[List[str]] = None) -> str:
        if not genres_list and content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        content_prefix = TelegramTemplates.get_content_type_prefix(content.content_type)
        year = TelegramTemplates.format_year(content.release_date)
        genres = TelegramTemplates.format_genres(genres_list, limit=3)
        
        caption_text = caption or "This scene will hook you instantly"
        
        message = f"""<b>{caption_text}</b>
🎥 <b>{content_prefix} {content.title}{year}</b>
<i>{genres}</i>
{DIVIDER}
⚡ Watch the clip above.
If this hooks you, the full movie will blow your mind.

🔎 <i>More scenes — @cinebrain</i>
<i>🧠 CineBrain — Hidden Gems • Mind-Bending Sci-Fi • Rare Anime</i>

{CINEBRAIN_FOOTER}"""
        
        return message
    
    @staticmethod
    def get_template_prompts() -> Dict[str, Dict[str, str]]:
        return {
            'mind_bending': {
                'purpose': 'For movies that are confusing, reality-breaking, psychological, sci-fi, or have a big twist.',
                'use_when': 'The movie is genius, underrated, brain-melting.',
                'poster_support': True,
                'prompt': '''Generate a mind-bending movie recommendation.
Fill these fields with strong, intense, viral-quality text:
• Title: The exact movie title
• Year: The release year in brackets (example: (2013))
• Genres: 2–3 genres (example: Sci-Fi • Thriller • Mystery)
• Rating: IMDb or TMDB rating like 7.6/10
• Runtime: Format inside (⏱ 1h 32m)
• Overview: 2–3 sentences, mysterious, raise questions, never spoil twists
• If_you_like: 2–4 similar brain-breaking movies (example: Inception, Dark, Predestination)
Make the overview gripping, non-generic, focused on tension, confusion, and mystery.'''
            },
            'hidden_gem': {
                'purpose': 'For underrated or lesser-known movies/series.',
                'use_when': 'Movies are not mainstream but excellent.',
                'poster_support': True,
                'prompt': '''Generate a hidden gem recommendation.
Fill these fields with sharp, concise descriptions:
• Title: Movie title
• Year: Release year in brackets
• Genres: 2–3 max
• Rating: IMDb or TMDB rating
• Hook/catch-line: 1–2 lines, "why nobody talks about this" feel
• If_you_like (optional): 2–3 movies with similar tone
The hook must be extremely catchy. Avoid long explanations. Focus on uniqueness, vibe, style.'''
            },
            'anime_gem': {
                'purpose': 'For emotional, psychological, sci-fi, or plot-heavy anime.',
                'use_when': 'The anime has depth, philosophy, or a big emotional theme.',
                'poster_support': True,
                'prompt': '''Generate an anime gem recommendation.
Fill these fields with emotional and engaging detail:
• Title: Name of the anime
• Year: Release year in brackets
• Genres: 2–4 combined genres
• Status: Completed or Ongoing
• Rating: Anime rating (MyAnimeList / TMDB)
• Overview: 2–3 lines, emotional and philosophical core, focus on themes not plot
• Emotion_hook: 1 strong emotional line (example: "A time-loop tragedy that hits harder the more you think about it.")
Make the tone emotional, powerful, and intense.'''
            },
            'top_list': {
                'purpose': 'For list-style posts (Top 5, Top 10).',
                'use_when': 'You want a viral post full of mini recommendations.',
                'poster_support': True,
                'prompt': '''Generate a top-list for CineBrain.
Fill each item with:
• Movie Title
• Year  
• Hook (example: "A mind-bending paradox that will fry your brain")
Provide 5–10 items MAX. Hooks should be short, punchy, and scroll-stopping.
Also provide:
• List Title: Make it catchy and niche-based
Examples: "Top 5 Mind-Bending Sci-Fi Gems", "Top 7 Psychological Thrillers You Missed"
No long paragraphs. Just title + year + short hook.'''
            },
            'scene_clip': {
                'purpose': 'For posting a clip/video from a movie or anime.',
                'use_when': 'You upload a 10–30 sec clip to Telegram.',
                'poster_support': False,
                'prompt': '''Generate text for a scene-clip post.
Fill these fields:
• Caption: A punchy line creating curiosity (example: "This scene will hook you instantly")
• Title: Movie or Anime title
• Year: Release year in brackets
• Genres: 2–3 genres that define the tone
The tone must be exciting and suspenseful. Keep text minimal because video is main attraction.'''
            }
        }
    
    @staticmethod
    def get_available_templates() -> Dict[str, str]:
        return {
            'standard_movie': 'Standard Movie Recommendation',
            'standard_tv': 'Standard TV Show Recommendation', 
            'standard_anime': 'Standard Anime Recommendation',
            'mind_bending': '🔥 Mind-Bending Movie',
            'hidden_gem': '💎 Hidden Gem',
            'anime_gem': '🎐 Anime Gem',
            'top_list': '🧠 Top List/Curation',
            'scene_clip': '🎥 Scene Clip with Video'
        }
    
    @staticmethod
    def get_template_fields(template_type: str) -> Dict[str, Any]:
        field_definitions = {
            'mind_bending': {
                'required': ['title', 'year', 'genres', 'rating', 'runtime', 'overview'],
                'optional': ['if_you_like'],
                'poster_support': True,
                'field_specs': {
                    'genres': '2-3 genres (Sci-Fi • Thriller • Mystery)',
                    'overview': '2-3 sentences, mysterious, never spoil twists',
                    'if_you_like': '2-4 similar brain-breaking movies'
                }
            },
            'hidden_gem': {
                'required': ['title', 'year', 'genres', 'rating', 'hook'],
                'optional': ['if_you_like'],
                'poster_support': True,
                'field_specs': {
                    'genres': '2-3 genres max',
                    'hook': '1-2 lines, "why nobody talks about this" feel',
                    'if_you_like': '2-3 movies with similar tone'
                }
            },
            'anime_gem': {
                'required': ['title', 'year', 'genres', 'status', 'rating', 'overview', 'emotion_hook'],
                'optional': [],
                'poster_support': True,
                'field_specs': {
                    'genres': '2-4 combined genres',
                    'status': 'Completed or Ongoing',
                    'overview': '2-3 lines, emotional/philosophical core, themes not plot',
                    'emotion_hook': '1 strong emotional line'
                }
            },
            'top_list': {
                'required': ['list_title', 'items'],
                'optional': ['poster_url'],
                'poster_support': True,
                'field_specs': {
                    'list_title': 'Catchy and niche-based',
                    'items': '5-10 items MAX, each with title, year, short punchy hook',
                    'poster_url': 'Optional custom poster for the list'
                }
            },
            'scene_clip': {
                'required': ['caption', 'title', 'year', 'genres'],
                'optional': [],
                'poster_support': False,
                'field_specs': {
                    'caption': 'Punchy line creating curiosity',
                    'genres': '2-3 genres that define the tone'
                }
            }
        }
        
        return field_definitions.get(template_type, {})
    
    @staticmethod
    def render_template(template_type: str, content: Any = None, admin_name: str = "", description: str = "", **kwargs) -> str:
        template_map = {
            'standard_movie': TelegramTemplates.movie_recommendation_template,
            'standard_tv': TelegramTemplates.tv_show_recommendation_template,
            'standard_anime': TelegramTemplates.anime_recommendation_template,
            'mind_bending': TelegramTemplates.mind_bending_template,
            'hidden_gem': TelegramTemplates.hidden_gem_template,
            'anime_gem': TelegramTemplates.anime_gem_template,
            'scene_clip': TelegramTemplates.scene_clip_template
        }
        
        if template_type == 'top_list':
            return TelegramTemplates.top_list_template(
                kwargs.get('list_title', 'Curated List'),
                kwargs.get('items', []),
                admin_name,
                description,
                kwargs.get('poster_url')
            )
        
        template_func = template_map.get(template_type)
        if not template_func:
            if content and content.content_type == 'anime':
                template_func = TelegramTemplates.anime_recommendation_template
            elif content and content.content_type in ['tv', 'series']:
                template_func = TelegramTemplates.tv_show_recommendation_template
            else:
                template_func = TelegramTemplates.movie_recommendation_template
        
        return template_func(content, admin_name, description, **kwargs)


class TelegramService:
    
    @staticmethod
    def send_admin_recommendation(content: Any, admin_name: str, description: str, template_type: str = 'auto', template_params: Dict = None) -> bool:
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram recommendation skipped - channel not configured")
                return False
            
            template_params = template_params or {}
            
            if template_type == 'auto':
                if content and content.content_type == 'anime':
                    template_type = 'standard_anime'
                elif content and content.content_type in ['tv', 'series']:
                    template_type = 'standard_tv'
                else:
                    template_type = 'standard_movie'
            
            template_info = TelegramTemplates.get_template_fields(template_type)
            poster_support = template_info.get('poster_support', True)
            
            if template_type == 'top_list':
                message = TelegramTemplates.top_list_template(
                    template_params.get('list_title', 'Curated List'),
                    template_params.get('items', []),
                    admin_name,
                    description,
                    template_params.get('poster_url')
                )
                poster_url = template_params.get('poster_url')
            else:
                message = TelegramTemplates.render_template(
                    template_type, 
                    content, 
                    admin_name, 
                    description,
                    **template_params
                )
                
                poster_url = None
                if poster_support and content and content.poster_path:
                    if content.poster_path.startswith('http'):
                        poster_url = content.poster_path
                    else:
                        poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            
            buttons = []
            
            if template_type != 'top_list' and content and hasattr(content, 'slug') and content.slug:
                campaign_type = f"{content.content_type}_recommendation"
                content_identifier = content.slug.replace('-', '_')
                
                detail_url = cinebrain_tracking_url(
                    content.slug, 
                    campaign_type, 
                    content_identifier
                )
                
                details_btn = types.InlineKeyboardButton(
                    text="Full Details",
                    url=detail_url
                )
                buttons.append(details_btn)
            
            explore_btn = types.InlineKeyboardButton(
                text="Explore More",
                url=f"https://cinebrain.vercel.app/?utm_source=telegram&utm_medium=bot&utm_campaign=recommendation&utm_content=explore_more"
            )
            buttons.append(explore_btn)
            
            if len(buttons) == 2:
                keyboard.row(buttons[0], buttons[1])
            elif len(buttons) == 1:
                keyboard.add(buttons[0])
            
            if poster_url and poster_support and template_type != 'scene_clip':
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                    logger.info(f"✅ {template_type} recommendation with poster sent: {content.title if content else 'List'}")
                except Exception as e:
                    logger.error(f"Photo send failed: {e}, sending text only")
                    bot.send_message(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        text=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                    logger.info(f"✅ {template_type} recommendation sent (text only): {content.title if content else 'List'}")
            else:
                bot.send_message(
                    chat_id=TELEGRAM_CHANNEL_ID,
                    text=message,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
                logger.info(f"✅ {template_type} recommendation sent: {content.title if content else 'List'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram send error: {e}")
            return False


class TelegramAdminService:
    
    @staticmethod
    def send_content_notification(content_title: str, admin_name: str, action_type: str = "added") -> bool:
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
    try:
        if bot:
            logger.info("✅ CineBrain Telegram service initialized successfully")
            logger.info("   ├─ Content type prefixes: ✓")
            logger.info("   ├─ Poster support (all except scene_clip): ✓")
            logger.info("   ├─ 5 custom templates: ✓")
            logger.info("   ├─ Fixed button layout (side by side): ✓")
            logger.info("   ├─ Fixed divider formatting: ✓")
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


__all__ = [
    'TelegramTemplates',
    'TelegramService', 
    'TelegramAdminService',
    'cinebrain_tracking_url',
    'init_telegram_service'
]