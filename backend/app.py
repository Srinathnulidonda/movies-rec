from flask import Flask, request, jsonify, session, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import requests
import os
import json
import logging
from functools import wraps
import sqlite3
from collections import defaultdict, Counter
import random
import hashlib
import time
from sqlalchemy import func, and_, or_, desc, text
import telebot
import threading
from geopy.geocoders import Nominatim
import jwt
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import re

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Database configuration
if os.environ.get('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_recommendations.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# API Keys
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '52260795')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '1002850793757')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-xmf5.onrender.com')
JUSTWATCH_API_KEY = os.environ.get('JUSTWATCH_API_KEY', 'your_justwatch_api_key')
WATCHMODE_API_KEY = os.environ.get('WATCHMODE_API_KEY', 'WtcKDji9i20pjOl5Lg0AiyG2bddfUs3nSZRZJIsY')

# Initialize Telegram bot
if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk':
    try:
        bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
    except:
        bot = None
        logging.warning("Failed to initialize Telegram bot")
else:
    bot = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    preferred_languages = db.Column(db.Text)
    preferred_genres = db.Column(db.Text)
    location = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    imdb_id = db.Column(db.String(20))
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    content_type = db.Column(db.String(20), nullable=False)
    genres = db.Column(db.Text)
    languages = db.Column(db.Text)
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float)
    vote_count = db.Column(db.Integer)
    popularity = db.Column(db.Float)
    overview = db.Column(db.Text)
    poster_path = db.Column(db.String(255))
    backdrop_path = db.Column(db.String(255))
    trailer_url = db.Column(db.String(255))
    ott_platforms = db.Column(db.Text)
    youtube_availability = db.Column(db.Text)  # New field for YouTube data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AdminRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recommendation_type = db.Column(db.String(50))
    description = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnonymousInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)
    ip_address = db.Column(db.String(45))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class TelegramService:
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            # Format genre list
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Parse OTT availability
            ott_info = ""
            watch_links = ""
            hashtags = "#AdminChoice #MovieRecommendation #CineScope"
            
            if content.ott_platforms:
                try:
                    ott_data = json.loads(content.ott_platforms)
                    platforms = ott_data.get('platforms', [])
                    
                    if platforms:
                        free_platforms = [p for p in platforms if p.get('is_free')]
                        paid_platforms = [p for p in platforms if not p.get('is_free')]
                        youtube_platforms = [p for p in platforms if 'youtube' in p.get('platform_id', '')]
                        
                        ott_info = "\n\n━━━━━━━━━━━━━━━━━\n📺 **WHERE TO WATCH**\n\n"
                        
                        # YouTube availability
                        youtube_free = [p for p in youtube_platforms if p.get('platform_id') == 'youtube']
                        if youtube_free:
                            ott_info += "🎥 **FREE on YouTube!**\n"
                            for yt in youtube_free[:1]:  # Show only the best YouTube link
                                ott_info += f"▶️ Watch Now: {yt['watch_url']}\n"
                            ott_info += "\n"
                            hashtags += " #FreeOnYouTube"
                        
                        # Other free platforms
                        other_free = [p for p in free_platforms if 'youtube' not in p.get('platform_id', '')]
                        if other_free:
                            ott_info += "🆓 **FREE Platforms:**\n"
                            for platform in other_free[:3]:
                                ott_info += f"• {platform['platform_name']} - {platform['watch_url']}\n"
                            ott_info += "\n"
                        
                        # Paid platforms
                        if paid_platforms:
                            ott_info += "💰 **Subscription/Paid:**\n"
                            for platform in paid_platforms[:5]:
                                platform_name = platform['platform_name']
                                if platform.get('availability_type') == 'rent':
                                    platform_name += " (Rent)"
                                elif platform.get('availability_type') == 'buy':
                                    platform_name += " (Buy)"
                                ott_info += f"• {platform_name} - {platform['watch_url']}\n"
                                
                                # Add platform hashtags
                                if 'netflix' in platform.get('platform_id', '').lower():
                                    hashtags += " #Netflix"
                                elif 'prime' in platform.get('platform_id', '').lower():
                                    hashtags += " #PrimeVideo"
                                elif 'hotstar' in platform.get('platform_id', '').lower():
                                    hashtags += " #Hotstar"
                        
                        ott_info += "\n━━━━━━━━━━━━━━━━━"
                
                except Exception as e:
                    logger.error(f"Error parsing OTT data: {e}")
            
            # Create message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}{ott_info}

{hashtags}"""
            
            # Send message with photo if available
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='Markdown',
                        disable_web_page_preview=False
                    )
                except Exception as photo_error:
                    logger.error(f"Failed to send photo, sending text only: {photo_error}")
                    bot.send_message(
                        TELEGRAM_CHANNEL_ID, 
                        message, 
                        parse_mode='Markdown',
                        disable_web_page_preview=False
                    )
            else:
                bot.send_message(
                    TELEGRAM_CHANNEL_ID, 
                    message, 
                    parse_mode='Markdown',
                    disable_web_page_preview=False
                )
            
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

# Enhanced OTT Platform Information
OTT_PLATFORMS = {
    # International Platforms
    'netflix': {
        'name': 'Netflix',
        'is_free': False,
        'base_url': 'https://netflix.com',
        'deep_link_pattern': 'https://netflix.com/title/{id}',
        'region_support': ['global'],
        'subscription_required': True,
        'logo_url': 'https://www.netflix.com/favicon.ico'
    },
    'amazon_prime': {
        'name': 'Amazon Prime Video',
        'is_free': False,
        'base_url': 'https://primevideo.com',
        'deep_link_pattern': 'https://primevideo.com/detail/{id}',
        'region_support': ['global'],
        'subscription_required': True,
        'logo_url': 'https://primevideo.com/favicon.ico'
    },
    'disney_plus': {
        'name': 'Disney+ Hotstar',
        'is_free': False,
        'base_url': 'https://hotstar.com',
        'deep_link_pattern': 'https://hotstar.com/{id}',
        'region_support': ['IN', 'US', 'CA'],
        'subscription_required': True,
        'logo_url': 'https://hotstar.com/favicon.ico'
    },
    'hulu': {
        'name': 'Hulu',
        'is_free': False,
        'base_url': 'https://hulu.com',
        'deep_link_pattern': 'https://hulu.com/watch/{id}',
        'region_support': ['US'],
        'subscription_required': True,
        'logo_url': 'https://hulu.com/favicon.ico'
    },
    
    # Free Platforms
    'youtube': {
        'name': 'YouTube',
        'is_free': True,
        'base_url': 'https://youtube.com',
        'deep_link_pattern': 'https://youtube.com/watch?v={id}',
        'region_support': ['global'],
        'subscription_required': False,
        'logo_url': 'https://youtube.com/favicon.ico'
    },
    'youtube_premium': {
        'name': 'YouTube Premium',
        'is_free': False,
        'base_url': 'https://youtube.com',
        'deep_link_pattern': 'https://youtube.com/watch?v={id}',
        'region_support': ['global'],
        'subscription_required': True,
        'logo_url': 'https://youtube.com/favicon.ico'
    },
    'jiocinema': {
        'name': 'JioCinema',
        'is_free': True,
        'base_url': 'https://jiocinema.com',
        'deep_link_pattern': 'https://jiocinema.com/movies/{id}',
        'region_support': ['IN'],
        'subscription_required': False,
        'logo_url': 'https://jiocinema.com/favicon.ico'
    },
    'mx_player': {
        'name': 'MX Player',
        'is_free': True,
        'base_url': 'https://mxplayer.in',
        'deep_link_pattern': 'https://mxplayer.in/movie/{id}',
        'region_support': ['IN'],
        'subscription_required': False,
        'logo_url': 'https://mxplayer.in/favicon.ico'
    },
    'voot': {
        'name': 'Voot',
        'is_free': True,
        'base_url': 'https://voot.com',
        'deep_link_pattern': 'https://voot.com/shows/{id}',
        'region_support': ['IN'],
        'subscription_required': False,
        'logo_url': 'https://voot.com/favicon.ico'
    },
    'zee5': {
        'name': 'ZEE5',
        'is_free': False,
        'base_url': 'https://zee5.com',
        'deep_link_pattern': 'https://zee5.com/movies/details/{id}',
        'region_support': ['IN'],
        'subscription_required': True,
        'logo_url': 'https://zee5.com/favicon.ico'
    },
    'sonyliv': {
        'name': 'SonyLIV',
        'is_free': False,
        'base_url': 'https://sonyliv.com',
        'deep_link_pattern': 'https://sonyliv.com/shows/{id}',
        'region_support': ['IN'],
        'subscription_required': True,
        'logo_url': 'https://sonyliv.com/favicon.ico'
    },
    'alt_balaji': {
        'name': 'ALTBalaji',
        'is_free': False,
        'base_url': 'https://altbalaji.com',
        'deep_link_pattern': 'https://altbalaji.com/show/{id}',
        'region_support': ['IN'],
        'subscription_required': True,
        'logo_url': 'https://altbalaji.com/favicon.ico'
    },
    'eros_now': {
        'name': 'Eros Now',
        'is_free': False,
        'base_url': 'https://erosnow.com',
        'deep_link_pattern': 'https://erosnow.com/movie/{id}',
        'region_support': ['IN'],
        'subscription_required': True,
        'logo_url': 'https://erosnow.com/favicon.ico'
    },
    'hoichoi': {
        'name': 'Hoichoi',
        'is_free': False,
        'base_url': 'https://hoichoi.tv',
        'deep_link_pattern': 'https://hoichoi.tv/films/{id}',
        'region_support': ['IN'],
        'subscription_required': True,
        'logo_url': 'https://hoichoi.tv/favicon.ico'
    },
    'shemaroo_me': {
        'name': 'ShemarooMe',
        'is_free': True,
        'base_url': 'https://shemaroome.com',
        'deep_link_pattern': 'https://shemaroome.com/movies/{id}',
        'region_support': ['IN'],
        'subscription_required': False,
        'logo_url': 'https://shemaroome.com/favicon.ico'
    }
}

# OTT Cache
class OTTCache:
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=6)
    
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < self.cache_duration:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, data):
        self.cache[key] = (data, datetime.utcnow())
    
    def clear_expired(self):
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= self.cache_duration
        ]
        for key in expired_keys:
            del self.cache[key]

ott_cache = OTTCache()

# Helper Functions
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

def require_admin(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user or not current_user.is_admin:
                return jsonify({'error': 'Admin access required'}), 403
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = hashlib.md5(f"{request.remote_addr}{time.time()}".encode()).hexdigest()
    return session['session_id']

# Enhanced YouTube Service
class EnhancedYouTubeService:
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    # Official movie channels and distributors
    OFFICIAL_CHANNELS = {
        'bollywood': [
            'UCq-Fj5jknLsUf-MWSy4_brA',  # T-Series
            'UCjvgGbPPn-FV0I2Ms_S3AeA',  # Zee Music Company
            'UCFFbwnve3yF62-tV3hIXhBg',  # Sony Music India
            'UCpEhnqL0y41EpW2TvWAHD7Q',  # Saregama
            'UCq-Fj5jknLsUf-MWSy4_brA',  # Shemaroo Entertainment
        ],
        'hollywood': [
            'UCiifkYAs_bq1pt_zbNAzYGg',  # Sony Pictures
            'UC_VEXHtGJKI34LHaFZu_9jQ',  # Warner Bros
            'UCZf2iKB7HZN6Kj7UrXrtUyA',  # Universal Pictures
            'UCy4P9dbGpJx4RAnXWCKZLGg',  # Disney Movie Trailers
            'UCjy6CpE2M2FHvd2Cv-6w0-Q',  # Paramount Pictures
        ],
        'regional': [
            'UCq-Fj5jknLsUf-MWSy4_brA',  # T-Series (Telugu/Tamil)
            'UCuFYtjZ8yaWOTZ6ZRbNAXzQ',  # Aditya Music
            'UCIJLWx_k1TnXhO6Vma0eXQw',  # Lahari Music
            'UC_sAgmPd5WLdWqINR1yVY6A',  # Mango Music
        ]
    }
    
    @staticmethod
    async def get_comprehensive_youtube_availability(title, original_title=None, release_year=None, content_type='movie', region='IN'):
        """Get comprehensive YouTube availability including free movies, trailers, and premium content"""
        if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == 'your_youtube_api_key':
            return []
        
        try:
            youtube_data = {
                'free_movies': [],
                'premium_content': [],
                'trailers': [],
                'clips': [],
                'official_content': [],
                'last_checked': datetime.utcnow().isoformat()
            }
            
            # Multiple search strategies
            search_results = await EnhancedYouTubeService._comprehensive_search(title, original_title, release_year, content_type, region)
            
            # Categorize results
            for video in search_results:
                category = EnhancedYouTubeService._categorize_video(video, title)
                if category:
                    youtube_data[category].append(video)
            
            # Get official channel content
            official_content = await EnhancedYouTubeService._search_official_channels(title, region)
            youtube_data['official_content'].extend(official_content)
            
            return youtube_data
            
        except Exception as e:
            logger.error(f"YouTube comprehensive search error: {e}")
            return []
    
    @staticmethod
    async def _comprehensive_search(title, original_title, release_year, content_type, region):
        """Perform comprehensive search with multiple strategies"""
        all_results = []
        
        # Search queries with different variations
        search_queries = EnhancedYouTubeService._generate_search_queries(title, original_title, release_year, content_type, region)
        
        async with aiohttp.ClientSession() as session:
            for query in search_queries[:10]:  # Limit to 10 queries to avoid quota issues
                try:
                    url = f"{EnhancedYouTubeService.BASE_URL}/search"
                    params = {
                        'key': YOUTUBE_API_KEY,
                        'q': query,
                        'part': 'snippet',
                        'type': 'video',
                        'maxResults': 10,
                        'order': 'relevance',
                        'regionCode': region,
                        'safeSearch': 'moderate'
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get('items', []):
                                video_data = await EnhancedYouTubeService._get_detailed_video_info(session, item)
                                if video_data:
                                    all_results.append(video_data)
                        
                        # Add delay to respect rate limits
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"YouTube search error for query '{query}': {e}")
                    continue
        
        # Remove duplicates
        seen_ids = set()
        unique_results = []
        for video in all_results:
            if video['video_id'] not in seen_ids:
                seen_ids.add(video['video_id'])
                unique_results.append(video)
        
        return unique_results
    
    @staticmethod
    def _generate_search_queries(title, original_title, release_year, content_type, region):
        """Generate comprehensive search queries"""
        queries = []
        year_str = str(release_year) if release_year else ""
        
        # Basic searches
        queries.extend([
            f"{title} full movie",
            f"{title} full {content_type}",
            f"{title} movie {year_str}",
            f"{title} complete movie",
            f"{title} HD full movie",
        ])
        
        # Original title searches
        if original_title and original_title != title:
            queries.extend([
                f"{original_title} full movie",
                f"{original_title} movie {year_str}",
                f"{original_title} complete movie"
            ])
        
        # Language-specific searches based on region
        if region == 'IN':
            queries.extend([
                f"{title} hindi movie",
                f"{title} tamil movie",
                f"{title} telugu movie",
                f"{title} bollywood movie",
                f"{title} full movie hindi",
                f"{title} full movie with subtitles"
            ])
        
        # Free movie searches
        queries.extend([
            f"{title} free movie",
            f"{title} movie free online",
            f"{title} full movie free",
            f"watch {title} free",
            f"{title} movie online free"
        ])
        
        # Official channel searches
        queries.extend([
            f"{title} official",
            f"{title} movie official",
            f"{title} trailer official",
            f"{title} songs",
            f"{title} clips"
        ])
        
        # Premium content searches
        queries.extend([
            f"{title} movie rent",
            f"{title} buy movie",
            f"{title} youtube movies",
            f"{title} pay per view"
        ])
        
        return queries
    
    @staticmethod
    async def _get_detailed_video_info(session, video_item):
        """Get detailed information about a video"""
        try:
            video_id = video_item['id']['videoId']
            
            # Get video details
            url = f"{EnhancedYouTubeService.BASE_URL}/videos"
            params = {
                'key': YOUTUBE_API_KEY,
                'id': video_id,
                'part': 'snippet,contentDetails,statistics,status',
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('items'):
                        video_details = data['items'][0]
                        
                        # Parse duration
                        duration = EnhancedYouTubeService._parse_duration(
                            video_details['contentDetails'].get('duration', 'PT0S')
                        )
                        
                        return {
                            'video_id': video_id,
                            'title': video_details['snippet']['title'],
                            'description': video_details['snippet']['description'],
                            'channel_title': video_details['snippet']['channelTitle'],
                            'channel_id': video_details['snippet']['channelId'],
                            'published_at': video_details['snippet']['publishedAt'],
                            'duration_seconds': duration,
                            'duration_formatted': EnhancedYouTubeService._format_duration(duration),
                            'view_count': int(video_details['statistics'].get('viewCount', 0)),
                            'like_count': int(video_details['statistics'].get('likeCount', 0)),
                            'thumbnail_url': video_details['snippet']['thumbnails'].get('high', {}).get('url'),
                            'watch_url': f"https://youtube.com/watch?v={video_id}",
                            'embed_url': f"https://youtube.com/embed/{video_id}",
                            'is_live': video_details['snippet'].get('liveBroadcastContent') == 'live',
                            'is_premium': EnhancedYouTubeService._check_if_premium(video_details),
                            'quality_score': EnhancedYouTubeService._calculate_quality_score(video_details, duration)
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Video details error for {video_item.get('id', {}).get('videoId')}: {e}")
            return None
    
    @staticmethod
    def _parse_duration(duration_str):
        """Parse YouTube duration format (PT1H2M3S) to seconds"""
        import re
        
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_str)
        
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds
        
        return 0
    
    @staticmethod
    def _format_duration(seconds):
        """Format seconds to readable duration"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    @staticmethod
    def _categorize_video(video, movie_title):
        """Categorize video based on content analysis"""
        title = video['title'].lower()
        description = video['description'].lower()
        duration = video['duration_seconds']
        
        # Check for full movies (typically longer than 60 minutes)
        if duration > 3600:  # More than 1 hour
            if any(keyword in title for keyword in ['full movie', 'complete movie', 'full film', 'movie', 'पूरी फिल्म']):
                if video['is_premium']:
                    return 'premium_content'
                else:
                    return 'free_movies'
        
        # Check for trailers (typically 1-5 minutes)
        elif 60 <= duration <= 300:
            if any(keyword in title for keyword in ['trailer', 'teaser', 'preview', 'promo']):
                return 'trailers'
        
        # Check for clips and songs (typically 3-15 minutes)
        elif 180 <= duration <= 900:
            if any(keyword in title for keyword in ['song', 'clip', 'scene', 'dialogue', 'making', 'behind']):
                return 'clips'
        
        # Default categorization based on quality score
        if video['quality_score'] > 0.7:
            if video['is_premium']:
                return 'premium_content'
            else:
                return 'free_movies'
        
        return None
    
    @staticmethod
    def _check_if_premium(video_details):
        """Check if video is premium content"""
        # Check for YouTube Premium indicators
        title = video_details['snippet']['title'].lower()
        description = video_details['snippet']['description'].lower()
        
        premium_indicators = [
            'youtube premium',
            'rent or buy',
            'purchase',
            'premium movie',
            'paid content'
        ]
        
        return any(indicator in title or indicator in description for indicator in premium_indicators)
    
    @staticmethod
    def _calculate_quality_score(video_details, duration):
        """Calculate quality score based on various factors"""
        score = 0.0
        
        # Duration score (prefer full-length content)
        if duration > 3600:  # Full movie length
            score += 0.4
        elif duration > 1800:  # Half movie length
            score += 0.2
        
        # View count score
        view_count = int(video_details['statistics'].get('viewCount', 0))
        if view_count > 1000000:  # 1M+ views
            score += 0.3
        elif view_count > 100000:  # 100K+ views
            score += 0.2
        elif view_count > 10000:  # 10K+ views
            score += 0.1
        
        # Like ratio score
        like_count = int(video_details['statistics'].get('likeCount', 0))
        if like_count > 1000:
            score += 0.1
        
        # Channel verification (simplified)
        channel_title = video_details['snippet']['channelTitle'].lower()
        verified_indicators = ['official', 'music', 'entertainment', 'movies', 'cinema']
        if any(indicator in channel_title for indicator in verified_indicators):
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    @staticmethod
    async def _search_official_channels(title, region):
        """Search in official channels for the content"""
        official_content = []
        
        if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == 'your_youtube_api_key':
            return official_content
        
        try:
            # Determine channel list based on region
            if region == 'IN':
                channels = EnhancedYouTubeService.OFFICIAL_CHANNELS['bollywood'] + EnhancedYouTubeService.OFFICIAL_CHANNELS['regional']
            else:
                channels = EnhancedYouTubeService.OFFICIAL_CHANNELS['hollywood']
            
            async with aiohttp.ClientSession() as session:
                for channel_id in channels[:5]:  # Limit to 5 channels
                    try:
                        url = f"{EnhancedYouTubeService.BASE_URL}/search"
                        params = {
                            'key': YOUTUBE_API_KEY,
                            'q': title,
                            'part': 'snippet',
                            'type': 'video',
                            'channelId': channel_id,
                            'maxResults': 5,
                            'order': 'relevance'
                        }
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                for item in data.get('items', []):
                                    video_data = await EnhancedYouTubeService._get_detailed_video_info(session, item)
                                    if video_data:
                                        video_data['is_official'] = True
                                        official_content.append(video_data)
                            
                            await asyncio.sleep(0.1)  # Rate limiting
                            
                    except Exception as e:
                        logger.error(f"Official channel search error for {channel_id}: {e}")
                        continue
            
            return official_content
            
        except Exception as e:
            logger.error(f"Official channels search error: {e}")
            return official_content

# Enhanced OTT Availability Service
class OTTAvailabilityService:
    
    @staticmethod
    async def get_comprehensive_availability(tmdb_id, content_type='movie', region='IN'):
        """Get availability from multiple sources including enhanced YouTube"""
        cache_key = f"{tmdb_id}_{content_type}_{region}"
        
        # Check cache first
        cached_result = ott_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Get title information first
        title_data = await OTTAvailabilityService.get_tmdb_title(tmdb_id, content_type)
        if not title_data:
            return {'platforms': [], 'youtube_data': {}, 'last_updated': datetime.utcnow().isoformat()}
        
        availability_data = {
            'platforms': [],
            'free_options': [],
            'paid_options': [],
            'rent_options': [],
            'buy_options': [],
            'youtube_data': {},
            'last_updated': datetime.utcnow().isoformat(),
            'region': region,
            'title': title_data['title']
        }
        
        try:
            # Get comprehensive YouTube availability
            youtube_data = await EnhancedYouTubeService.get_comprehensive_youtube_availability(
                title_data['title'],
                title_data.get('original_title'),
                title_data.get('release_year'),
                content_type,
                region
            )
            availability_data['youtube_data'] = youtube_data
            
            # Convert YouTube data to platform format
            youtube_platforms = OTTAvailabilityService._convert_youtube_to_platforms(youtube_data)
            availability_data['platforms'].extend(youtube_platforms)
            
            # Get other OTT platforms
            tmdb_data = await OTTAvailabilityService.get_tmdb_providers(tmdb_id, content_type, region)
            if tmdb_data:
                availability_data['platforms'].extend(tmdb_data)
            
            # Get Indian platform availability
            indian_platforms = await OTTAvailabilityService.get_indian_platform_availability(title_data['title'], tmdb_id)
            if indian_platforms:
                availability_data['platforms'].extend(indian_platforms)
            
            # Categorize platforms
            for platform in availability_data['platforms']:
                if platform['is_free']:
                    availability_data['free_options'].append(platform)
                elif platform['availability_type'] == 'subscription':
                    availability_data['paid_options'].append(platform)
                elif platform['availability_type'] == 'rent':
                    availability_data['rent_options'].append(platform)
                elif platform['availability_type'] == 'buy':
                    availability_data['buy_options'].append(platform)
            
            # Cache the result
            ott_cache.set(cache_key, availability_data)
            
            return availability_data
            
        except Exception as e:
            logger.error(f"OTT availability error: {e}")
            return availability_data
    
    @staticmethod
    def _convert_youtube_to_platforms(youtube_data):
        """Convert YouTube data to platform format"""
        platforms = []
        
        # Free movies
        for movie in youtube_data.get('free_movies', []):
            platforms.append({
                'platform_id': 'youtube',
                'platform_name': 'YouTube',
                'logo_url': 'https://youtube.com/favicon.ico',
                'watch_url': movie['watch_url'],
                'availability_type': 'free',
                'is_free': True,
                'price': None,
                'currency': None,
                'source': 'youtube',
                'video_title': movie['title'],
                'duration': movie['duration_formatted'],
                'quality_score': movie['quality_score'],
                'view_count': movie['view_count'],
                'channel_name': movie['channel_title'],
                'content_type': 'full_movie'
            })
        
        # Premium content
        for movie in youtube_data.get('premium_content', []):
            platforms.append({
                'platform_id': 'youtube_premium',
                'platform_name': 'YouTube Premium',
                'logo_url': 'https://youtube.com/favicon.ico',
                'watch_url': movie['watch_url'],
                'availability_type': 'subscription',
                'is_free': False,
                'price': None,
                'currency': 'USD',
                'source': 'youtube',
                'video_title': movie['title'],
                'duration': movie['duration_formatted'],
                'quality_score': movie['quality_score'],
                'view_count': movie['view_count'],
                'channel_name': movie['channel_title'],
                'content_type': 'premium_movie'
            })
        
        # Add best trailer if available
        trailers = youtube_data.get('trailers', [])
        if trailers:
            best_trailer = max(trailers, key=lambda x: x['quality_score'])
            platforms.append({
                'platform_id': 'youtube_trailer',
                'platform_name': 'YouTube (Trailer)',
                'logo_url': 'https://youtube.com/favicon.ico',
                'watch_url': best_trailer['watch_url'],
                'availability_type': 'free',
                'is_free': True,
                'price': None,
                'currency': None,
                'source': 'youtube',
                'video_title': best_trailer['title'],
                'duration': best_trailer['duration_formatted'],
                'quality_score': best_trailer['quality_score'],
                'view_count': best_trailer['view_count'],
                'channel_name': best_trailer['channel_title'],
                'content_type': 'trailer'
            })
        
        return platforms
    
    @staticmethod
    async def get_tmdb_providers(tmdb_id, content_type, region):
        """Get availability from TMDB Watch Providers API"""
        try:
            url = f"https://api.themoviedb.org/3/{content_type}/{tmdb_id}/watch/providers"
            params = {'api_key': TMDB_API_KEY}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        providers = []
                        
                        if 'results' in data and region in data['results']:
                            region_data = data['results'][region]
                            
                            # Streaming subscriptions
                            if 'flatrate' in region_data:
                                for provider in region_data['flatrate']:
                                    providers.append({
                                        'platform_id': provider['provider_name'].lower().replace(' ', '_'),
                                        'platform_name': provider['provider_name'],
                                        'logo_url': f"https://image.tmdb.org/t/p/original{provider['logo_path']}",
                                        'watch_url': OTTAvailabilityService.generate_watch_url(provider['provider_name'], tmdb_id),
                                        'availability_type': 'subscription',
                                        'is_free': False,
                                        'price': None,
                                        'currency': None,
                                        'source': 'tmdb'
                                    })
                            
                            # Rental options
                            if 'rent' in region_data:
                                for provider in region_data['rent']:
                                    providers.append({
                                        'platform_id': provider['provider_name'].lower().replace(' ', '_'),
                                        'platform_name': provider['provider_name'],
                                        'logo_url': f"https://image.tmdb.org/t/p/original{provider['logo_path']}",
                                        'watch_url': OTTAvailabilityService.generate_watch_url(provider['provider_name'], tmdb_id),
                                        'availability_type': 'rent',
                                        'is_free': False,
                                        'price': None,
                                        'currency': 'INR' if region == 'IN' else 'USD',
                                        'source': 'tmdb'
                                    })
                            
                            # Purchase options
                            if 'buy' in region_data:
                                for provider in region_data['buy']:
                                    providers.append({
                                        'platform_id': provider['provider_name'].lower().replace(' ', '_'),
                                        'platform_name': provider['provider_name'],
                                        'logo_url': f"https://image.tmdb.org/t/p/original{provider['logo_path']}",
                                        'watch_url': OTTAvailabilityService.generate_watch_url(provider['provider_name'], tmdb_id),
                                        'availability_type': 'buy',
                                        'is_free': False,
                                        'price': None,
                                        'currency': 'INR' if region == 'IN' else 'USD',
                                        'source': 'tmdb'
                                    })
                        
                        return providers
        except Exception as e:
            logger.error(f"TMDB providers error: {e}")
            return []
    
    @staticmethod
    async def get_indian_platform_availability(title, tmdb_id):
        """Check availability on Indian platforms (simplified simulation)"""
        try:
            providers = []
            
            # Simulate availability for popular Indian platforms
            indian_platforms = {
                'jiocinema': {'probability': 0.3, 'is_free': True},
                'mx_player': {'probability': 0.4, 'is_free': True},
                'zee5': {'probability': 0.25, 'is_free': False},
                'sonyliv': {'probability': 0.2, 'is_free': False},
                'voot': {'probability': 0.3, 'is_free': True}
            }
            
            # Simple probability-based availability (in real implementation, you'd call actual APIs)
            for platform_id, config in indian_platforms.items():
                if random.random() < config['probability']:  # Simulate availability
                    platform_info = OTT_PLATFORMS.get(platform_id)
                    if platform_info:
                        providers.append({
                            'platform_id': platform_id,
                            'platform_name': platform_info['name'],
                            'logo_url': platform_info.get('logo_url'),
                            'watch_url': platform_info['deep_link_pattern'].format(id=tmdb_id),
                            'availability_type': 'free' if config['is_free'] else 'subscription',
                            'is_free': config['is_free'],
                            'price': None,
                            'currency': 'INR',
                            'source': 'indian_platform'
                        })
            
            return providers
            
        except Exception as e:
            logger.error(f"Indian platform availability error: {e}")
            return []
    
    @staticmethod
    async def get_tmdb_title(tmdb_id, content_type):
        """Get title from TMDB"""
        try:
            url = f"https://api.themoviedb.org/3/{content_type}/{tmdb_id}"
            params = {'api_key': TMDB_API_KEY}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        release_date = data.get('release_date') or data.get('first_air_date')
                        release_year = None
                        if release_date:
                            try:
                                release_year = int(release_date.split('-')[0])
                            except:
                                pass
                        
                        return {
                            'title': data.get('title') or data.get('name'),
                            'original_title': data.get('original_title') or data.get('original_name'),
                            'release_date': release_date,
                            'release_year': release_year
                        }
            return None
            
        except Exception as e:
            logger.error(f"TMDB title fetch error: {e}")
            return None
    
    @staticmethod
    def generate_watch_url(provider_name, tmdb_id):
        """Generate watch URL for a provider"""
        provider_mapping = {
            'Netflix': f"https://netflix.com/search?q=tmdb{tmdb_id}",
            'Amazon Prime Video': f"https://primevideo.com/search/ref=atv_sr_def_c_unkc_1_1?phrase=tmdb{tmdb_id}",
            'Disney+ Hotstar': f"https://hotstar.com/search/{tmdb_id}",
            'Hulu': f"https://hulu.com/search?q={tmdb_id}",
            'YouTube': f"https://youtube.com/results?search_query=tmdb{tmdb_id}",
            'JioCinema': f"https://jiocinema.com/search/{tmdb_id}",
            'MX Player': f"https://mxplayer.in/search/{tmdb_id}",
            'Voot': f"https://voot.com/search/{tmdb_id}",
            'ZEE5': f"https://zee5.com/search/{tmdb_id}",
            'SonyLIV': f"https://sonyliv.com/search/{tmdb_id}"
        }
        
        return provider_mapping.get(provider_name, f"https://google.com/search?q={provider_name}+tmdb{tmdb_id}")

# [Include all other services and API routes from the previous backend/app.py file]
# [Due to length constraints, I'm showing the key enhanced sections]

# External API Services (include TMDBService, OMDbService, JikanService, YouTubeService from previous file)
class TMDBService:
    BASE_URL = 'https://api.themoviedb.org/3'
    
    @staticmethod
    def search_content(query, content_type='multi', language='en-US', page=1):
        url = f"{TMDBService.BASE_URL}/search/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'language': language,
            'page': page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB search error: {e}")
        return None
    
    @staticmethod
    def get_content_details(content_id, content_type='movie'):
        url = f"{TMDBService.BASE_URL}/{content_type}/{content_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'append_to_response': 'credits,videos,similar,watch/providers'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB details error: {e}")
        return None
    
    @staticmethod
    def get_trending(content_type='all', time_window='day', page=1):
        url = f"{TMDBService.BASE_URL}/trending/{content_type}/{time_window}"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB trending error: {e}")
        return None
    
    @staticmethod
    def get_popular(content_type='movie', page=1, region=None):
        url = f"{TMDBService.BASE_URL}/{content_type}/popular"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        if region:
            params['region'] = region
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB popular error: {e}")
        return None

class OMDbService:
    BASE_URL = 'http://www.omdbapi.com/'
    
    @staticmethod
    def get_content_by_imdb(imdb_id):
        params = {
            'apikey': OMDB_API_KEY,
            'i': imdb_id,
            'plot': 'full'
        }
        
        try:
            response = requests.get(OMDbService.BASE_URL, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"OMDb error: {e}")
        return None

class JikanService:
    BASE_URL = 'https://api.jikan.moe/v4'
    
    @staticmethod
    def search_anime(query, page=1):
        url = f"{JikanService.BASE_URL}/anime"
        params = {
            'q': query,
            'page': page,
            'limit': 20
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan search error: {e}")
        return None
    
    @staticmethod
    def get_top_anime(type='tv', page=1):
        url = f"{JikanService.BASE_URL}/top/anime"
        params = {
            'type': type,
            'page': page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan top anime error: {e}")
        return None

# Content Management Service
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            # Check if content already exists
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                return existing
            
            # Extract genres
            genres = []
            if 'genres' in tmdb_data:
                genres = [genre['name'] for genre in tmdb_data['genres']]
            elif 'genre_ids' in tmdb_data:
                genres = ContentService.map_genre_ids(tmdb_data['genre_ids'])
            
            # Extract languages
            languages = []
            if 'spoken_languages' in tmdb_data:
                languages = [lang['name'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
            
            # Get OTT platforms asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ott_data = loop.run_until_complete(
                    OTTAvailabilityService.get_comprehensive_availability(
                        tmdb_data['id'], content_type, 'IN'
                    )
                )
            finally:
                loop.close()
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
                title=tmdb_data.get('title') or tmdb_data.get('name'),
                original_title=tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                content_type=content_type,
                genres=json.dumps(genres),
                languages=json.dumps(languages),
                release_date=datetime.strptime(tmdb_data.get('release_date') or tmdb_data.get('first_air_date', '1900-01-01'), '%Y-%m-%d').date() if tmdb_data.get('release_date') or tmdb_data.get('first_air_date') else None,
                runtime=tmdb_data.get('runtime'),
                rating=tmdb_data.get('vote_average'),
                vote_count=tmdb_data.get('vote_count'),
                popularity=tmdb_data.get('popularity'),
                overview=tmdb_data.get('overview'),
                poster_path=tmdb_data.get('poster_path'),
                backdrop_path=tmdb_data.get('backdrop_path'),
                ott_platforms=json.dumps(ott_data),
                youtube_availability=json.dumps(ott_data.get('youtube_data', {}))
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def map_genre_ids(genre_ids):
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]

# Telegram Service
class TelegramService:
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            # Format genre list
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Add YouTube availability info
            youtube_info = ""
            if content.youtube_availability:
                try:
                    youtube_data = json.loads(content.youtube_availability)
                    if youtube_data.get('free_movies'):
                        youtube_info = "\n🎬 **Free on YouTube!**"
                    elif youtube_data.get('trailers'):
                        youtube_info = "\n📺 **Trailer on YouTube**"
                except:
                    pass
            
            # Create message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}{youtube_info}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

#AdminChoice #MovieRecommendation #CineScope"""
            
            # Send message with photo if available
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='Markdown'
                    )
                except Exception as photo_error:
                    logger.error(f"Failed to send photo, sending text only: {photo_error}")
                    bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            else:
                bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

# API Routes for Enhanced OTT

@app.route('/api/ott/availability/<int:content_id>', methods=['GET'])
def get_ott_availability(content_id):
    """Get real-time OTT availability for content including YouTube"""
    try:
        content = Content.query.get_or_404(content_id)
        region = request.args.get('region', 'IN')
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        if not content.tmdb_id:
            return jsonify({'error': 'TMDB ID not available for this content'}), 400
        
        # Check if we need to refresh
        if force_refresh or not content.ott_platforms or not content.youtube_availability:
            # Get fresh data
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                availability_data = loop.run_until_complete(
                    OTTAvailabilityService.get_comprehensive_availability(
                        content.tmdb_id,
                        content.content_type,
                        region
                    )
                )
                
                # Update content in database
                content.ott_platforms = json.dumps(availability_data)
                content.youtube_availability = json.dumps(availability_data.get('youtube_data', {}))
                content.updated_at = datetime.utcnow()
                db.session.commit()
                
            finally:
                loop.close()
        else:
            # Use cached data
            try:
                availability_data = json.loads(content.ott_platforms or '{}')
                if not availability_data.get('youtube_data') and content.youtube_availability:
                    availability_data['youtube_data'] = json.loads(content.youtube_availability)
            except:
                availability_data = {'platforms': [], 'youtube_data': {}}
        
        return jsonify({
            'content_id': content.id,
            'title': content.title,
            'availability': availability_data
        }), 200
        
    except Exception as e:
        logger.error(f"OTT availability error: {e}")
        return jsonify({'error': 'Failed to get OTT availability'}), 500

@app.route('/api/ott/youtube/<int:content_id>', methods=['GET'])
def get_youtube_availability(content_id):
    """Get detailed YouTube availability for content"""
    try:
        content = Content.query.get_or_404(content_id)
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        if not content.tmdb_id:
            return jsonify({'error': 'TMDB ID not available for this content'}), 400
        
        # Check if we need to refresh YouTube data
        youtube_data = {}
        if force_refresh or not content.youtube_availability:
            # Get fresh YouTube data
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                title_data = loop.run_until_complete(
                    OTTAvailabilityService.get_tmdb_title(content.tmdb_id, content.content_type)
                )
                
                if title_data:
                    youtube_data = loop.run_until_complete(
                        EnhancedYouTubeService.get_comprehensive_youtube_availability(
                            title_data['title'],
                            title_data.get('original_title'),
                            title_data.get('release_year'),
                            content.content_type,
                            'IN'
                        )
                    )
                    
                    # Update content
                    content.youtube_availability = json.dumps(youtube_data)
                    content.updated_at = datetime.utcnow()
                    db.session.commit()
                
            finally:
                loop.close()
        else:
            # Use cached data
            try:
                youtube_data = json.loads(content.youtube_availability or '{}')
            except:
                youtube_data = {}
        
        return jsonify({
            'content_id': content.id,
            'title': content.title,
            'youtube_data': youtube_data
        }), 200
        
    except Exception as e:
        logger.error(f"YouTube availability error: {e}")
        return jsonify({'error': 'Failed to get YouTube availability'}), 500

# [Include all other API routes from the previous backend/app.py file]
# Authentication Routes, Content Discovery Routes, Recommendation Routes, etc.
# [Due to length constraints, including key routes only]

# Authentication Routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        if not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password']),
            preferred_languages=json.dumps(data.get('preferred_languages', [])),
            preferred_genres=json.dumps(data.get('preferred_genres', []))
        )
        
        db.session.add(user)
        db.session.commit()
        
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Missing username or password'}), 400
        
        user = User.query.filter_by(username=data['username']).first()
        
        if not user or not check_password_hash(user.password_hash, data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

# Content Routes
@app.route('/api/content/<int:content_id>', methods=['GET'])
def get_content_details(content_id):
    try:
        content = Content.query.get_or_404(content_id)
        
        # Record view interaction
        session_id = get_session_id()
        interaction = AnonymousInteraction(
            session_id=session_id,
            content_id=content.id,
            interaction_type='view',
            ip_address=request.remote_addr
        )
        db.session.add(interaction)
        
        # Get additional details from TMDB
        additional_details = None
        if content.tmdb_id:
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        
        # Get trailers from YouTube data
        trailers = []
        if content.youtube_availability:
            try:
                youtube_data = json.loads(content.youtube_availability)
                trailers = youtube_data.get('trailers', [])[:5]
            except:
                pass
        
        # Get similar content
        similar_content = []
        if additional_details and 'similar' in additional_details:
            for item in additional_details['similar']['results'][:5]:
                similar = ContentService.save_content_from_tmdb(item, content.content_type)
                if similar:
                    similar_content.append({
                        'id': similar.id,
                        'title': similar.title,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{similar.poster_path}" if similar.poster_path else None,
                        'rating': similar.rating
                    })
        
        db.session.commit()
        
        return jsonify({
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'runtime': content.runtime,
            'rating': content.rating,
            'vote_count': content.vote_count,
            'overview': content.overview,
            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path else None,
            'ott_platforms': json.loads(content.ott_platforms or '{}'),
            'youtube_availability': json.loads(content.youtube_availability or '{}'),
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Admin Routes
@app.route('/api/admin/content', methods=['POST'])
@require_admin
def save_external_content(current_user):
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No content data provided'}), 400
        
        # Check if content already exists
        existing_content = None
        if data.get('id'):
            existing_content = Content.query.filter_by(tmdb_id=data['id']).first()
        
        if existing_content:
            return jsonify({
                'message': 'Content already exists',
                'content_id': existing_content.id
            }), 200
        
        # Create new content with enhanced OTT data
        try:
            release_date = None
            if data.get('release_date'):
                try:
                    release_date = datetime.strptime(data['release_date'], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            # Get comprehensive OTT availability
            ott_data = {}
            youtube_data = {}
            if data.get('id'):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    ott_data = loop.run_until_complete(
                        OTTAvailabilityService.get_comprehensive_availability(
                            data['id'],
                            data.get('content_type', 'movie'),
                            'IN'
                        )
                    )
                    youtube_data = ott_data.get('youtube_data', {})
                finally:
                    loop.close()
            
            content = Content(
                tmdb_id=data.get('id'),
                title=data.get('title'),
                original_title=data.get('original_title'),
                content_type=data.get('content_type', 'movie'),
                genres=json.dumps(data.get('genres', [])),
                languages=json.dumps(data.get('languages', ['en'])),
                release_date=release_date,
                runtime=data.get('runtime'),
                rating=data.get('rating'),
                vote_count=data.get('vote_count'),
                popularity=data.get('popularity'),
                overview=data.get('overview'),
                poster_path=data.get('poster_path'),
                backdrop_path=data.get('backdrop_path'),
                ott_platforms=json.dumps(ott_data),
                youtube_availability=json.dumps(youtube_data)
            )
            
            db.session.add(content)
            db.session.commit()
            
            return jsonify({
                'message': 'Content saved successfully with OTT data',
                'content_id': content.id,
                'youtube_available': len(youtube_data.get('free_movies', [])) > 0
            }), 201
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving content: {e}")
            return jsonify({'error': 'Failed to save content to database'}), 500
        
    except Exception as e:
        logger.error(f"Save content error: {e}")
        return jsonify({'error': 'Failed to process content'}), 500

@app.route('/api/admin/recommendations', methods=['POST'])
@require_admin
def create_admin_recommendation(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'recommendation_type', 'description']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        content = Content.query.get(data['content_id'])
        if not content:
            content = Content.query.filter_by(tmdb_id=data['content_id']).first()
        
        if not content:
            return jsonify({'error': 'Content not found. Please save content first.'}), 404
        
        admin_rec = AdminRecommendation(
            content_id=content.id,
            admin_id=current_user.id,
            recommendation_type=data['recommendation_type'],
            description=data['description']
        )
        
        db.session.add(admin_rec)
        db.session.commit()
        
        # Send to Telegram channel
        telegram_success = TelegramService.send_admin_recommendation(content, current_user.username, data['description'])
        
        return jsonify({
            'message': 'Admin recommendation created successfully',
            'telegram_sent': telegram_success
        }), 201
        
    except Exception as e:
        logger.error(f"Admin recommendation error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create recommendation'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'features': ['ott_availability', 'youtube_integration', 'telegram_posting']
    }), 200

# Initialize database
def create_tables():
    try:
        with app.app_context():
            db.create_all()
            
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(
                    username='admin',
                    email='admin@example.com',
                    password_hash=generate_password_hash('admin123'),
                    is_admin=True
                )
                db.session.add(admin)
                db.session.commit()
                logger.info("Admin user created with username: admin, password: admin123")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

# Initialize database when app starts
create_tables()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)