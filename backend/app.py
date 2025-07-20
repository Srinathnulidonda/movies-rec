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
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Database configuration
if os.environ.get('DATABASE_URL'):
    # Production - PostgreSQL on Render
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    # Local development - SQLite
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_recommendations.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# API Keys - Set these in your environment
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '52260795')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '-1002850793757')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-xmf5.onrender.com')
WATCHMODE_API_KEY = os.environ.get('WATCHMODE_API_KEY', 'WtcKDji9i20pjOl5Lg0AiyG2bddfUs3nSZRZJIsY')

# Initialize Telegram bot
if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != 'your_telegram_bot_token':
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

# Create a session with retry strategy
def create_session_with_retries():
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Global session for API calls
api_session = create_session_with_retries()

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    preferred_languages = db.Column(db.Text)  # JSON string
    preferred_genres = db.Column(db.Text)  # JSON string
    location = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    imdb_id = db.Column(db.String(20))
    mal_id = db.Column(db.Integer)  # MyAnimeList ID for anime
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    content_type = db.Column(db.String(20), nullable=False)  # movie, tv, anime
    genres = db.Column(db.Text)  # JSON string
    languages = db.Column(db.Text)  # JSON string
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float)
    vote_count = db.Column(db.Integer)
    popularity = db.Column(db.Float)
    overview = db.Column(db.Text)
    poster_path = db.Column(db.String(255))
    backdrop_path = db.Column(db.String(255))
    trailer_url = db.Column(db.String(255))
    ott_platforms = db.Column(db.Text)  # JSON string
    streaming_links = db.Column(db.Text)  # JSON string with language-specific links
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)  # view, like, favorite, watchlist, search
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AdminRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recommendation_type = db.Column(db.String(50))  # trending, popular, critics_choice, admin_choice
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

# OTT Platform Information with Streaming Links
OTT_PLATFORMS = {
    # Free Platforms
    'mx_player': {
        'name': 'MX Player',
        'is_free': True,
        'url': 'https://mxplayer.in',
        'base_watch_url': 'https://mxplayer.in/movie/'
    },
    'jio_hotstar': {
        'name': 'JioHotstar',
        'is_free': True,
        'url': 'https://jiocinema.com',
        'base_watch_url': 'https://jiocinema.com/movies/'
    },
    'sonyliv_free': {
        'name': 'SonyLIV (Free)',
        'is_free': True,
        'url': 'https://sonyliv.com',
        'base_watch_url': 'https://sonyliv.com/movies/'
    },
    'zee5_free': {
        'name': 'Zee5 (Free)',
        'is_free': True,
        'url': 'https://zee5.com',
        'base_watch_url': 'https://zee5.com/movies/'
    },
    'youtube': {
        'name': 'YouTube',
        'is_free': True,
        'url': 'https://youtube.com',
        'base_watch_url': 'https://youtube.com/watch?v='
    },
    'crunchyroll_free': {
        'name': 'Crunchyroll (Free)',
        'is_free': True,
        'url': 'https://crunchyroll.com',
        'base_watch_url': 'https://crunchyroll.com/watch/'
    },
    'airtel_xstream': {
        'name': 'Airtel Xstream',
        'is_free': True,
        'url': 'https://airtelxstream.in',
        'base_watch_url': 'https://airtelxstream.in/movies/'
    },
    
    # Paid Platforms
    'netflix': {
        'name': 'Netflix',
        'is_free': False,
        'url': 'https://netflix.com',
        'base_watch_url': 'https://netflix.com/title/'
    },
    'prime_video': {
        'name': 'Prime Video',
        'is_free': False,
        'url': 'https://primevideo.com',
        'base_watch_url': 'https://primevideo.com/detail/'
    },
    'disney_plus_hotstar': {
        'name': 'Disney+ Hotstar',
        'is_free': False,
        'url': 'https://hotstar.com',
        'base_watch_url': 'https://hotstar.com/in/movies/'
    },
    'zee5_premium': {
        'name': 'Zee5 Premium',
        'is_free': False,
        'url': 'https://zee5.com',
        'base_watch_url': 'https://zee5.com/movies/'
    },
    'sonyliv_premium': {
        'name': 'SonyLIV Premium',
        'is_free': False,
        'url': 'https://sonyliv.com',
        'base_watch_url': 'https://sonyliv.com/movies/'
    },
    'aha': {
        'name': 'Aha',
        'is_free': False,
        'url': 'https://aha.video',
        'base_watch_url': 'https://aha.video/player/movie/'
    },
    'sun_nxt': {
        'name': 'Sun NXT',
        'is_free': False,
        'url': 'https://sunnxt.com',
        'base_watch_url': 'https://sunnxt.com/movie/'
    }
}

# Regional Language Mapping with Priorities
REGIONAL_LANGUAGES = {
    'telugu': {
        'codes': ['te', 'telugu', 'tollywood'],
        'priority': 1,
        'region': 'IN',
        'industry': 'Tollywood'
    },
    'english': {
        'codes': ['en', 'english', 'hollywood'],
        'priority': 1,
        'region': 'US',
        'industry': 'Hollywood'
    },
    'hindi': {
        'codes': ['hi', 'hindi', 'bollywood'],
        'priority': 2,
        'region': 'IN',
        'industry': 'Bollywood'
    },
    'tamil': {
        'codes': ['ta', 'tamil', 'kollywood'],
        'priority': 2,
        'region': 'IN',
        'industry': 'Kollywood'
    },
    'malayalam': {
        'codes': ['ml', 'malayalam', 'mollywood'],
        'priority': 2,
        'region': 'IN',
        'industry': 'Mollywood'
    },
    'kannada': {
        'codes': ['kn', 'kannada', 'sandalwood'],
        'priority': 2,
        'region': 'IN',
        'industry': 'Sandalwood'
    }
}

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

def get_user_location(ip_address):
    try:
        response = api_session.get(f'http://ip-api.com/json/{ip_address}', timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return {
                    'country': data.get('country'),
                    'region': data.get('regionName'),
                    'city': data.get('city'),
                    'lat': data.get('lat'),
                    'lon': data.get('lon')
                }
    except:
        pass
    return None

# Enhanced Streaming Availability Service with better error handling
class StreamingAvailabilityService:
    @staticmethod
    def get_streaming_availability(title, tmdb_id=None, imdb_id=None):
        """Get streaming availability with timeout protection"""
        try:
            streaming_data = {
                'free_platforms': [],
                'paid_platforms': [],
                'language_specific_links': {}
            }
            
            # Use async approach to prevent timeouts
            with concurrent.futures.ThreadPoolExecutor(max_workers=2, timeout=5) as executor:
                futures = []
                
                # Only try external APIs if we have proper keys
                if WATCHMODE_API_KEY and WATCHMODE_API_KEY != 'your_watchmode_api_key':
                    futures.append(executor.submit(StreamingAvailabilityService._get_watchmode_data_safe, title, tmdb_id))
                
                # Always add fallback data
                futures.append(executor.submit(StreamingAvailabilityService._get_fallback_streaming_data, title))
                
                # Collect results with timeout
                for future in concurrent.futures.as_completed(futures, timeout=5):
                    try:
                        result = future.result()
                        if result:
                            streaming_data['free_platforms'].extend(result.get('free_platforms', []))
                            streaming_data['paid_platforms'].extend(result.get('paid_platforms', []))
                            streaming_data['language_specific_links'].update(result.get('language_specific_links', {}))
                    except:
                        continue
            
            # Remove duplicates
            streaming_data['free_platforms'] = list({p['platform']: p for p in streaming_data['free_platforms']}.values())
            streaming_data['paid_platforms'] = list({p['platform']: p for p in streaming_data['paid_platforms']}.values())
            
            return streaming_data
            
        except Exception as e:
            logger.error(f"Streaming availability error: {e}")
            # Return basic fallback data
            return StreamingAvailabilityService._get_fallback_streaming_data(title) or {
                'free_platforms': [],
                'paid_platforms': [],
                'language_specific_links': {}
            }
    
    @staticmethod
    def _get_watchmode_data_safe(title, tmdb_id=None):
        """Safe WatchMode API call with timeout"""
        try:
            headers = {'X-API-Key': WATCHMODE_API_KEY}
            search_url = 'https://api.watchmode.com/v1/search/'
            params = {'search_field': 'name', 'search_value': title}
            
            response = api_session.get(search_url, headers=headers, params=params, timeout=3)
            if response.status_code == 200:
                search_results = response.json()
                if search_results.get('title_results'):
                    title_id = search_results['title_results'][0]['id']
                    sources_url = f'https://api.watchmode.com/v1/title/{title_id}/sources/'
                    sources_response = api_session.get(sources_url, headers=headers, timeout=3)
                    
                    if sources_response.status_code == 200:
                        sources_data = sources_response.json()
                        return StreamingAvailabilityService._process_watchmode_sources(sources_data)
        except Exception as e:
            logger.error(f"WatchMode API error: {e}")
        return None
    
    @staticmethod
    def _get_fallback_streaming_data(title):
        """Fallback streaming data - always works"""
        try:
            streaming_data = {
                'free_platforms': [],
                'paid_platforms': [],
                'language_specific_links': {}
            }
            
            # Detect language from title
            detected_languages = StreamingAvailabilityService._detect_languages_from_title(title)
            
            # Add common free platforms
            free_platforms = ['mx_player', 'youtube', 'jio_hotstar']
            for platform in free_platforms:
                platform_info = OTT_PLATFORMS.get(platform, {})
                streaming_data['free_platforms'].append({
                    'platform': platform,
                    'name': platform_info.get('name', platform),
                    'url': platform_info.get('url', ''),
                    'is_free': True,
                    'quality': 'HD'
                })
            
            # Add common paid platforms
            paid_platforms = ['netflix', 'prime_video', 'disney_plus_hotstar']
            for platform in paid_platforms:
                platform_info = OTT_PLATFORMS.get(platform, {})
                streaming_data['paid_platforms'].append({
                    'platform': platform,
                    'name': platform_info.get('name', platform),
                    'url': platform_info.get('url', ''),
                    'is_free': False,
                    'quality': '4K'
                })
            
            # Add language-specific links
            for language in detected_languages:
                streaming_data['language_specific_links'][language] = {
                    'free_links': [
                        {
                            'platform': 'mx_player',
                            'name': f'🔘 Watch in {language.title()}',
                            'url': f"{OTT_PLATFORMS['mx_player']['base_watch_url']}{title.lower().replace(' ', '-')}-{language}",
                            'quality': 'HD'
                        },
                        {
                            'platform': 'youtube',
                            'name': f'🔘 Watch in {language.title()} (Free)',
                            'url': f"https://youtube.com/results?search_query={title}+{language}+full+movie",
                            'quality': 'HD'
                        }
                    ],
                    'paid_links': [
                        {
                            'platform': 'netflix',
                            'name': f'🔘 Watch in {language.title()} (Netflix)',
                            'url': f"https://netflix.com/search?q={title}",
                            'quality': '4K'
                        },
                        {
                            'platform': 'prime_video',
                            'name': f'🔘 Watch in {language.title()} (Prime)',
                            'url': f"https://primevideo.com/search?phrase={title}",
                            'quality': '4K'
                        }
                    ]
                }
            
            return streaming_data
            
        except Exception as e:
            logger.error(f"Fallback streaming data error: {e}")
            return {
                'free_platforms': [],
                'paid_platforms': [],
                'language_specific_links': {}
            }
    
    @staticmethod
    def _detect_languages_from_title(title):
        """Detect possible languages from title"""
        detected = []
        title_lower = title.lower()
        
        # Check for language indicators in title
        for language, info in REGIONAL_LANGUAGES.items():
            for code in info['codes']:
                if code in title_lower:
                    detected.append(language)
                    break
        
        # Default to major Indian languages if none detected
        if not detected:
            detected = ['hindi', 'english', 'telugu', 'tamil']
        
        return detected
    
    @staticmethod
    def _process_watchmode_sources(sources_data):
        """Process WatchMode API sources"""
        streaming_data = {
            'free_platforms': [],
            'paid_platforms': [],
            'language_specific_links': {}
        }
        
        for source in sources_data:
            platform_info = {
                'platform': source.get('name', '').lower().replace(' ', '_'),
                'name': source.get('name', ''),
                'url': source.get('web_url', ''),
                'is_free': source.get('type') == 'free',
                'quality': source.get('format', 'HD')
            }
            
            if platform_info['is_free']:
                streaming_data['free_platforms'].append(platform_info)
            else:
                streaming_data['paid_platforms'].append(platform_info)
        
        return streaming_data

# External API Services with better error handling
class TMDBService:
    BASE_URL = 'https://api.themoviedb.org/3'
    
    @staticmethod
    def search_content(query, content_type='multi', language='en-US', page=1, region=None):
        url = f"{TMDBService.BASE_URL}/search/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'language': language,
            'page': page
        }
        if region:
            params['region'] = region
        
        try:
            response = api_session.get(url, params=params, timeout=5)
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
            'append_to_response': 'credits,videos,similar,watch/providers,external_ids'
        }
        
        try:
            response = api_session.get(url, params=params, timeout=5)
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
            response = api_session.get(url, params=params, timeout=5)
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
            response = api_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB popular error: {e}")
        return None
    
    @staticmethod
    def get_top_rated(content_type='movie', page=1, region=None):
        url = f"{TMDBService.BASE_URL}/{content_type}/top_rated"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        if region:
            params['region'] = region
        
        try:
            response = api_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB top rated error: {e}")
        return None
    
    @staticmethod
    def get_now_playing(page=1, region=None):
        url = f"{TMDBService.BASE_URL}/movie/now_playing"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        if region:
            params['region'] = region
        
        try:
            response = api_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB now playing error: {e}")
        return None
    
    @staticmethod
    def discover_content(content_type='movie', **kwargs):
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            **kwargs
        }
        
        try:
            response = api_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB discover error: {e}")
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
            response = api_session.get(OMDbService.BASE_URL, params=params, timeout=5)
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
            response = api_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan search error: {e}")
        return None
    
    @staticmethod
    def get_anime_details(anime_id):
        url = f"{JikanService.BASE_URL}/anime/{anime_id}/full"
        
        try:
            response = api_session.get(url, params={}, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan anime details error: {e}")
        return None
    
    @staticmethod
    def get_top_anime(type='tv', page=1, filter=''):
        url = f"{JikanService.BASE_URL}/top/anime"
        params = {
            'type': type,
            'page': page
        }
        if filter:
            params['filter'] = filter
        
        try:
            response = api_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan top anime error: {e}")
        return None

class YouTubeService:
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    @staticmethod
    def search_trailers(query):
        if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == 'your_youtube_api_key':
            return None
            
        url = f"{YouTubeService.BASE_URL}/search"
        params = {
            'key': YOUTUBE_API_KEY,
            'q': f"{query} trailer",
            'part': 'snippet',
            'type': 'video',
            'maxResults': 5
        }
        
        try:
            response = api_session.get(url, params=params, timeout=3)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
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
            
            # Get streaming availability with fallback
            title = tmdb_data.get('title') or tmdb_data.get('name')
            imdb_id = tmdb_data.get('external_ids', {}).get('imdb_id') if 'external_ids' in tmdb_data else None
            
            try:
                streaming_data = StreamingAvailabilityService.get_streaming_availability(title, tmdb_data['id'], imdb_id)
            except:
                # Fallback streaming data
                streaming_data = {
                    'free_platforms': [{'platform': 'youtube', 'name': 'YouTube', 'url': 'https://youtube.com', 'is_free': True}],
                    'paid_platforms': [{'platform': 'netflix', 'name': 'Netflix', 'url': 'https://netflix.com', 'is_free': False}],
                    'language_specific_links': {}
                }
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
                imdb_id=imdb_id,
                title=title,
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
                ott_platforms=json.dumps(streaming_data.get('free_platforms', []) + streaming_data.get('paid_platforms', [])),
                streaming_links=json.dumps(streaming_data)
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def save_anime_content(anime_data):
        try:
            # Check if anime already exists
            existing = Content.query.filter_by(mal_id=anime_data['mal_id']).first()
            if existing:
                return existing
            
            # Extract genres
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            # Get streaming availability for anime (without external API calls to prevent timeouts)
            streaming_data = {
                'free_platforms': [
                    {'platform': 'crunchyroll_free', 'name': 'Crunchyroll (Free)', 'url': 'https://crunchyroll.com', 'is_free': True},
                    {'platform': 'youtube', 'name': 'YouTube', 'url': 'https://youtube.com', 'is_free': True}
                ],
                'paid_platforms': [
                    {'platform': 'crunchyroll', 'name': 'Crunchyroll Premium', 'url': 'https://crunchyroll.com', 'is_free': False}
                ],
                'language_specific_links': {
                    'japanese': {
                        'free_links': [
                            {'platform': 'crunchyroll_free', 'name': '🔘 Watch in Japanese (Sub)', 'url': f"https://crunchyroll.com/search?q={anime_data.get('title')}", 'quality': 'HD'}
                        ]
                    }
                }
            }
            
            # Create content object
            content = Content(
                mal_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                languages=json.dumps(['japanese']),
                release_date=datetime.strptime(anime_data.get('aired', {}).get('from', '1900-01-01T00:00:00+00:00')[:10], '%Y-%m-%d').date() if anime_data.get('aired', {}).get('from') else None,
                runtime=anime_data.get('duration', 24),  # Default anime episode duration
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                backdrop_path=anime_data.get('images', {}).get('jpg', {}).get('large_image_url'),
                ott_platforms=json.dumps(streaming_data.get('free_platforms', []) + streaming_data.get('paid_platforms', [])),
                streaming_links=json.dumps(streaming_data)
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def map_genre_ids(genre_ids):
        # TMDB Genre ID mapping
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western',
            10759: 'Action & Adventure', 10762: 'Kids', 10763: 'News',
            10764: 'Reality', 10765: 'Sci-Fi & Fantasy', 10766: 'Soap',
            10767: 'Talk', 10768: 'War & Politics'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]

# Enhanced Recommendation Engine
class RecommendationEngine:
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all'):
        """Get current trending content with Telugu and English priority"""
        try:
            # Get trending from TMDB
            trending_data = TMDBService.get_trending(content_type=content_type)
            if not trending_data:
                return []
            
            recommendations = []
            priority_items = []
            other_items = []
            
            for item in trending_data.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    # Check if it's Telugu or English content
                    languages = json.loads(content.languages or '[]')
                    is_priority = any(lang.lower() in ['telugu', 'english', 'te', 'en'] for lang in languages)
                    
                    if is_priority:
                        priority_items.append(content)
                    else:
                        other_items.append(content)
            
            # Combine with priority first
            recommendations = priority_items + other_items
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_popular_by_genre(genre, limit=20, language=None):
        """Get popular content by genre"""
        try:
            # Genre mapping
            genre_map = {
                'action': 28, 'adventure': 12, 'animation': 16, 'biography': 36,
                'comedy': 35, 'crime': 80, 'documentary': 99, 'drama': 18,
                'fantasy': 14, 'horror': 27, 'musical': 10402, 'mystery': 9648,
                'romance': 10749, 'sci-fi': 878, 'thriller': 53, 'western': 37
            }
            
            genre_id = genre_map.get(genre.lower())
            if not genre_id:
                return []
            
            discover_params = {
                'with_genres': genre_id,
                'sort_by': 'popularity.desc',
                'vote_count.gte': 50
            }
            
            # Add language filter
            if language and language in REGIONAL_LANGUAGES:
                discover_params['region'] = REGIONAL_LANGUAGES[language]['region']
                discover_params['with_original_language'] = REGIONAL_LANGUAGES[language]['codes'][0]
            
            genre_data = TMDBService.discover_content('movie', **discover_params)
            
            recommendations = []
            if genre_data:
                for item in genre_data.get('results', [])[:limit]:
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting popular by genre: {e}")
            return []
    
    @staticmethod
    def get_new_releases(language=None, limit=20):
        """Get new releases from last 30-60 days"""
        try:
            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=60)
            
            # Discover new releases
            discover_params = {
                'primary_release_date.gte': start_date.isoformat(),
                'primary_release_date.lte': end_date.isoformat(),
                'sort_by': 'release_date.desc',
                'vote_count.gte': 10
            }
            
            # Add region filter for Indian languages
            if language and language in ['hindi', 'telugu', 'tamil', 'malayalam', 'kannada']:
                discover_params['region'] = 'IN'
                discover_params['with_original_language'] = REGIONAL_LANGUAGES[language]['codes'][0]
            
            new_releases_data = TMDBService.discover_content('movie', **discover_params)
            
            recommendations = []
            if new_releases_data:
                for item in new_releases_data.get('results', [])[:limit]:
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return []
    
    @staticmethod
    def get_best_movies(limit=20):
        """Get all-time great movies"""
        try:
            # Get top rated movies
            top_rated_data = TMDBService.get_top_rated('movie')
            
            recommendations = []
            if top_rated_data:
                for item in top_rated_data.get('results', [])[:limit]:
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting best movies: {e}")
            return []
    
    @staticmethod
    def get_critics_choice(limit=20):
        """Get critically acclaimed titles"""
        try:
            # Get high-rated content with significant vote count
            discover_params = {
                'vote_average.gte': 7.5,
                'vote_count.gte': 1000,
                'sort_by': 'vote_average.desc'
            }
            
            critics_data = TMDBService.discover_content('movie', **discover_params)
            
            recommendations = []
            if critics_data:
                for item in critics_data.get('results', [])[:limit]:
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting critics choice: {e}")
            return []
    
    @staticmethod
    def get_regional_recommendations(language, limit=20):
        try:
            if language not in REGIONAL_LANGUAGES:
                return []
            
            lang_info = REGIONAL_LANGUAGES[language]
            
            recommendations = []
            
            # Use discover endpoint for better results
            discover_params = {
                'region': lang_info['region'],
                'sort_by': 'popularity.desc',
                'vote_count.gte': 10
            }
            
            if lang_info['codes'][0] != 'en':
                discover_params['with_original_language'] = lang_info['codes'][0]
            
            discover_data = TMDBService.discover_content('movie', **discover_params)
            
            if discover_data:
                for item in discover_data.get('results', []):
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
                    
                    if len(recommendations) >= limit:
                        break
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting regional recommendations: {e}")
            return []
    
    @staticmethod
    def get_anime_recommendations(limit=20):
        try:
            top_anime = JikanService.get_top_anime()
            if not top_anime:
                return []
            
            recommendations = []
            for anime in top_anime.get('data', [])[:limit]:
                content = ContentService.save_anime_content(anime)
                if content:
                    recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting anime recommendations: {e}")
            return []

# Anonymous User Recommendations
class AnonymousRecommendationEngine:
    @staticmethod
    def get_recommendations_for_anonymous(session_id, ip_address, limit=20):
        try:
            recommendations = []
            
            # Priority: Telugu and English content
            priority_languages = ['telugu', 'english']
            
            # Add trending Telugu and English content
            trending_recs = RecommendationEngine.get_trending_recommendations(limit=10)
            recommendations.extend(trending_recs)
            
            # Add regional content
            for language in priority_languages:
                regional_recs = RecommendationEngine.get_regional_recommendations(language, limit=5)
                recommendations.extend(regional_recs)
            
            # Remove duplicates and limit
            seen_ids = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec and rec.id not in seen_ids:
                    seen_ids.add(rec.id)
                    unique_recommendations.append(rec)
                    if len(unique_recommendations) >= limit:
                        break
            
            return unique_recommendations
        except Exception as e:
            logger.error(f"Error getting anonymous recommendations: {e}")
            return []

# Enhanced Telegram Service
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
            
            # Get streaming links
            streaming_data = {}
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                except:
                    streaming_data = {}
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Format streaming links by language
            streaming_text = TelegramService._format_streaming_links(streaming_data)
            
            # Create message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

{streaming_text}

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
    
    @staticmethod
    def _format_streaming_links(streaming_data):
        """Format streaming links for Telegram message"""
        if not streaming_data:
            return ""
        
        streaming_text = "\n🎥 **Watch Options:**\n"
        
        # Language-specific links
        language_links = streaming_data.get('language_specific_links', {})
        if language_links:
            for language, links in language_links.items():
                streaming_text += f"\n**{language.title()} Audio:**\n"
                
                # Free links
                free_links = links.get('free_links', [])
                for link in free_links[:2]:  # Limit to 2 free links per language
                    streaming_text += f"{link['name']} - [Watch Free]({link['url']})\n"
                
                # Paid links
                paid_links = links.get('paid_links', [])
                for link in paid_links[:2]:  # Limit to 2 paid links per language
                    streaming_text += f"{link['name']} - [Watch]({link['url']})\n"
        
        return streaming_text

# API Routes

# Authentication Routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validate input
        if not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if user exists
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        # Create user
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password']),
            preferred_languages=json.dumps(data.get('preferred_languages', ['telugu', 'english'])),
            preferred_genres=json.dumps(data.get('preferred_genres', []))
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Generate token
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
        
        # Update last active
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        # Generate token
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
                'preferred_languages': json.loads(user.preferred_languages or '["telugu", "english"]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

# Content Discovery Routes
@app.route('/api/search', methods=['GET'])
def search_content():
    try:
        query = request.args.get('query', '')
        content_type = request.args.get('type', 'multi')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        # Record search interaction
        session_id = get_session_id()
        
        # Search TMDB
        tmdb_results = TMDBService.search_content(query, content_type, page=page)
        
        # Search anime if content_type is anime or multi
        anime_results = None
        if content_type in ['anime', 'multi']:
            anime_results = JikanService.search_anime(query, page=page)
        
        # Process and save results
        results = []
        
        if tmdb_results:
            for item in tmdb_results.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    # Record anonymous interaction
                    interaction = AnonymousInteraction(
                        session_id=session_id,
                        content_id=content.id,
                        interaction_type='search',
                        ip_address=request.remote_addr
                    )
                    db.session.add(interaction)
                    
                    # Get streaming data
                    streaming_data = {}
                    if content.streaming_links:
                        try:
                            streaming_data = json.loads(content.streaming_links)
                        except:
                            streaming_data = {}
                    
                    results.append({
                        'id': content.id,
                        'tmdb_id': content.tmdb_id,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
                        'overview': content.overview,
                        'ott_platforms': json.loads(content.ott_platforms or '[]'),
                        'streaming_links': streaming_data
                    })
        
        # Add anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                # Save anime content
                anime_content = ContentService.save_anime_content(anime)
                if anime_content:
                    streaming_data = {}
                    if anime_content.streaming_links:
                        try:
                            streaming_data = json.loads(anime_content.streaming_links)
                        except:
                            streaming_data = {}
                    
                    results.append({
                        'id': anime_content.id,
                        'mal_id': anime['mal_id'],
                        'title': anime.get('title'),
                        'content_type': 'anime',
                        'genres': [genre['name'] for genre in anime.get('genres', [])],
                        'rating': anime.get('score'),
                        'release_date': anime.get('aired', {}).get('from'),
                        'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                        'overview': anime.get('synopsis'),
                        'ott_platforms': json.loads(anime_content.ott_platforms or '[]'),
                        'streaming_links': streaming_data
                    })
        
        db.session.commit()
        
        return jsonify({
            'results': results,
            'total_results': tmdb_results.get('total_results', 0) if tmdb_results else 0,
            'total_pages': tmdb_results.get('total_pages', 0) if tmdb_results else 0,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

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
        
        # Get additional details from TMDB if available
        additional_details = None
        if content.tmdb_id:
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        
        # Get YouTube trailers
        trailers = []
        youtube_results = YouTubeService.search_trailers(content.title)
        if youtube_results:
            for video in youtube_results.get('items', []):
                trailers.append({
                    'title': video['snippet']['title'],
                    'url': f"https://www.youtube.com/watch?v={video['id']['videoId']}",
                    'thumbnail': video['snippet']['thumbnails']['medium']['url']
                })
        
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
        
        # Get streaming data
        streaming_data = {}
        if content.streaming_links:
            try:
                streaming_data = json.loads(content.streaming_links)
            except:
                streaming_data = {}
        
        db.session.commit()
        
        return jsonify({
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'mal_id': content.mal_id,
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
            'ott_platforms': json.loads(content.ott_platforms or '[]'),
            'streaming_links': streaming_data,
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Fixed Anime Details Route
@app.route('/api/anime/<int:anime_id>', methods=['GET'])
def get_anime_details(anime_id):
    try:
        # Check if anime exists in our database
        anime_content = Content.query.filter_by(mal_id=anime_id).first()
        
        if not anime_content:
            # Fetch from Jikan API
            anime_data = JikanService.get_anime_details(anime_id)
            if not anime_data:
                return jsonify({'error': 'Anime not found'}), 404
            
            anime_info = anime_data.get('data', {})
            if not anime_info:
                return jsonify({'error': 'Anime data not available'}), 404
            
            # Save to database
            anime_content = ContentService.save_anime_content(anime_info)
            if not anime_content:
                return jsonify({'error': 'Failed to save anime data'}), 500
        
        # Record view interaction
        session_id = get_session_id()
        interaction = AnonymousInteraction(
            session_id=session_id,
            content_id=anime_content.id,
            interaction_type='view',
            ip_address=request.remote_addr
        )
        db.session.add(interaction)
        
        # Get streaming data
        streaming_data = {}
        if anime_content.streaming_links:
            try:
                streaming_data = json.loads(anime_content.streaming_links)
            except:
                streaming_data = {}
        
        # Get trailers for anime
        trailers = []
        youtube_results = YouTubeService.search_trailers(f"{anime_content.title} anime trailer")
        if youtube_results:
            for video in youtube_results.get('items', []):
                trailers.append({
                    'title': video['snippet']['title'],
                    'url': f"https://www.youtube.com/watch?v={video['id']['videoId']}",
                    'thumbnail': video['snippet']['thumbnails']['medium']['url']
                })
        
        db.session.commit()
        
        return jsonify({
            'id': anime_content.id,
            'mal_id': anime_content.mal_id,
            'title': anime_content.title,
            'original_title': anime_content.original_title,
            'content_type': anime_content.content_type,
            'genres': json.loads(anime_content.genres or '[]'),
            'languages': json.loads(anime_content.languages or '[]'),
            'release_date': anime_content.release_date.isoformat() if anime_content.release_date else None,
            'runtime': anime_content.runtime,
            'rating': anime_content.rating,
            'vote_count': anime_content.vote_count,
            'overview': anime_content.overview,
            'poster_path': anime_content.poster_path,
            'backdrop_path': anime_content.backdrop_path,
            'ott_platforms': json.loads(anime_content.ott_platforms or '[]'),
            'streaming_links': streaming_data,
            'trailers': trailers
        }), 200
        
    except Exception as e:
        logger.error(f"Anime details error: {e}")
        return jsonify({'error': 'Failed to get anime details'}), 500

def format_content_response(content):
    """Helper function to format content response consistently"""
    streaming_data = {}
    if content.streaming_links:
        try:
            streaming_data = json.loads(content.streaming_links)
        except:
            streaming_data = {}
    
    return {
        'id': content.id,
        'title': content.title,
        'content_type': content.content_type,
        'genres': json.loads(content.genres or '[]'),
        'rating': content.rating,
        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
        'overview': content.overview[:150] + '...' if content.overview else '',
        'ott_platforms': json.loads(content.ott_platforms or '[]'),
        'streaming_links': streaming_data
    }

# Enhanced Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = []
        for content in recommendations:
            result.append(format_content_response(content))
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

# FIXED: Popular genre routes that were missing
@app.route('/api/recommendations/popular/<genre>', methods=['GET'])
def get_popular_by_genre(genre):
    try:
        limit = int(request.args.get('limit', 20))
        language = request.args.get('language')
        
        recommendations = RecommendationEngine.get_popular_by_genre(genre, limit, language)
        
        result = []
        for content in recommendations:
            result.append(format_content_response(content))
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Popular by genre error: {e}")
        return jsonify({'error': 'Failed to get popular recommendations'}), 500

@app.route('/api/recommendations/new-releases', methods=['GET'])
def get_new_releases():
    try:
        language = request.args.get('language')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_new_releases(language, limit)
        
        result = []
        for content in recommendations:
            result.append(format_content_response(content))
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"New releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

@app.route('/api/recommendations/best-movies', methods=['GET'])
def get_best_movies():
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_best_movies(limit)
        
        result = []
        for content in recommendations:
            result.append(format_content_response(content))
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Best movies error: {e}")
        return jsonify({'error': 'Failed to get best movies'}), 500

@app.route('/api/recommendations/critics-choice', methods=['GET'])
def get_critics_choice():
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_critics_choice(limit)
        
        result = []
        for content in recommendations:
            result.append(format_content_response(content))
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Critics choice error: {e}")
        return jsonify({'error': 'Failed to get critics choice'}), 500

@app.route('/api/recommendations/genre/<genre>', methods=['GET'])
def get_genre_recommendations(genre):
    try:
        limit = int(request.args.get('limit', 20))
        language = request.args.get('language')
        
        recommendations = RecommendationEngine.get_popular_by_genre(genre, limit, language)
        
        result = []
        for content in recommendations:
            result.append(format_content_response(content))
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Genre recommendations error: {e}")
        return jsonify({'error': 'Failed to get genre recommendations'}), 500

@app.route('/api/recommendations/regional/<language>', methods=['GET'])
def get_regional(language):
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_regional_recommendations(language, limit)
        
        result = []
        for content in recommendations:
            result.append(format_content_response(content))
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Regional recommendations error: {e}")
        return jsonify({'error': 'Failed to get regional recommendations'}), 500

@app.route('/api/recommendations/anime', methods=['GET'])
def get_anime():
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_anime_recommendations(limit)
        
        result = []
        for content in recommendations:
            result.append(format_content_response(content))
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Anime recommendations error: {e}")
        return jsonify({'error': 'Failed to get anime recommendations'}), 500

@app.route('/api/recommendations/anonymous', methods=['GET'])
def get_anonymous_recommendations():
    try:
        session_id = get_session_id()
        limit = int(request.args.get('limit', 20))
        
        recommendations = AnonymousRecommendationEngine.get_recommendations_for_anonymous(
            session_id, request.remote_addr, limit
        )
        
        result = []
        for content in recommendations:
            result.append(format_content_response(content))
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Anonymous recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

# Personalized Recommendations (requires ML service)
@app.route('/api/recommendations/personalized', methods=['GET'])
@require_auth
def get_personalized_recommendations(current_user):
    try:
        # Get user interactions
        interactions = UserInteraction.query.filter_by(user_id=current_user.id).all()
        
        # Prepare data for ML service
        user_data = {
            'user_id': current_user.id,
            'preferred_languages': json.loads(current_user.preferred_languages or '["telugu", "english"]'),
            'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
            'interactions': [
                {
                    'content_id': interaction.content_id,
                    'interaction_type': interaction.interaction_type,
                    'rating': interaction.rating,
                    'timestamp': interaction.timestamp.isoformat()
                }
                for interaction in interactions
            ]
        }
        
        # Call ML service
        try:
            response = api_session.post(f"{ML_SERVICE_URL}/api/recommendations", json=user_data, timeout=5)
            
            if response.status_code == 200:
                ml_recommendations = response.json().get('recommendations', [])
                
                # Get content details for recommended content IDs
                content_ids = [rec['content_id'] for rec in ml_recommendations]
                contents = Content.query.filter(Content.id.in_(content_ids)).all()
                
                # Create response with ML scores
                result = []
                content_dict = {content.id: content for content in contents}
                
                for rec in ml_recommendations:
                    content = content_dict.get(rec['content_id'])
                    if content:
                        content_data = format_content_response(content)
                        content_data['recommendation_score'] = rec.get('score', 0)
                        content_data['recommendation_reason'] = rec.get('reason', '')
                        result.append(content_data)
                
                return jsonify({'recommendations': result}), 200
        except:
            pass
        
        # Fallback to basic recommendations with user preferences
        preferred_languages = json.loads(current_user.preferred_languages or '["telugu", "english"]')
        fallback_recs = []
        
        for language in preferred_languages[:2]:  # Top 2 preferred languages
            regional_recs = RecommendationEngine.get_regional_recommendations(language, limit=10)
            fallback_recs.extend(regional_recs)
        
        # Add trending content
        trending_recs = RecommendationEngine.get_trending_recommendations(limit=10)
        fallback_recs.extend(trending_recs)
        
        # Format response
        result = []
        seen_ids = set()
        for content in fallback_recs:
            if content and content.id not in seen_ids:
                seen_ids.add(content.id)
                result.append(format_content_response(content))
            
            if len(result) >= 20:
                break
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Personalized recommendations error: {e}")
        return get_trending()

# User Interaction Routes
@app.route('/api/interactions', methods=['POST'])
@require_auth
def record_interaction(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=data.get('rating')
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        return jsonify({'message': 'Interaction recorded successfully'}), 201
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

@app.route('/api/user/watchlist', methods=['GET'])
@require_auth
def get_watchlist(current_user):
    try:
        watchlist_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        ).all()
        
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        
        result = []
        for content in contents:
            result.append(format_content_response(content))
        
        return jsonify({'watchlist': result}), 200
        
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return jsonify({'error': 'Failed to get watchlist'}), 500

@app.route('/api/user/favorites', methods=['GET'])
@require_auth
def get_favorites(current_user):
    try:
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).all()
        
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        
        result = []
        for content in contents:
            result.append(format_content_response(content))
        
        return jsonify({'favorites': result}), 200
        
    except Exception as e:
        logger.error(f"Favorites error: {e}")
        return jsonify({'error': 'Failed to get favorites'}), 500

# Admin Routes
@app.route('/api/admin/search', methods=['GET'])
@require_admin
def admin_search(current_user):
    try:
        query = request.args.get('query', '')
        source = request.args.get('source', 'tmdb')  # tmdb, omdb, anime
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        results = []
        
        if source == 'tmdb':
            tmdb_results = TMDBService.search_content(query, page=page)
            if tmdb_results:
                for item in tmdb_results.get('results', []):
                    results.append({
                        'id': item['id'],
                        'title': item.get('title') or item.get('name'),
                        'content_type': 'movie' if 'title' in item else 'tv',
                        'release_date': item.get('release_date') or item.get('first_air_date'),
                        'poster_path': f"https://image.tmdb.org/t/p/w300{item['poster_path']}" if item.get('poster_path') else None,
                        'overview': item.get('overview'),
                        'rating': item.get('vote_average'),
                        'source': 'tmdb'
                    })
        
        elif source == 'anime':
            anime_results = JikanService.search_anime(query, page=page)
            if anime_results:
                for anime in anime_results.get('data', []):
                    results.append({
                        'id': anime['mal_id'],
                        'title': anime.get('title'),
                        'content_type': 'anime',
                        'release_date': anime.get('aired', {}).get('from'),
                        'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                        'overview': anime.get('synopsis'),
                        'rating': anime.get('score'),
                        'source': 'anime'
                    })
        
        return jsonify({'results': results}), 200
        
    except Exception as e:
        logger.error(f"Admin search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/admin/content', methods=['POST'])
@require_admin
def save_external_content(current_user):
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No content data provided'}), 400
        
        # Check if content already exists by external ID
        existing_content = None
        if data.get('source') == 'anime' and data.get('id'):
            existing_content = Content.query.filter_by(mal_id=data['id']).first()
        elif data.get('id'):
            existing_content = Content.query.filter_by(tmdb_id=data['id']).first()
        
        if existing_content:
            return jsonify({
                'message': 'Content already exists',
                'content_id': existing_content.id
            }), 200
        
        # Create new content from external data
        try:
            # Handle release date
            release_date = None
            if data.get('release_date'):
                try:
                    release_date = datetime.strptime(data['release_date'][:10], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            # Get streaming availability
            try:
                streaming_data = StreamingAvailabilityService.get_streaming_availability(data.get('title'))
            except:
                streaming_data = {'free_platforms': [], 'paid_platforms': [], 'language_specific_links': {}}
            
            # Create content object
            if data.get('source') == 'anime':
                content = Content(
                    mal_id=data.get('id'),
                    title=data.get('title'),
                    original_title=data.get('original_title'),
                    content_type='anime',
                    genres=json.dumps(data.get('genres', [])),
                    languages=json.dumps(['japanese']),
                    release_date=release_date,
                    runtime=data.get('runtime', 24),
                    rating=data.get('rating'),
                    vote_count=data.get('vote_count'),
                    popularity=data.get('popularity'),
                    overview=data.get('overview'),
                    poster_path=data.get('poster_path'),
                    backdrop_path=data.get('backdrop_path'),
                    ott_platforms=json.dumps(streaming_data.get('free_platforms', []) + streaming_data.get('paid_platforms', [])),
                    streaming_links=json.dumps(streaming_data)
                )
            else:
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
                    ott_platforms=json.dumps(streaming_data.get('free_platforms', []) + streaming_data.get('paid_platforms', [])),
                    streaming_links=json.dumps(streaming_data)
                )
            
            db.session.add(content)
            db.session.commit()
            
            return jsonify({
                'message': 'Content saved successfully',
                'content_id': content.id
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
        
        # Get content - handle both internal ID and external ID
        content = Content.query.get(data['content_id'])
        if not content:
            # Try to find by TMDB ID if direct ID lookup fails
            content = Content.query.filter_by(tmdb_id=data['content_id']).first()
            if not content:
                # Try to find by MAL ID for anime
                content = Content.query.filter_by(mal_id=data['content_id']).first()
        
        if not content:
            return jsonify({'error': 'Content not found. Please save content first.'}), 404
        
        # Create admin recommendation
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

@app.route('/api/admin/recommendations', methods=['GET'])
@require_admin
def get_admin_recommendations(current_user):
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        admin_recs = AdminRecommendation.query.filter_by(is_active=True)\
            .order_by(AdminRecommendation.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        result = []
        for rec in admin_recs.items:
            content = Content.query.get(rec.content_id)
            admin = User.query.get(rec.admin_id)
            
            result.append({
                'id': rec.id,
                'recommendation_type': rec.recommendation_type,
                'description': rec.description,
                'created_at': rec.created_at.isoformat(),
                'admin_name': admin.username if admin else 'Unknown',
                'content': {
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None
                }
            })
        
        return jsonify({
            'recommendations': result,
            'total': admin_recs.total,
            'pages': admin_recs.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Get admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@app.route('/api/admin/analytics', methods=['GET'])
@require_admin
def get_analytics(current_user):
    try:
        # Get basic analytics
        total_users = User.query.count()
        total_content = Content.query.count()
        total_interactions = UserInteraction.query.count()
        active_users_last_week = User.query.filter(
            User.last_active >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        # Popular content
        popular_content = db.session.query(
            Content.id, Content.title, func.count(UserInteraction.id).label('interaction_count')
        ).join(UserInteraction).group_by(Content.id, Content.title)\
         .order_by(desc('interaction_count')).limit(10).all()
        
        # Popular genres
        all_interactions = UserInteraction.query.join(Content).all()
        genre_counts = defaultdict(int)
        for interaction in all_interactions:
            content = Content.query.get(interaction.content_id)
            if content and content.genres:
                genres = json.loads(content.genres)
                for genre in genres:
                    genre_counts[genre] += 1
        
        popular_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Language analytics
        language_counts = defaultdict(int)
        all_content = Content.query.all()
        for content in all_content:
            if content.languages:
                languages = json.loads(content.languages)
                for language in languages:
                    language_counts[language] += 1
        
        popular_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return jsonify({
            'total_users': total_users,
            'total_content': total_content,
            'total_interactions': total_interactions,
            'active_users_last_week': active_users_last_week,
            'popular_content': [
                {'title': item.title, 'interactions': item.interaction_count}
                for item in popular_content
            ],
            'popular_genres': [
                {'genre': genre, 'count': count}
                for genre, count in popular_genres
            ],
            'popular_languages': [
                {'language': language, 'count': count}
                for language, count in popular_languages
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

# Public Admin Recommendations
@app.route('/api/recommendations/admin-choice', methods=['GET'])
def get_public_admin_recommendations():
    try:
        limit = int(request.args.get('limit', 20))
        rec_type = request.args.get('type', 'admin_choice')
        
        admin_recs = AdminRecommendation.query.filter_by(
            is_active=True,
            recommendation_type=rec_type
        ).order_by(AdminRecommendation.created_at.desc()).limit(limit).all()
        
        result = []
        for rec in admin_recs:
            content = Content.query.get(rec.content_id)
            admin = User.query.get(rec.admin_id)
            
            if content:
                content_data = format_content_response(content)
                content_data['admin_description'] = rec.description
                content_data['admin_name'] = admin.username if admin else 'Admin'
                content_data['recommended_at'] = rec.created_at.isoformat()
                result.append(content_data)
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Public admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get admin recommendations'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.1.0',
        'features': [
            'streaming_availability',
            'multi_language_support',
            'real_time_recommendations',
            'telugu_english_priority',
            'anime_support',
            'telegram_integration',
            'timeout_protection',
            'error_handling'
        ],
        'fixes': [
            'worker_timeout_protection',
            'missing_popular_routes_added',
            'improved_error_handling',
            'better_api_timeouts',
            'anime_details_fixed'
        ]
    }), 200

# Add root route
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Movie Recommendation API v2.1.0',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'recommendations': '/api/recommendations/*',
            'search': '/api/search',
            'auth': '/api/login, /api/register'
        }
    }), 200

# Initialize database
def create_tables():
    try:
        with app.app_context():
            db.create_all()
            
            # Create admin user if not exists
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(
                    username='admin',
                    email='admin@example.com',
                    password_hash=generate_password_hash('admin123'),
                    is_admin=True,
                    preferred_languages=json.dumps(['telugu', 'english', 'hindi']),
                    preferred_genres=json.dumps(['action', 'drama', 'comedy'])
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