#backend/app.py
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
from collections import defaultdict, Counter
import hashlib
import time
from sqlalchemy import func, desc
import telebot
import jwt
import urllib.parse

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
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '-1002850793757')

# OTT Platform API Keys
WATCHMODE_API_KEY = os.environ.get('WATCHMODE_API_KEY', 'WtcKDji9i20pjOl5Lg0AiyG2bddfUs3nSZRZJIsY')
RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY', 'c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37')
RAPIDAPI_HOST = 'streaming-availability.p.rapidapi.com'

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
    ott_platforms = db.Column(db.Text)  # JSON string with direct links
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
        response = requests.get(f'http://ip-api.com/json/{ip_address}', timeout=5)
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

# OTT Platform Configuration - ONLY SPECIFIED PLATFORMS
SUPPORTED_PLATFORMS = {
    # Platform ID mappings for different APIs
    'mx_player': {
        'name': 'MX Player',
        'base_url': 'https://www.mxplayer.in',
        'search_url': 'https://www.mxplayer.in/search?q={}',
        'tmdb_ids': [501, 541],
        'watchmode_ids': [541],
        'streaming_api_keys': ['mxplayer']
    },
    'jiocinema': {
        'name': 'JioCinema',
        'base_url': 'https://www.jiocinema.com',
        'search_url': 'https://www.jiocinema.com/search/{}',
        'tmdb_ids': [432],
        'watchmode_ids': [551],
        'streaming_api_keys': ['jiocinema']
    },
    'sonyliv': {
        'name': 'SonyLIV',
        'base_url': 'https://www.sonyliv.com',
        'search_url': 'https://www.sonyliv.com/search/{}',
        'tmdb_ids': [237, 442],
        'watchmode_ids': [442],
        'streaming_api_keys': ['sonyliv']
    },
    'youtube': {
        'name': 'YouTube',
        'base_url': 'https://www.youtube.com',
        'search_url': 'https://www.youtube.com/results?search_query={}',
        'tmdb_ids': [188],
        'watchmode_ids': [237],
        'streaming_api_keys': ['youtube']
    },
    'airtel_xstream': {
        'name': 'Airtel Xstream',
        'base_url': 'https://www.airtelxstream.in',
        'search_url': 'https://www.airtelxstream.in/search?q={}',
        'tmdb_ids': [],
        'watchmode_ids': [],
        'streaming_api_keys': ['airtel']
    },
    'netflix': {
        'name': 'Netflix',
        'base_url': 'https://www.netflix.com',
        'search_url': 'https://www.netflix.com/search?q={}',
        'tmdb_ids': [8],
        'watchmode_ids': [203],
        'streaming_api_keys': ['netflix']
    },
    'prime_video': {
        'name': 'Prime Video',
        'base_url': 'https://www.primevideo.com',
        'search_url': 'https://www.primevideo.com/search/ref=atv_nb_sr?phrase={}',
        'tmdb_ids': [119],
        'watchmode_ids': [26],
        'streaming_api_keys': ['prime', 'amazon']
    },
    'hotstar': {
        'name': 'Disney+ Hotstar',
        'base_url': 'https://www.hotstar.com',
        'search_url': 'https://www.hotstar.com/in/search?q={}',
        'tmdb_ids': [377],
        'watchmode_ids': [372],
        'streaming_api_keys': ['hotstar', 'disney']
    },
    'aha': {
        'name': 'Aha',
        'base_url': 'https://www.aha.video',
        'search_url': 'https://www.aha.video/search?q={}',
        'tmdb_ids': [],
        'watchmode_ids': [],
        'streaming_api_keys': ['aha']
    },
    'sun_nxt': {
        'name': 'Sun NXT',
        'base_url': 'https://www.sunnxt.com',
        'search_url': 'https://www.sunnxt.com/search?q={}',
        'tmdb_ids': [],
        'watchmode_ids': [],
        'streaming_api_keys': ['sunnxt']
    },
    'zee5': {
        'name': 'ZEE5',
        'base_url': 'https://www.zee5.com',
        'search_url': 'https://www.zee5.com/search/{}',
        'tmdb_ids': [232],
        'watchmode_ids': [457],
        'streaming_api_keys': ['zee5']
    }
}

# Enhanced OTT Availability Services - ONLY ACTUAL AVAILABILITY
class WatchModeService:
    BASE_URL = 'https://api.watchmode.com/v1'
    
    @staticmethod
    def search_title(title, title_type='movie'):
        """Search for content on WatchMode"""
        try:
            url = f"{WatchModeService.BASE_URL}/search/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'search_field': 'name',
                'search_value': title,
                'types': title_type
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"WatchMode search error: {e}")
        return None
    
    @staticmethod
    def get_title_sources(watchmode_id):
        """Get streaming sources for a title"""
        try:
            url = f"{WatchModeService.BASE_URL}/title/{watchmode_id}/sources/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'regions': 'IN,US'  # India and US regions
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"WatchMode sources error: {e}")
        return None

class StreamingAvailabilityService:
    BASE_URL = 'https://streaming-availability.p.rapidapi.com'
    
    @staticmethod
    def search_title(title, country='in'):
        """Search for content on Streaming Availability API"""
        try:
            url = f"{StreamingAvailabilityService.BASE_URL}/v2/search/title"
            headers = {
                'X-RapidAPI-Key': RAPIDAPI_KEY,
                'X-RapidAPI-Host': RAPIDAPI_HOST
            }
            params = {
                'title': title,
                'country': country,
                'show_type': 'all',
                'output_language': 'en'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Streaming Availability search error: {e}")
        return None
    
    @staticmethod
    def get_title_details(imdb_id, country='in'):
        """Get detailed streaming info by IMDB ID"""
        try:
            url = f"{StreamingAvailabilityService.BASE_URL}/v2/get/basic"
            headers = {
                'X-RapidAPI-Key': RAPIDAPI_KEY,
                'X-RapidAPI-Host': RAPIDAPI_HOST
            }
            params = {
                'country': country,
                'imdb_id': imdb_id,
                'output_language': 'en'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Streaming Availability details error: {e}")
        return None

# External API Services
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
            'append_to_response': 'credits,videos,similar,watch/providers,external_ids'
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

class YouTubeService:
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    @staticmethod
    def search_trailers(query):
        url = f"{YouTubeService.BASE_URL}/search"
        params = {
            'key': YOUTUBE_API_KEY,
            'q': f"{query} trailer",
            'part': 'snippet',
            'type': 'video',
            'maxResults': 5
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
        return None

# UPDATED: Content Management Service - ONLY ACTUAL AVAILABILITY
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            # Check if content already exists
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                # Update OTT platforms with fresh data
                updated_ott = ContentService.get_actual_availability(tmdb_data, existing.imdb_id)
                existing.ott_platforms = json.dumps(updated_ott)
                existing.updated_at = datetime.utcnow()
                db.session.commit()
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
            
            # Get IMDB ID
            imdb_id = None
            if 'external_ids' in tmdb_data:
                imdb_id = tmdb_data['external_ids'].get('imdb_id')
            
            # Get ACTUAL OTT platforms - Only real availability
            ott_platforms = ContentService.get_actual_availability(tmdb_data, imdb_id)
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
                imdb_id=imdb_id,
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
                ott_platforms=json.dumps(ott_platforms)
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
        """Save anime content from Jikan API to database"""
        try:
            # Check if anime already exists by MAL ID (using tmdb_id field to store MAL ID)
            existing = Content.query.filter_by(tmdb_id=anime_data['mal_id']).first()
            if existing:
                return existing
            
            # Extract genres
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            # Parse release date
            release_date = None
            if anime_data.get('aired') and anime_data['aired'].get('from'):
                try:
                    release_date = datetime.fromisoformat(anime_data['aired']['from'].replace('Z', '+00:00')).date()
                except:
                    release_date = None
            
            # Get anime streaming platforms (only actual availability)
            ott_platforms = ContentService.get_anime_availability(anime_data)
            
            # Create content object
            content = Content(
                tmdb_id=anime_data['mal_id'],  # Store MAL ID in tmdb_id field
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                languages=json.dumps(['japanese']),
                release_date=release_date,
                runtime=anime_data.get('duration'),
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                backdrop_path=anime_data.get('images', {}).get('jpg', {}).get('large_image_url'),
                ott_platforms=json.dumps(ott_platforms)
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
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]
    
    @staticmethod
    def get_actual_availability(tmdb_data, imdb_id=None):
        """Get ONLY actual OTT availability - No fake or placeholder links"""
        available_platforms = []
        title = tmdb_data.get('title') or tmdb_data.get('name', '')
        
        logger.info(f"Checking actual availability for: {title}")
        
        try:
            # Method 1: TMDB Watch Providers (Most reliable source)
            tmdb_platforms = ContentService.check_tmdb_providers(tmdb_data)
            if tmdb_platforms:
                available_platforms.extend(tmdb_platforms)
                logger.info(f"Found {len(tmdb_platforms)} platforms from TMDB")
            
            # Method 2: WatchMode API for additional verification
            if WATCHMODE_API_KEY:
                watchmode_platforms = ContentService.check_watchmode_availability(title)
                if watchmode_platforms:
                    available_platforms.extend(watchmode_platforms)
                    logger.info(f"Found {len(watchmode_platforms)} platforms from WatchMode")
            
            # Method 3: Streaming Availability API
            if RAPIDAPI_KEY and imdb_id:
                streaming_platforms = ContentService.check_streaming_api_availability(imdb_id)
                if streaming_platforms:
                    available_platforms.extend(streaming_platforms)
                    logger.info(f"Found {len(streaming_platforms)} platforms from Streaming API")
            
            # Remove duplicates and keep only supported platforms
            final_platforms = ContentService.filter_and_deduplicate(available_platforms)
            
            # Add audio language information
            for platform in final_platforms:
                platform['audio_languages'] = ContentService.get_audio_languages(platform['name'], tmdb_data)
                platform['last_verified'] = datetime.utcnow().isoformat()
            
            logger.info(f"Final result: {len(final_platforms)} verified platforms for {title}")
            return final_platforms
            
        except Exception as e:
            logger.error(f"Error checking availability for {title}: {e}")
            return []
    
    @staticmethod
    def check_tmdb_providers(tmdb_data):
        """Check TMDB watch providers for actual availability"""
        platforms = []
        
        if 'watch/providers' not in tmdb_data:
            return platforms
        
        providers_data = tmdb_data['watch/providers']
        
        # Check India and US regions
        for region in ['IN', 'US']:
            region_data = providers_data.get('results', {}).get(region, {})
            
            # Check all provider types
            for provider_type in ['free', 'flatrate', 'rent', 'buy']:
                if provider_type in region_data:
                    for provider in region_data[provider_type]:
                        platform_info = ContentService.map_tmdb_provider_to_supported(provider)
                        if platform_info:
                            platforms.append(platform_info)
        
        return platforms
    
    @staticmethod
    def map_tmdb_provider_to_supported(provider):
        """Map TMDB provider to our supported platforms"""
        provider_id = provider.get('provider_id')
        
        # Find matching supported platform
        for platform_key, platform_config in SUPPORTED_PLATFORMS.items():
            if provider_id in platform_config['tmdb_ids']:
                platform_info = {
                    'name': platform_config['name'],
                    'platform_key': platform_key,
                    'verified': True,
                    'source': 'tmdb'
                }
                
                # Add direct link if available from TMDB
                if provider.get('link'):
                    platform_info['direct_url'] = provider['link']
                    platform_info['has_direct_link'] = True
                else:
                    platform_info['has_direct_link'] = False
                
                return platform_info
        
        return None
    
    @staticmethod
    def check_watchmode_availability(title):
        """Check WatchMode API for actual availability"""
        platforms = []
        
        try:
            search_results = WatchModeService.search_title(title)
            if search_results and search_results.get('title_results'):
                first_result = search_results['title_results'][0]
                watchmode_id = first_result.get('id')
                
                if watchmode_id:
                    sources = WatchModeService.get_title_sources(watchmode_id)
                    if sources:
                        for source in sources:
                            platform_info = ContentService.map_watchmode_to_supported(source)
                            if platform_info:
                                platforms.append(platform_info)
        except Exception as e:
            logger.error(f"WatchMode check error: {e}")
        
        return platforms
    
    @staticmethod
    def map_watchmode_to_supported(source):
        """Map WatchMode source to our supported platforms"""
        source_id = source.get('source_id')
        
        # Find matching supported platform
        for platform_key, platform_config in SUPPORTED_PLATFORMS.items():
            if source_id in platform_config['watchmode_ids']:
                platform_info = {
                    'name': platform_config['name'],
                    'platform_key': platform_key,
                    'verified': True,
                    'source': 'watchmode'
                }
                
                # Add direct link if available
                if source.get('web_url'):
                    platform_info['direct_url'] = source['web_url']
                    platform_info['has_direct_link'] = True
                else:
                    platform_info['has_direct_link'] = False
                
                # Add quality and type info
                if source.get('type'):
                    platform_info['availability_type'] = source['type']
                if source.get('quality'):
                    platform_info['quality'] = source['quality']
                
                return platform_info
        
        return None
    
    @staticmethod
    def check_streaming_api_availability(imdb_id):
        """Check Streaming Availability API for actual availability"""
        platforms = []
        
        try:
            streaming_data = StreamingAvailabilityService.get_title_details(imdb_id)
            if streaming_data and streaming_data.get('streamingInfo'):
                streaming_info = streaming_data['streamingInfo']
                
                for service_key, service_data in streaming_info.items():
                    platform_info = ContentService.map_streaming_api_to_supported(service_key, service_data)
                    if platform_info:
                        platforms.append(platform_info)
        except Exception as e:
            logger.error(f"Streaming API check error: {e}")
        
        return platforms
    
    @staticmethod
    def map_streaming_api_to_supported(service_key, service_data):
        """Map Streaming API service to our supported platforms"""
        # Find matching supported platform
        for platform_key, platform_config in SUPPORTED_PLATFORMS.items():
            if service_key.lower() in platform_config['streaming_api_keys']:
                platform_info = {
                    'name': platform_config['name'],
                    'platform_key': platform_key,
                    'verified': True,
                    'source': 'streaming_api'
                }
                
                # Extract service data (could be list for different regions)
                if isinstance(service_data, list) and service_data:
                    service_info = service_data[0]  # Take first region
                else:
                    service_info = service_data
                
                # Add direct link if available
                if service_info.get('link'):
                    platform_info['direct_url'] = service_info['link']
                    platform_info['has_direct_link'] = True
                else:
                    platform_info['has_direct_link'] = False
                
                # Add additional info
                if service_info.get('quality'):
                    platform_info['quality'] = service_info['quality']
                if service_info.get('audioLanguages'):
                    platform_info['audio_languages'] = service_info['audioLanguages']
                if service_info.get('subtitleLanguages'):
                    platform_info['subtitle_languages'] = service_info['subtitleLanguages']
                
                # Add multiple audio language links if available
                if service_info.get('audios'):
                    platform_info['audio_links'] = []
                    for audio in service_info['audios']:
                        if audio.get('link'):
                            platform_info['audio_links'].append({
                                'language': audio.get('language'),
                                'link': audio['link']
                            })
                
                return platform_info
        
        return None
    
    @staticmethod
    def get_anime_availability(anime_data):
        """Get anime streaming availability - only actual availability"""
        platforms = []
        title = anime_data.get('title', '')
        
        # For anime, we need to check anime-specific platforms
        # This would require anime-specific APIs like Crunchyroll, Funimation etc.
        # For now, return empty list unless we have actual verification
        
        try:
            # You can add anime-specific API checks here
            # For example, checking Crunchyroll API, Funimation API etc.
            pass
        except Exception as e:
            logger.error(f"Error getting anime availability: {e}")
        
        return platforms
    
    @staticmethod
    def filter_and_deduplicate(platforms):
        """Remove duplicates and keep only supported platforms"""
        seen_platforms = {}
        
        # Sort by verification and source priority
        platforms.sort(key=lambda x: (
            not x.get('verified', False),
            x.get('source') != 'tmdb',  # TMDB is most reliable
            x.get('name', '')
        ))
        
        for platform in platforms:
            platform_key = platform.get('platform_key', '')
            if platform_key and platform_key not in seen_platforms:
                # Only include if it's a supported platform
                if platform_key in SUPPORTED_PLATFORMS:
                    seen_platforms[platform_key] = platform
        
        return list(seen_platforms.values())
    
    @staticmethod
    def get_audio_languages(platform_name, tmdb_data):
        """Get available audio languages for a platform"""
        # Platform-specific audio language mapping
        platform_audio_map = {
            'MX Player': ['hindi', 'english', 'tamil', 'telugu', 'kannada'],
            'JioCinema': ['hindi', 'english', 'tamil', 'telugu'],
            'SonyLIV': ['hindi', 'english', 'tamil', 'telugu', 'malayalam'],
            'YouTube': ['hindi', 'english', 'tamil', 'telugu', 'kannada', 'malayalam'],
            'Airtel Xstream': ['hindi', 'english', 'tamil', 'telugu'],
            'Netflix': ['hindi', 'english', 'tamil', 'telugu', 'malayalam'],
            'Prime Video': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada'],
            'Disney+ Hotstar': ['hindi', 'english', 'tamil', 'telugu'],
            'Aha': ['telugu', 'tamil'],
            'Sun NXT': ['tamil', 'telugu', 'kannada', 'malayalam'],
            'ZEE5': ['hindi', 'tamil', 'telugu', 'kannada', 'malayalam', 'bengali']
        }
        
        # Get languages from TMDB data
        content_languages = []
        if 'spoken_languages' in tmdb_data:
            content_languages = [lang['iso_639_1'] for lang in tmdb_data['spoken_languages']]
        elif 'original_language' in tmdb_data:
            content_languages = [tmdb_data['original_language']]
        
        # Map ISO codes to language names
        iso_to_name = {
            'hi': 'hindi', 'en': 'english', 'ta': 'tamil', 
            'te': 'telugu', 'kn': 'kannada', 'ml': 'malayalam',
            'bn': 'bengali', 'gu': 'gujarati', 'mr': 'marathi'
        }
        
        content_lang_names = [iso_to_name.get(code, code) for code in content_languages]
        
        # Return intersection of platform languages and content languages
        platform_langs = platform_audio_map.get(platform_name, ['original'])
        available_langs = list(set(platform_langs) & set(content_lang_names))
        
        return available_langs if available_langs else ['original']

# Recommendation Engine
class RecommendationEngine:
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all'):
        try:
            trending_data = TMDBService.get_trending(content_type=content_type)
            if not trending_data:
                return []
            
            recommendations = []
            for item in trending_data.get('results', [])[:limit]:
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_popular_by_genre(genre, limit=20, region=None):
        try:
            popular_movies = TMDBService.get_popular('movie', region=region)
            popular_tv = TMDBService.get_popular('tv', region=region)
            
            recommendations = []
            
            # Process movies
            if popular_movies:
                for item in popular_movies.get('results', []):
                    if genre.lower() in [g.lower() for g in ContentService.map_genre_ids(item.get('genre_ids', []))]:
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content:
                            recommendations.append(content)
            
            # Process TV shows
            if popular_tv:
                for item in popular_tv.get('results', []):
                    if genre.lower() in [g.lower() for g in ContentService.map_genre_ids(item.get('genre_ids', []))]:
                        content = ContentService.save_content_from_tmdb(item, 'tv')
                        if content:
                            recommendations.append(content)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting popular by genre: {e}")
            return []
    
    @staticmethod
    def get_regional_recommendations(language, limit=20):
        try:
            search_queries = {
                'hindi': ['bollywood', 'hindi movie'],
                'telugu': ['tollywood', 'telugu movie'],
                'tamil': ['kollywood', 'tamil movie'],
                'kannada': ['sandalwood', 'kannada movie']
            }
            
            recommendations = []
            queries = search_queries.get(language.lower(), [language])
            
            for query in queries:
                search_results = TMDBService.search_content(query)
                if search_results:
                    for item in search_results.get('results', []):
                        content_type_detected = 'movie' if 'title' in item else 'tv'
                        content = ContentService.save_content_from_tmdb(item, content_type_detected)
                        if content:
                            recommendations.append(content)
                        
                        if len(recommendations) >= limit:
                            break
                
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
            location = get_user_location(ip_address)
            interactions = AnonymousInteraction.query.filter_by(session_id=session_id).all()
            
            recommendations = []
            
            # If user has interactions, recommend similar content
            if interactions:
                viewed_content_ids = [interaction.content_id for interaction in interactions]
                viewed_contents = Content.query.filter(Content.id.in_(viewed_content_ids)).all()
                
                all_genres = []
                for content in viewed_contents:
                    if content.genres:
                        all_genres.extend(json.loads(content.genres))
                
                genre_counts = Counter(all_genres)
                top_genres = [genre for genre, _ in genre_counts.most_common(3)]
                
                for genre in top_genres:
                    genre_recs = RecommendationEngine.get_popular_by_genre(genre, limit=7)
                    recommendations.extend(genre_recs)
            
            # Add regional content based on location
            if location and location.get('country') == 'India':
                regional_recs = RecommendationEngine.get_regional_recommendations('hindi', limit=5)
                recommendations.extend(regional_recs)
            
            # Add trending content
            trending_recs = RecommendationEngine.get_trending_recommendations(limit=10)
            recommendations.extend(trending_recs)
            
            # Remove duplicates and limit
            seen_ids = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec.id not in seen_ids:
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
            
            # Get OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create OTT availability text - ONLY ACTUAL AVAILABILITY
            ott_text = ""
            if ott_platforms:
                ott_text += "\n\n📺 **AVAILABLE ON:**\n"
                for platform in ott_platforms[:5]:  # Show max 5 platforms
                    ott_text += f"▶️ **{platform['name']}**"
                    if platform.get('direct_url'):
                        ott_text += f" - [Watch Now]({platform['direct_url']})"
                    if platform.get('audio_languages'):
                        langs = ', '.join(platform['audio_languages'][:2])
                        ott_text += f" ({langs})"
                    ott_text += "\n"
            else:
                ott_text = "\n\n🔍 **Availability:** Check local platforms or search online"
            
            # Create message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}{ott_text}

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
    def send_ott_availability(content_title, ott_platforms):
        """Send OTT availability information to Telegram"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = f"📺 **Where to Watch: {content_title}**\n\n"
            
            if ott_platforms:
                message += "📱 **AVAILABLE ON:**\n"
                for platform in ott_platforms[:5]:
                    message += f"▶️ **{platform['name']}**"
                    if platform.get('direct_url'):
                        message += f" - [Watch Now]({platform['direct_url']})"
                    
                    # Add audio language info
                    if platform.get('audio_languages'):
                        langs = ', '.join(platform['audio_languages'][:2])
                        message += f"\n   🎵 Audio: {langs}"
                    
                    message += "\n\n"
            else:
                message += "🔍 **Availability:** Currently not available on tracked platforms"
            
            message += "#WhereToWatch #OTT #Streaming"
            
            bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            return True
            
        except Exception as e:
            logger.error(f"Error sending OTT availability to Telegram: {e}")
            return False

# API Routes

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

# UPDATED: Enhanced search with ONLY ACTUAL AVAILABILITY
@app.route('/api/search', methods=['GET'])
def search_content():
    try:
        query = request.args.get('query', '')
        content_type = request.args.get('type', 'multi')
        page = int(request.args.get('page', 1))
        platform_filter = request.args.get('platform', '')
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        session_id = get_session_id()
        
        # Search TMDB
        tmdb_results = TMDBService.search_content(query, content_type, page=page)
        
        # Search anime if requested
        anime_results = None
        if content_type in ['anime', 'multi']:
            anime_results = JikanService.search_anime(query, page=page)
        
        results = []
        
        if tmdb_results:
            for item in tmdb_results.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    # Parse OTT platforms - ONLY ACTUAL AVAILABILITY
                    ott_platforms = []
                    if content.ott_platforms:
                        try:
                            ott_platforms = json.loads(content.ott_platforms)
                        except:
                            ott_platforms = []
                    
                    # Apply platform filter if specified
                    if platform_filter:
                        filtered_platforms = [p for p in ott_platforms if platform_filter.lower() in p.get('name', '').lower()]
                        if not filtered_platforms:
                            continue  # Skip if platform not available
                        ott_platforms = filtered_platforms
                    
                    # Record interaction
                    interaction = AnonymousInteraction(
                        session_id=session_id,
                        content_id=content.id,
                        interaction_type='search',
                        ip_address=request.remote_addr
                    )
                    db.session.add(interaction)
                    
                    # Count platforms with direct links
                    platforms_with_links = len([p for p in ott_platforms if p.get('has_direct_link')])
                    
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
                        'availability_summary': {
                            'total_platforms': len(ott_platforms),
                            'platforms_with_direct_links': platforms_with_links,
                            'available_on': [p['name'] for p in ott_platforms[:3]],
                            'has_availability': len(ott_platforms) > 0
                        }
                    })
        
        # Process anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                content = ContentService.save_anime_content(anime)
                if content:
                    ott_platforms = []
                    if content.ott_platforms:
                        try:
                            ott_platforms = json.loads(content.ott_platforms)
                        except:
                            ott_platforms = []
                    
                    results.append({
                        'id': content.id,
                        'title': content.title,
                        'content_type': 'anime',
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': content.poster_path,
                        'overview': content.overview,
                        'availability_summary': {
                            'total_platforms': len(ott_platforms),
                            'platforms_with_direct_links': len([p for p in ott_platforms if p.get('has_direct_link')]),
                            'available_on': [p['name'] for p in ott_platforms[:3]],
                            'has_availability': len(ott_platforms) > 0
                        }
                    })
        
        db.session.commit()
        
        return jsonify({
            'results': results,
            'total_results': tmdb_results.get('total_results', 0) if tmdb_results else 0,
            'total_pages': tmdb_results.get('total_pages', 0) if tmdb_results else 0,
            'current_page': page,
            'filters_applied': {
                'platform_filter': platform_filter
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

# UPDATED: Content Discovery Route with ONLY ACTUAL AVAILABILITY
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
        if content.tmdb_id and content.content_type != 'anime':
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
            
            # Update OTT platforms with latest ACTUAL data
            if additional_details:
                updated_ott = ContentService.get_actual_availability(additional_details, content.imdb_id)
                content.ott_platforms = json.dumps(updated_ott)
                content.updated_at = datetime.utcnow()
        
        # Get YouTube trailers
        trailers = []
        if YOUTUBE_API_KEY:
            search_query = f"{content.title} {'anime' if content.content_type == 'anime' else ''} trailer"
            youtube_results = YouTubeService.search_trailers(search_query)
            if youtube_results:
                for video in youtube_results.get('items', []):
                    trailers.append({
                        'title': video['snippet']['title'],
                        'url': f"https://www.youtube.com/watch?v={video['id']['videoId']}",
                        'thumbnail': video['snippet']['thumbnails']['medium']['url']
                    })
        
        # Get similar content
        similar_content = []
        if additional_details and content.content_type != 'anime':
            if 'similar' in additional_details:
                for item in additional_details['similar']['results'][:5]:
                    similar = ContentService.save_content_from_tmdb(item, content.content_type)
                    if similar:
                        similar_content.append({
                            'id': similar.id,
                            'title': similar.title,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{similar.poster_path}" if similar.poster_path else None,
                            'rating': similar.rating
                        })
        
        # Parse OTT platforms - ONLY ACTUAL AVAILABILITY
        ott_platforms = []
        if content.ott_platforms:
            try:
                ott_platforms = json.loads(content.ott_platforms)
            except:
                ott_platforms = []
        
        # Separate platforms with and without direct links
        platforms_with_links = [p for p in ott_platforms if p.get('has_direct_link')]
        platforms_without_links = [p for p in ott_platforms if not p.get('has_direct_link')]
        
        db.session.commit()
        
        response_data = {
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'imdb_id': content.imdb_id,
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
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else [],
            
            # SIMPLIFIED OTT Information - ONLY ACTUAL AVAILABILITY
            'streaming_info': {
                'total_platforms': len(ott_platforms),
                'platforms_with_direct_links': len(platforms_with_links),
                'platforms_without_direct_links': len(platforms_without_links),
                'is_available': len(ott_platforms) > 0,
                'last_updated': content.updated_at.isoformat() if content.updated_at else None
            },
            'available_platforms': ott_platforms  # Simple list of all available platforms
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# New endpoint for getting OTT availability
@app.route('/api/content/<int:content_id>/ott', methods=['GET'])
def get_ott_availability_endpoint(content_id):
    """Get detailed OTT availability for a content"""
    try:
        content = Content.query.get_or_404(content_id)
        
        # Get fresh OTT data
        if content.tmdb_id:
            tmdb_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
            if tmdb_details:
                fresh_ott = ContentService.get_actual_availability(tmdb_details, content.imdb_id)
                
                # Update content with fresh data
                content.ott_platforms = json.dumps(fresh_ott)
                content.updated_at = datetime.utcnow()
                db.session.commit()
                
                return jsonify({
                    'content_title': content.title,
                    'last_updated': content.updated_at.isoformat(),
                    'available_platforms': fresh_ott,
                    'total_platforms': len(fresh_ott),
                    'platforms_with_direct_links': len([p for p in fresh_ott if p.get('has_direct_link')]),
                    'is_available': len(fresh_ott) > 0
                }), 200
        
        return jsonify({
            'content_title': content.title,
            'available_platforms': [],
            'is_available': False,
            'message': 'No verified OTT availability found'
        }), 200
        
    except Exception as e:
        logger.error(f"OTT availability error: {e}")
        return jsonify({'error': 'Failed to get OTT availability'}), 500

# Helper function to process platform data for API responses - SIMPLIFIED
def process_platforms_for_response(content):
    """Process OTT platforms for API response - only actual availability"""
    ott_platforms = []
    if content.ott_platforms:
        try:
            ott_platforms = json.loads(content.ott_platforms)
        except:
            ott_platforms = []
    
    return {
        'available_platforms': ott_platforms,
        'total_platforms': len(ott_platforms),
        'platforms_with_direct_links': len([p for p in ott_platforms if p.get('has_direct_link')]),
        'is_available': len(ott_platforms) > 0
    }

# UPDATED: Recommendation Routes with ONLY ACTUAL AVAILABILITY
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = []
        for content in recommendations:
            platforms_info = process_platforms_for_response(content)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_info': platforms_info
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/recommendations/popular/<genre>', methods=['GET'])
def get_popular_by_genre(genre):
    try:
        limit = int(request.args.get('limit', 20))
        region = request.args.get('region')
        
        recommendations = RecommendationEngine.get_popular_by_genre(genre, limit, region)
        
        result = []
        for content in recommendations:
            platforms_info = process_platforms_for_response(content)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_info': platforms_info
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Popular by genre error: {e}")
        return jsonify({'error': 'Failed to get popular recommendations'}), 500

@app.route('/api/recommendations/regional/<language>', methods=['GET'])
def get_regional(language):
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_regional_recommendations(language, limit)
        
        result = []
        for content in recommendations:
            platforms_info = process_platforms_for_response(content)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_info': platforms_info
            })
        
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
            platforms_info = process_platforms_for_response(content)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_info': platforms_info
            })
        
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
            platforms_info = process_platforms_for_response(content)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_info': platforms_info
            })
        
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
        
        # For now, fallback to trending recommendations
        # You can implement ML-based recommendations here
        trending_recommendations = RecommendationEngine.get_trending_recommendations(20)
        
        result = []
        for content in trending_recommendations:
            platforms_info = process_platforms_for_response(content)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_info': platforms_info,
                'recommendation_reason': 'Based on trending content'
            })
        
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
            platforms_info = process_platforms_for_response(content)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'streaming_info': platforms_info
            })
        
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
            platforms_info = process_platforms_for_response(content)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'streaming_info': platforms_info
            })
        
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
        source = request.args.get('source', 'tmdb')  # tmdb, anime
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
        if data.get('id'):
            existing_content = Content.query.filter_by(tmdb_id=data['id']).first()
        
        if existing_content:
            return jsonify({
                'message': 'Content already exists',
                'content_id': existing_content.id
            }), 200
        
        # Create new content from external data
        try:
            release_date = None
            if data.get('release_date'):
                try:
                    release_date = datetime.strptime(data['release_date'], '%Y-%m-%d').date()
                except:
                    release_date = None
            
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
                ott_platforms=json.dumps(data.get('ott_platforms', []))
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
        
        # Get content
        content = Content.query.get(data['content_id'])
        if not content:
            content = Content.query.filter_by(tmdb_id=data['content_id']).first()
        
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
                platforms_info = process_platforms_for_response(content)
                
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'streaming_info': platforms_info,
                    'admin_description': rec.description,
                    'admin_name': admin.username if admin else 'Admin',
                    'recommended_at': rec.created_at.isoformat()
                })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Public admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get admin recommendations'}), 500

# Quick search endpoint
@app.route('/api/quick-search', methods=['GET'])
def quick_search():
    """Quick search with immediate availability check"""
    try:
        query = request.args.get('query', '')
        if not query:
            return jsonify({'error': 'Query required'}), 400
        
        # Search TMDB quickly
        tmdb_results = TMDBService.search_content(query, 'multi', page=1)
        results = []
        
        if tmdb_results:
            for item in tmdb_results.get('results', [])[:5]:  # Limit to 5 results
                content_type = 'movie' if 'title' in item else 'tv'
                title = item.get('title') or item.get('name', '')
                
                results.append({
                    'title': title,
                    'content_type': content_type,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{item['poster_path']}" if item.get('poster_path') else None,
                    'rating': item.get('vote_average'),
                    'release_date': item.get('release_date') or item.get('first_air_date'),
                    'overview': item.get('overview', '')[:150] + '...',
                    'tmdb_id': item['id']
                })
        
        return jsonify({
            'results': results,
            'query': query,
            'total_found': len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Quick search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '3.0.0',
        'features': {
            'actual_availability_only': True,
            'direct_links_when_available': True,
            'supported_platforms': list(SUPPORTED_PLATFORMS.keys()),
            'no_fake_links': True,
            'verified_streaming_info': True,
            'telegram_bot': bot is not None,
            'watchmode_api': bool(WATCHMODE_API_KEY),
            'streaming_availability_api': bool(RAPIDAPI_KEY)
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