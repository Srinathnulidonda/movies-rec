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
RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY', 'c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37')

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
    mal_id = db.Column(db.Integer)  # MyAnimeList ID for anime
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    content_type = db.Column(db.String(20), nullable=False)  # movie, tv, anime
    genres = db.Column(db.Text)  # JSON string
    languages = db.Column(db.Text)  # JSON string
    available_languages = db.Column(db.Text)  # JSON string for audio languages
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float)
    vote_count = db.Column(db.Integer)
    popularity = db.Column(db.Float)
    overview = db.Column(db.Text)
    poster_path = db.Column(db.String(255))
    backdrop_path = db.Column(db.String(255))
    trailer_url = db.Column(db.String(255))
    streaming_platforms = db.Column(db.Text)  # JSON string with language-specific links
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

# Streaming Platform Information
STREAMING_PLATFORMS = {
    # Free Platforms
    'mx_player': {'name': 'MX Player', 'is_free': True, 'url': 'https://mxplayer.in'},
    'jio_hotstar': {'name': 'JioHotstar', 'is_free': True, 'url': 'https://jiocinema.com'},
    'sonyliv_free': {'name': 'SonyLIV', 'is_free': True, 'url': 'https://sonyliv.com'},
    'zee5_free': {'name': 'ZEE5', 'is_free': True, 'url': 'https://zee5.com'},
    'youtube': {'name': 'YouTube', 'is_free': True, 'url': 'https://youtube.com'},
    'crunchyroll_free': {'name': 'Crunchyroll', 'is_free': True, 'url': 'https://crunchyroll.com'},
    'airtel_xstream': {'name': 'Airtel Xstream', 'is_free': True, 'url': 'https://airtelxstream.in'},
    
    # Paid Platforms
    'netflix': {'name': 'Netflix', 'is_free': False, 'url': 'https://netflix.com'},
    'prime_video': {'name': 'Prime Video', 'is_free': False, 'url': 'https://primevideo.com'},
    'disney_plus_hotstar': {'name': 'Disney+ Hotstar', 'is_free': False, 'url': 'https://hotstar.com'},
    'zee5_premium': {'name': 'ZEE5 Premium', 'is_free': False, 'url': 'https://zee5.com'},
    'sonyliv_premium': {'name': 'SonyLIV Premium', 'is_free': False, 'url': 'https://sonyliv.com'},
    'aha': {'name': 'Aha', 'is_free': False, 'url': 'https://aha.video'},
    'sun_nxt': {'name': 'Sun NXT', 'is_free': False, 'url': 'https://sunnxt.com'}
}

# Language Mapping
LANGUAGE_MAPPING = {
    'hindi': {'name': 'Hindi', 'code': 'hi', 'flag': '🇮🇳'},
    'telugu': {'name': 'Telugu', 'code': 'te', 'flag': '🇮🇳'},
    'tamil': {'name': 'Tamil', 'code': 'ta', 'flag': '🇮🇳'},
    'malayalam': {'name': 'Malayalam', 'code': 'ml', 'flag': '🇮🇳'},
    'kannada': {'name': 'Kannada', 'code': 'kn', 'flag': '🇮🇳'},
    'english': {'name': 'English', 'code': 'en', 'flag': '🇺🇸'}
}

# Genre Categories
GENRE_CATEGORIES = [
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci-Fi', 'Thriller', 'Western'
]

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

# Streaming Service
class StreamingService:
    @staticmethod
    def check_streaming_availability(title, imdb_id=None, tmdb_id=None):
        """Check where content is available to stream with language-specific links"""
        try:
            platforms_data = []
            
            # Method 1: Use Streaming Availability API (RapidAPI)
            if RAPIDAPI_KEY:
                platforms_from_rapid = StreamingService._check_rapid_api(title, imdb_id)
                platforms_data.extend(platforms_from_rapid)
            
            # Method 2: Use WatchMode API
            if WATCHMODE_API_KEY and WATCHMODE_API_KEY != 'your_watchmode_api_key':
                platforms_from_watchmode = StreamingService._check_watchmode_api(title, imdb_id)
                platforms_data.extend(platforms_from_watchmode)
            
            # Method 3: Fallback - Smart assignment based on content type and language
            if not platforms_data:
                platforms_data = StreamingService._get_fallback_platforms(title, tmdb_id)
            
            return platforms_data
            
        except Exception as e:
            logger.error(f"Streaming availability check error: {e}")
            return StreamingService._get_fallback_platforms(title, tmdb_id)
    
    @staticmethod
    def _check_rapid_api(title, imdb_id=None):
        """Check using Streaming Availability API"""
        try:
            headers = {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': 'streaming-availability.p.rapidapi.com'
            }
            
            # Search for the title
            search_url = "https://streaming-availability.p.rapidapi.com/search/title"
            params = {
                'title': title,
                'country': 'in',  # India
                'show_type': 'all'
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                platforms = []
                
                for result in data.get('result', []):
                    streaming_info = result.get('streamingInfo', {})
                    
                    for country, services in streaming_info.items():
                        if country == 'in':  # India
                            for service_info in services:
                                service_name = service_info.get('service')
                                link = service_info.get('link')
                                
                                # Map service names to our platform IDs
                                platform_id = StreamingService._map_service_name(service_name)
                                if platform_id:
                                    platforms.append({
                                        'platform_id': platform_id,
                                        'platform_name': STREAMING_PLATFORMS[platform_id]['name'],
                                        'is_free': STREAMING_PLATFORMS[platform_id]['is_free'],
                                        'link': link,
                                        'languages': StreamingService._detect_available_languages(title)
                                    })
                
                return platforms
                
        except Exception as e:
            logger.error(f"RapidAPI streaming check error: {e}")
        
        return []
    
    @staticmethod
    def _check_watchmode_api(title, imdb_id=None):
        """Check using WatchMode API"""
        try:
            headers = {'X-API-Key': WATCHMODE_API_KEY}
            
            # Search for the title
            search_url = "https://api.watchmode.com/v1/search/"
            params = {
                'search_field': 'name',
                'search_value': title
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                results = response.json()
                platforms = []
                
                if results.get('title_results'):
                    # Get first result
                    title_id = results['title_results'][0]['id']
                    
                    # Get sources for this title
                    sources_url = f"https://api.watchmode.com/v1/title/{title_id}/sources/"
                    sources_response = requests.get(sources_url, headers=headers, timeout=10)
                    
                    if sources_response.status_code == 200:
                        sources_data = sources_response.json()
                        
                        for source in sources_data:
                            source_name = source.get('name', '').lower()
                            web_url = source.get('web_url')
                            
                            # Map to our platforms
                            platform_id = StreamingService._map_watchmode_source(source_name)
                            if platform_id:
                                platforms.append({
                                    'platform_id': platform_id,
                                    'platform_name': STREAMING_PLATFORMS[platform_id]['name'],
                                    'is_free': STREAMING_PLATFORMS[platform_id]['is_free'],
                                    'link': web_url,
                                    'languages': StreamingService._detect_available_languages(title)
                                })
                
                return platforms
                
        except Exception as e:
            logger.error(f"WatchMode API error: {e}")
        
        return []
    
    @staticmethod
    def _get_fallback_platforms(title, tmdb_id=None):
        """Smart fallback platform assignment"""
        platforms = []
        
        # Detect language from title
        languages = StreamingService._detect_available_languages(title)
        
        # Regional content assignment
        if any(lang in ['hindi', 'telugu', 'tamil', 'malayalam', 'kannada'] for lang in languages):
            # Indian regional content
            regional_platforms = [
                ('zee5_free', True), ('sonyliv_free', True), ('mx_player', True),
                ('youtube', True), ('disney_plus_hotstar', False), ('zee5_premium', False)
            ]
            
            for platform_id, is_free in regional_platforms:
                platforms.append({
                    'platform_id': platform_id,
                    'platform_name': STREAMING_PLATFORMS[platform_id]['name'],
                    'is_free': is_free,
                    'link': STREAMING_PLATFORMS[platform_id]['url'],
                    'languages': languages
                })
        
        # English content
        if 'english' in languages:
            english_platforms = [
                ('netflix', False), ('prime_video', False), ('youtube', True)
            ]
            
            for platform_id, is_free in english_platforms:
                platforms.append({
                    'platform_id': platform_id,
                    'platform_name': STREAMING_PLATFORMS[platform_id]['name'],
                    'is_free': is_free,
                    'link': STREAMING_PLATFORMS[platform_id]['url'],
                    'languages': ['english']
                })
        
        return platforms[:6]  # Limit to 6 platforms
    
    @staticmethod
    def _detect_available_languages(title):
        """Detect available languages based on title and other factors"""
        languages = []
        
        # Check for regional language keywords in title
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['hindi', 'bollywood', 'hindustani']):
            languages.append('hindi')
        if any(word in title_lower for word in ['telugu', 'tollywood']):
            languages.append('telugu')
        if any(word in title_lower for word in ['tamil', 'kollywood']):
            languages.append('tamil')
        if any(word in title_lower for word in ['malayalam', 'mollywood']):
            languages.append('malayalam')
        if any(word in title_lower for word in ['kannada', 'sandalwood']):
            languages.append('kannada')
        
        # Default to English if no regional language detected
        if not languages:
            languages.append('english')
        
        return languages
    
    @staticmethod
    def _map_service_name(service_name):
        """Map service names from APIs to our platform IDs"""
        service_mapping = {
            'netflix': 'netflix',
            'amazon': 'prime_video',
            'prime': 'prime_video',
            'disney': 'disney_plus_hotstar',
            'hotstar': 'disney_plus_hotstar',
            'zee5': 'zee5_premium',
            'sonyliv': 'sonyliv_premium',
            'youtube': 'youtube',
            'mx': 'mx_player',
            'jio': 'jio_hotstar',
            'aha': 'aha',
            'sun': 'sun_nxt'
        }
        
        service_lower = service_name.lower()
        for key, platform_id in service_mapping.items():
            if key in service_lower:
                return platform_id
        
        return None
    
    @staticmethod
    def _map_watchmode_source(source_name):
        """Map WatchMode source names to our platform IDs"""
        return StreamingService._map_service_name(source_name)

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
            'append_to_response': 'credits,videos,similar,watch/providers,translations'
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
    
    @staticmethod
    def discover_regional_content(language='hi', page=1):
        """Discover content by language"""
        url = f"{TMDBService.BASE_URL}/discover/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'with_original_language': language,
            'sort_by': 'popularity.desc',
            'page': page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
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
            # Add delay to respect rate limits
            time.sleep(1)
        except Exception as e:
            logger.error(f"Jikan search error: {e}")
        return None
    
    @staticmethod
    def get_anime_details(anime_id):
        url = f"{JikanService.BASE_URL}/anime/{anime_id}/full"
        
        try:
            response = requests.get(url, params=None, timeout=10)
            if response.status_code == 200:
                return response.json()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Jikan anime details error: {e}")
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
            time.sleep(1)
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
            
            # Extract languages and available audio languages
            languages = []
            available_languages = []
            
            if 'spoken_languages' in tmdb_data:
                languages = [lang['name'] for lang in tmdb_data['spoken_languages']]
                available_languages = [lang['iso_639_1'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
                available_languages = [tmdb_data['original_language']]
            
            # Map language codes to our language system
            mapped_languages = ContentService._map_languages(available_languages)
            
            # Get streaming platforms with language-specific availability
            title = tmdb_data.get('title') or tmdb_data.get('name')
            streaming_platforms = StreamingService.check_streaming_availability(
                title, 
                imdb_id=tmdb_data.get('imdb_id'),
                tmdb_id=tmdb_data['id']
            )
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
                title=title,
                original_title=tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                content_type=content_type,
                genres=json.dumps(genres),
                languages=json.dumps(languages),
                available_languages=json.dumps(mapped_languages),
                release_date=datetime.strptime(tmdb_data.get('release_date') or tmdb_data.get('first_air_date', '1900-01-01'), '%Y-%m-%d').date() if tmdb_data.get('release_date') or tmdb_data.get('first_air_date') else None,
                runtime=tmdb_data.get('runtime'),
                rating=tmdb_data.get('vote_average'),
                vote_count=tmdb_data.get('vote_count'),
                popularity=tmdb_data.get('popularity'),
                overview=tmdb_data.get('overview'),
                poster_path=tmdb_data.get('poster_path'),
                backdrop_path=tmdb_data.get('backdrop_path'),
                streaming_platforms=json.dumps(streaming_platforms)
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
            
            # Anime is typically in Japanese with subtitles
            available_languages = ['japanese', 'english']  # Japanese audio, English subs
            
            # Get streaming platforms for anime
            streaming_platforms = StreamingService.check_streaming_availability(
                anime_data.get('title'),
                tmdb_id=None
            )
            
            # Add anime-specific platforms
            anime_platforms = [
                {
                    'platform_id': 'crunchyroll_free',
                    'platform_name': 'Crunchyroll',
                    'is_free': True,
                    'link': 'https://crunchyroll.com',
                    'languages': ['japanese']
                }
            ]
            streaming_platforms.extend(anime_platforms)
            
            # Parse release date
            release_date = None
            if anime_data.get('aired', {}).get('from'):
                try:
                    release_date = datetime.strptime(anime_data['aired']['from'][:10], '%Y-%m-%d').date()
                except:
                    pass
            
            content = Content(
                mal_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                languages=json.dumps(['Japanese']),
                available_languages=json.dumps(available_languages),
                release_date=release_date,
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                streaming_platforms=json.dumps(streaming_platforms)
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
            10759: 'Action & Adventure', 10765: 'Sci-Fi & Fantasy'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]
    
    @staticmethod
    def _map_languages(language_codes):
        """Map ISO language codes to our language system"""
        code_mapping = {
            'hi': 'hindi',
            'te': 'telugu', 
            'ta': 'tamil',
            'ml': 'malayalam',
            'kn': 'kannada',
            'en': 'english',
            'ja': 'japanese',
            'ko': 'korean'
        }
        
        mapped = []
        for code in language_codes:
            if code in code_mapping:
                mapped.append(code_mapping[code])
            else:
                mapped.append('english')  # Default fallback
        
        return list(set(mapped))  # Remove duplicates

# Enhanced Recommendation Engine
class RecommendationEngine:
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all', region='IN'):
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
            
            # Prioritize Telugu and English content
            recommendations = RecommendationEngine._prioritize_languages(recommendations, ['telugu', 'english'])
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_popular_by_genre(genre, limit=20, region='IN'):
        try:
            popular_movies = TMDBService.get_popular('movie', region=region)
            popular_tv = TMDBService.get_popular('tv', region=region)
            
            # Also get regional content
            regional_content = []
            for lang_code in ['hi', 'te', 'ta', 'ml', 'kn']:
                regional_data = TMDBService.discover_regional_content(lang_code)
                if regional_data:
                    regional_content.extend(regional_data.get('results', []))
            
            recommendations = []
            all_content = []
            
            if popular_movies:
                all_content.extend(popular_movies.get('results', []))
            if popular_tv:
                all_content.extend(popular_tv.get('results', []))
            all_content.extend(regional_content)
            
            # Filter by genre and process
            for item in all_content:
                item_genres = ContentService.map_genre_ids(item.get('genre_ids', []))
                if genre.lower() in [g.lower() for g in item_genres]:
                    content_type_detected = 'movie' if 'title' in item else 'tv'
                    content = ContentService.save_content_from_tmdb(item, content_type_detected)
                    if content:
                        recommendations.append(content)
                        
                    if len(recommendations) >= limit * 2:  # Get extra for filtering
                        break
            
            # Prioritize Telugu and English
            recommendations = RecommendationEngine._prioritize_languages(recommendations, ['telugu', 'english'])
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting popular by genre: {e}")
            return []
    
    @staticmethod
    def get_regional_recommendations(language, limit=20):
        try:
            # Map language to TMDB language codes
            lang_map = {
                'hindi': 'hi',
                'telugu': 'te', 
                'tamil': 'ta',
                'malayalam': 'ml',
                'kannada': 'kn',
                'english': 'en'
            }
            
            lang_code = lang_map.get(language.lower(), 'hi')
            recommendations = []
            
            # Get content by language
            for page in range(1, 4):  # Get multiple pages
                regional_data = TMDBService.discover_regional_content(lang_code, page)
                if regional_data:
                    for item in regional_data.get('results', []):
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content:
                            recommendations.append(content)
                        
                        if len(recommendations) >= limit:
                            break
                
                if len(recommendations) >= limit:
                    break
            
            # Also search for popular content with language keywords
            search_queries = {
                'hindi': ['bollywood latest', 'hindi movie 2024'],
                'telugu': ['tollywood latest', 'telugu movie 2024'],  
                'tamil': ['kollywood latest', 'tamil movie 2024'],
                'malayalam': ['malayalam movie 2024'],
                'kannada': ['kannada movie 2024'],
                'english': ['hollywood latest', 'english movie 2024']
            }
            
            for query in search_queries.get(language.lower(), []):
                search_results = TMDBService.search_content(query)
                if search_results:
                    for item in search_results.get('results', [])[:5]:
                        content_type_detected = 'movie' if 'title' in item else 'tv'
                        content = ContentService.save_content_from_tmdb(item, content_type_detected)
                        if content:
                            recommendations.append(content)
            
            # Remove duplicates and prioritize by rating
            seen_ids = set()
            unique_recommendations = []
            
            # Sort by rating and popularity
            recommendations.sort(key=lambda x: (x.rating or 0) + (x.popularity or 0)/100, reverse=True)
            
            for rec in recommendations:
                if rec.id not in seen_ids:
                    seen_ids.add(rec.id)
                    unique_recommendations.append(rec)
                    if len(unique_recommendations) >= limit:
                        break
            
            return unique_recommendations
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
    
    @staticmethod
    def get_new_releases(limit=20, days=60):
        """Get new releases from the last N days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Get from database first
            new_content = Content.query.filter(
                Content.release_date >= cutoff_date.date()
            ).order_by(Content.release_date.desc()).limit(limit).all()
            
            # If not enough, fetch from TMDB
            if len(new_content) < limit:
                # Get latest releases from TMDB
                url = f"{TMDBService.BASE_URL}/movie/now_playing"
                params = {
                    'api_key': TMDB_API_KEY,
                    'region': 'IN'
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('results', []):
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content and content not in new_content:
                            new_content.append(content)
                            if len(new_content) >= limit:
                                break
            
            return RecommendationEngine._prioritize_languages(new_content, ['telugu', 'english'])[:limit]
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return []
    
    @staticmethod
    def get_critics_choice(limit=20):
        """Get critically acclaimed content"""
        try:
            # Get top rated content from database
            critics_choice = Content.query.filter(
                Content.rating >= 7.5,
                Content.vote_count >= 100
            ).order_by(Content.rating.desc()).limit(limit).all()
            
            # If not enough, get from TMDB
            if len(critics_choice) < limit:
                url = f"{TMDBService.BASE_URL}/movie/top_rated"
                params = {
                    'api_key': TMDB_API_KEY,
                    'region': 'IN'
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('results', []):
                        if item.get('vote_average', 0) >= 7.5:
                            content = ContentService.save_content_from_tmdb(item, 'movie')
                            if content and content not in critics_choice:
                                critics_choice.append(content)
                                if len(critics_choice) >= limit:
                                    break
            
            return RecommendationEngine._prioritize_languages(critics_choice, ['telugu', 'english'])[:limit]
        except Exception as e:
            logger.error(f"Error getting critics choice: {e}")
            return []
    
    @staticmethod
    def _prioritize_languages(content_list, priority_languages):
        """Prioritize content by language preference"""
        prioritized = []
        remaining = []
        
        for content in content_list:
            available_langs = json.loads(content.available_languages or '[]')
            
            # Check if content has priority languages
            has_priority = any(lang in available_langs for lang in priority_languages)
            
            if has_priority:
                prioritized.append(content)
            else:
                remaining.append(content)
        
        return prioritized + remaining

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
                
                # Extract preferred genres and languages
                all_genres = []
                all_languages = []
                
                for content in viewed_contents:
                    if content.genres:
                        all_genres.extend(json.loads(content.genres))
                    if content.available_languages:
                        all_languages.extend(json.loads(content.available_languages))
                
                # Get most common preferences
                genre_counts = Counter(all_genres)
                lang_counts = Counter(all_languages)
                
                top_genres = [genre for genre, _ in genre_counts.most_common(3)]
                top_languages = [lang for lang, _ in lang_counts.most_common(2)]
                
                # Get recommendations based on preferences
                for genre in top_genres:
                    genre_recs = RecommendationEngine.get_popular_by_genre(genre, limit=5)
                    recommendations.extend(genre_recs)
                
                for language in top_languages:
                    lang_recs = RecommendationEngine.get_regional_recommendations(language, limit=5)
                    recommendations.extend(lang_recs)
            
            # Add regional content based on location
            if location and location.get('country') == 'India':
                # Prioritize Telugu and Hindi for Indian users
                regional_recs = RecommendationEngine.get_regional_recommendations('telugu', limit=8)
                recommendations.extend(regional_recs)
                
                hindi_recs = RecommendationEngine.get_regional_recommendations('hindi', limit=5)
                recommendations.extend(hindi_recs)
            
            # Add trending content
            trending_recs = RecommendationEngine.get_trending_recommendations(limit=10)
            recommendations.extend(trending_recs)
            
            # Add new releases
            new_releases = RecommendationEngine.get_new_releases(limit=5)
            recommendations.extend(new_releases)
            
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
            
            # Get streaming platforms
            streaming_platforms = []
            if content.streaming_platforms:
                try:
                    streaming_platforms = json.loads(content.streaming_platforms)
                except:
                    streaming_platforms = []
            
            # Create language-specific watch links
            watch_links = TelegramService._create_watch_links(streaming_platforms)
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Find free platform
            free_platform = None
            for platform in streaming_platforms:
                if platform.get('is_free', False):
                    free_platform = platform['platform_name']
                    break
            
            # Create message
            free_text = f"\n🎬 Free on {free_platform}!" if free_platform else ""
            
            message = f"""**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}{free_text}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

{watch_links}

For More - http://recommendationwebsite.com

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
    def _create_watch_links(streaming_platforms):
        """Create language-specific watch links for Telegram"""
        if not streaming_platforms:
            return "🎯 Watch links not available"
        
        # Group platforms by language
        language_platforms = defaultdict(list)
        
        for platform in streaming_platforms:
            platform_languages = platform.get('languages', ['english'])
            for lang in platform_languages:
                language_platforms[lang].append(platform)
        
        # Create formatted links
        links_text = "🎯 Choose Your Language to Watch:\n"
        
        for language, platforms in language_platforms.items():
            lang_info = LANGUAGE_MAPPING.get(language, {'name': language.title(), 'flag': '🎬'})
            
            # Get best platform for this language (prefer free)
            best_platform = None
            for platform in platforms:
                if platform.get('is_free', False):
                    best_platform = platform
                    break
            
            if not best_platform and platforms:
                best_platform = platforms[0]
            
            if best_platform:
                free_indicator = " (Free)" if best_platform.get('is_free', False) else ""
                platform_name = best_platform.get('platform_name', 'Watch')
                
                links_text += f"[{lang_info['flag']} {lang_info['name']}{free_indicator}] "
        
        # Add trailer link
        links_text += "\n[📺 Watch Trailer]"
        
        return links_text

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
        
        session_id = get_session_id()
        results = []
        
        # Search TMDB
        if content_type in ['movie', 'tv', 'multi']:
            tmdb_results = TMDBService.search_content(query, content_type, page=page)
            
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
                        
                        # Get streaming platforms
                        streaming_platforms = json.loads(content.streaming_platforms or '[]')
                        
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
                            'available_languages': json.loads(content.available_languages or '[]'),
                            'streaming_platforms': streaming_platforms
                        })
        
        # Search anime if requested
        if content_type in ['anime', 'multi']:
            anime_results = JikanService.search_anime(query, page=page)
            
            if anime_results:
                for anime in anime_results.get('data', []):
                    content = ContentService.save_anime_content(anime)
                    if content:
                        streaming_platforms = json.loads(content.streaming_platforms or '[]')
                        
                        results.append({
                            'id': content.id,
                            'mal_id': content.mal_id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'release_date': content.release_date.isoformat() if content.release_date else None,
                            'poster_path': content.poster_path,
                            'overview': content.overview,
                            'available_languages': json.loads(content.available_languages or '[]'),
                            'streaming_platforms': streaming_platforms
                        })
        
        db.session.commit()
        
        return jsonify({
            'results': results,
            'total_results': len(results),
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
        
        # Get additional details from TMDB/Jikan if available
        additional_details = None
        if content.tmdb_id:
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        elif content.mal_id:
            additional_details = JikanService.get_anime_details(content.mal_id)
        
        # Get YouTube trailers
        trailers = []
        if YOUTUBE_API_KEY:
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
        
        # Get streaming platforms with language-specific information
        streaming_platforms = json.loads(content.streaming_platforms or '[]')
        
        # Organize streaming by language
        streaming_by_language = {}
        available_languages = json.loads(content.available_languages or '[]')
        
        for language in available_languages:
            streaming_by_language[language] = []
            
            for platform in streaming_platforms:
                platform_languages = platform.get('languages', [])
                if language in platform_languages or not platform_languages:
                    streaming_by_language[language].append({
                        'platform_name': platform.get('platform_name'),
                        'is_free': platform.get('is_free', False),
                        'link': platform.get('link'),
                        'button_text': f"🔘 Watch in {LANGUAGE_MAPPING.get(language, {'name': language.title()})['name']}"
                    })
        
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
            'available_languages': available_languages,
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'runtime': content.runtime,
            'rating': content.rating,
            'vote_count': content.vote_count,
            'overview': content.overview,
            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path else None,
            'streaming_platforms': streaming_platforms,
            'streaming_by_language': streaming_by_language,
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Enhanced Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = []
        for content in recommendations:
            streaming_platforms = json.loads(content.streaming_platforms or '[]')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'available_languages': json.loads(content.available_languages or '[]'),
                'streaming_platforms': streaming_platforms
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/recommendations/new-releases', methods=['GET'])
def get_new_releases():
    try:
        limit = int(request.args.get('limit', 20))
        days = int(request.args.get('days', 60))
        
        recommendations = RecommendationEngine.get_new_releases(limit, days)
        
        result = []
        for content in recommendations:
            streaming_platforms = json.loads(content.streaming_platforms or '[]')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'available_languages': json.loads(content.available_languages or '[]'),
                'streaming_platforms': streaming_platforms
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"New releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

@app.route('/api/recommendations/critics-choice', methods=['GET'])
def get_critics_choice():
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_critics_choice(limit)
        
        result = []
        for content in recommendations:
            streaming_platforms = json.loads(content.streaming_platforms or '[]')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'available_languages': json.loads(content.available_languages or '[]'),
                'streaming_platforms': streaming_platforms
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Critics choice error: {e}")
        return jsonify({'error': 'Failed to get critics choice'}), 500

@app.route('/api/recommendations/popular/<genre>', methods=['GET'])
def get_popular_by_genre(genre):
    try:
        limit = int(request.args.get('limit', 20))
        region = request.args.get('region', 'IN')
        
        recommendations = RecommendationEngine.get_popular_by_genre(genre, limit, region)
        
        result = []
        for content in recommendations:
            streaming_platforms = json.loads(content.streaming_platforms or '[]')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'available_languages': json.loads(content.available_languages or '[]'),
                'streaming_platforms': streaming_platforms
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
            streaming_platforms = json.loads(content.streaming_platforms or '[]')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'available_languages': json.loads(content.available_languages or '[]'),
                'streaming_platforms': streaming_platforms
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
            streaming_platforms = json.loads(content.streaming_platforms or '[]')
            result.append({
                'id': content.id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'available_languages': json.loads(content.available_languages or '[]'),
                'streaming_platforms': streaming_platforms
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
            streaming_platforms = json.loads(content.streaming_platforms or '[]')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'available_languages': json.loads(content.available_languages or '[]'),
                'streaming_platforms': streaming_platforms
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
            response = requests.post(f"{ML_SERVICE_URL}/api/recommendations", json=user_data, timeout=30)
            
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
                        streaming_platforms = json.loads(content.streaming_platforms or '[]')
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'available_languages': json.loads(content.available_languages or '[]'),
                            'streaming_platforms': streaming_platforms,
                            'recommendation_score': rec.get('score', 0),
                            'recommendation_reason': rec.get('reason', '')
                        })
                
                return jsonify({'recommendations': result}), 200
        except:
            pass
        
        # Fallback to basic recommendations with user preferences
        user_languages = json.loads(current_user.preferred_languages or '["telugu", "english"]')
        fallback_recommendations = []
        
        for language in user_languages:
            lang_recs = RecommendationEngine.get_regional_recommendations(language, 10)
            fallback_recommendations.extend(lang_recs)
        
        result = []
        for content in fallback_recommendations[:20]:
            streaming_platforms = json.loads(content.streaming_platforms or '[]')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'available_languages': json.loads(content.available_languages or '[]'),
                'streaming_platforms': streaming_platforms
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
            streaming_platforms = json.loads(content.streaming_platforms or '[]')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'available_languages': json.loads(content.available_languages or '[]'),
                'streaming_platforms': streaming_platforms
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
            streaming_platforms = json.loads(content.streaming_platforms or '[]')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'available_languages': json.loads(content.available_languages or '[]'),
                'streaming_platforms': streaming_platforms
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
        if data.get('source') == 'tmdb' and data.get('id'):
            existing_content = Content.query.filter_by(tmdb_id=data['id']).first()
        elif data.get('source') == 'anime' and data.get('id'):
            existing_content = Content.query.filter_by(mal_id=data['id']).first()
        
        if existing_content:
            return jsonify({
                'message': 'Content already exists',
                'content_id': existing_content.id
            }), 200
        
        # Create new content from external data
        try:
            if data.get('source') == 'tmdb':
                # Get full details from TMDB
                full_data = TMDBService.get_content_details(data['id'], data.get('content_type', 'movie'))
                if full_data:
                    content = ContentService.save_content_from_tmdb(full_data, data.get('content_type', 'movie'))
                else:
                    return jsonify({'error': 'Could not fetch full content details'}), 400
            
            elif data.get('source') == 'anime':
                # Get full details from Jikan
                full_data = JikanService.get_anime_details(data['id'])
                if full_data and 'data' in full_data:
                    content = ContentService.save_anime_content(full_data['data'])
                else:
                    return jsonify({'error': 'Could not fetch full anime details'}), 400
            
            else:
                return jsonify({'error': 'Unsupported content source'}), 400
            
            if content:
                return jsonify({
                    'message': 'Content saved successfully',
                    'content_id': content.id
                }), 201
            else:
                return jsonify({'error': 'Failed to save content'}), 500
            
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
            
            streaming_platforms = json.loads(content.streaming_platforms or '[]') if content else []
            
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
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'available_languages': json.loads(content.available_languages or '[]'),
                    'streaming_platforms': streaming_platforms
                } if content else None
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
        
        # Language preferences
        language_counts = defaultdict(int)
        for interaction in all_interactions:
            content = Content.query.get(interaction.content_id)
            if content and content.available_languages:
                languages = json.loads(content.available_languages)
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
                {'language': language.title(), 'count': count}
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
                streaming_platforms = json.loads(content.streaming_platforms or '[]')
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'available_languages': json.loads(content.available_languages or '[]'),
                    'streaming_platforms': streaming_platforms,
                    'admin_description': rec.description,
                    'admin_name': admin.username if admin else 'Admin',
                    'recommended_at': rec.created_at.isoformat()
                })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Public admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get admin recommendations'}), 500

# Utility Routes
@app.route('/api/genres', methods=['GET'])
def get_genres():
    """Get list of available genres"""
    return jsonify({'genres': GENRE_CATEGORIES}), 200

@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get list of supported languages"""
    languages = []
    for lang_code, lang_info in LANGUAGE_MAPPING.items():
        languages.append({
            'code': lang_code,
            'name': lang_info['name'],
            'flag': lang_info['flag']
        })
    
    return jsonify({'languages': languages}), 200

@app.route('/api/platforms', methods=['GET'])
def get_streaming_platforms():
    """Get list of streaming platforms"""
    platforms = []
    for platform_id, platform_info in STREAMING_PLATFORMS.items():
        platforms.append({
            'id': platform_id,
            'name': platform_info['name'],
            'is_free': platform_info['is_free'],
            'url': platform_info['url']
        })
    
    return jsonify({'platforms': platforms}), 200

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'features': {
            'streaming_integration': True,
            'multi_language_support': True,
            'anime_support': True,
            'telegram_integration': True,
            'regional_content': True
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
                    preferred_genres=json.dumps(['Action', 'Drama', 'Comedy'])
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