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
    streaming_links = db.Column(db.Text)  # JSON string for language-specific streaming links
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

# Enhanced OTT Platform Information with streaming support
OTT_PLATFORMS = {
    # Free Platforms
    'mx_player': {
        'name': 'MX Player', 
        'is_free': True, 
        'url': 'https://mxplayer.in',
        'supported_languages': ['hindi', 'english', 'telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://www.mxplayer.in/assets/images/logo.png'
    },
    'jio_hotstar': {
        'name': 'JioHotstar', 
        'is_free': True, 
        'url': 'https://jiocinema.com',
        'supported_languages': ['hindi', 'english', 'telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://jiocinema.com/logo.png'
    },
    'sony_liv_free': {
        'name': 'SonyLIV Free', 
        'is_free': True, 
        'url': 'https://sonyliv.com',
        'supported_languages': ['hindi', 'english', 'telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://sonyliv.com/logo.png'
    },
    'zee5_free': {
        'name': 'ZEE5 Free', 
        'is_free': True, 
        'url': 'https://zee5.com',
        'supported_languages': ['hindi', 'english', 'telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://zee5.com/logo.png'
    },
    'youtube': {
        'name': 'YouTube', 
        'is_free': True, 
        'url': 'https://youtube.com',
        'supported_languages': ['hindi', 'english', 'telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://youtube.com/logo.png'
    },
    'crunchyroll_free': {
        'name': 'Crunchyroll Free', 
        'is_free': True, 
        'url': 'https://crunchyroll.com',
        'supported_languages': ['english', 'japanese'],
        'logo': 'https://crunchyroll.com/logo.png'
    },
    'airtel_xstream': {
        'name': 'Airtel Xstream', 
        'is_free': True, 
        'url': 'https://airtelxstream.in',
        'supported_languages': ['hindi', 'english', 'telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://airtelxstream.in/logo.png'
    },
    
    # Paid Platforms
    'netflix': {
        'name': 'Netflix', 
        'is_free': False, 
        'url': 'https://netflix.com',
        'supported_languages': ['hindi', 'english', 'telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://netflix.com/logo.png'
    },
    'amazon_prime': {
        'name': 'Amazon Prime Video', 
        'is_free': False, 
        'url': 'https://primevideo.com',
        'supported_languages': ['hindi', 'english', 'telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://primevideo.com/logo.png'
    },
    'disney_plus_hotstar': {
        'name': 'Disney+ Hotstar', 
        'is_free': False, 
        'url': 'https://hotstar.com',
        'supported_languages': ['hindi', 'english', 'telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://hotstar.com/logo.png'
    },
    'zee5_premium': {
        'name': 'ZEE5 Premium', 
        'is_free': False, 
        'url': 'https://zee5.com',
        'supported_languages': ['hindi', 'english', 'telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://zee5.com/logo.png'
    },
    'sony_liv_premium': {
        'name': 'SonyLIV Premium', 
        'is_free': False, 
        'url': 'https://sonyliv.com',
        'supported_languages': ['hindi', 'english', 'telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://sonyliv.com/logo.png'
    },
    'aha': {
        'name': 'Aha', 
        'is_free': False, 
        'url': 'https://aha.video',
        'supported_languages': ['telugu', 'tamil'],
        'logo': 'https://aha.video/logo.png'
    },
    'sun_nxt': {
        'name': 'Sun NXT', 
        'is_free': False, 
        'url': 'https://sunnxt.com',
        'supported_languages': ['telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://sunnxt.com/logo.png'
    }
}

# Regional Language Mapping with priority
REGIONAL_LANGUAGES = {
    'telugu': {'code': 'te', 'priority': 1, 'tmdb_code': 'te', 'names': ['telugu', 'tollywood']},
    'english': {'code': 'en', 'priority': 1, 'tmdb_code': 'en', 'names': ['english', 'hollywood']},
    'hindi': {'code': 'hi', 'priority': 2, 'tmdb_code': 'hi', 'names': ['hindi', 'bollywood']},
    'tamil': {'code': 'ta', 'priority': 3, 'tmdb_code': 'ta', 'names': ['tamil', 'kollywood']},
    'malayalam': {'code': 'ml', 'priority': 4, 'tmdb_code': 'ml', 'names': ['malayalam', 'mollywood']},
    'kannada': {'code': 'kn', 'priority': 5, 'tmdb_code': 'kn', 'names': ['kannada', 'sandalwood']}
}

# Genre mapping for better recommendations
GENRE_CATEGORIES = {
    'Action': ['Action', 'Adventure', 'Thriller'],
    'Comedy': ['Comedy', 'Family'],
    'Drama': ['Drama', 'Biography'],
    'Horror': ['Horror', 'Thriller'],
    'Romance': ['Romance', 'Drama'],
    'Sci-Fi': ['Science Fiction', 'Fantasy'],
    'Crime': ['Crime', 'Mystery', 'Thriller'],
    'Animation': ['Animation', 'Family'],
    'Documentary': ['Documentary'],
    'Musical': ['Music', 'Musical'],
    'Western': ['Western']
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
        # Simple IP-based location detection
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

# Streaming Availability Service
class StreamingAvailabilityService:
    @staticmethod
    def get_streaming_availability(title, imdb_id=None, tmdb_id=None):
        """Get streaming availability using multiple APIs"""
        streaming_data = {}
        
        # Try RapidAPI Streaming Availability first
        rapidapi_data = StreamingAvailabilityService.get_rapidapi_streaming(title, imdb_id)
        if rapidapi_data:
            streaming_data.update(rapidapi_data)
        
        # Try Watchmode API as backup
        watchmode_data = StreamingAvailabilityService.get_watchmode_streaming(title, imdb_id)
        if watchmode_data:
            streaming_data.update(watchmode_data)
        
        # Fallback to content-based detection
        if not streaming_data:
            streaming_data = StreamingAvailabilityService.get_fallback_streaming(title)
        
        return streaming_data
    
    @staticmethod
    def get_rapidapi_streaming(title, imdb_id=None):
        """Get streaming data from RapidAPI Streaming Availability"""
        try:
            headers = {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': "streaming-availability.p.rapidapi.com"
            }
            
            # Search by title first
            search_url = "https://streaming-availability.p.rapidapi.com/search/title"
            params = {
                'title': title,
                'country': 'in',  # India
                'show_type': 'all',
                'output_language': 'en'
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                streaming_links = {}
                
                if data.get('result'):
                    for item in data['result']:
                        if item.get('streamingInfo'):
                            for country, services in item['streamingInfo'].items():
                                if country == 'in':  # India
                                    for service_name, service_data in services.items():
                                        platform_name = StreamingAvailabilityService.map_service_name(service_name)
                                        if platform_name:
                                            # Check for different audio languages
                                            for service_info in service_data:
                                                audio_lang = service_info.get('audios', [])
                                                if audio_lang:
                                                    for lang in audio_lang:
                                                        lang_code = lang.get('language', 'en')
                                                        lang_name = StreamingAvailabilityService.get_language_name(lang_code)
                                                        
                                                        if lang_name not in streaming_links:
                                                            streaming_links[lang_name] = []
                                                        
                                                        streaming_links[lang_name].append({
                                                            'platform': platform_name,
                                                            'url': service_info.get('link', ''),
                                                            'quality': service_info.get('quality', 'HD'),
                                                            'is_free': StreamingAvailabilityService.is_free_service(service_name),
                                                            'type': service_info.get('streamingType', 'subscription')
                                                        })
                
                return streaming_links
            
        except Exception as e:
            logger.error(f"RapidAPI streaming error: {e}")
        
        return {}
    
    @staticmethod
    def get_watchmode_streaming(title, imdb_id=None):
        """Get streaming data from Watchmode API"""
        try:
            if not WATCHMODE_API_KEY or WATCHMODE_API_KEY == 'your_watchmode_api_key':
                return {}
            
            # Search for title
            search_url = "https://api.watchmode.com/v1/search/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'search_field': 'name',
                'search_value': title
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                search_data = response.json()
                
                if search_data.get('title_results'):
                    title_id = search_data['title_results'][0]['id']
                    
                    # Get streaming sources
                    sources_url = f"https://api.watchmode.com/v1/title/{title_id}/sources/"
                    sources_params = {
                        'apiKey': WATCHMODE_API_KEY,
                        'regions': 'IN'  # India
                    }
                    
                    sources_response = requests.get(sources_url, params=sources_params, timeout=10)
                    
                    if sources_response.status_code == 200:
                        sources_data = sources_response.json()
                        streaming_links = {}
                        
                        for source in sources_data:
                            platform_name = StreamingAvailabilityService.map_watchmode_service(source.get('name', ''))
                            if platform_name:
                                # Default to English if no language specified
                                lang_name = 'english'
                                
                                if lang_name not in streaming_links:
                                    streaming_links[lang_name] = []
                                
                                streaming_links[lang_name].append({
                                    'platform': platform_name,
                                    'url': source.get('web_url', ''),
                                    'quality': 'HD',
                                    'is_free': source.get('type', '') == 'free',
                                    'type': source.get('type', 'subscription')
                                })
                        
                        return streaming_links
            
        except Exception as e:
            logger.error(f"Watchmode API error: {e}")
        
        return {}
    
    @staticmethod
    def get_fallback_streaming(title):
        """Fallback streaming detection based on content patterns"""
        streaming_links = {}
        
        # Basic language detection based on title patterns
        languages_to_check = ['hindi', 'telugu', 'tamil', 'english', 'kannada', 'malayalam']
        
        for lang in languages_to_check:
            streaming_links[lang] = []
            
            # Add common platforms based on language
            if lang in ['hindi', 'english']:
                streaming_links[lang].extend([
                    {
                        'platform': 'netflix',
                        'url': f'https://netflix.com/search?q={title.replace(" ", "%20")}',
                        'quality': 'HD',
                        'is_free': False,
                        'type': 'subscription'
                    },
                    {
                        'platform': 'amazon_prime',
                        'url': f'https://primevideo.com/search/ref=atv_nb_sr?phrase={title.replace(" ", "%20")}',
                        'quality': 'HD',
                        'is_free': False,
                        'type': 'subscription'
                    },
                    {
                        'platform': 'youtube',
                        'url': f'https://youtube.com/results?search_query={title.replace(" ", "+")}+full+movie',
                        'quality': 'HD',
                        'is_free': True,
                        'type': 'free'
                    }
                ])
            
            if lang in ['telugu', 'tamil']:
                streaming_links[lang].extend([
                    {
                        'platform': 'aha',
                        'url': f'https://aha.video/search?query={title.replace(" ", "%20")}',
                        'quality': 'HD',
                        'is_free': False,
                        'type': 'subscription'
                    },
                    {
                        'platform': 'sun_nxt',
                        'url': f'https://sunnxt.com/search?q={title.replace(" ", "%20")}',
                        'quality': 'HD',
                        'is_free': False,
                        'type': 'subscription'
                    }
                ])
            
            # Add free platforms for all languages
            streaming_links[lang].extend([
                {
                    'platform': 'mx_player',
                    'url': f'https://mxplayer.in/search?query={title.replace(" ", "%20")}',
                    'quality': 'HD',
                    'is_free': True,
                    'type': 'free'
                },
                {
                    'platform': 'jio_hotstar',
                    'url': f'https://jiocinema.com/search/{title.replace(" ", "-")}',
                    'quality': 'HD',
                    'is_free': True,
                    'type': 'free'
                }
            ])
        
        return streaming_links
    
    @staticmethod
    def map_service_name(service_name):
        """Map API service names to our platform names"""
        service_mapping = {
            'netflix': 'netflix',
            'prime': 'amazon_prime',
            'hotstar': 'disney_plus_hotstar',
            'zee5': 'zee5_premium',
            'sonyliv': 'sony_liv_premium',
            'youtube': 'youtube',
            'mxplayer': 'mx_player',
            'jiocinema': 'jio_hotstar',
            'aha': 'aha',
            'sunnxt': 'sun_nxt'
        }
        
        service_lower = service_name.lower()
        for key, value in service_mapping.items():
            if key in service_lower:
                return value
        
        return service_name
    
    @staticmethod
    def map_watchmode_service(service_name):
        """Map Watchmode service names to our platform names"""
        watchmode_mapping = {
            'Netflix': 'netflix',
            'Amazon Prime Video': 'amazon_prime',
            'Disney+ Hotstar': 'disney_plus_hotstar',
            'ZEE5': 'zee5_premium',
            'SonyLIV': 'sony_liv_premium',
            'YouTube': 'youtube',
            'MX Player': 'mx_player',
            'JioCinema': 'jio_hotstar',
            'Aha': 'aha',
            'Sun NXT': 'sun_nxt'
        }
        
        return watchmode_mapping.get(service_name, service_name.lower().replace(' ', '_'))
    
    @staticmethod
    def get_language_name(lang_code):
        """Convert language codes to readable names"""
        code_mapping = {
            'hi': 'hindi',
            'en': 'english', 
            'te': 'telugu',
            'ta': 'tamil',
            'kn': 'kannada',
            'ml': 'malayalam',
            'ja': 'japanese',
            'ko': 'korean'
        }
        
        return code_mapping.get(lang_code, 'english')
    
    @staticmethod
    def is_free_service(service_name):
        """Check if a service is free"""
        free_services = ['youtube', 'mx_player', 'jio_hotstar', 'crunchyroll_free', 'airtel_xstream']
        service_lower = service_name.lower()
        
        for free_service in free_services:
            if free_service.replace('_', '').replace(' ', '') in service_lower.replace('_', '').replace(' ', ''):
                return True
        
        return False

# External API Services
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
    def get_now_playing(content_type='movie', page=1, region=None):
        url = f"{TMDBService.BASE_URL}/{content_type}/now_playing"
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
            logger.error(f"TMDB now playing error: {e}")
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
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB top rated error: {e}")
        return None
    
    @staticmethod
    def discover_content(content_type='movie', **kwargs):
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            **kwargs
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
        except Exception as e:
            logger.error(f"Jikan search error: {e}")
        return None
    
    @staticmethod
    def get_anime_details(anime_id):
        url = f"{JikanService.BASE_URL}/anime/{anime_id}/full"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
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

# Enhanced Content Management Service
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            # Check if content already exists
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                # Update streaming links if they don't exist
                if not existing.streaming_links:
                    streaming_links = StreamingAvailabilityService.get_streaming_availability(
                        existing.title, 
                        existing.imdb_id, 
                        existing.tmdb_id
                    )
                    existing.streaming_links = json.dumps(streaming_links)
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
            imdb_id = tmdb_data.get('imdb_id')
            if not imdb_id and tmdb_data.get('external_ids'):
                imdb_id = tmdb_data['external_ids'].get('imdb_id')
            
            # Get streaming availability
            title = tmdb_data.get('title') or tmdb_data.get('name')
            streaming_links = StreamingAvailabilityService.get_streaming_availability(
                title, imdb_id, tmdb_data['id']
            )
            
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
                streaming_links=json.dumps(streaming_links),
                ott_platforms=json.dumps([])  # Will be populated from streaming_links
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
            # Create anime content
            streaming_links = StreamingAvailabilityService.get_streaming_availability(
                anime_data.get('title'), None, None
            )
            
            content = Content(
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps([genre['name'] for genre in anime_data.get('genres', [])]),
                languages=json.dumps(['japanese']),
                release_date=datetime.strptime(anime_data.get('aired', {}).get('from', '1900-01-01'), '%Y-%m-%d').date() if anime_data.get('aired', {}).get('from') else None,
                rating=anime_data.get('score'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                streaming_links=json.dumps(streaming_links),
                ott_platforms=json.dumps([])
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

# Enhanced Recommendation Engine
class RecommendationEngine:
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all', language=None):
        try:
            # Get trending from TMDB
            trending_data = TMDBService.get_trending(content_type=content_type)
            if not trending_data:
                return []
            
            recommendations = []
            for item in trending_data.get('results', [])[:limit]:
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    # Filter by language if specified
                    if language:
                        content_languages = json.loads(content.languages or '[]')
                        if not any(lang.lower() in [l.lower() for l in content_languages] for lang in REGIONAL_LANGUAGES[language]['names']):
                            continue
                    
                    recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_new_releases(limit=20, language=None, days_back=30):
        try:
            # Get recent releases
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Discover new movie releases
            movie_data = TMDBService.discover_content(
                content_type='movie',
                primary_release_date_gte=start_date.strftime('%Y-%m-%d'),
                primary_release_date_lte=end_date.strftime('%Y-%m-%d'),
                sort_by='popularity.desc',
                page=1
            )
            
            # Discover new TV releases
            tv_data = TMDBService.discover_content(
                content_type='tv',
                first_air_date_gte=start_date.strftime('%Y-%m-%d'),
                first_air_date_lte=end_date.strftime('%Y-%m-%d'),
                sort_by='popularity.desc',
                page=1
            )
            
            recommendations = []
            
            # Process movies
            if movie_data:
                for item in movie_data.get('results', []):
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
            
            # Process TV shows
            if tv_data:
                for item in tv_data.get('results', []):
                    content = ContentService.save_content_from_tmdb(item, 'tv')
                    if content:
                        recommendations.append(content)
            
            # Filter by language if specified
            if language:
                filtered_recs = []
                for content in recommendations:
                    content_languages = json.loads(content.languages or '[]')
                    if any(lang.lower() in [l.lower() for l in content_languages] for lang in REGIONAL_LANGUAGES[language]['names']):
                        filtered_recs.append(content)
                recommendations = filtered_recs
            
            # Sort by popularity and limit
            recommendations.sort(key=lambda x: x.popularity or 0, reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return []
    
    @staticmethod
    def get_best_movies(limit=20, language=None):
        try:
            # Get top rated movies
            top_rated_data = TMDBService.get_top_rated('movie')
            
            recommendations = []
            
            if top_rated_data:
                for item in top_rated_data.get('results', []):
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
            
            # Filter by language if specified
            if language:
                filtered_recs = []
                for content in recommendations:
                    content_languages = json.loads(content.languages or '[]')
                    if any(lang.lower() in [l.lower() for l in content_languages] for lang in REGIONAL_LANGUAGES[language]['names']):
                        filtered_recs.append(content)
                recommendations = filtered_recs
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting best movies: {e}")
            return []
    
    @staticmethod
    def get_critics_choice(limit=20, language=None):
        try:
            # Get highly rated content with good vote counts
            movie_data = TMDBService.discover_content(
                content_type='movie',
                sort_by='vote_average.desc',
                vote_count_gte=1000,
                page=1
            )
            
            tv_data = TMDBService.discover_content(
                content_type='tv',
                sort_by='vote_average.desc',
                vote_count_gte=500,
                page=1
            )
            
            recommendations = []
            
            # Process movies
            if movie_data:
                for item in movie_data.get('results', []):
                    if item.get('vote_average', 0) >= 7.5:  # High rating threshold
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content:
                            recommendations.append(content)
            
            # Process TV shows
            if tv_data:
                for item in tv_data.get('results', []):
                    if item.get('vote_average', 0) >= 7.5:  # High rating threshold
                        content = ContentService.save_content_from_tmdb(item, 'tv')
                        if content:
                            recommendations.append(content)
            
            # Filter by language if specified
            if language:
                filtered_recs = []
                for content in recommendations:
                    content_languages = json.loads(content.languages or '[]')
                    if any(lang.lower() in [l.lower() for l in content_languages] for lang in REGIONAL_LANGUAGES[language]['names']):
                        filtered_recs.append(content)
                recommendations = filtered_recs
            
            # Sort by rating
            recommendations.sort(key=lambda x: x.rating or 0, reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting critics choice: {e}")
            return []
    
    @staticmethod
    def get_popular_by_genre(genre, limit=20, region=None, language=None):
        try:
            # Get genre ID
            genre_ids = RecommendationEngine.get_genre_ids([genre])
            if not genre_ids:
                return []
            
            # Discover content by genre
            movie_data = TMDBService.discover_content(
                content_type='movie',
                with_genres=','.join(map(str, genre_ids)),
                sort_by='popularity.desc',
                page=1
            )
            
            tv_data = TMDBService.discover_content(
                content_type='tv',
                with_genres=','.join(map(str, genre_ids)),
                sort_by='popularity.desc',
                page=1
            )
            
            recommendations = []
            
            # Process movies
            if movie_data:
                for item in movie_data.get('results', []):
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
            
            # Process TV shows
            if tv_data:
                for item in tv_data.get('results', []):
                    content = ContentService.save_content_from_tmdb(item, 'tv')
                    if content:
                        recommendations.append(content)
            
            # Filter by language if specified
            if language:
                filtered_recs = []
                for content in recommendations:
                    content_languages = json.loads(content.languages or '[]')
                    if any(lang.lower() in [l.lower() for l in content_languages] for lang in REGIONAL_LANGUAGES[language]['names']):
                        filtered_recs.append(content)
                recommendations = filtered_recs
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting popular by genre: {e}")
            return []
    
    @staticmethod
    def get_regional_recommendations(language, limit=20):
        try:
            language_config = REGIONAL_LANGUAGES.get(language.lower(), {})
            language_code = language_config.get('tmdb_code', 'en')
            
            # Discover content in specific language
            movie_data = TMDBService.discover_content(
                content_type='movie',
                with_original_language=language_code,
                sort_by='popularity.desc',
                page=1
            )
            
            tv_data = TMDBService.discover_content(
                content_type='tv',
                with_original_language=language_code,
                sort_by='popularity.desc',
                page=1
            )
            
            recommendations = []
            
            # Process movies
            if movie_data:
                for item in movie_data.get('results', []):
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
            
            # Process TV shows
            if tv_data:
                for item in tv_data.get('results', []):
                    content = ContentService.save_content_from_tmdb(item, 'tv')
                    if content:
                        recommendations.append(content)
            
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
    
    @staticmethod
    def get_genre_ids(genre_names):
        """Get TMDB genre IDs from genre names"""
        genre_map = {
            'action': 28, 'adventure': 12, 'animation': 16, 'comedy': 35,
            'crime': 80, 'documentary': 99, 'drama': 18, 'family': 10751,
            'fantasy': 14, 'history': 36, 'horror': 27, 'music': 10402,
            'mystery': 9648, 'romance': 10749, 'science fiction': 878, 'sci-fi': 878,
            'tv movie': 10770, 'thriller': 53, 'war': 10752, 'western': 37
        }
        
        genre_ids = []
        for genre_name in genre_names:
            genre_id = genre_map.get(genre_name.lower())
            if genre_id:
                genre_ids.append(genre_id)
        
        return genre_ids

# Anonymous User Recommendations
class AnonymousRecommendationEngine:
    @staticmethod
    def get_recommendations_for_anonymous(session_id, ip_address, limit=20):
        try:
            # Get user location for regional content
            location = get_user_location(ip_address)
            
            # Get anonymous user's interaction history
            interactions = AnonymousInteraction.query.filter_by(session_id=session_id).all()
            
            recommendations = []
            
            # If user has interactions, recommend similar content
            if interactions:
                # Get genres from viewed content
                viewed_content_ids = [interaction.content_id for interaction in interactions]
                viewed_contents = Content.query.filter(Content.id.in_(viewed_content_ids)).all()
                
                # Extract preferred genres
                all_genres = []
                for content in viewed_contents:
                    if content.genres:
                        all_genres.extend(json.loads(content.genres))
                
                # Get most common genres
                genre_counts = Counter(all_genres)
                top_genres = [genre for genre, _ in genre_counts.most_common(3)]
                
                # Get recommendations based on top genres
                for genre in top_genres:
                    genre_recs = RecommendationEngine.get_popular_by_genre(genre, limit=7)
                    recommendations.extend(genre_recs)
            
            # Add regional content based on location (prioritize Telugu and English)
            if location and location.get('country') == 'India':
                # Prioritize Telugu content
                telugu_recs = RecommendationEngine.get_regional_recommendations('telugu', limit=8)
                recommendations.extend(telugu_recs)
                
                # Add English content
                english_recs = RecommendationEngine.get_regional_recommendations('english', limit=5)
                recommendations.extend(english_recs)
                
                # Add Hindi content
                hindi_recs = RecommendationEngine.get_regional_recommendations('hindi', limit=3)
                recommendations.extend(hindi_recs)
            
            # Add trending content
            trending_recs = RecommendationEngine.get_trending_recommendations(limit=8)
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
            
            # Get streaming links
            streaming_links = {}
            if content.streaming_links:
                try:
                    streaming_links = json.loads(content.streaming_links)
                except:
                    streaming_links = {}
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create streaming links text
            streaming_text = TelegramService.format_streaming_links(streaming_links)
            
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
    def format_streaming_links(streaming_links):
        """Format streaming links for Telegram message"""
        if not streaming_links:
            return "🔗 **Watch Online:** Not available yet"
        
        streaming_text = "🔗 **Watch Online:**\n"
        
        # Sort languages by priority
        language_priority = {
            'telugu': 1, 'english': 1, 'hindi': 2, 
            'tamil': 3, 'malayalam': 4, 'kannada': 5
        }
        
        sorted_languages = sorted(
            streaming_links.keys(), 
            key=lambda x: language_priority.get(x.lower(), 999)
        )
        
        for language in sorted_languages:
            links = streaming_links[language]
            if links:
                streaming_text += f"\n🔘 **{language.title()}:**\n"
                
                # Separate free and paid platforms
                free_links = [link for link in links if link.get('is_free', False)]
                paid_links = [link for link in links if not link.get('is_free', False)]
                
                # Show free platforms first
                if free_links:
                    streaming_text += "   🆓 Free: "
                    free_platforms = [link['platform'].replace('_', ' ').title() for link in free_links[:3]]
                    streaming_text += ", ".join(free_platforms) + "\n"
                
                # Show paid platforms
                if paid_links:
                    streaming_text += "   💰 Paid: "
                    paid_platforms = [link['platform'].replace('_', ' ').title() for link in paid_links[:3]]
                    streaming_text += ", ".join(paid_platforms) + "\n"
        
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
            preferred_languages=json.dumps(data.get('preferred_languages', [])),
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
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
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
        language = request.args.get('language', 'en')
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        # Record search interaction
        session_id = get_session_id()
        
        # Search TMDB
        tmdb_results = TMDBService.search_content(query, content_type, language=language, page=page)
        
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
                    
                    # Get streaming links
                    streaming_links = {}
                    if content.streaming_links:
                        try:
                            streaming_links = json.loads(content.streaming_links)
                        except:
                            streaming_links = {}
                    
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
                        'streaming_links': streaming_links
                    })
        
        # Add anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                # Save anime content
                content = ContentService.save_anime_content(anime)
                if content:
                    # Get streaming links
                    streaming_links = {}
                    if content.streaming_links:
                        try:
                            streaming_links = json.loads(content.streaming_links)
                        except:
                            streaming_links = {}
                    
                    results.append({
                        'id': content.id,
                        'title': anime.get('title'),
                        'content_type': 'anime',
                        'genres': [genre['name'] for genre in anime.get('genres', [])],
                        'rating': anime.get('score'),
                        'release_date': anime.get('aired', {}).get('from'),
                        'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                        'overview': anime.get('synopsis'),
                        'streaming_links': streaming_links
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
        
        # Get streaming links
        streaming_links = {}
        if content.streaming_links:
            try:
                streaming_links = json.loads(content.streaming_links)
            except:
                streaming_links = {}
        
        # If no streaming links, try to fetch them
        if not streaming_links and content.title:
            streaming_links = StreamingAvailabilityService.get_streaming_availability(
                content.title, content.imdb_id, content.tmdb_id
            )
            # Update content with streaming links
            content.streaming_links = json.dumps(streaming_links)
        
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
            'streaming_links': streaming_links,
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

@app.route('/api/anime/<int:anime_id>', methods=['GET'])
def get_anime_details(anime_id):
    try:
        # Try to get from our database first
        content = Content.query.filter_by(content_type='anime').filter(
            or_(Content.id == anime_id, Content.tmdb_id == anime_id)
        ).first()
        
        if not content:
            # Fetch from Jikan API
            anime_data = JikanService.get_anime_details(anime_id)
            if anime_data:
                content = ContentService.save_anime_content(anime_data['data'])
        
        if not content:
            return jsonify({'error': 'Anime not found'}), 404
        
        # Record view interaction
        session_id = get_session_id()
        interaction = AnonymousInteraction(
            session_id=session_id,
            content_id=content.id,
            interaction_type='view',
            ip_address=request.remote_addr
        )
        db.session.add(interaction)
        
        # Get streaming links
        streaming_links = {}
        if content.streaming_links:
            try:
                streaming_links = json.loads(content.streaming_links)
            except:
                streaming_links = {}
        
        db.session.commit()
        
        return jsonify({
            'id': content.id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'rating': content.rating,
            'overview': content.overview,
            'poster_path': content.poster_path,
            'streaming_links': streaming_links
        }), 200
        
    except Exception as e:
        logger.error(f"Anime details error: {e}")
        return jsonify({'error': 'Failed to get anime details'}), 500

# Enhanced Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        language = request.args.get('language')
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type, language)
        
        result = []
        for content in recommendations:
            # Get streaming links
            streaming_links = {}
            if content.streaming_links:
                try:
                    streaming_links = json.loads(content.streaming_links)
                except:
                    streaming_links = {}
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_links': streaming_links
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/recommendations/new-releases', methods=['GET'])
def get_new_releases():
    try:
        limit = int(request.args.get('limit', 20))
        language = request.args.get('language')
        days_back = int(request.args.get('days_back', 30))
        
        recommendations = RecommendationEngine.get_new_releases(limit, language, days_back)
        
        result = []
        for content in recommendations:
            # Get streaming links
            streaming_links = {}
            if content.streaming_links:
                try:
                    streaming_links = json.loads(content.streaming_links)
                except:
                    streaming_links = {}
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_links': streaming_links
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"New releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

@app.route('/api/recommendations/best-movies', methods=['GET'])
def get_best_movies():
    try:
        limit = int(request.args.get('limit', 20))
        language = request.args.get('language')
        
        recommendations = RecommendationEngine.get_best_movies(limit, language)
        
        result = []
        for content in recommendations:
            # Get streaming links
            streaming_links = {}
            if content.streaming_links:
                try:
                    streaming_links = json.loads(content.streaming_links)
                except:
                    streaming_links = {}
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_links': streaming_links
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Best movies error: {e}")
        return jsonify({'error': 'Failed to get best movies'}), 500

@app.route('/api/recommendations/critics-choice', methods=['GET'])
def get_critics_choice():
    try:
        limit = int(request.args.get('limit', 20))
        language = request.args.get('language')
        
        recommendations = RecommendationEngine.get_critics_choice(limit, language)
        
        result = []
        for content in recommendations:
            # Get streaming links
            streaming_links = {}
            if content.streaming_links:
                try:
                    streaming_links = json.loads(content.streaming_links)
                except:
                    streaming_links = {}
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_links': streaming_links
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Critics choice error: {e}")
        return jsonify({'error': 'Failed to get critics choice'}), 500

@app.route('/api/recommendations/popular/<genre>', methods=['GET'])
def get_popular_by_genre(genre):
    try:
        limit = int(request.args.get('limit', 20))
        region = request.args.get('region')
        language = request.args.get('language')
        
        recommendations = RecommendationEngine.get_popular_by_genre(genre, limit, region, language)
        
        result = []
        for content in recommendations:
            # Get streaming links
            streaming_links = {}
            if content.streaming_links:
                try:
                    streaming_links = json.loads(content.streaming_links)
                except:
                    streaming_links = {}
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_links': streaming_links
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
            # Get streaming links
            streaming_links = {}
            if content.streaming_links:
                try:
                    streaming_links = json.loads(content.streaming_links)
                except:
                    streaming_links = {}
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_links': streaming_links
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
            # Get streaming links
            streaming_links = {}
            if content.streaming_links:
                try:
                    streaming_links = json.loads(content.streaming_links)
                except:
                    streaming_links = {}
            
            result.append({
                'id': content.id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_links': streaming_links
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
            # Get streaming links
            streaming_links = {}
            if content.streaming_links:
                try:
                    streaming_links = json.loads(content.streaming_links)
                except:
                    streaming_links = {}
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_links': streaming_links
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
            'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
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
                        # Get streaming links
                        streaming_links = {}
                        if content.streaming_links:
                            try:
                                streaming_links = json.loads(content.streaming_links)
                            except:
                                streaming_links = {}
                        
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'streaming_links': streaming_links,
                            'recommendation_score': rec.get('score', 0),
                            'recommendation_reason': rec.get('reason', '')
                        })
                
                return jsonify({'recommendations': result}), 200
        except:
            pass
        
        # Fallback to basic recommendations
        return get_trending()
        
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
            # Get streaming links
            streaming_links = {}
            if content.streaming_links:
                try:
                    streaming_links = json.loads(content.streaming_links)
                except:
                    streaming_links = {}
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'streaming_links': streaming_links
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
            # Get streaming links
            streaming_links = {}
            if content.streaming_links:
                try:
                    streaming_links = json.loads(content.streaming_links)
                except:
                    streaming_links = {}
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'streaming_links': streaming_links
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
        if data.get('id'):
            # Check by TMDB ID or other external ID
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
                    release_date = datetime.strptime(data['release_date'], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            # Get streaming availability
            streaming_links = StreamingAvailabilityService.get_streaming_availability(
                data.get('title'), None, data.get('id')
            )
            
            # Create content object
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
                streaming_links=json.dumps(streaming_links),
                ott_platforms=json.dumps([])
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
            
            # Get streaming links
            streaming_links = {}
            if content and content.streaming_links:
                try:
                    streaming_links = json.loads(content.streaming_links)
                except:
                    streaming_links = {}
            
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
                    'streaming_links': streaming_links
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
                # Get streaming links
                streaming_links = {}
                if content.streaming_links:
                    try:
                        streaming_links = json.loads(content.streaming_links)
                    except:
                        streaming_links = {}
                
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'streaming_links': streaming_links,
                    'admin_description': rec.description,
                    'admin_name': admin.username if admin else 'Admin',
                    'recommended_at': rec.created_at.isoformat()
                })
        
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
        'version': '2.0.0',
        'features': [
            'streaming_availability',
            'multi_language_support',
            'enhanced_recommendations',
            'telegram_integration',
            'anime_support'
        ]
    }), 200

# Get available platforms
@app.route('/api/platforms', methods=['GET'])
def get_platforms():
    try:
        return jsonify({
            'platforms': OTT_PLATFORMS,
            'languages': REGIONAL_LANGUAGES
        }), 200
    except Exception as e:
        logger.error(f"Platforms error: {e}")
        return jsonify({'error': 'Failed to get platforms'}), 500

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