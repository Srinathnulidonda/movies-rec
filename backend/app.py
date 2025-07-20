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

# New API Keys for Streaming Availability
WATCHMODE_API_KEY = os.environ.get('WATCHMODE_API_KEY', 'WtcKDji9i20pjOl5Lg0AiyG2bddfUs3nSZRZJIsY')
RAPIDAPI_KEY = "c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37"
RAPIDAPI_HOST = "streaming-availability.p.rapidapi.com"

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
    ott_platforms = db.Column(db.Text)  # JSON string
    streaming_links = db.Column(db.Text)  # JSON string for language-specific links
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

# Enhanced OTT Platform Information with Regional Support
OTT_PLATFORMS = {
    # Free Platforms
    'mx_player': {
        'name': 'MX Player',
        'is_free': True,
        'url': 'https://mxplayer.com',
        'supports_languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'english'],
        'logo': 'https://img.mx-player.com/images/logos/mx-player-logo.png'
    },
    'jio_hotstar': {
        'name': 'JioHotstar',
        'is_free': True,
        'url': 'https://jiocinema.com',
        'supports_languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'english'],
        'logo': 'https://jiocinema.com/images/jio-cinema-logo.png'
    },
    'sonyliv_free': {
        'name': 'SonyLIV',
        'is_free': True,
        'url': 'https://sonyliv.com',
        'supports_languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'english'],
        'logo': 'https://sonyliv.com/images/sony-liv-logo.png'
    },
    'zee5_free': {
        'name': 'ZEE5',
        'is_free': True,
        'url': 'https://zee5.com',
        'supports_languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'english'],
        'logo': 'https://zee5.com/images/zee5-logo.png'
    },
    'youtube': {
        'name': 'YouTube',
        'is_free': True,
        'url': 'https://youtube.com',
        'supports_languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'english'],
        'logo': 'https://youtube.com/favicon.ico'
    },
    'crunchyroll': {
        'name': 'Crunchyroll',
        'is_free': True,
        'url': 'https://crunchyroll.com',
        'supports_languages': ['english', 'hindi'],
        'logo': 'https://crunchyroll.com/favicon.ico'
    },
    'airtel_xstream': {
        'name': 'Airtel Xstream',
        'is_free': True,
        'url': 'https://airtelxstream.in',
        'supports_languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'english'],
        'logo': 'https://airtelxstream.in/favicon.ico'
    },
    
    # Paid Platforms
    'netflix': {
        'name': 'Netflix',
        'is_free': False,
        'url': 'https://netflix.com',
        'supports_languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'english'],
        'logo': 'https://netflix.com/favicon.ico'
    },
    'amazon_prime': {
        'name': 'Amazon Prime Video',
        'is_free': False,
        'url': 'https://primevideo.com',
        'supports_languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'english'],
        'logo': 'https://primevideo.com/favicon.ico'
    },
    'disney_plus_hotstar': {
        'name': 'Disney+ Hotstar',
        'is_free': False,
        'url': 'https://hotstar.com',
        'supports_languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'english'],
        'logo': 'https://hotstar.com/favicon.ico'
    },
    'zee5_premium': {
        'name': 'ZEE5 Premium',
        'is_free': False,
        'url': 'https://zee5.com',
        'supports_languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'english'],
        'logo': 'https://zee5.com/images/zee5-logo.png'
    },
    'sonyliv_premium': {
        'name': 'SonyLIV Premium',
        'is_free': False,
        'url': 'https://sonyliv.com',
        'supports_languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'english'],
        'logo': 'https://sonyliv.com/images/sony-liv-logo.png'
    },
    'aha': {
        'name': 'Aha',
        'is_free': False,
        'url': 'https://aha.video',
        'supports_languages': ['telugu', 'tamil'],
        'logo': 'https://aha.video/favicon.ico'
    },
    'sun_nxt': {
        'name': 'Sun NXT',
        'is_free': False,
        'url': 'https://sunnxt.com',
        'supports_languages': ['telugu', 'tamil', 'kannada', 'malayalam'],
        'logo': 'https://sunnxt.com/favicon.ico'
    }
}

# Enhanced Regional Language Mapping with Priority
REGIONAL_LANGUAGES = {
    'telugu': {
        'codes': ['te', 'telugu', 'tollywood'],
        'priority': 1,
        'region': 'south_india',
        'search_terms': ['telugu movie', 'tollywood', 'telugu cinema', 'telugu film']
    },
    'english': {
        'codes': ['en', 'english', 'hollywood'],
        'priority': 1,
        'region': 'international',
        'search_terms': ['hollywood', 'english movie', 'english film']
    },
    'hindi': {
        'codes': ['hi', 'hindi', 'bollywood'],
        'priority': 2,
        'region': 'north_india',
        'search_terms': ['hindi movie', 'bollywood', 'hindi cinema', 'hindi film']
    },
    'tamil': {
        'codes': ['ta', 'tamil', 'kollywood'],
        'priority': 3,
        'region': 'south_india',
        'search_terms': ['tamil movie', 'kollywood', 'tamil cinema', 'tamil film']
    },
    'malayalam': {
        'codes': ['ml', 'malayalam', 'mollywood'],
        'priority': 4,
        'region': 'south_india',
        'search_terms': ['malayalam movie', 'mollywood', 'malayalam cinema', 'malayalam film']
    },
    'kannada': {
        'codes': ['kn', 'kannada', 'sandalwood'],
        'priority': 5,
        'region': 'south_india',
        'search_terms': ['kannada movie', 'sandalwood', 'kannada cinema', 'kannada film']
    }
}

# Genre Categories for Enhanced Recommendations
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

# Enhanced Streaming Availability Service
class StreamingAvailabilityService:
    @staticmethod
    def get_watchmode_availability(title, year=None):
        """Get streaming availability from Watchmode API"""
        try:
            if not WATCHMODE_API_KEY or WATCHMODE_API_KEY == 'your_watchmode_api_key':
                return []
            
            # Search for title
            search_url = f"https://api.watchmode.com/v1/search/"
            search_params = {
                'apiKey': WATCHMODE_API_KEY,
                'search_field': 'name',
                'search_value': title
            }
            if year:
                search_params['search_value'] += f" {year}"
            
            response = requests.get(search_url, params=search_params, timeout=10)
            if response.status_code != 200:
                return []
            
            search_results = response.json()
            if not search_results.get('title_results'):
                return []
            
            # Get details for first result
            title_id = search_results['title_results'][0]['id']
            details_url = f"https://api.watchmode.com/v1/title/{title_id}/details/"
            details_params = {'apiKey': WATCHMODE_API_KEY}
            
            details_response = requests.get(details_url, params=details_params, timeout=10)
            if details_response.status_code != 200:
                return []
            
            details = details_response.json()
            return StreamingAvailabilityService.parse_watchmode_sources(details.get('sources', []))
        
        except Exception as e:
            logger.error(f"Watchmode API error: {e}")
            return []
    
    @staticmethod
    def get_rapidapi_availability(title, country='IN'):
        """Get streaming availability from RapidAPI Streaming Availability"""
        try:
            url = f"https://{RAPIDAPI_HOST}/v2/search/title"
            headers = {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': RAPIDAPI_HOST
            }
            params = {
                'title': title,
                'country': country,
                'show_type': 'all',
                'output_language': 'en'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code != 200:
                return []
            
            data = response.json()
            if not data.get('result'):
                return []
            
            # Get streaming info for first result
            result = data['result'][0]
            return StreamingAvailabilityService.parse_rapidapi_sources(result.get('streamingInfo', {}))
        
        except Exception as e:
            logger.error(f"RapidAPI Streaming error: {e}")
            return []
    
    @staticmethod
    def parse_watchmode_sources(sources):
        """Parse Watchmode API sources"""
        streaming_links = []
        
        for source in sources:
            platform_name = source.get('name', '').lower()
            
            # Map to our platform IDs
            platform_mapping = {
                'netflix': 'netflix',
                'amazon prime video': 'amazon_prime',
                'disney plus': 'disney_plus_hotstar',
                'hotstar': 'disney_plus_hotstar',
                'sony liv': 'sonyliv_premium',
                'zee5': 'zee5_premium',
                'mx player': 'mx_player',
                'youtube': 'youtube',
                'jio cinema': 'jio_hotstar'
            }
            
            platform_id = None
            for key, value in platform_mapping.items():
                if key in platform_name:
                    platform_id = value
                    break
            
            if platform_id and platform_id in OTT_PLATFORMS:
                streaming_links.append({
                    'platform_id': platform_id,
                    'platform_name': OTT_PLATFORMS[platform_id]['name'],
                    'url': source.get('web_url', OTT_PLATFORMS[platform_id]['url']),
                    'is_free': OTT_PLATFORMS[platform_id]['is_free'],
                    'supported_languages': OTT_PLATFORMS[platform_id]['supports_languages'],
                    'logo': OTT_PLATFORMS[platform_id]['logo']
                })
        
        return streaming_links
    
    @staticmethod
    def parse_rapidapi_sources(streaming_info):
        """Parse RapidAPI Streaming Availability sources"""
        streaming_links = []
        
        for country, services in streaming_info.items():
            for service_id, service_data in services.items():
                # Map service IDs to our platform IDs
                service_mapping = {
                    'netflix': 'netflix',
                    'prime': 'amazon_prime',
                    'hotstar': 'disney_plus_hotstar',
                    'sonyliv': 'sonyliv_premium',
                    'zee5': 'zee5_premium',
                    'mx': 'mx_player',
                    'youtube': 'youtube',
                    'jiocinema': 'jio_hotstar'
                }
                
                platform_id = service_mapping.get(service_id.lower())
                
                if platform_id and platform_id in OTT_PLATFORMS:
                    streaming_links.append({
                        'platform_id': platform_id,
                        'platform_name': OTT_PLATFORMS[platform_id]['name'],
                        'url': service_data.get('link', OTT_PLATFORMS[platform_id]['url']),
                        'is_free': OTT_PLATFORMS[platform_id]['is_free'],
                        'supported_languages': OTT_PLATFORMS[platform_id]['supports_languages'],
                        'logo': OTT_PLATFORMS[platform_id]['logo'],
                        'available_languages': service_data.get('audios', [])
                    })
        
        return streaming_links
    
    @staticmethod
    def get_language_specific_links(title, year=None):
        """Get streaming links organized by language"""
        # Get all available streaming sources
        watchmode_links = StreamingAvailabilityService.get_watchmode_availability(title, year)
        rapidapi_links = StreamingAvailabilityService.get_rapidapi_availability(title)
        
        # Combine and deduplicate
        all_links = watchmode_links + rapidapi_links
        unique_links = {}
        
        for link in all_links:
            platform_id = link['platform_id']
            if platform_id not in unique_links:
                unique_links[platform_id] = link
        
        # Organize by language
        language_links = {}
        
        for lang_code, lang_info in REGIONAL_LANGUAGES.items():
            language_links[lang_code] = {
                'language': lang_code.title(),
                'priority': lang_info['priority'],
                'free_options': [],
                'paid_options': []
            }
            
            for link in unique_links.values():
                if lang_code in link['supported_languages']:
                    # Create language-specific streaming button
                    streaming_button = {
                        'platform_name': link['platform_name'],
                        'platform_id': link['platform_id'],
                        'url': link['url'],
                        'logo': link['logo'],
                        'button_text': f"🔘 Watch in {lang_code.title()} on {link['platform_name']}"
                    }
                    
                    if link['is_free']:
                        language_links[lang_code]['free_options'].append(streaming_button)
                    else:
                        language_links[lang_code]['paid_options'].append(streaming_button)
        
        return language_links

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
    
    @staticmethod
    def get_now_playing(page=1, region='IN'):
        """Get currently playing movies"""
        url = f"{TMDBService.BASE_URL}/movie/now_playing"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page,
            'region': region
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB now playing error: {e}")
        return None
    
    @staticmethod
    def get_top_rated(content_type='movie', page=1, region=None):
        """Get top rated content"""
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
    def discover_by_genre(genre_id, content_type='movie', page=1, language=None, region=None):
        """Discover content by genre"""
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': genre_id,
            'page': page,
            'sort_by': 'popularity.desc'
        }
        if language:
            params['with_original_language'] = language
        if region:
            params['region'] = region
        
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
        """Get detailed anime information"""
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
                return existing
            
            # Extract genres
            genres = []
            if 'genres' in tmdb_data:
                genres = [genre['name'] for genre in tmdb_data['genres']]
            elif 'genre_ids' in tmdb_data:
                genres = ContentService.map_genre_ids(tmdb_data['genre_ids'])
            
            # Extract languages
            languages = []
            available_languages = []
            if 'spoken_languages' in tmdb_data:
                languages = [lang['name'] for lang in tmdb_data['spoken_languages']]
                available_languages = [lang['iso_639_1'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
                available_languages = [tmdb_data['original_language']]
            
            # Get streaming availability
            title = tmdb_data.get('title') or tmdb_data.get('name')
            year = None
            if tmdb_data.get('release_date'):
                year = tmdb_data['release_date'][:4]
            elif tmdb_data.get('first_air_date'):
                year = tmdb_data['first_air_date'][:4]
            
            streaming_links = StreamingAvailabilityService.get_language_specific_links(title, year)
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
                title=title,
                original_title=tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                content_type=content_type,
                genres=json.dumps(genres),
                languages=json.dumps(languages),
                available_languages=json.dumps(available_languages),
                release_date=datetime.strptime(tmdb_data.get('release_date') or tmdb_data.get('first_air_date', '1900-01-01'), '%Y-%m-%d').date() if tmdb_data.get('release_date') or tmdb_data.get('first_air_date') else None,
                runtime=tmdb_data.get('runtime'),
                rating=tmdb_data.get('vote_average'),
                vote_count=tmdb_data.get('vote_count'),
                popularity=tmdb_data.get('popularity'),
                overview=tmdb_data.get('overview'),
                poster_path=tmdb_data.get('poster_path'),
                backdrop_path=tmdb_data.get('backdrop_path'),
                streaming_links=json.dumps(streaming_links)
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
        """Save anime content from Jikan API"""
        try:
            # Check if anime already exists
            existing = Content.query.filter_by(mal_id=anime_data['mal_id']).first()
            if existing:
                return existing
            
            # Extract anime genres
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            # Anime is typically in Japanese with subtitles
            languages = ['japanese']
            available_languages = ['ja', 'en']  # Japanese audio, English subtitles
            
            # Create anime content
            content = Content(
                mal_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                languages=json.dumps(languages),
                available_languages=json.dumps(available_languages),
                release_date=datetime.strptime(anime_data.get('aired', {}).get('from', '1900-01-01T00:00:00+00:00')[:10], '%Y-%m-%d').date() if anime_data.get('aired', {}).get('from') else None,
                runtime=anime_data.get('duration', {}).get('minutes') if isinstance(anime_data.get('duration'), dict) else None,
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('large_image_url'),
                backdrop_path=anime_data.get('images', {}).get('jpg', {}).get('large_image_url'),
                streaming_links=json.dumps({})  # Anime streaming would need separate service
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
            10759: 'Action & Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            10762: 'Kids', 9648: 'Mystery', 10763: 'News', 10764: 'Reality',
            10765: 'Sci-Fi & Fantasy', 10766: 'Soap', 10767: 'Talk',
            10768: 'War & Politics', 37: 'Western'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]

# Enhanced Recommendation Engine with Regional Priority
class RecommendationEngine:
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all', language_priority=None):
        """Get trending recommendations with language priority"""
        try:
            trending_data = TMDBService.get_trending(content_type=content_type)
            if not trending_data:
                return []
            
            recommendations = []
            prioritized_items = []
            other_items = []
            
            for item in trending_data.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                
                # Check language priority
                original_language = item.get('original_language', '')
                is_priority_language = False
                
                if language_priority:
                    for lang in language_priority:
                        if lang in REGIONAL_LANGUAGES:
                            if original_language in REGIONAL_LANGUAGES[lang]['codes']:
                                is_priority_language = True
                                break
                
                if is_priority_language:
                    prioritized_items.append((item, content_type_detected))
                else:
                    other_items.append((item, content_type_detected))
            
            # Process prioritized items first
            for item, content_type_detected in prioritized_items[:limit//2]:
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    recommendations.append(content)
            
            # Fill remaining with other items
            remaining_limit = limit - len(recommendations)
            for item, content_type_detected in other_items[:remaining_limit]:
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_new_releases(limit=20, language=None, days=60):
        """Get new releases in the last N days"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Get recent movies
            url = f"{TMDBService.BASE_URL}/discover/movie"
            params = {
                'api_key': TMDB_API_KEY,
                'primary_release_date.gte': start_date.isoformat(),
                'primary_release_date.lte': end_date.isoformat(),
                'sort_by': 'popularity.desc'
            }
            
            if language and language in REGIONAL_LANGUAGES:
                # Use first language code for the search
                lang_code = REGIONAL_LANGUAGES[language]['codes'][0]
                if len(lang_code) == 2:  # ISO language code
                    params['with_original_language'] = lang_code
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return []
            
            data = response.json()
            recommendations = []
            
            for item in data.get('results', [])[:limit]:
                content = ContentService.save_content_from_tmdb(item, 'movie')
                if content:
                    recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return []
    
    @staticmethod
    def get_best_movies_all_time(limit=20, language=None):
        """Get all-time best movies"""
        try:
            top_rated_data = TMDBService.get_top_rated('movie')
            if not top_rated_data:
                return []
            
            recommendations = []
            
            for item in top_rated_data.get('results', []):
                # Filter by language if specified
                if language and language in REGIONAL_LANGUAGES:
                    original_language = item.get('original_language', '')
                    lang_codes = REGIONAL_LANGUAGES[language]['codes']
                    if not any(code == original_language for code in lang_codes if len(code) == 2):
                        continue
                
                content = ContentService.save_content_from_tmdb(item, 'movie')
                if content:
                    recommendations.append(content)
                    if len(recommendations) >= limit:
                        break
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting best movies: {e}")
            return []
    
    @staticmethod
    def get_critics_choice(limit=20, language=None):
        """Get critics' choice recommendations"""
        try:
            # Get top rated content with high vote counts (critics + audience)
            url = f"{TMDBService.BASE_URL}/discover/movie"
            params = {
                'api_key': TMDB_API_KEY,
                'sort_by': 'vote_average.desc',
                'vote_count.gte': 1000,  # Minimum votes for credibility
                'vote_average.gte': 7.5   # High rating threshold
            }
            
            if language and language in REGIONAL_LANGUAGES:
                lang_code = REGIONAL_LANGUAGES[language]['codes'][0]
                if len(lang_code) == 2:
                    params['with_original_language'] = lang_code
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return []
            
            data = response.json()
            recommendations = []
            
            for item in data.get('results', [])[:limit]:
                content = ContentService.save_content_from_tmdb(item, 'movie')
                if content:
                    recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting critics choice: {e}")
            return []
    
    @staticmethod
    def get_genre_recommendations(genre, limit=20, language=None):
        """Get recommendations by genre"""
        try:
            # Map genre name to ID
            genre_mapping = {
                'action': 28, 'adventure': 12, 'animation': 16, 'biography': 99,
                'comedy': 35, 'crime': 80, 'documentary': 99, 'drama': 18,
                'fantasy': 14, 'horror': 27, 'musical': 10402, 'mystery': 9648,
                'romance': 10749, 'sci-fi': 878, 'thriller': 53, 'western': 37
            }
            
            genre_id = genre_mapping.get(genre.lower())
            if not genre_id:
                return []
            
            discover_data = TMDBService.discover_by_genre(
                genre_id, 
                language=REGIONAL_LANGUAGES.get(language, {}).get('codes', [None])[0] if language else None
            )
            
            if not discover_data:
                return []
            
            recommendations = []
            for item in discover_data.get('results', [])[:limit]:
                content = ContentService.save_content_from_tmdb(item, 'movie')
                if content:
                    recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting genre recommendations: {e}")
            return []
    
    @staticmethod
    def get_regional_recommendations(language, limit=20):
        """Enhanced regional recommendations with multiple search strategies"""
        try:
            recommendations = []
            lang_info = REGIONAL_LANGUAGES.get(language.lower())
            if not lang_info:
                return []
            
            # Strategy 1: Search by language-specific terms
            for search_term in lang_info['search_terms'][:2]:  # Use first 2 terms
                search_results = TMDBService.search_content(search_term)
                if search_results:
                    for item in search_results.get('results', []):
                        content_type_detected = 'movie' if 'title' in item else 'tv'
                        content = ContentService.save_content_from_tmdb(item, content_type_detected)
                        if content:
                            recommendations.append(content)
                        
                        if len(recommendations) >= limit // 2:
                            break
                
                if len(recommendations) >= limit // 2:
                    break
            
            # Strategy 2: Discover by original language
            lang_code = lang_info['codes'][0]
            if len(lang_code) == 2:  # ISO language code
                discover_data = TMDBService.discover_by_genre(
                    genre_id=None, 
                    language=lang_code
                )
                
                if discover_data:
                    for item in discover_data.get('results', []):
                        if len(recommendations) >= limit:
                            break
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content:
                            recommendations.append(content)
            
            # Remove duplicates
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
                    genre_recs = RecommendationEngine.get_genre_recommendations(genre, limit=7)
                    recommendations.extend(genre_recs)
            
            # Add regional content based on location (prioritize Telugu and English)
            if location and location.get('country') == 'India':
                # Priority languages for Indian users
                priority_languages = ['telugu', 'english', 'hindi']
                for lang in priority_languages:
                    regional_recs = RecommendationEngine.get_regional_recommendations(lang, limit=5)
                    recommendations.extend(regional_recs)
            else:
                # Default to English for international users
                english_recs = RecommendationEngine.get_regional_recommendations('english', limit=10)
                recommendations.extend(english_recs)
            
            # Add trending content with language priority
            trending_recs = RecommendationEngine.get_trending_recommendations(
                limit=10, 
                language_priority=['telugu', 'english']
            )
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
            
            # Create message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

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
                    
                    # Get streaming links
                    streaming_links = json.loads(content.streaming_links or '{}')
                    
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
                # Save anime to database for consistency
                anime_content = ContentService.save_anime_content(anime)
                if anime_content:
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
                        'streaming_links': {}
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
        
        # Get additional details based on content type
        additional_details = None
        if content.content_type == 'anime' and content.mal_id:
            # Get anime details from Jikan
            additional_details = JikanService.get_anime_details(content.mal_id)
        elif content.tmdb_id:
            # Get movie/TV details from TMDB
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
        if additional_details:
            if content.content_type == 'anime':
                # For anime, get from same genre
                genres = json.loads(content.genres or '[]')
                if genres:
                    similar_anime = RecommendationEngine.get_anime_recommendations(limit=5)
                    for similar in similar_anime:
                        if similar.id != content.id:
                            similar_content.append({
                                'id': similar.id,
                                'title': similar.title,
                                'poster_path': similar.poster_path,
                                'rating': similar.rating
                            })
            else:
                # For movies/TV, get from TMDB similar
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
        
        # Get streaming links with language options
        streaming_links = json.loads(content.streaming_links or '{}')
        
        # Format response
        response_data = {
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'mal_id': content.mal_id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'available_languages': json.loads(content.available_languages or '[]'),
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'runtime': content.runtime,
            'rating': content.rating,
            'vote_count': content.vote_count,
            'overview': content.overview,
            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path else None,
            'streaming_links': streaming_links,
            'trailers': trailers,
            'similar_content': similar_content
        }
        
        # Add cast/crew for non-anime content
        if content.content_type != 'anime' and additional_details:
            response_data.update({
                'cast': additional_details.get('credits', {}).get('cast', [])[:10],
                'crew': additional_details.get('credits', {}).get('crew', [])[:5]
            })
        
        db.session.commit()
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Enhanced Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        language = request.args.get('language')
        
        language_priority = ['telugu', 'english'] if not language else [language]
        recommendations = RecommendationEngine.get_trending_recommendations(
            limit, content_type, language_priority
        )
        
        result = []
        for content in recommendations:
            streaming_links = json.loads(content.streaming_links or '{}')
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
        language = request.args.get('language', 'telugu')
        limit = int(request.args.get('limit', 20))
        days = int(request.args.get('days', 60))
        
        recommendations = RecommendationEngine.get_new_releases(limit, language, days)
        
        result = []
        for content in recommendations:
            streaming_links = json.loads(content.streaming_links or '{}')
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
        language = request.args.get('language')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_best_movies_all_time(limit, language)
        
        result = []
        for content in recommendations:
            streaming_links = json.loads(content.streaming_links or '{}')
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
        language = request.args.get('language')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_critics_choice(limit, language)
        
        result = []
        for content in recommendations:
            streaming_links = json.loads(content.streaming_links or '{}')
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

@app.route('/api/recommendations/genre/<genre>', methods=['GET'])
def get_genre_recommendations(genre):
    try:
        language = request.args.get('language')
        limit = int(request.args.get('limit', 20))
        
        if genre.lower() not in [g.lower() for g in GENRE_CATEGORIES]:
            return jsonify({'error': 'Invalid genre'}), 400
        
        recommendations = RecommendationEngine.get_genre_recommendations(genre, limit, language)
        
        result = []
        for content in recommendations:
            streaming_links = json.loads(content.streaming_links or '{}')
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
        logger.error(f"Genre recommendations error: {e}")
        return jsonify({'error': 'Failed to get genre recommendations'}), 500

@app.route('/api/recommendations/regional/<language>', methods=['GET'])
def get_regional(language):
    try:
        limit = int(request.args.get('limit', 20))
        
        if language.lower() not in REGIONAL_LANGUAGES:
            return jsonify({'error': 'Unsupported language'}), 400
        
        recommendations = RecommendationEngine.get_regional_recommendations(language, limit)
        
        result = []
        for content in recommendations:
            streaming_links = json.loads(content.streaming_links or '{}')
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
            result.append({
                'id': content.id,
                'mal_id': content.mal_id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_links': json.loads(content.streaming_links or '{}')
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
            streaming_links = json.loads(content.streaming_links or '{}')
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

# New endpoint for genres list
@app.route('/api/genres', methods=['GET'])
def get_genres():
    return jsonify({'genres': GENRE_CATEGORIES}), 200

# New endpoint for supported languages
@app.route('/api/languages', methods=['GET'])
def get_languages():
    languages = []
    for lang_code, lang_info in REGIONAL_LANGUAGES.items():
        languages.append({
            'code': lang_code,
            'name': lang_code.title(),
            'priority': lang_info['priority'],
            'region': lang_info['region']
        })
    
    # Sort by priority
    languages.sort(key=lambda x: x['priority'])
    return jsonify({'languages': languages}), 200

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
                        streaming_links = json.loads(content.streaming_links or '{}')
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
        
        # Fallback to basic recommendations with user's preferred languages
        preferred_languages = json.loads(current_user.preferred_languages or '[]')
        if not preferred_languages:
            preferred_languages = ['telugu', 'english']
        
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
            streaming_links = json.loads(content.streaming_links or '{}')
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
            streaming_links = json.loads(content.streaming_links or '{}')
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
            
            # Create content object
            if data.get('source') == 'anime':
                content = Content(
                    mal_id=data.get('id'),
                    title=data.get('title'),
                    original_title=data.get('original_title'),
                    content_type='anime',
                    genres=json.dumps(data.get('genres', [])),
                    languages=json.dumps(['japanese']),
                    available_languages=json.dumps(['ja', 'en']),
                    release_date=release_date,
                    runtime=data.get('runtime'),
                    rating=data.get('rating'),
                    vote_count=data.get('vote_count'),
                    popularity=data.get('popularity'),
                    overview=data.get('overview'),
                    poster_path=data.get('poster_path'),
                    backdrop_path=data.get('backdrop_path'),
                    streaming_links=json.dumps({})
                )
            else:
                content = Content(
                    tmdb_id=data.get('id'),
                    title=data.get('title'),
                    original_title=data.get('original_title'),
                    content_type=data.get('content_type', 'movie'),
                    genres=json.dumps(data.get('genres', [])),
                    languages=json.dumps(data.get('languages', ['en'])),
                    available_languages=json.dumps(data.get('available_languages', ['en'])),
                    release_date=release_date,
                    runtime=data.get('runtime'),
                    rating=data.get('rating'),
                    vote_count=data.get('vote_count'),
                    popularity=data.get('popularity'),
                    overview=data.get('overview'),
                    poster_path=data.get('poster_path'),
                    backdrop_path=data.get('backdrop_path'),
                    streaming_links=json.dumps({})
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
        
        # Language distribution
        language_counts = defaultdict(int)
        all_content = Content.query.all()
        for content in all_content:
            if content.languages:
                languages = json.loads(content.languages)
                for lang in languages:
                    language_counts[lang] += 1
        
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
            'language_distribution': [
                {'language': lang, 'count': count}
                for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:10]
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
                streaming_links = json.loads(content.streaming_links or '{}')
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
        'features': {
            'multi_language_streaming': True,
            'regional_recommendations': True,
            'anime_support': True,
            'admin_panel': True
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
                    preferred_languages=json.dumps(['telugu', 'english']),
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