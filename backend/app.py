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

# RapidAPI Configuration for Streaming Availability
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
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float)
    vote_count = db.Column(db.Integer)
    popularity = db.Column(db.Float)
    overview = db.Column(db.Text)
    poster_path = db.Column(db.String(255))
    backdrop_path = db.Column(db.String(255))
    trailer_url = db.Column(db.String(255))
    ott_platforms = db.Column(db.Text)  # JSON string for streaming platforms
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

# Enhanced OTT Platform Information with Real-time Support
OTT_PLATFORMS = {
    # Free Platforms
    'mx_player': {'name': 'MX Player', 'is_free': True, 'url': 'https://mxplayer.in', 'deep_link': 'mxplayer://'},
    'jio_hotstar': {'name': 'JioCinema', 'is_free': True, 'url': 'https://jiocinema.com', 'deep_link': 'jiocinema://'},
    'sonyliv_free': {'name': 'SonyLIV', 'is_free': True, 'url': 'https://sonyliv.com', 'deep_link': 'sonyliv://'},
    'zee5_free': {'name': 'Zee5', 'is_free': True, 'url': 'https://zee5.com', 'deep_link': 'zee5://'},
    'youtube': {'name': 'YouTube', 'is_free': True, 'url': 'https://youtube.com', 'deep_link': 'youtube://'},
    'crunchyroll_free': {'name': 'Crunchyroll', 'is_free': True, 'url': 'https://crunchyroll.com', 'deep_link': 'crunchyroll://'},
    'airtel_xstream': {'name': 'Airtel Xstream', 'is_free': True, 'url': 'https://airtelxstream.in', 'deep_link': 'airtelxstream://'},
    
    # Paid Platforms
    'netflix': {'name': 'Netflix', 'is_free': False, 'url': 'https://netflix.com', 'deep_link': 'netflix://'},
    'prime_video': {'name': 'Prime Video', 'is_free': False, 'url': 'https://primevideo.com', 'deep_link': 'primevideo://'},
    'disney_plus_hotstar': {'name': 'Disney+ Hotstar', 'is_free': False, 'url': 'https://hotstar.com', 'deep_link': 'hotstar://'},
    'zee5_premium': {'name': 'Zee5 Premium', 'is_free': False, 'url': 'https://zee5.com', 'deep_link': 'zee5://'},
    'sonyliv_premium': {'name': 'SonyLIV Premium', 'is_free': False, 'url': 'https://sonyliv.com', 'deep_link': 'sonyliv://'},
    'aha': {'name': 'Aha', 'is_free': False, 'url': 'https://aha.video', 'deep_link': 'aha://'},
    'sun_nxt': {'name': 'Sun NXT', 'is_free': False, 'url': 'https://sunnxt.com', 'deep_link': 'sunnxt://'}
}

# Enhanced Regional Language Mapping with Priority (Telugu first)
REGIONAL_LANGUAGES = {
    'telugu': {
        'codes': ['te', 'tel', 'telugu'],
        'industry': 'Tollywood',
        'priority': 1,
        'search_terms': ['telugu', 'tollywood', 'prabhas', 'mahesh babu', 'allu arjun', 'jr ntr', 'ram charan']
    },
    'hindi': {
        'codes': ['hi', 'hin', 'hindi'],
        'industry': 'Bollywood',
        'priority': 2,
        'search_terms': ['hindi', 'bollywood', 'shah rukh khan', 'salman khan', 'aamir khan', 'akshay kumar']
    },
    'tamil': {
        'codes': ['ta', 'tam', 'tamil'],
        'industry': 'Kollywood',
        'priority': 3,
        'search_terms': ['tamil', 'kollywood', 'rajinikanth', 'kamal haasan', 'vijay', 'ajith', 'suriya']
    },
    'malayalam': {
        'codes': ['ml', 'mal', 'malayalam'],
        'industry': 'Mollywood',
        'priority': 4,
        'search_terms': ['malayalam', 'mollywood', 'mohanlal', 'mammootty', 'fahadh faasil']
    },
    'kannada': {
        'codes': ['kn', 'kan', 'kannada'],
        'industry': 'Sandalwood',
        'priority': 5,
        'search_terms': ['kannada', 'sandalwood', 'yash', 'darshan', 'puneeth rajkumar']
    },
    'english': {
        'codes': ['en', 'eng', 'english'],
        'industry': 'Hollywood',
        'priority': 6,
        'search_terms': ['english', 'hollywood', 'american', 'british']
    }
}

# Movie Categories and Genres
MOVIE_CATEGORIES = {
    'trending': 'Trending Now',
    'popular': 'Popular Movies',
    'top_rated': 'Top Rated',
    'new_releases': 'New Releases',
    'all_time_hits': 'All Time Hits',
    'critics_choice': "Critics' Choice",
    'box_office': 'Box Office Hits'
}

MOVIE_GENRES = [
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

# Real-time Streaming Availability Services
class StreamingAvailabilityService:
    @staticmethod
    def get_streaming_availability_rapidapi(tmdb_id, content_type='movie'):
        """Get streaming availability using RapidAPI Streaming Availability"""
        try:
            url = f"https://{RAPIDAPI_HOST}/get/basic"
            
            headers = {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': RAPIDAPI_HOST
            }
            
            params = {
                'country': 'in',  # India
                'tmdb_id': tmdb_id,
                'output_language': 'en'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return StreamingAvailabilityService.parse_rapidapi_response(data)
            else:
                logger.warning(f"RapidAPI streaming check failed: {response.status_code}")
        except Exception as e:
            logger.error(f"RapidAPI streaming availability error: {e}")
        
        return []
    
    @staticmethod
    def get_watchmode_availability(tmdb_id, content_type='movie'):
        """Get streaming availability using Watchmode API"""
        try:
            if not WATCHMODE_API_KEY or WATCHMODE_API_KEY == 'your_watchmode_api_key':
                return []
            
            # First get Watchmode ID from TMDB ID
            search_url = f"https://api.watchmode.com/v1/search/"
            search_params = {
                'apiKey': WATCHMODE_API_KEY,
                'search_field': 'tmdb_id',
                'search_value': tmdb_id
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=10)
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                if search_data.get('title_results'):
                    watchmode_id = search_data['title_results'][0]['id']
                    
                    # Get streaming sources
                    sources_url = f"https://api.watchmode.com/v1/title/{watchmode_id}/sources/"
                    sources_params = {
                        'apiKey': WATCHMODE_API_KEY,
                        'regions': 'IN'  # India
                    }
                    
                    sources_response = requests.get(sources_url, params=sources_params, timeout=10)
                    
                    if sources_response.status_code == 200:
                        sources_data = sources_response.json()
                        return StreamingAvailabilityService.parse_watchmode_response(sources_data)
        
        except Exception as e:
            logger.error(f"Watchmode API error: {e}")
        
        return []
    
    @staticmethod
    def parse_rapidapi_response(data):
        """Parse RapidAPI response to standard format"""
        streaming_options = []
        
        if 'result' in data and 'streamingOptions' in data['result']:
            for country_code, options in data['result']['streamingOptions'].items():
                if country_code == 'in':  # India
                    for option in options:
                        service = option.get('service', {})
                        platform_name = service.get('name', '').lower()
                        
                        # Map to our platform names
                        mapped_platform = StreamingAvailabilityService.map_platform_name(platform_name)
                        
                        if mapped_platform:
                            streaming_options.append({
                                'platform': mapped_platform,
                                'platform_name': OTT_PLATFORMS[mapped_platform]['name'],
                                'is_free': OTT_PLATFORMS[mapped_platform]['is_free'],
                                'url': option.get('link', OTT_PLATFORMS[mapped_platform]['url']),
                                'deep_link': OTT_PLATFORMS[mapped_platform].get('deep_link'),
                                'type': option.get('type', 'stream'),
                                'quality': option.get('quality', 'HD'),
                                'languages': option.get('audios', []),
                                'subtitles': option.get('subtitles', [])
                            })
        
        return streaming_options
    
    @staticmethod
    def parse_watchmode_response(data):
        """Parse Watchmode response to standard format"""
        streaming_options = []
        
        for source in data:
            platform_name = source.get('name', '').lower()
            mapped_platform = StreamingAvailabilityService.map_platform_name(platform_name)
            
            if mapped_platform:
                streaming_options.append({
                    'platform': mapped_platform,
                    'platform_name': OTT_PLATFORMS[mapped_platform]['name'],
                    'is_free': OTT_PLATFORMS[mapped_platform]['is_free'],
                    'url': source.get('web_url', OTT_PLATFORMS[mapped_platform]['url']),
                    'deep_link': OTT_PLATFORMS[mapped_platform].get('deep_link'),
                    'type': source.get('type', 'stream'),
                    'quality': 'HD',
                    'languages': [],
                    'subtitles': []
                })
        
        return streaming_options
    
    @staticmethod
    def map_platform_name(platform_name):
        """Map external platform names to our internal platform keys"""
        platform_mapping = {
            'netflix': 'netflix',
            'amazon prime video': 'prime_video',
            'amazon prime': 'prime_video',
            'disney+ hotstar': 'disney_plus_hotstar',
            'hotstar': 'disney_plus_hotstar',
            'zee5': 'zee5_premium',
            'sonyliv': 'sonyliv_premium',
            'mx player': 'mx_player',
            'jiocinema': 'jio_hotstar',
            'youtube': 'youtube',
            'crunchyroll': 'crunchyroll_free',
            'aha': 'aha',
            'sun nxt': 'sun_nxt',
            'airtel xstream': 'airtel_xstream'
        }
        
        for key, value in platform_mapping.items():
            if key in platform_name.lower():
                return value
        
        return None
    
    @staticmethod
    def get_comprehensive_streaming_data(tmdb_id, content_type='movie', title=''):
        """Get comprehensive streaming data from multiple sources"""
        all_streaming_options = []
        
        # Try RapidAPI first
        rapidapi_options = StreamingAvailabilityService.get_streaming_availability_rapidapi(tmdb_id, content_type)
        all_streaming_options.extend(rapidapi_options)
        
        # Try Watchmode
        watchmode_options = StreamingAvailabilityService.get_watchmode_availability(tmdb_id, content_type)
        all_streaming_options.extend(watchmode_options)
        
        # Remove duplicates based on platform
        seen_platforms = set()
        unique_options = []
        for option in all_streaming_options:
            if option['platform'] not in seen_platforms:
                seen_platforms.add(option['platform'])
                unique_options.append(option)
        
        # Add fallback options if no streaming data found
        if not unique_options:
            unique_options = StreamingAvailabilityService.get_fallback_options(title, content_type)
        
        return unique_options
    
    @staticmethod
    def get_fallback_options(title, content_type):
        """Provide fallback streaming options when real-time data is unavailable"""
        fallback_options = []
        
        # Add common free platforms
        free_platforms = ['youtube', 'mx_player', 'jio_hotstar']
        for platform in free_platforms:
            search_query = f"{title} full movie" if content_type == 'movie' else f"{title} episodes"
            search_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
            
            fallback_options.append({
                'platform': platform,
                'platform_name': OTT_PLATFORMS[platform]['name'],
                'is_free': True,
                'url': search_url if platform == 'youtube' else OTT_PLATFORMS[platform]['url'],
                'deep_link': OTT_PLATFORMS[platform].get('deep_link'),
                'type': 'search',
                'quality': 'Varies',
                'languages': ['Hindi', 'English'],
                'subtitles': ['English']
            })
        
        return fallback_options

class LanguageSpecificStreamingService:
    @staticmethod
    def get_language_specific_links(content_id, languages):
        """Get streaming links for specific languages"""
        language_links = {}
        
        content = Content.query.get(content_id) if content_id else None
        
        # Get base streaming data
        streaming_options = []
        if content and content.tmdb_id:
            streaming_options = StreamingAvailabilityService.get_comprehensive_streaming_data(
                content.tmdb_id, content.content_type, content.title
            )
        
        for lang in languages:
            lang_code = LanguageSpecificStreamingService.get_language_code(lang)
            lang_links = []
            
            for option in streaming_options:
                # Create language-specific link
                platform_link = {
                    'platform': option['platform'],
                    'platform_name': option['platform_name'],
                    'is_free': option['is_free'],
                    'url': LanguageSpecificStreamingService.create_language_url(
                        option['url'], lang_code, content.title if content else ''
                    ),
                    'deep_link': option.get('deep_link'),
                    'language': lang,
                    'quality': option.get('quality', 'HD'),
                    'button_label': f"Watch in {lang.title()}"
                }
                lang_links.append(platform_link)
            
            # Add fallback language-specific options
            if not lang_links:
                lang_links = LanguageSpecificStreamingService.get_fallback_language_links(lang, content.title if content else '')
            
            language_links[lang] = lang_links
        
        return language_links
    
    @staticmethod
    def get_language_code(language):
        """Convert language name to code"""
        lang_codes = {
            'hindi': 'hi',
            'telugu': 'te',
            'tamil': 'ta',
            'malayalam': 'ml',
            'kannada': 'kn',
            'english': 'en',
            'bengali': 'bn',
            'gujarati': 'gu',
            'marathi': 'mr',
            'punjabi': 'pa'
        }
        return lang_codes.get(language.lower(), 'en')
    
    @staticmethod
    def create_language_url(base_url, lang_code, title):
        """Create language-specific URL"""
        if 'youtube.com' in base_url:
            # For YouTube, modify search query
            if 'search_query=' in base_url:
                return base_url + f"+{lang_code}"
            else:
                return f"https://www.youtube.com/results?search_query={title.replace(' ', '+')}+{lang_code}+full+movie"
        elif 'netflix.com' in base_url:
            return f"{base_url}?audio={lang_code}"
        elif 'primevideo.com' in base_url:
            return f"{base_url}?language={lang_code}"
        else:
            return base_url
    
    @staticmethod
    def get_fallback_language_links(language, title):
        """Get fallback language-specific links"""
        lang_code = LanguageSpecificStreamingService.get_language_code(language)
        
        fallback_links = [
            {
                'platform': 'youtube',
                'platform_name': 'YouTube',
                'is_free': True,
                'url': f"https://www.youtube.com/results?search_query={title.replace(' ', '+')}+{language}+full+movie",
                'deep_link': 'youtube://',
                'language': language,
                'quality': 'Varies',
                'button_label': f"Watch in {language.title()}"
            },
            {
                'platform': 'mx_player',
                'platform_name': 'MX Player',
                'is_free': True,
                'url': 'https://mxplayer.in',
                'deep_link': 'mxplayer://',
                'language': language,
                'quality': 'HD',
                'button_label': f"Watch in {language.title()}"
            }
        ]
        
        return fallback_links

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
            'limit': 20,
            'order_by': 'popularity',
            'sort': 'desc'
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
            'page': page,
            'limit': 20
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan top anime error: {e}")
        return None
    
    @staticmethod
    def get_seasonal_anime(year=None, season=None):
        """Get seasonal anime"""
        if not year:
            year = datetime.now().year
        if not season:
            current_month = datetime.now().month
            seasons = {1: 'winter', 4: 'spring', 7: 'summer', 10: 'fall'}
            season = seasons.get(((current_month - 1) // 3) * 3 + 1, 'winter')
        
        url = f"{JikanService.BASE_URL}/seasons/{year}/{season}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan seasonal anime error: {e}")
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

# Enhanced Content Service with Real-time OTT Integration
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                # Update streaming data if it's old
                if not existing.updated_at or (datetime.utcnow() - existing.updated_at).total_seconds() > 3600:
                    ContentService.update_streaming_data(existing)
                return existing
            
            genres = []
            if 'genres' in tmdb_data:
                genres = [genre['name'] for genre in tmdb_data['genres']]
            elif 'genre_ids' in tmdb_data:
                genres = ContentService.map_genre_ids(tmdb_data['genre_ids'])
            
            languages = []
            if 'spoken_languages' in tmdb_data:
                languages = [lang['name'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
            
            # Get real-time OTT platforms
            ott_platforms = StreamingAvailabilityService.get_comprehensive_streaming_data(
                tmdb_data['id'], content_type, tmdb_data.get('title') or tmdb_data.get('name')
            )
            
            # Get language-specific streaming links
            streaming_links = LanguageSpecificStreamingService.get_language_specific_links(
                None, languages
            )
            
            release_date = None
            if tmdb_data.get('release_date') or tmdb_data.get('first_air_date'):
                try:
                    date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                except:
                    release_date = None
            
            content = Content(
                tmdb_id=tmdb_data['id'],
                title=tmdb_data.get('title') or tmdb_data.get('name'),
                original_title=tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                content_type=content_type,
                genres=json.dumps(genres),
                languages=json.dumps(languages),
                release_date=release_date,
                runtime=tmdb_data.get('runtime'),
                rating=tmdb_data.get('vote_average'),
                vote_count=tmdb_data.get('vote_count'),
                popularity=tmdb_data.get('popularity'),
                overview=tmdb_data.get('overview'),
                poster_path=tmdb_data.get('poster_path'),
                backdrop_path=tmdb_data.get('backdrop_path'),
                ott_platforms=json.dumps(ott_platforms),
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
        """Save anime content with proper anime handling"""
        try:
            existing = Content.query.filter_by(mal_id=anime_data['mal_id']).first()
            if existing:
                return existing
            
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            # Get anime streaming platforms (typically Crunchyroll, Funimation, etc.)
            anime_platforms = [
                {
                    'platform': 'crunchyroll_free',
                    'platform_name': 'Crunchyroll',
                    'is_free': True,
                    'url': f"https://www.crunchyroll.com/search?q={anime_data.get('title', '').replace(' ', '%20')}",
                    'type': 'stream',
                    'quality': 'HD'
                }
            ]
            
            # Language-specific links for anime
            anime_language_links = {
                'Japanese': [{
                    'platform': 'crunchyroll_free',
                    'platform_name': 'Crunchyroll',
                    'is_free': True,
                    'url': f"https://www.crunchyroll.com/search?q={anime_data.get('title', '').replace(' ', '%20')}",
                    'language': 'Japanese',
                    'quality': 'HD',
                    'button_label': 'Watch in Japanese (Sub)'
                }]
            }
            
            release_date = None
            if anime_data.get('aired', {}).get('from'):
                try:
                    release_date = datetime.strptime(anime_data['aired']['from'][:10], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            content = Content(
                mal_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                languages=json.dumps(['Japanese']),
                release_date=release_date,
                rating=anime_data.get('score'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                ott_platforms=json.dumps(anime_platforms),
                streaming_links=json.dumps(anime_language_links)
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def update_streaming_data(content):
        """Update streaming data for existing content"""
        try:
            if content.tmdb_id:
                ott_platforms = StreamingAvailabilityService.get_comprehensive_streaming_data(
                    content.tmdb_id, content.content_type, content.title
                )
                content.ott_platforms = json.dumps(ott_platforms)
                
                languages = json.loads(content.languages or '[]')
                streaming_links = LanguageSpecificStreamingService.get_language_specific_links(
                    content.id, languages
                )
                content.streaming_links = json.dumps(streaming_links)
                
                content.updated_at = datetime.utcnow()
                db.session.commit()
        except Exception as e:
            logger.error(f"Error updating streaming data: {e}")
    
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

# Enhanced Regional Movie Service with Telugu Priority
class RegionalMovieService:
    @staticmethod
    def get_regional_movies_by_category(language, category, limit=20, page=1):
        """Get regional movies by category with high accuracy"""
        try:
            lang_config = REGIONAL_LANGUAGES.get(language.lower())
            if not lang_config:
                return []
            
            movies = []
            
            if category == 'trending':
                movies = RegionalMovieService.get_trending_regional(language, limit, page)
            elif category == 'popular':
                movies = RegionalMovieService.get_popular_regional(language, limit, page)
            elif category == 'new_releases':
                movies = RegionalMovieService.get_new_releases_regional(language, limit, page)
            elif category == 'all_time_hits':
                movies = RegionalMovieService.get_all_time_hits_regional(language, limit, page)
            elif category == 'top_rated':
                movies = RegionalMovieService.get_top_rated_regional(language, limit, page)
            else:
                movies = RegionalMovieService.get_popular_regional(language, limit, page)
            
            return movies
        except Exception as e:
            logger.error(f"Error getting regional movies: {e}")
            return []
    
    @staticmethod
    def get_trending_regional(language, limit=20, page=1):
        """Get trending movies in specific language"""
        try:
            lang_config = REGIONAL_LANGUAGES.get(language.lower())
            search_terms = lang_config['search_terms']
            
            all_movies = []
            
            # Search with multiple terms for better accuracy
            for term in search_terms[:3]:  # Use top 3 search terms
                search_results = TMDBService.search_content(
                    f"{term} movie 2024 2023", 'movie', page=page
                )
                
                if search_results:
                    for item in search_results.get('results', []):
                        if RegionalMovieService.is_regional_content(item, language):
                            content = ContentService.save_content_from_tmdb(item, 'movie')
                            if content and content not in all_movies:
                                all_movies.append(content)
            
            # Get trending movies and filter by language
            trending = TMDBService.get_trending('movie', 'week', page=page)
            if trending:
                for item in trending.get('results', []):
                    if RegionalMovieService.is_regional_content(item, language):
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content and content not in all_movies:
                            all_movies.append(content)
            
            # Sort by popularity and release date
            all_movies.sort(key=lambda x: (x.popularity or 0, x.release_date or datetime.min.date()), reverse=True)
            
            return all_movies[:limit]
        except Exception as e:
            logger.error(f"Error getting trending regional movies: {e}")
            return []
    
    @staticmethod
    def get_popular_regional(language, limit=20, page=1):
        """Get popular movies in specific language"""
        try:
            lang_config = REGIONAL_LANGUAGES.get(language.lower())
            region_code = 'IN' if language.lower() != 'english' else 'US'
            
            popular_results = TMDBService.get_popular('movie', page=page, region=region_code)
            movies = []
            
            if popular_results:
                for item in popular_results.get('results', []):
                    if RegionalMovieService.is_regional_content(item, language):
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content:
                            movies.append(content)
            
            # If not enough movies, search with specific terms
            if len(movies) < limit:
                search_terms = lang_config['search_terms']
                for term in search_terms:
                    search_results = TMDBService.search_content(f"{term} popular movie", 'movie')
                    if search_results:
                        for item in search_results.get('results', []):
                            if RegionalMovieService.is_regional_content(item, language):
                                content = ContentService.save_content_from_tmdb(item, 'movie')
                                if content and content not in movies:
                                    movies.append(content)
                                    if len(movies) >= limit:
                                        break
                    if len(movies) >= limit:
                        break
            
            return movies[:limit]
        except Exception as e:
            logger.error(f"Error getting popular regional movies: {e}")
            return []
    
    @staticmethod
    def get_new_releases_regional(language, limit=20, page=1):
        """Get new releases in specific language"""
        try:
            current_year = datetime.now().year
            last_year = current_year - 1
            
            # Search for recent movies
            url = f"{TMDBService.BASE_URL}/discover/movie"
            params = {
                'api_key': TMDB_API_KEY,
                'primary_release_date.gte': f'{last_year}-01-01',
                'primary_release_date.lte': f'{current_year}-12-31',
                'sort_by': 'primary_release_date.desc',
                'page': page
            }
            
            # Add language filter if not English
            if language.lower() != 'english':
                lang_codes = REGIONAL_LANGUAGES[language.lower()]['codes']
                params['with_original_language'] = lang_codes[0]
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    results = response.json()
                    movies = []
                    
                    for item in results.get('results', []):
                        if RegionalMovieService.is_regional_content(item, language):
                            content = ContentService.save_content_from_tmdb(item, 'movie')
                            if content:
                                movies.append(content)
                    
                    return movies[:limit]
            except Exception as e:
                logger.error(f"TMDB discover error: {e}")
            
            # Fallback to search
            return RegionalMovieService.get_popular_regional(language, limit, page)
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return []
    
    @staticmethod
    def get_all_time_hits_regional(language, limit=20, page=1):
        """Get all-time hit movies in specific language"""
        try:
            lang_config = REGIONAL_LANGUAGES.get(language.lower())
            
            # Define classic search terms for each language
            classic_terms = {
                'telugu': ['baahubali', 'rangasthalam', 'arjun reddy', 'magadheera', 'eega', 'pokiri', 'athadu'],
                'hindi': ['sholay', 'ddlj', '3 idiots', 'dangal', 'bahubali', 'lagaan', 'mughal e azam'],
                'tamil': ['baahubali', 'vikram', 'master', 'bigil', 'sarkar', 'enthiran', 'anniyan'],
                'malayalam': ['drishyam', 'premam', 'bangalore days', 'maheshinte prathikaaram', 'kumbakonam gopals'],
                'kannada': ['kgf', 'kantara', 'kirik party', 'ugramm', 'rangitaranga'],
                'english': ['godfather', 'shawshank redemption', 'dark knight', 'pulp fiction', 'forrest gump']
            }
            
            search_terms = classic_terms.get(language.lower(), lang_config['search_terms'])
            movies = []
            
            # Search for classic movies
            for term in search_terms:
                search_results = TMDBService.search_content(term, 'movie')
                if search_results:
                    for item in search_results.get('results', []):
                        if RegionalMovieService.is_regional_content(item, language):
                            # Filter by high ratings and popularity
                            if item.get('vote_average', 0) >= 7.0 and item.get('vote_count', 0) >= 100:
                                content = ContentService.save_content_from_tmdb(item, 'movie')
                                if content and content not in movies:
                                    movies.append(content)
            
            # Sort by rating and vote count
            movies.sort(key=lambda x: (x.rating or 0) * (x.vote_count or 0), reverse=True)
            
            return movies[:limit]
        except Exception as e:
            logger.error(f"Error getting all-time hits: {e}")
            return []
    
    @staticmethod
    def get_top_rated_regional(language, limit=20, page=1):
        """Get top-rated movies in specific language"""
        try:
            url = f"{TMDBService.BASE_URL}/movie/top_rated"
            params = {
                'api_key': TMDB_API_KEY,
                'page': page
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    results = response.json()
                    movies = []
                    
                    for item in results.get('results', []):
                        if RegionalMovieService.is_regional_content(item, language):
                            content = ContentService.save_content_from_tmdb(item, 'movie')
                            if content:
                                movies.append(content)
                    
                    return movies[:limit]
            except Exception as e:
                logger.error(f"TMDB top rated error: {e}")
            
            # Fallback to all-time hits
            return RegionalMovieService.get_all_time_hits_regional(language, limit, page)
        except Exception as e:
            logger.error(f"Error getting top-rated movies: {e}")
            return []
    
    @staticmethod
    def get_movies_by_genre_regional(language, genre, limit=20, page=1):
        """Get movies by genre for specific language"""
        try:
            # First get genre ID
            genre_id = RegionalMovieService.get_genre_id(genre)
            if not genre_id:
                return []
            
            url = f"{TMDBService.BASE_URL}/discover/movie"
            params = {
                'api_key': TMDB_API_KEY,
                'with_genres': genre_id,
                'sort_by': 'popularity.desc',
                'page': page
            }
            
            # Add language filter if not English
            if language.lower() != 'english':
                lang_codes = REGIONAL_LANGUAGES[language.lower()]['codes']
                params['with_original_language'] = lang_codes[0]
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    results = response.json()
                    movies = []
                    
                    for item in results.get('results', []):
                        if RegionalMovieService.is_regional_content(item, language):
                            content = ContentService.save_content_from_tmdb(item, 'movie')
                            if content:
                                movies.append(content)
                    
                    return movies[:limit]
            except Exception as e:
                logger.error(f"TMDB genre discover error: {e}")
            
            return []
        except Exception as e:
            logger.error(f"Error getting movies by genre: {e}")
            return []
    
    @staticmethod
    def get_genre_id(genre_name):
        """Get TMDB genre ID from genre name"""
        genre_mapping = {
            'action': 28, 'adventure': 12, 'animation': 16, 'comedy': 35,
            'crime': 80, 'documentary': 99, 'drama': 18, 'family': 10751,
            'fantasy': 14, 'history': 36, 'horror': 27, 'music': 10402,
            'mystery': 9648, 'romance': 10749, 'science fiction': 878,
            'sci-fi': 878, 'thriller': 53, 'war': 10752, 'western': 37,
            'biography': 99, 'musical': 10402
        }
        return genre_mapping.get(genre_name.lower())
    
    @staticmethod
    def is_regional_content(item, language):
        """Check if content belongs to specific regional language"""
        try:
            original_language = item.get('original_language', '').lower()
            title = (item.get('title') or item.get('name', '')).lower()
            overview = (item.get('overview') or '').lower()
            
            lang_config = REGIONAL_LANGUAGES.get(language.lower())
            if not lang_config:
                return False
            
            # Check original language
            if original_language in lang_config['codes']:
                return True
            
            # Check for language-specific keywords in title
            search_terms = [term.lower() for term in lang_config['search_terms']]
            for term in search_terms:
                if term in title or term in overview:
                    return True
            
            # Special handling for English content
            if language.lower() == 'english':
                return original_language in ['en', 'english'] or any(
                    term in title for term in ['hollywood', 'american', 'british']
                )
            
            return False
        except Exception as e:
            logger.error(f"Error checking regional content: {e}")
            return False

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
            
            if popular_movies:
                for item in popular_movies.get('results', []):
                    if genre.lower() in [g.lower() for g in ContentService.map_genre_ids(item.get('genre_ids', []))]:
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content:
                            recommendations.append(content)
            
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
                'hindi': ['bollywood', 'hindi movie', 'hindi film'],
                'telugu': ['tollywood', 'telugu movie', 'telugu film'],
                'tamil': ['kollywood', 'tamil movie', 'tamil film'],
                'kannada': ['sandalwood', 'kannada movie', 'kannada film']
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
            
            if location and location.get('country') == 'India':
                regional_recs = RecommendationEngine.get_regional_recommendations('telugu', limit=5)
                recommendations.extend(regional_recs)
            
            trending_recs = RecommendationEngine.get_trending_recommendations(limit=10)
            recommendations.extend(trending_recs)
            
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

# Enhanced Telegram Service with Streaming Links
class TelegramService:
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Get streaming platforms
            streaming_info = ""
            if content.ott_platforms:
                try:
                    platforms = json.loads(content.ott_platforms)
                    free_platforms = [p['platform_name'] for p in platforms if p.get('is_free')]
                    paid_platforms = [p['platform_name'] for p in platforms if not p.get('is_free')]
                    
                    if free_platforms:
                        streaming_info += f"\n🆓 **Free on:** {', '.join(free_platforms)}"
                    if paid_platforms:
                        streaming_info += f"\n💰 **Premium on:** {', '.join(paid_platforms)}"
                except:
                    pass
            
            # Get language options
            language_info = ""
            if content.streaming_links:
                try:
                    lang_links = json.loads(content.streaming_links)
                    available_languages = list(lang_links.keys())
                    if available_languages:
                        language_info = f"\n🗣️ **Available in:** {', '.join(available_languages)}"
                except:
                    pass
            
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}{streaming_info}{language_info}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

#AdminChoice #MovieRecommendation #CineScope #Streaming"""
            
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
        
        tmdb_results = TMDBService.search_content(query, content_type, page=page)
        
        anime_results = None
        if content_type in ['anime', 'multi']:
            anime_results = JikanService.search_anime(query, page=page)
        
        results = []
        
        if tmdb_results:
            for item in tmdb_results.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    interaction = AnonymousInteraction(
                        session_id=session_id,
                        content_id=content.id,
                        interaction_type='search',
                        ip_address=request.remote_addr
                    )
                    db.session.add(interaction)
                    
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
                        'streaming_links': json.loads(content.streaming_links or '{}')
                    })
        
        if anime_results:
            for anime in anime_results.get('data', []):
                content = ContentService.save_anime_content(anime)
                if content:
                    results.append({
                        'id': content.id,
                        'mal_id': content.mal_id,
                        'title': content.title,
                        'content_type': 'anime',
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': content.poster_path,
                        'overview': content.overview,
                        'ott_platforms': json.loads(content.ott_platforms or '[]'),
                        'streaming_links': json.loads(content.streaming_links or '{}')
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

# Enhanced content details with real-time streaming
@app.route('/api/content/<int:content_id>', methods=['GET'])
def get_content_details(content_id):
    try:
        content = Content.query.get_or_404(content_id)
        
        session_id = get_session_id()
        interaction = AnonymousInteraction(
            session_id=session_id,
            content_id=content.id,
            interaction_type='view',
            ip_address=request.remote_addr
        )
        db.session.add(interaction)
        
        # Update streaming data if old
        if not content.updated_at or (datetime.utcnow() - content.updated_at).total_seconds() > 3600:
            ContentService.update_streaming_data(content)
        
        additional_details = None
        if content.tmdb_id:
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        elif content.mal_id:
            additional_details = JikanService.get_anime_details(content.mal_id)
        
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
        
        streaming_options = json.loads(content.ott_platforms or '[]')
        language_links = json.loads(content.streaming_links or '{}')
        
        db.session.commit()
        
        result = {
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
            'streaming_options': streaming_options,
            'language_specific_links': language_links,
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details and 'credits' in additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details and 'credits' in additional_details else [],
            'last_updated': content.updated_at.isoformat() if content.updated_at else None
        }
        
        # Add anime-specific details
        if content.content_type == 'anime' and additional_details:
            anime_data = additional_details.get('data', {})
            result.update({
                'episodes': anime_data.get('episodes'),
                'status': anime_data.get('status'),
                'aired': anime_data.get('aired'),
                'studios': anime_data.get('studios', []),
                'source': anime_data.get('source')
            })
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# NEW: Real-time streaming links endpoint
@app.route('/api/content/<int:content_id>/streaming', methods=['GET'])
def get_streaming_links(content_id):
    """Get real-time streaming links for content"""
    try:
        content = Content.query.get_or_404(content_id)
        
        # Update streaming data if it's old
        if not content.updated_at or (datetime.utcnow() - content.updated_at).total_seconds() > 3600:  # 1 hour
            ContentService.update_streaming_data(content)
        
        # Get language-specific links
        languages = json.loads(content.languages or '[]')
        language_links = LanguageSpecificStreamingService.get_language_specific_links(
            content_id, languages
        )
        
        # Get general streaming options
        streaming_options = json.loads(content.ott_platforms or '[]')
        
        return jsonify({
            'content_id': content.id,
            'title': content.title,
            'streaming_options': streaming_options,
            'language_specific_links': language_links,
            'last_updated': content.updated_at.isoformat() if content.updated_at else None
        }), 200
        
    except Exception as e:
        logger.error(f"Streaming links error: {e}")
        return jsonify({'error': 'Failed to get streaming links'}), 500

# NEW: Regional content endpoints
@app.route('/api/regional/<language>/<category>', methods=['GET'])
def get_regional_content(language, category):
    """Get regional content by language and category"""
    try:
        limit = int(request.args.get('limit', 20))
        page = int(request.args.get('page', 1))
        
        if language.lower() not in REGIONAL_LANGUAGES:
            return jsonify({'error': 'Language not supported'}), 400
        
        movies = RegionalMovieService.get_regional_movies_by_category(
            language, category, limit, page
        )
        
        result = []
        for movie in movies:
            streaming_options = json.loads(movie.ott_platforms or '[]')
            language_links = json.loads(movie.streaming_links or '{}')
            
            result.append({
                'id': movie.id,
                'title': movie.title,
                'original_title': movie.original_title,
                'content_type': movie.content_type,
                'genres': json.loads(movie.genres or '[]'),
                'languages': json.loads(movie.languages or '[]'),
                'rating': movie.rating,
                'release_date': movie.release_date.isoformat() if movie.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{movie.poster_path}" if movie.poster_path else None,
                'overview': movie.overview[:150] + '...' if movie.overview else '',
                'streaming_options': streaming_options,
                'language_links': language_links
            })
        
        return jsonify({
            'language': language,
            'category': category,
            'movies': result,
            'total': len(result),
            'page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Regional content error: {e}")
        return jsonify({'error': 'Failed to get regional content'}), 500

@app.route('/api/regional/<language>/genres/<genre>', methods=['GET'])
def get_regional_genre_content(language, genre):
    """Get regional content by language and genre"""
    try:
        limit = int(request.args.get('limit', 20))
        page = int(request.args.get('page', 1))
        
        movies = RegionalMovieService.get_movies_by_genre_regional(
            language, genre, limit, page
        )
        
        result = []
        for movie in movies:
            streaming_options = json.loads(movie.ott_platforms or '[]')
            language_links = json.loads(movie.streaming_links or '{}')
            
            result.append({
                'id': movie.id,
                'title': movie.title,
                'content_type': movie.content_type,
                'genres': json.loads(movie.genres or '[]'),
                'rating': movie.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{movie.poster_path}" if movie.poster_path else None,
                'overview': movie.overview[:100] + '...' if movie.overview else '',
                'streaming_options': streaming_options,
                'language_links': language_links
            })
        
        return jsonify({
            'language': language,
            'genre': genre,
            'movies': result
        }), 200
        
    except Exception as e:
        logger.error(f"Regional genre content error: {e}")
        return jsonify({'error': 'Failed to get regional genre content'}), 500

# NEW: Fixed anime details endpoint
@app.route('/api/anime/<int:anime_id>', methods=['GET'])
def get_anime_details(anime_id):
    """Get detailed anime information"""
    try:
        # First check if we have it in our database
        content = Content.query.filter_by(mal_id=anime_id).first()
        
        if not content:
            # Get from Jikan API
            anime_data = JikanService.get_anime_details(anime_id)
            if not anime_data:
                return jsonify({'error': 'Anime not found'}), 404
            
            # Save to database
            content = ContentService.save_anime_content(anime_data['data'])
            if not content:
                return jsonify({'error': 'Failed to save anime data'}), 500
        
        # Record view interaction
        session_id = get_session_id()
        interaction = AnonymousInteraction(
            session_id=session_id,
            content_id=content.id,
            interaction_type='view',
            ip_address=request.remote_addr
        )
        db.session.add(interaction)
        db.session.commit()
        
        # Get additional details if needed
        additional_data = JikanService.get_anime_details(anime_id)
        
        streaming_options = json.loads(content.ott_platforms or '[]')
        language_links = json.loads(content.streaming_links or '{}')
        
        # Add Crunchyroll search link if no streaming options
        if not streaming_options:
            streaming_options = [
                {
                    'platform': 'crunchyroll_free',
                    'platform_name': 'Crunchyroll',
                    'is_free': True,
                    'url': f"https://www.crunchyroll.com/search?q={content.title.replace(' ', '%20')}",
                    'type': 'search',
                    'quality': 'HD'
                }
            ]
        
        result = {
            'id': content.id,
            'mal_id': content.mal_id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'rating': content.rating,
            'overview': content.overview,
            'poster_path': content.poster_path,
            'streaming_options': streaming_options,
            'language_links': language_links
        }
        
        # Add additional anime details
        if additional_data and 'data' in additional_data:
            anime_details = additional_data['data']
            result.update({
                'episodes': anime_details.get('episodes'),
                'status': anime_details.get('status'),
                'aired': anime_details.get('aired'),
                'studios': anime_details.get('studios', []),
                'source': anime_details.get('source'),
                'duration': anime_details.get('duration'),
                'year': anime_details.get('year'),
                'season': anime_details.get('season')
            })
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Anime details error: {e}")
        return jsonify({'error': 'Failed to get anime details'}), 500

@app.route('/api/regional/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported regional languages"""
    try:
        languages = []
        for lang, config in REGIONAL_LANGUAGES.items():
            languages.append({
                'code': lang,
                'name': lang.title(),
                'industry': config['industry'],
                'priority': config['priority']
            })
        
        # Sort by priority (Telugu first)
        languages.sort(key=lambda x: x['priority'])
        
        return jsonify({
            'languages': languages,
            'categories': list(MOVIE_CATEGORIES.keys()),
            'genres': MOVIE_GENRES
        }), 200
        
    except Exception as e:
        logger.error(f"Languages error: {e}")
        return jsonify({'error': 'Failed to get languages'}), 500

# Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
                'streaming_links': json.loads(content.streaming_links or '{}')
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
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
                'streaming_links': json.loads(content.streaming_links or '{}')
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
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
                'streaming_links': json.loads(content.streaming_links or '{}')
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
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
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
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
                'streaming_links': json.loads(content.streaming_links or '{}')
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
        interactions = UserInteraction.query.filter_by(user_id=current_user.id).all()
        
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
        
        try:
            response = requests.post(f"{ML_SERVICE_URL}/api/recommendations", json=user_data, timeout=30)
            
            if response.status_code == 200:
                ml_recommendations = response.json().get('recommendations', [])
                
                content_ids = [rec['content_id'] for rec in ml_recommendations]
                contents = Content.query.filter(Content.id.in_(content_ids)).all()
                
                result = []
                content_dict = {content.id: content for content in contents}
                
                for rec in ml_recommendations:
                    content = content_dict.get(rec['content_id'])
                    if content:
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'ott_platforms': json.loads(content.ott_platforms or '[]'),
                            'streaming_links': json.loads(content.streaming_links or '{}'),
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
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
                'streaming_links': json.loads(content.streaming_links or '{}')
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
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
                'streaming_links': json.loads(content.streaming_links or '{}')
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
            if data.get('source') == 'anime':
                existing_content = Content.query.filter_by(mal_id=data['id']).first()
            else:
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
            
            # Create content object
            content_data = {
                'title': data.get('title'),
                'original_title': data.get('original_title'),
                'content_type': data.get('content_type', 'movie'),
                'genres': json.dumps(data.get('genres', [])),
                'languages': json.dumps(data.get('languages', ['en'])),
                'release_date': release_date,
                'runtime': data.get('runtime'),
                'rating': data.get('rating'),
                'vote_count': data.get('vote_count'),
                'popularity': data.get('popularity'),
                'overview': data.get('overview'),
                'poster_path': data.get('poster_path'),
                'backdrop_path': data.get('backdrop_path'),
                'ott_platforms': json.dumps(data.get('ott_platforms', [])),
                'streaming_links': json.dumps(data.get('streaming_links', {}))
            }
            
            # Add external ID based on source
            if data.get('source') == 'anime':
                content_data['mal_id'] = data.get('id')
            else:
                content_data['tmdb_id'] = data.get('id')
            
            content = Content(**content_data)
            
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
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'ott_platforms': json.loads(content.ott_platforms or '[]'),
                    'streaming_links': json.loads(content.streaming_links or '{}'),
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
            'real_time_streaming': True,
            'language_specific_links': True,
            'regional_content': True,
            'anime_support': True
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