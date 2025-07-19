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

# New API Keys for Streaming
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
    streaming_links = db.Column(db.Text)  # JSON string for language-specific links
    watch_providers = db.Column(db.Text)  # JSON string for detailed provider info
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

# Enhanced OTT Platform Information
OTT_PLATFORMS = {
    # Free Platforms
    'mx_player': {'name': 'MX Player', 'is_free': True, 'url': 'https://mxplayer.in', 'logo': 'mx_player_logo.png'},
    'jiocinema': {'name': 'JioCinema', 'is_free': True, 'url': 'https://jiocinema.com', 'logo': 'jio_logo.png'},
    'youtube': {'name': 'YouTube', 'is_free': True, 'url': 'https://youtube.com', 'logo': 'youtube_logo.png'},
    'zee5_free': {'name': 'ZEE5 Free', 'is_free': True, 'url': 'https://zee5.com', 'logo': 'zee5_logo.png'},
    'sonyliv_free': {'name': 'SonyLIV Free', 'is_free': True, 'url': 'https://sonyliv.com', 'logo': 'sonyliv_logo.png'},
    'airtel_xstream': {'name': 'Airtel Xstream', 'is_free': True, 'url': 'https://airtelxstream.in', 'logo': 'airtel_logo.png'},
    'crunchyroll_free': {'name': 'Crunchyroll Free', 'is_free': True, 'url': 'https://crunchyroll.com', 'logo': 'crunchyroll_logo.png'},
    
    # Paid Platforms
    'netflix': {'name': 'Netflix', 'is_free': False, 'url': 'https://netflix.com', 'logo': 'netflix_logo.png'},
    'prime_video': {'name': 'Prime Video', 'is_free': False, 'url': 'https://primevideo.com', 'logo': 'prime_logo.png'},
    'disney_hotstar': {'name': 'Disney+ Hotstar', 'is_free': False, 'url': 'https://hotstar.com', 'logo': 'hotstar_logo.png'},
    'zee5_premium': {'name': 'ZEE5 Premium', 'is_free': False, 'url': 'https://zee5.com', 'logo': 'zee5_logo.png'},
    'sonyliv_premium': {'name': 'SonyLIV Premium', 'is_free': False, 'url': 'https://sonyliv.com', 'logo': 'sonyliv_logo.png'},
    'aha': {'name': 'Aha', 'is_free': False, 'url': 'https://aha.video', 'logo': 'aha_logo.png'},
    'sun_nxt': {'name': 'Sun NXT', 'is_free': False, 'url': 'https://sunnxt.com', 'logo': 'sunnxt_logo.png'},
    'voot_select': {'name': 'Voot Select', 'is_free': False, 'url': 'https://voot.com', 'logo': 'voot_logo.png'},
    'alt_balaji': {'name': 'ALTBalaji', 'is_free': False, 'url': 'https://altbalaji.com', 'logo': 'altbalaji_logo.png'}
}

# Enhanced Regional Language Mapping
REGIONAL_LANGUAGES = {
    'telugu': {
        'codes': ['te', 'tel', 'telugu'],
        'name': 'Telugu',
        'industry': 'Tollywood',
        'country': 'IN',
        'primary_platforms': ['aha', 'sun_nxt', 'zee5_premium', 'disney_hotstar']
    },
    'hindi': {
        'codes': ['hi', 'hin', 'hindi'],
        'name': 'Hindi',
        'industry': 'Bollywood',
        'country': 'IN',
        'primary_platforms': ['netflix', 'prime_video', 'zee5_premium', 'disney_hotstar']
    },
    'tamil': {
        'codes': ['ta', 'tam', 'tamil'],
        'name': 'Tamil',
        'industry': 'Kollywood',
        'country': 'IN',
        'primary_platforms': ['sun_nxt', 'zee5_premium', 'disney_hotstar', 'netflix']
    },
    'malayalam': {
        'codes': ['ml', 'mal', 'malayalam'],
        'name': 'Malayalam',
        'industry': 'Mollywood',
        'country': 'IN',
        'primary_platforms': ['disney_hotstar', 'sun_nxt', 'netflix', 'prime_video']
    },
    'kannada': {
        'codes': ['kn', 'kan', 'kannada'],
        'name': 'Kannada',
        'industry': 'Sandalwood',
        'country': 'IN',
        'primary_platforms': ['sun_nxt', 'zee5_premium', 'voot_select', 'disney_hotstar']
    },
    'english': {
        'codes': ['en', 'eng', 'english'],
        'name': 'English',
        'industry': 'Hollywood',
        'country': 'US',
        'primary_platforms': ['netflix', 'prime_video', 'disney_hotstar', 'youtube']
    }
}

# Movie Genres
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

# Streaming Availability Service
class StreamingAvailabilityService:
    @staticmethod
    def get_watchmode_availability(title, content_type='movie', year=None):
        """Get streaming availability from Watchmode API"""
        try:
            if not WATCHMODE_API_KEY or WATCHMODE_API_KEY == 'your_watchmode_api_key':
                return []
            
            # Search for the title first
            search_url = "https://api.watchmode.com/v1/search/"
            search_params = {
                'apiKey': WATCHMODE_API_KEY,
                'search_field': 'name',
                'search_value': title,
                'types': content_type
            }
            
            if year:
                search_params['year'] = year
            
            search_response = requests.get(search_url, params=search_params, timeout=10)
            
            if search_response.status_code == 200:
                search_results = search_response.json()
                
                if search_results.get('title_results'):
                    title_id = search_results['title_results'][0]['id']
                    
                    # Get streaming sources
                    sources_url = f"https://api.watchmode.com/v1/title/{title_id}/sources/"
                    sources_params = {
                        'apiKey': WATCHMODE_API_KEY,
                        'regions': 'IN,US'  # India and US
                    }
                    
                    sources_response = requests.get(sources_url, params=sources_params, timeout=10)
                    
                    if sources_response.status_code == 200:
                        sources_data = sources_response.json()
                        return StreamingAvailabilityService._process_watchmode_sources(sources_data)
            
            return []
        except Exception as e:
            logger.error(f"Watchmode API error: {e}")
            return []
    
    @staticmethod
    def get_rapidapi_availability(title, content_type='movie', year=None):
        """Get streaming availability from RapidAPI Streaming Availability"""
        try:
            search_url = "https://streaming-availability.p.rapidapi.com/search/title"
            
            headers = {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': RAPIDAPI_HOST
            }
            
            params = {
                'title': title,
                'country': 'in',  # India
                'show_type': content_type,
                'output_language': 'en'
            }
            
            if year:
                params['year'] = year
            
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return StreamingAvailabilityService._process_rapidapi_sources(data)
            
            return []
        except Exception as e:
            logger.error(f"RapidAPI Streaming Availability error: {e}")
            return []
    
    @staticmethod
    def _process_watchmode_sources(sources_data):
        """Process Watchmode API response"""
        streaming_options = []
        
        for source in sources_data:
            platform_name = source.get('name', '').lower()
            
            # Map to our platform names
            mapped_platform = StreamingAvailabilityService._map_platform_name(platform_name)
            
            if mapped_platform:
                streaming_options.append({
                    'platform': mapped_platform,
                    'platform_name': OTT_PLATFORMS.get(mapped_platform, {}).get('name', platform_name),
                    'is_free': OTT_PLATFORMS.get(mapped_platform, {}).get('is_free', False),
                    'url': source.get('web_url', OTT_PLATFORMS.get(mapped_platform, {}).get('url')),
                    'type': source.get('type', 'subscription'),
                    'price': source.get('price'),
                    'currency': source.get('currency'),
                    'quality': source.get('format'),
                    'region': source.get('region')
                })
        
        return streaming_options
    
    @staticmethod
    def _process_rapidapi_sources(data):
        """Process RapidAPI Streaming Availability response"""
        streaming_options = []
        
        for result in data.get('result', []):
            streaming_info = result.get('streamingInfo', {})
            
            for country, platforms in streaming_info.items():
                if country.lower() in ['in', 'us']:  # India or US
                    for platform_info in platforms:
                        platform_name = platform_info.get('service', '').lower()
                        mapped_platform = StreamingAvailabilityService._map_platform_name(platform_name)
                        
                        if mapped_platform:
                            streaming_options.append({
                                'platform': mapped_platform,
                                'platform_name': OTT_PLATFORMS.get(mapped_platform, {}).get('name', platform_name),
                                'is_free': OTT_PLATFORMS.get(mapped_platform, {}).get('is_free', False),
                                'url': platform_info.get('link', OTT_PLATFORMS.get(mapped_platform, {}).get('url')),
                                'type': platform_info.get('streamingType', 'subscription'),
                                'quality': platform_info.get('quality'),
                                'region': country.upper(),
                                'available_since': platform_info.get('availableSince')
                            })
        
        return streaming_options
    
    @staticmethod
    def _map_platform_name(platform_name):
        """Map external platform names to our internal platform keys"""
        platform_mapping = {
            'netflix': 'netflix',
            'amazon prime video': 'prime_video',
            'amazon prime': 'prime_video',
            'prime video': 'prime_video',
            'disney+': 'disney_hotstar',
            'disney plus': 'disney_hotstar',
            'hotstar': 'disney_hotstar',
            'disney+ hotstar': 'disney_hotstar',
            'youtube': 'youtube',
            'youtube movies': 'youtube',
            'mx player': 'mx_player',
            'mxplayer': 'mx_player',
            'jiocinema': 'jiocinema',
            'jio cinema': 'jiocinema',
            'zee5': 'zee5_premium',
            'sonyliv': 'sonyliv_premium',
            'sony liv': 'sonyliv_premium',
            'voot': 'voot_select',
            'alt balaji': 'alt_balaji',
            'altbalaji': 'alt_balaji',
            'aha': 'aha',
            'sun nxt': 'sun_nxt',
            'sunnxt': 'sun_nxt',
            'airtel xstream': 'airtel_xstream',
            'crunchyroll': 'crunchyroll_free'
        }
        
        return platform_mapping.get(platform_name.lower())
    
    @staticmethod
    def get_combined_availability(title, content_type='movie', year=None):
        """Get combined streaming availability from both APIs"""
        all_options = []
        
        # Get from Watchmode
        watchmode_options = StreamingAvailabilityService.get_watchmode_availability(title, content_type, year)
        all_options.extend(watchmode_options)
        
        # Get from RapidAPI
        rapidapi_options = StreamingAvailabilityService.get_rapidapi_availability(title, content_type, year)
        all_options.extend(rapidapi_options)
        
        # Remove duplicates based on platform
        seen_platforms = set()
        unique_options = []
        
        for option in all_options:
            platform = option['platform']
            if platform not in seen_platforms:
                seen_platforms.add(platform)
                unique_options.append(option)
        
        # Separate free and paid options
        free_options = [opt for opt in unique_options if opt['is_free']]
        paid_options = [opt for opt in unique_options if not opt['is_free']]
        
        return {
            'free_options': free_options,
            'paid_options': paid_options,
            'all_options': unique_options
        }

# Multi-language Content Service
class MultiLanguageContentService:
    @staticmethod
    def get_language_versions(content_id, title):
        """Get different language versions of content"""
        try:
            # Search for the same content in different languages
            language_versions = {}
            
            for lang_key, lang_info in REGIONAL_LANGUAGES.items():
                for lang_code in lang_info['codes']:
                    # Search TMDB for content in specific language
                    search_results = TMDBService.search_content(
                        title, 
                        language=f"{lang_code}-{lang_info['country']}"
                    )
                    
                    if search_results and search_results.get('results'):
                        for result in search_results['results']:
                            # Check if it's the same content (similar title, same year)
                            result_title = result.get('title') or result.get('name', '')
                            if MultiLanguageContentService._is_same_content(title, result_title):
                                # Get streaming links for this language version
                                streaming_links = StreamingAvailabilityService.get_combined_availability(
                                    f"{result_title} {lang_info['name']}", 
                                    'movie' if 'title' in result else 'tv'
                                )
                                
                                language_versions[lang_key] = {
                                    'language': lang_info['name'],
                                    'language_code': lang_code,
                                    'title': result_title,
                                    'tmdb_id': result['id'],
                                    'poster_path': result.get('poster_path'),
                                    'overview': result.get('overview'),
                                    'streaming_links': streaming_links,
                                    'watch_buttons': MultiLanguageContentService._create_watch_buttons(
                                        streaming_links, lang_info['name']
                                    )
                                }
                                break
            
            return language_versions
        except Exception as e:
            logger.error(f"Error getting language versions: {e}")
            return {}
    
    @staticmethod
    def _is_same_content(original_title, compare_title):
        """Check if two titles refer to the same content"""
        import difflib
        similarity = difflib.SequenceMatcher(None, original_title.lower(), compare_title.lower()).ratio()
        return similarity > 0.6  # 60% similarity threshold
    
    @staticmethod
    def _create_watch_buttons(streaming_links, language_name):
        """Create watch buttons for each streaming option"""
        buttons = {
            'free_buttons': [],
            'paid_buttons': []
        }
        
        for option in streaming_links.get('free_options', []):
            buttons['free_buttons'].append({
                'platform': option['platform_name'],
                'url': option['url'],
                'label': f"Watch in {language_name} - {option['platform_name']} (Free)",
                'logo': OTT_PLATFORMS.get(option['platform'], {}).get('logo'),
                'is_free': True
            })
        
        for option in streaming_links.get('paid_options', []):
            buttons['paid_buttons'].append({
                'platform': option['platform_name'],
                'url': option['url'],
                'label': f"Watch in {language_name} - {option['platform_name']}",
                'logo': OTT_PLATFORMS.get(option['platform'], {}).get('logo'),
                'is_free': False,
                'price': option.get('price'),
                'currency': option.get('currency')
            })
        
        return buttons

# Enhanced TMDB Service
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
            'append_to_response': 'credits,videos,similar,watch/providers,translations,alternative_titles'
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
    def get_regional_content(language, region='IN', page=1):
        """Get regional content for specific language"""
        url = f"{TMDBService.BASE_URL}/discover/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'with_original_language': language,
            'region': region,
            'sort_by': 'popularity.desc',
            'page': page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB regional content error: {e}")
        return None
    
    @staticmethod
    def get_genre_content(genre_id, language=None, region='IN', page=1):
        """Get content by genre"""
        url = f"{TMDBService.BASE_URL}/discover/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': genre_id,
            'region': region,
            'sort_by': 'popularity.desc',
            'page': page
        }
        
        if language:
            params['with_original_language'] = language
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB genre content error: {e}")
        return None

# Enhanced Anime Service
class JikanService:
    BASE_URL = 'https://api.jikan.moe/v4'
    
    @staticmethod
    def search_anime(query, page=1):
        url = f"{JikanService.BASE_URL}/anime"
        params = {
            'q': query,
            'page': page,
            'limit': 20,
            'order_by': 'popularity'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            # Rate limiting handling
            elif response.status_code == 429:
                time.sleep(1)
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
            elif response.status_code == 429:
                time.sleep(1)
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
            'limit': 25
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(1)
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Jikan top anime error: {e}")
        return None

# Enhanced Regional Recommendation Engine
class RegionalRecommendationEngine:
    @staticmethod
    def get_regional_movies_by_category(language, category='popular', genre=None, limit=20):
        """Get regional movies by category (trending, popular, all-time hits, new releases)"""
        try:
            lang_info = REGIONAL_LANGUAGES.get(language.lower())
            if not lang_info:
                return []
            
            lang_code = lang_info['codes'][0]  # Primary language code
            
            recommendations = []
            
            if category == 'trending':
                # Get trending content for the language
                trending_data = TMDBService.get_trending('movie')
                if trending_data:
                    for item in trending_data.get('results', []):
                        if item.get('original_language') == lang_code:
                            content = ContentService.save_content_from_tmdb(item, 'movie')
                            if content:
                                recommendations.append(content)
            
            elif category == 'popular':
                # Get popular movies in the language
                regional_data = TMDBService.get_regional_content(lang_code)
                if regional_data:
                    for item in regional_data.get('results', []):
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content:
                            recommendations.append(content)
            
            elif category == 'all_time_hits':
                # Get highest rated movies in the language
                url = f"{TMDBService.BASE_URL}/discover/movie"
                params = {
                    'api_key': TMDB_API_KEY,
                    'with_original_language': lang_code,
                    'sort_by': 'vote_average.desc',
                    'vote_count.gte': 100,  # Minimum votes for credibility
                    'region': lang_info['country']
                }
                
                try:
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        for item in data.get('results', []):
                            content = ContentService.save_content_from_tmdb(item, 'movie')
                            if content:
                                recommendations.append(content)
                except:
                    pass
            
            elif category == 'new_releases':
                # Get recent releases in the language
                current_year = datetime.now().year
                url = f"{TMDBService.BASE_URL}/discover/movie"
                params = {
                    'api_key': TMDB_API_KEY,
                    'with_original_language': lang_code,
                    'primary_release_year': current_year,
                    'sort_by': 'release_date.desc',
                    'region': lang_info['country']
                }
                
                try:
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        for item in data.get('results', []):
                            content = ContentService.save_content_from_tmdb(item, 'movie')
                            if content:
                                recommendations.append(content)
                except:
                    pass
            
            # Filter by genre if specified
            if genre and recommendations:
                genre_filtered = []
                for content in recommendations:
                    if content.genres:
                        content_genres = json.loads(content.genres)
                        if genre.lower() in [g.lower() for g in content_genres]:
                            genre_filtered.append(content)
                recommendations = genre_filtered
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting regional recommendations: {e}")
            return []
    
    @staticmethod
    def get_all_regional_categories(language):
        """Get all categories for a regional language"""
        categories = {
            'trending': RegionalRecommendationEngine.get_regional_movies_by_category(language, 'trending', limit=10),
            'popular': RegionalRecommendationEngine.get_regional_movies_by_category(language, 'popular', limit=10),
            'all_time_hits': RegionalRecommendationEngine.get_regional_movies_by_category(language, 'all_time_hits', limit=10),
            'new_releases': RegionalRecommendationEngine.get_regional_movies_by_category(language, 'new_releases', limit=10)
        }
        
        # Add genre-wise categories
        for genre in MOVIE_GENRES[:8]:  # Top 8 genres
            categories[f"{genre.lower()}_movies"] = RegionalRecommendationEngine.get_regional_movies_by_category(
                language, 'popular', genre, limit=8
            )
        
        return categories

# Enhanced Telegram Service
class TelegramService:
    def __init__(self):
        self.bot = bot
        self.channel_id = TELEGRAM_CHANNEL_ID
        self.templates = {
            'admin_recommendation': self._get_admin_recommendation_template(),
            'trending_update': self._get_trending_update_template(),
            'new_release': self._get_new_release_template()
        }
    
    def send_admin_recommendation(self, content, admin_name, description):
        """Send professional admin recommendation with watch links"""
        try:
            if not self.bot or not self.channel_id:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            # Get content details and streaming links
            streaming_availability = StreamingAvailabilityService.get_combined_availability(
                content.title, 
                content.content_type, 
                content.release_date.year if content.release_date else None
            )
            
            # Get language versions
            language_versions = MultiLanguageContentService.get_language_versions(
                content.id, content.title
            )
            
            # Create the message
            message = self._format_admin_recommendation_message(
                content, admin_name, description, streaming_availability, language_versions
            )
            
            # Create inline keyboard with watch buttons
            keyboard = self._create_watch_keyboard(streaming_availability, language_versions)
            
            # Get poster
            poster_url = self._get_poster_url(content.poster_path)
            
            # Send message
            if poster_url:
                try:
                    self.bot.send_photo(
                        chat_id=self.channel_id,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                except Exception as photo_error:
                    logger.error(f"Failed to send photo: {photo_error}")
                    self.bot.send_message(
                        self.channel_id, 
                        message, 
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
            else:
                self.bot.send_message(
                    self.channel_id, 
                    message, 
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    def _format_admin_recommendation_message(self, content, admin_name, description, streaming_availability, language_versions):
        """Format the admin recommendation message"""
        # Format genre list
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        # Format languages available
        available_languages = list(language_versions.keys()) if language_versions else ['Original']
        
        # Count streaming options
        free_count = len(streaming_availability.get('free_options', []))
        paid_count = len(streaming_availability.get('paid_options', []))
        
        message = f"""🎬 <b>ADMIN'S CHOICE RECOMMENDATION</b> 🎬
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>👤 Recommended by:</b> {admin_name}
<b>🎭 Title:</b> {content.title}
<b>📺 Type:</b> {content.content_type.upper()}

<b>⭐ Rating:</b> {content.rating or 'N/A'}/10
<b>📅 Release:</b> {content.release_date.strftime('%B %d, %Y') if content.release_date else 'N/A'}
<b>🎭 Genres:</b> {', '.join(genres_list[:3]) if genres_list else 'N/A'}
<b>🌍 Languages:</b> {', '.join([lang.title() for lang in available_languages[:3]])}

<b>💬 Admin's Note:</b>
<i>{description}</i>

<b>📖 Synopsis:</b>
{(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

<b>🎥 WATCH OPTIONS:</b>
{"🆓 Free: " + str(free_count) + " platforms" if free_count > 0 else ""}
{"💰 Paid: " + str(paid_count) + " platforms" if paid_count > 0 else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>Click the buttons below to watch directly! 👇</i>

#AdminChoice #MovieRecommendation #CineScope #{content.content_type.title()}"""

        return message
    
    def _create_watch_keyboard(self, streaming_availability, language_versions):
        """Create inline keyboard with watch buttons"""
        try:
            from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
            
            keyboard = InlineKeyboardMarkup(row_width=2)
            
            # Add free options first
            free_options = streaming_availability.get('free_options', [])
            if free_options:
                keyboard.add(InlineKeyboardButton("🆓 FREE OPTIONS", callback_data="free_header"))
                
                free_buttons = []
                for option in free_options[:4]:  # Limit to 4 free options
                    button_text = f"🆓 {option['platform_name']}"
                    free_buttons.append(InlineKeyboardButton(button_text, url=option['url']))
                
                # Add buttons in pairs
                for i in range(0, len(free_buttons), 2):
                    if i + 1 < len(free_buttons):
                        keyboard.add(free_buttons[i], free_buttons[i + 1])
                    else:
                        keyboard.add(free_buttons[i])
            
            # Add paid options
            paid_options = streaming_availability.get('paid_options', [])
            if paid_options:
                keyboard.add(InlineKeyboardButton("💰 PREMIUM OPTIONS", callback_data="paid_header"))
                
                paid_buttons = []
                for option in paid_options[:4]:  # Limit to 4 paid options
                    button_text = f"💰 {option['platform_name']}"
                    paid_buttons.append(InlineKeyboardButton(button_text, url=option['url']))
                
                # Add buttons in pairs
                for i in range(0, len(paid_buttons), 2):
                    if i + 1 < len(paid_buttons):
                        keyboard.add(paid_buttons[i], paid_buttons[i + 1])
                    else:
                        keyboard.add(paid_buttons[i])
            
            # Add language-specific options if available
            if language_versions and len(language_versions) > 1:
                keyboard.add(InlineKeyboardButton("🌍 OTHER LANGUAGES", callback_data="lang_header"))
                
                lang_buttons = []
                for lang_key, lang_data in list(language_versions.items())[:3]:  # Limit to 3 languages
                    if lang_data.get('streaming_links', {}).get('all_options'):
                        button_text = f"🎬 {lang_data['language']}"
                        # Use the first available streaming link
                        first_option = lang_data['streaming_links']['all_options'][0]
                        lang_buttons.append(InlineKeyboardButton(button_text, url=first_option['url']))
                
                for button in lang_buttons:
                    keyboard.add(button)
            
            return keyboard
            
        except Exception as e:
            logger.error(f"Error creating keyboard: {e}")
            return None
    
    def _get_poster_url(self, poster_path):
        """Get full poster URL"""
        if not poster_path:
            return None
        
        if poster_path.startswith('http'):
            return poster_path
        else:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    
    def _get_admin_recommendation_template(self):
        """Get admin recommendation message template"""
        return {
            'format': 'html',
            'max_length': 1024,
            'include_poster': True,
            'include_buttons': True
        }
    
    def _get_trending_update_template(self):
        """Get trending update template"""
        return {
            'format': 'html',
            'max_length': 512,
            'include_poster': False,
            'include_buttons': False
        }
    
    def _get_new_release_template(self):
        """Get new release template"""
        return {
            'format': 'html',
            'max_length': 768,
            'include_poster': True,
            'include_buttons': True
        }
    
    def send_trending_update(self, trending_content):
        """Send trending content update"""
        try:
            if not self.bot or not self.channel_id:
                return False
            
            message = f"""📈 <b>TRENDING NOW</b> 📈
━━━━━━━━━━━━━━━━━━━━━━━

"""
            
            for i, content in enumerate(trending_content[:5], 1):
                genres = json.loads(content.genres or '[]')
                message += f"""<b>{i}. {content.title}</b>
⭐ {content.rating or 'N/A'}/10 | 🎭 {', '.join(genres[:2])}

"""
            
            message += """━━━━━━━━━━━━━━━━━━━━━━━
#Trending #Movies #CineScope"""
            
            self.bot.send_message(self.channel_id, message, parse_mode='HTML')
            return True
            
        except Exception as e:
            logger.error(f"Error sending trending update: {e}")
            return False

# Enhanced Content Service
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
            
            # Get streaming availability
            title = tmdb_data.get('title') or tmdb_data.get('name')
            release_year = None
            if tmdb_data.get('release_date') or tmdb_data.get('first_air_date'):
                try:
                    date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
                    release_year = datetime.strptime(date_str, '%Y-%m-%d').year
                except:
                    pass
            
            streaming_availability = StreamingAvailabilityService.get_combined_availability(
                title, content_type, release_year
            )
            
            # Get language versions
            language_versions = MultiLanguageContentService.get_language_versions(None, title)
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
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
                ott_platforms=json.dumps(streaming_availability.get('all_options', [])),
                streaming_links=json.dumps(streaming_availability),
                watch_providers=json.dumps(language_versions)
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
        # TMDB Genre ID mapping
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western',
            10759: 'Action', 10762: 'Kids', 10763: 'News', 10764: 'Reality',
            10765: 'Sci-Fi', 10766: 'Soap', 10767: 'Talk', 10768: 'War'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]

# API Routes

# Enhanced Content Details Route
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
        
        # Get fresh streaming availability
        streaming_availability = StreamingAvailabilityService.get_combined_availability(
            content.title,
            content.content_type,
            content.release_date.year if content.release_date else None
        )
        
        # Get language versions with watch links
        language_versions = MultiLanguageContentService.get_language_versions(content.id, content.title)
        
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
            'streaming_availability': streaming_availability,
            'language_versions': language_versions,
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Enhanced Anime Details Route
@app.route('/api/anime/<int:anime_id>', methods=['GET'])
def get_anime_details(anime_id):
    try:
        # Get anime details from Jikan API
        anime_data = JikanService.get_anime_details(anime_id)
        
        if not anime_data:
            return jsonify({'error': 'Anime not found'}), 404
        
        anime = anime_data.get('data', {})
        
        # Record view interaction for anonymous users
        session_id = get_session_id()
        
        # Create a temporary content object for anime
        anime_content = {
            'id': f"anime_{anime_id}",
            'mal_id': anime_id,
            'title': anime.get('title'),
            'title_english': anime.get('title_english'),
            'title_japanese': anime.get('title_japanese'),
            'content_type': 'anime',
            'type': anime.get('type'),
            'episodes': anime.get('episodes'),
            'status': anime.get('status'),
            'aired': anime.get('aired', {}),
            'duration': anime.get('duration'),
            'rating': anime.get('score'),
            'scored_by': anime.get('scored_by'),
            'rank': anime.get('rank'),
            'popularity': anime.get('popularity'),
            'synopsis': anime.get('synopsis'),
            'genres': [genre['name'] for genre in anime.get('genres', [])],
            'themes': [theme['name'] for theme in anime.get('themes', [])],
            'demographics': [demo['name'] for demo in anime.get('demographics', [])],
            'studios': [studio['name'] for studio in anime.get('studios', [])],
            'producers': [producer['name'] for producer in anime.get('producers', [])],
            'images': anime.get('images', {}),
            'trailer': anime.get('trailer', {}),
            'streaming': anime.get('streaming', []),
            'external': anime.get('external', [])
        }
        
        # Get streaming availability for anime
        streaming_availability = {
            'free_options': [
                {
                    'platform': 'crunchyroll_free',
                    'platform_name': 'Crunchyroll',
                    'is_free': True,
                    'url': f"https://crunchyroll.com/search?q={anime.get('title', '').replace(' ', '%20')}",
                    'type': 'free'
                },
                {
                    'platform': 'youtube',
                    'platform_name': 'YouTube',
                    'is_free': True,
                    'url': f"https://youtube.com/results?search_query={anime.get('title', '').replace(' ', '+')}+anime",
                    'type': 'free'
                }
            ],
            'paid_options': [
                {
                    'platform': 'crunchyroll_premium',
                    'platform_name': 'Crunchyroll Premium',
                    'is_free': False,
                    'url': f"https://crunchyroll.com/search?q={anime.get('title', '').replace(' ', '%20')}",
                    'type': 'subscription'
                }
            ]
        }
        
        # Add external streaming links if available
        for stream in anime.get('streaming', []):
            streaming_availability['paid_options'].append({
                'platform': 'external',
                'platform_name': stream.get('name', 'External'),
                'is_free': False,
                'url': stream.get('url'),
                'type': 'external'
            })
        
        return jsonify({
            'anime_details': anime_content,
            'streaming_availability': streaming_availability,
            'watch_buttons': {
                'free_buttons': [
                    {
                        'platform': opt['platform_name'],
                        'url': opt['url'],
                        'label': f"Watch on {opt['platform_name']} (Free)",
                        'is_free': True
                    }
                    for opt in streaming_availability['free_options']
                ],
                'paid_buttons': [
                    {
                        'platform': opt['platform_name'],
                        'url': opt['url'],
                        'label': f"Watch on {opt['platform_name']}",
                        'is_free': False
                    }
                    for opt in streaming_availability['paid_options']
                ]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Anime details error: {e}")
        return jsonify({'error': 'Failed to get anime details'}), 500

# Enhanced Regional Content Routes
@app.route('/api/regional/<language>', methods=['GET'])
def get_regional_content(language):
    try:
        category = request.args.get('category', 'popular')
        genre = request.args.get('genre')
        limit = int(request.args.get('limit', 20))
        page = int(request.args.get('page', 1))
        
        # Validate language
        if language.lower() not in REGIONAL_LANGUAGES:
            return jsonify({'error': 'Language not supported'}), 400
        
        if category == 'all_categories':
            # Get all categories for the language
            all_categories = RegionalRecommendationEngine.get_all_regional_categories(language)
            
            # Format response
            formatted_categories = {}
            for cat_name, content_list in all_categories.items():
                formatted_categories[cat_name] = [
                    {
                        'id': content.id,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                        'overview': content.overview[:150] + '...' if content.overview else '',
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'streaming_links': json.loads(content.streaming_links or '{}')
                    }
                    for content in content_list
                ]
            
            return jsonify({
                'language': REGIONAL_LANGUAGES[language.lower()]['name'],
                'categories': formatted_categories
            }), 200
        
        else:
            # Get specific category
            recommendations = RegionalRecommendationEngine.get_regional_movies_by_category(
                language, category, genre, limit
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
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'streaming_links': json.loads(content.streaming_links or '{}')
                })
            
            return jsonify({
                'language': REGIONAL_LANGUAGES[language.lower()]['name'],
                'category': category,
                'recommendations': result
            }), 200
        
    except Exception as e:
        logger.error(f"Regional content error: {e}")
        return jsonify({'error': 'Failed to get regional content'}), 500

@app.route('/api/regional/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported regional languages"""
    try:
        languages = []
        for lang_key, lang_info in REGIONAL_LANGUAGES.items():
            languages.append({
                'key': lang_key,
                'name': lang_info['name'],
                'industry': lang_info['industry'],
                'country': lang_info['country'],
                'primary_platforms': lang_info['primary_platforms']
            })
        
        return jsonify({'supported_languages': languages}), 200
        
    except Exception as e:
        logger.error(f"Error getting languages: {e}")
        return jsonify({'error': 'Failed to get languages'}), 500

@app.route('/api/genres', methods=['GET'])
def get_movie_genres():
    """Get list of movie genres"""
    try:
        return jsonify({'genres': MOVIE_GENRES}), 200
    except Exception as e:
        logger.error(f"Error getting genres: {e}")
        return jsonify({'error': 'Failed to get genres'}), 500

# Rest of the existing routes remain the same...
# (I'll continue with the other routes in the next part due to length constraints)

# Authentication Routes (unchanged)
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

# Enhanced Admin Routes
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
        
        # Send to Telegram channel with enhanced service
        telegram_service = TelegramService()
        telegram_success = telegram_service.send_admin_recommendation(
            content, current_user.username, data['description']
        )
        
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
        'features': {
            'streaming_availability': True,
            'multi_language_support': True,
            'regional_content': True,
            'anime_support': True,
            'telegram_integration': True
        }
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

create_tables()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)