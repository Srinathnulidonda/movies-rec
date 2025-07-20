# backend/app.py
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

# RapidAPI Streaming Availability
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
    ott_platforms = db.Column(db.Text)  # JSON string
    streaming_links = db.Column(db.Text)  # JSON string with language-specific links
    available_languages = db.Column(db.Text)  # JSON string
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
    'youtube': {'name': 'YouTube', 'is_free': True, 'url': 'https://youtube.com', 'icon': '📺'},
    'mxplayer': {'name': 'MX Player', 'is_free': True, 'url': 'https://mxplayer.in', 'icon': '🎬'},
    'jiocinema': {'name': 'JioCinema', 'is_free': True, 'url': 'https://jiocinema.com', 'icon': '🎪'},
    'sonyliv_free': {'name': 'SonyLIV (Free)', 'is_free': True, 'url': 'https://sonyliv.com', 'icon': '📡'},
    'zee5_free': {'name': 'ZEE5 (Free)', 'is_free': True, 'url': 'https://zee5.com', 'icon': '🌟'},
    'airtel_xstream': {'name': 'Airtel Xstream', 'is_free': True, 'url': 'https://airtelxstream.in', 'icon': '📻'},
    'crunchyroll_free': {'name': 'Crunchyroll (Free)', 'is_free': True, 'url': 'https://crunchyroll.com', 'icon': '🎌'},
    
    # Paid Platforms
    'netflix': {'name': 'Netflix', 'is_free': False, 'url': 'https://netflix.com', 'icon': '🎥'},
    'amazon_prime': {'name': 'Prime Video', 'is_free': False, 'url': 'https://primevideo.com', 'icon': '🏆'},
    'disney_hotstar': {'name': 'Disney+ Hotstar', 'is_free': False, 'url': 'https://hotstar.com', 'icon': '🏰'},
    'zee5_premium': {'name': 'ZEE5 Premium', 'is_free': False, 'url': 'https://zee5.com', 'icon': '⭐'},
    'sonyliv_premium': {'name': 'SonyLIV Premium', 'is_free': False, 'url': 'https://sonyliv.com', 'icon': '🎭'},
    'aha': {'name': 'Aha', 'is_free': False, 'url': 'https://aha.video', 'icon': '🌺'},
    'sun_nxt': {'name': 'Sun NXT', 'is_free': False, 'url': 'https://sunnxt.com', 'icon': '☀️'},
    'crunchyroll_premium': {'name': 'Crunchyroll Premium', 'is_free': False, 'url': 'https://crunchyroll.com', 'icon': '🎌'},
}

# Language Mapping with Priority
LANGUAGE_PRIORITY = {
    'telugu': {'priority': 1, 'codes': ['te', 'tel'], 'flag': '🇮🇳', 'name': 'Telugu'},
    'english': {'priority': 2, 'codes': ['en', 'eng'], 'flag': '🇺🇸', 'name': 'English'},
    'hindi': {'priority': 3, 'codes': ['hi', 'hin'], 'flag': '🇮🇳', 'name': 'Hindi'},
    'tamil': {'priority': 4, 'codes': ['ta', 'tam'], 'flag': '🇮🇳', 'name': 'Tamil'},
    'malayalam': {'priority': 5, 'codes': ['ml', 'mal'], 'flag': '🇮🇳', 'name': 'Malayalam'},
    'kannada': {'priority': 6, 'codes': ['kn', 'kan'], 'flag': '🇮🇳', 'name': 'Kannada'},
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
    def get_streaming_availability(title, imdb_id=None, tmdb_id=None):
        """Get streaming availability for a title across multiple platforms and languages"""
        try:
            streaming_data = {
                'platforms': [],
                'languages': {},
                'free_options': [],
                'paid_options': []
            }
            
            # Try RapidAPI Streaming Availability first
            rapidapi_data = StreamingAvailabilityService._get_rapidapi_availability(title, imdb_id)
            if rapidapi_data:
                streaming_data.update(rapidapi_data)
            
            # Try WatchMode API as backup
            watchmode_data = StreamingAvailabilityService._get_watchmode_availability(title)
            if watchmode_data:
                # Merge watchmode data with existing data
                StreamingAvailabilityService._merge_streaming_data(streaming_data, watchmode_data)
            
            # Add regional platform checks
            regional_data = StreamingAvailabilityService._get_regional_availability(title)
            if regional_data:
                StreamingAvailabilityService._merge_streaming_data(streaming_data, regional_data)
            
            return streaming_data
            
        except Exception as e:
            logger.error(f"Streaming availability error: {e}")
            return StreamingAvailabilityService._get_fallback_availability(title)
    
    @staticmethod
    def _get_rapidapi_availability(title, imdb_id=None):
        """Get availability from RapidAPI Streaming Availability"""
        try:
            headers = {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': RAPIDAPI_HOST
            }
            
            # Search by title first
            search_url = f"https://{RAPIDAPI_HOST}/v2/search/title"
            params = {
                'title': title,
                'country': 'in',  # India
                'show_type': 'all',
                'output_language': 'en'
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('result'):
                    # Get the first match
                    show = data['result'][0]
                    
                    streaming_data = {
                        'platforms': [],
                        'languages': {},
                        'free_options': [],
                        'paid_options': []
                    }
                    
                    # Process streaming options
                    streaming_info = show.get('streamingInfo', {})
                    
                    for country, platforms in streaming_info.items():
                        if country == 'in':  # Focus on India
                            for platform_key, platform_data in platforms.items():
                                for option in platform_data:
                                    platform_info = {
                                        'platform': platform_key,
                                        'type': option.get('streamingType', 'subscription'),
                                        'link': option.get('link', ''),
                                        'quality': option.get('quality', 'hd'),
                                        'audios': option.get('audios', []),
                                        'subtitles': option.get('subtitles', [])
                                    }
                                    
                                    # Categorize by free/paid
                                    if option.get('streamingType') == 'free':
                                        streaming_data['free_options'].append(platform_info)
                                    else:
                                        streaming_data['paid_options'].append(platform_info)
                                    
                                    streaming_data['platforms'].append(platform_info)
                                    
                                    # Group by languages
                                    for audio in option.get('audios', []):
                                        if audio not in streaming_data['languages']:
                                            streaming_data['languages'][audio] = []
                                        streaming_data['languages'][audio].append(platform_info)
                    
                    return streaming_data
            
        except Exception as e:
            logger.error(f"RapidAPI streaming error: {e}")
        
        return None
    
    @staticmethod
    def _get_watchmode_availability(title):
        """Get availability from WatchMode API"""
        try:
            if not WATCHMODE_API_KEY or WATCHMODE_API_KEY == 'your_watchmode_api_key':
                return None
                
            # Search for title
            search_url = "https://api.watchmode.com/v1/search/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'search_field': 'name',
                'search_value': title
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('title_results'):
                    # Get the first match
                    title_id = data['title_results'][0]['id']
                    
                    # Get streaming sources
                    sources_url = f"https://api.watchmode.com/v1/title/{title_id}/sources/"
                    params = {
                        'apiKey': WATCHMODE_API_KEY,
                        'regions': 'IN'  # India
                    }
                    
                    sources_response = requests.get(sources_url, params=params, timeout=10)
                    
                    if sources_response.status_code == 200:
                        sources_data = sources_response.json()
                        
                        streaming_data = {
                            'platforms': [],
                            'languages': {},
                            'free_options': [],
                            'paid_options': []
                        }
                        
                        for source in sources_data:
                            platform_info = {
                                'platform': source.get('name', '').lower().replace(' ', '_'),
                                'type': 'free' if source.get('type') == 'free' else 'subscription',
                                'link': source.get('web_url', ''),
                                'price': source.get('price', ''),
                                'format': source.get('format', '')
                            }
                            
                            if source.get('type') == 'free':
                                streaming_data['free_options'].append(platform_info)
                            else:
                                streaming_data['paid_options'].append(platform_info)
                            
                            streaming_data['platforms'].append(platform_info)
                        
                        return streaming_data
                        
        except Exception as e:
            logger.error(f"WatchMode API error: {e}")
        
        return None
    
    @staticmethod
    def _get_regional_availability(title):
        """Check regional Indian platforms"""
        try:
            # Mock regional platform availability
            # In a real implementation, you'd check specific regional APIs
            regional_platforms = [
                {
                    'platform': 'aha',
                    'type': 'subscription',
                    'link': f"https://aha.video/search?query={title.replace(' ', '%20')}",
                    'languages': ['telugu', 'tamil']
                },
                {
                    'platform': 'sun_nxt',
                    'type': 'subscription', 
                    'link': f"https://sunnxt.com/search?q={title.replace(' ', '%20')}",
                    'languages': ['tamil', 'malayalam', 'kannada']
                },
                {
                    'platform': 'zee5_premium',
                    'type': 'subscription',
                    'link': f"https://zee5.com/search?q={title.replace(' ', '%20')}",
                    'languages': ['hindi', 'tamil', 'telugu', 'kannada', 'malayalam']
                }
            ]
            
            streaming_data = {
                'platforms': [],
                'languages': {},
                'free_options': [],
                'paid_options': []
            }
            
            # Randomly assign some platforms (in real implementation, this would be actual API calls)
            available_platforms = random.sample(regional_platforms, random.randint(1, 2))
            
            for platform in available_platforms:
                streaming_data['paid_options'].append(platform)
                streaming_data['platforms'].append(platform)
                
                for lang in platform.get('languages', []):
                    if lang not in streaming_data['languages']:
                        streaming_data['languages'][lang] = []
                    streaming_data['languages'][lang].append(platform)
            
            return streaming_data
            
        except Exception as e:
            logger.error(f"Regional availability error: {e}")
        
        return None
    
    @staticmethod
    def _merge_streaming_data(main_data, new_data):
        """Merge streaming data from multiple sources"""
        try:
            if not new_data:
                return
            
            # Merge platforms
            main_data['platforms'].extend(new_data.get('platforms', []))
            main_data['free_options'].extend(new_data.get('free_options', []))
            main_data['paid_options'].extend(new_data.get('paid_options', []))
            
            # Merge languages
            for lang, platforms in new_data.get('languages', {}).items():
                if lang not in main_data['languages']:
                    main_data['languages'][lang] = []
                main_data['languages'][lang].extend(platforms)
            
            # Remove duplicates
            main_data['platforms'] = list({json.dumps(p, sort_keys=True): p for p in main_data['platforms']}.values())
            main_data['free_options'] = list({json.dumps(p, sort_keys=True): p for p in main_data['free_options']}.values())
            main_data['paid_options'] = list({json.dumps(p, sort_keys=True): p for p in main_data['paid_options']}.values())
            
        except Exception as e:
            logger.error(f"Merge streaming data error: {e}")
    
    @staticmethod
    def _get_fallback_availability(title):
        """Fallback availability when APIs fail"""
        # Provide basic platform suggestions based on content type and language
        fallback_data = {
            'platforms': [
                {
                    'platform': 'youtube',
                    'type': 'free',
                    'link': f"https://youtube.com/results?search_query={title.replace(' ', '+')}+full+movie",
                    'languages': ['hindi', 'english', 'telugu', 'tamil']
                },
                {
                    'platform': 'netflix',
                    'type': 'subscription',
                    'link': f"https://netflix.com/search?q={title.replace(' ', '%20')}",
                    'languages': ['english', 'hindi']
                }
            ],
            'languages': {
                'english': [{'platform': 'netflix', 'type': 'subscription'}],
                'hindi': [{'platform': 'youtube', 'type': 'free'}]
            },
            'free_options': [
                {
                    'platform': 'youtube',
                    'type': 'free',
                    'link': f"https://youtube.com/results?search_query={title.replace(' ', '+')}+full+movie"
                }
            ],
            'paid_options': [
                {
                    'platform': 'netflix',
                    'type': 'subscription',
                    'link': f"https://netflix.com/search?q={title.replace(' ', '%20')}"
                }
            ]
        }
        
        return fallback_data

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
            'append_to_response': 'credits,videos,similar,watch/providers,alternative_titles,translations'
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
    def get_regional_content(language='te', page=1):
        """Get content in specific regional language"""
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
            logger.error(f"TMDB regional content error: {e}")
        return None
    
    @staticmethod
    def get_by_genre(genre_id, content_type='movie', page=1, region=None):
        """Get content by genre"""
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': genre_id,
            'sort_by': 'popularity.desc',
            'page': page
        }
        if region:
            params['region'] = region
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB genre content error: {e}")
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
                # Update streaming availability
                ContentService._update_streaming_availability(existing)
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
            streaming_data = StreamingAvailabilityService.get_streaming_availability(
                title, 
                tmdb_data.get('imdb_id'),
                tmdb_data.get('id')
            )
            
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
                streaming_links=json.dumps(streaming_data),
                available_languages=json.dumps(list(streaming_data.get('languages', {}).keys()))
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
            
            # Get streaming availability for anime
            streaming_data = StreamingAvailabilityService.get_streaming_availability(
                anime_data.get('title'), 
                anime_data.get('mal_id')
            )
            
            # Create anime content
            content = Content(
                mal_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps([genre['name'] for genre in anime_data.get('genres', [])]),
                languages=json.dumps(['japanese']),
                release_date=datetime.strptime(anime_data.get('aired', {}).get('from', '1900-01-01T00:00:00+00:00')[:10], '%Y-%m-%d').date() if anime_data.get('aired', {}).get('from') else None,
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                streaming_links=json.dumps(streaming_data),
                available_languages=json.dumps(list(streaming_data.get('languages', {}).keys()))
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def _update_streaming_availability(content):
        """Update streaming availability for existing content"""
        try:
            streaming_data = StreamingAvailabilityService.get_streaming_availability(
                content.title,
                content.imdb_id,
                content.tmdb_id
            )
            
            content.streaming_links = json.dumps(streaming_data)
            content.available_languages = json.dumps(list(streaming_data.get('languages', {}).keys()))
            content.updated_at = datetime.utcnow()
            
            db.session.commit()
        except Exception as e:
            logger.error(f"Error updating streaming availability: {e}")
    
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
    def get_trending_recommendations(limit=20, content_type='all', prioritize_languages=None):
        try:
            if not prioritize_languages:
                prioritize_languages = ['te', 'en']  # Telugu and English priority
            
            recommendations = []
            
            # Get trending from TMDB for each priority language
            for lang in prioritize_languages:
                trending_data = TMDBService.get_trending(content_type=content_type)
                if trending_data:
                    for item in trending_data.get('results', []):
                        if len(recommendations) >= limit:
                            break
                        
                        content_type_detected = 'movie' if 'title' in item else 'tv'
                        content = ContentService.save_content_from_tmdb(item, content_type_detected)
                        if content:
                            recommendations.append(content)
            
            # Fill remaining with general trending if needed
            if len(recommendations) < limit:
                general_trending = TMDBService.get_trending(content_type=content_type)
                if general_trending:
                    for item in general_trending.get('results', []):
                        if len(recommendations) >= limit:
                            break
                        
                        content_type_detected = 'movie' if 'title' in item else 'tv'
                        content = ContentService.save_content_from_tmdb(item, content_type_detected)
                        if content and content not in recommendations:
                            recommendations.append(content)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_regional_recommendations(language='telugu', limit=20):
        try:
            recommendations = []
            
            # Language code mapping
            lang_codes = {
                'telugu': 'te',
                'tamil': 'ta', 
                'hindi': 'hi',
                'kannada': 'kn',
                'malayalam': 'ml',
                'english': 'en'
            }
            
            lang_code = lang_codes.get(language.lower(), 'en')
            
            # Get regional content from TMDB
            regional_data = TMDBService.get_regional_content(language=lang_code)
            if regional_data:
                for item in regional_data.get('results', []):
                    if len(recommendations) >= limit:
                        break
                    
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
            
            # Also search for language-specific content
            search_queries = {
                'telugu': ['tollywood', 'telugu movie', 'telugu cinema'],
                'tamil': ['kollywood', 'tamil movie', 'tamil cinema'],
                'hindi': ['bollywood', 'hindi movie', 'hindi cinema'],
                'kannada': ['sandalwood', 'kannada movie', 'kannada cinema'],
                'malayalam': ['mollywood', 'malayalam movie', 'malayalam cinema']
            }
            
            queries = search_queries.get(language.lower(), [language])
            
            for query in queries:
                if len(recommendations) >= limit:
                    break
                    
                search_results = TMDBService.search_content(query)
                if search_results:
                    for item in search_results.get('results', []):
                        if len(recommendations) >= limit:
                            break
                        
                        content_type_detected = 'movie' if 'title' in item else 'tv'
                        content = ContentService.save_content_from_tmdb(item, content_type_detected)
                        if content and content not in recommendations:
                            recommendations.append(content)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting regional recommendations: {e}")
            return []
    
    @staticmethod
    def get_genre_recommendations(genre, limit=20):
        try:
            recommendations = []
            
            # Genre ID mapping
            genre_ids = {
                'action': 28, 'adventure': 12, 'animation': 16, 'comedy': 35,
                'crime': 80, 'documentary': 99, 'drama': 18, 'family': 10751,
                'fantasy': 14, 'history': 36, 'horror': 27, 'music': 10402,
                'mystery': 9648, 'romance': 10749, 'sci-fi': 878, 'thriller': 53,
                'war': 10752, 'western': 37
            }
            
            genre_id = genre_ids.get(genre.lower())
            if not genre_id:
                return []
            
            # Get movies by genre
            genre_data = TMDBService.get_by_genre(genre_id, 'movie')
            if genre_data:
                for item in genre_data.get('results', []):
                    if len(recommendations) >= limit // 2:
                        break
                    
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
            
            # Get TV shows by genre
            tv_genre_data = TMDBService.get_by_genre(genre_id, 'tv')
            if tv_genre_data:
                for item in tv_genre_data.get('results', []):
                    if len(recommendations) >= limit:
                        break
                    
                    content = ContentService.save_content_from_tmdb(item, 'tv')
                    if content:
                        recommendations.append(content)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting genre recommendations: {e}")
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
        """Get new releases from the last 60 days"""
        try:
            recommendations = []
            
            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Search for recent releases in priority languages
            priority_languages = ['te', 'en', 'hi', 'ta', 'ml', 'kn']
            
            for lang in priority_languages:
                regional_data = TMDBService.get_regional_content(language=lang)
                if regional_data:
                    for item in regional_data.get('results', []):
                        if len(recommendations) >= limit:
                            break
                        
                        # Check release date
                        release_date_str = item.get('release_date')
                        if release_date_str:
                            try:
                                release_date = datetime.strptime(release_date_str, '%Y-%m-%d').date()
                                if start_date <= release_date <= end_date:
                                    content = ContentService.save_content_from_tmdb(item, 'movie')
                                    if content and content not in recommendations:
                                        recommendations.append(content)
                            except:
                                pass
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return []
    
    @staticmethod
    def get_critics_choice(limit=20):
        """Get critically acclaimed content"""
        try:
            recommendations = []
            
            # Get top rated movies
            url = f"{TMDBService.BASE_URL}/movie/top_rated"
            params = {
                'api_key': TMDB_API_KEY,
                'page': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('results', []):
                    if len(recommendations) >= limit // 2:
                        break
                    
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
            
            # Get top rated TV shows
            url = f"{TMDBService.BASE_URL}/tv/top_rated"
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('results', []):
                    if len(recommendations) >= limit:
                        break
                    
                    content = ContentService.save_content_from_tmdb(item, 'tv')
                    if content:
                        recommendations.append(content)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting critics choice: {e}")
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
            
            # Get streaming information
            streaming_data = {}
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                except:
                    streaming_data = {}
            
            # Create language-specific watch buttons
            language_buttons = []
            languages = streaming_data.get('languages', {})
            
            # Sort languages by priority
            sorted_languages = sorted(languages.keys(), key=lambda x: LANGUAGE_PRIORITY.get(x, {}).get('priority', 999))
            
            for lang in sorted_languages[:4]:  # Limit to 4 languages
                platforms = languages[lang]
                if platforms:
                    platform = platforms[0]  # Get first available platform
                    platform_name = STREAMING_PLATFORMS.get(platform.get('platform', ''), {}).get('name', platform.get('platform', ''))
                    platform_icon = STREAMING_PLATFORMS.get(platform.get('platform', ''), {}).get('icon', '🎬')
                    is_free = platform.get('type') == 'free'
                    
                    lang_info = LANGUAGE_PRIORITY.get(lang, {})
                    lang_flag = lang_info.get('flag', '🎬')
                    lang_name = lang_info.get('name', lang.title())
                    
                    button_text = f"{lang_flag} {lang_name} ({'Free' if is_free else platform_name})"
                    language_buttons.append(f"[{button_text}]")
            
            # Create free/paid platform info
            free_platforms = []
            paid_platforms = []
            
            for platform in streaming_data.get('free_options', []):
                platform_name = STREAMING_PLATFORMS.get(platform.get('platform', ''), {}).get('name', platform.get('platform', ''))
                if platform_name not in free_platforms:
                    free_platforms.append(platform_name)
            
            for platform in streaming_data.get('paid_options', []):
                platform_name = STREAMING_PLATFORMS.get(platform.get('platform', ''), {}).get('name', platform.get('platform', ''))
                if platform_name not in paid_platforms:
                    paid_platforms.append(platform_name)
            
            # Create availability text
            availability_text = ""
            if free_platforms:
                availability_text += f"🎬 Free on {', '.join(free_platforms[:2])}!"
            elif paid_platforms:
                availability_text += f"💎 Available on {', '.join(paid_platforms[:2])}"
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create message
            message = f"""**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}
{availability_text}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

🎯 **Choose Your Language to Watch:**
{' '.join(language_buttons)}

[📺 Watch Trailer] [⭐ More Details]

For More - https://movierecommendations.com

#AdminChoice #MovieRecommendation #CineScope #{content.content_type.title()}"""
            
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
                    
                    # Get streaming data
                    streaming_data = {}
                    if content.streaming_links:
                        try:
                            streaming_data = json.loads(content.streaming_links)
                        except:
                            pass
                    
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
                        'streaming_data': streaming_data,
                        'available_languages': json.loads(content.available_languages or '[]')
                    })
        
        # Add anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                content = ContentService.save_anime_content(anime)
                if content:
                    streaming_data = {}
                    if content.streaming_links:
                        try:
                            streaming_data = json.loads(content.streaming_links)
                        except:
                            pass
                    
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
                        'streaming_data': streaming_data,
                        'available_languages': json.loads(content.available_languages or '[]')
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
        if additional_details:
            if 'similar' in additional_details:
                # TMDB similar content
                for item in additional_details['similar']['results'][:5]:
                    similar = ContentService.save_content_from_tmdb(item, content.content_type)
                    if similar:
                        similar_content.append({
                            'id': similar.id,
                            'title': similar.title,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{similar.poster_path}" if similar.poster_path else None,
                            'rating': similar.rating
                        })
            elif content.content_type == 'anime' and 'data' in additional_details:
                # For anime, we might need to get recommendations differently
                # This would be implemented based on Jikan API structure
                pass
        
        # Get streaming data
        streaming_data = {}
        if content.streaming_links:
            try:
                streaming_data = json.loads(content.streaming_links)
            except:
                pass
        
        # Format language-specific watch links
        language_watch_links = []
        if streaming_data.get('languages'):
            for lang, platforms in streaming_data['languages'].items():
                if platforms:
                    platform = platforms[0]  # Get first available platform
                    lang_info = LANGUAGE_PRIORITY.get(lang, {})
                    platform_info = STREAMING_PLATFORMS.get(platform.get('platform', ''), {})
                    
                    language_watch_links.append({
                        'language': lang,
                        'language_name': lang_info.get('name', lang.title()),
                        'language_flag': lang_info.get('flag', '🎬'),
                        'platform': platform.get('platform'),
                        'platform_name': platform_info.get('name', platform.get('platform', '')),
                        'is_free': platform.get('type') == 'free',
                        'link': platform.get('link', ''),
                        'icon': platform_info.get('icon', '🎬')
                    })
        
        db.session.commit()
        
        response_data = {
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
            'streaming_data': streaming_data,
            'language_watch_links': language_watch_links,
            'available_languages': json.loads(content.available_languages or '[]'),
            'trailers': trailers,
            'similar_content': similar_content
        }
        
        # Add TMDB-specific data
        if additional_details and content.content_type != 'anime':
            response_data.update({
                'cast': additional_details.get('credits', {}).get('cast', [])[:10],
                'crew': additional_details.get('credits', {}).get('crew', [])[:5]
            })
        elif additional_details and content.content_type == 'anime':
            # Add anime-specific data
            anime_data = additional_details.get('data', {})
            response_data.update({
                'episodes': anime_data.get('episodes'),
                'status': anime_data.get('status'),
                'studios': [studio['name'] for studio in anime_data.get('studios', [])],
                'source': anime_data.get('source'),
                'duration': anime_data.get('duration')
            })
        
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
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = []
        for content in recommendations:
            streaming_data = {}
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_data': streaming_data,
                'available_languages': json.loads(content.available_languages or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/recommendations/regional/<language>', methods=['GET'])
def get_regional(language):
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_regional_recommendations(language, limit)
        
        result = []
        for content in recommendations:
            streaming_data = {}
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_data': streaming_data,
                'available_languages': json.loads(content.available_languages or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Regional recommendations error: {e}")
        return jsonify({'error': 'Failed to get regional recommendations'}), 500

@app.route('/api/recommendations/genre/<genre>', methods=['GET'])
def get_genre_recommendations(genre):
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_genre_recommendations(genre, limit)
        
        result = []
        for content in recommendations:
            streaming_data = {}
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_data': streaming_data,
                'available_languages': json.loads(content.available_languages or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Genre recommendations error: {e}")
        return jsonify({'error': 'Failed to get genre recommendations'}), 500

@app.route('/api/recommendations/anime', methods=['GET'])
def get_anime():
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_anime_recommendations(limit)
        
        result = []
        for content in recommendations:
            streaming_data = {}
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                except:
                    pass
            
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
                'streaming_data': streaming_data,
                'available_languages': json.loads(content.available_languages or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Anime recommendations error: {e}")
        return jsonify({'error': 'Failed to get anime recommendations'}), 500

@app.route('/api/recommendations/new-releases', methods=['GET'])
def get_new_releases():
    try:
        limit = int(request.args.get('limit', 20))
        days = int(request.args.get('days', 60))
        
        recommendations = RecommendationEngine.get_new_releases(limit, days)
        
        result = []
        for content in recommendations:
            streaming_data = {}
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_data': streaming_data,
                'available_languages': json.loads(content.available_languages or '[]')
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
            streaming_data = {}
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_data': streaming_data,
                'available_languages': json.loads(content.available_languages or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Critics choice error: {e}")
        return jsonify({'error': 'Failed to get critics choice'}), 500

# Anonymous User Recommendations
@app.route('/api/recommendations/anonymous', methods=['GET'])
def get_anonymous_recommendations():
    try:
        session_id = get_session_id()
        limit = int(request.args.get('limit', 20))
        
        # Get user location for regional content
        location = get_user_location(request.remote_addr)
        
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
                genre_recs = RecommendationEngine.get_genre_recommendations(genre.lower(), limit=7)
                recommendations.extend(genre_recs)
        
        # Add regional content based on location
        if location and location.get('country') == 'India':
            # Prioritize Telugu and English for Indian users
            regional_recs = RecommendationEngine.get_regional_recommendations('telugu', limit=5)
            recommendations.extend(regional_recs)
            
            english_recs = RecommendationEngine.get_regional_recommendations('english', limit=5)
            recommendations.extend(english_recs)
        
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
        
        # Format response
        result = []
        for content in unique_recommendations:
            streaming_data = {}
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_data': streaming_data,
                'available_languages': json.loads(content.available_languages or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Anonymous recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

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
            streaming_data = {}
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'streaming_data': streaming_data,
                'available_languages': json.loads(content.available_languages or '[]')
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
            streaming_data = {}
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'streaming_data': streaming_data,
                'available_languages': json.loads(content.available_languages or '[]')
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
            # Handle release date
            release_date = None
            if data.get('release_date'):
                try:
                    release_date = datetime.strptime(data['release_date'][:10], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            # Get streaming availability
            streaming_data = StreamingAvailabilityService.get_streaming_availability(
                data.get('title'),
                data.get('imdb_id'),
                data.get('id') if data.get('source') == 'tmdb' else None
            )
            
            # Create content object
            content = Content(
                tmdb_id=data.get('id') if data.get('source') == 'tmdb' else None,
                mal_id=data.get('id') if data.get('source') == 'anime' else None,
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
                streaming_links=json.dumps(streaming_data),
                available_languages=json.dumps(list(streaming_data.get('languages', {}).keys()))
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
        
        # Language preferences
        language_counts = defaultdict(int)
        for interaction in all_interactions:
            content = Content.query.get(interaction.content_id)
            if content and content.available_languages:
                languages = json.loads(content.available_languages)
                for lang in languages:
                    language_counts[lang] += 1
        
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
                {'language': lang, 'count': count}
                for lang, count in popular_languages
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
                streaming_data = {}
                if content.streaming_links:
                    try:
                        streaming_data = json.loads(content.streaming_links)
                    except:
                        pass
                
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'streaming_data': streaming_data,
                    'available_languages': json.loads(content.available_languages or '[]'),
                    'admin_description': rec.description,
                    'admin_name': admin.username if admin else 'Admin',
                    'recommended_at': rec.created_at.isoformat()
                })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Public admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get admin recommendations'}), 500

# Streaming availability endpoint
@app.route('/api/streaming/<int:content_id>', methods=['GET'])
def get_streaming_availability(content_id):
    try:
        content = Content.query.get_or_404(content_id)
        
        # Update streaming availability if data is old
        if not content.updated_at or (datetime.utcnow() - content.updated_at).days > 7:
            ContentService._update_streaming_availability(content)
        
        streaming_data = {}
        if content.streaming_links:
            try:
                streaming_data = json.loads(content.streaming_links)
            except:
                pass
        
        return jsonify({
            'content_id': content_id,
            'title': content.title,
            'streaming_data': streaming_data,
            'available_languages': json.loads(content.available_languages or '[]'),
            'last_updated': content.updated_at.isoformat() if content.updated_at else None
        }), 200
        
    except Exception as e:
        logger.error(f"Streaming availability error: {e}")
        return jsonify({'error': 'Failed to get streaming availability'}), 500

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
            'anime_support': True,
            'regional_content': True,
            'real_time_recommendations': True
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