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
    mal_id = db.Column(db.Integer)  # For anime
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

# OTT Platform Information
OTT_PLATFORMS = {
    'netflix': {'name': 'Netflix', 'is_free': False, 'url': 'https://netflix.com'},
    'amazon_prime': {'name': 'Amazon Prime Video', 'is_free': False, 'url': 'https://primevideo.com'},
    'disney_plus': {'name': 'Disney+ Hotstar', 'is_free': False, 'url': 'https://hotstar.com'},
    'youtube': {'name': 'YouTube', 'is_free': True, 'url': 'https://youtube.com'},
    'jiocinema': {'name': 'JioCinema', 'is_free': True, 'url': 'https://jiocinema.com'},
    'mx_player': {'name': 'MX Player', 'is_free': True, 'url': 'https://mxplayer.com'},
    'zee5': {'name': 'ZEE5', 'is_free': False, 'url': 'https://zee5.com'},
    'sonyliv': {'name': 'SonyLIV', 'is_free': False, 'url': 'https://sonyliv.com'},
    'voot': {'name': 'Voot', 'is_free': True, 'url': 'https://voot.com'},
    'alt_balaji': {'name': 'ALTBalaji', 'is_free': False, 'url': 'https://altbalaji.com'},
    'airtel_xstream': {'name': 'Airtel Xstream', 'is_free': True, 'url': 'https://airtelxstream.in'},
    'crunchyroll': {'name': 'Crunchyroll', 'is_free': True, 'url': 'https://crunchyroll.com'},
    'sun_nxt': {'name': 'Sun NXT', 'is_free': False, 'url': 'https://sunnxt.com'},
    'aha': {'name': 'Aha', 'is_free': False, 'url': 'https://aha.video'}
}

# Regional Language Mapping
REGIONAL_LANGUAGES = {
    'hindi': ['hi', 'hindi', 'bollywood'],
    'telugu': ['te', 'telugu', 'tollywood'],
    'tamil': ['ta', 'tamil', 'kollywood'],
    'kannada': ['kn', 'kannada', 'sandalwood'],
    'malayalam': ['ml', 'malayalam', 'mollywood'],
    'english': ['en', 'english', 'hollywood'],
    'bengali': ['bn', 'bengali', 'tollygunge'],
    'punjabi': ['pa', 'punjabi'],
    'gujarati': ['gu', 'gujarati'],
    'marathi': ['mr', 'marathi']
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
class StreamingService:
    RAPIDAPI_KEY = "c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37"
    RAPIDAPI_HOST = "streaming-availability.p.rapidapi.com"
    
    @staticmethod
    def get_streaming_availability(tmdb_id, content_type):
        """Get streaming availability from multiple sources"""
        try:
            # Try Streaming Availability API first
            streaming_data = StreamingService._get_from_streaming_availability_api(tmdb_id, content_type)
            
            if not streaming_data:
                # Fallback to WatchMode API
                streaming_data = StreamingService._get_from_watchmode_api(tmdb_id, content_type)
            
            if not streaming_data:
                # Fallback to default platforms
                streaming_data = StreamingService._get_default_platforms()
            
            return streaming_data
        except Exception as e:
            logger.error(f"Streaming availability error: {e}")
            return StreamingService._get_default_platforms()
    
    @staticmethod
    def _get_from_streaming_availability_api(tmdb_id, content_type):
        """Get data from Streaming Availability API"""
        try:
            url = f"https://{StreamingService.RAPIDAPI_HOST}/shows/{content_type}/{tmdb_id}"
            
            headers = {
                'x-rapidapi-key': StreamingService.RAPIDAPI_KEY,
                'x-rapidapi-host': StreamingService.RAPIDAPI_HOST
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return StreamingService._parse_streaming_availability_response(data)
        except Exception as e:
            logger.error(f"Streaming Availability API error: {e}")
        
        return []
    
    @staticmethod
    def _get_from_watchmode_api(tmdb_id, content_type):
        """Get data from WatchMode API"""
        try:
            if not WATCHMODE_API_KEY:
                return []
            
            # First, get the title details
            url = f"https://api.watchmode.com/v1/title/{tmdb_id}/details/"
            params = {
                'apikey': WATCHMODE_API_KEY,
                'append_to_response': 'sources'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return StreamingService._parse_watchmode_response(data)
        except Exception as e:
            logger.error(f"WatchMode API error: {e}")
        
        return []
    
    @staticmethod
    def _parse_streaming_availability_response(data):
        """Parse Streaming Availability API response"""
        platforms = []
        
        try:
            streaming_options = data.get('streamingOptions', {})
            
            for country_code, options in streaming_options.items():
                if country_code == 'in':  # India
                    for option in options:
                        service = option.get('service', {})
                        service_id = service.get('id', '').lower()
                        
                        # Map service IDs to our platform names
                        platform_mapping = {
                            'netflix': 'netflix',
                            'prime': 'amazon_prime',
                            'hotstar': 'disney_plus',
                            'zee5': 'zee5',
                            'sonyliv': 'sonyliv',
                            'jiocinema': 'jiocinema',
                            'mxplayer': 'mx_player',
                            'youtube': 'youtube',
                            'crunchyroll': 'crunchyroll',
                            'aha': 'aha',
                            'sunnxt': 'sun_nxt'
                        }
                        
                        platform_name = platform_mapping.get(service_id)
                        if platform_name:
                            platforms.append({
                                'platform': platform_name,
                                'platform_name': OTT_PLATFORMS.get(platform_name, {}).get('name', platform_name),
                                'url': option.get('link', ''),
                                'type': option.get('type', 'subscription'),
                                'quality': option.get('quality', 'hd'),
                                'languages': option.get('audios', []),
                                'is_free': OTT_PLATFORMS.get(platform_name, {}).get('is_free', False)
                            })
        except Exception as e:
            logger.error(f"Error parsing streaming availability: {e}")
        
        return platforms
    
    @staticmethod
    def _parse_watchmode_response(data):
        """Parse WatchMode API response"""
        platforms = []
        
        try:
            sources = data.get('sources', [])
            
            for source in sources:
                source_id = source.get('source_id')
                web_url = source.get('web_url', '')
                
                # Map WatchMode source IDs to our platform names
                source_mapping = {
                    203: 'netflix',
                    26: 'amazon_prime', 
                    372: 'disney_plus',
                    220: 'zee5',
                    237: 'sonyliv',
                    551: 'jiocinema',
                    473: 'mx_player',
                    142: 'youtube',
                    444: 'crunchyroll',
                    613: 'aha',
                    482: 'sun_nxt'
                }
                
                platform_name = source_mapping.get(source_id)
                if platform_name and web_url:
                    platforms.append({
                        'platform': platform_name,
                        'platform_name': OTT_PLATFORMS.get(platform_name, {}).get('name', platform_name),
                        'url': web_url,
                        'type': 'subscription',
                        'region': source.get('region', 'IN'),
                        'is_free': OTT_PLATFORMS.get(platform_name, {}).get('is_free', False)
                    })
        except Exception as e:
            logger.error(f"Error parsing WatchMode response: {e}")
        
        return platforms
    
    @staticmethod
    def _get_default_platforms():
        """Get default platform list when APIs fail"""
        default_platforms = []
        
        # Add popular Indian OTT platforms
        popular_platforms = ['netflix', 'amazon_prime', 'disney_plus', 'zee5', 'sonyliv', 'mx_player', 'youtube']
        
        for platform in popular_platforms:
            if platform in OTT_PLATFORMS:
                default_platforms.append({
                    'platform': platform,
                    'platform_name': OTT_PLATFORMS[platform]['name'],
                    'url': OTT_PLATFORMS[platform]['url'],
                    'type': 'subscription',
                    'is_free': OTT_PLATFORMS[platform]['is_free']
                })
        
        return default_platforms
    
    @staticmethod
    def get_language_specific_links(content_id, language):
        """Get language-specific streaming links"""
        try:
            content = Content.query.get(content_id)
            if not content:
                return []
            
            # Get base streaming data
            streaming_data = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type)
            
            # Filter and modify URLs for specific language
            language_links = []
            
            for platform in streaming_data:
                platform_url = platform.get('url', '')
                if platform_url:
                    # Add language parameter to URL (this would be platform-specific in production)
                    if '?' in platform_url:
                        lang_url = f"{platform_url}&audio={language}"
                    else:
                        lang_url = f"{platform_url}?audio={language}"
                    
                    language_links.append({
                        'platform': platform['platform'],
                        'platform_name': platform.get('platform_name', platform['platform']),
                        'url': lang_url,
                        'language': language,
                        'language_display': StreamingService._get_language_display_name(language),
                        'is_free': platform.get('is_free', False)
                    })
            
            return language_links
        except Exception as e:
            logger.error(f"Language-specific links error: {e}")
            return []
    
    @staticmethod
    def _get_language_display_name(lang_code):
        """Convert language code to display name"""
        lang_map = {
            'hi': 'Hindi',
            'te': 'Telugu', 
            'ta': 'Tamil',
            'ml': 'Malayalam',
            'kn': 'Kannada',
            'en': 'English',
            'bn': 'Bengali',
            'pa': 'Punjabi',
            'gu': 'Gujarati',
            'mr': 'Marathi'
        }
        return lang_map.get(lang_code.lower(), lang_code.title())

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
    def discover_content(content_type='movie', **kwargs):
        """Discover content with filters"""
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
                # Map genre IDs to names
                genres = ContentService.map_genre_ids(tmdb_data['genre_ids'])
            
            # Extract languages
            languages = []
            if 'spoken_languages' in tmdb_data:
                languages = [lang['iso_639_1'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
            
            # Get OTT platforms
            ott_platforms = StreamingService.get_streaming_availability(tmdb_data['id'], content_type)
            
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
    def save_anime_from_jikan(anime_data):
        """Save anime content from Jikan API"""
        try:
            # Check if anime already exists
            existing = Content.query.filter_by(mal_id=anime_data['mal_id']).first()
            if existing:
                return existing
            
            # Extract genres
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            # Create content object
            content = Content(
                mal_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                languages=json.dumps(['japanese']),
                release_date=datetime.strptime(anime_data.get('aired', {}).get('from', '1900-01-01T00:00:00+00:00')[:10], '%Y-%m-%d').date() if anime_data.get('aired', {}).get('from') else None,
                runtime=anime_data.get('duration'),
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                ott_platforms=json.dumps([])  # You would check anime streaming platforms
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime: {e}")
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

# Regional Content Service
class RegionalContentService:
    @staticmethod
    def get_accurate_regional_content(language, category='popular', genre=None, limit=20):
        """Get accurate regional content with multiple data sources"""
        try:
            content_list = []
            
            # Enhanced search queries for each language
            search_strategies = {
                'telugu': {
                    'all-time-hits': ['Baahubali', 'RRR', 'Pushpa', 'KGF Chapter 2', 'Arjun Reddy', 'Magadheera', 'Eega', 'Rangasthalam', 'Ala Vaikunthapurramuloo', 'Saaho'],
                    'trending': ['tollywood 2024', 'telugu latest movies', 'telugu blockbuster 2024'],
                    'new': ['telugu new releases 2024', 'latest telugu movies', 'telugu cinema 2024'],
                    'popular': ['telugu popular movies', 'tollywood hits', 'telugu blockbusters']
                },
                'hindi': {
                    'all-time-hits': ['Dangal', 'Sholay', '3 Idiots', 'Dilwale Dulhania Le Jayenge', 'Lagaan', 'Mughal-E-Azam', 'Zindagi Na Milegi Dobara', 'Taare Zameen Par', 'Queen', 'Andhadhun'],
                    'trending': ['bollywood 2024', 'hindi latest movies', 'bollywood blockbuster 2024'],
                    'new': ['hindi new releases 2024', 'latest bollywood movies', 'bollywood 2024'],
                    'popular': ['hindi popular movies', 'bollywood hits', 'hindi blockbusters']
                },
                'tamil': {
                    'all-time-hits': ['Vikram', 'KGF', 'Baahubali', 'Kabali', 'Enthiran', 'Sivaji', 'Anniyan', 'Ghilli', 'Thuppakki', 'Kaththi'],
                    'trending': ['kollywood 2024', 'tamil latest movies', 'tamil blockbuster 2024'],
                    'new': ['tamil new releases 2024', 'latest tamil movies', 'kollywood 2024'],
                    'popular': ['tamil popular movies', 'kollywood hits', 'tamil blockbusters']
                },
                'malayalam': {
                    'all-time-hits': ['Lucifer', 'Premam', 'Bangalore Days', 'Drishyam', 'Charlie', 'Kumbakonam Gopals', 'Mohanlal', 'Mammootty'],
                    'trending': ['mollywood 2024', 'malayalam latest movies', 'malayalam blockbuster 2024'],
                    'new': ['malayalam new releases 2024', 'latest malayalam movies', 'mollywood 2024'],
                    'popular': ['malayalam popular movies', 'mollywood hits', 'malayalam blockbusters']
                },
                'kannada': {
                    'all-time-hits': ['KGF', 'Kantara', 'Kotigobba', 'Mungaru Male', 'Lucia', 'U Turn', 'Kirik Party', 'RangiTaranga'],
                    'trending': ['sandalwood 2024', 'kannada latest movies', 'kannada blockbuster 2024'],
                    'new': ['kannada new releases 2024', 'latest kannada movies', 'sandalwood 2024'],
                    'popular': ['kannada popular movies', 'sandalwood hits', 'kannada blockbusters']
                },
                'english': {
                    'all-time-hits': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Avengers Endgame', 'Titanic', 'The Lord of the Rings', 'Pulp Fiction', 'Forrest Gump'],
                    'trending': ['hollywood 2024', 'english latest movies', 'hollywood blockbuster 2024'],
                    'new': ['english new releases 2024', 'latest hollywood movies', 'hollywood 2024'],
                    'popular': ['english popular movies', 'hollywood hits', 'english blockbusters']
                }
            }
            
            # Get search queries for the category
            queries = search_strategies.get(language, {}).get(category, [])
            
            # If it's all-time-hits, search for specific movies
            if category == 'all-time-hits':
                for movie_title in queries:
                    search_results = TMDBService.search_content(movie_title, 'movie')
                    if search_results and search_results.get('results'):
                        item = search_results['results'][0]  # Take the first result
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content and content not in content_list:
                            content_list.append(content)
                    
                    if len(content_list) >= limit:
                        break
            else:
                # For other categories, use search queries
                for query in queries:
                    search_results = TMDBService.search_content(query, 'movie')
                    if search_results:
                        for item in search_results.get('results', [])[:5]:
                            content = ContentService.save_content_from_tmdb(item, 'movie')
                            if content and content not in content_list:
                                content_list.append(content)
                    
                    if len(content_list) >= limit:
                        break
            
            # Use discover API for more accurate results
            if len(content_list) < limit:
                language_code = {
                    'telugu': 'te',
                    'hindi': 'hi',
                    'tamil': 'ta',
                    'malayalam': 'ml',
                    'kannada': 'kn',
                    'english': 'en'
                }.get(language, 'en')
                
                discover_params = {
                    'with_original_language': language_code,
                    'sort_by': 'popularity.desc' if category == 'popular' else 'primary_release_date.desc',
                    'page': 1
                }
                
                if genre:
                    # Map genre name to ID (you'd need a complete mapping)
                    genre_map = {
                        'Action': '28', 'Comedy': '35', 'Drama': '18', 'Romance': '10749',
                        'Thriller': '53', 'Horror': '27', 'Adventure': '12', 'Crime': '80',
                        'Fantasy': '14', 'Sci-Fi': '878', 'Animation': '16'
                    }
                    genre_id = genre_map.get(genre)
                    if genre_id:
                        discover_params['with_genres'] = genre_id
                
                discover_results = TMDBService.discover_content('movie', **discover_params)
                if discover_results:
                    for item in discover_results.get('results', []):
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content and content not in content_list:
                            content_list.append(content)
                        
                        if len(content_list) >= limit:
                            break
            
            return content_list[:limit]
            
        except Exception as e:
            logger.error(f"Regional content error: {e}")
            return []
    
    @staticmethod
    def get_genre_wise_content(language, genre, limit=20):
        """Get genre-wise content for a specific language"""
        try:
            content_list = []
            
            # First try with discover API
            language_code = {
                'telugu': 'te', 'hindi': 'hi', 'tamil': 'ta', 
                'malayalam': 'ml', 'kannada': 'kn', 'english': 'en'
            }.get(language, 'en')
            
            # Map genre name to ID
            genre_map = {
                'Action': '28', 'Adventure': '12', 'Animation': '16', 'Comedy': '35',
                'Crime': '80', 'Documentary': '99', 'Drama': '18', 'Family': '10751',
                'Fantasy': '14', 'History': '36', 'Horror': '27', 'Music': '10402',
                'Mystery': '9648', 'Romance': '10749', 'Sci-Fi': '878', 'Thriller': '53',
                'War': '10752', 'Western': '37'
            }
            
            genre_id = genre_map.get(genre)
            if genre_id:
                discover_params = {
                    'with_genres': genre_id,
                    'with_original_language': language_code,
                    'sort_by': 'popularity.desc',
                    'page': 1
                }
                
                discover_results = TMDBService.discover_content('movie', **discover_params)
                if discover_results:
                    for item in discover_results.get('results', []):
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content:
                            content_list.append(content)
                        
                        if len(content_list) >= limit:
                            break
            
            # If not enough results, search with query
            if len(content_list) < limit:
                search_query = f"{language} {genre} movies"
                search_results = TMDBService.search_content(search_query, 'movie')
                if search_results:
                    for item in search_results.get('results', []):
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content and content not in content_list:
                            content_list.append(content)
                        
                        if len(content_list) >= limit:
                            break
            
            return content_list[:limit]
            
        except Exception as e:
            logger.error(f"Genre-wise content error: {e}")
            return []

# Enhanced Telegram Service
class TelegramService:
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            # Get streaming availability
            streaming_data = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type)
            
            # Format genre list
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            # Format languages
            languages_list = []
            if content.languages:
                try:
                    languages_list = json.loads(content.languages)
                except:
                    languages_list = []
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create main message
            message = f"""🎬 **ADMIN'S CHOICE** by @{admin_name}

🎭 **{content.title}**
{'📺' if content.content_type == 'tv' else '🎬' if content.content_type == 'movie' else '🌸'} Type: {content.content_type.upper()}
⭐ Rating: {content.rating or 'N/A'}/10 {'⭐' * int(content.rating/2) if content.rating else ''}
📅 Release: {content.release_date.strftime('%Y') if content.release_date else 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🌐 Languages: {', '.join([TelegramService._get_language_display_name(lang) for lang in languages_list[:3]]) if languages_list else 'N/A'}

💭 **Admin's Recommendation:**
_{description}_

📖 **Synopsis:**
{(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

━━━━━━━━━━━━━━━━━━━━━━━━━━
🎬 **WHERE TO WATCH** 🎬
━━━━━━━━━━━━━━━━━━━━━━━━━━"""

            # Add streaming links by language
            if streaming_data:
                message += TelegramService._format_streaming_links(streaming_data, languages_list, content.title)
            else:
                message += "\n\n🔍 **Search on Popular Platforms:**"
                message += f"\n🔴 Netflix: netflix.com/search?q={content.title.replace(' ', '%20')}"
                message += f"\n🟦 Prime Video: primevideo.com/search?phrase={content.title.replace(' ', '%20')}"
                message += f"\n🟠 Disney+ Hotstar: hotstar.com/search?q={content.title.replace(' ', '%20')}"
                message += f"\n🟣 Zee5: zee5.com/search?q={content.title.replace(' ', '%20')}"
                message += f"\n🔵 SonyLIV: sonyliv.com/search?q={content.title.replace(' ', '%20')}"
            
            message += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━
🏷️ Tags: #AdminChoice #MovieRecommendation #CineScope #{content.content_type.title()}
{' '.join([f'#{genre.replace(" ", "")}' for genre in genres_list[:3]])}
━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 **Tip:** Use the language-specific links above for the best viewing experience!
📱 Save this message for easy access to watch links!"""
            
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
    def _format_streaming_links(streaming_data, languages, title):
        """Format streaming links with language-specific buttons"""
        links_text = ""
        
        # Group platforms by type
        free_platforms = []
        paid_platforms = []
        
        for platform_data in streaming_data:
            if platform_data.get('is_free', False):
                free_platforms.append(platform_data)
            else:
                paid_platforms.append(platform_data)
        
        # Add free platforms
        if free_platforms:
            links_text += "\n\n🆓 **FREE PLATFORMS:**"
            for platform in free_platforms:
                links_text += TelegramService._format_platform_links(platform, languages, title)
        
        # Add paid platforms
        if paid_platforms:
            links_text += "\n\n💰 **PREMIUM PLATFORMS:**"
            for platform in paid_platforms:
                links_text += TelegramService._format_platform_links(platform, languages, title)
        
        return links_text
    
    @staticmethod
    def _format_platform_links(platform_data, languages, title):
        """Format individual platform links with language options"""
        platform_name = platform_data.get('platform_name', platform_data.get('platform', '').title())
        base_url = platform_data.get('url', '')
        
        if not base_url:
            # Create search URL if direct link not available
            platform_key = platform_data.get('platform', '').lower()
            platform_info = OTT_PLATFORMS.get(platform_key, {})
            base_url = platform_info.get('url', '') + f"/search?q={title.replace(' ', '%20')}"
        
        platform_text = f"\n\n🎬 **{platform_name}**"
        
        # If multiple languages available, show language-specific links
        if len(languages) > 1:
            for lang in languages[:3]:  # Limit to 3 languages to avoid message being too long
                lang_display = TelegramService._get_language_display_name(lang)
                emoji = TelegramService._get_language_emoji(lang)
                
                # Create language-specific URL
                if '?' in base_url:
                    lang_url = f"{base_url}&audio={lang}"
                else:
                    lang_url = f"{base_url}?audio={lang}"
                platform_text += f"\n{emoji} [Watch in {lang_display}]({lang_url})"
        else:
            # Single language or default
            platform_text += f"\n🔗 [Watch Now]({base_url})"
        
        return platform_text
    
    @staticmethod
    def _get_language_display_name(lang_code):
        """Convert language code to display name"""
        lang_map = {
            'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil', 'ml': 'Malayalam',
            'kn': 'Kannada', 'en': 'English', 'bn': 'Bengali', 'pa': 'Punjabi',
            'gu': 'Gujarati', 'mr': 'Marathi'
        }
        return lang_map.get(lang_code.lower(), lang_code.title())
    
    @staticmethod
    def _get_language_emoji(lang_code):
        """Get emoji for language"""
        emoji_map = {
            'hi': '🇮🇳', 'te': '🎭', 'ta': '🎪', 'ml': '🌴', 'kn': '🏛️',
            'en': '🇺🇸', 'bn': '🐅', 'pa': '🎵', 'gu': '🦁', 'mr': '⚡'
        }
        return emoji_map.get(lang_code.lower(), '🎬')
    
    @staticmethod
    def send_trending_update(trending_content):
        """Send trending content updates"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = "🔥 **TRENDING NOW** 🔥\n\n"
            
            for idx, content in enumerate(trending_content[:5], 1):
                genres = json.loads(content.genres or '[]')
                message += f"{idx}. **{content.title}**\n"
                message += f"   ⭐ {content.rating or 'N/A'}/10 | {', '.join(genres[:2])}\n"
                message += f"   📱 Get details: /movie_{content.id}\n\n"
            
            message += "#Trending #MovieRecommendations #CineScope"
            
            bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            return True
        except Exception as e:
            logger.error(f"Trending update error: {e}")
            return False

# Recommendation Engine
class RecommendationEngine:
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all'):
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
                    recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_popular_by_genre(genre, limit=20, region=None):
        try:
            # Use discover API for better results
            genre_map = {
                'Action': '28', 'Adventure': '12', 'Animation': '16', 'Comedy': '35',
                'Crime': '80', 'Documentary': '99', 'Drama': '18', 'Family': '10751',
                'Fantasy': '14', 'History': '36', 'Horror': '27', 'Music': '10402',
                'Mystery': '9648', 'Romance': '10749', 'Sci-Fi': '878', 'Thriller': '53',
                'War': '10752', 'Western': '37'
            }
            
            genre_id = genre_map.get(genre)
            recommendations = []
            
            if genre_id:
                # Discover movies by genre
                discover_params = {
                    'with_genres': genre_id,
                    'sort_by': 'popularity.desc',
                    'page': 1
                }
                if region:
                    discover_params['region'] = region
                
                discover_results = TMDBService.discover_content('movie', **discover_params)
                if discover_results:
                    for item in discover_results.get('results', [])[:limit//2]:
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content:
                            recommendations.append(content)
                
                # Discover TV shows by genre
                discover_results = TMDBService.discover_content('tv', **discover_params)
                if discover_results:
                    for item in discover_results.get('results', [])[:limit//2]:
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
            return RegionalContentService.get_accurate_regional_content(language, 'popular', None, limit)
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
                # Save anime to database
                content = ContentService.save_anime_from_jikan(anime)
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
                    
                    # Get streaming platforms
                    streaming_platforms = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type)
                    
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
                        'ott_platforms': streaming_platforms
                    })
        
        # Add anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                # Save anime to database
                content = ContentService.save_anime_from_jikan(anime)
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
                        'ott_platforms': []
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
        
        # Get additional details from TMDB or Jikan
        additional_details = None
        if content.content_type == 'anime' and content.mal_id:
            additional_details = JikanService.get_anime_details(content.mal_id)
        elif content.tmdb_id:
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        
        # Get streaming availability with language support
        streaming_platforms = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type) if content.tmdb_id else []
        
        # Get language-specific streaming links
        languages = json.loads(content.languages or '[]')
        language_streaming_links = {}
        
        for lang in languages:
            lang_links = StreamingService.get_language_specific_links(content.id, lang)
            if lang_links:
                language_streaming_links[lang] = lang_links
        
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
                # For anime, get recommendations from the API response
                recommendations = additional_details.get('data', {}).get('recommendations', [])
                for rec in recommendations[:5]:
                    similar_content.append({
                        'title': rec.get('entry', {}).get('title'),
                        'poster_path': rec.get('entry', {}).get('images', {}).get('jpg', {}).get('image_url'),
                        'rating': None
                    })
            else:
                # For movies/TV, get similar content from TMDB
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
        
        db.session.commit()
        
        return jsonify({
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'mal_id': content.mal_id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': languages,
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'runtime': content.runtime,
            'rating': content.rating,
            'vote_count': content.vote_count,
            'overview': content.overview,
            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path else None,
            'ott_platforms': streaming_platforms,
            'language_streaming_links': language_streaming_links,
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details and content.content_type != 'anime' else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details and content.content_type != 'anime' else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = []
        for content in recommendations:
            streaming_platforms = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type) if content.tmdb_id else []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': streaming_platforms
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
            streaming_platforms = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type) if content.tmdb_id else []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': streaming_platforms
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
            streaming_platforms = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type) if content.tmdb_id else []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': streaming_platforms
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Regional recommendations error: {e}")
        return jsonify({'error': 'Failed to get regional recommendations'}), 500

@app.route('/api/recommendations/regional/<language>/<category>', methods=['GET'])
def get_enhanced_regional(language, category):
    try:
        limit = int(request.args.get('limit', 20))
        genre = request.args.get('genre')
        
        if category == 'all-time-hits':
            recommendations = RegionalContentService.get_accurate_regional_content(language, 'all-time-hits', genre, limit)
        elif category == 'trending':
            recommendations = RegionalContentService.get_accurate_regional_content(language, 'trending', genre, limit)
        elif category == 'new-releases':
            recommendations = RegionalContentService.get_accurate_regional_content(language, 'new', genre, limit)
        elif category == 'genre' and genre:
            recommendations = RegionalContentService.get_genre_wise_content(language, genre, limit)
        else:
            recommendations = RegionalContentService.get_accurate_regional_content(language, 'popular', genre, limit)
        
        result = []
        for content in recommendations:
            streaming_platforms = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type) if content.tmdb_id else []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': streaming_platforms,
                'languages': json.loads(content.languages or '[]')
            })
        
        return jsonify({
            'recommendations': result,
            'language': language,
            'category': category,
            'total': len(result)
        }), 200
        
    except Exception as e:
        logger.error(f"Enhanced regional recommendations error: {e}")
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
                'ott_platforms': json.loads(content.ott_platforms or '[]')
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
            streaming_platforms = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type) if content.tmdb_id else []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': streaming_platforms
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Anonymous recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@app.route('/api/content/<int:content_id>/streaming/<language>', methods=['GET'])
def get_language_streaming_links(content_id, language):
    """Get streaming links for specific language"""
    try:
        content = Content.query.get_or_404(content_id)
        
        # Get language-specific streaming links
        streaming_links = StreamingService.get_language_specific_links(content_id, language)
        
        return jsonify({
            'content_id': content_id,
            'language': language,
            'streaming_links': streaming_links,
            'content_title': content.title
        }), 200
        
    except Exception as e:
        logger.error(f"Language streaming links error: {e}")
        return jsonify({'error': 'Failed to get streaming links'}), 500

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
                        streaming_platforms = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type) if content.tmdb_id else []
                        
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'ott_platforms': streaming_platforms,
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
            streaming_platforms = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type) if content.tmdb_id else []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': streaming_platforms
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
            streaming_platforms = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type) if content.tmdb_id else []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': streaming_platforms
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
                    if 'T' in data['release_date']:
                        release_date = datetime.strptime(data['release_date'][:10], '%Y-%m-%d').date()
                    else:
                        release_date = datetime.strptime(data['release_date'], '%Y-%m-%d').date()
                except:
                    release_date = None
            
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
                streaming_platforms = StreamingService.get_streaming_availability(content.tmdb_id, content.content_type) if content.tmdb_id else []
                
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'ott_platforms': streaming_platforms,
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
        'version': '1.0.0'
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