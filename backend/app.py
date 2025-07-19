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

# Streaming Availability API Keys
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

# Enhanced Regional Language Mapping with Telugu Priority
REGIONAL_LANGUAGES = {
    'telugu': {
        'name': 'Telugu',
        'codes': ['te', 'telugu'],
        'search_terms': ['tollywood', 'telugu movie', 'telugu film', 'telugu cinema', 'andhra pradesh', 'telangana', 'nandamuri', 'allu arjun', 'mahesh babu', 'prabhas', 'ram charan', 'jr ntr', 'chiranjeevi'],
        'priority': 1,
        'emoji': '🎬',
        'flag': '🇮🇳',
        'industry': 'Tollywood'
    },
    'hindi': {
        'name': 'Hindi',
        'codes': ['hi', 'hindi'],
        'search_terms': ['bollywood', 'hindi movie', 'hindi film', 'hindi cinema', 'mumbai', 'shah rukh khan', 'salman khan', 'aamir khan', 'akshay kumar', 'hrithik roshan'],
        'priority': 2,
        'emoji': '🎭',
        'flag': '🇮🇳',
        'industry': 'Bollywood'
    },
    'tamil': {
        'name': 'Tamil',
        'codes': ['ta', 'tamil'],
        'search_terms': ['kollywood', 'tamil movie', 'tamil film', 'tamil cinema', 'tamil nadu', 'rajinikanth', 'kamal haasan', 'vijay', 'ajith', 'suriya', 'dhanush'],
        'priority': 3,
        'emoji': '🎪',
        'flag': '🇮🇳',
        'industry': 'Kollywood'
    },
    'malayalam': {
        'name': 'Malayalam',
        'codes': ['ml', 'malayalam'],
        'search_terms': ['mollywood', 'malayalam movie', 'malayalam film', 'malayalam cinema', 'kerala', 'mohanlal', 'mammootty', 'prithviraj', 'fahadh faasil'],
        'priority': 4,
        'emoji': '🎨',
        'flag': '🇮🇳',
        'industry': 'Mollywood'
    },
    'kannada': {
        'name': 'Kannada',
        'codes': ['kn', 'kannada'],
        'search_terms': ['sandalwood', 'kannada movie', 'kannada film', 'kannada cinema', 'karnataka', 'yash', 'puneeth rajkumar', 'sudeep', 'darshan'],
        'priority': 5,
        'emoji': '🎵',
        'flag': '🇮🇳',
        'industry': 'Sandalwood'
    },
    'english': {
        'name': 'English',
        'codes': ['en', 'english'],
        'search_terms': ['hollywood', 'english movie', 'english film', 'american cinema', 'british cinema'],
        'priority': 6,
        'emoji': '🎞️',
        'flag': '🇺🇸',
        'industry': 'Hollywood'
    }
}

# Enhanced Genre Categories
GENRE_CATEGORIES = {
    'Action': {'emoji': '💥', 'description': 'High-octane thrills and excitement'},
    'Adventure': {'emoji': '🗺️', 'description': 'Epic journeys and discoveries'},
    'Animation': {'emoji': '🎨', 'description': 'Animated masterpieces for all ages'},
    'Biography': {'emoji': '📖', 'description': 'Real-life stories of remarkable people'},
    'Comedy': {'emoji': '😂', 'description': 'Laugh-out-loud entertainment'},
    'Crime': {'emoji': '🔍', 'description': 'Gripping tales of law and disorder'},
    'Documentary': {'emoji': '🎞️', 'description': 'Real stories, real impact'},
    'Drama': {'emoji': '🎭', 'description': 'Emotional journeys and human stories'},
    'Fantasy': {'emoji': '🧙‍♂️', 'description': 'Magical worlds and mythical adventures'},
    'Horror': {'emoji': '👻', 'description': 'Spine-chilling scares and thrills'},
    'Musical': {'emoji': '🎵', 'description': 'Music, dance, and melodious storytelling'},
    'Mystery': {'emoji': '🕵️', 'description': 'Puzzles, secrets, and suspense'},
    'Romance': {'emoji': '💕', 'description': 'Love stories that touch the heart'},
    'Sci-Fi': {'emoji': '🚀', 'description': 'Future worlds and scientific wonders'},
    'Thriller': {'emoji': '⚡', 'description': 'Edge-of-your-seat suspense'},
    'Western': {'emoji': '🤠', 'description': 'Wild west adventures and cowboy tales'}
}

# Enhanced OTT Platform Information with regional focus
ENHANCED_OTT_PLATFORMS = {
    'netflix': {
        'name': 'Netflix',
        'is_free': False,
        'category': 'paid',
        'url': 'https://netflix.com',
        'search_url': 'https://www.netflix.com/search?q=',
        'description': 'Global streaming with regional content',
        'emoji': '🔴',
        'regional_strength': ['hindi', 'english', 'tamil']
    },
    'amazon_prime': {
        'name': 'Amazon Prime Video',
        'is_free': False,
        'category': 'paid',
        'url': 'https://primevideo.com',
        'search_url': 'https://www.primevideo.com/search/ref=atv_sr_sug_0?phrase=',
        'description': 'Prime membership with regional originals',
        'emoji': '📦',
        'regional_strength': ['hindi', 'tamil', 'telugu', 'malayalam']
    },
    'disney_plus': {
        'name': 'Disney+ Hotstar',
        'is_free': False,
        'category': 'paid',
        'url': 'https://hotstar.com',
        'search_url': 'https://www.hotstar.com/in/search?q=',
        'description': 'Disney content and Indian entertainment',
        'emoji': '⭐',
        'regional_strength': ['hindi', 'tamil', 'telugu', 'english']
    },
    'youtube': {
        'name': 'YouTube',
        'is_free': True,
        'category': 'free',
        'url': 'https://youtube.com',
        'search_url': 'https://www.youtube.com/results?search_query=',
        'description': 'Free movies and regional content',
        'emoji': '📺',
        'regional_strength': ['telugu', 'hindi', 'tamil', 'malayalam', 'kannada']
    },
    'jiocinema': {
        'name': 'JioCinema',
        'is_free': True,
        'category': 'free',
        'url': 'https://jiocinema.com',
        'search_url': 'https://www.jiocinema.com/search?q=',
        'description': 'Free regional and Bollywood content',
        'emoji': '🎬',
        'regional_strength': ['hindi', 'telugu', 'tamil', 'malayalam']
    },
    'mx_player': {
        'name': 'MX Player',
        'is_free': True,
        'category': 'free',
        'url': 'https://mxplayer.com',
        'search_url': 'https://www.mxplayer.in/search?q=',
        'description': 'Free regional movies and originals',
        'emoji': '🎭',
        'regional_strength': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam']
    },
    'zee5': {
        'name': 'ZEE5',
        'is_free': True,
        'category': 'freemium',
        'url': 'https://zee5.com',
        'search_url': 'https://www.zee5.com/search?q=',
        'description': 'Regional content hub',
        'emoji': '💎',
        'regional_strength': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam']
    },
    'sonyliv': {
        'name': 'SonyLIV',
        'is_free': True,
        'category': 'freemium',
        'url': 'https://sonyliv.com',
        'search_url': 'https://www.sonyliv.com/search?q=',
        'description': 'Sports and regional entertainment',
        'emoji': '📱',
        'regional_strength': ['hindi', 'telugu', 'tamil', 'malayalam']
    },
    'crunchyroll': {
        'name': 'Crunchyroll',
        'is_free': True,
        'category': 'freemium',
        'url': 'https://crunchyroll.com',
        'search_url': 'https://www.crunchyroll.com/search?q=',
        'description': 'Anime content with free and premium tiers',
        'emoji': '🎌'
    },
    'airtel_xstream': {
        'name': 'Airtel Xstream',
        'is_free': True,
        'category': 'free',
        'url': 'https://airtelxstream.in',
        'search_url': 'https://www.airtelxstream.in/search?q=',
        'description': 'Free for Airtel users',
        'emoji': '📡'
    },
    'aha': {
        'name': 'Aha',
        'is_free': False,
        'category': 'paid',
        'url': 'https://aha.video',
        'search_url': 'https://www.aha.video/search?q=',
        'description': 'Premium Telugu content platform',
        'emoji': '🎪',
        'regional_strength': ['telugu']
    },
    'sun_nxt': {
        'name': 'Sun NXT',
        'is_free': False,
        'category': 'paid',
        'url': 'https://sunnxt.com',
        'search_url': 'https://www.sunnxt.com/search?q=',
        'description': 'South Indian content specialist',
        'emoji': '☀️',
        'regional_strength': ['tamil', 'telugu', 'malayalam', 'kannada']
    },
    'etvwin': {
        'name': 'ETV Win',
        'is_free': False,
        'category': 'paid',
        'url': 'https://etvwin.com',
        'search_url': 'https://www.etvwin.com/search?q=',
        'description': 'Telugu entertainment platform',
        'emoji': '📽️',
        'regional_strength': ['telugu']
    }
}

# OTT Platform Information (legacy - maintaining backward compatibility)
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
    'alt_balaji': {'name': 'ALTBalaji', 'is_free': False, 'url': 'https://altbalaji.com'}
}

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
    regional_category = db.Column(db.String(50))  # telugu, hindi, etc.
    is_trending = db.Column(db.Boolean, default=False)
    is_all_time_hit = db.Column(db.Boolean, default=False)
    is_new_release = db.Column(db.Boolean, default=False)
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

class StreamingAvailabilityService:
    BASE_URL = 'https://streaming-availability.p.rapidapi.com'
    
    # Platform mapping for Indian OTT services
    PLATFORM_MAPPING = {
        'netflix': {'name': 'Netflix', 'is_free': False, 'category': 'paid', 'emoji': '🔴'},
        'prime': {'name': 'Amazon Prime Video', 'is_free': False, 'category': 'paid', 'emoji': '📦'},
        'hotstar': {'name': 'Disney+ Hotstar', 'is_free': False, 'category': 'paid', 'emoji': '⭐'},
        'zee5': {'name': 'ZEE5', 'is_free': True, 'category': 'freemium', 'emoji': '💎'},
        'zee5_premium': {'name': 'ZEE5 Premium', 'is_free': False, 'category': 'paid', 'emoji': '💎'},
        'sonyliv': {'name': 'SonyLIV', 'is_free': True, 'category': 'freemium', 'emoji': '📱'},
        'sonyliv_premium': {'name': 'SonyLIV Premium', 'is_free': False, 'category': 'paid', 'emoji': '📱'},
        'youtube': {'name': 'YouTube', 'is_free': True, 'category': 'free', 'emoji': '📺'},
        'mxplayer': {'name': 'MX Player', 'is_free': True, 'category': 'free', 'emoji': '🎭'},
        'jiocinema': {'name': 'JioCinema', 'is_free': True, 'category': 'free', 'emoji': '🎬'},
        'crunchyroll': {'name': 'Crunchyroll', 'is_free': True, 'category': 'freemium', 'emoji': '🎌'},
        'airtel': {'name': 'Airtel Xstream', 'is_free': True, 'category': 'free', 'emoji': '📡'},
        'aha': {'name': 'Aha', 'is_free': False, 'category': 'paid', 'emoji': '🎪'},
        'sunnxt': {'name': 'Sun NXT', 'is_free': False, 'category': 'paid', 'emoji': '☀️'}
    }
    
    @staticmethod
    def search_streaming_availability(title, imdb_id=None, tmdb_id=None, timeout=3):
        """Search for streaming availability with shorter timeout"""
        try:
            headers = {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': RAPIDAPI_HOST
            }
            
            # Try searching by IMDB ID first with very short timeout
            if imdb_id:
                try:
                    url = f"{StreamingAvailabilityService.BASE_URL}/shows/{imdb_id}"
                    response = requests.get(url, headers=headers, timeout=timeout)
                    
                    if response.status_code == 200:
                        return StreamingAvailabilityService._process_streaming_data(response.json())
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout fetching streaming data for IMDB ID: {imdb_id}")
                except Exception as e:
                    logger.warning(f"Error fetching streaming data for IMDB ID {imdb_id}: {e}")
            
            # If IMDB search fails, try title search with short timeout
            if title:
                try:
                    url = f"{StreamingAvailabilityService.BASE_URL}/shows/search/title"
                    params = {
                        'title': title,
                        'country': 'in',
                        'show_type': 'movie',
                        'output_language': 'en'
                    }
                    
                    response = requests.get(url, headers=headers, params=params, timeout=timeout)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) > 0:
                            return StreamingAvailabilityService._process_streaming_data(data[0])
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout fetching streaming data for title: {title}")
                except Exception as e:
                    logger.warning(f"Error fetching streaming data for title {title}: {e}")
            
            return StreamingAvailabilityService._get_fallback_streaming_data(title)
            
        except Exception as e:
            logger.error(f"Streaming availability error: {e}")
            return StreamingAvailabilityService._get_fallback_streaming_data(title)
    
    @staticmethod
    def _process_streaming_data(show_data):
        """Process API response and format streaming data"""
        try:
            streaming_info = {
                'free_options': [],
                'paid_options': [],
                'available_countries': [],
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Process streaming options
            streaming_options = show_data.get('streamingOptions', {})
            
            for country, platforms in streaming_options.items():
                if country == 'in':  # Focus on India
                    for platform_data in platforms:
                        service = platform_data.get('service', {})
                        service_id = service.get('id', '').lower()
                        
                        platform_mapping = StreamingAvailabilityService.PLATFORM_MAPPING.get(service_id, {})
                        
                        platform_info = {
                            'platform': service.get('name', 'Unknown'),
                            'platform_id': service_id,
                            'type': platform_data.get('type', 'subscription'),
                            'quality': platform_data.get('quality', 'hd'),
                            'link': platform_data.get('link', ''),
                            'price': platform_data.get('price', {}),
                            'available_since': platform_data.get('availableSince'),
                            'leaving_soon': platform_data.get('leavingSoon', False),
                            'emoji': platform_mapping.get('emoji', '🎬')
                        }
                        
                        # Categorize as free or paid
                        if platform_data.get('type') == 'free' or platform_mapping.get('is_free', False):
                            streaming_info['free_options'].append(platform_info)
                        else:
                            streaming_info['paid_options'].append(platform_info)
            
            return streaming_info
            
        except Exception as e:
            logger.error(f"Processing streaming data error: {e}")
            return StreamingAvailabilityService._get_fallback_streaming_data()
    
    @staticmethod
    def _get_fallback_streaming_data(title=None):
        """Provide fallback streaming options when API fails"""
        return {
            'free_options': [
                {
                    'platform': 'YouTube',
                    'platform_id': 'youtube',
                    'type': 'free',
                    'link': f"https://www.youtube.com/results?search_query={title.replace(' ', '+') if title else 'movie'}+full+movie",
                    'note': 'May have ads or be user-uploaded',
                    'emoji': '📺'
                },
                {
                    'platform': 'MX Player',
                    'platform_id': 'mxplayer',
                    'type': 'free',
                    'link': 'https://www.mxplayer.in/',
                    'note': 'Check availability on platform',
                    'emoji': '🎭'
                },
                {
                    'platform': 'JioCinema',
                    'platform_id': 'jiocinema',
                    'type': 'free',
                    'link': 'https://www.jiocinema.com/',
                    'note': 'Free for Jio users',
                    'emoji': '🎬'
                },
                {
                    'platform': 'Airtel Xstream',
                    'platform_id': 'airtel',
                    'type': 'free',
                    'link': 'https://www.airtelxstream.in/',
                    'note': 'Free for Airtel users',
                    'emoji': '📡'
                }
            ],
            'paid_options': [
                {
                    'platform': 'Netflix',
                    'platform_id': 'netflix',
                    'type': 'subscription',
                    'link': 'https://www.netflix.com/',
                    'note': 'Subscription required',
                    'emoji': '🔴'
                },
                {
                    'platform': 'Amazon Prime Video',
                    'platform_id': 'prime',
                    'type': 'subscription',
                    'link': 'https://www.primevideo.com/',
                    'note': 'Prime membership required',
                    'emoji': '📦'
                },
                {
                    'platform': 'Disney+ Hotstar',
                    'platform_id': 'hotstar',
                    'type': 'subscription',
                    'link': 'https://www.hotstar.com/',
                    'note': 'Subscription required',
                    'emoji': '⭐'
                },
                {
                    'platform': 'ZEE5 Premium',
                    'platform_id': 'zee5_premium',
                    'type': 'subscription',
                    'link': 'https://www.zee5.com/',
                    'note': 'Premium subscription required',
                    'emoji': '💎'
                },
                {
                    'platform': 'SonyLIV Premium',
                    'platform_id': 'sonyliv_premium',
                    'type': 'subscription',
                    'link': 'https://www.sonyliv.com/',
                    'note': 'Premium subscription required',
                    'emoji': '📱'
                }
            ],
            'available_countries': ['in'],
            'last_updated': datetime.utcnow().isoformat(),
            'note': 'Fallback data - please check platforms directly'
        }
    
    @staticmethod
    def get_platform_deep_links(title, platforms):
        """Generate deep links for specific platforms"""
        deep_links = {}
        
        # Platform-specific search URLs
        platform_urls = {
            'netflix': f"https://www.netflix.com/search?q={title.replace(' ', '%20')}",
            'prime': f"https://www.primevideo.com/search/ref=atv_sr_sug_0?phrase={title.replace(' ', '%20')}",
            'hotstar': f"https://www.hotstar.com/in/search?q={title.replace(' ', '%20')}",
            'zee5': f"https://www.zee5.com/search?q={title.replace(' ', '%20')}",
            'sonyliv': f"https://www.sonyliv.com/search?q={title.replace(' ', '%20')}",
            'youtube': f"https://www.youtube.com/results?search_query={title.replace(' ', '+')}+full+movie",
            'mxplayer': f"https://www.mxplayer.in/search?q={title.replace(' ', '%20')}",
            'jiocinema': f"https://www.jiocinema.com/search?q={title.replace(' ', '%20')}",
            'airtel': f"https://www.airtelxstream.in/search?q={title.replace(' ', '%20')}",
            'aha': f"https://www.aha.video/search?q={title.replace(' ', '%20')}",
            'sunnxt': f"https://www.sunnxt.com/search?q={title.replace(' ', '%20')}"
        }
        
        for platform in platforms:
            platform_id = platform.get('platform_id', '').lower()
            if platform_id in platform_urls:
                deep_links[platform_id] = platform_urls[platform_id]
        
        return deep_links

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
    
    @staticmethod
    def get_anime_details(anime_id):
        url = f"{JikanService.BASE_URL}/anime/{anime_id}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan anime details error: {e}")
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

# Enhanced Content Management Service with Regional Focus
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type, fetch_streaming=False):
        try:
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                return existing
            
            # Extract genres
            genres = []
            if 'genres' in tmdb_data:
                genres = [genre['name'] for genre in tmdb_data['genres']]
            elif 'genre_ids' in tmdb_data:
                genres = ContentService.map_genre_ids(tmdb_data['genre_ids'])
            
            # Extract languages and detect regional category
            languages = []
            regional_category = None
            
            if 'spoken_languages' in tmdb_data:
                languages = [lang['name'] for lang in tmdb_data['spoken_languages']]
                # Detect regional category from spoken languages
                for lang_data in tmdb_data['spoken_languages']:
                    lang_code = lang_data.get('iso_639_1', '').lower()
                    for region, config in REGIONAL_LANGUAGES.items():
                        if lang_code in config['codes'] or lang_data.get('name', '').lower() in config['codes']:
                            if not regional_category or config['priority'] < REGIONAL_LANGUAGES.get(regional_category, {}).get('priority', 999):
                                regional_category = region
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
                # Detect regional category from original language
                orig_lang = tmdb_data['original_language'].lower()
                for region, config in REGIONAL_LANGUAGES.items():
                    if orig_lang in config['codes']:
                        regional_category = region
                        break
            
            # Auto-detect Telugu content from title and overview
            title = tmdb_data.get('title') or tmdb_data.get('name', '')
            overview = tmdb_data.get('overview', '')
            
            if not regional_category:
                # Check for Telugu indicators in title/overview
                telugu_indicators = ['tollywood', 'telugu', 'andhra', 'hyderabad', 'telangana']
                content_text = (title + ' ' + overview).lower()
                
                for indicator in telugu_indicators:
                    if indicator in content_text:
                        regional_category = 'telugu'
                        break
                
                # Check for other regional indicators
                if not regional_category:
                    for region, config in REGIONAL_LANGUAGES.items():
                        for term in config['search_terms']:
                            if term.lower() in content_text:
                                if not regional_category or config['priority'] < REGIONAL_LANGUAGES.get(regional_category, {}).get('priority', 999):
                                    regional_category = region
            
            # Extract IMDB ID
            imdb_id = None
            if 'external_ids' in tmdb_data:
                imdb_id = tmdb_data['external_ids'].get('imdb_id')
            elif 'imdb_id' in tmdb_data:
                imdb_id = tmdb_data['imdb_id']
            
            # Determine content categories
            is_trending = tmdb_data.get('popularity', 0) > 100
            is_new_release = False
            is_all_time_hit = False
            
            release_date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
            if release_date_str:
                try:
                    release_date = datetime.strptime(release_date_str, '%Y-%m-%d').date()
                    current_date = datetime.now().date()
                    days_since_release = (current_date - release_date).days
                    
                    # New release if released within last 6 months
                    is_new_release = days_since_release <= 180
                    
                    # All-time hit if high rating and older than 1 year
                    rating = tmdb_data.get('vote_average', 0)
                    vote_count = tmdb_data.get('vote_count', 0)
                    is_all_time_hit = rating >= 7.5 and vote_count >= 1000 and days_since_release > 365
                except:
                    pass
            
            # Get OTT platforms
            ott_platforms = {}
            if fetch_streaming:
                try:
                    ott_platforms = ContentService.get_ott_availability(
                        tmdb_data, 
                        title=title,
                        imdb_id=imdb_id,
                        regional_category=regional_category
                    )
                except Exception as e:
                    logger.error(f"Failed to fetch streaming data: {e}")
                    ott_platforms = ContentService._get_regional_fallback_platforms(regional_category)
            else:
                ott_platforms = ContentService._get_regional_fallback_platforms(regional_category)
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
                imdb_id=imdb_id,
                title=title,
                original_title=tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                content_type=content_type,
                genres=json.dumps(genres),
                languages=json.dumps(languages),
                release_date=datetime.strptime(release_date_str, '%Y-%m-%d').date() if release_date_str else None,
                runtime=tmdb_data.get('runtime'),
                rating=tmdb_data.get('vote_average'),
                vote_count=tmdb_data.get('vote_count'),
                popularity=tmdb_data.get('popularity'),
                overview=tmdb_data.get('overview'),
                poster_path=tmdb_data.get('poster_path'),
                backdrop_path=tmdb_data.get('backdrop_path'),
                ott_platforms=json.dumps(ott_platforms),
                regional_category=regional_category,
                is_trending=is_trending,
                is_new_release=is_new_release,
                is_all_time_hit=is_all_time_hit
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
        """Save anime content from Jikan API to database"""
        try:
            existing = Content.query.filter_by(tmdb_id=anime_data['mal_id'], content_type='anime').first()
            if existing:
                return existing
            
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            aired_data = anime_data.get('aired', {})
            release_date = None
            if aired_data.get('from'):
                try:
                    date_str = aired_data['from'][:10]
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                except:
                    release_date = None
            
            poster_path = None
            images = anime_data.get('images', {})
            if images.get('jpg', {}).get('large_image_url'):
                poster_path = images['jpg']['large_image_url']
            elif images.get('jpg', {}).get('image_url'):
                poster_path = images['jpg']['image_url']
            
            runtime = None
            duration_str = anime_data.get('duration', '')
            if duration_str:
                import re
                minutes = re.findall(r'(\d+)\s*min', duration_str)
                if minutes:
                    runtime = int(minutes[0])
            
            content = Content(
                tmdb_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                languages=json.dumps(['japanese']),
                release_date=release_date,
                runtime=runtime,
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=poster_path,
                ott_platforms=json.dumps(ContentService._get_anime_platforms()),
                regional_category='anime'
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def _get_regional_fallback_platforms(regional_category):
        """Get region-specific fallback platforms"""
        platforms = []
        
        # Telugu-specific platforms
        if regional_category == 'telugu':
            telugu_platforms = ['aha', 'etvwin', 'zee5', 'sun_nxt', 'youtube', 'mx_player']
            for platform_id in telugu_platforms:
                platform_info = ENHANCED_OTT_PLATFORMS.get(platform_id, {})
                if 'telugu' in platform_info.get('regional_strength', []):
                    platforms.append({
                        'platform': platform_id,
                        'name': platform_info.get('name', 'Unknown'),
                        'url': platform_info.get('url', ''),
                        'is_free': platform_info.get('is_free', False),
                        'category': platform_info.get('category', 'unknown'),
                        'emoji': platform_info.get('emoji', '🎬'),
                        'regional_focus': True
                    })
        
        # Add general regional platforms
        general_platforms = ['youtube', 'mx_player', 'jiocinema', 'zee5', 'sonyliv']
        for platform_id in general_platforms:
            platform_info = ENHANCED_OTT_PLATFORMS.get(platform_id, {})
            if regional_category in platform_info.get('regional_strength', []) and platform_id not in [p['platform'] for p in platforms]:
                platforms.append({
                    'platform': platform_id,
                    'name': platform_info.get('name', 'Unknown'),
                    'url': platform_info.get('url', ''),
                    'is_free': platform_info.get('is_free', False),
                    'category': platform_info.get('category', 'unknown'),
                    'emoji': platform_info.get('emoji', '🎬')
                })
        
        return {
            'platforms': platforms,
            'last_updated': datetime.utcnow().isoformat(),
            'total_free': sum(1 for p in platforms if p.get('is_free', False)),
            'total_paid': sum(1 for p in platforms if not p.get('is_free', False)),
            'available_countries': ['in'],
            'regional_focus': regional_category or 'general'
        }
    
    @staticmethod
    def _get_anime_platforms():
        return {
            'platforms': [
                {
                    'platform': 'crunchyroll',
                    'name': 'Crunchyroll',
                    'url': 'https://crunchyroll.com',
                    'is_free': True,
                    'category': 'freemium',
                    'emoji': '🎌'
                },
                {
                    'platform': 'youtube',
                    'name': 'YouTube',
                    'url': 'https://youtube.com',
                    'is_free': True,
                    'category': 'free',
                    'emoji': '📺'
                }
            ],
            'last_updated': datetime.utcnow().isoformat(),
            'total_free': 2,
            'total_paid': 0,
            'available_countries': ['global']
        }
    
    @staticmethod
    def _get_minimal_platforms():
        """Minimal platform data for faster content saving"""
        return {
            'platforms': [],
            'last_updated': datetime.utcnow().isoformat(),
            'total_free': 0,
            'total_paid': 0,
            'available_countries': ['in'],
            'note': 'Streaming data not fetched during initial save'
        }
    
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
    
    @staticmethod
    def get_ott_availability(tmdb_data, title=None, imdb_id=None, regional_category=None):
        """Get OTT platform availability with regional focus"""
        try:
            # Get streaming availability from API
            streaming_data = StreamingAvailabilityService.search_streaming_availability(
                title or tmdb_data.get('title') or tmdb_data.get('name'),
                imdb_id=imdb_id,
                tmdb_id=tmdb_data.get('id')
            )
            
            platforms = []
            
            # Add free options
            for option in streaming_data.get('free_options', []):
                platforms.append({
                    'platform': option.get('platform_id', 'unknown'),
                    'name': option.get('platform', 'Unknown'),
                    'url': option.get('link', ''),
                    'is_free': True,
                    'type': option.get('type', 'free'),
                    'quality': option.get('quality', 'hd'),
                    'category': 'free',
                    'emoji': option.get('emoji', '🎬'),
                    'regional_focus': regional_category in ENHANCED_OTT_PLATFORMS.get(option.get('platform_id', ''), {}).get('regional_strength', [])
                })
            
            # Add paid options
            for option in streaming_data.get('paid_options', []):
                platforms.append({
                    'platform': option.get('platform_id', 'unknown'),
                    'name': option.get('platform', 'Unknown'),
                    'url': option.get('link', ''),
                    'is_free': False,
                    'type': option.get('type', 'subscription'),
                    'quality': option.get('quality', 'hd'),
                    'category': 'paid',
                    'price': option.get('price', {}),
                    'emoji': option.get('emoji', '💎'),
                    'regional_focus': regional_category in ENHANCED_OTT_PLATFORMS.get(option.get('platform_id', ''), {}).get('regional_strength', [])
                })
            
            # Add regional platforms if not found
            if regional_category and not any(p.get('regional_focus') for p in platforms):
                regional_platforms = ContentService._get_regional_fallback_platforms(regional_category)
                platforms.extend(regional_platforms.get('platforms', []))
            
            return {
                'platforms': platforms,
                'last_updated': streaming_data.get('last_updated'),
                'total_free': len([p for p in platforms if p.get('is_free', False)]),
                'total_paid': len([p for p in platforms if not p.get('is_free', False)]),
                'available_countries': streaming_data.get('available_countries', ['in']),
                'regional_focus': regional_category
            }
            
        except Exception as e:
            logger.error(f"OTT availability error: {e}")
            return ContentService._get_regional_fallback_platforms(regional_category)
    
    @staticmethod
    def _get_fallback_platforms():
        """Fallback OTT platforms when API fails"""
        fallback_platforms = []
        
        # Add major platforms as fallback
        major_platforms = ['netflix', 'amazon_prime', 'disney_plus', 'youtube', 'mx_player']
        
        for platform_id in major_platforms:
            platform_info = ENHANCED_OTT_PLATFORMS.get(platform_id, {})
            fallback_platforms.append({
                'platform': platform_id,
                'name': platform_info.get('name', 'Unknown'),
                'url': platform_info.get('url', ''),
                'is_free': platform_info.get('is_free', False),
                'category': platform_info.get('category', 'unknown'),
                'note': 'Check platform for availability',
                'emoji': platform_info.get('emoji', '🎬')
            })
        
        return {
            'platforms': fallback_platforms,
            'last_updated': datetime.utcnow().isoformat(),
            'total_free': sum(1 for p in fallback_platforms if p['is_free']),
            'total_paid': sum(1 for p in fallback_platforms if not p['is_free']),
            'available_countries': ['in'],
            'note': 'Fallback data - verify on platforms'
        }
    
    @staticmethod
    def _format_content_summary(content):
        """Helper method to format content summary"""
        return {
            'id': content.id,
            'title': content.title,
            'content_type': content.content_type,
            'rating': content.rating,
            'poster_path': f"https://image.tmdb.org/t/p/w200{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
            'regional_category': content.regional_category,
            'is_trending': content.is_trending,
            'is_all_time_hit': content.is_all_time_hit,
            'is_new_release': content.is_new_release
        }

# Enhanced Recommendation Engine with Telugu Priority
class RecommendationEngine:
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all', language_preference='telugu'):
        try:
            trending_data = TMDBService.get_trending(content_type=content_type)
            if not trending_data:
                return []
            
            recommendations = []
            telugu_content = []
            other_content = []
            
            for item in trending_data.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected, fetch_streaming=False)
                if content:
                    if content.regional_category == 'telugu':
                        telugu_content.append(content)
                    else:
                        other_content.append(content)
            
            # Prioritize Telugu content
            recommendations.extend(telugu_content[:limit//2])
            recommendations.extend(other_content[:limit - len(recommendations)])
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_popular_by_genre(genre, limit=20, region=None, language_preference='telugu'):
        """Enhanced genre-based recommendations with Telugu priority"""
        try:
            # First get popular movies and TV shows
            popular_movies = TMDBService.get_popular('movie', region=region)
            popular_tv = TMDBService.get_popular('tv', region=region)
            
            recommendations = []
            telugu_content = []
            other_content = []
            
            # Process movies
            if popular_movies:
                for item in popular_movies.get('results', []):
                    item_genres = ContentService.map_genre_ids(item.get('genre_ids', []))
                    if genre.lower() in [g.lower() for g in item_genres]:
                        content = ContentService.save_content_from_tmdb(item, 'movie', fetch_streaming=False)
                        if content:
                            if content.regional_category == 'telugu':
                                telugu_content.append(content)
                            else:
                                other_content.append(content)
            
            # Process TV shows
            if popular_tv:
                for item in popular_tv.get('results', []):
                    item_genres = ContentService.map_genre_ids(item.get('genre_ids', []))
                    if genre.lower() in [g.lower() for g in item_genres]:
                        content = ContentService.save_content_from_tmdb(item, 'tv', fetch_streaming=False)
                        if content:
                            if content.regional_category == 'telugu':
                                telugu_content.append(content)
                            else:
                                other_content.append(content)
            
            # Prioritize Telugu content
            recommendations.extend(telugu_content[:limit//2])
            recommendations.extend(other_content[:limit - len(recommendations)])
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting popular by genre: {e}")
            return []
    
    @staticmethod
    def get_regional_recommendations(language, limit=20):
        """Enhanced regional recommendations with Telugu priority"""
        try:
            if language not in REGIONAL_LANGUAGES:
                language = 'telugu'  # Default to Telugu
            
            lang_config = REGIONAL_LANGUAGES[language]
            recommendations = []
            
            # Search with enhanced terms
            for query in lang_config['search_terms'][:3]:  # Use top 3 search terms
                search_results = TMDBService.search_content(query)
                if search_results:
                    for item in search_results.get('results', []):
                        content_type_detected = 'movie' if 'title' in item else 'tv'
                        content = ContentService.save_content_from_tmdb(item, content_type_detected, fetch_streaming=False)
                        if content and content.regional_category == language:
                            recommendations.append(content)
                        
                        if len(recommendations) >= limit:
                            break
                
                if len(recommendations) >= limit:
                    break
            
            # If not enough content found, get popular regional content from database
            if len(recommendations) < limit:
                db_regional = Content.query.filter_by(regional_category=language)\
                    .order_by(Content.popularity.desc())\
                    .limit(limit - len(recommendations)).all()
                recommendations.extend(db_regional)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting regional recommendations: {e}")
            return []
    
    @staticmethod
    def get_all_time_hits(language='telugu', limit=20):
        """Get all-time hit movies with language preference"""
        try:
            # Get from database first
            db_hits = Content.query.filter_by(
                is_all_time_hit=True,
                regional_category=language
            ).order_by(Content.rating.desc()).limit(limit//2).all()
            
            recommendations = list(db_hits)
            
            # Search for classic hits
            lang_config = REGIONAL_LANGUAGES.get(language, REGIONAL_LANGUAGES['telugu'])
            classic_terms = [f"{lang_config['name']} classic movies", f"{lang_config['industry']} hits"]
            
            for query in classic_terms:
                search_results = TMDBService.search_content(query)
                if search_results:
                    for item in search_results.get('results', []):
                        if item.get('vote_average', 0) >= 7.5 and item.get('vote_count', 0) >= 500:
                            content_type_detected = 'movie' if 'title' in item else 'tv'
                            content = ContentService.save_content_from_tmdb(item, content_type_detected, fetch_streaming=False)
                            if content and content.id not in [r.id for r in recommendations]:
                                content.is_all_time_hit = True
                                db.session.commit()
                                recommendations.append(content)
                            
                            if len(recommendations) >= limit:
                                break
                
                if len(recommendations) >= limit:
                    break
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting all-time hits: {e}")
            return []
    
    @staticmethod
    def get_new_releases(language='telugu', limit=20):
        """Get new releases with language preference"""
        try:
            # Get from database first
            db_new = Content.query.filter_by(
                is_new_release=True,
                regional_category=language
            ).order_by(Content.release_date.desc()).limit(limit//2).all()
            
            recommendations = list(db_new)
            
            # Search for recent releases
            lang_config = REGIONAL_LANGUAGES.get(language, REGIONAL_LANGUAGES['telugu'])
            current_year = datetime.now().year
            new_terms = [f"{lang_config['name']} {current_year} movies", f"latest {lang_config['industry']} movies"]
            
            for query in new_terms:
                search_results = TMDBService.search_content(query)
                if search_results:
                    for item in search_results.get('results', []):
                        release_date = item.get('release_date') or item.get('first_air_date')
                        if release_date:
                            try:
                                release_year = int(release_date[:4])
                                if release_year >= current_year - 1:  # Last 2 years
                                    content_type_detected = 'movie' if 'title' in item else 'tv'
                                    content = ContentService.save_content_from_tmdb(item, content_type_detected, fetch_streaming=False)
                                    if content and content.id not in [r.id for r in recommendations]:
                                        content.is_new_release = True
                                        db.session.commit()
                                        recommendations.append(content)
                                    
                                    if len(recommendations) >= limit:
                                        break
                            except:
                                continue
                
                if len(recommendations) >= limit:
                    break
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return []
    
    @staticmethod
    def get_anime_recommendations(limit=20):
        try:
            top_anime = JikanService.get_top_anime()
            if not top_anime:
                return []
            
            recommendations = []
            for anime in top_anime.get('data', [])[:limit]:
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
            
            # Add regional content based on location (prioritize Telugu)
            if location and location.get('country') == 'India':
                telugu_recs = RecommendationEngine.get_regional_recommendations('telugu', limit=8)
                recommendations.extend(telugu_recs)
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

# Enhanced Telegram Service with Regional Design
class TelegramService:
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            # Get regional language info
            regional_info = REGIONAL_LANGUAGES.get(content.regional_category, {})
            language_emoji = regional_info.get('emoji', '🎬')
            language_flag = regional_info.get('flag', '🇮🇳')
            industry_name = regional_info.get('industry', 'Cinema')
            
            # Format genre list
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            # Get enhanced streaming info
            streaming_info = None
            free_platforms_text = ""
            paid_platforms_text = ""
            
            try:
                streaming_info = StreamingAvailabilityService.search_streaming_availability(
                    content.title,
                    imdb_id=content.imdb_id,
                    tmdb_id=content.tmdb_id,
                    timeout=8
                )
                
                # Enhanced platform formatting with regional focus
                free_options = streaming_info.get('free_options', [])
                if free_options:
                    for option in free_options[:4]:
                        platform_name = option.get('platform', 'Unknown')
                        link = option.get('link', '')
                        emoji = option.get('emoji', '🎬')
                        quality = option.get('quality', '').upper()
                        
                        if not link:
                            platform_search_urls = {
                                'YouTube': f"https://www.youtube.com/results?search_query={content.title.replace(' ', '+')}+{content.regional_category}+full+movie",
                                'MX Player': f"https://www.mxplayer.in/search?q={content.title.replace(' ', '%20')}",
                                'JioCinema': f"https://www.jiocinema.com/search?q={content.title.replace(' ', '%20')}",
                                'Aha': f"https://www.aha.video/search?q={content.title.replace(' ', '%20')}",
                                'ZEE5': f"https://www.zee5.com/search?q={content.title.replace(' ', '%20')}"
                            }
                            link = platform_search_urls.get(platform_name, f"https://www.google.com/search?q={content.title.replace(' ', '+')}+{platform_name.replace(' ', '+')}")
                        
                        quality_badge = f" • `{quality}`" if quality and quality != 'UNKNOWN' else ""
                        free_platforms_text += f"{emoji} [{platform_name}]({link}){quality_badge}\n"
                
                paid_options = streaming_info.get('paid_options', [])
                if paid_options:
                    for option in paid_options[:4]:
                        platform_name = option.get('platform', 'Unknown')
                        link = option.get('link', '')
                        emoji = option.get('emoji', '💎')
                        quality = option.get('quality', '').upper()
                        
                        if not link:
                            platform_search_urls = {
                                'Netflix': f"https://www.netflix.com/search?q={content.title.replace(' ', '%20')}",
                                'Amazon Prime Video': f"https://www.primevideo.com/search/ref=atv_sr_sug_0?phrase={content.title.replace(' ', '%20')}",
                                'Disney+ Hotstar': f"https://www.hotstar.com/in/search?q={content.title.replace(' ', '%20')}",
                                'Aha': f"https://www.aha.video/search?q={content.title.replace(' ', '%20')}",
                                'Sun NXT': f"https://www.sunnxt.com/search?q={content.title.replace(' ', '%20')}"
                            }
                            link = platform_search_urls.get(platform_name, f"https://www.google.com/search?q={content.title.replace(' ', '+')}+{platform_name.replace(' ', '+')}")
                        
                        quality_badge = f" • `{quality}`" if quality and quality != 'UNKNOWN' else ""
                        paid_platforms_text += f"{emoji} [{platform_name}]({link}){quality_badge}\n"
                
            except Exception as e:
                logger.error(f"Error getting streaming info for Telegram: {e}")
                # Enhanced regional fallback
                if content.regional_category == 'telugu':
                    free_platforms_text = f"📺 [YouTube](https://www.youtube.com/results?search_query={content.title.replace(' ', '+')}+telugu+full+movie) • `FREE`\n"
                    free_platforms_text += f"🎭 [MX Player](https://www.mxplayer.in/) • `FREE`\n"
                    free_platforms_text += f"🎪 [Aha](https://www.aha.video/) • `TELUGU SPECIAL`\n"
                    paid_platforms_text = f"🎪 [Aha Premium](https://www.aha.video/) • `TELUGU ORIGINALS`\n"
                    paid_platforms_text += f"☀️ [Sun NXT](https://www.sunnxt.com/) • `SOUTH CINEMA`\n"
                else:
                    free_platforms_text = f"📺 [YouTube](https://www.youtube.com/results?search_query={content.title.replace(' ', '+')}+full+movie) • `FREE`\n"
                    free_platforms_text += f"🎭 [MX Player](https://www.mxplayer.in/) • `FREE`\n"
                    paid_platforms_text = f"🔴 [Netflix](https://www.netflix.com/search?q={content.title.replace(' ', '%20')}) • `SUBSCRIPTION`\n"
                    paid_platforms_text += f"📦 [Prime Video](https://www.primevideo.com/search/ref=atv_sr_sug_0?phrase={content.title.replace(' ', '%20')}) • `SUBSCRIPTION`\n"
            
            # Enhanced poster handling
            poster_url = None
            poster_urls = []
            
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_urls.append(content.poster_path)
                else:
                    poster_urls.extend([
                        f"https://image.tmdb.org/t/p/w500{content.poster_path}",
                        f"https://image.tmdb.org/t/p/w342{content.poster_path}",
                        f"https://image.tmdb.org/t/p/w185{content.poster_path}"
                    ])
            
            if content.backdrop_path:
                if content.backdrop_path.startswith('http'):
                    poster_urls.append(content.backdrop_path)
                else:
                    poster_urls.append(f"https://image.tmdb.org/t/p/w780{content.backdrop_path}")
            
            # Test poster URLs
            for url in poster_urls:
                try:
                    response = requests.head(url, timeout=3)
                    if response.status_code == 200:
                        poster_url = url
                        break
                except:
                    continue
            
            # Enhanced rating display
            rating_stars = ""
            rating_text = ""
            if content.rating:
                rating_value = float(content.rating)
                full_stars = int(rating_value // 2)
                half_star = 1 if (rating_value % 2) >= 1 else 0
                empty_stars = 5 - full_stars - half_star
                rating_stars = "⭐" * full_stars + "💫" * half_star + "☆" * empty_stars
                rating_text = f"{rating_stars} `{content.rating}/10`"
            
            # Content type and regional emoji
            type_emoji_map = {
                'movie': '🎬',
                'tv': '📺', 
                'anime': '🎌'
            }
            type_emoji = type_emoji_map.get(content.content_type, '🎬')
            
            # Runtime formatting
            runtime_text = ""
            if content.runtime:
                hours = content.runtime // 60
                minutes = content.runtime % 60
                if hours > 0:
                    runtime_text = f" • ⏱️ `{hours}h {minutes}m`"
                else:
                    runtime_text = f" • ⏱️ `{minutes}m`"
            
            # Build spectacular regional message
            message_parts = []
            
            # Regional header with language-specific styling
            message_parts.append("✨ ═══════════════════════════════════════ ✨")
            message_parts.append(f"👑 **ADMIN'S CHOICE** 👑")
            message_parts.append(f"{language_flag} *{industry_name} Special by {admin_name}* {language_flag}")
            message_parts.append("✨ ═══════════════════════════════════════ ✨")
            
            # Movie title with regional emphasis
            message_parts.append(f"\n{language_emoji} {type_emoji} **{content.title}**")
            if content.original_title and content.original_title != content.title:
                message_parts.append(f"🌐 *{content.original_title}*")
            
            # Rating with enhanced visual appeal
            if rating_text:
                message_parts.append(f"\n{rating_text}")
            
            # Enhanced movie details box with regional info
            details_box = "┌─ 📋 **MOVIE DETAILS** ─────────────────────┐\n"
            details_box += f"│ 🎭 **Industry:** `{industry_name}`\n"
            details_box += f"│ 📅 **Release:** `{content.release_date or 'N/A'}`\n"
            details_box += f"│ 🎬 **Genres:** `{', '.join(genres_list[:3]) if genres_list else 'N/A'}`\n"
            details_box += f"│ 🎪 **Type:** `{content.content_type.upper()}`{runtime_text}\n"
            if content.vote_count:
                details_box += f"│ 👥 **Votes:** `{content.vote_count:,}`\n"
            details_box += "└─────────────────────────────────────────────┘"
            message_parts.append(f"\n{details_box}")
            
            # Admin's special note with regional context
            message_parts.append(f"\n💭 **ADMIN'S {industry_name.upper()} PICK:**")
            message_parts.append(f"*\"{description}\"*")
            
            # Synopsis
            if content.overview:
                synopsis = content.overview[:250] + "..." if len(content.overview) > 250 else content.overview
                message_parts.append(f"\n📖 **SYNOPSIS:**")
                message_parts.append(f"_{synopsis}_")
            
            # Regional streaming section
            message_parts.append(f"\n🎯 ═══════ **WHERE TO WATCH** ═══════ 🎯")
            
            if free_platforms_text:
                message_parts.append(f"\n🆓 **FREE STREAMING:**")
                message_parts.append(free_platforms_text.rstrip())
            
            if paid_platforms_text:
                message_parts.append(f"\n💰 **PREMIUM PLATFORMS:**")
                message_parts.append(paid_platforms_text.rstrip())
            
            # Regional call to action
            message_parts.append(f"\n🍿 ═══════════════════════════════════════ 🍿")
            message_parts.append(f"🎬 **Ready for {industry_name} Magic?** {language_emoji}")
            message_parts.append("✨ *Click the links above to start watching!* ✨")
            message_parts.append(f"🍿 ═══════════════════════════════════════ 🍿")
            
            # Enhanced hashtags with regional focus
            hashtags = ["#AdminChoice", f"#{industry_name}", f"#{content.regional_category.title()}Cinema", "#WatchNow", "#MovieNight"]
            if genres_list:
                hashtags.extend([f"#{genre.replace(' ', '').replace('-', '')}" for genre in genres_list[:2]])
            if content.content_type:
                hashtags.append(f"#{content.content_type.title()}")
            if content.is_new_release:
                hashtags.append("#NewRelease")
            elif content.is_all_time_hit:
                hashtags.append("#AllTimeHit")
            
            message_parts.append(f"\n{' '.join(hashtags)}")
            
            # Join all parts
            message = '\n'.join(message_parts)
            
            # Send with enhanced error handling
            message_sent = False
            
            # Try with poster first
            if poster_url:
                for attempt in range(3):
                    try:
                        bot.send_photo(
                            chat_id=TELEGRAM_CHANNEL_ID,
                            photo=poster_url,
                            caption=message,
                            parse_mode='Markdown',
                            disable_web_page_preview=False
                        )
                        message_sent = True
                        logger.info(f"Successfully sent regional Telegram recommendation for: {content.title}")
                        break
                    except Exception as photo_error:
                        logger.warning(f"Attempt {attempt + 1} failed to send photo: {photo_error}")
                        if attempt < 2 and len(poster_urls) > attempt + 1:
                            poster_url = poster_urls[attempt + 1]
                        continue
            
            # Fallback to text message
            if not message_sent:
                try:
                    simple_message = f"""{language_flag} {industry_name} ADMIN'S CHOICE by {admin_name}

{language_emoji} {type_emoji} {content.title}
{rating_text if rating_text else '⭐ Rating: N/A'}

🎭 Industry: {industry_name}
📅 Release: {content.release_date or 'N/A'}
🎬 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}

💭 Admin Says: {description}

📖 Synopsis: {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

🎯 WHERE TO WATCH:
{free_platforms_text if free_platforms_text else ''}
{paid_platforms_text if paid_platforms_text else ''}

🎬 Ready for {industry_name} Magic? 🍿

{' '.join(hashtags)}"""
                    
                    bot.send_message(
                        TELEGRAM_CHANNEL_ID, 
                        simple_message, 
                        parse_mode='Markdown',
                        disable_web_page_preview=False
                    )
                    message_sent = True
                    logger.info(f"Successfully sent regional text recommendation for: {content.title}")
                    
                except Exception as text_error:
                    logger.error(f"Failed to send text message: {text_error}")
                    # Final fallback - plain text
                    try:
                        plain_message = f"""ADMIN'S CHOICE by {admin_name}

Movie: {content.title}
Rating: {content.rating or 'N/A'}/10
Release: {content.release_date or 'N/A'}
Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}

Admin Says: {description}

Synopsis: {(content.overview[:150] + '...') if content.overview else 'No synopsis available'}

WHERE TO WATCH:
Check Netflix, Prime Video, YouTube, and other platforms

{' '.join(hashtags)}"""
                        
                        bot.send_message(TELEGRAM_CHANNEL_ID, plain_message)
                        message_sent = True
                        logger.info(f"Successfully sent plain Telegram recommendation for: {content.title}")
                    except Exception as final_error:
                        logger.error(f"Final fallback failed: {final_error}")
            
            return message_sent
            
        except Exception as e:
            logger.error(f"Regional Telegram send error: {e}")
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

# Enhanced Content Discovery Routes
@app.route('/api/search', methods=['GET'])
def search_content():
    try:
        query = request.args.get('query', '')
        content_type = request.args.get('type', 'multi')
        language_filter = request.args.get('language', '')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        session_id = get_session_id()
        
        # Enhanced search with regional terms
        search_query = query
        if language_filter and language_filter in REGIONAL_LANGUAGES:
            lang_config = REGIONAL_LANGUAGES[language_filter]
            search_query = f"{query} {lang_config['name']}"
        
        # Search TMDB
        tmdb_results = TMDBService.search_content(search_query, content_type, page=page)
        
        # Search anime if needed
        anime_results = None
        if content_type in ['anime', 'multi']:
            anime_results = JikanService.search_anime(query, page=page)
        
        # Process results with regional priority
        results = []
        telugu_results = []
        other_results = []
        
        if tmdb_results:
            for item in tmdb_results.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected, fetch_streaming=False)
                if content:
                    # Record interaction
                    interaction = AnonymousInteraction(
                        session_id=session_id,
                        content_id=content.id,
                        interaction_type='search',
                        ip_address=request.remote_addr
                    )
                    db.session.add(interaction)
                    
                    content_data = {
                        'id': content.id,
                        'tmdb_id': content.tmdb_id,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                        'overview': content.overview,
                        'regional_category': content.regional_category,
                        'industry': REGIONAL_LANGUAGES.get(content.regional_category, {}).get('industry', 'Cinema'),
                        'ott_platforms': json.loads(content.ott_platforms or '[]')
                    }
                    
                    if content.regional_category == 'telugu':
                        telugu_results.append(content_data)
                    else:
                        other_results.append(content_data)
        
        # Process anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                content = ContentService.save_anime_from_jikan(anime)
                if content:
                    interaction = AnonymousInteraction(
                        session_id=session_id,
                        content_id=content.id,
                        interaction_type='search',
                        ip_address=request.remote_addr
                    )
                    db.session.add(interaction)
                    
                    other_results.append({
                        'id': content.id,
                        'tmdb_id': content.tmdb_id,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': content.poster_path,
                        'overview': content.overview,
                        'regional_category': content.regional_category,
                        'ott_platforms': json.loads(content.ott_platforms or '[]')
                    })
        
        # Combine results with Telugu priority
        results = telugu_results + other_results
        
        db.session.commit()
        
        return jsonify({
            'results': results,
            'telugu_count': len(telugu_results),
            'other_count': len(other_results),
            'total_results': tmdb_results.get('total_results', 0) if tmdb_results else 0,
            'total_pages': tmdb_results.get('total_pages', 0) if tmdb_results else 0,
            'current_page': page,
            'language_filter': language_filter
        }), 200
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
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
        trailers = []
        similar_content = []
        
        if content.content_type == 'anime':
            # Get anime details from Jikan API
            try:
                jikan_url = f"https://api.jikan.moe/v4/anime/{content.tmdb_id}"
                response = requests.get(jikan_url, timeout=10)
                if response.status_code == 200:
                    anime_data = response.json().get('data', {})
                    
                    # Get anime trailers
                    trailers_data = anime_data.get('trailer', {})
                    if trailers_data.get('youtube_id'):
                        trailers.append({
                            'title': f"{content.title} Trailer",
                            'url': f"https://www.youtube.com/watch?v={trailers_data['youtube_id']}",
                            'thumbnail': trailers_data.get('images', {}).get('medium_image_url', '')
                        })
                    
                    # Get similar anime
                    similar_anime_url = f"https://api.jikan.moe/v4/anime/{content.tmdb_id}/recommendations"
                    try:
                        similar_response = requests.get(similar_anime_url, timeout=5)
                        if similar_response.status_code == 200:
                            similar_data = similar_response.json().get('data', [])
                            for similar in similar_data[:5]:
                                entry = similar.get('entry', {})
                                # Save similar anime and get proper ID
                                similar_anime_data = {
                                    'mal_id': entry.get('mal_id'),
                                    'title': entry.get('title', ''),
                                    'images': entry.get('images', {}),
                                    'genres': [],
                                    'aired': {'from': None},
                                    'synopsis': '',
                                    'score': None
                                }
                                similar_content_obj = ContentService.save_anime_from_jikan(similar_anime_data)
                                if similar_content_obj:
                                    similar_content.append({
                                        'id': similar_content_obj.id,
                                        'title': similar_content_obj.title,
                                        'poster_path': similar_content_obj.poster_path,
                                        'rating': similar_content_obj.rating
                                    })
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Failed to fetch anime details: {e}")
        
        else:
            # Get TMDB details for movies/TV shows
            if content.tmdb_id:
                additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
                
                # Get YouTube trailers
                if YOUTUBE_API_KEY:
                    try:
                        youtube_results = YouTubeService.search_trailers(content.title)
                        if youtube_results:
                            for video in youtube_results.get('items', []):
                                trailers.append({
                                    'title': video['snippet']['title'],
                                    'url': f"https://www.youtube.com/watch?v={video['id']['videoId']}",
                                    'thumbnail': video['snippet']['thumbnails']['medium']['url']
                                })
                    except Exception as e:
                        logger.warning(f"Failed to fetch YouTube trailers: {e}")
                
                # Get similar content
                if additional_details and 'similar' in additional_details:
                    for item in additional_details['similar']['results'][:5]:
                        similar = ContentService.save_content_from_tmdb(item, content.content_type, fetch_streaming=False)
                        if similar:
                            similar_content.append({
                                'id': similar.id,
                                'title': similar.title,
                                'poster_path': f"https://image.tmdb.org/t/p/w300{similar.poster_path}" if similar.poster_path and not similar.poster_path.startswith('http') else similar.poster_path,
                                'rating': similar.rating
                            })
        
        # Get real-time streaming availability (with timeout protection)
        streaming_availability = {'free_options': [], 'paid_options': [], 'last_updated': None}
        
        if content.content_type != 'anime':  # Skip streaming API for anime
            try:
                streaming_availability = StreamingAvailabilityService.search_streaming_availability(
                    content.title,
                    imdb_id=content.imdb_id,
                    tmdb_id=content.tmdb_id,
                    timeout=5
                )
            except Exception as e:
                logger.warning(f"Failed to fetch real-time streaming data for content {content_id}: {e}")
                # Use stored data or fallback
                try:
                    stored_platforms = json.loads(content.ott_platforms or '{}')
                    if stored_platforms and 'platforms' in stored_platforms:
                        free_options = [p for p in stored_platforms['platforms'] if p.get('is_free', False)]
                        paid_options = [p for p in stored_platforms['platforms'] if not p.get('is_free', False)]
                        streaming_availability = {
                            'free_options': free_options,
                            'paid_options': paid_options,
                            'last_updated': stored_platforms.get('last_updated')
                        }
                except:
                    pass
        else:
            # For anime, use stored platforms
            try:
                stored_platforms = json.loads(content.ott_platforms or '{}')
                if stored_platforms and 'platforms' in stored_platforms:
                    free_options = [p for p in stored_platforms['platforms'] if p.get('is_free', False)]
                    paid_options = [p for p in stored_platforms['platforms'] if not p.get('is_free', False)]
                    streaming_availability = {
                        'free_options': free_options,
                        'paid_options': paid_options,
                        'last_updated': stored_platforms.get('last_updated')
                    }
            except:
                pass
        
        # Generate platform deep links
        all_platforms = streaming_availability.get('free_options', []) + streaming_availability.get('paid_options', [])
        deep_links = StreamingAvailabilityService.get_platform_deep_links(content.title, all_platforms)
        
        db.session.commit()
        
        # Enhanced response with streaming availability
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
            'poster_path': content.poster_path,  # Keep original URL for anime
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path and not content.backdrop_path.startswith('http') else content.backdrop_path,
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else [],
            'regional_category': content.regional_category,
            'industry': REGIONAL_LANGUAGES.get(content.regional_category, {}).get('industry', 'Cinema'),
            'is_trending': content.is_trending,
            'is_all_time_hit': content.is_all_time_hit,
            'is_new_release': content.is_new_release,
            
            # Enhanced streaming availability
            'streaming_availability': {
                'free_options': streaming_availability.get('free_options', []),
                'paid_options': streaming_availability.get('paid_options', []),
                'platform_deep_links': deep_links,
                'total_platforms': len(all_platforms),
                'last_updated': streaming_availability.get('last_updated'),
                'summary': {
                    'total_free': len(streaming_availability.get('free_options', [])),
                    'total_paid': len(streaming_availability.get('paid_options', [])),
                    'available_countries': streaming_availability.get('available_countries', ['in'])
                }
            },
            
            # Legacy field for backward compatibility
            'ott_platforms': json.loads(content.ott_platforms or '[]')
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Fallback endpoint for anime details by MAL ID
@app.route('/api/content/anime/<int:mal_id>', methods=['GET'])
def get_anime_details_by_mal_id(mal_id):
    """Fallback endpoint for anime details by MAL ID"""
    try:
        # Try to find existing anime content by MAL ID
        content = Content.query.filter_by(tmdb_id=mal_id, content_type='anime').first()
        
        if not content:
            # Fetch from Jikan API and save
            try:
                jikan_url = f"https://api.jikan.moe/v4/anime/{mal_id}"
                response = requests.get(jikan_url, timeout=10)
                if response.status_code == 200:
                    anime_data = response.json().get('data', {})
                    content = ContentService.save_anime_from_jikan(anime_data)
                    if content:
                        db.session.commit()
            except Exception as e:
                logger.error(f"Failed to fetch anime from Jikan: {e}")
                return jsonify({'error': 'Anime not found'}), 404
        
        if content:
            # Redirect to regular content details
            return get_content_details(content.id)
        else:
            return jsonify({'error': 'Anime not found'}), 404
            
    except Exception as e:
        logger.error(f"Anime details error: {e}")
        return jsonify({'error': 'Failed to get anime details'}), 500

# Streaming Availability Routes
@app.route('/api/content/<int:content_id>/streaming', methods=['GET'])
def get_streaming_availability(content_id):
    """Get streaming availability for specific content"""
    try:
        content = Content.query.get_or_404(content_id)
        
        # Get real-time streaming availability
        streaming_data = StreamingAvailabilityService.search_streaming_availability(
            content.title,
            imdb_id=content.imdb_id,
            tmdb_id=content.tmdb_id
        )
        
        # Generate platform deep links
        all_platforms = streaming_data.get('free_options', []) + streaming_data.get('paid_options', [])
        deep_links = StreamingAvailabilityService.get_platform_deep_links(content.title, all_platforms)
        
        return jsonify({
            'content_id': content.id,
            'title': content.title,
            'streaming_availability': {
                'free_options': streaming_data.get('free_options', []),
                'paid_options': streaming_data.get('paid_options', []),
                'platform_deep_links': deep_links,
                'last_updated': streaming_data.get('last_updated'),
                'summary': {
                    'total_free': len(streaming_data.get('free_options', [])),
                    'total_paid': len(streaming_data.get('paid_options', [])),
                    'platforms_count': len(all_platforms)
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Streaming availability error: {e}")
        return jsonify({'error': 'Failed to get streaming availability'}), 500

@app.route('/api/streaming/search', methods=['GET'])
def search_streaming_by_title():
    """Search streaming availability by title"""
    try:
        title = request.args.get('title', '')
        if not title:
            return jsonify({'error': 'Title parameter required'}), 400
        
        streaming_data = StreamingAvailabilityService.search_streaming_availability(title)
        
        # Generate platform deep links
        all_platforms = streaming_data.get('free_options', []) + streaming_data.get('paid_options', [])
        deep_links = StreamingAvailabilityService.get_platform_deep_links(title, all_platforms)
        
        return jsonify({
            'title': title,
            'streaming_availability': {
                'free_options': streaming_data.get('free_options', []),
                'paid_options': streaming_data.get('paid_options', []),
                'platform_deep_links': deep_links,
                'last_updated': streaming_data.get('last_updated'),
                'summary': {
                    'total_free': len(streaming_data.get('free_options', [])),
                    'total_paid': len(streaming_data.get('paid_options', [])),
                    'platforms_count': len(all_platforms)
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Streaming search error: {e}")
        return jsonify({'error': 'Failed to search streaming availability'}), 500

# Enhanced Regional Content Routes
@app.route('/api/regional/<language>', methods=['GET'])
def get_regional_content(language):
    """Get regional content with Telugu priority"""
    try:
        if language not in REGIONAL_LANGUAGES:
            return jsonify({'error': 'Language not supported'}), 400
        
        category = request.args.get('category', 'popular')  # popular, trending, hits, new
        limit = int(request.args.get('limit', 20))
        
        if category == 'trending':
            recommendations = RecommendationEngine.get_trending_recommendations(limit, language_preference=language)
        elif category == 'hits':
            recommendations = RecommendationEngine.get_all_time_hits(language, limit)
        elif category == 'new':
            recommendations = RecommendationEngine.get_new_releases(language, limit)
        else:  # popular
            recommendations = RecommendationEngine.get_regional_recommendations(language, limit)
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'regional_category': content.regional_category,
                'industry': REGIONAL_LANGUAGES.get(content.regional_category, {}).get('industry', 'Cinema'),
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
                'is_trending': content.is_trending,
                'is_all_time_hit': content.is_all_time_hit,
                'is_new_release': content.is_new_release
            })
        
        return jsonify({
            'language': language,
            'language_info': REGIONAL_LANGUAGES.get(language, {}),
            'category': category,
            'recommendations': result,
            'total': len(result)
        }), 200
        
    except Exception as e:
        logger.error(f"Regional content error: {e}")
        return jsonify({'error': 'Failed to get regional content'}), 500

@app.route('/api/regional/all-languages', methods=['GET'])
def get_multi_language_content():
    """Get content from all languages with Telugu priority"""
    try:
        limit_per_language = int(request.args.get('limit_per_language', 5))
        category = request.args.get('category', 'popular')
        
        all_content = {}
        
        # Process languages in priority order
        sorted_languages = sorted(REGIONAL_LANGUAGES.items(), key=lambda x: x[1]['priority'])
        
        for language, config in sorted_languages:
            try:
                if category == 'trending':
                    recommendations = RecommendationEngine.get_trending_recommendations(limit_per_language, language_preference=language)
                elif category == 'hits':
                    recommendations = RecommendationEngine.get_all_time_hits(language, limit_per_language)
                elif category == 'new':
                    recommendations = RecommendationEngine.get_new_releases(language, limit_per_language)
                else:
                    recommendations = RecommendationEngine.get_regional_recommendations(language, limit_per_language)
                
                content_list = []
                for content in recommendations:
                    content_list.append({
                        'id': content.id,
                        'title': content.title,
                        'content_type': content.content_type,
                        'rating': content.rating,
                        'poster_path': f"https://image.tmdb.org/t/p/w200{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                        'regional_category': content.regional_category,
                        'industry': config.get('industry', 'Cinema')
                    })
                
                all_content[language] = {
                    'language_info': config,
                    'content': content_list
                }
            except Exception as e:
                logger.warning(f"Failed to get content for {language}: {e}")
                continue
        
        return jsonify({
            'category': category,
            'languages': all_content,
            'priority_order': [lang for lang, _ in sorted_languages]
        }), 200
        
    except Exception as e:
        logger.error(f"Multi-language content error: {e}")
        return jsonify({'error': 'Failed to get multi-language content'}), 500

@app.route('/api/genre/<genre>/regional', methods=['GET'])
def get_genre_by_language(genre):
    """Get genre-based content with regional preference"""
    try:
        language = request.args.get('language', 'telugu')
        limit = int(request.args.get('limit', 20))
        
        if genre not in GENRE_CATEGORIES:
            return jsonify({'error': 'Genre not supported'}), 400
        
        recommendations = RecommendationEngine.get_popular_by_genre(genre, limit, language_preference=language)
        
        # Separate by regional category
        regional_content = defaultdict(list)
        for content in recommendations:
            regional_content[content.regional_category or 'other'].append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:100] + '...' if content.overview else '',
                'regional_category': content.regional_category
            })
        
        return jsonify({
            'genre': genre,
            'genre_info': GENRE_CATEGORIES.get(genre, {}),
            'preferred_language': language,
            'content_by_language': dict(regional_content),
            'total': len(recommendations)
        }), 200
        
    except Exception as e:
        logger.error(f"Genre by language error: {e}")
        return jsonify({'error': 'Failed to get genre content'}), 500

@app.route('/api/dashboard/regional', methods=['GET'])
def get_regional_dashboard():
    """Get comprehensive regional dashboard with Telugu focus"""
    try:
        dashboard_data = {}
        
        # Telugu priority content
        telugu_trending = RecommendationEngine.get_trending_recommendations(5, language_preference='telugu')
        telugu_hits = RecommendationEngine.get_all_time_hits('telugu', 5)
        telugu_new = RecommendationEngine.get_new_releases('telugu', 5)
        
        dashboard_data['telugu_special'] = {
            'trending': [ContentService._format_content_summary(c) for c in telugu_trending],
            'all_time_hits': [ContentService._format_content_summary(c) for c in telugu_hits],
            'new_releases': [ContentService._format_content_summary(c) for c in telugu_new]
        }
        
        # Other regional languages
        for language, config in REGIONAL_LANGUAGES.items():
            if language != 'telugu':
                try:
                    regional_content = RecommendationEngine.get_regional_recommendations(language, 3)
                    dashboard_data[language] = {
                        'language_info': config,
                        'popular': [ContentService._format_content_summary(c) for c in regional_content]
                    }
                except:
                    continue
        
        # Genre highlights with Telugu preference
        popular_genres = ['Action', 'Drama', 'Comedy', 'Romance', 'Thriller']
        dashboard_data['genre_highlights'] = {}
        
        for genre in popular_genres:
            try:
                genre_content = RecommendationEngine.get_popular_by_genre(genre, 3, language_preference='telugu')
                dashboard_data['genre_highlights'][genre] = {
                    'genre_info': GENRE_CATEGORIES.get(genre, {}),
                    'content': [ContentService._format_content_summary(c) for c in genre_content]
                }
            except:
                continue
        
        return jsonify({
            'dashboard': dashboard_data,
            'last_updated': datetime.utcnow().isoformat(),
            'featured_language': 'telugu',
            'supported_languages': list(REGIONAL_LANGUAGES.keys())
        }), 200
        
    except Exception as e:
        logger.error(f"Regional dashboard error: {e}")
        return jsonify({'error': 'Failed to get regional dashboard'}), 500

# Existing Recommendation Routes (Enhanced)
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        language_preference = request.args.get('language', 'telugu')
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type, language_preference)
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'regional_category': content.regional_category,
                'industry': REGIONAL_LANGUAGES.get(content.regional_category, {}).get('industry', 'Cinema'),
                'ott_platforms': json.loads(content.ott_platforms or '[]')
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
        language_preference = request.args.get('language', 'telugu')
        
        recommendations = RecommendationEngine.get_popular_by_genre(genre, limit, region, language_preference)
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'regional_category': content.regional_category,
                'industry': REGIONAL_LANGUAGES.get(content.regional_category, {}).get('industry', 'Cinema'),
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Popular by genre error: {e}")
        return jsonify({'error': 'Failed to get popular recommendations'}), 500

@app.route('/api/recommendations/anime', methods=['GET'])
def get_anime():
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_anime_recommendations(limit)
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,  # Use database ID
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
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'regional_category': content.regional_category,
                'industry': REGIONAL_LANGUAGES.get(content.regional_category, {}).get('industry', 'Cinema'),
                'ott_platforms': json.loads(content.ott_platforms or '[]')
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
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'regional_category': content.regional_category,
                            'industry': REGIONAL_LANGUAGES.get(content.regional_category, {}).get('industry', 'Cinema'),
                            'ott_platforms': json.loads(content.ott_platforms or '[]'),
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
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'regional_category': content.regional_category,
                'industry': REGIONAL_LANGUAGES.get(content.regional_category, {}).get('industry', 'Cinema'),
                'ott_platforms': json.loads(content.ott_platforms or '[]')
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
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'regional_category': content.regional_category,
                'industry': REGIONAL_LANGUAGES.get(content.regional_category, {}).get('industry', 'Cinema'),
                'ott_platforms': json.loads(content.ott_platforms or '[]')
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
            # Check by TMDB ID or MAL ID for anime
            if data.get('source') == 'anime':
                existing_content = Content.query.filter_by(tmdb_id=data['id'], content_type='anime').first()
            else:
                existing_content = Content.query.filter_by(tmdb_id=data['id']).first()
        
        if existing_content:
            return jsonify({
                'message': 'Content already exists',
                'content_id': existing_content.id
            }), 200
        
        # Create new content from external data
        try:
            if data.get('source') == 'anime':
                # Handle anime content
                anime_data = {
                    'mal_id': data.get('id'),
                    'title': data.get('title'),
                    'title_japanese': data.get('original_title'),
                    'genres': [{'name': genre} for genre in (data.get('genres') or [])],
                    'aired': {'from': data.get('release_date')},
                    'synopsis': data.get('overview'),
                    'score': data.get('rating'),
                    'images': {'jpg': {'large_image_url': data.get('poster_path')}}
                }
                content = ContentService.save_anime_from_jikan(anime_data)
            else:
                # Handle regular TMDB content
                content = Content(
                    tmdb_id=data.get('id'),
                    title=data.get('title'),
                    original_title=data.get('original_title'),
                    content_type=data.get('content_type', 'movie'),
                    genres=json.dumps(data.get('genres', [])),
                    languages=json.dumps(data.get('languages', ['en'])),
                    release_date=datetime.strptime(data['release_date'], '%Y-%m-%d').date() if data.get('release_date') else None,
                    runtime=data.get('runtime'),
                    rating=data.get('rating'),
                    vote_count=data.get('vote_count'),
                    popularity=data.get('popularity'),
                    overview=data.get('overview'),
                    poster_path=data.get('poster_path'),
                    backdrop_path=data.get('backdrop_path'),
                    ott_platforms=json.dumps(data.get('ott_platforms', [])),
                    regional_category=data.get('regional_category')
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
        
        # Send to Telegram channel with enhanced design
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
                    'regional_category': content.regional_category,
                    'industry': REGIONAL_LANGUAGES.get(content.regional_category, {}).get('industry', 'Cinema'),
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path
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
        
        # Regional content breakdown
        regional_breakdown = {}
        for language in REGIONAL_LANGUAGES.keys():
            count = Content.query.filter_by(regional_category=language).count()
            regional_breakdown[language] = count
        
        # Popular content
        popular_content = db.session.query(
            Content.id, Content.title, Content.regional_category, func.count(UserInteraction.id).label('interaction_count')
        ).join(UserInteraction).group_by(Content.id, Content.title, Content.regional_category)\
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
            'regional_breakdown': regional_breakdown,
            'popular_content': [
                {
                    'title': item.title, 
                    'interactions': item.interaction_count,
                    'regional_category': item.regional_category,
                    'industry': REGIONAL_LANGUAGES.get(item.regional_category, {}).get('industry', 'Cinema')
                }
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
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'regional_category': content.regional_category,
                    'industry': REGIONAL_LANGUAGES.get(content.regional_category, {}).get('industry', 'Cinema'),
                    'ott_platforms': json.loads(content.ott_platforms or '[]'),
                    'admin_description': rec.description,
                    'admin_name': admin.username if admin else 'Admin',
                    'recommended_at': rec.created_at.isoformat()
                })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Public admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get admin recommendations'}), 500

# Background task to update streaming data
def update_streaming_data_background():
    """Background task to update streaming data for popular content"""
    while True:
        try:
            with app.app_context():
                # Get popular content without streaming data
                contents = Content.query.filter(
                    or_(
                        Content.ott_platforms == '{}',
                        Content.ott_platforms == None,
                        Content.ott_platforms.like('%"note": "Streaming data not fetched during initial save"%')
                    )
                ).limit(10).all()
                
                for content in contents:
                    try:
                        # Update streaming data with timeout
                        streaming_data = StreamingAvailabilityService.search_streaming_availability(
                            content.title,
                            imdb_id=content.imdb_id,
                            tmdb_id=content.tmdb_id,
                            timeout=5
                        )
                        
                        platforms_data = {
                            'platforms': streaming_data.get('free_options', []) + streaming_data.get('paid_options', []),
                            'last_updated': streaming_data.get('last_updated'),
                            'total_free': len(streaming_data.get('free_options', [])),
                            'total_paid': len(streaming_data.get('paid_options', [])),
                            'available_countries': streaming_data.get('available_countries', ['in']),
                            'regional_focus': content.regional_category
                        }
                        
                        content.ott_platforms = json.dumps(platforms_data)
                        db.session.commit()
                        
                        logger.info(f"Updated streaming data for: {content.title}")
                        time.sleep(2)  # Rate limiting
                        
                    except Exception as e:
                        logger.warning(f"Failed to update streaming data for {content.title}: {e}")
                        continue
                
            # Sleep for 1 hour before next batch
            time.sleep(3600)
            
        except Exception as e:
            logger.error(f"Background streaming update error: {e}")
            time.sleep(300)  # Wait 5 minutes before retry

# Start background thread if enabled
if os.environ.get('ENABLE_BACKGROUND_STREAMING_UPDATE', 'false').lower() == 'true':
    streaming_thread = threading.Thread(target=update_streaming_data_background, daemon=True)
    streaming_thread.start()

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'features': [
            'Enhanced Telugu-priority regional content',
            'Multi-language support with priority system',
            'Advanced streaming availability',
            'Regional OTT platform recommendations',
            'Beautiful Telegram post designs',
            'Comprehensive genre categorization',
            'All-time hits and new releases detection',
            'Regional dashboard and analytics'
        ],
        'supported_languages': list(REGIONAL_LANGUAGES.keys()),
        'streaming_api_enabled': bool(RAPIDAPI_KEY),
        'telegram_enabled': bool(bot),
        'regional_focus': 'Telugu-first with multi-language support'
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
                    preferred_languages=json.dumps(['telugu', 'hindi', 'english']),
                    preferred_genres=json.dumps(['Action', 'Drama', 'Comedy'])
                )
                db.session.add(admin)
                db.session.commit()
                logger.info("Admin user created with username: admin, password: admin123")
                logger.info("Admin preferences set to Telugu-first with multi-language support")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

# Initialize database when app starts
create_tables()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info("🎬 Starting Enhanced Regional Movie Recommendation System")
    logger.info("🇮🇳 Telugu-first priority with multi-language support")
    logger.info("✨ Features: Regional content, Beautiful Telegram posts, OTT recommendations")
    
    app.run(host='0.0.0.0', port=port, debug=debug)