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
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '52260795')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '-1002850793757')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-xmf5.onrender.com')

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

# Enhanced OTT Platform Information with direct links support
OTT_PLATFORMS = {
    # Free Platforms
    'mx_player': {
        'name': 'MX Player', 
        'is_free': True, 
        'url': 'https://www.mxplayer.in',
        'deep_link': 'https://www.mxplayer.in/movie/{content_id}',
        'logo': 'https://www.mxplayer.in/assets/images/logo.png'
    },
    'jiocinema': {
        'name': 'JioCinema', 
        'is_free': True, 
        'url': 'https://www.jiocinema.com',
        'deep_link': 'https://www.jiocinema.com/movies/{content_id}',
        'logo': 'https://www.jiocinema.com/images/jiocinema_logo.png'
    },
    'sonyliv_free': {
        'name': 'SonyLIV Free', 
        'is_free': True, 
        'url': 'https://www.sonyliv.com',
        'deep_link': 'https://www.sonyliv.com/movies/{content_id}',
        'logo': 'https://www.sonyliv.com/images/logo.png'
    },
    'youtube': {
        'name': 'YouTube', 
        'is_free': True, 
        'url': 'https://www.youtube.com',
        'deep_link': 'https://www.youtube.com/watch?v={content_id}',
        'logo': 'https://www.youtube.com/img/desktop/yt_1200.png'
    },
    'airtel_xstream': {
        'name': 'Airtel Xstream', 
        'is_free': True, 
        'url': 'https://www.airtelxstream.in',
        'deep_link': 'https://www.airtelxstream.in/movies/{content_id}',
        'logo': 'https://www.airtelxstream.in/images/logo.png'
    },
    'voot': {
        'name': 'Voot', 
        'is_free': True, 
        'url': 'https://www.voot.com',
        'deep_link': 'https://www.voot.com/movies/{content_id}',
        'logo': 'https://www.voot.com/images/logo.png'
    },
    
    # Paid Platforms
    'netflix': {
        'name': 'Netflix', 
        'is_free': False, 
        'url': 'https://www.netflix.com',
        'deep_link': 'https://www.netflix.com/title/{content_id}',
        'logo': 'https://assets.nflxext.com/us/ffe/siteui/common/icons/nficon2016.png'
    },
    'amazon_prime': {
        'name': 'Amazon Prime Video', 
        'is_free': False, 
        'url': 'https://www.primevideo.com',
        'deep_link': 'https://www.primevideo.com/detail/{content_id}',
        'logo': 'https://images-na.ssl-images-amazon.com/images/G/01/digital/video/web/Logo-min.png'
    },
    'disney_plus_hotstar': {
        'name': 'Disney+ Hotstar', 
        'is_free': False, 
        'url': 'https://www.hotstar.com',
        'deep_link': 'https://www.hotstar.com/in/movies/{content_id}',
        'logo': 'https://secure-media.hotstarext.com/web/images/logo.svg'
    },
    'zee5': {
        'name': 'ZEE5', 
        'is_free': False, 
        'url': 'https://www.zee5.com',
        'deep_link': 'https://www.zee5.com/movies/details/{content_id}',
        'logo': 'https://www.zee5.com/images/logo.png'
    },
    'sonyliv': {
        'name': 'SonyLIV', 
        'is_free': False, 
        'url': 'https://www.sonyliv.com',
        'deep_link': 'https://www.sonyliv.com/movies/{content_id}',
        'logo': 'https://www.sonyliv.com/images/logo.png'
    },
    'aha': {
        'name': 'Aha', 
        'is_free': False, 
        'url': 'https://www.aha.video',
        'deep_link': 'https://www.aha.video/movies/{content_id}',
        'logo': 'https://www.aha.video/images/logo.png'
    },
    'sun_nxt': {
        'name': 'Sun NXT', 
        'is_free': False, 
        'url': 'https://www.sunnxt.com',
        'deep_link': 'https://www.sunnxt.com/movie/{content_id}',
        'logo': 'https://www.sunnxt.com/images/logo.png'
    },
    'alt_balaji': {
        'name': 'ALTBalaji', 
        'is_free': False, 
        'url': 'https://www.altbalaji.com',
        'deep_link': 'https://www.altbalaji.com/movie/{content_id}',
        'logo': 'https://www.altbalaji.com/images/logo.png'
    }
}

# Regional Language Mapping
REGIONAL_LANGUAGES = {
    'hindi': ['hi', 'hindi', 'bollywood'],
    'telugu': ['te', 'telugu', 'tollywood'],
    'tamil': ['ta', 'tamil', 'kollywood'],
    'kannada': ['kn', 'kannada', 'sandalwood'],
    'malayalam': ['ml', 'malayalam', 'mollywood'],
    'english': ['en', 'english', 'hollywood'],
    'bengali': ['bn', 'bengali', 'tollywood'],
    'punjabi': ['pa', 'punjabi'],
    'marathi': ['mr', 'marathi'],
    'gujarati': ['gu', 'gujarati']
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

# Enhanced OTT Availability Services
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
                # Update OTT platforms if content exists
                updated_ott = ContentService.get_ott_availability(tmdb_data, existing.imdb_id)
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
            
            # Get OTT platforms with direct links
            ott_platforms = ContentService.get_ott_availability(tmdb_data, imdb_id)
            
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
    def get_ott_availability(tmdb_data, imdb_id=None):
        """Get real OTT availability using multiple APIs with enhanced fallbacks"""
        platforms = []
        title = tmdb_data.get('title') or tmdb_data.get('name', '')
        
        try:
            # Try WatchMode API first
            watchmode_results = WatchModeService.search_title(title)
            if watchmode_results and watchmode_results.get('title_results'):
                first_result = watchmode_results['title_results'][0]
                watchmode_id = first_result.get('id')
                
                if watchmode_id:
                    sources = WatchModeService.get_title_sources(watchmode_id)
                    if sources:
                        platforms.extend(ContentService.parse_watchmode_sources(sources))
            
            # Try Streaming Availability API
            if imdb_id:
                streaming_data = StreamingAvailabilityService.get_title_details(imdb_id)
                if streaming_data:
                    platforms.extend(ContentService.parse_streaming_availability(streaming_data))
            
            # If no results from APIs, try search by title
            if not platforms:
                streaming_search = StreamingAvailabilityService.search_title(title)
                if streaming_search and streaming_search.get('result'):
                    for result in streaming_search['result'][:1]:  # Take first result
                        platforms.extend(ContentService.parse_streaming_availability(result))
            
            # ENHANCED: Add fallback platforms with search links if no API results
            if not platforms:
                platforms = ContentService.get_fallback_ott_platforms(title, tmdb_data)
            
            # Remove duplicates and sort by free vs paid
            unique_platforms = ContentService.deduplicate_platforms(platforms)
            
            # Add Telegram deep links
            for platform in unique_platforms:
                platform['telegram_link'] = ContentService.generate_telegram_link(title, platform)
            
            return unique_platforms
            
        except Exception as e:
            logger.error(f"Error getting OTT availability: {e}")
            # Return fallback platforms if API fails
            return ContentService.get_fallback_ott_platforms(title, tmdb_data)   
    @staticmethod
    def get_fallback_ott_platforms(title, tmdb_data):
        """Get fallback OTT platforms with search links when APIs fail"""
        encoded_title = urllib.parse.quote(title.replace(' ', '+'))
        platforms = []
        
        # Free platforms with search links
        free_platforms = [
            {
                'name': 'YouTube',
                'is_free': True,
                'url': 'https://www.youtube.com',
                'direct_url': f"https://www.youtube.com/results?search_query={encoded_title}+full+movie",
                'search_url': f"https://www.youtube.com/results?search_query={encoded_title}",
                'logo': 'https://www.youtube.com/img/desktop/yt_1200.png',
                'type': 'free_search',
                'audio_languages': ['hindi', 'english', 'tamil', 'telugu'],
                'note': 'Search for full movies and clips'
            },
            {
                'name': 'MX Player',
                'is_free': True,
                'url': 'https://www.mxplayer.in',
                'direct_url': f"https://www.mxplayer.in/search?q={encoded_title}",
                'search_url': f"https://www.mxplayer.in/search?q={encoded_title}",
                'logo': 'https://www.mxplayer.in/assets/images/logo.png',
                'type': 'free_search',
                'audio_languages': ['hindi', 'english', 'tamil', 'telugu', 'kannada'],
                'note': 'Free movies and shows'
            },
            {
                'name': 'JioCinema',
                'is_free': True,
                'url': 'https://www.jiocinema.com',
                'direct_url': f"https://www.jiocinema.com/search/{encoded_title}",
                'search_url': f"https://www.jiocinema.com/search/{encoded_title}",
                'logo': 'https://www.jiocinema.com/images/jiocinema_logo.png',
                'type': 'free_search',
                'audio_languages': ['hindi', 'english', 'tamil', 'telugu'],
                'note': 'Free with ads'
            }
        ]
        
        # Paid platforms with search links
        paid_platforms = [
            {
                'name': 'Netflix',
                'is_free': False,
                'url': 'https://www.netflix.com',
                'direct_url': f"https://www.netflix.com/search?q={encoded_title}",
                'search_url': f"https://www.netflix.com/search?q={encoded_title}",
                'logo': 'https://assets.nflxext.com/us/ffe/siteui/common/icons/nficon2016.png',
                'type': 'subscription',
                'audio_languages': ['hindi', 'english', 'tamil', 'telugu'],
                'note': 'Subscription required'
            },
            {
                'name': 'Amazon Prime Video',
                'is_free': False,
                'url': 'https://www.primevideo.com',
                'direct_url': f"https://www.primevideo.com/search/ref=atv_nb_sr?phrase={encoded_title}",
                'search_url': f"https://www.primevideo.com/search/ref=atv_nb_sr?phrase={encoded_title}",
                'logo': 'https://images-na.ssl-images-amazon.com/images/G/01/digital/video/web/Logo-min.png',
                'type': 'subscription',
                'audio_languages': ['hindi', 'english', 'tamil', 'telugu'],
                'note': 'Prime membership required'
            },
            {
                'name': 'Disney+ Hotstar',
                'is_free': False,
                'url': 'https://www.hotstar.com',
                'direct_url': f"https://www.hotstar.com/in/search?q={encoded_title}",
                'search_url': f"https://www.hotstar.com/in/search?q={encoded_title}",
                'logo': 'https://secure-media.hotstarext.com/web/images/logo.svg',
                'type': 'subscription',
                'audio_languages': ['hindi', 'english', 'tamil', 'telugu'],
                'note': 'Subscription required'
            }
        ]
        
        # Combine and return based on content type
        all_platforms = free_platforms + paid_platforms
        
        # For anime, prioritize anime-specific platforms
        if tmdb_data.get('genre_ids') and 16 in tmdb_data.get('genre_ids', []):  # Animation genre
            anime_platforms = [
                {
                    'name': 'Crunchyroll',
                    'is_free': False,
                    'url': 'https://www.crunchyroll.com',
                    'direct_url': f"https://www.crunchyroll.com/search?q={encoded_title}",
                    'search_url': f"https://www.crunchyroll.com/search?q={encoded_title}",
                    'logo': 'https://www.crunchyroll.com/img/header_logo.png',
                    'type': 'subscription',
                    'audio_languages': ['japanese'],
                    'subtitle_languages': ['english', 'hindi'],
                    'note': 'Anime streaming service'
                }
            ]
            all_platforms = anime_platforms + all_platforms
        
        return all_platforms[:6]  # Return top 6 platforms
    @staticmethod
    def parse_watchmode_sources(sources_data):
        """Parse WatchMode API response to extract platform information"""
        platforms = []
        
        # WatchMode source ID to platform mapping
        watchmode_platform_map = {
            203: 'netflix',
            26: 'amazon_prime',
            372: 'disney_plus_hotstar',
            457: 'zee5',
            237: 'youtube',
            # Add more mappings as needed
        }
        
        try:
            for source in sources_data:
                source_id = source.get('source_id')
                platform_key = watchmode_platform_map.get(source_id)
                
                if platform_key and platform_key in OTT_PLATFORMS:
                    platform_info = OTT_PLATFORMS[platform_key].copy()
                    
                    # Add direct watch link
                    if source.get('web_url'):
                        platform_info['direct_url'] = source['web_url']
                    
                    # Add pricing info
                    platform_info['type'] = source.get('type', 'subscription')
                    platform_info['price'] = source.get('price')
                    
                    # Add audio languages if available
                    platform_info['audio_languages'] = source.get('audio_languages', ['original'])
                    
                    platforms.append(platform_info)
                    
        except Exception as e:
            logger.error(f"Error parsing WatchMode sources: {e}")
        
        return platforms
    
    @staticmethod
    def parse_streaming_availability(streaming_data):
        """Parse Streaming Availability API response"""
        platforms = []
        
        # Streaming service mapping
        streaming_platform_map = {
            'netflix': 'netflix',
            'prime': 'amazon_prime',
            'hotstar': 'disney_plus_hotstar',
            'zee5': 'zee5',
            'sonyliv': 'sonyliv',
            'jiocinema': 'jiocinema',
            'mxplayer': 'mx_player',
            'youtube': 'youtube',
            'voot': 'voot',
            'aha': 'aha',
            'sunnxt': 'sun_nxt'
        }
        
        try:
            streaming_info = streaming_data.get('streamingInfo', {})
            
            for service_key, service_data in streaming_info.items():
                platform_key = streaming_platform_map.get(service_key.lower())
                
                if platform_key and platform_key in OTT_PLATFORMS:
                    platform_info = OTT_PLATFORMS[platform_key].copy()
                    
                    # Extract service data (could be list for different regions)
                    if isinstance(service_data, list) and service_data:
                        service_info = service_data[0]  # Take first region
                    else:
                        service_info = service_data
                    
                    # Add direct watch link
                    if service_info.get('link'):
                        platform_info['direct_url'] = service_info['link']
                    
                    # Add quality and audio info
                    platform_info['quality'] = service_info.get('quality', 'HD')
                    platform_info['audio_languages'] = service_info.get('audioLanguages', ['original'])
                    platform_info['subtitle_languages'] = service_info.get('subtitleLanguages', [])
                    
                    # Add multiple audio language links if available
                    if 'audios' in service_info:
                        platform_info['audio_links'] = []
                        for audio in service_info['audios']:
                            platform_info['audio_links'].append({
                                'language': audio.get('language'),
                                'link': audio.get('link', service_info.get('link'))
                            })
                    
                    platforms.append(platform_info)
                    
        except Exception as e:
            logger.error(f"Error parsing streaming availability: {e}")
        
        return platforms
    
    @staticmethod
    def deduplicate_platforms(platforms):
        """Remove duplicate platforms and sort by availability"""
        seen_platforms = set()
        unique_platforms = []
        
        # Sort: free platforms first, then paid
        platforms.sort(key=lambda x: (not x.get('is_free', False), x.get('name', '')))
        
        for platform in platforms:
            platform_name = platform.get('name', '')
            if platform_name not in seen_platforms:
                seen_platforms.add(platform_name)
                unique_platforms.append(platform)
        
        return unique_platforms
    
    @staticmethod
    def generate_telegram_link(title, platform):
        """Generate Telegram deep link for sharing"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return None
            
            message = f"🎬 Watch '{title}' on {platform.get('name')}"
            if platform.get('direct_url'):
                message += f"\n🔗 {platform['direct_url']}"
            
            if platform.get('is_free'):
                message += "\n✅ FREE"
            else:
                message += "\n💰 PAID"
            
            # Create Telegram share link
            encoded_message = urllib.parse.quote(message)
            telegram_link = f"https://t.me/share/url?url={encoded_message}"
            
            return telegram_link
            
        except Exception as e:
            logger.error(f"Error generating Telegram link: {e}")
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
            
            # Get anime streaming platforms
            ott_platforms = ContentService.get_anime_ott_platforms(anime_data)
            
            # Create content object
            content = Content(
                tmdb_id=anime_data['mal_id'],  # Store MAL ID in tmdb_id field
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                languages=json.dumps(['japanese']),
                release_date=release_date,
                runtime=anime_data.get('duration'),  # Duration in minutes
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
    def get_anime_ott_platforms(anime_data):
        """Get OTT platforms for anime content"""
        platforms = []
        
        # Common anime streaming platforms with direct links
        anime_platforms = [
            {
                'name': 'YouTube',
                'is_free': True,
                'url': 'https://www.youtube.com',
                'direct_url': f"https://www.youtube.com/results?search_query={anime_data.get('title', '').replace(' ', '+')}_anime",
                'logo': 'https://www.youtube.com/img/desktop/yt_1200.png',
                'type': 'free',
                'audio_languages': ['japanese'],
                'subtitle_languages': ['english', 'hindi']
            },
            {
                'name': 'MX Player',
                'is_free': True,
                'url': 'https://www.mxplayer.in',
                'direct_url': f"https://www.mxplayer.in/search?q={anime_data.get('title', '').replace(' ', '%20')}",
                'logo': 'https://www.mxplayer.in/assets/images/logo.png',
                'type': 'free',
                'audio_languages': ['japanese', 'hindi'],
                'subtitle_languages': ['english', 'hindi']
            },
            {
                'name': 'Crunchyroll',
                'is_free': False,
                'url': 'https://www.crunchyroll.com',
                'direct_url': f"https://www.crunchyroll.com/search?q={anime_data.get('title', '').replace(' ', '%20')}",
                'logo': 'https://www.crunchyroll.com/img/header_logo.png',
                'type': 'subscription',
                'audio_languages': ['japanese'],
                'subtitle_languages': ['english', 'spanish', 'portuguese']
            }
        ]
        
        # Add more platforms based on anime popularity or genre
        if anime_data.get('score', 0) > 8.0:  # Popular anime
            anime_platforms.append({
                'name': 'Netflix',
                'is_free': False,
                'url': 'https://www.netflix.com',
                'direct_url': f"https://www.netflix.com/search?q={anime_data.get('title', '').replace(' ', '%20')}",
                'logo': 'https://assets.nflxext.com/us/ffe/siteui/common/icons/nficon2016.png',
                'type': 'subscription',
                'audio_languages': ['japanese', 'english'],
                'subtitle_languages': ['english', 'hindi', 'spanish']
            })
        
        return anime_platforms
    
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
            # First get popular movies
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
            # Search for content in specific language
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
                # Convert anime data to our content format
                content = Content(
                    title=anime.get('title'),
                    original_title=anime.get('title_japanese'),
                    content_type='anime',
                    genres=json.dumps([genre['name'] for genre in anime.get('genres', [])]),
                    languages=json.dumps(['japanese']),
                    rating=anime.get('score'),
                    overview=anime.get('synopsis'),
                    poster_path=anime.get('images', {}).get('jpg', {}).get('image_url'),
                    ott_platforms=json.dumps([])  # You would check anime streaming platforms
                )
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
            
            # Create OTT availability text
            ott_text = ""
            free_platforms = [p for p in ott_platforms if p.get('is_free')]
            paid_platforms = [p for p in ott_platforms if not p.get('is_free')]
            
            if free_platforms:
                ott_text += "\n\n🆓 **FREE TO WATCH:**\n"
                for platform in free_platforms[:3]:  # Limit to 3 platforms
                    ott_text += f"▶️ {platform['name']}"
                    if platform.get('direct_url'):
                        ott_text += f" - [Watch Now]({platform['direct_url']})"
                    ott_text += "\n"
            
            if paid_platforms:
                ott_text += "\n💰 **PREMIUM PLATFORMS:**\n"
                for platform in paid_platforms[:3]:  # Limit to 3 platforms
                    ott_text += f"▶️ {platform['name']}"
                    if platform.get('direct_url'):
                        ott_text += f" - [Watch Now]({platform['direct_url']})"
                    ott_text += "\n"
            
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
            
            # Group by free and paid
            free_platforms = [p for p in ott_platforms if p.get('is_free')]
            paid_platforms = [p for p in ott_platforms if not p.get('is_free')]
            
            if free_platforms:
                message += "🆓 **FREE PLATFORMS:**\n"
                for platform in free_platforms:
                    message += f"▶️ **{platform['name']}**"
                    if platform.get('direct_url'):
                        message += f" - [Watch Now]({platform['direct_url']})"
                    
                    # Add audio language info
                    if platform.get('audio_languages'):
                        langs = ', '.join(platform['audio_languages'][:3])
                        message += f"\n   🎵 Audio: {langs}"
                    
                    message += "\n\n"
            
            if paid_platforms:
                message += "💰 **PREMIUM PLATFORMS:**\n"
                for platform in paid_platforms:
                    message += f"▶️ **{platform['name']}**"
                    if platform.get('direct_url'):
                        message += f" - [Watch Now]({platform['direct_url']})"
                    
                    # Add audio language info
                    if platform.get('audio_languages'):
                        langs = ', '.join(platform['audio_languages'][:3])
                        message += f"\n   🎵 Audio: {langs}"
                    
                    message += "\n\n"
            
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

# Enhanced search with OTT filtering
@app.route('/api/search', methods=['GET'])
def search_content():
    try:
        query = request.args.get('query', '')
        content_type = request.args.get('type', 'multi')
        page = int(request.args.get('page', 1))
        free_only = request.args.get('free_only', '').lower() == 'true'
        platform_filter = request.args.get('platform', '')
        
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
                    # Parse OTT platforms for filtering
                    ott_platforms = []
                    if content.ott_platforms:
                        try:
                            ott_platforms = json.loads(content.ott_platforms)
                        except:
                            ott_platforms = []
                    
                    # Apply filters
                    if free_only and not any(p.get('is_free') for p in ott_platforms):
                        continue
                    
                    if platform_filter:
                        if not any(platform_filter.lower() in p.get('name', '').lower() for p in ott_platforms):
                            continue
                    
                    # Record anonymous interaction
                    interaction = AnonymousInteraction(
                        session_id=session_id,
                        content_id=content.id,
                        interaction_type='search',
                        ip_address=request.remote_addr
                    )
                    db.session.add(interaction)
                    
                    # Group platforms
                    free_platforms = [p for p in ott_platforms if p.get('is_free')]
                    paid_platforms = [p for p in ott_platforms if not p.get('is_free')]
                    
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
                        'ott_summary': {
                            'has_free': len(free_platforms) > 0,
                            'free_count': len(free_platforms),
                            'paid_count': len(paid_platforms),
                            'top_platforms': [p['name'] for p in (free_platforms + paid_platforms)[:3]]
                        }
                    })
        
        # FIXED: Properly save anime results to database
        if anime_results:
            for anime in anime_results.get('data', []):
                # Save anime to database properly
                content = ContentService.save_anime_content(anime)
                if content:
                    results.append({
                        'id': content.id,  # Use database ID, not MAL ID
                        'title': content.title,
                        'content_type': 'anime',
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': content.poster_path,
                        'overview': content.overview,
                        'ott_summary': {
                            'has_free': True,  # Most anime available on free platforms
                            'free_count': 2,
                            'paid_count': 1,
                            'top_platforms': ['YouTube', 'MX Player', 'Crunchyroll']
                        }
                    })
        
        db.session.commit()
        
        return jsonify({
            'results': results,
            'total_results': tmdb_results.get('total_results', 0) if tmdb_results else 0,
            'total_pages': tmdb_results.get('total_pages', 0) if tmdb_results else 0,
            'current_page': page,
            'filters_applied': {
                'free_only': free_only,
                'platform_filter': platform_filter
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

# Updated Content Discovery Route with Enhanced OTT Info
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
            
            # Update OTT platforms with latest data
            if additional_details:
                updated_ott = ContentService.get_ott_availability(additional_details, content.imdb_id)
                if updated_ott:
                    content.ott_platforms = json.dumps(updated_ott)
                    content.updated_at = datetime.utcnow()
        
        # For anime, get anime-specific details
        elif content.content_type == 'anime':
            # Try to get more anime details from Jikan if needed
            try:
                anime_details = requests.get(f"https://api.jikan.moe/v4/anime/{content.tmdb_id}", timeout=10)
                if anime_details.status_code == 200:
                    additional_details = anime_details.json().get('data', {})
            except:
                additional_details = {}
        
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
                        'thumbnail': video['snippet']['thumbnails']['medium']['url'],
                        'telegram_share': f"https://t.me/share/url?url=https://www.youtube.com/watch?v={video['id']['videoId']}"
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
        
        # Parse OTT platforms
        ott_platforms = []
        if content.ott_platforms:
            try:
                ott_platforms = json.loads(content.ott_platforms)
            except:
                ott_platforms = []
        
        # If no OTT platforms found, get fallback
        if not ott_platforms:
            tmdb_data = {'title': content.title, 'name': content.title}
            ott_platforms = ContentService.get_fallback_ott_platforms(content.title, tmdb_data)
            # Update content with fallback platforms
            content.ott_platforms = json.dumps(ott_platforms)
            content.updated_at = datetime.utcnow()
        
        # Group OTT platforms by type
        free_platforms = [p for p in ott_platforms if p.get('is_free')]
        paid_platforms = [p for p in ott_platforms if not p.get('is_free')]
        
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
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details and content.content_type != 'anime' else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details and content.content_type != 'anime' else [],
            
            # Enhanced OTT Information with direct links
            'streaming_info': {
                'available': len(ott_platforms) > 0,
                'free_options': len(free_platforms),
                'paid_options': len(paid_platforms),
                'last_updated': content.updated_at.isoformat() if content.updated_at else None
            },
            'ott_platforms': {
                'free_platforms': free_platforms,
                'paid_platforms': paid_platforms,
                'total_platforms': len(ott_platforms)
            },
            
            # Direct sharing links
            'sharing': {
                'telegram_channel': f"https://t.me/{TELEGRAM_CHANNEL_ID.replace('@', '').replace('-100', '')}",
                'content_telegram_share': f"Watch {content.title} - Check our recommendations!"
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500
@app.route('/api/quick-search', methods=['GET'])
def quick_search():
    """Quick search with immediate OTT availability"""
    try:
        query = request.args.get('query', '')
        if not query:
            return jsonify({'error': 'Query required'}), 400
        
        # Search multiple sources quickly
        results = []
        
        # Search TMDB
        tmdb_results = TMDBService.search_content(query, 'multi', page=1)
        if tmdb_results:
            for item in tmdb_results.get('results', [])[:5]:  # Limit to 5 results
                content_type = 'movie' if 'title' in item else 'tv'
                title = item.get('title') or item.get('name', '')
                
                # Get quick OTT info
                encoded_title = urllib.parse.quote(title.replace(' ', '+'))
                quick_platforms = [
                    {
                        'name': 'YouTube',
                        'direct_url': f"https://www.youtube.com/results?search_query={encoded_title}+full+movie",
                        'is_free': True
                    },
                    {
                        'name': 'Netflix',
                        'direct_url': f"https://www.netflix.com/search?q={encoded_title}",
                        'is_free': False
                    },
                    {
                        'name': 'Prime Video',
                        'direct_url': f"https://www.primevideo.com/search/ref=atv_nb_sr?phrase={encoded_title}",
                        'is_free': False
                    }
                ]
                
                results.append({
                    'title': title,
                    'content_type': content_type,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{item['poster_path']}" if item.get('poster_path') else None,
                    'rating': item.get('vote_average'),
                    'release_date': item.get('release_date') or item.get('first_air_date'),
                    'overview': item.get('overview', '')[:150] + '...',
                    'quick_watch_links': quick_platforms,
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
                fresh_ott = ContentService.get_ott_availability(tmdb_details, content.imdb_id)
                
                # Update content with fresh data
                content.ott_platforms = json.dumps(fresh_ott)
                content.updated_at = datetime.utcnow()
                db.session.commit()
                
                # Group platforms
                free_platforms = [p for p in fresh_ott if p.get('is_free')]
                paid_platforms = [p for p in fresh_ott if not p.get('is_free')]
                
                return jsonify({
                    'content_title': content.title,
                    'last_updated': content.updated_at.isoformat(),
                    'free_platforms': free_platforms,
                    'paid_platforms': paid_platforms,
                    'total_platforms': len(fresh_ott),
                    'availability_summary': {
                        'has_free_options': len(free_platforms) > 0,
                        'free_count': len(free_platforms),
                        'paid_count': len(paid_platforms)
                    }
                }), 200
        
        return jsonify({'error': 'No OTT data available'}), 404
        
    except Exception as e:
        logger.error(f"OTT availability error: {e}")
        return jsonify({'error': 'Failed to get OTT availability'}), 500

# Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = []
        for content in recommendations:
            # Parse OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            # Group platforms
            free_platforms = [p for p in ott_platforms if p.get('is_free')]
            paid_platforms = [p for p in ott_platforms if not p.get('is_free')]
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': {
                    'free_platforms': free_platforms,
                    'paid_platforms': paid_platforms,
                    'total_platforms': len(ott_platforms)
                }
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
            # Parse OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            # Group platforms
            free_platforms = [p for p in ott_platforms if p.get('is_free')]
            paid_platforms = [p for p in ott_platforms if not p.get('is_free')]
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': {
                    'free_platforms': free_platforms,
                    'paid_platforms': paid_platforms,
                    'total_platforms': len(ott_platforms)
                }
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
            # Parse OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            # Group platforms
            free_platforms = [p for p in ott_platforms if p.get('is_free')]
            paid_platforms = [p for p in ott_platforms if not p.get('is_free')]
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': {
                    'free_platforms': free_platforms,
                    'paid_platforms': paid_platforms,
                    'total_platforms': len(ott_platforms)
                }
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
            # Parse OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            # Group platforms
            free_platforms = [p for p in ott_platforms if p.get('is_free')]
            paid_platforms = [p for p in ott_platforms if not p.get('is_free')]
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': {
                    'free_platforms': free_platforms,
                    'paid_platforms': paid_platforms,
                    'total_platforms': len(ott_platforms)
                }
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
                        # Parse OTT platforms
                        ott_platforms = []
                        if content.ott_platforms:
                            try:
                                ott_platforms = json.loads(content.ott_platforms)
                            except:
                                ott_platforms = []
                        
                        # Group platforms
                        free_platforms = [p for p in ott_platforms if p.get('is_free')]
                        paid_platforms = [p for p in ott_platforms if not p.get('is_free')]
                        
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'ott_platforms': {
                                'free_platforms': free_platforms,
                                'paid_platforms': paid_platforms,
                                'total_platforms': len(ott_platforms)
                            },
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
            # Parse OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': ott_platforms
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
            # Parse OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': ott_platforms
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
                # Parse OTT platforms
                ott_platforms = []
                if content.ott_platforms:
                    try:
                        ott_platforms = json.loads(content.ott_platforms)
                    except:
                        ott_platforms = []
                
                # Group platforms
                free_platforms = [p for p in ott_platforms if p.get('is_free')]
                paid_platforms = [p for p in ott_platforms if not p.get('is_free')]
                
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'ott_platforms': {
                        'free_platforms': free_platforms,
                        'paid_platforms': paid_platforms,
                        'total_platforms': len(ott_platforms)
                    },
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
        'version': '1.0.0',
        'features': {
            'ott_integration': True,
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