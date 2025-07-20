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
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    content_type = db.Column(db.String(20), nullable=False)  # movie, tv, anime
    genres = db.Column(db.Text)  # JSON string
    languages = db.Column(db.Text)  # JSON string
    audio_languages = db.Column(db.Text)  # JSON string - Available audio languages
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
    streaming_info = db.Column(db.Text)  # JSON string with language-specific streaming links
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
    'prime': {'name': 'Amazon Prime Video', 'is_free': False, 'url': 'https://primevideo.com'},
    'hotstar': {'name': 'Disney+ Hotstar', 'is_free': False, 'url': 'https://hotstar.com'},
    'youtube': {'name': 'YouTube', 'is_free': True, 'url': 'https://youtube.com'},
    'jiocinema': {'name': 'JioCinema', 'is_free': True, 'url': 'https://jiocinema.com'},
    'mxplayer': {'name': 'MX Player', 'is_free': True, 'url': 'https://mxplayer.in'},
    'zee5': {'name': 'ZEE5', 'is_free': False, 'url': 'https://zee5.com'},
    'sonyliv': {'name': 'SonyLIV', 'is_free': False, 'url': 'https://sonyliv.com'},
    'voot': {'name': 'Voot', 'is_free': True, 'url': 'https://voot.com'},
    'altbalaji': {'name': 'ALTBalaji', 'is_free': False, 'url': 'https://altbalaji.com'},
    'aha': {'name': 'Aha', 'is_free': False, 'url': 'https://aha.video'},
    'sunnxt': {'name': 'Sun NXT', 'is_free': False, 'url': 'https://sunnxt.com'},
    'crunchyroll': {'name': 'Crunchyroll', 'is_free': True, 'url': 'https://crunchyroll.com'},
    'airtelxstream': {'name': 'Airtel Xstream', 'is_free': True, 'url': 'https://airtelxstream.in'}
}

# Regional Language Mapping
REGIONAL_LANGUAGES = {
    'hindi': ['hi', 'hindi', 'bollywood'],
    'telugu': ['te', 'telugu', 'tollywood'],
    'tamil': ['ta', 'tamil', 'kollywood'],
    'kannada': ['kn', 'kannada', 'sandalwood'],
    'malayalam': ['ml', 'malayalam', 'mollywood'],
    'english': ['en', 'english', 'hollywood']
}

# Free and Premium Platform Classification
FREE_PLATFORMS = ['mxplayer', 'jiocinema', 'youtube', 'crunchyroll', 'airtelxstream']
PREMIUM_PLATFORMS = ['netflix', 'prime', 'hotstar', 'zee5', 'sonyliv', 'aha', 'sunnxt']

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
    def get_new_releases(content_type='movie', page=1, region=None):
        # For movies, use now_playing. For TV shows, use on_the_air
        endpoint = 'now_playing' if content_type == 'movie' else 'on_the_air'
        url = f"{TMDBService.BASE_URL}/{content_type}/{endpoint}"
        
        current_date = datetime.now().date()
        date_from = (current_date - timedelta(days=60)).strftime('%Y-%m-%d')
        
        params = {
            'api_key': TMDB_API_KEY,
            'page': page,
            'primary_release_date.gte': date_from
        }
        if region:
            params['region'] = region
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB new releases error: {e}")
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
    def get_content_by_genre(genre_id, content_type='movie', page=1):
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': genre_id,
            'page': page,
            'sort_by': 'popularity.desc'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB genre discover error: {e}")
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

class WatchModeService:
    BASE_URL = 'https://api.watchmode.com/v1'
    
    @staticmethod
    def search_content(query):
        url = f"{WatchModeService.BASE_URL}/search"
        params = {
            'apiKey': WATCHMODE_API_KEY,
            'search_field': 'name',
            'search_value': query
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"WatchMode search error: {e}")
        return None
    
    @staticmethod
    def get_title_details(title_id):
        url = f"{WatchModeService.BASE_URL}/title/{title_id}/details"
        params = {
            'apiKey': WATCHMODE_API_KEY,
            'append_to_response': 'sources'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"WatchMode title details error: {e}")
        return None
    
    @staticmethod
    def get_title_sources(title_id):
        url = f"{WatchModeService.BASE_URL}/title/{title_id}/sources"
        params = {
            'apiKey': WATCHMODE_API_KEY,
            'regions': 'IN'  # Focus on Indian streaming services
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"WatchMode sources error: {e}")
        return None

class StreamingAvailabilityService:
    BASE_URL = 'https://streaming-availability.p.rapidapi.com/v2'
    
    @staticmethod
    def get_by_tmdb_id(tmdb_id, content_type='movie'):
        url = f"{StreamingAvailabilityService.BASE_URL}/get/tmdb"
        
        headers = {
            'x-rapidapi-key': "c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37",
            'x-rapidapi-host': "streaming-availability.p.rapidapi.com"
        }
        
        params = {
            'tmdb_id': f"{content_type}/{tmdb_id}",
            'country': 'in'  # Focus on Indian streaming services
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Streaming Availability error: {e}")
        return None
    
    @staticmethod
    def search(title, content_type=None):
        url = f"{StreamingAvailabilityService.BASE_URL}/search/title"
        
        headers = {
            'x-rapidapi-key': "c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37",
            'x-rapidapi-host': "streaming-availability.p.rapidapi.com"
        }
        
        params = {
            'title': title,
            'country': 'in',
            'show_type': content_type if content_type else 'all'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Streaming Availability search error: {e}")
        return None

# Content Management Service
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            # Check if content already exists
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                # Update streaming info if it doesn't exist or is outdated
                if not existing.streaming_info or (existing.updated_at < datetime.utcnow() - timedelta(days=7)):
                    streaming_info = ContentService.get_streaming_availability(existing.tmdb_id, content_type, existing.title)
                    if streaming_info:
                        existing.streaming_info = json.dumps(streaming_info)
                        existing.updated_at = datetime.utcnow()
                        db.session.commit()
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
                languages = [lang['name'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
            
            # Get IMDB ID if available
            imdb_id = None
            if 'external_ids' in tmdb_data:
                imdb_id = tmdb_data['external_ids'].get('imdb_id')
            
            # Get streaming availability
            streaming_info = ContentService.get_streaming_availability(tmdb_data['id'], content_type, 
                                                                    tmdb_data.get('title') or tmdb_data.get('name'))
            
            # Determine available audio languages
            audio_languages = ContentService.extract_audio_languages(streaming_info)
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
                imdb_id=imdb_id,
                title=tmdb_data.get('title') or tmdb_data.get('name'),
                original_title=tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                content_type=content_type,
                genres=json.dumps(genres),
                languages=json.dumps(languages),
                audio_languages=json.dumps(audio_languages),
                release_date=datetime.strptime(tmdb_data.get('release_date') or tmdb_data.get('first_air_date', '1900-01-01'), '%Y-%m-%d').date() if tmdb_data.get('release_date') or tmdb_data.get('first_air_date') else None,
                runtime=tmdb_data.get('runtime'),
                rating=tmdb_data.get('vote_average'),
                vote_count=tmdb_data.get('vote_count'),
                popularity=tmdb_data.get('popularity'),
                overview=tmdb_data.get('overview'),
                poster_path=tmdb_data.get('poster_path'),
                backdrop_path=tmdb_data.get('backdrop_path'),
                streaming_info=json.dumps(streaming_info) if streaming_info else None
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
        try:
            # Check if anime already exists
            anime_id = f"anime_{anime_data['mal_id']}"
            existing = Content.query.filter_by(tmdb_id=anime_id).first()
            if existing:
                return existing
            
            # Extract genres
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            # Get streaming availability for anime
            streaming_info = ContentService.get_anime_streaming(anime_data.get('title'))
            
            # Determine available audio languages for anime
            audio_languages = ContentService.extract_audio_languages(streaming_info)
            if not audio_languages:
                audio_languages = ['japanese']  # Default for anime
                
                # Check if English dub is mentioned in title or synopsis
                title = anime_data.get('title', '').lower()
                synopsis = anime_data.get('synopsis', '').lower()
                if 'dub' in title or 'english dub' in synopsis:
                    audio_languages.append('english')
            
            # Create content object for anime
            content = Content(
                tmdb_id=anime_id,
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                languages=json.dumps(['japanese']),
                audio_languages=json.dumps(audio_languages),
                release_date=datetime.strptime(anime_data.get('aired', {}).get('from', '1900-01-01T00:00:00+00:00')[:10], '%Y-%m-%d').date() if anime_data.get('aired', {}).get('from') else None,
                runtime=anime_data.get('duration', 0),
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('large_image_url'),
                streaming_info=json.dumps(streaming_info) if streaming_info else None
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
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]
    
    @staticmethod
    def get_streaming_availability(tmdb_id, content_type, title=None):
        # First try using Streaming Availability API with TMDB ID
        streaming_data = StreamingAvailabilityService.get_by_tmdb_id(tmdb_id, content_type)
        
        # If that fails, try using the title
        if not streaming_data and title:
            streaming_data = StreamingAvailabilityService.search(title, content_type)
        
        # If still no data, try WatchMode API
        if not streaming_data and title:
            watchmode_search = WatchModeService.search_content(title)
            if watchmode_search and watchmode_search.get('titles'):
                watchmode_id = watchmode_search['titles'][0]['id']
                watchmode_data = WatchModeService.get_title_sources(watchmode_id)
                
                if watchmode_data:
                    return ContentService.process_watchmode_data(watchmode_data)
        
        if streaming_data:
            return ContentService.process_streaming_data(streaming_data)
        
        # If no data found, return default structure
        return ContentService.generate_default_streaming_info(title, content_type)
    
    @staticmethod
    def process_streaming_data(streaming_data):
        result = {
            'free': [],
            'subscription': [],
            'rent': [],
            'buy': [],
            'by_language': {}
        }
        
        # Default language mappings if not provided by API
        default_langs = ['hindi', 'english', 'telugu', 'tamil', 'malayalam', 'kannada']
        
        try:
            if 'result' in streaming_data:
                streaming_info = streaming_data['result']
                
                # Process streaming services from API response
                if 'streamingInfo' in streaming_info and 'in' in streaming_info['streamingInfo']:
                    services = streaming_info['streamingInfo']['in']
                    
                    for service_name, service_data in services.items():
                        service_info = {
                            'name': service_name.capitalize(),
                            'logo': f"https://www.google.com/s2/favicons?domain={service_name}.com&sz=64",
                            'url': service_data[0].get('link', f"https://{service_name}.com")
                        }
                        
                        # Categorize by service type
                        if service_name in FREE_PLATFORMS:
                            result['free'].append(service_info)
                        else:
                            result['subscription'].append(service_info)
                        
                        # Process language-specific links if available
                        for entry in service_data:
                            if 'audios' in entry:
                                for lang in entry['audios']:
                                    lang_code = lang.lower()
                                    
                                    # Map language codes to full names
                                    for full_lang, codes in REGIONAL_LANGUAGES.items():
                                        if lang_code in codes:
                                            lang_code = full_lang
                                            break
                                    
                                    if lang_code not in result['by_language']:
                                        result['by_language'][lang_code] = []
                                    
                                    # Add link with specific language
                                    lang_service = service_info.copy()
                                    lang_service['url'] = entry.get('link', lang_service['url'])
                                    result['by_language'][lang_code].append(lang_service)
                
                # If no language-specific data is available, use imdbInfo for educated guesses
                if not result['by_language'] and 'imdbInfo' in streaming_info:
                    countries = streaming_info.get('imdbInfo', {}).get('countries', [])
                    languages = streaming_info.get('imdbInfo', {}).get('languages', [])
                    
                    # If it's an Indian movie, assume it has the primary language
                    is_indian = any(country == 'India' for country in countries)
                    
                    if is_indian and languages:
                        primary_lang = languages[0].lower()
                        # Map from ISO codes to our language keys
                        mapped_lang = None
                        for lang_key, lang_codes in REGIONAL_LANGUAGES.items():
                            if primary_lang in lang_codes:
                                mapped_lang = lang_key
                                break
                        
                        if mapped_lang:
                            result['by_language'][mapped_lang] = result['subscription'] + result['free']
                    
                    # For all movies, assume English is available
                    if 'english' not in result['by_language'] and (result['subscription'] or result['free']):
                        result['by_language']['english'] = result['subscription'] + result['free']
        except Exception as e:
            logger.error(f"Error processing streaming data: {e}")
        
        # Ensure we have at least some language categories
        if not result['by_language']:
            # If we have any streaming services, assume they have content in these languages
            if result['subscription'] or result['free']:
                services = result['subscription'] + result['free']
                for lang in default_langs:
                    result['by_language'][lang] = services
        
        return result
    
    @staticmethod
    def process_watchmode_data(watchmode_data):
        result = {
            'free': [],
            'subscription': [],
            'rent': [],
            'buy': [],
            'by_language': {}
        }
        
        try:
            for source in watchmode_data:
                service_name = source.get('name', '').lower().replace(' ', '')
                
                # Map WatchMode service names to our platform keys
                platform_key = None
                for key in OTT_PLATFORMS.keys():
                    if key in service_name:
                        platform_key = key
                        break
                
                if not platform_key:
                    continue
                
                service_info = {
                    'name': OTT_PLATFORMS[platform_key]['name'],
                    'logo': f"https://www.google.com/s2/favicons?domain={platform_key}.com&sz=64",
                    'url': source.get('web_url') or OTT_PLATFORMS[platform_key]['url']
                }
                
                # Categorize by service type
                if platform_key in FREE_PLATFORMS:
                    result['free'].append(service_info)
                else:
                    result['subscription'].append(service_info)
                
                # Add to rent/buy categories if applicable
                if source.get('type') == 'rent':
                    result['rent'].append(service_info)
                elif source.get('type') == 'buy':
                    result['buy'].append(service_info)
            
            # Assume all services have content in these languages
            # WatchMode doesn't provide language-specific info, so we make educated guesses
            services = result['subscription'] + result['free']
            if services:
                for lang in ['english', 'hindi']:
                    result['by_language'][lang] = services
                
        except Exception as e:
            logger.error(f"Error processing WatchMode data: {e}")
        
        return result
    
    @staticmethod
    def get_anime_streaming(title):
        # Anime streaming services are more specific
        result = {
            'free': [],
            'subscription': [],
            'by_language': {
                'japanese': [],
                'english': []
            }
        }
        
        # Common anime streaming services
        anime_services = [
            {
                'name': 'Crunchyroll',
                'key': 'crunchyroll',
                'url': f"https://www.crunchyroll.com/search?q={title.replace(' ', '+')}"
            },
            {
                'name': 'Netflix',
                'key': 'netflix',
                'url': f"https://www.netflix.com/search?q={title.replace(' ', '+')}"
            },
            {
                'name': 'Amazon Prime Video',
                'key': 'prime',
                'url': f"https://www.primevideo.com/search?k={title.replace(' ', '+')}"
            }
        ]
        
        # Add services to the result
        for service in anime_services:
            service_info = {
                'name': service['name'],
                'logo': f"https://www.google.com/s2/favicons?domain={service['key']}.com&sz=64",
                'url': service['url']
            }
            
            if service['key'] in FREE_PLATFORMS:
                result['free'].append(service_info)
            else:
                result['subscription'].append(service_info)
            
            # Add to both language categories
            result['by_language']['japanese'].append(service_info)
            
            # Most anime on these platforms also have English dubs
            if service['key'] in ['netflix', 'prime', 'crunchyroll']:
                result['by_language']['english'].append(service_info)
        
        return result
    
    @staticmethod
    def generate_default_streaming_info(title, content_type):
        # Generate default streaming info based on content type and title
        result = {
            'free': [],
            'subscription': [],
            'by_language': {}
        }
        
        # Determine likely platforms based on content type
        if content_type == 'movie':
            # For movies, include major platforms
            platforms = ['netflix', 'prime', 'hotstar', 'jiocinema', 'zee5', 'mxplayer']
        elif content_type == 'tv':
            # For TV shows, include major streaming platforms
            platforms = ['netflix', 'prime', 'hotstar', 'sonyliv', 'zee5']
        elif content_type == 'anime':
            # For anime, include anime-specific platforms
            platforms = ['crunchyroll', 'netflix', 'prime']
            
            # Default anime languages
            result['by_language']['japanese'] = []
            result['by_language']['english'] = []
        else:
            platforms = ['netflix', 'prime', 'hotstar']
        
        # Add platforms to the result
        for platform in platforms:
            if platform in OTT_PLATFORMS:
                service_info = {
                    'name': OTT_PLATFORMS[platform]['name'],
                    'logo': f"https://www.google.com/s2/favicons?domain={platform}.com&sz=64",
                    'url': f"{OTT_PLATFORMS[platform]['url']}/search?q={title.replace(' ', '+')}"
                }
                
                if platform in FREE_PLATFORMS:
                    result['free'].append(service_info)
                else:
                    result['subscription'].append(service_info)
                
                # For anime, add to language categories
                if content_type == 'anime':
                    result['by_language']['japanese'].append(service_info)
                    if platform in ['netflix', 'prime', 'crunchyroll']:
                        result['by_language']['english'].append(service_info)
        
        # For regular content, add default languages
        if content_type != 'anime':
            services = result['subscription'] + result['free']
            if services:
                for lang in ['english', 'hindi']:
                    result['by_language'][lang] = services
        
        return result
    
    @staticmethod
    def extract_audio_languages(streaming_info):
        if not streaming_info or not isinstance(streaming_info, dict):
            return ['english']  # Default fallback
        
        # Extract languages from streaming info
        languages = []
        if 'by_language' in streaming_info:
            languages = list(streaming_info['by_language'].keys())
        
        # Ensure we have at least one language
        if not languages:
            languages = ['english']
        
        return languages

# Recommendation Engine
class RecommendationEngine:
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all', languages=None):
        try:
            # Get trending from TMDB
            trending_data = TMDBService.get_trending(content_type=content_type)
            if not trending_data:
                return []
            
            recommendations = []
            for item in trending_data.get('results', [])[:limit * 2]:  # Get more items to filter by language
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    recommendations.append(content)
            
            # Filter by language preference if specified
            if languages:
                filtered_recommendations = []
                for content in recommendations:
                    content_languages = json.loads(content.audio_languages or '["english"]')
                    # Check if any preferred language is available
                    if any(lang in content_languages for lang in languages):
                        filtered_recommendations.append(content)
                
                # If we have enough filtered recommendations, use those
                if len(filtered_recommendations) >= limit // 2:
                    recommendations = filtered_recommendations
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_new_releases(limit=20, content_type='movie', languages=None):
        try:
            # Get new releases from TMDB
            new_releases = TMDBService.get_new_releases(content_type)
            if not new_releases:
                return []
            
            recommendations = []
            for item in new_releases.get('results', [])[:limit * 2]:
                content = ContentService.save_content_from_tmdb(item, content_type)
                if content:
                    recommendations.append(content)
            
            # Filter by language preference if specified
            if languages:
                filtered_recommendations = []
                for content in recommendations:
                    content_languages = json.loads(content.audio_languages or '["english"]')
                    # Check if any preferred language is available
                    if any(lang in content_languages for lang in languages):
                        filtered_recommendations.append(content)
                
                # If we have enough filtered recommendations, use those
                if len(filtered_recommendations) >= limit // 2:
                    recommendations = filtered_recommendations
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return []
    
    @staticmethod
    def get_best_movies(limit=20, languages=None):
        try:
            # Get top rated movies from TMDB
            top_rated = TMDBService.get_top_rated('movie')
            if not top_rated:
                return []
            
            recommendations = []
            for item in top_rated.get('results', [])[:limit * 2]:
                content = ContentService.save_content_from_tmdb(item, 'movie')
                if content:
                    recommendations.append(content)
            
            # Filter by language preference if specified
            if languages:
                filtered_recommendations = []
                for content in recommendations:
                    content_languages = json.loads(content.audio_languages or '["english"]')
                    # Check if any preferred language is available
                    if any(lang in content_languages for lang in languages):
                        filtered_recommendations.append(content)
                
                # If we have enough filtered recommendations, use those
                if len(filtered_recommendations) >= limit // 2:
                    recommendations = filtered_recommendations
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting best movies: {e}")
            return []
    
    @staticmethod
    def get_critics_choice(limit=20, languages=None):
        try:
            # This is similar to top rated but we can further filter by vote count
            # to ensure they're truly critically acclaimed
            url = f"{TMDBService.BASE_URL}/discover/movie"
            params = {
                'api_key': TMDB_API_KEY,
                'sort_by': 'vote_average.desc',
                'vote_count.gte': 1000,  # Minimum vote count for credibility
                'page': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return []
            
            critics_data = response.json()
            
            recommendations = []
            for item in critics_data.get('results', [])[:limit * 2]:
                content = ContentService.save_content_from_tmdb(item, 'movie')
                if content:
                    recommendations.append(content)
            
            # Filter by language preference if specified
            if languages:
                filtered_recommendations = []
                for content in recommendations:
                    content_languages = json.loads(content.audio_languages or '["english"]')
                    # Check if any preferred language is available
                    if any(lang in content_languages for lang in languages):
                        filtered_recommendations.append(content)
                
                # If we have enough filtered recommendations, use those
                if len(filtered_recommendations) >= limit // 2:
                    recommendations = filtered_recommendations
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting critics choice: {e}")
            return []
    
    @staticmethod
    def get_genre_recommendations(genre, limit=20, languages=None):
        try:
            # Map genre name to ID
            genre_map = {
                'action': 28, 'adventure': 12, 'animation': 16, 'biography': 99,
                'comedy': 35, 'crime': 80, 'documentary': 99, 'drama': 18,
                'fantasy': 14, 'horror': 27, 'musical': 10402, 'mystery': 9648,
                'romance': 10749, 'sci-fi': 878, 'thriller': 53, 'western': 37
            }
            
            genre_id = genre_map.get(genre.lower())
            if not genre_id:
                return []
            
            # Get genre recommendations from TMDB
            genre_data = TMDBService.get_content_by_genre(genre_id)
            if not genre_data:
                return []
            
            recommendations = []
            for item in genre_data.get('results', [])[:limit * 2]:
                content = ContentService.save_content_from_tmdb(item, 'movie')
                if content:
                    recommendations.append(content)
            
            # Filter by language preference if specified
            if languages:
                filtered_recommendations = []
                for content in recommendations:
                    content_languages = json.loads(content.audio_languages or '["english"]')
                    # Check if any preferred language is available
                    if any(lang in content_languages for lang in languages):
                        filtered_recommendations.append(content)
                
                # If we have enough filtered recommendations, use those
                if len(filtered_recommendations) >= limit // 2:
                    recommendations = filtered_recommendations
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting genre recommendations: {e}")
            return []
    
    @staticmethod
    def get_regional_recommendations(language, limit=20):
        try:
            # Search for content in specific language
            search_queries = {
                'hindi': ['bollywood', 'hindi movie', 'hindi film'],
                'telugu': ['tollywood', 'telugu movie', 'telugu film'],
                'tamil': ['kollywood', 'tamil movie', 'tamil film'],
                'kannada': ['sandalwood', 'kannada movie', 'kannada film'],
                'malayalam': ['malayalam movie', 'malayalam film'],
                'english': ['hollywood', 'english movie', 'english film']
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
                            # Filter by audio language availability
                            audio_languages = json.loads(content.audio_languages or '["english"]')
                            if language in audio_languages:
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
                content = ContentService.save_anime_from_jikan(anime)
                if content:
                    recommendations.append(content)
            
            return recommendations[:limit]
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
            
            # Add regional content based on location - prioritize Telugu and English
            priority_languages = ['telugu', 'english', 'hindi', 'tamil', 'malayalam', 'kannada']
            
            if location and location.get('country') == 'India':
                # Prioritize Telugu content
                telugu_recs = RecommendationEngine.get_regional_recommendations('telugu', limit=5)
                recommendations.extend(telugu_recs)
                
                # Add some Hindi content too
                hindi_recs = RecommendationEngine.get_regional_recommendations('hindi', limit=3)
                recommendations.extend(hindi_recs)
            else:
                # For non-Indian users, prioritize English content
                english_recs = RecommendationEngine.get_regional_recommendations('english', limit=5)
                recommendations.extend(english_recs)
            
            # Add trending content with priority for Telugu and English
            trending_recs = RecommendationEngine.get_trending_recommendations(limit=10, languages=priority_languages)
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
            
            # Get streaming links by language
            streaming_links = ""
            if content.streaming_info:
                try:
                    streaming_data = json.loads(content.streaming_info)
                    if 'by_language' in streaming_data:
                        streaming_links = "\n🎬 *Watch Now:*\n"
                        for language, services in streaming_data['by_language'].items():
                            if services:
                                # Only include the first service for each language to keep message clean
                                service = services[0]
                                streaming_links += f"• Watch in {language.capitalize()}: {service['url']}\n"
                except:
                    pass
            
            # Create message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}
{streaming_links}
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
                    
                    # Get streaming info
                    streaming_data = {}
                    if content.streaming_info:
                        try:
                            streaming_data = json.loads(content.streaming_info)
                        except:
                            streaming_data = {}
                    
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
                        'streaming_info': streaming_data,
                        'audio_languages': json.loads(content.audio_languages or '["english"]')
                    })
        
        # Add anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                # Process anime data
                anime_id = f"anime_{anime['mal_id']}"
                existing_anime = Content.query.filter_by(tmdb_id=anime_id).first()
                
                if existing_anime:
                    anime_content = existing_anime
                else:
                    anime_content = ContentService.save_anime_from_jikan(anime)
                
                if anime_content:
                    # Record anonymous interaction
                    interaction = AnonymousInteraction(
                        session_id=session_id,
                        content_id=anime_content.id,
                        interaction_type='search',
                        ip_address=request.remote_addr
                    )
                    db.session.add(interaction)
                    
                    # Get streaming info
                    streaming_data = {}
                    if anime_content.streaming_info:
                        try:
                            streaming_data = json.loads(anime_content.streaming_info)
                        except:
                            streaming_data = {}
                    
                    results.append({
                        'id': anime_content.id,
                        'tmdb_id': anime_content.tmdb_id,
                        'title': anime_content.title,
                        'content_type': 'anime',
                        'genres': json.loads(anime_content.genres or '[]'),
                        'rating': anime_content.rating,
                        'release_date': anime_content.release_date.isoformat() if anime_content.release_date else None,
                        'poster_path': anime_content.poster_path,
                        'overview': anime_content.overview,
                        'streaming_info': streaming_data,
                        'audio_languages': json.loads(anime_content.audio_languages or '["japanese", "english"]')
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
        if content.tmdb_id and not content.tmdb_id.startswith('anime_'):
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        elif content.tmdb_id and content.tmdb_id.startswith('anime_'):
            # Get anime details
            anime_id = content.tmdb_id.replace('anime_', '')
            additional_details = JikanService.get_anime_details(anime_id)
        
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
        
        # Get streaming information
        streaming_info = None
        if content.streaming_info:
            try:
                streaming_info = json.loads(content.streaming_info)
            except:
                pass
        
        # If streaming info is missing or outdated, refresh it
        if not streaming_info or content.updated_at < datetime.utcnow() - timedelta(days=7):
            streaming_info = ContentService.get_streaming_availability(
                content.tmdb_id, content.content_type, content.title)
            
            if streaming_info:
                content.streaming_info = json.dumps(streaming_info)
                content.updated_at = datetime.utcnow()
                db.session.commit()
        
        # Get similar content
        similar_content = []
        if additional_details:
            if content.content_type != 'anime' and 'similar' in additional_details:
                for item in additional_details['similar']['results'][:5]:
                    similar = ContentService.save_content_from_tmdb(item, content.content_type)
                    if similar:
                        similar_content.append({
                            'id': similar.id,
                            'title': similar.title,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{similar.poster_path}" if similar.poster_path else None,
                            'rating': similar.rating
                        })
            elif content.content_type == 'anime' and 'recommendations' in additional_details.get('data', {}):
                for item in additional_details['data']['recommendations'][:5]:
                    rec_anime = JikanService.get_anime_details(item['entry']['mal_id'])
                    if rec_anime and 'data' in rec_anime:
                        similar = ContentService.save_anime_from_jikan(rec_anime['data'])
                        if similar:
                            similar_content.append({
                                'id': similar.id,
                                'title': similar.title,
                                'poster_path': similar.poster_path,
                                'rating': similar.rating
                            })
        
        # Format audio languages for display
        audio_languages = json.loads(content.audio_languages or '["english"]')
        
        db.session.commit()
        
        return jsonify({
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'imdb_id': content.imdb_id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'audio_languages': audio_languages,
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'runtime': content.runtime,
            'rating': content.rating,
            'vote_count': content.vote_count,
            'overview': content.overview,
            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path and not content.backdrop_path.startswith('http') else content.backdrop_path,
            'streaming_info': streaming_info or {'free': [], 'subscription': [], 'by_language': {}},
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details and content.content_type != 'anime' else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details and content.content_type != 'anime' else [],
            'voice_actors': additional_details.get('data', {}).get('characters', [])[:10] if additional_details and content.content_type == 'anime' else []
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
        language_pref = request.args.get('language')
        
        # Set language preferences, prioritizing Telugu and English
        languages = ['telugu', 'english']
        if language_pref and language_pref not in languages:
            languages.insert(0, language_pref)
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type, languages)
        
        result = []
        for content in recommendations:
            # Get streaming info
            streaming_info = {}
            if content.streaming_info:
                try:
                    streaming_info = json.loads(content.streaming_info)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'audio_languages': json.loads(content.audio_languages or '["english"]'),
                'streaming_info': streaming_info
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/recommendations/new-releases', methods=['GET'])
def get_new_releases():
    try:
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        language_pref = request.args.get('language')
        
        # Set language preferences, prioritizing Telugu and English
        languages = ['telugu', 'english']
        if language_pref and language_pref not in languages:
            languages.insert(0, language_pref)
        
        recommendations = RecommendationEngine.get_new_releases(limit, content_type, languages)
        
        result = []
        for content in recommendations:
            # Get streaming info
            streaming_info = {}
            if content.streaming_info:
                try:
                    streaming_info = json.loads(content.streaming_info)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'audio_languages': json.loads(content.audio_languages or '["english"]'),
                'streaming_info': streaming_info
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"New releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

@app.route('/api/recommendations/best-movies', methods=['GET'])
def get_best_movies():
    try:
        limit = int(request.args.get('limit', 20))
        language_pref = request.args.get('language')
        
        # Set language preferences, prioritizing Telugu and English
        languages = ['telugu', 'english']
        if language_pref and language_pref not in languages:
            languages.insert(0, language_pref)
        
        recommendations = RecommendationEngine.get_best_movies(limit, languages)
        
        result = []
        for content in recommendations:
            # Get streaming info
            streaming_info = {}
            if content.streaming_info:
                try:
                    streaming_info = json.loads(content.streaming_info)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'audio_languages': json.loads(content.audio_languages or '["english"]'),
                'streaming_info': streaming_info
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Best movies error: {e}")
        return jsonify({'error': 'Failed to get best movies'}), 500

@app.route('/api/recommendations/critics-choice', methods=['GET'])
def get_critics_choice():
    try:
        limit = int(request.args.get('limit', 20))
        language_pref = request.args.get('language')
        
        # Set language preferences, prioritizing Telugu and English
        languages = ['telugu', 'english']
        if language_pref and language_pref not in languages:
            languages.insert(0, language_pref)
        
        recommendations = RecommendationEngine.get_critics_choice(limit, languages)
        
        result = []
        for content in recommendations:
            # Get streaming info
            streaming_info = {}
            if content.streaming_info:
                try:
                    streaming_info = json.loads(content.streaming_info)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'audio_languages': json.loads(content.audio_languages or '["english"]'),
                'streaming_info': streaming_info
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Critics choice error: {e}")
        return jsonify({'error': 'Failed to get critics choice recommendations'}), 500

@app.route('/api/recommendations/genre/<genre>', methods=['GET'])
def get_genre_recommendations(genre):
    try:
        limit = int(request.args.get('limit', 20))
        language_pref = request.args.get('language')
        
        # Set language preferences, prioritizing Telugu and English
        languages = ['telugu', 'english']
        if language_pref and language_pref not in languages:
            languages.insert(0, language_pref)
        
        recommendations = RecommendationEngine.get_genre_recommendations(genre, limit, languages)
        
        result = []
        for content in recommendations:
            # Get streaming info
            streaming_info = {}
            if content.streaming_info:
                try:
                    streaming_info = json.loads(content.streaming_info)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'audio_languages': json.loads(content.audio_languages or '["english"]'),
                'streaming_info': streaming_info
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Genre recommendations error: {e}")
        return jsonify({'error': 'Failed to get genre recommendations'}), 500

@app.route('/api/recommendations/regional/<language>', methods=['GET'])
def get_regional(language):
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_regional_recommendations(language, limit)
        
        result = []
        for content in recommendations:
            # Get streaming info
            streaming_info = {}
            if content.streaming_info:
                try:
                    streaming_info = json.loads(content.streaming_info)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'audio_languages': json.loads(content.audio_languages or '["english"]'),
                'streaming_info': streaming_info
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
            # Get streaming info
            streaming_info = {}
            if content.streaming_info:
                try:
                    streaming_info = json.loads(content.streaming_info)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'audio_languages': json.loads(content.audio_languages or '["japanese", "english"]'),
                'streaming_info': streaming_info
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
            # Get streaming info
            streaming_info = {}
            if content.streaming_info:
                try:
                    streaming_info = json.loads(content.streaming_info)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'audio_languages': json.loads(content.audio_languages or '["english"]'),
                'streaming_info': streaming_info
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
                        # Get streaming info
                        streaming_info = {}
                        if content.streaming_info:
                            try:
                                streaming_info = json.loads(content.streaming_info)
                            except:
                                pass
                        
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'audio_languages': json.loads(content.audio_languages or '["english"]'),
                            'streaming_info': streaming_info,
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
            # Get streaming info
            streaming_info = {}
            if content.streaming_info:
                try:
                    streaming_info = json.loads(content.streaming_info)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'audio_languages': json.loads(content.audio_languages or '["english"]'),
                'streaming_info': streaming_info
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
            # Get streaming info
            streaming_info = {}
            if content.streaming_info:
                try:
                    streaming_info = json.loads(content.streaming_info)
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'audio_languages': json.loads(content.audio_languages or '["english"]'),
                'streaming_info': streaming_info
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
        
        # Check streaming availability for the results
        for result in results:
            if source == 'tmdb':
                streaming_data = StreamingAvailabilityService.get_by_tmdb_id(
                    result['id'], result['content_type'])
                
                if streaming_data and 'result' in streaming_data:
                    # Extract available audio languages
                    audio_languages = []
                    if 'streamingInfo' in streaming_data['result'] and 'in' in streaming_data['result']['streamingInfo']:
                        for service_name, service_data in streaming_data['result']['streamingInfo']['in'].items():
                            for entry in service_data:
                                if 'audios' in entry:
                                    for lang in entry['audios']:
                                        lang_code = lang.lower()
                                        
                                        # Map language codes to full names
                                        for full_lang, codes in REGIONAL_LANGUAGES.items():
                                            if lang_code in codes:
                                                if full_lang not in audio_languages:
                                                    audio_languages.append(full_lang)
                    
                    result['audio_languages'] = audio_languages if audio_languages else ['english']
        
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
            streaming_info = None
            if data.get('content_type') and data.get('id'):
                streaming_info = ContentService.get_streaming_availability(
                    data['id'], data['content_type'], data.get('title'))
            elif data.get('title'):
                streaming_info = ContentService.get_streaming_availability(
                    None, data.get('content_type', 'movie'), data.get('title'))
            
            # Determine available audio languages
            audio_languages = ContentService.extract_audio_languages(streaming_info)
            
            # Create content object
            content = Content(
                tmdb_id=data.get('id'),
                title=data.get('title'),
                original_title=data.get('original_title'),
                content_type=data.get('content_type', 'movie'),
                genres=json.dumps(data.get('genres', [])),
                languages=json.dumps(data.get('languages', ['en'])),
                audio_languages=json.dumps(audio_languages),
                release_date=release_date,
                runtime=data.get('runtime'),
                rating=data.get('rating'),
                vote_count=data.get('vote_count'),
                popularity=data.get('popularity'),
                overview=data.get('overview'),
                poster_path=data.get('poster_path'),
                backdrop_path=data.get('backdrop_path'),
                streaming_info=json.dumps(streaming_info) if streaming_info else None
            )
            
            db.session.add(content)
            db.session.commit()
            
            return jsonify({
                'message': 'Content saved successfully',
                'content_id': content.id,
                'streaming_info': streaming_info
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
            
            # Get streaming info
            streaming_info = {}
            if content and content.streaming_info:
                try:
                    streaming_info = json.loads(content.streaming_info)
                except:
                    pass
            
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
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'audio_languages': json.loads(content.audio_languages or '["english"]'),
                    'streaming_info': streaming_info
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
            if content and content.audio_languages:
                languages = json.loads(content.audio_languages)
                for lang in languages:
                    language_counts[lang] += 1
        
        popular_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:6]
        
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
                # Get streaming info
                streaming_info = {}
                if content.streaming_info:
                    try:
                        streaming_info = json.loads(content.streaming_info)
                    except:
                        pass
                
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'audio_languages': json.loads(content.audio_languages or '["english"]'),
                    'streaming_info': streaming_info,
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