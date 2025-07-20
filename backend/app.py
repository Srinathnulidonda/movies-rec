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

# Streaming API Keys
WATCHMODE_API_KEY = os.environ.get('WATCHMODE_API_KEY', 'WtcKDji9i20pjOl5Lg0AiyG2bddfUs3nSZRZJIsY')
RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY', 'c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37')
RAPIDAPI_HOST = os.environ.get('RAPIDAPI_HOST', 'streaming-availability.p.rapidapi.com')

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
    streaming_links = db.Column(db.Text)  # JSON string for streaming links with languages
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

# Enhanced OTT Platform Information with Indian platforms
OTT_PLATFORMS = {
    'netflix': {'name': 'Netflix', 'is_free': False, 'url': 'https://netflix.com', 'country': 'global'},
    'amazon_prime': {'name': 'Amazon Prime Video', 'is_free': False, 'url': 'https://primevideo.com', 'country': 'global'},
    'disney_plus': {'name': 'Disney+ Hotstar', 'is_free': False, 'url': 'https://hotstar.com', 'country': 'india'},
    'youtube': {'name': 'YouTube', 'is_free': True, 'url': 'https://youtube.com', 'country': 'global'},
    'jiocinema': {'name': 'JioCinema', 'is_free': True, 'url': 'https://jiocinema.com', 'country': 'india'},
    'mx_player': {'name': 'MX Player', 'is_free': True, 'url': 'https://mxplayer.in', 'country': 'india'},
    'zee5': {'name': 'ZEE5', 'is_free': False, 'url': 'https://zee5.com', 'country': 'india'},
    'sonyliv': {'name': 'SonyLIV', 'is_free': False, 'url': 'https://sonyliv.com', 'country': 'india'},
    'sonyliv_free': {'name': 'SonyLIV Free', 'is_free': True, 'url': 'https://sonyliv.com', 'country': 'india'},
    'voot': {'name': 'Voot', 'is_free': True, 'url': 'https://voot.com', 'country': 'india'},
    'alt_balaji': {'name': 'ALTBalaji', 'is_free': False, 'url': 'https://altbalaji.com', 'country': 'india'},
    'aha': {'name': 'Aha', 'is_free': False, 'url': 'https://aha.video', 'country': 'india'},
    'sun_nxt': {'name': 'Sun NXT', 'is_free': False, 'url': 'https://sunnxt.com', 'country': 'india'},
    'airtel_xstream': {'name': 'Airtel Xstream', 'is_free': True, 'url': 'https://airtelxstream.in', 'country': 'india'},
    'erosnow': {'name': 'Eros Now', 'is_free': False, 'url': 'https://erosnow.com', 'country': 'india'},
    'hungama': {'name': 'Hungama Play', 'is_free': False, 'url': 'https://hungama.com', 'country': 'india'},
    'shemaroo': {'name': 'ShemarooMe', 'is_free': True, 'url': 'https://shemaroome.com', 'country': 'india'}
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

# Streaming Availability Services
class WatchModeService:
    BASE_URL = 'https://api.watchmode.com/v1'
    
    @staticmethod
    def search_title(title, content_type='movie'):
        """Search for title on WatchMode"""
        url = f"{WatchModeService.BASE_URL}/search/"
        params = {
            'apiKey': WATCHMODE_API_KEY,
            'search_field': 'name',
            'search_value': title,
            'types': content_type
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"WatchMode search error: {e}")
        return None
    
    @staticmethod
    def get_streaming_sources(title_id):
        """Get streaming sources for a title"""
        url = f"{WatchModeService.BASE_URL}/title/{title_id}/sources/"
        params = {
            'apiKey': WATCHMODE_API_KEY,
            'regions': 'IN'  # India region
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"WatchMode sources error: {e}")
        return None

class RapidAPIStreamingService:
    BASE_URL = 'https://streaming-availability.p.rapidapi.com'
    
    @staticmethod
    def search_title(title, country='in'):
        """Search for title on RapidAPI Streaming Availability"""
        url = f"{RapidAPIStreamingService.BASE_URL}/search/title"
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
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"RapidAPI streaming search error: {e}")
        return None
    
    @staticmethod
    def get_streaming_info(imdb_id, country='in'):
        """Get streaming info by IMDB ID"""
        url = f"{RapidAPIStreamingService.BASE_URL}/get"
        headers = {
            'X-RapidAPI-Key': RAPIDAPI_KEY,
            'X-RapidAPI-Host': RAPIDAPI_HOST
        }
        params = {
            'imdb_id': imdb_id,
            'country': country
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"RapidAPI streaming info error: {e}")
        return None

# Enhanced Streaming Service
class StreamingAvailabilityService:
    @staticmethod
    def get_comprehensive_streaming_info(title, imdb_id=None, content_type='movie'):
        """Get comprehensive streaming information from multiple sources"""
        streaming_info = {
            'free_platforms': [],
            'paid_platforms': [],
            'language_specific_links': {},
            'direct_links': []
        }
        
        try:
            # Try WatchMode first
            watchmode_data = WatchModeService.search_title(title, content_type)
            if watchmode_data and watchmode_data.get('title_results'):
                title_id = watchmode_data['title_results'][0]['id']
                sources = WatchModeService.get_streaming_sources(title_id)
                
                if sources:
                    streaming_info = StreamingAvailabilityService._process_watchmode_sources(sources, streaming_info)
            
            # Try RapidAPI as backup/additional source
            rapidapi_data = RapidAPIStreamingService.search_title(title)
            if rapidapi_data:
                streaming_info = StreamingAvailabilityService._process_rapidapi_sources(rapidapi_data, streaming_info)
            
            # If we have IMDB ID, try RapidAPI with it
            if imdb_id:
                rapidapi_imdb_data = RapidAPIStreamingService.get_streaming_info(imdb_id)
                if rapidapi_imdb_data:
                    streaming_info = StreamingAvailabilityService._process_rapidapi_sources(rapidapi_imdb_data, streaming_info)
            
        except Exception as e:
            logger.error(f"Comprehensive streaming info error: {e}")
        
        return streaming_info
    
    @staticmethod
    def _process_watchmode_sources(sources, streaming_info):
        """Process WatchMode API sources"""
        try:
            for source in sources:
                platform_name = source.get('name', '').lower()
                source_type = source.get('type', '')
                web_url = source.get('web_url', '')
                
                platform_info = {
                    'platform': platform_name,
                    'name': source.get('name', ''),
                    'url': web_url,
                    'type': source_type,
                    'price': source.get('price', ''),
                    'quality': source.get('format', ''),
                    'audio_languages': source.get('audios', []),
                    'subtitle_languages': source.get('subtitles', [])
                }
                
                # Categorize as free or paid
                if source_type in ['free', 'ads'] or 'free' in platform_name:
                    streaming_info['free_platforms'].append(platform_info)
                else:
                    streaming_info['paid_platforms'].append(platform_info)
                
                # Add language-specific links
                for lang in source.get('audios', []):
                    if lang.lower() in REGIONAL_LANGUAGES:
                        if lang not in streaming_info['language_specific_links']:
                            streaming_info['language_specific_links'][lang] = []
                        streaming_info['language_specific_links'][lang].append(platform_info)
                
                # Add to direct links
                if web_url:
                    streaming_info['direct_links'].append({
                        'platform': platform_name,
                        'url': web_url,
                        'language': ', '.join(source.get('audios', [])),
                        'type': source_type
                    })
        
        except Exception as e:
            logger.error(f"Error processing WatchMode sources: {e}")
        
        return streaming_info
    
    @staticmethod
    def _process_rapidapi_sources(api_data, streaming_info):
        """Process RapidAPI Streaming Availability sources"""
        try:
            if isinstance(api_data, dict) and 'streamingInfo' in api_data:
                streaming_data = api_data['streamingInfo']
            elif isinstance(api_data, list) and len(api_data) > 0:
                streaming_data = api_data[0].get('streamingInfo', {})
            else:
                return streaming_info
            
            for country, services in streaming_data.items():
                if country.lower() != 'in':  # Focus on India
                    continue
                    
                for service_name, service_info in services.items():
                    platform_info = {
                        'platform': service_name.lower(),
                        'name': service_name,
                        'url': service_info.get('link', ''),
                        'type': service_info.get('type', ''),
                        'quality': service_info.get('quality', ''),
                        'audio_languages': service_info.get('audios', []),
                        'subtitle_languages': service_info.get('subtitles', [])
                    }
                    
                    # Determine if free or paid based on service type and name
                    is_free = (
                        service_info.get('type') == 'free' or
                        'free' in service_name.lower() or
                        service_name.lower() in ['youtube', 'mx player', 'voot', 'jiocinema']
                    )
                    
                    if is_free:
                        streaming_info['free_platforms'].append(platform_info)
                    else:
                        streaming_info['paid_platforms'].append(platform_info)
                    
                    # Add language-specific links
                    for lang in service_info.get('audios', []):
                        if lang.lower() in REGIONAL_LANGUAGES:
                            if lang not in streaming_info['language_specific_links']:
                                streaming_info['language_specific_links'][lang] = []
                            streaming_info['language_specific_links'][lang].append(platform_info)
                    
                    # Add to direct links
                    if service_info.get('link'):
                        streaming_info['direct_links'].append({
                            'platform': service_name,
                            'url': service_info.get('link'),
                            'language': ', '.join(service_info.get('audios', [])),
                            'type': service_info.get('type', 'subscription')
                        })
        
        except Exception as e:
            logger.error(f"Error processing RapidAPI sources: {e}")
        
        return streaming_info

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
                # Update streaming information if content exists
                ContentService._update_streaming_info(existing, tmdb_data)
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
            
            # Get IMDB ID if available
            imdb_id = None
            if 'external_ids' in tmdb_data:
                imdb_id = tmdb_data['external_ids'].get('imdb_id')
            elif 'imdb_id' in tmdb_data:
                imdb_id = tmdb_data['imdb_id']
            
            # Get comprehensive streaming information
            title = tmdb_data.get('title') or tmdb_data.get('name')
            streaming_info = StreamingAvailabilityService.get_comprehensive_streaming_info(
                title, imdb_id, content_type
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
                ott_platforms=json.dumps(ContentService._format_ott_platforms(streaming_info)),
                streaming_links=json.dumps(streaming_info)
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def _update_streaming_info(content, tmdb_data):
        """Update streaming information for existing content"""
        try:
            title = content.title
            imdb_id = content.imdb_id
            
            streaming_info = StreamingAvailabilityService.get_comprehensive_streaming_info(
                title, imdb_id, content.content_type
            )
            
            content.streaming_links = json.dumps(streaming_info)
            content.ott_platforms = json.dumps(ContentService._format_ott_platforms(streaming_info))
            content.updated_at = datetime.utcnow()
            
            db.session.commit()
        except Exception as e:
            logger.error(f"Error updating streaming info: {e}")
            db.session.rollback()
    
    @staticmethod
    def _format_ott_platforms(streaming_info):
        """Format streaming info for ott_platforms field"""
        platforms = []
        
        # Add free platforms
        for platform in streaming_info.get('free_platforms', []):
            platforms.append({
                'platform': platform['platform'],
                'name': platform['name'],
                'url': platform['url'],
                'is_free': True,
                'type': platform.get('type', 'free')
            })
        
        # Add paid platforms
        for platform in streaming_info.get('paid_platforms', []):
            platforms.append({
                'platform': platform['platform'],
                'name': platform['name'],
                'url': platform['url'],
                'is_free': False,
                'type': platform.get('type', 'subscription')
            })
        
        return platforms
    
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
    def get_ott_availability(tmdb_data):
        # This would integrate with actual OTT APIs or databases
        # For now, return sample data
        platforms = []
        
        # You would implement actual OTT checking logic here
        # For demo purposes, randomly assign some platforms
        sample_platforms = ['netflix', 'amazon_prime', 'youtube', 'jiocinema']
        
        for platform in random.sample(sample_platforms, random.randint(1, 3)):
            platforms.append({
                'platform': platform,
                'url': OTT_PLATFORMS[platform]['url'],
                'is_free': OTT_PLATFORMS[platform]['is_free']
            })
        
        return platforms

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
                    ott_platforms=json.dumps([]),  # You would check anime streaming platforms
                    streaming_links=json.dumps({})
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
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Get and format streaming links
            streaming_text = TelegramService._format_streaming_links(content)
            
            # Create message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}{streaming_text}

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
                    # Try without markdown if it fails
                    try:
                        bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
                    except:
                        # Send as plain text if markdown fails
                        plain_message = TelegramService._convert_to_plain_text(message)
                        bot.send_message(TELEGRAM_CHANNEL_ID, plain_message)
            else:
                try:
                    bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
                except:
                    # Send as plain text if markdown fails
                    plain_message = TelegramService._convert_to_plain_text(message)
                    bot.send_message(TELEGRAM_CHANNEL_ID, plain_message)
            
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    @staticmethod
    def _format_streaming_links(content):
        """Format streaming links for Telegram message"""
        streaming_text = ""
        
        if not content.streaming_links:
            return streaming_text
        
        try:
            streaming_data = json.loads(content.streaming_links)
            
            # Collect all streaming links
            free_links = []
            paid_links = []
            language_links = {}
            
            # Process free platforms
            for platform in streaming_data.get('free_platforms', []):
                if platform.get('url') and platform.get('name'):
                    link_text = f"[{platform['name']}]({platform['url']})"
                    free_links.append(link_text)
            
            # Process paid platforms
            for platform in streaming_data.get('paid_platforms', []):
                if platform.get('url') and platform.get('name'):
                    link_text = f"[{platform['name']}]({platform['url']})"
                    paid_links.append(link_text)
            
            # Process language-specific links
            for lang, platforms in streaming_data.get('language_specific_links', {}).items():
                lang_links = []
                for platform in platforms:
                    if platform.get('url') and platform.get('name'):
                        link_text = f"[{platform['name']}]({platform['url']})"
                        if link_text not in lang_links:
                            lang_links.append(link_text)
                
                if lang_links:
                    language_links[lang.upper()] = lang_links
            
            # Process direct links as fallback
            direct_links = []
            for link in streaming_data.get('direct_links', []):
                if link.get('url') and link.get('platform'):
                    platform_name = link['platform'].replace('_', ' ').title()
                    language_info = f" ({link['language']})" if link.get('language') and link['language'] != 'N/A' else ""
                    link_text = f"[{platform_name}{language_info}]({link['url']})"
                    direct_links.append(link_text)
            
            # Build streaming text
            if free_links or paid_links or language_links or direct_links:
                streaming_text = "\n\n🎬 **WATCH NOW:**"
                
                # Add free platforms
                if free_links:
                    streaming_text += f"\n\n🆓 **FREE:**\n" + " • ".join(free_links[:4])
                    if len(free_links) > 4:
                        streaming_text += f" • +{len(free_links) - 4} more"
                
                # Add paid platforms
                if paid_links:
                    streaming_text += f"\n\n💰 **SUBSCRIPTION:**\n" + " • ".join(paid_links[:4])
                    if len(paid_links) > 4:
                        streaming_text += f" • +{len(paid_links) - 4} more"
                
                # Add language-specific links
                if language_links:
                    streaming_text += "\n\n🎭 **BY LANGUAGE:**"
                    for lang, links in list(language_links.items())[:3]:  # Limit to 3 languages
                        streaming_text += f"\n**{lang}:** " + " • ".join(links[:3])
                        if len(links) > 3:
                            streaming_text += f" • +{len(links) - 3} more"
                
                # Add direct links if no categorized links available
                if not free_links and not paid_links and direct_links:
                    streaming_text += f"\n\n📺 **PLATFORMS:**\n" + " • ".join(direct_links[:6])
                    if len(direct_links) > 6:
                        streaming_text += f" • +{len(direct_links) - 6} more"
                
                # Add note about links
                streaming_text += "\n\n💡 *Click links to watch directly*"
        
        except Exception as e:
            logger.error(f"Error formatting streaming links: {e}")
        
        return streaming_text
    
    @staticmethod
    def _convert_to_plain_text(markdown_message):
        """Convert markdown message to plain text as fallback"""
        import re
        
        # Remove markdown formatting
        plain_text = re.sub(r'\*\*(.*?)\*\*', r'\1', markdown_message)  # Bold
        plain_text = re.sub(r'\*(.*?)\*', r'\1', plain_text)  # Italic
        plain_text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1: \2', plain_text)  # Links
        plain_text = re.sub(r'`(.*?)`', r'\1', plain_text)  # Code
        
        return plain_text
    
    @staticmethod
    def send_trending_update(trending_content, limit=5):
        """Send trending content update to Telegram channel"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            message = "🔥 **TRENDING NOW** 🔥\n\n"
            
            for i, content in enumerate(trending_content[:limit], 1):
                genres = []
                if content.genres:
                    try:
                        genres = json.loads(content.genres)[:2]  # Limit to 2 genres
                    except:
                        pass
                
                genre_text = f" | {', '.join(genres)}" if genres else ""
                rating_text = f" ⭐ {content.rating}" if content.rating else ""
                
                message += f"{i}. **{content.title}**{genre_text}{rating_text}\n"
                
                # Add top streaming link if available
                if content.streaming_links:
                    try:
                        streaming_data = json.loads(content.streaming_links)
                        
                        # Get first available link
                        first_link = None
                        if streaming_data.get('free_platforms'):
                            platform = streaming_data['free_platforms'][0]
                            first_link = f"🆓 [{platform['name']}]({platform['url']})"
                        elif streaming_data.get('paid_platforms'):
                            platform = streaming_data['paid_platforms'][0]
                            first_link = f"💰 [{platform['name']}]({platform['url']})"
                        elif streaming_data.get('direct_links'):
                            link = streaming_data['direct_links'][0]
                            platform_name = link['platform'].replace('_', ' ').title()
                            first_link = f"📺 [{platform_name}]({link['url']})"
                        
                        if first_link:
                            message += f"   {first_link}\n"
                    except:
                        pass
                
                message += "\n"
            
            message += "#Trending #Movies #TVShows #CineScope"
            
            try:
                bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
                return True
            except:
                # Fallback to plain text
                plain_message = TelegramService._convert_to_plain_text(message)
                bot.send_message(TELEGRAM_CHANNEL_ID, plain_message)
                return True
                
        except Exception as e:
            logger.error(f"Trending update send error: {e}")
            return False
    
    @staticmethod
    def send_new_content_alert(content):
        """Send alert for new content addition"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            # Get streaming info
            streaming_text = TelegramService._format_streaming_links(content)
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Format genres
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            message = f"""🆕 **NEW ADDITION**

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📖 {(content.overview[:150] + '...') if content.overview else 'No synopsis available'}{streaming_text}

#NewContent #CineScope"""
            
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='Markdown'
                    )
                except:
                    try:
                        bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
                    except:
                        plain_message = TelegramService._convert_to_plain_text(message)
                        bot.send_message(TELEGRAM_CHANNEL_ID, plain_message)
            else:
                try:
                    bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
                except:
                    plain_message = TelegramService._convert_to_plain_text(message)
                    bot.send_message(TELEGRAM_CHANNEL_ID, plain_message)
            
            return True
        except Exception as e:
            logger.error(f"New content alert error: {e}")
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
                    
                    # Parse streaming links
                    streaming_links = []
                    if content.streaming_links:
                        try:
                            streaming_data = json.loads(content.streaming_links)
                            streaming_links = streaming_data.get('direct_links', [])
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
                        'ott_platforms': json.loads(content.ott_platforms or '[]'),
                        'streaming_links': streaming_links
                    })
        
        # Add anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                results.append({
                    'id': f"anime_{anime['mal_id']}",
                    'title': anime.get('title'),
                    'content_type': 'anime',
                    'genres': [genre['name'] for genre in anime.get('genres', [])],
                    'rating': anime.get('score'),
                    'release_date': anime.get('aired', {}).get('from'),
                    'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                    'overview': anime.get('synopsis'),
                    'ott_platforms': [],
                    'streaming_links': []
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
        
        # Parse streaming information
        streaming_info = {}
        if content.streaming_links:
            try:
                streaming_info = json.loads(content.streaming_links)
            except:
                streaming_info = {}
        
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
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'runtime': content.runtime,
            'rating': content.rating,
            'vote_count': content.vote_count,
            'overview': content.overview,
            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path else None,
            'ott_platforms': json.loads(content.ott_platforms or '[]'),
            'streaming_info': streaming_info,
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
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
            # Parse streaming links
            streaming_links = []
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                    streaming_links = streaming_data.get('direct_links', [])
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
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
                'streaming_links': streaming_links
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
            # Parse streaming links
            streaming_links = []
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                    streaming_links = streaming_data.get('direct_links', [])
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
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
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
            # Parse streaming links
            streaming_links = []
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                    streaming_links = streaming_data.get('direct_links', [])
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
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
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
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
                'streaming_links': []
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
            # Parse streaming links
            streaming_links = []
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                    streaming_links = streaming_data.get('direct_links', [])
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
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
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
                        # Parse streaming links
                        streaming_links = []
                        if content.streaming_links:
                            try:
                                streaming_data = json.loads(content.streaming_links)
                                streaming_links = streaming_data.get('direct_links', [])
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
                            'ott_platforms': json.loads(content.ott_platforms or '[]'),
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
            # Parse streaming links
            streaming_links = []
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                    streaming_links = streaming_data.get('direct_links', [])
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
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
            # Parse streaming links
            streaming_links = []
            if content.streaming_links:
                try:
                    streaming_data = json.loads(content.streaming_links)
                    streaming_links = streaming_data.get('direct_links', [])
                except:
                    pass
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
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
            
            # Get streaming information for the content
            streaming_info = StreamingAvailabilityService.get_comprehensive_streaming_info(
                data.get('title'), None, data.get('content_type', 'movie')
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
                ott_platforms=json.dumps(ContentService._format_ott_platforms(streaming_info)),
                streaming_links=json.dumps(streaming_info)
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
                # Parse streaming links
                streaming_links = []
                if content.streaming_links:
                    try:
                        streaming_data = json.loads(content.streaming_links)
                        streaming_links = streaming_data.get('direct_links', [])
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
                    'ott_platforms': json.loads(content.ott_platforms or '[]'),
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