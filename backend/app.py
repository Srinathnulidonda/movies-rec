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
from typing import Dict, List, Optional

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
ENHANCED_OTT_PLATFORMS = {
    'netflix': {'name': 'Netflix', 'is_free': False, 'icon': '🔴', 'priority': 1, 'base_url': 'https://netflix.com'},
    'amazon_prime': {'name': 'Prime Video', 'is_free': False, 'icon': '🟦', 'priority': 2, 'base_url': 'https://primevideo.com'},
    'disney_plus': {'name': 'Disney+ Hotstar', 'is_free': False, 'icon': '🌟', 'priority': 3, 'base_url': 'https://hotstar.com'},
    'zee5': {'name': 'ZEE5', 'is_free': False, 'icon': '🟣', 'priority': 4, 'base_url': 'https://zee5.com'},
    'zee5_free': {'name': 'ZEE5 Free', 'is_free': True, 'icon': '🟣', 'priority': 5, 'base_url': 'https://zee5.com'},
    'sonyliv': {'name': 'SonyLIV Premium', 'is_free': False, 'icon': '🔵', 'priority': 6, 'base_url': 'https://sonyliv.com'},
    'sonyliv_free': {'name': 'SonyLIV Free', 'is_free': True, 'icon': '🔵', 'priority': 7, 'base_url': 'https://sonyliv.com'},
    'youtube': {'name': 'YouTube', 'is_free': True, 'icon': '🔴', 'priority': 8, 'base_url': 'https://youtube.com'},
    'jiocinema': {'name': 'JioCinema', 'is_free': True, 'icon': '🟪', 'priority': 9, 'base_url': 'https://jiocinema.com'},
    'mx_player': {'name': 'MX Player', 'is_free': True, 'icon': '🟠', 'priority': 10, 'base_url': 'https://mxplayer.com'},
    'voot': {'name': 'Voot', 'is_free': True, 'icon': '🟤', 'priority': 11, 'base_url': 'https://voot.com'},
    'alt_balaji': {'name': 'ALTBalaji', 'is_free': False, 'icon': '🟡', 'priority': 12, 'base_url': 'https://altbalaji.com'},
    'aha': {'name': 'Aha', 'is_free': False, 'icon': '🟢', 'priority': 13, 'base_url': 'https://aha.video'},
    'sun_nxt': {'name': 'Sun NXT', 'is_free': False, 'icon': '🟠', 'priority': 14, 'base_url': 'https://sunnxt.com'},
    'airtel_xstream': {'name': 'Airtel Xstream', 'is_free': True, 'icon': '🔴', 'priority': 15, 'base_url': 'https://airtelxstream.in'},
    'crunchyroll': {'name': 'Crunchyroll', 'is_free': True, 'icon': '🟠', 'priority': 16, 'base_url': 'https://crunchyroll.com'}
}

# Regional Language Mapping
REGIONAL_LANGUAGES = {
    'hindi': ['hi', 'hindi', 'bollywood'],
    'telugu': ['te', 'telugu', 'tollywood'],
    'tamil': ['ta', 'tamil', 'kollywood'],
    'kannada': ['kn', 'kannada', 'sandalwood'],
    'malayalam': ['ml', 'malayalam', 'mollywood'],
    'english': ['en', 'english', 'hollywood'],
    'japanese': ['ja', 'japanese', 'anime']
}

# Regional Priority Order
REGIONAL_PRIORITY = ['telugu', 'hindi', 'tamil', 'malayalam', 'kannada', 'english']

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
    def get_streaming_availability(imdb_id: str, title: str, content_type: str = 'movie') -> Dict:
        """Get real-time streaming availability from multiple sources"""
        try:
            # Try RapidAPI first
            rapidapi_result = StreamingAvailabilityService._get_from_rapidapi(imdb_id, title, content_type)
            if rapidapi_result:
                return rapidapi_result
            
            # Fallback to WatchMode API
            watchmode_result = StreamingAvailabilityService._get_from_watchmode(title, content_type)
            if watchmode_result:
                return watchmode_result
            
            # Return sample data for demonstration
            return StreamingAvailabilityService._get_sample_availability(title, content_type)
            
        except Exception as e:
            logger.error(f"Streaming availability error: {e}")
            return StreamingAvailabilityService._get_sample_availability(title, content_type)
    
    @staticmethod
    def _get_from_rapidapi(imdb_id: str, title: str, content_type: str) -> Optional[Dict]:
        """Get availability from RapidAPI Streaming Availability"""
        try:
            headers = {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': RAPIDAPI_HOST
            }
            
            # Search by IMDB ID first
            if imdb_id:
                url = f"https://{RAPIDAPI_HOST}/get"
                params = {
                    'imdb_id': imdb_id,
                    'country': 'in'  # India
                }
            else:
                # Search by title
                url = f"https://{RAPIDAPI_HOST}/search/title"
                params = {
                    'title': title,
                    'country': 'in',
                    'show_type': content_type
                }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return StreamingAvailabilityService._process_rapidapi_response(data)
            
        except Exception as e:
            logger.error(f"RapidAPI streaming error: {e}")
        
        return None
    
    @staticmethod
    def _get_from_watchmode(title: str, content_type: str) -> Optional[Dict]:
        """Get availability from WatchMode API"""
        try:
            if not WATCHMODE_API_KEY or WATCHMODE_API_KEY == 'your_watchmode_api_key':
                return None
            
            # Search for the title
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
                    
                    # Get detailed info with sources
                    detail_url = f"https://api.watchmode.com/v1/title/{title_id}/details/"
                    detail_params = {
                        'apiKey': WATCHMODE_API_KEY,
                        'append_to_response': 'sources'
                    }
                    
                    detail_response = requests.get(detail_url, params=detail_params, timeout=10)
                    
                    if detail_response.status_code == 200:
                        detail_data = detail_response.json()
                        return StreamingAvailabilityService._process_watchmode_response(detail_data)
            
        except Exception as e:
            logger.error(f"WatchMode API error: {e}")
        
        return None
    
    @staticmethod
    def _get_sample_availability(title: str, content_type: str) -> Dict:
        """Generate sample streaming availability for demonstration"""
        # Determine likely platforms based on content characteristics
        sample_platforms = []
        
        # Add some common platforms with sample data
        if 'bollywood' in title.lower() or 'hindi' in title.lower():
            sample_platforms.extend([
                {
                    'platform_id': 'zee5',
                    'platform_name': 'ZEE5',
                    'is_free': False,
                    'icon': '🟣',
                    'priority': 4,
                    'watch_url': f"https://zee5.com/search?q={title}",
                    'quality': 'HD',
                    'type': 'subscription',
                    'audios': ['hindi', 'english'],
                    'subtitles': ['english', 'hindi'],
                    'availability': 'Available Now'
                },
                {
                    'platform_id': 'netflix',
                    'platform_name': 'Netflix',
                    'is_free': False,
                    'icon': '🔴',
                    'priority': 1,
                    'watch_url': f"https://netflix.com/search?q={title}",
                    'quality': 'HD',
                    'type': 'subscription',
                    'audios': ['hindi', 'english'],
                    'subtitles': ['english', 'hindi'],
                    'availability': 'Available Now'
                }
            ])
        
        if 'tollywood' in title.lower() or 'telugu' in title.lower():
            sample_platforms.extend([
                {
                    'platform_id': 'aha',
                    'platform_name': 'Aha',
                    'is_free': False,
                    'icon': '🟢',
                    'priority': 13,
                    'watch_url': f"https://aha.video/search?q={title}",
                    'quality': 'HD',
                    'type': 'subscription',
                    'audios': ['telugu', 'hindi'],
                    'subtitles': ['english', 'telugu'],
                    'availability': 'Available Now'
                },
                {
                    'platform_id': 'zee5',
                    'platform_name': 'ZEE5',
                    'is_free': False,
                    'icon': '🟣',
                    'priority': 4,
                    'watch_url': f"https://zee5.com/search?q={title}",
                    'quality': 'HD',
                    'type': 'subscription',
                    'audios': ['telugu', 'hindi'],
                    'subtitles': ['english', 'telugu'],
                    'availability': 'Available Now'
                }
            ])
        
        # Add some free platforms
        sample_platforms.extend([
            {
                'platform_id': 'youtube',
                'platform_name': 'YouTube',
                'is_free': True,
                'icon': '🔴',
                'priority': 8,
                'watch_url': f"https://youtube.com/results?search_query={title}",
                'quality': 'HD',
                'type': 'free',
                'audios': ['hindi', 'english', 'telugu'],
                'subtitles': ['english'],
                'availability': 'Available Now'
            },
            {
                'platform_id': 'mx_player',
                'platform_name': 'MX Player',
                'is_free': True,
                'icon': '🟠',
                'priority': 10,
                'watch_url': f"https://mxplayer.com/search?q={title}",
                'quality': 'HD',
                'type': 'free',
                'audios': ['hindi', 'english'],
                'subtitles': ['english'],
                'availability': 'Available Now'
            }
        ])
        
        # Remove duplicates and sort by priority
        unique_platforms = {p['platform_id']: p for p in sample_platforms}.values()
        sorted_platforms = sorted(unique_platforms, key=lambda x: x['priority'])
        
        # Extract unique languages
        all_languages = set()
        for platform in sorted_platforms:
            all_languages.update(platform.get('audios', []))
        
        return {
            'platforms': list(sorted_platforms),
            'languages': list(all_languages)
        }
    
    @staticmethod
    def _process_rapidapi_response(data: Dict) -> Dict:
        """Process RapidAPI response to our format"""
        platforms = []
        languages = set()
        
        if isinstance(data, list):
            data = data[0] if data else {}
        
        streaming_options = data.get('streamingOptions', {})
        
        for country, options in streaming_options.items():
            if country == 'in':  # India
                for option in options:
                    service = option.get('service', {})
                    service_id = service.get('id', '').lower()
                    
                    # Map service ID to our platform IDs
                    platform_mapping = {
                        'netflix': 'netflix',
                        'prime': 'amazon_prime',
                        'hotstar': 'disney_plus',
                        'zee5': 'zee5',
                        'sonyliv': 'sonyliv',
                        'youtube': 'youtube',
                        'jiocinema': 'jiocinema',
                        'mxplayer': 'mx_player',
                        'voot': 'voot'
                    }
                    
                    platform_id = None
                    for key, mapped_id in platform_mapping.items():
                        if key in service_id:
                            platform_id = mapped_id
                            break
                    
                    if platform_id and platform_id in ENHANCED_OTT_PLATFORMS:
                        # Get audio languages
                        audios = option.get('audios', [])
                        subtitles = option.get('subtitles', [])
                        
                        for audio in audios:
                            languages.add(audio.get('language', 'en'))
                        
                        platforms.append({
                            'platform_id': platform_id,
                            'platform_name': ENHANCED_OTT_PLATFORMS[platform_id]['name'],
                            'is_free': ENHANCED_OTT_PLATFORMS[platform_id]['is_free'],
                            'icon': ENHANCED_OTT_PLATFORMS[platform_id]['icon'],
                            'priority': ENHANCED_OTT_PLATFORMS[platform_id]['priority'],
                            'watch_url': option.get('link', ''),
                            'quality': option.get('quality', 'HD'),
                            'type': option.get('type', 'subscription'),
                            'audios': [audio.get('language') for audio in audios],
                            'subtitles': [sub.get('language') for sub in subtitles],
                            'availability': 'Available Now'
                        })
        
        return {
            'platforms': sorted(platforms, key=lambda x: x['priority']),
            'languages': list(languages)
        }
    
    @staticmethod
    def _process_watchmode_response(data: Dict) -> Dict:
        """Process WatchMode response to our format"""
        platforms = []
        languages = set()
        
        sources = data.get('sources', [])
        
        for source in sources:
            source_name = source.get('name', '').lower()
            
            # Map source names to our platform IDs
            platform_mapping = {
                'netflix': 'netflix',
                'amazon prime video': 'amazon_prime',
                'disney+ hotstar': 'disney_plus',
                'zee5': 'zee5',
                'sonyliv': 'sonyliv',
                'youtube': 'youtube',
                'jiocinema': 'jiocinema',
                'mx player': 'mx_player',
                'voot': 'voot'
            }
            
            platform_id = platform_mapping.get(source_name)
            
            if platform_id and platform_id in ENHANCED_OTT_PLATFORMS:
                platforms.append({
                    'platform_id': platform_id,
                    'platform_name': ENHANCED_OTT_PLATFORMS[platform_id]['name'],
                    'is_free': ENHANCED_OTT_PLATFORMS[platform_id]['is_free'],
                    'icon': ENHANCED_OTT_PLATFORMS[platform_id]['icon'],
                    'priority': ENHANCED_OTT_PLATFORMS[platform_id]['priority'],
                    'watch_url': source.get('web_url', ''),
                    'quality': 'HD',
                    'type': source.get('type', 'subscription'),
                    'audios': ['hindi', 'english'],  # Default languages
                    'subtitles': ['english'],
                    'availability': 'Available Now'
                })
        
        return {
            'platforms': sorted(platforms, key=lambda x: x['priority']),
            'languages': list(languages)
        }

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
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limit, wait and retry
                time.sleep(2)
                response = requests.get(url, params=params, timeout=15)
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
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(2)
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"Jikan top anime error: {e}")
        return None
    
    @staticmethod
    def get_anime_details(anime_id: int) -> Optional[Dict]:
        """Get detailed anime information - FIXED VERSION"""
        try:
            url = f"{JikanService.BASE_URL}/anime/{anime_id}"
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('data')
            elif response.status_code == 429:
                # Rate limit hit, wait and retry
                time.sleep(2)
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    return data.get('data')
            
        except Exception as e:
            logger.error(f"Jikan anime details error: {e}")
        
        return None

class YouTubeService:
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    @staticmethod
    def search_trailers(query):
        if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == 'your_youtube_api_key':
            return None
            
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
            if 'spoken_languages' in tmdb_data:
                languages = [lang['name'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
            
            # Get IMDB ID if available
            imdb_id = None
            if 'external_ids' in tmdb_data and tmdb_data['external_ids'].get('imdb_id'):
                imdb_id = tmdb_data['external_ids']['imdb_id']
            
            # Get OTT platforms with real-time data
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
        """Get real-time OTT availability"""
        title = tmdb_data.get('title') or tmdb_data.get('name', '')
        content_type = 'movie' if 'title' in tmdb_data else 'tv'
        
        # Get real-time streaming data
        streaming_data = StreamingAvailabilityService.get_streaming_availability(imdb_id, title, content_type)
        
        return streaming_data.get('platforms', [])
    
    @staticmethod
    def get_language_specific_availability(content_id: int, language: str) -> List[Dict]:
        """Get streaming platforms available for specific language"""
        try:
            content = Content.query.get(content_id)
            if not content:
                return []
            
            ott_platforms = json.loads(content.ott_platforms or '[]')
            
            # Filter platforms that support the requested language
            language_platforms = []
            for platform in ott_platforms:
                audios = platform.get('audios', [])
                if language.lower() in [audio.lower() for audio in audios]:
                    # Create language-specific watch URL
                    base_url = platform.get('watch_url', '')
                    if base_url:
                        lang_url = f"{base_url}?lang={language}"
                    else:
                        lang_url = f"{ENHANCED_OTT_PLATFORMS.get(platform.get('platform_id', ''), {}).get('base_url', '')}?search={content.title}&lang={language}"
                    
                    language_platforms.append({
                        **platform,
                        'language': language,
                        'watch_url': lang_url
                    })
            
            return language_platforms
        except Exception as e:
            logger.error(f"Language availability error: {e}")
            return []

# Regional Priority Service
class RegionalPriorityService:
    @staticmethod
    def get_prioritized_homepage_content(user_location=None, limit_per_section=20):
        """Get homepage content with regional priorities"""
        try:
            homepage_content = {
                'trending': [],
                'all_time_hits': [],
                'new_releases': [],
                'by_genre': {},
                'by_language': {}
            }
            
            # Determine primary language based on location or default to Telugu
            primary_language = RegionalPriorityService._get_primary_language(user_location)
            
            # Get trending content with regional priority
            trending = RecommendationEngine.get_trending_recommendations(limit=limit_per_section * 2)
            homepage_content['trending'] = RegionalPriorityService._apply_language_priority(
                trending, primary_language, limit_per_section
            )
            
            # Get new releases (last 6 months)
            six_months_ago = datetime.utcnow() - timedelta(days=180)
            new_releases = Content.query.filter(
                Content.release_date >= six_months_ago.date()
            ).order_by(Content.popularity.desc()).limit(limit_per_section * 2).all()
            
            homepage_content['new_releases'] = RegionalPriorityService._apply_language_priority(
                new_releases, primary_language, limit_per_section
            )
            
            # Get all-time hits (high rating + high vote count)
            all_time_hits = Content.query.filter(
                and_(Content.rating >= 7.5, Content.vote_count >= 1000)
            ).order_by(Content.rating.desc()).limit(limit_per_section * 2).all()
            
            homepage_content['all_time_hits'] = RegionalPriorityService._apply_language_priority(
                all_time_hits, primary_language, limit_per_section
            )
            
            # Get content by genre with regional priority
            popular_genres = ['Action', 'Drama', 'Comedy', 'Romance', 'Thriller', 'Sci-Fi']
            for genre in popular_genres:
                genre_content = RecommendationEngine.get_popular_by_genre(genre, limit=limit_per_section)
                homepage_content['by_genre'][genre] = RegionalPriorityService._apply_language_priority(
                    genre_content, primary_language, limit_per_section // 2
                )
            
            # Get content by language in priority order
            for language in REGIONAL_PRIORITY:
                lang_content = RecommendationEngine.get_regional_recommendations(language, limit=limit_per_section)
                if lang_content:
                    homepage_content['by_language'][language] = lang_content
            
            return homepage_content
            
        except Exception as e:
            logger.error(f"Regional priority error: {e}")
            return {}
    
    @staticmethod
    def _get_primary_language(user_location):
        """Determine primary language based on user location"""
        if not user_location:
            return 'telugu'  # Default to Telugu as requested
        
        region = user_location.get('region', '').lower()
        
        # Map Indian states to primary languages
        state_language_map = {
            'telangana': 'telugu',
            'andhra pradesh': 'telugu',
            'tamil nadu': 'tamil',
            'kerala': 'malayalam',
            'karnataka': 'kannada',
            'maharashtra': 'hindi',
            'uttar pradesh': 'hindi',
            'bihar': 'hindi',
            'west bengal': 'hindi',
            'rajasthan': 'hindi'
        }
        
        for state, language in state_language_map.items():
            if state in region:
                return language
        
        return 'telugu'  # Default fallback
    
    @staticmethod
    def _apply_language_priority(content_list, primary_language, limit):
        """Apply language-based priority to content list"""
        try:
            # Separate content by language preference
            prioritized = []
            others = []
            
            for content in content_list:
                languages = json.loads(content.languages or '[]')
                
                # Check if content is in primary language
                if any(primary_language.lower() in lang.lower() for lang in languages):
                    prioritized.append(content)
                else:
                    others.append(content)
            
            # Combine with priority language first
            result = prioritized[:limit//2] + others[:limit//2]
            
            return result[:limit]
            
        except Exception as e:
            logger.error(f"Language priority error: {e}")
            return content_list[:limit]
    
    @staticmethod
    def _format_content_list(content_list):
        """Format content list for API response"""
        result = []
        for content in content_list:
            # Get streaming platforms
            platforms = json.loads(content.ott_platforms or '[]')
            languages = json.loads(content.languages or '[]')
            
            # Create language-specific watch buttons
            watch_buttons = {}
            for language in languages:
                lang_platforms = ContentService.get_language_specific_availability(content.id, language)
                if lang_platforms:
                    watch_buttons[language] = {
                        'free_platforms': [p for p in lang_platforms if p.get('is_free', False)],
                        'paid_platforms': [p for p in lang_platforms if not p.get('is_free', False)]
                    }
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': languages,
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'watch_buttons': watch_buttons,
                'streaming_summary': {
                    'total_platforms': len(platforms),
                    'free_available': len([p for p in platforms if p.get('is_free', False)]) > 0,
                    'primary_platform': platforms[0] if platforms else None
                }
            })
        
        return result

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
                'kannada': ['sandalwood', 'kannada movie', 'kannada film'],
                'malayalam': ['mollywood', 'malayalam movie', 'malayalam film']
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
                    ott_platforms=json.dumps([{
                        'platform_id': 'crunchyroll',
                        'platform_name': 'Crunchyroll',
                        'is_free': True,
                        'icon': '🟠',
                        'watch_url': f"https://crunchyroll.com/search?q={anime.get('title', '')}",
                        'availability': 'Available Now'
                    }])
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
                primary_lang = RegionalPriorityService._get_primary_language(location)
                regional_recs = RecommendationEngine.get_regional_recommendations(primary_lang, limit=5)
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
            
            # Format the message exactly as requested
            message = TelegramService._format_recommendation_message(content, admin_name, description)
            
            # Get poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Send message with photo if available
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='HTML'
                    )
                except Exception as photo_error:
                    logger.error(f"Failed to send photo, sending text only: {photo_error}")
                    bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='HTML')
            else:
                bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='HTML')
            
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    @staticmethod
    def _format_recommendation_message(content, admin_name, description):
        """Format message exactly as requested in the example"""
        
        # Get genres
        genres_list = []
        if content.genres:
            try:
                genres_list = json.loads(content.genres)
            except:
                genres_list = []
        
        # Get languages
        languages_list = []
        if content.languages:
            try:
                languages_list = json.loads(content.languages)
            except:
                languages_list = []
        
        # Get streaming platforms
        platforms = []
        if content.ott_platforms:
            try:
                platforms = json.loads(content.ott_platforms)
            except:
                platforms = []
        
        # Build the title with year
        title_with_year = content.title
        if content.release_date:
            title_with_year = f"{content.title} ({content.release_date.year})"
        
        # Format genres (max 3)
        genres_text = ', '.join(genres_list[:3]) if genres_list else 'N/A'
        
        # Format available languages
        languages_text = ', '.join(languages_list[:4]) if languages_list else 'N/A'
        
        # Format synopsis (first 200 characters)
        synopsis = content.overview if content.overview else "No synopsis available"
        if len(synopsis) > 200:
            synopsis = synopsis[:200] + "..."
        
        # Build streaming platforms text with language-specific links
        streaming_text = ""
        if platforms:
            # Group platforms by language
            platform_by_lang = {}
            for platform in platforms:
                audios = platform.get('audios', [])
                for audio in audios:
                    if audio not in platform_by_lang:
                        platform_by_lang[audio] = []
                    platform_by_lang[audio].append(platform)
            
            # Create streaming links for each language
            streaming_lines = []
            for language in languages_list[:3]:  # Show max 3 languages
                lang_platforms = platform_by_lang.get(language.lower(), [])
                if lang_platforms:
                    # Get the best platform (highest priority)
                    best_platform = min(lang_platforms, key=lambda x: x.get('priority', 999))
                    platform_icon = best_platform.get('icon', '📺')
                    platform_name = best_platform.get('platform_name', 'Unknown')
                    
                    streaming_lines.append(f"        {platform_icon} {platform_name} - Watch in {language.title()}")
            
            # If no language-specific platforms, show general platforms
            if not streaming_lines and platforms:
                for platform in platforms[:2]:  # Show max 2 platforms
                    platform_icon = platform.get('icon', '📺')
                    platform_name = platform.get('platform_name', 'Unknown')
                    streaming_lines.append(f"        {platform_icon} {platform_name}")
            
            streaming_text = '\n'.join(streaming_lines) if streaming_lines else "        📺 Check local streaming platforms"
        else:
            streaming_text = "        📺 Check local streaming platforms"
        
        # Build the complete message
        message = f"""🎬 <b>{title_with_year}</b>
🎭 <b>Genre:</b> {genres_text}
🌐 <b>Available in:</b> {languages_text}

🧾 <b>Synopsis:</b> {synopsis}
📺 <b>Where to Watch:</b>
{streaming_text}

<i>✨ Recommended by Admin {admin_name}</i>
<i>💬 "{description}"</i>

#AdminChoice #MovieRecommendation #CineScope"""
        
        return message
    
    @staticmethod
    def send_trending_update(trending_content):
        """Send trending content update to Telegram channel"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            message = "🔥 <b>Trending Now on CineScope</b>\n\n"
            
            for i, content in enumerate(trending_content[:5], 1):
                genres = json.loads(content.genres or '[]')
                message += f"{i}. <b>{content.title}</b>\n"
                message += f"   ⭐ {content.rating or 'N/A'}/10 | 🎭 {', '.join(genres[:2])}\n\n"
            
            message += "#Trending #CineScope"
            
            bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='HTML')
            return True
            
        except Exception as e:
            logger.error(f"Telegram trending update error: {e}")
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

# Enhanced Homepage with Regional Priority
@app.route('/api/homepage', methods=['GET'])
def get_prioritized_homepage():
    try:
        # Get user location for regional prioritization
        user_location = get_user_location(request.remote_addr)
        
        # Get prioritized content
        homepage_data = RegionalPriorityService.get_prioritized_homepage_content(user_location)
        
        # Format response for frontend
        formatted_response = {}
        
        for section, content_list in homepage_data.items():
            if section == 'by_genre' or section == 'by_language':
                formatted_response[section] = {}
                for subsection, subcontent in content_list.items():
                    formatted_response[section][subsection] = RegionalPriorityService._format_content_list(subcontent)
            else:
                formatted_response[section] = RegionalPriorityService._format_content_list(content_list)
        
        return jsonify(formatted_response), 200
        
    except Exception as e:
        logger.error(f"Homepage error: {e}")
        return jsonify({'error': 'Failed to load homepage'}), 500

# Enhanced Search with Language Support
@app.route('/api/search', methods=['GET'])
def enhanced_search():
    try:
        query = request.args.get('query', '')
        content_type = request.args.get('type', 'multi')
        language = request.args.get('language', '')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        # Record search interaction
        session_id = get_session_id()
        
        # Search with language preference
        results = []
        
        # TMDB search
        tmdb_results = TMDBService.search_content(query, content_type, page=page)
        
        if tmdb_results:
            for item in tmdb_results.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                
                if content:
                    # Record search interaction
                    interaction = AnonymousInteraction(
                        session_id=session_id,
                        content_id=content.id,
                        interaction_type='search',
                        ip_address=request.remote_addr
                    )
                    db.session.add(interaction)
                    
                    # Check language match
                    content_languages = json.loads(content.languages or '[]')
                    language_match = not language or any(
                        language.lower() in lang.lower() for lang in content_languages
                    )
                    
                    # Get language-specific streaming options
                    streaming_by_language = {}
                    if language:
                        lang_platforms = ContentService.get_language_specific_availability(content.id, language)
                        streaming_by_language[language] = lang_platforms
                    
                    results.append({
                        'id': content.id,
                        'tmdb_id': content.tmdb_id,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'languages': content_languages,
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
                        'overview': content.overview,
                        'language_match': language_match,
                        'streaming_availability': {
                            'all_platforms': json.loads(content.ott_platforms or '[]'),
                            'by_language': streaming_by_language if language else {}
                        }
                    })
        
        # Handle anime search for anime-specific queries
        if content_type in ['anime', 'multi'] or 'anime' in query.lower():
            anime_results = JikanService.search_anime(query, page=page)
            if anime_results:
                for anime in anime_results.get('data', []):
                    results.append({
                        'id': f"anime_{anime['mal_id']}",
                        'mal_id': anime['mal_id'],
                        'title': anime.get('title'),
                        'content_type': 'anime',
                        'genres': [genre['name'] for genre in anime.get('genres', [])],
                        'languages': ['japanese'],
                        'rating': anime.get('score'),
                        'release_date': anime.get('aired', {}).get('from'),
                        'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                        'overview': anime.get('synopsis'),
                        'language_match': language.lower() == 'japanese' if language else True,
                        'streaming_availability': {
                            'all_platforms': [
                                {
                                    'platform_id': 'crunchyroll',
                                    'platform_name': 'Crunchyroll',
                                    'is_free': True,
                                    'icon': '🟠',
                                    'watch_url': f"https://crunchyroll.com/search?q={anime.get('title', '')}",
                                    'availability': 'Available Now'
                                }
                            ],
                            'by_language': {}
                        }
                    })
        
        # Sort results by language match if language specified
        if language:
            results.sort(key=lambda x: x['language_match'], reverse=True)
        
        db.session.commit()
        
        return jsonify({
            'results': results,
            'total_results': len(results),
            'current_page': page,
            'language_filter': language
        }), 200
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

# Enhanced Content Details with Language-Specific Streaming
@app.route('/api/content/<int:content_id>', methods=['GET'])
def get_enhanced_content_details(content_id):
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
        
        # Get additional details from TMDB
        additional_details = None
        if content.tmdb_id:
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
            
            # Update streaming availability with real-time data
            if additional_details:
                imdb_id = additional_details.get('external_ids', {}).get('imdb_id') or content.imdb_id
                updated_platforms = ContentService.get_ott_availability(additional_details, imdb_id)
                
                # Update content with latest platform data
                content.ott_platforms = json.dumps(updated_platforms)
                content.imdb_id = imdb_id
                db.session.commit()
        
        # Get streaming platforms by language
        languages_available = json.loads(content.languages or '[]')
        streaming_by_language = {}
        
        for language in languages_available:
            lang_platforms = ContentService.get_language_specific_availability(content.id, language)
            if lang_platforms:
                streaming_by_language[language] = {
                    'free_platforms': [p for p in lang_platforms if p.get('is_free', False)],
                    'paid_platforms': [p for p in lang_platforms if not p.get('is_free', False)]
                }
        
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
            
            # Enhanced streaming information
            'streaming_availability': {
                'all_platforms': json.loads(content.ott_platforms or '[]'),
                'by_language': streaming_by_language,
                'free_platforms': [p for p in json.loads(content.ott_platforms or '[]') if p.get('is_free', False)],
                'paid_platforms': [p for p in json.loads(content.ott_platforms or '[]') if not p.get('is_free', False)]
            },
            
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Enhanced content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Fixed Anime Details Endpoint
@app.route('/api/anime/<int:anime_id>', methods=['GET'])
def get_anime_details(anime_id):
    try:
        # Get anime details from Jikan API
        anime_data = JikanService.get_anime_details(anime_id)
        
        if not anime_data:
            return jsonify({'error': 'Anime not found'}), 404
        
        # Record view interaction
        session_id = get_session_id()
        
        # Convert to our content format and save
        content = Content.query.filter_by(tmdb_id=f"anime_{anime_id}").first()
        
        if not content:
            content = Content(
                tmdb_id=f"anime_{anime_id}",
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps([genre['name'] for genre in anime_data.get('genres', [])]),
                languages=json.dumps(['japanese']),
                rating=anime_data.get('score'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                ott_platforms=json.dumps([{
                    'platform_id': 'crunchyroll',
                    'platform_name': 'Crunchyroll',
                    'is_free': True,
                    'icon': '🟠',
                    'priority': 16,
                    'watch_url': f"https://crunchyroll.com/search?q={anime_data.get('title', '')}",
                    'audios': ['japanese'],
                    'subtitles': ['english'],
                    'availability': 'Available Now'
                }])
            )
            db.session.add(content)
            db.session.commit()
        
        # Record interaction
        interaction = AnonymousInteraction(
            session_id=session_id,
            content_id=content.id,
            interaction_type='view',
            ip_address=request.remote_addr
        )
        db.session.add(interaction)
        db.session.commit()
        
        # Get streaming platforms for anime
        anime_platforms = json.loads(content.ott_platforms or '[]')
        
        return jsonify({
            'id': content.id,
            'mal_id': anime_id,
            'title': anime_data.get('title'),
            'title_japanese': anime_data.get('title_japanese'),
            'content_type': 'anime',
            'genres': [genre['name'] for genre in anime_data.get('genres', [])],
            'rating': anime_data.get('score'),
            'episodes': anime_data.get('episodes'),
            'status': anime_data.get('status'),
            'overview': anime_data.get('synopsis'),
            'poster_path': anime_data.get('images', {}).get('jpg', {}).get('image_url'),
            'streaming_availability': {
                'all_platforms': anime_platforms,
                'by_language': {
                    'japanese': {
                        'free_platforms': [p for p in anime_platforms if p.get('is_free', False)],
                        'paid_platforms': [p for p in anime_platforms if not p.get('is_free', False)]
                    }
                },
                'free_platforms': [p for p in anime_platforms if p.get('is_free', False)],
                'paid_platforms': [p for p in anime_platforms if not p.get('is_free', False)]
            },
            'aired': anime_data.get('aired', {}),
            'studios': [studio['name'] for studio in anime_data.get('studios', [])],
            'languages': ['japanese']
        }), 200
        
    except Exception as e:
        logger.error(f"Anime details error: {e}")
        return jsonify({'error': 'Failed to get anime details'}), 500

# Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = RegionalPriorityService._format_content_list(recommendations)
        
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
        
        result = RegionalPriorityService._format_content_list(recommendations)
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Popular by genre error: {e}")
        return jsonify({'error': 'Failed to get popular recommendations'}), 500

@app.route('/api/recommendations/regional/<language>', methods=['GET'])
def get_regional(language):
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_regional_recommendations(language, limit)
        
        result = RegionalPriorityService._format_content_list(recommendations)
        
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
                'streaming_availability': {
                    'all_platforms': json.loads(content.ott_platforms or '[]'),
                    'by_language': {
                        'japanese': {
                            'free_platforms': json.loads(content.ott_platforms or '[]'),
                            'paid_platforms': []
                        }
                    }
                }
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
        
        result = RegionalPriorityService._format_content_list(recommendations)
        
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
        
        result = RegionalPriorityService._format_content_list(contents)
        
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
        
        result = RegionalPriorityService._format_content_list(contents)
        
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
                existing_content = Content.query.filter_by(tmdb_id=f"anime_{data['id']}").first()
            else:
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
            
            # Set tmdb_id based on source
            tmdb_id = data.get('id')
            if data.get('source') == 'anime':
                tmdb_id = f"anime_{data['id']}"
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_id,
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
                # Get streaming info
                platforms = json.loads(content.ott_platforms or '[]')
                languages = json.loads(content.languages or '[]')
                
                # Create language-specific watch buttons
                watch_buttons = {}
                for language in languages:
                    lang_platforms = ContentService.get_language_specific_availability(content.id, language)
                    if lang_platforms:
                        watch_buttons[language] = {
                            'free_platforms': [p for p in lang_platforms if p.get('is_free', False)],
                            'paid_platforms': [p for p in lang_platforms if not p.get('is_free', False)]
                        }
                
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': languages,
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'watch_buttons': watch_buttons,
                    'streaming_availability': {
                        'all_platforms': platforms,
                        'free_platforms': [p for p in platforms if p.get('is_free', False)],
                        'paid_platforms': [p for p in platforms if not p.get('is_free', False)]
                    },
                    'admin_description': rec.description,
                    'admin_name': admin.username if admin else 'Admin',
                    'recommended_at': rec.created_at.isoformat()
                })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Public admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get admin recommendations'}), 500

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
                        formatted_content = RegionalPriorityService._format_content_list([content])[0]
                        formatted_content.update({
                            'recommendation_score': rec.get('score', 0),
                            'recommendation_reason': rec.get('reason', '')
                        })
                        result.append(formatted_content)
                
                return jsonify({'recommendations': result}), 200
        except:
            pass
        
        # Fallback to basic recommendations
        return get_trending()
        
    except Exception as e:
        logger.error(f"Personalized recommendations error: {e}")
        return get_trending()

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'features': {
            'streaming_integration': True,
            'language_support': True,
            'regional_priority': True,
            'anime_support': True,
            'telegram_integration': bool(bot and TELEGRAM_CHANNEL_ID)
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