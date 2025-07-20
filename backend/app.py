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

# ✅ NEW: Streaming API Keys
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
    ott_platforms = db.Column(db.Text)  # JSON string
    streaming_links = db.Column(db.Text)  # ✅ NEW: JSON string for language-specific links
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

# ✅ ENHANCED: OTT Platform Information with Indian platforms
OTT_PLATFORMS = {
    # International Platforms
    'netflix': {'name': 'Netflix', 'is_free': False, 'url': 'https://netflix.com', 'logo': '🔴'},
    'amazon_prime': {'name': 'Amazon Prime Video', 'is_free': False, 'url': 'https://primevideo.com', 'logo': '🟦'},
    'disney_plus': {'name': 'Disney+ Hotstar', 'is_free': False, 'url': 'https://hotstar.com', 'logo': '🌟'},
    'youtube': {'name': 'YouTube', 'is_free': True, 'url': 'https://youtube.com', 'logo': '📺'},
    
    # Indian Free Platforms
    'jiocinema': {'name': 'JioCinema', 'is_free': True, 'url': 'https://jiocinema.com', 'logo': '🎬'},
    'mx_player': {'name': 'MX Player', 'is_free': True, 'url': 'https://mxplayer.in', 'logo': '🎭'},
    'voot': {'name': 'Voot', 'is_free': True, 'url': 'https://voot.com', 'logo': '🎪'},
    'sonyliv_free': {'name': 'SonyLIV Free', 'is_free': True, 'url': 'https://sonyliv.com', 'logo': '📱'},
    'airtel_xstream': {'name': 'Airtel Xstream', 'is_free': True, 'url': 'https://airtelxstream.in', 'logo': '📡'},
    'hungama': {'name': 'Hungama Play', 'is_free': True, 'url': 'https://hungama.com', 'logo': '🎵'},
    
    # Indian Paid Platforms
    'zee5': {'name': 'ZEE5', 'is_free': False, 'url': 'https://zee5.com', 'logo': '⚡'},
    'sonyliv': {'name': 'SonyLIV Premium', 'is_free': False, 'url': 'https://sonyliv.com', 'logo': '🏆'},
    'alt_balaji': {'name': 'ALTBalaji', 'is_free': False, 'url': 'https://altbalaji.com', 'logo': '🎯'},
    'aha': {'name': 'Aha', 'is_free': False, 'url': 'https://aha.video', 'logo': '🎊'},
    'sun_nxt': {'name': 'Sun NXT', 'is_free': False, 'url': 'https://sunnxt.com', 'logo': '☀️'},
    'hoichoi': {'name': 'Hoichoi', 'is_free': False, 'url': 'https://hoichoi.tv', 'logo': '🎨'},
    'addatimes': {'name': 'Addatimes', 'is_free': False, 'url': 'https://addatimes.com', 'logo': '⏰'}
}

# ✅ ENHANCED: Regional Language Mapping
REGIONAL_LANGUAGES = {
    'hindi': {'codes': ['hi', 'hindi', 'bollywood'], 'name': 'Hindi', 'flag': '🇮🇳'},
    'telugu': {'codes': ['te', 'telugu', 'tollywood'], 'name': 'Telugu', 'flag': '🎭'},
    'tamil': {'codes': ['ta', 'tamil', 'kollywood'], 'name': 'Tamil', 'flag': '🎬'},
    'kannada': {'codes': ['kn', 'kannada', 'sandalwood'], 'name': 'Kannada', 'flag': '🎪'},
    'malayalam': {'codes': ['ml', 'malayalam', 'mollywood'], 'name': 'Malayalam', 'flag': '🌴'},
    'english': {'codes': ['en', 'english', 'hollywood'], 'name': 'English', 'flag': '🇺🇸'},
    'bengali': {'codes': ['bn', 'bengali', 'tollygunge'], 'name': 'Bengali', 'flag': '🐟'},
    'marathi': {'codes': ['mr', 'marathi'], 'name': 'Marathi', 'flag': '🦁'},
    'gujarati': {'codes': ['gu', 'gujarati'], 'name': 'Gujarati', 'flag': '🪔'},
    'punjabi': {'codes': ['pa', 'punjabi'], 'name': 'Punjabi', 'flag': '👳'}
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

# ✅ NEW: Streaming Services
class WatchModeService:
    BASE_URL = 'https://api.watchmode.com/v1'
    
    @staticmethod
    def search_title(title, content_type='movie'):
        """Search for a title and get streaming info"""
        try:
            url = f"{WatchModeService.BASE_URL}/search/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'search_field': 'name',
                'search_value': title,
                'types': content_type if content_type != 'tv' else 'tv_series'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('title_results'):
                    return data['title_results'][0]  # Return first match
        except Exception as e:
            logger.error(f"WatchMode search error: {e}")
        return None
    
    @staticmethod
    def get_streaming_sources(watchmode_id):
        """Get streaming sources for a title"""
        try:
            url = f"{WatchModeService.BASE_URL}/title/{watchmode_id}/sources/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'regions': 'IN,US'  # India and US
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
    def search_content(title, content_type='movie'):
        """Search for content on streaming platforms"""
        try:
            url = f"{StreamingAvailabilityService.BASE_URL}/search/title"
            headers = {
                'X-RapidAPI-Key': RAPIDAPI_KEY,
                'X-RapidAPI-Host': RAPIDAPI_HOST
            }
            params = {
                'title': title,
                'country': 'in',  # India
                'show_type': content_type if content_type != 'tv' else 'series',
                'output_language': 'en'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('result'):
                    return data['result'][0]  # Return first match
        except Exception as e:
            logger.error(f"Streaming Availability search error: {e}")
        return None
    
    @staticmethod
    def get_streaming_info(imdb_id):
        """Get streaming info by IMDB ID"""
        try:
            url = f"{StreamingAvailabilityService.BASE_URL}/get"
            headers = {
                'X-RapidAPI-Key': RAPIDAPI_KEY,
                'X-RapidAPI-Host': RAPIDAPI_HOST
            }
            params = {
                'imdb_id': imdb_id,
                'country': 'in'  # India
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Streaming Availability get error: {e}")
        return None

class EnhancedStreamingService:
    """Enhanced streaming service that combines multiple APIs"""
    
    @staticmethod
    def get_comprehensive_streaming_info(title, content_type='movie', imdb_id=None):
        """Get comprehensive streaming information from multiple sources"""
        streaming_info = {
            'platforms': [],
            'language_links': {},
            'free_platforms': [],
            'paid_platforms': []
        }
        
        try:
            # Try WatchMode API first
            watchmode_result = WatchModeService.search_title(title, content_type)
            if watchmode_result:
                watchmode_sources = WatchModeService.get_streaming_sources(watchmode_result.get('id'))
                if watchmode_sources:
                    streaming_info = EnhancedStreamingService._process_watchmode_data(watchmode_sources, streaming_info)
            
            # Try Streaming Availability API
            if imdb_id:
                sa_result = StreamingAvailabilityService.get_streaming_info(imdb_id)
                if sa_result:
                    streaming_info = EnhancedStreamingService._process_streaming_availability_data(sa_result, streaming_info)
            
            # Add fallback Indian platforms based on content analysis
            streaming_info = EnhancedStreamingService._add_indian_platform_suggestions(title, content_type, streaming_info)
            
        except Exception as e:
            logger.error(f"Enhanced streaming service error: {e}")
        
        return streaming_info
    
    @staticmethod
    def _process_watchmode_data(sources_data, streaming_info):
        """Process WatchMode API response"""
        try:
            for source in sources_data:
                platform_name = source.get('name', '').lower()
                platform_type = source.get('type')
                web_url = source.get('web_url')
                
                # Map to our platform names
                mapped_platform = EnhancedStreamingService._map_platform_name(platform_name)
                if mapped_platform and web_url:
                    platform_info = {
                        'platform': mapped_platform,
                        'name': OTT_PLATFORMS.get(mapped_platform, {}).get('name', platform_name),
                        'url': web_url,
                        'is_free': platform_type == 'free' or OTT_PLATFORMS.get(mapped_platform, {}).get('is_free', False),
                        'logo': OTT_PLATFORMS.get(mapped_platform, {}).get('logo', '📺'),
                        'type': platform_type
                    }
                    
                    streaming_info['platforms'].append(platform_info)
                    
                    if platform_info['is_free']:
                        streaming_info['free_platforms'].append(platform_info)
                    else:
                        streaming_info['paid_platforms'].append(platform_info)
        except Exception as e:
            logger.error(f"Error processing WatchMode data: {e}")
        
        return streaming_info
    
    @staticmethod
    def _process_streaming_availability_data(sa_data, streaming_info):
        """Process Streaming Availability API response"""
        try:
            streaming_info_sa = sa_data.get('streamingInfo', {})
            
            for country, platforms in streaming_info_sa.items():
                if country == 'in':  # Focus on India
                    for platform_key, platform_data in platforms.items():
                        mapped_platform = EnhancedStreamingService._map_platform_name(platform_key)
                        if mapped_platform:
                            for stream_data in platform_data:
                                platform_info = {
                                    'platform': mapped_platform,
                                    'name': OTT_PLATFORMS.get(mapped_platform, {}).get('name', platform_key),
                                    'url': stream_data.get('link', ''),
                                    'is_free': stream_data.get('streamingType') == 'free',
                                    'logo': OTT_PLATFORMS.get(mapped_platform, {}).get('logo', '📺'),
                                    'quality': stream_data.get('quality', ''),
                                    'audio_languages': stream_data.get('audios', [])
                                }
                                
                                streaming_info['platforms'].append(platform_info)
                                
                                # Add language-specific links
                                for audio in stream_data.get('audios', []):
                                    lang_code = audio.get('language')
                                    if lang_code:
                                        lang_name = EnhancedStreamingService._get_language_name(lang_code)
                                        if lang_name not in streaming_info['language_links']:
                                            streaming_info['language_links'][lang_name] = []
                                        
                                        streaming_info['language_links'][lang_name].append({
                                            'platform': platform_info['name'],
                                            'url': platform_info['url'],
                                            'logo': platform_info['logo'],
                                            'is_free': platform_info['is_free']
                                        })
        except Exception as e:
            logger.error(f"Error processing Streaming Availability data: {e}")
        
        return streaming_info
    
    @staticmethod
    def _add_indian_platform_suggestions(title, content_type, streaming_info):
        """Add likely Indian platform suggestions based on content analysis"""
        try:
            title_lower = title.lower()
            
            # Hindi content suggestions
            if any(word in title_lower for word in ['bollywood', 'hindi', 'shah rukh', 'salman', 'aamir', 'akshay']):
                suggested_platforms = ['jiocinema', 'zee5', 'sonyliv', 'voot', 'mx_player']
                EnhancedStreamingService._add_suggested_platforms(suggested_platforms, streaming_info, 'hindi')
            
            # Telugu content suggestions
            elif any(word in title_lower for word in ['tollywood', 'telugu', 'prabhas', 'allu arjun', 'mahesh babu']):
                suggested_platforms = ['aha', 'zee5', 'sun_nxt', 'mx_player']
                EnhancedStreamingService._add_suggested_platforms(suggested_platforms, streaming_info, 'telugu')
            
            # Tamil content suggestions
            elif any(word in title_lower for word in ['kollywood', 'tamil', 'rajinikanth', 'kamal', 'vijay', 'ajith']):
                suggested_platforms = ['sun_nxt', 'zee5', 'hotstar', 'mx_player']
                EnhancedStreamingService._add_suggested_platforms(suggested_platforms, streaming_info, 'tamil')
            
            # Malayalam content suggestions
            elif any(word in title_lower for word in ['malayalam', 'mollywood', 'mohanlal', 'mammootty']):
                suggested_platforms = ['sun_nxt', 'zee5', 'mx_player']
                EnhancedStreamingService._add_suggested_platforms(suggested_platforms, streaming_info, 'malayalam')
            
            # Kannada content suggestions
            elif any(word in title_lower for word in ['kannada', 'sandalwood', 'yash', 'sudeep']):
                suggested_platforms = ['zee5', 'sun_nxt', 'voot', 'mx_player']
                EnhancedStreamingService._add_suggested_platforms(suggested_platforms, streaming_info, 'kannada')
            
            # General Indian content
            else:
                suggested_platforms = ['netflix', 'amazon_prime', 'disney_plus', 'jiocinema', 'mx_player']
                EnhancedStreamingService._add_suggested_platforms(suggested_platforms, streaming_info, 'hindi')
        
        except Exception as e:
            logger.error(f"Error adding Indian platform suggestions: {e}")
        
        return streaming_info
    
    @staticmethod
    def _add_suggested_platforms(platforms, streaming_info, primary_language):
        """Add suggested platforms to streaming info"""
        for platform_key in platforms:
            if platform_key in OTT_PLATFORMS:
                platform_info = OTT_PLATFORMS[platform_key].copy()
                platform_info['platform'] = platform_key
                platform_info['suggested'] = True
                
                # Check if platform already exists
                existing = next((p for p in streaming_info['platforms'] if p['platform'] == platform_key), None)
                if not existing:
                    streaming_info['platforms'].append(platform_info)
                    
                    if platform_info['is_free']:
                        streaming_info['free_platforms'].append(platform_info)
                    else:
                        streaming_info['paid_platforms'].append(platform_info)
                
                # Add to language links
                if primary_language not in streaming_info['language_links']:
                    streaming_info['language_links'][primary_language] = []
                
                streaming_info['language_links'][primary_language].append({
                    'platform': platform_info['name'],
                    'url': platform_info['url'],
                    'logo': platform_info['logo'],
                    'is_free': platform_info['is_free'],
                    'suggested': True
                })
    
    @staticmethod
    def _map_platform_name(platform_name):
        """Map external platform names to our internal platform keys"""
        platform_mapping = {
            'netflix': 'netflix',
            'amazon prime video': 'amazon_prime',
            'prime video': 'amazon_prime',
            'disney plus': 'disney_plus',
            'disney+ hotstar': 'disney_plus',
            'hotstar': 'disney_plus',
            'youtube': 'youtube',
            'jiocinema': 'jiocinema',
            'mx player': 'mx_player',
            'zee5': 'zee5',
            'sonyliv': 'sonyliv',
            'voot': 'voot',
            'alt balaji': 'alt_balaji',
            'aha': 'aha',
            'sun nxt': 'sun_nxt',
            'hoichoi': 'hoichoi'
        }
        
        platform_lower = platform_name.lower().strip()
        return platform_mapping.get(platform_lower)
    
    @staticmethod
    def _get_language_name(lang_code):
        """Get language name from language code"""
        lang_mapping = {
            'hi': 'hindi',
            'te': 'telugu', 
            'ta': 'tamil',
            'ml': 'malayalam',
            'kn': 'kannada',
            'en': 'english',
            'bn': 'bengali',
            'mr': 'marathi',
            'gu': 'gujarati',
            'pa': 'punjabi'
        }
        
        return lang_mapping.get(lang_code.lower(), lang_code)

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
            'append_to_response': 'credits,videos,similar,watch/providers,external_ids'  # ✅ Added external_ids
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

# ✅ ENHANCED: Content Management Service
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
                # Map genre IDs to names (you'll need a genre mapping)
                genres = ContentService.map_genre_ids(tmdb_data['genre_ids'])
            
            # Extract languages
            languages = []
            if 'spoken_languages' in tmdb_data:
                languages = [lang['name'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
            
            # ✅ ENHANCED: Get comprehensive streaming availability
            title = tmdb_data.get('title') or tmdb_data.get('name')
            imdb_id = None
            if 'external_ids' in tmdb_data:
                imdb_id = tmdb_data['external_ids'].get('imdb_id')
            
            streaming_info = EnhancedStreamingService.get_comprehensive_streaming_info(
                title, content_type, imdb_id
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
                ott_platforms=json.dumps(streaming_info['platforms']),
                streaming_links=json.dumps(streaming_info['language_links'])  # ✅ NEW
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
    def get_ott_availability(tmdb_data):
        # This method is now enhanced and handled by EnhancedStreamingService
        # Keep for backward compatibility
        title = tmdb_data.get('title') or tmdb_data.get('name', '')
        content_type = 'movie' if 'title' in tmdb_data else 'tv'
        
        streaming_info = EnhancedStreamingService.get_comprehensive_streaming_info(title, content_type)
        return streaming_info['platforms']

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
                    streaming_links=json.dumps({})  # ✅ NEW
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

# ✅ ENHANCED: Telegram Service with Streaming Links
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
            
            # ✅ NEW: Get streaming links
            streaming_links_text = TelegramService._format_streaming_links(content)
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # ✅ ENHANCED: Create message with streaming links
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

{streaming_links_text}

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
    def _format_streaming_links(content):
        """Format streaming links for Telegram message"""
        try:
            streaming_text = "\n📺 **Available to Watch:**\n"
            
            # Get language-specific links
            language_links = {}
            if content.streaming_links:
                try:
                    language_links = json.loads(content.streaming_links)
                except:
                    language_links = {}
            
            # Get general platform links
            platforms = []
            if content.ott_platforms:
                try:
                    platforms = json.loads(content.ott_platforms)
                except:
                    platforms = []
            
            # Format language-specific links first
            if language_links:
                for language, links in language_links.items():
                    lang_info = REGIONAL_LANGUAGES.get(language.lower(), {})
                    lang_name = lang_info.get('name', language.title())
                    flag = lang_info.get('flag', '🎭')
                    
                    # Show only first 2 platforms per language to avoid clutter
                    for link in links[:2]:
                        platform_name = link.get('platform', 'Unknown')
                        platform_url = link.get('url', '#')
                        logo = link.get('logo', '📺')
                        is_free = link.get('is_free', False)
                        free_text = " (Free)" if is_free else ""
                        
                        streaming_text += f"🔘 Watch in {lang_name} {flag} → [{platform_name}{free_text}]({platform_url})\n"
            
            # If no language-specific links, show general platform links
            elif platforms:
                streaming_text += "🔘 Available on:\n"
                
                # Separate free and paid platforms
                free_platforms = [p for p in platforms if p.get('is_free', False)]
                paid_platforms = [p for p in platforms if not p.get('is_free', False)]
                
                # Show free platforms first
                for platform in free_platforms[:3]:
                    name = platform.get('name', 'Unknown')
                    url = platform.get('url', '#')
                    logo = platform.get('logo', '📺')
                    streaming_text += f"   {logo} [{name} (Free)]({url})\n"
                
                # Then paid platforms
                for platform in paid_platforms[:2]:
                    name = platform.get('name', 'Unknown')
                    url = platform.get('url', '#')
                    logo = platform.get('logo', '📺')
                    streaming_text += f"   {logo} [{name}]({url})\n"
            
            else:
                streaming_text += "🔍 Check your local streaming platforms\n"
            
            return streaming_text
            
        except Exception as e:
            logger.error(f"Error formatting streaming links: {e}")
            return "\n📺 **Streaming info unavailable**\n"

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
                    
                    # ✅ ENHANCED: Include streaming info in search results
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
                        'streaming_links': json.loads(content.streaming_links or '{}')  # ✅ NEW
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
                    'streaming_links': {}  # ✅ NEW
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

# ✅ ENHANCED: Content Details Route with Streaming Info
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
        
        # ✅ NEW: Refresh streaming info if needed (cache for 24 hours)
        should_refresh_streaming = (
            datetime.utcnow() - content.updated_at > timedelta(hours=24) or
            not content.streaming_links
        )
        
        if should_refresh_streaming and content.title:
            try:
                # Get fresh streaming info
                streaming_info = EnhancedStreamingService.get_comprehensive_streaming_info(
                    content.title, content.content_type, content.imdb_id
                )
                
                # Update content with fresh streaming data
                content.ott_platforms = json.dumps(streaming_info['platforms'])
                content.streaming_links = json.dumps(streaming_info['language_links'])
                content.updated_at = datetime.utcnow()
                db.session.commit()
            except Exception as e:
                logger.error(f"Error refreshing streaming info: {e}")
        
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
                        'rating': similar.rating,
                        'streaming_links': json.loads(similar.streaming_links or '{}')  # ✅ NEW
                    })
        
        # ✅ NEW: Format language-specific streaming links
        formatted_streaming_links = []
        streaming_links = json.loads(content.streaming_links or '{}')
        
        for language, links in streaming_links.items():
            lang_info = REGIONAL_LANGUAGES.get(language.lower(), {})
            formatted_links = []
            
            for link in links:
                formatted_links.append({
                    'platform': link.get('platform'),
                    'url': link.get('url'),
                    'logo': link.get('logo', '📺'),
                    'is_free': link.get('is_free', False),
                    'suggested': link.get('suggested', False)
                })
            
            formatted_streaming_links.append({
                'language': language,
                'language_name': lang_info.get('name', language.title()),
                'language_flag': lang_info.get('flag', '🎭'),
                'links': formatted_links
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
            'ott_platforms': json.loads(content.ott_platforms or '[]'),
            'streaming_links': formatted_streaming_links,  # ✅ NEW: Language-specific links
            'raw_streaming_links': json.loads(content.streaming_links or '{}'),  # ✅ NEW: Raw data for frontend
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
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
                'streaming_links': json.loads(content.streaming_links or '{}')  # ✅ NEW
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
                'streaming_links': json.loads(content.streaming_links or '{}')  # ✅ NEW
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
                'streaming_links': json.loads(content.streaming_links or '{}')  # ✅ NEW
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
                'streaming_links': json.loads(content.streaming_links or '{}')  # ✅ NEW
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
                'streaming_links': json.loads(content.streaming_links or '{}')  # ✅ NEW
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
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'ott_platforms': json.loads(content.ott_platforms or '[]'),
                            'streaming_links': json.loads(content.streaming_links or '{}'),  # ✅ NEW
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
                'streaming_links': json.loads(content.streaming_links or '{}')  # ✅ NEW
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
                'streaming_links': json.loads(content.streaming_links or '{}')  # ✅ NEW
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
            
            # ✅ ENHANCED: Get streaming info for new content
            title = data.get('title', '')
            content_type = data.get('content_type', 'movie')
            
            streaming_info = EnhancedStreamingService.get_comprehensive_streaming_info(title, content_type)
            
            # Create content object
            content = Content(
                tmdb_id=data.get('id'),
                title=title,
                original_title=data.get('original_title'),
                content_type=content_type,
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
                ott_platforms=json.dumps(streaming_info['platforms']),  # ✅ ENHANCED
                streaming_links=json.dumps(streaming_info['language_links'])  # ✅ NEW
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
        
        # ✅ ENHANCED: Send to Telegram channel with streaming links
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
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'streaming_links': json.loads(content.streaming_links or '{}')  # ✅ NEW
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
        
        # ✅ NEW: Streaming platform analytics
        platform_counts = defaultdict(int)
        all_content = Content.query.all()
        for content in all_content:
            if content.ott_platforms:
                try:
                    platforms = json.loads(content.ott_platforms)
                    for platform in platforms:
                        platform_name = platform.get('name', 'Unknown')
                        platform_counts[platform_name] += 1
                except:
                    pass
        
        popular_platforms = sorted(platform_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
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
            'popular_platforms': [  # ✅ NEW
                {'platform': platform, 'count': count}
                for platform, count in popular_platforms
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
                    'streaming_links': json.loads(content.streaming_links or '{}'),  # ✅ NEW
                    'admin_description': rec.description,
                    'admin_name': admin.username if admin else 'Admin',
                    'recommended_at': rec.created_at.isoformat()
                })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Public admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get admin recommendations'}), 500

# ✅ NEW: Streaming Platform Routes
@app.route('/api/streaming/platforms', methods=['GET'])
def get_streaming_platforms():
    """Get list of all supported streaming platforms"""
    try:
        platforms = []
        for key, info in OTT_PLATFORMS.items():
            platforms.append({
                'key': key,
                'name': info['name'],
                'is_free': info['is_free'],
                'logo': info['logo'],
                'url': info['url']
            })
        
        return jsonify({'platforms': platforms}), 200
    except Exception as e:
        logger.error(f"Get platforms error: {e}")
        return jsonify({'error': 'Failed to get platforms'}), 500

@app.route('/api/streaming/refresh/<int:content_id>', methods=['POST'])
def refresh_streaming_info(content_id):
    """Manually refresh streaming info for a specific content"""
    try:
        content = Content.query.get_or_404(content_id)
        
        if not content.title:
            return jsonify({'error': 'Content title required for streaming lookup'}), 400
        
        # Get fresh streaming info
        streaming_info = EnhancedStreamingService.get_comprehensive_streaming_info(
            content.title, content.content_type, content.imdb_id
        )
        
        # Update content
        content.ott_platforms = json.dumps(streaming_info['platforms'])
        content.streaming_links = json.dumps(streaming_info['language_links'])
        content.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Streaming info refreshed successfully',
            'platforms': streaming_info['platforms'],
            'language_links': streaming_info['language_links']
        }), 200
        
    except Exception as e:
        logger.error(f"Refresh streaming info error: {e}")
        return jsonify({'error': 'Failed to refresh streaming info'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'features': {
            'streaming_apis': {
                'watchmode': bool(WATCHMODE_API_KEY),
                'rapidapi': bool(RAPIDAPI_KEY)
            }
        }
    }), 200

# ✅ NEW: Details route for frontend compatibility
@app.route('/details')
def details_page():
    """Route for details.html frontend page"""
    return render_template('details.html')

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