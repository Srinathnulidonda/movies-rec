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
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import re

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Database configuration
if os.environ.get('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_recommendations.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# API Keys - Updated to use WatchMode instead of JustWatch
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', 'your_tmdb_api_key')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', 'your_omdb_api_key')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'your_youtube_api_key')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'your_telegram_bot_token')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', 'your_channel_id')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'http://localhost:5001')
WATCHMODE_API_KEY = os.environ.get('WATCHMODE_API_KEY', 'your_watchmode_api_key')

# WatchMode API Configuration (using RapidAPI)
RAPIDAPI_KEY = "c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37"
RAPIDAPI_HOST = "streaming-availability.p.rapidapi.com"

# Website URL for Telegram links
WEBSITE_URL = os.environ.get('WEBSITE_URL', 'http://recommendationwebsite.com')

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

# Language Priority Configuration
LANGUAGE_PRIORITY = {
    'telugu': 1,
    'english': 1,
    'hindi': 2,
    'tamil': 3,
    'malayalam': 4,
    'kannada': 5
}

# Language Mapping for Display
LANGUAGE_DISPLAY = {
    'hindi': {'name': 'Hindi', 'flag': '🇮🇳', 'code': 'hi'},
    'telugu': {'name': 'Telugu', 'flag': '📺', 'code': 'te'},
    'tamil': {'name': 'Tamil', 'flag': '🎭', 'code': 'ta'},
    'malayalam': {'name': 'Malayalam', 'flag': '🌴', 'code': 'ml'},
    'kannada': {'name': 'Kannada', 'flag': '🎪', 'code': 'kn'},
    'english': {'name': 'English', 'flag': '🇺🇸', 'code': 'en'}
}

# Database Models (same as before but with enhanced fields)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    preferred_languages = db.Column(db.Text)
    preferred_genres = db.Column(db.Text)
    location = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    imdb_id = db.Column(db.String(20))
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    content_type = db.Column(db.String(20), nullable=False)
    genres = db.Column(db.Text)
    languages = db.Column(db.Text)
    audio_languages = db.Column(db.Text)  # New field for audio language availability
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float)
    vote_count = db.Column(db.Integer)
    popularity = db.Column(db.Float)
    overview = db.Column(db.Text)
    poster_path = db.Column(db.String(255))
    backdrop_path = db.Column(db.String(255))
    trailer_url = db.Column(db.String(255))
    ott_platforms = db.Column(db.Text)
    language_availability = db.Column(db.Text)  # New field for language-specific availability
    youtube_availability = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AdminRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recommendation_type = db.Column(db.String(50))
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

# Enhanced OTT Platform Information with Language Support
OTT_PLATFORMS = {
    'netflix': {
        'name': 'Netflix',
        'is_free': False,
        'base_url': 'https://netflix.com',
        'deep_link_pattern': 'https://netflix.com/title/{id}',
        'language_support': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada'],
        'audio_language_support': True,
        'subtitle_support': True
    },
    'amazon_prime': {
        'name': 'Prime Video',
        'is_free': False,
        'base_url': 'https://primevideo.com',
        'deep_link_pattern': 'https://primevideo.com/detail/{id}',
        'language_support': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada'],
        'audio_language_support': True,
        'subtitle_support': True
    },
    'disney_plus_hotstar': {
        'name': 'Disney+ Hotstar',
        'is_free': False,
        'base_url': 'https://hotstar.com',
        'deep_link_pattern': 'https://hotstar.com/in/movies/{title}/{id}',
        'language_support': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada'],
        'audio_language_support': True,
        'subtitle_support': True
    },
    'youtube': {
        'name': 'YouTube',
        'is_free': True,
        'base_url': 'https://youtube.com',
        'deep_link_pattern': 'https://youtube.com/watch?v={id}',
        'language_support': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada'],
        'audio_language_support': True,
        'subtitle_support': True
    },
    'jiocinema': {
        'name': 'JioCinema',
        'is_free': True,
        'base_url': 'https://jiocinema.com',
        'deep_link_pattern': 'https://jiocinema.com/movies/{title}/{id}',
        'language_support': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada'],
        'audio_language_support': True
    },
    'zee5': {
        'name': 'ZEE5',
        'is_free': False,
        'base_url': 'https://zee5.com',
        'deep_link_pattern': 'https://zee5.com/movies/details/{title}/{id}',
        'language_support': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada'],
        'audio_language_support': True
    },
    'sonyliv': {
        'name': 'SonyLIV',
        'is_free': False,
        'base_url': 'https://sonyliv.com',
        'deep_link_pattern': 'https://sonyliv.com/movies/{title}/{id}',
        'language_support': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada'],
        'audio_language_support': True
    }
}

# OTT Cache
class OTTCache:
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=6)
    
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < self.cache_duration:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, data):
        self.cache[key] = (data, datetime.utcnow())
    
    def clear_expired(self):
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= self.cache_duration
        ]
        for key in expired_keys:
            del self.cache[key]

ott_cache = OTTCache()

# Helper Functions (same as before)
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

# Enhanced WatchMode Service (Replacing JustWatch)
class WatchModeService:
    BASE_URL = 'https://streaming-availability.p.rapidapi.com'
    
    @staticmethod
    async def get_comprehensive_availability(tmdb_id, imdb_id=None, title=None, content_type='movie', region='in'):
        """Get comprehensive availability using WatchMode API via RapidAPI"""
        try:
            headers = {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': RAPIDAPI_HOST
            }
            
            availability_data = {
                'platforms': [],
                'language_availability': {},
                'free_options': [],
                'paid_options': [],
                'last_updated': datetime.utcnow().isoformat(),
                'region': region
            }
            
            # Get by IMDB ID if available (most accurate)
            if imdb_id:
                availability_data = await WatchModeService._get_by_imdb(imdb_id, headers, availability_data, region)
            
            # Search by title if no IMDB ID
            if not availability_data['platforms'] and title:
                availability_data = await WatchModeService._search_by_title(title, headers, availability_data, region, content_type)
            
            # Add language-specific availability
            await WatchModeService._enhance_with_language_data(availability_data, title, tmdb_id)
            
            return availability_data
            
        except Exception as e:
            logger.error(f"WatchMode service error: {e}")
            return {
                'platforms': [],
                'language_availability': {},
                'free_options': [],
                'paid_options': [],
                'last_updated': datetime.utcnow().isoformat(),
                'region': region,
                'error': str(e)
            }
    
    @staticmethod
    async def _get_by_imdb(imdb_id, headers, availability_data, region):
        """Get availability by IMDB ID"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{WatchModeService.BASE_URL}/get"
                params = {
                    'imdb_id': imdb_id,
                    'output_language': 'en',
                    'country': region
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        availability_data = WatchModeService._parse_streaming_data(data, availability_data)
                    else:
                        logger.warning(f"WatchMode API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"WatchMode IMDB lookup error: {e}")
        
        return availability_data
    
    @staticmethod
    async def _search_by_title(title, headers, availability_data, region, content_type):
        """Search availability by title"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{WatchModeService.BASE_URL}/search/title"
                params = {
                    'title': title,
                    'country': region,
                    'show_type': 'movie' if content_type == 'movie' else 'series',
                    'output_language': 'en'
                }
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('result') and len(data['result']) > 0:
                            # Use first result
                            first_result = data['result'][0]
                            availability_data = WatchModeService._parse_streaming_data(first_result, availability_data)
                    else:
                        logger.warning(f"WatchMode search error: {response.status}")
                        
        except Exception as e:
            logger.error(f"WatchMode title search error: {e}")
        
        return availability_data
    
    @staticmethod
    def _parse_streaming_data(data, availability_data):
        """Parse streaming data from WatchMode response"""
        try:
            streaming_options = data.get('streamingOptions', {})
            
            for country, services in streaming_options.items():
                for service_data in services:
                    service_name = service_data.get('service', {}).get('name', '')
                    service_id = service_data.get('service', {}).get('id', '')
                    
                    # Map to our platform IDs
                    platform_id = WatchModeService._map_service_to_platform(service_id, service_name)
                    
                    platform_info = {
                        'platform_id': platform_id,
                        'platform_name': service_name,
                        'watch_url': service_data.get('link', ''),
                        'availability_type': WatchModeService._determine_availability_type(service_data),
                        'is_free': service_data.get('type') == 'free',
                        'price': service_data.get('price', {}).get('amount'),
                        'currency': service_data.get('price', {}).get('currency'),
                        'quality': service_data.get('quality'),
                        'audio_languages': service_data.get('audios', []),
                        'subtitle_languages': service_data.get('subtitles', []),
                        'source': 'watchmode'
                    }
                    
                    availability_data['platforms'].append(platform_info)
                    
                    # Categorize
                    if platform_info['is_free']:
                        availability_data['free_options'].append(platform_info)
                    else:
                        availability_data['paid_options'].append(platform_info)
                    
                    # Language availability
                    for lang in platform_info['audio_languages']:
                        lang_key = WatchModeService._normalize_language(lang)
                        if lang_key not in availability_data['language_availability']:
                            availability_data['language_availability'][lang_key] = []
                        availability_data['language_availability'][lang_key].append(platform_info)
            
            return availability_data
            
        except Exception as e:
            logger.error(f"Error parsing streaming data: {e}")
            return availability_data
    
    @staticmethod
    def _map_service_to_platform(service_id, service_name):
        """Map WatchMode service to our platform IDs"""
        service_mapping = {
            'netflix': 'netflix',
            'prime': 'amazon_prime',
            'hotstar': 'disney_plus_hotstar',
            'youtube': 'youtube',
            'jio': 'jiocinema',
            'zee5': 'zee5',
            'sony': 'sonyliv'
        }
        
        service_lower = service_name.lower()
        for key, value in service_mapping.items():
            if key in service_lower or key in service_id.lower():
                return value
        
        return service_id.lower().replace(' ', '_')
    
    @staticmethod
    def _determine_availability_type(service_data):
        """Determine availability type from service data"""
        service_type = service_data.get('type', '').lower()
        
        if service_type in ['free', 'ads']:
            return 'free'
        elif service_type == 'subscription':
            return 'subscription'
        elif service_type == 'rent':
            return 'rent'
        elif service_type == 'buy':
            return 'buy'
        else:
            return 'subscription'  # default
    
    @staticmethod
    def _normalize_language(language):
        """Normalize language names to our standard format"""
        lang_mapping = {
            'hindi': 'hindi',
            'english': 'english',
            'tamil': 'tamil',
            'telugu': 'telugu',
            'malayalam': 'malayalam',
            'kannada': 'kannada',
            'en': 'english',
            'hi': 'hindi',
            'ta': 'tamil',
            'te': 'telugu',
            'ml': 'malayalam',
            'kn': 'kannada'
        }
        
        lang_lower = language.lower()
        return lang_mapping.get(lang_lower, lang_lower)
    
    @staticmethod
    async def _enhance_with_language_data(availability_data, title, tmdb_id):
        """Enhance with additional language availability data"""
        try:
            # Get YouTube language availability
            youtube_data = await EnhancedYouTubeService.get_comprehensive_youtube_availability(
                title, None, None, 'movie', 'IN'
            )
            
            # Add YouTube language options
            if youtube_data.get('free_movies'):
                for movie in youtube_data['free_movies']:
                    # Try to detect language from title/description
                    detected_languages = WatchModeService._detect_languages_from_video(movie)
                    
                    for lang in detected_languages:
                        lang_key = WatchModeService._normalize_language(lang)
                        
                        youtube_platform = {
                            'platform_id': 'youtube',
                            'platform_name': 'YouTube',
                            'watch_url': movie['watch_url'],
                            'availability_type': 'free',
                            'is_free': True,
                            'quality_score': movie.get('quality_score', 0),
                            'video_title': movie['title'],
                            'channel_name': movie['channel_title'],
                            'language': lang_key,
                            'source': 'youtube'
                        }
                        
                        if lang_key not in availability_data['language_availability']:
                            availability_data['language_availability'][lang_key] = []
                        
                        availability_data['language_availability'][lang_key].append(youtube_platform)
            
        except Exception as e:
            logger.error(f"Error enhancing with language data: {e}")
    
    @staticmethod
    def _detect_languages_from_video(video_data):
        """Detect languages from video title and description"""
        title = video_data.get('title', '').lower()
        description = video_data.get('description', '').lower()
        
        detected_languages = []
        
        # Language keywords
        language_keywords = {
            'hindi': ['hindi', 'हिंदी', 'bollywood', 'हिन्दी'],
            'telugu': ['telugu', 'తెలుగు', 'tollywood', 'తెలుగు'],
            'tamil': ['tamil', 'தமிழ்', 'kollywood', 'தமிழ்'],
            'malayalam': ['malayalam', 'മലയാളം', 'mollywood'],
            'kannada': ['kannada', 'ಕನ್ನಡ', 'sandalwood'],
            'english': ['english', 'hollywood', 'dubbed']
        }
        
        for lang, keywords in language_keywords.items():
            if any(keyword in title or keyword in description for keyword in keywords):
                detected_languages.append(lang)
        
        # Default to English if no specific language detected
        if not detected_languages:
            detected_languages = ['english']
        
        return detected_languages

# Enhanced YouTube Service (keeping existing function names)
class EnhancedYouTubeService:
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    @staticmethod
    async def get_comprehensive_youtube_availability(title, original_title=None, release_year=None, content_type='movie', region='IN'):
        """Get comprehensive YouTube availability including free movies, trailers, and premium content"""
        if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == 'your_youtube_api_key':
            return []
        
        try:
            youtube_data = {
                'free_movies': [],
                'premium_content': [],
                'trailers': [],
                'clips': [],
                'official_content': [],
                'language_specific': {},
                'last_checked': datetime.utcnow().isoformat()
            }
            
            # Multiple search strategies
            search_results = await EnhancedYouTubeService._comprehensive_search(title, original_title, release_year, content_type, region)
            
            # Categorize results
            for video in search_results:
                category = EnhancedYouTubeService._categorize_video(video, title)
                if category:
                    youtube_data[category].append(video)
                
                # Categorize by language
                languages = EnhancedYouTubeService._detect_video_language(video)
                for lang in languages:
                    if lang not in youtube_data['language_specific']:
                        youtube_data['language_specific'][lang] = []
                    youtube_data['language_specific'][lang].append(video)
            
            return youtube_data
            
        except Exception as e:
            logger.error(f"YouTube comprehensive search error: {e}")
            return []
    
    @staticmethod
    def _detect_video_language(video_data):
        """Enhanced language detection for videos"""
        title = video_data.get('title', '').lower()
        description = video_data.get('description', '').lower()
        
        languages = []
        
        # Enhanced language detection patterns
        language_patterns = {
            'hindi': [
                r'\bhindi\b', r'\bहिंदी\b', r'\bbollywood\b', r'\bhindi movie\b',
                r'\bhindi film\b', r'\bhindi cinema\b', r'\bहिन्दी\b'
            ],
            'telugu': [
                r'\btelugu\b', r'\bతెలుగు\b', r'\btollywood\b', r'\btelugu movie\b',
                r'\btelugu film\b', r'\btelugu cinema\b'
            ],
            'tamil': [
                r'\btamil\b', r'\bதமிழ்\b', r'\bkollywood\b', r'\btamil movie\b',
                r'\btamil film\b', r'\btamil cinema\b'
            ],
            'malayalam': [
                r'\bmalayalam\b', r'\bമലയാളം\b', r'\bmollywood\b', r'\bmalayalam movie\b',
                r'\bmalayalam film\b'
            ],
            'kannada': [
                r'\bkannada\b', r'\bಕನ್ನಡ\b', r'\bsandalwood\b', r'\bkannada movie\b',
                r'\bkannada film\b'
            ],
            'english': [
                r'\benglish\b', r'\bhollywood\b', r'\benglish movie\b', r'\benglish film\b',
                r'\bdubbed\b', r'\bsubtitles\b'
            ]
        }
        
        for lang, patterns in language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, title) or re.search(pattern, description):
                    languages.append(lang)
                    break  # Found this language, move to next
        
        # Default to English if no language detected
        if not languages:
            languages = ['english']
        
        return languages
    
    @staticmethod
    async def _comprehensive_search(title, original_title, release_year, content_type, region):
        """Perform comprehensive search with multiple strategies"""
        all_results = []
        
        # Enhanced search queries for Indian languages
        search_queries = EnhancedYouTubeService._generate_enhanced_search_queries(title, original_title, release_year, content_type, region)
        
        async with aiohttp.ClientSession() as session:
            for query in search_queries[:15]:  # Increased limit for better coverage
                try:
                    url = f"{EnhancedYouTubeService.BASE_URL}/search"
                    params = {
                        'key': YOUTUBE_API_KEY,
                        'q': query,
                        'part': 'snippet',
                        'type': 'video',
                        'maxResults': 15,
                        'order': 'relevance',
                        'regionCode': region,
                        'safeSearch': 'moderate'
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get('items', []):
                                video_data = await EnhancedYouTubeService._get_detailed_video_info(session, item)
                                if video_data:
                                    all_results.append(video_data)
                        
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"YouTube search error for query '{query}': {e}")
                    continue
        
        # Remove duplicates and sort by quality
        seen_ids = set()
        unique_results = []
        for video in sorted(all_results, key=lambda x: x.get('quality_score', 0), reverse=True):
            if video['video_id'] not in seen_ids:
                seen_ids.add(video['video_id'])
                unique_results.append(video)
        
        return unique_results[:50]  # Return top 50 results
    
    @staticmethod
    def _generate_enhanced_search_queries(title, original_title, release_year, content_type, region):
        """Generate enhanced search queries for better Indian content discovery"""
        queries = []
        year_str = str(release_year) if release_year else ""
        
        # Basic searches
        queries.extend([
            f"{title} full movie",
            f"{title} full {content_type}",
            f"{title} movie {year_str}",
            f"{title} complete movie"
        ])
        
        # Language-specific searches with higher priority for Telugu and English
        priority_languages = ['telugu', 'english', 'hindi', 'tamil', 'malayalam', 'kannada']
        
        for lang in priority_languages:
            queries.extend([
                f"{title} {lang} full movie",
                f"{title} {lang} movie",
                f"{title} full movie {lang}",
                f"{title} {lang} cinema",
                f"{title} {lang} film"
            ])
        
        # Original title searches
        if original_title and original_title != title:
            queries.extend([
                f"{original_title} full movie",
                f"{original_title} movie {year_str}"
            ])
        
        # Free content searches
        queries.extend([
            f"{title} free full movie",
            f"{title} movie free online",
            f"watch {title} free",
            f"{title} full movie no ads"
        ])
        
        # Industry-specific searches
        queries.extend([
            f"{title} bollywood movie",
            f"{title} tollywood movie",
            f"{title} kollywood movie",
            f"{title} mollywood movie",
            f"{title} sandalwood movie"
        ])
        
        return queries
    
    @staticmethod
    async def _get_detailed_video_info(session, video_item):
        """Get detailed information about a video (keeping existing function)"""
        try:
            video_id = video_item['id']['videoId']
            
            url = f"{EnhancedYouTubeService.BASE_URL}/videos"
            params = {
                'key': YOUTUBE_API_KEY,
                'id': video_id,
                'part': 'snippet,contentDetails,statistics,status',
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('items'):
                        video_details = data['items'][0]
                        
                        duration = EnhancedYouTubeService._parse_duration(
                            video_details['contentDetails'].get('duration', 'PT0S')
                        )
                        
                        return {
                            'video_id': video_id,
                            'title': video_details['snippet']['title'],
                            'description': video_details['snippet']['description'],
                            'channel_title': video_details['snippet']['channelTitle'],
                            'channel_id': video_details['snippet']['channelId'],
                            'published_at': video_details['snippet']['publishedAt'],
                            'duration_seconds': duration,
                            'duration_formatted': EnhancedYouTubeService._format_duration(duration),
                            'view_count': int(video_details['statistics'].get('viewCount', 0)),
                            'like_count': int(video_details['statistics'].get('likeCount', 0)),
                            'thumbnail_url': video_details['snippet']['thumbnails'].get('high', {}).get('url'),
                            'watch_url': f"https://youtube.com/watch?v={video_id}",
                            'embed_url': f"https://youtube.com/embed/{video_id}",
                            'is_live': video_details['snippet'].get('liveBroadcastContent') == 'live',
                            'is_premium': EnhancedYouTubeService._check_if_premium(video_details),
                            'quality_score': EnhancedYouTubeService._calculate_quality_score(video_details, duration),
                            'detected_languages': EnhancedYouTubeService._detect_video_language(video_details['snippet'])
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Video details error for {video_item.get('id', {}).get('videoId')}: {e}")
            return None
    
    @staticmethod
    def _parse_duration(duration_str):
        """Parse YouTube duration format (keeping existing function)"""
        import re
        
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_str)
        
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds
        
        return 0
    
    @staticmethod
    def _format_duration(seconds):
        """Format seconds to readable duration (keeping existing function)"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    @staticmethod
    def _categorize_video(video, movie_title):
        """Categorize video based on content analysis (keeping existing function)"""
        title = video['title'].lower()
        description = video['description'].lower()
        duration = video['duration_seconds']
        
        if duration > 3600:  # More than 1 hour
            if any(keyword in title for keyword in ['full movie', 'complete movie', 'full film', 'movie', 'पूरी फिल्म']):
                if video['is_premium']:
                    return 'premium_content'
                else:
                    return 'free_movies'
        
        elif 60 <= duration <= 300:
            if any(keyword in title for keyword in ['trailer', 'teaser', 'preview', 'promo']):
                return 'trailers'
        
        elif 180 <= duration <= 900:
            if any(keyword in title for keyword in ['song', 'clip', 'scene', 'dialogue', 'making', 'behind']):
                return 'clips'
        
        if video['quality_score'] > 0.7:
            if video['is_premium']:
                return 'premium_content'
            else:
                return 'free_movies'
        
        return None
    
    @staticmethod
    def _check_if_premium(video_details):
        """Check if video is premium content (keeping existing function)"""
        title = video_details['snippet']['title'].lower()
        description = video_details['snippet']['description'].lower()
        
        premium_indicators = [
            'youtube premium', 'rent or buy', 'purchase', 'premium movie', 'paid content'
        ]
        
        return any(indicator in title or indicator in description for indicator in premium_indicators)
    
    @staticmethod
    def _calculate_quality_score(video_details, duration):
        """Calculate quality score based on various factors (keeping existing function)"""
        score = 0.0
        
        if duration > 3600:
            score += 0.4
        elif duration > 1800:
            score += 0.2
        
        view_count = int(video_details['statistics'].get('viewCount', 0))
        if view_count > 1000000:
            score += 0.3
        elif view_count > 100000:
            score += 0.2
        elif view_count > 10000:
            score += 0.1
        
        like_count = int(video_details['statistics'].get('likeCount', 0))
        if like_count > 1000:
            score += 0.1
        
        channel_title = video_details['snippet']['channelTitle'].lower()
        verified_indicators = ['official', 'music', 'entertainment', 'movies', 'cinema']
        if any(indicator in channel_title for indicator in verified_indicators):
            score += 0.2
        
        return min(score, 1.0)

# Enhanced OTT Availability Service (keeping existing function names)
class OTTAvailabilityService:
    
    @staticmethod
    async def get_comprehensive_availability(tmdb_id, content_type='movie', region='IN'):
        """Enhanced availability with WatchMode API and language support"""
        cache_key = f"{tmdb_id}_{content_type}_{region}"
        
        cached_result = ott_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Get title information first
        title_data = await OTTAvailabilityService.get_tmdb_title(tmdb_id, content_type)
        if not title_data:
            return {'platforms': [], 'language_availability': {}, 'last_updated': datetime.utcnow().isoformat()}
        
        availability_data = {
            'platforms': [],
            'language_availability': {},
            'free_options': [],
            'paid_options': [],
            'rent_options': [],
            'buy_options': [],
            'youtube_data': {},
            'last_updated': datetime.utcnow().isoformat(),
            'region': region,
            'title': title_data['title']
        }
        
        try:
            # Get WatchMode data (replacing JustWatch)
            watchmode_data = await WatchModeService.get_comprehensive_availability(
                tmdb_id,
                title_data.get('imdb_id'),
                title_data['title'],
                content_type,
                region.lower()
            )
            
            if watchmode_data:
                availability_data['platforms'].extend(watchmode_data.get('platforms', []))
                availability_data['language_availability'].update(watchmode_data.get('language_availability', {}))
                availability_data['free_options'].extend(watchmode_data.get('free_options', []))
                availability_data['paid_options'].extend(watchmode_data.get('paid_options', []))
            
            # Get comprehensive YouTube availability
            youtube_data = await EnhancedYouTubeService.get_comprehensive_youtube_availability(
                title_data['title'],
                title_data.get('original_title'),
                title_data.get('release_year'),
                content_type,
                region
            )
            availability_data['youtube_data'] = youtube_data
            
            # Convert YouTube data to platforms with language support
            youtube_platforms = OTTAvailabilityService._convert_youtube_to_platforms_with_languages(youtube_data)
            availability_data['platforms'].extend(youtube_platforms)
            
            # Add YouTube language availability
            for lang, videos in youtube_data.get('language_specific', {}).items():
                if lang not in availability_data['language_availability']:
                    availability_data['language_availability'][lang] = []
                
                for video in videos:
                    platform_info = {
                        'platform_id': 'youtube',
                        'platform_name': 'YouTube',
                        'watch_url': video['watch_url'],
                        'availability_type': 'free',
                        'is_free': True,
                        'language': lang,
                        'video_title': video['title'],
                        'quality_score': video.get('quality_score', 0)
                    }
                    availability_data['language_availability'][lang].append(platform_info)
            
            # Get TMDB providers
            tmdb_data = await OTTAvailabilityService.get_tmdb_providers(tmdb_id, content_type, region)
            if tmdb_data:
                availability_data['platforms'].extend(tmdb_data)
            
            # Sort by language priority
            OTTAvailabilityService._sort_by_language_priority(availability_data)
            
            # Cache the result
            ott_cache.set(cache_key, availability_data)
            
            return availability_data
            
        except Exception as e:
            logger.error(f"OTT availability error: {e}")
            return availability_data
    
    @staticmethod
    def _convert_youtube_to_platforms_with_languages(youtube_data):
        """Convert YouTube data to platform format with language support"""
        platforms = []
        
        # Free movies with language detection
        for movie in youtube_data.get('free_movies', []):
            for language in movie.get('detected_languages', ['english']):
                platforms.append({
                    'platform_id': 'youtube',
                    'platform_name': 'YouTube',
                    'watch_url': movie['watch_url'],
                    'availability_type': 'free',
                    'is_free': True,
                    'price': None,
                    'currency': None,
                    'source': 'youtube',
                    'video_title': movie['title'],
                    'duration': movie['duration_formatted'],
                    'quality_score': movie['quality_score'],
                    'view_count': movie['view_count'],
                    'channel_name': movie['channel_title'],
                    'content_type': 'full_movie',
                    'language': language,
                    'audio_languages': movie.get('detected_languages', [])
                })
        
        return platforms
    
    @staticmethod
    def _sort_by_language_priority(availability_data):
        """Sort availability data by language priority"""
        for lang in availability_data.get('language_availability', {}):
            platforms = availability_data['language_availability'][lang]
            # Sort by quality score and platform preference
            platforms.sort(key=lambda x: (
                LANGUAGE_PRIORITY.get(lang, 10),  # Language priority
                0 if x.get('is_free') else 1,     # Free first
                -x.get('quality_score', 0)        # Higher quality first
            ))
    
    @staticmethod
    async def get_tmdb_providers(tmdb_id, content_type, region):
        """Get availability from TMDB (keeping existing function)"""
        try:
            url = f"https://api.themoviedb.org/3/{content_type}/{tmdb_id}/watch/providers"
            params = {'api_key': TMDB_API_KEY}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        providers = []
                        
                        if 'results' in data and region in data['results']:
                            region_data = data['results'][region]
                            
                            # Process different provider types
                            for provider_type in ['flatrate', 'rent', 'buy']:
                                if provider_type in region_data:
                                    for provider in region_data[provider_type]:
                                        providers.append({
                                            'platform_id': provider['provider_name'].lower().replace(' ', '_'),
                                            'platform_name': provider['provider_name'],
                                            'logo_url': f"https://image.tmdb.org/t/p/original{provider['logo_path']}",
                                            'watch_url': OTTAvailabilityService.generate_watch_url(provider['provider_name'], tmdb_id),
                                            'availability_type': 'subscription' if provider_type == 'flatrate' else provider_type,
                                            'is_free': False,
                                            'price': None,
                                            'currency': 'INR' if region == 'IN' else 'USD',
                                            'source': 'tmdb'
                                        })
                        
                        return providers
        except Exception as e:
            logger.error(f"TMDB providers error: {e}")
            return []
    
    @staticmethod
    async def get_tmdb_title(tmdb_id, content_type):
        """Get title from TMDB (keeping existing function)"""
        try:
            url = f"https://api.themoviedb.org/3/{content_type}/{tmdb_id}"
            params = {'api_key': TMDB_API_KEY}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        release_date = data.get('release_date') or data.get('first_air_date')
                        release_year = None
                        if release_date:
                            try:
                                release_year = int(release_date.split('-')[0])
                            except:
                                pass
                        
                        return {
                            'title': data.get('title') or data.get('name'),
                            'original_title': data.get('original_title') or data.get('original_name'),
                            'release_date': release_date,
                            'release_year': release_year,
                            'imdb_id': data.get('imdb_id')
                        }
            return None
            
        except Exception as e:
            logger.error(f"TMDB title fetch error: {e}")
            return None
    
    @staticmethod
    def generate_watch_url(provider_name, tmdb_id):
        """Generate watch URL for a provider (keeping existing function)"""
        provider_mapping = {
            'Netflix': f"https://netflix.com/search?q=tmdb{tmdb_id}",
            'Amazon Prime Video': f"https://primevideo.com/search/ref=atv_sr_def_c_unkc_1_1?phrase=tmdb{tmdb_id}",
            'Disney+ Hotstar': f"https://hotstar.com/search/{tmdb_id}",
            'YouTube': f"https://youtube.com/results?search_query=tmdb{tmdb_id}",
            'JioCinema': f"https://jiocinema.com/search/{tmdb_id}",
            'ZEE5': f"https://zee5.com/search/{tmdb_id}",
            'SonyLIV': f"https://sonyliv.com/search/{tmdb_id}"
        }
        
        return provider_mapping.get(provider_name, f"https://google.com/search?q={provider_name}+tmdb{tmdb_id}")

# Enhanced Telegram Service with new formatting
class TelegramService:
    @staticmethod
    def format_language_buttons(language_availability):
        """Format language-specific viewing options"""
        buttons = []
        
        # Priority order for buttons
        priority_order = ['telugu', 'english', 'hindi', 'tamil', 'malayalam', 'kannada']
        
        for lang in priority_order:
            if lang in language_availability and language_availability[lang]:
                lang_info = LANGUAGE_DISPLAY.get(lang, {'name': lang.title(), 'flag': '🎬'})
                
                # Get best option for this language
                best_option = None
                for option in language_availability[lang]:
                    if not best_option or (option.get('is_free') and not best_option.get('is_free')):
                        best_option = option
                
                if best_option:
                    platform_name = best_option['platform_name']
                    availability_type = "Free" if best_option.get('is_free') else platform_name
                    
                    button_text = f"{lang_info['flag']} {lang_info['name']} ({availability_type})"
                    buttons.append(button_text)
        
        return buttons
    
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        """Enhanced Telegram posting with new format"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            # Get language availability
            language_availability = {}
            if content.language_availability:
                try:
                    language_availability = json.loads(content.language_availability)
                except:
                    pass
            
            # Format genre list
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    pass
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Check for YouTube availability
            youtube_info = ""
            has_youtube_free = False
            if content.youtube_availability:
                try:
                    youtube_data = json.loads(content.youtube_availability)
                    if youtube_data.get('free_movies'):
                        youtube_info = "\n🎬 Free on YouTube!"
                        has_youtube_free = True
                    elif youtube_data.get('trailers'):
                        youtube_info = "\n📺 Trailer Available"
                except:
                    pass
            
            # Format language buttons
            language_buttons = TelegramService.format_language_buttons(language_availability)
            language_section = ""
            if language_buttons:
                language_section = f"\n\n🎯 Choose Your Language to Watch:\n"
                # Split buttons into rows of 2
                for i in range(0, len(language_buttons), 2):
                    row_buttons = language_buttons[i:i+2]
                    language_section += f"[{'] ['.join(row_buttons)}]\n"
            
            # Add YouTube and trailer buttons
            action_buttons = []
            if has_youtube_free:
                action_buttons.append("Watch Free on YouTube")
            action_buttons.append("📺 Watch Trailer")
            
            if action_buttons:
                language_section += f"[{'] ['.join(action_buttons)}]"
            
            # Create enhanced message
            message = f"""**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}{youtube_info}

📝 Admin's Note: {description}

📖 Synopsis: {(content.overview[:150] + '...') if content.overview else 'A must-watch recommendation from our admin team!'}{language_section}

For More - {WEBSITE_URL}

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

# External API Services (keeping existing function names)
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
    def get_trending(content_type='all', time_window='day', page=1, region='IN'):
        """Enhanced trending with region support"""
        url = f"{TMDBService.BASE_URL}/trending/{content_type}/{time_window}"
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
            logger.error(f"TMDB trending error: {e}")
        return None
    
    @staticmethod
    def get_popular(content_type='movie', page=1, region='IN'):
        """Enhanced popular with language filtering"""
        url = f"{TMDBService.BASE_URL}/{content_type}/popular"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page,
            'region': region
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB popular error: {e}")
        return None
    
    @staticmethod
    def discover_by_language(language='te', content_type='movie', page=1):
        """Discover content by language (prioritizing Telugu and English)"""
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_original_language': language,
            'sort_by': 'popularity.desc',
            'page': page,
            'vote_count.gte': 10
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
        """Enhanced anime details function"""
        url = f"{JikanService.BASE_URL}/anime/{anime_id}"
        
        try:
            response = requests.get(url, params={'fields': 'id,title,main_picture,alternative_titles,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,status,genres,media_type,num_episodes,start_season,rating,studios'}, timeout=10)
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

# Enhanced Content Service (keeping existing function names)
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
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
            
            # Extract languages
            languages = []
            if 'spoken_languages' in tmdb_data:
                languages = [lang['english_name'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
            
            # Get comprehensive OTT data with language support
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ott_data = loop.run_until_complete(
                    OTTAvailabilityService.get_comprehensive_availability(
                        tmdb_data['id'], content_type, 'IN'
                    )
                )
            finally:
                loop.close()
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
                imdb_id=tmdb_data.get('imdb_id'),
                title=tmdb_data.get('title') or tmdb_data.get('name'),
                original_title=tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                content_type=content_type,
                genres=json.dumps(genres),
                languages=json.dumps(languages),
                audio_languages=json.dumps(languages),  # Initialize with same as languages
                release_date=datetime.strptime(tmdb_data.get('release_date') or tmdb_data.get('first_air_date', '1900-01-01'), '%Y-%m-%d').date() if tmdb_data.get('release_date') or tmdb_data.get('first_air_date') else None,
                runtime=tmdb_data.get('runtime'),
                rating=tmdb_data.get('vote_average'),
                vote_count=tmdb_data.get('vote_count'),
                popularity=tmdb_data.get('popularity'),
                overview=tmdb_data.get('overview'),
                poster_path=tmdb_data.get('poster_path'),
                backdrop_path=tmdb_data.get('backdrop_path'),
                ott_platforms=json.dumps(ott_data),
                language_availability=json.dumps(ott_data.get('language_availability', {})),
                youtube_availability=json.dumps(ott_data.get('youtube_data', {}))
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
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]

# Enhanced Recommendation Service
class RecommendationService:
    @staticmethod
    def get_language_priority_recommendations(language_preferences=None, limit=20):
        """Get recommendations based on language priority"""
        try:
            # Default to Telugu and English priority
            if not language_preferences:
                language_preferences = ['telugu', 'english', 'hindi', 'tamil', 'malayalam', 'kannada']
            
            recommendations = []
            
            # Get trending content for each priority language
            for i, lang in enumerate(language_preferences[:3]):  # Top 3 languages
                lang_code = LANGUAGE_DISPLAY.get(lang, {}).get('code', 'en')
                
                # Get trending from TMDB
                trending_data = TMDBService.discover_by_language(lang_code, 'movie', 1)
                if trending_data and trending_data.get('results'):
                    for item in trending_data['results'][:limit//3]:
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content:
                            recommendations.append({
                                'content': content,
                                'reason': f'Trending in {lang.title()}',
                                'language_priority': i + 1,
                                'category': 'trending'
                            })
            
            # Sort by language priority and rating
            recommendations.sort(key=lambda x: (x['language_priority'], -x['content'].rating if x['content'].rating else 0))
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Language priority recommendations error: {e}")
            return []
    
    @staticmethod
    def get_new_releases(days=30, language_preferences=None):
        """Get new releases with language priority"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = Content.query.filter(
                Content.release_date >= cutoff_date.date(),
                Content.content_type == 'movie'
            ).order_by(Content.release_date.desc())
            
            new_releases = []
            for content in query.limit(50).all():
                if content.languages:
                    try:
                        content_languages = json.loads(content.languages)
                        # Check if content has priority languages
                        for lang in (language_preferences or ['telugu', 'english']):
                            if any(lang.lower() in cl.lower() for cl in content_languages):
                                new_releases.append({
                                    'content': content,
                                    'reason': 'New Release',
                                    'category': 'new_release'
                                })
                                break
                    except:
                        pass
            
            return new_releases[:20]
            
        except Exception as e:
            logger.error(f"New releases error: {e}")
            return []

# API Routes

# Enhanced OTT Availability Routes
@app.route('/api/ott/availability/<int:content_id>', methods=['GET'])
def get_ott_availability(content_id):
    """Get real-time OTT availability with language support"""
    try:
        content = Content.query.get_or_404(content_id)
        region = request.args.get('region', 'IN')
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        if not content.tmdb_id:
            return jsonify({'error': 'TMDB ID not available for this content'}), 400
        
        if force_refresh or not content.ott_platforms or not content.language_availability:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                availability_data = loop.run_until_complete(
                    OTTAvailabilityService.get_comprehensive_availability(
                        content.tmdb_id,
                        content.content_type,
                        region
                    )
                )
                
                # Update content in database
                content.ott_platforms = json.dumps(availability_data)
                content.language_availability = json.dumps(availability_data.get('language_availability', {}))
                content.youtube_availability = json.dumps(availability_data.get('youtube_data', {}))
                content.updated_at = datetime.utcnow()
                db.session.commit()
                
            finally:
                loop.close()
        else:
            # Use cached data
            try:
                availability_data = json.loads(content.ott_platforms or '{}')
                if not availability_data.get('language_availability') and content.language_availability:
                    availability_data['language_availability'] = json.loads(content.language_availability)
                if not availability_data.get('youtube_data') and content.youtube_availability:
                    availability_data['youtube_data'] = json.loads(content.youtube_availability)
            except:
                availability_data = {'platforms': [], 'language_availability': {}, 'youtube_data': {}}
        
        return jsonify({
            'content_id': content.id,
            'title': content.title,
            'availability': availability_data,
            'language_buttons': TelegramService.format_language_buttons(availability_data.get('language_availability', {}))
        }), 200
        
    except Exception as e:
        logger.error(f"OTT availability error: {e}")
        return jsonify({'error': 'Failed to get OTT availability'}), 500

# Enhanced Content Routes
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
        
        # Get language availability
        language_availability = {}
        if content.language_availability:
            try:
                language_availability = json.loads(content.language_availability)
            except:
                pass
        
        # Get additional details from TMDB
        additional_details = None
        if content.tmdb_id:
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        
        # Get trailers with language info
        trailers = []
        if content.youtube_availability:
            try:
                youtube_data = json.loads(content.youtube_availability)
                trailers = youtube_data.get('trailers', [])[:5]
            except:
                pass
        
        db.session.commit()
        
        return jsonify({
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'audio_languages': json.loads(content.audio_languages or '[]'),
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'runtime': content.runtime,
            'rating': content.rating,
            'vote_count': content.vote_count,
            'overview': content.overview,
            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path else None,
            'ott_platforms': json.loads(content.ott_platforms or '{}'),
            'language_availability': language_availability,
            'youtube_availability': json.loads(content.youtube_availability or '{}'),
            'language_buttons': TelegramService.format_language_buttons(language_availability),
            'trailers': trailers,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Fixed Anime Routes
@app.route('/api/anime/<int:anime_id>', methods=['GET'])
def get_anime_details(anime_id):
    """Fixed anime details endpoint"""
    try:
        # Get details from Jikan API
        anime_data = JikanService.get_anime_details(anime_id)
        
        if not anime_data:
            return jsonify({'error': 'Anime not found'}), 404
        
        # Record view interaction for anonymous users
        session_id = get_session_id()
        # Note: For anime, we don't save to Content table but can track views separately
        
        # Format the response
        formatted_data = {
            'id': anime_data.get('mal_id'),
            'title': anime_data.get('title'),
            'title_english': anime_data.get('title_english'),
            'title_japanese': anime_data.get('title_japanese'),
            'type': anime_data.get('type'),
            'episodes': anime_data.get('episodes'),
            'status': anime_data.get('status'),
            'aired': anime_data.get('aired', {}),
            'rating': anime_data.get('rating'),
            'score': anime_data.get('score'),
            'scored_by': anime_data.get('scored_by'),
            'rank': anime_data.get('rank'),
            'popularity': anime_data.get('popularity'),
            'synopsis': anime_data.get('synopsis'),
            'background': anime_data.get('background'),
            'season': anime_data.get('season'),
            'year': anime_data.get('year'),
            'broadcast': anime_data.get('broadcast'),
            'producers': anime_data.get('producers', []),
            'licensors': anime_data.get('licensors', []),
            'studios': anime_data.get('studios', []),
            'genres': anime_data.get('genres', []),
            'themes': anime_data.get('themes', []),
            'demographics': anime_data.get('demographics', []),
            'images': anime_data.get('images', {}),
            'trailer': anime_data.get('trailer'),
            'url': anime_data.get('url')
        }
        
        return jsonify(formatted_data), 200
        
    except Exception as e:
        logger.error(f"Anime details error: {e}")
        return jsonify({'error': 'Failed to get anime details'}), 500

@app.route('/api/anime/search', methods=['GET'])
def search_anime():
    """Search anime with proper error handling"""
    try:
        query = request.args.get('q')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        # Search using Jikan API
        search_results = JikanService.search_anime(query, page)
        
        if not search_results:
            return jsonify({'results': [], 'pagination': {}}), 200
        
        return jsonify(search_results), 200
        
    except Exception as e:
        logger.error(f"Anime search error: {e}")
        return jsonify({'error': 'Failed to search anime'}), 500

# Enhanced Recommendation Routes
@app.route('/api/recommendations/language-priority', methods=['GET'])
def get_language_priority_recommendations():
    """Get recommendations based on language priority"""
    try:
        languages = request.args.getlist('languages')
        limit = int(request.args.get('limit', 20))
        
        if not languages:
            languages = ['telugu', 'english']
        
        recommendations = RecommendationService.get_language_priority_recommendations(languages, limit)
        
        formatted_recommendations = []
        for rec in recommendations:
            content = rec['content']
            formatted_recommendations.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'reason': rec['reason'],
                'category': rec['category'],
                'language_priority': rec.get('language_priority', 1)
            })
        
        return jsonify({
            'recommendations': formatted_recommendations,
            'total': len(formatted_recommendations),
            'language_priority': languages
        }), 200
        
    except Exception as e:
        logger.error(f"Language priority recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@app.route('/api/recommendations/new-releases', methods=['GET'])
def get_new_releases():
    """Get new releases with language priority"""
    try:
        days = int(request.args.get('days', 30))
        languages = request.args.getlist('languages')
        
        if not languages:
            languages = ['telugu', 'english']
        
        new_releases = RecommendationService.get_new_releases(days, languages)
        
        formatted_releases = []
        for release in new_releases:
            content = release['content']
            formatted_releases.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'reason': release['reason'],
                'category': release['category']
            })
        
        return jsonify({
            'new_releases': formatted_releases,
            'total': len(formatted_releases),
            'days_range': days
        }), 200
        
    except Exception as e:
        logger.error(f"New releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

# Authentication Routes (keeping existing function names)
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
        
        # Set default language preferences with Telugu priority
        default_languages = data.get('preferred_languages', ['telugu', 'english'])
        
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password']),
            preferred_languages=json.dumps(default_languages),
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
                'is_admin': user.is_admin,
                'preferred_languages': default_languages
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
                'preferred_languages': json.loads(user.preferred_languages or '["telugu", "english"]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

# Admin Routes (keeping existing function names)
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
        
        # Send to Telegram channel with enhanced format
        telegram_success = TelegramService.send_admin_recommendation(content, current_user.username, data['description'])
        
        return jsonify({
            'message': 'Admin recommendation created successfully',
            'telegram_sent': telegram_success,
            'recommendation_id': admin_rec.id
        }), 201
        
    except Exception as e:
        logger.error(f"Admin recommendation error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create recommendation'}), 500

# Search Routes
@app.route('/api/search', methods=['GET'])
def search_content():
    try:
        query = request.args.get('q')
        content_type = request.args.get('type', 'multi')
        page = int(request.args.get('page', 1))
        language = request.args.get('language', 'en-US')
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        # Search TMDB
        tmdb_results = TMDBService.search_content(query, content_type, language, page)
        
        if not tmdb_results:
            return jsonify({'results': [], 'total_pages': 0, 'total_results': 0}), 200
        
        # Save content to database and format results
        formatted_results = []
        for item in tmdb_results.get('results', []):
            # Determine content type
            item_type = item.get('media_type', content_type)
            if item_type == 'person':
                continue  # Skip person results
            
            if item_type not in ['movie', 'tv']:
                item_type = 'movie' if 'title' in item else 'tv'
            
            # Save to database
            saved_content = ContentService.save_content_from_tmdb(item, item_type)
            
            if saved_content:
                formatted_results.append({
                    'id': saved_content.id,
                    'tmdb_id': saved_content.tmdb_id,
                    'title': saved_content.title,
                    'original_title': saved_content.original_title,
                    'content_type': saved_content.content_type,
                    'release_date': saved_content.release_date.isoformat() if saved_content.release_date else None,
                    'rating': saved_content.rating,
                    'overview': saved_content.overview,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{saved_content.poster_path}" if saved_content.poster_path else None,
                    'genres': json.loads(saved_content.genres or '[]'),
                    'languages': json.loads(saved_content.languages or '[]')
                })
        
        return jsonify({
            'results': formatted_results,
            'page': tmdb_results.get('page', 1),
            'total_pages': tmdb_results.get('total_pages', 1),
            'total_results': tmdb_results.get('total_results', 0)
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.1.0',
        'features': [
            'watchmode_api_integration',
            'language_priority_recommendations',
            'enhanced_youtube_integration',
            'enhanced_telegram_posting',
            'fixed_anime_support'
        ],
        'language_priority': list(LANGUAGE_PRIORITY.keys())
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
                    is_admin=True,
                    preferred_languages=json.dumps(['telugu', 'english'])
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