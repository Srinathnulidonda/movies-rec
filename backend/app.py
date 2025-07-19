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
from urllib.parse import quote

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

# API Keys - Updated with provided keys
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '52260795')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '-1002850793757')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-xmf5.onrender.com')

# Updated API Keys
WATCHMODE_API_KEY = os.environ.get('WATCHMODE_API_KEY', 'WtcKDji9i20pjOl5Lg0AiyG2bddfUs3nSZRZJIsY')
RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY', 'c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37')

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
    streaming_updated_at = db.Column(db.DateTime)
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

# Enhanced Streaming Availability Service with Multiple APIs
class StreamingAvailabilityService:
    # Streaming Availability API
    STREAMING_BASE_URL = 'https://streaming-availability.p.rapidapi.com'
    STREAMING_HEADERS = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': "streaming-availability.p.rapidapi.com"
    }
    
    # Watchmode API
    WATCHMODE_BASE_URL = 'https://api.watchmode.com/v1'
    
    # Platform mapping for better display with language support
    PLATFORM_MAPPING = {
        'netflix': {'name': 'Netflix', 'is_free': False, 'url': 'https://netflix.com', 'priority': 1},
        'prime': {'name': 'Amazon Prime Video', 'is_free': False, 'url': 'https://primevideo.com', 'priority': 2},
        'hotstar': {'name': 'Disney+ Hotstar', 'is_free': False, 'url': 'https://hotstar.com', 'priority': 3},
        'zee5': {'name': 'ZEE5', 'is_free': False, 'url': 'https://zee5.com', 'priority': 4},
        'sonyliv': {'name': 'SonyLIV', 'is_free': False, 'url': 'https://sonyliv.com', 'priority': 5},
        'voot': {'name': 'Voot', 'is_free': True, 'url': 'https://voot.com', 'priority': 6},
        'mx': {'name': 'MX Player', 'is_free': True, 'url': 'https://mxplayer.in', 'priority': 7},
        'youtube': {'name': 'YouTube', 'is_free': True, 'url': 'https://youtube.com', 'priority': 8},
        'jiocinema': {'name': 'JioCinema', 'is_free': True, 'url': 'https://jiocinema.com', 'priority': 9},
        'airtel': {'name': 'Airtel Xstream', 'is_free': True, 'url': 'https://airtelxstream.in', 'priority': 10},
        'crunchyroll': {'name': 'Crunchyroll', 'is_free': True, 'url': 'https://crunchyroll.com', 'priority': 11},
        'aha': {'name': 'Aha', 'is_free': False, 'url': 'https://aha.video', 'priority': 12},
        'sunnxt': {'name': 'Sun NXT', 'is_free': False, 'url': 'https://sunnxt.com', 'priority': 13},
        'erosnow': {'name': 'Eros Now', 'is_free': False, 'url': 'https://erosnow.com', 'priority': 14}
    }
    
    # Language mapping
    LANGUAGE_MAPPING = {
        'hi': 'Hindi',
        'te': 'Telugu', 
        'ta': 'Tamil',
        'ml': 'Malayalam',
        'kn': 'Kannada',
        'bn': 'Bengali',
        'gu': 'Gujarati',
        'mr': 'Marathi',
        'pa': 'Punjabi',
        'en': 'English',
        'ja': 'Japanese',
        'ko': 'Korean',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian'
    }
    
    @staticmethod
    def search_by_title_streaming_api(title, country='in'):
        """Search for content by title using Streaming Availability API"""
        try:
            url = f"{StreamingAvailabilityService.STREAMING_BASE_URL}/search/title"
            params = {
                'title': title,
                'country': country,
                'show_type': 'all',
                'output_language': 'en'
            }
            
            response = requests.get(
                url, 
                headers=StreamingAvailabilityService.STREAMING_HEADERS, 
                params=params, 
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Streaming API returned status {response.status_code} for title: {title}")
                return None
        except Exception as e:
            logger.error(f"Streaming availability search error: {e}")
            return None
    
    @staticmethod
    def get_streaming_info_by_tmdb(tmdb_id, content_type='movie'):
        """Get streaming information by TMDB ID using Streaming Availability API"""
        try:
            url = f"{StreamingAvailabilityService.STREAMING_BASE_URL}/get"
            params = {
                'tmdb_id': tmdb_id,
                'output_language': 'en'
            }
            
            response = requests.get(
                url, 
                headers=StreamingAvailabilityService.STREAMING_HEADERS, 
                params=params, 
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Streaming API returned status {response.status_code} for TMDB ID: {tmdb_id}")
                return None
        except Exception as e:
            logger.error(f"Streaming info error: {e}")
            return None
    
    @staticmethod
    def search_watchmode_api(title, imdb_id=None):
        """Search using Watchmode API as fallback"""
        try:
            if not WATCHMODE_API_KEY or WATCHMODE_API_KEY == 'your_watchmode_api_key':
                return None
                
            url = f"{StreamingAvailabilityService.WATCHMODE_BASE_URL}/search/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'search_field': 'name',
                'search_value': title
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('title_results'):
                    # Get detailed info for first result
                    first_result = data['title_results'][0]
                    title_id = first_result.get('id')
                    
                    if title_id:
                        return StreamingAvailabilityService.get_watchmode_details(title_id)
            else:
                logger.warning(f"Watchmode API returned status {response.status_code} for title: {title}")
                
        except Exception as e:
            logger.error(f"Watchmode search error: {e}")
        
        return None
    
    @staticmethod
    def get_watchmode_details(title_id):
        """Get detailed streaming info from Watchmode"""
        try:
            url = f"{StreamingAvailabilityService.WATCHMODE_BASE_URL}/title/{title_id}/details/"
            params = {'apiKey': WATCHMODE_API_KEY}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Watchmode details API returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Watchmode details error: {e}")
        
        return None
    
    @staticmethod
    def format_streaming_data_enhanced(streaming_data, title=None):
        """Enhanced formatting with language-specific buttons"""
        if not streaming_data:
            return []
        
        formatted_platforms = []
        
        try:
            # Handle different response formats
            if 'streamingInfo' in streaming_data:
                streaming_info = streaming_data['streamingInfo']
            elif 'result' in streaming_data:
                if isinstance(streaming_data['result'], list) and streaming_data['result']:
                    streaming_info = streaming_data['result'][0].get('streamingInfo', {})
                else:
                    streaming_info = streaming_data['result'].get('streamingInfo', {})
            else:
                streaming_info = streaming_data
            
            # Process India's streaming info first, then other countries
            countries_to_check = ['in', 'us', 'gb', 'ca']
            
            for country in countries_to_check:
                if country in streaming_info:
                    platforms = streaming_info[country]
                    
                    for platform_key, platform_data in platforms.items():
                        platform_info = StreamingAvailabilityService.PLATFORM_MAPPING.get(
                            platform_key.lower(), 
                            {'name': platform_key.title(), 'is_free': False, 'url': '', 'priority': 999}
                        )
                        
                        # Handle multiple entries for same platform (different languages)
                        if isinstance(platform_data, list):
                            for idx, item in enumerate(platform_data):
                                # Extract audio languages
                                audio_languages = []
                                if item.get('audios'):
                                    for audio in item['audios']:
                                        lang_code = audio.get('language', 'en')
                                        lang_name = StreamingAvailabilityService.LANGUAGE_MAPPING.get(
                                            lang_code, lang_code.upper()
                                        )
                                        audio_languages.append({
                                            'code': lang_code,
                                            'name': lang_name
                                        })
                                
                                # Extract subtitle languages
                                subtitle_languages = []
                                if item.get('subtitles'):
                                    for subtitle in item['subtitles']:
                                        lang_code = subtitle.get('language', 'en')
                                        lang_name = StreamingAvailabilityService.LANGUAGE_MAPPING.get(
                                            lang_code, lang_code.upper()
                                        )
                                        subtitle_languages.append({
                                            'code': lang_code,
                                            'name': lang_name
                                        })
                                
                                # Create separate entries for each audio language
                                if audio_languages:
                                    for audio_lang in audio_languages:
                                        formatted_platforms.append({
                                            'platform': platform_key,
                                            'platform_name': platform_info['name'],
                                            'is_free': platform_info['is_free'],
                                            'url': item.get('link', platform_info['url']),
                                            'audio_language': audio_lang,
                                            'subtitle_languages': subtitle_languages,
                                            'quality': item.get('quality', 'HD'),
                                            'type': item.get('streamingType', 'subscription'),
                                            'country': country,
                                            'priority': platform_info['priority']
                                        })
                                else:
                                    # Default entry if no language info
                                    formatted_platforms.append({
                                        'platform': platform_key,
                                        'platform_name': platform_info['name'],
                                        'is_free': platform_info['is_free'],
                                        'url': item.get('link', platform_info['url']),
                                        'audio_language': {'code': 'en', 'name': 'English'},
                                        'subtitle_languages': subtitle_languages,
                                        'quality': item.get('quality', 'HD'),
                                        'type': item.get('streamingType', 'subscription'),
                                        'country': country,
                                        'priority': platform_info['priority']
                                    })
                        else:
                            # Single platform entry
                            audio_languages = []
                            if platform_data.get('audios'):
                                for audio in platform_data['audios']:
                                    lang_code = audio.get('language', 'en')
                                    lang_name = StreamingAvailabilityService.LANGUAGE_MAPPING.get(
                                        lang_code, lang_code.upper()
                                    )
                                    audio_languages.append({
                                        'code': lang_code,
                                        'name': lang_name
                                    })
                            
                            subtitle_languages = []
                            if platform_data.get('subtitles'):
                                for subtitle in platform_data['subtitles']:
                                    lang_code = subtitle.get('language', 'en')
                                    lang_name = StreamingAvailabilityService.LANGUAGE_MAPPING.get(
                                        lang_code, lang_code.upper()
                                    )
                                    subtitle_languages.append({
                                        'code': lang_code,
                                        'name': lang_name
                                    })
                            
                            if audio_languages:
                                for audio_lang in audio_languages:
                                    formatted_platforms.append({
                                        'platform': platform_key,
                                        'platform_name': platform_info['name'],
                                        'is_free': platform_info['is_free'],
                                        'url': platform_data.get('link', platform_info['url']),
                                        'audio_language': audio_lang,
                                        'subtitle_languages': subtitle_languages,
                                        'quality': platform_data.get('quality', 'HD'),
                                        'type': platform_data.get('streamingType', 'subscription'),
                                        'country': country,
                                        'priority': platform_info['priority']
                                    })
                            else:
                                formatted_platforms.append({
                                    'platform': platform_key,
                                    'platform_name': platform_info['name'],
                                    'is_free': platform_info['is_free'],
                                    'url': platform_data.get('link', platform_info['url']),
                                    'audio_language': {'code': 'en', 'name': 'English'},
                                    'subtitle_languages': subtitle_languages,
                                    'quality': platform_data.get('quality', 'HD'),
                                    'type': platform_data.get('streamingType', 'subscription'),
                                    'country': country,
                                    'priority': platform_info['priority']
                                })
                    
                    # If we found platforms in India, prioritize them
                    if country == 'in' and formatted_platforms:
                        break
        
        except Exception as e:
            logger.error(f"Error formatting streaming data: {e}")
        
        # Remove duplicates and sort by priority
        seen_platforms = set()
        unique_platforms = []
        
        # Sort by priority first
        formatted_platforms.sort(key=lambda x: (x.get('priority', 999), x.get('platform', '')))
        
        for platform in formatted_platforms:
            platform_key = f"{platform['platform']}_{platform['audio_language']['code']}"
            if platform_key not in seen_platforms:
                seen_platforms.add(platform_key)
                unique_platforms.append(platform)
        
        return unique_platforms[:15]  # Limit to 15 platforms
    
    @staticmethod
    def format_watchmode_data(watchmode_data):
        """Format Watchmode API data"""
        if not watchmode_data:
            return []
        
        formatted_platforms = []
        
        try:
            sources = watchmode_data.get('sources', [])
            
            for source in sources:
                source_id = source.get('source_id')
                region = source.get('region', 'US')
                
                # Map Watchmode source IDs to our platform names
                platform_name = StreamingAvailabilityService.map_watchmode_source(source_id)
                
                if platform_name:
                    formatted_platforms.append({
                        'platform': platform_name.lower().replace(' ', ''),
                        'platform_name': platform_name,
                        'is_free': source.get('type') == 'free',
                        'url': source.get('web_url', ''),
                        'audio_language': {'code': 'en', 'name': 'English'},
                        'subtitle_languages': [],
                        'quality': 'HD',
                        'type': source.get('type', 'subscription'),
                        'country': region.lower(),
                        'priority': 50  # Lower priority for Watchmode data
                    })
        
        except Exception as e:
            logger.error(f"Error formatting Watchmode data: {e}")
        
        return formatted_platforms
    
    @staticmethod
    def map_watchmode_source(source_id):
        """Map Watchmode source IDs to platform names"""
        watchmode_mapping = {
            203: 'Netflix',
            26: 'Amazon Prime Video',
            387: 'Disney+ Hotstar',
            73: 'YouTube',
            142: 'Crunchyroll',
            # Add more mappings as needed
        }
        return watchmode_mapping.get(source_id, f"Platform_{source_id}")
    
    @staticmethod
    def get_comprehensive_streaming_info(tmdb_id, title, imdb_id=None):
        """Get streaming info from multiple sources with enhanced language support"""
        all_platforms = []
        
        # Try Streaming Availability API first
        if tmdb_id:
            streaming_data = StreamingAvailabilityService.get_streaming_info_by_tmdb(tmdb_id)
            if streaming_data:
                platforms = StreamingAvailabilityService.format_streaming_data_enhanced(streaming_data)
                all_platforms.extend(platforms)
        
        # If no data from TMDB ID, try by title
        if not all_platforms and title:
            search_results = StreamingAvailabilityService.search_by_title_streaming_api(title)
            if search_results:
                platforms = StreamingAvailabilityService.format_streaming_data_enhanced(search_results, title)
                all_platforms.extend(platforms)
        
        # Fallback to Watchmode API if still no data
        if not all_platforms and title:
            watchmode_data = StreamingAvailabilityService.search_watchmode_api(title, imdb_id)
            if watchmode_data:
                platforms = StreamingAvailabilityService.format_watchmode_data(watchmode_data)
                all_platforms.extend(platforms)
        
        # Remove duplicates and sort
        seen_platforms = set()
        unique_platforms = []
        
        # Sort by priority and then by platform name
        all_platforms.sort(key=lambda x: (x.get('priority', 999), x.get('platform', '')))
        
        for platform in all_platforms:
            platform_key = f"{platform['platform']}_{platform['audio_language']['code']}"
            if platform_key not in seen_platforms:
                seen_platforms.add(platform_key)
                unique_platforms.append(platform)
        
        return unique_platforms

# Enhanced Language Support Service
class LanguageService:
    @staticmethod
    def group_platforms_by_language(platforms):
        """Group streaming platforms by language for better UI display"""
        language_groups = defaultdict(list)
        
        for platform in platforms:
            audio_lang = platform.get('audio_language', {})
            lang_code = audio_lang.get('code', 'en')
            lang_name = audio_lang.get('name', 'English')
            
            language_groups[lang_name].append(platform)
        
        # Sort language groups by priority (Indian languages first)
        priority_languages = ['Hindi', 'Telugu', 'Tamil', 'Malayalam', 'Kannada', 'English']
        
        sorted_groups = {}
        for lang in priority_languages:
            if lang in language_groups:
                sorted_groups[lang] = language_groups[lang]
        
        # Add remaining languages
        for lang, platforms in language_groups.items():
            if lang not in sorted_groups:
                sorted_groups[lang] = platforms
        
        return sorted_groups
    
    @staticmethod
    def create_watch_buttons(platforms):
        """Create language-specific watch buttons"""
        language_groups = LanguageService.group_platforms_by_language(platforms)
        watch_buttons = []
        
        for language, lang_platforms in language_groups.items():
            # Group by platform within language
            platform_groups = defaultdict(list)
            for platform in lang_platforms:
                platform_groups[platform['platform_name']].append(platform)
            
            # Create buttons for each platform in this language
            for platform_name, platform_instances in platform_groups.items():
                # Use the first instance for button creation
                platform = platform_instances[0]
                
                button = {
                    'language': language,
                    'platform_name': platform_name,
                    'platform_key': platform['platform'],
                    'url': platform['url'],
                    'is_free': platform['is_free'],
                    'quality': platform.get('quality', 'HD'),
                    'type': platform.get('type', 'subscription'),
                    'button_text': f"Watch in {language} on {platform_name}",
                    'button_id': f"{platform['platform']}_{platform['audio_language']['code']}",
                    'subtitle_languages': platform.get('subtitle_languages', [])
                }
                
                watch_buttons.append(button)
        
        return watch_buttons

# Regional Content Service
class RegionalContentService:
    # Regional language priorities
    LANGUAGE_PRIORITY = ['telugu', 'hindi', 'tamil', 'malayalam', 'kannada', 'english']
    
    # Genre mapping
    GENRES = [
        'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
        'Documentary', 'Drama', 'Fantasy', 'Horror', 'Musical', 'Mystery', 
        'Romance', 'Sci-Fi', 'Thriller', 'Western'
    ]
    
    # Language code mapping
    LANG_CODES = {
        'telugu': 'te',
        'hindi': 'hi', 
        'tamil': 'ta',
        'malayalam': 'ml',
        'kannada': 'kn',
        'english': 'en'
    }
    
    # Genre ID mapping
    GENRE_MAP = {
        'action': 28, 'adventure': 12, 'animation': 16, 'biography': 18,
        'comedy': 35, 'crime': 80, 'documentary': 99, 'drama': 18,
        'fantasy': 14, 'horror': 27, 'musical': 10402, 'mystery': 9648,
        'romance': 10749, 'sci-fi': 878, 'thriller': 53, 'western': 37
    }
    
    @staticmethod
    def get_discover_params(language, genre=None, sort_by='popularity.desc', year=None, min_vote_count=10):
        """Get TMDB discover parameters for regional content"""
        params = {
            'api_key': TMDB_API_KEY,
            'with_original_language': RegionalContentService.LANG_CODES.get(language.lower(), 'en'),
            'sort_by': sort_by,
            'page': 1,
            'vote_count.gte': min_vote_count
        }
        
        if genre:
            genre_id = RegionalContentService.GENRE_MAP.get(genre.lower())
            if genre_id:
                params['with_genres'] = genre_id
        
        if year:
            params['year'] = year
        
        return params
    
    @staticmethod
    def get_regional_best_movies(language, limit=20):
        """Get best/all-time hit movies for a language"""
        try:
            params = RegionalContentService.get_discover_params(
                language, 
                sort_by='vote_average.desc',
                min_vote_count=100
            )
            params['vote_average.gte'] = 7.0  # High rating threshold
            
            url = f"{TMDBService.BASE_URL}/discover/movie"
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return RegionalContentService.process_tmdb_results(data.get('results', []), limit)
        except Exception as e:
            logger.error(f"Regional best movies error: {e}")
        return []
    
    @staticmethod
    def get_regional_trending(language, limit=20):
        """Get trending movies for a language"""
        try:
            # Get current trending and filter by language
            trending_data = TMDBService.get_trending('movie', 'week')
            if trending_data:
                target_lang = RegionalContentService.LANG_CODES.get(language.lower(), 'en')
                
                filtered_results = []
                for item in trending_data.get('results', []):
                    if item.get('original_language') == target_lang:
                        filtered_results.append(item)
                
                # If not enough trending, get popular for the language
                if len(filtered_results) < limit:
                    params = RegionalContentService.get_discover_params(language, sort_by='popularity.desc')
                    url = f"{TMDBService.BASE_URL}/discover/movie"
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        popular_data = response.json()
                        filtered_results.extend(popular_data.get('results', []))
                
                return RegionalContentService.process_tmdb_results(filtered_results, limit)
        except Exception as e:
            logger.error(f"Regional trending error: {e}")
        return []
    
    @staticmethod
    def get_regional_new_releases(language, limit=20):
        """Get new releases for a language"""
        try:
            current_year = datetime.now().year
            params = RegionalContentService.get_discover_params(
                language,
                sort_by='release_date.desc'
            )
            params['primary_release_year'] = current_year
            params['release_date.lte'] = datetime.now().strftime('%Y-%m-%d')
            
            url = f"{TMDBService.BASE_URL}/discover/movie"
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                # If current year doesn't have enough, try previous year
                if len(results) < limit:
                    params['primary_release_year'] = current_year - 1
                    response2 = requests.get(url, params=params, timeout=10)
                    if response2.status_code == 200:
                        results.extend(response2.json().get('results', []))
                
                return RegionalContentService.process_tmdb_results(results, limit)
        except Exception as e:
            logger.error(f"Regional new releases error: {e}")
        return []
    
    @staticmethod
    def get_regional_by_genre(language, genre, limit=20):
        """Get movies by genre for a language"""
        try:
            params = RegionalContentService.get_discover_params(language, genre, sort_by='popularity.desc')
            
            url = f"{TMDBService.BASE_URL}/discover/movie"
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return RegionalContentService.process_tmdb_results(data.get('results', []), limit)
        except Exception as e:
            logger.error(f"Regional genre error: {e}")
        return []
    
    @staticmethod
    def process_tmdb_results(results, limit):
        """Process TMDB results and save to database"""
        processed_content = []
        
        for item in results[:limit]:
            try:
                content = ContentService.save_content_from_tmdb(item, 'movie')
                if content:
                    processed_content.append({
                        'id': content.id,
                        'tmdb_id': content.tmdb_id,
                        'title': content.title,
                        'original_title': content.original_title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'languages': json.loads(content.languages or '[]'),
                        'rating': content.rating,
                        'vote_count': content.vote_count,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                        'overview': content.overview[:150] + '...' if content.overview else '',
                        'ott_platforms': json.loads(content.ott_platforms or '[]')
                    })
            except Exception as e:
                logger.error(f"Error processing TMDB result: {e}")
                continue
        
        return processed_content

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
            'append_to_response': 'credits,videos,similar,watch/providers,translations'
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
            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"Jikan search error: {e}")
        return None
    
    @staticmethod
    def get_anime_details(anime_id):
        url = f"{JikanService.BASE_URL}/anime/{anime_id}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            time.sleep(1)  # Rate limiting
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
            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"Jikan top anime error: {e}")
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
                # Update streaming info if it's old
                if not existing.streaming_updated_at or \
                   existing.streaming_updated_at < datetime.utcnow() - timedelta(days=7):
                    ContentService.update_streaming_info(existing)
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
            
            # Get OTT platforms with enhanced language support
            ott_platforms = ContentService.get_enhanced_ott_availability(tmdb_data)
            
            # Parse release date
            release_date = None
            date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
            if date_str:
                try:
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                except:
                    pass
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
                title=tmdb_data.get('title') or tmdb_data.get('name'),
                original_title=tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                content_type=content_type,
                genres=json.dumps(genres),
                languages=json.dumps(languages),
                release_date=release_date,
                runtime=tmdb_data.get('runtime'),
                rating=tmdb_data.get('vote_average'),
                vote_count=tmdb_data.get('vote_count'),
                popularity=tmdb_data.get('popularity'),
                overview=tmdb_data.get('overview'),
                poster_path=tmdb_data.get('poster_path'),
                backdrop_path=tmdb_data.get('backdrop_path'),
                ott_platforms=json.dumps(ott_platforms),
                streaming_updated_at=datetime.utcnow()
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
            
            # Parse release date
            release_date = None
            if anime_data.get('aired', {}).get('from'):
                try:
                    release_date = datetime.strptime(anime_data['aired']['from'][:10], '%Y-%m-%d').date()
                except:
                    pass
            
            content = Content(
                mal_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps([genre['name'] for genre in anime_data.get('genres', [])]),
                languages=json.dumps(['japanese']),
                release_date=release_date,
                runtime=anime_data.get('duration_minutes'),
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('large_image_url'),
                ott_platforms=json.dumps([]),
                streaming_updated_at=datetime.utcnow()
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def update_streaming_info(content):
        """Update streaming information for existing content"""
        try:
            if content.tmdb_id:
                ott_platforms = ContentService.get_enhanced_ott_availability({
                    'id': content.tmdb_id, 
                    'title': content.title
                })
                content.ott_platforms = json.dumps(ott_platforms)
                content.streaming_updated_at = datetime.utcnow()
                db.session.commit()
        except Exception as e:
            logger.error(f"Error updating streaming info: {e}")
    
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
    
    @staticmethod
    def get_enhanced_ott_availability(tmdb_data):
        """Get OTT availability with enhanced language support using multiple APIs"""
        platforms = []
        
        try:
            tmdb_id = tmdb_data.get('id')
            title = tmdb_data.get('title') or tmdb_data.get('name')
            imdb_id = tmdb_data.get('imdb_id')
            
            if tmdb_id or title:
                platforms = StreamingAvailabilityService.get_comprehensive_streaming_info(
                    tmdb_id, title, imdb_id
                )
        except Exception as e:
            logger.error(f"Error getting enhanced OTT availability: {e}")
        
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
                        try:
                            all_genres.extend(json.loads(content.genres))
                        except:
                            pass
                
                # Get most common genres
                if all_genres:
                    genre_counts = Counter(all_genres)
                    top_genres = [genre for genre, _ in genre_counts.most_common(3)]
                    
                    # Get recommendations based on top genres
                    for genre in top_genres:
                        genre_recs = RecommendationEngine.get_popular_by_genre(genre, limit=7)
                        recommendations.extend(genre_recs)
            
            # Add regional content based on location
            if location and location.get('country') == 'India':
                regional_recs = RegionalContentService.get_regional_trending('hindi', limit=5)
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
            
            # Get real-time streaming availability with language support
            streaming_platforms = []
            if content.tmdb_id:
                streaming_platforms = StreamingAvailabilityService.get_comprehensive_streaming_info(
                    content.tmdb_id, content.title
                )
            
            # If no streaming data from API, use stored data
            if not streaming_platforms:
                try:
                    streaming_platforms = json.loads(content.ott_platforms or '[]')
                except:
                    streaming_platforms = []
            
            # Create language-specific watch buttons
            watch_buttons = LanguageService.create_watch_buttons(streaming_platforms)
            
            # Group platforms by language for telegram display
            language_groups = LanguageService.group_platforms_by_language(streaming_platforms)
            
            # Build streaming availability text with language support
            streaming_text = ""
            for language, platforms in language_groups.items():
                if platforms:
                    streaming_text += f"\n🎭 **{language}:**\n"
                    
                    free_platforms = []
                    paid_platforms = []
                    
                    for platform in platforms[:3]:  # Limit to 3 platforms per language
                        platform_name = platform.get('platform_name', platform.get('platform', '').title())
                        if platform.get('is_free'):
                            free_platforms.append(platform_name)
                        else:
                            paid_platforms.append(platform_name)
                    
                    if free_platforms:
                        streaming_text += f"🆓 Free: {', '.join(free_platforms)}\n"
                    if paid_platforms:
                        streaming_text += f"💰 Paid: {', '.join(paid_platforms)}\n"
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

🎥 **Available On:**{streaming_text}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

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

# Enhanced Content Discovery Routes
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
                    
                    # Get streaming platforms with language support
                    streaming_platforms = json.loads(content.ott_platforms or '[]')
                    watch_buttons = LanguageService.create_watch_buttons(streaming_platforms)
                    
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
                        'ott_platforms': streaming_platforms,
                        'watch_buttons': watch_buttons
                    })
        
        # Add anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                results.append({
                    'id': f"anime_{anime['mal_id']}",
                    'mal_id': anime['mal_id'],
                    'title': anime.get('title'),
                    'content_type': 'anime',
                    'genres': [genre['name'] for genre in anime.get('genres', [])],
                    'rating': anime.get('score'),
                    'release_date': anime.get('aired', {}).get('from'),
                    'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                    'overview': anime.get('synopsis'),
                    'ott_platforms': [],
                    'watch_buttons': []
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
        
        # Get real-time streaming availability with enhanced language support
        streaming_platforms = []
        if content.tmdb_id:
            streaming_platforms = StreamingAvailabilityService.get_comprehensive_streaming_info(
                content.tmdb_id, content.title
            )
        
        # If no streaming data from API, use stored data
        if not streaming_platforms:
            try:
                streaming_platforms = json.loads(content.ott_platforms or '[]')
            except:
                streaming_platforms = []
        
        # Update stored streaming data if we got new data
        if streaming_platforms and content.tmdb_id:
            try:
                content.ott_platforms = json.dumps(streaming_platforms)
                content.streaming_updated_at = datetime.utcnow()
            except:
                pass
        
        # Create language-specific watch buttons
        watch_buttons = LanguageService.create_watch_buttons(streaming_platforms)
        language_groups = LanguageService.group_platforms_by_language(streaming_platforms)
        
        # Get YouTube trailers
        trailers = []
        if YOUTUBE_API_KEY and YOUTUBE_API_KEY != 'your_youtube_api_key':
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
            'ott_platforms': streaming_platforms,  # Raw platform data
            'watch_buttons': watch_buttons,  # Language-specific watch buttons
            'language_groups': language_groups,  # Grouped by language for UI
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# New route for streaming platforms with language filtering
@app.route('/api/content/<int:content_id>/streaming', methods=['GET'])
def get_content_streaming_info(content_id):
    try:
        content = Content.query.get_or_404(content_id)
        language_filter = request.args.get('language')  # Optional language filter
        
        # Get real-time streaming availability
        streaming_platforms = []
        if content.tmdb_id:
            streaming_platforms = StreamingAvailabilityService.get_comprehensive_streaming_info(
                content.tmdb_id, content.title
            )
        
        # If no data from API, use stored data
        if not streaming_platforms:
            try:
                streaming_platforms = json.loads(content.ott_platforms or '[]')
            except:
                streaming_platforms = []
        
        # Filter by language if requested
        if language_filter:
            streaming_platforms = [
                platform for platform in streaming_platforms
                if platform.get('audio_language', {}).get('name', '').lower() == language_filter.lower()
            ]
        
        # Create watch buttons
        watch_buttons = LanguageService.create_watch_buttons(streaming_platforms)
        language_groups = LanguageService.group_platforms_by_language(streaming_platforms)
        
        return jsonify({
            'content_id': content.id,
            'content_title': content.title,
            'streaming_platforms': streaming_platforms,
            'watch_buttons': watch_buttons,
            'language_groups': language_groups,
            'last_updated': content.streaming_updated_at.isoformat() if content.streaming_updated_at else None
        }), 200
        
    except Exception as e:
        logger.error(f"Streaming info error: {e}")
        return jsonify({'error': 'Failed to get streaming information'}), 500

# Anime Details Route
@app.route('/api/anime/<int:anime_id>', methods=['GET'])
def get_anime_details(anime_id):
    try:
        # Get anime details from Jikan API
        anime_data = JikanService.get_anime_details(anime_id)
        
        if not anime_data or 'data' not in anime_data:
            return jsonify({'error': 'Anime not found'}), 404
        
        anime = anime_data['data']
        
        # Record view interaction for anonymous users
        session_id = get_session_id()
        
        # Save anime to database
        content = ContentService.save_anime_content(anime)
        
        if content:
            interaction = AnonymousInteraction(
                session_id=session_id,
                content_id=content.id,
                interaction_type='view',
                ip_address=request.remote_addr
            )
            db.session.add(interaction)
        
        # Format anime details
        anime_details = {
            'id': anime_id,
            'mal_id': anime['mal_id'],
            'title': anime.get('title'),
            'title_english': anime.get('title_english'),
            'title_japanese': anime.get('title_japanese'),
            'content_type': 'anime',
            'type': anime.get('type'),
            'source': anime.get('source'),
            'episodes': anime.get('episodes'),
            'status': anime.get('status'),
            'duration': anime.get('duration'),
            'rating': anime.get('score'),
            'scored_by': anime.get('scored_by'),
            'rank': anime.get('rank'),
            'popularity': anime.get('popularity'),
            'synopsis': anime.get('synopsis'),
            'year': anime.get('year'),
            'season': anime.get('season'),
            'genres': [genre['name'] for genre in anime.get('genres', [])],
            'themes': [theme['name'] for theme in anime.get('themes', [])],
            'demographics': [demo['name'] for demo in anime.get('demographics', [])],
            'studios': [studio['name'] for studio in anime.get('studios', [])],
            'producers': [producer['name'] for producer in anime.get('producers', [])],
            'poster_path': anime.get('images', {}).get('jpg', {}).get('large_image_url'),
            'trailer_url': anime.get('trailer', {}).get('url'),
            'mal_url': anime.get('url'),
            'aired': {
                'from': anime.get('aired', {}).get('from'),
                'to': anime.get('aired', {}).get('to'),
                'string': anime.get('aired', {}).get('string')
            }
        }
        
        # Get streaming platforms for anime
        anime_streaming_platforms = [
            {'platform': 'crunchyroll', 'platform_name': 'Crunchyroll', 'is_free': True, 'url': 'https://crunchyroll.com', 'audio_language': {'code': 'ja', 'name': 'Japanese'}},
            {'platform': 'funimation', 'platform_name': 'Funimation', 'is_free': False, 'url': 'https://funimation.com', 'audio_language': {'code': 'en', 'name': 'English'}},
        ]
        
        watch_buttons = LanguageService.create_watch_buttons(anime_streaming_platforms)
        
        anime_details['ott_platforms'] = anime_streaming_platforms
        anime_details['watch_buttons'] = watch_buttons
        
        db.session.commit()
        
        return jsonify(anime_details), 200
        
    except Exception as e:
        logger.error(f"Anime details error: {e}")
        return jsonify({'error': 'Failed to get anime details'}), 500

# Regional Movie Routes
@app.route('/api/regional/<language>/best', methods=['GET'])
def get_regional_best(language):
    try:
        limit = int(request.args.get('limit', 20))
        movies = RegionalContentService.get_regional_best_movies(language, limit)
        return jsonify({'movies': movies, 'category': 'best', 'language': language}), 200
    except Exception as e:
        logger.error(f"Regional best error: {e}")
        return jsonify({'error': 'Failed to get best movies'}), 500

@app.route('/api/regional/<language>/trending', methods=['GET'])
def get_regional_trending_movies(language):
    try:
        limit = int(request.args.get('limit', 20))
        movies = RegionalContentService.get_regional_trending(language, limit)
        return jsonify({'movies': movies, 'category': 'trending', 'language': language}), 200
    except Exception as e:
        logger.error(f"Regional trending error: {e}")
        return jsonify({'error': 'Failed to get trending movies'}), 500

@app.route('/api/regional/<language>/new-releases', methods=['GET'])
def get_regional_new_releases_route(language):
    try:
        limit = int(request.args.get('limit', 20))
        movies = RegionalContentService.get_regional_new_releases(language, limit)
        return jsonify({'movies': movies, 'category': 'new-releases', 'language': language}), 200
    except Exception as e:
        logger.error(f"Regional new releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

@app.route('/api/regional/<language>/genre/<genre>', methods=['GET'])
def get_regional_genre_movies(language, genre):
    try:
        limit = int(request.args.get('limit', 20))
        movies = RegionalContentService.get_regional_by_genre(language, genre, limit)
        return jsonify({'movies': movies, 'category': f'{genre}', 'language': language}), 200
    except Exception as e:
        logger.error(f"Regional genre error: {e}")
        return jsonify({'error': 'Failed to get genre movies'}), 500

@app.route('/api/regional/languages', methods=['GET'])
def get_supported_languages():
    return jsonify({
        'languages': RegionalContentService.LANGUAGE_PRIORITY,
        'genres': RegionalContentService.GENRES
    }), 200

@app.route('/api/regional/<language>/all-categories', methods=['GET'])
def get_all_regional_categories(language):
    try:
        limit_per_category = int(request.args.get('limit', 10))
        
        categories = {
            'best_movies': RegionalContentService.get_regional_best_movies(language, limit_per_category),
            'trending': RegionalContentService.get_regional_trending(language, limit_per_category),
            'new_releases': RegionalContentService.get_regional_new_releases(language, limit_per_category),
            'action': RegionalContentService.get_regional_by_genre(language, 'action', limit_per_category),
            'drama': RegionalContentService.get_regional_by_genre(language, 'drama', limit_per_category),
            'comedy': RegionalContentService.get_regional_by_genre(language, 'comedy', limit_per_category),
            'romance': RegionalContentService.get_regional_by_genre(language, 'romance', limit_per_category),
            'thriller': RegionalContentService.get_regional_by_genre(language, 'thriller', limit_per_category)
        }
        
        return jsonify({
            'language': language,
            'categories': categories
        }), 200
    except Exception as e:
        logger.error(f"All regional categories error: {e}")
        return jsonify({'error': 'Failed to get regional categories'}), 500

# Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = []
        for content in recommendations:
            # Get streaming platforms with watch buttons
            streaming_platforms = json.loads(content.ott_platforms or '[]')
            watch_buttons = LanguageService.create_watch_buttons(streaming_platforms)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': streaming_platforms,
                'watch_buttons': watch_buttons
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
            # Get streaming platforms with watch buttons
            streaming_platforms = json.loads(content.ott_platforms or '[]')
            watch_buttons = LanguageService.create_watch_buttons(streaming_platforms)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': streaming_platforms,
                'watch_buttons': watch_buttons
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
                'id': content.id,
                'mal_id': content.mal_id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]'),
                'watch_buttons': []
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
            # Get streaming platforms with watch buttons
            streaming_platforms = json.loads(content.ott_platforms or '[]')
            watch_buttons = LanguageService.create_watch_buttons(streaming_platforms)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': streaming_platforms,
                'watch_buttons': watch_buttons
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
            # Get streaming platforms with watch buttons
            streaming_platforms = json.loads(content.ott_platforms or '[]')
            watch_buttons = LanguageService.create_watch_buttons(streaming_platforms)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': streaming_platforms,
                'watch_buttons': watch_buttons
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
            # Get streaming platforms with watch buttons
            streaming_platforms = json.loads(content.ott_platforms or '[]')
            watch_buttons = LanguageService.create_watch_buttons(streaming_platforms)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': streaming_platforms,
                'watch_buttons': watch_buttons
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
            if data.get('source') == 'anime':
                existing_content = Content.query.filter_by(mal_id=data['id']).first()
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
                    release_date = datetime.strptime(data['release_date'][:10], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            # Create content object
            content = Content(
                tmdb_id=data.get('id') if data.get('source') != 'anime' else None,
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
                ott_platforms=json.dumps(data.get('ott_platforms', [])),
                streaming_updated_at=datetime.utcnow()
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
                try:
                    genres = json.loads(content.genres)
                    for genre in genres:
                        genre_counts[genre] += 1
                except:
                    pass
        
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
                # Get streaming platforms with watch buttons
                streaming_platforms = json.loads(content.ott_platforms or '[]')
                watch_buttons = LanguageService.create_watch_buttons(streaming_platforms)
                
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'ott_platforms': streaming_platforms,
                    'watch_buttons': watch_buttons,
                    'admin_description': rec.description,
                    'admin_name': admin.username if admin else 'Admin',
                    'recommended_at': rec.created_at.isoformat()
                })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Public admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get admin recommendations'}), 500

# Test endpoint for streaming APIs
@app.route('/api/test/streaming', methods=['GET'])
def test_streaming_apis():
    """Test endpoint to check streaming API functionality"""
    try:
        tmdb_id = request.args.get('tmdb_id', '248852')  # Default to one of the failing IDs
        title = request.args.get('title', 'UNTAMED')
        
        results = {
            'tmdb_id': tmdb_id,
            'title': title,
            'streaming_api_result': None,
            'watchmode_api_result': None,
            'comprehensive_result': None
        }
        
        # Test Streaming Availability API
        try:
            streaming_data = StreamingAvailabilityService.get_streaming_info_by_tmdb(tmdb_id)
            results['streaming_api_result'] = {
                'success': streaming_data is not None,
                'data': streaming_data if streaming_data else 'No data or 404 error'
            }
        except Exception as e:
            results['streaming_api_result'] = {'success': False, 'error': str(e)}
        
        # Test Watchmode API
        try:
            watchmode_data = StreamingAvailabilityService.search_watchmode_api(title)
            results['watchmode_api_result'] = {
                'success': watchmode_data is not None,
                'data': watchmode_data if watchmode_data else 'No data found'
            }
        except Exception as e:
            results['watchmode_api_result'] = {'success': False, 'error': str(e)}
        
        # Test comprehensive approach
        try:
            comprehensive_data = StreamingAvailabilityService.get_comprehensive_streaming_info(
                tmdb_id, title
            )
            results['comprehensive_result'] = {
                'success': len(comprehensive_data) > 0,
                'platform_count': len(comprehensive_data),
                'platforms': comprehensive_data
            }
        except Exception as e:
            results['comprehensive_result'] = {'success': False, 'error': str(e)}
        
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.1.0',
        'features': {
            'multi_language_streaming': True,
            'enhanced_api_fallback': True,
            'language_specific_buttons': True,
            'watchmode_fallback': True,
            'comprehensive_streaming_search': True
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