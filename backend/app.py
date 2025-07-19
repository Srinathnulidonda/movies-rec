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

# API Keys - Set these in your environment
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '52260795')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '-1002850793757')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-xmf5.onrender.com')

# Streaming API Configuration
WATCHMODE_API_KEY = 'WtcKDji9i20pjOl5Lg0AiyG2bddfUs3nSZRZJIsY'
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

# Enhanced Streaming Availability Service
class StreamingAvailabilityService:
    # Use Watchmode API as primary
    WATCHMODE_BASE_URL = 'https://api.watchmode.com/v1'
    RAPIDAPI_BASE_URL = 'https://streaming-availability.p.rapidapi.com'
    
    RAPIDAPI_HEADERS = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': RAPIDAPI_HOST
    }
    
    # Enhanced platform mapping with language support
    PLATFORM_MAPPING = {
        'netflix': {
            'name': 'Netflix', 
            'is_free': False, 
            'url': 'https://netflix.com',
            'languages': ['hindi', 'english', 'telugu', 'tamil'],
            'source_ids': [203]
        },
        'prime': {
            'name': 'Amazon Prime Video', 
            'is_free': False, 
            'url': 'https://primevideo.com',
            'languages': ['hindi', 'english', 'telugu', 'tamil'],
            'source_ids': [26]
        },
        'hotstar': {
            'name': 'Disney+ Hotstar', 
            'is_free': False, 
            'url': 'https://hotstar.com',
            'languages': ['hindi', 'english', 'telugu', 'tamil'],
            'source_ids': [377]
        },
        'zee5': {
            'name': 'ZEE5', 
            'is_free': False, 
            'url': 'https://zee5.com',
            'languages': ['hindi', 'english', 'telugu', 'tamil'],
            'source_ids': [218]
        },
        'sonyliv': {
            'name': 'SonyLIV', 
            'is_free': False, 
            'url': 'https://sonyliv.com',
            'languages': ['hindi', 'english', 'telugu', 'tamil'],
            'source_ids': [283]
        },
        'voot': {
            'name': 'Voot', 
            'is_free': True, 
            'url': 'https://voot.com',
            'languages': ['hindi', 'english'],
            'source_ids': [102]
        },
        'mx': {
            'name': 'MX Player', 
            'is_free': True, 
            'url': 'https://mxplayer.in',
            'languages': ['hindi', 'english', 'telugu', 'tamil'],
            'source_ids': [75]
        },
        'youtube': {
            'name': 'YouTube', 
            'is_free': True, 
            'url': 'https://youtube.com',
            'languages': ['hindi', 'english', 'telugu', 'tamil'],
            'source_ids': [371]
        },
        'jiocinema': {
            'name': 'JioCinema', 
            'is_free': True, 
            'url': 'https://jiocinema.com',
            'languages': ['hindi', 'english', 'telugu', 'tamil'],
            'source_ids': [105]
        },
        'aha': {
            'name': 'Aha', 
            'is_free': False, 
            'url': 'https://aha.video',
            'languages': ['telugu'],
            'source_ids': [452]
        },
        'sunnxt': {
            'name': 'Sun NXT', 
            'is_free': False, 
            'url': 'https://sunnxt.com',
            'languages': ['tamil', 'telugu'],
            'source_ids': [309]
        },
        'erosnow': {
            'name': 'Eros Now', 
            'is_free': False, 
            'url': 'https://erosnow.com',
            'languages': ['hindi', 'english'],
            'source_ids': [148]
        },
        'airtel': {
            'name': 'Airtel Xstream', 
            'is_free': True, 
            'url': 'https://airtelxstream.in',
            'languages': ['hindi', 'english'],
            'source_ids': [236]
        },
        'crunchyroll': {
            'name': 'Crunchyroll', 
            'is_free': True, 
            'url': 'https://crunchyroll.com',
            'languages': ['english'],
            'source_ids': [35]
        },
    }
    
    @staticmethod
    def search_by_watchmode(title, content_type='movie'):
        """Search using Watchmode API"""
        try:
            url = f"{StreamingAvailabilityService.WATCHMODE_BASE_URL}/search/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'search_field': 'name',
                'search_value': title,
                'types': content_type
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('title_results'):
                    # Get the first result and fetch its sources
                    first_result = data['title_results'][0]
                    title_id = first_result.get('id')
                    if title_id:
                        return StreamingAvailabilityService.get_title_sources(title_id)
            else:
                logger.warning(f"Watchmode API returned status {response.status_code} for title: {title}")
        except Exception as e:
            logger.error(f"Watchmode search error: {e}")
        return None
    
    @staticmethod
    def get_title_sources(title_id):
        """Get streaming sources for a specific title"""
        try:
            url = f"{StreamingAvailabilityService.WATCHMODE_BASE_URL}/title/{title_id}/sources/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'regions': 'IN'  # India region
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Watchmode sources API returned status {response.status_code} for title ID: {title_id}")
        except Exception as e:
            logger.error(f"Watchmode sources error: {e}")
        return None
    
    @staticmethod
    def search_by_rapidapi_fallback(title, country='in'):
        """Fallback to RapidAPI if Watchmode fails"""
        try:
            url = f"{StreamingAvailabilityService.RAPIDAPI_BASE_URL}/search/title"
            params = {
                'title': title,
                'country': country,
                'show_type': 'all',
                'output_language': 'en'
            }
            
            response = requests.get(
                url, 
                headers=StreamingAvailabilityService.RAPIDAPI_HEADERS, 
                params=params, 
                timeout=15
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"RapidAPI returned status {response.status_code} for title: {title}")
        except Exception as e:
            logger.error(f"RapidAPI fallback error: {e}")
        return None
    
    @staticmethod
    def get_streaming_info(tmdb_id, title, content_type='movie'):
        """Get streaming information using multiple APIs"""
        # First try Watchmode
        streaming_data = StreamingAvailabilityService.search_by_watchmode(title, content_type)
        
        # If Watchmode fails, try RapidAPI
        if not streaming_data:
            streaming_data = StreamingAvailabilityService.search_by_rapidapi_fallback(title)
        
        # If both fail, return mock data for common platforms
        if not streaming_data:
            return StreamingAvailabilityService.get_mock_streaming_data(title)
        
        return streaming_data
    
    @staticmethod
    def get_mock_streaming_data(title):
        """Generate mock streaming data when APIs fail"""
        # Common platforms that likely have content
        mock_platforms = [
            {
                'name': 'Netflix', 
                'source_id': 203, 
                'type': 'sub', 
                'region': 'IN', 
                'android_url': 'https://netflix.com/search?q=' + quote(title), 
                'web_url': 'https://netflix.com/search?q=' + quote(title)
            },
            {
                'name': 'Amazon Prime Video', 
                'source_id': 26, 
                'type': 'sub', 
                'region': 'IN', 
                'android_url': 'https://primevideo.com/search/ref=atv_nb_sr?phrase=' + quote(title), 
                'web_url': 'https://primevideo.com/search/ref=atv_nb_sr?phrase=' + quote(title)
            },
            {
                'name': 'Disney+ Hotstar', 
                'source_id': 377, 
                'type': 'sub', 
                'region': 'IN', 
                'android_url': 'https://hotstar.com/search?q=' + quote(title), 
                'web_url': 'https://hotstar.com/search?q=' + quote(title)
            },
            {
                'name': 'YouTube', 
                'source_id': 371, 
                'type': 'free', 
                'region': 'IN', 
                'android_url': 'https://youtube.com/results?search_query=' + quote(title), 
                'web_url': 'https://youtube.com/results?search_query=' + quote(title)
            },
            {
                'name': 'MX Player', 
                'source_id': 75, 
                'type': 'free', 
                'region': 'IN', 
                'android_url': 'https://mxplayer.in/search?q=' + quote(title), 
                'web_url': 'https://mxplayer.in/search?q=' + quote(title)
            }
        ]
        return mock_platforms
    
    @staticmethod
    def format_streaming_data_with_languages(streaming_data, title=None):
        """Format streaming data with proper language separation"""
        if not streaming_data:
            return {'hindi': [], 'telugu': [], 'tamil': [], 'english': []}
        
        language_platforms = {
            'hindi': [],
            'telugu': [],
            'tamil': [],
            'english': []
        }
        
        try:
            # Handle Watchmode format
            if isinstance(streaming_data, list):
                platforms = streaming_data
            else:
                platforms = streaming_data.get('sources', streaming_data)
            
            # If no platforms found, return empty
            if not platforms:
                return language_platforms
            
            for platform in platforms:
                platform_name = platform.get('name', 'Unknown')
                platform_id = platform.get('source_id', 0)
                
                # Map to our platform system
                mapped_platform = None
                for key, info in StreamingAvailabilityService.PLATFORM_MAPPING.items():
                    if (info['name'].lower() in platform_name.lower() or 
                        platform_id in info.get('source_ids', [])):
                        mapped_platform = info
                        break
                
                if not mapped_platform:
                    # Create default mapping for unknown platforms
                    mapped_platform = {
                        'name': platform_name,
                        'is_free': platform.get('type') in ['free', 'ad', 'ads'],
                        'url': platform.get('web_url', ''),
                        'languages': ['hindi', 'english']  # Default assumption
                    }
                
                # Create platform entry for each supported language
                for language in mapped_platform.get('languages', ['english']):
                    if language in language_platforms:
                        # Generate language-specific URLs
                        base_url = platform.get('web_url') or platform.get('android_url') or mapped_platform['url']
                        language_url = StreamingAvailabilityService.generate_language_url(base_url, title, language)
                        
                        platform_entry = {
                            'platform': platform_name.lower().replace(' ', '').replace('+', ''),
                            'platform_name': platform_name,
                            'is_free': mapped_platform['is_free'],
                            'url': language_url,
                            'language': language,
                            'type': platform.get('type', 'subscription'),
                            'region': platform.get('region', 'IN')
                        }
                        
                        # Avoid duplicates
                        if not any(p['platform'] == platform_entry['platform'] for p in language_platforms[language]):
                            language_platforms[language].append(platform_entry)
        
        except Exception as e:
            logger.error(f"Error formatting streaming data: {e}")
        
        return language_platforms
    
    @staticmethod
    def generate_language_url(base_url, title, language):
        """Generate language-specific URLs for platforms"""
        if not base_url or not title:
            return base_url
        
        try:
            # Platform-specific URL generation
            if 'netflix.com' in base_url:
                return f"https://netflix.com/search?q={quote(title)}"
            elif 'primevideo.com' in base_url:
                return f"https://primevideo.com/search/ref=atv_nb_sr?phrase={quote(title)}"
            elif 'hotstar.com' in base_url:
                return f"https://hotstar.com/search?q={quote(title)}"
            elif 'zee5.com' in base_url:
                return f"https://zee5.com/search?q={quote(title)}"
            elif 'sonyliv.com' in base_url:
                return f"https://sonyliv.com/search?q={quote(title)}"
            elif 'youtube.com' in base_url:
                search_query = f"{title} {language} movie" if language != 'english' else f"{title} movie"
                return f"https://youtube.com/results?search_query={quote(search_query)}"
            elif 'mxplayer.in' in base_url:
                return f"https://mxplayer.in/search?q={quote(title)}"
            elif 'jiocinema.com' in base_url:
                return f"https://jiocinema.com/search?q={quote(title)}"
            elif 'aha.video' in base_url:
                return f"https://aha.video/search?q={quote(title)}"
            elif 'sunnxt.com' in base_url:
                return f"https://sunnxt.com/search?q={quote(title)}"
            else:
                return base_url
        except:
            return base_url

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
                        'ott_platforms': json.loads(content.ott_platforms or '{}')
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

# Content Management Service
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
            
            # Get OTT platforms with language separation
            ott_platforms = ContentService.get_ott_availability(tmdb_data)
            
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
                ott_platforms=json.dumps({'hindi': [], 'telugu': [], 'tamil': [], 'english': []}),
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
                ott_platforms = ContentService.get_ott_availability({'id': content.tmdb_id, 'title': content.title})
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
    def get_ott_availability(tmdb_data):
        """Get OTT availability using improved streaming service"""
        language_platforms = {'hindi': [], 'telugu': [], 'tamil': [], 'english': []}
        
        try:
            title = tmdb_data.get('title') or tmdb_data.get('name')
            content_type = 'movie' if 'title' in tmdb_data else 'tv'
            
            if title:
                # Get streaming info using the updated service
                streaming_data = StreamingAvailabilityService.get_streaming_info(
                    tmdb_data.get('id'), title, content_type
                )
                language_platforms = StreamingAvailabilityService.format_streaming_data_with_languages(streaming_data, title)
        except Exception as e:
            logger.error(f"Error getting OTT availability: {e}")
        
        return language_platforms

# Enhanced Telegram Service with Classic Design
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
            
            # Get real-time streaming availability with language separation
            streaming_platforms = {'hindi': [], 'telugu': [], 'tamil': [], 'english': []}
            if content.tmdb_id:
                streaming_data = StreamingAvailabilityService.get_streaming_info(
                    content.tmdb_id, content.title, content.content_type
                )
                streaming_platforms = StreamingAvailabilityService.format_streaming_data_with_languages(streaming_data)
            
            # If no real-time data, try stored data
            if not any(streaming_platforms.values()):
                try:
                    stored_platforms = json.loads(content.ott_platforms or '{}')
                    if isinstance(stored_platforms, dict):
                        streaming_platforms = stored_platforms
                except:
                    pass
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Build classic Telegram message design
            message = TelegramService.create_classic_movie_post(
                content, admin_name, description, genres_list, streaming_platforms
            )
            
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
    def create_classic_movie_post(content, admin_name, description, genres_list, streaming_platforms):
        """Create a classic cinema-style Telegram post"""
        
        # Header with movie title
        message = f"""🎬═══════════════════════🎬
<b>✨ ADMIN'S CHOICE ✨</b>
<i>Curated by {admin_name}</i>
🎬═══════════════════════🎬

🎭 <b>{content.title}</b>"""
        
        if content.original_title and content.original_title != content.title:
            message += f"\n<i>({content.original_title})</i>"
        
        # Movie details section
        message += f"""

📊 <b>MOVIE DETAILS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━
⭐ <b>Rating:</b> {content.rating or 'N/A'}/10
📅 <b>Release:</b> {content.release_date or 'N/A'}
🎭 <b>Genres:</b> {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 <b>Type:</b> {content.content_type.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        # Admin's note
        message += f"""

💭 <b>ADMIN'S REVIEW</b>
━━━━━━━━━━━━━━━━━━━━━━━━━
{description}
━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        # Streaming availability by language
        message += f"\n\n🔗 <b>WHERE TO WATCH</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Hindi content
        if streaming_platforms.get('hindi'):
            message += f"\n\n🇮🇳 <b>HINDI</b>"
            free_platforms = []
            paid_platforms = []
            
            for platform in streaming_platforms['hindi'][:4]:  # Limit to 4 platforms
                platform_name = platform.get('platform_name', platform.get('platform', '').title())
                platform_url = platform.get('url', '')
                
                if platform.get('is_free'):
                    if platform_url:
                        free_platforms.append(f"<a href='{platform_url}'>▶️ {platform_name}</a>")
                    else:
                        free_platforms.append(f"▶️ {platform_name}")
                else:
                    if platform_url:
                        paid_platforms.append(f"<a href='{platform_url}'>💰 {platform_name}</a>")
                    else:
                        paid_platforms.append(f"💰 {platform_name}")
            
            if free_platforms:
                message += f"\n🆓 <b>Free:</b> {' | '.join(free_platforms)}"
            if paid_platforms:
                message += f"\n💰 <b>Paid:</b> {' | '.join(paid_platforms)}"
        
        # Telugu content
        if streaming_platforms.get('telugu'):
            message += f"\n\n🅰️ <b>TELUGU</b>"
            free_platforms = []
            paid_platforms = []
            
            for platform in streaming_platforms['telugu'][:4]:
                platform_name = platform.get('platform_name', platform.get('platform', '').title())
                platform_url = platform.get('url', '')
                
                if platform.get('is_free'):
                    if platform_url:
                        free_platforms.append(f"<a href='{platform_url}'>▶️ {platform_name}</a>")
                    else:
                        free_platforms.append(f"▶️ {platform_name}")
                else:
                    if platform_url:
                        paid_platforms.append(f"<a href='{platform_url}'>💰 {platform_name}</a>")
                    else:
                        paid_platforms.append(f"💰 {platform_name}")
            
            if free_platforms:
                message += f"\n🆓 <b>Free:</b> {' | '.join(free_platforms)}"
            if paid_platforms:
                message += f"\n💰 <b>Paid:</b> {' | '.join(paid_platforms)}"
        
        # Tamil content
        if streaming_platforms.get('tamil'):
            message += f"\n\n🔤 <b>TAMIL</b>"
            free_platforms = []
            paid_platforms = []
            
            for platform in streaming_platforms['tamil'][:4]:
                platform_name = platform.get('platform_name', platform.get('platform', '').title())
                platform_url = platform.get('url', '')
                
                if platform.get('is_free'):
                    if platform_url:
                        free_platforms.append(f"<a href='{platform_url}'>▶️ {platform_name}</a>")
                    else:
                        free_platforms.append(f"▶️ {platform_name}")
                else:
                    if platform_url:
                        paid_platforms.append(f"<a href='{platform_url}'>💰 {platform_name}</a>")
                    else:
                        paid_platforms.append(f"💰 {platform_name}")
            
            if free_platforms:
                message += f"\n🆓 <b>Free:</b> {' | '.join(free_platforms)}"
            if paid_platforms:
                message += f"\n💰 <b>Paid:</b> {' | '.join(paid_platforms)}"
        
        # English content
        if streaming_platforms.get('english'):
            message += f"\n\n🇺🇸 <b>ENGLISH</b>"
            free_platforms = []
            paid_platforms = []
            
            for platform in streaming_platforms['english'][:4]:
                platform_name = platform.get('platform_name', platform.get('platform', '').title())
                platform_url = platform.get('url', '')
                
                if platform.get('is_free'):
                    if platform_url:
                        free_platforms.append(f"<a href='{platform_url}'>▶️ {platform_name}</a>")
                    else:
                        free_platforms.append(f"▶️ {platform_name}")
                else:
                    if platform_url:
                        paid_platforms.append(f"<a href='{platform_url}'>💰 {platform_name}</a>")
                    else:
                        paid_platforms.append(f"💰 {platform_name}")
            
            if free_platforms:
                message += f"\n🆓 <b>Free:</b> {' | '.join(free_platforms)}"
            if paid_platforms:
                message += f"\n💰 <b>Paid:</b> {' | '.join(paid_platforms)}"
        
        # If no streaming platforms found
        if not any(streaming_platforms.values()):
            message += f"\n\n🔍 <b>Search on popular platforms:</b>"
            search_title = quote(content.title)
            message += f"\n<a href='https://netflix.com/search?q={search_title}'>🔴 Netflix</a> | "
            message += f"<a href='https://primevideo.com/search/ref=atv_nb_sr?phrase={search_title}'>📺 Prime Video</a> | "
            message += f"<a href='https://hotstar.com/search?q={search_title}'>⭐ Hotstar</a>"
        
        # Synopsis
        if content.overview:
            synopsis = content.overview[:300] + '...' if len(content.overview) > 300 else content.overview
            message += f"""

📖 <b>SYNOPSIS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━
{synopsis}
━━━━━━━━━━━━━━━━━━━━━━━━━"""
        
        # Footer
        message += f"""

🎬═══════════════════════🎬
<b>#AdminChoice #MovieRecommendation #CineScope</b>
<i>Join us for more amazing recommendations!</i>
🎬═══════════════════════🎬"""
        
        return message

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
                        'ott_platforms': json.loads(content.ott_platforms or '{}')
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
                    'ott_platforms': {'hindi': [], 'telugu': [], 'tamil': [], 'english': []}
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
        
        # Get real-time streaming availability
        streaming_platforms = {'hindi': [], 'telugu': [], 'tamil': [], 'english': []}
        if content.tmdb_id:
            streaming_data = StreamingAvailabilityService.get_streaming_info(content.tmdb_id, content.title, content.content_type)
            streaming_platforms = StreamingAvailabilityService.format_streaming_data_with_languages(streaming_data)
        
        # If no streaming data from API, use stored data
        if not any(streaming_platforms.values()):
            try:
                stored_platforms = json.loads(content.ott_platforms or '{}')
                if isinstance(stored_platforms, dict):
                    streaming_platforms = stored_platforms
            except:
                streaming_platforms = {'hindi': [], 'telugu': [], 'tamil': [], 'english': []}
        
        # Update stored streaming data if we got new data
        if any(streaming_platforms.values()) and content.tmdb_id:
            try:
                content.ott_platforms = json.dumps(streaming_platforms)
                content.streaming_updated_at = datetime.utcnow()
            except:
                pass
        
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
            'ott_platforms': streaming_platforms,  # Language-separated streaming data
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

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
                'ott_platforms': json.loads(content.ott_platforms or '{}')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

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
                'ott_platforms': json.loads(content.ott_platforms or '{}')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Anonymous recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

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
                ott_platforms=json.dumps({'hindi': [], 'telugu': [], 'tamil': [], 'english': []}),
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

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0'
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