# backend/app.py
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
from collections import defaultdict, Counter
import hashlib
import time
from sqlalchemy import func, desc
import telebot
import jwt
import urllib.parse

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

# API Keys
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
try:
    bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
except:
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

# Known OTT Platforms with direct watch capabilities
KNOWN_OTT_PLATFORMS = {
    'netflix': {
        'name': 'Netflix',
        'base_url': 'https://www.netflix.com',
        'watch_url': 'https://www.netflix.com/watch/{content_id}',
        'search_url': 'https://www.netflix.com/search?q={query}',
        'is_free': False,
        'regions': ['IN', 'US', 'UK'],
        'languages': ['hindi', 'english', 'tamil', 'telugu']
    },
    'prime': {
        'name': 'Amazon Prime Video',
        'base_url': 'https://www.primevideo.com',
        'watch_url': 'https://www.primevideo.com/detail/{content_id}',
        'search_url': 'https://www.primevideo.com/search/ref=atv_nb_sr?phrase={query}',
        'is_free': False,
        'regions': ['IN', 'US', 'UK'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'kannada', 'malayalam']
    },
    'hotstar': {
        'name': 'Disney+ Hotstar',
        'base_url': 'https://www.hotstar.com',
        'watch_url': 'https://www.hotstar.com/{country}/movies/{slug}/{content_id}',
        'search_url': 'https://www.hotstar.com/in/search?q={query}',
        'is_free': False,
        'regions': ['IN'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'kannada', 'malayalam', 'bengali']
    },
    'jiocinema': {
        'name': 'JioCinema',
        'base_url': 'https://www.jiocinema.com',
        'watch_url': 'https://www.jiocinema.com/movies/{slug}/{content_id}',
        'search_url': 'https://www.jiocinema.com/search/{query}',
        'is_free': True,
        'regions': ['IN'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'kannada', 'malayalam']
    },
    'zee5': {
        'name': 'ZEE5',
        'base_url': 'https://www.zee5.com',
        'watch_url': 'https://www.zee5.com/movies/details/{slug}/{content_id}',
        'search_url': 'https://www.zee5.com/search?q={query}',
        'is_free': False,
        'regions': ['IN'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'kannada', 'malayalam', 'bengali']
    },
    'sonyliv': {
        'name': 'SonyLIV',
        'base_url': 'https://www.sonyliv.com',
        'watch_url': 'https://www.sonyliv.com/movies/{slug}/{content_id}',
        'search_url': 'https://www.sonyliv.com/search?q={query}',
        'is_free': False,
        'regions': ['IN'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'kannada', 'malayalam']
    },
    'mxplayer': {
        'name': 'MX Player',
        'base_url': 'https://www.mxplayer.in',
        'watch_url': 'https://www.mxplayer.in/movie/{slug}/{content_id}',
        'search_url': 'https://www.mxplayer.in/search?q={query}',
        'is_free': True,
        'regions': ['IN'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'kannada', 'malayalam']
    },
    'youtube': {
        'name': 'YouTube',
        'base_url': 'https://www.youtube.com',
        'watch_url': 'https://www.youtube.com/watch?v={content_id}',
        'search_url': 'https://www.youtube.com/results?search_query={query}',
        'is_free': True,
        'regions': ['GLOBAL'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'kannada', 'malayalam']
    }
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

# Enhanced OTT Availability Services
class WatchModeService:
    BASE_URL = 'https://api.watchmode.com/v1'
    
    @staticmethod
    def search_title(title, title_type='movie'):
        """Search for content on WatchMode"""
        try:
            logger.info(f"WatchMode: Searching for '{title}'")
            url = f"{WatchModeService.BASE_URL}/search/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'search_field': 'name',
                'search_value': title,
                'types': title_type
            }
            
            response = requests.get(url, params=params, timeout=10)
            logger.info(f"WatchMode search response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"WatchMode found {len(data.get('title_results', []))} results")
                return data
            else:
                logger.error(f"WatchMode API error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"WatchMode search error: {e}")
        return None
    
    @staticmethod
    def get_title_sources(watchmode_id):
        """Get streaming sources for a title"""
        try:
            logger.info(f"WatchMode: Getting sources for ID {watchmode_id}")
            url = f"{WatchModeService.BASE_URL}/title/{watchmode_id}/sources/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'regions': 'IN,US'
            }
            
            response = requests.get(url, params=params, timeout=10)
            logger.info(f"WatchMode sources response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"WatchMode found {len(data)} sources")
                return data
            else:
                logger.error(f"WatchMode sources error: {response.status_code}")
        except Exception as e:
            logger.error(f"WatchMode sources error: {e}")
        return None

class StreamingAvailabilityService:
    BASE_URL = 'https://streaming-availability.p.rapidapi.com'
    
    @staticmethod
    def search_by_title(title, country='in'):
        """Search for content using the updated API endpoint"""
        try:
            logger.info(f"StreamingAvailability: Searching for '{title}' in {country}")
            url = f"{StreamingAvailabilityService.BASE_URL}/shows/search/title"
            headers = {
                'X-RapidAPI-Key': RAPIDAPI_KEY,
                'X-RapidAPI-Host': RAPIDAPI_HOST
            }
            params = {
                'title': title,
                'country': country,
                'series_granularity': 'show',
                'output_language': 'en'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            logger.info(f"StreamingAvailability response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"StreamingAvailability found results")
                return data
            else:
                logger.error(f"StreamingAvailability error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"StreamingAvailability search error: {e}")
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
            'page': page,
            'include_adult': False
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
    def get_watch_providers(content_id, content_type='movie'):
        """Get watch providers from TMDB"""
        url = f"{TMDBService.BASE_URL}/{content_type}/{content_id}/watch/providers"
        params = {
            'api_key': TMDB_API_KEY
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB watch providers error: {e}")
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

class JikanService:
    BASE_URL = 'https://api.jikan.moe/v4'
    
    @staticmethod
    def search_anime(query, page=1):
        url = f"{JikanService.BASE_URL}/anime"
        params = {
            'q': query,
            'page': page,
            'limit': 20,
            'sfw': True
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
            'page': page,
            'sfw': True
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
            'q': f"{query} official trailer",
            'part': 'snippet',
            'type': 'video',
            'maxResults': 5,
            'videoDefinition': 'high'
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
            elif 'imdb_id' in tmdb_data:
                imdb_id = tmdb_data['imdb_id']
            
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
        """Get OTT availability with multiple fallback methods"""
        platforms = []
        title = tmdb_data.get('title') or tmdb_data.get('name', '')
        content_type = 'movie' if tmdb_data.get('title') else 'tv'
        
        logger.info(f"Getting OTT availability for: {title}")
        
        try:
            # Method 1: Try TMDB Watch Providers
            if 'watch/providers' in tmdb_data and tmdb_data['watch/providers'].get('results'):
                tmdb_platforms = ContentService.parse_tmdb_providers(tmdb_data['watch/providers']['results'])
                if tmdb_platforms:
                    platforms.extend(tmdb_platforms)
                    logger.info(f"Found {len(tmdb_platforms)} platforms from TMDB")
            
            # Method 2: Try WatchMode API
            if WATCHMODE_API_KEY and not platforms:
                watchmode_results = WatchModeService.search_title(title, content_type)
                if watchmode_results and watchmode_results.get('title_results'):
                    first_result = watchmode_results['title_results'][0]
                    watchmode_id = first_result.get('id')
                    
                    if watchmode_id:
                        sources = WatchModeService.get_title_sources(watchmode_id)
                        if sources:
                            watchmode_platforms = ContentService.parse_watchmode_sources(sources)
                            platforms.extend(watchmode_platforms)
                            logger.info(f"Found {len(watchmode_platforms)} platforms from WatchMode")
            
            # Method 3: Try Streaming Availability API
            if RAPIDAPI_KEY and not platforms:
                streaming_data = StreamingAvailabilityService.search_by_title(title)
                if streaming_data:
                    streaming_platforms = ContentService.parse_streaming_availability(streaming_data)
                    platforms.extend(streaming_platforms)
                    logger.info(f"Found {len(streaming_platforms)} platforms from StreamingAvailability")
            
            # Method 4: Use known platforms based on content metadata
            if not platforms:
                logger.info("No API results, using intelligent platform matching")
                platforms = ContentService.get_intelligent_platform_suggestions(tmdb_data)
            
            # Deduplicate and validate platforms
            unique_platforms = ContentService.deduplicate_platforms(platforms)
            
            logger.info(f"Total unique platforms found: {len(unique_platforms)}")
            return unique_platforms
            
        except Exception as e:
            logger.error(f"Error getting OTT availability: {e}")
            # Return intelligent suggestions on error
            return ContentService.get_intelligent_platform_suggestions(tmdb_data)
    
    @staticmethod
    def parse_tmdb_providers(providers_data):
        """Parse TMDB watch providers"""
        platforms = []
        
        # Check for India first, then US
        for region in ['IN', 'US']:
            if region in providers_data:
                region_data = providers_data[region]
                
                # Parse different types of availability
                for availability_type in ['flatrate', 'rent', 'buy', 'free']:
                    if availability_type in region_data:
                        for provider in region_data[availability_type]:
                            platform_key = ContentService.map_tmdb_provider_to_key(provider['provider_name'])
                            if platform_key and platform_key in KNOWN_OTT_PLATFORMS:
                                platform_info = KNOWN_OTT_PLATFORMS[platform_key].copy()
                                platform_info.update({
                                    'availability_type': availability_type,
                                    'region': region,
                                    'logo_path': f"https://image.tmdb.org/t/p/original{provider.get('logo_path')}" if provider.get('logo_path') else None,
                                    'display_priority': provider.get('display_priority', 999)
                                })
                                platforms.append(platform_info)
        
        return platforms
    
    @staticmethod
    def map_tmdb_provider_to_key(provider_name):
        """Map TMDB provider names to our platform keys"""
        provider_map = {
            'Netflix': 'netflix',
            'Amazon Prime Video': 'prime',
            'Disney Plus': 'hotstar',
            'Hotstar': 'hotstar',
            'Zee5': 'zee5',
            'SonyLIV': 'sonyliv',
            'Jio Cinema': 'jiocinema',
            'MX Player': 'mxplayer',
            'YouTube': 'youtube'
        }
        
        for key, value in provider_map.items():
            if key.lower() in provider_name.lower():
                return value
        return None
    
    @staticmethod
    def parse_watchmode_sources(sources_data):
        """Parse WatchMode API response"""
        platforms = []
        
        watchmode_source_map = {
            203: 'netflix',
            26: 'prime',
            372: 'hotstar',
            457: 'zee5',
            444: 'sonyliv',
            1899: 'mxplayer',
            1971: 'jiocinema',
            237: 'youtube'
        }
        
        for source in sources_data:
            source_id = source.get('source_id')
            if source_id in watchmode_source_map:
                platform_key = watchmode_source_map[source_id]
                if platform_key in KNOWN_OTT_PLATFORMS:
                    platform_info = KNOWN_OTT_PLATFORMS[platform_key].copy()
                    
                    if source.get('web_url'):
                        platform_info['direct_url'] = source['web_url']
                        platform_info['verified'] = True
                    
                    platform_info['price'] = source.get('price', 0)
                    platform_info['format'] = source.get('format', 'HD')
                    platform_info['type'] = source.get('type', 'subscription')
                    
                    platforms.append(platform_info)
        
        return platforms
    
    @staticmethod
    def parse_streaming_availability(streaming_data):
        """Parse Streaming Availability API response"""
        platforms = []
        
        service_map = {
            'netflix': 'netflix',
            'prime': 'prime',
            'hotstar': 'hotstar',
            'zee5': 'zee5',
            'sonyliv': 'sonyliv',
            'jiocinema': 'jiocinema',
            'mxplayer': 'mxplayer',
            'youtube': 'youtube'
        }
        
        # Handle different response formats
        if isinstance(streaming_data, list):
            # It's a list of results
            for result in streaming_data[:1]:  # Take first result
                if 'streamingInfo' in result:
                    platforms.extend(ContentService.parse_streaming_info(result['streamingInfo'], service_map))
        elif isinstance(streaming_data, dict) and 'streamingInfo' in streaming_data:
            # Direct result
            platforms.extend(ContentService.parse_streaming_info(streaming_data['streamingInfo'], service_map))
        
        return platforms
    
    @staticmethod
    def parse_streaming_info(streaming_info, service_map):
        """Parse streaming info object"""
        platforms = []
        
        for country, services in streaming_info.items():
            if isinstance(services, dict):
                for service_name, service_data in services.items():
                    platform_key = service_map.get(service_name.lower())
                    if platform_key and platform_key in KNOWN_OTT_PLATFORMS:
                        platform_info = KNOWN_OTT_PLATFORMS[platform_key].copy()
                        
                        # Handle different data structures
                        if isinstance(service_data, list) and service_data:
                            service_info = service_data[0]
                        else:
                            service_info = service_data
                        
                        if isinstance(service_info, dict):
                            if service_info.get('link'):
                                platform_info['direct_url'] = service_info['link']
                                platform_info['verified'] = True
                            
                            platform_info['quality'] = service_info.get('quality', 'HD')
                            platform_info['audio_languages'] = service_info.get('audios', [])
                            platform_info['subtitle_languages'] = service_info.get('subtitles', [])
                            platform_info['leaving_date'] = service_info.get('leaving')
                            platform_info['region'] = country.upper()
                        
                        platforms.append(platform_info)
        
        return platforms
    
    @staticmethod
    def get_intelligent_platform_suggestions(tmdb_data):
        """Intelligent platform suggestions based on content metadata"""
        platforms = []
        title = tmdb_data.get('title') or tmdb_data.get('name', '')
        original_language = tmdb_data.get('original_language', 'en')
        release_date = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
        popularity = tmdb_data.get('popularity', 0)
        vote_average = tmdb_data.get('vote_average', 0)
        
        # Parse release year
        release_year = None
        if release_date:
            try:
                release_year = int(release_date[:4])
            except:
                pass
        
        # Encode title for search URLs
        encoded_title = urllib.parse.quote(title)
        
        # High-popularity content is likely on major platforms
        if popularity > 50 or vote_average > 7:
            # Netflix (usually has popular content)
            platform = KNOWN_OTT_PLATFORMS['netflix'].copy()
            platform['search_url'] = platform['search_url'].format(query=encoded_title)
            platform['confidence'] = 'high' if popularity > 100 else 'medium'
            platform['availability_note'] = 'Popular content - likely available'
            platforms.append(platform)
            
            # Prime Video
            platform = KNOWN_OTT_PLATFORMS['prime'].copy()
            platform['search_url'] = platform['search_url'].format(query=encoded_title)
            platform['confidence'] = 'high' if popularity > 100 else 'medium'
            platform['availability_note'] = 'Check availability in your region'
            platforms.append(platform)
        
        # Indian content
        if original_language in ['hi', 'ta', 'te', 'kn', 'ml', 'bn', 'mr', 'gu', 'pa']:
            # Hotstar (strong Indian content)
            platform = KNOWN_OTT_PLATFORMS['hotstar'].copy()
            platform['search_url'] = platform['search_url'].format(query=encoded_title)
            platform['confidence'] = 'high'
            platform['availability_note'] = 'Indian content - likely available'
            platforms.append(platform)
            
            # Zee5
            platform = KNOWN_OTT_PLATFORMS['zee5'].copy()
            platform['search_url'] = platform['search_url'].format(query=encoded_title)
            platform['confidence'] = 'medium'
            platforms.append(platform)
            
            # JioCinema (free option)
            platform = KNOWN_OTT_PLATFORMS['jiocinema'].copy()
            platform['search_url'] = platform['search_url'].format(query=encoded_title)
            platform['confidence'] = 'medium'
            platform['availability_note'] = 'Free with ads'
            platforms.append(platform)
            
            # MX Player (free option)
            platform = KNOWN_OTT_PLATFORMS['mxplayer'].copy()
            platform['search_url'] = platform['search_url'].format(query=encoded_title)
            platform['confidence'] = 'medium'
            platform['availability_note'] = 'Free with ads'
            platforms.append(platform)
        
        # Recent releases
        if release_year and release_year >= datetime.now().year - 2:
            # Recent content often on premium platforms first
            if 'prime' not in [p.get('base_url') for p in platforms]:
                platform = KNOWN_OTT_PLATFORMS['prime'].copy()
                platform['search_url'] = platform['search_url'].format(query=encoded_title)
                platform['confidence'] = 'medium'
                platform['availability_note'] = 'Recent release'
                platforms.append(platform)
        
        # Older content might be free
        if release_year and release_year < datetime.now().year - 5:
            # YouTube (often has older content)
            platform = KNOWN_OTT_PLATFORMS['youtube'].copy()
            platform['search_url'] = platform['search_url'].format(query=encoded_title)
            platform['confidence'] = 'low'
            platform['availability_note'] = 'May be available for free or rent'
            platforms.append(platform)
        
        # Always include YouTube as a fallback for trailers/clips
        if not any(p.get('name') == 'YouTube' for p in platforms):
            platform = KNOWN_OTT_PLATFORMS['youtube'].copy()
            platform['search_url'] = platform['search_url'].format(query=encoded_title)
            platform['confidence'] = 'low'
            platform['availability_note'] = 'Check for trailers and clips'
            platforms.append(platform)
        
        return platforms
    
    @staticmethod
    def deduplicate_platforms(platforms):
        """Remove duplicate platforms and prioritize verified links"""
        seen = {}
        unique = []
        
        # Sort by: verified first, then free platforms, then by priority
        platforms.sort(key=lambda x: (
            not x.get('verified', False),
            not x.get('is_free', False),
            x.get('display_priority', 999)
        ))
        
        for platform in platforms:
            key = platform.get('name', '').lower()
            if key not in seen:
                seen[key] = True
                unique.append(platform)
            elif platform.get('verified') and not seen.get(f"{key}_verified"):
                # Replace with verified version
                unique = [p for p in unique if p.get('name', '').lower() != key]
                unique.append(platform)
                seen[f"{key}_verified"] = True
        
        return unique
    
    @staticmethod
    def save_anime_content(anime_data):
        """Save anime content from Jikan API"""
        try:
            existing = Content.query.filter_by(tmdb_id=anime_data['mal_id']).first()
            if existing:
                return existing
            
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            release_date = None
            if anime_data.get('aired') and anime_data['aired'].get('from'):
                try:
                    release_date = datetime.fromisoformat(anime_data['aired']['from'].replace('Z', '+00:00')).date()
                except:
                    release_date = None
            
            # Anime platforms (most anime is not on mainstream platforms in India)
            ott_platforms = ContentService.get_anime_platforms(anime_data)
            
            content = Content(
                tmdb_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                languages=json.dumps(['japanese']),
                release_date=release_date,
                runtime=anime_data.get('duration'),
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
    def get_anime_platforms(anime_data):
        """Get platforms for anime content"""
        platforms = []
        title = anime_data.get('title', '')
        encoded_title = urllib.parse.quote(title + ' anime')
        
        # YouTube (often has anime clips/episodes)
        platform = KNOWN_OTT_PLATFORMS['youtube'].copy()
        platform['search_url'] = platform['search_url'].format(query=encoded_title)
        platform['confidence'] = 'medium'
        platform['availability_note'] = 'Check for episodes and clips'
        platforms.append(platform)
        
        # Netflix (has some popular anime)
        platform = KNOWN_OTT_PLATFORMS['netflix'].copy()
        platform['search_url'] = platform['search_url'].format(query=encoded_title)
        platform['confidence'] = 'low'
        platform['availability_note'] = 'Limited anime selection'
        platforms.append(platform)
        
        # Prime Video (has some anime)
        platform = KNOWN_OTT_PLATFORMS['prime'].copy()
        platform['search_url'] = platform['search_url'].format(query=encoded_title)
        platform['confidence'] = 'low'
        platform['availability_note'] = 'Limited anime selection'
        platforms.append(platform)
        
        return platforms

# Recommendation Engine
class RecommendationEngine:
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all'):
        try:
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
            popular_movies = TMDBService.get_popular('movie', region=region)
            popular_tv = TMDBService.get_popular('tv', region=region)
            
            recommendations = []
            
            if popular_movies:
                for item in popular_movies.get('results', []):
                    if genre.lower() in [g.lower() for g in ContentService.map_genre_ids(item.get('genre_ids', []))]:
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content:
                            recommendations.append(content)
            
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
            search_queries = {
                'hindi': ['bollywood', 'hindi movie', 'hindi film'],
                'telugu': ['tollywood', 'telugu movie', 'telugu film'],
                'tamil': ['kollywood', 'tamil movie', 'tamil film'],
                'kannada': ['sandalwood', 'kannada movie', 'kannada film'],
                'malayalam': ['malayalam movie', 'malayalam film'],
                'bengali': ['bengali movie', 'bengali film'],
                'marathi': ['marathi movie', 'marathi film']
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
            interactions = AnonymousInteraction.query.filter_by(session_id=session_id).all()
            
            recommendations = []
            
            if interactions:
                viewed_content_ids = [interaction.content_id for interaction in interactions]
                viewed_contents = Content.query.filter(Content.id.in_(viewed_content_ids)).all()
                
                all_genres = []
                for content in viewed_contents:
                    if content.genres:
                        all_genres.extend(json.loads(content.genres))
                
                genre_counts = Counter(all_genres)
                top_genres = [genre for genre, _ in genre_counts.most_common(3)]
                
                for genre in top_genres:
                    genre_recs = RecommendationEngine.get_popular_by_genre(genre, limit=7)
                    recommendations.extend(genre_recs)
            
            # Add trending content
            trending_recs = RecommendationEngine.get_trending_recommendations(limit=10)
            recommendations.extend(trending_recs)
            
            # Remove duplicates
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
            
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            ott_text = ""
            free_platforms = [p for p in ott_platforms if p.get('is_free')]
            paid_platforms = [p for p in ott_platforms if not p.get('is_free')]
            
            if free_platforms:
                ott_text += "\n\n🆓 **FREE TO WATCH:**\n"
                for platform in free_platforms[:3]:
                    ott_text += f"▶️ {platform['name']}"
                    if platform.get('direct_url'):
                        ott_text += f" - [Watch Now]({platform['direct_url']})"
                    elif platform.get('search_url'):
                        ott_text += f" - [Search]({platform['search_url']})"
                    ott_text += "\n"
            
            if paid_platforms:
                ott_text += "\n💰 **PREMIUM PLATFORMS:**\n"
                for platform in paid_platforms[:3]:
                    ott_text += f"▶️ {platform['name']}"
                    if platform.get('direct_url'):
                        ott_text += f" - [Watch Now]({platform['direct_url']})"
                    elif platform.get('search_url'):
                        ott_text += f" - [Search]({platform['search_url']})"
                    ott_text += "\n"
            
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}{ott_text}

#AdminChoice #MovieRecommendation #CineScope"""
            
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
        
        if not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password']),
            preferred_languages=json.dumps(data.get('preferred_languages', [])),
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
        
        session_id = get_session_id()
        
        # Search TMDB
        tmdb_results = TMDBService.search_content(query, content_type, page=page)
        
        # Search anime if applicable
        anime_results = None
        if content_type in ['anime', 'multi']:
            anime_results = JikanService.search_anime(query, page=page)
        
        results = []
        
        if tmdb_results:
            for item in tmdb_results.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
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
                    
                    # Record interaction
                    interaction = AnonymousInteraction(
                        session_id=session_id,
                        content_id=content.id,
                        interaction_type='search',
                        ip_address=request.remote_addr
                    )
                    db.session.add(interaction)
                    
                    # Format result
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
                            'platforms_available': len(ott_platforms) > 0
                        }
                    })
        
        # Process anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                content = ContentService.save_anime_content(anime)
                if content:
                    results.append({
                        'id': content.id,
                        'title': content.title,
                        'content_type': 'anime',
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': content.poster_path,
                        'overview': content.overview,
                        'ott_summary': {
                            'has_free': True,
                            'free_count': 1,
                            'paid_count': 2,
                            'platforms_available': True
                        }
                    })
        
        db.session.commit()
        
        return jsonify({
            'results': results,
            'total_results': len(results),
            'current_page': page,
            'filters_applied': {
                'free_only': free_only,
                'platform_filter': platform_filter
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

# Content Details Route
@app.route('/api/content/<int:content_id>', methods=['GET'])
def get_content_details(content_id):
    try:
        content = Content.query.get_or_404(content_id)
        
        # Record view
        session_id = get_session_id()
        interaction = AnonymousInteraction(
            session_id=session_id,
            content_id=content.id,
            interaction_type='view',
            ip_address=request.remote_addr
        )
        db.session.add(interaction)
        
        # Get additional details if needed
        additional_details = None
        if content.tmdb_id and content.content_type != 'anime':
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
            
            # Update OTT platforms with fresh data
            if additional_details:
                updated_ott = ContentService.get_ott_availability(additional_details, content.imdb_id)
                if updated_ott:
                    content.ott_platforms = json.dumps(updated_ott)
                    content.updated_at = datetime.utcnow()
        
        # Get trailers
        trailers = []
        if YOUTUBE_API_KEY:
            youtube_results = YouTubeService.search_trailers(content.title)
            if youtube_results:
                for video in youtube_results.get('items', [])[:3]:
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
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else [],
            
            # OTT Information
            'streaming_info': {
                'available': len(ott_platforms) > 0,
                'free_options': len(free_platforms),
                'paid_options': len(paid_platforms),
                'last_updated': content.updated_at.isoformat() if content.updated_at else None
            },
            'ott_platforms': {
                'free_platforms': free_platforms,
                'paid_platforms': paid_platforms,
                'all_platforms': ott_platforms
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# OTT Availability Endpoint
@app.route('/api/content/<int:content_id>/ott', methods=['GET'])
def get_ott_availability_endpoint(content_id):
    """Get fresh OTT availability for content"""
    try:
        content = Content.query.get_or_404(content_id)
        
        # Force refresh OTT data
        if content.tmdb_id:
            tmdb_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
            if tmdb_details:
                fresh_ott = ContentService.get_ott_availability(tmdb_details, content.imdb_id)
                
                # Update content
                content.ott_platforms = json.dumps(fresh_ott)
                content.updated_at = datetime.utcnow()
                db.session.commit()
                
                # Group platforms
                free_platforms = [p for p in fresh_ott if p.get('is_free')]
                paid_platforms = [p for p in fresh_ott if not p.get('is_free')]
                
                return jsonify({
                    'content_title': content.title,
                    'last_updated': content.updated_at.isoformat(),
                    'streaming_available': len(fresh_ott) > 0,
                    'free_platforms': free_platforms,
                    'paid_platforms': paid_platforms,
                    'total_platforms': len(fresh_ott),
                    'data_sources': {
                        'tmdb': bool(tmdb_details.get('watch/providers')),
                        'watchmode': any(p.get('verified') for p in fresh_ott if 'watchmode' in str(p).lower()),
                        'streaming_availability': any(p.get('verified') for p in fresh_ott if 'streaming' in str(p).lower()),
                        'intelligent_suggestions': any(p.get('confidence') for p in fresh_ott)
                    }
                }), 200
        
        # Return existing data if no TMDB ID
        ott_platforms = json.loads(content.ott_platforms or '[]')
        free_platforms = [p for p in ott_platforms if p.get('is_free')]
        paid_platforms = [p for p in ott_platforms if not p.get('is_free')]
        
        return jsonify({
            'content_title': content.title,
            'last_updated': content.updated_at.isoformat() if content.updated_at else None,
            'streaming_available': len(ott_platforms) > 0,
            'free_platforms': free_platforms,
            'paid_platforms': paid_platforms,
            'total_platforms': len(ott_platforms)
        }), 200
        
    except Exception as e:
        logger.error(f"OTT availability error: {e}")
        return jsonify({'error': 'Failed to get OTT availability'}), 500

# Debug endpoint for OTT services
@app.route('/api/debug/ott/<string:title>', methods=['GET'])
def debug_ott_search(title):
    """Debug endpoint to test OTT API responses"""
    try:
        results = {
            'query': title,
            'services_checked': [],
            'raw_responses': {}
        }
        
        # Test WatchMode
        if WATCHMODE_API_KEY:
            watchmode_data = WatchModeService.search_title(title)
            results['services_checked'].append('watchmode')
            results['raw_responses']['watchmode'] = watchmode_data
        
        # Test Streaming Availability
        if RAPIDAPI_KEY:
            streaming_data = StreamingAvailabilityService.search_by_title(title)
            results['services_checked'].append('streaming_availability')
            results['raw_responses']['streaming_availability'] = streaming_data
        
        # Test TMDB
        tmdb_data = TMDBService.search_content(title)
        if tmdb_data and tmdb_data.get('results'):
            first_result = tmdb_data['results'][0]
            content_type = 'movie' if 'title' in first_result else 'tv'
            tmdb_id = first_result['id']
            
            # Get watch providers
            providers_data = TMDBService.get_watch_providers(tmdb_id, content_type)
            results['services_checked'].append('tmdb_providers')
            results['raw_responses']['tmdb_providers'] = providers_data
        
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Debug OTT error: {e}")
        return jsonify({'error': str(e)}), 500

# Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = []
        for content in recommendations:
            ott_platforms = json.loads(content.ott_platforms or '[]')
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
                    'available': len(ott_platforms) > 0,
                    'free_count': len(free_platforms),
                    'paid_count': len(paid_platforms)
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
            ott_platforms = json.loads(content.ott_platforms or '[]')
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
                    'available': len(ott_platforms) > 0,
                    'free_count': len(free_platforms),
                    'paid_count': len(paid_platforms)
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
            ott_platforms = json.loads(content.ott_platforms or '[]')
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
                    'available': len(ott_platforms) > 0,
                    'free_count': len(free_platforms),
                    'paid_count': len(paid_platforms)
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
            ott_platforms = json.loads(content.ott_platforms or '[]')
            
            result.append({
                'id': content.id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': {
                    'available': len(ott_platforms) > 0,
                    'platforms': [p['name'] for p in ott_platforms]
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
        
        result = []
        for content in recommendations:
            ott_platforms = json.loads(content.ott_platforms or '[]')
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
                    'available': len(ott_platforms) > 0,
                    'free_count': len(free_platforms),
                    'paid_count': len(paid_platforms)
                }
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
            ott_platforms = json.loads(content.ott_platforms or '[]')
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_available': len(ott_platforms) > 0
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
            ott_platforms = json.loads(content.ott_platforms or '[]')
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_available': len(ott_platforms) > 0
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
        source = request.args.get('source', 'tmdb')
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
            return jsonify({'error': 'Content not found'}), 404
        
        admin_rec = AdminRecommendation(
            content_id=content.id,
            admin_id=current_user.id,
            recommendation_type=data['recommendation_type'],
            description=data['description']
        )
        
        db.session.add(admin_rec)
        db.session.commit()
        
        telegram_success = TelegramService.send_admin_recommendation(content, current_user.username, data['description'])
        
        return jsonify({
            'message': 'Admin recommendation created successfully',
            'telegram_sent': telegram_success
        }), 201
        
    except Exception as e:
        logger.error(f"Admin recommendation error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create recommendation'}), 500

@app.route('/api/admin/analytics', methods=['GET'])
@require_admin
def get_analytics(current_user):
    try:
        total_users = User.query.count()
        total_content = Content.query.count()
        total_interactions = UserInteraction.query.count()
        active_users_last_week = User.query.filter(
            User.last_active >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        popular_content = db.session.query(
            Content.id, Content.title, func.count(UserInteraction.id).label('interaction_count')
        ).join(UserInteraction).group_by(Content.id, Content.title)\
         .order_by(desc('interaction_count')).limit(10).all()
        
        return jsonify({
            'total_users': total_users,
            'total_content': total_content,
            'total_interactions': total_interactions,
            'active_users_last_week': active_users_last_week,
            'popular_content': [
                {'title': item.title, 'interactions': item.interaction_count}
                for item in popular_content
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
                ott_platforms = json.loads(content.ott_platforms or '[]')
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
                        'available': len(ott_platforms) > 0,
                        'free_count': len(free_platforms),
                        'paid_count': len(paid_platforms)
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
        'version': '2.0.0',
        'features': {
            'ott_integration': True,
            'intelligent_suggestions': True,
            'multiple_api_sources': True,
            'telegram_bot': bot is not None,
            'watchmode_api': bool(WATCHMODE_API_KEY),
            'streaming_availability_api': bool(RAPIDAPI_KEY),
            'tmdb_providers': True
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