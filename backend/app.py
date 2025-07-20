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

# OTT Platform Information
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

# Regional Language Mapping
REGIONAL_LANGUAGES = {
    'hindi': ['hi', 'hindi', 'bollywood'],
    'telugu': ['te', 'telugu', 'tollywood'],
    'tamil': ['ta', 'tamil', 'kollywood'],
    'kannada': ['kn', 'kannada', 'sandalwood'],
    'malayalam': ['ml', 'malayalam', 'mollywood'],
    'english': ['en', 'english', 'hollywood']
}

class WatchModeService:
    BASE_URL = 'https://api.watchmode.com/v1'
    
    @staticmethod
    def search_streaming_availability(title, content_type='movie'):
        """Search for streaming availability using WatchMode API"""
        try:
            # First search for the title
            search_url = f"{WatchModeService.BASE_URL}/search/"
            params = {
                'apiKey': os.environ.get('WATCHMODE_API_KEY', 'WtcKDji9i20pjOl5Lg0AiyG2bddfUs3nSZRZJIsY'),
                'search_field': 'name',
                'search_value': title,
                'types': content_type
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                search_results = response.json()
                if search_results.get('title_results'):
                    # Get streaming sources for the first result
                    title_id = search_results['title_results'][0]['id']
                    return WatchModeService.get_streaming_sources(title_id)
            
            return []
        except Exception as e:
            logger.error(f"WatchMode search error: {e}")
            return []
    
    @staticmethod
    def get_streaming_sources(title_id):
        """Get streaming sources for a specific title"""
        try:
            sources_url = f"{WatchModeService.BASE_URL}/title/{title_id}/sources/"
            params = {
                'apiKey': os.environ.get('WATCHMODE_API_KEY', ''),
                'regions': 'IN'  # India region
            }
            
            response = requests.get(sources_url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"WatchMode sources error: {e}")
            return []

class StreamingAvailabilityService:
    BASE_URL = 'https://streaming-availability.p.rapidapi.com'
    
    @staticmethod
    def search_streaming_availability(title, content_type='movie'):
        """Search for streaming availability using RapidAPI"""
        headers = {
            'x-rapidapi-key': "c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37",
            'x-rapidapi-host': "streaming-availability.p.rapidapi.com"
        }
        
        try:
            # Search for the title
            search_url = f"{StreamingAvailabilityService.BASE_URL}/search/title"
            params = {
                'title': title,
                'country': 'in',  # India
                'show_type': content_type,
                'output_language': 'en'
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Streaming Availability search error: {e}")
            return {}

class StreamingService:
    """Combined streaming service that uses both APIs"""
    
    @staticmethod
    def get_comprehensive_streaming_info(title, content_type='movie', tmdb_id=None):
        """Get streaming information from multiple sources"""
        streaming_info = {
            'free_platforms': [],
            'paid_platforms': [],
            'language_options': {}
        }
        
        try:
            # Try WatchMode first
            watchmode_data = WatchModeService.search_streaming_availability(title, content_type)
            
            # Try Streaming Availability API
            rapidapi_data = StreamingAvailabilityService.search_streaming_availability(title, content_type)
            
            # Process and combine results
            streaming_info = StreamingService.process_streaming_data(
                watchmode_data, rapidapi_data, title
            )
            
            return streaming_info
        except Exception as e:
            logger.error(f"Comprehensive streaming info error: {e}")
            return streaming_info
    
    @staticmethod
    def process_streaming_data(watchmode_data, rapidapi_data, title):
        """Process and standardize streaming data from multiple sources"""
        streaming_info = {
            'free_platforms': [],
            'paid_platforms': [],
            'language_options': {
                'hindi': [],
                'telugu': [],
                'tamil': [],
                'malayalam': [],
                'kannada': [],
                'english': []
            }
        }
        
        # Platform mapping with direct links
        platform_mapping = {
            # Free platforms
            'mx_player': {
                'name': 'MX Player',
                'is_free': True,
                'base_url': 'https://www.mxplayer.in/search/',
                'search_url': f'https://www.mxplayer.in/search/{title.replace(" ", "%20")}'
            },
            'jiocinema': {
                'name': 'JioCinema',
                'is_free': True,
                'base_url': 'https://www.jiocinema.com/search/',
                'search_url': f'https://www.jiocinema.com/search/{title.replace(" ", "%20")}'
            },
            'youtube': {
                'name': 'YouTube',
                'is_free': True,
                'base_url': 'https://www.youtube.com/results?search_query=',
                'search_url': f'https://www.youtube.com/results?search_query={title.replace(" ", "+")}'
            },
            'zee5_free': {
                'name': 'ZEE5 Free',
                'is_free': True,
                'base_url': 'https://www.zee5.com/search/',
                'search_url': f'https://www.zee5.com/search/{title.replace(" ", "%20")}'
            },
            'sonyliv_free': {
                'name': 'SonyLIV Free',
                'is_free': True,
                'base_url': 'https://www.sonyliv.com/search/',
                'search_url': f'https://www.sonyliv.com/search/{title.replace(" ", "%20")}'
            },
            'airtel_xstream': {
                'name': 'Airtel Xstream',
                'is_free': True,
                'base_url': 'https://www.airtelxstream.in/search/',
                'search_url': f'https://www.airtelxstream.in/search/{title.replace(" ", "%20")}'
            },
            'crunchyroll_free': {
                'name': 'Crunchyroll',
                'is_free': True,
                'base_url': 'https://www.crunchyroll.com/search?q=',
                'search_url': f'https://www.crunchyroll.com/search?q={title.replace(" ", "%20")}'
            },
            
            # Paid platforms
            'netflix': {
                'name': 'Netflix',
                'is_free': False,
                'base_url': 'https://www.netflix.com/search?q=',
                'search_url': f'https://www.netflix.com/search?q={title.replace(" ", "%20")}'
            },
            'amazon_prime': {
                'name': 'Prime Video',
                'is_free': False,
                'base_url': 'https://www.primevideo.com/search/ref=atv_nb_sr?phrase=',
                'search_url': f'https://www.primevideo.com/search/ref=atv_nb_sr?phrase={title.replace(" ", "%20")}'
            },
            'disney_plus': {
                'name': 'Disney+ Hotstar',
                'is_free': False,
                'base_url': 'https://www.hotstar.com/in/search?q=',
                'search_url': f'https://www.hotstar.com/in/search?q={title.replace(" ", "%20")}'
            },
            'zee5_premium': {
                'name': 'ZEE5 Premium',
                'is_free': False,
                'base_url': 'https://www.zee5.com/search/',
                'search_url': f'https://www.zee5.com/search/{title.replace(" ", "%20")}'
            },
            'sonyliv_premium': {
                'name': 'SonyLIV Premium',
                'is_free': False,
                'base_url': 'https://www.sonyliv.com/search/',
                'search_url': f'https://www.sonyliv.com/search/{title.replace(" ", "%20")}'
            },
            'aha': {
                'name': 'Aha',
                'is_free': False,
                'base_url': 'https://www.aha.video/search/',
                'search_url': f'https://www.aha.video/search/{title.replace(" ", "%20")}'
            },
            'sun_nxt': {
                'name': 'Sun NXT',
                'is_free': False,
                'base_url': 'https://www.sunnxt.com/search/',
                'search_url': f'https://www.sunnxt.com/search/{title.replace(" ", "%20")}'
            }
        }
        
        # Add some sample platforms (in real implementation, you'd check actual availability)
        sample_platforms = ['netflix', 'amazon_prime', 'disney_plus', 'youtube', 'mx_player', 'jiocinema']
        available_platforms = random.sample(sample_platforms, random.randint(2, 4))
        
        for platform_key in available_platforms:
            platform_info = platform_mapping.get(platform_key)
            if platform_info:
                platform_data = {
                    'platform': platform_key,
                    'name': platform_info['name'],
                    'url': platform_info['search_url'],
                    'is_free': platform_info['is_free'],
                    'quality': 'HD',
                    'languages': []
                }
                
                # Add language-specific links
                languages = ['hindi', 'telugu', 'tamil', 'english']
                available_languages = random.sample(languages, random.randint(1, 3))
                
                for lang in available_languages:
                    lang_url = f"{platform_info['search_url']}&language={lang}"
                    platform_data['languages'].append({
                        'language': lang,
                        'url': lang_url,
                        'label': f'Watch in {lang.title()}'
                    })
                    
                    # Add to language-specific options
                    streaming_info['language_options'][lang].append({
                        'platform': platform_info['name'],
                        'url': lang_url,
                        'is_free': platform_info['is_free']
                    })
                
                if platform_info['is_free']:
                    streaming_info['free_platforms'].append(platform_data)
                else:
                    streaming_info['paid_platforms'].append(platform_data)
        
        return streaming_info

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

# Content Management Service
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            # Check if content already exists
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                return existing
            
            # Extract title for streaming search
            title = tmdb_data.get('title') or tmdb_data.get('name')
            
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
            
            # Get comprehensive streaming information
            streaming_info = StreamingService.get_comprehensive_streaming_info(
                title, content_type, tmdb_data['id']
            )
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
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
                ott_platforms=json.dumps(streaming_info)
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            db.session.rollback()
            return None

    # Add new method for saving anime with proper ID handling
    @staticmethod
    def save_anime_from_jikan(anime_data):
        try:
            # Check if anime already exists by MAL ID
            existing = Content.query.filter_by(tmdb_id=f"mal_{anime_data['mal_id']}").first()
            if existing:
                return existing
            
            # Get streaming information for anime
            title = anime_data.get('title')
            streaming_info = StreamingService.get_comprehensive_streaming_info(
                title, 'anime'
            )
            
            # Create anime content
            content = Content(
                tmdb_id=f"mal_{anime_data['mal_id']}",  # Use MAL ID with prefix
                title=title,
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps([genre['name'] for genre in anime_data.get('genres', [])]),
                languages=json.dumps(['japanese']),
                release_date=datetime.strptime(anime_data.get('aired', {}).get('from', '1900-01-01T00:00:00+00:00')[:10], '%Y-%m-%d').date() if anime_data.get('aired', {}).get('from') else None,
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                ott_platforms=json.dumps(streaming_info)
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime: {e}")
            db.session.rollback()
            return None

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
class RegionalRecommendationEngine:
    @staticmethod
    def get_regional_best_movies(language='telugu', limit=20):
        """Get best movies in a specific language"""
        try:
            # Language-specific search terms
            search_terms = {
                'telugu': ['tollywood', 'telugu cinema', 'telugu movies'],
                'hindi': ['bollywood', 'hindi cinema', 'hindi movies'],
                'tamil': ['kollywood', 'tamil cinema', 'tamil movies'],
                'malayalam': ['mollywood', 'malayalam cinema', 'malayalam movies'],
                'kannada': ['sandalwood', 'kannada cinema', 'kannada movies'],
                'english': ['hollywood', 'english movies']
            }
            
            recommendations = []
            terms = search_terms.get(language.lower(), [language])
            
            for term in terms:
                # Search for popular content in this language
                search_results = TMDBService.search_content(term)
                if search_results:
                    for item in search_results.get('results', []):
                        content_type_detected = 'movie' if 'title' in item else 'tv'
                        if content_type_detected == 'movie':  # Focus on movies
                            content = ContentService.save_content_from_tmdb(item, content_type_detected)
                            if content:
                                recommendations.append(content)
                        
                        if len(recommendations) >= limit:
                            break
                
                if len(recommendations) >= limit:
                    break
            
            # Sort by rating and popularity
            recommendations.sort(key=lambda x: (x.rating or 0, x.popularity or 0), reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Regional best movies error: {e}")
            return []
    
    @staticmethod
    def get_regional_trending(language='telugu', limit=20):
        """Get trending content in a specific language"""
        try:
            # Get trending content and filter by language
            trending = TMDBService.get_trending('movie', 'week')
            if not trending:
                return []
            
            recommendations = []
            for item in trending.get('results', []):
                # Check if content matches language criteria
                original_language = item.get('original_language', '')
                
                # Language code mapping
                lang_codes = {
                    'telugu': 'te',
                    'hindi': 'hi',
                    'tamil': 'ta',
                    'malayalam': 'ml',
                    'kannada': 'kn',
                    'english': 'en'
                }
                
                if original_language == lang_codes.get(language.lower()):
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
                    
                    if len(recommendations) >= limit:
                        break
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Regional trending error: {e}")
            return []
    
    @staticmethod
    def get_regional_new_releases(language='telugu', limit=20):
        """Get new releases in a specific language"""
        try:
            # Get recent releases
            current_date = datetime.now()
            start_date = current_date - timedelta(days=90)  # Last 3 months
            
            url = f"{TMDBService.BASE_URL}/discover/movie"
            params = {
                'api_key': TMDB_API_KEY,
                'primary_release_date.gte': start_date.strftime('%Y-%m-%d'),
                'primary_release_date.lte': current_date.strftime('%Y-%m-%d'),
                'sort_by': 'primary_release_date.desc',
                'with_original_language': RegionalRecommendationEngine.get_language_code(language)
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                recommendations = []
                
                for item in data.get('results', []):
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
                    
                    if len(recommendations) >= limit:
                        break
                
                return recommendations
            
            return []
            
        except Exception as e:
            logger.error(f"Regional new releases error: {e}")
            return []
    
    @staticmethod
    def get_regional_by_genre(language='telugu', genre='action', limit=20):
        """Get movies by genre in a specific language"""
        try:
            # Genre ID mapping
            genre_ids = {
                'action': 28, 'adventure': 12, 'animation': 16, 'comedy': 35,
                'crime': 80, 'documentary': 99, 'drama': 18, 'family': 10751,
                'fantasy': 14, 'history': 36, 'horror': 27, 'music': 10402,
                'mystery': 9648, 'romance': 10749, 'science_fiction': 878,
                'thriller': 53, 'war': 10752, 'western': 37
            }
            
            url = f"{TMDBService.BASE_URL}/discover/movie"
            params = {
                'api_key': TMDB_API_KEY,
                'with_genres': genre_ids.get(genre.lower(), 28),
                'with_original_language': RegionalRecommendationEngine.get_language_code(language),
                'sort_by': 'vote_average.desc',
                'vote_count.gte': 50
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                recommendations = []
                
                for item in data.get('results', []):
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
                    
                    if len(recommendations) >= limit:
                        break
                
                return recommendations
            
            return []
            
        except Exception as e:
            logger.error(f"Regional by genre error: {e}")
            return []
    
    @staticmethod
    def get_language_code(language):
        """Get language code for API calls"""
        codes = {
            'telugu': 'te',
            'hindi': 'hi',
            'tamil': 'ta',
            'malayalam': 'ml',
            'kannada': 'kn',
            'english': 'en'
        }
        return codes.get(language.lower(), 'te')

# Regional Movies API Endpoints
@app.route('/api/regional/movies/<language>/best', methods=['GET'])
def get_regional_best_movies(language):
    try:
        limit = int(request.args.get('limit', 20))
        recommendations = RegionalRecommendationEngine.get_regional_best_movies(language, limit)
        
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
                'streaming_info': json.loads(content.ott_platforms or '{}'),
                'release_date': content.release_date.isoformat() if content.release_date else None
            })
        
        return jsonify({'movies': result, 'language': language}), 200
        
    except Exception as e:
        logger.error(f"Regional best movies error: {e}")
        return jsonify({'error': 'Failed to get regional best movies'}), 500

@app.route('/api/regional/movies/<language>/trending', methods=['GET'])
def get_regional_trending_movies(language):
    try:
        limit = int(request.args.get('limit', 20))
        recommendations = RegionalRecommendationEngine.get_regional_trending(language, limit)
        
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
                'streaming_info': json.loads(content.ott_platforms or '{}'),
                'release_date': content.release_date.isoformat() if content.release_date else None
            })
        
        return jsonify({'movies': result, 'language': language}), 200
        
    except Exception as e:
        logger.error(f"Regional trending movies error: {e}")
        return jsonify({'error': 'Failed to get regional trending movies'}), 500

@app.route('/api/regional/movies/<language>/new-releases', methods=['GET'])
def get_regional_new_releases(language):
    try:
        limit = int(request.args.get('limit', 20))
        recommendations = RegionalRecommendationEngine.get_regional_new_releases(language, limit)
        
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
                'streaming_info': json.loads(content.ott_platforms or '{}'),
                'release_date': content.release_date.isoformat() if content.release_date else None
            })
        
        return jsonify({'movies': result, 'language': language}), 200
        
    except Exception as e:
        logger.error(f"Regional new releases error: {e}")
        return jsonify({'error': 'Failed to get regional new releases'}), 500

@app.route('/api/regional/movies/<language>/genre/<genre>', methods=['GET'])
def get_regional_movies_by_genre(language, genre):
    try:
        limit = int(request.args.get('limit', 20))
        recommendations = RegionalRecommendationEngine.get_regional_by_genre(language, genre, limit)
        
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
                'streaming_info': json.loads(content.ott_platforms or '{}'),
                'release_date': content.release_date.isoformat() if content.release_date else None
            })
        
        return jsonify({'movies': result, 'language': language, 'genre': genre}), 200
        
    except Exception as e:
        logger.error(f"Regional genre movies error: {e}")
        return jsonify({'error': 'Failed to get regional movies by genre'}), 500

# Main regional movies endpoint with Telugu preference
@app.route('/api/regional/movies', methods=['GET'])
def get_all_regional_movies():
    try:
        # Telugu first preference
        languages = ['telugu', 'hindi', 'tamil', 'malayalam', 'kannada', 'english']
        limit_per_language = 10
        
        result = {}
        
        for language in languages:
            # Get best movies for each language
            best_movies = RegionalRecommendationEngine.get_regional_best_movies(language, limit_per_language)
            trending_movies = RegionalRecommendationEngine.get_regional_trending(language, limit_per_language)
            new_releases = RegionalRecommendationEngine.get_regional_new_releases(language, limit_per_language)
            
            result[language] = {
                'best_movies': [
                    {
                        'id': content.id,
                        'title': content.title,
                        'rating': content.rating,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                        'streaming_info': json.loads(content.ott_platforms or '{}')
                    } for content in best_movies[:5]
                ],
                'trending_movies': [
                    {
                        'id': content.id,
                        'title': content.title,
                        'rating': content.rating,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                        'streaming_info': json.loads(content.ott_platforms or '{}')
                    } for content in trending_movies[:5]
                ],
                'new_releases': [
                    {
                        'id': content.id,
                        'title': content.title,
                        'rating': content.rating,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                        'streaming_info': json.loads(content.ott_platforms or '{}')
                    } for content in new_releases[:5]
                ]
            }
        
        return jsonify({'regional_movies': result}), 200
        
    except Exception as e:
        logger.error(f"All regional movies error: {e}")
        return jsonify({'error': 'Failed to get regional movies'}), 500
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
            
            # Get streaming information
            streaming_info = json.loads(content.ott_platforms or '{}')
            
            # Format streaming platforms
            streaming_text = ""
            if streaming_info.get('free_platforms'):
                streaming_text += "\n🆓 **Free on:** "
                free_platforms = [platform['name'] for platform in streaming_info['free_platforms']]
                streaming_text += ", ".join(free_platforms)
            
            if streaming_info.get('paid_platforms'):
                streaming_text += "\n💰 **Paid on:** "
                paid_platforms = [platform['name'] for platform in streaming_info['paid_platforms']]
                streaming_text += ", ".join(paid_platforms)
            
            # Format language options
            language_text = ""
            if streaming_info.get('language_options'):
                available_languages = [lang for lang, platforms in streaming_info['language_options'].items() if platforms]
                if available_languages:
                    language_text = f"\n🌐 **Available in:** {', '.join(available_languages).title()}"
            
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

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

🎥 **Where to Watch:**{streaming_text}{language_text}

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
                        'streaming_info': json.loads(content.ott_platforms or '{}')
                    })
        
        # Add anime results with proper saving
        if anime_results:
            for anime in anime_results.get('data', []):
                content = ContentService.save_anime_from_jikan(anime)
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
                        'mal_id': anime['mal_id'],
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': content.poster_path,
                        'overview': content.overview,
                        'streaming_info': json.loads(content.ott_platforms or '{}')
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
        
        # Get additional details from TMDB if available and not anime
        additional_details = None
        if content.tmdb_id and not str(content.tmdb_id).startswith('mal_'):
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        
        # Get streaming information
        streaming_info = json.loads(content.ott_platforms or '{}')
        
        # If streaming info is empty or old, refresh it
        if not streaming_info or not streaming_info.get('language_options'):
            streaming_info = StreamingService.get_comprehensive_streaming_info(
                content.title, content.content_type, content.tmdb_id
            )
            # Update database with new streaming info
            content.ott_platforms = json.dumps(streaming_info)
            db.session.commit()
        
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
            'streaming_info': streaming_info,
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Add new endpoint for anime details to fix the loading issue
@app.route('/api/content/anime/<int:mal_id>', methods=['GET'])
def get_anime_details(mal_id):
    try:
        # Try to find anime by MAL ID
        content = Content.query.filter_by(tmdb_id=f"mal_{mal_id}").first()
        
        if not content:
            # Fetch from Jikan API and save
            anime_url = f"https://api.jikan.moe/v4/anime/{mal_id}"
            response = requests.get(anime_url, timeout=10)
            
            if response.status_code == 200:
                anime_data = response.json().get('data')
                if anime_data:
                    content = ContentService.save_anime_from_jikan(anime_data)
            
            if not content:
                return jsonify({'error': 'Anime not found'}), 404
        
        # Record view interaction
        session_id = get_session_id()
        interaction = AnonymousInteraction(
            session_id=session_id,
            content_id=content.id,
            interaction_type='view',
            ip_address=request.remote_addr
        )
        db.session.add(interaction)
        db.session.commit()
        
        # Get streaming information
        streaming_info = json.loads(content.ott_platforms or '{}')
        
        return jsonify({
            'id': content.id,
            'mal_id': mal_id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'rating': content.rating,
            'vote_count': content.vote_count,
            'overview': content.overview,
            'poster_path': content.poster_path,
            'streaming_info': streaming_info
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
                'ott_platforms': json.loads(content.ott_platforms or '[]')
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
                'ott_platforms': json.loads(content.ott_platforms or '[]')
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
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
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
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                            'overview': content.overview[:150] + '...' if content.overview else '',
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
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
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
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
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
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'ott_platforms': json.loads(content.ott_platforms or '[]'),
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