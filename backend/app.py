from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_caching import Cache
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta
import requests
import os
import json
import logging
from functools import wraps
from collections import defaultdict, Counter
import random
from sqlalchemy import func, desc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import atexit

# Import separated modules
from trending import init_advanced_trending_service, get_trending_service
from admin import admin_bp, init_admin
from users import users_bp, init_users, require_auth
import auth

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Database Configuration
DATABASE_URL = 'postgresql://movies_rec_panf_user:BO5X3d2QihK7GG9hxgtBiCtni8NTbbIi@dpg-d2q7gamr433s73e0hcm0-a/movies_rec_panf'

if os.environ.get('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Redis Configuration
REDIS_URL = os.environ.get('REDIS_URL', 'redis://red-d2qlbuje5dus73c71qog:xp7inVzgblGCbo9I4taSGLdKUg0xY91I@red-d2qlbuje5dus73c71qog:6379')

if REDIS_URL and REDIS_URL.startswith(('redis://', 'rediss://')):
    app.config['CACHE_TYPE'] = 'redis'
    app.config['CACHE_REDIS_URL'] = REDIS_URL
    app.config['CACHE_DEFAULT_TIMEOUT'] = 3600
else:
    app.config['CACHE_TYPE'] = 'simple'
    app.config['CACHE_DEFAULT_TIMEOUT'] = 1800

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)
cache = Cache(app)

# API Keys and Configuration
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '52260795')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '-1002850793757')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-xmf5.onrender.com')

# Store configuration in app config for modules
app.config['TELEGRAM_BOT_TOKEN'] = TELEGRAM_BOT_TOKEN
app.config['TELEGRAM_CHANNEL_ID'] = TELEGRAM_CHANNEL_ID
app.config['ML_SERVICE_URL'] = ML_SERVICE_URL
app.config['TMDB_API_KEY'] = TMDB_API_KEY

# Initialize services (will be done after database is ready)
trending_service = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_http_session():
    """Create HTTP session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 504)
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

http_session = create_http_session()
executor = ThreadPoolExecutor(max_workers=5)

# Language Priority Configuration for New Releases
LANGUAGE_PRIORITY = {
    'telugu': {
        'code': 'te',
        'priority': 1,
        'min_items': 5,
        'regions': ['IN'],
        'keywords': ['telugu', 'tollywood', 'tfi', 'telugu cinema']
    },
    'hindi': {
        'code': 'hi',
        'priority': 2,
        'min_items': 5,
        'regions': ['IN'],
        'keywords': ['hindi', 'bollywood', 'hindi cinema', 'hindi film']
    },
    'english': {
        'code': 'en',
        'priority': 3,
        'min_items': 3,
        'regions': ['US', 'GB', 'IN'],
        'keywords': ['hollywood', 'english', 'blockbuster']
    },
    'tamil': {
        'code': 'ta',
        'priority': 4,
        'min_items': 4,
        'regions': ['IN'],
        'keywords': ['tamil', 'kollywood', 'tamil cinema', 'tamil film']
    },
    'malayalam': {
        'code': 'ml',
        'priority': 5,
        'min_items': 3,
        'regions': ['IN'],
        'keywords': ['malayalam', 'mollywood', 'malayalam cinema']
    }
}

# Language and Genre configurations
REGIONAL_LANGUAGES = {
    'hindi': ['hi', 'hindi', 'bollywood'],
    'telugu': ['te', 'telugu', 'tollywood'],
    'tamil': ['ta', 'tamil', 'kollywood'],
    'kannada': ['kn', 'kannada', 'sandalwood'],
    'malayalam': ['ml', 'malayalam', 'mollywood'],
    'english': ['en', 'english', 'hollywood']
}

ANIME_GENRES = {
    'shonen': ['Action', 'Adventure', 'Martial Arts', 'School', 'Shounen'],
    'shojo': ['Romance', 'Drama', 'School', 'Slice of Life', 'Shoujo'],
    'seinen': ['Action', 'Drama', 'Thriller', 'Psychological', 'Seinen'],
    'josei': ['Romance', 'Drama', 'Slice of Life', 'Josei'],
    'kodomomuke': ['Kids', 'Family', 'Adventure', 'Comedy']
}

# Cache key helpers
def make_cache_key(*args, **kwargs):
    path = request.path
    args_str = str(hash(frozenset(request.args.items())))
    return f"{path}:{args_str}"

def content_cache_key(content_id):
    return f"content:{content_id}"

def search_cache_key(query, content_type, page):
    return f"search:{query}:{content_type}:{page}"

def recommendations_cache_key(rec_type, **kwargs):
    params = ':'.join([f"{k}={v}" for k, v in sorted(kwargs.items())])
    return f"recommendations:{rec_type}:{params}"

# ================== Database Models ==================

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
    mal_id = db.Column(db.Integer)
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    content_type = db.Column(db.String(20), nullable=False)
    genres = db.Column(db.Text)
    anime_genres = db.Column(db.Text)
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
    youtube_trailer_id = db.Column(db.String(255))
    is_trending = db.Column(db.Boolean, default=False)
    is_new_release = db.Column(db.Boolean, default=False)
    is_critics_choice = db.Column(db.Boolean, default=False)
    critics_score = db.Column(db.Float)
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

# ================== Initialize Modules ==================

# Initialize auth module
from auth import init_auth, auth_bp
app.register_blueprint(auth_bp)
init_auth(app, db, User)

# Initialize admin module
init_admin(app, db, cache)
app.register_blueprint(admin_bp)

# Initialize users module
init_users(app, db, cache)
app.register_blueprint(users_bp)

def init_services():
    """Initialize all services after app context is ready"""
    global trending_service
    trending_service = init_advanced_trending_service(db, cache, TMDB_API_KEY)
    logger.info("All services initialized")
    return trending_service

def cleanup_services():
    """Cleanup services on shutdown"""
    global trending_service
    if trending_service:
        try:
            trending_service.stop_background_updates()
            logger.info("Stopped trending service background updates")
        except:
            pass

# Register cleanup
atexit.register(cleanup_services)

# ================== Service Classes ==================

class MLServiceClient:
    @staticmethod
    def call_ml_service(endpoint, params=None, timeout=15, use_cache=True):
        try:
            if not ML_SERVICE_URL:
                return None
            
            cache_key = f"ml:{endpoint}:{json.dumps(params, sort_keys=True)}"
            
            if use_cache:
                cached_result = cache.get(cache_key)
                if cached_result:
                    logger.info(f"ML service cache hit for {endpoint}")
                    return cached_result
            
            url = f"{ML_SERVICE_URL}{endpoint}"
            response = http_session.get(url, params=params, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                if use_cache:
                    cache.set(cache_key, result, timeout=1800)
                return result
            else:
                logger.warning(f"ML service returned {response.status_code} for {endpoint}")
                return None
                
        except Exception as e:
            logger.warning(f"ML service call failed for {endpoint}: {e}")
            return None
    
    @staticmethod
    def process_ml_recommendations(ml_response, limit=20):
        try:
            if not ml_response or 'recommendations' not in ml_response:
                return []
            
            recommendations = []
            ml_recs = ml_response['recommendations'][:limit]
            
            content_ids = []
            for rec in ml_recs:
                if isinstance(rec, dict) and 'content_id' in rec:
                    content_ids.append(rec['content_id'])
                elif isinstance(rec, int):
                    content_ids.append(rec)
            
            if not content_ids:
                return []
            
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_dict = {content.id: content for content in contents}
            
            for i, rec in enumerate(ml_recs):
                content_id = rec['content_id'] if isinstance(rec, dict) else rec
                content = content_dict.get(content_id)
                
                if content:
                    content_data = {
                        'content': content,
                        'ml_score': rec.get('score', 0) if isinstance(rec, dict) else 0,
                        'ml_reason': rec.get('reason', '') if isinstance(rec, dict) else '',
                        'ml_rank': i + 1
                    }
                    recommendations.append(content_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error processing ML recommendations: {e}")
            return []

class TMDBService:
    BASE_URL = 'https://api.themoviedb.org/3'
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def search_content(query, content_type='multi', language='en-US', page=1):
        url = f"{TMDBService.BASE_URL}/search/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'language': language,
            'page': page
        }
        
        try:
            response = http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB search error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=7200)
    def get_content_details(content_id, content_type='movie'):
        url = f"{TMDBService.BASE_URL}/{content_type}/{content_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'append_to_response': 'credits,videos,similar,reviews,recommendations'
        }
        
        try:
            response = http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB details error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=1800)
    def get_trending(content_type='all', time_window='day', page=1):
        url = f"{TMDBService.BASE_URL}/trending/{content_type}/{time_window}"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        
        try:
            response = http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB trending error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_popular(content_type='movie', page=1, region=None):
        url = f"{TMDBService.BASE_URL}/{content_type}/popular"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        if region:
            params['region'] = region
        
        try:
            response = http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB popular error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_new_releases(content_type='movie', region=None, page=1):
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'primary_release_date.gte': start_date,
            'primary_release_date.lte': end_date,
            'sort_by': 'release_date.desc',
            'page': page
        }
        
        if region:
            params['region'] = region
        
        try:
            response = http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB new releases error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_critics_choice(content_type='movie', page=1):
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'vote_average.gte': 7.5,
            'vote_count.gte': 100,
            'sort_by': 'vote_average.desc',
            'page': page
        }
        
        try:
            response = http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB critics choice error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_by_genre(genre_id, content_type='movie', page=1, region=None):
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': genre_id,
            'sort_by': 'popularity.desc',
            'page': page
        }
        
        if region:
            params['region'] = region
        
        try:
            response = http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB genre search error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_language_specific(language_code, content_type='movie', page=1):
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_original_language': language_code,
            'sort_by': 'popularity.desc',
            'page': page
        }
        
        try:
            response = http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB language search error: {e}")
        return None

class OMDbService:
    BASE_URL = 'http://www.omdbapi.com/'
    
    @staticmethod
    @cache.memoize(timeout=7200)
    def get_content_by_imdb(imdb_id):
        params = {
            'apikey': OMDB_API_KEY,
            'i': imdb_id,
            'plot': 'full'
        }
        
        try:
            response = http_session.get(OMDbService.BASE_URL, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"OMDb error: {e}")
        return None

class JikanService:
    BASE_URL = 'https://api.jikan.moe/v4'
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def search_anime(query, page=1):
        url = f"{JikanService.BASE_URL}/anime"
        params = {
            'q': query,
            'page': page,
            'limit': 20
        }
        
        try:
            response = http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan search error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=7200)
    def get_anime_details(anime_id):
        url = f"{JikanService.BASE_URL}/anime/{anime_id}/full"
        
        try:
            response = http_session.get(url, params={}, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan anime details error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_top_anime(type='tv', page=1):
        url = f"{JikanService.BASE_URL}/top/anime"
        params = {
            'type': type,
            'page': page
        }
        
        try:
            response = http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan top anime error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_anime_by_genre(genre_name, page=1):
        url = f"{JikanService.BASE_URL}/anime"
        params = {
            'genres': genre_name,
            'order_by': 'score',
            'sort': 'desc',
            'page': page
        }
        
        try:
            response = http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan genre search error: {e}")
        return None

class YouTubeService:
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    @staticmethod
    @cache.memoize(timeout=86400)
    def search_trailers(query, content_type='movie'):
        url = f"{YouTubeService.BASE_URL}/search"
        
        if content_type == 'anime':
            search_query = f"{query} anime trailer PV"
        else:
            search_query = f"{query} official trailer"
        
        params = {
            'key': YOUTUBE_API_KEY,
            'q': search_query,
            'part': 'snippet',
            'type': 'video',
            'maxResults': 5,
            'order': 'relevance'
        }
        
        try:
            response = http_session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
        return None

class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                if existing.updated_at < datetime.utcnow() - timedelta(hours=24):
                    ContentService.update_content_from_tmdb(existing, tmdb_data)
                return existing
            
            genres = []
            if 'genres' in tmdb_data:
                genres = [genre['name'] for genre in tmdb_data['genres']]
            elif 'genre_ids' in tmdb_data:
                genres = ContentService.map_genre_ids(tmdb_data['genre_ids'])
            
            languages = []
            if 'spoken_languages' in tmdb_data:
                languages = [lang['name'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                # Map language code to language name
                lang_code = tmdb_data['original_language']
                lang_map = {'te': 'telugu', 'hi': 'hindi', 'ta': 'tamil', 'ml': 'malayalam', 'en': 'english'}
                languages = [lang_map.get(lang_code, lang_code)]
            
            is_new_release = False
            release_date = None
            if tmdb_data.get('release_date') or tmdb_data.get('first_air_date'):
                date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
                try:
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    if release_date >= (datetime.now() - timedelta(days=90)).date():
                        is_new_release = True
                except:
                    pass
            
            is_critics_choice = False
            critics_score = tmdb_data.get('vote_average', 0)
            vote_count = tmdb_data.get('vote_count', 0)
            if critics_score >= 7.5 and vote_count >= 100:
                is_critics_choice = True
            
            youtube_trailer_id = ContentService.get_youtube_trailer(tmdb_data.get('title') or tmdb_data.get('name'))
            
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
                youtube_trailer_id=youtube_trailer_id,
                is_new_release=is_new_release,
                is_critics_choice=is_critics_choice,
                critics_score=critics_score
            )
            
            db.session.add(content)
            db.session.commit()
            
            cache.set(content_cache_key(content.id), content, timeout=7200)
            
            return content
            
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def update_content_from_tmdb(content, tmdb_data):
        try:
            content.rating = tmdb_data.get('vote_average', content.rating)
            content.vote_count = tmdb_data.get('vote_count', content.vote_count)
            content.popularity = tmdb_data.get('popularity', content.popularity)
            content.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            cache.delete(content_cache_key(content.id))
            
        except Exception as e:
            logger.error(f"Error updating content: {e}")
            db.session.rollback()
    
    @staticmethod
    def save_anime_content(anime_data):
        try:
            existing = Content.query.filter_by(mal_id=anime_data['mal_id']).first()
            if existing:
                return existing
            
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            anime_genre_categories = []
            for genre in genres:
                for category, category_genres in ANIME_GENRES.items():
                    if genre in category_genres:
                        anime_genre_categories.append(category)
            
            anime_genre_categories = list(set(anime_genre_categories))
            
            release_date = None
            if anime_data.get('aired', {}).get('from'):
                try:
                    release_date = datetime.strptime(anime_data['aired']['from'][:10], '%Y-%m-%d').date()
                except:
                    pass
            
            youtube_trailer_id = ContentService.get_youtube_trailer(anime_data.get('title'), 'anime')
            
            content = Content(
                mal_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                anime_genres=json.dumps(anime_genre_categories),
                languages=json.dumps(['japanese']),
                release_date=release_date,
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                youtube_trailer_id=youtube_trailer_id
            )
            
            db.session.add(content)
            db.session.commit()
            
            cache.set(content_cache_key(content.id), content, timeout=7200)
            
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def get_youtube_trailer(title, content_type='movie'):
        try:
            youtube_results = YouTubeService.search_trailers(title, content_type)
            if youtube_results and youtube_results.get('items'):
                return youtube_results['items'][0]['id']['videoId']
        except Exception as e:
            logger.error(f"Error getting YouTube trailer: {e}")
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

class RecommendationEngine:
    @staticmethod
    @cache.memoize(timeout=1800)
    def get_trending_recommendations(limit=20, content_type='all', region=None):
        """Get trending recommendations using advanced algorithms"""
        try:
            global trending_service
            if not trending_service:
                logger.warning("Trending service not initialized, using fallback")
                return RecommendationEngine._get_trending_fallback(limit, content_type, region)
            
            if region and region.upper() == 'IN':
                languages = ['telugu', 'hindi', 'tamil', 'malayalam', 'english']
            else:
                languages = ['english']
            
            categories_map = {
                'all': ['trending_movies', 'trending_tv', 'trending_anime'],
                'movie': ['trending_movies'],
                'tv': ['trending_tv'],
                'anime': ['trending_anime']
            }
            
            categories = categories_map.get(content_type, ['trending_movies'])
            
            results = trending_service.get_trending(
                languages=languages,
                categories=categories,
                limit=limit
            )
            
            all_trending = []
            for category, items in results.items():
                all_trending.extend(items[:limit])
            
            content_objects = []
            
            for item in all_trending[:limit]:
                if 'tmdb_id' in item:
                    content = Content.query.filter_by(tmdb_id=item['tmdb_id']).first()
                    if content:
                        content_objects.append(content)
            
            return content_objects if content_objects else all_trending[:limit]
            
        except Exception as e:
            logger.error(f"Error in trending recommendations: {e}")
            return RecommendationEngine._get_trending_fallback(limit, content_type, region)
    
    @staticmethod
    def _get_trending_fallback(limit=20, content_type='all', region=None):
        """Fallback method for trending if advanced service fails"""
        try:
            trending_day = TMDBService.get_trending(content_type=content_type, time_window='day')
            recommendations = []
            
            if trending_day and 'results' in trending_day:
                for item in trending_day['results'][:limit]:
                    content_type_detected = 'movie' if 'title' in item else 'tv'
                    content = ContentService.save_content_from_tmdb(item, content_type_detected)
                    if content:
                        recommendations.append(content)
            
            return recommendations
        except:
            return []
    
    @staticmethod
    @cache.memoize(timeout=1800)
    def get_new_releases(limit=20, language=None, content_type='movie'):
        """
        Get new releases with language priority:
        Priority Order: Telugu > Hindi > English > Tamil > Malayalam
        """
        try:
            # If specific language requested
            if language and language.lower() in LANGUAGE_PRIORITY:
                return RecommendationEngine._get_language_new_releases(
                    language.lower(), limit, content_type
                )
            
            # Get new releases for all priority languages
            all_new_releases = []
            remaining_limit = limit
            
            # Sort languages by priority
            sorted_languages = sorted(
                LANGUAGE_PRIORITY.items(), 
                key=lambda x: x[1]['priority']
            )
            
            for lang_name, lang_config in sorted_languages:
                if remaining_limit <= 0:
                    break
                
                # Calculate how many items to fetch for this language
                lang_limit = min(lang_config['min_items'], remaining_limit)
                
                # Fetch new releases for this language
                lang_releases = RecommendationEngine._get_language_new_releases(
                    lang_name, lang_limit, content_type
                )
                
                if lang_releases:
                    all_new_releases.extend(lang_releases)
                    remaining_limit -= len(lang_releases)
                    logger.info(f"Added {len(lang_releases)} new releases for {lang_name}")
            
            # If we still need more items, add more from priority languages
            if len(all_new_releases) < limit:
                for lang_name, lang_config in sorted_languages:
                    if len(all_new_releases) >= limit:
                        break
                    
                    additional_limit = limit - len(all_new_releases)
                    additional_releases = RecommendationEngine._get_language_new_releases(
                        lang_name, additional_limit, content_type, offset=lang_config['min_items']
                    )
                    
                    if additional_releases:
                        # Filter out duplicates
                        existing_ids = {r.id for r in all_new_releases}
                        for release in additional_releases:
                            if release.id not in existing_ids:
                                all_new_releases.append(release)
                                if len(all_new_releases) >= limit:
                                    break
            
            return all_new_releases[:limit]
            
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return RecommendationEngine._get_new_releases_fallback(limit, language, content_type)
    
    @staticmethod
    def _get_language_new_releases(language, limit, content_type='movie', offset=0):
        """Get new releases for a specific language with high accuracy"""
        try:
            if language not in LANGUAGE_PRIORITY:
                return []
            
            lang_config = LANGUAGE_PRIORITY[language]
            lang_code = lang_config['code']
            
            all_releases = []
            
            # Method 1: Get by original language with region filter
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            for region in lang_config['regions']:
                if len(all_releases) >= limit:
                    break
                
                url = f"{TMDBService.BASE_URL}/discover/{content_type}"
                params = {
                    'api_key': TMDB_API_KEY,
                    'with_original_language': lang_code,
                    'sort_by': 'release_date.desc' if content_type == 'movie' else 'first_air_date.desc',
                    'page': 1 + (offset // 20),
                    'region': region
                }
                
                # Add date filters based on content type
                if content_type == 'movie':
                    params['primary_release_date.gte'] = start_date
                    params['primary_release_date.lte'] = end_date
                else:  # TV shows
                    params['first_air_date.gte'] = start_date
                    params['first_air_date.lte'] = end_date
                
                try:
                    response = http_session.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Process results
                        for item in data.get('results', []):
                            if len(all_releases) >= limit:
                                break
                            
                            # Skip items for offset
                            if offset > 0:
                                offset -= 1
                                continue
                            
                            # Verify language match and it's a new release
                            if 'original_language' in item and item['original_language'] == lang_code:
                                content = ContentService.save_content_from_tmdb(item, content_type)
                                if content:
                                    # Ensure language is set correctly
                                    content.languages = json.dumps([language])
                                    content.is_new_release = True
                                    db.session.commit()
                                    
                                    # Check for duplicates
                                    if content not in all_releases:
                                        all_releases.append(content)
                                        logger.debug(f"Added {content.title} to {language} new releases")
                                        
                except Exception as e:
                    logger.error(f"Error fetching {language} releases from region {region}: {e}")
            
            # Method 2: Search with language-specific keywords for recent releases
            if len(all_releases) < limit:
                for keyword in lang_config['keywords']:
                    if len(all_releases) >= limit:
                        break
                    
                    search_url = f"{TMDBService.BASE_URL}/search/{content_type}"
                    search_params = {
                        'api_key': TMDB_API_KEY,
                        'query': keyword,
                        'year': datetime.now().year,
                        'page': 1,
                        'language': 'en-US',
                        'include_adult': False
                    }
                    
                    try:
                        response = http_session.get(search_url, params=search_params, timeout=10)
                        if response.status_code == 200:
                            search_data = response.json()
                            
                            for item in search_data.get('results', []):
                                if len(all_releases) >= limit:
                                    break
                                
                                # Check if it's a new release
                                release_date_field = 'release_date' if content_type == 'movie' else 'first_air_date'
                                if release_date_field in item and item[release_date_field]:
                                    try:
                                        release_date = datetime.strptime(item[release_date_field], '%Y-%m-%d')
                                        days_old = (datetime.now() - release_date).days
                                        
                                        if days_old <= 90:  # New release within 90 days
                                            # Check if not already added
                                            existing_ids = {r.tmdb_id for r in all_releases if r.tmdb_id}
                                            if item['id'] not in existing_ids:
                                                content = ContentService.save_content_from_tmdb(item, content_type)
                                                if content:
                                                    content.languages = json.dumps([language])
                                                    content.is_new_release = True
                                                    db.session.commit()
                                                    all_releases.append(content)
                                                    logger.debug(f"Added {content.title} from keyword search")
                                    except:
                                        pass
                    except Exception as e:
                        logger.error(f"Error searching {keyword}: {e}")
            
            # Method 3: Get from ML service if available
            if len(all_releases) < limit and ML_SERVICE_URL:
                ml_params = {
                    'limit': limit - len(all_releases),
                    'language': language,
                    'content_type': content_type
                }
                
                ml_response = MLServiceClient.call_ml_service('/api/new-releases', ml_params)
                if ml_response:
                    ml_recommendations = MLServiceClient.process_ml_recommendations(
                        ml_response, limit - len(all_releases)
                    )
                    if ml_recommendations:
                        for rec in ml_recommendations:
                            if rec['content'] not in all_releases:
                                content = rec['content']
                                content.languages = json.dumps([language])
                                content.is_new_release = True
                                db.session.commit()
                                all_releases.append(content)
            
            logger.info(f"Found {len(all_releases)} new releases for {language}")
            return all_releases
            
        except Exception as e:
            logger.error(f"Error getting {language} new releases: {e}")
            return []
    
    @staticmethod
    def _get_new_releases_fallback(limit, language, content_type):
        """Fallback method for new releases"""
        try:
            logger.info("Using fallback method for new releases")
            
            language_code = None
            if language:
                lang_mapping = {'hindi': 'hi', 'telugu': 'te', 'tamil': 'ta', 
                              'kannada': 'kn', 'malayalam': 'ml', 'english': 'en'}
                language_code = lang_mapping.get(language.lower())
            
            recommendations = []
            
            if language_code:
                new_releases = TMDBService.get_language_specific(language_code, content_type)
            else:
                new_releases = TMDBService.get_new_releases(content_type)
            
            if new_releases:
                for item in new_releases.get('results', [])[:limit]:
                    content = ContentService.save_content_from_tmdb(item, content_type)
                    if content:
                        recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Fallback new releases error: {e}")
            return []
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_critics_choice(limit=20, content_type='movie'):
        try:
            ml_params = {
                'limit': limit,
                'content_type': content_type
            }
            
            ml_response = MLServiceClient.call_ml_service('/api/critics-choice', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for critics choice: {len(ml_recommendations)} items")
                    return [rec['content'] for rec in ml_recommendations]
            
            logger.info("Falling back to TMDB for critics choice")
            
            critics_choice = TMDBService.get_critics_choice(content_type)
            
            recommendations = []
            if critics_choice:
                for item in critics_choice.get('results', [])[:limit]:
                    content = ContentService.save_content_from_tmdb(item, content_type)
                    if content:
                        recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting critics choice: {e}")
            return []
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_genre_recommendations(genre, limit=20, content_type='movie', region=None):
        try:
            ml_params = {
                'limit': limit,
                'content_type': content_type,
                'region': region
            }
            
            ml_response = MLServiceClient.call_ml_service(f'/api/genre/{genre}', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for genre {genre}: {len(ml_recommendations)} items")
                    return [rec['content'] for rec in ml_recommendations]
            
            logger.info(f"Falling back to TMDB for genre {genre}")
            
            genre_ids = {
                'action': 28, 'adventure': 12, 'animation': 16, 'biography': -1,
                'comedy': 35, 'crime': 80, 'documentary': 99, 'drama': 18,
                'fantasy': 14, 'horror': 27, 'musical': 10402, 'mystery': 9648,
                'romance': 10749, 'sci-fi': 878, 'thriller': 53, 'western': 37
            }
            
            genre_id = genre_ids.get(genre.lower())
            if not genre_id or genre_id == -1:
                search_results = TMDBService.search_content(genre, content_type)
                recommendations = []
                if search_results:
                    for item in search_results.get('results', [])[:limit]:
                        content = ContentService.save_content_from_tmdb(item, content_type)
                        if content:
                            recommendations.append(content)
                return recommendations
            
            genre_content = TMDBService.get_by_genre(genre_id, content_type, region=region)
            
            recommendations = []
            if genre_content:
                for item in genre_content.get('results', [])[:limit]:
                    content = ContentService.save_content_from_tmdb(item, content_type)
                    if content:
                        recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting genre recommendations: {e}")
            return []
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_regional_recommendations(language, limit=20, content_type='movie'):
        try:
            ml_params = {
                'limit': limit,
                'content_type': content_type
            }
            
            ml_response = MLServiceClient.call_ml_service(f'/api/regional/{language}', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for regional {language}: {len(ml_recommendations)} items")
                    return [rec['content'] for rec in ml_recommendations]
            
            logger.info(f"Falling back to TMDB for regional {language}")
            
            lang_mapping = {
                'hindi': 'hi', 'telugu': 'te', 'tamil': 'ta', 
                'kannada': 'kn', 'malayalam': 'ml', 'english': 'en'
            }
            
            language_code = lang_mapping.get(language.lower())
            recommendations = []
            
            if language_code:
                lang_content = TMDBService.get_language_specific(language_code, content_type)
                if lang_content:
                    for item in lang_content.get('results', [])[:limit]:
                        content = ContentService.save_content_from_tmdb(item, content_type)
                        if content:
                            recommendations.append(content)
            
            if len(recommendations) < limit:
                search_queries = REGIONAL_LANGUAGES.get(language.lower(), [language])
                for query in search_queries:
                    if len(recommendations) >= limit:
                        break
                    
                    search_results = TMDBService.search_content(query, content_type)
                    if search_results:
                        for item in search_results.get('results', []):
                            if len(recommendations) >= limit:
                                break
                            content = ContentService.save_content_from_tmdb(item, content_type)
                            if content:
                                recommendations.append(content)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting regional recommendations: {e}")
            return []
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_anime_recommendations(limit=20, genre=None):
        try:
            ml_params = {
                'limit': limit,
                'genre': genre
            }
            
            ml_response = MLServiceClient.call_ml_service('/api/anime', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for anime recommendations: {len(ml_recommendations)} items")
                    return [rec['content'] for rec in ml_recommendations]
            
            logger.info("Falling back to Jikan API for anime recommendations")
            
            recommendations = []
            
            if genre and genre.lower() in ANIME_GENRES:
                genre_keywords = ANIME_GENRES[genre.lower()]
                for keyword in genre_keywords[:2]:
                    anime_results = JikanService.get_anime_by_genre(keyword)
                    if anime_results:
                        for anime in anime_results.get('data', []):
                            if len(recommendations) >= limit:
                                break
                            content = ContentService.save_anime_content(anime)
                            if content:
                                recommendations.append(content)
                    if len(recommendations) >= limit:
                        break
            else:
                top_anime = JikanService.get_top_anime()
                if top_anime:
                    for anime in top_anime.get('data', [])[:limit]:
                        content = ContentService.save_anime_content(anime)
                        if content:
                            recommendations.append(content)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting anime recommendations: {e}")
            return []
    
    @staticmethod
    def get_similar_recommendations(content_id, limit=20):
        try:
            cache_key = f"similar:{content_id}:{limit}"
            cached_result = cache.get(cache_key)
            if cached_result:
                return cached_result
            
            ml_params = {
                'limit': limit
            }
            
            ml_response = MLServiceClient.call_ml_service(f'/api/similar/{content_id}', ml_params, use_cache=False)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for similar recommendations: {len(ml_recommendations)} items")
                    result = [rec['content'] for rec in ml_recommendations]
                    cache.set(cache_key, result, timeout=3600)
                    return result
            
            logger.info("Falling back to TMDB/database for similar recommendations")
            
            base_content = Content.query.get(content_id)
            if not base_content:
                return []
            
            similar_content = []
            
            if base_content.tmdb_id and base_content.content_type != 'anime':
                tmdb_details = TMDBService.get_content_details(base_content.tmdb_id, base_content.content_type)
                if tmdb_details:
                    if 'similar' in tmdb_details:
                        for item in tmdb_details['similar']['results'][:10]:
                            content = ContentService.save_content_from_tmdb(item, base_content.content_type)
                            if content:
                                similar_content.append(content)
                    
                    if 'recommendations' in tmdb_details:
                        for item in tmdb_details['recommendations']['results'][:10]:
                            content = ContentService.save_content_from_tmdb(item, base_content.content_type)
                            if content:
                                similar_content.append(content)
            
            if base_content.genres:
                genres = json.loads(base_content.genres)
                if genres:
                    db_similar = Content.query.filter(
                        Content.id != content_id,
                        Content.content_type == base_content.content_type
                    ).all()
                    
                    scored_content = []
                    for content in db_similar:
                        if content.genres:
                            content_genres = json.loads(content.genres)
                            overlap = len(set(genres) & set(content_genres))
                            if overlap > 0:
                                scored_content.append((content, overlap))
                    
                    scored_content.sort(key=lambda x: x[1], reverse=True)
                    for content, score in scored_content[:10]:
                        if content not in similar_content:
                            similar_content.append(content)
            
            seen_ids = set()
            unique_similar = []
            for content in similar_content:
                if content.id not in seen_ids:
                    seen_ids.add(content.id)
                    unique_similar.append(content)
                    if len(unique_similar) >= limit:
                        break
            
            cache.set(cache_key, unique_similar, timeout=3600)
            
            return unique_similar
        except Exception as e:
            logger.error(f"Error getting similar recommendations: {e}")
            return []

# ================== Content Routes ==================

@app.route('/api/search', methods=['GET'])
@cache.cached(timeout=300, key_prefix=make_cache_key)
def search_content():
    try:
        from users import get_session_id
        
        query = request.args.get('query', '')
        content_type = request.args.get('type', 'multi')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        session_id = get_session_id()
        
        futures = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures.append(executor.submit(TMDBService.search_content, query, content_type, page=page))
            
            if content_type in ['anime', 'multi']:
                futures.append(executor.submit(JikanService.search_anime, query, page=page))
        
        tmdb_results = futures[0].result()
        anime_results = futures[1].result() if len(futures) > 1 else None
        
        results = []
        
        if tmdb_results:
            for item in tmdb_results.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    interaction = AnonymousInteraction(
                        session_id=session_id,
                        content_id=content.id,
                        interaction_type='search',
                        ip_address=request.remote_addr
                    )
                    db.session.add(interaction)
                    
                    youtube_url = None
                    if content.youtube_trailer_id:
                        youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                    
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
                        'youtube_trailer': youtube_url
                    })
        
        if anime_results:
            for anime in anime_results.get('data', []):
                content = ContentService.save_anime_content(anime)
                if content:
                    youtube_url = None
                    if content.youtube_trailer_id:
                        youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                    
                    results.append({
                        'id': content.id,
                        'mal_id': content.mal_id,
                        'title': content.title,
                        'content_type': 'anime',
                        'genres': json.loads(content.genres or '[]'),
                        'anime_genres': json.loads(content.anime_genres or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': content.poster_path,
                        'overview': content.overview,
                        'youtube_trailer': youtube_url
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
        from users import get_session_id
        
        cache_key = content_cache_key(content_id)
        cached_content = cache.get(cache_key)
        
        if cached_content:
            content = cached_content
        else:
            content = Content.query.get_or_404(content_id)
            cache.set(cache_key, content, timeout=7200)
        
        session_id = get_session_id()
        interaction = AnonymousInteraction(
            session_id=session_id,
            content_id=content.id,
            interaction_type='view',
            ip_address=request.remote_addr
        )
        db.session.add(interaction)
        
        additional_details = None
        cast = []
        crew = []
        
        if content.content_type == 'anime' and content.mal_id:
            additional_details = JikanService.get_anime_details(content.mal_id)
            if additional_details:
                anime_data = additional_details.get('data', {})
                if 'voices' in anime_data:
                    cast = anime_data['voices'][:10]
                if 'staff' in anime_data:
                    crew = anime_data['staff'][:5]
        elif content.tmdb_id:
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
            if additional_details:
                cast = additional_details.get('credits', {}).get('cast', [])[:10]
                crew = additional_details.get('credits', {}).get('crew', [])[:5]
        
        similar_content = RecommendationEngine.get_similar_recommendations(content.id, limit=10)
        
        similar_formatted = []
        for similar in similar_content:
            youtube_url = None
            if similar.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={similar.youtube_trailer_id}"
            
            similar_formatted.append({
                'id': similar.id,
                'title': similar.title,
                'poster_path': f"https://image.tmdb.org/t/p/w300{similar.poster_path}" if similar.poster_path and not similar.poster_path.startswith('http') else similar.poster_path,
                'rating': similar.rating,
                'content_type': similar.content_type,
                'youtube_trailer': youtube_url
            })
        
        db.session.commit()
        
        youtube_trailer_url = None
        if content.youtube_trailer_id:
            youtube_trailer_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
        
        response_data = {
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'mal_id': content.mal_id,
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
            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path and not content.backdrop_path.startswith('http') else content.backdrop_path,
            'youtube_trailer': youtube_trailer_url,
            'similar_content': similar_formatted,
            'cast': cast,
            'crew': crew,
            'is_trending': content.is_trending,
            'is_new_release': content.is_new_release,
            'is_critics_choice': content.is_critics_choice
        }
        
        if content.content_type == 'anime':
            response_data['anime_genres'] = json.loads(content.anime_genres or '[]')
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# ================== Recommendation Routes ==================

@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    """Get trending content using advanced algorithms"""
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        languages = request.args.getlist('languages')
        categories = request.args.getlist('categories')
        
        if not languages:
            languages = ['telugu', 'english', 'hindi', 'tamil', 'malayalam']
        
        if not categories:
            categories = ['trending_movies', 'trending_tv', 'trending_anime', 
                        'rising_fast', 'popular_regional']
        
        global trending_service
        if not trending_service:
            trending_service = init_services()
        
        results = trending_service.get_trending(
            languages=languages,
            categories=categories,
            limit=limit
        )
        
        if content_type == 'all':
            return jsonify(results), 200
        elif content_type == 'movie':
            return jsonify({
                'recommendations': results.get('trending_movies', []),
                'type': 'movie'
            }), 200
        elif content_type == 'tv':
            return jsonify({
                'recommendations': results.get('trending_tv', []),
                'type': 'tv'
            }), 200
        elif content_type == 'anime':
            return jsonify({
                'recommendations': results.get('trending_anime', []),
                'type': 'anime'
            }), 200
        elif content_type == 'rising':
            return jsonify({
                'recommendations': results.get('rising_fast', []),
                'type': 'rising'
            }), 200
        elif content_type == 'regional':
            return jsonify({
                'recommendations': results.get('popular_regional', []),
                'type': 'regional'
            }), 200
        else:
            return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/recommendations/trending/categories', methods=['GET'])
def get_trending_categories():
    """Get all trending categories with detailed information"""
    try:
        languages = request.args.getlist('languages')
        if not languages:
            languages = ['telugu', 'english', 'hindi', 'tamil', 'malayalam']
        
        global trending_service
        if not trending_service:
            trending_service = init_services()
        
        results = trending_service.get_trending(
            languages=languages,
            categories=['trending_movies', 'trending_tv', 'trending_anime', 
                       'rising_fast', 'popular_regional'],
            limit=20
        )
        
        response = {
            'timestamp': datetime.utcnow().isoformat(),
            'languages': languages,
            'categories': {}
        }
        
        for category, items in results.items():
            response['categories'][category] = {
                'count': len(items),
                'items': items
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Trending categories error: {e}")
        return jsonify({'error': 'Failed to get trending categories'}), 500

@app.route('/api/recommendations/new-releases', methods=['GET'])
@cache.cached(timeout=300, key_prefix=make_cache_key)
def get_new_releases():
    """
    Get new releases with language priority
    Priority: Telugu > Hindi > English > Tamil > Malayalam
    """
    try:
        language = request.args.get('language')  # Specific language filter
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        include_all_languages = request.args.get('all_languages', 'true').lower() == 'true'
        
        # Get new releases
        if language:
            # Get for specific language only
            recommendations = RecommendationEngine.get_new_releases(limit, language, content_type)
        elif include_all_languages:
            # Get with language priority (default behavior)
            recommendations = RecommendationEngine.get_new_releases(limit, None, content_type)
        else:
            # Get only English releases
            recommendations = RecommendationEngine.get_new_releases(limit, 'english', content_type)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            # Determine the primary language
            languages = json.loads(content.languages or '[]')
            primary_language = languages[0] if languages else 'unknown'
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': languages,
                'primary_language': primary_language,
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'youtube_trailer': youtube_url,
                'is_new_release': content.is_new_release
            })
        
        # Sort by language priority and then by release date
        def sort_key(item):
            lang = item['primary_language'].lower()
            priority = LANGUAGE_PRIORITY.get(lang, {}).get('priority', 999)
            release_date = item['release_date'] or '1900-01-01'
            return (priority, release_date)
        
        result.sort(key=sort_key)
        
        # Calculate language distribution
        language_distribution = Counter([r['primary_language'] for r in result])
        
        return jsonify({
            'recommendations': result,
            'total': len(result),
            'language_distribution': dict(language_distribution),
            'priority_order': ['telugu', 'hindi', 'english', 'tamil', 'malayalam']
        }), 200
        
    except Exception as e:
        logger.error(f"New releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

@app.route('/api/recommendations/new-releases/<language>', methods=['GET'])
@cache.cached(timeout=300, key_prefix=make_cache_key)
def get_language_new_releases(language):
    """Get new releases for a specific language"""
    try:
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        days = int(request.args.get('days', 90))  # How many days back to look
        
        if language.lower() not in LANGUAGE_PRIORITY:
            return jsonify({'error': f'Language {language} not supported'}), 400
        
        # Get language-specific new releases
        recommendations = RecommendationEngine._get_language_new_releases(
            language.lower(), limit, content_type
        )
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'youtube_trailer': youtube_url,
                'is_new_release': True
            })
        
        # Sort by release date (newest first)
        result.sort(key=lambda x: x['release_date'] or '1900-01-01', reverse=True)
        
        return jsonify({
            'language': language,
            'recommendations': result,
            'total': len(result)
        }), 200
        
    except Exception as e:
        logger.error(f"Language new releases error: {e}")
        return jsonify({'error': f'Failed to get {language} new releases'}), 500

@app.route('/api/recommendations/critics-choice', methods=['GET'])
@cache.cached(timeout=600, key_prefix=make_cache_key)
def get_critics_choice():
    try:
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_critics_choice(limit, content_type)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url,
                'is_critics_choice': content.is_critics_choice,
                'critics_score': content.critics_score
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Critics choice error: {e}")
        return jsonify({'error': 'Failed to get critics choice'}), 500

@app.route('/api/recommendations/genre/<genre>', methods=['GET'])
@cache.cached(timeout=600, key_prefix=make_cache_key)
def get_genre_recommendations(genre):
    try:
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        region = request.args.get('region')
        
        recommendations = RecommendationEngine.get_genre_recommendations(genre, limit, content_type, region)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Genre recommendations error: {e}")
        return jsonify({'error': 'Failed to get genre recommendations'}), 500

@app.route('/api/recommendations/regional/<language>', methods=['GET'])
@cache.cached(timeout=600, key_prefix=make_cache_key)
def get_regional(language):
    try:
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_regional_recommendations(language, limit, content_type)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Regional recommendations error: {e}")
        return jsonify({'error': 'Failed to get regional recommendations'}), 500

@app.route('/api/recommendations/anime', methods=['GET'])
@cache.cached(timeout=600, key_prefix=make_cache_key)
def get_anime():
    try:
        genre = request.args.get('genre')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_anime_recommendations(limit, genre)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'mal_id': content.mal_id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'anime_genres': json.loads(content.anime_genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Anime recommendations error: {e}")
        return jsonify({'error': 'Failed to get anime recommendations'}), 500

@app.route('/api/recommendations/similar/<int:content_id>', methods=['GET'])
def get_similar_recommendations(content_id):
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_similar_recommendations(content_id, limit)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Similar recommendations error: {e}")
        return jsonify({'error': 'Failed to get similar recommendations'}), 500

@app.route('/api/recommendations/admin-choice', methods=['GET'])
@cache.cached(timeout=600, key_prefix=make_cache_key)
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
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'youtube_trailer': youtube_url,
                    'admin_description': rec.description,
                    'admin_name': admin.username if admin else 'Admin',
                    'recommended_at': rec.created_at.isoformat()
                })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Public admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get admin recommendations'}), 500

# ================== Health Check ==================

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        health_info = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '3.0.0'  # Updated version
        }
        
        # Database check
        try:
            db.session.execute('SELECT 1')
            health_info['database'] = 'connected'
        except:
            health_info['database'] = 'disconnected'
            health_info['status'] = 'degraded'
        
        # Cache check
        try:
            cache.set('health_check', 'ok', timeout=10)
            if cache.get('health_check') == 'ok':
                health_info['cache'] = 'connected'
            else:
                health_info['cache'] = 'error'
                health_info['status'] = 'degraded'
        except:
            health_info['cache'] = 'disconnected'
            health_info['status'] = 'degraded'
        
        # Trending service check
        try:
            global trending_service
            if trending_service:
                health_info['trending_service'] = 'active'
                health_info['trending_algorithms'] = [
                    'MRTSA', 'VBTD', 'GCTM', 'BOPP', 
                    'OPAA', 'TPRT', 'ADVC', 'MLCPD', 'UTSA'
                ]
            else:
                health_info['trending_service'] = 'not_initialized'
        except:
            health_info['trending_service'] = 'error'
        
        # Services check
        health_info['services'] = {
            'tmdb': bool(TMDB_API_KEY),
            'omdb': bool(OMDB_API_KEY),
            'youtube': bool(YOUTUBE_API_KEY),
            'ml_service': bool(ML_SERVICE_URL),
            'trending_service': trending_service is not None,
            'modules': {
                'admin': True,
                'users': True,
                'auth': True,
                'trending': True
            }
        }
        
        return jsonify(health_info), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# ================== Database Initialization ==================

def create_tables():
    try:
        with app.app_context():
            db.create_all()
            
            # Create default admin user if not exists
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
            
            # Initialize services after database is ready
            init_services()
            logger.info("All services initialized successfully")
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

# Initialize database
create_tables()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)