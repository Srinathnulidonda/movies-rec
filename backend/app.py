#backend/app.py
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
from collections import defaultdict
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

# Priority Languages Configuration
PRIORITY_LANGUAGES = ['telugu', 'english', 'hindi', 'tamil', 'malayalam', 'kannada']

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
        """Save content from TMDB and return dictionary representation"""
        try:
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                if existing.updated_at < datetime.utcnow() - timedelta(hours=24):
                    ContentService.update_content_from_tmdb(existing, tmdb_data)
                # Return a dictionary representation
                return ContentService._content_to_dict(existing)
            
            # Process genres
            genres = []
            if 'genres' in tmdb_data:
                genres = [genre['name'] for genre in tmdb_data['genres']]
            elif 'genre_ids' in tmdb_data:
                genres = ContentService.map_genre_ids(tmdb_data['genre_ids'])
            
            # Process languages
            languages = []
            if 'spoken_languages' in tmdb_data:
                languages = [lang['name'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
            
            # Process release date
            is_new_release = False
            release_date = None
            if tmdb_data.get('release_date') or tmdb_data.get('first_air_date'):
                date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
                try:
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    if release_date >= (datetime.now() - timedelta(days=60)).date():
                        is_new_release = True
                except:
                    pass
            
            # Check critics choice
            is_critics_choice = False
            critics_score = tmdb_data.get('vote_average', 0)
            vote_count = tmdb_data.get('vote_count', 0)
            if critics_score >= 7.5 and vote_count >= 100:
                is_critics_choice = True
            
            # Get YouTube trailer
            youtube_trailer_id = ContentService.get_youtube_trailer(
                tmdb_data.get('title') or tmdb_data.get('name')
            )
            
            # Create new content
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
            
            # Cache the actual object for internal use
            cache.set(content_cache_key(content.id), content, timeout=7200)
            
            # Return dictionary representation
            return ContentService._content_to_dict(content)
            
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def update_content_from_tmdb(content, tmdb_data):
        """Update existing content from TMDB data"""
        try:
            content.rating = tmdb_data.get('vote_average', content.rating)
            content.vote_count = tmdb_data.get('vote_count', content.vote_count)
            content.popularity = tmdb_data.get('popularity', content.popularity)
            content.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            # Clear cache
            cache.delete(content_cache_key(content.id))
            
        except Exception as e:
            logger.error(f"Error updating content: {e}")
            db.session.rollback()
    
    @staticmethod
    def save_anime_content(anime_data):
        """Save anime content and return dictionary representation"""
        try:
            existing = Content.query.filter_by(mal_id=anime_data['mal_id']).first()
            if existing:
                return ContentService._content_to_dict(existing)
            
            # Process genres
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            # Process anime genre categories
            anime_genre_categories = []
            for genre in genres:
                for category, category_genres in ANIME_GENRES.items():
                    if genre in category_genres:
                        anime_genre_categories.append(category)
            
            anime_genre_categories = list(set(anime_genre_categories))
            
            # Process release date
            release_date = None
            if anime_data.get('aired', {}).get('from'):
                try:
                    release_date = datetime.strptime(anime_data['aired']['from'][:10], '%Y-%m-%d').date()
                except:
                    pass
            
            # Get YouTube trailer
            youtube_trailer_id = ContentService.get_youtube_trailer(anime_data.get('title'), 'anime')
            
            # Create new content
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
            
            # Cache the actual object
            cache.set(content_cache_key(content.id), content, timeout=7200)
            
            # Return dictionary representation
            return ContentService._content_to_dict(content)
            
        except Exception as e:
            logger.error(f"Error saving anime content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def get_youtube_trailer(title, content_type='movie'):
        """Get YouTube trailer ID for content"""
        try:
            youtube_results = YouTubeService.search_trailers(title, content_type)
            if youtube_results and youtube_results.get('items'):
                return youtube_results['items'][0]['id']['videoId']
        except Exception as e:
            logger.error(f"Error getting YouTube trailer: {e}")
        return None
    
    @staticmethod
    def map_genre_ids(genre_ids):
        """Map TMDB genre IDs to genre names"""
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]
    
    @staticmethod
    def _content_to_dict(content):
        """Convert Content object to dictionary"""
        if not content:
            return None
        
        # Handle date serialization
        release_date = None
        if content.release_date:
            if hasattr(content.release_date, 'isoformat'):
                release_date = content.release_date.isoformat()
            else:
                release_date = str(content.release_date)
        
        return {
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'imdb_id': content.imdb_id,
            'mal_id': content.mal_id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': content.genres,
            'anime_genres': content.anime_genres,
            'languages': content.languages,
            'release_date': release_date,
            'runtime': content.runtime,
            'rating': content.rating,
            'vote_count': content.vote_count,
            'popularity': content.popularity,
            'overview': content.overview,
            'poster_path': content.poster_path,
            'backdrop_path': content.backdrop_path,
            'trailer_url': content.trailer_url,
            'youtube_trailer_id': content.youtube_trailer_id,
            'is_trending': content.is_trending,
            'is_new_release': content.is_new_release,
            'is_critics_choice': content.is_critics_choice,
            'critics_score': content.critics_score
        }
class RecommendationEngine:
    @staticmethod
    @cache.memoize(timeout=1800)
    def get_trending_recommendations(limit=20, content_type='all', region=None):
        """Get trending recommendations using advanced algorithms with priority languages"""
        try:
            global trending_service
            if not trending_service:
                logger.warning("Trending service not initialized, using fallback")
                return RecommendationEngine._get_trending_fallback(limit, content_type, region)
            
            # Always use priority languages
            languages = PRIORITY_LANGUAGES
            
            categories_map = {
                'all': ['trending_movies', 'trending_tv', 'trending_anime', 'priority_trending'],
                'movie': ['trending_movies', 'priority_trending'],
                'tv': ['trending_tv'],
                'anime': ['trending_anime']
            }
            
            categories = categories_map.get(content_type, ['trending_movies', 'priority_trending'])
            
            results = trending_service.get_trending(
                languages=languages,
                categories=categories,
                limit=limit
            )
            
            all_trending = []
            for category, items in results.items():
                all_trending.extend(items[:limit])
            
            # Convert to content dictionaries
            content_dicts = []
            for item in all_trending[:limit]:
                if isinstance(item, dict) and 'tmdb_id' in item:
                    # Fetch from database and convert to dict
                    content = Content.query.filter_by(tmdb_id=item['tmdb_id']).first()
                    if content:
                        content_dicts.append(ContentService._content_to_dict(content))
                    else:
                        content_dicts.append(item)
                else:
                    content_dicts.append(item)
            
            return content_dicts if content_dicts else all_trending[:limit]
            
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
                    content_dict = ContentService.save_content_from_tmdb(item, content_type_detected)
                    if content_dict:
                        recommendations.append(content_dict)
            
            return recommendations
        except:
            return []
    
    @staticmethod
    @cache.memoize(timeout=1800)
    def get_new_releases(limit=20, language=None, content_type='movie'):
        """Get new releases - returns list of dictionaries"""
        try:
            # Try ML service first
            ml_params = {
                'limit': limit,
                'language': language,
                'content_type': content_type
            }
            
            ml_response = MLServiceClient.call_ml_service('/api/new-releases', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for new releases: {len(ml_recommendations)} items")
                    # Convert Content objects to dictionaries
                    return [ContentService._content_to_dict(rec['content']) for rec in ml_recommendations]
            
            logger.info("Falling back to TMDB for new releases")
            
            # Language mapping
            language_code = None
            if language:
                lang_mapping = {'hindi': 'hi', 'telugu': 'te', 'tamil': 'ta', 
                              'kannada': 'kn', 'malayalam': 'ml', 'english': 'en'}
                language_code = lang_mapping.get(language.lower())
            
            recommendations = []
            
            # Fetch from TMDB
            if language_code:
                new_releases = TMDBService.get_language_specific(language_code, content_type)
            else:
                new_releases = TMDBService.get_new_releases(content_type)
            
            if new_releases:
                for item in new_releases.get('results', [])[:limit]:
                    content_dict = ContentService.save_content_from_tmdb(item, content_type)
                    if content_dict:
                        recommendations.append(content_dict)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return []
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_critics_choice(limit=20, content_type='movie'):
        """Get critics choice - returns list of dictionaries"""
        try:
            # Try ML service first
            ml_params = {
                'limit': limit,
                'content_type': content_type
            }
            
            ml_response = MLServiceClient.call_ml_service('/api/critics-choice', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for critics choice: {len(ml_recommendations)} items")
                    return [ContentService._content_to_dict(rec['content']) for rec in ml_recommendations]
            
            logger.info("Falling back to TMDB for critics choice")
            
            critics_choice = TMDBService.get_critics_choice(content_type)
            
            recommendations = []
            if critics_choice:
                for item in critics_choice.get('results', [])[:limit]:
                    content_dict = ContentService.save_content_from_tmdb(item, content_type)
                    if content_dict:
                        recommendations.append(content_dict)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting critics choice: {e}")
            return []
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_genre_recommendations(genre, limit=20, content_type='movie', region=None):
        """Get genre recommendations - returns list of dictionaries"""
        try:
            # Try ML service first
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
                    return [ContentService._content_to_dict(rec['content']) for rec in ml_recommendations]
            
            logger.info(f"Falling back to TMDB for genre {genre}")
            
            # Genre ID mapping
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
                        content_dict = ContentService.save_content_from_tmdb(item, content_type)
                        if content_dict:
                            recommendations.append(content_dict)
                return recommendations
            
            genre_content = TMDBService.get_by_genre(genre_id, content_type, region=region)
            
            recommendations = []
            if genre_content:
                for item in genre_content.get('results', [])[:limit]:
                    content_dict = ContentService.save_content_from_tmdb(item, content_type)
                    if content_dict:
                        recommendations.append(content_dict)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting genre recommendations: {e}")
            return []
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_regional_recommendations(language, limit=20, content_type='movie'):
        """Get regional recommendations - returns list of dictionaries"""
        try:
            # Try ML service first
            ml_params = {
                'limit': limit,
                'content_type': content_type
            }
            
            ml_response = MLServiceClient.call_ml_service(f'/api/regional/{language}', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for regional {language}: {len(ml_recommendations)} items")
                    return [ContentService._content_to_dict(rec['content']) for rec in ml_recommendations]
            
            logger.info(f"Falling back to TMDB for regional {language}")
            
            # Language code mapping
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
                        content_dict = ContentService.save_content_from_tmdb(item, content_type)
                        if content_dict:
                            recommendations.append(content_dict)
            
            # Fill with search results if needed
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
                            content_dict = ContentService.save_content_from_tmdb(item, content_type)
                            if content_dict and content_dict not in recommendations:
                                recommendations.append(content_dict)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting regional recommendations: {e}")
            return []
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_anime_recommendations(limit=20, genre=None):
        """Get anime recommendations - returns list of dictionaries"""
        try:
            # Try ML service first
            ml_params = {
                'limit': limit,
                'genre': genre
            }
            
            ml_response = MLServiceClient.call_ml_service('/api/anime', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for anime recommendations: {len(ml_recommendations)} items")
                    return [ContentService._content_to_dict(rec['content']) for rec in ml_recommendations]
            
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
                            content_dict = ContentService.save_anime_content(anime)
                            if content_dict:
                                recommendations.append(content_dict)
                    if len(recommendations) >= limit:
                        break
            else:
                top_anime = JikanService.get_top_anime()
                if top_anime:
                    for anime in top_anime.get('data', [])[:limit]:
                        content_dict = ContentService.save_anime_content(anime)
                        if content_dict:
                            recommendations.append(content_dict)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting anime recommendations: {e}")
            return []
    
    @staticmethod
    def get_similar_recommendations(content_id, limit=20):
        """Get similar recommendations - returns list of dictionaries"""
        try:
            cache_key = f"similar:{content_id}:{limit}"
            cached_result = cache.get(cache_key)
            if cached_result:
                # Ensure cached result is list of dicts
                if cached_result and isinstance(cached_result[0], dict):
                    return cached_result
                else:
                    # Convert to dicts if needed
                    return [ContentService._content_to_dict(c) for c in cached_result]
            
            # Try ML service
            ml_params = {
                'limit': limit
            }
            
            ml_response = MLServiceClient.call_ml_service(f'/api/similar/{content_id}', ml_params, use_cache=False)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for similar recommendations: {len(ml_recommendations)} items")
                    result = [ContentService._content_to_dict(rec['content']) for rec in ml_recommendations]
                    cache.set(cache_key, result, timeout=3600)
                    return result
            
            logger.info("Falling back to TMDB/database for similar recommendations")
            
            base_content = Content.query.get(content_id)
            if not base_content:
                return []
            
            similar_content = []
            
            # Get TMDB similar/recommendations
            if base_content.tmdb_id and base_content.content_type != 'anime':
                tmdb_details = TMDBService.get_content_details(base_content.tmdb_id, base_content.content_type)
                if tmdb_details:
                    if 'similar' in tmdb_details:
                        for item in tmdb_details['similar']['results'][:10]:
                            content_dict = ContentService.save_content_from_tmdb(item, base_content.content_type)
                            if content_dict:
                                similar_content.append(content_dict)
                    
                    if 'recommendations' in tmdb_details:
                        for item in tmdb_details['recommendations']['results'][:10]:
                            content_dict = ContentService.save_content_from_tmdb(item, base_content.content_type)
                            if content_dict and content_dict not in similar_content:
                                similar_content.append(content_dict)
            
            # Get genre-based similar content
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
                        content_dict = ContentService._content_to_dict(content)
                        if content_dict and content_dict not in similar_content:
                            similar_content.append(content_dict)
            
            # Remove duplicates and limit
            seen_ids = set()
            unique_similar = []
            for content_dict in similar_content:
                if content_dict['id'] not in seen_ids:
                    seen_ids.add(content_dict['id'])
                    unique_similar.append(content_dict)
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
    """Get trending content using advanced algorithms with priority languages"""
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        languages = request.args.getlist('languages')
        categories = request.args.getlist('categories')
        
        # Always include priority languages
        if not languages:
            languages = PRIORITY_LANGUAGES
        else:
            # Ensure priority languages are included
            languages = list(set(PRIORITY_LANGUAGES + languages))
        
        if not categories:
            categories = ['trending_movies', 'trending_tv', 'trending_anime', 
                        'rising_fast', 'popular_regional', 'priority_trending']
        
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
                'priority_trending': results.get('priority_trending', []),
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
            languages = PRIORITY_LANGUAGES
        
        global trending_service
        if not trending_service:
            trending_service = init_services()
        
        results = trending_service.get_trending(
            languages=languages,
            categories=['trending_movies', 'trending_tv', 'trending_anime', 
                       'rising_fast', 'popular_regional', 'priority_trending'],
            limit=20
        )
        
        response = {
            'timestamp': datetime.utcnow().isoformat(),
            'languages': languages,
            'priority_languages': PRIORITY_LANGUAGES,
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
    try:
        language = request.args.get('language')
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_new_releases(limit, language, content_type)
        
        result = []
        for content_dict in recommendations:
            # Now content_dict is already a dictionary
            youtube_url = None
            if content_dict.get('youtube_trailer_id'):
                youtube_url = f"https://www.youtube.com/watch?v={content_dict['youtube_trailer_id']}"
            
            # Parse JSON fields if needed
            genres = content_dict.get('genres', '[]')
            if isinstance(genres, str):
                try:
                    genres = json.loads(genres)
                except:
                    genres = []
            
            result.append({
                'id': content_dict.get('id'),
                'title': content_dict.get('title'),
                'content_type': content_dict.get('content_type'),
                'genres': genres,
                'rating': content_dict.get('rating'),
                'poster_path': f"https://image.tmdb.org/t/p/w300{content_dict.get('poster_path')}" if content_dict.get('poster_path') and not content_dict.get('poster_path', '').startswith('http') else content_dict.get('poster_path'),
                'overview': content_dict.get('overview', '')[:150] + '...' if content_dict.get('overview') else '',
                'release_date': content_dict.get('release_date').isoformat() if content_dict.get('release_date') and hasattr(content_dict.get('release_date'), 'isoformat') else content_dict.get('release_date'),
                'youtube_trailer': youtube_url,
                'is_new_release': content_dict.get('is_new_release', False)
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"New releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

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
            'version': '4.0.0',  # Updated version with priority languages
            'priority_languages': PRIORITY_LANGUAGES
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
                    'OPAA', 'TPRT', 'ADVC', 'MLCPD', 'UTSA',
                    'VSMTA', 'LPOE', 'NITP', 'QIST'  # New algorithms
                ]
                # Test trending service
                test_result = trending_service.get_trending(
                    languages=['telugu'],
                    categories=['priority_trending'],
                    limit=1
                )
                if test_result:
                    health_info['trending_test'] = 'passed'
            else:
                health_info['trending_service'] = 'not_initialized'
                health_info['status'] = 'degraded'
        except Exception as e:
            health_info['trending_service'] = f'error: {str(e)}'
            health_info['status'] = 'degraded'
        
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

# ================== Service and Database Initialization ==================

def init_services():
    """Initialize all services after app context is ready"""
    global trending_service
    try:
        # Pass app instance to trending service
        trending_service = init_advanced_trending_service(db, cache, TMDB_API_KEY, app)
        logger.info(f"Trending service initialized with priority languages: {PRIORITY_LANGUAGES}")
        
        # Background updates are started automatically in init_advanced_trending_service
        logger.info("Background trending updates started")
        
        return trending_service
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return None

def cleanup_services():
    """Cleanup services on shutdown"""
    global trending_service
    if trending_service:
        try:
            trending_service.stop_background_updates()
            logger.info("Stopped trending service background updates")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Register cleanup
atexit.register(cleanup_services)

def init_app():
    """Initialize application, database, and services"""
    with app.app_context():
        try:
            # Create all database tables
            db.create_all()
            logger.info("Database tables created")
            
            # Create default admin user if not exists
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(
                    username='admin',
                    email='admin@example.com',
                    password_hash=generate_password_hash('admin123'),
                    is_admin=True,
                    preferred_languages=json.dumps(PRIORITY_LANGUAGES[:3]),  # Top 3 priority languages
                    location='India'
                )
                db.session.add(admin)
                db.session.commit()
                logger.info("Admin user created with username: admin, password: admin123")
            
            # Initialize trending service
            global trending_service
            trending_service = init_services()
            
            if trending_service:
                logger.info("All services initialized successfully")
                return True
            else:
                logger.warning("Trending service initialization failed - some features may not work")
                return False
            
        except Exception as e:
            logger.error(f"Application initialization failed: {e}")
            return False

# Initialize the application
init_success = init_app()

if not init_success:
    logger.warning("Application initialization incomplete - some features may not work")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)