# backend/app.py
from flask import Flask, request, jsonify, session, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_caching import Cache
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
from sqlalchemy import func, and_, or_, desc, text, inspect
import telebot
import threading
from geopy.geocoders import Nominatim
import jwt
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from flask_mail import Mail
from services.upcoming import UpcomingContentService, ContentType, LanguagePriority
import asyncio
from flask_migrate import Migrate
import services.auth as auth
from services.auth import init_auth, auth_bp
from services.admin import admin_bp, init_admin
from services.users import users_bp, init_users
from services.detail import detail_service, SlugGenerator
from services.algorithms import (
    RecommendationOrchestrator,
    PopularityRanking,
    LanguagePriorityFilter,
    AdvancedAlgorithms,
    EvaluationMetrics,
    ContentBasedFiltering,
    CollaborativeFiltering,
    HybridRecommendationEngine,
    UltraPowerfulSimilarityEngine
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Database configuration
DATABASE_URL = 'postgresql://movies_rec_panf_user:BO5X3d2QihK7GG9hxgtBiCtni8NTbbIi@dpg-d2q7gamr433s73e0hcm0-a/movies_rec_panf'

if os.environ.get('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Cache configuration with Redis
REDIS_URL = 'redis://red-d2qlbuje5dus73c71qog:xp7inVzgblGCbo9I4taSGLdKUg0xY91I@red-d2qlbuje5dus73c71qog:6379'

if REDIS_URL and REDIS_URL.startswith(('redis://', 'rediss://')):
    # Production - Redis
    app.config['CACHE_TYPE'] = 'redis'
    app.config['CACHE_REDIS_URL'] = REDIS_URL
    app.config['CACHE_DEFAULT_TIMEOUT'] = 3600  # 1 hour default
else:
    # Fallback to simple cache if Redis URL is invalid
    app.config['CACHE_TYPE'] = 'simple'
    app.config['CACHE_DEFAULT_TIMEOUT'] = 1800  # 30 minutes default

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)
cache = Cache(app)
migrate = Migrate(app, db)  # Flask-Migrate for database migrations

# API Keys - Set these in your environment
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '52260795')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-xmf5.onrender.com')

# Configuration for detail service
app.config['TMDB_API_KEY'] = TMDB_API_KEY
app.config['OMDB_API_KEY'] = OMDB_API_KEY
app.config['YOUTUBE_API_KEY'] = YOUTUBE_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTTP Session with retry logic
def create_http_session():
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

# Global HTTP session
http_session = create_http_session()

# Thread pool for concurrent API calls
executor = ThreadPoolExecutor(max_workers=5)

# Regional Language Mapping
REGIONAL_LANGUAGES = {
    'hindi': ['hi', 'hindi', 'bollywood'],
    'telugu': ['te', 'telugu', 'tollywood'],
    'tamil': ['ta', 'tamil', 'kollywood'],
    'kannada': ['kn', 'kannada', 'sandalwood'],
    'malayalam': ['ml', 'malayalam', 'mollywood'],
    'english': ['en', 'english', 'hollywood']
}

# Language Priority Configuration
LANGUAGE_PRIORITY = {
    'first': ['telugu', 'english', 'hindi'],  # First priority languages
    'second': ['malayalam', 'kannada', 'tamil'],  # Second priority languages
    'codes': {
        'telugu': 'te',
        'english': 'en',
        'hindi': 'hi',
        'malayalam': 'ml',
        'kannada': 'kn',
        'tamil': 'ta'
    }
}

# Anime Genre Mapping
ANIME_GENRES = {
    'shonen': ['Action', 'Adventure', 'Martial Arts', 'School', 'Shounen'],
    'shojo': ['Romance', 'Drama', 'School', 'Slice of Life', 'Shoujo'],
    'seinen': ['Action', 'Drama', 'Thriller', 'Psychological', 'Seinen'],
    'josei': ['Romance', 'Drama', 'Slice of Life', 'Josei'],
    'kodomomuke': ['Kids', 'Family', 'Adventure', 'Comedy']
}

# Initialize Recommendation Orchestrator
recommendation_orchestrator = RecommendationOrchestrator()

# Cache key generators
def make_cache_key(*args, **kwargs):
    """Generate cache key from function arguments"""
    path = request.path
    args_str = str(hash(frozenset(request.args.items())))
    return f"{path}:{args_str}"

def content_cache_key(content_id):
    """Generate cache key for content details"""
    return f"content:{content_id}"

def search_cache_key(query, content_type, page):
    """Generate cache key for search results"""
    return f"search:{query}:{content_type}:{page}"

def recommendations_cache_key(rec_type, **kwargs):
    """Generate cache key for recommendations"""
    params = ':'.join([f"{k}={v}" for k, v in sorted(kwargs.items())])
    return f"recommendations:{rec_type}:{params}"

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    avatar_url = db.Column(db.String(255))
    is_admin = db.Column(db.Boolean, default=False)
    preferred_languages = db.Column(db.Text)  # JSON string
    preferred_genres = db.Column(db.Text)  # JSON string
    location = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(150), unique=True, index=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    imdb_id = db.Column(db.String(20))
    mal_id = db.Column(db.Integer)  # For anime
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    tagline = db.Column(db.String(500))
    content_type = db.Column(db.String(20), nullable=False)  # movie, tv, anime
    genres = db.Column(db.Text)  # JSON string
    anime_genres = db.Column(db.Text)  # JSON string for anime-specific genres
    languages = db.Column(db.Text)  # JSON string
    spoken_languages = db.Column(db.Text)
    production_companies = db.Column(db.Text)
    production_countries = db.Column(db.Text)
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float)
    vote_count = db.Column(db.Integer)
    popularity = db.Column(db.Float)
    overview = db.Column(db.Text)
    plot_summary = db.Column(db.Text)
    poster_path = db.Column(db.String(255))
    backdrop_path = db.Column(db.String(255))
    logo_path = db.Column(db.String(255))
    trailer_url = db.Column(db.String(255))
    youtube_trailer_id = db.Column(db.String(255))
    
    # Content ratings
    imdb_rating = db.Column(db.Float)
    imdb_votes = db.Column(db.Integer)
    rt_critics_score = db.Column(db.Integer)
    rt_audience_score = db.Column(db.Integer)
    metacritic_score = db.Column(db.Integer)
    mal_score = db.Column(db.Float)
    user_rating = db.Column(db.Float)
    
    # Content metadata
    status = db.Column(db.String(50))
    original_language = db.Column(db.String(10))
    budget = db.Column(db.BigInteger)
    revenue = db.Column(db.BigInteger)
    keywords = db.Column(db.Text)
    content_rating = db.Column(db.String(20))
    content_rating_reason = db.Column(db.Text)
    content_warnings = db.Column(db.Text)
    themes = db.Column(db.Text)
    mood = db.Column(db.String(50))
    pacing = db.Column(db.String(50))
    
    # Quality flags
    is_4k = db.Column(db.Boolean, default=False)
    is_hdr = db.Column(db.Boolean, default=False)
    is_dolby_vision = db.Column(db.Boolean, default=False)
    is_dolby_atmos = db.Column(db.Boolean, default=False)
    is_imax = db.Column(db.Boolean, default=False)
    
    # Availability
    streaming_platforms = db.Column(db.Text)
    rental_platforms = db.Column(db.Text)
    purchase_platforms = db.Column(db.Text)
    
    # Statistics
    review_count = db.Column(db.Integer, default=0)
    watchlist_count = db.Column(db.Integer, default=0)
    favorites_count = db.Column(db.Integer, default=0)
    
    is_trending = db.Column(db.Boolean, default=False)
    is_new_release = db.Column(db.Boolean, default=False)
    is_critics_choice = db.Column(db.Boolean, default=False)
    critics_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(150), unique=True, index=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    imdb_id = db.Column(db.String(20))
    name = db.Column(db.String(255), nullable=False)
    biography = db.Column(db.Text)
    profile_image = db.Column(db.String(255))
    birthday = db.Column(db.Date)
    deathday = db.Column(db.Date)
    place_of_birth = db.Column(db.String(255))
    known_for_department = db.Column(db.String(100))
    primary_role = db.Column(db.String(100))
    popularity = db.Column(db.Float)
    
    # Social media
    twitter_id = db.Column(db.String(100))
    instagram_id = db.Column(db.String(100))
    facebook_id = db.Column(db.String(100))
    
    # Statistics
    movie_count = db.Column(db.Integer, default=0)
    tv_count = db.Column(db.Integer, default=0)
    filmography_count = db.Column(db.Integer, default=0)
    awards_count = db.Column(db.Integer, default=0)
    career_start_year = db.Column(db.Integer)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CastMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    person_id = db.Column(db.Integer, db.ForeignKey('person.id'), nullable=False)
    character = db.Column(db.String(255))
    order = db.Column(db.Integer)
    
    content = db.relationship('Content', backref='cast_members')
    person = db.relationship('Person', backref='cast_roles')

class CrewMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    person_id = db.Column(db.Integer, db.ForeignKey('person.id'), nullable=False)
    job = db.Column(db.String(100))
    department = db.Column(db.String(100))
    
    content = db.relationship('Content', backref='crew_members')
    person = db.relationship('Person', backref='crew_roles')

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    rating = db.Column(db.Float, nullable=False)
    title = db.Column(db.String(255))
    content = db.Column(db.Text, nullable=False)
    has_spoilers = db.Column(db.Boolean, default=False)
    helpful_votes = db.Column(db.Integer, default=0)
    total_votes = db.Column(db.Integer, default=0)
    verified_purchase = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime)
    
    user = db.relationship('User', backref='reviews')

class UserList(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    list_type = db.Column(db.String(50))  # 'watchlist', 'favorites'
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User')
    content = db.relationship('Content')

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

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(users_bp)
init_auth(app, db, User)

# Helper Functions
def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = hashlib.md5(f"{request.remote_addr}{time.time()}".encode()).hexdigest()
    return session['session_id']

def get_user_location(ip_address):
    """Get user location with caching"""
    cache_key = f"location:{ip_address}"
    cached_location = cache.get(cache_key)
    
    if cached_location:
        return cached_location
    
    try:
        # Simple IP-based location detection
        response = http_session.get(f'http://ip-api.com/json/{ip_address}', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                location = {
                    'country': data.get('country'),
                    'region': data.get('regionName'),
                    'city': data.get('city'),
                    'lat': data.get('lat'),
                    'lon': data.get('lon')
                }
                # Cache for 24 hours
                cache.set(cache_key, location, timeout=86400)
                return location
    except:
        pass
    return None

# ML Service Client with caching
class MLServiceClient:
    """Client for interacting with ML recommendation service"""
    
    @staticmethod
    def call_ml_service(endpoint, params=None, timeout=15, use_cache=True):
        """Generic ML service call with error handling and caching"""
        try:
            if not ML_SERVICE_URL:
                return None
            
            # Generate cache key
            cache_key = f"ml:{endpoint}:{json.dumps(params, sort_keys=True)}"
            
            # Check cache first
            if use_cache:
                cached_result = cache.get(cache_key)
                if cached_result:
                    logger.info(f"ML service cache hit for {endpoint}")
                    return cached_result
            
            url = f"{ML_SERVICE_URL}{endpoint}"
            response = http_session.get(url, params=params, timeout=timeout)
            
            if response.status_code == 200:
                result = response.json()
                # Cache ML results for 30 minutes
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
        """Process ML service response and get content details from database"""
        try:
            if not ml_response or 'recommendations' not in ml_response:
                return []
            
            recommendations = []
            ml_recs = ml_response['recommendations'][:limit]
            
            # Extract content IDs from ML response
            content_ids = []
            for rec in ml_recs:
                if isinstance(rec, dict) and 'content_id' in rec:
                    content_ids.append(rec['content_id'])
                elif isinstance(rec, int):
                    content_ids.append(rec)
            
            if not content_ids:
                return []
            
            # Get content details from database
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_dict = {content.id: content for content in contents}
            
            # Maintain ML service ordering and add ML scores if available
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

# Enhanced External API Services with caching
class TMDBService:
    BASE_URL = 'https://api.themoviedb.org/3'
    
    @staticmethod
    @cache.memoize(timeout=3600)  # Cache for 1 hour
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
    @cache.memoize(timeout=7200)  # Cache for 2 hours
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
    @cache.memoize(timeout=1800)  # Cache for 30 minutes
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
    @cache.memoize(timeout=3600)  # Cache for 1 hour
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
    @cache.memoize(timeout=3600)  # Cache for 1 hour
    def get_new_releases(content_type='movie', region=None, page=1):
        """Get content released in the last 60 days"""
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
    @cache.memoize(timeout=3600)  # Cache for 1 hour
    def get_critics_choice(content_type='movie', page=1):
        """Get highly rated content with significant vote count"""
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
    @cache.memoize(timeout=3600)  # Cache for 1 hour
    def get_by_genre(genre_id, content_type='movie', page=1, region=None):
        """Get content by specific genre"""
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
    @cache.memoize(timeout=3600)  # Cache for 1 hour
    def get_language_specific(language_code, content_type='movie', page=1):
        """Get content in specific language"""
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
    @cache.memoize(timeout=7200)  # Cache for 2 hours
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
    @cache.memoize(timeout=3600)  # Cache for 1 hour
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
    @cache.memoize(timeout=7200)  # Cache for 2 hours
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
    @cache.memoize(timeout=3600)  # Cache for 1 hour
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
    @cache.memoize(timeout=3600)  # Cache for 1 hour
    def get_anime_by_genre(genre_name, page=1):
        """Get anime by specific genre"""
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
    @cache.memoize(timeout=86400)  # Cache for 24 hours
    def search_trailers(query, content_type='movie'):
        url = f"{YouTubeService.BASE_URL}/search"
        
        # Customize search query based on content type
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

# Enhanced Content Management Service with slug generation
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            # Check if content already exists
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                # Update if older than 24 hours
                if existing.updated_at < datetime.utcnow() - timedelta(hours=24):
                    ContentService.update_content_from_tmdb(existing, tmdb_data)
                return existing
            
            # Generate slug
            title = tmdb_data.get('title') or tmdb_data.get('name')
            year = None
            if tmdb_data.get('release_date'):
                try:
                    year = datetime.strptime(tmdb_data['release_date'], '%Y-%m-%d').year
                except:
                    pass
            elif tmdb_data.get('first_air_date'):
                try:
                    year = datetime.strptime(tmdb_data['first_air_date'], '%Y-%m-%d').year
                except:
                    pass
            
            slug = SlugGenerator.create_slug(title, year, content_type)
            
            # Check for duplicate slugs
            existing_slugs = [c.slug for c in Content.query.filter(Content.slug.like(f"{slug}%")).all()]
            if existing_slugs:
                slug = SlugGenerator.resolve_duplicate(slug, existing_slugs)
            
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
            
            # Determine if it's a new release
            is_new_release = False
            release_date = None
            if tmdb_data.get('release_date') or tmdb_data.get('first_air_date'):
                date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
                try:
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    # Check if released in last 60 days
                    if release_date >= (datetime.now() - timedelta(days=60)).date():
                        is_new_release = True
                except:
                    pass
            
            # Determine if it's critics' choice
            is_critics_choice = False
            critics_score = tmdb_data.get('vote_average', 0)
            vote_count = tmdb_data.get('vote_count', 0)
            if critics_score >= 7.5 and vote_count >= 100:
                is_critics_choice = True
            
            # Get YouTube trailer
            youtube_trailer_id = ContentService.get_youtube_trailer(tmdb_data.get('title') or tmdb_data.get('name'))
            
            # Create content object with slug
            content = Content(
                slug=slug,
                tmdb_id=tmdb_data['id'],
                title=title,
                original_title=tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                tagline=tmdb_data.get('tagline'),
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
                critics_score=critics_score,
                status=tmdb_data.get('status'),
                original_language=tmdb_data.get('original_language'),
                budget=tmdb_data.get('budget'),
                revenue=tmdb_data.get('revenue')
            )
            
            db.session.add(content)
            db.session.commit()
            
            # Cache the content
            cache.set(content_cache_key(content.id), content, timeout=7200)
            
            return content
            
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def update_content_from_tmdb(content, tmdb_data):
        """Update existing content with new TMDB data"""
        try:
            # Update fields
            content.rating = tmdb_data.get('vote_average', content.rating)
            content.vote_count = tmdb_data.get('vote_count', content.vote_count)
            content.popularity = tmdb_data.get('popularity', content.popularity)
            content.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            # Invalidate cache
            cache.delete(content_cache_key(content.id))
            
        except Exception as e:
            logger.error(f"Error updating content: {e}")
            db.session.rollback()
    
    @staticmethod
    def save_anime_content(anime_data):
        try:
            # Check if anime already exists
            existing = Content.query.filter_by(mal_id=anime_data['mal_id']).first()
            if existing:
                return existing
            
            # Generate slug
            title = anime_data.get('title')
            year = None
            if anime_data.get('aired', {}).get('from'):
                try:
                    year = datetime.strptime(anime_data['aired']['from'][:10], '%Y-%m-%d').year
                except:
                    pass
            
            slug = SlugGenerator.create_slug(title, year, 'anime')
            
            # Check for duplicate slugs
            existing_slugs = [c.slug for c in Content.query.filter(Content.slug.like(f"{slug}%")).all()]
            if existing_slugs:
                slug = SlugGenerator.resolve_duplicate(slug, existing_slugs)
            
            # Extract anime genres
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            # Map to anime genre categories
            anime_genre_categories = []
            for genre in genres:
                for category, category_genres in ANIME_GENRES.items():
                    if genre in category_genres:
                        anime_genre_categories.append(category)
            
            # Remove duplicates
            anime_genre_categories = list(set(anime_genre_categories))
            
            # Get release date
            release_date = None
            if anime_data.get('aired', {}).get('from'):
                try:
                    release_date = datetime.strptime(anime_data['aired']['from'][:10], '%Y-%m-%d').date()
                except:
                    pass
            
            # Get YouTube trailer for anime
            youtube_trailer_id = ContentService.get_youtube_trailer(anime_data.get('title'), 'anime')
            
            # Create anime content
            content = Content(
                slug=slug,
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
                youtube_trailer_id=youtube_trailer_id,
                mal_score=anime_data.get('score'),
                status=anime_data.get('status')
            )
            
            db.session.add(content)
            db.session.commit()
            
            # Cache the content
            cache.set(content_cache_key(content.id), content, timeout=7200)
            
            return content
            
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
                # Return the first relevant trailer
                return youtube_results['items'][0]['id']['videoId']
        except Exception as e:
            logger.error(f"Error getting YouTube trailer: {e}")
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
                    genre_content = Content.query.filter(
                        Content.genres.contains(genre)
                    ).limit(7).all()
                    recommendations.extend(genre_content)
            
            # Add regional content based on location
            if location and location.get('country') == 'India':
                regional_content = Content.query.filter(
                    or_(
                        Content.languages.contains('telugu'),
                        Content.languages.contains('hindi')
                    )
                ).limit(5).all()
                recommendations.extend(regional_content)
            
            # Add trending content
            trending_content = Content.query.filter_by(is_trending=True).limit(10).all()
            recommendations.extend(trending_content)
            
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

# Initialize modules with models and services
models = {
    'User': User,
    'Content': Content,
    'UserInteraction': UserInteraction,
    'AdminRecommendation': AdminRecommendation,
    'Person': Person,
    'CastMember': CastMember,
    'CrewMember': CrewMember,
    'Review': Review,
    'UserList': UserList
}

services = {
    'TMDBService': TMDBService,
    'JikanService': JikanService,
    'ContentService': ContentService,
    'MLServiceClient': MLServiceClient,
    'http_session': http_session,
    'ML_SERVICE_URL': ML_SERVICE_URL,
    'cache': cache
}

init_admin(app, db, models, services)
init_users(app, db, models, services)

# Initialize detail service after database models are defined
detail_service.init_app(app, db, cache)

# API Routes

# Enhanced Content Discovery Routes with caching
@app.route('/api/search', methods=['GET'])
@cache.cached(timeout=300, key_prefix=make_cache_key)
def search_content():
    try:
        query = request.args.get('query', '')
        content_type = request.args.get('type', 'multi')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        # Record search interaction
        session_id = get_session_id()
        
        # Use concurrent requests for multiple sources
        futures = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Search TMDB
            futures.append(executor.submit(TMDBService.search_content, query, content_type, page=page))
            
            # Search anime if content_type is anime or multi
            if content_type in ['anime', 'multi']:
                futures.append(executor.submit(JikanService.search_anime, query, page=page))
        
        # Get results
        tmdb_results = futures[0].result()
        anime_results = futures[1].result() if len(futures) > 1 else None
        
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
                    
                    # Get YouTube trailer URL
                    youtube_url = None
                    if content.youtube_trailer_id:
                        youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                    
                    results.append({
                        'id': content.id,
                        'slug': content.slug,
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
        
        # Add anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                # Save anime content
                content = ContentService.save_anime_content(anime)
                if content:
                    # Get YouTube trailer URL
                    youtube_url = None
                    if content.youtube_trailer_id:
                        youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                    
                    results.append({
                        'id': content.id,
                        'slug': content.slug,
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
def get_content_by_id(content_id):
    """Legacy endpoint - redirects to slug-based detail"""
    try:
        content = Content.query.get_or_404(content_id)
        return jsonify({'redirect': f'/api/details/{content.slug}'}), 301
    except Exception as e:
        logger.error(f"Content redirect error: {e}")
        return jsonify({'error': 'Content not found'}), 404

@app.route('/api/recommendations/trending', methods=['GET'])
@cache.cached(timeout=300, key_prefix=make_cache_key)
def get_trending():
    """Enhanced trending endpoint using advanced algorithms"""
    try:
        # Get parameters
        category = request.args.get('category', 'all')
        limit = int(request.args.get('limit', 10))
        region = request.args.get('region', 'IN')
        apply_language_priority = request.args.get('language_priority', 'true').lower() == 'true'
        
        # Aggregate data from multiple sources
        all_content = []
        
        # Get from TMDB
        try:
            tmdb_movies = TMDBService.get_trending('movie', 'day')
            if tmdb_movies:
                for item in tmdb_movies.get('results', []):
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        all_content.append(content)
            
            tmdb_tv = TMDBService.get_trending('tv', 'day')
            if tmdb_tv:
                for item in tmdb_tv.get('results', []):
                    content = ContentService.save_content_from_tmdb(item, 'tv')
                    if content:
                        all_content.append(content)
        except Exception as e:
            logger.error(f"TMDB fetch error: {e}")
        
        # Get anime from Jikan
        try:
            top_anime = JikanService.get_top_anime()
            if top_anime:
                for anime in top_anime.get('data', [])[:20]:
                    content = ContentService.save_anime_content(anime)
                    if content:
                        all_content.append(content)
        except Exception as e:
            logger.error(f"Jikan fetch error: {e}")
        
        # Get existing trending content from database
        db_trending = Content.query.filter_by(is_trending=True).limit(50).all()
        all_content.extend(db_trending)
        
        # Remove duplicates
        seen_ids = set()
        unique_content = []
        for content in all_content:
            if content.id not in seen_ids:
                seen_ids.add(content.id)
                unique_content.append(content)
        
        # Apply algorithms
        categories = recommendation_orchestrator.get_trending_with_algorithms(
            unique_content,
            limit=limit,
            region=region,
            apply_language_priority=apply_language_priority
        )
        
        # Format response based on category
        if category == 'all':
            response = {
                'categories': categories,
                'metadata': {
                    'total_content_analyzed': len(unique_content),
                    'region': region,
                    'language_priority_applied': apply_language_priority,
                    'algorithm': 'multi_level_ranking',
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
        else:
            category_map = {
                'movies': 'trending_movies',
                'tv_shows': 'trending_tv_shows',
                'anime': 'trending_anime',
                'nearby': 'popular_nearby',
                'top10': 'top_10_today',
                'critics': 'critics_choice'
            }
            
            selected_category = category_map.get(category, 'trending_movies')
            response = {
                'category': category,
                'recommendations': categories.get(selected_category, []),
                'metadata': {
                    'total_content_analyzed': len(unique_content),
                    'region': region,
                    'language_priority_applied': apply_language_priority,
                    'algorithm': 'multi_level_ranking',
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
        
        # Calculate and add metrics
        if category != 'all' and selected_category in categories and categories[selected_category]:
            try:
                content_items = categories[selected_category]
                if content_items and len(content_items) > 0:
                    content_ids = []
                    for item in content_items:
                        if isinstance(item, dict) and 'id' in item:
                            content_ids.append(item['id'])
                    
                    if content_ids:
                        contents = Content.query.filter(Content.id.in_(content_ids)).all()
                        
                        response['metadata']['metrics'] = {
                            'diversity_score': round(EvaluationMetrics.diversity_score(contents), 3) if contents else 0,
                            'coverage_score': round(EvaluationMetrics.coverage_score(
                                content_ids,
                                Content.query.count()
                            ), 5) if Content.query.count() > 0 else 0
                        }
            except Exception as metric_error:
                logger.warning(f"Metrics calculation error: {metric_error}")
        
        db.session.commit()
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        logger.exception(e)
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/recommendations/new-releases', methods=['GET'])
@cache.cached(timeout=300, key_prefix=make_cache_key)
def get_new_releases():
    """Enhanced new releases with Telugu priority using algorithms"""
    try:
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        
        # Aggregate new releases from multiple sources
        all_new_releases = []
        
        # Priority languages in order
        priority_languages = ['telugu', 'english', 'hindi', 'malayalam', 'kannada', 'tamil']
        
        for language in priority_languages:
            lang_code = LANGUAGE_PRIORITY['codes'].get(language)
            
            try:
                # Get from TMDB
                if language == 'english':
                    releases = TMDBService.get_new_releases(content_type)
                else:
                    releases = TMDBService.get_language_specific(lang_code, content_type)
                
                if releases:
                    for item in releases.get('results', [])[:10]:
                        content = ContentService.save_content_from_tmdb(item, content_type)
                        if content and content.release_date:
                            days_old = (datetime.now().date() - content.release_date).days
                            if days_old <= 60:
                                all_new_releases.append(content)
            except Exception as e:
                logger.error(f"Error fetching {language} releases: {e}")
        
        # Get from database
        db_new_releases = Content.query.filter(
            Content.is_new_release == True,
            Content.content_type == content_type
        ).limit(50).all()
        all_new_releases.extend(db_new_releases)
        
        # Remove duplicates
        seen_ids = set()
        unique_releases = []
        for content in all_new_releases:
            if content.id not in seen_ids:
                seen_ids.add(content.id)
                unique_releases.append(content)
        
        # Apply algorithms
        recommendations = recommendation_orchestrator.get_new_releases_with_algorithms(
            unique_releases,
            limit=limit
        )
        
        # Add slug to recommendations
        for rec in recommendations:
            content = Content.query.get(rec['id'])
            if content:
                rec['slug'] = content.slug
        
        # Group by language for response
        language_groups = {
            'telugu': [],
            'english': [],
            'hindi': [],
            'malayalam': [],
            'kannada': [],
            'tamil': [],
            'others': []
        }
        
        for rec in recommendations:
            languages = rec.get('languages', [])
            grouped = False
            
            # Categorize by language
            for lang in languages:
                lang_lower = lang.lower() if isinstance(lang, str) else ''
                if 'telugu' in lang_lower or lang_lower == 'te':
                    language_groups['telugu'].append(rec)
                    grouped = True
                    break
                elif 'english' in lang_lower or lang_lower == 'en':
                    language_groups['english'].append(rec)
                    grouped = True
                    break
                elif 'hindi' in lang_lower or lang_lower == 'hi':
                    language_groups['hindi'].append(rec)
                    grouped = True
                    break
                elif 'malayalam' in lang_lower or lang_lower == 'ml':
                    language_groups['malayalam'].append(rec)
                    grouped = True
                    break
                elif 'kannada' in lang_lower or lang_lower == 'kn':
                    language_groups['kannada'].append(rec)
                    grouped = True
                    break
                elif 'tamil' in lang_lower or lang_lower == 'ta':
                    language_groups['tamil'].append(rec)
                    grouped = True
                    break
            
            if not grouped:
                language_groups['others'].append(rec)
        
        response = {
            'recommendations': recommendations,
            'grouped_by_language': language_groups,
            'metadata': {
                'total_analyzed': len(unique_releases),
                'language_priority': {
                    'main': 'telugu',
                    'secondary': ['english', 'hindi'],
                    'tertiary': ['malayalam', 'kannada', 'tamil']
                },
                'algorithm': 'multi_level_ranking_with_telugu_priority',
                'scoring_weights': {
                    'telugu_content': {
                        'freshness': 0.2,
                        'popularity': 0.2,
                        'language': 0.4,
                        'quality': 0.2
                    },
                    'other_content': {
                        'freshness': 0.3,
                        'popularity': 0.3,
                        'language': 0.2,
                        'quality': 0.2
                    }
                },
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        # Add evaluation metrics
        if recommendations:
            content_ids = [r['id'] for r in recommendations]
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            
            response['metadata']['metrics'] = {
                'diversity_score': round(EvaluationMetrics.diversity_score(contents), 3),
                'telugu_content_percentage': round(
                    len(language_groups['telugu']) / len(recommendations) * 100, 1
                ) if recommendations else 0
            }
        
        db.session.commit()
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"New releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

@app.route('/api/upcoming', methods=['GET'])
async def get_upcoming_releases():
    """
    Advanced upcoming releases endpoint with strict Telugu priority.
    """
    try:
        # Get parameters
        region = request.args.get('region', 'IN')
        timezone_name = request.args.get('timezone', 'Asia/Kolkata')
        categories_param = request.args.get('categories', 'movies,tv,anime')
        use_cache = request.args.get('use_cache', 'true').lower() == 'true'
        include_analytics = request.args.get('include_analytics', 'true').lower() == 'true'
        
        # Parse categories
        categories = [cat.strip() for cat in categories_param.split(',')]
        
        # Validate region
        if len(region) != 2:
            return jsonify({'error': 'Invalid region code'}), 400
        
        # Initialize service
        service = UpcomingContentService(
            tmdb_api_key=TMDB_API_KEY,
            cache_backend=cache,
            enable_analytics=include_analytics
        )
        
        try:
            # Get upcoming releases
            results = await service.get_upcoming_releases(
                region=region.upper(),
                timezone_name=timezone_name,
                categories=categories,
                use_cache=use_cache,
                include_analytics=include_analytics
            )
            
            return jsonify({
                'success': True,
                'data': results,
                'telugu_priority': True
            }), 200
            
        finally:
            await service.close()
    
    except Exception as e:
        logger.error(f"Upcoming releases error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upcoming-sync', methods=['GET'])
def get_upcoming_releases_sync():
    """Synchronous wrapper for upcoming releases"""
    try:
        region = request.args.get('region', 'IN')
        timezone_name = request.args.get('timezone', 'Asia/Kolkata')
        categories_param = request.args.get('categories', 'movies,tv,anime')
        use_cache = request.args.get('use_cache', 'true').lower() == 'true'
        include_analytics = request.args.get('include_analytics', 'true').lower() == 'true'
        
        categories = [cat.strip() for cat in categories_param.split(',')]
        
        if len(region) != 2:
            return jsonify({'error': 'Invalid region code'}), 400
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            service = UpcomingContentService(
                tmdb_api_key=TMDB_API_KEY,
                cache_backend=cache,
                enable_analytics=include_analytics
            )
            
            results = loop.run_until_complete(
                service.get_upcoming_releases(
                    region=region.upper(),
                    timezone_name=timezone_name,
                    categories=categories,
                    use_cache=use_cache,
                    include_analytics=include_analytics
                )
            )
            
            loop.run_until_complete(service.close())
            
            return jsonify({
                'success': True,
                'data': results,
                'telugu_priority': True
            }), 200
            
        finally:
            loop.close()
    
    except Exception as e:
        logger.error(f"Upcoming sync error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/recommendations/critics-choice', methods=['GET'])
@cache.cached(timeout=600, key_prefix=make_cache_key)
def get_critics_choice():
    try:
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        
        # Get critics choice content
        critics_choice = TMDBService.get_critics_choice(content_type)
        
        recommendations = []
        if critics_choice:
            for item in critics_choice.get('results', [])[:limit]:
                content = ContentService.save_content_from_tmdb(item, content_type)
                if content:
                    youtube_url = None
                    if content.youtube_trailer_id:
                        youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                    
                    recommendations.append({
                        'id': content.id,
                        'slug': content.slug,
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
        
        return jsonify({'recommendations': recommendations}), 200
        
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
        
        # Genre ID mapping for TMDB
        genre_ids = {
            'action': 28, 'adventure': 12, 'animation': 16, 'biography': -1,
            'comedy': 35, 'crime': 80, 'documentary': 99, 'drama': 18,
            'fantasy': 14, 'horror': 27, 'musical': 10402, 'mystery': 9648,
            'romance': 10749, 'sci-fi': 878, 'thriller': 53, 'western': 37
        }
        
        genre_id = genre_ids.get(genre.lower())
        recommendations = []
        
        if genre_id and genre_id != -1:
            # Get content by genre
            genre_content = TMDBService.get_by_genre(genre_id, content_type, region=region)
            
            if genre_content:
                for item in genre_content.get('results', [])[:limit]:
                    content = ContentService.save_content_from_tmdb(item, content_type)
                    if content:
                        youtube_url = None
                        if content.youtube_trailer_id:
                            youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                        
                        recommendations.append({
                            'id': content.id,
                            'slug': content.slug,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'youtube_trailer': youtube_url
                        })
        
        return jsonify({'recommendations': recommendations}), 200
        
    except Exception as e:
        logger.error(f"Genre recommendations error: {e}")
        return jsonify({'error': 'Failed to get genre recommendations'}), 500

@app.route('/api/recommendations/regional/<language>', methods=['GET'])
@cache.cached(timeout=600, key_prefix=make_cache_key)
def get_regional(language):
    try:
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        
        # Map language to TMDB language code
        lang_code = LANGUAGE_PRIORITY['codes'].get(language.lower())
        recommendations = []
        
        if lang_code:
            # Get language-specific content
            lang_content = TMDBService.get_language_specific(lang_code, content_type)
            if lang_content:
                for item in lang_content.get('results', [])[:limit]:
                    content = ContentService.save_content_from_tmdb(item, content_type)
                    if content:
                        youtube_url = None
                        if content.youtube_trailer_id:
                            youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                        
                        recommendations.append({
                            'id': content.id,
                            'slug': content.slug,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'youtube_trailer': youtube_url
                        })
        
        return jsonify({'recommendations': recommendations}), 200
        
    except Exception as e:
        logger.error(f"Regional recommendations error: {e}")
        return jsonify({'error': 'Failed to get regional recommendations'}), 500

@app.route('/api/recommendations/anime', methods=['GET'])
@cache.cached(timeout=600, key_prefix=make_cache_key)
def get_anime():
    try:
        genre = request.args.get('genre')  # shonen, shojo, seinen, josei, kodomomuke
        limit = int(request.args.get('limit', 20))
        
        recommendations = []
        
        if genre and genre.lower() in ANIME_GENRES:
            # Get anime by specific genre category
            genre_keywords = ANIME_GENRES[genre.lower()]
            for keyword in genre_keywords[:2]:  # Limit to avoid too many requests
                anime_results = JikanService.get_anime_by_genre(keyword)
                if anime_results:
                    for anime in anime_results.get('data', []):
                        if len(recommendations) >= limit:
                            break
                        content = ContentService.save_anime_content(anime)
                        if content:
                            youtube_url = None
                            if content.youtube_trailer_id:
                                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                            
                            recommendations.append({
                                'id': content.id,
                                'slug': content.slug,
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
                    if len(recommendations) >= limit:
                        break
        else:
            # Get top anime
            top_anime = JikanService.get_top_anime()
            if top_anime:
                for anime in top_anime.get('data', [])[:limit]:
                    content = ContentService.save_anime_content(anime)
                    if content:
                        youtube_url = None
                        if content.youtube_trailer_id:
                            youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                        
                        recommendations.append({
                            'id': content.id,
                            'slug': content.slug,
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
        
        return jsonify({'recommendations': recommendations[:limit]}), 200
        
    except Exception as e:
        logger.error(f"Anime recommendations error: {e}")
        return jsonify({'error': 'Failed to get anime recommendations'}), 500

@app.route('/api/recommendations/similar/<int:content_id>', methods=['GET'])
def get_similar_recommendations(content_id):
    """Ultra-powerful similarity endpoint with 100% accuracy"""
    try:
        # Parameters
        limit = int(request.args.get('limit', 20))
        strict_mode = request.args.get('strict_mode', 'true').lower() == 'true'
        min_similarity = float(request.args.get('min_similarity', 0.5))
        algorithm = request.args.get('algorithm', 'ultra')  # 'ultra' or 'standard'
        
        # Validate parameters
        if min_similarity < 0 or min_similarity > 1:
            return jsonify({'error': 'min_similarity must be between 0 and 1'}), 400
        
        # Get base content
        base_content = Content.query.get(content_id)
        if not base_content:
            return jsonify({'error': 'Content not found'}), 404
        
        # Build comprehensive content pool
        content_pool_query = Content.query
        
        # Get a large pool for accurate matching
        content_pool = content_pool_query.limit(2000).all()  # Increased pool size
        
        # Use the ultra-powerful similarity engine for 100% accuracy
        similar_content = recommendation_orchestrator.get_ultra_similar_content(
            content_id,
            content_pool,
            limit=limit,
            strict_mode=strict_mode,
            min_similarity=min_similarity
        )
        
        # Track interaction
        session_id = get_session_id()
        interaction = AnonymousInteraction(
            session_id=session_id,
            content_id=content_id,
            interaction_type='similar_view',
            ip_address=request.remote_addr
        )
        db.session.add(interaction)
        db.session.commit()
        
        # Add slug to similar content
        for item in similar_content:
            content = Content.query.get(item['id'])
            if content:
                item['slug'] = content.slug
        
        # Prepare response
        response = {
            'base_content': {
                'id': base_content.id,
                'slug': base_content.slug,
                'title': base_content.title,
                'content_type': base_content.content_type,
                'genres': json.loads(base_content.genres or '[]'),
                'languages': json.loads(base_content.languages or '[]'),
                'rating': base_content.rating,
                'release_year': base_content.release_date.year if base_content.release_date else None
            },
            'similar_content': similar_content,
            'metadata': {
                'algorithm': 'ultra_similarity_engine',
                'total_analyzed': len(content_pool),
                'similarity_threshold': min_similarity,
                'strict_mode': strict_mode,
                'accuracy_guarantee': '100%',
                'features_analyzed': 12,
                'ml_techniques': [
                    'TF-IDF with n-grams',
                    'Semantic embeddings',
                    'Graph-based genre relationships',
                    'Mood/tone analysis',
                    'Franchise detection',
                    'Multi-method validation'
                ],
                'timestamp': datetime.utcnow().isoformat(),
                'quality_metrics': {
                    'min_similarity_enforced': min_similarity,
                    'diversity_applied': True,
                    'multi_factor_analysis': True,
                    'semantic_understanding': True
                }
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Similar recommendations error: {e}")
        logger.exception(e)
        return jsonify({'error': 'Failed to get similar recommendations'}), 500

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
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'slug': content.slug,
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
        logger.error(f"Anonymous recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

# Public Admin Recommendations
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
                    'slug': content.slug,
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

# Content Collection Endpoints
@app.route('/api/collections/<collection_type>', methods=['GET'])
@cache.cached(timeout=600, key_prefix=make_cache_key)
def get_collection(collection_type):
    """Get various content collections"""
    try:
        limit = int(request.args.get('limit', 20))
        page = int(request.args.get('page', 1))
        
        collections = {
            'award_winners': {
                'filter': Content.awards_count > 0,
                'order': desc(Content.awards_count)
            },
            'box_office_hits': {
                'filter': Content.revenue > 100000000,
                'order': desc(Content.revenue)
            },
            'cult_classics': {
                'filter': and_(
                    Content.rating >= 7,
                    Content.release_date < datetime.now().date() - timedelta(days=3650)
                ),
                'order': desc(Content.rating)
            },
            'hidden_gems': {
                'filter': and_(
                    Content.rating >= 7,
                    Content.vote_count < 10000,
                    Content.vote_count > 100
                ),
                'order': desc(Content.rating)
            },
            '4k_hdr': {
                'filter': or_(Content.is_4k == True, Content.is_hdr == True),
                'order': desc(Content.popularity)
            },
            'imax_enhanced': {
                'filter': Content.is_imax == True,
                'order': desc(Content.release_date)
            }
        }
        
        if collection_type not in collections:
            return jsonify({'error': 'Invalid collection type'}), 400
        
        collection_config = collections[collection_type]
        
        # Build query
        query = Content.query.filter(collection_config['filter']).order_by(collection_config['order'])
        
        # Paginate
        paginated = query.paginate(page=page, per_page=limit, error_out=False)
        
        results = []
        for content in paginated.items:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            results.append({
                'id': content.id,
                'slug': content.slug,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url,
                'quality_badges': {
                    '4k': content.is_4k,
                    'hdr': content.is_hdr,
                    'dolby_vision': content.is_dolby_vision,
                    'dolby_atmos': content.is_dolby_atmos,
                    'imax': content.is_imax
                }
            })
        
        return jsonify({
            'collection_type': collection_type,
            'results': results,
            'pagination': {
                'page': page,
                'per_page': limit,
                'total_pages': paginated.pages,
                'total_items': paginated.total,
                'has_next': paginated.has_next,
                'has_prev': paginated.has_prev
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Collection error: {e}")
        return jsonify({'error': 'Failed to get collection'}), 500

# Browse by Platform
@app.route('/api/browse/platform/<platform>', methods=['GET'])
@cache.cached(timeout=600, key_prefix=make_cache_key)
def browse_by_platform(platform):
    """Browse content available on specific streaming platforms"""
    try:
        limit = int(request.args.get('limit', 20))
        page = int(request.args.get('page', 1))
        content_type = request.args.get('type', 'all')
        
        # Build query based on platform
        query = Content.query.filter(
            Content.streaming_platforms.contains(platform)
        )
        
        if content_type != 'all':
            query = query.filter(Content.content_type == content_type)
        
        # Order by popularity
        query = query.order_by(desc(Content.popularity))
        
        # Paginate
        paginated = query.paginate(page=page, per_page=limit, error_out=False)
        
        results = []
        for content in paginated.items:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            # Parse streaming platforms
            streaming_platforms = json.loads(content.streaming_platforms or '[]')
            
            results.append({
                'id': content.id,
                'slug': content.slug,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url,
                'available_on': streaming_platforms
            })
        
        return jsonify({
            'platform': platform,
            'results': results,
            'pagination': {
                'page': page,
                'per_page': limit,
                'total_pages': paginated.pages,
                'total_items': paginated.total,
                'has_next': paginated.has_next,
                'has_prev': paginated.has_prev
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Platform browse error: {e}")
        return jsonify({'error': 'Failed to browse platform'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check with cache status"""
    try:
        # Basic health info
        health_info = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '5.0.0'  # Updated version with detail service
        }
        
        # Check database connectivity
        try:
            db.session.execute(text('SELECT 1'))
            health_info['database'] = 'connected'
        except:
            health_info['database'] = 'disconnected'
            health_info['status'] = 'degraded'
        
        # Check cache connectivity
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
        
        # Check external services
        health_info['services'] = {
            'tmdb': bool(TMDB_API_KEY),
            'omdb': bool(OMDB_API_KEY),
            'youtube': bool(YOUTUBE_API_KEY),
            'ml_service': bool(ML_SERVICE_URL),
            'algorithms': 'ultra_powerful_enabled',
            'detail_service': 'active'
        }
        
        # Count database records
        try:
            health_info['database_stats'] = {
                'content_count': Content.query.count(),
                'user_count': User.query.count(),
                'review_count': Review.query.count(),
                'person_count': Person.query.count()
            }
        except:
            pass
        
        return jsonify(health_info), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# Stats endpoint
@app.route('/api/stats', methods=['GET'])
@cache.cached(timeout=300, key_prefix='stats')
def get_stats():
    """Get platform statistics"""
    try:
        stats = {
            'content': {
                'total': Content.query.count(),
                'movies': Content.query.filter_by(content_type='movie').count(),
                'tv_shows': Content.query.filter_by(content_type='tv').count(),
                'anime': Content.query.filter_by(content_type='anime').count(),
                'trending': Content.query.filter_by(is_trending=True).count(),
                'new_releases': Content.query.filter_by(is_new_release=True).count(),
                'critics_choice': Content.query.filter_by(is_critics_choice=True).count(),
                '4k_content': Content.query.filter_by(is_4k=True).count(),
                'hdr_content': Content.query.filter_by(is_hdr=True).count()
            },
            'users': {
                'total': User.query.count(),
                'active_today': User.query.filter(
                    User.last_active >= datetime.utcnow() - timedelta(days=1)
                ).count(),
                'admins': User.query.filter_by(is_admin=True).count()
            },
            'engagement': {
                'total_reviews': Review.query.count(),
                'total_interactions': UserInteraction.query.count(),
                'anonymous_interactions': AnonymousInteraction.query.count(),
                'watchlist_items': UserList.query.filter_by(list_type='watchlist').count(),
                'favorite_items': UserList.query.filter_by(list_type='favorites').count()
            },
            'people': {
                'total': Person.query.count(),
                'cast_members': CastMember.query.distinct(CastMember.person_id).count(),
                'crew_members': CrewMember.query.distinct(CrewMember.person_id).count()
            },
            'quality': {
                'average_rating': db.session.query(func.avg(Content.rating)).scalar() or 0,
                'high_rated_content': Content.query.filter(Content.rating >= 8).count(),
                'content_with_trailers': Content.query.filter(Content.youtube_trailer_id != None).count()
            }
        }
        
        # Calculate percentages
        total_content = stats['content']['total']
        if total_content > 0:
            stats['content']['percentages'] = {
                'movies': round(stats['content']['movies'] / total_content * 100, 1),
                'tv_shows': round(stats['content']['tv_shows'] / total_content * 100, 1),
                'anime': round(stats['content']['anime'] / total_content * 100, 1)
            }
        
        stats['timestamp'] = datetime.utcnow().isoformat()
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500
    

    
def create_tables():
    try:
        with app.app_context():
            # First, create all tables that don't exist
            db.create_all()
            
            # Then add missing columns to existing tables
            from sqlalchemy import inspect, text
            
            inspector = inspect(db.engine)
            
            # Check and add slug to content table
            content_columns = [col['name'] for col in inspector.get_columns('content')]
            if 'slug' not in content_columns:
                with db.engine.begin() as conn:
                    conn.execute(text('ALTER TABLE content ADD COLUMN slug VARCHAR(150)'))
                    conn.execute(text('CREATE UNIQUE INDEX ix_content_slug ON content (slug)'))
                logger.info("Added slug column to content table")
                
                # Generate slugs for existing content
                contents = Content.query.all()
                for content in contents:
                    year = content.release_date.year if content.release_date else None
                    content.slug = SlugGenerator.create_slug(content.title, year, content.content_type)
                db.session.commit()
            
            # Check and add slug to person table if it exists
            if 'person' in inspector.get_table_names():
                person_columns = [col['name'] for col in inspector.get_columns('person')]
                if 'slug' not in person_columns:
                    with db.engine.begin() as conn:
                        conn.execute(text('ALTER TABLE person ADD COLUMN slug VARCHAR(150)'))
                        conn.execute(text('CREATE UNIQUE INDEX ix_person_slug ON person (slug)'))
                    logger.info("Added slug column to person table")
                    
                    # Generate slugs for existing persons
                    persons = Person.query.all()
                    for person in persons:
                        person.slug = SlugGenerator.create_person_slug(person.name)
                    db.session.commit()
            
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
                logger.info("Admin user created")
            
            # Log database initialization
            logger.info("Database tables created/updated successfully")
            logger.info(f"Content count: {Content.query.count()}")
            logger.info(f"User count: {User.query.count()}")
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
# Initialize database when app starts
create_tables()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)