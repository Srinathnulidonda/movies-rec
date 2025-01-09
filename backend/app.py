#backend/app.py 
from typing import Optional
from flask import Flask, request, jsonify, session, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_caching import Cache
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from sqlalchemy import func, and_, or_, desc, text
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
import telebot
import threading
from geopy.geocoders import Nominatim
import jwt
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask_mail import Mail
from services.upcoming import UpcomingContentService, ContentType, LanguagePriority
import asyncio
from auth.routes import auth_bp
from services.admin import admin_bp, init_admin
from services.support import support_bp, init_support
from services.critics_choice import critics_choice_bp, init_critics_choice_service
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
from services.personalized import init_personalized
from services.details import init_details_service, SlugManager, ContentService
from services.new_releases import init_cinebrain_new_releases_service
from user.routes import user_bp, init_user_routes
from recommendation import recommendation_bp, init_recommendation_routes
from personalized import init_personalized_system, personalized_bp
from system.routes import system_bp, init_system_routes
from operations.routes import operations_bp, init_operations_routes
import re
import click
import traceback
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
DATABASE_URL = os.environ.get('DATABASE_URL')
if os.environ.get('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is required")
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
REDIS_URL = os.environ.get('REDIS_URL')

if REDIS_URL and REDIS_URL.startswith(('redis://', 'rediss://')):
    app.config['CACHE_TYPE'] = 'redis'
    app.config['CACHE_REDIS_URL'] = REDIS_URL
    app.config['CACHE_DEFAULT_TIMEOUT'] = 3600
else:
    app.config['CACHE_TYPE'] = 'simple'
    app.config['CACHE_DEFAULT_TIMEOUT'] = 1800

if DATABASE_URL and DATABASE_URL.startswith('postgresql://'):
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL + '?sslmode=require'
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'connect_args': {
            'sslmode': 'require',
            'connect_timeout': 10
        }
    }

db = SQLAlchemy(app)
CORS(app)
cache = Cache(app)
TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY environment variable is required")
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY environment variable is required")

app.config['TMDB_API_KEY'] = TMDB_API_KEY
app.config['YOUTUBE_API_KEY'] = YOUTUBE_API_KEY
app.config['SECRET_KEY'] = app.secret_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_http_session():
    session = requests.Session()
    retry = Retry(
        total=2,
        read=2,
        connect=2,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 504)
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=5)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

http_session = create_http_session()
executor = ThreadPoolExecutor(max_workers=3)

CINEBRAIN_REGIONAL_LANGUAGES = {
    'hindi': ['hi', 'hindi', 'bollywood'],
    'telugu': ['te', 'telugu', 'tollywood'],
    'tamil': ['ta', 'tamil', 'kollywood'],
    'kannada': ['kn', 'kannada', 'sandalwood'],
    'malayalam': ['ml', 'malayalam', 'mollywood'],
    'english': ['en', 'english', 'hollywood']
}

CINEBRAIN_LANGUAGE_PRIORITY = {
    'first': ['telugu', 'english', 'hindi'],
    'second': ['malayalam', 'kannada', 'tamil'],
    'codes': {
        'telugu': 'te',
        'english': 'en',
        'hindi': 'hi',
        'malayalam': 'ml',
        'kannada': 'kn',
        'tamil': 'ta'
    }
}

CINEBRAIN_ANIME_GENRES = {
    'shonen': ['Action', 'Adventure', 'Martial Arts', 'School', 'Shounen'],
    'shojo': ['Romance', 'Drama', 'School', 'Slice of Life', 'Shoujo'],
    'seinen': ['Action', 'Drama', 'Thriller', 'Psychological', 'Seinen'],
    'josei': ['Romance', 'Drama', 'Slice of Life', 'Josei'],
    'kodomomuke': ['Kids', 'Family', 'Adventure', 'Comedy']
}

def make_cache_key(*args, **kwargs):
    path = request.path
    args_str = str(hash(frozenset(request.args.items())))
    return f"cinebrain:{path}:{args_str}"

def content_cache_key(content_id):
    return f"cinebrain:content:{content_id}"

def search_cache_key(query, content_type, page):
    return f"cinebrain:search:{query}:{content_type}:{page}"

def recommendations_cache_key(rec_type, **kwargs):
    params = ':'.join([f"{k}={v}" for k, v in sorted(kwargs.items())])
    return f"cinebrain:recommendations:{rec_type}:{params}"

def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'CineBrain authentication required'}), 401
        
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user_id = payload.get('user_id')
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'CineBrain token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid CineBrain token'}), 401
    
    return decorated_function

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    preferred_languages = db.Column(db.Text)
    preferred_genres = db.Column(db.Text)
    location = db.Column(db.String(100))
    avatar_url = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    
    reviews = db.relationship('Review', backref='user', lazy='dynamic')

class Content(db.Model):
    __tablename__ = 'content'
    
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(150), unique=True, nullable=False, index=True)
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
    
    reviews = db.relationship('Review', backref='content', lazy='dynamic')
    cast_crew = db.relationship('ContentPerson', backref='content', lazy='dynamic')

    def ensure_slug(self):
        if not self.slug and self.title:
            try:
                year = self.release_date.year if self.release_date else None
                self.slug = SlugManager.generate_unique_slug(
                    db, Content, self.title, year, self.content_type, existing_id=self.id
                )
            except Exception as e:
                logger.error(f"CineBrain error ensuring slug for content {self.id}: {e}")
                self.slug = f"cinebrain-content-{self.id}-{int(time.time())}"
        return self.slug

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)
    rating = db.Column(db.Float)
    interaction_metadata = db.Column(db.JSON)
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

class Person(db.Model):
    __tablename__ = 'persons'
    
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(150), unique=True, nullable=False, index=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    name = db.Column(db.String(255), nullable=False)
    biography = db.Column(db.Text)
    birthday = db.Column(db.Date)
    deathday = db.Column(db.Date)
    place_of_birth = db.Column(db.String(255))
    profile_path = db.Column(db.String(255))
    popularity = db.Column(db.Float)
    known_for_department = db.Column(db.String(50))
    gender = db.Column(db.Integer)
    also_known_as = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ContentPerson(db.Model):
    __tablename__ = 'content_persons'
    
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    person_id = db.Column(db.Integer, db.ForeignKey('persons.id'), nullable=False)
    role_type = db.Column(db.String(20))
    character = db.Column(db.String(255))
    job = db.Column(db.String(100))
    department = db.Column(db.String(100))
    order = db.Column(db.Integer)
    
    __table_args__ = (
        db.UniqueConstraint('content_id', 'person_id', 'role_type', 'character', 'job'),
    )

class Review(db.Model):
    __tablename__ = 'reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    rating = db.Column(db.Float)
    title = db.Column(db.String(255))
    review_text = db.Column(db.Text)
    has_spoilers = db.Column(db.Boolean, default=False)
    helpful_count = db.Column(db.Integer, default=0)
    is_approved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('content_id', 'user_id'),
    )

def get_session_id():
    if 'cinebrain_session_id' not in session:
        session['cinebrain_session_id'] = hashlib.md5(f"cinebrain{request.remote_addr}{time.time()}".encode()).hexdigest()
    return session['cinebrain_session_id']

def get_user_location(ip_address):
    cache_key = f"cinebrain:location:{ip_address}"
    cached_location = cache.get(cache_key)
    
    if cached_location:
        return cached_location
    
    try:
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
                cache.set(cache_key, location, timeout=86400)
                return location
    except:
        pass
    return None

class CineBrainTMDBService:
    BASE_URL = 'https://api.themoviedb.org/3'
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def search_content(query, content_type='multi', language='en-US', page=1):
        url = f"{CineBrainTMDBService.BASE_URL}/search/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'language': language,
            'page': page
        }
        
        try:
            response = http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain TMDB search error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=7200)
    def get_content_details(content_id, content_type='movie'):
        url = f"{CineBrainTMDBService.BASE_URL}/{content_type}/{content_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'append_to_response': 'credits,videos,similar,reviews,recommendations'
        }
        
        try:
            response = http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain TMDB details error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=1800)
    def get_trending(content_type='all', time_window='day', page=1):
        url = f"{CineBrainTMDBService.BASE_URL}/trending/{content_type}/{time_window}"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        
        try:
            response = http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain TMDB trending error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_popular(content_type='movie', page=1, region=None):
        url = f"{CineBrainTMDBService.BASE_URL}/{content_type}/popular"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        if region:
            params['region'] = region
        
        try:
            response = http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain TMDB popular error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_critics_choice(content_type='movie', page=1):
        url = f"{CineBrainTMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'vote_average.gte': 7.5,
            'vote_count.gte': 100,
            'sort_by': 'vote_average.desc',
            'page': page
        }
        
        try:
            response = http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain TMDB critics choice error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_by_genre(genre_id, content_type='movie', page=1, region=None):
        url = f"{CineBrainTMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': genre_id,
            'sort_by': 'popularity.desc',
            'page': page
        }
        
        if region:
            params['region'] = region
        
        try:
            response = http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain TMDB genre search error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_language_specific(language_code, content_type='movie', page=1):
        url = f"{CineBrainTMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_original_language': language_code,
            'sort_by': 'popularity.desc',
            'page': page
        }
        
        try:
            response = http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain TMDB language search error: {e}")
        return None

class CineBrainJikanService:
    BASE_URL = 'https://api.jikan.moe/v4'
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def search_anime(query, page=1):
        url = f"{CineBrainJikanService.BASE_URL}/anime"
        params = {
            'q': query,
            'page': page,
            'limit': 20
        }
        
        try:
            response = http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain Jikan search error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=7200)
    def get_anime_details(anime_id):
        url = f"{CineBrainJikanService.BASE_URL}/anime/{anime_id}/full"
        
        try:
            response = http_session.get(url, params={}, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain Jikan anime details error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_top_anime(type='tv', page=1):
        url = f"{CineBrainJikanService.BASE_URL}/top/anime"
        params = {
            'type': type,
            'page': page
        }
        
        try:
            response = http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain Jikan top anime error: {e}")
        return None
    
    @staticmethod
    @cache.memoize(timeout=3600)
    def get_anime_by_genre(genre_name, page=1):
        url = f"{CineBrainJikanService.BASE_URL}/anime"
        params = {
            'genres': genre_name,
            'order_by': 'score',
            'sort': 'desc',
            'page': page
        }
        
        try:
            response = http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain Jikan genre search error: {e}")
        return None

class CineBrainYouTubeService:
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    @staticmethod
    @cache.memoize(timeout=86400)
    def search_trailers(query, content_type='movie'):
        url = f"{CineBrainYouTubeService.BASE_URL}/search"
        
        if content_type == 'anime':
            search_query = f"{query} anime trailer PV"
        else:
            search_query = f"{query} official trailer"
        
        params = {
            'key': YOUTUBE_API_KEY,
            'q': search_query,
            'part': 'snippet',
            'type': 'video',
            'maxResults': 3,
            'order': 'relevance'
        }
        
        try:
            response = http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"CineBrain YouTube search error: {e}")
        return None

class CineBrainAnonymousRecommendationEngine:
    @staticmethod
    def get_recommendations_for_anonymous(session_id, ip_address, limit=20):
        try:
            location = get_user_location(ip_address)
            interactions = AnonymousInteraction.query.filter_by(session_id=session_id).all()
            
            recommendations = []
            
            if interactions:
                viewed_content_ids = [interaction.content_id for interaction in interactions]
                viewed_contents = Content.query.filter(Content.id.in_(viewed_content_ids)).all()
                
                all_genres = []
                for content in viewed_contents:
                    if content.genres:
                        try:
                            all_genres.extend(json.loads(content.genres))
                        except (json.JSONDecodeError, TypeError):
                            pass
                
                genre_counts = Counter(all_genres)
                top_genres = [genre for genre, _ in genre_counts.most_common(3)]
                
                for genre in top_genres:
                    genre_content = Content.query.filter(
                        Content.genres.contains(genre)
                    ).limit(7).all()
                    recommendations.extend(genre_content)
            
            if location and location.get('country') == 'India':
                regional_content = Content.query.filter(
                    or_(
                        Content.languages.contains('telugu'),
                        Content.languages.contains('hindi')
                    )
                ).limit(5).all()
                recommendations.extend(regional_content)
            
            trending_content = Content.query.filter_by(is_trending=True).limit(10).all()
            recommendations.extend(trending_content)
            
            seen_ids = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec.id not in seen_ids:
                    seen_ids.add(rec.id)
                    if not rec.slug:
                        rec.ensure_slug()
                    unique_recommendations.append(rec)
                    if len(unique_recommendations) >= limit:
                        break
            
            return unique_recommendations
        except Exception as e:
            logger.error(f"CineBrain error getting anonymous recommendations: {e}")
            return []

models = {
    'User': User,
    'Content': Content,
    'UserInteraction': UserInteraction,
    'AdminRecommendation': AdminRecommendation,
    'Review': Review,
    'Person': Person,
    'ContentPerson': ContentPerson,
    'AnonymousInteraction': AnonymousInteraction
}

details_service = None
content_service = None
cinebrain_new_releases_service = None

try:
    with app.app_context():
        details_service = init_details_service(app, db, models, cache)
        content_service = ContentService(db, models)
        logger.info("CineBrain details and content services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize CineBrain details/content services: {e}")

services = {
    'TMDBService': CineBrainTMDBService,
    'JikanService': CineBrainJikanService,
    'ContentService': content_service,
    'http_session': http_session,
    'cache': cache
}

# Initialize the new modular reviews system
try:
    from reviews import init_reviews_service
    
    review_services = init_reviews_service(app, db, models, cache)
    if review_services:
        logger.info("✅ CineBrain Reviews system initialized successfully")
        services.update(review_services)
    else:
        logger.warning("⚠️ CineBrain Reviews system failed to initialize")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain Reviews system: {e}")

try:
    cinebrain_new_releases_service = init_cinebrain_new_releases_service(app, db, models, services)
    if cinebrain_new_releases_service:
        logger.info("CineBrain new releases service integrated successfully")
        services['new_releases_service'] = cinebrain_new_releases_service
    else:
        logger.warning("CineBrain new releases service failed to initialize")
except Exception as e:
    logger.error(f"Failed to initialize CineBrain new releases service: {e}")

try:
    init_support(app, db, models, services)
    app.register_blueprint(support_bp)
    logger.info("CineBrain support service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize CineBrain support service: {e}")

# Replace the existing personalized initialization block with the new advanced system
try:
    # Initialize the new modular personalized recommendation system
    profile_analyzer, personalized_recommendation_engine = init_personalized_system(
        app, db, models, services, cache
    )
    
    if profile_analyzer and personalized_recommendation_engine:
        logger.info("✅ CineBrain Advanced Personalized Recommendation System initialized successfully")
        logger.info(f"   - Profile Analyzer: Active")
        logger.info(f"   - Recommendation Engine: Active with Telugu-first priority")
        logger.info(f"   - Real-time Learning: Enabled")
        
        # Add to services dictionary
        services['profile_analyzer'] = profile_analyzer
        services['personalized_recommendation_engine'] = personalized_recommendation_engine
        
        # Register the personalized blueprint
        app.register_blueprint(personalized_bp, url_prefix='/api')
        logger.info("✅ CineBrain Personalized routes registered at /api/personalized/*")
    else:
        logger.warning("⚠️ CineBrain Personalized Recommendation System failed to initialize fully")
        
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain Personalized Recommendation System: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")

# Initialize the new modular auth system with Resend
try:
    from auth.service import init_auth
    from auth.routes import auth_bp
    
    init_auth(app, db, User)
    app.register_blueprint(auth_bp)
    logger.info("✅ CineBrain authentication service with Resend email initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain authentication service: {e}")

try:
    init_admin(app, db, models, services)
    app.register_blueprint(admin_bp)
    logger.info("CineBrain admin service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize CineBrain admin service: {e}")

try:
    init_user_routes(app, db, models, {**services, 'cache': cache})
    app.register_blueprint(user_bp)
    logger.info("CineBrain user module initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize CineBrain user module: {e}")

try:
    critics_choice_service = init_critics_choice_service(app, db, models, services, cache)
    app.register_blueprint(critics_choice_bp)
    if critics_choice_service:
        logger.info("CineBrain Critics Choice service integrated successfully")
        services['critics_choice_service'] = critics_choice_service
    else:
        logger.warning("CineBrain Critics Choice service failed to initialize")
except Exception as e:
    logger.error(f"Failed to initialize CineBrain Critics Choice service: {e}")

# Add RecommendationOrchestrator to services
try:
    cinebrain_recommendation_orchestrator = RecommendationOrchestrator()
    services['recommendation_orchestrator'] = cinebrain_recommendation_orchestrator
    logger.info("CineBrain RecommendationOrchestrator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize CineBrain RecommendationOrchestrator: {e}")

# Initialize recommendation routes
try:
    init_recommendation_routes(app, db, models, services, cache)
    app.register_blueprint(recommendation_bp)
    logger.info("CineBrain recommendation routes initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize CineBrain recommendation routes: {e}")

# Initialize system routes
try:
    init_system_routes(app, db, models, services)
    app.register_blueprint(system_bp)
    logger.info("✅ CineBrain system monitoring service initialized successfully")
    services['system_service'] = True
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain system monitoring service: {e}")

# Initialize operations routes
try:
    init_operations_routes(app, db, models, services)
    app.register_blueprint(operations_bp)
    logger.info("✅ CineBrain operations service initialized successfully")
    services['operations_service'] = True
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain operations service: {e}")

def setup_support_monitoring():
    def support_monitor():
        while True:
            try:
                if not app or not db:
                    time.sleep(60)
                    continue
                
                with app.app_context():
                    try:
                        from services.admin import AdminNotificationService
                        from services.support import TicketStatus, TicketPriority
                        
                        if 'SupportTicket' in models and models['SupportTicket']:
                            SupportTicket = models['SupportTicket']
                            overdue_tickets = db.session.query(SupportTicket).filter(
                                SupportTicket.sla_deadline < datetime.utcnow(),
                                SupportTicket.sla_breached == False,
                                SupportTicket.status.in_([TicketStatus.OPEN, TicketStatus.IN_PROGRESS])
                            ).all()
                            
                            for ticket in overdue_tickets:
                                ticket.sla_breached = True
                                AdminNotificationService.notify_sla_breach(ticket)
                            
                            if overdue_tickets:
                                db.session.commit()
                        
                        if 'SupportTicket' in models and models['SupportTicket']:
                            SupportTicket = models['SupportTicket']
                            urgent_tickets = db.session.query(SupportTicket).filter(
                                SupportTicket.priority == TicketPriority.URGENT,
                                SupportTicket.first_response_at.is_(None),
                                SupportTicket.created_at < datetime.utcnow() - timedelta(hours=1)
                            ).all()
                            
                            for ticket in urgent_tickets:
                                AdminNotificationService.create_notification(
                                    'urgent_ticket',
                                    f"CineBrain Urgent Ticket Needs Attention",
                                    f"Ticket #{ticket.ticket_number} has been open for over 1 hour without response",
                                    related_ticket_id=ticket.id,
                                    is_urgent=True,
                                    action_required=True
                                )
                    except Exception as e:
                        logger.error(f"CineBrain support monitoring inner error: {e}")
                
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"CineBrain support monitoring error: {e}")
                time.sleep(300)
    
    monitor_thread = threading.Thread(target=support_monitor, daemon=True)
    monitor_thread.start()
    logger.info("CineBrain support monitoring thread started")

def on_new_support_ticket(ticket):
    try:
        from services.admin import AdminNotificationService
        AdminNotificationService.notify_new_ticket(ticket)
    except Exception as e:
        logger.error(f"CineBrain error handling new ticket notification: {e}")

def on_new_feedback(feedback):
    try:
        from services.admin import AdminNotificationService
        AdminNotificationService.notify_feedback_received(feedback)
    except Exception as e:
        logger.error(f"CineBrain error handling new feedback notification: {e}")

@app.route('/api/webhooks/support/ticket-created', methods=['POST'])
def webhook_ticket_created():
    try:
        data = request.get_json()
        ticket_id = data.get('ticket_id')
        
        if ticket_id and 'SupportTicket' in globals():
            ticket = globals()['SupportTicket'].query.get(ticket_id)
            if ticket:
                on_new_support_ticket(ticket)
        
        return jsonify({'success': True, 'cinebrain_service': 'support_webhook'}), 200
    except Exception as e:
        logger.error(f"CineBrain webhook error: {e}")
        return jsonify({'error': 'CineBrain webhook processing failed'}), 500

@app.route('/api/webhooks/support/feedback-created', methods=['POST'])
def webhook_feedback_created():
    try:
        data = request.get_json()
        feedback_id = data.get('feedback_id')
        
        if feedback_id and 'Feedback' in globals():
            feedback = globals()['Feedback'].query.get(feedback_id)
            if feedback:
                on_new_feedback(feedback)
        
        return jsonify({'success': True, 'cinebrain_service': 'support_webhook'}), 200
    except Exception as e:
        logger.error(f"CineBrain feedback webhook error: {e}")
        return jsonify({'error': 'CineBrain webhook processing failed'}), 500

@app.route('/api/details/<slug>', methods=['GET'])
def get_content_details_by_slug(slug):
    try:
        user_id = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
                user_id = payload.get('user_id')
            except:
                pass
        
        if details_service:
            details = details_service.get_details_by_slug(slug, user_id)
        else:
            logger.error("CineBrain details service not available")
            return jsonify({'error': 'CineBrain service unavailable'}), 503
        
        if not details:
            return jsonify({'error': 'CineBrain content not found'}), 404
        
        return jsonify(details), 200
        
    except Exception as e:
        logger.error(f"CineBrain error getting details for slug {slug}: {e}")
        return jsonify({'error': 'Failed to get CineBrain content details'}), 500

@app.route('/api/search', methods=['GET'])
@cache.cached(timeout=300, key_prefix=make_cache_key)
def search_content():
    try:
        query = request.args.get('query', '')
        content_type = request.args.get('type', 'multi')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'CineBrain search query parameter required'}), 400
        
        session_id = get_session_id()
        
        futures = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures.append(executor.submit(CineBrainTMDBService.search_content, query, content_type, page=page))
            
            if content_type in ['anime', 'multi']:
                futures.append(executor.submit(CineBrainJikanService.search_anime, query, page=page))
        
        tmdb_results = None
        anime_results = None
        
        try:
            tmdb_results = futures[0].result(timeout=5)
        except Exception as e:
            logger.warning(f"CineBrain TMDB search timeout/error: {e}")
        
        if len(futures) > 1:
            try:
                anime_results = futures[1].result(timeout=5)
            except Exception as e:
                logger.warning(f"CineBrain anime search timeout/error: {e}")
        
        results = []
        
        if tmdb_results:
            for item in tmdb_results.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = content_service.save_content_from_tmdb(item, content_type_detected)
                if content:
                    try:
                        interaction = AnonymousInteraction(
                            session_id=session_id,
                            content_id=content.id,
                            interaction_type='search',
                            ip_address=request.remote_addr
                        )
                        db.session.add(interaction)
                    except Exception as e:
                        logger.warning(f"CineBrain failed to record interaction: {e}")
                    
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
        
        if anime_results:
            for anime in anime_results.get('data', []):
                content = content_service.save_anime_content(anime)
                if content:
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
        
        try:
            db.session.commit()
        except Exception as e:
            logger.warning(f"CineBrain failed to commit search interactions: {e}")
            db.session.rollback()
        
        return jsonify({
            'results': results,
            'total_results': tmdb_results.get('total_results', 0) if tmdb_results else 0,
            'total_pages': tmdb_results.get('total_pages', 0) if tmdb_results else 0,
            'current_page': page,
            'cinebrain_service': 'search'
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain search error: {e}")
        return jsonify({'error': 'CineBrain search failed'}), 500

@app.route('/api/content/<int:content_id>', methods=['GET'])
def get_content_details(content_id):
    try:
        cache_key = content_cache_key(content_id)
        cached_content = cache.get(cache_key)
        
        if cached_content:
            content = cached_content
        else:
            content = Content.query.get_or_404(content_id)
            cache.set(cache_key, content, timeout=3600)
        
        if not content.slug:
            try:
                content.ensure_slug()
                db.session.commit()
            except Exception as e:
                logger.warning(f"CineBrain failed to ensure slug: {e}")
        
        try:
            session_id = get_session_id()
            interaction = AnonymousInteraction(
                session_id=session_id,
                content_id=content.id,
                interaction_type='view',
                ip_address=request.remote_addr
            )
            db.session.add(interaction)
        except Exception as e:
            logger.warning(f"CineBrain failed to record view interaction: {e}")
        
        additional_details = None
        cast = []
        crew = []
        
        try:
            if content.content_type == 'anime' and content.mal_id:
                additional_details = CineBrainJikanService.get_anime_details(content.mal_id)
                if additional_details:
                    anime_data = additional_details.get('data', {})
                    if 'voices' in anime_data:
                        cast = anime_data['voices'][:10]
                    if 'staff' in anime_data:
                        crew = anime_data['staff'][:5]
            elif content.tmdb_id:
                additional_details = CineBrainTMDBService.get_content_details(content.tmdb_id, content.content_type)
                if additional_details:
                    cast = additional_details.get('credits', {}).get('cast', [])[:10]
                    crew = additional_details.get('credits', {}).get('crew', [])[:5]
        except Exception as e:
            logger.warning(f"CineBrain failed to get additional details: {e}")
        
        similar_content = []
        try:
            try:
                genres = json.loads(content.genres) if content.genres else []
            except (json.JSONDecodeError, TypeError):
                genres = []
            
            if genres:
                primary_genre = genres[0]
                similar_items = Content.query.filter(
                    Content.id != content_id,
                    Content.content_type == content.content_type,
                    Content.genres.contains(primary_genre)
                ).order_by(Content.rating.desc()).limit(8).all()
                
                for similar in similar_items:
                    if not similar.slug:
                        try:
                            similar.ensure_slug()
                        except Exception:
                            similar.slug = f"cinebrain-content-{similar.id}"
                    
                    youtube_url = None
                    if similar.youtube_trailer_id:
                        youtube_url = f"https://www.youtube.com/watch?v={similar.youtube_trailer_id}"
                    
                    similar_content.append({
                        'id': similar.id,
                        'slug': similar.slug,
                        'title': similar.title,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{similar.poster_path}" if similar.poster_path and not similar.poster_path.startswith('http') else similar.poster_path,
                        'rating': similar.rating,
                        'content_type': similar.content_type,
                        'youtube_trailer': youtube_url,
                        'similarity_score': 0.8,
                        'match_type': 'genre_based'
                    })
        except Exception as e:
            logger.warning(f"CineBrain failed to get similar content: {e}")
        
        try:
            db.session.commit()
        except Exception as e:
            logger.warning(f"CineBrain failed to commit view interaction: {e}")
            db.session.rollback()
        
        youtube_trailer_url = None
        if content.youtube_trailer_id:
            youtube_trailer_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
        
        response_data = {
            'id': content.id,
            'slug': content.slug,
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
            'similar_content': similar_content,
            'cast': cast,
            'crew': crew,
            'is_trending': content.is_trending,
            'is_new_release': content.is_new_release,
            'is_critics_choice': content.is_critics_choice,
            'cinebrain_service': 'content_details'
        }
        
        if content.content_type == 'anime':
            response_data['anime_genres'] = json.loads(content.anime_genres or '[]')
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"CineBrain content details error: {e}")
        return jsonify({'error': 'Failed to get CineBrain content details'}), 500

@app.route('/api/person/<slug>', methods=['GET'])
def get_person_details(slug):
    try:
        user_id = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
                user_id = payload.get('user_id')
            except:
                pass
        
        if details_service:
            person_details = details_service.get_person_details(slug)
        else:
            logger.error("CineBrain details service not available")
            return jsonify({'error': 'CineBrain service unavailable'}), 503
        
        if not person_details:
            return jsonify({'error': 'CineBrain person not found'}), 404
        
        return jsonify(person_details), 200
        
    except Exception as e:
        logger.error(f"CineBrain error getting person details for slug {slug}: {e}")
        return jsonify({'error': 'Failed to get CineBrain person details'}), 500

@app.route('/api/admin/slugs/migrate', methods=['POST'])
@auth_required
def migrate_all_slugs():
    try:
        user = User.query.get(request.user_id)
        if not user or not user.is_admin:
            return jsonify({'error': 'CineBrain admin access required'}), 403
        
        if details_service:
            batch_size = int(request.json.get('batch_size', 50))
            stats = details_service.migrate_all_slugs(batch_size)
            return jsonify({
                'success': True,
                'migration_stats': stats,
                'cinebrain_service': 'slug_migration'
            }), 200
        else:
            return jsonify({'error': 'CineBrain service unavailable'}), 503
            
    except Exception as e:
        logger.error(f"CineBrain error migrating slugs: {e}")
        return jsonify({'error': 'Failed to migrate CineBrain slugs'}), 500

@app.route('/api/admin/content/<int:content_id>/slug', methods=['PUT'])
@auth_required
def update_content_slug(content_id):
    try:
        user = User.query.get(request.user_id)
        if not user or not user.is_admin:
            return jsonify({'error': 'CineBrain admin access required'}), 403
        
        if details_service:
            force_update = request.json.get('force_update', False)
            new_slug = details_service.update_content_slug(content_id, force_update)
            
            if new_slug:
                return jsonify({
                    'success': True,
                    'new_slug': new_slug,
                    'cinebrain_service': 'slug_update'
                }), 200
            else:
                return jsonify({'error': 'CineBrain content not found or update failed'}), 404
        else:
            return jsonify({'error': 'CineBrain service unavailable'}), 503
            
    except Exception as e:
        logger.error(f"CineBrain error updating content slug: {e}")
        return jsonify({'error': 'Failed to update CineBrain slug'}), 500

@app.route('/api/content/<int:content_id>/refresh-cast-crew', methods=['POST'])
def refresh_cast_crew(content_id):
    try:
        content = Content.query.get_or_404(content_id)
        
        if not content.tmdb_id:
            return jsonify({'error': 'No TMDB ID available for CineBrain content'}), 400
        
        if details_service:
            cast_crew = details_service._fetch_and_save_all_cast_crew(content)
            return jsonify({
                'success': True,
                'cast_count': len(cast_crew['cast']),
                'crew_count': sum(len(crew_list) for crew_list in cast_crew['crew'].values()),
                'cinebrain_service': 'cast_crew_refresh'
            })
        else:
            return jsonify({'error': 'CineBrain details service not available'}), 503
            
    except Exception as e:
        logger.error(f"CineBrain error refreshing cast/crew: {e}")
        return jsonify({'error': 'Failed to refresh CineBrain cast/crew data'}), 500

@app.route('/api/admin/populate-cast-crew', methods=['POST'])
@auth_required
def populate_all_cast_crew():
    try:
        user = User.query.get(request.user_id)
        if not user or not user.is_admin:
            return jsonify({'error': 'CineBrain admin access required'}), 403
        
        batch_size = int(request.json.get('batch_size', 10))
        
        content_items = Content.query.filter(
            Content.tmdb_id.isnot(None),
            ~Content.id.in_(
                db.session.query(ContentPerson.content_id).distinct()
            )
        ).limit(batch_size).all()
        
        processed = 0
        errors = 0
        
        for content in content_items:
            try:
                if details_service:
                    cast_crew = details_service._fetch_and_save_all_cast_crew(content)
                    processed += 1
                    logger.info(f"CineBrain populated cast/crew for {content.title}")
            except Exception as e:
                logger.error(f"CineBrain error processing {content.title}: {e}")
                errors += 1
        
        return jsonify({
            'success': True,
            'processed': processed,
            'errors': errors,
            'total_available': len(content_items),
            'message': f"Successfully populated CineBrain cast/crew for {processed} content items",
            'cinebrain_service': 'cast_crew_population'
        }), 200
        
    except Exception as e:
        logger.error(f"CineBrain error in bulk cast/crew population: {e}")
        return jsonify({'error': 'Failed to populate CineBrain cast/crew'}), 500

# CLI Commands

@app.cli.command('generate-slugs')
def generate_slugs():
    try:
        print("Starting CineBrain comprehensive slug generation...")
        
        if not details_service:
            print("Error: CineBrain details service not available")
            return
        
        stats = details_service.migrate_all_slugs(batch_size=50)
        
        print("CineBrain slug migration completed successfully!")
        print(f"Content updated: {stats['content_updated']}")
        print(f"Persons updated: {stats['persons_updated']}")
        print(f"Total processed: {stats['total_processed']}")
        print(f"Errors: {stats['errors']}")
        
        try:
            total_content = Content.query.count()
            content_with_slugs = Content.query.filter(
                and_(Content.slug != None, Content.slug != '')
            ).count()
            
            total_persons = Person.query.count()
            persons_with_slugs = Person.query.filter(
                and_(Person.slug != None, Person.slug != '')
            ).count()
            
            print(f"\nCineBrain Verification Results:")
            print(f"Content: {content_with_slugs}/{total_content} ({round(content_with_slugs/total_content*100, 1)}% coverage)")
            print(f"Persons: {persons_with_slugs}/{total_persons} ({round(persons_with_slugs/total_persons*100, 1)}% coverage)")
            
        except Exception as e:
            print(f"CineBrain error during verification: {e}")
        
    except Exception as e:
        print(f"Failed to generate CineBrain slugs: {e}")
        logger.error(f"CineBrain CLI slug generation error: {e}")

@app.cli.command('populate-cast-crew')
def populate_cast_crew_cli():
    try:
        print("Starting CineBrain cast/crew population...")
        
        if not details_service:
            print("Error: CineBrain details service not available")
            return
        
        content_items = Content.query.filter(
            Content.tmdb_id.isnot(None),
            ~Content.id.in_(
                db.session.query(ContentPerson.content_id).distinct()
            )
        ).all()
        
        print(f"Found {len(content_items)} CineBrain content items without cast/crew")
        
        processed = 0
        errors = 0
        
        for i, content in enumerate(content_items):
            try:
                print(f"Processing {i+1}/{len(content_items)}: {content.title}")
                cast_crew = details_service._fetch_and_save_all_cast_crew(content)
                processed += 1
                print(f"  Added {len(cast_crew['cast'])} cast members and crew")
            except Exception as e:
                print(f"  Error: {e}")
                errors += 1
        
        print(f"\nCineBrain cast/crew population completed!")
        print(f"Processed: {processed}")
        print(f"Errors: {errors}")
        
    except Exception as e:
        print(f"Failed to populate CineBrain cast/crew: {e}")
        logger.error(f"CineBrain CLI cast/crew population error: {e}")

@app.cli.command('cinebrain-new-releases-refresh')
def cinebrain_new_releases_refresh_cli():
    try:
        print("Starting CineBrain new releases manual refresh...")
        
        if not cinebrain_new_releases_service:
            print("Error: CineBrain new releases service not available")
            return
        
        cinebrain_new_releases_service.refresh_new_releases()
        stats = cinebrain_new_releases_service.get_stats()
        
        print("CineBrain new releases refresh completed!")
        print(f"Total items: {stats.get('total_items', 0)}")
        print(f"Priority items: {stats.get('priority_items', 0)}")
        print(f"Movies: {stats.get('movies', 0)}")
        print(f"TV Shows: {stats.get('tv_shows', 0)}")
        print(f"Anime: {stats.get('anime', 0)}")
        
    except Exception as e:
        print(f"Failed to refresh CineBrain new releases: {e}")
        logger.error(f"CineBrain CLI new releases refresh error: {e}")

@app.cli.command('analyze-user-profiles')
def analyze_user_profiles_cli():
    """Analyze all user profiles and generate Cinematic DNA"""
    try:
        print("Starting CineBrain user profile analysis...")
        
        if 'profile_analyzer' not in services or not services['profile_analyzer']:
            print("Error: Profile Analyzer not available")
            return
        
        profile_analyzer = services['profile_analyzer']
        
        users = User.query.all()
        print(f"Found {len(users)} users to analyze")
        
        successful = 0
        failed = 0
        
        for i, user in enumerate(users, 1):
            try:
                print(f"\nAnalyzing user {i}/{len(users)}: {user.username} (ID: {user.id})")
                
                profile = profile_analyzer.build_comprehensive_profile(user.id)
                
                if profile:
                    print(f"  ✓ Profile built successfully")
                    print(f"  - Cinematic Sophistication: {profile['cinematic_dna']['cinematic_sophistication_score']:.2f}")
                    print(f"  - Telugu Affinity: {profile['cinematic_dna']['telugu_cultural_affinity']:.2f}")
                    print(f"  - Profile Confidence: {profile['profile_confidence']:.2f}")
                    print(f"  - Recommendation Strategy: {profile['recommendations_strategy']}")
                    successful += 1
                else:
                    print(f"  ✗ Failed to build profile")
                    failed += 1
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                failed += 1
        
        print(f"\nCineBrain Profile Analysis Complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
    except Exception as e:
        print(f"Failed to analyze user profiles: {e}")
        logger.error(f"CineBrain CLI profile analysis error: {e}")

@app.cli.command('test-personalized-recommendations')
@click.argument('username')
def test_personalized_recommendations_cli(username):
    """Test personalized recommendations for a specific user"""
    try:
        print(f"Testing CineBrain personalized recommendations for user: {username}")
        
        user = User.query.filter_by(username=username).first()
        if not user:
            print(f"Error: User '{username}' not found")
            return
        
        if 'personalized_recommendation_engine' not in services:
            print("Error: Personalized Recommendation Engine not available")
            return
        
        engine = services['personalized_recommendation_engine']
        
        print(f"\nGenerating recommendations for {username} (ID: {user.id})...")
        
        recommendations = engine.get_personalized_recommendations(
            user_id=user.id,
            recommendation_type='for_you',
            limit=10
        )
        
        if recommendations['success']:
            print(f"\n✅ Successfully generated {len(recommendations['recommendations'])} recommendations")
            
            print("\nTop 5 Recommendations:")
            for i, rec in enumerate(recommendations['recommendations'][:5], 1):
                print(f"{i}. {rec['title']}")
                print(f"   - Type: {rec['content_type']}")
                print(f"   - Languages: {', '.join(rec['languages'])}")
                print(f"   - Personalization Score: {rec['personalization_score']:.3f}")
                print(f"   - Recommendation Strength: {rec['recommendation_strength']}")
                print(f"   - Why: {rec['personalized_explanation']}")
                print()
            
            insights = recommendations.get('profile_insights', {})
            print("User Profile Insights:")
            print(f"- Profile Strength: {insights.get('profile_strength', 'unknown')}")
            print(f"- Cinematic Sophistication: {insights.get('cinematic_sophistication', 0):.2f}")
            print(f"- Dominant Themes: {', '.join(insights.get('dominant_themes', []))}")
            print(f"- Language Priority: {', '.join(insights.get('language_priority', []))}")
            
        else:
            print(f"❌ Failed to generate recommendations")
            
    except Exception as e:
        print(f"Error testing recommendations: {e}")
        logger.error(f"CineBrain CLI recommendation test error: {e}")

def create_tables():
    try:
        with app.app_context():
            db.create_all()
            
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                print("Creating CineBrain admin user...")
                admin = User(
                    username='admin',
                    email='srinathnulidonda.dev@gmail.com',
                    password_hash=generate_password_hash('admin123'),
                    is_admin=True
                )
                db.session.add(admin)
                db.session.commit()
                print("CineBrain admin user created successfully")
                print(f"Admin username: admin")
                print(f"Admin password: admin123")
            else:
                print("CineBrain admin user already exists")
                print(f"Admin ID: {admin.id}")
                print(f"Admin username: {admin.username}")
                print(f"Admin email: {admin.email}")
                print(f"Password hash exists: {bool(admin.password_hash)}")
            
            test_check = check_password_hash(admin.password_hash, 'admin123')
            print(f"Password verification test: {test_check}")
            
            setup_support_monitoring()
            
            logger.info("CineBrain database tables created successfully")
    except Exception as e:
        logger.error(f"CineBrain database initialization error: {e}")

create_tables()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    print("=== Running CineBrain Flask with Advanced Personalized Recommendation System & Comprehensive Operations ===")
    print("Features:")
    print("  ✅ Cinematic DNA Analysis")
    print("  ✅ Advanced Behavioral Analysis") 
    print("  ✅ Preference Embedding Engine")
    print("  ✅ Telugu-first Cultural Priority")
    print("  ✅ Real-time Learning")
    print("  ✅ Multi-strategy Recommendations")
    print("  ✅ Comprehensive System Monitoring")
    print("  ✅ Performance Analytics")
    print("  ✅ Automated Cache Refresh System")
    print("  ✅ Background Operations Management")
    print("  ✅ Resend Email Service Integration")
    print("  ✅ Modular Authentication System")
    print("  ✅ Advanced Reviews & Rating System")
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
else:
    print("=== CineBrain Flask app with Advanced Systems ===")
    print(f"App name: {app.name}")
    print(f"Python version: 3.13.4")
    print(f"CineBrain brand: CineBrain Entertainment Platform")
    print(f"Database URI configured: {'Yes' if app.config.get('SQLALCHEMY_DATABASE_URI') else 'No'}")
    print(f"Cache type: {app.config.get('CACHE_TYPE', 'Not configured')}")
    print(f"CineBrain details service status: {'Initialized' if details_service else 'Failed to initialize'}")
    print(f"CineBrain content service status: {'Initialized' if content_service else 'Failed to initialize'}")
    print(f"CineBrain reviews system status: {'Integrated' if 'review_service' in services else 'Failed to initialize'}")
    print(f"CineBrain new releases service status: {'Integrated' if cinebrain_new_releases_service else 'Failed to initialize'}")
    print(f"CineBrain critics choice service status: {'Integrated' if 'critics_choice_service' in services else 'Failed to initialize'}")
    print(f"CineBrain support service status: {'Integrated' if 'support_bp' in app.blueprints else 'Not integrated'}")
    print(f"CineBrain auth service status: {'Integrated with Resend' if 'auth' in app.blueprints else 'Not integrated'}")
    print(f"CineBrain user service status: {'Integrated' if 'user_bp' in app.blueprints else 'Not integrated'}")
    print(f"CineBrain recommendation service status: {'Integrated' if 'recommendations' in app.blueprints else 'Not integrated'}")
    print(f"CineBrain system monitoring service status: {'Integrated' if 'system_bp' in app.blueprints else 'Not integrated'}")
    print(f"CineBrain operations service status: {'Integrated' if 'operations_bp' in app.blueprints else 'Not integrated'}")
    print(f"   Personalized System: {'Active' if 'profile_analyzer' in services else 'Not Initialized'}")
    
    print("\n=== CineBrain Advanced Features ===")
    print("✅ Modular recommendation architecture")
    print("✅ All original endpoints preserved")
    print("✅ Advanced algorithms integration")
    print("✅ Telugu-first priority system")
    print("✅ Ultra-powerful similarity engine")
    print("✅ Critics choice with auto-refresh")
    print("✅ New releases with 45-day strict filter")
    print("✅ Upcoming content with Telugu focus")
    print("✅ Anonymous recommendation engine")
    print("✅ Admin recommendation system")
    print("✅ Advanced Personalized System with Cinematic DNA")
    print("✅ Real-time Learning and Profile Analysis")
    print("✅ Multi-strategy Recommendation Engine")
    print("✅ Behavioral Analysis and Preference Embeddings")
    print("✅ Comprehensive System Monitoring & Health Checks")
    print("✅ Performance Analytics & Alerts")
    print("✅ Real-time Metrics & Database Statistics")
    print("✅ Automated Cache Refresh with UptimeRobot Support")
    print("✅ Background Operations & Maintenance Tasks")
    print("✅ Resend Email Service with Professional Templates")
    print("✅ Modular Authentication System")
    print("✅ Advanced Reviews & Rating System with Moderation")
    
    print(f"\n=== CineBrain Registered Routes ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.rule} [{', '.join(rule.methods)}]")
    print("=== End of CineBrain Routes ===\n")