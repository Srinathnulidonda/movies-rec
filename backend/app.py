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
import services.auth as auth
from services.auth import init_auth, auth_bp
from services.admin import admin_bp, init_admin
from services.users import users_bp, init_users
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
from services.personalized import personalized_bp, init_personalized
from services.details import init_details_service, SlugManager, ContentService
import re

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

DATABASE_URL = 'postgresql://movies_rec_panf_user:BO5X3d2QihK7GG9hxgtBiCtni8NTbbIi@dpg-d2q7gamr433s73e0hcm0-a/movies_rec_panf'

if os.environ.get('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

REDIS_URL = os.environ.get('REDIS_URL', 'redis://red-d2qlbuje5dus73c71qog:xp7inVzgblGCbo9I4taSGLdKUg0xY91I@red-d2qlbuje5dus73c71qog:6379')

if REDIS_URL and REDIS_URL.startswith(('redis://', 'rediss://')):
    app.config['CACHE_TYPE'] = 'redis'
    app.config['CACHE_REDIS_URL'] = REDIS_URL
    app.config['CACHE_DEFAULT_TIMEOUT'] = 3600
else:
    app.config['CACHE_TYPE'] = 'simple'
    app.config['CACHE_DEFAULT_TIMEOUT'] = 1800

db = SQLAlchemy(app)
CORS(app)
cache = Cache(app)

TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '52260795')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')

app.config['TMDB_API_KEY'] = TMDB_API_KEY
app.config['OMDB_API_KEY'] = OMDB_API_KEY
app.config['YOUTUBE_API_KEY'] = YOUTUBE_API_KEY

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

REGIONAL_LANGUAGES = {
    'hindi': ['hi', 'hindi', 'bollywood'],
    'telugu': ['te', 'telugu', 'tollywood'],
    'tamil': ['ta', 'tamil', 'kollywood'],
    'kannada': ['kn', 'kannada', 'sandalwood'],
    'malayalam': ['ml', 'malayalam', 'mollywood'],
    'english': ['en', 'english', 'hollywood']
}

LANGUAGE_PRIORITY = {
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

ANIME_GENRES = {
    'shonen': ['Action', 'Adventure', 'Martial Arts', 'School', 'Shounen'],
    'shojo': ['Romance', 'Drama', 'School', 'Slice of Life', 'Shoujo'],
    'seinen': ['Action', 'Drama', 'Thriller', 'Psychological', 'Seinen'],
    'josei': ['Romance', 'Drama', 'Slice of Life', 'Josei'],
    'kodomomuke': ['Kids', 'Family', 'Adventure', 'Comedy']
}

recommendation_orchestrator = RecommendationOrchestrator()

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

def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401
        
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user_id = payload.get('user_id')
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
    
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
                logger.error(f"Error ensuring slug for content {self.id}: {e}")
                self.slug = f"content-{self.id}-{int(time.time())}"
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
    if 'session_id' not in session:
        session['session_id'] = hashlib.md5(f"{request.remote_addr}{time.time()}".encode()).hexdigest()
    return session['session_id']

def get_user_location(ip_address):
    cache_key = f"location:{ip_address}"
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
            response = http_session.get(url, params=params, timeout=5)
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
            response = http_session.get(url, params=params, timeout=5)
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
            response = http_session.get(url, params=params, timeout=5)
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
            response = http_session.get(url, params=params, timeout=5)
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
            response = http_session.get(url, params=params, timeout=5)
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
            response = http_session.get(url, params=params, timeout=5)
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
            response = http_session.get(url, params=params, timeout=5)
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
            response = http_session.get(url, params=params, timeout=5)
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
            response = http_session.get(OMDbService.BASE_URL, params=params, timeout=5)
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
            response = http_session.get(url, params=params, timeout=5)
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
            response = http_session.get(url, params={}, timeout=5)
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
            response = http_session.get(url, params=params, timeout=5)
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
            response = http_session.get(url, params=params, timeout=5)
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
            'maxResults': 3,
            'order': 'relevance'
        }
        
        try:
            response = http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
        return None

class AnonymousRecommendationEngine:
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
            logger.error(f"Error getting anonymous recommendations: {e}")
            return []

models = {
    'User': User,
    'Content': Content,
    'UserInteraction': UserInteraction,
    'AdminRecommendation': AdminRecommendation,
    'Review': Review,
    'Person': Person,
    'ContentPerson': ContentPerson
}

details_service = None
content_service = None
try:
    with app.app_context():
        details_service = init_details_service(app, db, models, cache)
        content_service = ContentService(db, models)
        logger.info("Details and Content services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize details/content services: {e}")

services = {
    'TMDBService': TMDBService,
    'JikanService': JikanService,
    'ContentService': content_service,
    'http_session': http_session,
    'cache': cache
}

try:
    init_personalized(app, db, models, services, cache)
    app.register_blueprint(personalized_bp)
    logger.info("Personalized recommendation service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize personalized service: {e}")


app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(users_bp)
init_auth(app, db, User)

init_admin(app, db, models, services)
init_users(app, db, models, services)
init_users(app, db, models, {**services, 'cache': cache})

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
            logger.error("Details service not available")
            return jsonify({'error': 'Service unavailable'}), 503
        
        if not details:
            return jsonify({'error': 'Content not found'}), 404
        
        return jsonify(details), 200
        
    except Exception as e:
        logger.error(f"Error getting details for slug {slug}: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

@app.route('/api/search', methods=['GET'])
@cache.cached(timeout=300, key_prefix=make_cache_key)
def search_content():
    try:
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
        
        tmdb_results = None
        anime_results = None
        
        try:
            tmdb_results = futures[0].result(timeout=5)
        except Exception as e:
            logger.warning(f"TMDB search timeout/error: {e}")
        
        if len(futures) > 1:
            try:
                anime_results = futures[1].result(timeout=5)
            except Exception as e:
                logger.warning(f"Anime search timeout/error: {e}")
        
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
                        logger.warning(f"Failed to record interaction: {e}")
                    
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
            logger.warning(f"Failed to commit search interactions: {e}")
            db.session.rollback()
        
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
                logger.warning(f"Failed to ensure slug: {e}")
        
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
            logger.warning(f"Failed to record view interaction: {e}")
        
        additional_details = None
        cast = []
        crew = []
        
        try:
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
        except Exception as e:
            logger.warning(f"Failed to get additional details: {e}")
        
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
                            similar.slug = f"content-{similar.id}"
                    
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
            logger.warning(f"Failed to get similar content: {e}")
        
        try:
            db.session.commit()
        except Exception as e:
            logger.warning(f"Failed to commit view interaction: {e}")
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
            'is_critics_choice': content.is_critics_choice
        }
        
        if content.content_type == 'anime':
            response_data['anime_genres'] = json.loads(content.anime_genres or '[]')
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

@app.route('/api/recommendations/trending', methods=['GET'])
@cache.cached(timeout=300, key_prefix=make_cache_key)
def get_trending():
    try:
        category = request.args.get('category', 'all')
        limit = int(request.args.get('limit', 10))
        region = request.args.get('region', 'IN')
        apply_language_priority = request.args.get('language_priority', 'true').lower() == 'true'
        
        all_content = []
        
        try:
            tmdb_movies = TMDBService.get_trending('movie', 'day')
            if tmdb_movies:
                for item in tmdb_movies.get('results', []):
                    content = content_service.save_content_from_tmdb(item, 'movie')
                    if content:
                        all_content.append(content)
            
            tmdb_tv = TMDBService.get_trending('tv', 'day')
            if tmdb_tv:
                for item in tmdb_tv.get('results', []):
                    content = content_service.save_content_from_tmdb(item, 'tv')
                    if content:
                        all_content.append(content)
        except Exception as e:
            logger.error(f"TMDB fetch error: {e}")
        
        try:
            top_anime = JikanService.get_top_anime()
            if top_anime:
                for anime in top_anime.get('data', [])[:20]:
                    content = content_service.save_anime_content(anime)
                    if content:
                        all_content.append(content)
        except Exception as e:
            logger.error(f"Jikan fetch error: {e}")
        
        db_trending = Content.query.filter_by(is_trending=True).limit(50).all()
        all_content.extend(db_trending)
        
        seen_ids = set()
        unique_content = []
        for content in all_content:
            if content.id not in seen_ids:
                seen_ids.add(content.id)
                if not content.slug:
                    try:
                        content.ensure_slug()
                    except Exception as e:
                        logger.warning(f"Failed to ensure slug for content {content.id}: {e}")
                        content.slug = f"content-{content.id}"
                unique_content.append(content)
        
        categories = recommendation_orchestrator.get_trending_with_algorithms(
            unique_content,
            limit=limit,
            region=region,
            apply_language_priority=apply_language_priority
        )
        
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
        
        try:
            db.session.commit()
        except Exception as e:
            logger.warning(f"Failed to commit trending updates: {e}")
            db.session.rollback()
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        logger.exception(e)
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/recommendations/new-releases', methods=['GET'])
@cache.cached(timeout=300, key_prefix=make_cache_key)
def get_new_releases():
    try:
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        
        all_new_releases = []
        priority_languages = ['telugu', 'english', 'hindi', 'malayalam', 'kannada', 'tamil']
        
        for language in priority_languages:
            lang_code = LANGUAGE_PRIORITY['codes'].get(language)
            
            try:
                if language == 'english':
                    releases = TMDBService.get_new_releases(content_type)
                else:
                    releases = TMDBService.get_language_specific(lang_code, content_type)
                
                if releases:
                    for item in releases.get('results', [])[:10]:
                        content = content_service.save_content_from_tmdb(item, content_type)
                        if content and content.release_date:
                            days_old = (datetime.now().date() - content.release_date).days
                            if days_old <= 60:
                                all_new_releases.append(content)
            except Exception as e:
                logger.error(f"Error fetching {language} releases: {e}")
        
        db_new_releases = Content.query.filter(
            Content.is_new_release == True,
            Content.content_type == content_type
        ).limit(50).all()
        all_new_releases.extend(db_new_releases)
        
        seen_ids = set()
        unique_releases = []
        for content in all_new_releases:
            if content.id not in seen_ids:
                seen_ids.add(content.id)
                if not content.slug:
                    try:
                        content.ensure_slug()
                    except Exception as e:
                        logger.warning(f"Failed to ensure slug for content {content.id}: {e}")
                        content.slug = f"content-{content.id}"
                unique_releases.append(content)
        
        recommendations = recommendation_orchestrator.get_new_releases_with_algorithms(
            unique_releases,
            limit=limit
        )
        
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
        
        if recommendations:
            content_ids = [r['id'] for r in recommendations]
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            
            response['metadata']['metrics'] = {
                'diversity_score': round(EvaluationMetrics.diversity_score(contents), 3),
                'telugu_content_percentage': round(
                    len(language_groups['telugu']) / len(recommendations) * 100, 1
                ) if recommendations else 0
            }
        
        try:
            db.session.commit()
        except Exception as e:
            logger.warning(f"Failed to commit new releases updates: {e}")
            db.session.rollback()
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"New releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

@app.route('/api/upcoming', methods=['GET'])
async def get_upcoming_releases():
    try:
        region = request.args.get('region', 'IN')
        timezone_name = request.args.get('timezone', 'Asia/Kolkata')
        categories_param = request.args.get('categories', 'movies,tv,anime')
        use_cache = request.args.get('use_cache', 'true').lower() == 'true'
        include_analytics = request.args.get('include_analytics', 'true').lower() == 'true'
        
        categories = [cat.strip() for cat in categories_param.split(',')]
        
        if len(region) != 2:
            return jsonify({'error': 'Invalid region code'}), 400
        
        service = UpcomingContentService(
            tmdb_api_key=TMDB_API_KEY,
            cache_backend=cache,
            enable_analytics=include_analytics
        )
        
        try:
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
    try:
        region = request.args.get('region', 'IN')
        timezone_name = request.args.get('timezone', 'Asia/Kolkata')
        categories_param = request.args.get('categories', 'movies,tv,anime')
        use_cache = request.args.get('use_cache', 'true').lower() == 'true'
        include_analytics = request.args.get('include_analytics', 'true').lower() == 'true'
        
        categories = [cat.strip() for cat in categories_param.split(',')]
        
        if len(region) != 2:
            return jsonify({'error': 'Invalid region code'}), 400
        
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
        
        critics_choice = TMDBService.get_critics_choice(content_type)
        
        recommendations = []
        if critics_choice:
            for item in critics_choice.get('results', [])[:limit]:
                content = content_service.save_content_from_tmdb(item, content_type)
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
        
        genre_ids = {
            'action': 28, 'adventure': 12, 'animation': 16, 'biography': -1,
            'comedy': 35, 'crime': 80, 'documentary': 99, 'drama': 18,
            'fantasy': 14, 'horror': 27, 'musical': 10402, 'mystery': 9648,
            'romance': 10749, 'sci-fi': 878, 'science fiction': 878, 'thriller': 53, 'western': 37
        }
        
        genre_id = genre_ids.get(genre.lower())
        recommendations = []
        
        if genre_id and genre_id != -1:
            genre_content = TMDBService.get_by_genre(genre_id, content_type, region=region)
            
            if genre_content:
                for item in genre_content.get('results', [])[:limit]:
                    content = content_service.save_content_from_tmdb(item, content_type)
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
        
        lang_code = LANGUAGE_PRIORITY['codes'].get(language.lower())
        recommendations = []
        
        if lang_code:
            lang_content = TMDBService.get_language_specific(lang_code, content_type)
            if lang_content:
                for item in lang_content.get('results', [])[:limit]:
                    content = content_service.save_content_from_tmdb(item, content_type)
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
        genre = request.args.get('genre')
        limit = int(request.args.get('limit', 20))
        
        recommendations = []
        
        if genre and genre.lower() in ANIME_GENRES:
            genre_keywords = ANIME_GENRES[genre.lower()]
            for keyword in genre_keywords[:2]:
                anime_results = JikanService.get_anime_by_genre(keyword)
                if anime_results:
                    for anime in anime_results.get('data', []):
                        if len(recommendations) >= limit:
                            break
                        content = content_service.save_anime_content(anime)
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
            top_anime = JikanService.get_top_anime()
            if top_anime:
                for anime in top_anime.get('data', [])[:limit]:
                    content = content_service.save_anime_content(anime)
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
    try:
        limit = min(int(request.args.get('limit', 8)), 15)
        strict_mode = request.args.get('strict_mode', 'false').lower() == 'true'
        min_similarity = float(request.args.get('min_similarity', 0.3))
        
        cache_key = f"similar:{content_id}:{limit}:{strict_mode}:{min_similarity}"
        if cache:
            try:
                cached_result = cache.get(cache_key)
                if cached_result:
                    return jsonify(cached_result), 200
            except Exception as e:
                logger.warning(f"Cache get error: {e}")
        
        base_content = Content.query.get(content_id)
        if not base_content:
            return jsonify({'error': 'Content not found'}), 404
        
        if not base_content.slug:
            try:
                base_content.ensure_slug()
                db.session.commit()
            except Exception as e:
                logger.warning(f"Slug generation failed for base content: {e}")
                base_content.slug = f"content-{base_content.id}"
        
        similar_content = []
        
        try:
            try:
                base_genres = json.loads(base_content.genres or '[]')
            except (json.JSONDecodeError, TypeError):
                base_genres = []
            
            if base_genres:
                primary_genre = base_genres[0]
                
                similar_items = Content.query.filter(
                    Content.id != content_id,
                    Content.content_type == base_content.content_type,
                    Content.genres.contains(primary_genre)
                ).order_by(
                    Content.rating.desc()
                ).limit(limit * 2).all()
                
                for item in similar_items[:limit]:
                    try:
                        if not item.slug:
                            item.slug = f"content-{item.id}"
                        
                        try:
                            item_genres = json.loads(item.genres or '[]')
                        except (json.JSONDecodeError, TypeError):
                            item_genres = []
                        
                        similar_content.append({
                            'id': item.id,
                            'slug': item.slug,
                            'title': item.title,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{item.poster_path}" if item.poster_path and not item.poster_path.startswith('http') else item.poster_path,
                            'rating': item.rating,
                            'content_type': item.content_type,
                            'genres': item_genres,
                            'similarity_score': 0.8,
                            'match_type': 'genre_based'
                        })
                        
                        if len(similar_content) >= limit:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error processing similar item {item.id}: {e}")
                        continue
            
            if not similar_content:
                fallback_items = Content.query.filter(
                    Content.id != content_id,
                    Content.content_type == base_content.content_type
                ).order_by(
                    Content.popularity.desc()
                ).limit(limit).all()
                
                for item in fallback_items:
                    if not item.slug:
                        item.slug = f"content-{item.id}"
                    
                    similar_content.append({
                        'id': item.id,
                        'slug': item.slug,
                        'title': item.title,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{item.poster_path}" if item.poster_path and not item.poster_path.startswith('http') else item.poster_path,
                        'rating': item.rating,
                        'content_type': item.content_type,
                        'similarity_score': 0.5,
                        'match_type': 'popularity_fallback'
                    })
        
        except Exception as e:
            logger.error(f"Error in similarity calculation: {e}")
            similar_content = []
        
        try:
            session_id = get_session_id()
            interaction = AnonymousInteraction(
                session_id=session_id,
                content_id=content_id,
                interaction_type='similar_view',
                ip_address=request.remote_addr
            )
            db.session.add(interaction)
            db.session.commit()
        except Exception as e:
            logger.warning(f"Interaction tracking failed: {e}")
        
        response = {
            'base_content': {
                'id': base_content.id,
                'slug': base_content.slug or f"content-{base_content.id}",
                'title': base_content.title,
                'content_type': base_content.content_type,
                'rating': base_content.rating
            },
            'similar_content': similar_content,
            'metadata': {
                'algorithm': 'optimized_genre_based',
                'total_results': len(similar_content),
                'similarity_threshold': min_similarity,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        if cache:
            try:
                cache.set(cache_key, response, timeout=900)
            except Exception as e:
                logger.warning(f"Caching failed: {e}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Similar recommendations error: {e}")
        return jsonify({
            'error': 'Failed to get similar recommendations',
            'similar_content': [],
            'metadata': {'error': str(e)}
        }), 500

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
                if not content.slug:
                    try:
                        content.ensure_slug()
                    except Exception as e:
                        logger.warning(f"Failed to ensure slug for admin rec content: {e}")
                        content.slug = f"content-{content.id}"
                
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
            logger.error("Details service not available")
            return jsonify({'error': 'Service unavailable'}), 503
        
        if not person_details:
            return jsonify({'error': 'Person not found'}), 404
        
        return jsonify(person_details), 200
        
    except Exception as e:
        logger.error(f"Error getting person details for slug {slug}: {e}")
        return jsonify({'error': 'Failed to get person details'}), 500

@app.route('/api/details/<slug>/reviews', methods=['POST'])
@auth_required
def add_review(slug):
    try:
        content = Content.query.filter_by(slug=slug).first()
        if not content:
            return jsonify({'error': 'Content not found'}), 404
        
        user_id = request.user_id
        review_data = request.json
        
        if details_service:
            result = details_service.add_review(content.id, user_id, review_data)
        else:
            return jsonify({'error': 'Service unavailable'}), 503
        
        if result['success']:
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error adding review: {e}")
        return jsonify({'error': 'Failed to add review'}), 500

@app.route('/api/reviews/<int:review_id>/helpful', methods=['POST'])
@auth_required
def vote_review_helpful(review_id):
    try:
        user_id = request.user_id
        is_helpful = request.json.get('is_helpful', True)
        
        if details_service:
            success = details_service.vote_review_helpful(review_id, user_id, is_helpful)
        else:
            return jsonify({'error': 'Service unavailable'}), 503
        
        if success:
            return jsonify({'success': True}), 200
        else:
            return jsonify({'error': 'Failed to vote'}), 400
            
    except Exception as e:
        logger.error(f"Error voting on review: {e}")
        return jsonify({'error': 'Failed to vote'}), 500

@app.route('/api/admin/slugs/migrate', methods=['POST'])
@auth_required
def migrate_all_slugs():
    try:
        user = User.query.get(request.user_id)
        if not user or not user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        if details_service:
            batch_size = int(request.json.get('batch_size', 50))
            stats = details_service.migrate_all_slugs(batch_size)
            return jsonify({
                'success': True,
                'migration_stats': stats
            }), 200
        else:
            return jsonify({'error': 'Service unavailable'}), 503
            
    except Exception as e:
        logger.error(f"Error migrating slugs: {e}")
        return jsonify({'error': 'Failed to migrate slugs'}), 500

@app.route('/api/admin/content/<int:content_id>/slug', methods=['PUT'])
@auth_required
def update_content_slug(content_id):
    try:
        user = User.query.get(request.user_id)
        if not user or not user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        if details_service:
            force_update = request.json.get('force_update', False)
            new_slug = details_service.update_content_slug(content_id, force_update)
            
            if new_slug:
                return jsonify({
                    'success': True,
                    'new_slug': new_slug
                }), 200
            else:
                return jsonify({'error': 'Content not found or update failed'}), 404
        else:
            return jsonify({'error': 'Service unavailable'}), 503
            
    except Exception as e:
        logger.error(f"Error updating content slug: {e}")
        return jsonify({'error': 'Failed to update slug'}), 500

@app.route('/api/content/<int:content_id>/refresh-cast-crew', methods=['POST'])
def refresh_cast_crew(content_id):
    try:
        content = Content.query.get_or_404(content_id)
        
        if not content.tmdb_id:
            return jsonify({'error': 'No TMDB ID available'}), 400
        
        if details_service:
            cast_crew = details_service._fetch_and_save_all_cast_crew(content)
            return jsonify({
                'success': True,
                'cast_count': len(cast_crew['cast']),
                'crew_count': sum(len(crew_list) for crew_list in cast_crew['crew'].values())
            })
        else:
            return jsonify({'error': 'Details service not available'}), 503
            
    except Exception as e:
        logger.error(f"Error refreshing cast/crew: {e}")
        return jsonify({'error': 'Failed to refresh cast/crew data'}), 500

@app.route('/api/admin/populate-cast-crew', methods=['POST'])
@auth_required
def populate_all_cast_crew():
    try:
        user = User.query.get(request.user_id)
        if not user or not user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
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
                    logger.info(f"Populated cast/crew for {content.title}")
            except Exception as e:
                logger.error(f"Error processing {content.title}: {e}")
                errors += 1
        
        return jsonify({
            'success': True,
            'processed': processed,
            'errors': errors,
            'total_available': len(content_items),
            'message': f"Successfully populated cast/crew for {processed} content items"
        }), 200
        
    except Exception as e:
        logger.error(f"Error in bulk cast/crew population: {e}")
        return jsonify({'error': 'Failed to populate cast/crew'}), 500

@app.route('/api/performance', methods=['GET'])
def performance_check():
    try:
        total_content = Content.query.count()
        content_with_slugs = Content.query.filter(
            and_(Content.slug != None, Content.slug != '')
        ).count()
        
        stats = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'database': {
                'total_content': total_content,
                'content_with_slugs': content_with_slugs,
                'content_without_slugs': total_content - content_with_slugs,
                'slug_coverage': round((content_with_slugs / total_content * 100), 2) if total_content > 0 else 0
            },
            'cache_type': app.config.get('CACHE_TYPE', 'unknown'),
            'services': {
                'details_service': 'enabled' if details_service else 'disabled',
                'content_service': 'enabled' if content_service else 'disabled'
            },
            'performance': {
                'thread_pool_workers': 3,
                'api_timeouts': '5s',
                'cache_timeout': '15min-30min',
                'optimization_level': 'high'
            }
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Performance check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        health_info = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '5.2.0',
            'python_version': '3.13.4'
        }
        
        try:
            db.session.execute(text('SELECT 1'))
            health_info['database'] = 'connected'
        except:
            health_info['database'] = 'disconnected'
            health_info['status'] = 'degraded'
        
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
        
        health_info['services'] = {
            'tmdb': bool(TMDB_API_KEY),
            'omdb': bool(OMDB_API_KEY),
            'youtube': bool(YOUTUBE_API_KEY),
            'algorithms': 'optimized_enabled',
            'slug_support': 'comprehensive_enabled',
            'details_service': 'enabled' if details_service else 'disabled',
            'content_service': 'enabled' if content_service else 'disabled',
            'cast_crew': 'fully_enabled'
        }
        
        try:
            total_content = Content.query.count()
            content_with_slugs = Content.query.filter(
                and_(Content.slug != None, Content.slug != '')
            ).count()
            
            health_info['slug_status'] = {
                'total_content': total_content,
                'with_slugs': content_with_slugs,
                'without_slugs': total_content - content_with_slugs,
                'coverage_percentage': round((content_with_slugs / total_content * 100), 2) if total_content > 0 else 0
            }
        except Exception as e:
            health_info['slug_status'] = {'error': str(e)}
        
        try:
            total_content_persons = ContentPerson.query.count()
            total_persons = Person.query.count()
            
            health_info['cast_crew_status'] = {
                'total_relations': total_content_persons,
                'total_persons': total_persons,
                'content_with_cast': db.session.query(ContentPerson.content_id).distinct().count()
            }
        except Exception as e:
            health_info['cast_crew_status'] = {'error': str(e)}
        
        health_info['performance'] = {
            'optimizations_applied': [
                'python_3.13_compatibility',
                'reduced_api_timeouts', 
                'optimized_thread_pools',
                'enhanced_caching',
                'error_handling_improvements',
                'cast_crew_optimization'
            ],
            'memory_optimizations': 'enabled',
            'unicode_fixes': 'applied'
        }
        
        return jsonify(health_info), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.cli.command('generate-slugs')
def generate_slugs():
    try:
        print("Starting comprehensive slug generation...")
        
        if not details_service:
            print("Error: Details service not available")
            return
        
        stats = details_service.migrate_all_slugs(batch_size=50)
        
        print("Slug migration completed successfully!")
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
            
            print(f"\nVerification Results:")
            print(f"Content: {content_with_slugs}/{total_content} ({round(content_with_slugs/total_content*100, 1)}% coverage)")
            print(f"Persons: {persons_with_slugs}/{total_persons} ({round(persons_with_slugs/total_persons*100, 1)}% coverage)")
            
        except Exception as e:
            print(f"Error during verification: {e}")
        
    except Exception as e:
        print(f"Failed to generate slugs: {e}")
        logger.error(f"CLI slug generation error: {e}")

@app.cli.command('populate-cast-crew')
def populate_cast_crew_cli():
    try:
        print("Starting cast/crew population...")
        
        if not details_service:
            print("Error: Details service not available")
            return
        
        content_items = Content.query.filter(
            Content.tmdb_id.isnot(None),
            ~Content.id.in_(
                db.session.query(ContentPerson.content_id).distinct()
            )
        ).all()
        
        print(f"Found {len(content_items)} content items without cast/crew")
        
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
        
        print(f"\nCast/crew population completed!")
        print(f"Processed: {processed}")
        print(f"Errors: {errors}")
        
    except Exception as e:
        print(f"Failed to populate cast/crew: {e}")
        logger.error(f"CLI cast/crew population error: {e}")

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
                    is_admin=True
                )
                db.session.add(admin)
                db.session.commit()
                logger.info("Admin user created with username: admin, password: admin123")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

create_tables()

if __name__ == '__main__':
    print("=== Running Flask in development mode ===")
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
else:
    print("=== Flask app imported by Gunicorn - OPTIMIZED VERSION WITH CAST/CREW ===")
    print(f"App name: {app.name}")
    print(f"Python version: 3.13.4")
    print(f"Database URI configured: {'Yes' if app.config.get('SQLALCHEMY_DATABASE_URI') else 'No'}")
    print(f"Cache type: {app.config.get('CACHE_TYPE', 'Not configured')}")
    print(f"Details service status: {'Initialized' if details_service else 'Failed to initialize'}")
    print(f"Content service status: {'Initialized' if content_service else 'Failed to initialize'}")
    print(f"Performance optimizations: Applied")
    print(f"Unicode fixes: Applied")
    print(f"Cast/Crew support: Fully enabled")
    
    print("\n=== Registered Routes ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.rule} [{', '.join(rule.methods)}]")
    print("=== End of Routes ===\n")