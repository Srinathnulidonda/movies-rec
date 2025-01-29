# backend/app.py

from typing import Optional
from flask import Flask, request, jsonify, session, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_caching import Cache
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from sqlalchemy import func, and_, or_, desc, text
from sqlalchemy.dialects.postgresql import ENUM
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
from admin import admin_bp, init_admin
from support import support_bp, init_support
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
from reviews import init_reviews_service
import re
import click
import traceback
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
DATABASE_URL = os.environ.get('DATABASE_URL')

# Database Configuration - Fixed
if DATABASE_URL:
    # Handle both postgres:// and postgresql:// formats
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
else:
    raise ValueError("DATABASE_URL environment variable is required")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Redis Configuration - Fixed
REDIS_URL = os.environ.get('REDIS_URL')
if REDIS_URL and REDIS_URL.startswith(('redis://', 'rediss://')):
    app.config['CACHE_TYPE'] = 'redis'
    app.config['CACHE_REDIS_URL'] = REDIS_URL
    app.config['CACHE_DEFAULT_TIMEOUT'] = 3600
else:
    app.config['CACHE_TYPE'] = 'simple'
    app.config['CACHE_DEFAULT_TIMEOUT'] = 1800

# PostgreSQL SSL Configuration - Fixed
if DATABASE_URL and 'postgresql://' in DATABASE_URL:
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_timeout': 20,
        'max_overflow': 0,
        'connect_args': {
            'sslmode': 'prefer',  # Changed from 'require' to 'prefer'
            'connect_timeout': 10
        }
    }

db = SQLAlchemy(app)

# CORS Configuration - Updated with specific origins and credentials support
CORS(app, origins=[
    'https://cinebrain.vercel.app',
    'http://127.0.0.1:5500', 
    'http://127.0.0.1:5501',
], supports_credentials=True)

cache = Cache(app)

# API Keys
TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY environment variable is required")
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY environment variable is required")

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
    secure=True
)

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

# Database Models - Fixed with String-based Enums
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

# Support System Models - Fixed to use strings instead of enums
class SupportCategory(db.Model):
    __tablename__ = 'support_categories'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.Text)
    icon = db.Column(db.String(50))
    sort_order = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    tickets = db.relationship('SupportTicket', backref='category', lazy='dynamic')

class SupportTicket(db.Model):
    __tablename__ = 'support_tickets'
    
    id = db.Column(db.Integer, primary_key=True)
    ticket_number = db.Column(db.String(20), unique=True, nullable=False)
    subject = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    user_email = db.Column(db.String(255), nullable=False)
    user_name = db.Column(db.String(255), nullable=False)
    
    category_id = db.Column(db.Integer, db.ForeignKey('support_categories.id'), nullable=False)
    
    # Fixed: Use string columns instead of enums
    ticket_type = db.Column(db.String(20), nullable=False, default='general')
    priority = db.Column(db.String(20), default='normal')
    status = db.Column(db.String(20), default='open')
    
    browser_info = db.Column(db.Text)
    device_info = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    page_url = db.Column(db.String(500))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    first_response_at = db.Column(db.DateTime)
    resolved_at = db.Column(db.DateTime)
    closed_at = db.Column(db.DateTime)
    sla_deadline = db.Column(db.DateTime)
    sla_breached = db.Column(db.Boolean, default=False)
    
    assigned_to = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    activities = db.relationship('TicketActivity', backref='ticket', lazy='dynamic', cascade='all, delete-orphan')

class TicketActivity(db.Model):
    __tablename__ = 'ticket_activities'
    
    id = db.Column(db.Integer, primary_key=True)
    ticket_id = db.Column(db.Integer, db.ForeignKey('support_tickets.id'), nullable=False)
    
    action = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    old_value = db.Column(db.Text)
    new_value = db.Column(db.Text)
    
    actor_type = db.Column(db.String(20), nullable=False)
    actor_id = db.Column(db.Integer, nullable=True)
    actor_name = db.Column(db.String(255))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ContactMessage(db.Model):
    __tablename__ = 'contact_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    subject = db.Column(db.String(255), nullable=False)
    message = db.Column(db.Text, nullable=False)
    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    phone = db.Column(db.String(20))
    company = db.Column(db.String(255))
    
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    page_url = db.Column(db.String(500))
    
    is_read = db.Column(db.Boolean, default=False)
    admin_notes = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class IssueReport(db.Model):
    __tablename__ = 'issue_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    issue_id = db.Column(db.String(100), unique=True, nullable=False)
    
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    issue_type = db.Column(db.String(50), nullable=False)
    severity = db.Column(db.String(20), nullable=False)
    issue_title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    
    browser_version = db.Column(db.String(255))
    device_os = db.Column(db.String(255))
    page_url_reported = db.Column(db.String(500))
    steps_to_reproduce = db.Column(db.Text)
    expected_behavior = db.Column(db.Text)
    
    screenshots = db.Column(db.JSON)
    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    ticket_id = db.Column(db.Integer, db.ForeignKey('support_tickets.id'), nullable=True)
    
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    page_url = db.Column(db.String(500))
    
    is_resolved = db.Column(db.Boolean, default=False)
    admin_notes = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime)

# Admin System Models - Fixed
class AdminNotification(db.Model):
    __tablename__ = 'admin_notifications'
    
    id = db.Column(db.Integer, primary_key=True)
    notification_type = db.Column(db.String(50), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    message = db.Column(db.Text, nullable=False)
    
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    related_ticket_id = db.Column(db.Integer, nullable=True)
    related_content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=True)
    
    is_read = db.Column(db.Boolean, default=False)
    is_urgent = db.Column(db.Boolean, default=False)
    action_required = db.Column(db.Boolean, default=False)
    action_url = db.Column(db.String(500))
    
    notification_metadata = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    read_at = db.Column(db.DateTime)

class CannedResponse(db.Model):
    __tablename__ = 'canned_responses'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    
    category_id = db.Column(db.Integer, nullable=True)
    tags = db.Column(db.JSON)
    
    is_active = db.Column(db.Boolean, default=True)
    usage_count = db.Column(db.Integer, default=0)
    
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SupportMetrics(db.Model):
    __tablename__ = 'support_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    
    tickets_created = db.Column(db.Integer, default=0)
    tickets_resolved = db.Column(db.Integer, default=0)
    tickets_closed = db.Column(db.Integer, default=0)
    
    avg_first_response_time = db.Column(db.Float)
    avg_resolution_time = db.Column(db.Float)
    
    sla_breaches = db.Column(db.Integer, default=0)
    escalations = db.Column(db.Integer, default=0)
    
    customer_satisfaction = db.Column(db.Float)
    feedback_count = db.Column(db.Integer, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

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

# Updated models dictionary
models = {
    'User': User,
    'Content': Content,
    'UserInteraction': UserInteraction,
    'AdminRecommendation': AdminRecommendation,
    'Review': Review,
    'Person': Person,
    'ContentPerson': ContentPerson,
    'AnonymousInteraction': AnonymousInteraction,
    'SupportCategory': SupportCategory,
    'SupportTicket': SupportTicket,
    'TicketActivity': TicketActivity,
    'ContactMessage': ContactMessage,
    'IssueReport': IssueReport,
    'AdminNotification': AdminNotification,
    'CannedResponse': CannedResponse,
    'SupportMetrics': SupportMetrics
}

# Initialize services
details_service = None
content_service = None
cinebrain_new_releases_service = None
email_service = None
admin_notification_service = None

# Details and Content Services
try:
    with app.app_context():
        details_service = init_details_service(app, db, models, cache)
        content_service = ContentService(db, models)
        logger.info("✅ CineBrain details and content services initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain details/content services: {e}")

services = {
    'TMDBService': CineBrainTMDBService,
    'JikanService': CineBrainJikanService,
    'ContentService': content_service,
    'http_session': http_session,
    'cache': cache
}

# Initialize Reviews System
try:
    review_services = init_reviews_service(app, db, models, cache)
    if review_services:
        logger.info("✅ CineBrain Reviews system initialized successfully")
        services.update(review_services)
    else:
        logger.warning("⚠️ CineBrain Reviews system failed to initialize")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain Reviews system: {e}")

# Initialize New Releases Service
try:
    cinebrain_new_releases_service = init_cinebrain_new_releases_service(app, db, models, services)
    if cinebrain_new_releases_service:
        logger.info("✅ CineBrain new releases service integrated successfully")
        services['new_releases_service'] = cinebrain_new_releases_service
    else:
        logger.warning("⚠️ CineBrain new releases service failed to initialize")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain new releases service: {e}")

# Initialize Authentication System - FIXED: Removed url_prefix
try:
    from auth.service import init_auth, email_service as auth_email_service
    from auth.routes import auth_bp
    
    init_auth(app, db, User)
    app.register_blueprint(auth_bp)  # FIXED: No url_prefix
    
    email_service = auth_email_service
    services['email_service'] = email_service
    
    logger.info("✅ CineBrain authentication service with Brevo email initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain authentication service: {e}")

# Initialize Support System - FIXED: Removed url_prefix
try:
    if 'email_service' not in services and email_service:
        services['email_service'] = email_service
    
    support_models = init_support(app, db, models, services)
    app.register_blueprint(support_bp)  # FIXED: No url_prefix
    
    if support_models:
        logger.info("✅ CineBrain Support System initialized successfully")
        models.update(support_models)
        services['support_models'] = support_models
    else:
        logger.warning("⚠️ CineBrain Support System failed to initialize")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain Support System: {e}")

# Initialize Admin System - FIXED: Removed url_prefix
try:
    if email_service:
        services['email_service'] = email_service
    
    admin_models = init_admin(app, db, models, services)
    app.register_blueprint(admin_bp)  # FIXED: No url_prefix
    
    if 'admin_notification_service' not in services:
        try:
            from admin.service import AdminNotificationService
            admin_notification_service = AdminNotificationService(app, db, models, services)
            services['admin_notification_service'] = admin_notification_service
            app.admin_notification_service = admin_notification_service
            logger.info("✅ Admin notification service stored for monitoring")
        except Exception as e:
            logger.error(f"❌ Failed to create admin notification service: {e}")
    
    logger.info("✅ CineBrain admin service initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain admin service: {e}")

# Initialize Personalized System
try:
    profile_analyzer, personalized_recommendation_engine = init_personalized_system(
        app, db, models, services, cache
    )
    
    if profile_analyzer and personalized_recommendation_engine:
        logger.info("✅ CineBrain Advanced Personalized Recommendation System initialized successfully")
        services['profile_analyzer'] = profile_analyzer
        services['personalized_recommendation_engine'] = personalized_recommendation_engine
        app.register_blueprint(personalized_bp)  # FIXED: No url_prefix
        logger.info("✅ CineBrain Personalized routes registered")
    else:
        logger.warning("⚠️ CineBrain Personalized Recommendation System failed to initialize fully")
        
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain Personalized Recommendation System: {e}")

# Initialize User Routes
try:
    init_user_routes(app, db, models, {**services, 'cache': cache})
    app.register_blueprint(user_bp)  # FIXED: No url_prefix
    logger.info("✅ CineBrain user module initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain user module: {e}")

# Initialize Critics Choice Service
try:
    critics_choice_service = init_critics_choice_service(app, db, models, services, cache)
    app.register_blueprint(critics_choice_bp)  # FIXED: No url_prefix
    if critics_choice_service:
        logger.info("✅ CineBrain Critics Choice service integrated successfully")
        services['critics_choice_service'] = critics_choice_service
    else:
        logger.warning("⚠️ CineBrain Critics Choice service failed to initialize")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain Critics Choice service: {e}")

# Add RecommendationOrchestrator
try:
    cinebrain_recommendation_orchestrator = RecommendationOrchestrator()
    services['recommendation_orchestrator'] = cinebrain_recommendation_orchestrator
    logger.info("✅ CineBrain RecommendationOrchestrator initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain RecommendationOrchestrator: {e}")

# Initialize Recommendation Routes
try:
    init_recommendation_routes(app, db, models, services, cache)
    app.register_blueprint(recommendation_bp)  # FIXED: No url_prefix
    logger.info("✅ CineBrain recommendation routes initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain recommendation routes: {e}")

# Initialize System Routes
try:
    init_system_routes(app, db, models, services)
    app.register_blueprint(system_bp)  # FIXED: No url_prefix
    logger.info("✅ CineBrain system monitoring service initialized successfully")
    services['system_service'] = True
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain system monitoring service: {e}")

# Initialize Operations Routes
try:
    init_operations_routes(app, db, models, services)
    app.register_blueprint(operations_bp)  # FIXED: No url_prefix
    logger.info("✅ CineBrain operations service initialized successfully")
    services['operations_service'] = True
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain operations service: {e}")

def setup_support_monitoring():
    """Enhanced support monitoring - EMAIL ONLY, NO TELEGRAM - FIXED FOR STRINGS"""
    def support_monitor():
        while True:
            try:
                if not app or not db:
                    time.sleep(60)
                    continue
                
                with app.app_context():
                    try:
                        # FIXED: Use SQLAlchemy ORM instead of raw SQL to handle enum/string mismatch
                        current_time = datetime.utcnow()
                        
                        # Use ORM query instead of raw SQL
                        overdue_tickets = db.session.query(SupportTicket).filter(
                            and_(
                                SupportTicket.sla_deadline < current_time,
                                SupportTicket.sla_breached == False,
                                or_(
                                    SupportTicket.status == 'open',
                                    SupportTicket.status == 'in_progress'
                                )
                            )
                        ).all()
                        
                        for ticket in overdue_tickets:
                            # Update using ORM
                            ticket.sla_breached = True
                            
                            logger.warning(f"SLA breached for ticket #{ticket.ticket_number}")
                            
                            # ONLY EMAIL notification to admin - NO TELEGRAM
                            try:
                                if admin_notification_service:
                                    admin_notification_service.notify_sla_breach(ticket)
                            except Exception as e:
                                logger.error(f"Error handling SLA breach notification: {e}")
                        
                        if overdue_tickets:
                            db.session.commit()
                            logger.info(f"✅ Marked {len(overdue_tickets)} tickets as SLA breached")
                        
                    except Exception as e:
                        logger.error(f"❌ Support monitoring inner error: {e}")
                        db.session.rollback()
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"❌ Support monitoring error: {e}")
                time.sleep(300)
    
    monitor_thread = threading.Thread(target=support_monitor, daemon=True)
    monitor_thread.start()
    logger.info("✅ CineBrain support monitoring (EMAIL ONLY) started")

# Core API Routes
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

# Import and initialize CLI commands
from system.cli_commands import init_cli_commands

# Initialize CLI commands
try:
    cli_manager = init_cli_commands(app, db, models, services)
    if cli_manager:
        logger.info("✅ CineBrain CLI commands initialized successfully")
    else:
        logger.warning("⚠️ CineBrain CLI commands failed to initialize")
except Exception as e:
    logger.error(f"❌ Failed to initialize CineBrain CLI commands: {e}")

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
            
            # Create default support categories
            try:
                categories = [
                    {'name': 'Account & Login', 'description': 'Issues with account creation, login, password reset', 'icon': '👤'},
                    {'name': 'Technical Issues', 'description': 'App crashes, loading issues, performance problems', 'icon': '🔧'},
                    {'name': 'Features & Functions', 'description': 'How to use features, feature requests', 'icon': '⚡'},
                    {'name': 'Content & Recommendations', 'description': 'Issues with movies, shows, recommendations', 'icon': '🎬'},
                    {'name': 'General Support', 'description': 'Other questions and general inquiries', 'icon': '❓'}
                ]
                
                for i, cat_data in enumerate(categories):
                    existing = SupportCategory.query.filter_by(name=cat_data['name']).first()
                    if not existing:
                        category = SupportCategory(
                            name=cat_data['name'],
                            description=cat_data['description'],
                            icon=cat_data['icon'],
                            sort_order=i+1
                        )
                        db.session.add(category)
                
                db.session.commit()
                print("✅ Default support categories created")
            except Exception as e:
                logger.warning(f"Failed to create support categories: {e}")
            
            setup_support_monitoring()
            
            logger.info("✅ CineBrain database tables created successfully")
    except Exception as e:
        logger.error(f"❌ CineBrain database initialization error: {e}")

create_tables()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.rule} [{', '.join(rule.methods)}]")
    print("=== CINEBRAIN ===\n")