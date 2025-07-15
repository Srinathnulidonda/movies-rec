## Enhanced Backend/app.py
from flask import Flask, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os
import asyncio
import aiohttp
from functools import wraps
import json
from threading import Thread
import time
from flask_cors import CORS
import redis
import telegram
from telegram.error import TelegramError
from sqlalchemy import text, or_, and_, func, desc
from flask_caching import Cache
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, 
     origins=["http://127.0.0.1:5500", 
              "http://localhost:5500", 
              "https://movies-rec.vercel.app",
              "https://movies-frontend-jade.vercel.app",
              "https://backend-app-970m.onrender.com"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///movie_rec.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)
app.config['CACHE_TYPE'] = 'redis' if os.getenv('REDIS_URL') else 'simple'
app.config['CACHE_REDIS_URL'] = os.getenv('REDIS_URL')
app.config['CACHE_DEFAULT_TIMEOUT'] = 300

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)
cache = Cache(app)

# API Keys and URLs
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.getenv('OMDB_API_KEY', '52260795')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7689567537:AAGvDtu94OlLlTiWpfjSfpl_dd_Osi_2W7c')
TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID', '-1002566510721')

# Enhanced Genre Mapping
GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western", 10759: "Action & Adventure",
    10762: "Kids", 10763: "News", 10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap",
    10767: "Talk", 10768: "War & Politics"
}

REGIONAL_LANGUAGES = {
    'telugu': 'te', 'hindi': 'hi', 'tamil': 'ta', 'kannada': 'kn',
    'malayalam': 'ml', 'bengali': 'bn', 'punjabi': 'pa', 'gujarati': 'gu',
    'marathi': 'mr', 'urdu': 'ur'
}

# Enhanced Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    preferences = db.Column(db.JSON, default={})
    demographics = db.Column(db.JSON, default={})
    is_active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.String(50), unique=True)
    imdb_id = db.Column(db.String(50))
    title = db.Column(db.String(200), nullable=False, index=True)
    original_title = db.Column(db.String(200))
    overview = db.Column(db.Text)
    tagline = db.Column(db.String(500))
    genres = db.Column(db.JSON)
    language = db.Column(db.String(10), index=True)
    spoken_languages = db.Column(db.JSON)
    release_date = db.Column(db.Date, index=True)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float, index=True)
    vote_count = db.Column(db.Integer, default=0)
    poster_path = db.Column(db.String(200))
    backdrop_path = db.Column(db.String(200))
    content_type = db.Column(db.String(20), index=True)  # movie, tv, anime
    status = db.Column(db.String(20))  # released, upcoming, etc.
    production_companies = db.Column(db.JSON)
    production_countries = db.Column(db.JSON)
    budget = db.Column(db.BigInteger)
    revenue = db.Column(db.BigInteger)
    keywords = db.Column(db.JSON)
    cast = db.Column(db.JSON)
    crew = db.Column(db.JSON)
    videos = db.Column(db.JSON)
    meta_data = db.Column(db.JSON)
    popularity = db.Column(db.Float, default=0, index=True)
    trending_score = db.Column(db.Float, default=0)
    quality_score = db.Column(db.Float, default=0)
    content_embedding = db.Column(db.Text)  # Serialized embedding vector
    is_featured = db.Column(db.Boolean, default=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False, index=True)
    interaction_type = db.Column(db.String(20), nullable=False, index=True)  # view, like, favorite, wishlist, rating, watch_time
    rating = db.Column(db.Integer)
    watch_time = db.Column(db.Integer)  # in seconds
    watch_percentage = db.Column(db.Float)
    session_id = db.Column(db.String(100))
    device_type = db.Column(db.String(50))
    metadata = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = db.relationship('User', backref='interactions')
    content = db.relationship('Content', backref='interactions')

class UserSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    anonymous_id = db.Column(db.String(100))  # For anonymous users
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    preferences = db.Column(db.JSON, default={})
    viewed_content = db.Column(db.JSON, default=[])
    session_data = db.Column(db.JSON, default={})
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime)

class ContentSimilarity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    similar_content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    similarity_score = db.Column(db.Float, nullable=False)
    similarity_type = db.Column(db.String(50))  # content_based, collaborative, hybrid
    computed_at = db.Column(db.DateTime, default=datetime.utcnow)

class AdminRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'))
    category = db.Column(db.String(50))  # critics_choice, trending, featured, regional_spotlight
    priority = db.Column(db.Integer, default=1)
    region = db.Column(db.String(10))  # language/region specific
    target_demographics = db.Column(db.JSON)
    expires_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AdminPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'))
    admin_user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    custom_tags = db.Column(db.JSON)
    priority = db.Column(db.Integer, default=1)
    target_audience = db.Column(db.JSON)  # demographics, preferences
    post_to_website = db.Column(db.Boolean, default=True)
    post_to_telegram = db.Column(db.Boolean, default=False)
    telegram_message_id = db.Column(db.String(50))
    scheduled_for = db.Column(db.DateTime)
    expires_at = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    engagement_stats = db.Column(db.JSON, default={})
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    content = db.relationship('Content', backref='admin_posts')
    admin_user = db.relationship('User', backref='admin_posts')

class SystemAnalytics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(50), nullable=False)
    metric_value = db.Column(db.Float, nullable=False)
    dimensions = db.Column(db.JSON)  # Additional metadata
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)

class ContentTrending(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'))
    trending_score = db.Column(db.Float, nullable=False)
    time_window = db.Column(db.String(20))  # daily, weekly, monthly
    region = db.Column(db.String(10))
    genre = db.Column(db.String(50))
    computed_at = db.Column(db.DateTime, default=datetime.utcnow)

# Enhanced Content Aggregator Service
class EnhancedContentAggregator:
    def __init__(self):
        self.tmdb_base = "https://api.themoviedb.org/3"
        self.omdb_base = "http://www.omdbapi.com"
        self.jikan_base = "https://api.jikan.moe/v4"
        self.youtube_base = "https://www.googleapis.com/youtube/v3"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'MovieRecommendationApp/1.0'})
        
    async def fetch_with_retry(self, session, url, params, max_retries=3):
        """Fetch with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url}, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None
        return None
    
    async def fetch_trending_enhanced(self, content_type='movie', time_window='week', region=None):
        """Enhanced trending content with regional support"""
        url = f"{self.tmdb_base}/trending/{content_type}/{time_window}"
        params = {'api_key': TMDB_API_KEY}
        if region:
            params['region'] = region
        
        async with aiohttp.ClientSession() as session:
            data = await self.fetch_with_retry(session, url, params)
            return data.get('results', []) if data else []
    
    async def fetch_popular_by_genre_enhanced(self, genre_id, content_type='movie', region=None, year=None):
        """Enhanced popular content by genre with more filters"""
        url = f"{self.tmdb_base}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': genre_id,
            'sort_by': 'popularity.desc',
            'vote_count.gte': 100  # Minimum vote count for quality
        }
        
        if region:
            params['region'] = region
        if year:
            params['year'] = year
        
        async with aiohttp.ClientSession() as session:
            data = await self.fetch_with_retry(session, url, params)
            return data.get('results', []) if data else []
    
    async def fetch_regional_content_enhanced(self, language='hi', year=None, genre=None):
        """Enhanced regional content fetching"""
        url = f"{self.tmdb_base}/discover/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'with_original_language': language,
            'sort_by': 'popularity.desc',
            'vote_count.gte': 50
        }
        
        if year:
            params['year'] = year
        if genre:
            params['with_genres'] = genre
        
        async with aiohttp.ClientSession() as session:
            data = await self.fetch_with_retry(session, url, params)
            return data.get('results', []) if data else []
    
    async def get_content_complete_details(self, content_id, content_type='movie'):
        """Get complete content details including cast, crew, videos, etc."""
        url = f"{self.tmdb_base}/{content_type}/{content_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'append_to_response': 'credits,videos,similar,reviews,keywords,images,external_ids'
        }
        
        async with aiohttp.ClientSession() as session:
            return await self.fetch_with_retry(session, url, params)
    
    async def get_omdb_details(self, imdb_id):
        """Get additional details from OMDb"""
        if not OMDB_API_KEY or not imdb_id:
            return None
        
        url = self.omdb_base
        params = {
            'apikey': OMDB_API_KEY,
            'i': imdb_id,
            'plot': 'full'
        }
        
        async with aiohttp.ClientSession() as session:
            return await self.fetch_with_retry(session, url, params)
    
    async def search_youtube_trailers(self, title, year=None):
        """Search for trailers on YouTube"""
        if not YOUTUBE_API_KEY:
            return []
        
        query = f"{title} official trailer"
        if year:
            query += f" {year}"
        
        url = f"{self.youtube_base}/search"
        params = {
            'key': YOUTUBE_API_KEY,
            'q': query,
            'part': 'snippet',
            'type': 'video',
            'maxResults': 5,
            'order': 'relevance'
        }
        
        async with aiohttp.ClientSession() as session:
            data = await self.fetch_with_retry(session, url, params)
            return data.get('items', []) if data else []

# Enhanced Recommendation Engine
class EnhancedRecommendationEngine:
    def __init__(self):
        self.content_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2
        )
        self.content_matrix = None
        self.content_similarity = None
        self.user_item_matrix = None
        self.content_features_df = None
        
    def build_enhanced_content_matrix(self):
        """Build enhanced content similarity matrix"""
        contents = Content.query.all()
        if not contents:
            return
        
        # Create comprehensive content features
        features = []
        content_data = []
        
        for content in contents:
            # Text features
            text_features = []
            if content.title:
                text_features.append(content.title)
            if content.overview:
                text_features.append(content.overview)
            if content.tagline:
                text_features.append(content.tagline)
            
            # Genre features
            if content.genres:
                genre_names = []
                for genre_id in content.genres:
                    if isinstance(genre_id, int) and genre_id in GENRE_MAP:
                        genre_names.append(GENRE_MAP[genre_id])
                    elif isinstance(genre_id, str):
                        genre_names.append(genre_id)
                text_features.extend(genre_names * 3)  # Weight genres more
            
            # Cast and crew features
            if content.cast:
                cast_names = [person.get('name', '') for person in content.cast[:5]]
                text_features.extend(cast_names)
            
            if content.crew:
                directors = [person.get('name', '') for person in content.crew 
                           if person.get('job') == 'Director']
                text_features.extend(directors * 2)  # Weight directors more
            
            # Keywords
            if content.keywords:
                keyword_names = [kw.get('name', '') for kw in content.keywords]
                text_features.extend(keyword_names)
            
            feature_text = ' '.join(text_features)
            features.append(feature_text)
            
            # Store additional numeric features
            content_data.append({
                'id': content.id,
                'rating': content.rating or 0,
                'popularity': content.popularity or 0,
                'runtime': content.runtime or 0,
                'year': content.release_date.year if content.release_date else 0,
                'language': content.language or 'unknown'
            })
        
        # Build text similarity matrix
        if features:
            self.content_matrix = self.content_vectorizer.fit_transform(features)
            self.content_similarity = cosine_similarity(self.content_matrix)
            self.content_features_df = pd.DataFrame(content_data)
        
    def calculate_diversity_score(self, recommendations):
        """Calculate diversity of recommendations"""
        if not recommendations or len(recommendations) < 2:
            return 1.0
        
        genres_set = set()
        languages_set = set()
        years_set = set()
        
        for content in recommendations:
            if content.genres:
                genres_set.update(content.genres)
            if content.language:
                languages_set.add(content.language)
            if content.release_date:
                years_set.add(content.release_date.year)
        
        # Normalize diversity scores
        genre_diversity = min(len(genres_set) / 10, 1.0)
        language_diversity = min(len(languages_set) / 5, 1.0)
        year_diversity = min(len(years_set) / 10, 1.0)
        
        return (genre_diversity + language_diversity + year_diversity) / 3
    
    def get_ml_enhanced_recommendations(self, user_id, limit=20):
        """Get recommendations from ML service with fallback"""
        try:
            response = requests.post(
                f"{ML_SERVICE_URL}/recommend",
                json={'user_id': user_id, 'limit': limit},
                timeout=5
            )
            if response.status_code == 200:
                ml_data = response.json()
                content_ids = ml_data.get('recommendations', [])
                
                # Fetch content objects
                contents = []
                for content_id in content_ids:
                    content = Content.query.get(content_id)
                    if content:
                        contents.append(content)
                
                return contents
        except Exception as e:
            logger.warning(f"ML service error: {e}")
        
        # Fallback to local recommendations
        return self.get_hybrid_recommendations(user_id, limit)
    
    def get_trending_recommendations(self, limit=20, region=None, time_window='week'):
        """Get trending recommendations"""
        query = Content.query.filter(Content.popularity > 0)
        
        if region:
            query = query.filter(Content.language == region)
        
        # Weight recent content more
        recent_threshold = datetime.utcnow() - timedelta(days=365)
        
        trending_content = query.filter(
            or_(
                Content.release_date >= recent_threshold,
                Content.popularity > 50
            )
        ).order_by(
            desc(Content.popularity),
            desc(Content.rating)
        ).limit(limit).all()
        
        return trending_content
    
    def get_personalized_homepage(self, user_id=None, session_data=None, limit=50):
        """Generate personalized homepage recommendations"""
        if user_id:
            # Logged-in user recommendations
            user = User.query.get(user_id)
            preferences = user.preferences or {}
            
            # Get user interaction history
            recent_interactions = UserInteraction.query.filter_by(
                user_id=user_id
            ).filter(
                UserInteraction.created_at >= datetime.utcnow() - timedelta(days=30)
            ).all()
            
            # Analyze user preferences
            preferred_genres = set()
            preferred_languages = set()
            
            for interaction in recent_interactions:
                content = Content.query.get(interaction.content_id)
                if content:
                    if content.genres:
                        preferred_genres.update(content.genres)
                    if content.language:
                        preferred_languages.add(content.language)
            
            # Get ML recommendations
            ml_recs = self.get_ml_enhanced_recommendations(user_id, limit//2)
            
            # Get content-based recommendations
            content_recs = self.get_content_based_recommendations(user_id, limit//4)
            
            # Get genre-based recommendations
            genre_recs = []
            for genre in list(preferred_genres)[:3]:
                genre_content = Content.query.filter(
                    Content.genres.contains([genre])
                ).order_by(desc(Content.popularity)).limit(5).all()
                genre_recs.extend(genre_content)
            
            # Combine and diversify
            all_recs = ml_recs + content_recs + genre_recs
            
        else:
            # Anonymous user recommendations
            all_recs = self.get_trending_recommendations(limit)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recs = []
        for rec in all_recs:
            if rec.id not in seen:
                seen.add(rec.id)
                unique_recs.append(rec)
        
        return unique_recs[:limit]

# Initialize enhanced services
aggregator = EnhancedContentAggregator()
recommender = EnhancedRecommendationEngine()

# Enhanced caching and rate limiting
class EnhancedRateLimiter:
    def __init__(self):
        try:
            self.redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
            self.redis_client.ping()
        except:
            self.redis_client = None
            logger.warning("Redis not available, rate limiting disabled")
    
    def is_allowed(self, key, limit=100, window=3600):
        if not self.redis_client:
            return True
        
        try:
            current = self.redis_client.get(key)
            if current is None:
                self.redis_client.setex(key, window, 1)
                return True
            elif int(current) < limit:
                self.redis_client.incr(key)
                return True
            return False
        except:
            return True

rate_limiter = EnhancedRateLimiter()

def enhanced_rate_limit(limit=100, window=3600):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            key = f"rate_limit:{request.remote_addr}:{f.__name__}"
            if not rate_limiter.is_allowed(key, limit, window):
                return jsonify({'error': 'Rate limit exceeded'}), 429
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Session management for anonymous users
def get_or_create_session():
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        return None
    
    session = UserSession.query.filter_by(session_id=session_id).first()
    if not session:
        session = UserSession(
            session_id=session_id,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent', '')
        )
        db.session.add(session)
        db.session.commit()
    else:
        session.last_activity = datetime.utcnow()
        db.session.commit()
    
    return session

# Helper functions
def serialize_content_enhanced(content, include_details=False):
    """Enhanced content serialization"""
    base_data = {
        'id': content.id,
        'tmdb_id': content.tmdb_id,
        'imdb_id': content.imdb_id,
        'title': content.title,
        'original_title': content.original_title,
        'overview': content.overview,
        'tagline': content.tagline,
        'genres': content.genres,
        'language': content.language,
        'release_date': content.release_date.isoformat() if content.release_date else None,
        'runtime': content.runtime,
        'rating': content.rating,
        'vote_count': content.vote_count,
        'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
        'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path else None,
        'content_type': content.content_type,
        'popularity': content.popularity,
        'trending_score': content.trending_score
    }
    
    if include_details:
        base_data.update({
            'status': content.status,
            'production_companies': content.production_companies,
            'production_countries': content.production_countries,
            'budget': content.budget,
            'revenue': content.revenue,
            'keywords': content.keywords,
            'cast': content.cast[:10] if content.cast else [],  # Limit cast for performance
            'crew': content.crew[:5] if content.crew else [],   # Limit crew for performance
            'videos': content.videos,
            'spoken_languages': content.spoken_languages
        })
    
    return base_data

def async_to_sync_enhanced(async_func):
    """Enhanced async to sync converter with better error handling"""
    def wrapper(*args, **kwargs):
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new thread for async execution
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(async_func(*args, **kwargs))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(async_func(*args, **kwargs))
        except Exception as e:
            logger.error(f"Error in async_to_sync: {e}")
            return None
    return wrapper

# Enhanced API Routes

@app.route('/api/register', methods=['POST'])
@enhanced_rate_limit(limit=10, window=3600)
def register():
    try:
        data = request.get_json() or request.form.to_dict()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Enhanced validation
        if not username or len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        if not email or '@' not in email:
            return jsonify({'error': 'Valid email is required'}), 400
        if not password or len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        # Get demographics and preferences from registration data
        demographics = {
            'age_group': data.get('age_group'),
            'gender': data.get('gender'),
            'location': data.get('location'),
            'preferred_languages': data.get('preferred_languages', ['en'])
        }
        
        preferences = {
            'favorite_genres': data.get('favorite_genres', []),
            'content_types': data.get('content_types', ['movie', 'tv']),
            'regional_preference': data.get('regional_preference', 'global')
        }
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            demographics=demographics,
            preferences=preferences
        )
        db.session.add(user)
        db.session.commit()
        
        token = create_access_token(identity=user.id)
        return jsonify({
            'token': token, 
            'user_id': user.id,
            'username': user.username,
            'preferences': preferences
        })
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/homepage')
@cache.cached(timeout=300, key_prefix='homepage')
@enhanced_rate_limit(limit=200, window=3600)
def enhanced_homepage():
    """Enhanced homepage recommendations with caching and personalization"""
    try:
        # Check if user is logged in
        user_id = None
        if 'Authorization' in request.headers:
            try:
                from flask_jwt_extended import decode_token
                token = request.headers['Authorization'].replace('Bearer ', '')
                decoded = decode_token(token)
                user_id = decoded['sub']
            except:
                pass
        
        # Get session for anonymous users
        session = get_or_create_session()
        
        # Personalized recommendations
        personalized_recs = recommender.get_personalized_homepage(user_id, session)
        
        # Trending content by categories
        trending_data = {
            'global_trending': async_to_sync_enhanced(aggregator.fetch_trending_enhanced)('movie', 'week'),
            'tv_trending': async_to_sync_enhanced(aggregator.fetch_trending_enhanced)('tv', 'week'),
            'anime_trending': async_to_sync_enhanced(aggregator.fetch_trending_enhanced)('movie', 'week')  # Will need anime API
        }
        
        # Popular by genre with enhanced filtering
        HOMEPAGE_GENRES = {
            'Action': 28, 'Comedy': 35, 'Drama': 18, 'Horror': 27,
            'Sci-Fi': 878, 'Romance': 10749, 'Thriller': 53, 'Animation': 16
        }
        
        popular_by_genre = {}
        for genre_name, genre_id in HOMEPAGE_GENRES.items():
            genre_content = async_to_sync_enhanced(
                aggregator.fetch_popular_by_genre_enhanced
            )(genre_id, 'movie')
            popular_by_genre[genre_name] = genre_content[:8] if genre_content else []
        
        # Regional content with multiple languages
        regional_content = {}
        for lang_name, lang_code in REGIONAL_LANGUAGES.items():
            regional_data = async_to_sync_enhanced(
                aggregator.fetch_regional_content_enhanced
            )(lang_code, year=datetime.now().year)
            if regional_data:
                regional_content[lang_name] = regional_data[:6]
        
        # Admin curated content
        admin_posts = AdminPost.query.filter_by(
            is_active=True, 
            post_to_website=True
        ).filter(
            or_(AdminPost.expires_at.is_(None), AdminPost.expires_at > datetime.utcnow())
        ).order_by(AdminPost.priority.desc(), AdminPost.created_at.desc()).limit(15).all()
        
        curated_content = []
        for post in admin_posts:
            content_data = serialize_content_enhanced(post.content)
            content_data.update({
                'admin_title': post.title,
                'admin_description': post.description,
                'custom_tags': post.custom_tags,
                'priority': post.priority
            })
            curated_content.append(content_data)
        
        # Critics' choice and user favorites
        critics_choice = Content.query.filter(
            and_(Content.rating >= 7.5, Content.vote_count >= 1000)
        ).order_by(desc(Content.rating)).limit(10).all()
        
        user_favorites = Content.query.join(UserInteraction).filter(
            UserInteraction.interaction_type == 'favorite'
        ).group_by(Content.id).order_by(
            func.count(UserInteraction.id).desc()
        ).limit(10).all()
        
        # What's hot - recently popular content
        whats_hot = Content.query.filter(
            Content.release_date >= datetime.utcnow().date() - timedelta(days=90)
        ).order_by(desc(Content.popularity)).limit(12).all()
        
        return jsonify({
            'personalized': [serialize_content_enhanced(c) for c in personalized_recs[:20]],
            'trending': trending_data,
            'popular_by_genre': popular_by_genre,
            'regional': regional_content,
            'admin_curated': curated_content,
            'critics_choice': [serialize_content_enhanced(c) for c in critics_choice],
            'user_favorites': [serialize_content_enhanced(c) for c in user_favorites],
            'whats_hot': [serialize_content_enhanced(c) for c in whats_hot],
            'diversity_score': recommender.calculate_diversity_score(personalized_recs)
        })
        
    except Exception as e:
        logger.error(f"Homepage error: {e}")
        return jsonify({'error': 'Failed to load homepage'}), 500

@app.route('/api/recommendations')
@jwt_required()
@enhanced_rate_limit(limit=50, window=3600)
def get_enhanced_recommendations():
    """Enhanced personalized recommendations with multiple algorithms"""
    try:
        user_id = get_jwt_identity()
        
        # Update user's last activity
        user = User.query.get(user_id)
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Get ML-enhanced recommendations
        ml_recommendations = recommender.get_ml_enhanced_recommendations(user_id, 25)
        
        # Get collaborative filtering recommendations
        collaborative_recs = recommender.get_collaborative_recommendations(user_id, 15)
        
        # Get content-based recommendations
        content_recs = recommender.get_content_based_recommendations(user_id, 15)
        
        # Get user's interaction history for analysis
        recent_interactions = UserInteraction.query.filter_by(
            user_id=user_id
        ).filter(
            UserInteraction.created_at >= datetime.utcnow() - timedelta(days=60)
        ).order_by(UserInteraction.created_at.desc()).limit(20).all()
        
        # Analyze user preferences
        user_analytics = {
            'total_interactions': UserInteraction.query.filter_by(user_id=user_id).count(),
            'favorite_genres': {},
            'preferred_languages': {},
            'average_rating': 0,
            'content_type_preference': {}
        }
        
        # Calculate user preference analytics
        user_ratings = [i.rating for i in recent_interactions if i.rating]
        if user_ratings:
            user_analytics['average_rating'] = sum(user_ratings) / len(user_ratings)
        
        for interaction in recent_interactions:
            content = Content.query.get(interaction.content_id)
            if content:
                # Genre preferences
                if content.genres:
                    for genre in content.genres:
                        genre_name = GENRE_MAP.get(genre, str(genre))
                        user_analytics['favorite_genres'][genre_name] = \
                            user_analytics['favorite_genres'].get(genre_name, 0) + 1
                
                # Language preferences
                if content.language:
                    user_analytics['preferred_languages'][content.language] = \
                        user_analytics['preferred_languages'].get(content.language, 0) + 1
                
                # Content type preferences
                if content.content_type:
                    user_analytics['content_type_preference'][content.content_type] = \
                        user_analytics['content_type_preference'].get(content.content_type, 0) + 1
        
        # Genre-based recommendations based on user's top genres
        top_genres = sorted(user_analytics['favorite_genres'].items(), 
                          key=lambda x: x[1], reverse=True)[:3]
        
        genre_recommendations = []
        for genre_name, _ in top_genres:
            genre_id = next((k for k, v in GENRE_MAP.items() if v == genre_name), None)
            if genre_id:
                genre_content = Content.query.filter(
                    Content.genres.contains([genre_id])
                ).order_by(desc(Content.popularity)).limit(5).all()
                genre_recommendations.extend(genre_content)
        
        # Similar users recommendations
        similar_users_recs = []
        # Find users with similar preferences (simplified)
        similar_user_ids = db.session.query(UserInteraction.user_id).filter(
            UserInteraction.content_id.in_([i.content_id for i in recent_interactions[:10]]),
            UserInteraction.user_id != user_id,
            UserInteraction.interaction_type.in_(['favorite', 'like'])
        ).group_by(UserInteraction.user_id).having(
            func.count(UserInteraction.id) >= 2
        ).limit(5).all()
        
        for (similar_user_id,) in similar_user_ids:
            similar_user_content = UserInteraction.query.filter_by(
                user_id=similar_user_id,
                interaction_type='favorite'
            ).order_by(UserInteraction.created_at.desc()).limit(3).all()
            
            for interaction in similar_user_content:
                content = Content.query.get(interaction.content_id)
                if content and content not in [Content.query.get(i.content_id) for i in recent_interactions]:
                    similar_users_recs.append(content)
        
        # Trending recommendations based on user preferences
        user_languages = list(user_analytics['preferred_languages'].keys())
        trending_for_user = []
        
        if user_languages:
            trending_for_user = Content.query.filter(
                Content.language.in_(user_languages[:2])
            ).order_by(desc(Content.trending_score)).limit(10).all()
        
        return jsonify({
            'ml_recommendations': [serialize_content_enhanced(r) for r in ml_recommendations],
            'collaborative_filtering': [serialize_content_enhanced(r) for r in collaborative_recs],
            'content_based': [serialize_content_enhanced(r) for r in content_recs],
            'genre_based': [serialize_content_enhanced(r) for r in genre_recommendations],
            'similar_users': [serialize_content_enhanced(r) for r in similar_users_recs[:10]],
            'trending_for_you': [serialize_content_enhanced(r) for r in trending_for_user],
            'user_analytics': user_analytics,
            'recommendation_explanation': {
                'total_sources': 6,
                'personalization_score': min(len(recent_interactions) / 20, 1.0),
                'diversity_score': recommender.calculate_diversity_score(ml_recommendations)
            }
        })
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@app.route('/api/content/<int:content_id>')
@cache.cached(timeout=600, key_prefix='content_details')
@enhanced_rate_limit(limit=100, window=3600)
def get_enhanced_content_details(content_id):
    """Get comprehensive content details"""
    try:
        content = Content.query.get_or_404(content_id)
        
        # Get complete TMDB details if available
        tmdb_details = {}
        if content.tmdb_id:
            tmdb_details = async_to_sync_enhanced(
                aggregator.get_content_complete_details
            )(content.tmdb_id, content.content_type or 'movie') or {}
        
        # Get OMDb details for additional information
        omdb_details = {}
        if content.imdb_id:
            omdb_details = async_to_sync_enhanced(
                aggregator.get_omdb_details
            )(content.imdb_id) or {}
        
        # Get user reviews and ratings
        reviews = UserInteraction.query.filter_by(
            content_id=content.id
        ).filter(UserInteraction.rating.isnot(None)).order_by(
            UserInteraction.created_at.desc()
        ).limit(20).all()
        
        # Calculate rating statistics
        ratings = [r.rating for r in reviews]
        rating_stats = {
            'average': sum(ratings) / len(ratings) if ratings else 0,
            'count': len(ratings),
            'distribution': {i: ratings.count(i) for i in range(1, 6)}
        }
        
        # Get similar content
        similar_content = []
        similarities = ContentSimilarity.query.filter_by(
            content_id=content.id
        ).order_by(desc(ContentSimilarity.similarity_score)).limit(10).all()
        
        for sim in similarities:
            similar = Content.query.get(sim.similar_content_id)
            if similar:
                similar_data = serialize_content_enhanced(similar)
                similar_data['similarity_score'] = sim.similarity_score
                similar_data['similarity_type'] = sim.similarity_type
                similar_content.append(similar_data)
        
        # If no pre-computed similarities, use real-time computation
        if not similar_content and recommender.content_similarity is not None:
            try:
                content_idx = Content.query.filter(Content.id <= content.id).count() - 1
                if content_idx < len(recommender.content_similarity):
                    similar_indices = np.argsort(
                        recommender.content_similarity[content_idx]
                    )[::-1][1:11]
                    
                    for idx in similar_indices:
                        similar = Content.query.offset(idx).first()
                        if similar:
                            similar_data = serialize_content_enhanced(similar)
                            similar_data['similarity_score'] = \
                                recommender.content_similarity[content_idx][idx]
                            similar_content.append(similar_data)
            except Exception as e:
                logger.warning(f"Error computing similar content: {e}")
        
        # Get viewing statistics
        view_stats = {
            'total_views': UserInteraction.query.filter_by(
                content_id=content.id, 
                interaction_type='view'
            ).count(),
            'favorites': UserInteraction.query.filter_by(
                content_id=content.id, 
                interaction_type='favorite'
            ).count(),
            'wishlist_adds': UserInteraction.query.filter_by(
                content_id=content.id, 
                interaction_type='wishlist'
            ).count()
        }
        
        # YouTube trailers
        youtube_videos = []
        if content.title:
            year = content.release_date.year if content.release_date else None
            youtube_videos = async_to_sync_enhanced(
                aggregator.search_youtube_trailers
            )(content.title, year) or []
        
        # Trending score calculation
        recent_views = UserInteraction.query.filter_by(
            content_id=content.id,
            interaction_type='view'
        ).filter(
            UserInteraction.created_at >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        trending_score = (recent_views * 0.4 + content.popularity * 0.3 + 
                         (content.rating or 0) * 10 * 0.3)
        
        return jsonify({
            'content': serialize_content_enhanced(content, include_details=True),
            'tmdb_details': tmdb_details,
            'omdb_details': omdb_details,
            'rating_stats': rating_stats,
            'reviews': [
                {
                    'user_id': r.user_id,
                    'rating': r.rating,
                    'created_at': r.created_at.isoformat(),
                    'interaction_type': r.interaction_type
                } for r in reviews
            ],
            'similar_content': similar_content,
            'view_stats': view_stats,
            'youtube_videos': youtube_videos,
            'trending_score': trending_score,
            'recommendations': {
                'watch_next': similar_content[:5],
                'if_you_liked': similar_content[5:10] if len(similar_content) > 5 else []
            }
        })
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Background tasks and monitoring
def update_content_similarities():
    """Background task to update content similarities"""
    try:
        with app.app_context():
            logger.info("Starting content similarity update")
            recommender.build_enhanced_content_matrix()
            
            # Update trending scores
            recent_interactions = UserInteraction.query.filter(
                UserInteraction.created_at >= datetime.utcnow() - timedelta(days=7)
            ).all()
            
            content_scores = defaultdict(float)
            for interaction in recent_interactions:
                weight = 1.0
                if interaction.interaction_type == 'favorite':
                    weight = 3.0
                elif interaction.interaction_type == 'like':
                    weight = 2.0
                elif interaction.interaction_type == 'view':
                    weight = 1.0
                
                content_scores[interaction.content_id] += weight
            
            # Update trending scores in database
            for content_id, score in content_scores.items():
                content = Content.query.get(content_id)
                if content:
                    content.trending_score = score
            
            db.session.commit()
            logger.info("Content similarity update completed")
            
    except Exception as e:
        logger.error(f"Error updating content similarities: {e}")

def sync_external_content():
    """Background task to sync content from external APIs"""
    try:
        with app.app_context():
            logger.info("Starting external content sync")
            
            # Sync trending movies and TV shows
            trending_movies = async_to_sync_enhanced(aggregator.fetch_trending_enhanced)('movie')
            trending_tv = async_to_sync_enhanced(aggregator.fetch_trending_enhanced)('tv')
            
            for items, content_type in [(trending_movies, 'movie'), (trending_tv, 'tv')]:
                for item in items[:20]:  # Limit to top 20
                    existing = Content.query.filter_by(tmdb_id=str(item['id'])).first()
                    if not existing:
                        try:
                            content = create_content_from_tmdb(item, content_type)
                            db.session.add(content)
                        except Exception as e:
                            logger.warning(f"Error creating content from TMDB data: {e}")
            
            db.session.commit()
            logger.info("External content sync completed")
            
    except Exception as e:
        logger.error(f"Error syncing external content: {e}")

# Initialize scheduler for background tasks
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=update_content_similarities,
    trigger=IntervalTrigger(hours=6),
    id='update_similarities',
    name='Update content similarities',
    replace_existing=True
)
scheduler.add_job(
    func=sync_external_content,
    trigger=IntervalTrigger(hours=12),
    id='sync_content',
    name='Sync external content',
    replace_existing=True
)

# Start scheduler
try:
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
except Exception as e:
    logger.error(f"Failed to start scheduler: {e}")

# Enhanced helper functions
def create_content_from_tmdb(tmdb_data, content_type='movie'):
    """Enhanced content creation from TMDB data"""
    try:
        # Extract release date
        release_date = None
        date_field = 'release_date' if content_type == 'movie' else 'first_air_date'
        if tmdb_data.get(date_field):
            try:
                release_date = datetime.strptime(tmdb_data[date_field], '%Y-%m-%d').date()
            except:
                pass
        
        # Create content object
        content = Content(
            tmdb_id=str(tmdb_data['id']),
            title=tmdb_data.get('title', tmdb_data.get('name', 'Unknown')),
            original_title=tmdb_data.get('original_title', tmdb_data.get('original_name')),
            overview=tmdb_data.get('overview'),
            genres=tmdb_data.get('genre_ids', []) if 'genre_ids' in tmdb_data else [g['id'] for g in tmdb_data.get('genres', [])],
            language=tmdb_data.get('original_language'),
            release_date=release_date,
            runtime=tmdb_data.get('runtime'),
            rating=tmdb_data.get('vote_average'),
            vote_count=tmdb_data.get('vote_count', 0),
            poster_path=tmdb_data.get('poster_path'),
            backdrop_path=tmdb_data.get('backdrop_path'),
            content_type=content_type,
            popularity=tmdb_data.get('popularity', 0),
            status=tmdb_data.get('status'),
            budget=tmdb_data.get('budget'),
            revenue=tmdb_data.get('revenue'),
            production_companies=tmdb_data.get('production_companies'),
            production_countries=tmdb_data.get('production_countries'),
            spoken_languages=tmdb_data.get('spoken_languages')
        )
        
        return content
        
    except Exception as e:
        logger.error(f"Error creating content from TMDB data: {e}")
        raise

# Add the rest of your API routes here (interact, admin routes, etc.)
# ... (continuing with the rest of the enhanced routes)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Create admin user if not exists
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin_user = User(
                username='admin',
                email='admin@movieapp.com',
                password_hash=generate_password_hash('admin123'),
                preferences={'role': 'admin'}
            )
            db.session.add(admin_user)
            db.session.commit()
    
    app.run(debug=True, port=5000)