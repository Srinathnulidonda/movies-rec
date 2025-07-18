import os
import sys
from datetime import datetime, timedelta
from functools import wraps
import requests
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_caching import Cache
from sqlalchemy import func, and_, or_, desc, text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON
import redis
from apscheduler.schedulers.background import BackgroundScheduler
import logging
from typing import List, Dict, Any, Optional
import hashlib
import json
from dataclasses import dataclass
from enum import Enum

# Initialize Flask app
app = Flask(__name__)

# Configuration
class Config:
    # Database configuration
    if os.environ.get('DATABASE_URL'):
        SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
        if SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
            SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)
    else:
        SQLALCHEMY_DATABASE_URI = 'sqlite:///recommendation_system.db'
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True
    }
    
    # JWT configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=7)
    
    # API Keys
    TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
    OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '52260795')
    
    # Cache configuration
    CACHE_TYPE = 'redis' if os.environ.get('REDIS_URL') else 'simple'
    CACHE_REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    CACHE_DEFAULT_TIMEOUT = 300
    
    # ML Service URL
    ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'http://localhost:5001')
    
    # Telegram configuration
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '')

app.config.from_object(Config)

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
cache = Cache(app)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Models
class ContentType(Enum):
    MOVIE = "movie"
    TV_SHOW = "tv_show"
    ANIME = "anime"

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # User preferences
    preferred_genres = db.Column(JSON, default=list)
    preferred_languages = db.Column(JSON, default=list)
    
    # Relationships
    watch_history = relationship('WatchHistory', back_populates='user', lazy='dynamic')
    favorites = relationship('Favorite', back_populates='user', lazy='dynamic')
    ratings = relationship('Rating', back_populates='user', lazy='dynamic')
    wishlist = relationship('Wishlist', back_populates='user', lazy='dynamic')
    reviews = relationship('Review', back_populates='user', lazy='dynamic')

class Content(db.Model):
    __tablename__ = 'content'
    
    id = db.Column(db.Integer, primary_key=True)
    external_id = db.Column(db.String(100), index=True)
    source = db.Column(db.String(50))  # tmdb, omdb, jikan, regional
    content_type = db.Column(db.String(20))
    title = db.Column(db.String(200), nullable=False)
    original_title = db.Column(db.String(200))
    description = db.Column(db.Text)
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    poster_path = db.Column(db.String(500))
    backdrop_path = db.Column(db.String(500))
    trailer_url = db.Column(db.String(500))
    
    # Metadata
    genres = db.Column(JSON, default=list)
    cast = db.Column(JSON, default=list)
    crew = db.Column(JSON, default=list)
    languages = db.Column(JSON, default=list)
    countries = db.Column(JSON, default=list)
    
    # Ratings
    tmdb_rating = db.Column(db.Float)
    imdb_rating = db.Column(db.Float)
    user_rating = db.Column(db.Float)
    critic_score = db.Column(db.Float)
    
    # Regional content flags
    is_telugu = db.Column(db.Boolean, default=False)
    is_hindi = db.Column(db.Boolean, default=False)
    is_tamil = db.Column(db.Boolean, default=False)
    is_kannada = db.Column(db.Boolean, default=False)
    
    # Analytics
    view_count = db.Column(db.Integer, default=0)
    popularity_score = db.Column(db.Float, default=0.0)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    watch_history = relationship('WatchHistory', back_populates='content', lazy='dynamic')
    favorites = relationship('Favorite', back_populates='content', lazy='dynamic')
    ratings = relationship('Rating', back_populates='content', lazy='dynamic')
    wishlist = relationship('Wishlist', back_populates='content', lazy='dynamic')
    reviews = relationship('Review', back_populates='content', lazy='dynamic')

class WatchHistory(db.Model):
    __tablename__ = 'watch_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    watched_at = db.Column(db.DateTime, default=datetime.utcnow)
    watch_duration = db.Column(db.Integer)  # in minutes
    completed = db.Column(db.Boolean, default=False)
    
    user = relationship('User', back_populates='watch_history')
    content = relationship('Content', back_populates='watch_history')

class Favorite(db.Model):
    __tablename__ = 'favorites'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = relationship('User', back_populates='favorites')
    content = relationship('Content', back_populates='favorites')

class Rating(db.Model):
    __tablename__ = 'ratings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    rating = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = relationship('User', back_populates='ratings')
    content = relationship('Content', back_populates='ratings')

class Wishlist(db.Model):
    __tablename__ = 'wishlist'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    priority = db.Column(db.Integer, default=0)
    
    user = relationship('User', back_populates='wishlist')
    content = relationship('Content', back_populates='wishlist')

class Review(db.Model):
    __tablename__ = 'reviews'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    review_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship('User', back_populates='reviews')
    content = relationship('Content', back_populates='reviews')

class CuratedRecommendation(db.Model):
    __tablename__ = 'curated_recommendations'
    
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(50))  # critics_choice, editors_pick, etc
    priority = db.Column(db.Integer, default=0)
    active = db.Column(db.Boolean, default=True)
    expires_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    content = relationship('Content')

# API Integration Classes
class TMDBClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.themoviedb.org/3"
    
    def search(self, query, content_type='movie'):
        endpoint = f"{self.base_url}/search/{content_type}"
        params = {'api_key': self.api_key, 'query': query}
        response = requests.get(endpoint, params=params)
        return response.json() if response.status_code == 200 else None
    
    def get_details(self, content_id, content_type='movie'):
        endpoint = f"{self.base_url}/{content_type}/{content_id}"
        params = {'api_key': self.api_key, 'append_to_response': 'credits,videos'}
        response = requests.get(endpoint, params=params)
        return response.json() if response.status_code == 200 else None
    
    def get_trending(self, content_type='movie', time_window='week'):
        endpoint = f"{self.base_url}/trending/{content_type}/{time_window}"
        params = {'api_key': self.api_key}
        response = requests.get(endpoint, params=params)
        return response.json() if response.status_code == 200 else None

class OMDBClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://www.omdbapi.com/"
    
    def search(self, title):
        params = {'apikey': self.api_key, 't': title}
        response = requests.get(self.base_url, params=params)
        return response.json() if response.status_code == 200 else None

class JikanClient:
    def __init__(self):
        self.base_url = "https://api.jikan.moe/v4"
    
    def search(self, query):
        endpoint = f"{self.base_url}/anime"
        params = {'q': query}
        response = requests.get(endpoint, params=params)
        return response.json() if response.status_code == 200 else None
    
    def get_details(self, anime_id):
        endpoint = f"{self.base_url}/anime/{anime_id}/full"
        response = requests.get(endpoint)
        return response.json() if response.status_code == 200 else None

# Helper Functions
def create_content_from_tmdb(tmdb_data, content_type='movie'):
    """Create or update content from TMDB data"""
    external_id = f"tmdb_{tmdb_data['id']}"
    
    content = Content.query.filter_by(external_id=external_id).first()
    if not content:
        content = Content(external_id=external_id)
    
    content.source = 'tmdb'
    content.content_type = content_type
    content.title = tmdb_data.get('title') or tmdb_data.get('name')
    content.original_title = tmdb_data.get('original_title') or tmdb_data.get('original_name')
    content.description = tmdb_data.get('overview')
    content.release_date = datetime.strptime(tmdb_data.get('release_date', '1900-01-01'), '%Y-%m-%d').date()
    content.poster_path = f"https://image.tmdb.org/t/p/w500{tmdb_data.get('poster_path')}" if tmdb_data.get('poster_path') else None
    content.backdrop_path = f"https://image.tmdb.org/t/p/original{tmdb_data.get('backdrop_path')}" if tmdb_data.get('backdrop_path') else None
    content.tmdb_rating = tmdb_data.get('vote_average')
    
    # Extract genres
    if 'genres' in tmdb_data:
        content.genres = [genre['name'] for genre in tmdb_data['genres']]
    
    # Extract cast and crew if available
    if 'credits' in tmdb_data:
        content.cast = [
            {'name': person['name'], 'character': person['character']} 
            for person in tmdb_data['credits'].get('cast', [])[:10]
        ]
        content.crew = [
            {'name': person['name'], 'job': person['job']} 
            for person in tmdb_data['credits'].get('crew', [])
            if person['job'] in ['Director', 'Producer', 'Writer']
        ]
    
    # Extract trailer
    if 'videos' in tmdb_data:
        for video in tmdb_data['videos'].get('results', []):
            if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                content.trailer_url = f"https://www.youtube.com/watch?v={video['key']}"
                break
    
    db.session.add(content)
    db.session.commit()
    
    return content

# Recommendation Engine
class RecommendationEngine:
    def __init__(self):
        self.tmdb_client = TMDBClient(app.config['TMDB_API_KEY'])
        self.omdb_client = OMDBClient(app.config['OMDB_API_KEY'])
        self.jikan_client = JikanClient()
    
    @cache.memoize(timeout=3600)
    def get_homepage_recommendations(self):
        """Generate recommendations for non-logged users"""
        recommendations = {
            'trending': self._get_trending_content(),
            'popular_by_genre': self._get_popular_by_genre(),
            'whats_hot': self._get_whats_hot(),
            'regional_spotlight': self._get_regional_spotlight(),
            'critics_choice': self._get_critics_choice(),
            'user_favorites': self._get_user_favorites()
        }
        return recommendations
    
    def get_personalized_recommendations(self, user_id):
        """Generate personalized recommendations for logged-in users"""
        user = User.query.get(user_id)
        if not user:
            return None
        
        recommendations = {
            'based_on_history': self._get_history_based_recommendations(user),
            'based_on_favorites': self._get_favorites_based_recommendations(user),
            'based_on_wishlist': self._get_wishlist_based_recommendations(user),
            'genre_recommendations': self._get_genre_recommendations(user),
            'collaborative_filtering': self._get_collaborative_recommendations(user),
            'hybrid_recommendations': self._get_hybrid_recommendations(user)
        }
        
        return recommendations
    
    def _get_trending_content(self):
        """Get trending content from various sources"""
        trending = []
        
        # Get trending from TMDB
        tmdb_trending = self.tmdb_client.get_trending()
        if tmdb_trending:
            for item in tmdb_trending.get('results', [])[:10]:
                content = create_content_from_tmdb(item)
                trending.append(self._serialize_content(content))
        
        # Get trending from database based on view count
        db_trending = Content.query.order_by(
            desc(Content.view_count)
        ).limit(10).all()
        
        for content in db_trending:
            trending.append(self._serialize_content(content))
        
        return trending[:20]  # Return top 20
    
    def _get_popular_by_genre(self):
        """Get popular content organized by genre"""
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance']
        popular_by_genre = {}
        
        for genre in genres:
            content_list = Content.query.filter(
                Content.genres.contains([genre])
            ).order_by(desc(Content.popularity_score)).limit(10).all()
            
            popular_by_genre[genre] = [
                self._serialize_content(content) for content in content_list
            ]
        
        return popular_by_genre
    
    def _get_whats_hot(self):
        """Get currently hot content based on recent views"""
        # Calculate content viewed in last 7 days
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        
        hot_content = db.session.query(
            Content,
            func.count(WatchHistory.id).label('recent_views')
        ).join(
            WatchHistory, Content.id == WatchHistory.content_id
        ).filter(
            WatchHistory.watched_at >= seven_days_ago
        ).group_by(Content.id).order_by(
            desc('recent_views')
        ).limit(15).all()
        
        return [self._serialize_content(item[0]) for item in hot_content]
    
    def _get_regional_spotlight(self):
        """Get popular regional content"""
        regional = {
            'telugu': Content.query.filter_by(is_telugu=True).order_by(
                desc(Content.popularity_score)
            ).limit(10).all(),
            'hindi': Content.query.filter_by(is_hindi=True).order_by(
                desc(Content.popularity_score)
            ).limit(10).all(),
            'tamil': Content.query.filter_by(is_tamil=True).order_by(
                desc(Content.popularity_score)
            ).limit(10).all(),
            'kannada': Content.query.filter_by(is_kannada=True).order_by(
                desc(Content.popularity_score)
            ).limit(10).all()
        }
        
        return {
            lang: [self._serialize_content(content) for content in content_list]
            for lang, content_list in regional.items()
        }
    
    def _get_critics_choice(self):
        """Get curated critic recommendations"""
        curated = CuratedRecommendation.query.filter_by(
            category='critics_choice',
            active=True
        ).filter(
            or_(
                CuratedRecommendation.expires_at == None,
                CuratedRecommendation.expires_at > datetime.utcnow()
            )
        ).order_by(desc(CuratedRecommendation.priority)).limit(10).all()
        
        return [
            self._serialize_content(rec.content) for rec in curated
        ]
    
    def _get_user_favorites(self):
        """Get most favorited content"""
        favorites = db.session.query(
            Content,
            func.count(Favorite.id).label('favorite_count')
        ).join(
            Favorite, Content.id == Favorite.content_id
        ).group_by(Content.id).order_by(
            desc('favorite_count')
        ).limit(15).all()
        
        return [self._serialize_content(item[0]) for item in favorites]
    
    def _get_history_based_recommendations(self, user):
        """Get recommendations based on watch history"""
        # Get user's recent watch history
        recent_history = user.watch_history.order_by(
            desc(WatchHistory.watched_at)
        ).limit(20).all()
        
        if not recent_history:
            return []
        
        # Extract genres from watched content
        watched_genres = []
        for history in recent_history:
            watched_genres.extend(history.content.genres or [])
        
        # Get most common genres
        from collections import Counter
        genre_counts = Counter(watched_genres)
        top_genres = [genre for genre, _ in genre_counts.most_common(3)]
        
        # Find similar content
        recommendations = Content.query.filter(
            and_(
                Content.id.notin_([h.content_id for h in recent_history]),
                or_(*[Content.genres.contains([genre]) for genre in top_genres])
            )
        ).order_by(desc(Content.popularity_score)).limit(10).all()
        
        return [self._serialize_content(content) for content in recommendations]
    
    def _get_favorites_based_recommendations(self, user):
        """Get recommendations based on user favorites"""
        favorites = user.favorites.all()
        if not favorites:
            return []
        
        # Similar logic to history-based but with favorites
        favorite_genres = []
        for fav in favorites:
            favorite_genres.extend(fav.content.genres or [])
        
        from collections import Counter
        genre_counts = Counter(favorite_genres)
        top_genres = [genre for genre, _ in genre_counts.most_common(3)]
        
        recommendations = Content.query.filter(
            and_(
                Content.id.notin_([f.content_id for f in favorites]),
                or_(*[Content.genres.contains([genre]) for genre in top_genres])
            )
        ).order_by(desc(Content.popularity_score)).limit(10).all()
        
        return [self._serialize_content(content) for content in recommendations]
    
    def _get_wishlist_based_recommendations(self, user):
        """Get recommendations based on wishlist"""
        wishlist = user.wishlist.all()
        if not wishlist:
            return []
        
        # Get genres from wishlist
        wishlist_genres = []
        for item in wishlist:
            wishlist_genres.extend(item.content.genres or [])
        
        from collections import Counter
        genre_counts = Counter(wishlist_genres)
        top_genres = [genre for genre, _ in genre_counts.most_common(2)]
        
        recommendations = Content.query.filter(
            and_(
                Content.id.notin_([w.content_id for w in wishlist]),
                or_(*[Content.genres.contains([genre]) for genre in top_genres])
            )
        ).order_by(desc(Content.popularity_score)).limit(8).all()
        
        return [self._serialize_content(content) for content in recommendations]
    
    def _get_genre_recommendations(self, user):
        """Get recommendations based on user's preferred genres"""
        if not user.preferred_genres:
            return []
        
        recommendations = Content.query.filter(
            or_(*[Content.genres.contains([genre]) for genre in user.preferred_genres])
        ).order_by(desc(Content.popularity_score)).limit(10).all()
        
        return [self._serialize_content(content) for content in recommendations]
    
    def _get_collaborative_recommendations(self, user):
        """Get recommendations using collaborative filtering"""
        # Find similar users based on ratings
        user_ratings = {r.content_id: r.rating for r in user.ratings.all()}
        
        if not user_ratings:
            return []
        
        # Find users who rated similar content
        similar_users = []
        other_users = User.query.filter(User.id != user.id).all()
        
        for other_user in other_users:
            other_ratings = {r.content_id: r.rating for r in other_user.ratings.all()}
            
            # Calculate similarity (simple approach)
            common_items = set(user_ratings.keys()) & set(other_ratings.keys())
            if len(common_items) > 3:
                similarity = sum(
                    abs(user_ratings[item] - other_ratings[item]) 
                    for item in common_items
                ) / len(common_items)
                similar_users.append((other_user, similarity))
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x[1])
        
        # Get recommendations from similar users
        recommendations = []
        for similar_user, _ in similar_users[:5]:
            similar_user_favorites = similar_user.favorites.filter(
                Favorite.content_id.notin_(user_ratings.keys())
            ).limit(2).all()
            
            for fav in similar_user_favorites:
                recommendations.append(self._serialize_content(fav.content))
        
        return recommendations[:10]
    
    def _get_hybrid_recommendations(self, user):
        """Combine multiple recommendation approaches"""
        try:
            # Call ML service for advanced recommendations
            ml_response = requests.post(
                f"{app.config['ML_SERVICE_URL']}/recommend",
                json={'user_id': user.id}
            )
            
            if ml_response.status_code == 200:
                ml_recommendations = ml_response.json().get('recommendations', [])
                
                # Get content objects for ML recommendations
                content_ids = [rec['content_id'] for rec in ml_recommendations]
                contents = Content.query.filter(Content.id.in_(content_ids)).all()
                
                return [self._serialize_content(content) for content in contents]
        except:
            logger.error("ML service unavailable, falling back to basic recommendations")
        
        # Fallback to combining other recommendation methods
        history_recs = self._get_history_based_recommendations(user)[:3]
        favorite_recs = self._get_favorites_based_recommendations(user)[:3]
        genre_recs = self._get_genre_recommendations(user)[:4]
        
        return history_recs + favorite_recs + genre_recs
    
    def _serialize_content(self, content):
        """Serialize content object to dictionary"""
        return {
            'id': content.id,
            'external_id': content.external_id,
            'title': content.title,
            'original_title': content.original_title,
            'description': content.description,
            'content_type': content.content_type,
            'poster_path': content.poster_path,
            'backdrop_path': content.backdrop_path,
            'genres': content.genres,
            'tmdb_rating': content.tmdb_rating,
            'imdb_rating': content.imdb_rating,
            'user_rating': content.user_rating,
            'release_date': content.release_date.isoformat() if content.release_date else None
        }

# Authentication decorator for admin routes
def admin_required(f):
    @wraps(f)
    @jwt_required()
    def decorated_function(*args, **kwargs):
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if not user or user.username != 'admin':  # Simple admin check
            return jsonify({'message': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

# Initialize recommendation engine
recommendation_engine = RecommendationEngine()

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration"""
    data = request.get_json()
    
    # Validate input
    if not all(k in data for k in ('username', 'email', 'password')):
        return jsonify({'message': 'Missing required fields'}), 400
    
    # Check if user exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'message': 'Username already exists'}), 409
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'Email already exists'}), 409
    
    # Create new user
    user = User(
        username=data['username'],
        email=data['email'],
        password_hash=bcrypt.generate_password_hash(data['password']).decode('utf-8'),
        preferred_genres=data.get('preferred_genres', []),
        preferred_languages=data.get('preferred_languages', [])
    )
    
    db.session.add(user)
    db.session.commit()
    
    # Generate access token
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'message': 'User created successfully',
        'access_token': access_token,
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email
        }
    }), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    data = request.get_json()
    
    if not all(k in data for k in ('username', 'password')):
        return jsonify({'message': 'Missing username or password'}), 400
    
    user = User.query.filter_by(username=data['username']).first()
    
    if not user or not bcrypt.check_password_hash(user.password_hash, data['password']):
        return jsonify({'message': 'Invalid credentials'}), 401
    
    access_token = create_access_token(identity=user.id)
    
    return jsonify({
        'access_token': access_token,
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email
        }
    }), 200

@app.route('/api/recommendations/homepage', methods=['GET'])
@cache.cached(timeout=3600)
def homepage_recommendations():
    """Get homepage recommendations for non-logged users"""
    recommendations = recommendation_engine.get_homepage_recommendations()
    return jsonify(recommendations), 200

@app.route('/api/recommendations/personalized', methods=['GET'])
@jwt_required()
def personalized_recommendations():
    """Get personalized recommendations for logged-in users"""
    user_id = get_jwt_identity()
    recommendations = recommendation_engine.get_personalized_recommendations(user_id)
    
    if recommendations is None:
        return jsonify({'message': 'User not found'}), 404
    
    return jsonify(recommendations), 200

@app.route('/api/content/<int:content_id>/details', methods=['GET'])
def content_details(content_id):
    """Get complete content details"""
    content = Content.query.get_or_404(content_id)
    
    # Increment view count
    content.view_count += 1
    db.session.commit()
    
    # Get additional data
    ratings = db.session.query(func.avg(Rating.rating)).filter_by(content_id=content_id).scalar()
    total_favorites = Favorite.query.filter_by(content_id=content_id).count()
    
    # Get reviews
    reviews = Review.query.filter_by(content_id=content_id).order_by(
        desc(Review.created_at)
    ).limit(10).all()
    
    # Get similar content
    similar_content = Content.query.filter(
        and_(
            Content.id != content_id,
            or_(*[Content.genres.contains([genre]) for genre in content.genres])
        )
    ).order_by(desc(Content.popularity_score)).limit(6).all()
    
    return jsonify({
        'content': recommendation_engine._serialize_content(content),
        'additional_info': {
            'runtime': content.runtime,
            'languages': content.languages,
            'countries': content.countries,
            'cast': content.cast,
            'crew': content.crew,
            'trailer_url': content.trailer_url,
            'critic_score': content.critic_score
        },
        'stats': {
            'user_rating': ratings or 0,
            'total_favorites': total_favorites,
            'view_count': content.view_count
        },
        'reviews': [
            {
                'id': review.id,
                'user': review.user.username,
                'text': review.review_text,
                'created_at': review.created_at.isoformat()
            } for review in reviews
        ],
        'similar_content': [
            recommendation_engine._serialize_content(c) for c in similar_content
        ]
    }), 200

@app.route('/api/search', methods=['GET'])
def search():
    """Multi-source content search"""
    query = request.args.get('q', '')
    content_type = request.args.get('type', 'all')
    source = request.args.get('source', 'all')
    
    if not query:
        return jsonify({'message': 'Search query required'}), 400
    
    results = []
    
    # Search in database first
    db_results = Content.query.filter(
        Content.title.ilike(f'%{query}%')
    )
    
    if content_type != 'all':
        db_results = db_results.filter_by(content_type=content_type)
    
    results.extend([
        recommendation_engine._serialize_content(content) 
        for content in db_results.limit(20).all()
    ])
    
    # Search external sources if requested
    if source in ['all', 'tmdb'] and app.config['TMDB_API_KEY']:
        tmdb_results = recommendation_engine.tmdb_client.search(query, content_type)
        if tmdb_results:
            for item in tmdb_results.get('results', [])[:10]:
                content = create_content_from_tmdb(item, content_type)
                results.append(recommendation_engine._serialize_content(content))
    
    if source in ['all', 'jikan'] and content_type in ['all', 'anime']:
        jikan_results = recommendation_engine.jikan_client.search(query)
        if jikan_results and 'data' in jikan_results:
            for anime in jikan_results['data'][:5]:
                # Create anime content
                external_id = f"jikan_{anime['mal_id']}"
                content = Content.query.filter_by(external_id=external_id).first()
                if not content:
                    content = Content(
                        external_id=external_id,
                        source='jikan',
                        content_type='anime',
                        title=anime['title'],
                        description=anime.get('synopsis'),
                        poster_path=anime['images']['jpg']['image_url'] if 'images' in anime else None,
                        genres=[g['name'] for g in anime.get('genres', [])]
                    )
                    db.session.add(content)
                    db.session.commit()
                
                results.append(recommendation_engine._serialize_content(content))
    
    return jsonify({
        'results': results,
        'total': len(results)
    }), 200

@app.route('/api/trending', methods=['GET'])
@cache.cached(timeout=1800)
def trending():
    """Get trending content globally and regionally"""
    region = request.args.get('region', 'global')
    
    trending_data = {
        'global': recommendation_engine._get_trending_content(),
        'regional': {}
    }
    
    if region in ['all', 'india']:
        trending_data['regional'] = recommendation_engine._get_regional_spotlight()
    
    return jsonify(trending_data), 200

@app.route('/api/user/interactions', methods=['POST'])
@jwt_required()
def user_interactions():
    """Handle user interactions (watchlist, favorites, ratings)"""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    action = data.get('action')
    content_id = data.get('content_id')
    
    if not all([action, content_id]):
        return jsonify({'message': 'Missing required fields'}), 400
    
    # Verify content exists
    content = Content.query.get(content_id)
    if not content:
        return jsonify({'message': 'Content not found'}), 404
    
    if action == 'add_favorite':
        existing = Favorite.query.filter_by(user_id=user_id, content_id=content_id).first()
        if not existing:
            favorite = Favorite(user_id=user_id, content_id=content_id)
            db.session.add(favorite)
            db.session.commit()
        return jsonify({'message': 'Added to favorites'}), 200
    
    elif action == 'remove_favorite':
        favorite = Favorite.query.filter_by(user_id=user_id, content_id=content_id).first()
        if favorite:
            db.session.delete(favorite)
            db.session.commit()
        return jsonify({'message': 'Removed from favorites'}), 200
    
    elif action == 'add_wishlist':
        existing = Wishlist.query.filter_by(user_id=user_id, content_id=content_id).first()
        if not existing:
            wishlist = Wishlist(
                user_id=user_id, 
                content_id=content_id,
                priority=data.get('priority', 0)
            )
            db.session.add(wishlist)
            db.session.commit()
        return jsonify({'message': 'Added to wishlist'}), 200
    
    elif action == 'remove_wishlist':
        wishlist = Wishlist.query.filter_by(user_id=user_id, content_id=content_id).first()
        if wishlist:
            db.session.delete(wishlist)
            db.session.commit()
        return jsonify({'message': 'Removed from wishlist'}), 200
    
    elif action == 'rate':
        rating_value = data.get('rating')
        if not rating_value or not (0 <= rating_value <= 10):
            return jsonify({'message': 'Invalid rating value'}), 400
        
        rating = Rating.query.filter_by(user_id=user_id, content_id=content_id).first()
        if rating:
            rating.rating = rating_value
        else:
            rating = Rating(user_id=user_id, content_id=content_id, rating=rating_value)
            db.session.add(rating)
        
        # Update content user rating
        avg_rating = db.session.query(func.avg(Rating.rating)).filter_by(content_id=content_id).scalar()
        content.user_rating = avg_rating
        
        db.session.commit()
        return jsonify({'message': 'Rating updated'}), 200
    
    elif action == 'add_history':
        history = WatchHistory(
            user_id=user_id,
            content_id=content_id,
            watch_duration=data.get('duration', 0),
            completed=data.get('completed', False)
        )
        db.session.add(history)
        db.session.commit()
        return jsonify({'message': 'Added to watch history'}), 200
    
    else:
        return jsonify({'message': 'Invalid action'}), 400

@app.route('/api/admin/curate', methods=['POST'])
@admin_required
def admin_curate():
    """Admin content curation endpoint"""
    data = request.get_json()
    
    content_id = data.get('content_id')
    if not content_id:
        return jsonify({'message': 'Content ID required'}), 400
    
    # Verify content exists
    content = Content.query.get(content_id)
    if not content:
        return jsonify({'message': 'Content not found'}), 404
    
    # Create curated recommendation
    curated = CuratedRecommendation(
        content_id=content_id,
        title=data.get('title', content.title),
        description=data.get('description'),
        category=data.get('category', 'editors_pick'),
        priority=data.get('priority', 0),
        expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None
    )
    
    db.session.add(curated)
    db.session.commit()
    
    # Send Telegram notification if configured
    if app.config['TELEGRAM_BOT_TOKEN'] and app.config['TELEGRAM_CHANNEL_ID']:
        telegram_message = f"🎬 New Recommendation!\n\n{curated.title}\n\n{curated.description or content.description}"
        
        try:
            telegram_url = f"https://api.telegram.org/bot{app.config['TELEGRAM_BOT_TOKEN']}/sendMessage"
            requests.post(telegram_url, data={
                'chat_id': app.config['TELEGRAM_CHANNEL_ID'],
                'text': telegram_message,
                'parse_mode': 'HTML'
            })
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
    
    return jsonify({
        'message': 'Content curated successfully',
        'curated_id': curated.id
    }), 201

@app.route('/api/admin/content/search', methods=['GET'])
@admin_required
def admin_content_search():
    """Admin search across multiple content sources"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'message': 'Search query required'}), 400
    
    results = {
        'tmdb': [],
        'omdb': [],
        'jikan': [],
        'database': []
    }
    
    # Search TMDB
    if app.config['TMDB_API_KEY']:
        tmdb_movies = recommendation_engine.tmdb_client.search(query, 'movie')
        tmdb_tv = recommendation_engine.tmdb_client.search(query, 'tv')
        
        if tmdb_movies:
            results['tmdb'].extend(tmdb_movies.get('results', [])[:5])
        if tmdb_tv:
            results['tmdb'].extend(tmdb_tv.get('results', [])[:5])
    
    # Search OMDB
    if app.config['OMDB_API_KEY']:
        omdb_result = recommendation_engine.omdb_client.search(query)
        if omdb_result and omdb_result.get('Response') == 'True':
            results['omdb'].append(omdb_result)
    
    # Search Jikan
    jikan_results = recommendation_engine.jikan_client.search(query)
    if jikan_results and 'data' in jikan_results:
        results['jikan'] = jikan_results['data'][:5]
    
    # Search database
    db_results = Content.query.filter(
        Content.title.ilike(f'%{query}%')
    ).limit(10).all()
    
    results['database'] = [
        recommendation_engine._serialize_content(content) for content in db_results
    ]
    
    return jsonify(results), 200

# Background tasks
def update_popularity_scores():
    """Update content popularity scores based on various metrics"""
    with app.app_context():
        contents = Content.query.all()
        
        for content in contents:
            # Calculate popularity score
            view_weight = 1.0
            favorite_weight = 3.0
            rating_weight = 2.0
            recent_view_weight = 5.0
            
            # Base metrics
            view_score = content.view_count * view_weight
            favorite_count = Favorite.query.filter_by(content_id=content.id).count()
            favorite_score = favorite_count * favorite_weight
            
            # Rating score
            avg_rating = db.session.query(func.avg(Rating.rating)).filter_by(content_id=content.id).scalar()
            rating_score = (avg_rating or 0) * rating_weight * 10
            
            # Recent views (last 7 days)
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            recent_views = WatchHistory.query.filter(
                and_(
                    WatchHistory.content_id == content.id,
                    WatchHistory.watched_at >= seven_days_ago
                )
            ).count()
            recent_view_score = recent_views * recent_view_weight
            
            # Calculate total popularity score
            content.popularity_score = view_score + favorite_score + rating_score + recent_view_score
        
        db.session.commit()
        logger.info("Updated popularity scores for all content")

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_popularity_scores, trigger="interval", hours=6)
scheduler.start()

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'message': 'Internal server error'}), 500

# Create tables
with app.app_context():
    db.create_all()
    
    # Create admin user if not exists
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@example.com',
            password_hash=bcrypt.generate_password_hash('admin123').decode('utf-8')
        )
        db.session.add(admin)
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True, port=5000)