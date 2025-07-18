import os
import secrets
import hashlib
import sqlite3
import psycopg2
import requests
import redis
import json
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import QueuePool
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import geoip2.database
from telegram import Bot
from cachetools import TTLCache
import asyncio
import aiohttp

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///movie_recommendations.db')
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Redis configuration
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
except:
    print("Redis connection failed, using in-memory cache")
    redis_client = None

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# API Keys
TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID')

# ML Service URL
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-xmf5.onrender.com')

# Database setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, poolclass=QueuePool, pool_size=20, max_overflow=0)
Session = sessionmaker(bind=engine)

# Cache setup
content_cache = TTLCache(maxsize=10000, ttl=3600)  # 1 hour cache
recommendation_cache = TTLCache(maxsize=5000, ttl=1800)  # 30 min cache

# Database Models
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_admin = Column(Boolean, default=False)
    preferred_languages = Column(Text, default='["en"]')
    preferred_genres = Column(Text, default='[]')
    region = Column(String(50))
    
    watchlist = relationship('Watchlist', back_populates='user')
    favorites = relationship('Favorite', back_populates='user')
    ratings = relationship('Rating', back_populates='user')
    search_history = relationship('SearchHistory', back_populates='user')

class Content(Base):
    __tablename__ = 'content'
    id = Column(Integer, primary_key=True)
    tmdb_id = Column(String(50))
    imdb_id = Column(String(50))
    mal_id = Column(String(50))  # MyAnimeList ID
    title = Column(String(255), nullable=False)
    original_title = Column(String(255))
    content_type = Column(String(50))  # movie, tv, anime
    language = Column(String(10))
    region = Column(String(50))
    release_date = Column(DateTime)
    runtime = Column(Integer)
    synopsis = Column(Text)
    plot = Column(Text)
    genres = Column(Text)  # JSON array
    cast_crew = Column(Text)  # JSON
    ratings = Column(Text)  # JSON with multiple rating sources
    poster_url = Column(String(500))
    backdrop_url = Column(String(500))
    trailer_urls = Column(Text)  # JSON array
    content_metadata = Column(Text)  # JSON for additional data - CHANGED THIS LINE
    popularity_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Watchlist(Base):
    __tablename__ = 'watchlist'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    content_id = Column(Integer, ForeignKey('content.id'))
    added_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship('User', back_populates='watchlist')
    content = relationship('Content')

class Favorite(Base):
    __tablename__ = 'favorites'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    content_id = Column(Integer, ForeignKey('content.id'))
    added_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship('User', back_populates='favorites')
    content = relationship('Content')

class Rating(Base):
    __tablename__ = 'ratings'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    content_id = Column(Integer, ForeignKey('content.id'))
    rating = Column(Float, nullable=False)
    review = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship('User', back_populates='ratings')
    content = relationship('Content')

class SearchHistory(Base):
    __tablename__ = 'search_history'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    session_id = Column(String(100))
    query = Column(String(255))
    results_count = Column(Integer)
    clicked_content_ids = Column(Text)  # JSON array
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship('User', back_populates='search_history')

class AdminRecommendation(Base):
    __tablename__ = 'admin_recommendations'
    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey('content.id'))
    title = Column(String(255))
    description = Column(Text)
    priority = Column(Integer, default=1)
    tags = Column(Text)  # JSON array
    expires_at = Column(DateTime)
    created_by = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    content = relationship('Content')
    admin = relationship('User')

# Create tables
Base.metadata.create_all(engine)

# Admin initialization
def initialize_admin():
    """Create default admin user if not exists"""
    db_session = Session()
    try:
        # Check if admin exists
        admin_email = os.environ.get('ADMIN_EMAIL', 'admin@movierecommender.com')
        admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
        admin_password = os.environ.get('ADMIN_PASSWORD', 'AdminPass123!')
        
        existing_admin = db_session.query(User).filter(
            (User.email == admin_email) | (User.username == admin_username)
        ).first()
        
        if not existing_admin:
            # Create admin user
            admin_user = User(
                username=admin_username,
                email=admin_email,
                password_hash=generate_password_hash(admin_password),
                is_admin=True,
                region='global',
                created_at=datetime.utcnow()
            )
            db_session.add(admin_user)
            db_session.commit()
            print(f"Admin user created: {admin_username}")
        else:
            print(f"Admin user already exists: {existing_admin.username}")
            
    except Exception as e:
        print(f"Error creating admin user: {e}")
        db_session.rollback()
    finally:
        db_session.close()

# Call admin initialization after creating tables
initialize_admin()

# Helper functions
def get_user_location(ip_address):
    """Get user's location from IP address"""
    try:
        if os.path.exists('GeoLite2-City.mmdb'):
            reader = geoip2.database.Reader('GeoLite2-City.mmdb')
            response = reader.city(ip_address)
            country = response.country.iso_code
            
            # Map countries to regions
            region_mapping = {
                'IN': 'india',
                'US': 'usa',
                'GB': 'uk',
                'JP': 'japan',
                'KR': 'korea'
            }
            
            return region_mapping.get(country, 'global')
    except:
        pass
    return 'global'

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            token = token.split(' ')[1]  # Bearer token
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            current_user_id = data['user_id']
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(current_user_id, *args, **kwargs)
    
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(current_user_id, *args, **kwargs):
        db_session = Session()
        try:
            user = db_session.query(User).filter_by(id=current_user_id).first()
            
            if not user or not user.is_admin:
                return jsonify({'message': 'Admin access required'}), 403
            
            return f(current_user_id, *args, **kwargs)
        finally:
            db_session.close()
    
    return decorated

# Cache helpers
def cache_get(key):
    if redis_client:
        return redis_client.get(key)
    return None

def cache_set(key, value, ttl=3600):
    if redis_client:
        redis_client.setex(key, ttl, value)

def cache_delete(key):
    if redis_client:
        redis_client.delete(key)

# External API Integration Functions
async def fetch_tmdb_content(query=None, content_type='movie', page=1, region=None, language=None):
    """Fetch content from TMDB API"""
    if not TMDB_API_KEY:
        return {'results': []}
        
    async with aiohttp.ClientSession() as session:
        base_url = f"https://api.themoviedb.org/3"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        
        if region:
            params['region'] = region
        if language:
            params['language'] = language
            
        if query:
            url = f"{base_url}/search/{content_type}"
            params['query'] = query
        else:
            url = f"{base_url}/{content_type}/popular"
            
        try:
            async with session.get(url, params=params) as response:
                return await response.json()
        except:
            return {'results': []}

async def fetch_omdb_details(imdb_id):
    """Fetch additional details from OMDb API"""
    if not OMDB_API_KEY:
        return {}
        
    async with aiohttp.ClientSession() as session:
        url = f"http://www.omdbapi.com/"
        params = {
            'apikey': OMDB_API_KEY,
            'i': imdb_id,
            'plot': 'full'
        }
        
        try:
            async with session.get(url, params=params) as response:
                return await response.json()
        except:
            return {}

async def fetch_anime_content(query=None, page=1):
    """Fetch anime content from Jikan API"""
    async with aiohttp.ClientSession() as session:
        base_url = "https://api.jikan.moe/v4"
        
        if query:
            url = f"{base_url}/anime"
            params = {'q': query, 'page': page}
        else:
            url = f"{base_url}/top/anime"
            params = {'page': page}
            
        try:
            async with session.get(url, params=params) as response:
                return await response.json()
        except:
            return {'data': []}

async def fetch_youtube_trailers(content_title, content_type):
    """Fetch trailer URLs from YouTube"""
    if not YOUTUBE_API_KEY:
        return []
        
    async with aiohttp.ClientSession() as session:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            'part': 'snippet',
            'q': f"{content_title} {content_type} trailer",
            'key': YOUTUBE_API_KEY,
            'maxResults': 3,
            'type': 'video'
        }
        
        try:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
            trailers = []
            if 'items' in data:
                for item in data['items']:
                    trailers.append({
                        'title': item['snippet']['title'],
                        'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                        'thumbnail': item['snippet']['thumbnails']['high']['url']
                    })
                    
            return trailers
        except:
            return []

def merge_content_data(tmdb_data, omdb_data=None, anime_data=None):
    """Merge data from multiple sources"""
    merged = {}
    
    if anime_data:
        merged.update({
            'content_type': 'anime',
            'mal_id': str(anime_data.get('mal_id')),
            'title': anime_data.get('title'),
            'synopsis': anime_data.get('synopsis'),
            'genres': [g['name'] for g in anime_data.get('genres', [])],
            'poster_url': anime_data.get('images', {}).get('jpg', {}).get('large_image_url'),
            'popularity_score': anime_data.get('score', 0)
        })
    else:
        merged.update({
            'title': tmdb_data.get('title') or tmdb_data.get('name'),
            'original_title': tmdb_data.get('original_title') or tmdb_data.get('original_name'),
            'content_type': 'movie' if 'title' in tmdb_data else 'tv',
            'language': tmdb_data.get('original_language'),
            'release_date': tmdb_data.get('release_date') or tmdb_data.get('first_air_date'),
            'synopsis': tmdb_data.get('overview'),
            'genres': [g['name'] for g in tmdb_data.get('genres', [])] if 'genres' in tmdb_data else [],
            'poster_url': f"https://image.tmdb.org/t/p/w500{tmdb_data.get('poster_path')}" if tmdb_data.get('poster_path') else None,
            'backdrop_url': f"https://image.tmdb.org/t/p/original{tmdb_data.get('backdrop_path')}" if tmdb_data.get('backdrop_path') else None,
            'tmdb_id': str(tmdb_data.get('id')),
            'popularity_score': tmdb_data.get('popularity', 0)
        })
    
    if omdb_data and omdb_data.get('Response') == 'True':
        merged.update({
            'imdb_id': omdb_data.get('imdbID'),
            'plot': omdb_data.get('Plot'),
            'runtime': int(omdb_data.get('Runtime', '0').split()[0]) if omdb_data.get('Runtime') and 'min' in omdb_data.get('Runtime') else None,
            'ratings': {
                'imdb': omdb_data.get('imdbRating'),
                'metascore': omdb_data.get('Metascore'),
                'ratings': omdb_data.get('Ratings', [])
            },
            'cast_crew': {
                'director': omdb_data.get('Director'),
                'writer': omdb_data.get('Writer'),
                'actors': omdb_data.get('Actors')
            }
        })
        
    return merged

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
@limiter.limit("5 per hour")
def register():
    db_session = Session()
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or not email or not password:
            return jsonify({'message': 'Missing required fields'}), 400
            
        # Check if user exists
        existing_user = db_session.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            return jsonify({'message': 'User already exists'}), 409
            
        # Get user region from IP
        user_ip = request.remote_addr
        region = get_user_location(user_ip)
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            region=region
        )
        
        db_session.add(new_user)
        db_session.commit()
        
        # Generate token
        token = jwt.encode({
            'user_id': new_user.id,
            'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
        }, app.config['JWT_SECRET_KEY'])
        
        return jsonify({
            'token': token,
            'user': {
                'id': new_user.id,
                'username': new_user.username,
                'email': new_user.email,
                'region': new_user.region
            }
        }), 201
        
    except Exception as e:
        db_session.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

@app.route('/api/auth/login', methods=['POST'])
def login():
    db_session = Session()
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'message': 'Missing credentials'}), 400
            
        # Find user
        user = db_session.query(User).filter(
            (User.username == username) | (User.email == username)
        ).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'message': 'Invalid credentials'}), 401
            
        # Generate token
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
        }, app.config['JWT_SECRET_KEY'])
        
        return jsonify({
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'region': user.region
            }
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

# Content Discovery Routes
@app.route('/api/content/search', methods=['GET'])
def search_content():
    db_session = Session()
    try:
        query = request.args.get('q')
        content_type = request.args.get('type', 'all')
        page = int(request.args.get('page', 1))
        language = request.args.get('language')
        region = request.args.get('region')
        
        # Get or create session ID for anonymous users
        session_id = request.headers.get('Session-ID') or session.get('session_id')
        if not session_id:
            session_id = secrets.token_urlsafe(16)
            session['session_id'] = session_id
            
        # Check cache
        cache_key = f"search:{query}:{content_type}:{page}:{language}:{region}"
        cached_results = cache_get(cache_key)
        if cached_results:
            results = json.loads(cached_results)
        else:
            results = []
            
            # Search across multiple sources
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            if content_type in ['all', 'movie', 'tv']:
                # TMDB search
                if content_type == 'all':
                    movie_data = loop.run_until_complete(
                        fetch_tmdb_content(query, 'movie', page, region, language)
                    )
                    tv_data = loop.run_until_complete(
                        fetch_tmdb_content(query, 'tv', page, region, language)
                    )
                    tmdb_results = movie_data.get('results', []) + tv_data.get('results', [])
                else:
                    tmdb_data = loop.run_until_complete(
                        fetch_tmdb_content(query, content_type, page, region, language)
                    )
                    tmdb_results = tmdb_data.get('results', [])
                    
                for item in tmdb_results:
                    content_data = merge_content_data(item)
                    results.append(content_data)
                    
            if content_type in ['all', 'anime']:
                # Anime search
                anime_data = loop.run_until_complete(fetch_anime_content(query, page))
                
                for item in anime_data.get('data', []):
                    content_data = merge_content_data({}, anime_data=item)
                    results.append(content_data)
                    
            # Cache results
            cache_set(cache_key, json.dumps(results), 3600)
            
        # Store search history
        search_history = SearchHistory(
            session_id=session_id,
            query=query,
            results_count=len(results)
        )
        
        # Get user ID if authenticated
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.split(' ')[1]
                data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
                search_history.user_id = data['user_id']
            except:
                pass
                
        db_session.add(search_history)
        db_session.commit()
        
        return jsonify({
            'results': results,
            'page': page,
            'total_results': len(results),
            'session_id': session_id
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

@app.route('/api/content/<int:content_id>', methods=['GET'])
def get_content_details(content_id):
    db_session = Session()
    try:
        # Check if content exists in database
        content = db_session.query(Content).filter_by(id=content_id).first()
        
        if not content:
            return jsonify({'message': 'Content not found'}), 404
            
        # Fetch additional details if needed
        if content.imdb_id and not content.plot:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            omdb_data = loop.run_until_complete(fetch_omdb_details(content.imdb_id))
            
            if omdb_data.get('Response') == 'True':
                content.plot = omdb_data.get('Plot')
                content.cast_crew = json.dumps({
                    'director': omdb_data.get('Director'),
                    'writer': omdb_data.get('Writer'),
                    'actors': omdb_data.get('Actors')
                })
                db_session.commit()
                
        # Fetch trailers if not available
        if not content.trailer_urls:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            trailers = loop.run_until_complete(
                fetch_youtube_trailers(content.title, content.content_type)
            )
            content.trailer_urls = json.dumps(trailers)
            db_session.commit()
            
        # Get similar content recommendations
        similar_content = []
        try:
            response = requests.post(f"{ML_SERVICE_URL}/recommend/similar", 
                                   json={'content_id': content_id})
            if response.status_code == 200:
                similar_ids = response.json().get('recommendations', [])
                similar_content = db_session.query(Content).filter(
                    Content.id.in_(similar_ids[:10])
                ).all()
        except:
            pass
            
        # Get user rating if authenticated
        user_rating = None
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.split(' ')[1]
                data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
                rating = db_session.query(Rating).filter_by(
                    user_id=data['user_id'],
                    content_id=content_id
                ).first()
                if rating:
                    user_rating = rating.rating
            except:
                pass
                
        return jsonify({
            'id': content.id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'language': content.language,
            'region': content.region,
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'runtime': content.runtime,
            'synopsis': content.synopsis,
            'plot': content.plot,
            'genres': json.loads(content.genres) if content.genres else [],
            'cast_crew': json.loads(content.cast_crew) if content.cast_crew else {},
            'ratings': json.loads(content.ratings) if content.ratings else {},
            'poster_url': content.poster_url,
            'backdrop_url': content.backdrop_url,
            'trailers': json.loads(content.trailer_urls) if content.trailer_urls else [],
            'user_rating': user_rating,
            'similar_content': [
                {
                    'id': c.id,
                    'title': c.title,
                    'poster_url': c.poster_url,
                    'content_type': c.content_type
                } for c in similar_content
            ]
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

# Recommendation Routes
@app.route('/api/recommendations/homepage', methods=['GET'])
def get_homepage_recommendations():
    db_session = Session()
    try:
        # Get session ID or user region
        session_id = request.headers.get('Session-ID') or session.get('session_id')
        user_ip = request.remote_addr
        region = get_user_location(user_ip)
        
        recommendations = {
            'trending': [],
            'popular': [],
            'regional': [],
            'anime': [],
            'by_genre': {},
            'recently_added': []
        }
        
        # Check cache
        cache_key = f"homepage:{region}:{session_id}"
        cached = cache_get(cache_key)
        if cached:
            return jsonify(json.loads(cached)), 200
            
        # Get trending content
        trending = db_session.query(Content).order_by(
            Content.popularity_score.desc()
        ).limit(20).all()
        
        recommendations['trending'] = [
            {
                'id': c.id,
                'title': c.title,
                'poster_url': c.poster_url,
                'content_type': c.content_type,
                'rating': json.loads(c.ratings).get('imdb') if c.ratings else None
            } for c in trending
        ]
        
        # Get popular by genre
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
        for genre in genres:
            genre_content = db_session.query(Content).filter(
                Content.genres.like(f'%{genre}%')
            ).order_by(Content.popularity_score.desc()).limit(10).all()
            
            recommendations['by_genre'][genre] = [
                {
                    'id': c.id,
                    'title': c.title,
                    'poster_url': c.poster_url,
                    'content_type': c.content_type
                } for c in genre_content
            ]
            
        # Get regional content
        if region == 'india':
            regional_languages = ['hi', 'te', 'ta', 'kn']
            regional_content = db_session.query(Content).filter(
                Content.language.in_(regional_languages)
            ).order_by(Content.popularity_score.desc()).limit(20).all()
            
            recommendations['regional'] = [
                {
                    'id': c.id,
                    'title': c.title,
                    'poster_url': c.poster_url,
                    'content_type': c.content_type,
                    'language': c.language
                } for c in regional_content
            ]
            
        # Get anime recommendations
        anime_content = db_session.query(Content).filter(
            Content.content_type == 'anime'
        ).order_by(Content.popularity_score.desc()).limit(20).all()
        
        recommendations['anime'] = [
            {
                'id': c.id,
                'title': c.title,
                'poster_url': c.poster_url,
                'mal_id': c.mal_id
            } for c in anime_content
        ]
        
        # Get recently added
        recent = db_session.query(Content).order_by(
            Content.created_at.desc()
        ).limit(20).all()
        
        recommendations['recently_added'] = [
            {
                'id': c.id,
                'title': c.title,
                'poster_url': c.poster_url,
                'content_type': c.content_type,
                'added': c.created_at.isoformat()
            } for c in recent
        ]
        
        # If user has search history, get personalized recommendations
        if session_id:
            search_history = db_session.query(SearchHistory).filter_by(
                session_id=session_id
            ).order_by(SearchHistory.timestamp.desc()).limit(10).all()
            
            if search_history:
                # Extract genres from search queries
                search_genres = []
                for search in search_history:
                    # Simple genre extraction logic
                    query_lower = search.query.lower()
                    for genre in genres:
                        if genre.lower() in query_lower:
                            search_genres.append(genre)
                            
                # Get recommendations based on search history
                if search_genres:
                    personalized = db_session.query(Content).filter(
                        Content.genres.like(f'%{search_genres[0]}%')
                    ).order_by(Content.popularity_score.desc()).limit(10).all()
                    
                    recommendations['personalized'] = [
                        {
                            'id': c.id,
                            'title': c.title,
                            'poster_url': c.poster_url,
                            'content_type': c.content_type
                        } for c in personalized
                    ]
                    
        # Cache recommendations
        cache_set(cache_key, json.dumps(recommendations), 1800)
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

@app.route('/api/recommendations/personalized', methods=['GET'])
@token_required
def get_personalized_recommendations(current_user_id):
    db_session = Session()
    try:
        # Check cache
        cache_key = f"personalized:{current_user_id}"
        cached = cache_get(cache_key)
        if cached:
            return jsonify(json.loads(cached)), 200
            
        recommendations = {
            'for_you': [],
            'based_on_favorites': [],
            'based_on_watchlist': [],
            'based_on_ratings': [],
            'genre_deep_dive': {},
            'mood_based': {},
            'cross_genre': [],
            'weekend_picks': [],
            'binge_worthy': []
        }
        
        # Get user data
        user = db_session.query(User).filter_by(id=current_user_id).first()
        favorites = db_session.query(Favorite).filter_by(user_id=current_user_id).all()
        watchlist = db_session.query(Watchlist).filter_by(user_id=current_user_id).all()
        ratings = db_session.query(Rating).filter_by(user_id=current_user_id).all()
        
        # Get ML recommendations
        try:
            response = requests.post(f"{ML_SERVICE_URL}/recommend/user", 
                                   json={'user_id': current_user_id})
            if response.status_code == 200:
                ml_recommendations = response.json().get('recommendations', [])
                
                for_you_content = db_session.query(Content).filter(
                    Content.id.in_(ml_recommendations[:20])
                ).all()
                
                recommendations['for_you'] = [
                    {
                        'id': c.id,
                        'title': c.title,
                        'poster_url': c.poster_url,
                        'content_type': c.content_type,
                        'match_score': 0.95  # Placeholder
                    } for c in for_you_content
                ]
        except:
            pass
            
        # Based on favorites
        if favorites:
            favorite_ids = [f.content_id for f in favorites]
            
            # Get similar content for each favorite
            similar_ids = []
            for fav_id in favorite_ids[:5]:  # Top 5 favorites
                try:
                    response = requests.post(f"{ML_SERVICE_URL}/recommend/similar", 
                                           json={'content_id': fav_id})
                    if response.status_code == 200:
                        similar_ids.extend(response.json().get('recommendations', [])[:5])
                except:
                    pass
                    
            if similar_ids:
                similar_content = db_session.query(Content).filter(
                    Content.id.in_(similar_ids)
                ).all()
                
                recommendations['based_on_favorites'] = [
                    {
                        'id': c.id,
                        'title': c.title,
                        'poster_url': c.poster_url,
                        'content_type': c.content_type
                    } for c in similar_content[:10]
                ]
                
        # Genre deep dive
        preferred_genres = json.loads(user.preferred_genres) if user.preferred_genres else []
        if not preferred_genres and favorites:
            # Extract genres from favorites
            genre_counts = {}
            for fav in favorites:
                content = fav.content
                if content.genres:
                    genres = json.loads(content.genres)
                    for genre in genres:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
                        
            preferred_genres = sorted(genre_counts.keys(), 
                                    key=lambda x: genre_counts[x], 
                                    reverse=True)[:3]
            
        for genre in preferred_genres:
            genre_content = db_session.query(Content).filter(
                Content.genres.like(f'%{genre}%')
            ).order_by(Content.popularity_score.desc()).limit(10).all()
            
            recommendations['genre_deep_dive'][genre] = [
                {
                    'id': c.id,
                    'title': c.title,
                    'poster_url': c.poster_url,
                    'content_type': c.content_type
                } for c in genre_content
            ]
            
        # Mood-based recommendations
        moods = {
            'action': ['Action', 'Adventure', 'Thriller'],
            'comedy': ['Comedy', 'Animation'],
            'drama': ['Drama', 'Romance'],
            'horror': ['Horror', 'Mystery', 'Thriller']
        }
        
        from sqlalchemy import or_
        
        for mood, genres in moods.items():
            mood_content = db_session.query(Content).filter(
                or_(*[Content.genres.like(f'%{g}%') for g in genres])
            ).order_by(Content.popularity_score.desc()).limit(10).all()
            
            recommendations['mood_based'][mood] = [
                {
                    'id': c.id,
                    'title': c.title,
                    'poster_url': c.poster_url,
                    'content_type': c.content_type
                } for c in mood_content[:5]
            ]
            
        # Weekend picks (movies with good ratings)
        weekend_picks = db_session.query(Content).filter(
            Content.content_type == 'movie',
            Content.runtime.between(90, 150)
        ).order_by(Content.popularity_score.desc()).limit(10).all()
        
        recommendations['weekend_picks'] = [
            {
                'id': c.id,
                'title': c.title,
                'poster_url': c.poster_url,
                'runtime': c.runtime
            } for c in weekend_picks
        ]
        
        # Binge-worthy series
        binge_worthy = db_session.query(Content).filter(
            Content.content_type.in_(['tv', 'anime'])
        ).order_by(Content.popularity_score.desc()).limit(10).all()
        
        recommendations['binge_worthy'] = [
            {
                'id': c.id,
                'title': c.title,
                'poster_url': c.poster_url,
                'content_type': c.content_type
            } for c in binge_worthy
        ]
        
        # Cache recommendations
        cache_set(cache_key, json.dumps(recommendations), 1800)
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

# User Interaction Routes
@app.route('/api/user/watchlist', methods=['GET', 'POST', 'DELETE'])
@token_required
def manage_watchlist(current_user_id):
    db_session = Session()
    try:
        if request.method == 'GET':
            watchlist = db_session.query(Watchlist).filter_by(
                user_id=current_user_id
            ).order_by(Watchlist.added_at.desc()).all()
            
            return jsonify({
                'watchlist': [
                    {
                        'id': w.content.id,
                        'title': w.content.title,
                        'poster_url': w.content.poster_url,
                        'content_type': w.content.content_type,
                        'added_at': w.added_at.isoformat()
                    } for w in watchlist
                ]
            }), 200
            
        elif request.method == 'POST':
            data = request.get_json()
            content_id = data.get('content_id')
            
            # Check if already in watchlist
            existing = db_session.query(Watchlist).filter_by(
                user_id=current_user_id,
                content_id=content_id
            ).first()
            
            if existing:
                return jsonify({'message': 'Already in watchlist'}), 400
                
            # Add to watchlist
            watchlist_item = Watchlist(
                user_id=current_user_id,
                content_id=content_id
            )
            db_session.add(watchlist_item)
            db_session.commit()
            
            # Clear user's recommendation cache
            cache_delete(f"personalized:{current_user_id}")
            
            return jsonify({'message': 'Added to watchlist'}), 201
            
        elif request.method == 'DELETE':
            content_id = request.args.get('content_id')
            
            watchlist_item = db_session.query(Watchlist).filter_by(
                user_id=current_user_id,
                content_id=content_id
            ).first()
            
            if watchlist_item:
                db_session.delete(watchlist_item)
                db_session.commit()
                
                # Clear user's recommendation cache
                cache_delete(f"personalized:{current_user_id}")
                
                return jsonify({'message': 'Removed from watchlist'}), 200
            else:
                return jsonify({'message': 'Not in watchlist'}), 404
                
    except Exception as e:
        db_session.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

@app.route('/api/user/favorites', methods=['GET', 'POST', 'DELETE'])
@token_required
def manage_favorites(current_user_id):
    db_session = Session()
    try:
        if request.method == 'GET':
            favorites = db_session.query(Favorite).filter_by(
                user_id=current_user_id
            ).order_by(Favorite.added_at.desc()).all()
            
            return jsonify({
                'favorites': [
                    {
                        'id': f.content.id,
                        'title': f.content.title,
                        'poster_url': f.content.poster_url,
                        'content_type': f.content.content_type,
                        'added_at': f.added_at.isoformat()
                    } for f in favorites
                ]
            }), 200
            
        elif request.method == 'POST':
            data = request.get_json()
            content_id = data.get('content_id')
            
            # Check if already favorited
            existing = db_session.query(Favorite).filter_by(
                user_id=current_user_id,
                content_id=content_id
            ).first()
            
            if existing:
                return jsonify({'message': 'Already in favorites'}), 400
                
            # Add to favorites
            favorite = Favorite(
                user_id=current_user_id,
                content_id=content_id
            )
            db_session.add(favorite)
            db_session.commit()
            
            # Update user preferences based on favorite
            content = db_session.query(Content).filter_by(id=content_id).first()
            if content and content.genres:
                user = db_session.query(User).filter_by(id=current_user_id).first()
                current_genres = json.loads(user.preferred_genres) if user.preferred_genres else []
                new_genres = json.loads(content.genres)
                
                # Update preferred genres
                for genre in new_genres:
                    if genre not in current_genres:
                        current_genres.append(genre)
                        
                user.preferred_genres = json.dumps(current_genres[:10])  # Keep top 10
                db_session.commit()
                
            # Clear user's recommendation cache
            cache_delete(f"personalized:{current_user_id}")
            
            return jsonify({'message': 'Added to favorites'}), 201
            
        elif request.method == 'DELETE':
            content_id = request.args.get('content_id')
            
            favorite = db_session.query(Favorite).filter_by(
                user_id=current_user_id,
                content_id=content_id
            ).first()
            
            if favorite:
                db_session.delete(favorite)
                db_session.commit()
                
                # Clear user's recommendation cache
                cache_delete(f"personalized:{current_user_id}")
                
                return jsonify({'message': 'Removed from favorites'}), 200
            else:
                return jsonify({'message': 'Not in favorites'}), 404
                
    except Exception as e:
        db_session.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

@app.route('/api/user/rate', methods=['POST'])
@token_required
def rate_content(current_user_id):
    db_session = Session()
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        rating_value = data.get('rating')
        review = data.get('review', '')
        
        if not content_id or rating_value is None:
            return jsonify({'message': 'Missing required fields'}), 400
            
        if not 0 <= rating_value <= 10:
            return jsonify({'message': 'Rating must be between 0 and 10'}), 400
            
        # Check if already rated
        existing_rating = db_session.query(Rating).filter_by(
            user_id=current_user_id,
            content_id=content_id
        ).first()
        
        if existing_rating:
            # Update existing rating
            existing_rating.rating = rating_value
            existing_rating.review = review
        else:
            # Create new rating
            rating = Rating(
                user_id=current_user_id,
                content_id=content_id,
                rating=rating_value,
                review=review
            )
            db_session.add(rating)
            
        db_session.commit()
        
        # Update content's average rating
        all_ratings = db_session.query(Rating).filter_by(content_id=content_id).all()
        avg_rating = sum(r.rating for r in all_ratings) / len(all_ratings)
        
        content = db_session.query(Content).filter_by(id=content_id).first()
        if content:
            ratings_data = json.loads(content.ratings) if content.ratings else {}
            ratings_data['user_average'] = round(avg_rating, 1)
            ratings_data['user_count'] = len(all_ratings)
            content.ratings = json.dumps(ratings_data)
            db_session.commit()
            
        # Clear user's recommendation cache
        cache_delete(f"personalized:{current_user_id}")
        
        # Send rating data to ML service for real-time learning
        try:
            requests.post(f"{ML_SERVICE_URL}/learn/rating", json={
                'user_id': current_user_id,
                'content_id': content_id,
                'rating': rating_value
            })
        except:
            pass
            
        return jsonify({'message': 'Rating saved'}), 200
        
    except Exception as e:
        db_session.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

# Admin Routes
@app.route('/api/admin/search-content', methods=['GET'])
@token_required
@admin_required
def admin_search_content(current_user_id):
    """Admin endpoint to search content across all sources"""
    try:
        query = request.args.get('q')
        source = request.args.get('source', 'all')  # tmdb, omdb, anime, all
        
        results = []
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if source in ['all', 'tmdb']:
            # Search TMDB
            movie_data = loop.run_until_complete(fetch_tmdb_content(query, 'movie'))
            tv_data = loop.run_until_complete(fetch_tmdb_content(query, 'tv'))
            
            for item in movie_data.get('results', []) + tv_data.get('results', []):
                results.append({
                    'source': 'tmdb',
                    'data': merge_content_data(item)
                })
                
        if source in ['all', 'anime']:
            # Search anime
            anime_data = loop.run_until_complete(fetch_anime_content(query))
            
            for item in anime_data.get('data', []):
                results.append({
                    'source': 'jikan',
                    'data': merge_content_data({}, anime_data=item)
                })
                
        return jsonify({'results': results}), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/admin/add-content', methods=['POST'])
@token_required
@admin_required
def admin_add_content(current_user_id):
    """Admin endpoint to add content to database"""
    db_session = Session()
    try:
        data = request.get_json()
        content_data = data.get('content_data')
        
        # Check if content already exists
        existing = None
        if content_data.get('tmdb_id'):
            existing = db_session.query(Content).filter_by(
                tmdb_id=content_data['tmdb_id']
            ).first()
        elif content_data.get('imdb_id'):
            existing = db_session.query(Content).filter_by(
                imdb_id=content_data['imdb_id']
            ).first()
            
        if existing:
            return jsonify({'message': 'Content already exists', 'content_id': existing.id}), 200
            
        # Create new content
        new_content = Content(
            tmdb_id=content_data.get('tmdb_id'),
            imdb_id=content_data.get('imdb_id'),
            mal_id=content_data.get('mal_id'),
            title=content_data['title'],
            original_title=content_data.get('original_title'),
            content_type=content_data['content_type'],
            language=content_data.get('language'),
            region=content_data.get('region'),
            release_date=datetime.fromisoformat(content_data['release_date']) if content_data.get('release_date') else None,
            runtime=content_data.get('runtime'),
            synopsis=content_data.get('synopsis'),
            plot=content_data.get('plot'),
            genres=json.dumps(content_data.get('genres', [])),
            cast_crew=json.dumps(content_data.get('cast_crew', {})),
            ratings=json.dumps(content_data.get('ratings', {})),
            poster_url=content_data.get('poster_url'),
            backdrop_url=content_data.get('backdrop_url'),
            popularity_score=content_data.get('popularity_score', 0)
        )
        
        db_session.add(new_content)
        db_session.commit()
        
        # Fetch trailers
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        trailers = loop.run_until_complete(
            fetch_youtube_trailers(new_content.title, new_content.content_type)
        )
        new_content.trailer_urls = json.dumps(trailers)
        db_session.commit()
        
        return jsonify({
            'message': 'Content added successfully',
            'content_id': new_content.id
        }), 201
        
    except Exception as e:
        db_session.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

@app.route('/api/admin/recommendations', methods=['GET', 'POST'])
@token_required
@admin_required
def manage_admin_recommendations(current_user_id):
    """Admin endpoint to manage public recommendations"""
    db_session = Session()
    try:
        if request.method == 'GET':
            from sqlalchemy import or_
            
            recommendations = db_session.query(AdminRecommendation).filter(
                or_(
                    AdminRecommendation.expires_at > datetime.utcnow(),
                    AdminRecommendation.expires_at.is_(None)
                )
            ).order_by(AdminRecommendation.priority.desc()).all()
            
            return jsonify({
                'recommendations': [
                    {
                        'id': r.id,
                        'content': {
                            'id': r.content.id,
                            'title': r.content.title,
                            'poster_url': r.content.poster_url,
                            'content_type': r.content.content_type
                        },
                        'title': r.title,
                        'description': r.description,
                        'priority': r.priority,
                        'tags': json.loads(r.tags) if r.tags else [],
                        'expires_at': r.expires_at.isoformat() if r.expires_at else None,
                        'created_at': r.created_at.isoformat()
                    } for r in recommendations
                ]
            }), 200
            
        elif request.method == 'POST':
            data = request.get_json()
            
            recommendation = AdminRecommendation(
                content_id=data['content_id'],
                title=data['title'],
                description=data['description'],
                priority=data.get('priority', 1),
                tags=json.dumps(data.get('tags', [])),
                expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
                created_by=current_user_id
            )
            
            db_session.add(recommendation)
            db_session.commit()
            
            # Send to Telegram if configured
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID:
                content = db_session.query(Content).filter_by(id=data['content_id']).first()
                
                bot = Bot(token=TELEGRAM_BOT_TOKEN)
                message = f"🎬 *{recommendation.title}*\n\n"
                message += f"{recommendation.description}\n\n"
                message += f"Title: {content.title}\n"
                message += f"Type: {content.content_type.title()}\n"
                if content.genres:
                    genres = json.loads(content.genres)
                    message += f"Genres: {', '.join(genres[:3])}\n"
                
                try:
                    bot.send_message(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        text=message,
                        parse_mode='Markdown'
                    )
                except:
                    pass
                    
            return jsonify({
                'message': 'Recommendation created',
                'id': recommendation.id
            }), 201
            
    except Exception as e:
        db_session.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

@app.route('/api/admin/create-admin', methods=['POST'])
@token_required
@admin_required
def create_admin_user(current_user_id):
    """Create a new admin user"""
    db_session = Session()
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or not email or not password:
            return jsonify({'message': 'Missing required fields'}), 400
            
        # Check if user exists
        existing_user = db_session.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            return jsonify({'message': 'User already exists'}), 409
            
        # Create new admin user
        new_admin = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            is_admin=True,
            region='global',
            created_at=datetime.utcnow()
        )
        
        db_session.add(new_admin)
        db_session.commit()
        
        return jsonify({
            'message': 'Admin user created successfully',
            'user': {
                'id': new_admin.id,
                'username': new_admin.username,
                'email': new_admin.email
            }
        }), 201
        
    except Exception as e:
        db_session.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

@app.route('/api/admin/promote-user/<int:user_id>', methods=['PUT'])
@token_required
@admin_required
def promote_to_admin(current_user_id, user_id):
    """Promote an existing user to admin"""
    db_session = Session()
    try:
        user = db_session.query(User).filter_by(id=user_id).first()
        
        if not user:
            return jsonify({'message': 'User not found'}), 404
            
        if user.is_admin:
            return jsonify({'message': 'User is already an admin'}), 400
            
        user.is_admin = True
        db_session.commit()
        
        return jsonify({
            'message': 'User promoted to admin successfully',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            }
        }), 200
        
    except Exception as e:
        db_session.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

# Public Recommendations Route
@app.route('/api/public/recommendations', methods=['GET'])
def get_public_recommendations():
    """Get admin-curated public recommendations"""
    db_session = Session()
    try:
        from sqlalchemy import or_
        
        recommendations = db_session.query(AdminRecommendation).filter(
            or_(
                AdminRecommendation.expires_at > datetime.utcnow(),
                AdminRecommendation.expires_at.is_(None)
            )
        ).order_by(AdminRecommendation.priority.desc()).limit(20).all()
        
        return jsonify({
            'recommendations': [
                {
                    'id': r.id,
                    'content': {
                        'id': r.content.id,
                        'title': r.content.title,
                        'poster_url': r.content.poster_url,
                        'content_type': r.content.content_type,
                        'synopsis': r.content.synopsis,
                        'genres': json.loads(r.content.genres) if r.content.genres else []
                    },
                    'title': r.title,
                    'description': r.description,
                    'tags': json.loads(r.tags) if r.tags else []
                } for r in recommendations
            ]
        }), 200
        
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        db_session.close()

# Background Tasks
scheduler = BackgroundScheduler()

def update_trending_content():
    """Background task to update trending content"""
    db_session = Session()
    try:
        # Fetch trending from TMDB
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        trending_movies = loop.run_until_complete(
            fetch_tmdb_content(content_type='movie')
        )
        trending_tv = loop.run_until_complete(
            fetch_tmdb_content(content_type='tv')
        )
        
        # Update database
        for item in trending_movies.get('results', []) + trending_tv.get('results', []):
            tmdb_id = str(item['id'])
            existing = db_session.query(Content).filter_by(tmdb_id=tmdb_id).first()
            
            if existing:
                existing.popularity_score = item.get('popularity', 0)
            else:
                # Add new trending content
                content_data = merge_content_data(item)
                new_content = Content(
                    tmdb_id=tmdb_id,
                    title=content_data['title'],
                    original_title=content_data.get('original_title'),
                    content_type=content_data['content_type'],
                    language=content_data.get('language'),
                    synopsis=content_data.get('synopsis'),
                    genres=json.dumps(content_data.get('genres', [])),
                    poster_url=content_data.get('poster_url'),
                    backdrop_url=content_data.get('backdrop_url'),
                    popularity_score=content_data.get('popularity_score', 0)
                )
                db_session.add(new_content)
                
        db_session.commit()
        
        # Clear trending cache
        if redis_client:
            for key in redis_client.scan_iter("homepage:*"):
                redis_client.delete(key)
        
    except Exception as e:
        print(f"Error updating trending content: {e}")
    finally:
        db_session.close()

# Schedule background tasks
scheduler.add_job(update_trending_content, 'interval', hours=6)
scheduler.start()

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)