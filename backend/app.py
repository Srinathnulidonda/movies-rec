# backend/app.py
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
from functools import wraps
import json
from datetime import datetime, timedelta
import requests

app = Flask(__name__)
CORS(app, 
     origins=["http://127.0.0.1:5500", 
              "http://localhost:5500", 
              "https://movies-rec.vercel.app",
              "https://backend-app-970m.onrender.com"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_rec.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your-secret-key'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)

db = SQLAlchemy(app)
jwt = JWTManager(app)

# API Keys
WATCHMODE_API_KEY = os.getenv('WATCHMODE_API_KEY', 'WtcKDji9i20pjOl5Lg0AiyG2bddfUs3nSZRZJIsY')
JUSTWATCH_API_BASE = "https://apis.justwatch.com/content"
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.getenv('OMDB_API_KEY', '52260795')
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'https://movies-rec-xmf5.onrender.com')

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    preferences = db.Column(db.JSON, default={})
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.String(50), unique=True)
    title = db.Column(db.String(200), nullable=False)
    original_title = db.Column(db.String(200))
    overview = db.Column(db.Text)
    genres = db.Column(db.JSON)
    language = db.Column(db.String(10))
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float)
    poster_path = db.Column(db.String(200))
    backdrop_path = db.Column(db.String(200))
    content_type = db.Column(db.String(20))  # movie, tv, anime
    meta_data = db.Column(db.JSON)
    popularity = db.Column(db.Float, default=0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'))
    interaction_type = db.Column(db.String(20))  # view, like, favorite, wishlist
    rating = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AdminRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'))
    category = db.Column(db.String(50))  # critics_choice, trending, featured
    priority = db.Column(db.Integer, default=1)
    expires_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class StreamingPlatform(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'))
    platform_name = db.Column(db.String(100), nullable=False)
    platform_type = db.Column(db.String(50))  # streaming, rental, purchase, free, theater
    watch_url = db.Column(db.String(500))
    price = db.Column(db.Float)
    currency = db.Column(db.String(10), default='USD')
    quality = db.Column(db.String(20))  # HD, 4K, SD
    region = db.Column(db.String(10), default='US')
    is_free = db.Column(db.Boolean, default=False)
    expires_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

# Add new model for theater showtimes
class TheaterShowtime(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'))
    theater_name = db.Column(db.String(200))
    theater_address = db.Column(db.String(300))
    showtime = db.Column(db.DateTime)
    ticket_price = db.Column(db.Float)
    booking_url = db.Column(db.String(500))
    city = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Content Aggregator Service
class ContentAggregator:
    def __init__(self):
        self.tmdb_base = "https://api.themoviedb.org/3"
        self.omdb_base = "http://www.omdbapi.com"
        self.jikan_base = "https://api.jikan.moe/v4"
        self.watchmode_base = "https://api.watchmode.com/v1"
        self.justwatch_base = "https://apis.justwatch.com/content"
        
    async def get_streaming_platforms(self, tmdb_id, content_type='movie', region='US'):
        """Get streaming platforms for content"""
        platforms = []
        
        # Method 1: TMDB Watch Providers
        try:
            url = f"{self.tmdb_base}/{content_type}/{tmdb_id}/watch/providers"
            params = {'api_key': TMDB_API_KEY}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if 'results' in data and region in data['results']:
                        region_data = data['results'][region]
                        
                        # Free platforms
                        if 'free' in region_data:
                            for provider in region_data['free']:
                                platforms.append({
                                    'platform_name': provider['provider_name'],
                                    'platform_type': 'free',
                                    'watch_url': provider.get('link', ''),
                                    'is_free': True,
                                    'logo_path': provider.get('logo_path', ''),
                                    'provider_id': provider.get('provider_id')
                                })
                        
                        # Streaming platforms
                        if 'flatrate' in region_data:
                            for provider in region_data['flatrate']:
                                platforms.append({
                                    'platform_name': provider['provider_name'],
                                    'platform_type': 'streaming',
                                    'watch_url': provider.get('link', ''),
                                    'is_free': False,
                                    'logo_path': provider.get('logo_path', ''),
                                    'provider_id': provider.get('provider_id')
                                })
                        
                        # Rental platforms
                        if 'rent' in region_data:
                            for provider in region_data['rent']:
                                platforms.append({
                                    'platform_name': provider['provider_name'],
                                    'platform_type': 'rental',
                                    'watch_url': provider.get('link', ''),
                                    'is_free': False,
                                    'logo_path': provider.get('logo_path', ''),
                                    'provider_id': provider.get('provider_id')
                                })
                        
                        # Purchase platforms
                        if 'buy' in region_data:
                            for provider in region_data['buy']:
                                platforms.append({
                                    'platform_name': provider['provider_name'],
                                    'platform_type': 'purchase',
                                    'watch_url': provider.get('link', ''),
                                    'is_free': False,
                                    'logo_path': provider.get('logo_path', ''),
                                    'provider_id': provider.get('provider_id')
                                })
                                
        except Exception as e:
            print(f"Error fetching TMDB watch providers: {e}")
        
        # Method 2: Add popular free platforms manually for better coverage
        free_platforms = [
            {'name': 'YouTube', 'url': f'https://www.youtube.com/results?search_query=', 'type': 'free'},
            {'name': 'Tubi', 'url': 'https://tubitv.com/search/', 'type': 'free'},
            {'name': 'Crackle', 'url': 'https://www.crackle.com/search?q=', 'type': 'free'},
            {'name': 'Pluto TV', 'url': 'https://pluto.tv/search?q=', 'type': 'free'},
            {'name': 'IMDb TV', 'url': 'https://www.imdb.com/find?q=', 'type': 'free'}
        ]
        
        return platforms
    
    async def get_theater_showtimes(self, tmdb_id, city='New York', content_type='movie'):
        """Get theater showtimes - using mock data for demonstration"""
        # In a real implementation, you would integrate with:
        # - Fandango API
        # - MovieTickets.com API
        # - Local theater APIs
        
        # Mock theater data for demonstration
        theaters = [
            {
                'theater_name': 'AMC Empire 25',
                'theater_address': '234 W 42nd St, New York, NY 10036',
                'showtimes': ['2:00 PM', '5:30 PM', '8:00 PM', '10:45 PM'],
                'ticket_price': 15.99,
                'booking_url': 'https://www.fandango.com/amc-empire-25-AACQX/theater-page'
            },
            {
                'theater_name': 'Regal Union Square',
                'theater_address': '850 Broadway, New York, NY 10003',
                'showtimes': ['1:15 PM', '4:00 PM', '7:30 PM', '10:15 PM'],
                'ticket_price': 14.50,
                'booking_url': 'https://www.fandango.com/regal-union-square-AABKX/theater-page'
            }
        ]
        
        return theaters
    
    async def check_if_in_theaters(self, tmdb_id, content_type='movie'):
        """Check if content is currently in theaters"""
        try:
            url = f"{self.tmdb_base}/{content_type}/{tmdb_id}"
            params = {'api_key': TMDB_API_KEY}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if content_type == 'movie' and 'release_date' in data:
                        release_date = datetime.strptime(data['release_date'], '%Y-%m-%d')
                        now = datetime.now()
                        
                        # Consider movie in theaters if released within last 3 months
                        # and not yet available on streaming (simplified logic)
                        if (now - release_date).days <= 90 and (now - release_date).days >= 0:
                            return True
                    
                    return False
        except:
            return False
        
    # Add these methods to the ContentAggregator class

    async def fetch_trending(self, content_type='movie'):
        """Fetch trending content from TMDB"""
        try:
            url = f"{self.tmdb_base}/trending/{content_type}/week"
            params = {'api_key': TMDB_API_KEY}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return data.get('results', [])
        except Exception as e:
            print(f"Error fetching trending {content_type}: {e}")
            return []

    async def fetch_popular_by_genre(self, genre_id, content_type='movie'):
        """Fetch popular content by genre"""
        try:
            url = f"{self.tmdb_base}/discover/{content_type}"
            params = {
                'api_key': TMDB_API_KEY,
                'with_genres': genre_id,
                'sort_by': 'popularity.desc'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return data.get('results', [])[:10]
        except Exception as e:
            print(f"Error fetching popular content for genre {genre_id}: {e}")
            return []

    async def fetch_regional_content(self, language_code):
        """Fetch regional content by language"""
        try:
            url = f"{self.tmdb_base}/discover/movie"
            params = {
                'api_key': TMDB_API_KEY,
                'with_original_language': language_code,
                'sort_by': 'popularity.desc'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return data.get('results', [])[:10]
        except Exception as e:
            print(f"Error fetching regional content for {language_code}: {e}")
            return []

    async def fetch_anime_trending(self):
        """Fetch trending anime from Jikan API"""
        try:
            url = f"{self.jikan_base}/top/anime"
            params = {'limit': 10}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    anime_list = data.get('data', [])
                    
                    # Convert to TMDB-like format
                    formatted_anime = []
                    for anime in anime_list:
                        formatted_anime.append({
                            'id': anime.get('mal_id'),
                            'title': anime.get('title'),
                            'name': anime.get('title'),
                            'overview': anime.get('synopsis', ''),
                            'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                            'backdrop_path': anime.get('images', {}).get('jpg', {}).get('large_image_url'),
                            'vote_average': anime.get('score'),
                            'popularity': anime.get('popularity', 0),
                            'genre_ids': [genre.get('mal_id') for genre in anime.get('genres', [])],
                            'content_type': 'anime'
                        })
                    
                    return formatted_anime
        except Exception as e:
            print(f"Error fetching anime trending: {e}")
            return []

    async def get_content_details(self, content_id, content_type='movie'):
        """Get detailed content information from TMDB"""
        try:
            url = f"{self.tmdb_base}/{content_type}/{content_id}"
            params = {'api_key': TMDB_API_KEY}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return data
        except Exception as e:
            print(f"Error fetching content details for {content_id}: {e}")
            return {}
class RedisRateLimiter:
    def __init__(self, redis_url=None):
        try:
            self.redis_client = redis.from_url(redis_url or 'redis://red-d1l75ap5pdvs73bk295g:rE0xu32o3U2bNUQKz6mG7KIybWzle9xf@red-d1l75ap5pdvs73bk295g:6379')
        except:
            self.redis_client = None
    
    def is_allowed(self, key, limit=100, window=3600):
        if not self.redis_client:
            return True
        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            results = pipe.execute()
            
            return results[0] <= limit
        except:
            # If Redis fails, allow the request
            return True

# Initialize (only if Redis is available)
redis_limiter = RedisRateLimiter(os.getenv('REDIS_URL'))

# Rate limiting decorator
def redis_rate_limit(limit=100, window=3600):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if redis_limiter and redis_limiter.redis_client:
                key = f"rate_limit:{request.remote_addr}:{f.__name__}"
                if not redis_limiter.is_allowed(key, limit, window):
                    return jsonify({'error': 'Rate limit exceeded'}), 429
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Recommendation Engine
class RecommendationEngine:
    def __init__(self):
        self.content_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.content_matrix = None
        self.content_similarity = None
        
        # TMDB Genre mapping
        self.genre_map = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
            99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
            27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western", 10759: "Action & Adventure",
            10762: "Kids", 10763: "News", 10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap",
            10767: "Talk", 10768: "War & Politics"
        }
        
    def build_content_matrix(self):
        """Build content similarity matrix"""
        contents = Content.query.all()
        if not contents:
            return
            
        # Create content features
        features = []
        for content in contents:
            feature_text = f"{content.title} {content.overview or ''}"
            if content.genres:
                # Convert genre IDs to genre names
                genre_names = []
                for genre_id in content.genres:
                    if isinstance(genre_id, int) and genre_id in self.genre_map:
                        genre_names.append(self.genre_map[genre_id])
                    elif isinstance(genre_id, str):
                        genre_names.append(genre_id)
                
                if genre_names:
                    feature_text += " " + " ".join(genre_names)
            features.append(feature_text)
        
        self.content_matrix = self.content_vectorizer.fit_transform(features)
        self.content_similarity = cosine_similarity(self.content_matrix)
    
    def get_content_based_recommendations(self, user_id, limit=10):
        """Content-based filtering recommendations"""
        user_interactions = UserInteraction.query.filter_by(
            user_id=user_id, interaction_type='favorite'
        ).all()
        
        if not user_interactions:
            return []
        
        # Get user's favorite content indices
        favorite_indices = []
        for interaction in user_interactions:
            content = Content.query.get(interaction.content_id)
            if content:
                idx = Content.query.filter(Content.id <= content.id).count() - 1
                favorite_indices.append(idx)
        
        # Calculate average similarity scores
        if not favorite_indices or self.content_similarity is None:
            return []
        
        avg_similarity = np.mean(self.content_similarity[favorite_indices], axis=0)
        similar_indices = np.argsort(avg_similarity)[::-1]
        
        # Get content recommendations
        recommendations = []
        user_content_ids = {i.content_id for i in user_interactions}
        
        for idx in similar_indices:
            if len(recommendations) >= limit:
                break
            content = Content.query.offset(idx).first()
            if content and content.id not in user_content_ids:
                recommendations.append(content)
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, limit=10):
        """Collaborative filtering recommendations"""
        # Find similar users
        user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        user_content_ratings = {i.content_id: i.rating or 5 for i in user_interactions}
        
        # Find users with similar preferences
        similar_users = []
        all_users = User.query.filter(User.id != user_id).all()
        
        for user in all_users:
            other_interactions = UserInteraction.query.filter_by(user_id=user.id).all()
            other_ratings = {i.content_id: i.rating or 5 for i in other_interactions}
            
            # Calculate similarity
            common_items = set(user_content_ratings.keys()) & set(other_ratings.keys())
            if len(common_items) > 2:
                similarity = self.calculate_user_similarity(
                    user_content_ratings, other_ratings, common_items
                )
                similar_users.append((user.id, similarity))
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)
        
        # Get recommendations from similar users
        recommendations = []
        user_content_ids = set(user_content_ratings.keys())
        
        for similar_user_id, _ in similar_users[:10]:
            similar_interactions = UserInteraction.query.filter_by(
                user_id=similar_user_id
            ).filter(
                UserInteraction.rating >= 4
            ).all()
            
            for interaction in similar_interactions:
                if (interaction.content_id not in user_content_ids and
                    len(recommendations) < limit):
                    content = Content.query.get(interaction.content_id)
                    if content:
                        recommendations.append(content)
        
        return recommendations
    
    def calculate_user_similarity(self, ratings1, ratings2, common_items):
        """Calculate cosine similarity between users"""
        if not common_items:
            return 0
        
        vec1 = [ratings1[item] for item in common_items]
        vec2 = [ratings2[item] for item in common_items]
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def get_hybrid_recommendations(self, user_id, limit=10):
        """Hybrid recommendation combining multiple approaches"""
        content_recs = self.get_content_based_recommendations(user_id, limit//2)
        collab_recs = self.get_collaborative_recommendations(user_id, limit//2)
        
        # Combine and deduplicate
        all_recs = content_recs + collab_recs
        seen = set()
        unique_recs = []
        
        for rec in all_recs:
            if rec.id not in seen:
                seen.add(rec.id)
                unique_recs.append(rec)
        
        return unique_recs[:limit]

# Initialize services
aggregator = ContentAggregator()
recommender = RecommendationEngine()

# Helper functions
def async_to_sync(async_func):
    """Convert async function to sync"""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

def serialize_content(content):
    """Serialize content object"""
    return {
        'id': content.id,
        'tmdb_id': content.tmdb_id,
        'title': content.title,
        'original_title': content.original_title,
        'overview': content.overview,
        'genres': content.genres,
        'language': content.language,
        'release_date': content.release_date.isoformat() if content.release_date else None,
        'runtime': content.runtime,
        'rating': content.rating,
        'poster_path': content.poster_path,
        'backdrop_path': content.backdrop_path,
        'content_type': content.content_type,
        'popularity': content.popularity
    }




# API Routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        if not data:
            data = request.form.to_dict()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or not email or not password:
            return jsonify({'error': 'Username, email, and password are required'}), 400
        
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        token = create_access_token(identity=user.id)
        return jsonify({
            'token': token, 
            'user_id': user.id,
            'username': user.username
        })
        
    except Exception as e:
        if app.debug:
            print(f"Register error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        # Get raw data for debugging
        raw_data = request.get_data()
        content_type = request.content_type
        
        print(f"Raw data: {raw_data}")
        print(f"Content type: {content_type}")
        
        # Try multiple ways to get data
        data = None
        
        # Method 1: JSON
        try:
            data = request.get_json(force=True)
            print(f"JSON data: {data}")
        except Exception as e:
            print(f"JSON parsing failed: {e}")
        
        # Method 2: Form data
        if not data:
            try:
                data = request.form.to_dict()
                print(f"Form data: {data}")
            except Exception as e:
                print(f"Form parsing failed: {e}")
        
        # Method 3: Raw JSON parsing
        if not data and raw_data:
            try:
                import json
                data = json.loads(raw_data.decode('utf-8'))
                print(f"Raw JSON data: {data}")
            except Exception as e:
                print(f"Raw JSON parsing failed: {e}")
        
        # Method 4: Args
        if not data:
            data = request.args.to_dict()
            print(f"Args data: {data}")
        
        if not data:
            return jsonify({'error': 'No data provided', 'debug': {
                'raw_data': raw_data.decode('utf-8') if raw_data else None,
                'content_type': content_type
            }}), 400
        
        username = data.get('username') or data.get('email')
        password = data.get('password')
        
        print(f"Extracted - Username: {username}, Password: {'*' * len(password) if password else None}")
        
        if not username or not password:
            return jsonify({
                'error': 'Username and password are required',
                'received': data,
                'debug': {
                    'username_received': bool(username),
                    'password_received': bool(password)
                }
            }), 400
        
        # Check both username and email fields
        user = User.query.filter(
            (User.username == username) | (User.email == username)
        ).first()
        
        if user and check_password_hash(user.password_hash, password):
            token = create_access_token(identity=user.id)
            return jsonify({
                'token': token, 
                'user_id': user.id,
                'username': user.username
            })
        
        return jsonify({'error': 'Invalid credentials'}), 401
        
    except Exception as e:
        print(f"Login error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'debug': str(e)}), 500
    

@app.route('/api/test', methods=['GET', 'POST'])
def test_api():
    if request.method == 'POST':
        data = request.get_json()
        return jsonify({
            'message': 'Test successful',
            'received_data': data,
            'content_type': request.content_type
        })
    return jsonify({'message': 'API is working', 'method': 'GET'})


@app.route('/api/test-login', methods=['POST'])
def test_login():
    try:
        raw_data = request.get_data()
        json_data = request.get_json(force=True)
        form_data = request.form.to_dict()
        
        return jsonify({
            'raw_data': raw_data.decode('utf-8') if raw_data else None,
            'json_data': json_data,
            'form_data': form_data,
            'content_type': request.content_type,
            'headers': dict(request.headers)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/homepage')
def homepage():
    """Get homepage recommendations for non-logged users"""
    # Get trending content
    trending_movies = async_to_sync(aggregator.fetch_trending)('movie')
    trending_tv = async_to_sync(aggregator.fetch_trending)('tv')
    
    # Genre mapping
    genre_map = {
        'Action': 28, 'Comedy': 35, 'Drama': 18, 'Horror': 27,
        'Sci-Fi': 878, 'Romance': 10749
    }
    
    # Get popular by genre
    popular_by_genre = {}
    for genre, genre_id in genre_map.items():
        popular_by_genre[genre] = async_to_sync(aggregator.fetch_popular_by_genre)(genre_id)
    
    # Regional content
    regional_content = {
        'Telugu': async_to_sync(aggregator.fetch_regional_content)('te'),
        'Hindi': async_to_sync(aggregator.fetch_regional_content)('hi'),
        'Tamil': async_to_sync(aggregator.fetch_regional_content)('ta'),
        'Kannada': async_to_sync(aggregator.fetch_regional_content)('kn')
    }
    
    # Anime trending
    anime_trending = async_to_sync(aggregator.fetch_anime_trending)()
    
    # Admin curated content
    critics_choice = AdminRecommendation.query.filter_by(category='critics_choice').all()
    user_favorites = Content.query.order_by(Content.popularity.desc()).limit(10).all()
    
    return jsonify({
        'trending': {
            'movies': trending_movies[:10],
            'tv': trending_tv[:10],
            'anime': anime_trending[:10]
        },
        'popular_by_genre': popular_by_genre,
        'regional': regional_content,
        'critics_choice': [serialize_content(Content.query.get(r.content_id)) for r in critics_choice],
        'user_favorites': [serialize_content(c) for c in user_favorites]
    })

@app.route('/api/recommendations')
@jwt_required()
def get_recommendations():
    """Get personalized recommendations for logged-in users"""
    user_id = get_jwt_identity()
    
    # Get hybrid recommendations
    recommendations = recommender.get_hybrid_recommendations(user_id, 20)
    
    # Get ML service recommendations
    try:
        ml_response = requests.post(f"{ML_SERVICE_URL}/recommend", 
                                  json={'user_id': user_id}, timeout=5)
        ml_recommendations = ml_response.json().get('recommendations', [])
    except:
        ml_recommendations = []
    
    # Get user preferences
    user = User.query.get(user_id)
    preferences = user.preferences or {}
    
    # Genre-based recommendations
    favorite_genres = preferences.get('favorite_genres', [])
    genre_recommendations = []
    
    for genre in favorite_genres:
        genre_content = Content.query.filter(
            Content.genres.contains(genre)
        ).order_by(Content.popularity.desc()).limit(5).all()
        genre_recommendations.extend(genre_content)
    
    # Recent interactions analysis
    recent_interactions = UserInteraction.query.filter_by(
        user_id=user_id
    ).order_by(UserInteraction.created_at.desc()).limit(10).all()
    
    return jsonify({
        'hybrid_recommendations': [serialize_content(r) for r in recommendations],
        'ml_recommendations': ml_recommendations,
        'genre_based': [serialize_content(r) for r in genre_recommendations],
        'based_on_recent': [serialize_content(Content.query.get(i.content_id)) 
                           for i in recent_interactions if Content.query.get(i.content_id)]
    })

@app.route('/api/content/<int:content_id>')
def get_content_details(content_id):
    """Get detailed content information including watch options"""
    content = Content.query.get(content_id)
    
    # If content doesn't exist in database, try to fetch from TMDB
    if not content:
        try:
            # Fetch content details from TMDB
            details = async_to_sync(aggregator.get_content_details)(content_id, 'movie')
            
            # If TMDB returns valid data, create content in database
            if details and 'id' in details:
                content = Content(
                    tmdb_id=str(details['id']),
                    title=details.get('title', details.get('name', 'Unknown')),
                    original_title=details.get('original_title', details.get('original_name')),
                    overview=details.get('overview'),
                    genres=details.get('genre_ids', []),
                    language=details.get('original_language'),
                    release_date=datetime.strptime(details['release_date'], '%Y-%m-%d').date() 
                            if details.get('release_date') else None,
                    runtime=details.get('runtime'),
                    rating=details.get('vote_average'),
                    poster_path=details.get('poster_path'),
                    backdrop_path=details.get('backdrop_path'),
                    content_type='movie' if 'title' in details else 'tv',
                    popularity=details.get('popularity', 0)
                )
                db.session.add(content)
                db.session.commit()
            else:
                return jsonify({'error': 'Content not found'}), 404
                
        except Exception as e:
            return jsonify({'error': 'Content not found'}), 404
    
    # Get additional details from TMDB
    details = {}
    if content.tmdb_id:
        try:
            details = async_to_sync(aggregator.get_content_details)(
                content.tmdb_id, content.content_type
            )
        except:
            details = {}
    
    # Get streaming platforms and theater info
    streaming_platforms = StreamingPlatform.query.filter_by(content_id=content.id).all()
    theater_showtimes = TheaterShowtime.query.filter_by(content_id=content.id).all()
    
    # Get user reviews
    reviews = UserInteraction.query.filter_by(
        content_id=content.id
    ).filter(UserInteraction.rating.isnot(None)).all()
    
    # Similar content
    similar_content = []
    if recommender.content_similarity is not None:
        try:
            content_idx = Content.query.filter(Content.id <= content.id).count() - 1
            if content_idx < len(recommender.content_similarity):
                similar_indices = np.argsort(recommender.content_similarity[content_idx])[::-1][1:6]
                for idx in similar_indices:
                    similar = Content.query.offset(idx).first()
                    if similar:
                        similar_content.append(serialize_content(similar))
        except:
            pass
    
    # Format streaming platforms
    streaming_data = {
        'free': [],
        'paid': [],
        'all': []
    }
    
    for platform in streaming_platforms:
        platform_data = {
            'platform_name': platform.platform_name,
            'platform_type': platform.platform_type,
            'watch_url': platform.watch_url,
            'is_free': platform.is_free,
            'price': platform.price,
            'currency': platform.currency,
            'quality': platform.quality
        }
        
        if platform.is_free:
            streaming_data['free'].append(platform_data)
        else:
            streaming_data['paid'].append(platform_data)
        
        streaming_data['all'].append(platform_data)
    
    # Format theater info
    theater_data = []
    for showtime in theater_showtimes:
        theater_data.append({
            'theater_name': showtime.theater_name,
            'theater_address': showtime.theater_address,
            'showtime': showtime.showtime.strftime('%I:%M %p'),
            'ticket_price': showtime.ticket_price,
            'booking_url': showtime.booking_url
        })
    
    return jsonify({
        'content': serialize_content(content),
        'details': details,
        'reviews': [{'user_id': r.user_id, 'rating': r.rating, 'created_at': r.created_at} 
                   for r in reviews],
        'similar': similar_content,
        'watch_options': {
            'streaming_platforms': streaming_data,
            'theater_info': theater_data,
            'in_theaters': len(theater_data) > 0
        }
    })

@app.route('/api/interact', methods=['POST'])
@jwt_required()
def user_interact():
    """Record user interaction"""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    # Check if interaction already exists
    existing = UserInteraction.query.filter_by(
        user_id=user_id,
        content_id=data['content_id'],
        interaction_type=data['interaction_type']
    ).first()
    
    if existing:
        existing.rating = data.get('rating', existing.rating)
        existing.created_at = datetime.utcnow()
    else:
        interaction = UserInteraction(
            user_id=user_id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=data.get('rating')
        )
        db.session.add(interaction)
    
    db.session.commit()
    
    # Update user preferences
    user = User.query.get(user_id)
    content = Content.query.get(data['content_id'])
    
    if content and content.genres:
        preferences = user.preferences or {}
        genre_weights = preferences.get('genre_weights', {})
        
        weight_change = 1 if data['interaction_type'] in ['favorite', 'like'] else 0.5
        
        for genre in content.genres:
            genre_weights[genre] = genre_weights.get(genre, 0) + weight_change
        
        preferences['genre_weights'] = genre_weights
        user.preferences = preferences
        db.session.commit()
    
    return jsonify({'status': 'success'})

@app.route('/api/content/tmdb/<int:tmdb_id>')
def get_tmdb_content(tmdb_id):
    """Get content by TMDB ID, create if doesn't exist"""
    # Check if content exists in database
    content = Content.query.filter_by(tmdb_id=str(tmdb_id)).first()
    
    if not content:
        try:
            # Fetch from TMDB
            details = async_to_sync(aggregator.get_content_details)(tmdb_id, 'movie')
            
            if details and 'id' in details:
                content = Content(
                    tmdb_id=str(details['id']),
                    title=details.get('title', details.get('name', 'Unknown')),
                    original_title=details.get('original_title', details.get('original_name')),
                    overview=details.get('overview'),
                    genres=[g['id'] for g in details.get('genres', [])],
                    language=details.get('original_language'),
                    release_date=datetime.strptime(details['release_date'], '%Y-%m-%d').date() 
                            if details.get('release_date') else None,
                    runtime=details.get('runtime'),
                    rating=details.get('vote_average'),
                    poster_path=details.get('poster_path'),
                    backdrop_path=details.get('backdrop_path'),
                    content_type='movie' if 'title' in details else 'tv',
                    popularity=details.get('popularity', 0)
                )
                db.session.add(content)
                db.session.commit()
                
                return jsonify({
                    'content': serialize_content(content),
                    'details': details,
                    'reviews': [],
                    'similar': []
                })
        except Exception as e:
            return jsonify({'error': 'Content not found'}), 404
    
    # If content exists, return it with full details
    return get_content_details(content.id)



def admin_required(f):
    @wraps(f)
    @jwt_required()
    def decorated_function(*args, **kwargs):
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        if not user or user.preferences.get('role') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/admin/curate', methods=['POST'])
@admin_required
def admin_curate():
    """Admin content curation"""
    data = request.get_json()
    
    recommendation = AdminRecommendation(
        content_id=data['content_id'],
        category=data['category'],
        priority=data.get('priority', 1),
        expires_at=datetime.strptime(data['expires_at'], '%Y-%m-%d') if data.get('expires_at') else None
    )
    
    db.session.add(recommendation)
    db.session.commit()
    
    return jsonify({'status': 'success'})

@app.route('/api/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard with stats"""
    total_users = User.query.count()
    total_content = Content.query.count()
    total_interactions = UserInteraction.query.count()
    
    # Recent activity
    recent_users = User.query.order_by(User.created_at.desc()).limit(10).all()
    recent_interactions = UserInteraction.query.order_by(UserInteraction.created_at.desc()).limit(20).all()
    
    return jsonify({
        'stats': {
            'total_users': total_users,
            'total_content': total_content,
            'total_interactions': total_interactions
        },
        'recent_users': [{'id': u.id, 'username': u.username, 'created_at': u.created_at} for u in recent_users],
        'recent_interactions': [{'user_id': i.user_id, 'content_id': i.content_id, 'type': i.interaction_type} for i in recent_interactions]
    })




@app.route('/api/search')
def search_content():
    """Search content across all sources"""
    query = request.args.get('q', '')
    content_type = request.args.get('type', 'movie')
    
    # Search in database
    db_results = Content.query.filter(
        Content.title.contains(query) | 
        Content.overview.contains(query)
    ).limit(10).all()
    
    # Search TMDB
    tmdb_url = f"{aggregator.tmdb_base}/search/{content_type}"
    tmdb_params = {'api_key': TMDB_API_KEY, 'query': query}
    
    try:
        tmdb_response = requests.get(tmdb_url, params=tmdb_params)
        tmdb_results = tmdb_response.json().get('results', [])
        
        # Add tmdb_id to each result for frontend routing
        for result in tmdb_results:
            result['tmdb_id'] = result['id']
            
    except:
        tmdb_results = []
    
    return jsonify({
        'database_results': [serialize_content(c) for c in db_results],
        'tmdb_results': tmdb_results[:10]
    })
@app.route('/api/sync-content', methods=['POST'])
def sync_content():
    """Sync content from external APIs"""
    def sync_task():
        with app.app_context():  # Add this line
            # Sync trending content
            trending_movies = async_to_sync(aggregator.fetch_trending)('movie')
            trending_tv = async_to_sync(aggregator.fetch_trending)('tv')
            
            for item in trending_movies + trending_tv:
                existing = Content.query.filter_by(tmdb_id=str(item['id'])).first()
                if not existing:
                    content = Content(
                        tmdb_id=str(item['id']),
                        title=item['title'] if 'title' in item else item['name'],
                        original_title=item.get('original_title', item.get('original_name')),
                        overview=item.get('overview'),
                        genres=item.get('genre_ids', []),
                        language=item.get('original_language'),
                        release_date=datetime.strptime(item['release_date'], '%Y-%m-%d').date() 
                                if item.get('release_date') else None,
                        rating=item.get('vote_average'),
                        poster_path=item.get('poster_path'),
                        backdrop_path=item.get('backdrop_path'),
                        content_type='movie' if 'title' in item else 'tv',
                        popularity=item.get('popularity', 0)
                    )
                    db.session.add(content)
            
            db.session.commit()
            
            # Rebuild recommendation matrix
            recommender.build_content_matrix()    
    # Run sync in background
    thread = Thread(target=sync_task)
    thread.start()
    
    return jsonify({'status': 'sync_started'})


@app.route('/api/content/<int:content_id>/watch-options')
def get_watch_options(content_id):
    """Get streaming platforms and theater information for content"""
    content = Content.query.get_or_404(content_id)
    
    # Get streaming platforms
    streaming_platforms = []
    theater_info = []
    
    if content.tmdb_id:
        try:
            # Get streaming platforms
            platforms = async_to_sync(aggregator.get_streaming_platforms)(
                content.tmdb_id, content.content_type
            )
            
            # Check if in theaters
            in_theaters = async_to_sync(aggregator.check_if_in_theaters)(
                content.tmdb_id, content.content_type
            )
            if in_theaters:
                theater_info = async_to_sync(aggregator.get_theater_showtimes)(
                    content.tmdb_id, content_type=content.content_type
                )
            StreamingPlatform.query.filter_by(content_id=content.id).delete()            
            for platform in platforms:
                db_platform = StreamingPlatform(
                    content_id=content.id,
                    platform_name=platform['platform_name'],
                    platform_type=platform['platform_type'],
                    watch_url=platform.get('watch_url', ''),
                    is_free=platform.get('is_free', False),
                    updated_at=datetime.utcnow()
                )
                db.session.add(db_platform)            
            if theater_info:
                TheaterShowtime.query.filter_by(content_id=content.id).delete()
                for theater in theater_info:
                    for showtime in theater['showtimes']:
                        # Parse showtime (simplified)
                        showtime_dt = datetime.now().replace(
                            hour=int(showtime.split(':')[0]) % 12 + (12 if 'PM' in showtime else 0),
                            minute=int(showtime.split(':')[1].split()[0])
                        )
                        
                        db_showtime = TheaterShowtime(
                            content_id=content.id,
                            theater_name=theater['theater_name'],
                            theater_address=theater['theater_address'],
                            showtime=showtime_dt,
                            ticket_price=theater['ticket_price'],
                            booking_url=theater['booking_url']
                        )
                        db.session.add(db_showtime)
            
            db.session.commit()
            streaming_platforms = platforms
            
        except Exception as e:
            print(f"Error fetching watch options: {e}")
            db_platforms = StreamingPlatform.query.filter_by(content_id=content.id).all()
            streaming_platforms = [
                {
                    'platform_name': p.platform_name,
                    'platform_type': p.platform_type,
                    'watch_url': p.watch_url,
                    'is_free': p.is_free,
                    'price': p.price,
                    'currency': p.currency
                } for p in db_platforms
            ]
            
            db_theaters = TheaterShowtime.query.filter_by(content_id=content.id).all()
            theater_info = [
                {
                    'theater_name': t.theater_name,
                    'theater_address': t.theater_address,
                    'showtime': t.showtime.strftime('%I:%M %p'),
                    'ticket_price': t.ticket_price,
                    'booking_url': t.booking_url
                } for t in db_theaters
            ]
    free_platforms = [p for p in streaming_platforms if p.get('is_free', False)]
    paid_platforms = [p for p in streaming_platforms if not p.get('is_free', False)]
    
    return jsonify({
        'content_id': content.id,
        'title': content.title,
        'streaming_platforms': {
            'free': free_platforms,
            'paid': paid_platforms,
            'all': streaming_platforms
        },
        'theater_info': theater_info,
        'in_theaters': len(theater_info) > 0,
        'last_updated': datetime.utcnow().isoformat()
    })
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})
def create_tables():
    try:
        with app.app_context():
            db.create_all()
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
                print("Admin user created - Username: admin, Password: admin123")
            if Content.query.count() == 0:
                print("Starting initial content sync...")
                
    except Exception as e:
        print(f"Error creating tables: {e}")
create_tables()
if __name__ == '__main__':
    CORS(app)
    app.run(debug=True, port=5000)