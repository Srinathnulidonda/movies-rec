#backend/app.py 
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
from sqlalchemy import text
from sqlalchemy import text, or_

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_rec.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your-secret-key'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7689567537:AAGvDtu94OlLlTiWpfjSfpl_dd_Osi_2W7c')
TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID', '-1002566510721')
db = SQLAlchemy(app)
jwt = JWTManager(app)

# API Keys
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.getenv('OMDB_API_KEY', '52260795')
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'https://movies-rec-xmf5.onrender.com')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')
YOUTUBE_BASE_URL = "https://www.googleapis.com/youtube/v3"

# Global Genre Map
GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western", 10759: "Action & Adventure",
    10762: "Kids", 10763: "News", 10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap",
    10767: "Talk", 10768: "War & Politics"
}
REGIONAL_GENRE_MAP = {
    'telugu': ['Action', 'Comedy', 'Drama', 'Family', 'Romance', 'Thriller'],
    'hindi': ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller', 'Musical'],
    'tamil': ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller', 'Historical'],
    'kannada': ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller', 'Family']
}

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

class AdminPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'))
    admin_user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    custom_tags = db.Column(db.JSON)
    priority = db.Column(db.Integer, default=1)
    post_to_website = db.Column(db.Boolean, default=True)
    post_to_telegram = db.Column(db.Boolean, default=False)
    telegram_message_id = db.Column(db.String(50))
    expires_at = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    content = db.relationship('Content', backref='admin_posts')
    admin_user = db.relationship('User', backref='admin_posts')

class SystemAnalytics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(50), nullable=False)
    metric_value = db.Column(db.Float, nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)

# Content Aggregator Service
class ContentAggregator:
    def __init__(self):
        self.tmdb_base = "https://api.themoviedb.org/3"
        self.omdb_base = "http://www.omdbapi.com"
        self.jikan_base = "https://api.jikan.moe/v4"
        self.youtube_base = YOUTUBE_BASE_URL
        
    async def fetch_trending(self, content_type='movie', time_window='week'):
        """Fetch trending content from TMDB"""
        url = f"{self.tmdb_base}/trending/{content_type}/{time_window}"
        params = {'api_key': TMDB_API_KEY}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data.get('results', [])
    
    async def fetch_popular_by_genre(self, genre_id, content_type='movie'):
        """Fetch popular content by genre"""
        url = f"{self.tmdb_base}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': genre_id,
            'sort_by': 'popularity.desc'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data.get('results', [])
    
    async def fetch_regional_content(self, language='te'):
        """Fetch regional content"""
        url = f"{self.tmdb_base}/discover/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'with_original_language': language,
            'sort_by': 'popularity.desc'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data.get('results', [])
    
    async def fetch_anime_trending(self):
        """Fetch trending anime from Jikan API"""
        url = f"{self.jikan_base}/top/anime"
        params = {'filter': 'bypopularity', 'limit': 20}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data.get('data', [])
    
    async def get_content_details(self, content_id, content_type='movie'):
        """Get detailed content information"""
        url = f"{self.tmdb_base}/{content_type}/{content_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'append_to_response': 'credits,videos,similar,reviews'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                return await response.json()
            
    async def fetch_youtube_videos(self, search_query, content_type='trailer'):
        """Fetch trailers and teasers from YouTube"""
        search_terms = f"{search_query} {content_type}"
        url = f"{self.youtube_base}/search"
        params = {
            'part': 'snippet',
            'q': search_terms,
            'type': 'video',
            'key': YOUTUBE_API_KEY,
            'maxResults': 5,
            'order': 'relevance'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                videos = []
                for item in data.get('items', []):
                    videos.append({
                        'video_id': item['id']['videoId'],
                        'title': item['snippet']['title'],
                        'thumbnail': item['snippet']['thumbnails']['high']['url'],
                        'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                        'type': content_type
                    })
                return videos
    async def fetch_enhanced_trending(self, content_type='movie', region='US'):
        """Enhanced trending with regional support"""
        url = f"{self.tmdb_base}/trending/{content_type}/week"
        params = {
            'api_key': TMDB_API_KEY,
            'region': region
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                results = data.get('results', [])
                
                # Enhance with YouTube videos
                for item in results:
                    title = item.get('title', item.get('name', ''))
                    item['youtube_videos'] = await self.fetch_youtube_videos(title)
                    
                return results

    async def fetch_critics_choice(self, content_type='movie'):
        """Fetch critically acclaimed content"""
        url = f"{self.tmdb_base}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'sort_by': 'vote_average.desc',
            'vote_count.gte': 1000,
            'with_watch_monetization_types': 'flatrate'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data.get('results', [])

    async def fetch_whats_hot(self, time_window='day'):
        """Fetch what's hot based on popularity"""
        url = f"{self.tmdb_base}/trending/all/{time_window}"
        params = {'api_key': TMDB_API_KEY}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data.get('results', [])


# Redis Rate Limiter
class RedisRateLimiter:
    def __init__(self, redis_url=None):
        try:
            self.redis_client = redis.from_url(redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379'))
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
        self.user_similarity_cache = {}

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
                    if isinstance(genre_id, int) and genre_id in GENRE_MAP:
                        genre_names.append(GENRE_MAP[genre_id])
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
    def get_user_preference_vector(self, user_id):
        """Generate user preference vector based on interactions"""
        interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        genre_weights = defaultdict(float)
        total_interactions = len(interactions)
        
        for interaction in interactions:
            content = Content.query.get(interaction.content_id)
            if content and content.genres:
                weight = 1.0
                if interaction.interaction_type == 'favorite':
                    weight = 2.0
                elif interaction.interaction_type == 'like':
                    weight = 1.5
                elif interaction.rating:
                    weight = interaction.rating / 5.0
                
                for genre in content.genres:
                    genre_name = GENRE_MAP.get(genre, str(genre))
                    genre_weights[genre_name] += weight
        
        # Normalize weights
        if total_interactions > 0:
            for genre in genre_weights:
                genre_weights[genre] /= total_interactions
        
        return dict(genre_weights)

    def get_watch_history_recommendations(self, user_id, limit=10):
        """Get recommendations based on watch history"""
        recent_interactions = UserInteraction.query.filter_by(
            user_id=user_id, interaction_type='view'
        ).order_by(UserInteraction.created_at.desc()).limit(20).all()
        
        genre_scores = defaultdict(float)
        viewed_content_ids = set()
        
        for interaction in recent_interactions:
            content = Content.query.get(interaction.content_id)
            viewed_content_ids.add(content.id)
            
            if content and content.genres:
                for genre in content.genres:
                    genre_scores[genre] += 1.0
        
        # Get recommendations based on preferred genres
        recommendations = []
        for genre, score in sorted(genre_scores.items(), key=lambda x: x[1], reverse=True):
            genre_content = Content.query.filter(
                Content.genres.contains(genre),
                ~Content.id.in_(viewed_content_ids)
            ).order_by(Content.popularity.desc()).limit(5).all()
            
            recommendations.extend(genre_content)
            if len(recommendations) >= limit:
                break
        
        return recommendations[:limit]

    def get_regional_recommendations(self, user_id, language='te', limit=10):
        """Get regional content recommendations"""
        user_preferences = self.get_user_preference_vector(user_id)
        
        regional_content = Content.query.filter_by(language=language).order_by(
            Content.popularity.desc()
        ).limit(limit * 2).all()
        
        # Score based on user preferences
        scored_content = []
        for content in regional_content:
            score = 0
            if content.genres:
                for genre in content.genres:
                    genre_name = GENRE_MAP.get(genre, str(genre))
                    score += user_preferences.get(genre_name, 0)
            
            scored_content.append((content, score))
        
        # Sort by score and return top items
        scored_content.sort(key=lambda x: x[1], reverse=True)
        return [content for content, _ in scored_content[:limit]]


# Initialize services
aggregator = ContentAggregator()
recommender = RecommendationEngine()

# Helper functions
def async_to_sync(async_func):
    """Convert async function to sync"""
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, use asyncio.run_coroutine_threadsafe
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_func(*args, **kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(async_func(*args, **kwargs))
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_func(*args, **kwargs))
            finally:
                loop.close()
                asyncio.set_event_loop(None)
    return wrapper


def serialize_content(content):
    """Enhanced content serialization"""
    return {
        'id': content.id,
        'tmdb_id': content.tmdb_id,
        'title': content.title,
        'original_title': content.original_title,
        'overview': content.overview,
        'genres': content.genres,
        'genre_names': [GENRE_MAP.get(g, str(g)) for g in (content.genres or [])],
        'language': content.language,
        'release_date': content.release_date.isoformat() if content.release_date else None,
        'runtime': content.runtime,
        'rating': content.rating,
        'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
        'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path else None,
        'content_type': content.content_type,
        'popularity': content.popularity,
        'updated_at': content.updated_at.isoformat() if content.updated_at else None
    }

def create_content_from_tmdb(tmdb_data, content_type='movie'):
    """Helper function to create content from TMDB data"""
    return Content(
        tmdb_id=str(tmdb_data['id']),
        title=tmdb_data.get('title', tmdb_data.get('name', 'Unknown')),
        original_title=tmdb_data.get('original_title', tmdb_data.get('original_name')),
        overview=tmdb_data.get('overview'),
        genres=tmdb_data.get('genre_ids', []) if 'genre_ids' in tmdb_data else [g['id'] for g in tmdb_data.get('genres', [])],
        language=tmdb_data.get('original_language'),
        release_date=datetime.strptime(tmdb_data['release_date'], '%Y-%m-%d').date() 
                if tmdb_data.get('release_date') else None,
        runtime=tmdb_data.get('runtime'),
        rating=tmdb_data.get('vote_average'),
        poster_path=tmdb_data.get('poster_path'),
        backdrop_path=tmdb_data.get('backdrop_path'),
        content_type=content_type,
        popularity=tmdb_data.get('popularity', 0)
    )
def create_content_from_anime(anime_data):
    """Helper function to create content from anime data"""
    return Content(
        tmdb_id=str(anime_data['mal_id']),
        title=anime_data.get('title', anime_data.get('title_english', 'Unknown')),
        original_title=anime_data.get('title_japanese'),
        overview=anime_data.get('synopsis'),
        genres=[g['mal_id'] for g in anime_data.get('genres', [])],
        language='ja',
        release_date=datetime.strptime(anime_data['aired']['from'][:10], '%Y-%m-%d').date() 
                    if anime_data.get('aired') and anime_data['aired'].get('from') else None,
        runtime=anime_data.get('duration'),
        rating=anime_data.get('score'),
        poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
        backdrop_path=anime_data.get('images', {}).get('jpg', {}).get('large_image_url'),
        content_type='anime',
        popularity=anime_data.get('popularity', 0)
    )
class TelegramService:
    def __init__(self, bot_token=None, channel_id=None):
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.channel_id = channel_id or TELEGRAM_CHANNEL_ID
        self.bot = telegram.Bot(token=self.bot_token) if self.bot_token else None
    
    def send_recommendation(self, content, admin_post):
        """Send recommendation to Telegram channel"""
        if not self.bot or not self.channel_id:
            return False
        
        try:
            message = f"🎬 **{admin_post.title}**\n\n"
            message += f"📺 {content.title}"
            if content.release_date:
                message += f" ({content.release_date.year})"
            message += f"\n\n{admin_post.description}"
            
            if content.genres:
                genres = [GENRE_MAP.get(g, str(g)) for g in content.genres]
                message += f"\n\n🎭 **Genres:** {', '.join(genres)}"
            
            if content.rating:
                message += f"\n⭐ **Rating:** {content.rating}/10"
            
            if admin_post.custom_tags:
                message += f"\n\n🏷️ **Tags:** {', '.join(admin_post.custom_tags)}"
            
            sent_message = self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode='Markdown'
            )
            
            return sent_message.message_id
        except Exception as e:
            print(f"Error sending to Telegram: {e}")
            return False

# Initialize Telegram service
telegram_service = TelegramService()

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
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data:
            data = request.form.to_dict()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        username = data.get('username') or data.get('email')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        
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
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/homepage')
@redis_rate_limit(limit=50, window=300)  # 50 requests per 5 minutes
def enhanced_homepage():
    """Enhanced homepage with comprehensive recommendations"""
    try:
        # Trending content with YouTube videos
        trending_movies = async_to_sync(aggregator.fetch_enhanced_trending)('movie')
        trending_tv = async_to_sync(aggregator.fetch_enhanced_trending)('tv')
        
        # What's Hot section
        whats_hot = async_to_sync(aggregator.fetch_whats_hot)('day')
        
        # Critics' Choice
        critics_choice = async_to_sync(aggregator.fetch_critics_choice)('movie')
        
        # Regional content for all languages
        regional_content = {}
        for lang_code, lang_name in [('te', 'Telugu'), ('hi', 'Hindi'), ('ta', 'Tamil'), ('kn', 'Kannada')]:
            regional_content[lang_name] = async_to_sync(aggregator.fetch_regional_content)(lang_code)
        
        # User Favorites (most interacted content)
        user_favorites = db.session.query(
            Content, db.func.count(UserInteraction.id).label('interaction_count')
        ).join(UserInteraction).group_by(Content.id).order_by(
            db.func.count(UserInteraction.id).desc()
        ).limit(10).all()
        
        # Admin curated content
        admin_curated = AdminPost.query.filter_by(
            is_active=True, post_to_website=True
        ).filter(
            db.or_(AdminPost.expires_at.is_(None), AdminPost.expires_at > datetime.utcnow())
        ).order_by(AdminPost.priority.desc()).limit(15).all()
        
        curated_content = []
        for post in admin_curated:
            content_data = serialize_content(post.content)
            content_data.update({
                'admin_title': post.title,
                'admin_description': post.description,
                'custom_tags': post.custom_tags,
                'priority': post.priority
            })
            curated_content.append(content_data)
        
        return jsonify({
            'trending': {
                'movies': trending_movies[:12],
                'tv': trending_tv[:12],
                'anime': async_to_sync(aggregator.fetch_anime_trending)()[:12]
            },
            'whats_hot': whats_hot[:15],
            'critics_choice': critics_choice[:10],
            'regional': regional_content,
            'user_favorites': [serialize_content(content) for content, _ in user_favorites],
            'admin_curated': curated_content
        })
    except Exception as e:
        return jsonify({'error': 'Failed to fetch homepage data'}), 500
    
@app.route('/api/recommendations/personalized')
@jwt_required()
@redis_rate_limit(limit=30, window=300)
def enhanced_personalized_recommendations():
    """Enhanced personalized recommendations"""
    user_id = get_jwt_identity()
    
    try:
        # Watch history recommendations
        watch_history_recs = recommender.get_watch_history_recommendations(user_id, 15)
        
        # Favorites-based recommendations
        favorites_recs = recommender.get_content_based_recommendations(user_id, 15)
        
        # Wishlist-influenced suggestions
        wishlist_items = UserInteraction.query.filter_by(
            user_id=user_id, interaction_type='wishlist'
        ).all()
        
        wishlist_recs = []
        for item in wishlist_items:
            content = Content.query.get(item.content_id)
            if content and content.genres:
                similar_content = Content.query.filter(
                    Content.genres.overlap(content.genres),
                    Content.id != content.id
                ).order_by(Content.popularity.desc()).limit(3).all()
                wishlist_recs.extend(similar_content)
        
        # Regional recommendations based on user location
        regional_recs = recommender.get_regional_recommendations(user_id, 'te', 10)
        
        # Collaborative filtering
        collab_recs = recommender.get_collaborative_recommendations(user_id, 15)
        
        # Hybrid approach
        hybrid_recs = recommender.get_hybrid_recommendations(user_id, 20)
        
        return jsonify({
            'watch_history_based': [serialize_content(r) for r in watch_history_recs],
            'favorites_based': [serialize_content(r) for r in favorites_recs],
            'wishlist_influenced': [serialize_content(r) for r in wishlist_recs[:10]],
            'regional_suggestions': [serialize_content(r) for r in regional_recs],
            'collaborative_filtering': [serialize_content(r) for r in collab_recs],
            'hybrid_recommendations': [serialize_content(r) for r in hybrid_recs]
        })
    except Exception as e:
        return jsonify({'error': 'Failed to generate recommendations'}), 500


@app.route('/api/recommendations')
@jwt_required()
def get_recommendations():
    """Get personalized recommendations for logged-in users"""
    user_id = get_jwt_identity()
    
    # Get hybrid recommendations
    try:
        recommendations = recommender.get_hybrid_recommendations(user_id, 20)
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        recommendations = []
    
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
    """Get detailed content information"""
    content = Content.query.get(content_id)
    
    if not content:
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
    
    return jsonify({
        'content': serialize_content(content),
        'details': details,
        'reviews': [{'user_id': r.user_id, 'rating': r.rating, 'created_at': r.created_at} 
                   for r in reviews],
        'similar': similar_content
    })

@app.route('/api/content/<int:content_id>/details')
@redis_rate_limit(limit=100, window=300)
def enhanced_content_details(content_id):
    """Enhanced content details with YouTube integration"""
    content = Content.query.get_or_404(content_id)
    
    try:
        # Get YouTube videos for this content
        title = content.title
        trailers = async_to_sync(aggregator.fetch_youtube_videos)(f"{title} trailer")
        teasers = async_to_sync(aggregator.fetch_youtube_videos)(f"{title} teaser")
        
        # Get TMDB details
        tmdb_details = {}
        if content.tmdb_id:
            tmdb_details = async_to_sync(aggregator.get_content_details)(
                content.tmdb_id, content.content_type
            )
        
        # Get user reviews and ratings
        reviews = db.session.query(
            UserInteraction, User.username
        ).join(User).filter(
            UserInteraction.content_id == content.id,
            UserInteraction.rating.isnot(None)
        ).order_by(UserInteraction.created_at.desc()).limit(10).all()
        
        # Similar content recommendations
        similar_content = []
        if content.genres:
            similar_content = Content.query.filter(
                Content.genres.overlap(content.genres),
                Content.id != content.id
            ).order_by(Content.popularity.desc()).limit(8).all()
        
        return jsonify({
            'content': serialize_content(content),
            'tmdb_details': tmdb_details,
            'youtube_videos': {
                'trailers': trailers,
                'teasers': teasers
            },
            'user_reviews': [{
                'username': username,
                'rating': interaction.rating,
                'created_at': interaction.created_at.isoformat(),
                'interaction_type': interaction.interaction_type
            } for interaction, username in reviews],
            'similar_content': [serialize_content(c) for c in similar_content]
        })
    except Exception as e:
        return jsonify({'error': 'Failed to fetch content details'}), 500


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

@app.route('/api/admin/enhanced-browse')
@admin_required
def enhanced_admin_browse():
    """Enhanced admin content browsing with multi-source support"""
    source = request.args.get('source', 'tmdb')
    query = request.args.get('q', '')
    content_type = request.args.get('type', 'movie')
    language = request.args.get('language', 'en')
    page = int(request.args.get('page', 1))
    
    try:
        if source == 'tmdb':
            if query:
                url = f"{aggregator.tmdb_base}/search/{content_type}"
                params = {
                    'api_key': TMDB_API_KEY,
                    'query': query,
                    'page': page,
                    'language': language
                }
            else:
                url = f"{aggregator.tmdb_base}/discover/{content_type}"
                params = {
                    'api_key': TMDB_API_KEY,
                    'page': page,
                    'sort_by': 'popularity.desc',
                    'with_original_language': language if language != 'en' else None
                }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            # Enhance results with YouTube videos
            for item in data.get('results', []):
                title = item.get('title', item.get('name', ''))
                item['youtube_videos'] = async_to_sync(aggregator.fetch_youtube_videos)(title)
            
            return jsonify({
                'source': 'tmdb',
                'results': data.get('results', []),
                'total_pages': data.get('total_pages', 1),
                'current_page': page
            })
        
        elif source == 'regional':
            regional_content = async_to_sync(aggregator.fetch_regional_content)(language)
            return jsonify({
                'source': 'regional',
                'results': regional_content,
                'language': language
            })
        
        elif source == 'anime':
            if query:
                url = f"{aggregator.jikan_base}/anime"
                params = {'q': query, 'page': page}
            else:
                url = f"{aggregator.jikan_base}/top/anime"
                params = {'page': page}
            
            response = requests.get(url, params=params)
            data = response.json()
            
            return jsonify({
                'source': 'anime',
                'results': data.get('data', []),
                'pagination': data.get('pagination', {}),
                'current_page': page
            })
        
        else:
            return jsonify({'error': 'Invalid source'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Failed to browse content: {str(e)}'}), 500
    
@app.route('/api/admin/create-post', methods=['POST'])
@admin_required
def admin_create_post():
    """Create admin curated post"""
    user_id = get_jwt_identity()
    data = request.get_json()
    content = None
    if data.get('content_id'):
        content = Content.query.get(data['content_id'])
    elif data.get('tmdb_id'):
        content = Content.query.filter_by(tmdb_id=str(data['tmdb_id'])).first()
        if not content:
            # Create content from TMDB data
            try:
                tmdb_data = data.get('tmdb_data', {})
                content = create_content_from_tmdb(tmdb_data, 'movie' if 'title' in tmdb_data else 'tv')
                db.session.add(content)
                db.session.flush()
            except Exception as e:
                return jsonify({'error': 'Failed to create content'}), 500
    
    if not content:
        return jsonify({'error': 'Content not found'}), 404
    
    # Create admin post
    admin_post = AdminPost(
        content_id=content.id,
        admin_user_id=user_id,
        title=data['title'],
        description=data.get('description', ''),
        custom_tags=data.get('custom_tags', []),
        priority=data.get('priority', 1),
        post_to_website=data.get('post_to_website', True),
        post_to_telegram=data.get('post_to_telegram', False),
        expires_at=datetime.strptime(data['expires_at'], '%Y-%m-%d') if data.get('expires_at') else None
    )
    db.session.add(admin_post)
    db.session.commit()
    
    # Send to Telegram if requested
    telegram_message_id = None
    if admin_post.post_to_telegram:
        telegram_message_id = telegram_service.send_recommendation(content, admin_post)
        if telegram_message_id:
            admin_post.telegram_message_id = str(telegram_message_id)
            db.session.commit()
    
    return jsonify({
        'status': 'success',
        'post_id': admin_post.id,
        'telegram_sent': bool(telegram_message_id)
    })

@app.route('/api/admin/posts')
@admin_required
def admin_get_posts():
    """Get all admin posts"""
    posts = AdminPost.query.order_by(AdminPost.created_at.desc()).all()
    
    return jsonify({
        'posts': [{
            'id': post.id,
            'title': post.title,
            'description': post.description,
            'content': serialize_content(post.content),
            'custom_tags': post.custom_tags,
            'priority': post.priority,
            'post_to_website': post.post_to_website,
            'post_to_telegram': post.post_to_telegram,
            'telegram_message_id': post.telegram_message_id,
            'expires_at': post.expires_at.isoformat() if post.expires_at else None,
            'is_active': post.is_active,
            'created_at': post.created_at.isoformat(),
            'admin_user': post.admin_user.username
        } for post in posts]
    })

@app.route('/api/admin/posts/<int:post_id>', methods=['PUT'])
@admin_required
def admin_update_post(post_id):
    """Update admin post"""
    post = AdminPost.query.get_or_404(post_id)
    data = request.get_json()
    post.title = data.get('title', post.title)
    post.description = data.get('description', post.description)
    post.custom_tags = data.get('custom_tags', post.custom_tags)
    post.priority = data.get('priority', post.priority)
    post.is_active = data.get('is_active', post.is_active)
    post.expires_at = datetime.strptime(data['expires_at'], '%Y-%m-%d') if data.get('expires_at') else None
    
    db.session.commit()
    
    return jsonify({'status': 'success'})

@app.route('/api/admin/posts/<int:post_id>', methods=['DELETE'])
@admin_required
def admin_delete_post(post_id):
    """Delete admin post"""
    post = AdminPost.query.get_or_404(post_id)
    db.session.delete(post)
    db.session.commit()
    
    return jsonify({'status': 'success'})

@app.route('/api/admin/analytics')
@admin_required
def admin_analytics():
    """Get detailed analytics"""
    # User analytics
    total_users = User.query.count()
    active_users = UserInteraction.query.filter(
        UserInteraction.created_at >= datetime.utcnow() - timedelta(days=30)
    ).distinct(UserInteraction.user_id).count()
    
    # Content analytics
    total_content = Content.query.count()
    popular_content = db.session.query(
        Content.title, 
        db.func.count(UserInteraction.id).label('interaction_count')
    ).join(UserInteraction).group_by(Content.id).order_by(
        db.func.count(UserInteraction.id).desc()
    ).limit(10).all()
    
    # Interaction analytics
    interactions_by_type = db.session.query(
        UserInteraction.interaction_type,
        db.func.count(UserInteraction.id).label('count')
    ).group_by(UserInteraction.interaction_type).all()
    
    # Genre preferences
    genre_stats = defaultdict(int)
    interactions = UserInteraction.query.filter_by(interaction_type='favorite').all()
    for interaction in interactions:
        content = Content.query.get(interaction.content_id)
        if content and content.genres:
            for genre in content.genres:
                genre_name = GENRE_MAP.get(genre, str(genre))
                genre_stats[genre_name] += 1
    
    # Admin posts analytics
    admin_posts_count = AdminPost.query.count()
    active_admin_posts = AdminPost.query.filter_by(is_active=True).count()
    telegram_posts = AdminPost.query.filter_by(post_to_telegram=True).count()
    
    return jsonify({
        'users': {
            'total': total_users,
            'active_monthly': active_users,
            'engagement_rate': (active_users / total_users * 100) if total_users > 0 else 0
        },
        'content': {
            'total': total_content,
            'popular': [{'title': title, 'interactions': count} for title, count in popular_content]
        },
        'interactions': {
            'by_type': [{'type': type_, 'count': count} for type_, count in interactions_by_type],
            'total': UserInteraction.query.count()
        },
        'preferences': {
            'top_genres': dict(sorted(genre_stats.items(), key=lambda x: x[1], reverse=True)[:10])
        },
        'admin_posts': {
            'total': admin_posts_count,
            'active': active_admin_posts,
            'telegram_posts': telegram_posts
        }
    })

@app.route('/api/admin/system-status')
@admin_required
def admin_system_status():
    """Get system status and monitoring info"""
    try:
        db.session.execute(text('SELECT 1'))
        db_status = 'healthy'
    except:
        db_status = 'error'
    api_status = {}
    try:
        response = requests.get(f"{aggregator.tmdb_base}/configuration", 
                              params={'api_key': TMDB_API_KEY}, timeout=5)
        api_status['tmdb'] = 'healthy' if response.status_code == 200 else 'error'
    except:
        api_status['tmdb'] = 'error'
    telegram_status = 'disabled'
    if telegram_service.bot:
        try:
            telegram_service.bot.get_me()
            telegram_status = 'healthy'
        except:
            telegram_status = 'error'
    
    return jsonify({
        'database': db_status,
        'external_apis': api_status,
        'telegram_bot': telegram_status,
        'recommendation_matrix': 'built' if recommender.content_matrix is not None else 'empty',
        'content_count': Content.query.count(),
        'user_count': User.query.count(),
        'uptime': datetime.utcnow().isoformat()
    })

@app.route('/api/search')
def search_content():
    """Search content across all sources"""
    query = request.args.get('q', '')
    content_type = request.args.get('type', 'movie')
    db_results = Content.query.filter(
        Content.title.contains(query) | 
        Content.overview.contains(query)
    ).limit(10).all()
    tmdb_url = f"{aggregator.tmdb_base}/search/{content_type}"
    tmdb_params = {'api_key': TMDB_API_KEY, 'query': query}
    try:
        tmdb_response = requests.get(tmdb_url, params=tmdb_params)
        tmdb_results = tmdb_response.json().get('results', [])
        for result in tmdb_results:
            result['tmdb_id'] = result['id']
    except:
        tmdb_results = []
    return jsonify({
        'database_results': [serialize_content(c) for c in db_results],
        'tmdb_results': tmdb_results[:10]
    })

@app.route('/api/enhanced-sync', methods=['POST'])
@admin_required
def enhanced_sync_content():
    """Enhanced content sync with YouTube integration"""
    def enhanced_sync_task():
        with app.app_context():
            try:
                # Sync trending content
                trending_movies = async_to_sync(aggregator.fetch_enhanced_trending)('movie')
                trending_tv = async_to_sync(aggregator.fetch_enhanced_trending)('tv')
                
                for item in trending_movies + trending_tv:
                    existing = Content.query.filter_by(tmdb_id=str(item['id'])).first()
                    if not existing:
                        content = create_content_from_tmdb(item, 'movie' if 'title' in item else 'tv')
                        db.session.add(content)
                    else:
                        # Update existing content
                        existing.popularity = item.get('popularity', existing.popularity)
                        existing.rating = item.get('vote_average', existing.rating)
                        existing.updated_at = datetime.utcnow()
                
                # Sync regional content
                for lang in ['te', 'hi', 'ta', 'kn']:
                    regional_content = async_to_sync(aggregator.fetch_regional_content)(lang)
                    for item in regional_content:
                        existing = Content.query.filter_by(tmdb_id=str(item['id'])).first()
                        if not existing:
                            content = create_content_from_tmdb(item, 'movie')
                            db.session.add(content)
                
                db.session.commit()
                recommender.build_content_matrix()
                
                return True
            except Exception as e:
                print(f"Sync error: {e}")
                return False
    
    thread = Thread(target=enhanced_sync_task)
    thread.start()
    return jsonify({'status': 'enhanced_sync_started'})

# Initialize enhanced services
aggregator = ContentAggregator()
recommender = RecommendationEngine()

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
    try:
        # Environment-based configuration
        debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        port = int(os.getenv('PORT', 5000))
        host = os.getenv('HOST', '0.0.0.0' if not debug_mode else '127.0.0.1')
        
        # Initialize recommendation matrix in background
        def init_recommendations():
            try:
                with app.app_context():
                    if Content.query.count() > 0:
                        print("Building recommendation matrix...")
                        recommender.build_content_matrix()
                        print("Recommendation matrix built successfully")
            except Exception as e:
                print(f"Error building recommendation matrix: {e}")
        
        # Start recommendation matrix building in background
        init_thread = Thread(target=init_recommendations, daemon=True)
        init_thread.start()
        
        print(f"Starting Movie Recommendation API on {host}:{port}")
        print(f"Debug mode: {debug_mode}")
        print(f"Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
        
        app.run(
            debug=debug_mode,
            host=host,
            port=port,
            threaded=True,
            use_reloader=debug_mode
        )
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Failed to start application: {e}")
        exit(1)