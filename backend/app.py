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
import time
from flask_cors import CORS
import redis
from functools import lru_cache
import hashlib
import telegram
from telegram import Bot, ParseMode, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError
import html
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram import InputMediaPhoto


# App initialization
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)

# Database configuration
if os.getenv('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_recommendations.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# API Keys
TMDB_API_KEY = os.getenv('TMDB_API_KEY', 'your-tmdb-api-key')
OMDB_API_KEY = os.getenv('OMDB_API_KEY', 'your-omdb-api-key')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'your-youtube-api-key')
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'http://localhost:5001')

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    preferences = db.Column(db.JSON, default={})
    location = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.String(50))
    imdb_id = db.Column(db.String(50))
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
    trailers = db.Column(db.JSON)
    cast = db.Column(db.JSON)
    crew = db.Column(db.JSON)
    keywords = db.Column(db.JSON)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'))
    interaction_type = db.Column(db.String(20))  # view, like, favorite, wishlist, rating
    rating = db.Column(db.Float)
    watch_time = db.Column(db.Integer)  # in minutes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    session_id = db.Column(db.String(100))
    search_query = db.Column(db.String(200))
    results_count = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AdminRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'))
    category = db.Column(db.String(50))  # critics_choice, trending, featured, festival
    priority = db.Column(db.Integer, default=1)
    description = db.Column(db.Text)
    expires_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))

class UserReview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'))
    review_text = db.Column(db.Text)
    rating = db.Column(db.Float)
    likes = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TelegramChannel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    channel_id = db.Column(db.String(100), unique=True, nullable=False)
    channel_name = db.Column(db.String(200))
    channel_type = db.Column(db.String(50))  # channel, group, supergroup
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TelegramPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    admin_recommendation_id = db.Column(db.Integer, db.ForeignKey('admin_recommendation.id'))
    channel_id = db.Column(db.String(100))
    message_id = db.Column(db.Integer)  # Telegram message ID
    post_type = db.Column(db.String(50))  # instant, scheduled
    scheduled_time = db.Column(db.DateTime)
    posted_at = db.Column(db.DateTime)
    status = db.Column(db.String(20))  # pending, sent, failed
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Redis Cache Manager
class CacheManager:
    def __init__(self):
        self.redis_client = None
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
        except:
            print("Redis not available, using in-memory cache")
            self.memory_cache = {}
    
    def get(self, key):
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            except:
                return None
        return self.memory_cache.get(key)
    
    def set(self, key, value, expiry=3600):
        if self.redis_client:
            try:
                self.redis_client.setex(key, expiry, json.dumps(value))
            except:
                pass
        else:
            self.memory_cache[key] = value
    
    def delete(self, key):
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except:
                pass
        elif key in self.memory_cache:
            del self.memory_cache[key]

cache = CacheManager()

# Content Aggregator Service
class ContentAggregator:
    def __init__(self):
        self.tmdb_base = "https://api.themoviedb.org/3"
        self.omdb_base = "http://www.omdbapi.com"
        self.jikan_base = "https://api.jikan.moe/v4"
        self.youtube_base = "https://www.googleapis.com/youtube/v3"
        
    async def fetch_trending(self, content_type='movie', time_window='week', page=1):
        """Fetch trending content from TMDB"""
        cache_key = f"trending_{content_type}_{time_window}_{page}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        url = f"{self.tmdb_base}/trending/{content_type}/{time_window}"
        params = {'api_key': TMDB_API_KEY, 'page': page}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                results = data.get('results', [])
                cache.set(cache_key, results, 3600)
                return results
    
    async def fetch_popular_by_genre(self, genre_id, content_type='movie', page=1):
        """Fetch popular content by genre"""
        cache_key = f"genre_{genre_id}_{content_type}_{page}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        url = f"{self.tmdb_base}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': genre_id,
            'sort_by': 'popularity.desc',
            'page': page
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                results = data.get('results', [])
                cache.set(cache_key, results, 3600)
                return results
    
    async def fetch_regional_content(self, language='te', page=1):
        """Fetch regional content"""
        cache_key = f"regional_{language}_{page}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        url = f"{self.tmdb_base}/discover/movie"
        params = {
            'api_key': TMDB_API_KEY,
            'with_original_language': language,
            'sort_by': 'popularity.desc',
            'page': page,
            'region': 'IN'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                results = data.get('results', [])
                cache.set(cache_key, results, 3600)
                return results
    
    async def fetch_anime_trending(self, page=1):
        """Fetch trending anime from Jikan API"""
        cache_key = f"anime_trending_{page}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        url = f"{self.jikan_base}/top/anime"
        params = {'filter': 'bypopularity', 'limit': 25, 'page': page}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    results = data.get('data', [])
                    cache.set(cache_key, results, 7200)
                    return results
        except:
            return []
    
    async def get_content_details(self, content_id, content_type='movie'):
        """Get detailed content information with all metadata"""
        cache_key = f"details_{content_type}_{content_id}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        url = f"{self.tmdb_base}/{content_type}/{content_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'append_to_response': 'credits,videos,similar,reviews,keywords'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                # Fetch additional data from OMDb
                if data.get('imdb_id'):
                    omdb_data = await self.fetch_omdb_details(data['imdb_id'])
                    data['omdb'] = omdb_data
                
                # Fetch YouTube trailers
                if data.get('videos', {}).get('results'):
                    data['trailers'] = await self.fetch_youtube_trailers(data['videos']['results'])
                
                cache.set(cache_key, data, 86400)
                return data
    
    async def fetch_omdb_details(self, imdb_id):
        """Fetch additional details from OMDb"""
        url = self.omdb_base
        params = {
            'apikey': OMDB_API_KEY,
            'i': imdb_id,
            'plot': 'full'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    return await response.json()
        except:
            return {}
    
    async def fetch_youtube_trailers(self, videos):
        """Get YouTube trailer information"""
        trailers = []
        for video in videos[:5]:  # Limit to 5 trailers
            if video.get('site') == 'YouTube':
                trailers.append({
                    'key': video['key'],
                    'name': video['name'],
                    'type': video['type'],
                    'url': f"https://www.youtube.com/watch?v={video['key']}"
                })
        return trailers
    
    async def search_multi_source(self, query, page=1):
        """Search across multiple sources"""
        results = {
            'tmdb': [],
            'anime': []
        }
        
        # TMDB search
        tmdb_url = f"{self.tmdb_base}/search/multi"
        tmdb_params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'page': page,
            'include_adult': False
        }
        
        # Anime search
        anime_url = f"{self.jikan_base}/anime"
        anime_params = {
            'q': query,
            'limit': 20,
            'page': page
        }
        
        async with aiohttp.ClientSession() as session:
            # TMDB search
            try:
                async with session.get(tmdb_url, params=tmdb_params) as response:
                    data = await response.json()
                    results['tmdb'] = data.get('results', [])
            except:
                pass
            
            # Anime search
            try:
                async with session.get(anime_url, params=anime_params) as response:
                    data = await response.json()
                    results['anime'] = data.get('data', [])
            except:
                pass
        
        return results

# Telegram Service
class TelegramService:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN','7689567537:AAGvDtu94OlLlTiWpfjSfpl_dd_Osi_2W7c')
        self.bot = Bot(token=self.bot_token) if self.bot_token else None
        self.admin_channel = os.getenv('TELEGRAM_ADMIN_CHANNEL', '@movie_updates_1')
        
    def format_movie_post(self, content, recommendation):
        """Format content for Telegram post"""
        # Emoji mapping for genres
        genre_emojis = {
            'Action': '💥', 'Comedy': '😄', 'Drama': '🎭', 'Horror': '👻',
            'Romance': '❤️', 'Sci-Fi': '🚀', 'Thriller': '😱', 'Animation': '🎨',
            'Documentary': '📹', 'Fantasy': '🧙', 'Mystery': '🔍', 'Adventure': '🗺️'
        }
        
        # Build the message
        message = []
        
        # Title and rating
        title = content.title
        if content.release_date:
            title += f" ({content.release_date.year})"
        
        rating_stars = '⭐' * int(content.rating / 2) if content.rating else ''
        message.append(f"🎬 <b>{html.escape(title)}</b> {rating_stars}")
        message.append("")
        
        # Admin's custom description or overview
        if recommendation.description:
            message.append(f"<i>{html.escape(recommendation.description)}</i>")
        else:
            overview = content.overview[:300] + "..." if len(content.overview) > 300 else content.overview
            message.append(f"<i>{html.escape(overview)}</i>")
        message.append("")
        
        # Genres
        if content.genres:
            genre_text = []
            for genre in content.genres[:5]:  # Limit to 5 genres
                genre_name = genre.get('name') if isinstance(genre, dict) else str(genre)
                emoji = genre_emojis.get(genre_name, '🎭')
                genre_text.append(f"{emoji} {genre_name}")
            message.append(' '.join(genre_text))
            message.append("")
        
        # Details
        details = []
        if content.language:
            lang_map = {'en': 'English', 'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil', 'kn': 'Kannada'}
            language = lang_map.get(content.language, content.language.upper())
            details.append(f"🗣️ {language}")
        
        if content.runtime:
            hours = content.runtime // 60
            minutes = content.runtime % 60
            runtime = f"{hours}h {minutes}m" if hours else f"{minutes}m"
            details.append(f"⏱️ {runtime}")
        
        if content.rating:
            details.append(f"⭐ {content.rating}/10")
        
        if details:
            message.append(' | '.join(details))
        
        # Category tag
        category_emojis = {
            'critics_choice': '🏆 Critics\' Choice',
            'trending': '🔥 Trending Now',
            'featured': '✨ Featured',
            'festival': '🎪 Festival Special',
            'new_release': '🆕 New Release',
            'hidden_gem': '💎 Hidden Gem',
            'weekend_special': '🎉 Weekend Special'
        }
        
        category_tag = category_emojis.get(recommendation.category, f"📌 {recommendation.category}")
        message.append("")
        message.append(category_tag)
        
        return '\n'.join(message)
    
    def create_inline_keyboard(self, content):
        """Create inline keyboard with action buttons"""
        keyboard = []
        
        # Watch trailer button (if available)
        if content.trailers and len(content.trailers) > 0:
            trailer_url = content.trailers[0].get('url', '')
            if trailer_url:
                keyboard.append([InlineKeyboardButton("🎬 Watch Trailer", url=trailer_url)])
        
        # More info button (link to your website)
        website_url = os.getenv('WEBSITE_URL', 'https://movieapp.com')
        info_url = f"{website_url}/content/{content.id}"
        keyboard.append([InlineKeyboardButton("ℹ️ More Info", url=info_url)])
        
        # TMDB/IMDb links
        buttons_row = []
        if content.tmdb_id:
            tmdb_url = f"https://www.themoviedb.org/movie/{content.tmdb_id}"
            buttons_row.append(InlineKeyboardButton("TMDB", url=tmdb_url))
        
        if content.imdb_id:
            imdb_url = f"https://www.imdb.com/title/{content.imdb_id}"
            buttons_row.append(InlineKeyboardButton("IMDb", url=imdb_url))
        
        if buttons_row:
            keyboard.append(buttons_row)
        
        return InlineKeyboardMarkup(keyboard)
    
    async def send_recommendation(self, content, recommendation, channel_id=None):
        """Send recommendation to Telegram channel"""
        if not self.bot:
            return {'success': False, 'error': 'Telegram bot not configured'}
        
        channel = channel_id or self.admin_channel
        
        try:
            # Format message
            message_text = self.format_movie_post(content, recommendation)
            keyboard = self.create_inline_keyboard(content)
            
            # Send photo with caption if poster available
            if content.poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
                
                # Send photo with caption
                message = await self.bot.send_photo(
                            chat_id=channel,
                            photo=poster_url,
                            caption=message_text,
                            parse_mode=ParseMode.HTML,
                            reply_markup=keyboard
                )
            else:
                # Send text message
                message = await self.bot.send_message(
                    chat_id=channel,
                    text=message_text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=keyboard,
                    disable_web_page_preview=False
                )
            
            return {
                'success': True,
                'message_id': message.message_id,
                'channel': channel
            }
            
        except TelegramError as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def send_batch_recommendations(self, recommendations):
        """Send multiple recommendations as an album or carousel"""
        if not self.bot:
            return {'success': False, 'error': 'Telegram bot not configured'}
        
        try:
            media_group = []
            
            for i, (content, rec) in enumerate(recommendations[:10]):  # Limit to 10
                if content.poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
                    caption = self.format_movie_post(content, rec) if i == 0 else None
                    
                    media_group.append(
                        InputMediaPhoto(
                            media=poster_url,
                            caption=caption,
                            parse_mode=ParseMode.HTML
                        )
                    )
            
            if media_group:
                messages = await self.bot.send_media_group(
                    chat_id=self.admin_channel,
                    media=media_group
                )
                
                return {
                    'success': True,
                    'message_count': len(messages)
                }
            
        except TelegramError as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def schedule_post(self, content, recommendation, channel_id, scheduled_time):
        """Schedule a post for later"""
        post = TelegramPost(
            admin_recommendation_id=recommendation.id,
            channel_id=channel_id,
            post_type='scheduled',
            scheduled_time=scheduled_time,
            status='pending'
        )
        db.session.add(post)
        db.session.commit()
        
        return post

# Recommendation Engine
class RecommendationEngine:
    def __init__(self):
        self.content_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.content_matrix = None
        self.content_similarity = None
        self.user_item_matrix = None
        
    def build_content_matrix(self):
        """Build content similarity matrix"""
        contents = Content.query.all()
        if not contents:
            return
        
        # Create content features
        features = []
        for content in contents:
            feature_text = f"{content.title} {content.overview or ''}"
            
            # Add genres
            if content.genres:
                if isinstance(content.genres, list):
                    genre_names = []
                    for genre in content.genres:
                        if isinstance(genre, dict) and 'name' in genre:
                            genre_names.append(genre['name'])
                        elif isinstance(genre, str):
                            genre_names.append(genre)
                    feature_text += " " + " ".join(genre_names)
            
            # Add keywords
            if content.keywords:
                keyword_names = []
                if isinstance(content.keywords, list):
                    for keyword in content.keywords[:10]:  # Limit keywords
                        if isinstance(keyword, dict) and 'name' in keyword:
                            keyword_names.append(keyword['name'])
                    feature_text += " " + " ".join(keyword_names)
            
            # Add cast
            if content.cast:
                cast_names = []
                if isinstance(content.cast, list):
                    for actor in content.cast[:5]:  # Top 5 cast members
                        if isinstance(actor, dict) and 'name' in actor:
                            cast_names.append(actor['name'])
                    feature_text += " " + " ".join(cast_names)
            
            features.append(feature_text)
        
        self.content_matrix = self.content_vectorizer.fit_transform(features)
        self.content_similarity = cosine_similarity(self.content_matrix)
    
    def get_content_based_recommendations(self, user_id, limit=20):
        """Enhanced content-based filtering"""
        # Get user's liked content
        liked_interactions = UserInteraction.query.filter_by(
            user_id=user_id
        ).filter(
            UserInteraction.interaction_type.in_(['favorite', 'like', 'rating'])
        ).all()
        
        if not liked_interactions:
            return []
        
        # Weight interactions by type and rating
        content_weights = {}
        for interaction in liked_interactions:
            weight = 1.0
            if interaction.interaction_type == 'favorite':
                weight = 2.0
            elif interaction.interaction_type == 'rating' and interaction.rating:
                weight = interaction.rating / 5.0
            
            content_weights[interaction.content_id] = weight
        
        # Get weighted similarity scores
        weighted_similarities = np.zeros(len(Content.query.all()))
        
        for content_id, weight in content_weights.items():
            content = Content.query.get(content_id)
            if content:
                idx = Content.query.filter(Content.id <= content.id).count() - 1
                if idx < len(self.content_similarity):
                    weighted_similarities += self.content_similarity[idx] * weight
        
        # Normalize by number of interactions
        weighted_similarities /= len(content_weights)
        
        # Get top similar content
        similar_indices = np.argsort(weighted_similarities)[::-1]
        
        recommendations = []
        user_content_ids = {i.content_id for i in liked_interactions}
        
        for idx in similar_indices:
            if len(recommendations) >= limit:
                break
            content = Content.query.offset(idx).first()
            if content and content.id not in user_content_ids:
                recommendations.append(content)
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, limit=20):
        """Enhanced collaborative filtering"""
        # Get all user interactions
        all_interactions = UserInteraction.query.all()
        
        # Build user-item matrix
        user_item_dict = defaultdict(dict)
        for interaction in all_interactions:
            score = 1.0
            if interaction.interaction_type == 'favorite':
                score = 5.0
            elif interaction.interaction_type == 'like':
                score = 4.0
            elif interaction.interaction_type == 'rating' and interaction.rating:
                score = interaction.rating
            
            user_item_dict[interaction.user_id][interaction.content_id] = score
        
        # Find similar users using cosine similarity
        target_user_vector = user_item_dict[user_id]
        if not target_user_vector:
            return []
        
        similar_users = []
        for other_user_id, other_vector in user_item_dict.items():
            if other_user_id != user_id:
                similarity = self.calculate_user_similarity(target_user_vector, other_vector)
                if similarity > 0.1:  # Threshold
                    similar_users.append((other_user_id, similarity))
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)
        
        # Get recommendations from similar users
        content_scores = defaultdict(float)
        for similar_user_id, similarity in similar_users[:20]:  # Top 20 similar users
            for content_id, score in user_item_dict[similar_user_id].items():
                if content_id not in target_user_vector:
                    content_scores[content_id] += score * similarity
        
        # Sort by score and get top content
        sorted_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for content_id, _ in sorted_content[:limit]:
            content = Content.query.get(content_id)
            if content:
                recommendations.append(content)
        
        return recommendations
    
    def calculate_user_similarity(self, vector1, vector2):
        """Calculate cosine similarity between two user vectors"""
        common_items = set(vector1.keys()) & set(vector2.keys())
        if not common_items:
            return 0.0
        
        dot_product = sum(vector1[item] * vector2[item] for item in common_items)
        magnitude1 = sum(v * v for v in vector1.values()) ** 0.5
        magnitude2 = sum(v * v for v in vector2.values()) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def get_popularity_based_recommendations(self, limit=20, timeframe_days=30):
        """Get popularity-based recommendations"""
        cutoff_date = datetime.utcnow() - timedelta(days=timeframe_days)
        
        # Get recent popular content
        popular_content = db.session.query(
            Content,
            db.func.count(UserInteraction.id).label('interaction_count')
        ).join(
            UserInteraction
        ).filter(
            UserInteraction.created_at >= cutoff_date
        ).group_by(
            Content.id
        ).order_by(
            db.func.count(UserInteraction.id).desc()
        ).limit(limit).all()
        
        return [content for content, _ in popular_content]
    
    def get_genre_based_recommendations(self, user_id, limit=20):
        """Get recommendations based on user's preferred genres"""
        user = User.query.get(user_id)
        if not user or not user.preferences:
            return []
        
        # Get user's genre preferences
        genre_weights = user.preferences.get('genre_weights', {})
        if not genre_weights:
            # Calculate from interactions
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            for interaction in interactions:
                content = Content.query.get(interaction.content_id)
                if content and content.genres:
                    for genre in content.genres:
                        genre_name = genre.get('name') if isinstance(genre, dict) else genre
                        weight = 1.0
                        if interaction.interaction_type == 'favorite':
                            weight = 2.0
                        elif interaction.interaction_type == 'rating' and interaction.rating:
                            weight = interaction.rating / 5.0
                        
                        genre_weights[genre_name] = genre_weights.get(genre_name, 0) + weight
        
        if not genre_weights:
            return []
        
        # Sort genres by weight
        sorted_genres = sorted(genre_weights.items(), key=lambda x: x[1], reverse=True)
        
        # Get content for top genres
        recommendations = []
        user_content_ids = {i.content_id for i in UserInteraction.query.filter_by(user_id=user_id).all()}
        
        for genre_name, _ in sorted_genres[:5]:  # Top 5 genres
            genre_content = Content.query.filter(
                Content.genres.contains(genre_name)
            ).order_by(
                Content.popularity.desc()
            ).limit(10).all()
            
            for content in genre_content:
                if content.id not in user_content_ids and content not in recommendations:
                    recommendations.append(content)
                    if len(recommendations) >= limit:
                        return recommendations
        
        return recommendations
    
    def get_hybrid_recommendations(self, user_id, limit=30):
        """Hybrid recommendation combining multiple approaches"""
        recommendations = []
        weights = {
            'content': 0.35,
            'collaborative': 0.25,
            'genre': 0.2,
            'ml': 0.2
        }
        
        # Get recommendations from each method
        content_recs = self.get_content_based_recommendations(user_id, limit)
        collab_recs = self.get_collaborative_recommendations(user_id, limit)
        genre_recs = self.get_genre_based_recommendations(user_id, limit)
        
        # Get ML recommendations
        ml_recs = []
        try:
            response = requests.post(
                f"{ML_SERVICE_URL}/recommend",
                json={'user_id': user_id, 'limit': limit},
                timeout=5
            )
            if response.status_code == 200:
                ml_data = response.json()
                ml_content_ids = ml_data.get('recommendations', [])
                ml_recs = [Content.query.get(cid) for cid in ml_content_ids if Content.query.get(cid)]
        except:
            pass
        
        # Score each content
        content_scores = defaultdict(float)
        
        # Add scores from each method
        for i, content in enumerate(content_recs):
            content_scores[content.id] += weights['content'] * (1 - i / len(content_recs))
        
        for i, content in enumerate(collab_recs):
            content_scores[content.id] += weights['collaborative'] * (1 - i / len(collab_recs))
        
        for i, content in enumerate(genre_recs):
            content_scores[content.id] += weights['genre'] * (1 - i / len(genre_recs))
        
        for i, content in enumerate(ml_recs):
            content_scores[content.id] += weights['ml'] * (1 - i / len(ml_recs))
        
        # Sort by combined score
        sorted_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get final recommendations
        for content_id, _ in sorted_content[:limit]:
            content = Content.query.get(content_id)
            if content:
                recommendations.append(content)
        
        return recommendations
    
    def get_time_based_recommendations(self, user_id):
        """Get recommendations based on time of day and day of week"""
        current_hour = datetime.utcnow().hour
        current_day = datetime.utcnow().weekday()
        
        recommendations = {
            'morning': [],  # 5-12
            'afternoon': [],  # 12-17
            'evening': [],  # 17-22
            'night': []  # 22-5
        }
        
        # Determine time of day
        if 5 <= current_hour < 12:
            time_slot = 'morning'
        elif 12 <= current_hour < 17:
            time_slot = 'afternoon'
        elif 17 <= current_hour < 22:
            time_slot = 'evening'
        else:
            time_slot = 'night'
        
        # Weekend vs weekday
        is_weekend = current_day >= 5
        
        # Get user's viewing patterns
        user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        time_patterns = defaultdict(list)
        
        for interaction in user_interactions:
            hour = interaction.created_at.hour
            content = Content.query.get(interaction.content_id)
            if content:
                if 5 <= hour < 12:
                    time_patterns['morning'].append(content)
                elif 12 <= hour < 17:
                    time_patterns['afternoon'].append(content)
                elif 17 <= hour < 22:
                    time_patterns['evening'].append(content)
                else:
                    time_patterns['night'].append(content)
        
        # Generate recommendations based on patterns
        if time_patterns[time_slot]:
            # Find similar content to what user watches at this time
            for content in time_patterns[time_slot][-5:]:  # Last 5 items
                idx = Content.query.filter(Content.id <= content.id).count() - 1
                if idx < len(self.content_similarity):
                    similar_indices = np.argsort(self.content_similarity[idx])[::-1][1:6]
                    for sim_idx in similar_indices:
                        sim_content = Content.query.offset(sim_idx).first()
                        if sim_content and sim_content not in recommendations[time_slot]:
                            recommendations[time_slot].append(sim_content)
        
        # Default recommendations for different times
        if not recommendations[time_slot]:
            if time_slot == 'morning':
                # Light content for morning
                recommendations['morning'] = Content.query.filter(
                    Content.genres.contains('Comedy') | Content.genres.contains('Animation')
                ).order_by(Content.popularity.desc()).limit(10).all()
            elif time_slot == 'evening':
                # Drama/thriller for evening
                recommendations['evening'] = Content.query.filter(
                    Content.genres.contains('Drama') | Content.genres.contains('Thriller')
                ).order_by(Content.popularity.desc()).limit(10).all()
            elif time_slot == 'night' and is_weekend:
                # Longer content for weekend nights
                recommendations['night'] = Content.query.filter(
                    Content.runtime > 120
                ).order_by(Content.popularity.desc()).limit(10).all()
        
        return recommendations

# Initialize services
aggregator = ContentAggregator()
recommender = RecommendationEngine()
telegram_service = TelegramService()

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
    """Serialize content object with full details"""
    return {
        'id': content.id,
        'tmdb_id': content.tmdb_id,
        'imdb_id': content.imdb_id,
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
        'popularity': content.popularity,
        'trailers': content.trailers,
        'cast': content.cast[:10] if content.cast else [],  # Top 10 cast
        'crew': content.crew[:5] if content.crew else []  # Top 5 crew
    }

def get_user_location(ip_address):
    """Get user location from IP address"""
    try:
        response = requests.get(f'http://ip-api.com/json/{ip_address}')
        data = response.json()
        if data['status'] == 'success':
            return {
                'country': data.get('country'),
                'region': data.get('regionName'),
                'city': data.get('city'),
                'country_code': data.get('countryCode')
            }
    except:
        pass
    return None

# Authentication routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or not email or not password:
            return jsonify({'error': 'All fields are required'}), 400
        
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        # Get user location from IP
        location_data = get_user_location(request.remote_addr)
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            location=location_data.get('country_code') if location_data else None
        )
        db.session.add(user)
        db.session.commit()
        
        token = create_access_token(identity=user.id)
        return jsonify({
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        user = User.query.filter(
            (User.username == username) | (User.email == username)
        ).first()
        
        if user and check_password_hash(user.password_hash, password):
            token = create_access_token(identity=user.id)
            return jsonify({
                'token': token,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'is_admin': user.is_admin
                }
            })
        
        return jsonify({'error': 'Invalid credentials'}), 401
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Homepage routes
@app.route('/api/homepage')
def homepage():
    """Get homepage recommendations for non-logged users"""
    # Get session ID for anonymous tracking
    session_id = request.headers.get('X-Session-ID') or request.remote_addr
    
    # Get user's search history
    search_history = SearchHistory.query.filter_by(
        session_id=session_id
    ).order_by(SearchHistory.created_at.desc()).limit(10).all()
    
    # Get location-based recommendations
    location_data = get_user_location(request.remote_addr)
    
    # Language mapping for regional content
    region_language_map = {
        'IN': ['hi', 'te', 'ta', 'kn'],  # India
        'JP': ['ja'],  # Japan
        'KR': ['ko'],  # Korea
        'FR': ['fr'],  # France
        'ES': ['es'],  # Spain
        'DE': ['de']   # Germany
    }
    
    # Get regional languages
    regional_languages = []
    if location_data:
        country_code = location_data.get('country_code', 'US')
        regional_languages = region_language_map.get(country_code, ['en'])
    
    # Fetch content
    trending_movies = async_to_sync(aggregator.fetch_trending)('movie')
    trending_tv = async_to_sync(aggregator.fetch_trending)('tv')
    anime_trending = async_to_sync(aggregator.fetch_anime_trending)()
    
    # Genre-based recommendations
    genres = {
        'Action': 28, 'Comedy': 35, 'Drama': 18, 'Horror': 27,
        'Sci-Fi': 878, 'Romance': 10749, 'Thriller': 53, 'Animation': 16
    }
    
    popular_by_genre = {}
    for genre_name, genre_id in genres.items():
        popular_by_genre[genre_name] = async_to_sync(aggregator.fetch_popular_by_genre)(genre_id)[:10]
    
    # Regional content
    regional_content = {}
    for lang in regional_languages[:3]:  # Limit to 3 languages
        regional_content[lang] = async_to_sync(aggregator.fetch_regional_content)(lang)[:10]
    
    # Search-based recommendations
    search_recommendations = []
    if search_history:
        # Get content related to recent searches
        for search in search_history[:3]:
            results = Content.query.filter(
                Content.title.ilike(f'%{search.search_query}%')
            ).limit(5).all()
            search_recommendations.extend(results)
    
    # Admin curated content
    featured = AdminRecommendation.query.filter(
        AdminRecommendation.category == 'featured',
        (AdminRecommendation.expires_at.is_(None)) | (AdminRecommendation.expires_at > datetime.utcnow())
    ).order_by(AdminRecommendation.priority.desc()).limit(10).all()
    
    featured_content = []
    for rec in featured:
        content = Content.query.get(rec.content_id)
        if content:
            serialized = serialize_content(content)
            serialized['admin_description'] = rec.description
            featured_content.append(serialized)
    
    # Popular content
    popular_content = Content.query.order_by(
        Content.popularity.desc()
    ).limit(20).all()
    
    # Recently added
    recent_content = Content.query.order_by(
        Content.updated_at.desc()
    ).limit(20).all()
    
    return jsonify({
        'trending': {
            'movies': trending_movies[:15],
            'tv': trending_tv[:15],
            'anime': anime_trending[:15]
        },
        'popular_by_genre': popular_by_genre,
        'regional': regional_content,
        'featured': featured_content,
        'popular': [serialize_content(c) for c in popular_content],
        'recent': [serialize_content(c) for c in recent_content],
        'search_based': [serialize_content(c) for c in search_recommendations[:10]],
        'location': location_data
    })

# Personalized recommendations
@app.route('/api/recommendations')
@jwt_required()
def get_recommendations():
    """Get personalized recommendations for logged-in users"""
    user_id = get_jwt_identity()
    
    # Update recommendation matrix if needed
    if not recommender.content_similarity is not None:
        recommender.build_content_matrix()
    
    # Get different types of recommendations
    hybrid_recs = recommender.get_hybrid_recommendations(user_id, 30)
    time_based_recs = recommender.get_time_based_recommendations(user_id)
    genre_based_recs = recommender.get_genre_based_recommendations(user_id, 15)
    
    # Get user's viewing history for "continue watching"
    recent_views = UserInteraction.query.filter_by(
        user_id=user_id,
        interaction_type='view'
    ).filter(
        UserInteraction.watch_time < 80  # Assuming 80% completion
    ).order_by(
        UserInteraction.created_at.desc()
    ).limit(10).all()
    
    continue_watching = []
    for view in recent_views:
        content = Content.query.get(view.content_id)
        if content:
            serialized = serialize_content(content)
            serialized['progress'] = view.watch_time
            continue_watching.append(serialized)
    
    # Get recommendations based on watchlist
    watchlist = UserInteraction.query.filter_by(
        user_id=user_id,
        interaction_type='wishlist'
    ).all()
    
    watchlist_recommendations = []
    for item in watchlist[:5]:
        content = Content.query.get(item.content_id)
        if content:
            idx = Content.query.filter(Content.id <= content.id).count() - 1
            if idx < len(recommender.content_similarity):
                similar_indices = np.argsort(recommender.content_similarity[idx])[::-1][1:4]
                for sim_idx in similar_indices:
                    sim_content = Content.query.offset(sim_idx).first()
                    if sim_content and sim_content not in watchlist_recommendations:
                        watchlist_recommendations.append(sim_content)
    
    # Trending in user's preferred genres
    user = User.query.get(user_id)
    genre_trending = []
    if user.preferences and 'favorite_genres' in user.preferences:
        for genre in user.preferences['favorite_genres'][:3]:
            trending = Content.query.filter(
                Content.genres.contains(genre)
            ).order_by(
                Content.popularity.desc()
            ).limit(5).all()
            genre_trending.extend(trending)
    
    return jsonify({
        'personalized': [serialize_content(c) for c in hybrid_recs],
        'continue_watching': continue_watching,
        'time_based': {
            'morning': [serialize_content(c) for c in time_based_recs.get('morning', [])[:10]],
            'evening': [serialize_content(c) for c in time_based_recs.get('evening', [])[:10]],
            'night': [serialize_content(c) for c in time_based_recs.get('night', [])[:10]]
        },
        'genre_picks': [serialize_content(c) for c in genre_based_recs],
        'watchlist_based': [serialize_content(c) for c in watchlist_recommendations[:10]],
        'trending_for_you': [serialize_content(c) for c in genre_trending[:15]]
    })

# Content details
@app.route('/api/content/<int:content_id>')
def get_content_details(content_id):
    """Get detailed content information"""
    content = Content.query.get(content_id)
    
    if not content:
        return jsonify({'error': 'Content not found'}), 404
    
    # Get full details from external API if needed
    if content.tmdb_id and (not content.cast or not content.trailers):
        details = async_to_sync(aggregator.get_content_details)(
            content.tmdb_id, content.content_type
        )
        
        # Update content with new details
        if details:
            content.cast = details.get('credits', {}).get('cast', [])
            content.crew = details.get('credits', {}).get('crew', [])
            content.trailers = details.get('trailers', [])
            content.keywords = details.get('keywords', {}).get('keywords', [])
            content.meta_data = {
                'budget': details.get('budget'),
                'revenue': details.get('revenue'),
                'production_companies': details.get('production_companies', []),
                'spoken_languages': details.get('spoken_languages', [])
            }
            db.session.commit()
    
    # Get user reviews
    reviews = db.session.query(
        UserReview, User
    ).join(User).filter(
        UserReview.content_id == content_id
    ).order_by(
        UserReview.created_at.desc()
    ).limit(20).all()
    
    review_data = []
    for review, user in reviews:
        review_data.append({
            'id': review.id,
            'user': {
                'id': user.id,
                'username': user.username
            },
            'rating': review.rating,
            'review': review.review_text,
            'likes': review.likes,
            'created_at': review.created_at.isoformat()
        })
    
    # Get similar content
    similar_content = []
    if recommender.content_similarity is not None:
        idx = Content.query.filter(Content.id <= content.id).count() - 1
        if idx < len(recommender.content_similarity):
            similar_indices = np.argsort(recommender.content_similarity[idx])[::-1][1:11]
            for sim_idx in similar_indices:
                sim_content = Content.query.offset(sim_idx).first()
                if sim_content:
                    similar_content.append(serialize_content(sim_content))
    
    # Average rating
    avg_rating = db.session.query(
        db.func.avg(UserReview.rating)
    ).filter(
        UserReview.content_id == content_id
    ).scalar() or 0
    
    return jsonify({
        'content': serialize_content(content),
        'reviews': review_data,
        'similar': similar_content,
        'average_rating': round(avg_rating, 1),
        'total_reviews': len(review_data),
        'meta_data': content.meta_data
    })

# User interactions
@app.route('/api/interact', methods=['POST'])
@jwt_required()
def user_interact():
    """Record user interaction"""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    content_id = data.get('content_id')
    interaction_type = data.get('interaction_type')
    rating = data.get('rating')
    watch_time = data.get('watch_time')
    
    # Validate
    if not content_id or not interaction_type:
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Check if interaction exists
    existing = UserInteraction.query.filter_by(
        user_id=user_id,
        content_id=content_id,
        interaction_type=interaction_type
    ).first()
    
    if existing:
        # Update existing interaction
        if rating:
            existing.rating = rating
        if watch_time:
            existing.watch_time = watch_time
        existing.created_at = datetime.utcnow()
    else:
        # Create new interaction
        interaction = UserInteraction(
            user_id=user_id,
            content_id=content_id,
            interaction_type=interaction_type,
            rating=rating,
            watch_time=watch_time
        )
        db.session.add(interaction)
    
    # Update user preferences
    content = Content.query.get(content_id)
    if content:
        user = User.query.get(user_id)
        preferences = user.preferences or {}
        
        # Update genre weights
        if content.genres:
            genre_weights = preferences.get('genre_weights', {})
            for genre in content.genres:
                genre_name = genre.get('name') if isinstance(genre, dict) else genre
                current_weight = genre_weights.get(genre_name, 0)
                
                # Calculate weight change based on interaction
                weight_change = 0
                if interaction_type == 'favorite':
                    weight_change = 2.0
                elif interaction_type == 'like':
                    weight_change = 1.5
                elif interaction_type == 'rating' and rating:
                    weight_change = rating / 5.0
                elif interaction_type == 'view':
                    weight_change = 0.5
                
                genre_weights[genre_name] = current_weight + weight_change
            
            preferences['genre_weights'] = genre_weights
        
        # Update favorite genres
        if 'favorite_genres' not in preferences:
            sorted_genres = sorted(
                preferences.get('genre_weights', {}).items(),
                key=lambda x: x[1],
                reverse=True
            )
            preferences['favorite_genres'] = [g[0] for g in sorted_genres[:5]]
        
        user.preferences = preferences
    
    db.session.commit()
    
    # Send to ML service for learning
    try:
        requests.post(
            f"{ML_SERVICE_URL}/learn",
            json={
                'user_id': user_id,
                'content_id': content_id,
                'interaction_type': interaction_type,
                'rating': rating
            },
            timeout=2
        )
    except:
        pass
    
    return jsonify({'status': 'success'})

# Search
@app.route('/api/search')
def search_content():
    """Multi-source search with caching"""
    query = request.args.get('q', '')
    page = int(request.args.get('page', 1))
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    # Cache key
    cache_key = f"search_{hashlib.md5(query.encode()).hexdigest()}_{page}"
    cached = cache.get(cache_key)
    if cached:
        return jsonify(cached)
    
    # Record search history
    session_id = request.headers.get('X-Session-ID') or request.remote_addr
    user_id = None
    
    # Check if user is authenticated
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            token = auth_header.split(' ')[1]
            decoded = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            user_id = decoded['sub']
        except:
            pass
    
    search_history = SearchHistory(
        user_id=user_id,
        session_id=session_id,
        search_query=query
    )
    db.session.add(search_history)
    
    # Search database
    db_results = Content.query.filter(
        Content.title.ilike(f'%{query}%') |
        Content.original_title.ilike(f'%{query}%') |
        Content.overview.ilike(f'%{query}%')
    ).limit(20).all()
    
    # Search external sources
    external_results = async_to_sync(aggregator.search_multi_source)(query, page)
    
    # Combine and deduplicate results
    all_results = {
        'database': [serialize_content(c) for c in db_results],
        'tmdb': external_results.get('tmdb', []),
        'anime': external_results.get('anime', [])
    }
    
    # Update search history with results count
    search_history.results_count = (
        len(all_results['database']) +
        len(all_results['tmdb']) +
        len(all_results['anime'])
    )
    db.session.commit()
    
    # Cache results
    cache.set(cache_key, all_results, 3600)
    
    return jsonify(all_results)

# User profile
@app.route('/api/user/profile')
@jwt_required()
def get_user_profile():
    """Get user profile with stats"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Get user stats
    total_watched = UserInteraction.query.filter_by(
        user_id=user_id,
        interaction_type='view'
    ).count()
    
    total_favorites = UserInteraction.query.filter_by(
        user_id=user_id,
        interaction_type='favorite'
    ).count()
    
    total_ratings = UserInteraction.query.filter_by(
        user_id=user_id
    ).filter(UserInteraction.rating.isnot(None)).count()
    
    # Get favorite genres
    favorite_genres = user.preferences.get('favorite_genres', []) if user.preferences else []
    
    # Get recent activity
    recent_activity = UserInteraction.query.filter_by(
        user_id=user_id
    ).order_by(
        UserInteraction.created_at.desc()
    ).limit(10).all()
    
    activity_data = []
    for activity in recent_activity:
        content = Content.query.get(activity.content_id)
        if content:
            activity_data.append({
                'content': serialize_content(content),
                'interaction_type': activity.interaction_type,
                'rating': activity.rating,
                'created_at': activity.created_at.isoformat()
            })
    
    return jsonify({
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'created_at': user.created_at.isoformat(),
            'is_admin': user.is_admin
        },
        'stats': {
            'total_watched': total_watched,
            'total_favorites': total_favorites,
            'total_ratings': total_ratings
        },
        'preferences': {
            'favorite_genres': favorite_genres,
            'location': user.location
        },
        'recent_activity': activity_data
    })

@app.route('/api/user/watchlist')
@jwt_required()
def get_user_watchlist():
    """Get user's watchlist"""
    user_id = get_jwt_identity()
    
    watchlist = db.session.query(
        UserInteraction, Content
    ).join(Content).filter(
        UserInteraction.user_id == user_id,
        UserInteraction.interaction_type == 'wishlist'
    ).order_by(
        UserInteraction.created_at.desc()
    ).all()
    
    return jsonify({
        'watchlist': [
            {
                'content': serialize_content(content),
                'added_at': interaction.created_at.isoformat()
            }
            for interaction, content in watchlist
        ]
    })

@app.route('/api/user/favorites')
@jwt_required()
def get_user_favorites():
    """Get user's favorite content"""
    user_id = get_jwt_identity()
    
    favorites = db.session.query(
        UserInteraction, Content
    ).join(Content).filter(
        UserInteraction.user_id == user_id,
        UserInteraction.interaction_type == 'favorite'
    ).order_by(
        UserInteraction.created_at.desc()
    ).all()
    
    return jsonify({
        'favorites': [
            {
                'content': serialize_content(content),
                'added_at': interaction.created_at.isoformat()
            }
            for interaction, content in favorites
        ]
    })

# Reviews
@app.route('/api/content/<int:content_id>/review', methods=['POST'])
@jwt_required()
def add_review():
    """Add or update user review"""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    review_text = data.get('review')
    rating = data.get('rating')
    
    if not rating:
        return jsonify({'error': 'Rating required'}), 400
    
    # Check if review exists
    existing = UserReview.query.filter_by(
        user_id=user_id,
        content_id=content_id
    ).first()
    
    if existing:
        existing.review_text = review_text
        existing.rating = rating
        existing.created_at = datetime.utcnow()
    else:
        review = UserReview(
            user_id=user_id,
            content_id=content_id,
            review_text=review_text,
            rating=rating
        )
        db.session.add(review)
    
    # Also update interaction
    interaction = UserInteraction.query.filter_by(
        user_id=user_id,
        content_id=content_id,
        interaction_type='rating'
    ).first()
    
    if interaction:
        interaction.rating = rating
    else:
        interaction = UserInteraction(
            user_id=user_id,
            content_id=content_id,
            interaction_type='rating',
            rating=rating
        )
        db.session.add(interaction)
    
    db.session.commit()
    
    return jsonify({'status': 'success'})

# Admin routes
@app.route('/api/admin/content/search')
@jwt_required()
def admin_search_content():
    """Admin content search across all sources"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    query = request.args.get('q', '')
    source = request.args.get('source', 'all')  # all, tmdb, omdb, anime
    
    results = {
        'tmdb': [],
        'omdb': [],
        'anime': []
    }
    
    if source in ['all', 'tmdb']:
        # Search TMDB
        tmdb_response = requests.get(
            f"{aggregator.tmdb_base}/search/multi",
            params={
                'api_key': TMDB_API_KEY,
                'query': query
            }
        )
        if tmdb_response.status_code == 200:
            results['tmdb'] = tmdb_response.json().get('results', [])
    
    if source in ['all', 'omdb']:
        # Search OMDb
        omdb_response = requests.get(
            aggregator.omdb_base,
            params={
                'apikey': OMDB_API_KEY,
                's': query
            }
        )
        if omdb_response.status_code == 200:
            data = omdb_response.json()
            if data.get('Response') == 'True':
                results['omdb'] = data.get('Search', [])
    
    if source in ['all', 'anime']:
        # Search anime
        results['anime'] = async_to_sync(aggregator.search_multi_source)(query)['anime']
    
    return jsonify(results)

@app.route('/api/admin/content/add', methods=['POST'])
@jwt_required()
def admin_add_content():
    """Add content to database from external source"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    data = request.get_json()
    source = data.get('source')
    external_id = data.get('external_id')
    content_type = data.get('content_type', 'movie')
    
    if source == 'tmdb':
        # Fetch from TMDB
        details = async_to_sync(aggregator.get_content_details)(external_id, content_type)
        
        if not details:
            return jsonify({'error': 'Content not found'}), 404
        
        # Check if already exists
        existing = Content.query.filter_by(tmdb_id=str(external_id)).first()
        if existing:
            return jsonify({'error': 'Content already exists', 'content_id': existing.id}), 400
        
        # Create content
        content = Content(
            tmdb_id=str(details['id']),
            imdb_id=details.get('imdb_id'),
            title=details.get('title') or details.get('name'),
            original_title=details.get('original_title') or details.get('original_name'),
            overview=details.get('overview'),
            genres=[{'id': g['id'], 'name': g['name']} for g in details.get('genres', [])],
            language=details.get('original_language'),
            release_date=datetime.strptime(details['release_date'], '%Y-%m-%d').date()
                    if details.get('release_date') else None,
            runtime=details.get('runtime'),
            rating=details.get('vote_average'),
            poster_path=details.get('poster_path'),
            backdrop_path=details.get('backdrop_path'),
            content_type=content_type,
            popularity=details.get('popularity', 0),
            cast=details.get('credits', {}).get('cast', []),
            crew=details.get('credits', {}).get('crew', []),
            trailers=details.get('trailers', [])
        )
        
        db.session.add(content)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'content': serialize_content(content)
        })
    
    return jsonify({'error': 'Unsupported source'}), 400

@app.route('/api/admin/recommendations/create', methods=['POST'])
@jwt_required()
def create_admin_recommendation():
    """Enhanced admin recommendation creation with Telegram option"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    data = request.get_json()
    
    # Create recommendation
    recommendation = AdminRecommendation(
        content_id=data['content_id'],
        category=data['category'],
        priority=data.get('priority', 1),
        description=data.get('description'),
        expires_at=datetime.strptime(data['expires_at'], '%Y-%m-%d')
                  if data.get('expires_at') else None,
        created_by=user_id
    )
    
    db.session.add(recommendation)
    db.session.commit()
    
    # Send to Telegram if requested
    if data.get('send_to_telegram'):
        content = Content.query.get(data['content_id'])
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            channels = data.get('telegram_channels', [telegram_service.admin_channel])
            
            for channel in channels:
                result = loop.run_until_complete(
                    telegram_service.send_recommendation(content, recommendation, channel)
                )
                
                if result['success']:
                    post = TelegramPost(
                        admin_recommendation_id=recommendation.id,
                        channel_id=channel,
                        message_id=result['message_id'],
                        post_type='instant',
                        posted_at=datetime.utcnow(),
                        status='sent'
                    )
                    db.session.add(post)
            
            db.session.commit()
            
        finally:
            loop.close()
    
    return jsonify({
        'status': 'success',
        'recommendation_id': recommendation.id,
        'telegram_sent': data.get('send_to_telegram', False)
    })

@app.route('/api/admin/telegram/channels', methods=['GET', 'POST'])
@jwt_required()
def manage_telegram_channels():
    """Manage Telegram channels"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    if request.method == 'POST':
        data = request.get_json()
        
        channel = TelegramChannel(
            channel_id=data['channel_id'],
            channel_name=data.get('channel_name'),
            channel_type=data.get('channel_type', 'channel')
        )
        
        db.session.add(channel)
        db.session.commit()
        
        return jsonify({'status': 'success', 'channel_id': channel.id})
    
    # GET - List channels
    channels = TelegramChannel.query.filter_by(is_active=True).all()
    
    return jsonify({
        'channels': [
            {
                'id': ch.id,
                'channel_id': ch.channel_id,
                'channel_name': ch.channel_name,
                'channel_type': ch.channel_type,
                'created_at': ch.created_at.isoformat()
            }
            for ch in channels
        ]
    })

@app.route('/api/admin/telegram/send', methods=['POST'])
@jwt_required()
def send_telegram_recommendation():
    """Send recommendation to Telegram"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    data = request.get_json()
    recommendation_id = data.get('recommendation_id')
    channel_id = data.get('channel_id')
    schedule_time = data.get('schedule_time')
    
    # Get recommendation and content
    recommendation = AdminRecommendation.query.get(recommendation_id)
    if not recommendation:
        return jsonify({'error': 'Recommendation not found'}), 404
    
    content = Content.query.get(recommendation.content_id)
    if not content:
        return jsonify({'error': 'Content not found'}), 404
    
    # Schedule or send immediately
    if schedule_time:
        scheduled_dt = datetime.strptime(schedule_time, '%Y-%m-%d %H:%M:%S')
        post = telegram_service.schedule_post(content, recommendation, channel_id, scheduled_dt)
        
        return jsonify({
            'status': 'scheduled',
            'post_id': post.id,
            'scheduled_time': scheduled_dt.isoformat()
        })
    else:
        # Send immediately
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                telegram_service.send_recommendation(content, recommendation, channel_id)
            )
            
            if result['success']:
                # Record the post
                post = TelegramPost(
                    admin_recommendation_id=recommendation_id,
                    channel_id=channel_id or telegram_service.admin_channel,
                    message_id=result['message_id'],
                    post_type='instant',
                    posted_at=datetime.utcnow(),
                    status='sent'
                )
                db.session.add(post)
                db.session.commit()
                
                return jsonify({
                    'status': 'sent',
                    'message_id': result['message_id'],
                    'channel': result['channel']
                })
            else:
                return jsonify({
                    'status': 'failed',
                    'error': result['error']
                }), 500
                
        finally:
            loop.close()

@app.route('/api/admin/telegram/batch', methods=['POST'])
@jwt_required()
def send_telegram_batch():
    """Send multiple recommendations as a batch"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    data = request.get_json()
    recommendation_ids = data.get('recommendation_ids', [])
    post_type = data.get('post_type', 'album')  # album or separate
    
    recommendations = []
    for rec_id in recommendation_ids:
        rec = AdminRecommendation.query.get(rec_id)
        if rec:
            content = Content.query.get(rec.content_id)
            if content:
                recommendations.append((content, rec))
    
    if not recommendations:
        return jsonify({'error': 'No valid recommendations found'}), 404
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        if post_type == 'album':
            # Send as media group
            result = loop.run_until_complete(
                telegram_service.send_batch_recommendations(recommendations)
            )
        else:
            # Send separately
            results = []
            for content, rec in recommendations:
                result = loop.run_until_complete(
                    telegram_service.send_recommendation(content, rec)
                )
                results.append(result)
                
                # Add delay between posts
                time.sleep(2)
            
            return jsonify({
                'status': 'sent',
                'results': results
            })
        
        return jsonify(result)
        
    finally:
        loop.close()

@app.route('/api/admin/telegram/templates', methods=['GET', 'POST'])
@jwt_required()
def telegram_templates():
    """Manage Telegram post templates"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    if request.method == 'POST':
        data = request.get_json()
        
        # Save template to user preferences
        templates = user.preferences.get('telegram_templates', [])
        templates.append({
            'name': data['name'],
            'format': data['format'],
            'created_at': datetime.utcnow().isoformat()
        })
        
        user.preferences['telegram_templates'] = templates
        db.session.commit()
        
        return jsonify({'status': 'success'})
    
    # GET templates
    templates = user.preferences.get('telegram_templates', [])
    
    # Default templates
    default_templates = [
        {
            'name': 'Weekend Special',
            'format': '🎉 WEEKEND SPECIAL 🎉\n\n{title}\n\n{description}\n\n#WeekendWatch #Movies'
        },
        {
            'name': 'Critics Choice',
            'format': '🏆 CRITICS\' CHOICE 🏆\n\n{title}\n⭐ {rating}/10\n\n{description}\n\n#CriticsChoice'
        },
        {
            'name': 'Hidden Gem',
            'format': '💎 HIDDEN GEM ALERT 💎\n\n{title}\n\n{description}\n\nDon\'t miss this underrated masterpiece!\n\n#HiddenGem'
        }
    ]
    
    return jsonify({
        'templates': templates,
        'default_templates': default_templates
    })

@app.route('/api/admin/telegram/analytics')
@jwt_required()
def telegram_analytics():
    """Get Telegram post analytics"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    # Get post statistics
    total_posts = TelegramPost.query.count()
    successful_posts = TelegramPost.query.filter_by(status='sent').count()
    failed_posts = TelegramPost.query.filter_by(status='failed').count()
    scheduled_posts = TelegramPost.query.filter_by(status='pending').count()
    
    # Recent posts
    recent_posts = db.session.query(
        TelegramPost, AdminRecommendation, Content
    ).join(
        AdminRecommendation
    ).join(
        Content, AdminRecommendation.content_id == Content.id
    ).order_by(
        TelegramPost.created_at.desc()
    ).limit(20).all()
    
    # Posts by category
    category_stats = db.session.query(
        AdminRecommendation.category,
        db.func.count(TelegramPost.id)
    ).join(
        TelegramPost
    ).group_by(
        AdminRecommendation.category
    ).all()
    
    return jsonify({
        'stats': {
            'total_posts': total_posts,
            'successful_posts': successful_posts,
            'failed_posts': failed_posts,
            'scheduled_posts': scheduled_posts
        },
        'recent_posts': [
            {
                'id': post.id,
                'content_title': content.title,
                'category': rec.category,
                'status': post.status,
                'posted_at': post.posted_at.isoformat() if post.posted_at else None,
                'channel': post.channel_id
            }
            for post, rec, content in recent_posts
        ],
        'category_distribution': dict(category_stats)
    })

@app.route('/api/admin/dashboard')
@jwt_required()
def admin_dashboard():
    """Admin dashboard with comprehensive stats"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    # Basic stats
    total_users = User.query.count()
    total_content = Content.query.count()
    total_interactions = UserInteraction.query.count()
    total_reviews = UserReview.query.count()
    
    # Recent activity
    recent_users = User.query.order_by(
        User.created_at.desc()
    ).limit(10).all()
    
    recent_interactions = db.session.query(
        UserInteraction, User, Content
    ).join(User).join(Content).order_by(
        UserInteraction.created_at.desc()
    ).limit(20).all()
    
    # Popular content
    popular_content = db.session.query(
        Content,
        db.func.count(UserInteraction.id).label('interaction_count')
    ).join(UserInteraction).group_by(
        Content.id
    ).order_by(
        db.func.count(UserInteraction.id).desc()
    ).limit(10).all()
    
    # Genre distribution
    genre_stats = {}
    all_content = Content.query.all()
    for content in all_content:
        if content.genres:
            for genre in content.genres:
                genre_name = genre.get('name') if isinstance(genre, dict) else genre
                genre_stats[genre_name] = genre_stats.get(genre_name, 0) + 1
    
    return jsonify({
        'stats': {
            'total_users': total_users,
            'total_content': total_content,
            'total_interactions': total_interactions,
            'total_reviews': total_reviews
        },
        'recent_users': [
            {
                'id': u.id,
                'username': u.username,
                'created_at': u.created_at.isoformat()
            }
            for u in recent_users
        ],
        'recent_activity': [
            {
                'user': user.username,
                'content': content.title,
                'interaction': interaction.interaction_type,
                'created_at': interaction.created_at.isoformat()
            }
            for interaction, user, content in recent_interactions
        ],
        'popular_content': [
            {
                'content': serialize_content(content),
                'interactions': count
            }
            for content, count in popular_content
        ],
        'genre_distribution': genre_stats
    })

# Content sync
@app.route('/api/sync/content', methods=['POST'])
@jwt_required()
def sync_content():
    """Sync content from external APIs (admin only)"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    sync_type = request.json.get('type', 'trending')
    
    def sync_task():
        with app.app_context():
            synced = 0
            
            if sync_type in ['trending', 'all']:
                # Sync trending movies and TV shows
                trending_movies = async_to_sync(aggregator.fetch_trending)('movie')
                trending_tv = async_to_sync(aggregator.fetch_trending)('tv')
                
                for item in trending_movies + trending_tv:
                    existing = Content.query.filter_by(tmdb_id=str(item['id'])).first()
                    if not existing:
                        content = Content(
                            tmdb_id=str(item['id']),
                            title=item.get('title') or item.get('name'),
                            original_title=item.get('original_title') or item.get('original_name'),
                            overview=item.get('overview'),
                            genres=[{'id': g, 'name': 'Unknown'} for g in item.get('genre_ids', [])],
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
                        synced += 1
            
            if sync_type in ['regional', 'all']:
                # Sync regional content
                languages = ['hi', 'te', 'ta', 'kn']
                for lang in languages:
                    regional = async_to_sync(aggregator.fetch_regional_content)(lang)
                    for item in regional:
                        existing = Content.query.filter_by(tmdb_id=str(item['id'])).first()
                        if not existing:
                            content = Content(
                                tmdb_id=str(item['id']),
                                title=item.get('title'),
                                original_title=item.get('original_title'),
                                overview=item.get('overview'),
                                genres=[{'id': g, 'name': 'Unknown'} for g in item.get('genre_ids', [])],
                                language=lang,
                                release_date=datetime.strptime(item['release_date'], '%Y-%m-%d').date()
                                        if item.get('release_date') else None,
                                rating=item.get('vote_average'),
                                poster_path=item.get('poster_path'),
                                backdrop_path=item.get('backdrop_path'),
                                content_type='movie',
                                popularity=item.get('popularity', 0)
                            )
                            db.session.add(content)
                            synced += 1
            
            db.session.commit()
            
            # Rebuild recommendation matrix
            recommender.build_content_matrix()
            
            print(f"Synced {synced} new content items")
    
    # Run in background
    import threading
    thread = threading.Thread(target=sync_task)
    thread.start()
    
    return jsonify({'status': 'sync_started'})

# Health check
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'database': 'connected' if db.session.is_active else 'disconnected',
        'cache': 'connected' if cache.redis_client else 'memory',
        'telegram': 'configured' if telegram_service.bot else 'not configured'
    })

# Root route
@app.route('/')
def index():
    return jsonify({
        'name': 'Movie Recommendation API',
        'version': '2.0',
        'endpoints': {
            'auth': {
                'register': '/api/register',
                'login': '/api/login'
            },
            'content': {
                'homepage': '/api/homepage',
                'recommendations': '/api/recommendations',
                'details': '/api/content/<id>',
                'search': '/api/search'
            },
            'user': {
                'profile': '/api/user/profile',
                'watchlist': '/api/user/watchlist',
                'favorites': '/api/user/favorites'
            },
            'admin': {
                'dashboard': '/api/admin/dashboard',
                'content_search': '/api/admin/content/search',
                'add_content': '/api/admin/content/add',
                'telegram': {
                    'channels': '/api/admin/telegram/channels',
                    'send': '/api/admin/telegram/send',
                    'batch': '/api/admin/telegram/batch',
                    'analytics': '/api/admin/telegram/analytics'
                }
            }
        }
    })

# Background task for scheduled posts
def process_scheduled_posts():
    """Process scheduled Telegram posts"""
    with app.app_context():
        pending_posts = TelegramPost.query.filter_by(
            status='pending'
        ).filter(
            TelegramPost.scheduled_time <= datetime.utcnow()
        ).all()
        
        for post in pending_posts:
            recommendation = AdminRecommendation.query.get(post.admin_recommendation_id)
            content = Content.query.get(recommendation.content_id)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    telegram_service.send_recommendation(
                        content, recommendation, post.channel_id
                    )
                )
                
                if result['success']:
                    post.status = 'sent'
                    post.posted_at = datetime.utcnow()
                    post.message_id = result['message_id']
                else:
                    post.status = 'failed'
                    post.error_message = result['error']
                
            except Exception as e:
                post.status = 'failed'
                post.error_message = str(e)
            
            finally:
                loop.close()
            
            db.session.commit()

# Initialize database
def init_db():
    with app.app_context():
        db.create_all()
        
        # Create admin user if not exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@movieapp.com',
                password_hash=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
            print("Admin user created - Username: admin, Password: admin123")
        
        # Build initial recommendation matrix
        if Content.query.count() > 0:
            recommender.build_content_matrix()

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)