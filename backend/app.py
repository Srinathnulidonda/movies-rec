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




app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_rec.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your-secret-key'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)

db = SQLAlchemy(app)
jwt = JWTManager(app)

# API Keys
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.getenv('OMDB_API_KEY', '52260795')
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'https://ml-service-s2pr.onrender.com')

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

# Content Aggregator Service
class ContentAggregator:
    def __init__(self):
        self.tmdb_base = "https://api.themoviedb.org/3"
        self.omdb_base = "http://www.omdbapi.com"
        self.jikan_base = "https://api.jikan.moe/v4"
        
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
    data = request.get_json()
    
    # Add validation for required fields
    if not data or 'username' not in data or 'email' not in data or 'password' not in data:
        return jsonify({'error': 'Username, email, and password are required'}), 400
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    user = User(
        username=data['username'],
        email=data['email'],
        password_hash=generate_password_hash(data['password'])
    )
    db.session.add(user)
    db.session.commit()
    
    token = create_access_token(identity=user.id)
    return jsonify({'token': token, 'user_id': user.id})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    
    # Add validation for required fields
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Username and password are required'}), 400
    
    user = User.query.filter_by(username=data['username']).first()
    
    if user and check_password_hash(user.password_hash, data['password']):
        token = create_access_token(identity=user.id)
        return jsonify({'token': token, 'user_id': user.id})
    
    return jsonify({'error': 'Invalid credentials'}), 401

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
    """Get detailed content information"""
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

def create_tables():
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
            print("Admin user created - Username: admin, Password: admin123")
        
        # Initial content sync
        if Content.query.count() == 0:
            sync_content()

@app.route('/api/homepage-recommendations')
def homepage_recommendations():
    return jsonify(get_homepage_data())

# 3. Add admin authentication middleware
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

if __name__ == '__main__':
    create_tables() 
    CORS(app)
    app.run(debug=True, port=5000)