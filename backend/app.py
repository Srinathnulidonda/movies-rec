import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functools import wraps
from urllib.parse import quote
import hashlib
import pickle
import threading
import time
from collections import defaultdict
import logging
import requests
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor


from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import asyncio
import aiohttp
from apscheduler.schedulers.background import BackgroundScheduler

# Initialize Flask app
app = Flask(__name__)
app.config.update({
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key'),
    'SQLALCHEMY_DATABASE_URI': os.environ.get('DATABASE_URL', 'sqlite:///movies.db').replace('postgres://', 'postgresql://'),
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'JWT_SECRET_KEY': os.environ.get('JWT_SECRET_KEY', 'jwt-secret'),
    'JWT_ACCESS_TOKEN_EXPIRES': timedelta(days=7),
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
limiter.init_app(app)
cache = Cache(app)

# API Keys
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '52260795')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7689567537:AAGvDtu94OlLlTiWpfjSfpl_dd_Osi_2W7c')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '-1002566510721')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-ml-service.onrender.com')
ML_SERVICE_TIMEOUT = 5  # seconds


# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    preferences = db.Column(db.JSON, default=lambda: {'genres': [], 'languages': [], 'min_rating': 0})
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    region = db.Column(db.String(10), default='IN')

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    external_id = db.Column(db.String(50), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    original_title = db.Column(db.String(200))
    type = db.Column(db.String(20), nullable=False)  # movie, tv, anime
    genres = db.Column(db.JSON)
    overview = db.Column(db.Text)
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float)
    poster_path = db.Column(db.String(200))
    backdrop_path = db.Column(db.String(200))
    language = db.Column(db.String(10))
    popularity = db.Column(db.Float, default=0)
    metadata = db.Column(db.JSON)
    source = db.Column(db.String(20))  # tmdb, omdb, jikan, custom
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)  # watched, favorite, wishlist, rated
    value = db.Column(db.Float)  # rating value
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AdminRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # critics_choice, trending, featured
    priority = db.Column(db.Integer, default=1)
    description = db.Column(db.Text)
    expires_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ContentSimilarity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    similar_content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    similarity_score = db.Column(db.Float, nullable=False)
    algorithm = db.Column(db.String(20), nullable=False)

# Content Fetcher Class
class ContentFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'MovieRecommendationSystem/1.0'})
    
    @cache.memoize(timeout=3600)
    def fetch_tmdb_content(self, content_type, page=1, language='en'):
        """Fetch content from TMDB API"""
        urls = {
            'popular_movies': f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&page={page}&language={language}",
            'popular_tv': f"https://api.themoviedb.org/3/tv/popular?api_key={TMDB_API_KEY}&page={page}&language={language}",
            'trending': f"https://api.themoviedb.org/3/trending/all/day?api_key={TMDB_API_KEY}&page={page}",
            'bollywood': f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_original_language=hi&page={page}",
            'regional': f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_original_language=te,ta,kn&page={page}"
        }
        
        try:
            response = self.session.get(urls.get(content_type, urls['popular_movies']))
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logging.error(f"TMDB API error: {e}")
            return None
    
    @cache.memoize(timeout=3600)
    def fetch_anime_content(self, page=1):
        """Fetch anime content from Jikan API"""
        try:
            url = f"https://api.jikan.moe/v4/anime?page={page}&order_by=popularity&sort=desc"
            response = self.session.get(url)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logging.error(f"Jikan API error: {e}")
            return None
    
    def get_content_details(self, content_id, source='tmdb', content_type='movie'):
        """Get detailed content information"""
        if source == 'tmdb':
            url = f"https://api.themoviedb.org/3/{content_type}/{content_id}?api_key={TMDB_API_KEY}&append_to_response=credits,videos"
        elif source == 'omdb':
            url = f"http://www.omdbapi.com/?i={content_id}&apikey={OMDB_API_KEY}&plot=full"
        elif source == 'jikan':
            url = f"https://api.jikan.moe/v4/anime/{content_id}/full"
        
        try:
            response = self.session.get(url)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logging.error(f"Content details error: {e}")
            return None

# ML Recommendation Engine
class RecommendationEngine:
    def __init__(self):
        self.content_features = None
        self.user_item_matrix = None
        self.content_similarity_matrix = None
        self.svd_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
    def update_models(self):
        """Update recommendation models with latest data"""
        try:
            # Get all content and user interactions
            contents = Content.query.all()
            interactions = UserInteraction.query.all()
            
            # Build content features matrix
            content_data = []
            for content in contents:
                features = f"{' '.join(content.genres or [])} {content.overview or ''} {content.language or ''}"
                content_data.append(features)
            
            if content_data:
                self.content_features = self.tfidf_vectorizer.fit_transform(content_data)
                self.content_similarity_matrix = cosine_similarity(self.content_features)
            
            # Build user-item interaction matrix
            user_ids = list(set([i.user_id for i in interactions]))
            content_ids = list(set([i.content_id for i in interactions]))
            
            if user_ids and content_ids:
                self.user_item_matrix = pd.DataFrame(0, index=user_ids, columns=content_ids)
                
                for interaction in interactions:
                    if interaction.interaction_type == 'rated':
                        self.user_item_matrix.loc[interaction.user_id, interaction.content_id] = interaction.value
                    elif interaction.interaction_type == 'favorite':
                        self.user_item_matrix.loc[interaction.user_id, interaction.content_id] = 5
                    elif interaction.interaction_type == 'watched':
                        self.user_item_matrix.loc[interaction.user_id, interaction.content_id] = 3
                
                # Train SVD model
                self.svd_model = TruncatedSVD(n_components=min(50, len(user_ids)-1))
                self.svd_model.fit(self.user_item_matrix.fillna(0))
                
        except Exception as e:
            logging.error(f"Model update error: {e}")
    
    def get_content_based_recommendations(self, content_id, n_recommendations=10):
        """Get recommendations based on content similarity"""
        if self.content_similarity_matrix is None:
            return []
        
        try:
            content_idx = Content.query.filter_by(id=content_id).first()
            if not content_idx:
                return []
            
            # Get content index in the matrix
            all_contents = Content.query.all()
            content_index = next((i for i, c in enumerate(all_contents) if c.id == content_id), None)
            
            if content_index is None:
                return []
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_similarity_matrix[content_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar content
            similar_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
            return [all_contents[i].id for i in similar_indices if i < len(all_contents)]
            
        except Exception as e:
            logging.error(f"Content-based recommendation error: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations based on user similarity"""
        if self.user_item_matrix is None or self.svd_model is None:
            return []
        
        try:
            if user_id not in self.user_item_matrix.index:
                return []
            
            # Get user's rating vector
            user_ratings = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
            
            # Transform using SVD
            user_vector = self.svd_model.transform(user_ratings)
            
            # Get all users' vectors
            all_users_vectors = self.svd_model.transform(self.user_item_matrix.fillna(0))
            
            # Calculate similarity with all users
            similarities = cosine_similarity(user_vector, all_users_vectors)[0]
            
            # Get most similar users
            similar_users = np.argsort(similarities)[::-1][1:6]  # Top 5 similar users
            
            # Get recommendations based on similar users' preferences
            recommendations = defaultdict(float)
            for similar_user_idx in similar_users:
                similar_user_id = self.user_item_matrix.index[similar_user_idx]
                similar_user_ratings = self.user_item_matrix.loc[similar_user_id]
                
                for content_id, rating in similar_user_ratings.items():
                    if rating > 0 and self.user_item_matrix.loc[user_id, content_id] == 0:
                        recommendations[content_id] += rating * similarities[similar_user_idx]
            
            # Sort and return top recommendations
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            return [content_id for content_id, _ in sorted_recs[:n_recommendations]]
            
        except Exception as e:
            logging.error(f"Collaborative recommendation error: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=10):
        """Get hybrid recommendations combining multiple approaches"""
        content_recs = []
        collab_recs = []
        
        # Get user's recent interactions
        recent_interactions = UserInteraction.query.filter_by(user_id=user_id).order_by(UserInteraction.timestamp.desc()).limit(5).all()
        
        # Get content-based recommendations from recent interactions
        for interaction in recent_interactions:
            content_recs.extend(self.get_content_based_recommendations(interaction.content_id, 5))
        
        # Get collaborative recommendations
        collab_recs = self.get_collaborative_recommendations(user_id, 10)
        
        # Combine and weight recommendations
        combined_recs = defaultdict(float)
        for content_id in content_recs:
            combined_recs[content_id] += 0.7  # Content-based weight
        
        for content_id in collab_recs:
            combined_recs[content_id] += 0.3  # Collaborative weight
        
        # Sort by combined score
        sorted_recs = sorted(combined_recs.items(), key=lambda x: x[1], reverse=True)
        return [content_id for content_id, _ in sorted_recs[:n_recommendations]]

# Initialize instances
content_fetcher = ContentFetcher()
recommendation_engine = RecommendationEngine()

# Background Tasks
def sync_content_data():
    """Sync content data from various sources"""
    try:
        # Fetch from TMDB
        for content_type in ['popular_movies', 'popular_tv', 'trending', 'bollywood', 'regional']:
            data = content_fetcher.fetch_tmdb_content(content_type)
            if data and 'results' in data:
                for item in data['results']:
                    existing = Content.query.filter_by(external_id=str(item['id']), source='tmdb').first()
                    if not existing:
                        content = Content(
                            external_id=str(item['id']),
                            title=item.get('title', item.get('name', '')),
                            original_title=item.get('original_title', item.get('original_name', '')),
                            type='movie' if 'title' in item else 'tv',
                            genres=item.get('genre_ids', []),
                            overview=item.get('overview', ''),
                            release_date=datetime.strptime(item.get('release_date', item.get('first_air_date', '1900-01-01')), '%Y-%m-%d').date() if item.get('release_date') or item.get('first_air_date') else None,
                            rating=item.get('vote_average', 0),
                            poster_path=item.get('poster_path', ''),
                            backdrop_path=item.get('backdrop_path', ''),
                            language=item.get('original_language', 'en'),
                            popularity=item.get('popularity', 0),
                            metadata=item,
                            source='tmdb'
                        )
                        db.session.add(content)
        
        # Fetch anime content
        anime_data = content_fetcher.fetch_anime_content()
        if anime_data and 'data' in anime_data:
            for item in anime_data['data']:
                existing = Content.query.filter_by(external_id=str(item['mal_id']), source='jikan').first()
                if not existing:
                    content = Content(
                        external_id=str(item['mal_id']),
                        title=item.get('title', ''),
                        original_title=item.get('title_japanese', ''),
                        type='anime',
                        genres=[g['name'] for g in item.get('genres', [])],
                        overview=item.get('synopsis', ''),
                        release_date=datetime.strptime(item.get('aired', {}).get('from', '1900-01-01T00:00:00'), '%Y-%m-%dT%H:%M:%S').date() if item.get('aired', {}).get('from') else None,
                        rating=item.get('score', 0),
                        poster_path=item.get('images', {}).get('jpg', {}).get('image_url', ''),
                        language='ja',
                        popularity=item.get('popularity', 0),
                        metadata=item,
                        source='jikan'
                    )
                    db.session.add(content)
        
        db.session.commit()
        logging.info("Content sync completed")
        
    except Exception as e:
        logging.error(f"Content sync error: {e}")
        db.session.rollback()

def update_similarity_matrix():
    """Update content similarity matrix"""
    try:
        recommendation_engine.update_models()
        
        # Store similarities in database
        ContentSimilarity.query.delete()
        
        if recommendation_engine.content_similarity_matrix is not None:
            contents = Content.query.all()
            for i, content in enumerate(contents):
                similarities = recommendation_engine.content_similarity_matrix[i]
                top_similar = np.argsort(similarities)[::-1][1:21]  # Top 20 similar
                
                for j in top_similar:
                    if j < len(contents) and similarities[j] > 0.1:
                        similarity = ContentSimilarity(
                            content_id=content.id,
                            similar_content_id=contents[j].id,
                            similarity_score=similarities[j],
                            algorithm='tfidf_cosine'
                        )
                        db.session.add(similarity)
        
        db.session.commit()
        logging.info("Similarity matrix updated")
        
    except Exception as e:
        logging.error(f"Similarity update error: {e}")
        db.session.rollback()

def send_telegram_notification(message):
    """Send notification to Telegram"""
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
            requests.post(url, json=data)
        except Exception as e:
            logging.error(f"Telegram notification error: {e}")

# API Routes
@app.route('/api/auth/register', methods=['POST'])
@limiter.limit("5 per minute")
def register():
    data = request.get_json()
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"error": "Username already exists"}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "Email already exists"}), 400
    
    user = User(
        username=data['username'],
        email=data['email'],
        password_hash=generate_password_hash(data['password']),
        preferences=data.get('preferences', {}),
        region=data.get('region', 'IN')
    )
    
    db.session.add(user)
    db.session.commit()
    
    token = create_access_token(identity=user.id)
    return jsonify({"token": token, "user": {"id": user.id, "username": user.username}})

@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    
    if user and check_password_hash(user.password_hash, data['password']):
        token = create_access_token(identity=user.id)
        return jsonify({"token": token, "user": {"id": user.id, "username": user.username}})
    
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/recommendations/homepage')
@cache.cached(timeout=600)
def homepage_recommendations():
    """Get homepage recommendations for non-logged users"""
    try:
        recommendations = {}
        
        # Trending content
        trending = Content.query.filter_by(source='tmdb').order_by(Content.popularity.desc()).limit(20).all()
        recommendations['trending'] = [serialize_content(c) for c in trending]
        
        # Popular by genre
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance']
        for genre in genres:
            genre_content = Content.query.filter(Content.genres.contains([genre])).order_by(Content.rating.desc()).limit(10).all()
            recommendations[f'popular_{genre.lower()}'] = [serialize_content(c) for c in genre_content]
        
        # Regional content
        regional_langs = ['hi', 'te', 'ta', 'kn']
        for lang in regional_langs:
            regional_content = Content.query.filter_by(language=lang).order_by(Content.popularity.desc()).limit(10).all()
            recommendations[f'regional_{lang}'] = [serialize_content(c) for c in regional_content]
        
        # Critics' choice (admin curated)
        critics_choice = db.session.query(Content).join(AdminRecommendation).filter(AdminRecommendation.category == 'critics_choice').order_by(AdminRecommendation.priority.desc()).limit(10).all()
        recommendations['critics_choice'] = [serialize_content(c) for c in critics_choice]
        
        # Anime content
        anime_content = Content.query.filter_by(type='anime').order_by(Content.rating.desc()).limit(10).all()
        recommendations['anime'] = [serialize_content(c) for c in anime_content]
        
        return jsonify(recommendations)
        
    except Exception as e:
        logging.error(f"Homepage recommendations error: {e}")
        return jsonify({"error": "Failed to fetch recommendations"}), 500

@app.route('/api/recommendations/personalized')
@jwt_required()
def personalized_recommendations():
    """Get personalized recommendations using ML service"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        recommendations = {}
        
        # Try ML service first
        if ml_client.is_healthy():
            ml_recommendations = ml_client.get_hybrid_recommendations(user_id, 20)
            if ml_recommendations:
                # Get content details for ML recommendations
                content_ids = [r['content_id'] for r in ml_recommendations]
                ml_contents = Content.query.filter(Content.id.in_(content_ids)).all()
                
                recommendations['ml_powered'] = [serialize_content(c) for c in ml_contents]
        
        # Fallback to existing logic
        if not recommendations.get('ml_powered'):
            # Get hybrid recommendations
            hybrid_recs = recommendation_engine.get_hybrid_recommendations(user_id, 20)
            
            # Get content-based recommendations from user's favorites
            favorite_interactions = UserInteraction.query.filter_by(user_id=user_id, interaction_type='favorite').all()
            content_recs = []
            for interaction in favorite_interactions:
                content_recs.extend(recommendation_engine.get_content_based_recommendations(interaction.content_id, 5))
            
            # Get trending content in user's preferred genres
            user_genres = user.preferences.get('genres', [])
            trending_in_genres = []
            if user_genres:
                trending_in_genres = Content.query.filter(Content.genres.op('&&')(user_genres)).order_by(Content.popularity.desc()).limit(15).all()
            
            # Combine all recommendations
            all_recs = list(set(hybrid_recs + content_recs + [c.id for c in trending_in_genres]))
            
            # Get content details
            recommended_content = Content.query.filter(Content.id.in_(all_recs)).all()
            
            # Organize by categories
            recommendations['for_you'] = [serialize_content(c) for c in recommended_content[:10]]
            recommendations['because_you_liked'] = [serialize_content(c) for c in recommended_content[10:20]]
            recommendations['trending_in_your_genres'] = [serialize_content(c) for c in trending_in_genres]
        
        # Add other sections
        recommendations['continue_watching'] = get_continue_watching(user_id)
        recommendations['your_watchlist'] = get_user_watchlist(user_id)
        
        return jsonify(recommendations)
        
    except Exception as e:
        logging.error(f"Personalized recommendations error: {e}")
        return jsonify({"error": "Failed to fetch recommendations"}), 500
    

@app.route('/api/content/<int:content_id>/similar')
def similar_content(content_id):
    """Get similar content using ML service"""
    try:
        similar_content = []
        
        # Try ML service first
        if ml_client.is_healthy():
            ml_similar = ml_client.get_content_recommendations(content_id, 10)
            if ml_similar:
                content_ids = [r['content_id'] for r in ml_similar]
                similar_content_objects = Content.query.filter(Content.id.in_(content_ids)).all()
                similar_content = [serialize_content(c) for c in similar_content_objects]
        
        # Fallback to existing logic
        if not similar_content:
            similar_db = ContentSimilarity.query.filter_by(content_id=content_id).order_by(ContentSimilarity.similarity_score.desc()).limit(10).all()
            content_ids = [s.similar_content_id for s in similar_db]
            similar_content_objects = Content.query.filter(Content.id.in_(content_ids)).all()
            similar_content = [serialize_content(c) for c in similar_content_objects]
        
        return jsonify({
            'similar_content': similar_content,
            'content_id': content_id
        })
        
    except Exception as e:
        logging.error(f"Similar content error: {e}")
        return jsonify({"error": "Failed to fetch similar content"}), 500
    

@app.route('/api/trending/ml')
@cache.cached(timeout=1800)
def ml_trending():
    """Get ML-powered trending predictions"""
    try:
        trending = []
        
        if ml_client.is_healthy():
            ml_trending = ml_client.get_trending_predictions()
            if ml_trending:
                content_ids = [t['content_id'] for t in ml_trending[:20]]
                trending_content = Content.query.filter(Content.id.in_(content_ids)).all()
                trending = [serialize_content(c) for c in trending_content]
        
        return jsonify({
            'trending': trending,
            'method': 'ml-prediction'
        })
        
    except Exception as e:
        logging.error(f"ML trending error: {e}")
        return jsonify({"error": "Failed to fetch ML trending"}), 500

# Add ML service health check endpoint
@app.route('/api/ml/status')
def ml_service_status():
    """Check ML service status"""
    try:
        is_healthy = ml_client.is_healthy()
        return jsonify({
            'ml_service_healthy': is_healthy,
            'ml_service_url': ML_SERVICE_URL
        })
    except Exception as e:
        return jsonify({
            'ml_service_healthy': False,
            'error': str(e)
        })

def train_ml_models():
    """Train ML models periodically"""
    try:
        if ml_client.is_healthy():
            success = ml_client.train_models()
            if success:
                logging.info("ML models trained successfully")
                send_telegram_notification("ML models updated successfully")
            else:
                logging.error("ML model training failed")
        else:
            logging.error("ML service not healthy")
    except Exception as e:
        logging.error(f"ML training error: {e}")



@app.route('/api/content/<int:content_id>/details')
@cache.cached(timeout=1800)
def content_details(content_id):
    """Get detailed content information"""
    try:
        content = Content.query.get(content_id)
        if not content:
            return jsonify({"error": "Content not found"}), 404
        
        # Get additional details from external API
        external_details = content_fetcher.get_content_details(content.external_id, content.source, content.type)
        
        # Get similar content
        similar_content = ContentSimilarity.query.filter_by(content_id=content_id).order_by(ContentSimilarity.similarity_score.desc()).limit(10).all()
        similar_ids = [s.similar_content_id for s in similar_content]
        similar_content_objects = Content.query.filter(Content.id.in_(similar_ids)).all()
        
        # Get user ratings and reviews
        ratings = UserInteraction.query.filter_by(content_id=content_id, interaction_type='rated').all()
        avg_rating = sum([r.value for r in ratings]) / len(ratings) if ratings else 0
        
        details = {
            **serialize_content(content),
            'external_details': external_details,
            'similar_content': [serialize_content(c) for c in similar_content_objects],
            'user_rating': avg_rating,
            'total_ratings': len(ratings)
        }
        
        return jsonify(details)
        
    except Exception as e:
        logging.error(f"Content details error: {e}")
        return jsonify({"error": "Failed to fetch content details"}), 500

@app.route('/api/search')
@limiter.limit("30 per minute")
def search_content():
    """Search across all content sources"""
    try:
        query = request.args.get('q', '')
        content_type = request.args.get('type', 'all')
        limit = min(int(request.args.get('limit', 20)), 50)
        
        if not query:
            return jsonify({"error": "Search query required"}), 400
        
        # Search in database
        search_filter = Content.title.ilike(f'%{query}%') | Content.overview.ilike(f'%{query}%')
        
        if content_type != 'all':
            search_filter = search_filter & (Content.type == content_type)
        
        results = Content.query.filter(search_filter).order_by(Content.popularity.desc()).limit(limit).all()
        
        # If not enough results, search external APIs
        if len(results) < limit:
            # Search TMDB
            tmdb_search = content_fetcher.session.get(f"https://api.themoviedb.org/3/search/multi?api_key={TMDB_API_KEY}&query={quote(query)}")
            if tmdb_search.status_code == 200:
                tmdb_results = tmdb_search.json().get('results', [])
                for item in tmdb_results[:10]:
                    # Add to database if not exists
                    existing = Content.query.filter_by(external_id=str(item['id']), source='tmdb').first()
                    if not existing:
                        content = Content(
                            external_id=str(item['id']),
                            title=item.get('title', item.get('name', '')),
                            type='movie' if 'title' in item else 'tv',
                            genres=item.get('genre_ids', []),
                            overview=item.get('overview', ''),
                            rating=item.get('vote_average', 0),
                            poster_path=item.get('poster_path', ''),
                            language=item.get('original_language', 'en'),
                            popularity=item.get('popularity', 0),
                            metadata=item,
                            source='tmdb'
                        )
                        db.session.add(content)
                        results.append(content)
        
        db.session.commit()
        
        return jsonify({
            'results': [serialize_content(c) for c in results],
            'total': len(results),
            'query': query
        })
        
    except Exception as e:
        logging.error(f"Search error: {e}")
        return jsonify({"error": "Search failed"}), 500

@app.route('/api/user/interact', methods=['POST'])
@jwt_required()
def user_interact():
    """Handle user interactions (rating, favorite, watchlist)"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        content_id = data.get('content_id')
        interaction_type = data.get('type')  # rated, favorite, wishlist, watched
        value = data.get('value', 1)
        
        if not content_id or not interaction_type:
            return jsonify({"error": "Missing required fields"}), 400
        
        # Check if content exists
        content = Content.query.get(content_id)
        if not content:
            return jsonify({"error": "Content not found"}), 404
        
        # Remove existing interaction of same type
        existing = UserInteraction.query.filter_by(
            user_id=user_id, 
            content_id=content_id, 
            interaction_type=interaction_type
        ).first()
        
        if existing:
            if interaction_type == 'rated':
                existing.value = value
            else:
                db.session.delete(existing)
                db.session.commit()
                return jsonify({"message": "Interaction removed"})
        else:
            interaction = UserInteraction(
                user_id=user_id,
                content_id=content_id,
                interaction_type=interaction_type,
                value=value if interaction_type == 'rated' else None
            )
            db.session.add(interaction)
        
        db.session.commit()
        return jsonify({"message": "Interaction recorded"})
        
    except Exception as e:
        logging.error(f"User interaction error: {e}")
        return jsonify({"error": "Failed to record interaction"}), 500

@app.route('/api/user/profile')
@jwt_required()
def user_profile():
    """Get user profile and stats"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Get user stats
        interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        
        stats = {
            'total_watched': len([i for i in interactions if i.interaction_type == 'watched']),
            'total_favorites': len([i for i in interactions if i.interaction_type == 'favorite']),
            'total_ratings': len([i for i in interactions if i.interaction_type == 'rated']),
            'avg_rating': sum([i.value for i in interactions if i.interaction_type == 'rated' and i.value]) / len([i for i in interactions if i.interaction_type == 'rated']) if [i for i in interactions if i.interaction_type == 'rated'] else 0
        }
        
        # Get favorite genres
        favorite_content = [i.content_id for i in interactions if i.interaction_type == 'favorite']
        favorite_genres = []
        if favorite_content:
            contents = Content.query.filter(Content.id.in_(favorite_content)).all()
            genre_count = defaultdict(int)
            for content in contents:
                for genre in content.genres or []:
                    genre_count[genre] += 1
            favorite_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)[:5]
        
        profile = {
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'preferences': user.preferences,
                'region': user.region,
                'created_at': user.created_at.isoformat()
            },
            'stats': stats,
            'favorite_genres': favorite_genres
        }
        
        return jsonify(profile)
        
    except Exception as e:
        logging.error(f"User profile error: {e}")
        return jsonify({"error": "Failed to fetch profile"}), 500

@app.route('/api/trending')
@cache.cached(timeout=3600)
def trending_content():
    """Get trending content by region and type"""
    try:
        region = request.args.get('region', 'global')
        content_type = request.args.get('type', 'all')
        limit = min(int(request.args.get('limit', 20)), 50)
        
        query = Content.query
        
        if region != 'global':
            query = query.filter_by(language=region)
        
        if content_type != 'all':
            query = query.filter_by(type=content_type)
        
        # Get trending based on recent interactions and popularity
        trending = query.order_by(Content.popularity.desc()).limit(limit).all()
        
        return jsonify({
            'trending': [serialize_content(c) for c in trending],
            'region': region,
            'type': content_type
        })
        
    except Exception as e:
        logging.error(f"Trending content error: {e}")
        return jsonify({"error": "Failed to fetch trending content"}), 500

@app.route('/api/admin/curate', methods=['POST'])
@jwt_required()
def admin_curate():
    """Admin endpoint to curate content"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user or not user.is_admin:
            return jsonify({"error": "Admin access required"}), 403
        
        data = request.get_json()
        content_id = data.get('content_id')
        category = data.get('category')  # critics_choice, featured, trending
        priority = data.get('priority', 1)
        description = data.get('description', '')
        expires_at = data.get('expires_at')
        
        if not content_id or not category:
            return jsonify({"error": "Missing required fields"}), 400
        
        # Check if content exists
        content = Content.query.get(content_id)
        if not content:
            return jsonify({"error": "Content not found"}), 404
        
        # Remove existing curation
        existing = AdminRecommendation.query.filter_by(content_id=content_id, category=category).first()
        if existing:
            db.session.delete(existing)
        
        # Add new curation
        recommendation = AdminRecommendation(
            content_id=content_id,
            category=category,
            priority=priority,
            description=description,
            expires_at=datetime.strptime(expires_at, '%Y-%m-%d') if expires_at else None
        )
        
        db.session.add(recommendation)
        db.session.commit()
        
        # Send Telegram notification
        send_telegram_notification(f"New admin curation: {content.title} added to {category}")
        
        return jsonify({"message": "Content curated successfully"})
        
    except Exception as e:
        logging.error(f"Admin curation error: {e}")
        return jsonify({"error": "Failed to curate content"}), 500

@app.route('/api/admin/browse')
@jwt_required()
def admin_browse():
    """Admin endpoint to browse content for curation"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user or not user.is_admin:
            return jsonify({"error": "Admin access required"}), 403
        
        source = request.args.get('source', 'tmdb')
        page = int(request.args.get('page', 1))
        content_type = request.args.get('type', 'popular_movies')
        
        # Fetch content from external sources
        if source == 'tmdb':
            data = content_fetcher.fetch_tmdb_content(content_type, page)
        elif source == 'jikan':
            data = content_fetcher.fetch_anime_content(page)
        else:
            return jsonify({"error": "Invalid source"}), 400
        
        if not data:
            return jsonify({"error": "Failed to fetch content"}), 500
        
        # Format response
        results = []
        if source == 'tmdb' and 'results' in data:
            for item in data['results']:
                results.append({
                    'external_id': item['id'],
                    'title': item.get('title', item.get('name', '')),
                    'overview': item.get('overview', ''),
                    'rating': item.get('vote_average', 0),
                    'poster_path': f"https://image.tmdb.org/t/p/w500{item.get('poster_path', '')}" if item.get('poster_path') else '',
                    'release_date': item.get('release_date', item.get('first_air_date', '')),
                    'source': 'tmdb'
                })
        elif source == 'jikan' and 'data' in data:
            for item in data['data']:
                results.append({
                    'external_id': item['mal_id'],
                    'title': item.get('title', ''),
                    'overview': item.get('synopsis', ''),
                    'rating': item.get('score', 0),
                    'poster_path': item.get('images', {}).get('jpg', {}).get('image_url', ''),
                    'release_date': item.get('aired', {}).get('from', ''),
                    'source': 'jikan'
                })
        
        return jsonify({
            'results': results,
            'page': page,
            'source': source,
            'type': content_type
        })
        
    except Exception as e:
        logging.error(f"Admin browse error: {e}")
        return jsonify({"error": "Failed to browse content"}), 500

@app.route('/api/genres')
@cache.cached(timeout=86400)
def get_genres():
    """Get all available genres"""
    try:
        # Get genres from TMDB
        tmdb_genres = content_fetcher.session.get(f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}")
        tv_genres = content_fetcher.session.get(f"https://api.themoviedb.org/3/genre/tv/list?api_key={TMDB_API_KEY}")
        
        all_genres = []
        if tmdb_genres.status_code == 200:
            all_genres.extend(tmdb_genres.json().get('genres', []))
        if tv_genres.status_code == 200:
            all_genres.extend(tv_genres.json().get('genres', []))
        
        # Remove duplicates
        unique_genres = {g['id']: g['name'] for g in all_genres}
        
        # Add anime genres
        anime_genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Slice of Life', 'Supernatural', 'Thriller']
        for genre in anime_genres:
            unique_genres[genre] = genre
        
        return jsonify(list(unique_genres.values()))
        
    except Exception as e:
        logging.error(f"Genres error: {e}")
        return jsonify({"error": "Failed to fetch genres"}), 500

# Helper Functions
def serialize_content(content):
    """Serialize content object to dictionary"""
    return {
        'id': content.id,
        'external_id': content.external_id,
        'title': content.title,
        'original_title': content.original_title,
        'type': content.type,
        'genres': content.genres,
        'overview': content.overview,
        'release_date': content.release_date.isoformat() if content.release_date else None,
        'runtime': content.runtime,
        'rating': content.rating,
        'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path and content.source == 'tmdb' else content.poster_path,
        'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path and content.source == 'tmdb' else content.backdrop_path,
        'language': content.language,
        'popularity': content.popularity,
        'source': content.source,
        'created_at': content.created_at.isoformat()
    }

def get_continue_watching(user_id):
    """Get user's continue watching list"""
    watched = UserInteraction.query.filter_by(user_id=user_id, interaction_type='watched').order_by(UserInteraction.timestamp.desc()).limit(10).all()
    content_ids = [w.content_id for w in watched]
    contents = Content.query.filter(Content.id.in_(content_ids)).all()
    return [serialize_content(c) for c in contents]

def get_user_watchlist(user_id):
    """Get user's watchlist"""
    wishlist = UserInteraction.query.filter_by(user_id=user_id, interaction_type='wishlist').order_by(UserInteraction.timestamp.desc()).limit(20).all()
    content_ids = [w.content_id for w in wishlist]
    contents = Content.query.filter(Content.id.in_(content_ids)).all()
    return [serialize_content(c) for c in contents]

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded"}), 429

# Initialize Database
@app.before_first_request
def create_tables():
    db.create_all()
    
    # Create admin user if doesn't exist
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

# Background Scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=sync_content_data,
    trigger="interval",
    hours=6,
    id='sync_content',
    name='Sync content data from external APIs',
    replace_existing=True
)
scheduler.add_job(
    func=train_ml_models,
    trigger="interval",
    hours=8,
    id='train_ml_models',
    name='Train ML recommendation models',
    replace_existing=True
)
scheduler.start()


class MLServiceClient:
    def __init__(self, base_url=ML_SERVICE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = ML_SERVICE_TIMEOUT
    
    def is_healthy(self):
        """Check if ML service is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def get_content_recommendations(self, content_id, limit=10):
        """Get content-based recommendations"""
        try:
            response = self.session.get(f"{self.base_url}/recommendations/content/{content_id}?limit={limit}")
            return response.json().get('recommendations', []) if response.status_code == 200 else []
        except:
            return []
    
    def get_user_recommendations(self, user_id, limit=10):
        """Get collaborative filtering recommendations"""
        try:
            response = self.session.get(f"{self.base_url}/recommendations/user/{user_id}?limit={limit}")
            return response.json().get('recommendations', []) if response.status_code == 200 else []
        except:
            return []
    
    def get_hybrid_recommendations(self, user_id, limit=20):
        """Get hybrid recommendations"""
        try:
            response = self.session.get(f"{self.base_url}/recommendations/hybrid/{user_id}?limit={limit}")
            return response.json().get('recommendations', []) if response.status_code == 200 else []
        except:
            return []
    
    def get_trending_predictions(self):
        """Get ML-based trending predictions"""
        try:
            response = self.session.get(f"{self.base_url}/trending")
            return response.json().get('trending', []) if response.status_code == 200 else []
        except:
            return []
    
    def train_models(self):
        """Trigger ML model training"""
        try:
            response = self.session.post(f"{self.base_url}/train")
            return response.status_code == 200
        except:
            return False

# Initialize ML service client (add this after your existing initializations)
ml_client = MLServiceClient()


@app.route('/api/ml/content-data')
def ml_content_data():
    """Provide content data to ML service"""
    try:
        contents = Content.query.all()
        content_data = []
        
        for content in contents:
            content_data.append({
                'id': content.id,
                'title': content.title,
                'genres': content.genres or [],
                'overview': content.overview or '',
                'language': content.language or 'en',
                'rating': content.rating or 0,
                'popularity': content.popularity or 0,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'type': content.type
            })
        
        return jsonify(content_data)
        
    except Exception as e:
        logging.error(f"ML content data error: {e}")
        return jsonify([]), 500

@app.route('/api/ml/user-interactions')
def ml_user_interactions():
    """Provide user interaction data to ML service"""
    try:
        interactions = UserInteraction.query.all()
        interaction_data = []
        
        for interaction in interactions:
            interaction_data.append({
                'user_id': interaction.user_id,
                'content_id': interaction.content_id,
                'interaction_type': interaction.interaction_type,
                'value': interaction.value,
                'timestamp': interaction.timestamp.isoformat()
            })
        
        return jsonify(interaction_data)
        
    except Exception as e:
        logging.error(f"ML interaction data error: {e}")
        return jsonify([]), 500



# Health Check
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})

# Production Configuration
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    # Initialize background tasks
    sync_content_data()
    update_similarity_matrix()
    
    app.run(host='0.0.0.0', port=port, debug=debug)