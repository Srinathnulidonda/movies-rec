# requirements.txt
Flask==2.3.3
Flask-SQLAlchemy==3.0.5
Flask-CORS==4.0.0
Flask-JWT-Extended==4.5.2
psycopg2-binary==2.9.7
requests==2.31.0
python-dotenv==1.0.0
redis==4.6.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
gunicorn==21.2.0
APScheduler==3.10.4

# app.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import os
import requests
import redis
import json
from datetime import datetime, timedelta
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import logging

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/moviedb')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)

db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app)

# Redis connection (optional)
try:
    redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
    redis_client.ping()
except:
    redis_client = None

# API Keys
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
OMDB_API_KEY = os.getenv('OMDB_API_KEY')

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    preferred_genres = db.Column(db.JSON, default=[])
    preferred_languages = db.Column(db.JSON, default=['en'])
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    title = db.Column(db.String(200), nullable=False)
    overview = db.Column(db.Text)
    genres = db.Column(db.JSON, default=[])
    languages = db.Column(db.JSON, default=[])
    release_date = db.Column(db.Date)
    rating = db.Column(db.Float, default=0.0)
    poster_path = db.Column(db.String(200))
    backdrop_path = db.Column(db.String(200))
    content_type = db.Column(db.String(20))  # movie, tv, anime
    region = db.Column(db.String(10))  # telugu, hindi, tamil, kannada
    popularity = db.Column(db.Float, default=0.0)
    cast = db.Column(db.JSON, default=[])
    crew = db.Column(db.JSON, default=[])
    runtime = db.Column(db.Integer)
    trailer_url = db.Column(db.String(200))
    is_trending = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20))  # watch, favorite, wishlist, rating
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AdminRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(50))  # critics_choice, user_favorites, trending
    priority = db.Column(db.Integer, default=1)
    expires_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Content Fetching Service
class ContentService:
    @staticmethod
    def fetch_tmdb_content(page=1, region='US'):
        try:
            # Fetch movies
            movies_url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&page={page}&region={region}"
            movies_response = requests.get(movies_url).json()
            
            # Fetch TV shows
            tv_url = f"https://api.themoviedb.org/3/tv/popular?api_key={TMDB_API_KEY}&page={page}"
            tv_response = requests.get(tv_url).json()
            
            content_list = []
            
            # Process movies
            for movie in movies_response.get('results', []):
                content = Content(
                    tmdb_id=movie['id'],
                    title=movie['title'],
                    overview=movie.get('overview', ''),
                    genres=movie.get('genre_ids', []),
                    release_date=datetime.strptime(movie.get('release_date', '1970-01-01'), '%Y-%m-%d').date() if movie.get('release_date') else None,
                    rating=movie.get('vote_average', 0.0),
                    poster_path=movie.get('poster_path', ''),
                    backdrop_path=movie.get('backdrop_path', ''),
                    content_type='movie',
                    popularity=movie.get('popularity', 0.0)
                )
                content_list.append(content)
            
            # Process TV shows
            for show in tv_response.get('results', []):
                content = Content(
                    tmdb_id=show['id'],
                    title=show['name'],
                    overview=show.get('overview', ''),
                    genres=show.get('genre_ids', []),
                    release_date=datetime.strptime(show.get('first_air_date', '1970-01-01'), '%Y-%m-%d').date() if show.get('first_air_date') else None,
                    rating=show.get('vote_average', 0.0),
                    poster_path=show.get('poster_path', ''),
                    backdrop_path=show.get('backdrop_path', ''),
                    content_type='tv',
                    popularity=show.get('popularity', 0.0)
                )
                content_list.append(content)
            
            return content_list
        except Exception as e:
            logging.error(f"Error fetching TMDB content: {e}")
            return []

    @staticmethod
    def fetch_regional_content(language='hi'):
        # Fetch regional content using TMDB with language filters
        try:
            url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_original_language={language}&sort_by=popularity.desc"
            response = requests.get(url).json()
            
            content_list = []
            for movie in response.get('results', []):
                content = Content(
                    tmdb_id=movie['id'],
                    title=movie['title'],
                    overview=movie.get('overview', ''),
                    genres=movie.get('genre_ids', []),
                    release_date=datetime.strptime(movie.get('release_date', '1970-01-01'), '%Y-%m-%d').date() if movie.get('release_date') else None,
                    rating=movie.get('vote_average', 0.0),
                    poster_path=movie.get('poster_path', ''),
                    backdrop_path=movie.get('backdrop_path', ''),
                    content_type='movie',
                    region=language,
                    popularity=movie.get('popularity', 0.0)
                )
                content_list.append(content)
            
            return content_list
        except Exception as e:
            logging.error(f"Error fetching regional content: {e}")
            return []

    @staticmethod
    def fetch_content_details(tmdb_id, content_type='movie'):
        try:
            url = f"https://api.themoviedb.org/3/{content_type}/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits,videos"
            response = requests.get(url).json()
            
            # Get cast and crew
            cast = [{'name': person['name'], 'character': person.get('character', ''), 'profile_path': person.get('profile_path', '')} for person in response.get('credits', {}).get('cast', [])[:10]]
            crew = [{'name': person['name'], 'job': person.get('job', ''), 'profile_path': person.get('profile_path', '')} for person in response.get('credits', {}).get('crew', [])[:10]]
            
            # Get trailer
            trailer_url = ''
            for video in response.get('videos', {}).get('results', []):
                if video.get('type') == 'Trailer' and video.get('site') == 'YouTube':
                    trailer_url = f"https://www.youtube.com/watch?v={video['key']}"
                    break
            
            return {
                'cast': cast,
                'crew': crew,
                'trailer_url': trailer_url,
                'runtime': response.get('runtime', 0)
            }
        except Exception as e:
            logging.error(f"Error fetching content details: {e}")
            return {'cast': [], 'crew': [], 'trailer_url': '', 'runtime': 0}

# Recommendation Engine
class RecommendationEngine:
    @staticmethod
    def get_homepage_recommendations():
        cache_key = 'homepage_recommendations'
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Trending content
        trending = Content.query.filter_by(is_trending=True).order_by(Content.popularity.desc()).limit(20).all()
        
        # Popular by genre
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance']
        popular_by_genre = {}
        for genre in genres:
            popular_by_genre[genre] = Content.query.filter(Content.genres.contains([genre])).order_by(Content.popularity.desc()).limit(10).all()
        
        # Regional trending
        regional_trending = {}
        for region in ['hi', 'te', 'ta', 'kn']:
            regional_trending[region] = Content.query.filter_by(region=region).order_by(Content.popularity.desc()).limit(10).all()
        
        # Critics choice and user favorites
        critics_choice = AdminRecommendation.query.filter_by(category='critics_choice').order_by(AdminRecommendation.priority.desc()).limit(10).all()
        user_favorites = AdminRecommendation.query.filter_by(category='user_favorites').order_by(AdminRecommendation.priority.desc()).limit(10).all()
        
        recommendations = {
            'trending': [{'id': c.id, 'title': c.title, 'poster_path': c.poster_path, 'rating': c.rating} for c in trending],
            'popular_by_genre': {genre: [{'id': c.id, 'title': c.title, 'poster_path': c.poster_path, 'rating': c.rating} for c in content] for genre, content in popular_by_genre.items()},
            'regional_trending': {region: [{'id': c.id, 'title': c.title, 'poster_path': c.poster_path, 'rating': c.rating} for c in content] for region, content in regional_trending.items()},
            'critics_choice': [{'id': r.content_id, 'title': r.title, 'description': r.description} for r in critics_choice],
            'user_favorites': [{'id': r.content_id, 'title': r.title, 'description': r.description} for r in user_favorites]
        }
        
        if redis_client:
            redis_client.setex(cache_key, 3600, json.dumps(recommendations))
        
        return recommendations

    @staticmethod
    def get_personalized_recommendations(user_id):
        user = User.query.get(user_id)
        if not user:
            return []
        
        # Get user interactions
        interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        
        # Content-based filtering
        content_based = RecommendationEngine.content_based_recommendations(user_id)
        
        # Collaborative filtering
        collaborative = RecommendationEngine.collaborative_filtering(user_id)
        
        # Combine recommendations
        recommendations = list(set(content_based + collaborative))
        
        # Sort by popularity and rating
        content_objects = Content.query.filter(Content.id.in_(recommendations)).order_by(Content.popularity.desc(), Content.rating.desc()).limit(50).all()
        
        return [{'id': c.id, 'title': c.title, 'poster_path': c.poster_path, 'rating': c.rating, 'genres': c.genres} for c in content_objects]

    @staticmethod
    def content_based_recommendations(user_id):
        # Get user's watched/liked content
        user_interactions = UserInteraction.query.filter_by(user_id=user_id).filter(UserInteraction.interaction_type.in_(['watch', 'favorite', 'rating'])).all()
        
        if not user_interactions:
            return []
        
        # Get content IDs user has interacted with
        interacted_content_ids = [i.content_id for i in user_interactions]
        
        # Get all content
        all_content = Content.query.all()
        
        # Create feature vectors based on genres and overview
        content_features = []
        content_ids = []
        
        for content in all_content:
            features = ' '.join(content.genres) + ' ' + (content.overview or '')
            content_features.append(features)
            content_ids.append(content.id)
        
        # Calculate similarity
        if len(content_features) > 1:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(content_features)
            
            recommendations = []
            for content_id in interacted_content_ids:
                if content_id in content_ids:
                    idx = content_ids.index(content_id)
                    sim_scores = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
                    similar_indices = sim_scores.argsort()[-20:][::-1]
                    
                    for i in similar_indices:
                        if content_ids[i] not in interacted_content_ids:
                            recommendations.append(content_ids[i])
            
            return list(set(recommendations))
        
        return []

    @staticmethod
    def collaborative_filtering(user_id):
        # Get users with similar preferences
        user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        user_content_ids = set([i.content_id for i in user_interactions])
        
        # Find similar users
        all_users = User.query.filter(User.id != user_id).all()
        similar_users = []
        
        for user in all_users:
            other_interactions = UserInteraction.query.filter_by(user_id=user.id).all()
            other_content_ids = set([i.content_id for i in other_interactions])
            
            # Calculate Jaccard similarity
            intersection = len(user_content_ids.intersection(other_content_ids))
            union = len(user_content_ids.union(other_content_ids))
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0.1:  # Threshold for similarity
                    similar_users.append((user.id, similarity))
        
        # Get recommendations from similar users
        recommendations = []
        for similar_user_id, similarity in sorted(similar_users, key=lambda x: x[1], reverse=True)[:10]:
            similar_user_interactions = UserInteraction.query.filter_by(user_id=similar_user_id).filter(UserInteraction.interaction_type.in_(['favorite', 'rating'])).all()
            
            for interaction in similar_user_interactions:
                if interaction.content_id not in user_content_ids:
                    recommendations.append(interaction.content_id)
        
        return list(set(recommendations))

# API Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 400
    
    user = User(
        username=data['username'],
        email=data['email'],
        password_hash=generate_password_hash(data['password']),
        preferred_genres=data.get('preferred_genres', []),
        preferred_languages=data.get('preferred_languages', ['en'])
    )
    
    db.session.add(user)
    db.session.commit()
    
    access_token = create_access_token(identity=user.id)
    return jsonify({'access_token': access_token, 'user': {'id': user.id, 'username': user.username, 'email': user.email}})

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data['email']).first()
    
    if user and check_password_hash(user.password_hash, data['password']):
        access_token = create_access_token(identity=user.id)
        return jsonify({'access_token': access_token, 'user': {'id': user.id, 'username': user.username, 'email': user.email}})
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/recommendations/homepage', methods=['GET'])
def homepage_recommendations():
    recommendations = RecommendationEngine.get_homepage_recommendations()
    return jsonify(recommendations)

@app.route('/api/recommendations/personalized', methods=['GET'])
@jwt_required()
def personalized_recommendations():
    user_id = get_jwt_identity()
    recommendations = RecommendationEngine.get_personalized_recommendations(user_id)
    return jsonify(recommendations)

@app.route('/api/content/<int:content_id>/details', methods=['GET'])
def content_details(content_id):
    content = Content.query.get_or_404(content_id)
    
    # Fetch additional details if not cached
    if not content.cast or not content.crew:
        details = ContentService.fetch_content_details(content.tmdb_id, content.content_type)
        content.cast = details['cast']
        content.crew = details['crew']
        content.trailer_url = details['trailer_url']
        content.runtime = details['runtime']
        db.session.commit()
    
    # Get similar content
    similar_content = Content.query.filter(Content.id != content_id).filter(Content.genres.op('&&')(content.genres)).order_by(Content.popularity.desc()).limit(10).all()
    
    return jsonify({
        'id': content.id,
        'title': content.title,
        'overview': content.overview,
        'genres': content.genres,
        'rating': content.rating,
        'poster_path': content.poster_path,
        'backdrop_path': content.backdrop_path,
        'cast': content.cast,
        'crew': content.crew,
        'trailer_url': content.trailer_url,
        'runtime': content.runtime,
        'release_date': content.release_date.isoformat() if content.release_date else None,
        'similar_content': [{'id': c.id, 'title': c.title, 'poster_path': c.poster_path} for c in similar_content]
    })

@app.route('/api/search', methods=['GET'])
def search_content():
    query = request.args.get('q', '')
    content_type = request.args.get('type', 'all')
    
    search_query = Content.query.filter(Content.title.ilike(f'%{query}%'))
    
    if content_type != 'all':
        search_query = search_query.filter_by(content_type=content_type)
    
    results = search_query.limit(50).all()
    
    return jsonify([{
        'id': c.id,
        'title': c.title,
        'poster_path': c.poster_path,
        'rating': c.rating,
        'content_type': c.content_type,
        'genres': c.genres
    } for c in results])

@app.route('/api/trending', methods=['GET'])
def trending_content():
    region = request.args.get('region', 'all')
    
    if region == 'all':
        trending = Content.query.filter_by(is_trending=True).order_by(Content.popularity.desc()).limit(20).all()
    else:
        trending = Content.query.filter_by(region=region).order_by(Content.popularity.desc()).limit(20).all()
    
    return jsonify([{
        'id': c.id,
        'title': c.title,
        'poster_path': c.poster_path,
        'rating': c.rating,
        'content_type': c.content_type
    } for c in trending])

@app.route('/api/user/interactions', methods=['POST'])
@jwt_required()
def user_interaction():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    interaction = UserInteraction(
        user_id=user_id,
        content_id=data['content_id'],
        interaction_type=data['interaction_type'],
        rating=data.get('rating')
    )
    
    db.session.add(interaction)
    db.session.commit()
    
    return jsonify({'message': 'Interaction recorded'})

@app.route('/api/user/watchlist', methods=['GET'])
@jwt_required()
def get_watchlist():
    user_id = get_jwt_identity()
    watchlist = db.session.query(Content).join(UserInteraction).filter(UserInteraction.user_id == user_id, UserInteraction.interaction_type == 'wishlist').all()
    
    return jsonify([{
        'id': c.id,
        'title': c.title,
        'poster_path': c.poster_path,
        'rating': c.rating
    } for c in watchlist])

@app.route('/api/admin/curate', methods=['POST'])
@jwt_required()
def admin_curate():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    data = request.get_json()
    
    recommendation = AdminRecommendation(
        content_id=data['content_id'],
        title=data['title'],
        description=data.get('description', ''),
        category=data['category'],
        priority=data.get('priority', 1),
        expires_at=datetime.strptime(data['expires_at'], '%Y-%m-%d') if data.get('expires_at') else None
    )
    
    db.session.add(recommendation)
    db.session.commit()
    
    return jsonify({'message': 'Recommendation added'})

@app.route('/api/admin/content/browse', methods=['GET'])
@jwt_required()
def admin_browse_content():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    page = request.args.get('page', 1, type=int)
    source = request.args.get('source', 'tmdb')
    
    if source == 'tmdb':
        content_list = ContentService.fetch_tmdb_content(page)
    else:
        content_list = ContentService.fetch_regional_content(source)
    
    return jsonify([{
        'tmdb_id': c.tmdb_id,
        'title': c.title,
        'overview': c.overview,
        'poster_path': c.poster_path,
        'rating': c.rating
    } for c in content_list])

# Background Tasks
def update_content_database():
    try:
        # Fetch and update content from various sources
        for page in range(1, 6):  # Fetch 5 pages
            content_list = ContentService.fetch_tmdb_content(page)
            for content in content_list:
                existing = Content.query.filter_by(tmdb_id=content.tmdb_id).first()
                if not existing:
                    db.session.add(content)
        
        # Fetch regional content
        for lang in ['hi', 'te', 'ta', 'kn']:
            content_list = ContentService.fetch_regional_content(lang)
            for content in content_list:
                existing = Content.query.filter_by(tmdb_id=content.tmdb_id).first()
                if not existing:
                    db.session.add(content)
        
        db.session.commit()
        logging.info("Content database updated successfully")
    except Exception as e:
        logging.error(f"Error updating content database: {e}")
        db.session.rollback()

def update_trending_content():
    try:
        # Mark trending content based on popularity
        Content.query.update({Content.is_trending: False})
        trending_content = Content.query.order_by(Content.popularity.desc()).limit(50).all()
        
        for content in trending_content:
            content.is_trending = True
        
        db.session.commit()
        logging.info("Trending content updated successfully")
    except Exception as e:
        logging.error(f"Error updating trending content: {e}")
        db.session.rollback()

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_content_database, trigger="interval", hours=6)
scheduler.add_job(func=update_trending_content, trigger="interval", hours=1)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    # Start scheduler
    scheduler.start()
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_ENV') == 'development')

# Procfile (for Render deployment)
web: gunicorn app:app

# .env template
# DATABASE_URL=postgresql://username:password@hostname:port/database
# JWT_SECRET_KEY=your-super-secret-jwt-key
# TMDB_API_KEY=your-tmdb-api-key
# OMDB_API_KEY=your-omdb-api-key
# REDIS_URL=redis://localhost:6379
# FLASK_ENV=production

# render.yaml (Render deployment configuration)
services:
  - type: web
    name: movie-recommendation-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: movie-db
          property: connectionString
      - key: JWT_SECRET_KEY
        generateValue: true
      - key: TMDB_API_KEY
        value: your-tmdb-api-key
      - key: OMDB_API_KEY
        value: your-omdb-api-key

databases:
  - name: movie-db
    databaseName: moviedb
    user: movieuser