from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from datetime import datetime, timedelta
import requests
import os
from functools import wraps
import jwt
import logging
from typing import List, Dict, Optional
import telebot

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'jwt-secret-change-this'

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
CORS(app)

# Configuration
TMDB_API_KEY = '1cf86635f20bb2aff8e70940e7c3ddd5'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
JUSTWATCH_API_URL = 'https://apis.justwatch.com/content'
ML_SERVICE_URL = 'http://localhost:5001'
TELEGRAM_BOT_TOKEN = '7689567537:AAGvDtu94OlLlTiWpfjSfpl_dd_Osi_2W7c'
TELEGRAM_CHANNEL_ID = '-1002566510721'

# Initialize Telegram bot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    watchlist = db.relationship('Watchlist', backref='user', lazy=True, cascade='all, delete-orphan')
    favorites = db.relationship('Favorite', backref='user', lazy=True, cascade='all, delete-orphan')
    watch_history = db.relationship('WatchHistory', backref='user', lazy=True, cascade='all, delete-orphan')

class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, unique=True, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    overview = db.Column(db.Text)
    release_date = db.Column(db.String(20))
    poster_path = db.Column(db.String(200))
    backdrop_path = db.Column(db.String(200))
    genre_ids = db.Column(db.String(100))
    vote_average = db.Column(db.Float)
    vote_count = db.Column(db.Integer)
    popularity = db.Column(db.Float)
    content_type = db.Column(db.String(20))  # movie, tv, anime
    runtime = db.Column(db.Integer)
    status = db.Column(db.String(20))
    tagline = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Watchlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

class WatchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    watched_at = db.Column(db.DateTime, default=datetime.utcnow)
    rating = db.Column(db.Float, nullable=True)

class FeaturedSuggestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    movie = db.relationship('Movie', backref='featured_suggestions')
    creator = db.relationship('User', backref='created_suggestions')

class StreamingPlatform(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    platform_name = db.Column(db.String(100), nullable=False)
    platform_id = db.Column(db.String(50))
    stream_type = db.Column(db.String(20))  # rent, buy, stream, free
    url = db.Column(db.String(500))
    price = db.Column(db.String(20))
    currency = db.Column(db.String(5))
    quality = db.Column(db.String(10))
    
    movie = db.relationship('Movie', backref='streaming_platforms')

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            token = token.split(' ')[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'message': 'User not found'}), 401
        except Exception as e:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(current_user, *args, **kwargs):
        if not current_user.is_admin:
            return jsonify({'message': 'Admin access required'}), 403
        return f(current_user, *args, **kwargs)
    return decorated

# TMDb API Service
class TMDbService:
    @staticmethod
    def search_content(query: str, content_type: str = 'multi') -> Dict:
        """Search for movies, TV shows, or multi content"""
        url = f"{TMDB_BASE_URL}/search/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'language': 'en-US',
            'page': 1
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            app.logger.error(f"TMDb search error: {e}")
            return {'results': []}
    
    @staticmethod
    def get_content_details(content_id: int, content_type: str) -> Dict:
        """Get detailed information about a movie or TV show"""
        url = f"{TMDB_BASE_URL}/{content_type}/{content_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'en-US',
            'append_to_response': 'videos,credits,similar,recommendations,watch/providers'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            app.logger.error(f"TMDb details error: {e}")
            return {}
    
    @staticmethod
    def get_trending(content_type: str = 'all', time_window: str = 'week') -> Dict:
        """Get trending content"""
        url = f"{TMDB_BASE_URL}/trending/{content_type}/{time_window}"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'en-US'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            app.logger.error(f"TMDb trending error: {e}")
            return {'results': []}

# JustWatch Service for Streaming Platforms
class JustWatchService:
    @staticmethod
    def get_streaming_info(tmdb_id: int, content_type: str) -> List[Dict]:
        """Get streaming platform information"""
        # This is a simplified version - JustWatch doesn't have a public API
        # You would need to implement web scraping or use alternative services
        
        # Placeholder implementation
        platforms = [
            {
                'platform_name': 'Netflix',
                'stream_type': 'stream',
                'url': f'https://netflix.com/title/{tmdb_id}',
                'quality': 'HD'
            },
            {
                'platform_name': 'Amazon Prime',
                'stream_type': 'stream',
                'url': f'https://amazon.com/dp/{tmdb_id}',
                'quality': 'HD'
            }
        ]
        return platforms

# ML Service Communication
class MLService:
    @staticmethod
    def get_recommendations(user_id: int, content_preferences: Dict) -> List[int]:
        """Get personalized recommendations from ML microservice"""
        try:
            response = requests.post(
                f"{ML_SERVICE_URL}/recommend",
                json={
                    'user_id': user_id,
                    'preferences': content_preferences
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json().get('recommendations', [])
        except requests.RequestException as e:
            app.logger.error(f"ML service error: {e}")
            return []

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validate input
        if not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'message': 'Missing required fields'}), 400
        
        # Check if user exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'message': 'Email already registered'}), 400
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'message': 'Username already taken'}), 400
        
        # Create new user
        hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=hashed_password
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.config['JWT_SECRET_KEY'])
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            }
        }), 201
        
    except Exception as e:
        app.logger.error(f"Registration error: {e}")
        return jsonify({'message': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data.get('email') or not data.get('password'):
            return jsonify({'message': 'Email and password required'}), 400
        
        user = User.query.filter_by(email=data['email']).first()
        
        if user and bcrypt.check_password_hash(user.password_hash, data['password']):
            token = jwt.encode({
                'user_id': user.id,
                'exp': datetime.utcnow() + timedelta(days=30)
            }, app.config['JWT_SECRET_KEY'])
            
            return jsonify({
                'message': 'Login successful',
                'token': token,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'is_admin': user.is_admin
                }
            })
        else:
            return jsonify({'message': 'Invalid credentials'}), 401
            
    except Exception as e:
        app.logger.error(f"Login error: {e}")
        return jsonify({'message': 'Login failed'}), 500

# Content Routes
@app.route('/api/search', methods=['GET'])
def search_content():
    try:
        query = request.args.get('q', '')
        content_type = request.args.get('type', 'multi')
        
        if not query:
            return jsonify({'message': 'Search query required'}), 400
        
        # Search using TMDb
        results = TMDbService.search_content(query, content_type)
        
        # Process and store results in database
        processed_results = []
        for item in results.get('results', [])[:20]:  # Limit to 20 results
            # Store in database if not exists
            existing_movie = Movie.query.filter_by(tmdb_id=item['id']).first()
            if not existing_movie:
                movie = Movie(
                    tmdb_id=item['id'],
                    title=item.get('title') or item.get('name', ''),
                    overview=item.get('overview', ''),
                    release_date=item.get('release_date') or item.get('first_air_date', ''),
                    poster_path=item.get('poster_path', ''),
                    backdrop_path=item.get('backdrop_path', ''),
                    genre_ids=','.join(map(str, item.get('genre_ids', []))),
                    vote_average=item.get('vote_average', 0),
                    vote_count=item.get('vote_count', 0),
                    popularity=item.get('popularity', 0),
                    content_type=item.get('media_type', content_type)
                )
                db.session.add(movie)
                db.session.commit()
                existing_movie = movie
            
            processed_results.append({
                'id': existing_movie.id,
                'tmdb_id': existing_movie.tmdb_id,
                'title': existing_movie.title,
                'overview': existing_movie.overview,
                'release_date': existing_movie.release_date,
                'poster_path': f"https://image.tmdb.org/t/p/w500{existing_movie.poster_path}" if existing_movie.poster_path else None,
                'backdrop_path': f"https://image.tmdb.org/t/p/w1280{existing_movie.backdrop_path}" if existing_movie.backdrop_path else None,
                'vote_average': existing_movie.vote_average,
                'content_type': existing_movie.content_type
            })
        
        return jsonify({
            'results': processed_results,
            'total_results': len(processed_results)
        })
        
    except Exception as e:
        app.logger.error(f"Search error: {e}")
        return jsonify({'message': 'Search failed'}), 500

@app.route('/api/content/<int:content_id>', methods=['GET'])
def get_content_details(content_id):
    try:
        movie = Movie.query.get_or_404(content_id)
        
        # Get detailed info from TMDb
        content_type = 'movie' if movie.content_type == 'movie' else 'tv'
        details = TMDbService.get_content_details(movie.tmdb_id, content_type)
        
        # Get streaming platforms
        streaming_platforms = StreamingPlatform.query.filter_by(movie_id=movie.id).all()
        
        # If no streaming platforms cached, fetch them
        if not streaming_platforms:
            platform_data = JustWatchService.get_streaming_info(movie.tmdb_id, content_type)
            for platform in platform_data:
                sp = StreamingPlatform(
                    movie_id=movie.id,
                    platform_name=platform['platform_name'],
                    stream_type=platform['stream_type'],
                    url=platform['url'],
                    quality=platform.get('quality', 'HD')
                )
                db.session.add(sp)
            db.session.commit()
            streaming_platforms = StreamingPlatform.query.filter_by(movie_id=movie.id).all()
        
        return jsonify({
            'id': movie.id,
            'tmdb_id': movie.tmdb_id,
            'title': movie.title,
            'overview': movie.overview,
            'release_date': movie.release_date,
            'poster_path': f"https://image.tmdb.org/t/p/w500{movie.poster_path}" if movie.poster_path else None,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{movie.backdrop_path}" if movie.backdrop_path else None,
            'vote_average': movie.vote_average,
            'vote_count': movie.vote_count,
            'content_type': movie.content_type,
            'runtime': details.get('runtime') or details.get('episode_run_time', [0])[0] if details.get('episode_run_time') else None,
            'tagline': details.get('tagline', ''),
            'genres': details.get('genres', []),
            'cast': details.get('credits', {}).get('cast', [])[:10],
            'crew': details.get('credits', {}).get('crew', [])[:5],
            'similar': [
                {
                    'id': item['id'],
                    'title': item.get('title') or item.get('name'),
                    'poster_path': f"https://image.tmdb.org/t/p/w300{item['poster_path']}" if item.get('poster_path') else None
                }
                for item in details.get('similar', {}).get('results', [])[:6]
            ],
            'streaming_platforms': [
                {
                    'platform_name': sp.platform_name,
                    'stream_type': sp.stream_type,
                    'url': sp.url,
                    'quality': sp.quality,
                    'price': sp.price
                }
                for sp in streaming_platforms
            ],
            'videos': details.get('videos', {}).get('results', [])[:3]
        })
        
    except Exception as e:
        app.logger.error(f"Content details error: {e}")
        return jsonify({'message': 'Failed to fetch content details'}), 500

# User Lists Routes
@app.route('/api/watchlist', methods=['GET', 'POST'])
@token_required
def manage_watchlist(current_user):
    if request.method == 'GET':
        watchlist_items = db.session.query(Watchlist, Movie).join(Movie).filter(Watchlist.user_id == current_user.id).all()
        
        return jsonify({
            'watchlist': [
                {
                    'id': movie.id,
                    'title': movie.title,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{movie.poster_path}" if movie.poster_path else None,
                    'added_at': watchlist.added_at.isoformat()
                }
                for watchlist, movie in watchlist_items
            ]
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            movie_id = data.get('movie_id')
            
            if not movie_id:
                return jsonify({'message': 'Movie ID required'}), 400
            
            # Check if already in watchlist
            existing = Watchlist.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
            if existing:
                return jsonify({'message': 'Already in watchlist'}), 400
            
            # Add to watchlist
            watchlist_item = Watchlist(user_id=current_user.id, movie_id=movie_id)
            db.session.add(watchlist_item)
            db.session.commit()
            
            return jsonify({'message': 'Added to watchlist'}), 201
            
        except Exception as e:
            app.logger.error(f"Watchlist error: {e}")
            return jsonify({'message': 'Failed to add to watchlist'}), 500

@app.route('/api/watchlist/<int:movie_id>', methods=['DELETE'])
@token_required
def remove_from_watchlist(current_user, movie_id):
    try:
        watchlist_item = Watchlist.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
        if not watchlist_item:
            return jsonify({'message': 'Not in watchlist'}), 404
        
        db.session.delete(watchlist_item)
        db.session.commit()
        
        return jsonify({'message': 'Removed from watchlist'})
        
    except Exception as e:
        app.logger.error(f"Remove watchlist error: {e}")
        return jsonify({'message': 'Failed to remove from watchlist'}), 500

# Similar routes for favorites and watch history...
@app.route('/api/favorites', methods=['GET', 'POST'])
@token_required
def manage_favorites(current_user):
    if request.method == 'GET':
        favorites = db.session.query(Favorite, Movie).join(Movie).filter(Favorite.user_id == current_user.id).all()
        
        return jsonify({
            'favorites': [
                {
                    'id': movie.id,
                    'title': movie.title,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{movie.poster_path}" if movie.poster_path else None,
                    'added_at': favorite.added_at.isoformat()
                }
                for favorite, movie in favorites
            ]
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            movie_id = data.get('movie_id')
            
            existing = Favorite.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
            if existing:
                return jsonify({'message': 'Already in favorites'}), 400
            
            favorite = Favorite(user_id=current_user.id, movie_id=movie_id)
            db.session.add(favorite)
            db.session.commit()
            
            return jsonify({'message': 'Added to favorites'}), 201
            
        except Exception as e:
            return jsonify({'message': 'Failed to add to favorites'}), 500

# Recommendations Route
@app.route('/api/recommendations', methods=['GET'])
@token_required
def get_recommendations(current_user):
    try:
        # Get user preferences based on history
        watch_history = WatchHistory.query.filter_by(user_id=current_user.id).all()
        favorites = Favorite.query.filter_by(user_id=current_user.id).all()
        
        preferences = {
            'watched_movies': [wh.movie_id for wh in watch_history],
            'favorite_movies': [fav.movie_id for fav in favorites],
            'user_ratings': {str(wh.movie_id): wh.rating for wh in watch_history if wh.rating}
        }
        
        # Get recommendations from ML service
        recommended_ids = MLService.get_recommendations(current_user.id, preferences)
        
        # Fallback to trending if ML service fails
        if not recommended_ids:
            trending = TMDbService.get_trending()
            recommended_ids = [item['id'] for item in trending.get('results', [])[:10]]
        
        # Get movie details for recommendations
        recommendations = []
        for tmdb_id in recommended_ids[:10]:
            movie = Movie.query.filter_by(tmdb_id=tmdb_id).first()
            if movie:
                recommendations.append({
                    'id': movie.id,
                    'title': movie.title,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{movie.poster_path}" if movie.poster_path else None,
                    'vote_average': movie.vote_average,
                    'overview': movie.overview[:200] + '...' if len(movie.overview) > 200 else movie.overview
                })
        
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        app.logger.error(f"Recommendations error: {e}")
        return jsonify({'message': 'Failed to get recommendations'}), 500

# Admin Routes
@app.route('/api/admin/suggestions', methods=['GET', 'POST'])
@token_required
@admin_required
def manage_featured_suggestions(current_user):
    if request.method == 'GET':
        suggestions = FeaturedSuggestion.query.filter_by(is_active=True).all()
        return jsonify({
            'suggestions': [
                {
                    'id': suggestion.id,
                    'title': suggestion.title,
                    'description': suggestion.description,
                    'movie_id': suggestion.movie_id,
                    'created_at': suggestion.created_at.isoformat()
                }
                for suggestion in suggestions
            ]
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            suggestion = FeaturedSuggestion(
                movie_id=data['movie_id'],
                title=data['title'],
                description=data.get('description', ''),
                created_by=current_user.id
            )
            
            db.session.add(suggestion)
            db.session.commit()
            
            # Post to Telegram channel
            try:
                movie = Movie.query.get(data['movie_id'])
                message = f"🎬 New Featured Suggestion!\n\n{data['title']}\n\n{data.get('description', '')}\n\n#{movie.content_type} #Featured"
                bot.send_message(TELEGRAM_CHANNEL_ID, message)
            except Exception as e:
                app.logger.error(f"Telegram post error: {e}")
            
            return jsonify({'message': 'Suggestion created successfully'}), 201
            
        except Exception as e:
            app.logger.error(f"Create suggestion error: {e}")
            return jsonify({'message': 'Failed to create suggestion'}), 500

# Trending content
@app.route('/api/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        time_window = request.args.get('time', 'week')
        
        trending_data = TMDbService.get_trending(content_type, time_window)
        
        return jsonify({
            'trending': [
                {
                    'id': item['id'],
                    'title': item.get('title') or item.get('name'),
                    'poster_path': f"https://image.tmdb.org/t/p/w300{item['poster_path']}" if item.get('poster_path') else None,
                    'vote_average': item.get('vote_average', 0),
                    'media_type': item.get('media_type', content_type)
                }
                for item in trending_data.get('results', [])[:20]
            ]
        })
        
    except Exception as e:
        app.logger.error(f"Trending error: {e}")
        return jsonify({'message': 'Failed to get trending content'}), 500

# Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'message': 'Internal server error'}), 500

# Initialize database
@app.before_first_request
def create_tables():
    db.create_all()
    
    # Create admin user if doesn't exist
    admin = User.query.filter_by(email='admin@movieapp.com').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@movieapp.com',
            password_hash=bcrypt.generate_password_hash('admin123').decode('utf-8'),
            is_admin=True
        )
        db.session.add(admin)
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)