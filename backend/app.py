#backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import requests
import os
from datetime import datetime, timedelta
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///movies.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
CORS(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    title = db.Column(db.String(200), nullable=False)
    overview = db.Column(db.Text)
    poster_path = db.Column(db.String(200))
    release_date = db.Column(db.String(20))
    genre = db.Column(db.String(100))
    rating = db.Column(db.Float)
    type = db.Column(db.String(20), default='movie')  # movie, anime, series

class Wishlist(db.Model):
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

# API Keys
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID')
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'https://movies-rec-ml-service.onrender.com')

# Helper Functions
def get_tmdb_data(endpoint, params=None):
    base_url = "https://api.themoviedb.org/3"
    if params is None:
        params = {}
    params['api_key'] = TMDB_API_KEY
    
    try:
        response = requests.get(f"{base_url}/{endpoint}", params=params)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def post_to_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHANNEL_ID:
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        'chat_id': TELEGRAM_CHANNEL_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, data=data)
        return response.status_code == 200
    except:
        return False

# Routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    password_hash = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    user = User(username=data['username'], email=data['email'], password_hash=password_hash)
    
    db.session.add(user)
    db.session.commit()
    
    access_token = create_access_token(identity=user.id)
    return jsonify({'access_token': access_token, 'user_id': user.id}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    
    if user and bcrypt.check_password_hash(user.password_hash, data['password']):
        access_token = create_access_token(identity=user.id)
        return jsonify({'access_token': access_token, 'user_id': user.id}), 200
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/movies/popular', methods=['GET'])
def get_popular_movies():
    data = get_tmdb_data('movie/popular')
    return jsonify(data) if data else jsonify({'error': 'Failed to fetch data'}), 500

@app.route('/api/movies/search', methods=['GET'])
def search_movies():
    query = request.args.get('q', '')
    data = get_tmdb_data('search/movie', {'query': query})
    return jsonify(data) if data else jsonify({'error': 'Failed to fetch data'}), 500

@app.route('/api/anime/popular', methods=['GET'])
def get_popular_anime():
    # Using TMDB's discover endpoint with anime genres
    data = get_tmdb_data('discover/movie', {
        'with_genres': '16',  # Animation genre
        'with_keywords': '210024',  # Anime keyword
        'sort_by': 'popularity.desc'
    })
    return jsonify(data) if data else jsonify({'error': 'Failed to fetch data'}), 500

@app.route('/api/series/popular', methods=['GET'])
def get_popular_series():
    data = get_tmdb_data('tv/popular')
    return jsonify(data) if data else jsonify({'error': 'Failed to fetch data'}), 500

@app.route('/api/wishlist', methods=['GET', 'POST', 'DELETE'])
@jwt_required()
def handle_wishlist():
    user_id = get_jwt_identity()
    
    if request.method == 'GET':
        wishlist_items = db.session.query(Wishlist, Movie).join(Movie).filter(Wishlist.user_id == user_id).all()
        result = []
        for wishlist, movie in wishlist_items:
            result.append({
                'id': movie.id,
                'title': movie.title,
                'poster_path': movie.poster_path,
                'overview': movie.overview,
                'rating': movie.rating,
                'added_at': wishlist.added_at.isoformat()
            })
        return jsonify(result)
    
    elif request.method == 'POST':
        data = request.get_json()
        movie = Movie.query.filter_by(tmdb_id=data['tmdb_id']).first()
        
        if not movie:
            movie = Movie(
                tmdb_id=data['tmdb_id'],
                title=data['title'],
                overview=data.get('overview', ''),
                poster_path=data.get('poster_path', ''),
                release_date=data.get('release_date', ''),
                genre=data.get('genre', ''),
                rating=data.get('rating', 0),
                type=data.get('type', 'movie')
            )
            db.session.add(movie)
            db.session.commit()
        
        existing = Wishlist.query.filter_by(user_id=user_id, movie_id=movie.id).first()
        if not existing:
            wishlist_item = Wishlist(user_id=user_id, movie_id=movie.id)
            db.session.add(wishlist_item)
            db.session.commit()
        
        return jsonify({'message': 'Added to wishlist'}), 201
    
    elif request.method == 'DELETE':
        movie_id = request.args.get('movie_id')
        wishlist_item = Wishlist.query.filter_by(user_id=user_id, movie_id=movie_id).first()
        if wishlist_item:
            db.session.delete(wishlist_item)
            db.session.commit()
        return jsonify({'message': 'Removed from wishlist'}), 200

@app.route('/api/favorites', methods=['GET', 'POST', 'DELETE'])
@jwt_required()
def handle_favorites():
    user_id = get_jwt_identity()
    
    if request.method == 'GET':
        favorite_items = db.session.query(Favorite, Movie).join(Movie).filter(Favorite.user_id == user_id).all()
        result = []
        for favorite, movie in favorite_items:
            result.append({
                'id': movie.id,
                'title': movie.title,
                'poster_path': movie.poster_path,
                'overview': movie.overview,
                'rating': movie.rating,
                'added_at': favorite.added_at.isoformat()
            })
        return jsonify(result)
    
    elif request.method == 'POST':
        data = request.get_json()
        movie = Movie.query.filter_by(tmdb_id=data['tmdb_id']).first()
        
        if not movie:
            movie = Movie(
                tmdb_id=data['tmdb_id'],
                title=data['title'],
                overview=data.get('overview', ''),
                poster_path=data.get('poster_path', ''),
                release_date=data.get('release_date', ''),
                genre=data.get('genre', ''),
                rating=data.get('rating', 0),
                type=data.get('type', 'movie')
            )
            db.session.add(movie)
            db.session.commit()
        
        existing = Favorite.query.filter_by(user_id=user_id, movie_id=movie.id).first()
        if not existing:
            favorite_item = Favorite(user_id=user_id, movie_id=movie.id)
            db.session.add(favorite_item)
            db.session.commit()
        
        return jsonify({'message': 'Added to favorites'}), 201
    
    elif request.method == 'DELETE':
        movie_id = request.args.get('movie_id')
        favorite_item = Favorite.query.filter_by(user_id=user_id, movie_id=movie_id).first()
        if favorite_item:
            db.session.delete(favorite_item)
            db.session.commit()
        return jsonify({'message': 'Removed from favorites'}), 200

@app.route('/api/recommendations', methods=['GET'])
@jwt_required()
def get_recommendations():
    user_id = get_jwt_identity()
    
    # Get user's watch history and favorites for ML service
    watch_history = WatchHistory.query.filter_by(user_id=user_id).all()
    favorites = Favorite.query.filter_by(user_id=user_id).all()
    
    user_data = {
        'user_id': user_id,
        'watch_history': [w.movie_id for w in watch_history],
        'favorites': [f.movie_id for f in favorites]
    }
    
    try:
        response = requests.post(f"{ML_SERVICE_URL}/recommend", json=user_data)
        if response.status_code == 200:
            return jsonify(response.json())
    except:
        pass
    
    # Fallback to popular movies if ML service fails
    data = get_tmdb_data('movie/popular')
    return jsonify(data) if data else jsonify({'error': 'Failed to fetch recommendations'}), 500

@app.route('/api/admin/post-suggestion', methods=['POST'])
@jwt_required()
def admin_post_suggestion():
    data = request.get_json()
    message = f"🎬 <b>{data['title']}</b>\n\n{data['description']}\n\n⭐ Rating: {data.get('rating', 'N/A')}\n🎭 Genre: {data.get('genre', 'N/A')}"
    
    if post_to_telegram(message):
        return jsonify({'message': 'Posted to Telegram successfully'}), 200
    else:
        return jsonify({'error': 'Failed to post to Telegram'}), 500

@app.route('/api/watch-history', methods=['POST'])
@jwt_required()
def add_to_watch_history():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    movie = Movie.query.filter_by(tmdb_id=data['tmdb_id']).first()
    if not movie:
        movie = Movie(
            tmdb_id=data['tmdb_id'],
            title=data['title'],
            overview=data.get('overview', ''),
            poster_path=data.get('poster_path', ''),
            release_date=data.get('release_date', ''),
            genre=data.get('genre', ''),
            rating=data.get('rating', 0),
            type=data.get('type', 'movie')
        )
        db.session.add(movie)
        db.session.commit()
    
    # Remove existing entry if present
    existing = WatchHistory.query.filter_by(user_id=user_id, movie_id=movie.id).first()
    if existing:
        db.session.delete(existing)
    
    # Add new entry
    history_item = WatchHistory(user_id=user_id, movie_id=movie.id)
    db.session.add(history_item)
    db.session.commit()
    
    return jsonify({'message': 'Added to watch history'}), 201

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))