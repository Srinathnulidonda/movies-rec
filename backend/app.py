from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import os
from datetime import datetime, timedelta
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///movies.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-string')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app)

# API Keys
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
JUSTWATCH_API_KEY = os.getenv('JUSTWATCH_API_KEY', '7689567537:AAGvDtu94OlLlTiWpfjSfpl_dd_Osi_2W7c')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7689567537:AAGvDtu94OlLlTiWpfjSfpl_dd_Osi_2W7c')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '-1002566510721')
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'http://localhost:5001')

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, unique=True, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    overview = db.Column(db.Text)
    poster_path = db.Column(db.String(200))
    release_date = db.Column(db.String(20))
    rating = db.Column(db.Float)
    genres = db.Column(db.String(500))
    runtime = db.Column(db.Integer)
    content_type = db.Column(db.String(20))  # movie, tv, anime
    watch_providers = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Wishlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Favorites(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class WatchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    watched_at = db.Column(db.DateTime, default=datetime.utcnow)
    rating = db.Column(db.Integer)  # 1-10

class FeaturedContent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    description = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Helper Functions
def get_movie_from_tmdb(tmdb_id, content_type='movie'):
    """Fetch movie/TV show details from TMDb API"""
    endpoint = 'movie' if content_type == 'movie' else 'tv'
    url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}"
    params = {'api_key': TMDB_API_KEY, 'append_to_response': 'watch/providers'}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logging.error(f"TMDb API error: {e}")
    return None

def get_watch_providers(tmdb_id, content_type='movie'):
    """Get streaming providers for content"""
    endpoint = 'movie' if content_type == 'movie' else 'tv'
    url = f"https://api.themoviedb.org/3/{endpoint}/{tmdb_id}/watch/providers"
    params = {'api_key': TMDB_API_KEY}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            providers = []
            us_providers = data.get('results', {}).get('US', {})
            for provider_type in ['flatrate', 'buy', 'rent']:
                if provider_type in us_providers:
                    for provider in us_providers[provider_type]:
                        providers.append({
                            'name': provider['provider_name'],
                            'logo': f"https://image.tmdb.org/t/p/w45{provider['logo_path']}",
                            'link': us_providers.get('link', '#')
                        })
            return providers
    except Exception as e:
        logging.error(f"Watch providers API error: {e}")
    return []

def search_tmdb(query, page=1):
    """Search movies and TV shows on TMDb"""
    url = "https://api.themoviedb.org/3/search/multi"
    params = {
        'api_key': TMDB_API_KEY,
        'query': query,
        'page': page
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logging.error(f"TMDb search error: {e}")
    return {'results': []}

def get_trending_content():
    """Get trending movies and TV shows"""
    url = "https://api.themoviedb.org/3/trending/all/week"
    params = {'api_key': TMDB_API_KEY}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logging.error(f"TMDb trending error: {e}")
    return {'results': []}

def format_content_item(item, content_type=None):
    """Format TMDb item for frontend"""
    if not content_type:
        content_type = item.get('media_type', 'movie')
    
    title = item.get('title') or item.get('name', 'Unknown Title')
    poster_path = item.get('poster_path')
    poster = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else '/static/placeholder.jpg'
    
    return {
        'id': item.get('id'),
        'title': title,
        'poster': poster,
        'year': (item.get('release_date') or item.get('first_air_date', ''))[:4],
        'rating': round(item.get('vote_average', 0), 1),
        'type': content_type,
        'overview': item.get('overview', '')[:200] + '...' if item.get('overview', '') else ''
    }

def send_telegram_message(message):
    """Send message to Telegram channel"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, data=data)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Telegram API error: {e}")
        return False

# Routes
@app.route('/')
def home():
    return jsonify({'message': 'Movie Recommendation API is running!'})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
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
    return jsonify({'token': token, 'user_id': user.id})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password_hash, password):
        token = create_access_token(identity=user.id)
        return jsonify({'token': token, 'user_id': user.id})
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/featured')
def get_featured():
    """Get featured content"""
    try:
        # First check database for featured content
        featured = FeaturedContent.query.filter_by(is_active=True).limit(20).all()
        if featured:
            results = []
            for item in featured:
                movie = Movie.query.get(item.movie_id)
                if movie:
                    results.append({
                        'id': movie.tmdb_id,
                        'title': movie.title,
                        'poster': f"https://image.tmdb.org/t/p/w500{movie.poster_path}" if movie.poster_path else '/static/placeholder.jpg',
                        'year': movie.release_date[:4] if movie.release_date else '',
                        'rating': movie.rating or 0,
                        'type': movie.content_type,
                        'overview': movie.overview[:200] + '...' if movie.overview else ''
                    })
            if results:
                return jsonify({'results': results})
        
        # Fallback to trending content from TMDb
        trending = get_trending_content()
        results = [format_content_item(item) for item in trending.get('results', [])[:20]]
        return jsonify({'results': results})
    
    except Exception as e:
        logging.error(f"Featured content error: {e}")
        return jsonify({'error': 'Failed to fetch featured content'}), 500

@app.route('/api/search')
def search():
    """Search for movies, TV shows, and anime"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'results': []})
    
    try:
        search_results = search_tmdb(query)
        results = [format_content_item(item) for item in search_results.get('results', [])[:20]]
        return jsonify({'results': results})
    
    except Exception as e:
        logging.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/filter')
def filter_content():
    """Filter content by genre"""
    genre = request.args.get('genre', '')
    
    try:
        if genre == 'anime':
            # Search for anime content
            url = "https://api.themoviedb.org/3/discover/tv"
            params = {
                'api_key': TMDB_API_KEY,
                'with_keywords': '210024',  # Anime keyword ID
                'sort_by': 'popularity.desc'
            }
        else:
            # Search by genre
            genre_map = {
                'action': 28, 'comedy': 35, 'drama': 18, 'horror': 27,
                'romance': 10749, 'thriller': 53, 'sci-fi': 878
            }
            genre_id = genre_map.get(genre.lower(), 28)
            
            url = "https://api.themoviedb.org/3/discover/movie"
            params = {
                'api_key': TMDB_API_KEY,
                'with_genres': genre_id,
                'sort_by': 'popularity.desc'
            }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            content_type = 'tv' if genre == 'anime' else 'movie'
            results = [format_content_item(item, content_type) for item in data.get('results', [])[:20]]
            return jsonify({'results': results})
        
        return jsonify({'results': []})
    
    except Exception as e:
        logging.error(f"Filter error: {e}")
        return jsonify({'error': 'Filter failed'}), 500

@app.route('/api/details/<int:tmdb_id>')
def get_details(tmdb_id):
    """Get detailed information about a movie/TV show"""
    content_type = request.args.get('type', 'movie')
    
    try:
        # Check if movie exists in database
        movie = Movie.query.filter_by(tmdb_id=tmdb_id).first()
        if not movie:
            # Fetch from TMDb and save to database
            tmdb_data = get_movie_from_tmdb(tmdb_id, content_type)
            if not tmdb_data:
                return jsonify({'error': 'Content not found'}), 404
            
            # Save to database
            watch_providers = get_watch_providers(tmdb_id, content_type)
            movie = Movie(
                tmdb_id=tmdb_id,
                title=tmdb_data.get('title') or tmdb_data.get('name', ''),
                overview=tmdb_data.get('overview', ''),
                poster_path=tmdb_data.get('poster_path', ''),
                release_date=tmdb_data.get('release_date') or tmdb_data.get('first_air_date', ''),
                rating=tmdb_data.get('vote_average', 0),
                genres=','.join([g['name'] for g in tmdb_data.get('genres', [])]),
                runtime=tmdb_data.get('runtime') or tmdb_data.get('episode_run_time', [0])[0] if tmdb_data.get('episode_run_time') else 0,
                content_type=content_type,
                watch_providers=str(watch_providers)
            )
            db.session.add(movie)
            db.session.commit()
        
        # Get watch providers
        providers = eval(movie.watch_providers) if movie.watch_providers else []
        
        return jsonify({
            'id': movie.tmdb_id,
            'title': movie.title,
            'overview': movie.overview,
            'poster': f"https://image.tmdb.org/t/p/w500{movie.poster_path}" if movie.poster_path else '/static/placeholder.jpg',
            'release_date': movie.release_date,
            'rating': movie.rating,
            'genres': movie.genres.split(',') if movie.genres else [],
            'runtime': movie.runtime,
            'watch_providers': providers
        })
    
    except Exception as e:
        logging.error(f"Details error: {e}")
        return jsonify({'error': 'Failed to fetch details'}), 500

@app.route('/api/recommendations')
@jwt_required()
def get_recommendations():
    """Get personalized recommendations"""
    user_id = get_jwt_identity()
    
    try:
        # Get user's watch history, favorites, and wishlist
        watch_history = db.session.query(WatchHistory.movie_id).filter_by(user_id=user_id).all()
        favorites = db.session.query(Favorites.movie_id).filter_by(user_id=user_id).all()
        
        user_data = {
            'user_id': user_id,
            'watch_history': [w[0] for w in watch_history],
            'favorites': [f[0] for f in favorites]
        }
        
        # Call ML service for recommendations
        try:
            ml_response = requests.post(f"{ML_SERVICE_URL}/recommend", 
                                      json=user_data, timeout=10)
            if ml_response.status_code == 200:
                ml_data = ml_response.json()
                recommended_ids = ml_data.get('recommendations', [])
            else:
                # Fallback to popular content
                trending = get_trending_content()
                recommended_ids = [item['id'] for item in trending.get('results', [])[:10]]
        except:
            # Fallback to trending if ML service is down
            trending = get_trending_content()
            recommended_ids = [item['id'] for item in trending.get('results', [])[:10]]
        
        # Get movie details for recommendations
        results = []
        for tmdb_id in recommended_ids[:10]:
            movie = Movie.query.filter_by(tmdb_id=tmdb_id).first()
            if movie:
                results.append({
                    'id': movie.tmdb_id,
                    'title': movie.title,
                    'poster': f"https://image.tmdb.org/t/p/w500{movie.poster_path}" if movie.poster_path else '/static/placeholder.jpg',
                    'year': movie.release_date[:4] if movie.release_date else '',
                    'rating': movie.rating,
                    'type': movie.content_type
                })
        
        return jsonify({'results': results})
    
    except Exception as e:
        logging.error(f"Recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@app.route('/api/wishlist', methods=['GET', 'POST'])
@jwt_required()
def handle_wishlist():
    user_id = get_jwt_identity()
    
    if request.method == 'GET':
        try:
            wishlist_items = db.session.query(Wishlist, Movie).join(Movie).filter(Wishlist.user_id == user_id).all()
            results = []
            for wishlist, movie in wishlist_items:
                results.append({
                    'id': movie.tmdb_id,
                    'title': movie.title,
                    'poster': f"https://image.tmdb.org/t/p/w500{movie.poster_path}" if movie.poster_path else '/static/placeholder.jpg',
                    'year': movie.release_date[:4] if movie.release_date else '',
                    'rating': movie.rating,
                    'type': movie.content_type
                })
            return jsonify({'results': results})
        except Exception as e:
            logging.error(f"Wishlist GET error: {e}")
            return jsonify({'error': 'Failed to fetch wishlist'}), 500
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            movie_tmdb_id = data.get('movie_id')
            
            # Find or create movie record
            movie = Movie.query.filter_by(tmdb_id=movie_tmdb_id).first()
            if not movie:
                return jsonify({'error': 'Movie not found'}), 404
            
            # Check if already in wishlist
            existing = Wishlist.query.filter_by(user_id=user_id, movie_id=movie.id).first()
            if existing:
                return jsonify({'message': 'Already in wishlist'})
            
            wishlist_item = Wishlist(user_id=user_id, movie_id=movie.id)
            db.session.add(wishlist_item)
            db.session.commit()
            
            return jsonify({'message': 'Added to wishlist'})
        except Exception as e:
            logging.error(f"Wishlist POST error: {e}")
            return jsonify({'error': 'Failed to add to wishlist'}), 500

@app.route('/api/favorites', methods=['GET', 'POST'])
@jwt_required()
def handle_favorites():
    user_id = get_jwt_identity()
    
    if request.method == 'GET':
        try:
            favorite_items = db.session.query(Favorites, Movie).join(Movie).filter(Favorites.user_id == user_id).all()
            results = []
            for favorite, movie in favorite_items:
                results.append({
                    'id': movie.tmdb_id,
                    'title': movie.title,
                    'poster': f"https://image.tmdb.org/t/p/w500{movie.poster_path}" if movie.poster_path else '/static/placeholder.jpg',
                    'year': movie.release_date[:4] if movie.release_date else '',
                    'rating': movie.rating,
                    'type': movie.content_type
                })
            return jsonify({'results': results})
        except Exception as e:
            logging.error(f"Favorites GET error: {e}")
            return jsonify({'error': 'Failed to fetch favorites'}), 500
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            movie_tmdb_id = data.get('movie_id')
            
            movie = Movie.query.filter_by(tmdb_id=movie_tmdb_id).first()
            if not movie:
                return jsonify({'error': 'Movie not found'}), 404
            
            existing = Favorites.query.filter_by(user_id=user_id, movie_id=movie.id).first()
            if existing:
                return jsonify({'message': 'Already in favorites'})
            
            favorite_item = Favorites(user_id=user_id, movie_id=movie.id)
            db.session.add(favorite_item)
            db.session.commit()
            
            return jsonify({'message': 'Added to favorites'})
        except Exception as e:
            logging.error(f"Favorites POST error: {e}")
            return jsonify({'error': 'Failed to add to favorites'}), 500

@app.route('/api/admin/featured', methods=['POST'])
@jwt_required()
def post_featured():
    """Admin: Add featured content"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    try:
        data = request.get_json()
        movie_tmdb_id = data.get('movie_id')
        description = data.get('description', '')
        
        # Find or create movie record
        movie = Movie.query.filter_by(tmdb_id=movie_tmdb_id).first()
        if not movie:
            # Fetch from TMDb
            tmdb_data = get_movie_from_tmdb(movie_tmdb_id)
            if not tmdb_data:
                return jsonify({'error': 'Movie not found'}), 404
            
            movie = Movie(
                tmdb_id=movie_tmdb_id,
                title=tmdb_data.get('title', ''),
                overview=tmdb_data.get('overview', ''),
                poster_path=tmdb_data.get('poster_path', ''),
                release_date=tmdb_data.get('release_date', ''),
                rating=tmdb_data.get('vote_average', 0),
                genres=','.join([g['name'] for g in tmdb_data.get('genres', [])]),
                runtime=tmdb_data.get('runtime', 0),
                content_type='movie'
            )
            db.session.add(movie)
            db.session.commit()
        
        # Add to featured
        featured = FeaturedContent(movie_id=movie.id, description=description)
        db.session.add(featured)
        db.session.commit()
        
        return jsonify({'message': 'Added to featured content'})
    
    except Exception as e:
        logging.error(f"Featured POST error: {e}")
        return jsonify({'error': 'Failed to add featured content'}), 500

@app.route('/api/admin/telegram', methods=['POST'])
@jwt_required()
def post_to_telegram():
    """Admin: Post message to Telegram channel"""
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if send_telegram_message(message):
            return jsonify({'message': 'Posted to Telegram successfully'})
        else:
            return jsonify({'error': 'Failed to post to Telegram'}), 500
    
    except Exception as e:
        logging.error(f"Telegram POST error: {e}")
        return jsonify({'error': 'Failed to post to Telegram'}), 500

@app.route('/api/watch_history', methods=['POST'])
@jwt_required()
def add_to_watch_history():
    """Add movie to user's watch history"""
    user_id = get_jwt_identity()
    
    try:
        data = request.get_json()
        movie_tmdb_id = data.get('movie_id')
        rating = data.get('rating', None)
        
        movie = Movie.query.filter_by(tmdb_id=movie_tmdb_id).first()
        if not movie:
            return jsonify({'error': 'Movie not found'}), 404
        
        # Check if already watched
        existing = WatchHistory.query.filter_by(user_id=user_id, movie_id=movie.id).first()
        if existing:
            existing.watched_at = datetime.utcnow()
            if rating:
                existing.rating = rating
        else:
            watch_item = WatchHistory(user_id=user_id, movie_id=movie.id, rating=rating)
            db.session.add(watch_item)
        
        db.session.commit()
        return jsonify({'message': 'Added to watch history'})
    
    except Exception as e:
        logging.error(f"Watch history error: {e}")
        return jsonify({'error': 'Failed to add to watch history'}), 500

# Initialize database
@app.before_first_request
def create_tables():
    db.create_all()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Create admin user if doesn't exist
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin_user = User(
                username='admin',
                email='admin@movieapp.com',
                password_hash=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created: username=admin, password=admin123")
    
    app.run(debug=True, host='0.0.0.0', port=5000)