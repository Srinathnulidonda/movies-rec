from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import os
import requests
import sqlite3
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Initialize extensions
jwt = JWTManager(app)
CORS(app, origins=["*"])

# API Keys
TMDB_API_KEY = os.getenv('TMDB_API_KEY', 'your-tmdb-api-key')
JUSTWATCH_API_KEY = os.getenv('JUSTWATCH_API_KEY', 'your-justwatch-api-key')
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'https://your-ml-service.render.com')

# Database setup
def init_db():
    conn = sqlite3.connect('cinema.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Wishlist table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wishlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            movie_title TEXT NOT NULL,
            movie_poster TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, movie_id)
        )
    ''')
    
    # Favorites table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            movie_title TEXT NOT NULL,
            movie_poster TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, movie_id)
        )
    ''')
    
    # Watch history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watch_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            movie_title TEXT NOT NULL,
            watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Admin suggestions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            poster_url TEXT,
            is_featured BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Helper functions
def get_db_connection():
    conn = sqlite3.connect('cinema.db')
    conn.row_factory = sqlite3.Row
    return conn

def tmdb_request(endpoint, params=None):
    """Make request to TMDB API"""
    if params is None:
        params = {}
    params['api_key'] = TMDB_API_KEY
    
    try:
        response = requests.get(f'https://api.themoviedb.org/3{endpoint}', params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"TMDB API error: {e}")
        return None

def get_watch_providers(movie_id):
    """Get streaming providers for a movie"""
    try:
        # Using TMDB watch providers
        providers = tmdb_request(f'/movie/{movie_id}/watch/providers')
        if providers and 'results' in providers:
            us_providers = providers['results'].get('US', {})
            watch_providers = []
            
            # Get streaming providers
            if 'flatrate' in us_providers:
                for provider in us_providers['flatrate']:
                    watch_providers.append({
                        'provider': provider['provider_name'],
                        'link': f"https://www.themoviedb.org/movie/{movie_id}/watch"
                    })
            
            # Get rent/buy providers
            if 'rent' in us_providers:
                for provider in us_providers['rent'][:3]:  # Limit to 3
                    watch_providers.append({
                        'provider': f"{provider['provider_name']} (Rent)",
                        'link': f"https://www.themoviedb.org/movie/{movie_id}/watch"
                    })
            
            return watch_providers
        
        return []
    except Exception as e:
        logger.error(f"Error getting watch providers: {e}")
        return []

def format_movie_data(movie):
    """Format movie data for frontend"""
    return {
        'id': movie.get('id'),
        'title': movie.get('title') or movie.get('name', 'Unknown Title'),
        'overview': movie.get('overview', 'No description available'),
        'poster': f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else None,
        'backdrop': f"https://image.tmdb.org/t/p/w1280{movie.get('backdrop_path')}" if movie.get('backdrop_path') else None,
        'year': movie.get('release_date', movie.get('first_air_date', ''))[:4] if movie.get('release_date') or movie.get('first_air_date') else 'Unknown',
        'rating': round(movie.get('vote_average', 0), 1),
        'genres': [genre['name'] for genre in movie.get('genres', [])] if movie.get('genres') else []
    }

# Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create new user
        password_hash = generate_password_hash(password)
        cursor.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)', 
                      (email, password_hash))
        conn.commit()
        
        user_id = cursor.lastrowid
        conn.close()
        
        # Create token
        token = create_access_token(identity=user_id)
        
        return jsonify({
            'token': token,
            'user': {'id': user_id, 'email': email}
        })
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, password_hash FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        token = create_access_token(identity=user['id'])
        
        return jsonify({
            'token': token,
            'user': {'id': user['id'], 'email': email}
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/featured', methods=['GET'])
def get_featured():
    try:
        # Get trending movies
        trending = tmdb_request('/trending/movie/day')
        
        if not trending or 'results' not in trending:
            return jsonify({'error': 'Failed to fetch featured content'}), 500
        
        # Format movies
        movies = [format_movie_data(movie) for movie in trending['results'][:12]]
        
        return jsonify({'results': movies})
        
    except Exception as e:
        logger.error(f"Featured content error: {e}")
        return jsonify({'error': 'Failed to fetch featured content'}), 500

@app.route('/api/search', methods=['GET'])
def search_content():
    try:
        query = request.args.get('q', '').strip()
        content_type = request.args.get('type', 'multi')
        page = int(request.args.get('page', 1))
        
        if not query and content_type == 'multi':
            return jsonify({'results': []})
        
        if query:
            # Search by query
            if content_type == 'anime':
                # Search for anime using TV search with anime keywords
                search_result = tmdb_request('/search/tv', {
                    'query': f"{query} anime",
                    'page': page
                })
            else:
                endpoint = '/search/multi' if content_type == 'multi' else f'/search/{content_type}'
                search_result = tmdb_request(endpoint, {
                    'query': query,
                    'page': page
                })
        else:
            # Get popular content by type
            if content_type == 'movie':
                search_result = tmdb_request('/movie/popular', {'page': page})
            elif content_type == 'tv':
                search_result = tmdb_request('/tv/popular', {'page': page})
            elif content_type == 'anime':
                search_result = tmdb_request('/discover/tv', {
                    'with_genres': '16',  # Animation genre
                    'with_origin_country': 'JP',
                    'page': page
                })
            else:
                search_result = tmdb_request('/movie/popular', {'page': page})
        
        if not search_result or 'results' not in search_result:
            return jsonify({'results': []})
        
        movies = [format_movie_data(movie) for movie in search_result['results']]
        
        return jsonify({
            'results': movies,
            'page': page,
            'total_pages': search_result.get('total_pages', 1)
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    try:
        # Get movie details
        movie = tmdb_request(f'/movie/{movie_id}')
        if not movie:
            return jsonify({'error': 'Movie not found'}), 404
        
        # Get credits
        credits = tmdb_request(f'/movie/{movie_id}/credits')
        cast = []
        if credits and 'cast' in credits:
            cast = [actor['name'] for actor in credits['cast'][:10]]
        
        # Get watch providers
        watch_providers = get_watch_providers(movie_id)
        
        # Format movie details
        movie_details = format_movie_data(movie)
        movie_details.update({
            'runtime': movie.get('runtime', 0),
            'cast': cast,
            'watch_providers': watch_providers,
            'budget': movie.get('budget', 0),
            'revenue': movie.get('revenue', 0),
            'homepage': movie.get('homepage', ''),
            'imdb_id': movie.get('imdb_id', ''),
            'tagline': movie.get('tagline', ''),
            'production_companies': [company['name'] for company in movie.get('production_companies', [])]
        })
        
        return jsonify({'movie': movie_details})
        
    except Exception as e:
        logger.error(f"Movie details error: {e}")
        return jsonify({'error': 'Failed to fetch movie details'}), 500

@app.route('/api/recommendations', methods=['GET'])
@jwt_required()
def get_recommendations():
    try:
        user_id = get_jwt_identity()
        
        # Get user's watch history, favorites, and wishlist
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT movie_id FROM favorites WHERE user_id = ?
            UNION
            SELECT movie_id FROM wishlist WHERE user_id = ?
            UNION
            SELECT movie_id FROM watch_history WHERE user_id = ?
        ''', (user_id, user_id, user_id))
        
        user_movies = [row['movie_id'] for row in cursor.fetchall()]
        conn.close()
        
        if user_movies:
            # Call ML service for personalized recommendations
            try:
                ml_response = requests.post(f'{ML_SERVICE_URL}/recommend', 
                                          json={'user_movies': user_movies},
                                          timeout=10)
                if ml_response.status_code == 200:
                    recommended_ids = ml_response.json().get('recommendations', [])
                else:
                    recommended_ids = []
            except:
                recommended_ids = []
        else:
            recommended_ids = []
        
        # Fallback to popular movies if no ML recommendations
        if not recommended_ids:
            popular = tmdb_request('/movie/popular')
            if popular and 'results' in popular:
                movies = [format_movie_data(movie) for movie in popular['results'][:12]]
            else:
                movies = []
        else:
            # Get details for recommended movies
            movies = []
            for movie_id in recommended_ids[:12]:
                movie = tmdb_request(f'/movie/{movie_id}')
                if movie:
                    movies.append(format_movie_data(movie))
        
        return jsonify({'results': movies})
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return jsonify({'error': 'Failed to fetch recommendations'}), 500

@app.route('/api/wishlist', methods=['GET', 'POST', 'DELETE'])
@jwt_required()
def manage_wishlist():
    try:
        user_id = get_jwt_identity()
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if request.method == 'GET':
            cursor.execute('''
                SELECT movie_id, movie_title, movie_poster, added_at
                FROM wishlist WHERE user_id = ?
                ORDER BY added_at DESC
            ''', (user_id,))
            
            wishlist_items = cursor.fetchall()
            movies = []
            
            for item in wishlist_items:
                movies.append({
                    'id': item['movie_id'],
                    'title': item['movie_title'],
                    'poster': item['movie_poster'],
                    'added_at': item['added_at']
                })
            
            conn.close()
            return jsonify({'results': movies})
        
        elif request.method == 'POST':
            data = request.get_json()
            movie_id = data.get('movie_id')
            
            if not movie_id:
                return jsonify({'error': 'Movie ID required'}), 400
            
            # Get movie details from TMDB
            movie = tmdb_request(f'/movie/{movie_id}')
            if not movie:
                return jsonify({'error': 'Movie not found'}), 404
            
            try:
                cursor.execute('''
                    INSERT INTO wishlist (user_id, movie_id, movie_title, movie_poster)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, movie_id, movie['title'], 
                     f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None))
                conn.commit()
                conn.close()
                return jsonify({'message': 'Added to wishlist'})
            except sqlite3.IntegrityError:
                conn.close()
                return jsonify({'error': 'Already in wishlist'}), 400
        
        elif request.method == 'DELETE':
            movie_id = request.args.get('movie_id')
            if not movie_id:
                return jsonify({'error': 'Movie ID required'}), 400
            
            cursor.execute('DELETE FROM wishlist WHERE user_id = ? AND movie_id = ?', 
                          (user_id, movie_id))
            conn.commit()
            conn.close()
            return jsonify({'message': 'Removed from wishlist'})
        
    except Exception as e:
        logger.error(f"Wishlist error: {e}")
        return jsonify({'error': 'Wishlist operation failed'}), 500

@app.route('/api/favorites', methods=['GET', 'POST', 'DELETE'])
@jwt_required()
def manage_favorites():
    try:
        user_id = get_jwt_identity()
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if request.method == 'GET':
            cursor.execute('''
                SELECT movie_id, movie_title, movie_poster, added_at
                FROM favorites WHERE user_id = ?
                ORDER BY added_at DESC
            ''', (user_id,))
            
            favorite_items = cursor.fetchall()
            movies = []
            
            for item in favorite_items:
                movies.append({
                    'id': item['movie_id'],
                    'title': item['movie_title'],
                    'poster': item['movie_poster'],
                    'added_at': item['added_at']
                })
            
            conn.close()
            return jsonify({'results': movies})
        
        elif request.method == 'POST':
            data = request.get_json()
            movie_id = data.get('movie_id')
            
            if not movie_id:
                return jsonify({'error': 'Movie ID required'}), 400
            
            # Get movie details from TMDB
            movie = tmdb_request(f'/movie/{movie_id}')
            if not movie:
                return jsonify({'error': 'Movie not found'}), 404
            
            try:
                cursor.execute('''
                    INSERT INTO favorites (user_id, movie_id, movie_title, movie_poster)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, movie_id, movie['title'], 
                     f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None))
                conn.commit()
                conn.close()
                return jsonify({'message': 'Added to favorites'})
            except sqlite3.IntegrityError:
                conn.close()
                return jsonify({'error': 'Already in favorites'}), 400
        
        elif request.method == 'DELETE':
            movie_id = request.args.get('movie_id')
            if not movie_id:
                return jsonify({'error': 'Movie ID required'}), 400
            
            cursor.execute('DELETE FROM favorites WHERE user_id = ? AND movie_id = ?', 
                          (user_id, movie_id))
            conn.commit()
            conn.close()
            return jsonify({'message': 'Removed from favorites'})
        
    except Exception as e:
        logger.error(f"Favorites error: {e}")
        return jsonify({'error': 'Favorites operation failed'}), 500

@app.route('/api/watch-history', methods=['GET', 'POST'])
@jwt_required()
def manage_watch_history():
    try:
        user_id = get_jwt_identity()
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if request.method == 'GET':
            cursor.execute('''
                SELECT movie_id, movie_title, watched_at
                FROM watch_history WHERE user_id = ?
                ORDER BY watched_at DESC
                LIMIT 50
            ''', (user_id,))
            
            history = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return jsonify({'results': history})
        
        elif request.method == 'POST':
            data = request.get_json()
            movie_id = data.get('movie_id')
            movie_title = data.get('movie_title')
            
            if not movie_id or not movie_title:
                return jsonify({'error': 'Movie ID and title required'}), 400
            
            cursor.execute('''
                INSERT INTO watch_history (user_id, movie_id, movie_title)
                VALUES (?, ?, ?)
            ''', (user_id, movie_id, movie_title))
            conn.commit()
            conn.close()
            return jsonify({'message': 'Added to watch history'})
        
    except Exception as e:
        logger.error(f"Watch history error: {e}")
        return jsonify({'error': 'Watch history operation failed'}), 500

# Admin routes
@app.route('/api/admin/suggestions', methods=['GET', 'POST'])
@jwt_required()
def admin_suggestions():
    try:
        # For demo purposes, no admin verification
        # In production, add admin role checking
        
        if request.method == 'GET':
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM admin_suggestions
                ORDER BY created_at DESC
            ''')
            suggestions = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return jsonify({'results': suggestions})
        
        elif request.method == 'POST':
            data = request.get_json()
            movie_id = data.get('movie_id')
            title = data.get('title')
            description = data.get('description')
            poster_url = data.get('poster_url')
            is_featured = data.get('is_featured', False)
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO admin_suggestions 
                (movie_id, title, description, poster_url, is_featured)
                VALUES (?, ?, ?, ?, ?)
            ''', (movie_id, title, description, poster_url, is_featured))
            conn.commit()
            conn.close()
            
            return jsonify({'message': 'Suggestion added successfully'})
        
    except Exception as e:
        logger.error(f"Admin suggestions error: {e}")
        return jsonify({'error': 'Admin operation failed'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has expired'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({'error': 'Invalid token'}), 401

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)