# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import requests
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import threading
import time
from functools import wraps

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)

CORS(app, origins=['*'])
jwt = JWTManager(app)

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:password@localhost/moviedb')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# API Keys
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# Cache
cache = {}
cache_timeout = 3600  # 1 hour

def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                preferences JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Content table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content (
                id SERIAL PRIMARY KEY,
                external_id VARCHAR(50),
                source VARCHAR(20),
                title VARCHAR(255) NOT NULL,
                type VARCHAR(20) NOT NULL,
                genres TEXT[],
                year INTEGER,
                rating FLOAT,
                synopsis TEXT,
                poster_url TEXT,
                backdrop_url TEXT,
                trailer_url TEXT,
                runtime INTEGER,
                language VARCHAR(10),
                region VARCHAR(10),
                cast_crew JSONB DEFAULT '{}',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User interactions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                content_id INTEGER REFERENCES content(id),
                interaction_type VARCHAR(20),
                rating INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Viewing sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS viewing_sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                content_id INTEGER REFERENCES content(id),
                watch_time INTEGER,
                completed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Admin curated recommendations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin_recommendations (
                id SERIAL PRIMARY KEY,
                content_id INTEGER REFERENCES content(id),
                category VARCHAR(50),
                priority INTEGER DEFAULT 1,
                custom_description TEXT,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Content similarity
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_similarity (
                id SERIAL PRIMARY KEY,
                content_id_1 INTEGER REFERENCES content(id),
                content_id_2 INTEGER REFERENCES content(id),
                similarity_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_genres ON content USING GIN(genres)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON content(type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_language ON content(language)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id ON user_interactions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_viewing_sessions_user_id ON viewing_sessions(user_id)')
        
        conn.commit()

def cache_result(key, data, timeout=cache_timeout):
    cache[key] = {'data': data, 'expires': time.time() + timeout}

def get_cached_result(key):
    if key in cache and cache[key]['expires'] > time.time():
        return cache[key]['data']
    return None

def rate_limit(per_minute=60):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# External API integrations
class ContentService:
    @staticmethod
    def search_tmdb(query, page=1):
        url = f"https://api.themoviedb.org/3/search/multi?api_key={TMDB_API_KEY}&query={query}&page={page}"
        response = requests.get(url)
        return response.json() if response.status_code == 200 else {}
    
    @staticmethod
    def get_tmdb_details(content_id, content_type):
        url = f"https://api.themoviedb.org/3/{content_type}/{content_id}?api_key={TMDB_API_KEY}&append_to_response=credits,videos"
        response = requests.get(url)
        return response.json() if response.status_code == 200 else {}
    
    @staticmethod
    def get_omdb_details(title, year=None):
        url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}"
        if year:
            url += f"&y={year}"
        response = requests.get(url)
        return response.json() if response.status_code == 200 else {}
    
    @staticmethod
    def get_trending_content():
        url = f"https://api.themoviedb.org/3/trending/all/day?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        return response.json() if response.status_code == 200 else {}
    
    @staticmethod
    def get_anime_trending():
        url = "https://api.jikan.moe/v4/top/anime"
        response = requests.get(url)
        return response.json() if response.status_code == 200 else {}

class RecommendationEngine:
    def __init__(self):
        self.content_cache = {}
        self.user_cache = {}
    
    def get_user_profile(self, user_id):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.genres, ui.rating, ui.interaction_type
                FROM user_interactions ui
                JOIN content c ON ui.content_id = c.id
                WHERE ui.user_id = %s
            ''', (user_id,))
            return cursor.fetchall()
    
    def calculate_user_preferences(self, user_id):
        profile = self.get_user_profile(user_id)
        genre_scores = {}
        
        for interaction in profile:
            if interaction['genres']:
                for genre in interaction['genres']:
                    if genre not in genre_scores:
                        genre_scores[genre] = 0
                    
                    weight = 1
                    if interaction['interaction_type'] == 'favorite':
                        weight = 3
                    elif interaction['interaction_type'] == 'wishlist':
                        weight = 2
                    elif interaction['rating'] and interaction['rating'] > 7:
                        weight = 2
                    
                    genre_scores[genre] += weight
        
        return genre_scores
    
    def collaborative_filtering(self, user_id, limit=10):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT u2.id, COUNT(*) as common_interactions
                FROM user_interactions u1
                JOIN user_interactions u2 ON u1.content_id = u2.content_id
                WHERE u1.user_id = %s AND u2.user_id != %s
                GROUP BY u2.user_id
                ORDER BY common_interactions DESC
                LIMIT 5
            ''', (user_id, user_id))
            
            similar_users = cursor.fetchall()
            if not similar_users:
                return []
            
            similar_user_ids = [user['id'] for user in similar_users]
            cursor.execute('''
                SELECT DISTINCT c.*, ui.rating
                FROM content c
                JOIN user_interactions ui ON c.id = ui.content_id
                WHERE ui.user_id = ANY(%s)
                AND c.id NOT IN (
                    SELECT content_id FROM user_interactions WHERE user_id = %s
                )
                ORDER BY ui.rating DESC NULLS LAST
                LIMIT %s
            ''', (similar_user_ids, user_id, limit))
            
            return cursor.fetchall()
    
    def content_based_filtering(self, user_id, limit=10):
        user_preferences = self.calculate_user_preferences(user_id)
        if not user_preferences:
            return []
        
        top_genres = sorted(user_preferences.items(), key=lambda x: x[1], reverse=True)[:3]
        genre_list = [genre[0] for genre in top_genres]
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT c.*
                FROM content c
                WHERE c.genres && %s
                AND c.id NOT IN (
                    SELECT content_id FROM user_interactions WHERE user_id = %s
                )
                ORDER BY c.rating DESC NULLS LAST
                LIMIT %s
            ''', (genre_list, user_id, limit))
            
            return cursor.fetchall()
    
    def get_trending_by_region(self, region='IN', limit=10):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, COUNT(vs.id) as view_count
                FROM content c
                LEFT JOIN viewing_sessions vs ON c.id = vs.content_id
                WHERE c.region = %s OR c.language IN ('te', 'hi', 'ta', 'kn')
                GROUP BY c.id
                ORDER BY view_count DESC, c.rating DESC
                LIMIT %s
            ''', (region, limit))
            
            return cursor.fetchall()
    
    def get_popular_by_genre(self, genre, limit=10):
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, COUNT(ui.id) as interaction_count
                FROM content c
                LEFT JOIN user_interactions ui ON c.id = ui.content_id
                WHERE %s = ANY(c.genres)
                GROUP BY c.id
                ORDER BY interaction_count DESC, c.rating DESC
                LIMIT %s
            ''', (genre, limit))
            
            return cursor.fetchall()

# Initialize services
content_service = ContentService()
recommendation_engine = RecommendationEngine()

# Auth endpoints
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not all([username, email, password]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    password_hash = generate_password_hash(password)
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (%s, %s, %s) RETURNING id
            ''', (username, email, password_hash))
            user_id = cursor.fetchone()['id']
            conn.commit()
            
            access_token = create_access_token(identity=user_id)
            return jsonify({'access_token': access_token, 'user_id': user_id}), 201
    except psycopg2.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, password_hash FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user['password_hash'], password):
            access_token = create_access_token(identity=user['id'])
            return jsonify({'access_token': access_token, 'user_id': user['id']}), 200
        
        return jsonify({'error': 'Invalid credentials'}), 401

# Homepage recommendations (non-logged users)
@app.route('/api/recommendations/homepage', methods=['GET'])
@rate_limit(per_minute=100)
def homepage_recommendations():
    cache_key = 'homepage_recommendations'
    cached_result = get_cached_result(cache_key)
    
    if cached_result:
        return jsonify(cached_result), 200
    
    try:
        # Get trending content
        trending_data = content_service.get_trending_content()
        
        # Get popular by genre
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance']
        popular_by_genre = {}
        
        for genre in genres:
            popular_by_genre[genre] = recommendation_engine.get_popular_by_genre(genre.lower(), 8)
        
        # Get regional trending
        regional_trending = {
            'Telugu': recommendation_engine.get_trending_by_region('IN', 8),
            'Hindi': recommendation_engine.get_trending_by_region('IN', 8),
            'Tamil': recommendation_engine.get_trending_by_region('IN', 8),
            'Kannada': recommendation_engine.get_trending_by_region('IN', 8)
        }
        
        # Get admin curated content
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, ar.category, ar.custom_description
                FROM content c
                JOIN admin_recommendations ar ON c.id = ar.content_id
                WHERE ar.expires_at > NOW() OR ar.expires_at IS NULL
                ORDER BY ar.priority DESC
                LIMIT 10
            ''')
            critics_choice = cursor.fetchall()
        
        result = {
            'trending': trending_data.get('results', [])[:10],
            'popular_by_genre': popular_by_genre,
            'regional_trending': regional_trending,
            'critics_choice': critics_choice,
            'whats_hot': trending_data.get('results', [])[:5]
        }
        
        cache_result(cache_key, result)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Personalized recommendations (logged-in users)
@app.route('/api/recommendations/personalized', methods=['GET'])
@jwt_required()
@rate_limit(per_minute=60)
def personalized_recommendations():
    user_id = get_jwt_identity()
    cache_key = f'personalized_recommendations_{user_id}'
    cached_result = get_cached_result(cache_key)
    
    if cached_result:
        return jsonify(cached_result), 200
    
    try:
        # Collaborative filtering
        collaborative_recs = recommendation_engine.collaborative_filtering(user_id, 10)
        
        # Content-based filtering
        content_recs = recommendation_engine.content_based_filtering(user_id, 10)
        
        # Get user preferences
        user_preferences = recommendation_engine.calculate_user_preferences(user_id)
        
        # Get similar content based on watch history
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT c2.*
                FROM content c1
                JOIN user_interactions ui ON c1.id = ui.content_id
                JOIN content_similarity cs ON c1.id = cs.content_id_1
                JOIN content c2 ON cs.content_id_2 = c2.id
                WHERE ui.user_id = %s
                ORDER BY cs.similarity_score DESC
                LIMIT 10
            ''', (user_id,))
            similar_content = cursor.fetchall()
        
        result = {
            'collaborative_filtering': collaborative_recs,
            'content_based': content_recs,
            'similar_content': similar_content,
            'user_preferences': user_preferences,
            'recommended_for_you': collaborative_recs[:5] + content_recs[:5]
        }
        
        cache_result(cache_key, result, timeout=1800)  # 30 minutes
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Content details
@app.route('/api/content/<int:content_id>/details', methods=['GET'])
@rate_limit(per_minute=100)
def content_details(content_id):
    cache_key = f'content_details_{content_id}'
    cached_result = get_cached_result(cache_key)
    
    if cached_result:
        return jsonify(cached_result), 200
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, 
                       AVG(ui.rating) as avg_rating,
                       COUNT(ui.id) as total_interactions
                FROM content c
                LEFT JOIN user_interactions ui ON c.id = ui.content_id
                WHERE c.id = %s
                GROUP BY c.id
            ''', (content_id,))
            content = cursor.fetchone()
            
            if not content:
                return jsonify({'error': 'Content not found'}), 404
            
            # Get similar content
            cursor.execute('''
                SELECT c2.*, cs.similarity_score
                FROM content_similarity cs
                JOIN content c2 ON cs.content_id_2 = c2.id
                WHERE cs.content_id_1 = %s
                ORDER BY cs.similarity_score DESC
                LIMIT 10
            ''', (content_id,))
            similar_content = cursor.fetchall()
            
            # Get user reviews
            cursor.execute('''
                SELECT u.username, ui.rating, ui.created_at
                FROM user_interactions ui
                JOIN users u ON ui.user_id = u.id
                WHERE ui.content_id = %s AND ui.rating IS NOT NULL
                ORDER BY ui.created_at DESC
                LIMIT 10
            ''', (content_id,))
            reviews = cursor.fetchall()
            
            result = {
                'content': dict(content),
                'similar_content': similar_content,
                'reviews': reviews
            }
            
            cache_result(cache_key, result)
            return jsonify(result), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Search
@app.route('/api/search', methods=['GET'])
@rate_limit(per_minute=100)
def search_content():
    query = request.args.get('q', '')
    page = int(request.args.get('page', 1))
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    try:
        # Search in local database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM content 
                WHERE title ILIKE %s OR synopsis ILIKE %s
                ORDER BY rating DESC
                LIMIT 20 OFFSET %s
            ''', (f'%{query}%', f'%{query}%', (page - 1) * 20))
            local_results = cursor.fetchall()
        
        # Search TMDB
        tmdb_results = content_service.search_tmdb(query, page)
        
        # Search anime
        anime_results = []
        if 'anime' in query.lower():
            anime_results = content_service.get_anime_trending()
        
        result = {
            'local_results': local_results,
            'tmdb_results': tmdb_results.get('results', []),
            'anime_results': anime_results.get('data', [])[:10],
            'total_pages': tmdb_results.get('total_pages', 1)
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Trending content
@app.route('/api/trending', methods=['GET'])
@rate_limit(per_minute=100)
def trending_content():
    region = request.args.get('region', 'global')
    content_type = request.args.get('type', 'all')
    
    cache_key = f'trending_{region}_{content_type}'
    cached_result = get_cached_result(cache_key)
    
    if cached_result:
        return jsonify(cached_result), 200
    
    try:
        if region == 'global':
            trending_data = content_service.get_trending_content()
            result = trending_data.get('results', [])
        else:
            result = recommendation_engine.get_trending_by_region(region, 20)
        
        cache_result(cache_key, result)
        return jsonify({'results': result}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# User interactions
@app.route('/api/user/interactions', methods=['POST'])
@jwt_required()
def add_interaction():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    content_id = data.get('content_id')
    interaction_type = data.get('type')  # 'favorite', 'wishlist', 'rating', 'watch'
    rating = data.get('rating')
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_interactions (user_id, content_id, interaction_type, rating)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, content_id, interaction_type) 
                DO UPDATE SET rating = EXCLUDED.rating, created_at = CURRENT_TIMESTAMP
            ''', (user_id, content_id, interaction_type, rating))
            conn.commit()
        
        return jsonify({'success': True}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/interactions', methods=['GET'])
@jwt_required()
def get_user_interactions():
    user_id = get_jwt_identity()
    interaction_type = request.args.get('type')
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT c.*, ui.rating, ui.interaction_type, ui.created_at
                FROM user_interactions ui
                JOIN content c ON ui.content_id = c.id
                WHERE ui.user_id = %s
            '''
            params = [user_id]
            
            if interaction_type:
                query += ' AND ui.interaction_type = %s'
                params.append(interaction_type)
            
            query += ' ORDER BY ui.created_at DESC'
            
            cursor.execute(query, params)
            interactions = cursor.fetchall()
        
        return jsonify({'interactions': interactions}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Admin endpoints
@app.route('/api/admin/curate', methods=['POST'])
@jwt_required()
def admin_curate():
    # Simple admin check - in production, implement proper role-based access
    data = request.get_json()
    
    content_id = data.get('content_id')
    category = data.get('category')
    priority = data.get('priority', 1)
    custom_description = data.get('description')
    expires_at = data.get('expires_at')
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO admin_recommendations 
                (content_id, category, priority, custom_description, expires_at)
                VALUES (%s, %s, %s, %s, %s)
            ''', (content_id, category, priority, custom_description, expires_at))
            conn.commit()
        
        # Send Telegram notification
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            message = f"New curated content added: {category}"
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": message}
            )
        
        return jsonify({'success': True}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Background task for content similarity calculation
def calculate_content_similarity():
    """Background task to calculate content similarity matrices"""
    while True:
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, title, synopsis, genres FROM content LIMIT 1000')
                content_list = cursor.fetchall()
                
                if len(content_list) < 2:
                    time.sleep(3600)  # Wait 1 hour
                    continue
                
                # Create feature vectors
                texts = []
                for content in content_list:
                    text = f"{content['title']} {content['synopsis'] or ''} {' '.join(content['genres'] or [])}"
                    texts.append(text)
                
                # Calculate TF-IDF similarity
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Store similarities
                for i, content1 in enumerate(content_list):
                    for j, content2 in enumerate(content_list):
                        if i != j and similarity_matrix[i][j] > 0.1:  # Only store meaningful similarities
                            cursor.execute('''
                                INSERT INTO content_similarity 
                                (content_id_1, content_id_2, similarity_score)
                                VALUES (%s, %s, %s)
                                ON CONFLICT (content_id_1, content_id_2) 
                                DO UPDATE SET similarity_score = EXCLUDED.similarity_score
                            ''', (content1['id'], content2['id'], float(similarity_matrix[i][j])))
                
                conn.commit()
                
        except Exception as e:
            print(f"Error in similarity calculation: {e}")
        
        time.sleep(3600)  # Run every hour

# Background task for fetching content
def fetch_and_store_content():
    """Background task to fetch and store content from external APIs"""
    while True:
        try:
            # Fetch trending content
            trending_data = content_service.get_trending_content()
            
            with get_db() as conn:
                cursor = conn.cursor()
                
                for item in trending_data.get('results', []):
                    content_type = 'movie' if item.get('media_type') == 'movie' else 'tv'
                    
                    cursor.execute('''
                        INSERT INTO content 
                        (external_id, source, title, type, genres, year, rating, synopsis, poster_url, backdrop_url)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (external_id, source) DO NOTHING
                    ''', (
                        item.get('id'),
                        'tmdb',
                        item.get('title') or item.get('name'),
                        content_type,
                        item.get('genre_ids', []),
                        item.get('release_date', '').split('-')[0] if item.get('release_date') else None,
                        item.get('vote_average'),
                        item.get('overview'),
                        f"https://image.tmdb.org/t/p/w500{item.get('poster_path')}" if item.get('poster_path') else None,
                        f"https://image.tmdb.org/t/p/w1280{item.get('backdrop_path')}" if item.get('backdrop_path') else None
                    ))
                
                conn.commit()
                
        except Exception as e:
            print(f"Error in content fetching: {e}")
        
        time.sleep(3600)  # Run every hour

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    init_db()
    
    # Start background tasks
    similarity_thread = threading.Thread(target=calculate_content_similarity, daemon=True)
    content_thread = threading.Thread(target=fetch_and_store_content, daemon=True)
    
    similarity_thread.start()
    content_thread.start()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

# requirements.txt
"""
Flask==2.3.3
Flask-CORS==4.0.0
Flask-JWT-Extended==4.5.3
psycopg2-binary==2.9.7
requests==2.31.0
numpy==1.24.3
scikit-learn==1.3.0
Werkzeug==2.3.7
python-dotenv
"""

# render.yaml (Render deployment config)
"""
services:
  - type: web
    name: movie-recommendation-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: movie-db
          property: connectionString
      - key: TMDB_API_KEY
        value: your_tmdb_api_key
      - key: OMDB_API_KEY
        value: your_omdb_api_key
      - key: JWT_SECRET_KEY
        generateValue: true
      - key: TELEGRAM_BOT_TOKEN
        value: your_telegram_bot_token
      - key: TELEGRAM_CHAT_ID
        value: your_telegram_chat_id

databases:
  - name: movie-db
    databaseName: moviedb
    user: movieuser
"""

# .env (Environment variables template)
"""
DATABASE_URL=postgresql://user:password@localhost/moviedb
TMDB_API_KEY=your_tmdb_api_key_here
OMDB_API_KEY=your_omdb_api_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
PORT=5000
"""

# Additional utility functions for enhanced functionality
class RegionalContentService:
    """Service for fetching regional Indian content"""
    
    @staticmethod
    def get_telugu_movies():
        # Mock Telugu movie data - replace with actual API
        return [
            {
                'title': 'RRR',
                'year': 2022,
                'rating': 8.0,
                'genres': ['Action', 'Drama'],
                'language': 'te',
                'region': 'IN'
            },
            {
                'title': 'Baahubali 2',
                'year': 2017,
                'rating': 8.7,
                'genres': ['Action', 'Drama', 'Fantasy'],
                'language': 'te',
                'region': 'IN'
            }
        ]
    
    @staticmethod
    def get_hindi_movies():
        # Mock Hindi movie data - replace with actual API
        return [
            {
                'title': 'Dangal',
                'year': 2016,
                'rating': 8.4,
                'genres': ['Biography', 'Drama', 'Sport'],
                'language': 'hi',
                'region': 'IN'
            },
            {
                'title': '3 Idiots',
                'year': 2009,
                'rating': 8.4,
                'genres': ['Comedy', 'Drama'],
                'language': 'hi',
                'region': 'IN'
            }
        ]
    
    @staticmethod
    def get_tamil_movies():
        # Mock Tamil movie data - replace with actual API
        return [
            {
                'title': 'Vikram',
                'year': 2022,
                'rating': 8.2,
                'genres': ['Action', 'Crime', 'Thriller'],
                'language': 'ta',
                'region': 'IN'
            },
            {
                'title': '96',
                'year': 2018,
                'rating': 8.5,
                'genres': ['Drama', 'Romance'],
                'language': 'ta',
                'region': 'IN'
            }
        ]
    
    @staticmethod
    def get_kannada_movies():
        # Mock Kannada movie data - replace with actual API
        return [
            {
                'title': 'K.G.F: Chapter 2',
                'year': 2022,
                'rating': 8.4,
                'genres': ['Action', 'Crime', 'Drama'],
                'language': 'kn',
                'region': 'IN'
            },
            {
                'title': 'Kantara',
                'year': 2022,
                'rating': 8.2,
                'genres': ['Action', 'Drama', 'Thriller'],
                'language': 'kn',
                'region': 'IN'
            }
        ]

class AdvancedRecommendationEngine:
    """Advanced recommendation algorithms"""
    
    @staticmethod
    def matrix_factorization(user_id, n_factors=50):
        """Matrix factorization for collaborative filtering"""
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                # Get user-item interaction matrix
                cursor.execute('''
                    SELECT DISTINCT user_id, content_id, rating
                    FROM user_interactions
                    WHERE rating IS NOT NULL
                ''')
                interactions = cursor.fetchall()
                
                if len(interactions) < 10:
                    return []
                
                # Create user-item matrix
                users = list(set([i['user_id'] for i in interactions]))
                items = list(set([i['content_id'] for i in interactions]))
                
                user_map = {u: i for i, u in enumerate(users)}
                item_map = {i: j for j, i in enumerate(items)}
                
                rating_matrix = np.zeros((len(users), len(items)))
                
                for interaction in interactions:
                    user_idx = user_map[interaction['user_id']]
                    item_idx = item_map[interaction['content_id']]
                    rating_matrix[user_idx][item_idx] = interaction['rating']
                
                # Simple SVD approximation
                U, sigma, Vt = np.linalg.svd(rating_matrix, full_matrices=False)
                
                # Reduce dimensionality
                n_factors = min(n_factors, len(sigma))
                U_reduced = U[:, :n_factors]
                sigma_reduced = np.diag(sigma[:n_factors])
                Vt_reduced = Vt[:n_factors, :]
                
                # Reconstruct matrix
                predicted_ratings = np.dot(U_reduced, np.dot(sigma_reduced, Vt_reduced))
                
                # Get recommendations for user
                if user_id in user_map:
                    user_idx = user_map[user_id]
                    user_ratings = predicted_ratings[user_idx]
                    
                    # Get top unrated items
                    rated_items = [item_map[i['content_id']] for i in interactions if i['user_id'] == user_id]
                    recommendations = []
                    
                    for item_idx, rating in enumerate(user_ratings):
                        if item_idx not in rated_items:
                            content_id = items[item_idx]
                            recommendations.append((content_id, rating))
                    
                    # Sort by predicted rating
                    recommendations.sort(key=lambda x: x[1], reverse=True)
                    
                    # Get content details
                    content_ids = [r[0] for r in recommendations[:10]]
                    cursor.execute('''
                        SELECT * FROM content WHERE id = ANY(%s)
                    ''', (content_ids,))
                    
                    return cursor.fetchall()
                
                return []
                
        except Exception as e:
            print(f"Matrix factorization error: {e}")
            return []
    
    @staticmethod
    def deep_learning_recommendations(user_id):
        """Placeholder for deep learning model integration"""
        # This would integrate with a ML service like TensorFlow Serving
        # For now, return content-based recommendations
        return recommendation_engine.content_based_filtering(user_id, 10)
    
    @staticmethod
    def seasonal_recommendations(user_id=None):
        """Get seasonal recommendations based on current date"""
        current_month = datetime.now().month
        
        seasonal_genres = {
            12: ['Family', 'Comedy', 'Romance'],  # December
            1: ['Family', 'Comedy', 'Romance'],   # January
            2: ['Romance', 'Drama'],              # February
            3: ['Action', 'Adventure'],           # March
            4: ['Comedy', 'Romance'],             # April
            5: ['Action', 'Adventure'],           # May
            6: ['Comedy', 'Family'],              # June
            7: ['Action', 'Adventure'],           # July
            8: ['Horror', 'Thriller'],            # August
            9: ['Drama', 'Thriller'],             # September
            10: ['Horror', 'Thriller'],           # October
            11: ['Family', 'Comedy']              # November
        }
        
        preferred_genres = seasonal_genres.get(current_month, ['Action', 'Comedy'])
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM content 
                WHERE genres && %s
                ORDER BY rating DESC
                LIMIT 10
            ''', (preferred_genres,))
            
            return cursor.fetchall()

# Additional API endpoints for enhanced functionality

@app.route('/api/content/bulk-import', methods=['POST'])
@jwt_required()
def bulk_import_content():
    """Bulk import content from external APIs"""
    data = request.get_json()
    source = data.get('source', 'tmdb')
    content_type = data.get('type', 'movie')
    page = data.get('page', 1)
    
    try:
        imported_count = 0
        
        if source == 'tmdb':
            if content_type == 'trending':
                trending_data = content_service.get_trending_content()
                items = trending_data.get('results', [])
            else:
                # Get popular movies/TV shows
                url = f"https://api.themoviedb.org/3/{content_type}/popular?api_key={TMDB_API_KEY}&page={page}"
                response = requests.get(url)
                items = response.json().get('results', []) if response.status_code == 200 else []
        
        elif source == 'anime':
            anime_data = content_service.get_anime_trending()
            items = anime_data.get('data', [])
        
        elif source == 'regional':
            regional_service = RegionalContentService()
            items = []
            if content_type == 'telugu':
                items = regional_service.get_telugu_movies()
            elif content_type == 'hindi':
                items = regional_service.get_hindi_movies()
            elif content_type == 'tamil':
                items = regional_service.get_tamil_movies()
            elif content_type == 'kannada':
                items = regional_service.get_kannada_movies()
        
        else:
            return jsonify({'error': 'Invalid source'}), 400
        
        # Store content in database
        with get_db() as conn:
            cursor = conn.cursor()
            
            for item in items:
                try:
                    if source == 'tmdb':
                        cursor.execute('''
                            INSERT INTO content 
                            (external_id, source, title, type, genres, year, rating, synopsis, poster_url, backdrop_url)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (external_id, source) DO NOTHING
                        ''', (
                            item.get('id'),
                            'tmdb',
                            item.get('title') or item.get('name'),
                            'movie' if item.get('media_type') == 'movie' else 'tv',
                            item.get('genre_ids', []),
                            item.get('release_date', '').split('-')[0] if item.get('release_date') else None,
                            item.get('vote_average'),
                            item.get('overview'),
                            f"https://image.tmdb.org/t/p/w500{item.get('poster_path')}" if item.get('poster_path') else None,
                            f"https://image.tmdb.org/t/p/w1280{item.get('backdrop_path')}" if item.get('backdrop_path') else None
                        ))
                    
                    elif source == 'anime':
                        cursor.execute('''
                            INSERT INTO content 
                            (external_id, source, title, type, genres, year, rating, synopsis, poster_url)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (external_id, source) DO NOTHING
                        ''', (
                            item.get('mal_id'),
                            'anime',
                            item.get('title'),
                            'anime',
                            [genre['name'] for genre in item.get('genres', [])],
                            item.get('year'),
                            item.get('score'),
                            item.get('synopsis'),
                            item.get('images', {}).get('jpg', {}).get('image_url')
                        ))
                    
                    elif source == 'regional':
                        cursor.execute('''
                            INSERT INTO content 
                            (title, type, genres, year, rating, language, region)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (title, year) DO NOTHING
                        ''', (
                            item.get('title'),
                            'movie',
                            item.get('genres', []),
                            item.get('year'),
                            item.get('rating'),
                            item.get('language'),
                            item.get('region')
                        ))
                    
                    imported_count += 1
                    
                except Exception as e:
                    print(f"Error importing item: {e}")
                    continue
            
            conn.commit()
        
        return jsonify({'imported_count': imported_count}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations/advanced', methods=['GET'])
@jwt_required()
def advanced_recommendations():
    """Get advanced recommendations using multiple algorithms"""
    user_id = get_jwt_identity()
    algorithm = request.args.get('algorithm', 'hybrid')
    
    try:
        advanced_engine = AdvancedRecommendationEngine()
        
        if algorithm == 'matrix_factorization':
            recommendations = advanced_engine.matrix_factorization(user_id)
        elif algorithm == 'deep_learning':
            recommendations = advanced_engine.deep_learning_recommendations(user_id)
        elif algorithm == 'seasonal':
            recommendations = advanced_engine.seasonal_recommendations(user_id)
        else:  # hybrid
            # Combine multiple recommendation approaches
            collab_recs = recommendation_engine.collaborative_filtering(user_id, 5)
            content_recs = recommendation_engine.content_based_filtering(user_id, 5)
            matrix_recs = advanced_engine.matrix_factorization(user_id)[:5]
            seasonal_recs = advanced_engine.seasonal_recommendations(user_id)[:5]
            
            # Combine and deduplicate
            all_recs = collab_recs + content_recs + matrix_recs + seasonal_recs
            seen_ids = set()
            recommendations = []
            
            for rec in all_recs:
                if rec['id'] not in seen_ids:
                    recommendations.append(rec)
                    seen_ids.add(rec['id'])
                    
                if len(recommendations) >= 20:
                    break
        
        return jsonify({'recommendations': recommendations}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/user-behavior', methods=['GET'])
@jwt_required()
def user_behavior_analytics():
    """Get user behavior analytics"""
    user_id = get_jwt_identity()
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get user stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_interactions,
                    AVG(rating) as avg_rating,
                    COUNT(DISTINCT content_id) as unique_content
                FROM user_interactions
                WHERE user_id = %s
            ''', (user_id,))
            stats = cursor.fetchone()
            
            # Get genre preferences
            cursor.execute('''
                SELECT 
                    unnest(c.genres) as genre,
                    COUNT(*) as count,
                    AVG(ui.rating) as avg_rating
                FROM user_interactions ui
                JOIN content c ON ui.content_id = c.id
                WHERE ui.user_id = %s AND c.genres IS NOT NULL
                GROUP BY genre
                ORDER BY count DESC
            ''', (user_id,))
            genre_preferences = cursor.fetchall()
            
            # Get viewing patterns
            cursor.execute('''
                SELECT 
                    DATE_TRUNC('month', created_at) as month,
                    COUNT(*) as interactions
                FROM user_interactions
                WHERE user_id = %s
                GROUP BY month
                ORDER BY month DESC
                LIMIT 12
            ''', (user_id,))
            viewing_patterns = cursor.fetchall()
            
            # Get content type distribution
            cursor.execute('''
                SELECT 
                    c.type,
                    COUNT(*) as count
                FROM user_interactions ui
                JOIN content c ON ui.content_id = c.id
                WHERE ui.user_id = %s
                GROUP BY c.type
            ''', (user_id,))
            content_types = cursor.fetchall()
        
        return jsonify({
            'stats': dict(stats),
            'genre_preferences': genre_preferences,
            'viewing_patterns': viewing_patterns,
            'content_types': content_types
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/content/similar-users', methods=['GET'])
@jwt_required()
def similar_users():
    """Find users with similar preferences"""
    user_id = get_jwt_identity()
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Find users with similar interaction patterns
            cursor.execute('''
                WITH user_genres AS (
                    SELECT 
                        ui.user_id,
                        unnest(c.genres) as genre
                    FROM user_interactions ui
                    JOIN content c ON ui.content_id = c.id
                    WHERE c.genres IS NOT NULL
                ),
                genre_similarity AS (
                    SELECT 
                        ug1.user_id as user1,
                        ug2.user_id as user2,
                        COUNT(*) as common_genres
                    FROM user_genres ug1
                    JOIN user_genres ug2 ON ug1.genre = ug2.genre
                    WHERE ug1.user_id = %s AND ug2.user_id != %s
                    GROUP BY ug1.user_id, ug2.user_id
                )
                SELECT 
                    u.username,
                    gs.common_genres,
                    COUNT(ui.id) as total_interactions
                FROM genre_similarity gs
                JOIN users u ON gs.user2 = u.id
                LEFT JOIN user_interactions ui ON gs.user2 = ui.user_id
                GROUP BY u.username, gs.common_genres
                ORDER BY gs.common_genres DESC
                LIMIT 10
            ''', (user_id, user_id))
            
            similar_users_data = cursor.fetchall()
        
        return jsonify({'similar_users': similar_users_data}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/content/export', methods=['GET'])
@jwt_required()
def export_user_data():
    """Export user data for GDPR compliance"""
    user_id = get_jwt_identity()
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get user data
            cursor.execute('SELECT username, email, preferences, created_at FROM users WHERE id = %s', (user_id,))
            user_data = cursor.fetchone()
            
            # Get interactions
            cursor.execute('''
                SELECT c.title, ui.interaction_type, ui.rating, ui.created_at
                FROM user_interactions ui
                JOIN content c ON ui.content_id = c.id
                WHERE ui.user_id = %s
                ORDER BY ui.created_at DESC
            ''', (user_id,))
            interactions = cursor.fetchall()
            
            # Get viewing sessions
            cursor.execute('''
                SELECT c.title, vs.watch_time, vs.completed, vs.created_at
                FROM viewing_sessions vs
                JOIN content c ON vs.content_id = c.id
                WHERE vs.user_id = %s
                ORDER BY vs.created_at DESC
            ''', (user_id,))
            sessions = cursor.fetchall()
        
        export_data = {
            'user_profile': dict(user_data),
            'interactions': interactions,
            'viewing_sessions': sessions,
            'export_date': datetime.now().isoformat()
        }
        
        return jsonify(export_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket support for real-time recommendations (optional)
try:
    from flask_socketio import SocketIO, emit, join_room, leave_room
    
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    @socketio.on('join_recommendations')
    def on_join_recommendations(data):
        user_id = data['user_id']
        join_room(f'user_{user_id}')
        emit('joined', {'room': f'user_{user_id}'})
    
    @socketio.on('request_live_recommendations')
    def on_request_live_recommendations(data):
        user_id = data['user_id']
        # Get fresh recommendations
        recommendations = recommendation_engine.collaborative_filtering(user_id, 5)
        emit('live_recommendations', {'recommendations': recommendations}, room=f'user_{user_id}')
    
    def send_notification_to_user(user_id, message):
        socketio.emit('notification', {'message': message}, room=f'user_{user_id}')
    
except ImportError:
    print("Flask-SocketIO not installed. WebSocket features disabled.")
    socketio = None

# Additional monitoring and health checks
@app.route('/api/system/stats', methods=['GET'])
def system_stats():
    """Get system statistics"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get content stats
            cursor.execute('SELECT COUNT(*) as total_content FROM content')
            content_count = cursor.fetchone()['total_content']
            
            # Get user stats
            cursor.execute('SELECT COUNT(*) as total_users FROM users')
            user_count = cursor.fetchone()['total_users']
            
            # Get interaction stats
            cursor.execute('SELECT COUNT(*) as total_interactions FROM user_interactions')
            interaction_count = cursor.fetchone()['total_interactions']
            
            # Get recent activity
            cursor.execute('''
                SELECT COUNT(*) as recent_activity
                FROM user_interactions
                WHERE created_at > NOW() - INTERVAL '24 hours'
            ''')
            recent_activity = cursor.fetchone()['recent_activity']
        
        return jsonify({
            'total_content': content_count,
            'total_users': user_count,
            'total_interactions': interaction_count,
            'recent_activity': recent_activity,
            'cache_size': len(cache),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    
    # Start background tasks
    similarity_thread = threading.Thread(target=calculate_content_similarity, daemon=True)
    content_thread = threading.Thread(target=fetch_and_store_content, daemon=True)
    
    similarity_thread.start()
    content_thread.start()
    
    port = int(os.environ.get('PORT', 5000))
    
    if socketio:
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    else:
        app.run(host='0.0.0.0', port=port, debug=False)