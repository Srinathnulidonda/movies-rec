#backend/app.py
import os
import logging
from datetime import datetime, timedelta
from functools import wraps
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from concurrent.futures import ThreadPoolExecutor
import json
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import threading

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# CORS configuration
CORS(app, origins=['https://movies-rec.vercel.app/', 'http://localhost:3000'])

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Caching configuration
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://localhost/moviedb')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# API Keys
TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-ml-service.onrender.com')

# Database connection pool
db_pool = None

def init_db_pool():
    global db_pool
    try:
        db_pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=20,
            dsn=DATABASE_URL
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")

def get_db():
    if db_pool is None:
        init_db_pool()
    return db_pool.getconn()

def return_db(conn):
    if db_pool:
        db_pool.putconn(conn)

# Database schema initialization
def create_tables():
    conn = get_db()
    cur = conn.cursor()
    
    try:
        # Users table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                preferences JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Content table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS content (
                id SERIAL PRIMARY KEY,
                external_id VARCHAR(255) NOT NULL,
                source VARCHAR(50) NOT NULL,
                title VARCHAR(500) NOT NULL,
                original_title VARCHAR(500),
                overview TEXT,
                genres JSONB,
                release_date DATE,
                runtime INTEGER,
                rating FLOAT,
                vote_count INTEGER,
                poster_path VARCHAR(500),
                backdrop_path VARCHAR(500),
                language VARCHAR(10),
                region VARCHAR(10),
                type VARCHAR(20) DEFAULT 'movie',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(external_id, source)
            )
        ''')
        
        # User interactions
        cur.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                content_id INTEGER REFERENCES content(id) ON DELETE CASCADE,
                interaction_type VARCHAR(50) NOT NULL,
                rating FLOAT,
                watched_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Admin recommendations
        cur.execute('''
            CREATE TABLE IF NOT EXISTS admin_recommendations (
                id SERIAL PRIMARY KEY,
                content_id INTEGER REFERENCES content(id) ON DELETE CASCADE,
                category VARCHAR(100) NOT NULL,
                priority INTEGER DEFAULT 1,
                description TEXT,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Content similarity matrix
        cur.execute('''
            CREATE TABLE IF NOT EXISTS content_similarity (
                id SERIAL PRIMARY KEY,
                content_id_1 INTEGER REFERENCES content(id) ON DELETE CASCADE,
                content_id_2 INTEGER REFERENCES content(id) ON DELETE CASCADE,
                similarity_score FLOAT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(content_id_1, content_id_2)
            )
        ''')
        
        # Indexes for performance
        cur.execute('CREATE INDEX IF NOT EXISTS idx_content_type ON content(type)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_content_region ON content(region)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_content_genres ON content USING GIN(genres)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id ON user_interactions(user_id)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_user_interactions_content_id ON user_interactions(content_id)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_admin_recommendations_category ON admin_recommendations(category)')
        
        conn.commit()
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        conn.rollback()
    finally:
        cur.close()
        return_db(conn)

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Token is invalid'}), 401
        
        return f(current_user_id, *args, **kwargs)
    
    return decorated

# Content fetching services
class ContentService:
    def __init__(self):
        self.tmdb_base_url = 'https://api.themoviedb.org/3'
        self.omdb_base_url = 'http://www.omdbapi.com'
        self.jikan_base_url = 'https://api.jikan.moe/v4'
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def fetch_tmdb_content(self, endpoint, params=None):
        if not TMDB_API_KEY:
            return None
        
        params = params or {}
        params['api_key'] = TMDB_API_KEY
        
        try:
            response = requests.get(f"{self.tmdb_base_url}/{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"TMDB API error: {e}")
            return None
    
    def fetch_omdb_content(self, params):
        if not OMDB_API_KEY:
            return None
        
        params['apikey'] = OMDB_API_KEY
        
        try:
            response = requests.get(self.omdb_base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"OMDB API error: {e}")
            return None
    
    def fetch_anime_content(self, endpoint):
        try:
            response = requests.get(f"{self.jikan_base_url}/{endpoint}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Jikan API error: {e}")
            return None
    
    def get_trending_content(self, media_type='movie', time_window='week'):
        return self.fetch_tmdb_content(f'trending/{media_type}/{time_window}')
    
    def get_popular_content(self, media_type='movie', region=None):
        params = {}
        if region:
            params['region'] = region
        return self.fetch_tmdb_content(f'{media_type}/popular', params)
    
    def get_content_details(self, content_id, media_type='movie'):
        return self.fetch_tmdb_content(f'{media_type}/{content_id}', {'append_to_response': 'credits,videos,similar'})
    
    def search_content(self, query, media_type='movie'):
        return self.fetch_tmdb_content(f'search/{media_type}', {'query': query})

content_service = ContentService()

# Recommendation engine
class RecommendationEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
    def get_content_based_recommendations(self, user_id, limit=20):
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Get user's favorite genres and rated content
            cur.execute('''
                SELECT c.*, ui.rating, ui.interaction_type
                FROM content c
                JOIN user_interactions ui ON c.id = ui.content_id
                WHERE ui.user_id = %s AND ui.rating >= 4
                ORDER BY ui.rating DESC, ui.created_at DESC
                LIMIT 50
            ''', (user_id,))
            
            liked_content = cur.fetchall()
            
            if not liked_content:
                return self.get_popular_recommendations(limit)
            
            # Extract genres and create user profile
            user_genres = defaultdict(float)
            for content in liked_content:
                if content['genres']:
                    for genre in content['genres']:
                        user_genres[genre['name']] += content['rating'] / 5.0
            
            # Find similar content
            cur.execute('''
                SELECT c.*, 
                       COUNT(CASE WHEN ui.rating >= 4 THEN 1 END) as positive_ratings,
                       AVG(ui.rating) as avg_rating
                FROM content c
                LEFT JOIN user_interactions ui ON c.id = ui.content_id
                WHERE c.id NOT IN (
                    SELECT content_id FROM user_interactions WHERE user_id = %s
                )
                GROUP BY c.id
                ORDER BY positive_ratings DESC, avg_rating DESC
                LIMIT 200
            ''', (user_id,))
            
            candidates = cur.fetchall()
            
            # Score based on genre similarity
            recommendations = []
            for content in candidates:
                score = 0
                if content['genres']:
                    for genre in content['genres']:
                        if genre['name'] in user_genres:
                            score += user_genres[genre['name']]
                
                if score > 0:
                    recommendations.append({
                        'content': dict(content),
                        'score': score,
                        'reason': 'Based on your favorite genres'
                    })
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return []
        finally:
            cur.close()
            return_db(conn)
    
    def get_collaborative_recommendations(self, user_id, limit=20):
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Find users with similar preferences
            cur.execute('''
                WITH user_ratings AS (
                    SELECT ui1.user_id, ui1.content_id, ui1.rating
                    FROM user_interactions ui1
                    WHERE ui1.rating IS NOT NULL
                ),
                similar_users AS (
                    SELECT ur2.user_id, 
                           COUNT(*) as common_items,
                           AVG(ABS(ur1.rating - ur2.rating)) as avg_diff
                    FROM user_ratings ur1
                    JOIN user_ratings ur2 ON ur1.content_id = ur2.content_id
                    WHERE ur1.user_id = %s AND ur2.user_id != %s
                    GROUP BY ur2.user_id
                    HAVING COUNT(*) >= 3
                    ORDER BY avg_diff ASC, common_items DESC
                    LIMIT 20
                )
                SELECT c.*, ui.rating, ui.user_id
                FROM content c
                JOIN user_interactions ui ON c.id = ui.content_id
                JOIN similar_users su ON ui.user_id = su.user_id
                WHERE ui.rating >= 4
                AND c.id NOT IN (
                    SELECT content_id FROM user_interactions WHERE user_id = %s
                )
                ORDER BY ui.rating DESC
                LIMIT %s
            ''', (user_id, user_id, user_id, limit))
            
            recommendations = cur.fetchall()
            
            return [{'content': dict(rec), 'score': rec['rating'], 'reason': 'Users with similar taste liked this'} 
                    for rec in recommendations]
            
        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return []
        finally:
            cur.close()
            return_db(conn)
    
    def get_popular_recommendations(self, limit=20, region=None):
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            where_clause = ""
            params = []
            
            if region:
                where_clause = "WHERE c.region = %s"
                params.append(region)
            
            cur.execute(f'''
                SELECT c.*, 
                       COUNT(ui.id) as interaction_count,
                       AVG(ui.rating) as avg_rating
                FROM content c
                LEFT JOIN user_interactions ui ON c.id = ui.content_id
                {where_clause}
                GROUP BY c.id
                ORDER BY interaction_count DESC, avg_rating DESC NULLS LAST
                LIMIT %s
            ''', params + [limit])
            
            recommendations = cur.fetchall()
            
            return [{'content': dict(rec), 'score': rec['avg_rating'] or 0, 'reason': 'Popular content'} 
                    for rec in recommendations]
            
        except Exception as e:
            logger.error(f"Error in popular recommendations: {e}")
            return []
        finally:
            cur.close()
            return_db(conn)

recommendation_engine = RecommendationEngine()

# API Routes
@app.route('/api/auth/register', methods=['POST'])
@limiter.limit("5 per minute")
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not all([username, email, password]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    conn = get_db()
    cur = conn.cursor()
    
    try:
        password_hash = generate_password_hash(password)
        cur.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES (%s, %s, %s)
            RETURNING id
        ''', (username, email, password_hash))
        
        user_id = cur.fetchone()[0]
        conn.commit()
        
        token = jwt.encode({
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(days=7)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user_id': user_id
        }), 201
        
    except psycopg2.IntegrityError:
        conn.rollback()
        return jsonify({'error': 'Username or email already exists'}), 409
    except Exception as e:
        conn.rollback()
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500
    finally:
        cur.close()
        return_db(conn)

@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not all([username, password]):
        return jsonify({'error': 'Missing credentials'}), 400
    
    conn = get_db()
    cur = conn.cursor()
    
    try:
        cur.execute('''
            SELECT id, password_hash FROM users
            WHERE username = %s OR email = %s
        ''', (username, username))
        
        user = cur.fetchone()
        
        if user and check_password_hash(user[1], password):
            token = jwt.encode({
                'user_id': user[0],
                'exp': datetime.utcnow() + timedelta(days=7)
            }, app.config['SECRET_KEY'], algorithm='HS256')
            
            return jsonify({
                'message': 'Login successful',
                'token': token,
                'user_id': user[0]
            }), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500
    finally:
        cur.close()
        return_db(conn)

@app.route('/api/recommendations/homepage', methods=['GET'])
@cache.cached(timeout=300)
def homepage_recommendations():
    """Get recommendations for non-logged users"""
    region = request.args.get('region', 'US')
    
    try:
        # Get trending content
        trending_movies = content_service.get_trending_content('movie', 'week')
        trending_tv = content_service.get_trending_content('tv', 'week')
        
        # Get popular content by region
        popular_movies = content_service.get_popular_content('movie', region)
        
        # Get popular content from database
        db_popular = recommendation_engine.get_popular_recommendations(20, region)
        
        response = {
            'trending_movies': trending_movies.get('results', [])[:10] if trending_movies else [],
            'trending_tv': trending_tv.get('results', [])[:10] if trending_tv else [],
            'popular_movies': popular_movies.get('results', [])[:10] if popular_movies else [],
            'whats_hot': db_popular[:10],
            'regional_content': db_popular[:20] if region in ['IN', 'US'] else []
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Homepage recommendations error: {e}")
        return jsonify({'error': 'Failed to fetch recommendations'}), 500

@app.route('/api/recommendations/personalized', methods=['GET'])
@token_required
def personalized_recommendations(current_user_id):
    """Get personalized recommendations for logged-in users"""
    try:
        # Get ML service recommendations
        ml_recommendations = []
        try:
            ml_response = requests.post(f"{ML_SERVICE_URL}/recommendations", 
                                       json={'user_id': current_user_id}, 
                                       timeout=5)
            if ml_response.status_code == 200:
                ml_recommendations = ml_response.json().get('recommendations', [])
        except Exception as e:
            logger.warning(f"ML service unavailable: {e}")
        
        # Get content-based recommendations
        content_based = recommendation_engine.get_content_based_recommendations(current_user_id, 15)
        
        # Get collaborative filtering recommendations
        collaborative = recommendation_engine.get_collaborative_recommendations(current_user_id, 15)
        
        # Get popular fallback
        popular = recommendation_engine.get_popular_recommendations(10)
        
        response = {
            'ml_recommendations': ml_recommendations,
            'content_based': content_based,
            'collaborative': collaborative,
            'popular': popular,
            'hybrid': content_based + collaborative
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Personalized recommendations error: {e}")
        return jsonify({'error': 'Failed to fetch personalized recommendations'}), 500

@app.route('/api/content/<int:content_id>/details', methods=['GET'])
def content_details(content_id):
    """Get detailed information about a specific content"""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get content from database
        cur.execute('''
            SELECT c.*, 
                   COUNT(ui.id) as total_interactions,
                   AVG(ui.rating) as avg_rating,
                   COUNT(CASE WHEN ui.interaction_type = 'favorite' THEN 1 END) as favorite_count
            FROM content c
            LEFT JOIN user_interactions ui ON c.id = ui.content_id
            WHERE c.id = %s
            GROUP BY c.id
        ''', (content_id,))
        
        content = cur.fetchone()
        
        if not content:
            return jsonify({'error': 'Content not found'}), 404
        
        # Get detailed info from TMDB
        tmdb_details = content_service.get_content_details(content['external_id'], content['type'])
        
        # Get similar content
        cur.execute('''
            SELECT c2.*, cs.similarity_score
            FROM content_similarity cs
            JOIN content c2 ON cs.content_id_2 = c2.id
            WHERE cs.content_id_1 = %s
            ORDER BY cs.similarity_score DESC
            LIMIT 10
        ''', (content_id,))
        
        similar_content = cur.fetchall()
        
        response = {
            'content': dict(content),
            'tmdb_details': tmdb_details,
            'similar_content': [dict(item) for item in similar_content]
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to fetch content details'}), 500
    finally:
        cur.close()
        return_db(conn)

@app.route('/api/search', methods=['GET'])
def search_content():
    """Search across multiple content sources"""
    query = request.args.get('q', '')
    media_type = request.args.get('type', 'movie')
    
    if not query:
        return jsonify({'error': 'Search query is required'}), 400
    
    try:
        # Search TMDB
        tmdb_results = content_service.search_content(query, media_type)
        
        # Search database
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute('''
            SELECT c.*, 
                   COUNT(ui.id) as interaction_count,
                   AVG(ui.rating) as avg_rating
            FROM content c
            LEFT JOIN user_interactions ui ON c.id = ui.content_id
            WHERE c.title ILIKE %s OR c.original_title ILIKE %s
            GROUP BY c.id
            ORDER BY interaction_count DESC, avg_rating DESC NULLS LAST
            LIMIT 20
        ''', (f'%{query}%', f'%{query}%'))
        
        db_results = cur.fetchall()
        
        response = {
            'tmdb_results': tmdb_results.get('results', []) if tmdb_results else [],
            'db_results': [dict(item) for item in db_results],
            'query': query
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500
    finally:
        cur.close()
        return_db(conn)

@app.route('/api/user/interactions', methods=['POST'])
@token_required
def user_interaction(current_user_id):
    """Record user interaction with content"""
    data = request.get_json()
    content_id = data.get('content_id')
    interaction_type = data.get('interaction_type')
    rating = data.get('rating')
    
    if not all([content_id, interaction_type]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    conn = get_db()
    cur = conn.cursor()
    
    try:
        # Check if interaction already exists
        cur.execute('''
            SELECT id FROM user_interactions
            WHERE user_id = %s AND content_id = %s AND interaction_type = %s
        ''', (current_user_id, content_id, interaction_type))
        
        existing = cur.fetchone()
        
        if existing:
            # Update existing interaction
            cur.execute('''
                UPDATE user_interactions
                SET rating = %s, watched_at = CURRENT_TIMESTAMP
                WHERE id = %s
            ''', (rating, existing[0]))
        else:
            # Create new interaction
            cur.execute('''
                INSERT INTO user_interactions (user_id, content_id, interaction_type, rating, watched_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            ''', (current_user_id, content_id, interaction_type, rating))
        
        conn.commit()
        
        return jsonify({'message': 'Interaction recorded successfully'}), 201
        
    except Exception as e:
        conn.rollback()
        logger.error(f"User interaction error: {e}")
        return jsonify({'error': 'Failed to record interaction'}), 500
    finally:
        cur.close()
        return_db(conn)

@app.route('/api/trending', methods=['GET'])
@cache.cached(timeout=600)
def trending_content():
    """Get trending content by region"""
    region = request.args.get('region', 'US')
    media_type = request.args.get('type', 'movie')
    
    try:
        # Get trending from TMDB
        tmdb_trending = content_service.get_trending_content(media_type, 'week')
        
        # Get trending from database
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute('''
            SELECT c.*, 
                   COUNT(ui.id) as recent_interactions,
                   AVG(ui.rating) as avg_rating
            FROM content c
            LEFT JOIN user_interactions ui ON c.id = ui.content_id
            WHERE ui.created_at >= CURRENT_DATE - INTERVAL '7 days'
            AND c.region = %s
            GROUP BY c.id
            ORDER BY recent_interactions DESC, avg_rating DESC NULLS LAST
            LIMIT 20
        ''', (region,))
        
        db_trending = cur.fetchall()
        
        response = {
            'tmdb_trending': tmdb_trending.get('results', []) if tmdb_trending else [],
            'db_trending': [dict(item) for item in db_trending],
            'region': region
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Trending content error: {e}")
        return jsonify({'error': 'Failed to fetch trending content'}), 500
    finally:
        cur.close()
        return_db(conn)

@app.route('/api/admin/curate', methods=['POST'])
@token_required
def admin_curate_content(current_user_id):
    """Admin endpoint to curate content recommendations"""
    # Add admin role check here
    data = request.get_json()
    content_id = data.get('content_id')
    category = data.get('category')
    priority = data.get('priority', 1)
    description = data.get('description', '')
    
    if not all([content_id, category]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    conn = get_db()
    cur = conn.cursor()
    
    try:
        cur.execute('''
            INSERT INTO admin_recommendations (content_id, category, priority, description)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (content_id, category) DO UPDATE SET
            priority = EXCLUDED.priority,
            description = EXCLUDED.description
        ''', (content_id, category, priority, description))
        
        conn.commit()
        
        return jsonify({'message': 'Content curated successfully'}), 201
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Admin curation error: {e}")
        return jsonify({'error': 'Failed to curate content'}), 500
    finally:
        cur.close()
        return_db(conn)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize database on startup
@app.before_first_request
def initialize_database():
    create_tables()

if __name__ == '__main__':
    # Initialize database pool
    init_db_pool()
    
    # Create tables
    create_tables()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)