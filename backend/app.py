# backend/app.py
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, requests, json, os, hashlib
from datetime import datetime, timedelta
from functools import wraps
import logging
from collections import defaultdict
import asyncio
import aiohttp
from datetime import datetime, timedelta
from threading import Thread
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app, supports_credentials=True)

# API Configuration
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
JIKAN_BASE_URL = 'https://api.jikan.moe/v4'
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-ml-service.onrender.com')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# Database initialization
def init_db():
    conn = sqlite3.connect('recommendations.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_admin BOOLEAN DEFAULT 0
    )''')
    
    conn.execute('''CREATE TABLE IF NOT EXISTS content (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tmdb_id INTEGER,
        mal_id INTEGER,
        title TEXT NOT NULL,
        content_type TEXT NOT NULL,
        genre_ids TEXT,
        rating REAL,
        release_date TEXT,
        overview TEXT,
        poster_path TEXT,
        backdrop_path TEXT,
        runtime INTEGER,
        status TEXT,
        popularity REAL,
        vote_average REAL,
        vote_count INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(tmdb_id, content_type),
        UNIQUE(mal_id, content_type)
    )''')
    
    conn.execute('''CREATE TABLE IF NOT EXISTS user_interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        content_id INTEGER,
        interaction_type TEXT NOT NULL,
        rating REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (content_id) REFERENCES content (id),
        UNIQUE(user_id, content_id, interaction_type)
    )''')
    
    conn.execute('''CREATE TABLE IF NOT EXISTS featured_content (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content_id INTEGER,
        featured_by INTEGER,
        reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (content_id) REFERENCES content (id),
        FOREIGN KEY (featured_by) REFERENCES users (id)
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS user_preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        genre_weights TEXT,
        content_type_weights TEXT,
        viewing_time_patterns TEXT,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    
    conn.execute('''CREATE TABLE IF NOT EXISTS viewing_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        content_id INTEGER,
        watch_duration INTEGER,
        completion_rate REAL,
        session_start TIMESTAMP,
        session_end TIMESTAMP,
        device_type TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (content_id) REFERENCES content (id)
    )''')
    
    conn.execute('''CREATE TABLE IF NOT EXISTS public_recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content_id INTEGER,
        title TEXT,
        description TEXT,
        tags TEXT,
        priority INTEGER DEFAULT 0,
        created_by INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        is_active BOOLEAN DEFAULT 1,
        FOREIGN KEY (content_id) REFERENCES content (id),
        FOREIGN KEY (created_by) REFERENCES users (id)
    )''')

    conn.commit()
    conn.close()

# Utility functions
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        conn = sqlite3.connect('recommendations.db')
        user = conn.execute('SELECT is_admin FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        conn.close()
        if not user or not user[0]:
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

def get_db_connection():
    conn = sqlite3.connect('recommendations.db')
    conn.row_factory = sqlite3.Row
    return conn

def cache_content(content_data, content_type):
    """Cache content in database with conflict resolution"""
    conn = get_db_connection()
    try:
        if content_type == 'anime':
            conn.execute('''INSERT OR REPLACE INTO content 
                (mal_id, title, content_type, genre_ids, rating, release_date, overview, poster_path, popularity, vote_average, vote_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (content_data.get('mal_id'), content_data.get('title'), content_type,
                 json.dumps(content_data.get('genres', [])), content_data.get('score'),
                 content_data.get('aired', {}).get('from'), content_data.get('synopsis'),
                 content_data.get('images', {}).get('jpg', {}).get('large_image_url'),
                 content_data.get('popularity'), content_data.get('score'), content_data.get('scored_by')))
        else:
            conn.execute('''INSERT OR REPLACE INTO content 
                (tmdb_id, title, content_type, genre_ids, rating, release_date, overview, poster_path, backdrop_path, popularity, vote_average, vote_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (content_data.get('id'), content_data.get('title') or content_data.get('name'),
                 content_type, json.dumps(content_data.get('genre_ids', [])),
                 content_data.get('vote_average'), content_data.get('release_date') or content_data.get('first_air_date'),
                 content_data.get('overview'), content_data.get('poster_path'),
                 content_data.get('backdrop_path'), content_data.get('popularity'),
                 content_data.get('vote_average'), content_data.get('vote_count')))
        conn.commit()
        return conn.lastrowid
    except Exception as e:
        logger.error(f"Error caching content: {e}")
        return None
    finally:
        conn.close()

def get_ml_recommendations(user_id, limit=20):
    """Get recommendations from ML service with fallback"""
    try:
        response = requests.post(f"{ML_SERVICE_URL}/recommend", 
                               json={'user_id': user_id, 'limit': limit}, timeout=5)
        if response.status_code == 200:
            return response.json().get('recommendations', [])
    except Exception as e:
        logger.error(f"ML service error: {e}")
    
    # Fallback: Popular content
    conn = get_db_connection()
    popular = conn.execute('''SELECT * FROM content 
                             ORDER BY popularity DESC, vote_average DESC 
                             LIMIT ?''', (limit,)).fetchall()
    conn.close()
    return [dict(item) for item in popular]

# Authentication endpoints
@app.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not all(k in data for k in ['username', 'email', 'password']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                    (data['username'], data['email'], generate_password_hash(data['password'])))
        conn.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409
    finally:
        conn.close()

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not all(k in data for k in ['username', 'password']):
        return jsonify({'error': 'Missing username or password'}), 400
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (data['username'],)).fetchone()
    conn.close()
    
    if user and check_password_hash(user['password_hash'], data['password']):
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['is_admin'] = user['is_admin']
        return jsonify({'message': 'Login successful', 'user': {'id': user['id'], 'username': user['username'], 'is_admin': user['is_admin']}})
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/auth/logout', methods=['POST'])
@login_required
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'})

# Content discovery endpoints
@app.route('/search')
def search():
    query = request.args.get('q', '')
    content_type = request.args.get('type', 'all')
    page = int(request.args.get('page', 1))
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    results = {'movies': [], 'tv': [], 'anime': [], 'suggestions': []}
    
    try:
        # Get ML-powered search suggestions if user is logged in
        if 'user_id' in session:
            suggestions = get_ml_search_suggestions(session['user_id'], query)
            results['suggestions'] = suggestions
        
        # ... existing search logic ...
        
        # Enhanced caching with user context
        for search_type in (['movie', 'tv'] if content_type == 'all' else [content_type]):
            if search_type in ['movie', 'tv']:
                response = requests.get(f"{TMDB_BASE_URL}/search/{search_type}",
                                      params={'api_key': TMDB_API_KEY, 'query': query, 'page': page})
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('results', []):
                        # Enhanced caching with metadata
                        cache_content_enhanced(item, search_type)
                        results[search_type].append(item)
        
        # Anime search with enhanced metadata
        if content_type in ['all', 'anime']:
            response = requests.get(f"{JIKAN_BASE_URL}/anime", params={'q': query, 'page': page})
            if response.status_code == 200:
                data = response.json()
                for item in data.get('data', []):
                    cache_content_enhanced(item, 'anime')
                    results['anime'].append(item)
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search service unavailable'}), 503
    

def get_ml_search_suggestions(user_id, query, limit=5):
    """Get ML-powered search suggestions"""
    try:
        response = requests.post(f"{ML_SERVICE_URL}/search_suggestions", 
                               json={'user_id': user_id, 'query': query, 'limit': limit}, 
                               timeout=2)
        if response.status_code == 200:
            return response.json().get('suggestions', [])
    except Exception as e:
        logger.error(f"ML search suggestions error: {e}")
    return []
    

def cache_content_enhanced(content_data, content_type):
    """Enhanced content caching with metadata"""
    conn = get_db_connection()
    try:
        # Extract additional metadata
        popularity = content_data.get('popularity', 0)
        vote_average = content_data.get('vote_average', 0)
        vote_count = content_data.get('vote_count', 0)
        
        if content_type == 'anime':
            conn.execute('''INSERT OR REPLACE INTO content 
                (mal_id, title, content_type, genre_ids, rating, release_date, overview, 
                 poster_path, popularity, vote_average, vote_count, runtime, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (content_data.get('mal_id'), content_data.get('title'), content_type,
                 json.dumps([g.get('name') for g in content_data.get('genres', [])]),
                 content_data.get('score'), content_data.get('aired', {}).get('from'),
                 content_data.get('synopsis'), content_data.get('images', {}).get('jpg', {}).get('large_image_url'),
                 popularity, content_data.get('score'), content_data.get('scored_by'),
                 content_data.get('duration'), content_data.get('status')))
        else:
            conn.execute('''INSERT OR REPLACE INTO content 
                (tmdb_id, title, content_type, genre_ids, rating, release_date, overview, 
                 poster_path, backdrop_path, popularity, vote_average, vote_count, runtime, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (content_data.get('id'), content_data.get('title') or content_data.get('name'),
                 content_type, json.dumps(content_data.get('genre_ids', [])),
                 vote_average, content_data.get('release_date') or content_data.get('first_air_date'),
                 content_data.get('overview'), content_data.get('poster_path'),
                 content_data.get('backdrop_path'), popularity, vote_average, vote_count,
                 content_data.get('runtime'), content_data.get('status')))
        
        conn.commit()
        return conn.lastrowid
    except Exception as e:
        logger.error(f"Enhanced caching error: {e}")
        return None
    finally:
        conn.close()




@app.route('/public-recommendations')
def public_recommendations():
    limit = min(int(request.args.get('limit', 20)), 50)
    category = request.args.get('category', 'all')
    
    conn = get_db_connection()
    query = '''SELECT pr.*, c.title, c.poster_path, c.vote_average, c.content_type,
                      u.username as created_by_name
               FROM public_recommendations pr
               JOIN content c ON pr.content_id = c.id
               JOIN users u ON pr.created_by = u.id
               WHERE pr.is_active = 1 AND (pr.expires_at IS NULL OR pr.expires_at > datetime('now'))'''
    
    if category != 'all':
        query += f" AND c.content_type = '{category}'"
    
    query += ' ORDER BY pr.priority DESC, pr.created_at DESC LIMIT ?'
    
    recommendations = conn.execute(query, (limit,)).fetchall()
    conn.close()
    
    return jsonify({
        'public_recommendations': [dict(rec) for rec in recommendations],
        'category': category,
        'count': len(recommendations)
    })


@app.route('/admin/public-recommendation', methods=['POST'])
@admin_required
def create_public_recommendation():
    data = request.get_json()
    content_id = data.get('content_id')
    title = data.get('title')
    description = data.get('description')
    tags = data.get('tags', [])
    priority = data.get('priority', 0)
    expires_at = data.get('expires_at')
    
    if not all([content_id, title, description]):
        return jsonify({'error': 'Content ID, title, and description required'}), 400
    
    conn = get_db_connection()
    try:
        # Create public recommendation
        conn.execute('''INSERT INTO public_recommendations 
                       (content_id, title, description, tags, priority, created_by, expires_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (content_id, title, description, json.dumps(tags), priority, 
                     session['user_id'], expires_at))
        conn.commit()
        
        # Get content details for notifications
        content = conn.execute('SELECT title, poster_path FROM content WHERE id = ?', (content_id,)).fetchone()
        
        # Send to Telegram channel
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            send_telegram_recommendation(content, title, description, tags)
        
        # Trigger ML model update with new recommendation
        trigger_ml_update('public_recommendation_added', {
            'content_id': content_id,
            'admin_id': session['user_id']
        })
        
        return jsonify({'message': 'Public recommendation created successfully'})
    except Exception as e:
        logger.error(f"Public recommendation error: {e}")
        return jsonify({'error': 'Failed to create public recommendation'}), 500
    finally:
        conn.close()

def send_telegram_recommendation(content, title, description, tags):
    """Send recommendation to Telegram channel"""
    try:
        message = f"🎬 **New Recommendation: {title}**\n\n"
        message += f"📽️ **Content:** {content['title']}\n"
        message += f"📋 **Description:** {description}\n"
        if tags:
            message += f"🏷️ **Tags:** {', '.join(tags)}\n"
        message += f"\n🔗 Check it out now!"
        
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                     json={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'})
    except Exception as e:
        logger.error(f"Telegram send error: {e}")

def get_ml_recommendations(user_id, limit=20):
    """Enhanced ML recommendations with fallback strategies"""
    try:
        # Primary ML service
        response = requests.post(f"{ML_SERVICE_URL}/recommend", 
                               json={
                                   'user_id': user_id, 
                                   'limit': limit,
                                   'algorithm': 'hybrid_advanced',
                                   'real_time': True
                               }, timeout=5)
        if response.status_code == 200:
            return response.json().get('recommendations', [])
    except Exception as e:
        logger.error(f"ML service error: {e}")
    
    # Enhanced fallback with user history
    return get_enhanced_fallback_recommendations(user_id, limit)

def get_enhanced_fallback_recommendations(user_id, limit):
    """Enhanced fallback recommendations"""
    conn = get_db_connection()
    
    # Get user's preferred genres and content types
    user_prefs = conn.execute('''
        SELECT c.genre_ids, c.content_type, AVG(ui.rating) as avg_rating
        FROM content c
        JOIN user_interactions ui ON c.id = ui.content_id
        WHERE ui.user_id = ? AND ui.interaction_type = 'rating' AND ui.rating >= 7
        GROUP BY c.genre_ids, c.content_type
        ORDER BY avg_rating DESC
    ''', (user_id,)).fetchall()
    
    # Build preference weights
    genre_weights = defaultdict(float)
    content_type_weights = defaultdict(float)
    
    for pref in user_prefs:
        genres = json.loads(pref['genre_ids'] or '[]')
        for genre in genres:
            genre_weights[genre] += pref['avg_rating']
        content_type_weights[pref['content_type']] += pref['avg_rating']
    
    # Get recommendations based on preferences
    recommendations = conn.execute('''
        SELECT *, 
               (popularity * 0.3 + vote_average * 0.7) as score
        FROM content
        WHERE id NOT IN (
            SELECT content_id FROM user_interactions WHERE user_id = ?
        )
        ORDER BY score DESC, created_at DESC
        LIMIT ?
    ''', (user_id, limit * 2)).fetchall()
    
    # Score and rank recommendations
    scored_recs = []
    for rec in recommendations:
        score = rec['score']
        
        # Boost based on genre preferences
        genres = json.loads(rec['genre_ids'] or '[]')
        genre_boost = sum(genre_weights.get(str(g), 0) for g in genres) / len(genres) if genres else 0
        
        # Boost based on content type preferences
        content_type_boost = content_type_weights.get(rec['content_type'], 0)
        
        final_score = score + (genre_boost * 0.4) + (content_type_boost * 0.3)
        
        rec_dict = dict(rec)
        rec_dict['recommendation_score'] = final_score
        scored_recs.append(rec_dict)
    
    # Sort by final score and return top results
    scored_recs.sort(key=lambda x: x['recommendation_score'], reverse=True)
    
    conn.close()
    return scored_recs[:limit]

def track_content_view(user_id, content_id):
    """Track content view for real-time learning"""
    conn = get_db_connection()
    try:
        conn.execute('''INSERT INTO viewing_sessions 
                       (user_id, content_id, session_start, device_type)
                       VALUES (?, ?, datetime('now'), ?)''',
                    (user_id, content_id, request.headers.get('User-Agent', 'unknown')))
        conn.commit()
    except Exception as e:
        logger.error(f"View tracking error: {e}")
    finally:
        conn.close()

def trigger_ml_update(event_type, data):
    """Trigger ML model update for real-time learning"""
    try:
        requests.post(f"{ML_SERVICE_URL}/update", 
                     json={'event': event_type, 'data': data}, 
                     timeout=1)
    except Exception as e:
        logger.error(f"ML update trigger error: {e}")

# Background task for real-time learning
def start_background_learning():
    """Start background learning processes"""
    def learning_loop():
        while True:
            try:
                # Update user preferences every 5 minutes
                update_all_user_preferences()
                time.sleep(300)  # 5 minutes
            except Exception as e:
                logger.error(f"Background learning error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    thread = Thread(target=learning_loop, daemon=True)
    thread.start()

def update_all_user_preferences():
    """Update user preferences based on recent interactions"""
    conn = get_db_connection()
    try:
        # Get recent interactions
        recent_interactions = conn.execute('''
            SELECT user_id, content_id, interaction_type, rating, created_at
            FROM user_interactions
            WHERE created_at > datetime('now', '-1 hour')
        ''').fetchall()
        
        # Group by user and update preferences
        user_updates = defaultdict(list)
        for interaction in recent_interactions:
            user_updates[interaction['user_id']].append(interaction)
        
        for user_id, interactions in user_updates.items():
            update_user_preferences_realtime(user_id, interactions)
    
    except Exception as e:
        logger.error(f"User preferences update error: {e}")
    finally:
        conn.close()

def update_user_preferences_realtime(user_id, data):
    """Update user preferences in real-time"""
    # This would typically send data to ML service for real-time learning
    try:
        requests.post(f"{ML_SERVICE_URL}/update_preferences", 
                     json={'user_id': user_id, 'data': data}, 
                     timeout=2)
    except Exception as e:
        logger.error(f"Real-time preferences update error: {e}")

# Initialize background learning on startup
if __name__ == '__main__':
    init_db()
    start_background_learning()
    app.run(debug=True, host='0.0.0.0', port=5000)

# Real-time learning endpoint
@app.route('/learn', methods=['POST'])
@login_required
def real_time_learning():
    data = request.get_json()
    user_id = session['user_id']
    action = data.get('action')
    content_id = data.get('content_id')
    context = data.get('context', {})
    
    # Track user behavior for real-time learning
    learning_data = {
        'user_id': user_id,
        'action': action,
        'content_id': content_id,
        'timestamp': datetime.now().isoformat(),
        'context': context
    }
    
    # Send to ML service for real-time learning
    try:
        response = requests.post(f"{ML_SERVICE_URL}/learn", json=learning_data, timeout=2)
        if response.status_code == 200:
            return jsonify({'message': 'Learning data recorded'})
    except Exception as e:
        logger.error(f"Real-time learning error: {e}")
    
    return jsonify({'message': 'Learning data queued'})


@app.route('/trending')
def trending():
    content_type = request.args.get('type', 'all')
    time_window = request.args.get('time_window', 'week')
    
    try:
        results = {}
        
        if content_type in ['all', 'movie']:
            response = requests.get(f"{TMDB_BASE_URL}/trending/movie/{time_window}",
                                  params={'api_key': TMDB_API_KEY})
            if response.status_code == 200:
                movies = response.json().get('results', [])
                for movie in movies:
                    cache_content(movie, 'movie')
                results['movies'] = movies
        
        if content_type in ['all', 'tv']:
            response = requests.get(f"{TMDB_BASE_URL}/trending/tv/{time_window}",
                                  params={'api_key': TMDB_API_KEY})
            if response.status_code == 200:
                tv_shows = response.json().get('results', [])
                for show in tv_shows:
                    cache_content(show, 'tv')
                results['tv'] = tv_shows
        
        if content_type in ['all', 'anime']:
            response = requests.get(f"{JIKAN_BASE_URL}/top/anime")
            if response.status_code == 200:
                anime_list = response.json().get('data', [])[:20]
                for anime in anime_list:
                    cache_content(anime, 'anime')
                results['anime'] = anime_list
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Trending error: {e}")
        return jsonify({'error': 'Trending service unavailable'}), 503

@app.route('/recommendations')
@login_required
def recommendations():
    user_id = session['user_id']
    limit = min(int(request.args.get('limit', 20)), 50)
    algorithm = request.args.get('algorithm', 'hybrid')
    
    try:
        # Get recommendations using specified algorithm
        if algorithm == 'hybrid':
            recommendations = get_hybrid_recommendations(user_id, limit)
        elif algorithm == 'collaborative':
            recommendations = get_collaborative_recommendations(user_id, limit)
        elif algorithm == 'content_based':
            recommendations = get_content_based_recommendations(user_id, limit)
        elif algorithm == 'deep_learning':
            recommendations = get_deep_learning_recommendations(user_id, limit)
        else:
            recommendations = get_ml_recommendations(user_id, limit)
        
        # Real-time learning update
        update_user_preferences_realtime(user_id, recommendations)
        
        return jsonify({
            'recommendations': recommendations,
            'algorithm': algorithm,
            'personalization_score': calculate_personalization_score(user_id),
            'diversity_score': calculate_diversity_score(recommendations)
        })
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return jsonify({'error': 'Recommendations service unavailable'}), 503
    
def get_hybrid_recommendations(user_id, limit=20):
    """Hybrid recommendation algorithm combining multiple approaches"""
    # Get collaborative filtering recommendations
    collab_recs = get_collaborative_recommendations(user_id, limit // 2)
    
    # Get content-based recommendations
    content_recs = get_content_based_recommendations(user_id, limit // 2)
    
    # Combine and deduplicate
    all_recs = collab_recs + content_recs
    seen_ids = set()
    unique_recs = []
    
    for rec in all_recs:
        if rec['id'] not in seen_ids:
            seen_ids.add(rec['id'])
            unique_recs.append(rec)
    
    return unique_recs[:limit]

def get_collaborative_recommendations(user_id, limit=20):
    """Collaborative filtering recommendations"""
    conn = get_db_connection()
    try:
        # Find users with similar preferences
        similar_users = conn.execute('''
            SELECT ui2.user_id, COUNT(*) as common_ratings,
                   AVG(ABS(ui1.rating - ui2.rating)) as rating_diff
            FROM user_interactions ui1
            JOIN user_interactions ui2 ON ui1.content_id = ui2.content_id
            WHERE ui1.user_id = ? AND ui2.user_id != ? 
            AND ui1.interaction_type = 'rating' AND ui2.interaction_type = 'rating'
            GROUP BY ui2.user_id
            HAVING common_ratings >= 3
            ORDER BY rating_diff ASC, common_ratings DESC
            LIMIT 10
        ''', (user_id, user_id)).fetchall()
        
        if not similar_users:
            return get_enhanced_fallback_recommendations(user_id, limit)
        
        # Get recommendations from similar users
        similar_user_ids = [str(u['user_id']) for u in similar_users]
        placeholders = ','.join(['?' for _ in similar_user_ids])
        
        recommendations = conn.execute(f'''
            SELECT c.*, AVG(ui.rating) as avg_rating, COUNT(*) as rating_count
            FROM content c
            JOIN user_interactions ui ON c.id = ui.content_id
            WHERE ui.user_id IN ({placeholders})
            AND ui.interaction_type = 'rating' AND ui.rating >= 7
            AND c.id NOT IN (
                SELECT content_id FROM user_interactions WHERE user_id = ?
            )
            GROUP BY c.id
            ORDER BY avg_rating DESC, rating_count DESC
            LIMIT ?
        ''', similar_user_ids + [user_id, limit]).fetchall()
        
        return [dict(rec) for rec in recommendations]
    
    except Exception as e:
        logger.error(f"Collaborative filtering error: {e}")
        return get_enhanced_fallback_recommendations(user_id, limit)
    finally:
        conn.close()

def get_content_based_recommendations(user_id, limit=20):
    """Content-based recommendations using user's content preferences"""
    conn = get_db_connection()
    try:
        # Get user's highly rated content
        user_content = conn.execute('''
            SELECT c.genre_ids, c.content_type, ui.rating
            FROM content c
            JOIN user_interactions ui ON c.id = ui.content_id
            WHERE ui.user_id = ? AND ui.interaction_type = 'rating' AND ui.rating >= 7
        ''', (user_id,)).fetchall()
        
        if not user_content:
            return get_enhanced_fallback_recommendations(user_id, limit)
        
        # Analyze preferences
        genre_scores = defaultdict(float)
        content_type_scores = defaultdict(float)
        
        for content in user_content:
            genres = json.loads(content['genre_ids'] or '[]')
            weight = content['rating'] / 10.0
            
            for genre in genres:
                genre_scores[genre] += weight
            content_type_scores[content['content_type']] += weight
        
        # Find similar content
        recommendations = conn.execute('''
            SELECT c.*, c.vote_average * c.popularity as content_score
            FROM content c
            WHERE c.id NOT IN (
                SELECT content_id FROM user_interactions WHERE user_id = ?
            )
            AND c.vote_average >= 6.0
            ORDER BY content_score DESC
            LIMIT ?
        ''', (user_id, limit * 3)).fetchall()
        
        # Score recommendations based on user preferences
        scored_recs = []
        for rec in recommendations:
            score = rec['content_score']
            
            # Boost score based on genre preferences
            genres = json.loads(rec['genre_ids'] or '[]')
            genre_boost = sum(genre_scores.get(str(g), 0) for g in genres)
            
            # Boost score based on content type preferences
            content_type_boost = content_type_scores.get(rec['content_type'], 0)
            
            final_score = score + (genre_boost * 50) + (content_type_boost * 30)
            
            rec_dict = dict(rec)
            rec_dict['recommendation_score'] = final_score
            scored_recs.append(rec_dict)
        
        # Sort by score and return top results
        scored_recs.sort(key=lambda x: x['recommendation_score'], reverse=True)
        return scored_recs[:limit]
    
    except Exception as e:
        logger.error(f"Content-based recommendations error: {e}")
        return get_enhanced_fallback_recommendations(user_id, limit)
    finally:
        conn.close()

def get_deep_learning_recommendations(user_id, limit=20):
    """Deep learning recommendations (placeholder for ML service)"""
    try:
        response = requests.post(f"{ML_SERVICE_URL}/deep_recommend", 
                               json={'user_id': user_id, 'limit': limit}, 
                               timeout=10)
        if response.status_code == 200:
            return response.json().get('recommendations', [])
    except Exception as e:
        logger.error(f"Deep learning recommendations error: {e}")
    
    # Fallback to hybrid approach
    return get_hybrid_recommendations(user_id, limit)

def get_ml_similar_content(content_id, user_id=None):
    """Get similar content using ML service"""
    try:
        payload = {'content_id': content_id}
        if user_id:
            payload['user_id'] = user_id
        
        response = requests.post(f"{ML_SERVICE_URL}/similar", 
                               json=payload, timeout=5)
        if response.status_code == 200:
            return response.json().get('similar', [])
    except Exception as e:
        logger.error(f"ML similar content error: {e}")
    
    # Fallback: Simple genre-based similarity
    return get_genre_based_similar_content(content_id)

def get_genre_based_similar_content(content_id, limit=10):
    """Fallback genre-based similarity"""
    conn = get_db_connection()
    try:
        # Get the content's genres
        content = conn.execute('SELECT genre_ids, content_type FROM content WHERE id = ?', 
                              (content_id,)).fetchone()
        if not content:
            return []
        
        genres = json.loads(content['genre_ids'] or '[]')
        if not genres:
            return []
        
        # Find similar content with overlapping genres
        placeholders = ','.join(['?' for _ in genres])
        similar = conn.execute(f'''
            SELECT c.*, 
                   (SELECT COUNT(*) FROM json_each(c.genre_ids) 
                    WHERE value IN ({placeholders})) as genre_overlap
            FROM content c
            WHERE c.id != ? AND c.content_type = ?
            AND genre_overlap > 0
            ORDER BY genre_overlap DESC, c.vote_average DESC
            LIMIT ?
        ''', genres + [content_id, content['content_type'], limit]).fetchall()
        
        return [dict(item) for item in similar]
    
    except Exception as e:
        logger.error(f"Genre-based similarity error: {e}")
        return []
    finally:
        conn.close()

def calculate_personalization_score(user_id):
    """Calculate how personalized the recommendations are"""
    conn = get_db_connection()
    try:
        # Get user's interaction count
        interaction_count = conn.execute('''
            SELECT COUNT(*) as count FROM user_interactions 
            WHERE user_id = ?
        ''', (user_id,)).fetchone()['count']
        
        # Calculate personalization score (0-1)
        if interaction_count < 5:
            return 0.2
        elif interaction_count < 20:
            return 0.5
        elif interaction_count < 50:
            return 0.8
        else:
            return 1.0
    
    except Exception as e:
        logger.error(f"Personalization score error: {e}")
        return 0.5
    finally:
        conn.close()

def calculate_diversity_score(recommendations):
    """Calculate diversity score of recommendations"""
    if not recommendations:
        return 0.0
    
    # Count unique genres and content types
    genres = set()
    content_types = set()
    
    for rec in recommendations:
        content_types.add(rec.get('content_type', ''))
        rec_genres = json.loads(rec.get('genre_ids', '[]'))
        genres.update(rec_genres)
    
    # Simple diversity calculation
    genre_diversity = min(len(genres) / 10.0, 1.0)  # Normalize to 0-1
    content_type_diversity = min(len(content_types) / 3.0, 1.0)  # 3 main types
    
    return (genre_diversity + content_type_diversity) / 2.0


# User interaction endpoints
@app.route('/watchlist', methods=['GET', 'POST', 'DELETE'])
@login_required
def watchlist():
    user_id = session['user_id']
    
    if request.method == 'GET':
        conn = get_db_connection()
        items = conn.execute('''SELECT c.* FROM content c
                              JOIN user_interactions ui ON c.id = ui.content_id
                              WHERE ui.user_id = ? AND ui.interaction_type = "watchlist"
                              ORDER BY ui.created_at DESC''', (user_id,)).fetchall()
        conn.close()
        return jsonify({'watchlist': [dict(item) for item in items]})
    
    elif request.method == 'POST':
        data = request.get_json()
        content_id = data.get('content_id')
        if not content_id:
            return jsonify({'error': 'Content ID required'}), 400
        
        conn = get_db_connection()
        try:
            conn.execute('''INSERT OR REPLACE INTO user_interactions 
                           (user_id, content_id, interaction_type, updated_at)
                           VALUES (?, ?, "watchlist", CURRENT_TIMESTAMP)''',
                        (user_id, content_id))
            conn.commit()
            return jsonify({'message': 'Added to watchlist'})
        except Exception as e:
            logger.error(f"Watchlist add error: {e}")
            return jsonify({'error': 'Failed to add to watchlist'}), 500
        finally:
            conn.close()
    
    elif request.method == 'DELETE':
        content_id = request.args.get('content_id')
        if not content_id:
            return jsonify({'error': 'Content ID required'}), 400
        
        conn = get_db_connection()
        conn.execute('DELETE FROM user_interactions WHERE user_id = ? AND content_id = ? AND interaction_type = "watchlist"',
                    (user_id, content_id))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Removed from watchlist'})

@app.route('/rate', methods=['POST'])
@login_required
def rate_content():
    data = request.get_json()
    content_id, rating = data.get('content_id'), data.get('rating')
    
    if not content_id or not (1 <= rating <= 10):
        return jsonify({'error': 'Valid content ID and rating (1-10) required'}), 400
    
    conn = get_db_connection()
    try:
        conn.execute('''INSERT OR REPLACE INTO user_interactions 
                       (user_id, content_id, interaction_type, rating, updated_at)
                       VALUES (?, ?, "rating", ?, CURRENT_TIMESTAMP)''',
                    (session['user_id'], content_id, rating))
        conn.commit()
        return jsonify({'message': 'Rating saved'})
    except Exception as e:
        logger.error(f"Rating error: {e}")
        return jsonify({'error': 'Failed to save rating'}), 500
    finally:
        conn.close()

# Admin endpoints
@app.route('/admin/feature', methods=['POST'])
@admin_required
def feature_content():
    data = request.get_json()
    content_id, reason = data.get('content_id'), data.get('reason', '')
    
    if not content_id:
        return jsonify({'error': 'Content ID required'}), 400
    
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO featured_content (content_id, featured_by, reason) VALUES (?, ?, ?)',
                    (content_id, session['user_id'], reason))
        conn.commit()
        
        # Send to Telegram if configured
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            content = conn.execute('SELECT title FROM content WHERE id = ?', (content_id,)).fetchone()
            if content:
                message = f"🎬 Featured: {content['title']}\n{reason}"
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                            json={'chat_id': TELEGRAM_CHAT_ID, 'text': message})
        
        return jsonify({'message': 'Content featured successfully'})
    except Exception as e:
        logger.error(f"Feature error: {e}")
        return jsonify({'error': 'Failed to feature content'}), 500
    finally:
        conn.close()

@app.route('/featured')
def get_featured():
    conn = get_db_connection()
    featured = conn.execute('''SELECT c.*, fc.reason, fc.created_at as featured_at, u.username as featured_by
                             FROM content c
                             JOIN featured_content fc ON c.id = fc.content_id
                             JOIN users u ON fc.featured_by = u.id
                             ORDER BY fc.created_at DESC
                             LIMIT 20''').fetchall()
    conn.close()
    return jsonify({'featured': [dict(item) for item in featured]})

# Content details
@app.route('/content/<int:content_id>')
def get_content_details(content_id):
    conn = get_db_connection()
    content = conn.execute('SELECT * FROM content WHERE id = ?', (content_id,)).fetchone()
    
    if not content:
        return jsonify({'error': 'Content not found'}), 404
    
    # Get detailed metadata from external APIs
    detailed_info = get_detailed_content_info(content)
    
    # Get user interactions if logged in
    user_interactions = {}
    similar_content = []
    
    if 'user_id' in session:
        interactions = conn.execute('''SELECT interaction_type, rating FROM user_interactions 
                                     WHERE user_id = ? AND content_id = ?''',
                                  (session['user_id'], content_id)).fetchall()
        user_interactions = {i['interaction_type']: i['rating'] for i in interactions}
        
        # Get ML-powered similar content
        similar_content = get_ml_similar_content(content_id, session['user_id'])
        
        # Track viewing session
        track_content_view(session['user_id'], content_id)
    
    conn.close()
    
    result = dict(content)
    result.update(detailed_info)
    result['user_interactions'] = user_interactions
    result['similar_content'] = similar_content
    
    return jsonify(result)

# Helper functions
def get_detailed_content_info(content):
    """Get detailed content information from external APIs"""
    detailed_info = {
        'cast': [],
        'crew': [],
        'videos': [],
        'reviews': [],
        'keywords': [],
        'similar': []
    }
    
    try:
        content_type = content['content_type']
        if content_type in ['movie', 'tv']:
            tmdb_id = content['tmdb_id']
            if tmdb_id:
                # Get cast and crew
                credits_response = requests.get(f"{TMDB_BASE_URL}/{content_type}/{tmdb_id}/credits",
                                              params={'api_key': TMDB_API_KEY})
                if credits_response.status_code == 200:
                    credits = credits_response.json()
                    detailed_info['cast'] = credits.get('cast', [])[:10]
                    detailed_info['crew'] = credits.get('crew', [])[:10]
                
                # Get videos (trailers, teasers)
                videos_response = requests.get(f"{TMDB_BASE_URL}/{content_type}/{tmdb_id}/videos",
                                             params={'api_key': TMDB_API_KEY})
                if videos_response.status_code == 200:
                    videos = videos_response.json()
                    detailed_info['videos'] = videos.get('results', [])[:5]
                
                # Get keywords
                keywords_response = requests.get(f"{TMDB_BASE_URL}/{content_type}/{tmdb_id}/keywords",
                                               params={'api_key': TMDB_API_KEY})
                if keywords_response.status_code == 200:
                    keywords = keywords_response.json()
                    detailed_info['keywords'] = keywords.get('keywords' if content_type == 'movie' else 'results', [])
        
        elif content_type == 'anime':
            mal_id = content['mal_id']
            if mal_id:
                # Get anime details
                anime_response = requests.get(f"{JIKAN_BASE_URL}/anime/{mal_id}/full")
                if anime_response.status_code == 200:
                    anime_data = anime_response.json().get('data', {})
                    detailed_info['cast'] = anime_data.get('characters', [])[:10]
                    detailed_info['crew'] = anime_data.get('staff', [])[:10]
                    detailed_info['videos'] = anime_data.get('trailer', {})
                    detailed_info['genres'] = anime_data.get('genres', [])
    
    except Exception as e:
        logger.error(f"Detailed content info error: {e}")
    
    return detailed_info


# Health check
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)