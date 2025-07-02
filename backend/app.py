from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import sqlite3
import os
from datetime import datetime, timedelta
import jwt
from functools import wraps
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
CORS(app, origins=["*"])

# API Keys
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7689567537:AAGvDtu94OlLlTiWpfjSfpl_dd_Osi_2W7c')
TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID', '-1002566510721')
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'https://movies-rec-ml-service.onrender.com')

# Database setup
def init_db():
    conn = sqlite3.connect('movies.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, 
                  email TEXT UNIQUE, password TEXT, created_at TIMESTAMP)''')
    
    # Wishlist table
    c.execute('''CREATE TABLE IF NOT EXISTS wishlist
                 (id INTEGER PRIMARY KEY, user_id INTEGER, movie_id INTEGER,
                  title TEXT, poster_path TEXT, added_at TIMESTAMP)''')
    
    # Favorites table
    c.execute('''CREATE TABLE IF NOT EXISTS favorites
                 (id INTEGER PRIMARY KEY, user_id INTEGER, movie_id INTEGER,
                  title TEXT, poster_path TEXT, added_at TIMESTAMP)''')
    
    # Watch history table
    c.execute('''CREATE TABLE IF NOT EXISTS watch_history
                 (id INTEGER PRIMARY KEY, user_id INTEGER, movie_id INTEGER,
                  title TEXT, poster_path TEXT, watched_at TIMESTAMP)''')
    
    # Admin suggestions table
    c.execute('''CREATE TABLE IF NOT EXISTS admin_suggestions
                 (id INTEGER PRIMARY KEY, movie_id INTEGER, title TEXT,
                  description TEXT, poster_path TEXT, featured BOOLEAN,
                  created_at TIMESTAMP)''')
    
    # Admin users table
    c.execute('''CREATE TABLE IF NOT EXISTS admin_users
                 (id INTEGER PRIMARY KEY, username TEXT UNIQUE, 
                  password TEXT, created_at TIMESTAMP)''')
    
    conn.commit()
    conn.close()

init_db()

# JWT token required decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token missing'}), 401
        try:
            token = token.split(' ')[1]  # Remove 'Bearer '
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user_id = data['user_id']
        except:
            return jsonify({'message': 'Token invalid'}), 401
        return f(current_user_id, *args, **kwargs)
    return decorated

# TMDB API helper functions
def search_tmdb(query, media_type='multi'):
    url = f"https://api.themoviedb.org/3/search/{media_type}"
    params = {'api_key': TMDB_API_KEY, 'query': query}
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else {}

def get_tmdb_details(movie_id, media_type='movie'):
    url = f"https://api.themoviedb.org/3/{media_type}/{movie_id}"
    params = {'api_key': TMDB_API_KEY, 'append_to_response': 'videos,credits,watch/providers'}
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else {}

def get_trending():
    url = "https://api.themoviedb.org/3/trending/all/week"
    params = {'api_key': TMDB_API_KEY}
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else {}

# Routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not all([username, email, password]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    conn = sqlite3.connect('movies.db')
    c = conn.cursor()
    
    try:
        hashed_password = generate_password_hash(password)
        c.execute('INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)',
                  (username, email, hashed_password, datetime.now()))
        conn.commit()
        user_id = c.lastrowid
        
        token = jwt.encode({
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.config['SECRET_KEY'])
        
        return jsonify({'token': token, 'username': username}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 400
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    conn = sqlite3.connect('movies.db')
    c = conn.cursor()
    c.execute('SELECT id, username, password FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    
    if user and check_password_hash(user[2], password):
        token = jwt.encode({
            'user_id': user[0],
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.config['SECRET_KEY'])
        return jsonify({'token': token, 'username': user[1]}), 200
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    results = search_tmdb(query)
    return jsonify(results)

@app.route('/api/trending', methods=['GET'])
def trending():
    results = get_trending()
    return jsonify(results)

@app.route('/api/details/<int:movie_id>')
def movie_details(movie_id):
    media_type = request.args.get('type', 'movie')
    details = get_tmdb_details(movie_id, media_type)
    return jsonify(details)

@app.route('/api/recommendations')
@token_required
def get_recommendations(current_user_id):
    try:
        # Get user's watch history and favorites for ML service
        conn = sqlite3.connect('movies.db')
        c = conn.cursor()
        c.execute('SELECT movie_id FROM watch_history WHERE user_id = ? ORDER BY watched_at DESC LIMIT 50', (current_user_id,))
        watch_history = [row[0] for row in c.fetchall()]
        
        c.execute('SELECT movie_id FROM favorites WHERE user_id = ?', (current_user_id,))
        favorites = [row[0] for row in c.fetchall()]
        conn.close()
        
        # Call ML service
        ml_response = requests.post(f'{ML_SERVICE_URL}/api/recommend', 
                                   json={'user_id': current_user_id, 'watch_history': watch_history, 'favorites': favorites})
        
        if ml_response.status_code == 200:
            return jsonify(ml_response.json())
        else:
            # Fallback to trending if ML service fails
            return trending()
    except:
        return trending()

@app.route('/api/wishlist', methods=['GET', 'POST', 'DELETE'])
@token_required
def wishlist(current_user_id):
    conn = sqlite3.connect('movies.db')
    c = conn.cursor()
    
    if request.method == 'GET':
        c.execute('SELECT * FROM wishlist WHERE user_id = ? ORDER BY added_at DESC', (current_user_id,))
        items = [{'id': row[0], 'movie_id': row[2], 'title': row[3], 'poster_path': row[4], 'added_at': row[5]} 
                for row in c.fetchall()]
        conn.close()
        return jsonify(items)
    
    elif request.method == 'POST':
        data = request.get_json()
        c.execute('INSERT OR IGNORE INTO wishlist (user_id, movie_id, title, poster_path, added_at) VALUES (?, ?, ?, ?, ?)',
                  (current_user_id, data['movie_id'], data['title'], data['poster_path'], datetime.now()))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Added to wishlist'}), 201
    
    elif request.method == 'DELETE':
        movie_id = request.args.get('movie_id')
        c.execute('DELETE FROM wishlist WHERE user_id = ? AND movie_id = ?', (current_user_id, movie_id))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Removed from wishlist'}), 200

@app.route('/api/favorites', methods=['GET', 'POST', 'DELETE'])
@token_required
def favorites(current_user_id):
    conn = sqlite3.connect('movies.db')
    c = conn.cursor()
    
    if request.method == 'GET':
        c.execute('SELECT * FROM favorites WHERE user_id = ? ORDER BY added_at DESC', (current_user_id,))
        items = [{'id': row[0], 'movie_id': row[2], 'title': row[3], 'poster_path': row[4], 'added_at': row[5]} 
                for row in c.fetchall()]
        conn.close()
        return jsonify(items)
    
    elif request.method == 'POST':
        data = request.get_json()
        c.execute('INSERT OR IGNORE INTO favorites (user_id, movie_id, title, poster_path, added_at) VALUES (?, ?, ?, ?, ?)',
                  (current_user_id, data['movie_id'], data['title'], data['poster_path'], datetime.now()))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Added to favorites'}), 201
    
    elif request.method == 'DELETE':
        movie_id = request.args.get('movie_id')
        c.execute('DELETE FROM favorites WHERE user_id = ? AND movie_id = ?', (current_user_id, movie_id))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Removed from favorites'}), 200

@app.route('/api/watch-history', methods=['GET', 'POST'])
@token_required
def watch_history(current_user_id):
    conn = sqlite3.connect('movies.db')
    c = conn.cursor()
    
    if request.method == 'GET':
        c.execute('SELECT * FROM watch_history WHERE user_id = ? ORDER BY watched_at DESC LIMIT 100', (current_user_id,))
        items = [{'id': row[0], 'movie_id': row[2], 'title': row[3], 'poster_path': row[4], 'watched_at': row[5]} 
                for row in c.fetchall()]
        conn.close()
        return jsonify(items)
    
    elif request.method == 'POST':
        data = request.get_json()
        c.execute('INSERT INTO watch_history (user_id, movie_id, title, poster_path, watched_at) VALUES (?, ?, ?, ?, ?)',
                  (current_user_id, data['movie_id'], data['title'], data['poster_path'], datetime.now()))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Added to watch history'}), 201

# Admin routes
@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    conn = sqlite3.connect('movies.db')
    c = conn.cursor()
    c.execute('SELECT id, password FROM admin_users WHERE username = ?', (username,))
    admin = c.fetchone()
    conn.close()
    
    if admin and check_password_hash(admin[1], password):
        token = jwt.encode({
            'admin_id': admin[0],
            'exp': datetime.utcnow() + timedelta(hours=8)
        }, app.config['SECRET_KEY'])
        return jsonify({'token': token}), 200
    
    return jsonify({'error': 'Invalid admin credentials'}), 401

@app.route('/api/admin/suggestions', methods=['GET', 'POST'])
def admin_suggestions():
    if request.method == 'GET':
        conn = sqlite3.connect('movies.db')
        c = conn.cursor()
        c.execute('SELECT * FROM admin_suggestions ORDER BY created_at DESC')
        suggestions = [{'id': row[0], 'movie_id': row[1], 'title': row[2], 'description': row[3], 
                       'poster_path': row[4], 'featured': row[5], 'created_at': row[6]} 
                      for row in c.fetchall()]
        conn.close()
        return jsonify(suggestions)
    
    elif request.method == 'POST':
        data = request.get_json()
        conn = sqlite3.connect('movies.db')
        c = conn.cursor()
        c.execute('INSERT INTO admin_suggestions (movie_id, title, description, poster_path, featured, created_at) VALUES (?, ?, ?, ?, ?, ?)',
                  (data['movie_id'], data['title'], data['description'], data['poster_path'], data.get('featured', False), datetime.now()))
        conn.commit()
        
        # Post to Telegram if enabled
        if data.get('post_to_telegram', False):
            post_to_telegram(data['title'], data['description'], data['poster_path'])
        
        conn.close()
        return jsonify({'message': 'Suggestion added'}), 201

def post_to_telegram(title, description, poster_url):
    try:
        message = f"🎬 *{title}*\n\n{description}\n\n#MovieRecommendation"
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        
        data = {
            'chat_id': TELEGRAM_CHANNEL_ID,
            'photo': f"https://image.tmdb.org/t/p/w500{poster_url}",
            'caption': message,
            'parse_mode': 'Markdown'
        }
        
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Telegram post failed: {e}")

@app.route('/api/featured', methods=['GET'])
def get_featured():
    conn = sqlite3.connect('movies.db')
    c = conn.cursor()
    c.execute('SELECT * FROM admin_suggestions WHERE featured = 1 ORDER BY created_at DESC LIMIT 10')
    featured = [{'id': row[0], 'movie_id': row[1], 'title': row[2], 'description': row[3], 
                'poster_path': row[4], 'created_at': row[6]} 
               for row in c.fetchall()]
    conn.close()
    return jsonify(featured)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)