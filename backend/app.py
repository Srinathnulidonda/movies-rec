from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from datetime import datetime
import sqlite3

app = Flask(__name__)
CORS(app)

# Configuration
TMDB_API_KEY = os.getenv('TMDB_API_KEY', 'your_tmdb_api_key')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your_telegram_token')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_chat_id')
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'http://127.0.0.1:5001/')

# Database setup
def init_db():
    conn = sqlite3.connect('movies.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS wishlist (id INTEGER PRIMARY KEY, movie_id INTEGER, title TEXT, poster TEXT, added_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS favorites (id INTEGER PRIMARY KEY, movie_id INTEGER, title TEXT, poster TEXT, added_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS watch_history (id INTEGER PRIMARY KEY, movie_id INTEGER, title TEXT, poster TEXT, watched_at TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/api/search')
def search():
    query = request.args.get('q', '')
    response = requests.get(f'https://api.themoviedb.org/3/search/multi?api_key={TMDB_API_KEY}&query={query}')
    data = response.json()
    
    results = []
    for item in data.get('results', [])[:20]:
        results.append({
            'id': item.get('id'),
            'title': item.get('title') or item.get('name'),
            'overview': item.get('overview'),
            'poster': f"https://image.tmdb.org/t/p/w500{item.get('poster_path')}" if item.get('poster_path') else None,
            'type': item.get('media_type')
        })
    
    return jsonify({'results': results})

@app.route('/api/recommendations')
def get_recommendations():
    try:
        # Get user's favorites and wishlist for ML service
        conn = sqlite3.connect('movies.db')
        c = conn.cursor()
        c.execute('SELECT movie_id FROM favorites LIMIT 10')
        favorites = [row[0] for row in c.fetchall()]
        c.execute('SELECT movie_id FROM wishlist LIMIT 10')
        wishlist = [row[0] for row in c.fetchall()]
        conn.close()
        
        # Call ML service
        ml_response = requests.post(f'{ML_SERVICE_URL}/recommend', 
                                  json={'favorites': favorites, 'wishlist': wishlist})
        
        if ml_response.status_code == 200:
            recommendations = ml_response.json().get('recommendations', [])
        else:
            # Fallback to popular movies
            response = requests.get(f'https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}')
            data = response.json()
            recommendations = []
            for item in data.get('results', [])[:10]:
                recommendations.append({
                    'id': item.get('id'),
                    'title': item.get('title'),
                    'overview': item.get('overview'),
                    'poster': f"https://image.tmdb.org/t/p/w500{item.get('poster_path')}" if item.get('poster_path') else None
                })
        
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/wishlist', methods=['GET', 'POST'])
def wishlist():
    if request.method == 'POST':
        data = request.json
        movie_id = data.get('movie_id')
        
        # Get movie details from TMDB
        response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}')
        movie_data = response.json()
        
        conn = sqlite3.connect('movies.db')
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO wishlist (movie_id, title, poster, added_at) VALUES (?, ?, ?, ?)',
                 (movie_id, movie_data.get('title'), 
                  f"https://image.tmdb.org/t/p/w500{movie_data.get('poster_path')}" if movie_data.get('poster_path') else None,
                  datetime.now()))
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'added'})
    
    else:
        conn = sqlite3.connect('movies.db')
        c = conn.cursor()
        c.execute('SELECT movie_id, title, poster FROM wishlist ORDER BY added_at DESC')
        wishlist = [{'id': row[0], 'title': row[1], 'poster': row[2]} for row in c.fetchall()]
        conn.close()
        
        return jsonify({'wishlist': wishlist})

@app.route('/api/favorites', methods=['GET', 'POST'])
def favorites():
    if request.method == 'POST':
        data = request.json
        movie_id = data.get('movie_id')
        
        # Get movie details from TMDB
        response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}')
        movie_data = response.json()
        
        conn = sqlite3.connect('movies.db')
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO favorites (movie_id, title, poster, added_at) VALUES (?, ?, ?, ?)',
                 (movie_id, movie_data.get('title'), 
                  f"https://image.tmdb.org/t/p/w500{movie_data.get('poster_path')}" if movie_data.get('poster_path') else None,
                  datetime.now()))
        conn.commit()
        conn.close()
        
        return jsonify({'status': 'added'})
    
    else:
        conn = sqlite3.connect('movies.db')
        c = conn.cursor()
        c.execute('SELECT movie_id, title, poster FROM favorites ORDER BY added_at DESC')
        favorites = [{'id': row[0], 'title': row[1], 'poster': row[2]} for row in c.fetchall()]
        conn.close()
        
        return jsonify({'favorites': favorites})

@app.route('/api/admin/post', methods=['POST'])
def admin_post():
    data = request.json
    content = data.get('content')
    
    try:
        telegram_url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': f'🎬 Movie Recommendation:\n\n{content}',
            'parse_mode': 'HTML'
        }
        response = requests.post(telegram_url, json=payload)
        
        if response.status_code == 200:
            return jsonify({'status': 'posted'})
        else:
            return jsonify({'error': 'Failed to post'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))