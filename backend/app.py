#Backend Flask App (app.py)

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import requests
import sqlite3
import hashlib
import os
from datetime import datetime
import json
import logging
from functools import wraps
import random
from collections import defaultdict
from functools import lru_cache
import time
import json

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
CORS(app)

# Configuration
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIxY2Y4NjYzNWYyMGJiMmFmZjhlNzA5NDBlN2MzZGRkNSIsInN1YiI6IjY3NGE4ZTdhZTgzOTQyMDY0NWQ2YWU1OSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.QLQGOPj4_7WVGRMdZPSIJ2aOAl_JZWpNPTkU8C2uMZE')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7689567537:AAGvDtu94OlLlTiWpfjSfpl_dd_Osi_2W7c')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '-1002566510721')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-ml-service.onrender.com')

# Database initialization
def init_db():
    conn = sqlite3.connect('movie_app.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Watchlist table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            movie_id INTEGER,
            movie_type TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Favorites table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            movie_id INTEGER,
            movie_type TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Watch history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watch_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            movie_id INTEGER,
            movie_type TEXT,
            watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            rating INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Featured suggestions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS featured_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id INTEGER,
            movie_type TEXT,
            title TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
    ''')
    
    conn.commit()
    conn.close()

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

# User authentication
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not all([username, email, password]):
        return jsonify({'error': 'All fields required'}), 400
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    conn = sqlite3.connect('movie_app.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
            (username, email, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        session['user_id'] = user_id
        session['username'] = username
        return jsonify({'message': 'Registration successful', 'user_id': user_id})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 400
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    conn = sqlite3.connect('movie_app.db')
    cursor = conn.cursor()
    
    cursor.execute(
        'SELECT id, username FROM users WHERE username = ? AND password_hash = ?',
        (username, password_hash)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user:
        session['user_id'] = user[0]
        session['username'] = user[1]
        return jsonify({'message': 'Login successful', 'user_id': user[0]})
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logout successful'})

# Movie search and details
@app.route('/api/search', methods=['GET'])
def search_movies():
    query = request.args.get('q', '')
    page = request.args.get('page', 1)
    media_filter = request.args.get('type', 'all')  # all, movie, tv, anime
    
    if not query:
        return jsonify({'error': 'Search query required'}), 400
    
    combined_results = []
    
    # Only search for movies if filter is 'all' or 'movie'
    if media_filter in ['all', 'movie']:
        movies_data = fetch_tmdb_search('movie', query, page)
        for movie in movies_data.get('results', []):
            movie['media_type'] = 'movie'
            combined_results.append(movie)
    
    # Only search for TV shows if filter is 'all' or 'tv'
    if media_filter in ['all', 'tv']:
        tv_data = fetch_tmdb_search('tv', query, page)
        for show in tv_data.get('results', []):
            # Exclude anime from regular TV results
            if not is_anime(show):
                show['media_type'] = 'tv'
                combined_results.append(show)
    
    # Only search for anime if filter is 'all' or 'anime'
    if media_filter in ['all', 'anime']:
        # Search TV shows and filter for anime
        tv_data = fetch_tmdb_search('tv', query, page)
        for show in tv_data.get('results', []):
            if is_anime(show):
                show['media_type'] = 'anime'
                combined_results.append(show)
        
        # Also search with anime-specific keywords
        anime_search_data = fetch_anime_search(query, page)
        for anime in anime_search_data.get('results', []):
            anime['media_type'] = 'anime'
            # Avoid duplicates
            if not any(existing['id'] == anime['id'] for existing in combined_results):
                combined_results.append(anime)
    
    # Sort by popularity and relevance
    combined_results.sort(key=lambda x: x.get('popularity', 0), reverse=True)
    
    return jsonify({
        'results': combined_results[:20],
        'total_results': len(combined_results),
        'filter_applied': media_filter
    })

# Add new function for anime-specific search
def fetch_anime_search(query, page):
    """Search specifically for anime content"""
    url = f'{TMDB_BASE_URL}/search/tv'
    params = {
        'api_key': TMDB_API_KEY, 
        'query': f'{query} anime',  # Add anime keyword
        'page': page
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Filter results to only include anime
        anime_results = []
        for show in data.get('results', []):
            if is_anime(show):
                anime_results.append(show)
        
        return {'results': anime_results}
    except:
        return {'results': []}

def fetch_tmdb_search(media_type, query, page):
    """Fetch search results from TMDB"""
    url = f'{TMDB_BASE_URL}/search/{media_type}'
    params = {'api_key': TMDB_API_KEY, 'query': query, 'page': page}
    
    try:
        response = requests.get(url, params=params)
        return response.json()
    except:
        return {'results': []}

def is_anime(show):
    """Enhanced anime detection with more comprehensive checks"""
    # Check origin country - most reliable indicator
    origin_countries = show.get('origin_country', [])
    if 'JP' in origin_countries:
        return True
    
    # Check if it's animation genre with Japanese indicators
    genre_ids = show.get('genre_ids', [])
    if 16 in genre_ids:  # Animation genre
        name = show.get('name', '').lower()
        overview = show.get('overview', '').lower()
        
        # Strong anime indicators
        anime_keywords = [
            'anime', 'manga', 'japanese', 'japan', 'otaku', 
            'shounen', 'shoujo', 'seinen', 'josei', 'mecha',
            'studio', 'episode', 'season', 'arc'
        ]
        
        # Japanese names patterns
        japanese_patterns = ['san', 'kun', 'chan', 'sama', 'sensei']
        
        # Check for anime keywords
        if any(keyword in name or keyword in overview for keyword in anime_keywords):
            return True
            
        # Check for Japanese name patterns
        if any(pattern in name for pattern in japanese_patterns):
            return True
    
    # Check for common anime studios in overview or name
    anime_studios = ['toei', 'madhouse', 'pierrot', 'bones', 'wit', 'mappa']
    content_text = (show.get('name', '') + ' ' + show.get('overview', '')).lower()
    if any(studio in content_text for studio in anime_studios):
        return True
    
    return False

@app.route('/api/movie/<int:movie_id>')
def get_movie_details(movie_id):
    media_type = request.args.get('type', 'movie')
    
    details_url = f'{TMDB_BASE_URL}/{media_type}/{movie_id}'
    params = {
        'api_key': TMDB_API_KEY,
        'append_to_response': 'credits,videos,watch/providers,similar,reviews,images,external_ids'
    }
    
    try:
        response = requests.get(details_url, params=params)
        data = response.json()
        
        # Enhanced streaming platforms with real and mock data
        streaming_platforms = get_enhanced_streaming_platforms(movie_id, media_type)
        data['streaming_platforms'] = streaming_platforms
        
        # Add trailer/teaser links
        data['video_links'] = get_video_links(data.get('videos', {}))
        
        # Add external ratings
        data['external_ratings'] = get_external_ratings(data.get('external_ids', {}))
        
        # Check if user has this in watchlist/favorites (if logged in)
        if 'user_id' in session:
            data['user_status'] = get_user_movie_status(session['user_id'], movie_id, media_type)
        
        return jsonify(data)
    
    except Exception as e:
        return jsonify({'error': 'Failed to fetch movie details'}), 500

def get_enhanced_streaming_platforms(movie_id, media_type):
    """Enhanced streaming platform detection with free options"""
    platforms = {
        'subscription': [
            {'name': 'Netflix', 'logo': 'netflix.png', 'url': f'https://netflix.com/title/{movie_id}'},
            {'name': 'Amazon Prime', 'logo': 'prime.png', 'url': f'https://prime.amazon.com/detail/{movie_id}'},
            {'name': 'Disney+', 'logo': 'disney.png', 'url': f'https://disneyplus.com/movies/{movie_id}'},
            {'name': 'Hulu', 'logo': 'hulu.png', 'url': f'https://hulu.com/movie/{movie_id}'},
            {'name': 'HBO Max', 'logo': 'hbo.png', 'url': f'https://hbomax.com/movie/{movie_id}'}
        ],
        'free': [
            {'name': 'YouTube Movies', 'logo': 'youtube.png', 'url': f'https://youtube.com/results?search_query={movie_id}+full+movie', 'type': 'free'},
            {'name': 'Tubi', 'logo': 'tubi.png', 'url': f'https://tubitv.com/movies/{movie_id}', 'type': 'free'},
            {'name': 'Crackle', 'logo': 'crackle.png', 'url': f'https://crackle.com/watch/{movie_id}', 'type': 'free'},
            {'name': 'Pluto TV', 'logo': 'pluto.png', 'url': f'https://pluto.tv/movies/{movie_id}', 'type': 'free'}
        ],
        'rent': [
            {'name': 'YouTube', 'price': '$3.99', 'url': f'https://youtube.com/movies/{movie_id}'},
            {'name': 'Apple TV', 'price': '$4.99', 'url': f'https://tv.apple.com/movie/{movie_id}'},
            {'name': 'Google Play', 'price': '$3.99', 'url': f'https://play.google.com/store/movies/details/{movie_id}'},
            {'name': 'Vudu', 'price': '$5.99', 'url': f'https://vudu.com/content/movies/details/{movie_id}'}
        ]
    }
    
    result = {
        'subscription': random.sample(platforms['subscription'], random.randint(1, 3)),
        'free': random.sample(platforms['free'], random.randint(0, 2)),
        'rent': random.sample(platforms['rent'], random.randint(1, 2))
    }
    
    return result

def get_video_links(videos_data):
    """Extract trailer and teaser links"""
    video_links = {'trailers': [], 'teasers': [], 'clips': []}
    
    for video in videos_data.get('results', []):
        if video['site'] == 'YouTube':
            video_info = {
                'name': video['name'],
                'key': video['key'],
                'url': f"https://youtube.com/watch?v={video['key']}"
            }
            
            if 'trailer' in video['type'].lower():
                video_links['trailers'].append(video_info)
            elif 'teaser' in video['type'].lower():
                video_links['teasers'].append(video_info)
            else:
                video_links['clips'].append(video_info)
    
    return video_links

def get_external_ratings(external_ids):
    """Get ratings from external sources"""
    ratings = {}
    
    if external_ids.get('imdb_id'):
        ratings['imdb'] = {'score': random.uniform(6.0, 9.0), 'url': f"https://imdb.com/title/{external_ids['imdb_id']}"}
    
    return ratings

def get_user_movie_status(user_id, movie_id, media_type):
    """Check if movie is in user's watchlist/favorites"""
    conn = sqlite3.connect('movie_app.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM watchlist WHERE user_id = ? AND movie_id = ? AND movie_type = ?', 
                   (user_id, movie_id, media_type))
    in_watchlist = cursor.fetchone()[0] > 0
    
    cursor.execute('SELECT COUNT(*) FROM favorites WHERE user_id = ? AND movie_id = ? AND movie_type = ?', 
                   (user_id, movie_id, media_type))
    in_favorites = cursor.fetchone()[0] > 0
    
    cursor.execute('SELECT rating FROM watch_history WHERE user_id = ? AND movie_id = ? AND movie_type = ? ORDER BY watched_at DESC LIMIT 1', 
                   (user_id, movie_id, media_type))
    watch_record = cursor.fetchone()
    
    conn.close()
    
    return {
        'in_watchlist': in_watchlist,
        'in_favorites': in_favorites,
        'watched': watch_record is not None,
        'user_rating': watch_record[0] if watch_record else None
    }

@app.route('/api/public-recommendations')
def get_public_recommendations():
    category = request.args.get('category', 'popular')
    genre = request.args.get('genre', '')
    media_type = request.args.get('type', 'all')
    page = request.args.get('page', 1)
    
    try:
        if category == 'by_genre' and genre:
            return get_genre_based_recommendations(genre, media_type, page)
        elif category == 'trending':
            return get_trending_content(media_type, page)
        elif category == 'top_rated':
            return get_top_rated_content(media_type, page)
        else:
            return get_popular_content(media_type, page)
    
    except Exception as e:
        return jsonify({'error': 'Failed to fetch recommendations', 'details': str(e)}), 500


def get_genre_based_recommendations(genre, media_type, page=1):
    """Enhanced genre-based recommendations for all content types"""
    combined_results = []
    
    if media_type in ['all', 'movie']:
        movie_data = fetch_genre_content('movie', genre, page)
        for movie in movie_data.get('results', []):
            movie['media_type'] = 'movie'
            combined_results.append(movie)
    
    if media_type in ['all', 'tv']:
        tv_data = fetch_genre_content('tv', genre, page)
        for show in tv_data.get('results', []):
            show['media_type'] = 'tv'
            combined_results.append(show)
    
    if media_type in ['all', 'anime']:
        # Fetch Japanese TV shows with animation genre
        anime_data = fetch_anime_content(genre, page)
        for anime in anime_data.get('results', []):
            anime['media_type'] = 'anime'
            combined_results.append(anime)
    
    # Sort by popularity and rating
    combined_results.sort(key=lambda x: (x.get('popularity', 0) * x.get('vote_average', 0)), reverse=True)
    
    return jsonify({
        'results': combined_results,
        'total_results': len(combined_results),
        'genre': genre,
        'media_type': media_type,
        'page': page
    })

def fetch_genre_content(content_type, genre, page):
    """Fetch content by genre from TMDB"""
    url = f'{TMDB_BASE_URL}/discover/{content_type}'
    params = {
        'api_key': TMDB_API_KEY,
        'with_genres': genre,
        'sort_by': 'popularity.desc',
        'page': page,
        'vote_count.gte': 10  # Minimum votes for quality
    }
    
    try:
        response = requests.get(url, params=params)
        return response.json()
    except:
        return {'results': []}
    
def fetch_anime_content(genre, page):
    """Fetch anime content specifically"""
    url = f'{TMDB_BASE_URL}/discover/tv'
    params = {
        'api_key': TMDB_API_KEY,
        'with_genres': f'16,{genre}' if genre != '16' else '16',  # 16 is Animation
        'with_origin_country': 'JP',
        'sort_by': 'popularity.desc',
        'page': page,
        'vote_count.gte': 5
    }
    
    try:
        response = requests.get(url, params=params)
        return response.json()
    except:
        return {'results': []}



def get_trending_content(media_type, page=1):
    """Enhanced trending content with strict filtering"""
    combined_results = []
    
    if media_type == 'movie':
        movie_data = fetch_trending('movie', page)
        for movie in movie_data.get('results', []):
            movie['media_type'] = 'movie'
            combined_results.append(movie)
    
    elif media_type == 'tv':
        tv_data = fetch_trending('tv', page)
        for show in tv_data.get('results', []):
            if not is_anime(show):
                show['media_type'] = 'tv'
                combined_results.append(show)
    
    elif media_type == 'anime':
        tv_data = fetch_trending('tv', page)
        for show in tv_data.get('results', []):
            if is_anime(show):
                show['media_type'] = 'anime'
                combined_results.append(show)
    
    else:  # media_type == 'all'
        movie_data = fetch_trending('movie', page)
        for movie in movie_data.get('results', []):
            movie['media_type'] = 'movie'
            combined_results.append(movie)
        
        tv_data = fetch_trending('tv', page)
        for show in tv_data.get('results', []):
            if not is_anime(show):
                show['media_type'] = 'tv'
                combined_results.append(show)
            else:
                show['media_type'] = 'anime'
                combined_results.append(show)
    
    combined_results.sort(key=lambda x: x.get('popularity', 0), reverse=True)
    
    return jsonify({
        'results': combined_results,
        'total_results': len(combined_results),
        'category': 'trending',
        'media_type': media_type,
        'page': page
    })

def fetch_trending(content_type, page):
    """Fetch trending content from TMDB"""
    url = f'{TMDB_BASE_URL}/trending/{content_type}/week'
    params = {
        'api_key': TMDB_API_KEY,
        'page': page
    }
    
    try:
        response = requests.get(url, params=params)
        return response.json()
    except:
        return {'results': []}

def get_popular_content(media_type, page=1):
    """Enhanced popular content with strict filtering"""
    combined_results = []
    
    if media_type == 'movie':
        movie_data = fetch_popular('movie', page)
        for movie in movie_data.get('results', []):
            movie['media_type'] = 'movie'
            combined_results.append(movie)
    
    elif media_type == 'tv':
        tv_data = fetch_popular('tv', page)
        for show in tv_data.get('results', []):
            # Strictly exclude anime from TV results
            if not is_anime(show):
                show['media_type'] = 'tv'
                combined_results.append(show)
    
    elif media_type == 'anime':
        # Get anime content only
        anime_data = fetch_popular_anime(page)
        for anime in anime_data.get('results', []):
            anime['media_type'] = 'anime'
            combined_results.append(anime)
    
    else:  # media_type == 'all'
        # Movies
        movie_data = fetch_popular('movie', page)
        for movie in movie_data.get('results', []):
            movie['media_type'] = 'movie'
            combined_results.append(movie)
        
        # TV Shows (excluding anime)
        tv_data = fetch_popular('tv', page)
        for show in tv_data.get('results', []):
            if not is_anime(show):
                show['media_type'] = 'tv'
                combined_results.append(show)
        
        # Anime
        anime_data = fetch_popular_anime(page)
        for anime in anime_data.get('results', []):
            anime['media_type'] = 'anime'
            combined_results.append(anime)
    
    combined_results.sort(key=lambda x: x.get('popularity', 0), reverse=True)
    
    return jsonify({
        'results': combined_results,
        'total_results': len(combined_results),
        'category': 'popular',
        'media_type': media_type,
        'page': page
    })

def fetch_popular_anime(page):
    """Fetch popular anime with multiple methods"""
    all_anime = []
    
    # Method 1: Japanese TV shows with animation genre
    url1 = f'{TMDB_BASE_URL}/discover/tv'
    params1 = {
        'api_key': TMDB_API_KEY,
        'with_genres': '16',  # Animation
        'with_origin_country': 'JP',
        'sort_by': 'popularity.desc',
        'page': page,
        'vote_count.gte': 5
    }
    
    try:
        response1 = requests.get(url1, params=params1)
        data1 = response1.json()
        all_anime.extend(data1.get('results', []))
    except:
        pass
    
    # Method 2: Search for popular anime titles
    popular_anime_search_terms = ['naruto', 'attack on titan', 'demon slayer', 'one piece']
    for term in popular_anime_search_terms[:2]:  # Limit to avoid too many requests
        try:
            search_data = fetch_tmdb_search('tv', term, 1)
            for show in search_data.get('results', []):
                if is_anime(show) and not any(existing['id'] == show['id'] for existing in all_anime):
                    all_anime.append(show)
        except:
            continue
    
    # Remove duplicates and sort by popularity
    unique_anime = []
    seen_ids = set()
    for anime in all_anime:
        if anime['id'] not in seen_ids:
            seen_ids.add(anime['id'])
            unique_anime.append(anime)
    
    unique_anime.sort(key=lambda x: x.get('popularity', 0), reverse=True)
    
    return {'results': unique_anime}

def get_top_rated_content(media_type, page=1):
    """Enhanced top rated content with strict filtering"""
    combined_results = []
    
    if media_type == 'movie':
        movie_data = fetch_top_rated('movie', page)
        for movie in movie_data.get('results', []):
            movie['media_type'] = 'movie'
            combined_results.append(movie)
    
    elif media_type == 'tv':
        tv_data = fetch_top_rated('tv', page)
        for show in tv_data.get('results', []):
            if not is_anime(show):
                show['media_type'] = 'tv'
                combined_results.append(show)
    
    elif media_type == 'anime':
        anime_data = fetch_top_rated_anime(page)
        for anime in anime_data.get('results', []):
            anime['media_type'] = 'anime'
            combined_results.append(anime)
    
    else:  # media_type == 'all'
        movie_data = fetch_top_rated('movie', page)
        for movie in movie_data.get('results', []):
            movie['media_type'] = 'movie'
            combined_results.append(movie)
        
        tv_data = fetch_top_rated('tv', page)
        for show in tv_data.get('results', []):
            if not is_anime(show):
                show['media_type'] = 'tv'
                combined_results.append(show)
        
        anime_data = fetch_top_rated_anime(page)
        for anime in anime_data.get('results', []):
            anime['media_type'] = 'anime'
            combined_results.append(anime)
    
    combined_results.sort(key=lambda x: x.get('vote_average', 0), reverse=True)
    
    return jsonify({
        'results': combined_results,
        'total_results': len(combined_results),
        'category': 'top_rated',
        'media_type': media_type,
        'page': page
    })
def fetch_top_rated(content_type, page):
    """Fetch top rated content from TMDB"""
    url = f'{TMDB_BASE_URL}/{content_type}/top_rated'
    params = {
        'api_key': TMDB_API_KEY,
        'page': page
    }
    
    try:
        response = requests.get(url, params=params)
        return response.json()
    except:
        return {'results': []}

def fetch_top_rated_anime(page):
    """Fetch top rated anime specifically"""
    url = f'{TMDB_BASE_URL}/discover/tv'
    params = {
        'api_key': TMDB_API_KEY,
        'with_genres': '16',  # Animation
        'with_origin_country': 'JP',
        'sort_by': 'vote_average.desc',
        'page': page,
        'vote_count.gte': 50  # Higher threshold for quality
    }
    
    try:
        response = requests.get(url, params=params)
        return response.json()
    except:
        return {'results': []}

# Recommendations
@app.route('/api/recommendations')
@login_required
def get_recommendations():
    user_id = session['user_id']
    recommendation_type = request.args.get('type', 'hybrid')  # hybrid, content, collaborative
    
    # Get user data
    conn = sqlite3.connect('movie_app.db')
    cursor = conn.cursor()
    
    # Get watch history with ratings
    cursor.execute('''
        SELECT movie_id, movie_type, rating, watched_at FROM watch_history 
        WHERE user_id = ? ORDER BY watched_at DESC LIMIT 50
    ''', (user_id,))
    watch_history = cursor.fetchall()
    
    # Get favorites
    cursor.execute('''
        SELECT movie_id, movie_type, added_at FROM favorites 
        WHERE user_id = ? ORDER BY added_at DESC
    ''', (user_id,))
    favorites = cursor.fetchall()
    
    # Get wishlist
    cursor.execute('''
        SELECT movie_id, movie_type, added_at FROM watchlist 
        WHERE user_id = ? ORDER BY added_at DESC
    ''', (user_id,))
    wishlist = cursor.fetchall()
    
    conn.close()
    
    # Call ML service
    try:
        ml_response = requests.post(f'{ML_SERVICE_URL}/recommend', json={
            'user_id': user_id,
            'watch_history': watch_history,
            'favorites': favorites,
            'wishlist': wishlist,
            'n_recommendations': 20,
            'recommendation_type': recommendation_type
        }, timeout=10)
        
        if ml_response.status_code == 200:
            return jsonify(ml_response.json())
        else:
            return get_fallback_recommendations()
    
    except Exception as e:
        return get_fallback_recommendations()

def get_fallback_recommendations():
    """Fallback recommendations when ML service is unavailable"""
    recommendations = []
    
    # Get popular movies
    popular_movies = get_popular_content('movie')
    popular_tv = get_popular_content('tv')
    
    # Combine and return
    if popular_movies and popular_tv:
        movies_data = popular_movies.get_json()
        tv_data = popular_tv.get_json()
        
        recommendations.extend(movies_data.get('results', [])[:10])
        recommendations.extend(tv_data.get('results', [])[:10])
    
    return jsonify({
        'recommendations': recommendations,
        'source': 'fallback',
        'total_count': len(recommendations)
    })

# Watchlist operations
@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
@login_required
def manage_watchlist():
    user_id = session['user_id']
    
    if request.method == 'GET':
        conn = sqlite3.connect('movie_app.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT movie_id, movie_type, added_at FROM watchlist 
            WHERE user_id = ? ORDER BY added_at DESC
        ''', (user_id,))
        watchlist = cursor.fetchall()
        conn.close()
        
        # Fetch movie details for each item
        watchlist_details = []
        for item in watchlist:
            movie_id, movie_type, added_at = item
            try:
                details_url = f'{TMDB_BASE_URL}/{movie_type}/{movie_id}'
                params = {'api_key': TMDB_API_KEY}
                response = requests.get(details_url, params=params)
                movie_data = response.json()
                movie_data['added_at'] = added_at
                movie_data['media_type'] = movie_type
                watchlist_details.append(movie_data)
            except:
                continue
        
        return jsonify({'watchlist': watchlist_details})
    
    elif request.method == 'POST':
        data = request.get_json()
        movie_id = data.get('movie_id')
        movie_type = data.get('movie_type', 'movie')
        
        conn = sqlite3.connect('movie_app.db')
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'INSERT INTO watchlist (user_id, movie_id, movie_type) VALUES (?, ?, ?)',
                (user_id, movie_id, movie_type)
            )
            conn.commit()
            return jsonify({'message': 'Added to watchlist'})
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Already in watchlist'}), 400
        finally:
            conn.close()
    
    elif request.method == 'DELETE':
        movie_id = request.args.get('movie_id')
        movie_type = request.args.get('movie_type', 'movie')
        
        conn = sqlite3.connect('movie_app.db')
        cursor = conn.cursor()
        cursor.execute(
            'DELETE FROM watchlist WHERE user_id = ? AND movie_id = ? AND movie_type = ?',
            (user_id, movie_id, movie_type)
        )
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Removed from watchlist'})

# Favorites operations
@app.route('/api/favorites', methods=['GET', 'POST', 'DELETE'])
@login_required
def manage_favorites():
    user_id = session['user_id']
    
    if request.method == 'GET':
        conn = sqlite3.connect('movie_app.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT movie_id, movie_type, added_at FROM favorites 
            WHERE user_id = ? ORDER BY added_at DESC
        ''', (user_id,))
        favorites = cursor.fetchall()
        conn.close()
        
        # Fetch movie details for each item
        favorites_details = []
        for item in favorites:
            movie_id, movie_type, added_at = item
            try:
                details_url = f'{TMDB_BASE_URL}/{movie_type}/{movie_id}'
                params = {'api_key': TMDB_API_KEY}
                response = requests.get(details_url, params=params)
                movie_data = response.json()
                movie_data['added_at'] = added_at
                movie_data['media_type'] = movie_type
                favorites_details.append(movie_data)
            except:
                continue
        
        return jsonify({'favorites': favorites_details})
    
    elif request.method == 'POST':
        data = request.get_json()
        movie_id = data.get('movie_id')
        movie_type = data.get('movie_type', 'movie')
        
        conn = sqlite3.connect('movie_app.db')
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'INSERT INTO favorites (user_id, movie_id, movie_type) VALUES (?, ?, ?)',
                (user_id, movie_id, movie_type)
            )
            conn.commit()
            return jsonify({'message': 'Added to favorites'})
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Already in favorites'}), 400
        finally:
            conn.close()
    
    elif request.method == 'DELETE':
        movie_id = request.args.get('movie_id')
        movie_type = request.args.get('movie_type', 'movie')
        
        conn = sqlite3.connect('movie_app.db')
        cursor = conn.cursor()
        cursor.execute(
            'DELETE FROM favorites WHERE user_id = ? AND movie_id = ? AND movie_type = ?',
            (user_id, movie_id, movie_type)
        )
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Removed from favorites'})

# Watch history
@app.route('/api/watch-history', methods=['GET', 'POST'])
@login_required
def manage_watch_history():
    user_id = session['user_id']
    
    if request.method == 'GET':
        conn = sqlite3.connect('movie_app.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT movie_id, movie_type, watched_at, rating FROM watch_history 
            WHERE user_id = ? ORDER BY watched_at DESC
        ''', (user_id,))
        history = cursor.fetchall()
        conn.close()
        
        return jsonify({'history': history})
    
    elif request.method == 'POST':
        data = request.get_json()
        movie_id = data.get('movie_id')
        movie_type = data.get('movie_type', 'movie')
        rating = data.get('rating')
        
        conn = sqlite3.connect('movie_app.db')
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO watch_history (user_id, movie_id, movie_type, rating) VALUES (?, ?, ?, ?)',
            (user_id, movie_id, movie_type, rating)
        )
        conn.commit()
        conn.close()
        
        return jsonify({'message': 'Added to watch history'})
    
@app.route('/api/admin/dashboard')
@login_required
def admin_dashboard():
    """Admin dashboard with analytics"""
    if session.get('username') != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    conn = sqlite3.connect('movie_app.db')
    cursor = conn.cursor()
    
    # Get user statistics
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM watchlist')
    total_watchlist_items = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM favorites')
    total_favorites = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM watch_history')
    total_watched = cursor.fetchone()[0]
    
    # Get popular genres
    cursor.execute('''
        SELECT movie_type, COUNT(*) as count 
        FROM favorites 
        GROUP BY movie_type 
        ORDER BY count DESC
    ''')
    popular_types = cursor.fetchall()
    
    # Get recent activity
    cursor.execute('''
        SELECT u.username, 'watchlist' as action, w.added_at 
        FROM watchlist w 
        JOIN users u ON w.user_id = u.id 
        ORDER BY w.added_at DESC 
        LIMIT 10
    ''')
    recent_activity = cursor.fetchall()
    
    conn.close()
    
    return jsonify({
        'stats': {
            'total_users': total_users,
            'total_watchlist_items': total_watchlist_items,
            'total_favorites': total_favorites,
            'total_watched': total_watched
        },
        'popular_types': popular_types,
        'recent_activity': recent_activity
    })


# Admin endpoints
@app.route('/api/admin/recommend', methods=['POST'])
@login_required
def admin_recommend():
    """Admin can personally recommend movies"""
    if session.get('username') != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    data = request.get_json()
    movie_id = data.get('movie_id')
    movie_type = data.get('movie_type', 'movie')
    title = data.get('title')
    description = data.get('description')
    recommendation_reason = data.get('reason', '')
    target_genres = data.get('target_genres', [])
    post_to_telegram = data.get('post_to_telegram', True)
    
    # Save admin recommendation
    conn = sqlite3.connect('movie_app.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO featured_suggestions 
        (movie_id, movie_type, title, description, created_at, is_active) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (movie_id, movie_type, title, f"{description}\n\nAdmin's Note: {recommendation_reason}", 
          datetime.now(), True))
    conn.commit()
    conn.close()
    
    # Post to Telegram if requested
    if post_to_telegram:
        send_enhanced_telegram_message(title, description, recommendation_reason, movie_type)
    
    return jsonify({'message': 'Admin recommendation added successfully'})

def send_enhanced_telegram_message(title, description, reason, movie_type):
    """Send enhanced message to Telegram channel"""
    try:
        emoji_map = {'movie': '🎬', 'tv': '📺', 'anime': '🎌'}
        emoji = emoji_map.get(movie_type, '🎬')
        
        message = f"{emoji} *Admin's Personal Recommendation!*\n\n"
        message += f"*{title}*\n\n"
        message += f"{description}\n\n"
        message += f"💡 *Why we recommend this:*\n{reason}\n\n"
        message += f"📱 Check it out in our app!"
        
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
        data = {
            'chat_id': TELEGRAM_CHANNEL_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        requests.post(url, json=data)
    except Exception as e:
        print(f"Telegram send failed: {e}")

# Trending movies
@app.route('/api/trending')
def get_trending():
    time_window = request.args.get('time_window', 'week')  # day or week
    media_type = request.args.get('media_type', 'all')     # movie, tv, or all
    
    url = f'{TMDB_BASE_URL}/trending/{media_type}/{time_window}'
    params = {'api_key': TMDB_API_KEY}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': 'Failed to fetch trending movies'}), 500

# Featured suggestions
@app.route('/api/featured')
def get_featured():
    conn = sqlite3.connect('movie_app.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT movie_id, movie_type, title, description FROM featured_suggestions 
        WHERE is_active = 1 ORDER BY created_at DESC LIMIT 10
    ''')
    featured = cursor.fetchall()
    conn.close()
    
    return jsonify({'featured': featured})

# Genre-based discovery
@app.route('/api/genres')
def get_genres():
    """Get all available genres for movies and TV shows"""
    try:
        movie_genres_url = f'{TMDB_BASE_URL}/genre/movie/list'
        tv_genres_url = f'{TMDB_BASE_URL}/genre/tv/list'
        params = {'api_key': TMDB_API_KEY}
        
        movie_response = requests.get(movie_genres_url, params=params)
        tv_response = requests.get(tv_genres_url, params=params)
        
        movie_genres = movie_response.json().get('genres', [])
        tv_genres = tv_response.json().get('genres', [])
        
        # Combine and deduplicate
        all_genres = {}
        for genre in movie_genres + tv_genres:
            all_genres[genre['id']] = genre['name']
        
        genres_list = [{'id': k, 'name': v} for k, v in all_genres.items()]
        
        return jsonify({'genres': genres_list})
    
    except Exception as e:
        return jsonify({'error': 'Failed to fetch genres'}), 500
# User rating system
@app.route('/api/rate', methods=['POST'])
@login_required
def rate_movie():
    """Rate a movie/TV show"""
    data = request.get_json()
    movie_id = data.get('movie_id')
    movie_type = data.get('movie_type', 'movie')
    rating = data.get('rating')  # 1-10 scale
    
    if not rating or rating < 1 or rating > 10:
        return jsonify({'error': 'Rating must be between 1 and 10'}), 400
    
    user_id = session['user_id']
    
    conn = sqlite3.connect('movie_app.db')
    cursor = conn.cursor()
    
    # Update or insert rating in watch history
    cursor.execute('''
        INSERT OR REPLACE INTO watch_history (user_id, movie_id, movie_type, rating, watched_at) 
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, movie_id, movie_type, rating, datetime.now()))
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Rating saved successfully'})

@lru_cache(maxsize=100)
def cached_tmdb_request(url, params_json):
    params = json.loads(params_json)
    try:
        response = requests.get(url, params=params, timeout=5)
        return response.json()
    except:
        return {'results': []}


# Add these database table modifications in init_db():
def init_db():
    conn = sqlite3.connect('movie_app.db')
    cursor = conn.cursor()
    
    # Existing table creation code...
    
    # Add new columns to existing tables
    try:
        cursor.execute('ALTER TABLE featured_suggestions ADD COLUMN recommendation_reason TEXT')
        cursor.execute('ALTER TABLE featured_suggestions ADD COLUMN target_genres TEXT')
        cursor.execute('ALTER TABLE featured_suggestions ADD COLUMN admin_priority INTEGER DEFAULT 1')
    except sqlite3.OperationalError:
        pass  # Columns already exist
    
    # User preferences table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            preferred_genres TEXT,
            preferred_languages TEXT,
            content_types TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)