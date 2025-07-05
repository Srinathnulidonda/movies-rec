#backend/app.py
from flask import Flask, request, jsonify, session, g
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import requests
import os
import logging
from functools import wraps
import json
import hashlib
from collections import defaultdict
import random

# App configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///recommendations.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
CORS(app, supports_credentials=True)

# External API configurations
TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
JIKAN_BASE_URL = 'https://api.jikan.moe/v4'
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'http://localhost:5001')

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    preferred_languages = db.Column(db.Text)  # JSON string
    region = db.Column(db.String(10), default='US')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    external_id = db.Column(db.String(50), nullable=False)
    source = db.Column(db.String(20), nullable=False)  # tmdb, jikan, regional
    content_type = db.Column(db.String(20), nullable=False)  # movie, tv, anime
    title = db.Column(db.String(200), nullable=False)
    original_title = db.Column(db.String(200))
    overview = db.Column(db.Text)
    genres = db.Column(db.Text)  # JSON string
    languages = db.Column(db.Text)  # JSON string
    poster_path = db.Column(db.String(200))
    backdrop_path = db.Column(db.String(200))
    release_date = db.Column(db.DateTime)
    rating = db.Column(db.Float)
    metadata = db.Column(db.Text)  # JSON string for additional data
    cached_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)  # rating, watchlist, favorite, viewed
    value = db.Column(db.Float)  # rating value or duration watched
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AdminRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(50))
    languages = db.Column(db.Text)  # JSON string
    priority = db.Column(db.Integer, default=1)
    expires_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

class AnonymousSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(128), unique=True, nullable=False)
    search_history = db.Column(db.Text)  # JSON string
    viewed_content = db.Column(db.Text)  # JSON string
    preferences = db.Column(db.Text)  # JSON string
    last_active = db.Column(db.DateTime, default=datetime.utcnow)

# Helper functions
def get_session_id():
    """Generate or retrieve session ID for anonymous users"""
    if 'session_id' not in session:
        session['session_id'] = hashlib.md5(f"{datetime.utcnow()}{random.random()}".encode()).hexdigest()
    return session['session_id']

def get_user_location():
    """Get user's approximate location from IP"""
    try:
        ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        if ip and ip != '127.0.0.1':
            response = requests.get(f'http://ip-api.com/json/{ip}', timeout=2)
            if response.status_code == 200:
                data = response.json()
                return data.get('countryCode', 'US')
    except:
        pass
    return 'US'

def cache_content(content_data, source, content_type):
    """Cache content in database"""
    existing = Content.query.filter_by(
        external_id=content_data['id'], 
        source=source
    ).first()
    
    if existing:
        # Update cache timestamp
        existing.cached_at = datetime.utcnow()
        db.session.commit()
        return existing
    
    # Create new content entry
    content = Content(
        external_id=str(content_data['id']),
        source=source,
        content_type=content_type,
        title=content_data.get('title', content_data.get('name', '')),
        original_title=content_data.get('original_title', content_data.get('original_name', '')),
        overview=content_data.get('overview', content_data.get('synopsis', '')),
        genres=json.dumps(content_data.get('genres', [])),
        poster_path=content_data.get('poster_path'),
        backdrop_path=content_data.get('backdrop_path'),
        rating=content_data.get('vote_average', content_data.get('score', 0)),
        metadata=json.dumps(content_data)
    )
    
    if 'release_date' in content_data:
        try:
            content.release_date = datetime.strptime(content_data['release_date'], '%Y-%m-%d')
        except:
            pass
    
    db.session.add(content)
    db.session.commit()
    return content

def fetch_tmdb_content(endpoint, params=None):
    """Fetch content from TMDB API"""
    if not TMDB_API_KEY:
        return None
    
    url = f"{TMDB_BASE_URL}/{endpoint}"
    params = params or {}
    params['api_key'] = TMDB_API_KEY
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logging.error(f"TMDB API error: {e}")
    
    return None

def fetch_jikan_content(endpoint, params=None):
    """Fetch anime content from Jikan API"""
    url = f"{JIKAN_BASE_URL}/{endpoint}"
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logging.error(f"Jikan API error: {e}")
    
    return None

def get_ml_recommendations(user_id, content_preferences=None):
    """Get ML-powered recommendations"""
    try:
        payload = {'user_id': user_id}
        if content_preferences:
            payload['preferences'] = content_preferences
        
        response = requests.post(f"{ML_SERVICE_URL}/recommend", json=payload, timeout=5)
        if response.status_code == 200:
            return response.json().get('recommendations', [])
    except Exception as e:
        logging.error(f"ML service error: {e}")
    
    return []

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        g.user = User.query.get(session['user_id'])
        if not g.user:
            return jsonify({'error': 'User not found'}), 401
        return f(*args, **kwargs)
    return decorated_function

def require_admin(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        g.user = User.query.get(session['user_id'])
        if not g.user or not g.user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

# Authentication routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    user = User(
        username=data['username'],
        email=data['email'],
        region=data.get('region', get_user_location()),
        preferred_languages=json.dumps(data.get('preferred_languages', ['en']))
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    session['user_id'] = user.id
    return jsonify({'message': 'User registered successfully', 'user_id': user.id})

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    
    if user and user.check_password(data['password']):
        session['user_id'] = user.id
        return jsonify({'message': 'Login successful', 'user_id': user.id, 'is_admin': user.is_admin})
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'message': 'Logged out successfully'})

# Homepage recommendations for non-logged users
@app.route('/api/homepage-recommendations')
def homepage_recommendations():
    session_id = get_session_id()
    user_region = get_user_location()
    
    # Get or create anonymous session
    anon_session = AnonymousSession.query.filter_by(session_id=session_id).first()
    if not anon_session:
        anon_session = AnonymousSession(session_id=session_id)
        db.session.add(anon_session)
        db.session.commit()
    
    recommendations = {}
    
    # Popular movies
    tmdb_popular = fetch_tmdb_content('movie/popular', {'region': user_region})
    if tmdb_popular:
        recommendations['popular_movies'] = []
        for movie in tmdb_popular['results'][:10]:
            content = cache_content(movie, 'tmdb', 'movie')
            recommendations['popular_movies'].append({
                'id': content.id,
                'title': content.title,
                'overview': content.overview,
                'poster_path': content.poster_path,
                'rating': content.rating
            })
    
    # Popular TV shows
    tmdb_tv = fetch_tmdb_content('tv/popular', {'region': user_region})
    if tmdb_tv:
        recommendations['popular_tv'] = []
        for show in tmdb_tv['results'][:10]:
            content = cache_content(show, 'tmdb', 'tv')
            recommendations['popular_tv'].append({
                'id': content.id,
                'title': content.title,
                'overview': content.overview,
                'poster_path': content.poster_path,
                'rating': content.rating
            })
    
    # Trending content
    tmdb_trending = fetch_tmdb_content('trending/all/day')
    if tmdb_trending:
        recommendations['trending'] = []
        for item in tmdb_trending['results'][:10]:
            content_type = 'movie' if item['media_type'] == 'movie' else 'tv'
            content = cache_content(item, 'tmdb', content_type)
            recommendations['trending'].append({
                'id': content.id,
                'title': content.title,
                'overview': content.overview,
                'poster_path': content.poster_path,
                'rating': content.rating,
                'type': content_type
            })
    
    # Popular anime
    jikan_anime = fetch_jikan_content('top/anime', {'limit': 10})
    if jikan_anime:
        recommendations['popular_anime'] = []
        for anime in jikan_anime['data']:
            content = cache_content(anime, 'jikan', 'anime')
            recommendations['popular_anime'].append({
                'id': content.id,
                'title': content.title,
                'overview': content.overview,
                'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                'rating': content.rating
            })
    
    # Admin curated recommendations
    admin_recs = AdminRecommendation.query.filter(
        AdminRecommendation.is_active == True,
        AdminRecommendation.expires_at > datetime.utcnow()
    ).order_by(AdminRecommendation.priority.desc()).limit(5).all()
    
    if admin_recs:
        recommendations['featured'] = []
        for rec in admin_recs:
            content = Content.query.get(rec.content_id)
            if content:
                recommendations['featured'].append({
                    'id': content.id,
                    'title': rec.title,
                    'description': rec.description,
                    'poster_path': content.poster_path,
                    'rating': content.rating,
                    'category': rec.category
                })
    
    return jsonify(recommendations)

# Personalized recommendations for logged-in users
@app.route('/api/personalized-recommendations')
@require_auth
def personalized_recommendations():
    user = g.user
    recommendations = {}
    
    # Get user interactions
    interactions = UserInteraction.query.filter_by(user_id=user.id).all()
    
    # Based on favorites
    favorite_content = [i.content_id for i in interactions if i.interaction_type == 'favorite']
    if favorite_content:
        # Get similar content based on genres
        similar_content = []
        for content_id in favorite_content[:5]:  # Limit to avoid too many queries
            content = Content.query.get(content_id)
            if content and content.genres:
                genres = json.loads(content.genres)
                genre_ids = [g['id'] for g in genres if isinstance(g, dict)]
                
                if genre_ids:
                    genre_based = fetch_tmdb_content('discover/movie', {
                        'with_genres': ','.join(map(str, genre_ids)),
                        'sort_by': 'popularity.desc'
                    })
                    
                    if genre_based:
                        for movie in genre_based['results'][:5]:
                            cached = cache_content(movie, 'tmdb', 'movie')
                            similar_content.append({
                                'id': cached.id,
                                'title': cached.title,
                                'overview': cached.overview,
                                'poster_path': cached.poster_path,
                                'rating': cached.rating
                            })
        
        recommendations['based_on_favorites'] = similar_content[:10]
    
    # Based on watchlist
    watchlist_content = [i.content_id for i in interactions if i.interaction_type == 'watchlist']
    if watchlist_content:
        recommendations['complete_watchlist'] = []
        for content_id in watchlist_content[:10]:
            content = Content.query.get(content_id)
            if content:
                recommendations['complete_watchlist'].append({
                    'id': content.id,
                    'title': content.title,
                    'overview': content.overview,
                    'poster_path': content.poster_path,
                    'rating': content.rating
                })
    
    # ML-powered recommendations
    ml_recs = get_ml_recommendations(user.id)
    if ml_recs:
        recommendations['ai_powered'] = ml_recs
    
    # Language-based recommendations
    if user.preferred_languages:
        langs = json.loads(user.preferred_languages)
        for lang in langs[:2]:  # Limit to top 2 languages
            lang_movies = fetch_tmdb_content('discover/movie', {
                'with_original_language': lang,
                'sort_by': 'popularity.desc'
            })
            
            if lang_movies:
                lang_content = []
                for movie in lang_movies['results'][:5]:
                    cached = cache_content(movie, 'tmdb', 'movie')
                    lang_content.append({
                        'id': cached.id,
                        'title': cached.title,
                        'overview': cached.overview,
                        'poster_path': cached.poster_path,
                        'rating': cached.rating
                    })
                
                recommendations[f'{lang}_movies'] = lang_content
    
    return jsonify(recommendations)

# Content details
@app.route('/api/content/<int:content_id>/details')
def content_details(content_id):
    content = Content.query.get_or_404(content_id)
    metadata = json.loads(content.metadata) if content.metadata else {}
    
    details = {
        'id': content.id,
        'title': content.title,
        'original_title': content.original_title,
        'overview': content.overview,
        'genres': json.loads(content.genres) if content.genres else [],
        'poster_path': content.poster_path,
        'backdrop_path': content.backdrop_path,
        'release_date': content.release_date.isoformat() if content.release_date else None,
        'rating': content.rating,
        'source': content.source,
        'content_type': content.content_type,
        'metadata': metadata
    }
    
    # Get additional details from source APIs
    if content.source == 'tmdb':
        endpoint = f"{content.content_type}/{content.external_id}"
        extra_data = fetch_tmdb_content(endpoint, {'append_to_response': 'credits,videos,similar'})
        if extra_data:
            details.update({
                'cast': extra_data.get('credits', {}).get('cast', [])[:10],
                'crew': extra_data.get('credits', {}).get('crew', [])[:5],
                'videos': extra_data.get('videos', {}).get('results', [])[:3],
                'similar': extra_data.get('similar', {}).get('results', [])[:6]
            })
    
    # Get user interaction if logged in
    if 'user_id' in session:
        interaction = UserInteraction.query.filter_by(
            user_id=session['user_id'],
            content_id=content.id
        ).first()
        if interaction:
            details['user_interaction'] = {
                'type': interaction.interaction_type,
                'value': interaction.value
            }
    
    return jsonify(details)

# Search functionality
@app.route('/api/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify({'results': []})
    
    results = []
    
    # Search TMDB
    tmdb_results = fetch_tmdb_content('search/multi', {'query': query})
    if tmdb_results:
        for item in tmdb_results['results'][:10]:
            if item['media_type'] in ['movie', 'tv']:
                content = cache_content(item, 'tmdb', item['media_type'])
                results.append({
                    'id': content.id,
                    'title': content.title,
                    'overview': content.overview,
                    'poster_path': content.poster_path,
                    'rating': content.rating,
                    'type': content.content_type,
                    'source': 'tmdb'
                })
    
    # Search Jikan for anime
    jikan_results = fetch_jikan_content('anime', {'q': query, 'limit': 10})
    if jikan_results:
        for anime in jikan_results['data']:
            content = cache_content(anime, 'jikan', 'anime')
            results.append({
                'id': content.id,
                'title': content.title,
                'overview': content.overview,
                'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                'rating': content.rating,
                'type': 'anime',
                'source': 'jikan'
            })
    
    # Track search for anonymous users
    session_id = get_session_id()
    anon_session = AnonymousSession.query.filter_by(session_id=session_id).first()
    if anon_session:
        history = json.loads(anon_session.search_history) if anon_session.search_history else []
        history.append({'query': query, 'timestamp': datetime.utcnow().isoformat()})
        anon_session.search_history = json.dumps(history[-20:])  # Keep last 20 searches
        anon_session.last_active = datetime.utcnow()
        db.session.commit()
    
    return jsonify({'results': results})

# Trending content
@app.route('/api/trending')
def trending():
    region = request.args.get('region', get_user_location())
    time_window = request.args.get('time_window', 'day')
    
    trending_data = {}
    
    # TMDB trending
    tmdb_trending = fetch_tmdb_content(f'trending/all/{time_window}')
    if tmdb_trending:
        trending_data['all'] = []
        for item in tmdb_trending['results']:
            content_type = 'movie' if item['media_type'] == 'movie' else 'tv'
            content = cache_content(item, 'tmdb', content_type)
            trending_data['all'].append({
                'id': content.id,
                'title': content.title,
                'overview': content.overview,
                'poster_path': content.poster_path,
                'rating': content.rating,
                'type': content_type
            })
    
    # Trending anime
    jikan_trending = fetch_jikan_content('top/anime', {'limit': 10, 'filter': 'airing'})
    if jikan_trending:
        trending_data['anime'] = []
        for anime in jikan_trending['data']:
            content = cache_content(anime, 'jikan', 'anime')
            trending_data['anime'].append({
                'id': content.id,
                'title': content.title,
                'overview': content.overview,
                'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                'rating': content.rating,
                'type': 'anime'
            })
    
    return jsonify(trending_data)

# User interactions
@app.route('/api/user/interaction', methods=['POST'])
@require_auth
def user_interaction():
    data = request.get_json()
    content_id = data.get('content_id')
    interaction_type = data.get('type')  # rating, watchlist, favorite, viewed
    value = data.get('value')
    
    # Remove existing interaction of same type
    existing = UserInteraction.query.filter_by(
        user_id=g.user.id,
        content_id=content_id,
        interaction_type=interaction_type
    ).first()
    
    if existing:
        if interaction_type in ['watchlist', 'favorite'] and existing:
            # Toggle behavior for watchlist and favorites
            db.session.delete(existing)
            db.session.commit()
            return jsonify({'message': 'Removed from ' + interaction_type})
        else:
            existing.value = value
            existing.timestamp = datetime.utcnow()
    else:
        interaction = UserInteraction(
            user_id=g.user.id,
            content_id=content_id,
            interaction_type=interaction_type,
            value=value
        )
        db.session.add(interaction)
    
    db.session.commit()
    
    # Send interaction to ML service for learning
    try:
        requests.post(f"{ML_SERVICE_URL}/learn", json={
            'user_id': g.user.id,
            'content_id': content_id,
            'interaction_type': interaction_type,
            'value': value
        }, timeout=2)
    except:
        pass  # ML service is optional
    
    return jsonify({'message': 'Interaction recorded'})

# Admin routes
@app.route('/api/admin/search-content')
@require_admin
def admin_search_content():
    query = request.args.get('q', '')
    source = request.args.get('source', 'tmdb')
    
    if not query:
        return jsonify({'results': []})
    
    results = []
    
    if source == 'tmdb':
        tmdb_results = fetch_tmdb_content('search/multi', {'query': query})
        if tmdb_results:
            for item in tmdb_results['results']:
                if item['media_type'] in ['movie', 'tv']:
                    results.append({
                        'external_id': item['id'],
                        'title': item.get('title', item.get('name', '')),
                        'overview': item.get('overview', ''),
                        'poster_path': item.get('poster_path'),
                        'release_date': item.get('release_date', item.get('first_air_date')),
                        'rating': item.get('vote_average', 0),
                        'media_type': item['media_type']
                    })
    
    elif source == 'jikan':
        jikan_results = fetch_jikan_content('anime', {'q': query, 'limit': 20})
        if jikan_results:
            for anime in jikan_results['data']:
                results.append({
                    'external_id': anime['mal_id'],
                    'title': anime['title'],
                    'overview': anime.get('synopsis', ''),
                    'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                    'rating': anime.get('score', 0),
                    'media_type': 'anime'
                })
    
    return jsonify({'results': results})

@app.route('/api/admin/curate-content', methods=['POST'])
@require_admin
def curate_content():
    data = request.get_json()
    
    # Cache the content first
    content_data = {
        'id': data['external_id'],
        'title': data['title'],
        'overview': data['overview'],
        'poster_path': data['poster_path'],
        'vote_average': data['rating']
    }
    
    content = cache_content(content_data, data['source'], data['media_type'])
    
    # Create admin recommendation
    admin_rec = AdminRecommendation(
        admin_id=g.user.id,
        content_id=content.id,
        title=data['title'],
        description=data.get('description', ''),
        category=data.get('category', 'featured'),
        languages=json.dumps(data.get('languages', [])),
        priority=data.get('priority', 1),
        expires_at=datetime.utcnow() + timedelta(days=data.get('expires_in_days', 30))
    )
    
    db.session.add(admin_rec)
    db.session.commit()
    
    return jsonify({'message': 'Content curated successfully'})

@app.route('/api/admin/recommendations')
@require_admin
def admin_recommendations():
    recs = AdminRecommendation.query.filter_by(admin_id=g.user.id).order_by(
        AdminRecommendation.created_at.desc()
    ).all()
    
    result = []
    for rec in recs:
        content = Content.query.get(rec.content_id)
        result.append({
            'id': rec.id,
            'title': rec.title,
            'description': rec.description,
            'category': rec.category,
            'priority': rec.priority,
            'is_active': rec.is_active,
            'expires_at': rec.expires_at.isoformat() if rec.expires_at else None,
            'created_at': rec.created_at.isoformat(),
            'content': {
                'title': content.title,
                'poster_path': content.poster_path
            } if content else None
        })
    
    return jsonify({'recommendations': result})

# Regional content
@app.route('/api/regional-content')
def regional_content():
    region = request.args.get('region', get_user_location())
    language = request.args.get('language', 'en')
    
    content = {}
    
    # Regional movies
    regional_movies = fetch_tmdb_content('discover/movie', {
        'region': region,
        'with_original_language': language,
        'sort_by': 'popularity.desc'
    })
    
    if regional_movies:
        content['movies'] = []
        for movie in regional_movies['results'][:10]:
            cached = cache_content(movie, 'tmdb', 'movie')
            content['movies'].append({
                'id': cached.id,
                'title': cached.title,
                'overview': cached.overview,
                'poster_path': cached.poster_path,
                'rating': cached.rating
            })
    
    # Regional TV shows
    regional_tv = fetch_tmdb_content('discover/tv', {
        'region': region,
        'with_original_language': language,
        'sort_by': 'popularity.desc'
    })
    
    if regional_tv:
        content['tv_shows'] = []
        for show in regional_tv['results'][:10]:
            cached = cache_content(show, 'tmdb', 'tv')
            content['tv_shows'].append({
                'id': cached.id,
                'title': cached.title,
                'overview': cached.overview,
                'poster_path': cached.poster_path,
                'rating': cached.rating
            })
    
    return jsonify(content)

# Health check
@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'tmdb': bool(TMDB_API_KEY),
            'ml_service': bool(ML_SERVICE_URL)
        }
    })

# User profile
@app.route('/api/user/profile')
@require_auth
def user_profile():
    interactions = UserInteraction.query.filter_by(user_id=g.user.id).all()
    
    profile = {
        'id': g.user.id,
        'username': g.user.username,
        'email': g.user.email,
        'region': g.user.region,
        'preferred_languages': json.loads(g.user.preferred_languages) if g.user.preferred_languages else [],
        'created_at': g.user.created_at.isoformat(),
        'stats': {
            'ratings_count': len([i for i in interactions if i.interaction_type == 'rating']),
            'watchlist_count': len([i for i in interactions if i.interaction_type == 'watchlist']),
            'favorites_count': len([i for i in interactions if i.interaction_type == 'favorite']),
            'viewed_count': len([i for i in interactions if i.interaction_type == 'viewed'])
        }
    }
    
    return jsonify(profile)

@app.route('/api/user/profile', methods=['PUT'])
@require_auth
def update_profile():
    data = request.get_json()
    
    if 'preferred_languages' in data:
        g.user.preferred_languages = json.dumps(data['preferred_languages'])
    
    if 'region' in data:
        g.user.region = data['region']
    
    db.session.commit()
    return jsonify({'message': 'Profile updated successfully'})

# User lists
@app.route('/api/user/watchlist')
@require_auth
def user_watchlist():
    interactions = UserInteraction.query.filter_by(
        user_id=g.user.id,
        interaction_type='watchlist'
    ).all()
    
    watchlist = []
    for interaction in interactions:
        content = Content.query.get(interaction.content_id)
        if content:
            watchlist.append({
                'id': content.id,
                'title': content.title,
                'overview': content.overview,
                'poster_path': content.poster_path,
                'rating': content.rating,
                'type': content.content_type,
                'added_at': interaction.timestamp.isoformat()
            })
    
    return jsonify({'watchlist': watchlist})

@app.route('/api/user/favorites')
@require_auth
def user_favorites():
    interactions = UserInteraction.query.filter_by(
        user_id=g.user.id,
        interaction_type='favorite'
    ).all()
    
    favorites = []
    for interaction in interactions:
        content = Content.query.get(interaction.content_id)
        if content:
            favorites.append({
                'id': content.id,
                'title': content.title,
                'overview': content.overview,
                'poster_path': content.poster_path,
                'rating': content.rating,
                'type': content.content_type,
                'added_at': interaction.timestamp.isoformat()
            })
    
    return jsonify({'favorites': favorites})

# Initialize database
@app.before_first_request
def create_tables():
    db.create_all()

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)