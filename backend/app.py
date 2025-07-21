#backend/app.py

from flask import Flask, request, jsonify, session, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import requests
import os
import json
import logging
from functools import wraps
import sqlite3
from collections import defaultdict, Counter
import random
import hashlib
import time
from sqlalchemy import func, and_, or_, desc, text
import telebot
import threading
from geopy.geocoders import Nominatim
import jwt

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Database configuration
if os.environ.get('DATABASE_URL'):
    # Production - PostgreSQL on Render
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    # Local development - SQLite
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_recommendations.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# API Keys - Set these in your environment
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '52260795')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '-1002850793757')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-xmf5.onrender.com')

# Initialize Telegram bot
if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != 'your_telegram_bot_token':
    try:
        bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
    except:
        bot = None
        logging.warning("Failed to initialize Telegram bot")
else:
    bot = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    preferred_languages = db.Column(db.Text)  # JSON string
    preferred_genres = db.Column(db.Text)  # JSON string
    location = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    imdb_id = db.Column(db.String(20))
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    content_type = db.Column(db.String(20), nullable=False)  # movie, tv, anime
    genres = db.Column(db.Text)  # JSON string
    languages = db.Column(db.Text)  # JSON string
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float)
    vote_count = db.Column(db.Integer)
    popularity = db.Column(db.Float)
    overview = db.Column(db.Text)
    poster_path = db.Column(db.String(255))
    backdrop_path = db.Column(db.String(255))
    trailer_url = db.Column(db.String(255))
    ott_platforms = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)  # view, like, favorite, watchlist, search
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AdminRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recommendation_type = db.Column(db.String(50))  # trending, popular, critics_choice, admin_choice
    description = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnonymousInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)
    ip_address = db.Column(db.String(45))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Enhanced OTT Platform Information
OTT_PLATFORMS = {
    'netflix': {'name': 'Netflix', 'is_free': False, 'url': 'https://netflix.com'},
    'amazon_prime': {'name': 'Amazon Prime Video', 'is_free': False, 'url': 'https://primevideo.com'},
    'disney_plus': {'name': 'Disney+ Hotstar', 'is_free': False, 'url': 'https://hotstar.com'},
    'youtube': {'name': 'YouTube', 'is_free': True, 'url': 'https://youtube.com'},
    'jiocinema': {'name': 'JioCinema', 'is_free': True, 'url': 'https://jiocinema.com'},
    'mx_player': {'name': 'MX Player', 'is_free': True, 'url': 'https://mxplayer.com'},
    'zee5': {'name': 'ZEE5', 'is_free': False, 'url': 'https://zee5.com'},
    'sonyliv': {'name': 'SonyLIV', 'is_free': False, 'url': 'https://sonyliv.com'},
    'voot': {'name': 'Voot', 'is_free': True, 'url': 'https://voot.com'},
    'alt_balaji': {'name': 'ALTBalaji', 'is_free': False, 'url': 'https://altbalaji.com'},
    'aha': {'name': 'Aha', 'is_free': False, 'url': 'https://aha.video'},
    'sun_nxt': {'name': 'Sun NXT', 'is_free': False, 'url': 'https://sunnxt.com'},
    'eros_now': {'name': 'Eros Now', 'is_free': False, 'url': 'https://erosnow.com'},
    'hoichoi': {'name': 'Hoichoi', 'is_free': False, 'url': 'https://hoichoi.tv'},
    'addatimes': {'name': 'Addatimes', 'is_free': False, 'url': 'https://addatimes.com'}
}

# Enhanced Regional Language Mapping with Priority
REGIONAL_LANGUAGES = {
    'telugu': {
        'codes': ['te', 'telugu'],
        'priority': 1,
        'region': 'IN',
        'industry': 'tollywood'
    },
    'english': {
        'codes': ['en', 'english'],
        'priority': 1,
        'region': 'US',
        'industry': 'hollywood'
    },
    'hindi': {
        'codes': ['hi', 'hindi'],
        'priority': 2,
        'region': 'IN',
        'industry': 'bollywood'
    },
    'tamil': {
        'codes': ['ta', 'tamil'],
        'priority': 2,
        'region': 'IN',
        'industry': 'kollywood'
    },
    'malayalam': {
        'codes': ['ml', 'malayalam'],
        'priority': 2,
        'region': 'IN',
        'industry': 'mollywood'
    },
    'kannada': {
        'codes': ['kn', 'kannada'],
        'priority': 2,
        'region': 'IN',
        'industry': 'sandalwood'
    }
}

# Helper Functions
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

def require_admin(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user or not current_user.is_admin:
                return jsonify({'error': 'Admin access required'}), 403
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = hashlib.md5(f"{request.remote_addr}{time.time()}".encode()).hexdigest()
    return session['session_id']

def get_user_location(ip_address):
    try:
        response = requests.get(f'http://ip-api.com/json/{ip_address}', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return {
                    'country': data.get('country'),
                    'region': data.get('regionName'),
                    'city': data.get('city'),
                    'lat': data.get('lat'),
                    'lon': data.get('lon')
                }
    except:
        pass
    return None

# External API Services
class TMDBService:
    BASE_URL = 'https://api.themoviedb.org/3'
    
    @staticmethod
    def search_content(query, content_type='multi', language='en-US', page=1, region=None):
        url = f"{TMDBService.BASE_URL}/search/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'language': language,
            'page': page
        }
        if region:
            params['region'] = region
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB search error: {e}")
        return None
    
    @staticmethod
    def get_content_details(content_id, content_type='movie'):
        url = f"{TMDBService.BASE_URL}/{content_type}/{content_id}"
        params = {
            'api_key': TMDB_API_KEY,
            'append_to_response': 'credits,videos,similar,watch/providers'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB details error: {e}")
        return None
    
    @staticmethod
    def get_trending(content_type='all', time_window='day', page=1):
        url = f"{TMDBService.BASE_URL}/trending/{content_type}/{time_window}"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB trending error: {e}")
        return None
    
    @staticmethod
    def get_popular(content_type='movie', page=1, region=None):
        url = f"{TMDBService.BASE_URL}/{content_type}/popular"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        if region:
            params['region'] = region
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB popular error: {e}")
        return None
    
    @staticmethod
    def get_now_playing(content_type='movie', page=1, region=None):
        url = f"{TMDBService.BASE_URL}/{content_type}/now_playing"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        if region:
            params['region'] = region
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB now playing error: {e}")
        return None
    
    @staticmethod
    def get_top_rated(content_type='movie', page=1, region=None):
        url = f"{TMDBService.BASE_URL}/{content_type}/top_rated"
        params = {
            'api_key': TMDB_API_KEY,
            'page': page
        }
        if region:
            params['region'] = region
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB top rated error: {e}")
        return None
    
    @staticmethod
    def discover_content(content_type='movie', **kwargs):
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            **kwargs
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB discover error: {e}")
        return None

class OMDbService:
    BASE_URL = 'http://www.omdbapi.com/'
    
    @staticmethod
    def get_content_by_imdb(imdb_id):
        params = {
            'apikey': OMDB_API_KEY,
            'i': imdb_id,
            'plot': 'full'
        }
        
        try:
            response = requests.get(OMDbService.BASE_URL, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"OMDb error: {e}")
        return None

class JikanService:
    BASE_URL = 'https://api.jikan.moe/v4'
    
    @staticmethod
    def search_anime(query, page=1):
        url = f"{JikanService.BASE_URL}/anime"
        params = {
            'q': query,
            'page': page,
            'limit': 20
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"Jikan search error: {e}")
        return None
    
    @staticmethod
    def get_anime_details(anime_id):
        url = f"{JikanService.BASE_URL}/anime/{anime_id}/full"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"Jikan anime details error: {e}")
        return None
    
    @staticmethod
    def get_top_anime(type='tv', page=1):
        url = f"{JikanService.BASE_URL}/top/anime"
        params = {
            'type': type,
            'page': page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"Jikan top anime error: {e}")
        return None

class YouTubeService:
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    @staticmethod
    def search_trailers(query):
        if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == 'your_youtube_api_key':
            return None
            
        url = f"{YouTubeService.BASE_URL}/search"
        params = {
            'key': YOUTUBE_API_KEY,
            'q': f"{query} trailer",
            'part': 'snippet',
            'type': 'video',
            'maxResults': 5
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
        return None

# Enhanced Content Management Service
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                return existing
            
            genres = []
            if 'genres' in tmdb_data:
                genres = [genre['name'] for genre in tmdb_data['genres']]
            elif 'genre_ids' in tmdb_data:
                genres = ContentService.map_genre_ids(tmdb_data['genre_ids'])
            
            languages = []
            if 'spoken_languages' in tmdb_data:
                languages = [lang['name'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
            
            ott_platforms = ContentService.get_ott_availability(tmdb_data)
            
            # Handle release date parsing
            release_date = None
            date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
            if date_str:
                try:
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                except:
                    pass
            
            content = Content(
                tmdb_id=tmdb_data['id'],
                title=tmdb_data.get('title') or tmdb_data.get('name'),
                original_title=tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                content_type=content_type,
                genres=json.dumps(genres),
                languages=json.dumps(languages),
                release_date=release_date,
                runtime=tmdb_data.get('runtime'),
                rating=tmdb_data.get('vote_average'),
                vote_count=tmdb_data.get('vote_count'),
                popularity=tmdb_data.get('popularity'),
                overview=tmdb_data.get('overview'),
                poster_path=tmdb_data.get('poster_path'),
                backdrop_path=tmdb_data.get('backdrop_path'),
                ott_platforms=json.dumps(ott_platforms)
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def save_anime_content(anime_data):
        try:
            existing = Content.query.filter_by(tmdb_id=anime_data['mal_id']).first()
            if existing:
                return existing
            
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            # Handle anime release date
            release_date = None
            aired_from = anime_data.get('aired', {}).get('from')
            if aired_from:
                try:
                    release_date = datetime.strptime(aired_from[:10], '%Y-%m-%d').date()
                except:
                    pass
            
            content = Content(
                tmdb_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                languages=json.dumps(['japanese']),
                release_date=release_date,
                runtime=anime_data.get('duration'),
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                ott_platforms=json.dumps(ContentService.get_anime_ott_platforms())
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def map_genre_ids(genre_ids):
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western',
            10759: 'Action & Adventure', 10762: 'Kids', 10763: 'News',
            10764: 'Reality', 10765: 'Sci-Fi & Fantasy', 10766: 'Soap',
            10767: 'Talk', 10768: 'War & Politics'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]
    
    @staticmethod
    def get_ott_availability(tmdb_data):
        platforms = []
        
        # Enhanced OTT detection based on content language and region
        content_language = tmdb_data.get('original_language', 'en')
        
        if content_language in ['te', 'hi', 'ta', 'ml', 'kn']:
            # Indian regional content
            regional_platforms = ['aha', 'sun_nxt', 'zee5', 'sonyliv', 'jiocinema']
            selected_platforms = random.sample(regional_platforms, random.randint(1, 3))
        else:
            # Global content
            selected_platforms = random.sample(['netflix', 'amazon_prime', 'disney_plus'], random.randint(1, 2))
        
        for platform in selected_platforms:
            platforms.append({
                'platform': platform,
                'name': OTT_PLATFORMS[platform]['name'],
                'url': OTT_PLATFORMS[platform]['url'],
                'is_free': OTT_PLATFORMS[platform]['is_free']
            })
        
        return platforms
    
    @staticmethod
    def get_anime_ott_platforms():
        anime_platforms = ['youtube', 'netflix', 'jiocinema']
        selected = random.sample(anime_platforms, random.randint(1, 2))
        return [{
            'platform': platform,
            'name': OTT_PLATFORMS[platform]['name'],
            'url': OTT_PLATFORMS[platform]['url'],
            'is_free': OTT_PLATFORMS[platform]['is_free']
        } for platform in selected]

# Enhanced Recommendation Engine
class RecommendationEngine:
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all', language_priority=True):
        try:
            trending_data = TMDBService.get_trending(content_type=content_type)
            if not trending_data:
                return []
            
            recommendations = []
            prioritized_content = []
            other_content = []
            
            for item in trending_data.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    # Check if content is in priority languages (Telugu/English)
                    content_languages = json.loads(content.languages or '[]')
                    is_priority = any(lang.lower() in ['telugu', 'english', 'te', 'en'] for lang in content_languages)
                    
                    if is_priority:
                        prioritized_content.append(content)
                    else:
                        other_content.append(content)
            
            # Combine prioritized content first
            if language_priority:
                recommendations = prioritized_content[:limit//2] + other_content[:limit//2]
            else:
                recommendations = prioritized_content + other_content
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_new_releases(limit=20, days_back=60):
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Get recent releases from database
            recent_content = Content.query.filter(
                Content.release_date >= cutoff_date.date()
            ).order_by(Content.release_date.desc()).limit(limit).all()
            
            # If not enough, fetch from TMDB
            if len(recent_content) < limit:
                now_playing = TMDBService.get_now_playing('movie', region='IN')
                if now_playing:
                    for item in now_playing.get('results', []):
                        if len(recent_content) >= limit:
                            break
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content and content not in recent_content:
                            recent_content.append(content)
            
            return recent_content[:limit]
            
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return []
    
    @staticmethod
    def get_best_movies(limit=20):
        try:
            # Get top rated from TMDB
            top_rated_movies = TMDBService.get_top_rated('movie', region='IN')
            top_rated_tv = TMDBService.get_top_rated('tv', region='IN')
            
            recommendations = []
            
            # Process movies
            if top_rated_movies:
                for item in top_rated_movies.get('results', []):
                    if len(recommendations) >= limit:
                        break
                    content = ContentService.save_content_from_tmdb(item, 'movie')
                    if content:
                        recommendations.append(content)
            
            # Process TV shows
            if top_rated_tv and len(recommendations) < limit:
                for item in top_rated_tv.get('results', []):
                    if len(recommendations) >= limit:
                        break
                    content = ContentService.save_content_from_tmdb(item, 'tv')
                    if content:
                        recommendations.append(content)
            
            # Sort by rating
            recommendations.sort(key=lambda x: x.rating or 0, reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting best movies: {e}")
            return []
    
    @staticmethod
    def get_critics_choice(limit=20):
        try:
            # Get admin recommendations marked as critics choice
            admin_recs = AdminRecommendation.query.filter_by(
                recommendation_type='critics_choice',
                is_active=True
            ).order_by(AdminRecommendation.created_at.desc()).limit(limit).all()
            
            recommendations = []
            for rec in admin_recs:
                content = Content.query.get(rec.content_id)
                if content:
                    recommendations.append(content)
            
            # If not enough admin recommendations, supplement with high-rated content
            if len(recommendations) < limit:
                high_rated = Content.query.filter(
                    Content.rating >= 8.0,
                    Content.vote_count >= 1000
                ).order_by(Content.rating.desc()).limit(limit - len(recommendations)).all()
                
                for content in high_rated:
                    if content not in recommendations:
                        recommendations.append(content)
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting critics choice: {e}")
            return []
    
    @staticmethod
    def get_genre_recommendations(genre, limit=20, language_priority=True):
        try:
            # Search for content by genre with language priority
            db_content = Content.query.filter(
                Content.genres.contains(f'"{genre}"')
            ).order_by(Content.rating.desc()).limit(limit * 2).all()
            
            # Filter and prioritize
            prioritized = []
            other = []
            
            for content in db_content:
                languages = json.loads(content.languages or '[]')
                is_priority = any(lang.lower() in ['telugu', 'english', 'te', 'en'] for lang in languages)
                
                if is_priority:
                    prioritized.append(content)
                else:
                    other.append(content)
            
            if language_priority:
                recommendations = prioritized[:limit//2] + other[:limit//2]
            else:
                recommendations = prioritized + other
            
            # If not enough, fetch from TMDB
            if len(recommendations) < limit:
                genre_map = {v: k for k, v in {
                    28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
                    80: 'Crime', 99: 'Documentary', 18: 'Drama', 14: 'Fantasy',
                    27: 'Horror', 9648: 'Mystery', 10749: 'Romance', 878: 'Sci-Fi',
                    53: 'Thriller', 37: 'Western'
                }.items()}
                
                genre_id = genre_map.get(genre)
                if genre_id:
                    discover_data = TMDBService.discover_content(
                        content_type='movie',
                        with_genres=genre_id,
                        sort_by='popularity.desc',
                        page=1
                    )
                    
                    if discover_data:
                        for item in discover_data.get('results', []):
                            if len(recommendations) >= limit:
                                break
                            content = ContentService.save_content_from_tmdb(item, 'movie')
                            if content and content not in recommendations:
                                recommendations.append(content)
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting genre recommendations: {e}")
            return []
    
    @staticmethod
    def get_popular_by_genre(genre, limit=20, region=None):
        try:
            popular_movies = TMDBService.get_popular('movie', region=region)
            popular_tv = TMDBService.get_popular('tv', region=region)
            
            recommendations = []
            
            # Process movies
            if popular_movies:
                for item in popular_movies.get('results', []):
                    if genre.lower() in [g.lower() for g in ContentService.map_genre_ids(item.get('genre_ids', []))]:
                        content = ContentService.save_content_from_tmdb(item, 'movie')
                        if content:
                            recommendations.append(content)
            
            # Process TV shows
            if popular_tv:
                for item in popular_tv.get('results', []):
                    if genre.lower() in [g.lower() for g in ContentService.map_genre_ids(item.get('genre_ids', []))]:
                        content = ContentService.save_content_from_tmdb(item, 'tv')
                        if content:
                            recommendations.append(content)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting popular by genre: {e}")
            return []
    
    @staticmethod
    def get_regional_recommendations(language, limit=20):
        try:
            search_queries = {
                'hindi': ['bollywood', 'hindi movie', 'hindi film'],
                'telugu': ['tollywood', 'telugu movie', 'telugu film'],
                'tamil': ['kollywood', 'tamil movie', 'tamil film'],
                'kannada': ['sandalwood', 'kannada movie', 'kannada film'],
                'malayalam': ['mollywood', 'malayalam movie', 'malayalam film'],
                'english': ['hollywood', 'english movie', 'english film']
            }
            
            recommendations = []
            queries = search_queries.get(language.lower(), [language])
            
            for query in queries:
                search_results = TMDBService.search_content(query, region='IN')
                if search_results:
                    for item in search_results.get('results', []):
                        content_type_detected = 'movie' if 'title' in item else 'tv'
                        content = ContentService.save_content_from_tmdb(item, content_type_detected)
                        if content:
                            recommendations.append(content)
                        
                        if len(recommendations) >= limit:
                            break
                
                if len(recommendations) >= limit:
                    break
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting regional recommendations: {e}")
            return []
    
    @staticmethod
    def get_anime_recommendations(limit=20):
        try:
            top_anime = JikanService.get_top_anime()
            if not top_anime:
                return []
            
            recommendations = []
            for anime in top_anime.get('data', [])[:limit]:
                content = ContentService.save_anime_content(anime)
                if content:
                    recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting anime recommendations: {e}")
            return []

# Anonymous User Recommendations
class AnonymousRecommendationEngine:
    @staticmethod
    def get_recommendations_for_anonymous(session_id, ip_address, limit=20):
        try:
            location = get_user_location(ip_address)
            
            interactions = AnonymousInteraction.query.filter_by(session_id=session_id).all()
            
            recommendations = []
            
            if interactions:
                viewed_content_ids = [interaction.content_id for interaction in interactions]
                viewed_contents = Content.query.filter(Content.id.in_(viewed_content_ids)).all()
                
                all_genres = []
                for content in viewed_contents:
                    if content.genres:
                        all_genres.extend(json.loads(content.genres))
                
                genre_counts = Counter(all_genres)
                top_genres = [genre for genre, _ in genre_counts.most_common(3)]
                
                for genre in top_genres:
                    genre_recs = RecommendationEngine.get_popular_by_genre(genre, limit=7)
                    recommendations.extend(genre_recs)
            
            # Add regional content based on location
            if location and location.get('country') == 'India':
                regional_recs = RecommendationEngine.get_regional_recommendations('hindi', limit=5)
                recommendations.extend(regional_recs)
            
            # Add trending content
            trending_recs = RecommendationEngine.get_trending_recommendations(limit=10)
            recommendations.extend(trending_recs)
            
            # Remove duplicates and limit
            seen_ids = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec.id not in seen_ids:
                    seen_ids.add(rec.id)
                    unique_recommendations.append(rec)
                    if len(unique_recommendations) >= limit:
                        break
            
            return unique_recommendations
        except Exception as e:
            logger.error(f"Error getting anonymous recommendations: {e}")
            return []

# Telegram Service
class TelegramService:
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

#AdminChoice #MovieRecommendation #CineScope"""
            
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='Markdown'
                    )
                except Exception as photo_error:
                    logger.error(f"Failed to send photo, sending text only: {photo_error}")
                    bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            else:
                bot.send_message(TELEGRAM_CHANNEL_ID, message, parse_mode='Markdown')
            
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

# API Routes

# Authentication Routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        if not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password']),
            preferred_languages=json.dumps(data.get('preferred_languages', [])),
            preferred_genres=json.dumps(data.get('preferred_genres', []))
        )
        
        db.session.add(user)
        db.session.commit()
        
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'message': 'User registered successfully',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Missing username or password'}), 400
        
        user = User.query.filter_by(username=data['username']).first()
        
        if not user or not check_password_hash(user.password_hash, data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=30)
        }, app.secret_key, algorithm='HS256')
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'preferred_languages': json.loads(user.preferred_languages or '[]'),
                'preferred_genres': json.loads(user.preferred_genres or '[]')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

# Enhanced Content Discovery Routes
@app.route('/api/search', methods=['GET'])
def search_content():
    try:
        query = request.args.get('query', '')
        content_type = request.args.get('type', 'multi')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        session_id = get_session_id()
        results = []
        
        # Search TMDB
        if content_type in ['multi', 'movie', 'tv']:
            tmdb_results = TMDBService.search_content(query, content_type, page=page)
            
            if tmdb_results:
                for item in tmdb_results.get('results', []):
                    content_type_detected = 'movie' if 'title' in item else 'tv'
                    content = ContentService.save_content_from_tmdb(item, content_type_detected)
                    if content:
                        interaction = AnonymousInteraction(
                            session_id=session_id,
                            content_id=content.id,
                            interaction_type='search',
                            ip_address=request.remote_addr
                        )
                        db.session.add(interaction)
                        
                        results.append({
                            'id': content.id,
                            'tmdb_id': content.tmdb_id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'release_date': content.release_date.isoformat() if content.release_date else None,
                            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
                            'overview': content.overview
                        })
        
        # Search anime
        if content_type in ['anime', 'multi']:
            anime_results = JikanService.search_anime(query, page=page)
            
            if anime_results:
                for anime in anime_results.get('data', []):
                    content = ContentService.save_anime_content(anime)
                    if content:
                        interaction = AnonymousInteraction(
                            session_id=session_id,
                            content_id=content.id,
                            interaction_type='search',
                            ip_address=request.remote_addr
                        )
                        db.session.add(interaction)
                        
                        results.append({
                            'id': content.id,
                            'tmdb_id': content.tmdb_id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'release_date': content.release_date.isoformat() if content.release_date else None,
                            'poster_path': content.poster_path,
                            'overview': content.overview
                        })
        
        db.session.commit()
        
        return jsonify({
            'results': results,
            'total_results': len(results),
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

# Enhanced Content Details Route - Returns only OTT platforms when requested
@app.route('/api/content/<int:content_id>', methods=['GET'])
def get_content_details(content_id):
    try:
        content = Content.query.get_or_404(content_id)
        ott_only = request.args.get('ott_only', 'false').lower() == 'true'
        
        # Record view interaction
        session_id = get_session_id()
        interaction = AnonymousInteraction(
            session_id=session_id,
            content_id=content.id,
            interaction_type='view',
            ip_address=request.remote_addr
        )
        db.session.add(interaction)
        
        # If only OTT platforms requested
        if ott_only:
            db.session.commit()
            return jsonify({
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            }), 200
        
        # Full content details
        additional_details = None
        if content.tmdb_id and content.content_type != 'anime':
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        elif content.content_type == 'anime':
            additional_details = JikanService.get_anime_details(content.tmdb_id)
        
        # Get YouTube trailers
        trailers = []
        if YOUTUBE_API_KEY and YOUTUBE_API_KEY != 'your_youtube_api_key':
            youtube_results = YouTubeService.search_trailers(content.title)
            if youtube_results:
                for video in youtube_results.get('items', []):
                    trailers.append({
                        'title': video['snippet']['title'],
                        'url': f"https://www.youtube.com/watch?v={video['id']['videoId']}",
                        'thumbnail': video['snippet']['thumbnails']['medium']['url']
                    })
        
        # Get similar content
        similar_content = []
        if additional_details:
            if content.content_type == 'anime' and 'data' in additional_details:
                pass  # Anime API doesn't provide similar content in the same call
            elif 'similar' in additional_details:
                for item in additional_details['similar']['results'][:5]:
                    similar = ContentService.save_content_from_tmdb(item, content.content_type)
                    if similar:
                        similar_content.append({
                            'id': similar.id,
                            'title': similar.title,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{similar.poster_path}" if similar.poster_path else None,
                            'rating': similar.rating
                        })
        
        db.session.commit()
        
        return jsonify({
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'runtime': content.runtime,
            'rating': content.rating,
            'vote_count': content.vote_count,
            'overview': content.overview,
            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path else None,
            'ott_platforms': json.loads(content.ott_platforms or '[]'),
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details and 'credits' in additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details and 'credits' in additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Enhanced Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        language_priority = request.args.get('language_priority', 'true').lower() == 'true'
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type, language_priority)
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/recommendations/new-releases', methods=['GET'])
def get_new_releases():
    try:
        limit = int(request.args.get('limit', 20))
        days_back = int(request.args.get('days_back', 60))
        
        recommendations = RecommendationEngine.get_new_releases(limit, days_back)
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"New releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

@app.route('/api/recommendations/best-movies', methods=['GET'])
def get_best_movies():
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_best_movies(limit)
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Best movies error: {e}")
        return jsonify({'error': 'Failed to get best movies'}), 500

@app.route('/api/recommendations/critics-choice', methods=['GET'])
def get_critics_choice():
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_critics_choice(limit)
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Critics choice error: {e}")
        return jsonify({'error': 'Failed to get critics choice'}), 500

@app.route('/api/recommendations/genre/<genre>', methods=['GET'])
def get_genre_recommendations(genre):
    try:
        limit = int(request.args.get('limit', 20))
        language_priority = request.args.get('language_priority', 'true').lower() == 'true'
        
        recommendations = RecommendationEngine.get_genre_recommendations(genre, limit, language_priority)
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Genre recommendations error: {e}")
        return jsonify({'error': 'Failed to get genre recommendations'}), 500

@app.route('/api/recommendations/popular/<genre>', methods=['GET'])
def get_popular_by_genre(genre):
    try:
        limit = int(request.args.get('limit', 20))
        region = request.args.get('region')
        
        recommendations = RecommendationEngine.get_popular_by_genre(genre, limit, region)
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Popular by genre error: {e}")
        return jsonify({'error': 'Failed to get popular recommendations'}), 500

@app.route('/api/recommendations/regional/<language>', methods=['GET'])
def get_regional(language):
    try:
        limit = int(request.args.get('limit', 20))
        
        # Enhanced regional search with multiple queries
        search_terms = {
            'telugu': ['telugu movie', 'tollywood', 'telugu film', 'telugu cinema'],
            'tamil': ['tamil movie', 'kollywood', 'tamil film', 'tamil cinema'],
            'hindi': ['hindi movie', 'bollywood', 'hindi film', 'hindi cinema'],
            'malayalam': ['malayalam movie', 'mollywood', 'malayalam film'],
            'kannada': ['kannada movie', 'sandalwood', 'kannada film'],
            'english': ['hollywood', 'english movie', 'english film']
        }
        
        recommendations = []
        terms = search_terms.get(language.lower(), [language])
        
        for term in terms:
            search_results = TMDBService.search_content(term, region='IN')
            if search_results:
                for item in search_results.get('results', []):
                    if len(recommendations) >= limit:
                        break
                    content_type_detected = 'movie' if 'title' in item else 'tv'
                    content = ContentService.save_content_from_tmdb(item, content_type_detected)
                    if content and content not in recommendations:
                        recommendations.append(content)
            
            if len(recommendations) >= limit:
                break
        
        result = []
        for content in recommendations[:limit]:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Regional recommendations error: {e}")
        return jsonify({'error': 'Failed to get regional recommendations'}), 500

@app.route('/api/recommendations/anime', methods=['GET'])
def get_anime():
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_anime_recommendations(limit)
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Anime recommendations error: {e}")
        return jsonify({'error': 'Failed to get anime recommendations'}), 500

@app.route('/api/recommendations/anonymous', methods=['GET'])
def get_anonymous_recommendations():
    try:
        session_id = get_session_id()
        limit = int(request.args.get('limit', 20))
        
        recommendations = AnonymousRecommendationEngine.get_recommendations_for_anonymous(
            session_id, request.remote_addr, limit
        )
        
        result = []
        for content in recommendations:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Anonymous recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

# Personalized Recommendations (requires ML service)
@app.route('/api/recommendations/personalized', methods=['GET'])
@require_auth
def get_personalized_recommendations(current_user):
    try:
        interactions = UserInteraction.query.filter_by(user_id=current_user.id).all()
        
        user_data = {
            'user_id': current_user.id,
            'preferred_languages': json.loads(current_user.preferred_languages or '[]'),
            'preferred_genres': json.loads(current_user.preferred_genres or '[]'),
            'interactions': [
                {
                    'content_id': interaction.content_id,
                    'interaction_type': interaction.interaction_type,
                    'rating': interaction.rating,
                    'timestamp': interaction.timestamp.isoformat()
                }
                for interaction in interactions
            ]
        }
        
        try:
            response = requests.post(f"{ML_SERVICE_URL}/api/recommendations", json=user_data, timeout=30)
            
            if response.status_code == 200:
                ml_recommendations = response.json().get('recommendations', [])
                
                content_ids = [rec['content_id'] for rec in ml_recommendations]
                contents = Content.query.filter(Content.id.in_(content_ids)).all()
                
                result = []
                content_dict = {content.id: content for content in contents}
                
                for rec in ml_recommendations:
                    content = content_dict.get(rec['content_id'])
                    if content:
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'ott_platforms': json.loads(content.ott_platforms or '[]'),
                            'recommendation_score': rec.get('score', 0),
                            'recommendation_reason': rec.get('reason', '')
                        })
                
                return jsonify({'recommendations': result}), 200
        except:
            pass
        
        # Fallback to basic recommendations
        return get_trending()
        
    except Exception as e:
        logger.error(f"Personalized recommendations error: {e}")
        return get_trending()

# User Interaction Routes
@app.route('/api/interactions', methods=['POST'])
@require_auth
def record_interaction(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=data.get('rating')
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        return jsonify({'message': 'Interaction recorded successfully'}), 201
        
    except Exception as e:
        logger.error(f"Interaction recording error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record interaction'}), 500

@app.route('/api/user/watchlist', methods=['GET'])
@require_auth
def get_watchlist(current_user):
    try:
        watchlist_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='watchlist'
        ).all()
        
        content_ids = [interaction.content_id for interaction in watchlist_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        
        result = []
        for content in contents:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            })
        
        return jsonify({'watchlist': result}), 200
        
    except Exception as e:
        logger.error(f"Watchlist error: {e}")
        return jsonify({'error': 'Failed to get watchlist'}), 500

@app.route('/api/user/favorites', methods=['GET'])
@require_auth
def get_favorites(current_user):
    try:
        favorite_interactions = UserInteraction.query.filter_by(
            user_id=current_user.id,
            interaction_type='favorite'
        ).all()
        
        content_ids = [interaction.content_id for interaction in favorite_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        
        result = []
        for content in contents:
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': json.loads(content.ott_platforms or '[]')
            })
        
        return jsonify({'favorites': result}), 200
        
    except Exception as e:
        logger.error(f"Favorites error: {e}")
        return jsonify({'error': 'Failed to get favorites'}), 500

# Admin Routes
@app.route('/api/admin/search', methods=['GET'])
@require_admin
def admin_search(current_user):
    try:
        query = request.args.get('query', '')
        source = request.args.get('source', 'tmdb')  # tmdb, omdb, anime
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        results = []
        
        if source == 'tmdb':
            tmdb_results = TMDBService.search_content(query, page=page)
            if tmdb_results:
                for item in tmdb_results.get('results', []):
                    results.append({
                        'id': item['id'],
                        'title': item.get('title') or item.get('name'),
                        'content_type': 'movie' if 'title' in item else 'tv',
                        'release_date': item.get('release_date') or item.get('first_air_date'),
                        'poster_path': f"https://image.tmdb.org/t/p/w300{item['poster_path']}" if item.get('poster_path') else None,
                        'overview': item.get('overview'),
                        'rating': item.get('vote_average'),
                        'source': 'tmdb'
                    })
        
        elif source == 'anime':
            anime_results = JikanService.search_anime(query, page=page)
            if anime_results:
                for anime in anime_results.get('data', []):
                    results.append({
                        'id': anime['mal_id'],
                        'title': anime.get('title'),
                        'content_type': 'anime',
                        'release_date': anime.get('aired', {}).get('from'),
                        'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                        'overview': anime.get('synopsis'),
                        'rating': anime.get('score'),
                        'source': 'anime'
                    })
        
        return jsonify({'results': results}), 200
        
    except Exception as e:
        logger.error(f"Admin search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/admin/content', methods=['POST'])
@require_admin
def save_external_content(current_user):
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No content data provided'}), 400
        
        existing_content = None
        if data.get('id'):
            existing_content = Content.query.filter_by(tmdb_id=data['id']).first()
        
        if existing_content:
            return jsonify({
                'message': 'Content already exists',
                'content_id': existing_content.id
            }), 200
        
        try:
            release_date = None
            if data.get('release_date'):
                try:
                    release_date = datetime.strptime(data['release_date'], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            content = Content(
                tmdb_id=data.get('id'),
                title=data.get('title'),
                original_title=data.get('original_title'),
                content_type=data.get('content_type', 'movie'),
                genres=json.dumps(data.get('genres', [])),
                languages=json.dumps(data.get('languages', ['en'])),
                release_date=release_date,
                runtime=data.get('runtime'),
                rating=data.get('rating'),
                vote_count=data.get('vote_count'),
                popularity=data.get('popularity'),
                overview=data.get('overview'),
                poster_path=data.get('poster_path'),
                backdrop_path=data.get('backdrop_path'),
                ott_platforms=json.dumps(data.get('ott_platforms', []))
            )
            
            db.session.add(content)
            db.session.commit()
            
            return jsonify({
                'message': 'Content saved successfully',
                'content_id': content.id
            }), 201
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving content: {e}")
            return jsonify({'error': 'Failed to save content to database'}), 500
        
    except Exception as e:
        logger.error(f"Save content error: {e}")
        return jsonify({'error': 'Failed to process content'}), 500

@app.route('/api/admin/recommendations', methods=['POST'])
@require_admin
def create_admin_recommendation(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'recommendation_type', 'description']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        content = Content.query.get(data['content_id'])
        if not content:
            content = Content.query.filter_by(tmdb_id=data['content_id']).first()
        
        if not content:
            return jsonify({'error': 'Content not found. Please save content first.'}), 404
        
        admin_rec = AdminRecommendation(
            content_id=content.id,
            admin_id=current_user.id,
            recommendation_type=data['recommendation_type'],
            description=data['description']
        )
        
        db.session.add(admin_rec)
        db.session.commit()
        
        telegram_success = TelegramService.send_admin_recommendation(content, current_user.username, data['description'])
        
        return jsonify({
            'message': 'Admin recommendation created successfully',
            'telegram_sent': telegram_success
        }), 201
        
    except Exception as e:
        logger.error(f"Admin recommendation error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create recommendation'}), 500

@app.route('/api/admin/recommendations', methods=['GET'])
@require_admin
def get_admin_recommendations(current_user):
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        admin_recs = AdminRecommendation.query.filter_by(is_active=True)\
            .order_by(AdminRecommendation.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        result = []
        for rec in admin_recs.items:
            content = Content.query.get(rec.content_id)
            admin = User.query.get(rec.admin_id)
            
            result.append({
                'id': rec.id,
                'recommendation_type': rec.recommendation_type,
                'description': rec.description,
                'created_at': rec.created_at.isoformat(),
                'admin_name': admin.username if admin else 'Unknown',
                'content': {
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None
                }
            })
        
        return jsonify({
            'recommendations': result,
            'total': admin_recs.total,
            'pages': admin_recs.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Get admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@app.route('/api/admin/analytics', methods=['GET'])
@require_admin
def get_analytics(current_user):
    try:
        total_users = User.query.count()
        total_content = Content.query.count()
        total_interactions = UserInteraction.query.count()
        active_users_last_week = User.query.filter(
            User.last_active >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        popular_content = db.session.query(
            Content.id, Content.title, func.count(UserInteraction.id).label('interaction_count')
        ).join(UserInteraction).group_by(Content.id, Content.title)\
         .order_by(desc('interaction_count')).limit(10).all()
        
        all_interactions = UserInteraction.query.join(Content).all()
        genre_counts = defaultdict(int)
        for interaction in all_interactions:
            content = Content.query.get(interaction.content_id)
            if content and content.genres:
                genres = json.loads(content.genres)
                for genre in genres:
                    genre_counts[genre] += 1
        
        popular_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return jsonify({
            'total_users': total_users,
            'total_content': total_content,
            'total_interactions': total_interactions,
            'active_users_last_week': active_users_last_week,
            'popular_content': [
                {'title': item.title, 'interactions': item.interaction_count}
                for item in popular_content
            ],
            'popular_genres': [
                {'genre': genre, 'count': count}
                for genre, count in popular_genres
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

# Public Admin Recommendations
@app.route('/api/recommendations/admin-choice', methods=['GET'])
def get_public_admin_recommendations():
    try:
        limit = int(request.args.get('limit', 20))
        rec_type = request.args.get('type', 'admin_choice')
        
        admin_recs = AdminRecommendation.query.filter_by(
            is_active=True,
            recommendation_type=rec_type
        ).order_by(AdminRecommendation.created_at.desc()).limit(limit).all()
        
        result = []
        for rec in admin_recs:
            content = Content.query.get(rec.content_id)
            admin = User.query.get(rec.admin_id)
            
            if content:
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'ott_platforms': json.loads(content.ott_platforms or '[]'),
                    'admin_description': rec.description,
                    'admin_name': admin.username if admin else 'Admin',
                    'recommended_at': rec.created_at.isoformat()
                })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Public admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get admin recommendations'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'features': {
            'language_priority': True,
            'ott_only_mode': True,
            'enhanced_anime_support': True,
            'regional_content': True,
            'telugu_english_priority': True
        }
    }), 200

# Initialize database
def create_tables():
    try:
        with app.app_context():
            db.create_all()
            
            admin = User.query.filter_by(username='admin').first()
            if not admin:
                admin = User(
                    username='admin',
                    email='admin@example.com',
                    password_hash=generate_password_hash('admin123'),
                    is_admin=True
                )
                db.session.add(admin)
                db.session.commit()
                logger.info("Admin user created with username: admin, password: admin123")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

create_tables()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)