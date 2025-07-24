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

# Regional Language Mapping
REGIONAL_LANGUAGES = {
    'hindi': ['hi', 'hindi', 'bollywood'],
    'telugu': ['te', 'telugu', 'tollywood'],
    'tamil': ['ta', 'tamil', 'kollywood'],
    'kannada': ['kn', 'kannada', 'sandalwood'],
    'malayalam': ['ml', 'malayalam', 'mollywood'],
    'english': ['en', 'english', 'hollywood']
}

# Anime Genre Mapping
ANIME_GENRES = {
    'shonen': ['Action', 'Adventure', 'Martial Arts', 'School', 'Shounen'],
    'shojo': ['Romance', 'Drama', 'School', 'Slice of Life', 'Shoujo'],
    'seinen': ['Action', 'Drama', 'Thriller', 'Psychological', 'Seinen'],
    'josei': ['Romance', 'Drama', 'Slice of Life', 'Josei'],
    'kodomomuke': ['Kids', 'Family', 'Adventure', 'Comedy']
}

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
    mal_id = db.Column(db.Integer)  # For anime
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    content_type = db.Column(db.String(20), nullable=False)  # movie, tv, anime
    genres = db.Column(db.Text)  # JSON string
    anime_genres = db.Column(db.Text)  # JSON string for anime-specific genres
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
    youtube_trailer_id = db.Column(db.String(255))
    is_trending = db.Column(db.Boolean, default=False)
    is_new_release = db.Column(db.Boolean, default=False)
    is_critics_choice = db.Column(db.Boolean, default=False)
    critics_score = db.Column(db.Float)
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
        # Simple IP-based location detection
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

# ML Service Client
class MLServiceClient:
    """Client for interacting with ML recommendation service"""
    
    @staticmethod
    def call_ml_service(endpoint, params=None, timeout=15):
        """Generic ML service call with error handling"""
        try:
            if not ML_SERVICE_URL:
                return None
                
            url = f"{ML_SERVICE_URL}{endpoint}"
            response = requests.get(url, params=params, timeout=timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"ML service returned {response.status_code} for {endpoint}")
                return None
                
        except Exception as e:
            logger.warning(f"ML service call failed for {endpoint}: {e}")
            return None
    
    @staticmethod
    def process_ml_recommendations(ml_response, limit=20):
        """Process ML service response and get content details from database"""
        try:
            if not ml_response or 'recommendations' not in ml_response:
                return []
            
            recommendations = []
            ml_recs = ml_response['recommendations'][:limit]
            
            # Extract content IDs from ML response
            content_ids = []
            for rec in ml_recs:
                if isinstance(rec, dict) and 'content_id' in rec:
                    content_ids.append(rec['content_id'])
                elif isinstance(rec, int):
                    content_ids.append(rec)
            
            if not content_ids:
                return []
            
            # Get content details from database
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_dict = {content.id: content for content in contents}
            
            # Maintain ML service ordering and add ML scores if available
            for i, rec in enumerate(ml_recs):
                content_id = rec['content_id'] if isinstance(rec, dict) else rec
                content = content_dict.get(content_id)
                
                if content:
                    content_data = {
                        'content': content,
                        'ml_score': rec.get('score', 0) if isinstance(rec, dict) else 0,
                        'ml_reason': rec.get('reason', '') if isinstance(rec, dict) else '',
                        'ml_rank': i + 1
                    }
                    recommendations.append(content_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error processing ML recommendations: {e}")
            return []

# External API Services
class TMDBService:
    BASE_URL = 'https://api.themoviedb.org/3'
    
    @staticmethod
    def search_content(query, content_type='multi', language='en-US', page=1):
        url = f"{TMDBService.BASE_URL}/search/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'language': language,
            'page': page
        }
        
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
            'append_to_response': 'credits,videos,similar,reviews,recommendations'
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
    def get_new_releases(content_type='movie', region=None, page=1):
        """Get content released in the last 60 days"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'primary_release_date.gte': start_date,
            'primary_release_date.lte': end_date,
            'sort_by': 'release_date.desc',
            'page': page
        }
        
        if region:
            params['region'] = region
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB new releases error: {e}")
        return None
    
    @staticmethod
    def get_critics_choice(content_type='movie', page=1):
        """Get highly rated content with significant vote count"""
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'vote_average.gte': 7.5,
            'vote_count.gte': 100,
            'sort_by': 'vote_average.desc',
            'page': page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB critics choice error: {e}")
        return None
    
    @staticmethod
    def get_by_genre(genre_id, content_type='movie', page=1, region=None):
        """Get content by specific genre"""
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_genres': genre_id,
            'sort_by': 'popularity.desc',
            'page': page
        }
        
        if region:
            params['region'] = region
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB genre search error: {e}")
        return None
    
    @staticmethod
    def get_language_specific(language_code, content_type='movie', page=1):
        """Get content in specific language"""
        url = f"{TMDBService.BASE_URL}/discover/{content_type}"
        params = {
            'api_key': TMDB_API_KEY,
            'with_original_language': language_code,
            'sort_by': 'popularity.desc',
            'page': page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB language search error: {e}")
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
        except Exception as e:
            logger.error(f"Jikan search error: {e}")
        return None
    
    @staticmethod
    def get_anime_details(anime_id):
        url = f"{JikanService.BASE_URL}/anime/{anime_id}/full"
        
        try:
            response = requests.get(url, params={}, timeout=10)
            if response.status_code == 200:
                return response.json()
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
        except Exception as e:
            logger.error(f"Jikan top anime error: {e}")
        return None
    
    @staticmethod
    def get_anime_by_genre(genre_name, page=1):
        """Get anime by specific genre"""
        url = f"{JikanService.BASE_URL}/anime"
        params = {
            'genres': genre_name,
            'order_by': 'score',
            'sort': 'desc',
            'page': page
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan genre search error: {e}")
        return None

class YouTubeService:
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    @staticmethod
    def search_trailers(query, content_type='movie'):
        url = f"{YouTubeService.BASE_URL}/search"
        
        # Customize search query based on content type
        if content_type == 'anime':
            search_query = f"{query} anime trailer PV"
        else:
            search_query = f"{query} official trailer"
        
        params = {
            'key': YOUTUBE_API_KEY,
            'q': search_query,
            'part': 'snippet',
            'type': 'video',
            'maxResults': 5,
            'order': 'relevance'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
        return None

# Content Management Service
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            # Check if content already exists
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                return existing
            
            # Extract genres
            genres = []
            if 'genres' in tmdb_data:
                genres = [genre['name'] for genre in tmdb_data['genres']]
            elif 'genre_ids' in tmdb_data:
                genres = ContentService.map_genre_ids(tmdb_data['genre_ids'])
            
            # Extract languages
            languages = []
            if 'spoken_languages' in tmdb_data:
                languages = [lang['name'] for lang in tmdb_data['spoken_languages']]
            elif 'original_language' in tmdb_data:
                languages = [tmdb_data['original_language']]
            
            # Determine if it's a new release
            is_new_release = False
            release_date = None
            if tmdb_data.get('release_date') or tmdb_data.get('first_air_date'):
                date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
                try:
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    # Check if released in last 60 days
                    if release_date >= (datetime.now() - timedelta(days=60)).date():
                        is_new_release = True
                except:
                    pass
            
            # Determine if it's critics' choice
            is_critics_choice = False
            critics_score = tmdb_data.get('vote_average', 0)
            vote_count = tmdb_data.get('vote_count', 0)
            if critics_score >= 7.5 and vote_count >= 100:
                is_critics_choice = True
            
            # Get YouTube trailer
            youtube_trailer_id = ContentService.get_youtube_trailer(tmdb_data.get('title') or tmdb_data.get('name'))
            
            # Create content object
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
                youtube_trailer_id=youtube_trailer_id,
                is_new_release=is_new_release,
                is_critics_choice=is_critics_choice,
                critics_score=critics_score
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
            # Check if anime already exists
            existing = Content.query.filter_by(mal_id=anime_data['mal_id']).first()
            if existing:
                return existing
            
            # Extract anime genres
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            # Map to anime genre categories
            anime_genre_categories = []
            for genre in genres:
                for category, category_genres in ANIME_GENRES.items():
                    if genre in category_genres:
                        anime_genre_categories.append(category)
            
            # Remove duplicates
            anime_genre_categories = list(set(anime_genre_categories))
            
            # Get release date
            release_date = None
            if anime_data.get('aired', {}).get('from'):
                try:
                    release_date = datetime.strptime(anime_data['aired']['from'][:10], '%Y-%m-%d').date()
                except:
                    pass
            
            # Get YouTube trailer for anime
            youtube_trailer_id = ContentService.get_youtube_trailer(anime_data.get('title'), 'anime')
            
            # Create anime content
            content = Content(
                mal_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps(genres),
                anime_genres=json.dumps(anime_genre_categories),
                languages=json.dumps(['japanese']),
                release_date=release_date,
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                youtube_trailer_id=youtube_trailer_id
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def get_youtube_trailer(title, content_type='movie'):
        """Get YouTube trailer ID for content"""
        try:
            youtube_results = YouTubeService.search_trailers(title, content_type)
            if youtube_results and youtube_results.get('items'):
                # Return the first relevant trailer
                return youtube_results['items'][0]['id']['videoId']
        except Exception as e:
            logger.error(f"Error getting YouTube trailer: {e}")
        return None
    
    @staticmethod
    def map_genre_ids(genre_ids):
        # TMDB Genre ID mapping
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]

# Enhanced Recommendation Engine with ML Integration
class RecommendationEngine:
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all', region=None):
        """Enhanced trending recommendations with ML service integration"""
        try:
            # First try ML service
            ml_params = {
                'limit': limit,
                'content_type': content_type,
                'region': region
            }
            
            ml_response = MLServiceClient.call_ml_service('/api/trending', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for trending recommendations: {len(ml_recommendations)} items")
                    return [rec['content'] for rec in ml_recommendations]
            
            # Fallback to original logic
            logger.info("Falling back to TMDB for trending recommendations")
            
            # Get trending from TMDB for both day and week
            trending_day = TMDBService.get_trending(content_type=content_type, time_window='day')
            trending_week = TMDBService.get_trending(content_type=content_type, time_window='week')
            
            # Combine and remove duplicates
            all_trending = []
            seen_ids = set()
            
            # Process day trending first (more current)
            if trending_day:
                for item in trending_day.get('results', []):
                    if item['id'] not in seen_ids:
                        all_trending.append(item)
                        seen_ids.add(item['id'])
            
            # Add week trending
            if trending_week:
                for item in trending_week.get('results', []):
                    if item['id'] not in seen_ids and len(all_trending) < limit * 2:
                        all_trending.append(item)
                        seen_ids.add(item['id'])
            
            recommendations = []
            for item in all_trending[:limit]:
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    # Mark as trending
                    content.is_trending = True
                    db.session.commit()
                    recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_new_releases(limit=20, language=None, content_type='movie'):
        """Enhanced new releases with ML service integration"""
        try:
            # First try ML service
            ml_params = {
                'limit': limit,
                'language': language,
                'content_type': content_type
            }
            
            ml_response = MLServiceClient.call_ml_service('/api/new-releases', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for new releases: {len(ml_recommendations)} items")
                    return [rec['content'] for rec in ml_recommendations]
            
            # Fallback to original logic
            logger.info("Falling back to TMDB for new releases")
            
            # Map language to TMDB language code
            language_code = None
            if language:
                lang_mapping = {'hindi': 'hi', 'telugu': 'te', 'tamil': 'ta', 
                              'kannada': 'kn', 'malayalam': 'ml', 'english': 'en'}
                language_code = lang_mapping.get(language.lower())
            
            recommendations = []
            
            if language_code:
                # Get language-specific new releases
                new_releases = TMDBService.get_language_specific(language_code, content_type)
            else:
                # Get general new releases
                new_releases = TMDBService.get_new_releases(content_type)
            
            if new_releases:
                for item in new_releases.get('results', [])[:limit]:
                    content = ContentService.save_content_from_tmdb(item, content_type)
                    if content:
                        recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return []
    
    @staticmethod
    def get_critics_choice(limit=20, content_type='movie'):
        """Enhanced critics choice with ML service integration"""
        try:
            # First try ML service
            ml_params = {
                'limit': limit,
                'content_type': content_type
            }
            
            ml_response = MLServiceClient.call_ml_service('/api/critics-choice', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for critics choice: {len(ml_recommendations)} items")
                    return [rec['content'] for rec in ml_recommendations]
            
            # Fallback to original logic
            logger.info("Falling back to TMDB for critics choice")
            
            critics_choice = TMDBService.get_critics_choice(content_type)
            
            recommendations = []
            if critics_choice:
                for item in critics_choice.get('results', [])[:limit]:
                    content = ContentService.save_content_from_tmdb(item, content_type)
                    if content:
                        recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting critics choice: {e}")
            return []
    
    @staticmethod
    def get_genre_recommendations(genre, limit=20, content_type='movie', region=None):
        """Enhanced genre recommendations with ML service integration"""
        try:
            # First try ML service
            ml_params = {
                'limit': limit,
                'content_type': content_type,
                'region': region
            }
            
            ml_response = MLServiceClient.call_ml_service(f'/api/genre/{genre}', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for genre {genre}: {len(ml_recommendations)} items")
                    return [rec['content'] for rec in ml_recommendations]
            
            # Fallback to original logic
            logger.info(f"Falling back to TMDB for genre {genre}")
            
            # Genre ID mapping for TMDB
            genre_ids = {
                'action': 28, 'adventure': 12, 'animation': 16, 'biography': -1,
                'comedy': 35, 'crime': 80, 'documentary': 99, 'drama': 18,
                'fantasy': 14, 'horror': 27, 'musical': 10402, 'mystery': 9648,
                'romance': 10749, 'sci-fi': 878, 'thriller': 53, 'western': 37
            }
            
            genre_id = genre_ids.get(genre.lower())
            if not genre_id or genre_id == -1:
                # Fallback to search for biography and other genres
                search_results = TMDBService.search_content(genre, content_type)
                recommendations = []
                if search_results:
                    for item in search_results.get('results', [])[:limit]:
                        content = ContentService.save_content_from_tmdb(item, content_type)
                        if content:
                            recommendations.append(content)
                return recommendations
            
            # Get content by genre
            genre_content = TMDBService.get_by_genre(genre_id, content_type, region=region)
            
            recommendations = []
            if genre_content:
                for item in genre_content.get('results', [])[:limit]:
                    content = ContentService.save_content_from_tmdb(item, content_type)
                    if content:
                        recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting genre recommendations: {e}")
            return []
    
    @staticmethod
    def get_regional_recommendations(language, limit=20, content_type='movie'):
        """Enhanced regional recommendations with ML service integration"""
        try:
            # First try ML service
            ml_params = {
                'limit': limit,
                'content_type': content_type
            }
            
            ml_response = MLServiceClient.call_ml_service(f'/api/regional/{language}', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for regional {language}: {len(ml_recommendations)} items")
                    return [rec['content'] for rec in ml_recommendations]
            
            # Fallback to original logic
            logger.info(f"Falling back to TMDB for regional {language}")
            
            # Map language to TMDB language code
            lang_mapping = {
                'hindi': 'hi', 'telugu': 'te', 'tamil': 'ta', 
                'kannada': 'kn', 'malayalam': 'ml', 'english': 'en'
            }
            
            language_code = lang_mapping.get(language.lower())
            recommendations = []
            
            if language_code:
                # Get language-specific content
                lang_content = TMDBService.get_language_specific(language_code, content_type)
                if lang_content:
                    for item in lang_content.get('results', [])[:limit]:
                        content = ContentService.save_content_from_tmdb(item, content_type)
                        if content:
                            recommendations.append(content)
            
            # If not enough results, search with language keywords
            if len(recommendations) < limit:
                search_queries = REGIONAL_LANGUAGES.get(language.lower(), [language])
                for query in search_queries:
                    if len(recommendations) >= limit:
                        break
                    
                    search_results = TMDBService.search_content(query, content_type)
                    if search_results:
                        for item in search_results.get('results', []):
                            if len(recommendations) >= limit:
                                break
                            content = ContentService.save_content_from_tmdb(item, content_type)
                            if content:
                                recommendations.append(content)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting regional recommendations: {e}")
            return []
    
    @staticmethod
    def get_anime_recommendations(limit=20, genre=None):
        """Enhanced anime recommendations with ML service integration"""
        try:
            # First try ML service
            ml_params = {
                'limit': limit,
                'genre': genre
            }
            
            ml_response = MLServiceClient.call_ml_service('/api/anime', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for anime recommendations: {len(ml_recommendations)} items")
                    return [rec['content'] for rec in ml_recommendations]
            
            # Fallback to original logic
            logger.info("Falling back to Jikan API for anime recommendations")
            
            recommendations = []
            
            if genre and genre.lower() in ANIME_GENRES:
                # Get anime by specific genre category
                genre_keywords = ANIME_GENRES[genre.lower()]
                for keyword in genre_keywords[:2]:  # Limit to avoid too many requests
                    anime_results = JikanService.get_anime_by_genre(keyword)
                    if anime_results:
                        for anime in anime_results.get('data', []):
                            if len(recommendations) >= limit:
                                break
                            content = ContentService.save_anime_content(anime)
                            if content:
                                recommendations.append(content)
                    if len(recommendations) >= limit:
                        break
            else:
                # Get top anime
                top_anime = JikanService.get_top_anime()
                if top_anime:
                    for anime in top_anime.get('data', [])[:limit]:
                        content = ContentService.save_anime_content(anime)
                        if content:
                            recommendations.append(content)
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting anime recommendations: {e}")
            return []
    
    @staticmethod
    def get_similar_recommendations(content_id, limit=20):
        """Enhanced similar recommendations with ML service integration"""
        try:
            # First try ML service
            ml_params = {
                'limit': limit
            }
            
            ml_response = MLServiceClient.call_ml_service(f'/api/similar/{content_id}', ml_params)
            if ml_response:
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                if ml_recommendations:
                    logger.info(f"Using ML service for similar recommendations: {len(ml_recommendations)} items")
                    return [rec['content'] for rec in ml_recommendations]
            
            # Fallback to original logic
            logger.info("Falling back to TMDB/database for similar recommendations")
            
            base_content = Content.query.get(content_id)
            if not base_content:
                return []
            
            similar_content = []
            
            # Get similar from TMDB if available
            if base_content.tmdb_id and base_content.content_type != 'anime':
                tmdb_details = TMDBService.get_content_details(base_content.tmdb_id, base_content.content_type)
                if tmdb_details:
                    # Get similar content
                    if 'similar' in tmdb_details:
                        for item in tmdb_details['similar']['results'][:10]:
                            content = ContentService.save_content_from_tmdb(item, base_content.content_type)
                            if content:
                                similar_content.append(content)
                    
                    # Get recommendations
                    if 'recommendations' in tmdb_details:
                        for item in tmdb_details['recommendations']['results'][:10]:
                            content = ContentService.save_content_from_tmdb(item, base_content.content_type)
                            if content:
                                similar_content.append(content)
            
            # Add genre-based recommendations
            if base_content.genres:
                genres = json.loads(base_content.genres)
                if genres:
                    # Get content with similar genres from database
                    db_similar = Content.query.filter(
                        Content.id != content_id,
                        Content.content_type == base_content.content_type
                    ).all()
                    
                    # Score by genre overlap
                    scored_content = []
                    for content in db_similar:
                        if content.genres:
                            content_genres = json.loads(content.genres)
                            overlap = len(set(genres) & set(content_genres))
                            if overlap > 0:
                                scored_content.append((content, overlap))
                    
                    # Sort by score and add to similar content
                    scored_content.sort(key=lambda x: x[1], reverse=True)
                    for content, score in scored_content[:10]:
                        if content not in similar_content:
                            similar_content.append(content)
            
            # Remove duplicates and limit
            seen_ids = set()
            unique_similar = []
            for content in similar_content:
                if content.id not in seen_ids:
                    seen_ids.add(content.id)
                    unique_similar.append(content)
                    if len(unique_similar) >= limit:
                        break
            
            return unique_similar
        except Exception as e:
            logger.error(f"Error getting similar recommendations: {e}")
            return []

# Anonymous User Recommendations
class AnonymousRecommendationEngine:
    @staticmethod
    def get_recommendations_for_anonymous(session_id, ip_address, limit=20):
        try:
            # Get user location for regional content
            location = get_user_location(ip_address)
            
            # Get anonymous user's interaction history
            interactions = AnonymousInteraction.query.filter_by(session_id=session_id).all()
            
            recommendations = []
            
            # If user has interactions, recommend similar content
            if interactions:
                # Get genres from viewed content
                viewed_content_ids = [interaction.content_id for interaction in interactions]
                viewed_contents = Content.query.filter(Content.id.in_(viewed_content_ids)).all()
                
                # Extract preferred genres
                all_genres = []
                for content in viewed_contents:
                    if content.genres:
                        all_genres.extend(json.loads(content.genres))
                
                # Get most common genres
                genre_counts = Counter(all_genres)
                top_genres = [genre for genre, _ in genre_counts.most_common(3)]
                
                # Get recommendations based on top genres
                for genre in top_genres:
                    genre_recs = RecommendationEngine.get_genre_recommendations(genre, limit=7)
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
            
            # Format genre list
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

#AdminChoice #MovieRecommendation #CineScope"""
            
            # Send message with photo if available
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
        
        # Validate input
        if not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if user exists
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        # Create user
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password']),
            preferred_languages=json.dumps(data.get('preferred_languages', [])),
            preferred_genres=json.dumps(data.get('preferred_genres', []))
        )
        
        db.session.add(user)
        db.session.commit()
        
        # Generate token
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
        
        # Update last active
        user.last_active = datetime.utcnow()
        db.session.commit()
        
        # Generate token
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

# Content Discovery Routes
@app.route('/api/search', methods=['GET'])
def search_content():
    try:
        query = request.args.get('query', '')
        content_type = request.args.get('type', 'multi')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        # Record search interaction
        session_id = get_session_id()
        
        # Search TMDB
        tmdb_results = TMDBService.search_content(query, content_type, page=page)
        
        # Search anime if content_type is anime or multi
        anime_results = None
        if content_type in ['anime', 'multi']:
            anime_results = JikanService.search_anime(query, page=page)
        
        # Process and save results
        results = []
        
        if tmdb_results:
            for item in tmdb_results.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    # Record anonymous interaction
                    interaction = AnonymousInteraction(
                        session_id=session_id,
                        content_id=content.id,
                        interaction_type='search',
                        ip_address=request.remote_addr
                    )
                    db.session.add(interaction)
                    
                    # Get YouTube trailer URL
                    youtube_url = None
                    if content.youtube_trailer_id:
                        youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                    
                    results.append({
                        'id': content.id,
                        'tmdb_id': content.tmdb_id,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
                        'overview': content.overview,
                        'youtube_trailer': youtube_url
                    })
        
        # Add anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                # Save anime content
                content = ContentService.save_anime_content(anime)
                if content:
                    # Get YouTube trailer URL
                    youtube_url = None
                    if content.youtube_trailer_id:
                        youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                    
                    results.append({
                        'id': content.id,
                        'mal_id': content.mal_id,
                        'title': content.title,
                        'content_type': 'anime',
                        'genres': json.loads(content.genres or '[]'),
                        'anime_genres': json.loads(content.anime_genres or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': content.poster_path,
                        'overview': content.overview,
                        'youtube_trailer': youtube_url
                    })
        
        db.session.commit()
        
        return jsonify({
            'results': results,
            'total_results': tmdb_results.get('total_results', 0) if tmdb_results else 0,
            'total_pages': tmdb_results.get('total_pages', 0) if tmdb_results else 0,
            'current_page': page
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/content/<int:content_id>', methods=['GET'])
def get_content_details(content_id):
    try:
        content = Content.query.get_or_404(content_id)
        
        # Record view interaction
        session_id = get_session_id()
        interaction = AnonymousInteraction(
            session_id=session_id,
            content_id=content.id,
            interaction_type='view',
            ip_address=request.remote_addr
        )
        db.session.add(interaction)
        
        # Get additional details
        additional_details = None
        cast = []
        crew = []
        
        if content.content_type == 'anime' and content.mal_id:
            # Get anime details
            additional_details = JikanService.get_anime_details(content.mal_id)
            if additional_details:
                anime_data = additional_details.get('data', {})
                # Extract voice actors as cast
                if 'voices' in anime_data:
                    cast = anime_data['voices'][:10]
                # Extract staff as crew
                if 'staff' in anime_data:
                    crew = anime_data['staff'][:5]
        elif content.tmdb_id:
            # Get TMDB details
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
            if additional_details:
                cast = additional_details.get('credits', {}).get('cast', [])[:10]
                crew = additional_details.get('credits', {}).get('crew', [])[:5]
        
        # Get similar content using enhanced algorithm
        similar_content = RecommendationEngine.get_similar_recommendations(content.id, limit=10)
        
        # Format similar content
        similar_formatted = []
        for similar in similar_content:
            youtube_url = None
            if similar.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={similar.youtube_trailer_id}"
            
            similar_formatted.append({
                'id': similar.id,
                'title': similar.title,
                'poster_path': f"https://image.tmdb.org/t/p/w300{similar.poster_path}" if similar.poster_path and not similar.poster_path.startswith('http') else similar.poster_path,
                'rating': similar.rating,
                'content_type': similar.content_type,
                'youtube_trailer': youtube_url
            })
        
        db.session.commit()
        
        # Get YouTube trailer URL
        youtube_trailer_url = None
        if content.youtube_trailer_id:
            youtube_trailer_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
        
        response_data = {
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'mal_id': content.mal_id,
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
            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path and not content.backdrop_path.startswith('http') else content.backdrop_path,
            'youtube_trailer': youtube_trailer_url,
            'similar_content': similar_formatted,
            'cast': cast,
            'crew': crew,
            'is_trending': content.is_trending,
            'is_new_release': content.is_new_release,
            'is_critics_choice': content.is_critics_choice
        }
        
        # Add anime-specific data
        if content.content_type == 'anime':
            response_data['anime_genres'] = json.loads(content.anime_genres or '[]')
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Enhanced Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        region = request.args.get('region')
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type, region)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url,
                'is_trending': content.is_trending
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/recommendations/new-releases', methods=['GET'])
def get_new_releases():
    try:
        language = request.args.get('language')
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_new_releases(limit, language, content_type)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'youtube_trailer': youtube_url,
                'is_new_release': content.is_new_release
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"New releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

@app.route('/api/recommendations/critics-choice', methods=['GET'])
def get_critics_choice():
    try:
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_critics_choice(limit, content_type)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url,
                'is_critics_choice': content.is_critics_choice,
                'critics_score': content.critics_score
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Critics choice error: {e}")
        return jsonify({'error': 'Failed to get critics choice'}), 500

@app.route('/api/recommendations/genre/<genre>', methods=['GET'])
def get_genre_recommendations(genre):
    try:
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        region = request.args.get('region')
        
        recommendations = RecommendationEngine.get_genre_recommendations(genre, limit, content_type, region)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Genre recommendations error: {e}")
        return jsonify({'error': 'Failed to get genre recommendations'}), 500

@app.route('/api/recommendations/regional/<language>', methods=['GET'])
def get_regional(language):
    try:
        content_type = request.args.get('type', 'movie')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_regional_recommendations(language, limit, content_type)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Regional recommendations error: {e}")
        return jsonify({'error': 'Failed to get regional recommendations'}), 500

@app.route('/api/recommendations/anime', methods=['GET'])
def get_anime():
    try:
        genre = request.args.get('genre')  # shonen, shojo, seinen, josei, kodomomuke
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_anime_recommendations(limit, genre)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'mal_id': content.mal_id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'anime_genres': json.loads(content.anime_genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Anime recommendations error: {e}")
        return jsonify({'error': 'Failed to get anime recommendations'}), 500

@app.route('/api/recommendations/similar/<int:content_id>', methods=['GET'])
def get_similar_recommendations(content_id):
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_similar_recommendations(content_id, limit)
        
        result = []
        for content in recommendations:
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Similar recommendations error: {e}")
        return jsonify({'error': 'Failed to get similar recommendations'}), 500

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
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'youtube_trailer': youtube_url
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
        # Get user interactions
        interactions = UserInteraction.query.filter_by(user_id=current_user.id).all()
        
        # Prepare data for ML service
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
        
        # Call ML service
        try:
            response = requests.post(f"{ML_SERVICE_URL}/api/recommendations", json=user_data, timeout=30)
            
            if response.status_code == 200:
                ml_recommendations = response.json().get('recommendations', [])
                
                # Get content details for recommended content IDs
                content_ids = [rec['content_id'] for rec in ml_recommendations]
                contents = Content.query.filter(Content.id.in_(content_ids)).all()
                
                # Create response with ML scores
                result = []
                content_dict = {content.id: content for content in contents}
                
                for rec in ml_recommendations:
                    content = content_dict.get(rec['content_id'])
                    if content:
                        youtube_url = None
                        if content.youtube_trailer_id:
                            youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                        
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'youtube_trailer': youtube_url,
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

# ML-Enhanced Personalized Recommendations
@app.route('/api/recommendations/ml-personalized', methods=['GET'])
@require_auth
def get_ml_personalized_recommendations(current_user):
    """ML-enhanced personalized recommendations with detailed ML insights"""
    try:
        limit = int(request.args.get('limit', 20))
        
        # Get user interactions
        interactions = UserInteraction.query.filter_by(user_id=current_user.id).all()
        
        # Prepare data for ML service
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
        
        # Call ML service
        try:
            response = requests.post(f"{ML_SERVICE_URL}/api/recommendations", json=user_data, timeout=30)
            
            if response.status_code == 200:
                ml_response = response.json()
                ml_recommendations = MLServiceClient.process_ml_recommendations(ml_response, limit)
                
                if ml_recommendations:
                    # Create detailed response with ML insights
                    result = []
                    for rec in ml_recommendations:
                        content = rec['content']
                        youtube_url = None
                        if content.youtube_trailer_id:
                            youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                        
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'youtube_trailer': youtube_url,
                            'ml_score': rec['ml_score'],
                            'ml_reason': rec['ml_reason'],
                            'ml_rank': rec['ml_rank'],
                            'recommendation_source': 'ml_service'
                        })
                    
                    return jsonify({
                        'recommendations': result,
                        'ml_strategy': ml_response.get('strategy', 'unknown'),
                        'ml_cached': ml_response.get('cached', False),
                        'total_interactions': len(interactions),
                        'source': 'ml_service'
                    }), 200
        except:
            pass
        
        # Fallback to basic personalized recommendations
        return get_trending()
        
    except Exception as e:
        logger.error(f"ML personalized recommendations error: {e}")
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
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'youtube_trailer': youtube_url
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
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                'youtube_trailer': youtube_url
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
        
        # Check if content already exists by external ID
        existing_content = None
        if data.get('source') == 'anime' and data.get('id'):
            existing_content = Content.query.filter_by(mal_id=data['id']).first()
        elif data.get('id'):
            existing_content = Content.query.filter_by(tmdb_id=data['id']).first()
        
        if existing_content:
            return jsonify({
                'message': 'Content already exists',
                'content_id': existing_content.id
            }), 200
        
        # Create new content from external data
        try:
            # Handle release date
            release_date = None
            if data.get('release_date'):
                try:
                    release_date = datetime.strptime(data['release_date'][:10], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            # Get YouTube trailer
            youtube_trailer_id = ContentService.get_youtube_trailer(data.get('title'), data.get('content_type'))
            
            # Create content object
            if data.get('source') == 'anime':
                content = Content(
                    mal_id=data.get('id'),
                    title=data.get('title'),
                    original_title=data.get('original_title'),
                    content_type='anime',
                    genres=json.dumps(data.get('genres', [])),
                    anime_genres=json.dumps([]),
                    languages=json.dumps(['japanese']),
                    release_date=release_date,
                    rating=data.get('rating'),
                    overview=data.get('overview'),
                    poster_path=data.get('poster_path'),
                    youtube_trailer_id=youtube_trailer_id
                )
            else:
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
                    youtube_trailer_id=youtube_trailer_id
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
        
        # Get content - handle both internal ID and external ID
        content = Content.query.get(data['content_id'])
        if not content:
            # Try to find by TMDB ID if direct ID lookup fails
            content = Content.query.filter_by(tmdb_id=data['content_id']).first()
        
        if not content:
            return jsonify({'error': 'Content not found. Please save content first.'}), 404
        
        # Create admin recommendation
        admin_rec = AdminRecommendation(
            content_id=content.id,
            admin_id=current_user.id,
            recommendation_type=data['recommendation_type'],
            description=data['description']
        )
        
        db.session.add(admin_rec)
        db.session.commit()
        
        # Send to Telegram channel
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
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path
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
        # Get basic analytics
        total_users = User.query.count()
        total_content = Content.query.count()
        total_interactions = UserInteraction.query.count()
        active_users_last_week = User.query.filter(
            User.last_active >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        # Popular content
        popular_content = db.session.query(
            Content.id, Content.title, func.count(UserInteraction.id).label('interaction_count')
        ).join(UserInteraction).group_by(Content.id, Content.title)\
         .order_by(desc('interaction_count')).limit(10).all()
        
        # Popular genres
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

# ML Service Admin Routes
@app.route('/api/admin/ml-service-check', methods=['GET'])
@require_admin
def ml_service_comprehensive_check(current_user):
    """Simple comprehensive ML service check"""
    try:
        ml_url = ML_SERVICE_URL
        if not ml_url:
            return jsonify({
                'status': 'error',
                'message': 'ML_SERVICE_URL not configured',
                'checks': {}
            }), 500

        checks = {}
        overall_status = 'healthy'
        
        # 1. Basic Health Check
        try:
            start_time = time.time()
            health_resp = requests.get(f"{ml_url}/api/health", timeout=10)
            health_time = time.time() - start_time
            
            if health_resp.status_code == 200:
                health_data = health_resp.json()
                checks['connectivity'] = {
                    'status': 'pass',
                    'response_time': f"{health_time:.2f}s",
                    'models_initialized': health_data.get('models_initialized', False),
                    'data_status': health_data.get('data_status', {})
                }
            else:
                checks['connectivity'] = {'status': 'fail', 'error': f'HTTP {health_resp.status_code}'}
                overall_status = 'unhealthy'
        except Exception as e:
            checks['connectivity'] = {'status': 'fail', 'error': str(e)}
            overall_status = 'unhealthy'

        # 2. Recommendation Test (only if connectivity passes)
        if checks['connectivity']['status'] == 'pass':
            try:
                start_time = time.time()
                test_request = {
                    'user_id': 1,
                    'preferred_languages': ['english'],
                    'preferred_genres': ['Action'],
                    'interactions': [{
                        'content_id': 1,
                        'interaction_type': 'view',
                        'timestamp': datetime.utcnow().isoformat()
                    }]
                }
                
                rec_resp = requests.post(f"{ml_url}/api/recommendations", json=test_request, timeout=20)
                rec_time = time.time() - start_time
                
                if rec_resp.status_code == 200:
                    rec_data = rec_resp.json()
                    checks['recommendations'] = {
                        'status': 'pass',
                        'response_time': f"{rec_time:.2f}s",
                        'count': len(rec_data.get('recommendations', [])),
                        'strategy': rec_data.get('strategy', 'unknown'),
                        'cached': rec_data.get('cached', False)
                    }
                else:
                    checks['recommendations'] = {'status': 'fail', 'error': f'HTTP {rec_resp.status_code}'}
                    overall_status = 'partial'
            except Exception as e:
                checks['recommendations'] = {'status': 'fail', 'error': str(e)}
                overall_status = 'partial'

        # 3. Statistics Check
        if checks['connectivity']['status'] == 'pass':
            try:
                start_time = time.time()
                stats_resp = requests.get(f"{ml_url}/api/stats", timeout=10)
                stats_time = time.time() - start_time
                
                if stats_resp.status_code == 200:
                    stats_data = stats_resp.json()
                    checks['statistics'] = {
                        'status': 'pass',
                        'response_time': f"{stats_time:.2f}s",
                        'data_count': stats_data.get('data_statistics', {}).get('total_content', 0),
                        'user_count': stats_data.get('data_statistics', {}).get('unique_users', 0)
                    }
                else:
                    checks['statistics'] = {'status': 'fail', 'error': f'HTTP {stats_resp.status_code}'}
            except Exception as e:
                checks['statistics'] = {'status': 'fail', 'error': str(e)}

        # 4. Quick Performance Test
        endpoints = [
            {'name': 'trending', 'url': '/api/trending?limit=3'},
        ]
        
        performance = {}
        for endpoint in endpoints:
            try:
                start_time = time.time()
                resp = requests.get(f"{ml_url}{endpoint['url']}", timeout=10)
                response_time = time.time() - start_time
                
                performance[endpoint['name']] = {
                    'status': 'pass' if resp.status_code == 200 else 'fail',
                    'response_time': f"{response_time:.2f}s"
                }
            except Exception as e:
                performance[endpoint['name']] = {'status': 'fail', 'error': str(e)}

        checks['performance'] = performance

        # 5. Database Integration Check
        try:
            total_users = User.query.count()
            total_content = Content.query.count() 
            total_interactions = UserInteraction.query.count()
            
            checks['database_integration'] = {
                'status': 'pass',
                'backend_users': total_users,
                'backend_content': total_content,
                'backend_interactions': total_interactions,
                'data_ready': total_content > 0 and total_interactions > 0
            }
        except Exception as e:
            checks['database_integration'] = {'status': 'fail', 'error': str(e)}

        # Summary
        failed_checks = sum(1 for check in checks.values() 
                           if isinstance(check, dict) and check.get('status') == 'fail')
        total_checks = len([check for check in checks.values() if isinstance(check, dict)])
        
        if failed_checks == 0:
            overall_status = 'healthy'
        elif failed_checks < total_checks:
            overall_status = 'partial'
        else:
            overall_status = 'unhealthy'

        return jsonify({
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'ml_service_url': ml_url,
            'summary': {
                'total_checks': total_checks,
                'passed': total_checks - failed_checks,
                'failed': failed_checks
            },
            'checks': checks,
            'quick_actions': {
                'force_update_available': checks.get('connectivity', {}).get('status') == 'pass',
                'recommendations_working': checks.get('recommendations', {}).get('status') == 'pass'
            }
        }), 200

    except Exception as e:
        logger.error(f"ML service check error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/admin/ml-service-update', methods=['POST'])
@require_admin  
def ml_service_force_update(current_user):
    """Force ML service model update"""
    try:
        if not ML_SERVICE_URL:
            return jsonify({'success': False, 'message': 'ML service not configured'}), 400
            
        response = requests.post(f"{ML_SERVICE_URL}/api/update-models", timeout=30)
        
        if response.status_code == 200:
            return jsonify({'success': True, 'message': 'Model update initiated'})
        else:
            return jsonify({'success': False, 'message': f'Update failed: {response.status_code}'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/admin/ml-stats', methods=['GET'])
@require_admin
def get_ml_service_stats(current_user):
    """Get ML service statistics and performance metrics"""
    try:
        if not ML_SERVICE_URL:
            return jsonify({'error': 'ML service not configured'}), 400
        
        # Get ML service stats
        ml_stats = MLServiceClient.call_ml_service('/api/stats')
        
        if ml_stats:
            # Add backend stats for comparison
            backend_stats = {
                'total_users': User.query.count(),
                'total_content': Content.query.count(),
                'total_interactions': UserInteraction.query.count(),
                'active_users_last_week': User.query.filter(
                    User.last_active >= datetime.utcnow() - timedelta(days=7)
                ).count()
            }
            
            return jsonify({
                'ml_service_stats': ml_stats,
                'backend_stats': backend_stats,
                'data_sync_status': {
                    'content_match': backend_stats['total_content'] == ml_stats.get('data_statistics', {}).get('total_content', 0),
                    'user_match': backend_stats['total_users'] == ml_stats.get('data_statistics', {}).get('unique_users', 0)
                }
            }), 200
        else:
            return jsonify({'error': 'Failed to get ML service stats'}), 500
            
    except Exception as e:
        logger.error(f"ML stats error: {e}")
        return jsonify({'error': 'Failed to get ML statistics'}), 500

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
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'youtube_trailer': youtube_url,
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
        'version': '2.0.0'
    }), 200

# Initialize database
def create_tables():
    try:
        with app.app_context():
            db.create_all()
            
            # Create admin user if not exists
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

# Initialize database when app starts
create_tables()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)