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
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, quote, quote_plus
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

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

# Configuration
IS_PRODUCTION = os.environ.get('FLASK_ENV') == 'production' or os.environ.get('DATABASE_URL') is not None
ENABLE_ADVANCED_OTT = os.environ.get('ENABLE_ADVANCED_OTT', 'true').lower() == 'true'

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

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=3)

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
    ott_platforms = db.Column(db.Text)  # JSON string - Enhanced with direct links and languages
    ott_last_updated = db.Column(db.DateTime, default=datetime.utcnow)
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

# Enhanced OTT Platform Information with Real URLs and Comprehensive Data
OTT_PLATFORMS = {
    'netflix': {
        'name': 'Netflix',
        'is_free': False,
        'base_url': 'https://www.netflix.com',
        'search_url': 'https://www.netflix.com/search?q={}',
        'direct_url_pattern': 'https://www.netflix.com/title/{}',
        'logo': 'https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg',
        'supported_regions': ['US', 'IN', 'UK', 'CA', 'AU', 'DE', 'FR', 'BR', 'JP', 'KR'],
        'content_types': ['movie', 'tv', 'documentary', 'anime'],
        'languages': ['english', 'hindi', 'spanish', 'french', 'german', 'japanese', 'korean'],
        'quality': ['HD', '4K', 'HDR'],
        'features': ['offline_download', 'multiple_profiles', 'kids_content']
    },
    'prime_video': {
        'name': 'Amazon Prime Video',
        'is_free': False,
        'base_url': 'https://www.primevideo.com',
        'search_url': 'https://www.primevideo.com/search/ref=atv_nb_sr?phrase={}',
        'direct_url_pattern': 'https://www.primevideo.com/detail/{}',
        'logo': 'https://upload.wikimedia.org/wikipedia/commons/f/f1/Prime_Video.png',
        'supported_regions': ['US', 'IN', 'UK', 'DE', 'JP', 'CA', 'AU', 'FR', 'IT', 'ES'],
        'content_types': ['movie', 'tv', 'documentary', 'sports'],
        'languages': ['english', 'hindi', 'tamil', 'telugu', 'german', 'spanish', 'french'],
        'quality': ['HD', '4K', 'HDR'],
        'features': ['offline_download', 'x-ray', 'prime_shipping']
    },
    'hotstar': {
        'name': 'Disney+ Hotstar',
        'is_free': False,
        'base_url': 'https://www.hotstar.com',
        'search_url': 'https://www.hotstar.com/in/search?q={}',
        'direct_url_pattern': 'https://www.hotstar.com/in/movies/{}/{}',
        'logo': 'https://upload.wikimedia.org/wikipedia/commons/1/1e/Disney%2B_Hotstar_logo.svg',
        'supported_regions': ['IN', 'US', 'UK', 'CA', 'SG', 'MY', 'TH', 'ID'],
        'content_types': ['movie', 'tv', 'sports', 'documentary', 'kids'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'bengali', 'marathi', 'kannada', 'malayalam'],
        'quality': ['HD', '4K'],
        'features': ['live_sports', 'disney_content', 'marvel_content', 'star_wars']
    },
    'aha': {
        'name': 'Aha',
        'is_free': False,
        'base_url': 'https://www.aha.video',
        'search_url': 'https://www.aha.video/search?q={}',
        'direct_url_pattern': 'https://www.aha.video/player/movie/{}',
        'logo': 'https://www.aha.video/images/aha-logo.svg',
        'supported_regions': ['IN'],
        'content_types': ['movie', 'tv', 'web_series'],
        'languages': ['telugu', 'tamil'],
        'quality': ['HD'],
        'features': ['regional_content', 'original_series', 'offline_download']
    },
    'sun_nxt': {
        'name': 'Sun NXT',
        'is_free': False,
        'base_url': 'https://www.sunnxt.com',
        'search_url': 'https://www.sunnxt.com/search?q={}',
        'direct_url_pattern': 'https://www.sunnxt.com/movie/watch/{}',
        'logo': 'https://www.sunnxt.com/assets/images/sun-nxt-logo.png',
        'supported_regions': ['IN', 'US', 'MY', 'SG'],
        'content_types': ['movie', 'tv', 'music'],
        'languages': ['tamil', 'telugu', 'malayalam', 'kannada'],
        'quality': ['HD'],
        'features': ['south_indian_content', 'music_videos', 'devotional_content']
    },
    'mx_player': {
        'name': 'MX Player',
        'is_free': True,
        'base_url': 'https://www.mxplayer.in',
        'search_url': 'https://www.mxplayer.in/search?q={}',
        'direct_url_pattern': 'https://www.mxplayer.in/movie/watch-{}-movie-online-{}',
        'logo': 'https://upload.wikimedia.org/wikipedia/commons/3/3a/MX_Player_Logo.png',
        'supported_regions': ['IN'],
        'content_types': ['movie', 'tv', 'web_series', 'music'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'bengali', 'marathi', 'gujarati', 'punjabi', 'bhojpuri'],
        'quality': ['SD', 'HD'],
        'features': ['free_content', 'ad_supported', 'offline_download', 'regional_content']
    },
    'jiocinema': {
        'name': 'JioCinema',
        'is_free': True,
        'base_url': 'https://www.jiocinema.com',
        'search_url': 'https://www.jiocinema.com/search?q={}',
        'direct_url_pattern': 'https://www.jiocinema.com/movies/{}/{}',
        'logo': 'https://www.jiocinema.com/images/jio-cinema-logo.svg',
        'supported_regions': ['IN'],
        'content_types': ['movie', 'tv', 'sports', 'music'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada', 'bengali', 'marathi', 'gujarati', 'punjabi'],
        'quality': ['SD', 'HD', '4K'],
        'features': ['free_content', 'live_tv', 'sports', 'jio_exclusive']
    },
    'sonyliv': {
        'name': 'SonyLIV',
        'is_free': False,
        'base_url': 'https://www.sonyliv.com',
        'search_url': 'https://www.sonyliv.com/search?q={}',
        'direct_url_pattern': 'https://www.sonyliv.com/movies/{}',
        'logo': 'https://www.sonyliv.com/images/common/sonyliv_logo.png',
        'supported_regions': ['IN', 'US', 'UK', 'AU'],
        'content_types': ['movie', 'tv', 'sports', 'news'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada', 'bengali', 'marathi'],
        'quality': ['HD', '4K'],
        'features': ['live_sports', 'sony_content', 'wwe', 'uefa']
    },
    'youtube': {
        'name': 'YouTube',
        'is_free': True,
        'base_url': 'https://www.youtube.com',
        'search_url': 'https://www.youtube.com/results?search_query={}',
        'direct_url_pattern': 'https://www.youtube.com/watch?v={}',
        'logo': 'https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png',
        'supported_regions': ['global'],
        'content_types': ['movie', 'tv', 'documentary', 'trailer', 'music', 'educational'],
        'languages': ['all'],
        'quality': ['SD', 'HD', '4K'],
        'features': ['free_content', 'ad_supported', 'user_generated', 'live_streaming']
    },
    'airtel_xstream': {
        'name': 'Airtel Xstream',
        'is_free': False,
        'base_url': 'https://www.airtelxstream.in',
        'search_url': 'https://www.airtelxstream.in/search?q={}',
        'direct_url_pattern': 'https://www.airtelxstream.in/movies/{}',
        'logo': 'https://www.airtelxstream.in/assets/images/airtel-xstream-logo.png',
        'supported_regions': ['IN'],
        'content_types': ['movie', 'tv', 'music'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada', 'bengali', 'marathi'],
        'quality': ['HD'],
        'features': ['airtel_exclusive', 'live_tv', 'music']
    },
    'zee5': {
        'name': 'ZEE5',
        'is_free': False,
        'base_url': 'https://www.zee5.com',
        'search_url': 'https://www.zee5.com/search?q={}',
        'direct_url_pattern': 'https://www.zee5.com/movies/details/{}/{}',
        'logo': 'https://upload.wikimedia.org/wikipedia/commons/d/d7/Zee5_Official_Logo.png',
        'supported_regions': ['IN', 'US', 'UK', 'BD', 'MY'],
        'content_types': ['movie', 'tv', 'music', 'kids'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada', 'bengali', 'marathi', 'gujarati', 'punjabi', 'oriya'],
        'quality': ['HD', '4K'],
        'features': ['zee_originals', 'regional_content', 'music_videos', 'kids_content']
    },
    'voot': {
        'name': 'Voot',
        'is_free': True,
        'base_url': 'https://www.voot.com',
        'search_url': 'https://www.voot.com/search?q={}',
        'direct_url_pattern': 'https://www.voot.com/movies/{}/{}',
        'logo': 'https://www.voot.com/images/voot-logo.png',
        'supported_regions': ['IN', 'UK'],
        'content_types': ['movie', 'tv', 'kids'],
        'languages': ['hindi', 'english', 'tamil', 'telugu', 'malayalam', 'kannada', 'bengali', 'marathi', 'gujarati'],
        'quality': ['SD', 'HD'],
        'features': ['free_content', 'colors_content', 'kids_content', 'reality_shows']
    },
    'eros_now': {
        'name': 'Eros Now',
        'is_free': False,
        'base_url': 'https://erosnow.com',
        'search_url': 'https://erosnow.com/search?q={}',
        'direct_url_pattern': 'https://erosnow.com/movie/watch/{}',
        'logo': 'https://erosnow.com/images/eros-now-logo.png',
        'supported_regions': ['IN', 'US', 'UK', 'CA', 'AU'],
        'content_types': ['movie', 'tv', 'music'],
        'languages': ['hindi', 'tamil', 'telugu', 'malayalam', 'kannada', 'bengali', 'marathi', 'gujarati', 'punjabi'],
        'quality': ['HD'],
        'features': ['bollywood_content', 'music_videos', 'devotional_content']
    },
    'alt_balaji': {
        'name': 'ALTBalaji',
        'is_free': False,
        'base_url': 'https://www.altbalaji.com',
        'search_url': 'https://www.altbalaji.com/search?q={}',
        'direct_url_pattern': 'https://www.altbalaji.com/detail/{}',
        'logo': 'https://www.altbalaji.com/images/altbalaji-logo.png',
        'supported_regions': ['IN'],
        'content_types': ['tv', 'web_series'],
        'languages': ['hindi', 'english'],
        'quality': ['HD'],
        'features': ['original_content', 'bold_content', 'web_series']
    },
    'ullu': {
        'name': 'Ullu',
        'is_free': False,
        'base_url': 'https://www.ullu.app',
        'search_url': 'https://www.ullu.app/search?q={}',
        'direct_url_pattern': 'https://www.ullu.app/details/{}',
        'logo': 'https://www.ullu.app/images/ullu-logo.png',
        'supported_regions': ['IN'],
        'content_types': ['web_series', 'movie'],
        'languages': ['hindi', 'english'],
        'quality': ['HD'],
        'features': ['original_content', 'adult_content', 'web_series']
    }
}

# Regional Language Mapping with Enhanced Detection
REGIONAL_LANGUAGES = {
    'hindi': ['hi', 'hindi', 'bollywood', 'हिंदी', 'hindhi', 'bollywood movies'],
    'telugu': ['te', 'telugu', 'tollywood', 'తెలుగు', 'tollywood movies', 'andhra', 'telangana'],
    'tamil': ['ta', 'tamil', 'kollywood', 'தமிழ்', 'kollywood movies', 'tamilnadu', 'chennai'],
    'kannada': ['kn', 'kannada', 'sandalwood', 'ಕನ್ನಡ', 'sandalwood movies', 'karnataka', 'bangalore'],
    'malayalam': ['ml', 'malayalam', 'mollywood', 'മലയാളം', 'mollywood movies', 'kerala', 'kochi'],
    'english': ['en', 'english', 'hollywood', 'american', 'british', 'english movies'],
    'bengali': ['bn', 'bengali', 'tollywood bengali', 'বাংলা', 'kolkata', 'west bengal', 'bangla'],
    'marathi': ['mr', 'marathi', 'मराठी', 'maharashtra', 'mumbai', 'pune'],
    'gujarati': ['gu', 'gujarati', 'ગુજરાતી', 'gujarat', 'ahmedabad'],
    'punjabi': ['pa', 'punjabi', 'ਪੰਜਾਬੀ', 'punjab', 'chandigarh'],
    'bhojpuri': ['bho', 'bhojpuri', 'भोजपुरी', 'bihar', 'uttar pradesh'],
    'oriya': ['or', 'oriya', 'odia', 'ଓଡ଼ିଆ', 'odisha', 'bhubaneswar'],
    'assamese': ['as', 'assamese', 'অসমীয়া', 'assam', 'guwahati'],
    'urdu': ['ur', 'urdu', 'اردو', 'hyderabad', 'lucknow']
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

# Enhanced OTT Services with Advanced Detection and Real API Integration
class AdvancedOTTService:
    @staticmethod
    def detect_content_language(title, overview=None, original_language=None, genres=None):
        """Advanced content language detection with multiple factors"""
        languages = set()
        confidence_scores = {}
        
        # Check original language first (highest confidence)
        if original_language:
            lang_map = {
                'hi': 'hindi', 'te': 'telugu', 'ta': 'tamil', 'kn': 'kannada', 
                'ml': 'malayalam', 'bn': 'bengali', 'mr': 'marathi', 'gu': 'gujarati',
                'pa': 'punjabi', 'or': 'oriya', 'as': 'assamese', 'ur': 'urdu',
                'en': 'english', 'ja': 'japanese', 'ko': 'korean', 'zh': 'chinese'
            }
            if original_language in lang_map:
                detected_lang = lang_map[original_language]
                languages.add(detected_lang)
                confidence_scores[detected_lang] = 0.9
        
        # Analyze title and overview content
        text_to_analyze = (title or '').lower()
        if overview:
            text_to_analyze += ' ' + overview.lower()
        
        # Check for language indicators in text
        for language, indicators in REGIONAL_LANGUAGES.items():
            for indicator in indicators:
                if indicator.lower() in text_to_analyze:
                    languages.add(language)
                    confidence_scores[language] = confidence_scores.get(language, 0) + 0.3
        
        # Genre-based language hints
        if genres:
            genre_text = ' '.join(genres).lower()
            if 'bollywood' in genre_text or 'masala' in genre_text:
                languages.add('hindi')
                confidence_scores['hindi'] = confidence_scores.get('hindi', 0) + 0.2
        
        # Default to English if no specific language detected
        if not languages:
            languages.add('english')
            confidence_scores['english'] = 0.5
        
        # Sort by confidence and return top languages
        sorted_languages = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        return [lang for lang, score in sorted_languages if score > 0.2][:3]  # Top 3 languages
    
    @staticmethod
    def get_platform_availability(title, content_type, languages, genres=None, year=None, popularity=None):
        """Advanced platform availability detection with intelligent matching"""
        platforms = []
        
        # Content analysis
        is_bollywood = 'hindi' in languages
        is_south_indian = any(lang in languages for lang in ['telugu', 'tamil', 'kannada', 'malayalam'])
        is_english = 'english' in languages
        is_regional = any(lang in languages for lang in ['bengali', 'marathi', 'gujarati', 'punjabi', 'bhojpuri'])
        is_recent = year and year >= 2020
        is_very_recent = year and year >= 2022
        is_popular = popularity and popularity > 50
        is_high_rated = True  # Would check rating if available
        
        # Genre analysis
        action_thriller = genres and any(g.lower() in ['action', 'thriller', 'crime', 'adventure'] for g in genres)
        family_content = genres and any(g.lower() in ['family', 'animation', 'comedy', 'kids'] for g in genres)
        drama_romance = genres and any(g.lower() in ['drama', 'romance', 'biographical'] for g in genres)
        horror_adult = genres and any(g.lower() in ['horror', 'thriller', 'adult'] for g in genres)
        documentary = genres and any(g.lower() in ['documentary', 'biography'] for g in genres)
        
        # Netflix - Global platform with premium content
        netflix_confidence = 0.3  # Base confidence
        if is_english: netflix_confidence += 0.4
        if is_bollywood and is_recent: netflix_confidence += 0.3
        if is_popular: netflix_confidence += 0.2
        if action_thriller or drama_romance: netflix_confidence += 0.1
        if is_very_recent: netflix_confidence += 0.2
        
        if netflix_confidence > 0.4:
            platforms.append(AdvancedOTTService._create_platform_entry(
                'netflix', title, languages, content_type, netflix_confidence,
                note="Premium global content with multi-language support"
            ))
        
        # Amazon Prime Video - Wide coverage, especially for Indian content
        prime_confidence = 0.4  # Base confidence
        if is_bollywood: prime_confidence += 0.3
        if is_south_indian: prime_confidence += 0.3
        if is_english: prime_confidence += 0.2
        if is_recent: prime_confidence += 0.2
        if content_type == 'tv': prime_confidence += 0.1
        
        if prime_confidence > 0.4:
            platforms.append(AdvancedOTTService._create_platform_entry(
                'prime_video', title, languages, content_type, prime_confidence,
                note="Wide content library with strong Indian collection"
            ))
        
        # Disney+ Hotstar - Indian content + Disney/Marvel
        hotstar_confidence = 0.2
        if is_bollywood: hotstar_confidence += 0.4
        if is_south_indian: hotstar_confidence += 0.3
        if family_content: hotstar_confidence += 0.3
        if content_type == 'tv': hotstar_confidence += 0.2
        if genres and any(g.lower() in ['superhero', 'marvel', 'disney'] for g in genres): hotstar_confidence += 0.4
        
        if hotstar_confidence > 0.4:
            platforms.append(AdvancedOTTService._create_platform_entry(
                'hotstar', title, languages, content_type, hotstar_confidence,
                note="Disney content, sports, and Indian entertainment"
            ))
        
        # Regional platforms
        if 'telugu' in languages or 'tamil' in languages:
            platforms.append(AdvancedOTTService._create_platform_entry(
                'aha', title, languages, content_type, 0.8,
                note="Dedicated Telugu and Tamil content platform"
            ))
        
        if is_south_indian:
            platforms.append(AdvancedOTTService._create_platform_entry(
                'sun_nxt', title, languages, content_type, 0.7,
                note="South Indian movies and music"
            ))
        
        # Free platforms with good Indian content
        if is_bollywood or is_south_indian or is_regional:
            platforms.append(AdvancedOTTService._create_platform_entry(
                'mx_player', title, languages, content_type, 0.6,
                note="Free Indian movies and shows with ads"
            ))
            
            platforms.append(AdvancedOTTService._create_platform_entry(
                'jiocinema', title, languages, content_type, 0.5,
                note="Free streaming with live TV and sports"
            ))
        
        # Premium Indian platforms
        if is_bollywood or is_south_indian:
            platforms.append(AdvancedOTTService._create_platform_entry(
                'zee5', title, languages, content_type, 0.6,
                note="ZEE originals and regional content"
            ))
            
            platforms.append(AdvancedOTTService._create_platform_entry(
                'sonyliv', title, languages, content_type, 0.5,
                note="Sony content with sports and entertainment"
            ))
        
        # YouTube - Universal availability
        youtube_confidence = 0.9
        youtube_note = "Trailers available"
        if content_type == 'documentary': 
            youtube_note = "May have full documentary"
            youtube_confidence = 0.95
        elif is_bollywood and year and year < 2015:
            youtube_note = "Older movies often available in full"
            youtube_confidence = 0.8
        elif content_type == 'music':
            youtube_note = "Music videos and songs available"
            youtube_confidence = 0.99
        
        platforms.append(AdvancedOTTService._create_platform_entry(
            'youtube', title, languages, content_type, youtube_confidence, note=youtube_note
        ))
        
        # Other platforms based on content type
        if horror_adult:
            platforms.append(AdvancedOTTService._create_platform_entry(
                'ullu', title, languages, content_type, 0.4,
                note="Adult and bold content"
            ))
        
        if content_type == 'web_series' and is_bollywood:
            platforms.append(AdvancedOTTService._create_platform_entry(
                'alt_balaji', title, languages, content_type, 0.5,
                note="Original web series and bold content"
            ))
        
        # Sort by confidence and return top platforms
        platforms.sort(key=lambda x: x.get('availability_confidence', 0), reverse=True)
        return platforms[:8]  # Return top 8 platforms
    
    @staticmethod
    def _create_platform_entry(platform_key, title, languages, content_type, availability_confidence=0.5, note=None):
        """Create enhanced platform entry with working URLs"""
        platform_info = OTT_PLATFORMS.get(platform_key, {})
        
        # Generate language-specific links
        links = {}
        supported_languages = platform_info.get('languages', [])
        
        for language in languages:
            # Check if platform supports this language
            if language in supported_languages or 'all' in supported_languages:
                # Create optimized search query
                search_query = AdvancedOTTService._create_search_query(title, language, content_type)
                
                # Generate URLs
                search_url = AdvancedOTTService._generate_search_url(platform_key, search_query)
                direct_url = AdvancedOTTService._generate_direct_url(platform_key, title, content_type, language)
                
                # Determine quality based on platform and subscription
                quality = AdvancedOTTService._determine_quality(platform_key, availability_confidence)
                
                links[language] = {
                    'watch_url': direct_url or search_url,
                    'search_url': search_url,
                    'subscription_required': not platform_info.get('is_free', False),
                    'quality': quality,
                    'availability_confidence': availability_confidence,
                    'language': language,
                    'platform_features': platform_info.get('features', [])
                }
        
        # Create default link if no language-specific links
        if not links:
            search_query = AdvancedOTTService._create_search_query(title, 'english', content_type)
            search_url = AdvancedOTTService._generate_search_url(platform_key, search_query)
            direct_url = AdvancedOTTService._generate_direct_url(platform_key, title, content_type)
            quality = AdvancedOTTService._determine_quality(platform_key, availability_confidence)
            
            links['default'] = {
                'watch_url': direct_url or search_url,
                'search_url': search_url,
                'subscription_required': not platform_info.get('is_free', False),
                'quality': quality,
                'availability_confidence': availability_confidence
            }
        
        # Create comprehensive platform entry
        platform_entry = {
            'platform': platform_key,
            'name': platform_info.get('name', platform_key),
            'is_free': platform_info.get('is_free', False),
            'logo': platform_info.get('logo', ''),
            'base_url': platform_info.get('base_url', ''),
            'links': links,
            'availability_confidence': availability_confidence,
            'supported_regions': platform_info.get('supported_regions', []),
            'content_types': platform_info.get('content_types', []),
            'supported_languages': platform_info.get('languages', []),
            'quality_options': platform_info.get('quality', ['HD']),
            'features': platform_info.get('features', [])
        }
        
        if note:
            platform_entry['note'] = note
        
        return platform_entry
    
    @staticmethod
    def _create_search_query(title, language, content_type):
        """Create optimized search query"""
        query = title.strip()
        
        # Add language if not English
        if language != 'english':
            language_names = {
                'hindi': 'hindi', 'telugu': 'telugu', 'tamil': 'tamil',
                'kannada': 'kannada', 'malayalam': 'malayalam', 'bengali': 'bengali'
            }
            if language in language_names:
                query += f" {language_names[language]}"
        
        # Add content type hint
        if content_type == 'movie':
            query += " movie"
        elif content_type == 'tv':
            query += " series"
        
        return query
    
    @staticmethod
    def _generate_search_url(platform_key, search_query):
        """Generate properly formatted search URL"""
        platform_info = OTT_PLATFORMS.get(platform_key, {})
        search_url_template = platform_info.get('search_url', '')
        
        if search_url_template and '{}' in search_url_template:
            # Properly encode the search query
            encoded_query = quote_plus(search_query)
            return search_url_template.format(encoded_query)
        
        return platform_info.get('base_url', '')
    
    @staticmethod
    def _generate_direct_url(platform_key, title, content_type, language=None):
        """Generate direct content URL where possible"""
        platform_info = OTT_PLATFORMS.get(platform_key, {})
        
        # For YouTube, try to get actual video URL
        if platform_key == 'youtube':
            return AdvancedOTTService._get_youtube_direct_url(title, content_type, language)
        
        # For other platforms, use pattern if available
        direct_pattern = platform_info.get('direct_url_pattern')
        if direct_pattern and '{}' in direct_pattern:
            # Create URL-safe title
            safe_title = re.sub(r'[^\w\s-]', '', title).strip()
            safe_title = re.sub(r'[-\s]+', '-', safe_title).lower()
            
            try:
                if platform_key in ['hotstar', 'zee5']:
                    # These platforms need additional ID parameter
                    return None  # Fall back to search URL
                else:
                    return direct_pattern.format(safe_title)
            except:
                pass
        
        return None
    
    @staticmethod
    def _get_youtube_direct_url(title, content_type, language=None):
        """Get direct YouTube URL using API"""
        if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == 'your_youtube_api_key':
            return None
        
        try:
            # Create search query
            search_query = title
            if language and language != 'english':
                search_query += f" {language}"
            
            if content_type == 'movie':
                search_query += " full movie"
            elif content_type == 'tv':
                search_query += " full episodes"
            else:
                search_query += " trailer"
            
            # Search YouTube API
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                'key': YOUTUBE_API_KEY,
                'q': search_query,
                'part': 'snippet',
                'type': 'video',
                'maxResults': 1,
                'order': 'relevance'
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    video_id = data['items'][0]['id']['videoId']
                    return f"https://www.youtube.com/watch?v={video_id}"
        except Exception as e:
            logger.error(f"YouTube URL generation error: {e}")
        
        return None
    
    @staticmethod
    def _determine_quality(platform_key, confidence):
        """Determine video quality based on platform and confidence"""
        platform_info = OTT_PLATFORMS.get(platform_key, {})
        available_qualities = platform_info.get('quality', ['HD'])
        
        if platform_info.get('is_free', False):
            # Free platforms typically offer lower quality
            if confidence > 0.7:
                return 'HD' if 'HD' in available_qualities else 'SD'
            else:
                return 'SD'
        else:
            # Paid platforms offer better quality
            if '4K' in available_qualities and confidence > 0.8:
                return '4K'
            elif 'HD' in available_qualities:
                return 'HD'
            else:
                return 'SD'
    
    @staticmethod
    def search_streaming_apis(title, content_type, tmdb_id=None, imdb_id=None):
        """Search real streaming APIs for verified availability"""
        verified_platforms = []
        
        # TMDB Watch Providers API
        if tmdb_id and TMDB_API_KEY:
            try:
                url = f"https://api.themoviedb.org/3/{content_type}/{tmdb_id}/watch/providers"
                params = {'api_key': TMDB_API_KEY}
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for Indian providers first, then other regions
                    regions_to_check = ['IN', 'US', 'GB']
                    
                    for region in regions_to_check:
                        if region in data.get('results', {}):
                            providers = data['results'][region]
                            
                            # Process different availability types
                            for provider_type in ['flatrate', 'rent', 'buy', 'free']:
                                if provider_type in providers:
                                    for provider in providers[provider_type]:
                                        platform_name = provider.get('provider_name', '').lower()
                                        platform_key = AdvancedOTTService._map_tmdb_provider(platform_name)
                                        
                                        if platform_key:
                                            verified_platforms.append({
                                                'platform': platform_key,
                                                'name': OTT_PLATFORMS[platform_key]['name'],
                                                'type': provider_type,
                                                'region': region,
                                                'logo': f"https://image.tmdb.org/t/p/original{provider.get('logo_path', '')}",
                                                'verified': True,
                                                'tmdb_provider_id': provider.get('provider_id')
                                            })
                            
                            if verified_platforms:
                                break  # Use first region with results
                                
            except Exception as e:
                logger.error(f"TMDB providers API error: {e}")
        
        # JustWatch API (if available)
        # AdvancedOTTService._search_justwatch_api(title, content_type)
        
        return verified_platforms
    
    @staticmethod
    def _map_tmdb_provider(provider_name):
        """Map TMDB provider names to our platform keys"""
        mapping = {
            'netflix': 'netflix',
            'amazon prime video': 'prime_video',
            'disney plus hotstar': 'hotstar',
            'disney+ hotstar': 'hotstar',
            'mx player': 'mx_player',
            'jiocinema': 'jiocinema',
            'sonyliv': 'sonyliv',
            'zee5': 'zee5',
            'youtube': 'youtube',
            'aha': 'aha',
            'sun nxt': 'sun_nxt',
            'voot': 'voot',
            'airtel xstream': 'airtel_xstream',
            'eros now': 'eros_now',
            'altbalaji': 'alt_balaji'
        }
        
        provider_lower = provider_name.lower()
        for key, value in mapping.items():
            if key in provider_lower:
                return value
        
        return None

# Enhanced Content Management Service
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            # Check if content already exists
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                # Update OTT data if it's stale (6 hours)
                if existing.ott_last_updated < datetime.utcnow() - timedelta(hours=6):
                    executor.submit(ContentService.update_ott_availability_async, existing.id)
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
            
            # Get IMDB ID if available
            imdb_id = None
            if 'imdb_id' in tmdb_data:
                imdb_id = tmdb_data['imdb_id']
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
                imdb_id=imdb_id,
                title=tmdb_data.get('title') or tmdb_data.get('name'),
                original_title=tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                content_type=content_type,
                genres=json.dumps(genres),
                languages=json.dumps(languages),
                release_date=datetime.strptime(tmdb_data.get('release_date') or tmdb_data.get('first_air_date', '1900-01-01'), '%Y-%m-%d').date() if tmdb_data.get('release_date') or tmdb_data.get('first_air_date') else None,
                runtime=tmdb_data.get('runtime'),
                rating=tmdb_data.get('vote_average'),
                vote_count=tmdb_data.get('vote_count'),
                popularity=tmdb_data.get('popularity'),
                overview=tmdb_data.get('overview'),
                poster_path=tmdb_data.get('poster_path'),
                backdrop_path=tmdb_data.get('backdrop_path'),
                ott_platforms=json.dumps([]),  # Will be populated
                ott_last_updated=datetime.utcnow()
            )
            
            db.session.add(content)
            db.session.commit()
            
            # Get OTT availability immediately for better UX
            ContentService.update_ott_availability(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def update_ott_availability_async(content_id):
        """Update OTT availability asynchronously"""
        try:
            with app.app_context():
                content = Content.query.get(content_id)
                if content:
                    ContentService.update_ott_availability(content)
        except Exception as e:
            logger.error(f"Error in async OTT update: {e}")
    
    @staticmethod
    def update_ott_availability(content):
        """Update OTT availability with advanced detection"""
        try:
            # Get content languages using advanced detection
            original_lang = None
            if content.languages:
                try:
                    lang_list = json.loads(content.languages)
                    original_lang = lang_list[0] if lang_list else None
                except:
                    pass
            
            # Get genres
            genres = []
            if content.genres:
                try:
                    genres = json.loads(content.genres)
                except:
                    pass
            
            # Detect content languages with advanced algorithm
            content_languages = AdvancedOTTService.detect_content_language(
                content.title, 
                content.overview,
                original_lang,
                genres
            )
            
            # Get release year
            year = content.release_date.year if content.release_date else None
            
            # Get platform availability using advanced service
            ott_data = AdvancedOTTService.get_platform_availability(
                content.title,
                content.content_type,
                content_languages,
                genres,
                year,
                content.popularity
            )
            
            # Try to get verified streaming data from APIs
            if ENABLE_ADVANCED_OTT and content.tmdb_id:
                verified_platforms = AdvancedOTTService.search_streaming_apis(
                    content.title,
                    content.content_type,
                    content.tmdb_id,
                    content.imdb_id
                )
                
                # Merge verified data with predictions
                ContentService._merge_verified_data(ott_data, verified_platforms)
            
            if ott_data:
                content.ott_platforms = json.dumps(ott_data)
                content.ott_last_updated = datetime.utcnow()
                db.session.commit()
                logger.info(f"Updated OTT data for '{content.title}' with {len(ott_data)} platforms")
                
        except Exception as e:
            logger.error(f"Error updating OTT availability for '{content.title}': {e}")
    
    @staticmethod
    def _merge_verified_data(predicted_platforms, verified_platforms):
        """Merge verified API data with predictions"""
        verified_platform_keys = {vp['platform'] for vp in verified_platforms}
        
        for platform in predicted_platforms:
            platform_key = platform['platform']
            if platform_key in verified_platform_keys:
                platform['verified'] = True
                platform['availability_confidence'] = min(platform['availability_confidence'] + 0.3, 1.0)
                
                # Add verified information
                verified_info = next((vp for vp in verified_platforms if vp['platform'] == platform_key), None)
                if verified_info:
                    platform['verification_source'] = 'tmdb'
                    platform['provider_type'] = verified_info.get('type', 'unknown')
                    if verified_info.get('region'):
                        platform['available_regions'] = platform.get('available_regions', [])
                        if verified_info['region'] not in platform['available_regions']:
                            platform['available_regions'].append(verified_info['region'])
    
    @staticmethod
    def map_genre_ids(genre_ids):
        # Enhanced TMDB Genre ID mapping
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

# External API Services (keeping all existing ones)
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
            'append_to_response': 'credits,videos,similar,watch/providers,translations'
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

class YouTubeService:
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    @staticmethod
    def search_trailers(query):
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

# Recommendation Engines (keeping all existing)
class RecommendationEngine:
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all'):
        try:
            # Get trending from TMDB
            trending_data = TMDBService.get_trending(content_type=content_type)
            if not trending_data:
                return []
            
            recommendations = []
            for item in trending_data.get('results', [])[:limit]:
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    recommendations.append(content)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_popular_by_genre(genre, limit=20, region=None):
        try:
            # First get popular movies
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
            # Search for content in specific language
            search_queries = {
                'hindi': ['bollywood', 'hindi movie', 'hindi film'],
                'telugu': ['tollywood', 'telugu movie', 'telugu film'],
                'tamil': ['kollywood', 'tamil movie', 'tamil film'],
                'kannada': ['sandalwood', 'kannada movie', 'kannada film'],
                'malayalam': ['mollywood', 'malayalam movie', 'malayalam film'],
                'bengali': ['bengali movie', 'bengali film', 'tollywood bengali'],
                'marathi': ['marathi movie', 'marathi film'],
                'gujarati': ['gujarati movie', 'gujarati film'],
                'punjabi': ['punjabi movie', 'punjabi film']
            }
            
            recommendations = []
            queries = search_queries.get(language.lower(), [language])
            
            for query in queries:
                search_results = TMDBService.search_content(query)
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
                # Convert anime data to our content format
                content = Content(
                    title=anime.get('title'),
                    original_title=anime.get('title_japanese'),
                    content_type='anime',
                    genres=json.dumps([genre['name'] for genre in anime.get('genres', [])]),
                    languages=json.dumps(['japanese']),
                    rating=anime.get('score'),
                    overview=anime.get('synopsis'),
                    poster_path=anime.get('images', {}).get('jpg', {}).get('image_url'),
                    ott_platforms=json.dumps([])
                )
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
# Enhanced Telegram Service with Direct Watch Links
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
            
            # Get OTT platforms with direct watch links
            ott_links = TelegramService._format_ott_links(content.ott_platforms)
            
            # Create enhanced message with watch links
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}

{ott_links}

#AdminChoice #MovieRecommendation #CineScope"""
            
            # Send message with photo if available
            if poster_url:
                try:
                    bot.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=poster_url,
                        caption=message,
                        parse_mode='Markdown',
                        disable_web_page_preview=False
                    )
                except Exception as photo_error:
                    logger.error(f"Failed to send photo, sending text only: {photo_error}")
                    bot.send_message(
                        TELEGRAM_CHANNEL_ID, 
                        message, 
                        parse_mode='Markdown',
                        disable_web_page_preview=False
                    )
            else:
                bot.send_message(
                    TELEGRAM_CHANNEL_ID, 
                    message, 
                    parse_mode='Markdown',
                    disable_web_page_preview=False
                )
            
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    @staticmethod
    def _format_ott_links(ott_platforms_json):
        """Format OTT platform links for Telegram message"""
        if not ott_platforms_json:
            return "📺 **Watch On:**\n🔍 Check your favorite streaming platforms!"
        
        try:
            platforms = json.loads(ott_platforms_json)
            if not platforms:
                return "📺 **Watch On:**\n🔍 Check your favorite streaming platforms!"
            
            # Separate free and paid platforms
            free_platforms = []
            paid_platforms = []
            
            for platform in platforms[:8]:  # Limit to top 8 platforms
                platform_info = TelegramService._extract_platform_info(platform)
                if platform_info:
                    if platform.get('is_free', False):
                        free_platforms.append(platform_info)
                    else:
                        paid_platforms.append(platform_info)
            
            # Build the formatted message
            watch_section = "📺 **Watch On:**\n"
            
            # Add free platforms first
            if free_platforms:
                watch_section += "\n🆓 **FREE:**\n"
                for i, platform_info in enumerate(free_platforms[:4], 1):  # Max 4 free platforms
                    watch_section += f"{i}. {platform_info}\n"
            
            # Add paid platforms
            if paid_platforms:
                watch_section += "\n💰 **PREMIUM:**\n"
                for i, platform_info in enumerate(paid_platforms[:4], 1):  # Max 4 paid platforms
                    watch_section += f"{i}. {platform_info}\n"
            
            # Add helpful note
            if len(platforms) > 8:
                watch_section += f"\n*+{len(platforms) - 8} more platforms available*"
            
            watch_section += "\n💡 *Tip: Click the links to watch directly!*"
            
            return watch_section
            
        except Exception as e:
            logger.error(f"Error formatting OTT links: {e}")
            return "📺 **Watch On:**\n🔍 Check your favorite streaming platforms!"
    
    @staticmethod
    def _extract_platform_info(platform):
        """Extract and format platform information with links"""
        try:
            platform_name = platform.get('name', platform.get('platform', 'Unknown'))
            is_free = platform.get('is_free', False)
            confidence = platform.get('availability_confidence', 0)
            verified = platform.get('verified', False)
            links = platform.get('links', {})
            
            # Get the best available link
            best_link = TelegramService._get_best_link(links, platform_name)
            
            if not best_link:
                return None
            
            # Create status indicators
            status_indicators = []
            if verified:
                status_indicators.append("✅")
            elif confidence > 0.7:
                status_indicators.append("🎯")
            elif confidence > 0.5:
                status_indicators.append("🎲")
            
            if platform.get('note'):
                status_indicators.append("ℹ️")
            
            # Format quality info
            quality_info = ""
            if best_link.get('quality'):
                quality = best_link['quality']
                if quality == '4K':
                    quality_info = " (4K)"
                elif quality == 'HD':
                    quality_info = " (HD)"
            
            # Create the formatted link
            status_text = "".join(status_indicators)
            watch_url = best_link['watch_url']
            
            # Create clickable link
            if watch_url and watch_url.startswith('http'):
                platform_text = f"[{platform_name}{quality_info}]({watch_url})"
            else:
                platform_text = f"{platform_name}{quality_info}"
            
            # Add language info if multiple languages available
            language_info = ""
            if len(links) > 1:
                languages = list(links.keys())
                if 'default' in languages:
                    languages.remove('default')
                if languages:
                    language_info = f" ({', '.join(languages[:2])}{'...' if len(languages) > 2 else ''})"
            
            # Add subscription info for paid platforms
            subscription_info = ""
            if not is_free and best_link.get('subscription_required', True):
                subscription_info = " 🔐"
            
            return f"{status_text} {platform_text}{language_info}{subscription_info}"
            
        except Exception as e:
            logger.error(f"Error extracting platform info: {e}")
            return None
    
    @staticmethod
    def _get_best_link(links, platform_name):
        """Get the best available link from platform links"""
        if not links:
            return None
        
        # Priority order for language selection
        priority_languages = ['hindi', 'english', 'tamil', 'telugu', 'default']
        
        # Special handling for YouTube - prefer English for wider audience
        if 'youtube' in platform_name.lower():
            priority_languages = ['english', 'hindi', 'default']
        
        # Find the best link based on language priority
        for lang in priority_languages:
            if lang in links:
                link_info = links[lang]
                if link_info.get('watch_url'):
                    return link_info
        
        # If no priority language found, return first available link
        for lang, link_info in links.items():
            if link_info.get('watch_url'):
                return link_info
        
        return None
    
    @staticmethod
    def send_weekly_digest(top_content, stats):
        """Send weekly digest with top content and watch links"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                logger.warning("Telegram bot or channel ID not configured")
                return False
            
            message = f"""📊 **Weekly CineScope Digest**

🔥 **This Week's Top Picks:**

"""
            
            for i, content in enumerate(top_content[:5], 1):
                # Get OTT links for each content
                ott_summary = TelegramService._get_ott_summary(content.ott_platforms)
                
                message += f"""**{i}. {content.title}**
⭐ {content.rating or 'N/A'}/10 | 🎭 {content.content_type.upper()}
{ott_summary}

"""
            
            message += f"""📈 **Platform Stats:**
🎬 Total Content: {stats.get('total_content', 0)}
👥 Active Users: {stats.get('active_users', 0)}
🔥 Most Popular: {stats.get('top_platform', 'Netflix')}

#WeeklyDigest #CineScope #Streaming"""
            
            bot.send_message(
                TELEGRAM_CHANNEL_ID, 
                message, 
                parse_mode='Markdown',
                disable_web_page_preview=False
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Weekly digest send error: {e}")
            return False
    
    @staticmethod
    def _get_ott_summary(ott_platforms_json):
        """Get a brief summary of OTT availability"""
        try:
            if not ott_platforms_json:
                return "📺 Check streaming platforms"
            
            platforms = json.loads(ott_platforms_json)
            if not platforms:
                return "📺 Check streaming platforms"
            
            # Get top 3 platforms
            top_platforms = platforms[:3]
            platform_names = []
            
            for platform in top_platforms:
                name = platform.get('name', platform.get('platform', ''))
                if platform.get('is_free'):
                    platform_names.append(f"{name} (Free)")
                else:
                    platform_names.append(name)
            
            return f"📺 {', '.join(platform_names)}"
            
        except:
            return "📺 Check streaming platforms"
    
    @staticmethod
    def send_new_content_alert(content, content_type="new_release"):
        """Send alert for new content with watch links"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            # Get content details
            genres = []
            if content.genres:
                try:
                    genres = json.loads(content.genres)[:2]  # Top 2 genres
                except:
                    pass
            
            # Get poster
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Format OTT links
            ott_links = TelegramService._format_ott_links(content.ott_platforms)
            
            # Create alert message
            alert_emoji = "🆕" if content_type == "new_release" else "🔥"
            content_type_text = "NEW RELEASE" if content_type == "new_release" else "HOT PICK"
            
            message = f"""{alert_emoji} **{content_type_text}**

**{content.title}**
⭐ {content.rating or 'N/A'}/10
🎭 {', '.join(genres) if genres else 'Entertainment'}
📅 {content.release_date or 'Recent'}

📖 {(content.overview[:150] + '...') if content.overview else 'New content alert!'}

{ott_links}

#NewRelease #CineScope #Streaming"""
            
            if poster_url:
                bot.send_photo(
                    chat_id=TELEGRAM_CHANNEL_ID,
                    photo=poster_url,
                    caption=message,
                    parse_mode='Markdown',
                    disable_web_page_preview=False
                )
            else:
                bot.send_message(
                    TELEGRAM_CHANNEL_ID,
                    message,
                    parse_mode='Markdown',
                    disable_web_page_preview=False
                )
            
            return True
            
        except Exception as e:
            logger.error(f"New content alert error: {e}")
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
                    
                    # Format OTT platforms for response
                    ott_platforms = []
                    if content.ott_platforms:
                        try:
                            ott_platforms = json.loads(content.ott_platforms)
                        except:
                            ott_platforms = []
                    
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
                        'ott_platforms': ott_platforms
                    })
        
        # Add anime results
        if anime_results:
            for anime in anime_results.get('data', []):
                results.append({
                    'id': f"anime_{anime['mal_id']}",
                    'title': anime.get('title'),
                    'content_type': 'anime',
                    'genres': [genre['name'] for genre in anime.get('genres', [])],
                    'rating': anime.get('score'),
                    'release_date': anime.get('aired', {}).get('from'),
                    'poster_path': anime.get('images', {}).get('jpg', {}).get('image_url'),
                    'overview': anime.get('synopsis'),
                    'ott_platforms': []
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
        
        # Update OTT data if it's stale (6 hours)
        if content.ott_last_updated < datetime.utcnow() - timedelta(hours=6):
            executor.submit(ContentService.update_ott_availability_async, content.id)
        
        # Get additional details from TMDB if available
        additional_details = None
        if content.tmdb_id:
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        
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
        if additional_details and 'similar' in additional_details:
            for item in additional_details['similar']['results'][:5]:
                similar = ContentService.save_content_from_tmdb(item, content.content_type)
                if similar:
                    similar_content.append({
                        'id': similar.id,
                        'title': similar.title,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{similar.poster_path}" if similar.poster_path else None,
                        'rating': similar.rating
                    })
        
        # Format OTT platforms for response with enhanced details
        ott_platforms = []
        if content.ott_platforms:
            try:
                ott_platforms = json.loads(content.ott_platforms)
            except:
                ott_platforms = []
        
        db.session.commit()
        
        return jsonify({
            'id': content.id,
            'tmdb_id': content.tmdb_id,
            'imdb_id': content.imdb_id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'runtime': content.runtime,
            'rating': content.rating,
            'vote_count': content.vote_count,
            'popularity': content.popularity,
            'overview': content.overview,
            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path else None,
            'ott_platforms': ott_platforms,
            'ott_last_updated': content.ott_last_updated.isoformat() if content.ott_last_updated else None,
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Enhanced OTT refresh endpoint
@app.route('/api/content/<int:content_id>/refresh-ott', methods=['POST'])
def refresh_ott_data(content_id):
    try:
        content = Content.query.get_or_404(content_id)
        
        # Force update OTT availability
        ContentService.update_ott_availability(content)
        
        # Return updated OTT platforms
        ott_platforms = []
        if content.ott_platforms:
            try:
                ott_platforms = json.loads(content.ott_platforms)
            except:
                ott_platforms = []
        
        return jsonify({
            'message': 'OTT data refreshed successfully',
            'ott_platforms': ott_platforms,
            'last_updated': content.ott_last_updated.isoformat(),
            'platforms_found': len(ott_platforms),
            'verified_platforms': len([p for p in ott_platforms if p.get('verified', False)])
        }), 200
        
    except Exception as e:
        logger.error(f"OTT refresh error: {e}")
        return jsonify({'error': 'Failed to refresh OTT data'}), 500

# Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = []
        for content in recommendations:
            # Format OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': ott_platforms
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Trending recommendations error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/recommendations/popular/<genre>', methods=['GET'])
def get_popular_by_genre(genre):
    try:
        limit = int(request.args.get('limit', 20))
        region = request.args.get('region')
        
        recommendations = RecommendationEngine.get_popular_by_genre(genre, limit, region)
        
        result = []
        for content in recommendations:
            # Format OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': ott_platforms
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Popular by genre error: {e}")
        return jsonify({'error': 'Failed to get popular recommendations'}), 500

@app.route('/api/recommendations/regional/<language>', methods=['GET'])
def get_regional(language):
    try:
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_regional_recommendations(language, limit)
        
        result = []
        for content in recommendations:
            # Format OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': ott_platforms
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
            # Format OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': ott_platforms
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
                        # Format OTT platforms
                        ott_platforms = []
                        if content.ott_platforms:
                            try:
                                ott_platforms = json.loads(content.ott_platforms)
                            except:
                                ott_platforms = []
                        
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'ott_platforms': ott_platforms,
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
            # Format OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': ott_platforms
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
            # Format OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': ott_platforms
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
        if data.get('id'):
            # Check by TMDB ID or other external ID
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
                    release_date = datetime.strptime(data['release_date'], '%Y-%m-%d').date()
                except:
                    release_date = None
            
            # Create content object
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
                ott_platforms=json.dumps([]),  # Will be populated
                ott_last_updated=datetime.utcnow()
            )
            
            db.session.add(content)
            db.session.commit()
            
            # Generate OTT info immediately
            ContentService.update_ott_availability(content)
            
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
            
            # Format OTT platforms
            ott_platforms = []
            if content and content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
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
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'ott_platforms': ott_platforms
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
        
        # OTT platform analytics
        ott_platform_counts = defaultdict(int)
        ott_platform_availability = defaultdict(int)
        contents_with_ott = Content.query.filter(Content.ott_platforms.isnot(None)).all()
        
        for content in contents_with_ott:
            try:
                platforms = json.loads(content.ott_platforms)
                for platform in platforms:
                    platform_name = platform.get('name', platform.get('platform', 'Unknown'))
                    ott_platform_counts[platform_name] += 1
                    if platform.get('verified', False):
                        ott_platform_availability[platform_name] += 1
            except:
                continue
        
        popular_ott_platforms = sorted(ott_platform_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
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
            ],
            'popular_ott_platforms': [
                {
                    'platform': platform, 
                    'content_count': count,
                    'verified_count': ott_platform_availability.get(platform, 0)
                }
                for platform, count in popular_ott_platforms
            ],
            'ott_detection_stats': {
                'total_content_with_ott': len(contents_with_ott),
                'total_platforms_tracked': len(OTT_PLATFORMS),
                'average_platforms_per_content': sum(ott_platform_counts.values()) / len(contents_with_ott) if contents_with_ott else 0
            }
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
                # Format OTT platforms
                ott_platforms = []
                if content.ott_platforms:
                    try:
                        ott_platforms = json.loads(content.ott_platforms)
                    except:
                        ott_platforms = []
                
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'ott_platforms': ott_platforms,
                    'admin_description': rec.description,
                    'admin_name': admin.username if admin else 'Admin',
                    'recommended_at': rec.created_at.isoformat()
                })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Public admin recommendations error: {e}")
        return jsonify({'error': 'Failed to get admin recommendations'}), 500

# Enhanced OTT Platform Routes
@app.route('/api/ott-platforms', methods=['GET'])
def get_ott_platforms():
    """Get comprehensive list of all supported OTT platforms"""
    try:
        platforms = []
        for platform_key, platform_info in OTT_PLATFORMS.items():
            platforms.append({
                'key': platform_key,
                'name': platform_info['name'],
                'is_free': platform_info['is_free'],
                'base_url': platform_info['base_url'],
                'logo': platform_info.get('logo', ''),
                'supported_regions': platform_info.get('supported_regions', []),
                'content_types': platform_info.get('content_types', []),
                'languages': platform_info.get('languages', []),
                'quality_options': platform_info.get('quality', ['HD']),
                'features': platform_info.get('features', [])
            })
        
        # Get usage statistics
        platform_usage = defaultdict(int)
        contents_with_ott = Content.query.filter(Content.ott_platforms.isnot(None)).all()
        
        for content in contents_with_ott:
            try:
                ott_platforms = json.loads(content.ott_platforms)
                for platform in ott_platforms:
                    platform_key = platform.get('platform', '')
                    if platform_key:
                        platform_usage[platform_key] += 1
            except:
                continue
        
        # Add usage stats to platforms
        for platform in platforms:
            platform['usage_count'] = platform_usage.get(platform['key'], 0)
        
        # Sort by usage
        platforms.sort(key=lambda x: x['usage_count'], reverse=True)
        
        return jsonify({
            'platforms': platforms,
            'total_platforms': len(platforms),
            'free_platforms': len([p for p in platforms if p['is_free']]),
            'paid_platforms': len([p for p in platforms if not p['is_free']]),
            'regional_platforms': len([p for p in platforms if 'IN' in p.get('supported_regions', [])]),
            'global_platforms': len([p for p in platforms if 'global' in p.get('supported_regions', []) or len(p.get('supported_regions', [])) > 3])
        }), 200
        
    except Exception as e:
        logger.error(f"Get OTT platforms error: {e}")
        return jsonify({'error': 'Failed to get OTT platforms'}), 500

@app.route('/api/ott-platforms/<platform_key>/content', methods=['GET'])
def get_platform_content(platform_key):
    """Get content available on specific platform"""
    try:
        if platform_key not in OTT_PLATFORMS:
            return jsonify({'error': 'Platform not found'}), 404
        
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        # Find content with this platform
        contents = Content.query.filter(Content.ott_platforms.isnot(None)).all()
        platform_content = []
        
        for content in contents:
            try:
                ott_platforms = json.loads(content.ott_platforms)
                for platform in ott_platforms:
                    if platform.get('platform') == platform_key:
                        platform_content.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                            'availability_confidence': platform.get('availability_confidence', 0),
                            'verified': platform.get('verified', False),
                            'links': platform.get('links', {})
                        })
                        break
            except:
                continue
        
        # Paginate results
        total = len(platform_content)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_content = platform_content[start:end]
        
        return jsonify({
            'platform': OTT_PLATFORMS[platform_key],
            'content': paginated_content,
            'total_content': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        }), 200
        
    except Exception as e:
        logger.error(f"Get platform content error: {e}")
        return jsonify({'error': 'Failed to get platform content'}), 500

# Enhanced Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    # Test database connection
    db_status = 'healthy'
    try:
        db.session.execute(text('SELECT 1'))
    except:
        db_status = 'unhealthy'
    
    # Test external APIs
    api_status = {}
    try:
        if TMDB_API_KEY and TMDB_API_KEY != 'your_tmdb_api_key':
            response = requests.get(f"https://api.themoviedb.org/3/configuration?api_key={TMDB_API_KEY}", timeout=5)
            api_status['tmdb'] = 'healthy' if response.status_code == 200 else 'unhealthy'
        else:
            api_status['tmdb'] = 'not_configured'
    except:
        api_status['tmdb'] = 'unhealthy'
    
    return jsonify({
        'status': 'healthy' if db_status == 'healthy' else 'degraded',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '3.0.0',
        'environment': 'production' if IS_PRODUCTION else 'development',
        'database': db_status,
        'external_apis': api_status,
        'features': {
            'advanced_ott_detection': True,
            'real_streaming_apis': ENABLE_ADVANCED_OTT,
            'multiple_languages': True,
            'direct_links': True,
            'verified_availability': True,
            'smart_platform_matching': True,
            'youtube_integration': YOUTUBE_API_KEY != 'your_youtube_api_key',
            'telegram_notifications': bot is not None
        },
        'supported_platforms': len(OTT_PLATFORMS),
        'supported_languages': len(REGIONAL_LANGUAGES),
        'total_content': Content.query.count(),
        'total_users': User.query.count()
    }), 200

# Root endpoint
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Enhanced Movie Recommendation API with Smart OTT Integration',
        'version': '3.0.0',
        'status': 'running',
        'environment': 'production' if IS_PRODUCTION else 'development',
        'features': [
            'Smart OTT platform detection',
            'Multi-language content support',
            'Real streaming availability',
            'Direct watch links',
            'Verified platform data',
            'Advanced language detection',
            'Regional content recommendations',
            'Admin content management',
            'Telegram notifications',
            'Personalized recommendations'
        ],
        'endpoints': {
            'search': '/api/search',
            'recommendations': '/api/recommendations',
            'ott_platforms': '/api/ott-platforms',
            'admin': '/api/admin',
            'health': '/api/health'
        }
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