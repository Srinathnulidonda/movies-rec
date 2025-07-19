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

# Streaming API Keys
WATCHMODE_API_KEY = os.environ.get('WATCHMODE_API_KEY', 'your_watchmode_api_key')
RAPIDAPI_KEY = "c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37"
RAPIDAPI_HOST = "streaming-availability.p.rapidapi.com"

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
    ott_platforms = db.Column(db.Text)  # JSON string - Enhanced for streaming
    streaming_links = db.Column(db.Text)  # JSON string - New field for detailed streaming info
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

# Enhanced OTT Platform Information with direct links and categories
OTT_PLATFORMS = {
    # Free Platforms
    'mx_player': {
        'name': 'MX Player', 'is_free': True, 'url': 'https://mxplayer.in',
        'search_url': 'https://mxplayer.in/search?q=',
        'category': 'free', 'country': 'IN', 'icon': '🎬'
    },
    'jiocinema': {
        'name': 'JioCinema', 'is_free': True, 'url': 'https://jiocinema.com',
        'search_url': 'https://jiocinema.com/search/',
        'category': 'free', 'country': 'IN', 'icon': '🎭'
    },
    'sonyliv_free': {
        'name': 'SonyLIV Free', 'is_free': True, 'url': 'https://sonyliv.com',
        'search_url': 'https://sonyliv.com/search?q=',
        'category': 'free', 'country': 'IN', 'icon': '📺'
    },
    'zee5_free': {
        'name': 'Zee5 Free', 'is_free': True, 'url': 'https://zee5.com',
        'search_url': 'https://zee5.com/search?q=',
        'category': 'free', 'country': 'IN', 'icon': '🎪'
    },
    'youtube': {
        'name': 'YouTube', 'is_free': True, 'url': 'https://youtube.com',
        'search_url': 'https://youtube.com/results?search_query=',
        'category': 'free', 'country': 'GLOBAL', 'icon': '▶️'
    },
    'crunchyroll_free': {
        'name': 'Crunchyroll', 'is_free': True, 'url': 'https://crunchyroll.com',
        'search_url': 'https://crunchyroll.com/search?q=',
        'category': 'free', 'country': 'GLOBAL', 'icon': '🌸'
    },
    'airtel_xstream': {
        'name': 'Airtel Xstream', 'is_free': True, 'url': 'https://airtelxstream.in',
        'search_url': 'https://airtelxstream.in/search?q=',
        'category': 'free', 'country': 'IN', 'icon': '📡'
    },
    
    # Paid Platforms
    'netflix': {
        'name': 'Netflix', 'is_free': False, 'url': 'https://netflix.com',
        'search_url': 'https://netflix.com/search?q=',
        'category': 'paid', 'country': 'GLOBAL', 'icon': '🔴'
    },
    'amazon_prime': {
        'name': 'Prime Video', 'is_free': False, 'url': 'https://primevideo.com',
        'search_url': 'https://primevideo.com/search/ref=atv_nb_sr?phrase=',
        'category': 'paid', 'country': 'GLOBAL', 'icon': '📦'
    },
    'disney_plus_hotstar': {
        'name': 'Disney+ Hotstar', 'is_free': False, 'url': 'https://hotstar.com',
        'search_url': 'https://hotstar.com/search?q=',
        'category': 'paid', 'country': 'IN', 'icon': '🏰'
    },
    'zee5_premium': {
        'name': 'Zee5 Premium', 'is_free': False, 'url': 'https://zee5.com',
        'search_url': 'https://zee5.com/search?q=',
        'category': 'paid', 'country': 'IN', 'icon': '👑'
    },
    'sonyliv_premium': {
        'name': 'SonyLIV Premium', 'is_free': False, 'url': 'https://sonyliv.com',
        'search_url': 'https://sonyliv.com/search?q=',
        'category': 'paid', 'country': 'IN', 'icon': '💎'
    },
    'aha': {
        'name': 'Aha', 'is_free': False, 'url': 'https://aha.video',
        'search_url': 'https://aha.video/search?q=',
        'category': 'paid', 'country': 'IN', 'icon': '🎨'
    },
    'sun_nxt': {
        'name': 'Sun NXT', 'is_free': False, 'url': 'https://sunnxt.com',
        'search_url': 'https://sunnxt.com/search?q=',
        'category': 'paid', 'country': 'IN', 'icon': '☀️'
    }
}

# Enhanced Regional Language Mapping with priorities
REGIONAL_LANGUAGES = {
    'telugu': {
        'codes': ['te', 'telugu'], 
        'priority': 1, 
        'industry': 'Tollywood',
        'region': 'AP/TG',
        'platforms': ['aha', 'zee5_premium', 'sun_nxt', 'disney_plus_hotstar']
    },
    'hindi': {
        'codes': ['hi', 'hindi'], 
        'priority': 2, 
        'industry': 'Bollywood',
        'region': 'Hindi Belt',
        'platforms': ['zee5_premium', 'netflix', 'amazon_prime', 'disney_plus_hotstar']
    },
    'tamil': {
        'codes': ['ta', 'tamil'], 
        'priority': 3, 
        'industry': 'Kollywood',
        'region': 'TN',
        'platforms': ['sun_nxt', 'zee5_premium', 'disney_plus_hotstar', 'netflix']
    },
    'malayalam': {
        'codes': ['ml', 'malayalam'], 
        'priority': 4, 
        'industry': 'Mollywood',
        'region': 'Kerala',
        'platforms': ['disney_plus_hotstar', 'netflix', 'amazon_prime', 'zee5_premium']
    },
    'kannada': {
        'codes': ['kn', 'kannada'], 
        'priority': 5, 
        'industry': 'Sandalwood',
        'region': 'Karnataka',
        'platforms': ['zee5_premium', 'sun_nxt', 'disney_plus_hotstar', 'netflix']
    },
    'english': {
        'codes': ['en', 'english'], 
        'priority': 6, 
        'industry': 'Hollywood',
        'region': 'Global',
        'platforms': ['netflix', 'amazon_prime', 'disney_plus_hotstar', 'youtube']
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

# Streaming Availability Service
class StreamingAvailabilityService:
    RAPIDAPI_HEADERS = {
        'x-rapidapi-key': RAPIDAPI_KEY,
        'x-rapidapi-host': RAPIDAPI_HOST
    }
    
    @staticmethod
    def search_streaming_by_title(title, country='IN'):
        """Search for streaming availability by title"""
        try:
            url = f"https://{RAPIDAPI_HOST}/search/title"
            params = {
                'title': title,
                'country': country,
                'show_type': 'all',
                'output_language': 'en'
            }
            
            response = requests.get(url, headers=StreamingAvailabilityService.RAPIDAPI_HEADERS, 
                                  params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Streaming API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Streaming search error: {e}")
            return None
    
    @staticmethod
    def get_streaming_by_imdb(imdb_id, country='IN'):
        """Get streaming availability by IMDB ID"""
        try:
            url = f"https://{RAPIDAPI_HOST}/get"
            params = {
                'imdb_id': imdb_id,
                'country': country,
                'output_language': 'en'
            }
            
            response = requests.get(url, headers=StreamingAvailabilityService.RAPIDAPI_HEADERS, 
                                  params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Streaming API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Streaming get error: {e}")
            return None
    
    @staticmethod
    def search_watchmode(title, content_type='movie'):
        """Search using WatchMode API as fallback"""
        try:
            if not WATCHMODE_API_KEY or WATCHMODE_API_KEY == 'your_watchmode_api_key':
                return None
                
            url = "https://api.watchmode.com/v1/search/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'search_field': 'name',
                'search_value': title,
                'types': content_type
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                results = response.json()
                if results.get('title_results'):
                    # Get detailed info for first result
                    title_id = results['title_results'][0]['id']
                    return StreamingAvailabilityService.get_watchmode_details(title_id)
            
            return None
        except Exception as e:
            logger.error(f"WatchMode search error: {e}")
            return None
    
    @staticmethod
    def get_watchmode_details(title_id):
        """Get detailed streaming info from WatchMode"""
        try:
            url = f"https://api.watchmode.com/v1/title/{title_id}/details/"
            params = {
                'apiKey': WATCHMODE_API_KEY,
                'append_to_response': 'sources'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            
            return None
        except Exception as e:
            logger.error(f"WatchMode details error: {e}")
            return None
    
    @staticmethod
    def parse_streaming_data(streaming_data, title):
        """Parse and normalize streaming data from various sources"""
        streaming_links = {
            'free_options': [],
            'paid_options': [],
            'languages': {},
            'last_updated': datetime.utcnow().isoformat()
        }
        
        if not streaming_data:
            return streaming_links
        
        try:
            # Handle RapidAPI response format
            if 'result' in streaming_data:
                result = streaming_data['result']
                streaming_services = result.get('streamingInfo', {})
                
                for country, services in streaming_services.items():
                    if country.lower() != 'in':  # Focus on India
                        continue
                        
                    for service_name, service_data in services.items():
                        platform_info = StreamingAvailabilityService.map_service_to_platform(service_name)
                        
                        if platform_info:
                            stream_info = {
                                'platform': service_name,
                                'platform_name': platform_info['name'],
                                'url': service_data.get('link', platform_info['url']),
                                'is_free': platform_info['is_free'],
                                'quality': service_data.get('quality', 'HD'),
                                'availability': 'Available Now',
                                'icon': platform_info.get('icon', '🎬'),
                                'languages': service_data.get('audios', []),
                                'subtitles': service_data.get('subtitles', [])
                            }
                            
                            if platform_info['is_free']:
                                streaming_links['free_options'].append(stream_info)
                            else:
                                streaming_links['paid_options'].append(stream_info)
            
            # Handle WatchMode response format
            elif 'sources' in streaming_data:
                for source in streaming_data['sources']:
                    platform_info = StreamingAvailabilityService.map_service_to_platform(source.get('name', ''))
                    
                    if platform_info:
                        stream_info = {
                            'platform': source.get('name', '').lower().replace(' ', '_'),
                            'platform_name': platform_info['name'],
                            'url': source.get('web_url', platform_info['url']),
                            'is_free': source.get('type') == 'free' or platform_info['is_free'],
                            'quality': 'HD',
                            'availability': 'Available Now',
                            'icon': platform_info.get('icon', '🎬')
                        }
                        
                        if stream_info['is_free']:
                            streaming_links['free_options'].append(stream_info)
                        else:
                            streaming_links['paid_options'].append(stream_info)
            
            # Add fallback platforms if no results found
            if not streaming_links['free_options'] and not streaming_links['paid_options']:
                streaming_links = StreamingAvailabilityService.get_fallback_platforms(title)
            
            # Add language-specific versions
            streaming_links['languages'] = StreamingAvailabilityService.create_language_links(
                streaming_links, title
            )
            
        except Exception as e:
            logger.error(f"Error parsing streaming data: {e}")
            # Return fallback data on error
            streaming_links = StreamingAvailabilityService.get_fallback_platforms(title)
        
        return streaming_links
    
    @staticmethod
    def get_fallback_platforms(title):
        """Provide fallback streaming options when API fails"""
        return {
            'free_options': [
                {
                    'platform': 'youtube',
                    'platform_name': 'YouTube',
                    'url': f"https://youtube.com/results?search_query={title.replace(' ', '+')}+full+movie",
                    'is_free': True,
                    'quality': 'HD',
                    'availability': 'Search Results',
                    'icon': '▶️'
                },
                {
                    'platform': 'mx_player',
                    'platform_name': 'MX Player',
                    'url': f"https://mxplayer.in/search?q={title.replace(' ', '%20')}",
                    'is_free': True,
                    'quality': 'HD',
                    'availability': 'Search Results',
                    'icon': '🎬'
                },
                {
                    'platform': 'jiocinema',
                    'platform_name': 'JioCinema',
                    'url': f"https://jiocinema.com/search/{title.replace(' ', '%20')}",
                    'is_free': True,
                    'quality': 'HD',
                    'availability': 'Search Results',
                    'icon': '🎭'
                }
            ],
            'paid_options': [
                {
                    'platform': 'netflix',
                    'platform_name': 'Netflix',
                    'url': f"https://netflix.com/search?q={title.replace(' ', '%20')}",
                    'is_free': False,
                    'quality': 'HD',
                    'availability': 'Search Results',
                    'icon': '🔴'
                },
                {
                    'platform': 'amazon_prime',
                    'platform_name': 'Prime Video',
                    'url': f"https://primevideo.com/search/ref=atv_nb_sr?phrase={title.replace(' ', '%20')}",
                    'is_free': False,
                    'quality': 'HD',
                    'availability': 'Search Results',
                    'icon': '📦'
                },
                {
                    'platform': 'disney_plus_hotstar',
                    'platform_name': 'Disney+ Hotstar',
                    'url': f"https://hotstar.com/search?q={title.replace(' ', '%20')}",
                    'is_free': False,
                    'quality': 'HD',
                    'availability': 'Search Results',
                    'icon': '🏰'
                }
            ],
            'languages': {},
            'last_updated': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def map_service_to_platform(service_name):
        """Map external service names to our platform definitions"""
        service_mapping = {
            'netflix': 'netflix',
            'amazon': 'amazon_prime',
            'prime': 'amazon_prime',
            'primevideo': 'amazon_prime',
            'disney': 'disney_plus_hotstar',
            'hotstar': 'disney_plus_hotstar',
            'zee5': 'zee5_premium',
            'sonyliv': 'sonyliv_premium',
            'jiocinema': 'jiocinema',
            'mx': 'mx_player',
            'mxplayer': 'mx_player',
            'youtube': 'youtube',
            'aha': 'aha',
            'sunnxt': 'sun_nxt',
            'airtel': 'airtel_xstream',
            'crunchyroll': 'crunchyroll_free'
        }
        
        service_lower = service_name.lower().replace(' ', '').replace('+', '')
        
        for key, platform in service_mapping.items():
            if key in service_lower:
                return OTT_PLATFORMS.get(platform)
        
        return None
    
    @staticmethod
    def create_language_links(streaming_links, title):
        """Create language-specific streaming links"""
        language_links = {}
        
        # Get all available platforms
        all_platforms = streaming_links['free_options'] + streaming_links['paid_options']
        
        for lang_code, lang_info in REGIONAL_LANGUAGES.items():
            language_links[lang_code] = {
                'language': lang_code.title(),
                'industry': lang_info['industry'],
                'links': []
            }
            
            # Prioritize platforms based on language
            preferred_platforms = lang_info.get('platforms', [])
            
            for platform_data in all_platforms:
                platform_key = platform_data.get('platform', '')
                
                # Create language-specific URL
                lang_url = StreamingAvailabilityService.create_language_url(
                    platform_data['url'], title, lang_code
                )
                
                lang_link = {
                    'platform': platform_data['platform_name'],
                    'url': lang_url,
                    'is_free': platform_data['is_free'],
                    'icon': platform_data.get('icon', '🎬'),
                    'quality': platform_data.get('quality', 'HD'),
                    'button_text': f"🔘 Watch in {lang_code.title()}",
                    'priority': 1 if platform_key in preferred_platforms else 2
                }
                
                language_links[lang_code]['links'].append(lang_link)
            
            # Sort by priority and free options first
            language_links[lang_code]['links'].sort(
                key=lambda x: (x['priority'], not x['is_free'])
            )
        
        return language_links
    
    @staticmethod
    def create_language_url(base_url, title, language):
        """Create language-specific URLs for streaming platforms"""
        # Create search queries with language specification
        lang_queries = {
            'hindi': f"{title} hindi",
            'telugu': f"{title} telugu",
            'tamil': f"{title} tamil",
            'malayalam': f"{title} malayalam",
            'kannada': f"{title} kannada",
            'english': f"{title} english"
        }
        
        search_query = lang_queries.get(language, title)
        
        # Platform-specific URL modifications
        if 'netflix.com' in base_url:
            return f"https://netflix.com/search?q={search_query.replace(' ', '%20')}"
        elif 'primevideo.com' in base_url:
            return f"https://primevideo.com/search/ref=atv_nb_sr?phrase={search_query.replace(' ', '%20')}"
        elif 'hotstar.com' in base_url:
            return f"https://hotstar.com/search?q={search_query.replace(' ', '%20')}"
        elif 'zee5.com' in base_url:
            return f"https://zee5.com/search?q={search_query.replace(' ', '%20')}"
        elif 'sonyliv.com' in base_url:
            return f"https://sonyliv.com/search?q={search_query.replace(' ', '%20')}"
        elif 'jiocinema.com' in base_url:
            return f"https://jiocinema.com/search/{search_query.replace(' ', '%20')}"
        elif 'mxplayer.in' in base_url:
            return f"https://mxplayer.in/search?q={search_query.replace(' ', '%20')}"
        elif 'aha.video' in base_url:
            return f"https://aha.video/search?q={search_query.replace(' ', '%20')}"
        elif 'youtube.com' in base_url:
            return f"https://youtube.com/results?search_query={search_query.replace(' ', '+')}"
        else:
            return base_url

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
            'append_to_response': 'credits,videos,similar,watch/providers,external_ids'
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
            # Add delay to respect Jikan rate limits
            time.sleep(0.5)
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Jikan search error: {e}")
        return None
    
    @staticmethod
    def get_anime_details(anime_id):
        """Get detailed anime information by ID - FIX FOR ANIME DETAILS BUG"""
        url = f"{JikanService.BASE_URL}/anime/{anime_id}/full"
        
        try:
            time.sleep(0.5)  # Rate limiting
            response = requests.get(url, timeout=15)
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
            time.sleep(0.5)
            response = requests.get(url, params=params, timeout=15)
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

# Enhanced Content Management Service with Streaming Integration
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            # Check if content already exists
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                # Update streaming info if content exists and is old
                ContentService.update_streaming_info(existing)
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
            
            # Get IMDB ID for streaming lookup
            imdb_id = None
            if 'external_ids' in tmdb_data:
                imdb_id = tmdb_data['external_ids'].get('imdb_id')
            
            # Get streaming availability
            streaming_data = ContentService.get_streaming_availability(
                tmdb_data.get('title') or tmdb_data.get('name'),
                imdb_id
            )
            
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
                ott_platforms=json.dumps(ContentService.extract_basic_platforms(streaming_data)),
                streaming_links=json.dumps(streaming_data)
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def get_streaming_availability(title, imdb_id=None):
        """Get comprehensive streaming availability"""
        try:
            streaming_data = None
            
            # Try RapidAPI first
            if imdb_id:
                streaming_data = StreamingAvailabilityService.get_streaming_by_imdb(imdb_id)
            
            if not streaming_data:
                streaming_data = StreamingAvailabilityService.search_streaming_by_title(title)
            
            # Try WatchMode as fallback
            if not streaming_data:
                streaming_data = StreamingAvailabilityService.search_watchmode(title)
            
            # Parse and return structured data
            return StreamingAvailabilityService.parse_streaming_data(streaming_data, title)
            
        except Exception as e:
            logger.error(f"Error getting streaming availability: {e}")
            return StreamingAvailabilityService.get_fallback_platforms(title)
    
    @staticmethod
    def update_streaming_info(content):
        """Update streaming information for existing content"""
        try:
            if content.updated_at and content.updated_at > datetime.utcnow() - timedelta(hours=24):
                return  # Skip if updated recently
            
            streaming_data = ContentService.get_streaming_availability(content.title, content.imdb_id)
            content.streaming_links = json.dumps(streaming_data)
            content.ott_platforms = json.dumps(ContentService.extract_basic_platforms(streaming_data))
            content.updated_at = datetime.utcnow()
            
            db.session.commit()
        except Exception as e:
            logger.error(f"Error updating streaming info: {e}")
    
    @staticmethod
    def extract_basic_platforms(streaming_data):
        """Extract basic platform list for backward compatibility"""
        platforms = []
        
        if streaming_data:
            for option in streaming_data.get('free_options', []):
                platforms.append({
                    'platform': option.get('platform'),
                    'name': option.get('platform_name'),
                    'url': option.get('url'),
                    'is_free': True
                })
            
            for option in streaming_data.get('paid_options', []):
                platforms.append({
                    'platform': option.get('platform'),
                    'name': option.get('platform_name'),
                    'url': option.get('url'),
                    'is_free': False
                })
        
        return platforms
    
    @staticmethod
    def save_anime_content(anime_data):
        """Save anime content with streaming info - FIX FOR ANIME BUG"""
        try:
            # Check if anime already exists
            existing = Content.query.filter_by(tmdb_id=anime_data['mal_id']).first()
            
            if existing and existing.content_type == 'anime':
                return existing
            
            # Get streaming availability for anime
            streaming_data = ContentService.get_anime_streaming(anime_data)
            
            # Create anime content
            content = Content(
                tmdb_id=anime_data['mal_id'],
                title=anime_data.get('title'),
                original_title=anime_data.get('title_japanese'),
                content_type='anime',
                genres=json.dumps([genre['name'] for genre in anime_data.get('genres', [])]),
                languages=json.dumps(['japanese']),
                release_date=datetime.strptime(anime_data.get('aired', {}).get('from', '1900-01-01T00:00:00+00:00')[:10], '%Y-%m-%d').date() if anime_data.get('aired', {}).get('from') else None,
                runtime=anime_data.get('duration', 24),  # Default episode duration
                rating=anime_data.get('score'),
                vote_count=anime_data.get('scored_by'),
                popularity=anime_data.get('popularity'),
                overview=anime_data.get('synopsis'),
                poster_path=anime_data.get('images', {}).get('jpg', {}).get('image_url'),
                backdrop_path=anime_data.get('images', {}).get('jpg', {}).get('large_image_url'),
                ott_platforms=json.dumps(ContentService.extract_basic_platforms(streaming_data)),
                streaming_links=json.dumps(streaming_data)
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def get_anime_streaming(anime_data):
        """Get streaming availability for anime"""
        anime_platforms = {
            'free_options': [
                {
                    'platform': 'crunchyroll_free',
                    'platform_name': 'Crunchyroll',
                    'url': f"https://crunchyroll.com/search?q={anime_data.get('title', '').replace(' ', '%20')}",
                    'is_free': True,
                    'quality': 'HD',
                    'availability': 'Available Now',
                    'icon': '🌸'
                },
                {
                    'platform': 'youtube',
                    'platform_name': 'YouTube',
                    'url': f"https://youtube.com/results?search_query={anime_data.get('title', '').replace(' ', '+')}+anime",
                    'is_free': True,
                    'quality': 'HD',
                    'availability': 'Available Now',
                    'icon': '▶️'
                }
            ],
            'paid_options': [],
            'languages': {
                'japanese': {
                    'language': 'Japanese',
                    'industry': 'Anime',
                    'links': [
                        {
                            'platform': 'Crunchyroll',
                            'url': f"https://crunchyroll.com/search?q={anime_data.get('title', '').replace(' ', '%20')}",
                            'is_free': True,
                            'icon': '🌸',
                            'quality': 'HD',
                            'button_text': '🔘 Watch in Japanese',
                            'priority': 1
                        }
                    ]
                }
            },
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return anime_platforms
    
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

# Enhanced Recommendation Engine with Regional Prioritization
class RecommendationEngine:
    @staticmethod
    def get_streaming_summary(streaming_info):
        """Get a summary of streaming availability"""
        if not streaming_info:
            return {'total_platforms': 0, 'free_count': 0, 'paid_count': 0, 'preview': []}
        
        free_count = len(streaming_info.get('free_options', []))
        paid_count = len(streaming_info.get('paid_options', []))
        
        # Get preview of top platforms
        preview = []
        all_options = streaming_info.get('free_options', []) + streaming_info.get('paid_options', [])
        for option in all_options[:3]:
            preview.append({
                'name': option.get('platform_name'),
                'icon': option.get('icon', '🎬'),
                'is_free': option.get('is_free', False)
            })
        
        return {
            'total_platforms': free_count + paid_count,
            'free_count': free_count,
            'paid_count': paid_count,
            'preview': preview
        }
    
    @staticmethod
    def format_watch_links(streaming_info):
        """Format watch links in the requested example format"""
        if not streaming_info:
            return []
        
        watch_links = []
        languages = streaming_info.get('languages', {})
        
        # Prioritize languages based on regional priority
        lang_priorities = [(lang, REGIONAL_LANGUAGES.get(lang, {}).get('priority', 999)) 
                          for lang in languages.keys()]
        lang_priorities.sort(key=lambda x: x[1])
        
        for lang, _ in lang_priorities:
            lang_data = languages[lang]
            for link in lang_data.get('links', []):
                watch_links.append({
                    'platform': link['platform'],
                    'language': lang.title(),
                    'url': link['url'],
                    'is_free': link['is_free'],
                    'button_text': link['button_text'],
                    'icon': link.get('icon', '🎬')
                })
        
        return watch_links
    
    @staticmethod
    def get_trending_recommendations(limit=20, content_type='all', prioritize_regional=True):
        try:
            # Get trending from TMDB
            trending_data = TMDBService.get_trending(content_type=content_type)
            if not trending_data:
                return []
            
            recommendations = []
            regional_content = []
            international_content = []
            
            for item in trending_data.get('results', []):
                content_type_detected = 'movie' if 'title' in item else 'tv'
                content = ContentService.save_content_from_tmdb(item, content_type_detected)
                if content:
                    # Check if it's regional content
                    if RecommendationEngine.is_regional_content(content):
                        regional_content.append(content)
                    else:
                        international_content.append(content)
            
            # Prioritize regional content if requested
            if prioritize_regional:
                # Sort regional content by language priority
                regional_content.sort(key=lambda x: RecommendationEngine.get_language_priority(x))
                recommendations = regional_content + international_content
            else:
                recommendations = international_content + regional_content
            
            return recommendations[:limit]
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    @staticmethod
    def get_homepage_recommendations():
        """Get categorized recommendations for homepage with regional priority"""
        try:
            homepage_data = {
                '🔥 Trending': {
                    'content': [],
                    'description': 'What\'s hot right now'
                },
                '🕰️ All-time Hits': {
                    'content': [],
                    'description': 'Timeless classics that never get old'
                },
                '🆕 New Releases': {
                    'content': [],
                    'description': 'Fresh content just for you'
                },
                '🎭 Action': {
                    'content': [],
                    'description': 'High-octane entertainment'
                },
                '💕 Romance': {
                    'content': [],
                    'description': 'Love stories that touch the heart'
                },
                '😂 Comedy': {
                    'content': [],
                    'description': 'Laugh out loud moments'
                },
                '🎨 Regional Gems': {
                    'content': [],
                    'description': 'Best of regional cinema'
                }
            }
            
            # Get trending with regional priority
            trending = RecommendationEngine.get_trending_recommendations(15, prioritize_regional=True)
            homepage_data['🔥 Trending']['content'] = trending
            
            # Get genre-specific recommendations
            action_content = RecommendationEngine.get_popular_by_genre('Action', 10)
            homepage_data['🎭 Action']['content'] = action_content
            
            romance_content = RecommendationEngine.get_popular_by_genre('Romance', 10)
            homepage_data['💕 Romance']['content'] = romance_content
            
            comedy_content = RecommendationEngine.get_popular_by_genre('Comedy', 10)
            homepage_data['😂 Comedy']['content'] = comedy_content
            
            # Get regional recommendations prioritized by Telugu > Hindi > Tamil > Malayalam > Kannada
            regional_recs = []
            for lang in ['telugu', 'hindi', 'tamil', 'malayalam', 'kannada']:
                lang_content = RecommendationEngine.get_regional_recommendations(lang, 3)
                regional_recs.extend(lang_content)
            
            homepage_data['🎨 Regional Gems']['content'] = regional_recs[:15]
            
            # Get new releases (content from last 6 months)
            recent_date = datetime.utcnow() - timedelta(days=180)
            new_releases = Content.query.filter(
                Content.release_date >= recent_date.date()
            ).order_by(Content.popularity.desc()).limit(15).all()
            
            homepage_data['🆕 New Releases']['content'] = new_releases
            
            # Get all-time hits (high-rated content)
            all_time_hits = Content.query.filter(
                Content.rating >= 7.5,
                Content.vote_count >= 1000
            ).order_by(Content.rating.desc()).limit(15).all()
            
            homepage_data['🕰️ All-time Hits']['content'] = all_time_hits
            
            return homepage_data
            
        except Exception as e:
            logger.error(f"Error getting homepage recommendations: {e}")
            return {}
    
    @staticmethod
    def is_regional_content(content):
        """Check if content is regional (Indian languages)"""
        try:
            if content.languages:
                languages = json.loads(content.languages)
                regional_codes = []
                for lang_info in REGIONAL_LANGUAGES.values():
                    regional_codes.extend(lang_info['codes'])
                
                return any(lang.lower() in regional_codes for lang in languages)
        except:
            pass
        return False
    
    @staticmethod
    def get_language_priority(content):
        """Get priority score for content based on language (lower = higher priority)"""
        try:
            if content.languages:
                languages = json.loads(content.languages)
                min_priority = 999
                
                for lang in languages:
                    for lang_code, lang_info in REGIONAL_LANGUAGES.items():
                        if lang.lower() in lang_info['codes']:
                            min_priority = min(min_priority, lang_info['priority'])
                
                return min_priority
        except:
            pass
        return 999
    
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
                'hindi': ['bollywood', 'hindi movie', 'hindi film', 'hindi cinema'],
                'telugu': ['tollywood', 'telugu movie', 'telugu film', 'telugu cinema'],
                'tamil': ['kollywood', 'tamil movie', 'tamil film', 'tamil cinema'],
                'kannada': ['sandalwood', 'kannada movie', 'kannada film', 'kannada cinema'],
                'malayalam': ['mollywood', 'malayalam movie', 'malayalam film', 'malayalam cinema']
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
                # Save anime using the fixed method
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
            
            # Get streaming info
            streaming_info = json.loads(content.streaming_links or '{}')
            platforms_text = ""
            if streaming_info.get('free_options') or streaming_info.get('paid_options'):
                free_count = len(streaming_info.get('free_options', []))
                paid_count = len(streaming_info.get('paid_options', []))
                platforms_text = f"\n📺 **Available on:** {free_count} free, {paid_count} paid platforms"
            
            # Create message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}{platforms_text}

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

# Enhanced Content Discovery Routes
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
                    
                    # Get streaming info
                    streaming_info = json.loads(content.streaming_links or '{}')
                    
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
                        'streaming_summary': RecommendationEngine.get_streaming_summary(streaming_info)
                    })
        
        # Add anime results with proper handling
        if anime_results:
            for anime in anime_results.get('data', []):
                content = ContentService.save_anime_content(anime)
                if content:
                    streaming_info = json.loads(content.streaming_links or '{}')
                    
                    results.append({
                        'id': content.id,
                        'title': content.title,
                        'content_type': 'anime',
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': content.poster_path,
                        'overview': content.overview,
                        'streaming_summary': RecommendationEngine.get_streaming_summary(streaming_info)
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

# Enhanced content details route with comprehensive streaming info
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
        
        # Update streaming info if it's old
        ContentService.update_streaming_info(content)
        
        # Get additional details from TMDB if available
        additional_details = None
        if content.tmdb_id and content.content_type != 'anime':
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        elif content.content_type == 'anime':
            # Fixed anime details retrieval
            additional_details = JikanService.get_anime_details(content.tmdb_id)
        
        # Get YouTube trailers
        trailers = []
        if YOUTUBE_API_KEY:
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
        
        # Get comprehensive streaming information
        streaming_info = json.loads(content.streaming_links or '{}')
        
        db.session.commit()
        
        response_data = {
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
            'trailers': trailers,
            'similar_content': similar_content,
            
            # Enhanced streaming information
            'streaming_availability': {
                'free_options': streaming_info.get('free_options', []),
                'paid_options': streaming_info.get('paid_options', []),
                'language_links': streaming_info.get('languages', {}),
                'last_updated': streaming_info.get('last_updated'),
                'total_platforms': len(streaming_info.get('free_options', [])) + len(streaming_info.get('paid_options', []))
            },
            
            # Example format as requested
            'where_to_watch': RecommendationEngine.format_watch_links(streaming_info),
            
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details and 'credits' in additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details and 'credits' in additional_details else []
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Enhanced homepage endpoint with regional prioritization
@app.route('/api/homepage', methods=['GET'])
def get_homepage():
    try:
        homepage_data = RecommendationEngine.get_homepage_recommendations()
        
        # Format response with streaming info
        formatted_data = {}
        for category, data in homepage_data.items():
            formatted_data[category] = {
                'description': data['description'],
                'content': []
            }
            
            for content in data['content']:
                streaming_info = json.loads(content.streaming_links or '{}')
                
                formatted_data[category]['content'].append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'streaming_summary': RecommendationEngine.get_streaming_summary(streaming_info),
                    'available_languages': list(streaming_info.get('languages', {}).keys())
                })
        
        return jsonify(formatted_data), 200
        
    except Exception as e:
        logger.error(f"Homepage error: {e}")
        return jsonify({'error': 'Failed to get homepage data'}), 500

# Anime details fix route
@app.route('/api/anime/<int:anime_id>', methods=['GET'])
def get_anime_details(anime_id):
    try:
        # Get anime details from Jikan API
        anime_data = JikanService.get_anime_details(anime_id)
        
        if not anime_data:
            return jsonify({'error': 'Anime not found'}), 404
        
        anime_info = anime_data.get('data', {})
        
        # Save or update anime content
        content = ContentService.save_anime_content(anime_info)
        
        if content:
            # Record view interaction
            session_id = get_session_id()
            interaction = AnonymousInteraction(
                session_id=session_id,
                content_id=content.id,
                interaction_type='view',
                ip_address=request.remote_addr
            )
            db.session.add(interaction)
            db.session.commit()
            
            # Get streaming info
            streaming_info = json.loads(content.streaming_links or '{}')
            
            return jsonify({
                'id': content.id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': 'anime',
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'overview': content.overview,
                'poster_path': content.poster_path,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'streaming_availability': {
                    'free_options': streaming_info.get('free_options', []),
                    'paid_options': streaming_info.get('paid_options', []),
                    'language_links': streaming_info.get('languages', {}),
                    'last_updated': streaming_info.get('last_updated')
                },
                'where_to_watch': RecommendationEngine.format_watch_links(streaming_info),
                'episodes': anime_info.get('episodes'),
                'status': anime_info.get('status'),
                'studios': [studio['name'] for studio in anime_info.get('studios', [])],
                'themes': [theme['name'] for theme in anime_info.get('themes', [])]
            }), 200
        else:
            return jsonify({'error': 'Failed to process anime data'}), 500
        
    except Exception as e:
        logger.error(f"Anime details error: {e}")
        return jsonify({'error': 'Failed to get anime details'}), 500

# Recommendation Routes
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = []
        for content in recommendations:
            streaming_info = json.loads(content.streaming_links or '{}')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_summary': RecommendationEngine.get_streaming_summary(streaming_info),
                'where_to_watch': RecommendationEngine.format_watch_links(streaming_info)
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
            streaming_info = json.loads(content.streaming_links or '{}')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_summary': RecommendationEngine.get_streaming_summary(streaming_info),
                'where_to_watch': RecommendationEngine.format_watch_links(streaming_info)
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
            streaming_info = json.loads(content.streaming_links or '{}')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_summary': RecommendationEngine.get_streaming_summary(streaming_info),
                'where_to_watch': RecommendationEngine.format_watch_links(streaming_info)
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
            streaming_info = json.loads(content.streaming_links or '{}')
            result.append({
                'id': content.id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': content.poster_path,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_summary': RecommendationEngine.get_streaming_summary(streaming_info),
                'where_to_watch': RecommendationEngine.format_watch_links(streaming_info)
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
            streaming_info = json.loads(content.streaming_links or '{}')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'streaming_summary': RecommendationEngine.get_streaming_summary(streaming_info),
                'where_to_watch': RecommendationEngine.format_watch_links(streaming_info)
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
                        streaming_info = json.loads(content.streaming_links or '{}')
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'streaming_summary': RecommendationEngine.get_streaming_summary(streaming_info),
                            'where_to_watch': RecommendationEngine.format_watch_links(streaming_info),
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
            streaming_info = json.loads(content.streaming_links or '{}')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'streaming_summary': RecommendationEngine.get_streaming_summary(streaming_info),
                'where_to_watch': RecommendationEngine.format_watch_links(streaming_info)
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
            streaming_info = json.loads(content.streaming_links or '{}')
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'streaming_summary': RecommendationEngine.get_streaming_summary(streaming_info),
                'where_to_watch': RecommendationEngine.format_watch_links(streaming_info)
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
            
            # Get streaming availability
            streaming_data = ContentService.get_streaming_availability(data.get('title', ''))
            
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
                ott_platforms=json.dumps(ContentService.extract_basic_platforms(streaming_data)),
                streaming_links=json.dumps(streaming_data)
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
                streaming_info = json.loads(content.streaming_links or '{}')
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'streaming_summary': RecommendationEngine.get_streaming_summary(streaming_info),
                    'where_to_watch': RecommendationEngine.format_watch_links(streaming_info),
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