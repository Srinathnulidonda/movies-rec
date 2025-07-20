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
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import re

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Database configuration
if os.environ.get('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_recommendations.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# API Keys
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY', '52260795')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyDU-JLASTdIdoLOmlpWuJYLTZDUspqw2T4')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '-1002850793757')
WATCHMODE_API_KEY = os.environ.get('WATCHMODE_API_KEY', 'WtcKDji9i20pjOl5Lg0AiyG2bddfUs3nSZRZJIsY')

# Streaming Availability API credentials
RAPID_API_KEY = "c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37"
RAPID_API_HOST = "streaming-availability.p.rapidapi.com"

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
    preferred_languages = db.Column(db.Text)
    preferred_genres = db.Column(db.Text)
    location = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    imdb_id = db.Column(db.String(20))
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    content_type = db.Column(db.String(20), nullable=False)
    genres = db.Column(db.Text)
    languages = db.Column(db.Text)
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float)
    vote_count = db.Column(db.Integer)
    popularity = db.Column(db.Float)
    overview = db.Column(db.Text)
    poster_path = db.Column(db.String(255))
    backdrop_path = db.Column(db.String(255))
    trailer_url = db.Column(db.String(255))
    ott_platforms = db.Column(db.Text)
    youtube_availability = db.Column(db.Text)
    streaming_languages = db.Column(db.Text)  # New field for language-specific streaming
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AdminRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recommendation_type = db.Column(db.String(50))
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

# Cache for OTT data
class OTTCache:
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=6)
    
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < self.cache_duration:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, data):
        self.cache[key] = (data, datetime.utcnow())

ott_cache = OTTCache()

# Language mapping for Indian content
LANGUAGE_MAPPING = {
    'hi': 'Hindi',
    'te': 'Telugu', 
    'ta': 'Tamil',
    'ml': 'Malayalam',
    'kn': 'Kannada',
    'en': 'English',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'mr': 'Marathi',
    'pa': 'Punjabi'
}

# Priority order for recommendations
LANGUAGE_PRIORITY = ['te', 'en', 'hi', 'ta', 'ml', 'kn']

# Genre mapping
GENRE_MAPPING = {
    'Action': [28], 'Adventure': [12], 'Animation': [16], 'Biography': [36],
    'Comedy': [35], 'Crime': [80], 'Documentary': [99], 'Drama': [18],
    'Fantasy': [14], 'Horror': [27], 'Musical': [10402], 'Mystery': [9648],
    'Romance': [10749], 'Sci-Fi': [878], 'Thriller': [53], 'Western': [37]
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

# Enhanced YouTube Service
class EnhancedYouTubeService:
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    @staticmethod
    async def get_comprehensive_youtube_availability(title, original_title=None, release_year=None, content_type='movie', region='IN'):
        """Get comprehensive YouTube availability including free movies, trailers, and premium content"""
        if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == 'your_youtube_api_key':
            return {}
        
        try:
            youtube_data = {
                'free_movies': [],
                'premium_content': [],
                'trailers': [],
                'clips': [],
                'official_content': [],
                'last_checked': datetime.utcnow().isoformat()
            }
            
            search_queries = EnhancedYouTubeService._generate_search_queries(title, original_title, release_year, content_type, region)
            
            async with aiohttp.ClientSession() as session:
                for query in search_queries[:8]:  # Limit queries
                    try:
                        url = f"{EnhancedYouTubeService.BASE_URL}/search"
                        params = {
                            'key': YOUTUBE_API_KEY,
                            'q': query,
                            'part': 'snippet',
                            'type': 'video',
                            'maxResults': 5,
                            'order': 'relevance',
                            'regionCode': region,
                            'safeSearch': 'moderate'
                        }
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                for item in data.get('items', []):
                                    video_data = await EnhancedYouTubeService._get_detailed_video_info(session, item)
                                    if video_data:
                                        category = EnhancedYouTubeService._categorize_video(video_data, title)
                                        if category:
                                            youtube_data[category].append(video_data)
                            
                            await asyncio.sleep(0.1)
                            
                    except Exception as e:
                        logger.error(f"YouTube search error for query '{query}': {e}")
                        continue
            
            return youtube_data
            
        except Exception as e:
            logger.error(f"YouTube comprehensive search error: {e}")
            return {}
    
    @staticmethod
    def _generate_search_queries(title, original_title, release_year, content_type, region):
        """Generate comprehensive search queries"""
        queries = []
        year_str = str(release_year) if release_year else ""
        
        # Basic searches
        queries.extend([
            f"{title} full movie",
            f"{title} movie {year_str}",
            f"{title} complete movie",
            f"{title} HD full movie",
            f"{title} free movie"
        ])
        
        # Original title searches
        if original_title and original_title != title:
            queries.extend([
                f"{original_title} full movie",
                f"{original_title} movie {year_str}"
            ])
        
        # Language-specific searches for Indian region
        if region == 'IN':
            queries.extend([
                f"{title} hindi movie",
                f"{title} tamil movie",
                f"{title} telugu movie",
                f"{title} full movie with subtitles"
            ])
        
        return queries
    
    @staticmethod
    async def _get_detailed_video_info(session, video_item):
        """Get detailed information about a video"""
        try:
            video_id = video_item['id']['videoId']
            
            url = f"{EnhancedYouTubeService.BASE_URL}/videos"
            params = {
                'key': YOUTUBE_API_KEY,
                'id': video_id,
                'part': 'snippet,contentDetails,statistics',
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('items'):
                        video_details = data['items'][0]
                        duration = EnhancedYouTubeService._parse_duration(
                            video_details['contentDetails'].get('duration', 'PT0S')
                        )
                        
                        return {
                            'video_id': video_id,
                            'title': video_details['snippet']['title'],
                            'description': video_details['snippet']['description'],
                            'channel_title': video_details['snippet']['channelTitle'],
                            'published_at': video_details['snippet']['publishedAt'],
                            'duration_seconds': duration,
                            'duration_formatted': EnhancedYouTubeService._format_duration(duration),
                            'view_count': int(video_details['statistics'].get('viewCount', 0)),
                            'like_count': int(video_details['statistics'].get('likeCount', 0)),
                            'thumbnail_url': video_details['snippet']['thumbnails'].get('high', {}).get('url'),
                            'watch_url': f"https://youtube.com/watch?v={video_id}",
                            'embed_url': f"https://youtube.com/embed/{video_id}",
                            'quality_score': EnhancedYouTubeService._calculate_quality_score(video_details, duration)
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Video details error: {e}")
            return None
    
    @staticmethod
    def _parse_duration(duration_str):
        """Parse YouTube duration format (PT1H2M3S) to seconds"""
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration_str)
        
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds
        
        return 0
    
    @staticmethod
    def _format_duration(seconds):
        """Format seconds to readable duration"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    @staticmethod
    def _categorize_video(video, movie_title):
        """Categorize video based on content analysis"""
        title = video['title'].lower()
        duration = video['duration_seconds']
        
        # Check for full movies (typically longer than 60 minutes)
        if duration > 3600:  # More than 1 hour
            if any(keyword in title for keyword in ['full movie', 'complete movie', 'full film', 'movie']):
                return 'free_movies'
        
        # Check for trailers (typically 1-5 minutes)
        elif 60 <= duration <= 300:
            if any(keyword in title for keyword in ['trailer', 'teaser', 'preview', 'promo']):
                return 'trailers'
        
        # Check for clips and songs (typically 3-15 minutes)
        elif 180 <= duration <= 900:
            if any(keyword in title for keyword in ['song', 'clip', 'scene', 'dialogue']):
                return 'clips'
        
        return None
    
    @staticmethod
    def _calculate_quality_score(video_details, duration):
        """Calculate quality score based on various factors"""
        score = 0.0
        
        # Duration score
        if duration > 3600:  # Full movie length
            score += 0.4
        elif duration > 1800:  # Half movie length
            score += 0.2
        
        # View count score
        view_count = int(video_details['statistics'].get('viewCount', 0))
        if view_count > 1000000:  # 1M+ views
            score += 0.3
        elif view_count > 100000:  # 100K+ views
            score += 0.2
        
        # Like count score
        like_count = int(video_details['statistics'].get('likeCount', 0))
        if like_count > 1000:
            score += 0.1
        
        # Channel verification
        channel_title = video_details['snippet']['channelTitle'].lower()
        verified_indicators = ['official', 'music', 'entertainment', 'movies']
        if any(indicator in channel_title for indicator in verified_indicators):
            score += 0.2
        
        return min(score, 1.0)

# Enhanced OTT Availability Service with Streaming Availability API
class OTTAvailabilityService:
    STREAMING_API_BASE = 'https://streaming-availability.p.rapidapi.com'
    
    @staticmethod
    async def get_comprehensive_availability(tmdb_id, content_type='movie', region='in'):
        """Get availability from Streaming Availability API with language-specific options"""
        cache_key = f"streaming_{tmdb_id}_{content_type}_{region}"
        
        # Check cache first
        cached_result = ott_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        availability_data = {
            'platforms': [],
            'language_specific': {},
            'free_options': [],
            'paid_options': [],
            'rent_options': [],
            'buy_options': [],
            'youtube_data': {},
            'last_updated': datetime.utcnow().isoformat(),
            'region': region
        }
        
        try:
            # Get title information first
            title_data = await OTTAvailabilityService.get_tmdb_title(tmdb_id, content_type)
            if not title_data:
                return availability_data
            
            availability_data['title'] = title_data['title']
            
            # Get streaming availability from RapidAPI
            streaming_data = await OTTAvailabilityService.get_streaming_availability(tmdb_id, content_type, region)
            if streaming_data:
                availability_data['platforms'].extend(streaming_data['platforms'])
                availability_data['language_specific'] = streaming_data['language_specific']
            
            # Get comprehensive YouTube availability
            youtube_data = await EnhancedYouTubeService.get_comprehensive_youtube_availability(
                title_data['title'],
                title_data.get('original_title'),
                title_data.get('release_year'),
                content_type,
                region
            )
            availability_data['youtube_data'] = youtube_data
            
            # Convert YouTube data to platform format
            youtube_platforms = OTTAvailabilityService._convert_youtube_to_platforms(youtube_data)
            availability_data['platforms'].extend(youtube_platforms)
            
            # Categorize platforms
            for platform in availability_data['platforms']:
                if platform['is_free']:
                    availability_data['free_options'].append(platform)
                elif platform['availability_type'] == 'subscription':
                    availability_data['paid_options'].append(platform)
                elif platform['availability_type'] == 'rent':
                    availability_data['rent_options'].append(platform)
                elif platform['availability_type'] == 'buy':
                    availability_data['buy_options'].append(platform)
            
            # Cache the result
            ott_cache.set(cache_key, availability_data)
            
            return availability_data
            
        except Exception as e:
            logger.error(f"OTT availability error: {e}")
            return availability_data
    
    @staticmethod
    async def get_streaming_availability(tmdb_id, content_type, region):
        """Get streaming availability from RapidAPI Streaming Availability"""
        try:
            headers = {
                'x-rapidapi-key': RAPID_API_KEY,
                'x-rapidapi-host': RAPID_API_HOST
            }
            
            # Convert content type for API
            api_content_type = 'movie' if content_type == 'movie' else 'series'
            
            platforms = []
            language_specific = {}
            
            async with aiohttp.ClientSession() as session:
                # Get detailed streaming info using TMDB ID
                detail_url = f"{OTTAvailabilityService.STREAMING_API_BASE}/get/basic"
                detail_params = {
                    'country': region,
                    'tmdb_id': tmdb_id,
                    'output_language': 'en'
                }
                
                async with session.get(detail_url, headers=headers, params=detail_params) as detail_response:
                    if detail_response.status == 200:
                        detail_data = await detail_response.json()
                        
                        if detail_data.get('result'):
                            streaming_info = detail_data['result']
                            
                            # Process streaming options
                            processed_data = await OTTAvailabilityService._process_streaming_data(
                                streaming_info, region
                            )
                            
                            platforms.extend(processed_data['platforms'])
                            language_specific = processed_data['language_specific']
                    
                    await asyncio.sleep(0.2)  # Rate limiting
            
            return {
                'platforms': platforms,
                'language_specific': language_specific
            }
            
        except Exception as e:
            logger.error(f"Streaming availability API error: {e}")
            return {'platforms': [], 'language_specific': {}}
    
    @staticmethod
    async def _process_streaming_data(streaming_info, region):
        """Process streaming data and organize by language"""
        platforms = []
        language_specific = {}
        
        try:
            streaming_options = streaming_info.get('streamingOptions', {})
            
            for country_code, options in streaming_options.items():
                if country_code.lower() != region.lower():
                    continue
                
                for option in options:
                    service = option.get('service', {})
                    
                    # Extract basic platform info
                    platform_data = {
                        'platform_id': service.get('id', '').lower(),
                        'platform_name': service.get('name', 'Unknown'),
                        'logo_url': service.get('imageSet', {}).get('lightThemeImage', ''),
                        'availability_type': OTTAvailabilityService._map_streaming_type(option.get('type')),
                        'is_free': option.get('type') == 'free',
                        'price': option.get('price', {}).get('amount'),
                        'currency': option.get('price', {}).get('currency'),
                        'source': 'streaming_availability_api',
                        'quality': option.get('quality', 'HD'),
                        'watch_url': option.get('link')
                    }
                    
                    # Extract audio languages for this platform
                    audio_languages = option.get('audios', [])
                    subtitle_languages = option.get('subtitles', [])
                    
                    # If multiple audio languages available, create separate entries
                    if audio_languages and len(audio_languages) > 1:
                        for audio_lang in audio_languages:
                            lang_code = audio_lang.get('language')
                            lang_name = LANGUAGE_MAPPING.get(lang_code, lang_code.upper())
                            
                            lang_platform = platform_data.copy()
                            lang_platform['audio_language'] = lang_name
                            lang_platform['audio_language_code'] = lang_code
                            lang_platform['available_subtitles'] = [
                                LANGUAGE_MAPPING.get(sub.get('language'), sub.get('language', ''))
                                for sub in subtitle_languages
                            ]
                            
                            # Create language-specific watch URL
                            lang_platform['watch_url'] = OTTAvailabilityService._create_language_specific_url(
                                platform_data['watch_url'], 
                                platform_data['platform_id'], 
                                lang_code
                            )
                            
                            platforms.append(lang_platform)
                            
                            # Add to language-specific grouping
                            if lang_name not in language_specific:
                                language_specific[lang_name] = []
                            language_specific[lang_name].append(lang_platform)
                    else:
                        # Single language or no language info
                        if audio_languages:
                            lang_code = audio_languages[0].get('language')
                            platform_data['audio_language'] = LANGUAGE_MAPPING.get(lang_code, lang_code.upper())
                            platform_data['audio_language_code'] = lang_code
                        
                        platform_data['available_subtitles'] = [
                            LANGUAGE_MAPPING.get(sub.get('language'), sub.get('language', ''))
                            for sub in subtitle_languages
                        ]
                        
                        platforms.append(platform_data)
            
            return {
                'platforms': platforms,
                'language_specific': language_specific
            }
            
        except Exception as e:
            logger.error(f"Error processing streaming data: {e}")
            return {'platforms': [], 'language_specific': {}}
    
    @staticmethod
    def _create_language_specific_url(base_url, platform_id, language_code):
        """Create language-specific streaming URLs"""
        if not base_url:
            return base_url
        
        # Platform-specific URL modifications for language selection
        language_params = {
            'netflix': f"{base_url}?audio={language_code}",
            'amazon_prime': f"{base_url}&language={language_code}",
            'disney_plus': f"{base_url}?lang={language_code}",
            'hotstar': f"{base_url}?audio={language_code}",
            'zee5': f"{base_url}?lang={language_code}",
            'sonyliv': f"{base_url}?language={language_code}",
            'jiocinema': f"{base_url}?lang={language_code}",
            'voot': f"{base_url}?audio={language_code}"
        }
        
        return language_params.get(platform_id, base_url)
    
    @staticmethod
    def _map_streaming_type(stream_type):
        """Map streaming API types to our format"""
        type_mapping = {
            'subscription': 'subscription',
            'free': 'free',
            'rent': 'rent', 
            'buy': 'buy',
            'addon': 'subscription'
        }
        return type_mapping.get(stream_type, 'subscription')
    
    @staticmethod
    def _convert_youtube_to_platforms(youtube_data):
        """Convert YouTube data to platform format"""
        platforms = []
        
        # Free movies
        for movie in youtube_data.get('free_movies', []):
            platforms.append({
                'platform_id': 'youtube',
                'platform_name': 'YouTube',
                'logo_url': 'https://youtube.com/favicon.ico',
                'watch_url': movie['watch_url'],
                'availability_type': 'free',
                'is_free': True,
                'source': 'youtube',
                'video_title': movie['title'],
                'duration': movie['duration_formatted'],
                'quality_score': movie['quality_score'],
                'view_count': movie['view_count'],
                'channel_name': movie['channel_title'],
                'content_type': 'full_movie'
            })
        
        # Add best trailer if available
        trailers = youtube_data.get('trailers', [])
        if trailers:
            best_trailer = max(trailers, key=lambda x: x['quality_score'])
            platforms.append({
                'platform_id': 'youtube_trailer',
                'platform_name': 'YouTube (Trailer)',
                'logo_url': 'https://youtube.com/favicon.ico',
                'watch_url': best_trailer['watch_url'],
                'availability_type': 'free',
                'is_free': True,
                'source': 'youtube',
                'video_title': best_trailer['title'],
                'duration': best_trailer['duration_formatted'],
                'content_type': 'trailer'
            })
        
        return platforms
    
    @staticmethod
    async def get_tmdb_title(tmdb_id, content_type):
        """Get title from TMDB"""
        try:
            url = f"https://api.themoviedb.org/3/{content_type}/{tmdb_id}"
            params = {'api_key': TMDB_API_KEY}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        release_date = data.get('release_date') or data.get('first_air_date')
                        release_year = None
                        if release_date:
                            try:
                                release_year = int(release_date.split('-')[0])
                            except:
                                pass
                        
                        return {
                            'title': data.get('title') or data.get('name'),
                            'original_title': data.get('original_title') or data.get('original_name'),
                            'release_date': release_date,
                            'release_year': release_year
                        }
            return None
            
        except Exception as e:
            logger.error(f"TMDB title fetch error: {e}")
            return None

# Enhanced Anime Service
class EnhancedAnimeService:
    """Enhanced anime service with better error handling"""
    
    @staticmethod
    async def get_anime_details(anime_id):
        """Get comprehensive anime details with multiple fallbacks"""
        try:
            return await EnhancedAnimeService._get_jikan_details(anime_id)
        except Exception as e:
            logger.error(f"Anime details error for ID {anime_id}: {e}")
            return None
    
    @staticmethod
    async def _get_jikan_details(anime_id):
        """Get anime details from Jikan API with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"https://api.jikan.moe/v4/anime/{anime_id}/full"
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            anime_data = data.get('data')
                            
                            if anime_data:
                                processed_data = EnhancedAnimeService._process_anime_data(anime_data)
                                streaming_data = await EnhancedAnimeService._get_anime_streaming(anime_data)
                                processed_data['streaming_options'] = streaming_data
                                return processed_data
                        elif response.status == 429:  # Rate limited
                            await asyncio.sleep(retry_delay * (attempt + 1))
                            continue
                        else:
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
                                continue
                            
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for anime {anime_id}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                
        return None
    
    @staticmethod
    def _process_anime_data(anime_data):
        """Process and normalize anime data"""
        try:
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            studios = [studio['name'] for studio in anime_data.get('studios', [])]
            themes = [theme['name'] for theme in anime_data.get('themes', [])]
            
            aired = anime_data.get('aired', {})
            
            return {
                'id': anime_data.get('mal_id'),
                'title': anime_data.get('title'),
                'title_english': anime_data.get('title_english'),
                'title_japanese': anime_data.get('title_japanese'),
                'type': anime_data.get('type'),
                'source': anime_data.get('source'),
                'episodes': anime_data.get('episodes'),
                'status': anime_data.get('status'),
                'aired_from': aired.get('from'),
                'aired_to': aired.get('to'),
                'duration': anime_data.get('duration'),
                'rating': anime_data.get('rating'),
                'score': anime_data.get('score'),
                'scored_by': anime_data.get('scored_by'),
                'rank': anime_data.get('rank'),
                'popularity': anime_data.get('popularity'),
                'synopsis': anime_data.get('synopsis'),
                'genres': genres,
                'studios': studios,
                'themes': themes,
                'images': anime_data.get('images', {}),
                'trailer': anime_data.get('trailer'),
                'year': anime_data.get('year'),
                'season': anime_data.get('season')
            }
            
        except Exception as e:
            logger.error(f"Error processing anime data: {e}")
            return {}
    
    @staticmethod
    async def _get_anime_streaming(anime_data):
        """Get anime streaming availability"""
        try:
            streaming_options = []
            
            # Check external links for streaming platforms
            external_links = anime_data.get('external', [])
            
            for link in external_links:
                platform_name = link.get('name', '').lower()
                url = link.get('url', '')
                
                if 'crunchyroll' in platform_name:
                    streaming_options.append({
                        'platform': 'Crunchyroll',
                        'url': url,
                        'type': 'subscription',
                        'region': 'Global',
                        'languages': ['Japanese', 'English']
                    })
                elif 'netflix' in platform_name:
                    streaming_options.append({
                        'platform': 'Netflix',
                        'url': url,
                        'type': 'subscription',
                        'region': 'Global',
                        'languages': ['Japanese', 'English', 'Hindi']
                    })
            
            # Add YouTube search for anime
            youtube_data = await EnhancedYouTubeService.get_comprehensive_youtube_availability(
                anime_data.get('title', ''),
                anime_data.get('title_english'),
                anime_data.get('year'),
                'anime',
                'IN'
            )
            
            # Convert YouTube results to streaming format
            for movie in youtube_data.get('free_movies', []):
                streaming_options.append({
                    'platform': 'YouTube',
                    'url': movie['watch_url'],
                    'type': 'free',
                    'region': 'Global',
                    'title': movie['title'],
                    'duration': movie['duration_formatted'],
                    'languages': ['Japanese', 'English', 'Hindi']
                })
            
            return streaming_options
            
        except Exception as e:
            logger.error(f"Error getting anime streaming: {e}")
            return []

# Enhanced Recommendation Service
class EnhancedRecommendationService:
    """Enhanced recommendation service with language priorities and real-time data"""
    
    @staticmethod
    async def get_comprehensive_recommendations(region='IN', language_priority=None, page=1):
        """Get comprehensive recommendations with language priorities"""
        if not language_priority:
            language_priority = LANGUAGE_PRIORITY
        
        recommendations = {
            'trending_movies': [],
            'new_releases': [],
            'best_movies': [],
            'critics_choice': [],
            'genre_picks': {},
            'language_specific': {},
            'free_to_watch': [],
            'last_updated': datetime.utcnow().isoformat()
        }
        
        try:
            # Get trending content
            trending = await EnhancedRecommendationService._get_trending_with_languages(language_priority, region)
            recommendations['trending_movies'] = trending
            
            # Get new releases by language
            for lang in language_priority[:3]:  # Top 3 languages
                new_releases = await EnhancedRecommendationService._get_new_releases_by_language(lang, region)
                recommendations['language_specific'][LANGUAGE_MAPPING.get(lang, lang)] = new_releases
            
            # Get best rated movies
            best_movies = await EnhancedRecommendationService._get_best_rated_movies(language_priority, region)
            recommendations['best_movies'] = best_movies
            
            # Get critics' choice
            critics_choice = await EnhancedRecommendationService._get_critics_choice(language_priority, region)
            recommendations['critics_choice'] = critics_choice
            
            # Get genre-wise recommendations
            for genre, genre_ids in list(GENRE_MAPPING.items())[:8]:  # Limit to 8 genres
                genre_content = await EnhancedRecommendationService._get_genre_recommendations(
                    genre_ids[0], language_priority, region
                )
                recommendations['genre_picks'][genre] = genre_content
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Comprehensive recommendations error: {e}")
            return recommendations
    
    @staticmethod
    async def _get_trending_with_languages(language_priority, region):
        """Get trending content prioritized by language"""
        trending_content = []
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.themoviedb.org/3/trending/movie/day"
                params = {
                    'api_key': TMDB_API_KEY,
                    'region': region
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for movie in data.get('results', [])[:15]:
                            detail_url = f"https://api.themoviedb.org/3/movie/{movie['id']}"
                            detail_params = {'api_key': TMDB_API_KEY}
                            
                            async with session.get(detail_url, params=detail_params) as detail_response:
                                if detail_response.status == 200:
                                    detail_data = await detail_response.json()
                                    
                                    original_language = detail_data.get('original_language', '')
                                    spoken_languages = [lang.get('iso_639_1') for lang in detail_data.get('spoken_languages', [])]
                                    
                                    language_priority_score = EnhancedRecommendationService._calculate_language_priority(
                                        original_language, spoken_languages, language_priority
                                    )
                                    
                                    if language_priority_score > 0:
                                        streaming_data = await OTTAvailabilityService.get_comprehensive_availability(
                                            movie['id'], 'movie', region.lower()
                                        )
                                        
                                        movie_data = {
                                            'id': movie['id'],
                                            'title': movie['title'],
                                            'original_title': detail_data.get('original_title'),
                                            'overview': movie['overview'][:200] + '...' if len(movie.get('overview', '')) > 200 else movie.get('overview', ''),
                                            'poster_path': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None,
                                            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{movie['backdrop_path']}" if movie.get('backdrop_path') else None,
                                            'release_date': movie.get('release_date'),
                                            'vote_average': movie.get('vote_average'),
                                            'vote_count': movie.get('vote_count'),
                                            'popularity': movie.get('popularity'),
                                            'original_language': original_language,
                                            'language_priority_score': language_priority_score,
                                            'streaming_availability': streaming_data,
                                            'genres': [genre['name'] for genre in detail_data.get('genres', [])],
                                            'is_free_available': len(streaming_data.get('free_options', [])) > 0,
                                            'language_options': streaming_data.get('language_specific', {})
                                        }
                                        
                                        trending_content.append(movie_data)
                            
                            await asyncio.sleep(0.1)  # Rate limiting
                
                # Sort by language priority and popularity
                trending_content.sort(key=lambda x: (x['language_priority_score'], x['popularity']), reverse=True)
                
            return trending_content[:12]
            
        except Exception as e:
            logger.error(f"Trending with languages error: {e}")
            return []
    
    @staticmethod
    async def _get_new_releases_by_language(language_code, region, days_back=60):
        """Get new releases for specific language"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.themoviedb.org/3/discover/movie"
                params = {
                    'api_key': TMDB_API_KEY,
                    'primary_release_date.gte': start_date.strftime('%Y-%m-%d'),
                    'primary_release_date.lte': end_date.strftime('%Y-%m-%d'),
                    'with_original_language': language_code,
                    'region': region,
                    'sort_by': 'primary_release_date.desc',
                    'vote_count.gte': 10
                }
                
                new_releases = []
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for movie in data.get('results', [])[:8]:
                            streaming_data = await OTTAvailabilityService.get_comprehensive_availability(
                                movie['id'], 'movie', region.lower()
                            )
                            
                            movie_data = {
                                'id': movie['id'],
                                'title': movie['title'],
                                'overview': movie['overview'][:150] + '...' if len(movie.get('overview', '')) > 150 else movie.get('overview', ''),
                                'poster_path': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None,
                                'release_date': movie.get('release_date'),
                                'vote_average': movie.get('vote_average'),
                                'vote_count': movie.get('vote_count'),
                                'language': language_code,
                                'streaming_availability': streaming_data,
                                'is_free_available': len(streaming_data.get('free_options', [])) > 0,
                                'language_options': streaming_data.get('language_specific', {})
                            }
                            
                            new_releases.append(movie_data)
                        
                return new_releases
                
        except Exception as e:
            logger.error(f"New releases by language error for {language_code}: {e}")
            return []
    
    @staticmethod
    def _calculate_language_priority(original_language, spoken_languages, language_priority):
        """Calculate language priority score"""
        score = 0
        
        # Check original language
        if original_language in language_priority:
            score += (len(language_priority) - language_priority.index(original_language)) * 2
        
        # Check spoken languages
        for lang in spoken_languages:
            if lang in language_priority:
                score += (len(language_priority) - language_priority.index(lang))
        
        return score
    
    @staticmethod
    async def _get_best_rated_movies(language_priority, region):
        """Get best rated movies across languages"""
        try:
            best_movies = []
            
            async with aiohttp.ClientSession() as session:
                for lang in language_priority[:3]:
                    url = f"https://api.themoviedb.org/3/discover/movie"
                    params = {
                        'api_key': TMDB_API_KEY,
                        'with_original_language': lang,
                        'vote_average.gte': 7.0,
                        'vote_count.gte': 100,
                        'sort_by': 'vote_average.desc',
                        'region': region
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for movie in data.get('results', [])[:4]:
                                streaming_data = await OTTAvailabilityService.get_comprehensive_availability(
                                    movie['id'], 'movie', region.lower()
                                )
                                
                                movie_data = {
                                    'id': movie['id'],
                                    'title': movie['title'],
                                    'overview': movie['overview'][:150] + '...' if len(movie.get('overview', '')) > 150 else movie.get('overview', ''),
                                    'poster_path': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None,
                                    'release_date': movie.get('release_date'),
                                    'vote_average': movie.get('vote_average'),
                                    'vote_count': movie.get('vote_count'),
                                    'language': lang,
                                    'streaming_availability': streaming_data,
                                    'is_free_available': len(streaming_data.get('free_options', [])) > 0,
                                    'language_options': streaming_data.get('language_specific', {})
                                }
                                
                                best_movies.append(movie_data)
                    
                    await asyncio.sleep(0.1)
                
                best_movies.sort(key=lambda x: x['vote_average'], reverse=True)
                
            return best_movies[:12]
            
        except Exception as e:
            logger.error(f"Best rated movies error: {e}")
            return []
    
    @staticmethod
    async def _get_critics_choice(language_priority, region):
        """Get critics' choice movies"""
        try:
            critics_movies = []
            current_year = datetime.now().year
            
            async with aiohttp.ClientSession() as session:
                for year in [current_year, current_year - 1]:
                    url = f"https://api.themoviedb.org/3/discover/movie"
                    params = {
                        'api_key': TMDB_API_KEY,
                        'primary_release_year': year,
                        'vote_average.gte': 8.0,
                        'vote_count.gte': 50,
                        'sort_by': 'vote_average.desc',
                        'region': region
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for movie in data.get('results', [])[:4]:
                                streaming_data = await OTTAvailabilityService.get_comprehensive_availability(
                                    movie['id'], 'movie', region.lower()
                                )
                                
                                movie_data = {
                                    'id': movie['id'],
                                    'title': movie['title'],
                                    'overview': movie['overview'][:150] + '...' if len(movie.get('overview', '')) > 150 else movie.get('overview', ''),
                                    'poster_path': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None,
                                    'release_date': movie.get('release_date'),
                                    'vote_average': movie.get('vote_average'),
                                    'vote_count': movie.get('vote_count'),
                                    'streaming_availability': streaming_data,
                                    'is_free_available': len(streaming_data.get('free_options', [])) > 0,
                                    'language_options': streaming_data.get('language_specific', {})
                                }
                                
                                critics_movies.append(movie_data)
                    
                    await asyncio.sleep(0.1)
                
            return critics_movies[:8]
            
        except Exception as e:
            logger.error(f"Critics choice error: {e}")
            return []
    
    @staticmethod
    async def _get_genre_recommendations(genre_id, language_priority, region):
        """Get recommendations for specific genre"""
        try:
            genre_movies = []
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.themoviedb.org/3/discover/movie"
                params = {
                    'api_key': TMDB_API_KEY,
                    'with_genres': genre_id,
                    'vote_average.gte': 6.0,
                    'vote_count.gte': 20,
                    'sort_by': 'popularity.desc',
                    'region': region
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for movie in data.get('results', [])[:6]:
                            streaming_data = await OTTAvailabilityService.get_comprehensive_availability(
                                movie['id'], 'movie', region.lower()
                            )
                            
                            movie_data = {
                                'id': movie['id'],
                                'title': movie['title'],
                                'poster_path': f"https://image.tmdb.org/t/p/w300{movie['poster_path']}" if movie.get('poster_path') else None,
                                'vote_average': movie.get('vote_average'),
                                'release_date': movie.get('release_date'),
                                'streaming_availability': streaming_data,
                                'is_free_available': len(streaming_data.get('free_options', [])) > 0
                            }
                            
                            genre_movies.append(movie_data)
                
            return genre_movies
            
        except Exception as e:
            logger.error(f"Genre recommendations error for genre {genre_id}: {e}")
            return []

# TMDB Service
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
            'append_to_response': 'credits,videos,similar'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB details error: {e}")
        return None

# Content Service
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
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
            
            # Get OTT platforms asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ott_data = loop.run_until_complete(
                    OTTAvailabilityService.get_comprehensive_availability(
                        tmdb_data['id'], content_type, 'in'
                    )
                )
            finally:
                loop.close()
            
            content = Content(
                tmdb_id=tmdb_data['id'],
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
                ott_platforms=json.dumps(ott_data),
                youtube_availability=json.dumps(ott_data.get('youtube_data', {})),
                streaming_languages=json.dumps(ott_data.get('language_specific', {}))
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def map_genre_ids(genre_ids):
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]

# Telegram Service
class TelegramService:
    @staticmethod
    def send_admin_recommendation(content, admin_name, description):
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            # Extract content information
            genres_list = []
            if content.genres:
                try:
                    genres_list = json.loads(content.genres)
                except:
                    genres_list = []
            
            # Get streaming availability data
            streaming_data = {}
            language_options = {}
            youtube_info = ""
            
            if content.ott_platforms:
                try:
                    streaming_data = json.loads(content.ott_platforms)
                except:
                    streaming_data = {}
            
            if content.streaming_languages:
                try:
                    language_options = json.loads(content.streaming_languages)
                except:
                    language_options = {}
            
            if content.youtube_availability:
                try:
                    youtube_data = json.loads(content.youtube_availability)
                    if youtube_data.get('free_movies'):
                        youtube_info = "\n🎬 **Free on YouTube!**"
                    elif youtube_data.get('trailers'):
                        youtube_info = "\n📺 **Trailer Available**"
                except:
                    pass
            
            # Create poster URL
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w500{content.poster_path}"
            
            # Create base message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}{youtube_info}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}"""
            
            # Check if multiple language options are available
            if language_options and len(language_options) > 1:
                # Create inline keyboard with language-specific buttons
                keyboard = TelegramService._create_language_keyboard(language_options, streaming_data)
                
                message += "\n\n🎯 **Choose Your Language to Watch:**"
                
                # Send with inline keyboard
                if poster_url:
                    try:
                        bot.send_photo(
                            chat_id=TELEGRAM_CHANNEL_ID,
                            photo=poster_url,
                            caption=message,
                            parse_mode='Markdown',
                            reply_markup=keyboard
                        )
                    except Exception as photo_error:
                        logger.error(f"Failed to send photo with keyboard: {photo_error}")
                        bot.send_message(
                            TELEGRAM_CHANNEL_ID, 
                            message, 
                            parse_mode='Markdown',
                            reply_markup=keyboard
                        )
                else:
                    bot.send_message(
                        TELEGRAM_CHANNEL_ID, 
                        message, 
                        parse_mode='Markdown',
                        reply_markup=keyboard
                    )
            else:
                # Single language or no language-specific options - include direct links in message
                watch_links = TelegramService._create_direct_links_text(streaming_data, language_options)
                
                if watch_links:
                    message += f"\n\n🔗 **Watch Now:**\n{watch_links}"
                
                message += "\n\n#AdminChoice #MovieRecommendation #CineScope"
                
                # Send regular message
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
    
    @staticmethod
    def _create_language_keyboard(language_options, streaming_data):
        """Create inline keyboard with language-specific watch buttons"""
        try:
            from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
            
            keyboard = InlineKeyboardMarkup(row_width=2)
            buttons = []
            
            # Priority order for language display
            language_priority = ['Hindi', 'Telugu', 'Tamil', 'Malayalam', 'Kannada', 'English']
            
            # Sort languages by priority
            sorted_languages = []
            for lang in language_priority:
                if lang in language_options:
                    sorted_languages.append(lang)
            
            # Add any remaining languages
            for lang in language_options.keys():
                if lang not in sorted_languages:
                    sorted_languages.append(lang)
            
            for language in sorted_languages:
                platforms = language_options[language]
                
                # Find the best platform for this language (prefer free, then subscription)
                best_platform = TelegramService._find_best_platform(platforms)
                
                if best_platform and best_platform.get('watch_url'):
                    # Create language-specific emoji
                    lang_emoji = TelegramService._get_language_emoji(language)
                    
                    # Create button text with platform info
                    platform_name = best_platform.get('platform_name', 'Stream')
                    if best_platform.get('is_free'):
                        button_text = f"{lang_emoji} {language} (Free)"
                    else:
                        button_text = f"{lang_emoji} {language} ({platform_name})"
                    
                    # Create button with direct watch URL
                    button = InlineKeyboardButton(
                        text=button_text,
                        url=best_platform['watch_url']
                    )
                    buttons.append(button)
            
            # Add YouTube options if available
            youtube_buttons = TelegramService._create_youtube_buttons(streaming_data)
            buttons.extend(youtube_buttons)
            
            # Add buttons to keyboard (2 per row)
            for i in range(0, len(buttons), 2):
                if i + 1 < len(buttons):
                    keyboard.row(buttons[i], buttons[i + 1])
                else:
                    keyboard.row(buttons[i])
            
            return keyboard
            
        except Exception as e:
            logger.error(f"Error creating language keyboard: {e}")
            return None
    
    @staticmethod
    def _find_best_platform(platforms):
        """Find the best platform from available options"""
        if not platforms:
            return None
        
        # Prioritize free platforms
        free_platforms = [p for p in platforms if p.get('is_free', False)]
        if free_platforms:
            # Among free platforms, prefer higher quality scores
            return max(free_platforms, key=lambda x: x.get('quality_score', 0))
        
        # If no free platforms, prefer subscription over rent/buy
        subscription_platforms = [p for p in platforms if p.get('availability_type') == 'subscription']
        if subscription_platforms:
            return subscription_platforms[0]
        
        # Fallback to any available platform
        return platforms[0]
    
    @staticmethod
    def _get_language_emoji(language):
        """Get emoji for language"""
        language_emojis = {
            'Hindi': '🇮🇳',
            'Telugu': '🎭',
            'Tamil': '🎪',
            'Malayalam': '🌴',
            'Kannada': '🎨',
            'English': '🇺🇸',
            'Bengali': '🐅',
            'Gujarati': '🦚',
            'Marathi': '🏺',
            'Punjabi': '🎺'
        }
        return language_emojis.get(language, '🎬')
    
    @staticmethod
    def _create_youtube_buttons(streaming_data):
        """Create YouTube-specific buttons"""
        try:
            from telebot.types import InlineKeyboardButton
            
            youtube_buttons = []
            youtube_data = streaming_data.get('youtube_data', {})
            
            # Free movies on YouTube
            free_movies = youtube_data.get('free_movies', [])
            if free_movies:
                best_free_movie = max(free_movies, key=lambda x: x.get('quality_score', 0))
                youtube_buttons.append(InlineKeyboardButton(
                    text="🎬 Watch Free on YouTube",
                    url=best_free_movie['watch_url']
                ))
            
            # Best trailer
            trailers = youtube_data.get('trailers', [])
            if trailers:
                best_trailer = max(trailers, key=lambda x: x.get('quality_score', 0))
                youtube_buttons.append(InlineKeyboardButton(
                    text="📺 Watch Trailer",
                    url=best_trailer['watch_url']
                ))
            
            return youtube_buttons
            
        except Exception as e:
            logger.error(f"Error creating YouTube buttons: {e}")
            return []
    
    @staticmethod
    def _create_direct_links_text(streaming_data, language_options):
        """Create direct links text for single language or fallback"""
        try:
            links_text = ""
            
            # Get all available platforms
            all_platforms = streaming_data.get('platforms', [])
            
            # Separate free and paid options
            free_platforms = [p for p in all_platforms if p.get('is_free', False)]
            paid_platforms = [p for p in all_platforms if not p.get('is_free', False)]
            
            # Add free options first
            if free_platforms:
                links_text += "🆓 **Free Options:**\n"
                for platform in free_platforms[:3]:  # Limit to 3 free options
                    platform_name = platform.get('platform_name', 'Unknown')
                    watch_url = platform.get('watch_url', '')
                    
                    if watch_url:
                        # Create language info if available
                        lang_info = ""
                        if platform.get('audio_language'):
                            lang_info = f" ({platform['audio_language']})"
                        
                        links_text += f"▶️ [{platform_name}{lang_info}]({watch_url})\n"
                
                links_text += "\n"
            
            # Add paid options
            if paid_platforms:
                links_text += "💰 **Subscription/Rent Options:**\n"
                for platform in paid_platforms[:3]:  # Limit to 3 paid options
                    platform_name = platform.get('platform_name', 'Unknown')
                    watch_url = platform.get('watch_url', '')
                    
                    if watch_url:
                        # Create language info if available
                        lang_info = ""
                        if platform.get('audio_language'):
                            lang_info = f" ({platform['audio_language']})"
                        
                        # Add pricing info if available
                        price_info = ""
                        if platform.get('price') and platform.get('currency'):
                            price_info = f" - {platform['currency']} {platform['price']}"
                        
                        links_text += f"▶️ [{platform_name}{lang_info}]({watch_url}){price_info}\n"
            
            # Add YouTube options
            youtube_data = streaming_data.get('youtube_data', {})
            youtube_links = TelegramService._create_youtube_links_text(youtube_data)
            if youtube_links:
                links_text += f"\n{youtube_links}"
            
            return links_text.strip()
            
        except Exception as e:
            logger.error(f"Error creating direct links text: {e}")
            return ""
    
    @staticmethod
    def _create_youtube_links_text(youtube_data):
        """Create YouTube links text"""
        try:
            youtube_text = ""
            
            # Free movies
            free_movies = youtube_data.get('free_movies', [])
            if free_movies:
                best_free_movie = max(free_movies, key=lambda x: x.get('quality_score', 0))
                youtube_text += f"🎬 [Watch Free on YouTube]({best_free_movie['watch_url']}) ({best_free_movie.get('duration_formatted', 'Unknown duration')})\n"
            
            # Trailers
            trailers = youtube_data.get('trailers', [])
            if trailers:
                best_trailer = max(trailers, key=lambda x: x.get('quality_score', 0))
                youtube_text += f"📺 [Watch Trailer]({best_trailer['watch_url']}) ({best_trailer.get('duration_formatted', 'Unknown duration')})\n"
            
            return youtube_text.strip()
            
        except Exception as e:
            logger.error(f"Error creating YouTube links text: {e}")
            return ""

    @staticmethod
    def send_enhanced_recommendation_with_languages(content, admin_name, description, language_options=None):
        """Enhanced method specifically for multi-language content"""
        try:
            if not bot or not TELEGRAM_CHANNEL_ID:
                return False
            
            # Get fresh streaming data if language_options not provided
            if not language_options and content.tmdb_id:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        availability_data = loop.run_until_complete(
                            OTTAvailabilityService.get_comprehensive_availability(
                                content.tmdb_id,
                                content.content_type,
                                'in'
                            )
                        )
                        language_options = availability_data.get('language_specific', {})
                    finally:
                        loop.close()
                        
                except Exception as e:
                    logger.error(f"Error getting fresh streaming data: {e}")
                    language_options = {}
            
            # Update content with latest streaming data
            if language_options:
                content.streaming_languages = json.dumps(language_options)
                try:
                    db.session.commit()
                except:
                    pass
            
            # Use the regular send method which now handles language options
            return TelegramService.send_admin_recommendation(content, admin_name, description)
            
        except Exception as e:
            logger.error(f"Enhanced recommendation send error: {e}")
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

# Content Routes
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
        
        # Get additional details from TMDB
        additional_details = None
        if content.tmdb_id:
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        
        # Get trailers from YouTube data
        trailers = []
        if content.youtube_availability:
            try:
                youtube_data = json.loads(content.youtube_availability)
                trailers = youtube_data.get('trailers', [])[:5]
            except:
                pass
        
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
            'ott_platforms': json.loads(content.ott_platforms or '{}'),
            'youtube_availability': json.loads(content.youtube_availability or '{}'),
            'streaming_languages': json.loads(content.streaming_languages or '{}'),
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

@app.route('/api/anime/<int:anime_id>', methods=['GET'])
def get_anime_details(anime_id):
    """Get detailed anime information with streaming options"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            anime_data = loop.run_until_complete(
                EnhancedAnimeService.get_anime_details(anime_id)
            )
        finally:
            loop.close()
        
        if not anime_data:
            return jsonify({'error': 'Anime not found'}), 404
        
        return jsonify(anime_data), 200
        
    except Exception as e:
        logger.error(f"Anime details error: {e}")
        return jsonify({'error': 'Failed to get anime details'}), 500

@app.route('/api/search', methods=['GET'])
def search_content():
    try:
        query = request.args.get('q', '')
        content_type = request.args.get('type', 'multi')
        page = int(request.args.get('page', 1))
        
        if not query:
            return jsonify({'error': 'Search query required'}), 400
        
        # Search in TMDB
        tmdb_results = TMDBService.search_content(query, content_type, 'en-US', page)
        
        results = []
        if tmdb_results and tmdb_results.get('results'):
            for item in tmdb_results['results'][:20]:
                # Determine content type
                item_type = item.get('media_type', content_type)
                if item_type == 'person':
                    continue
                
                # Save/update content in database
                saved_content = ContentService.save_content_from_tmdb(item, item_type)
                
                if saved_content:
                    results.append({
                        'id': saved_content.id,
                        'tmdb_id': saved_content.tmdb_id,
                        'title': saved_content.title,
                        'overview': saved_content.overview[:200] + '...' if saved_content.overview and len(saved_content.overview) > 200 else saved_content.overview,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{saved_content.poster_path}" if saved_content.poster_path else None,
                        'release_date': saved_content.release_date.isoformat() if saved_content.release_date else None,
                        'rating': saved_content.rating,
                        'content_type': saved_content.content_type,
                        'genres': json.loads(saved_content.genres or '[]'),
                        'is_free_available': len(json.loads(saved_content.ott_platforms or '{}').get('free_options', [])) > 0
                    })
        
        return jsonify({
            'query': query,
            'results': results,
            'page': page,
            'total_pages': tmdb_results.get('total_pages', 1) if tmdb_results else 1,
            'total_results': len(results)
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

@app.route('/api/recommendations/enhanced', methods=['GET'])
def get_enhanced_recommendations():
    """Get enhanced recommendations with language priorities"""
    try:
        region = request.args.get('region', 'IN')
        language_priority = request.args.getlist('languages') or LANGUAGE_PRIORITY
        page = int(request.args.get('page', 1))
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            recommendations = loop.run_until_complete(
                EnhancedRecommendationService.get_comprehensive_recommendations(
                    region, language_priority, page
                )
            )
        finally:
            loop.close()
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        logger.error(f"Enhanced recommendations error: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

@app.route('/api/ott/availability/<int:content_id>', methods=['GET'])
def get_ott_availability(content_id):
    """Get real-time OTT availability for content including YouTube"""
    try:
        content = Content.query.get_or_404(content_id)
        region = request.args.get('region', 'in')
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        if not content.tmdb_id:
            return jsonify({'error': 'TMDB ID not available for this content'}), 400
        
        if force_refresh or not content.ott_platforms:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                availability_data = loop.run_until_complete(
                    OTTAvailabilityService.get_comprehensive_availability(
                        content.tmdb_id,
                        content.content_type,
                        region
                    )
                )
                
                content.ott_platforms = json.dumps(availability_data)
                content.youtube_availability = json.dumps(availability_data.get('youtube_data', {}))
                content.streaming_languages = json.dumps(availability_data.get('language_specific', {}))
                content.updated_at = datetime.utcnow()
                db.session.commit()
                
            finally:
                loop.close()
        else:
            try:
                availability_data = json.loads(content.ott_platforms or '{}')
            except:
                availability_data = {'platforms': [], 'youtube_data': {}}
        
        return jsonify({
            'content_id': content.id,
            'title': content.title,
            'availability': availability_data
        }), 200
        
    except Exception as e:
        logger.error(f"OTT availability error: {e}")
        return jsonify({'error': 'Failed to get OTT availability'}), 500

@app.route('/api/streaming/language-options/<int:content_id>', methods=['GET'])
def get_language_streaming_options(content_id):
    """Get language-specific streaming options for content"""
    try:
        content = Content.query.get_or_404(content_id)
        region = request.args.get('region', 'in')
        
        if not content.tmdb_id:
            return jsonify({'error': 'TMDB ID not available'}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            availability_data = loop.run_until_complete(
                OTTAvailabilityService.get_comprehensive_availability(
                    content.tmdb_id,
                    content.content_type,
                    region
                )
            )
        finally:
            loop.close()
        
        # Format language-specific options
        language_options = {}
        for lang, options in availability_data.get('language_specific', {}).items():
            language_options[lang] = {
                'language_name': lang,
                'platforms': options,
                'free_options': [opt for opt in options if opt['is_free']],
                'paid_options': [opt for opt in options if not opt['is_free']]
            }
        
        return jsonify({
            'content_id': content.id,
            'title': content.title,
            'language_options': language_options,
            'all_platforms': availability_data.get('platforms', [])
        }), 200
        
    except Exception as e:
        logger.error(f"Language streaming options error: {e}")
        return jsonify({'error': 'Failed to get language streaming options'}), 500

# Admin Routes
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
            
            # Get comprehensive OTT availability
            ott_data = {}
            youtube_data = {}
            language_data = {}
            
            if data.get('id'):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    ott_data = loop.run_until_complete(
                        OTTAvailabilityService.get_comprehensive_availability(
                            data['id'],
                            data.get('content_type', 'movie'),
                            'in'
                        )
                    )
                    youtube_data = ott_data.get('youtube_data', {})
                    language_data = ott_data.get('language_specific', {})
                finally:
                    loop.close()
            
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
                ott_platforms=json.dumps(ott_data),
                youtube_availability=json.dumps(youtube_data),
                streaming_languages=json.dumps(language_data)
            )
            
            db.session.add(content)
            db.session.commit()
            
            return jsonify({
                'message': 'Content saved successfully with streaming data',
                'content_id': content.id,
                'youtube_available': len(youtube_data.get('free_movies', [])) > 0,
                'languages_available': list(language_data.keys())
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
        
        # Get latest streaming data for enhanced posting
        language_options = {}
        if content.tmdb_id:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    availability_data = loop.run_until_complete(
                        OTTAvailabilityService.get_comprehensive_availability(
                            content.tmdb_id,
                            content.content_type,
                            'in'
                        )
                    )
                    language_options = availability_data.get('language_specific', {})
                    
                    # Update content with latest data
                    content.ott_platforms = json.dumps(availability_data)
                    content.youtube_availability = json.dumps(availability_data.get('youtube_data', {}))
                    content.streaming_languages = json.dumps(language_options)
                    content.updated_at = datetime.utcnow()
                    db.session.commit()
                    
                finally:
                    loop.close()
                    
            except Exception as e:
                logger.error(f"Error getting streaming data for Telegram: {e}")
        
        # Send enhanced recommendation with language options
        telegram_success = TelegramService.send_enhanced_recommendation_with_languages(
            content, 
            current_user.username, 
            data['description'], 
            language_options
        )
        
        return jsonify({
            'message': 'Admin recommendation created successfully',
            'telegram_sent': telegram_success,
            'languages_available': list(language_options.keys()) if language_options else [],
            'streaming_platforms_count': len(language_options) if language_options else 0
        }), 201
        
    except Exception as e:
        logger.error(f"Admin recommendation error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create recommendation'}), 500
    
# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'features': ['streaming_availability_api', 'language_specific_streaming', 'youtube_integration', 'telegram_posting', 'enhanced_recommendations']
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

# Initialize database when app starts
create_tables()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)