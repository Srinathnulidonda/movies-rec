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
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '1002850793757')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'https://movies-rec-xmf5.onrender.com')

# Streaming Availability API Configuration
RAPIDAPI_KEY = "c50f156591mshac38b14b2f02d6fp1da925jsn4b816e4dae37"
RAPIDAPI_HOST = "streaming-availability.p.rapidapi.com"

# Initialize Telegram bot
if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != '7974343726:AAFUCW444L6jbj1tVLRyf8V7Isz2Ua1SxSk':
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

# Database Models (same as before)
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
    streaming_availability = db.Column(db.Text)  # New field for Streaming Availability API data
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

# Enhanced Streaming Availability Service
class StreamingAvailabilityService:
    BASE_URL = "https://streaming-availability.p.rapidapi.com"
    
    # Service mapping for better names and additional info
    SERVICE_MAPPING = {
        'netflix': {
            'name': 'Netflix',
            'logo': 'https://image.tmdb.org/t/p/original/wwemzKWzjKYJFfCeiB57q3r4Bcm.png',
            'type': 'subscription',
            'is_free': False
        },
        'prime': {
            'name': 'Amazon Prime Video',
            'logo': 'https://image.tmdb.org/t/p/original/68MNrwlkpF7WnmNPXLah69CR5cb.png',
            'type': 'subscription',
            'is_free': False
        },
        'hulu': {
            'name': 'Hulu',
            'logo': 'https://image.tmdb.org/t/p/original/pqUTCObHOqLs6MaaSnJ4az6fKAq.png',
            'type': 'subscription',
            'is_free': False
        },
        'disney': {
            'name': 'Disney+ Hotstar',
            'logo': 'https://image.tmdb.org/t/p/original/dgPueyEdOwpQ10fjuhL2WYFQwQs.png',
            'type': 'subscription',
            'is_free': False
        },
        'hbo': {
            'name': 'HBO Max',
            'logo': 'https://image.tmdb.org/t/p/original/nmU4DUFjpoaeFBoi83HnGlnGdvo.png',
            'type': 'subscription',
            'is_free': False
        },
        'peacock': {
            'name': 'Peacock',
            'logo': 'https://image.tmdb.org/t/p/original/gEA3x02KkJ4qcL5kUYJPBHgUMzj.png',
            'type': 'freemium',
            'is_free': True
        },
        'tubi': {
            'name': 'Tubi',
            'logo': 'https://image.tmdb.org/t/p/original/fj9Y8yNNjuPbOx4cydMlDOhA8IZ.png',
            'type': 'free',
            'is_free': True
        },
        'crackle': {
            'name': 'Crackle',
            'logo': 'https://image.tmdb.org/t/p/original/lDiEpnKa0s8kWXnvkfgBePvyGDj.png',
            'type': 'free',
            'is_free': True
        },
        'youtube': {
            'name': 'YouTube',
            'logo': 'https://image.tmdb.org/t/p/original/aS2zvJWn9mwiCOeaVQGBu7t0Kg.png',
            'type': 'freemium',
            'is_free': True
        },
        'apple': {
            'name': 'Apple TV+',
            'logo': 'https://image.tmdb.org/t/p/original/peURlLlr8jggOwK53fJ5wdQl05y.png',
            'type': 'subscription',
            'is_free': False
        },
        'paramount': {
            'name': 'Paramount+',
            'logo': 'https://image.tmdb.org/t/p/original/vRs0srfOonYjlxUFCGZRr6LRIc2.png',
            'type': 'subscription',
            'is_free': False
        },
        'showtime': {
            'name': 'Showtime',
            'logo': 'https://image.tmdb.org/t/p/original/NtY0Hl4A8VEpZlIvVhqjB3qrsF.png',
            'type': 'subscription',
            'is_free': False
        },
        'starz': {
            'name': 'Starz',
            'logo': 'https://image.tmdb.org/t/p/original/4b6b9Vvx4GkhnqjSRYhavIXTKJ1.png',
            'type': 'subscription',
            'is_free': False
        },
        # Indian Services
        'hotstar': {
            'name': 'Disney+ Hotstar',
            'logo': 'https://image.tmdb.org/t/p/original/dgPueyEdOwpQ10fjuhL2WYFQwQs.png',
            'type': 'subscription',
            'is_free': False
        },
        'jiocinema': {
            'name': 'JioCinema',
            'logo': 'https://image.tmdb.org/t/p/original/bPCaFjWY3X8a4ZEJHsYf1zUjXO2.png',
            'type': 'freemium',
            'is_free': True
        },
        'zee5': {
            'name': 'ZEE5',
            'logo': 'https://image.tmdb.org/t/p/original/1FErLGUVSvNLj9jNUMfQGZnOKPd.png',
            'type': 'freemium',
            'is_free': False
        },
        'sonyliv': {
            'name': 'SonyLIV',
            'logo': 'https://image.tmdb.org/t/p/original/nL3ctf2Vq4Qz3oSqMQ6PH2g4h6N.png',
            'type': 'freemium',
            'is_free': False
        },
        'voot': {
            'name': 'Voot',
            'logo': 'https://image.tmdb.org/t/p/original/aBAZC9TQ6K4E7zZP6YoNKPWXhwo.png',
            'type': 'freemium',
            'is_free': True
        },
        'mxplayer': {
            'name': 'MX Player',
            'logo': 'https://image.tmdb.org/t/p/original/tnAuB8q5vv7Ax9UAEje5Xi4BXik.png',
            'type': 'freemium',
            'is_free': True
        }
    }
    
    @staticmethod
    async def get_streaming_availability_by_imdb(imdb_id, country='in'):
        """Get streaming availability using IMDB ID"""
        try:
            headers = {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': RAPIDAPI_HOST
            }
            
            url = f"{StreamingAvailabilityService.BASE_URL}/get"
            params = {
                'imdb_id': imdb_id,
                'country': country,
                'output_language': 'en'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return StreamingAvailabilityService._parse_streaming_data(data)
                    else:
                        logger.error(f"Streaming API error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Streaming availability error: {e}")
            return None
    
    @staticmethod
    async def search_streaming_availability(title, year=None, country='in', content_type='movie'):
        """Search for content and get streaming availability"""
        try:
            headers = {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': RAPIDAPI_HOST
            }
            
            url = f"{StreamingAvailabilityService.BASE_URL}/search/title"
            params = {
                'title': title,
                'country': country,
                'output_language': 'en',
                'show_type': content_type
            }
            
            if year:
                params['year'] = year
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Get the first result
                        if data.get('result') and len(data['result']) > 0:
                            first_result = data['result'][0]
                            return StreamingAvailabilityService._parse_streaming_data(first_result)
                        else:
                            return None
                    else:
                        logger.error(f"Streaming search API error: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Streaming search error: {e}")
            return None
    
    @staticmethod
    def _parse_streaming_data(data):
        """Parse streaming data from API response"""
        try:
            streaming_info = {
                'title': data.get('title'),
                'imdb_id': data.get('imdbId'),
                'tmdb_id': data.get('tmdbId'),
                'year': data.get('year'),
                'platforms': [],
                'free_options': [],
                'subscription_options': [],
                'rent_options': [],
                'buy_options': [],
                'last_updated': datetime.utcnow().isoformat()
            }
            
            # Parse streaming options
            streaming_options = data.get('streamingOptions', {})
            
            for country, services in streaming_options.items():
                for service in services:
                    service_id = service.get('service', {}).get('id', '')
                    service_name = service.get('service', {}).get('name', service_id)
                    
                    # Get service info from mapping
                    service_info = StreamingAvailabilityService.SERVICE_MAPPING.get(
                        service_id.lower(), 
                        {
                            'name': service_name,
                            'logo': None,
                            'type': 'subscription',
                            'is_free': False
                        }
                    )
                    
                    platform_data = {
                        'platform_id': service_id.lower(),
                        'platform_name': service_info['name'],
                        'logo_url': service_info.get('logo'),
                        'watch_url': service.get('link', ''),
                        'availability_type': service.get('type', 'subscription'),
                        'is_free': service_info['is_free'] or service.get('type') == 'free',
                        'price': service.get('price', {}).get('amount'),
                        'currency': service.get('price', {}).get('currency'),
                        'quality': service.get('quality', 'HD'),
                        'audio_languages': service.get('audios', []),
                        'subtitle_languages': service.get('subtitles', []),
                        'expires_on': service.get('expiresOn'),
                        'available_since': service.get('availableSince'),
                        'source': 'streaming_availability_api'
                    }
                    
                    streaming_info['platforms'].append(platform_data)
                    
                    # Categorize platforms
                    if platform_data['is_free']:
                        streaming_info['free_options'].append(platform_data)
                    elif platform_data['availability_type'] == 'subscription':
                        streaming_info['subscription_options'].append(platform_data)
                    elif platform_data['availability_type'] == 'rent':
                        streaming_info['rent_options'].append(platform_data)
                    elif platform_data['availability_type'] == 'buy':
                        streaming_info['buy_options'].append(platform_data)
            
            return streaming_info
            
        except Exception as e:
            logger.error(f"Error parsing streaming data: {e}")
            return None

# Enhanced YouTube Service (keeping the existing one)
class EnhancedYouTubeService:
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    # Official movie channels and distributors
    OFFICIAL_CHANNELS = {
        'bollywood': [
            'UCq-Fj5jknLsUf-MWSy4_brA',  # T-Series
            'UCjvgGbPPn-FV0I2Ms_S3AeA',  # Zee Music Company
            'UCFFbwnve3yF62-tV3hIXhBg',  # Sony Music India
            'UCpEhnqL0y41EpW2TvWAHD7Q',  # Saregama
        ],
        'hollywood': [
            'UCiifkYAs_bq1pt_zbNAzYGg',  # Sony Pictures
            'UC_VEXHtGJKI34LHaFZu_9jQ',  # Warner Bros
            'UCZf2iKB7HZN6Kj7UrXrtUyA',  # Universal Pictures
            'UCy4P9dbGpJx4RAnXWCKZLGg',  # Disney Movie Trailers
        ]
    }
    
    @staticmethod
    async def get_comprehensive_youtube_availability(title, original_title=None, release_year=None, content_type='movie', region='IN'):
        """Get comprehensive YouTube availability including free movies, trailers, and premium content"""
        if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == 'your_youtube_api_key':
            return []
        
        try:
            youtube_data = {
                'free_movies': [],
                'premium_content': [],
                'trailers': [],
                'clips': [],
                'official_content': [],
                'last_checked': datetime.utcnow().isoformat()
            }
            
            # Multiple search strategies
            search_results = await EnhancedYouTubeService._comprehensive_search(title, original_title, release_year, content_type, region)
            
            # Categorize results
            for video in search_results:
                category = EnhancedYouTubeService._categorize_video(video, title)
                if category:
                    youtube_data[category].append(video)
            
            return youtube_data
            
        except Exception as e:
            logger.error(f"YouTube comprehensive search error: {e}")
            return []
    
    @staticmethod
    async def _comprehensive_search(title, original_title, release_year, content_type, region):
        """Perform comprehensive search with multiple strategies"""
        all_results = []
        
        # Search queries with different variations
        search_queries = EnhancedYouTubeService._generate_search_queries(title, original_title, release_year, content_type, region)
        
        async with aiohttp.ClientSession() as session:
            for query in search_queries[:5]:  # Limit to 5 queries to avoid quota issues
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
                                    all_results.append(video_data)
                        
                        # Add delay to respect rate limits
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"YouTube search error for query '{query}': {e}")
                    continue
        
        # Remove duplicates
        seen_ids = set()
        unique_results = []
        for video in all_results:
            if video['video_id'] not in seen_ids:
                seen_ids.add(video['video_id'])
                unique_results.append(video)
        
        return unique_results
    
    @staticmethod
    def _generate_search_queries(title, original_title, release_year, content_type, region):
        """Generate comprehensive search queries"""
        queries = []
        year_str = str(release_year) if release_year else ""
        
        # Basic searches
        queries.extend([
            f"{title} full movie",
            f"{title} trailer",
            f"{title} movie {year_str}",
        ])
        
        # Original title searches
        if original_title and original_title != title:
            queries.extend([
                f"{original_title} full movie",
                f"{original_title} trailer"
            ])
        
        # Language-specific searches based on region
        if region == 'IN':
            queries.extend([
                f"{title} hindi movie",
                f"{title} bollywood movie",
            ])
        
        return queries
    
    @staticmethod
    async def _get_detailed_video_info(session, video_item):
        """Get detailed information about a video"""
        try:
            video_id = video_item['id']['videoId']
            
            # Get video details
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
                        
                        # Parse duration
                        duration = EnhancedYouTubeService._parse_duration(
                            video_details['contentDetails'].get('duration', 'PT0S')
                        )
                        
                        return {
                            'video_id': video_id,
                            'title': video_details['snippet']['title'],
                            'description': video_details['snippet']['description'],
                            'channel_title': video_details['snippet']['channelTitle'],
                            'channel_id': video_details['snippet']['channelId'],
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
            logger.error(f"Video details error for {video_item.get('id', {}).get('videoId')}: {e}")
            return None
    
    @staticmethod
    def _parse_duration(duration_str):
        """Parse YouTube duration format (PT1H2M3S) to seconds"""
        import re
        
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
            if any(keyword in title for keyword in ['full movie', 'complete movie', 'full film']):
                return 'free_movies'
        
        # Check for trailers (typically 1-5 minutes)
        elif 60 <= duration <= 300:
            if any(keyword in title for keyword in ['trailer', 'teaser', 'preview']):
                return 'trailers'
        
        # Check for clips and songs (typically 3-15 minutes)
        elif 180 <= duration <= 900:
            if any(keyword in title for keyword in ['song', 'clip', 'scene']):
                return 'clips'
        
        return None
    
    @staticmethod
    def _calculate_quality_score(video_details, duration):
        """Calculate quality score based on various factors"""
        score = 0.0
        
        # Duration score (prefer full-length content)
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
        elif view_count > 10000:  # 10K+ views
            score += 0.1
        
        # Like ratio score
        like_count = int(video_details['statistics'].get('likeCount', 0))
        if like_count > 1000:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0

# Enhanced OTT Availability Service using Streaming Availability API
class OTTAvailabilityService:
    
    @staticmethod
    async def get_comprehensive_availability(tmdb_id, content_type='movie', region='in'):
        """Get availability from Streaming Availability API and YouTube"""
        
        availability_data = {
            'platforms': [],
            'free_options': [],
            'subscription_options': [],
            'rent_options': [],
            'buy_options': [],
            'youtube_data': {},
            'streaming_data': {},
            'last_updated': datetime.utcnow().isoformat(),
            'region': region
        }
        
        try:
            # Get title information from TMDB first
            title_data = await OTTAvailabilityService.get_tmdb_title(tmdb_id, content_type)
            if not title_data:
                return availability_data
            
            availability_data['title'] = title_data['title']
            
            # Get IMDB ID for better streaming availability search
            imdb_id = None
            if title_data.get('imdb_id'):
                imdb_id = title_data['imdb_id']
            
            # Get streaming availability using the new API
            if imdb_id:
                streaming_data = await StreamingAvailabilityService.get_streaming_availability_by_imdb(
                    imdb_id, region
                )
            else:
                # Fallback to title search
                streaming_data = await StreamingAvailabilityService.search_streaming_availability(
                    title_data['title'], 
                    title_data.get('release_year'), 
                    region, 
                    content_type
                )
            
            if streaming_data:
                availability_data['streaming_data'] = streaming_data
                availability_data['platforms'].extend(streaming_data.get('platforms', []))
                availability_data['free_options'].extend(streaming_data.get('free_options', []))
                availability_data['subscription_options'].extend(streaming_data.get('subscription_options', []))
                availability_data['rent_options'].extend(streaming_data.get('rent_options', []))
                availability_data['buy_options'].extend(streaming_data.get('buy_options', []))
            
            # Get comprehensive YouTube availability
            youtube_data = await EnhancedYouTubeService.get_comprehensive_youtube_availability(
                title_data['title'],
                title_data.get('original_title'),
                title_data.get('release_year'),
                content_type,
                'IN'
            )
            availability_data['youtube_data'] = youtube_data
            
            # Convert YouTube data to platform format and add to platforms
            youtube_platforms = OTTAvailabilityService._convert_youtube_to_platforms(youtube_data)
            availability_data['platforms'].extend(youtube_platforms)
            
            # Add YouTube platforms to appropriate categories
            for platform in youtube_platforms:
                if platform['is_free']:
                    availability_data['free_options'].append(platform)
            
            return availability_data
            
        except Exception as e:
            logger.error(f"Comprehensive availability error: {e}")
            return availability_data
    
    @staticmethod
    def _convert_youtube_to_platforms(youtube_data):
        """Convert YouTube data to platform format"""
        platforms = []
        
        # Free movies
        for movie in youtube_data.get('free_movies', []):
            platforms.append({
                'platform_id': 'youtube',
                'platform_name': 'YouTube',
                'logo_url': 'https://image.tmdb.org/t/p/original/aS2zvJWn9mwiCOeaVQGBu7t0Kg.png',
                'watch_url': movie['watch_url'],
                'availability_type': 'free',
                'is_free': True,
                'price': None,
                'currency': None,
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
                'logo_url': 'https://image.tmdb.org/t/p/original/aS2zvJWn9mwiCOeaVQGBu7t0Kg.png',
                'watch_url': best_trailer['watch_url'],
                'availability_type': 'free',
                'is_free': True,
                'price': None,
                'currency': None,
                'source': 'youtube',
                'video_title': best_trailer['title'],
                'duration': best_trailer['duration_formatted'],
                'quality_score': best_trailer['quality_score'],
                'view_count': best_trailer['view_count'],
                'channel_name': best_trailer['channel_title'],
                'content_type': 'trailer'
            })
        
        return platforms
    
    @staticmethod
    async def get_tmdb_title(tmdb_id, content_type):
        """Get title and IMDB ID from TMDB"""
        try:
            url = f"https://api.themoviedb.org/3/{content_type}/{tmdb_id}"
            params = {
                'api_key': TMDB_API_KEY,
                'append_to_response': 'external_ids'
            }
            
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
                            'release_year': release_year,
                            'imdb_id': data.get('external_ids', {}).get('imdb_id')
                        }
            return None
            
        except Exception as e:
            logger.error(f"TMDB title fetch error: {e}")
            return None

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
            'append_to_response': 'credits,videos,similar,external_ids'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"TMDB details error: {e}")
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
            
            # Get comprehensive availability data
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                availability_data = loop.run_until_complete(
                    OTTAvailabilityService.get_comprehensive_availability(
                        tmdb_data['id'], content_type, 'in'
                    )
                )
            finally:
                loop.close()
            
            # Create content object
            content = Content(
                tmdb_id=tmdb_data['id'],
                imdb_id=tmdb_data.get('external_ids', {}).get('imdb_id'),
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
                ott_platforms=json.dumps(availability_data),
                youtube_availability=json.dumps(availability_data.get('youtube_data', {})),
                streaming_availability=json.dumps(availability_data.get('streaming_data', {}))
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
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]

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
            
            # Parse streaming availability
            streaming_info = ""
            if content.streaming_availability:
                try:
                    streaming_data = json.loads(content.streaming_availability)
                    platforms = streaming_data.get('platforms', [])
                    
                    if platforms:
                        free_platforms = [p for p in platforms if p.get('is_free')]
                        paid_platforms = [p for p in platforms if not p.get('is_free')]
                        
                        streaming_info = "\n\n━━━━━━━━━━━━━━━━━\n📺 **WHERE TO WATCH**\n\n"
                        
                        # Free platforms
                        if free_platforms:
                            streaming_info += "🆓 **FREE:**\n"
                            for platform in free_platforms[:3]:
                                streaming_info += f"• {platform['platform_name']}\n"
                            streaming_info += "\n"
                        
                        # Paid platforms
                        if paid_platforms:
                            streaming_info += "💰 **SUBSCRIPTION/RENT:**\n"
                            for platform in paid_platforms[:5]:
                                platform_name = platform['platform_name']
                                if platform.get('availability_type') == 'rent':
                                    platform_name += " (Rent)"
                                elif platform.get('availability_type') == 'buy':
                                    platform_name += " (Buy)"
                                streaming_info += f"• {platform_name}\n"
                        
                        streaming_info += "\n━━━━━━━━━━━━━━━━━"
                
                except Exception as e:
                    logger.error(f"Error parsing streaming data: {e}")
            
            # Add YouTube availability info
            youtube_info = ""
            if content.youtube_availability:
                try:
                    youtube_data = json.loads(content.youtube_availability)
                    if youtube_data.get('free_movies'):
                        youtube_info = "\n🎬 **Free on YouTube!**"
                    elif youtube_data.get('trailers'):
                        youtube_info = "\n📺 **Trailer on YouTube**"
                except:
                    pass
            
            # Create message
            message = f"""🎬 **Admin's Choice** by {admin_name}

**{content.title}**
⭐ Rating: {content.rating or 'N/A'}/10
📅 Release: {content.release_date or 'N/A'}
🎭 Genres: {', '.join(genres_list[:3]) if genres_list else 'N/A'}
🎬 Type: {content.content_type.upper()}{youtube_info}

📝 **Admin's Note:** {description}

📖 **Synopsis:** {(content.overview[:200] + '...') if content.overview else 'No synopsis available'}{streaming_info}

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

# API Routes

@app.route('/api/ott/availability/<int:content_id>', methods=['GET'])
def get_ott_availability(content_id):
    """Get real-time streaming availability for content"""
    try:
        content = Content.query.get_or_404(content_id)
        region = request.args.get('region', 'in')
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        if not content.tmdb_id:
            return jsonify({'error': 'TMDB ID not available for this content'}), 400
        
        # Check if we need to refresh
        if force_refresh or not content.streaming_availability:
            # Get fresh data
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
                
                # Update content in database
                content.ott_platforms = json.dumps(availability_data)
                content.youtube_availability = json.dumps(availability_data.get('youtube_data', {}))
                content.streaming_availability = json.dumps(availability_data.get('streaming_data', {}))
                content.updated_at = datetime.utcnow()
                db.session.commit()
                
            finally:
                loop.close()
        else:
            # Use cached data
            try:
                availability_data = json.loads(content.ott_platforms or '{}')
                if not availability_data.get('streaming_data') and content.streaming_availability:
                    availability_data['streaming_data'] = json.loads(content.streaming_availability)
                if not availability_data.get('youtube_data') and content.youtube_availability:
                    availability_data['youtube_data'] = json.loads(content.youtube_availability)
            except:
                availability_data = {'platforms': [], 'streaming_data': {}, 'youtube_data': {}}
        
        return jsonify({
            'content_id': content.id,
            'title': content.title,
            'availability': availability_data
        }), 200
        
    except Exception as e:
        logger.error(f"OTT availability error: {e}")
        return jsonify({'error': 'Failed to get streaming availability'}), 500

@app.route('/api/streaming/search', methods=['POST'])
def search_streaming_content():
    """Search for content and return streaming availability"""
    try:
        data = request.get_json()
        
        if not data.get('title'):
            return jsonify({'error': 'Title is required'}), 400
        
        title = data['title']
        year = data.get('year')
        country = data.get('country', 'in')
        content_type = data.get('content_type', 'movie')
        
        # Get streaming availability
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            streaming_data = loop.run_until_complete(
                StreamingAvailabilityService.search_streaming_availability(
                    title, year, country, content_type
                )
            )
            
            if streaming_data:
                return jsonify({
                    'success': True,
                    'streaming_data': streaming_data
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': 'No streaming data found'
                }), 404
                
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Streaming search error: {e}")
        return jsonify({'error': 'Failed to search streaming availability'}), 500

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
            'overview': content.overview,
            'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
            'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path else None,
            'streaming_availability': json.loads(content.streaming_availability or '{}'),
            'youtube_availability': json.loads(content.youtube_availability or '{}'),
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Search Route
@app.route('/api/search', methods=['GET'])
def search_content():
    try:
        query = request.args.get('q', '').strip()
        page = int(request.args.get('page', 1))
        content_type = request.args.get('type', 'multi')
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        # Search TMDB
        results = TMDBService.search_content(query, content_type, page=page)
        
        if not results:
            return jsonify({'error': 'Search failed'}), 500
        
        formatted_results = []
        for item in results.get('results', []):
            # Save to database and get streaming info
            content = ContentService.save_content_from_tmdb(
                item, 
                item.get('media_type', content_type if content_type != 'multi' else 'movie')
            )
            
            if content:
                formatted_results.append({
                    'id': content.id,
                    'tmdb_id': content.tmdb_id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview,
                    'has_streaming_data': bool(content.streaming_availability),
                    'has_youtube_data': bool(content.youtube_availability)
                })
        
        return jsonify({
            'results': formatted_results,
            'page': results.get('page', 1),
            'total_pages': results.get('total_pages', 1),
            'total_results': results.get('total_results', 0)
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Search failed'}), 500

# Admin Routes
@app.route('/api/admin/content', methods=['POST'])
@require_admin
def save_external_content(current_user):
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No content data provided'}), 400
        
        # Check if content already exists
        existing_content = None
        if data.get('id'):
            existing_content = Content.query.filter_by(tmdb_id=data['id']).first()
        
        if existing_content:
            return jsonify({
                'message': 'Content already exists',
                'content_id': existing_content.id
            }), 200
        
        # Create new content with enhanced streaming data
        content = ContentService.save_content_from_tmdb(data, data.get('content_type', 'movie'))
        
        if content:
            return jsonify({
                'message': 'Content saved successfully with streaming data',
                'content_id': content.id,
                'has_streaming_data': bool(content.streaming_availability),
                'has_youtube_data': bool(content.youtube_availability)
            }), 201
        else:
            return jsonify({'error': 'Failed to save content'}), 500
        
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

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.1.0',
        'features': ['streaming_availability_api', 'youtube_integration', 'telegram_posting'],
        'api_status': {
            'streaming_availability': bool(RAPIDAPI_KEY),
            'youtube': bool(YOUTUBE_API_KEY),
            'tmdb': bool(TMDB_API_KEY),
            'telegram': bool(bot)
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

# Initialize database when app starts
create_tables()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)