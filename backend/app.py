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
# Add new imports for web scraping
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib.parse
import re

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
    ott_platforms = db.Column(db.Text)  # JSON string with detailed platform info
    ott_links = db.Column(db.Text)  # JSON string with direct watch links by language
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
    'netflix': {
        'name': 'Netflix',
        'is_free': False,
        'base_url': 'https://www.netflix.com',
        'search_url': 'https://www.netflix.com/search?q={query}',
        'logo': 'https://assets.nflxext.com/ffe/siteui/common/icons/nficon2016.png'
    },
    'amazon_prime': {
        'name': 'Amazon Prime Video',
        'is_free': False,
        'base_url': 'https://www.primevideo.com',
        'search_url': 'https://www.primevideo.com/search/ref=atv_sr_sug_1?phrase={query}',
        'logo': 'https://m.media-amazon.com/images/G/01/digital/video/web/Logo-min.png'
    },
    'hotstar': {
        'name': 'Disney+ Hotstar',
        'is_free': False,
        'base_url': 'https://www.hotstar.com',
        'search_url': 'https://www.hotstar.com/in/search?q={query}',
        'logo': 'https://img.hotstar.com/image/upload/v1656431456/web-images/logo-d-plus.svg'
    },
    'aha': {
        'name': 'Aha',
        'is_free': False,
        'base_url': 'https://www.aha.video',
        'search_url': 'https://www.aha.video/search?query={query}',
        'logo': 'https://d1j5aduzn6k3m5.cloudfront.net/images/ahaLogo.png'
    },
    'sun_nxt': {
        'name': 'Sun NXT',
        'is_free': False,
        'base_url': 'https://www.sunnxt.com',
        'search_url': 'https://www.sunnxt.com/search?q={query}',
        'logo': 'https://d1j5aduzn6k3m5.cloudfront.net/images/sunNxtLogo.png'
    },
    'mx_player': {
        'name': 'MX Player',
        'is_free': True,
        'base_url': 'https://www.mxplayer.in',
        'search_url': 'https://www.mxplayer.in/search?query={query}',
        'logo': 'https://www.mxplayer.in/assets/icon/apple-icon-180x180.png'
    },
    'jiocinema': {
        'name': 'JioCinema',
        'is_free': True,
        'base_url': 'https://www.jiocinema.com',
        'search_url': 'https://www.jiocinema.com/search/{query}',
        'logo': 'https://v3img.voot.com/jiocinemaIconRounded.png'
    },
    'sonyliv': {
        'name': 'SonyLIV',
        'is_free': False,
        'base_url': 'https://www.sonyliv.com',
        'search_url': 'https://www.sonyliv.com/search?query={query}',
        'logo': 'https://images.slivcdn.com/UI/sony_liv_logo.png'
    },
    'youtube': {
        'name': 'YouTube',
        'is_free': True,
        'base_url': 'https://www.youtube.com',
        'search_url': 'https://www.youtube.com/results?search_query={query}',
        'logo': 'https://www.youtube.com/s/desktop/img/favicon_144x144.png'
    },
    'airtel_xstream': {
        'name': 'Airtel Xstream',
        'is_free': False,
        'base_url': 'https://www.airtelxstream.in',
        'search_url': 'https://www.airtelxstream.in/search?q={query}',
        'logo': 'https://assets-global.website-files.com/63a4972d70e79e7aeafbf9c9/63a4972d70e79e0e78fc04bd_airtel-xstream-logo.png'
    },
    'zee5': {
        'name': 'ZEE5',
        'is_free': False,
        'base_url': 'https://www.zee5.com',
        'search_url': 'https://www.zee5.com/search?q={query}',
        'logo': 'https://akamaividz2.zee5.com/image/upload/resources/0-1-6z1791/portrait/zee5icon_1920_1080.jpg'
    },
    'voot': {
        'name': 'Voot',
        'is_free': True,
        'base_url': 'https://www.voot.com',
        'search_url': 'https://www.voot.com/search?q={query}',
        'logo': 'https://v3img.voot.com/v3Storage/assets/voot-logo-1671702595500.png'
    },
    'alt_balaji': {
        'name': 'ALTBalaji',
        'is_free': False,
        'base_url': 'https://www.altbalaji.com',
        'search_url': 'https://www.altbalaji.com/search?q={query}',
        'logo': 'https://www.altbalaji.com/assets/images/alt-balaji-logo.png'
    }
}

# Regional Language Mapping
REGIONAL_LANGUAGES = {
    'hindi': ['hi', 'hindi', 'bollywood'],
    'telugu': ['te', 'telugu', 'tollywood'],
    'tamil': ['ta', 'tamil', 'kollywood'],
    'kannada': ['kn', 'kannada', 'sandalwood'],
    'malayalam': ['ml', 'malayalam', 'mollywood'],
    'english': ['en', 'english', 'hollywood'],
    'marathi': ['mr', 'marathi'],
    'bengali': ['bn', 'bengali'],
    'gujarati': ['gu', 'gujarati'],
    'punjabi': ['pa', 'punjabi']
}

# OTT Scraping Services
class OTTScrapingService:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.selenium_driver = None
    
    def get_selenium_driver(self):
        """Initialize Selenium WebDriver for JavaScript-heavy sites"""
        if not self.selenium_driver:
            try:
                chrome_options = Options()
                chrome_options.add_argument('--headless')
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument('--window-size=1920,1080')
                self.selenium_driver = webdriver.Chrome(options=chrome_options)
            except Exception as e:
                logger.error(f"Failed to initialize Selenium driver: {e}")
                self.selenium_driver = None
        return self.selenium_driver
    
    def close_selenium_driver(self):
        """Close Selenium WebDriver"""
        if self.selenium_driver:
            try:
                self.selenium_driver.quit()
                self.selenium_driver = None
            except:
                pass

class JustWatchScraper(OTTScrapingService):
    def __init__(self):
        super().__init__()
        self.base_url = 'https://www.justwatch.com'
        self.search_url = 'https://www.justwatch.com/in/search'
    
    def search_content(self, title, year=None):
        """Search for content on JustWatch India"""
        try:
            search_query = title
            if year:
                search_query += f" {year}"
            
            params = {
                'q': search_query
            }
            
            response = self.session.get(self.search_url, params=params, timeout=15)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for content items
            content_items = soup.find_all('div', class_='title-list-row')
            
            ott_availability = []
            
            for item in content_items[:3]:  # Check first 3 results
                # Get content title to match
                title_elem = item.find('h3')
                if not title_elem:
                    continue
                
                item_title = title_elem.get_text(strip=True)
                
                # Simple title matching (you can improve this)
                if self._is_title_match(title, item_title):
                    # Look for streaming platforms
                    providers = item.find_all('img', {'data-original': True})
                    
                    for provider in providers:
                        provider_name = provider.get('alt', '').lower()
                        provider_img = provider.get('data-original', '')
                        
                        # Map JustWatch provider names to our platform IDs
                        platform_id = self._map_provider_to_platform(provider_name)
                        
                        if platform_id:
                            # Try to get direct link
                            link_elem = item.find('a', href=True)
                            direct_link = None
                            if link_elem:
                                content_url = self.base_url + link_elem['href']
                                direct_link = self._get_direct_watch_link(content_url, platform_id)
                            
                            ott_availability.append({
                                'platform': platform_id,
                                'platform_name': OTT_PLATFORMS[platform_id]['name'],
                                'is_free': OTT_PLATFORMS[platform_id]['is_free'],
                                'direct_link': direct_link,
                                'languages': ['Hindi', 'English'],  # Default languages
                                'quality': 'HD',
                                'subscription_required': not OTT_PLATFORMS[platform_id]['is_free']
                            })
                    
                    break
            
            return ott_availability
            
        except Exception as e:
            logger.error(f"JustWatch scraping error: {e}")
            return []
    
    def _is_title_match(self, search_title, found_title):
        """Simple title matching logic"""
        search_words = set(search_title.lower().split())
        found_words = set(found_title.lower().split())
        
        # Check if at least 70% of search words are in found title
        if len(search_words) == 0:
            return False
        
        match_count = len(search_words.intersection(found_words))
        return match_count / len(search_words) >= 0.7
    
    def _map_provider_to_platform(self, provider_name):
        """Map JustWatch provider names to our platform IDs"""
        mapping = {
            'netflix': 'netflix',
            'amazon prime video': 'amazon_prime',
            'disney+ hotstar': 'hotstar',
            'hotstar': 'hotstar',
            'sony liv': 'sonyliv',
            'sonyliv': 'sonyliv',
            'zee5': 'zee5',
            'mx player': 'mx_player',
            'voot': 'voot',
            'jiocinema': 'jiocinema',
            'youtube': 'youtube',
            'aha': 'aha',
            'sun nxt': 'sun_nxt'
        }
        
        for key, value in mapping.items():
            if key in provider_name.lower():
                return value
        return None
    
    def _get_direct_watch_link(self, content_url, platform_id):
        """Extract direct watch link from content page"""
        try:
            response = self.session.get(content_url, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for streaming links
            streaming_links = soup.find_all('a', {'data-track-event': 'selectOffer'})
            
            for link in streaming_links:
                href = link.get('href', '')
                if platform_id in href.lower() or any(keyword in href.lower() for keyword in [platform_id]):
                    return href
            
            return None
            
        except Exception as e:
            logger.error(f"Direct link extraction error: {e}")
            return None

class OTTPlayScraper(OTTScrapingService):
    def __init__(self):
        super().__init__()
        self.base_url = 'https://www.ottplay.com'
        self.search_url = 'https://www.ottplay.com/search'
    
    def search_content(self, title, year=None):
        """Search for content on OTTPlay"""
        try:
            params = {
                'q': title,
                'type': 'all'
            }
            
            response = self.session.get(self.search_url, params=params, timeout=15)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for search results
            result_items = soup.find_all('div', class_='search-result-item')
            
            ott_availability = []
            
            for item in result_items[:3]:
                # Get title and match
                title_elem = item.find('h3') or item.find('h2')
                if not title_elem:
                    continue
                
                item_title = title_elem.get_text(strip=True)
                
                if self._is_title_match(title, item_title):
                    # Look for OTT platform badges/icons
                    platform_elems = item.find_all('img', class_='platform-logo') or item.find_all('div', class_='platform-badge')
                    
                    for platform_elem in platform_elems:
                        platform_name = platform_elem.get('alt', '') or platform_elem.get('title', '')
                        platform_id = self._map_platform_name(platform_name)
                        
                        if platform_id:
                            # Get content page link for more details
                            content_link = item.find('a', href=True)
                            languages = self._extract_languages(item)
                            
                            availability_info = {
                                'platform': platform_id,
                                'platform_name': OTT_PLATFORMS[platform_id]['name'],
                                'is_free': OTT_PLATFORMS[platform_id]['is_free'],
                                'languages': languages,
                                'quality': 'HD',
                                'subscription_required': not OTT_PLATFORMS[platform_id]['is_free']
                            }
                            
                            # Try to get direct link
                            if content_link:
                                content_url = self.base_url + content_link['href']
                                direct_links = self._get_direct_links_by_language(content_url, platform_id)
                                availability_info['direct_links'] = direct_links
                            
                            ott_availability.append(availability_info)
                    
                    break
            
            return ott_availability
            
        except Exception as e:
            logger.error(f"OTTPlay scraping error: {e}")
            return []
    
    def _extract_languages(self, item_elem):
        """Extract available languages from the item"""
        languages = []
        
        # Look for language indicators
        lang_elems = item_elem.find_all(['span', 'div'], class_=['language', 'lang', 'audio-lang'])
        
        for lang_elem in lang_elems:
            lang_text = lang_elem.get_text(strip=True)
            if lang_text:
                languages.append(lang_text)
        
        # Default languages if none found
        if not languages:
            languages = ['Hindi', 'English']
        
        return languages
    
    def _get_direct_links_by_language(self, content_url, platform_id):
        """Get direct watch links for different languages"""
        try:
            response = self.session.get(content_url, timeout=10)
            if response.status_code != 200:
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            language_links = {}
            
            # Look for language-specific watch buttons
            watch_buttons = soup.find_all('a', class_=['watch-btn', 'stream-btn', 'play-btn'])
            
            for button in watch_buttons:
                # Try to extract language info
                lang_elem = button.find_parent().find(['span', 'div'], class_=['lang', 'language'])
                language = lang_elem.get_text(strip=True) if lang_elem else 'Default'
                
                # Get the actual streaming link
                href = button.get('href', '')
                if href and (platform_id in href.lower() or 'watch' in href.lower()):
                    language_links[language] = href
            
            return language_links
            
        except Exception as e:
            logger.error(f"Language links extraction error: {e}")
            return {}
    
    def _map_platform_name(self, platform_name):
        """Map platform name to our platform ID"""
        platform_name_lower = platform_name.lower()
        
        mapping = {
            'netflix': 'netflix',
            'prime': 'amazon_prime',
            'amazon': 'amazon_prime',
            'hotstar': 'hotstar',
            'disney': 'hotstar',
            'sony': 'sonyliv',
            'sonyliv': 'sonyliv',
            'zee5': 'zee5',
            'mx': 'mx_player',
            'mxplayer': 'mx_player',
            'voot': 'voot',
            'jio': 'jiocinema',
            'jiocinema': 'jiocinema',
            'youtube': 'youtube',
            'aha': 'aha',
            'sun': 'sun_nxt',
            'airtel': 'airtel_xstream'
        }
        
        for key, value in mapping.items():
            if key in platform_name_lower:
                return value
        return None
    
    def _is_title_match(self, search_title, found_title):
        """Check if titles match"""
        search_words = set(search_title.lower().split())
        found_words = set(found_title.lower().split())
        
        if len(search_words) == 0:
            return False
        
        match_count = len(search_words.intersection(found_words))
        return match_count / len(search_words) >= 0.6

class PlayPilotScraper(OTTScrapingService):
    def __init__(self):
        super().__init__()
        self.base_url = 'https://www.playpilot.com'
        self.search_url = 'https://www.playpilot.com/in/search'
    
    def search_content(self, title, year=None):
        """Search for content on PlayPilot"""
        try:
            # PlayPilot often requires JavaScript, so we'll use Selenium
            driver = self.get_selenium_driver()
            if not driver:
                return []
            
            search_query = title
            if year:
                search_query += f" {year}"
            
            # Navigate to search page
            driver.get(f"{self.search_url}?q={urllib.parse.quote(search_query)}")
            
            # Wait for results to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'search-result'))
            )
            
            # Get page source and parse
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Look for search results
            result_items = soup.find_all('div', class_=['search-result', 'content-item'])
            
            ott_availability = []
            
            for item in result_items[:3]:
                # Extract title
                title_elem = item.find(['h2', 'h3', 'h4'])
                if not title_elem:
                    continue
                
                item_title = title_elem.get_text(strip=True)
                
                if self._is_title_match(title, item_title):
                    # Look for streaming service icons/badges
                    service_elems = item.find_all(['img', 'div'], class_=['service-logo', 'platform-icon', 'streaming-service'])
                    
                    for service_elem in service_elems:
                        service_name = service_elem.get('alt', '') or service_elem.get('title', '') or service_elem.get_text(strip=True)
                        platform_id = self._map_service_to_platform(service_name)
                        
                        if platform_id:
                            # Check for language options
                            languages = self._get_available_languages(item)
                            
                            # Try to get direct watch link
                            watch_link = self._find_watch_link(item, platform_id)
                            
                            ott_info = {
                                'platform': platform_id,
                                'platform_name': OTT_PLATFORMS[platform_id]['name'],
                                'is_free': OTT_PLATFORMS[platform_id]['is_free'],
                                'direct_link': watch_link,
                                'languages': languages,
                                'quality': 'HD',
                                'subscription_required': not OTT_PLATFORMS[platform_id]['is_free']
                            }
                            
                            ott_availability.append(ott_info)
                    
                    break
            
            return ott_availability
            
        except Exception as e:
            logger.error(f"PlayPilot scraping error: {e}")
            return []
        finally:
            # Don't close driver here as it might be reused
            pass
    
    def _get_available_languages(self, item_elem):
        """Extract available languages"""
        languages = []
        
        # Look for language indicators
        lang_indicators = item_elem.find_all(['span', 'div'], class_=['language', 'audio', 'subtitle'])
        
        for indicator in lang_indicators:
            lang_text = indicator.get_text(strip=True)
            if lang_text and len(lang_text) < 20:  # Avoid getting long descriptions
                languages.append(lang_text)
        
        return languages if languages else ['English', 'Hindi']
    
    def _find_watch_link(self, item_elem, platform_id):
        """Find direct watch link for the platform"""
        # Look for watch/play buttons
        watch_buttons = item_elem.find_all('a', class_=['watch-btn', 'play-btn', 'stream-btn'])
        
        for button in watch_buttons:
            href = button.get('href', '')
            if href and platform_id in href.lower():
                return href
        
        return None
    
    def _map_service_to_platform(self, service_name):
        """Map service name to platform ID"""
        service_lower = service_name.lower()
        
        mapping = {
            'netflix': 'netflix',
            'prime video': 'amazon_prime',
            'amazon prime': 'amazon_prime',
            'hotstar': 'hotstar',
            'disney+': 'hotstar',
            'sonyliv': 'sonyliv',
            'sony liv': 'sonyliv',
            'zee5': 'zee5',
            'mx player': 'mx_player',
            'voot': 'voot',
            'jiocinema': 'jiocinema',
            'youtube': 'youtube',
            'aha': 'aha',
            'sun nxt': 'sun_nxt'
        }
        
        for key, value in mapping.items():
            if key in service_lower:
                return value
        return None
    
    def _is_title_match(self, search_title, found_title):
        """Check if titles match"""
        search_words = set(search_title.lower().split())
        found_words = set(found_title.lower().split())
        
        if len(search_words) == 0:
            return False
        
        match_count = len(search_words.intersection(found_words))
        return match_count / len(search_words) >= 0.6

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

# Enhanced Content Management Service
class ContentService:
    @staticmethod
    def save_content_from_tmdb(tmdb_data, content_type):
        try:
            # Check if content already exists
            existing = Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            if existing:
                # Update OTT availability if content is older than 24 hours
                if existing.updated_at < datetime.utcnow() - timedelta(hours=24):
                    ContentService.update_ott_availability(existing)
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
            
            # Get OTT platforms and direct links
            ott_data = ContentService.get_comprehensive_ott_availability(tmdb_data)
            
            # Create content object
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
                ott_platforms=json.dumps(ott_data['platforms']),
                ott_links=json.dumps(ott_data['links'])
            )
            
            db.session.add(content)
            db.session.commit()
            return content
            
        except Exception as e:
            logger.error(f"Error saving content: {e}")
            db.session.rollback()
            return None
    
    @staticmethod
    def get_comprehensive_ott_availability(tmdb_data):
        """Get comprehensive OTT availability using multiple scrapers"""
        title = tmdb_data.get('title') or tmdb_data.get('name', '')
        year = None
        
        # Extract year from release date
        release_date = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
        if release_date:
            try:
                year = datetime.strptime(release_date, '%Y-%m-%d').year
            except:
                pass
        
        all_platforms = []
        all_links = {}
        
        # Try JustWatch scraper
        try:
            justwatch_scraper = JustWatchScraper()
            justwatch_results = justwatch_scraper.search_content(title, year)
            if justwatch_results:
                all_platforms.extend(justwatch_results)
                logger.info(f"JustWatch found {len(justwatch_results)} platforms for {title}")
        except Exception as e:
            logger.error(f"JustWatch scraper error: {e}")
        
        # Try OTTPlay scraper
        try:
            ottplay_scraper = OTTPlayScraper()
            ottplay_results = ottplay_scraper.search_content(title, year)
            if ottplay_results:
                for result in ottplay_results:
                    # Merge with existing platforms or add new ones
                    existing_platform = next((p for p in all_platforms if p['platform'] == result['platform']), None)
                    if existing_platform:
                        # Merge language information
                        existing_platform['languages'] = list(set(existing_platform['languages'] + result['languages']))
                        if 'direct_links' in result:
                            existing_platform['direct_links'] = result['direct_links']
                    else:
                        all_platforms.append(result)
                
                logger.info(f"OTTPlay found {len(ottplay_results)} platforms for {title}")
        except Exception as e:
            logger.error(f"OTTPlay scraper error: {e}")
        
        # Try PlayPilot scraper (optional, as it's heavier)
        try:
            playpilot_scraper = PlayPilotScraper()
            playpilot_results = playpilot_scraper.search_content(title, year)
            if playpilot_results:
                for result in playpilot_results:
                    existing_platform = next((p for p in all_platforms if p['platform'] == result['platform']), None)
                    if not existing_platform:  # Only add if not already found
                        all_platforms.append(result)
                
                logger.info(f"PlayPilot found {len(playpilot_results)} platforms for {title}")
            
            # Close PlayPilot's Selenium driver
            playpilot_scraper.close_selenium_driver()
        except Exception as e:
            logger.error(f"PlayPilot scraper error: {e}")
        
        # Organize direct links by platform and language
        for platform_info in all_platforms:
            platform_id = platform_info['platform']
            
            if 'direct_links' in platform_info:
                all_links[platform_id] = platform_info['direct_links']
            elif 'direct_link' in platform_info and platform_info['direct_link']:
                all_links[platform_id] = {'default': platform_info['direct_link']}
        
        # Remove duplicate platforms
        unique_platforms = []
        seen_platforms = set()
        
        for platform in all_platforms:
            platform_id = platform['platform']
            if platform_id not in seen_platforms:
                seen_platforms.add(platform_id)
                unique_platforms.append(platform)
        
        return {
            'platforms': unique_platforms,
            'links': all_links
        }
    
    @staticmethod
    def update_ott_availability(content):
        """Update OTT availability for existing content"""
        try:
            # Create fake TMDB data for updating
            fake_tmdb_data = {
                'title': content.title,
                'name': content.title,
                'release_date': content.release_date.isoformat() if content.release_date else None
            }
            
            ott_data = ContentService.get_comprehensive_ott_availability(fake_tmdb_data)
            
            content.ott_platforms = json.dumps(ott_data['platforms'])
            content.ott_links = json.dumps(ott_data['links'])
            content.updated_at = datetime.utcnow()
            
            db.session.commit()
            logger.info(f"Updated OTT availability for {content.title}")
            
        except Exception as e:
            logger.error(f"Error updating OTT availability: {e}")
            db.session.rollback()
    
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

# Recommendation Engine
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
                'kannada': ['sandalwood', 'kannada movie', 'kannada film']
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
                    ott_platforms=json.dumps([]),  # You would check anime streaming platforms
                    ott_links=json.dumps({})
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
            
            # Format OTT platforms
            ott_platforms = []
            if content.ott_platforms:
                try:
                    ott_platforms = json.loads(content.ott_platforms)
                except:
                    ott_platforms = []
            
            platform_names = [platform.get('platform_name', '') for platform in ott_platforms[:3]]
            
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
📺 Available on: {', '.join(platform_names) if platform_names else 'Check local platforms'}

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
                    
                    # Parse OTT platforms and links
                    ott_platforms = json.loads(content.ott_platforms or '[]')
                    ott_links = json.loads(content.ott_links or '{}')
                    
                    # Enhance platform info with direct links
                    enhanced_platforms = []
                    for platform in ott_platforms:
                        platform_id = platform['platform']
                        platform_links = ott_links.get(platform_id, {})
                        
                        enhanced_platform = {
                            **platform,
                            'direct_links': platform_links,
                            'logo': OTT_PLATFORMS.get(platform_id, {}).get('logo', ''),
                            'base_url': OTT_PLATFORMS.get(platform_id, {}).get('base_url', '')
                        }
                        enhanced_platforms.append(enhanced_platform)
                    
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
                        'ott_platforms': enhanced_platforms
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
        
        # Get additional details from TMDB if available
        additional_details = None
        if content.tmdb_id:
            additional_details = TMDBService.get_content_details(content.tmdb_id, content.content_type)
        
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
        
        # Parse and enhance OTT platform information
        ott_platforms = json.loads(content.ott_platforms or '[]')
        ott_links = json.loads(content.ott_links or '{}')
        
        enhanced_platforms = []
        for platform in ott_platforms:
            platform_id = platform['platform']
            platform_links = ott_links.get(platform_id, {})
            
            enhanced_platform = {
                **platform,
                'direct_links': platform_links,
                'logo': OTT_PLATFORMS.get(platform_id, {}).get('logo', ''),
                'base_url': OTT_PLATFORMS.get(platform_id, {}).get('base_url', ''),
                'search_url': OTT_PLATFORMS.get(platform_id, {}).get('search_url', '').format(query=content.title) if OTT_PLATFORMS.get(platform_id, {}).get('search_url') else ''
            }
            enhanced_platforms.append(enhanced_platform)
        
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
            'ott_platforms': enhanced_platforms,
            'trailers': trailers,
            'similar_content': similar_content,
            'cast': additional_details.get('credits', {}).get('cast', [])[:10] if additional_details else [],
            'crew': additional_details.get('credits', {}).get('crew', [])[:5] if additional_details else []
        }), 200
        
    except Exception as e:
        logger.error(f"Content details error: {e}")
        return jsonify({'error': 'Failed to get content details'}), 500

# Enhanced OTT-specific routes
@app.route('/api/content/<int:content_id>/ott-links', methods=['GET'])
def get_ott_links(content_id):
    """Get detailed OTT links with language options for specific content"""
    try:
        content = Content.query.get_or_404(content_id)
        
        # Parse OTT data
        ott_platforms = json.loads(content.ott_platforms or '[]')
        ott_links = json.loads(content.ott_links or '{}')
        
        detailed_links = {}
        
        for platform in ott_platforms:
            platform_id = platform['platform']
            platform_info = OTT_PLATFORMS.get(platform_id, {})
            platform_links = ott_links.get(platform_id, {})
            
            detailed_links[platform_id] = {
                'platform_name': platform_info.get('name', platform_id),
                'is_free': platform_info.get('is_free', False),
                'logo': platform_info.get('logo', ''),
                'base_url': platform_info.get('base_url', ''),
                'languages': platform.get('languages', []),
                'direct_links': platform_links,
                'search_url': platform_info.get('search_url', '').format(query=content.title),
                'subscription_required': platform.get('subscription_required', True)
            }
        
        return jsonify({
            'content_id': content_id,
            'content_title': content.title,
            'ott_links': detailed_links,
            'total_platforms': len(detailed_links),
            'free_platforms': len([p for p in detailed_links.values() if p['is_free']]),
            'paid_platforms': len([p for p in detailed_links.values() if not p['is_free']])
        }), 200
        
    except Exception as e:
        logger.error(f"OTT links error: {e}")
        return jsonify({'error': 'Failed to get OTT links'}), 500

@app.route('/api/ott-platforms', methods=['GET'])
def get_all_ott_platforms():
    """Get information about all supported OTT platforms"""
    try:
        platform_list = []
        
        for platform_id, platform_info in OTT_PLATFORMS.items():
            platform_list.append({
                'id': platform_id,
                'name': platform_info['name'],
                'is_free': platform_info['is_free'],
                'base_url': platform_info['base_url'],
                'logo': platform_info.get('logo', ''),
                'description': f"{'Free' if platform_info['is_free'] else 'Subscription-based'} streaming platform"
            })
        
        return jsonify({
            'platforms': platform_list,
            'total_platforms': len(platform_list),
            'free_platforms': len([p for p in platform_list if p['is_free']]),
            'paid_platforms': len([p for p in platform_list if not p['is_free']])
        }), 200
        
    except Exception as e:
        logger.error(f"OTT platforms error: {e}")
        return jsonify({'error': 'Failed to get OTT platforms'}), 500

@app.route('/api/content/by-platform/<platform_id>', methods=['GET'])
def get_content_by_platform(platform_id):
    """Get content available on a specific OTT platform"""
    try:
        if platform_id not in OTT_PLATFORMS:
            return jsonify({'error': 'Invalid platform ID'}), 400
        
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        # Query content that has this platform in their ott_platforms
        content_query = Content.query.filter(Content.ott_platforms.like(f'%"{platform_id}"%'))
        
        paginated_content = content_query.paginate(page=page, per_page=per_page, error_out=False)
        
        result = []
        for content in paginated_content.items:
            ott_platforms = json.loads(content.ott_platforms or '[]')
            ott_links = json.loads(content.ott_links or '{}')
            
            # Find this platform's info
            platform_info = next((p for p in ott_platforms if p['platform'] == platform_id), None)
            platform_links = ott_links.get(platform_id, {})
            
            if platform_info:
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'platform_info': {
                        **platform_info,
                        'direct_links': platform_links
                    }
                })
        
        return jsonify({
            'platform': OTT_PLATFORMS[platform_id],
            'content': result,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': paginated_content.total,
                'pages': paginated_content.pages
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Content by platform error: {e}")
        return jsonify({'error': 'Failed to get content by platform'}), 500

# Recommendation Routes (Enhanced with OTT information)
@app.route('/api/recommendations/trending', methods=['GET'])
def get_trending():
    try:
        content_type = request.args.get('type', 'all')
        limit = int(request.args.get('limit', 20))
        
        recommendations = RecommendationEngine.get_trending_recommendations(limit, content_type)
        
        result = []
        for content in recommendations:
            ott_platforms = json.loads(content.ott_platforms or '[]')
            ott_links = json.loads(content.ott_links or '{}')
            
            # Enhance platform info
            enhanced_platforms = []
            for platform in ott_platforms:
                platform_id = platform['platform']
                platform_links = ott_links.get(platform_id, {})
                
                enhanced_platform = {
                    **platform,
                    'direct_links': platform_links,
                    'logo': OTT_PLATFORMS.get(platform_id, {}).get('logo', '')
                }
                enhanced_platforms.append(enhanced_platform)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': enhanced_platforms
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
            ott_platforms = json.loads(content.ott_platforms or '[]')
            ott_links = json.loads(content.ott_links or '{}')
            
            # Enhance platform info
            enhanced_platforms = []
            for platform in ott_platforms:
                platform_id = platform['platform']
                platform_links = ott_links.get(platform_id, {})
                
                enhanced_platform = {
                    **platform,
                    'direct_links': platform_links,
                    'logo': OTT_PLATFORMS.get(platform_id, {}).get('logo', '')
                }
                enhanced_platforms.append(enhanced_platform)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': enhanced_platforms
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
            ott_platforms = json.loads(content.ott_platforms or '[]')
            ott_links = json.loads(content.ott_links or '{}')
            
            # Enhance platform info
            enhanced_platforms = []
            for platform in ott_platforms:
                platform_id = platform['platform']
                platform_links = ott_links.get(platform_id, {})
                
                enhanced_platform = {
                    **platform,
                    'direct_links': platform_links,
                    'logo': OTT_PLATFORMS.get(platform_id, {}).get('logo', '')
                }
                enhanced_platforms.append(enhanced_platform)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': enhanced_platforms
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
            ott_platforms = json.loads(content.ott_platforms or '[]')
            ott_links = json.loads(content.ott_links or '{}')
            
            # Enhance platform info
            enhanced_platforms = []
            for platform in ott_platforms:
                platform_id = platform['platform']
                platform_links = ott_links.get(platform_id, {})
                
                enhanced_platform = {
                    **platform,
                    'direct_links': platform_links,
                    'logo': OTT_PLATFORMS.get(platform_id, {}).get('logo', '')
                }
                enhanced_platforms.append(enhanced_platform)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'overview': content.overview[:150] + '...' if content.overview else '',
                'ott_platforms': enhanced_platforms
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
                        ott_platforms = json.loads(content.ott_platforms or '[]')
                        ott_links = json.loads(content.ott_links or '{}')
                        
                        # Enhance platform info
                        enhanced_platforms = []
                        for platform in ott_platforms:
                            platform_id = platform['platform']
                            platform_links = ott_links.get(platform_id, {})
                            
                            enhanced_platform = {
                                **platform,
                                'direct_links': platform_links,
                                'logo': OTT_PLATFORMS.get(platform_id, {}).get('logo', '')
                            }
                            enhanced_platforms.append(enhanced_platform)
                        
                        result.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'rating': content.rating,
                            'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'ott_platforms': enhanced_platforms,
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
            ott_platforms = json.loads(content.ott_platforms or '[]')
            ott_links = json.loads(content.ott_links or '{}')
            
            # Enhance platform info
            enhanced_platforms = []
            for platform in ott_platforms:
                platform_id = platform['platform']
                platform_links = ott_links.get(platform_id, {})
                
                enhanced_platform = {
                    **platform,
                    'direct_links': platform_links,
                    'logo': OTT_PLATFORMS.get(platform_id, {}).get('logo', '')
                }
                enhanced_platforms.append(enhanced_platform)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': enhanced_platforms
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
            ott_platforms = json.loads(content.ott_platforms or '[]')
            ott_links = json.loads(content.ott_links or '{}')
            
            # Enhance platform info
            enhanced_platforms = []
            for platform in ott_platforms:
                platform_id = platform['platform']
                platform_links = ott_links.get(platform_id, {})
                
                enhanced_platform = {
                    **platform,
                    'direct_links': platform_links,
                    'logo': OTT_PLATFORMS.get(platform_id, {}).get('logo', '')
                }
                enhanced_platforms.append(enhanced_platform)
            
            result.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'rating': content.rating,
                'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                'ott_platforms': enhanced_platforms
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
                ott_platforms=json.dumps(data.get('ott_platforms', [])),
                ott_links=json.dumps(data.get('ott_links', {}))
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
        
        # OTT platform analytics
        platform_counts = defaultdict(int)
        all_content = Content.query.all()
        for content in all_content:
            if content.ott_platforms:
                platforms = json.loads(content.ott_platforms)
                for platform in platforms:
                    platform_counts[platform.get('platform_name', 'Unknown')] += 1
        
        popular_platforms = sorted(platform_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
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
            'popular_platforms': [
                {'platform': platform, 'content_count': count}
                for platform, count in popular_platforms
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500

# Admin Routes for OTT Management
@app.route('/api/admin/content/<int:content_id>/update-ott', methods=['POST'])
@require_admin
def admin_update_ott(current_user, content_id):
    """Admin endpoint to manually update OTT availability for content"""
    try:
        content = Content.query.get_or_404(content_id)
        
        # Force update OTT availability
        ContentService.update_ott_availability(content)
        
        return jsonify({
            'message': 'OTT availability updated successfully',
            'content_id': content_id,
            'ott_platforms': json.loads(content.ott_platforms or '[]'),
            'ott_links': json.loads(content.ott_links or '{}')
        }), 200
        
    except Exception as e:
        logger.error(f"Admin OTT update error: {e}")
        return jsonify({'error': 'Failed to update OTT availability'}), 500

@app.route('/api/admin/bulk-ott-update', methods=['POST'])
@require_admin
def admin_bulk_ott_update(current_user):
    """Admin endpoint to bulk update OTT availability for all content"""
    try:
        data = request.get_json()
        limit = data.get('limit', 50)
        
        # Get content that hasn't been updated recently
        old_content = Content.query.filter(
            Content.updated_at < datetime.utcnow() - timedelta(days=7)
        ).limit(limit).all()
        
        updated_count = 0
        for content in old_content:
            try:
                ContentService.update_ott_availability(content)
                updated_count += 1
            except Exception as e:
                logger.error(f"Failed to update OTT for content {content.id}: {e}")
                continue
        
        return jsonify({
            'message': f'Updated OTT availability for {updated_count} content items',
            'total_processed': len(old_content),
            'success_count': updated_count
        }), 200
        
    except Exception as e:
        logger.error(f"Bulk OTT update error: {e}")
        return jsonify({'error': 'Failed to perform bulk update'}), 500

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
                ott_platforms = json.loads(content.ott_platforms or '[]')
                ott_links = json.loads(content.ott_links or '{}')
                
                # Enhance platform info
                enhanced_platforms = []
                for platform in ott_platforms:
                    platform_id = platform['platform']
                    platform_links = ott_links.get(platform_id, {})
                    
                    enhanced_platform = {
                        **platform,
                        'direct_links': platform_links,
                        'logo': OTT_PLATFORMS.get(platform_id, {}).get('logo', '')
                    }
                    enhanced_platforms.append(enhanced_platform)
                
                result.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path else None,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'ott_platforms': enhanced_platforms,
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
            'ott_scraping': True,
            'direct_links': True,
            'language_support': True,
            'platform_count': len(OTT_PLATFORMS),
            'supported_platforms': list(OTT_PLATFORMS.keys())
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