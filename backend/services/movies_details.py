# backend/services/movies_details.py
import re
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from functools import wraps
from slugify import slugify
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from sqlalchemy import and_, or_, func, desc, text
from sqlalchemy.orm import Session
from flask import current_app, has_app_context
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import unicodedata
from dotenv import load_dotenv
from difflib import SequenceMatcher

load_dotenv()

logger = logging.getLogger(__name__)

TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
OMDB_API_KEY = os.environ.get('OMDB_API_KEY')
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY environment variable is required")
if not OMDB_API_KEY:
    raise ValueError("OMDB_API_KEY environment variable is required")
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY environment variable is required")

TMDB_BASE_URL = 'https://api.themoviedb.org/3'
OMDB_BASE_URL = 'http://www.omdbapi.com/'
YOUTUBE_BASE_URL = 'https://www.googleapis.com/youtube/v3'
JIKAN_BASE_URL = 'https://api.jikan.moe/v4'

POSTER_BASE = 'https://image.tmdb.org/t/p/w500'
BACKDROP_BASE = 'https://image.tmdb.org/t/p/w1280'
PROFILE_BASE = 'https://image.tmdb.org/t/p/w185'
STILL_BASE = 'https://image.tmdb.org/t/p/w780'

def ensure_app_context(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if has_app_context():
            return func(*args, **kwargs)
        else:
            try:
                if current_app:
                    with current_app.app_context():
                        return func(*args, **kwargs)
                else:
                    logger.warning(f"No app context available for {func.__name__}")
                    if 'cast_crew' in func.__name__:
                        return {'cast': [], 'crew': {'directors': [], 'writers': [], 'producers': []}}
                    elif 'reviews' in func.__name__:
                        return []
                    elif 'similar' in func.__name__:
                        return []
                    elif 'gallery' in func.__name__:
                        return {'posters': [], 'backdrops': [], 'stills': []}
                    elif 'user_data' in func.__name__:
                        return {'in_watchlist': False, 'in_favorites': False, 'user_rating': None, 'watch_status': None}
                    else:
                        return None
            except Exception as e:
                logger.error(f"Error in app context wrapper for {func.__name__}: {e}")
                if 'cast_crew' in func.__name__:
                    return {'cast': [], 'crew': {'directors': [], 'writers': [], 'producers': []}}
                elif 'reviews' in func.__name__:
                    return []
                elif 'similar' in func.__name__:
                    return []
                elif 'gallery' in func.__name__:
                    return {'posters': [], 'backdrops': [], 'stills': []}
                elif 'user_data' in func.__name__:
                    return {'in_watchlist': False, 'in_favorites': False, 'user_rating': None, 'watch_status': None}
                else:
                    return None
    return wrapper

@dataclass
class ContentDetails:
    slug: str
    id: int
    title: str
    original_title: Optional[str]
    overview: str
    content_type: str
    poster_url: Optional[str]
    backdrop_url: Optional[str]
    trailer: Optional[Dict]
    synopsis: Dict
    cast_crew: Dict
    ratings: Dict
    metadata: Dict
    more_like_this: List[Dict]
    reviews: List[Dict]
    gallery: Dict
    streaming_info: Optional[Dict]
    seasons_episodes: Optional[Dict]
    
    def to_dict(self):
        return asdict(self)

class SlugManager:
    
    @staticmethod
    def normalize_title(title: str) -> str:
        if not title or not isinstance(title, str):
            return ""
        
        try:
            clean_title = str(title).strip()
            if not clean_title:
                return ""
            
            normalized = unicodedata.normalize('NFKD', clean_title)
            normalized = normalized.encode('ascii', 'ignore').decode('ascii')
            
            normalized = re.sub(r'[^\w\s\-\']', '', normalized)
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing title '{title}': {e}")
            return str(title).strip()
    
    @staticmethod
    def extract_year_from_title(title: str) -> Tuple[str, Optional[int]]:
        try:
            year_patterns = [
                r'\((\d{4})\)$',
                r'\s(\d{4})$',
                r'-(\d{4})$',
                r'\[(\d{4})\]$'
            ]
            
            for pattern in year_patterns:
                match = re.search(pattern, title)
                if match:
                    year = int(match.group(1))
                    if 1900 <= year <= 2030:
                        clean_title = re.sub(pattern, '', title).strip()
                        return clean_title, year
            
            return title, None
            
        except Exception as e:
            logger.error(f"Error extracting year from title '{title}': {e}")
            return title, None
    
    @staticmethod
    def detect_content_type(title: str, original_title: str = None, genres: List[str] = None) -> str:
        try:
            title_lower = title.lower() if title else ""
            original_lower = original_title.lower() if original_title else ""
            
            anime_indicators = [
                'anime', 'manga', 'otaku', 'chan', 'kun', 'san', 'sama',
                'senpai', 'kouhai', 'sensei', 'dojo', 'ninja', 'samurai',
                'yokai', 'kami', 'studio ghibli', 'madhouse', 'pierrot',
                'bones', 'shaft', 'trigger', 'mappa', 'wit studio'
            ]
            
            tv_indicators = [
                'series', 'season', 'episode', 'tv show', 'television',
                'mini-series', 'limited series', 'anthology'
            ]
            
            if any(indicator in title_lower or indicator in original_lower for indicator in anime_indicators):
                return 'anime'
            
            if genres and any(genre.lower() in ['animation', 'anime'] for genre in genres):
                if any(indicator in title_lower for indicator in anime_indicators):
                    return 'anime'
            
            if any(indicator in title_lower for indicator in tv_indicators):
                return 'tv'
            
            return 'movie'
            
        except Exception as e:
            logger.error(f"Error detecting content type: {e}")
            return 'movie'
    
    @staticmethod
    def generate_slug(title: str, year: Optional[int] = None, content_type: str = 'movie', 
                     original_title: str = None, tmdb_id: int = None) -> str:
        try:
            if not title or not isinstance(title, str):
                fallback = f"content-{tmdb_id or int(time.time())}"
                logger.warning(f"Invalid title provided, using fallback: {fallback}")
                return fallback
            
            clean_title, extracted_year = SlugManager.extract_year_from_title(title)
            
            if not year and extracted_year:
                year = extracted_year
            
            normalized_title = SlugManager.normalize_title(clean_title)
            
            if not normalized_title:
                fallback = f"content-{tmdb_id or int(time.time())}"
                logger.warning(f"Title normalization failed, using fallback: {fallback}")
                return fallback
            
            try:
                slug = slugify(normalized_title, max_length=70, word_boundary=True, save_order=True)
            except Exception as slugify_error:
                logger.warning(f"Slugify failed for '{normalized_title}': {slugify_error}")
                slug = SlugManager._manual_slugify(normalized_title)
            
            if not slug or len(slug) < 2:
                slug = SlugManager._manual_slugify(normalized_title)
            
            if not slug:
                type_prefix = {
                    'movie': 'movie',
                    'tv': 'tv-show',
                    'anime': 'anime',
                    'person': 'person'
                }.get(content_type, 'content')
                return f"{type_prefix}-{tmdb_id or int(time.time())}"
            
            if content_type == 'anime' and not slug.startswith('anime-'):
                slug = f"anime-{slug}"
            
            if year and content_type in ['movie', 'anime'] and isinstance(year, int):
                if 1900 <= year <= 2030:
                    slug = f"{slug}-{year}"
            
            if len(slug) > 120:
                parts = slug[:117].split('-')
                if len(parts) > 1:
                    slug = '-'.join(parts[:-1])
                else:
                    slug = slug[:117]
            
            return slug
            
        except Exception as e:
            logger.error(f"Critical error generating slug for title '{title}': {e}")
            type_prefix = {
                'movie': 'movie',
                'tv': 'tv-show',
                'anime': 'anime',
                'person': 'person'
            }.get(content_type, 'content')
            return f"{type_prefix}-{tmdb_id or int(time.time())}"
    
    @staticmethod
    def _manual_slugify(text: str) -> str:
        try:
            slug = text.lower()
            slug = re.sub(r'[^\w\s-]', '', slug)
            slug = re.sub(r'[-\s]+', '-', slug)
            slug = slug.strip('-')
            return slug[:70] if slug else ""
        except Exception:
            return ""
            
    @staticmethod
    def generate_unique_slug(db, model, title: str, year: Optional[int] = None, 
                           content_type: str = 'movie', existing_id: Optional[int] = None,
                           original_title: str = None, tmdb_id: int = None) -> str:
        try:
            base_slug = SlugManager.generate_slug(title, year, content_type, original_title, tmdb_id)
            
            if not base_slug:
                type_prefix = {
                    'movie': 'movie',
                    'tv': 'tv-show',
                    'anime': 'anime',
                    'person': 'person'
                }.get(content_type, 'content')
                base_slug = f"{type_prefix}-{tmdb_id or int(time.time())}"
            
            slug = base_slug
            counter = 1
            max_attempts = 100
            
            while counter <= max_attempts:
                try:
                    query = db.session.query(model.id).filter_by(slug=slug)
                    
                    if existing_id:
                        query = query.filter(model.id != existing_id)
                    
                    exists = query.first() is not None
                    
                    if not exists:
                        break
                    
                    if counter == 1:
                        if year and content_type in ['movie', 'anime']:
                            slug = f"{base_slug.replace(f'-{year}', '')}-{year}-{counter}"
                        else:
                            slug = f"{base_slug}-{counter}"
                    else:
                        base_part = base_slug.replace(f'-{counter-1}', '') if counter > 2 else base_slug
                        slug = f"{base_part}-{counter}"
                    
                    counter += 1
                    
                except Exception as e:
                    logger.error(f"Error checking slug uniqueness: {e}")
                    slug = f"{base_slug}-{int(time.time())}"
                    break
            
            if counter > max_attempts:
                timestamp_slug = f"{base_slug}-{int(time.time())}"
                logger.warning(f"Hit max attempts for slug generation, using timestamp: {timestamp_slug}")
                return timestamp_slug
            
            return slug
            
        except Exception as e:
            logger.error(f"Critical error generating unique slug: {e}")
            type_prefix = {
                'movie': 'movie',
                'tv': 'tv-show',
                'anime': 'anime',
                'person': 'person'
            }.get(content_type, 'content')
            return f"{type_prefix}-{tmdb_id or int(time.time())}-{abs(hash(str(title)))[:6]}"
    
    @staticmethod
    def extract_info_from_slug(slug: str) -> Dict:
        try:
            if not slug:
                return {'title': 'Unknown', 'year': None, 'content_type': 'movie'}
            
            content_type = 'movie'
            clean_slug = slug
            
            type_patterns = [
                ('anime-', 'anime'),
                ('tv-show-', 'tv'),
                ('tv-', 'tv'),
                ('series-', 'tv'),
                ('person-', 'person')
            ]
            
            for prefix, ctype in type_patterns:
                if slug.startswith(prefix):
                    content_type = ctype
                    clean_slug = slug[len(prefix):]
                    break
            
            year_pattern = r'-(\d{4})(?:-\d+)?$'
            year_match = re.search(year_pattern, clean_slug)
            year = None
            title_slug = clean_slug
            
            if year_match:
                potential_year = int(year_match.group(1))
                if 1900 <= potential_year <= 2030:
                    year = potential_year
                    title_slug = clean_slug[:year_match.start()]
            
            title = SlugManager._slug_to_title(title_slug)
            
            return {
                'title': title,
                'year': year,
                'content_type': content_type
            }
            
        except Exception as e:
            logger.error(f"Error extracting info from slug '{slug}': {e}")
            return {
                'title': slug.replace('-', ' ').title() if slug else 'Unknown',
                'year': None,
                'content_type': 'movie'
            }
    
    @staticmethod
    def _slug_to_title(slug: str) -> str:
        try:
            title = slug.replace('-', ' ').title()
            
            title_fixes = {
                'Dc': 'DC',
                'Mcu': 'MCU',
                'Uk': 'UK',
                'Us': 'US',
                'Tv': 'TV',
                'Ai': 'AI',
                'Fbi': 'FBI',
                'Cia': 'CIA',
                'Ufc': 'UFC',
                'Wwe': 'WWE',
                'Nba': 'NBA',
                'Nfl': 'NFL'
            }
            
            for wrong, correct in title_fixes.items():
                title = re.sub(f'\\b{wrong}\\b', correct, title)
            
            roman_numerals = ['Ii', 'Iii', 'Iv', 'Vi', 'Vii', 'Viii', 'Ix', 'Xi', 'Xii', 'Xiii', 'Xiv', 'Xv']
            for numeral in roman_numerals:
                title = re.sub(f'\\b{numeral}\\b', numeral.upper(), title)
            
            return title
            
        except Exception as e:
            logger.error(f"Error converting slug to title: {e}")
            return slug.replace('-', ' ').title()
    
    @staticmethod
    def update_content_slug(db, content, force_update: bool = False) -> str:
        try:
            if content.slug and not force_update:
                return content.slug
            
            content_type = getattr(content, 'content_type', 'movie')
            if hasattr(content, '__tablename__') and content.__tablename__ == 'persons':
                content_type = 'person'
            
            title = getattr(content, 'title', '') or getattr(content, 'name', '')
            original_title = getattr(content, 'original_title', None)
            tmdb_id = getattr(content, 'tmdb_id', None)
            year = None
            
            if hasattr(content, 'release_date') and content.release_date:
                try:
                    year = content.release_date.year
                except:
                    pass
            elif hasattr(content, 'birthday') and content.birthday:
                try:
                    year = content.birthday.year
                except:
                    pass
            
            if not title:
                title = f"Content {getattr(content, 'id', 'Unknown')}"
            
            new_slug = SlugManager.generate_unique_slug(
                db, 
                content.__class__, 
                title, 
                year, 
                content_type,
                existing_id=getattr(content, 'id', None),
                original_title=original_title,
                tmdb_id=tmdb_id
            )
            
            content.slug = new_slug
            
            return new_slug
            
        except Exception as e:
            logger.error(f"Error updating content slug: {e}")
            fallback = f"content-{getattr(content, 'id', int(time.time()))}"
            content.slug = fallback
            return fallback

class ContentService:
    
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.Content = models.get('Content')
        self.Person = models.get('Person')
    
    def save_content_from_tmdb(self, tmdb_data: Dict, content_type: str) -> Any:
        try:
            tmdb_id = tmdb_data.get('id')
            if not tmdb_id:
                logger.warning("No TMDB ID provided in data")
                return None
            
            existing = self.Content.query.filter_by(tmdb_id=tmdb_id).first()
            
            if existing:
                if not existing.slug or existing.slug.startswith('content-'):
                    try:
                        SlugManager.update_content_slug(self.db, existing, force_update=True)
                        self.db.session.commit()
                    except Exception as e:
                        logger.warning(f"Failed to update existing content slug: {e}")
                return existing
            
            title = tmdb_data.get('title') or tmdb_data.get('name') or 'Unknown Title'
            original_title = tmdb_data.get('original_title') or tmdb_data.get('original_name')
            
            release_date = None
            year = None
            date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
            if date_str:
                try:
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    year = release_date.year
                except Exception as e:
                    logger.warning(f"Error parsing date '{date_str}': {e}")
            
            genres = []
            if 'genres' in tmdb_data:
                genres = [genre['name'] for genre in tmdb_data['genres']]
            elif 'genre_ids' in tmdb_data:
                genres = self._map_genre_ids(tmdb_data['genre_ids'])
            
            detected_type = SlugManager.detect_content_type(title, original_title, genres)
            final_content_type = detected_type if detected_type != 'movie' else content_type
            
            slug = SlugManager.generate_unique_slug(
                self.db, self.Content, title, year, final_content_type, 
                original_title=original_title, tmdb_id=tmdb_id
            )
            
            content_data = {
                'slug': slug,
                'tmdb_id': tmdb_id,
                'title': title,
                'original_title': original_title,
                'content_type': final_content_type,
                'genres': json.dumps(genres) if genres else None,
                'release_date': release_date,
                'rating': tmdb_data.get('vote_average'),
                'vote_count': tmdb_data.get('vote_count'),
                'popularity': tmdb_data.get('popularity'),
                'overview': tmdb_data.get('overview'),
                'poster_path': tmdb_data.get('poster_path'),
                'backdrop_path': tmdb_data.get('backdrop_path')
            }
            
            content = self.Content(**content_data)
            self.db.session.add(content)
            self.db.session.commit()
            
            logger.info(f"Saved content: {title} with slug: {slug}")
            return content
            
        except Exception as e:
            logger.error(f"Error saving content from TMDB: {e}")
            self.db.session.rollback()
            return None
    
    def save_anime_content(self, anime_data: Dict) -> Any:
        try:
            mal_id = anime_data.get('mal_id')
            if not mal_id:
                logger.warning("No MAL ID provided in anime data")
                return None
            
            existing = self.Content.query.filter_by(mal_id=mal_id).first()
            
            if existing:
                if not existing.slug or existing.slug.startswith('content-'):
                    try:
                        SlugManager.update_content_slug(self.db, existing, force_update=True)
                        self.db.session.commit()
                    except Exception as e:
                        logger.warning(f"Failed to update existing anime slug: {e}")
                return existing
            
            title = anime_data.get('title') or 'Unknown Anime'
            original_title = anime_data.get('title_japanese')
            
            release_date = None
            year = None
            if anime_data.get('aired', {}).get('from'):
                try:
                    release_date = datetime.strptime(anime_data['aired']['from'][:10], '%Y-%m-%d').date()
                    year = release_date.year
                except Exception as e:
                    logger.warning(f"Error parsing anime date: {e}")
            
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            slug = SlugManager.generate_unique_slug(
                self.db, self.Content, title, year, 'anime',
                original_title=original_title, tmdb_id=mal_id
            )
            
            content_data = {
                'slug': slug,
                'mal_id': mal_id,
                'title': title,
                'original_title': original_title,
                'content_type': 'anime',
                'genres': json.dumps(genres) if genres else None,
                'release_date': release_date,
                'rating': anime_data.get('score'),
                'vote_count': anime_data.get('scored_by'),
                'popularity': anime_data.get('popularity'),
                'overview': anime_data.get('synopsis'),
                'poster_path': anime_data.get('images', {}).get('jpg', {}).get('image_url')
            }
            
            content = self.Content(**content_data)
            self.db.session.add(content)
            self.db.session.commit()
            
            logger.info(f"Saved anime: {title} with slug: {slug}")
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime content: {e}")
            self.db.session.rollback()
            return None
    
    def get_or_create_person(self, person_data: Dict) -> Any:
        try:
            tmdb_id = person_data.get('id')
            if not tmdb_id:
                return None
            
            existing = self.Person.query.filter_by(tmdb_id=tmdb_id).first()
            
            if existing:
                if not existing.slug or existing.slug.startswith('person-'):
                    try:
                        name = existing.name or f"Person {existing.id}"
                        slug = SlugManager.generate_unique_slug(
                            self.db, self.Person, name, content_type='person',
                            existing_id=existing.id, tmdb_id=tmdb_id
                        )
                        existing.slug = slug
                        self.db.session.commit()
                    except Exception as e:
                        logger.warning(f"Failed to update person slug: {e}")
                return existing
            
            name = person_data.get('name') or 'Unknown Person'
            slug = SlugManager.generate_unique_slug(
                self.db, self.Person, name, content_type='person', tmdb_id=tmdb_id
            )
            
            person_data_clean = {
                'slug': slug,
                'tmdb_id': tmdb_id,
                'name': name,
                'profile_path': person_data.get('profile_path'),
                'popularity': person_data.get('popularity'),
                'known_for_department': person_data.get('known_for_department'),
                'gender': person_data.get('gender')
            }
            
            person = self.Person(**person_data_clean)
            self.db.session.add(person)
            self.db.session.flush()
            
            return person
            
        except Exception as e:
            logger.error(f"Error creating person: {e}")
            return None
    
    def _map_genre_ids(self, genre_ids: List[int]) -> List[str]:
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

class TMDBService:
    @staticmethod
    def search_content(query, content_type='multi', page=1):
        try:
            if not TMDB_API_KEY:
                return None
            
            url = f"{TMDB_BASE_URL}/search/{content_type}"
            params = {
                'api_key': TMDB_API_KEY,
                'query': query,
                'page': page
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"TMDB search error: {e}")
            return None

class MoviesDetailsService:
    
    def __init__(self, db, models, cache=None):
        self.db = db
        self.Content = models.get('Content')
        self.User = models.get('User')
        self.UserInteraction = models.get('UserInteraction')
        self.Review = models.get('Review')
        self.Person = models.get('Person')
        self.ContentPerson = models.get('ContentPerson')
        self.cache = cache
        self.models = models
        
        self.content_service = ContentService(db, models)
        self.session = self._create_http_session()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._verify_api_keys()
    
    def _verify_api_keys(self):
        try:
            if not TMDB_API_KEY:
                logger.error("TMDB_API_KEY not found in environment variables")
                raise ValueError("TMDB_API_KEY environment variable is required")
            
            if not OMDB_API_KEY:
                logger.error("OMDB_API_KEY not found in environment variables")
                raise ValueError("OMDB_API_KEY environment variable is required")
            
            if not YOUTUBE_API_KEY:
                logger.error("YOUTUBE_API_KEY not found in environment variables")
                raise ValueError("YOUTUBE_API_KEY environment variable is required")
            
            logger.info("All required API keys verified successfully")
            
        except Exception as e:
            logger.error(f"API key verification failed: {e}")
            raise e
    
    def _create_http_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            read=3,
            connect=3,
            backoff_factor=0.3,
            status_forcelist=(500, 502, 504)
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def get_details_by_slug(self, slug: str, user_id: Optional[int] = None, force_refresh: bool = False) -> Optional[Dict]:
        try:
            if not has_app_context():
                logger.warning("No app context available for get_details_by_slug")
                return None
            
            cache_key = f"details:slug:{slug}"
            if self.cache and not force_refresh:
                try:
                    cached = self.cache.get(cache_key)
                    if cached:
                        logger.info(f"Cache hit for slug: {slug}")
                        if user_id:
                            cached = self._add_user_data(cached, user_id)
                        return cached
                except Exception as e:
                    logger.warning(f"Cache error for slug {slug}: {e}")
            
            content = self.Content.query.filter_by(slug=slug).first()
            
            if not content:
                content = self._find_content_fuzzy(slug)
                
                if not content:
                    info = SlugManager.extract_info_from_slug(slug)
                    if self._should_fetch_from_external(slug, info['title'], info['year']):
                        content = self._try_fetch_from_external(info['title'], info['year'], info['content_type'])
                        if content:
                            logger.info(f"Fetched content from external API for slug: {slug}")
                    
                    if not content:
                        logger.debug(f"Content not found for slug: {slug}")
                        return None
            
            if not content.slug or content.slug.startswith('content-'):
                try:
                    SlugManager.update_content_slug(self.db, content, force_update=True)
                    self.db.session.commit()
                    if self.cache:
                        old_cache_key = f"details:slug:{slug}"
                        new_cache_key = f"details:slug:{content.slug}"
                        self.cache.delete(old_cache_key)
                        self.cache.delete(new_cache_key)
                except Exception as e:
                    logger.warning(f"Failed to update content slug: {e}")
            
            details = self._build_content_details(content, user_id)
            
            if self.cache and details and not force_refresh:
                try:
                    cache_data = details.copy()
                    cache_data.pop('user_data', None)
                    self.cache.set(cache_key, cache_data, timeout=1800)
                except Exception as e:
                    logger.warning(f"Cache set error for slug {slug}: {e}")
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting details for slug {slug}: {e}")
            return None
    
    def get_details_with_reviews(self, slug, user_id=None):
        try:
            # Get basic content details
            details = self.get_details_by_slug(slug, user_id)
            if not details:
                return None
            
            # Get reviews using review service
            if hasattr(current_app, 'review_service'):
                review_result = current_app.review_service.get_content_reviews(
                    slug, page=1, limit=20, sort_by='newest', user_id=user_id
                )
                if review_result['success']:
                    details['reviews'] = review_result['reviews']
                    details['review_stats'] = review_result['stats']
                    details['pagination'] = review_result['pagination']
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting details with reviews: {e}")
            return details if 'details' in locals() else None
    
    def _find_content_fuzzy(self, slug: str) -> Optional[Any]:
        try:
            info = SlugManager.extract_info_from_slug(slug)
            title = info['title']
            year = info['year']
            content_type = info['content_type']
            
            logger.debug(f"Fuzzy search for slug '{slug}': title='{title}', year={year}, type={content_type}")
            
            slug_results = self._search_by_slug_patterns(slug)
            if slug_results:
                logger.info(f"Found {len(slug_results)} results by slug pattern for '{slug}'")
                return slug_results[0]
            
            title_variations = self._generate_comprehensive_title_variations(title)
            
            results = []
            
            strategies = [
                ('exact_match_with_year', lambda v: self._search_exact_with_year(v, content_type, year)),
                ('flexible_year_match', lambda v: self._search_flexible_year(v, content_type, year)),
                ('normalized_title_match', lambda v: self._search_normalized_title(v, content_type, year)),
                ('content_type_match', lambda v: self._search_by_content_type(v, content_type)),
                ('tmdb_id_match', lambda v: self._search_by_tmdb_data(v, content_type, year)),
                ('contains_search', lambda v: self._search_contains(v, content_type)),
                ('partial_word_match', lambda v: self._search_partial_words(v, content_type)),
                ('fuzzy_distance_match', lambda v: self._search_fuzzy_distance(v, content_type)),
                ('broad_search', lambda v: self._search_broad(v)),
                ('super_fuzzy', lambda v: self._search_super_fuzzy(v, content_type, year))
            ]
            
            for strategy_name, search_func in strategies:
                if results and len(results) >= 5:
                    break
                    
                logger.debug(f"Trying strategy: {strategy_name}")
                
                for i, variation in enumerate(title_variations):
                    if results and len(results) >= 10:
                        break
                    
                    try:
                        strategy_results = search_func(variation)
                        if strategy_results:
                            results.extend(strategy_results)
                            logger.debug(f"Strategy '{strategy_name}' found {len(strategy_results)} results")
                    except Exception as e:
                        logger.debug(f"Strategy {strategy_name} error: {e}")
                        continue
            
            seen_ids = set()
            unique_results = []
            for result in results:
                if result.id not in seen_ids:
                    seen_ids.add(result.id)
                    unique_results.append(result)
            
            if unique_results:
                best_match = self._find_best_match(unique_results, title, year, content_type)
                logger.info(f"Found fuzzy match for '{slug}': {best_match.title} (ID: {best_match.id})")
                return best_match
            
            if self._should_fetch_from_external(slug, title, year):
                logger.info(f"No local match for '{slug}'. Trying external API fetch.")
                external_content = self._try_fetch_from_external(title, year, content_type)
                if external_content:
                    return external_content
            
            logger.debug(f"No match found for '{slug}'")
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy content search: {e}")
            return None
    
    def _search_by_slug_patterns(self, slug: str) -> List[Any]:
        try:
            results = []
            
            direct = self.Content.query.filter_by(slug=slug).first()
            if direct:
                return [direct]
            
            slug_without_year = re.sub(r'-\d{4}(?:-\d+)?$', '', slug)
            if slug_without_year != slug:
                results.extend(
                    self.Content.query.filter(
                        self.Content.slug.like(f"{slug_without_year}%")
                    ).limit(5).all()
                )
            
            slug_base = slug.split('-')[0] if '-' in slug else slug
            if len(slug_base) > 3:
                results.extend(
                    self.Content.query.filter(
                        self.Content.slug.like(f"{slug_base}%")
                    ).limit(5).all()
                )
            
            if len(slug) > 5:
                results.extend(
                    self.Content.query.filter(
                        self.Content.slug.contains(slug[:10])
                    ).limit(5).all()
                )
            
            return results
            
        except Exception as e:
            logger.debug(f"Slug pattern search error: {e}")
            return []
    
    def _search_by_tmdb_data(self, variation: str, content_type: str, year: Optional[int]) -> List[Any]:
        try:
            search_results = TMDBService.search_content(variation, 'multi' if not content_type else content_type)
            
            if not search_results or not search_results.get('results'):
                return []
            
            found_content = []
            for tmdb_item in search_results['results'][:3]:
                tmdb_id = tmdb_item.get('id')
                if tmdb_id:
                    existing = self.Content.query.filter_by(tmdb_id=tmdb_id).first()
                    if existing:
                        found_content.append(existing)
                    else:
                        item_type = 'movie' if 'title' in tmdb_item else 'tv'
                        saved_content = self.content_service.save_content_from_tmdb(tmdb_item, item_type)
                        if saved_content:
                            found_content.append(saved_content)
            
            return found_content
            
        except Exception as e:
            logger.debug(f"TMDB search error: {e}")
            return []
    
    def _search_normalized_title(self, variation: str, content_type: str, year: Optional[int]) -> List[Any]:
        try:
            normalized = self._normalize_title_for_search(variation)
            
            query = self.Content.query.filter(
                func.lower(
                    func.regexp_replace(
                        self.Content.title, 
                        '[^a-zA-Z0-9 ]', 
                        '', 
                        'g'
                    )
                ).like(f"%{normalized}%")
            )
            
            if content_type and content_type != 'multi':
                query = query.filter(self.Content.content_type == content_type)
            
            if year:
                query = query.filter(
                    func.extract('year', self.Content.release_date).between(year - 2, year + 2)
                )
            
            return query.order_by(self.Content.popularity.desc()).limit(10).all()
            
        except Exception as e:
            logger.debug(f"Normalized search error: {e}")
            return self._search_contains(variation, content_type)
    
    def _normalize_title_for_search(self, title: str) -> str:
        try:
            normalized = unicodedata.normalize('NFKD', title)
            normalized = normalized.encode('ascii', 'ignore').decode('ascii')
            
            normalized = re.sub(r'[^a-zA-Z0-9 ]', '', normalized)
            normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
            
            return normalized
        except Exception:
            return title.lower().strip()
    
    def _search_fuzzy_distance(self, variation: str, content_type: str) -> List[Any]:
        try:
            query = self.Content.query
            if content_type and content_type != 'multi':
                query = query.filter(self.Content.content_type == content_type)
            
            candidates = query.limit(1000).all()
            
            scored_results = []
            for candidate in candidates:
                if candidate.title:
                    similarity = self._calculate_similarity(candidate.title.lower(), variation.lower())
                    if similarity > 0.7:
                        scored_results.append((candidate, similarity))
            
            scored_results.sort(key=lambda x: x[1], reverse=True)
            return [result[0] for result in scored_results[:10]]
            
        except Exception as e:
            logger.debug(f"Fuzzy distance search error: {e}")
            return []
    
    def _try_fetch_from_external(self, title: str, year: Optional[int], content_type: str) -> Optional[Any]:
        try:
            if not TMDB_API_KEY:
                return None
            
            search_type = 'multi'
            if content_type == 'movie':
                search_type = 'movie'
            elif content_type == 'tv':
                search_type = 'tv'
            
            search_results = TMDBService.search_content(title, search_type, page=1)
            if search_results and search_results.get('results'):
                for result in search_results['results'][:5]:
                    if year:
                        result_date = result.get('release_date') or result.get('first_air_date')
                        if result_date:
                            try:
                                result_year = int(result_date[:4])
                                if abs(result_year - year) > 3:
                                    continue
                            except:
                                pass
                    
                    existing = self.Content.query.filter_by(tmdb_id=result['id']).first()
                    if existing:
                        return existing
                    
                    detected_type = 'movie' if 'title' in result else 'tv'
                    saved_content = self.content_service.save_content_from_tmdb(result, detected_type)
                    
                    if saved_content:
                        logger.info(f"Fetched and saved content from TMDB: {saved_content.title}")
                        return saved_content
            
            return None
            
        except Exception as e:
            logger.debug(f"External fetch error: {e}")
            return None
    
    def _generate_comprehensive_title_variations(self, title: str) -> List[str]:
        try:
            variations = []
            title_lower = title.lower().strip()
            
            if not title_lower or len(title_lower) < 1:
                return [title_lower] if title_lower else []
            
            variations.append(title_lower)
            
            special_char_replacements = {
                'œ': 'oe', 'æ': 'ae', 'ø': 'o', 'å': 'a', 'ö': 'o', 'ä': 'a', 'ü': 'u', 'ß': 'ss',
                'ñ': 'n', 'ç': 'c', 'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e', 'à': 'a', 'â': 'a',
                'ô': 'o', 'û': 'u', 'ù': 'u', 'ï': 'i', 'î': 'i', 'á': 'a', 'í': 'i', 'ó': 'o',
                'ú': 'u', 'ý': 'y', '&': 'and', '@': 'at'
            }
            
            normalized = title_lower
            for old, new in special_char_replacements.items():
                if old in normalized:
                    normalized = normalized.replace(old, new)
                    variations.append(normalized)
            
            patterns = [
                (r'^the\s+', ''),
                (r'^a\s+', ''),
                (r'^an\s+', ''),
                (r'\s*:\s*', ' '),
                (r'\s*-\s*', ' '),
                (r'\s*&\s*', ' and '),
                (r'\s+', ' '),
                (r'[^\w\s]', ''),
            ]
            
            for pattern, replacement in patterns:
                modified = re.sub(pattern, replacement, title_lower).strip()
                if modified and modified != title_lower:
                    variations.append(modified)
            
            if len(title_lower) <= 15:
                variations.extend([
                    f"the {title_lower}",
                    f"{title_lower} movie",
                    f"{title_lower} film",
                    f"{title_lower} series",
                    f"{title_lower} tv",
                    f"{title_lower} show"
                ])
            
            words = title_lower.split()
            if len(words) > 1:
                variations.append(''.join(words))
                variations.append('-'.join(words))
                variations.append('_'.join(words))
                
                for i in range(1, len(words)):
                    if len(words[:i]) > 0:
                        variations.append(' '.join(words[:i]))
                    if len(words[i:]) > 0:
                        variations.append(' '.join(words[i:]))
            
            seen = set()
            unique_variations = []
            for var in variations:
                clean_var = var.strip()
                if clean_var and clean_var not in seen and len(clean_var) >= 1:
                    seen.add(clean_var)
                    unique_variations.append(clean_var)
            
            return unique_variations[:40]
            
        except Exception as e:
            logger.error(f"Error generating title variations: {e}")
            return [title.lower().strip()] if title else []
    
    def _search_exact_with_year(self, variation: str, content_type: str, year: Optional[int]) -> List[Any]:
        if not year or not content_type or content_type == 'multi':
            return []
        
        try:
            return self.Content.query.filter(
                func.lower(self.Content.title) == variation,
                self.Content.content_type == content_type,
                func.extract('year', self.Content.release_date) == year
            ).limit(5).all()
        except Exception:
            return []
    
    def _search_flexible_year(self, variation: str, content_type: str, year: Optional[int]) -> List[Any]:
        if not year:
            return []
        
        try:
            query = self.Content.query.filter(
                func.lower(self.Content.title).like(f"%{variation}%"),
                func.extract('year', self.Content.release_date).between(year - 3, year + 3)
            )
            
            if content_type and content_type != 'multi':
                query = query.filter(self.Content.content_type == content_type)
                
            return query.order_by(self.Content.popularity.desc()).limit(10).all()
        except Exception:
            return []
    
    def _search_by_content_type(self, variation: str, content_type: str) -> List[Any]:
        try:
            query = self.Content.query.filter(
                func.lower(self.Content.title).like(f"%{variation}%")
            )
            
            if content_type and content_type != 'multi':
                query = query.filter(self.Content.content_type == content_type)
                
            return query.order_by(self.Content.popularity.desc()).limit(15).all()
        except Exception:
            return []
    
    def _search_contains(self, variation: str, content_type: str) -> List[Any]:
        try:
            results = []
            
            exact_results = self.Content.query.filter(
                func.lower(self.Content.title) == variation
            ).order_by(self.Content.popularity.desc()).limit(10).all()
            results.extend(exact_results)
            
            if len(results) < 10:
                contains_results = self.Content.query.filter(
                    func.lower(self.Content.title).like(f"%{variation}%"),
                    ~self.Content.id.in_([r.id for r in exact_results])
                ).order_by(self.Content.popularity.desc()).limit(15 - len(results)).all()
                results.extend(contains_results)
            
            return results
        except Exception:
            return []
    
    def _search_partial_words(self, variation: str, content_type: str) -> List[Any]:
        try:
            words = [w.strip() for w in variation.split() if len(w.strip()) >= 2]
            if not words:
                return []
            
            conditions = []
            for word in words[:3]:
                conditions.append(func.lower(self.Content.title).like(f"%{word}%"))
            
            if conditions:
                query = self.Content.query.filter(and_(*conditions))
                if content_type and content_type != 'multi':
                    query = query.filter(self.Content.content_type == content_type)
                return query.order_by(self.Content.popularity.desc()).limit(10).all()
            
            return []
        except Exception:
            return []
    
    def _search_broad(self, variation: str) -> List[Any]:
        try:
            if len(variation) < 2:
                return []
            
            return self.Content.query.filter(
                func.lower(self.Content.title).like(f"%{variation}%")
            ).order_by(self.Content.popularity.desc()).limit(20).all()
        except Exception:
            return []
    
    def _search_super_fuzzy(self, variation: str, content_type: str, year: Optional[int]) -> List[Any]:
        try:
            results = []
            
            if len(variation) >= 3:
                for i in range(max(2, len(variation) - 3), len(variation)):
                    partial = variation[:i]
                    if len(partial) >= 3:
                        partial_results = self.Content.query.filter(
                            func.lower(self.Content.title).like(f"{partial}%")
                        ).order_by(self.Content.popularity.desc()).limit(5).all()
                        results.extend(partial_results)
                
                for i in range(3, min(len(variation), 8)):
                    partial = variation[:i]
                    partial_results = self.Content.query.filter(
                        func.lower(self.Content.title).like(f"%{partial}%")
                    ).order_by(self.Content.popularity.desc()).limit(5).all()
                    results.extend(partial_results)
            
            return results[:15]
        except Exception:
            return []
    
    def _find_best_match(self, results: List[Any], title: str, year: Optional[int], content_type: str) -> Any:
        try:
            if not results:
                return None
            
            if len(results) == 1:
                return results[0]
            
            scored_results = []
            
            for result in results:
                score = 0
                
                try:
                    title_similarity = self._calculate_similarity(result.title.lower(), title.lower())
                    score += title_similarity * 60
                except Exception:
                    pass
                
                if year and result.release_date:
                    try:
                        year_diff = abs(result.release_date.year - year)
                        if year_diff == 0:
                            score += 40
                        elif year_diff <= 1:
                            score += 25
                        elif year_diff <= 3:
                            score += 15
                        elif year_diff <= 5:
                            score += 5
                    except Exception:
                        pass
                
                if content_type and content_type != 'multi' and result.content_type == content_type:
                    score += 20
                
                try:
                    if result.popularity:
                        score += min(result.popularity / 100, 15)
                except Exception:
                    pass
                
                try:
                    if result.rating:
                        score += min(result.rating, 10)
                except Exception:
                    pass
                
                try:
                    if result.vote_count:
                        score += min(result.vote_count / 1000, 10)
                except Exception:
                    pass
                
                scored_results.append((result, score))
            
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            best_result = scored_results[0][0]
            logger.info(f"Best match: {best_result.title} (Score: {scored_results[0][1]:.2f})")
            
            return best_result
            
        except Exception as e:
            logger.error(f"Error finding best match: {e}")
            return results[0] if results else None
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        try:
            clean_str1 = str1.lower().strip()
            clean_str2 = str2.lower().strip()
            
            if clean_str1 == clean_str2:
                return 1.0
            
            basic_sim = SequenceMatcher(None, clean_str1, clean_str2).ratio()
            
            words1 = set(clean_str1.split())
            words2 = set(clean_str2.split())
            
            if words1 and words2:
                word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                return 0.6 * basic_sim + 0.4 * word_overlap
            
            return basic_sim
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def _should_fetch_from_external(self, slug: str, title: str, year: Optional[int]) -> bool:
        try:
            current_year = datetime.now().year
            
            if year and (current_year - 2 <= year <= current_year + 3):
                return True
            
            if not year and any(str(y) in slug for y in range(current_year - 1, current_year + 3)):
                return True
            
            upcoming_keywords = ['sequel', 'part', '2024', '2025', '2026', 'remake', 'reboot']
            if any(keyword in slug.lower() for keyword in upcoming_keywords):
                return True
            
            if len(title) <= 20 and not year:
                return True
            
            return False
        except:
            return False
    
    def _with_app_context(self, app, func, *args, **kwargs):
        if app:
            with app.app_context():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _build_content_details(self, content: Any, user_id: Optional[int] = None) -> Dict:
        try:
            from flask import current_app
            app = current_app._get_current_object() if has_app_context() else None
            
            futures = {}
            
            with ThreadPoolExecutor(max_workers=6) as executor:
                if content.tmdb_id and TMDB_API_KEY:
                    futures['tmdb'] = executor.submit(self._fetch_tmdb_details, content.tmdb_id, content.content_type)
                if content.imdb_id and OMDB_API_KEY:
                    futures['omdb'] = executor.submit(self._fetch_omdb_details, content.imdb_id)
                
                if app:
                    futures['cast_crew'] = executor.submit(self._with_app_context, app, self._get_cast_crew, content.id)
                    futures['reviews'] = executor.submit(self._with_app_context, app, self._get_reviews, content.id, 10)
                    futures['similar'] = executor.submit(self._with_app_context, app, self._get_similar_content, content.id, 12)
                    futures['gallery'] = executor.submit(self._with_app_context, app, self._get_gallery, content.id)
                else:
                    futures['cast_crew'] = executor.submit(self._get_cast_crew, content.id)
                    futures['reviews'] = executor.submit(self._get_reviews, content.id, 10)
                    futures['similar'] = executor.submit(self._get_similar_content, content.id, 12)
                    futures['gallery'] = executor.submit(self._get_gallery, content.id)
                
                futures['trailer'] = executor.submit(self._get_trailer, content.title, content.content_type)
            
            tmdb_data = {}
            omdb_data = {}
            cast_crew = {'cast': [], 'crew': {'directors': [], 'writers': [], 'producers': []}}
            reviews = []
            similar = []
            gallery = {'posters': [], 'backdrops': [], 'stills': []}
            trailer = None
            
            try:
                if 'tmdb' in futures:
                    tmdb_data = futures['tmdb'].result(timeout=12) or {}
            except Exception as e:
                logger.warning(f"TMDB fetch error/timeout: {e}")
            
            try:
                if 'omdb' in futures:
                    omdb_data = futures['omdb'].result(timeout=10) or {}
            except Exception as e:
                logger.warning(f"OMDB fetch error/timeout: {e}")
            
            try:
                cast_crew = futures['cast_crew'].result(timeout=15) or cast_crew
            except Exception as e:
                logger.warning(f"Cast/crew fetch error/timeout: {e}")
            
            try:
                reviews = futures['reviews'].result(timeout=5) or []
            except Exception as e:
                logger.warning(f"Reviews fetch error/timeout: {e}")
            
            try:
                similar = futures['similar'].result(timeout=10) or []
            except Exception as e:
                logger.warning(f"Similar content fetch error/timeout: {e}")
            
            try:
                gallery = futures['gallery'].result(timeout=10) or gallery
            except Exception as e:
                logger.warning(f"Gallery fetch error/timeout: {e}")
            
            try:
                trailer = futures['trailer'].result(timeout=10)
            except Exception as e:
                logger.warning(f"Trailer fetch error/timeout: {e}")
            
            synopsis = self._build_synopsis(content, tmdb_data, omdb_data)
            ratings = self._build_ratings(content, tmdb_data, omdb_data)
            metadata = self._build_metadata(content, tmdb_data, omdb_data)
            streaming_info = self._get_streaming_info(content, tmdb_data)
            
            seasons_episodes = None
            if content.content_type in ['tv', 'anime']:
                seasons_episodes = self._get_seasons_episodes(content, tmdb_data)
            
            details = {
                'slug': content.slug,
                'id': content.id,
                'title': content.title,
                'original_title': content.original_title,
                'overview': content.overview,
                'content_type': content.content_type,
                'poster_url': self._format_image_url(content.poster_path, 'poster'),
                'backdrop_url': self._format_image_url(content.backdrop_path or tmdb_data.get('backdrop_path'), 'backdrop'),
                'trailer': trailer,
                'synopsis': synopsis,
                'cast_crew': cast_crew,
                'ratings': ratings,
                'metadata': metadata,
                'more_like_this': similar,
                'reviews': reviews,
                'gallery': gallery,
                'streaming_info': streaming_info,
                'seasons_episodes': seasons_episodes
            }
            
            if user_id:
                if app:
                    with app.app_context():
                        details['user_data'] = self._get_user_data(content.id, user_id)
                else:
                    details['user_data'] = self._get_user_data(content.id, user_id)
            
            return details
            
        except Exception as e:
            logger.error(f"Error building content details: {e}")
            return self._get_minimal_details(content)

    def _get_trailer(self, title: str, content_type: str) -> Optional[Dict]:
        try:
            if not YOUTUBE_API_KEY:
                logger.warning("YOUTUBE_API_KEY not available")
                return None
            
            # Improved search queries
            search_queries = []
            
            if content_type == 'anime':
                search_queries = [
                    f"{title} official trailer",
                    f"{title} anime trailer",
                    f"{title} PV trailer",
                    f"{title} anime PV",
                    f"{title} official PV"
                ]
            elif content_type == 'tv':
                search_queries = [
                    f"{title} official trailer",
                    f"{title} series trailer", 
                    f"{title} season 1 trailer",
                    f"{title} TV series trailer",
                    f"{title} teaser trailer"
                ]
            else:  # movie
                search_queries = [
                    f"{title} official trailer",
                    f"{title} movie trailer",
                    f"{title} film trailer",
                    f"{title} teaser trailer",
                    f"{title} final trailer"
                ]
            
            for search_query in search_queries:
                try:
                    url = f"{YOUTUBE_BASE_URL}/search"
                    params = {
                        'key': YOUTUBE_API_KEY,
                        'q': search_query,
                        'part': 'snippet',
                        'type': 'video',
                        'maxResults': 5,
                        'order': 'relevance',
                        'videoDefinition': 'any',
                        'videoDuration': 'any'
                    }
                    
                    response = self.session.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('items'):
                            for video in data['items']:
                                video_title = video['snippet']['title'].lower()
                                
                                # Filter for actual trailers
                                if any(keyword in video_title for keyword in ['trailer', 'teaser', 'preview', 'tv spot', 'official']):
                                    return {
                                        'youtube_id': video['id']['videoId'],
                                        'title': video['snippet']['title'],
                                        'description': video['snippet']['description'][:200] + '...' if len(video['snippet']['description']) > 200 else video['snippet']['description'],
                                        'thumbnail': video['snippet']['thumbnails']['high']['url'],
                                        'embed_url': f"https://www.youtube.com/embed/{video['id']['videoId']}",
                                        'watch_url': f"https://www.youtube.com/watch?v={video['id']['videoId']}",
                                        'channel_title': video['snippet']['channelTitle'],
                                        'published_at': video['snippet']['publishedAt']
                                    }
                    
                except Exception as e:
                    logger.warning(f"Error with search query '{search_query}': {e}")
                    continue
            
            logger.info(f"No trailer found for '{title}' ({content_type})")
            return None
            
        except Exception as e:
            logger.error(f"Error getting trailer: {e}")
            return None

    @ensure_app_context
    def _get_cast_crew(self, content_id: int) -> Dict:
        try:
            cast_crew = {
                'cast': [],
                'crew': {
                    'directors': [],
                    'writers': [],
                    'producers': []
                }
            }
            
            content = self.Content.query.get(content_id) if content_id else None
            if not content:
                logger.warning(f"Content not found for ID: {content_id}")
                return cast_crew
            
            if self.ContentPerson and self.Person:
                cast_crew = self._get_cast_crew_from_db(content_id)
                
                total_cast_crew = len(cast_crew['cast']) + sum(len(crew_list) for crew_list in cast_crew['crew'].values())
                if total_cast_crew >= 5:
                    logger.info(f"Found {total_cast_crew} cast/crew members in database for content {content_id}")
                    return cast_crew
            
            if content.tmdb_id and TMDB_API_KEY:
                logger.info(f"Fetching comprehensive cast/crew from TMDB for content {content_id}")
                cast_crew = self._fetch_and_save_all_cast_crew(content)
            
            return cast_crew
            
        except Exception as e:
            logger.error(f"Error getting cast/crew: {e}")
            return {'cast': [], 'crew': {'directors': [], 'writers': [], 'producers': []}}

    def _get_cast_crew_from_db(self, content_id: int) -> Dict:
        cast_crew = {
            'cast': [],
            'crew': {
                'directors': [],
                'writers': [],
                'producers': []
            }
        }
        
        try:
            cast_entries = self.db.session.query(
                self.ContentPerson, self.Person
            ).join(
                self.Person
            ).filter(
                self.ContentPerson.content_id == content_id,
                self.ContentPerson.role_type == 'cast'
            ).order_by(
                self.ContentPerson.order.asc()
            ).all()
            
            for cp, person in cast_entries:
                if not person.slug:
                    try:
                        SlugManager.update_content_slug(self.db, person)
                    except Exception:
                        person.slug = f"person-{person.id}"
                
                cast_crew['cast'].append({
                    'id': person.id,
                    'name': person.name,
                    'character': cp.character,
                    'profile_path': self._format_image_url(person.profile_path, 'profile'),
                    'slug': person.slug,
                    'popularity': person.popularity or 0,
                    'order': cp.order or 999
                })
            
            crew_entries = self.db.session.query(
                self.ContentPerson, self.Person
            ).join(
                self.Person
            ).filter(
                self.ContentPerson.content_id == content_id,
                self.ContentPerson.role_type == 'crew'
            ).all()
            
            for cp, person in crew_entries:
                if not person.slug:
                    try:
                        SlugManager.update_content_slug(self.db, person)
                    except Exception:
                        person.slug = f"person-{person.id}"
                
                crew_data = {
                    'id': person.id,
                    'name': person.name,
                    'job': cp.job,
                    'department': cp.department,
                    'profile_path': self._format_image_url(person.profile_path, 'profile'),
                    'slug': person.slug
                }
                
                if cp.department == 'Directing' or cp.job == 'Director':
                    cast_crew['crew']['directors'].append(crew_data)
                elif cp.department == 'Writing' or cp.job in ['Writer', 'Screenplay', 'Story', 'Novel', 'Characters']:
                    cast_crew['crew']['writers'].append(crew_data)
                elif cp.department == 'Production' or 'Producer' in (cp.job or ''):
                    cast_crew['crew']['producers'].append(crew_data)
            
            return cast_crew
            
        except Exception as e:
            logger.error(f"Error getting cast/crew from database: {e}")
            return cast_crew

    def _fetch_and_save_all_cast_crew(self, content) -> Dict:
        cast_crew = {
            'cast': [],
            'crew': {
                'directors': [],
                'writers': [],
                'producers': []
            }
        }
        
        try:
            if not TMDB_API_KEY:
                logger.warning("TMDB_API_KEY not available for cast/crew fetch")
                return cast_crew
            
            endpoint = 'movie' if content.content_type == 'movie' else 'tv'
            url = f"{TMDB_BASE_URL}/{endpoint}/{content.tmdb_id}/credits"
            
            params = {
                'api_key': TMDB_API_KEY
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                cast_data = data.get('cast', [])
                logger.info(f"Processing {len(cast_data)} cast members for {content.title}")
                
                for i, cast_member in enumerate(cast_data):
                    try:
                        person = self._get_or_create_person(cast_member)
                        if person:
                            content_person = self._get_or_create_content_person(
                                content.id, person.id, 'cast', 
                                character=cast_member.get('character'),
                                order=i
                            )
                            
                            cast_crew['cast'].append({
                                'id': person.id,
                                'name': person.name,
                                'character': cast_member.get('character', ''),
                                'profile_path': self._format_image_url(person.profile_path, 'profile'),
                                'slug': person.slug,
                                'popularity': person.popularity or 0,
                                'order': i
                            })
                    except Exception as e:
                        logger.warning(f"Error processing cast member: {e}")
                        continue
                
                crew_data = data.get('crew', [])
                logger.info(f"Processing {len(crew_data)} crew members for {content.title}")
                
                directors = []
                writers = []
                producers = []
                
                for crew_member in crew_data:
                    try:
                        person = self._get_or_create_person(crew_member)
                        if person:
                            content_person = self._get_or_create_content_person(
                                content.id, person.id, 'crew',
                                job=crew_member.get('job'),
                                department=crew_member.get('department')
                            )
                            
                            crew_data_item = {
                                'id': person.id,
                                'name': person.name,
                                'job': crew_member.get('job', ''),
                                'department': crew_member.get('department', ''),
                                'profile_path': self._format_image_url(person.profile_path, 'profile'),
                                'slug': person.slug
                            }
                            
                            job = crew_member.get('job', '').lower()
                            department = crew_member.get('department', '').lower()
                            
                            if department == 'directing' or job == 'director':
                                directors.append(crew_data_item)
                            elif department == 'writing' or job in ['writer', 'screenplay', 'story', 'novel', 'characters']:
                                writers.append(crew_data_item)
                            elif department == 'production' or 'producer' in job:
                                producers.append(crew_data_item)
                                
                    except Exception as e:
                        logger.warning(f"Error processing crew member: {e}")
                        continue
                
                cast_crew['crew']['directors'] = directors
                cast_crew['crew']['writers'] = writers
                cast_crew['crew']['producers'] = producers
                
                self.db.session.commit()
                
                total_saved = len(cast_crew['cast']) + len(directors) + len(writers) + len(producers)
                logger.info(f"Successfully saved {total_saved} cast/crew members for content {content.id} ({content.title})")
                
            else:
                logger.warning(f"TMDB credits API returned {response.status_code} for {content.title}")
                
        except Exception as e:
            logger.error(f"Error fetching cast/crew from TMDB: {e}")
            self.db.session.rollback()
        
        return cast_crew

    def _get_or_create_person(self, person_data) -> Any:
        try:
            tmdb_id = person_data.get('id')
            if not tmdb_id:
                return None
            
            person = self.Person.query.filter_by(tmdb_id=tmdb_id).first()
            
            if not person:
                name = person_data.get('name', 'Unknown')
                slug = SlugManager.generate_unique_slug(
                    self.db, self.Person, name, content_type='person', tmdb_id=tmdb_id
                )
                
                person = self.Person(
                    slug=slug,
                    tmdb_id=tmdb_id,
                    name=name,
                    profile_path=person_data.get('profile_path'),
                    popularity=person_data.get('popularity', 0),
                    known_for_department=person_data.get('known_for_department'),
                    gender=person_data.get('gender')
                )
                
                self.db.session.add(person)
                self.db.session.flush()
            else:
                if not person.slug or person.slug.startswith('person-'):
                    person.slug = SlugManager.generate_unique_slug(
                        self.db, self.Person, person.name, content_type='person',
                        existing_id=person.id, tmdb_id=tmdb_id
                    )
                
                if not person.popularity and person_data.get('popularity'):
                    person.popularity = person_data.get('popularity')
                if not person.known_for_department and person_data.get('known_for_department'):
                    person.known_for_department = person_data.get('known_for_department')
            
            return person
            
        except Exception as e:
            logger.error(f"Error creating/getting person: {e}")
            return None

    def _get_or_create_content_person(self, content_id, person_id, role_type, 
                                     character=None, job=None, department=None, order=None):
        try:
            existing = None
            
            if role_type == 'cast' and character:
                existing = self.ContentPerson.query.filter_by(
                    content_id=content_id,
                    person_id=person_id,
                    role_type=role_type,
                    character=character
                ).first()
            elif role_type == 'crew' and job:
                existing = self.ContentPerson.query.filter_by(
                    content_id=content_id,
                    person_id=person_id,
                    role_type=role_type,
                    job=job
                ).first()
            else:
                existing = self.ContentPerson.query.filter_by(
                    content_id=content_id,
                    person_id=person_id,
                    role_type=role_type
                ).first()
            
            if not existing:
                content_person = self.ContentPerson(
                    content_id=content_id,
                    person_id=person_id,
                    role_type=role_type,
                    character=character,
                    job=job,
                    department=department,
                    order=order
                )
                
                self.db.session.add(content_person)
                return content_person
            
            return existing
            
        except Exception as e:
            logger.error(f"Error creating content-person relationship: {e}")
            return None

    def _get_minimal_details(self, content: Any) -> Dict:
        try:
            return {
                'slug': getattr(content, 'slug', ''),
                'id': getattr(content, 'id', 0),
                'title': getattr(content, 'title', 'Unknown'),
                'original_title': getattr(content, 'original_title', None),
                'overview': getattr(content, 'overview', ''),
                'content_type': getattr(content, 'content_type', 'movie'),
                'poster_url': self._format_image_url(getattr(content, 'poster_path', None), 'poster'),
                'backdrop_url': self._format_image_url(getattr(content, 'backdrop_path', None), 'backdrop'),
                'trailer': None,
                'synopsis': {
                    'overview': getattr(content, 'overview', ''),
                    'plot': '',
                    'tagline': '',
                    'content_warnings': [],
                    'themes': [],
                    'keywords': []
                },
                'cast_crew': {'cast': [], 'crew': {'directors': [], 'writers': [], 'producers': []}},
                'ratings': {
                    'tmdb': {'score': getattr(content, 'rating', 0), 'count': getattr(content, 'vote_count', 0)},
                    'imdb': {'score': 0, 'votes': 'N/A'},
                    'composite_score': getattr(content, 'rating', 0),
                    'critics': [],
                    'audience_score': None
                },
                'metadata': {
                    'genres': [],
                    'release_date': getattr(content, 'release_date', None),
                    'runtime': getattr(content, 'runtime', None),
                    'status': 'Released',
                    'original_language': None,
                    'spoken_languages': [],
                    'production_companies': [],
                    'production_countries': [],
                    'budget': 0,
                    'revenue': 0,
                    'advisories': {},
                    'certifications': {},
                    'awards': '',
                    'popularity': getattr(content, 'popularity', 0),
                    'is_critics_choice': getattr(content, 'is_critics_choice', False),
                    'is_trending': getattr(content, 'is_trending', False),
                    'is_new_release': getattr(content, 'is_new_release', False)
                },
                'more_like_this': [],
                'reviews': [],
                'gallery': {'posters': [], 'backdrops': [], 'stills': []},
                'streaming_info': None,
                'seasons_episodes': None
            }
        except Exception as e:
            logger.error(f"Error creating minimal details: {e}")
            return {'error': 'Failed to create content details'}
    
    def _fetch_tmdb_details(self, tmdb_id: int, content_type: str) -> Dict:
        try:
            if not TMDB_API_KEY:
                logger.warning("TMDB_API_KEY not available")
                return {}
                
            endpoint = 'movie' if content_type == 'movie' else 'tv'
            url = f"{TMDB_BASE_URL}/{endpoint}/{tmdb_id}"
            
            params = {
                'api_key': TMDB_API_KEY,
                'append_to_response': 'videos,images,credits,similar,recommendations,reviews,external_ids,watch/providers,content_ratings,release_dates,keywords'
            }
            
            response = self.session.get(url, params=params, timeout=12)
            
            if response.status_code == 200:
                return response.json()
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching TMDB details: {e}")
            return {}
    
    def _fetch_omdb_details(self, imdb_id: str) -> Dict:
        try:
            if not OMDB_API_KEY:
                logger.warning("OMDB_API_KEY not available")
                return {}
                
            params = {
                'apikey': OMDB_API_KEY,
                'i': imdb_id,
                'plot': 'full'
            }
            
            response = self.session.get(OMDB_BASE_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching OMDB details: {e}")
            return {}
    
    def _build_synopsis(self, content: Any, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        try:
            synopsis = {
                'overview': content.overview or tmdb_data.get('overview', ''),
                'plot': omdb_data.get('Plot', content.overview or ''),
                'tagline': tmdb_data.get('tagline', ''),
                'content_warnings': [],
                'themes': [],
                'keywords': []
            }
            
            if tmdb_data.get('content_ratings'):
                ratings = tmdb_data['content_ratings'].get('results', [])
                us_rating = next((r for r in ratings if r['iso_3166_1'] == 'US'), None)
                if us_rating:
                    synopsis['content_warnings'].append({
                        'rating': us_rating.get('rating'),
                        'descriptors': us_rating.get('descriptors', [])
                    })
            
            if tmdb_data.get('keywords'):
                keywords = tmdb_data['keywords'].get('keywords', []) or tmdb_data['keywords'].get('results', [])
                synopsis['keywords'] = [kw['name'] for kw in keywords[:15]]
            
            return synopsis
        except Exception as e:
            logger.error(f"Error building synopsis: {e}")
            return {
                'overview': getattr(content, 'overview', ''),
                'plot': '',
                'tagline': '',
                'content_warnings': [],
                'themes': [],
                'keywords': []
            }
    
    def _build_ratings(self, content: Any, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        try:
            ratings = {
                'tmdb': {
                    'score': content.rating or tmdb_data.get('vote_average', 0),
                    'count': content.vote_count or tmdb_data.get('vote_count', 0)
                },
                'imdb': {
                    'score': 0,
                    'votes': 'N/A'
                },
                'composite_score': 0,
                'critics': [],
                'audience_score': None
            }
            
            try:
                imdb_rating = omdb_data.get('imdbRating', 'N/A')
                if imdb_rating and imdb_rating != 'N/A':
                    ratings['imdb']['score'] = float(imdb_rating)
                    ratings['imdb']['votes'] = omdb_data.get('imdbVotes', 'N/A')
            except (ValueError, TypeError):
                pass
            
            if omdb_data.get('Ratings'):
                for rating in omdb_data['Ratings']:
                    ratings['critics'].append({
                        'source': rating['Source'],
                        'value': rating['Value']
                    })
            
            scores = []
            if ratings['tmdb']['score'] > 0:
                scores.append(ratings['tmdb']['score'])
            if ratings['imdb']['score'] > 0:
                scores.append(ratings['imdb']['score'])
            
            if scores:
                ratings['composite_score'] = round(sum(scores) / len(scores), 1)
            
            return ratings
        except Exception as e:
            logger.error(f"Error building ratings: {e}")
            return {
                'tmdb': {'score': 0, 'count': 0},
                'imdb': {'score': 0, 'votes': 'N/A'},
                'composite_score': 0,
                'critics': [],
                'audience_score': None
            }
    
    def _build_metadata(self, content: Any, tmdb_data: Dict, omdb_data: Dict) -> Dict:
        try:
            try:
                genres = json.loads(content.genres) if content.genres else []
            except (json.JSONDecodeError, TypeError):
                genres = []
            
            metadata = {
                'genres': genres,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'runtime': content.runtime or tmdb_data.get('runtime'),
                'status': tmdb_data.get('status', 'Released'),
                'original_language': tmdb_data.get('original_language'),
                'spoken_languages': [],
                'production_companies': [],
                'production_countries': [],
                'budget': tmdb_data.get('budget', 0),
                'revenue': tmdb_data.get('revenue', 0),
                'advisories': {},
                'certifications': {},
                'awards': omdb_data.get('Awards', ''),
                'popularity': content.popularity or 0,
                'is_critics_choice': getattr(content, 'is_critics_choice', False),
                'is_trending': getattr(content, 'is_trending', False),
                'is_new_release': getattr(content, 'is_new_release', False)
            }
            
            if tmdb_data.get('spoken_languages'):
                metadata['spoken_languages'] = [
                    {'code': lang['iso_639_1'], 'name': lang['name']} 
                    for lang in tmdb_data['spoken_languages']
                ]
            
            if tmdb_data.get('production_companies'):
                metadata['production_companies'] = [
                    {
                        'id': company['id'],
                        'name': company['name'],
                        'logo': self._format_image_url(company.get('logo_path'), 'logo') if company.get('logo_path') else None
                    }
                    for company in tmdb_data['production_companies'][:5]
                ]
            
            if tmdb_data.get('release_dates'):
                for country in tmdb_data['release_dates'].get('results', []):
                    if country['iso_3166_1'] in ['US', 'GB', 'IN']:
                        for release in country['release_dates']:
                            if release.get('certification'):
                                metadata['certifications'][country['iso_3166_1']] = release['certification']
                                break
            
            return metadata
        except Exception as e:
            logger.error(f"Error building metadata: {e}")
            return {
                'genres': [],
                'release_date': None,
                'runtime': None,
                'status': 'Released',
                'original_language': None,
                'spoken_languages': [],
                'production_companies': [],
                'production_countries': [],
                'budget': 0,
                'revenue': 0,
                'advisories': {},
                'certifications': {},
                'awards': '',
                'popularity': 0,
                'is_critics_choice': False,
                'is_trending': False,
                'is_new_release': False
            }
    
    @ensure_app_context
    def _get_reviews(self, content_id: int, limit: int = 10) -> List[Dict]:
        try:
            reviews = []
            
            if self.Review and self.User:
                review_entries = self.db.session.query(
                    self.Review, self.User
                ).join(
                    self.User
                ).filter(
                    self.Review.content_id == content_id,
                    self.Review.is_approved == True
                ).order_by(
                    self.Review.helpful_count.desc(),
                    self.Review.created_at.desc()
                ).limit(limit).all()
                
                for review, user in review_entries:
                    reviews.append({
                        'id': review.id,
                        'user': {
                            'id': user.id,
                            'username': user.username,
                            'avatar': user.avatar_url
                        },
                        'rating': review.rating,
                        'title': review.title,
                        'review_text': review.review_text,
                        'has_spoilers': review.has_spoilers,
                        'helpful_count': review.helpful_count,
                        'created_at': review.created_at.isoformat(),
                        'updated_at': review.updated_at.isoformat() if review.updated_at else None
                    })
            
            return reviews
            
        except Exception as e:
            logger.error(f"Error getting reviews: {e}")
            return []
    
    @ensure_app_context
    def _get_similar_content(self, content_id: int, limit: int = 12) -> List[Dict]:
        try:
            similar = []
            
            content = self.Content.query.get(content_id)
            if not content:
                return []
            
            try:
                genres = json.loads(content.genres) if content.genres else []
            except (json.JSONDecodeError, TypeError):
                genres = []
            
            query = self.Content.query.filter(
                self.Content.id != content_id,
                self.Content.content_type == content.content_type
            )
            
            if genres:
                primary_genre = genres[0]
                query = query.filter(self.Content.genres.contains(primary_genre))
            
            similar_content = query.order_by(
                self.Content.rating.desc(),
                self.Content.popularity.desc()
            ).limit(limit).all()
            
            for item in similar_content:
                if not item.slug or item.slug.startswith('content-'):
                    try:
                        SlugManager.update_content_slug(self.db, item, force_update=True)
                    except Exception:
                        item.slug = f"content-{item.id}"
                
                try:
                    item_genres = json.loads(item.genres) if item.genres else []
                except (json.JSONDecodeError, TypeError):
                    item_genres = []
                
                similar.append({
                    'id': item.id,
                    'slug': item.slug,
                    'title': item.title,
                    'content_type': item.content_type,
                    'poster_path': self._format_image_url(item.poster_path, 'poster'),
                    'poster_url': self._format_image_url(item.poster_path, 'poster'),
                    'rating': item.rating,
                    'release_date': item.release_date.isoformat() if item.release_date else None,
                    'year': item.release_date.year if item.release_date else None,
                    'genres': item_genres,
                    'runtime': item.runtime
                })
            
            return similar
            
        except Exception as e:
            logger.error(f"Error getting similar content: {e}")
            return []
    
    def _get_streaming_info(self, content: Any, tmdb_data: Dict) -> Optional[Dict]:
        try:
            streaming = {
                'rent': [],
                'buy': [],
                'stream': [],
                'free': []
            }
            
            if tmdb_data.get('watch/providers'):
                providers = tmdb_data['watch/providers'].get('results', {})
                region_data = providers.get('US', {})
                
                if region_data.get('flatrate'):
                    for provider in region_data['flatrate'][:8]:
                        streaming['stream'].append({
                            'provider_id': provider['provider_id'],
                            'provider_name': provider['provider_name'],
                            'logo_path': self._format_image_url(provider.get('logo_path'), 'logo')
                        })
                
                if region_data.get('rent'):
                    for provider in region_data['rent'][:8]:
                        streaming['rent'].append({
                            'provider_id': provider['provider_id'],
                            'provider_name': provider['provider_name'],
                            'logo_path': self._format_image_url(provider.get('logo_path'), 'logo')
                        })
                
                if region_data.get('buy'):
                    for provider in region_data['buy'][:8]:
                        streaming['buy'].append({
                            'provider_id': provider['provider_id'],
                            'provider_name': provider['provider_name'],
                            'logo_path': self._format_image_url(provider.get('logo_path'), 'logo')
                        })
                
                if region_data.get('free'):
                    for provider in region_data['free'][:8]:
                        streaming['free'].append({
                            'provider_id': provider['provider_id'],
                            'provider_name': provider['provider_name'],
                            'logo_path': self._format_image_url(provider.get('logo_path'), 'logo')
                        })
            
            if not any([streaming['rent'], streaming['buy'], streaming['stream'], streaming['free']]):
                return None
            
            return streaming
            
        except Exception as e:
            logger.error(f"Error getting streaming info: {e}")
            return None
    
    def _get_seasons_episodes(self, content: Any, tmdb_data: Dict) -> Optional[Dict]:
        try:
            if content.content_type not in ['tv', 'anime']:
                return None
            
            seasons_data = {
                'season_count': tmdb_data.get('number_of_seasons', 0),
                'episode_count': tmdb_data.get('number_of_episodes', 0),
                'seasons': []
            }
            
            if tmdb_data.get('seasons'):
                for season in tmdb_data['seasons'][:15]:
                    if season['season_number'] == 0 and len(tmdb_data['seasons']) > 1:
                        continue
                    
                    seasons_data['seasons'].append({
                        'season_number': season['season_number'],
                        'name': season['name'],
                        'episode_count': season['episode_count'],
                        'air_date': season.get('air_date'),
                        'overview': season.get('overview'),
                        'poster_path': self._format_image_url(season.get('poster_path'), 'poster')
                    })
            
            if tmdb_data.get('next_episode_to_air'):
                next_ep = tmdb_data['next_episode_to_air']
                seasons_data['next_episode'] = {
                    'name': next_ep['name'],
                    'season_number': next_ep['season_number'],
                    'episode_number': next_ep['episode_number'],
                    'air_date': next_ep['air_date'],
                    'overview': next_ep.get('overview')
                }
            
            if tmdb_data.get('last_episode_to_air'):
                last_ep = tmdb_data['last_episode_to_air']
                seasons_data['last_episode'] = {
                    'name': last_ep['name'],
                    'season_number': last_ep['season_number'],
                    'episode_number': last_ep['episode_number'],
                    'air_date': last_ep['air_date']
                }
            
            return seasons_data
            
        except Exception as e:
            logger.error(f"Error getting seasons/episodes: {e}")
            return None
    
    @ensure_app_context
    def _get_user_data(self, content_id: int, user_id: int) -> Dict:
        try:
            user_data = {
                'in_watchlist': False,
                'in_favorites': False,
                'user_rating': None,
                'watch_status': None
            }
            
            if not self.UserInteraction:
                return user_data
            
            interactions = self.UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=content_id
            ).all()
            
            for interaction in interactions:
                if interaction.interaction_type == 'watchlist':
                    user_data['in_watchlist'] = True
                elif interaction.interaction_type == 'favorite':
                    user_data['in_favorites'] = True
                elif interaction.interaction_type == 'rating':
                    user_data['user_rating'] = interaction.rating
                elif interaction.interaction_type == 'watch_status' and interaction.interaction_metadata:
                    user_data['watch_status'] = interaction.interaction_metadata.get('status')
            
            return user_data
            
        except Exception as e:
            logger.error(f"Error getting user data: {e}")
            return {
                'in_watchlist': False,
                'in_favorites': False,
                'user_rating': None,
                'watch_status': None
            }
    
    def _add_user_data(self, details: Dict, user_id: int) -> Dict:
        try:
            if details and 'id' in details:
                details['user_data'] = self._get_user_data(details['id'], user_id)
            return details
        except Exception as e:
            logger.error(f"Error adding user data: {e}")
            return details
    
    def _format_image_url(self, path: str, image_type: str = 'poster') -> Optional[str]:
        if not path:
            return None
        
        if path.startswith('http'):
            return path
        
        try:
            if image_type == 'poster':
                return f"{POSTER_BASE}{path}"
            elif image_type == 'backdrop':
                return f"{BACKDROP_BASE}{path}"
            elif image_type == 'profile':
                return f"{PROFILE_BASE}{path}"
            elif image_type == 'still':
                return f"{STILL_BASE}{path}"
            elif image_type == 'thumbnail':
                return f"https://image.tmdb.org/t/p/w185{path}"
            elif image_type == 'logo':
                return f"https://image.tmdb.org/t/p/w92{path}"
            else:
                return f"{POSTER_BASE}{path}"
        except Exception as e:
            logger.error(f"Error formatting image URL: {e}")
            return None
    
    def _get_gallery(self, content_id: int) -> Dict:
        try:
            gallery = {
                'posters': [],
                'backdrops': [],
                'stills': []
            }
            
            content = None
            try:
                if has_app_context():
                    content = self.Content.query.get(content_id)
                else:
                    return gallery
            except Exception as e:
                logger.warning(f"Error getting content for gallery: {e}")
                return gallery
            
            if content and content.tmdb_id and TMDB_API_KEY:
                endpoint = 'movie' if content.content_type == 'movie' else 'tv'
                url = f"{TMDB_BASE_URL}/{endpoint}/{content.tmdb_id}/images"
                
                params = {
                    'api_key': TMDB_API_KEY
                }
                
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for img in data.get('posters', [])[:15]:
                        gallery['posters'].append({
                            'url': self._format_image_url(img['file_path'], 'poster'),
                            'thumbnail': self._format_image_url(img['file_path'], 'thumbnail'),
                            'aspect_ratio': img.get('aspect_ratio'),
                            'width': img.get('width'),
                            'height': img.get('height')
                        })
                    
                    for img in data.get('backdrops', [])[:15]:
                        gallery['backdrops'].append({
                            'url': self._format_image_url(img['file_path'], 'backdrop'),
                            'thumbnail': self._format_image_url(img['file_path'], 'still'),
                            'aspect_ratio': img.get('aspect_ratio'),
                            'width': img.get('width'),
                            'height': img.get('height')
                        })
                    
                    if content.content_type == 'tv' and data.get('stills'):
                        for img in data.get('stills', [])[:15]:
                            gallery['stills'].append({
                                'url': self._format_image_url(img['file_path'], 'still'),
                                'thumbnail': self._format_image_url(img['file_path'], 'thumbnail'),
                                'aspect_ratio': img.get('aspect_ratio'),
                                'width': img.get('width'),
                                'height': img.get('height')
                            })
            
            if not gallery['posters'] and content and content.poster_path:
                gallery['posters'].append({
                    'url': self._format_image_url(content.poster_path, 'poster'),
                    'thumbnail': self._format_image_url(content.poster_path, 'thumbnail')
                })
            
            return gallery
            
        except Exception as e:
            logger.error(f"Error getting gallery: {e}")
            return {'posters': [], 'backdrops': [], 'stills': []}

    def add_review(self, content_id: int, user_id: int, review_data: Dict) -> Dict:
        try:
            if not self.Review:
                return {
                    'success': False,
                    'message': 'Review system not available'
                }
            
            existing_review = self.Review.query.filter_by(
                content_id=content_id,
                user_id=user_id
            ).first()
            
            if existing_review:
                return {
                    'success': False,
                    'message': 'You have already reviewed this content'
                }
            
            user = self.User.query.get(user_id) if self.User else None
            should_auto_approve = self._should_auto_approve_review(user, review_data)
            
            review = self.Review(
                content_id=content_id,
                user_id=user_id,
                rating=review_data.get('rating'),
                title=review_data.get('title'),
                review_text=review_data.get('review_text'),
                has_spoilers=review_data.get('has_spoilers', False),
                is_approved=should_auto_approve
            )
            
            self.db.session.add(review)
            self.db.session.commit()
            
            content = self.Content.query.get(content_id)
            if content and self.cache:
                try:
                    cache_key = f"details:slug:{content.slug}"
                    self.cache.delete(cache_key)
                except Exception as e:
                    logger.warning(f"Cache invalidation error: {e}")
            
            message = 'Review published successfully' if should_auto_approve else 'Review submitted for moderation'
            
            return {
                'success': True,
                'review_id': review.id,
                'message': message,
                'auto_approved': should_auto_approve
            }
            
        except Exception as e:
            logger.error(f"Error adding review: {e}")
            self.db.session.rollback()
            return {
                'success': False,
                'message': 'Failed to add review'
            }

    def _should_auto_approve_review(self, user, review_data: Dict) -> bool:
        try:
            if user and getattr(user, 'is_admin', False):
                return True
            
            review_text = review_data.get('review_text', '')
            rating = review_data.get('rating', 0)
            
            if len(review_text.strip()) >= 20 and 1 <= rating <= 10:
                if user and self.Review:
                    approved_reviews_count = self.Review.query.filter_by(
                        user_id=user.id,
                        is_approved=True
                    ).count()
                    
                    if approved_reviews_count >= 1:
                        return True
                    
                    if len(review_text.strip()) >= 50:
                        return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error in auto-approval logic: {e}")
            return False
    
    def vote_review_helpful(self, review_id: int, user_id: int, is_helpful: bool = True) -> bool:
        try:
            if not self.Review:
                return False
            
            review = self.Review.query.get(review_id)
            if not review:
                return False
            
            if is_helpful:
                review.helpful_count = (review.helpful_count or 0) + 1
            else:
                review.helpful_count = max(0, (review.helpful_count or 0) - 1)
            
            self.db.session.commit()
            
            content = self.Content.query.get(review.content_id)
            if content and self.cache:
                try:
                    cache_key = f"details:slug:{content.slug}"
                    self.cache.delete(cache_key)
                except Exception as e:
                    logger.warning(f"Cache invalidation error: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error voting on review: {e}")
            self.db.session.rollback()
            return False
    
    def clear_cache_for_slug(self, slug: str) -> bool:
        try:
            if not self.cache:
                return False
            
            cache_key = f"details:slug:{slug}"
            self.cache.delete(cache_key)
            logger.info(f"Cleared cache for slug: {slug}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache for slug {slug}: {e}")
            return False
    
    def update_content_slug(self, content_id: int, force_update: bool = False) -> Optional[str]:
        try:
            content = self.Content.query.get(content_id)
            if content:
                new_slug = SlugManager.update_content_slug(self.db, content, force_update)
                self.db.session.commit()
                
                if self.cache:
                    try:
                        cache_key = f"details:slug:{new_slug}"
                        self.cache.delete(cache_key)
                    except Exception as e:
                        logger.warning(f"Cache invalidation error: {e}")
                
                return new_slug
            return None
        except Exception as e:
            logger.error(f"Error updating content slug: {e}")
            self.db.session.rollback()
            return None
    
    def get_content_by_slug_or_id(self, identifier: Union[str, int]) -> Optional[Any]:
        try:
            if isinstance(identifier, int):
                return self.Content.query.get(identifier)
            else:
                content = self.Content.query.filter_by(slug=identifier).first()
                if not content:
                    content = self._find_content_fuzzy(identifier)
                return content
        except Exception as e:
            logger.error(f"Error getting content by slug/id: {e}")
            return None

def init_movies_details_service(app, db, models, cache):
    try:
        with app.app_context():
            service = MoviesDetailsService(db, models, cache)
            logger.info("CineBrain Movies Details Service initialized successfully")
            return service
    except ValueError as e:
        logger.error(f"Failed to initialize CineBrain Movies Details Service: {e}")
        raise e
    except Exception as e:
        logger.error(f"Failed to initialize CineBrain Movies Details Service: {e}")
        raise e