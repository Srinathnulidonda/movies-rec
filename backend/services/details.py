# backend/services/details.py
import re
import json
import logging
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

logger = logging.getLogger(__name__)

TMDB_API_KEY = None
OMDB_API_KEY = None
YOUTUBE_API_KEY = None
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
OMDB_BASE_URL = 'http://www.omdbapi.com/'
YOUTUBE_BASE_URL = 'https://www.googleapis.com/youtube/v3'
JIKAN_BASE_URL = 'https://api.jikan.moe/v4'

POSTER_BASE = 'https://image.tmdb.org/t/p/w500'
BACKDROP_BASE = 'https://image.tmdb.org/t/p/w1280'
PROFILE_BASE = 'https://image.tmdb.org/t/p/w185'
STILL_BASE = 'https://image.tmdb.org/t/p/w780'

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

@dataclass
class PersonDetails:
    slug: str
    id: int
    name: str
    biography: str
    birthday: Optional[str]
    deathday: Optional[str]
    place_of_birth: Optional[str]
    profile_url: Optional[str]
    popularity: float
    known_for_department: Optional[str]
    also_known_as: List[str]
    gender: Optional[int]
    filmography: Dict
    images: List[str]
    social_media: Dict
    total_works: int
    awards: List[Dict]
    personal_info: Dict
    career_highlights: Dict
    
    def to_dict(self):
        return asdict(self)

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

class SlugManager:
    
    @staticmethod
    def generate_slug(title: str, year: Optional[int] = None, content_type: str = 'movie') -> str:
        if not title or not isinstance(title, str):
            return f"content-{int(time.time())}"
        
        try:
            clean_title = str(title).strip()
            if not clean_title:
                return f"content-{int(time.time())}"
            
            try:
                slug = slugify(clean_title, max_length=80, word_boundary=True, save_order=True)
            except Exception as slugify_error:
                logger.warning(f"Slugify failed for '{clean_title}': {slugify_error}")
                slug = None
            
            if not slug:
                slug = clean_title.lower()
                slug = re.sub(r'[^\w\s-]', '', slug, flags=re.ASCII)
                slug = re.sub(r'[-\s]+', '-', slug)
                slug = slug.strip('-')
            
            if not slug or len(slug) < 1:
                type_prefix = {
                    'movie': 'movie',
                    'tv': 'tv-show',
                    'anime': 'anime',
                    'person': 'person'
                }.get(content_type, 'content')
                return f"{type_prefix}-{int(time.time())}"
            
            if len(slug) > 80:
                truncated = slug[:80]
                last_dash = truncated.rfind('-')
                if last_dash > 40:
                    slug = truncated[:last_dash]
                else:
                    slug = truncated
            
            if year and content_type == 'movie' and isinstance(year, int):
                if 1800 <= year <= 2100:
                    slug = f"{slug}-{year}"
            
            if content_type == 'anime':
                if not slug.startswith('anime-'):
                    slug = f"anime-{slug}"
            
            if len(slug) > 100:
                parts = slug[:97].split('-')
                if len(parts) > 1:
                    slug = '-'.join(parts[:-1])
                else:
                    slug = slug[:97]
            
            return slug
            
        except Exception as e:
            logger.error(f"Error generating slug for title '{title}': {e}")
            safe_slug = ''
            for c in str(title):
                if c.isalnum():
                    safe_slug += c.lower()
                elif c in ' -_':
                    safe_slug += '-'
            
            safe_slug = re.sub(r'-+', '-', safe_slug).strip('-')
            
            if not safe_slug:
                type_prefix = {
                    'movie': 'movie',
                    'tv': 'tv-show', 
                    'anime': 'anime',
                    'person': 'person'
                }.get(content_type, 'content')
                return f"{type_prefix}-{int(time.time())}"
            return safe_slug[:50]
            
    @staticmethod
    def generate_unique_slug(db, model, title: str, year: Optional[int] = None, 
                        content_type: str = 'movie', existing_id: Optional[int] = None) -> str:
        try:
            base_slug = SlugManager.generate_slug(title, year, content_type)
            
            if not base_slug:
                type_prefix = {
                    'movie': 'movie',
                    'tv': 'tv-show',
                    'anime': 'anime', 
                    'person': 'person'
                }.get(content_type, 'content')
                base_slug = f"{type_prefix}-{int(time.time())}"
            
            slug = base_slug
            counter = 1
            max_attempts = 50
            
            while counter <= max_attempts:
                try:
                    query = db.session.query(model.id).filter_by(slug=slug)
                    
                    if existing_id:
                        query = query.filter(model.id != existing_id)
                    
                    exists = query.first() is not None
                    
                    if not exists:
                        break
                    
                    slug = f"{base_slug}-{counter}"
                    counter += 1
                    
                except Exception as e:
                    logger.error(f"Error checking slug uniqueness: {e}")
                    slug = f"{base_slug}-{int(time.time())}"
                    break
            
            if counter > max_attempts:
                slug = f"{base_slug}-{int(time.time())}"
                logger.warning(f"Hit max attempts for slug generation, using timestamp: {slug}")
            
            return slug
            
        except Exception as e:
            logger.error(f"Error generating unique slug: {e}")
            type_prefix = {
                'movie': 'movie',
                'tv': 'tv-show',
                'anime': 'anime',
                'person': 'person'
            }.get(content_type, 'content')
            return f"{type_prefix}-{int(time.time())}-{abs(hash(str(title)))[:6]}"    
    
    @staticmethod
    def extract_info_from_slug(slug: str) -> Dict:
        try:
            if not slug:
                return {'title': 'Unknown', 'year': None, 'content_type': 'movie'}
            
            content_type = 'movie'
            clean_slug = slug
            
            if slug.startswith('anime-'):
                content_type = 'anime'
                clean_slug = slug[6:]
            elif slug.startswith('tv-') or 'tv-show' in slug or '-tv-' in slug:
                content_type = 'tv'
                clean_slug = slug.replace('tv-show-', '').replace('tv-', '').replace('-tv-', '-')
            elif slug.startswith('series-') or '-series-' in slug:
                content_type = 'tv'
                clean_slug = slug.replace('series-', '').replace('-series-', '-')
            elif slug.startswith('person-'):
                content_type = 'person'
                clean_slug = slug[7:]
            
            year_match = re.search(r'-(\d{4})$', clean_slug)
            year = None
            title_slug = clean_slug
            
            if year_match:
                year = int(year_match.group(1))
                title_slug = clean_slug[:year_match.start()]
                
                if 1900 <= year <= 2030:
                    title = SlugManager._slug_to_title(title_slug)
                    return {
                        'title': title,
                        'year': year,
                        'content_type': content_type
                    }
            
            title = SlugManager._slug_to_title(title_slug)
            return {
                'title': title,
                'year': None,
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
            
            if 'mission impossible' in title.lower():
                title = re.sub(r'Mission Impossible', 'Mission: Impossible', title, flags=re.IGNORECASE)
            
            title = re.sub(r'\bDc\b', 'DC', title)
            title = re.sub(r'\bMcu\b', 'MCU', title)
            title = re.sub(r'\bUk\b', 'UK', title)
            title = re.sub(r'\bUs\b', 'US', title)
            title = re.sub(r'\bTv\b', 'TV', title)
            
            roman_numerals = ['Ii', 'Iii', 'Iv', 'Vi', 'Vii', 'Viii', 'Ix', 'Xi', 'Xii']
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
                existing_id=getattr(content, 'id', None)
            )
            
            content.slug = new_slug
            
            return new_slug
            
        except Exception as e:
            logger.error(f"Error updating content slug: {e}")
            fallback = f"content-{getattr(content, 'id', int(time.time()))}"
            content.slug = fallback
            return fallback
    
    @staticmethod
    def migrate_slugs(db, models, batch_size: int = 50) -> Dict:
        stats = {
            'content_updated': 0,
            'persons_updated': 0,
            'errors': 0,
            'total_processed': 0
        }
        
        try:
            if 'Content' in models:
                Content = models['Content']
                
                offset = 0
                batch_count = 0
                
                while True:
                    content_items = Content.query.filter(
                        or_(Content.slug == None, Content.slug == '')
                    ).offset(offset).limit(batch_size).all()
                    
                    if not content_items:
                        break
                    
                    batch_count += 1
                    logger.info(f"Processing content batch {batch_count}, items: {len(content_items)}")
                    
                    for i, content in enumerate(content_items):
                        try:
                            SlugManager.update_content_slug(db, content)
                            stats['content_updated'] += 1
                            
                        except Exception as e:
                            logger.error(f"Error updating content {getattr(content, 'id', 'unknown')}: {e}")
                            stats['errors'] += 1
                    
                    try:
                        db.session.commit()
                        logger.info(f"Committed content batch {batch_count}: {stats['content_updated']} total updated")
                    except Exception as e:
                        logger.error(f"Batch commit failed: {e}")
                        db.session.rollback()
                        stats['errors'] += len(content_items)
                    
                    offset += batch_size
                    time.sleep(0.1)
            
            if 'Person' in models:
                Person = models['Person']
                
                offset = 0
                batch_count = 0
                
                while True:
                    person_items = Person.query.filter(
                        or_(Person.slug == None, Person.slug == '')
                    ).offset(offset).limit(batch_size).all()
                    
                    if not person_items:
                        break
                    
                    batch_count += 1
                    logger.info(f"Processing person batch {batch_count}, items: {len(person_items)}")
                    
                    for i, person in enumerate(person_items):
                        try:
                            name = getattr(person, 'name', '')
                            if not name:
                                name = f"Person {getattr(person, 'id', 'Unknown')}"
                            
                            new_slug = SlugManager.generate_unique_slug(
                                db, 
                                Person, 
                                name, 
                                content_type='person',
                                existing_id=getattr(person, 'id', None)
                            )
                            
                            person.slug = new_slug
                            stats['persons_updated'] += 1
                            
                        except Exception as e:
                            logger.error(f"Error updating person {getattr(person, 'id', 'unknown')}: {e}")
                            stats['errors'] += 1
                    
                    try:
                        db.session.commit()
                        logger.info(f"Committed person batch {batch_count}: {stats['persons_updated']} total updated")
                    except Exception as e:
                        logger.error(f"Person batch commit failed: {e}")
                        db.session.rollback()
                        stats['errors'] += len(person_items)
                    
                    offset += batch_size
                    time.sleep(0.1)
            
            stats['total_processed'] = stats['content_updated'] + stats['persons_updated']
            logger.info(f"Slug migration completed: {stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in slug migration: {e}")
            db.session.rollback()
            stats['errors'] += 1
            return stats

class ContentService:
    
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.Content = models.get('Content')
        self.Person = models.get('Person')
    
    def save_content_from_tmdb(self, tmdb_data: Dict, content_type: str) -> Any:
        try:
            existing = None
            if self.Content and tmdb_data.get('id'):
                existing = self.Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            
            if existing:
                if not existing.slug:
                    try:
                        SlugManager.update_content_slug(self.db, existing)
                        self.db.session.commit()
                    except Exception as e:
                        logger.warning(f"Failed to update existing content slug: {e}")
                return existing
            
            title = tmdb_data.get('title') or tmdb_data.get('name') or 'Unknown Title'
            
            release_date = None
            year = None
            date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
            if date_str:
                try:
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    year = release_date.year
                except:
                    pass
            
            slug = SlugManager.generate_unique_slug(
                self.db, self.Content, title, year, content_type
            )
            
            genres = []
            if 'genres' in tmdb_data:
                genres = [genre['name'] for genre in tmdb_data['genres']]
            elif 'genre_ids' in tmdb_data:
                genres = self._map_genre_ids(tmdb_data['genre_ids'])
            
            content_data = {
                'slug': slug,
                'tmdb_id': tmdb_data['id'],
                'title': title,
                'original_title': tmdb_data.get('original_title') or tmdb_data.get('original_name'),
                'content_type': content_type,
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
            
            return content
            
        except Exception as e:
            logger.error(f"Error saving content from TMDB: {e}")
            self.db.session.rollback()
            return None
    
    def save_anime_content(self, anime_data: Dict) -> Any:
        try:
            existing = None
            if self.Content and anime_data.get('mal_id'):
                existing = self.Content.query.filter_by(mal_id=anime_data['mal_id']).first()
            
            if existing:
                if not existing.slug:
                    try:
                        SlugManager.update_content_slug(self.db, existing)
                        self.db.session.commit()
                    except Exception as e:
                        logger.warning(f"Failed to update existing anime slug: {e}")
                return existing
            
            title = anime_data.get('title') or 'Unknown Anime'
            
            release_date = None
            year = None
            if anime_data.get('aired', {}).get('from'):
                try:
                    release_date = datetime.strptime(anime_data['aired']['from'][:10], '%Y-%m-%d').date()
                    year = release_date.year
                except:
                    pass
            
            slug = SlugManager.generate_unique_slug(
                self.db, self.Content, title, year, 'anime'
            )
            
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            content_data = {
                'slug': slug,
                'mal_id': anime_data['mal_id'],
                'title': title,
                'original_title': anime_data.get('title_japanese'),
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
            
            return content
            
        except Exception as e:
            logger.error(f"Error saving anime content: {e}")
            self.db.session.rollback()
            return None
    
    def get_or_create_person(self, person_data: Dict) -> Any:
        try:
            existing = None
            if self.Person and person_data.get('id'):
                existing = self.Person.query.filter_by(tmdb_id=person_data['id']).first()
            
            if existing:
                if not existing.slug:
                    try:
                        name = existing.name or f"Person {existing.id}"
                        slug = SlugManager.generate_unique_slug(
                            self.db, self.Person, name, content_type='person',
                            existing_id=existing.id
                        )
                        existing.slug = slug
                        self.db.session.commit()
                    except Exception as e:
                        logger.warning(f"Failed to update person slug: {e}")
                return existing
            
            name = person_data.get('name') or 'Unknown Person'
            slug = SlugManager.generate_unique_slug(
                self.db, self.Person, name, content_type='person'
            )
            
            person_data_clean = {
                'slug': slug,
                'tmdb_id': person_data['id'],
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
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]

class DetailsService:
    
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
        self._init_api_keys()
    
    def _init_api_keys(self):
        global TMDB_API_KEY, OMDB_API_KEY, YOUTUBE_API_KEY
        
        try:
            if has_app_context() and current_app:
                TMDB_API_KEY = current_app.config.get('TMDB_API_KEY')
                OMDB_API_KEY = current_app.config.get('OMDB_API_KEY')
                YOUTUBE_API_KEY = current_app.config.get('YOUTUBE_API_KEY')
        except Exception as e:
            logger.warning(f"Could not initialize API keys: {e}")
    
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
    
    def get_details_by_slug(self, slug: str, user_id: Optional[int] = None) -> Optional[Dict]:
        try:
            if not has_app_context():
                logger.warning("No app context available for get_details_by_slug")
                return None
            
            cache_key = f"details:slug:{slug}"
            if self.cache:
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
                        logger.info(f"Content not found for slug: {slug}, consider implementing external API fetch")
                    else:
                        logger.warning(f"Content not found for slug: {slug}")
                    return None
            
            if not content.slug:
                try:
                    SlugManager.update_content_slug(self.db, content)
                    self.db.session.commit()
                except Exception as e:
                    logger.warning(f"Failed to update content slug: {e}")
            
            details = self._build_content_details(content, user_id)
            
            if self.cache and details:
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
    
    def _find_content_fuzzy(self, slug: str) -> Optional[Any]:
        try:
            info = SlugManager.extract_info_from_slug(slug)
            title = info['title']
            year = info['year']
            content_type = info['content_type']
            
            logger.info(f"Fuzzy search for slug '{slug}': title='{title}', year={year}, type={content_type}")
            
            title_variations = [title.lower()]
            
            if 'mission' in title.lower() and 'impossible' in title.lower():
                parts = title.lower().split()
                mission_idx = next((i for i, part in enumerate(parts) if 'mission' in part), -1)
                impossible_idx = next((i for i, part in enumerate(parts) if 'impossible' in part), -1)
                
                if mission_idx >= 0 and impossible_idx >= 0:
                    before_impossible = ' '.join(parts[:impossible_idx])
                    after_impossible = ' '.join(parts[impossible_idx+1:])
                    
                    title_variations.extend([
                        f"mission: impossible {after_impossible}".strip(),
                        f"mission: impossible – {after_impossible}".strip(),
                        f"mission: impossible - {after_impossible}".strip(),
                        f"mission impossible {after_impossible}".strip(),
                    ])
            
            title_variations.extend([
                title.replace(':', '').lower(),
                title.replace(':', ' -').lower(),
                title.replace(':', ' –').lower(),
                title.replace(':', ' ').lower(),
                title.replace(' ', '').lower(),
                title.replace('-', ' ').lower(),
                title.replace('–', ' ').lower(),
            ])
            
            if ':' in title or '–' in title or '-' in title:
                for separator in [':', '–', '-']:
                    if separator in title:
                        main_title = title.split(separator)[0].strip()
                        title_variations.append(main_title.lower())
                        subtitle = title.split(separator, 1)[1].strip() if separator in title else ''
                        if subtitle:
                            title_variations.extend([
                                f"{main_title.lower()}: {subtitle.lower()}",
                                f"{main_title.lower()} – {subtitle.lower()}",
                                f"{main_title.lower()} - {subtitle.lower()}",
                                f"{main_title.lower()} {subtitle.lower()}",
                            ])
                        break
            
            if title.lower().startswith('the '):
                title_variations.append(title[4:].lower())
            else:
                title_variations.append(f"the {title.lower()}")
            
            seen = set()
            unique_variations = []
            for variation in title_variations:
                clean_variation = variation.strip()
                if clean_variation and clean_variation not in seen and len(clean_variation) > 2:
                    seen.add(clean_variation)
                    unique_variations.append(clean_variation)
            
            logger.info(f"Trying {len(unique_variations)} title variations: {unique_variations[:5]}...")
            
            results = []
            
            for i, variation in enumerate(unique_variations):
                if results and len(results) >= 3:
                    break
                    
                logger.info(f"Trying variation {i+1}: '{variation}'")
                
                if content_type and year:
                    query_results = self.Content.query.filter(
                        func.lower(self.Content.title).like(f"%{variation}%"),
                        self.Content.content_type == content_type,
                        func.extract('year', self.Content.release_date) == year
                    ).limit(3).all()
                    
                    if query_results:
                        logger.info(f"Found {len(query_results)} matches with exact type and year for '{variation}'")
                        results.extend(query_results)
                        continue
                
                if content_type and year:
                    query_results = self.Content.query.filter(
                        func.lower(self.Content.title).like(f"%{variation}%"),
                        self.Content.content_type == content_type,
                        func.extract('year', self.Content.release_date).between(year - 2, year + 2)
                    ).limit(3).all()
                    
                    if query_results:
                        logger.info(f"Found {len(query_results)} matches with type and flexible year for '{variation}'")
                        results.extend(query_results)
                        continue
                
                if content_type:
                    query_results = self.Content.query.filter(
                        func.lower(self.Content.title).like(f"%{variation}%"),
                        self.Content.content_type == content_type
                    ).order_by(self.Content.popularity.desc()).limit(5).all()
                    
                    if query_results:
                        logger.info(f"Found {len(query_results)} matches with type only for '{variation}'")
                        results.extend(query_results)
                        continue
            
            if not results:
                logger.info("No matches found with specific strategies, trying broader search...")
                best_variations = unique_variations[:5]
                
                for variation in best_variations:
                    if year:
                        query_results = self.Content.query.filter(
                            func.lower(self.Content.title).like(f"%{variation}%"),
                            func.extract('year', self.Content.release_date).between(year - 3, year + 3)
                        ).order_by(self.Content.popularity.desc()).limit(5).all()
                        
                        if query_results:
                            logger.info(f"Found {len(query_results)} matches with flexible criteria for '{variation}'")
                            results.extend(query_results)
                            break
                    
                    query_results = self.Content.query.filter(
                        func.lower(self.Content.title).like(f"%{variation}%")
                    ).order_by(self.Content.popularity.desc()).limit(10).all()
                    
                    if query_results:
                        logger.info(f"Found {len(query_results)} matches with title only for '{variation}'")
                        results.extend(query_results)
                        break
            
            if not results and len(unique_variations) > 0:
                logger.info("Trying partial word matching...")
                words = [word.strip() for word in title.lower().split() if len(word.strip()) > 2]
                if len(words) >= 2:
                    if 'mission' in words and 'impossible' in words:
                        partial_search = "mission impossible"
                    else:
                        partial_search = ' '.join(words[:min(2, len(words))])
                    
                    query_results = self.Content.query.filter(
                        func.lower(self.Content.title).like(f"%{partial_search}%")
                    ).order_by(self.Content.popularity.desc()).limit(5).all()
                    
                    if query_results:
                        logger.info(f"Found {len(query_results)} matches with partial search '{partial_search}'")
                        results.extend(query_results)
            
            seen_ids = set()
            unique_results = []
            for result in results:
                if result.id not in seen_ids:
                    seen_ids.add(result.id)
                    unique_results.append(result)
            
            if unique_results:
                best_match = max(unique_results, key=lambda x: self._calculate_similarity(x.title.lower(), title.lower()))
                logger.info(f"Best fuzzy match for '{slug}': {best_match.title} (ID: {best_match.id}, slug: {best_match.slug})")
                return best_match
            
            logger.warning(f"No fuzzy match found for '{slug}' with title '{title}' and year {year}")
            logger.info(f"Consider fetching '{title}' ({year}) from external APIs")
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy content search: {e}")
            return None
            
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        try:
            from difflib import SequenceMatcher
            
            clean_str1 = str1.lower().strip()
            clean_str2 = str2.lower().strip()
            
            basic_sim = SequenceMatcher(None, clean_str1, clean_str2).ratio()
            
            words1 = set(clean_str1.split())
            words2 = set(clean_str2.split())
            
            if words1 and words2:
                word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                return 0.7 * basic_sim + 0.3 * word_overlap
            
            return basic_sim
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def _should_fetch_from_external(self, slug: str, title: str, year: Optional[int]) -> bool:
        try:
            current_year = datetime.now().year
            if year and (current_year - 5 <= year <= current_year + 2):
                logger.info(f"Content '{title}' ({year}) not found in database, could fetch from external APIs")
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
            self._init_api_keys()
            
            from flask import current_app
            app = current_app._get_current_object() if has_app_context() else None
            
            futures = {}
            
            with ThreadPoolExecutor(max_workers=4) as executor:
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
                    tmdb_data = futures['tmdb'].result(timeout=10) or {}
            except Exception as e:
                logger.warning(f"TMDB fetch error/timeout: {e}")
            
            try:
                if 'omdb' in futures:
                    omdb_data = futures['omdb'].result(timeout=8) or {}
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
                similar = futures['similar'].result(timeout=8) or []
            except Exception as e:
                logger.warning(f"Similar content fetch error/timeout: {e}")
            
            try:
                gallery = futures['gallery'].result(timeout=8) or gallery
            except Exception as e:
                logger.warning(f"Gallery fetch error/timeout: {e}")
            
            try:
                trailer = futures['trailer'].result(timeout=8)
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
                    self.db, self.Person, name, content_type='person'
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
                if not person.slug:
                    person.slug = SlugManager.generate_unique_slug(
                        self.db, self.Person, person.name, content_type='person',
                        existing_id=person.id
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

    @ensure_app_context
    def get_person_details(self, person_slug: str) -> Optional[Dict]:
        try:
            person = self.Person.query.filter_by(slug=person_slug).first()
            
            if not person:
                person = self._find_person_fuzzy(person_slug)
                
                if not person:
                    logger.warning(f"Person not found for slug: {person_slug}")
                    return None
            
            if not person.slug:
                try:
                    SlugManager.update_content_slug(self.db, person)
                    self.db.session.commit()
                except Exception as e:
                    logger.warning(f"Failed to update person slug: {e}")
            
            tmdb_data = {}
            if person.tmdb_id and TMDB_API_KEY:
                logger.info(f"Fetching comprehensive TMDB data for {person.name} (ID: {person.tmdb_id})")
                tmdb_data = self._fetch_complete_person_details(person.tmdb_id)
                
                self._update_person_from_tmdb(person, tmdb_data)
                
                if tmdb_data:
                    self._fetch_and_save_person_credits(person, tmdb_data)
            
            filmography = self._get_enhanced_filmography(person.id, tmdb_data)
            
            person_details = self._build_comprehensive_person_details(person, tmdb_data, filmography)
            
            return person_details
            
        except Exception as e:
            logger.error(f"Error getting person details for slug {person_slug}: {e}")
            logger.exception(e)
            return None

    def _fetch_complete_person_details(self, tmdb_id: int) -> Dict:
        try:
            url = f"{TMDB_BASE_URL}/person/{tmdb_id}"
            params = {
                'api_key': TMDB_API_KEY,
                'append_to_response': 'images,external_ids,combined_credits,movie_credits,tv_credits'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"TMDB person API returned {response.status_code} for person {tmdb_id}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching complete person details from TMDB: {e}")
            return {}

    def _fetch_and_save_person_credits(self, person: Any, tmdb_data: Dict):
        try:
            if not tmdb_data:
                return
            
            movie_credits = tmdb_data.get('movie_credits', {})
            self._process_person_credits(person, movie_credits.get('cast', []), 'cast', 'movie')
            self._process_person_credits(person, movie_credits.get('crew', []), 'crew', 'movie')
            
            tv_credits = tmdb_data.get('tv_credits', {})
            self._process_person_credits(person, tv_credits.get('cast', []), 'cast', 'tv')
            self._process_person_credits(person, tv_credits.get('crew', []), 'crew', 'tv')
            
            combined_credits = tmdb_data.get('combined_credits', {})
            if combined_credits:
                self._process_person_credits(person, combined_credits.get('cast', []), 'cast', None)
                self._process_person_credits(person, combined_credits.get('crew', []), 'crew', None)
            
            self.db.session.commit()
            logger.info(f"Successfully saved credits for {person.name}")
            
        except Exception as e:
            logger.error(f"Error fetching/saving person credits: {e}")
            self.db.session.rollback()

    def _process_person_credits(self, person: Any, credits: List[Dict], role_type: str, content_type: Optional[str]):
        try:
            for credit in credits:
                try:
                    content = self._get_or_create_content_from_credit(credit, content_type)
                    if not content:
                        continue
                    
                    if role_type == 'cast':
                        self._get_or_create_content_person(
                            content.id, 
                            person.id, 
                            'cast',
                            character=credit.get('character'),
                            order=credit.get('order', 999)
                        )
                    else:
                        self._get_or_create_content_person(
                            content.id,
                            person.id,
                            'crew',
                            job=credit.get('job'),
                            department=credit.get('department')
                        )
                        
                except Exception as e:
                    logger.warning(f"Error processing credit: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing person credits: {e}")

    def _get_or_create_content_from_credit(self, credit: Dict, content_type: Optional[str]) -> Any:
        try:
            tmdb_id = credit.get('id')
            if not tmdb_id:
                return None
            
            existing = self.Content.query.filter_by(tmdb_id=tmdb_id).first()
            if existing:
                return existing
            
            if not content_type:
                media_type = credit.get('media_type', 'movie')
                content_type = 'tv' if media_type == 'tv' else 'movie'
            
            title = credit.get('title') or credit.get('name') or 'Unknown Title'
            
            release_date = None
            year = None
            date_str = credit.get('release_date') or credit.get('first_air_date')
            if date_str:
                try:
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    year = release_date.year
                except:
                    pass
            
            slug = SlugManager.generate_unique_slug(
                self.db, self.Content, title, year, content_type
            )
            
            content = self.Content(
                slug=slug,
                tmdb_id=tmdb_id,
                title=title,
                original_title=credit.get('original_title') or credit.get('original_name'),
                content_type=content_type,
                release_date=release_date,
                rating=credit.get('vote_average'),
                vote_count=credit.get('vote_count'),
                popularity=credit.get('popularity'),
                overview=credit.get('overview'),
                poster_path=credit.get('poster_path'),
                backdrop_path=credit.get('backdrop_path')
            )
            
            self.db.session.add(content)
            self.db.session.flush()
            
            return content
            
        except Exception as e:
            logger.error(f"Error creating content from credit: {e}")
            return None

    def _get_enhanced_filmography(self, person_id: int, tmdb_data: Dict) -> Dict:
        try:
            filmography = {
                'as_actor': [],
                'as_director': [],
                'as_writer': [],
                'as_producer': [],
                'as_other_crew': [],
                'upcoming': [],
                'by_year': {},
                'by_decade': {},
                'statistics': {
                    'total_projects': 0,
                    'years_active': 0,
                    'debut_year': None,
                    'latest_year': None,
                    'highest_rated': None,
                    'most_popular': None,
                    'average_rating': 0,
                    'total_votes': 0
                }
            }
            
            filmography_entries = self.db.session.query(
                self.ContentPerson, self.Content
            ).join(
                self.Content
            ).filter(
                self.ContentPerson.person_id == person_id
            ).order_by(
                self.Content.release_date.desc().nullslast()
            ).all()
            
            years = []
            all_ratings = []
            all_popularity = []
            total_votes = 0
            current_year = datetime.now().year
            
            for cp, content in filmography_entries:
                try:
                    if not content.slug:
                        try:
                            SlugManager.update_content_slug(self.db, content)
                        except Exception:
                            content.slug = f"content-{content.id}"
                    
                    year = None
                    if content.release_date:
                        try:
                            year = int(content.release_date.year)
                            years.append(year)
                        except (AttributeError, TypeError, ValueError):
                            pass
                    
                    rating = 0
                    try:
                        if content.rating and isinstance(content.rating, (int, float)):
                            rating = float(content.rating)
                            if rating > 0:
                                all_ratings.append(rating)
                    except (TypeError, ValueError):
                        pass
                    
                    popularity = 0
                    try:
                        if content.popularity and isinstance(content.popularity, (int, float)):
                            popularity = float(content.popularity)
                            if popularity > 0:
                                all_popularity.append(popularity)
                    except (TypeError, ValueError):
                        pass
                    
                    try:
                        if content.vote_count and isinstance(content.vote_count, (int, float)):
                            total_votes += int(content.vote_count)
                    except (TypeError, ValueError):
                        pass
                    
                    work = {
                        'id': content.id,
                        'slug': content.slug,
                        'title': content.title,
                        'original_title': content.original_title,
                        'year': year,
                        'content_type': content.content_type,
                        'poster_path': self._format_image_url(content.poster_path, 'poster'),
                        'backdrop_path': self._format_image_url(content.backdrop_path, 'backdrop'),
                        'rating': rating,
                        'popularity': popularity,
                        'vote_count': content.vote_count or 0,
                        'character': cp.character,
                        'job': cp.job,
                        'department': cp.department,
                        'role_type': cp.role_type,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'overview': content.overview or ''
                    }
                    
                    year_key = str(year) if year else 'Unknown'
                    if year_key not in filmography['by_year']:
                        filmography['by_year'][year_key] = []
                    filmography['by_year'][year_key].append(work)
                    
                    if year:
                        decade = (year // 10) * 10
                        decade_key = f"{decade}s"
                        if decade_key not in filmography['by_decade']:
                            filmography['by_decade'][decade_key] = []
                        filmography['by_decade'][decade_key].append(work)
                    
                    if year and year > current_year:
                        filmography['upcoming'].append(work)
                    
                    if cp.role_type == 'cast':
                        filmography['as_actor'].append(work)
                    elif cp.role_type == 'crew':
                        job = (cp.job or '').lower()
                        department = (cp.department or '').lower()
                        
                        if 'director' in job or department == 'directing':
                            filmography['as_director'].append(work)
                        elif 'writer' in job or 'screenplay' in job or department == 'writing':
                            filmography['as_writer'].append(work)
                        elif 'producer' in job or department == 'production':
                            filmography['as_producer'].append(work)
                        else:
                            filmography['as_other_crew'].append(work)
                            
                except Exception as e:
                    logger.warning(f"Error processing filmography entry: {e}")
                    continue
            
            try:
                filmography['statistics']['total_projects'] = len(filmography_entries)
                
                if years:
                    valid_years = [y for y in years if isinstance(y, int) and 1900 <= y <= 2030]
                    if valid_years:
                        filmography['statistics']['debut_year'] = min(valid_years)
                        filmography['statistics']['latest_year'] = max(valid_years)
                        filmography['statistics']['years_active'] = max(valid_years) - min(valid_years) + 1
                
                if all_ratings:
                    filmography['statistics']['average_rating'] = round(sum(all_ratings) / len(all_ratings), 1)
                    
                    highest_rated_score = max(all_ratings)
                    for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer', 'as_other_crew']:
                        for work in filmography[role_type]:
                            if work['rating'] == highest_rated_score:
                                filmography['statistics']['highest_rated'] = work
                                break
                        if filmography['statistics']['highest_rated']:
                            break
                
                if all_popularity:
                    highest_popularity = max(all_popularity)
                    for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer', 'as_other_crew']:
                        for work in filmography[role_type]:
                            if work['popularity'] == highest_popularity:
                                filmography['statistics']['most_popular'] = work
                                break
                        if filmography['statistics']['most_popular']:
                            break
                
                filmography['statistics']['total_votes'] = total_votes
                
            except Exception as e:
                logger.warning(f"Error calculating filmography statistics: {e}")
            
            logger.info(f"Retrieved enhanced filmography: {filmography['statistics']['total_projects']} projects")
            return filmography
            
        except Exception as e:
            logger.error(f"Error getting enhanced filmography: {e}")
            return {
                'as_actor': [], 'as_director': [], 'as_writer': [], 'as_producer': [], 'as_other_crew': [],
                'upcoming': [], 'by_year': {}, 'by_decade': {}, 'statistics': {}
            }

    def _build_comprehensive_person_details(self, person: Any, tmdb_data: Dict, filmography: Dict) -> Dict:
        try:
            also_known_as = []
            try:
                if person.also_known_as:
                    if isinstance(person.also_known_as, str):
                        also_known_as = json.loads(person.also_known_as)
                    elif isinstance(person.also_known_as, list):
                        also_known_as = person.also_known_as
            except (json.JSONDecodeError, TypeError):
                pass
            
            if not also_known_as and tmdb_data.get('also_known_as'):
                also_known_as = tmdb_data['also_known_as']
            
            personal_info = self._build_enhanced_personal_info(person, tmdb_data, filmography)
            career_highlights = self._build_enhanced_career_highlights(filmography)
            images = self._get_person_images(tmdb_data)
            social_media = self._get_person_social_media(tmdb_data)
            
            person_details = {
                'id': person.id,
                'slug': person.slug,
                'tmdb_id': person.tmdb_id,
                'name': person.name,
                'biography': person.biography or tmdb_data.get('biography', ''),
                'birthday': person.birthday.isoformat() if person.birthday else tmdb_data.get('birthday'),
                'deathday': person.deathday.isoformat() if person.deathday else tmdb_data.get('deathday'),
                'place_of_birth': person.place_of_birth or tmdb_data.get('place_of_birth'),
                'profile_path': self._format_image_url(person.profile_path, 'profile'),
                'popularity': person.popularity or tmdb_data.get('popularity', 0),
                'known_for_department': person.known_for_department or tmdb_data.get('known_for_department'),
                'also_known_as': also_known_as,
                'gender': person.gender,
                'filmography': filmography,
                'images': images,
                'social_media': social_media,
                'total_works': self._calculate_total_works(filmography),
                'personal_info': personal_info,
                'career_highlights': career_highlights,
                'external_ids': tmdb_data.get('external_ids', {}),
                'known_for': self._get_known_for_works(filmography),
                'awards_recognition': self._build_awards_info(tmdb_data, filmography),
                'trivia': self._extract_trivia(person, tmdb_data),
                'collaborations': self._analyze_collaborations(filmography)
            }
            
            return person_details
            
        except Exception as e:
            logger.error(f"Error building comprehensive person details: {e}")
            return self._get_minimal_person_details(person)

    def _build_enhanced_personal_info(self, person: Any, tmdb_data: Dict, filmography: Dict) -> Dict:
        try:
            personal_info = {
                'age': None,
                'age_at_death': None,
                'zodiac_sign': None,
                'nationality': None,
                'birth_name': None,
                'height': None,
                'career_span': None,
                'status': 'Active',
                'family': {},
                'education': []
            }
            
            if person.birthday:
                try:
                    birth_date = person.birthday
                    if person.deathday:
                        end_date = person.deathday
                        personal_info['status'] = 'Deceased'
                        personal_info['age_at_death'] = end_date.year - birth_date.year
                        if end_date.month < birth_date.month or (end_date.month == birth_date.month and end_date.day < birth_date.day):
                            personal_info['age_at_death'] -= 1
                    else:
                        today = datetime.now().date()
                        personal_info['age'] = today.year - birth_date.year
                        if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
                            personal_info['age'] -= 1
                    
                    personal_info['zodiac_sign'] = self._get_zodiac_sign(birth_date.month, birth_date.day)
                    
                except Exception as e:
                    logger.warning(f"Error calculating age: {e}")
            
            if person.place_of_birth:
                try:
                    place_parts = person.place_of_birth.split(',')
                    if len(place_parts) > 0:
                        country = place_parts[-1].strip()
                        personal_info['nationality'] = country
                except Exception:
                    pass
            
            stats = filmography.get('statistics', {})
            if stats.get('debut_year') and stats.get('latest_year'):
                debut = stats['debut_year']
                latest = stats['latest_year']
                if isinstance(debut, int) and isinstance(latest, int):
                    personal_info['career_span'] = f"{debut} - {latest if latest != datetime.now().year else 'Present'}"
            
            return personal_info
            
        except Exception as e:
            logger.error(f"Error building enhanced personal info: {e}")
            return {}

    def _get_zodiac_sign(self, month: int, day: int) -> str:
        try:
            zodiac_signs = [
                (120, "Capricorn"), (218, "Aquarius"), (320, "Pisces"), (420, "Aries"),
                (521, "Taurus"), (621, "Gemini"), (722, "Cancer"), (823, "Leo"),
                (923, "Virgo"), (1023, "Libra"), (1122, "Scorpio"), (1222, "Sagittarius"), (1231, "Capricorn")
            ]
            
            date_number = month * 100 + day
            for date_limit, sign in zodiac_signs:
                if date_number <= date_limit:
                    return sign
            return "Capricorn"
        except:
            return ""

    def _build_enhanced_career_highlights(self, filmography: Dict) -> Dict:
        try:
            highlights = {
                'debut_work': None,
                'breakthrough_role': None,
                'latest_work': None,
                'most_successful_decade': None,
                'genre_expertise': {},
                'collaboration_frequency': {},
                'career_phases': [],
                'notable_achievements': []
            }
            
            stats = filmography.get('statistics', {})
            
            if stats.get('debut_year'):
                debut_year = stats['debut_year']
                for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                    for work in filmography.get(role_type, []):
                        if work.get('year') == debut_year:
                            highlights['debut_work'] = work
                            break
                    if highlights['debut_work']:
                        break
            
            if stats.get('latest_year'):
                latest_year = stats['latest_year']
                for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                    for work in filmography.get(role_type, []):
                        if work.get('year') == latest_year:
                            highlights['latest_work'] = work
                            break
                    if highlights['latest_work']:
                        break
            
            decade_stats = {}
            for decade, works in filmography.get('by_decade', {}).items():
                decade_stats[decade] = {
                    'count': len(works),
                    'avg_rating': 0,
                    'total_popularity': 0
                }
                
                ratings = [w['rating'] for w in works if w.get('rating', 0) > 0]
                if ratings:
                    decade_stats[decade]['avg_rating'] = sum(ratings) / len(ratings)
                
                popularity = [w['popularity'] for w in works if w.get('popularity', 0) > 0]
                if popularity:
                    decade_stats[decade]['total_popularity'] = sum(popularity)
            
            if decade_stats:
                best_decade = max(decade_stats.items(), 
                                key=lambda x: x[1]['count'] * x[1]['avg_rating'])
                highlights['most_successful_decade'] = {
                    'decade': best_decade[0],
                    'projects': best_decade[1]['count'],
                    'avg_rating': round(best_decade[1]['avg_rating'], 1)
                }
            
            return highlights
            
        except Exception as e:
            logger.error(f"Error building enhanced career highlights: {e}")
            return {}

    def _get_known_for_works(self, filmography: Dict) -> List[Dict]:
        try:
            all_works = []
            
            for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                for work in filmography.get(role_type, []):
                    if work.get('rating', 0) > 0 or work.get('popularity', 0) > 0:
                        rating_score = work.get('rating', 0) * 10
                        popularity_score = min(work.get('popularity', 0), 100)
                        combined_score = (rating_score + popularity_score) / 2
                        
                        work_copy = work.copy()
                        work_copy['combined_score'] = combined_score
                        work_copy['role_description'] = work.get('character') or work.get('job') or 'Unknown Role'
                        all_works.append(work_copy)
            
            all_works.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            return all_works[:10]
            
        except Exception as e:
            logger.error(f"Error getting known for works: {e}")
            return []

    def _build_awards_info(self, tmdb_data: Dict, filmography: Dict) -> Dict:
        try:
            awards_info = {
                'major_awards': [],
                'nominations': [],
                'recognition': [],
                'honors': []
            }
            
            stats = filmography.get('statistics', {})
            
            if stats.get('highest_rated'):
                highest = stats['highest_rated']
                if highest.get('rating', 0) >= 8.0:
                    awards_info['recognition'].append({
                        'type': 'Critical Acclaim',
                        'description': f"Highly rated performance in '{highest.get('title', 'Unknown')}'",
                        'rating': highest.get('rating'),
                        'year': highest.get('year')
                    })
            
            if stats.get('total_projects', 0) >= 50:
                awards_info['honors'].append({
                    'type': 'Prolific Career',
                    'description': f"Over {stats['total_projects']} film and television credits"
                })
            
            return awards_info
            
        except Exception as e:
            logger.error(f"Error building awards info: {e}")
            return {}

    def _extract_trivia(self, person: Any, tmdb_data: Dict) -> List[str]:
        try:
            trivia = []
            
            if person.birthday and person.place_of_birth:
                try:
                    age = datetime.now().year - person.birthday.year
                    trivia.append(f"Born in {person.place_of_birth}")
                    if age > 0:
                        trivia.append(f"Currently {age} years old")
                except:
                    pass
            
            if person.known_for_department:
                trivia.append(f"Primarily known for work in {person.known_for_department}")
            
            return trivia
            
        except Exception as e:
            logger.error(f"Error extracting trivia: {e}")
            return []

    def _analyze_collaborations(self, filmography: Dict) -> Dict:
        try:
            return {
                'frequent_co_stars': [],
                'frequent_directors': [],
                'frequent_writers': [],
                'production_companies': []
            }
            
        except Exception as e:
            logger.error(f"Error analyzing collaborations: {e}")
            return {}

    def _get_minimal_person_details(self, person: Any) -> Dict:
        try:
            return {
                'id': person.id,
                'slug': person.slug or f"person-{person.id}",
                'name': person.name,
                'biography': person.biography or '',
                'profile_path': self._format_image_url(person.profile_path, 'profile'),
                'popularity': person.popularity or 0,
                'known_for_department': person.known_for_department,
                'filmography': {
                    'as_actor': [], 'as_director': [], 'as_writer': [], 'as_producer': [],
                    'statistics': {'total_projects': 0}
                },
                'error': 'Detailed information temporarily unavailable'
            }
        except Exception as e:
            logger.error(f"Error creating minimal person details: {e}")
            return {'error': 'Person details unavailable'}

    def _update_person_from_tmdb(self, person: Any, tmdb_data: Dict):
        try:
            if not tmdb_data:
                return
            
            updated = False
            
            if not person.biography and tmdb_data.get('biography'):
                person.biography = tmdb_data['biography']
                updated = True
            
            if not person.birthday and tmdb_data.get('birthday'):
                try:
                    person.birthday = datetime.strptime(tmdb_data['birthday'], '%Y-%m-%d').date()
                    updated = True
                except:
                    pass
            
            if not person.deathday and tmdb_data.get('deathday'):
                try:
                    person.deathday = datetime.strptime(tmdb_data['deathday'], '%Y-%m-%d').date()
                    updated = True
                except:
                    pass
            
            if not person.place_of_birth and tmdb_data.get('place_of_birth'):
                person.place_of_birth = tmdb_data['place_of_birth']
                updated = True
            
            if not person.also_known_as and tmdb_data.get('also_known_as'):
                person.also_known_as = json.dumps(tmdb_data['also_known_as'])
                updated = True
            
            if tmdb_data.get('popularity') and tmdb_data['popularity'] > (person.popularity or 0):
                person.popularity = tmdb_data['popularity']
                updated = True
            
            if updated:
                self.db.session.commit()
                logger.info(f"Updated person {person.name} with TMDB data")
                
        except Exception as e:
            logger.error(f"Error updating person from TMDB: {e}")
            self.db.session.rollback()

    def _find_person_fuzzy(self, slug: str) -> Optional[Any]:
        try:
            info = SlugManager.extract_info_from_slug(slug)
            name = info['title']
            
            results = self.Person.query.filter(
                func.lower(self.Person.name).like(f"%{name.lower()}%")
            ).limit(10).all()
            
            if results:
                best_match = max(results, key=lambda x: self._calculate_similarity(x.name, name))
                logger.info(f"Found fuzzy match for person '{slug}': {best_match.name}")
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy person search: {e}")
            return None

    def _get_person_images(self, tmdb_data: Dict) -> List[Dict]:
        try:
            images = []
            if tmdb_data.get('images', {}).get('profiles'):
                for img in tmdb_data['images']['profiles'][:20]:
                    images.append({
                        'url': self._format_image_url(img['file_path'], 'profile'),
                        'width': img.get('width'),
                        'height': img.get('height'),
                        'aspect_ratio': img.get('aspect_ratio')
                    })
            return images
        except Exception as e:
            logger.error(f"Error getting person images: {e}")
            return []

    def _get_person_social_media(self, tmdb_data: Dict) -> Dict:
        try:
            social = {}
            if tmdb_data.get('external_ids'):
                ids = tmdb_data['external_ids']
                if ids.get('twitter_id'):
                    social['twitter'] = f"https://twitter.com/{ids['twitter_id']}"
                if ids.get('instagram_id'):
                    social['instagram'] = f"https://instagram.com/{ids['instagram_id']}"
                if ids.get('facebook_id'):
                    social['facebook'] = f"https://facebook.com/{ids['facebook_id']}"
                if ids.get('imdb_id'):
                    social['imdb'] = f"https://www.imdb.com/name/{ids['imdb_id']}"
                if ids.get('tiktok_id'):
                    social['tiktok'] = f"https://tiktok.com/@{ids['tiktok_id']}"
                if ids.get('youtube_id'):
                    social['youtube'] = f"https://youtube.com/channel/{ids['youtube_id']}"
                if ids.get('wikidata_id'):
                    social['wikidata'] = f"https://www.wikidata.org/wiki/{ids['wikidata_id']}"
            return social
        except Exception as e:
            logger.error(f"Error getting person social media: {e}")
            return {}

    def _calculate_total_works(self, filmography: Dict) -> int:
        try:
            total = 0
            for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                total += len(filmography.get(role_type, []))
            return total
        except:
            return 0

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
                
                response = self.session.get(url, params=params, timeout=8)
                
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
                return {}
                
            endpoint = 'movie' if content_type == 'movie' else 'tv'
            url = f"{TMDB_BASE_URL}/{endpoint}/{tmdb_id}"
            
            params = {
                'api_key': TMDB_API_KEY,
                'append_to_response': 'videos,images,credits,similar,recommendations,reviews,external_ids,watch/providers,content_ratings,release_dates,keywords'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching TMDB details: {e}")
            return {}
    
    def _fetch_omdb_details(self, imdb_id: str) -> Dict:
        try:
            if not OMDB_API_KEY:
                return {}
                
            params = {
                'apikey': OMDB_API_KEY,
                'i': imdb_id,
                'plot': 'full'
            }
            
            response = self.session.get(OMDB_BASE_URL, params=params, timeout=8)
            
            if response.status_code == 200:
                return response.json()
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching OMDB details: {e}")
            return {}
    
    def _get_trailer(self, title: str, content_type: str) -> Optional[Dict]:
        try:
            if not YOUTUBE_API_KEY:
                return None
            
            search_query = f"{title} official trailer"
            if content_type == 'anime':
                search_query = f"{title} anime trailer PV"
            
            url = f"{YOUTUBE_BASE_URL}/search"
            params = {
                'key': YOUTUBE_API_KEY,
                'q': search_query,
                'part': 'snippet',
                'type': 'video',
                'maxResults': 5,
                'order': 'relevance'
            }
            
            response = self.session.get(url, params=params, timeout=8)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('items'):
                    video = data['items'][0]
                    return {
                        'youtube_id': video['id']['videoId'],
                        'title': video['snippet']['title'],
                        'thumbnail': video['snippet']['thumbnails']['high']['url'],
                        'embed_url': f"https://www.youtube.com/embed/{video['id']['videoId']}",
                        'watch_url': f"https://www.youtube.com/watch?v={video['id']['videoId']}"
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting trailer: {e}")
            return None
    
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
                if not item.slug:
                    try:
                        SlugManager.update_content_slug(self.db, item)
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
    
    @ensure_app_context
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
            
            review = self.Review(
                content_id=content_id,
                user_id=user_id,
                rating=review_data.get('rating'),
                title=review_data.get('title'),
                review_text=review_data.get('review_text'),
                has_spoilers=review_data.get('has_spoilers', False),
                is_approved=False
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
            
            return {
                'success': True,
                'review_id': review.id,
                'message': 'Review submitted for moderation'
            }
            
        except Exception as e:
            logger.error(f"Error adding review: {e}")
            self.db.session.rollback()
            return {
                'success': False,
                'message': 'Failed to add review'
            }
    
    @ensure_app_context
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
    
    def migrate_all_slugs(self, batch_size: int = 50) -> Dict:
        return SlugManager.migrate_slugs(self.db, self.models, batch_size)
    
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

def init_details_service(app, db, models, cache):
    with app.app_context():
        service = DetailsService(db, models, cache)
        return service