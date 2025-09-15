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

# Configuration
TMDB_API_KEY = None
OMDB_API_KEY = None
YOUTUBE_API_KEY = None
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
OMDB_BASE_URL = 'http://www.omdbapi.com/'
YOUTUBE_BASE_URL = 'https://www.googleapis.com/youtube/v3'
JIKAN_BASE_URL = 'https://api.jikan.moe/v4'

# Image bases
POSTER_BASE = 'https://image.tmdb.org/t/p/w500'
BACKDROP_BASE = 'https://image.tmdb.org/t/p/w1280'
PROFILE_BASE = 'https://image.tmdb.org/t/p/w185'
STILL_BASE = 'https://image.tmdb.org/t/p/w780'

@dataclass
class ContentDetails:
    """Data class for content details"""
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

def ensure_app_context(func):
    """Decorator to ensure Flask app context for database operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if has_app_context():
            return func(*args, **kwargs)
        else:
            # If we don't have an app context, try to get one
            try:
                if current_app:
                    with current_app.app_context():
                        return func(*args, **kwargs)
                else:
                    logger.warning(f"No app context available for {func.__name__}")
                    # Return safe defaults based on function name
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
                # Return safe defaults
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
    """
    Centralized slug management for all content types - Python 3.13 Compatible
    Handles movies, TV shows, anime, and persons
    """
    
    @staticmethod
    def generate_slug(title: str, year: Optional[int] = None, content_type: str = 'movie') -> str:
        """
        Generate URL-safe slug from title - Python 3.13 compatible version
        
        Args:
            title: Content title
            year: Release year (for movies)
            content_type: Type of content (movie, tv, anime, person)
            
        Returns:
            URL-safe slug string
            
        Examples:
            - "The Dark Knight" (2008) → "the-dark-knight-2008"
            - "Attack on Titan" → "attack-on-titan"
            - "Christopher Nolan" → "christopher-nolan"
        """
        if not title or not isinstance(title, str):
            return f"content-{int(time.time())}"
        
        try:
            # Clean the title first
            clean_title = str(title).strip()
            if not clean_title:
                return f"content-{int(time.time())}"
            
            # Use slugify with correct parameters - Python 3.13 compatible
            try:
                slug = slugify(clean_title, max_length=80, word_boundary=True, save_order=True)
            except Exception as slugify_error:
                logger.warning(f"Slugify failed for '{clean_title}': {slugify_error}")
                slug = None
            
            # If slugify returns empty or None, use manual fallback
            if not slug:
                # Manual cleaning as fallback - Python 3.13 compatible
                slug = clean_title.lower()
                # Remove special characters and replace with hyphens
                # Use re.ASCII flag for Python 3 compatibility
                slug = re.sub(r'[^\w\s-]', '', slug, flags=re.ASCII)
                slug = re.sub(r'[-\s]+', '-', slug)
                slug = slug.strip('-')
            
            # Ensure we have a valid slug
            if not slug or len(slug) < 1:
                # Generate based on content type and timestamp
                type_prefix = {
                    'movie': 'movie',
                    'tv': 'tv-show',
                    'anime': 'anime',
                    'person': 'person'
                }.get(content_type, 'content')
                return f"{type_prefix}-{int(time.time())}"
            
            # Truncate if too long (manual truncation since max_length isn't always supported)
            if len(slug) > 80:
                # Try to cut at word boundary
                truncated = slug[:80]
                last_dash = truncated.rfind('-')
                if last_dash > 40:  # Only use word boundary if we're not cutting too much
                    slug = truncated[:last_dash]
                else:
                    slug = truncated
            
            # Add year suffix for movies to avoid conflicts
            if year and content_type == 'movie' and isinstance(year, int):
                if 1800 <= year <= 2100:  # Reasonable year range
                    slug = f"{slug}-{year}"
            
            # Add content type prefix for disambiguation if needed
            if content_type == 'anime':
                # Only add prefix if not already present
                if not slug.startswith('anime-'):
                    slug = f"anime-{slug}"
            
            # Final length check after additions
            if len(slug) > 100:
                # Truncate at word boundary if possible
                parts = slug[:97].split('-')
                if len(parts) > 1:
                    slug = '-'.join(parts[:-1])
                else:
                    slug = slug[:97]
            
            return slug
            
        except Exception as e:
            logger.error(f"Error generating slug for title '{title}': {e}")
            # Ultimate fallback - guaranteed to work for Python 3.13
            # Use only ASCII characters for safety
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
            return safe_slug[:50]  # Limit length
            
    @staticmethod
    def generate_unique_slug(db, model, title: str, year: Optional[int] = None, 
                        content_type: str = 'movie', existing_id: Optional[int] = None) -> str:
        """
        Generate unique slug, adding suffix if necessary - Optimized for Python 3.13
        
        Args:
            db: Database session
            model: SQLAlchemy model class
            title: Content title
            year: Release year
            content_type: Type of content
            existing_id: ID to exclude from uniqueness check (for updates)
            
        Returns:
            Unique slug string
        """
        try:
            base_slug = SlugManager.generate_slug(title, year, content_type)
            
            # Ensure base_slug is valid
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
            max_attempts = 50  # Reduced from 100 for better performance
            
            # Keep trying until we find a unique slug
            while counter <= max_attempts:
                try:
                    # Check if slug exists - optimized query
                    query = db.session.query(model.id).filter_by(slug=slug)
                    
                    # Exclude existing record if updating
                    if existing_id:
                        query = query.filter(model.id != existing_id)
                    
                    # Use first() instead of exists() for better performance in some cases
                    exists = query.first() is not None
                    
                    if not exists:
                        break
                    
                    # Generate new slug with counter
                    slug = f"{base_slug}-{counter}"
                    counter += 1
                    
                except Exception as e:
                    logger.error(f"Error checking slug uniqueness: {e}")
                    # Fallback to timestamp-based slug
                    slug = f"{base_slug}-{int(time.time())}"
                    break
            
            # If we hit max attempts, use timestamp
            if counter > max_attempts:
                slug = f"{base_slug}-{int(time.time())}"
                logger.warning(f"Hit max attempts for slug generation, using timestamp: {slug}")
            
            return slug
            
        except Exception as e:
            logger.error(f"Error generating unique slug: {e}")
            # Emergency fallback
            type_prefix = {
                'movie': 'movie',
                'tv': 'tv-show',
                'anime': 'anime',
                'person': 'person'
            }.get(content_type, 'content')
            # Use timestamp and a small hash for uniqueness
            return f"{type_prefix}-{int(time.time())}-{abs(hash(str(title)))[:6]}"    
    
    @staticmethod
    def extract_info_from_slug(slug: str) -> Dict:
        """Extract potential title and year from slug"""
        try:
            if not slug:
                return {'title': 'Unknown', 'year': None, 'content_type': 'movie'}
            
            # Handle anime prefix
            content_type = 'movie'
            clean_slug = slug
            
            if slug.startswith('anime-'):
                content_type = 'anime'
                clean_slug = slug[6:]  # Remove 'anime-' prefix
            elif 'tv-show' in slug:
                content_type = 'tv'
            
            # Check if slug ends with a year (4 digits)
            year_match = re.search(r'-(\d{4})$', clean_slug)
            if year_match:
                year = int(year_match.group(1))
                title_slug = clean_slug[:year_match.start()]
                if content_type == 'movie' and 1800 <= year <= 2100:
                    return {
                        'title': title_slug.replace('-', ' ').title(),
                        'year': year,
                        'content_type': content_type
                    }
            
            # No year found or invalid year
            return {
                'title': clean_slug.replace('-', ' ').title(),
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
    def update_content_slug(db, content, force_update: bool = False) -> str:
        """
        Update content slug if missing or if forced - Optimized version
        
        Args:
            db: Database session
            content: Content object
            force_update: Whether to update even if slug exists
            
        Returns:
            Updated slug
        """
        try:
            # Check if update is needed
            if content.slug and not force_update:
                return content.slug
            
            # Determine content type
            content_type = getattr(content, 'content_type', 'movie')
            
            # Get title and year
            title = getattr(content, 'title', '') or getattr(content, 'name', '')
            year = None
            
            if hasattr(content, 'release_date') and content.release_date:
                try:
                    year = content.release_date.year
                except:
                    pass
            
            if not title:
                title = f"Content {getattr(content, 'id', 'Unknown')}"
            
            # Generate new slug
            new_slug = SlugManager.generate_unique_slug(
                db, 
                content.__class__, 
                title, 
                year, 
                content_type,
                existing_id=getattr(content, 'id', None)
            )
            
            # Update content
            content.slug = new_slug
            
            return new_slug
            
        except Exception as e:
            logger.error(f"Error updating content slug: {e}")
            # Fallback slug
            fallback = f"content-{getattr(content, 'id', int(time.time()))}"
            content.slug = fallback
            return fallback
    
    @staticmethod
    def migrate_slugs(db, models, batch_size: int = 50) -> Dict:
        """
        Migrate all content without slugs - Optimized for large datasets
        
        Args:
            db: Database session
            models: Dictionary of model classes
            batch_size: Number of items to process per batch
            
        Returns:
            Migration statistics
        """
        stats = {
            'content_updated': 0,
            'persons_updated': 0,
            'errors': 0,
            'total_processed': 0
        }
        
        try:
            # Migrate content in batches
            if 'Content' in models:
                Content = models['Content']
                
                # Process in smaller batches to avoid memory issues
                offset = 0
                batch_count = 0
                
                while True:
                    # Find content without slugs in batches
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
                    
                    # Commit batch
                    try:
                        db.session.commit()
                        logger.info(f"Committed content batch {batch_count}: {stats['content_updated']} total updated")
                    except Exception as e:
                        logger.error(f"Batch commit failed: {e}")
                        db.session.rollback()
                        stats['errors'] += len(content_items)
                    
                    offset += batch_size
                    
                    # Add small delay to prevent overwhelming the system
                    time.sleep(0.1)
            
            # Migrate persons in batches
            if 'Person' in models:
                Person = models['Person']
                
                offset = 0
                batch_count = 0
                
                while True:
                    # Find persons without slugs in batches
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
                    
                    # Commit batch
                    try:
                        db.session.commit()
                        logger.info(f"Committed person batch {batch_count}: {stats['persons_updated']} total updated")
                    except Exception as e:
                        logger.error(f"Person batch commit failed: {e}")
                        db.session.rollback()
                        stats['errors'] += len(person_items)
                    
                    offset += batch_size
                    
                    # Add small delay
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
    """Enhanced content service with comprehensive slug support"""
    
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.Content = models.get('Content')
        self.Person = models.get('Person')
    
    def save_content_from_tmdb(self, tmdb_data: Dict, content_type: str) -> Any:
        """Save content from TMDB with proper slug generation"""
        try:
            # Check if content already exists
            existing = None
            if self.Content and tmdb_data.get('id'):
                existing = self.Content.query.filter_by(tmdb_id=tmdb_data['id']).first()
            
            if existing:
                # Update existing content slug if missing
                if not existing.slug:
                    try:
                        SlugManager.update_content_slug(self.db, existing)
                        self.db.session.commit()
                    except Exception as e:
                        logger.warning(f"Failed to update existing content slug: {e}")
                return existing
            
            # Extract basic info
            title = tmdb_data.get('title') or tmdb_data.get('name') or 'Unknown Title'
            
            # Extract release date and year
            release_date = None
            year = None
            date_str = tmdb_data.get('release_date') or tmdb_data.get('first_air_date')
            if date_str:
                try:
                    release_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    year = release_date.year
                except:
                    pass
            
            # Generate unique slug
            slug = SlugManager.generate_unique_slug(
                self.db, self.Content, title, year, content_type
            )
            
            # Extract other data
            genres = []
            if 'genres' in tmdb_data:
                genres = [genre['name'] for genre in tmdb_data['genres']]
            elif 'genre_ids' in tmdb_data:
                genres = self._map_genre_ids(tmdb_data['genre_ids'])
            
            # Create content object
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
        """Save anime content with proper slug generation"""
        try:
            # Check if anime already exists
            existing = None
            if self.Content and anime_data.get('mal_id'):
                existing = self.Content.query.filter_by(mal_id=anime_data['mal_id']).first()
            
            if existing:
                # Update existing anime slug if missing
                if not existing.slug:
                    try:
                        SlugManager.update_content_slug(self.db, existing)
                        self.db.session.commit()
                    except Exception as e:
                        logger.warning(f"Failed to update existing anime slug: {e}")
                return existing
            
            # Extract basic info
            title = anime_data.get('title') or 'Unknown Anime'
            
            # Extract release date and year
            release_date = None
            year = None
            if anime_data.get('aired', {}).get('from'):
                try:
                    release_date = datetime.strptime(anime_data['aired']['from'][:10], '%Y-%m-%d').date()
                    year = release_date.year
                except:
                    pass
            
            # Generate unique slug for anime
            slug = SlugManager.generate_unique_slug(
                self.db, self.Content, title, year, 'anime'
            )
            
            # Extract genres
            genres = [genre['name'] for genre in anime_data.get('genres', [])]
            
            # Create anime content
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
        """Get or create person with proper slug"""
        try:
            # Check if person exists
            existing = None
            if self.Person and person_data.get('id'):
                existing = self.Person.query.filter_by(tmdb_id=person_data['id']).first()
            
            if existing:
                # Update slug if missing
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
            
            # Create new person
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
        """Map TMDB genre IDs to names"""
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids if gid in genre_map]

class DetailsService:
    """Main service for handling content details with comprehensive slug support"""
    
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
        
        # Initialize content service
        self.content_service = ContentService(db, models)
        
        # Setup HTTP session with retry
        self.session = self._create_http_session()
        
        # Thread pool for concurrent API calls - optimized for Python 3.13
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Initialize API keys from environment or config
        self._init_api_keys()
    
    def _init_api_keys(self):
        """Initialize API keys from app config"""
        global TMDB_API_KEY, OMDB_API_KEY, YOUTUBE_API_KEY
        
        try:
            if has_app_context() and current_app:
                TMDB_API_KEY = current_app.config.get('TMDB_API_KEY')
                OMDB_API_KEY = current_app.config.get('OMDB_API_KEY')
                YOUTUBE_API_KEY = current_app.config.get('YOUTUBE_API_KEY')
        except Exception as e:
            logger.warning(f"Could not initialize API keys: {e}")
    
    def _create_http_session(self) -> requests.Session:
        """Create HTTP session with retry strategy - optimized"""
        session = requests.Session()
        retry = Retry(
            total=2,  # Reduced retries for performance
            read=2,
            connect=2,
            backoff_factor=0.3,
            status_forcelist=(500, 502, 504)
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=5)  # Reduced pool size
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def get_details_by_slug(self, slug: str, user_id: Optional[int] = None) -> Optional[Dict]:
        """
        Main method to get all content details by slug
        Returns comprehensive details for the details page
        """
        try:
            # Ensure we have app context
            if not has_app_context():
                logger.warning("No app context available for get_details_by_slug")
                return None
            
            # Try cache first
            cache_key = f"details:slug:{slug}"
            if self.cache:
                try:
                    cached = self.cache.get(cache_key)
                    if cached:
                        logger.info(f"Cache hit for slug: {slug}")
                        # Add user-specific data if authenticated
                        if user_id:
                            cached = self._add_user_data(cached, user_id)
                        return cached
                except Exception as e:
                    logger.warning(f"Cache error for slug {slug}: {e}")
            
            # Find content by slug
            content = self.Content.query.filter_by(slug=slug).first()
            
            if not content:
                # Try fuzzy matching if exact slug not found
                content = self._find_content_fuzzy(slug)
                
                if not content:
                    logger.warning(f"Content not found for slug: {slug}")
                    return None
            
            # Ensure content has slug
            if not content.slug:
                try:
                    SlugManager.update_content_slug(self.db, content)
                    self.db.session.commit()
                except Exception as e:
                    logger.warning(f"Failed to update content slug: {e}")
            
            # Build comprehensive details
            details = self._build_content_details(content, user_id)
            
            # Cache the result (without user-specific data)
            if self.cache and details:
                try:
                    cache_data = details.copy()
                    # Remove user-specific fields before caching
                    cache_data.pop('user_data', None)
                    self.cache.set(cache_key, cache_data, timeout=1800)  # 30 minutes
                except Exception as e:
                    logger.warning(f"Cache set error for slug {slug}: {e}")
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting details for slug {slug}: {e}")
            return None
    
    def _find_content_fuzzy(self, slug: str) -> Optional[Any]:
        """Find content using fuzzy matching if exact slug not found"""
        try:
            # Extract potential title from slug
            info = SlugManager.extract_info_from_slug(slug)
            title = info['title']
            year = info['year']
            content_type = info['content_type']
            
            # Build query with all filters FIRST
            query = self.Content.query.filter(
                func.lower(self.Content.title).like(f"%{title.lower()}%")
            )
            
            # Filter by content type if detected
            if content_type:
                query = query.filter(self.Content.content_type == content_type)
            
            # Filter by year if available
            if year:
                query = query.filter(
                    func.extract('year', self.Content.release_date) == year
                )
            
            # Apply limit AFTER all filters
            results = query.limit(10).all()
            
            if results:
                # Return best match (highest similarity)
                best_match = max(results, key=lambda x: self._calculate_similarity(x.title, title))
                logger.info(f"Found fuzzy match for '{slug}': {best_match.title} (slug: {best_match.slug})")
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy content search: {e}")
            return None
            
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity score"""
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
        except Exception:
            return 0.0
    
    def _with_app_context(self, app, func, *args, **kwargs):
        """Execute function with Flask app context in thread"""
        if app:
            with app.app_context():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _build_content_details(self, content: Any, user_id: Optional[int] = None) -> Dict:
        """Build comprehensive content details - optimized for performance"""
        try:
            # Initialize API keys if not already done
            self._init_api_keys()
            
            # Capture the current app for thread context
            from flask import current_app
            app = current_app._get_current_object() if has_app_context() else None
            
            # Prepare futures for concurrent API calls - reduced for performance
            futures = {}
            
            with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers
                # External API calls - only essential ones
                if content.tmdb_id and TMDB_API_KEY:
                    futures['tmdb'] = executor.submit(self._fetch_tmdb_details, content.tmdb_id, content.content_type)
                if content.imdb_id and OMDB_API_KEY:
                    futures['omdb'] = executor.submit(self._fetch_omdb_details, content.imdb_id)
                
                # Internal data fetching with app context - reduced limits
                if app:
                    futures['cast_crew'] = executor.submit(self._with_app_context, app, self._get_cast_crew, content.id)
                    futures['reviews'] = executor.submit(self._with_app_context, app, self._get_reviews, content.id, 5)
                    futures['similar'] = executor.submit(self._with_app_context, app, self._get_similar_content, content.id, 8)
                    futures['gallery'] = executor.submit(self._with_app_context, app, self._get_gallery, content.id)
                else:
                    # Fallback to direct calls if no app context
                    futures['cast_crew'] = executor.submit(self._get_cast_crew, content.id)
                    futures['reviews'] = executor.submit(self._get_reviews, content.id, 5)
                    futures['similar'] = executor.submit(self._get_similar_content, content.id, 8)
                    futures['gallery'] = executor.submit(self._get_gallery, content.id)
                
                futures['trailer'] = executor.submit(self._get_trailer, content.title, content.content_type)
            
            # Collect results with error handling and timeouts
            tmdb_data = {}
            omdb_data = {}
            cast_crew = {'cast': [], 'crew': {'directors': [], 'writers': [], 'producers': []}}
            reviews = []
            similar = []
            gallery = {'posters': [], 'backdrops': [], 'stills': []}
            trailer = None
            
            # Process results with timeout protection
            try:
                if 'tmdb' in futures:
                    tmdb_data = futures['tmdb'].result(timeout=5) or {}
            except Exception as e:
                logger.warning(f"TMDB fetch error/timeout: {e}")
            
            try:
                if 'omdb' in futures:
                    omdb_data = futures['omdb'].result(timeout=5) or {}
            except Exception as e:
                logger.warning(f"OMDB fetch error/timeout: {e}")
            
            try:
                cast_crew = futures['cast_crew'].result(timeout=3) or cast_crew
            except Exception as e:
                logger.warning(f"Cast/crew fetch error/timeout: {e}")
            
            try:
                reviews = futures['reviews'].result(timeout=3) or []
            except Exception as e:
                logger.warning(f"Reviews fetch error/timeout: {e}")
            
            try:
                similar = futures['similar'].result(timeout=3) or []
            except Exception as e:
                logger.warning(f"Similar content fetch error/timeout: {e}")
            
            try:
                gallery = futures['gallery'].result(timeout=3) or gallery
            except Exception as e:
                logger.warning(f"Gallery fetch error/timeout: {e}")
            
            try:
                trailer = futures['trailer'].result(timeout=5)
            except Exception as e:
                logger.warning(f"Trailer fetch error/timeout: {e}")
            
            # Build synopsis
            synopsis = self._build_synopsis(content, tmdb_data, omdb_data)
            
            # Build ratings
            ratings = self._build_ratings(content, tmdb_data, omdb_data)
            
            # Build metadata
            metadata = self._build_metadata(content, tmdb_data, omdb_data)
            
            # Get streaming information
            streaming_info = self._get_streaming_info(content, tmdb_data)
            
            # Get seasons/episodes for TV shows
            seasons_episodes = None
            if content.content_type in ['tv', 'anime']:
                seasons_episodes = self._get_seasons_episodes(content, tmdb_data)
            
            # Construct final details object
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
            
            # Add user-specific data if authenticated
            if user_id:
                if app:
                    # Get user data with app context
                    with app.app_context():
                        details['user_data'] = self._get_user_data(content.id, user_id)
                else:
                    details['user_data'] = self._get_user_data(content.id, user_id)
            
            return details
            
        except Exception as e:
            logger.error(f"Error building content details: {e}")
            # Return minimal details to prevent complete failure
            return self._get_minimal_details(content)

    def _get_gallery(self, content_id: int) -> Dict:
        """Get content gallery (posters, backdrops, stills) with performance optimization"""
        try:
            gallery = {
                'posters': [],
                'backdrops': [],
                'stills': []
            }
            
            # Get content from database within app context
            content = None
            try:
                if has_app_context():
                    content = self.Content.query.get(content_id)
                else:
                    # If no app context, return basic gallery
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
                
                response = self.session.get(url, params=params, timeout=5)  # Reduced timeout
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Add posters - reduced limit
                    for img in data.get('posters', [])[:5]:  # Reduced from 10
                        gallery['posters'].append({
                            'url': self._format_image_url(img['file_path'], 'poster'),
                            'thumbnail': self._format_image_url(img['file_path'], 'thumbnail'),
                            'aspect_ratio': img.get('aspect_ratio'),
                            'width': img.get('width'),
                            'height': img.get('height')
                        })
                    
                    # Add backdrops - reduced limit
                    for img in data.get('backdrops', [])[:5]:  # Reduced from 10
                        gallery['backdrops'].append({
                            'url': self._format_image_url(img['file_path'], 'backdrop'),
                            'thumbnail': self._format_image_url(img['file_path'], 'still'),
                            'aspect_ratio': img.get('aspect_ratio'),
                            'width': img.get('width'),
                            'height': img.get('height')
                        })
                    
                    # Add stills for TV shows - reduced limit
                    if content.content_type == 'tv' and data.get('stills'):
                        for img in data.get('stills', [])[:5]:  # Reduced from 10
                            gallery['stills'].append({
                                'url': self._format_image_url(img['file_path'], 'still'),
                                'thumbnail': self._format_image_url(img['file_path'], 'thumbnail'),
                                'aspect_ratio': img.get('aspect_ratio'),
                                'width': img.get('width'),
                                'height': img.get('height')
                            })
            
            # Add default poster if no gallery images
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
        """Return minimal details as fallback"""
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
        """Fetch comprehensive details from TMDB with timeout"""
        try:
            if not TMDB_API_KEY:
                return {}
                
            endpoint = 'movie' if content_type == 'movie' else 'tv'
            url = f"{TMDB_BASE_URL}/{endpoint}/{tmdb_id}"
            
            params = {
                'api_key': TMDB_API_KEY,
                'append_to_response': 'videos,images,credits,similar,recommendations,reviews,external_ids,watch/providers,content_ratings,release_dates'
            }
            
            response = self.session.get(url, params=params, timeout=5)  # Reduced timeout
            
            if response.status_code == 200:
                return response.json()
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching TMDB details: {e}")
            return {}
    
    def _fetch_omdb_details(self, imdb_id: str) -> Dict:
        """Fetch details from OMDB with timeout"""
        try:
            if not OMDB_API_KEY:
                return {}
                
            params = {
                'apikey': OMDB_API_KEY,
                'i': imdb_id,
                'plot': 'full'
            }
            
            response = self.session.get(OMDB_BASE_URL, params=params, timeout=5)  # Reduced timeout
            
            if response.status_code == 200:
                return response.json()
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching OMDB details: {e}")
            return {}
    
    def _get_trailer(self, title: str, content_type: str) -> Optional[Dict]:
        """Get trailer information from YouTube with timeout"""
        try:
            if not YOUTUBE_API_KEY:
                return None
            
            # Search for trailer
            search_query = f"{title} official trailer"
            if content_type == 'anime':
                search_query = f"{title} anime trailer PV"
            
            url = f"{YOUTUBE_BASE_URL}/search"
            params = {
                'key': YOUTUBE_API_KEY,
                'q': search_query,
                'part': 'snippet',
                'type': 'video',
                'maxResults': 3,  # Reduced for performance
                'order': 'relevance'
            }
            
            response = self.session.get(url, params=params, timeout=5)  # Reduced timeout
            
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
        """Build comprehensive synopsis information"""
        try:
            synopsis = {
                'overview': content.overview or tmdb_data.get('overview', ''),
                'plot': omdb_data.get('Plot', content.overview or ''),
                'tagline': tmdb_data.get('tagline', ''),
                'content_warnings': [],
                'themes': [],
                'keywords': []
            }
            
            # Extract content warnings from ratings
            if tmdb_data.get('content_ratings'):
                ratings = tmdb_data['content_ratings'].get('results', [])
                us_rating = next((r for r in ratings if r['iso_3166_1'] == 'US'), None)
                if us_rating:
                    synopsis['content_warnings'].append({
                        'rating': us_rating.get('rating'),
                        'descriptors': us_rating.get('descriptors', [])
                    })
            
            # Extract keywords/themes
            if tmdb_data.get('keywords'):
                keywords = tmdb_data['keywords'].get('keywords', []) or tmdb_data['keywords'].get('results', [])
                synopsis['keywords'] = [kw['name'] for kw in keywords[:10]]
            
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
        """Build comprehensive ratings information"""
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
            
            # Parse IMDB rating safely
            try:
                imdb_rating = omdb_data.get('imdbRating', 'N/A')
                if imdb_rating and imdb_rating != 'N/A':
                    ratings['imdb']['score'] = float(imdb_rating)
                    ratings['imdb']['votes'] = omdb_data.get('imdbVotes', 'N/A')
            except (ValueError, TypeError):
                pass
            
            # Parse critic ratings from OMDB
            if omdb_data.get('Ratings'):
                for rating in omdb_data['Ratings']:
                    ratings['critics'].append({
                        'source': rating['Source'],
                        'value': rating['Value']
                    })
            
            # Calculate composite score
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
        """Build comprehensive metadata"""
        try:
            # Parse genres safely
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
            
            # Languages
            if tmdb_data.get('spoken_languages'):
                metadata['spoken_languages'] = [
                    {'code': lang['iso_639_1'], 'name': lang['name']} 
                    for lang in tmdb_data['spoken_languages']
                ]
            
            # Production info - limited for performance
            if tmdb_data.get('production_companies'):
                metadata['production_companies'] = [
                    {
                        'id': company['id'],
                        'name': company['name'],
                        'logo': self._format_image_url(company.get('logo_path'), 'logo') if company.get('logo_path') else None
                    }
                    for company in tmdb_data['production_companies'][:3]  # Reduced limit
                ]
            
            # Certifications
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
    def _get_cast_crew(self, content_id: int) -> Dict:
        """Get cast and crew information with performance optimization"""
        try:
            cast_crew = {
                'cast': [],
                'crew': {
                    'directors': [],
                    'writers': [],
                    'producers': []
                }
            }
            
            # Query from database if we have person data
            if self.ContentPerson and self.Person:
                # Get cast - reduced limit
                cast_entries = self.db.session.query(
                    self.ContentPerson, self.Person
                ).join(
                    self.Person
                ).filter(
                    self.ContentPerson.content_id == content_id,
                    self.ContentPerson.role_type == 'cast'
                ).order_by(
                    self.ContentPerson.order
                ).limit(15).all()  # Reduced from 20
                
                for cp, person in cast_entries:
                    # Ensure person has slug
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
                        'popularity': person.popularity
                    })
                
                # Get crew - reduced limit
                crew_entries = self.db.session.query(
                    self.ContentPerson, self.Person
                ).join(
                    self.Person
                ).filter(
                    self.ContentPerson.content_id == content_id,
                    self.ContentPerson.role_type == 'crew'
                ).limit(10).all()  # Reduced limit
                
                for cp, person in crew_entries:
                    # Ensure person has slug
                    if not person.slug:
                        try:
                            SlugManager.update_content_slug(self.db, person)
                        except Exception:
                            person.slug = f"person-{person.id}"
                    
                    crew_data = {
                        'id': person.id,
                        'name': person.name,
                        'job': cp.job,
                        'profile_path': self._format_image_url(person.profile_path, 'profile'),
                        'slug': person.slug
                    }
                    
                    if cp.department == 'Directing' or cp.job == 'Director':
                        cast_crew['crew']['directors'].append(crew_data)
                    elif cp.department == 'Writing' or cp.job in ['Writer', 'Screenplay']:
                        cast_crew['crew']['writers'].append(crew_data)
                    elif cp.department == 'Production' or 'Producer' in (cp.job or ''):
                        cast_crew['crew']['producers'].append(crew_data)
            
            return cast_crew
            
        except Exception as e:
            logger.error(f"Error getting cast/crew: {e}")
            return {'cast': [], 'crew': {'directors': [], 'writers': [], 'producers': []}}
    
    @ensure_app_context
    def get_person_details(self, person_slug: str) -> Optional[Dict]:
        """Get comprehensive person details by slug"""
        try:
            # Find person by slug
            person = self.Person.query.filter_by(slug=person_slug).first()
            
            if not person:
                # Try fuzzy matching
                person = self._find_person_fuzzy(person_slug)
                
                if not person:
                    return None
            
            # Ensure person has slug
            if not person.slug:
                try:
                    SlugManager.update_content_slug(self.db, person)
                    self.db.session.commit()
                except Exception as e:
                    logger.warning(f"Failed to update person slug: {e}")
            
            # Get filmography - with limit for performance
            filmography = self.db.session.query(
                self.ContentPerson, self.Content
            ).join(
                self.Content
            ).filter(
                self.ContentPerson.person_id == person.id
            ).order_by(
                self.Content.release_date.desc()
            ).limit(50).all()  # Added limit for performance
            
            # Organize filmography
            works = {
                'as_actor': [],
                'as_director': [],
                'as_writer': [],
                'as_producer': [],
                'upcoming': []
            }
            
            for cp, content in filmography:
                # Ensure content has slug
                if not content.slug:
                    try:
                        SlugManager.update_content_slug(self.db, content)
                    except Exception:
                        content.slug = f"content-{content.id}"
                
                work = {
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'year': content.release_date.year if content.release_date else None,
                    'content_type': content.content_type,
                    'poster_path': self._format_image_url(content.poster_path, 'poster'),
                    'rating': content.rating,
                    'character': cp.character,
                    'job': cp.job
                }
                
                # Check if upcoming
                if content.release_date and content.release_date > datetime.now().date():
                    works['upcoming'].append(work)
                elif cp.role_type == 'cast':
                    works['as_actor'].append(work)
                elif cp.department == 'Directing' or cp.job == 'Director':
                    works['as_director'].append(work)
                elif cp.department == 'Writing' or cp.job in ['Writer', 'Screenplay']:
                    works['as_writer'].append(work)
                elif 'Producer' in (cp.job or ''):
                    works['as_producer'].append(work)
            
            # Fetch additional data from TMDB if available
            tmdb_data = {}
            if person.tmdb_id and TMDB_API_KEY:
                try:
                    url = f"{TMDB_BASE_URL}/person/{person.tmdb_id}"
                    params = {
                        'api_key': TMDB_API_KEY,
                        'append_to_response': 'images,external_ids,combined_credits'
                    }
                    response = self.session.get(url, params=params, timeout=5)  # Reduced timeout
                    if response.status_code == 200:
                        tmdb_data = response.json()
                except Exception as e:
                    logger.error(f"Error fetching person from TMDB: {e}")
            
            # Parse also_known_as safely
            also_known_as = []
            try:
                if person.also_known_as:
                    also_known_as = json.loads(person.also_known_as)
            except (json.JSONDecodeError, TypeError):
                pass
            
            if not also_known_as and tmdb_data.get('also_known_as'):
                also_known_as = tmdb_data['also_known_as']
            
            return {
                'id': person.id,
                'slug': person.slug,
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
                'filmography': works,
                'images': self._get_person_images(tmdb_data),
                'social_media': self._get_person_social_media(tmdb_data),
                'total_works': len(filmography),
                'awards': []  # Could be expanded with awards data
            }
            
        except Exception as e:
            logger.error(f"Error getting person details: {e}")
            return None
    
    def _find_person_fuzzy(self, slug: str) -> Optional[Any]:
        """Find person using fuzzy matching"""
        try:
            # Extract name from slug
            name = slug.replace('-', ' ').title()
            
            # Build query and apply limit at the end
            results = self.Person.query.filter(
                func.lower(self.Person.name).like(f"%{name.lower()}%")
            ).limit(5).all()  # Apply limit after filter
            
            if results:
                # Return best match
                best_match = max(results, key=lambda x: self._calculate_similarity(x.name, name))
                logger.info(f"Found fuzzy match for person '{slug}': {best_match.name}")
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy person search: {e}")
            return None
            
    def _get_person_images(self, tmdb_data: Dict) -> List[str]:
        """Get person images from TMDB data"""
        try:
            images = []
            if tmdb_data.get('images', {}).get('profiles'):
                for img in tmdb_data['images']['profiles'][:5]:  # Reduced limit
                    images.append(self._format_image_url(img['file_path'], 'profile'))
            return images
        except Exception as e:
            logger.error(f"Error getting person images: {e}")
            return []
    
    def _get_person_social_media(self, tmdb_data: Dict) -> Dict:
        """Get person's social media links"""
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
            return social
        except Exception as e:
            logger.error(f"Error getting person social media: {e}")
            return {}
    
    @ensure_app_context
    def _get_reviews(self, content_id: int, limit: int = 5) -> List[Dict]:
        """Get user reviews for content with performance optimization"""
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
    def _get_similar_content(self, content_id: int, limit: int = 8) -> List[Dict]:
        """Get similar/recommended content with slug support and performance optimization"""
        try:
            similar = []
            
            # Get the current content
            content = self.Content.query.get(content_id)
            if not content:
                return []
            
            # Parse genres safely
            try:
                genres = json.loads(content.genres) if content.genres else []
            except (json.JSONDecodeError, TypeError):
                genres = []
            
            # Find similar content based on genres and type - optimized query
            query = self.Content.query.filter(
                self.Content.id != content_id,
                self.Content.content_type == content.content_type
            )
            
            # Filter by genres if available - use only primary genre for performance
            if genres:
                primary_genre = genres[0]
                query = query.filter(self.Content.genres.contains(primary_genre))
            
            # Order by rating and popularity with limit
            similar_content = query.order_by(
                self.Content.rating.desc(),
                self.Content.popularity.desc()
            ).limit(limit).all()
            
            for item in similar_content:
                # Ensure each item has a slug
                if not item.slug:
                    try:
                        SlugManager.update_content_slug(self.db, item)
                    except Exception:
                        item.slug = f"content-{item.id}"
                
                # Parse genres safely
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
    
    def _get_gallery(self, content_id: int) -> Dict:
        """Get content gallery (posters, backdrops, stills) with performance optimization"""
        try:
            gallery = {
                'posters': [],
                'backdrops': [],
                'stills': []
            }
            
            # Get from TMDB if available
            if not has_app_context():
                return gallery
            
            content = self.Content.query.get(content_id)
            if content and content.tmdb_id and TMDB_API_KEY:
                endpoint = 'movie' if content.content_type == 'movie' else 'tv'
                url = f"{TMDB_BASE_URL}/{endpoint}/{content.tmdb_id}/images"
                
                params = {
                    'api_key': TMDB_API_KEY
                }
                
                response = self.session.get(url, params=params, timeout=5)  # Reduced timeout
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Add posters - reduced limit
                    for img in data.get('posters', [])[:5]:  # Reduced from 10
                        gallery['posters'].append({
                            'url': self._format_image_url(img['file_path'], 'poster'),
                            'thumbnail': self._format_image_url(img['file_path'], 'thumbnail'),
                            'aspect_ratio': img.get('aspect_ratio'),
                            'width': img.get('width'),
                            'height': img.get('height')
                        })
                    
                    # Add backdrops - reduced limit
                    for img in data.get('backdrops', [])[:5]:  # Reduced from 10
                        gallery['backdrops'].append({
                            'url': self._format_image_url(img['file_path'], 'backdrop'),
                            'thumbnail': self._format_image_url(img['file_path'], 'still'),
                            'aspect_ratio': img.get('aspect_ratio'),
                            'width': img.get('width'),
                            'height': img.get('height')
                        })
                    
                    # Add stills for TV shows - reduced limit
                    if content.content_type == 'tv' and data.get('stills'):
                        for img in data.get('stills', [])[:5]:  # Reduced from 10
                            gallery['stills'].append({
                                'url': self._format_image_url(img['file_path'], 'still'),
                                'thumbnail': self._format_image_url(img['file_path'], 'thumbnail'),
                                'aspect_ratio': img.get('aspect_ratio'),
                                'width': img.get('width'),
                                'height': img.get('height')
                            })
            
            # Add default poster if no gallery images
            if not gallery['posters'] and content and content.poster_path:
                gallery['posters'].append({
                    'url': self._format_image_url(content.poster_path, 'poster'),
                    'thumbnail': self._format_image_url(content.poster_path, 'thumbnail')
                })
            
            return gallery
            
        except Exception as e:
            logger.error(f"Error getting gallery: {e}")
            return {'posters': [], 'backdrops': [], 'stills': []}
    
    def _get_streaming_info(self, content: Any, tmdb_data: Dict) -> Optional[Dict]:
        """Get streaming availability information"""
        try:
            streaming = {
                'rent': [],
                'buy': [],
                'stream': [],
                'free': []
            }
            
            if tmdb_data.get('watch/providers'):
                providers = tmdb_data['watch/providers'].get('results', {})
                
                # Get providers for user's region (defaulting to US)
                region_data = providers.get('US', {})
                
                if region_data.get('flatrate'):
                    for provider in region_data['flatrate'][:5]:  # Limit for performance
                        streaming['stream'].append({
                            'provider_id': provider['provider_id'],
                            'provider_name': provider['provider_name'],
                            'logo_path': self._format_image_url(provider.get('logo_path'), 'logo')
                        })
                
                if region_data.get('rent'):
                    for provider in region_data['rent'][:5]:  # Limit for performance
                        streaming['rent'].append({
                            'provider_id': provider['provider_id'],
                            'provider_name': provider['provider_name'],
                            'logo_path': self._format_image_url(provider.get('logo_path'), 'logo')
                        })
                
                if region_data.get('buy'):
                    for provider in region_data['buy'][:5]:  # Limit for performance
                        streaming['buy'].append({
                            'provider_id': provider['provider_id'],
                            'provider_name': provider['provider_name'],
                            'logo_path': self._format_image_url(provider.get('logo_path'), 'logo')
                        })
                
                if region_data.get('free'):
                    for provider in region_data['free'][:5]:  # Limit for performance
                        streaming['free'].append({
                            'provider_id': provider['provider_id'],
                            'provider_name': provider['provider_name'],
                            'logo_path': self._format_image_url(provider.get('logo_path'), 'logo')
                        })
            
            # Return None if no streaming options available
            if not any([streaming['rent'], streaming['buy'], streaming['stream'], streaming['free']]):
                return None
            
            return streaming
            
        except Exception as e:
            logger.error(f"Error getting streaming info: {e}")
            return None
    
    def _get_seasons_episodes(self, content: Any, tmdb_data: Dict) -> Optional[Dict]:
        """Get seasons and episodes for TV shows"""
        try:
            if content.content_type not in ['tv', 'anime']:
                return None
            
            seasons_data = {
                'season_count': tmdb_data.get('number_of_seasons', 0),
                'episode_count': tmdb_data.get('number_of_episodes', 0),
                'seasons': []
            }
            
            # Get season details - limit for performance
            if tmdb_data.get('seasons'):
                for season in tmdb_data['seasons'][:10]:  # Limit seasons
                    # Skip specials (season 0) unless it's the only season
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
            
            # Get next episode info
            if tmdb_data.get('next_episode_to_air'):
                next_ep = tmdb_data['next_episode_to_air']
                seasons_data['next_episode'] = {
                    'name': next_ep['name'],
                    'season_number': next_ep['season_number'],
                    'episode_number': next_ep['episode_number'],
                    'air_date': next_ep['air_date'],
                    'overview': next_ep.get('overview')
                }
            
            # Get last episode info
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
        """Get user-specific data for content with performance optimization"""
        try:
            user_data = {
                'in_watchlist': False,
                'in_favorites': False,
                'user_rating': None,
                'watch_status': None  # 'watching', 'completed', 'plan_to_watch', 'dropped'
            }
            
            if not self.UserInteraction:
                return user_data
            
            # Efficient single query for all interactions
            interactions = self.UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=content_id
            ).all()
            
            # Process all interactions at once
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
        """Add user-specific data to cached details"""
        try:
            if details and 'id' in details:
                details['user_data'] = self._get_user_data(details['id'], user_id)
            return details
        except Exception as e:
            logger.error(f"Error adding user data: {e}")
            return details
    
    def _format_image_url(self, path: str, image_type: str = 'poster') -> Optional[str]:
        """Format image URL based on type"""
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
        """Add a new review for content"""
        try:
            if not self.Review:
                return {
                    'success': False,
                    'message': 'Review system not available'
                }
            
            # Check if user already has a review for this content
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
                is_approved=False  # Reviews need moderation
            )
            
            self.db.session.add(review)
            self.db.session.commit()
            
            # Invalidate cache
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
        """Vote on review helpfulness"""
        try:
            if not self.Review:
                return False
            
            review = self.Review.query.get(review_id)
            if not review:
                return False
            
            # Update helpful count
            if is_helpful:
                review.helpful_count = (review.helpful_count or 0) + 1
            else:
                review.helpful_count = max(0, (review.helpful_count or 0) - 1)
            
            self.db.session.commit()
            
            # Invalidate cache
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
    
    # Slug management methods
    def migrate_all_slugs(self, batch_size: int = 50) -> Dict:
        """Migrate all content and persons without slugs"""
        return SlugManager.migrate_slugs(self.db, self.models, batch_size)
    
    def update_content_slug(self, content_id: int, force_update: bool = False) -> Optional[str]:
        """Update slug for specific content"""
        try:
            content = self.Content.query.get(content_id)
            if content:
                new_slug = SlugManager.update_content_slug(self.db, content, force_update)
                self.db.session.commit()
                
                # Invalidate cache
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
        """Get content by slug or ID"""
        try:
            if isinstance(identifier, int):
                return self.Content.query.get(identifier)
            else:
                content = self.Content.query.filter_by(slug=identifier).first()
                if not content:
                    # Try fuzzy matching
                    content = self._find_content_fuzzy(identifier)
                return content
        except Exception as e:
            logger.error(f"Error getting content by slug/id: {e}")
            return None

def init_details_service(app, db, models, cache):
    """Initialize the details service with app context"""
    with app.app_context():
        service = DetailsService(db, models, cache)
        return service