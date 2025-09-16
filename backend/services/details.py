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
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process  # Add this import for better fuzzy matching

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

@dataclass
class PersonDetails:
    """Data class for person details"""
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

class AdvancedFuzzyMatcher:
    """Advanced fuzzy matching for content discovery with multiple strategies"""
    
    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize title for better matching"""
        if not title:
            return ""
        
        # Convert to lowercase
        normalized = title.lower().strip()
        
        # Remove common punctuation and normalize spacing
        normalized = re.sub(r'[:\-–—_\.]', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    @staticmethod
    def generate_title_variations(title: str) -> List[str]:
        """Generate multiple variations of a title for better matching"""
        variations = [title]
        
        if not title:
            return variations
        
        # Original title
        original = title.strip()
        variations.append(original)
        
        # Normalized version
        normalized = AdvancedFuzzyMatcher.normalize_title(title)
        if normalized not in variations:
            variations.append(normalized)
        
        # Remove subtitle patterns (everything after colon, dash, etc.)
        main_title_patterns = [
            r'^([^:]+)',  # Everything before first colon
            r'^([^–]+)',  # Everything before em dash
            r'^([^—]+)',  # Everything before en dash
            r'^([^-]+)',  # Everything before hyphen
        ]
        
        for pattern in main_title_patterns:
            match = re.search(pattern, original)
            if match:
                main_title = match.group(1).strip()
                if main_title and len(main_title) > 3:  # Avoid too short titles
                    normalized_main = AdvancedFuzzyMatcher.normalize_title(main_title)
                    if normalized_main not in variations:
                        variations.append(normalized_main)
        
        # Add subtitle-only versions (everything after separator)
        subtitle_patterns = [
            r':\s*(.+)$',  # Everything after colon
            r'–\s*(.+)$',  # Everything after em dash
            r'—\s*(.+)$',  # Everything after en dash
            r'-\s*(.+)$',  # Everything after hyphen
        ]
        
        for pattern in subtitle_patterns:
            match = re.search(pattern, original)
            if match:
                subtitle = match.group(1).strip()
                if subtitle and len(subtitle) > 3:
                    normalized_subtitle = AdvancedFuzzyMatcher.normalize_title(subtitle)
                    if normalized_subtitle not in variations:
                        variations.append(normalized_subtitle)
        
        # Handle common franchise patterns
        franchise_patterns = [
            (r'^(.+?)\s*\d+$', r'\1'),  # "Movie 2" -> "Movie"
            (r'^(.+?)\s*part\s*\d+$', r'\1'),  # "Movie Part 2" -> "Movie"
            (r'^(.+?)\s*chapter\s*\d+$', r'\1'),  # "Movie Chapter 2" -> "Movie"
            (r'^(.+?)\s*episode\s*\d+$', r'\1'),  # "Movie Episode 2" -> "Movie"
        ]
        
        for pattern, replacement in franchise_patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                franchise_base = match.group(1).strip()
                if franchise_base and len(franchise_base) > 3:
                    if franchise_base not in variations:
                        variations.append(franchise_base)
        
        # Remove duplicates while preserving order
        unique_variations = []
        for var in variations:
            if var and var not in unique_variations:
                unique_variations.append(var)
        
        return unique_variations
    
    @staticmethod
    def calculate_similarity_score(title1: str, title2: str) -> float:
        """Calculate comprehensive similarity score using multiple algorithms"""
        if not title1 or not title2:
            return 0.0
        
        # Normalize both titles
        norm1 = AdvancedFuzzyMatcher.normalize_title(title1)
        norm2 = AdvancedFuzzyMatcher.normalize_title(title2)
        
        # Multiple similarity measures
        scores = []
        
        # 1. Basic sequence matching
        try:
            seq_score = SequenceMatcher(None, norm1, norm2).ratio()
            scores.append(seq_score)
        except:
            pass
        
        # 2. Fuzzy matching (if available)
        try:
            fuzz_score = fuzz.ratio(norm1, norm2) / 100.0
            scores.append(fuzz_score)
            
            # Token sort ratio (good for word order differences)
            token_score = fuzz.token_sort_ratio(norm1, norm2) / 100.0
            scores.append(token_score)
            
            # Partial ratio (good for substring matches)
            partial_score = fuzz.partial_ratio(norm1, norm2) / 100.0
            scores.append(partial_score)
        except:
            # Fallback if fuzzywuzzy is not available
            pass
        
        # 3. Word-based matching
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if words1 and words2:
            # Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard_score = intersection / union if union > 0 else 0
            scores.append(jaccard_score)
            
            # Word overlap ratio
            overlap_score = intersection / min(len(words1), len(words2))
            scores.append(overlap_score)
        
        # Return the maximum score (best match)
        return max(scores) if scores else 0.0

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
        """Extract potential title and year from slug with improved parsing"""
        try:
            if not slug:
                return {'title': 'Unknown', 'year': None, 'content_type': 'movie'}
            
            # Handle different content type patterns
            content_type = 'movie'  # default
            clean_slug = slug
            
            # Check for explicit type indicators
            if slug.startswith('anime-'):
                content_type = 'anime'
                clean_slug = slug[6:]  # Remove 'anime-' prefix
            elif slug.startswith('tv-') or 'tv-show' in slug or '-tv-' in slug:
                content_type = 'tv'
                clean_slug = slug.replace('tv-show-', '').replace('tv-', '').replace('-tv-', '-')
            elif slug.startswith('series-') or '-series-' in slug:
                content_type = 'tv'
                clean_slug = slug.replace('series-', '').replace('-series-', '-')
            elif slug.startswith('person-'):
                content_type = 'person'
                clean_slug = slug[7:]  # Remove 'person-' prefix
            
            # Check if slug ends with a year (4 digits)
            year_match = re.search(r'-(\d{4})$', clean_slug)
            if year_match:
                year = int(year_match.group(1))
                title_slug = clean_slug[:year_match.start()]
                
                # More flexible year range for upcoming content
                if 1900 <= year <= 2030:
                    title = title_slug.replace('-', ' ').title()
                    
                    # Handle common title patterns
                    title = re.sub(r'\bThe\b', 'The', title)  # Proper "The" capitalization
                    title = re.sub(r'\bAnd\b', 'and', title)  # Lowercase "and"
                    title = re.sub(r'\bOf\b', 'of', title)    # Lowercase "of"
                    title = re.sub(r'\bIn\b', 'in', title)    # Lowercase "in"
                    title = re.sub(r'\bTo\b', 'to', title)    # Lowercase "to"
                    title = re.sub(r'\bA\b', 'a', title)      # Lowercase "a"
                    title = re.sub(r'\bAn\b', 'an', title)    # Lowercase "an"
                    
                    return {
                        'title': title,
                        'year': year,
                        'content_type': content_type
                    }
            
            # No year found or invalid year
            title = clean_slug.replace('-', ' ').title()
            
            # Apply same title formatting
            title = re.sub(r'\bThe\b', 'The', title)
            title = re.sub(r'\bAnd\b', 'and', title)
            title = re.sub(r'\bOf\b', 'of', title)
            title = re.sub(r'\bIn\b', 'in', title)
            title = re.sub(r'\bTo\b', 'to', title)
            title = re.sub(r'\bA\b', 'a', title)
            title = re.sub(r'\bAn\b', 'an', title)
            
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
            if hasattr(content, '__tablename__') and content.__tablename__ == 'persons':
                content_type = 'person'
            
            # Get title and year
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
    """Main service for handling content details with enhanced fuzzy matching"""
    
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
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize API keys from environment or config
        self._init_api_keys()
        
        # Initialize fuzzy matcher
        self.fuzzy_matcher = AdvancedFuzzyMatcher()
    
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
        """
        Main method to get all content details by slug with improved fuzzy matching
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
                # Try enhanced fuzzy matching
                content = self._find_content_enhanced_fuzzy(slug)
                
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
    
    def _find_content_enhanced_fuzzy(self, slug: str) -> Optional[Any]:
        """Enhanced fuzzy matching with multiple strategies and better logging"""
        try:
            # Extract potential title from slug
            info = SlugManager.extract_info_from_slug(slug)
            search_title = info['title']
            search_year = info['year']
            content_type = info['content_type']
            
            logger.info(f"Fuzzy search for '{slug}' - extracted title: '{search_title}', year: {search_year}, type: {content_type}")
            
            # Generate multiple title variations for better matching
            title_variations = self.fuzzy_matcher.generate_title_variations(search_title)
            logger.info(f"Generated title variations: {title_variations}")
            
            # Strategy 1: Exact matches with variations
            candidates = []
            
            for variation in title_variations:
                # Try exact title match (case insensitive)
                exact_matches = self.Content.query.filter(
                    func.lower(self.Content.title) == variation.lower()
                ).all()
                candidates.extend(exact_matches)
                
                # Try with original title
                if hasattr(self.Content, 'original_title'):
                    original_matches = self.Content.query.filter(
                        func.lower(self.Content.original_title) == variation.lower()
                    ).all()
                    candidates.extend(original_matches)
            
            # Strategy 2: Partial matches with LIKE operator
            for variation in title_variations:
                if len(variation) > 3:  # Avoid too short searches
                    partial_matches = self.Content.query.filter(
                        func.lower(self.Content.title).like(f"%{variation.lower()}%")
                    ).limit(20).all()
                    candidates.extend(partial_matches)
            
            # Strategy 3: Word-based matching (each word as a separate LIKE)
            words = search_title.lower().split()
            if len(words) > 1:
                word_query = self.Content.query
                for word in words:
                    if len(word) > 2:  # Skip very short words
                        word_query = word_query.filter(
                            func.lower(self.Content.title).like(f"%{word}%")
                        )
                word_matches = word_query.limit(10).all()
                candidates.extend(word_matches)
            
            # Strategy 4: Genre + year matching (if we have year)
            if search_year and content_type:
                year_matches = self.Content.query.filter(
                    self.Content.content_type == content_type,
                    func.extract('year', self.Content.release_date) == search_year
                ).limit(20).all()
                candidates.extend(year_matches)
            
            # Strategy 5: Content type only matching
            if content_type:
                type_matches = self.Content.query.filter(
                    self.Content.content_type == content_type
                ).order_by(self.Content.popularity.desc()).limit(50).all()
                candidates.extend(type_matches)
            
            # Remove duplicates
            seen_ids = set()
            unique_candidates = []
            for candidate in candidates:
                if candidate.id not in seen_ids:
                    seen_ids.add(candidate.id)
                    unique_candidates.append(candidate)
            
            logger.info(f"Found {len(unique_candidates)} unique candidates")
            
            if not unique_candidates:
                logger.warning(f"No candidates found for fuzzy matching")
                return None
            
            # Calculate similarity scores for all candidates
            scored_candidates = []
            for candidate in unique_candidates:
                # Calculate similarity with original title
                title_score = self.fuzzy_matcher.calculate_similarity_score(
                    search_title, candidate.title
                )
                
                # Calculate similarity with original title if available
                original_score = 0
                if candidate.original_title:
                    original_score = self.fuzzy_matcher.calculate_similarity_score(
                        search_title, candidate.original_title
                    )
                
                # Use the higher score
                best_score = max(title_score, original_score)
                
                # Bonus for matching year
                year_bonus = 0
                if search_year and candidate.release_date:
                    if candidate.release_date.year == search_year:
                        year_bonus = 0.2
                    elif abs(candidate.release_date.year - search_year) <= 1:
                        year_bonus = 0.1  # Close years get smaller bonus
                
                # Bonus for matching content type
                type_bonus = 0.1 if candidate.content_type == content_type else 0
                
                # Popularity bonus (normalized)
                popularity_bonus = min(0.1, (candidate.popularity or 0) / 1000)
                
                # Final score
                final_score = best_score + year_bonus + type_bonus + popularity_bonus
                
                scored_candidates.append((candidate, final_score, {
                    'title_score': title_score,
                    'original_score': original_score,
                    'year_bonus': year_bonus,
                    'type_bonus': type_bonus,
                    'popularity_bonus': popularity_bonus
                }))
            
            # Sort by score (highest first)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Log top candidates for debugging
            logger.info("Top 5 fuzzy match candidates:")
            for i, (candidate, score, details) in enumerate(scored_candidates[:5]):
                logger.info(f"{i+1}. '{candidate.title}' ({candidate.release_date.year if candidate.release_date else 'No year'}) - Score: {score:.3f} - {details}")
            
            # Return best match if score is above threshold
            if scored_candidates:
                best_candidate, best_score, score_details = scored_candidates[0]
                
                # Dynamic threshold based on search quality
                min_threshold = 0.3  # Base threshold
                if search_year:
                    min_threshold = 0.25  # Lower threshold if we have year
                if len(title_variations) > 3:
                    min_threshold = 0.2   # Lower threshold if we have many variations
                
                if best_score >= min_threshold:
                    logger.info(f"Fuzzy match found: '{best_candidate.title}' (slug: {best_candidate.slug}) - Score: {best_score:.3f}")
                    return best_candidate
                else:
                    logger.warning(f"Best candidate score {best_score:.3f} below threshold {min_threshold}")
            
            logger.warning(f"No suitable fuzzy match found for '{slug}' with title '{search_title}' and year {search_year}")
            return None
            
        except Exception as e:
            logger.error(f"Error in enhanced fuzzy content search: {e}")
            return None
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity score using the enhanced fuzzy matcher"""
        return self.fuzzy_matcher.calculate_similarity_score(str1, str2)
    
    # Keep all other methods unchanged...
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
            
            # Prepare futures for concurrent API calls
            futures = {}
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # External API calls
                if content.tmdb_id and TMDB_API_KEY:
                    futures['tmdb'] = executor.submit(self._fetch_tmdb_details, content.tmdb_id, content.content_type)
                if content.imdb_id and OMDB_API_KEY:
                    futures['omdb'] = executor.submit(self._fetch_omdb_details, content.imdb_id)
                
                # Internal data fetching with app context
                if app:
                    futures['cast_crew'] = executor.submit(self._with_app_context, app, self._get_cast_crew, content.id)
                    futures['reviews'] = executor.submit(self._with_app_context, app, self._get_reviews, content.id, 10)
                    futures['similar'] = executor.submit(self._with_app_context, app, self._get_similar_content, content.id, 12)
                    futures['gallery'] = executor.submit(self._with_app_context, app, self._get_gallery, content.id)
                else:
                    # Fallback to direct calls if no app context
                    futures['cast_crew'] = executor.submit(self._get_cast_crew, content.id)
                    futures['reviews'] = executor.submit(self._get_reviews, content.id, 10)
                    futures['similar'] = executor.submit(self._get_similar_content, content.id, 12)
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

    # Include all other methods from the original DetailsService class...
    # (I'll include the key methods here, but all other methods remain the same)

    @ensure_app_context
    def _get_cast_crew(self, content_id: int) -> Dict:
        """Get cast and crew information with comprehensive data and fallback to TMDB"""
        try:
            cast_crew = {
                'cast': [],
                'crew': {
                    'directors': [],
                    'writers': [],
                    'producers': []
                }
            }
            
            # Get content to check for TMDB ID
            content = self.Content.query.get(content_id) if content_id else None
            if not content:
                logger.warning(f"Content not found for ID: {content_id}")
                return cast_crew
            
            # First, try to get from database
            if self.ContentPerson and self.Person:
                cast_crew = self._get_cast_crew_from_db(content_id)
                
                # If we have substantial data in database, return it
                total_cast_crew = len(cast_crew['cast']) + sum(len(crew_list) for crew_list in cast_crew['crew'].values())
                if total_cast_crew >= 5:  # Threshold for "enough" data
                    logger.info(f"Found {total_cast_crew} cast/crew members in database for content {content_id}")
                    return cast_crew
            
            # If no/insufficient data in database and we have TMDB ID, fetch from TMDB
            if content.tmdb_id and TMDB_API_KEY:
                logger.info(f"Fetching comprehensive cast/crew from TMDB for content {content_id}")
                cast_crew = self._fetch_and_save_all_cast_crew(content)
            
            return cast_crew
            
        except Exception as e:
            logger.error(f"Error getting cast/crew: {e}")
            return {'cast': [], 'crew': {'directors': [], 'writers': [], 'producers': []}}

    def _get_cast_crew_from_db(self, content_id: int) -> Dict:
        """Get ALL cast and crew from database without limits"""
        cast_crew = {
            'cast': [],
            'crew': {
                'directors': [],
                'writers': [],
                'producers': []
            }
        }
        
        try:
            # Get ALL cast from database (no limit)
            cast_entries = self.db.session.query(
                self.ContentPerson, self.Person
            ).join(
                self.Person
            ).filter(
                self.ContentPerson.content_id == content_id,
                self.ContentPerson.role_type == 'cast'
            ).order_by(
                self.ContentPerson.order.asc()
            ).all()  # No limit - get all cast
            
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
                    'popularity': person.popularity or 0,
                    'order': cp.order or 999
                })
            
            # Get ALL crew from database (no limit)
            crew_entries = self.db.session.query(
                self.ContentPerson, self.Person
            ).join(
                self.Person
            ).filter(
                self.ContentPerson.content_id == content_id,
                self.ContentPerson.role_type == 'crew'
            ).all()  # No limit - get all crew
            
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
                    'department': cp.department,
                    'profile_path': self._format_image_url(person.profile_path, 'profile'),
                    'slug': person.slug
                }
                
                # Categorize crew members
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
        """Fetch ALL cast/crew from TMDB and save to database"""
        cast_crew = {
            'cast': [],
            'crew': {
                'directors': [],
                'writers': [],
                'producers': []
            }
        }
        
        try:
            # Determine TMDB endpoint
            endpoint = 'movie' if content.content_type == 'movie' else 'tv'
            url = f"{TMDB_BASE_URL}/{endpoint}/{content.tmdb_id}/credits"
            
            params = {
                'api_key': TMDB_API_KEY
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Process ALL cast (remove any limits)
                cast_data = data.get('cast', [])  # Get ALL cast members
                logger.info(f"Processing {len(cast_data)} cast members for {content.title}")
                
                for i, cast_member in enumerate(cast_data):
                    try:
                        # Create or get person
                        person = self._get_or_create_person(cast_member)
                        if person:
                            # Create content-person relationship
                            content_person = self._get_or_create_content_person(
                                content.id, person.id, 'cast', 
                                character=cast_member.get('character'),
                                order=i
                            )
                            
                            # Add to response
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
                
                # Process ALL crew (remove any limits)
                crew_data = data.get('crew', [])  # Get ALL crew members
                logger.info(f"Processing {len(crew_data)} crew members for {content.title}")
                
                directors = []
                writers = []
                producers = []
                
                for crew_member in crew_data:
                    try:
                        # Create or get person
                        person = self._get_or_create_person(crew_member)
                        if person:
                            # Create content-person relationship
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
                            
                            # Categorize crew members (more comprehensive)
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
                
                # Store ALL crew members (no limits)
                cast_crew['crew']['directors'] = directors
                cast_crew['crew']['writers'] = writers
                cast_crew['crew']['producers'] = producers
                
                # Commit all changes
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
        """Get or create person from TMDB data with enhanced information"""
        try:
            tmdb_id = person_data.get('id')
            if not tmdb_id:
                return None
            
            # Check if person exists
            person = self.Person.query.filter_by(tmdb_id=tmdb_id).first()
            
            if not person:
                # Create new person
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
                self.db.session.flush()  # Get the ID
            else:
                # Update existing person if needed
                if not person.slug:
                    person.slug = SlugManager.generate_unique_slug(
                        self.db, self.Person, person.name, content_type='person',
                        existing_id=person.id
                    )
                
                # Update basic info if missing
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
        """Get or create content-person relationship"""
        try:
            # Check if relationship exists
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

    # Include all other methods from the original service...
    # (For brevity, I'm not copying all methods, but they should all be included)

    def _fetch_tmdb_details(self, tmdb_id: int, content_type: str) -> Dict:
        """Fetch comprehensive details from TMDB with timeout"""
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
        """Fetch details from OMDB with timeout"""
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
            
            # Production info
            if tmdb_data.get('production_companies'):
                metadata['production_companies'] = [
                    {
                        'id': company['id'],
                        'name': company['name'],
                        'logo': self._format_image_url(company.get('logo_path'), 'logo') if company.get('logo_path') else None
                    }
                    for company in tmdb_data['production_companies'][:5]
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
    def _get_reviews(self, content_id: int, limit: int = 10) -> List[Dict]:
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
    def _get_similar_content(self, content_id: int, limit: int = 12) -> List[Dict]:
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
            
            # Get season details
            if tmdb_data.get('seasons'):
                for season in tmdb_data['seasons'][:15]:  # Get more seasons
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
                
                response = self.session.get(url, params=params, timeout=8)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Add posters
                    for img in data.get('posters', [])[:15]:  # Get more images
                        gallery['posters'].append({
                            'url': self._format_image_url(img['file_path'], 'poster'),
                            'thumbnail': self._format_image_url(img['file_path'], 'thumbnail'),
                            'aspect_ratio': img.get('aspect_ratio'),
                            'width': img.get('width'),
                            'height': img.get('height')
                        })
                    
                    # Add backdrops
                    for img in data.get('backdrops', [])[:15]:  # Get more images
                        gallery['backdrops'].append({
                            'url': self._format_image_url(img['file_path'], 'backdrop'),
                            'thumbnail': self._format_image_url(img['file_path'], 'still'),
                            'aspect_ratio': img.get('aspect_ratio'),
                            'width': img.get('width'),
                            'height': img.get('height')
                        })
                    
                    # Add stills for TV shows
                    if content.content_type == 'tv' and data.get('stills'):
                        for img in data.get('stills', [])[:15]:  # Get more images
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
    
    @ensure_app_context
    def get_person_details(self, person_slug: str) -> Optional[Dict]:
        """Get comprehensive person details by slug with complete filmography and career information"""
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
            
            # Fetch complete person data from TMDB if available
            tmdb_data = {}
            if person.tmdb_id and TMDB_API_KEY:
                tmdb_data = self._fetch_complete_person_details(person.tmdb_id)
            
            # Get complete filmography from database
            filmography = self._get_complete_filmography(person.id)
            
            # Update person record with TMDB data if needed
            self._update_person_from_tmdb(person, tmdb_data)
            
            # Parse also_known_as safely
            also_known_as = []
            try:
                if person.also_known_as:
                    also_known_as = json.loads(person.also_known_as)
            except (json.JSONDecodeError, TypeError):
                pass
            
            if not also_known_as and tmdb_data.get('also_known_as'):
                also_known_as = tmdb_data['also_known_as']
            
            # Build comprehensive person details
            person_details = {
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
                'filmography': filmography,
                'images': self._get_person_images(tmdb_data),
                'social_media': self._get_person_social_media(tmdb_data),
                'total_works': self._calculate_total_works(filmography),
                'awards': [],  # Could be expanded with awards data
                'personal_info': self._build_personal_info(person, tmdb_data),
                'career_highlights': self._build_career_highlights(filmography),
                'external_ids': tmdb_data.get('external_ids', {}),
                'tmdb_id': person.tmdb_id
            }
            
            return person_details
            
        except Exception as e:
            logger.error(f"Error getting person details: {e}")
            return None

    def _fetch_complete_person_details(self, tmdb_id: int) -> Dict:
        """Fetch complete person details from TMDB including all career information"""
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

    def _update_person_from_tmdb(self, person: Any, tmdb_data: Dict):
        """Update person record with comprehensive TMDB data"""
        try:
            if not tmdb_data:
                return
            
            updated = False
            
            # Update biography
            if not person.biography and tmdb_data.get('biography'):
                person.biography = tmdb_data['biography']
                updated = True
            
            # Update birthday
            if not person.birthday and tmdb_data.get('birthday'):
                try:
                    person.birthday = datetime.strptime(tmdb_data['birthday'], '%Y-%m-%d').date()
                    updated = True
                except:
                    pass
            
            # Update deathday
            if not person.deathday and tmdb_data.get('deathday'):
                try:
                    person.deathday = datetime.strptime(tmdb_data['deathday'], '%Y-%m-%d').date()
                    updated = True
                except:
                    pass
            
            # Update place of birth
            if not person.place_of_birth and tmdb_data.get('place_of_birth'):
                person.place_of_birth = tmdb_data['place_of_birth']
                updated = True
            
            # Update also known as
            if not person.also_known_as and tmdb_data.get('also_known_as'):
                person.also_known_as = json.dumps(tmdb_data['also_known_as'])
                updated = True
            
            # Update popularity
            if tmdb_data.get('popularity') and tmdb_data['popularity'] > (person.popularity or 0):
                person.popularity = tmdb_data['popularity']
                updated = True
            
            if updated:
                self.db.session.commit()
                logger.info(f"Updated person {person.name} with TMDB data")
                
        except Exception as e:
            logger.error(f"Error updating person from TMDB: {e}")
            self.db.session.rollback()

    def _get_complete_filmography(self, person_id: int) -> Dict:
        """Get complete filmography for a person organized by role and type"""
        try:
            filmography = {
                'as_actor': [],
                'as_director': [],
                'as_writer': [],
                'as_producer': [],
                'upcoming': [],
                'by_year': {},
                'statistics': {
                    'total_projects': 0,
                    'years_active': 0,
                    'highest_rated': None,
                    'most_popular': None
                }
            }
            
            # Get ALL filmography entries (no limit)
            filmography_entries = self.db.session.query(
                self.ContentPerson, self.Content
            ).join(
                self.Content
            ).filter(
                self.ContentPerson.person_id == person_id
            ).order_by(
                self.Content.release_date.desc()
            ).all()  # Get ALL entries
            
            years = set()
            all_ratings = []
            all_popularity = []
            
            for cp, content in filmography_entries:
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
                    'popularity': content.popularity,
                    'character': cp.character,
                    'job': cp.job,
                    'department': cp.department,
                    'release_date': content.release_date.isoformat() if content.release_date else None
                }
                
                # Track statistics
                if content.release_date:
                    years.add(content.release_date.year)
                if content.rating:
                    all_ratings.append((content.rating, work))
                if content.popularity:
                    all_popularity.append((content.popularity, work))
                
                # Organize by year
                year_key = work['year'] or 'Unknown'
                if year_key not in filmography['by_year']:
                    filmography['by_year'][year_key] = []
                filmography['by_year'][year_key].append(work)
                
                # Check if upcoming
                if content.release_date and content.release_date > datetime.now().date():
                    filmography['upcoming'].append(work)
                elif cp.role_type == 'cast':
                    filmography['as_actor'].append(work)
                elif cp.department == 'Directing' or cp.job == 'Director':
                    filmography['as_director'].append(work)
                elif cp.department == 'Writing' or cp.job in ['Writer', 'Screenplay', 'Story']:
                    filmography['as_writer'].append(work)
                elif 'Producer' in (cp.job or ''):
                    filmography['as_producer'].append(work)
            
            # Calculate statistics
            filmography['statistics']['total_projects'] = len(filmography_entries)
            if years:
                filmography['statistics']['years_active'] = max(years) - min(years) + 1
            
            if all_ratings:
                filmography['statistics']['highest_rated'] = max(all_ratings, key=lambda x: x[0])[1]
            
            if all_popularity:
                filmography['statistics']['most_popular'] = max(all_popularity, key=lambda x: x[0])[1]
            
            logger.info(f"Retrieved complete filmography: {filmography['statistics']['total_projects']} projects")
            return filmography
            
        except Exception as e:
            logger.error(f"Error getting complete filmography: {e}")
            return {
                'as_actor': [], 'as_director': [], 'as_writer': [], 'as_producer': [],
                'upcoming': [], 'by_year': {}, 'statistics': {}
            }

    def _build_personal_info(self, person: Any, tmdb_data: Dict) -> Dict:
        """Build comprehensive personal information"""
        try:
            personal_info = {
                'age': None,
                'zodiac_sign': None,
                'nationality': None,
                'height': None,
                'awards_count': 0,
                'trivia': []
            }
            
            # Calculate age
            if person.birthday:
                today = datetime.now().date()
                if person.deathday:
                    end_date = person.deathday
                else:
                    end_date = today
                
                personal_info['age'] = end_date.year - person.birthday.year
                if end_date.month < person.birthday.month or (end_date.month == person.birthday.month and end_date.day < person.birthday.day):
                    personal_info['age'] -= 1
            
            # Determine nationality from place of birth
            if person.place_of_birth:
                # Simple nationality extraction (could be enhanced with a proper mapping)
                place_parts = person.place_of_birth.split(',')
                if len(place_parts) > 0:
                    country = place_parts[-1].strip()
                    personal_info['nationality'] = country
            
            return personal_info
            
        except Exception as e:
            logger.error(f"Error building personal info: {e}")
            return {}

    def _build_career_highlights(self, filmography: Dict) -> Dict:
        """Build career highlights and milestones"""
        try:
            highlights = {
                'debut_year': None,
                'breakthrough_role': None,
                'most_successful_decade': None,
                'collaboration_frequency': {},
                'genre_distribution': {},
                'career_phases': []
            }
            
            # Find debut year
            all_years = []
            for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                for work in filmography.get(role_type, []):
                    if work.get('year'):
                        all_years.append(work['year'])
            
            if all_years:
                highlights['debut_year'] = min(all_years)
            
            # Analyze decades
            decade_count = {}
            for year in all_years:
                decade = (year // 10) * 10
                decade_count[decade] = decade_count.get(decade, 0) + 1
            
            if decade_count:
                most_successful_decade = max(decade_count.items(), key=lambda x: x[1])
                highlights['most_successful_decade'] = f"{most_successful_decade[0]}s"
            
            return highlights
            
        except Exception as e:
            logger.error(f"Error building career highlights: {e}")
            return {}

    def _calculate_total_works(self, filmography: Dict) -> int:
        """Calculate total number of works across all roles"""
        try:
            total = 0
            for role_type in ['as_actor', 'as_director', 'as_writer', 'as_producer']:
                total += len(filmography.get(role_type, []))
            return total
        except:
            return 0

    def _find_person_fuzzy(self, slug: str) -> Optional[Any]:
        """Find person using enhanced fuzzy matching"""
        try:
            # Extract name from slug
            info = SlugManager.extract_info_from_slug(slug)
            name = info['title']  # For persons, title is the name
            
            logger.info(f"Person fuzzy search for '{slug}' - extracted name: '{name}'")
            
            # Generate name variations
            name_variations = self.fuzzy_matcher.generate_title_variations(name)
            logger.info(f"Generated name variations: {name_variations}")
            
            # Find candidates
            candidates = []
            
            for variation in name_variations:
                # Exact matches
                exact_matches = self.Person.query.filter(
                    func.lower(self.Person.name) == variation.lower()
                ).all()
                candidates.extend(exact_matches)
                
                # Partial matches
                if len(variation) > 3:
                    partial_matches = self.Person.query.filter(
                        func.lower(self.Person.name).like(f"%{variation.lower()}%")
                    ).limit(20).all()
                    candidates.extend(partial_matches)
            
            # Remove duplicates
            seen_ids = set()
            unique_candidates = []
            for candidate in candidates:
                if candidate.id not in seen_ids:
                    seen_ids.add(candidate.id)
                    unique_candidates.append(candidate)
            
            if not unique_candidates:
                logger.warning(f"No person candidates found for '{slug}'")
                return None
            
            # Score candidates
            scored_candidates = []
            for candidate in unique_candidates:
                score = self.fuzzy_matcher.calculate_similarity_score(name, candidate.name)
                
                # Popularity bonus
                popularity_bonus = min(0.1, (candidate.popularity or 0) / 1000)
                final_score = score + popularity_bonus
                
                scored_candidates.append((candidate, final_score))
            
            # Sort by score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Log top candidates
            logger.info(f"Top 3 person candidates:")
            for i, (candidate, score) in enumerate(scored_candidates[:3]):
                logger.info(f"{i+1}. '{candidate.name}' - Score: {score:.3f}")
            
            # Return best match if above threshold
            if scored_candidates:
                best_candidate, best_score = scored_candidates[0]
                if best_score >= 0.3:  # Lower threshold for persons
                    logger.info(f"Person fuzzy match found: '{best_candidate.name}' - Score: {best_score:.3f}")
                    return best_candidate
                else:
                    logger.warning(f"Best person candidate score {best_score:.3f} below threshold 0.3")
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy person search: {e}")
            return None

    def _get_person_images(self, tmdb_data: Dict) -> List[Dict]:
        """Get person images from TMDB data"""
        try:
            images = []
            if tmdb_data.get('images', {}).get('profiles'):
                for img in tmdb_data['images']['profiles'][:20]:  # Get more images
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
                    # Try enhanced fuzzy matching
                    content = self._find_content_enhanced_fuzzy(identifier)
                return content
        except Exception as e:
            logger.error(f"Error getting content by slug/id: {e}")
            return None

def init_details_service(app, db, models, cache):
    """Initialize the details service with app context"""
    with app.app_context():
        service = DetailsService(db, models, cache)
        return service