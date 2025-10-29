# backend/services/slug_manager.py
import re
import logging
import time
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from functools import wraps
from slugify import slugify
from flask import has_app_context, current_app
from sqlalchemy import or_, and_

logger = logging.getLogger(__name__)

def ensure_app_context(func):
    """Decorator to ensure Flask app context is available"""
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
                    # Return appropriate defaults based on function name
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
                # Return appropriate defaults based on function name
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
    Comprehensive slug management for CineBrain platform.
    Handles slug generation, validation, and migration for all content types.
    """
    
    @staticmethod
    def normalize_title(title: str) -> str:
        """
        Normalize a title for slug generation.
        Handles unicode characters, special characters, and whitespace.
        """
        if not title or not isinstance(title, str):
            return ""
        
        try:
            # Clean and strip the title
            clean_title = str(title).strip()
            if not clean_title:
                return ""
            
            # Normalize unicode characters
            normalized = unicodedata.normalize('NFKD', clean_title)
            normalized = normalized.encode('ascii', 'ignore').decode('ascii')
            
            # Remove non-alphanumeric characters except spaces, hyphens, and apostrophes
            normalized = re.sub(r'[^\w\s\-\']', '', normalized)
            
            # Normalize whitespace
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing title '{title}': {e}")
            return str(title).strip()
    
    @staticmethod
    def extract_year_from_title(title: str) -> Tuple[str, Optional[int]]:
        """
        Extract year from title if present.
        Handles various year formats like (2024), [2024], -2024, etc.
        """
        try:
            # Year patterns to check
            year_patterns = [
                r'\((\d{4})\)$',    # (2024)
                r'\s(\d{4})$',      # 2024
                r'-(\d{4})$',       # -2024
                r'\[(\d{4})\]$',    # [2024]
                r'\{(\d{4})\}$',    # {2024}
                r'\.(\d{4})$',      # .2024
            ]
            
            for pattern in year_patterns:
                match = re.search(pattern, title)
                if match:
                    year = int(match.group(1))
                    # Validate year is reasonable
                    if 1900 <= year <= 2030:
                        clean_title = re.sub(pattern, '', title).strip()
                        # Remove trailing punctuation
                        clean_title = re.sub(r'[\s\-\.\,]+$', '', clean_title).strip()
                        return clean_title, year
            
            return title, None
            
        except Exception as e:
            logger.error(f"Error extracting year from title '{title}': {e}")
            return title, None
    
    @staticmethod
    def detect_content_type(title: str, original_title: str = None, genres: List[str] = None) -> str:
        """
        Detect content type based on title, original title, and genres.
        Returns: 'movie', 'tv', or 'anime'
        """
        try:
            title_lower = title.lower() if title else ""
            original_lower = original_title.lower() if original_title else ""
            
            # Anime indicators
            anime_indicators = [
                # Common anime terms
                'anime', 'manga', 'otaku', 'chan', 'kun', 'san', 'sama',
                'senpai', 'kouhai', 'sensei', 'dojo', 'ninja', 'samurai',
                'yokai', 'kami', 'kawaii', 'chibi', 'neko', 'baka',
                
                # Anime studios
                'studio ghibli', 'madhouse', 'pierrot', 'bones', 'shaft', 
                'trigger', 'mappa', 'wit studio', 'kyoani', 'toei', 
                'gainax', 'sunrise', 'a-1 pictures', 'production i.g',
                
                # Common anime title patterns
                'no hero academia', 'piece', 'ball', 'slayer', 'note',
                'titan', 'ghoul', 'bizarre', 'stone', 'saga'
            ]
            
            # TV indicators
            tv_indicators = [
                'series', 'season', 'episode', 'tv show', 'television',
                'mini-series', 'limited series', 'anthology', 'miniseries',
                'documentary series', 'reality', 'talk show', 'game show'
            ]
            
            # Check for anime indicators
            if any(indicator in title_lower or indicator in original_lower for indicator in anime_indicators):
                return 'anime'
            
            # Check genres for anime
            if genres and any(genre.lower() in ['animation', 'anime'] for genre in genres):
                # If it has Japanese origin indicators, it's likely anime
                if any(indicator in title_lower for indicator in ['chan', 'kun', 'san', 'sama']):
                    return 'anime'
            
            # Check for TV indicators
            if any(indicator in title_lower for indicator in tv_indicators):
                return 'tv'
            
            # Default to movie
            return 'movie'
            
        except Exception as e:
            logger.error(f"Error detecting content type: {e}")
            return 'movie'
    
    @staticmethod
    def generate_slug(title: str, year: Optional[int] = None, content_type: str = 'movie', 
                     original_title: str = None, tmdb_id: int = None) -> str:
        """
        Generate a slug from title with optional year and content type.
        Handles special cases and ensures valid slug format.
        """
        try:
            # Validate input
            if not title or not isinstance(title, str):
                fallback = f"content-{tmdb_id or int(time.time())}"
                logger.warning(f"Invalid title provided, using fallback: {fallback}")
                return fallback
            
            # Extract year if embedded in title
            clean_title, extracted_year = SlugManager.extract_year_from_title(title)
            
            # Use extracted year if no year provided
            if not year and extracted_year:
                year = extracted_year
            
            # Normalize the title
            normalized_title = SlugManager.normalize_title(clean_title)
            
            if not normalized_title:
                fallback = f"content-{tmdb_id or int(time.time())}"
                logger.warning(f"Title normalization failed, using fallback: {fallback}")
                return fallback
            
            # Generate slug using slugify library
            try:
                slug = slugify(normalized_title, max_length=70, word_boundary=True, save_order=True)
            except Exception as slugify_error:
                logger.warning(f"Slugify failed for '{normalized_title}': {slugify_error}")
                slug = SlugManager._manual_slugify(normalized_title)
            
            # Validate slug
            if not slug or len(slug) < 2:
                slug = SlugManager._manual_slugify(normalized_title)
            
            if not slug:
                # Use content type specific fallback
                type_prefix = {
                    'movie': 'movie',
                    'tv': 'tv-show',
                    'anime': 'anime',
                    'person': 'person'
                }.get(content_type, 'content')
                return f"{type_prefix}-{tmdb_id or int(time.time())}"
            
            # Add content type prefix for anime
            if content_type == 'anime' and not slug.startswith('anime-'):
                slug = f"anime-{slug}"
            
            # Add year for movies and anime (not for TV shows as they span multiple years)
            if year and content_type in ['movie', 'anime'] and isinstance(year, int):
                if 1900 <= year <= 2030:
                    slug = f"{slug}-{year}"
            
            # Ensure slug length is within limits
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
        """
        Manual slugification when slugify library fails.
        Basic but reliable slug generation.
        """
        try:
            slug = text.lower()
            # Remove non-alphanumeric characters except spaces and hyphens
            slug = re.sub(r'[^\w\s-]', '', slug)
            # Replace spaces and multiple hyphens with single hyphen
            slug = re.sub(r'[-\s]+', '-', slug)
            # Remove leading/trailing hyphens
            slug = slug.strip('-')
            # Limit length
            return slug[:70] if slug else ""
        except Exception:
            return ""
            
    @staticmethod
    def generate_unique_slug(db, model, title: str, year: Optional[int] = None, 
                           content_type: str = 'movie', existing_id: Optional[int] = None,
                           original_title: str = None, tmdb_id: int = None) -> str:
        """
        Generate a unique slug by checking database for conflicts.
        Adds numeric suffix if necessary to ensure uniqueness.
        """
        try:
            # Generate base slug
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
            
            # Check for uniqueness
            while counter <= max_attempts:
                try:
                    # Build query to check if slug exists
                    query = db.session.query(model.id).filter_by(slug=slug)
                    
                    # Exclude current record if updating
                    if existing_id:
                        query = query.filter(model.id != existing_id)
                    
                    exists = query.first() is not None
                    
                    if not exists:
                        break
                    
                    # Generate new slug with counter
                    if counter == 1:
                        # For first duplicate, try adding year if not already present
                        if year and content_type in ['movie', 'anime'] and str(year) not in slug:
                            slug = f"{base_slug.replace(f'-{year}', '')}-{year}-{counter}"
                        else:
                            slug = f"{base_slug}-{counter}"
                    else:
                        # Remove previous counter and add new one
                        base_part = base_slug.replace(f'-{counter-1}', '') if counter > 2 else base_slug
                        slug = f"{base_part}-{counter}"
                    
                    counter += 1
                    
                except Exception as e:
                    logger.error(f"Error checking slug uniqueness: {e}")
                    # Use timestamp to ensure uniqueness
                    slug = f"{base_slug}-{int(time.time())}"
                    break
            
            if counter > max_attempts:
                # Fallback: use timestamp
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
        """
        Extract information from a slug.
        Returns dict with title, year, and content_type.
        """
        try:
            if not slug:
                return {'title': 'Unknown', 'year': None, 'content_type': 'movie'}
            
            content_type = 'movie'
            clean_slug = slug
            
            # Detect content type from prefix
            type_patterns = [
                ('anime-', 'anime'),
                ('tv-show-', 'tv'),
                ('tv-', 'tv'),
                ('series-', 'tv'),
                ('person-', 'person'),
                ('movie-', 'movie')
            ]
            
            for prefix, ctype in type_patterns:
                if slug.startswith(prefix):
                    content_type = ctype
                    clean_slug = slug[len(prefix):]
                    break
            
            # Extract year if present
            year_pattern = r'-(\d{4})(?:-\d+)?$'
            year_match = re.search(year_pattern, clean_slug)
            year = None
            title_slug = clean_slug
            
            if year_match:
                potential_year = int(year_match.group(1))
                if 1900 <= potential_year <= 2030:
                    year = potential_year
                    title_slug = clean_slug[:year_match.start()]
            
            # Convert slug back to title
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
        """
        Convert slug back to human-readable title.
        Handles common abbreviations and formatting.
        """
        try:
            # Replace hyphens with spaces and title case
            title = slug.replace('-', ' ').title()
            
            # Common fixes for acronyms and special cases
            title_fixes = {
                # Acronyms
                'Dc': 'DC',
                'Mcu': 'MCU',
                'Uk': 'UK',
                'Us': 'US',
                'Usa': 'USA',
                'Tv': 'TV',
                'Ai': 'AI',
                'Fbi': 'FBI',
                'Cia': 'CIA',
                'Nsa': 'NSA',
                'Ufc': 'UFC',
                'Wwe': 'WWE',
                'Nba': 'NBA',
                'Nfl': 'NFL',
                'Mlb': 'MLB',
                'Nhl': 'NHL',
                'Cgi': 'CGI',
                'Vfx': 'VFX',
                'Dvd': 'DVD',
                'Hd': 'HD',
                '4k': '4K',
                
                # Name suffixes
                'Jr': 'Jr.',
                'Sr': 'Sr.',
                'Phd': 'PhD',
                'Md': 'MD',
                
                # Common words
                'And': 'and',
                'Of': 'of',
                'The': 'the',
                'In': 'in',
                'To': 'to',
                'For': 'for',
                'With': 'with',
                'From': 'from',
                'By': 'by',
                'At': 'at',
                'On': 'on',
                'As': 'as',
                'Or': 'or',
                'But': 'but',
                'An': 'an',
                'A': 'a'
            }
            
            # Apply fixes
            for wrong, correct in title_fixes.items():
                title = re.sub(f'\\b{wrong}\\b', correct, title)
            
            # Fix Roman numerals
            roman_numerals = {
                'Ii': 'II', 'Iii': 'III', 'Iv': 'IV', 'Vi': 'VI', 
                'Vii': 'VII', 'Viii': 'VIII', 'Ix': 'IX', 'Xi': 'XI', 
                'Xii': 'XII', 'Xiii': 'XIII', 'Xiv': 'XIV', 'Xv': 'XV',
                'Xx': 'XX', 'Xxi': 'XXI', 'Xxii': 'XXII', 'Xxiii': 'XXIII'
            }
            
            for wrong, correct in roman_numerals.items():
                title = re.sub(f'\\b{wrong}\\b', correct, title)
            
            # Ensure first letter is capitalized after fixing articles
            if title and title[0].islower():
                title = title[0].upper() + title[1:]
            
            return title
            
        except Exception as e:
            logger.error(f"Error converting slug to title: {e}")
            return slug.replace('-', ' ').title()
                    
    @staticmethod
    def update_content_slug(db, content, force_update: bool = False) -> str:
        """
        Update slug for existing content.
        Only updates if slug is missing or force_update is True.
        """
        try:
            # Check if update is needed
            if content.slug and not force_update:
                return content.slug
            
            # Determine content type
            content_type = getattr(content, 'content_type', 'movie')
            if hasattr(content, '__tablename__'):
                if content.__tablename__ == 'persons':
                    content_type = 'person'
                elif content.__tablename__ == 'content':
                    content_type = getattr(content, 'content_type', 'movie')
            
            # Get title/name
            title = getattr(content, 'title', '') or getattr(content, 'name', '')
            original_title = getattr(content, 'original_title', None)
            tmdb_id = getattr(content, 'tmdb_id', None)
            year = None
            
            # Extract year from release date or birthday
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
            
            # Fallback title if empty
            if not title:
                if content_type == 'person':
                    title = f"Person {getattr(content, 'id', 'Unknown')}"
                else:
                    title = f"Content {getattr(content, 'id', 'Unknown')}"
            
            # Generate unique slug
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
            
            # Update content
            content.slug = new_slug
            
            return new_slug
            
        except Exception as e:
            logger.error(f"Error updating content slug: {e}")
            # Generate fallback slug
            fallback = f"content-{getattr(content, 'id', int(time.time()))}"
            content.slug = fallback
            return fallback
    
    @staticmethod
    def migrate_slugs(db, models, batch_size: int = 50) -> Dict:
        """
        Migrate all content and person slugs in batches.
        Returns statistics about the migration.
        """
        stats = {
            'content_updated': 0,
            'persons_updated': 0,
            'errors': 0,
            'total_processed': 0,
            'duplicates_fixed': 0,
            'skipped': 0
        }
        
        try:
            # Migrate content slugs
            if 'Content' in models:
                Content = models['Content']
                
                offset = 0
                batch_count = 0
                
                while True:
                    # Query for content with missing or problematic slugs
                    content_items = Content.query.filter(
                        or_(
                            Content.slug == None, 
                            Content.slug == '', 
                            Content.slug.like('content-%'),
                            # Also catch old format slugs that might need updating
                            and_(
                                Content.slug.notlike('%-20%'),  # Missing year
                                Content.release_date != None
                            )
                        )
                    ).offset(offset).limit(batch_size).all()
                    
                    if not content_items:
                        break
                    
                    batch_count += 1
                    logger.info(f"Processing content batch {batch_count}, items: {len(content_items)}")
                    
                    for i, content in enumerate(content_items):
                        try:
                            old_slug = content.slug
                            
                            # Force update if slug is generic or missing year
                            force = (not old_slug or 
                                   old_slug.startswith('content-') or
                                   (content.release_date and str(content.release_date.year) not in (old_slug or '')))
                            
                            SlugManager.update_content_slug(db, content, force_update=force)
                            
                            if old_slug != content.slug:
                                stats['content_updated'] += 1
                                if old_slug and old_slug.startswith('content-'):
                                    stats['duplicates_fixed'] += 1
                                logger.debug(f"Updated slug: '{old_slug}' -> '{content.slug}'")
                            else:
                                stats['skipped'] += 1
                            
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
                    time.sleep(0.1)  # Small delay to prevent database overload
            
            # Migrate person slugs
            if 'Person' in models:
                Person = models['Person']
                
                offset = 0
                batch_count = 0
                
                while True:
                    # Query for persons with missing or problematic slugs
                    person_items = Person.query.filter(
                        or_(
                            Person.slug == None, 
                            Person.slug == '', 
                            Person.slug.like('person-%')
                        )
                    ).offset(offset).limit(batch_size).all()
                    
                    if not person_items:
                        break
                    
                    batch_count += 1
                    logger.info(f"Processing person batch {batch_count}, items: {len(person_items)}")
                    
                    for i, person in enumerate(person_items):
                        try:
                            old_slug = person.slug
                            name = getattr(person, 'name', '')
                            
                            if not name:
                                name = f"Person {getattr(person, 'id', 'Unknown')}"
                            
                            new_slug = SlugManager.generate_unique_slug(
                                db, 
                                Person, 
                                name, 
                                content_type='person',
                                existing_id=getattr(person, 'id', None),
                                tmdb_id=getattr(person, 'tmdb_id', None)
                            )
                            
                            person.slug = new_slug
                            
                            if old_slug != new_slug:
                                stats['persons_updated'] += 1
                                logger.debug(f"Updated person slug: '{old_slug}' -> '{new_slug}'")
                            
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
                    time.sleep(0.1)  # Small delay
            
            stats['total_processed'] = stats['content_updated'] + stats['persons_updated'] + stats['skipped']
            logger.info(f"Slug migration completed: {stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in slug migration: {e}")
            db.session.rollback()
            stats['errors'] += 1
            return stats
    
    @staticmethod
    def validate_slug(slug: str) -> bool:
        """
        Validate that a slug meets CineBrain standards.
        """
        if not slug or not isinstance(slug, str):
            return False
        
        # Check length
        if len(slug) < 2 or len(slug) > 150:
            return False
        
        # Check format (alphanumeric, hyphens only)
        if not re.match(r'^[a-z0-9]+(?:-[a-z0-9]+)*$', slug):
            return False
        
        # Check for double hyphens
        if '--' in slug:
            return False
        
        # Check for leading/trailing hyphens
        if slug.startswith('-') or slug.endswith('-'):
            return False
        
        return True
    
    @staticmethod
    def suggest_slug_corrections(slug: str) -> List[str]:
        """
        Suggest corrections for invalid slugs.
        """
        suggestions = []
        
        if not slug:
            return suggestions
        
        # Basic cleanup
        cleaned = slug.lower().strip()
        cleaned = re.sub(r'[^\w\s-]', '', cleaned)
        cleaned = re.sub(r'[-\s]+', '-', cleaned)
        cleaned = cleaned.strip('-')
        
        if cleaned and cleaned != slug:
            suggestions.append(cleaned)
        
        # Try with slugify
        try:
            slugified = slugify(slug)
            if slugified and slugified not in suggestions:
                suggestions.append(slugified)
        except:
            pass
        
        # Try extracting info and regenerating
        info = SlugManager.extract_info_from_slug(slug)
        if info['title'] != 'Unknown':
            regenerated = SlugManager.generate_slug(
                info['title'], 
                info['year'], 
                info['content_type']
            )
            if regenerated and regenerated not in suggestions:
                suggestions.append(regenerated)
        
        return suggestions[:15]  # Return max 5 suggestions

# Export public interface
__all__ = ['SlugManager', 'ensure_app_context']