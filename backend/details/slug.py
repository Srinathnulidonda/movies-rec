"""
CineBrain Slug Management
Comprehensive slug generation and management system
"""

import re
import time
import unicodedata
import logging
from typing import Optional, Tuple, Dict, Any
from slugify import slugify
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class SlugManager:
    """Advanced slug management for CineBrain content"""
    
    # Content type prefixes for better SEO and organization
    TYPE_PREFIXES = {
        'movie': 'movie',
        'tv': 'tv-show',
        'anime': 'anime',
        'person': 'person'
    }
    
    # Reserved slugs that shouldn't be used
    RESERVED_SLUGS = {
        'admin', 'api', 'search', 'trending', 'popular', 'new', 'details',
        'person', 'movie', 'tv', 'anime', 'login', 'register', 'about',
        'contact', 'privacy', 'terms', 'help', 'support', 'feedback'
    }
    
    @classmethod
    def generate_slug(cls, title: str, year: Optional[int] = None, 
                     content_type: str = 'movie', original_title: Optional[str] = None,
                     tmdb_id: Optional[int] = None) -> str:
        """Generate a slug from title and metadata"""
        try:
            if not title or not isinstance(title, str):
                fallback = f"{cls.TYPE_PREFIXES.get(content_type, 'content')}-{tmdb_id or int(time.time())}"
                logger.warning(f"Invalid title provided, using fallback: {fallback}")
                return fallback
            
            # Clean and extract year from title if present
            clean_title, extracted_year = cls._extract_year_from_title(title)
            working_year = year or extracted_year
            
            # Normalize title
            normalized_title = cls._normalize_title(clean_title)
            if not normalized_title:
                fallback = f"{cls.TYPE_PREFIXES.get(content_type, 'content')}-{tmdb_id or int(time.time())}"
                logger.warning(f"Title normalization failed, using fallback: {fallback}")
                return fallback
            
            # Generate base slug
            try:
                base_slug = slugify(
                    normalized_title,
                    max_length=80,
                    word_boundary=True,
                    save_order=True,
                    stopwords=['the', 'a', 'an']
                )
            except Exception as e:
                logger.warning(f"Slugify failed for '{normalized_title}': {e}")
                base_slug = cls._manual_slugify(normalized_title)
            
            if not base_slug or len(base_slug) < 2:
                base_slug = cls._manual_slugify(normalized_title)
            
            # Add type prefix for better organization
            type_prefix = cls.TYPE_PREFIXES.get(content_type, 'content')
            
            # Build final slug
            if content_type == 'person':
                # For persons, simpler slug structure
                final_slug = f"{type_prefix}-{base_slug}"
            else:
                # For content, add year if available and appropriate
                if working_year and cls._should_include_year(content_type, working_year):
                    final_slug = f"{type_prefix}-{base_slug}-{working_year}"
                else:
                    final_slug = f"{type_prefix}-{base_slug}"
            
            # Add TMDB ID for additional uniqueness if slug is short
            if tmdb_id and len(base_slug) < 10:
                final_slug = f"{final_slug}-{tmdb_id}"
            
            # Ensure reasonable length
            if len(final_slug) > 150:
                # Truncate base part but keep important components
                max_base_length = 150 - len(type_prefix) - 20  # Reserve space for year/id
                if working_year:
                    final_slug = f"{type_prefix}-{base_slug[:max_base_length]}-{working_year}"
                else:
                    final_slug = f"{type_prefix}-{base_slug[:max_base_length]}"
            
            # Validate final slug
            if cls._is_reserved_slug(final_slug):
                final_slug = f"{final_slug}-content"
            
            return final_slug
            
        except Exception as e:
            logger.error(f"Critical error generating slug for title '{title}': {e}")
            type_prefix = cls.TYPE_PREFIXES.get(content_type, 'content')
            return f"{type_prefix}-{tmdb_id or int(time.time())}-{abs(hash(str(title)))[:6]}"
    
    @classmethod
    def generate_unique_slug(cls, db, model, title: str, year: Optional[int] = None,
                           content_type: str = 'movie', existing_id: Optional[int] = None,
                           original_title: Optional[str] = None, tmdb_id: Optional[int] = None) -> str:
        """Generate a unique slug that doesn't conflict with existing entries"""
        try:
            base_slug = cls.generate_slug(title, year, content_type, original_title, tmdb_id)
            
            if not base_slug:
                type_prefix = cls.TYPE_PREFIXES.get(content_type, 'content')
                base_slug = f"{type_prefix}-{tmdb_id or int(time.time())}"
            
            slug = base_slug
            counter = 1
            max_attempts = 100
            
            while counter <= max_attempts:
                try:
                    # Check if slug exists
                    query = db.session.query(model.id).filter_by(slug=slug)
                    
                    if existing_id:
                        query = query.filter(model.id != existing_id)
                    
                    exists = query.first() is not None
                    
                    if not exists:
                        break
                    
                    # Generate next variant
                    if counter == 1:
                        # First attempt: add counter
                        slug = f"{base_slug}-{counter}"
                    else:
                        # Replace previous counter
                        slug = f"{base_slug}-{counter}"
                    
                    counter += 1
                    
                except Exception as e:
                    logger.error(f"Error checking slug uniqueness: {e}")
                    # Fallback to timestamp-based slug
                    slug = f"{base_slug}-{int(time.time())}"
                    break
            
            if counter > max_attempts:
                timestamp_slug = f"{base_slug}-{int(time.time())}"
                logger.warning(f"Hit max attempts for slug generation, using timestamp: {timestamp_slug}")
                return timestamp_slug
            
            return slug
            
        except Exception as e:
            logger.error(f"Critical error generating unique slug: {e}")
            type_prefix = cls.TYPE_PREFIXES.get(content_type, 'content')
            return f"{type_prefix}-{tmdb_id or int(time.time())}-{abs(hash(str(title)))[:6]}"
    
    @classmethod
    def extract_info_from_slug(cls, slug: str) -> Dict[str, Any]:
        """Extract information from a slug"""
        try:
            if not slug:
                return cls._default_slug_info()
            
            # Determine content type from prefix
            content_type = 'movie'  # default
            clean_slug = slug
            
            for prefix_type, prefix in cls.TYPE_PREFIXES.items():
                if slug.startswith(f"{prefix}-"):
                    content_type = prefix_type
                    clean_slug = slug[len(prefix) + 1:]
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
            
            # Extract TMDB ID if present (trailing number after year)
            tmdb_id = None
            tmdb_pattern = r'-(\d+)$'
            if not year_match:  # Only check for TMDB ID if no year found
                tmdb_match = re.search(tmdb_pattern, title_slug)
                if tmdb_match:
                    tmdb_id = int(tmdb_match.group(1))
                    title_slug = title_slug[:tmdb_match.start()]
            
            # Convert slug back to title
            title = cls._slug_to_title(title_slug)
            
            return {
                'title': title,
                'year': year,
                'content_type': content_type,
                'tmdb_id': tmdb_id,
                'clean_slug': title_slug
            }
            
        except Exception as e:
            logger.error(f"Error extracting info from slug '{slug}': {e}")
            return cls._default_slug_info(slug)
    
    @classmethod
    def update_content_slug(cls, db, content, force_update: bool = False) -> str:
        """Update slug for existing content"""
        try:
            if content.slug and not force_update and not cls._needs_slug_update(content.slug):
                return content.slug
            
            # Determine content type
            content_type = getattr(content, 'content_type', 'movie')
            if hasattr(content, '__tablename__') and content.__tablename__ == 'persons':
                content_type = 'person'
            
            # Get title and year
            title = getattr(content, 'title', '') or getattr(content, 'name', '')
            original_title = getattr(content, 'original_title', None)
            tmdb_id = getattr(content, 'tmdb_id', None)
            year = None
            
            # Extract year from dates
            if hasattr(content, 'release_date') and content.release_date:
                try:
                    year = content.release_date.year
                except (AttributeError, TypeError):
                    pass
            elif hasattr(content, 'birthday') and content.birthday:
                try:
                    year = content.birthday.year
                except (AttributeError, TypeError):
                    pass
            
            if not title:
                title = f"Content {getattr(content, 'id', 'Unknown')}"
            
            # Generate new slug
            new_slug = cls.generate_unique_slug(
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
    
    @classmethod
    def validate_slug(cls, slug: str) -> bool:
        """Validate slug format and content"""
        try:
            if not slug or not isinstance(slug, str):
                return False
            
            # Length check
            if len(slug) < 3 or len(slug) > 150:
                return False
            
            # Format check
            if not re.match(r'^[a-z0-9\-]+$', slug):
                return False
            
            # Structure check
            if slug.startswith('-') or slug.endswith('-') or '--' in slug:
                return False
            
            # Reserved slug check
            if cls._is_reserved_slug(slug):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating slug '{slug}': {e}")
            return False
    
    @classmethod
    def find_similar_slugs(cls, db, model, target_slug: str, limit: int = 5) -> List[str]:
        """Find similar existing slugs for suggestions"""
        try:
            # Get all slugs that start with similar pattern
            base_pattern = target_slug.split('-')[0] if '-' in target_slug else target_slug
            
            similar_slugs = db.session.query(model.slug).filter(
                model.slug.like(f"{base_pattern}%")
            ).limit(limit * 2).all()
            
            # Calculate similarity scores
            scored_slugs = []
            for (slug,) in similar_slugs:
                if slug and slug != target_slug:
                    similarity = SequenceMatcher(None, target_slug, slug).ratio()
                    scored_slugs.append((slug, similarity))
            
            # Sort by similarity and return top results
            scored_slugs.sort(key=lambda x: x[1], reverse=True)
            return [slug for slug, _ in scored_slugs[:limit]]
            
        except Exception as e:
            logger.error(f"Error finding similar slugs: {e}")
            return []
    
    @classmethod
    def _normalize_title(cls, title: str) -> str:
        """Normalize title for slug generation"""
        try:
            if not title or not isinstance(title, str):
                return ""
            
            # Clean and normalize
            clean_title = str(title).strip()
            if not clean_title:
                return ""
            
            # Unicode normalization
            normalized = unicodedata.normalize('NFKD', clean_title)
            normalized = normalized.encode('ascii', 'ignore').decode('ascii')
            
            # Remove special characters except basic punctuation
            normalized = re.sub(r'[^\w\s\-\']', '', normalized)
            
            # Normalize whitespace
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing title '{title}': {e}")
            return str(title).strip() if title else ""
    
    @classmethod
    def _extract_year_from_title(cls, title: str) -> Tuple[str, Optional[int]]:
        """Extract year from title"""
        try:
            year_patterns = [
                r'\((\d{4})\)$',      # (2023)
                r'\s(\d{4})$',        # 2023
                r'-(\d{4})$',         # -2023
                r'\[(\d{4})\]$'       # [2023]
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
    
    @classmethod
    def _manual_slugify(cls, text: str) -> str:
        """Manual slugify as fallback"""
        try:
            slug = text.lower()
            slug = re.sub(r'[^\w\s-]', '', slug)
            slug = re.sub(r'[-\s]+', '-', slug)
            slug = slug.strip('-')
            return slug[:100] if slug else ""
        except Exception:
            return ""
    
    @classmethod
    def _should_include_year(cls, content_type: str, year: int) -> bool:
        """Determine if year should be included in slug"""
        try:
            current_year = time.gmtime().tm_year
            
            # Always include for movies and anime
            if content_type in ['movie', 'anime']:
                return True
            
            # Include for TV shows if not current year
            if content_type == 'tv' and abs(year - current_year) > 1:
                return True
            
            # Don't include for persons
            if content_type == 'person':
                return False
            
            return False
            
        except Exception:
            return False
    
    @classmethod
    def _is_reserved_slug(cls, slug: str) -> bool:
        """Check if slug is reserved"""
        try:
            # Check exact matches
            if slug in cls.RESERVED_SLUGS:
                return True
            
            # Check if it starts with reserved words
            slug_parts = slug.split('-')
            if slug_parts[0] in cls.RESERVED_SLUGS:
                return True
            
            return False
            
        except Exception:
            return False
    
    @classmethod
    def _needs_slug_update(cls, current_slug: str) -> bool:
        """Check if current slug needs updating"""
        try:
            # Update if it's a generic/fallback slug
            if current_slug.startswith('content-') and current_slug.count('-') <= 1:
                return True
            
            # Update if it's very short
            if len(current_slug) < 5:
                return True
            
            # Update if it contains only numbers
            if re.match(r'^[a-z]+-\d+$', current_slug):
                return True
            
            return False
            
        except Exception:
            return True
    
    @classmethod
    def _slug_to_title(cls, slug: str) -> str:
        """Convert slug back to readable title"""
        try:
            title = slug.replace('-', ' ').title()
            
            # Fix common abbreviations
            title_fixes = {
                'Dc': 'DC', 'Mcu': 'MCU', 'Uk': 'UK', 'Us': 'US',
                'Tv': 'TV', 'Ai': 'AI', 'Fbi': 'FBI', 'Cia': 'CIA',
                'Ufc': 'UFC', 'Wwe': 'WWE', 'Nba': 'NBA', 'Nfl': 'NFL'
            }
            
            for wrong, correct in title_fixes.items():
                title = re.sub(f'\\b{wrong}\\b', correct, title)
            
            # Fix Roman numerals
            roman_numerals = ['Ii', 'Iii', 'Iv', 'Vi', 'Vii', 'Viii', 'Ix', 'Xi', 'Xii']
            for numeral in roman_numerals:
                title = re.sub(f'\\b{numeral}\\b', numeral.upper(), title)
            
            return title
            
        except Exception as e:
            logger.error(f"Error converting slug to title: {e}")
            return slug.replace('-', ' ').title()
    
    @classmethod
    def _default_slug_info(cls, slug: str = '') -> Dict[str, Any]:
        """Return default slug info structure"""
        return {
            'title': slug.replace('-', ' ').title() if slug else 'Unknown',
            'year': None,
            'content_type': 'movie',
            'tmdb_id': None,
            'clean_slug': slug
        }