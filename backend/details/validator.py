"""
CineBrain Details Input Validation
Comprehensive validation for all inputs to the details module
"""

import re
import logging
from typing import Any, Optional, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class DetailsValidator:
    """Input validation for details module"""
    
    # Validation patterns
    SLUG_PATTERN = re.compile(r'^[a-z0-9\-]+$')
    TMDB_ID_PATTERN = re.compile(r'^\d+$')
    IMDB_ID_PATTERN = re.compile(r'^tt\d{7,8}$')
    YOUTUBE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{11}$')
    
    # Allowed values
    CONTENT_TYPES = {'movie', 'tv', 'anime', 'person'}
    SORT_OPTIONS = {'newest', 'oldest', 'helpful', 'rating_high', 'rating_low'}
    ALGORITHM_OPTIONS = {'genre_based', 'collaborative', 'hybrid', 'popularity'}
    
    @classmethod
    def validate_slug(cls, slug: str) -> bool:
        """Validate content/person slug"""
        try:
            if not slug or not isinstance(slug, str):
                return False
            
            if len(slug) < 2 or len(slug) > 150:
                return False
            
            if not cls.SLUG_PATTERN.match(slug.lower()):
                return False
            
            # Additional checks
            if slug.startswith('-') or slug.endswith('-'):
                return False
            
            if '--' in slug:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating slug '{slug}': {e}")
            return False
    
    @classmethod
    def validate_tmdb_id(cls, tmdb_id: Any) -> bool:
        """Validate TMDB ID"""
        try:
            if isinstance(tmdb_id, int):
                return tmdb_id > 0
            
            if isinstance(tmdb_id, str):
                return cls.TMDB_ID_PATTERN.match(tmdb_id) is not None
            
            return False
            
        except Exception:
            return False
    
    @classmethod
    def validate_imdb_id(cls, imdb_id: str) -> bool:
        """Validate IMDb ID"""
        try:
            if not imdb_id or not isinstance(imdb_id, str):
                return False
            
            return cls.IMDB_ID_PATTERN.match(imdb_id) is not None
            
        except Exception:
            return False
    
    @classmethod
    def validate_youtube_id(cls, youtube_id: str) -> bool:
        """Validate YouTube video ID"""
        try:
            if not youtube_id or not isinstance(youtube_id, str):
                return False
            
            return cls.YOUTUBE_ID_PATTERN.match(youtube_id) is not None
            
        except Exception:
            return False
    
    @classmethod
    def validate_content_type(cls, content_type: str) -> bool:
        """Validate content type"""
        try:
            return content_type in cls.CONTENT_TYPES
        except Exception:
            return False
    
    @classmethod
    def validate_pagination(cls, page: int, limit: int, max_limit: int = 100) -> bool:
        """Validate pagination parameters"""
        try:
            if not isinstance(page, int) or not isinstance(limit, int):
                return False
            
            if page < 1:
                return False
            
            if limit < 1 or limit > max_limit:
                return False
            
            return True
            
        except Exception:
            return False
    
    @classmethod
    def validate_limit(cls, limit: int, max_limit: int = 50) -> bool:
        """Validate limit parameter"""
        try:
            if not isinstance(limit, int):
                return False
            
            return 1 <= limit <= max_limit
            
        except Exception:
            return False
    
    @classmethod
    def validate_sort_option(cls, sort_by: str) -> bool:
        """Validate sort option"""
        try:
            return sort_by in cls.SORT_OPTIONS
        except Exception:
            return False
    
    @classmethod
    def validate_algorithm(cls, algorithm: str) -> bool:
        """Validate recommendation algorithm"""
        try:
            return algorithm in cls.ALGORITHM_OPTIONS
        except Exception:
            return False
    
    @classmethod
    def validate_rating(cls, rating: Any) -> bool:
        """Validate rating value"""
        try:
            if isinstance(rating, (int, float)):
                return 0 <= rating <= 10
            
            if isinstance(rating, str):
                try:
                    rating_float = float(rating)
                    return 0 <= rating_float <= 10
                except ValueError:
                    return False
            
            return False
            
        except Exception:
            return False
    
    @classmethod
    def validate_year(cls, year: Any) -> bool:
        """Validate year value"""
        try:
            if isinstance(year, int):
                return 1900 <= year <= 2030
            
            if isinstance(year, str):
                try:
                    year_int = int(year)
                    return 1900 <= year_int <= 2030
                except ValueError:
                    return False
            
            return False
            
        except Exception:
            return False
    
    @classmethod
    def validate_title(cls, title: str) -> bool:
        """Validate content title"""
        try:
            if not title or not isinstance(title, str):
                return False
            
            title = title.strip()
            if len(title) < 1 or len(title) > 500:
                return False
            
            # Check for minimum meaningful content
            if len(title.replace(' ', '').replace('-', '')) < 1:
                return False
            
            return True
            
        except Exception:
            return False
    
    @classmethod
    def validate_search_query(cls, query: str) -> bool:
        """Validate search query"""
        try:
            if not query or not isinstance(query, str):
                return False
            
            query = query.strip()
            if len(query) < 1 or len(query) > 200:
                return False
            
            # Check for malicious patterns
            malicious_patterns = [
                r'<script',
                r'javascript:',
                r'onclick=',
                r'onerror=',
                r'onload='
            ]
            
            query_lower = query.lower()
            for pattern in malicious_patterns:
                if re.search(pattern, query_lower):
                    return False
            
            return True
            
        except Exception:
            return False
    
    @classmethod
    def sanitize_input(cls, input_string: str) -> str:
        """Sanitize input string"""
        try:
            if not isinstance(input_string, str):
                return ""
            
            # Remove potential XSS characters
            sanitized = re.sub(r'[<>"\']', '', input_string)
            
            # Normalize whitespace
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()
            
            return sanitized[:1000]  # Limit length
            
        except Exception:
            return ""
    
    @classmethod
    def validate_boolean_param(cls, param: Any) -> bool:
        """Validate boolean parameter from query string"""
        try:
            if isinstance(param, bool):
                return True
            
            if isinstance(param, str):
                return param.lower() in {'true', 'false', '1', '0', 'yes', 'no'}
            
            if isinstance(param, int):
                return param in {0, 1}
            
            return False
            
        except Exception:
            return False
    
    @classmethod
    def convert_to_boolean(cls, param: Any) -> bool:
        """Convert parameter to boolean"""
        try:
            if isinstance(param, bool):
                return param
            
            if isinstance(param, str):
                return param.lower() in {'true', '1', 'yes'}
            
            if isinstance(param, int):
                return param == 1
            
            return False
            
        except Exception:
            return False