"""
CineBrain TMDB Service
Primary data source for movies and TV shows
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta

from .errors import APIError, handle_api_error
from .validator import DetailsValidator

logger = logging.getLogger(__name__)

class TMDBService:
    """TMDB API integration service"""
    
    BASE_URL = 'https://api.themoviedb.org/3'
    IMAGE_BASE_URL = 'https://image.tmdb.org/t/p'
    
    def __init__(self):
        self.api_key = os.environ.get('TMDB_API_KEY')
        if not self.api_key:
            raise ValueError("TMDB_API_KEY environment variable is required")
        
        self.session = self._create_session()
        self.last_request_time = {}
        self.rate_limit_delay = 0.25  # 4 requests per second max
        
        logger.info("TMDB Service initialized successfully")
    
    def _create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()
        
        retry = Retry(
            total=3,
            read=3,
            connect=3,
            backoff_factor=0.3,
            status_forcelist=(429, 500, 502, 503, 504)
        )
        
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    def _rate_limit(self):
        """Simple rate limiting"""
        import time
        current_time = time.time()
        if hasattr(self, '_last_request_time'):
            time_diff = current_time - self._last_request_time
            if time_diff < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - time_diff)
        self._last_request_time = time.time()
    
    @handle_api_error
    def get_content_details(self, tmdb_id: int, content_type: str) -> Optional[Dict]:
        """Get comprehensive content details from TMDB"""
        try:
            if not DetailsValidator.validate_tmdb_id(tmdb_id):
                raise APIError("TMDB", f"Invalid TMDB ID: {tmdb_id}")
            
            if not DetailsValidator.validate_content_type(content_type):
                content_type = 'movie'  # Default fallback
            
            endpoint = 'movie' if content_type == 'movie' else 'tv'
            url = f"{self.BASE_URL}/{endpoint}/{tmdb_id}"
            
            params = {
                'api_key': self.api_key,
                'append_to_response': (
                    'videos,images,credits,similar,recommendations,reviews,'
                    'external_ids,watch/providers,content_ratings,release_dates,'
                    'keywords,translations'
                )
            }
            
            self._rate_limit()
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched TMDB details for {endpoint} {tmdb_id}")
                return self._process_content_details(data, content_type)
            elif response.status_code == 404:
                logger.warning(f"TMDB content not found: {tmdb_id}")
                return None
            else:
                raise APIError("TMDB", f"HTTP {response.status_code}", response.status_code)
                
        except requests.exceptions.RequestException as e:
            raise APIError("TMDB", f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in TMDB service: {e}")
            raise APIError("TMDB", str(e))
    
    @handle_api_error
    def search_content(self, query: str, content_type: str = 'multi', page: int = 1) -> Optional[Dict]:
        """Search for content on TMDB"""
        try:
            if not DetailsValidator.validate_search_query(query):
                raise APIError("TMDB", f"Invalid search query: {query}")
            
            url = f"{self.BASE_URL}/search/{content_type}"
            params = {
                'api_key': self.api_key,
                'query': query,
                'page': page,
                'include_adult': False
            }
            
            self._rate_limit()
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise APIError("TMDB", f"Search failed: HTTP {response.status_code}", response.status_code)
                
        except requests.exceptions.RequestException as e:
            raise APIError("TMDB", f"Search request failed: {str(e)}")
    
    @handle_api_error
    def get_person_details(self, person_id: int) -> Optional[Dict]:
        """Get person details from TMDB"""
        try:
            if not DetailsValidator.validate_tmdb_id(person_id):
                raise APIError("TMDB", f"Invalid person ID: {person_id}")
            
            url = f"{self.BASE_URL}/person/{person_id}"
            params = {
                'api_key': self.api_key,
                'append_to_response': (
                    'images,external_ids,combined_credits,movie_credits,'
                    'tv_credits,tagged_images'
                )
            }
            
            self._rate_limit()
            response = self.session.get(url, params=params, timeout=12)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched TMDB person details for {person_id}")
                return data
            elif response.status_code == 404:
                return None
            else:
                raise APIError("TMDB", f"HTTP {response.status_code}", response.status_code)
                
        except requests.exceptions.RequestException as e:
            raise APIError("TMDB", f"Person request failed: {str(e)}")
    
    @handle_api_error
    def get_trending(self, content_type: str = 'all', time_window: str = 'day', page: int = 1) -> Optional[Dict]:
        """Get trending content from TMDB"""
        try:
            url = f"{self.BASE_URL}/trending/{content_type}/{time_window}"
            params = {
                'api_key': self.api_key,
                'page': page
            }
            
            self._rate_limit()
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise APIError("TMDB", f"Trending request failed: HTTP {response.status_code}", response.status_code)
                
        except requests.exceptions.RequestException as e:
            raise APIError("TMDB", f"Trending request failed: {str(e)}")
    
    def _process_content_details(self, data: Dict, content_type: str) -> Dict:
        """Process and normalize TMDB content details"""
        try:
            # Add processed fields
            processed_data = data.copy()
            
            # Normalize image URLs
            if 'poster_path' in data and data['poster_path']:
                processed_data['poster_url'] = f"{self.IMAGE_BASE_URL}/w500{data['poster_path']}"
            
            if 'backdrop_path' in data and data['backdrop_path']:
                processed_data['backdrop_url'] = f"{self.IMAGE_BASE_URL}/w1280{data['backdrop_path']}"
            
            # Process videos for trailers
            if 'videos' in data and 'results' in data['videos']:
                trailers = [
                    video for video in data['videos']['results']
                    if video.get('type') == 'Trailer' and video.get('site') == 'YouTube'
                ]
                processed_data['trailers'] = trailers
            
            # Process cast and crew
            if 'credits' in data:
                processed_data['cast'] = data['credits'].get('cast', [])[:20]  # Limit cast
                processed_data['crew'] = data['credits'].get('crew', [])
            
            # Process watch providers
            if 'watch/providers' in data:
                processed_data['streaming_providers'] = data['watch/providers']
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing TMDB data: {e}")
            return data
    
    def get_image_url(self, path: str, size: str = 'w500') -> Optional[str]:
        """Get full image URL"""
        if not path:
            return None
        return f"{self.IMAGE_BASE_URL}/{size}{path}"
    
    def check_health(self) -> Dict:
        """Check TMDB service health"""
        try:
            url = f"{self.BASE_URL}/configuration"
            params = {'api_key': self.api_key}
            
            response = self.session.get(url, params=params, timeout=5)
            
            return {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': response.elapsed.total_seconds(),
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }