"""
CineBrain Jikan Service
Anime data from MyAnimeList via Jikan API
"""

import logging
import requests
from typing import Dict, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

from .errors import APIError, handle_api_error
from .validator import DetailsValidator

logger = logging.getLogger(__name__)

class JikanService:
    """Jikan API integration for anime data"""
    
    BASE_URL = 'https://api.jikan.moe/v4'
    
    def __init__(self):
        self.session = self._create_session()
        self.last_request_time = None
        self.rate_limit_delay = 1.0  # Jikan has strict rate limits
        
        logger.info("Jikan Service initialized successfully")
    
    def _create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()
        
        retry = Retry(
            total=2,
            read=2,
            connect=2,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504)
        )
        
        adapter = HTTPAdapter(max_retries=retry, pool_connections=3, pool_maxsize=3)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    def _rate_limit(self):
        """Strict rate limiting for Jikan API"""
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    @handle_api_error
    def get_anime_details(self, mal_id: int) -> Optional[Dict]:
        """Get anime details from Jikan"""
        try:
            if not isinstance(mal_id, int) or mal_id <= 0:
                raise APIError("Jikan", f"Invalid MAL ID: {mal_id}")
            
            url = f"{self.BASE_URL}/anime/{mal_id}/full"
            
            self._rate_limit()
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    logger.info(f"Successfully fetched Jikan anime details for {mal_id}")
                    return self._process_anime_data(data['data'])
                return None
            elif response.status_code == 404:
                logger.warning(f"Anime not found on MAL: {mal_id}")
                return None
            elif response.status_code == 429:
                logger.warning("Jikan rate limit exceeded")
                time.sleep(2)  # Wait longer on rate limit
                return None
            else:
                raise APIError("Jikan", f"HTTP {response.status_code}", response.status_code)
                
        except requests.exceptions.RequestException as e:
            raise APIError("Jikan", f"Request failed: {str(e)}")
    
    @handle_api_error
    def search_anime(self, query: str, page: int = 1) -> Optional[Dict]:
        """Search for anime on Jikan"""
        try:
            if not DetailsValidator.validate_search_query(query):
                raise APIError("Jikan", f"Invalid search query: {query}")
            
            url = f"{self.BASE_URL}/anime"
            params = {
                'q': query,
                'page': page,
                'limit': 20,
                'order_by': 'score',
                'sort': 'desc'
            }
            
            self._rate_limit()
            response = self.session.get(url, params=params, timeout=12)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Jikan search rate limit exceeded")
                return None
            else:
                raise APIError("Jikan", f"Search failed: HTTP {response.status_code}", response.status_code)
                
        except requests.exceptions.RequestException as e:
            raise APIError("Jikan", f"Search request failed: {str(e)}")
    
    @handle_api_error
    def get_top_anime(self, anime_type: str = 'tv', page: int = 1) -> Optional[Dict]:
        """Get top anime from Jikan"""
        try:
            url = f"{self.BASE_URL}/top/anime"
            params = {
                'type': anime_type,
                'page': page
            }
            
            self._rate_limit()
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Jikan top anime rate limit exceeded")
                return None
            else:
                raise APIError("Jikan", f"Top anime request failed: HTTP {response.status_code}", response.status_code)
                
        except requests.exceptions.RequestException as e:
            raise APIError("Jikan", f"Top anime request failed: {str(e)}")
    
    @handle_api_error
    def get_anime_characters(self, mal_id: int) -> Optional[Dict]:
        """Get anime characters from Jikan"""
        try:
            url = f"{self.BASE_URL}/anime/{mal_id}/characters"
            
            self._rate_limit()
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            elif response.status_code == 429:
                return None
            else:
                raise APIError("Jikan", f"Characters request failed: HTTP {response.status_code}", response.status_code)
                
        except requests.exceptions.RequestException as e:
            raise APIError("Jikan", f"Characters request failed: {str(e)}")
    
    def _process_anime_data(self, data: Dict) -> Dict:
        """Process and normalize Jikan anime data"""
        try:
            processed = data.copy()
            
            # Normalize image URLs
            if 'images' in data and 'jpg' in data['images']:
                processed['poster_url'] = data['images']['jpg'].get('large_image_url')
                processed['poster_path'] = data['images']['jpg'].get('image_url')
            
            # Process genres
            if 'genres' in data:
                processed['genre_names'] = [genre['name'] for genre in data['genres']]
            
            # Process studios
            if 'studios' in data:
                processed['studio_names'] = [studio['name'] for studio in data['studios']]
            
            # Process aired dates
            if 'aired' in data and 'from' in data['aired']:
                processed['release_date'] = data['aired']['from']
            
            # Normalize rating
            if 'score' in data:
                processed['rating'] = data['score']
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing Jikan data: {e}")
            return data
    
    def check_health(self) -> Dict:
        """Check Jikan service health"""
        try:
            url = f"{self.BASE_URL}/anime/1"  # Cowboy Bebop - reliable test
            
            start_time = time.time()
            response = self.session.get(url, timeout=5)
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': response_time,
                'status_code': response.status_code,
                'rate_limited': response.status_code == 429
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }