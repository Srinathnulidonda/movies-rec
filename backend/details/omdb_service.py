"""
CineBrain OMDb Service
Supplementary data source for ratings and additional metadata
"""

import os
import logging
import requests
from typing import Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import APIError, handle_api_error
from .validator import DetailsValidator

logger = logging.getLogger(__name__)

class OMDBService:
    """OMDb API integration service"""
    
    BASE_URL = 'http://www.omdbapi.com/'
    
    def __init__(self):
        self.api_key = os.environ.get('OMDB_API_KEY')
        if not self.api_key:
            raise ValueError("OMDB_API_KEY environment variable is required")
        
        self.session = self._create_session()
        logger.info("OMDb Service initialized successfully")
    
    def _create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()
        
        retry = Retry(
            total=2,
            read=2,
            connect=2,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504)
        )
        
        adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=5)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    @handle_api_error
    def get_content_details(self, imdb_id: str) -> Optional[Dict]:
        """Get content details from OMDb using IMDb ID"""
        try:
            if not DetailsValidator.validate_imdb_id(imdb_id):
                logger.warning(f"Invalid IMDb ID format: {imdb_id}")
                return None
            
            params = {
                'apikey': self.api_key,
                'i': imdb_id,
                'plot': 'full',
                'r': 'json'
            }
            
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Response') == 'True':
                    logger.info(f"Successfully fetched OMDb details for {imdb_id}")
                    return self._process_omdb_data(data)
                else:
                    logger.warning(f"OMDb returned error for {imdb_id}: {data.get('Error')}")
                    return None
            else:
                raise APIError("OMDb", f"HTTP {response.status_code}", response.status_code)
                
        except requests.exceptions.RequestException as e:
            raise APIError("OMDb", f"Request failed: {str(e)}")
    
    @handle_api_error
    def search_by_title(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Search content by title in OMDb"""
        try:
            if not DetailsValidator.validate_title(title):
                raise APIError("OMDb", f"Invalid title: {title}")
            
            params = {
                'apikey': self.api_key,
                't': title,
                'plot': 'full',
                'r': 'json'
            }
            
            if year and DetailsValidator.validate_year(year):
                params['y'] = str(year)
            
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Response') == 'True':
                    return self._process_omdb_data(data)
                else:
                    logger.debug(f"OMDb search failed for '{title}': {data.get('Error')}")
                    return None
            else:
                raise APIError("OMDb", f"HTTP {response.status_code}", response.status_code)
                
        except requests.exceptions.RequestException as e:
            raise APIError("OMDb", f"Search request failed: {str(e)}")
    
    def _process_omdb_data(self, data: Dict) -> Dict:
        """Process and normalize OMDb data"""
        try:
            processed = {
                'imdb_rating': self._parse_rating(data.get('imdbRating')),
                'imdb_votes': data.get('imdbVotes', 'N/A'),
                'metascore': self._parse_rating(data.get('Metascore')),
                'plot': data.get('Plot', ''),
                'awards': data.get('Awards', ''),
                'box_office': data.get('BoxOffice', ''),
                'production': data.get('Production', ''),
                'website': data.get('Website', ''),
                'ratings': self._parse_ratings(data.get('Ratings', [])),
                'runtime': data.get('Runtime', ''),
                'rated': data.get('Rated', ''),
                'director': data.get('Director', ''),
                'writer': data.get('Writer', ''),
                'actors': data.get('Actors', ''),
                'country': data.get('Country', ''),
                'language': data.get('Language', ''),
                'dvd': data.get('DVD', ''),
                'type': data.get('Type', '')
            }
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing OMDb data: {e}")
            return data
    
    def _parse_rating(self, rating_str: str) -> Optional[float]:
        """Parse rating string to float"""
        try:
            if not rating_str or rating_str in ['N/A', 'n/a']:
                return None
            
            # Remove any non-numeric characters except decimal point
            import re
            clean_rating = re.sub(r'[^\d.]', '', rating_str)
            
            if clean_rating:
                return float(clean_rating)
            
            return None
            
        except (ValueError, TypeError):
            return None
    
    def _parse_ratings(self, ratings_list: List[Dict]) -> List[Dict]:
        """Parse and normalize ratings from different sources"""
        try:
            normalized_ratings = []
            
            for rating in ratings_list:
                source = rating.get('Source', '')
                value = rating.get('Value', '')
                
                normalized_rating = {
                    'source': source,
                    'value': value,
                    'normalized_score': self._normalize_rating_score(source, value)
                }
                
                normalized_ratings.append(normalized_rating)
            
            return normalized_ratings
            
        except Exception as e:
            logger.error(f"Error parsing ratings: {e}")
            return ratings_list
    
    def _normalize_rating_score(self, source: str, value: str) -> Optional[float]:
        """Normalize different rating scales to 0-10"""
        try:
            if not value or value == 'N/A':
                return None
            
            source_lower = source.lower()
            
            if 'imdb' in source_lower:
                # IMDb: already 0-10 scale
                return self._parse_rating(value)
            
            elif 'rotten tomatoes' in source_lower:
                # Rotten Tomatoes: percentage to 0-10
                rating = self._parse_rating(value)
                return rating / 10 if rating else None
            
            elif 'metacritic' in source_lower:
                # Metacritic: 0-100 to 0-10
                rating = self._parse_rating(value)
                return rating / 10 if rating else None
            
            else:
                # Try to parse as-is
                return self._parse_rating(value)
                
        except Exception:
            return None
    
    def check_health(self) -> Dict:
        """Check OMDb service health"""
        try:
            params = {
                'apikey': self.api_key,
                'i': 'tt0111161',  # The Shawshank Redemption - reliable test
                'plot': 'short'
            }
            
            response = self.session.get(self.BASE_URL, params=params, timeout=5)
            
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