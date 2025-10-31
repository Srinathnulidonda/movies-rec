"""
CineBrain YouTube Service
Intelligent trailer fetching with validation and caching
"""

import os
import logging
import requests
import re
from typing import Dict, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from difflib import SequenceMatcher

from .errors import APIError, handle_api_error
from .validator import DetailsValidator

logger = logging.getLogger(__name__)

class YouTubeService:
    """YouTube API integration for trailer fetching"""
    
    BASE_URL = 'https://www.googleapis.com/youtube/v3'
    
    def __init__(self):
        self.api_key = os.environ.get('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError("YOUTUBE_API_KEY environment variable is required")
        
        self.session = self._create_session()
        self.last_request_time = None
        self.rate_limit_delay = 0.1  # Conservative rate limiting
        
        logger.info("YouTube Service initialized successfully")
    
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
    
    def _rate_limit(self):
        """Rate limiting to avoid quota exhaustion"""
        import time
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    @handle_api_error
    def search_and_validate_trailer(self, title: str, content_type: str) -> Optional[Dict]:
        """
        Intelligent trailer search with validation
        Only called when content details are requested
        """
        try:
            if not DetailsValidator.validate_title(title):
                logger.warning(f"Invalid title for trailer search: {title}")
                return None
            
            # Clean title for better search results
            clean_title = self._clean_title_for_search(title)
            
            # Generate search queries based on content type
            search_queries = self._generate_search_queries(clean_title, content_type)
            
            # Try each search query until we find a valid trailer
            for query in search_queries:
                logger.debug(f"Searching YouTube for: {query}")
                
                videos = self._search_videos(query)
                if not videos:
                    continue
                
                # Validate and rank results
                trailer = self._validate_and_select_trailer(videos, clean_title, content_type)
                if trailer:
                    logger.info(f"Found valid trailer for '{title}': {trailer['youtube_id']}")
                    return trailer
            
            logger.info(f"No suitable trailer found for '{title}' ({content_type})")
            return None
            
        except Exception as e:
            logger.error(f"Error searching trailer for '{title}': {e}")
            return None
    
    def _clean_title_for_search(self, title: str) -> str:
        """Clean title for optimal YouTube search"""
        try:
            clean = title.strip()
            
            # Remove year in parentheses
            clean = re.sub(r'\s*\(\d{4}\)\s*', '', clean)
            
            # Remove subtitle after colon if main title is substantial
            if ':' in clean:
                parts = clean.split(':')
                if len(parts[0].strip()) >= 3:
                    clean = parts[0].strip()
            
            # Remove special characters that might interfere
            clean = re.sub(r'[^\w\s\-&]', '', clean)
            
            # Clean up whitespace
            clean = re.sub(r'\s+', ' ', clean).strip()
            
            return clean
            
        except Exception as e:
            logger.error(f"Error cleaning title: {e}")
            return title
    
    def _generate_search_queries(self, title: str, content_type: str) -> List[str]:
        """Generate prioritized search queries"""
        queries = []
        
        if content_type == 'anime':
            queries = [
                f"{title} official trailer",
                f"{title} anime trailer",
                f"{title} PV",
                f"{title} anime PV",
                f"{title} official PV",
                f"{title} trailer anime"
            ]
        elif content_type == 'tv':
            queries = [
                f"{title} official trailer",
                f"{title} series trailer",
                f"{title} season 1 trailer",
                f"{title} TV series trailer",
                f"{title} show trailer",
                f"{title} teaser"
            ]
        else:  # movie
            queries = [
                f"{title} official trailer",
                f"{title} movie trailer",
                f"{title} film trailer",
                f"{title} final trailer",
                f"{title} teaser trailer",
                f"{title} trailer"
            ]
        
        return queries[:4]  # Limit to prevent excessive API calls
    
    @handle_api_error
    def _search_videos(self, query: str) -> List[Dict]:
        """Search for videos on YouTube"""
        try:
            url = f"{self.BASE_URL}/search"
            params = {
                'key': self.api_key,
                'q': query,
                'part': 'snippet',
                'type': 'video',
                'maxResults': 8,  # Reduced to save quota
                'order': 'relevance',
                'videoDefinition': 'any',
                'videoDuration': 'any',
                'safeSearch': 'moderate'
            }
            
            self._rate_limit()
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            elif response.status_code == 403:
                logger.error("YouTube API quota exceeded")
                return []
            else:
                raise APIError("YouTube", f"HTTP {response.status_code}", response.status_code)
                
        except requests.exceptions.RequestException as e:
            raise APIError("YouTube", f"Search request failed: {str(e)}")
    
    def _validate_and_select_trailer(self, videos: List[Dict], title: str, content_type: str) -> Optional[Dict]:
        """Validate videos and select the best trailer"""
        try:
            scored_trailers = []
            
            for video in videos:
                score = self._score_trailer_candidate(video, title, content_type)
                if score > 0:
                    trailer_data = self._build_trailer_data(video, score)
                    scored_trailers.append(trailer_data)
            
            if not scored_trailers:
                return None
            
            # Sort by score and return the best one
            scored_trailers.sort(key=lambda x: x['score'], reverse=True)
            best_trailer = scored_trailers[0]
            
            logger.debug(f"Selected trailer with score {best_trailer['score']}: {best_trailer['title']}")
            return best_trailer
            
        except Exception as e:
            logger.error(f"Error validating trailers: {e}")
            return None
    
    def _score_trailer_candidate(self, video: Dict, title: str, content_type: str) -> float:
        """Score a video based on how likely it is to be a legitimate trailer"""
        try:
            snippet = video.get('snippet', {})
            video_title = snippet.get('title', '').lower()
            video_description = snippet.get('description', '').lower()
            channel_title = snippet.get('channelTitle', '').lower()
            
            score = 0.0
            
            # Title similarity (most important factor)
            title_similarity = SequenceMatcher(None, title.lower(), video_title).ratio()
            score += title_similarity * 50
            
            # Trailer keywords in title (positive)
            trailer_keywords = ['trailer', 'teaser', 'preview', 'official']
            for keyword in trailer_keywords:
                if keyword in video_title:
                    score += 15
            
            # Content type specific keywords
            if content_type == 'anime' and any(word in video_title for word in ['pv', 'anime', 'preview']):
                score += 10
            elif content_type == 'tv' and any(word in video_title for word in ['series', 'season', 'episode']):
                score += 10
            
            # Official channels (positive)
            official_indicators = [
                'official', 'studios', 'entertainment', 'pictures', 'films',
                'movie', 'cinema', 'warner', 'disney', 'sony', 'universal',
                'paramount', 'fox', 'marvel', 'dc', 'netflix', 'hbo'
            ]
            if any(indicator in channel_title for indicator in official_indicators):
                score += 20
            
            # Negative indicators
            negative_keywords = [
                'reaction', 'review', 'breakdown', 'analysis', 'behind the scenes',
                'making of', 'interview', 'music video', 'soundtrack', 'theme song',
                'fan made', 'fanmade', 'parody', 'spoof', 'fake', 'leaked'
            ]
            
            for negative in negative_keywords:
                if negative in video_title or negative in video_description[:200]:
                    score -= 25
            
            # Duration check (trailers are usually 30s-5min)
            # Note: We can't check duration from search API, would need additional call
            
            # Minimum score threshold
            if score < 30:
                return 0
            
            return score
            
        except Exception as e:
            logger.error(f"Error scoring trailer candidate: {e}")
            return 0
    
    def _build_trailer_data(self, video: Dict, score: float) -> Dict:
        """Build standardized trailer data structure"""
        try:
            snippet = video.get('snippet', {})
            video_id = video.get('id', {}).get('videoId', '')
            
            return {
                'youtube_id': video_id,
                'title': snippet.get('title', ''),
                'description': snippet.get('description', '')[:300] + '...' if len(snippet.get('description', '')) > 300 else snippet.get('description', ''),
                'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                'embed_url': f"https://www.youtube.com/embed/{video_id}",
                'watch_url': f"https://www.youtube.com/watch?v={video_id}",
                'channel_title': snippet.get('channelTitle', ''),
                'published_at': snippet.get('publishedAt', ''),
                'score': score,
                'source': 'youtube_api'
            }
            
        except Exception as e:
            logger.error(f"Error building trailer data: {e}")
            return {}
    
    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """Get additional details for a specific video"""
        try:
            if not DetailsValidator.validate_youtube_id(video_id):
                return None
            
            url = f"{self.BASE_URL}/videos"
            params = {
                'key': self.api_key,
                'id': video_id,
                'part': 'snippet,contentDetails,statistics',
            }
            
            self._rate_limit()
            response = self.session.get(url, params=params, timeout=8)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                return items[0] if items else None
            else:
                raise APIError("YouTube", f"HTTP {response.status_code}", response.status_code)
                
        except requests.exceptions.RequestException as e:
            raise APIError("YouTube", f"Video details request failed: {str(e)}")
    
    def check_health(self) -> Dict:
        """Check YouTube service health"""
        try:
            url = f"{self.BASE_URL}/search"
            params = {
                'key': self.api_key,
                'q': 'test',
                'part': 'snippet',
                'type': 'video',
                'maxResults': 1
            }
            
            response = self.session.get(url, params=params, timeout=5)
            
            return {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': response.elapsed.total_seconds(),
                'status_code': response.status_code,
                'quota_exceeded': response.status_code == 403
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }