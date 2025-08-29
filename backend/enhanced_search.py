# backend/enhanced_search.py

import json
import re
import unicodedata
import logging
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)


class EnhancedSearchEngine:
    """
    Advanced search engine with multi-source integration and intelligent ranking
    Integrates TMDB, OMDB, and Jikan APIs for comprehensive search results
    """
    
    def __init__(self, tmdb_service=None, omdb_service=None, jikan_service=None, youtube_service=None):
        """
        Initialize search engine with API services
        
        Args:
            tmdb_service: TMDBService instance
            omdb_service: OMDbService instance
            jikan_service: JikanService instance
            youtube_service: YouTubeService instance
        """
        self.tmdb = tmdb_service
        self.omdb = omdb_service
        self.jikan = jikan_service
        self.youtube = youtube_service
        
        # Stop words for better search matching
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 
            'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'could'
        }
        
        # Content type indicators
        self.content_type_keywords = {
            'movie': ['movie', 'film', 'cinema', 'flick'],
            'tv': ['tv', 'series', 'show', 'season', 'episode', 'serial'],
            'anime': ['anime', 'manga', 'ova', 'ona', 'japanese animation']
        }
        
        # Genre mapping for better matching
        self.genre_keywords = {
            'action': ['action', 'fight', 'battle', 'war', 'combat', 'explosive'],
            'comedy': ['comedy', 'funny', 'laugh', 'humor', 'hilarious', 'comic'],
            'drama': ['drama', 'emotional', 'serious', 'intense', 'dramatic'],
            'horror': ['horror', 'scary', 'terror', 'ghost', 'nightmare', 'frightening'],
            'romance': ['romance', 'love', 'romantic', 'relationship', 'dating'],
            'sci-fi': ['sci-fi', 'science fiction', 'space', 'future', 'alien', 'futuristic'],
            'thriller': ['thriller', 'suspense', 'mystery', 'tense', 'psychological'],
            'anime': ['anime', 'manga', 'japanese', 'animation', 'otaku'],
            'documentary': ['documentary', 'real', 'true story', 'factual', 'non-fiction']
        }
        
        # Regional language mapping
        self.language_mapping = {
            'hindi': 'hi',
            'telugu': 'te',
            'tamil': 'ta',
            'kannada': 'kn',
            'malayalam': 'ml',
            'bengali': 'bn',
            'marathi': 'mr',
            'gujarati': 'gu',
            'punjabi': 'pa',
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'italian': 'it',
            'japanese': 'ja',
            'korean': 'ko',
            'chinese': 'zh'
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        if not text:
            return ""
        
        # Remove accents and diacritics
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text) 
            if unicodedata.category(c) != 'Mn'
        )
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace special characters with spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str, remove_stop_words: bool = True) -> List[str]:
        """Tokenize text into searchable terms"""
        normalized = self.normalize_text(text)
        tokens = normalized.split()
        
        if remove_stop_words:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        
        return tokens
    
    def extract_year_from_query(self, query: str) -> Optional[int]:
        """Extract year from search query"""
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        if year_match:
            return int(year_match.group())
        return None
    
    def detect_content_type_from_query(self, query: str) -> Optional[str]:
        """Detect if query is asking for specific content type"""
        query_lower = query.lower()
        for content_type, keywords in self.content_type_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return content_type
        return None
    
    def detect_language_from_query(self, query: str) -> Optional[str]:
        """Detect language preference from query"""
        query_lower = query.lower()
        for language, code in self.language_mapping.items():
            if language in query_lower:
                return code
        return None
    
    async def search_multi_source(self, query: str, content_type: str = 'multi', 
                                 page: int = 1, limit: int = 20) -> Dict[str, Any]:
        """
        Search across multiple sources (TMDB, OMDB, Jikan) concurrently
        
        Args:
            query: Search query
            content_type: Type of content to search for
            page: Page number for pagination
            limit: Maximum number of results
            
        Returns:
            Dictionary containing search results from all sources
        """
        results = {
            'tmdb': [],
            'omdb': [],
            'jikan': [],
            'combined': []
        }
        
        # Extract search parameters from query
        year = self.extract_year_from_query(query)
        detected_type = self.detect_content_type_from_query(query) or content_type
        language = self.detect_language_from_query(query)
        
        # Clean query for better matching
        clean_query = self.clean_search_query(query)
        
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            # Search TMDB
            if self.tmdb and detected_type in ['multi', 'movie', 'tv']:
                futures['tmdb'] = executor.submit(
                    self._search_tmdb, clean_query, detected_type, language, page
                )
            
            # Search OMDB (only for specific titles)
            if self.omdb and len(clean_query.split()) <= 5:
                futures['omdb'] = executor.submit(
                    self._search_omdb, clean_query, year
                )
            
            # Search Jikan for anime
            if self.jikan and detected_type in ['multi', 'anime']:
                futures['jikan'] = executor.submit(
                    self._search_jikan, clean_query, page
                )
            
            # Collect results
            for source, future in futures.items():
                try:
                    result = future.result(timeout=10)
                    if result:
                        results[source] = result
                except Exception as e:
                    logger.error(f"Error searching {source}: {e}")
        
        # Combine and rank results
        combined_results = self._combine_and_rank_results(
            results, query, detected_type, limit
        )
        
        results['combined'] = combined_results
        
        return results
    
    def _search_tmdb(self, query: str, content_type: str, 
                     language: Optional[str], page: int) -> List[Dict]:
        """Search TMDB for content"""
        if not self.tmdb:
            return []
        
        try:
            results = []
            
            # Search based on content type
            if content_type == 'multi':
                tmdb_response = self.tmdb.search_content(query, 'multi', page=page)
            elif content_type == 'movie':
                tmdb_response = self.tmdb.search_content(query, 'movie', page=page)
            elif content_type == 'tv':
                tmdb_response = self.tmdb.search_content(query, 'tv', page=page)
            else:
                tmdb_response = self.tmdb.search_content(query, 'multi', page=page)
            
            if tmdb_response and 'results' in tmdb_response:
                for item in tmdb_response['results']:
                    processed_item = self._process_tmdb_item(item)
                    if processed_item:
                        results.append(processed_item)
            
            return results
            
        except Exception as e:
            logger.error(f"TMDB search error: {e}")
            return []
    
    def _search_omdb(self, query: str, year: Optional[int]) -> List[Dict]:
        """Search OMDB for detailed movie information"""
        if not self.omdb:
            return []
        
        try:
            results = []
            
            # OMDB search is more limited, so we'll search by title
            params = {'s': query, 'type': 'movie'}
            if year:
                params['y'] = year
            
            # Note: You'll need to implement a search method in OMDbService
            # For now, we'll return empty as OMDB is typically used for detailed info
            # by IMDB ID rather than search
            
            return results
            
        except Exception as e:
            logger.error(f"OMDB search error: {e}")
            return []
    
    def _search_jikan(self, query: str, page: int) -> List[Dict]:
        """Search Jikan for anime content"""
        if not self.jikan:
            return []
        
        try:
            results = []
            
            jikan_response = self.jikan.search_anime(query, page)
            
            if jikan_response and 'data' in jikan_response:
                for anime in jikan_response['data']:
                    processed_anime = self._process_jikan_item(anime)
                    if processed_anime:
                        results.append(processed_anime)
            
            return results
            
        except Exception as e:
            logger.error(f"Jikan search error: {e}")
            return []
    
    def _process_tmdb_item(self, item: Dict) -> Dict:
        """Process TMDB item to standardized format"""
        try:
            # Determine content type
            content_type = 'movie' if 'title' in item else 'tv'
            
            # Build poster URL
            poster_path = None
            if item.get('poster_path'):
                poster_path = f"https://image.tmdb.org/t/p/w500{item['poster_path']}"
            
            # Build backdrop URL
            backdrop_path = None
            if item.get('backdrop_path'):
                backdrop_path = f"https://image.tmdb.org/t/p/w1280{item['backdrop_path']}"
            
            return {
                'source': 'tmdb',
                'tmdb_id': item.get('id'),
                'title': item.get('title') or item.get('name'),
                'original_title': item.get('original_title') or item.get('original_name'),
                'content_type': content_type,
                'release_date': item.get('release_date') or item.get('first_air_date'),
                'poster_path': poster_path,
                'backdrop_path': backdrop_path,
                'overview': item.get('overview'),
                'rating': item.get('vote_average'),
                'vote_count': item.get('vote_count'),
                'popularity': item.get('popularity'),
                'genre_ids': item.get('genre_ids', []),
                'original_language': item.get('original_language')
            }
        except Exception as e:
            logger.error(f"Error processing TMDB item: {e}")
            return None
    
    def _process_jikan_item(self, anime: Dict) -> Dict:
        """Process Jikan anime item to standardized format"""
        try:
            # Get poster image
            poster_path = anime.get('images', {}).get('jpg', {}).get('large_image_url') or \
                         anime.get('images', {}).get('jpg', {}).get('image_url')
            
            # Get genres
            genres = [genre.get('name') for genre in anime.get('genres', [])]
            
            # Get release date
            release_date = None
            if anime.get('aired', {}).get('from'):
                release_date = anime['aired']['from'][:10] if anime['aired']['from'] else None
            
            return {
                'source': 'jikan',
                'mal_id': anime.get('mal_id'),
                'title': anime.get('title'),
                'original_title': anime.get('title_japanese'),
                'content_type': 'anime',
                'release_date': release_date,
                'poster_path': poster_path,
                'backdrop_path': poster_path,  # Jikan doesn't provide separate backdrop
                'overview': anime.get('synopsis'),
                'rating': anime.get('score'),
                'vote_count': anime.get('scored_by'),
                'popularity': anime.get('popularity'),
                'genres': genres,
                'episodes': anime.get('episodes'),
                'status': anime.get('status')
            }
        except Exception as e:
            logger.error(f"Error processing Jikan item: {e}")
            return None
    
    def clean_search_query(self, query: str) -> str:
        """Clean search query by removing common additions"""
        # Remove year from query for better matching
        query = re.sub(r'\b(19|20)\d{2}\b', '', query)
        
        # Remove content type keywords
        for keywords in self.content_type_keywords.values():
            for keyword in keywords:
                query = re.sub(r'\b' + keyword + r'\b', '', query, flags=re.IGNORECASE)
        
        # Remove language indicators
        for language in self.language_mapping.keys():
            query = re.sub(r'\b' + language + r'\b', '', query, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        query = ' '.join(query.split())
        
        return query.strip()
    
    def calculate_relevance_score(self, query: str, item: Dict) -> float:
        """
        Calculate comprehensive relevance score for a search result
        
        Args:
            query: Original search query
            item: Search result item
            
        Returns:
            Float score representing relevance (higher is better)
        """
        score = 0.0
        
        # Normalize query and title
        query_normalized = self.normalize_text(query)
        title_normalized = self.normalize_text(item.get('title', ''))
        original_title_normalized = self.normalize_text(item.get('original_title', ''))
        
        # 1. Title matching (highest weight - 40%)
        if query_normalized == title_normalized:
            score += 100.0  # Exact match
        elif query_normalized in title_normalized:
            score += 70.0  # Contains query
            if title_normalized.startswith(query_normalized):
                score += 20.0  # Starts with query
        
        # Check original title
        if original_title_normalized:
            if query_normalized == original_title_normalized:
                score += 80.0
            elif query_normalized in original_title_normalized:
                score += 40.0
        
        # Fuzzy matching
        title_similarity = SequenceMatcher(None, query_normalized, title_normalized).ratio()
        score += title_similarity * 50.0
        
        # 2. Release date relevance (15%)
        release_date = item.get('release_date')
        if release_date:
            try:
                if isinstance(release_date, str):
                    release_date = datetime.strptime(release_date[:10], '%Y-%m-%d').date()
                
                days_old = (datetime.now().date() - release_date).days
                
                if days_old < 30:  # Last month
                    score += 20.0
                elif days_old < 90:  # Last 3 months
                    score += 15.0
                elif days_old < 365:  # Last year
                    score += 10.0
                elif days_old < 365 * 2:  # Last 2 years
                    score += 5.0
                
            except:
                pass
        
        # 3. Popularity and ratings (25%)
        rating = item.get('rating', 0) or 0
        vote_count = item.get('vote_count', 0) or 0
        popularity = item.get('popularity', 0) or 0
        
        # Weighted rating (IMDB formula)
        if vote_count > 10:
            weighted_rating = (vote_count / (vote_count + 100)) * rating
            score += weighted_rating * 5.0
        
        # Popularity boost (logarithmic scale)
        if popularity > 0:
            import math
            score += min(math.log10(popularity + 1) * 5, 15.0)
        
        # 4. Overview matching (10%)
        overview = item.get('overview', '')
        if overview:
            overview_normalized = self.normalize_text(overview)
            if query_normalized in overview_normalized:
                score += 10.0
            
            # Token matching
            query_tokens = self.tokenize(query)
            overview_tokens = self.tokenize(overview)
            matching_tokens = set(query_tokens) & set(overview_tokens)
            score += len(matching_tokens) * 2.0
        
        # 5. Source reliability bonus (10%)
        source = item.get('source', '')
        if source == 'tmdb':
            score += 10.0  # TMDB is most comprehensive
        elif source == 'jikan':
            score += 8.0  # Jikan for anime
        elif source == 'omdb':
            score += 7.0  # OMDB for detailed info
        
        # 6. Content type matching
        detected_type = self.detect_content_type_from_query(query)
        if detected_type and item.get('content_type') == detected_type:
            score += 15.0
        
        # 7. Year matching if specified
        year_query = self.extract_year_from_query(query)
        if year_query and release_date:
            try:
                item_year = release_date.year if hasattr(release_date, 'year') else int(release_date[:4])
                if year_query == item_year:
                    score += 30.0  # Exact year match
                elif abs(year_query - item_year) <= 1:
                    score += 15.0  # Close year match
            except:
                pass
        
        return score
    
    def _combine_and_rank_results(self, results: Dict, query: str, 
                                  content_type: str, limit: int) -> List[Dict]:
        """
        Combine results from multiple sources and rank by relevance
        
        Args:
            results: Dictionary containing results from each source
            query: Original search query
            content_type: Type of content being searched
            limit: Maximum number of results to return
            
        Returns:
            List of ranked search results
        """
        all_results = []
        seen_titles = set()
        
        # Combine all results
        for source in ['tmdb', 'jikan', 'omdb']:
            for item in results.get(source, []):
                if not item:
                    continue
                
                # Create unique identifier to avoid duplicates
                title_key = self.normalize_text(item.get('title', ''))
                
                # Skip if we've seen this title
                if title_key in seen_titles:
                    continue
                
                seen_titles.add(title_key)
                
                # Calculate relevance score
                item['relevance_score'] = self.calculate_relevance_score(query, item)
                
                all_results.append(item)
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Add ranking position
        for idx, result in enumerate(all_results[:limit]):
            result['search_rank'] = idx + 1
        
        return all_results[:limit]
    
    async def get_enhanced_details(self, content_id: int, content_type: str, 
                                  tmdb_id: Optional[int] = None,
                                  mal_id: Optional[int] = None) -> Dict:
        """
        Get enhanced details from multiple sources
        
        Args:
            content_id: Internal content ID
            content_type: Type of content
            tmdb_id: TMDB ID if available
            mal_id: MyAnimeList ID if available
            
        Returns:
            Dictionary containing enhanced details
        """
        details = {
            'basic_info': {},
            'cast_crew': [],
            'videos': [],
            'images': [],
            'reviews': [],
            'similar': [],
            'recommendations': []
        }
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            # Get TMDB details
            if tmdb_id and self.tmdb:
                futures['tmdb'] = executor.submit(
                    self.tmdb.get_content_details, tmdb_id, content_type
                )
            
            # Get Jikan details for anime
            if mal_id and self.jikan and content_type == 'anime':
                futures['jikan'] = executor.submit(
                    self.jikan.get_anime_details, mal_id
                )
            
            # Get YouTube trailers
            if self.youtube:
                futures['youtube'] = executor.submit(
                    self._get_youtube_trailers, content_type
                )
            
            # Collect results
            for source, future in futures.items():
                try:
                    result = future.result(timeout=10)
                    if result:
                        if source == 'tmdb':
                            details.update(self._process_tmdb_details(result))
                        elif source == 'jikan':
                            details.update(self._process_jikan_details(result))
                        elif source == 'youtube':
                            details['videos'] = result
                except Exception as e:
                    logger.error(f"Error getting {source} details: {e}")
        
        return details
    
    def _process_tmdb_details(self, tmdb_data: Dict) -> Dict:
        """Process TMDB detailed data"""
        processed = {
            'cast_crew': [],
            'videos': [],
            'images': [],
            'similar': [],
            'recommendations': []
        }
        
        # Process cast and crew
        if 'credits' in tmdb_data:
            cast = tmdb_data['credits'].get('cast', [])[:10]
            crew = tmdb_data['credits'].get('crew', [])[:5]
            
            processed['cast_crew'] = {
                'cast': [
                    {
                        'name': person.get('name'),
                        'character': person.get('character'),
                        'profile_path': f"https://image.tmdb.org/t/p/w200{person['profile_path']}" 
                                      if person.get('profile_path') else None
                    }
                    for person in cast
                ],
                'crew': [
                    {
                        'name': person.get('name'),
                        'job': person.get('job'),
                        'department': person.get('department')
                    }
                    for person in crew
                ]
            }
        
        # Process videos
        if 'videos' in tmdb_data:
            videos = tmdb_data['videos'].get('results', [])
            processed['videos'] = [
                {
                    'key': video.get('key'),
                    'name': video.get('name'),
                    'type': video.get('type'),
                    'site': video.get('site')
                }
                for video in videos if video.get('site') == 'YouTube'
            ][:5]
        
        # Process similar content
        if 'similar' in tmdb_data:
            similar = tmdb_data['similar'].get('results', [])[:10]
            processed['similar'] = [self._process_tmdb_item(item) for item in similar]
        
        # Process recommendations
        if 'recommendations' in tmdb_data:
            recommendations = tmdb_data['recommendations'].get('results', [])[:10]
            processed['recommendations'] = [self._process_tmdb_item(item) for item in recommendations]
        
        return processed
    
    def _process_jikan_details(self, jikan_data: Dict) -> Dict:
        """Process Jikan anime detailed data"""
        processed = {
            'anime_info': {},
            'characters': [],
            'staff': []
        }
        
        if 'data' in jikan_data:
            anime = jikan_data['data']
            
            # Basic anime info
            processed['anime_info'] = {
                'episodes': anime.get('episodes'),
                'status': anime.get('status'),
                'aired': anime.get('aired'),
                'duration': anime.get('duration'),
                'rating': anime.get('rating'),
                'source': anime.get('source'),
                'studios': [studio.get('name') for studio in anime.get('studios', [])]
            }
            
            # Characters
            if 'characters' in anime:
                processed['characters'] = [
                    {
                        'name': char.get('character', {}).get('name'),
                        'role': char.get('role'),
                        'image': char.get('character', {}).get('images', {}).get('jpg', {}).get('image_url')
                    }
                    for char in anime.get('characters', [])[:10]
                ]
        
        return processed
    
    def _get_youtube_trailers(self, content_type: str) -> List[Dict]:
        """Get YouTube trailers for content"""
        # This would be implemented based on your YouTube service
        return []
    
    @lru_cache(maxsize=128)
    def get_image_urls(self, poster_path: Optional[str], backdrop_path: Optional[str]) -> Dict:
        """
        Get properly formatted image URLs
        
        Args:
            poster_path: Poster image path
            backdrop_path: Backdrop image path
            
        Returns:
            Dictionary with formatted image URLs
        """
        images = {
            'poster': {
                'small': None,
                'medium': None,
                'large': None,
                'original': None
            },
            'backdrop': {
                'small': None,
                'medium': None,
                'large': None,
                'original': None
            }
        }
        
        # Process poster
        if poster_path:
            if poster_path.startswith('http'):
                # Already a full URL (probably from Jikan)
                images['poster']['original'] = poster_path
                images['poster']['large'] = poster_path
                images['poster']['medium'] = poster_path
                images['poster']['small'] = poster_path
            else:
                # TMDB path
                images['poster']['small'] = f"https://image.tmdb.org/t/p/w200{poster_path}"
                images['poster']['medium'] = f"https://image.tmdb.org/t/p/w300{poster_path}"
                images['poster']['large'] = f"https://image.tmdb.org/t/p/w500{poster_path}"
                images['poster']['original'] = f"https://image.tmdb.org/t/p/original{poster_path}"
        
        # Process backdrop
        if backdrop_path:
            if backdrop_path.startswith('http'):
                images['backdrop']['original'] = backdrop_path
                images['backdrop']['large'] = backdrop_path
            else:
                # TMDB path
                images['backdrop']['small'] = f"https://image.tmdb.org/t/p/w300{backdrop_path}"
                images['backdrop']['medium'] = f"https://image.tmdb.org/t/p/w780{backdrop_path}"
                images['backdrop']['large'] = f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
                images['backdrop']['original'] = f"https://image.tmdb.org/t/p/original{backdrop_path}"
        
        return images


class SearchOptimizer:
    """
    Optimizes search queries and results for better performance
    """
    
    @staticmethod
    def optimize_query(query: str) -> str:
        """
        Optimize search query for better results
        
        Args:
            query: Original search query
            
        Returns:
            Optimized query string
        """
        # Remove special characters but keep spaces
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Remove duplicate words
        words = query.lower().split()
        seen = set()
        unique_words = []
        for word in words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        # Rejoin words
        optimized = ' '.join(unique_words)
        
        return optimized.strip()
    
    @staticmethod
    def create_search_hash(query: str, content_type: str, page: int) -> str:
        """
        Create a hash for caching search results
        
        Args:
            query: Search query
            content_type: Type of content
            page: Page number
            
        Returns:
            Hash string for caching
        """
        cache_string = f"{query}:{content_type}:{page}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    @staticmethod
    def filter_duplicate_results(results: List[Dict]) -> List[Dict]:
        """
        Filter duplicate results based on title similarity
        
        Args:
            results: List of search results
            
        Returns:
            Filtered list without duplicates
        """
        filtered = []
        seen_titles = set()
        
        for result in results:
            # Normalize title for comparison
            title = result.get('title', '').lower()
            title = re.sub(r'[^\w\s]', '', title)
            title = ' '.join(title.split())
            
            if title and title not in seen_titles:
                seen_titles.add(title)
                filtered.append(result)
        
        return filtered


# Export the main class
__all__ = ['EnhancedSearchEngine', 'SearchOptimizer']