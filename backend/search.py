# backend/search.py
import json
import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import hashlib
from functools import lru_cache

import numpy as np
from flask import current_app
from sqlalchemy import or_, and_, func, text
from sqlalchemy.orm import Query
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
import redis
import pickle

logger = logging.getLogger(__name__)

class SearchIndexer:
    """In-memory search index for ultra-fast searching"""
    
    def __init__(self):
        self.index = {
            'titles': {},  # title -> content_id mapping
            'normalized_titles': {},  # normalized_title -> content_id mapping
            'keywords': defaultdict(set),  # keyword -> set of content_ids
            'genres': defaultdict(set),  # genre -> set of content_ids
            'languages': defaultdict(set),  # language -> set of content_ids
            'years': defaultdict(set),  # year -> set of content_ids
            'content_map': {},  # content_id -> content_data
            'autocomplete': {},  # prefix -> list of suggestions
            'phonetic': defaultdict(set),  # soundex/metaphone -> content_ids
        }
        self.last_update = None
        self.update_interval = 300  # 5 minutes
        
    def needs_update(self) -> bool:
        """Check if index needs updating"""
        if not self.last_update:
            return True
        return (datetime.now() - self.last_update).seconds > self.update_interval
    
    def build_index(self, content_list: List[Any]) -> None:
        """Build or rebuild the search index"""
        logger.info(f"Building search index with {len(content_list)} items")
        start_time = time.time()
        
        # Clear existing index
        self.index = {
            'titles': {},
            'normalized_titles': {},
            'keywords': defaultdict(set),
            'genres': defaultdict(set),
            'languages': defaultdict(set),
            'years': defaultdict(set),
            'content_map': {},
            'autocomplete': {},
            'phonetic': defaultdict(set),
        }
        
        for content in content_list:
            # Store content mapping
            self.index['content_map'][content.id] = {
                'id': content.id,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'release_date': content.release_date,
                'rating': content.rating,
                'popularity': content.popularity,
                'overview': content.overview,
                'poster_path': content.poster_path,
                'youtube_trailer_id': content.youtube_trailer_id
            }
            
            # Index titles
            if content.title:
                self.index['titles'][content.title.lower()] = content.id
                normalized = self._normalize_title(content.title)
                self.index['normalized_titles'][normalized] = content.id
                
                # Build autocomplete index
                self._build_autocomplete(content.title, content.id)
                
                # Phonetic indexing for typo tolerance
                phonetic_key = self._get_phonetic_key(content.title)
                if phonetic_key:
                    self.index['phonetic'][phonetic_key].add(content.id)
            
            # Index original title
            if content.original_title and content.original_title != content.title:
                self.index['titles'][content.original_title.lower()] = content.id
                self._build_autocomplete(content.original_title, content.id)
            
            # Index keywords from title and overview
            keywords = self._extract_keywords(content.title, content.overview)
            for keyword in keywords:
                self.index['keywords'][keyword].add(content.id)
            
            # Index genres
            genres = json.loads(content.genres or '[]')
            for genre in genres:
                self.index['genres'][genre.lower()].add(content.id)
            
            # Index languages
            languages = json.loads(content.languages or '[]')
            for language in languages:
                self.index['languages'][language.lower()].add(content.id)
            
            # Index year
            if content.release_date:
                year = content.release_date.year
                self.index['years'][year].add(content.id)
        
        self.last_update = datetime.now()
        elapsed = time.time() - start_time
        logger.info(f"Search index built in {elapsed:.2f} seconds")
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for better matching"""
        # Remove special characters and extra spaces
        normalized = re.sub(r'[^\w\s]', '', title.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _extract_keywords(self, title: str, overview: str = None) -> Set[str]:
        """Extract searchable keywords from text"""
        keywords = set()
        
        # Extract from title
        if title:
            # Split by spaces and common delimiters
            words = re.findall(r'\w+', title.lower())
            keywords.update(words)
            
            # Add n-grams for compound words
            if len(words) > 1:
                for i in range(len(words) - 1):
                    keywords.add(f"{words[i]}_{words[i+1]}")
        
        # Extract from overview (limited)
        if overview:
            # Get first 50 words from overview
            overview_words = re.findall(r'\w+', overview.lower())[:50]
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'were'}
            keywords.update(w for w in overview_words if w not in stop_words and len(w) > 2)
        
        return keywords
    
    def _build_autocomplete(self, title: str, content_id: int) -> None:
        """Build autocomplete index for progressive search"""
        if not title:
            return
        
        title_lower = title.lower()
        words = title_lower.split()
        
        # Index full title prefixes
        for i in range(1, min(len(title_lower) + 1, 50)):  # Limit to 50 chars
            prefix = title_lower[:i]
            if prefix not in self.index['autocomplete']:
                self.index['autocomplete'][prefix] = []
            if content_id not in self.index['autocomplete'][prefix]:
                self.index['autocomplete'][prefix].append(content_id)
        
        # Index word prefixes
        for word in words:
            for i in range(1, min(len(word) + 1, 20)):  # Limit to 20 chars per word
                prefix = word[:i]
                if prefix not in self.index['autocomplete']:
                    self.index['autocomplete'][prefix] = []
                if content_id not in self.index['autocomplete'][prefix]:
                    self.index['autocomplete'][prefix].append(content_id)
    
    def _get_phonetic_key(self, text: str) -> Optional[str]:
        """Generate phonetic key for fuzzy matching (simplified Soundex)"""
        if not text:
            return None
        
        text = text.upper()
        # Remove non-letters
        text = re.sub(r'[^A-Z]', '', text)
        
        if not text:
            return None
        
        # Simplified Soundex algorithm
        soundex = text[0]
        
        # Soundex mappings
        mappings = {
            'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3',
            'L': '4', 'MN': '5', 'R': '6'
        }
        
        for char in text[1:]:
            for key, value in mappings.items():
                if char in key:
                    if len(soundex) == 1 or soundex[-1] != value:
                        soundex += value
                    break
        
        # Pad with zeros
        soundex = (soundex + '000')[:4]
        return soundex
    
    def search(self, query: str, limit: int = 20, filters: Dict = None) -> List[int]:
        """Fast in-memory search"""
        query_lower = query.lower()
        normalized_query = self._normalize_title(query)
        results = defaultdict(float)
        
        # 1. Exact title match (highest weight)
        if query_lower in self.index['titles']:
            results[self.index['titles'][query_lower]] += 100
        
        # 2. Normalized title match
        if normalized_query in self.index['normalized_titles']:
            results[self.index['normalized_titles'][normalized_query]] += 90
        
        # 3. Prefix match (autocomplete)
        if query_lower in self.index['autocomplete']:
            for content_id in self.index['autocomplete'][query_lower]:
                results[content_id] += 80
        
        # 4. Fuzzy title match
        for title, content_id in self.index['titles'].items():
            similarity = fuzz.ratio(query_lower, title)
            if similarity > 70:  # Threshold for fuzzy match
                results[content_id] += similarity * 0.7
        
        # 5. Keyword match
        query_keywords = self._extract_keywords(query)
        for keyword in query_keywords:
            if keyword in self.index['keywords']:
                for content_id in self.index['keywords'][keyword]:
                    results[content_id] += 30
        
        # 6. Phonetic match (for typos)
        phonetic_query = self._get_phonetic_key(query)
        if phonetic_query and phonetic_query in self.index['phonetic']:
            for content_id in self.index['phonetic'][phonetic_query]:
                results[content_id] += 40
        
        # Apply filters if provided
        if filters:
            results = self._apply_filters(results, filters)
        
        # Sort by score and return top results
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [content_id for content_id, _ in sorted_results[:limit]]
    
    def _apply_filters(self, results: Dict[int, float], filters: Dict) -> Dict[int, float]:
        """Apply filters to search results"""
        filtered = {}
        
        for content_id, score in results.items():
            content = self.index['content_map'].get(content_id)
            if not content:
                continue
            
            # Genre filter
            if filters.get('genre'):
                if filters['genre'].lower() not in [g.lower() for g in content['genres']]:
                    continue
            
            # Language filter
            if filters.get('language'):
                if filters['language'].lower() not in [l.lower() for l in content['languages']]:
                    continue
            
            # Year filter
            if filters.get('year') and content['release_date']:
                if content['release_date'].year != filters['year']:
                    continue
            
            # Content type filter
            if filters.get('content_type'):
                if content['content_type'] != filters['content_type']:
                    continue
            
            # Rating filter
            if filters.get('min_rating'):
                if not content['rating'] or content['rating'] < filters['min_rating']:
                    continue
            
            filtered[content_id] = score
        
        return filtered

class HybridSearch:
    """Hybrid search combining multiple search strategies"""
    
    def __init__(self, db, cache, services):
        self.db = db
        self.cache = cache
        self.services = services
        self.indexer = SearchIndexer()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def search(self, query: str, search_type: str = 'multi', page: int = 1, 
               limit: int = 20, filters: Dict = None) -> Dict[str, Any]:
        """
        Perform hybrid search across multiple sources
        Returns results with instant response using caching and indexing
        """
        # Generate cache key
        cache_key = self._generate_cache_key(query, search_type, page, filters)
        
        # Check cache first for instant results
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Cache hit for search: {query}")
            return cached_result
        
        # Perform search
        start_time = time.time()
        
        # Update index if needed
        if self.indexer.needs_update():
            self._update_search_index()
        
        # Execute parallel search strategies
        results = {
            'local': [],
            'external': [],
            'suggestions': [],
            'instant_results': []
        }
        
        # 1. Instant results from in-memory index
        instant_ids = self.indexer.search(query, limit * 2, filters)
        if instant_ids:
            instant_results = self._get_content_by_ids(instant_ids[:limit])
            results['instant_results'] = self._format_results(instant_results)
        
        # 2. Database search (fuzzy + full-text)
        db_results = self._database_search(query, search_type, filters, limit)
        results['local'] = self._format_results(db_results)
        
        # 3. External API search (async)
        if not results['instant_results'] or len(results['instant_results']) < 5:
            external_results = self._external_search(query, search_type, page)
            results['external'] = external_results
        
        # 4. Generate suggestions
        results['suggestions'] = self._generate_suggestions(query, results['instant_results'][:5])
        
        # Merge and rank results
        final_results = self._merge_and_rank_results(results, query, limit)
        
        # Prepare response
        response = {
            'results': final_results,
            'total_results': len(final_results),
            'page': page,
            'query': query,
            'search_time': round(time.time() - start_time, 3),
            'suggestions': results['suggestions'],
            'instant': len(results['instant_results']) > 0
        }
        
        # Cache the result
        self._cache_result(cache_key, response)
        
        return response
    
    def autocomplete(self, prefix: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Provide instant autocomplete suggestions
        """
        if len(prefix) < 2:
            return []
        
        # Check cache
        cache_key = f"autocomplete:{prefix.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            return json.loads(cached)
        
        suggestions = []
        
        # Get from index
        if prefix.lower() in self.indexer.index['autocomplete']:
            content_ids = self.indexer.index['autocomplete'][prefix.lower()][:limit]
            for content_id in content_ids:
                content_data = self.indexer.index['content_map'].get(content_id)
                if content_data:
                    suggestions.append({
                        'id': content_id,
                        'title': content_data['title'],
                        'type': content_data['content_type'],
                        'year': content_data['release_date'].year if content_data['release_date'] else None,
                        'poster': self._get_poster_url(content_data['poster_path'])
                    })
        
        # Cache for 1 hour
        self.cache.set(cache_key, json.dumps(suggestions), timeout=3600)
        
        return suggestions[:limit]
    
    def _update_search_index(self):
        """Update the search index with latest content"""
        try:
            from models import Content  # Import here to avoid circular imports
            content_list = Content.query.limit(10000).all()  # Limit for memory
            self.indexer.build_index(content_list)
        except Exception as e:
            logger.error(f"Failed to update search index: {e}")
    
    def _database_search(self, query: str, search_type: str, filters: Dict, limit: int) -> List[Any]:
        """
        Perform database search with fuzzy matching
        """
        from models import Content  # Import here to avoid circular imports
        
        # Build search conditions
        search_conditions = []
        
        # Title matching (exact, prefix, fuzzy)
        search_conditions.append(Content.title.ilike(f'%{query}%'))
        search_conditions.append(Content.original_title.ilike(f'%{query}%'))
        
        # Overview matching
        if len(query) > 3:  # Only for longer queries
            search_conditions.append(Content.overview.ilike(f'%{query}%'))
        
        # Build query
        db_query = Content.query.filter(or_(*search_conditions))
        
        # Apply type filter
        if search_type != 'multi':
            if search_type == 'anime':
                db_query = db_query.filter(Content.content_type == 'anime')
            elif search_type == 'movie':
                db_query = db_query.filter(Content.content_type == 'movie')
            elif search_type == 'tv':
                db_query = db_query.filter(Content.content_type == 'tv')
        
        # Apply additional filters
        if filters:
            if filters.get('genre'):
                db_query = db_query.filter(Content.genres.contains(filters['genre']))
            if filters.get('language'):
                db_query = db_query.filter(Content.languages.contains(filters['language']))
            if filters.get('year'):
                year_start = datetime(filters['year'], 1, 1)
                year_end = datetime(filters['year'], 12, 31)
                db_query = db_query.filter(Content.release_date.between(year_start, year_end))
            if filters.get('min_rating'):
                db_query = db_query.filter(Content.rating >= filters['min_rating'])
        
        # Order by relevance (simplified)
        db_query = db_query.order_by(
            Content.popularity.desc(),
            Content.rating.desc()
        )
        
        return db_query.limit(limit).all()
    
    def _external_search(self, query: str, search_type: str, page: int) -> List[Dict]:
        """
        Search external APIs (TMDB, Jikan) with concurrent requests
        """
        external_results = []
        futures = []
        
        with self.executor as executor:
            # Search TMDB
            if search_type in ['multi', 'movie', 'tv']:
                futures.append(
                    executor.submit(
                        self.services['TMDBService'].search_content,
                        query, search_type, page=page
                    )
                )
            
            # Search anime
            if search_type in ['multi', 'anime']:
                futures.append(
                    executor.submit(
                        self.services['JikanService'].search_anime,
                        query, page
                    )
                )
            
            # Process results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=5)
                    if result:
                        if 'results' in result:  # TMDB format
                            for item in result.get('results', [])[:10]:
                                content_type = 'movie' if 'title' in item else 'tv'
                                content = self.services['ContentService'].save_content_from_tmdb(item, content_type)
                                if content:
                                    external_results.append(self._format_content(content))
                        elif 'data' in result:  # Jikan format
                            for anime in result.get('data', [])[:10]:
                                content = self.services['ContentService'].save_anime_content(anime)
                                if content:
                                    external_results.append(self._format_content(content))
                except Exception as e:
                    logger.error(f"External search error: {e}")
        
        return external_results
    
    def _generate_suggestions(self, query: str, top_results: List[Dict]) -> List[str]:
        """
        Generate search suggestions based on query and results
        """
        suggestions = []
        
        # Add query variations
        query_lower = query.lower()
        
        # Suggest corrections for common typos
        corrections = self._get_spelling_corrections(query)
        suggestions.extend(corrections[:2])
        
        # Suggest related searches based on top results
        if top_results:
            # Extract genres from top results
            genres = set()
            for result in top_results[:3]:
                if 'genres' in result:
                    genres.update(result['genres'])
            
            # Suggest genre-based searches
            for genre in list(genres)[:2]:
                suggestions.append(f"{query} {genre}")
        
        # Suggest different content types
        if 'movie' not in query_lower and 'film' not in query_lower:
            suggestions.append(f"{query} movies")
        if 'series' not in query_lower and 'tv' not in query_lower:
            suggestions.append(f"{query} series")
        if 'anime' not in query_lower:
            suggestions.append(f"{query} anime")
        
        # Remove duplicates and limit
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion.lower() not in seen and suggestion.lower() != query_lower:
                seen.add(suggestion.lower())
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:5]
    
    def _get_spelling_corrections(self, query: str) -> List[str]:
        """
        Get spelling corrections for the query
        """
        corrections = []
        
        # Use fuzzy matching against known titles
        all_titles = list(self.indexer.index['titles'].keys())
        if all_titles:
            # Get best matches
            matches = process.extract(
                query.lower(),
                all_titles,
                scorer=fuzz.ratio,
                limit=3
            )
            
            for match, score, _ in matches:
                if score > 60 and score < 95:  # Not exact but close
                    corrections.append(match.title())
        
        return corrections
    
    def _merge_and_rank_results(self, results: Dict, query: str, limit: int) -> List[Dict]:
        """
        Merge and rank results from different sources
        """
        all_results = []
        seen_ids = set()
        
        # Priority order: instant > local > external
        for source in ['instant_results', 'local', 'external']:
            for result in results.get(source, []):
                if isinstance(result, dict) and 'id' in result:
                    if result['id'] not in seen_ids:
                        seen_ids.add(result['id'])
                        # Add source info
                        result['source'] = source
                        all_results.append(result)
        
        # Re-rank based on relevance
        query_lower = query.lower()
        for result in all_results:
            score = 0
            
            # Title match scoring
            title_lower = result.get('title', '').lower()
            if query_lower == title_lower:
                score += 100
            elif query_lower in title_lower:
                score += 50
            else:
                # Fuzzy match
                score += fuzz.ratio(query_lower, title_lower) * 0.3
            
            # Boost by popularity and rating
            score += (result.get('popularity', 0) / 100) * 10
            score += (result.get('rating', 0) / 10) * 20
            
            # Boost by source
            if result['source'] == 'instant_results':
                score += 30
            elif result['source'] == 'local':
                score += 20
            
            result['relevance_score'] = score
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Remove relevance score from final output
        for result in all_results:
            result.pop('relevance_score', None)
            result.pop('source', None)
        
        return all_results[:limit]
    
    def _format_results(self, content_list: List[Any]) -> List[Dict]:
        """Format content objects for response"""
        formatted = []
        for content in content_list:
            formatted.append(self._format_content(content))
        return formatted
    
    def _format_content(self, content: Any) -> Dict:
        """Format a single content object"""
        youtube_url = None
        if hasattr(content, 'youtube_trailer_id') and content.youtube_trailer_id:
            youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
        
        return {
            'id': content.id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'rating': content.rating,
            'popularity': content.popularity,
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'poster_path': self._get_poster_url(content.poster_path),
            'overview': content.overview[:200] + '...' if content.overview and len(content.overview) > 200 else content.overview,
            'youtube_trailer': youtube_url
        }
    
    def _get_poster_url(self, poster_path: str) -> Optional[str]:
        """Get full poster URL"""
        if not poster_path:
            return None
        if poster_path.startswith('http'):
            return poster_path
        return f"https://image.tmdb.org/t/p/w300{poster_path}"
    
    def _get_content_by_ids(self, content_ids: List[int]) -> List[Any]:
        """Get content objects by IDs"""
        from models import Content  # Import here to avoid circular imports
        return Content.query.filter(Content.id.in_(content_ids)).all()
    
    def _generate_cache_key(self, query: str, search_type: str, page: int, filters: Dict) -> str:
        """Generate cache key for search"""
        filter_str = json.dumps(filters, sort_keys=True) if filters else ''
        key_data = f"{query}:{search_type}:{page}:{filter_str}"
        return f"search:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached search result"""
        try:
            cached = self.cache.get(cache_key)
            if cached:
                return json.loads(cached) if isinstance(cached, str) else cached
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        return None
    
    def _cache_result(self, cache_key: str, result: Dict) -> None:
        """Cache search result"""
        try:
            # Cache for 5 minutes for instant results
            self.cache.set(cache_key, json.dumps(result), timeout=300)
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

class SmartSuggestionEngine:
    """Intelligent suggestion engine for search"""
    
    def __init__(self, cache):
        self.cache = cache
        self.trending_queries = []
        self.user_history = defaultdict(list)
        
    def get_trending_searches(self, limit: int = 10) -> List[str]:
        """Get trending search queries"""
        cache_key = "trending_searches"
        cached = self.cache.get(cache_key)
        
        if cached:
            return json.loads(cached)[:limit]
        
        # Default trending searches
        trending = [
            "Telugu movies",
            "New releases",
            "Action movies",
            "Romantic series",
            "Anime",
            "Thriller",
            "Comedy shows",
            "Marvel",
            "Netflix originals",
            "2024 movies"
        ]
        
        # Cache for 1 hour
        self.cache.set(cache_key, json.dumps(trending), timeout=3600)
        return trending[:limit]
    
    def record_search(self, query: str, user_id: Optional[int] = None) -> None:
        """Record a search query for analytics"""
        try:
            # Update trending queries
            cache_key = "search_analytics"
            analytics = self.cache.get(cache_key)
            
            if analytics:
                analytics = json.loads(analytics)
            else:
                analytics = {'queries': {}, 'last_update': datetime.now().isoformat()}
            
            # Increment query count
            if 'queries' not in analytics:
                analytics['queries'] = {}
            
            analytics['queries'][query] = analytics['queries'].get(query, 0) + 1
            analytics['last_update'] = datetime.now().isoformat()
            
            # Cache updated analytics
            self.cache.set(cache_key, json.dumps(analytics), timeout=86400)  # 24 hours
            
            # Update user history if user_id provided
            if user_id:
                user_key = f"user_search_history:{user_id}"
                history = self.cache.get(user_key)
                
                if history:
                    history = json.loads(history)
                else:
                    history = []
                
                # Add to history (keep last 20)
                history.insert(0, {'query': query, 'timestamp': datetime.now().isoformat()})
                history = history[:20]
                
                self.cache.set(user_key, json.dumps(history), timeout=2592000)  # 30 days
                
        except Exception as e:
            logger.error(f"Failed to record search: {e}")
    
    def get_personalized_suggestions(self, user_id: int, limit: int = 5) -> List[str]:
        """Get personalized search suggestions for a user"""
        try:
            user_key = f"user_search_history:{user_id}"
            history = self.cache.get(user_key)
            
            if history:
                history = json.loads(history)
                # Extract unique recent queries
                recent_queries = []
                seen = set()
                for item in history:
                    query = item['query']
                    if query not in seen:
                        seen.add(query)
                        recent_queries.append(query)
                
                return recent_queries[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get personalized suggestions: {e}")
        
        return []

# Export the main search interface
def create_search_engine(db, cache, services):
    """Factory function to create search engine"""
    return HybridSearch(db, cache, services)

def create_suggestion_engine(cache):
    """Factory function to create suggestion engine"""
    return SmartSuggestionEngine(cache)