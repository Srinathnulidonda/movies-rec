# backend/search.py
import json
import logging
import re
import time
import unicodedata
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from collections import defaultdict, Counter, OrderedDict
import hashlib
from functools import lru_cache, wraps
from dataclasses import dataclass, field

import numpy as np
from flask import current_app
from sqlalchemy import or_, and_, func, text, case
from sqlalchemy.orm import Query, joinedload
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
import redis
import pickle
from threading import Lock

logger = logging.getLogger(__name__)

# Circuit breaker for external APIs
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = Lock()
    
    def call(self, func, *args, **kwargs):
        with self._lock:
            if self.state == 'open':
                if datetime.now().timestamp() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'half-open'
                else:
                    return None
            
            try:
                result = func(*args, **kwargs)
                if self.state == 'half-open':
                    self.state = 'closed'
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.now().timestamp()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                    logger.warning(f"Circuit breaker opened for {func.__name__}")
                
                raise e

@dataclass
class SearchResult:
    """Data class for search results with scoring"""
    id: int
    title: str
    content_type: str
    score: float = 0.0
    exact_match: bool = False
    fuzzy_score: float = 0.0
    popularity_score: float = 0.0
    recency_score: float = 0.0
    language_boost: float = 0.0
    source: str = 'local'
    metadata: Dict = field(default_factory=dict)

class AdvancedSearchIndexer:
    """Enhanced in-memory search index with n-gram support and better scoring"""
    
    def __init__(self):
        self.index = {
            'titles': {},  # exact title -> content_id
            'normalized': {},  # normalized title -> content_id
            'ngrams': defaultdict(set),  # ngram -> set of content_ids
            'tokens': defaultdict(set),  # token -> set of content_ids
            'prefixes': defaultdict(set),  # prefix -> set of content_ids
            'genres': defaultdict(set),
            'languages': defaultdict(set),
            'years': defaultdict(set),
            'content_map': {},  # content_id -> full content data
            'popularity_scores': {},  # content_id -> popularity score
            'title_vectors': {},  # content_id -> title vector for similarity
        }
        self.last_update = None
        self.update_interval = 180  # 3 minutes
        self._lock = Lock()
        self.stop_words = self._load_stop_words()
        
    def _load_stop_words(self) -> Set[str]:
        """Load common stop words for multiple languages"""
        return {
            # English
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
            # Hindi
            'का', 'के', 'की', 'में', 'है', 'और', 'को', 'से', 'ने', 'पर',
            # Telugu (common particles)
            'మరియు', 'లో', 'నుండి', 'కోసం'
        }
    
    def needs_update(self) -> bool:
        """Check if index needs updating"""
        if not self.last_update:
            return True
        return (datetime.now() - self.last_update).seconds > self.update_interval
    
    def build_index(self, content_list: List[Any]) -> None:
        """Build comprehensive search index with better tokenization"""
        logger.info(f"Building advanced search index with {len(content_list)} items")
        start_time = time.time()
        
        with self._lock:
            # Reset index
            self._reset_index()
            
            for content in content_list:
                try:
                    self._index_content(content)
                except Exception as e:
                    logger.error(f"Error indexing content {content.id}: {e}")
                    continue
            
            # Build derived indexes
            self._build_popularity_scores()
            self._build_title_vectors()
            
            self.last_update = datetime.now()
        
        elapsed = time.time() - start_time
        logger.info(f"Advanced search index built in {elapsed:.2f} seconds")
    
    def _reset_index(self):
        """Reset all index structures"""
        self.index = {
            'titles': {},
            'normalized': {},
            'ngrams': defaultdict(set),
            'tokens': defaultdict(set),
            'prefixes': defaultdict(set),
            'genres': defaultdict(set),
            'languages': defaultdict(set),
            'years': defaultdict(set),
            'content_map': {},
            'popularity_scores': {},
            'title_vectors': {},
        }
    
    def _index_content(self, content):
        """Index a single content item comprehensively"""
        content_id = content.id
        
        # Store full content data
        self.index['content_map'][content_id] = {
            'id': content_id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'release_date': content.release_date,
            'rating': content.rating or 0,
            'vote_count': content.vote_count or 0,
            'popularity': content.popularity or 0,
            'overview': content.overview,
            'poster_path': content.poster_path,
            'youtube_trailer_id': content.youtube_trailer_id,
            'is_trending': getattr(content, 'is_trending', False),
            'is_new_release': getattr(content, 'is_new_release', False),
        }
        
        # Index titles
        if content.title:
            # Exact title
            title_lower = content.title.lower().strip()
            self.index['titles'][title_lower] = content_id
            
            # Normalized title
            normalized = self._normalize_text(content.title)
            if normalized:
                self.index['normalized'][normalized] = content_id
            
            # Tokenize and index
            tokens = self._tokenize(content.title)
            for token in tokens:
                self.index['tokens'][token].add(content_id)
            
            # N-grams for fuzzy matching
            ngrams = self._generate_ngrams(title_lower, n=3)
            for ngram in ngrams:
                self.index['ngrams'][ngram].add(content_id)
            
            # Prefixes for autocomplete
            self._index_prefixes(title_lower, content_id)
            
            # Also index words from title
            words = title_lower.split()
            for word in words:
                if len(word) > 2 and word not in self.stop_words:
                    self.index['tokens'][word].add(content_id)
                    self._index_prefixes(word, content_id)
        
        # Index original title if different
        if content.original_title and content.original_title != content.title:
            orig_lower = content.original_title.lower().strip()
            self.index['titles'][orig_lower] = content_id
            tokens = self._tokenize(content.original_title)
            for token in tokens:
                self.index['tokens'][token].add(content_id)
        
        # Index metadata
        genres = json.loads(content.genres or '[]')
        for genre in genres:
            self.index['genres'][genre.lower()].add(content_id)
        
        languages = json.loads(content.languages or '[]')
        for language in languages:
            self.index['languages'][language.lower()].add(content_id)
        
        if content.release_date:
            self.index['years'][content.release_date.year].add(content_id)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching across languages"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove accents and diacritics
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into searchable tokens"""
        if not text:
            return []
        
        normalized = self._normalize_text(text)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+', normalized.lower())
        
        # Filter out stop words and very short tokens
        tokens = [t for t in tokens if len(t) > 1 and t not in self.stop_words]
        
        # Add bigrams for compound words
        bigrams = []
        for i in range(len(tokens) - 1):
            bigrams.append(f"{tokens[i]}_{tokens[i+1]}")
        
        return tokens + bigrams
    
    def _generate_ngrams(self, text: str, n: int = 3) -> Set[str]:
        """Generate character n-grams for fuzzy matching"""
        ngrams = set()
        text = text.lower().replace(' ', '')
        
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i+n])
        
        return ngrams
    
    def _index_prefixes(self, text: str, content_id: int, max_length: int = 10):
        """Index prefixes for autocomplete"""
        text = text.lower().strip()
        
        # Index progressive prefixes
        for i in range(1, min(len(text) + 1, max_length + 1)):
            prefix = text[:i]
            self.index['prefixes'][prefix].add(content_id)
    
    def _build_popularity_scores(self):
        """Calculate normalized popularity scores"""
        max_pop = 0
        max_rating = 10
        max_votes = 0
        
        # Find maximums
        for content_data in self.index['content_map'].values():
            max_pop = max(max_pop, content_data.get('popularity', 0))
            max_votes = max(max_votes, content_data.get('vote_count', 0))
        
        # Calculate normalized scores
        for content_id, content_data in self.index['content_map'].items():
            pop_score = 0
            
            # Popularity component (0-40)
            if max_pop > 0:
                pop_score += (content_data.get('popularity', 0) / max_pop) * 40
            
            # Rating component (0-30)
            rating = content_data.get('rating', 0)
            pop_score += (rating / max_rating) * 30
            
            # Vote count component (0-20)
            if max_votes > 0:
                votes = content_data.get('vote_count', 0)
                pop_score += (votes / max_votes) * 20
            
            # Recency bonus (0-10)
            if content_data.get('is_new_release'):
                pop_score += 10
            elif content_data.get('release_date'):
                days_old = (datetime.now().date() - content_data['release_date']).days
                if days_old < 365:
                    pop_score += max(0, 10 - (days_old / 36.5))
            
            self.index['popularity_scores'][content_id] = pop_score
    
    def _build_title_vectors(self):
        """Build simple title vectors for similarity computation"""
        # This is a simplified version - in production, use proper embeddings
        for content_id, content_data in self.index['content_map'].items():
            title = content_data.get('title', '')
            tokens = self._tokenize(title)
            
            # Simple bag-of-words vector
            vector = Counter(tokens)
            self.index['title_vectors'][content_id] = vector
    
    def search(self, query: str, limit: int = 20, filters: Dict = None) -> List[SearchResult]:
        """Perform intelligent search with multiple strategies"""
        query_lower = query.lower().strip()
        query_normalized = self._normalize_text(query)
        query_tokens = self._tokenize(query)
        query_ngrams = self._generate_ngrams(query_lower)
        
        results = {}  # content_id -> SearchResult
        
        with self._lock:
            # 1. Exact title match (highest priority)
            if query_lower in self.index['titles']:
                content_id = self.index['titles'][query_lower]
                results[content_id] = SearchResult(
                    id=content_id,
                    title=self.index['content_map'][content_id]['title'],
                    content_type=self.index['content_map'][content_id]['content_type'],
                    score=1000,
                    exact_match=True
                )
            
            # 2. Normalized title match
            if query_normalized in self.index['normalized']:
                content_id = self.index['normalized'][query_normalized]
                if content_id not in results:
                    results[content_id] = SearchResult(
                        id=content_id,
                        title=self.index['content_map'][content_id]['title'],
                        content_type=self.index['content_map'][content_id]['content_type'],
                        score=900
                    )
            
            # 3. Prefix match for autocomplete
            prefix_matches = self.index['prefixes'].get(query_lower, set())
            for content_id in prefix_matches:
                if content_id not in results:
                    content_data = self.index['content_map'][content_id]
                    title_lower = content_data['title'].lower()
                    
                    # Calculate prefix score based on position
                    if title_lower.startswith(query_lower):
                        score = 800
                    else:
                        # Check if any word starts with query
                        words = title_lower.split()
                        word_match = any(w.startswith(query_lower) for w in words)
                        score = 700 if word_match else 600
                    
                    results[content_id] = SearchResult(
                        id=content_id,
                        title=content_data['title'],
                        content_type=content_data['content_type'],
                        score=score
                    )
            
            # 4. Token-based search
            token_matches = defaultdict(int)
            for token in query_tokens:
                if token in self.index['tokens']:
                    for content_id in self.index['tokens'][token]:
                        token_matches[content_id] += 1
            
            # Score based on number of matching tokens
            for content_id, match_count in token_matches.items():
                if content_id not in results:
                    content_data = self.index['content_map'][content_id]
                    score = (match_count / len(query_tokens)) * 500 if query_tokens else 0
                    
                    results[content_id] = SearchResult(
                        id=content_id,
                        title=content_data['title'],
                        content_type=content_data['content_type'],
                        score=score
                    )
                else:
                    # Boost existing result
                    results[content_id].score += (match_count / len(query_tokens)) * 200
            
            # 5. N-gram similarity for fuzzy matching
            if len(query_ngrams) > 0:
                ngram_scores = defaultdict(float)
                
                for ngram in query_ngrams:
                    if ngram in self.index['ngrams']:
                        for content_id in self.index['ngrams'][ngram]:
                            ngram_scores[content_id] += 1
                
                # Calculate Jaccard similarity
                for content_id, ngram_count in ngram_scores.items():
                    if content_id not in results:
                        content_data = self.index['content_map'][content_id]
                        title_ngrams = self._generate_ngrams(content_data['title'].lower())
                        
                        if len(title_ngrams) > 0:
                            jaccard = ngram_count / len(title_ngrams.union(query_ngrams))
                            score = jaccard * 400
                            
                            results[content_id] = SearchResult(
                                id=content_id,
                                title=content_data['title'],
                                content_type=content_data['content_type'],
                                score=score,
                                fuzzy_score=jaccard
                            )
                    else:
                        # Add fuzzy bonus to existing results
                        content_data = self.index['content_map'][content_id]
                        title_ngrams = self._generate_ngrams(content_data['title'].lower())
                        if len(title_ngrams) > 0:
                            jaccard = ngram_count / len(title_ngrams.union(query_ngrams))
                            results[content_id].score += jaccard * 100
                            results[content_id].fuzzy_score = jaccard
            
            # 6. Apply popularity boost
            for content_id, result in results.items():
                pop_score = self.index['popularity_scores'].get(content_id, 0)
                result.popularity_score = pop_score
                result.score += pop_score * 2  # Popularity contributes up to 200 points
            
            # 7. Apply language boost (Telugu priority)
            telugu_boost = 50
            for content_id, result in results.items():
                content_data = self.index['content_map'][content_id]
                languages = content_data.get('languages', [])
                
                # Check for Telugu content
                for lang in languages:
                    if isinstance(lang, str):
                        lang_lower = lang.lower()
                        if 'telugu' in lang_lower or lang_lower == 'te':
                            result.language_boost = telugu_boost
                            result.score += telugu_boost
                            break
            
            # 8. Apply filters
            if filters:
                filtered_results = {}
                for content_id, result in results.items():
                    if self._apply_filter(content_id, filters):
                        filtered_results[content_id] = result
                results = filtered_results
        
        # Convert to list and sort by score
        result_list = list(results.values())
        result_list.sort(key=lambda x: x.score, reverse=True)
        
        return result_list[:limit]
    
    def _apply_filter(self, content_id: int, filters: Dict) -> bool:
        """Apply filters to a content item"""
        content_data = self.index['content_map'].get(content_id)
        if not content_data:
            return False
        
        # Genre filter
        if filters.get('genre'):
            genres = [g.lower() for g in content_data.get('genres', [])]
            if filters['genre'].lower() not in genres:
                return False
        
        # Language filter
        if filters.get('language'):
            languages = [l.lower() for l in content_data.get('languages', [])]
            if filters['language'].lower() not in languages:
                return False
        
        # Year filter
        if filters.get('year'):
            if not content_data.get('release_date'):
                return False
            if content_data['release_date'].year != filters['year']:
                return False
        
        # Content type filter
        if filters.get('content_type'):
            if content_data['content_type'] != filters['content_type']:
                return False
        
        # Rating filter
        if filters.get('min_rating'):
            if content_data.get('rating', 0) < filters['min_rating']:
                return False
        
        return True

class ImprovedHybridSearch:
    """Improved hybrid search with better error handling and result quality"""
    
    def __init__(self, db, cache, services):
        self.db = db
        self.cache = cache
        self.services = services
        self.indexer = AdvancedSearchIndexer()
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.circuit_breaker = CircuitBreaker()
        self._ensure_index_built()
    
    def _ensure_index_built(self):
        """Ensure search index is built on initialization"""
        try:
            if self.indexer.needs_update():
                self._update_search_index()
        except Exception as e:
            logger.error(f"Failed to build initial index: {e}")
    
    def search(self, query: str, search_type: str = 'multi', page: int = 1,
               limit: int = 20, filters: Dict = None) -> Dict[str, Any]:
        """
        Improved search with better result quality and error handling
        """
        # Validate input
        if not query or len(query.strip()) < 2:
            return {
                'results': [],
                'total_results': 0,
                'page': page,
                'query': query,
                'search_time': 0,
                'suggestions': [],
                'error': 'Query must be at least 2 characters'
            }
        
        query = query.strip()
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, search_type, page, filters)
        
        # Try cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Cache hit for search: {query}")
            cached_result['cached'] = True
            cached_result['search_time'] = 0.001
            return cached_result
        
        # Update index if needed
        if self.indexer.needs_update():
            self._update_search_index()
        
        # Perform multi-strategy search
        all_results = []
        
        try:
            # 1. In-memory index search (fastest)
            index_results = self.indexer.search(query, limit * 2, filters)
            
            # 2. Database search (comprehensive)
            db_results = self._enhanced_database_search(query, search_type, filters, limit)
            
            # 3. External API search (if needed)
            external_results = []
            if len(index_results) < 5:  # Only call external APIs if we have few local results
                external_results = self._safe_external_search(query, search_type, page)
            
            # Merge all results
            all_results = self._merge_all_results(
                index_results, db_results, external_results, query, limit
            )
            
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
            # Fallback to basic database search
            try:
                all_results = self._fallback_search(query, limit)
            except:
                all_results = []
        
        # Format final results
        formatted_results = self._format_final_results(all_results[:limit])
        
        # Generate suggestions
        suggestions = self._generate_smart_suggestions(query, formatted_results[:5])
        
        # Prepare response
        response = {
            'results': formatted_results,
            'total_results': len(formatted_results),
            'page': page,
            'query': query,
            'search_time': round(time.time() - start_time, 3),
            'suggestions': suggestions,
            'filters_applied': filters or {},
            'cached': False
        }
        
        # Cache successful results
        if formatted_results:
            self._cache_result(cache_key, response)
        
        return response
    
    def autocomplete(self, prefix: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fast autocomplete with instant results
        """
        if len(prefix) < 2:
            return []
        
        prefix_lower = prefix.lower().strip()
        
        # Check cache
        cache_key = f"autocomplete:{prefix_lower}"
        cached = self.cache.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except:
                pass
        
        suggestions = []
        
        # Search in index
        results = self.indexer.search(prefix, limit * 2)
        
        for result in results[:limit]:
            content_data = self.indexer.index['content_map'].get(result.id)
            if content_data:
                suggestions.append({
                    'id': result.id,
                    'title': content_data['title'],
                    'type': content_data['content_type'],
                    'year': content_data['release_date'].year if content_data['release_date'] else None,
                    'poster': self._get_poster_url(content_data['poster_path']),
                    'rating': content_data.get('rating', 0),
                    'match_type': 'exact' if result.exact_match else 'fuzzy'
                })
        
        # Cache for 30 minutes
        if suggestions:
            try:
                self.cache.set(cache_key, json.dumps(suggestions), timeout=1800)
            except:
                pass
        
        return suggestions
    
    def _update_search_index(self):
        """Update search index with latest content"""
        try:
            # Import Content model
            from app import Content
            
            # Get all content (with limit for memory management)
            content_query = Content.query.order_by(
                Content.popularity.desc(),
                Content.rating.desc()
            ).limit(50000)  # Increased limit
            
            content_list = content_query.all()
            
            if content_list:
                self.indexer.build_index(content_list)
                logger.info(f"Search index updated with {len(content_list)} items")
            
        except Exception as e:
            logger.error(f"Failed to update search index: {e}")
    
    def _enhanced_database_search(self, query: str, search_type: str, 
                                  filters: Dict, limit: int) -> List[Any]:
        """
        Enhanced database search with better ranking
        """
        try:
            from app import Content
            
            query_lower = query.lower()
            
            # Build weighted search query
            search_query = self.db.session.query(
                Content,
                case(
                    [
                        (func.lower(Content.title) == query_lower, 100),
                        (func.lower(Content.title).startswith(query_lower), 90),
                        (func.lower(Content.title).contains(query_lower), 70),
                        (func.lower(Content.original_title) == query_lower, 80),
                        (func.lower(Content.original_title).contains(query_lower), 60),
                        (func.lower(Content.overview).contains(query_lower), 30),
                    ],
                    else_=0
                ).label('relevance_score')
            )
            
            # Add filters
            conditions = []
            
            # Content type filter
            if search_type != 'multi':
                if search_type == 'movie':
                    conditions.append(Content.content_type == 'movie')
                elif search_type == 'tv':
                    conditions.append(Content.content_type == 'tv')
                elif search_type == 'anime':
                    conditions.append(Content.content_type == 'anime')
            
            # Apply additional filters
            if filters:
                if filters.get('genre'):
                    conditions.append(Content.genres.contains(filters['genre']))
                if filters.get('language'):
                    conditions.append(Content.languages.contains(filters['language']))
                if filters.get('year'):
                    year = int(filters['year'])
                    conditions.append(
                        func.extract('year', Content.release_date) == year
                    )
                if filters.get('min_rating'):
                    conditions.append(Content.rating >= filters['min_rating'])
            
            # Apply all conditions
            if conditions:
                search_query = search_query.filter(and_(*conditions))
            
            # Filter by relevance score > 0
            search_query = search_query.filter(
                case(
                    [
                        (func.lower(Content.title) == query_lower, 100),
                        (func.lower(Content.title).startswith(query_lower), 90),
                        (func.lower(Content.title).contains(query_lower), 70),
                        (func.lower(Content.original_title) == query_lower, 80),
                        (func.lower(Content.original_title).contains(query_lower), 60),
                        (func.lower(Content.overview).contains(query_lower), 30),
                    ],
                    else_=0
                ) > 0
            )
            
            # Order by relevance, then popularity
            search_query = search_query.order_by(
                text('relevance_score DESC'),
                Content.popularity.desc(),
                Content.rating.desc()
            )
            
            # Execute query
            results = search_query.limit(limit).all()
            
            # Extract Content objects
            return [result[0] for result in results]
            
        except Exception as e:
            logger.error(f"Database search error: {e}")
            return []
    
    def _safe_external_search(self, query: str, search_type: str, page: int) -> List[Dict]:
        """
        Safe external API search with circuit breaker
        """
        external_results = []
        
        try:
            # Use circuit breaker for external calls
            def search_tmdb():
                if search_type in ['multi', 'movie', 'tv']:
                    return self.services['TMDBService'].search_content(
                        query, search_type, page=page
                    )
                return None
            
            def search_anime():
                if search_type in ['multi', 'anime']:
                    return self.services['JikanService'].search_anime(query, page)
                return None
            
            # Try TMDB
            tmdb_result = self.circuit_breaker.call(search_tmdb)
            if tmdb_result and 'results' in tmdb_result:
                for item in tmdb_result['results'][:5]:
                    try:
                        content_type = 'movie' if 'title' in item else 'tv'
                        content = self.services['ContentService'].save_content_from_tmdb(
                            item, content_type
                        )
                        if content:
                            external_results.append({
                                'content': content,
                                'source': 'tmdb'
                            })
                    except Exception as e:
                        logger.error(f"Error processing TMDB result: {e}")
            
            # Try Jikan for anime
            anime_result = self.circuit_breaker.call(search_anime)
            if anime_result and 'data' in anime_result:
                for anime in anime_result['data'][:5]:
                    try:
                        content = self.services['ContentService'].save_anime_content(anime)
                        if content:
                            external_results.append({
                                'content': content,
                                'source': 'jikan'
                            })
                    except Exception as e:
                        logger.error(f"Error processing anime result: {e}")
            
        except Exception as e:
            logger.error(f"External search error: {e}")
        
        return external_results
    
    def _fallback_search(self, query: str, limit: int) -> List[Any]:
        """
        Simple fallback search when main search fails
        """
        try:
            from app import Content
            
            return Content.query.filter(
                or_(
                    Content.title.ilike(f'%{query}%'),
                    Content.original_title.ilike(f'%{query}%')
                )
            ).order_by(
                Content.popularity.desc()
            ).limit(limit).all()
            
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return []
    
    def _merge_all_results(self, index_results: List[SearchResult],
                          db_results: List[Any], 
                          external_results: List[Dict],
                          query: str, limit: int) -> List[Any]:
        """
        Intelligently merge results from all sources
        """
        merged = OrderedDict()  # Preserve order while removing duplicates
        query_lower = query.lower()
        
        # Process index results (highest priority)
        for result in index_results:
            content = self._get_content_by_id(result.id)
            if content and content.id not in merged:
                # Add relevance metadata
                content._search_score = result.score
                content._exact_match = result.exact_match
                content._source = 'index'
                merged[content.id] = content
        
        # Process database results
        for content in db_results:
            if content and content.id not in merged:
                # Calculate simple relevance score
                title_lower = (content.title or '').lower()
                if query_lower == title_lower:
                    score = 900
                elif title_lower.startswith(query_lower):
                    score = 700
                elif query_lower in title_lower:
                    score = 500
                else:
                    score = 300
                
                content._search_score = score
                content._exact_match = False
                content._source = 'database'
                merged[content.id] = content
        
        # Process external results
        for result in external_results:
            content = result.get('content')
            if content and content.id not in merged:
                content._search_score = 200  # Lower score for external
                content._exact_match = False
                content._source = result.get('source', 'external')
                merged[content.id] = content
        
        # Sort by score
        sorted_results = sorted(
            merged.values(),
            key=lambda x: getattr(x, '_search_score', 0),
            reverse=True
        )
        
        return sorted_results[:limit]
    
    def _format_final_results(self, results: List[Any]) -> List[Dict]:
        """
        Format results for API response
        """
        formatted = []
        
        for content in results:
            try:
                youtube_url = None
                if hasattr(content, 'youtube_trailer_id') and content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                formatted.append({
                    'id': content.id,
                    'title': content.title,
                    'original_title': content.original_title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': float(content.rating) if content.rating else 0,
                    'vote_count': content.vote_count or 0,
                    'popularity': float(content.popularity) if content.popularity else 0,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'year': content.release_date.year if content.release_date else None,
                    'poster_path': self._get_poster_url(content.poster_path),
                    'overview': self._truncate_text(content.overview, 200),
                    'youtube_trailer': youtube_url,
                    'is_trending': getattr(content, 'is_trending', False),
                    'is_new_release': getattr(content, 'is_new_release', False),
                    'match_info': {
                        'exact_match': getattr(content, '_exact_match', False),
                        'source': getattr(content, '_source', 'unknown'),
                        'score': getattr(content, '_search_score', 0)
                    }
                })
            except Exception as e:
                logger.error(f"Error formatting result: {e}")
                continue
        
        return formatted
    
    def _generate_smart_suggestions(self, query: str, top_results: List[Dict]) -> List[str]:
        """
        Generate intelligent search suggestions
        """
        suggestions = []
        query_lower = query.lower()
        
        # Extract patterns from results
        if top_results:
            # Get common genres
            all_genres = []
            for result in top_results:
                all_genres.extend(result.get('genres', []))
            
            genre_counts = Counter(all_genres)
            top_genres = [genre for genre, _ in genre_counts.most_common(2)]
            
            # Suggest genre refinements
            for genre in top_genres:
                if genre.lower() not in query_lower:
                    suggestions.append(f"{query} {genre}")
            
            # Get common content types
            content_types = [r.get('content_type') for r in top_results]
            type_counts = Counter(content_types)
            
            # Suggest type refinements
            if type_counts:
                most_common_type = type_counts.most_common(1)[0][0]
                if most_common_type == 'movie' and 'movie' not in query_lower:
                    suggestions.append(f"{query} movies")
                elif most_common_type == 'tv' and 'series' not in query_lower:
                    suggestions.append(f"{query} series")
                elif most_common_type == 'anime' and 'anime' not in query_lower:
                    suggestions.append(f"{query} anime")
        
        # Add year suggestions
        current_year = datetime.now().year
        if str(current_year) not in query:
            suggestions.append(f"{query} {current_year}")
        
        # Add language suggestions
        if 'telugu' not in query_lower:
            suggestions.append(f"{query} Telugu")
        if 'hindi' not in query_lower:
            suggestions.append(f"{query} Hindi")
        
        # Remove duplicates and limit
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion.lower() not in seen and suggestion.lower() != query_lower:
                seen.add(suggestion.lower())
                unique_suggestions.append(suggestion)
                if len(unique_suggestions) >= 5:
                    break
        
        return unique_suggestions
    
    def _get_content_by_id(self, content_id: int):
        """Get content object by ID"""
        try:
            from app import Content
            return Content.query.get(content_id)
        except Exception as e:
            logger.error(f"Error getting content by ID: {e}")
            return None
    
    def _get_poster_url(self, poster_path: str) -> Optional[str]:
        """Get full poster URL"""
        if not poster_path:
            return None
        if poster_path.startswith('http'):
            return poster_path
        return f"https://image.tmdb.org/t/p/w300{poster_path}"
    
    def _truncate_text(self, text: str, length: int) -> str:
        """Truncate text to specified length"""
        if not text:
            return ""
        if len(text) <= length:
            return text
        return text[:length] + "..."
    
    def _generate_cache_key(self, query: str, search_type: str, 
                           page: int, filters: Dict) -> str:
        """Generate unique cache key"""
        filter_str = json.dumps(filters, sort_keys=True) if filters else ''
        key_data = f"{query}:{search_type}:{page}:{filter_str}"
        return f"search:v2:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached result safely"""
        try:
            cached = self.cache.get(cache_key)
            if cached:
                if isinstance(cached, str):
                    return json.loads(cached)
                return cached
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        return None
    
    def _cache_result(self, cache_key: str, result: Dict) -> None:
        """Cache result safely"""
        try:
            # Don't cache error responses
            if result.get('results'):
                # Cache for 5 minutes
                self.cache.set(cache_key, json.dumps(result), timeout=300)
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

# Keep the SmartSuggestionEngine as is (it's already good)
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
            try:
                return json.loads(cached)[:limit]
            except:
                pass
        
        # Default trending searches with Telugu priority
        trending = [
            "Telugu movies 2024",
            "Latest Telugu releases",
            "Pushpa",
            "RRR",
            "New releases",
            "Action movies",
            "Romantic series",
            "Best anime",
            "Thriller movies",
            "Comedy shows"
        ]
        
        # Cache for 1 hour
        try:
            self.cache.set(cache_key, json.dumps(trending), timeout=3600)
        except:
            pass
        
        return trending[:limit]
    
    def record_search(self, query: str, user_id: Optional[int] = None) -> None:
        """Record a search query for analytics"""
        try:
            # Update trending queries
            cache_key = "search_analytics"
            analytics = self.cache.get(cache_key)
            
            if analytics:
                try:
                    analytics = json.loads(analytics)
                except:
                    analytics = {'queries': {}, 'last_update': datetime.now().isoformat()}
            else:
                analytics = {'queries': {}, 'last_update': datetime.now().isoformat()}
            
            # Increment query count
            analytics['queries'][query] = analytics['queries'].get(query, 0) + 1
            analytics['last_update'] = datetime.now().isoformat()
            
            # Update trending based on frequency
            if len(analytics['queries']) > 10:
                # Get top 10 queries
                top_queries = sorted(
                    analytics['queries'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                # Update trending cache
                trending = [q[0] for q in top_queries]
                self.cache.set("trending_searches", json.dumps(trending), timeout=3600)
            
            # Cache updated analytics
            self.cache.set(cache_key, json.dumps(analytics), timeout=86400)
            
            # Update user history if user_id provided
            if user_id:
                user_key = f"user_search_history:{user_id}"
                history = self.cache.get(user_key)
                
                if history:
                    try:
                        history = json.loads(history)
                    except:
                        history = []
                else:
                    history = []
                
                # Add to history (keep last 20)
                history.insert(0, {
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                })
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
                try:
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
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Failed to get personalized suggestions: {e}")
        
        return []

# Export factory functions
def create_search_engine(db, cache, services):
    """Factory function to create improved search engine"""
    return ImprovedHybridSearch(db, cache, services)

def create_suggestion_engine(cache):
    """Factory function to create suggestion engine"""
    return SmartSuggestionEngine(cache)