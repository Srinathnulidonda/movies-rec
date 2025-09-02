# backend/search.py
import json
import logging
import re
import time
import unicodedata
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter, OrderedDict
import hashlib
from functools import lru_cache, wraps
import threading

import numpy as np
from flask import current_app
from sqlalchemy import or_, and_, func, text, case
from sqlalchemy.orm import Query
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
import redis
import pickle
import Levenshtein as lev

logger = logging.getLogger(__name__)

# Language priorities from app.py
LANGUAGE_PRIORITY = {
    'telugu': 100,
    'english': 90,
    'hindi': 85,
    'malayalam': 80,
    'kannada': 75,
    'tamil': 70
}

# Telugu transliteration patterns for better search
TELUGU_TRANSLITERATIONS = {
    'baahubali': ['bahubali', 'baahubali', 'బాహుబలి'],
    'pushpa': ['pushpa', 'పుష్ప'],
    'rrr': ['rrr', 'rajamouli', 'ఆర్ఆర్ఆర్'],
    'kgf': ['kgf', 'కేజీఎఫ్'],
    # Add more common Telugu movie transliterations
}

class UltraFastSearchIndexer:
    """Enhanced in-memory search index for ultra-fast, accurate searching"""
    
    def __init__(self):
        self.index = {
            # Primary indexes
            'titles': {},  # exact title -> content_id
            'normalized_titles': {},  # normalized -> content_id
            'title_tokens': defaultdict(set),  # token -> set of content_ids
            'title_ngrams': defaultdict(set),  # ngram -> set of content_ids
            
            # Language-specific indexes
            'telugu_titles': defaultdict(set),  # telugu title variations -> content_ids
            'language_content': defaultdict(set),  # language -> content_ids
            
            # Content metadata
            'content_map': {},  # content_id -> full content data
            'content_vectors': {},  # content_id -> search vector
            
            # Autocomplete and suggestions
            'autocomplete_trie': {},  # trie structure for instant autocomplete
            'popular_searches': OrderedDict(),  # popularity-ordered searches
            
            # Fuzzy matching
            'phonetic': defaultdict(set),  # phonetic key -> content_ids
            'metaphone': defaultdict(set),  # metaphone key -> content_ids
            
            # Category indexes
            'genres': defaultdict(set),
            'years': defaultdict(set),
            'content_types': defaultdict(set),
            'ratings': defaultdict(set),  # rating ranges -> content_ids
            
            # Performance optimization
            'hot_cache': OrderedDict(),  # LRU cache for frequent searches
            'search_stats': defaultdict(int),  # query -> search count
        }
        
        self.last_update = None
        self.update_interval = 180  # 3 minutes for more frequent updates
        self.lock = threading.RLock()
        self.is_building = False
        
    def build_index(self, content_list: List[Any]) -> None:
        """Build comprehensive search index with Telugu priority"""
        with self.lock:
            if self.is_building:
                return
            self.is_building = True
        
        try:
            logger.info(f"Building optimized search index with {len(content_list)} items")
            start_time = time.time()
            
            # Clear existing index safely
            self._clear_index()
            
            # Build in batches for better performance
            batch_size = 100
            for i in range(0, len(content_list), batch_size):
                batch = content_list[i:i + batch_size]
                self._process_batch(batch)
            
            # Build autocomplete trie
            self._build_autocomplete_trie()
            
            # Calculate content vectors for similarity
            self._calculate_content_vectors()
            
            self.last_update = datetime.now()
            elapsed = time.time() - start_time
            logger.info(f"Search index built successfully in {elapsed:.2f} seconds")
            
        finally:
            with self.lock:
                self.is_building = False
    
    def _clear_index(self):
        """Clear index while preserving structure"""
        for key in self.index:
            if isinstance(self.index[key], dict):
                self.index[key].clear()
            elif isinstance(self.index[key], defaultdict):
                self.index[key] = defaultdict(set)
            elif isinstance(self.index[key], OrderedDict):
                self.index[key] = OrderedDict()
    
    def _process_batch(self, batch: List[Any]) -> None:
        """Process a batch of content items"""
        for content in batch:
            self._index_content(content)
    
    def _index_content(self, content: Any) -> None:
        """Index a single content item comprehensively"""
        content_id = content.id
        
        # Store complete content data
        content_data = {
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
            'backdrop_path': content.backdrop_path,
            'youtube_trailer_id': content.youtube_trailer_id,
            'is_trending': content.is_trending,
            'is_new_release': content.is_new_release,
            'is_critics_choice': content.is_critics_choice
        }
        self.index['content_map'][content_id] = content_data
        
        # Index titles with multiple strategies
        if content.title:
            self._index_title(content.title, content_id, content_data)
        
        if content.original_title and content.original_title != content.title:
            self._index_title(content.original_title, content_id, content_data)
        
        # Index by language with priority
        for language in content_data['languages']:
            self.index['language_content'][language.lower()].add(content_id)
            
            # Special handling for Telugu content
            if 'telugu' in language.lower() or language.lower() == 'te':
                self.index['telugu_titles']['all'].add(content_id)
                # Add transliteration support
                self._index_telugu_variations(content.title, content_id)
        
        # Index metadata
        for genre in content_data['genres']:
            self.index['genres'][genre.lower()].add(content_id)
        
        if content.release_date:
            self.index['years'][content.release_date.year].add(content_id)
        
        self.index['content_types'][content.content_type].add(content_id)
        
        # Index by rating range
        if content.rating:
            rating_range = int(content.rating)
            self.index['ratings'][rating_range].add(content_id)
    
    def _index_title(self, title: str, content_id: int, content_data: Dict) -> None:
        """Index title with multiple strategies for accurate matching"""
        if not title:
            return
        
        title_lower = title.lower().strip()
        
        # 1. Exact title index
        self.index['titles'][title_lower] = content_id
        
        # 2. Normalized title (remove special chars, accents)
        normalized = self._normalize_title(title)
        self.index['normalized_titles'][normalized] = content_id
        
        # 3. Token-based indexing
        tokens = self._tokenize(title_lower)
        for token in tokens:
            self.index['title_tokens'][token].add(content_id)
        
        # 4. N-gram indexing for partial matches
        ngrams = self._generate_ngrams(title_lower, n=3)
        for ngram in ngrams:
            self.index['title_ngrams'][ngram].add(content_id)
        
        # 5. Phonetic indexing for typo tolerance
        phonetic_key = self._get_phonetic_key(title)
        if phonetic_key:
            self.index['phonetic'][phonetic_key].add(content_id)
        
        metaphone_key = self._get_metaphone_key(title)
        if metaphone_key:
            self.index['metaphone'][metaphone_key].add(content_id)
    
    def _index_telugu_variations(self, title: str, content_id: int) -> None:
        """Index Telugu title variations for better search"""
        if not title:
            return
        
        title_lower = title.lower()
        
        # Check for known transliterations
        for telugu_title, variations in TELUGU_TRANSLITERATIONS.items():
            for variation in variations:
                if variation in title_lower:
                    self.index['telugu_titles'][telugu_title].add(content_id)
                    for var in variations:
                        self.index['telugu_titles'][var].add(content_id)
        
        # Add common Telugu movie patterns
        telugu_patterns = ['thala', 'anna', 'garu', 'babu', 'deva', 'raja']
        for pattern in telugu_patterns:
            if pattern in title_lower:
                self.index['telugu_titles'][pattern].add(content_id)
    
    def _normalize_title(self, title: str) -> str:
        """Advanced title normalization"""
        # Remove accents
        title = unicodedata.normalize('NFKD', title).encode('ASCII', 'ignore').decode('utf-8')
        
        # Convert to lowercase and remove special characters
        title = re.sub(r'[^\w\s]', ' ', title.lower())
        
        # Remove extra spaces
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove common words that don't affect search
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = title.split()
        title = ' '.join(w for w in words if w not in stop_words or len(words) <= 2)
        
        return title
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for indexing"""
        # Split on spaces and special characters
        tokens = re.findall(r'\w+', text.lower())
        
        # Add compound tokens for better matching
        if len(tokens) > 1:
            # Add bigrams
            for i in range(len(tokens) - 1):
                tokens.append(f"{tokens[i]}_{tokens[i+1]}")
        
        return tokens
    
    def _generate_ngrams(self, text: str, n: int = 3) -> Set[str]:
        """Generate n-grams for fuzzy matching"""
        ngrams = set()
        text = text.replace(' ', '')
        
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i + n])
        
        return ngrams
    
    def _get_phonetic_key(self, text: str) -> Optional[str]:
        """Enhanced Soundex algorithm"""
        if not text:
            return None
        
        text = re.sub(r'[^A-Za-z]', '', text.upper())
        if not text:
            return None
        
        # Soundex mappings
        soundex_map = {
            'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3',
            'L': '4', 'MN': '5', 'R': '6'
        }
        
        # Keep first letter
        soundex = text[0]
        
        # Process remaining letters
        prev_code = ''
        for char in text[1:]:
            for key, code in soundex_map.items():
                if char in key:
                    if code != prev_code:
                        soundex += code
                        prev_code = code
                    break
            else:
                prev_code = ''
        
        # Pad to 4 characters
        return (soundex + '0000')[:4]
    
    def _get_metaphone_key(self, text: str) -> Optional[str]:
        """Simple Metaphone algorithm for alternative phonetic matching"""
        if not text:
            return None
        
        text = text.upper()
        text = re.sub(r'[^A-Z]', '', text)
        
        if not text:
            return None
        
        # Simplified Metaphone rules
        replacements = [
            (r'^KN', 'N'), (r'^WR', 'R'), (r'^X', 'S'),
            (r'MB$', 'M'), (r'SCH', 'SK'), (r'PH', 'F'),
            (r'([AEIOU])GH', r'\1'), (r'GH', 'G'),
            (r'CK', 'K'), (r'C([IEY])', r'S\1'),
            (r'C', 'K'), (r'DG([IEY])', r'J\1'),
            (r'Q', 'K'), (r'W([^AEIOU])', r'\1'),
            (r'Z', 'S'), (r'([^AEIOU])Y', r'\1'),
        ]
        
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        
        return text[:6]  # Limit length
    
    def _build_autocomplete_trie(self):
        """Build trie structure for instant autocomplete"""
        for content_id, content_data in self.index['content_map'].items():
            title = content_data['title']
            if title:
                self._add_to_trie(title.lower(), content_id)
                
                # Add individual words to trie
                words = title.lower().split()
                for word in words:
                    if len(word) > 2:  # Skip very short words
                        self._add_to_trie(word, content_id)
    
    def _add_to_trie(self, word: str, content_id: int):
        """Add word to trie structure"""
        node = self.index['autocomplete_trie']
        
        for char in word:
            if char not in node:
                node[char] = {'ids': set(), 'children': {}}
            node[char]['ids'].add(content_id)
            node = node[char]['children']
    
    def _calculate_content_vectors(self):
        """Calculate search vectors for content similarity"""
        for content_id, content_data in self.index['content_map'].items():
            vector = {
                'genres': set(content_data['genres']),
                'languages': set(content_data['languages']),
                'year': content_data['release_date'].year if content_data['release_date'] else None,
                'rating': content_data['rating'],
                'popularity': content_data['popularity'],
                'type': content_data['content_type']
            }
            self.index['content_vectors'][content_id] = vector
    
    def search(self, query: str, limit: int = 20, filters: Dict = None) -> List[Tuple[int, float]]:
        """Ultra-fast, accurate search with scoring"""
        query_lower = query.lower().strip()
        
        # Check hot cache first
        cache_key = f"{query_lower}:{json.dumps(filters, sort_keys=True)}"
        if cache_key in self.index['hot_cache']:
            return self.index['hot_cache'][cache_key][:limit]
        
        results = defaultdict(float)
        
        # Multi-strategy search with weighted scoring
        
        # 1. Exact title match (highest priority)
        if query_lower in self.index['titles']:
            content_id = self.index['titles'][query_lower]
            results[content_id] += 1000
        
        # 2. Normalized title match
        normalized_query = self._normalize_title(query)
        if normalized_query in self.index['normalized_titles']:
            content_id = self.index['normalized_titles'][normalized_query]
            results[content_id] += 900
        
        # 3. Telugu content boost (check for Telugu variations)
        for telugu_key, content_ids in self.index['telugu_titles'].items():
            if telugu_key in query_lower or query_lower in telugu_key:
                for content_id in content_ids:
                    results[content_id] += 500  # High boost for Telugu content
        
        # 4. Token matching with position weighting
        query_tokens = self._tokenize(query_lower)
        for i, token in enumerate(query_tokens):
            if token in self.index['title_tokens']:
                position_weight = 1.0 / (i + 1)  # Earlier tokens more important
                for content_id in self.index['title_tokens'][token]:
                    results[content_id] += 200 * position_weight
        
        # 5. N-gram matching for fuzzy search
        query_ngrams = self._generate_ngrams(query_lower)
        for ngram in query_ngrams:
            if ngram in self.index['title_ngrams']:
                for content_id in self.index['title_ngrams'][ngram]:
                    results[content_id] += 50
        
        # 6. Phonetic matching for typos
        phonetic_query = self._get_phonetic_key(query)
        if phonetic_query and phonetic_query in self.index['phonetic']:
            for content_id in self.index['phonetic'][phonetic_query]:
                results[content_id] += 150
        
        # 7. Metaphone matching
        metaphone_query = self._get_metaphone_key(query)
        if metaphone_query and metaphone_query in self.index['metaphone']:
            for content_id in self.index['metaphone'][metaphone_query]:
                results[content_id] += 100
        
        # 8. Advanced fuzzy matching for remaining candidates
        if len(results) < limit:
            self._fuzzy_search(query_lower, results, limit * 2)
        
        # Apply filters
        if filters:
            results = self._apply_advanced_filters(results, filters)
        
        # Apply ranking algorithm
        ranked_results = self._rank_results(results, query_lower)
        
        # Cache hot searches
        if len(self.index['hot_cache']) > 1000:
            self.index['hot_cache'].popitem(last=False)
        self.index['hot_cache'][cache_key] = ranked_results
        
        # Update search stats
        self.index['search_stats'][query_lower] += 1
        
        return ranked_results[:limit]
    
    def _fuzzy_search(self, query: str, results: Dict[int, float], max_candidates: int):
        """Advanced fuzzy search using Levenshtein distance"""
        all_titles = list(self.index['titles'].keys())[:1000]  # Limit for performance
        
        # Use rapidfuzz for efficient fuzzy matching
        matches = process.extract(
            query,
            all_titles,
            scorer=fuzz.WRatio,  # Weighted ratio for better matching
            limit=max_candidates
        )
        
        for match, score, _ in matches:
            if score > 65:  # Threshold for relevance
                content_id = self.index['titles'].get(match)
                if content_id:
                    results[content_id] += score * 2
    
    def _apply_advanced_filters(self, results: Dict[int, float], filters: Dict) -> Dict[int, float]:
        """Apply advanced filters with scoring adjustments"""
        filtered = {}
        
        for content_id, score in results.items():
            content = self.index['content_map'].get(content_id)
            if not content:
                continue
            
            passes_filter = True
            
            # Genre filter
            if filters.get('genre'):
                if not any(g.lower() == filters['genre'].lower() for g in content['genres']):
                    passes_filter = False
            
            # Language filter with Telugu priority
            if filters.get('language'):
                lang_filter = filters['language'].lower()
                content_langs = [l.lower() for l in content['languages']]
                if lang_filter not in content_langs:
                    passes_filter = False
                elif lang_filter in ['telugu', 'te']:
                    score *= 1.5  # Boost Telugu content
            
            # Year filter
            if filters.get('year'):
                if not content['release_date'] or content['release_date'].year != filters['year']:
                    passes_filter = False
            
            # Content type filter
            if filters.get('content_type'):
                if content['content_type'] != filters['content_type']:
                    passes_filter = False
            
            # Rating filter
            if filters.get('min_rating'):
                if content['rating'] < filters['min_rating']:
                    passes_filter = False
            
            # New release filter
            if filters.get('new_releases'):
                if not content.get('is_new_release'):
                    passes_filter = False
            
            # Trending filter
            if filters.get('trending'):
                if not content.get('is_trending'):
                    passes_filter = False
                else:
                    score *= 1.2  # Boost trending content
            
            if passes_filter:
                filtered[content_id] = score
        
        return filtered
    
    def _rank_results(self, results: Dict[int, float], query: str) -> List[Tuple[int, float]]:
        """Advanced ranking algorithm with multiple factors"""
        ranked = []
        
        for content_id, base_score in results.items():
            content = self.index['content_map'].get(content_id)
            if not content:
                continue
            
            final_score = base_score
            
            # Language boost (Telugu priority)
            for lang in content['languages']:
                lang_lower = lang.lower()
                if lang_lower in LANGUAGE_PRIORITY:
                    final_score += LANGUAGE_PRIORITY[lang_lower]
            
            # Popularity and rating boost
            if content['popularity']:
                final_score += min(content['popularity'] / 10, 50)
            if content['rating']:
                final_score += content['rating'] * 10
            
            # Vote count relevance
            if content['vote_count']:
                final_score += min(content['vote_count'] / 100, 30)
            
            # Recency boost for new releases
            if content.get('is_new_release'):
                final_score += 100
            
            # Critics choice boost
            if content.get('is_critics_choice'):
                final_score += 80
            
            # Trending boost
            if content.get('is_trending'):
                final_score += 150
            
            # Title length similarity (prefer exact matches)
            title_lower = content['title'].lower() if content['title'] else ''
            if query == title_lower:
                final_score *= 2
            elif len(query) > 3 and query in title_lower:
                final_score *= 1.5
            
            ranked.append((content_id, final_score))
        
        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def autocomplete(self, prefix: str, limit: int = 10) -> List[int]:
        """Instant autocomplete using trie"""
        prefix_lower = prefix.lower().strip()
        if len(prefix_lower) < 2:
            return []
        
        # Navigate trie
        node = self.index['autocomplete_trie']
        content_ids = set()
        
        for char in prefix_lower:
            if char in node:
                content_ids.update(node[char]['ids'])
                node = node[char]['children']
            else:
                break
        
        # Rank by popularity
        ranked_ids = []
        for content_id in content_ids:
            content = self.index['content_map'].get(content_id)
            if content:
                score = content['popularity'] + content['rating'] * 10
                ranked_ids.append((content_id, score))
        
        ranked_ids.sort(key=lambda x: x[1], reverse=True)
        
        return [cid for cid, _ in ranked_ids[:limit]]

class InstantHybridSearch:
    """Enhanced hybrid search with instant results"""
    
    def __init__(self, db, cache, services):
        self.db = db
        self.cache = cache
        self.services = services
        self.indexer = UltraFastSearchIndexer()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize search index on startup"""
        try:
            threading.Thread(target=self._build_initial_index, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to initialize search index: {e}")
    
    def _build_initial_index(self):
        """Build initial index in background"""
        try:
            time.sleep(2)  # Small delay to let app initialize
            with self.db.app.app_context():
                Content = self.db.Model._decl_class_registry.get('Content')
                if Content:
                    content_list = Content.query.limit(10000).all()
                    self.indexer.build_index(content_list)
                    logger.info("Initial search index built successfully")
        except Exception as e:
            logger.error(f"Failed to build initial index: {e}")
    
    def search(self, query: str, search_type: str = 'multi', page: int = 1,
               limit: int = 20, filters: Dict = None) -> Dict[str, Any]:
        """
        Ultra-fast search with instant results
        """
        start_time = time.time()
        
        # Validate and clean query
        query = query.strip()
        if not query:
            return {'results': [], 'total_results': 0, 'search_time': 0}
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, search_type, page, filters)
        
        # Try cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            cached_result['cached'] = True
            cached_result['search_time'] = time.time() - start_time
            return cached_result
        
        # Update index if needed
        if self.indexer.last_update is None or \
           (datetime.now() - self.indexer.last_update).seconds > self.indexer.update_interval:
            self._update_index_async()
        
        # Perform instant in-memory search
        instant_results = self._instant_search(query, search_type, filters, limit)
        
        # If we have good instant results, return immediately
        if len(instant_results) >= min(limit, 10):
            response = self._prepare_response(instant_results, query, page, start_time, True)
            self._cache_result(cache_key, response)
            return response
        
        # Otherwise, perform comprehensive search
        all_results = {
            'instant': instant_results,
            'database': [],
            'external': []
        }
        
        # Parallel search operations
        with self.executor as executor:
            futures = []
            
            # Database search
            futures.append(
                executor.submit(self._database_search, query, search_type, filters, limit)
            )
            
            # External API search (only if needed)
            if len(instant_results) < 5:
                futures.append(
                    executor.submit(self._external_search_parallel, query, search_type, page)
                )
            
            # Collect results
            for future in as_completed(futures, timeout=5):
                try:
                    result = future.result()
                    if isinstance(result, list):
                        if result and hasattr(result[0], 'id'):
                            all_results['database'] = result
                        else:
                            all_results['external'] = result
                except Exception as e:
                    logger.error(f"Search operation failed: {e}")
        
        # Merge and rank all results
        final_results = self._merge_and_rank_all(all_results, query, limit)
        
        response = self._prepare_response(final_results, query, page, start_time, False)
        self._cache_result(cache_key, response)
        
        return response
    
    def _instant_search(self, query: str, search_type: str, filters: Dict, limit: int) -> List[Dict]:
        """Perform instant in-memory search"""
        # Add content type to filters if specified
        if search_type != 'multi':
            if not filters:
                filters = {}
            filters['content_type'] = search_type
        
        # Get results from indexer
        search_results = self.indexer.search(query, limit * 2, filters)
        
        # Format results
        formatted_results = []
        for content_id, score in search_results[:limit]:
            content_data = self.indexer.index['content_map'].get(content_id)
            if content_data:
                formatted_results.append(self._format_indexed_content(content_data, score))
        
        return formatted_results
    
    def _format_indexed_content(self, content_data: Dict, relevance_score: float) -> Dict:
        """Format indexed content for response"""
        youtube_url = None
        if content_data.get('youtube_trailer_id'):
            youtube_url = f"https://www.youtube.com/watch?v={content_data['youtube_trailer_id']}"
        
        return {
            'id': content_data['id'],
            'title': content_data['title'],
            'original_title': content_data.get('original_title'),
            'content_type': content_data['content_type'],
            'genres': content_data.get('genres', []),
            'languages': content_data.get('languages', []),
            'rating': content_data.get('rating', 0),
            'popularity': content_data.get('popularity', 0),
            'vote_count': content_data.get('vote_count', 0),
            'release_date': content_data['release_date'].isoformat() if content_data.get('release_date') else None,
            'poster_path': self._get_poster_url(content_data.get('poster_path')),
            'backdrop_path': self._get_backdrop_url(content_data.get('backdrop_path')),
            'overview': self._truncate_overview(content_data.get('overview')),
            'youtube_trailer': youtube_url,
            'is_trending': content_data.get('is_trending', False),
            'is_new_release': content_data.get('is_new_release', False),
            'is_critics_choice': content_data.get('is_critics_choice', False),
            'relevance_score': round(relevance_score, 2)
        }
    
    def _database_search(self, query: str, search_type: str, filters: Dict, limit: int) -> List[Any]:
        """Enhanced database search with Telugu priority"""
        try:
            Content = self.db.Model._decl_class_registry.get('Content')
            if not Content:
                return []
            
            # Build search query with Telugu priority
            search_query = Content.query
            
            # Title search conditions
            title_conditions = [
                Content.title.ilike(f'{query}%'),  # Starts with
                Content.title.ilike(f'%{query}%'),  # Contains
                Content.original_title.ilike(f'%{query}%')
            ]
            
            # Add overview search for longer queries
            if len(query) > 4:
                title_conditions.append(Content.overview.ilike(f'%{query}%'))
            
            search_query = search_query.filter(or_(*title_conditions))
            
            # Apply type filter
            if search_type != 'multi':
                search_query = search_query.filter(Content.content_type == search_type)
            
            # Apply additional filters
            if filters:
                search_query = self._apply_database_filters(search_query, filters, Content)
            
            # Order with Telugu content priority
            search_query = search_query.order_by(
                # Telugu content first
                case(
                    [(Content.languages.contains('telugu'), 0),
                     (Content.languages.contains('te'), 0)],
                    else_=1
                ),
                # Then by trending, new releases, critics choice
                Content.is_trending.desc(),
                Content.is_new_release.desc(),
                Content.is_critics_choice.desc(),
                # Then by popularity and rating
                Content.popularity.desc(),
                Content.rating.desc()
            )
            
            return search_query.limit(limit).all()
            
        except Exception as e:
            logger.error(f"Database search error: {e}")
            return []
    
    def _apply_database_filters(self, query: Query, filters: Dict, Content: Any) -> Query:
        """Apply filters to database query"""
        if filters.get('genre'):
            query = query.filter(Content.genres.contains(filters['genre']))
        
        if filters.get('language'):
            lang = filters['language']
            query = query.filter(
                or_(
                    Content.languages.contains(lang),
                    Content.languages.contains(lang.lower()),
                    Content.languages.contains(lang.title())
                )
            )
        
        if filters.get('year'):
            year = filters['year']
            query = query.filter(
                func.extract('year', Content.release_date) == year
            )
        
        if filters.get('min_rating'):
            query = query.filter(Content.rating >= filters['min_rating'])
        
        if filters.get('new_releases'):
            query = query.filter(Content.is_new_release == True)
        
        if filters.get('trending'):
            query = query.filter(Content.is_trending == True)
        
        if filters.get('critics_choice'):
            query = query.filter(Content.is_critics_choice == True)
        
        return query
    
    def _external_search_parallel(self, query: str, search_type: str, page: int) -> List[Dict]:
        """Parallel external API search"""
        external_results = []
        futures = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # TMDB search
            if search_type in ['multi', 'movie', 'tv']:
                futures.append(
                    executor.submit(
                        self._search_tmdb,
                        query, search_type, page
                    )
                )
            
            # Jikan anime search
            if search_type in ['multi', 'anime']:
                futures.append(
                    executor.submit(
                        self._search_jikan,
                        query, page
                    )
                )
            
            # Collect results
            for future in as_completed(futures, timeout=3):
                try:
                    results = future.result()
                    if results:
                        external_results.extend(results)
                except Exception as e:
                    logger.error(f"External search error: {e}")
        
        return external_results
    
    def _search_tmdb(self, query: str, search_type: str, page: int) -> List[Dict]:
        """Search TMDB API"""
        try:
            results = []
            tmdb_results = self.services['TMDBService'].search_content(query, search_type, page=page)
            
            if tmdb_results and 'results' in tmdb_results:
                for item in tmdb_results['results'][:10]:
                    content_type = 'movie' if 'title' in item else 'tv'
                    content = self.services['ContentService'].save_content_from_tmdb(item, content_type)
                    if content:
                        results.append(self._format_content(content))
            
            return results
        except Exception as e:
            logger.error(f"TMDB search error: {e}")
            return []
    
    def _search_jikan(self, query: str, page: int) -> List[Dict]:
        """Search Jikan API for anime"""
        try:
            results = []
            anime_results = self.services['JikanService'].search_anime(query, page)
            
            if anime_results and 'data' in anime_results:
                for anime in anime_results['data'][:10]:
                    content = self.services['ContentService'].save_anime_content(anime)
                    if content:
                        results.append(self._format_content(content))
            
            return results
        except Exception as e:
            logger.error(f"Jikan search error: {e}")
            return []
    
    def _merge_and_rank_all(self, all_results: Dict, query: str, limit: int) -> List[Dict]:
        """Merge and rank all search results"""
        merged = []
        seen_ids = set()
        
        # Priority: instant > database > external
        for source in ['instant', 'database', 'external']:
            results = all_results.get(source, [])
            
            for result in results:
                # Handle different result types
                if hasattr(result, 'id'):  # Database object
                    result = self._format_content(result)
                
                if isinstance(result, dict) and 'id' in result:
                    if result['id'] not in seen_ids:
                        seen_ids.add(result['id'])
                        result['source'] = source
                        merged.append(result)
        
        # Re-rank merged results
        query_lower = query.lower()
        for result in merged:
            score = result.get('relevance_score', 0)
            
            # Title matching
            title_lower = (result.get('title') or '').lower()
            if query_lower == title_lower:
                score += 1000
            elif query_lower in title_lower:
                score += 500
            else:
                score += fuzz.ratio(query_lower, title_lower) * 2
            
            # Language priority (Telugu boost)
            languages = result.get('languages', [])
            for lang in languages:
                if 'telugu' in lang.lower() or lang.lower() == 'te':
                    score += 300
                elif 'english' in lang.lower() or lang.lower() == 'en':
                    score += 200
                elif 'hindi' in lang.lower() or lang.lower() == 'hi':
                    score += 150
            
            # Quality signals
            score += (result.get('rating', 0) * 20)
            score += min(result.get('popularity', 0), 100)
            score += (result.get('vote_count', 0) / 100) if result.get('vote_count') else 0
            
            # Special flags
            if result.get('is_trending'):
                score += 200
            if result.get('is_new_release'):
                score += 150
            if result.get('is_critics_choice'):
                score += 100
            
            # Source bonus
            if result.get('source') == 'instant':
                score += 100
            elif result.get('source') == 'database':
                score += 50
            
            result['final_score'] = score
        
        # Sort by final score
        merged.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Clean up temporary fields
        for result in merged:
            result.pop('source', None)
            result.pop('final_score', None)
            result.pop('relevance_score', None)
        
        return merged[:limit]
    
    def _format_content(self, content: Any) -> Dict:
        """Format content object for response"""
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
            'rating': content.rating or 0,
            'popularity': content.popularity or 0,
            'vote_count': content.vote_count or 0,
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'poster_path': self._get_poster_url(content.poster_path),
            'backdrop_path': self._get_backdrop_url(content.backdrop_path),
            'overview': self._truncate_overview(content.overview),
            'youtube_trailer': youtube_url,
            'is_trending': content.is_trending,
            'is_new_release': content.is_new_release,
            'is_critics_choice': content.is_critics_choice
        }
    
    def _prepare_response(self, results: List[Dict], query: str, page: int, 
                         start_time: float, instant: bool) -> Dict:
        """Prepare final search response"""
        search_time = round(time.time() - start_time, 3)
        
        # Generate smart suggestions
        suggestions = self._generate_smart_suggestions(query, results[:5])
        
        return {
            'results': results,
            'total_results': len(results),
            'page': page,
            'query': query,
            'search_time': search_time,
            'instant': instant,
            'suggestions': suggestions,
            'filters_available': {
                'genres': self._get_available_genres(results),
                'languages': self._get_available_languages(results),
                'years': self._get_available_years(results),
                'types': self._get_available_types(results)
            }
        }
    
    def _generate_smart_suggestions(self, query: str, top_results: List[Dict]) -> List[str]:
        """Generate intelligent search suggestions"""
        suggestions = []
        query_lower = query.lower()
        
        # Language-specific suggestions
        if not any(lang in query_lower for lang in ['telugu', 'hindi', 'tamil', 'malayalam']):
            suggestions.append(f"{query} telugu")
            suggestions.append(f"{query} hindi")
        
        # Content type suggestions
        if 'movie' not in query_lower and 'film' not in query_lower:
            suggestions.append(f"{query} movies")
        if 'series' not in query_lower and 'show' not in query_lower:
            suggestions.append(f"{query} series")
        if 'anime' not in query_lower:
            suggestions.append(f"{query} anime")
        
        # Genre-based suggestions from results
        if top_results:
            genres = set()
            for result in top_results[:3]:
                genres.update(result.get('genres', []))
            
            for genre in list(genres)[:2]:
                if genre.lower() not in query_lower:
                    suggestions.append(f"{query} {genre}")
        
        # Year suggestions
        current_year = datetime.now().year
        if str(current_year) not in query:
            suggestions.append(f"{query} {current_year}")
        
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
    
    def _get_available_genres(self, results: List[Dict]) -> List[str]:
        """Get available genres from results"""
        genres = set()
        for result in results:
            genres.update(result.get('genres', []))
        return sorted(list(genres))
    
    def _get_available_languages(self, results: List[Dict]) -> List[str]:
        """Get available languages from results"""
        languages = set()
        for result in results:
            languages.update(result.get('languages', []))
        return sorted(list(languages))
    
    def _get_available_years(self, results: List[Dict]) -> List[int]:
        """Get available years from results"""
        years = set()
        for result in results:
            if result.get('release_date'):
                try:
                    year = int(result['release_date'][:4])
                    years.add(year)
                except:
                    pass
        return sorted(list(years), reverse=True)
    
    def _get_available_types(self, results: List[Dict]) -> List[str]:
        """Get available content types from results"""
        types = set()
        for result in results:
            if result.get('content_type'):
                types.add(result['content_type'])
        return sorted(list(types))
    
    def _get_poster_url(self, poster_path: str) -> Optional[str]:
        """Get full poster URL"""
        if not poster_path:
            return None
        if poster_path.startswith('http'):
            return poster_path
        return f"https://image.tmdb.org/t/p/w300{poster_path}"
    
    def _get_backdrop_url(self, backdrop_path: str) -> Optional[str]:
        """Get full backdrop URL"""
        if not backdrop_path:
            return None
        if backdrop_path.startswith('http'):
            return backdrop_path
        return f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
    
    def _truncate_overview(self, overview: str) -> str:
        """Truncate overview for response"""
        if not overview:
            return ""
        if len(overview) > 200:
            return overview[:197] + "..."
        return overview
    
    def _generate_cache_key(self, query: str, search_type: str, page: int, filters: Dict) -> str:
        """Generate cache key for search"""
        filter_str = json.dumps(filters, sort_keys=True) if filters else ''
        key_data = f"search:v2:{query}:{search_type}:{page}:{filter_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached search result"""
        try:
            cached = self.cache.get(cache_key)
            if cached:
                if isinstance(cached, str):
                    return json.loads(cached)
                return cached
        except Exception as e:
            logger.debug(f"Cache retrieval error: {e}")
        return None
    
    def _cache_result(self, cache_key: str, result: Dict) -> None:
        """Cache search result"""
        try:
            # Shorter cache time for instant results
            timeout = 180 if result.get('instant') else 300
            self.cache.set(cache_key, json.dumps(result), timeout=timeout)
        except Exception as e:
            logger.debug(f"Cache storage error: {e}")
    
    def _update_index_async(self):
        """Update search index asynchronously"""
        def update():
            try:
                Content = self.db.Model._decl_class_registry.get('Content')
                if Content:
                    with self.db.app.app_context():
                        content_list = Content.query.limit(10000).all()
                        self.indexer.build_index(content_list)
            except Exception as e:
                logger.error(f"Index update error: {e}")
        
        threading.Thread(target=update, daemon=True).start()
    
    def autocomplete(self, prefix: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Ultra-fast autocomplete with Telugu priority"""
        if len(prefix) < 2:
            return []
        
        # Check cache
        cache_key = f"autocomplete:v2:{prefix.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            return json.loads(cached) if isinstance(cached, str) else cached
        
        # Get suggestions from indexer
        content_ids = self.indexer.autocomplete(prefix, limit * 2)
        
        suggestions = []
        for content_id in content_ids[:limit]:
            content_data = self.indexer.index['content_map'].get(content_id)
            if content_data:
                # Prioritize Telugu content
                is_telugu = any('telugu' in lang.lower() or lang.lower() == 'te' 
                               for lang in content_data.get('languages', []))
                
                suggestions.append({
                    'id': content_id,
                    'title': content_data['title'],
                    'type': content_data['content_type'],
                    'year': content_data['release_date'].year if content_data.get('release_date') else None,
                    'poster': self._get_poster_url(content_data.get('poster_path')),
                    'rating': content_data.get('rating', 0),
                    'is_telugu': is_telugu,
                    'priority': 1 if is_telugu else 0
                })
        
        # Sort with Telugu priority
        suggestions.sort(key=lambda x: (-x['priority'], -x['rating']))
        
        # Remove priority field before returning
        for s in suggestions:
            s.pop('priority', None)
        
        # Cache for 30 minutes
        self.cache.set(cache_key, json.dumps(suggestions), timeout=1800)
        
        return suggestions[:limit]

class IntelligentSuggestionEngine:
    """Advanced suggestion engine with personalization"""
    
    def __init__(self, cache):
        self.cache = cache
        self.trending_cache_key = "trending_searches:v2"
        self.analytics_cache_key = "search_analytics:v2"
    
    def get_trending_searches(self, limit: int = 10) -> List[str]:
        """Get trending searches with Telugu content priority"""
        cached = self.cache.get(self.trending_cache_key)
        
        if cached:
            trending = json.loads(cached) if isinstance(cached, str) else cached
            return trending[:limit]
        
        # Default trending with Telugu priority
        trending = [
            "Pushpa 2",
            "RRR",
            "Telugu movies 2024",
            "Bahubali",
            "Latest Telugu releases",
            "New movies today",
            "Action movies",
            "Romantic Telugu films",
            "Netflix Telugu",
            "Prime Video Telugu",
            "Anime series",
            "Marvel movies",
            "Thriller movies",
            "Comedy shows"
        ]
        
        # Cache for 2 hours
        self.cache.set(self.trending_cache_key, json.dumps(trending), timeout=7200)
        return trending[:limit]
    
    def record_search(self, query: str, user_id: Optional[int] = None) -> None:
        """Record search with analytics"""
        try:
            # Get current analytics
            analytics = self.cache.get(self.analytics_cache_key)
            if analytics:
                analytics = json.loads(analytics) if isinstance(analytics, str) else analytics
            else:
                analytics = {
                    'queries': {},
                    'user_queries': {},
                    'language_stats': {},
                    'last_update': datetime.now().isoformat()
                }
            
            # Update query count
            analytics['queries'][query] = analytics['queries'].get(query, 0) + 1
            
            # Detect language preference
            query_lower = query.lower()
            if any(term in query_lower for term in ['telugu', 'tollywood']):
                analytics['language_stats']['telugu'] = analytics['language_stats'].get('telugu', 0) + 1
            elif any(term in query_lower for term in ['hindi', 'bollywood']):
                analytics['language_stats']['hindi'] = analytics['language_stats'].get('hindi', 0) + 1
            elif any(term in query_lower for term in ['tamil', 'kollywood']):
                analytics['language_stats']['tamil'] = analytics['language_stats'].get('tamil', 0) + 1
            
            # Update user-specific data
            if user_id:
                if str(user_id) not in analytics['user_queries']:
                    analytics['user_queries'][str(user_id)] = []
                
                analytics['user_queries'][str(user_id)].append({
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only last 50 queries per user
                analytics['user_queries'][str(user_id)] = analytics['user_queries'][str(user_id)][-50:]
            
            analytics['last_update'] = datetime.now().isoformat()
            
            # Save analytics
            self.cache.set(self.analytics_cache_key, json.dumps(analytics), timeout=86400)
            
            # Update trending if query is popular
            if analytics['queries'][query] > 5:
                self._update_trending(query)
            
        except Exception as e:
            logger.error(f"Failed to record search: {e}")
    
    def _update_trending(self, query: str):
        """Update trending searches"""
        try:
            trending = self.get_trending_searches(20)
            if query not in trending:
                trending.insert(0, query)
                trending = trending[:15]  # Keep top 15
                self.cache.set(self.trending_cache_key, json.dumps(trending), timeout=7200)
        except Exception as e:
            logger.error(f"Failed to update trending: {e}")
    
    def get_personalized_suggestions(self, user_id: int, limit: int = 5) -> List[str]:
        """Get personalized suggestions based on user history"""
        try:
            # Get user search history
            analytics = self.cache.get(self.analytics_cache_key)
            if not analytics:
                return []
            
            analytics = json.loads(analytics) if isinstance(analytics, str) else analytics
            user_queries = analytics.get('user_queries', {}).get(str(user_id), [])
            
            if not user_queries:
                return []
            
            # Extract unique recent queries
            suggestions = []
            seen = set()
            
            # Process recent queries
            for item in reversed(user_queries[-20:]):
                query = item['query']
                if query not in seen:
                    seen.add(query)
                    suggestions.append(query)
            
            # Add related suggestions
            if suggestions:
                recent_query = suggestions[0]
                # Add variations
                if 'telugu' not in recent_query.lower():
                    suggestions.append(f"{recent_query} telugu")
                if '2024' not in recent_query:
                    suggestions.append(f"{recent_query} 2024")
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get personalized suggestions: {e}")
            return []

# Export factory functions
def create_search_engine(db, cache, services):
    """Create optimized search engine"""
    return InstantHybridSearch(db, cache, services)

def create_suggestion_engine(cache):
    """Create intelligent suggestion engine"""
    return IntelligentSuggestionEngine(cache)