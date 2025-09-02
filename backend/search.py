# backend/search.py
import json
import logging
import re
import time
import string
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import hashlib
from functools import lru_cache

import numpy as np
from flask import current_app
from sqlalchemy import or_, and_, func, text, desc
from sqlalchemy.orm import Query
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
import redis
import pickle

logger = logging.getLogger(__name__)

class NGramAnalyzer:
    """Advanced N-gram analyzer for partial matching and auto-complete"""
    
    def __init__(self, min_gram=2, max_gram=4):
        self.min_gram = min_gram
        self.max_gram = max_gram
        
    def generate_ngrams(self, text: str) -> Set[str]:
        """Generate n-grams from text"""
        if not text:
            return set()
        
        text = self._normalize_text(text)
        ngrams = set()
        
        # Character-level n-grams for typo tolerance
        for n in range(self.min_gram, min(len(text) + 1, self.max_gram + 1)):
            for i in range(len(text) - n + 1):
                ngrams.add(text[i:i+n])
        
        # Word-level n-grams
        words = text.split()
        for n in range(1, min(len(words) + 1, 4)):  # Up to 3-word phrases
            for i in range(len(words) - n + 1):
                ngrams.add(' '.join(words[i:i+n]))
        
        return ngrams
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for n-gram generation"""
        # Convert to lowercase and remove extra spaces
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove punctuation but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def calculate_similarity(self, query_ngrams: Set[str], content_ngrams: Set[str]) -> float:
        """Calculate similarity based on n-gram overlap"""
        if not query_ngrams or not content_ngrams:
            return 0.0
        
        intersection = len(query_ngrams & content_ngrams)
        union = len(query_ngrams | content_ngrams)
        
        return intersection / union if union > 0 else 0.0

class FuzzyMatcher:
    """Advanced fuzzy matching for typos and misspellings"""
    
    def __init__(self):
        self.algorithms = {
            'ratio': fuzz.ratio,
            'partial_ratio': fuzz.partial_ratio,
            'token_sort_ratio': fuzz.token_sort_ratio,
            'token_set_ratio': fuzz.token_set_ratio
        }
        
    def get_best_match(self, query: str, candidates: List[str], threshold: float = 70.0) -> List[Tuple[str, float]]:
        """Get best fuzzy matches with scores"""
        matches = []
        
        for candidate in candidates:
            scores = []
            for name, algorithm in self.algorithms.items():
                score = algorithm(query, candidate)
                scores.append(score)
            
            # Use weighted average of all algorithms
            final_score = (
                scores[0] * 0.4 +  # ratio
                scores[1] * 0.3 +  # partial_ratio
                scores[2] * 0.2 +  # token_sort_ratio
                scores[3] * 0.1    # token_set_ratio
            )
            
            if final_score >= threshold:
                matches.append((candidate, final_score))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance for edit distance"""
        return Levenshtein.distance(s1, s2)

class FieldBooster:
    """Field boosting system for prioritizing different content fields"""
    
    def __init__(self):
        self.field_weights = {
            'exact_title_match': 100.0,
            'title_prefix': 80.0,
            'title_contains': 60.0,
            'original_title': 50.0,
            'genre_match': 40.0,
            'keyword_match': 30.0,
            'overview_match': 20.0,
            'language_match': 15.0,
            'fuzzy_title': 25.0,
            'ngram_match': 35.0,
            'popularity_boost': 10.0,
            'rating_boost': 8.0,
            'recent_boost': 5.0
        }
    
    def calculate_score(self, content: Dict, query: str, match_types: List[str]) -> float:
        """Calculate weighted score for content based on match types"""
        score = 0.0
        
        for match_type in match_types:
            base_score = self.field_weights.get(match_type, 0.0)
            
            # Apply additional boosts
            if match_type in ['exact_title_match', 'title_prefix']:
                # Boost for exact matches
                score += base_score
            elif match_type == 'popularity_boost':
                # Boost based on popularity (normalized 0-1)
                popularity = content.get('popularity', 0)
                normalized_popularity = min(popularity / 1000, 1.0)
                score += base_score * normalized_popularity
            elif match_type == 'rating_boost':
                # Boost based on rating
                rating = content.get('rating', 0)
                if rating:
                    normalized_rating = rating / 10.0
                    score += base_score * normalized_rating
            elif match_type == 'recent_boost':
                # Boost for recent content
                release_date = content.get('release_date')
                if release_date:
                    try:
                        if isinstance(release_date, str):
                            release_date = datetime.fromisoformat(release_date.replace('Z', '+00:00'))
                        days_ago = (datetime.now() - release_date.replace(tzinfo=None)).days
                        if days_ago <= 365:  # Within last year
                            recency_factor = max(0, (365 - days_ago) / 365)
                            score += base_score * recency_factor
                    except:
                        pass
            else:
                score += base_score
        
        return score

class AdvancedSearchIndexer:
    """Advanced in-memory search index with n-grams and fuzzy matching"""
    
    def __init__(self):
        self.index = {
            'content_map': {},  # content_id -> content_data
            'title_exact': {},  # exact_title -> content_id
            'title_normalized': {},  # normalized_title -> content_id
            'title_ngrams': defaultdict(set),  # ngram -> set of content_ids
            'keywords': defaultdict(set),  # keyword -> set of content_ids
            'genres': defaultdict(set),  # genre -> set of content_ids
            'languages': defaultdict(set),  # language -> set of content_ids
            'years': defaultdict(set),  # year -> set of content_ids
            'autocomplete_trie': {},  # prefix -> suggestions
            'fuzzy_titles': [],  # list of titles for fuzzy matching
            'content_scores': {},  # content_id -> base_score
        }
        self.ngram_analyzer = NGramAnalyzer()
        self.fuzzy_matcher = FuzzyMatcher()
        self.field_booster = FieldBooster()
        self.last_update = None
        self.update_interval = 180  # 3 minutes for faster updates
        
    def needs_update(self) -> bool:
        """Check if index needs updating"""
        if not self.last_update:
            return True
        return (datetime.now() - self.last_update).seconds > self.update_interval
    
    def build_index(self, content_list: List[Any]) -> None:
        """Build comprehensive search index"""
        logger.info(f"Building advanced search index with {len(content_list)} items")
        start_time = time.time()
        
        # Clear existing index
        self.index = {
            'content_map': {},
            'title_exact': {},
            'title_normalized': {},
            'title_ngrams': defaultdict(set),
            'keywords': defaultdict(set),
            'genres': defaultdict(set),
            'languages': defaultdict(set),
            'years': defaultdict(set),
            'autocomplete_trie': {},
            'fuzzy_titles': [],
            'content_scores': {},
        }
        
        for content in content_list:
            self._index_content(content)
        
        # Build autocomplete trie
        self._build_autocomplete_trie()
        
        self.last_update = datetime.now()
        elapsed = time.time() - start_time
        logger.info(f"Advanced search index built in {elapsed:.2f} seconds")
    
    def _index_content(self, content: Any) -> None:
        """Index a single content item"""
        content_id = content.id
        
        # Store content data
        content_data = {
            'id': content_id,
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
            'youtube_trailer_id': content.youtube_trailer_id,
            'vote_count': getattr(content, 'vote_count', 0)
        }
        
        self.index['content_map'][content_id] = content_data
        
        # Calculate base score for this content
        base_score = self._calculate_base_score(content_data)
        self.index['content_scores'][content_id] = base_score
        
        # Index titles
        if content.title:
            title_lower = content.title.lower()
            self.index['title_exact'][title_lower] = content_id
            self.index['title_normalized'][self._normalize_title(content.title)] = content_id
            self.index['fuzzy_titles'].append(content.title)
            
            # Generate and index n-grams
            title_ngrams = self.ngram_analyzer.generate_ngrams(content.title)
            for ngram in title_ngrams:
                self.index['title_ngrams'][ngram].add(content_id)
        
        # Index original title
        if content.original_title and content.original_title != content.title:
            orig_title_lower = content.original_title.lower()
            self.index['title_exact'][orig_title_lower] = content_id
            
            # N-grams for original title
            orig_ngrams = self.ngram_analyzer.generate_ngrams(content.original_title)
            for ngram in orig_ngrams:
                self.index['title_ngrams'][ngram].add(content_id)
        
        # Index keywords
        keywords = self._extract_enhanced_keywords(content.title, content.overview)
        for keyword in keywords:
            self.index['keywords'][keyword].add(content_id)
        
        # Index genres
        genres = json.loads(content.genres or '[]')
        for genre in genres:
            self.index['genres'][genre.lower()].add(content_id)
            # Also index genre n-grams
            genre_ngrams = self.ngram_analyzer.generate_ngrams(genre)
            for ngram in genre_ngrams:
                self.index['keywords'][ngram].add(content_id)
        
        # Index languages
        languages = json.loads(content.languages or '[]')
        for language in languages:
            self.index['languages'][language.lower()].add(content_id)
        
        # Index year
        if content.release_date:
            year = content.release_date.year
            self.index['years'][year].add(content_id)
    
    def _calculate_base_score(self, content_data: Dict) -> float:
        """Calculate base score for content based on intrinsic quality"""
        score = 0.0
        
        # Rating contribution (0-10 scale)
        if content_data.get('rating'):
            score += content_data['rating'] * 2  # Max 20 points
        
        # Popularity contribution (normalized)
        if content_data.get('popularity'):
            score += min(content_data['popularity'] / 100, 10)  # Max 10 points
        
        # Vote count contribution (normalized)
        if content_data.get('vote_count'):
            score += min(content_data['vote_count'] / 1000, 5)  # Max 5 points
        
        # Recency bonus
        release_date = content_data.get('release_date')
        if release_date:
            try:
                if isinstance(release_date, str):
                    release_date = datetime.fromisoformat(release_date.replace('Z', '+00:00'))
                days_ago = (datetime.now() - release_date.replace(tzinfo=None)).days
                if days_ago <= 365:  # Within last year
                    score += (365 - days_ago) / 365 * 5  # Max 5 points
            except:
                pass
        
        return score
    
    def _normalize_title(self, title: str) -> str:
        """Advanced title normalization"""
        if not title:
            return ""
        
        # Convert to lowercase
        normalized = title.lower()
        
        # Remove articles and common words
        articles = ['the', 'a', 'an']
        words = normalized.split()
        filtered_words = [w for w in words if w not in articles]
        
        # Remove special characters and normalize spaces
        normalized = ' '.join(filtered_words)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _extract_enhanced_keywords(self, title: str, overview: str = None) -> Set[str]:
        """Extract enhanced keywords with better filtering"""
        keywords = set()
        
        # Process title
        if title:
            # Basic word extraction
            title_words = re.findall(r'\w+', title.lower())
            keywords.update(w for w in title_words if len(w) > 2)
            
            # Add title n-grams
            title_ngrams = self.ngram_analyzer.generate_ngrams(title)
            keywords.update(ngram for ngram in title_ngrams if len(ngram) > 2)
        
        # Process overview (limited and filtered)
        if overview:
            # Get first 100 words from overview
            overview_words = re.findall(r'\w+', overview.lower())[:100]
            
            # Filter out stop words and short words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 
                'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 
                'might', 'must', 'can', 'this', 'that', 'these', 'those', 'his', 'her',
                'their', 'our', 'my', 'your', 'him', 'she', 'they', 'we', 'you', 'it'
            }
            
            filtered_words = [
                w for w in overview_words 
                if w not in stop_words and len(w) > 3 and w.isalpha()
            ]
            
            # Add most relevant overview words
            word_freq = Counter(filtered_words)
            top_words = [word for word, count in word_freq.most_common(20)]
            keywords.update(top_words)
        
        return keywords
    
    def _build_autocomplete_trie(self) -> None:
        """Build trie structure for fast autocomplete"""
        for title in self.index['fuzzy_titles']:
            if not title:
                continue
                
            title_lower = title.lower()
            words = title_lower.split()
            
            # Index full title prefixes
            for i in range(1, min(len(title_lower) + 1, 30)):
                prefix = title_lower[:i]
                if prefix not in self.index['autocomplete_trie']:
                    self.index['autocomplete_trie'][prefix] = []
                if title not in self.index['autocomplete_trie'][prefix]:
                    self.index['autocomplete_trie'][prefix].append(title)
            
            # Index word prefixes
            for word in words:
                if len(word) > 2:
                    for i in range(2, min(len(word) + 1, 15)):
                        prefix = word[:i]
                        if prefix not in self.index['autocomplete_trie']:
                            self.index['autocomplete_trie'][prefix] = []
                        if title not in self.index['autocomplete_trie'][prefix]:
                            self.index['autocomplete_trie'][prefix].append(title)
    
    def advanced_search(self, query: str, limit: int = 20, filters: Dict = None) -> List[Tuple[int, float]]:
        """Perform advanced search with scoring"""
        if not query.strip():
            return []
        
        query_lower = query.lower()
        query_normalized = self._normalize_title(query)
        query_ngrams = self.ngram_analyzer.generate_ngrams(query)
        
        # Collect all potential matches with their scores
        scored_results = defaultdict(float)
        match_types = defaultdict(list)
        
        # 1. Exact title matches (highest priority)
        if query_lower in self.index['title_exact']:
            content_id = self.index['title_exact'][query_lower]
            scored_results[content_id] += 100
            match_types[content_id].append('exact_title_match')
        
        # 2. Normalized title matches
        if query_normalized in self.index['title_normalized']:
            content_id = self.index['title_normalized'][query_normalized]
            scored_results[content_id] += 90
            match_types[content_id].append('title_prefix')
        
        # 3. Title prefix matches
        for title, content_id in self.index['title_exact'].items():
            if title.startswith(query_lower):
                scored_results[content_id] += 80
                match_types[content_id].append('title_prefix')
            elif query_lower in title:
                scored_results[content_id] += 60
                match_types[content_id].append('title_contains')
        
        # 4. N-gram matching
        for ngram in query_ngrams:
            if ngram in self.index['title_ngrams']:
                for content_id in self.index['title_ngrams'][ngram]:
                    # Calculate n-gram similarity
                    content_data = self.index['content_map'][content_id]
                    content_title = content_data.get('title', '')
                    content_ngrams = self.ngram_analyzer.generate_ngrams(content_title)
                    similarity = self.ngram_analyzer.calculate_similarity(query_ngrams, content_ngrams)
                    
                    scored_results[content_id] += similarity * 50
                    match_types[content_id].append('ngram_match')
        
        # 5. Keyword matching
        query_keywords = self._extract_enhanced_keywords(query)
        for keyword in query_keywords:
            if keyword in self.index['keywords']:
                for content_id in self.index['keywords'][keyword]:
                    scored_results[content_id] += 30
                    match_types[content_id].append('keyword_match')
        
        # 6. Genre matching
        for genre, content_ids in self.index['genres'].items():
            if query_lower in genre or any(word in genre for word in query_lower.split()):
                for content_id in content_ids:
                    scored_results[content_id] += 40
                    match_types[content_id].append('genre_match')
        
        # 7. Fuzzy matching (for typos)
        if len(scored_results) < limit * 2:  # Only if we don't have enough results
            fuzzy_matches = self.fuzzy_matcher.get_best_match(
                query, self.index['fuzzy_titles'][:500], threshold=70
            )
            
            for matched_title, score in fuzzy_matches[:20]:
                # Find content ID for this title
                title_lower = matched_title.lower()
                if title_lower in self.index['title_exact']:
                    content_id = self.index['title_exact'][title_lower]
                    fuzzy_score = (score / 100) * 40  # Normalize fuzzy score
                    scored_results[content_id] += fuzzy_score
                    match_types[content_id].append('fuzzy_title')
        
        # Apply field boosting and base scores
        final_scores = {}
        for content_id, search_score in scored_results.items():
            content_data = self.index['content_map'][content_id]
            
            # Add base quality score
            base_score = self.index['content_scores'].get(content_id, 0)
            
            # Apply field boosting
            boosted_score = self.field_booster.calculate_score(
                content_data, query, match_types[content_id]
            )
            
            # Combine all scores
            final_score = search_score + boosted_score + (base_score * 0.1)
            final_scores[content_id] = final_score
        
        # Apply filters
        if filters:
            final_scores = self._apply_advanced_filters(final_scores, filters)
        
        # Sort by score and return top results
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]
    
    def _apply_advanced_filters(self, scored_results: Dict[int, float], filters: Dict) -> Dict[int, float]:
        """Apply advanced filters to search results"""
        filtered = {}
        
        for content_id, score in scored_results.items():
            content_data = self.index['content_map'].get(content_id)
            if not content_data:
                continue
            
            # Genre filter
            if filters.get('genre'):
                content_genres = [g.lower() for g in content_data.get('genres', [])]
                if filters['genre'].lower() not in content_genres:
                    continue
            
            # Language filter
            if filters.get('language'):
                content_languages = [l.lower() for l in content_data.get('languages', [])]
                if filters['language'].lower() not in content_languages:
                    continue
            
            # Year filter
            if filters.get('year') and content_data.get('release_date'):
                try:
                    content_year = content_data['release_date'].year
                    if content_year != filters['year']:
                        continue
                except:
                    continue
            
            # Content type filter
            if filters.get('content_type'):
                if content_data.get('content_type') != filters['content_type']:
                    continue
            
            # Rating filter
            if filters.get('min_rating'):
                content_rating = content_data.get('rating', 0)
                if not content_rating or content_rating < filters['min_rating']:
                    continue
            
            filtered[content_id] = score
        
        return filtered
    
    def get_autocomplete_suggestions(self, prefix: str, limit: int = 10) -> List[Dict]:
        """Get ranked autocomplete suggestions"""
        if len(prefix) < 2:
            return []
        
        prefix_lower = prefix.lower()
        suggestions = []
        
        # Get suggestions from trie
        if prefix_lower in self.index['autocomplete_trie']:
            titles = self.index['autocomplete_trie'][prefix_lower]
            
            # Score and rank suggestions
            scored_suggestions = []
            for title in titles[:50]:  # Limit for performance
                title_lower = title.lower()
                if title_lower in self.index['title_exact']:
                    content_id = self.index['title_exact'][title_lower]
                    content_data = self.index['content_map'].get(content_id)
                    
                    if content_data:
                        # Calculate suggestion score
                        score = 0
                        
                        # Prefix match score
                        if title_lower.startswith(prefix_lower):
                            score += 50
                        else:
                            score += 20
                        
                        # Add base quality score
                        score += self.index['content_scores'].get(content_id, 0) * 0.1
                        
                        # Popularity boost
                        popularity = content_data.get('popularity', 0)
                        score += min(popularity / 100, 10)
                        
                        scored_suggestions.append((title, content_id, score, content_data))
            
            # Sort by score and format
            scored_suggestions.sort(key=lambda x: x[2], reverse=True)
            
            for title, content_id, score, content_data in scored_suggestions[:limit]:
                suggestions.append({
                    'id': content_id,
                    'title': title,
                    'type': content_data.get('content_type'),
                    'year': content_data.get('release_date').year if content_data.get('release_date') else None,
                    'poster': self._format_poster_url(content_data.get('poster_path')),
                    'rating': content_data.get('rating'),
                    'score': round(score, 2)
                })
        
        return suggestions
    
    def _format_poster_url(self, poster_path: str) -> Optional[str]:
        """Format poster URL"""
        if not poster_path:
            return None
        if poster_path.startswith('http'):
            return poster_path
        return f"https://image.tmdb.org/t/p/w92{poster_path}"

class AdvancedHybridSearch:
    """Advanced hybrid search system with real-time data and optimized performance"""
    
    def __init__(self, db, cache, services):
        self.db = db
        self.cache = cache
        self.services = services
        self.indexer = AdvancedSearchIndexer()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache keys
        self.INDEX_CACHE_KEY = "search_index_timestamp"
        self.TRENDING_CACHE_KEY = "trending_searches"
        
    def search(self, query: str, search_type: str = 'multi', page: int = 1, 
               limit: int = 20, filters: Dict = None) -> Dict[str, Any]:
        """
        Advanced hybrid search with instant results and real-time data
        """
        if not query or not query.strip():
            return self._empty_response(query, page)
        
        # Normalize inputs
        query = query.strip()
        filters = filters or {}
        
        # Check cache for instant results
        cache_key = self._generate_cache_key(query, search_type, page, filters)
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for search: '{query}'")
            cached_result['cached'] = True
            return cached_result
        
        start_time = time.time()
        
        # Update index if needed with real-time data
        self._ensure_fresh_index()
        
        # Perform multi-strategy search
        search_results = {
            'primary': [],
            'secondary': [],
            'external': [],
            'suggestions': []
        }
        
        # 1. Primary search - Advanced in-memory index
        try:
            primary_results = self.indexer.advanced_search(query, limit * 2, filters)
            if primary_results:
                content_objects = self._get_content_by_ids([r[0] for r in primary_results])
                content_dict = {c.id: c for c in content_objects}
                
                for content_id, score in primary_results:
                    if content_id in content_dict:
                        content = content_dict[content_id]
                        formatted = self._format_content_with_score(content, score)
                        search_results['primary'].append(formatted)
        except Exception as e:
            logger.error(f"Primary search error: {e}")
        
        # 2. Secondary search - Database fallback for comprehensive results
        if len(search_results['primary']) < limit:
            try:
                db_results = self._database_search(query, search_type, filters, limit)
                for content in db_results:
                    # Avoid duplicates
                    if not any(r['id'] == content.id for r in search_results['primary']):
                        formatted = self._format_content_with_score(content, 50.0)
                        search_results['secondary'].append(formatted)
            except Exception as e:
                logger.error(f"Database search error: {e}")
        
        # 3. External search - Only if insufficient results
        total_results = len(search_results['primary']) + len(search_results['secondary'])
        if total_results < 5:
            try:
                external_results = self._external_search(query, search_type, page)
                search_results['external'] = external_results[:10]
            except Exception as e:
                logger.error(f"External search error: {e}")
        
        # 4. Generate intelligent suggestions
        search_results['suggestions'] = self._generate_smart_suggestions(
            query, search_results['primary'][:3]
        )
        
        # Merge and finalize results
        final_results = self._merge_and_rank_results(search_results, query, limit, page)
        
        # Prepare response
        response = {
            'results': final_results,
            'total_results': len(final_results),
            'page': page,
            'per_page': limit,
            'query': query,
            'search_time': round(time.time() - start_time, 3),
            'suggestions': search_results['suggestions'],
            'has_more': len(final_results) >= limit,
            'cached': False,
            'algorithm': 'advanced_hybrid_v2'
        }
        
        # Cache the response
        self._cache_result(cache_key, response)
        
        # Record search analytics
        self._record_search_analytics(query, len(final_results))
        
        return response
    
    def autocomplete(self, prefix: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ultra-fast autocomplete with intelligent ranking
        """
        if len(prefix) < 2:
            return []
        
        # Check cache
        cache_key = f"autocomplete:{prefix.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            try:
                return json.loads(cached)[:limit]
            except:
                pass
        
        # Ensure fresh index
        self._ensure_fresh_index()
        
        # Get suggestions from advanced indexer
        suggestions = self.indexer.get_autocomplete_suggestions(prefix, limit)
        
        # Cache for 30 minutes
        try:
            self.cache.set(cache_key, json.dumps(suggestions), timeout=1800)
        except Exception as e:
            logger.error(f"Autocomplete cache error: {e}")
        
        return suggestions
    
    def _ensure_fresh_index(self) -> None:
        """Ensure search index is up-to-date with real-time data"""
        try:
            # Check if index needs update
            if not self.indexer.needs_update():
                return
            
            # Check cache for last update timestamp
            last_db_update = self.cache.get(self.INDEX_CACHE_KEY)
            
            # Get latest content from database
            from models import Content  # Dynamic import to avoid circular dependency
            
            # Query for all content, ordered by update time
            content_query = Content.query.order_by(Content.updated_at.desc())
            
            # If we have a last update timestamp, only get newer content
            if last_db_update:
                try:
                    last_update_time = datetime.fromisoformat(last_db_update)
                    content_query = content_query.filter(Content.updated_at > last_update_time)
                except:
                    pass
            
            # Limit to prevent memory issues
            content_list = content_query.limit(20000).all()
            
            if content_list or not last_db_update:
                # Get all content if it's first time or we have new content
                if not last_db_update:
                    all_content = Content.query.limit(20000).all()
                    self.indexer.build_index(all_content)
                else:
                    # Incremental update for new content
                    for content in content_list:
                        self.indexer._index_content(content)
                    self.indexer.last_update = datetime.now()
                
                # Update cache timestamp
                self.cache.set(self.INDEX_CACHE_KEY, datetime.now().isoformat(), timeout=3600)
                
                logger.info(f"Search index updated with {len(content_list)} items")
            
        except Exception as e:
            logger.error(f"Index update error: {e}")
    
    def _database_search(self, query: str, search_type: str, filters: Dict, limit: int) -> List[Any]:
        """
        Optimized database search with advanced PostgreSQL features
        """
        from models import Content  # Dynamic import
        
        # Build search conditions with different strategies
        search_conditions = []
        
        # 1. Full-text search on title and overview
        search_conditions.append(
            func.lower(Content.title).contains(query.lower())
        )
        
        if Content.original_title:
            search_conditions.append(
                func.lower(Content.original_title).contains(query.lower())
            )
        
        # 2. LIKE patterns for partial matches
        query_pattern = f"%{query.lower()}%"
        search_conditions.append(Content.title.ilike(query_pattern))
        
        # 3. Genre and keyword matching
        words = query.lower().split()
        for word in words:
            if len(word) > 2:
                search_conditions.append(Content.genres.ilike(f"%{word}%"))
                if len(word) > 3:
                    search_conditions.append(Content.overview.ilike(f"%{word}%"))
        
        # Build query with OR conditions
        base_query = Content.query.filter(or_(*search_conditions))
        
        # Apply type filter
        if search_type != 'multi':
            type_mapping = {
                'movie': 'movie',
                'tv': 'tv',
                'anime': 'anime'
            }
            if search_type in type_mapping:
                base_query = base_query.filter(Content.content_type == type_mapping[search_type])
        
        # Apply additional filters
        if filters:
            if filters.get('genre'):
                base_query = base_query.filter(Content.genres.contains(filters['genre']))
            
            if filters.get('language'):
                base_query = base_query.filter(Content.languages.contains(filters['language']))
            
            if filters.get('year'):
                year_start = datetime(filters['year'], 1, 1).date()
                year_end = datetime(filters['year'], 12, 31).date()
                base_query = base_query.filter(Content.release_date.between(year_start, year_end))
            
            if filters.get('min_rating'):
                base_query = base_query.filter(Content.rating >= filters['min_rating'])
            
            if filters.get('content_type'):
                base_query = base_query.filter(Content.content_type == filters['content_type'])
        
        # Advanced ordering for relevance
        # PostgreSQL similarity and ranking
        query_lower = query.lower()
        
        # Custom relevance scoring using CASE WHEN
        relevance_score = func.coalesce(
            # Exact title match gets highest score
            func.sum(
                func.case(
                    (func.lower(Content.title) == query_lower, 100),
                    (func.lower(Content.title).like(f"{query_lower}%"), 80),
                    (func.lower(Content.title).contains(query_lower), 60),
                    else_=0
                )
            ), 0
        ).label('relevance')
        
        # Execute query with ranking
        try:
            results = base_query.order_by(
                desc(Content.popularity * 0.1 + Content.rating * 2),
                desc(Content.vote_count),
                desc(Content.release_date)
            ).limit(limit).all()
            
            return results
            
        except Exception as e:
            logger.error(f"Database search error: {e}")
            # Fallback to simple search
            return Content.query.filter(
                Content.title.ilike(f"%{query}%")
            ).order_by(desc(Content.popularity)).limit(limit).all()
    
    def _external_search(self, query: str, search_type: str, page: int) -> List[Dict]:
        """
        Search external APIs with optimized concurrent requests
        """
        external_results = []
        
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                
                # Search TMDB for movies/TV
                if search_type in ['multi', 'movie', 'tv']:
                    futures.append(
                        executor.submit(
                            self._search_tmdb, query, search_type, page
                        )
                    )
                
                # Search anime
                if search_type in ['multi', 'anime']:
                    futures.append(
                        executor.submit(
                            self._search_anime, query, page
                        )
                    )
                
                # Process results with timeout
                for future in as_completed(futures, timeout=8):
                    try:
                        results = future.result()
                        if results:
                            external_results.extend(results)
                    except Exception as e:
                        logger.error(f"External search future error: {e}")
        
        except Exception as e:
            logger.error(f"External search error: {e}")
        
        return external_results
    
    def _search_tmdb(self, query: str, search_type: str, page: int) -> List[Dict]:
        """Search TMDB and save to database"""
        results = []
        try:
            tmdb_response = self.services['TMDBService'].search_content(
                query, search_type, page=page
            )
            
            if tmdb_response and 'results' in tmdb_response:
                for item in tmdb_response['results'][:10]:
                    # Determine content type
                    content_type = 'movie' if 'title' in item else 'tv'
                    
                    # Save to database
                    content = self.services['ContentService'].save_content_from_tmdb(
                        item, content_type
                    )
                    
                    if content:
                        results.append(self._format_content_with_score(content, 40.0))
        
        except Exception as e:
            logger.error(f"TMDB search error: {e}")
        
        return results
    
    def _search_anime(self, query: str, page: int) -> List[Dict]:
        """Search anime and save to database"""
        results = []
        try:
            anime_response = self.services['JikanService'].search_anime(query, page)
            
            if anime_response and 'data' in anime_response:
                for anime in anime_response['data'][:10]:
                    # Save to database
                    content = self.services['ContentService'].save_anime_content(anime)
                    
                    if content:
                        results.append(self._format_content_with_score(content, 35.0))
        
        except Exception as e:
            logger.error(f"Anime search error: {e}")
        
        return results
    
    def _merge_and_rank_results(self, search_results: Dict, query: str, 
                               limit: int, page: int) -> List[Dict]:
        """
        Intelligent merging and ranking of results from different sources
        """
        all_results = []
        seen_ids = set()
        
        # Priority order: primary (index) -> secondary (db) -> external
        for source in ['primary', 'secondary', 'external']:
            for result in search_results.get(source, []):
                if result.get('id') not in seen_ids:
                    seen_ids.add(result['id'])
                    result['source'] = source
                    all_results.append(result)
        
        # Re-rank based on combined relevance
        query_lower = query.lower()
        
        for result in all_results:
            # Base score from search
            base_score = result.get('search_score', 0)
            
            # Title relevance bonus
            title = result.get('title', '').lower()
            if query_lower == title:
                base_score += 50
            elif title.startswith(query_lower):
                base_score += 30
            elif query_lower in title:
                base_score += 20
            
            # Quality indicators
            rating = result.get('rating', 0)
            popularity = result.get('popularity', 0)
            
            if rating:
                base_score += (rating / 10) * 15
            if popularity:
                base_score += min(popularity / 1000, 10)
            
            # Source priority
            source_bonus = {
                'primary': 20,
                'secondary': 10,
                'external': 5
            }
            base_score += source_bonus.get(result['source'], 0)
            
            # Recency bonus
            release_date = result.get('release_date')
            if release_date:
                try:
                    if isinstance(release_date, str):
                        release_date = datetime.fromisoformat(release_date.replace('Z', '+00:00'))
                    days_ago = (datetime.now() - release_date.replace(tzinfo=None)).days
                    if days_ago <= 365:
                        base_score += (365 - days_ago) / 365 * 10
                except:
                    pass
            
            result['final_score'] = base_score
        
        # Sort by final score
        all_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Handle pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        # Clean up result objects
        for result in all_results:
            result.pop('search_score', None)
            result.pop('source', None)
            result.pop('final_score', None)
        
        return all_results[start_idx:end_idx]
    
    def _generate_smart_suggestions(self, query: str, top_results: List[Dict]) -> List[str]:
        """
        Generate intelligent search suggestions
        """
        suggestions = []
        query_lower = query.lower()
        
        # 1. Spelling corrections using fuzzy matching
        if self.indexer.index['fuzzy_titles']:
            corrections = self.indexer.fuzzy_matcher.get_best_match(
                query, self.indexer.index['fuzzy_titles'][:1000], threshold=60
            )
            
            for correction, score in corrections[:2]:
                if correction.lower() != query_lower and score < 95:
                    suggestions.append(correction)
        
        # 2. Related searches based on top results
        if top_results:
            genres = set()
            for result in top_results[:3]:
                genres.update(result.get('genres', []))
            
            # Suggest genre combinations
            for genre in list(genres)[:2]:
                if genre.lower() not in query_lower:
                    suggestions.append(f"{query} {genre}")
        
        # 3. Content type suggestions
        type_suggestions = []
        if 'movie' not in query_lower and 'film' not in query_lower:
            type_suggestions.append(f"{query} movies")
        if 'series' not in query_lower and 'tv' not in query_lower:
            type_suggestions.append(f"{query} series")
        if 'anime' not in query_lower:
            type_suggestions.append(f"{query} anime")
        
        suggestions.extend(type_suggestions[:2])
        
        # 4. Year-based suggestions
        current_year = datetime.now().year
        if str(current_year) not in query:
            suggestions.append(f"{query} {current_year}")
        
        # Remove duplicates and limit
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            if suggestion_lower not in seen and suggestion_lower != query_lower:
                seen.add(suggestion_lower)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:6]
    
    def _format_content_with_score(self, content: Any, score: float = 0) -> Dict:
        """Format content object with search score"""
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
            'vote_count': getattr(content, 'vote_count', 0),
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'poster_path': self._format_poster_url(content.poster_path),
            'overview': self._truncate_overview(content.overview),
            'youtube_trailer': youtube_url,
            'search_score': round(score, 2)
        }
    
    def _format_poster_url(self, poster_path: str) -> Optional[str]:
        """Format poster URL"""
        if not poster_path:
            return None
        if poster_path.startswith('http'):
            return poster_path
        return f"https://image.tmdb.org/t/p/w300{poster_path}"
    
    def _truncate_overview(self, overview: str, max_length: int = 200) -> str:
        """Truncate overview text"""
        if not overview:
            return ""
        if len(overview) <= max_length:
            return overview
        return overview[:max_length].rsplit(' ', 1)[0] + '...'
    
    def _get_content_by_ids(self, content_ids: List[int]) -> List[Any]:
        """Get content objects by IDs efficiently"""
        if not content_ids:
            return []
        
        try:
            from models import Content  # Dynamic import
            return Content.query.filter(Content.id.in_(content_ids)).all()
        except Exception as e:
            logger.error(f"Error fetching content by IDs: {e}")
            return []
    
    def _empty_response(self, query: str, page: int) -> Dict[str, Any]:
        """Return empty search response"""
        return {
            'results': [],
            'total_results': 0,
            'page': page,
            'per_page': 0,
            'query': query,
            'search_time': 0.001,
            'suggestions': [],
            'has_more': False,
            'cached': False,
            'algorithm': 'empty'
        }
    
    def _generate_cache_key(self, query: str, search_type: str, page: int, filters: Dict) -> str:
        """Generate optimized cache key"""
        filter_str = json.dumps(filters, sort_keys=True) if filters else ''
        key_data = f"v2:{query}:{search_type}:{page}:{filter_str}"
        return f"search:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached search result with error handling"""
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
        """Cache search result with error handling"""
        try:
            # Cache for 5 minutes for dynamic results
            self.cache.set(cache_key, json.dumps(result), timeout=300)
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def _record_search_analytics(self, query: str, result_count: int) -> None:
        """Record search analytics for optimization"""
        try:
            analytics_key = "search_analytics_v2"
            analytics = self.cache.get(analytics_key)
            
            if analytics:
                analytics = json.loads(analytics)
            else:
                analytics = {
                    'queries': {},
                    'no_results': [],
                    'popular_terms': {},
                    'last_update': datetime.now().isoformat()
                }
            
            # Record query
            analytics['queries'][query] = analytics['queries'].get(query, 0) + 1
            
            # Record no results
            if result_count == 0:
                analytics['no_results'].append({
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                })
                # Keep only last 100 no-result queries
                analytics['no_results'] = analytics['no_results'][-100:]
            
            # Extract popular terms
            words = query.lower().split()
            for word in words:
                if len(word) > 2:
                    analytics['popular_terms'][word] = analytics['popular_terms'].get(word, 0) + 1
            
            analytics['last_update'] = datetime.now().isoformat()
            
            # Cache analytics
            self.cache.set(analytics_key, json.dumps(analytics), timeout=86400)
            
        except Exception as e:
            logger.error(f"Analytics recording error: {e}")

class AdvancedSuggestionEngine:
    """Advanced suggestion engine with machine learning-like features"""
    
    def __init__(self, cache):
        self.cache = cache
        
    def get_trending_searches(self, limit: int = 10) -> List[str]:
        """Get trending search queries with real-time data"""
        cache_key = "trending_searches_v2"
        cached = self.cache.get(cache_key)
        
        if cached:
            try:
                trending = json.loads(cached)
                return trending[:limit]
            except:
                pass
        
        # Get from analytics
        analytics_key = "search_analytics_v2"
        analytics = self.cache.get(analytics_key)
        
        trending = []
        if analytics:
            try:
                analytics = json.loads(analytics)
                queries = analytics.get('queries', {})
                
                # Sort by frequency and recency
                sorted_queries = sorted(
                    queries.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                trending = [query for query, count in sorted_queries[:limit]]
                
            except Exception as e:
                logger.error(f"Trending extraction error: {e}")
        
        # Fallback trending searches
        if not trending:
            trending = [
                "Telugu movies 2024",
                "Latest releases",
                "Action thriller",
                "Romantic comedy",
                "Popular anime",
                "Marvel movies",
                "Netflix series",
                "Horror films",
                "Bollywood hits",
                "Korean drama"
            ]
        
        # Cache for 1 hour
        try:
            self.cache.set(cache_key, json.dumps(trending), timeout=3600)
        except:
            pass
        
        return trending[:limit]
    
    def record_search(self, query: str, user_id: Optional[int] = None) -> None:
        """Record search with enhanced analytics"""
        try:
            # Record in main analytics
            analytics_key = "search_analytics_v2"
            analytics = self.cache.get(analytics_key)
            
            if analytics:
                analytics = json.loads(analytics)
            else:
                analytics = {
                    'queries': {},
                    'user_queries': {},
                    'hourly_stats': {},
                    'last_update': datetime.now().isoformat()
                }
            
            # Record query
            analytics['queries'][query] = analytics['queries'].get(query, 0) + 1
            
            # Record hourly stats
            current_hour = datetime.now().strftime('%Y-%m-%d-%H')
            if current_hour not in analytics['hourly_stats']:
                analytics['hourly_stats'][current_hour] = 0
            analytics['hourly_stats'][current_hour] += 1
            
            # Keep only last 7 days of hourly stats
            cutoff_time = datetime.now() - timedelta(days=7)
            analytics['hourly_stats'] = {
                k: v for k, v in analytics['hourly_stats'].items()
                if datetime.strptime(k, '%Y-%m-%d-%H') > cutoff_time
            }
            
            analytics['last_update'] = datetime.now().isoformat()
            
            # Cache updated analytics
            self.cache.set(analytics_key, json.dumps(analytics), timeout=86400)
            
            # Record user-specific history
            if user_id:
                user_key = f"user_search_history_v2:{user_id}"
                history = self.cache.get(user_key)
                
                if history:
                    history = json.loads(history)
                else:
                    history = []
                
                # Add to history with timestamp
                history.insert(0, {
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep last 50 searches
                history = history[:50]
                
                # Cache for 30 days
                self.cache.set(user_key, json.dumps(history), timeout=2592000)
                
        except Exception as e:
            logger.error(f"Search recording error: {e}")
    
    def get_personalized_suggestions(self, user_id: int, limit: int = 5) -> List[str]:
        """Get personalized suggestions based on user history"""
        try:
            user_key = f"user_search_history_v2:{user_id}"
            history = self.cache.get(user_key)
            
            if history:
                history = json.loads(history)
                
                # Extract unique recent queries
                recent_queries = []
                seen = set()
                
                for item in history[:20]:  # Last 20 searches
                    query = item['query']
                    if query not in seen and len(query) > 2:
                        seen.add(query)
                        recent_queries.append(query)
                
                return recent_queries[:limit]
            
        except Exception as e:
            logger.error(f"Personalized suggestions error: {e}")
        
        return []
    
    def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get query completion suggestions"""
        if len(partial_query) < 2:
            return []
        
        try:
            # Get popular queries that start with or contain the partial query
            analytics_key = "search_analytics_v2"
            analytics = self.cache.get(analytics_key)
            
            if analytics:
                analytics = json.loads(analytics)
                queries = analytics.get('queries', {})
                
                partial_lower = partial_query.lower()
                suggestions = []
                
                # Find matching queries
                for query, count in queries.items():
                    query_lower = query.lower()
                    if partial_lower in query_lower and query_lower != partial_lower:
                        suggestions.append((query, count))
                
                # Sort by popularity
                suggestions.sort(key=lambda x: x[1], reverse=True)
                
                return [query for query, count in suggestions[:limit]]
            
        except Exception as e:
            logger.error(f"Query suggestions error: {e}")
        
        return []

# Factory functions for integration with app.py
def create_search_engine(db, cache, services):
    """Factory function to create the advanced search engine"""
    return AdvancedHybridSearch(db, cache, services)

def create_suggestion_engine(cache):
    """Factory function to create the advanced suggestion engine"""
    return AdvancedSuggestionEngine(cache)

# Additional utility functions
def warm_up_search_index(search_engine):
    """Warm up the search index on application startup"""
    try:
        logger.info("Warming up search index...")
        search_engine._ensure_fresh_index()
        logger.info("Search index warmed up successfully")
    except Exception as e:
        logger.error(f"Search index warm-up failed: {e}")

def get_search_statistics(cache):
    """Get search statistics for monitoring"""
    try:
        analytics_key = "search_analytics_v2"
        analytics = cache.get(analytics_key)
        
        if analytics:
            analytics = json.loads(analytics)
            
            total_searches = sum(analytics.get('queries', {}).values())
            unique_queries = len(analytics.get('queries', {}))
            no_results_count = len(analytics.get('no_results', []))
            
            return {
                'total_searches': total_searches,
                'unique_queries': unique_queries,
                'no_results_count': no_results_count,
                'last_update': analytics.get('last_update'),
                'top_queries': sorted(
                    analytics.get('queries', {}).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
    except Exception as e:
        logger.error(f"Statistics retrieval error: {e}")
    
    return {
        'total_searches': 0,
        'unique_queries': 0,
        'no_results_count': 0,
        'last_update': None,
        'top_queries': []
    }