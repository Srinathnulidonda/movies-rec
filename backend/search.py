# backend/search.py
import json
import re
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set
from difflib import SequenceMatcher
from functools import lru_cache
import hashlib
import unicodedata
from math import log10, sqrt
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

class TMDBFetcher:
    """Fetches content from TMDB when not found locally"""
    
    def __init__(self, api_key: str, http_session=None):
        self.api_key = api_key
        self.base_url = 'https://api.themoviedb.org/3'
        self.http_session = http_session or requests.Session()
        self._cache = {}
        self._lock = threading.Lock()
        
    def search_tmdb(self, query: str, content_type: str = 'multi', page: int = 1) -> List[Dict]:
        """Search TMDB for content"""
        cache_key = f"tmdb_{query}_{content_type}_{page}"
        
        # Check cache first
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).seconds < 3600:  # 1 hour cache
                return cache_entry['data']
        
        url = f"{self.base_url}/search/{content_type}"
        params = {
            'api_key': self.api_key,
            'query': query,
            'page': page
        }
        
        try:
            response = self.http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                # Cache the results
                with self._lock:
                    self._cache[cache_key] = {
                        'data': results,
                        'timestamp': datetime.now()
                    }
                
                return results
        except Exception as e:
            logger.error(f"TMDB search error: {e}")
        
        return []
    
    def get_content_details(self, tmdb_id: int, content_type: str) -> Optional[Dict]:
        """Get detailed content information from TMDB"""
        cache_key = f"tmdb_details_{tmdb_id}_{content_type}"
        
        # Check cache
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).seconds < 7200:  # 2 hour cache
                return cache_entry['data']
        
        url = f"{self.base_url}/{content_type}/{tmdb_id}"
        params = {
            'api_key': self.api_key,
            'append_to_response': 'credits,videos'
        }
        
        try:
            response = self.http_session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Cache the result
                with self._lock:
                    self._cache[cache_key] = {
                        'data': data,
                        'timestamp': datetime.now()
                    }
                
                return data
        except Exception as e:
            logger.error(f"TMDB details error: {e}")
        
        return None

class NGramAnalyzer:
    """N-gram analyzer for partial matching and auto-complete"""
    
    def __init__(self, min_gram: int = 2, max_gram: int = 5):
        self.min_gram = min_gram
        self.max_gram = max_gram
        self._cache = {}
    
    def analyze(self, text: str) -> Set[str]:
        """Generate n-grams from text"""
        if not text:
            return set()
        
        # Check cache
        cache_key = f"{text}_{self.min_gram}_{self.max_gram}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Normalize text
        text = self._normalize_text(text)
        ngrams = set()
        
        # Generate character n-grams
        for n in range(self.min_gram, min(len(text) + 1, self.max_gram + 1)):
            for i in range(len(text) - n + 1):
                ngrams.add(text[i:i+n])
        
        # Add word n-grams
        words = text.split()
        for n in range(1, min(len(words) + 1, 4)):
            for i in range(len(words) - n + 1):
                ngrams.add(' '.join(words[i:i+n]))
        
        # Cache result
        self._cache[cache_key] = ngrams
        return ngrams
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for n-gram generation"""
        # Convert to lowercase
        text = text.lower()
        # Remove accents
        text = ''.join(c for c in unicodedata.normalize('NFD', text) 
                      if unicodedata.category(c) != 'Mn')
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

class FuzzyMatcher:
    """Fuzzy search implementation for handling typos and misspellings"""
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self._cache = {}
    
    def fuzzy_score(self, query: str, target: str) -> float:
        """Calculate fuzzy match score between query and target"""
        # Check cache
        cache_key = f"{query}|{target}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        query = query.lower()
        target = target.lower()
        
        # Exact match
        if query == target:
            score = 1.0
        # Contains query
        elif query in target:
            score = 0.9 * (len(query) / len(target))
        else:
            # Calculate similarity using multiple methods
            seq_score = SequenceMatcher(None, query, target).ratio()
            
            # Levenshtein distance normalized
            lev_score = 1 - (self._levenshtein_distance(query, target) / max(len(query), len(target)))
            
            # Jaro-Winkler similarity
            jw_score = self._jaro_winkler_similarity(query, target)
            
            # Weighted average
            score = (seq_score * 0.4 + lev_score * 0.3 + jw_score * 0.3)
        
        # Cache result
        self._cache[cache_key] = score
        return score
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro-Winkler similarity"""
        jaro = self._jaro_similarity(s1, s2)
        
        # Find common prefix up to 4 characters
        prefix = 0
        for i in range(min(len(s1), len(s2), 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        
        return jaro + (prefix * 0.1 * (1 - jaro))
    
    def _jaro_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro similarity"""
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        match_distance = (max(len1, len2) // 2) - 1
        match_distance = max(0, match_distance)
        
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        
        matches = 0
        transpositions = 0
        
        # Find matches
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Find transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        return (matches / len1 + matches / len2 + 
                (matches - transpositions / 2) / matches) / 3.0

class SearchIndex:
    """Optimized in-memory search index for ultra-fast searching"""
    
    def __init__(self):
        self.inverted_index = defaultdict(set)  # term -> set of content IDs
        self.content_data = {}  # content ID -> content data
        self.ngram_index = defaultdict(set)  # n-gram -> set of content IDs
        self.field_index = defaultdict(lambda: defaultdict(set))  # field -> term -> set of content IDs
        self.exact_title_index = {}  # lowercase title -> content ID for exact matches
        self.prefix_index = defaultdict(set)  # prefix -> set of content IDs
        self.ngram_analyzer = NGramAnalyzer()
        self.last_update = None
        self._lock = threading.RLock()
    
    def build_index(self, contents: List[Any]) -> None:
        """Build search index from content list"""
        with self._lock:
            start_time = time.time()
            
            # Clear existing index
            self.inverted_index.clear()
            self.content_data.clear()
            self.ngram_index.clear()
            self.field_index.clear()
            self.exact_title_index.clear()
            self.prefix_index.clear()
            
            for content in contents:
                self._index_content(content)
            
            self.last_update = datetime.utcnow()
            
            build_time = time.time() - start_time
            logger.info(f"Search index built for {len(contents)} items in {build_time:.3f}s")
    
    def add_content(self, content: Any) -> None:
        """Add single content to index (for TMDB fetched content)"""
        with self._lock:
            self._index_content(content)
    
    def _index_content(self, content: Any) -> None:
        """Index a single content item"""
        content_id = content.id
        
        # Store content data with proper None handling
        self.content_data[content_id] = {
            'id': content.id,
            'title': content.title or '',
            'original_title': content.original_title or '',
            'content_type': content.content_type or 'unknown',
            'genres': json.loads(content.genres or '[]'),
            'languages': json.loads(content.languages or '[]'),
            'overview': content.overview or '',
            'rating': float(content.rating) if content.rating is not None else 0.0,
            'vote_count': int(content.vote_count) if content.vote_count is not None else 0,
            'popularity': float(content.popularity) if content.popularity is not None else 0.0,
            'release_date': content.release_date,
            'poster_path': content.poster_path or '',
            'backdrop_path': content.backdrop_path or '',
            'youtube_trailer_id': content.youtube_trailer_id or '',
            'is_trending': bool(content.is_trending) if content.is_trending is not None else False,
            'is_new_release': bool(content.is_new_release) if content.is_new_release is not None else False,
            'is_critics_choice': bool(content.is_critics_choice) if content.is_critics_choice is not None else False,
            'tmdb_id': content.tmdb_id,
            'mal_id': content.mal_id
        }
        
        # Build exact title index for 100% matches
        if content.title:
            normalized_title = content.title.lower().strip()
            self.exact_title_index[normalized_title] = content_id
            
            # Also index without special characters for better matching
            clean_title = re.sub(r'[^\w\s]', '', normalized_title)
            if clean_title != normalized_title:
                self.exact_title_index[clean_title] = content_id
            
            # Build prefix index for instant autocomplete
            for i in range(2, min(len(normalized_title) + 1, 20)):
                prefix = normalized_title[:i]
                self.prefix_index[prefix].add(content_id)
        
        # Index original title
        if content.original_title:
            normalized_orig = content.original_title.lower().strip()
            self.exact_title_index[normalized_orig] = content_id
            
            # Build prefix index for original title
            for i in range(2, min(len(normalized_orig) + 1, 20)):
                prefix = normalized_orig[:i]
                self.prefix_index[prefix].add(content_id)
        
        # Index all fields
        self._index_text(content.title, content_id, 'title')
        self._index_text(content.original_title, content_id, 'original_title')
        self._index_text(content.overview, content_id, 'overview')
        self._index_text(content.content_type, content_id, 'content_type')
        
        # Index genres
        try:
            genres = json.loads(content.genres or '[]')
            for genre in genres:
                if genre:
                    self._index_text(genre, content_id, 'genre')
        except:
            pass
        
        # Index languages
        try:
            languages = json.loads(content.languages or '[]')
            for language in languages:
                if language:
                    self._index_text(language, content_id, 'language')
        except:
            pass
    
    def _index_text(self, text: str, content_id: int, field: str) -> None:
        """Index text for a specific field"""
        if not text:
            return
        
        # Normalize text
        normalized = text.lower()
        
        # Index full terms
        terms = self._tokenize(normalized)
        for term in terms:
            self.inverted_index[term].add(content_id)
            self.field_index[field][term].add(content_id)
        
        # Index n-grams for fuzzy matching
        ngrams = self.ngram_analyzer.analyze(text)
        for ngram in ngrams:
            self.ngram_index[ngram].add(content_id)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into searchable terms"""
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split into words
        words = text.split()
        return [word for word in words if len(word) > 1]

class SearchEngine:
    """Enhanced search engine with TMDB fetching and instant accurate results"""
    
    def __init__(self, db_session, Content, tmdb_api_key=None, ContentService=None, http_session=None):
        self.db = db_session
        self.Content = Content
        self.ContentService = ContentService
        self.search_index = SearchIndex()
        self.fuzzy_matcher = FuzzyMatcher()
        self.ngram_analyzer = NGramAnalyzer()
        self.tmdb_fetcher = TMDBFetcher(tmdb_api_key, http_session) if tmdb_api_key else None
        self._index_lock = threading.Lock()
        self._last_index_update = None
        self._index_update_interval = 60  # Reduced to 1 minute for faster updates
        self._executor = ThreadPoolExecutor(max_workers=3)
        
        # Field boost weights for relevance scoring
        self.field_boosts = {
            'title': 10.0,
            'original_title': 8.0,
            'genre': 5.0,
            'content_type': 3.0,
            'language': 3.0,
            'overview': 1.0
        }
        
        # Initialize index
        self._ensure_index_updated()
    
    def _ensure_index_updated(self) -> None:
        """Ensure search index is up to date"""
        current_time = datetime.utcnow()
        
        if (not self._last_index_update or 
            (current_time - self._last_index_update).seconds > self._index_update_interval):
            
            if not self._index_lock.locked():
                with self._index_lock:
                    try:
                        # Get all content from database
                        contents = self.Content.query.all()
                        self.search_index.build_index(contents)
                        self._last_index_update = current_time
                    except Exception as e:
                        logger.error(f"Index update error: {e}")
    
    def _fetch_from_tmdb_if_needed(self, query: str, existing_results: List) -> List:
        """Fetch content from TMDB if local results are insufficient"""
        if not self.tmdb_fetcher or not self.ContentService:
            return []
        
        # Only fetch from TMDB if we have fewer than 5 results
        if len(existing_results) >= 5:
            return []
        
        try:
            # Search TMDB
            tmdb_results = self.tmdb_fetcher.search_tmdb(query, 'multi')
            
            new_contents = []
            for item in tmdb_results[:10]:  # Limit to 10 items to avoid too many DB writes
                # Determine content type
                if 'media_type' in item:
                    content_type = 'tv' if item['media_type'] == 'tv' else 'movie'
                elif 'first_air_date' in item:
                    content_type = 'tv'
                else:
                    content_type = 'movie'
                
                # Check if content already exists
                tmdb_id = item.get('id')
                existing = self.Content.query.filter_by(tmdb_id=tmdb_id).first()
                
                if not existing:
                    # Save new content to database
                    content = self.ContentService.save_content_from_tmdb(item, content_type)
                    if content:
                        new_contents.append(content)
                        # Add to search index immediately
                        self.search_index.add_content(content)
            
            return new_contents
            
        except Exception as e:
            logger.error(f"TMDB fetch error: {e}")
            return []
    
    def search(self, 
               query: str, 
               content_type: Optional[str] = None,
               genres: Optional[List[str]] = None,
               languages: Optional[List[str]] = None,
               min_rating: Optional[float] = None,
               page: int = 1,
               per_page: int = 20,
               sort_by: str = 'relevance') -> Dict[str, Any]:
        """
        Perform instant 100% accurate search with TMDB fallback
        """
        start_time = time.time()
        
        # Ensure index is updated
        self._ensure_index_updated()
        
        # Normalize and prepare query
        query = query.strip()
        if not query:
            return self._empty_result()
        
        query_lower = query.lower()
        
        # Step 1: Check for exact title matches (100% accurate)
        exact_matches = []
        exact_match_ids = set()
        
        # Check exact title index
        if query_lower in self.search_index.exact_title_index:
            exact_match_id = self.search_index.exact_title_index[query_lower]
            exact_matches.append((exact_match_id, 1000.0))
            exact_match_ids.add(exact_match_id)
        
        # Check without special characters
        clean_query = re.sub(r'[^\w\s]', '', query_lower)
        if clean_query != query_lower and clean_query in self.search_index.exact_title_index:
            exact_match_id = self.search_index.exact_title_index[clean_query]
            if exact_match_id not in exact_match_ids:
                exact_matches.append((exact_match_id, 999.0))
                exact_match_ids.add(exact_match_id)
        
        # Step 2: Check for phrase matches in titles
        for content_id, content_data in self.search_index.content_data.items():
            if content_id in exact_match_ids:
                continue
                
            title = content_data.get('title', '').lower()
            original_title = content_data.get('original_title', '').lower()
            
            # Check if query matches exactly as a phrase in title
            if query_lower in title:
                position_score = 1 - (title.index(query_lower) / len(title))
                exact_matches.append((content_id, 900 + position_score * 50))
                exact_match_ids.add(content_id)
            elif original_title and query_lower in original_title:
                position_score = 1 - (original_title.index(query_lower) / len(original_title))
                exact_matches.append((content_id, 850 + position_score * 50))
                exact_match_ids.add(content_id)
        
        # Step 3: Get other candidates for fuzzy/partial matching
        candidate_ids = self._get_candidates(query_lower)
        candidate_ids = candidate_ids - exact_match_ids
        
        # Score and rank non-exact candidates
        scored_results = self._score_candidates(query_lower, candidate_ids)
        
        # Combine exact matches with other results
        all_results = exact_matches + scored_results
        
        # Step 4: If results are insufficient, fetch from TMDB
        if len(all_results) < 5 and self.tmdb_fetcher:
            # Run TMDB fetch in background while processing current results
            future = self._executor.submit(
                self._fetch_from_tmdb_if_needed, 
                query, 
                all_results
            )
            
            # Continue with current results processing
            # We'll add TMDB results later if they arrive in time
        else:
            future = None
        
        # Apply filters
        filtered_results = self._apply_filters(
            all_results, 
            content_type, 
            genres, 
            languages, 
            min_rating
        )
        
        # Check if TMDB fetch completed and add new results
        if future:
            try:
                # Wait max 1 second for TMDB results
                new_contents = future.result(timeout=1.0)
                
                # Score and add new TMDB content
                for content in new_contents:
                    score = self._calculate_relevance_score(query_lower, content)
                    if score > 0:
                        filtered_results.append((content.id, score))
            except:
                pass  # Continue without TMDB results if timeout
        
        # Sort results
        sorted_results = self._sort_results_optimized(filtered_results, sort_by)
        
        # Paginate
        total_results = len(sorted_results)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = sorted_results[start_idx:end_idx]
        
        # Format results
        formatted_results = self._format_results(paginated_results)
        
        # Calculate search time
        search_time = time.time() - start_time
        
        return {
            'query': query,
            'results': formatted_results,
            'total_results': total_results,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_results + per_page - 1) // per_page if per_page > 0 else 0,
            'search_time': f"{search_time:.3f}s",
            'filters_applied': {
                'content_type': content_type,
                'genres': genres,
                'languages': languages,
                'min_rating': min_rating
            },
            'data_source': 'hybrid'  # Indicates both local and TMDB
        }
    
    def autocomplete(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get instant autocomplete suggestions using prefix index
        """
        if not query or len(query) < 2:
            return []
        
        # Ensure index is updated
        self._ensure_index_updated()
        
        query = query.lower().strip()
        suggestions = []
        seen_titles = set()
        
        # Use prefix index for instant results
        if query in self.search_index.prefix_index:
            candidate_ids = self.search_index.prefix_index[query]
            
            # Score candidates by popularity and relevance
            scored_candidates = []
            for content_id in candidate_ids:
                content_data = self.search_index.content_data.get(content_id)
                if not content_data:
                    continue
                
                title = content_data.get('title', '')
                if not title or title.lower() in seen_titles:
                    continue
                
                # Calculate score based on exact prefix match and popularity
                title_lower = title.lower()
                score = 1.0 if title_lower.startswith(query) else 0.5
                
                popularity = content_data.get('popularity', 0) or 0
                score += log10(popularity + 1) * 0.1
                
                scored_candidates.append({
                    'title': title,
                    'content_type': content_data.get('content_type', 'unknown'),
                    'poster_path': content_data.get('poster_path', ''),
                    'score': score
                })
                seen_titles.add(title.lower())
            
            # Sort by score
            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # Format suggestions
            for candidate in scored_candidates[:limit]:
                suggestions.append({
                    'title': candidate['title'],
                    'content_type': candidate['content_type'],
                    'poster_path': self._format_poster_path(candidate['poster_path'])
                })
        
        return suggestions
    
    def _calculate_relevance_score(self, query: str, content: Any) -> float:
        """Calculate relevance score for a content item"""
        score = 0.0
        
        # Title matching
        if content.title:
            title_score = self._calculate_field_score(
                query, 
                content.title.lower(), 
                self.field_boosts['title']
            )
            score += title_score
        
        # Original title matching
        if content.original_title:
            orig_title_score = self._calculate_field_score(
                query, 
                content.original_title.lower(), 
                self.field_boosts['original_title']
            )
            score += orig_title_score
        
        # Overview matching
        if content.overview:
            overview_score = self._calculate_field_score(
                query, 
                content.overview.lower(), 
                self.field_boosts['overview']
            )
            score += overview_score
        
        # Popularity boost
        if content.popularity and content.popularity > 0:
            score += log10(content.popularity + 1) * 0.1
        
        # Rating boost
        if content.rating and content.vote_count and content.rating > 0 and content.vote_count > 10:
            score += (content.rating / 10) * log10(content.vote_count + 1) * 0.05
        
        return score
    
    def _get_candidates(self, query: str) -> Set[int]:
        """Get candidate content IDs for the query"""
        candidates = set()
        
        # Tokenize query
        query_terms = self._tokenize_query(query)
        
        # Get exact matches from inverted index
        for term in query_terms:
            if term in self.search_index.inverted_index:
                candidates.update(self.search_index.inverted_index[term])
        
        # Get fuzzy matches from n-gram index
        query_ngrams = self.ngram_analyzer.analyze(query)
        ngram_candidates = set()
        for ngram in query_ngrams:
            if ngram in self.search_index.ngram_index:
                ngram_candidates.update(self.search_index.ngram_index[ngram])
        
        # Add top fuzzy matches
        if len(candidates) < 100:
            candidates.update(list(ngram_candidates)[:200])
        
        return candidates
    
    def _score_candidates(self, query: str, candidate_ids: Set[int]) -> List[Tuple[int, float]]:
        """Score and rank candidate results"""
        scored_results = []
        
        for content_id in candidate_ids:
            content_data = self.search_index.content_data.get(content_id)
            if not content_data:
                continue
            
            # Calculate relevance score
            score = 0.0
            
            # Title matching with boost
            title = content_data.get('title', '')
            if title:
                title_score = self._calculate_field_score(
                    query, 
                    title, 
                    self.field_boosts['title']
                )
                score += title_score
            
            # Original title matching
            original_title = content_data.get('original_title', '')
            if original_title:
                orig_title_score = self._calculate_field_score(
                    query, 
                    original_title, 
                    self.field_boosts['original_title']
                )
                score += orig_title_score
            
            # Genre matching
            genres = content_data.get('genres', [])
            for genre in genres:
                if genre:
                    genre_score = self._calculate_field_score(
                        query, 
                        genre, 
                        self.field_boosts['genre']
                    )
                    score += genre_score
            
            # Overview matching
            overview = content_data.get('overview', '')
            if overview:
                overview_score = self._calculate_field_score(
                    query, 
                    overview, 
                    self.field_boosts['overview']
                )
                score += overview_score
            
            # Popularity boost
            popularity = content_data.get('popularity', 0) or 0
            if popularity > 0:
                score += log10(popularity + 1) * 0.1
            
            # Rating boost
            rating = content_data.get('rating', 0) or 0
            vote_count = content_data.get('vote_count', 0) or 0
            if rating > 0 and vote_count > 10:
                score += (rating / 10) * log10(vote_count + 1) * 0.05
            
            # Recency boost
            if content_data.get('is_new_release'):
                score += 0.5
            
            # Critics choice boost
            if content_data.get('is_critics_choice'):
                score += 0.3
            
            # Trending boost
            if content_data.get('is_trending'):
                score += 0.4
            
            if score > 0:
                scored_results.append((content_id, score))
        
        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results
    
    def _calculate_field_score(self, query: str, field_value: str, boost: float) -> float:
        """Calculate score for a specific field"""
        if not field_value:
            return 0.0
        
        field_lower = field_value.lower()
        query_lower = query.lower()
        
        # Exact match
        if query_lower == field_lower:
            return boost * 5.0
        
        # Prefix match
        if field_lower.startswith(query_lower):
            return boost * 4.0
        
        # Contains exact query
        if query_lower in field_lower:
            position_factor = 1 - (field_lower.index(query_lower) / len(field_lower))
            return boost * 3.0 * position_factor
        
        # Word match
        query_words = query_lower.split()
        field_words = field_lower.split()
        matched_words = sum(1 for word in query_words if word in field_words)
        if matched_words > 0:
            return boost * 2.0 * (matched_words / len(query_words))
        
        # Fuzzy match
        fuzzy_score = self.fuzzy_matcher.fuzzy_score(query_lower, field_lower)
        if fuzzy_score > self.fuzzy_matcher.threshold:
            return boost * 1.0 * fuzzy_score
        
        return 0.0
    
    def _apply_filters(self, 
                       scored_results: List[Tuple[int, float]], 
                       content_type: Optional[str],
                       genres: Optional[List[str]],
                       languages: Optional[List[str]],
                       min_rating: Optional[float]) -> List[Tuple[int, float]]:
        """Apply filters to search results"""
        if not any([content_type, genres, languages, min_rating]):
            return scored_results
        
        filtered = []
        for content_id, score in scored_results:
            content_data = self.search_index.content_data.get(content_id)
            if not content_data:
                continue
            
            # Content type filter
            if content_type and content_data.get('content_type') != content_type:
                continue
            
            # Genre filter
            if genres:
                content_genres = [g.lower() for g in content_data.get('genres', [])]
                if not any(genre.lower() in content_genres for genre in genres):
                    continue
            
            # Language filter
            if languages:
                content_languages = [l.lower() for l in content_data.get('languages', [])]
                if not any(lang.lower() in content_languages for lang in languages):
                    continue
            
            # Rating filter
            if min_rating:
                rating = content_data.get('rating', 0) or 0
                if rating < min_rating:
                    continue
            
            filtered.append((content_id, score))
        
        return filtered
    
    def _sort_results_optimized(self, 
                                scored_results: List[Tuple[int, float]], 
                                sort_by: str) -> List[Tuple[int, float]]:
        """Optimized sorting with caching"""
        if sort_by == 'relevance':
            return scored_results  # Already sorted by relevance
        
        # Pre-extract sort keys for optimization
        if sort_by == 'rating':
            keyed_results = [
                (item, self.search_index.content_data.get(item[0], {}).get('rating', 0) or 0)
                for item in scored_results
            ]
            keyed_results.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in keyed_results]
        
        elif sort_by == 'popularity':
            keyed_results = [
                (item, self.search_index.content_data.get(item[0], {}).get('popularity', 0) or 0)
                for item in scored_results
            ]
            keyed_results.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in keyed_results]
        
        elif sort_by == 'date':
            keyed_results = [
                (item, self.search_index.content_data.get(item[0], {}).get('release_date') or datetime.min.date())
                for item in scored_results
            ]
            keyed_results.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in keyed_results]
        
        return scored_results
    
    def _format_results(self, scored_results: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        """Format search results for response"""
        formatted = []
        
        for content_id, score in scored_results:
            content_data = self.search_index.content_data.get(content_id)
            if not content_data:
                continue
            
            # Format poster path
            poster_path = self._format_poster_path(content_data.get('poster_path', ''))
            
            # Format backdrop path
            backdrop_path = self._format_backdrop_path(content_data.get('backdrop_path', ''))
            
            # Format YouTube URL
            youtube_url = None
            youtube_trailer_id = content_data.get('youtube_trailer_id')
            if youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={youtube_trailer_id}"
            
            # Format release date
            release_date = None
            if content_data.get('release_date'):
                try:
                    release_date = content_data['release_date'].isoformat()
                except:
                    release_date = None
            
            # Add match type indicator
            match_type = 'fuzzy'
            if score >= 1000:
                match_type = 'exact'
            elif score >= 850:
                match_type = 'phrase'
            elif score >= 100:
                match_type = 'partial'
            
            formatted.append({
                'id': content_data.get('id'),
                'tmdb_id': content_data.get('tmdb_id'),
                'mal_id': content_data.get('mal_id'),
                'title': content_data.get('title', ''),
                'original_title': content_data.get('original_title', ''),
                'content_type': content_data.get('content_type', 'unknown'),
                'genres': content_data.get('genres', []),
                'languages': content_data.get('languages', []),
                'overview': content_data.get('overview', ''),
                'rating': content_data.get('rating', 0) or 0,
                'vote_count': content_data.get('vote_count', 0) or 0,
                'popularity': content_data.get('popularity', 0) or 0,
                'release_date': release_date,
                'poster_path': poster_path,
                'backdrop_path': backdrop_path,
                'youtube_trailer': youtube_url,
                'is_trending': content_data.get('is_trending', False),
                'is_new_release': content_data.get('is_new_release', False),
                'is_critics_choice': content_data.get('is_critics_choice', False),
                'relevance_score': round(score, 3),
                'match_type': match_type
            })
        
        return formatted
    
    def _format_poster_path(self, poster_path: str) -> Optional[str]:
        """Format poster path to full URL"""
        if not poster_path:
            return None
        if poster_path.startswith('http'):
            return poster_path
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    
    def _format_backdrop_path(self, backdrop_path: str) -> Optional[str]:
        """Format backdrop path to full URL"""
        if not backdrop_path:
            return None
        if backdrop_path.startswith('http'):
            return backdrop_path
        return f"https://image.tmdb.org/t/p/w1280{backdrop_path}"
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize search query"""
        # Remove special characters
        query = re.sub(r'[^\w\s]', ' ', query)
        # Split into words
        words = query.split()
        # Filter short words
        return [word for word in words if len(word) > 1]
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty search result"""
        return {
            'query': '',
            'results': [],
            'total_results': 0,
            'page': 1,
            'per_page': 20,
            'total_pages': 0,
            'search_time': '0.000s',
            'filters_applied': {},
            'data_source': 'local'
        }

# Singleton instance
_search_engine_instance = None

def get_search_engine(db_session, Content, tmdb_api_key=None, ContentService=None, http_session=None):
    """Get or create search engine instance with TMDB support"""
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = SearchEngine(
            db_session, 
            Content, 
            tmdb_api_key, 
            ContentService,
            http_session
        )
    return _search_engine_instance

# Public API functions for use in app.py
def search_content(db_session, Content, query, tmdb_api_key=None, ContentService=None, http_session=None, **kwargs):
    """
    Enhanced search function with TMDB fetching
    
    Usage:
        from search import search_content
        results = search_content(
            db.session, 
            Content, 
            "avatar",
            tmdb_api_key=TMDB_API_KEY,
            ContentService=ContentService,
            http_session=http_session,
            content_type="movie", 
            page=1
        )
    """
    try:
        engine = get_search_engine(db_session, Content, tmdb_api_key, ContentService, http_session)
        return engine.search(query, **kwargs)
    except Exception as e:
        logger.error(f"Search error: {e}")
        # Return empty result on error
        return {
            'query': query,
            'results': [],
            'total_results': 0,
            'page': kwargs.get('page', 1),
            'per_page': kwargs.get('per_page', 20),
            'total_pages': 0,
            'search_time': '0.000s',
            'filters_applied': kwargs,
            'error': str(e),
            'data_source': 'error'
        }

def get_autocomplete_suggestions(db_session, Content, query, limit=10):
    """
    Instant autocomplete function
    
    Usage:
        from search import get_autocomplete_suggestions
        suggestions = get_autocomplete_suggestions(db.session, Content, "ava", limit=5)
    """
    try:
        engine = get_search_engine(db_session, Content)
        return engine.autocomplete(query, limit)
    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        return []

def rebuild_search_index(db_session, Content):
    """
    Force rebuild of search index
    
    Usage:
        from search import rebuild_search_index
        rebuild_search_index(db.session, Content)
    """
    try:
        engine = get_search_engine(db_session, Content)
        contents = Content.query.all()
        engine.search_index.build_index(contents)
        engine._last_index_update = datetime.utcnow()
        return len(contents)
    except Exception as e:
        logger.error(f"Index rebuild error: {e}")
        return 0