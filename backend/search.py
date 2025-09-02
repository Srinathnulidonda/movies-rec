# backend/search.py
import json
import re
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from difflib import SequenceMatcher
from functools import lru_cache, wraps
import hashlib
import unicodedata
from math import log10, sqrt, exp
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pickle
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Performance monitoring decorator
def measure_performance(func):
    """Decorator to measure function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug(f"{func.__name__} took {(end - start)*1000:.2f}ms")
        return result
    return wrapper

@dataclass
class SearchConfig:
    """Configuration for search engine"""
    # N-gram settings
    min_gram: int = 2
    max_gram: int = 5
    edge_ngram_min: int = 1
    edge_ngram_max: int = 20
    
    # Fuzzy search settings
    fuzzy_threshold: float = 0.6
    fuzzy_max_expansions: int = 50
    typo_tolerance: int = 2
    
    # Field boost settings
    field_boosts: Dict[str, float] = field(default_factory=lambda: {
        'title': 15.0,              # Highest priority for title matches
        'original_title': 12.0,      # High priority for original title
        'title_prefix': 20.0,        # Boost for prefix matches in title
        'title_exact': 50.0,         # Massive boost for exact title matches
        'genre': 8.0,                # Genre matches are important
        'genre_exact': 10.0,         # Exact genre match boost
        'keywords': 6.0,             # Keywords/tags
        'cast': 5.0,                 # Actor/director names
        'content_type': 4.0,         # Content type match
        'language': 3.0,             # Language match
        'overview': 2.0,             # Overview/description
        'year': 3.0,                 # Release year match
        'collection': 7.0            # Movie collection/series
    })
    
    # Scoring weights
    popularity_weight: float = 0.15
    rating_weight: float = 0.10
    recency_weight: float = 0.08
    trending_weight: float = 0.12
    
    # Cache settings
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 10000
    
    # Performance settings
    index_update_interval: int = 60  # seconds
    max_candidates: int = 1000
    min_score_threshold: float = 0.1

class AdvancedNGramAnalyzer:
    """Advanced N-gram analyzer with multiple strategies"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self._cache = {}
        self._phonetic_cache = {}
        self._lock = threading.RLock()
    
    @measure_performance
    def analyze(self, text: str, mode: str = 'standard') -> Set[str]:
        """
        Generate n-grams with multiple strategies
        
        Modes:
        - standard: Regular n-grams
        - edge: Edge n-grams for prefix matching
        - positional: N-grams with position information
        - phonetic: Phonetic n-grams for sound-alike matching
        """
        if not text:
            return set()
        
        cache_key = f"{text}_{mode}_{self.config.min_gram}_{self.config.max_gram}"
        
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        text_normalized = self._normalize_text(text)
        ngrams = set()
        
        if mode == 'standard':
            ngrams = self._generate_standard_ngrams(text_normalized)
        elif mode == 'edge':
            ngrams = self._generate_edge_ngrams(text_normalized)
        elif mode == 'positional':
            ngrams = self._generate_positional_ngrams(text_normalized)
        elif mode == 'phonetic':
            ngrams = self._generate_phonetic_ngrams(text_normalized)
        else:
            # Combine all strategies
            ngrams = self._generate_standard_ngrams(text_normalized)
            ngrams.update(self._generate_edge_ngrams(text_normalized))
            ngrams.update(self._generate_phonetic_ngrams(text_normalized))
        
        # Add word-level n-grams
        ngrams.update(self._generate_word_ngrams(text_normalized))
        
        with self._lock:
            if len(self._cache) < self.config.max_cache_size:
                self._cache[cache_key] = ngrams
        
        return ngrams
    
    def _normalize_text(self, text: str) -> str:
        """Advanced text normalization"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove accents and diacritics
        text = ''.join(c for c in unicodedata.normalize('NFD', text) 
                      if unicodedata.category(c) != 'Mn')
        
        # Normalize special characters
        replacements = {
            '&': ' and ',
            '@': ' at ',
            '#': ' number ',
            '%': ' percent ',
            '+': ' plus ',
            '=': ' equals '
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove other special characters but keep spaces and alphanumeric
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _generate_standard_ngrams(self, text: str) -> Set[str]:
        """Generate standard character n-grams"""
        ngrams = set()
        
        for n in range(self.config.min_gram, min(len(text) + 1, self.config.max_gram + 1)):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                ngrams.add(ngram)
                
                # Add skip-grams for better fuzzy matching
                if n > 2 and i + n + 1 <= len(text):
                    skip_gram = text[i] + text[i+n-1:i+n+1]
                    ngrams.add(f"skip_{skip_gram}")
        
        return ngrams
    
    def _generate_edge_ngrams(self, text: str) -> Set[str]:
        """Generate edge n-grams for prefix/suffix matching"""
        ngrams = set()
        
        # Prefix n-grams
        for n in range(self.config.edge_ngram_min, 
                      min(len(text) + 1, self.config.edge_ngram_max + 1)):
            ngrams.add(f"prefix_{text[:n]}")
        
        # Suffix n-grams
        for n in range(self.config.edge_ngram_min, 
                      min(len(text) + 1, self.config.edge_ngram_max + 1)):
            ngrams.add(f"suffix_{text[-n:]}")
        
        return ngrams
    
    def _generate_positional_ngrams(self, text: str) -> Set[str]:
        """Generate n-grams with position information"""
        ngrams = set()
        
        for n in range(self.config.min_gram, min(len(text) + 1, self.config.max_gram + 1)):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                # Add position information
                position = "start" if i == 0 else "end" if i == len(text) - n else "mid"
                ngrams.add(f"{position}_{ngram}")
                # Also add without position
                ngrams.add(ngram)
        
        return ngrams
    
    def _generate_phonetic_ngrams(self, text: str) -> Set[str]:
        """Generate phonetic n-grams for sound-alike matching"""
        ngrams = set()
        
        # Simple phonetic encoding (can be replaced with Metaphone or Soundex)
        phonetic = self._simple_phonetic_encoding(text)
        
        for n in range(2, min(len(phonetic) + 1, 6)):
            for i in range(len(phonetic) - n + 1):
                ngrams.add(f"phonetic_{phonetic[i:i+n]}")
        
        return ngrams
    
    def _generate_word_ngrams(self, text: str) -> Set[str]:
        """Generate word-level n-grams"""
        ngrams = set()
        words = text.split()
        
        # Single words
        for word in words:
            if len(word) > 1:
                ngrams.add(f"word_{word}")
        
        # Word bigrams and trigrams
        for n in range(2, min(len(words) + 1, 4)):
            for i in range(len(words) - n + 1):
                word_ngram = ' '.join(words[i:i+n])
                ngrams.add(f"words_{word_ngram}")
        
        return ngrams
    
    def _simple_phonetic_encoding(self, text: str) -> str:
        """Simple phonetic encoding for sound-alike matching"""
        # Remove vowels except first letter
        if not text:
            return ""
        
        result = text[0]
        for char in text[1:]:
            if char not in 'aeiou':
                result += char
        
        # Common phonetic replacements
        replacements = {
            'ph': 'f', 'ck': 'k', 'q': 'k', 'x': 'ks',
            'wr': 'r', 'kn': 'n', 'gn': 'n', 'mb': 'm',
            'ps': 's', 'wh': 'w'
        }
        
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        return result

class AdvancedFuzzyMatcher:
    """Advanced fuzzy matching with multiple algorithms"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self._cache = {}
        self._lock = threading.RLock()
    
    @measure_performance
    def fuzzy_score(self, query: str, target: str, method: str = 'hybrid') -> float:
        """
        Calculate fuzzy match score using multiple methods
        
        Methods:
        - levenshtein: Edit distance based
        - jaro_winkler: Position-based similarity
        - ngram: N-gram overlap
        - hybrid: Combination of all methods
        - semantic: Semantic similarity (simplified)
        """
        cache_key = f"{query}|{target}|{method}"
        
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        query_lower = query.lower()
        target_lower = target.lower()
        
        # Exact match
        if query_lower == target_lower:
            score = 1.0
        # Contains exact query
        elif query_lower in target_lower:
            # Position-based scoring
            position = target_lower.index(query_lower)
            position_penalty = position / len(target_lower)
            length_ratio = len(query_lower) / len(target_lower)
            score = 0.95 * (1 - position_penalty * 0.3) * (0.5 + length_ratio * 0.5)
        else:
            if method == 'levenshtein':
                score = self._levenshtein_similarity(query_lower, target_lower)
            elif method == 'jaro_winkler':
                score = self._jaro_winkler_similarity(query_lower, target_lower)
            elif method == 'ngram':
                score = self._ngram_similarity(query_lower, target_lower)
            elif method == 'semantic':
                score = self._semantic_similarity(query_lower, target_lower)
            else:  # hybrid
                scores = [
                    self._levenshtein_similarity(query_lower, target_lower) * 0.25,
                    self._jaro_winkler_similarity(query_lower, target_lower) * 0.25,
                    self._ngram_similarity(query_lower, target_lower) * 0.3,
                    self._semantic_similarity(query_lower, target_lower) * 0.2
                ]
                score = sum(scores)
        
        # Apply threshold
        if score < self.config.fuzzy_threshold:
            score = 0.0
        
        with self._lock:
            if len(self._cache) < self.config.max_cache_size:
                self._cache[cache_key] = score
        
        return score
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Optimized Levenshtein distance with early termination"""
        if abs(len(s1) - len(s2)) > self.config.typo_tolerance * 2:
            return 0.0
        
        # Use dynamic programming with optimization
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        distances = range(len(s1) + 1)
        
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min(
                        distances[i1],      # substitution
                        distances[i1 + 1],  # insertion
                        new_distances[-1]   # deletion
                    ))
            distances = new_distances
            
            # Early termination if distance is too high
            if min(distances) > self.config.typo_tolerance:
                return 0.0
        
        distance = distances[-1]
        max_len = max(len(s1), len(s2))
        
        return 1 - (distance / max_len) if max_len > 0 else 0.0
    
    def _jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """Enhanced Jaro-Winkler with prefix scaling"""
        if not s1 or not s2:
            return 0.0
        
        # Quick check for very different lengths
        len_ratio = min(len(s1), len(s2)) / max(len(s1), len(s2))
        if len_ratio < 0.5:
            return len_ratio * 0.5
        
        # Calculate Jaro similarity
        jaro = self._jaro_similarity(s1, s2)
        
        # Find common prefix (up to 4 chars)
        prefix_len = 0
        for i in range(min(len(s1), len(s2), 4)):
            if s1[i] == s2[i]:
                prefix_len += 1
            else:
                break
        
        # Apply Winkler modification
        # Use scaling factor of 0.1 for prefix, with max prefix of 4
        jw_score = jaro + (prefix_len * 0.1 * (1 - jaro))
        
        # Additional boost for exact prefix match
        if prefix_len >= 3:
            jw_score = min(1.0, jw_score * 1.1)
        
        return jw_score
    
    def _jaro_similarity(self, s1: str, s2: str) -> float:
        """Optimized Jaro similarity calculation"""
        len1, len2 = len(s1), len(s2)
        
        if len1 == 0 and len2 == 0:
            return 1.0
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Calculate match window
        match_window = max(len1, len2) // 2 - 1
        match_window = max(0, match_window)
        
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        
        matches = 0
        transpositions = 0
        
        # Find matches
        for i in range(len1):
            start = max(0, i - match_window)
            end = min(i + match_window + 1, len2)
            
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
        
        jaro = (matches / len1 + 
                matches / len2 + 
                (matches - transpositions / 2) / matches) / 3.0
        
        return jaro
    
    def _ngram_similarity(self, s1: str, s2: str) -> float:
        """N-gram based similarity with Jaccard coefficient"""
        if not s1 or not s2:
            return 0.0
        
        # Generate character bigrams and trigrams
        ngrams1 = set()
        ngrams2 = set()
        
        for n in [2, 3]:
            for i in range(len(s1) - n + 1):
                ngrams1.add(s1[i:i+n])
            for i in range(len(s2) - n + 1):
                ngrams2.add(s2[i:i+n])
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        jaccard = intersection / union if union > 0 else 0.0
        
        # Apply Dice coefficient for better partial matching
        dice = 2 * intersection / (len(ngrams1) + len(ngrams2))
        
        # Weighted average
        return jaccard * 0.6 + dice * 0.4
    
    def _semantic_similarity(self, s1: str, s2: str) -> float:
        """Simplified semantic similarity based on word overlap and synonyms"""
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Direct word overlap
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        if total == 0:
            return 0.0
        
        base_score = overlap / total
        
        # Check for common variations
        variations = {
            'movie': ['film', 'motion picture', 'flick'],
            'show': ['series', 'program', 'tv show'],
            'episode': ['ep', 'chapter', 'part'],
            '1': ['one', 'i', 'first'],
            '2': ['two', 'ii', 'second'],
            '3': ['three', 'iii', 'third']
        }
        
        # Check for synonym matches
        synonym_matches = 0
        for word1 in words1:
            for base, syns in variations.items():
                if word1 == base or word1 in syns:
                    for word2 in words2:
                        if word2 in syns or word2 == base:
                            synonym_matches += 0.5
                            break
        
        # Combine scores
        synonym_bonus = min(0.3, synonym_matches * 0.1)
        
        return min(1.0, base_score + synonym_bonus)
    
    def find_typo_corrections(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """Find possible typo corrections from candidates"""
        corrections = []
        
        for candidate in candidates[:self.config.fuzzy_max_expansions]:
            score = self.fuzzy_score(query, candidate, 'hybrid')
            
            if score >= self.config.fuzzy_threshold:
                corrections.append((candidate, score))
        
        # Sort by score
        corrections.sort(key=lambda x: x[1], reverse=True)
        
        return corrections[:10]  # Return top 10 corrections

class OptimizedSearchIndex:
    """Highly optimized search index with advanced indexing strategies"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.ngram_analyzer = AdvancedNGramAnalyzer(config)
        
        # Multiple index structures for different query types
        self.inverted_index = defaultdict(set)  # term -> content IDs
        self.exact_index = {}  # exact term -> content IDs
        self.prefix_trie = {}  # Trie structure for prefix search
        self.ngram_index = defaultdict(set)  # ngram -> content IDs
        self.field_index = defaultdict(lambda: defaultdict(set))  # field -> term -> IDs
        self.phonetic_index = defaultdict(set)  # phonetic -> content IDs
        
        # Content storage
        self.content_data = {}  # content ID -> content data
        self.content_vectors = {}  # content ID -> feature vector (for ML-based ranking)
        
        # Statistics for BM25 scoring
        self.doc_lengths = {}  # content ID -> document length
        self.avg_doc_length = 0
        self.total_docs = 0
        self.term_doc_freq = defaultdict(int)  # term -> document frequency
        
        # Cache structures
        self.score_cache = {}
        self.last_update = None
        self._lock = threading.RLock()
    
    @measure_performance
    def build_index(self, contents: List[Any]) -> None:
        """Build optimized search index with multiple strategies"""
        with self._lock:
            start_time = time.time()
            
            # Clear all indices
            self._clear_indices()
            
            # Build indices
            for content in contents:
                self._index_content(content)
            
            # Calculate statistics
            self._calculate_statistics()
            
            # Build trie for prefix search
            self._build_prefix_trie()
            
            self.last_update = datetime.utcnow()
            self.total_docs = len(contents)
            
            build_time = time.time() - start_time
            logger.info(f"Optimized index built for {len(contents)} items in {build_time:.3f}s")
    
    def _clear_indices(self):
        """Clear all index structures"""
        self.inverted_index.clear()
        self.exact_index.clear()
        self.prefix_trie.clear()
        self.ngram_index.clear()
        self.field_index.clear()
        self.phonetic_index.clear()
        self.content_data.clear()
        self.content_vectors.clear()
        self.doc_lengths.clear()
        self.term_doc_freq.clear()
        self.score_cache.clear()
    
    def _index_content(self, content: Any) -> None:
        """Index single content with all strategies"""
        content_id = content.id
        
        # Store content data
        content_data = self._extract_content_data(content)
        self.content_data[content_id] = content_data
        
        # Create feature vector for ML ranking
        self.content_vectors[content_id] = self._create_feature_vector(content_data)
        
        # Index title with multiple strategies
        if content_data['title']:
            self._index_field(content_data['title'], content_id, 'title')
            
            # Exact title indexing
            title_lower = content_data['title'].lower().strip()
            self.exact_index[title_lower] = content_id
            
            # Clean title (without special chars)
            clean_title = re.sub(r'[^\w\s]', '', title_lower)
            if clean_title != title_lower:
                self.exact_index[clean_title] = content_id
        
        # Index original title
        if content_data['original_title']:
            self._index_field(content_data['original_title'], content_id, 'original_title')
            self.exact_index[content_data['original_title'].lower().strip()] = content_id
        
        # Index other fields
        self._index_field(content_data['overview'], content_id, 'overview')
        self._index_field(content_data['content_type'], content_id, 'content_type')
        
        # Index genres with exact matching
        for genre in content_data['genres']:
            self._index_field(genre, content_id, 'genre')
            self.field_index['genre_exact'][genre.lower()].add(content_id)
        
        # Index languages
        for language in content_data['languages']:
            self._index_field(language, content_id, 'language')
        
        # Index year
        if content_data['release_date']:
            year = str(content_data['release_date'].year) if hasattr(content_data['release_date'], 'year') else ''
            if year:
                self._index_field(year, content_id, 'year')
        
        # Index cast/crew if available
        for person in content_data.get('cast', []):
            self._index_field(person, content_id, 'cast')
        
        # Index keywords/tags
        for keyword in content_data.get('keywords', []):
            self._index_field(keyword, content_id, 'keywords')
    
    def _index_field(self, text: str, content_id: int, field: str) -> None:
        """Index text for specific field with all strategies"""
        if not text:
            return
        
        text_lower = text.lower()
        
        # Tokenize
        tokens = self._tokenize(text_lower)
        
        # Update document length
        if content_id not in self.doc_lengths:
            self.doc_lengths[content_id] = 0
        self.doc_lengths[content_id] += len(tokens)
        
        # Index tokens
        for token in tokens:
            # Main inverted index
            self.inverted_index[token].add(content_id)
            
            # Field-specific index
            self.field_index[field][token].add(content_id)
            
            # Update term document frequency
            self.term_doc_freq[token] += 1
        
        # Generate and index n-grams
        ngrams = self.ngram_analyzer.analyze(text, 'standard')
        for ngram in ngrams:
            self.ngram_index[ngram].add(content_id)
        
        # Generate and index edge n-grams
        edge_ngrams = self.ngram_analyzer.analyze(text, 'edge')
        for ngram in edge_ngrams:
            self.ngram_index[ngram].add(content_id)
        
        # Generate and index phonetic n-grams
        phonetic_ngrams = self.ngram_analyzer.analyze(text, 'phonetic')
        for ngram in phonetic_ngrams:
            self.phonetic_index[ngram].add(content_id)
    
    def _extract_content_data(self, content: Any) -> Dict[str, Any]:
        """Extract and prepare content data"""
        return {
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
            'is_trending': bool(content.is_trending),
            'is_new_release': bool(content.is_new_release),
            'is_critics_choice': bool(content.is_critics_choice),
            'tmdb_id': content.tmdb_id,
            'mal_id': content.mal_id,
            'cast': [],  # Can be populated if available
            'keywords': []  # Can be populated if available
        }
    
    def _create_feature_vector(self, content_data: Dict) -> np.ndarray:
        """Create feature vector for ML-based ranking"""
        features = []
        
        # Numeric features (normalized)
        features.append(content_data['rating'] / 10.0 if content_data['rating'] else 0.0)
        features.append(min(1.0, content_data['vote_count'] / 10000.0) if content_data['vote_count'] else 0.0)
        features.append(min(1.0, content_data['popularity'] / 100.0) if content_data['popularity'] else 0.0)
        
        # Boolean features
        features.append(1.0 if content_data['is_trending'] else 0.0)
        features.append(1.0 if content_data['is_new_release'] else 0.0)
        features.append(1.0 if content_data['is_critics_choice'] else 0.0)
        
        # Release year (normalized to recent = 1.0)
        if content_data['release_date']:
            try:
                year = content_data['release_date'].year if hasattr(content_data['release_date'], 'year') else 2000
                years_old = datetime.now().year - year
                features.append(max(0.0, 1.0 - years_old / 50.0))
            except:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # Genre diversity
        features.append(min(1.0, len(content_data['genres']) / 5.0))
        
        return np.array(features)
    
    def _tokenize(self, text: str) -> List[str]:
        """Advanced tokenization with stemming simulation"""
        # Remove special characters
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Filter and process tokens
        processed = []
        for token in tokens:
            if len(token) > 1:
                # Simple stemming (remove common suffixes)
                if len(token) > 4:
                    if token.endswith('ing'):
                        token = token[:-3]
                    elif token.endswith('ed'):
                        token = token[:-2]
                    elif token.endswith('s') and not token.endswith('ss'):
                        token = token[:-1]
                
                processed.append(token)
        
        return processed
    
    def _calculate_statistics(self):
        """Calculate statistics for BM25 scoring"""
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        else:
            self.avg_doc_length = 0
    
    def _build_prefix_trie(self):
        """Build trie structure for ultra-fast prefix search"""
        self.prefix_trie = {}
        
        for content_id, content_data in self.content_data.items():
            title = content_data['title'].lower()
            
            # Insert into trie
            node = self.prefix_trie
            for char in title:
                if char not in node:
                    node[char] = {'ids': set(), 'children': {}}
                node[char]['ids'].add(content_id)
                node = node[char]['children']
    
    def search_prefix(self, prefix: str) -> Set[int]:
        """Ultra-fast prefix search using trie"""
        prefix = prefix.lower()
        node = self.prefix_trie
        
        for char in prefix:
            if char in node:
                node = node[char]['children']
            else:
                return set()
        
        # Collect all IDs from this node
        return self._collect_trie_ids(node)
    
    def _collect_trie_ids(self, node: Dict) -> Set[int]:
        """Recursively collect all IDs from trie node"""
        ids = set()
        
        for char, data in node.items():
            ids.update(data['ids'])
            ids.update(self._collect_trie_ids(data['children']))
        
        return ids

class EnhancedSearchEngine:
    """Enhanced search engine with all advanced features"""
    
    def __init__(self, db_session, Content, tmdb_api_key=None, ContentService=None, http_session=None):
        self.db = db_session
        self.Content = Content
        self.ContentService = ContentService
        
        # Initialize configuration
        self.config = SearchConfig()
        
        # Initialize components
        self.search_index = OptimizedSearchIndex(self.config)
        self.fuzzy_matcher = AdvancedFuzzyMatcher(self.config)
        self.ngram_analyzer = AdvancedNGramAnalyzer(self.config)
        
        # TMDB fetcher for external content
        from .search import TMDBFetcher  # Import from original file
        self.tmdb_fetcher = TMDBFetcher(tmdb_api_key, http_session) if tmdb_api_key else None
        
        # Threading and caching
        self._index_lock = threading.Lock()
        self._last_index_update = None
        self._executor = ThreadPoolExecutor(max_workers=5)
        
        # Query cache for instant repeated searches
        self._query_cache = {}
        self._cache_lock = threading.RLock()
        
        # Initialize index
        self._ensure_index_updated()
    
    def _ensure_index_updated(self) -> None:
        """Ensure search index is up to date"""
        current_time = datetime.utcnow()
        
        if (not self._last_index_update or 
            (current_time - self._last_index_update).seconds > self.config.index_update_interval):
            
            if not self._index_lock.locked():
                with self._index_lock:
                    try:
                        contents = self.Content.query.all()
                        self.search_index.build_index(contents)
                        self._last_index_update = current_time
                        # Clear query cache on index update
                        self._query_cache.clear()
                    except Exception as e:
                        logger.error(f"Index update error: {e}")
    
    @measure_performance
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
        Perform advanced search with all optimizations
        """
        start_time = time.perf_counter()
        
        # Check query cache first
        cache_key = self._generate_cache_key(query, content_type, genres, languages, min_rating, sort_by)
        
        with self._cache_lock:
            if cache_key in self._query_cache:
                cached_result = self._query_cache[cache_key]
                # Return paginated cached result
                return self._paginate_cached_result(cached_result, page, per_page)
        
        # Ensure index is updated
        self._ensure_index_updated()
        
        # Normalize query
        query = query.strip()
        if not query:
            return self._empty_result()
        
        query_lower = query.lower()
        
        # Step 1: Exact matching (highest priority)
        exact_matches = self._find_exact_matches(query_lower)
        
        # Step 2: Prefix matching (very high priority)
        prefix_matches = self._find_prefix_matches(query_lower)
        
        # Step 3: Field-boosted search
        field_matches = self._find_field_matches(query_lower)
        
        # Step 4: Fuzzy matching for typos
        fuzzy_matches = self._find_fuzzy_matches(query_lower)
        
        # Step 5: N-gram based matching
        ngram_matches = self._find_ngram_matches(query_lower)
        
        # Combine all matches with proper scoring
        all_matches = self._combine_and_score_matches(
            query_lower,
            exact_matches,
            prefix_matches,
            field_matches,
            fuzzy_matches,
            ngram_matches
        )
        
        # Apply filters
        filtered_results = self._apply_advanced_filters(
            all_matches,
            content_type,
            genres,
            languages,
            min_rating
        )
        
        # Apply advanced sorting
        sorted_results = self._apply_advanced_sorting(filtered_results, sort_by)
        
        # Cache the results
        with self._cache_lock:
            if len(self._query_cache) < self.config.max_cache_size:
                self._query_cache[cache_key] = sorted_results
        
        # Paginate
        total_results = len(sorted_results)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = sorted_results[start_idx:end_idx]
        
        # Format results
        formatted_results = self._format_advanced_results(paginated_results)
        
        # Calculate metrics
        search_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        return {
            'query': query,
            'results': formatted_results,
            'total_results': total_results,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_results + per_page - 1) // per_page if per_page > 0 else 0,
            'search_time': f"{search_time:.2f}ms",
            'search_metadata': {
                'exact_matches': len(exact_matches),
                'prefix_matches': len(prefix_matches),
                'fuzzy_matches': len(fuzzy_matches),
                'total_candidates': len(all_matches),
                'filters_applied': {
                    'content_type': content_type,
                    'genres': genres,
                    'languages': languages,
                    'min_rating': min_rating
                },
                'sort_method': sort_by,
                'index_size': len(self.search_index.content_data),
                'cache_hit': False
            }
        }
    
    def _find_exact_matches(self, query: str) -> List[Tuple[int, float]]:
        """Find exact matches with maximum score"""
        matches = []
        
        # Check exact index
        if query in self.search_index.exact_index:
            content_id = self.search_index.exact_index[query]
            matches.append((content_id, self.config.field_boosts['title_exact']))
        
        # Check clean query
        clean_query = re.sub(r'[^\w\s]', '', query)
        if clean_query != query and clean_query in self.search_index.exact_index:
            content_id = self.search_index.exact_index[clean_query]
            if not any(m[0] == content_id for m in matches):
                matches.append((content_id, self.config.field_boosts['title_exact'] * 0.95))
        
        return matches
    
    def _find_prefix_matches(self, query: str) -> List[Tuple[int, float]]:
        """Find prefix matches using trie"""
        matches = []
        
        # Use trie for ultra-fast prefix search
        prefix_ids = self.search_index.search_prefix(query)
        
        for content_id in prefix_ids:
            content_data = self.search_index.content_data.get(content_id)
            if content_data:
                title = content_data['title'].lower()
                
                # Calculate position-based score
                if title.startswith(query):
                    score = self.config.field_boosts['title_prefix']
                elif query in title:
                    position = title.index(query)
                    position_penalty = position / len(title)
                    score = self.config.field_boosts['title_prefix'] * (1 - position_penalty * 0.5)
                else:
                    continue
                
                matches.append((content_id, score))
        
        return matches
    
    def _find_field_matches(self, query: str) -> List[Tuple[int, float]]:
        """Find matches with field boosting"""
        matches = defaultdict(float)
        query_tokens = self.search_index._tokenize(query)
        
        for token in query_tokens:
            # Search each field with appropriate boost
            for field, boost in self.config.field_boosts.items():
                if field in ['title_exact', 'title_prefix']:
                    continue  # These are handled separately
                
                if field in self.search_index.field_index:
                    if token in self.search_index.field_index[field]:
                        for content_id in self.search_index.field_index[field][token]:
                            # Calculate BM25 score
                            bm25_score = self._calculate_bm25_score(token, content_id, field)
                            matches[content_id] += bm25_score * boost
        
        return list(matches.items())
    
    def _find_fuzzy_matches(self, query: str) -> List[Tuple[int, float]]:
        """Find fuzzy matches for handling typos"""
        matches = []
        seen_ids = set()
        
        # Generate fuzzy candidates
        for content_id, content_data in self.search_index.content_data.items():
            if content_id in seen_ids:
                continue
            
            title = content_data['title']
            
            # Calculate fuzzy score
            fuzzy_score = self.fuzzy_matcher.fuzzy_score(query, title, 'hybrid')
            
            if fuzzy_score >= self.config.fuzzy_threshold:
                matches.append((content_id, fuzzy_score * 10))  # Scale the score
                seen_ids.add(content_id)
        
        return matches
    
    def _find_ngram_matches(self, query: str) -> List[Tuple[int, float]]:
        """Find matches using n-gram analysis"""
        matches = defaultdict(float)
        
        # Generate query n-grams
        query_ngrams = self.ngram_analyzer.analyze(query, 'standard')
        query_edge_ngrams = self.ngram_analyzer.analyze(query, 'edge')
        
        # Score based on n-gram overlap
        for ngram in query_ngrams:
            if ngram in self.search_index.ngram_index:
                for content_id in self.search_index.ngram_index[ngram]:
                    matches[content_id] += 1.0
        
        for ngram in query_edge_ngrams:
            if ngram in self.search_index.ngram_index:
                for content_id in self.search_index.ngram_index[ngram]:
                    matches[content_id] += 0.5
        
        # Normalize scores
        max_score = max(matches.values()) if matches else 1.0
        
        return [(cid, (score / max_score) * 5) for cid, score in matches.items()]
    
    def _calculate_bm25_score(self, term: str, doc_id: int, field: str) -> float:
        """Calculate BM25 relevance score"""
        k1 = 1.2  # Term saturation parameter
        b = 0.75  # Length normalization parameter
        
        # Document frequency
        df = len(self.search_index.inverted_index[term])
        
        # Inverse document frequency
        N = self.search_index.total_docs or 1
        idf = log10((N - df + 0.5) / (df + 0.5) + 1)
        
        # Term frequency in document
        tf = 1  # Simplified - count term occurrences in field
        
        # Document length
        doc_len = self.search_index.doc_lengths.get(doc_id, 0)
        avg_doc_len = self.search_index.avg_doc_length or 1
        
        # BM25 formula
        score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        
        return score
    
    def _combine_and_score_matches(self, query: str, *match_lists) -> List[Tuple[int, float]]:
        """Combine and score all matches"""
        combined = defaultdict(float)
        
        # Combine all match lists
        for matches in match_lists:
            for content_id, score in matches:
                combined[content_id] += score
        
        # Add popularity and rating boosts
        for content_id in combined.keys():
            content_data = self.search_index.content_data.get(content_id)
            if content_data:
                # Popularity boost
                popularity = content_data.get('popularity', 0) or 0
                if popularity > 0:
                    combined[content_id] += log10(popularity + 1) * self.config.popularity_weight
                
                # Rating boost
                rating = content_data.get('rating', 0) or 0
                vote_count = content_data.get('vote_count', 0) or 0
                if rating > 0 and vote_count > 10:
                    combined[content_id] += (rating / 10) * log10(vote_count + 1) * self.config.rating_weight
                
                # Recency boost
                if content_data.get('is_new_release'):
                    combined[content_id] += self.config.recency_weight * 10
                
                # Trending boost
                if content_data.get('is_trending'):
                    combined[content_id] += self.config.trending_weight * 10
        
        # Filter by minimum score threshold
        filtered = [(cid, score) for cid, score in combined.items() 
                   if score >= self.config.min_score_threshold]
        
        # Sort by score
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        return filtered[:self.config.max_candidates]
    
    def _apply_advanced_filters(self, results, content_type, genres, languages, min_rating):
        """Apply advanced filtering with optimization"""
        if not any([content_type, genres, languages, min_rating]):
            return results
        
        filtered = []
        
        for content_id, score in results:
            content_data = self.search_index.content_data.get(content_id)
            if not content_data:
                continue
            
            # Fast filtering checks
            if content_type and content_data['content_type'] != content_type:
                continue
            
            if genres:
                content_genres = set(g.lower() for g in content_data['genres'])
                query_genres = set(g.lower() for g in genres)
                if not content_genres & query_genres:
                    continue
            
            if languages:
                content_langs = set(l.lower() for l in content_data['languages'])
                query_langs = set(l.lower() for l in languages)
                if not content_langs & query_langs:
                    continue
            
            if min_rating:
                if (content_data['rating'] or 0) < min_rating:
                    continue
            
            filtered.append((content_id, score))
        
        return filtered
    
    def _apply_advanced_sorting(self, results, sort_by):
        """Apply advanced sorting strategies"""
        if sort_by == 'relevance':
            return results  # Already sorted by relevance
        
        # Create sort key functions
        def get_sort_value(item):
            content_id, relevance_score = item
            content_data = self.search_index.content_data.get(content_id, {})
            
            if sort_by == 'rating':
                rating = content_data.get('rating', 0) or 0
                vote_count = content_data.get('vote_count', 0) or 0
                # Weighted rating to account for vote count
                if vote_count > 0:
                    return rating * log10(vote_count + 1)
                return rating
            
            elif sort_by == 'popularity':
                return content_data.get('popularity', 0) or 0
            
            elif sort_by == 'date':
                date = content_data.get('release_date')
                if date:
                    return date.timestamp() if hasattr(date, 'timestamp') else 0
                return 0
            
            elif sort_by == 'trending':
                # Custom trending score
                popularity = content_data.get('popularity', 0) or 0
                is_trending = 10 if content_data.get('is_trending') else 0
                is_new = 5 if content_data.get('is_new_release') else 0
                return popularity + is_trending + is_new
            
            return 0
        
        # Sort with relevance as secondary key
        sorted_results = sorted(results, 
                              key=lambda x: (get_sort_value(x), x[1]), 
                              reverse=True)
        
        return sorted_results
    
    def _format_advanced_results(self, results):
        """Format results with advanced metadata"""
        formatted = []
        
        for content_id, score in results:
            content_data = self.search_index.content_data.get(content_id)
            if not content_data:
                continue
            
            # Determine match quality
            match_quality = 'excellent' if score > 30 else 'good' if score > 10 else 'fair'
            
            formatted_item = {
                'id': content_data['id'],
                'title': content_data['title'],
                'original_title': content_data['original_title'],
                'content_type': content_data['content_type'],
                'genres': content_data['genres'],
                'languages': content_data['languages'],
                'overview': content_data['overview'][:200] + '...' if len(content_data['overview']) > 200 else content_data['overview'],
                'rating': content_data['rating'],
                'vote_count': content_data['vote_count'],
                'popularity': content_data['popularity'],
                'release_date': content_data['release_date'].isoformat() if content_data['release_date'] else None,
                'poster_path': self._format_image_path(content_data['poster_path']),
                'backdrop_path': self._format_image_path(content_data['backdrop_path'], 'backdrop'),
                'is_trending': content_data['is_trending'],
                'is_new_release': content_data['is_new_release'],
                'is_critics_choice': content_data['is_critics_choice'],
                'relevance_score': round(score, 2),
                'match_quality': match_quality
            }
            
            # Add YouTube trailer if available
            if content_data['youtube_trailer_id']:
                formatted_item['youtube_trailer'] = f"https://www.youtube.com/watch?v={content_data['youtube_trailer_id']}"
            
            formatted.append(formatted_item)
        
        return formatted
    
    def _format_image_path(self, path, type='poster'):
        """Format image path to full URL"""
        if not path:
            return None
        if path.startswith('http'):
            return path
        
        size = 'w500' if type == 'poster' else 'w1280'
        return f"https://image.tmdb.org/t/p/{size}{path}"
    
    def _generate_cache_key(self, query, content_type, genres, languages, min_rating, sort_by):
        """Generate unique cache key for query"""
        key_parts = [
            query,
            content_type or '',
            ','.join(genres) if genres else '',
            ','.join(languages) if languages else '',
            str(min_rating) if min_rating else '',
            sort_by
        ]
        
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _paginate_cached_result(self, cached_results, page, per_page):
        """Paginate cached results"""
        total_results = len(cached_results)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated = cached_results[start_idx:end_idx]
        
        formatted = self._format_advanced_results(paginated)
        
        return {
            'query': '',  # Would need to store this in cache
            'results': formatted,
            'total_results': total_results,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_results + per_page - 1) // per_page if per_page > 0 else 0,
            'search_time': '0.00ms',  # Instant from cache
            'search_metadata': {
                'cache_hit': True
            }
        }
    
    def _empty_result(self):
        """Return empty result structure"""
        return {
            'query': '',
            'results': [],
            'total_results': 0,
            'page': 1,
            'per_page': 20,
            'total_pages': 0,
            'search_time': '0.00ms',
            'search_metadata': {}
        }
    
    @measure_performance
    def autocomplete(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Ultra-fast autocomplete with advanced features"""
        if not query or len(query) < 1:
            return []
        
        query_lower = query.lower()
        suggestions = []
        seen_titles = set()
        
        # Use prefix trie for instant results
        prefix_ids = self.search_index.search_prefix(query_lower)
        
        # Score and rank suggestions
        scored_suggestions = []
        
        for content_id in prefix_ids:
            content_data = self.search_index.content_data.get(content_id)
            if not content_data or content_data['title'].lower() in seen_titles:
                continue
            
            title = content_data['title']
            title_lower = title.lower()
            
            # Calculate relevance score
            score = 0
            
            # Exact prefix match
            if title_lower.startswith(query_lower):
                score = 1.0
            # Word prefix match
            elif any(word.startswith(query_lower) for word in title_lower.split()):
                score = 0.8
            # Contains query
            elif query_lower in title_lower:
                score = 0.6
            else:
                # Fuzzy match
                score = self.fuzzy_matcher.fuzzy_score(query_lower, title_lower, 'hybrid') * 0.5
            
            if score > 0.3:
                # Add popularity boost
                popularity = content_data.get('popularity', 0) or 0
                score += min(0.2, log10(popularity + 1) / 10)
                
                scored_suggestions.append({
                    'title': title,
                    'content_type': content_data['content_type'],
                    'poster_path': self._format_image_path(content_data['poster_path']),
                    'year': content_data['release_date'].year if content_data['release_date'] and hasattr(content_data['release_date'], 'year') else None,
                    'rating': content_data['rating'],
                    'score': score
                })
                seen_titles.add(title_lower)
        
        # Sort by score
        scored_suggestions.sort(key=lambda x: x['score'], reverse=True)
        
        # Format final suggestions
        for item in scored_suggestions[:limit]:
            suggestions.append({
                'title': item['title'],
                'content_type': item['content_type'],
                'poster_path': item['poster_path'],
                'year': item['year'],
                'rating': item['rating']
            })
        
        return suggestions

# Singleton instance management
_enhanced_search_engine = None

def get_enhanced_search_engine(db_session, Content, tmdb_api_key=None, ContentService=None, http_session=None):
    """Get or create enhanced search engine instance"""
    global _enhanced_search_engine
    if _enhanced_search_engine is None:
        _enhanced_search_engine = EnhancedSearchEngine(
            db_session, 
            Content, 
            tmdb_api_key, 
            ContentService,
            http_session
        )
    return _enhanced_search_engine

# Public API functions for use in app.py
def search_content(db_session, Content, query, tmdb_api_key=None, ContentService=None, http_session=None, **kwargs):
    """
    Enhanced search function with all advanced features
    
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
        engine = get_enhanced_search_engine(db_session, Content, tmdb_api_key, ContentService, http_session)
        return engine.search(query, **kwargs)
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return {
            'query': query,
            'results': [],
            'total_results': 0,
            'page': kwargs.get('page', 1),
            'per_page': kwargs.get('per_page', 20),
            'total_pages': 0,
            'search_time': '0.00ms',
            'error': str(e)
        }

def get_autocomplete_suggestions(db_session, Content, query, limit=10):
    """
    Enhanced autocomplete with ultra-fast response
    
    Usage:
        from search import get_autocomplete_suggestions
        suggestions = get_autocomplete_suggestions(db.session, Content, "ava", limit=5)
    """
    try:
        engine = get_enhanced_search_engine(db_session, Content)
        return engine.autocomplete(query, limit)
    except Exception as e:
        logger.error(f"Autocomplete error: {e}", exc_info=True)
        return []

def rebuild_search_index(db_session, Content):
    """
    Force rebuild of search index with all optimizations
    
    Usage:
        from search import rebuild_search_index
        count = rebuild_search_index(db.session, Content)
    """
    try:
        engine = get_enhanced_search_engine(db_session, Content)
        contents = Content.query.all()
        engine.search_index.build_index(contents)
        engine._last_index_update = datetime.utcnow()
        engine._query_cache.clear()  # Clear query cache after rebuild
        logger.info(f"Search index rebuilt with {len(contents)} items")
        return len(contents)
    except Exception as e:
        logger.error(f"Index rebuild error: {e}", exc_info=True)
        return 0