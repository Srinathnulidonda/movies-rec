# backend/search.py

import json
import re
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Set
from difflib import SequenceMatcher
from functools import lru_cache, wraps
import hashlib
import unicodedata
from math import log10, sqrt, exp
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import threading
from queue import PriorityQueue
import heapq

logger = logging.getLogger(__name__)

# API Configuration
TMDB_API_KEY = '1cf86635f20bb2aff8e70940e7c3ddd5'
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

def timed_lru_cache(seconds: int, maxsize: int = 128):
    """LRU cache with time expiration"""
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime
        
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime
            return func(*args, **kwargs)
        
        return wrapped_func
    return wrapper_cache

class PhoneticMatcher:
    """Phonetic matching for handling similar sounding words"""
    
    @staticmethod
    def metaphone(word: str, max_length: int = 10) -> str:
        """Generate metaphone code for phonetic matching"""
        if not word:
            return ""
        
        word = word.upper()
        
        # Metaphone rules
        vowels = set('AEIOU')
        metaph = []
        
        # Preprocess
        if word.startswith('KN') or word.startswith('GN') or word.startswith('PN'):
            word = word[1:]
        elif word.startswith('WR'):
            word = word[1:]
        elif word.startswith('X'):
            word = 'S' + word[1:]
        
        i = 0
        while i < len(word) and len(metaph) < max_length:
            ch = word[i]
            
            # Skip duplicates except 'C'
            if i > 0 and ch == word[i-1] and ch != 'C':
                i += 1
                continue
            
            # Main rules
            if ch in vowels:
                if i == 0:
                    metaph.append(ch)
            elif ch == 'B':
                if i != len(word) - 1 or (i > 0 and word[i-1] != 'M'):
                    metaph.append('B')
            elif ch == 'C':
                if i + 1 < len(word):
                    if word[i+1] in 'IEY':
                        metaph.append('S')
                    elif word[i+1] == 'H':
                        metaph.append('X')
                        i += 1
                    else:
                        metaph.append('K')
                else:
                    metaph.append('K')
            elif ch == 'D':
                if i + 2 < len(word) and word[i+1:i+3] == 'GE':
                    metaph.append('J')
                else:
                    metaph.append('T')
            elif ch == 'G':
                if i + 1 < len(word) and word[i+1] in 'IEY':
                    metaph.append('J')
                else:
                    metaph.append('K')
            elif ch == 'H':
                if i == 0 or word[i-1] not in vowels:
                    if i + 1 < len(word) and word[i+1] in vowels:
                        metaph.append('H')
            elif ch == 'K':
                if i == 0 or word[i-1] != 'C':
                    metaph.append('K')
            elif ch == 'P':
                if i + 1 < len(word) and word[i+1] == 'H':
                    metaph.append('F')
                    i += 1
                else:
                    metaph.append('P')
            elif ch == 'Q':
                metaph.append('K')
            elif ch == 'S':
                if i + 2 < len(word) and word[i+1:i+3] == 'CH':
                    metaph.append('X')
                    i += 2
                else:
                    metaph.append('S')
            elif ch == 'T':
                if i + 2 < len(word) and word[i+1:i+3] == 'CH':
                    metaph.append('X')
                    i += 2
                elif i + 2 < len(word) and word[i+1:i+3] == 'IA':
                    metaph.append('X')
                else:
                    metaph.append('T')
            elif ch == 'V':
                metaph.append('F')
            elif ch == 'W' or ch == 'Y':
                if i + 1 < len(word) and word[i+1] in vowels:
                    metaph.append(ch)
            elif ch == 'X':
                metaph.append('KS')
            elif ch == 'Z':
                metaph.append('S')
            else:
                metaph.append(ch)
            
            i += 1
        
        return ''.join(metaph[:max_length])
    
    @staticmethod
    def soundex(word: str) -> str:
        """Generate Soundex code for phonetic matching"""
        if not word:
            return ""
        
        word = word.upper()
        
        # Soundex mappings
        soundex_map = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }
        
        # Keep first letter
        soundex = word[0]
        
        # Map remaining letters
        for char in word[1:]:
            if char in soundex_map:
                code = soundex_map[char]
                if len(soundex) == 1 or soundex[-1] != code:
                    soundex += code
        
        # Remove vowels (except first letter)
        soundex = soundex[0] + ''.join(c for c in soundex[1:] if c != '0')
        
        # Pad with zeros or truncate to length 4
        soundex = (soundex + '000')[:4]
        
        return soundex

class AdvancedNGramAnalyzer:
    """Advanced N-gram analyzer with edge n-grams and skip-grams"""
    
    def __init__(self, min_gram: int = 2, max_gram: int = 5, edge_gram: bool = True):
        self.min_gram = min_gram
        self.max_gram = max_gram
        self.edge_gram = edge_gram
        self._cache = {}
        self._edge_cache = {}
    
    def analyze(self, text: str, use_edge: bool = False) -> Set[str]:
        """Generate n-grams from text"""
        if not text:
            return set()
        
        # Check cache
        cache_key = f"{text}_{self.min_gram}_{self.max_gram}_{use_edge}"
        cache = self._edge_cache if use_edge else self._cache
        
        if cache_key in cache:
            return cache[cache_key]
        
        # Normalize text
        text = self._normalize_text(text)
        ngrams = set()
        
        if use_edge:
            # Edge n-grams (for autocomplete)
            ngrams.update(self._generate_edge_ngrams(text))
        else:
            # Standard n-grams
            ngrams.update(self._generate_standard_ngrams(text))
        
        # Add skip-grams for better partial matching
        ngrams.update(self._generate_skip_grams(text))
        
        # Add word-level n-grams
        ngrams.update(self._generate_word_ngrams(text))
        
        # Cache result
        cache[cache_key] = ngrams
        return ngrams
    
    def _generate_standard_ngrams(self, text: str) -> Set[str]:
        """Generate standard character n-grams"""
        ngrams = set()
        for n in range(self.min_gram, min(len(text) + 1, self.max_gram + 1)):
            for i in range(len(text) - n + 1):
                ngrams.add(text[i:i+n])
        return ngrams
    
    def _generate_edge_ngrams(self, text: str) -> Set[str]:
        """Generate edge n-grams (prefixes) for autocomplete"""
        ngrams = set()
        for n in range(self.min_gram, min(len(text) + 1, self.max_gram + 1)):
            ngrams.add(text[:n])  # Prefix n-grams
            if n <= len(text):
                ngrams.add(text[-n:])  # Suffix n-grams
        return ngrams
    
    def _generate_skip_grams(self, text: str, skip: int = 1) -> Set[str]:
        """Generate skip-grams for handling insertions/deletions"""
        ngrams = set()
        if len(text) > 3:
            for i in range(len(text) - 2 - skip):
                # Skip one character
                skip_gram = text[i] + text[i+1+skip:i+3+skip]
                if len(skip_gram) >= self.min_gram:
                    ngrams.add(skip_gram)
        return ngrams
    
    def _generate_word_ngrams(self, text: str) -> Set[str]:
        """Generate word-level n-grams"""
        words = text.split()
        ngrams = set()
        
        # Single words
        ngrams.update(words)
        
        # Word pairs and triples
        for n in range(2, min(len(words) + 1, 4)):
            for i in range(len(words) - n + 1):
                ngrams.add(' '.join(words[i:i+n]))
        
        # First letters of words (acronyms)
        if len(words) > 1:
            acronym = ''.join(word[0] for word in words if word)
            if len(acronym) >= 2:
                ngrams.add(acronym)
        
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

class EnhancedFuzzyMatcher:
    """Advanced fuzzy search with multiple algorithms and typo correction"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._cache = {}
        self.phonetic_matcher = PhoneticMatcher()
        
        # Common keyboard typos mapping
        self.keyboard_adjacency = {
            'q': 'wa', 'w': 'qeas', 'e': 'wrd', 'r': 'etf', 't': 'ryg',
            'y': 'tuh', 'u': 'yij', 'i': 'uok', 'o': 'ipl', 'p': 'ol',
            'a': 'qwsz', 's': 'awedx', 'd': 'serfxc', 'f': 'drtgcv',
            'g': 'ftyhvb', 'h': 'gyujbn', 'j': 'huikmn', 'k': 'jiolm',
            'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
            'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
        }
    
    @timed_lru_cache(seconds=300, maxsize=1000)
    def fuzzy_score(self, query: str, target: str) -> float:
        """Calculate comprehensive fuzzy match score"""
        query = query.lower()
        target = target.lower()
        
        # Exact match
        if query == target:
            return 1.0
        
        # Contains exact query
        if query in target:
            position_factor = 1 - (target.index(query) / len(target))
            return 0.9 + (0.05 * position_factor)
        
        # Calculate multiple similarity scores
        scores = []
        
        # 1. Sequence Matcher (good for general similarity)
        seq_score = SequenceMatcher(None, query, target).ratio()
        scores.append(seq_score * 0.25)
        
        # 2. Levenshtein distance (edit distance)
        lev_score = 1 - (self._levenshtein_distance(query, target) / max(len(query), len(target)))
        scores.append(lev_score * 0.25)
        
        # 3. Jaro-Winkler (good for typos)
        jw_score = self._jaro_winkler_similarity(query, target)
        scores.append(jw_score * 0.2)
        
        # 4. Phonetic similarity (for sound-alike matches)
        phonetic_score = self._phonetic_similarity(query, target)
        scores.append(phonetic_score * 0.15)
        
        # 5. Keyboard distance (for typos)
        keyboard_score = self._keyboard_similarity(query, target)
        scores.append(keyboard_score * 0.1)
        
        # 6. N-gram similarity
        ngram_score = self._ngram_similarity(query, target)
        scores.append(ngram_score * 0.05)
        
        return sum(scores)
    
    def _phonetic_similarity(self, s1: str, s2: str) -> float:
        """Calculate phonetic similarity using metaphone and soundex"""
        meta1 = self.phonetic_matcher.metaphone(s1)
        meta2 = self.phonetic_matcher.metaphone(s2)
        
        sound1 = self.phonetic_matcher.soundex(s1)
        sound2 = self.phonetic_matcher.soundex(s2)
        
        meta_score = 1.0 if meta1 == meta2 else SequenceMatcher(None, meta1, meta2).ratio()
        sound_score = 1.0 if sound1 == sound2 else SequenceMatcher(None, sound1, sound2).ratio()
        
        return (meta_score * 0.7 + sound_score * 0.3)
    
    def _keyboard_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity based on keyboard layout (typos)"""
        if len(s1) != len(s2):
            return 0.0
        
        matches = 0
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                matches += 1
            elif c1 in self.keyboard_adjacency and c2 in self.keyboard_adjacency[c1]:
                matches += 0.8  # Adjacent keys are likely typos
        
        return matches / len(s1)
    
    def _ngram_similarity(self, s1: str, s2: str, n: int = 2) -> float:
        """Calculate n-gram based similarity"""
        if len(s1) < n or len(s2) < n:
            return 0.0
        
        ngrams1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
        ngrams2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Optimized Levenshtein distance calculation"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        # Use only two rows for space optimization
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _jaro_winkler_similarity(self, s1: str, s2: str, p: float = 0.1) -> float:
        """Enhanced Jaro-Winkler similarity"""
        jaro = self._jaro_similarity(s1, s2)
        
        # Find common prefix up to 4 characters
        prefix = 0
        for i in range(min(len(s1), len(s2), 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        
        # Jaro-Winkler formula with scaling factor p
        return jaro + (prefix * p * (1 - jaro))
    
    def _jaro_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro similarity"""
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Match window
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
    
    def suggest_corrections(self, query: str, candidates: List[str], max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Suggest spelling corrections based on candidates"""
        suggestions = []
        
        for candidate in candidates:
            score = self.fuzzy_score(query, candidate)
            if score > self.threshold:
                suggestions.append((candidate, score))
        
        # Sort by score and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]

class OptimizedSearchIndex:
    """High-performance search index with advanced data structures"""
    
    def __init__(self):
        # Multiple index types for different search strategies
        self.inverted_index = defaultdict(set)  # term -> content IDs
        self.content_data = {}  # content ID -> content data
        self.ngram_index = defaultdict(set)  # n-gram -> content IDs
        self.edge_gram_index = defaultdict(set)  # edge n-gram -> content IDs
        self.phonetic_index = defaultdict(set)  # phonetic code -> content IDs
        self.field_index = defaultdict(lambda: defaultdict(set))  # field -> term -> content IDs
        self.exact_title_index = {}  # exact title -> content ID
        self.prefix_tree = {}  # Trie for prefix matching
        
        # Scoring metadata
        self.term_frequency = defaultdict(lambda: defaultdict(int))  # term -> content_id -> frequency
        self.document_frequency = defaultdict(int)  # term -> document count
        self.field_length = defaultdict(lambda: defaultdict(int))  # field -> content_id -> length
        
        # Analyzers
        self.ngram_analyzer = AdvancedNGramAnalyzer()
        self.phonetic_matcher = PhoneticMatcher()
        
        # Cache
        self.last_update = None
        self._total_documents = 0
    
    def build_index(self, contents: List[Any]) -> None:
        """Build comprehensive search index"""
        start_time = time.time()
        
        # Clear existing indexes
        self._clear_indexes()
        
        self._total_documents = len(contents)
        
        # Use thread pool for parallel indexing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for content in contents:
                future = executor.submit(self._index_content, content)
                futures.append(future)
            
            # Wait for all indexing to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error indexing content: {e}")
        
        # Calculate IDF scores
        self._calculate_idf_scores()
        
        self.last_update = datetime.utcnow()
        
        build_time = time.time() - start_time
        logger.info(f"Advanced index built for {len(contents)} items in {build_time:.3f}s")
    
    def _index_content(self, content: Any) -> None:
        """Index a single content item"""
        content_id = content.id
        
        # Store content data
        self.content_data[content_id] = self._extract_content_data(content)
        
        # Index title with multiple strategies
        if content.title:
            self._index_field(content.title, content_id, 'title')
            
            # Exact title index
            normalized_title = content.title.lower().strip()
            self.exact_title_index[normalized_title] = content_id
            
            # Prefix tree for autocomplete
            self._add_to_prefix_tree(normalized_title, content_id)
            
            # Phonetic index
            phonetic_code = self.phonetic_matcher.metaphone(content.title)
            self.phonetic_index[phonetic_code].add(content_id)
        
        # Index original title
        if content.original_title:
            self._index_field(content.original_title, content_id, 'original_title')
        
        # Index other fields
        self._index_json_field(content.genres, content_id, 'genre')
        self._index_json_field(content.languages, content_id, 'language')
        
        if content.overview:
            self._index_field(content.overview, content_id, 'overview')
        
        if content.content_type:
            self._index_field(content.content_type, content_id, 'content_type')
    
    def _index_field(self, text: str, content_id: int, field: str) -> None:
        """Index text for a specific field with multiple strategies"""
        if not text:
            return
        
        normalized = text.lower()
        
        # Count field length
        self.field_length[field][content_id] = len(normalized.split())
        
        # Token indexing with term frequency
        terms = self._tokenize(normalized)
        term_counts = Counter(terms)
        
        for term, count in term_counts.items():
            self.inverted_index[term].add(content_id)
            self.field_index[field][term].add(content_id)
            self.term_frequency[term][content_id] = count
            self.document_frequency[term] += 1
        
        # N-gram indexing
        ngrams = self.ngram_analyzer.analyze(text)
        for ngram in ngrams:
            self.ngram_index[ngram].add(content_id)
        
        # Edge n-gram indexing (for autocomplete)
        edge_grams = self.ngram_analyzer.analyze(text, use_edge=True)
        for edge_gram in edge_grams:
            self.edge_gram_index[edge_gram].add(content_id)
    
    def _index_json_field(self, json_data: str, content_id: int, field: str) -> None:
        """Index JSON field data"""
        try:
            items = json.loads(json_data or '[]')
            for item in items:
                if item:
                    self._index_field(item, content_id, field)
        except:
            pass
    
    def _add_to_prefix_tree(self, text: str, content_id: int) -> None:
        """Add text to prefix tree for autocomplete"""
        node = self.prefix_tree
        for char in text:
            if char not in node:
                node[char] = {'ids': set(), 'children': {}}
            node[char]['ids'].add(content_id)
            node = node[char]['children']
    
    def _extract_content_data(self, content: Any) -> Dict:
        """Extract and normalize content data"""
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
            'mal_id': content.mal_id
        }
    
    def _calculate_idf_scores(self) -> None:
        """Calculate IDF scores for terms"""
        self.idf_scores = {}
        for term, doc_count in self.document_frequency.items():
            # IDF = log(total_documents / documents_containing_term)
            self.idf_scores[term] = log10((self._total_documents + 1) / (doc_count + 1))
    
    def _tokenize(self, text: str) -> List[str]:
        """Advanced tokenization with stemming"""
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split into words
        words = text.split()
        # Simple stemming (remove common suffixes)
        stemmed = []
        for word in words:
            if len(word) > 3:
                # Basic stemming rules
                if word.endswith('ing'):
                    word = word[:-3]
                elif word.endswith('ed'):
                    word = word[:-2]
                elif word.endswith('ly'):
                    word = word[:-2]
                elif word.endswith('es'):
                    word = word[:-2]
                elif word.endswith('s') and not word.endswith('ss'):
                    word = word[:-1]
            if len(word) > 1:
                stemmed.append(word)
        return stemmed
    
    def _clear_indexes(self) -> None:
        """Clear all indexes"""
        self.inverted_index.clear()
        self.content_data.clear()
        self.ngram_index.clear()
        self.edge_gram_index.clear()
        self.phonetic_index.clear()
        self.field_index.clear()
        self.exact_title_index.clear()
        self.prefix_tree.clear()
        self.term_frequency.clear()
        self.document_frequency.clear()
        self.field_length.clear()

class AdvancedSearchEngine:
    """High-performance search engine with advanced features"""
    
    def __init__(self, db_session, Content):
        self.db = db_session
        self.Content = Content
        self.search_index = OptimizedSearchIndex()
        self.fuzzy_matcher = EnhancedFuzzyMatcher()
        self.ngram_analyzer = AdvancedNGramAnalyzer()
        self._index_lock = threading.Lock()
        self._last_index_update = None
        self._index_update_interval = 300  # 5 minutes
        
        # Dynamic field boost weights based on search context
        self.base_field_boosts = {
            'title': 15.0,
            'original_title': 12.0,
            'genre': 8.0,
            'content_type': 5.0,
            'language': 4.0,
            'overview': 2.0
        }
        
        # Query expansion settings
        self.enable_query_expansion = True
        self.enable_spell_correction = True
        
        # Initialize index
        self._ensure_index_updated()
    
    def _ensure_index_updated(self) -> None:
        """Ensure search index is up to date"""
        current_time = datetime.utcnow()
        
        if (not self._last_index_update or 
            (current_time - self._last_index_update).seconds > self._index_update_interval):
            
            with self._index_lock:
                if (not self._last_index_update or 
                    (current_time - self._last_index_update).seconds > self._index_update_interval):
                    
                    # Get all content from database
                    contents = self.Content.query.all()
                    self.search_index.build_index(contents)
                    self._last_index_update = current_time
    
    def search(self, 
               query: str, 
               content_type: Optional[str] = None,
               genres: Optional[List[str]] = None,
               languages: Optional[List[str]] = None,
               min_rating: Optional[float] = None,
               page: int = 1,
               per_page: int = 20,
               sort_by: str = 'relevance',
               search_mode: str = 'smart') -> Dict[str, Any]:
        """
        Advanced search with multiple strategies
        
        search_mode options:
        - 'exact': Only exact matches
        - 'fuzzy': Fuzzy matching with typo tolerance
        - 'smart': Automatic mode selection based on query
        - 'phonetic': Sound-based matching
        """
        start_time = time.time()
        
        # Ensure index is updated
        self._ensure_index_updated()
        
        # Normalize query
        original_query = query.strip()
        if not original_query:
            return self._empty_result()
        
        query_lower = original_query.lower()
        
        # Query expansion and spell correction
        expanded_queries = self._expand_query(query_lower) if self.enable_query_expansion else [query_lower]
        
        # Get candidates using multiple strategies
        all_candidates = set()
        strategy_results = {}
        
        # 1. Exact match strategy (highest priority)
        exact_matches = self._find_exact_matches(query_lower)
        strategy_results['exact'] = exact_matches
        
        # 2. Prefix match strategy (for autocomplete-like results)
        prefix_matches = self._find_prefix_matches(query_lower)
        strategy_results['prefix'] = prefix_matches
        
        # 3. Phonetic match strategy (for sound-alike)
        if search_mode in ['smart', 'phonetic']:
            phonetic_matches = self._find_phonetic_matches(query_lower)
            strategy_results['phonetic'] = phonetic_matches
        
        # 4. N-gram fuzzy match strategy
        if search_mode in ['smart', 'fuzzy']:
            fuzzy_matches = self._find_fuzzy_matches(expanded_queries)
            strategy_results['fuzzy'] = fuzzy_matches
        
        # 5. Field-specific search
        field_matches = self._find_field_matches(expanded_queries)
        strategy_results['field'] = field_matches
        
        # Combine and score all candidates
        scored_results = self._combine_and_score(
            strategy_results, 
            original_query,
            search_mode
        )
        
        # Apply filters
        filtered_results = self._apply_filters(
            scored_results,
            content_type,
            genres,
            languages,
            min_rating
        )
        
        # Sort results
        sorted_results = self._sort_results(filtered_results, sort_by)
        
        # Paginate
        total_results = len(sorted_results)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = sorted_results[start_idx:end_idx]
        
        # Format results
        formatted_results = self._format_results(paginated_results)
        
        # Calculate search metrics
        search_time = time.time() - start_time
        
        return {
            'query': original_query,
            'results': formatted_results,
            'total_results': total_results,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_results + per_page - 1) // per_page if per_page > 0 else 0,
            'search_time': f"{search_time:.3f}s",
            'search_mode': search_mode,
            'strategies_used': list(strategy_results.keys()),
            'filters_applied': {
                'content_type': content_type,
                'genres': genres,
                'languages': languages,
                'min_rating': min_rating
            }
        }
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and variations"""
        expanded = [query]
        
        # Common movie/TV synonyms
        synonyms = {
            'film': ['movie', 'cinema', 'picture'],
            'movie': ['film', 'cinema', 'picture'],
            'series': ['tv', 'show', 'television'],
            'tv': ['series', 'show', 'television'],
            'anime': ['animation', 'japanese animation'],
            'sci-fi': ['science fiction', 'scifi'],
            'scifi': ['science fiction', 'sci-fi']
        }
        
        # Check if query contains any synonyms
        for word, syns in synonyms.items():
            if word in query:
                for syn in syns:
                    expanded.append(query.replace(word, syn))
        
        return expanded
    
    def _find_exact_matches(self, query: str) -> List[Tuple[int, float]]:
        """Find exact title matches"""
        matches = []
        
        # Check exact title index
        if query in self.search_index.exact_title_index:
            content_id = self.search_index.exact_title_index[query]
            matches.append((content_id, 1000.0))
        
        # Check for exact phrase in titles
        for content_id, content_data in self.search_index.content_data.items():
            title = content_data.get('title', '').lower()
            if query in title and content_id not in [m[0] for m in matches]:
                # Position-based scoring
                position = title.index(query)
                score = 900 * (1 - position / len(title))
                matches.append((content_id, score))
        
        return matches
    
    def _find_prefix_matches(self, query: str) -> List[Tuple[int, float]]:
        """Find prefix matches using prefix tree"""
        matches = []
        node = self.search_index.prefix_tree
        
        # Traverse prefix tree
        for i, char in enumerate(query):
            if char in node:
                node = node[char]
                # Add all content IDs at this node with decreasing score
                for content_id in node.get('ids', set()):
                    score = 800 * ((i + 1) / len(query))
                    matches.append((content_id, score))
                node = node.get('children', {})
            else:
                break
        
        return matches
    
    def _find_phonetic_matches(self, query: str) -> List[Tuple[int, float]]:
        """Find phonetically similar matches"""
        matches = []
        
        query_phonetic = PhoneticMatcher.metaphone(query)
        if query_phonetic in self.search_index.phonetic_index:
            for content_id in self.search_index.phonetic_index[query_phonetic]:
                matches.append((content_id, 700.0))
        
        return matches
    
    def _find_fuzzy_matches(self, queries: List[str]) -> List[Tuple[int, float]]:
        """Find fuzzy matches using n-grams"""
        matches = []
        candidates = set()
        
        for query in queries:
            # Get n-grams for query
            query_ngrams = self.ngram_analyzer.analyze(query)
            
            # Find candidates with matching n-grams
            ngram_matches = Counter()
            for ngram in query_ngrams:
                if ngram in self.search_index.ngram_index:
                    for content_id in self.search_index.ngram_index[ngram]:
                        ngram_matches[content_id] += 1
            
            # Score based on n-gram overlap
            for content_id, match_count in ngram_matches.items():
                overlap_ratio = match_count / len(query_ngrams)
                if overlap_ratio > 0.3:  # Threshold
                    content_data = self.search_index.content_data.get(content_id)
                    if content_data:
                        title = content_data.get('title', '')
                        fuzzy_score = self.fuzzy_matcher.fuzzy_score(query, title)
                        if fuzzy_score > self.fuzzy_matcher.threshold:
                            final_score = 600 * fuzzy_score * overlap_ratio
                            matches.append((content_id, final_score))
        
        return matches
    
    def _find_field_matches(self, queries: List[str]) -> List[Tuple[int, float]]:
        """Find matches in specific fields"""
        matches = []
        
        for query in queries:
            query_terms = self.search_index._tokenize(query)
            
            for field, boost in self.base_field_boosts.items():
                field_index = self.search_index.field_index[field]
                
                for term in query_terms:
                    if term in field_index:
                        for content_id in field_index[term]:
                            # TF-IDF scoring
                            tf = self.search_index.term_frequency[term][content_id]
                            idf = self.search_index.idf_scores.get(term, 1.0)
                            field_length = self.search_index.field_length[field][content_id]
                            
                            # Normalized TF-IDF
                            normalized_tf = tf / (1 + field_length)
                            score = normalized_tf * idf * boost * 100
                            
                            matches.append((content_id, score))
        
        return matches
    
    def _combine_and_score(self, 
                          strategy_results: Dict[str, List[Tuple[int, float]]], 
                          query: str,
                          search_mode: str) -> List[Tuple[int, float]]:
        """Combine results from different strategies with intelligent scoring"""
        combined_scores = defaultdict(float)
        strategy_weights = {
            'exact': 10.0,
            'prefix': 5.0,
            'phonetic': 3.0,
            'fuzzy': 2.0,
            'field': 1.0
        }
        
        # Adjust weights based on search mode
        if search_mode == 'exact':
            strategy_weights['fuzzy'] = 0.1
            strategy_weights['phonetic'] = 0.1
        elif search_mode == 'fuzzy':
            strategy_weights['fuzzy'] = 4.0
            strategy_weights['phonetic'] = 2.0
        
        # Combine scores from all strategies
        for strategy, results in strategy_results.items():
            weight = strategy_weights.get(strategy, 1.0)
            for content_id, score in results:
                combined_scores[content_id] += score * weight
        
        # Add popularity and quality boosts
        for content_id in combined_scores:
            content_data = self.search_index.content_data.get(content_id)
            if content_data:
                # Popularity boost
                popularity = content_data.get('popularity', 0) or 0
                if popularity > 0:
                    combined_scores[content_id] += log10(popularity + 1) * 10
                
                # Rating boost
                rating = content_data.get('rating', 0) or 0
                vote_count = content_data.get('vote_count', 0) or 0
                if rating > 0 and vote_count > 10:
                    combined_scores[content_id] += (rating / 10) * log10(vote_count + 1) * 5
                
                # Recency boost
                if content_data.get('is_new_release'):
                    combined_scores[content_id] *= 1.2
                
                # Trending boost
                if content_data.get('is_trending'):
                    combined_scores[content_id] *= 1.3
        
        # Convert to sorted list
        results = [(content_id, score) for content_id, score in combined_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def autocomplete(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Advanced autocomplete with multiple strategies"""
        if not query or len(query) < 2:
            return []
        
        self._ensure_index_updated()
        
        query = query.lower().strip()
        suggestions = []
        seen_titles = set()
        
        # 1. Prefix tree matches (fastest)
        node = self.search_index.prefix_tree
        prefix_ids = set()
        
        for char in query:
            if char in node:
                node = node[char]
                prefix_ids.update(node.get('ids', set()))
                node = node.get('children', {})
            else:
                break
        
        # Score and rank prefix matches
        prefix_scores = []
        for content_id in prefix_ids:
            content_data = self.search_index.content_data.get(content_id)
            if content_data:
                title = content_data.get('title', '')
                if title.lower().startswith(query):
                    score = 1.0
                else:
                    score = 0.8
                
                popularity = content_data.get('popularity', 0) or 0
                prefix_scores.append({
                    'title': title,
                    'content_type': content_data.get('content_type', 'unknown'),
                    'poster_path': content_data.get('poster_path', ''),
                    'score': score,
                    'popularity': popularity
                })
        
        # Sort by score and popularity
        prefix_scores.sort(key=lambda x: (x['score'], x['popularity']), reverse=True)
        
        # Add to suggestions
        for item in prefix_scores[:limit]:
            if item['title'].lower() not in seen_titles:
                suggestions.append({
                    'title': item['title'],
                    'content_type': item['content_type'],
                    'poster_path': self._format_poster_path(item['poster_path'])
                })
                seen_titles.add(item['title'].lower())
        
        # 2. Edge n-gram matches if needed
        if len(suggestions) < limit:
            edge_gram_ids = set()
            for edge_gram in self.ngram_analyzer.analyze(query, use_edge=True):
                if edge_gram in self.search_index.edge_gram_index:
                    edge_gram_ids.update(self.search_index.edge_gram_index[edge_gram])
            
            edge_scores = []
            for content_id in edge_gram_ids:
                if content_id not in prefix_ids:
                    content_data = self.search_index.content_data.get(content_id)
                    if content_data:
                        title = content_data.get('title', '')
                        if title.lower() not in seen_titles:
                            score = self.fuzzy_matcher.fuzzy_score(query, title.lower())
                            if score > 0.5:
                                edge_scores.append({
                                    'title': title,
                                    'content_type': content_data.get('content_type', 'unknown'),
                                    'poster_path': content_data.get('poster_path', ''),
                                    'score': score,
                                    'popularity': content_data.get('popularity', 0) or 0
                                })
            
            edge_scores.sort(key=lambda x: (x['score'], x['popularity']), reverse=True)
            
            for item in edge_scores[:limit - len(suggestions)]:
                suggestions.append({
                    'title': item['title'],
                    'content_type': item['content_type'],
                    'poster_path': self._format_poster_path(item['poster_path'])
                })
                seen_titles.add(item['title'].lower())
        
        return suggestions
    
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
            
            # Apply each filter
            if content_type and content_data.get('content_type') != content_type:
                continue
            
            if genres:
                content_genres = [g.lower() for g in content_data.get('genres', [])]
                if not any(genre.lower() in content_genres for genre in genres):
                    continue
            
            if languages:
                content_languages = [l.lower() for l in content_data.get('languages', [])]
                if not any(lang.lower() in content_languages for lang in languages):
                    continue
            
            if min_rating:
                rating = content_data.get('rating', 0) or 0
                if rating < min_rating:
                    continue
            
            filtered.append((content_id, score))
        
        return filtered
    
    def _sort_results(self, 
                     scored_results: List[Tuple[int, float]], 
                     sort_by: str) -> List[Tuple[int, float]]:
        """Sort results by specified criteria"""
        if sort_by == 'relevance':
            return scored_results  # Already sorted by relevance
        
        elif sort_by == 'rating':
            def get_rating(item):
                content = self.search_index.content_data.get(item[0], {})
                return content.get('rating', 0) or 0
            
            return sorted(scored_results, key=get_rating, reverse=True)
        
        elif sort_by == 'popularity':
            def get_popularity(item):
                content = self.search_index.content_data.get(item[0], {})
                return content.get('popularity', 0) or 0
            
            return sorted(scored_results, key=get_popularity, reverse=True)
        
        elif sort_by == 'date':
            def get_date(item):
                content = self.search_index.content_data.get(item[0], {})
                return content.get('release_date') or datetime.min.date()
            
            return sorted(scored_results, key=get_date, reverse=True)
        
        return scored_results
    
    def _format_results(self, scored_results: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        """Format search results for response"""
        formatted = []
        
        for content_id, score in scored_results:
            content_data = self.search_index.content_data.get(content_id)
            if not content_data:
                continue
            
            # Determine match quality
            match_quality = 'low'
            if score >= 1000:
                match_quality = 'perfect'
            elif score >= 800:
                match_quality = 'excellent'
            elif score >= 600:
                match_quality = 'very_good'
            elif score >= 400:
                match_quality = 'good'
            elif score >= 200:
                match_quality = 'fair'
            
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
                'release_date': content_data.get('release_date').isoformat() if content_data.get('release_date') else None,
                'poster_path': self._format_poster_path(content_data.get('poster_path', '')),
                'backdrop_path': self._format_backdrop_path(content_data.get('backdrop_path', '')),
                'youtube_trailer': f"https://www.youtube.com/watch?v={content_data.get('youtube_trailer_id')}" if content_data.get('youtube_trailer_id') else None,
                'is_trending': content_data.get('is_trending', False),
                'is_new_release': content_data.get('is_new_release', False),
                'is_critics_choice': content_data.get('is_critics_choice', False),
                'relevance_score': round(score, 3),
                'match_quality': match_quality
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
            'search_mode': 'smart',
            'strategies_used': [],
            'filters_applied': {}
        }

# Singleton instance
_search_engine_instance = None

def get_search_engine(db_session, Content):
    """Get or create search engine instance"""
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = AdvancedSearchEngine(db_session, Content)
    return _search_engine_instance

# Public API functions
def search_content(db_session, Content, query, **kwargs):
    """
    Advanced search function for app.py
    
    Usage:
        from search import search_content
        results = search_content(db.session, Content, "avatr", search_mode='fuzzy')
    """
    try:
        engine = get_search_engine(db_session, Content)
        return engine.search(query, **kwargs)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            'query': query,
            'results': [],
            'total_results': 0,
            'page': kwargs.get('page', 1),
            'per_page': kwargs.get('per_page', 20),
            'total_pages': 0,
            'search_time': '0.000s',
            'error': str(e)
        }

def get_autocomplete_suggestions(db_session, Content, query, limit=10):
    """
    Advanced autocomplete function
    """
    try:
        engine = get_search_engine(db_session, Content)
        return engine.autocomplete(query, limit)
    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        return []

def rebuild_search_index(db_session, Content):
    """Force rebuild of search index"""
    try:
        engine = get_search_engine(db_session, Content)
        contents = Content.query.all()
        engine.search_index.build_index(contents)
        engine._last_index_update = datetime.utcnow()
        return len(contents)
    except Exception as e:
        logger.error(f"Index rebuild error: {e}")
        return 0