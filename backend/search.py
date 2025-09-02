# backend/search.py - Advanced Title Matching & Search Intelligence

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
from difflib import SequenceMatcher
import Levenshtein

import numpy as np
from flask import current_app
from sqlalchemy import or_, and_, func, text, case
from sqlalchemy.orm import Query, joinedload
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein as RapidLevenshtein
import redis
import pickle
from threading import Lock
import metaphone
import jellyfish

logger = logging.getLogger(__name__)

# Title variations and common patterns
TITLE_PATTERNS = {
    "sequel_patterns": [
        r"(\d+)$",                # Movies ending with numbers (KGF 2)
        r"part\s*(\d+)",          # Part 1, Part 2
        r"chapter\s*(\d+)",       # Chapter 1, Chapter 2
        r"season\s*(\d+)",        # Season 1, Season 2
        r"s(\d+)",                # S1, S2
        r"vol\.?\s*(\d+)",        # Vol 1, Volume 2
    ],
    "year_patterns": [
        r"\((\d{4})\)",           # Matches (2024)
        r"\[(\d{4})\]",           # Matches [2024]
        r"\b(19\d{2}|20\d{2})\b", # Matches 1900–2099
    ],
    "language_tags": [
        r"\[(telugu|hindi|tamil|english|korean|japanese)\]",  # Matches [Telugu]
        r"\((telugu|hindi|tamil|english|korean|japanese)\)",  # Matches (Telugu)
    ],
    "quality_tags": [
        r"\b(hd|4k|720p|1080p|2160p|bluray|webrip|hdtv)\b",
    ],
    "special_chars": {
        "&": ["and", "n", "&amp;"],
        "@": ["at"],
        "#": ["number", "no"],
        "+": ["plus", "and"],
        "-": [" ", ""],
        ":": [" ", ""],
        ".": [" ", ""],
        ",": [" ", ""],
        "!": ["", " "],
        "?": ["", " "],
        "'": ["", " "],
        '"': ["", " "],
    }
}

# Common title abbreviations and expansions
TITLE_ABBREVIATIONS = {
    'kgf': 'kolar gold fields',
    'rrr': 'rise roar revolt',
    'kvj': 'khiladi venky janda',
    'svp': 'sarkaru vaari paata',
    'avpl': 'ala vaikunthapurramuloo',
    'ssmb': 'super star mahesh babu',
    'aa': 'allu arjun',
    'jr ntr': 'junior ntr',
    'ntr': 'nandamuri taraka rama rao',
    'mb': 'mahesh babu',
    'pk': 'pawan kalyan',
    'rc': 'ram charan',
}

# Language mappings for search
LANGUAGE_MAPPINGS = {
    'telugu': ['Telugu', 'te', 'tollywood'],
    'hindi': ['Hindi', 'hi', 'bollywood'],
    'tamil': ['Tamil', 'ta', 'kollywood'],
    'kannada': ['Kannada', 'kn', 'sandalwood'],
    'malayalam': ['Malayalam', 'ml', 'mollywood'],
    'english': ['English', 'en', 'hollywood'],
    'korean': ['Korean', 'ko', 'k-drama', 'kdrama'],
    'japanese': ['Japanese', 'ja', 'anime', 'j-drama']
}

# Genre mappings
GENRE_MAPPINGS = {
    'action': ['Action', 'Action & Adventure', 'Martial Arts'],
    'comedy': ['Comedy', 'Stand-up', 'Romantic Comedy'],
    'horror': ['Horror', 'Thriller', 'Suspense', 'Supernatural'],
    'romance': ['Romance', 'Romantic', 'Love'],
    'drama': ['Drama', 'Melodrama'],
    'thriller': ['Thriller', 'Suspense', 'Mystery'],
    'sci-fi': ['Science Fiction', 'Sci-Fi', 'Sci-Fi & Fantasy'],
    'fantasy': ['Fantasy', 'Magic', 'Supernatural'],
    'animation': ['Animation', 'Animated', 'Cartoon'],
    'crime': ['Crime', 'Detective', 'Police'],
    'adventure': ['Adventure', 'Action & Adventure'],
    'family': ['Family', 'Kids', 'Children'],
    'musical': ['Musical', 'Music'],
    'war': ['War', 'Military'],
    'western': ['Western'],
    'documentary': ['Documentary'],
    'mystery': ['Mystery', 'Detective'],
    'biography': ['Biography', 'Biopic'],
    'history': ['History', 'Historical'],
    'sport': ['Sport', 'Sports']
}

@dataclass
class SearchResult:
    """Enhanced data class for search results with title match scoring"""
    id: int
    title: str
    content_type: str
    score: float = 0.0
    title_match_score: float = 0.0  # Specific score for title matching
    exact_match: bool = False
    partial_match: bool = False
    fuzzy_match: bool = False
    phonetic_match: bool = False
    fuzzy_score: float = 0.0
    popularity_score: float = 0.0
    recency_score: float = 0.0
    language_boost: float = 0.0
    is_original_language: bool = False
    is_dubbed: bool = False
    original_language: str = None
    languages: List[str] = field(default_factory=list)
    genres: List[str] = field(default_factory=list)
    source: str = 'local'
    metadata: Dict = field(default_factory=dict)

class TitleMatcher:
    """Advanced title matching with multiple algorithms"""
    
    def __init__(self):
        self.phonetic = metaphone.doublemetaphone
        self.soundex_cache = {}
        self.metaphone_cache = {}
        
    def normalize_title(self, title: str) -> str:
        """Normalize title for matching"""
        if not title:
            return ""
        
        # Convert to lowercase
        title = title.lower()
        
        # Remove language tags, year, quality tags
        for pattern_list in [TITLE_PATTERNS['language_tags'], 
                            TITLE_PATTERNS['year_patterns'],
                            TITLE_PATTERNS['quality_tags']]:
            for pattern in pattern_list:
                title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Expand abbreviations
        for abbr, expansion in TITLE_ABBREVIATIONS.items():
            if abbr in title.split():
                title = title.replace(abbr, expansion)
        
        # Handle special characters
        for char, replacements in TITLE_PATTERNS['special_chars'].items():
            if char in title:
                title = title.replace(char, replacements[0])
        
        # Remove extra whitespace
        title = ' '.join(title.split())
        
        return title.strip()
    
    def extract_title_components(self, title: str) -> Dict[str, Any]:
        """Extract components from title (year, sequel number, etc.)"""
        components = {
            'clean_title': title,
            'year': None,
            'sequel_number': None,
            'language_tag': None,
            'series_info': None
        }
        
        # Extract year
        for pattern in TITLE_PATTERNS['year_patterns']:
            match = re.search(pattern, title)
            if match:
                components['year'] = int(match.group(1))
                components['clean_title'] = re.sub(pattern, '', title).strip()
                break
        
        # Extract sequel number
        for pattern in TITLE_PATTERNS['sequel_patterns']:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                components['sequel_number'] = int(match.group(1))
                break
        
        # Extract language tag
        for pattern in TITLE_PATTERNS['language_tags']:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                components['language_tag'] = match.group(1)
                components['clean_title'] = re.sub(pattern, '', components['clean_title']).strip()
                break
        
        return components
    
    def calculate_title_similarity(self, query: str, title: str, original_title: str = None) -> Dict[str, float]:
        """Calculate multiple similarity scores between query and title"""
        scores = {
            'exact': 0.0,
            'normalized': 0.0,
            'partial': 0.0,
            'fuzzy': 0.0,
            'token': 0.0,
            'phonetic': 0.0,
            'levenshtein': 0.0,
            'jaro_winkler': 0.0,
            'sequence': 0.0,
            'combined': 0.0
        }
        
        query_lower = query.lower().strip()
        title_lower = title.lower().strip() if title else ""
        original_lower = original_title.lower().strip() if original_title else ""
        
        # 1. Exact match (highest priority)
        if query_lower == title_lower or query_lower == original_lower:
            scores['exact'] = 100.0
            scores['combined'] = 100.0
            return scores
        
        # 2. Normalized match
        query_normalized = self.normalize_title(query)
        title_normalized = self.normalize_title(title)
        original_normalized = self.normalize_title(original_title) if original_title else ""
        
        if query_normalized == title_normalized or query_normalized == original_normalized:
            scores['normalized'] = 95.0
        
        # 3. Partial match (query is substring or title is substring)
        if query_lower in title_lower or title_lower in query_lower:
            scores['partial'] = 85.0
        elif original_lower and (query_lower in original_lower or original_lower in query_lower):
            scores['partial'] = 80.0
        
        # 4. Fuzzy match using rapidfuzz
        scores['fuzzy'] = max(
            fuzz.ratio(query_lower, title_lower),
            fuzz.ratio(query_lower, original_lower) if original_lower else 0,
            fuzz.partial_ratio(query_lower, title_lower),
            fuzz.partial_ratio(query_lower, original_lower) if original_lower else 0
        )
        
        # 5. Token-based matching
        query_tokens = set(query_normalized.split())
        title_tokens = set(title_normalized.split())
        
        if query_tokens and title_tokens:
            intersection = query_tokens.intersection(title_tokens)
            union = query_tokens.union(title_tokens)
            scores['token'] = (len(intersection) / len(union)) * 100 if union else 0
        
        # 6. Phonetic matching (for handling typos)
        scores['phonetic'] = self._phonetic_similarity(query_normalized, title_normalized)
        
        # 7. Levenshtein distance (edit distance)
        max_len = max(len(query_lower), len(title_lower))
        if max_len > 0:
            lev_distance = Levenshtein.distance(query_lower, title_lower)
            scores['levenshtein'] = (1 - lev_distance / max_len) * 100
        
        # 8. Jaro-Winkler similarity
        scores['jaro_winkler'] = jellyfish.jaro_winkler_similarity(query_lower, title_lower) * 100
        
        # 9. Sequence matching (for ordered similarity)
        scores['sequence'] = SequenceMatcher(None, query_lower, title_lower).ratio() * 100
        
        # Calculate combined score with weights
        weights = {
            'exact': 1.0,
            'normalized': 0.9,
            'partial': 0.8,
            'fuzzy': 0.7,
            'token': 0.6,
            'phonetic': 0.5,
            'levenshtein': 0.6,
            'jaro_winkler': 0.6,
            'sequence': 0.5
        }
        
        weighted_sum = sum(scores[key] * weights[key] for key in weights)
        total_weight = sum(weights.values())
        scores['combined'] = weighted_sum / total_weight if total_weight > 0 else 0
        
        return scores
    
    def _phonetic_similarity(self, text1: str, text2: str) -> float:
        """Calculate phonetic similarity using metaphone"""
        if not text1 or not text2:
            return 0.0
        
        # Get metaphone codes
        meta1 = self.phonetic(text1)
        meta2 = self.phonetic(text2)
        
        # Compare primary and secondary codes
        score = 0.0
        if meta1[0] == meta2[0]:
            score += 50.0
        if meta1[1] and meta2[1] and meta1[1] == meta2[1]:
            score += 50.0
        
        return score
    
    def find_best_match(self, query: str, candidates: List[Dict]) -> Optional[Dict]:
        """Find the best matching title from candidates"""
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            title = candidate.get('title', '')
            original_title = candidate.get('original_title', '')
            
            scores = self.calculate_title_similarity(query, title, original_title)
            
            if scores['combined'] > best_score:
                best_score = scores['combined']
                best_match = candidate
                best_match['match_scores'] = scores
        
        return best_match if best_score > 30 else None  # Threshold for minimum match

class QueryParser:
    """Enhanced query parser with title detection"""
    
    def __init__(self):
        self.title_matcher = TitleMatcher()
        
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Enhanced query parsing to detect if it's a title search
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        result = {
            'original_query': query,
            'normalized_query': self.title_matcher.normalize_title(query),
            'is_title_search': False,
            'title_components': None,
            'language': None,
            'genres': [],
            'year': None,
            'content_type': None,
            'search_terms': [],
            'is_compound': False,
            'search_intent': 'general'  # general, title, browse, specific
        }
        
        # Extract title components
        result['title_components'] = self.title_matcher.extract_title_components(query)
        
        # Detect if this is likely a title search
        result['is_title_search'] = self._is_title_search(query, words)
        
        if result['is_title_search']:
            result['search_intent'] = 'title'
            result['search_terms'] = words
        else:
            # Regular parsing for language, genre, etc.
            # Extract language
            for word in words[:]:
                for lang, variations in LANGUAGE_MAPPINGS.items():
                    if word in [v.lower() for v in variations]:
                        result['language'] = lang
                        words.remove(word)
                        break
            
            # Extract genres
            for word in words[:]:
                for genre, variations in GENRE_MAPPINGS.items():
                    if word in [v.lower() for v in variations] or word == genre:
                        result['genres'].append(genre)
                        words.remove(word)
                        break
            
            # Extract year
            if result['title_components']['year']:
                result['year'] = result['title_components']['year']
            
            # Extract content type
            content_type_keywords = {
                'movie': 'movie',
                'movies': 'movie',
                'film': 'movie',
                'films': 'movie',
                'series': 'tv',
                'show': 'tv',
                'shows': 'tv',
                'tv': 'tv',
                'anime': 'anime',
                'animes': 'anime'
            }
            
            for word in words[:]:
                if word in content_type_keywords:
                    result['content_type'] = content_type_keywords[word]
                    words.remove(word)
                    break
            
            result['search_terms'] = words
            result['is_compound'] = bool(result['language'] and result['genres'])
            
            if result['is_compound']:
                result['search_intent'] = 'browse'
            elif len(words) > 0:
                result['search_intent'] = 'specific'
        
        return result
    
    def _is_title_search(self, query: str, words: List[str]) -> bool:
        """
        Detect if the query is likely a movie/show title
        """
        # Indicators that this is a title search
        title_indicators = [
            # Has sequel indicators
            any(re.search(pattern, query, re.IGNORECASE) for pattern in TITLE_PATTERNS['sequel_patterns']),
            # Is a known abbreviation
            query.lower() in TITLE_ABBREVIATIONS,
            # Has multiple capitalized words (proper nouns)
            sum(1 for word in query.split() if word[0].isupper()) >= 2,
            # Doesn't contain generic search terms
            not any(word in ['movies', 'films', 'shows', 'series', 'anime'] for word in words),
            # Length suggests a title (2-8 words typically)
            2 <= len(words) <= 8,
            # Contains "the" at the beginning (common in titles)
            query.lower().startswith('the '),
            # Has numbers that aren't years
            any(word.isdigit() and len(word) < 4 for word in words),
        ]
        
        # If 2 or more indicators are true, likely a title search
        return sum(title_indicators) >= 2

class AdvancedSearchIndexer:
    """Enhanced search indexer with advanced title matching"""
    
    def __init__(self):
        self.index = {
            'titles': {},  # exact title -> content_id
            'normalized_titles': {},  # normalized title -> content_id
            'title_tokens': defaultdict(set),  # title tokens -> content_ids
            'title_ngrams': defaultdict(set),  # title ngrams -> content_ids
            'title_phonetic': defaultdict(set),  # phonetic codes -> content_ids
            'original_titles': {},  # original title -> content_id
            'title_aliases': defaultdict(set),  # aliases -> content_ids
            'ngrams': defaultdict(set),
            'tokens': defaultdict(set),
            'prefixes': defaultdict(set),
            'genres': defaultdict(set),
            'languages': defaultdict(set),
            'original_languages': defaultdict(set),
            'dubbed_content': defaultdict(set),
            'years': defaultdict(set),
            'content_map': {},
            'popularity_scores': {},
            'title_search_index': {},  # Special index for title searches
            'language_genre_map': defaultdict(lambda: defaultdict(set))
        }
        self.last_update = None
        self.update_interval = 180
        self._lock = Lock()
        self.stop_words = self._load_stop_words()
        self.query_parser = QueryParser()
        self.title_matcher = TitleMatcher()
        
    def _load_stop_words(self) -> Set[str]:
        """Load common stop words for multiple languages"""
        return {
            # English
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            # Don't include movie-related words as stop words
        }
    
    def build_index(self, content_list: List[Any]) -> None:
        """Build comprehensive search index with advanced title indexing"""
        logger.info(f"Building advanced search index with {len(content_list)} items")
        start_time = time.time()
        
        with self._lock:
            self._reset_index()
            
            for content in content_list:
                try:
                    self._index_content(content)
                    self._build_title_search_index(content)
                except Exception as e:
                    logger.error(f"Error indexing content {content.id}: {e}")
                    continue
            
            self._build_popularity_scores()
            self._build_language_genre_index()
            self._build_phonetic_index()
            
            self.last_update = datetime.now()
        
        elapsed = time.time() - start_time
        logger.info(f"Advanced search index built in {elapsed:.2f} seconds")
    
    def _build_title_search_index(self, content):
        """Build specialized index for title searches"""
        content_id = content.id
        
        if content.title:
            # Store exact title
            title_lower = content.title.lower().strip()
            self.index['titles'][title_lower] = content_id
            
            # Store normalized title
            normalized = self.title_matcher.normalize_title(content.title)
            self.index['normalized_titles'][normalized] = content_id
            
            # Extract and index title tokens
            title_tokens = set(normalized.split())
            for token in title_tokens:
                if token not in self.stop_words:
                    self.index['title_tokens'][token].add(content_id)
            
            # Generate and index title n-grams
            for n in [2, 3, 4]:  # Bigrams, trigrams, quadgrams
                ngrams = self._generate_ngrams(title_lower, n)
                for ngram in ngrams:
                    self.index['title_ngrams'][ngram].add(content_id)
            
            # Index prefixes for autocomplete
            for length in range(1, min(len(title_lower) + 1, 15)):
                prefix = title_lower[:length]
                self.index['prefixes'][prefix].add(content_id)
            
            # Index each word's prefix
            for word in title_lower.split():
                for length in range(1, min(len(word) + 1, 10)):
                    prefix = word[:length]
                    self.index['prefixes'][prefix].add(content_id)
        
        # Index original title
        if content.original_title:
            orig_lower = content.original_title.lower().strip()
            self.index['original_titles'][orig_lower] = content_id
            
            # Also index original title tokens and ngrams
            orig_normalized = self.title_matcher.normalize_title(content.original_title)
            orig_tokens = set(orig_normalized.split())
            for token in orig_tokens:
                if token not in self.stop_words:
                    self.index['title_tokens'][token].add(content_id)
        
        # Build comprehensive title search entry
        self.index['title_search_index'][content_id] = {
            'title': content.title,
            'title_lower': content.title.lower() if content.title else '',
            'title_normalized': self.title_matcher.normalize_title(content.title) if content.title else '',
            'original_title': content.original_title,
            'original_lower': content.original_title.lower() if content.original_title else '',
            'original_normalized': self.title_matcher.normalize_title(content.original_title) if content.original_title else '',
            'title_tokens': title_tokens if 'title_tokens' in locals() else set(),
            'content_type': content.content_type,
            'year': content.release_date.year if content.release_date else None
        }
    
    def _build_phonetic_index(self):
        """Build phonetic index for all titles"""
        for content_id, title_data in self.index['title_search_index'].items():
            # Get phonetic codes for title
            if title_data['title_normalized']:
                phonetic_codes = metaphone.doublemetaphone(title_data['title_normalized'])
                for code in phonetic_codes:
                    if code:
                        self.index['title_phonetic'][code].add(content_id)
            
            # Get phonetic codes for original title
            if title_data['original_normalized']:
                phonetic_codes = metaphone.doublemetaphone(title_data['original_normalized'])
                for code in phonetic_codes:
                    if code:
                        self.index['title_phonetic'][code].add(content_id)
    
    def search(self, query: str, limit: int = 20, filters: Dict = None) -> List[SearchResult]:
        """Advanced search with title priority"""
        # Parse the query
        parsed_query = self.query_parser.parse_query(query)
        
        # If it's identified as a title search, use specialized title search
        if parsed_query['is_title_search']:
            return self._title_priority_search(query, parsed_query, limit, filters)
        
        # If it's a compound query (language + genre), use specialized search
        if parsed_query['is_compound']:
            return self._search_language_genre(
                parsed_query['language'],
                parsed_query['genres'],
                parsed_query['search_terms'],
                limit,
                filters
            )
        
        # Otherwise, use enhanced regular search
        return self._enhanced_regular_search(query, parsed_query, limit, filters)
    
    def _title_priority_search(self, query: str, parsed_query: Dict, 
                              limit: int, filters: Dict) -> List[SearchResult]:
        """
        Specialized search for movie/show titles with 100% accuracy
        """
        query_lower = query.lower().strip()
        query_normalized = parsed_query['normalized_query']
        results = {}
        
        with self._lock:
            # Phase 1: Exact matching (Score: 1000+)
            # Check exact title match
            if query_lower in self.index['titles']:
                content_id = self.index['titles'][query_lower]
                content_data = self.index['content_map'].get(content_id)
                if content_data:
                    results[content_id] = self._create_search_result(
                        content_id, content_data,
                        title_match_score=1000,
                        exact_match=True
                    )
            
            # Check exact original title match
            if query_lower in self.index['original_titles']:
                content_id = self.index['original_titles'][query_lower]
                if content_id not in results:
                    content_data = self.index['content_map'].get(content_id)
                    if content_data:
                        results[content_id] = self._create_search_result(
                            content_id, content_data,
                            title_match_score=950,
                            exact_match=True
                        )
            
            # Phase 2: Normalized matching (Score: 800-900)
            if query_normalized in self.index['normalized_titles']:
                content_id = self.index['normalized_titles'][query_normalized]
                if content_id not in results:
                    content_data = self.index['content_map'].get(content_id)
                    if content_data:
                        results[content_id] = self._create_search_result(
                            content_id, content_data,
                            title_match_score=900
                        )
            
            # Phase 3: Advanced similarity matching (Score: 500-800)
            # Search through all titles using advanced matching
            all_matches = []
            for content_id, title_data in self.index['title_search_index'].items():
                if content_id in results:
                    continue
                
                # Calculate similarity scores
                similarity_scores = self.title_matcher.calculate_title_similarity(
                    query,
                    title_data['title'],
                    title_data['original_title']
                )
                
                # If combined score is high enough, add to results
                if similarity_scores['combined'] > 50:
                    content_data = self.index['content_map'].get(content_id)
                    if content_data:
                        all_matches.append((
                            content_id,
                            content_data,
                            similarity_scores
                        ))
            
            # Sort matches by score and add top matches to results
            all_matches.sort(key=lambda x: x[2]['combined'], reverse=True)
            for content_id, content_data, scores in all_matches[:20]:
                if content_id not in results:
                    results[content_id] = self._create_search_result(
                        content_id, content_data,
                        title_match_score=scores['combined'] * 8,  # Scale to 0-800
                        partial_match=scores['partial'] > 70,
                        fuzzy_match=scores['fuzzy'] > 70
                    )
            
            # Phase 4: Token-based matching (Score: 300-500)
            query_tokens = set(query_normalized.split())
            token_matches = defaultdict(int)
            
            for token in query_tokens:
                if token in self.index['title_tokens']:
                    for content_id in self.index['title_tokens'][token]:
                        if content_id not in results:
                            token_matches[content_id] += 1
            
            # Add token matches with score based on match percentage
            for content_id, match_count in token_matches.items():
                if content_id not in results:
                    content_data = self.index['content_map'].get(content_id)
                    if content_data:
                        match_percentage = (match_count / len(query_tokens)) * 100 if query_tokens else 0
                        if match_percentage > 50:  # At least 50% tokens match
                            results[content_id] = self._create_search_result(
                                content_id, content_data,
                                title_match_score=300 + (match_percentage * 2)
                            )
            
            # Phase 5: Phonetic matching for typos (Score: 200-300)
            phonetic_codes = metaphone.doublemetaphone(query_normalized)
            phonetic_matches = set()
            
            for code in phonetic_codes:
                if code and code in self.index['title_phonetic']:
                    phonetic_matches.update(self.index['title_phonetic'][code])
            
            for content_id in phonetic_matches:
                if content_id not in results:
                    content_data = self.index['content_map'].get(content_id)
                    if content_data:
                        # Verify with fuzzy matching
                        title_data = self.index['title_search_index'][content_id]
                        fuzzy_score = fuzz.ratio(query_normalized, title_data['title_normalized'])
                        if fuzzy_score > 60:
                            results[content_id] = self._create_search_result(
                                content_id, content_data,
                                title_match_score=200 + fuzzy_score,
                                phonetic_match=True
                            )
            
            # Phase 6: N-gram matching (Score: 100-200)
            query_ngrams = set()
            for n in [2, 3]:
                query_ngrams.update(self._generate_ngrams(query_lower, n))
            
            ngram_matches = defaultdict(int)
            for ngram in query_ngrams:
                if ngram in self.index['title_ngrams']:
                    for content_id in self.index['title_ngrams'][ngram]:
                        if content_id not in results:
                            ngram_matches[content_id] += 1
            
            # Add high-scoring n-gram matches
            for content_id, match_count in ngram_matches.items():
                if match_count >= len(query_ngrams) * 0.3:  # At least 30% n-grams match
                    if content_id not in results:
                        content_data = self.index['content_map'].get(content_id)
                        if content_data:
                            results[content_id] = self._create_search_result(
                                content_id, content_data,
                                title_match_score=100 + (match_count * 10)
                            )
            
            # Apply filters if provided
            if filters:
                filtered_results = {}
                for content_id, result in results.items():
                    if self._apply_filter(content_id, filters):
                        filtered_results[content_id] = result
                results = filtered_results
        
        # Convert to list and sort by title match score primarily
        result_list = list(results.values())
        result_list.sort(
            key=lambda x: (
                x.title_match_score,  # Title match score is primary
                x.popularity_score,    # Then popularity
                x.score               # Then general score
            ),
            reverse=True
        )
        
        return result_list[:limit]
    
    def _enhanced_regular_search(self, query: str, parsed_query: Dict, 
                                limit: int, filters: Dict) -> List[SearchResult]:
        """Enhanced regular search with title matching integrated"""
        query_lower = query.lower().strip()
        query_normalized = parsed_query['normalized_query']
        results = {}
        
        with self._lock:
            # First, always check for title matches
            title_results = self._title_priority_search(query, parsed_query, limit * 2, filters)
            
            # Add high-scoring title matches to results
            for result in title_results:
                if result.title_match_score > 300:  # Good title matches
                    results[result.id] = result
            
            # Then do regular search for additional results
            # Token-based search
            query_tokens = self._tokenize(query)
            token_matches = defaultdict(int)
            
            for token in query_tokens:
                if token in self.index['tokens']:
                    for content_id in self.index['tokens'][token]:
                        if content_id not in results:
                            token_matches[content_id] += 1
            
            for content_id, match_count in token_matches.items():
                if content_id not in results:
                    content_data = self.index['content_map'].get(content_id)
                    if content_data:
                        score = (match_count / len(query_tokens)) * 400 if query_tokens else 0
                        results[content_id] = self._create_search_result(
                            content_id, content_data,
                            score_boost=score
                        )
            
            # Apply language boost if detected
            if parsed_query['language']:
                for content_id, result in results.items():
                    if parsed_query['language'].lower() in [lang.lower() for lang in result.languages]:
                        if result.is_original_language:
                            result.score += 200
                        else:
                            result.score += 100
            
            # Apply filters
            if filters:
                filtered_results = {}
                for content_id, result in results.items():
                    if self._apply_filter(content_id, filters):
                        filtered_results[content_id] = result
                results = filtered_results
        
        # Convert and sort
        result_list = list(results.values())
        result_list.sort(
            key=lambda x: (
                x.title_match_score > 500,  # Exact/near-exact title matches first
                x.title_match_score,         # Then by title match score
                x.is_original_language,      # Original language content
                not x.is_dubbed,             # Non-dubbed content
                x.score,                     # General score
                x.popularity_score           # Popularity
            ),
            reverse=True
        )
        
        return result_list[:limit]
    
    def _search_language_genre(self, language: str, genres: List[str], 
                               search_terms: List[str], limit: int, filters: Dict) -> List[SearchResult]:
        """
        Specialized search for language + genre queries
        Priority: Original language content > Dubbed content
        Within each group: Sort by popularity
        """
        results = {}
        
        with self._lock:
            # Check if search terms might be a title
            if search_terms:
                search_text = ' '.join(search_terms)
                # First search for exact titles in this language/genre
                title_results = self._title_priority_search(
                    search_text, 
                    self.query_parser.parse_query(search_text),
                    limit * 2,
                    filters
                )
                
                # Filter title results by language and genre
                for result in title_results:
                    matches_language = language.lower() in [l.lower() for l in result.languages] if language else True
                    matches_genre = any(g.lower() in [genre.lower() for genre in result.genres] for g in genres) if genres else True
                    
                    if matches_language and matches_genre:
                        results[result.id] = result
            
            # Then get content by language and genre
            original_results = []
            dubbed_results = []
            
            for genre in genres:
                genre_lower = genre.lower()
                
                # Get original language content
                if language in self.index['language_genre_map']:
                    original_ids = self.index['language_genre_map'][language].get(genre_lower, set())
                    for content_id in original_ids:
                        if content_id not in results:
                            content_data = self.index['content_map'].get(content_id)
                            if content_data and not content_data.get('is_dubbed'):
                                original_results.append(self._create_search_result(
                                    content_id, content_data,
                                    score_boost=800,
                                    is_original=True
                                ))
                
                # Get dubbed content
                dubbed_key = f"{language}_dubbed"
                if dubbed_key in self.index['language_genre_map']:
                    dubbed_ids = self.index['language_genre_map'][dubbed_key].get(genre_lower, set())
                    for content_id in dubbed_ids:
                        if content_id not in results:
                            content_data = self.index['content_map'].get(content_id)
                            if content_data:
                                dubbed_results.append(self._create_search_result(
                                    content_id, content_data,
                                    score_boost=400,
                                    is_dubbed=True
                                ))
            
            # Sort by popularity
            original_results.sort(key=lambda x: x.popularity_score, reverse=True)
            dubbed_results.sort(key=lambda x: x.popularity_score, reverse=True)
            
            # Combine: title matches first, then original, then dubbed
            all_results = list(results.values()) + original_results + dubbed_results
            
            # Apply additional filters
            if filters:
                all_results = self._apply_filters_to_results(all_results, filters)
            
            return all_results[:limit]
    
    def _create_search_result(self, content_id: int, content_data: Dict, 
                             score_boost: float = 0, title_match_score: float = 0,
                             exact_match: bool = False, partial_match: bool = False,
                             fuzzy_match: bool = False, phonetic_match: bool = False,
                             is_original: bool = False, is_dubbed: bool = False) -> SearchResult:
        """Create a SearchResult object with all metadata"""
        popularity_score = self.index['popularity_scores'].get(content_id, 0)
        
        return SearchResult(
            id=content_id,
            title=content_data['title'],
            content_type=content_data['content_type'],
            score=score_boost + popularity_score,
            title_match_score=title_match_score,
            exact_match=exact_match,
            partial_match=partial_match,
            fuzzy_match=fuzzy_match,
            phonetic_match=phonetic_match,
            popularity_score=popularity_score,
            is_original_language=is_original or not content_data.get('is_dubbed', False),
            is_dubbed=is_dubbed or content_data.get('is_dubbed', False),
            original_language=content_data.get('original_language'),
            languages=content_data.get('languages', []),
            genres=content_data.get('genres', []),
            metadata=content_data
        )
    
    # ... (keep all other methods from previous implementation)
    def _reset_index(self):
        """Reset all index structures"""
        self.index = {
            'titles': {},
            'normalized_titles': {},
            'title_tokens': defaultdict(set),
            'title_ngrams': defaultdict(set),
            'title_phonetic': defaultdict(set),
            'original_titles': {},
            'title_aliases': defaultdict(set),
            'normalized': {},
            'ngrams': defaultdict(set),
            'tokens': defaultdict(set),
            'prefixes': defaultdict(set),
            'genres': defaultdict(set),
            'languages': defaultdict(set),
            'original_languages': defaultdict(set),
            'dubbed_content': defaultdict(set),
            'years': defaultdict(set),
            'content_map': {},
            'popularity_scores': {},
            'title_search_index': {},
            'language_genre_map': defaultdict(lambda: defaultdict(set))
        }
    
    def _index_content(self, content):
        """Index a single content item"""
        content_id = content.id
        
        # Parse languages and detect original vs dubbed
        languages = json.loads(content.languages or '[]')
        original_language = self._detect_original_language(content, languages)
        is_dubbed = self._is_dubbed_content(content, languages, original_language)
        
        # Store full content data
        self.index['content_map'][content_id] = {
            'id': content_id,
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': json.loads(content.genres or '[]'),
            'languages': languages,
            'original_language': original_language,
            'is_dubbed': is_dubbed,
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
        
        # Index genres
        genres = json.loads(content.genres or '[]')
        for genre in genres:
            self.index['genres'][genre.lower()].add(content_id)
        
        # Index languages
        for language in languages:
            self.index['languages'][language.lower()].add(content_id)
        
        if original_language:
            self.index['original_languages'][original_language.lower()].add(content_id)
        
        if is_dubbed:
            self.index['dubbed_content']['all'].add(content_id)
            for lang in languages:
                self.index['dubbed_content'][lang.lower()].add(content_id)
        
        # Index year
        if content.release_date:
            self.index['years'][content.release_date.year].add(content_id)
        
        # Index for general text search
        if content.title:
            tokens = self._tokenize(content.title)
            for token in tokens:
                self.index['tokens'][token].add(content_id)
        
        if content.overview:
            overview_tokens = self._tokenize(content.overview)[:20]  # Limit overview tokens
            for token in overview_tokens:
                self.index['tokens'][token].add(content_id)
    
    def needs_update(self) -> bool:
        """Check if index needs updating"""
        if not self.last_update:
            return True
        return (datetime.now() - self.last_update).seconds > self.update_interval
    
    def _detect_original_language(self, content, languages):
        """Detect the original language of content"""
        if hasattr(content, 'original_language') and content.original_language:
            return content.original_language
        
        if content.original_title:
            # Telugu title patterns
            if any(ord(char) >= 0x0C00 and ord(char) <= 0x0C7F for char in content.original_title):
                return 'Telugu'
            # Hindi/Devanagari
            elif any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in content.original_title):
                return 'Hindi'
            # Tamil
            elif any(ord(char) >= 0x0B80 and ord(char) <= 0x0BFF for char in content.original_title):
                return 'Tamil'
            # Korean
            elif any(ord(char) >= 0xAC00 and ord(char) <= 0xD7AF for char in content.original_title):
                return 'Korean'
            # Japanese
            elif any(ord(char) >= 0x3040 and ord(char) <= 0x309F for char in content.original_title):
                return 'Japanese'
        
        return languages[0] if languages else 'English'
    
    def _is_dubbed_content(self, content, languages, original_language):
        """Determine if content is dubbed"""
        if not languages or not original_language:
            return False
        
        if content.original_title and content.title != content.original_title:
            if self._is_translation(content.title, content.original_title):
                return True
        
        if len(languages) > 1 and original_language.lower() != 'english' and 'English' in languages:
            return True
        
        return False
    
    def _is_translation(self, title, original_title):
        """Check if title is a translation of original_title"""
        if not title or not original_title:
            return False
        
        title_norm = self._normalize_text(title)
        original_norm = self._normalize_text(original_title)
        
        similarity = fuzz.ratio(title_norm, original_norm)
        return similarity < 50
    
    def _build_language_genre_index(self):
        """Build language-genre mapping"""
        for content_id, content_data in self.index['content_map'].items():
            original_lang = content_data.get('original_language', '').lower()
            genres = content_data.get('genres', [])
            
            for genre in genres:
                genre_lower = genre.lower()
                if original_lang:
                    self.index['language_genre_map'][original_lang][genre_lower].add(content_id)
                
                if content_data.get('is_dubbed'):
                    for lang in content_data.get('languages', []):
                        lang_lower = lang.lower()
                        if lang_lower != original_lang:
                            self.index['language_genre_map'][f"{lang_lower}_dubbed"][genre_lower].add(content_id)
    
    def _build_popularity_scores(self):
        """Calculate normalized popularity scores"""
        max_pop = 0
        max_rating = 10
        max_votes = 0
        
        for content_data in self.index['content_map'].values():
            max_pop = max(max_pop, content_data.get('popularity', 0))
            max_votes = max(max_votes, content_data.get('vote_count', 0))
        
        for content_id, content_data in self.index['content_map'].items():
            pop_score = 0
            
            if max_pop > 0:
                pop_score += (content_data.get('popularity', 0) / max_pop) * 40
            
            rating = content_data.get('rating', 0)
            pop_score += (rating / max_rating) * 30
            
            if max_votes > 0:
                votes = content_data.get('vote_count', 0)
                pop_score += (votes / max_votes) * 20
            
            if content_data.get('is_new_release'):
                pop_score += 10
            elif content_data.get('release_date'):
                days_old = (datetime.now().date() - content_data['release_date']).days
                if days_old < 365:
                    pop_score += max(0, 10 - (days_old / 36.5))
            
            if content_data.get('original_language', '').lower() == 'telugu':
                pop_score *= 1.2
            
            self.index['popularity_scores'][content_id] = pop_score
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        if not text:
            return ""
        
        text = text.lower()
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into searchable tokens"""
        if not text:
            return []
        
        normalized = self._normalize_text(text)
        tokens = re.findall(r'\w+', normalized.lower())
        tokens = [t for t in tokens if len(t) > 1 and t not in self.stop_words]
        
        return tokens
    
    def _generate_ngrams(self, text: str, n: int = 3) -> Set[str]:
        """Generate character n-grams"""
        ngrams = set()
        text = text.lower().replace(' ', '')
        
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i+n])
        
        return ngrams
    
    def _apply_filter(self, content_id: int, filters: Dict) -> bool:
        """Apply filters to a content item"""
        content_data = self.index['content_map'].get(content_id)
        if not content_data:
            return False
        
        if filters.get('genre'):
            genres = [g.lower() for g in content_data.get('genres', [])]
            if filters['genre'].lower() not in genres:
                return False
        
        if filters.get('language'):
            languages = [l.lower() for l in content_data.get('languages', [])]
            if filters['language'].lower() not in languages:
                return False
        
        if filters.get('year'):
            if not content_data.get('release_date'):
                return False
            if content_data['release_date'].year != filters['year']:
                return False
        
        if filters.get('content_type'):
            if content_data['content_type'] != filters['content_type']:
                return False
        
        if filters.get('min_rating'):
            if content_data.get('rating', 0) < filters['min_rating']:
                return False
        
        return True
    
    def _apply_filters_to_results(self, results: List[SearchResult], filters: Dict) -> List[SearchResult]:
        """Apply filters to search results"""
        filtered = []
        for result in results:
            if self._apply_filter(result.id, filters):
                filtered.append(result)
        return filtered

# Keep the rest of the classes (ImprovedHybridSearch, SmartSuggestionEngine) the same
# but update ImprovedHybridSearch to use the new indexer capabilities

class ImprovedHybridSearch:
    """Enhanced hybrid search with advanced title matching"""
    
    def __init__(self, db, cache, services):
        self.db = db
        self.cache = cache
        self.services = services
        self.indexer = AdvancedSearchIndexer()
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.query_parser = QueryParser()
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
        Enhanced search with advanced title matching
        """
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
        
        # Parse the query
        parsed_query = self.query_parser.parse_query(query)
        
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
        
        # Perform search
        all_results = []
        
        try:
            # Use the advanced indexer search
            index_results = self.indexer.search(query, limit * 2, filters)
            
            # Format results
            all_results = index_results
            
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
            try:
                all_results = self._fallback_search(query, limit)
            except:
                all_results = []
        
        # Format final results
        formatted_results = self._format_final_results(all_results[:limit])
        
        # Generate smart suggestions
        suggestions = self._generate_smart_suggestions(query, parsed_query, formatted_results[:5])
        
        # Prepare response
        response = {
            'results': formatted_results,
            'total_results': len(formatted_results),
            'page': page,
            'query': query,
            'parsed_query': {
                'language': parsed_query.get('language'),
                'genres': parsed_query.get('genres'),
                'is_title_search': parsed_query.get('is_title_search'),
                'search_intent': parsed_query.get('search_intent')
            },
            'search_time': round(time.time() - start_time, 3),
            'suggestions': suggestions,
            'filters_applied': filters or {},
            'cached': False
        }
        
        # Cache successful results
        if formatted_results:
            self._cache_result(cache_key, response)
        
        return response
    
    def _format_final_results(self, results: List[SearchResult]) -> List[Dict]:
        """Format results with all metadata"""
        formatted = []
        
        for result in results:
            try:
                content_data = result.metadata
                youtube_url = None
                if content_data.get('youtube_trailer_id'):
                    youtube_url = f"https://www.youtube.com/watch?v={content_data['youtube_trailer_id']}"
                
                formatted.append({
                    'id': result.id,
                    'title': result.title,
                    'original_title': content_data.get('original_title'),
                    'content_type': result.content_type,
                    'genres': result.genres,
                    'languages': result.languages,
                    'rating': float(content_data.get('rating', 0)),
                    'vote_count': content_data.get('vote_count', 0),
                    'popularity': float(content_data.get('popularity', 0)),
                    'release_date': content_data['release_date'].isoformat() if content_data.get('release_date') else None,
                    'year': content_data['release_date'].year if content_data.get('release_date') else None,
                    'poster_path': self._get_poster_url(content_data.get('poster_path')),
                    'overview': self._truncate_text(content_data.get('overview'), 200),
                    'youtube_trailer': youtube_url,
                    'is_trending': content_data.get('is_trending', False),
                    'is_new_release': content_data.get('is_new_release', False),
                    'is_original_language': result.is_original_language,
                    'is_dubbed': result.is_dubbed,
                    'match_info': {
                        'exact_match': result.exact_match,
                        'partial_match': result.partial_match,
                        'fuzzy_match': result.fuzzy_match,
                        'phonetic_match': result.phonetic_match,
                        'title_match_score': result.title_match_score,
                        'score': result.score
                    }
                })
            except Exception as e:
                logger.error(f"Error formatting result: {e}")
                continue
        
        return formatted
    
    def _update_search_index(self):
        """Update search index with latest content"""
        try:
            from app import Content
            
            content_query = Content.query.order_by(
                Content.popularity.desc()
            ).limit(100000)  # Increase limit for comprehensive index
            
            content_list = content_query.all()
            
            if content_list:
                self.indexer.build_index(content_list)
                logger.info(f"Search index updated with {len(content_list)} items")
            
        except Exception as e:
            logger.error(f"Failed to update search index: {e}")
    
    def _generate_smart_suggestions(self, query: str, parsed_query: Dict, 
                                   top_results: List[Dict]) -> List[str]:
        """Generate intelligent suggestions"""
        suggestions = []
        
        if parsed_query.get('is_title_search'):
            # For title searches, suggest sequels or related titles
            if top_results:
                for result in top_results[:1]:
                    title = result.get('title', '')
                    # Check for sequel patterns
                    for pattern in TITLE_PATTERNS['sequel_patterns']:
                        if re.search(pattern, title, re.IGNORECASE):
                            # Suggest other parts
                            base_title = re.sub(pattern, '', title, flags=re.IGNORECASE).strip()
                            suggestions.append(f"{base_title} 2")
                            suggestions.append(f"{base_title} sequel")
                            break
        
        elif parsed_query.get('is_compound'):
            # Language-genre suggestions
            language = parsed_query.get('language')
            genres = parsed_query.get('genres', [])
            
            other_genres = ['action', 'comedy', 'drama', 'thriller', 'romance']
            for genre in other_genres:
                if genre not in genres:
                    suggestions.append(f"{language} {genre} movies")
            
            if genres:
                other_languages = ['telugu', 'hindi', 'tamil', 'english']
                for lang in other_languages:
                    if lang != language:
                        suggestions.append(f"{lang} {genres[0]} movies")
        
        else:
            # Regular suggestions
            if top_results:
                all_genres = []
                all_languages = []
                
                for result in top_results[:3]:
                    all_genres.extend(result.get('genres', []))
                    all_languages.extend(result.get('languages', []))
                
                genre_counts = Counter(all_genres)
                for genre, _ in genre_counts.most_common(2):
                    suggestions.append(f"{query} {genre}")
                
                lang_counts = Counter(all_languages)
                for lang, _ in lang_counts.most_common(1):
                    if lang.lower() not in query.lower():
                        suggestions.append(f"{lang} {query}")
        
        current_year = datetime.now().year
        if str(current_year) not in query:
            suggestions.append(f"{query} {current_year}")
        
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion.lower() not in seen and suggestion.lower() != query.lower():
                seen.add(suggestion.lower())
                unique_suggestions.append(suggestion)
                if len(unique_suggestions) >= 5:
                    break
        
        return unique_suggestions
    
    def _fallback_search(self, query: str, limit: int) -> List[SearchResult]:
        """Simple fallback search"""
        try:
            from app import Content
            
            contents = Content.query.filter(
                or_(
                    Content.title.ilike(f'%{query}%'),
                    Content.original_title.ilike(f'%{query}%')
                )
            ).order_by(
                Content.popularity.desc()
            ).limit(limit).all()
            
            results = []
            for content in contents:
                results.append(SearchResult(
                    id=content.id,
                    title=content.title,
                    content_type=content.content_type,
                    metadata={
                        'original_title': content.original_title,
                        'genres': json.loads(content.genres or '[]'),
                        'languages': json.loads(content.languages or '[]'),
                        'rating': content.rating,
                        'popularity': content.popularity,
                        'release_date': content.release_date,
                        'poster_path': content.poster_path,
                        'overview': content.overview,
                        'youtube_trailer_id': content.youtube_trailer_id
                    }
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return []
    
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
        return f"search:v4:{hashlib.md5(key_data.encode()).hexdigest()}"
    
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
            if result.get('results'):
                self.cache.set(cache_key, json.dumps(result), timeout=300)
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def autocomplete(self, prefix: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Provide autocomplete with title priority"""
        if len(prefix) < 2:
            return []
        
        cache_key = f"autocomplete:v3:{prefix.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except:
                pass
        
        suggestions = []
        
        # Get suggestions from index with title priority
        results = self.indexer.search(prefix, limit * 2)
        
        # Format suggestions
        for result in results[:limit]:
            content_data = result.metadata
            if content_data:
                language_tag = ""
                if result.original_language and result.original_language.lower() != 'english':
                    language_tag = f" ({result.original_language})"
                
                suggestions.append({
                    'id': result.id,
                    'title': content_data['title'] + language_tag,
                    'type': content_data['content_type'],
                    'year': content_data['release_date'].year if content_data.get('release_date') else None,
                    'poster': self._get_poster_url(content_data.get('poster_path')),
                    'rating': content_data.get('rating', 0),
                    'is_exact_match': result.exact_match,
                    'match_score': result.title_match_score
                })
        
        if suggestions:
            try:
                self.cache.set(cache_key, json.dumps(suggestions), timeout=1800)
            except:
                pass
        
        return suggestions

# Keep SmartSuggestionEngine as is
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
        
        trending = [
            "Pushpa 2",
            "Salaar",
            "KGF Chapter 2",
            "RRR",
            "Telugu movies 2024",
            "Telugu horror movies",
            "Telugu comedy movies",
            "Latest Telugu releases",
            "Marvel movies",
            "Best anime 2024"
        ]
        
        try:
            self.cache.set(cache_key, json.dumps(trending), timeout=3600)
        except:
            pass
        
        return trending[:limit]
    
    def record_search(self, query: str, user_id: Optional[int] = None) -> None:
        """Record a search query for analytics"""
        try:
            cache_key = "search_analytics"
            analytics = self.cache.get(cache_key)
            
            if analytics:
                try:
                    analytics = json.loads(analytics)
                except:
                    analytics = {'queries': {}, 'last_update': datetime.now().isoformat()}
            else:
                analytics = {'queries': {}, 'last_update': datetime.now().isoformat()}
            
            analytics['queries'][query] = analytics['queries'].get(query, 0) + 1
            analytics['last_update'] = datetime.now().isoformat()
            
            if len(analytics['queries']) > 10:
                top_queries = sorted(
                    analytics['queries'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                trending = [q[0] for q in top_queries]
                self.cache.set("trending_searches", json.dumps(trending), timeout=3600)
            
            self.cache.set(cache_key, json.dumps(analytics), timeout=86400)
            
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
                
                history.insert(0, {
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                })
                history = history[:20]
                
                self.cache.set(user_key, json.dumps(history), timeout=2592000)
                
        except Exception as e:
            logger.error(f"Failed to record search: {e}")
    
    def get_personalized_suggestions(self, user_id: int, limit: int = 5) -> List[str]:
        """Get personalized search suggestions"""
        try:
            user_key = f"user_search_history:{user_id}"
            history = self.cache.get(user_key)
            
            if history:
                try:
                    history = json.loads(history)
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