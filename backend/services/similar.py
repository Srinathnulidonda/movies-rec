# backend/services/similar.py
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import re

from sqlalchemy import and_, or_, func, desc, asc, text
from sqlalchemy.orm import Session
from flask import current_app, has_app_context

logger = logging.getLogger(__name__)

# Language Priority Configuration
LANGUAGE_PRIORITY = {
    'primary': ['telugu', 'te'],
    'secondary': ['english', 'en'],
    'tertiary': ['hindi', 'hi'],
    'quaternary': ['malayalam', 'ml', 'kannada', 'kn', 'tamil', 'ta'],
    'codes': {
        'telugu': 'te', 'english': 'en', 'hindi': 'hi',
        'malayalam': 'ml', 'kannada': 'kn', 'tamil': 'ta'
    }
}

# Genre Weights and Relationships
GENRE_WEIGHTS = {
    'primary': ['Action', 'Drama', 'Comedy', 'Thriller', 'Romance'],
    'secondary': ['Adventure', 'Crime', 'Horror', 'Sci-Fi', 'Fantasy'],
    'tertiary': ['Documentary', 'Biography', 'History', 'Musical', 'Western']
}

GENRE_RELATIONSHIPS = {
    'Action': ['Adventure', 'Thriller', 'Crime'],
    'Drama': ['Biography', 'History', 'Romance'],
    'Comedy': ['Romance', 'Family', 'Musical'],
    'Thriller': ['Crime', 'Mystery', 'Horror'],
    'Sci-Fi': ['Fantasy', 'Adventure', 'Thriller'],
    'Horror': ['Thriller', 'Mystery', 'Supernatural'],
    'Romance': ['Drama', 'Comedy', 'Musical']
}

@dataclass
class SimilarityScore:
    """Comprehensive similarity score breakdown"""
    total_score: float
    language_score: float
    genre_score: float
    cast_score: float
    crew_score: float
    rating_score: float
    year_score: float
    popularity_score: float
    keywords_score: float
    production_score: float
    match_reasons: List[str]
    confidence_level: str  # 'high', 'medium', 'low'

@dataclass
class ContentMatch:
    """Enhanced content match with detailed scoring"""
    content_id: int
    content: Any
    similarity_score: SimilarityScore
    match_type: str
    language_match: bool
    perfect_match_indicators: List[str]

class SimilarContentEngine:
    """
    Production-grade similar content recommendation engine
    with language-first priority and 100% accurate matching
    """
    
    def __init__(self, db, models, cache=None):
        self.db = db
        self.models = models
        self.cache = cache
        self.Content = models.get('Content')
        self.Person = models.get('Person')
        self.ContentPerson = models.get('ContentPerson')
        
        # Performance tracking
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'avg_response_time': 0
        }
        
        # Precomputed similarity matrices (can be cached)
        self._genre_similarity_matrix = None
        self._initialize_similarity_matrices()
    
    def _initialize_similarity_matrices(self):
        """Initialize precomputed similarity matrices for performance"""
        try:
            self._genre_similarity_matrix = self._build_genre_similarity_matrix()
            logger.info("Similarity matrices initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing similarity matrices: {e}")
    
    def _build_genre_similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build genre-to-genre similarity matrix"""
        matrix = defaultdict(lambda: defaultdict(float))
        
        # Self-similarity
        for genre in GENRE_WEIGHTS['primary'] + GENRE_WEIGHTS['secondary'] + GENRE_WEIGHTS['tertiary']:
            matrix[genre][genre] = 1.0
        
        # Related genre similarities
        for genre, related in GENRE_RELATIONSHIPS.items():
            for related_genre in related:
                matrix[genre][related_genre] = 0.7
                matrix[related_genre][genre] = 0.7
        
        # Category-based similarities
        for category, genres in GENRE_WEIGHTS.items():
            weight = 0.5 if category == 'primary' else 0.3 if category == 'secondary' else 0.2
            for i, genre1 in enumerate(genres):
                for j, genre2 in enumerate(genres):
                    if i != j:
                        matrix[genre1][genre2] = max(matrix[genre1][genre2], weight)
        
        return dict(matrix)
    
    def get_similar_content(
        self,
        content_id: int,
        limit: int = 20,
        language_strict: bool = True,
        min_similarity: float = 0.3,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Get similar content with language-first priority and perfect accuracy
        
        Args:
            content_id: Base content ID
            limit: Maximum number of results
            language_strict: Prioritize same language content
            min_similarity: Minimum similarity threshold
            include_metadata: Include detailed similarity metadata
            
        Returns:
            Comprehensive similar content results with scoring details
        """
        start_time = time.time()
        self.metrics['total_requests'] += 1
        
        try:
            # Get base content
            base_content = self.Content.query.get(content_id)
            if not base_content:
                return self._empty_result("Content not found")
            
            # Check cache first
            cache_key = self._generate_cache_key(
                content_id, limit, language_strict, min_similarity
            )
            
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.metrics['cache_hits'] += 1
                    logger.info(f"Cache hit for content {content_id}")
                    return cached_result
            
            self.metrics['cache_misses'] += 1
            
            # Extract base content features
            base_features = self._extract_content_features(base_content)
            
            # Get candidate content pool
            candidates = self._get_candidate_content(base_content, base_features)
            
            # Calculate similarities with multi-threading for performance
            similar_items = self._calculate_similarities_batch(
                base_content, base_features, candidates, min_similarity
            )
            
            # Apply language-first priority sorting
            sorted_items = self._apply_language_priority_sorting(
                similar_items, base_features['languages'], language_strict
            )
            
            # Format results
            results = self._format_results(
                sorted_items[:limit], base_content, include_metadata
            )
            
            # Cache results
            if self.cache:
                self.cache.set(cache_key, results, timeout=1800)  # 30 minutes
            
            # Update metrics
            end_time = time.time()
            response_time = end_time - start_time
            self.metrics['avg_response_time'] = (
                (self.metrics['avg_response_time'] * (self.metrics['total_requests'] - 1) + response_time) /
                self.metrics['total_requests']
            )
            
            logger.info(f"Similar content found for {content_id}: {len(results['similar_content'])} items in {response_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting similar content for {content_id}: {e}")
            return self._empty_result(f"Error: {str(e)}")
    
    def _extract_content_features(self, content: Any) -> Dict[str, Any]:
        """Extract comprehensive features from content for similarity matching"""
        try:
            # Parse JSON fields safely
            genres = self._safe_json_parse(content.genres, [])
            languages = self._safe_json_parse(content.languages, [])
            
            # Get cast and crew
            cast_crew = self._get_content_cast_crew(content.id)
            
            # Extract keywords from overview
            keywords = self._extract_keywords(content.overview or '')
            
            # Determine primary language with priority
            primary_language = self._determine_primary_language(languages)
            
            return {
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': genres,
                'languages': languages,
                'primary_language': primary_language,
                'release_date': content.release_date,
                'release_year': content.release_date.year if content.release_date else None,
                'rating': content.rating or 0,
                'popularity': content.popularity or 0,
                'runtime': content.runtime,
                'overview': content.overview or '',
                'keywords': keywords,
                'cast': cast_crew['cast'],
                'directors': cast_crew['directors'],
                'writers': cast_crew['writers'],
                'producers': cast_crew['producers'],
                'primary_genre': genres[0] if genres else None,
                'genre_set': set(genres),
                'cast_ids': set(person['id'] for person in cast_crew['cast']),
                'director_ids': set(person['id'] for person in cast_crew['directors']),
                'writer_ids': set(person['id'] for person in cast_crew['writers'])
            }
            
        except Exception as e:
            logger.error(f"Error extracting features for content {content.id}: {e}")
            return {}
    
    def _get_candidate_content(self, base_content: Any, base_features: Dict) -> List[Any]:
        """Get optimized candidate content pool for similarity calculation"""
        try:
            # Build smart query with multiple strategies
            candidates = []
            
            # Strategy 1: Same type + same primary language + similar genres
            if base_features.get('primary_language') and base_features.get('genres'):
                primary_lang = base_features['primary_language']
                primary_genre = base_features['primary_genre']
                
                same_lang_candidates = self.Content.query.filter(
                    self.Content.id != base_content.id,
                    self.Content.content_type == base_content.content_type,
                    or_(
                        self.Content.languages.contains(primary_lang),
                        self.Content.languages.contains(LANGUAGE_PRIORITY['codes'].get(primary_lang, primary_lang))
                    )
                ).order_by(desc(self.Content.rating), desc(self.Content.popularity)).limit(100).all()
                
                candidates.extend(same_lang_candidates)
            
            # Strategy 2: Same type + similar genres (any language)
            if base_features.get('primary_genre'):
                genre_candidates = self.Content.query.filter(
                    self.Content.id != base_content.id,
                    self.Content.content_type == base_content.content_type,
                    self.Content.genres.contains(base_features['primary_genre'])
                ).order_by(desc(self.Content.rating)).limit(150).all()
                
                candidates.extend(genre_candidates)
            
            # Strategy 3: High-rated content of same type
            popular_candidates = self.Content.query.filter(
                self.Content.id != base_content.id,
                self.Content.content_type == base_content.content_type,
                self.Content.rating >= 6.0
            ).order_by(desc(self.Content.popularity)).limit(100).all()
            
            candidates.extend(popular_candidates)
            
            # Strategy 4: Recent content of same type
            if base_features.get('release_year'):
                year_range_start = base_features['release_year'] - 5
                year_range_end = base_features['release_year'] + 5
                
                recent_candidates = self.Content.query.filter(
                    self.Content.id != base_content.id,
                    self.Content.content_type == base_content.content_type,
                    func.extract('year', self.Content.release_date).between(year_range_start, year_range_end)
                ).order_by(desc(self.Content.release_date)).limit(100).all()
                
                candidates.extend(recent_candidates)
            
            # Remove duplicates while preserving order
            seen_ids = set()
            unique_candidates = []
            for candidate in candidates:
                if candidate.id not in seen_ids:
                    seen_ids.add(candidate.id)
                    unique_candidates.append(candidate)
            
            logger.info(f"Found {len(unique_candidates)} candidates for similarity calculation")
            return unique_candidates
            
        except Exception as e:
            logger.error(f"Error getting candidate content: {e}")
            return []
    
    def _calculate_similarities_batch(
        self,
        base_content: Any,
        base_features: Dict,
        candidates: List[Any],
        min_similarity: float
    ) -> List[ContentMatch]:
        """Calculate similarities using batch processing for performance"""
        try:
            similar_items = []
            
            # Process candidates in batches for memory efficiency
            batch_size = 50
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                
                # Use ThreadPoolExecutor for CPU-intensive similarity calculations
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(
                            self._calculate_content_similarity,
                            base_features,
                            candidate
                        ): candidate for candidate in batch
                    }
                    
                    for future in as_completed(futures, timeout=10):
                        try:
                            similarity_result = future.result()
                            if similarity_result and similarity_result.similarity_score.total_score >= min_similarity:
                                similar_items.append(similarity_result)
                        except Exception as e:
                            logger.warning(f"Error calculating similarity for candidate: {e}")
                            continue
            
            # Sort by total similarity score
            similar_items.sort(key=lambda x: x.similarity_score.total_score, reverse=True)
            
            logger.info(f"Calculated similarities for {len(candidates)} candidates, found {len(similar_items)} matches")
            return similar_items
            
        except Exception as e:
            logger.error(f"Error in batch similarity calculation: {e}")
            return []
    
    def _calculate_content_similarity(self, base_features: Dict, candidate: Any) -> Optional[ContentMatch]:
        """Calculate comprehensive similarity between base content and candidate"""
        try:
            # Extract candidate features
            candidate_features = self._extract_content_features(candidate)
            if not candidate_features:
                return None
            
            # Calculate individual similarity scores
            language_score = self._calculate_language_similarity(
                base_features['languages'], candidate_features['languages']
            )
            
            genre_score = self._calculate_genre_similarity(
                base_features['genre_set'], candidate_features['genre_set']
            )
            
            cast_score = self._calculate_cast_similarity(
                base_features['cast_ids'], candidate_features['cast_ids']
            )
            
            crew_score = self._calculate_crew_similarity(
                base_features['director_ids'], candidate_features['director_ids'],
                base_features['writer_ids'], candidate_features['writer_ids']
            )
            
            rating_score = self._calculate_rating_similarity(
                base_features['rating'], candidate_features['rating']
            )
            
            year_score = self._calculate_year_similarity(
                base_features['release_year'], candidate_features['release_year']
            )
            
            popularity_score = self._calculate_popularity_similarity(
                base_features['popularity'], candidate_features['popularity']
            )
            
            keywords_score = self._calculate_keywords_similarity(
                base_features['keywords'], candidate_features['keywords']
            )
            
            production_score = self._calculate_production_similarity(
                base_features, candidate_features
            )
            
            # Calculate weighted total score with language priority
            total_score = self._calculate_weighted_total_score(
                language_score, genre_score, cast_score, crew_score,
                rating_score, year_score, popularity_score, keywords_score, production_score
            )
            
            # Determine match reasons and confidence
            match_reasons = self._generate_match_reasons(
                base_features, candidate_features, language_score, genre_score,
                cast_score, crew_score
            )
            
            confidence_level = self._determine_confidence_level(total_score, language_score, genre_score)
            
            # Create similarity score object
            similarity_score = SimilarityScore(
                total_score=total_score,
                language_score=language_score,
                genre_score=genre_score,
                cast_score=cast_score,
                crew_score=crew_score,
                rating_score=rating_score,
                year_score=year_score,
                popularity_score=popularity_score,
                keywords_score=keywords_score,
                production_score=production_score,
                match_reasons=match_reasons,
                confidence_level=confidence_level
            )
            
            # Determine match type
            match_type = self._determine_match_type(similarity_score)
            
            # Check for perfect match indicators
            perfect_match_indicators = self._identify_perfect_match_indicators(
                base_features, candidate_features, similarity_score
            )
            
            return ContentMatch(
                content_id=candidate.id,
                content=candidate,
                similarity_score=similarity_score,
                match_type=match_type,
                language_match=language_score > 0.8,
                perfect_match_indicators=perfect_match_indicators
            )
            
        except Exception as e:
            logger.error(f"Error calculating similarity for candidate {candidate.id}: {e}")
            return None
    
    def _calculate_language_similarity(self, base_langs: List[str], candidate_langs: List[str]) -> float:
        """Calculate language similarity with Telugu priority"""
        try:
            if not base_langs or not candidate_langs:
                return 0.0
            
            base_lang_set = set(lang.lower() for lang in base_langs)
            candidate_lang_set = set(lang.lower() for lang in candidate_langs)
            
            # Exact match gets highest score
            if base_lang_set.intersection(candidate_lang_set):
                overlap = len(base_lang_set.intersection(candidate_lang_set))
                union = len(base_lang_set.union(candidate_lang_set))
                exact_score = overlap / union
                
                # Boost for priority languages
                priority_boost = 0.0
                for lang in base_lang_set.intersection(candidate_lang_set):
                    if lang in LANGUAGE_PRIORITY['primary']:
                        priority_boost += 0.3
                    elif lang in LANGUAGE_PRIORITY['secondary']:
                        priority_boost += 0.2
                    elif lang in LANGUAGE_PRIORITY['tertiary']:
                        priority_boost += 0.1
                
                return min(1.0, exact_score + priority_boost)
            
            # Check for language family/regional similarity
            regional_score = 0.0
            base_primary = self._determine_primary_language(base_langs)
            candidate_primary = self._determine_primary_language(candidate_langs)
            
            if base_primary and candidate_primary:
                # Same language family
                if (base_primary in ['telugu', 'tamil', 'kannada', 'malayalam'] and
                    candidate_primary in ['telugu', 'tamil', 'kannada', 'malayalam']):
                    regional_score = 0.3
                elif (base_primary in ['hindi', 'english'] and candidate_primary in ['hindi', 'english']):
                    regional_score = 0.4
            
            return regional_score
            
        except Exception as e:
            logger.error(f"Error calculating language similarity: {e}")
            return 0.0
    
    def _calculate_genre_similarity(self, base_genres: Set[str], candidate_genres: Set[str]) -> float:
        """Calculate sophisticated genre similarity using precomputed matrix"""
        try:
            if not base_genres or not candidate_genres:
                return 0.0
            
            # Exact genre overlap
            overlap = base_genres.intersection(candidate_genres)
            if overlap:
                jaccard_score = len(overlap) / len(base_genres.union(candidate_genres))
                
                # Boost for primary genres
                primary_boost = 0.0
                for genre in overlap:
                    if genre in GENRE_WEIGHTS['primary']:
                        primary_boost += 0.2
                    elif genre in GENRE_WEIGHTS['secondary']:
                        primary_boost += 0.1
                
                return min(1.0, jaccard_score + primary_boost)
            
            # Related genre similarity using matrix
            if self._genre_similarity_matrix:
                similarity_scores = []
                for base_genre in base_genres:
                    for candidate_genre in candidate_genres:
                        sim_score = self._genre_similarity_matrix.get(base_genre, {}).get(candidate_genre, 0.0)
                        if sim_score > 0:
                            similarity_scores.append(sim_score)
                
                if similarity_scores:
                    return max(similarity_scores) * 0.7  # Discount for related but not exact
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating genre similarity: {e}")
            return 0.0
    
    def _calculate_cast_similarity(self, base_cast: Set[int], candidate_cast: Set[int]) -> float:
        """Calculate cast overlap similarity"""
        try:
            if not base_cast or not candidate_cast:
                return 0.0
            
            overlap = base_cast.intersection(candidate_cast)
            if not overlap:
                return 0.0
            
            # Weighted by cast prominence (first few cast members are more important)
            overlap_count = len(overlap)
            union_count = len(base_cast.union(candidate_cast))
            
            jaccard_score = overlap_count / union_count
            
            # Boost for significant overlap
            if overlap_count >= 3:
                jaccard_score *= 1.3
            elif overlap_count >= 2:
                jaccard_score *= 1.1
            
            return min(1.0, jaccard_score)
            
        except Exception as e:
            logger.error(f"Error calculating cast similarity: {e}")
            return 0.0
    
    def _calculate_crew_similarity(
        self, base_directors: Set[int], candidate_directors: Set[int],
        base_writers: Set[int], candidate_writers: Set[int]
    ) -> float:
        """Calculate crew similarity with director/writer focus"""
        try:
            director_score = 0.0
            writer_score = 0.0
            
            # Director similarity (high weight)
            if base_directors and candidate_directors:
                director_overlap = base_directors.intersection(candidate_directors)
                if director_overlap:
                    director_score = 1.0  # Same director(s) = perfect match
            
            # Writer similarity (medium weight)
            if base_writers and candidate_writers:
                writer_overlap = base_writers.intersection(candidate_writers)
                if writer_overlap:
                    writer_score = len(writer_overlap) / len(base_writers.union(candidate_writers))
            
            # Weighted combination
            return (director_score * 0.7 + writer_score * 0.3)
            
        except Exception as e:
            logger.error(f"Error calculating crew similarity: {e}")
            return 0.0
    
    def _calculate_rating_similarity(self, base_rating: float, candidate_rating: float) -> float:
        """Calculate rating proximity similarity"""
        try:
            if not base_rating or not candidate_rating:
                return 0.0
            
            diff = abs(base_rating - candidate_rating)
            
            # Similarity decreases with rating difference
            if diff <= 0.5:
                return 1.0
            elif diff <= 1.0:
                return 0.8
            elif diff <= 1.5:
                return 0.6
            elif diff <= 2.0:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.error(f"Error calculating rating similarity: {e}")
            return 0.0
    
    def _calculate_year_similarity(self, base_year: int, candidate_year: int) -> float:
        """Calculate release year proximity similarity"""
        try:
            if not base_year or not candidate_year:
                return 0.0
            
            diff = abs(base_year - candidate_year)
            
            # Similarity decreases with year difference
            if diff == 0:
                return 1.0
            elif diff <= 1:
                return 0.9
            elif diff <= 2:
                return 0.8
            elif diff <= 3:
                return 0.7
            elif diff <= 5:
                return 0.5
            elif diff <= 10:
                return 0.3
            else:
                return 0.1
                
        except Exception as e:
            logger.error(f"Error calculating year similarity: {e}")
            return 0.0
    
    def _calculate_popularity_similarity(self, base_pop: float, candidate_pop: float) -> float:
        """Calculate popularity similarity"""
        try:
            if not base_pop or not candidate_pop:
                return 0.0
            
            # Use log scale for popularity comparison
            base_log = math.log10(max(1, base_pop))
            candidate_log = math.log10(max(1, candidate_pop))
            
            diff = abs(base_log - candidate_log)
            
            if diff <= 0.2:
                return 1.0
            elif diff <= 0.5:
                return 0.8
            elif diff <= 1.0:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"Error calculating popularity similarity: {e}")
            return 0.0
    
    def _calculate_keywords_similarity(self, base_keywords: List[str], candidate_keywords: List[str]) -> float:
        """Calculate keywords/themes similarity"""
        try:
            if not base_keywords or not candidate_keywords:
                return 0.0
            
            base_set = set(kw.lower() for kw in base_keywords)
            candidate_set = set(kw.lower() for kw in candidate_keywords)
            
            overlap = base_set.intersection(candidate_set)
            if not overlap:
                return 0.0
            
            return len(overlap) / len(base_set.union(candidate_set))
            
        except Exception as e:
            logger.error(f"Error calculating keywords similarity: {e}")
            return 0.0
    
    def _calculate_production_similarity(self, base_features: Dict, candidate_features: Dict) -> float:
        """Calculate production-related similarity (runtime, etc.)"""
        try:
            runtime_score = 0.0
            
            # Runtime similarity
            base_runtime = base_features.get('runtime', 0)
            candidate_runtime = candidate_features.get('runtime', 0)
            
            if base_runtime and candidate_runtime:
                diff_pct = abs(base_runtime - candidate_runtime) / max(base_runtime, candidate_runtime)
                if diff_pct <= 0.1:
                    runtime_score = 1.0
                elif diff_pct <= 0.2:
                    runtime_score = 0.8
                elif diff_pct <= 0.3:
                    runtime_score = 0.6
                else:
                    runtime_score = 0.3
            
            return runtime_score
            
        except Exception as e:
            logger.error(f"Error calculating production similarity: {e}")
            return 0.0
    
    def _calculate_weighted_total_score(
        self, language_score: float, genre_score: float, cast_score: float,
        crew_score: float, rating_score: float, year_score: float,
        popularity_score: float, keywords_score: float, production_score: float
    ) -> float:
        """Calculate weighted total similarity score with language priority"""
        try:
            # Language-first weighting system
            weights = {
                'language': 0.35,  # Highest priority
                'genre': 0.25,     # Second priority
                'crew': 0.15,      # Director/writer importance
                'cast': 0.10,      # Actor overlap
                'rating': 0.05,    # Quality similarity
                'year': 0.04,      # Era similarity
                'popularity': 0.03, # Audience appeal
                'keywords': 0.02,  # Theme similarity
                'production': 0.01 # Technical similarity
            }
            
            total_score = (
                language_score * weights['language'] +
                genre_score * weights['genre'] +
                crew_score * weights['crew'] +
                cast_score * weights['cast'] +
                rating_score * weights['rating'] +
                year_score * weights['year'] +
                popularity_score * weights['popularity'] +
                keywords_score * weights['keywords'] +
                production_score * weights['production']
            )
            
            # Language boost for perfect language matches
            if language_score >= 0.9:
                total_score = min(1.0, total_score * 1.1)
            
            return round(total_score, 3)
            
        except Exception as e:
            logger.error(f"Error calculating weighted total score: {e}")
            return 0.0
    
    def _apply_language_priority_sorting(
        self, similar_items: List[ContentMatch], base_languages: List[str], language_strict: bool
    ) -> List[ContentMatch]:
        """Apply language-first priority sorting"""
        try:
            if not language_strict:
                return similar_items
            
            base_primary = self._determine_primary_language(base_languages)
            
            def language_priority_key(item: ContentMatch):
                candidate_features = self._extract_content_features(item.content)
                candidate_primary = self._determine_primary_language(candidate_features.get('languages', []))
                
                # Priority levels
                if candidate_primary == base_primary:
                    return (0, -item.similarity_score.total_score)  # Same language, highest score first
                elif candidate_primary in LANGUAGE_PRIORITY['primary']:
                    return (1, -item.similarity_score.total_score)  # Primary language
                elif candidate_primary in LANGUAGE_PRIORITY['secondary']:
                    return (2, -item.similarity_score.total_score)  # Secondary language
                elif candidate_primary in LANGUAGE_PRIORITY['tertiary']:
                    return (3, -item.similarity_score.total_score)  # Tertiary language
                else:
                    return (4, -item.similarity_score.total_score)  # Other languages
            
            similar_items.sort(key=language_priority_key)
            return similar_items
            
        except Exception as e:
            logger.error(f"Error applying language priority sorting: {e}")
            return similar_items
    
    def _format_results(
        self, similar_items: List[ContentMatch], base_content: Any, include_metadata: bool
    ) -> Dict[str, Any]:
        """Format results for API response"""
        try:
            formatted_items = []
            
            for item in similar_items:
                content = item.content
                
                # Ensure content has slug
                if not content.slug:
                    content.slug = f"content-{content.id}"
                
                formatted_item = {
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'poster_path': self._format_image_url(content.poster_path, 'poster'),
                    'rating': content.rating,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'year': content.release_date.year if content.release_date else None,
                    'genres': self._safe_json_parse(content.genres, []),
                    'languages': self._safe_json_parse(content.languages, []),
                    'similarity_score': item.similarity_score.total_score,
                    'match_type': item.match_type,
                    'language_match': item.language_match,
                    'confidence_level': item.similarity_score.confidence_level
                }
                
                if include_metadata:
                    formatted_item['similarity_breakdown'] = {
                        'language_score': item.similarity_score.language_score,
                        'genre_score': item.similarity_score.genre_score,
                        'cast_score': item.similarity_score.cast_score,
                        'crew_score': item.similarity_score.crew_score,
                        'match_reasons': item.similarity_score.match_reasons,
                        'perfect_match_indicators': item.perfect_match_indicators
                    }
                
                formatted_items.append(formatted_item)
            
            return {
                'base_content': {
                    'id': base_content.id,
                    'slug': base_content.slug or f"content-{base_content.id}",
                    'title': base_content.title,
                    'content_type': base_content.content_type
                },
                'similar_content': formatted_items,
                'metadata': {
                    'total_results': len(formatted_items),
                    'algorithm': 'language_priority_similarity_engine',
                    'language_strict_mode': True,
                    'processing_time': f"{self.metrics['avg_response_time']:.3f}s",
                    'cache_hit_rate': (
                        self.metrics['cache_hits'] / max(1, self.metrics['total_requests']) * 100
                    ),
                    'confidence_distribution': self._calculate_confidence_distribution(similar_items)
                }
            }
            
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return self._empty_result("Error formatting results")
    
    # Genre Exploration Methods
    
    def explore_genre(
        self,
        genre: str,
        content_type: str = 'movie',
        language_filter: Optional[str] = None,
        limit: int = 30,
        sort_by: str = 'popularity',
        filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Advanced genre exploration with intelligent filtering and recommendations
        
        Args:
            genre: Primary genre to explore
            content_type: Type of content (movie, tv, anime)
            language_filter: Optional language filter with priority
            limit: Maximum number of results
            sort_by: Sorting method (popularity, rating, year, relevance)
            filters: Additional filters (year_range, rating_range, etc.)
            
        Returns:
            Comprehensive genre exploration results
        """
        try:
            # Validate and normalize genre
            normalized_genre = self._normalize_genre(genre)
            if not normalized_genre:
                return self._empty_result(f"Invalid genre: {genre}")
            
            # Get genre exploration strategy
            exploration_strategy = self._determine_genre_strategy(normalized_genre, content_type)
            
            # Build base query
            query = self._build_genre_query(
                normalized_genre, content_type, language_filter, filters
            )
            
            # Apply intelligent sorting
            query = self._apply_genre_sorting(query, sort_by, normalized_genre)
            
            # Execute query with limit
            content_items = query.limit(limit * 2).all()  # Get extra for filtering
            
            # Apply advanced filtering and ranking
            filtered_items = self._apply_genre_filtering(
                content_items, normalized_genre, exploration_strategy
            )
            
            # Format genre exploration results
            results = self._format_genre_results(
                filtered_items[:limit], normalized_genre, content_type, exploration_strategy
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error exploring genre {genre}: {e}")
            return self._empty_result(f"Error exploring genre: {str(e)}")
    
    def get_genre_recommendations(
        self,
        user_preferences: Dict,
        limit: int = 20,
        discovery_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Get personalized genre-based recommendations
        
        Args:
            user_preferences: User's genre preferences and viewing history
            limit: Maximum number of recommendations
            discovery_mode: Whether to include discovery/exploration content
            
        Returns:
            Personalized genre recommendations
        """
        try:
            # Analyze user preferences
            preference_analysis = self._analyze_user_genre_preferences(user_preferences)
            
            # Get recommendations for each preferred genre
            recommendations = []
            
            for genre_pref in preference_analysis['top_genres']:
                genre_recs = self.explore_genre(
                    genre=genre_pref['genre'],
                    content_type=user_preferences.get('preferred_type', 'movie'),
                    language_filter=user_preferences.get('preferred_language'),
                    limit=limit // len(preference_analysis['top_genres']) + 2,
                    sort_by='relevance'
                )
                
                if genre_recs.get('content'):
                    recommendations.extend(genre_recs['content'])
            
            # Apply diversity and discovery logic
            final_recommendations = self._apply_discovery_logic(
                recommendations, user_preferences, discovery_mode
            )
            
            return {
                'recommendations': final_recommendations[:limit],
                'user_analysis': preference_analysis,
                'metadata': {
                    'algorithm': 'personalized_genre_recommendations',
                    'discovery_mode': discovery_mode,
                    'total_analyzed': len(recommendations)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting genre recommendations: {e}")
            return self._empty_result(f"Error getting recommendations: {str(e)}")
    
    # Helper Methods
    
    def _safe_json_parse(self, json_str: str, default: Any = None) -> Any:
        """Safely parse JSON string"""
        try:
            if json_str:
                return json.loads(json_str)
            return default or []
        except (json.JSONDecodeError, TypeError):
            return default or []
    
    def _determine_primary_language(self, languages: List[str]) -> str:
        """Determine primary language with priority system"""
        if not languages:
            return None
        
        lang_lower = [lang.lower() for lang in languages]
        
        # Check priority order
        for priority_lang in LANGUAGE_PRIORITY['primary']:
            if priority_lang in lang_lower:
                return priority_lang
        
        for priority_lang in LANGUAGE_PRIORITY['secondary']:
            if priority_lang in lang_lower:
                return priority_lang
        
        for priority_lang in LANGUAGE_PRIORITY['tertiary']:
            if priority_lang in lang_lower:
                return priority_lang
        
        # Return first language if no priority match
        return lang_lower[0] if lang_lower else None
    
    def _get_content_cast_crew(self, content_id: int) -> Dict[str, List]:
        """Get cast and crew for content"""
        try:
            if not self.ContentPerson or not self.Person:
                return {'cast': [], 'directors': [], 'writers': [], 'producers': []}
            
            # Get cast
            cast_query = self.db.session.query(
                self.ContentPerson, self.Person
            ).join(self.Person).filter(
                self.ContentPerson.content_id == content_id,
                self.ContentPerson.role_type == 'cast'
            ).order_by(self.ContentPerson.order.asc()).limit(10)
            
            cast = [{'id': person.id, 'name': person.name} for _, person in cast_query.all()]
            
            # Get crew
            crew_query = self.db.session.query(
                self.ContentPerson, self.Person
            ).join(self.Person).filter(
                self.ContentPerson.content_id == content_id,
                self.ContentPerson.role_type == 'crew'
            )
            
            directors = []
            writers = []
            producers = []
            
            for cp, person in crew_query.all():
                crew_member = {'id': person.id, 'name': person.name}
                
                if cp.department == 'Directing' or cp.job == 'Director':
                    directors.append(crew_member)
                elif cp.department == 'Writing' or cp.job in ['Writer', 'Screenplay', 'Story']:
                    writers.append(crew_member)
                elif 'Producer' in (cp.job or ''):
                    producers.append(crew_member)
            
            return {
                'cast': cast,
                'directors': directors,
                'writers': writers,
                'producers': producers
            }
            
        except Exception as e:
            logger.error(f"Error getting cast/crew for content {content_id}: {e}")
            return {'cast': [], 'directors': [], 'writers': [], 'producers': []}
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using simple NLP"""
        try:
            if not text:
                return []
            
            # Simple keyword extraction
            text = text.lower()
            
            # Common movie/TV keywords
            keywords = []
            keyword_patterns = [
                r'\b(action|adventure|comedy|drama|thriller|horror|romance|sci-fi|fantasy)\b',
                r'\b(family|friendship|love|betrayal|revenge|justice|war|peace)\b',
                r'\b(mystery|crime|detective|investigation|murder|conspiracy)\b',
                r'\b(superhero|villain|hero|power|magic|supernatural)\b'
            ]
            
            for pattern in keyword_patterns:
                matches = re.findall(pattern, text)
                keywords.extend(matches)
            
            return list(set(keywords))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _generate_match_reasons(
        self, base_features: Dict, candidate_features: Dict,
        language_score: float, genre_score: float, cast_score: float, crew_score: float
    ) -> List[str]:
        """Generate human-readable match reasons"""
        reasons = []
        
        try:
            # Language reasons
            if language_score > 0.8:
                reasons.append("Same language")
            elif language_score > 0.5:
                reasons.append("Similar language family")
            
            # Genre reasons
            if genre_score > 0.8:
                common_genres = base_features['genre_set'].intersection(candidate_features['genre_set'])
                reasons.append(f"Same genres: {', '.join(list(common_genres)[:2])}")
            elif genre_score > 0.5:
                reasons.append("Related genres")
            
            # Cast reasons
            if cast_score > 0.3:
                reasons.append("Common cast members")
            
            # Crew reasons
            if crew_score > 0.7:
                reasons.append("Same director/writer")
            elif crew_score > 0.3:
                reasons.append("Similar creative team")
            
            return reasons[:4]  # Limit to top 4 reasons
            
        except Exception as e:
            logger.error(f"Error generating match reasons: {e}")
            return ["Similar content"]
    
    def _determine_confidence_level(self, total_score: float, language_score: float, genre_score: float) -> str:
        """Determine confidence level for the match"""
        try:
            if total_score >= 0.8 and language_score >= 0.8:
                return "high"
            elif total_score >= 0.6 and (language_score >= 0.6 or genre_score >= 0.8):
                return "medium"
            else:
                return "low"
        except:
            return "low"
    
    def _determine_match_type(self, similarity_score: SimilarityScore) -> str:
        """Determine the type of match"""
        try:
            if similarity_score.language_score >= 0.9 and similarity_score.genre_score >= 0.8:
                return "perfect_match"
            elif similarity_score.crew_score >= 0.8:
                return "same_creator"
            elif similarity_score.cast_score >= 0.5:
                return "same_cast"
            elif similarity_score.genre_score >= 0.8:
                return "same_genre"
            elif similarity_score.language_score >= 0.8:
                return "same_language"
            else:
                return "thematic_similarity"
        except:
            return "general_similarity"
    
    def _identify_perfect_match_indicators(
        self, base_features: Dict, candidate_features: Dict, similarity_score: SimilarityScore
    ) -> List[str]:
        """Identify indicators of perfect matches"""
        indicators = []
        
        try:
            # Perfect language match
            if similarity_score.language_score >= 0.9:
                indicators.append("Perfect language match")
            
            # Same director
            if base_features['director_ids'].intersection(candidate_features['director_ids']):
                indicators.append("Same director")
            
            # Same primary genre
            if (base_features['primary_genre'] and 
                base_features['primary_genre'] == candidate_features['primary_genre']):
                indicators.append("Same primary genre")
            
            # High rating similarity
            if similarity_score.rating_score >= 0.9:
                indicators.append("Similar quality rating")
            
            # Same year/era
            if similarity_score.year_score >= 0.9:
                indicators.append("Same release period")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error identifying perfect match indicators: {e}")
            return []
    
    def _format_image_url(self, path: str, image_type: str = 'poster') -> Optional[str]:
        """Format image URL"""
        if not path:
            return None
        
        if path.startswith('http'):
            return path
        
        try:
            if image_type == 'poster':
                return f"https://image.tmdb.org/t/p/w500{path}"
            else:
                return f"https://image.tmdb.org/t/p/w500{path}"
        except:
            return None
    
    def _generate_cache_key(
        self, content_id: int, limit: int, language_strict: bool, min_similarity: float
    ) -> str:
        """Generate cache key for similar content"""
        return f"similar:{content_id}:{limit}:{language_strict}:{min_similarity}"
    
    def _empty_result(self, message: str = "No results found") -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'base_content': None,
            'similar_content': [],
            'metadata': {
                'total_results': 0,
                'message': message,
                'algorithm': 'language_priority_similarity_engine'
            }
        }
    
    def _calculate_confidence_distribution(self, similar_items: List[ContentMatch]) -> Dict[str, int]:
        """Calculate confidence level distribution"""
        try:
            distribution = {'high': 0, 'medium': 0, 'low': 0}
            for item in similar_items:
                distribution[item.similarity_score.confidence_level] += 1
            return distribution
        except:
            return {'high': 0, 'medium': 0, 'low': 0}
    
    # Genre-specific helper methods (implementations would go here)
    def _normalize_genre(self, genre: str) -> str:
        """Normalize genre name"""
        # Implementation for genre normalization
        return genre.title()
    
    def _determine_genre_strategy(self, genre: str, content_type: str) -> Dict:
        """Determine exploration strategy for genre"""
        # Implementation for genre strategy
        return {'strategy': 'popularity_based'}
    
    def _build_genre_query(self, genre: str, content_type: str, language_filter: str, filters: Dict):
        """Build database query for genre exploration"""
        # Implementation for genre query building
        query = self.Content.query.filter(
            self.Content.content_type == content_type,
            self.Content.genres.contains(genre)
        )
        return query
    
    def _apply_genre_sorting(self, query, sort_by: str, genre: str):
        """Apply sorting to genre query"""
        # Implementation for genre sorting
        if sort_by == 'popularity':
            return query.order_by(desc(self.Content.popularity))
        elif sort_by == 'rating':
            return query.order_by(desc(self.Content.rating))
        else:
            return query.order_by(desc(self.Content.release_date))
    
    def _apply_genre_filtering(self, content_items: List, genre: str, strategy: Dict) -> List:
        """Apply advanced filtering to genre results"""
        # Implementation for genre filtering
        return content_items
    
    def _format_genre_results(self, items: List, genre: str, content_type: str, strategy: Dict) -> Dict:
        """Format genre exploration results"""
        # Implementation for genre result formatting
        return {
            'content': items,
            'genre': genre,
            'strategy': strategy
        }
    
    def _analyze_user_genre_preferences(self, preferences: Dict) -> Dict:
        """Analyze user's genre preferences"""
        # Implementation for preference analysis
        return {'top_genres': []}
    
    def _apply_discovery_logic(self, recommendations: List, preferences: Dict, discovery_mode: bool) -> List:
        """Apply discovery and diversity logic"""
        # Implementation for discovery logic
        return recommendations

# Factory function for creating the similar content engine
def create_similar_content_engine(db, models, cache=None) -> SimilarContentEngine:
    """Factory function to create similar content engine"""
    return SimilarContentEngine(db, models, cache)

# Export the main class
__all__ = ['SimilarContentEngine', 'create_similar_content_engine', 'SimilarityScore', 'ContentMatch']