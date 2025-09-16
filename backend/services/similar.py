# backend/services/similar.py

import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import math
import re

from sqlalchemy import and_, or_, func, text
from sqlalchemy.orm import sessionmaker, joinedload
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

logger = logging.getLogger(__name__)

class SimilarityType(Enum):
    """Types of similarity calculations"""
    LANGUAGE_EXACT = "language_exact"
    GENRE_BASED = "genre_based"
    CAST_SIMILARITY = "cast_similarity"
    DIRECTOR_SIMILARITY = "director_similarity"
    CONTENT_BASED = "content_based"
    COLLABORATIVE = "collaborative"
    HYBRID = "hybrid"
    TEMPORAL_PROXIMITY = "temporal_proximity"
    RATING_SIMILARITY = "rating_similarity"
    PLOT_SIMILARITY = "plot_similarity"

class LanguagePriority(Enum):
    """Language priority levels"""
    EXACT_MATCH = 100
    SAME_FAMILY = 80
    REGIONAL = 60
    INTERNATIONAL = 40
    FALLBACK = 20

@dataclass
class SimilarityScore:
    """Similarity score with detailed breakdown"""
    content_id: int
    total_score: float
    language_score: float = 0.0
    genre_score: float = 0.0
    cast_score: float = 0.0
    director_score: float = 0.0
    plot_score: float = 0.0
    rating_score: float = 0.0
    temporal_score: float = 0.0
    algorithm_used: str = ""
    match_reasons: List[str] = field(default_factory=list)
    confidence_level: float = 0.0
    
    def __post_init__(self):
        """Calculate confidence level based on component scores"""
        scores = [
            self.language_score, self.genre_score, self.cast_score,
            self.director_score, self.plot_score, self.rating_score
        ]
        non_zero_scores = [s for s in scores if s > 0]
        if non_zero_scores:
            self.confidence_level = (sum(non_zero_scores) / len(non_zero_scores)) * 100

@dataclass
class ContentFeatures:
    """Extracted features for similarity calculation"""
    content_id: int
    languages: Set[str]
    genres: Set[str]
    cast_ids: Set[int]
    crew_ids: Set[int]
    director_ids: Set[int]
    plot_vector: Optional[np.ndarray] = None
    release_year: Optional[int] = None
    rating: Optional[float] = None
    popularity: Optional[float] = None
    content_type: str = ""
    runtime: Optional[int] = None

class SimilarityCalculator(ABC):
    """Abstract base class for similarity calculators"""
    
    @abstractmethod
    def calculate_similarity(
        self, 
        source_features: ContentFeatures, 
        target_features: ContentFeatures
    ) -> float:
        """Calculate similarity score between two content items"""
        pass
    
    @abstractmethod
    def get_weight(self) -> float:
        """Get the weight of this calculator in overall score"""
        pass

class LanguageSimilarityCalculator(SimilarityCalculator):
    """Ultra-precise language similarity calculator"""
    
    def __init__(self):
        self.language_families = {
            'dravidian': {'telugu', 'tamil', 'kannada', 'malayalam', 'te', 'ta', 'kn', 'ml'},
            'indo_aryan': {'hindi', 'bengali', 'gujarati', 'marathi', 'hi', 'bn', 'gu', 'mr'},
            'sino_tibetan': {'chinese', 'mandarin', 'cantonese', 'zh', 'zh-cn', 'zh-tw'},
            'germanic': {'english', 'german', 'dutch', 'en', 'de', 'nl'},
            'romance': {'spanish', 'french', 'italian', 'portuguese', 'es', 'fr', 'it', 'pt'},
            'japanese': {'japanese', 'ja'},
            'korean': {'korean', 'ko'}
        }
        
        self.priority_weights = {
            LanguagePriority.EXACT_MATCH: 1.0,
            LanguagePriority.SAME_FAMILY: 0.8,
            LanguagePriority.REGIONAL: 0.6,
            LanguagePriority.INTERNATIONAL: 0.4,
            LanguagePriority.FALLBACK: 0.2
        }
    
    def normalize_language(self, language: str) -> str:
        """Normalize language codes and names"""
        lang_mapping = {
            'te': 'telugu', 'ta': 'tamil', 'kn': 'kannada', 'ml': 'malayalam',
            'hi': 'hindi', 'en': 'english', 'ja': 'japanese', 'ko': 'korean',
            'zh': 'chinese', 'zh-cn': 'chinese', 'zh-tw': 'chinese',
            'es': 'spanish', 'fr': 'french', 'de': 'german', 'it': 'italian'
        }
        
        normalized = language.lower().strip()
        return lang_mapping.get(normalized, normalized)
    
    def get_language_family(self, language: str) -> Optional[str]:
        """Get language family for a given language"""
        normalized = self.normalize_language(language)
        for family, languages in self.language_families.items():
            if normalized in languages:
                return family
        return None
    
    def calculate_similarity(
        self, 
        source_features: ContentFeatures, 
        target_features: ContentFeatures
    ) -> float:
        """Calculate language similarity with family grouping"""
        if not source_features.languages or not target_features.languages:
            return 0.0
        
        source_normalized = {self.normalize_language(lang) for lang in source_features.languages}
        target_normalized = {self.normalize_language(lang) for lang in target_features.languages}
        
        # Exact match gets highest priority
        exact_matches = source_normalized.intersection(target_normalized)
        if exact_matches:
            return LanguagePriority.EXACT_MATCH.value / 100
        
        # Same family match
        source_families = {self.get_language_family(lang) for lang in source_normalized}
        target_families = {self.get_language_family(lang) for lang in target_normalized}
        source_families.discard(None)
        target_families.discard(None)
        
        if source_families.intersection(target_families):
            return LanguagePriority.SAME_FAMILY.value / 100
        
        return 0.0
    
    def get_weight(self) -> float:
        return 0.4  # 40% weight for language similarity

class GenreSimilarityCalculator(SimilarityCalculator):
    """Advanced genre similarity with weighted importance"""
    
    def __init__(self):
        # Genre importance weights
        self.genre_weights = {
            'action': 1.0, 'drama': 1.0, 'comedy': 1.0, 'thriller': 1.0,
            'romance': 0.9, 'horror': 0.9, 'sci-fi': 0.9, 'fantasy': 0.9,
            'adventure': 0.8, 'mystery': 0.8, 'crime': 0.8,
            'family': 0.7, 'animation': 0.7, 'documentary': 0.6
        }
        
        # Genre compatibility matrix
        self.genre_compatibility = {
            'action': {'adventure', 'thriller', 'crime', 'sci-fi'},
            'drama': {'romance', 'thriller', 'crime', 'family'},
            'comedy': {'family', 'romance', 'adventure'},
            'horror': {'thriller', 'mystery', 'supernatural'},
            'sci-fi': {'action', 'adventure', 'fantasy', 'thriller'},
            'romance': {'drama', 'comedy', 'family'},
            'thriller': {'action', 'crime', 'mystery', 'horror'},
            'fantasy': {'adventure', 'family', 'sci-fi'},
            'crime': {'action', 'drama', 'thriller', 'mystery'},
            'mystery': {'thriller', 'crime', 'horror'},
            'adventure': {'action', 'fantasy', 'family', 'comedy'},
            'family': {'comedy', 'adventure', 'drama', 'animation'}
        }
    
    def calculate_similarity(
        self, 
        source_features: ContentFeatures, 
        target_features: ContentFeatures
    ) -> float:
        """Calculate weighted genre similarity"""
        if not source_features.genres or not target_features.genres:
            return 0.0
        
        source_genres = {g.lower() for g in source_features.genres}
        target_genres = {g.lower() for g in target_features.genres}
        
        # Direct matches
        direct_matches = source_genres.intersection(target_genres)
        direct_score = sum(self.genre_weights.get(genre, 0.5) for genre in direct_matches)
        
        # Compatible genre matches
        compatible_score = 0.0
        for source_genre in source_genres:
            compatible_genres = self.genre_compatibility.get(source_genre, set())
            compatible_matches = compatible_genres.intersection(target_genres)
            compatible_score += sum(self.genre_weights.get(genre, 0.3) * 0.7 for genre in compatible_matches)
        
        total_possible = max(len(source_genres), len(target_genres))
        if total_possible == 0:
            return 0.0
        
        return min(1.0, (direct_score + compatible_score) / total_possible)
    
    def get_weight(self) -> float:
        return 0.25  # 25% weight for genre similarity

class CastSimilarityCalculator(SimilarityCalculator):
    """Cast and crew similarity calculator"""
    
    def calculate_similarity(
        self, 
        source_features: ContentFeatures, 
        target_features: ContentFeatures
    ) -> float:
        """Calculate cast similarity with role importance weighting"""
        cast_score = self._calculate_cast_overlap(source_features.cast_ids, target_features.cast_ids)
        director_score = self._calculate_director_overlap(source_features.director_ids, target_features.director_ids)
        crew_score = self._calculate_crew_overlap(source_features.crew_ids, target_features.crew_ids)
        
        # Weighted combination
        return (cast_score * 0.5) + (director_score * 0.3) + (crew_score * 0.2)
    
    def _calculate_cast_overlap(self, source_cast: Set[int], target_cast: Set[int]) -> float:
        """Calculate cast overlap percentage"""
        if not source_cast or not target_cast:
            return 0.0
        
        overlap = len(source_cast.intersection(target_cast))
        total_unique = len(source_cast.union(target_cast))
        
        return overlap / total_unique if total_unique > 0 else 0.0
    
    def _calculate_director_overlap(self, source_directors: Set[int], target_directors: Set[int]) -> float:
        """Calculate director overlap with high importance"""
        if not source_directors or not target_directors:
            return 0.0
        
        return 1.0 if source_directors.intersection(target_directors) else 0.0
    
    def _calculate_crew_overlap(self, source_crew: Set[int], target_crew: Set[int]) -> float:
        """Calculate crew overlap percentage"""
        if not source_crew or not target_crew:
            return 0.0
        
        overlap = len(source_crew.intersection(target_crew))
        total_unique = len(source_crew.union(target_crew))
        
        return overlap / total_unique if total_unique > 0 else 0.0
    
    def get_weight(self) -> float:
        return 0.2  # 20% weight for cast similarity

class PlotSimilarityCalculator(SimilarityCalculator):
    """Advanced plot similarity using NLP techniques"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.plot_vectors = {}
    
    def calculate_similarity(
        self, 
        source_features: ContentFeatures, 
        target_features: ContentFeatures
    ) -> float:
        """Calculate plot similarity using TF-IDF and cosine similarity"""
        if source_features.plot_vector is None or target_features.plot_vector is None:
            return 0.0
        
        try:
            similarity_matrix = cosine_similarity(
                source_features.plot_vector.reshape(1, -1),
                target_features.plot_vector.reshape(1, -1)
            )
            return similarity_matrix[0][0]
        except Exception as e:
            logger.warning(f"Plot similarity calculation error: {e}")
            return 0.0
    
    def get_weight(self) -> float:
        return 0.1  # 10% weight for plot similarity

class RatingSimilarityCalculator(SimilarityCalculator):
    """Rating and popularity similarity calculator"""
    
    def calculate_similarity(
        self, 
        source_features: ContentFeatures, 
        target_features: ContentFeatures
    ) -> float:
        """Calculate rating similarity with quality preference"""
        if source_features.rating is None or target_features.rating is None:
            return 0.0
        
        rating_diff = abs(source_features.rating - target_features.rating)
        rating_similarity = max(0, 1 - (rating_diff / 10))  # Assuming 10-point scale
        
        # Bonus for high-quality content
        quality_bonus = 0.0
        if source_features.rating >= 7.0 and target_features.rating >= 7.0:
            quality_bonus = 0.2
        
        return min(1.0, rating_similarity + quality_bonus)
    
    def get_weight(self) -> float:
        return 0.05  # 5% weight for rating similarity

class FeatureExtractor:
    """Extract features from content for similarity calculation"""
    
    def __init__(self, db, models):
        self.db = db
        self.Content = models['Content']
        self.ContentPerson = models['ContentPerson']
        self.Person = models['Person']
        self.plot_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self._plot_corpus = []
        self._plot_vectors_fitted = False
    
    def extract_features(self, content_id: int) -> Optional[ContentFeatures]:
        """Extract comprehensive features for a content item"""
        try:
            content = self.Content.query.get(content_id)
            if not content:
                return None
            
            # Extract basic features
            languages = set()
            if content.languages:
                try:
                    lang_list = json.loads(content.languages)
                    languages = {lang.lower() for lang in lang_list if lang}
                except (json.JSONDecodeError, TypeError):
                    pass
            
            genres = set()
            if content.genres:
                try:
                    genre_list = json.loads(content.genres)
                    genres = {genre.lower() for genre in genre_list if genre}
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Extract cast and crew
            cast_crew = self.ContentPerson.query.filter_by(content_id=content_id).all()
            
            cast_ids = set()
            crew_ids = set()
            director_ids = set()
            
            for cp in cast_crew:
                if cp.role_type == 'cast':
                    cast_ids.add(cp.person_id)
                elif cp.role_type == 'crew':
                    crew_ids.add(cp.person_id)
                    if cp.job and cp.job.lower() in ['director', 'directing']:
                        director_ids.add(cp.person_id)
            
            # Extract plot vector
            plot_vector = None
            if content.overview and self._plot_vectors_fitted:
                try:
                    plot_vector = self.plot_vectorizer.transform([content.overview]).toarray()[0]
                except Exception as e:
                    logger.warning(f"Plot vectorization error for content {content_id}: {e}")
            
            return ContentFeatures(
                content_id=content_id,
                languages=languages,
                genres=genres,
                cast_ids=cast_ids,
                crew_ids=crew_ids,
                director_ids=director_ids,
                plot_vector=plot_vector,
                release_year=content.release_date.year if content.release_date else None,
                rating=content.rating,
                popularity=content.popularity,
                content_type=content.content_type,
                runtime=content.runtime
            )
            
        except Exception as e:
            logger.error(f"Feature extraction error for content {content_id}: {e}")
            return None
    
    def prepare_plot_vectors(self, content_ids: List[int]):
        """Prepare plot vectors for similarity calculation"""
        try:
            contents = self.Content.query.filter(
                self.Content.id.in_(content_ids),
                self.Content.overview.isnot(None)
            ).all()
            
            plots = [content.overview for content in contents if content.overview]
            
            if plots:
                self.plot_vectorizer.fit(plots)
                self._plot_vectors_fitted = True
                logger.info(f"Plot vectorizer fitted with {len(plots)} plot summaries")
            
        except Exception as e:
            logger.error(f"Plot vector preparation error: {e}")

class SimilarityEngine:
    """Main similarity calculation engine"""
    
    def __init__(self, db, models, cache=None):
        self.db = db
        self.models = models
        self.cache = cache
        self.feature_extractor = FeatureExtractor(db, models)
        
        # Initialize calculators
        self.calculators = {
            SimilarityType.LANGUAGE_EXACT: LanguageSimilarityCalculator(),
            SimilarityType.GENRE_BASED: GenreSimilarityCalculator(),
            SimilarityType.CAST_SIMILARITY: CastSimilarityCalculator(),
            SimilarityType.PLOT_SIMILARITY: PlotSimilarityCalculator(),
            SimilarityType.RATING_SIMILARITY: RatingSimilarityCalculator()
        }
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def calculate_similarity_score(
        self, 
        source_features: ContentFeatures, 
        target_features: ContentFeatures,
        algorithm_weights: Optional[Dict[SimilarityType, float]] = None
    ) -> SimilarityScore:
        """Calculate comprehensive similarity score"""
        
        if algorithm_weights is None:
            algorithm_weights = {
                SimilarityType.LANGUAGE_EXACT: 0.4,
                SimilarityType.GENRE_BASED: 0.25,
                SimilarityType.CAST_SIMILARITY: 0.2,
                SimilarityType.PLOT_SIMILARITY: 0.1,
                SimilarityType.RATING_SIMILARITY: 0.05
            }
        
        # Calculate individual similarities
        language_score = self.calculators[SimilarityType.LANGUAGE_EXACT].calculate_similarity(
            source_features, target_features
        )
        genre_score = self.calculators[SimilarityType.GENRE_BASED].calculate_similarity(
            source_features, target_features
        )
        cast_score = self.calculators[SimilarityType.CAST_SIMILARITY].calculate_similarity(
            source_features, target_features
        )
        plot_score = self.calculators[SimilarityType.PLOT_SIMILARITY].calculate_similarity(
            source_features, target_features
        )
        rating_score = self.calculators[SimilarityType.RATING_SIMILARITY].calculate_similarity(
            source_features, target_features
        )
        
        # Calculate weighted total score
        total_score = (
            language_score * algorithm_weights[SimilarityType.LANGUAGE_EXACT] +
            genre_score * algorithm_weights[SimilarityType.GENRE_BASED] +
            cast_score * algorithm_weights[SimilarityType.CAST_SIMILARITY] +
            plot_score * algorithm_weights[SimilarityType.PLOT_SIMILARITY] +
            rating_score * algorithm_weights[SimilarityType.RATING_SIMILARITY]
        )
        
        # Generate match reasons
        match_reasons = []
        if language_score > 0.8:
            match_reasons.append("Exact language match")
        elif language_score > 0.6:
            match_reasons.append("Same language family")
        
        if genre_score > 0.7:
            match_reasons.append("Similar genres")
        
        if cast_score > 0.3:
            match_reasons.append("Shared cast members")
        
        if plot_score > 0.5:
            match_reasons.append("Similar plot themes")
        
        if rating_score > 0.8:
            match_reasons.append("Similar quality rating")
        
        return SimilarityScore(
            content_id=target_features.content_id,
            total_score=total_score,
            language_score=language_score,
            genre_score=genre_score,
            cast_score=cast_score,
            plot_score=plot_score,
            rating_score=rating_score,
            algorithm_used="weighted_multi_factor",
            match_reasons=match_reasons
        )
    
    def find_similar_content(
        self, 
        content_id: int, 
        limit: int = 10,
        min_similarity: float = 0.3,
        language_priority: bool = True,
        include_same_franchise: bool = True
    ) -> List[Dict[str, Any]]:
        """Find similar content with advanced algorithms"""
        
        try:
            # Check cache first
            cache_key = f"similar_v2:{content_id}:{limit}:{min_similarity}:{language_priority}"
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    return cached_result
            
            # Extract source features
            source_features = self.feature_extractor.extract_features(content_id)
            if not source_features:
                return []
            
            # Get candidate content
            Content = self.models['Content']
            candidates = Content.query.filter(
                Content.id != content_id,
                Content.content_type == source_features.content_type
            ).limit(500).all()  # Pre-filter for performance
            
            if not candidates:
                return []
            
            # Prepare plot vectors if needed
            candidate_ids = [c.id for c in candidates] + [content_id]
            self.feature_extractor.prepare_plot_vectors(candidate_ids)
            
            # Calculate similarities
            similarities = []
            
            for candidate in candidates:
                target_features = self.feature_extractor.extract_features(candidate.id)
                if not target_features:
                    continue
                
                similarity_score = self.calculate_similarity_score(source_features, target_features)
                
                if similarity_score.total_score >= min_similarity:
                    similarities.append(similarity_score)
            
            # Sort by total score (language priority is built into weights)
            similarities.sort(key=lambda x: x.total_score, reverse=True)
            
            # Convert to response format
            result = []
            for sim_score in similarities[:limit]:
                content = Content.query.get(sim_score.content_id)
                if content:
                    # Ensure content has slug
                    if not content.slug:
                        content.slug = f"content-{content.id}"
                    
                    youtube_url = None
                    if content.youtube_trailer_id:
                        youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                    
                    result.append({
                        'id': content.id,
                        'slug': content.slug,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'languages': json.loads(content.languages or '[]'),
                        'rating': content.rating,
                        'release_date': content.release_date.isoformat() if content.release_date else None,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                        'overview': content.overview[:150] + '...' if content.overview else '',
                        'youtube_trailer': youtube_url,
                        'similarity_score': round(sim_score.total_score, 3),
                        'language_score': round(sim_score.language_score, 3),
                        'genre_score': round(sim_score.genre_score, 3),
                        'cast_score': round(sim_score.cast_score, 3),
                        'confidence_level': round(sim_score.confidence_level, 1),
                        'match_reasons': sim_score.match_reasons,
                        'algorithm_used': sim_score.algorithm_used
                    })
            
            # Cache result
            if self.cache:
                self.cache.set(cache_key, result, timeout=1800)  # 30 minutes
            
            return result
            
        except Exception as e:
            logger.error(f"Similar content calculation error: {e}")
            return []

class GenreExplorationService:
    """Advanced genre exploration with intelligent recommendations"""
    
    def __init__(self, db, models, cache=None):
        self.db = db
        self.models = models
        self.cache = cache
        self.similarity_engine = SimilarityEngine(db, models, cache)
        
        # Genre hierarchy and relationships
        self.genre_hierarchy = {
            'action': {
                'subgenres': ['martial_arts', 'spy', 'superhero', 'war'],
                'related': ['adventure', 'thriller', 'crime'],
                'opposite': ['drama', 'romance', 'comedy']
            },
            'drama': {
                'subgenres': ['family_drama', 'historical', 'biographical', 'social'],
                'related': ['romance', 'thriller', 'family'],
                'opposite': ['comedy', 'action', 'horror']
            },
            'comedy': {
                'subgenres': ['romantic_comedy', 'dark_comedy', 'parody', 'sitcom'],
                'related': ['family', 'romance', 'adventure'],
                'opposite': ['horror', 'thriller', 'drama']
            },
            'horror': {
                'subgenres': ['psychological', 'supernatural', 'slasher', 'zombie'],
                'related': ['thriller', 'mystery', 'supernatural'],
                'opposite': ['comedy', 'family', 'romance']
            },
            'sci-fi': {
                'subgenres': ['space_opera', 'cyberpunk', 'dystopian', 'time_travel'],
                'related': ['action', 'adventure', 'thriller'],
                'opposite': ['historical', 'period', 'slice_of_life']
            }
        }
        
        # Language-specific genre preferences
        self.language_genre_preferences = {
            'telugu': ['action', 'drama', 'romance', 'family', 'comedy'],
            'hindi': ['drama', 'action', 'romance', 'comedy', 'thriller'],
            'tamil': ['action', 'drama', 'comedy', 'romance', 'thriller'],
            'english': ['action', 'drama', 'comedy', 'sci-fi', 'thriller'],
            'japanese': ['anime', 'drama', 'romance', 'slice_of_life', 'supernatural']
        }
    
    def explore_genre(
        self, 
        genre: str, 
        language_preference: Optional[str] = None,
        content_type: str = 'movie',
        limit: int = 20,
        quality_threshold: float = 6.0,
        diversity_factor: float = 0.3
    ) -> Dict[str, Any]:
        """Explore genre with intelligent filtering and recommendations"""
        
        try:
            cache_key = f"genre_explore_v2:{genre}:{language_preference}:{content_type}:{limit}:{quality_threshold}"
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    return cached_result
            
            Content = self.models['Content']
            
            # Base query
            query = Content.query.filter(
                Content.content_type == content_type,
                Content.rating >= quality_threshold
            )
            
            # Genre filtering
            if genre.lower() != 'all':
                query = query.filter(Content.genres.contains(genre))
            
            # Language prioritization
            if language_preference:
                # Primary language content
                primary_content = query.filter(
                    Content.languages.contains(language_preference)
                ).order_by(Content.rating.desc()).limit(limit).all()
                
                # Fill remaining with other languages if needed
                remaining_limit = max(0, limit - len(primary_content))
                if remaining_limit > 0:
                    other_content = query.filter(
                        ~Content.languages.contains(language_preference)
                    ).order_by(Content.rating.desc()).limit(remaining_limit).all()
                    
                    all_content = primary_content + other_content
                else:
                    all_content = primary_content
            else:
                all_content = query.order_by(Content.rating.desc()).limit(limit * 2).all()
            
            # Apply diversity filtering
            if diversity_factor > 0:
                all_content = self._apply_diversity_filter(all_content, diversity_factor, limit)
            else:
                all_content = all_content[:limit]
            
            # Format results
            results = []
            for content in all_content:
                if not content.slug:
                    content.slug = f"content-{content.id}"
                
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                results.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'vote_count': content.vote_count,
                    'popularity': content.popularity,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'youtube_trailer': youtube_url
                })
            
            # Get related genres
            related_genres = self._get_related_genres(genre)
            
            # Get genre statistics
            stats = self._get_genre_statistics(genre, content_type, language_preference)
            
            response = {
                'genre': genre,
                'language_preference': language_preference,
                'content_type': content_type,
                'results': results,
                'related_genres': related_genres,
                'statistics': stats,
                'metadata': {
                    'total_results': len(results),
                    'quality_threshold': quality_threshold,
                    'diversity_applied': diversity_factor > 0,
                    'algorithm': 'advanced_genre_exploration_v2',
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            # Cache result
            if self.cache:
                self.cache.set(cache_key, response, timeout=1800)  # 30 minutes
            
            return response
            
        except Exception as e:
            logger.error(f"Genre exploration error: {e}")
            return {
                'genre': genre,
                'results': [],
                'error': str(e)
            }
    
    def _apply_diversity_filter(
        self, 
        content_list: List, 
        diversity_factor: float, 
        limit: int
    ) -> List:
        """Apply diversity filtering to avoid similar content clustering"""
        
        if len(content_list) <= limit:
            return content_list
        
        selected = []
        remaining = content_list.copy()
        
        # Always include the top-rated item
        if remaining:
            selected.append(remaining.pop(0))
        
        while len(selected) < limit and remaining:
            best_candidate = None
            best_diversity_score = -1
            
            for candidate in remaining:
                # Calculate diversity score
                diversity_score = self._calculate_diversity_score(candidate, selected)
                
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return selected
    
    def _calculate_diversity_score(self, candidate, selected_items) -> float:
        """Calculate how diverse a candidate is from already selected items"""
        
        if not selected_items:
            return 1.0
        
        try:
            candidate_genres = set(json.loads(candidate.genres or '[]'))
            candidate_year = candidate.release_date.year if candidate.release_date else 2000
            
            diversity_scores = []
            
            for selected in selected_items:
                selected_genres = set(json.loads(selected.genres or '[]'))
                selected_year = selected.release_date.year if selected.release_date else 2000
                
                # Genre diversity
                genre_overlap = len(candidate_genres.intersection(selected_genres))
                genre_diversity = 1.0 - (genre_overlap / max(len(candidate_genres), len(selected_genres), 1))
                
                # Temporal diversity
                year_diff = abs(candidate_year - selected_year)
                temporal_diversity = min(1.0, year_diff / 10)  # Normalize to 10-year span
                
                # Combined diversity
                combined_diversity = (genre_diversity + temporal_diversity) / 2
                diversity_scores.append(combined_diversity)
            
            return sum(diversity_scores) / len(diversity_scores)
            
        except Exception as e:
            logger.warning(f"Diversity calculation error: {e}")
            return 0.5  # Default moderate diversity
    
    def _get_related_genres(self, genre: str) -> List[str]:
        """Get related genres for exploration"""
        
        genre_lower = genre.lower()
        if genre_lower in self.genre_hierarchy:
            related = self.genre_hierarchy[genre_lower].get('related', [])
            subgenres = self.genre_hierarchy[genre_lower].get('subgenres', [])
            return related + subgenres
        
        return []
    
    def _get_genre_statistics(
        self, 
        genre: str, 
        content_type: str, 
        language_preference: Optional[str]
    ) -> Dict[str, Any]:
        """Get statistical information about the genre"""
        
        try:
            Content = self.models['Content']
            
            base_query = Content.query.filter(Content.content_type == content_type)
            
            if genre.lower() != 'all':
                base_query = base_query.filter(Content.genres.contains(genre))
            
            total_count = base_query.count()
            
            if language_preference:
                language_count = base_query.filter(
                    Content.languages.contains(language_preference)
                ).count()
            else:
                language_count = None
            
            # Rating statistics
            rating_stats = base_query.filter(
                Content.rating.isnot(None)
            ).with_entities(
                func.avg(Content.rating).label('avg_rating'),
                func.min(Content.rating).label('min_rating'),
                func.max(Content.rating).label('max_rating')
            ).first()
            
            return {
                'total_content': total_count,
                'language_specific_count': language_count,
                'average_rating': round(float(rating_stats.avg_rating), 2) if rating_stats.avg_rating else None,
                'rating_range': {
                    'min': float(rating_stats.min_rating) if rating_stats.min_rating else None,
                    'max': float(rating_stats.max_rating) if rating_stats.max_rating else None
                }
            }
            
        except Exception as e:
            logger.error(f"Genre statistics error: {e}")
            return {}

class SimilarTitlesService:
    """Main service for similar titles functionality"""
    
    def __init__(self, app, db, models, cache=None):
        self.app = app
        self.db = db
        self.models = models
        self.cache = cache
        self.similarity_engine = SimilarityEngine(db, models, cache)
        self.genre_service = GenreExplorationService(db, models, cache)
        
        # Performance monitoring
        self.performance_metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'average_response_time': 0.0,
            'error_count': 0
        }
    
    def get_similar_titles(
        self, 
        content_id: int, 
        limit: int = 10,
        min_similarity: float = 0.3,
        language_priority: bool = True,
        algorithm: str = 'weighted_multi_factor'
    ) -> Dict[str, Any]:
        """Get similar titles with production-ready implementation"""
        
        start_time = time.time()
        self.performance_metrics['total_requests'] += 1
        
        try:
            # Validate input
            if limit > 50:
                limit = 50  # Cap for performance
            
            if min_similarity < 0.1:
                min_similarity = 0.1  # Minimum threshold
            
            # Get similar content
            similar_content = self.similarity_engine.find_similar_content(
                content_id=content_id,
                limit=limit,
                min_similarity=min_similarity,
                language_priority=language_priority
            )
            
            # Get base content info
            Content = self.models['Content']
            base_content = Content.query.get(content_id)
            
            if not base_content:
                raise ValueError(f"Content with ID {content_id} not found")
            
            if not base_content.slug:
                base_content.slug = f"content-{base_content.id}"
            
            response = {
                'base_content': {
                    'id': base_content.id,
                    'slug': base_content.slug,
                    'title': base_content.title,
                    'content_type': base_content.content_type,
                    'languages': json.loads(base_content.languages or '[]'),
                    'genres': json.loads(base_content.genres or '[]'),
                    'rating': base_content.rating
                },
                'similar_content': similar_content,
                'metadata': {
                    'algorithm_used': algorithm,
                    'language_priority_enabled': language_priority,
                    'total_results': len(similar_content),
                    'min_similarity_threshold': min_similarity,
                    'response_time_ms': round((time.time() - start_time) * 1000, 2),
                    'cache_used': any('cache_hit' in str(item) for item in similar_content) if similar_content else False,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            # Update performance metrics
            response_time = time.time() - start_time
            self.performance_metrics['average_response_time'] = (
                (self.performance_metrics['average_response_time'] * (self.performance_metrics['total_requests'] - 1) + response_time) /
                self.performance_metrics['total_requests']
            )
            
            return response
            
        except Exception as e:
            self.performance_metrics['error_count'] += 1
            logger.error(f"Similar titles service error: {e}")
            
            return {
                'base_content': {'id': content_id},
                'similar_content': [],
                'error': str(e),
                'metadata': {
                    'algorithm_used': algorithm,
                    'error_occurred': True,
                    'response_time_ms': round((time.time() - start_time) * 1000, 2),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
    
    def explore_genre(
        self, 
        genre: str, 
        language_preference: Optional[str] = None,
        content_type: str = 'movie',
        limit: int = 20,
        quality_threshold: float = 6.0
    ) -> Dict[str, Any]:
        """Explore genre with advanced filtering"""
        
        start_time = time.time()
        
        try:
            result = self.genre_service.explore_genre(
                genre=genre,
                language_preference=language_preference,
                content_type=content_type,
                limit=limit,
                quality_threshold=quality_threshold
            )
            
            # Add performance metrics
            result['metadata']['response_time_ms'] = round((time.time() - start_time) * 1000, 2)
            
            return result
            
        except Exception as e:
            logger.error(f"Genre exploration service error: {e}")
            return {
                'genre': genre,
                'results': [],
                'error': str(e),
                'metadata': {
                    'response_time_ms': round((time.time() - start_time) * 1000, 2),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics"""
        return {
            'performance': self.performance_metrics,
            'cache_hit_rate': (
                self.performance_metrics['cache_hits'] / max(self.performance_metrics['total_requests'], 1) * 100
            ),
            'error_rate': (
                self.performance_metrics['error_count'] / max(self.performance_metrics['total_requests'], 1) * 100
            ),
            'timestamp': datetime.utcnow().isoformat()
        }

def init_similar_service(app, db, models, cache=None):
    """Initialize the similar titles service"""
    try:
        service = SimilarTitlesService(app, db, models, cache)
        logger.info("Similar titles service initialized successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize similar titles service: {e}")
        return None