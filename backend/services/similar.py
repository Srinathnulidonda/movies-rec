# backend/services/similar.py

import numpy as np
import json
import logging
import time
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import lru_cache
import re

# Scientific computing and ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# Text processing and embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, falling back to TF-IDF")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available, using basic text processing")

# Database and caching
from sqlalchemy import and_, or_, func, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

@dataclass
class SimilarityScore:
    """Data class for similarity scoring"""
    content_id: int
    score: float
    similarity_type: str
    language_match: bool
    genre_overlap: float
    text_similarity: float
    metadata: Dict[str, Any]

@dataclass
class SimilarityConfig:
    """Configuration for similarity calculations"""
    language_weight: float = 0.4
    genre_weight: float = 0.25
    text_weight: float = 0.2
    rating_weight: float = 0.1
    popularity_weight: float = 0.05
    min_similarity_threshold: float = 0.3
    max_results: int = 20
    enable_vector_similarity: bool = True
    enable_collaborative_filtering: bool = True
    cache_ttl: int = 3600  # 1 hour

class AdvancedSimilarityEngine:
    """
    Production-ready similarity engine with vector-based approaches
    and perfect language matching capabilities.
    """
    
    def __init__(self, db, models, cache=None, config=None):
        self.db = db
        self.models = models
        self.cache = cache
        self.config = config or SimilarityConfig()
        
        # Initialize models and vectorizers
        self._sentence_model = None
        self._tfidf_vectorizer = None
        self._genre_vectorizer = None
        self._scaler = StandardScaler()
        
        # Language mappings
        self.language_codes = {
            'telugu': ['te', 'tel', 'telugu'],
            'english': ['en', 'eng', 'english'],
            'hindi': ['hi', 'hin', 'hindi'],
            'tamil': ['ta', 'tam', 'tamil'],
            'malayalam': ['ml', 'mal', 'malayalam'],
            'kannada': ['kn', 'kan', 'kannada'],
            'japanese': ['ja', 'jpn', 'japanese'],
            'korean': ['ko', 'kor', 'korean'],
            'spanish': ['es', 'spa', 'spanish'],
            'french': ['fr', 'fra', 'french'],
            'german': ['de', 'deu', 'german'],
            'italian': ['it', 'ita', 'italian'],
            'portuguese': ['pt', 'por', 'portuguese'],
            'russian': ['ru', 'rus', 'russian'],
            'chinese': ['zh', 'chi', 'chinese', 'mandarin'],
            'arabic': ['ar', 'ara', 'arabic']
        }
        
        # Initialize components
        self._initialize_models()
        
        logger.info("Advanced Similarity Engine initialized successfully")
    
    def _initialize_models(self):
        """Initialize ML models and vectorizers"""
        try:
            # Initialize sentence transformer if available
            if SENTENCE_TRANSFORMERS_AVAILABLE and self.config.enable_vector_similarity:
                try:
                    self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Sentence transformer model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load sentence transformer: {e}")
                    self._sentence_model = None
            
            # Initialize TF-IDF vectorizers
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            self._genre_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 1),
                min_df=1
            )
            
            logger.info("Vectorizers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def normalize_language(self, language: str) -> str:
        """Normalize language codes and names to standard format"""
        if not language:
            return 'unknown'
        
        lang_lower = language.lower().strip()
        
        for standard_lang, variants in self.language_codes.items():
            if lang_lower in variants:
                return standard_lang
        
        return lang_lower
    
    def extract_languages_from_content(self, content) -> Set[str]:
        """Extract and normalize languages from content"""
        languages = set()
        
        try:
            # From languages field
            if content.languages:
                lang_list = json.loads(content.languages) if isinstance(content.languages, str) else content.languages
                for lang in lang_list:
                    normalized = self.normalize_language(lang)
                    languages.add(normalized)
            
            # From original title analysis
            if content.original_title and content.title:
                if content.original_title != content.title:
                    # Detect language from script/characters
                    detected_lang = self._detect_language_from_title(content.original_title)
                    if detected_lang:
                        languages.add(detected_lang)
            
        except Exception as e:
            logger.warning(f"Error extracting languages for content {content.id}: {e}")
        
        return languages if languages else {'unknown'}
    
    def _detect_language_from_title(self, title: str) -> Optional[str]:
        """Detect language from title using character analysis"""
        if not title:
            return None
        
        # Telugu script detection
        telugu_pattern = r'[\u0C00-\u0C7F]'
        if re.search(telugu_pattern, title):
            return 'telugu'
        
        # Tamil script detection
        tamil_pattern = r'[\u0B80-\u0BFF]'
        if re.search(tamil_pattern, title):
            return 'tamil'
        
        # Malayalam script detection
        malayalam_pattern = r'[\u0D00-\u0D7F]'
        if re.search(malayalam_pattern, title):
            return 'malayalam'
        
        # Kannada script detection
        kannada_pattern = r'[\u0C80-\u0CFF]'
        if re.search(kannada_pattern, title):
            return 'kannada'
        
        # Hindi/Devanagari script detection
        hindi_pattern = r'[\u0900-\u097F]'
        if re.search(hindi_pattern, title):
            return 'hindi'
        
        # Japanese script detection
        japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]'
        if re.search(japanese_pattern, title):
            return 'japanese'
        
        # Korean script detection
        korean_pattern = r'[\uAC00-\uD7AF]'
        if re.search(korean_pattern, title):
            return 'korean'
        
        # Chinese script detection
        chinese_pattern = r'[\u4E00-\u9FFF]'
        if re.search(chinese_pattern, title):
            return 'chinese'
        
        # Arabic script detection
        arabic_pattern = r'[\u0600-\u06FF]'
        if re.search(arabic_pattern, title):
            return 'arabic'
        
        return None
    
    def get_similar_content(self, content_id: int, limit: int = 20, 
                          language_strict: bool = True, 
                          include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Get similar content with perfect language matching and vector-based similarity.
        
        Args:
            content_id: ID of the base content
            limit: Maximum number of results
            language_strict: If True, prioritize exact language matches
            include_metadata: Include detailed similarity metadata
            
        Returns:
            List of similar content with similarity scores and metadata
        """
        try:
            # Check cache first
            cache_key = f"similar_advanced:{content_id}:{limit}:{language_strict}"
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for similar content {content_id}")
                    return cached_result
            
            # Get base content
            Content = self.models['Content']
            base_content = self.db.session.query(Content).filter_by(id=content_id).first()
            
            if not base_content:
                logger.warning(f"Content {content_id} not found")
                return []
            
            # Extract base content features
            base_languages = self.extract_languages_from_content(base_content)
            base_genres = self._extract_genres(base_content)
            base_text_features = self._extract_text_features(base_content)
            
            logger.info(f"Base content {content_id} languages: {base_languages}, genres: {base_genres}")
            
            # Get candidate content
            candidates = self._get_candidate_content(base_content, language_strict, base_languages)
            
            if not candidates:
                logger.warning(f"No candidates found for content {content_id}")
                return []
            
            # Calculate similarities
            similarities = []
            
            # Process in batches for memory efficiency
            batch_size = 100
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                batch_similarities = self._calculate_batch_similarities(
                    base_content, batch, base_languages, base_genres, base_text_features
                )
                similarities.extend(batch_similarities)
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x.score, reverse=True)
            
            # Convert to response format
            results = []
            for sim in similarities[:limit]:
                candidate = next(c for c in candidates if c.id == sim.content_id)
                
                # Ensure slug exists
                if not candidate.slug:
                    candidate.slug = f"content-{candidate.id}"
                
                result = {
                    'id': candidate.id,
                    'slug': candidate.slug,
                    'title': candidate.title,
                    'original_title': candidate.original_title,
                    'content_type': candidate.content_type,
                    'rating': candidate.rating,
                    'poster_path': self._format_poster_path(candidate.poster_path),
                    'release_date': candidate.release_date.isoformat() if candidate.release_date else None,
                    'similarity_score': round(sim.score, 4),
                    'language_match': sim.language_match,
                    'languages': list(self.extract_languages_from_content(candidate))
                }
                
                if include_metadata:
                    result['similarity_metadata'] = {
                        'similarity_type': sim.similarity_type,
                        'genre_overlap': round(sim.genre_overlap, 3),
                        'text_similarity': round(sim.text_similarity, 3),
                        'language_match_score': 1.0 if sim.language_match else 0.0,
                        'additional_metadata': sim.metadata
                    }
                
                results.append(result)
            
            # Cache the results
            if self.cache:
                self.cache.set(cache_key, results, timeout=self.config.cache_ttl)
            
            logger.info(f"Found {len(results)} similar content for {content_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error getting similar content for {content_id}: {e}")
            return []
    
    def _get_candidate_content(self, base_content, language_strict: bool, base_languages: Set[str]) -> List:
        """Get candidate content for similarity comparison"""
        Content = self.models['Content']
        
        query = self.db.session.query(Content).filter(
            Content.id != base_content.id,
            Content.content_type == base_content.content_type
        )
        
        if language_strict and 'unknown' not in base_languages:
            # Create language filter conditions
            language_conditions = []
            for lang in base_languages:
                # Check various language representations
                lang_variants = self.language_codes.get(lang, [lang])
                for variant in lang_variants:
                    language_conditions.append(Content.languages.contains(variant))
            
            if language_conditions:
                query = query.filter(or_(*language_conditions))
        
        # Add quality filters
        query = query.filter(
            Content.rating.isnot(None),
            Content.rating > 0
        ).order_by(
            Content.rating.desc(),
            Content.popularity.desc()
        ).limit(1000)  # Limit candidates for performance
        
        return query.all()
    
    def _calculate_batch_similarities(self, base_content, candidates: List, 
                                    base_languages: Set[str], base_genres: List[str], 
                                    base_text_features: Dict[str, Any]) -> List[SimilarityScore]:
        """Calculate similarities for a batch of candidates"""
        similarities = []
        
        for candidate in candidates:
            try:
                similarity = self._calculate_individual_similarity(
                    base_content, candidate, base_languages, base_genres, base_text_features
                )
                if similarity.score >= self.config.min_similarity_threshold:
                    similarities.append(similarity)
            except Exception as e:
                logger.warning(f"Error calculating similarity for candidate {candidate.id}: {e}")
                continue
        
        return similarities
    
    def _calculate_individual_similarity(self, base_content, candidate, 
                                       base_languages: Set[str], base_genres: List[str], 
                                       base_text_features: Dict[str, Any]) -> SimilarityScore:
        """Calculate similarity between base content and a candidate"""
        
        # Extract candidate features
        candidate_languages = self.extract_languages_from_content(candidate)
        candidate_genres = self._extract_genres(candidate)
        candidate_text_features = self._extract_text_features(candidate)
        
        # Language similarity (highest priority)
        language_match = bool(base_languages.intersection(candidate_languages))
        language_score = 1.0 if language_match else 0.0
        
        # Genre similarity
        genre_overlap = self._calculate_genre_similarity(base_genres, candidate_genres)
        
        # Text similarity (title + overview)
        text_similarity = self._calculate_text_similarity(base_text_features, candidate_text_features)
        
        # Rating similarity
        rating_similarity = self._calculate_rating_similarity(base_content.rating or 0, candidate.rating or 0)
        
        # Popularity similarity
        popularity_similarity = self._calculate_popularity_similarity(
            base_content.popularity or 0, candidate.popularity or 0
        )
        
        # Calculate weighted similarity score
        total_score = (
            language_score * self.config.language_weight +
            genre_overlap * self.config.genre_weight +
            text_similarity * self.config.text_weight +
            rating_similarity * self.config.rating_weight +
            popularity_similarity * self.config.popularity_weight
        )
        
        # Boost score for exact language matches
        if language_match:
            total_score *= 1.2  # 20% boost for language match
        
        # Determine similarity type
        similarity_type = self._determine_similarity_type(
            language_match, genre_overlap, text_similarity
        )
        
        return SimilarityScore(
            content_id=candidate.id,
            score=min(total_score, 1.0),  # Cap at 1.0
            similarity_type=similarity_type,
            language_match=language_match,
            genre_overlap=genre_overlap,
            text_similarity=text_similarity,
            metadata={
                'rating_similarity': rating_similarity,
                'popularity_similarity': popularity_similarity,
                'base_languages': list(base_languages),
                'candidate_languages': list(candidate_languages),
                'genre_match_count': len(set(base_genres).intersection(set(candidate_genres)))
            }
        )
    
    def _extract_genres(self, content) -> List[str]:
        """Extract genres from content"""
        try:
            if content.genres:
                genres = json.loads(content.genres) if isinstance(content.genres, str) else content.genres
                return [genre.lower().strip() for genre in genres if genre]
            return []
        except Exception as e:
            logger.warning(f"Error extracting genres for content {content.id}: {e}")
            return []
    
    def _extract_text_features(self, content) -> Dict[str, Any]:
        """Extract text features from content"""
        features = {
            'title': content.title or '',
            'original_title': content.original_title or '',
            'overview': content.overview or '',
            'combined_text': ''
        }
        
        # Combine all text features
        combined_parts = [
            features['title'],
            features['original_title'],
            features['overview'][:200]  # Limit overview length
        ]
        features['combined_text'] = ' '.join(filter(None, combined_parts))
        
        return features
    
    def _calculate_genre_similarity(self, genres1: List[str], genres2: List[str]) -> float:
        """Calculate genre similarity using Jaccard similarity"""
        if not genres1 or not genres2:
            return 0.0
        
        set1 = set(genres1)
        set2 = set(genres2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_text_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate text similarity using multiple approaches"""
        try:
            text1 = features1['combined_text']
            text2 = features2['combined_text']
            
            if not text1 or not text2:
                return 0.0
            
            # Use sentence transformer if available
            if self._sentence_model:
                return self._calculate_sentence_similarity(text1, text2)
            else:
                return self._calculate_tfidf_similarity(text1, text2)
                
        except Exception as e:
            logger.warning(f"Error calculating text similarity: {e}")
            return 0.0
    
    def _calculate_sentence_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using sentence transformers"""
        try:
            embeddings = self._sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Error in sentence similarity calculation: {e}")
            return 0.0
    
    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using TF-IDF vectors"""
        try:
            documents = [text1, text2]
            tfidf_matrix = self._tfidf_vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Error in TF-IDF similarity calculation: {e}")
            return 0.0
    
    def _calculate_rating_similarity(self, rating1: float, rating2: float) -> float:
        """Calculate rating similarity"""
        if rating1 <= 0 or rating2 <= 0:
            return 0.0
        
        # Normalize ratings to 0-1 scale (assuming 0-10 scale)
        norm_rating1 = min(rating1 / 10.0, 1.0)
        norm_rating2 = min(rating2 / 10.0, 1.0)
        
        # Calculate inverse of absolute difference
        diff = abs(norm_rating1 - norm_rating2)
        return 1.0 - diff
    
    def _calculate_popularity_similarity(self, pop1: float, pop2: float) -> float:
        """Calculate popularity similarity"""
        if pop1 <= 0 or pop2 <= 0:
            return 0.0
        
        # Use log scale for popularity to handle wide ranges
        log_pop1 = np.log1p(pop1)
        log_pop2 = np.log1p(pop2)
        
        # Normalize and calculate similarity
        max_log_pop = max(log_pop1, log_pop2)
        if max_log_pop == 0:
            return 1.0
        
        diff = abs(log_pop1 - log_pop2) / max_log_pop
        return 1.0 - diff
    
    def _determine_similarity_type(self, language_match: bool, genre_overlap: float, 
                                 text_similarity: float) -> str:
        """Determine the type of similarity"""
        if language_match and genre_overlap > 0.5 and text_similarity > 0.5:
            return 'perfect_match'
        elif language_match and genre_overlap > 0.3:
            return 'language_genre_match'
        elif language_match:
            return 'language_match'
        elif genre_overlap > 0.5:
            return 'genre_match'
        elif text_similarity > 0.5:
            return 'text_match'
        else:
            return 'general_similarity'
    
    def _format_poster_path(self, poster_path: str) -> Optional[str]:
        """Format poster path for display"""
        if not poster_path:
            return None
        
        if poster_path.startswith('http'):
            return poster_path
        else:
            return f"https://image.tmdb.org/t/p/w300{poster_path}"

class AdvancedGenreExplorer:
    """
    Production-ready genre exploration engine with advanced filtering
    and recommendation capabilities.
    """
    
    def __init__(self, db, models, cache=None, similarity_engine=None):
        self.db = db
        self.models = models
        self.cache = cache
        self.similarity_engine = similarity_engine
        
        # Genre mappings and hierarchies
        self.genre_hierarchies = {
            'action': {
                'subgenres': ['martial_arts', 'superhero', 'spy', 'war', 'heist'],
                'related': ['adventure', 'thriller', 'crime']
            },
            'drama': {
                'subgenres': ['family', 'romantic', 'historical', 'biographical', 'legal'],
                'related': ['romance', 'thriller', 'mystery']
            },
            'comedy': {
                'subgenres': ['romantic_comedy', 'dark_comedy', 'satire', 'parody'],
                'related': ['romance', 'family', 'adventure']
            },
            'horror': {
                'subgenres': ['psychological', 'supernatural', 'slasher', 'zombie'],
                'related': ['thriller', 'mystery', 'sci-fi']
            },
            'sci-fi': {
                'subgenres': ['space_opera', 'cyberpunk', 'dystopian', 'time_travel'],
                'related': ['action', 'adventure', 'thriller']
            },
            'romance': {
                'subgenres': ['romantic_comedy', 'romantic_drama', 'historical_romance'],
                'related': ['drama', 'comedy', 'family']
            }
        }
        
        # Language-specific genre preferences
        self.language_genre_preferences = {
            'telugu': ['action', 'drama', 'romance', 'comedy', 'family'],
            'hindi': ['action', 'drama', 'romance', 'comedy', 'musical'],
            'tamil': ['action', 'drama', 'comedy', 'thriller'],
            'malayalam': ['drama', 'comedy', 'thriller', 'family'],
            'kannada': ['action', 'drama', 'comedy', 'romance'],
            'english': ['action', 'drama', 'comedy', 'thriller', 'sci-fi', 'horror'],
            'japanese': ['anime', 'drama', 'action', 'romance', 'horror'],
            'korean': ['drama', 'thriller', 'romance', 'action', 'comedy']
        }
        
        logger.info("Advanced Genre Explorer initialized successfully")
    
    def explore_genre(self, genre: str, language: Optional[str] = None, 
                     content_type: str = 'movie', limit: int = 20,
                     sort_by: str = 'popularity', filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Explore content by genre with advanced filtering and language preferences.
        
        Args:
            genre: Primary genre to explore
            language: Preferred language (None for all languages)
            content_type: Type of content (movie, tv, anime)
            limit: Maximum number of results
            sort_by: Sorting criteria (popularity, rating, release_date, similarity)
            filters: Additional filters (year_range, rating_range, etc.)
            
        Returns:
            Dict containing categorized recommendations and metadata
        """
        try:
            # Check cache
            cache_key = f"genre_explore:{genre}:{language}:{content_type}:{limit}:{sort_by}:{hash(str(filters))}"
            if self.cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for genre exploration: {genre}")
                    return cached_result
            
            # Normalize genre
            genre_normalized = genre.lower().replace(' ', '_').replace('-', '_')
            
            # Get base query
            Content = self.models['Content']
            query = self.db.session.query(Content).filter(Content.content_type == content_type)
            
            # Apply genre filter
            genre_conditions = self._build_genre_conditions(genre_normalized, Content)
            if genre_conditions:
                query = query.filter(or_(*genre_conditions))
            
            # Apply language filter
            if language:
                language_conditions = self._build_language_conditions(language, Content)
                if language_conditions:
                    query = query.filter(or_(*language_conditions))
            
            # Apply additional filters
            if filters:
                query = self._apply_additional_filters(query, filters, Content)
            
            # Apply sorting
            query = self._apply_sorting(query, sort_by, Content)
            
            # Get results
            all_content = query.limit(limit * 3).all()  # Get more for categorization
            
            if not all_content:
                return {
                    'genre': genre,
                    'language': language,
                    'recommendations': [],
                    'categories': {},
                    'metadata': {'total_found': 0, 'message': 'No content found for specified criteria'}
                }
            
            # Categorize results
            categorized_results = self._categorize_genre_results(
                all_content, genre_normalized, language, limit
            )
            
            # Add metadata
            categorized_results['metadata'] = {
                'total_found': len(all_content),
                'genre': genre,
                'language': language,
                'content_type': content_type,
                'sort_by': sort_by,
                'applied_filters': filters or {},
                'genre_hierarchy': self.genre_hierarchies.get(genre_normalized, {}),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Cache results
            if self.cache:
                self.cache.set(cache_key, categorized_results, timeout=1800)  # 30 minutes
            
            logger.info(f"Genre exploration completed for {genre}: {len(all_content)} items found")
            return categorized_results
            
        except Exception as e:
            logger.error(f"Error exploring genre {genre}: {e}")
            return {
                'genre': genre,
                'recommendations': [],
                'categories': {},
                'metadata': {'error': str(e)}
            }
    
    def _build_genre_conditions(self, genre: str, Content):
        """Build genre filter conditions"""
        conditions = []
        
        # Direct genre match
        conditions.append(Content.genres.contains(genre))
        
        # Check for common variations
        genre_variations = [
            genre.replace('_', ' '),
            genre.replace('_', '-'),
            genre.replace('_', ''),
            genre.title(),
            genre.capitalize()
        ]
        
        for variation in genre_variations:
            conditions.append(Content.genres.contains(variation))
        
        # Add related genres if available
        if genre in self.genre_hierarchies:
            related_genres = self.genre_hierarchies[genre].get('related', [])
            for related in related_genres:
                conditions.append(Content.genres.contains(related))
        
        return conditions
    
    def _build_language_conditions(self, language: str, Content):
        """Build language filter conditions"""
        conditions = []
        
        # Normalize language
        if hasattr(self.similarity_engine, 'normalize_language'):
            normalized_lang = self.similarity_engine.normalize_language(language)
        else:
            normalized_lang = language.lower()
        
        # Get language variants
        language_variants = []
        if hasattr(self.similarity_engine, 'language_codes'):
            language_variants = self.similarity_engine.language_codes.get(normalized_lang, [normalized_lang])
        else:
            language_variants = [normalized_lang, language]
        
        # Add conditions for each variant
        for variant in language_variants:
            conditions.append(Content.languages.contains(variant))
        
        return conditions
    
    def _apply_additional_filters(self, query, filters: Dict[str, Any], Content):
        """Apply additional filters to the query"""
        
        # Year range filter
        if 'year_range' in filters:
            year_range = filters['year_range']
            if 'min' in year_range:
                query = query.filter(
                    func.extract('year', Content.release_date) >= year_range['min']
                )
            if 'max' in year_range:
                query = query.filter(
                    func.extract('year', Content.release_date) <= year_range['max']
                )
        
        # Rating range filter
        if 'rating_range' in filters:
            rating_range = filters['rating_range']
            if 'min' in rating_range:
                query = query.filter(Content.rating >= rating_range['min'])
            if 'max' in rating_range:
                query = query.filter(Content.rating <= rating_range['max'])
        
        # Minimum vote count
        if 'min_votes' in filters:
            query = query.filter(Content.vote_count >= filters['min_votes'])
        
        # Runtime filter
        if 'runtime_range' in filters:
            runtime_range = filters['runtime_range']
            if 'min' in runtime_range:
                query = query.filter(Content.runtime >= runtime_range['min'])
            if 'max' in runtime_range:
                query = query.filter(Content.runtime <= runtime_range['max'])
        
        return query
    
    def _apply_sorting(self, query, sort_by: str, Content):
        """Apply sorting to the query"""
        
        if sort_by == 'popularity':
            return query.order_by(Content.popularity.desc().nulls_last())
        elif sort_by == 'rating':
            return query.order_by(Content.rating.desc().nulls_last())
        elif sort_by == 'release_date':
            return query.order_by(Content.release_date.desc().nulls_last())
        elif sort_by == 'title':
            return query.order_by(Content.title.asc())
        elif sort_by == 'vote_count':
            return query.order_by(Content.vote_count.desc().nulls_last())
        else:
            # Default to popularity
            return query.order_by(Content.popularity.desc().nulls_last())
    
    def _categorize_genre_results(self, content_list: List, genre: str, 
                                language: Optional[str], limit: int) -> Dict[str, Any]:
        """Categorize genre exploration results"""
        
        categories = {
            'featured': [],           # Top picks
            'highly_rated': [],       # High rating content
            'recent': [],             # Recently released
            'popular': [],            # Most popular
            'hidden_gems': [],        # Lower popularity but good rating
            'classic': []             # Older but acclaimed content
        }
        
        main_recommendations = []
        
        # Sort content by composite score for main recommendations
        scored_content = []
        for content in content_list:
            score = self._calculate_genre_relevance_score(content, genre, language)
            scored_content.append((content, score))
        
        scored_content.sort(key=lambda x: x[1], reverse=True)
        
        # Fill main recommendations
        for content, score in scored_content[:limit]:
            if not content.slug:
                content.slug = f"content-{content.id}"
            
            item = self._format_content_item(content, score)
            main_recommendations.append(item)
        
        # Categorize content
        current_year = datetime.now().year
        
        for content, score in scored_content:
            if not content.slug:
                content.slug = f"content-{content.id}"
            
            item = self._format_content_item(content, score)
            
            # Featured: Top scoring items
            if score > 0.8 and len(categories['featured']) < 10:
                categories['featured'].append(item)
            
            # Highly rated: Rating > 8.0
            if (content.rating or 0) >= 8.0 and len(categories['highly_rated']) < 10:
                categories['highly_rated'].append(item)
            
            # Recent: Released in last 2 years
            if (content.release_date and 
                content.release_date.year >= current_year - 2 and 
                len(categories['recent']) < 10):
                categories['recent'].append(item)
            
            # Popular: High popularity
            if (content.popularity or 0) > 50 and len(categories['popular']) < 10:
                categories['popular'].append(item)
            
            # Hidden gems: Good rating but lower popularity
            if ((content.rating or 0) >= 7.5 and 
                (content.popularity or 0) < 20 and 
                len(categories['hidden_gems']) < 10):
                categories['hidden_gems'].append(item)
            
            # Classic: Older but acclaimed
            if (content.release_date and 
                content.release_date.year < current_year - 10 and
                (content.rating or 0) >= 7.5 and 
                len(categories['classic']) < 10):
                categories['classic'].append(item)
        
        return {
            'genre': genre,
            'language': language,
            'recommendations': main_recommendations,
            'categories': {k: v for k, v in categories.items() if v}  # Only include non-empty categories
        }
    
    def _calculate_genre_relevance_score(self, content, genre: str, language: Optional[str]) -> float:
        """Calculate relevance score for genre exploration"""
        score = 0.0
        
        # Base score from rating and popularity
        rating_score = min((content.rating or 0) / 10.0, 1.0) * 0.4
        popularity_score = min(np.log1p(content.popularity or 0) / 10.0, 1.0) * 0.3
        
        score += rating_score + popularity_score
        
        # Genre relevance bonus
        try:
            genres = json.loads(content.genres or '[]')
            if any(genre.lower() in g.lower() for g in genres):
                score += 0.2
        except:
            pass
        
        # Language preference bonus
        if language and hasattr(self.similarity_engine, 'extract_languages_from_content'):
            content_languages = self.similarity_engine.extract_languages_from_content(content)
            normalized_lang = self.similarity_engine.normalize_language(language)
            if normalized_lang in content_languages:
                score += 0.1
        
        return min(score, 1.0)
    
    def _format_content_item(self, content, score: float = None) -> Dict[str, Any]:
        """Format content item for API response"""
        # Extract genres safely
        try:
            genres = json.loads(content.genres or '[]')
        except:
            genres = []
        
        # Extract languages safely
        try:
            if hasattr(self.similarity_engine, 'extract_languages_from_content'):
                languages = list(self.similarity_engine.extract_languages_from_content(content))
            else:
                languages = json.loads(content.languages or '[]')
        except:
            languages = []
        
        # Format poster path
        poster_path = content.poster_path
        if poster_path and not poster_path.startswith('http'):
            poster_path = f"https://image.tmdb.org/t/p/w300{poster_path}"
        
        # Format YouTube trailer
        youtube_url = None
        if content.youtube_trailer_id:
            youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
        
        item = {
            'id': content.id,
            'slug': content.slug or f"content-{content.id}",
            'title': content.title,
            'original_title': content.original_title,
            'content_type': content.content_type,
            'genres': genres,
            'languages': languages,
            'rating': content.rating,
            'vote_count': content.vote_count,
            'popularity': content.popularity,
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'poster_path': poster_path,
            'overview': (content.overview[:150] + '...') if content.overview and len(content.overview) > 150 else content.overview,
            'youtube_trailer': youtube_url
        }
        
        if score is not None:
            item['relevance_score'] = round(score, 4)
        
        return item

def init_similarity_service(db, models, cache=None, config=None):
    """Initialize the similarity service with all components"""
    try:
        # Initialize similarity engine
        similarity_engine = AdvancedSimilarityEngine(db, models, cache, config)
        
        # Initialize genre explorer
        genre_explorer = AdvancedGenreExplorer(db, models, cache, similarity_engine)
        
        logger.info("Similarity service initialized successfully")
        
        return {
            'similarity_engine': similarity_engine,
            'genre_explorer': genre_explorer
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize similarity service: {e}")
        raise

# Export main classes and functions
__all__ = [
    'AdvancedSimilarityEngine',
    'AdvancedGenreExplorer', 
    'SimilarityScore',
    'SimilarityConfig',
    'init_similarity_service'
]