# backend/personalized/utils.py
"""
CineBrain Personalization Utilities
Advanced ML utilities for embeddings, similarity computation, and caching
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import json
import logging
import hashlib
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import pickle

logger = logging.getLogger(__name__)

# Telugu-first language configuration
TELUGU_PRIORITY_CONFIG = {
    'primary_languages': ['telugu', 'te'],
    'secondary_languages': ['english', 'en', 'hindi', 'hi'],
    'tertiary_languages': ['tamil', 'ta', 'malayalam', 'ml', 'kannada', 'kn'],
    'language_weights': {
        'telugu': 1.0, 'te': 1.0,
        'english': 0.9, 'en': 0.9,
        'hindi': 0.85, 'hi': 0.85,
        'tamil': 0.8, 'ta': 0.8,
        'malayalam': 0.75, 'ml': 0.75,
        'kannada': 0.7, 'kn': 0.7
    }
}

class TeluguPriorityManager:
    """
    Manages Telugu-first language prioritization across the system
    """
    
    @staticmethod
    def calculate_language_score(content_languages: List[str], 
                                user_languages: List[str] = None) -> float:
        """
        Calculate language priority score with Telugu-first approach
        
        Args:
            content_languages: List of content languages
            user_languages: User's preferred languages (optional)
        
        Returns:
            float: Language priority score (0.0 to 1.0)
        """
        if not content_languages:
            return 0.0
        
        max_score = 0.0
        weights = TELUGU_PRIORITY_CONFIG['language_weights']
        
        # Check each content language against priority weights
        for lang in content_languages:
            lang_lower = lang.lower() if isinstance(lang, str) else ''
            
            # Direct match with priority languages
            if lang_lower in weights:
                max_score = max(max_score, weights[lang_lower])
            
            # Partial match (e.g., "Telugu" contains "telugu")
            for priority_lang, weight in weights.items():
                if priority_lang in lang_lower:
                    max_score = max(max_score, weight * 0.9)  # Slight reduction for partial match
        
        # User preference bonus
        if user_languages:
            for user_lang in user_languages:
                user_lang_lower = user_lang.lower()
                if any(user_lang_lower in content_lang.lower() for content_lang in content_languages):
                    max_score *= 1.2  # 20% boost for user preference
        
        return min(max_score, 1.0)
    
    @staticmethod
    def sort_by_language_priority(content_list: List[Any], 
                                 user_languages: List[str] = None) -> List[Any]:
        """
        Sort content by Telugu-first language priority
        
        Args:
            content_list: List of content objects
            user_languages: User's preferred languages
        
        Returns:
            List[Any]: Sorted content list
        """
        def get_priority_score(content):
            languages = []
            if hasattr(content, 'languages') and content.languages:
                try:
                    languages = json.loads(content.languages) if isinstance(content.languages, str) else content.languages
                except (json.JSONDecodeError, TypeError):
                    languages = []
            
            return TeluguPriorityManager.calculate_language_score(languages, user_languages)
        
        return sorted(content_list, key=get_priority_score, reverse=True)

class EmbeddingManager:
    """
    Manages user and content embeddings with real-time updates
    """
    
    def __init__(self, embedding_dim: int = 128, cache_manager=None):
        self.embedding_dim = embedding_dim
        self.cache_manager = cache_manager
        
        # TF-IDF vectorizers for different content aspects
        self.title_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        self.overview_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        
        # Dimensionality reduction
        self.svd_reducer = TruncatedSVD(n_components=embedding_dim, random_state=42)
        self.scaler = StandardScaler()
        
        # Content and user embedding caches
        self._content_embeddings = {}
        self._user_embeddings = {}
        
        # Fitted status
        self._fitted = False
        
    def fit_content_embeddings(self, content_list: List[Any]) -> np.ndarray:
        """
        Fit embedding models on content corpus and generate embeddings
        
        Args:
            content_list: List of content objects
        
        Returns:
            np.ndarray: Content embeddings matrix
        """
        try:
            # Extract text features
            titles = []
            overviews = []
            
            for content in content_list:
                title = getattr(content, 'title', '') or ''
                overview = getattr(content, 'overview', '') or ''
                
                titles.append(title)
                overviews.append(overview)
            
            # Create TF-IDF features
            logger.info(f"Creating TF-IDF features for {len(content_list)} content items")
            
            title_features = self.title_vectorizer.fit_transform(titles)
            overview_features = self.overview_vectorizer.fit_transform(overviews)
            
            # Combine features
            combined_features = np.hstack([
                title_features.toarray(),
                overview_features.toarray()
            ])
            
            # Apply dimensionality reduction
            if combined_features.shape[1] > self.embedding_dim:
                reduced_features = self.svd_reducer.fit_transform(combined_features)
                embeddings = self.scaler.fit_transform(reduced_features)
            else:
                embeddings = self.scaler.fit_transform(combined_features)
                # Pad or truncate to target dimension
                if embeddings.shape[1] < self.embedding_dim:
                    padding = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
                    embeddings = np.hstack([embeddings, padding])
                else:
                    embeddings = embeddings[:, :self.embedding_dim]
            
            # Cache embeddings
            for i, content in enumerate(content_list):
                self._content_embeddings[content.id] = embeddings[i]
            
            self._fitted = True
            logger.info(f"✅ Content embeddings fitted: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error fitting content embeddings: {e}")
            return np.array([])
    
    def get_content_embedding(self, content: Any) -> np.ndarray:
        """
        Get or generate embedding for a single content item
        
        Args:
            content: Content object
        
        Returns:
            np.ndarray: Content embedding vector
        """
        # Check cache first
        if content.id in self._content_embeddings:
            return self._content_embeddings[content.id]
        
        # Generate embedding if not cached
        if not self._fitted:
            # If not fitted, create simple embedding
            return self._generate_simple_embedding(content)
        
        try:
            # Use fitted vectorizers
            title = getattr(content, 'title', '') or ''
            overview = getattr(content, 'overview', '') or ''
            
            title_features = self.title_vectorizer.transform([title])
            overview_features = self.overview_vectorizer.transform([overview])
            
            combined_features = np.hstack([
                title_features.toarray(),
                overview_features.toarray()
            ])
            
            if combined_features.shape[1] > self.embedding_dim:
                reduced_features = self.svd_reducer.transform(combined_features)
                embedding = self.scaler.transform(reduced_features)[0]
            else:
                embedding = self.scaler.transform(combined_features)[0]
                if len(embedding) < self.embedding_dim:
                    padding = np.zeros(self.embedding_dim - len(embedding))
                    embedding = np.concatenate([embedding, padding])
                else:
                    embedding = embedding[:self.embedding_dim]
            
            # Cache the embedding
            self._content_embeddings[content.id] = embedding
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Error generating content embedding for {content.id}: {e}")
            return self._generate_simple_embedding(content)
    
    def _generate_simple_embedding(self, content: Any) -> np.ndarray:
        """Generate simple embedding when advanced methods fail"""
        # Create basic embedding from available features
        embedding = np.zeros(self.embedding_dim)
        
        # Use content ID as seed for reproducible randomness
        np.random.seed(content.id % 10000)
        
        # Basic features
        if hasattr(content, 'rating') and content.rating:
            embedding[0] = content.rating / 10.0
        
        if hasattr(content, 'popularity') and content.popularity:
            embedding[1] = min(content.popularity / 100.0, 1.0)
        
        # Genre features (simplified)
        if hasattr(content, 'genres') and content.genres:
            try:
                genres = json.loads(content.genres) if isinstance(content.genres, str) else content.genres
                genre_hash = hash(' '.join(sorted(genres))) % 1000
                embedding[2:5] = [(genre_hash >> i) & 1 for i in range(3)]
            except:
                pass
        
        # Add some randomness based on content ID
        embedding[5:] = np.random.normal(0, 0.1, self.embedding_dim - 5)
        
        return embedding
    
    def update_user_embedding(self, user_id: int, 
                            interaction_data: Dict[str, Any],
                            learning_rate: float = 0.1) -> np.ndarray:
        """
        Update user embedding based on new interaction
        
        Args:
            user_id: User ID
            interaction_data: Interaction information
            learning_rate: Learning rate for updates
        
        Returns:
            np.ndarray: Updated user embedding
        """
        # Get current user embedding
        current_embedding = self._user_embeddings.get(user_id, np.zeros(self.embedding_dim))
        
        # Get content embedding
        content_id = interaction_data.get('content_id')
        if not content_id:
            return current_embedding
        
        # Load content embedding from cache or compute
        content_embedding = self._get_cached_content_embedding(content_id)
        if content_embedding is None:
            return current_embedding
        
        # Calculate update based on interaction type
        interaction_type = interaction_data.get('interaction_type', 'view')
        interaction_weights = {
            'favorite': 1.0,
            'like': 0.8,
            'view': 0.3,
            'search': 0.5,
            'rating': 0.9,
            'share': 1.2,
            'dislike': -0.5,
            'skip': -0.2
        }
        
        weight = interaction_weights.get(interaction_type, 0.3)
        
        # Apply rating weight if available
        if interaction_data.get('rating'):
            rating_weight = (interaction_data['rating'] - 5.0) / 5.0  # Normalize to [-1, 1]
            weight *= (1.0 + rating_weight)
        
        # Update embedding using exponential moving average
        updated_embedding = (1 - learning_rate) * current_embedding + learning_rate * weight * content_embedding
        
        # Normalize to prevent embedding drift
        updated_embedding = updated_embedding / (np.linalg.norm(updated_embedding) + 1e-10)
        
        # Cache updated embedding
        self._user_embeddings[user_id] = updated_embedding
        
        # Cache in external cache if available
        if self.cache_manager:
            cache_key = f"user_embedding:{user_id}"
            self.cache_manager.set(cache_key, updated_embedding.tolist(), ttl=3600)
        
        return updated_embedding
    
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """
        Get user embedding, loading from cache if available
        
        Args:
            user_id: User ID
        
        Returns:
            np.ndarray: User embedding vector
        """
        # Check memory cache first
        if user_id in self._user_embeddings:
            return self._user_embeddings[user_id]
        
        # Check external cache
        if self.cache_manager:
            cache_key = f"user_embedding:{user_id}"
            cached_embedding = self.cache_manager.get(cache_key)
            if cached_embedding:
                embedding = np.array(cached_embedding)
                self._user_embeddings[user_id] = embedding
                return embedding
        
        # Return zero embedding for new users
        return np.zeros(self.embedding_dim)
    
    def _get_cached_content_embedding(self, content_id: int) -> Optional[np.ndarray]:
        """Get content embedding from cache"""
        return self._content_embeddings.get(content_id)
    
    def compute_user_content_similarity(self, user_id: int, content_list: List[Any]) -> np.ndarray:
        """
        Compute similarity scores between user and multiple content items
        
        Args:
            user_id: User ID
            content_list: List of content objects
        
        Returns:
            np.ndarray: Similarity scores
        """
        user_embedding = self.get_user_embedding(user_id)
        
        if np.allclose(user_embedding, 0):
            # New user - return popularity-based scores
            scores = []
            for content in content_list:
                if hasattr(content, 'popularity') and content.popularity:
                    scores.append(min(content.popularity / 100.0, 1.0))
                else:
                    scores.append(0.5)
            return np.array(scores)
        
        # Compute similarities
        similarities = []
        for content in content_list:
            content_embedding = self.get_content_embedding(content)
            similarity = cosine_similarity([user_embedding], [content_embedding])[0][0]
            similarities.append(similarity)
        
        return np.array(similarities)

class SimilarityEngine:
    """
    Advanced similarity computation engine
    """
    
    def __init__(self, embedding_manager=None, cache_manager=None):
        self.embedding_manager = embedding_manager
        self.cache_manager = cache_manager
        
    def compute_content_similarity(self, content1: Any, content2: Any) -> Dict[str, float]:
        """
        Compute multi-dimensional similarity between two content items
        
        Args:
            content1: First content object
            content2: Second content object
        
        Returns:
            Dict[str, float]: Similarity scores by dimension
        """
        similarities = {}
        
        # Embedding-based similarity
        if self.embedding_manager:
            emb1 = self.embedding_manager.get_content_embedding(content1)
            emb2 = self.embedding_manager.get_content_embedding(content2)
            similarities['embedding'] = cosine_similarity([emb1], [emb2])[0][0]
        
        # Genre similarity
        similarities['genre'] = self._compute_genre_similarity(content1, content2)
        
        # Language similarity
        similarities['language'] = self._compute_language_similarity(content1, content2)
        
        # Rating similarity
        similarities['rating'] = self._compute_rating_similarity(content1, content2)
        
        # Temporal similarity
        similarities['temporal'] = self._compute_temporal_similarity(content1, content2)
        
        return similarities
    
    def _compute_genre_similarity(self, content1: Any, content2: Any) -> float:
        """Compute genre-based similarity"""
        try:
            genres1 = json.loads(getattr(content1, 'genres', '[]') or '[]')
            genres2 = json.loads(getattr(content2, 'genres', '[]') or '[]')
            
            if not genres1 or not genres2:
                return 0.0
            
            set1, set2 = set(genres1), set(genres2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _compute_language_similarity(self, content1: Any, content2: Any) -> float:
        """Compute language-based similarity with Telugu priority"""
        try:
            languages1 = json.loads(getattr(content1, 'languages', '[]') or '[]')
            languages2 = json.loads(getattr(content2, 'languages', '[]') or '[]')
            
            if not languages1 or not languages2:
                return 0.0
            
            # Apply Telugu priority scoring
            score1 = TeluguPriorityManager.calculate_language_score(languages1)
            score2 = TeluguPriorityManager.calculate_language_score(languages2)
            
            # Similarity based on priority scores
            return 1.0 - abs(score1 - score2)
            
        except Exception:
            return 0.0
    
    def _compute_rating_similarity(self, content1: Any, content2: Any) -> float:
        """Compute rating-based similarity"""
        rating1 = getattr(content1, 'rating', None)
        rating2 = getattr(content2, 'rating', None)
        
        if rating1 is None or rating2 is None:
            return 0.5
        
        # Normalize ratings and compute similarity
        diff = abs(rating1 - rating2)
        return max(0.0, 1.0 - diff / 10.0)
    
    def _compute_temporal_similarity(self, content1: Any, content2: Any) -> float:
        """Compute temporal similarity based on release dates"""
        date1 = getattr(content1, 'release_date', None)
        date2 = getattr(content2, 'release_date', None)
        
        if not date1 or not date2:
            return 0.5
        
        # Convert to datetime if needed
        if hasattr(date1, 'year'):
            year1 = date1.year
        else:
            return 0.5
            
        if hasattr(date2, 'year'):
            year2 = date2.year
        else:
            return 0.5
        
        year_diff = abs(year1 - year2)
        
        # Same year = 1.0, decreasing similarity with time difference
        if year_diff == 0:
            return 1.0
        elif year_diff <= 2:
            return 0.8
        elif year_diff <= 5:
            return 0.6
        elif year_diff <= 10:
            return 0.4
        else:
            return 0.2

class CacheManager:
    """
    Advanced caching manager with multiple backends
    """
    
    def __init__(self, cache_backend=None, default_ttl: int = 3600):
        self.cache_backend = cache_backend
        self.default_ttl = default_ttl
        self.memory_cache = {}
        self.max_memory_items = 1000
        
    def get(self, key: str, default=None):
        """Get item from cache with fallback to memory cache"""
        try:
            # Try external cache first
            if self.cache_backend:
                result = self.cache_backend.get(key)
                if result is not None:
                    return result
            
            # Fallback to memory cache
            if key in self.memory_cache:
                item, timestamp = self.memory_cache[key]
                if time.time() - timestamp < self.default_ttl:
                    return item
                else:
                    del self.memory_cache[key]
            
            return default
            
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set item in cache with TTL"""
        try:
            ttl = ttl or self.default_ttl
            
            # Set in external cache
            if self.cache_backend:
                self.cache_backend.set(key, value, timeout=ttl)
            
            # Set in memory cache
            self._memory_cache_set(key, value)
            
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
    
    def delete(self, key: str):
        """Delete item from cache"""
        try:
            if self.cache_backend:
                self.cache_backend.delete(key)
            
            if key in self.memory_cache:
                del self.memory_cache[key]
                
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
    
    def _memory_cache_set(self, key: str, value: Any):
        """Set item in memory cache with size limit"""
        # Remove oldest items if cache is full
        if len(self.memory_cache) >= self.max_memory_items:
            oldest_key = min(self.memory_cache.keys(), 
                           key=lambda k: self.memory_cache[k][1])
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = (value, time.time())
    
    def get_user_cache_key(self, user_id: int, suffix: str = "") -> str:
        """Generate cache key for user-specific data"""
        return f"cinebrain:user:{user_id}:{suffix}" if suffix else f"cinebrain:user:{user_id}"
    
    def get_content_cache_key(self, content_id: int, suffix: str = "") -> str:
        """Generate cache key for content-specific data"""
        return f"cinebrain:content:{content_id}:{suffix}" if suffix else f"cinebrain:content:{content_id}"

def safe_json_loads(json_str: str, default=None):
    """Safely load JSON string with fallback"""
    try:
        return json.loads(json_str) if json_str else (default or [])
    except (json.JSONDecodeError, TypeError):
        return default or []

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to 0-1 range"""
    if len(scores) == 0:
        return scores
    
    min_score, max_score = np.min(scores), np.max(scores)
    if max_score == min_score:
        return np.ones_like(scores) * 0.5
    
    return (scores - min_score) / (max_score - min_score)

def calculate_diversity_score(content_list: List[Any]) -> float:
    """Calculate diversity score for a list of content"""
    if len(content_list) < 2:
        return 1.0
    
    # Collect genres and types
    all_genres = []
    all_types = []
    all_languages = []
    
    for content in content_list:
        # Genres
        genres = safe_json_loads(getattr(content, 'genres', '[]'))
        all_genres.extend(genres)
        
        # Content types
        content_type = getattr(content, 'content_type', 'unknown')
        all_types.append(content_type)
        
        # Languages
        languages = safe_json_loads(getattr(content, 'languages', '[]'))
        all_languages.extend(languages)
    
    # Calculate diversity metrics
    genre_diversity = len(set(all_genres)) / max(len(all_genres), 1)
    type_diversity = len(set(all_types)) / max(len(all_types), 1)
    lang_diversity = len(set(all_languages)) / max(len(all_languages), 1)
    
    # Weighted average
    return (genre_diversity * 0.4 + type_diversity * 0.3 + lang_diversity * 0.3)