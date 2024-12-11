# backend/personalized/utils.py

"""
CineBrain Personalization Utilities
===================================

Core utilities for embedding management, similarity calculations, caching,
and performance optimization in the personalization system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import json
import hashlib
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import redis
from functools import wraps
import pickle
import lz4.frame
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)

# Telugu-first language configuration
LANGUAGE_PRIORITY = {
    'telugu': 1.0,
    'english': 0.9,
    'hindi': 0.85,
    'malayalam': 0.75,
    'kannada': 0.7,
    'tamil': 0.65
}

PRIORITY_LANGUAGES = ['telugu', 'english', 'hindi', 'malayalam', 'kannada', 'tamil']

class EmbeddingManager:
    """
    Advanced embedding manager for user preferences and content features.
    Creates and maintains high-dimensional vector representations.
    """
    
    def __init__(self, embedding_dim: int = 128, cache=None):
        """
        Initialize embedding manager
        
        Args:
            embedding_dim: Dimension of embedding vectors
            cache: Cache backend for storing embeddings
        """
        self.embedding_dim = embedding_dim
        self.cache = cache
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=min(embedding_dim, 50))
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        
        # Initialize embedding spaces
        self._user_embeddings = {}
        self._content_embeddings = {}
        self._genre_embeddings = {}
        self._director_embeddings = {}
        
        logger.info(f"CineBrain EmbeddingManager initialized with {embedding_dim}D vectors")
    
    def create_user_embedding(self, user_id: int, interaction_data: List[Dict]) -> np.ndarray:
        """
        Create comprehensive user embedding from interaction history
        
        Args:
            user_id: User identifier
            interaction_data: List of user interactions with metadata
            
        Returns:
            numpy.ndarray: User embedding vector
        """
        try:
            # Extract features from interactions
            features = self._extract_user_features(interaction_data)
            
            # Create multi-dimensional embedding
            embedding_components = []
            
            # 1. Genre preference embedding (Telugu cinema bias)
            genre_embedding = self._create_genre_embedding(features.get('genres', {}))
            embedding_components.append(genre_embedding)
            
            # 2. Language preference embedding (Telugu-first)
            language_embedding = self._create_language_embedding(features.get('languages', {}))
            embedding_components.append(language_embedding)
            
            # 3. Temporal behavior embedding
            temporal_embedding = self._create_temporal_embedding(features.get('temporal_patterns', {}))
            embedding_components.append(temporal_embedding)
            
            # 4. Quality preference embedding
            quality_embedding = self._create_quality_embedding(features.get('rating_patterns', {}))
            embedding_components.append(quality_embedding)
            
            # 5. Content type preference embedding
            type_embedding = self._create_content_type_embedding(features.get('content_types', {}))
            embedding_components.append(type_embedding)
            
            # 6. Cinematic sophistication embedding
            sophistication_embedding = self._create_sophistication_embedding(features.get('sophistication', {}))
            embedding_components.append(sophistication_embedding)
            
            # Combine all components
            combined_embedding = np.concatenate(embedding_components)
            
            # Normalize to target dimension
            if len(combined_embedding) != self.embedding_dim:
                combined_embedding = self._normalize_embedding_dimension(combined_embedding)
            
            # Cache the embedding
            self._cache_embedding(f"user_{user_id}", combined_embedding)
            
            # Update user embeddings dictionary
            self._user_embeddings[user_id] = combined_embedding
            
            logger.info(f"Created user embedding for user {user_id}: {combined_embedding.shape}")
            return combined_embedding
            
        except Exception as e:
            logger.error(f"Error creating user embedding for user {user_id}: {e}")
            return np.zeros(self.embedding_dim)
    
    def create_content_embedding(self, content_id: int, content_data: Dict) -> np.ndarray:
        """
        Create comprehensive content embedding
        
        Args:
            content_id: Content identifier
            content_data: Content metadata and features
            
        Returns:
            numpy.ndarray: Content embedding vector
        """
        try:
            embedding_components = []
            
            # 1. Genre embedding
            genres = content_data.get('genres', [])
            genre_vec = self._vectorize_genres(genres)
            embedding_components.append(genre_vec)
            
            # 2. Language embedding with Telugu priority
            languages = content_data.get('languages', [])
            language_vec = self._vectorize_languages(languages)
            embedding_components.append(language_vec)
            
            # 3. Plot/overview text embedding
            overview = content_data.get('overview', '')
            text_vec = self._vectorize_text(overview)
            embedding_components.append(text_vec)
            
            # 4. Metadata embedding (year, rating, runtime)
            metadata_vec = self._vectorize_metadata(content_data)
            embedding_components.append(metadata_vec)
            
            # 5. Cinematic style embedding
            style_vec = self._vectorize_cinematic_style(content_data)
            embedding_components.append(style_vec)
            
            # Combine and normalize
            combined_embedding = np.concatenate(embedding_components)
            combined_embedding = self._normalize_embedding_dimension(combined_embedding)
            
            # Cache the embedding
            self._cache_embedding(f"content_{content_id}", combined_embedding)
            self._content_embeddings[content_id] = combined_embedding
            
            return combined_embedding
            
        except Exception as e:
            logger.error(f"Error creating content embedding for content {content_id}: {e}")
            return np.zeros(self.embedding_dim)
    
    def update_user_embedding_realtime(self, user_id: int, new_interaction: Dict) -> np.ndarray:
        """
        Update user embedding in real-time based on new interaction
        
        Args:
            user_id: User identifier
            new_interaction: New interaction data
            
        Returns:
            numpy.ndarray: Updated user embedding
        """
        try:
            # Get current embedding
            current_embedding = self.get_user_embedding(user_id)
            if current_embedding is None:
                # Create new embedding if doesn't exist
                return self.create_user_embedding(user_id, [new_interaction])
            
            # Create embedding for new interaction
            interaction_embedding = self._create_interaction_embedding(new_interaction)
            
            # Calculate learning rate based on interaction type and recency
            learning_rate = self._calculate_learning_rate(new_interaction)
            
            # Update embedding using weighted average
            updated_embedding = (
                current_embedding * (1 - learning_rate) + 
                interaction_embedding * learning_rate
            )
            
            # Normalize
            updated_embedding = updated_embedding / (np.linalg.norm(updated_embedding) + 1e-10)
            
            # Cache updated embedding
            self._cache_embedding(f"user_{user_id}", updated_embedding)
            self._user_embeddings[user_id] = updated_embedding
            
            logger.info(f"Updated user embedding for user {user_id} with learning rate {learning_rate}")
            return updated_embedding
            
        except Exception as e:
            logger.error(f"Error updating user embedding for user {user_id}: {e}")
            return self.get_user_embedding(user_id) or np.zeros(self.embedding_dim)
    
    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Get user embedding from cache or memory"""
        try:
            # Try memory first
            if user_id in self._user_embeddings:
                return self._user_embeddings[user_id]
            
            # Try cache
            cached_embedding = self._get_cached_embedding(f"user_{user_id}")
            if cached_embedding is not None:
                self._user_embeddings[user_id] = cached_embedding
                return cached_embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user embedding for user {user_id}: {e}")
            return None
    
    def get_content_embedding(self, content_id: int) -> Optional[np.ndarray]:
        """Get content embedding from cache or memory"""
        try:
            # Try memory first
            if content_id in self._content_embeddings:
                return self._content_embeddings[content_id]
            
            # Try cache
            cached_embedding = self._get_cached_embedding(f"content_{content_id}")
            if cached_embedding is not None:
                self._content_embeddings[content_id] = cached_embedding
                return cached_embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting content embedding for content {content_id}: {e}")
            return None
    
    def _extract_user_features(self, interaction_data: List[Dict]) -> Dict:
        """Extract comprehensive features from user interactions"""
        features = {
            'genres': defaultdict(float),
            'languages': defaultdict(float),
            'content_types': defaultdict(float),
            'rating_patterns': defaultdict(float),
            'temporal_patterns': defaultdict(float),
            'sophistication': defaultdict(float)
        }
        
        if not interaction_data:
            return features
        
        # Interaction weights
        weights = {
            'favorite': 3.0,
            'watchlist': 2.5,
            'rating': 2.0,
            'view': 1.5,
            'search': 1.0,
            'like': 1.2
        }
        
        for interaction in interaction_data:
            weight = weights.get(interaction.get('interaction_type', 'view'), 1.0)
            
            # Recency decay
            timestamp = interaction.get('timestamp', datetime.utcnow())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            days_ago = (datetime.utcnow() - timestamp).days
            recency_weight = np.exp(-days_ago / 30)  # 30-day half-life
            
            final_weight = weight * recency_weight
            
            # Extract genre features
            if 'genres' in interaction:
                for genre in interaction['genres']:
                    features['genres'][genre] += final_weight
            
            # Extract language features with Telugu priority
            if 'languages' in interaction:
                for language in interaction['languages']:
                    lang_weight = LANGUAGE_PRIORITY.get(language.lower(), 0.5)
                    features['languages'][language] += final_weight * lang_weight
            
            # Extract content type features
            if 'content_type' in interaction:
                features['content_types'][interaction['content_type']] += final_weight
            
            # Extract rating patterns
            if 'rating' in interaction and interaction['rating']:
                rating = float(interaction['rating'])
                features['rating_patterns']['avg_rating'] += rating * final_weight
                features['rating_patterns']['rating_count'] += final_weight
                
                if rating >= 8.0:
                    features['sophistication']['high_quality_preference'] += final_weight
        
        return features
    
    def _create_genre_embedding(self, genre_features: Dict) -> np.ndarray:
        """Create genre preference embedding"""
        # Define standard genres
        standard_genres = [
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
            'Music', 'Musical', 'Mystery', 'Romance', 'Science Fiction', 'Sport',
            'Thriller', 'War', 'Western'
        ]
        
        embedding = np.zeros(len(standard_genres))
        total_weight = sum(genre_features.values()) if genre_features else 1
        
        for i, genre in enumerate(standard_genres):
            if genre in genre_features:
                embedding[i] = genre_features[genre] / total_weight
        
        return embedding
    
    def _create_language_embedding(self, language_features: Dict) -> np.ndarray:
        """Create language preference embedding with Telugu priority"""
        embedding = np.zeros(len(PRIORITY_LANGUAGES))
        total_weight = sum(language_features.values()) if language_features else 1
        
        for i, lang in enumerate(PRIORITY_LANGUAGES):
            if lang in language_features:
                embedding[i] = language_features[lang] / total_weight
            elif lang.title() in language_features:
                embedding[i] = language_features[lang.title()] / total_weight
        
        return embedding
    
    def _normalize_embedding_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to target dimension"""
        if len(embedding) == self.embedding_dim:
            return embedding
        elif len(embedding) > self.embedding_dim:
            # Reduce dimension using PCA
            return self.pca.fit_transform(embedding.reshape(1, -1)).flatten()[:self.embedding_dim]
        else:
            # Pad with zeros
            padded = np.zeros(self.embedding_dim)
            padded[:len(embedding)] = embedding
            return padded
    
    def _cache_embedding(self, key: str, embedding: np.ndarray):
        """Cache embedding with compression"""
        if not self.cache:
            return
        
        try:
            # Compress embedding
            compressed_data = lz4.frame.compress(pickle.dumps(embedding))
            cache_key = f"cinebrain:embedding:{key}"
            self.cache.set(cache_key, compressed_data, timeout=86400)  # 24 hours
        except Exception as e:
            logger.warning(f"Failed to cache embedding {key}: {e}")
    
    def _get_cached_embedding(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache with decompression"""
        if not self.cache:
            return None
        
        try:
            cache_key = f"cinebrain:embedding:{key}"
            compressed_data = self.cache.get(cache_key)
            
            if compressed_data:
                return pickle.loads(lz4.frame.decompress(compressed_data))
            
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached embedding {key}: {e}")
            return None

class SimilarityCalculator:
    """
    Advanced similarity calculator for users and content using multiple metrics
    """
    
    def __init__(self):
        """Initialize similarity calculator with multiple distance metrics"""
        self.metrics = {
            'cosine': self._cosine_similarity,
            'euclidean': self._euclidean_similarity,
            'pearson': self._pearson_similarity,
            'jaccard': self._jaccard_similarity,
            'hybrid': self._hybrid_similarity
        }
        
        logger.info("CineBrain SimilarityCalculator initialized with multiple metrics")
    
    def calculate_user_similarity(self, user1_embedding: np.ndarray, 
                                 user2_embedding: np.ndarray,
                                 metric: str = 'hybrid') -> float:
        """
        Calculate similarity between two user embeddings
        
        Args:
            user1_embedding: First user's embedding vector
            user2_embedding: Second user's embedding vector
            metric: Similarity metric to use
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            if metric not in self.metrics:
                metric = 'hybrid'
            
            return self.metrics[metric](user1_embedding, user2_embedding)
            
        except Exception as e:
            logger.error(f"Error calculating user similarity: {e}")
            return 0.0
    
    def calculate_content_similarity(self, content1_embedding: np.ndarray,
                                   content2_embedding: np.ndarray,
                                   metric: str = 'hybrid') -> float:
        """
        Calculate similarity between two content embeddings
        
        Args:
            content1_embedding: First content's embedding vector
            content2_embedding: Second content's embedding vector
            metric: Similarity metric to use
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            if metric not in self.metrics:
                metric = 'hybrid'
            
            return self.metrics[metric](content1_embedding, content2_embedding)
            
        except Exception as e:
            logger.error(f"Error calculating content similarity: {e}")
            return 0.0
    
    def find_similar_users(self, target_user_embedding: np.ndarray,
                          user_embeddings: Dict[int, np.ndarray],
                          top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find most similar users to target user
        
        Args:
            target_user_embedding: Target user's embedding
            user_embeddings: Dictionary of user embeddings
            top_k: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        try:
            similarities = []
            
            for user_id, embedding in user_embeddings.items():
                similarity = self.calculate_user_similarity(target_user_embedding, embedding)
                similarities.append((user_id, similarity))
            
            # Sort by similarity and return top K
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return []
    
    def find_similar_content(self, target_content_embedding: np.ndarray,
                           content_embeddings: Dict[int, np.ndarray],
                           top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Find most similar content to target content
        
        Args:
            target_content_embedding: Target content's embedding
            content_embeddings: Dictionary of content embeddings
            top_k: Number of similar content items to return
            
        Returns:
            List of (content_id, similarity_score) tuples
        """
        try:
            similarities = []
            
            for content_id, embedding in content_embeddings.items():
                similarity = self.calculate_content_similarity(target_content_embedding, embedding)
                similarities.append((content_id, similarity))
            
            # Sort by similarity and return top K
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar content: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        try:
            return cosine_similarity([vec1], [vec2])[0][0]
        except:
            return 0.0
    
    def _euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate euclidean similarity (inverted distance)"""
        try:
            distance = euclidean_distances([vec1], [vec2])[0][0]
            return 1 / (1 + distance)
        except:
            return 0.0
    
    def _hybrid_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate hybrid similarity combining multiple metrics"""
        try:
            cosine = self._cosine_similarity(vec1, vec2)
            euclidean = self._euclidean_similarity(vec1, vec2)
            
            # Weighted combination
            return cosine * 0.7 + euclidean * 0.3
        except:
            return 0.0

class CacheManager:
    """
    Advanced cache manager for personalization system with intelligent invalidation
    """
    
    def __init__(self, cache=None):
        """
        Initialize cache manager
        
        Args:
            cache: Cache backend (Redis or Simple cache)
        """
        self.cache = cache
        self.default_timeout = 3600  # 1 hour
        self.cache_stats = defaultdict(int)
        
        logger.info("CineBrain CacheManager initialized")
    
    def get_user_recommendations(self, user_id: int, category: str = 'general') -> Optional[List[Dict]]:
        """Get cached user recommendations"""
        try:
            cache_key = f"cinebrain:user_recs:{user_id}:{category}"
            cached_data = self.cache.get(cache_key) if self.cache else None
            
            if cached_data:
                self.cache_stats['hits'] += 1
                return json.loads(cached_data)
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.warning(f"Error getting cached recommendations: {e}")
            return None
    
    def set_user_recommendations(self, user_id: int, recommendations: List[Dict],
                               category: str = 'general', timeout: int = None):
        """Cache user recommendations"""
        try:
            if not self.cache:
                return
            
            cache_key = f"cinebrain:user_recs:{user_id}:{category}"
            cache_timeout = timeout or self.default_timeout
            
            self.cache.set(cache_key, json.dumps(recommendations), timeout=cache_timeout)
            self.cache_stats['sets'] += 1
            
        except Exception as e:
            logger.warning(f"Error caching recommendations: {e}")
    
    def invalidate_user_cache(self, user_id: int):
        """Invalidate all cache entries for a user"""
        try:
            if not self.cache:
                return
            
            # Define cache patterns to invalidate
            patterns = [
                f"cinebrain:user_recs:{user_id}:*",
                f"cinebrain:embedding:user_{user_id}",
                f"cinebrain:profile:{user_id}",
                f"cinebrain:similar_users:{user_id}"
            ]
            
            for pattern in patterns:
                try:
                    # For Redis
                    if hasattr(self.cache, 'delete_many'):
                        keys = self.cache.keys(pattern)
                        if keys:
                            self.cache.delete_many(keys)
                    # For simple cache
                    else:
                        self.cache.delete(pattern)
                except:
                    pass
            
            self.cache_stats['invalidations'] += 1
            logger.info(f"Invalidated cache for user {user_id}")
            
        except Exception as e:
            logger.warning(f"Error invalidating user cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'sets': self.cache_stats['sets'],
            'invalidations': self.cache_stats['invalidations'],
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests
        }

class PerformanceOptimizer:
    """
    Performance optimization utilities for the personalization system
    """
    
    def __init__(self):
        """Initialize performance optimizer"""
        self.execution_times = defaultdict(list)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("CineBrain PerformanceOptimizer initialized")
    
    def time_function(self, func_name: str):
        """Decorator to time function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                self.execution_times[func_name].append(execution_time)
                
                # Keep only last 100 measurements
                if len(self.execution_times[func_name]) > 100:
                    self.execution_times[func_name] = self.execution_times[func_name][-100:]
                
                return result
            return wrapper
        return decorator
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        
        for func_name, times in self.execution_times.items():
            if times:
                stats[func_name] = {
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'call_count': len(times),
                    'total_time': np.sum(times)
                }
        
        return stats
    
    def batch_process(self, items: List, process_func, batch_size: int = 10):
        """Process items in parallel batches"""
        try:
            results = []
            
            # Split into batches
            batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
            
            # Process batches in parallel
            future_to_batch = {
                self.thread_pool.submit(process_func, batch): batch 
                for batch in batches
            }
            
            for future in future_to_batch:
                try:
                    batch_result = future.result(timeout=30)
                    results.extend(batch_result)
                except Exception as e:
                    logger.warning(f"Batch processing error: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return []

# Utility functions
def create_cache_key(*args, **kwargs) -> str:
    """Create a consistent cache key from arguments"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def decay_weight(days_ago: int, half_life: int = 30) -> float:
    """Calculate exponential decay weight based on time"""
    return np.exp(-days_ago / half_life)
