# backend/personalized/utils.py

"""
CineBrain Personalization Utilities
Vector operations, caching, embeddings, and shared utilities
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
import json
import logging
import hashlib
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import redis
import pickle

logger = logging.getLogger(__name__)

# Telugu-first language priority configuration
LANGUAGE_WEIGHTS = {
    'telugu': 1.0, 'te': 1.0,
    'english': 0.9, 'en': 0.9,
    'hindi': 0.85, 'hi': 0.85,
    'malayalam': 0.75, 'ml': 0.75,
    'kannada': 0.7, 'kn': 0.7,
    'tamil': 0.65, 'ta': 0.65
}

PRIORITY_LANGUAGES = ['telugu', 'english', 'hindi', 'malayalam', 'kannada', 'tamil']

class VectorOperations:
    """Advanced vector operations for content and user embeddings"""
    
    @staticmethod
    def cosine_similarity_batch(vectors_a: np.ndarray, vectors_b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between two batches of vectors efficiently"""
        try:
            # Normalize vectors
            vectors_a_norm = vectors_a / (np.linalg.norm(vectors_a, axis=1, keepdims=True) + 1e-10)
            vectors_b_norm = vectors_b / (np.linalg.norm(vectors_b, axis=1, keepdims=True) + 1e-10)
            
            # Compute similarity
            similarity = np.dot(vectors_a_norm, vectors_b_norm.T)
            return similarity
            
        except Exception as e:
            logger.error(f"Error in batch cosine similarity: {e}")
            return np.zeros((len(vectors_a), len(vectors_b)))
    
    @staticmethod
    def compute_user_content_affinity(user_vector: np.ndarray, 
                                     content_vectors: np.ndarray) -> np.ndarray:
        """Compute affinity scores between user and content vectors"""
        if user_vector.ndim == 1:
            user_vector = user_vector.reshape(1, -1)
        
        # Multiple similarity metrics
        cosine_sim = cosine_similarity(user_vector, content_vectors)[0]
        
        # Euclidean distance (inverted and normalized)
        euclidean_dist = euclidean_distances(user_vector, content_vectors)[0]
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # Weighted combination
        affinity = cosine_sim * 0.7 + euclidean_sim * 0.3
        
        return affinity
    
    @staticmethod
    def normalize_scores(scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize scores using different methods"""
        if len(scores) == 0:
            return scores
        
        if method == 'minmax':
            min_score, max_score = np.min(scores), np.max(scores)
            if max_score == min_score:
                return np.ones_like(scores) * 0.5
            return (scores - min_score) / (max_score - min_score)
        
        elif method == 'zscore':
            mean_score, std_score = np.mean(scores), np.std(scores)
            if std_score == 0:
                return np.zeros_like(scores)
            normalized = (scores - mean_score) / std_score
            # Convert to 0-1 range using sigmoid
            return 1 / (1 + np.exp(-normalized))
        
        elif method == 'robust':
            median_score = np.median(scores)
            mad = np.median(np.abs(scores - median_score))
            if mad == 0:
                return np.ones_like(scores) * 0.5
            return (scores - median_score) / (1.4826 * mad)
        
        return scores
    
    @staticmethod
    def weighted_average_embeddings(embeddings: List[np.ndarray], 
                                   weights: List[float]) -> np.ndarray:
        """Compute weighted average of embeddings"""
        if not embeddings or not weights:
            return np.array([])
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        weighted_embedding = np.zeros_like(embeddings[0])
        for embedding, weight in zip(embeddings, weights):
            weighted_embedding += embedding * weight
        
        return weighted_embedding

class ContentEmbedding:
    """Generate and manage content embeddings for similarity computation"""
    
    def __init__(self, max_features: int = 5000):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=100, random_state=42)
        self.scaler = MinMaxScaler()
        self.fitted = False
        
    def fit_transform(self, content_list: List[Any]) -> np.ndarray:
        """Fit vectorizer and transform content to embeddings"""
        try:
            # Extract text features
            text_features = []
            for content in content_list:
                text = self._extract_content_text(content)
                text_features.append(text)
            
            # TF-IDF transformation
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
            
            # Dimensionality reduction
            reduced_embeddings = self.svd.fit_transform(tfidf_matrix)
            
            # Normalize embeddings
            normalized_embeddings = self.scaler.fit_transform(reduced_embeddings)
            
            self.fitted = True
            logger.info(f"Fitted content embeddings for {len(content_list)} items")
            
            return normalized_embeddings
            
        except Exception as e:
            logger.error(f"Error fitting content embeddings: {e}")
            return np.array([])
    
    def transform(self, content_list: List[Any]) -> np.ndarray:
        """Transform new content to embeddings (requires fitted vectorizer)"""
        if not self.fitted:
            raise ValueError("ContentEmbedding must be fitted before transform")
        
        try:
            text_features = []
            for content in content_list:
                text = self._extract_content_text(content)
                text_features.append(text)
            
            tfidf_matrix = self.tfidf_vectorizer.transform(text_features)
            reduced_embeddings = self.svd.transform(tfidf_matrix)
            normalized_embeddings = self.scaler.transform(reduced_embeddings)
            
            return normalized_embeddings
            
        except Exception as e:
            logger.error(f"Error transforming content embeddings: {e}")
            return np.array([])
    
    def _extract_content_text(self, content: Any) -> str:
        """Extract text features from content object"""
        text_parts = []
        
        # Title (most important)
        if hasattr(content, 'title') and content.title:
            text_parts.extend([content.title] * 3)  # Weight title heavily
        
        # Genres
        if hasattr(content, 'genres') and content.genres:
            try:
                genres = json.loads(content.genres)
                text_parts.extend(genres)
            except:
                pass
        
        # Overview/Description
        if hasattr(content, 'overview') and content.overview:
            text_parts.append(content.overview)
        
        # Languages (for Telugu prioritization)
        if hasattr(content, 'languages') and content.languages:
            try:
                languages = json.loads(content.languages)
                text_parts.extend(languages)
            except:
                pass
        
        # Content type
        if hasattr(content, 'content_type') and content.content_type:
            text_parts.append(content.content_type)
        
        return ' '.join(text_parts)

class CacheManager:
    """Advanced caching for recommendations and user profiles"""
    
    def __init__(self, cache_backend=None, default_ttl: int = 3600):
        self.cache = cache_backend
        self.default_ttl = default_ttl
        self.memory_cache = {}  # Fallback in-memory cache
        self.max_memory_items = 1000
    
    def get(self, key: str, default=None):
        """Get item from cache with fallback"""
        try:
            if self.cache:
                result = self.cache.get(key)
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
        """Set item in cache with fallback"""
        try:
            ttl = ttl or self.default_ttl
            
            if self.cache:
                self.cache.set(key, value, timeout=ttl)
            
            # Also store in memory cache
            self._memory_cache_set(key, value)
            
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
    
    def delete(self, key: str):
        """Delete item from cache"""
        try:
            if self.cache:
                self.cache.delete(key)
            
            if key in self.memory_cache:
                del self.memory_cache[key]
                
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
    
    def _memory_cache_set(self, key: str, value: Any):
        """Set item in memory cache with size limit"""
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest item
            oldest_key = min(self.memory_cache.keys(), 
                           key=lambda k: self.memory_cache[k][1])
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = (value, time.time())
    
    def get_user_profile_cache_key(self, user_id: int) -> str:
        """Generate cache key for user profile"""
        return f"cinebrain:profile:{user_id}"
    
    def get_recommendations_cache_key(self, user_id: int, 
                                    categories: List[str] = None,
                                    **kwargs) -> str:
        """Generate cache key for recommendations"""
        categories_str = ','.join(sorted(categories)) if categories else 'default'
        params_str = ','.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        cache_key = f"cinebrain:recs:{user_id}:{categories_str}:{params_str}"
        
        # Hash if too long
        if len(cache_key) > 200:
            cache_key = f"cinebrain:recs:{hashlib.md5(cache_key.encode()).hexdigest()}"
        
        return cache_key
    
    def get_similarity_cache_key(self, content_id: int, algorithm: str = 'default') -> str:
        """Generate cache key for content similarity"""
        return f"cinebrain:similarity:{content_id}:{algorithm}"

class LanguagePriorityManager:
    """Manage Telugu-first language prioritization across the system"""
    
    @staticmethod
    def apply_language_boost(content_list: List[Any], 
                           user_languages: List[str] = None) -> List[Tuple[Any, float]]:
        """Apply language priority boosting to content"""
        boosted_content = []
        
        for content in content_list:
            base_score = 1.0
            language_boost = LanguagePriorityManager.calculate_language_score(
                content, user_languages
            )
            
            final_score = base_score * (1 + language_boost)
            boosted_content.append((content, final_score))
        
        return boosted_content
    
    @staticmethod
    def calculate_language_score(content: Any, 
                               user_languages: List[str] = None) -> float:
        """Calculate language priority score for content"""
        if not hasattr(content, 'languages') or not content.languages:
            return 0.0
        
        try:
            content_languages = json.loads(content.languages)
        except:
            return 0.0
        
        max_score = 0.0
        
        # Check against priority languages (Telugu first)
        for i, priority_lang in enumerate(PRIORITY_LANGUAGES):
            for lang in content_languages:
                lang_lower = lang.lower() if isinstance(lang, str) else ''
                
                if priority_lang in lang_lower or lang_lower == LANGUAGE_WEIGHTS.get(priority_lang, ''):
                    # Telugu gets highest score, others decrease
                    position_score = 1.0 - (i * 0.1)
                    priority_weight = LANGUAGE_WEIGHTS.get(priority_lang, 0.5)
                    
                    score = position_score * priority_weight
                    max_score = max(max_score, score)
                    break
        
        # User preference bonus
        if user_languages:
            for user_lang in user_languages:
                for content_lang in content_languages:
                    if user_lang.lower() in content_lang.lower():
                        max_score *= 1.2  # 20% boost for user preference
                        break
        
        return min(max_score, 2.0)  # Cap the boost
    
    @staticmethod
    def sort_by_language_priority(content_list: List[Any], 
                                user_languages: List[str] = None) -> List[Any]:
        """Sort content by language priority (Telugu first)"""
        boosted_content = LanguagePriorityManager.apply_language_boost(
            content_list, user_languages
        )
        
        # Sort by boosted score
        boosted_content.sort(key=lambda x: x[1], reverse=True)
        
        return [content for content, _ in boosted_content]

class DataProcessor:
    """Process and clean data for machine learning models"""
    
    @staticmethod
    def extract_user_features(interactions: List[Any], 
                            content_pool: List[Any]) -> Dict[str, Any]:
        """Extract comprehensive user features from interactions"""
        if not interactions:
            return {}
        
        features = {
            'interaction_counts': Counter([i.interaction_type for i in interactions]),
            'rating_stats': DataProcessor._calculate_rating_stats(interactions),
            'temporal_patterns': DataProcessor._analyze_temporal_patterns(interactions),
            'content_preferences': DataProcessor._analyze_content_preferences(interactions, content_pool),
            'engagement_metrics': DataProcessor._calculate_engagement_metrics(interactions)
        }
        
        return features
    
    @staticmethod
    def _calculate_rating_stats(interactions: List[Any]) -> Dict[str, float]:
        """Calculate user rating statistics"""
        ratings = [i.rating for i in interactions if i.rating is not None]
        
        if not ratings:
            return {'count': 0, 'mean': 0, 'std': 0}
        
        return {
            'count': len(ratings),
            'mean': np.mean(ratings),
            'std': np.std(ratings),
            'median': np.median(ratings),
            'min': np.min(ratings),
            'max': np.max(ratings)
        }
    
    @staticmethod
    def _analyze_temporal_patterns(interactions: List[Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in user behavior"""
        if not interactions:
            return {}
        
        hours = [i.timestamp.hour for i in interactions]
        days = [i.timestamp.weekday() for i in interactions]
        
        return {
            'peak_hour': Counter(hours).most_common(1)[0][0] if hours else 12,
            'weekend_ratio': sum(1 for d in days if d >= 5) / len(days) if days else 0,
            'activity_spread': np.std(hours) if hours else 0,
            'most_active_day': Counter(days).most_common(1)[0][0] if days else 0
        }
    
    @staticmethod
    def _analyze_content_preferences(interactions: List[Any], 
                                   content_pool: List[Any]) -> Dict[str, Any]:
        """Analyze content preferences from interactions"""
        content_ids = [i.content_id for i in interactions]
        content_map = {c.id: c for c in content_pool if c.id in content_ids}
        
        genres = []
        content_types = []
        languages = []
        
        for content_id in content_ids:
            content = content_map.get(content_id)
            if not content:
                continue
            
            if content.genres:
                try:
                    genres.extend(json.loads(content.genres))
                except:
                    pass
            
            if content.content_type:
                content_types.append(content.content_type)
            
            if content.languages:
                try:
                    languages.extend(json.loads(content.languages))
                except:
                    pass
        
        return {
            'top_genres': [g for g, _ in Counter(genres).most_common(5)],
            'content_type_dist': dict(Counter(content_types)),
            'top_languages': [l for l, _ in Counter(languages).most_common(3)],
            'genre_diversity': len(set(genres)) / max(len(genres), 1)
        }
    
    @staticmethod
    def _calculate_engagement_metrics(interactions: List[Any]) -> Dict[str, float]:
        """Calculate user engagement metrics"""
        if not interactions:
            return {'total': 0, 'recent': 0, 'consistency': 0}
        
        total_interactions = len(interactions)
        recent_cutoff = datetime.utcnow() - timedelta(days=30)
        recent_interactions = len([i for i in interactions if i.timestamp > recent_cutoff])
        
        # Calculate consistency (activity spread over time)
        dates = [i.timestamp.date() for i in interactions]
        unique_dates = len(set(dates))
        date_span = (max(dates) - min(dates)).days + 1 if len(dates) > 1 else 1
        consistency = unique_dates / date_span
        
        return {
            'total': total_interactions,
            'recent': recent_interactions,
            'consistency': consistency,
            'avg_per_day': total_interactions / max(date_span, 1)
        }

def safe_json_loads(json_str: str, default=None):
    """Safely load JSON string with fallback"""
    try:
        return json.loads(json_str) if json_str else (default or [])
    except (json.JSONDecodeError, TypeError):
        return default or []

def calculate_content_quality_score(content: Any) -> float:
    """Calculate overall quality score for content"""
    score = 0.0
    
    # Rating component (40%)
    if hasattr(content, 'rating') and content.rating:
        score += (content.rating / 10.0) * 0.4
    
    # Vote count component (20%)
    if hasattr(content, 'vote_count') and content.vote_count:
        vote_score = min(content.vote_count / 1000.0, 1.0)
        score += vote_score * 0.2
    
    # Popularity component (20%)
    if hasattr(content, 'popularity') and content.popularity:
        pop_score = min(content.popularity / 100.0, 1.0)
        score += pop_score * 0.2
    
    # Language priority bonus (20%)
    if hasattr(content, 'languages') and content.languages:
        lang_score = LanguagePriorityManager.calculate_language_score(content)
        score += (lang_score / 2.0) * 0.2  # Normalize to 0-1 range
    
    return min(score, 1.0)