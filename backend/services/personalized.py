# backend/services/personalized.py

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
import json
import logging
import hashlib
import pickle
import redis
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import heapq
import random
from scipy import sparse
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss  # For approximate nearest neighbor search
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models and Enums
# ============================================================================

class InteractionType(Enum):
    """Types of user interactions with weights"""
    VIEW = ("view", 1.0)
    LIKE = ("like", 3.0)
    FAVORITE = ("favorite", 5.0)
    WATCHLIST = ("watchlist", 2.0)
    RATING = ("rating", 4.0)
    SEARCH = ("search", 0.5)
    CLICK = ("click", 0.8)
    COMPLETE = ("complete", 4.5)
    SKIP = ("skip", -2.0)
    DISLIKE = ("dislike", -3.0)
    
    def __init__(self, value, weight):
        self._value_ = value
        self.weight = weight

class ContentType(Enum):
    """Content types"""
    MOVIE = "movie"
    TV_SHOW = "tv"
    ANIME = "anime"
    DOCUMENTARY = "documentary"
    SPECIAL = "special"

class TimeContext(Enum):
    """Time of day contexts"""
    MORNING = (6, 12)  # 6 AM - 12 PM
    AFTERNOON = (12, 17)  # 12 PM - 5 PM
    EVENING = (17, 21)  # 5 PM - 9 PM
    NIGHT = (21, 24)  # 9 PM - 12 AM
    LATE_NIGHT = (0, 6)  # 12 AM - 6 AM

@dataclass
class UserActivity:
    """User activity tracking"""
    user_id: int
    content_id: int
    interaction_type: InteractionType
    timestamp: datetime
    duration: Optional[float] = None  # Watch duration in seconds
    completion_rate: Optional[float] = None  # 0-1
    rating: Optional[float] = None
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    device_type: Optional[str] = None

@dataclass
class UserPreferences:
    """User preference profile"""
    user_id: int
    preferred_genres: List[str] = field(default_factory=list)
    preferred_languages: List[str] = field(default_factory=list)
    preferred_content_types: List[ContentType] = field(default_factory=list)
    blacklisted_genres: List[str] = field(default_factory=list)
    viewing_time_preferences: Dict[TimeContext, float] = field(default_factory=dict)
    avg_session_duration: float = 0.0
    diversity_preference: float = 0.5  # 0-1, higher means more diverse
    exploration_rate: float = 0.2  # 0-1, exploration vs exploitation
    maturity_rating: str = "PG-13"
    subtitle_preference: Optional[str] = None  # "subbed", "dubbed", None

@dataclass
class ContentFeatures:
    """Content feature representation"""
    content_id: int
    title: str
    content_type: ContentType
    genres: List[str]
    languages: List[str]
    cast: List[str]
    director: Optional[str]
    release_date: datetime
    runtime: float  # in minutes
    rating: float
    popularity: float
    embeddings: Optional[np.ndarray] = None
    tags: List[str] = field(default_factory=list)
    mood: Optional[str] = None
    franchise: Optional[str] = None

@dataclass
class RecommendationContext:
    """Context for generating recommendations"""
    user_id: int
    timestamp: datetime
    device_type: Optional[str] = None
    location: Optional[str] = None
    mood: Optional[str] = None
    social_context: Optional[List[int]] = None  # Friend IDs watching together
    session_length_estimate: Optional[float] = None
    explicit_filters: Optional[Dict[str, Any]] = None

# ============================================================================
# Neural Collaborative Filtering Model
# ============================================================================

class NCFModel(nn.Module):
    """Neural Collaborative Filtering model for personalized recommendations"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64,
                 hidden_layers: List[int] = [128, 64, 32]):
        super(NCFModel, self).__init__()
        
        # User and item embeddings for MF pathway
        self.user_embedding_mf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mf = nn.Embedding(num_items, embedding_dim)
        
        # User and item embeddings for MLP pathway
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_size = embedding_dim * 2
        for hidden_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        self.prediction = nn.Linear(hidden_layers[-1] + embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.normal_(self.user_embedding_mf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # MF pathway
        user_mf = self.user_embedding_mf(user_ids)
        item_mf = self.item_embedding_mf(item_ids)
        mf_output = user_mf * item_mf
        
        # MLP pathway
        user_mlp = self.user_embedding_mlp(user_ids)
        item_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate and predict
        combined = torch.cat([mf_output, mlp_output], dim=-1)
        prediction = self.sigmoid(self.prediction(combined))
        
        return prediction.squeeze()

class InteractionDataset(Dataset):
    """Dataset for training NCF model"""
    
    def __init__(self, interactions: List[Tuple[int, int, float]]):
        self.interactions = interactions
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id, item_id, score = self.interactions[idx]
        return torch.tensor(user_id), torch.tensor(item_id), torch.tensor(score, dtype=torch.float32)

# ============================================================================
# Matrix Factorization
# ============================================================================

class MatrixFactorization:
    """Optimized matrix factorization for collaborative filtering"""
    
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01,
                 regularization: float = 0.01, n_epochs: int = 20):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = 0
    
    def fit(self, interaction_matrix: sparse.csr_matrix):
        """Train matrix factorization model"""
        n_users, n_items = interaction_matrix.shape
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.01, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        # Calculate global mean
        self.global_mean = interaction_matrix.data.mean()
        
        # SGD optimization
        for epoch in range(self.n_epochs):
            for user_id, item_id in zip(*interaction_matrix.nonzero()):
                rating = interaction_matrix[user_id, item_id]
                
                # Predict
                prediction = self.predict_single(user_id, item_id)
                error = rating - prediction
                
                # Update biases
                self.user_biases[user_id] += self.learning_rate * (error - self.regularization * self.user_biases[user_id])
                self.item_biases[item_id] += self.learning_rate * (error - self.regularization * self.item_biases[item_id])
                
                # Update factors
                user_factor = self.user_factors[user_id]
                item_factor = self.item_factors[item_id]
                
                self.user_factors[user_id] += self.learning_rate * (error * item_factor - self.regularization * user_factor)
                self.item_factors[item_id] += self.learning_rate * (error * user_factor - self.regularization * item_factor)
    
    def predict_single(self, user_id: int, item_id: int) -> float:
        """Predict rating for single user-item pair"""
        if self.user_factors is None:
            return self.global_mean
        
        prediction = self.global_mean + self.user_biases[user_id] + self.item_biases[item_id]
        prediction += np.dot(self.user_factors[user_id], self.item_factors[item_id])
        return prediction
    
    def predict_all(self, user_id: int) -> np.ndarray:
        """Predict ratings for all items for a user"""
        if self.user_factors is None:
            return np.full(len(self.item_factors), self.global_mean)
        
        predictions = self.global_mean + self.user_biases[user_id] + self.item_biases
        predictions += np.dot(self.user_factors[user_id], self.item_factors.T)
        return predictions

# ============================================================================
# Content-Based Filtering
# ============================================================================

class ContentBasedEngine:
    """Advanced content-based filtering engine"""
    
    def __init__(self, content_features: List[ContentFeatures]):
        self.content_features = {cf.content_id: cf for cf in content_features}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.genre_encoder = None
        self.content_embeddings = None
        self._build_content_profiles()
    
    def _build_content_profiles(self):
        """Build content profiles using various features"""
        if not self.content_features:
            return
        
        # Create text descriptions
        descriptions = []
        for content in self.content_features.values():
            desc = f"{' '.join(content.genres)} {' '.join(content.tags)} {content.mood or ''}"
            descriptions.append(desc)
        
        # TF-IDF on descriptions
        if descriptions:
            self.content_embeddings = self.tfidf_vectorizer.fit_transform(descriptions).toarray()
    
    def get_similar_content(self, content_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get similar content based on content features"""
        if content_id not in self.content_features or self.content_embeddings is None:
            return []
        
        # Get content index
        content_ids = list(self.content_features.keys())
        idx = content_ids.index(content_id)
        
        # Calculate similarities
        content_vec = self.content_embeddings[idx].reshape(1, -1)
        similarities = cosine_similarity(content_vec, self.content_embeddings)[0]
        
        # Get top similar items (excluding self)
        similar_indices = similarities.argsort()[-n_recommendations-1:-1][::-1]
        
        recommendations = []
        for sim_idx in similar_indices:
            if sim_idx != idx:
                recommendations.append((content_ids[sim_idx], similarities[sim_idx]))
        
        return recommendations[:n_recommendations]
    
    def get_user_profile(self, user_activities: List[UserActivity]) -> np.ndarray:
        """Build user profile based on interaction history"""
        if not user_activities or self.content_embeddings is None:
            return np.zeros(self.content_embeddings.shape[1])
        
        # Weight interactions
        weighted_embeddings = []
        weights = []
        
        content_ids = list(self.content_features.keys())
        for activity in user_activities:
            if activity.content_id in self.content_features:
                idx = content_ids.index(activity.content_id)
                weight = activity.interaction_type.weight
                
                # Adjust weight based on recency
                days_ago = (datetime.now() - activity.timestamp).days
                recency_weight = np.exp(-days_ago / 30)  # Exponential decay
                
                final_weight = weight * recency_weight
                weighted_embeddings.append(self.content_embeddings[idx] * final_weight)
                weights.append(final_weight)
        
        if not weighted_embeddings:
            return np.zeros(self.content_embeddings.shape[1])
        
        # Weighted average
        user_profile = np.sum(weighted_embeddings, axis=0) / np.sum(weights)
        return user_profile
    
    def recommend_for_user(self, user_profile: np.ndarray, n_recommendations: int = 10,
                           exclude_ids: Set[int] = None) -> List[Tuple[int, float]]:
        """Recommend content based on user profile"""
        if self.content_embeddings is None:
            return []
        
        exclude_ids = exclude_ids or set()
        
        # Calculate similarities
        similarities = cosine_similarity(user_profile.reshape(1, -1), self.content_embeddings)[0]
        
        # Get recommendations
        content_ids = list(self.content_features.keys())
        recommendations = []
        
        for idx, sim in enumerate(similarities):
            content_id = content_ids[idx]
            if content_id not in exclude_ids:
                recommendations.append((content_id, sim))
        
        # Sort and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

# ============================================================================
# Hybrid Recommendation Engine
# ============================================================================

class HybridRecommendationEngine:
    """Main hybrid recommendation engine combining multiple techniques"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.ncf_model: Optional[NCFModel] = None
        self.matrix_factorization: Optional[MatrixFactorization] = None
        self.content_engine: Optional[ContentBasedEngine] = None
        self.user_embeddings_index: Optional[faiss.IndexFlatL2] = None
        self.item_embeddings_index: Optional[faiss.IndexFlatL2] = None
        
        # Caches
        self.user_cache = {}
        self.recommendation_cache = {}
        self.similarity_cache = {}
        
        # Configuration
        self.ensemble_weights = {
            'ncf': 0.4,
            'mf': 0.2,
            'content': 0.2,
            'popularity': 0.1,
            'trending': 0.1
        }
        
        # Multi-armed bandit for exploration
        self.bandit_arms = defaultdict(lambda: {'successes': 1, 'trials': 2})
        
        # Metrics tracking
        self.metrics = {
            'recommendations_served': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'model_updates': 0
        }
    
    def initialize(self, user_activities: List[UserActivity],
                  content_features: List[ContentFeatures]):
        """Initialize all recommendation components"""
        logger.info("Initializing hybrid recommendation engine...")
        
        # Build interaction matrix
        interaction_matrix = self._build_interaction_matrix(user_activities)
        
        # Initialize content-based engine
        self.content_engine = ContentBasedEngine(content_features)
        
        # Initialize matrix factorization
        self.matrix_factorization = MatrixFactorization()
        if interaction_matrix is not None:
            self.matrix_factorization.fit(interaction_matrix)
        
        # Initialize NCF model (would be pre-trained in production)
        # self._initialize_ncf(interaction_matrix)
        
        # Build FAISS indices for fast similarity search
        self._build_faiss_indices(content_features)
        
        logger.info("Hybrid recommendation engine initialized successfully")
    
    def _build_interaction_matrix(self, user_activities: List[UserActivity]) -> Optional[sparse.csr_matrix]:
        """Build sparse interaction matrix from user activities"""
        if not user_activities:
            return None
        
        # Get unique users and items
        users = list(set(activity.user_id for activity in user_activities))
        items = list(set(activity.content_id for activity in user_activities))
        
        user_map = {user: idx for idx, user in enumerate(users)}
        item_map = {item: idx for idx, item in enumerate(items)}
        
        # Build matrix
        row_indices = []
        col_indices = []
        data = []
        
        for activity in user_activities:
            if activity.user_id in user_map and activity.content_id in item_map:
                row_indices.append(user_map[activity.user_id])
                col_indices.append(item_map[activity.content_id])
                
                # Calculate interaction score
                score = activity.interaction_type.weight
                if activity.rating:
                    score *= (activity.rating / 5.0)
                if activity.completion_rate:
                    score *= activity.completion_rate
                
                data.append(score)
        
        matrix = sparse.csr_matrix((data, (row_indices, col_indices)),
                                  shape=(len(users), len(items)))
        
        return matrix
    
    def _build_faiss_indices(self, content_features: List[ContentFeatures]):
        """Build FAISS indices for fast similarity search"""
        if not content_features or not self.content_engine:
            return
        
        embeddings = self.content_engine.content_embeddings
        if embeddings is not None and len(embeddings) > 0:
            # Normalize embeddings
            embeddings = embeddings.astype('float32')
            
            # Build index
            dimension = embeddings.shape[1]
            self.item_embeddings_index = faiss.IndexFlatL2(dimension)
            self.item_embeddings_index.add(embeddings)
    
    def get_recommendations(self, context: RecommendationContext,
                           n_recommendations: int = 20) -> List[Dict[str, Any]]:
        """Get personalized recommendations for a user"""
        
        # Check cache first
        cache_key = self._get_cache_key(context)
        if cache_key in self.recommendation_cache:
            self.metrics['cache_hits'] += 1
            cached_recs = self.recommendation_cache[cache_key]
            # Add small random variation to cached results
            return self._add_exploration(cached_recs, context)
        
        self.metrics['cache_misses'] += 1
        
        # Get user history
        user_activities = self._get_user_activities(context.user_id)
        
        # Generate recommendations from different sources
        recommendations = {}
        
        # Content-based recommendations
        if self.content_engine and user_activities:
            user_profile = self.content_engine.get_user_profile(user_activities)
            viewed_ids = set(activity.content_id for activity in user_activities)
            content_recs = self.content_engine.recommend_for_user(
                user_profile, n_recommendations * 2, viewed_ids
            )
            recommendations['content'] = content_recs
        
        # Collaborative filtering (Matrix Factorization)
        if self.matrix_factorization:
            mf_scores = self.matrix_factorization.predict_all(context.user_id)
            mf_recs = [(idx, score) for idx, score in enumerate(mf_scores)]
            mf_recs.sort(key=lambda x: x[1], reverse=True)
            recommendations['mf'] = mf_recs[:n_recommendations * 2]
        
        # Popularity-based (fallback)
        popular_recs = self._get_popular_content(n_recommendations)
        recommendations['popularity'] = popular_recs
        
        # Trending content
        trending_recs = self._get_trending_content(context, n_recommendations)
        recommendations['trending'] = trending_recs
        
        # Combine recommendations using ensemble
        final_recommendations = self._ensemble_recommendations(
            recommendations, context, n_recommendations
        )
        
        # Apply business rules and filters
        final_recommendations = self._apply_business_rules(
            final_recommendations, context
        )
        
        # Cache results
        self.recommendation_cache[cache_key] = final_recommendations
        
        # Update metrics
        self.metrics['recommendations_served'] += 1
        
        return final_recommendations
    
    def _ensemble_recommendations(self, recommendations: Dict[str, List[Tuple[int, float]]],
                                 context: RecommendationContext,
                                 n_recommendations: int) -> List[Dict[str, Any]]:
        """Combine recommendations from different sources"""
        
        # Score aggregation
        item_scores = defaultdict(float)
        item_sources = defaultdict(list)
        
        for source, items in recommendations.items():
            if not items:
                continue
            
            weight = self.ensemble_weights.get(source, 0.1)
            
            # Apply multi-armed bandit for dynamic weight adjustment
            bandit_weight = self._get_bandit_weight(source)
            weight = weight * 0.7 + bandit_weight * 0.3
            
            for item_id, score in items:
                item_scores[item_id] += score * weight
                item_sources[item_id].append(source)
        
        # Sort by combined score
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format recommendations
        final_recommendations = []
        for item_id, score in sorted_items[:n_recommendations]:
            rec = {
                'content_id': item_id,
                'score': score,
                'sources': item_sources[item_id],
                'explanation': self._generate_explanation(item_id, item_sources[item_id], context),
                'confidence': self._calculate_confidence(score, item_sources[item_id]),
                'timestamp': datetime.now().isoformat()
            }
            final_recommendations.append(rec)
        
        return final_recommendations
    
    def _get_bandit_weight(self, source: str) -> float:
        """Get weight using Thompson sampling for multi-armed bandit"""
        arm = self.bandit_arms[source]
        # Thompson sampling
        sample = np.random.beta(arm['successes'], arm['trials'] - arm['successes'] + 1)
        return sample
    
    def update_bandit(self, source: str, reward: bool):
        """Update bandit arm based on user feedback"""
        self.bandit_arms[source]['trials'] += 1
        if reward:
            self.bandit_arms[source]['successes'] += 1
    
    def _add_exploration(self, recommendations: List[Dict[str, Any]],
                        context: RecommendationContext) -> List[Dict[str, Any]]:
        """Add exploration to recommendations"""
        
        # Get user's exploration rate
        user_prefs = self._get_user_preferences(context.user_id)
        exploration_rate = user_prefs.exploration_rate if user_prefs else 0.2
        
        n_explore = int(len(recommendations) * exploration_rate)
        if n_explore == 0:
            return recommendations
        
        # Get random diverse content
        explore_items = self._get_diverse_content(n_explore, context)
        
        # Replace some recommendations with exploration items
        for i, item in enumerate(explore_items):
            if i < len(recommendations):
                idx = random.randint(len(recommendations) // 2, len(recommendations) - 1)
                recommendations[idx] = {
                    'content_id': item['content_id'],
                    'score': item['score'],
                    'sources': ['exploration'],
                    'explanation': 'Recommended for discovery',
                    'confidence': 0.3,
                    'timestamp': datetime.now().isoformat()
                }
        
        return recommendations
    
    def _get_diverse_content(self, n_items: int, context: RecommendationContext) -> List[Dict[str, Any]]:
        """Get diverse content for exploration"""
        # This would fetch diverse content from different genres/languages
        # Implementation depends on your content catalog
        return []
    
    def _generate_explanation(self, item_id: int, sources: List[str],
                             context: RecommendationContext) -> str:
        """Generate explanation for why item was recommended"""
        
        explanations = []
        
        if 'content' in sources:
            explanations.append("similar to your viewing history")
        if 'mf' in sources or 'ncf' in sources:
            explanations.append("users like you also watched")
        if 'popularity' in sources:
            explanations.append("popular right now")
        if 'trending' in sources:
            explanations.append("trending in your area")
        
        if not explanations:
            return "recommended for you"
        
        return f"Recommended because: {', '.join(explanations)}"
    
    def _calculate_confidence(self, score: float, sources: List[str]) -> float:
        """Calculate confidence score for recommendation"""
        
        # Base confidence from score
        confidence = min(score, 1.0)
        
        # Boost confidence if multiple sources agree
        if len(sources) > 1:
            confidence = min(confidence * 1.2, 1.0)
        
        # Higher confidence for NCF/ML models
        if 'ncf' in sources:
            confidence = min(confidence * 1.3, 1.0)
        
        return round(confidence, 3)
    
    def _apply_business_rules(self, recommendations: List[Dict[str, Any]],
                             context: RecommendationContext) -> List[Dict[str, Any]]:
        """Apply business rules and filters"""
        
        filtered = []
        
        for rec in recommendations:
            # Apply filters based on context
            if context.explicit_filters:
                # Check content type filter
                if 'content_type' in context.explicit_filters:
                    # Would need to fetch content metadata to filter
                    pass
                
                # Check language filter
                if 'language' in context.explicit_filters:
                    # Would need to fetch content metadata to filter
                    pass
            
            # Check maturity rating
            # Implementation depends on content metadata
            
            filtered.append(rec)
        
        # Ensure diversity
        filtered = self._ensure_diversity(filtered)
        
        return filtered
    
    def _ensure_diversity(self, recommendations: List[Dict[str, Any]],
                         min_diversity: float = 0.3) -> List[Dict[str, Any]]:
        """Ensure recommendations are diverse"""
        
        # This would check genre/type diversity
        # For now, returning as-is
        return recommendations
    
    def _get_cache_key(self, context: RecommendationContext) -> str:
        """Generate cache key for recommendations"""
        
        key_parts = [
            str(context.user_id),
            context.timestamp.strftime('%Y%m%d%H'),  # Hour-level caching
            context.device_type or 'unknown',
            context.mood or 'neutral'
        ]
        
        key_str = ':'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_user_activities(self, user_id: int) -> List[UserActivity]:
        """Get user activities from storage"""
        # This would fetch from database
        # Placeholder implementation
        return []
    
    def _get_user_preferences(self, user_id: int) -> Optional[UserPreferences]:
        """Get user preferences from storage"""
        # This would fetch from database or cache
        # Placeholder implementation
        return None
    
    def _get_popular_content(self, n_items: int) -> List[Tuple[int, float]]:
        """Get popular content"""
        # This would fetch from a popularity tracking service
        # Placeholder implementation
        return []
    
    def _get_trending_content(self, context: RecommendationContext,
                             n_items: int) -> List[Tuple[int, float]]:
        """Get trending content based on context"""
        # This would fetch location-based or time-based trending content
        # Placeholder implementation
        return []
    
    def update_user_embedding(self, user_id: int, embedding: np.ndarray):
        """Update user embedding in real-time"""
        if self.redis_client:
            key = f"user_embedding:{user_id}"
            self.redis_client.setex(key, 3600, embedding.tobytes())
    
    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Get user embedding from cache"""
        if self.redis_client:
            key = f"user_embedding:{user_id}"
            data = self.redis_client.get(key)
            if data:
                return np.frombuffer(data, dtype=np.float32)
        return None

# ============================================================================
# Real-time Recommendation Service
# ============================================================================

class PersonalizedRecommendationService:
    """Main service for personalized recommendations"""
    
    def __init__(self, db_connection, redis_client: Optional[redis.Redis] = None):
        self.db = db_connection
        self.redis_client = redis_client
        self.hybrid_engine = HybridRecommendationEngine(redis_client)
        
        # Background processors
        self.activity_processor = ActivityProcessor(db_connection, redis_client)
        self.model_trainer = ModelTrainer(self.hybrid_engine)
        
        # Thread pools
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        # Start activity processing
        self.executor.submit(self.activity_processor.start)
        
        # Start model training
        self.executor.submit(self.model_trainer.start_continuous_training)
        
        # Start monitoring
        self.executor.submit(self.performance_monitor.start)
    
    async def get_recommendations(self, user_id: int, 
                                 request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized recommendations for a user"""
        
        start_time = datetime.now()
        
        try:
            # Build recommendation context
            context = RecommendationContext(
                user_id=user_id,
                timestamp=datetime.now(),
                device_type=request_context.get('device_type'),
                location=request_context.get('location'),
                mood=request_context.get('mood'),
                explicit_filters=request_context.get('filters')
            )
            
            # Get recommendations
            recommendations = self.hybrid_engine.get_recommendations(
                context,
                n_recommendations=request_context.get('limit', 20)
            )
            
            # Track performance
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.performance_monitor.record_request(user_id, latency, len(recommendations))
            
            # Format response
            response = {
                'status': 'success',
                'user_id': user_id,
                'recommendations': recommendations,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'latency_ms': latency,
                    'algorithm_version': '2.0',
                    'personalization_level': self._calculate_personalization_level(user_id),
                    'cache_hit': recommendations[0].get('cached', False) if recommendations else False
                }
            }
            
            # Async log for analytics
            self.executor.submit(self._log_recommendation_event, user_id, recommendations)
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting recommendations for user {user_id}: {e}")
            
            # Fallback to popular content
            fallback = self._get_fallback_recommendations(user_id, request_context.get('limit', 20))
            
            return {
                'status': 'fallback',
                'user_id': user_id,
                'recommendations': fallback,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'fallback_reason': 'recommendation_engine_error'
                }
            }
    
    def record_interaction(self, interaction: UserActivity):
        """Record user interaction"""
        # Queue for processing
        self.activity_processor.queue_activity(interaction)
        
        # Update user embedding in real-time if significant interaction
        if interaction.interaction_type in [InteractionType.RATING, InteractionType.FAVORITE]:
            self.executor.submit(self._update_user_embedding_realtime, interaction)
    
    def _update_user_embedding_realtime(self, interaction: UserActivity):
        """Update user embedding in real-time"""
        try:
            # Get current embedding
            current_embedding = self.hybrid_engine.get_user_embedding(interaction.user_id)
            
            # Update embedding based on interaction
            # This is a simplified version - would use more sophisticated update in production
            if current_embedding is not None:
                # Placeholder for embedding update logic
                pass
            
        except Exception as e:
            logger.error(f"Error updating user embedding: {e}")
    
    def _calculate_personalization_level(self, user_id: int) -> str:
        """Calculate personalization level for user"""
        # Check user activity count
        # This would query from database
        activity_count = 0  # Placeholder
        
        if activity_count < 5:
            return 'cold_start'
        elif activity_count < 20:
            return 'warming_up'
        elif activity_count < 100:
            return 'personalized'
        else:
            return 'highly_personalized'
    
    def _get_fallback_recommendations(self, user_id: int, limit: int) -> List[Dict[str, Any]]:
        """Get fallback recommendations"""
        # This would return popular/trending content
        return []
    
    def _log_recommendation_event(self, user_id: int, recommendations: List[Dict[str, Any]]):
        """Log recommendation event for analytics"""
        try:
            event = {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'recommendations': [r['content_id'] for r in recommendations[:10]],
                'algorithm_version': '2.0'
            }
            
            # Log to analytics service
            # This would send to your analytics pipeline
            
        except Exception as e:
            logger.error(f"Error logging recommendation event: {e}")

# ============================================================================
# Supporting Components
# ============================================================================

class ActivityProcessor:
    """Process user activities in background"""
    
    def __init__(self, db_connection, redis_client: Optional[redis.Redis] = None):
        self.db = db_connection
        self.redis_client = redis_client
        self.activity_queue = deque(maxlen=10000)
        self.running = False
    
    def queue_activity(self, activity: UserActivity):
        """Queue activity for processing"""
        self.activity_queue.append(activity)
    
    def start(self):
        """Start processing activities"""
        self.running = True
        
        while self.running:
            if self.activity_queue:
                batch = []
                
                # Process in batches
                while self.activity_queue and len(batch) < 100:
                    batch.append(self.activity_queue.popleft())
                
                self._process_batch(batch)
            
            # Sleep briefly
            threading.Event().wait(0.1)
    
    def _process_batch(self, activities: List[UserActivity]):
        """Process batch of activities"""
        try:
            # Store in database
            # Update user profiles
            # Update statistics
            pass
        except Exception as e:
            logger.error(f"Error processing activity batch: {e}")
    
    def stop(self):
        """Stop processing"""
        self.running = False

class ModelTrainer:
    """Background model training"""
    
    def __init__(self, hybrid_engine: HybridRecommendationEngine):
        self.hybrid_engine = hybrid_engine
        self.training_interval = 3600  # Train every hour
        self.running = False
    
    def start_continuous_training(self):
        """Start continuous model training"""
        self.running = True
        
        while self.running:
            try:
                # Retrain models periodically
                self._retrain_models()
                
                # Sleep until next training
                threading.Event().wait(self.training_interval)
                
            except Exception as e:
                logger.error(f"Error in model training: {e}")
    
    def _retrain_models(self):
        """Retrain recommendation models"""
        logger.info("Starting model retraining...")
        
        # Fetch recent interactions
        # Retrain matrix factorization
        # Update content embeddings
        # Fine-tune NCF model if applicable
        
        logger.info("Model retraining completed")
    
    def stop(self):
        """Stop training"""
        self.running = False

class PerformanceMonitor:
    """Monitor recommendation system performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.running = False
    
    def record_request(self, user_id: int, latency: float, n_recommendations: int):
        """Record request metrics"""
        self.metrics['latencies'].append(latency)
        self.metrics['recommendation_counts'].append(n_recommendations)
        self.metrics['timestamps'].append(datetime.now())
    
    def start(self):
        """Start monitoring"""
        self.running = True
        
        while self.running:
            # Calculate and log metrics periodically
            if len(self.metrics['latencies']) > 0:
                avg_latency = np.mean(self.metrics['latencies'][-1000:])
                p95_latency = np.percentile(self.metrics['latencies'][-1000:], 95)
                
                logger.info(f"Performance metrics - Avg latency: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")
            
            # Clear old metrics periodically
            if len(self.metrics['latencies']) > 10000:
                for key in self.metrics:
                    self.metrics[key] = self.metrics[key][-5000:]
            
            threading.Event().wait(60)  # Log every minute
    
    def stop(self):
        """Stop monitoring"""
        self.running = False

# ============================================================================
# API Integration
# ============================================================================

def create_recommendation_service(app, db, cache_client=None):
    """Factory function to create recommendation service"""
    
    # Initialize Redis client if available
    redis_client = None
    if cache_client and hasattr(cache_client, 'cache'):
        redis_client = cache_client.cache._cache
    
    # Create service
    service = PersonalizedRecommendationService(db, redis_client)
    
    # Initialize with data
    # This would load initial data from database
    
    return service

# Export main components
__all__ = [
    'PersonalizedRecommendationService',
    'HybridRecommendationEngine',
    'UserActivity',
    'UserPreferences',
    'ContentFeatures',
    'RecommendationContext',
    'InteractionType',
    'ContentType',
    'create_recommendation_service'
]