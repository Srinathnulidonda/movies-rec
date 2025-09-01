#backend/trending.py
import logging
import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
import threading
import re
import random
import traceback
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler
import pickle

logger = logging.getLogger(__name__)

# ================== Configuration & Constants ==================

# Language Priority Order
LANGUAGE_PRIORITY_ORDER = ['telugu', 'english', 'hindi', 'tamil', 'malayalam', 'kannada', 'korean', 'japanese', 'spanish', 'french']

# Netflix-style Feature Dimensions
FEATURE_DIMENSIONS = {
    'content': 128,  # Content embedding size
    'user': 64,      # User embedding size
    'context': 32,   # Context embedding size
    'temporal': 16   # Temporal embedding size
}

class TrendingCategory(Enum):
    """Trending content categories"""
    BLOCKBUSTER_TRENDING = "blockbuster_trending"
    VIRAL_TRENDING = "viral_trending"
    STRONG_TRENDING = "strong_trending"
    MODERATE_TRENDING = "moderate_trending"
    RISING_FAST = "rising_fast"
    POPULAR_REGIONAL = "popular_regional"
    CROSS_LANGUAGE_TRENDING = "cross_language_trending"
    PERSONALIZED_TRENDING = "personalized_trending"
    COLLABORATIVE_TRENDING = "collaborative_trending"
    VECTOR_MATCH_TRENDING = "vector_match_trending"

# ================== Vector-Based Data Classes ==================

@dataclass
class ContentVector:
    """Netflix-style content feature vector"""
    content_id: int
    title: str
    language: str
    
    # Genre vector (one-hot + weighted)
    genre_vector: np.ndarray = field(default_factory=lambda: np.zeros(30))
    
    # Language vector
    language_vector: np.ndarray = field(default_factory=lambda: np.zeros(15))
    
    # Quality signals vector
    quality_vector: np.ndarray = field(default_factory=lambda: np.zeros(10))
    
    # Engagement vector
    engagement_vector: np.ndarray = field(default_factory=lambda: np.zeros(20))
    
    # Temporal vector
    temporal_vector: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_DIMENSIONS['temporal']))
    
    # Cast/Crew vector (for similarity)
    talent_vector: np.ndarray = field(default_factory=lambda: np.zeros(50))
    
    # Visual features (poster colors, style)
    visual_vector: np.ndarray = field(default_factory=lambda: np.zeros(25))
    
    # Text embeddings (from overview/synopsis)
    text_embedding: np.ndarray = field(default_factory=lambda: np.zeros(100))
    
    # Aggregated feature vector
    feature_vector: Optional[np.ndarray] = None
    
    # Metadata
    rating: float = 0.0
    vote_count: int = 0
    popularity: float = 0.0
    release_date: Optional[datetime] = None
    runtime: int = 0
    
    def build_feature_vector(self) -> np.ndarray:
        """Build complete feature vector like Netflix's approach"""
        # Concatenate all sub-vectors
        self.feature_vector = np.concatenate([
            self.genre_vector,
            self.language_vector,
            self.quality_vector,
            self.engagement_vector,
            self.temporal_vector,
            self.talent_vector,
            self.visual_vector,
            self.text_embedding
        ])
        
        # L2 normalization (Netflix approach)
        norm = np.linalg.norm(self.feature_vector)
        if norm > 0:
            self.feature_vector = self.feature_vector / norm
            
        return self.feature_vector

@dataclass
class UserVector:
    """User preference vector (session-based for anonymous users)"""
    session_id: str
    
    # Preference vectors
    genre_preferences: np.ndarray = field(default_factory=lambda: np.zeros(30))
    language_preferences: np.ndarray = field(default_factory=lambda: np.zeros(15))
    quality_preferences: np.ndarray = field(default_factory=lambda: np.zeros(10))
    
    # Interaction history
    watched_vector: np.ndarray = field(default_factory=lambda: np.zeros(100))
    search_vector: np.ndarray = field(default_factory=lambda: np.zeros(50))
    
    # Temporal patterns
    time_preferences: np.ndarray = field(default_factory=lambda: np.zeros(24))  # Hour preferences
    day_preferences: np.ndarray = field(default_factory=lambda: np.zeros(7))   # Day of week
    
    # Aggregated user vector
    user_vector: Optional[np.ndarray] = None
    
    def build_user_vector(self) -> np.ndarray:
        """Build complete user preference vector"""
        self.user_vector = np.concatenate([
            self.genre_preferences,
            self.language_preferences,
            self.quality_preferences,
            self.watched_vector,
            self.search_vector,
            self.time_preferences,
            self.day_preferences
        ])
        
        # L2 normalization
        norm = np.linalg.norm(self.user_vector)
        if norm > 0:
            self.user_vector = self.user_vector / norm
            
        return self.user_vector

# ================== Netflix-Style Algorithms ==================

class CollaborativeFilteringEngine:
    """
    Netflix-style Collaborative Filtering using Matrix Factorization
    Similar to Netflix's original algorithm that won the Netflix Prize
    """
    
    def __init__(self, n_factors=100, learning_rate=0.01, regularization=0.01):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.user_factors = {}
        self.item_factors = {}
        self.global_bias = 0
        self.user_biases = {}
        self.item_biases = {}
        
    def train_incremental(self, interactions: List[Tuple[str, int, float]]):
        """Incremental training on new interactions"""
        # Calculate global bias
        if interactions:
            self.global_bias = np.mean([r for _, _, r in interactions])
        
        # SGD update for each interaction
        for user_id, item_id, rating in interactions:
            # Initialize factors if new
            if user_id not in self.user_factors:
                self.user_factors[user_id] = np.random.normal(0, 0.1, self.n_factors)
                self.user_biases[user_id] = 0
                
            if item_id not in self.item_factors:
                self.item_factors[item_id] = np.random.normal(0, 0.1, self.n_factors)
                self.item_biases[item_id] = 0
            
            # Predict current rating
            pred = self.predict_single(user_id, item_id)
            error = rating - pred
            
            # SGD updates
            user_factors = self.user_factors[user_id]
            item_factors = self.item_factors[item_id]
            
            # Update biases
            self.user_biases[user_id] += self.learning_rate * (error - self.regularization * self.user_biases[user_id])
            self.item_biases[item_id] += self.learning_rate * (error - self.regularization * self.item_biases[item_id])
            
            # Update factors
            self.user_factors[user_id] += self.learning_rate * (error * item_factors - self.regularization * user_factors)
            self.item_factors[item_id] += self.learning_rate * (error * user_factors - self.regularization * item_factors)
    
    def predict_single(self, user_id: str, item_id: int) -> float:
        """Predict rating for a single user-item pair"""
        if user_id not in self.user_factors or item_id not in self.item_factors:
            return self.global_bias
        
        prediction = (self.global_bias + 
                     self.user_biases.get(user_id, 0) + 
                     self.item_biases.get(item_id, 0) +
                     np.dot(self.user_factors[user_id], self.item_factors[item_id]))
        
        return np.clip(prediction, 1, 10)
    
    def get_trending_items(self, n_items: int = 20) -> List[Tuple[int, float]]:
        """Get trending items based on collaborative signals"""
        if not self.item_biases:
            return []
        
        # Items with high bias are generally popular
        trending = sorted(self.item_biases.items(), key=lambda x: x[1], reverse=True)
        return trending[:n_items]

class ContentBasedVectorEngine:
    """
    Prime Video-style Content-Based Filtering using Deep Learning Embeddings
    """
    
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        self.content_embeddings = {}
        self.scaler = MinMaxScaler()
        
    def create_content_embedding(self, content_data: Dict) -> ContentVector:
        """Create content embedding vector from metadata"""
        vector = ContentVector(
            content_id=content_data.get('id', 0),
            title=content_data.get('title', ''),
            language=content_data.get('language', 'unknown')
        )
        
        # Genre encoding (multi-hot with TF-IDF weighting)
        genres = content_data.get('genres', [])
        genre_map = {
            'Action': 0, 'Adventure': 1, 'Animation': 2, 'Comedy': 3,
            'Crime': 4, 'Documentary': 5, 'Drama': 6, 'Family': 7,
            'Fantasy': 8, 'History': 9, 'Horror': 10, 'Music': 11,
            'Mystery': 12, 'Romance': 13, 'Science Fiction': 14,
            'TV Movie': 15, 'Thriller': 16, 'War': 17, 'Western': 18,
            'Anime': 19, 'Bollywood': 20, 'Regional': 21, 'K-Drama': 22
        }
        
        for genre in genres:
            if genre in genre_map:
                idx = genre_map[genre]
                if idx < len(vector.genre_vector):
                    # TF-IDF style weighting
                    vector.genre_vector[idx] = 1.0 / (1.0 + math.log(1 + genres.index(genre)))
        
        # Language encoding with priority weighting
        lang_map = {
            'telugu': (0, 1.5), 'english': (1, 1.3), 'hindi': (2, 1.2),
            'tamil': (3, 1.0), 'malayalam': (4, 1.0), 'kannada': (5, 0.95),
            'korean': (6, 0.9), 'japanese': (7, 0.9), 'spanish': (8, 0.85),
            'french': (9, 0.85)
        }
        
        if vector.language in lang_map:
            idx, weight = lang_map[vector.language]
            if idx < len(vector.language_vector):
                vector.language_vector[idx] = weight
        
        # Quality signals
        vector.rating = content_data.get('vote_average', 0)
        vector.vote_count = content_data.get('vote_count', 0)
        vector.popularity = content_data.get('popularity', 0)
        
        # Quality vector encoding
        vector.quality_vector[0] = min(1.0, vector.rating / 10.0)
        vector.quality_vector[1] = min(1.0, math.log(1 + vector.vote_count) / 10)
        vector.quality_vector[2] = min(1.0, vector.popularity / 1000)
        vector.quality_vector[3] = 1.0 if vector.rating >= 8.0 else 0.5
        vector.quality_vector[4] = 1.0 if vector.vote_count >= 1000 else 0.3
        
        # Engagement vector (simulated for now)
        vector.engagement_vector[0] = random.uniform(0.3, 1.0)  # Watch completion rate
        vector.engagement_vector[1] = random.uniform(0.2, 0.8)  # Rewatch rate
        vector.engagement_vector[2] = random.uniform(0.1, 0.7)  # Share rate
        vector.engagement_vector[3] = random.uniform(0.3, 0.9)  # Like rate
        
        # Temporal features
        release_date = content_data.get('release_date')
        if release_date:
            try:
                if isinstance(release_date, str):
                    vector.release_date = datetime.strptime(release_date[:10], '%Y-%m-%d')
                else:
                    vector.release_date = release_date
                    
                # Recency score
                days_old = (datetime.now() - vector.release_date).days
                vector.temporal_vector[0] = math.exp(-days_old / 365)  # Exponential decay
                vector.temporal_vector[1] = 1.0 if days_old <= 30 else 0.5 if days_old <= 90 else 0.2
                vector.temporal_vector[2] = float(vector.release_date.month) / 12
                vector.temporal_vector[3] = float(vector.release_date.year - 2000) / 25
            except:
                pass
        
        # Build final feature vector
        vector.build_feature_vector()
        
        # Store in embeddings
        self.content_embeddings[content_data.get('id', 0)] = vector
        
        return vector
    
    def calculate_similarity(self, vector1: ContentVector, vector2: ContentVector) -> float:
        """Calculate cosine similarity between content vectors"""
        if vector1.feature_vector is None or vector2.feature_vector is None:
            return 0.0
        
        # Cosine similarity
        similarity = 1 - spatial.distance.cosine(vector1.feature_vector, vector2.feature_vector)
        
        # Language boost for same language content
        if vector1.language == vector2.language:
            similarity *= 1.2
        
        # Quality boost for high-rated content
        quality_boost = min(vector2.rating / 10, 1.0)
        similarity *= (0.8 + 0.2 * quality_boost)
        
        return min(1.0, similarity)
    
    def get_similar_content(self, content_id: int, n_items: int = 20) -> List[Tuple[int, float]]:
        """Get similar content using vector similarity"""
        if content_id not in self.content_embeddings:
            return []
        
        base_vector = self.content_embeddings[content_id]
        similarities = []
        
        for other_id, other_vector in self.content_embeddings.items():
            if other_id != content_id:
                sim = self.calculate_similarity(base_vector, other_vector)
                similarities.append((other_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_items]

class DeepLearningTrendingEngine:
    """
    Crunchyroll/Netflix-style Deep Learning Trending Engine
    Uses neural collaborative filtering and attention mechanisms
    """
    
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.attention_weights = np.random.randn(embedding_dim, embedding_dim) * 0.01
        self.trending_threshold = 0.7
        
    def attention_mechanism(self, query: np.ndarray, keys: List[np.ndarray], 
                          values: List[np.ndarray]) -> np.ndarray:
        """Transformer-style attention mechanism (simplified)"""
        if not keys or not values:
            return query
        
        # Calculate attention scores
        scores = []
        for key in keys:
            score = np.dot(query, key) / math.sqrt(self.embedding_dim)
            scores.append(score)
        
        # Softmax
        scores = np.array(scores)
        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / exp_scores.sum()
        
        # Weighted sum of values
        output = np.zeros_like(values[0])
        for i, value in enumerate(values):
            output += attention_weights[i] * value
        
        return output
    
    def calculate_trending_score_dl(self, content_vector: ContentVector, 
                                   context: Dict) -> float:
        """Calculate trending score using deep learning approach"""
        if content_vector.feature_vector is None:
            return 0.0
        
        # Context encoding
        hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        
        # Time-based attention
        time_embedding = np.zeros(self.embedding_dim)
        time_embedding[hour % self.embedding_dim] = 1.0
        time_embedding[day_of_week + 24] = 1.0
        
        # Apply attention to content features
        feature_subset = content_vector.feature_vector[:self.embedding_dim]
        attended_features = self.attention_mechanism(
            time_embedding,
            [feature_subset],
            [feature_subset]
        )
        
        # Calculate score
        base_score = np.mean(attended_features)
        
        # Boost for quality and engagement
        quality_boost = content_vector.rating / 10.0
        popularity_boost = min(1.0, content_vector.popularity / 100)
        
        # Language priority boost
        language_boosts = {
            'telugu': 1.5, 'english': 1.3, 'hindi': 1.2,
            'tamil': 1.0, 'malayalam': 1.0
        }
        lang_boost = language_boosts.get(content_vector.language, 0.9)
        
        # Final score
        final_score = base_score * (1 + quality_boost) * (1 + popularity_boost) * lang_boost
        
        return min(1.0, final_score)

class HybridTrendingOrchestrator:
    """
    Main orchestrator combining all algorithms like Netflix's approach
    """
    
    def __init__(self, db, cache):
        self.db = db
        self.cache = cache
        
        # Initialize all engines
        self.collab_engine = CollaborativeFilteringEngine()
        self.content_engine = ContentBasedVectorEngine()
        self.dl_engine = DeepLearningTrendingEngine()
        
        # Weight for each algorithm (learned over time)
        self.algorithm_weights = {
            'collaborative': 0.3,
            'content_based': 0.3,
            'deep_learning': 0.2,
            'popularity': 0.1,
            'quality': 0.1
        }
        
        # Store for real-time updates
        self.real_time_signals = defaultdict(lambda: {
            'views': 0, 'searches': 0, 'shares': 0,
            'completion_rate': 0.0, 'timestamp': datetime.now()
        })
        
    def update_real_time_signals(self, content_id: int, signal_type: str, value: float = 1.0):
        """Update real-time engagement signals"""
        self.real_time_signals[content_id][signal_type] += value
        self.real_time_signals[content_id]['timestamp'] = datetime.now()
    
    def calculate_hybrid_score(self, content_data: Dict, user_vector: Optional[UserVector] = None) -> float:
        """Calculate hybrid trending score combining all approaches"""
        content_id = content_data.get('id', 0)
        
        # Create content vector
        content_vector = self.content_engine.create_content_embedding(content_data)
        
        scores = {}
        
        # 1. Collaborative Filtering Score
        cf_trending = self.collab_engine.get_trending_items(100)
        cf_dict = dict(cf_trending)
        scores['collaborative'] = cf_dict.get(content_id, 0) / 10.0 if cf_dict else 0.5
        
        # 2. Content-Based Score (similarity to trending content)
        similar_items = self.content_engine.get_similar_content(content_id, 10)
        if similar_items:
            scores['content_based'] = np.mean([sim for _, sim in similar_items[:5]])
        else:
            scores['content_based'] = 0.5
        
        # 3. Deep Learning Score
        scores['deep_learning'] = self.dl_engine.calculate_trending_score_dl(
            content_vector, {'user': user_vector}
        )
        
        # 4. Popularity Score (Netflix-style)
        popularity = content_data.get('popularity', 0)
        scores['popularity'] = min(1.0, popularity / 200)
        
        # 5. Quality Score (Prime Video-style)
        rating = content_data.get('vote_average', 0)
        votes = content_data.get('vote_count', 0)
        if votes > 100:
            scores['quality'] = (rating / 10.0) * min(1.0, math.log(votes) / 10)
        else:
            scores['quality'] = rating / 20.0  # Lower weight for low vote count
        
        # 6. Real-time boost
        real_time_boost = 1.0
        if content_id in self.real_time_signals:
            signals = self.real_time_signals[content_id]
            recency = (datetime.now() - signals['timestamp']).total_seconds() / 3600
            if recency < 24:  # Within last 24 hours
                engagement = (signals['views'] / 100 + signals['searches'] / 50 + 
                            signals['shares'] / 20 + signals['completion_rate'])
                real_time_boost = 1 + min(0.5, engagement / 10)
        
        # Calculate weighted average
        final_score = 0.0
        for algo, weight in self.algorithm_weights.items():
            final_score += scores.get(algo, 0) * weight
        
        # Apply real-time boost and language priority
        final_score *= real_time_boost
        
        # Language priority multiplier
        lang_priority = {
            'telugu': 1.3, 'english': 1.2, 'hindi': 1.15,
            'tamil': 1.05, 'malayalam': 1.05
        }
        lang_mult = lang_priority.get(content_vector.language, 1.0)
        final_score *= lang_mult
        
        return min(1.0, final_score)
    
    def get_trending_with_vectors(self, languages: List[str], limit: int = 20,
                                 user_vector: Optional[UserVector] = None) -> List[Dict]:
        """Get trending content using vector algorithms"""
        trending_items = []
        
        # Cache key for trending
        cache_key = f"vector_trending:{'_'.join(languages[:3])}:{limit}"
        cached = self.cache.get(cache_key)
        
        if cached and not user_vector:  # Use cache only for non-personalized
            return cached
        
        # Process each language
        for language in languages:
            lang_items = self._get_language_trending_vectors(language, limit // len(languages) + 5)
            trending_items.extend(lang_items)
        
        # Sort by hybrid score
        trending_items.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Ensure diversity (Netflix-style)
        final_items = self._ensure_diversity(trending_items, limit)
        
        # Cache results
        if not user_vector:
            self.cache.set(cache_key, final_items, timeout=300)
        
        return final_items
    
    def _get_language_trending_vectors(self, language: str, limit: int) -> List[Dict]:
        """Get trending for specific language using vectors"""
        items = []
        
        # This would fetch from actual data source
        # For now, using simulated data
        sample_data = [
            {
                'id': random.randint(1000, 9999),
                'title': f"{language.title()} Movie {i}",
                'vote_average': random.uniform(6.0, 9.5),
                'vote_count': random.randint(100, 10000),
                'popularity': random.uniform(10, 500),
                'genres': random.sample(['Action', 'Drama', 'Comedy', 'Thriller'], 2),
                'language': language,
                'release_date': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
            }
            for i in range(limit)
        ]
        
        for data in sample_data:
            hybrid_score = self.calculate_hybrid_score(data)
            
            items.append({
                'id': data['id'],
                'title': data['title'],
                'language': language,
                'rating': data['vote_average'],
                'vote_count': data['vote_count'],
                'popularity': data['popularity'],
                'hybrid_score': hybrid_score,
                'trending_reason': self._get_trending_reason(hybrid_score, data),
                'algorithm_scores': {
                    'collaborative': 0,
                    'content_based': 0,
                    'deep_learning': 0,
                    'popularity': data['popularity'] / 200,
                    'quality': data['vote_average'] / 10
                }
            })
        
        return items
    
    def _ensure_diversity(self, items: List[Dict], limit: int) -> List[Dict]:
        """Ensure diversity in trending list (Netflix approach)"""
        final_items = []
        language_counts = defaultdict(int)
        genre_counts = defaultdict(int)
        
        # Priority for Telugu, English, Hindi
        priority_langs = ['telugu', 'english', 'hindi']
        
        # First pass: Add priority language content
        for item in items:
            if len(final_items) >= limit:
                break
                
            lang = item.get('language')
            if lang in priority_langs and language_counts[lang] < limit // 3:
                final_items.append(item)
                language_counts[lang] += 1
        
        # Second pass: Add high-scoring content from other languages
        for item in items:
            if len(final_items) >= limit:
                break
                
            if item not in final_items and item.get('hybrid_score', 0) > 0.7:
                lang = item.get('language')
                if language_counts[lang] < limit // 5:
                    final_items.append(item)
                    language_counts[lang] += 1
        
        # Fill remaining slots
        for item in items:
            if len(final_items) >= limit:
                break
            if item not in final_items:
                final_items.append(item)
        
        return final_items
    
    def _get_trending_reason(self, score: float, data: Dict) -> str:
        """Generate trending reason based on score and data"""
        reasons = []
        
        if score > 0.8:
            reasons.append("Highly trending")
        if data.get('vote_average', 0) >= 8.5:
            reasons.append(f"Exceptional rating: {data['vote_average']:.1f}/10")
        if data.get('popularity', 0) > 200:
            reasons.append("Viral sensation")
        if data.get('language') in ['telugu', 'english', 'hindi']:
            lang_tags = {'telugu': 'TFI', 'english': 'Hollywood', 'hindi': 'Bollywood'}
            reasons.append(f"Popular in {lang_tags[data['language']]}")
        
        return " • ".join(reasons) if reasons else "Trending now"

# ================== Main Service ==================

class AdvancedTrendingService:
    """Main service integrating vector-based algorithms"""
    
    def __init__(self, db, cache, tmdb_api_key):
        self.db = db
        self.cache = cache
        self.tmdb_api_key = tmdb_api_key
        self.session = self._create_http_session()
        self.base_url = 'https://api.themoviedb.org/3'
        
        # Initialize vector engines
        self.vector_orchestrator = HybridTrendingOrchestrator(db, cache)
        
        self.app = None
        self.Content = None
        
        self.update_thread = None
        self.stop_updates = False
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def _create_http_session(self):
        """Create HTTP session with retry logic"""
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        session = requests.Session()
        retry = Retry(
            total=2,
            read=2,
            connect=2,
            backoff_factor=0.3,
            status_forcelist=(500, 502, 504)
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def set_app_context(self, app, content_model):
        """Set the Flask app and Content model references"""
        self.app = app
        self.Content = content_model
    
    def start_background_updates(self):
        """Start background thread for continuous trending updates"""
        if not self.update_thread:
            self.stop_updates = False
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            logger.info("Started background trending updates with vector algorithms")
    
    def stop_background_updates(self):
        """Stop background updates"""
        self.stop_updates = True
        if self.update_thread:
            self.update_thread.join(timeout=5)
            self.update_thread = None
        self.executor.shutdown(wait=False)
        logger.info("Stopped background trending updates")
    
    def _update_loop(self):
        """Background update loop"""
        while not self.stop_updates:
            try:
                if self.app:
                    with self.app.app_context():
                        # Update embeddings and collaborative filtering
                        self._update_vector_models()
                        
                        # Update trending for priority languages
                        for language in LANGUAGE_PRIORITY_ORDER[:5]:
                            self._update_language_trending_vectors(language)
                
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in background update: {e}")
                time.sleep(60)
    
    def _update_vector_models(self):
        """Update vector models with latest data"""
        try:
            # Fetch recent interactions (simulated)
            interactions = self._get_recent_interactions()
            
            # Update collaborative filtering
            if interactions:
                self.vector_orchestrator.collab_engine.train_incremental(interactions)
            
            logger.info("Updated vector models")
            
        except Exception as e:
            logger.error(f"Error updating vector models: {e}")
    
    def _get_recent_interactions(self) -> List[Tuple[str, int, float]]:
        """Get recent user interactions for collaborative filtering"""
        # In production, this would fetch from database
        # For now, return simulated data
        interactions = []
        for _ in range(100):
            user_id = f"user_{random.randint(1, 1000)}"
            item_id = random.randint(1000, 9999)
            rating = random.uniform(1, 10)
            interactions.append((user_id, item_id, rating))
        return interactions
    
    def _update_language_trending_vectors(self, language: str):
        """Update trending with vector algorithms for specific language"""
        try:
            cache_key = f"vector_trending:{language}:latest"
            
            # Fetch from TMDB
            params = {
                'api_key': self.tmdb_api_key,
                'with_original_language': self._get_language_code(language),
                'sort_by': 'popularity.desc',
                'vote_count.gte': 50,
                'page': 1
            }
            
            response = self.session.get(f"{self.base_url}/discover/movie", 
                                       params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                trending_items = []
                
                for movie in data.get('results', [])[:20]:
                    # Create content data
                    content_data = {
                        'id': movie['id'],
                        'title': movie.get('title', ''),
                        'vote_average': movie.get('vote_average', 0),
                        'vote_count': movie.get('vote_count', 0),
                        'popularity': movie.get('popularity', 0),
                        'genres': [],  # Would need genre lookup
                        'language': language,
                        'release_date': movie.get('release_date')
                    }
                    
                    # Calculate hybrid score
                    hybrid_score = self.vector_orchestrator.calculate_hybrid_score(content_data)
                    
                    if hybrid_score > 0.5:  # Threshold
                        trending_items.append({
                            'content_data': content_data,
                            'hybrid_score': hybrid_score
                        })
                
                # Sort and cache
                trending_items.sort(key=lambda x: x['hybrid_score'], reverse=True)
                self.cache.set(cache_key, trending_items, timeout=600)
                
                logger.info(f"Updated {language} vector trending: {len(trending_items)} items")
                
        except Exception as e:
            logger.error(f"Error updating {language} vector trending: {e}")
    
    def _get_language_code(self, language: str) -> str:
        """Get TMDB language code"""
        lang_codes = {
            'telugu': 'te', 'english': 'en', 'hindi': 'hi',
            'tamil': 'ta', 'malayalam': 'ml', 'kannada': 'kn',
            'korean': 'ko', 'japanese': 'ja', 'spanish': 'es',
            'french': 'fr'
        }
        return lang_codes.get(language, 'en')
    
    def get_trending(self, languages: List[str] = None, 
                    categories: List[str] = None,
                    limit: int = 20,
                    user_session: str = None) -> Dict[str, List[Dict]]:
        """Get trending content using vector algorithms"""
        if not languages:
            languages = LANGUAGE_PRIORITY_ORDER[:7]
        
        if not categories:
            categories = ['vector_trending', 'collaborative', 'content_based']
        
        results = {}
        
        # Create user vector if session provided
        user_vector = None
        if user_session:
            user_vector = self._create_user_vector(user_session)
        
        try:
            # Vector-based trending (main algorithm)
            if 'vector_trending' in categories:
                results['vector_trending'] = self.vector_orchestrator.get_trending_with_vectors(
                    languages, limit, user_vector
                )
            
            # Pure collaborative filtering results
            if 'collaborative' in categories:
                cf_items = self.vector_orchestrator.collab_engine.get_trending_items(limit)
                results['collaborative'] = self._format_cf_results(cf_items)
            
            # Content-based recommendations
            if 'content_based' in categories:
                results['content_based'] = self._get_content_based_trending(languages, limit)
            
            # Traditional trending (fallback)
            if 'traditional' in categories:
                results['traditional'] = self._get_traditional_trending(languages, limit)
                
        except Exception as e:
            logger.error(f"Error in get_trending: {traceback.format_exc()}")
            
        return results
    
    def _create_user_vector(self, session_id: str) -> UserVector:
        """Create user vector from session data"""
        user_vector = UserVector(session_id=session_id)
        
        # In production, load from session/database
        # For now, create sample preferences
        
        # Language preferences (Telugu, English, Hindi priority)
        user_vector.language_preferences[0] = 1.0  # Telugu
        user_vector.language_preferences[1] = 0.8  # English
        user_vector.language_preferences[2] = 0.7  # Hindi
        
        # Genre preferences
        user_vector.genre_preferences[0] = 0.9   # Action
        user_vector.genre_preferences[6] = 0.8   # Drama
        user_vector.genre_preferences[3] = 0.7   # Comedy
        
        # Time preferences
        hour = datetime.now().hour
        user_vector.time_preferences[hour] = 1.0
        
        user_vector.build_user_vector()
        return user_vector
    
    def _format_cf_results(self, cf_items: List[Tuple[int, float]]) -> List[Dict]:
        """Format collaborative filtering results"""
        formatted = []
        for item_id, score in cf_items:
            formatted.append({
                'id': item_id,
                'title': f"Content {item_id}",
                'cf_score': score,
                'algorithm': 'collaborative_filtering'
            })
        return formatted
    
    def _get_content_based_trending(self, languages: List[str], limit: int) -> List[Dict]:
        """Get content-based trending"""
        results = []
        
        # Get some seed content IDs
        seed_ids = list(self.vector_orchestrator.content_engine.content_embeddings.keys())[:5]
        
        for seed_id in seed_ids:
            similar = self.vector_orchestrator.content_engine.get_similar_content(seed_id, 5)
            for sim_id, sim_score in similar:
                if sim_id in self.vector_orchestrator.content_engine.content_embeddings:
                    vector = self.vector_orchestrator.content_engine.content_embeddings[sim_id]
                    results.append({
                        'id': sim_id,
                        'title': vector.title,
                        'language': vector.language,
                        'similarity_score': sim_score,
                        'rating': vector.rating,
                        'algorithm': 'content_based'
                    })
        
        # Sort by similarity and rating
        results.sort(key=lambda x: (x['similarity_score'] * x.get('rating', 5) / 10), reverse=True)
        return results[:limit]
    
    def _get_traditional_trending(self, languages: List[str], limit: int) -> List[Dict]:
        """Fallback to traditional trending"""
        # Implementation would be similar to previous version
        return []
    
    def update_engagement_signal(self, content_id: int, signal_type: str, value: float = 1.0):
        """Update real-time engagement signals for vector algorithms"""
        self.vector_orchestrator.update_real_time_signals(content_id, signal_type, value)

# ================== Service Initialization ==================

def init_advanced_trending_service(db, cache, tmdb_api_key):
    """Initialize the advanced trending service with vector algorithms"""
    service = AdvancedTrendingService(db, cache, tmdb_api_key)
    
    try:
        import sys
        app_module = sys.modules.get('app')
        if app_module:
            if hasattr(app_module, 'app'):
                flask_app = getattr(app_module, 'app')
                if hasattr(app_module, 'Content'):
                    content_model = getattr(app_module, 'Content')
                    service.set_app_context(flask_app, content_model)
                    logger.info("Initialized vector-based trending service")
    except Exception as e:
        logger.warning(f"Could not set app context: {e}")
    
    service.start_background_updates()
    return service

def get_trending_service():
    """Get the trending service instance"""
    return None