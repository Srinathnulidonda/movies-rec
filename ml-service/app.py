import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, text, pool
from sqlalchemy.orm import sessionmaker
import redis
from redis.exceptions import ConnectionError as RedisConnectionError

# Scientific computing
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Recommendation libraries
from scipy.sparse import csr_matrix, lil_matrix, hstack, vstack
import implicit
from implicit import als, bpr, lmf
import faiss

# Utilities
import joblib
import hashlib
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///movie_recommendations.db')
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
MODEL_UPDATE_INTERVAL = int(os.environ.get('MODEL_UPDATE_INTERVAL', '3600'))  # 1 hour

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# Initialize database connection with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=pool.QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
SessionLocal = sessionmaker(bind=engine)

# Redis client with fallback
class CacheManager:
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.cache_expiry = {}
        self._init_redis()
    
    def _init_redis(self):
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[str]:
        if self.redis_client:
            try:
                return self.redis_client.get(key)
            except RedisConnectionError:
                self._init_redis()
        
        # Fallback to memory cache
        if key in self.memory_cache:
            if self.cache_expiry.get(key, 0) > datetime.now().timestamp():
                return self.memory_cache[key]
            else:
                del self.memory_cache[key]
                del self.cache_expiry[key]
        return None
    
    def set(self, key: str, value: str, ttl: int = 3600):
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, value)
                return
            except RedisConnectionError:
                self._init_redis()
        
        # Fallback to memory cache
        self.memory_cache[key] = value
        self.cache_expiry[key] = datetime.now().timestamp() + ttl
    
    def delete(self, key: str):
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except RedisConnectionError:
                pass
        
        if key in self.memory_cache:
            del self.memory_cache[key]
            if key in self.cache_expiry:
                del self.cache_expiry[key]

# Initialize cache manager
cache_manager = CacheManager()

# Model storage
class ModelManager:
    def __init__(self):
        self.models = {
            'svd': None,
            'nmf': None,
            'als': None,
            'bpr': None,
            'neural_cf': None,
            'content_encoder': None,
            'hybrid_model': None,
            'popularity_model': None
        }
        self.encoders = {
            'tfidf': None,
            'count': None,
            'genre_encoder': None,
            'language_encoder': None
        }
        self.indices = {
            'faiss_item': None,
            'faiss_user': None
        }
        self.data = {
            'user_item_matrix': None,
            'item_features': None,
            'user_features': None,
            'item_mapping': {},
            'user_mapping': {},
            'reverse_item_mapping': {},
            'reverse_user_mapping': {}
        }
        self.last_update = None
        self.model_lock = threading.Lock()
    
    def update_models(self, new_models: Dict[str, Any]):
        with self.model_lock:
            self.models.update(new_models)
            self.last_update = datetime.now()
    
    def get_model(self, name: str):
        with self.model_lock:
            return self.models.get(name)
    
    def is_initialized(self) -> bool:
        return self.data['user_item_matrix'] is not None

model_manager = ModelManager()

# Enhanced Neural Collaborative Filtering Model
class NeuralCF(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_features: int = 0, 
                 embedding_dim: int = 64, hidden_layers: List[int] = [128, 64, 32]):
        super(NeuralCF, self).__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Feature processing
        self.n_features = n_features
        if n_features > 0:
            self.feature_processor = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32)
            )
            input_dim = embedding_dim * 2 + 32
        else:
            input_dim = embedding_dim * 2
        
        # MLP layers
        self.fc_layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_layers):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(0.2))
            self.fc_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        # Output layer
        self.output = nn.Linear(hidden_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, 
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        if self.n_features > 0 and features is not None:
            feature_embeds = self.feature_processor(features)
            x = torch.cat([user_embeds, item_embeds, feature_embeds], dim=1)
        else:
            x = torch.cat([user_embeds, item_embeds], dim=1)
        
        for layer in self.fc_layers:
            x = layer(x)
        
        output = self.sigmoid(self.output(x))
        return output.squeeze()

# Hybrid Recommendation Model
class HybridRecommender:
    def __init__(self):
        self.weights = {
            'collaborative': 0.4,
            'content': 0.2,
            'neural': 0.25,
            'popularity': 0.15
        }
    
    def get_recommendations(self, user_id: int, n_recommendations: int = 20,
                          filters: Optional[Dict] = None) -> List[int]:
        """Get hybrid recommendations combining multiple approaches"""
        recommendations = defaultdict(float)
        
        # Get recommendations from each model
        methods = {
            'collaborative': self._get_collaborative_recommendations,
            'content': self._get_content_recommendations,
            'neural': self._get_neural_recommendations,
            'popularity': self._get_popularity_recommendations
        }
        
        for method_name, method_func in methods.items():
            try:
                method_recs = method_func(user_id, n_recommendations * 2)
                weight = self.weights[method_name]
                
                for i, (item_id, score) in enumerate(method_recs):
                    # Apply positional decay
                    position_weight = 1.0 / (1.0 + i * 0.1)
                    recommendations[item_id] += weight * score * position_weight
            except Exception as e:
                logger.warning(f"Error in {method_name} recommendations: {e}")
        
        # Apply filters if provided
        if filters:
            recommendations = self._apply_filters(recommendations, filters)
        
        # Sort and return top recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_recs[:n_recommendations]]
    
    def _get_collaborative_recommendations(self, user_id: int, n_recs: int) -> List[Tuple[int, float]]:
        """Get collaborative filtering recommendations"""
        if not model_manager.data['user_mapping'] or user_id not in model_manager.data['user_mapping']:
            return []
        
        user_idx = model_manager.data['user_mapping'][user_id]
        recommendations = []
        
        # Try ALS model first
        als_model = model_manager.get_model('als')
        if als_model and model_manager.data['user_item_matrix'] is not None:
            try:
                ids, scores = als_model.recommend(
                    user_idx,
                    model_manager.data['user_item_matrix'][user_idx],
                    N=n_recs,
                    filter_already_liked_items=True
                )
                
                for idx, score in zip(ids, scores):
                    if idx in model_manager.data['reverse_item_mapping']:
                        item_id = model_manager.data['reverse_item_mapping'][idx]
                        recommendations.append((item_id, float(score)))
            except Exception as e:
                logger.error(f"ALS recommendation error: {e}")
        
        # Fallback to SVD
        if not recommendations:
            svd_model = model_manager.get_model('svd')
            if svd_model and model_manager.data.get('user_embeddings_svd') is not None:
                try:
                    user_embedding = model_manager.data['user_embeddings_svd'][user_idx]
                    scores = np.dot(model_manager.data['item_embeddings_svd'], user_embedding)
                    top_items = np.argsort(scores)[::-1][:n_recs]
                    
                    for idx in top_items:
                        if idx in model_manager.data['reverse_item_mapping']:
                            item_id = model_manager.data['reverse_item_mapping'][idx]
                            recommendations.append((item_id, float(scores[idx])))
                except Exception as e:
                    logger.error(f"SVD recommendation error: {e}")
        
        return recommendations
    
    def _get_content_recommendations(self, user_id: int, n_recs: int) -> List[Tuple[int, float]]:
        """Get content-based recommendations"""
        # Get user's interaction history
        with SessionLocal() as session:
            user_items_query = text("""
                SELECT DISTINCT content_id 
                FROM (
                    SELECT content_id FROM ratings WHERE user_id = :user_id
                    UNION
                    SELECT content_id FROM favorites WHERE user_id = :user_id
                    UNION
                    SELECT content_id FROM watchlist WHERE user_id = :user_id
                ) as interactions
                LIMIT 20
            """)
            result = session.execute(user_items_query, {'user_id': user_id})
            user_items = [row[0] for row in result]
        
        if not user_items or model_manager.data['item_features'] is None:
            return []
        
        recommendations = defaultdict(float)
        
        # Get similar items for each user item
        for item_id in user_items:
            if item_id in model_manager.data['item_mapping']:
                item_idx = model_manager.data['item_mapping'][item_id]
                similar_items = self._get_similar_items_content(item_idx, n_recs // len(user_items) + 1)
                
                for sim_item_id, score in similar_items:
                    if sim_item_id != item_id:
                        recommendations[sim_item_id] += score
        
        # Sort and return
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recs]
    
    def _get_neural_recommendations(self, user_id: int, n_recs: int) -> List[Tuple[int, float]]:
        """Get neural network recommendations"""
        neural_model = model_manager.get_model('neural_cf')
        if not neural_model or user_id not in model_manager.data['user_mapping']:
            return []
        
        user_idx = model_manager.data['user_mapping'][user_id]
        recommendations = []
        
        try:
            neural_model.eval()
            with torch.no_grad():
                # Score all items for the user
                user_tensor = torch.LongTensor([user_idx] * len(model_manager.data['item_mapping']))
                item_tensor = torch.LongTensor(list(range(len(model_manager.data['item_mapping']))))
                
                # Get predictions in batches
                batch_size = 1000
                scores = []
                
                for i in range(0, len(item_tensor), batch_size):
                    batch_users = user_tensor[i:i+batch_size]
                    batch_items = item_tensor[i:i+batch_size]
                    batch_scores = neural_model(batch_users, batch_items).cpu().numpy()
                    scores.extend(batch_scores)
                
                scores = np.array(scores)
                top_items = np.argsort(scores)[::-1][:n_recs]
                
                for idx in top_items:
                    if idx in model_manager.data['reverse_item_mapping']:
                        item_id = model_manager.data['reverse_item_mapping'][idx]
                        recommendations.append((item_id, float(scores[idx])))
        except Exception as e:
            logger.error(f"Neural recommendation error: {e}")
        
        return recommendations
    
    def _get_popularity_recommendations(self, user_id: int, n_recs: int) -> List[Tuple[int, float]]:
        """Get popularity-based recommendations"""
        with SessionLocal() as session:
            # Get user preferences
            user_query = text("""
                SELECT preferred_languages, preferred_genres, region
                FROM users WHERE id = :user_id
            """)
            user_result = session.execute(user_query, {'user_id': user_id}).fetchone()
            
            if user_result:
                languages = json.loads(user_result[0]) if user_result[0] else []
                genres = json.loads(user_result[1]) if user_result[1] else []
                region = user_result[2]
            else:
                languages, genres, region = [], [], None
            
            # Build query for popular items
            query = """
                SELECT id, popularity_score 
                FROM content 
                WHERE 1=1
            """
            params = {}
            
            if languages:
                query += " AND language IN :languages"
                params['languages'] = tuple(languages)
            
            if region:
                query += " AND (region = :region OR region = 'global')"
                params['region'] = region
            
            query += " ORDER BY popularity_score DESC LIMIT :limit"
            params['limit'] = n_recs * 2
            
            result = session.execute(text(query), params)
            
            recommendations = []
            for row in result:
                item_id, score = row
                # Apply genre boost
                if genres and hasattr(row, 'genres'):
                    item_genres = json.loads(row.genres) if row.genres else []
                    genre_overlap = len(set(genres) & set(item_genres))
                    score *= (1 + 0.1 * genre_overlap)
                
                recommendations.append((item_id, float(score)))
        
        return recommendations[:n_recs]
    
    def _get_similar_items_content(self, item_idx: int, n_similar: int) -> List[Tuple[int, float]]:
        """Get similar items based on content features"""
        if model_manager.data['item_features'] is None:
            return []
        
        try:
            item_features = model_manager.data['item_features'][item_idx]
            
            # Use FAISS for fast similarity search if available
            if model_manager.indices['faiss_item'] is not None:
                distances, indices = model_manager.indices['faiss_item'].search(
                    item_features.toarray().astype('float32'), n_similar + 1
                )
                
                similar_items = []
                for idx, dist in zip(indices[0][1:], distances[0][1:]):
                    if idx in model_manager.data['reverse_item_mapping']:
                        item_id = model_manager.data['reverse_item_mapping'][idx]
                        similar_items.append((item_id, 1.0 - dist))  # Convert distance to similarity
                
                return similar_items
            else:
                # Fallback to cosine similarity
                similarities = cosine_similarity(item_features, model_manager.data['item_features'])[0]
                top_similar = np.argsort(similarities)[::-1][1:n_similar + 1]
                
                similar_items = []
                for idx in top_similar:
                    if idx in model_manager.data['reverse_item_mapping']:
                        item_id = model_manager.data['reverse_item_mapping'][idx]
                        similar_items.append((item_id, float(similarities[idx])))
                
                return similar_items
        except Exception as e:
            logger.error(f"Content similarity error: {e}")
            return []
    
    def _apply_filters(self, recommendations: Dict[int, float], 
                      filters: Dict) -> Dict[int, float]:
        """Apply filters to recommendations"""
        if not filters:
            return recommendations
        
        filtered = {}
        
        with SessionLocal() as session:
            # Get content details for filtering
            item_ids = list(recommendations.keys())
            if not item_ids:
                return filtered
            
            query = text("""
                SELECT id, content_type, language, genres, release_date
                FROM content
                WHERE id IN :item_ids
            """)
            result = session.execute(query, {'item_ids': tuple(item_ids)})
            
            for row in result:
                item_id = row[0]
                passes_filter = True
                
                # Apply content type filter
                if 'content_type' in filters and row[1] != filters['content_type']:
                    passes_filter = False
                
                # Apply language filter
                if 'language' in filters and row[2] not in filters['language']:
                    passes_filter = False
                
                # Apply genre filter
                if 'genres' in filters and row[3]:
                    item_genres = json.loads(row[3])
                    if not any(g in item_genres for g in filters['genres']):
                        passes_filter = False
                
                # Apply release date filter
                if 'min_year' in filters and row[4]:
                    if row[4].year < filters['min_year']:
                        passes_filter = False
                
                if passes_filter:
                    filtered[item_id] = recommendations[item_id]
        
        return filtered

# Initialize hybrid recommender
hybrid_recommender = HybridRecommender()

# Data loading and preprocessing
class DataProcessor:
    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all necessary data from database"""
        with SessionLocal() as session:
            # Load ratings with timestamps
            ratings_query = """
                SELECT user_id, content_id, rating, created_at 
                FROM ratings
                ORDER BY created_at DESC
            """
            ratings_df = pd.read_sql(ratings_query, session.bind)
            
            # Load content with all features
            content_query = """
                SELECT id, title, original_title, synopsis, plot, genres, 
                       cast_crew, language, region, content_type, 
                       popularity_score, release_date, runtime,
                       tmdb_id, imdb_id, mal_id
                FROM content
            """
            content_df = pd.read_sql(content_query, session.bind)
            
            # Load user data
            user_query = """
                SELECT id, preferred_languages, preferred_genres, region,
                       created_at
                FROM users
            """
            user_df = pd.read_sql(user_query, session.bind)
            
            # Load additional interactions
            watchlist_query = """
                SELECT user_id, content_id, added_at as created_at,
                       5.0 as rating
                FROM watchlist
            """
            watchlist_df = pd.read_sql(watchlist_query, session.bind)
            
            favorites_query = """
                SELECT user_id, content_id, added_at as created_at,
                       8.0 as rating
                FROM favorites
            """
            favorites_df = pd.read_sql(favorites_query, session.bind)
            
            # Combine all interactions
            all_interactions = pd.concat([
                ratings_df,
                watchlist_df,
                favorites_df
            ], ignore_index=True)
            
            # Remove duplicates, keeping the most recent interaction
            all_interactions = all_interactions.sort_values('created_at', ascending=False)
            all_interactions = all_interactions.drop_duplicates(['user_id', 'content_id'], keep='first')
            
            return all_interactions, content_df, user_df
    
    @staticmethod
    def create_user_item_matrix(interactions_df: pd.DataFrame) -> csr_matrix:
        """Create sparse user-item interaction matrix"""
        # Create mappings
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['content_id'].unique()
        
        user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        
        # Store mappings
        model_manager.data['user_mapping'] = user_mapping
        model_manager.data['item_mapping'] = item_mapping
        model_manager.data['reverse_user_mapping'] = {idx: user for user, idx in user_mapping.items()}
        model_manager.data['reverse_item_mapping'] = {idx: item for item, idx in item_mapping.items()}
        
        # Create matrix
        row_indices = [user_mapping[user] for user in interactions_df['user_id']]
        col_indices = [item_mapping[item] for item in interactions_df['content_id']]
        
        # Normalize ratings to 0-1 scale
        ratings = interactions_df['rating'].values / 10.0
        
        matrix = csr_matrix(
            (ratings, (row_indices, col_indices)),
            shape=(len(unique_users), len(unique_items))
        )
        
        return matrix
    
    @staticmethod
    def create_content_features(content_df: pd.DataFrame) -> csr_matrix:
        """Create content feature matrix"""
        # Process text features
        content_df['text_features'] = (
            content_df['title'].fillna('') + ' ' +
            content_df['original_title'].fillna('') + ' ' +
            content_df['synopsis'].fillna('') + ' ' +
            content_df['plot'].fillna('')
        )
        
        # Process genres
        content_df['genre_text'] = content_df['genres'].apply(
            lambda x: ' '.join(json.loads(x)) if x else ''
        )
        
        # Process cast and crew
        content_df['cast_crew_text'] = content_df['cast_crew'].apply(
            lambda x: ' '.join([
                x.get('director', ''),
                x.get('writer', ''),
                x.get('actors', '')
            ]) if isinstance(x, dict) else ''
        )
        
        # Create TF-IDF features
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Combine all text
        all_text = (
            content_df['text_features'] + ' ' +
            content_df['genre_text'] + ' ' +
            content_df['cast_crew_text']
        )
        
        tfidf_features = tfidf.fit_transform(all_text)
        
        # Store vectorizer
        model_manager.encoders['tfidf'] = tfidf
        
        # Create additional features
        # Language encoding
        lang_encoder = LabelEncoder()
        lang_encoded = lang_encoder.fit_transform(content_df['language'].fillna('unknown'))
        
        # Content type encoding
        type_encoder = LabelEncoder()
        type_encoded = type_encoder.fit_transform(content_df['content_type'].fillna('unknown'))
        
        # Numerical features
        numerical_features = StandardScaler().fit_transform(
            content_df[['popularity_score', 'runtime']].fillna(0)
        )
        
        # Combine all features
        from scipy.sparse import hstack as sparse_hstack
        
        additional_features = np.column_stack([
            lang_encoded,
            type_encoded,
            numerical_features
        ])
        
        content_features = sparse_hstack([
            tfidf_features,
            csr_matrix(additional_features)
        ])
        
        return content_features
    
    @staticmethod
    def create_user_features(user_df: pd.DataFrame, interactions_df: pd.DataFrame,
                           content_df: pd.DataFrame) -> np.ndarray:
        """Create user feature matrix"""
        user_features = []
        
        for _, user in user_df.iterrows():
            features = []
            
            # User preferences
            pref_langs = json.loads(user['preferred_languages']) if user['preferred_languages'] else []
            pref_genres = json.loads(user['preferred_genres']) if user['preferred_genres'] else []
            
            # Get user's interaction statistics
            user_interactions = interactions_df[interactions_df['user_id'] == user['id']]
            
            # Average rating
            avg_rating = user_interactions['rating'].mean() if len(user_interactions) > 0 else 5.0
            features.append(avg_rating)
            
            # Number of interactions
            n_interactions = len(user_interactions)
            features.append(np.log1p(n_interactions))
            
            # Diversity of content types
            if len(user_interactions) > 0:
                interacted_items = user_interactions['content_id'].values
                interacted_content = content_df[content_df['id'].isin(interacted_items)]
                
                # Content type diversity
                content_types = interacted_content['content_type'].nunique()
                features.append(content_types)
                
                # Language diversity
                languages = interacted_content['language'].nunique()
                features.append(languages)
                
                # Genre diversity
                all_genres = []
                for genres in interacted_content['genres'].dropna():
                    if genres:
                        all_genres.extend(json.loads(genres))
                genre_diversity = len(set(all_genres))
                features.append(genre_diversity)
            else:
                features.extend([0, 0, 0])
            
            # Account age (in days)
            account_age = (datetime.now() - user['created_at']).days
            features.append(np.log1p(account_age))
            
            user_features.append(features)
        
        return np.array(user_features)

# Model training functions
class ModelTrainer:
    @staticmethod
    def train_svd(user_item_matrix: csr_matrix, n_components: int = 100):
        """Train SVD model"""
        logger.info("Training SVD model...")
        
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        user_embeddings = svd.fit_transform(user_item_matrix)
        item_embeddings = svd.components_.T
        
        # Store in model manager
        model_manager.models['svd'] = svd
        model_manager.data['user_embeddings_svd'] = user_embeddings
        model_manager.data['item_embeddings_svd'] = item_embeddings
        
        logger.info(f"SVD model trained with {n_components} components")
        return svd
    
    @staticmethod
    def train_nmf(user_item_matrix: csr_matrix, n_components: int = 100):
        """Train Non-negative Matrix Factorization model"""
        logger.info("Training NMF model...")
        
        # Ensure non-negative values
        user_item_positive = user_item_matrix.copy()
        user_item_positive.data = np.maximum(user_item_positive.data, 0)
        
        nmf = NMF(n_components=n_components, random_state=42, max_iter=200)
        user_embeddings = nmf.fit_transform(user_item_positive)
        item_embeddings = nmf.components_.T
        
        # Store in model manager
        model_manager.models['nmf'] = nmf
        model_manager.data['user_embeddings_nmf'] = user_embeddings
        model_manager.data['item_embeddings_nmf'] = item_embeddings
        
        logger.info(f"NMF model trained with {n_components} components")
        return nmf
    
    @staticmethod
    def train_als(user_item_matrix: csr_matrix):
        """Train Alternating Least Squares model"""
        logger.info("Training ALS model...")
        
        als_model = implicit.als.AlternatingLeastSquares(
            factors=128,
            regularization=0.01,
            iterations=50,
            calculate_training_loss=True,
            random_state=42
        )
        
        # implicit library expects item-user matrix
        als_model.fit(user_item_matrix.T)
        
        model_manager.models['als'] = als_model
        logger.info("ALS model trained successfully")
        return als_model
    
    @staticmethod
    def train_bpr(user_item_matrix: csr_matrix):
        """Train Bayesian Personalized Ranking model"""
        logger.info("Training BPR model...")
        
        bpr_model = implicit.bpr.BayesianPersonalizedRanking(
            factors=128,
            learning_rate=0.01,
            regularization=0.001,
            iterations=100,
            random_state=42
        )
        
        # Convert to binary implicit feedback
        binary_matrix = user_item_matrix.copy()
        binary_matrix.data = np.ones_like(binary_matrix.data)
        
        bpr_model.fit(binary_matrix.T)
        
        model_manager.models['bpr'] = bpr_model
        logger.info("BPR model trained successfully")
        return bpr_model
    
    @staticmethod
    def train_neural_cf(interactions_df: pd.DataFrame, content_features: Optional[csr_matrix] = None):
        """Train Neural Collaborative Filtering model"""
        logger.info("Training Neural CF model...")
        
        # Prepare data
        user_ids = [model_manager.data['user_mapping'][uid] for uid in interactions_df['user_id']]
        item_ids = [model_manager.data['item_mapping'][iid] for iid in interactions_df['content_id']]
        ratings = interactions_df['rating'].values / 10.0
        
        # Create dataset
        dataset = TensorDataset(
            torch.LongTensor(user_ids),
            torch.LongTensor(item_ids),
            torch.FloatTensor(ratings)
        )
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # Initialize model
        n_users = len(model_manager.data['user_mapping'])
        n_items = len(model_manager.data['item_mapping'])
        n_features = content_features.shape[1] if content_features is not None else 0
        
        model = NeuralCF(n_users, n_items, n_features)
        
        # Training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(50):
            # Training
            model.train()
            train_loss = 0
            for batch_users, batch_items, batch_ratings in train_loader:
                batch_users = batch_users.to(device)
                batch_items = batch_items.to(device)
                batch_ratings = batch_ratings.to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_users, batch_items, batch_ratings in val_loader:
                    batch_users = batch_users.to(device)
                    batch_items = batch_items.to(device)
                    batch_ratings = batch_ratings.to(device)
                    
                    predictions = model(batch_users, batch_items)
                    loss = criterion(predictions, batch_ratings)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    logger.info("Early stopping triggered")
                    break
        
        # Move model back to CPU for inference
        model = model.cpu()
        model_manager.models['neural_cf'] = model
        
        logger.info("Neural CF model trained successfully")
        return model
    
    @staticmethod
    def build_faiss_indices(item_embeddings: np.ndarray, user_embeddings: Optional[np.ndarray] = None):
        """Build FAISS indices for fast similarity search"""
        logger.info("Building FAISS indices...")
        
        # Item index
        item_embeddings_32 = item_embeddings.astype('float32')
        
        # Normalize embeddings
        faiss.normalize_L2(item_embeddings_32)
        
        # Build index
        d = item_embeddings_32.shape[1]
        item_index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
        item_index.add(item_embeddings_32)
        
        model_manager.indices['faiss_item'] = item_index
        
        # User index if available
        if user_embeddings is not None:
            user_embeddings_32 = user_embeddings.astype('float32')
            faiss.normalize_L2(user_embeddings_32)
            
            user_index = faiss.IndexFlatIP(d)
            user_index.add(user_embeddings_32)
            
            model_manager.indices['faiss_user'] = user_index
        
        logger.info("FAISS indices built successfully")

# Main training pipeline
def train_all_models():
    """Train all recommendation models"""
    try:
        logger.info("Starting model training pipeline...")
        
        # Load data
        data_processor = DataProcessor()
        interactions_df, content_df, user_df = data_processor.load_data()
        
        if interactions_df.empty:
            logger.warning("No interaction data available for training")
            return False
        
        logger.info(f"Loaded {len(interactions_df)} interactions, {len(content_df)} content items, {len(user_df)} users")
        
        # Create matrices
        user_item_matrix = data_processor.create_user_item_matrix(interactions_df)
        content_features = data_processor.create_content_features(content_df)
        user_features = data_processor.create_user_features(user_df, interactions_df, content_df)
        
        # Store matrices
        model_manager.data['user_item_matrix'] = user_item_matrix
        model_manager.data['item_features'] = content_features
        model_manager.data['user_features'] = user_features
        
        # Train models
        trainer = ModelTrainer()
        
        # Matrix factorization models
        trainer.train_svd(user_item_matrix)
        trainer.train_nmf(user_item_matrix)
        
        # Implicit feedback models
        trainer.train_als(user_item_matrix)
        trainer.train_bpr(user_item_matrix)
        
        # Neural model
        trainer.train_neural_cf(interactions_df, content_features)
        
        # Build FAISS indices
        if model_manager.data.get('item_embeddings_svd') is not None:
            trainer.build_faiss_indices(
                model_manager.data['item_embeddings_svd'],
                model_manager.data.get('user_embeddings_svd')
            )
        
        # Save models
        save_models()
        
        logger.info("Model training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in model training: {e}", exc_info=True)
        return False

# Model persistence
def save_models():
    """Save all trained models to disk"""
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Save scikit-learn models
        for name in ['svd', 'nmf']:
            if model_manager.models[name] is not None:
                joblib.dump(model_manager.models[name], os.path.join(model_dir, f'{name}_model.pkl'))
        
        # Save encoders
        for name, encoder in model_manager.encoders.items():
            if encoder is not None:
                joblib.dump(encoder, os.path.join(model_dir, f'{name}_encoder.pkl'))
        
        # Save implicit models
        for name in ['als', 'bpr']:
            if model_manager.models[name] is not None:
                with open(os.path.join(model_dir, f'{name}_model.pkl'), 'wb') as f:
                    pickle.dump(model_manager.models[name], f)
        
        # Save neural model
        if model_manager.models['neural_cf'] is not None:
            torch.save(
                model_manager.models['neural_cf'].state_dict(),
                os.path.join(model_dir, 'neural_cf_model.pth')
            )
            # Save model config
            model_config = {
                'n_users': len(model_manager.data['user_mapping']),
                'n_items': len(model_manager.data['item_mapping']),
                'n_features': model_manager.data['item_features'].shape[1] if model_manager.data['item_features'] is not None else 0
            }
            with open(os.path.join(model_dir, 'neural_cf_config.json'), 'w') as f:
                json.dump(model_config, f)
        
        # Save embeddings
        for name in ['user_embeddings_svd', 'item_embeddings_svd', 'user_embeddings_nmf', 'item_embeddings_nmf']:
            if model_manager.data.get(name) is not None:
                np.save(os.path.join(model_dir, f'{name}.npy'), model_manager.data[name])
        
        # Save FAISS indices
        for name, index in model_manager.indices.items():
            if index is not None:
                faiss.write_index(index, os.path.join(model_dir, f'{name}.idx'))
        
        # Save mappings and metadata
        metadata = {
            'user_mapping': model_manager.data['user_mapping'],
            'item_mapping': model_manager.data['item_mapping'],
            'reverse_user_mapping': model_manager.data['reverse_user_mapping'],
            'reverse_item_mapping': model_manager.data['reverse_item_mapping'],
            'last_update': model_manager.last_update.isoformat() if model_manager.last_update else None
        }
        
        with open(os.path.join(model_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save feature matrices
        if model_manager.data['item_features'] is not None:
            joblib.dump(model_manager.data['item_features'], os.path.join(model_dir, 'item_features.pkl'))
        
        if model_manager.data['user_features'] is not None:
            np.save(os.path.join(model_dir, 'user_features.npy'), model_manager.data['user_features'])
        
        logger.info("Models saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving models: {e}", exc_info=True)

def load_models():
    """Load pre-trained models from disk"""
    model_dir = 'models'
    
    if not os.path.exists(model_dir):
        logger.warning("Model directory not found")
        return False
    
    try:
        # Load metadata first
        metadata_path = os.path.join(model_dir, 'metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
            model_manager.data.update({
                'user_mapping': metadata['user_mapping'],
                'item_mapping': metadata['item_mapping'],
                'reverse_user_mapping': metadata['reverse_user_mapping'],
                'reverse_item_mapping': metadata['reverse_item_mapping']
            })
            
            if metadata.get('last_update'):
                model_manager.last_update = datetime.fromisoformat(metadata['last_update'])
        
        # Load scikit-learn models
        for name in ['svd', 'nmf']:
            model_path = os.path.join(model_dir, f'{name}_model.pkl')
            if os.path.exists(model_path):
                model_manager.models[name] = joblib.load(model_path)
        
        # Load encoders
        for name in ['tfidf', 'count', 'genre_encoder', 'language_encoder']:
            encoder_path = os.path.join(model_dir, f'{name}_encoder.pkl')
            if os.path.exists(encoder_path):
                model_manager.encoders[name] = joblib.load(encoder_path)
        
        # Load implicit models
        for name in ['als', 'bpr']:
            model_path = os.path.join(model_dir, f'{name}_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_manager.models[name] = pickle.load(f)
        
        # Load neural model
        neural_config_path = os.path.join(model_dir, 'neural_cf_config.json')
        neural_model_path = os.path.join(model_dir, 'neural_cf_model.pth')
        
        if os.path.exists(neural_config_path) and os.path.exists(neural_model_path):
            with open(neural_config_path, 'r') as f:
                config = json.load(f)
            
            model = NeuralCF(
                n_users=config['n_users'],
                n_items=config['n_items'],
                n_features=config.get('n_features', 0)
            )
            model.load_state_dict(torch.load(neural_model_path, map_location='cpu'))
            model.eval()
            model_manager.models['neural_cf'] = model
        
        # Load embeddings
        for name in ['user_embeddings_svd', 'item_embeddings_svd', 'user_embeddings_nmf', 'item_embeddings_nmf']:
            embedding_path = os.path.join(model_dir, f'{name}.npy')
            if os.path.exists(embedding_path):
                model_manager.data[name] = np.load(embedding_path)
        
        # Load FAISS indices
        for name in ['faiss_item', 'faiss_user']:
            index_path = os.path.join(model_dir, f'{name}.idx')
            if os.path.exists(index_path):
                model_manager.indices[name] = faiss.read_index(index_path)
        
        # Load feature matrices
        item_features_path = os.path.join(model_dir, 'item_features.pkl')
        if os.path.exists(item_features_path):
            model_manager.data['item_features'] = joblib.load(item_features_path)
        
        user_features_path = os.path.join(model_dir, 'user_features.npy')
        if os.path.exists(user_features_path):
            model_manager.data['user_features'] = np.load(user_features_path)
        
        logger.info("Models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        return False

# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'svd': model_manager.get_model('svd') is not None,
            'nmf': model_manager.get_model('nmf') is not None,
            'als': model_manager.get_model('als') is not None,
            'bpr': model_manager.get_model('bpr') is not None,
            'neural_cf': model_manager.get_model('neural_cf') is not None
        },
        'last_update': model_manager.last_update.isoformat() if model_manager.last_update else None,
        'cache_type': 'redis' if cache_manager.redis_client else 'memory'
    }), 200

@app.route('/train', methods=['POST'])
def train_models_endpoint():
    """Endpoint to trigger model training"""
    try:
        # Check if already training
        if hasattr(train_models_endpoint, 'is_training') and train_models_endpoint.is_training:
            return jsonify({'message': 'Training already in progress'}), 429
        
        train_models_endpoint.is_training = True
        
        # Run training in background
        future = executor.submit(train_all_models)
        result = future.result(timeout=1800)  # 30 minute timeout
        
        if result:
            return jsonify({
                'message': 'Models trained successfully',
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({'message': 'Training completed with warnings'}), 200
            
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        return jsonify({'message': str(e)}), 500
    finally:
        train_models_endpoint.is_training = False

@app.route('/recommend/user', methods=['POST'])
def recommend_for_user():
    """Get personalized recommendations for a user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        n_recommendations = data.get('n_recommendations', 20)
        filters = data.get('filters', {})
        
        if not user_id:
            return jsonify({'message': 'user_id is required'}), 400
        
        # Check cache
        cache_key = f"ml_rec:user:{user_id}:{n_recommendations}:{json.dumps(filters, sort_keys=True)}"
        cached = cache_manager.get(cache_key)
        if cached:
            return jsonify({'recommendations': json.loads(cached)}), 200
        
        # Check if models are loaded
        if not model_manager.is_initialized():
            # Try to load models
            if not load_models():
                # If no models exist, train them
                train_all_models()
        
        # Get recommendations
        recommendations = hybrid_recommender.get_recommendations(
            user_id, n_recommendations, filters
        )
        
        # Cache results
        cache_manager.set(cache_key, json.dumps(recommendations), ttl=1800)
        
        return jsonify({'recommendations': recommendations}), 200
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        return jsonify({'message': str(e)}), 500

@app.route('/recommend/similar', methods=['POST'])
def recommend_similar_items():
    """Get similar items"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        n_similar = data.get('n_similar', 10)
        
        if not content_id:
            return jsonify({'message': 'content_id is required'}), 400
        
        # Check cache
        cache_key = f"ml_rec:similar:{content_id}:{n_similar}"
        cached = cache_manager.get(cache_key)
        if cached:
            return jsonify({'recommendations': json.loads(cached)}), 200
        
        # Get similar items
        if content_id not in model_manager.data['item_mapping']:
            return jsonify({'message': 'Content not found in training data'}), 404
        
        item_idx = model_manager.data['item_mapping'][content_id]
        similar_items = hybrid_recommender._get_similar_items_content(item_idx, n_similar)
        
        recommendations = [item_id for item_id, _ in similar_items]
        
        # Cache results
        cache_manager.set(cache_key, json.dumps(recommendations), ttl=3600)
        
        return jsonify({'recommendations': recommendations}), 200
        
    except Exception as e:
        logger.error(f"Similar items error: {e}", exc_info=True)
        return jsonify({'message': str(e)}), 500

@app.route('/recommend/popular', methods=['POST'])
def recommend_popular():
    """Get popular recommendations"""
    try:
        data = request.get_json()
        region = data.get('region')
        language = data.get('language')
        genres = data.get('genres', [])
        n_items = data.get('n_items', 20)
        
        # Build cache key
        cache_key = f"ml_rec:popular:{region}:{language}:{':'.join(sorted(genres))}:{n_items}"
        cached = cache_manager.get(cache_key)
        if cached:
            return jsonify({'recommendations': json.loads(cached)}), 200
        
        # Get popular items from database
        with SessionLocal() as session:
            query = """
                SELECT id, popularity_score 
                FROM content 
                WHERE 1=1
            """
            params = {}
            
            if language:
                if isinstance(language, list):
                    query += " AND language IN :languages"
                    params['languages'] = tuple(language)
                else:
                    query += " AND language = :language"
                    params['language'] = language
            
            if region:
                query += " AND (region = :region OR region = 'global')"
                params['region'] = region
            
            if genres:
                # Filter by genres using JSON contains
                genre_conditions = []
                for i, genre in enumerate(genres):
                    param_name = f'genre_{i}'
                    genre_conditions.append(f"genres LIKE :{param_name}")
                    params[param_name] = f'%"{genre}"%'
                
                query += f" AND ({' OR '.join(genre_conditions)})"
            
            query += " ORDER BY popularity_score DESC LIMIT :limit"
            params['limit'] = n_items
            
            result = session.execute(text(query), params)
            recommendations = [row[0] for row in result]
        
        # Cache results
        cache_manager.set(cache_key, json.dumps(recommendations), ttl=3600)
        
        return jsonify({'recommendations': recommendations}), 200
        
    except Exception as e:
        logger.error(f"Popular recommendations error: {e}", exc_info=True)
        return jsonify({'message': str(e)}), 500

@app.route('/recommend/cold_start', methods=['POST'])
def recommend_cold_start():
    """Get recommendations for new users (cold start)"""
    try:
        data = request.get_json()
        preferences = data.get('preferences', {})
        n_recommendations = data.get('n_recommendations', 20)
        
        # Extract preferences
        preferred_genres = preferences.get('genres', [])
        preferred_languages = preferences.get('languages', [])
        preferred_types = preferences.get('content_types', [])
        region = preferences.get('region')
        
        recommendations = []
        
        with SessionLocal() as session:
            # Build query based on preferences
            query = """
                SELECT id, popularity_score,
                       CASE 
                           WHEN genres IS NOT NULL THEN genres
                           ELSE '[]'
                       END as genres
                FROM content 
                WHERE popularity_score > 0
            """
            params = {}
            
            if preferred_languages:
                query += " AND language IN :languages"
                params['languages'] = tuple(preferred_languages)
            
            if preferred_types:
                query += " AND content_type IN :types"
                params['types'] = tuple(preferred_types)
            
            if region:
                query += " AND (region = :region OR region = 'global')"
                params['region'] = region
            
            query += " ORDER BY popularity_score DESC LIMIT :limit"
            params['limit'] = n_recommendations * 3  # Get more to filter
            
            result = session.execute(text(query), params)
            
            # Score items based on genre overlap
            scored_items = []
            for row in result:
                item_id, popularity, item_genres_json = row
                
                score = popularity
                
                # Boost score based on genre overlap
                if preferred_genres and item_genres_json:
                    try:
                        item_genres = json.loads(item_genres_json)
                        overlap = len(set(preferred_genres) & set(item_genres))
                        score *= (1 + 0.3 * overlap)
                    except:
                        pass
                
                scored_items.append((item_id, score))
            
            # Sort by score and get top recommendations
            scored_items.sort(key=lambda x: x[1], reverse=True)
            recommendations = [item_id for item_id, _ in scored_items[:n_recommendations]]
        
        return jsonify({'recommendations': recommendations}), 200
        
    except Exception as e:
        logger.error(f"Cold start recommendation error: {e}", exc_info=True)
        return jsonify({'message': str(e)}), 500

@app.route('/learn/rating', methods=['POST'])
def learn_from_rating():
    """Update models based on new rating (online learning)"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        content_id = data.get('content_id')
        rating = data.get('rating')
        
        if not all([user_id, content_id, rating is not None]):
            return jsonify({'message': 'user_id, content_id, and rating are required'}), 400
        
        # Clear user's cache
        cache_pattern = f"ml_rec:user:{user_id}:*"
        # Note: This is a simplified cache clearing - in production, use Redis SCAN
        cache_manager.delete(f"ml_rec:user:{user_id}:20:{{}}")
        
        # Log for future batch retraining
        logger.info(f"New rating: user={user_id}, item={content_id}, rating={rating}")
        
        # In a production system, you might want to:
        # 1. Store this in a streaming queue (Kafka, RabbitMQ)
        # 2. Periodically retrain models with new data
        # 3. Implement true online learning algorithms
        
        return jsonify({'message': 'Rating recorded for model update'}), 200
        
    except Exception as e:
        logger.error(f"Learning error: {e}", exc_info=True)
        return jsonify({'message': str(e)}), 500

@app.route('/explain/<int:user_id>/<int:content_id>', methods=['GET'])
def explain_recommendation(user_id: int, content_id: int):
    """Explain why a content was recommended to a user"""
    try:
        explanation = {
            'user_id': user_id,
            'content_id': content_id,
            'factors': []
        }
        
        # Check if user and item exist in mappings
        if user_id not in model_manager.data['user_mapping']:
            return jsonify({'message': 'User not found in training data'}), 404
        
        if content_id not in model_manager.data['item_mapping']:
            return jsonify({'message': 'Content not found in training data'}), 404
        
        user_idx = model_manager.data['user_mapping'][user_id]
        item_idx = model_manager.data['item_mapping'][content_id]
        
        # Get user's interaction history
        with SessionLocal() as session:
            # Get similar users who liked this content
            similar_users_query = text("""
                SELECT u2.user_id, u2.rating
                FROM ratings u1
                JOIN ratings u2 ON u1.content_id = u2.content_id
                WHERE u1.user_id = :user_id 
                  AND u2.user_id != :user_id
                  AND u2.content_id = :content_id
                  AND u2.rating >= 7
                LIMIT 5
            """)
            
            similar_users = session.execute(
                similar_users_query,
                {'user_id': user_id, 'content_id': content_id}
            ).fetchall()
            
            if similar_users:
                explanation['factors'].append({
                    'type': 'collaborative',
                    'description': f"{len(similar_users)} similar users rated this highly",
                    'weight': 0.4
                })
            
            # Get content similarity
            user_history_query = text("""
                SELECT DISTINCT c.id, c.title, c.genres
                FROM content c
                JOIN ratings r ON c.id = r.content_id
                WHERE r.user_id = :user_id AND r.rating >= 7
                LIMIT 10
            """)
            
            user_liked_content = session.execute(
                user_history_query,
                {'user_id': user_id}
            ).fetchall()
            
            # Check genre overlap
            target_content_query = text("""
                SELECT title, genres FROM content WHERE id = :content_id
            """)
            target_content = session.execute(
                target_content_query,
                {'content_id': content_id}
            ).fetchone()
            
            if target_content and target_content[1]:
                target_genres = set(json.loads(target_content[1]))
                genre_overlaps = []
                
                for _, title, genres_json in user_liked_content:
                    if genres_json:
                        liked_genres = set(json.loads(genres_json))
                        overlap = target_genres & liked_genres
                        if overlap:
                            genre_overlaps.append((title, overlap))
                
                if genre_overlaps:
                    explanation['factors'].append({
                        'type': 'content',
                        'description': f"Similar genres to {len(genre_overlaps)} content you liked",
                        'examples': [f"{title} ({', '.join(genres)})" for title, genres in genre_overlaps[:3]],
                        'weight': 0.2
                    })
        
        # Add model scores if available
        if model_manager.get_model('als'):
            try:
                score = model_manager.get_model('als').similar_items(
                    item_idx, N=1
                )[0][1]
                explanation['factors'].append({
                    'type': 'matrix_factorization',
                    'description': 'High compatibility score from collaborative filtering',
                    'score': float(score),
                    'weight': 0.3
                })
            except:
                pass
        
        # Calculate final score
        total_weight = sum(factor['weight'] for factor in explanation['factors'])
        if total_weight > 0:
            explanation['confidence'] = min(total_weight, 1.0)
        else:
            explanation['confidence'] = 0.1
            explanation['factors'].append({
                'type': 'popularity',
                'description': 'Generally popular content',
                'weight': 0.1
            })
        
        return jsonify(explanation), 200
        
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        return jsonify({'message': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_statistics():
    """Get ML service statistics"""
    try:
        stats = {
            'models': {
                'svd': {
                    'loaded': model_manager.get_model('svd') is not None,
                    'components': model_manager.get_model('svd').n_components if model_manager.get_model('svd') else 0
                },
                'als': {
                    'loaded': model_manager.get_model('als') is not None,
                    'factors': model_manager.get_model('als').factors if model_manager.get_model('als') else 0
                },
                'neural_cf': {
                    'loaded': model_manager.get_model('neural_cf') is not None
                }
            },
            'data': {
                'n_users': len(model_manager.data['user_mapping']),
                'n_items': len(model_manager.data['item_mapping']),
                'matrix_shape': model_manager.data['user_item_matrix'].shape if model_manager.data['user_item_matrix'] is not None else None,
                'matrix_density': (model_manager.data['user_item_matrix'].nnz / 
                                 (model_manager.data['user_item_matrix'].shape[0] * 
                                  model_manager.data['user_item_matrix'].shape[1]))
                                 if model_manager.data['user_item_matrix'] is not None else 0
            },
            'last_update': model_manager.last_update.isoformat() if model_manager.last_update else None,
            'cache_type': 'redis' if cache_manager.redis_client else 'memory'
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Statistics error: {e}", exc_info=True)
        return jsonify({'message': str(e)}), 500

# Initialize models on startup
def initialize_models():
    """Initialize models on application startup"""
    try:
        logger.info("Initializing ML models...")
        
        # Try to load existing models
        if load_models():
            logger.info("Pre-trained models loaded successfully")
            
            # Check if models need updating
            if model_manager.last_update:
                time_since_update = datetime.now() - model_manager.last_update
                if time_since_update.total_seconds() > MODEL_UPDATE_INTERVAL:
                    logger.info("Models are outdated, scheduling retraining...")
                    executor.submit(train_all_models)
        else:
            logger.info("No pre-trained models found, training new models...")
            # Train models in background
            executor.submit(train_all_models)
            
    except Exception as e:
        logger.error(f"Error initializing models: {e}", exc_info=True)

# Run initialization when module loads
initialize_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    
    # Use production server
    try:
        import gunicorn
        logger.info(f"Starting ML service on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except ImportError:
        logger.warning("Gunicorn not available, using development server")
        app.run(host='0.0.0.0', port=port, debug=True)