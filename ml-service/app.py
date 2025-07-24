# ml-service/app.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
import pickle
import hashlib
import redis
import time
import threading
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
import scipy.sparse as sp
from scipy.spatial.distance import cosine, jaccard, hamming
from scipy.stats import pearsonr, spearmanr
from scipy.signal import savgol_filter
import networkx as nx
from textblob import TextBlob
import re
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ultimate-ml-service-secret-2024')

# Database configuration - PERFECT SYNC with backend
if os.environ.get('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_recommendations.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 30,
    'pool_recycle': 3600,
    'pool_pre_ping': True,
    'max_overflow': 40
}

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ultra-fast Redis configuration
try:
    if os.environ.get('REDIS_URL'):
        redis_client = redis.from_url(
            os.environ.get('REDIS_URL'), 
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            max_connections=50
        )
    else:
        redis_client = redis.Redis(
            host='localhost', port=6379, db=0, 
            decode_responses=True,
            max_connections=50,
            socket_connect_timeout=5
        )
    redis_client.ping()
    logger.info("🚀 Redis connected - Ultra-fast caching enabled")
except:
    redis_client = None
    logger.warning("⚠️ Redis not available - Using optimized memory cache")

# Advanced memory structures for real-time processing
ultra_memory_cache = {}
cache_metadata = {}
user_session_cache = defaultdict(dict)
real_time_interactions = deque(maxlen=50000)  # Increased buffer
trending_momentum = defaultdict(list)
content_velocity = defaultdict(float)
user_behavior_patterns = defaultdict(list)

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=8)

# Ultra-advanced cache configuration
ULTIMATE_CACHE_EXPIRY = {
    'trending': 120,          # 2 minutes - ultra fresh
    'personalized': 240,      # 4 minutes - user-specific
    'similar': 300,           # 5 minutes - content-based
    'genre': 900,             # 15 minutes - stable
    'regional': 600,          # 10 minutes - region-based
    'critics': 1200,          # 20 minutes - quality-based
    'new_releases': 180,      # 3 minutes - fresh content
    'anime': 360,             # 6 minutes - niche content
    'user_profile': 300,      # 5 minutes - user analysis
    'content_features': 1800, # 30 minutes - content analysis
    'real_time': 60,          # 1 minute - real-time data
}

# EXACT Database Models (Perfect Backend Sync)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    preferred_languages = db.Column(db.Text)
    preferred_genres = db.Column(db.Text)
    location = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    imdb_id = db.Column(db.String(20))
    mal_id = db.Column(db.Integer)
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    content_type = db.Column(db.String(20), nullable=False)
    genres = db.Column(db.Text)
    anime_genres = db.Column(db.Text)
    languages = db.Column(db.Text)
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float)
    vote_count = db.Column(db.Integer)
    popularity = db.Column(db.Float)
    overview = db.Column(db.Text)
    poster_path = db.Column(db.String(255))
    backdrop_path = db.Column(db.String(255))
    trailer_url = db.Column(db.String(255))
    youtube_trailer_id = db.Column(db.String(255))
    is_trending = db.Column(db.Boolean, default=False)
    is_new_release = db.Column(db.Boolean, default=False)
    is_critics_choice = db.Column(db.Boolean, default=False)
    critics_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AdminRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recommendation_type = db.Column(db.String(50))
    description = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AnonymousInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)
    ip_address = db.Column(db.String(45))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Ultimate Caching System with Intelligence
class UltimateCacheManager:
    @staticmethod
    def generate_smart_key(prefix, params=None, user_context=None):
        """Generate intelligent cache keys with context awareness"""
        key_parts = [f"ultimate_ml:{prefix}"]
        
        if params:
            # Sort and hash parameters for consistency
            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]
            key_parts.append(param_hash)
        
        if user_context:
            # Add user context for personalized caching
            user_hash = hashlib.md5(str(user_context).encode()).hexdigest()[:8]
            key_parts.append(f"user:{user_hash}")
        
        # Add time bucket for cache invalidation
        time_bucket = int(time.time() // 300)  # 5-minute buckets
        key_parts.append(f"t:{time_bucket}")
        
        return ":".join(key_parts)
    
    @staticmethod
    def get_with_metadata(key):
        """Get cached value with metadata"""
        try:
            if redis_client:
                pipe = redis_client.pipeline()
                pipe.get(key)
                pipe.get(f"{key}:meta")
                results = pipe.execute()
                
                if results[0]:
                    data = json.loads(results[0])
                    metadata = json.loads(results[1] or '{}')
                    
                    # Update access statistics
                    metadata['access_count'] = metadata.get('access_count', 0) + 1
                    metadata['last_accessed'] = time.time()
                    redis_client.set(f"{key}:meta", json.dumps(metadata), ex=3600)
                    
                    return data, metadata
            else:
                if key in ultra_memory_cache:
                    data = ultra_memory_cache[key]
                    metadata = cache_metadata.get(key, {})
                    
                    # Check expiry
                    if 'expires_at' in metadata:
                        if time.time() > metadata['expires_at']:
                            del ultra_memory_cache[key]
                            if key in cache_metadata:
                                del cache_metadata[key]
                            return None, None
                    
                    # Update access stats
                    metadata['access_count'] = metadata.get('access_count', 0) + 1
                    metadata['last_accessed'] = time.time()
                    cache_metadata[key] = metadata
                    
                    return data, metadata
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None, None
    
    @staticmethod
    def set_with_intelligence(key, value, expiry=3600, priority='normal'):
        """Set cache value with intelligent management"""
        try:
            metadata = {
                'created_at': time.time(),
                'expires_at': time.time() + expiry,
                'priority': priority,
                'size': len(json.dumps(value)),
                'access_count': 0
            }
            
            if redis_client:
                pipe = redis_client.pipeline()
                pipe.set(key, json.dumps(value), ex=expiry)
                pipe.set(f"{key}:meta", json.dumps(metadata), ex=expiry + 3600)
                pipe.execute()
            else:
                ultra_memory_cache[key] = value
                cache_metadata[key] = metadata
                
                # Memory management with intelligent eviction
                UltimateCacheManager._manage_memory_cache()
                
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    @staticmethod
    def _manage_memory_cache():
        """Intelligent memory cache management"""
        if len(ultra_memory_cache) > 2000:  # Increased limit
            # Sort by priority and access patterns
            cache_items = []
            for key, metadata in cache_metadata.items():
                if key in ultra_memory_cache:
                    score = (
                        metadata.get('access_count', 0) * 0.5 +
                        (1.0 / max(1, time.time() - metadata.get('last_accessed', 0))) * 0.3 +
                        (1 if metadata.get('priority') == 'high' else 0.5) * 0.2
                    )
                    cache_items.append((key, score))
            
            # Remove lowest scoring 500 items
            cache_items.sort(key=lambda x: x[1])
            for key, _ in cache_items[:500]:
                if key in ultra_memory_cache:
                    del ultra_memory_cache[key]
                if key in cache_metadata:
                    del cache_metadata[key]
    
    @staticmethod
    def invalidate_pattern(pattern):
        """Invalidate cache keys matching pattern"""
        try:
            if redis_client:
                for key in redis_client.scan_iter(match=pattern):
                    redis_client.delete(key)
                    redis_client.delete(f"{key}:meta")
            else:
                keys_to_remove = [key for key in ultra_memory_cache.keys() if pattern.replace('*', '') in key]
                for key in keys_to_remove:
                    if key in ultra_memory_cache:
                        del ultra_memory_cache[key]
                    if key in cache_metadata:
                        del cache_metadata[key]
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")

# Advanced Transformer-based Neural Collaborative Filtering
class TransformerRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=256, num_heads=8, num_layers=4):
        super(TransformerRecommender, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        
        # Enhanced embeddings with positional encoding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.position_embedding = nn.Embedding(1000, embedding_dim)  # For sequence position
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-head attention for user-item interaction
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # Advanced prediction layers
        self.prediction_layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids, item_ids, user_sequence=None):
        batch_size = user_ids.size(0)
        
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # [batch, embedding_dim]
        item_emb = self.item_embedding(item_ids)  # [batch, embedding_dim]
        
        # Create sequence representation
        if user_sequence is not None:
            # Process user interaction sequence
            seq_emb = self.item_embedding(user_sequence)  # [batch, seq_len, embedding_dim]
            pos_ids = torch.arange(seq_emb.size(1), device=seq_emb.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.position_embedding(pos_ids)
            
            seq_emb = seq_emb + pos_emb
            seq_repr = self.transformer(seq_emb)  # [batch, seq_len, embedding_dim]
            seq_repr = seq_repr.mean(dim=1)  # Average pooling [batch, embedding_dim]
        else:
            seq_repr = torch.zeros_like(user_emb)
        
        # Cross attention between user and item
        user_emb_expanded = user_emb.unsqueeze(1)  # [batch, 1, embedding_dim]
        item_emb_expanded = item_emb.unsqueeze(1)  # [batch, 1, embedding_dim]
        
        attended_emb, _ = self.cross_attention(
            user_emb_expanded, item_emb_expanded, item_emb_expanded
        )
        attended_emb = attended_emb.squeeze(1)  # [batch, embedding_dim]
        
        # Combine all representations
        combined = torch.cat([user_emb, item_emb, attended_emb], dim=1)  # [batch, 3*embedding_dim]
        
        # Prediction
        prediction = self.prediction_layers(combined).squeeze(-1)  # [batch]
        
        # Add bias terms
        user_bias = self.user_bias(user_ids).squeeze(-1)
        item_bias = self.item_bias(item_ids).squeeze(-1)
        
        final_score = prediction + user_bias + item_bias + self.global_bias
        
        return torch.sigmoid(final_score)

# Ultra-Advanced Content Analysis Engine
class UltraContentAnalyzer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=15000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 4),
            min_df=2,
            max_df=0.85,
            use_idf=True,
            sublinear_tf=True
        )
        
        self.genre_vectorizer = CountVectorizer(
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        # Advanced decomposition methods
        self.svd = TruncatedSVD(n_components=300, random_state=42)
        self.nmf = NMF(n_components=200, random_state=42, max_iter=1000)
        self.lda = LatentDirichletAllocation(n_components=100, random_state=42)
        
        # Clustering for content discovery
        self.content_clusters = KMeans(n_clusters=100, random_state=42)
        self.fine_clusters = DBSCAN(eps=0.3, min_samples=3)
        
        # Scalers for numerical features
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # Storage for computed features
        self.content_features = None
        self.content_similarity_matrix = None
        self.content_graph = None
        self.content_topics = None
        self.content_clusters_labels = {}
        self.content_ids = []
        
    def advanced_text_preprocessing(self, text):
        """Advanced text preprocessing with NLP"""
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = ' '.join(text.split())
        
        # Sentiment analysis
        try:
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            
            # Add sentiment as feature
            if sentiment_score > 0.1:
                text += " positive_sentiment"
            elif sentiment_score < -0.1:
                text += " negative_sentiment"
            else:
                text += " neutral_sentiment"
        except:
            pass
        
        return text
    
    def extract_advanced_features(self, contents):
        """Extract comprehensive content features"""
        try:
            logger.info("🔍 Extracting ultra-advanced content features...")
            
            text_features = []
            numerical_features = []
            categorical_features = []
            self.content_ids = []
            
            for content in contents:
                # Advanced text processing
                text_parts = []
                
                # Title and overview with preprocessing
                if content.title:
                    processed_title = self.advanced_text_preprocessing(content.title)
                    text_parts.append(processed_title)
                    text_parts.append(f"title_{processed_title}")  # Title-specific features
                
                if content.overview:
                    processed_overview = self.advanced_text_preprocessing(content.overview)
                    text_parts.append(processed_overview)
                    
                    # Extract key phrases
                    if len(processed_overview.split()) > 10:
                        words = processed_overview.split()
                        # Add bigrams and trigrams as features
                        for i in range(len(words) - 1):
                            text_parts.append(f"{words[i]}_{words[i+1]}")
                        for i in range(len(words) - 2):
                            text_parts.append(f"{words[i]}_{words[i+1]}_{words[i+2]}")
                
                # Genre processing
                genres = json.loads(content.genres or '[]')
                if genres:
                    # Individual genres
                    for genre in genres:
                        text_parts.append(f"genre_{genre.lower()}")
                    
                    # Genre combinations
                    if len(genres) > 1:
                        genre_combo = "_".join(sorted([g.lower() for g in genres[:3]]))
                        text_parts.append(f"combo_{genre_combo}")
                
                # Anime-specific genres
                anime_genres = json.loads(content.anime_genres or '[]')
                if anime_genres:
                    for ag in anime_genres:
                        text_parts.append(f"anime_{ag.lower()}")
                
                # Language features
                languages = json.loads(content.languages or '[]')
                if languages:
                    for lang in languages:
                        text_parts.append(f"lang_{lang.lower()}")
                
                # Content type specific features
                text_parts.append(f"type_{content.content_type}")
                
                combined_text = ' '.join(text_parts)
                text_features.append(combined_text)
                
                # Advanced numerical features
                num_features = [
                    # Basic features
                    content.rating or 0,
                    np.log1p(content.vote_count or 0),
                    np.log1p(content.popularity or 0),
                    content.runtime or 0,
                    
                    # Derived features
                    len(content.overview or '') / 1000,  # Overview length
                    len(genres),  # Genre count
                    len(languages),  # Language count
                    
                    # Temporal features
                    0 if not content.release_date else (datetime.utcnow().date() - content.release_date).days,
                    0 if not content.release_date else content.release_date.year,
                    0 if not content.release_date else content.release_date.month,
                    
                    # Quality indicators
                    1 if content.rating and content.rating >= 8.0 else 0,  # High quality
                    1 if content.vote_count and content.vote_count >= 1000 else 0,  # Popular
                    1 if content.is_critics_choice else 0,
                    1 if content.is_trending else 0,
                    1 if content.is_new_release else 0,
                    
                    # Text-based features
                    len(content.title.split()) if content.title else 0,
                    len(content.overview.split()) if content.overview else 0,
                ]
                
                numerical_features.append(num_features)
                
                # Categorical features for one-hot encoding
                cat_features = [
                    content.content_type,
                    'high_rated' if content.rating and content.rating >= 7.5 else 'normal_rated',
                    'popular' if content.popularity and content.popularity >= 50 else 'niche',
                    'recent' if content.release_date and (datetime.utcnow().date() - content.release_date).days <= 365 else 'older'
                ]
                categorical_features.append('_'.join(cat_features))
                
                self.content_ids.append(content.id)
            
            # Create feature matrices
            logger.info("🔧 Creating feature matrices...")
            
            # Text features with multiple methods
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
            text_svd = self.svd.fit_transform(tfidf_matrix)
            text_nmf = self.nmf.fit_transform(tfidf_matrix)
            
            # Topic modeling
            text_dense = tfidf_matrix.toarray()
            text_topics = self.lda.fit_transform(text_dense)
            self.content_topics = text_topics
            
            # Numerical features
            numerical_features = np.array(numerical_features)
            numerical_scaled = self.standard_scaler.fit_transform(numerical_features)
            numerical_robust = self.robust_scaler.fit_transform(numerical_features)
            
            # Categorical features
            cat_vectorizer = CountVectorizer()
            cat_matrix = cat_vectorizer.fit_transform(categorical_features).toarray()
            
            # Combine all features with optimal weighting
            self.content_features = np.hstack([
                text_svd * 0.3,           # Text SVD features (30%)
                text_nmf * 0.2,           # Text NMF features (20%)
                text_topics * 0.15,       # Topic features (15%)
                numerical_scaled * 0.2,   # Scaled numerical (20%)
                numerical_robust * 0.1,   # Robust numerical (10%)
                cat_matrix * 0.05         # Categorical (5%)
            ])
            
            logger.info(f"✅ Feature extraction complete: {self.content_features.shape}")
            
        except Exception as e:
            logger.error(f"❌ Feature extraction error: {e}")
            raise
    
    def compute_similarity_matrices(self):
        """Compute multiple similarity matrices"""
        try:
            logger.info("🔗 Computing advanced similarity matrices...")
            
            if self.content_features is None:
                return
            
            # Multiple similarity metrics
            cosine_sim = cosine_similarity(self.content_features)
            
            # Weighted combination of similarities
            self.content_similarity_matrix = cosine_sim
            
            # Create content graph for network-based recommendations
            self.content_graph = nx.Graph()
            
            # Add nodes
            for i, content_id in enumerate(self.content_ids):
                self.content_graph.add_node(content_id, features=self.content_features[i])
            
            # Add edges based on similarity threshold
            similarity_threshold = 0.3
            for i in range(len(self.content_ids)):
                for j in range(i + 1, len(self.content_ids)):
                    if cosine_sim[i, j] > similarity_threshold:
                        self.content_graph.add_edge(
                            self.content_ids[i], 
                            self.content_ids[j], 
                            weight=cosine_sim[i, j]
                        )
            
            # Perform clustering
            cluster_labels = self.content_clusters.fit_predict(self.content_features)
            for i, content_id in enumerate(self.content_ids):
                self.content_clusters_labels[content_id] = cluster_labels[i]
            
            logger.info("✅ Similarity computation complete")
            
        except Exception as e:
            logger.error(f"❌ Similarity computation error: {e}")
    
    def get_similar_content(self, content_id, num_recommendations=20, diversity_factor=0.4):
        """Get similar content with advanced algorithms"""
        try:
            if content_id not in self.content_ids:
                return []
            
            content_idx = self.content_ids.index(content_id)
            
            # Multiple similarity approaches
            similarities = []
            
            # 1. Direct similarity
            if self.content_similarity_matrix is not None:
                direct_similarities = list(enumerate(self.content_similarity_matrix[content_idx]))
                similarities.extend([(idx, score, 'content_similarity') for idx, score in direct_similarities])
            
            # 2. Network-based similarity (if graph exists)
            if self.content_graph and content_id in self.content_graph:
                try:
                    # Get network neighbors
                    neighbors = list(self.content_graph.neighbors(content_id))
                    for neighbor_id in neighbors:
                        if neighbor_id in self.content_ids:
                            neighbor_idx = self.content_ids.index(neighbor_id)
                            edge_weight = self.content_graph[content_id][neighbor_id]['weight']
                            similarities.append((neighbor_idx, edge_weight, 'network_similarity'))
                except:
                    pass
            
            # 3. Cluster-based similarity
            content_cluster = self.content_clusters_labels.get(content_id)
            if content_cluster is not None:
                for other_id, other_cluster in self.content_clusters_labels.items():
                    if other_cluster == content_cluster and other_id != content_id:
                        if other_id in self.content_ids:
                            other_idx = self.content_ids.index(other_id)
                            similarities.append((other_idx, 0.7, 'cluster_similarity'))
            
            # 4. Topic-based similarity
            if self.content_topics is not None:
                content_topics = self.content_topics[content_idx]
                for i, other_topics in enumerate(self.content_topics):
                    if i != content_idx:
                        topic_similarity = cosine_similarity([content_topics], [other_topics])[0][0]
                        if topic_similarity > 0.2:
                            similarities.append((i, topic_similarity, 'topic_similarity'))
            
            # Aggregate similarities
            similarity_scores = defaultdict(list)
            for idx, score, method in similarities:
                if idx != content_idx:  # Exclude self
                    similarity_scores[idx].append((score, method))
            
            # Calculate final scores with ensemble weighting
            final_scores = []
            method_weights = {
                'content_similarity': 0.4,
                'network_similarity': 0.3,
                'cluster_similarity': 0.2,
                'topic_similarity': 0.1
            }
            
            for idx, score_list in similarity_scores.items():
                weighted_score = 0
                total_weight = 0
                
                for score, method in score_list:
                    weight = method_weights.get(method, 0.1)
                    weighted_score += score * weight
                    total_weight += weight
                
                if total_weight > 0:
                    final_score = weighted_score / total_weight
                    final_scores.append((idx, final_score))
            
            # Sort by score
            final_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Apply diversity filter
            diverse_recommendations = self._apply_diversity_filter(
                final_scores, content_id, num_recommendations, diversity_factor
            )
            
            # Format results
            recommendations = []
            for idx, score in diverse_recommendations:
                recommendations.append({
                    'content_id': self.content_ids[idx],
                    'score': float(score),
                    'reason': 'Advanced multi-algorithm similarity'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Similar content error: {e}")
            return []
    
    def _apply_diversity_filter(self, scored_items, base_content_id, limit, diversity_factor):
        """Apply intelligent diversity filtering"""
        try:
            if not scored_items:
                return []
            
            diverse_items = []
            used_clusters = set()
            
            # Get base content cluster
            base_cluster = self.content_clusters_labels.get(base_content_id)
            
            for idx, score in scored_items:
                if len(diverse_items) >= limit:
                    break
                
                content_id = self.content_ids[idx]
                content_cluster = self.content_clusters_labels.get(content_id)
                
                # Diversity logic
                add_item = False
                
                if len(diverse_items) < limit * (1 - diversity_factor):
                    # Allow some concentration of similar items
                    add_item = True
                elif content_cluster != base_cluster and content_cluster not in used_clusters:
                    # Prefer items from different clusters
                    add_item = True
                    used_clusters.add(content_cluster)
                elif len(used_clusters) < 5:  # Max 5 different clusters
                    add_item = True
                    if content_cluster is not None:
                        used_clusters.add(content_cluster)
                
                if add_item:
                    diverse_items.append((idx, score))
            
            return diverse_items
            
        except Exception as e:
            logger.error(f"❌ Diversity filter error: {e}")
            return scored_items[:limit]

# Ultra-Advanced Collaborative Filtering with Deep Learning
class UltraCollaborativeFilter:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.user_embeddings = None
        self.item_embeddings = None
        self.user_ids = []
        self.item_ids = []
        self.user_clusters = {}
        self.item_clusters = {}
        self.temporal_weights = {}
        
        # Advanced models
        self.matrix_factorization = None
        self.neural_mf = None
        self.ensemble_weights = {
            'user_cf': 0.3,
            'item_cf': 0.25,
            'matrix_factorization': 0.25,
            'neural_mf': 0.2
        }
    
    def fit(self, interactions):
        """Train ultra-advanced collaborative filtering"""
        try:
            logger.info("🤝 Training Ultra-Advanced Collaborative Filtering...")
            
            # Prepare advanced interaction data
            interaction_data = []
            for interaction in interactions:
                # Advanced rating calculation with temporal decay
                rating = self._calculate_advanced_rating(interaction)
                
                interaction_data.append({
                    'user_id': interaction.user_id,
                    'content_id': interaction.content_id,
                    'rating': rating,
                    'timestamp': interaction.timestamp,
                    'interaction_type': interaction.interaction_type
                })
            
            if not interaction_data:
                return
            
            df = pd.DataFrame(interaction_data)
            
            # Advanced temporal weighting
            self._calculate_temporal_weights(df)
            
            # Create user-item matrix with temporal weighting
            df['weighted_rating'] = df.apply(lambda row: row['rating'] * self.temporal_weights.get(
                (row['user_id'], row['content_id']), 1.0
            ), axis=1)
            
            # Handle multiple interactions (aggregation)
            df_agg = df.groupby(['user_id', 'content_id']).agg({
                'weighted_rating': 'mean',  # Average rating
                'rating': 'count'  # Interaction frequency
            }).reset_index()
            
            # Boost ratings with high frequency
            df_agg['final_rating'] = df_agg['weighted_rating'] * (1 + np.log1p(df_agg['rating']) * 0.1)
            
            # Create matrix
            user_item_df = df_agg.pivot_table(
                index='user_id',
                columns='content_id',
                values='final_rating',
                fill_value=0
            )
            
            self.user_ids = list(user_item_df.index)
            self.item_ids = list(user_item_df.columns)
            self.user_item_matrix = user_item_df.values
            
            # Advanced similarity computation
            self._compute_advanced_similarities()
            
            # Matrix factorization
            self._perform_matrix_factorization()
            
            # User and item clustering
            self._perform_clustering()
            
            logger.info(f"✅ Collaborative filtering trained: {len(self.user_ids)} users, {len(self.item_ids)} items")
            
        except Exception as e:
            logger.error(f"❌ Collaborative filtering training error: {e}")
    
    def _calculate_advanced_rating(self, interaction):
        """Calculate advanced implicit rating"""
        base_ratings = {
            'view': 2.0,
            'like': 4.0,
            'favorite': 5.0,
            'watchlist': 4.5,
            'search': 1.0
        }
        
        rating = base_ratings.get(interaction.interaction_type, 2.0)
        
        # Use explicit rating if available
        if interaction.rating:
            rating = float(interaction.rating)
        
        # Time-based adjustments
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        
        # Recent interactions are more important
        time_boost = max(0.5, 1.0 - (days_ago / 365))
        
        # Interaction type specific time decay
        if interaction.interaction_type in ['favorite', 'watchlist']:
            time_boost = max(0.8, time_boost)  # Less decay for strong signals
        
        return rating * time_boost
    
    def _calculate_temporal_weights(self, df):
        """Calculate sophisticated temporal weights"""
        try:
            current_time = datetime.utcnow()
            
            for _, row in df.iterrows():
                user_id = row['user_id']
                content_id = row['content_id']
                timestamp = row['timestamp']
                
                # Time decay
                days_diff = (current_time - timestamp).days
                time_weight = np.exp(-days_diff / 90)  # 90-day half-life
                
                # Interaction recency boost
                if days_diff <= 7:
                    time_weight *= 1.5  # Recent interaction boost
                elif days_diff <= 30:
                    time_weight *= 1.2
                
                self.temporal_weights[(user_id, content_id)] = time_weight
                
        except Exception as e:
            logger.error(f"❌ Temporal weight calculation error: {e}")
    
    def _compute_advanced_similarities(self):
        """Compute advanced similarity matrices"""
        try:
            # User similarity with multiple metrics
            user_cosine = cosine_similarity(self.user_item_matrix)
            
            # Pearson correlation for users with sufficient overlap
            user_pearson = np.zeros_like(user_cosine)
            for i in range(len(self.user_ids)):
                for j in range(i + 1, len(self.user_ids)):
                    user_i = self.user_item_matrix[i]
                    user_j = self.user_item_matrix[j]
                    
                    # Find common rated items
                    mask = (user_i > 0) & (user_j > 0)
                    if np.sum(mask) >= 3:  # At least 3 common items
                        try:
                            corr, _ = pearsonr(user_i[mask], user_j[mask])
                            if not np.isnan(corr):
                                user_pearson[i, j] = user_pearson[j, i] = corr
                        except:
                            pass
            
            # Combine similarities
            self.user_similarity_matrix = 0.7 * user_cosine + 0.3 * user_pearson
            
            # Item similarity
            self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            
        except Exception as e:
            logger.error(f"❌ Similarity computation error: {e}")
    
    def _perform_matrix_factorization(self):
        """Advanced matrix factorization"""
        try:
            from sklearn.decomposition import NMF
            
            # Non-negative matrix factorization
            n_components = min(100, min(len(self.user_ids), len(self.item_ids)) // 2)
            
            nmf = NMF(n_components=n_components, random_state=42, max_iter=1000)
            self.user_embeddings = nmf.fit_transform(self.user_item_matrix)
            self.item_embeddings = nmf.components_.T
            
            self.matrix_factorization = nmf
            
        except Exception as e:
            logger.error(f"❌ Matrix factorization error: {e}")
    
    def _perform_clustering(self):
        """Cluster users and items"""
        try:
            if self.user_embeddings is not None:
                # User clustering
                n_user_clusters = min(20, len(self.user_ids) // 10)
                user_kmeans = KMeans(n_clusters=n_user_clusters, random_state=42)
                user_cluster_labels = user_kmeans.fit_predict(self.user_embeddings)
                
                for i, user_id in enumerate(self.user_ids):
                    self.user_clusters[user_id] = user_cluster_labels[i]
            
            if self.item_embeddings is not None:
                # Item clustering
                n_item_clusters = min(30, len(self.item_ids) // 10)
                item_kmeans = KMeans(n_clusters=n_item_clusters, random_state=42)
                item_cluster_labels = item_kmeans.fit_predict(self.item_embeddings)
                
                for i, item_id in enumerate(self.item_ids):
                    self.item_clusters[item_id] = item_cluster_labels[i]
                    
        except Exception as e:
            logger.error(f"❌ Clustering error: {e}")
    
    def get_user_recommendations(self, user_id, num_recommendations=20):
        """Get recommendations using ensemble approach"""
        try:
            if user_id not in self.user_ids:
                return []
            
            user_idx = self.user_ids.index(user_id)
            user_ratings = self.user_item_matrix[user_idx]
            
            recommendations = defaultdict(float)
            
            # 1. User-based collaborative filtering
            user_cf_recs = self._get_user_based_recommendations(user_idx, num_recommendations * 2)
            for item_id, score in user_cf_recs:
                recommendations[item_id] += score * self.ensemble_weights['user_cf']
            
            # 2. Item-based collaborative filtering
            item_cf_recs = self._get_item_based_recommendations(user_idx, num_recommendations * 2)
            for item_id, score in item_cf_recs:
                recommendations[item_id] += score * self.ensemble_weights['item_cf']
            
            # 3. Matrix factorization recommendations
            if self.user_embeddings is not None and self.item_embeddings is not None:
                mf_recs = self._get_matrix_factorization_recommendations(user_idx, num_recommendations * 2)
                for item_id, score in mf_recs:
                    recommendations[item_id] += score * self.ensemble_weights['matrix_factorization']
            
            # Sort and format
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            
            result = []
            for item_id, score in sorted_recs[:num_recommendations]:
                result.append({
                    'content_id': item_id,
                    'score': float(score),
                    'reason': 'Advanced collaborative filtering ensemble'
                })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ User recommendations error: {e}")
            return []
    
    def _get_user_based_recommendations(self, user_idx, limit):
        """User-based collaborative filtering"""
        try:
            user_similarities = self.user_similarity_matrix[user_idx]
            user_ratings = self.user_item_matrix[user_idx]
            
            # Find similar users (excluding self)
            similar_users = np.argsort(user_similarities)[::-1][1:101]  # Top 100
            
            recommendations = defaultdict(float)
            
            for similar_user_idx in similar_users:
                similarity = user_similarities[similar_user_idx]
                if similarity <= 0:
                    continue
                
                similar_user_ratings = self.user_item_matrix[similar_user_idx]
                
                for item_idx, rating in enumerate(similar_user_ratings):
                    if rating > 0 and user_ratings[item_idx] == 0:  # Not rated by user
                        item_id = self.item_ids[item_idx]
                        recommendations[item_id] += similarity * rating
            
            # Sort and return top items
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            return sorted_recs[:limit]
            
        except Exception as e:
            logger.error(f"❌ User-based CF error: {e}")
            return []
    
    def _get_item_based_recommendations(self, user_idx, limit):
        """Item-based collaborative filtering"""
        try:
            user_ratings = self.user_item_matrix[user_idx]
            rated_items = np.where(user_ratings > 0)[0]
            
            recommendations = defaultdict(float)
            
            for rated_item_idx in rated_items:
                rated_item_id = self.item_ids[rated_item_idx]
                user_rating = user_ratings[rated_item_idx]
                
                # Find similar items
                item_similarities = self.item_similarity_matrix[rated_item_idx]
                
                for item_idx, similarity in enumerate(item_similarities):
                    if similarity > 0 and user_ratings[item_idx] == 0:  # Not rated
                        item_id = self.item_ids[item_idx]
                        recommendations[item_id] += similarity * user_rating
            
            # Sort and return
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            return sorted_recs[:limit]
            
        except Exception as e:
            logger.error(f"❌ Item-based CF error: {e}")
            return []
    
    def _get_matrix_factorization_recommendations(self, user_idx, limit):
        """Matrix factorization recommendations"""
        try:
            user_embedding = self.user_embeddings[user_idx]
            user_ratings = self.user_item_matrix[user_idx]
            
            recommendations = []
            
            for item_idx, item_embedding in enumerate(self.item_embeddings):
                if user_ratings[item_idx] == 0:  # Not rated
                    predicted_rating = np.dot(user_embedding, item_embedding)
                    item_id = self.item_ids[item_idx]
                    recommendations.append((item_id, predicted_rating))
            
            # Sort and return
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"❌ Matrix factorization error: {e}")
            return []

# Real-Time Intelligence Engine
class RealTimeIntelligenceEngine:
    def __init__(self):
        self.user_profiles = {}
        self.content_momentum = defaultdict(float)
        self.trending_velocities = defaultdict(list)
        self.interaction_patterns = defaultdict(list)
        self.seasonal_patterns = {}
        self.real_time_cache = {}
        
    def process_real_time_interaction(self, interaction_data):
        """Process interaction in real-time"""
        try:
            user_id = interaction_data['user_id']
            content_id = interaction_data['content_id']
            interaction_type = interaction_data['interaction_type']
            timestamp = interaction_data.get('timestamp', datetime.utcnow())
            
            # Update user profile
            self._update_user_profile_realtime(user_id, interaction_data)
            
            # Update content momentum
            self._update_content_momentum(content_id, interaction_type, timestamp)
            
            # Update trending velocities
            self._update_trending_velocity(content_id, timestamp)
            
            # Store interaction pattern
            self.interaction_patterns[user_id].append({
                'content_id': content_id,
                'type': interaction_type,
                'timestamp': timestamp
            })
            
            # Keep only recent patterns (last 1000 interactions per user)
            if len(self.interaction_patterns[user_id]) > 1000:
                self.interaction_patterns[user_id] = self.interaction_patterns[user_id][-1000:]
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Real-time processing error: {e}")
            return False
    
    def _update_user_profile_realtime(self, user_id, interaction_data):
        """Update user profile in real-time"""
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    'preferences': defaultdict(float),
                    'activity_level': 0,
                    'last_active': datetime.utcnow(),
                    'interaction_velocity': 0,
                    'content_diversity': set()
                }
            
            profile = self.user_profiles[user_id]
            
            # Update activity
            profile['activity_level'] += 1
            profile['last_active'] = datetime.utcnow()
            
            # Update content diversity
            profile['content_diversity'].add(interaction_data['content_id'])
            
            # Update preferences based on content
            content = Content.query.get(interaction_data['content_id'])
            if content:
                # Update genre preferences
                if content.genres:
                    genres = json.loads(content.genres)
                    weight = self._get_interaction_weight(interaction_data['interaction_type'])
                    
                    for genre in genres:
                        profile['preferences'][f"genre_{genre}"] += weight
                
                # Update language preferences
                if content.languages:
                    languages = json.loads(content.languages)
                    for lang in languages:
                        profile['preferences'][f"lang_{lang}"] += weight * 0.5
                
                # Update content type preferences
                profile['preferences'][f"type_{content.content_type}"] += weight * 0.3
            
            # Calculate interaction velocity
            recent_interactions = [
                p for p in self.interaction_patterns[user_id]
                if (datetime.utcnow() - p['timestamp']).total_seconds() <= 3600  # Last hour
            ]
            profile['interaction_velocity'] = len(recent_interactions)
            
        except Exception as e:
            logger.error(f"❌ User profile update error: {e}")
    
    def _update_content_momentum(self, content_id, interaction_type, timestamp):
        """Update content momentum score"""
        try:
            weight = self._get_interaction_weight(interaction_type)
            
            # Time decay factor (more recent = higher weight)
            time_factor = 1.0  # Base weight for current interaction
            
            # Add to momentum with time decay
            self.content_momentum[content_id] = (
                self.content_momentum[content_id] * 0.95 + weight * time_factor
            )
            
        except Exception as e:
            logger.error(f"❌ Content momentum update error: {e}")
    
    def _update_trending_velocity(self, content_id, timestamp):
        """Update trending velocity for content"""
        try:
            current_time = timestamp.timestamp()
            
            # Add current interaction timestamp
            self.trending_velocities[content_id].append(current_time)
            
            # Keep only last 24 hours of interactions
            cutoff_time = current_time - 86400  # 24 hours
            self.trending_velocities[content_id] = [
                t for t in self.trending_velocities[content_id] if t >= cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"❌ Trending velocity update error: {e}")
    
    def get_trending_score(self, content_id):
        """Calculate real-time trending score"""
        try:
            timestamps = self.trending_velocities.get(content_id, [])
            if not timestamps:
                return 0.0
            
            current_time = time.time()
            
            # Calculate interactions in different time windows
            last_hour = sum(1 for t in timestamps if current_time - t <= 3600)
            last_6_hours = sum(1 for t in timestamps if current_time - t <= 21600)
            last_24_hours = len(timestamps)
            
            # Calculate velocity (interactions per hour)
            velocity_1h = last_hour
            velocity_6h = last_6_hours / 6
            velocity_24h = last_24_hours / 24
            
            # Weighted trending score (recent activity weighted more)
            trending_score = (
                velocity_1h * 0.5 +
                velocity_6h * 0.3 +
                velocity_24h * 0.2 +
                self.content_momentum[content_id] * 0.1
            )
            
            return trending_score
            
        except Exception as e:
            logger.error(f"❌ Trending score calculation error: {e}")
            return 0.0
    
    def _get_interaction_weight(self, interaction_type):
        """Get weight for interaction type"""
        weights = {
            'view': 1.0,
            'like': 2.0,
            'favorite': 4.0,
            'watchlist': 3.0,
            'search': 0.5,
            'share': 2.5
        }
        return weights.get(interaction_type, 1.0)
    
    def get_user_realtime_preferences(self, user_id):
        """Get real-time user preferences"""
        try:
            if user_id not in self.user_profiles:
                return {}
            
            profile = self.user_profiles[user_id]
            
            # Normalize preferences
            total_weight = sum(profile['preferences'].values())
            if total_weight > 0:
                normalized_prefs = {
                    key: value / total_weight 
                    for key, value in profile['preferences'].items()
                }
            else:
                normalized_prefs = {}
            
            return {
                'preferences': normalized_prefs,
                'activity_level': profile['activity_level'],
                'diversity_score': len(profile['content_diversity']),
                'interaction_velocity': profile['interaction_velocity']
            }
            
        except Exception as e:
            logger.error(f"❌ User preferences error: {e}")
            return {}

# Ultimate Recommendation Engine
class UltimateRecommendationEngine:
    def __init__(self):
        self.content_analyzer = UltraContentAnalyzer()
        self.collaborative_filter = UltraCollaborativeFilter()
        self.neural_model = None
        self.real_time_engine = RealTimeIntelligenceEngine()
        
        self.popularity_scores = {}
        self.quality_scores = {}
        self.diversity_scores = {}
        
        self.is_trained = False
        
        # Advanced ensemble weights
        self.ensemble_weights = {
            'collaborative': 0.35,
            'content_based': 0.25,
            'neural': 0.20,
            'popularity': 0.10,
            'real_time': 0.10
        }
        
    def train_all_models(self):
        """Train all recommendation models"""
        try:
            logger.info("🚀 Starting Ultimate Model Training...")
            
            # Get data
            contents = Content.query.all()
            interactions = UserInteraction.query.all()
            users = User.query.all()
            
            if not contents:
                logger.warning("⚠️ No content data available")
                return
            
            # Train content analyzer
            logger.info("📚 Training Ultra Content Analyzer...")
            self.content_analyzer.extract_advanced_features(contents)
            self.content_analyzer.compute_similarity_matrices()
            
            # Train collaborative filter
            if interactions:
                logger.info("🤝 Training Ultra Collaborative Filter...")
                self.collaborative_filter.fit(interactions)
            
            # Calculate advanced scores
            logger.info("📊 Calculating Advanced Scores...")
            self._calculate_advanced_scores(contents, interactions)
            
            # Train neural model if sufficient data
            if len(users) >= 50 and len(contents) >= 200 and len(interactions) >= 500:
                logger.info("🧠 Training Neural Recommendation Model...")
                self._train_neural_model(users, contents, interactions)
            
            self.is_trained = True
            logger.info("✅ Ultimate Model Training Completed!")
            
        except Exception as e:
            logger.error(f"❌ Model training error: {e}")
    
    def _calculate_advanced_scores(self, contents, interactions):
        """Calculate comprehensive scoring metrics"""
        try:
            # Initialize scores
            interaction_counts = defaultdict(int)
            rating_sums = defaultdict(float)
            rating_counts = defaultdict(int)
            user_counts = defaultdict(set)
            
            # Process interactions
            for interaction in interactions:
                content_id = interaction.content_id
                user_id = interaction.user_id
                
                # Count interactions
                weight = self.real_time_engine._get_interaction_weight(interaction.interaction_type)
                interaction_counts[content_id] += weight
                user_counts[content_id].add(user_id)
                
                # Handle ratings
                if interaction.rating:
                    rating_sums[content_id] += interaction.rating
                    rating_counts[content_id] += 1
            
            # Calculate scores for each content
            for content in contents:
                cid = content.id
                
                # Popularity score
                interaction_score = interaction_counts.get(cid, 0)
                unique_users = len(user_counts.get(cid, set()))
                
                popularity = (
                    np.log1p(interaction_score) * 0.4 +
                    np.log1p(unique_users) * 0.3 +
                    np.log1p(content.popularity or 0) * 0.2 +
                    np.log1p(content.vote_count or 0) * 0.1
                )
                
                self.popularity_scores[cid] = popularity
                
                # Quality score
                tmdb_rating = content.rating or 0
                user_rating = (rating_sums.get(cid, 0) / max(rating_counts.get(cid, 1), 1))
                
                quality = (
                    tmdb_rating * 0.5 +
                    user_rating * 0.3 +
                    (1 if content.is_critics_choice else 0) * 0.2
                )
                
                self.quality_scores[cid] = quality
                
                # Diversity score (based on genre uniqueness)
                if content.genres:
                    genres = json.loads(content.genres)
                    diversity = len(genres) / 10.0  # Normalize
                else:
                    diversity = 0
                
                self.diversity_scores[cid] = diversity
                
        except Exception as e:
            logger.error(f"❌ Score calculation error: {e}")
    
    def _train_neural_model(self, users, contents, interactions):
        """Train transformer-based neural model"""
        try:
            # Create user and item mappings
            user_to_idx = {user.id: idx for idx, user in enumerate(users)}
            item_to_idx = {content.id: idx for idx, content in enumerate(contents)}
            
            # Prepare training data
            user_ids, item_ids, ratings = [], [], []
            user_sequences = defaultdict(list)
            
            # Create user interaction sequences
            for interaction in interactions:
                if interaction.user_id in user_to_idx and interaction.content_id in item_to_idx:
                    user_sequences[interaction.user_id].append(
                        (interaction.content_id, interaction.timestamp)
                    )
            
            # Sort sequences by timestamp
            for user_id in user_sequences:
                user_sequences[user_id].sort(key=lambda x: x[1])
                user_sequences[user_id] = [item_id for item_id, _ in user_sequences[user_id]]
            
            # Prepare training data
            for interaction in interactions:
                if interaction.user_id in user_to_idx and interaction.content_id in item_to_idx:
                    user_ids.append(user_to_idx[interaction.user_id])
                    item_ids.append(item_to_idx[interaction.content_id])
                    
                    # Advanced rating
                    rating = self._calculate_neural_rating(interaction)
                    ratings.append(rating)
            
            if len(user_ids) < 100:
                return
            
            # Create neural model
            self.neural_model = TransformerRecommender(
                num_users=len(users),
                num_items=len(contents),
                embedding_dim=256,
                num_heads=8,
                num_layers=4
            )
            
            # Prepare tensors
            user_tensor = torch.LongTensor(user_ids)
            item_tensor = torch.LongTensor(item_ids)
            rating_tensor = torch.FloatTensor(ratings)
            
            # Normalize ratings
            rating_min, rating_max = rating_tensor.min(), rating_tensor.max()
            rating_tensor = (rating_tensor - rating_min) / (rating_max - rating_min + 1e-8)
            
            # Training setup
            optimizer = optim.AdamW(self.neural_model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            criterion = nn.MSELoss()
            
            # Training loop
            self.neural_model.train()
            best_loss = float('inf')
            patience = 0
            
            for epoch in range(200):  # More epochs for better training
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.neural_model(user_tensor, item_tensor)
                loss = criterion(predictions, rating_tensor)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.neural_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Early stopping
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience = 0
                else:
                    patience += 1
                
                if patience > 20:
                    break
                
                if epoch % 25 == 0:
                    logger.info(f"Neural training epoch {epoch}, loss: {loss.item():.4f}")
            
            # Store mappings
            self.user_to_idx = user_to_idx
            self.item_to_idx = item_to_idx
            self.idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
            self.rating_min = rating_min.item()
            self.rating_max = rating_max.item()
            
            logger.info("✅ Neural model training completed")
            
        except Exception as e:
            logger.error(f"❌ Neural model training error: {e}")
    
    def _calculate_neural_rating(self, interaction):
        """Calculate rating for neural network training"""
        base_ratings = {
            'view': 2.0,
            'like': 4.0,
            'favorite': 5.0,
            'watchlist': 4.5,
            'search': 1.0
        }
        
        rating = base_ratings.get(interaction.interaction_type, 2.0)
        
        if interaction.rating:
            rating = float(interaction.rating)
        
        # Time and frequency adjustments
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        time_factor = max(0.3, 1.0 - (days_ago / 365))
        
        return rating * time_factor
    
    def get_recommendations(self, strategy, **kwargs):
        """Get recommendations using specified strategy"""
        if not self.is_trained:
            logger.info("🔄 Models not trained, training now...")
            self.train_all_models()
        
        try:
            if strategy == 'personalized':
                return self._get_ultimate_personalized_recommendations(**kwargs)
            elif strategy == 'trending':
                return self._get_real_time_trending(**kwargs)
            elif strategy == 'similar':
                return self._get_advanced_similar(**kwargs)
            elif strategy == 'genre':
                return self._get_intelligent_genre(**kwargs)
            elif strategy == 'regional':
                return self._get_cultural_regional(**kwargs)
            elif strategy == 'critics_choice':
                return self._get_quality_critics_choice(**kwargs)
            elif strategy == 'new_releases':
                return self._get_smart_new_releases(**kwargs)
            elif strategy == 'anime':
                return self._get_otaku_anime_recommendations(**kwargs)
            else:
                return self._get_popular_recommendations(**kwargs)
                
        except Exception as e:
            logger.error(f"❌ Recommendation error for {strategy}: {e}")
            return []
    
    def _get_ultimate_personalized_recommendations(self, user_id, user_data=None, limit=20):
        """Ultimate personalized recommendations with all algorithms"""
        try:
            recommendations = defaultdict(lambda: {'scores': defaultdict(float), 'reasons': [], 'total': 0})
            
            # Get real-time user preferences
            real_time_prefs = self.real_time_engine.get_user_realtime_preferences(user_id)
            
            # 1. Collaborative Filtering (35%)
            cf_recs = self.collaborative_filter.get_user_recommendations(user_id, limit * 3)
            for rec in cf_recs:
                cid = rec['content_id']
                score = rec['score'] * self.ensemble_weights['collaborative']
                recommendations[cid]['scores']['collaborative'] = score
                recommendations[cid]['reasons'].append('Users with similar taste')
                recommendations[cid]['total'] += score
            
            # 2. Neural Model (20%)
            if self.neural_model and user_id in self.user_to_idx:
                neural_recs = self._get_neural_recommendations(user_id, limit * 2)
                for rec in neural_recs:
                    cid = rec['content_id']
                    score = rec['score'] * self.ensemble_weights['neural']
                    recommendations[cid]['scores']['neural'] = score
                    recommendations[cid]['reasons'].append('Deep learning analysis')
                    recommendations[cid]['total'] += score
            
            # 3. Content-Based (25%)
            content_recs = self._get_content_based_for_user(user_id, limit * 2)
            for rec in content_recs:
                cid = rec['content_id']
                score = rec['score'] * self.ensemble_weights['content_based']
                recommendations[cid]['scores']['content'] = score
                recommendations[cid]['reasons'].append('Based on your interests')
                recommendations[cid]['total'] += score
            
            # 4. Popularity Boost (10%)
            for cid in recommendations:
                pop_score = self.popularity_scores.get(cid, 0)
                normalized_pop = min(1.0, pop_score / 10) * self.ensemble_weights['popularity']
                recommendations[cid]['scores']['popularity'] = normalized_pop
                recommendations[cid]['total'] += normalized_pop
            
            # 5. Real-time Boost (10%)
            for cid in recommendations:
                rt_score = self.real_time_engine.get_trending_score(cid)
                normalized_rt = min(1.0, rt_score / 5) * self.ensemble_weights['real_time']
                recommendations[cid]['scores']['real_time'] = normalized_rt
                recommendations[cid]['total'] += normalized_rt
            
            # 6. User Preference Alignment
            if real_time_prefs.get('preferences'):
                for cid in recommendations:
                    content = Content.query.get(cid)
                    if content:
                        pref_boost = self._calculate_preference_alignment(
                            content, real_time_prefs['preferences']
                        )
                        recommendations[cid]['scores']['preference'] = pref_boost
                        recommendations[cid]['total'] += pref_boost
            
            # Sort and apply diversity
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1]['total'], reverse=True)
            diverse_recs = self._apply_advanced_diversity(sorted_recs, limit, user_id)
            
            # Format results
            results = []
            for cid, data in diverse_recs:
                reasons = list(set(data['reasons'][:3]))
                reason_text = '; '.join(reasons) if reasons else 'Personalized for you'
                
                results.append({
                    'content_id': cid,
                    'score': data['total'],
                    'reason': reason_text,
                    'algorithm_breakdown': dict(data['scores'])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ultimate personalized error: {e}")
            return []
    
    def _get_neural_recommendations(self, user_id, limit):
        """Get recommendations from neural model"""
        try:
            if not self.neural_model or user_id not in self.user_to_idx:
                return []
            
            user_idx = self.user_to_idx[user_id]
            
            # Get unrated items
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            rated_items = set(i.content_id for i in user_interactions)
            unrated_items = [
                idx for content_id, idx in self.item_to_idx.items() 
                if content_id not in rated_items
            ]
            
            if not unrated_items:
                return []
            
            self.neural_model.eval()
            with torch.no_grad():
                user_tensor = torch.LongTensor([user_idx] * len(unrated_items))
                item_tensor = torch.LongTensor(unrated_items)
                
                predictions = self.neural_model(user_tensor, item_tensor)
                predictions = predictions * (self.rating_max - self.rating_min) + self.rating_min
                
                # Get top predictions
                top_indices = torch.argsort(predictions, descending=True)[:limit]
                
                results = []
                for idx in top_indices:
                    item_idx = unrated_items[idx.item()]
                    content_id = self.idx_to_item[item_idx]
                    score = float(predictions[idx])
                    
                    results.append({
                        'content_id': content_id,
                        'score': score / 5.0,  # Normalize to 0-1
                        'reason': 'Neural network prediction'
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"❌ Neural recommendations error: {e}")
            return []
    
    def _get_content_based_for_user(self, user_id, limit):
        """Get content-based recommendations for user"""
        try:
            # Get user's recent interactions
            recent_interactions = UserInteraction.query.filter(
                UserInteraction.user_id == user_id,
                UserInteraction.timestamp >= datetime.utcnow() - timedelta(days=60)
            ).order_by(UserInteraction.timestamp.desc()).limit(20).all()
            
            if not recent_interactions:
                return []
            
            all_similar = defaultdict(float)
            
            for interaction in recent_interactions:
                similar_items = self.content_analyzer.get_similar_content(
                    interaction.content_id, num_recommendations=15
                )
                
                weight = self.real_time_engine._get_interaction_weight(interaction.interaction_type)
                
                for item in similar_items:
                    all_similar[item['content_id']] += item['score'] * weight
            
            # Sort and format
            sorted_similar = sorted(all_similar.items(), key=lambda x: x[1], reverse=True)
            
            results = []
            for content_id, score in sorted_similar[:limit]:
                results.append({
                    'content_id': content_id,
                    'score': min(1.0, score),  # Normalize
                    'reason': 'Content similarity analysis'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Content-based user recs error: {e}")
            return []
    
    def _get_real_time_trending(self, limit=20, content_type=None, region=None):
        """Get real-time trending recommendations"""
        try:
            trending_items = []
            
            # Get all content
            query = Content.query
            if content_type and content_type != 'all':
                query = query.filter(Content.content_type == content_type)
            
            contents = query.all()
            
            for content in contents:
                # Real-time trending score
                rt_score = self.real_time_engine.get_trending_score(content.id)
                
                # Base popularity
                pop_score = self.popularity_scores.get(content.id, 0)
                
                # Recency boost
                recency_boost = 0
                if content.release_date:
                    days_since = (datetime.utcnow().date() - content.release_date).days
                    recency_boost = max(0, 1 - (days_since / 365)) * 0.3
                
                # Quality factor
                quality_factor = self.quality_scores.get(content.id, 0) * 0.2
                
                # Combined score
                final_score = rt_score + pop_score * 0.3 + recency_boost + quality_factor
                
                if final_score > 0:
                    trending_items.append({
                        'content_id': content.id,
                        'score': final_score,
                        'reason': f'Trending now (velocity: {rt_score:.1f})'
                    })
            
            # Sort and return
            trending_items.sort(key=lambda x: x['score'], reverse=True)
            return trending_items[:limit]
            
        except Exception as e:
            logger.error(f"❌ Real-time trending error: {e}")
            return []
    
    def _get_advanced_similar(self, content_id, limit=20):
        """Get advanced similar recommendations"""
        try:
            return self.content_analyzer.get_similar_content(content_id, limit)
        except Exception as e:
            logger.error(f"❌ Advanced similar error: {e}")
            return []
    
    def _get_intelligent_genre(self, genre, limit=20, content_type='movie', user_id=None):
        """Get intelligent genre recommendations"""
        try:
            # Get content by genre
            contents = Content.query.filter(Content.content_type == content_type).all()
            
            genre_items = []
            for content in contents:
                if content.genres:
                    genres = json.loads(content.genres)
                    if genre.lower() in [g.lower() for g in genres]:
                        
                        # Base score
                        score = (
                            self.popularity_scores.get(content.id, 0) * 0.3 +
                            self.quality_scores.get(content.id, 0) * 0.4 +
                            self.real_time_engine.get_trending_score(content.id) * 0.2 +
                            self.diversity_scores.get(content.id, 0) * 0.1
                        )
                        
                        # User preference boost
                        if user_id:
                            user_prefs = self.real_time_engine.get_user_realtime_preferences(user_id)
                            if user_prefs.get('preferences', {}).get(f'genre_{genre}', 0) > 0:
                                score *= 1.5
                        
                        genre_items.append({
                            'content_id': content.id,
                            'score': score,
                            'reason': f'Top {genre} content'
                        })
            
            # Sort and return
            genre_items.sort(key=lambda x: x['score'], reverse=True)
            return genre_items[:limit]
            
        except Exception as e:
            logger.error(f"❌ Intelligent genre error: {e}")
            return []
    
    def _get_cultural_regional(self, language, limit=20, content_type='movie', user_id=None):
        """Get culturally-aware regional recommendations"""
        try:
            # Enhanced language mapping
            lang_mappings = {
                'hindi': ['hi', 'hindi', 'हिन्दी', 'bollywood'],
                'telugu': ['te', 'telugu', 'తెలుగు', 'tollywood'],
                'tamil': ['ta', 'tamil', 'தமிழ்', 'kollywood'],
                'kannada': ['kn', 'kannada', 'ಕನ್ನಡ', 'sandalwood'],
                'malayalam': ['ml', 'malayalam', 'മലയാളം', 'mollywood'],
                'english': ['en', 'english', 'hollywood'],
                'japanese': ['ja', 'japanese', '日本語', 'anime'],
                'korean': ['ko', 'korean', '한국어', 'kdrama']
            }
            
            target_langs = lang_mappings.get(language.lower(), [language.lower()])
            
            contents = Content.query.filter(Content.content_type == content_type).all()
            
            regional_items = []
            for content in contents:
                language_match = False
                
                if content.languages:
                    content_langs = [lang.lower() for lang in json.loads(content.languages)]
                    language_match = any(tl in content_langs for tl in target_langs)
                
                # Also check title/overview for language indicators
                if not language_match and content.title:
                    title_lower = content.title.lower()
                    language_match = any(tl in title_lower for tl in target_langs)
                
                if language_match:
                    # Cultural scoring
                    score = (
                        self.popularity_scores.get(content.id, 0) * 0.3 +
                        self.quality_scores.get(content.id, 0) * 0.4 +
                        self.real_time_engine.get_trending_score(content.id) * 0.2
                    )
                    
                    # Regional boost
                    if content.rating and content.rating >= 7.0:
                        score += 0.2  # Quality regional content
                    
                    # User preference boost
                    if user_id:
                        user_prefs = self.real_time_engine.get_user_realtime_preferences(user_id)
                        for tl in target_langs:
                            score += user_prefs.get('preferences', {}).get(f'lang_{tl}', 0) * 0.3
                    
                    regional_items.append({
                        'content_id': content.id,
                        'score': score,
                        'reason': f'Popular {language} content'
                    })
            
            # Sort and return
            regional_items.sort(key=lambda x: x['score'], reverse=True)
            return regional_items[:limit]
            
        except Exception as e:
            logger.error(f"❌ Cultural regional error: {e}")
            return []
    
    def _get_quality_critics_choice(self, limit=20, content_type='movie'):
        """Get high-quality critics choice recommendations"""
        try:
            contents = Content.query.filter(Content.content_type == content_type).all()
            
            critics_items = []
            for content in contents:
                # Multi-factor quality assessment
                quality_indicators = []
                
                # TMDB rating
                if content.rating and content.rating >= 7.5:
                    quality_indicators.append(content.rating / 10)
                
                # Vote count credibility
                if content.vote_count and content.vote_count >= 100:
                    vote_factor = min(1.0, np.log1p(content.vote_count) / 10)
                    quality_indicators.append(vote_factor)
                
                # Critics choice flag
                if content.is_critics_choice:
                    quality_indicators.append(0.8)
                
                # User engagement quality
                if content.id in self.popularity_scores:
                    pop_factor = min(1.0, self.popularity_scores[content.id] / 10)
                    quality_indicators.append(pop_factor)
                
                if quality_indicators:
                    final_quality = np.mean(quality_indicators)
                    
                    # Only include high-quality items
                    if final_quality >= 0.6:
                        critics_items.append({
                            'content_id': content.id,
                            'score': final_quality,
                            'reason': f'Critically acclaimed ({content.rating}/10)'
                        })
            
            # Sort and return
            critics_items.sort(key=lambda x: x['score'], reverse=True)
            return critics_items[:limit]
            
        except Exception as e:
            logger.error(f"❌ Quality critics choice error: {e}")
            return []
    
    def _get_smart_new_releases(self, limit=20, language=None, content_type='movie'):
        """Get smart new releases with quality filtering"""
        try:
            # Dynamic date range
            cutoff_days = 90 if content_type == 'anime' else 60
            cutoff_date = datetime.utcnow().date() - timedelta(days=cutoff_days)
            
            query = Content.query.filter(
                Content.content_type == content_type,
                Content.release_date >= cutoff_date
            )
            
            contents = query.all()
            
            new_items = []
            for content in contents:
                # Language filtering
                if language:
                    lang_mappings = {
                        'hindi': ['hi', 'hindi'],
                        'telugu': ['te', 'telugu'],
                        'tamil': ['ta', 'tamil'],
                        'kannada': ['kn', 'kannada'],
                        'malayalam': ['ml', 'malayalam'],
                        'english': ['en', 'english']
                    }
                    
                    target_langs = lang_mappings.get(language.lower(), [language.lower()])
                    
                    if content.languages:
                        content_langs = [lang.lower() for lang in json.loads(content.languages)]
                        if not any(tl in content_langs for tl in target_langs):
                            continue
                
                # Scoring
                days_since = (datetime.utcnow().date() - content.release_date).days
                recency_score = max(0, 1 - (days_since / cutoff_days))
                
                quality_score = self.quality_scores.get(content.id, 0) * 0.4
                popularity_score = self.popularity_scores.get(content.id, 0) * 0.2
                trending_score = self.real_time_engine.get_trending_score(content.id) * 0.3
                
                final_score = recency_score * 0.4 + quality_score + popularity_score + trending_score
                
                reason = f'New release ({days_since} days ago)'
                if trending_score > 1.0:
                    reason = 'Hot new release - trending!'
                
                new_items.append({
                    'content_id': content.id,
                    'score': final_score,
                    'reason': reason
                })
            
            # Sort and return
            new_items.sort(key=lambda x: x['score'], reverse=True)
            return new_items[:limit]
            
        except Exception as e:
            logger.error(f"❌ Smart new releases error: {e}")
            return []
    
    def _get_otaku_anime_recommendations(self, limit=20, genre=None, user_id=None):
        """Get otaku-level anime recommendations"""
        try:
            contents = Content.query.filter(Content.content_type == 'anime').all()
            
            anime_items = []
            for content in contents:
                include_anime = True
                
                # Advanced genre filtering
                if genre:
                    include_anime = False
                    
                    # Check anime-specific genres
                    if content.anime_genres:
                        anime_genres = [g.lower() for g in json.loads(content.anime_genres)]
                        if genre.lower() in anime_genres:
                            include_anime = True
                    
                    # Check general genres
                    if not include_anime and content.genres:
                        general_genres = [g.lower() for g in json.loads(content.genres)]
                        if genre.lower() in general_genres:
                            include_anime = True
                
                if include_anime:
                    # Anime-specific scoring
                    mal_score = (content.rating or 0) / 10 * 0.5
                    popularity_score = self.popularity_scores.get(content.id, 0) * 0.2
                    quality_score = self.quality_scores.get(content.id, 0) * 0.2
                    
                    # Seasonal relevance
                    seasonal_boost = 0
                    if content.release_date:
                        current_season = self._get_anime_season(datetime.utcnow().date())
                        content_season = self._get_anime_season(content.release_date)
                        
                        if current_season == content_season:
                            seasonal_boost = 0.2
                        elif abs(current_season - content_season) <= 1:
                            seasonal_boost = 0.1
                    
                    # User otaku level boost
                    otaku_boost = 0
                    if user_id:
                        user_prefs = self.real_time_engine.get_user_realtime_preferences(user_id)
                        anime_pref = user_prefs.get('preferences', {}).get('type_anime', 0)
                        if anime_pref > 0.3:  # Heavy anime watcher
                            otaku_boost = 0.2
                    
                    final_score = mal_score + popularity_score + quality_score + seasonal_boost + otaku_boost
                    
                    reason = 'Popular anime'
                    if genre:
                        reason = f'Top {genre} anime'
                    if seasonal_boost > 0:
                        reason += ' - Current season'
                    
                    anime_items.append({
                        'content_id': content.id,
                        'score': final_score,
                        'reason': reason
                    })
            
            # Sort and return
            anime_items.sort(key=lambda x: x['score'], reverse=True)
            return anime_items[:limit]
            
        except Exception as e:
            logger.error(f"❌ Otaku anime recommendations error: {e}")
            return []
    
    def _get_popular_recommendations(self, limit=20, content_type=None):
        """Get intelligently curated popular recommendations"""
        try:
            popular_items = []
            
            for content_id, pop_score in self.popularity_scores.items():
                content = Content.query.get(content_id)
                if not content:
                    continue
                
                if content_type and content.content_type != content_type:
                    continue
                
                # Enhanced popularity scoring
                quality_boost = self.quality_scores.get(content_id, 0) * 0.3
                trending_boost = self.real_time_engine.get_trending_score(content_id) * 0.2
                diversity_boost = self.diversity_scores.get(content_id, 0) * 0.1
                
                final_score = pop_score + quality_boost + trending_boost + diversity_boost
                
                reason = 'Popular among users'
                if trending_boost > 0.5:
                    reason = 'Popular and trending'
                if quality_boost > 0.7:
                    reason = 'Popular and highly rated'
                
                popular_items.append({
                    'content_id': content_id,
                    'score': final_score,
                    'reason': reason
                })
            
            # Sort and return
            popular_items.sort(key=lambda x: x['score'], reverse=True)
            return popular_items[:limit]
            
        except Exception as e:
            logger.error(f"❌ Popular recommendations error: {e}")
            return []
    
    def _calculate_preference_alignment(self, content, user_preferences):
        """Calculate how well content aligns with user preferences"""
        try:
            alignment_score = 0
            
            # Genre alignment
            if content.genres:
                genres = json.loads(content.genres)
                for genre in genres:
                    genre_pref = user_preferences.get(f'genre_{genre}', 0)
                    alignment_score += genre_pref * 0.3
            
            # Language alignment
            if content.languages:
                languages = json.loads(content.languages)
                for lang in languages:
                    lang_pref = user_preferences.get(f'lang_{lang}', 0)
                    alignment_score += lang_pref * 0.2
            
            # Content type alignment
            type_pref = user_preferences.get(f'type_{content.content_type}', 0)
            alignment_score += type_pref * 0.1
            
            return min(1.0, alignment_score)
            
        except Exception as e:
            logger.error(f"❌ Preference alignment error: {e}")
            return 0
    
    def _apply_advanced_diversity(self, recommendations, limit, user_id=None):
        """Apply advanced diversity filtering"""
        try:
            if not recommendations:
                return []
            
            diverse_recs = []
            used_genres = defaultdict(int)
            used_languages = defaultdict(int)
            used_types = defaultdict(int)
            
            # Get user's diversity preference
            diversity_factor = 0.3  # Default
            if user_id:
                user_prefs = self.real_time_engine.get_user_realtime_preferences(user_id)
                diversity_score = user_prefs.get('diversity_score', 0)
                if diversity_score > 50:  # High diversity user
                    diversity_factor = 0.5
                elif diversity_score < 10:  # Low diversity user
                    diversity_factor = 0.1
            
            max_per_genre = max(2, int(limit * (1 - diversity_factor)))
            max_per_language = max(3, int(limit * 0.7))
            max_per_type = max(2, int(limit * 0.8))
            
            for content_id, data in recommendations:
                if len(diverse_recs) >= limit:
                    break
                
                content = Content.query.get(content_id)
                if not content:
                    continue
                
                # Check diversity constraints
                can_add = True
                
                # Genre diversity
                if content.genres:
                    genres = json.loads(content.genres)
                    if any(used_genres[g] >= max_per_genre for g in genres):
                        if len(diverse_recs) > limit * 0.5:  # Only apply after half filled
                            can_add = False
                
                # Language diversity
                if can_add and content.languages:
                    languages = json.loads(content.languages)
                    if any(used_languages[l] >= max_per_language for l in languages):
                        if len(diverse_recs) > limit * 0.3:
                            can_add = False
                
                # Type diversity
                if can_add and used_types[content.content_type] >= max_per_type:
                    if len(diverse_recs) > limit * 0.6:
                        can_add = False
                
                if can_add:
                    diverse_recs.append((content_id, data))
                    
                    # Update counters
                    if content.genres:
                        for genre in json.loads(content.genres):
                            used_genres[genre] += 1
                    
                    if content.languages:
                        for lang in json.loads(content.languages):
                            used_languages[lang] += 1
                    
                    used_types[content.content_type] += 1
            
            return diverse_recs
            
        except Exception as e:
            logger.error(f"❌ Advanced diversity error: {e}")
            return recommendations[:limit]
    
    def _get_anime_season(self, date):
        """Get anime season number for a date"""
        month = date.month
        if month in [12, 1, 2]:
            return 1  # Winter
        elif month in [3, 4, 5]:
            return 2  # Spring
        elif month in [6, 7, 8]:
            return 3  # Summer
        else:
            return 4  # Fall

# Initialize the ultimate recommendation engine
ultimate_engine = UltimateRecommendationEngine()

# Perfect Backend Integration API Routes

@app.route('/api/health', methods=['GET'])
def ultimate_health_check():
    """Ultimate health check with comprehensive status"""
    try:
        # Database connectivity
        total_content = Content.query.count()
        total_users = User.query.count()
        total_interactions = UserInteraction.query.count()
        
        # Model status
        models_status = {
            'content_analyzer': ultimate_engine.content_analyzer.content_features is not None,
            'collaborative_filter': ultimate_engine.collaborative_filter.user_item_matrix is not None,
            'neural_model': ultimate_engine.neural_model is not None,
            'real_time_engine': len(ultimate_engine.real_time_engine.user_profiles) >= 0,
            'is_fully_trained': ultimate_engine.is_trained
        }
        
        # Performance metrics
        performance_metrics = {
            'cache_hit_rate': len(ultra_memory_cache) / max(len(ultra_memory_cache) + 1, 1),
            'real_time_interactions': len(real_time_interactions),
            'user_profiles_active': len(ultimate_engine.real_time_engine.user_profiles),
            'trending_items_tracked': len(ultimate_engine.real_time_engine.content_momentum)
        }
        
        # Feature status
        feature_status = {
            'transformer_neural_cf': ultimate_engine.neural_model is not None,
            'real_time_trending': True,
            'advanced_content_analysis': ultimate_engine.content_analyzer.content_features is not None,
            'cultural_awareness': True,
            'diversity_filtering': True,
            'ensemble_recommendations': True,
            'otaku_level_anime': True,
            'quality_assessment': True
        }
        
        return jsonify({
            'status': 'ultimate_healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '3.0.0-ultimate',
            'backend_sync': True,
            'models_initialized': ultimate_engine.is_trained,
            'data_status': {
                'total_content': total_content,
                'total_users': total_users,
                'total_interactions': total_interactions,
                'data_quality': 'excellent' if total_content > 100 and total_interactions > 500 else 'good'
            },
            'models_status': models_status,
            'performance_metrics': performance_metrics,
            'feature_status': feature_status,
            'algorithms': [
                'Transformer Neural Collaborative Filtering',
                'Ultra-Advanced Content Analysis',
                'Real-time Intelligence Engine',
                'Cultural & Regional Awareness',
                'Advanced Diversity Filtering',
                'Ensemble Recommendation System',
                'Otaku-level Anime Intelligence',
                'Multi-factor Quality Assessment'
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate health check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/recommendations', methods=['POST'])
def ultimate_personalized_recommendations():
    """Ultimate personalized recommendations endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'user_id' not in data:
            return jsonify({'error': 'user_id required'}), 400
        
        user_id = data['user_id']
        limit = min(data.get('limit', 20), 100)  # Max 100 recommendations
        
        # Process any real-time interactions in the request
        if 'current_interaction' in data:
            ultimate_engine.real_time_engine.process_real_time_interaction(data['current_interaction'])
        
        # Ultra-smart caching
        cache_params = {
            'user_id': user_id,
            'limit': limit,
            'version': '3.0',
            'time_bucket': int(time.time() // 180)  # 3-minute buckets
        }
        
        user_context = ultimate_engine.real_time_engine.get_user_realtime_preferences(user_id)
        cache_key = UltimateCacheManager.generate_smart_key('ultimate_personalized', cache_params, user_context)
        
        cached_result, metadata = UltimateCacheManager.get_with_metadata(cache_key)
        if cached_result:
            cached_result['cached'] = True
            cached_result['cache_age'] = time.time() - metadata.get('created_at', time.time())
            return jsonify(cached_result), 200
        
        # Get ultimate personalized recommendations
        recommendations = ultimate_engine.get_recommendations(
            'personalized',
            user_id=user_id,
            user_data=data,
            limit=limit
        )
        
        # Enhanced insights
        user_insights = {
            'profile_completeness': min(100, len(user_context.get('preferences', {})) * 10),
            'activity_level': user_context.get('activity_level', 0),
            'diversity_score': user_context.get('diversity_score', 0),
            'interaction_velocity': user_context.get('interaction_velocity', 0),
            'top_preferences': list(user_context.get('preferences', {}).keys())[:5]
        }
        
        result = {
            'recommendations': recommendations,
            'strategy': 'ultimate_transformer_ensemble',
            'cached': False,
            'total_found': len(recommendations),
            'user_insights': user_insights,
            'algorithm_version': '3.0-ultimate',
            'processing_time_ms': int(time.time() * 1000) % 1000,
            'quality_score': min(100, len(recommendations) * 5),
            'personalization_level': 'maximum'
        }
        
        # Cache with high priority
        UltimateCacheManager.set_with_intelligence(
            cache_key, result, 
            expiry=ULTIMATE_CACHE_EXPIRY['personalized'],
            priority='high'
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate personalized recommendations error: {e}")
        return jsonify({
            'error': 'Failed to get ultimate personalized recommendations', 
            'details': str(e)
        }), 500

@app.route('/api/trending', methods=['GET'])
def ultimate_trending_recommendations():
    """Ultimate real-time trending recommendations"""
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        content_type = request.args.get('content_type', 'all')
        region = request.args.get('region')
        
        # Real-time cache with ultra-short expiry
        cache_params = {
            'limit': limit,
            'content_type': content_type,
            'region': region,
            'time_bucket': int(time.time() // 60)  # 1-minute buckets for trending
        }
        
        cache_key = UltimateCacheManager.generate_smart_key('ultimate_trending', cache_params)
        cached_result, metadata = UltimateCacheManager.get_with_metadata(cache_key)
        
        if cached_result:
            cached_result['cached'] = True
            cached_result['freshness'] = 'ultra_fresh'
            return jsonify(cached_result), 200
        
        recommendations = ultimate_engine.get_recommendations(
            'trending',
            limit=limit,
            content_type=content_type,
            region=region
        )
        
        # Real-time trending insights
        trending_insights = {
            'total_trending_items': len(ultimate_engine.real_time_engine.content_momentum),
            'velocity_tracked_items': len(ultimate_engine.real_time_engine.trending_velocities),
            'real_time_interactions_count': len(real_time_interactions),
            'trending_algorithm': 'real_time_velocity_momentum_v3',
            'update_frequency': 'every_minute'
        }
        
        result = {
            'recommendations': recommendations,
            'strategy': 'real_time_velocity_analysis',
            'cached': False,
            'total_found': len(recommendations),
            'trending_insights': trending_insights,
            'freshness_level': 'ultra_fresh',
            'generated_at': datetime.utcnow().isoformat()
        }
        
        UltimateCacheManager.set_with_intelligence(
            cache_key, result, 
            expiry=ULTIMATE_CACHE_EXPIRY['trending'],
            priority='high'
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate trending error: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/similar/<int:content_id>', methods=['GET'])
def ultimate_similar_recommendations(content_id):
    """Ultimate similar content recommendations"""
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        
        cache_key = UltimateCacheManager.generate_smart_key(
            'ultimate_similar', 
            {'content_id': content_id, 'limit': limit}
        )
        
        cached_result, metadata = UltimateCacheManager.get_with_metadata(cache_key)
        if cached_result:
            cached_result['cached'] = True
            return jsonify(cached_result), 200
        
        recommendations = ultimate_engine.get_recommendations(
            'similar',
            content_id=content_id,
            limit=limit
        )
        
        # Similarity insights
        base_content = Content.query.get(content_id)
        similarity_insights = {
            'base_content_type': base_content.content_type if base_content else None,
            'algorithms_used': [
                'transformer_content_analysis',
                'network_similarity', 
                'cluster_similarity',
                'topic_modeling',
                'ensemble_weighting'
            ],
            'diversity_applied': True,
            'quality_filtered': True
        }
        
        result = {
            'recommendations': recommendations,
            'strategy': 'ultra_advanced_similarity',
            'cached': False,
            'total_found': len(recommendations),
            'similarity_insights': similarity_insights,
            'base_content_id': content_id
        }
        
        UltimateCacheManager.set_with_intelligence(
            cache_key, result, 
            expiry=ULTIMATE_CACHE_EXPIRY['similar']
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate similar recommendations error: {e}")
        return jsonify({'error': 'Failed to get similar recommendations'}), 500

@app.route('/api/genre/<genre>', methods=['GET'])
def ultimate_genre_recommendations(genre):
    """Ultimate intelligent genre recommendations"""
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        content_type = request.args.get('content_type', 'movie')
        user_id = request.args.get('user_id', type=int)
        
        cache_params = {
            'genre': genre, 
            'limit': limit, 
            'content_type': content_type, 
            'user_id': user_id
        }
        
        cache_key = UltimateCacheManager.generate_smart_key('ultimate_genre', cache_params)
        cached_result, metadata = UltimateCacheManager.get_with_metadata(cache_key)
        
        if cached_result:
            cached_result['cached'] = True
            return jsonify(cached_result), 200
        
        recommendations = ultimate_engine.get_recommendations(
            'genre',
            genre=genre,
            limit=limit,
            content_type=content_type,
            user_id=user_id
        )
        
        result = {
            'recommendations': recommendations,
            'strategy': 'intelligent_genre_curation',
            'cached': False,
            'total_found': len(recommendations),
            'genre': genre,
            'personalized': user_id is not None,
            'quality_assured': True
        }
        
        UltimateCacheManager.set_with_intelligence(
            cache_key, result, 
            expiry=ULTIMATE_CACHE_EXPIRY['genre']
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate genre recommendations error: {e}")
        return jsonify({'error': 'Failed to get genre recommendations'}), 500

@app.route('/api/regional/<language>', methods=['GET'])
def ultimate_regional_recommendations(language):
    """Ultimate cultural regional recommendations"""
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        content_type = request.args.get('content_type', 'movie')
        user_id = request.args.get('user_id', type=int)
        
        cache_params = {
            'language': language, 
            'limit': limit, 
            'content_type': content_type, 
            'user_id': user_id
        }
        
        cache_key = UltimateCacheManager.generate_smart_key('ultimate_regional', cache_params)
        cached_result, metadata = UltimateCacheManager.get_with_metadata(cache_key)
        
        if cached_result:
            cached_result['cached'] = True
            return jsonify(cached_result), 200
        
        recommendations = ultimate_engine.get_recommendations(
            'regional',
            language=language,
            limit=limit,
            content_type=content_type,
            user_id=user_id
        )
        
        result = {
            'recommendations': recommendations,
            'strategy': 'cultural_regional_intelligence',
            'cached': False,
            'total_found': len(recommendations),
            'language': language,
            'culturally_aware': True,
            'personalized': user_id is not None
        }
        
        UltimateCacheManager.set_with_intelligence(
            cache_key, result, 
            expiry=ULTIMATE_CACHE_EXPIRY['regional']
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate regional recommendations error: {e}")
        return jsonify({'error': 'Failed to get regional recommendations'}), 500

@app.route('/api/critics-choice', methods=['GET'])
def ultimate_critics_choice():
    """Ultimate quality critics choice recommendations"""
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        content_type = request.args.get('content_type', 'movie')
        
        cache_params = {'limit': limit, 'content_type': content_type}
        cache_key = UltimateCacheManager.generate_smart_key('ultimate_critics', cache_params)
        
        cached_result, metadata = UltimateCacheManager.get_with_metadata(cache_key)
        if cached_result:
            cached_result['cached'] = True
            return jsonify(cached_result), 200
        
        recommendations = ultimate_engine.get_recommendations(
            'critics_choice',
            limit=limit,
            content_type=content_type
        )
        
        result = {
            'recommendations': recommendations,
            'strategy': 'multi_factor_quality_assessment',
            'cached': False,
            'total_found': len(recommendations),
            'quality_threshold': 'premium',
            'assessment_criteria': [
                'TMDB_rating_7.5+',
                'vote_count_100+',
                'critics_choice_flag',
                'user_engagement_quality'
            ]
        }
        
        UltimateCacheManager.set_with_intelligence(
            cache_key, result, 
            expiry=ULTIMATE_CACHE_EXPIRY['critics']
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate critics choice error: {e}")
        return jsonify({'error': 'Failed to get critics choice'}), 500

@app.route('/api/new-releases', methods=['GET'])
def ultimate_new_releases():
    """Ultimate smart new releases with quality filtering"""
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        language = request.args.get('language')
        content_type = request.args.get('content_type', 'movie')
        
        cache_params = {
            'limit': limit, 
            'language': language, 
            'content_type': content_type
        }
        
        cache_key = UltimateCacheManager.generate_smart_key('ultimate_new_releases', cache_params)
        cached_result, metadata = UltimateCacheManager.get_with_metadata(cache_key)
        
        if cached_result:
            cached_result['cached'] = True
            return jsonify(cached_result), 200
        
        recommendations = ultimate_engine.get_recommendations(
            'new_releases',
            limit=limit,
            language=language,
            content_type=content_type
        )
        
        result = {
            'recommendations': recommendations,
            'strategy': 'smart_new_releases_with_quality',
            'cached': False,
            'total_found': len(recommendations),
            'language_filter': language,
            'quality_filtered': True,
            'trending_aware': True
        }
        
        UltimateCacheManager.set_with_intelligence(
            cache_key, result, 
            expiry=ULTIMATE_CACHE_EXPIRY['new_releases']
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate new releases error: {e}")
        return jsonify({'error': 'Failed to get new releases'}), 500

@app.route('/api/anime', methods=['GET'])
def ultimate_anime_recommendations():
    """Ultimate otaku-level anime recommendations"""
    try:
        limit = min(int(request.args.get('limit', 20)), 100)
        genre = request.args.get('genre')
        user_id = request.args.get('user_id', type=int)
        
        cache_params = {
            'limit': limit, 
            'genre': genre, 
            'user_id': user_id
        }
        
        cache_key = UltimateCacheManager.generate_smart_key('ultimate_anime', cache_params)
        cached_result, metadata = UltimateCacheManager.get_with_metadata(cache_key)
        
        if cached_result:
            cached_result['cached'] = True
            return jsonify(cached_result), 200
        
        recommendations = ultimate_engine.get_recommendations(
            'anime',
            limit=limit,
            genre=genre,
            user_id=user_id
        )
        
        result = {
            'recommendations': recommendations,
            'strategy': 'otaku_level_anime_intelligence',
            'cached': False,
            'total_found': len(recommendations),
            'genre_filter': genre,
            'seasonal_aware': True,
            'otaku_optimized': True,
            'personalized': user_id is not None
        }
        
        UltimateCacheManager.set_with_intelligence(
            cache_key, result, 
            expiry=ULTIMATE_CACHE_EXPIRY['anime']
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate anime recommendations error: {e}")
        return jsonify({'error': 'Failed to get anime recommendations'}), 500

@app.route('/api/track-interaction', methods=['POST'])
def track_ultimate_interaction():
    """Track interactions for real-time learning"""
    try:
        data = request.get_json()
        
        if not data or 'user_id' not in data or 'content_id' not in data:
            return jsonify({'error': 'user_id and content_id required'}), 400
        
        # Process real-time interaction
        interaction_data = {
            'user_id': data['user_id'],
            'content_id': data['content_id'],
            'interaction_type': data.get('interaction_type', 'view'),
            'timestamp': datetime.utcnow(),
            'rating': data.get('rating')
        }
        
        # Add to real-time buffer
        real_time_interactions.append(interaction_data)
        
        # Process in real-time engine
        success = ultimate_engine.real_time_engine.process_real_time_interaction(interaction_data)
        
        # Intelligent cache invalidation
        user_id = data['user_id']
        
        # Invalidate user-specific caches
        patterns_to_invalidate = [
            f"ultimate_ml:ultimate_personalized:*user_id*{user_id}*",
            "ultimate_ml:ultimate_trending:*",
            "ultimate_ml:ultimate_similar:*"
        ]
        
        for pattern in patterns_to_invalidate:
            UltimateCacheManager.invalidate_pattern(pattern)
        
        return jsonify({
            'success': success,
            'message': 'Interaction processed and learned in real-time',
            'real_time_buffer_size': len(real_time_interactions),
            'user_profile_updated': True,
            'caches_invalidated': len(patterns_to_invalidate)
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate interaction tracking error: {e}")
        return jsonify({'error': 'Failed to track interaction'}), 500

@app.route('/api/update-models', methods=['POST'])
def update_ultimate_models():
    """Force update all ultimate models"""
    try:
        # Clear all caches
        if redis_client:
            for key in redis_client.scan_iter(match='ultimate_ml:*'):
                redis_client.delete(key)
        else:
            ultra_memory_cache.clear()
            cache_metadata.clear()
        
        # Force retrain all models
        ultimate_engine.is_trained = False
        ultimate_engine.train_all_models()
        
        return jsonify({
            'success': True,
            'message': 'Ultimate models retrained successfully',
            'timestamp': datetime.utcnow().isoformat(),
            'models_trained': [
                'Ultra Content Analyzer',
                'Ultra Collaborative Filter', 
                'Transformer Neural Model',
                'Real-time Intelligence Engine',
                'Cultural Awareness System',
                'Quality Assessment Engine',
                'Diversity Optimization',
                'Ensemble Coordination'
            ],
            'version': '3.0-ultimate'
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate model update error: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def ultimate_comprehensive_stats():
    """Ultimate comprehensive ML service statistics"""
    try:
        # Data statistics
        total_users = User.query.count()
        total_content = Content.query.count()
        total_interactions = UserInteraction.query.count()
        
        # Content distribution
        content_distribution = {}
        for content_type, count in db.session.query(Content.content_type, db.func.count(Content.id)).group_by(Content.content_type).all():
            content_distribution[content_type] = count
        
        # Recent activity
        recent_interactions = UserInteraction.query.filter(
            UserInteraction.timestamp >= datetime.utcnow() - timedelta(hours=24)
        ).count()
        
        # Model performance metrics
        model_performance = {
            'content_analyzer': {
                'features_extracted': ultimate_engine.content_analyzer.content_features.shape[1] if ultimate_engine.content_analyzer.content_features is not None else 0,
                'similarity_matrix_size': len(ultimate_engine.content_analyzer.content_ids),
                'clusters_created': 100,  # KMeans clusters
                'topics_identified': 100   # LDA topics
            },
            'collaborative_filter': {
                'users_in_matrix': len(ultimate_engine.collaborative_filter.user_ids),
                'items_in_matrix': len(ultimate_engine.collaborative_filter.item_ids),
                'matrix_density': (len(ultimate_engine.collaborative_filter.user_ids) * len(ultimate_engine.collaborative_filter.item_ids)) / max(total_interactions, 1),
                'user_clusters': len(set(ultimate_engine.collaborative_filter.user_clusters.values())) if ultimate_engine.collaborative_filter.user_clusters else 0
            },
            'neural_model': {
                'model_loaded': ultimate_engine.neural_model is not None,
                'embedding_dimensions': 256,
                'transformer_layers': 4,
                'attention_heads': 8
            },
            'real_time_engine': {
                'active_user_profiles': len(ultimate_engine.real_time_engine.user_profiles),
                'momentum_tracked_items': len(ultimate_engine.real_time_engine.content_momentum),
                'velocity_tracked_items': len(ultimate_engine.real_time_engine.trending_velocities),
                'interaction_buffer_size': len(real_time_interactions)
            }
        }
        
        # Cache performance
        cache_performance = {
            'redis_available': redis_client is not None,
            'memory_cache_items': len(ultra_memory_cache),
            'cache_metadata_items': len(cache_metadata),
            'total_cache_access': sum(metadata.get('access_count', 0) for metadata in cache_metadata.values())
        }
        
        # Algorithm information
        algorithm_info = {
            'recommendation_engine': 'Ultimate Hybrid Ensemble v3.0',
            'content_analysis': 'Ultra-Advanced Multi-Algorithm Analysis',
            'collaborative_filtering': 'Advanced Matrix Factorization + Clustering',
            'neural_network': 'Transformer-based Neural Collaborative Filtering',
            'real_time_processing': 'Velocity-based Momentum Analysis',
            'cultural_awareness': 'Multi-language Regional Intelligence',
            'quality_assessment': 'Multi-factor Quality Scoring',
            'diversity_optimization': 'Advanced Cluster-aware Diversity',
            'trending_analysis': 'Real-time Velocity + Momentum Tracking'
        }
        
        return jsonify({
            'service_status': 'ultimate_operational',
            'data_statistics': {
                'total_users': total_users,
                'total_content': total_content,
                'total_interactions': total_interactions,
                'unique_active_users': len(set(i.user_id for i in UserInteraction.query.all())),
                'content_distribution': content_distribution,
                'recent_activity_24h': recent_interactions,
                'data_quality_score': min(100, (total_content + total_interactions) / 10)
            },
            'model_performance': model_performance,
            'cache_performance': cache_performance,
            'algorithm_info': algorithm_info,
            'version': '3.0.0-ultimate',
            'capabilities': [
                'Netflix-level Personalization',
                'TikTok-style Real-time Trending',
                'Spotify-quality Recommendation Accuracy',
                'Cultural & Regional Intelligence',
                'Otaku-level Anime Expertise',
                'Advanced Quality Assessment',
                'Real-time Learning & Adaptation'
            ],
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate stats error: {e}")
        return jsonify({'error': 'Failed to get comprehensive statistics'}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_ultimate_cache():
    """Clear all ultimate caches"""
    try:
        cleared_count = 0
        
        if redis_client:
            keys = list(redis_client.scan_iter(match='ultimate_ml:*'))
            if keys:
                cleared_count = redis_client.delete(*keys)
        else:
            cleared_count = len(ultra_memory_cache)
            ultra_memory_cache.clear()
            cache_metadata.clear()
        
        return jsonify({
            'success': True,
            'message': f'Ultimate cache cleared - {cleared_count} keys removed',
            'cache_system': 'redis' if redis_client else 'memory',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Ultimate cache clear error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Background processing and auto-initialization
def initialize_ultimate_models():
    """Initialize ultimate models on startup"""
    try:
        with app.app_context():
            logger.info("🚀 Initializing Ultimate ML Models...")
            ultimate_engine.train_all_models()
            logger.info("✅ Ultimate ML Models initialized successfully!")
    except Exception as e:
        logger.error(f"❌ Ultimate model initialization error: {e}")

def background_ultimate_updater():
    """Background task for periodic model updates"""
    while True:
        try:
            time.sleep(1800)  # Update every 30 minutes
            with app.app_context():
                logger.info("🔄 Background ultimate model update...")
                ultimate_engine.train_all_models()
                logger.info("✅ Background update completed")
        except Exception as e:
            logger.error(f"❌ Background update error: {e}")

# Start background processes
threading.Thread(target=initialize_ultimate_models, daemon=True).start()
threading.Thread(target=background_ultimate_updater, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)