# ml-service/app.py
import os
import json
import pickle
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Core ML & Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pandas as pd

# Advanced Recommendation Systems
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
import surprise
from surprise import SVD, NMF, KNNBasic, BaselineOnly
from surprise.model_selection import train_test_split
from surprise import Dataset as SurpriseDataset
from surprise import Reader

# Advanced Embeddings & NLP
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF as SKNMF
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models import Word2Vec, FastText, Doc2Vec

# High-Performance Computing
import faiss
import numba
from numba import jit
import networkx as nx

# Optimization & AutoML
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Database & Storage
import redis
import sqlite3
from sqlalchemy import create_engine, text
import scipy.sparse as sp

# API & Web Framework
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import aiohttp

# Monitoring & Utilities
from prometheus_client import Counter, Histogram, generate_latest
import joblib
import concurrent.futures
from functools import lru_cache, wraps
import threading
import time
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask App Setup
app = Flask(__name__)
CORS(app)

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///movie_rec.db')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 128
SEQUENCE_LENGTH = 50

# Redis Cache
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connected successfully")
except:
    REDIS_AVAILABLE = False
    redis_client = None
    logger.warning("Redis not available, caching disabled")

# Global Models Cache
MODEL_CACHE = {}
FAISS_INDEX = None
CONTENT_EMBEDDINGS = None

class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering Model"""
    def __init__(self, num_users, num_items, embedding_dim=128, hidden_dims=[256, 128, 64]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        
        return torch.sigmoid(self.mlp(x))

class ContentBasedModel:
    """Content-based recommendation model"""
    def __init__(self):
        self.sentence_transformer = None
        self.tfidf_vectorizer = None
        self.content_matrix = None
        self.faiss_index = None
        
    def initialize(self):
        """Initialize the models"""
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            logger.info("Content-based model initialized")
        except Exception as e:
            logger.error(f"Error initializing content model: {e}")
    
    def build_content_features(self, content_df):
        """Build content feature matrix"""
        try:
            # Combine text features
            content_df['combined_features'] = (
                content_df['title'].fillna('') + ' ' +
                content_df['overview'].fillna('') + ' ' +
                content_df['genres'].fillna('').astype(str)
            )
            
            # Get sentence embeddings
            embeddings = self.sentence_transformer.encode(
                content_df['combined_features'].tolist(),
                batch_size=32,
                show_progress_bar=False
            )
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings.astype('float32'))
            
            self.content_matrix = embeddings
            logger.info(f"Content features built: {embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error building content features: {e}")

class AdvancedRecommendationEngine:
    """Advanced Recommendation Engine with Multiple Algorithms"""
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.content_model = ContentBasedModel()
        self.users_df = pd.DataFrame()
        self.content_df = pd.DataFrame()
        self.interactions_df = pd.DataFrame()
        self.user_item_matrix = None
        
    def _get_database_connection(self):
        """Get database connection"""
        if DATABASE_URL.startswith('sqlite'):
            db_path = DATABASE_URL.replace('sqlite:///', '')
            return sqlite3.connect(db_path)
        else:
            from sqlalchemy import create_engine
            engine = create_engine(DATABASE_URL)
            return engine.connect()
    
    def load_data(self):
        """Load data from database"""
        try:
            conn = self._get_database_connection()
            
            # Load users
            self.users_df = pd.read_sql("SELECT * FROM user", conn)
            logger.info(f"Loaded {len(self.users_df)} users")
            
            # Load content
            self.content_df = pd.read_sql("SELECT * FROM content", conn)
            logger.info(f"Loaded {len(self.content_df)} content items")
            
            # Load interactions
            self.interactions_df = pd.read_sql("""
                SELECT user_id, content_id, interaction_type, rating, created_at
                FROM user_interaction
            """, conn)
            logger.info(f"Loaded {len(self.interactions_df)} interactions")
            
            if hasattr(conn, 'close'):
                conn.close()
            
            return len(self.interactions_df) > 0
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def prepare_interaction_matrix(self):
        """Prepare user-item interaction matrix"""
        try:
            if self.interactions_df.empty:
                return False
            
            # Create interaction weights
            interaction_weights = {
                'view': 1.0,
                'like': 2.0,
                'favorite': 3.0,
                'wishlist': 2.5
            }
            
            # Add weights to interactions
            self.interactions_df['weight'] = self.interactions_df['interaction_type'].map(
                lambda x: interaction_weights.get(x, 1.0)
            )
            
            # Add rating boost
            mask = self.interactions_df['rating'].notna()
            self.interactions_df.loc[mask, 'weight'] *= (
                self.interactions_df.loc[mask, 'rating'] / 5.0
            )
            
            # Create pivot table
            self.user_item_matrix = self.interactions_df.pivot_table(
                index='user_id',
                columns='content_id',
                values='weight',
                fill_value=0,
                aggfunc='sum'
            )
            
            logger.info(f"Interaction matrix shape: {self.user_item_matrix.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing interaction matrix: {e}")
            return False
    
    def train_collaborative_filtering(self):
        """Train collaborative filtering models"""
        try:
            if self.user_item_matrix is None or self.user_item_matrix.empty:
                return False
            
            # Convert to sparse matrix
            sparse_matrix = sp.csr_matrix(self.user_item_matrix.values)
            
            # Train ALS model
            als_model = AlternatingLeastSquares(
                factors=EMBEDDING_DIM,
                regularization=0.1,
                iterations=50,
                alpha=40
            )
            als_model.fit(sparse_matrix.T)  # Transpose for implicit library
            self.models['als'] = als_model
            
            # Train BPR model
            bpr_model = BayesianPersonalizedRanking(
                factors=EMBEDDING_DIM,
                learning_rate=0.01,
                regularization=0.01,
                iterations=100
            )
            bpr_model.fit(sparse_matrix.T)
            self.models['bpr'] = bpr_model
            
            # Train Surprise models
            reader = Reader(rating_scale=(0, 5))
            
            # Prepare data for Surprise
            surprise_data = []
            for _, row in self.interactions_df.iterrows():
                surprise_data.append((
                    row['user_id'],
                    row['content_id'],
                    row['weight']
                ))
            
            surprise_dataset = SurpriseDataset.load_from_df(
                pd.DataFrame(surprise_data, columns=['user_id', 'content_id', 'rating']),
                reader
            )
            
            # Train SVD
            svd_model = SVD(n_factors=EMBEDDING_DIM, n_epochs=20, lr_all=0.005, reg_all=0.02)
            trainset = surprise_dataset.build_full_trainset()
            svd_model.fit(trainset)
            self.models['svd'] = svd_model
            
            logger.info("Collaborative filtering models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering: {e}")
            return False
    
    def train_neural_model(self):
        """Train neural collaborative filtering model"""
        try:
            if self.interactions_df.empty:
                return False
            
            # Prepare encoders
            user_encoder = LabelEncoder()
            item_encoder = LabelEncoder()
            
            all_users = self.users_df['id'].unique()
            all_items = self.content_df['id'].unique()
            
            user_encoder.fit(all_users)
            item_encoder.fit(all_items)
            
            self.encoders['user'] = user_encoder
            self.encoders['item'] = item_encoder
            
            # Encode interactions
            interactions_encoded = self.interactions_df.copy()
            interactions_encoded['user_encoded'] = user_encoder.transform(
                interactions_encoded['user_id']
            )
            interactions_encoded['item_encoded'] = item_encoder.transform(
                interactions_encoded['content_id']
            )
            
            # Create neural model
            ncf_model = NeuralCollaborativeFiltering(
                num_users=len(all_users),
                num_items=len(all_items),
                embedding_dim=EMBEDDING_DIM
            ).to(DEVICE)
            
            # Prepare training data
            user_tensor = torch.tensor(
                interactions_encoded['user_encoded'].values,
                dtype=torch.long
            ).to(DEVICE)
            item_tensor = torch.tensor(
                interactions_encoded['item_encoded'].values,
                dtype=torch.long
            ).to(DEVICE)
            label_tensor = torch.ones(len(interactions_encoded), dtype=torch.float).to(DEVICE)
            
            # Train model
            optimizer = optim.Adam(ncf_model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            dataset = TensorDataset(user_tensor, item_tensor, label_tensor)
            dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
            
            ncf_model.train()
            for epoch in range(10):  # Reduced epochs for faster training
                total_loss = 0
                for batch_users, batch_items, batch_labels in dataloader:
                    optimizer.zero_grad()
                    outputs = ncf_model(batch_users, batch_items).squeeze()
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 2 == 0:
                    logger.info(f"NCF Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
            
            self.models['ncf'] = ncf_model
            logger.info("Neural model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training neural model: {e}")
            return False
    
    def train_models(self):
        """Train all recommendation models"""
        logger.info("Starting model training...")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load data")
            return False
        
        # Prepare interaction matrix
        if not self.prepare_interaction_matrix():
            logger.error("Failed to prepare interaction matrix")
            return False
        
        # Initialize content model
        self.content_model.initialize()
        
        # Build content features
        if not self.content_df.empty:
            self.content_model.build_content_features(self.content_df)
        
        # Train collaborative filtering
        self.train_collaborative_filtering()
        
        # Train neural model
        self.train_neural_model()
        
        logger.info("Model training completed")
        return True
    
    def get_collaborative_recommendations(self, user_id, k=10):
        """Get collaborative filtering recommendations"""
        recommendations = []
        
        try:
            # ALS recommendations
            if 'als' in self.models and user_id in self.user_item_matrix.index:
                user_idx = list(self.user_item_matrix.index).index(user_id)
                user_vector = sp.csr_matrix(self.user_item_matrix.iloc[user_idx].values)
                
                als_recs = self.models['als'].recommend(
                    user_idx, 
                    user_vector,
                    N=k,
                    filter_already_liked_items=True
                )
                
                for item_idx, score in als_recs:
                    content_id = self.user_item_matrix.columns[item_idx]
                    recommendations.append({
                        'content_id': int(content_id),
                        'score': float(score),
                        'method': 'als'
                    })
            
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {e}")
        
        return recommendations[:k]
    
    def get_content_based_recommendations(self, user_id, k=10):
        """Get content-based recommendations"""
        recommendations = []
        
        try:
            if self.content_model.faiss_index is None:
                return recommendations
            
            # Get user's liked content
            user_interactions = self.interactions_df[
                (self.interactions_df['user_id'] == user_id) &
                (self.interactions_df['interaction_type'].isin(['like', 'favorite']))
            ]
            
            if user_interactions.empty:
                return recommendations
            
            liked_content_ids = user_interactions['content_id'].unique()
            
            # Get embeddings for liked content
            liked_embeddings = []
            for content_id in liked_content_ids:
                content_row = self.content_df[self.content_df['id'] == content_id]
                if not content_row.empty:
                    idx = content_row.index[0]
                    if idx < len(self.content_model.content_matrix):
                        liked_embeddings.append(self.content_model.content_matrix[idx])
            
            if not liked_embeddings:
                return recommendations
            
            # Average embeddings
            avg_embedding = np.mean(liked_embeddings, axis=0).reshape(1, -1)
            faiss.normalize_L2(avg_embedding)
            
            # Search similar content
            scores, indices = self.content_model.faiss_index.search(
                avg_embedding.astype('float32'), k * 2
            )
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.content_df):
                    content_id = self.content_df.iloc[idx]['id']
                    if content_id not in liked_content_ids:
                        recommendations.append({
                            'content_id': int(content_id),
                            'score': float(score),
                            'method': 'content_based'
                        })
        
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
        
        return recommendations[:k]
    
    def get_neural_recommendations(self, user_id, k=10):
        """Get neural model recommendations"""
        recommendations = []
        
        try:
            if 'ncf' not in self.models or 'user' not in self.encoders:
                return recommendations
            
            ncf_model = self.models['ncf']
            user_encoder = self.encoders['user']
            item_encoder = self.encoders['item']
            
            # Check if user exists
            if user_id not in user_encoder.classes_:
                return recommendations
            
            user_encoded = user_encoder.transform([user_id])[0]
            
            # Get all items
            all_items = self.content_df['id'].values
            
            # Filter out already interacted items
            user_interactions = self.interactions_df[
                self.interactions_df['user_id'] == user_id
            ]['content_id'].values
            
            candidate_items = [item for item in all_items if item not in user_interactions]
            
            if not candidate_items:
                return recommendations
            
            # Filter items that exist in encoder
            valid_items = [item for item in candidate_items if item in item_encoder.classes_]
            
            if not valid_items:
                return recommendations
            
            # Encode items
            items_encoded = item_encoder.transform(valid_items)
            
            # Predict scores
            ncf_model.eval()
            with torch.no_grad():
                user_tensor = torch.tensor([user_encoded] * len(items_encoded), 
                                         dtype=torch.long).to(DEVICE)
                item_tensor = torch.tensor(items_encoded, dtype=torch.long).to(DEVICE)
                
                scores = ncf_model(user_tensor, item_tensor).squeeze().cpu().numpy()
            
            # Sort and get top recommendations
            item_scores = list(zip(valid_items, scores))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            for content_id, score in item_scores[:k]:
                recommendations.append({
                    'content_id': int(content_id),
                    'score': float(score),
                    'method': 'neural'
                })
        
        except Exception as e:
            logger.error(f"Error in neural recommendations: {e}")
        
        return recommendations[:k]
    
    def get_hybrid_recommendations(self, user_id, k=20):
        """Get hybrid recommendations"""
        all_recommendations = []
        
        # Get recommendations from different methods
        methods = [
            self.get_collaborative_recommendations,
            self.get_content_based_recommendations,
            self.get_neural_recommendations
        ]
        
        for method in methods:
            try:
                recs = method(user_id, k)
                all_recommendations.extend(recs)
            except Exception as e:
                logger.error(f"Error in method {method.__name__}: {e}")
        
        # Combine and rank recommendations
        content_scores = defaultdict(list)
        
        for rec in all_recommendations:
            content_scores[rec['content_id']].append(rec['score'])
        
        # Calculate final scores
        final_recommendations = []
        
        for content_id, scores in content_scores.items():
            # Use average score with diversity boost
            avg_score = np.mean(scores)
            diversity_boost = len(scores) / len(methods)
            final_score = avg_score * (1 + 0.1 * diversity_boost)
            
            final_recommendations.append({
                'content_id': content_id,
                'score': final_score,
                'method_count': len(scores)
            })
        
        # Sort by final score
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return final_recommendations[:k]

# Global recommendation engine
recommendation_engine = AdvancedRecommendationEngine()

# Caching decorator
def cache_result(expire_time=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not REDIS_AVAILABLE:
                return func(*args, **kwargs)
            
            # Create cache key
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = f"ml_cache:{hashlib.md5(key_data.encode()).hexdigest()}"
            
            try:
                # Try to get from cache
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
            except:
                pass
            
            # Execute function
            result = func(*args, **kwargs)
            
            try:
                # Cache result
                redis_client.setex(
                    cache_key, 
                    expire_time, 
                    json.dumps(result, default=str)
                )
            except:
                pass
            
            return result
        return wrapper
    return decorator

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'models_loaded': len(recommendation_engine.models),
        'gpu_available': torch.cuda.is_available(),
        'redis_available': REDIS_AVAILABLE
    })

@app.route('/recommend', methods=['POST'])
@cache_result(expire_time=1800)
def get_recommendations():
    """Main recommendation endpoint"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        limit = min(data.get('limit', 20), 50)
        method = data.get('method', 'hybrid')
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Get recommendations based on method
        if method == 'hybrid':
            recommendations = recommendation_engine.get_hybrid_recommendations(user_id, limit)
        elif method == 'collaborative':
            recommendations = recommendation_engine.get_collaborative_recommendations(user_id, limit)
        elif method == 'content_based':
            recommendations = recommendation_engine.get_content_based_recommendations(user_id, limit)
        elif method == 'neural':
            recommendations = recommendation_engine.get_neural_recommendations(user_id, limit)
        else:
            return jsonify({'error': 'Invalid method'}), 400
        
        return jsonify({
            'user_id': user_id,
            'method': method,
            'recommendations': recommendations,
            'total': len(recommendations),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in recommendations: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/similar/<int:content_id>', methods=['GET'])
@cache_result(expire_time=7200)
def get_similar_content(content_id):
    """Get similar content recommendations"""
    try:
        limit = min(int(request.args.get('limit', 10)), 20)
        
        if recommendation_engine.content_model.faiss_index is None:
            return jsonify({'similar_content': [], 'total': 0})
        
        # Find content
        content_row = recommendation_engine.content_df[
            recommendation_engine.content_df['id'] == content_id
        ]
        
        if content_row.empty:
            return jsonify({'similar_content': [], 'total': 0})
        
        content_idx = content_row.index[0]
        
        if content_idx >= len(recommendation_engine.content_model.content_matrix):
            return jsonify({'similar_content': [], 'total': 0})
        
        # Get content embedding
        content_embedding = recommendation_engine.content_model.content_matrix[content_idx].reshape(1, -1)
        
        # Search similar content
        scores, indices = recommendation_engine.content_model.faiss_index.search(
            content_embedding.astype('float32'), limit + 1
        )
        
        recommendations = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != content_idx and idx < len(recommendation_engine.content_df):
                similar_content_id = recommendation_engine.content_df.iloc[idx]['id']
                recommendations.append({
                    'content_id': int(similar_content_id),
                    'score': float(score)
                })
        
        return jsonify({
            'content_id': content_id,
            'similar_content': recommendations[:limit],
            'total': len(recommendations),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting similar content: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/train', methods=['POST'])
def train_models():
    """Trigger model training"""
    try:
        # Run training in background
        def train_task():
            recommendation_engine.train_models()
        
        thread = threading.Thread(target=train_task)
        thread.start()
        
        return jsonify({
            'status': 'training_started',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({'error': 'Failed to start training'}), 500

@app.route('/model/status', methods=['GET'])
def get_model_status():
    """Get model training status"""
    return jsonify({
        'models_loaded': list(recommendation_engine.models.keys()),
        'total_models': len(recommendation_engine.models),
        'content_model_ready': recommendation_engine.content_model.faiss_index is not None,
        'interaction_matrix_ready': recommendation_engine.user_item_matrix is not None,
        'device': str(DEVICE),
        'data_loaded': {
            'users': len(recommendation_engine.users_df),
            'content': len(recommendation_engine.content_df),
            'interactions': len(recommendation_engine.interactions_df)
        },
        'timestamp': datetime.utcnow().isoformat()
    })

# Initialize models on startup
def initialize_models():
    """Initialize models when the app starts"""
    try:
        logger.info("Initializing models on startup...")
        success = recommendation_engine.train_models()
        if success:
            logger.info("Models initialized successfully")
        else:
            logger.warning("Model initialization completed with some issues")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")

if __name__ == '__main__':
    # Train models on startup
    initialize_models()
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5001)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    )