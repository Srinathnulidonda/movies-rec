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
import pytorch_lightning as pl
import numpy as np
import pandas as pd

# Advanced Recommendation Systems
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from lightfm import LightFM
from lightfm.data import Dataset as LFMDataset
import surprise
from surprise import SVD, NMF, KNNBasic, BaselineOnly
from surprise.model_selection import train_test_split
import recbole
from recbole.quick_start import run_recbole

# Advanced Embeddings & NLP
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF as SKNMF
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models import Word2Vec, FastText, Doc2Vec

# Graph Neural Networks
import torch_geometric
from torch_geometric.data import Data, DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
import networkx as nx

# High-Performance Computing
import faiss
import numba
from numba import jit, cuda
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Optimization & AutoML
import optuna
from optuna.integration import LightGBMPruningCallback
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Time Series & Sequential
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import statsmodels.api as sm

# Database & Storage
import redis
import sqlite3
from sqlalchemy import create_engine, text
import psycopg2

# API & Web Framework
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import requests
import aiohttp

# Monitoring & MLOps
import mlflow
import mlflow.pytorch
from prometheus_client import Counter, Histogram, generate_latest
import joblib

# Utilities
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
except:
    REDIS_AVAILABLE = False
    redis_client = None

# Global Models Cache
MODEL_CACHE = {}
FAISS_INDEX = None
CONTENT_EMBEDDINGS = None

class ContentEmbeddingModel(nn.Module):
    """Deep learning model for content embeddings"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        return self.classifier(pooled)

class UserContentInteractionModel(nn.Module):
    """Neural collaborative filtering model"""
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

class SequentialRecommendationModel(nn.Module):
    """Transformer-based sequential recommendation"""
    def __init__(self, num_items, embedding_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(embedding_dim, num_items)
        
    def forward(self, sequences, positions):
        item_emb = self.item_embedding(sequences)
        pos_emb = self.position_embedding(positions)
        
        x = item_emb + pos_emb
        x = x.transpose(0, 1)  # (seq_len, batch, embedding_dim)
        
        transformer_out = self.transformer(x)
        transformer_out = transformer_out.transpose(0, 1)  # Back to (batch, seq_len, embedding_dim)
        
        # Use last item representation
        last_item_repr = transformer_out[:, -1, :]
        
        return self.output_layer(last_item_repr)

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for user-item interactions"""
    def __init__(self, num_users, num_items, embedding_dim=128, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Node embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # GNN layers
        self.convs = nn.ModuleList([
            GCNConv(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, edge_index, user_ids=None, item_ids=None):
        # Create node features
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Combine user and item embeddings
        x = torch.cat([user_emb, item_emb], dim=0)
        
        # Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x

class HybridRecommendationEngine:
    """Advanced hybrid recommendation engine"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.sentence_transformer = None
        self.content_vectorizer = None
        self.user_item_matrix = None
        self.content_features = None
        self.faiss_index = None
        self.graph_data = None
        
        # Load pre-trained models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all recommendation models"""
        try:
            # Sentence transformer for content embeddings
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # TF-IDF vectorizer
            self.content_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _get_database_connection(self):
        """Get database connection"""
        if DATABASE_URL.startswith('sqlite'):
            return sqlite3.connect(DATABASE_URL.replace('sqlite:///', ''))
        else:
            from sqlalchemy import create_engine
            engine = create_engine(DATABASE_URL)
            return engine.connect()
    
    def _load_data(self):
        """Load data from database"""
        try:
            conn = self._get_database_connection()
            
            # Load users
            users_df = pd.read_sql("SELECT * FROM user", conn)
            
            # Load content
            content_df = pd.read_sql("SELECT * FROM content", conn)
            
            # Load interactions
            interactions_df = pd.read_sql("""
                SELECT user_id, content_id, interaction_type, rating, created_at
                FROM user_interaction
            """, conn)
            
            conn.close()
            
            return users_df, content_df, interactions_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def _prepare_interaction_matrix(self, interactions_df):
        """Prepare user-item interaction matrix"""
        # Create implicit feedback matrix
        interaction_weights = {
            'view': 1.0,
            'like': 2.0,
            'favorite': 3.0,
            'wishlist': 2.5
        }
        
        # Add weights to interactions
        interactions_df['weight'] = interactions_df['interaction_type'].map(
            lambda x: interaction_weights.get(x, 1.0)
        )
        
        # Add rating boost
        interactions_df.loc[interactions_df['rating'].notna(), 'weight'] *= (
            interactions_df.loc[interactions_df['rating'].notna(), 'rating'] / 5.0
        )
        
        # Create pivot table
        user_item_matrix = interactions_df.pivot_table(
            index='user_id',
            columns='content_id',
            values='weight',
            fill_value=0,
            aggfunc='sum'
        )
        
        return user_item_matrix
    
    def _build_content_features(self, content_df):
        """Build content feature matrix"""
        # Combine text features
        content_df['combined_features'] = (
            content_df['title'].fillna('') + ' ' +
            content_df['overview'].fillna('') + ' ' +
            content_df['genres'].fillna('').astype(str)
        )
        
        # TF-IDF features
        tfidf_features = self.content_vectorizer.fit_transform(
            content_df['combined_features']
        )
        
        # Sentence transformer embeddings
        sentence_embeddings = self.sentence_transformer.encode(
            content_df['combined_features'].tolist(),
            batch_size=32,
            show_progress_bar=True
        )
        
        # Combine features
        from scipy.sparse import hstack, csr_matrix
        combined_features = hstack([
            tfidf_features,
            csr_matrix(sentence_embeddings)
        ])
        
        return combined_features, sentence_embeddings
    
    def _train_collaborative_filtering(self, user_item_matrix):
        """Train collaborative filtering models"""
        # Convert to sparse matrix
        import scipy.sparse as sp
        sparse_matrix = sp.csr_matrix(user_item_matrix.values)
        
        # ALS Model
        als_model = AlternatingLeastSquares(
            factors=EMBEDDING_DIM,
            regularization=0.1,
            iterations=50,
            alpha=40
        )
        als_model.fit(sparse_matrix.T)  # Transpose for implicit library
        
        # BPR Model
        bpr_model = BayesianPersonalizedRanking(
            factors=EMBEDDING_DIM,
            learning_rate=0.01,
            regularization=0.01,
            iterations=100
        )
        bpr_model.fit(sparse_matrix.T)
        
        # LightFM Model
        lightfm_dataset = LFMDataset()
        
        # Prepare data for LightFM
        user_ids = user_item_matrix.index.tolist()
        item_ids = user_item_matrix.columns.tolist()
        
        lightfm_dataset.fit(
            users=user_ids,
            items=item_ids
        )
        
        # Build interactions
        (interactions, weights) = lightfm_dataset.build_interactions(
            [(u, i, r) for u, row in user_item_matrix.iterrows() 
             for i, r in row.items() if r > 0]
        )
        
        lightfm_model = LightFM(
            loss='warp',
            no_components=EMBEDDING_DIM,
            learning_rate=0.05,
            max_sampled=10
        )
        lightfm_model.fit(interactions, sample_weight=weights, epochs=50)
        
        return {
            'als': als_model,
            'bpr': bpr_model,
            'lightfm': (lightfm_model, lightfm_dataset)
        }
    
    def _train_deep_learning_models(self, users_df, content_df, interactions_df):
        """Train deep learning models"""
        # Prepare data
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        all_users = users_df['id'].unique()
        all_items = content_df['id'].unique()
        
        user_encoder.fit(all_users)
        item_encoder.fit(all_items)
        
        # Encode interactions
        interactions_encoded = interactions_df.copy()
        interactions_encoded['user_encoded'] = user_encoder.transform(
            interactions_encoded['user_id']
        )
        interactions_encoded['item_encoded'] = item_encoder.transform(
            interactions_encoded['content_id']
        )
        
        # Create labels (implicit feedback)
        interactions_encoded['label'] = 1.0
        
        # Neural Collaborative Filtering
        ncf_model = UserContentInteractionModel(
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
        label_tensor = torch.tensor(
            interactions_encoded['label'].values,
            dtype=torch.float
        ).to(DEVICE)
        
        # Train NCF model
        optimizer = optim.Adam(ncf_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        dataset = TensorDataset(user_tensor, item_tensor, label_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        
        ncf_model.train()
        for epoch in range(20):
            total_loss = 0
            for batch_users, batch_items, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = ncf_model(batch_users, batch_items).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                logger.info(f"NCF Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        return {
            'ncf': ncf_model,
            'user_encoder': user_encoder,
            'item_encoder': item_encoder
        }
    
    def _build_faiss_index(self, content_embeddings):
        """Build FAISS index for fast similarity search"""
        dimension = content_embeddings.shape[1]
        
        # Use GPU index if available
        if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
            # GPU index
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, dimension)
        else:
            # CPU index with product quantization for memory efficiency
            index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings
        faiss.normalize_L2(content_embeddings)
        
        # Add to index
        index.add(content_embeddings.astype('float32'))
        
        return index
    
    def _prepare_graph_data(self, interactions_df, users_df, content_df):
        """Prepare graph data for GNN"""
        # Create user-item bipartite graph
        user_offset = 0
        item_offset = len(users_df)
        
        # Map user and item IDs
        user_mapping = {uid: i for i, uid in enumerate(users_df['id'])}
        item_mapping = {iid: i + item_offset for i, iid in enumerate(content_df['id'])}
        
        # Create edge list
        edges = []
        for _, row in interactions_df.iterrows():
            user_node = user_mapping.get(row['user_id'])
            item_node = item_mapping.get(row['content_id'])
            
            if user_node is not None and item_node is not None:
                edges.append([user_node, item_node])
                edges.append([item_node, user_node])  # Undirected
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Create GNN model
        gnn_model = GraphNeuralNetwork(
            num_users=len(users_df),
            num_items=len(content_df),
            embedding_dim=EMBEDDING_DIM
        ).to(DEVICE)
        
        return {
            'edge_index': edge_index.to(DEVICE),
            'model': gnn_model,
            'user_mapping': user_mapping,
            'item_mapping': item_mapping
        }
    
    def train_models(self):
        """Train all recommendation models"""
        logger.info("Starting model training...")
        
        # Load data
        users_df, content_df, interactions_df = self._load_data()
        
        if interactions_df.empty:
            logger.warning("No interaction data found")
            return
        
        # Prepare interaction matrix
        self.user_item_matrix = self._prepare_interaction_matrix(interactions_df)
        
        # Build content features
        self.content_features, content_embeddings = self._build_content_features(content_df)
        
        # Build FAISS index
        self.faiss_index = self._build_faiss_index(content_embeddings)
        
        # Train collaborative filtering models
        cf_models = self._train_collaborative_filtering(self.user_item_matrix)
        self.models.update(cf_models)
        
        # Train deep learning models
        dl_models = self._train_deep_learning_models(users_df, content_df, interactions_df)
        self.models.update(dl_models)
        
        # Prepare graph data
        self.graph_data = self._prepare_graph_data(interactions_df, users_df, content_df)
        
        # Store data
        self.users_df = users_df
        self.content_df = content_df
        self.interactions_df = interactions_df
        
        logger.info("Model training completed successfully")
    
    def get_collaborative_recommendations(self, user_id, k=10):
        """Get collaborative filtering recommendations"""
        recommendations = []
        
        try:
            # ALS recommendations
            if 'als' in self.models and user_id in self.user_item_matrix.index:
                user_idx = list(self.user_item_matrix.index).index(user_id)
                als_recs = self.models['als'].recommend(
                    user_idx, 
                    self.user_item_matrix.iloc[user_idx].values,
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
            # Get user's interaction history
            user_interactions = self.interactions_df[
                self.interactions_df['user_id'] == user_id
            ]
            
            if user_interactions.empty:
                return recommendations
            
            # Get liked content
            liked_content = user_interactions[
                user_interactions['interaction_type'].isin(['like', 'favorite'])
            ]['content_id'].unique()
            
            if len(liked_content) == 0:
                return recommendations
            
            # Get content indices
            liked_indices = []
            for content_id in liked_content:
                content_row = self.content_df[self.content_df['id'] == content_id]
                if not content_row.empty:
                    liked_indices.append(content_row.index[0])
            
            if not liked_indices:
                return recommendations
            
            # Use FAISS for similarity search
            if self.faiss_index is not None:
                # Get average embedding of liked content
                liked_embeddings = []
                for idx in liked_indices:
                    if idx < self.content_features.shape[0]:
                        embedding = self.sentence_transformer.encode([
                            self.content_df.iloc[idx]['combined_features']
                        ])[0]
                        liked_embeddings.append(embedding)
                
                if liked_embeddings:
                    avg_embedding = np.mean(liked_embeddings, axis=0).reshape(1, -1)
                    faiss.normalize_L2(avg_embedding)
                    
                    # Search similar content
                    scores, indices = self.faiss_index.search(
                        avg_embedding.astype('float32'), k * 2
                    )
                    
                    for score, idx in zip(scores[0], indices[0]):
                        if idx < len(self.content_df):
                            content_id = self.content_df.iloc[idx]['id']
                            if content_id not in liked_content:
                                recommendations.append({
                                    'content_id': int(content_id),
                                    'score': float(score),
                                    'method': 'content_based'
                                })
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
        
        return recommendations[:k]
    
    def get_deep_learning_recommendations(self, user_id, k=10):
        """Get deep learning recommendations"""
        recommendations = []
        
        try:
            if 'ncf' not in self.models:
                return recommendations
            
            ncf_model = self.models['ncf']
            user_encoder = self.models['user_encoder']
            item_encoder = self.models['item_encoder']
            
            # Check if user exists
            try:
                user_encoded = user_encoder.transform([user_id])[0]
            except:
                return recommendations
            
            # Get all items
            all_items = self.content_df['id'].values
            
            # Filter out already interacted items
            user_interactions = self.interactions_df[
                self.interactions_df['user_id'] == user_id
            ]['content_id'].values
            
            candidate_items = [item for item in all_items if item not in user_interactions]
            
            if not candidate_items:
                return recommendations
            
            # Encode candidate items
            try:
                items_encoded = item_encoder.transform(candidate_items)
            except:
                return recommendations
            
            # Predict scores
            ncf_model.eval()
            with torch.no_grad():
                user_tensor = torch.tensor([user_encoded] * len(items_encoded), 
                                         dtype=torch.long).to(DEVICE)
                item_tensor = torch.tensor(items_encoded, dtype=torch.long).to(DEVICE)
                
                scores = ncf_model(user_tensor, item_tensor).squeeze().cpu().numpy()
            
            # Sort and get top recommendations
            item_scores = list(zip(candidate_items, scores))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            for content_id, score in item_scores[:k]:
                recommendations.append({
                    'content_id': int(content_id),
                    'score': float(score),
                    'method': 'deep_learning'
                })
            
        except Exception as e:
            logger.error(f"Error in deep learning recommendations: {e}")
        
        return recommendations[:k]
    
    def get_sequential_recommendations(self, user_id, k=10):
        """Get sequential recommendations based on user history"""
        recommendations = []
        
        try:
            # Get user's interaction sequence
            user_interactions = self.interactions_df[
                self.interactions_df['user_id'] == user_id
            ].sort_values('created_at')
            
            if len(user_interactions) < 2:
                return recommendations
            
            # Create sequence
            sequence = user_interactions['content_id'].tolist()[-SEQUENCE_LENGTH:]
            
            # Pad sequence if needed
            if len(sequence) < SEQUENCE_LENGTH:
                sequence = [0] * (SEQUENCE_LENGTH - len(sequence)) + sequence
            
            # This is a simplified sequential recommendation
            # In a full implementation, you would train a sequential model
            
            # For now, use content similarity for next items
            last_items = sequence[-3:]  # Look at last 3 items
            
            for item_id in last_items:
                similar_recs = self.get_content_based_recommendations_for_item(
                    item_id, k=k//3
                )
                recommendations.extend(similar_recs)
            
        except Exception as e:
            logger.error(f"Error in sequential recommendations: {e}")
        
        return recommendations[:k]
    
    def get_content_based_recommendations_for_item(self, item_id, k=5):
        """Get recommendations similar to a specific item"""
        recommendations = []
        
        try:
            # Find item in content_df
            item_row = self.content_df[self.content_df['id'] == item_id]
            
            if item_row.empty:
                return recommendations
            
            item_idx = item_row.index[0]
            
            if self.faiss_index is not None and item_idx < len(self.content_df):
                # Get item embedding
                item_embedding = self.sentence_transformer.encode([
                    item_row.iloc[0]['combined_features']
                ]).reshape(1, -1)
                faiss.normalize_L2(item_embedding)
                
                # Search similar items
                scores, indices = self.faiss_index.search(
                    item_embedding.astype('float32'), k + 1
                )
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.content_df) and idx != item_idx:
                        content_id = self.content_df.iloc[idx]['id']
                        recommendations.append({
                            'content_id': int(content_id),
                            'score': float(score),
                            'method': 'item_similarity'
                        })
        
        except Exception as e:
            logger.error(f"Error in item-based recommendations: {e}")
        
        return recommendations[:k]
    
    def get_graph_based_recommendations(self, user_id, k=10):
        """Get graph neural network recommendations"""
        recommendations = []
        
        try:
            if not self.graph_data or 'model' not in self.graph_data:
                return recommendations
            
            gnn_model = self.graph_data['model']
            user_mapping = self.graph_data['user_mapping']
            item_mapping = self.graph_data['item_mapping']
            edge_index = self.graph_data['edge_index']
            
            if user_id not in user_mapping:
                return recommendations
            
            user_node = user_mapping[user_id]
            
            # Get node embeddings
            gnn_model.eval()
            with torch.no_grad():
                node_embeddings = gnn_model(edge_index)
                user_emb = node_embeddings[user_node].unsqueeze(0)
                
                # Get item embeddings
                item_start_idx = len(user_mapping)
                item_embeddings = node_embeddings[item_start_idx:]
                
                # Compute similarities
                similarities = torch.cosine_similarity(user_emb, item_embeddings)
                
                # Get top-k items
                top_scores, top_indices = torch.topk(similarities, k)
                
                # Convert back to content IDs
                reverse_item_mapping = {v - item_start_idx: k 
                                      for k, v in item_mapping.items()}
                
                for score, idx in zip(top_scores.cpu().numpy(), 
                                    top_indices.cpu().numpy()):
                    if idx in reverse_item_mapping:
                        content_id = reverse_item_mapping[idx]
                        recommendations.append({
                            'content_id': int(content_id),
                            'score': float(score),
                            'method': 'graph_neural_network'
                        })
        
        except Exception as e:
            logger.error(f"Error in graph-based recommendations: {e}")
        
        return recommendations[:k]
    
    def get_hybrid_recommendations(self, user_id, k=20):
        """Get hybrid recommendations combining all methods"""
        all_recommendations = []
        
        # Get recommendations from different methods
        methods = [
            ('collaborative', self.get_collaborative_recommendations),
            ('content_based', self.get_content_based_recommendations),
            ('deep_learning', self.get_deep_learning_recommendations),
            ('sequential', self.get_sequential_recommendations),
            ('graph_based', self.get_graph_based_recommendations)
        ]
        
        for method_name, method_func in methods:
            try:
                recs = method_func(user_id, k)
                all_recommendations.extend(recs)
            except Exception as e:
                logger.error(f"Error in {method_name}: {e}")
        
        # Combine and rank recommendations
        content_scores = defaultdict(list)
        
        for rec in all_recommendations:
            content_scores[rec['content_id']].append({
                'score': rec['score'],
                'method': rec['method']
            })
        
        # Calculate ensemble scores
        final_recommendations = []
        
        # Method weights
        method_weights = {
            'als': 0.25,
            'content_based': 0.20,
            'deep_learning': 0.25,
            'sequential': 0.15,
            'graph_neural_network': 0.15
        }
        
        for content_id, scores in content_scores.items():
            # Weighted average
            weighted_score = 0
            total_weight = 0
            
            for score_info in scores:
                method = score_info['method']
                weight = method_weights.get(method, 0.1)
                weighted_score += score_info['score'] * weight
                total_weight += weight
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
                
                # Boost score based on number of methods
                diversity_boost = len(scores) / len(method_weights)
                final_score *= (1 + 0.1 * diversity_boost)
                
                final_recommendations.append({
                    'content_id': content_id,
                    'score': final_score,
                    'methods': [s['method'] for s in scores],
                    'method_count': len(scores)
                })
        
        # Sort by final score
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return final_recommendations[:k]

# Global recommendation engine
recommendation_engine = HybridRecommendationEngine()

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
@cache_result(expire_time=1800)  # Cache for 30 minutes
def get_recommendations():
    """Main recommendation endpoint"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        limit = min(data.get('limit', 20), 50)  # Max 50 recommendations
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
        elif method == 'deep_learning':
            recommendations = recommendation_engine.get_deep_learning_recommendations(user_id, limit)
        elif method == 'sequential':
            recommendations = recommendation_engine.get_sequential_recommendations(user_id, limit)
        elif method == 'graph_based':
            recommendations = recommendation_engine.get_graph_based_recommendations(user_id, limit)
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
@cache_result(expire_time=7200)  # Cache for 2 hours
def get_similar_content(content_id):
    """Get similar content recommendations"""
    try:
        limit = min(int(request.args.get('limit', 10)), 20)
        
        recommendations = recommendation_engine.get_content_based_recommendations_for_item(
            content_id, limit
        )
        
        return jsonify({
            'content_id': content_id,
            'similar_content': recommendations,
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
        'faiss_index_ready': recommendation_engine.faiss_index is not None,
        'content_features_ready': recommendation_engine.content_features is not None,
        'graph_data_ready': recommendation_engine.graph_data is not None,
        'device': str(DEVICE),
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics for monitoring"""
    try:
        # Basic metrics
        metrics = {
            'models_count': len(recommendation_engine.models),
            'gpu_memory_used': 0,
            'gpu_memory_total': 0,
            'cache_hit_rate': 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # GPU metrics
        if torch.cuda.is_available():
            metrics['gpu_memory_used'] = torch.cuda.memory_allocated() / 1024**3  # GB
            metrics['gpu_memory_total'] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        # Redis metrics
        if REDIS_AVAILABLE:
            try:
                info = redis_client.info()
                metrics['redis_memory_used'] = info.get('used_memory_human', 'N/A')
                metrics['redis_connected_clients'] = info.get('connected_clients', 0)
            except:
                pass
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({'error': 'Failed to get metrics'}), 500

# Initialize models on startup
@app.before_first_request
def initialize_models():
    """Initialize models when the app starts"""
    try:
        logger.info("Initializing models on startup...")
        recommendation_engine.train_models()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")

if __name__ == '__main__':
    # Train models on startup if not running with a WSGI server
    try:
        recommendation_engine.train_models()
    except Exception as e:
        logger.error(f"Failed to train models on startup: {e}")
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5001)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    )