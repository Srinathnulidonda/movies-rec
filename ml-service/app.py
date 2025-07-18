import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
import xgboost as xgb
from surprise import Dataset as SurpriseDataset, Reader, SVD, SVDpp, NMF, KNNWithMeans
from surprise.model_selection import train_test_split as surprise_split
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
import faiss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, NMF as skNMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
import networkx as nx
from scipy import sparse
from scipy.spatial.distance import cosine
import joblib
import pickle
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import logging
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://localhost/recommendation_system')
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    MODEL_UPDATE_INTERVAL = int(os.environ.get('MODEL_UPDATE_INTERVAL', 3600))
    
    # Model parameters
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    N_FACTORS = 150
    MIN_INTERACTIONS = 3
    EXPLORATION_RATE = 0.1
    
    # Advanced parameters
    GRAPH_WALK_LENGTH = 10
    GRAPH_NUM_WALKS = 20
    SESSION_WINDOW = 30  # minutes
    TREND_WINDOW = 7  # days

app.config.from_object(Config)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis
try:
    redis_client = redis.from_url(app.config['REDIS_URL'])
    redis_client.ping()
    logger.info("Connected to Redis")
except:
    redis_client = None
    logger.warning("Redis not available, caching disabled")

# Database connection
def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(app.config['DATABASE_URL'], cursor_factory=RealDictCursor)

# PyTorch Models
class NeuralCollaborativeFiltering(nn.Module):
    """Advanced NCF model with attention mechanism"""
    
    def __init__(self, n_users, n_items, embedding_dim=128, hidden_dims=[256, 128, 64]):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # GMF pathway
        self.gmf_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP pathway
        mlp_dims = [embedding_dim * 2] + hidden_dims
        mlp_layers = []
        for i in range(len(mlp_dims) - 1):
            mlp_layers.append(nn.Linear(mlp_dims[i], mlp_dims[i+1]))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=4)
        
        # Final layers
        self.fusion = nn.Linear(hidden_dims[-1] + embedding_dim, 64)
        self.output = nn.Linear(64, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.gmf_user_embedding.weight)
        nn.init.xavier_uniform_(self.gmf_item_embedding.weight)
    
    def forward(self, user_ids, item_ids):
        # MLP pathway
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        mlp_input = torch.cat([user_embed, item_embed], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # GMF pathway
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user * gmf_item
        
        # Attention over embeddings
        stacked = torch.stack([user_embed, item_embed], dim=0)
        attended, _ = self.attention(stacked, stacked, stacked)
        attended_output = attended.mean(dim=0)
        
        # Fusion
        combined = torch.cat([mlp_output, gmf_output], dim=-1)
        fusion = torch.relu(self.fusion(combined))
        output = torch.sigmoid(self.output(fusion))
        
        return output.squeeze()

class SessionBasedRNN(nn.Module):
    """Session-based recommendations using GRU"""
    
    def __init__(self, n_items, embedding_dim=128, hidden_dim=256, n_layers=2):
        super(SessionBasedRNN, self).__init__()
        
        self.embedding = nn.Embedding(n_items, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, 
                          batch_first=True, dropout=0.2)
        self.output = nn.Linear(hidden_dim, n_items)
        
    def forward(self, sequences):
        embedded = self.embedding(sequences)
        output, hidden = self.gru(embedded)
        # Use last hidden state
        predictions = self.output(output[:, -1, :])
        return predictions

class GraphAttentionNetwork(nn.Module):
    """GAT for graph-based recommendations"""
    
    def __init__(self, n_nodes, embedding_dim=128, n_heads=4):
        super(GraphAttentionNetwork, self).__init__()
        
        self.embedding = nn.Embedding(n_nodes, embedding_dim)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, n_heads) 
            for _ in range(2)
        ])
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, node_ids, adjacency_matrix):
        embeddings = self.embedding(node_ids)
        
        # Apply attention layers
        for attention in self.attention_layers:
            attended, _ = attention(embeddings, embeddings, embeddings, 
                                   attn_mask=adjacency_matrix)
            embeddings = self.layer_norm(embeddings + attended)
            embeddings = self.layer_norm(embeddings + self.feed_forward(embeddings))
        
        return embeddings

# Advanced Recommendation Models
class ImplicitFeedbackModel:
    """Handle implicit feedback with multiple algorithms"""
    
    def __init__(self):
        self.als_model = AlternatingLeastSquares(
            factors=100, 
            regularization=0.01,
            iterations=50,
            use_gpu=torch.cuda.is_available()
        )
        self.bpr_model = BayesianPersonalizedRanking(
            factors=100,
            learning_rate=0.01,
            regularization=0.01,
            iterations=100
        )
        self.lmf_model = LogisticMatrixFactorization(
            factors=100,
            learning_rate=1.0,
            regularization=0.01,
            iterations=30
        )
        self.user_items = None
        self.item_users = None
        
    def fit(self, interaction_matrix):
        """Train all implicit feedback models"""
        self.user_items = interaction_matrix
        self.item_users = interaction_matrix.T
        
        # Train models
        self.als_model.fit(self.user_items)
        self.bpr_model.fit(self.user_items)
        self.lmf_model.fit(self.user_items)
        
        logger.info("Trained implicit feedback models")
        
    def recommend(self, user_id, n_recommendations=20):
        """Get recommendations combining all models"""
        recommendations = []
        
        # ALS recommendations
        als_recs = self.als_model.recommend(
            user_id, self.user_items[user_id], N=n_recommendations
        )
        
        # BPR recommendations
        bpr_recs = self.bpr_model.recommend(
            user_id, self.user_items[user_id], N=n_recommendations
        )
        
        # LMF recommendations
        lmf_recs = self.lmf_model.recommend(
            user_id, self.user_items[user_id], N=n_recommendations
        )
        
        # Combine recommendations with weighted voting
        item_scores = defaultdict(float)
        
        for item_id, score in als_recs:
            item_scores[item_id] += score * 0.4
            
        for item_id, score in bpr_recs:
            item_scores[item_id] += score * 0.3
            
        for item_id, score in lmf_recs:
            item_scores[item_id] += score * 0.3
        
        # Sort and return top items
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]

class GradientBoostingRecommender:
    """Use gradient boosting for recommendation ranking"""
    
    def __init__(self):
        self.lgb_model = None
        self.xgb_model = None
        self.feature_columns = []
        
    def prepare_features(self, interactions_df, content_df, user_df):
        """Engineer features for gradient boosting"""
        features = []
        
        # Merge dataframes
        data = interactions_df.merge(
            content_df, left_on='content_id', right_on='id', suffixes=('', '_content')
        ).merge(
            user_df, left_on='user_id', right_on='id', suffixes=('', '_user')
        )
        
        # User features
        user_features = [
            'user_total_views', 'user_avg_rating', 'user_genre_diversity',
            'user_activity_days', 'user_favorite_count'
        ]
        
        # Content features  
        content_features = [
            'content_popularity', 'content_rating', 'content_age_days',
            'content_genre_count', 'content_cast_popularity'
        ]
        
        # Interaction features
        interaction_features = [
            'user_content_genre_match', 'time_since_release',
            'user_language_match', 'trending_score'
        ]
        
        # Calculate features
        for _, row in data.iterrows():
            feat = {}
            
            # User features
            feat['user_total_views'] = row.get('user_view_count', 0)
            feat['user_avg_rating'] = row.get('user_avg_rating', 0)
            feat['user_genre_diversity'] = len(set(row.get('user_genres', [])))
            feat['user_activity_days'] = row.get('user_active_days', 0)
            feat['user_favorite_count'] = row.get('user_favorites', 0)
            
            # Content features
            feat['content_popularity'] = row.get('popularity_score', 0)
            feat['content_rating'] = row.get('tmdb_rating', 0)
            feat['content_age_days'] = (datetime.now() - row.get('release_date', datetime.now())).days
            feat['content_genre_count'] = len(row.get('genres', []))
            feat['content_cast_popularity'] = row.get('cast_popularity', 0)
            
            # Interaction features
            user_genres = set(row.get('user_preferred_genres', []))
            content_genres = set(row.get('genres', []))
            feat['user_content_genre_match'] = len(user_genres & content_genres) / max(len(content_genres), 1)
            
            feat['time_since_release'] = feat['content_age_days']
            feat['user_language_match'] = 1 if row.get('language') in row.get('user_languages', []) else 0
            feat['trending_score'] = row.get('trending_score', 0)
            
            features.append(feat)
        
        features_df = pd.DataFrame(features)
        self.feature_columns = features_df.columns.tolist()
        
        return features_df
    
    def train(self, features_df, labels):
        """Train gradient boosting models"""
        # LightGBM
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        train_data = lgb.Dataset(features_df, label=labels)
        self.lgb_model = lgb.train(
            lgb_params, 
            train_data, 
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # XGBoost
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8
        }
        
        dtrain = xgb.DMatrix(features_df, label=labels)
        self.xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        logger.info("Trained gradient boosting models")
    
    def predict(self, features_df):
        """Get predictions from ensemble"""
        lgb_pred = self.lgb_model.predict(features_df, num_iteration=self.lgb_model.best_iteration)
        xgb_pred = self.xgb_model.predict(xgb.DMatrix(features_df))
        
        # Ensemble predictions
        return (lgb_pred + xgb_pred) / 2

class GraphRecommender:
    """Graph-based recommendations using random walks and node embeddings"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.node_embeddings = {}
        
    def build_graph(self, interactions_df, content_df):
        """Build user-item interaction graph"""
        # Add nodes
        users = interactions_df['user_id'].unique()
        items = interactions_df['content_id'].unique()
        
        self.graph.add_nodes_from([f"u_{u}" for u in users], bipartite=0)
        self.graph.add_nodes_from([f"i_{i}" for i in items], bipartite=1)
        
        # Add edges with weights
        for _, row in interactions_df.iterrows():
            user_node = f"u_{row['user_id']}"
            item_node = f"i_{row['content_id']}"
            weight = row.get('rating', 1) / 10.0
            
            self.graph.add_edge(user_node, item_node, weight=weight)
        
        # Add item-item edges based on similarity
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                item1 = items[i]
                item2 = items[j]
                
                # Calculate similarity based on content features
                similarity = self._calculate_item_similarity(item1, item2, content_df)
                if similarity > 0.5:
                    self.graph.add_edge(f"i_{item1}", f"i_{item2}", weight=similarity)
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def _calculate_item_similarity(self, item1, item2, content_df):
        """Calculate similarity between two items"""
        item1_data = content_df[content_df['id'] == item1].iloc[0]
        item2_data = content_df[content_df['id'] == item2].iloc[0]
        
        # Genre similarity
        genres1 = set(item1_data.get('genres', []))
        genres2 = set(item2_data.get('genres', []))
        genre_sim = len(genres1 & genres2) / max(len(genres1 | genres2), 1)
        
        # Rating similarity
        rating1 = item1_data.get('tmdb_rating', 0)
        rating2 = item2_data.get('tmdb_rating', 0)
        rating_sim = 1 - abs(rating1 - rating2) / 10
        
        return (genre_sim + rating_sim) / 2
    
    def generate_embeddings(self, embedding_dim=128):
        """Generate node embeddings using Node2Vec-like approach"""
        # Random walk sampling
        walks = []
        nodes = list(self.graph.nodes())
        
        for _ in range(app.config['GRAPH_NUM_WALKS']):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self._random_walk(node, app.config['GRAPH_WALK_LENGTH'])
                walks.append(walk)
        
        # Train embedding model (simplified Word2Vec approach)
        from gensim.models import Word2Vec
        
        model = Word2Vec(
            walks,
            vector_size=embedding_dim,
            window=5,
            min_count=1,
            sg=1,  # Skip-gram
            workers=4
        )
        
        # Store embeddings
        for node in self.graph.nodes():
            if node in model.wv:
                self.node_embeddings[node] = model.wv[node]
    
    def _random_walk(self, start_node, walk_length):
        """Perform random walk from start node"""
        walk = [start_node]
        current = start_node
        
        for _ in range(walk_length - 1):
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
                
            # Weighted random selection
            weights = [self.graph[current][n].get('weight', 1) for n in neighbors]
            probabilities = np.array(weights) / sum(weights)
            
            current = np.random.choice(neighbors, p=probabilities)
            walk.append(current)
        
        return walk
    
    def recommend(self, user_id, n_recommendations=20):
        """Get recommendations using graph embeddings"""
        user_node = f"u_{user_id}"
        
        if user_node not in self.node_embeddings:
            return []
        
        user_embedding = self.node_embeddings[user_node]
        
        # Find similar items
        item_scores = []
        for node, embedding in self.node_embeddings.items():
            if node.startswith('i_'):
                # Cosine similarity
                similarity = np.dot(user_embedding, embedding) / (
                    np.linalg.norm(user_embedding) * np.linalg.norm(embedding)
                )
                item_id = int(node[2:])  # Remove 'i_' prefix
                item_scores.append((item_id, similarity))
        
        # Sort and return top items
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_recommendations]

class VectorSearchEngine:
    """Fast similarity search using FAISS"""
    
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        self.index = None
        self.id_map = {}
        
    def build_index(self, embeddings_dict):
        """Build FAISS index for fast similarity search"""
        # Convert embeddings to numpy array
        ids = []
        embeddings = []
        
        for item_id, embedding in embeddings_dict.items():
            ids.append(item_id)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings_array)
        
        # Build index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
        self.index.add(embeddings_array)
        
        # Store ID mapping
        self.id_map = {i: item_id for i, item_id in enumerate(ids)}
        
        logger.info(f"Built FAISS index with {len(ids)} items")
    
    def search(self, query_embedding, k=20):
        """Search for similar items"""
        query = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx in self.id_map:
                results.append((self.id_map[idx], float(dist)))
        
        return results

class MultiArmedBandit:
    """Multi-armed bandit for exploration vs exploitation"""
    
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms)
        
    def select_arm(self):
        """Select arm using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            # Exploration
            return np.random.randint(self.n_arms)
        else:
            # Exploitation
            return np.argmax(self.q_values)
    
    def update(self, arm, reward):
        """Update Q-values based on reward"""
        self.n_pulls[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.n_pulls[arm]

class AdvancedHybridRecommender:
    """Advanced hybrid recommender combining all approaches"""
    
    def __init__(self):
        # Traditional models
        self.cf_models = {
            'svd': SVD(n_factors=150, n_epochs=20),
            'svdpp': SVDpp(n_factors=100, n_epochs=20),
            'nmf': NMF(n_factors=100, n_epochs=20),
            'knn': KNNWithMeans(k=50, sim_options={'name': 'cosine'})
        }
        
        # Advanced models
        self.implicit_model = ImplicitFeedbackModel()
        self.gb_model = GradientBoostingRecommender()
        self.graph_model = GraphRecommender()
        self.vector_search = VectorSearchEngine()
        
        # Neural models
        self.ncf_model = None
        self.session_model = None
        self.gat_model = None
        
        # Bandits for online learning
        self.bandits = {}
        
        # Model weights
        self.model_weights = {
            'collaborative': 0.25,
            'implicit': 0.20,
            'content': 0.15,
            'graph': 0.15,
            'neural': 0.15,
            'trending': 0.10
        }
        
        # Feature engineering
        self.feature_extractor = FeatureExtractor()
        self.trend_analyzer = TrendAnalyzer()
        
        self.last_update = None
    
    def update_models(self):
        """Update all recommendation models"""
        try:
            conn = get_db_connection()
            
            # Load all necessary data
            data = self._load_training_data(conn)
            
            # Train collaborative filtering models
            self._train_collaborative_models(data['ratings'])
            
            # Train implicit feedback models
            self._train_implicit_models(data['interactions'])
            
            # Train gradient boosting models
            self._train_gradient_boosting(data)
            
            # Build graph and train graph models
            self._train_graph_models(data)
            
            # Train neural models
            self._train_neural_models(data)
            
            # Update feature extractors
            self.feature_extractor.fit(data['content'])
            
            # Analyze trends
            self.trend_analyzer.analyze(data['interactions'], data['content'])
            
            self.last_update = datetime.utcnow()
            logger.info("Successfully updated all recommendation models")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
            raise
    
    def _load_training_data(self, conn):
        """Load and prepare all training data"""
        # Load ratings
        ratings_query = """
            SELECT user_id, content_id, rating, created_at
            FROM ratings
        """
        ratings_df = pd.read_sql(ratings_query, conn)
        
        # Load all interactions
        interactions_query = """
            SELECT user_id, content_id, interaction_type, created_at, 
                   CASE interaction_type
                       WHEN 'view' THEN 3
                       WHEN 'favorite' THEN 8
                       WHEN 'wishlist' THEN 5
                       WHEN 'completed' THEN 7
                       ELSE 1
                   END as implicit_rating
            FROM (
                SELECT user_id, content_id, 'view' as interaction_type, watched_at as created_at
                FROM watch_history
                UNION ALL
                SELECT user_id, content_id, 'favorite' as interaction_type, added_at as created_at
                FROM favorites
                UNION ALL
                SELECT user_id, content_id, 'wishlist' as interaction_type, added_at as created_at
                FROM wishlist
            ) interactions
        """
        interactions_df = pd.read_sql(interactions_query, conn)
        
        # Load content features
        content_query = """
            SELECT id, title, description, genres, cast, crew,
                   tmdb_rating, imdb_rating, user_rating, critic_score,
                   view_count, popularity_score, release_date,
                   is_telugu, is_hindi, is_tamil, is_kannada,
                   languages, countries, runtime
            FROM content
        """
        content_df = pd.read_sql(content_query, conn)
        
        # Parse JSON fields
        content_df['genres'] = content_df['genres'].apply(lambda x: json.loads(x) if x else [])
        content_df['cast'] = content_df['cast'].apply(lambda x: json.loads(x) if x else [])
        content_df['languages'] = content_df['languages'].apply(lambda x: json.loads(x) if x else [])
        
        # Load user features
        user_query = """
            SELECT u.id, u.preferred_genres, u.preferred_languages,
                   COUNT(DISTINCT wh.content_id) as view_count,
                   COUNT(DISTINCT f.content_id) as favorite_count,
                   AVG(r.rating) as avg_rating
            FROM users u
            LEFT JOIN watch_history wh ON u.id = wh.user_id
            LEFT JOIN favorites f ON u.id = f.user_id
            LEFT JOIN ratings r ON u.id = r.user_id
            GROUP BY u.id, u.preferred_genres, u.preferred_languages
        """
        user_df = pd.read_sql(user_query, conn)
        
        # Parse JSON fields
        user_df['preferred_genres'] = user_df['preferred_genres'].apply(
            lambda x: json.loads(x) if x else []
        )
        
        return {
            'ratings': ratings_df,
            'interactions': interactions_df,
            'content': content_df,
            'users': user_df
        }
    
    def _train_collaborative_models(self, ratings_df):
        """Train collaborative filtering models"""
        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 10))
        data = SurpriseDataset.load_from_df(
            ratings_df[['user_id', 'content_id', 'rating']], 
            reader
        )
        
        # Train each model
        for name, model in self.cf_models.items():
            trainset = data.build_full_trainset()
            model.fit(trainset)
            logger.info(f"Trained collaborative filtering model: {name}")
    
    def _train_implicit_models(self, interactions_df):
        """Train implicit feedback models"""
        # Create sparse matrix
        users = interactions_df['user_id'].unique()
        items = interactions_df['content_id'].unique()
        
        user_map = {u: i for i, u in enumerate(users)}
        item_map = {i: idx for idx, i in enumerate(items)}
        
        row = [user_map[u] for u in interactions_df['user_id']]
        col = [item_map[i] for i in interactions_df['content_id']]
        data = interactions_df['implicit_rating'].values
        
        interaction_matrix = sparse.csr_matrix(
            (data, (row, col)), 
            shape=(len(users), len(items))
        )
        
        # Train implicit models
        self.implicit_model.fit(interaction_matrix)
    
    def _train_gradient_boosting(self, data):
        """Train gradient boosting models"""
        # Prepare features
        features_df = self.gb_model.prepare_features(
            data['interactions'], 
            data['content'], 
            data['users']
        )
        
        # Create labels (1 for positive interactions, 0 for negative sampling)
        labels = np.ones(len(features_df))
        
        # Add negative samples
        negative_samples = self._generate_negative_samples(data)
        negative_features = self.gb_model.prepare_features(
            negative_samples, 
            data['content'], 
            data['users']
        )
        negative_labels = np.zeros(len(negative_features))
        
        # Combine positive and negative
        all_features = pd.concat([features_df, negative_features])
        all_labels = np.concatenate([labels, negative_labels])
        
        # Train model
        self.gb_model.train(all_features, all_labels)
    
    def _generate_negative_samples(self, data, ratio=1.0):
        """Generate negative samples for training"""
        interactions = data['interactions']
        all_users = interactions['user_id'].unique()
        all_items = data['content']['id'].unique()
        
        # Get existing interactions
        existing = set(
            zip(interactions['user_id'], interactions['content_id'])
        )
        
        # Generate negative samples
        negative_samples = []
        n_negative = int(len(interactions) * ratio)
        
        while len(negative_samples) < n_negative:
            user = np.random.choice(all_users)
            item = np.random.choice(all_items)
            
            if (user, item) not in existing:
                negative_samples.append({
                    'user_id': user,
                    'content_id': item,
                    'interaction_type': 'negative'
                })
        
        return pd.DataFrame(negative_samples)
    
    def _train_graph_models(self, data):
        """Train graph-based models"""
        # Build graph
        self.graph_model.build_graph(data['interactions'], data['content'])
        
        # Generate embeddings
        self.graph_model.generate_embeddings()
        
        # Build vector search index
        content_embeddings = {
            int(node[2:]): embedding 
            for node, embedding in self.graph_model.node_embeddings.items()
            if node.startswith('i_')
        }
        self.vector_search.build_index(content_embeddings)
    
    def _train_neural_models(self, data):
        """Train PyTorch neural models"""
        # Prepare data
        interactions = data['interactions']
        n_users = interactions['user_id'].nunique()
        n_items = interactions['content_id'].nunique()
        
        # Map IDs to indices
        user_map = {uid: idx for idx, uid in enumerate(interactions['user_id'].unique())}
        item_map = {iid: idx for idx, iid in enumerate(interactions['content_id'].unique())}
        
        # Train NCF
        self.ncf_model = NeuralCollaborativeFiltering(n_users, n_items)
        self._train_ncf(interactions, user_map, item_map)
        
        # Train session-based model
        self.session_model = SessionBasedRNN(n_items)
        self._train_session_model(interactions, item_map)
    
    def _train_ncf(self, interactions_df, user_map, item_map, epochs=20):
        """Train Neural Collaborative Filtering model"""
        # Prepare data
        users = torch.LongTensor([user_map[u] for u in interactions_df['user_id']])
        items = torch.LongTensor([item_map[i] for i in interactions_df['content_id']])
        ratings = torch.FloatTensor(interactions_df['implicit_rating'].values / 10)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(users, items, ratings)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        # Training
        optimizer = optim.Adam(self.ncf_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        self.ncf_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_users, batch_items, batch_ratings in dataloader:
                optimizer.zero_grad()
                predictions = self.ncf_model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                logger.info(f"NCF Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def _train_session_model(self, interactions_df, item_map):
        """Train session-based RNN model"""
        # Group interactions by user and time
        sessions = []
        
        for user_id, user_interactions in interactions_df.groupby('user_id'):
            # Sort by time
            user_interactions = user_interactions.sort_values('created_at')
            
            # Create sessions (split by time gaps)
            session = []
            last_time = None
            
            for _, interaction in user_interactions.iterrows():
                if last_time and (interaction['created_at'] - last_time).total_seconds() > 1800:  # 30 min gap
                    if len(session) > 1:
                        sessions.append(session)
                    session = []
                
                session.append(item_map.get(interaction['content_id'], 0))
                last_time = interaction['created_at']
            
            if len(session) > 1:
                sessions.append(session)
        
        # Train model on sessions
        # ... (implementation details)
    
    def get_recommendations(self, user_id, n_recommendations=20):
        """Get hybrid recommendations combining all models"""
        recommendations = defaultdict(list)
        
        # 1. Collaborative filtering recommendations
        cf_recs = self._get_collaborative_recommendations(user_id, n_recommendations)
        recommendations['collaborative'] = cf_recs
        
        # 2. Implicit feedback recommendations
        implicit_recs = self._get_implicit_recommendations(user_id, n_recommendations)
        recommendations['implicit'] = implicit_recs
        
        # 3. Content-based recommendations
        content_recs = self._get_content_recommendations(user_id, n_recommendations)
        recommendations['content'] = content_recs
        
        # 4. Graph-based recommendations
        graph_recs = self.graph_model.recommend(user_id, n_recommendations)
        recommendations['graph'] = graph_recs
        
        # 5. Neural network recommendations
        neural_recs = self._get_neural_recommendations(user_id, n_recommendations)
        recommendations['neural'] = neural_recs
        
        # 6. Trending recommendations
        trending_recs = self.trend_analyzer.get_trending(n_recommendations)
        recommendations['trending'] = trending_recs
        
        # Combine all recommendations using weighted voting
        final_recommendations = self._combine_recommendations(
            recommendations, 
            self.model_weights,
            n_recommendations
        )
        
        # Apply multi-armed bandit for exploration
        final_recommendations = self._apply_exploration(
            user_id, 
            final_recommendations,
            n_recommendations
        )
        
        return final_recommendations
    
    def _get_collaborative_recommendations(self, user_id, n_recommendations):
        """Get collaborative filtering recommendations"""
        all_recs = []
        
        for name, model in self.cf_models.items():
            try:
                # Get predictions for all items
                predictions = []
                for item_id in range(1000):  # Assuming item IDs up to 1000
                    pred = model.predict(user_id, item_id)
                    predictions.append((item_id, pred.est))
                
                # Sort and get top N
                predictions.sort(key=lambda x: x[1], reverse=True)
                all_recs.extend(predictions[:n_recommendations])
            except:
                pass
        
        # Aggregate and rank
        item_scores = defaultdict(list)
        for item_id, score in all_recs:
            item_scores[item_id].append(score)
        
        # Average scores
        final_recs = [
            (item_id, np.mean(scores))
            for item_id, scores in item_scores.items()
        ]
        
        final_recs.sort(key=lambda x: x[1], reverse=True)
        return final_recs[:n_recommendations]
    
    def _get_implicit_recommendations(self, user_id, n_recommendations):
        """Get implicit feedback recommendations"""
        return self.implicit_model.recommend(user_id, n_recommendations)
    
    def _get_content_recommendations(self, user_id, n_recommendations):
        """Get content-based recommendations"""
        # Get user profile
        user_profile = self._build_user_profile(user_id)
        
        if not user_profile:
            return []
        
        # Search for similar content
        similar_items = self.vector_search.search(user_profile, n_recommendations * 2)
        
        # Filter out already interacted items
        interacted = self._get_user_interactions(user_id)
        filtered = [
            (item_id, score) 
            for item_id, score in similar_items 
            if item_id not in interacted
        ]
        
        return filtered[:n_recommendations]
    
    def _get_neural_recommendations(self, user_id, n_recommendations):
        """Get neural network recommendations"""
        if not self.ncf_model:
            return []
        
        # Get predictions for all items
        predictions = []
        user_tensor = torch.LongTensor([user_id])
        
        with torch.no_grad():
            for item_id in range(1000):  # Assuming item IDs up to 1000
                item_tensor = torch.LongTensor([item_id])
                score = self.ncf_model(user_tensor, item_tensor).item()
                predictions.append((item_id, score))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def _combine_recommendations(self, recommendations_dict, weights, n_recommendations):
        """Combine recommendations from multiple models"""
        combined_scores = defaultdict(float)
        
        for method, recs in recommendations_dict.items():
            weight = weights.get(method, 0.1)
            
            # Normalize scores to [0, 1]
            if recs:
                max_score = max(score for _, score in recs)
                min_score = min(score for _, score in recs)
                score_range = max_score - min_score if max_score != min_score else 1
                
                for item_id, score in recs:
                    normalized_score = (score - min_score) / score_range
                    combined_scores[item_id] += normalized_score * weight
        
        # Sort by combined score
        final_recs = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return final_recs[:n_recommendations]
    
    def _apply_exploration(self, user_id, recommendations, n_recommendations):
        """Apply multi-armed bandit for exploration vs exploitation"""
        if user_id not in self.bandits:
            self.bandits[user_id] = MultiArmedBandit(
                n_arms=len(recommendations),
                epsilon=app.config['EXPLORATION_RATE']
            )
        
        bandit = self.bandits[user_id]
        
        # Mix exploitation and exploration
        exploited = recommendations[:int(n_recommendations * 0.8)]
        
        # Get some random items for exploration
        try:
            conn = get_db_connection()
            explore_query = """
                SELECT id FROM content 
                WHERE id NOT IN (
                    SELECT content_id FROM watch_history WHERE user_id = %s
                    UNION
                    SELECT content_id FROM ratings WHERE user_id = %s
                )
                ORDER BY RANDOM() 
                LIMIT %s
            """
            
            with conn.cursor() as cursor:
                cursor.execute(explore_query, (user_id, user_id, int(n_recommendations * 0.2)))
                explored = [(row['id'], 0.5) for row in cursor.fetchall()]
            
            conn.close()
            
            return exploited + explored
            
        except:
            return recommendations
    
    def _build_user_profile(self, user_id):
        """Build user profile vector"""
        try:
            conn = get_db_connection()
            
            # Get user's interacted items
            query = """
                SELECT c.genres, c.cast, r.rating
                FROM ratings r
                JOIN content c ON r.content_id = c.id
                WHERE r.user_id = %s AND r.rating >= 7
                ORDER BY r.created_at DESC
                LIMIT 20
            """
            
            with conn.cursor() as cursor:
                cursor.execute(query, (user_id,))
                interactions = cursor.fetchall()
            
            conn.close()
            
            if not interactions:
                return None
            
            # Build profile vector
            # ... (implementation details)
            
            return np.random.rand(128)  # Placeholder
            
        except:
            return None
    
    def _get_user_interactions(self, user_id):
        """Get set of items user has interacted with"""
        try:
            conn = get_db_connection()
            
            query = """
                SELECT DISTINCT content_id 
                FROM (
                    SELECT content_id FROM watch_history WHERE user_id = %s
                    UNION
                    SELECT content_id FROM ratings WHERE user_id = %s
                    UNION
                    SELECT content_id FROM favorites WHERE user_id = %s
                ) interactions
            """
            
            with conn.cursor() as cursor:
                cursor.execute(query, (user_id, user_id, user_id))
                interacted = {row['content_id'] for row in cursor.fetchall()}
            
            conn.close()
            return interacted
            
        except:
            return set()

class FeatureExtractor:
    """Extract features from content for various models"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.genre_encoder = None
        self.pca = PCA(n_components=50)
        
    def fit(self, content_df):
        """Fit feature extractors"""
        # TF-IDF for text features
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        text_features = content_df['title'] + ' ' + content_df['description'].fillna('')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_vectorizer.fit(text_features)
        
        # Genre encoding
        all_genres = set()
        for genres in content_df['genres']:
            all_genres.update(genres)
        
        self.genre_encoder = {genre: i for i, genre in enumerate(all_genres)}
        
        logger.info("Fitted feature extractors")
    
    def extract(self, content):
        """Extract features from content"""
        features = []
        
        # Text features
        text = content.get('title', '') + ' ' + content.get('description', '')
        text_features = self.tfidf_vectorizer.transform([text]).toarray()[0]
        features.extend(text_features)
        
        # Genre features
        genre_vector = np.zeros(len(self.genre_encoder))
        for genre in content.get('genres', []):
            if genre in self.genre_encoder:
                genre_vector[self.genre_encoder[genre]] = 1
        features.extend(genre_vector)
        
        # Numerical features
        features.extend([
            content.get('tmdb_rating', 0) / 10,
            content.get('imdb_rating', 0) / 10,
            content.get('popularity_score', 0) / 1000,
            content.get('view_count', 0) / 1000,
            1 if content.get('is_telugu') else 0,
            1 if content.get('is_hindi') else 0,
            1 if content.get('is_tamil') else 0,
            1 if content.get('is_kannada') else 0
        ])
        
        return np.array(features)

class TrendAnalyzer:
    """Analyze trends in user behavior and content popularity"""
    
    def __init__(self):
        self.trending_items = []
        self.seasonal_patterns = {}
        self.genre_trends = {}
        
    def analyze(self, interactions_df, content_df):
        """Analyze trends from interaction data"""
        # Time-based trending
        recent = datetime.utcnow() - timedelta(days=app.config['TREND_WINDOW'])
        recent_interactions = interactions_df[
            pd.to_datetime(interactions_df['created_at']) > recent
        ]
        
        # Calculate trending score
        trending_scores = recent_interactions.groupby('content_id').agg({
            'user_id': 'count',
            'implicit_rating': 'mean'
        }).rename(columns={'user_id': 'interaction_count'})
        
        # Weighted trending score
        trending_scores['trend_score'] = (
            trending_scores['interaction_count'] * 0.7 +
            trending_scores['implicit_rating'] * 10 * 0.3
        )
        
        self.trending_items = trending_scores.sort_values(
            'trend_score', ascending=False
        ).head(100).index.tolist()
        
        # Genre trends
        for _, interaction in recent_interactions.iterrows():
            content = content_df[content_df['id'] == interaction['content_id']]
            if not content.empty:
                genres = content.iloc[0]['genres']
                for genre in genres:
                    if genre not in self.genre_trends:
                        self.genre_trends[genre] = []
                    self.genre_trends[genre].append(interaction['created_at'])
        
        logger.info("Analyzed trends")
    
    def get_trending(self, n_items=20):
        """Get current trending items"""
        return [(item_id, 1.0) for item_id in self.trending_items[:n_items]]
    
    def get_genre_trends(self):
        """Get trending genres"""
        genre_scores = {}
        
        for genre, timestamps in self.genre_trends.items():
            # Recent activity
            recent = sum(1 for ts in timestamps if 
                        (datetime.utcnow() - pd.to_datetime(ts)).days < 7)
            genre_scores[genre] = recent
        
        return sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)

# Initialize recommender
recommender = AdvancedHybridRecommender()

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'last_model_update': recommender.last_update.isoformat() if recommender.last_update else None,
        'models_loaded': {
            'collaborative': bool(recommender.cf_models),
            'implicit': bool(recommender.implicit_model),
            'gradient_boosting': bool(recommender.gb_model.lgb_model),
            'graph': bool(recommender.graph_model.graph),
            'neural': bool(recommender.ncf_model)
        }
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get recommendations for a user"""
    data = request.get_json()
    user_id = data.get('user_id')
    n_recommendations = data.get('n_recommendations', 20)
    method = data.get('method', 'hybrid')
    
    if not user_id:
        return jsonify({'error': 'user_id required'}), 400
    
    # Check cache
    cache_key = f"recommendations:{user_id}:{n_recommendations}:{method}"
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            return jsonify({
                'recommendations': json.loads(cached),
                'method': method,
                'cached': True
            })
    
    # Get recommendations
    try:
        if method == 'hybrid':
            recommendations = recommender.get_recommendations(user_id, n_recommendations)
        elif method == 'collaborative':
            recommendations = recommender._get_collaborative_recommendations(user_id, n_recommendations)
        elif method == 'content':
            recommendations = recommender._get_content_recommendations(user_id, n_recommendations)
        elif method == 'graph':
            recommendations = recommender.graph_model.recommend(user_id, n_recommendations)
        elif method == 'neural':
            recommendations = recommender._get_neural_recommendations(user_id, n_recommendations)
        elif method == 'trending':
            recommendations = recommender.trend_analyzer.get_trending(n_recommendations)
        else:
            return jsonify({'error': 'Invalid method'}), 400
        
        # Format recommendations
        formatted_recs = [
            {'content_id': int(item_id), 'score': float(score)}
            for item_id, score in recommendations
        ]
        
        # Cache results
        if redis_client:
            redis_client.setex(
                cache_key,
                3600,  # 1 hour cache
                json.dumps(formatted_recs)
            )
        
        return jsonify({
            'recommendations': formatted_recs,
            'method': method,
            'cached': False
        })
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({'error': 'Failed to generate recommendations'}), 500

@app.route('/similar', methods=['POST'])
def find_similar():
    """Find similar content"""
    data = request.get_json()
    content_id = data.get('content_id')
    n_similar = data.get('n_similar', 10)
    method = data.get('method', 'content')
    
    if not content_id:
        return jsonify({'error': 'content_id required'}), 400
    
    try:
        if method == 'content':
            # Use vector search
            if content_id in recommender.graph_model.node_embeddings:
                embedding = recommender.graph_model.node_embeddings[f"i_{content_id}"]
                similar_items = recommender.vector_search.search(embedding, n_similar)
            else:
                similar_items = []
        elif method == 'collaborative':
            # Find users who liked this item and what else they liked
            similar_items = []
        else:
            return jsonify({'error': 'Invalid method'}), 400
        
        formatted_items = [
            {'content_id': int(item_id), 'similarity': float(score)}
            for item_id, score in similar_items
        ]
        
        return jsonify({
            'similar_items': formatted_items,
            'method': method
        })
        
    except Exception as e:
        logger.error(f"Error finding similar items: {e}")
        return jsonify({'error': 'Failed to find similar items'}), 500

@app.route('/explain', methods=['POST'])
def explain_recommendation():
    """Explain why an item was recommended"""
    data = request.get_json()
    user_id = data.get('user_id')
    content_id = data.get('content_id')
    
    if not all([user_id, content_id]):
        return jsonify({'error': 'user_id and content_id required'}), 400
    
    explanation = {
        'user_id': user_id,
        'content_id': content_id,
        'reasons': []
    }
    
    try:
        # Check if user has similar taste
        conn = get_db_connection()
        
        # Find similar users who liked this content
        similar_users_query = """
            SELECT u2.user_id, COUNT(*) as common_items
            FROM ratings r1
            JOIN ratings r2 ON r1.content_id = r2.content_id
            JOIN ratings u2 ON r2.user_id = u2.user_id
            WHERE r1.user_id = %s AND r2.user_id != %s
                AND r1.rating >= 7 AND r2.rating >= 7
                AND u2.content_id = %s AND u2.rating >= 7
            GROUP BY u2.user_id
            ORDER BY common_items DESC
            LIMIT 5
        """
        
        with conn.cursor() as cursor:
            cursor.execute(similar_users_query, (user_id, user_id, content_id))
            similar_users = cursor.fetchall()
        
        if similar_users:
            explanation['reasons'].append({
                'type': 'collaborative',
                'description': f"{len(similar_users)} users with similar taste enjoyed this content"
            })
        
        # Check genre match
        user_genres_query = """
            SELECT DISTINCT unnest(c.genres) as genre
            FROM ratings r
            JOIN content c ON r.content_id = c.id
            WHERE r.user_id = %s AND r.rating >= 7
        """
        
        content_genres_query = """
            SELECT genres FROM content WHERE id = %s
        """
        
        with conn.cursor() as cursor:
            cursor.execute(user_genres_query, (user_id,))
            user_genres = {row['genre'] for row in cursor.fetchall()}
            
            cursor.execute(content_genres_query, (content_id,))
            content_data = cursor.fetchone()
            content_genres = set(json.loads(content_data['genres']) if content_data else [])
        
        matching_genres = user_genres & content_genres
        if matching_genres:
            explanation['reasons'].append({
                'type': 'content',
                'description': f"Matches your preferred genres: {', '.join(matching_genres)}"
            })
        
        # Check if trending
        if content_id in recommender.trend_analyzer.trending_items[:20]:
            explanation['reasons'].append({
                'type': 'trending',
                'description': "Currently trending and popular"
            })
        
        conn.close()
        
        return jsonify(explanation)
        
    except Exception as e:
        logger.error(f"Error explaining recommendation: {e}")
        return jsonify({'error': 'Failed to explain recommendation'}), 500

@app.route('/feedback', methods=['POST'])
def record_feedback():
    """Record user feedback on recommendations"""
    data = request.get_json()
    user_id = data.get('user_id')
    content_id = data.get('content_id')
    feedback_type = data.get('feedback_type')  # 'click', 'like', 'dislike', 'watch'
    
    if not all([user_id, content_id, feedback_type]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        # Update bandit if exists
        if user_id in recommender.bandits:
            # Calculate reward based on feedback type
            rewards = {
                'click': 0.3,
                'watch': 0.7,
                'like': 1.0,
                'dislike': -0.5
            }
            reward = rewards.get(feedback_type, 0)
            
            # Update bandit (simplified - would need to track which arm corresponds to which item)
            recommender.bandits[user_id].update(0, reward)
        
        # Log feedback for future model updates
        if redis_client:
            feedback_key = f"feedback:{user_id}:{content_id}"
            redis_client.setex(feedback_key, 86400, feedback_type)  # 24 hour TTL
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        return jsonify({'error': 'Failed to record feedback'}), 500

@app.route('/update_models', methods=['POST'])
def update_models():
    """Manually trigger model update"""
    try:
        recommender.update_models()
        return jsonify({
            'status': 'success',
            'updated_at': recommender.last_update.isoformat() if recommender.last_update else None
        })
    except Exception as e:
        logger.error(f"Error updating models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    stats = {
        'last_update': recommender.last_update.isoformat() if recommender.last_update else None,
        'models': {
            'collaborative': {
                'svd': 'trained' if recommender.cf_models.get('svd') else 'not trained',
                'svdpp': 'trained' if recommender.cf_models.get('svdpp') else 'not trained',
                'nmf': 'trained' if recommender.cf_models.get('nmf') else 'not trained',
                'knn': 'trained' if recommender.cf_models.get('knn') else 'not trained'
            },
            'implicit': {
                'als': 'trained' if recommender.implicit_model.als_model else 'not trained',
                'bpr': 'trained' if recommender.implicit_model.bpr_model else 'not trained',
                'lmf': 'trained' if recommender.implicit_model.lmf_model else 'not trained'
            },
            'gradient_boosting': {
                'lightgbm': 'trained' if recommender.gb_model.lgb_model else 'not trained',
                'xgboost': 'trained' if recommender.gb_model.xgb_model else 'not trained'
            },
            'graph': {
                'nodes': recommender.graph_model.graph.number_of_nodes() if recommender.graph_model.graph else 0,
                'edges': recommender.graph_model.graph.number_of_edges() if recommender.graph_model.graph else 0
            },
            'neural': {
                'ncf': 'trained' if recommender.ncf_model else 'not trained',
                'session_rnn': 'trained' if recommender.session_model else 'not trained'
            }
        },
        'trends': {
            'trending_items': len(recommender.trend_analyzer.trending_items),
            'genre_trends': recommender.trend_analyzer.get_genre_trends()[:5]
        }
    }
    
    return jsonify(stats)

@app.route('/ab_test', methods=['POST'])
def ab_test():
    """A/B testing endpoint for comparing recommendation strategies"""
    data = request.get_json()
    user_id = data.get('user_id')
    n_recommendations = data.get('n_recommendations', 20)
    
    if not user_id:
        return jsonify({'error': 'user_id required'}), 400
    
    # Randomly assign user to test group
    test_group = 'A' if hash(str(user_id)) % 2 == 0 else 'B'
    
    if test_group == 'A':
        # Traditional collaborative filtering
        recommendations = recommender._get_collaborative_recommendations(user_id, n_recommendations)
    else:
        # Advanced hybrid approach
        recommendations = recommender.get_recommendations(user_id, n_recommendations)
    
    formatted_recs = [
        {'content_id': int(item_id), 'score': float(score)}
        for item_id, score in recommendations
    ]
    
    return jsonify({
        'recommendations': formatted_recs,
        'test_group': test_group,
        'strategy': 'collaborative' if test_group == 'A' else 'hybrid'
    })

# Background model update
from threading import Thread
import time

def periodic_model_update():
    """Periodically update recommendation models"""
    while True:
        time.sleep(app.config['MODEL_UPDATE_INTERVAL'])
        try:
            logger.info("Starting periodic model update")
            recommender.update_models()
            
            # Clean up old bandits
            if len(recommender.bandits) > 10000:
                # Keep only recently active users
                recommender.bandits = dict(
                    list(recommender.bandits.items())[-5000:]
                )
                
        except Exception as e:
            logger.error(f"Error in periodic model update: {e}")

# Start background thread
update_thread = Thread(target=periodic_model_update, daemon=True)
update_thread.start()

# Initialize models on startup
with app.app_context():
    try:
        recommender.update_models()
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")

if __name__ == '__main__':
    app.run(debug=True, port=5001)