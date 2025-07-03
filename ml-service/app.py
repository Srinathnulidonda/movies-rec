from flask import Flask, request, jsonify
import sqlite3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from textblob import TextBlob
import implicit
from scipy.sparse import csr_matrix
import faiss
import redis
import json
import logging
import os
from datetime import datetime, timedelta
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import asyncio
import aiohttp
import warnings
import time

# Try importing surprise, but allow the app to continue if it fails
try:
    from surprise import Dataset, Reader, SVD, NMF, KNNBasic
    from surprise.model_selection import cross_validate
    SURPRISE_AVAILABLE = True
except ImportError as e:
    SURPRISE_AVAILABLE = False
    logging.warning(f"Surprise library not available: {e}. Collaborative filtering models will be skipped.")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
DB_PATH = os.environ.get('DATABASE_URL', 'recommendations.db')
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
CACHE_DURATION = 3600  # 1 hour
MODEL_UPDATE_INTERVAL = 3600  # 1 hour

# Global variables for caching
recommendation_cache = {}
model_cache = {}
cache_lock = threading.Lock()

# Initialize Redis for caching
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    redis_client.ping()  # Test connection
except Exception as e:
    redis_client = None
    logger.warning(f"Redis not available, using in-memory caching: {e}")

# Custom Sampling Layer for VAE
class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

# [TrendAnalyzer class remains unchanged]
class TrendAnalyzer:
    """Advanced trend analysis for content recommendations"""
    def __init__(self):
        self.trend_weights = {
            'popularity_momentum': 0.3,
            'rating_trend': 0.25,
            'genre_trend': 0.2,
            'seasonal_trend': 0.15,
            'social_buzz': 0.1
        }
        self.trend_cache = {}

    def analyze_content_trends(self, content_data, time_window_days=7):
        try:
            df = pd.DataFrame(content_data)
            current_time = datetime.now()

            recent_interactions = df[df['created_at'] > (current_time - timedelta(days=time_window_days))]
            popularity_momentum = recent_interactions.groupby('content_id')['interaction_count'].sum()

            rating_trends = self.calculate_rating_trends(df)
            genre_trends = self.analyze_genre_trends(df)
            seasonal_trends = self.detect_seasonal_patterns(df)
            social_buzz = self.calculate_social_buzz(df)

            trend_scores = {}
            all_content_ids = set(df['content_id'].unique())

            for content_id in all_content_ids:
                score = (
                    popularity_momentum.get(content_id, 0) * self.trend_weights['popularity_momentum'] +
                    rating_trends.get(content_id, 0) * self.trend_weights['rating_trend'] +
                    genre_trends.get(content_id, 0) * self.trend_weights['genre_trend'] +
                    seasonal_trends.get(content_id, 0) * self.trend_weights['seasonal_trend'] +
                    social_buzz.get(content_id, 0) * self.trend_weights['social_buzz']
                )
                trend_scores[content_id] = score

            return trend_scores
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return {}

    def calculate_rating_trends(self, df):
        rating_trends = {}
        for content_id in df['content_id'].unique():
            content_ratings = df[df['content_id'] == content_id]['rating'].dropna()
            if len(content_ratings) > 3:
                x = np.arange(len(content_ratings))
                slope, _ = np.polyfit(x, content_ratings, 1)
                rating_trends[content_id] = slope
            else:
                rating_trends[content_id] = 0
        return rating_trends

    def analyze_genre_trends(self, df):
        genre_trends = {}
        current_time = datetime.now()

        for _, row in df.iterrows():
            genres = json.loads(row.get('genre_ids', '[]'))
            days_since_interaction = (current_time - pd.to_datetime(row['created_at'])).days
            time_weight = max(0, 1 - (days_since_interaction / 30))

            for genre in genres:
                genre_trends[genre] = genre_trends.get(genre, 0) + time_weight

        content_genre_scores = {}
        for _, row in df.iterrows():
            content_id = row['content_id']
            genres = json.loads(row.get('genre_ids', '[]'))
            genre_score = sum(genre_trends.get(genre, 0) for genre in genres)
            content_genre_scores[content_id] = genre_score / len(genres) if genres else 0

        return content_genre_scores

    def detect_seasonal_patterns(self, df):
        seasonal_scores = {}
        current_month = datetime.now().month

        seasonal_genres = {
            'winter': [18, 10751, 10402],  # Drama, Family, Music
            'spring': [35, 10749, 16],    # Comedy, Romance, Animation
            'summer': [28, 12, 53],       # Action, Adventure, Thriller
            'fall': [18, 80, 99]          # Drama, Crime, Documentary
        }

        season_map = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        }

        current_season = season_map.get(current_month, 'spring')
        preferred_genres = seasonal_genres.get(current_season, [])

        for _, row in df.iterrows():
            content_id = row['content_id']
            genres = json.loads(row.get('genre_ids', '[]'))
            seasonal_match = len(set(genres) & set(preferred_genres))
            seasonal_scores[content_id] = seasonal_match / len(genres) if genres else 0

        return seasonal_scores

    def calculate_social_buzz(self, df):
        social_scores = {}
        for content_id in df['content_id'].unique():
            social_scores[content_id] = np.random.uniform(0, 1)  # Placeholder
        return social_scores

# [DiversityOptimizer class remains unchanged]
class DiversityOptimizer:
    """Optimize recommendation diversity while maintaining relevance"""
    def __init__(self):
        self.diversity_weights = {
            'genre_diversity': 0.35,
            'content_type_diversity': 0.25,
            'release_date_diversity': 0.2,
            'popularity_diversity': 0.15,
            'cultural_diversity': 0.05
        }

    def optimize_diversity(self, recommendations, target_diversity=0.75):
        if not recommendations:
            return recommendations

        current_diversity = self.calculate_diversity_score(recommendations)
        if current_diversity >= target_diversity:
            return recommendations

        return self.apply_diversity_algorithm(recommendations, target_diversity)

    def calculate_diversity_score(self, recommendations):
        if not recommendations:
            return 0

        scores = {
            'genre_diversity': self.calculate_genre_diversity(recommendations),
            'content_type_diversity': self.calculate_content_type_diversity(recommendations),
            'release_date_diversity': self.calculate_temporal_diversity(recommendations),
            'popularity_diversity': self.calculate_popularity_diversity(recommendations),
            'cultural_diversity': self.calculate_cultural_diversity(recommendations)
        }

        total_score = sum(scores[key] * self.diversity_weights[key] for key in scores)
        return total_score

    def calculate_genre_diversity(self, recommendations):
        genre_counts = {}
        total_genres = 0

        for rec in recommendations:
            genres = json.loads(rec.get('genre_ids', '[]'))
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                total_genres += 1

        if total_genres == 0:
            return 0

        entropy = 0
        for count in genre_counts.values():
            p = count / total_genres
            entropy -= p * np.log2(p) if p > 0 else 0

        max_entropy = np.log2(len(genre_counts)) if genre_counts else 1
        return entropy / max_entropy if max_entropy > 0 else 0

    def calculate_content_type_diversity(self, recommendations):
        content_types = [rec.get('content_type') for rec in recommendations]
        unique_types = len(set(content_types))
        total_types = len(content_types)
        return unique_types / total_types if total_types > 0 else 0

    def calculate_temporal_diversity(self, recommendations):
        release_years = []
        for rec in recommendations:
            date_str = rec.get('release_date')
            if date_str:
                try:
                    year = pd.to_datetime(date_str).year
                    release_years.append(year)
                except:
                    continue

        if not release_years:
            return 0

        year_span = max(release_years) - min(release_years)
        return min(1.0, year_span / 50)

    def calculate_popularity_diversity(self, recommendations):
        popularities = [rec.get('popularity', 0) for rec in recommendations]
        if not popularities:
            return 0

        mean_pop = np.mean(popularities)
        std_pop = np.std(popularities)
        return min(1.0, std_pop / mean_pop) if mean_pop > 0 else 0

    def calculate_cultural_diversity(self, recommendations):
        return np.random.uniform(0, 1)  # Placeholder

    def apply_diversity_algorithm(self, recommendations, target_diversity):
        sorted_recs = sorted(recommendations, key=lambda x: x.get('recommendation_score', 0), reverse=True)
        optimized_list = []
        candidate_pool = sorted_recs.copy()

        while len(optimized_list) < len(recommendations) and candidate_pool:
            best_candidate = None
            best_score = -1

            for candidate in candidate_pool:
                test_list = optimized_list + [candidate]
                diversity_score = self.calculate_diversity_score(test_list)
                relevance_score = candidate.get('recommendation_score', 0)
                combined_score = diversity_score * 0.7 + (relevance_score / 10) * 0.3

                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate

            if best_candidate:
                optimized_list.append(best_candidate)
                candidate_pool.remove(best_candidate)

        return optimized_list

# [AdvancedNeuralRecommender class remains unchanged]
class AdvancedNeuralRecommender(nn.Module):
    def __init__(self, num_users, num_items, input_dim=100, latent_dim=64):
        super(AdvancedNeuralRecommender, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.vae = self.build_vae()
        self.transformer = self.build_transformer()
        self.gnn_embeddings = None

    def build_vae(self):
        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super(VAE, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                self.z_mean = nn.Linear(128, latent_dim)
                self.z_log_var = nn.Linear(128, latent_dim)
                self.sampling = Sampling()
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, input_dim),
                    nn.Sigmoid()
                )

            def forward(self, x):
                h = self.encoder(x)
                z_mean = self.z_mean(h)
                z_log_var = self.z_log_var(h)
                z = self.sampling(z_mean, z_log_var)
                reconstructed = self.decoder(z)
                return reconstructed, z_mean, z_log_var

        return VAE(self.input_dim, self.latent_dim)

    def build_transformer(self):
        class Transformer(nn.Module):
            def __init__(self, num_users, num_items):
                super(Transformer, self).__init__()
                self.user_embedding = nn.Embedding(num_users, 64)
                self.item_embedding = nn.Embedding(num_items, 64)
                self.attention = nn.Linear(128, 64)
                self.attention_softmax = nn.Softmax(dim=-1)
                self.dense1 = nn.Linear(128, 128)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                self.dense2 = nn.Linear(128, 64)
                self.output = nn.Linear(64, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, user_ids, item_ids):
                user_vec = self.user_embedding(user_ids)
                item_vec = self.item_embedding(item_ids)
                concat = torch.cat([user_vec, item_vec], dim=-1)
                attention_weights = self.attention_softmax(self.attention(concat))
                attended = concat * attention_weights
                x = self.dense1(attended)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.dense2(x)
                x = self.relu(x)
                x = self.output(x)
                return self.sigmoid(x)

        return Transformer(self.num_users, self.num_items)

    def build_gnn(self, interactions):
        G = nx.Graph()
        for user_id, item_id, rating in interactions:
            G.add_edge(f"user_{user_id}", f"item_{item_id}", weight=rating)

        num_nodes = len(G.nodes())
        embeddings = np.random.normal(0, 0.1, (num_nodes, 64))
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}

        for _ in range(3):
            new_embeddings = np.zeros_like(embeddings)
            for node, idx in node_to_idx.items():
                neighbors = list(G.neighbors(node))
                if neighbors:
                    neighbor_indices = [node_to_idx[n] for n in neighbors]
                    neighbor_embeddings = embeddings[neighbor_indices]
                    new_embeddings[idx] = np.mean(neighbor_embeddings, axis=0)
                else:
                    new_embeddings[idx] = embeddings[idx]
            embeddings = new_embeddings

        self.gnn_embeddings = {node: emb for node, emb in zip(G.nodes(), embeddings)}
        return self.gnn_embeddings

    def train_vae(self, data, epochs=50, batch_size=32):
        optimizer = optim.Adam(self.vae.parameters(), lr=0.001)
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae.to(device)

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                reconstructed, z_mean, z_log_var = self.vae(x)
                recon_loss = nn.BCELoss(reduction='sum')(reconstructed, x)
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
                loss = recon_loss + kl_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"VAE Epoch {epoch+1}, Loss: {total_loss/len(dataloader.dataset):.4f}")

    def train_transformer(self, user_ids, item_ids, ratings, epochs=50, batch_size=32):
        optimizer = optim.Adam(self.transformer.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(user_ids),
            torch.LongTensor(item_ids),
            torch.FloatTensor(ratings)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer.to(device)

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                user_ids_batch, item_ids_batch, ratings_batch = [b.to(device) for b in batch]
                optimizer.zero_grad()
                predictions = self.transformer(user_ids_batch, item_ids_batch).squeeze()
                loss = criterion(predictions, ratings_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Transformer Epoch {epoch+1}, Loss: {total_loss/len(dataloader.dataset):.4f}")

# [ReinforcementLearningRecommender class remains unchanged]
class ReinforcementLearningRecommender:
    def __init__(self, action_space):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.action_space = action_space

    def get_state(self, user_profile, context):
        """Get current state representation with default values for missing context fields"""
        state = {
            'user_genre_preferences': user_profile.get('genre_preferences', {}),
            'time_of_day': context.get('time_of_day', 'unknown'),
            'device_type': context.get('device_type', 'unknown'),
            'session_length': context.get('session_length', 0)
        }
        return json.dumps(state, sort_keys=True)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            if state in self.q_table:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                return np.random.choice(self.action_space)

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0

        max_next_q = max(self.q_table.get(next_state, {}).values()) if next_state in self.q_table else 0
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

class AdvancedRecommendationEngine:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.content_features = None
        self.user_item_matrix = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.svd = TruncatedSVD(n_components=100, random_state=42)
        self.scaler = StandardScaler()
        self.content_similarity = None
        self.user_similarity = None
        self.item_similarity = None
        self.model_trained = False
        self.model_timestamp = None
        self.neural_recommender = None
        self.rl_recommender = None
        self.trend_analyzer = TrendAnalyzer()
        self.diversity_optimizer = DiversityOptimizer()
        self.surprise_models = {}
        self.implicit_model = None
        self.faiss_index = None
        self.knowledge_graph = None
        self.ensemble_models = {}
        self.online_learning_buffer = deque(maxlen=50000)
        self.feedback_weights = {'click': 0.1, 'rating': 1.0, 'watchlist': 0.3, 'favorite': 0.8}
        self.cold_start_strategies = {'popularity': 0.6, 'trending': 0.3, 'random': 0.1}
        self.user_clusters = None
        self.item_clusters = None

    def get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def extract_content_features(self):
        conn = self.get_db_connection()
        content_data = conn.execute('''
            SELECT id, title, overview, genre_ids, vote_average, popularity,
                   content_type, release_date, runtime
            FROM content
        ''').fetchall()

        if not content_data:
            logger.warning("No content data found")
            conn.close()
            return None

        df = pd.DataFrame([dict(row) for row in content_data])
        text_features = df['title'].fillna('') + ' ' + df['overview'].fillna('')
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)

        numerical_features = []
        for _, row in df.iterrows():
            try:
                release_year = pd.to_datetime(row['release_date']).year if row['release_date'] else 2000
            except:
                release_year = 2000
            features = [
                row['vote_average'] or 0,
                np.log1p(row['popularity'] or 0),
                len(json.loads(row['genre_ids'] or '[]')),
                1 if row['content_type'] == 'movie' else 0,
                1 if row['content_type'] == 'tv' else 0,
                1 if row['content_type'] == 'anime' else 0,
                row['runtime'] or 0,
                release_year
            ]
            numerical_features.append(features)

        numerical_features = np.array(numerical_features)
        numerical_features = self.scaler.fit_transform(numerical_features)
        content_features = np.hstack([tfidf_matrix.toarray(), numerical_features])
        content_features = self.svd.fit_transform(content_features)

        self.content_features = content_features
        self.content_ids = df['id'].values
        conn.close()
        return content_features

    def build_user_item_matrix(self):
        conn = self.get_db_connection()
        interactions = conn.execute('''
            SELECT user_id, content_id, interaction_type, rating
            FROM user_interactions
            WHERE interaction_type IN ('rating', 'watchlist', 'favorite')
        ''').fetchall()

        if not interactions:
            logger.warning("No user interactions found")
            conn.close()
            return None

        interaction_data = []
        for interaction in interactions:
            weight = self.feedback_weights.get(interaction['interaction_type'], 0.1)
            if interaction['interaction_type'] == 'rating':
                weight = (interaction['rating'] or 0) / 10.0
            interaction_data.append({
                'user_id': interaction['user_id'],
                'content_id': interaction['content_id'],
                'weight': weight
            })

        df = pd.DataFrame(interaction_data)
        user_item_matrix = df.pivot_table(index='user_id', columns='content_id', values='weight', fill_value=0)
        self.user_item_matrix = user_item_matrix
        self.user_ids = user_item_matrix.index.values
        self.item_ids = user_item_matrix.columns.values
        conn.close()
        return user_item_matrix

    def compute_similarity_matrices(self):
        if self.content_features is not None:
            self.content_similarity = cosine_similarity(self.content_features)
            logger.info("Content similarity matrix computed")

        if self.user_item_matrix is not None:
            self.user_similarity = cosine_similarity(self.user_item_matrix.values)
            self.item_similarity = cosine_similarity(self.user_item_matrix.T.values)
            logger.info("User and item similarity matrices computed")

    def train_neural_recommender(self):
        conn = self.get_db_connection()
        max_user_id = conn.execute('SELECT MAX(user_id) FROM user_interactions').fetchone()[0] or 1000
        max_content_id = conn.execute('SELECT MAX(content_id) FROM user_interactions').fetchone()[0] or 10000
        conn.close()

        input_dim = self.content_features.shape[1] if self.content_features is not None else 100
        self.neural_recommender = AdvancedNeuralRecommender(max_user_id + 1, max_content_id + 1, input_dim)
        interactions = [(row['user_id'], row['content_id'], row['rating'] or 5) for row in
                        self.get_db_connection().execute('SELECT * FROM user_interactions WHERE rating IS NOT NULL').fetchall()]

        if self.content_features is not None:
            self.neural_recommender.train_vae(self.content_features)

        user_ids, item_ids, ratings = zip(*interactions)
        ratings = np.array(ratings) / 10.0
        self.neural_recommender.train_transformer(user_ids, item_ids, ratings)
        self.neural_recommender.build_gnn(interactions)

    def train_surprise_models(self):
        """Train Surprise library models if available"""
        if not SURPRISE_AVAILABLE:
            logger.warning("Surprise library not available. Skipping collaborative filtering models.")
            self.surprise_models = {}
            return False

        conn = self.get_db_connection()
        ratings_data = conn.execute('''
            SELECT user_id, content_id, rating
            FROM user_interactions
            WHERE interaction_type = 'rating' AND rating IS NOT NULL
        ''').fetchall()

        if not ratings_data:
            conn.close()
            logger.warning("No ratings data available for Surprise models")
            return False

        df = pd.DataFrame(ratings_data)
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(df[['user_id', 'content_id', 'rating']], reader)

        algorithms = {
            'SVD': SVD(n_factors=100, reg_all=0.02),
            'NMF': NMF(n_factors=50, reg_pu=0.06, reg_qi=0.06),
            'KNN_User': KNNBasic(k=40, sim_options={'user_based': True}),
            'KNN_Item': KNNBasic(k=40, sim_options={'user_based': False})
        }

        for name, algo in algorithms.items():
            try:
                cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
                trainset = data.build_full_trainset()
                algo.fit(trainset)
                self.surprise_models[name] = {
                    'model': algo,
                    'rmse': np.mean(cv_results['test_rmse']),
                    'mae': np.mean(cv_results['test_mae'])
                }
                logger.info(f"Trained {name} - RMSE: {np.mean(cv_results['test_rmse']):.4f}")
            except Exception as e:
                logger.error(f"Failed to train {name} model: {e}")
                continue

        conn.close()
        return bool(self.surprise_models)

    def train_implicit_model(self):
        conn = self.get_db_connection()
        interactions = conn.execute('''
            SELECT user_id, content_id,
                   CASE interaction_type
                       WHEN 'rating' THEN rating
                       WHEN 'favorite' THEN 10
                       WHEN 'watchlist' THEN 5
                       ELSE 1
                   END as confidence
            FROM user_interactions
        ''').fetchall()

        if not interactions:
            conn.close()
            return False

        df = pd.DataFrame(interactions)
        user_ids = df['user_id'].unique()
        item_ids = df['content_id'].unique()

        user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
        item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}

        rows = [user_id_map[uid] for uid in df['user_id']]
        cols = [item_id_map[iid] for iid in df['content_id']]
        data = df['confidence'].values

        interaction_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
        model = implicit.als.AlternatingLeastSquares(factors=64, iterations=20, regularization=0.1)
        model.fit(interaction_matrix)

        self.implicit_model = {
            'model': model,
            'user_id_map': user_id_map,
            'item_id_map': item_id_map,
            'interaction_matrix': interaction_matrix
        }
        conn.close()
        logger.info("Implicit ALS model trained successfully")
        return True

    def build_faiss_index(self):
        if self.content_features is None:
            return False

        dimension = self.content_features.shape[1]
        index = faiss.IndexFlatIP(dimension)
        normalized_features = self.content_features / np.linalg.norm(self.content_features, axis=1, keepdims=True)
        index.add(normalized_features.astype('float32'))
        self.faiss_index = index
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        return True

    def train_ensemble_models(self):
        X, y = self.prepare_training_data()
        if X is None or len(X) == 0:
            return False

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(128, 64, 32), random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            self.ensemble_models[name] = {'model': model, 'score': score}
            logger.info(f"Trained {name} with score: {score:.4f}")

        return True

    def prepare_training_data(self):
        conn = self.get_db_connection()
        data = conn.execute('''
            SELECT
                ui.user_id,
                ui.content_id,
                ui.rating,
                c.vote_average,
                c.popularity,
                c.content_type,
                c.genre_ids,
                u.created_at as user_created,
                ui.created_at as interaction_created
            FROM user_interactions ui
            JOIN content c ON ui.content_id = c.id
            JOIN users u ON ui.user_id = u.id
            WHERE ui.interaction_type = 'rating' AND ui.rating IS NOT NULL
        ''').fetchall()

        if not data:
            conn.close()
            return None, None

        X = []
        y = []
        for row in data:
            try:
                interaction_age = (datetime.now() - datetime.fromisoformat(row['interaction_created'])).days
                user_age = (datetime.now() - datetime.fromisoformat(row['user_created'])).days
            except:
                interaction_age = user_age = 0

            features = [
                row['user_id'],
                row['content_id'],
                row['vote_average'] or 0,
                np.log1p(row['popularity'] or 0),
                1 if row['content_type'] == 'movie' else 0,
                1 if row['content_type'] == 'tv' else 0,
                1 if row['content_type'] == 'anime' else 0,
                len(json.loads(row['genre_ids'] or '[]')),
                interaction_age,
                user_age
            ]
            X.append(features)
            y.append(row['rating'])

        conn.close()
        return np.array(X), np.array(y)

    def build_knowledge_graph(self):
        conn = self.get_db_connection()
        content_data = conn.execute('''
            SELECT c.*, GROUP_CONCAT(DISTINCT ui.user_id) as user_interactions
            FROM content c
            LEFT JOIN user_interactions ui ON c.id = ui.content_id
            GROUP BY c.id
        ''').fetchall()

        G = nx.Graph()
        for content in content_data:
            content_id = content['id']
            G.add_node(content_id,
                       content_type=content['content_type'],
                       genres=json.loads(content['genre_ids'] or '[]'),
                       rating=content['vote_average'],
                       popularity=content['popularity'])

            genres = set(json.loads(content['genre_ids'] or '[]'))
            for other_content in content_data:
                if other_content['id'] != content_id:
                    other_genres = set(json.loads(other_content['genre_ids'] or '[]'))
                    if genres and other_genres:
                        similarity = len(genres & other_genres) / len(genres | other_genres)
                        if similarity > 0.3:
                            G.add_edge(content_id, other_content['id'], weight=similarity)

        self.knowledge_graph = G
        conn.close()
        logger.info(f"Knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return True

    def train_model(self):
        try:
            logger.info("Training recommendation models...")
            self.extract_content_features()
            self.build_user_item_matrix()
            self.compute_similarity_matrices()
            self.train_neural_recommender()
            success = self.train_surprise_models()
            if not success:
                logger.warning("Surprise models not trained; relying on other models")
            self.train_implicit_model()
            self.train_ensemble_models()
            self.build_faiss_index()
            self.build_knowledge_graph()

            conn = self.get_db_connection()
            content_data = conn.execute('SELECT id, content_type, genre_ids FROM content').fetchall()
            conn.close()
            self.rl_recommender = ReinforcementLearningRecommender([f"recommend_{row['content_type']}_{row['genre_ids']}" for row in content_data])
            self.model_trained = True
            self.model_timestamp = datetime.now()
            logger.info("Model training completed successfully")
            return True
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def get_content_based_recommendations(self, user_id, limit=20):
        if not self.model_trained or self.content_features is None:
            return self.get_cold_start_recommendations(limit)

        conn = self.get_db_connection()
        try:
            user_ratings = conn.execute('''
                SELECT content_id, rating FROM user_interactions
                WHERE user_id = ? AND interaction_type = 'rating'
                ORDER BY rating DESC
            ''', (user_id,)).fetchall()

            if not user_ratings:
                return self.get_cold_start_recommendations(limit)

            recommendations = {}
            for rating in user_ratings[:10]:
                content_id = rating['content_id']
                user_rating = rating['rating']
                try:
                    content_idx = np.where(self.content_ids == content_id)[0][0]
                    similarities = self.content_similarity[content_idx]
                    similar_indices = np.argsort(similarities)[::-1][1:limit+1]
                    for idx in similar_indices:
                        similar_content_id = self.content_ids[idx]
                        score = similarities[idx] * (user_rating / 10.0)
                        recommendations[similar_content_id] = recommendations.get(similar_content_id, 0) + score
                except IndexError:
                    continue

            top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:limit]
            content_details = []
            for content_id, score in top_recommendations:
                content = conn.execute('SELECT * FROM content WHERE id = ?', (content_id,)).fetchone()
                if content:
                    content_dict = dict(content)
                    content_dict['recommendation_score'] = score
                    content_details.append(content_dict)

            return self.diversity_optimizer.optimize_diversity(content_details)
        finally:
            conn.close()

    def get_collaborative_recommendations(self, user_id, limit=20):
        if not self.model_trained or self.user_item_matrix is None or not self.surprise_models:
            logger.warning("Collaborative filtering unavailable; using cold-start recommendations")
            return self.get_cold_start_recommendations(limit)

        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
            user_similarities = self.user_similarity[user_idx]
            similar_users = np.argsort(user_similarities)[::-1][1:11]
            recommendations = {}
            user_items = set(self.user_item_matrix.iloc[user_idx].nonzero()[0])

            for similar_user_idx in similar_users:
                similarity_score = user_similarities[similar_user_idx]
                similar_user_items = self.user_item_matrix.iloc[similar_user_idx]
                for item_idx, rating in similar_user_items.items():
                    if rating > 0 and item_idx not in user_items:
                        item_id = self.item_ids[item_idx]
                        weighted_score = similarity_score * rating
                        recommendations[item_id] = recommendations.get(item_id, 0) + weighted_score

            top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:limit]
            conn = self.get_db_connection()
            content_details = []
            for content_id, score in top_recommendations:
                content = conn.execute('SELECT * FROM content WHERE id = ?', (content_id,)).fetchone()
                if content:
                    content_dict = dict(content)
                    content_dict['recommendation_score'] = score
                    content_details.append(content_dict)

            conn.close()
            return self.diversity_optimizer.optimize_diversity(content_details)
        except IndexError:
            return self.get_cold_start_recommendations(limit)

    def get_deep_learning_recommendations(self, user_id, limit=20):
        try:
            recommendations = []
            if self.neural_recommender and self.neural_recommender.transformer:
                conn = self.get_db_connection()
                items = conn.execute('SELECT id FROM content LIMIT ?', (limit * 2,)).fetchall()
                item_ids = [item['id'] for item in items]
                user_ids = [user_id] * len(item_ids)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                user_ids_tensor = torch.LongTensor(user_ids).to(device)
                item_ids_tensor = torch.LongTensor(item_ids).to(device)
                self.neural_recommender.transformer.eval()
                with torch.no_grad():
                    predictions = self.neural_recommender.transformer(user_ids_tensor, item_ids_tensor).cpu().numpy()
                scores = predictions.flatten()
                top_indices = np.argsort(scores)[::-1][:limit]
                for idx in top_indices:
                    content = conn.execute('SELECT * FROM content WHERE id = ?', (item_ids[idx],)).fetchone()
                    if content:
                        content_dict = dict(content)
                        content_dict['recommendation_score'] = float(scores[idx])
                        recommendations.append(content_dict)
                conn.close()

            return self.diversity_optimizer.optimize_diversity(recommendations)
        except Exception as e:
            logger.error(f"Deep learning recommendations error: {e}")
            return self.get_hybrid_recommendations(user_id, limit)

    def get_hybrid_recommendations(self, user_id, limit=20):
        content_recs = self.get_content_based_recommendations(user_id, limit)
        collab_recs = self.get_collaborative_recommendations(user_id, limit)
        deep_recs = self.get_deep_learning_recommendations(user_id, limit)

        combined_recs = {}
        for rec in content_recs:
            combined_recs[rec['id']] = {'content': rec, 'score': rec.get('recommendation_score', 0) * 0.4}
        for rec in collab_recs:
            if rec['id'] in combined_recs:
                combined_recs[rec['id']]['score'] += rec.get('recommendation_score', 0) * 0.3
            else:
                combined_recs[rec['id']] = {'content': rec, 'score': rec.get('recommendation_score', 0) * 0.3}
        for rec in deep_recs:
            if rec['id'] in combined_recs:
                combined_recs[rec['id']]['score'] += rec.get('recommendation_score', 0) * 0.3
            else:
                combined_recs[rec['id']] = {'content': rec, 'score': rec.get('recommendation_score', 0) * 0.3}

        sorted_recs = sorted(combined_recs.items(), key=lambda x: x[1]['score'], reverse=True)
        final_recs = [rec_data['content'] for _, rec_data in sorted_recs[:limit]]
        return self.diversity_optimizer.optimize_diversity(final_recs)

    def get_cold_start_recommendations(self, limit=20):
        conn = self.get_db_connection()
        content_data = conn.execute('''
            SELECT *, (popularity * 0.7 + vote_average * 0.3) as score
            FROM content
            WHERE created_at > datetime('now', '-30 days')
            ORDER BY score DESC
            LIMIT ?
        ''', (limit,)).fetchall()
        conn.close()
        return [dict(row) | {'recommendation_score': row['score']} for row in content_data]

    def get_rl_recommendations(self, user_id, context, limit=20):
        conn = self.get_db_connection()
        user_profile = conn.execute('''
            SELECT genre_ids, AVG(rating) as avg_rating
            FROM content c
            JOIN user_interactions ui ON c.id = ui.content_id
            WHERE ui.user_id = ? AND ui.interaction_type = 'rating'
            GROUP BY genre_ids
        ''', (user_id,)).fetchall()
        conn.close()

        genre_preferences = {}
        for row in user_profile:
            genres = json.loads(row['genre_ids'] or '[]')
            for genre in genres:
                genre_preferences[genre] = genre_preferences.get(genre, 0) + row['avg_rating']

        user_profile_dict = {'genre_preferences': genre_preferences}
        state = self.rl_recommender.get_state(user_profile_dict, context)
        action = self.rl_recommender.choose_action(state)

        conn = self.get_db_connection()
        content_data = conn.execute('SELECT * FROM content LIMIT ?', (limit,)).fetchall()
        recommendations = [dict(row) | {'recommendation_score': 1.0} for row in content_data if f"recommend_{row['content_type']}_{row['genre_ids']}" == action]
        conn.close()

        return self.diversity_optimizer.optimize_diversity(recommendations[:limit])

    def update_online_learning(self, user_id, content_id, action, context):
        reward = self.feedback_weights.get(action, 0.1)
        user_profile = self.get_user_profile(user_id)
        state = self.rl_recommender.get_state(user_profile, context)
        action_id = f"recommend_content_{content_id}"
        next_state = state

        self.rl_recommender.update_q_table(state, action_id, reward, next_state)
        self.online_learning_buffer.append({
            'user_id': user_id,
            'content_id': content_id,
            'action': action,
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        })

        if len(self.online_learning_buffer) >= 1000:
            self.retrain_online_models()

    def retrain_online_models(self):
        try:
            X, y = [], []
            for interaction in self.online_learning_buffer:
                user_id = interaction['user_id']
                content_id = interaction['content_id']
                reward = interaction['reward']
                conn = self.get_db_connection()
                content = conn.execute('SELECT * FROM content WHERE id = ?', (content_id,)).fetchone()
                if content:
                    features = [
                        user_id,
                        content_id,
                        content['vote_average'] or 0,
                        np.log1p(content['popularity'] or 0),
                        len(json.loads(content['genre_ids'] or '[]'))
                    ]
                    X.append(features)
                    y.append(reward)
                conn.close()

            if X:
                X_array, y_array = np.array(X), np.array(y)
                if 'neural_network' in self.ensemble_models:
                    try:
                        self.ensemble_models['neural_network']['model'].partial_fit(X_array, y_array)
                    except:
                        pass
                logger.info("Online models updated with new interactions")
        except Exception as e:
            logger.error(f"Online model retraining error: {e}")

    def get_user_profile(self, user_id):
        conn = self.get_db_connection()
        genre_stats = conn.execute('''
            SELECT genre_ids, AVG(rating) as avg_rating
            FROM content c
            JOIN user_interactions ui ON c.id = ui.content_id
            WHERE ui.user_id = ? AND ui.interaction_type = 'rating'
            GROUP BY genre_ids
        ''', (user_id,)).fetchall()
        conn.close()

        genre_preferences = {}
        for row in genre_stats:
            genres = json.loads(row['genre_ids'] or '[]')
            for genre in genres:
                genre_preferences[genre] = genre_preferences.get(genre, 0) + row['avg_rating']
        return {'genre_preferences': genre_preferences}

# Global recommendation engine instance
rec_engine = AdvancedRecommendationEngine()

def get_cached_recommendations(user_id, rec_type='hybrid', limit=20):
    cache_key = f"{user_id}_{rec_type}_{limit}"
    if redis_client:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            cached_data = json.loads(cached_data)
            if datetime.now() - datetime.fromisoformat(cached_data['timestamp']) < timedelta(seconds=CACHE_DURATION):
                return cached_data['recommendations']

    recommendations = []
    if rec_type in ('hybrid', 'hybrid_advanced'):  # Handle backend's hybrid_advanced
        recommendations = rec_engine.get_hybrid_recommendations(user_id, limit)
    elif rec_type == 'content':
        recommendations = rec_engine.get_content_based_recommendations(user_id, limit)
    elif rec_type == 'collaborative':
        recommendations = rec_engine.get_collaborative_recommendations(user_id, limit)
    elif rec_type == 'deep_learning':
        recommendations = rec_engine.get_deep_learning_recommendations(user_id, limit)
    elif rec_type == 'reinforcement':
        context = request.get_json().get('context', {}) if request.is_json else {}
        recommendations = rec_engine.get_rl_recommendations(user_id, context, limit)

    if redis_client:
        redis_client.setex(cache_key, CACHE_DURATION, json.dumps({
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }))
    else:
        with cache_lock:
            recommendation_cache[cache_key] = {
                'recommendations': recommendations,
                'timestamp': datetime.now()
            }

    return recommendations

def retrain_model_if_needed():
    if not rec_engine.model_trained or not rec_engine.model_timestamp:
        return rec_engine.train_model()
    if datetime.now() - rec_engine.model_timestamp > timedelta(hours=24):
        logger.info("Model is outdated, retraining...")
        return rec_engine.train_model()
    return True

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_trained': rec_engine.model_trained,
        'model_timestamp': rec_engine.model_timestamp.isoformat() if rec_engine.model_timestamp else None,
        'cache_size': len(recommendation_cache) if not redis_client else redis_client.dbsize(),
        'surprise_available': SURPRISE_AVAILABLE,  # Added for debugging
        'timestamp': datetime.now().isoformat()
    })

@app.route('/train', methods=['POST'])
def train_model():
    try:
        success = rec_engine.train_model()
        if success:
            if redis_client:
                redis_client.flushdb()
            else:
                with cache_lock:
                    recommendation_cache.clear()
            return jsonify({'message': 'Model trained successfully'})
        return jsonify({'error': 'Model training failed'}), 500
    except Exception as e:
        logger.error(f"Training endpoint error: {e}")
        return jsonify({'error': 'Training failed'}), 500

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        limit = min(data.get('limit', 20), 100)
        rec_type = data.get('type', data.get('algorithm', 'hybrid'))  # Support both 'type' and 'algorithm'

        if not user_id:
            return jsonify({'error': 'User ID required'}), 400

        if not retrain_model_if_needed():
            return jsonify({'error': 'Model training failed'}), 500

        recommendations = get_cached_recommendations(user_id, rec_type, limit)
        return jsonify({
            'recommendations': recommendations,
            'user_id': user_id,
            'type': rec_type,
            'count': len(recommendations),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Recommendation endpoint error: {e}")
        return jsonify({'error': 'Recommendation service unavailable'}), 500

@app.route('/similar', methods=['POST'])
def get_similar_content():
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        user_id = data.get('user_id')
        if not content_id:
            return jsonify({'error': 'Content ID required'}), 400

        if not rec_engine.model_trained or rec_engine.faiss_index is None:
            return jsonify({'error': 'Model not trained'}), 503

        try:
            content_idx = np.where(rec_engine.content_ids == content_id)[0][0]
        except IndexError:
            return jsonify({'error': 'Content not found'}), 404

        content_vector = rec_engine.content_features[content_idx:content_idx+1]
        distances, indices = rec_engine.faiss_index.search(content_vector.astype('float32'), 21)

        conn = rec_engine.get_db_connection()
        try:
            similar_content = []
            for i, idx in enumerate(indices[0][1:]):
                similar_content_id = rec_engine.content_ids[idx]
                content = conn.execute('SELECT * FROM content WHERE id = ?', (similar_content_id,)).fetchone()
                if content:
                    content_dict = dict(content)
                    content_dict['similarity_score'] = float(distances[0][i+1])
                    similar_content.append(content_dict)

            return jsonify({
                'similar': similar_content,
                'content_id': content_id,
                'count': len(similar_content)
            })
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Similar content endpoint error: {e}")
        return jsonify({'error': 'Similar content service unavailable'}), 500

@app.route('/search_suggestions', methods=['POST'])
def get_search_suggestions():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        query = data.get('query', '')
        limit = min(data.get('limit', 5), 20)

        if not user_id or not query:
            return jsonify({'error': 'User ID and query required'}), 400

        if not retrain_model_if_needed():
            return jsonify({'error': 'Model not trained'}), 503

        query_vector = rec_engine.tfidf_vectorizer.transform([query]).toarray()
        query_embedding = rec_engine.svd.transform(query_vector)
        user_profile = rec_engine.get_user_profile(user_id)
        genre_preferences = user_profile.get('genre_preferences', {})
        similarities = cosine_similarity(query_embedding, rec_engine.content_features)[0]
        recommendations = []
        conn = rec_engine.get_db_connection()
        try:
            for idx in np.argsort(similarities)[::-1][:limit * 2]:
                content_id = rec_engine.content_ids[idx]
                content = conn.execute('SELECT * FROM content WHERE id = ?', (content_id,)).fetchone()
                if content:
                    content_dict = dict(content)
                    genres = json.loads(content['genre_ids'] or '[]')
                    genre_score = sum(genre_preferences.get(str(genre), 0) for genre in genres) / (len(genres) or 1)
                    content_dict['suggestion_score'] = similarities[idx] * 0.7 + genre_score * 0.3
                    recommendations.append(content_dict)

            recommendations.sort(key=lambda x: x['suggestion_score'], reverse=True)
            recommendations = rec_engine.diversity_optimizer.optimize_diversity(recommendations[:limit])
            return jsonify({
                'suggestions': recommendations,
                'count': len(recommendations)
            })
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        return jsonify({'error': 'Search suggestions unavailable'}), 500

@app.route('/update', methods=['POST'])
def trigger_model_update():
    try:
        data = request.get_json()
        event_type = data.get('event')
        event_data = data.get('data', {})

        if not event_type:
            return jsonify({'error': 'Event type required'}), 400

        if event_type == 'public_recommendation_added':
            content_id = event_data.get('content_id')
            if content_id:
                conn = rec_engine.get_db_connection()
                conn.execute('UPDATE content SET popularity = popularity * 1.2 WHERE id = ?', (content_id,))
                conn.commit()
                conn.close()
                logger.info(f"Boosted popularity for content_id {content_id} due to public recommendation")

        rec_engine.online_learning_buffer.append({
            'event': event_type,
            'data': event_data,
            'timestamp': datetime.now().isoformat()
        })

        return jsonify({'message': 'Model update triggered'})
    except Exception as e:
        logger.error(f"Model update error: {e}")
        return jsonify({'error': 'Failed to trigger model update'}), 500

@app.route('/update_preferences', methods=['POST'])
def update_preferences():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        interactions = data.get('data', [])

        if not user_id or not interactions:
            return jsonify({'error': 'User ID and interaction data required'}), 400

        for interaction in interactions:
            content_id = interaction.get('content_id')
            action = interaction.get('interaction_type')
            rating = interaction.get('rating')
            if content_id and action:
                reward = rec_engine.feedback_weights.get(action, 0.1)
                if action == 'rating' and rating:
                    reward = rating / 10.0
                rec_engine.online_learning_buffer.append({
                    'user_id': user_id,
                    'content_id': content_id,
                    'action': action,
                    'reward': reward,
                    'timestamp': datetime.now().isoformat()
                })

        if len(rec_engine.online_learning_buffer) >= 1000:
            rec_engine.retrain_online_models()

        return jsonify({'message': 'Preferences update queued'})
    except Exception as e:
        logger.error(f"Preferences update error: {e}")
        return jsonify({'error': 'Failed to update preferences'}), 500

@app.route('/learn', methods=['POST'])
def real_time_learning():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        content_id = data.get('content_id')
        action = data.get('action')
        context = data.get('context', {})

        if not user_id or not content_id or not action:
            return jsonify({'error': 'Missing required fields'}), 400

        rec_engine.update_online_learning(user_id, content_id, action, context)
        return jsonify({'message': 'Learning data recorded'})
    except Exception as e:
        logger.error(f"Real-time learning error: {e}")
        return jsonify({'error': 'Failed to record learning data'}), 500

def background_model_update():
    while True:
        try:
            time.sleep(MODEL_UPDATE_INTERVAL)
            if rec_engine.model_trained:
                logger.info("Running background model update...")
                rec_engine.train_model()
                if redis_client:
                    redis_client.flushdb()
                else:
                    with cache_lock:
                        recommendation_cache.clear()
                logger.info("Background model update completed")
        except Exception as e:
            logger.error(f"Background model update failed: {e}")

if __name__ == '__main__':
    logger.info("Starting ML Recommendation Service...")
    background_thread = threading.Thread(target=background_model_update, daemon=True)
    background_thread.start()
    rec_engine.train_model()
    logger.info("ML Recommendation Service ready")
    app.run(debug=True, host='0.0.0.0', port=5001)