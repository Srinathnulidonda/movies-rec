# ml-service/app.py
from flask import Flask, request, jsonify
import sqlite3, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import json, logging, os
from datetime import datetime, timedelta
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from collections import deque
import redis
import asyncio
import aiohttp
import joblib
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx
from textblob import TextBlob
from surprise import Dataset, Reader, SVD, NMF, KNNBasic, accuracy
from surprise.model_selection import cross_validate, train_test_split
import implicit
from scipy.sparse import csr_matrix
import faiss
import warnings
warnings.filterwarnings('ignore')


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for caching
recommendation_cache = {}
model_cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 3600  # 1 hour


class TrendAnalyzer:
    """Advanced trend analysis for content recommendations"""
    
    def __init__(self):
        self.trend_weights = {
            'popularity_momentum': 0.3,
            'rating_trend': 0.25,
            'genre_trend': 0.2,
            'seasonal_trend': 0.15,
            'viral_coefficient': 0.1
        }
        self.trend_cache = {}
    
    def analyze_content_trends(self, content_data, time_window_days=7):
        """Analyze trending patterns in content"""
        try:
            df = pd.DataFrame(content_data)
            
            # Calculate popularity momentum
            recent_interactions = df[df['created_at'] > (datetime.now() - timedelta(days=time_window_days))]
            popularity_momentum = recent_interactions.groupby('content_id')['interaction_count'].sum()
            
            # Calculate rating trends
            rating_trends = self.calculate_rating_trends(df)
            
            # Genre trend analysis
            genre_trends = self.analyze_genre_trends(df)
            
            # Seasonal trend detection
            seasonal_trends = self.detect_seasonal_patterns(df)
            
            # Viral coefficient calculation
            viral_scores = self.calculate_viral_coefficient(df)
            
            # Combine all trend factors
            trend_scores = {}
            all_content_ids = set(df['content_id'].unique())
            
            for content_id in all_content_ids:
                score = (
                    popularity_momentum.get(content_id, 0) * self.trend_weights['popularity_momentum'] +
                    rating_trends.get(content_id, 0) * self.trend_weights['rating_trend'] +
                    genre_trends.get(content_id, 0) * self.trend_weights['genre_trend'] +
                    seasonal_trends.get(content_id, 0) * self.trend_weights['seasonal_trend'] +
                    viral_scores.get(content_id, 0) * self.trend_weights['viral_coefficient']
                )
                trend_scores[content_id] = score
            
            return trend_scores
            
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return {}
    
    def calculate_rating_trends(self, df):
        """Calculate rating momentum and trends"""
        rating_trends = {}
        for content_id in df['content_id'].unique():
            content_ratings = df[df['content_id'] == content_id]['rating'].dropna()
            if len(content_ratings) > 1:
                # Calculate trend using linear regression
                x = np.arange(len(content_ratings))
                slope = np.polyfit(x, content_ratings, 1)[0]
                rating_trends[content_id] = slope
        return rating_trends
    
    def analyze_genre_trends(self, df):
        """Analyze trending genres"""
        genre_trends = {}
        current_time = datetime.now()
        
        for _, row in df.iterrows():
            genres = json.loads(row.get('genre_ids', '[]'))
            days_since_interaction = (current_time - pd.to_datetime(row['created_at'])).days
            
            # More recent interactions get higher weight
            time_weight = max(0, 1 - (days_since_interaction / 30))
            
            for genre in genres:
                if genre not in genre_trends:
                    genre_trends[genre] = 0
                genre_trends[genre] += time_weight
        
        # Apply genre trends to content
        content_genre_scores = {}
        for _, row in df.iterrows():
            content_id = row['content_id']
            genres = json.loads(row.get('genre_ids', '[]'))
            
            genre_score = sum(genre_trends.get(genre, 0) for genre in genres)
            content_genre_scores[content_id] = genre_score / len(genres) if genres else 0
        
        return content_genre_scores
    
    def detect_seasonal_patterns(self, df):
        """Detect seasonal viewing patterns"""
        seasonal_scores = {}
        current_month = datetime.now().month
        
        # Define seasonal preferences
        seasonal_genres = {
            'winter': [28, 18, 10402],  # Horror, Drama, Music
            'spring': [35, 10749, 16],  # Comedy, Romance, Animation
            'summer': [28, 12, 53],     # Action, Adventure, Thriller
            'fall': [18, 80, 99]        # Drama, Crime, Documentary
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
            
            # Calculate seasonal relevance
            seasonal_match = len(set(genres) & set(preferred_genres))
            seasonal_scores[content_id] = seasonal_match / len(genres) if genres else 0
        
        return seasonal_scores
    
    def calculate_viral_coefficient(self, df):
        """Calculate viral coefficient based on interaction patterns"""
        viral_scores = {}
        
        # Group interactions by content and time
        df['interaction_date'] = pd.to_datetime(df['created_at']).dt.date
        daily_interactions = df.groupby(['content_id', 'interaction_date']).size().reset_index(name='daily_count')
        
        for content_id in daily_interactions['content_id'].unique():
            content_interactions = daily_interactions[daily_interactions['content_id'] == content_id]
            
            if len(content_interactions) > 1:
                # Calculate acceleration in interactions
                daily_counts = content_interactions['daily_count'].values
                acceleration = np.diff(daily_counts, n=1)
                viral_coefficient = np.mean(acceleration) if len(acceleration) > 0 else 0
                viral_scores[content_id] = max(0, viral_coefficient)
        
        return viral_scores

class DiversityOptimizer:
    """Optimize recommendation diversity while maintaining relevance"""
    
    def __init__(self):
        self.diversity_weights = {
            'genre_diversity': 0.3,
            'content_type_diversity': 0.2,
            'release_date_diversity': 0.2,
            'popularity_diversity': 0.15,
            'rating_diversity': 0.15
        }
    
    def optimize_diversity(self, recommendations, target_diversity=0.7):
        """Optimize recommendation list for diversity"""
        if not recommendations:
            return recommendations
        
        # Calculate current diversity metrics
        current_diversity = self.calculate_diversity_score(recommendations)
        
        if current_diversity >= target_diversity:
            return recommendations
        
        # Apply diversity optimization
        optimized_recs = self.apply_diversity_algorithm(recommendations, target_diversity)
        
        return optimized_recs
    
    def calculate_diversity_score(self, recommendations):
        """Calculate overall diversity score"""
        if not recommendations:
            return 0
        
        scores = {
            'genre_diversity': self.calculate_genre_diversity(recommendations),
            'content_type_diversity': self.calculate_content_type_diversity(recommendations),
            'release_date_diversity': self.calculate_temporal_diversity(recommendations),
            'popularity_diversity': self.calculate_popularity_diversity(recommendations),
            'rating_diversity': self.calculate_rating_diversity(recommendations)
        }
        
        # Weighted average
        total_score = sum(scores[key] * self.diversity_weights[key] for key in scores)
        return total_score
    
    def calculate_genre_diversity(self, recommendations):
        """Calculate genre diversity using Shannon entropy"""
        genre_counts = {}
        total_genres = 0
        
        for rec in recommendations:
            genres = json.loads(rec.get('genre_ids', '[]'))
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                total_genres += 1
        
        if total_genres == 0:
            return 0
        
        # Calculate Shannon entropy
        entropy = 0
        for count in genre_counts.values():
            p = count / total_genres
            entropy -= p * np.log2(p)
        
        # Normalize to 0-1 range
        max_entropy = np.log2(len(genre_counts)) if genre_counts else 1
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def calculate_content_type_diversity(self, recommendations):
        """Calculate content type diversity"""
        content_types = [rec.get('content_type') for rec in recommendations]
        unique_types = len(set(content_types))
        total_types = len(content_types)
        
        return unique_types / total_types if total_types > 0 else 0
    
    def calculate_temporal_diversity(self, recommendations):
        """Calculate temporal diversity across release dates"""
        release_dates = []
        for rec in recommendations:
            date_str = rec.get('release_date')
            if date_str:
                try:
                    date = pd.to_datetime(date_str)
                    release_dates.append(date.year)
                except:
                    continue
        
        if not release_dates:
            return 0
        
        # Calculate year span diversity
        year_span = max(release_dates) - min(release_dates)
        return min(1.0, year_span / 50)  # Normalize to 50-year span
    
    def calculate_popularity_diversity(self, recommendations):
        """Calculate popularity diversity"""
        popularities = [rec.get('popularity', 0) for rec in recommendations]
        if not popularities:
            return 0
        
        # Calculate coefficient of variation
        mean_pop = np.mean(popularities)
        std_pop = np.std(popularities)
        
        return min(1.0, std_pop / mean_pop) if mean_pop > 0 else 0
    
    def calculate_rating_diversity(self, recommendations):
        """Calculate rating diversity"""
        ratings = [rec.get('vote_average', 0) for rec in recommendations]
        if not ratings:
            return 0
        
        # Calculate rating spread
        rating_range = max(ratings) - min(ratings)
        return min(1.0, rating_range / 10)  # Normalize to 0-10 scale
    
    def apply_diversity_algorithm(self, recommendations, target_diversity):
        """Apply diversity optimization algorithm"""
        # Sort by recommendation score
        sorted_recs = sorted(recommendations, key=lambda x: x.get('recommendation_score', 0), reverse=True)
        
        optimized_list = []
        candidate_pool = sorted_recs.copy()
        
        # Greedy diversity selection
        while len(optimized_list) < len(recommendations) and candidate_pool:
            best_candidate = None
            best_score = -1
            
            for candidate in candidate_pool:
                # Calculate diversity if this candidate is added
                test_list = optimized_list + [candidate]
                diversity_score = self.calculate_diversity_score(test_list)
                relevance_score = candidate.get('recommendation_score', 0)
                
                # Combined score (diversity + relevance)
                combined_score = diversity_score * 0.6 + (relevance_score / 10) * 0.4
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                optimized_list.append(best_candidate)
                candidate_pool.remove(best_candidate)
        
        return optimized_list

class AdvancedNeuralRecommender:
    """Advanced neural network recommender with deep learning"""
    
    def __init__(self):
        self.autoencoder = None
        self.vae_model = None
        self.gan_model = None
        self.transformer_model = None
        self.model_trained = False
    
    def build_variational_autoencoder(self, input_dim, latent_dim=64):
        """Build Variational Autoencoder for recommendation"""
        # Encoder
        encoder_input = Input(shape=(input_dim,))
        h = Dense(256, activation='relu')(encoder_input)
        h = Dropout(0.3)(h)
        h = Dense(128, activation='relu')(h)
        h = Dropout(0.3)(h)
        
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)
        
        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
        
        # Decoder
        decoder_h = Dense(128, activation='relu')(z)
        decoder_h = Dropout(0.3)(decoder_h)
        decoder_h = Dense(256, activation='relu')(decoder_h)
        decoder_h = Dropout(0.3)(decoder_h)
        decoder_output = Dense(input_dim, activation='sigmoid')(decoder_h)
        
        # VAE model
        vae = Model(encoder_input, decoder_output)
        
        # VAE loss
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(encoder_input, decoder_output)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        vae_loss = reconstruction_loss + kl_loss
        vae.add_loss(vae_loss)
        
        vae.compile(optimizer='adam')
        self.vae_model = vae
        
        return vae
    
    def build_attention_model(self, num_users, num_items, embedding_dim=64):
        """Build attention-based recommendation model"""
        # User and item embeddings
        user_input = Input(shape=(), name='user_id')
        item_input = Input(shape=(), name='item_id')
        
        user_embedding = Embedding(num_users, embedding_dim)(user_input)
        item_embedding = Embedding(num_items, embedding_dim)(item_input)
        
        user_vec = Flatten()(user_embedding)
        item_vec = Flatten()(item_embedding)
        
        # Attention mechanism
        attention_weights = Dense(1, activation='softmax')(
            tf.keras.layers.Concatenate()([user_vec, item_vec])
        )
        
        # Attended representations
        attended_user = tf.keras.layers.Multiply()([user_vec, attention_weights])
        attended_item = tf.keras.layers.Multiply()([item_vec, attention_weights])
        
        # Final prediction
        concat = tf.keras.layers.Concatenate()([attended_user, attended_item])
        dense1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        output = Dense(1, activation='sigmoid')(dropout2)
        
        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def build_graph_neural_network(self, user_item_interactions):
        """Build Graph Neural Network for recommendations"""
        # Create user-item interaction graph
        G = nx.Graph()
        
        for user_id, item_id, rating in user_item_interactions:
            G.add_edge(f"user_{user_id}", f"item_{item_id}", weight=rating)
        
        # Node embeddings using Graph Neural Network concepts
        node_embeddings = {}
        
        # Simple GNN implementation using adjacency matrix
        adjacency = nx.adjacency_matrix(G)
        
        # Initialize embeddings
        num_nodes = len(G.nodes())
        embeddings = np.random.normal(0, 0.1, (num_nodes, 64))
        
        # Message passing iterations
        for _ in range(3):
            new_embeddings = np.zeros_like(embeddings)
            for i, node in enumerate(G.nodes()):
                neighbors = list(G.neighbors(node))
                if neighbors:
                    neighbor_embeddings = [embeddings[j] for j, n in enumerate(G.nodes()) if n in neighbors]
                    new_embeddings[i] = np.mean(neighbor_embeddings, axis=0)
                else:
                    new_embeddings[i] = embeddings[i]
            embeddings = new_embeddings
        
        # Store embeddings
        for i, node in enumerate(G.nodes()):
            node_embeddings[node] = embeddings[i]
        
        return node_embeddings

class ReinforcementLearningRecommender:
    """Reinforcement Learning based recommender"""
    
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.action_space = []
        self.state_space = []
    
    def get_state(self, user_profile, context):
        """Get current state representation"""
        state = {
            'user_genre_preferences': user_profile.get('genre_preferences', {}),
            'time_of_day': context.get('time_of_day', 'unknown'),
            'device_type': context.get('device_type', 'unknown'),
            'session_length': context.get('session_length', 0),
            'last_interactions': context.get('last_interactions', [])
        }
        return json.dumps(state, sort_keys=True)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(self.action_space)
        else:
            # Exploit: best known action
            if state in self.q_table:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                return np.random.choice(self.action_space)
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table with new experience"""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        
        # Q-learning update
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0
        
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def get_rl_recommendations(self, user_profile, context, available_content):
        """Get recommendations using reinforcement learning"""
        state = self.get_state(user_profile, context)
        
        # Score each content item
        content_scores = {}
        for content in available_content:
            action = f"recommend_{content['content_type']}_{content.get('genre_ids', '[]')}"
            
            if state in self.q_table and action in self.q_table[state]:
                score = self.q_table[state][action]
            else:
                score = 0
            
            content_scores[content['id']] = score
        
        # Sort by Q-values
        sorted_content = sorted(available_content, 
                              key=lambda x: content_scores.get(x['id'], 0), 
                              reverse=True)
        
        return sorted_content


class AdvancedRecommendationEngine:
    def __init__(self, db_path='../backend/recommendations.db'):
        self.db_path = db_path
        self.content_features = None
        self.user_features = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        self.scaler = StandardScaler()
        self.similarity_matrix = None
        self.user_item_matrix = None
        self.model_trained = False
        self.model_timestamp = None
        # New advanced components
        self.deep_learning_model = None
        self.ensemble_models = {}
        self.real_time_buffer = deque(maxlen=10000)
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.contextual_weights = {}
        self.neural_cf_model = None
        self.trend_analyzer = TrendAnalyzer()
        self.diversity_optimizer = DiversityOptimizer()
        # Add new advanced components
        self.trend_analyzer = TrendAnalyzer()
        self.diversity_optimizer = DiversityOptimizer()
        self.neural_recommender = AdvancedNeuralRecommender()
        self.rl_recommender = ReinforcementLearningRecommender()
        
        # Advanced ML models
        self.surprise_models = {}
        self.implicit_model = None
        self.faiss_index = None
        self.knowledge_graph = None
        
        # Multi-armed bandit components
        self.bandit_arms = {}
        self.arm_rewards = {}
        self.arm_counts = {}
        
        # Contextual features
        self.contextual_features = {}
        self.user_clusters = {}
        self.item_clusters = {}
        
        # Real-time learning
        self.online_models = {}
        self.streaming_buffer = deque(maxlen=50000)
        self.model_update_frequency = 1000  # Update every 1000 interactions
        
        # Cold start handling
        self.cold_start_strategies = {}
        self.bootstrap_samples = 100
        
        # Real-time learning components
        self.online_learning_rate = 0.01
        self.adaptation_threshold = 0.1
        self.feedback_weights = {'click': 0.1, 'rating': 1.0, 'watchlist': 0.3, 'favorite': 0.8}
                # Initialize Redis for caching (optional)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        except:
            self.redis_client = None
            logger.warning("Redis not available, using in-memory caching")
        
    def get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def extract_content_features(self):
        """Extract and vectorize content features"""
        conn = self.get_db_connection()
        
        # Get content data
        content_data = conn.execute('''
            SELECT id, title, overview, genre_ids, vote_average, popularity, 
                   content_type, release_date, runtime
            FROM content
        ''').fetchall()
        
        if not content_data:
            logger.warning("No content data found")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in content_data])
        
        # Text features from title and overview
        text_features = df['title'].fillna('') + ' ' + df['overview'].fillna('')
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        
        # Numerical features
        numerical_features = []
        for _, row in df.iterrows():
            features = [
                row['vote_average'] or 0,
                np.log1p(row['popularity'] or 0),
                len(json.loads(row['genre_ids'] or '[]')),
                1 if row['content_type'] == 'movie' else 0,
                1 if row['content_type'] == 'tv' else 0,
                1 if row['content_type'] == 'anime' else 0,
                row['runtime'] or 0
            ]
            numerical_features.append(features)
        
        numerical_features = np.array(numerical_features)
        numerical_features = self.scaler.fit_transform(numerical_features)
        
        # Combine features
        content_features = np.hstack([tfidf_matrix.toarray(), numerical_features])
        
        # Dimensionality reduction
        content_features = self.svd.fit_transform(content_features)
        
        self.content_features = content_features
        self.content_ids = df['id'].values
        
        conn.close()
        return content_features
    

    def build_neural_collaborative_filtering(self):
        """Build Neural Collaborative Filtering model"""
        try:
            # Get user-item matrix dimensions
            conn = self.get_db_connection()
            
            max_user_id = conn.execute('SELECT MAX(user_id) FROM user_interactions').fetchone()[0]
            max_content_id = conn.execute('SELECT MAX(content_id) FROM user_interactions').fetchone()[0]
            
            if not max_user_id or not max_content_id:
                return None
            
            # Neural CF architecture
            user_input = Input(shape=(), name='user_id')
            item_input = Input(shape=(), name='item_id')
            
            # Embedding layers
            user_embedding = Embedding(max_user_id + 1, 64, name='user_embedding')(user_input)
            item_embedding = Embedding(max_content_id + 1, 64, name='item_embedding')(item_input)
            
            user_vec = Flatten()(user_embedding)
            item_vec = Flatten()(item_embedding)
            
            # Neural MF layers
            concat = Concatenate()([user_vec, item_vec])
            dense1 = Dense(128, activation='relu')(concat)
            dropout1 = Dropout(0.2)(dense1)
            dense2 = Dense(64, activation='relu')(dropout1)
            dropout2 = Dropout(0.2)(dense2)
            dense3 = Dense(32, activation='relu')(dropout2)
            output = Dense(1, activation='sigmoid')(dense3)
            
            model = Model(inputs=[user_input, item_input], outputs=output)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            self.neural_cf_model = model
            conn.close()
            
            return model
            
        except Exception as e:
            logger.error(f"Neural CF build error: {e}")
            return None
    
    def train_ensemble_models(self):
        """Train ensemble of different ML models"""
        try:
            # Prepare training data
            X, y = self.prepare_training_data()
            if X is None or len(X) == 0:
                return False
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'neural_network': MLPRegressor(hidden_layer_sizes=(128, 64, 32), random_state=42)
            }
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    self.ensemble_models[name] = {'model': model, 'score': score}
                    logger.info(f"Trained {name} with score: {score:.4f}")
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ensemble training error: {e}")
            return False
    
    def prepare_training_data(self):
        """Prepare training data for ML models"""
        try:
            conn = self.get_db_connection()
            
            # Get user-item interactions with features
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
                return None, None
            
            # Feature engineering
            X = []
            y = []
            
            for row in data:
                features = [
                    row['user_id'],
                    row['content_id'],
                    row['vote_average'] or 0,
                    np.log1p(row['popularity'] or 0),
                    1 if row['content_type'] == 'movie' else 0,
                    1 if row['content_type'] == 'tv' else 0,
                    1 if row['content_type'] == 'anime' else 0,
                    len(json.loads(row['genre_ids'] or '[]')),
                    # Time-based features
                    (datetime.now() - datetime.fromisoformat(row['interaction_created'])).days,
                    (datetime.now() - datetime.fromisoformat(row['user_created'])).days
                ]
                
                X.append(features)
                y.append(row['rating'])
            
            conn.close()
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Training data preparation error: {e}")
            return None, None
        
    def get_deep_learning_recommendations(self, user_id, limit=20):

    def build_user_item_matrix(self):
        """Build user-item interaction matrix"""
        conn = self.get_db_connection()
        
        # Get user interactions
        interactions = conn.execute('''
            SELECT user_id, content_id, interaction_type, rating
            FROM user_interactions
            WHERE interaction_type IN ('rating', 'watchlist', 'favorite')
        ''').fetchall()
        
        if not interactions:
            logger.warning("No user interactions found")
            return None
        
        # Create interaction matrix
        interaction_data = []
        for interaction in interactions:
            weight = 1.0
            if interaction['interaction_type'] == 'rating':
                weight = interaction['rating'] / 10.0
            elif interaction['interaction_type'] == 'favorite':
                weight = 1.0
            elif interaction['interaction_type'] == 'watchlist':
                weight = 0.5
            
            interaction_data.append({
                'user_id': interaction['user_id'],
                'content_id': interaction['content_id'],
                'weight': weight
            })
        
        df_interactions = pd.DataFrame(interaction_data)
        
        # Create pivot table
        user_item_matrix = df_interactions.pivot_table(
            index='user_id', 
            columns='content_id', 
            values='weight', 
            fill_value=0
        )
        
        self.user_item_matrix = user_item_matrix
        self.user_ids = user_item_matrix.index.values
        self.item_ids = user_item_matrix.columns.values
        
        conn.close()
        return user_item_matrix
    
    def compute_similarity_matrices(self):
        """Compute content and user similarity matrices"""
        if self.content_features is not None:
            self.content_similarity = cosine_similarity(self.content_features)
            logger.info("Content similarity matrix computed")
        
        if self.user_item_matrix is not None:
            # User-based collaborative filtering
            user_similarity = cosine_similarity(self.user_item_matrix.values)
            self.user_similarity = user_similarity
            
            # Item-based collaborative filtering
            item_similarity = cosine_similarity(self.user_item_matrix.T.values)
            self.item_similarity = item_similarity
            
            logger.info("User and item similarity matrices computed")
    
    def train_model(self):
        """Train the recommendation model"""
        try:
            logger.info("Training recommendation model...")
            
            # Extract features
            self.extract_content_features()
            self.build_user_item_matrix()
            
            # Compute similarity matrices
            self.compute_similarity_matrices()
            
            self.model_trained = True
            self.model_timestamp = datetime.now()
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def get_content_based_recommendations(self, user_id, limit=20):
        """Get content-based recommendations"""
        if not self.model_trained or self.content_features is None:
            return []
        
        conn = self.get_db_connection()
        
        # Get user's rated content
        user_ratings = conn.execute('''
            SELECT content_id, rating FROM user_interactions
            WHERE user_id = ? AND interaction_type = 'rating'
            ORDER BY rating DESC
        ''', (user_id,)).fetchall()
        
        if not user_ratings:
            # Return popular content for new users
            popular_content = conn.execute('''
                SELECT id, title, vote_average, popularity
                FROM content
                ORDER BY popularity DESC, vote_average DESC
                LIMIT ?
            ''', (limit,)).fetchall()
            conn.close()
            return [dict(item) for item in popular_content]
        
        # Find similar content based on high-rated items
        recommendations = {}
        
        for rating in user_ratings[:10]:  # Top 10 rated items
            content_id = rating['content_id']
            user_rating = rating['rating']
            
            # Find content index
            try:
                content_idx = np.where(self.content_ids == content_id)[0][0]
            except IndexError:
                continue
            
            # Get similar content
            similarities = self.content_similarity[content_idx]
            similar_indices = np.argsort(similarities)[::-1][1:limit+1]
            
            for idx in similar_indices:
                similar_content_id = self.content_ids[idx]
                similarity_score = similarities[idx]
                weighted_score = similarity_score * (user_rating / 10.0)
                
                if similar_content_id in recommendations:
                    recommendations[similar_content_id] += weighted_score
                else:
                    recommendations[similar_content_id] = weighted_score
        
        # Get top recommendations
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Fetch content details
        content_details = []
        for content_id, score in top_recommendations:
            content = conn.execute('SELECT * FROM content WHERE id = ?', (content_id,)).fetchone()
            if content:
                content_dict = dict(content)
                content_dict['recommendation_score'] = score
                content_details.append(content_dict)
        
        conn.close()
        return content_details
    
    def get_collaborative_recommendations(self, user_id, limit=20):
        """Get collaborative filtering recommendations"""
        if not self.model_trained or self.user_item_matrix is None:
            return []
        
        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
        except IndexError:
            return []
        
        # Get similar users
        user_similarities = self.user_similarity[user_idx]
        similar_users = np.argsort(user_similarities)[::-1][1:11]  # Top 10 similar users
        
        # Get recommendations from similar users
        recommendations = {}
        user_items = set(self.user_item_matrix.iloc[user_idx].nonzero()[0])
        
        for similar_user_idx in similar_users:
            similarity_score = user_similarities[similar_user_idx]
            similar_user_items = self.user_item_matrix.iloc[similar_user_idx]
            
            for item_idx, rating in similar_user_items.items():
                if rating > 0 and item_idx not in user_items:
                    item_id = self.item_ids[item_idx]
                    weighted_score = similarity_score * rating
                    
                    if item_id in recommendations:
                        recommendations[item_id] += weighted_score
                    else:
                        recommendations[item_id] = weighted_score
        
        # Get top recommendations
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Fetch content details
        conn = self.get_db_connection()
        content_details = []
        for content_id, score in top_recommendations:
            content = conn.execute('SELECT * FROM content WHERE id = ?', (content_id,)).fetchone()
            if content:
                content_dict = dict(content)
                content_dict['recommendation_score'] = score
                content_details.append(content_dict)
        
        conn.close()
        return content_details
    
    def get_hybrid_recommendations(self, user_id, limit=20):
        """Get hybrid recommendations combining multiple approaches"""
        content_recs = self.get_content_based_recommendations(user_id, limit)
        collab_recs = self.get_collaborative_recommendations(user_id, limit)
        
        # Combine recommendations with weighted scoring
        combined_recs = {}
        
        # Weight content-based recommendations (60%)
        for rec in content_recs:
            content_id = rec['id']
            score = rec.get('recommendation_score', 0) * 0.6
            combined_recs[content_id] = {'content': rec, 'score': score}
        
        # Weight collaborative recommendations (40%)
        for rec in collab_recs:
            content_id = rec['id']
            score = rec.get('recommendation_score', 0) * 0.4
            
            if content_id in combined_recs:
                combined_recs[content_id]['score'] += score
            else:
                combined_recs[content_id] = {'content': rec, 'score': score}
        
        # Sort by combined score
        sorted_recs = sorted(combined_recs.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Add diversity boost for different content types
        final_recs = []
        content_types_seen = set()
        
        for content_id, rec_data in sorted_recs:
            content = rec_data['content']
            content_type = content.get('content_type', 'unknown')
            
            # Boost score for content type diversity
            if content_type not in content_types_seen:
                rec_data['score'] *= 1.2
                content_types_seen.add(content_type)
            
            content['recommendation_score'] = rec_data['score']
            final_recs.append(content)
            
            if len(final_recs) >= limit:
                break
        
        return final_recs
    
    def get_trending_personalized(self, user_id, limit=20):
        """Get personalized trending content based on user preferences"""
        conn = self.get_db_connection()
        
        # Get user's preferred genres
        user_genres = conn.execute('''
            SELECT genre_ids FROM content c
            JOIN user_interactions ui ON c.id = ui.content_id
            WHERE ui.user_id = ? AND ui.interaction_type = 'rating' AND ui.rating >= 7
        ''', (user_id,)).fetchall()
        
        # Extract genre preferences
        genre_counts = {}
        for row in user_genres:
            genres = json.loads(row['genre_ids'] or '[]')
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Get trending content
        trending = conn.execute('''
            SELECT * FROM content
            WHERE created_at > datetime('now', '-7 days')
            ORDER BY popularity DESC, vote_average DESC
            LIMIT ?
        ''', (limit * 2,)).fetchall()  # Get more to filter
        
        # Score trending content based on user preferences
        scored_content = []
        for content in trending:
            base_score = (content['popularity'] or 0) * 0.7 + (content['vote_average'] or 0) * 0.3
            
            # Boost based on genre preferences
            content_genres = json.loads(content['genre_ids'] or '[]')
            genre_boost = sum(genre_counts.get(genre, 0) for genre in content_genres)
            
            final_score = base_score + genre_boost
            
            content_dict = dict(content)
            content_dict['recommendation_score'] = final_score
            scored_content.append(content_dict)
        
        # Sort and return top results
        scored_content.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        conn.close()
        return scored_content[:limit]

# Global recommendation engine instance
rec_engine = AdvancedRecommendationEngine()

def get_cached_recommendations(user_id, rec_type='hybrid', limit=20):
    """Get cached recommendations or compute new ones"""
    cache_key = f"{user_id}_{rec_type}_{limit}"
    
    with cache_lock:
        if cache_key in recommendation_cache:
            cached_data = recommendation_cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(seconds=CACHE_DURATION):
                return cached_data['recommendations']
    
    # Compute new recommendations
    if rec_type == 'hybrid':
        recommendations = rec_engine.get_hybrid_recommendations(user_id, limit)
    elif rec_type == 'content':
        recommendations = rec_engine.get_content_based_recommendations(user_id, limit)
    elif rec_type == 'collaborative':
        recommendations = rec_engine.get_collaborative_recommendations(user_id, limit)
    elif rec_type == 'trending':
        recommendations = rec_engine.get_trending_personalized(user_id, limit)
    else:
        recommendations = []
    
    # Cache the results
    with cache_lock:
        recommendation_cache[cache_key] = {
            'recommendations': recommendations,
            'timestamp': datetime.now()
        }
    
    return recommendations

def retrain_model_if_needed():
    """Retrain model if it's outdated"""
    if not rec_engine.model_trained or not rec_engine.model_timestamp:
        return rec_engine.train_model()
    
    # Retrain if model is older than 24 hours
    if datetime.now() - rec_engine.model_timestamp > timedelta(hours=24):
        logger.info("Model is outdated, retraining...")
        return rec_engine.train_model()
    
    return True

# API Routes
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_trained': rec_engine.model_trained,
        'model_timestamp': rec_engine.model_timestamp.isoformat() if rec_engine.model_timestamp else None,
        'cache_size': len(recommendation_cache),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/train', methods=['POST'])
def train_model():
    """Train/retrain the recommendation model"""
    try:
        success = rec_engine.train_model()
        if success:
            # Clear cache after retraining
            with cache_lock:
                recommendation_cache.clear()
            return jsonify({'message': 'Model trained successfully'})
        else:
            return jsonify({'error': 'Model training failed'}), 500
    except Exception as e:
        logger.error(f"Training endpoint error: {e}")
        return jsonify({'error': 'Training failed'}), 500

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get personalized recommendations for a user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        limit = min(data.get('limit', 20), 100)  # Cap at 100
        rec_type = data.get('type', 'hybrid')
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        # Ensure model is trained and up to date
        if not retrain_model_if_needed():
            return jsonify({'error': 'Model training failed'}), 500
        
        # Get recommendations
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

@app.route('/similar/<int:content_id>')
def get_similar_content(content_id):
    """Get similar content to a specific item"""
    try:
        if not rec_engine.model_trained or rec_engine.content_features is None:
            return jsonify({'error': 'Model not trained'}), 503
        
        # Find content index
        try:
            content_idx = np.where(rec_engine.content_ids == content_id)[0][0]
        except IndexError:
            return jsonify({'error': 'Content not found'}), 404
        
        # Get similar content
        similarities = rec_engine.content_similarity[content_idx]
        similar_indices = np.argsort(similarities)[::-1][1:21]  # Top 20 similar
        
        # Fetch content details
        conn = rec_engine.get_db_connection()
        similar_content = []
        
        for idx in similar_indices:
            similar_content_id = rec_engine.content_ids[idx]
            similarity_score = similarities[idx]
            
            content = conn.execute('SELECT * FROM content WHERE id = ?', (similar_content_id,)).fetchone()
            if content:
                content_dict = dict(content)
                content_dict['similarity_score'] = similarity_score
                similar_content.append(content_dict)
        
        conn.close()
        
        return jsonify({
            'similar_content': similar_content,
            'content_id': content_id,
            'count': len(similar_content)
        })
        
    except Exception as e:
        logger.error(f"Similar content endpoint error: {e}")
        return jsonify({'error': 'Similar content service unavailable'}), 500

@app.route('/user_profile/<int:user_id>')
def get_user_profile(user_id):
    """Get user's content preferences and statistics"""
    try:
        conn = rec_engine.get_db_connection()
        
        # Get user's interaction statistics
        stats = conn.execute('''
            SELECT 
                interaction_type,
                COUNT(*) as count,
                AVG(rating) as avg_rating
            FROM user_interactions
            WHERE user_id = ?
            GROUP BY interaction_type
        ''', (user_id,)).fetchall()
        
        # Get genre preferences
        genre_stats = conn.execute('''
            SELECT 
                genre_ids,
                AVG(ui.rating) as avg_rating,
                COUNT(*) as count
            FROM content c
            JOIN user_interactions ui ON c.id = ui.content_id
            WHERE ui.user_id = ? AND ui.interaction_type = 'rating'
            GROUP BY genre_ids
        ''', (user_id,)).fetchall()
        
        # Process genre preferences
        genre_preferences = {}
        for row in genre_stats:
            genres = json.loads(row['genre_ids'] or '[]')
            for genre in genres:
                if genre not in genre_preferences:
                    genre_preferences[genre] = {'total_rating': 0, 'count': 0}
                genre_preferences[genre]['total_rating'] += row['avg_rating'] * row['count']
                genre_preferences[genre]['count'] += row['count']
        
        # Calculate average ratings per genre
        for genre in genre_preferences:
            genre_preferences[genre]['avg_rating'] = (
                genre_preferences[genre]['total_rating'] / genre_preferences[genre]['count']
            )
        
        conn.close()
        
        return jsonify({
            'user_id': user_id,
            'interaction_stats': [dict(row) for row in stats],
            'genre_preferences': genre_preferences,
            'profile_generated': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"User profile endpoint error: {e}")
        return jsonify({'error': 'User profile service unavailable'}), 500

@app.route('/batch_recommend', methods=['POST'])
def batch_recommendations():
    """Get recommendations for multiple users efficiently"""
    try:
        data = request.get_json()
        user_ids = data.get('user_ids', [])
        limit = min(data.get('limit', 10), 50)
        rec_type = data.get('type', 'hybrid')
        
        if not user_ids or len(user_ids) > 100:
            return jsonify({'error': 'Invalid user_ids (max 100 users)'}), 400
        
        # Ensure model is trained
        if not retrain_model_if_needed():
            return jsonify({'error': 'Model training failed'}), 500
        
        # Get recommendations for all users in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(get_cached_recommendations, user_id, rec_type, limit): user_id
                for user_id in user_ids
            }
            
            batch_results = {}
            for future in futures:
                user_id = futures[future]
                try:
                    recommendations = future.result(timeout=30)
                    batch_results[user_id] = recommendations
                except Exception as e:
                    logger.error(f"Batch recommendation failed for user {user_id}: {e}")
                    batch_results[user_id] = []
        
        return jsonify({
            'batch_recommendations': batch_results,
            'type': rec_type,
            'processed_users': len(batch_results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch recommendation endpoint error: {e}")
        return jsonify({'error': 'Batch recommendation service unavailable'}), 500

# Background task to periodically retrain model
def background_model_update():
    """Background task to update model periodically"""
    import time
    while True:
        try:
            time.sleep(3600)  # Wait 1 hour
            if rec_engine.model_trained:
                logger.info("Running background model update...")
                rec_engine.train_model()
                with cache_lock:
                    recommendation_cache.clear()
                logger.info("Background model update completed")
        except Exception as e:
            logger.error(f"Background model update failed: {e}")


    def build_knowledge_graph(self):
        """Build knowledge graph from content relationships"""
        try:
            conn = self.get_db_connection()
            
            # Get content with detailed metadata
            content_data = conn.execute('''
                SELECT c.*, 
                       GROUP_CONCAT(DISTINCT ui.user_id) as user_interactions,
                       AVG(ui.rating) as avg_user_rating
                FROM content c
                LEFT JOIN user_interactions ui ON c.id = ui.content_id
                GROUP BY c.id
            ''').fetchall()
            
            # Build graph
            G = nx.Graph()
            
            for content in content_data:
                content_id = content['id']
                G.add_node(content_id, 
                          content_type=content['content_type'],
                          genres=json.loads(content['genre_ids'] or '[]'),
                          rating=content['vote_average'],
                          popularity=content['popularity'])
                
                # Add edges based on genre similarity
                genres = set(json.loads(content['genre_ids'] or '[]'))
                
                for other_content in content_data:
                    if other_content['id'] != content_id:
                        other_genres = set(json.loads(other_content['genre_ids'] or '[]'))
                        
                        # Calculate Jaccard similarity
                        if genres and other_genres:
                            similarity = len(genres & other_genres) / len(genres | other_genres)
                            if similarity > 0.3:  # Threshold for connection
                                G.add_edge(content_id, other_content['id'], weight=similarity)
            
            self.knowledge_graph = G
            conn.close()
            
            logger.info(f"Knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return True
            
        except Exception as e:
            logger.error(f"Knowledge graph building error: {e}")
            return False
    
    def train_surprise_models(self):
        """Train multiple Surprise library models"""
        try:
            conn = self.get_db_connection()
            
            # Get ratings data
            ratings_data = conn.execute('''
                SELECT user_id, content_id, rating
                FROM user_interactions
                WHERE interaction_type = 'rating' AND rating IS NOT NULL
            ''').fetchall()
            
            if not ratings_data:
                return False
            
            # Prepare data for Surprise
            df = pd.DataFrame(ratings_data)
            reader = Reader(rating_scale=(1, 10))
            data = Dataset.load_from_df(df[['user_id', 'content_id', 'rating']], reader)
            
            # Train different models
            algorithms = {
                'SVD': SVD(n_factors=100, reg_all=0.02),
                'NMF': NMF(n_factors=50, reg_pu=0.06, reg_qi=0.06),
                'KNN_User': KNNBasic(k=40, sim_options={'user_based': True}),
                'KNN_Item': KNNBasic(k=40, sim_options={'user_based': False})
            }
            
            for name, algo in algorithms.items():
                try:
                    # Cross-validation
                    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
                    
                    # Train on full dataset
                    trainset = data.build_full_trainset()
                    algo.fit(trainset)
                    
                    self.surprise_models[name] = {
                        'model': algo,
                        'rmse': np.mean(cv_results['test_rmse']),
                        'mae': np.mean(cv_results['test_mae'])
                    }
                    
                    logger.info(f"Trained {name} - RMSE: {np.mean(cv_results['test_rmse']):.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Surprise models training error: {e}")
            return False
    
    def train_implicit_model(self):
        """Train implicit feedback model using ALS"""
        try:
            conn = self.get_db_connection()
            
            # Get implicit feedback data
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
                return False
            
            # Create sparse matrix
            df = pd.DataFrame(interactions)
            
            # Map IDs to indices
            user_ids = df['user_id'].unique()
            item_ids = df['content_id'].unique()
            
            user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
            item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}
            
            # Create interaction matrix
            rows = [user_id_map[uid] for uid in df['user_id']]
            cols = [item_id_map[iid] for iid in df['content_id']]
            data = df['confidence'].values
            
            interaction_matrix = csr_matrix((data, (rows, cols)), 
                                          shape=(len(user_ids), len(item_ids)))
            
            # Train ALS model
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
            
        except Exception as e:
            logger.error(f"Implicit model training error: {e}")
            return False
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        try:
            if self.content_features is None:
                return False
            
            # Initialize FAISS index
            dimension = self.content_features.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize features for cosine similarity
            normalized_features = self.content_features / np.linalg.norm(self.content_features, axis=1, keepdims=True)
            
            # Add vectors to index
            index.add(normalized_features.astype('float32'))
            
            self.faiss_index = index
            logger.info(f"FAISS index built with {index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"FAISS index building error: {e}")
            return False
    
    def get_deep_learning_recommendations(self, user_id, limit=20):
        """Get recommendations using deep learning models"""
        try:
            recommendations = []
            
            # VAE-based recommendations
            if self.neural_recommender.vae_model:
                vae_recs = self.get_vae_recommendations(user_id, limit)
                recommendations.extend(vae_recs)
            
            # Attention-based recommendations
            if self.neural_recommender.transformer_model:
                attention_recs = self.get_attention_recommendations(user_id, limit)
                recommendations.extend(attention_recs)
            
            # GNN-based recommendations
            if self.knowledge_graph:
                gnn_recs = self.get_gnn_recommendations(user_id, limit)
                recommendations.extend(gnn_recs)
            
            # Combine and deduplicate
            seen_ids = set()
            final_recs = []
            
            for rec in recommendations:
                if rec['id'] not in seen_ids:
                    seen_ids.add(rec['id'])
                    final_recs.append(rec)
            
            return final_recs[:limit]
            
        except Exception as e:
            logger.error(f"Deep learning recommendations error: {e}")
            return []
    
    def get_contextual

if __name__ == '__main__':
    # Initialize and train model on startup
    logger.info("Starting ML Recommendation Service...")
    
    # Start background thread for model updates
    import threading
    background_thread = threading.Thread(target=background_model_update, daemon=True)
    background_thread.start()
    
    # Train initial model
    rec_engine.train_model()
    
    logger.info("ML Recommendation Service ready")
    app.run(debug=True, host='0.0.0.0', port=5001)