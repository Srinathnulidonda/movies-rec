# ml-service/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, euclidean_distances
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import os
import requests
import pickle
import joblib
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import cdist
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///movie_recommendations.db')
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:5000')
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', './models')
CACHE_EXPIRY_HOURS = int(os.environ.get('CACHE_EXPIRY_HOURS', '24'))

# Ensure directories exist
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs('./cache', exist_ok=True)

# Enhanced Regional Language Configuration with Telugu Priority
REGIONAL_LANGUAGES = {
    'telugu': {
        'name': 'Telugu',
        'priority': 1,
        'boost_factor': 1.8,
        'industry': 'Tollywood',
        'cultural_keywords': ['tollywood', 'andhra', 'telangana', 'hyderabad', 'telugu', 'nandamuri', 'allu arjun', 'mahesh babu', 'prabhas'],
        'weight': 0.4
    },
    'hindi': {
        'name': 'Hindi',
        'priority': 2,
        'boost_factor': 1.5,
        'industry': 'Bollywood',
        'cultural_keywords': ['bollywood', 'mumbai', 'hindi', 'shah rukh', 'salman', 'aamir'],
        'weight': 0.3
    },
    'tamil': {
        'name': 'Tamil',
        'priority': 3,
        'boost_factor': 1.3,
        'industry': 'Kollywood',
        'cultural_keywords': ['kollywood', 'tamil nadu', 'chennai', 'rajinikanth', 'kamal'],
        'weight': 0.25
    },
    'malayalam': {
        'name': 'Malayalam',
        'priority': 4,
        'boost_factor': 1.2,
        'industry': 'Mollywood',
        'cultural_keywords': ['mollywood', 'kerala', 'kochi', 'mohanlal', 'mammootty'],
        'weight': 0.2
    },
    'kannada': {
        'name': 'Kannada',
        'priority': 5,
        'boost_factor': 1.2,
        'industry': 'Sandalwood',
        'cultural_keywords': ['sandalwood', 'karnataka', 'bangalore', 'yash', 'puneeth'],
        'weight': 0.2
    },
    'english': {
        'name': 'English',
        'priority': 6,
        'boost_factor': 0.8,
        'industry': 'Hollywood',
        'cultural_keywords': ['hollywood', 'american', 'british'],
        'weight': 0.15
    }
}

class AdvancedRecommendationEngine:
    def __init__(self):
        # Content-based models
        self.content_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3))
        self.genre_vectorizer = TfidfVectorizer(max_features=200)
        self.language_vectorizer = TfidfVectorizer(max_features=100)
        
        # Collaborative filtering models
        self.user_similarity_model = NearestNeighbors(n_neighbors=50, metric='cosine')
        self.item_similarity_model = NearestNeighbors(n_neighbors=30, metric='cosine')
        
        # Matrix factorization models
        self.svd_model = TruncatedSVD(n_components=100, random_state=42)
        self.nmf_model = NMF(n_components=80, random_state=42)
        self.pca_model = PCA(n_components=50, random_state=42)
        
        # Neural network model
        self.neural_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # Ensemble models
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.label_encoders = {}
        
        # Clustering for user segmentation
        self.user_cluster_model = KMeans(n_clusters=10, random_state=42)
        self.content_cluster_model = KMeans(n_clusters=20, random_state=42)
        
        # Feature matrices
        self.content_features = None
        self.genre_features = None
        self.language_features = None
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.content_similarity_matrix = None
        self.user_similarity_matrix = None
        
        # Advanced features
        self.regional_embeddings = None
        self.temporal_features = None
        self.popularity_features = None
        
        # Model states
        self.is_fitted = False
        self.models_trained = {}
        
        # Data caches
        self.content_df = None
        self.interactions_df = None
        self.user_profiles = {}
        self.content_profiles = {}
        self.regional_preferences = {}
        
        # Caching
        self.cache = {}
        self.cache_timestamps = {}
        
        # Initialization flag
        self._initialized = False
        
        logger.info("Advanced Recommendation Engine initialized")
    
    def initialize_models(self):
        """Initialize models - called once when app starts"""
        if self._initialized:
            return True
        
        try:
            logger.info("Initializing Advanced Recommendation Service...")
            
            # Try to load existing models
            if not self._load_models():
                logger.info("No existing models found. Training new models...")
                success = self.fit()
                if not success:
                    logger.warning("Model training failed - will train when data becomes available")
            
            self._initialized = True
            logger.info("Advanced Recommendation Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self._initialized = True  # Mark as initialized to prevent repeated failures
            return False
    
    def fetch_data_from_database(self):
        """Fetch and process data from database"""
        try:
            # Connect to database
            if DATABASE_URL.startswith('sqlite'):
                conn = sqlite3.connect(DATABASE_URL.replace('sqlite:///', ''))
            else:
                import psycopg2
                conn = psycopg2.connect(DATABASE_URL)
            
            # Fetch content data with all fields
            content_query = """
            SELECT 
                id, tmdb_id, imdb_id, title, original_title, content_type, genres, languages,
                release_date, runtime, rating, vote_count, popularity, overview,
                poster_path, backdrop_path, regional_category, is_trending, 
                is_all_time_hit, is_new_release, created_at, updated_at
            FROM content
            ORDER BY created_at DESC
            """
            
            self.content_df = pd.read_sql_query(content_query, conn)
            
            # Fetch user interactions
            interactions_query = """
            SELECT 
                ui.user_id, ui.content_id, ui.interaction_type, ui.rating, ui.timestamp,
                c.regional_category, c.content_type, c.genres, c.rating as content_rating,
                c.popularity, c.is_trending, c.is_all_time_hit, c.is_new_release
            FROM user_interaction ui
            LEFT JOIN content c ON ui.content_id = c.id
            WHERE ui.timestamp >= datetime('now', '-730 days')
            ORDER BY ui.timestamp DESC
            """
            
            self.interactions_df = pd.read_sql_query(interactions_query, conn)
            
            # Fetch anonymous interactions
            anon_query = """
            SELECT 
                ai.session_id as user_id, ai.content_id, ai.interaction_type, ai.timestamp,
                ai.ip_address, c.regional_category, c.content_type, c.genres,
                c.rating as content_rating, c.popularity
            FROM anonymous_interaction ai
            LEFT JOIN content c ON ai.content_id = c.id
            WHERE ai.timestamp >= datetime('now', '-90 days')
            ORDER BY ai.timestamp DESC
            """
            
            anon_df = pd.read_sql_query(anon_query, conn)
            anon_df['rating'] = None
            anon_df['is_trending'] = False
            anon_df['is_all_time_hit'] = False
            anon_df['is_new_release'] = False
            
            # Combine interactions
            self.interactions_df = pd.concat([self.interactions_df, anon_df], ignore_index=True)
            
            conn.close()
            
            if self.content_df.empty:
                logger.warning("No content data found in database")
                return False
            
            # Process the data
            self._process_content_data()
            self._process_interaction_data()
            self._create_advanced_features()
            
            logger.info(f"Fetched {len(self.content_df)} content items and {len(self.interactions_df)} interactions")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching data from database: {e}")
            return False
    
    def _process_content_data(self):
        """Advanced content data processing"""
        try:
            # Parse JSON fields safely
            def safe_json_parse(x, default=None):
                if default is None:
                    default = []
                try:
                    if pd.isna(x) or x is None or x == 'null':
                        return default
                    if isinstance(x, str):
                        return json.loads(x)
                    return x if isinstance(x, list) else default
                except:
                    return default
            
            self.content_df['genres_list'] = self.content_df['genres'].apply(safe_json_parse)
            self.content_df['languages_list'] = self.content_df['languages'].apply(safe_json_parse)
            
            # Advanced text features
            self.content_df['content_text'] = (
                self.content_df['title'].fillna('') + ' ' +
                self.content_df['original_title'].fillna('') + ' ' +
                self.content_df['overview'].fillna('') + ' ' +
                self.content_df['genres_list'].apply(lambda x: ' '.join(x) if x else '') + ' ' +
                self.content_df['languages_list'].apply(lambda x: ' '.join(x) if x else '') + ' ' +
                self.content_df['regional_category'].fillna('') + ' ' +
                self.content_df['content_type'].fillna('')
            )
            
            # Create genre and language text
            self.content_df['genres_text'] = self.content_df['genres_list'].apply(
                lambda x: ' '.join(x) if x else ''
            )
            self.content_df['languages_text'] = self.content_df['languages_list'].apply(
                lambda x: ' '.join(x) if x else ''
            )
            
            # Advanced numerical features
            self.content_df['rating'] = pd.to_numeric(self.content_df['rating'], errors='coerce').fillna(5.0)
            self.content_df['vote_count'] = pd.to_numeric(self.content_df['vote_count'], errors='coerce').fillna(0)
            self.content_df['popularity'] = pd.to_numeric(self.content_df['popularity'], errors='coerce').fillna(0)
            self.content_df['runtime'] = pd.to_numeric(self.content_df['runtime'], errors='coerce').fillna(120)
            
            # Regional features
            self.content_df['regional_category'] = self.content_df['regional_category'].fillna('other')
            self.content_df['regional_priority'] = self.content_df['regional_category'].map(
                lambda x: REGIONAL_LANGUAGES.get(x, {}).get('priority', 10)
            )
            self.content_df['regional_boost'] = self.content_df['regional_category'].map(
                lambda x: REGIONAL_LANGUAGES.get(x, {}).get('boost_factor', 1.0)
            )
            self.content_df['regional_weight'] = self.content_df['regional_category'].map(
                lambda x: REGIONAL_LANGUAGES.get(x, {}).get('weight', 0.1)
            )
            
            # Temporal features
            self.content_df['release_date'] = pd.to_datetime(self.content_df['release_date'], errors='coerce')
            current_date = datetime.now()
            
            self.content_df['days_since_release'] = (
                current_date - self.content_df['release_date']
            ).dt.days.fillna(365)
            
            self.content_df['release_year'] = self.content_df['release_date'].dt.year.fillna(2020)
            self.content_df['is_recent'] = (self.content_df['days_since_release'] <= 365).astype(int)
            self.content_df['is_classic'] = (self.content_df['days_since_release'] > 1825).astype(int)
            
            # Quality scores
            self.content_df['normalized_rating'] = self.min_max_scaler.fit_transform(
                self.content_df[['rating']]
            ).flatten()
            
            self.content_df['log_vote_count'] = np.log1p(self.content_df['vote_count'])
            self.content_df['log_popularity'] = np.log1p(self.content_df['popularity'])
            
            self.content_df['quality_score'] = (
                self.content_df['normalized_rating'] * 0.4 +
                self.min_max_scaler.fit_transform(self.content_df[['log_vote_count']]).flatten() * 0.3 +
                self.min_max_scaler.fit_transform(self.content_df[['log_popularity']]).flatten() * 0.3
            )
            
            # Content categories
            self.content_df['is_trending'] = self.content_df['is_trending'].fillna(False)
            self.content_df['is_all_time_hit'] = self.content_df['is_all_time_hit'].fillna(False)
            self.content_df['is_new_release'] = self.content_df['is_new_release'].fillna(False)
            
            # Category scores
            self.content_df['category_boost'] = (
                self.content_df['is_trending'].astype(int) * 0.15 +
                self.content_df['is_all_time_hit'].astype(int) * 0.25 +
                self.content_df['is_new_release'].astype(int) * 0.1
            )
            
            # Final content score
            self.content_df['final_score'] = (
                self.content_df['quality_score'] * 0.6 +
                self.content_df['regional_boost'] * 0.3 +
                self.content_df['category_boost'] * 0.1
            )
            
            logger.info("Content data processing completed")
            
        except Exception as e:
            logger.error(f"Error processing content data: {e}")
    
    def _process_interaction_data(self):
        """Advanced interaction data processing"""
        try:
            if self.interactions_df.empty:
                logger.warning("No interaction data available")
                return
            
            # Interaction weights with advanced scoring
            interaction_weights = {
                'view': 1.0,
                'like': 2.5,
                'favorite': 4.0,
                'watchlist': 3.0,
                'search': 0.8,
                'rating': 0.0  # Handled separately
            }
            
            self.interactions_df['interaction_weight'] = self.interactions_df['interaction_type'].map(
                lambda x: interaction_weights.get(x, 1.0)
            )
            
            # Advanced rating processing
            self.interactions_df['rating'] = pd.to_numeric(self.interactions_df['rating'], errors='coerce')
            rating_mask = self.interactions_df['rating'].notna()
            self.interactions_df.loc[rating_mask, 'interaction_weight'] = (
                self.interactions_df.loc[rating_mask, 'rating'] / 2.0
            )
            
            # Temporal decay
            self.interactions_df['timestamp'] = pd.to_datetime(self.interactions_df['timestamp'])
            current_time = datetime.now()
            self.interactions_df['days_ago'] = (
                current_time - self.interactions_df['timestamp']
            ).dt.days
            
            # Temporal decay factor (more recent interactions have higher weight)
            self.interactions_df['temporal_decay'] = np.exp(-self.interactions_df['days_ago'] / 30.0)
            self.interactions_df['final_weight'] = (
                self.interactions_df['interaction_weight'] * self.interactions_df['temporal_decay']
            )
            
            # Regional interaction analysis
            self._analyze_regional_interactions()
            
            # Create user and content profiles
            self._create_user_profiles()
            self._create_content_profiles()
            
            logger.info("Interaction data processing completed")
            
        except Exception as e:
            logger.error(f"Error processing interaction data: {e}")
    
    def _analyze_regional_interactions(self):
        """Analyze regional content interaction patterns"""
        try:
            # Regional preference analysis
            regional_stats = self.interactions_df.groupby('regional_category').agg({
                'final_weight': ['sum', 'mean', 'count'],
                'rating': 'mean'
            }).round(3)
            
            regional_stats.columns = ['total_weight', 'avg_weight', 'interaction_count', 'avg_rating']
            
            # Calculate regional popularity scores
            total_interactions = len(self.interactions_df)
            for region in REGIONAL_LANGUAGES.keys():
                region_data = regional_stats.loc[regional_stats.index == region]
                if not region_data.empty:
                    popularity = region_data['interaction_count'].iloc[0] / total_interactions
                    self.regional_preferences[region] = {
                        'popularity': popularity,
                        'avg_rating': region_data['avg_rating'].iloc[0],
                        'total_weight': region_data['total_weight'].iloc[0],
                        'interaction_count': region_data['interaction_count'].iloc[0]
                    }
                else:
                    self.regional_preferences[region] = {
                        'popularity': 0.0,
                        'avg_rating': 5.0,
                        'total_weight': 0.0,
                        'interaction_count': 0
                    }
            
            logger.info(f"Regional preferences analyzed for {len(self.regional_preferences)} regions")
            
        except Exception as e:
            logger.error(f"Error analyzing regional interactions: {e}")
    
    def _create_user_profiles(self):
        """Create comprehensive user profiles"""
        try:
            for user_id in self.interactions_df['user_id'].unique():
                if pd.isna(user_id):
                    continue
                
                user_interactions = self.interactions_df[
                    self.interactions_df['user_id'] == user_id
                ].copy()
                
                if user_interactions.empty:
                    continue
                
                # Basic statistics
                total_interactions = len(user_interactions)
                avg_rating = user_interactions['rating'].mean() if user_interactions['rating'].notna().any() else 5.0
                total_weight = user_interactions['final_weight'].sum()
                
                # Regional preferences with advanced scoring
                regional_weights = user_interactions.groupby('regional_category')['final_weight'].sum().to_dict()
                regional_counts = user_interactions['regional_category'].value_counts().to_dict()
                
                # Normalize regional preferences
                total_regional_weight = sum(regional_weights.values()) if regional_weights else 1
                regional_preferences = {
                    region: weight / total_regional_weight 
                    for region, weight in regional_weights.items()
                }
                
                # Genre preferences
                all_genres = []
                for genres_str in user_interactions['genres'].dropna():
                    try:
                        genres = json.loads(genres_str) if genres_str else []
                        all_genres.extend(genres)
                    except:
                        continue
                
                genre_counts = Counter(all_genres)
                genre_preferences = dict(genre_counts.most_common(15))
                
                # Content type preferences
                content_type_preferences = user_interactions['content_type'].value_counts().to_dict()
                
                # Advanced preference features
                primary_region = max(regional_preferences.keys(), key=regional_preferences.get) if regional_preferences else 'telugu'
                telugu_preference = regional_preferences.get('telugu', 0.0)
                
                # Diversity metrics
                region_diversity = len(set(regional_counts.keys()))
                genre_diversity = len(set(all_genres))
                
                # Quality preferences
                high_rated_content = user_interactions[user_interactions['content_rating'] >= 7.0]
                quality_preference = len(high_rated_content) / total_interactions if total_interactions > 0 else 0.5
                
                # Temporal patterns
                recent_interactions = user_interactions[user_interactions['days_ago'] <= 30]
                recent_activity_ratio = len(recent_interactions) / total_interactions if total_interactions > 0 else 0.0
                
                # Trending content preference
                trending_interactions = user_interactions[user_interactions['is_trending'] == True]
                trending_preference = len(trending_interactions) / total_interactions if total_interactions > 0 else 0.0
                
                self.user_profiles[user_id] = {
                    'regional_preferences': regional_preferences,
                    'genre_preferences': genre_preferences,
                    'content_type_preferences': content_type_preferences,
                    'primary_region': primary_region,
                    'telugu_preference': telugu_preference,
                    'average_rating': avg_rating,
                    'total_interactions': total_interactions,
                    'total_weight': total_weight,
                    'region_diversity': region_diversity,
                    'genre_diversity': genre_diversity,
                    'quality_preference': quality_preference,
                    'recent_activity_ratio': recent_activity_ratio,
                    'trending_preference': trending_preference,
                    'user_cluster': None  # Will be filled after clustering
                }
            
            logger.info(f"Created {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            logger.error(f"Error creating user profiles: {e}")
    
    def _create_content_profiles(self):
        """Create comprehensive content profiles"""
        try:
            for content_id in self.content_df['id'].unique():
                content_interactions = self.interactions_df[
                    self.interactions_df['content_id'] == content_id
                ]
                
                if content_interactions.empty:
                    # Basic profile for content without interactions
                    content_row = self.content_df[self.content_df['id'] == content_id].iloc[0]
                    self.content_profiles[content_id] = {
                        'interaction_count': 0,
                        'avg_rating': content_row.get('rating', 5.0),
                        'total_weight': 0.0,
                        'user_diversity': 0,
                        'regional_appeal': 0.0,
                        'content_cluster': None
                    }
                    continue
                
                # Interaction statistics
                interaction_count = len(content_interactions)
                avg_user_rating = content_interactions['rating'].mean() if content_interactions['rating'].notna().any() else 5.0
                total_weight = content_interactions['final_weight'].sum()
                user_diversity = len(content_interactions['user_id'].unique())
                
                # Regional appeal
                content_region = content_interactions['regional_category'].iloc[0] if not content_interactions['regional_category'].empty else 'other'
                regional_boost = REGIONAL_LANGUAGES.get(content_region, {}).get('boost_factor', 1.0)
                
                # Calculate regional appeal score
                regional_appeal = (total_weight * regional_boost) / max(interaction_count, 1)
                
                self.content_profiles[content_id] = {
                    'interaction_count': interaction_count,
                    'avg_rating': avg_user_rating,
                    'total_weight': total_weight,
                    'user_diversity': user_diversity,
                    'regional_appeal': regional_appeal,
                    'content_cluster': None
                }
            
            logger.info(f"Created {len(self.content_profiles)} content profiles")
            
        except Exception as e:
            logger.error(f"Error creating content profiles: {e}")
    
    def _create_advanced_features(self):
        """Create advanced feature matrices and embeddings"""
        try:
            # Content-based features
            if not self.content_df.empty:
                content_texts = self.content_df['content_text'].fillna('')
                self.content_features = self.content_vectorizer.fit_transform(content_texts)
                
                genre_texts = self.content_df['genres_text'].fillna('')
                self.genre_features = self.genre_vectorizer.fit_transform(genre_texts)
                
                language_texts = self.content_df['languages_text'].fillna('')
                self.language_features = self.language_vectorizer.fit_transform(language_texts)
            
            # Collaborative filtering matrices
            if not self.interactions_df.empty:
                self._create_interaction_matrices()
            
            # Regional embeddings
            self._create_regional_embeddings()
            
            # Temporal and popularity features
            self._create_temporal_features()
            self._create_popularity_features()
            
            logger.info("Advanced features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating advanced features: {e}")
    
    def _create_interaction_matrices(self):
        """Create user-item and item-user interaction matrices"""
        try:
            # Create user-item matrix
            user_item_data = self.interactions_df.groupby(['user_id', 'content_id'])['final_weight'].sum().reset_index()
            
            # Get unique users and items
            users = sorted(user_item_data['user_id'].unique())
            items = sorted(user_item_data['content_id'].unique())
            
            # Create mapping dictionaries
            user_to_idx = {user: idx for idx, user in enumerate(users)}
            item_to_idx = {item: idx for idx, item in enumerate(items)}
            
            # Create sparse matrix
            n_users = len(users)
            n_items = len(items)
            
            self.user_item_matrix = lil_matrix((n_users, n_items))
            
            for _, row in user_item_data.iterrows():
                user_idx = user_to_idx[row['user_id']]
                item_idx = item_to_idx[row['content_id']]
                self.user_item_matrix[user_idx, item_idx] = row['final_weight']
            
            self.user_item_matrix = self.user_item_matrix.tocsr()
            self.item_user_matrix = self.user_item_matrix.T.tocsr()
            
            # Store mappings
            self.user_to_idx = user_to_idx
            self.item_to_idx = item_to_idx
            self.idx_to_user = {idx: user for user, idx in user_to_idx.items()}
            self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
            
            logger.info(f"Created interaction matrices: {n_users} users x {n_items} items")
            
        except Exception as e:
            logger.error(f"Error creating interaction matrices: {e}")
    
    def _create_regional_embeddings(self):
        """Create regional content embeddings"""
        try:
            if self.content_df.empty:
                return
            
            # Create regional feature matrix
            regional_features = []
            
            for _, row in self.content_df.iterrows():
                region = row['regional_category']
                region_config = REGIONAL_LANGUAGES.get(region, {})
                
                # One-hot encode region
                region_vector = [0] * len(REGIONAL_LANGUAGES)
                if region in REGIONAL_LANGUAGES:
                    region_idx = list(REGIONAL_LANGUAGES.keys()).index(region)
                    region_vector[region_idx] = 1
                
                # Add regional metadata
                region_features = region_vector + [
                    region_config.get('priority', 10),
                    region_config.get('boost_factor', 1.0),
                    region_config.get('weight', 0.1),
                    row['regional_boost'],
                    row['regional_weight']
                ]
                
                regional_features.append(region_features)
            
            self.regional_embeddings = np.array(regional_features)
            
            logger.info(f"Created regional embeddings: {self.regional_embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error creating regional embeddings: {e}")
    
    def _create_temporal_features(self):
        """Create temporal feature matrix"""
        try:
            if self.content_df.empty:
                return
            
            temporal_features = []
            
            for _, row in self.content_df.iterrows():
                features = [
                    row['days_since_release'],
                    row['release_year'],
                    row['is_recent'],
                    row['is_classic'],
                    row['is_trending'],
                    row['is_all_time_hit'],
                    row['is_new_release']
                ]
                temporal_features.append(features)
            
            self.temporal_features = np.array(temporal_features)
            self.temporal_features = self.scaler.fit_transform(self.temporal_features)
            
            logger.info(f"Created temporal features: {self.temporal_features.shape}")
            
        except Exception as e:
            logger.error(f"Error creating temporal features: {e}")
    
    def _create_popularity_features(self):
        """Create popularity feature matrix"""
        try:
            if self.content_df.empty:
                return
            
            popularity_features = []
            
            for _, row in self.content_df.iterrows():
                content_id = row['id']
                content_profile = self.content_profiles.get(content_id, {})
                
                features = [
                    row['rating'],
                    row['vote_count'],
                    row['popularity'],
                    row['quality_score'],
                    row['final_score'],
                    content_profile.get('interaction_count', 0),
                    content_profile.get('avg_rating', 5.0),
                    content_profile.get('total_weight', 0.0),
                    content_profile.get('user_diversity', 0),
                    content_profile.get('regional_appeal', 0.0)
                ]
                popularity_features.append(features)
            
            self.popularity_features = np.array(popularity_features)
            self.popularity_features = self.scaler.fit_transform(self.popularity_features)
            
            logger.info(f"Created popularity features: {self.popularity_features.shape}")
            
        except Exception as e:
            logger.error(f"Error creating popularity features: {e}")
    
    def fit(self):
        """Train all recommendation models"""
        try:
            logger.info("Training Advanced Recommendation Models...")
            
            # Fetch and process data
            if not self.fetch_data_from_database():
                logger.error("Failed to fetch data from database")
                return False
            
            if self.content_df.empty:
                logger.error("No content data available for training")
                return False
            
            # Train individual models
            training_results = {}
            
            # 1. Content-based filtering
            training_results['content_based'] = self._train_content_based_model()
            
            # 2. Collaborative filtering models
            training_results['user_based_cf'] = self._train_user_based_collaborative_filtering()
            training_results['item_based_cf'] = self._train_item_based_collaborative_filtering()
            
            # 3. Matrix factorization models
            training_results['svd'] = self._train_svd_model()
            training_results['nmf'] = self._train_nmf_model()
            
            # 4. Neural network model
            training_results['neural_network'] = self._train_neural_network()
            
            # 5. Ensemble models
            training_results['random_forest'] = self._train_random_forest()
            training_results['gradient_boosting'] = self._train_gradient_boosting()
            
            # 6. Clustering models
            training_results['user_clustering'] = self._train_user_clustering()
            training_results['content_clustering'] = self._train_content_clustering()
            
            # 7. Regional preference model
            training_results['regional_model'] = self._train_regional_model()
            
            # Store training results
            self.models_trained = training_results
            self.is_fitted = True
            
            # Save models
            self._save_models()
            
            # Log training summary
            successful_models = sum(1 for result in training_results.values() if result)
            logger.info(f"Model training completed: {successful_models}/{len(training_results)} models trained successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False
    
    def _train_content_based_model(self):
        """Train content-based filtering model"""
        try:
            if self.content_features is None:
                logger.warning("No content features available")
                return False
            
            # Calculate content similarity matrix
            self.content_similarity_matrix = cosine_similarity(self.content_features)
            
            logger.info("Content-based model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training content-based model: {e}")
            return False
    
    def _train_user_based_collaborative_filtering(self):
        """Train user-based collaborative filtering"""
        try:
            if self.user_item_matrix is None or self.user_item_matrix.shape[0] == 0:
                logger.warning("No user-item matrix available")
                return False
            
            # Calculate user similarity
            self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
            
            # Fit KNN model for user similarity
            self.user_similarity_model.fit(self.user_item_matrix)
            
            logger.info("User-based collaborative filtering trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training user-based CF: {e}")
            return False
    
    def _train_item_based_collaborative_filtering(self):
        """Train item-based collaborative filtering"""
        try:
            if self.item_user_matrix is None or self.item_user_matrix.shape[0] == 0:
                logger.warning("No item-user matrix available")
                return False
            
            # Fit KNN model for item similarity
            self.item_similarity_model.fit(self.item_user_matrix)
            
            logger.info("Item-based collaborative filtering trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training item-based CF: {e}")
            return False
    
    def _train_svd_model(self):
        """Train SVD matrix factorization model"""
        try:
            if self.user_item_matrix is None or self.user_item_matrix.shape[1] < self.svd_model.n_components:
                logger.warning("Insufficient data for SVD model")
                return False
            
            self.svd_model.fit(self.user_item_matrix)
            
            logger.info("SVD model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training SVD model: {e}")
            return False
    
    def _train_nmf_model(self):
        """Train NMF matrix factorization model"""
        try:
            if self.user_item_matrix is None or self.user_item_matrix.shape[1] < self.nmf_model.n_components:
                logger.warning("Insufficient data for NMF model")
                return False
            
            # NMF requires non-negative values
            user_item_dense = self.user_item_matrix.toarray()
            user_item_dense = np.maximum(user_item_dense, 0)
            
            self.nmf_model.fit(user_item_dense)
            
            logger.info("NMF model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training NMF model: {e}")
            return False
    
    def _train_neural_network(self):
        """Train neural network recommendation model"""
        try:
            if self.user_item_matrix is None or self.user_item_matrix.nnz < 100:
                logger.warning("Insufficient interaction data for neural network")
                return False
            
            # Prepare training data
            X_features = []
            y_ratings = []
            
            for user_id, user_profile in self.user_profiles.items():
                if user_id not in self.user_to_idx:
                    continue
                
                user_idx = self.user_to_idx[user_id]
                user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
                
                for _, interaction in user_interactions.iterrows():
                    content_id = interaction['content_id']
                    if content_id not in self.item_to_idx:
                        continue
                    
                    item_idx = self.item_to_idx[content_id]
                    
                    # Create feature vector
                    user_features = [
                        user_profile.get('telugu_preference', 0.0),
                        user_profile.get('average_rating', 5.0),
                        user_profile.get('total_interactions', 0),
                        user_profile.get('region_diversity', 0),
                        user_profile.get('genre_diversity', 0),
                        user_profile.get('quality_preference', 0.5)
                    ]
                    
                    # Get content features
                    content_row = self.content_df[self.content_df['id'] == content_id]
                    if content_row.empty:
                        continue
                    
                    content_row = content_row.iloc[0]
                    content_features = [
                        content_row['rating'],
                        content_row['popularity'],
                        content_row['regional_boost'],
                        content_row['quality_score'],
                        content_row['final_score']
                    ]
                    
                    # Combine features
                    features = user_features + content_features
                    X_features.append(features)
                    y_ratings.append(interaction['final_weight'])
            
            if len(X_features) < 50:
                logger.warning("Insufficient training data for neural network")
                return False
            
            X = np.array(X_features)
            y = np.array(y_ratings)
            
            # Train neural network
            self.neural_model.fit(X, y)
            
            logger.info(f"Neural network trained on {len(X_features)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training neural network: {e}")
            return False
    
    def _train_random_forest(self):
        """Train random forest ensemble model"""
        try:
            # Prepare training data similar to neural network
            X_features = []
            y_ratings = []
            
            for user_id, user_profile in self.user_profiles.items():
                user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
                
                for _, interaction in user_interactions.iterrows():
                    content_id = interaction['content_id']
                    content_row = self.content_df[self.content_df['id'] == content_id]
                    
                    if content_row.empty:
                        continue
                    
                    content_row = content_row.iloc[0]
                    
                    # Extended feature set for ensemble
                    features = [
                        user_profile.get('telugu_preference', 0.0),
                        user_profile.get('average_rating', 5.0),
                        user_profile.get('total_interactions', 0),
                        user_profile.get('region_diversity', 0),
                        user_profile.get('genre_diversity', 0),
                        user_profile.get('quality_preference', 0.5),
                        user_profile.get('trending_preference', 0.0),
                        content_row['rating'],
                        content_row['popularity'],
                        content_row['regional_boost'],
                        content_row['quality_score'],
                        content_row['final_score'],
                        content_row['is_trending'],
                        content_row['is_all_time_hit'],
                        content_row['is_new_release']
                    ]
                    
                    X_features.append(features)
                    y_ratings.append(interaction['final_weight'])
            
            if len(X_features) < 50:
                logger.warning("Insufficient training data for random forest")
                return False
            
            X = np.array(X_features)
            y = np.array(y_ratings)
            
            self.rf_model.fit(X, y)
            
            logger.info(f"Random forest trained on {len(X_features)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training random forest: {e}")
            return False
    
    def _train_gradient_boosting(self):
        """Train gradient boosting ensemble model"""
        try:
            # Use same data preparation as random forest
            X_features = []
            y_ratings = []
            
            for user_id, user_profile in self.user_profiles.items():
                user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
                
                for _, interaction in user_interactions.iterrows():
                    content_id = interaction['content_id']
                    content_row = self.content_df[self.content_df['id'] == content_id]
                    
                    if content_row.empty:
                        continue
                    
                    content_row = content_row.iloc[0]
                    
                    features = [
                        user_profile.get('telugu_preference', 0.0),
                        user_profile.get('average_rating', 5.0),
                        user_profile.get('total_interactions', 0),
                        user_profile.get('region_diversity', 0),
                        user_profile.get('genre_diversity', 0),
                        user_profile.get('quality_preference', 0.5),
                        user_profile.get('trending_preference', 0.0),
                        content_row['rating'],
                        content_row['popularity'],
                        content_row['regional_boost'],
                        content_row['quality_score'],
                        content_row['final_score'],
                        content_row['is_trending'],
                        content_row['is_all_time_hit'],
                        content_row['is_new_release']
                    ]
                    
                    X_features.append(features)
                    y_ratings.append(interaction['final_weight'])
            
            if len(X_features) < 50:
                logger.warning("Insufficient training data for gradient boosting")
                return False
            
            X = np.array(X_features)
            y = np.array(y_ratings)
            
            self.gbm_model.fit(X, y)
            
            logger.info(f"Gradient boosting trained on {len(X_features)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training gradient boosting: {e}")
            return False
    
    def _train_user_clustering(self):
        """Train user clustering model"""
        try:
            if len(self.user_profiles) < 10:
                logger.warning("Insufficient users for clustering")
                return False
            
            # Prepare user feature matrix
            user_features = []
            user_ids = []
            
            for user_id, profile in self.user_profiles.items():
                features = [
                    profile.get('telugu_preference', 0.0),
                    profile.get('average_rating', 5.0),
                    profile.get('total_interactions', 0),
                    profile.get('region_diversity', 0),
                    profile.get('genre_diversity', 0),
                    profile.get('quality_preference', 0.5),
                    profile.get('recent_activity_ratio', 0.0),
                    profile.get('trending_preference', 0.0)
                ]
                
                user_features.append(features)
                user_ids.append(user_id)
            
            # Adjust number of clusters based on user count
            n_clusters = min(self.user_cluster_model.n_clusters, len(user_features) // 2)
            if n_clusters < 2:
                logger.warning("Too few users for clustering")
                return False
            
            self.user_cluster_model.n_clusters = n_clusters
            
            X = np.array(user_features)
            X = self.scaler.fit_transform(X)
            
            cluster_labels = self.user_cluster_model.fit_predict(X)
            
            # Update user profiles with cluster information
            for i, user_id in enumerate(user_ids):
                self.user_profiles[user_id]['user_cluster'] = cluster_labels[i]
            
            logger.info(f"User clustering completed: {n_clusters} clusters for {len(user_ids)} users")
            return True
            
        except Exception as e:
            logger.error(f"Error training user clustering: {e}")
            return False
    
    def _train_content_clustering(self):
        """Train content clustering model"""
        try:
            if self.content_df.empty or len(self.content_df) < 20:
                logger.warning("Insufficient content for clustering")
                return False
            
            # Combine multiple feature matrices
            feature_matrices = []
            
            if self.regional_embeddings is not None:
                feature_matrices.append(self.regional_embeddings)
            
            if self.temporal_features is not None:
                feature_matrices.append(self.temporal_features)
            
            if self.popularity_features is not None:
                feature_matrices.append(self.popularity_features)
            
            if not feature_matrices:
                logger.warning("No feature matrices available for content clustering")
                return False
            
            # Combine features
            combined_features = np.hstack(feature_matrices)
            
            # Adjust number of clusters
            n_clusters = min(self.content_cluster_model.n_clusters, len(combined_features) // 5)
            if n_clusters < 2:
                logger.warning("Too few content items for clustering")
                return False
            
            self.content_cluster_model.n_clusters = n_clusters
            
            cluster_labels = self.content_cluster_model.fit_predict(combined_features)
            
            # Update content profiles with cluster information
            for i, content_id in enumerate(self.content_df['id']):
                if content_id in self.content_profiles:
                    self.content_profiles[content_id]['content_cluster'] = cluster_labels[i]
            
            logger.info(f"Content clustering completed: {n_clusters} clusters for {len(self.content_df)} items")
            return True
            
        except Exception as e:
            logger.error(f"Error training content clustering: {e}")
            return False
    
    def _train_regional_model(self):
        """Train regional preference model"""
        try:
            # This is already computed in regional preferences analysis
            # Additional regional-specific model training can be added here
            
            logger.info("Regional model training completed")
            return True
            
        except Exception as e:
            logger.error(f"Error training regional model: {e}")
            return False
    
    def get_recommendations(self, user_id, user_preferences=None, num_recommendations=20):
        """Generate hybrid recommendations using all trained models"""
        try:
            if not self.is_fitted:
                logger.warning("Models not fitted. Training now...")
                if not self.fit():
                    logger.error("Model training failed. Cannot generate recommendations.")
                    return []
            
            # Check cache first
            cache_key = f"user_{user_id}_{num_recommendations}"
            if self._check_cache(cache_key):
                logger.info(f"Returning cached recommendations for user {user_id}")
                return self.cache[cache_key]
            
            # Get or create user profile
            user_profile = self._get_enhanced_user_profile(user_id, user_preferences)
            
            if not user_profile:
                logger.warning(f"Could not create user profile for user {user_id}")
                return []
            
            # Generate recommendations using hybrid approach
            recommendations = self._generate_hybrid_recommendations(user_id, user_profile, num_recommendations)
            
            # Cache results
            self._cache_results(cache_key, recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return []
    
    def _get_enhanced_user_profile(self, user_id, user_preferences=None):
        """Get or create enhanced user profile"""
        try:
            # Start with existing profile or create new one
            user_profile = self.user_profiles.get(user_id, {})
            
            # If no existing profile and no preferences, try to infer from IP
            if not user_profile and not user_preferences:
                # Create minimal profile with Telugu preference
                user_profile = {
                    'regional_preferences': {'telugu': 0.4, 'hindi': 0.3, 'tamil': 0.2, 'english': 0.1},
                    'genre_preferences': {},
                    'content_type_preferences': {'movie': 0.7, 'tv': 0.3},
                    'primary_region': 'telugu',
                    'telugu_preference': 0.4,
                    'average_rating': 7.0,
                    'total_interactions': 0,
                    'region_diversity': 1,
                    'genre_diversity': 0,
                    'quality_preference': 0.7,
                    'recent_activity_ratio': 0.0,
                    'trending_preference': 0.3,
                    'user_cluster': 0
                }
            
            # Merge with explicit preferences if provided
            if user_preferences:
                user_profile = self._merge_user_preferences(user_profile, user_preferences)
            
            return user_profile
            
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    def _merge_user_preferences(self, user_profile, user_preferences):
        """Merge explicit user preferences with existing profile"""
        try:
            merged_profile = user_profile.copy()
            
            # Update regional preferences
            preferred_languages = user_preferences.get('preferred_languages', [])
            if preferred_languages:
                regional_prefs = merged_profile.get('regional_preferences', {})
                
                # Reset and set new preferences
                total_weight = 1.0
                weight_per_lang = total_weight / len(preferred_languages)
                
                # Clear existing and set new
                for lang in REGIONAL_LANGUAGES.keys():
                    regional_prefs[lang] = 0.0
                
                # Set preferred languages with Telugu boost
                for i, lang in enumerate(preferred_languages):
                    if lang == 'telugu':
                        regional_prefs[lang] = weight_per_lang * 1.5  # Telugu boost
                    else:
                        regional_prefs[lang] = weight_per_lang
                
                # Normalize
                total_sum = sum(regional_prefs.values())
                if total_sum > 0:
                    regional_prefs = {k: v/total_sum for k, v in regional_prefs.items()}
                
                merged_profile['regional_preferences'] = regional_prefs
                merged_profile['primary_region'] = preferred_languages[0] if preferred_languages else 'telugu'
                merged_profile['telugu_preference'] = regional_prefs.get('telugu', 0.0)
            
            # Update genre preferences
            preferred_genres = user_preferences.get('preferred_genres', [])
            if preferred_genres:
                genre_prefs = merged_profile.get('genre_preferences', {})
                for genre in preferred_genres:
                    genre_prefs[genre] = genre_prefs.get(genre, 0) + 5
                merged_profile['genre_preferences'] = genre_prefs
                merged_profile['genre_diversity'] = len(genre_prefs)
            
            # Update interaction data if provided
            interactions = user_preferences.get('interactions', [])
            if interactions:
                # Process interaction data to update profile
                total_weight = sum(self._get_interaction_weight(interaction) for interaction in interactions)
                merged_profile['total_interactions'] = len(interactions)
                merged_profile['total_weight'] = total_weight
            
            return merged_profile
            
        except Exception as e:
            logger.error(f"Error merging user preferences: {e}")
            return user_profile
    
    def _get_interaction_weight(self, interaction):
        """Calculate weight for an interaction"""
        interaction_weights = {
            'view': 1.0,
            'like': 2.5,
            'favorite': 4.0,
            'watchlist': 3.0,
            'search': 0.8
        }
        
        interaction_type = interaction.get('interaction_type', 'view')
        rating = interaction.get('rating')
        
        if rating:
            return float(rating) / 2.0
        else:
            return interaction_weights.get(interaction_type, 1.0)
    
    def _generate_hybrid_recommendations(self, user_id, user_profile, num_recommendations):
        """Generate hybrid recommendations using multiple algorithms"""
        try:
            algorithm_scores = defaultdict(lambda: defaultdict(float))
            algorithm_weights = {
                'content_based': 0.2,
                'user_collaborative': 0.15,
                'item_collaborative': 0.15,
                'svd': 0.1,
                'neural_network': 0.1,
                'random_forest': 0.1,
                'regional_boost': 0.15,
                'popularity': 0.05
            }
            
            # 1. Content-based recommendations
            if self.models_trained.get('content_based'):
                content_scores = self._get_content_based_scores(user_profile)
                for content_id, score in content_scores.items():
                    algorithm_scores['content_based'][content_id] = score
            
            # 2. User-based collaborative filtering
            if self.models_trained.get('user_based_cf'):
                user_cf_scores = self._get_user_collaborative_scores(user_id, user_profile)
                for content_id, score in user_cf_scores.items():
                    algorithm_scores['user_collaborative'][content_id] = score
            
            # 3. Item-based collaborative filtering
            if self.models_trained.get('item_based_cf'):
                item_cf_scores = self._get_item_collaborative_scores(user_id, user_profile)
                for content_id, score in item_cf_scores.items():
                    algorithm_scores['item_collaborative'][content_id] = score
            
            # 4. SVD-based recommendations
            if self.models_trained.get('svd'):
                svd_scores = self._get_svd_scores(user_id, user_profile)
                for content_id, score in svd_scores.items():
                    algorithm_scores['svd'][content_id] = score
            
            # 5. Neural network recommendations
            if self.models_trained.get('neural_network'):
                nn_scores = self._get_neural_network_scores(user_id, user_profile)
                for content_id, score in nn_scores.items():
                    algorithm_scores['neural_network'][content_id] = score
            
            # 6. Random forest recommendations
            if self.models_trained.get('random_forest'):
                rf_scores = self._get_random_forest_scores(user_id, user_profile)
                for content_id, score in rf_scores.items():
                    algorithm_scores['random_forest'][content_id] = score
            
            # 7. Regional preference boost
            regional_scores = self._get_regional_scores(user_profile)
            for content_id, score in regional_scores.items():
                algorithm_scores['regional_boost'][content_id] = score
            
            # 8. Popularity-based scores
            popularity_scores = self._get_popularity_scores()
            for content_id, score in popularity_scores.items():
                algorithm_scores['popularity'][content_id] = score
            
            # Combine scores from all algorithms
            final_scores = self._combine_algorithm_scores(algorithm_scores, algorithm_weights)
            
            # Apply diversification
            diversified_scores = self._apply_diversification(final_scores, user_profile)
            
            # Filter out already seen content
            filtered_scores = self._filter_seen_content(diversified_scores, user_id)
            
            # Sort and format recommendations
            recommendations = self._format_recommendations(filtered_scores, num_recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendation generation: {e}")
            return []
    
    def _get_content_based_scores(self, user_profile):
        """Get content-based recommendation scores"""
        scores = defaultdict(float)
        
        try:
            if self.content_similarity_matrix is None:
                return scores
            
            genre_prefs = user_profile.get('genre_preferences', {})
            regional_prefs = user_profile.get('regional_preferences', {})
            
            for idx, row in self.content_df.iterrows():
                content_id = row['id']
                
                # Base quality score
                base_score = row.get('quality_score', 0.5)
                
                # Genre matching
                content_genres = row.get('genres_list', [])
                genre_boost = 1.0
                for genre in content_genres:
                    if genre in genre_prefs:
                        genre_boost += genre_prefs[genre] * 0.1
                
                # Regional matching
                content_region = row.get('regional_category', 'other')
                regional_boost = regional_prefs.get(content_region, 0.1) * 2.0
                
                # Final score
                scores[content_id] = base_score * genre_boost * (1 + regional_boost)
            
        except Exception as e:
            logger.error(f"Error in content-based scoring: {e}")
        
        return scores
    
    def _get_user_collaborative_scores(self, user_id, user_profile):
        """Get user-based collaborative filtering scores"""
        scores = defaultdict(float)
        
        try:
            if self.user_similarity_matrix is None or user_id not in self.user_to_idx:
                return scores
            
            user_idx = self.user_to_idx[user_id]
            
            # Find similar users
            user_similarities = self.user_similarity_matrix[user_idx]
            similar_user_indices = np.argsort(user_similarities)[::-1][1:21]  # Top 20 similar users
            
            for similar_user_idx in similar_user_indices:
                if user_similarities[similar_user_idx] < 0.1:  # Minimum similarity threshold
                    continue
                
                similar_user_id = self.idx_to_user.get(similar_user_idx)
                if not similar_user_id:
                    continue
                
                # Get content liked by similar user
                similar_user_interactions = self.interactions_df[
                    self.interactions_df['user_id'] == similar_user_id
                ]
                
                for _, interaction in similar_user_interactions.iterrows():
                    content_id = interaction['content_id']
                    weight = interaction['final_weight']
                    similarity = user_similarities[similar_user_idx]
                    
                    scores[content_id] += weight * similarity * 0.1
            
        except Exception as e:
            logger.error(f"Error in user collaborative scoring: {e}")
        
        return scores
    
    def _get_item_collaborative_scores(self, user_id, user_profile):
        """Get item-based collaborative filtering scores"""
        scores = defaultdict(float)
        
        try:
            # Get user's interaction history
            user_interactions = self.interactions_df[
                self.interactions_df['user_id'] == user_id
            ]
            
            if user_interactions.empty:
                return scores
            
            # For each item user interacted with, find similar items
            for _, interaction in user_interactions.iterrows():
                content_id = interaction['content_id']
                user_weight = interaction['final_weight']
                
                if content_id not in self.item_to_idx:
                    continue
                
                item_idx = self.item_to_idx[content_id]
                
                # Find similar items
                try:
                    item_vector = self.item_user_matrix[item_idx].reshape(1, -1)
                    distances, indices = self.item_similarity_model.kneighbors(item_vector, n_neighbors=11)
                    
                    for i, similar_item_idx in enumerate(indices[0][1:]):  # Skip self
                        similarity = 1 / (1 + distances[0][i])  # Convert distance to similarity
                        similar_content_id = self.idx_to_item.get(similar_item_idx)
                        
                        if similar_content_id and similarity > 0.1:
                            scores[similar_content_id] += user_weight * similarity * 0.1
                            
                except Exception:
                    continue
            
        except Exception as e:
            logger.error(f"Error in item collaborative scoring: {e}")
        
        return scores
    
    def _get_svd_scores(self, user_id, user_profile):
        """Get SVD-based recommendation scores"""
        scores = defaultdict(float)
        
        try:
            if not hasattr(self.svd_model, 'components_') or user_id not in self.user_to_idx:
                return scores
            
            user_idx = self.user_to_idx[user_id]
            
            # Get user's latent factors
            user_vector = self.user_item_matrix[user_idx].toarray().flatten()
            user_factors = self.svd_model.transform([user_vector])[0]
            
            # Calculate scores for all items
            item_factors = self.svd_model.components_.T
            predicted_ratings = np.dot(user_factors, item_factors.T)
            
            # Map back to content IDs
            for item_idx, score in enumerate(predicted_ratings):
                content_id = self.idx_to_item.get(item_idx)
                if content_id:
                    scores[content_id] = float(score)
            
        except Exception as e:
            logger.error(f"Error in SVD scoring: {e}")
        
        return scores
    
    def _get_neural_network_scores(self, user_id, user_profile):
        """Get neural network-based recommendation scores"""
        scores = defaultdict(float)
        
        try:
            if not hasattr(self.neural_model, 'predict'):
                return scores
            
            # Prepare user features
            user_features = [
                user_profile.get('telugu_preference', 0.0),
                user_profile.get('average_rating', 5.0),
                user_profile.get('total_interactions', 0),
                user_profile.get('region_diversity', 0),
                user_profile.get('genre_diversity', 0),
                user_profile.get('quality_preference', 0.5)
            ]
            
            # Score all content
            for _, row in self.content_df.iterrows():
                content_id = row['id']
                
                content_features = [
                    row['rating'],
                    row['popularity'],
                    row['regional_boost'],
                    row['quality_score'],
                    row['final_score']
                ]
                
                # Combine features
                features = np.array([user_features + content_features])
                
                try:
                    predicted_score = self.neural_model.predict(features)[0]
                    scores[content_id] = float(predicted_score)
                except:
                    continue
            
        except Exception as e:
            logger.error(f"Error in neural network scoring: {e}")
        
        return scores
    
    def _get_random_forest_scores(self, user_id, user_profile):
        """Get random forest-based recommendation scores"""
        scores = defaultdict(float)
        
        try:
            if not hasattr(self.rf_model, 'predict'):
                return scores
            
            # Prepare user features (extended)
            user_features = [
                user_profile.get('telugu_preference', 0.0),
                user_profile.get('average_rating', 5.0),
                user_profile.get('total_interactions', 0),
                user_profile.get('region_diversity', 0),
                user_profile.get('genre_diversity', 0),
                user_profile.get('quality_preference', 0.5),
                user_profile.get('trending_preference', 0.0)
            ]
            
            # Score all content
            for _, row in self.content_df.iterrows():
                content_id = row['id']
                
                content_features = [
                    row['rating'],
                    row['popularity'],
                    row['regional_boost'],
                    row['quality_score'],
                    row['final_score'],
                    row['is_trending'],
                    row['is_all_time_hit'],
                    row['is_new_release']
                ]
                
                # Combine features
                features = np.array([user_features + content_features])
                
                try:
                    predicted_score = self.rf_model.predict(features)[0]
                    scores[content_id] = float(predicted_score)
                except:
                    continue
            
        except Exception as e:
            logger.error(f"Error in random forest scoring: {e}")
        
        return scores
    
    def _get_regional_scores(self, user_profile):
        """Get regional preference-based scores"""
        scores = defaultdict(float)
        
        try:
            regional_prefs = user_profile.get('regional_preferences', {})
            primary_region = user_profile.get('primary_region', 'telugu')
            
            for _, row in self.content_df.iterrows():
                content_id = row['id']
                content_region = row.get('regional_category', 'other')
                
                # Base regional score
                regional_score = regional_prefs.get(content_region, 0.1)
                
                # Extra boost for Telugu content
                if content_region == 'telugu':
                    regional_score *= 1.5
                
                # Boost for primary region
                if content_region == primary_region:
                    regional_score *= 1.3
                
                # Content quality boost
                quality_boost = row.get('regional_boost', 1.0)
                
                scores[content_id] = regional_score * quality_boost
            
        except Exception as e:
            logger.error(f"Error in regional scoring: {e}")
        
        return scores
    
    def _get_popularity_scores(self):
        """Get popularity-based scores"""
        scores = defaultdict(float)
        
        try:
            for _, row in self.content_df.iterrows():
                content_id = row['id']
                
                # Combine multiple popularity signals
                pop_score = (
                    row.get('popularity', 0) * 0.3 +
                    row.get('vote_count', 0) * 0.2 +
                    row.get('rating', 5.0) * 0.3 +
                    self.content_profiles.get(content_id, {}).get('interaction_count', 0) * 0.2
                )
                
                scores[content_id] = pop_score
            
            # Normalize scores
            if scores:
                max_score = max(scores.values())
                if max_score > 0:
                    scores = {k: v / max_score for k, v in scores.items()}
            
        except Exception as e:
            logger.error(f"Error in popularity scoring: {e}")
        
        return scores
    
    def _combine_algorithm_scores(self, algorithm_scores, algorithm_weights):
        """Combine scores from multiple algorithms"""
        combined_scores = defaultdict(float)
        
        try:
            # Get all content IDs
            all_content_ids = set()
            for algorithm_name, scores in algorithm_scores.items():
                all_content_ids.update(scores.keys())
            
            # Combine scores with weights
            for content_id in all_content_ids:
                total_score = 0.0
                total_weight = 0.0
                
                for algorithm_name, weight in algorithm_weights.items():
                    if algorithm_name in algorithm_scores:
                        score = algorithm_scores[algorithm_name].get(content_id, 0.0)
                        total_score += score * weight
                        total_weight += weight
                
                if total_weight > 0:
                    combined_scores[content_id] = total_score / total_weight
            
        except Exception as e:
            logger.error(f"Error combining algorithm scores: {e}")
        
        return combined_scores
    
    def _apply_diversification(self, scores, user_profile):
        """Apply diversification to avoid filter bubbles"""
        try:
            diversified_scores = scores.copy()
            
            # Get user's diversity preference
            region_diversity = user_profile.get('region_diversity', 1)
            genre_diversity = user_profile.get('genre_diversity', 0)
            
            # If user shows high diversity, boost diverse content
            if region_diversity > 2 or genre_diversity > 5:
                for content_id, score in scores.items():
                    content_row = self.content_df[self.content_df['id'] == content_id]
                    if content_row.empty:
                        continue
                    
                    content_row = content_row.iloc[0]
                    content_region = content_row.get('regional_category', 'other')
                    
                    # Boost non-primary region content for diverse users
                    primary_region = user_profile.get('primary_region', 'telugu')
                    if content_region != primary_region:
                        diversified_scores[content_id] = score * 1.2
            
        except Exception as e:
            logger.error(f"Error in diversification: {e}")
            return scores
        
        return diversified_scores
    
    def _filter_seen_content(self, scores, user_id):
        """Filter out content user has already seen"""
        try:
            # Get user's interaction history
            user_interactions = self.interactions_df[
                self.interactions_df['user_id'] == user_id
            ]
            
            seen_content = set(user_interactions['content_id'].tolist())
            
            # Filter out seen content
            filtered_scores = {
                content_id: score for content_id, score in scores.items()
                if content_id not in seen_content
            }
            
            return filtered_scores
            
        except Exception as e:
            logger.error(f"Error filtering seen content: {e}")
            return scores
    
    def _format_recommendations(self, scores, num_recommendations):
        """Format final recommendations"""
        try:
            # Sort by score
            sorted_recommendations = sorted(
                scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:num_recommendations]
            
            recommendations = []
            
            for content_id, score in sorted_recommendations:
                content_row = self.content_df[self.content_df['id'] == content_id]
                
                if content_row.empty:
                    continue
                
                content_row = content_row.iloc[0]
                
                recommendation = {
                    'content_id': int(content_id),
                    'score': float(score),
                    'title': content_row.get('title', 'Unknown'),
                    'regional_category': content_row.get('regional_category', 'other'),
                    'content_type': content_row.get('content_type', 'movie'),
                    'rating': float(content_row.get('rating', 0)),
                    'genres': content_row.get('genres_list', []),
                    'is_telugu': content_row.get('regional_category') == 'telugu',
                    'is_trending': bool(content_row.get('is_trending', False)),
                    'is_all_time_hit': bool(content_row.get('is_all_time_hit', False)),
                    'is_new_release': bool(content_row.get('is_new_release', False)),
                    'reason': self._generate_recommendation_reason(content_row, score)
                }
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error formatting recommendations: {e}")
            return []
    
    def _generate_recommendation_reason(self, content_row, score):
        """Generate explanation for recommendation"""
        reasons = []
        
        try:
            # Regional reason
            region = content_row.get('regional_category', 'other')
            if region == 'telugu':
                reasons.append("Telugu content priority")
            elif region in REGIONAL_LANGUAGES:
                reasons.append(f"{REGIONAL_LANGUAGES[region]['industry']} content")
            
            # Content category reasons
            if content_row.get('is_trending'):
                reasons.append("Trending now")
            if content_row.get('is_all_time_hit'):
                reasons.append("All-time hit")
            if content_row.get('is_new_release'):
                reasons.append("New release")
            
            # Quality reason
            if content_row.get('rating', 0) >= 8.0:
                reasons.append("Highly rated")
            
            return "; ".join(reasons[:2]) if reasons else "Recommended for you"
            
        except:
            return "Recommended for you"
    
    def _check_cache(self, cache_key):
        """Check if cached results are available and valid"""
        try:
            if cache_key not in self.cache:
                return False
            
            # Check if cache is expired
            cache_time = self.cache_timestamps.get(cache_key)
            if not cache_time:
                return False
            
            hours_elapsed = (datetime.now() - cache_time).total_seconds() / 3600
            return hours_elapsed < CACHE_EXPIRY_HOURS
            
        except:
            return False
    
    def _cache_results(self, cache_key, results):
        """Cache recommendation results"""
        try:
            self.cache[cache_key] = results
            self.cache_timestamps[cache_key] = datetime.now()
            
            # Clean old cache entries
            self._clean_cache()
            
        except Exception as e:
            logger.error(f"Error caching results: {e}")
    
    def _clean_cache(self):
        """Clean expired cache entries"""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for key, timestamp in self.cache_timestamps.items():
                hours_elapsed = (current_time - timestamp).total_seconds() / 3600
                if hours_elapsed >= CACHE_EXPIRY_HOURS:
                    expired_keys.append(key)
            
            for key in expired_keys:
                if key in self.cache:
                    del self.cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
            
            if expired_keys:
                logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error cleaning cache: {e}")
    
    def get_similar_content(self, content_id, num_similar=10):
        """Get similar content using content-based filtering"""
        try:
            if not self.is_fitted or self.content_similarity_matrix is None:
                return []
            
            content_idx = None
            for idx, row in self.content_df.iterrows():
                if row['id'] == content_id:
                    content_idx = idx
                    break
            
            if content_idx is None:
                return []
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_similarity_matrix[content_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar content (excluding the content itself)
            similar_content = []
            for idx, score in sim_scores[1:num_similar+1]:
                if idx < len(self.content_df):
                    content_row = self.content_df.iloc[idx]
                    similar_content.append({
                        'content_id': int(content_row['id']),
                        'similarity_score': float(score),
                        'title': content_row.get('title', 'Unknown'),
                        'regional_category': content_row.get('regional_category', 'other'),
                        'rating': float(content_row.get('rating', 0))
                    })
            
            return similar_content
            
        except Exception as e:
            logger.error(f"Error getting similar content: {e}")
            return []
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            models_data = {
                'content_vectorizer': self.content_vectorizer,
                'genre_vectorizer': self.genre_vectorizer,
                'language_vectorizer': self.language_vectorizer,
                'user_similarity_model': self.user_similarity_model,
                'item_similarity_model': self.item_similarity_model,
                'svd_model': self.svd_model,
                'nmf_model': self.nmf_model,
                'neural_model': self.neural_model,
                'rf_model': self.rf_model,
                'gbm_model': self.gbm_model,
                'user_cluster_model': self.user_cluster_model,
                'content_cluster_model': self.content_cluster_model,
                'scaler': self.scaler,
                'min_max_scaler': self.min_max_scaler,
                'content_similarity_matrix': self.content_similarity_matrix,
                'user_similarity_matrix': self.user_similarity_matrix,
                'regional_embeddings': self.regional_embeddings,
                'temporal_features': self.temporal_features,
                'popularity_features': self.popularity_features,
                'user_profiles': self.user_profiles,
                'content_profiles': self.content_profiles,
                'regional_preferences': self.regional_preferences,
                'user_to_idx': getattr(self, 'user_to_idx', {}),
                'item_to_idx': getattr(self, 'item_to_idx', {}),
                'idx_to_user': getattr(self, 'idx_to_user', {}),
                'idx_to_item': getattr(self, 'idx_to_item', {}),
                'models_trained': self.models_trained,
                'is_fitted': self.is_fitted
            }
            
            model_path = os.path.join(MODEL_CACHE_DIR, 'advanced_recommendation_models.pkl')
            joblib.dump(models_data, model_path)
            
            logger.info(f"Models saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            model_path = os.path.join(MODEL_CACHE_DIR, 'advanced_recommendation_models.pkl')
            
            if os.path.exists(model_path):
                models_data = joblib.load(model_path)
                
                # Load all models and data
                for key, value in models_data.items():
                    setattr(self, key, value)
                
                logger.info("Models loaded successfully")
                return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
        
        return False

# Initialize the recommendation engine
recommendation_engine = AdvancedRecommendationEngine()

# Initialize models flag for one-time initialization
models_initialized = False

def ensure_models_initialized():
    """Ensure models are initialized exactly once"""
    global models_initialized
    if not models_initialized:
        try:
            recommendation_engine.initialize_models()
            models_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            models_initialized = True  # Prevent infinite retry

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Advanced ML Recommendation Service',
        'version': '3.0.0',
        'features': [
            'Hybrid recommendation system',
            'Multiple collaborative filtering approaches',
            'Content-based filtering with NLP',
            'Matrix factorization (SVD, NMF)',
            'Neural network recommendations',
            'Ensemble methods (RF, GBM)',
            'User and content clustering',
            'Regional and language preferences',
            'Advanced feature extraction',
            'Recommendation diversification',
            'Caching and batch processing',
            'Cold start handling'
        ],
        'algorithms': [
            'Content-based filtering',
            'User-based collaborative filtering',
            'Item-based collaborative filtering',
            'SVD matrix factorization',
            'NMF matrix factorization',
            'Neural network (MLP)',
            'Random forest ensemble',
            'Gradient boosting ensemble',
            'K-means clustering',
            'Regional preference modeling'
        ],
        'models_fitted': recommendation_engine.is_fitted,
        'supported_languages': list(REGIONAL_LANGUAGES.keys()),
        'telugu_priority': True,
        'no_default_recommendations': True
    })

@app.route('/api/train', methods=['POST'])
def train_models():
    """Train all recommendation models"""
    try:
        ensure_models_initialized()
        logger.info("Starting comprehensive model training...")
        success = recommendation_engine.fit()
        
        if success:
            training_summary = {
                'status': 'success',
                'message': 'All models trained successfully',
                'models_trained': recommendation_engine.models_trained,
                'content_count': len(recommendation_engine.content_df) if recommendation_engine.content_df is not None else 0,
                'interaction_count': len(recommendation_engine.interactions_df) if recommendation_engine.interactions_df is not None else 0,
                'user_profiles': len(recommendation_engine.user_profiles),
                'content_profiles': len(recommendation_engine.content_profiles),
                'regional_preferences': recommendation_engine.regional_preferences,
                'training_timestamp': datetime.utcnow().isoformat()
            }
            
            return jsonify(training_summary)
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model training failed - insufficient data'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in training endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Training failed: {str(e)}'
        }), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get personalized recommendations using hybrid approach"""
    try:
        ensure_models_initialized()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        user_preferences = {
            'preferred_languages': data.get('preferred_languages', []),
            'preferred_genres': data.get('preferred_genres', []),
            'interactions': data.get('interactions', [])
        }
        
        num_recommendations = data.get('num_recommendations', 20)
        
        # Get recommendations using advanced hybrid approach
        recommendations = recommendation_engine.get_recommendations(
            user_id=user_id,
            user_preferences=user_preferences,
            num_recommendations=num_recommendations
        )
        
        if not recommendations:
            return jsonify({
                'status': 'no_recommendations',
                'message': 'No recommendations available - insufficient data or user preferences',
                'user_id': user_id,
                'recommendations': [],
                'total_recommendations': 0
            })
        
        # Calculate Telugu content statistics
        telugu_count = sum(1 for r in recommendations if r.get('is_telugu', False))
        trending_count = sum(1 for r in recommendations if r.get('is_trending', False))
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'telugu_recommendations': telugu_count,
            'trending_recommendations': trending_count,
            'algorithm_info': {
                'hybrid_approach': True,
                'algorithms_used': list(recommendation_engine.models_trained.keys()),
                'telugu_priority': True,
                'diversification_applied': True
            },
            'model_version': '3.0.0',
            'generation_timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Recommendation generation failed: {str(e)}'
        }), 500

@app.route('/api/similar', methods=['POST'])
def get_similar_content():
    """Get similar content using content-based filtering"""
    try:
        ensure_models_initialized()
        data = request.get_json()
        
        if not data or 'content_id' not in data:
            return jsonify({'error': 'content_id is required'}), 400
        
        content_id = data['content_id']
        num_similar = data.get('num_similar', 10)
        
        similar_content = recommendation_engine.get_similar_content(
            content_id=content_id,
            num_similar=num_similar
        )
        
        if not similar_content:
            return jsonify({
                'status': 'no_similar_content',
                'message': 'No similar content found',
                'content_id': content_id,
                'similar_content': []
            })
        
        return jsonify({
            'status': 'success',
            'content_id': content_id,
            'similar_content': similar_content,
            'total_similar': len(similar_content),
            'algorithm': 'content_based_filtering'
        })
        
    except Exception as e:
        logger.error(f"Error in similar content endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Similar content generation failed: {str(e)}'
        }), 500

@app.route('/api/regional/recommendations', methods=['POST'])
def get_regional_recommendations():
    """Get recommendations optimized for specific regional language"""
    try:
        ensure_models_initialized()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        language = data.get('language', 'telugu')
        user_id = data.get('user_id', f'regional_user_{language}')
        num_recommendations = data.get('num_recommendations', 20)
        
        if language not in REGIONAL_LANGUAGES:
            return jsonify({'error': f'Unsupported language: {language}'}), 400
        
        # Create language-focused user preferences
        user_preferences = {
            'preferred_languages': [language] + data.get('preferred_languages', []),
            'preferred_genres': data.get('preferred_genres', []),
            'interactions': data.get('interactions', [])
        }
        
        # Get recommendations with regional optimization
        recommendations = recommendation_engine.get_recommendations(
            user_id=user_id,
            user_preferences=user_preferences,
            num_recommendations=num_recommendations
        )
        
        if not recommendations:
            return jsonify({
                'status': 'no_recommendations',
                'message': f'No {language} recommendations available',
                'language': language,
                'recommendations': []
            })
        
        # Filter and boost regional content
        regional_recommendations = []
        for rec in recommendations:
            if rec.get('regional_category') == language:
                rec['regional_match'] = True
                rec['score'] *= REGIONAL_LANGUAGES[language]['boost_factor']
            else:
                rec['regional_match'] = False
            regional_recommendations.append(rec)
        
        # Re-sort after regional boosting
        regional_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        regional_count = sum(1 for r in regional_recommendations if r.get('regional_match', False))
        
        return jsonify({
            'status': 'success',
            'language': language,
            'language_info': REGIONAL_LANGUAGES[language],
            'recommendations': regional_recommendations,
            'total_recommendations': len(regional_recommendations),
            'regional_matches': regional_count,
            'optimization': f'{language}_priority'
        })
        
    except Exception as e:
        logger.error(f"Error in regional recommendations endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Regional recommendation generation failed: {str(e)}'
        }), 500

@app.route('/api/analytics/user', methods=['POST'])
def get_user_analytics():
    """Get comprehensive user analytics and preferences"""
    try:
        ensure_models_initialized()
        data = request.get_json()
        
        if not data or 'user_id' not in data:
            return jsonify({'error': 'user_id is required'}), 400
        
        user_id = data['user_id']
        
        user_profile = recommendation_engine.user_profiles.get(user_id, {})
        
        if not user_profile:
            return jsonify({
                'status': 'no_profile',
                'user_id': user_id,
                'message': 'No user profile available',
                'recommendation': 'User needs to interact with content to build profile'
            })
        
        # Calculate additional insights
        regional_prefs = user_profile.get('regional_preferences', {})
        top_region = max(regional_prefs.keys(), key=regional_prefs.get) if regional_prefs else 'telugu'
        
        genre_prefs = user_profile.get('genre_preferences', {})
        top_genres = sorted(genre_prefs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'profile_summary': {
                'primary_region': user_profile.get('primary_region', 'telugu'),
                'top_region_preference': top_region,
                'telugu_affinity': user_profile.get('telugu_preference', 0.0),
                'average_rating': user_profile.get('average_rating', 5.0),
                'total_interactions': user_profile.get('total_interactions', 0),
                'diversity_scores': {
                    'regional': user_profile.get('region_diversity', 0),
                    'genre': user_profile.get('genre_diversity', 0)
                },
                'preferences': {
                    'quality_focused': user_profile.get('quality_preference', 0.5),
                    'trending_focused': user_profile.get('trending_preference', 0.0),
                    'recent_activity': user_profile.get('recent_activity_ratio', 0.0)
                }
            },
            'detailed_preferences': {
                'regional_breakdown': regional_prefs,
                'top_genres': dict(top_genres),
                'content_types': user_profile.get('content_type_preferences', {}),
                'user_cluster': user_profile.get('user_cluster')
            },
            'insights': {
                'most_preferred_region': top_region,
                'regional_diversity': len(regional_prefs),
                'genre_diversity': len(genre_prefs),
                'engagement_level': 'high' if user_profile.get('total_interactions', 0) > 10 else 'low'
            }
        })
        
    except Exception as e:
        logger.error(f"Error in user analytics endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': f'User analytics failed: {str(e)}'
        }), 500

@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    """Refresh data and retrain models"""
    try:
        ensure_models_initialized()
        logger.info("Refreshing data and retraining models...")
        
        # Clear cache
        recommendation_engine.cache.clear()
        recommendation_engine.cache_timestamps.clear()
        
        # Fetch latest data and retrain
        success = recommendation_engine.fit()
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Data refreshed and models retrained successfully',
                'timestamp': datetime.utcnow().isoformat(),
                'models_trained': recommendation_engine.models_trained,
                'cache_cleared': True,
                'content_count': len(recommendation_engine.content_df) if recommendation_engine.content_df is not None else 0,
                'interaction_count': len(recommendation_engine.interactions_df) if recommendation_engine.interactions_df is not None else 0
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model retraining failed - insufficient data'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in refresh endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Data refresh failed: {str(e)}'
        }), 500

@app.route('/api/models/status', methods=['GET'])
def get_model_status():
    """Get detailed model training status"""
    try:
        ensure_models_initialized()
        return jsonify({
            'is_fitted': recommendation_engine.is_fitted,
            'models_trained': recommendation_engine.models_trained,
            'data_summary': {
                'content_count': len(recommendation_engine.content_df) if recommendation_engine.content_df is not None else 0,
                'interaction_count': len(recommendation_engine.interactions_df) if recommendation_engine.interactions_df is not None else 0,
                'user_profiles': len(recommendation_engine.user_profiles),
                'content_profiles': len(recommendation_engine.content_profiles)
            },
            'regional_summary': recommendation_engine.regional_preferences,
            'cache_stats': {
                'cached_items': len(recommendation_engine.cache),
                'cache_hits': 'Not tracked'  # Could be implemented
            },
            'initialization_status': models_initialized
        })
    except Exception as e:
        logger.error(f"Error in model status endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Model status retrieval failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info("🤖 Starting Advanced ML Recommendation Service")
    logger.info("🎯 Multi-Algorithm Hybrid Approach with Telugu Priority")
    logger.info("🔧 Algorithms: Content-based, Collaborative, SVD, Neural Networks, Ensembles")
    logger.info("🚫 No default recommendations - Data-driven only")
    
    # Initialize models on startup
    try:
        recommendation_engine.initialize_models()
        models_initialized = True
        logger.info("✅ Models initialized successfully on startup")
    except Exception as e:
        logger.error(f"❌ Failed to initialize models on startup: {e}")
        models_initialized = True  # Prevent infinite retry
    
    app.run(host='0.0.0.0', port=port, debug=debug)