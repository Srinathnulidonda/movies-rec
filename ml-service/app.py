# ml-service/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import logging
import time
import hashlib
import pickle
import math
import threading
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
from functools import lru_cache, wraps
import requests
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Dict, List, Tuple, Optional, Any

# Advanced ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, rbf_kernel
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine, euclidean, manhattan
from scipy.special import expit
import surprise
from surprise import Dataset, Reader, SVD, NMF as SurpriseNMF, KNNBasic, KNNWithMeans, BaselineOnly
from surprise.model_selection import train_test_split as surprise_train_test_split, cross_validate
from sentence_transformers import SentenceTransformer
import networkx as nx
from textblob import TextBlob
import re
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'ml-service-ultra-secret-key')
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:5000')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
models_lock = threading.RLock()
models_initialized = False
last_model_update = None
executor = ThreadPoolExecutor(max_workers=4)

# Advanced Model Store
class AdvancedModelStore:
    def __init__(self):
        # Data storage
        self.content_df = None
        self.interactions_df = None
        self.users_df = None
        
        # Feature matrices
        self.content_tfidf_matrix = None
        self.content_embeddings = None
        self.user_item_matrix = None
        self.item_features_matrix = None
        
        # Models
        self.collaborative_models = {}
        self.content_similarity_matrix = None
        self.user_similarity_matrix = None
        self.semantic_model = None
        
        # Advanced analytics
        self.trending_analyzer = None
        self.user_profiler = None
        self.content_clusterer = None
        
        # Caches
        self.recommendation_cache = {}
        self.similarity_cache = {}
        self.user_profiles_cache = {}
        
        # Metadata
        self.genre_mappings = {}
        self.language_mappings = {}
        self.content_metadata = {}
        self.user_metadata = {}
        
        # Performance metrics
        self.model_performance = {}
        self.last_update = None
        self.update_count = 0
        
    def is_initialized(self):
        return (self.content_df is not None and 
                len(self.content_df) > 0 and
                self.collaborative_models.get('svd') is not None)

# Global model store
model_store = AdvancedModelStore()

# Advanced Data Processor
class AdvancedDataProcessor:
    """Advanced data processing with comprehensive feature engineering"""
    
    @staticmethod
    def fetch_comprehensive_data():
        """Fetch all data from backend with error handling and retries"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching data from backend (attempt {attempt + 1})")
                
                # Fetch content data
                content_response = requests.get(f"{BACKEND_URL}/api/admin/content/all", timeout=30)
                if content_response.status_code == 200:
                    content_data = content_response.json().get('content', [])
                else:
                    logger.warning(f"Content fetch failed with status {content_response.status_code}")
                    content_data = []
                
                # Fetch interactions data
                interactions_response = requests.get(f"{BACKEND_URL}/api/admin/interactions/all", timeout=30)
                if interactions_response.status_code == 200:
                    interactions_data = interactions_response.json().get('interactions', [])
                else:
                    logger.warning(f"Interactions fetch failed with status {interactions_response.status_code}")
                    interactions_data = []
                
                # Fetch users data
                users_response = requests.get(f"{BACKEND_URL}/api/admin/users/all", timeout=30)
                if users_response.status_code == 200:
                    users_data = users_response.json().get('users', [])
                else:
                    logger.warning(f"Users fetch failed with status {users_response.status_code}")
                    users_data = []
                
                # If we got some data, proceed
                if content_data or interactions_data:
                    logger.info(f"Successfully fetched: {len(content_data)} content, {len(interactions_data)} interactions, {len(users_data)} users")
                    return content_data, interactions_data, users_data
                
                # If no data and not last attempt, wait and retry
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    
            except Exception as e:
                logger.error(f"Data fetch attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        # Fallback to comprehensive sample data
        logger.warning("Using comprehensive sample data")
        return AdvancedDataProcessor.create_comprehensive_sample_data()
    
    @staticmethod
    def create_comprehensive_sample_data():
        """Create comprehensive sample data with realistic patterns"""
        logger.info("Creating comprehensive sample data for ML service")
        
        # Advanced content data
        content_data = []
        
        # Realistic genre combinations
        genre_combinations = [
            ['Action', 'Adventure'], ['Comedy', 'Romance'], ['Drama', 'Thriller'],
            ['Horror', 'Mystery'], ['Sci-Fi', 'Action'], ['Animation', 'Family'],
            ['Crime', 'Drama'], ['Fantasy', 'Adventure'], ['Romance', 'Drama'],
            ['Thriller', 'Crime'], ['Comedy', 'Family'], ['Documentary'],
            ['Musical', 'Romance'], ['War', 'Drama'], ['Western', 'Action'],
            ['Biography', 'Drama'], ['History', 'War'], ['Sport', 'Drama']
        ]
        
        languages = ['english', 'hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'japanese', 'korean', 'spanish', 'french']
        content_types = ['movie', 'tv', 'anime']
        
        # Create realistic movie titles and overviews
        for i in range(1, 1001):  # 1000 content items
            genres = genre_combinations[i % len(genre_combinations)]
            content_type = np.random.choice(content_types, p=[0.6, 0.25, 0.15])
            language = np.random.choice(languages, p=[0.4, 0.15, 0.1, 0.1, 0.05, 0.05, 0.1, 0.02, 0.02, 0.01])
            
            # Generate realistic ratings with bias
            if 'Action' in genres or 'Adventure' in genres:
                rating_base = np.random.normal(7.2, 1.2)
            elif 'Drama' in genres:
                rating_base = np.random.normal(7.8, 1.0)
            elif 'Comedy' in genres:
                rating_base = np.random.normal(6.8, 1.3)
            elif content_type == 'anime':
                rating_base = np.random.normal(8.1, 0.9)
            else:
                rating_base = np.random.normal(7.0, 1.5)
            
            rating = max(1.0, min(10.0, rating_base))
            
            # Generate realistic popularity
            if content_type == 'movie':
                popularity = np.random.lognormal(3.5, 1.2)
            elif content_type == 'anime':
                popularity = np.random.lognormal(3.0, 1.0)
            else:
                popularity = np.random.lognormal(3.2, 1.1)
            
            # Generate realistic release dates
            if content_type == 'anime':
                release_days_ago = np.random.randint(0, 2000)
            else:
                release_days_ago = np.random.randint(0, 5000)
            
            release_date = (datetime.now() - timedelta(days=release_days_ago))
            
            # Determine if trending/new/critics
            is_new_release = release_days_ago <= 60
            is_trending = (release_days_ago <= 30 and popularity > 50) or np.random.random() < 0.05
            is_critics_choice = (rating >= 8.5 and np.random.random() < 0.3) or np.random.random() < 0.1
            
            content_data.append({
                'id': i,
                'tmdb_id': i * 10,
                'mal_id': i if content_type == 'anime' else None,
                'title': f'{genres[0]} {content_type.title()} {i}',
                'original_title': f'Original {genres[0]} {i}',
                'content_type': content_type,
                'genres': json.dumps(genres),
                'languages': json.dumps([language]),
                'rating': round(rating, 1),
                'popularity': round(popularity, 2),
                'release_date': release_date.strftime('%Y-%m-%d'),
                'overview': f'An exciting {" ".join(genres).lower()} {content_type} that follows the journey of compelling characters through various challenges and adventures. This {content_type} explores themes of {", ".join(genres).lower()} with outstanding performances.',
                'runtime': np.random.randint(80, 180) if content_type == 'movie' else np.random.randint(20, 60),
                'vote_count': max(10, int(np.random.lognormal(5, 1.5))),
                'is_trending': is_trending,
                'is_new_release': is_new_release,
                'is_critics_choice': is_critics_choice,
                'poster_path': f'/poster_{i}.jpg',
                'backdrop_path': f'/backdrop_{i}.jpg',
                'youtube_trailer_id': f'trailer_{i}',
                'created_at': (datetime.now() - timedelta(days=release_days_ago)).isoformat(),
                'updated_at': datetime.now().isoformat()
            })
        
        # Realistic user data
        users_data = []
        for i in range(1, 101):  # 100 users
            preferred_genres = np.random.choice(
                ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Animation'],
                size=np.random.randint(2, 5), replace=False
            ).tolist()
            
            preferred_languages = np.random.choice(
                languages, size=np.random.randint(1, 3), replace=False
            ).tolist()
            
            users_data.append({
                'id': i,
                'username': f'user_{i}',
                'email': f'user_{i}@example.com',
                'preferred_languages': json.dumps(preferred_languages),
                'preferred_genres': json.dumps(preferred_genres),
                'location': np.random.choice(['India', 'USA', 'Japan', 'UK', 'Canada']),
                'created_at': (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat(),
                'last_active': (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat(),
                'is_admin': i <= 5  # First 5 users are admins
            })
        
        # Realistic interaction patterns
        interactions_data = []
        interaction_types = ['view', 'like', 'favorite', 'watchlist', 'search']
        interaction_weights = [0.5, 0.2, 0.1, 0.15, 0.05]
        
        interaction_id = 1
        for user_id in range(1, 101):
            user_preferences = json.loads(users_data[user_id - 1]['preferred_genres'])
            num_interactions = np.random.poisson(25)  # Average 25 interactions per user
            
            for _ in range(num_interactions):
                # Bias content selection towards user preferences
                if np.random.random() < 0.7:  # 70% chance of preference-based selection
                    # Select content matching user preferences
                    matching_content = [
                        c for c in content_data 
                        if any(genre in json.loads(c['genres']) for genre in user_preferences)
                    ]
                    if matching_content:
                        content = np.random.choice(matching_content)
                    else:
                        content = np.random.choice(content_data)
                else:
                    content = np.random.choice(content_data)
                
                interaction_type = np.random.choice(interaction_types, p=interaction_weights)
                
                # Generate realistic ratings
                if interaction_type in ['like', 'favorite']:
                    rating = np.random.randint(7, 11)
                elif interaction_type == 'view':
                    rating = np.random.randint(5, 10) if np.random.random() < 0.3 else None
                else:
                    rating = None
                
                # Generate realistic timestamps
                days_ago = np.random.exponential(30)  # More recent interactions are more likely
                timestamp = datetime.now() - timedelta(days=days_ago)
                
                interactions_data.append({
                    'id': interaction_id,
                    'user_id': user_id,
                    'content_id': content['id'],
                    'interaction_type': interaction_type,
                    'rating': rating,
                    'timestamp': timestamp.isoformat()
                })
                interaction_id += 1
        
        logger.info(f"Created sample data: {len(content_data)} content, {len(interactions_data)} interactions, {len(users_data)} users")
        return content_data, interactions_data, users_data
    
    @staticmethod
    def preprocess_content_data(content_data):
        """Advanced content data preprocessing with feature engineering"""
        try:
            df = pd.DataFrame(content_data)
            if df.empty:
                return df
            
            # Parse JSON fields
            df['genres_list'] = df['genres'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else [])
            df['languages_list'] = df['languages'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else [])
            
            # Text processing
            df['genre_text'] = df['genres_list'].apply(lambda x: ' '.join(x))
            df['language_text'] = df['languages_list'].apply(lambda x: ' '.join(x))
            
            # Advanced text features
            df['overview_clean'] = df['overview'].fillna('').apply(
                lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower())
            )
            
            # Sentiment analysis
            df['overview_sentiment'] = df['overview'].fillna('').apply(
                lambda x: TextBlob(str(x)).sentiment.polarity
            )
            
            # Combined features for content-based filtering
            df['combined_features'] = (
                df['title'].fillna('') + ' ' +
                df['overview_clean'] + ' ' +
                df['genre_text'] + ' ' +
                df['language_text'] + ' ' +
                df['content_type'].fillna('')
            )
            
            # Numerical feature engineering
            scaler = RobustScaler()
            
            # Handle missing values
            numerical_cols = ['rating', 'popularity', 'runtime', 'vote_count']
            for col in numerical_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())
            
            # Normalize numerical features
            if len(df) > 1:
                for col in numerical_cols:
                    if col in df.columns:
                        df[f'{col}_norm'] = scaler.fit_transform(df[[col]].fillna(0))
            
            # Date processing
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['release_year'] = df['release_date'].dt.year
            df['release_month'] = df['release_date'].dt.month
            
            # Calculate content age and recency
            current_date = datetime.now()
            df['content_age_days'] = (current_date - df['release_date']).dt.days
            df['content_age_years'] = df['content_age_days'] / 365.25
            
            # Quality score calculation
            df['quality_score'] = (
                (df['rating'] / 10.0) * 0.4 +
                (np.log1p(df['vote_count']) / np.log1p(df['vote_count'].max())) * 0.3 +
                (df['popularity'] / df['popularity'].max()) * 0.2 +
                df['overview_sentiment'].clip(-1, 1) * 0.1
            )
            
            # Genre encoding for clustering
            all_genres = set()
            for genres in df['genres_list']:
                all_genres.update(genres)
            
            for genre in all_genres:
                df[f'genre_{genre.lower().replace(" ", "_")}'] = df['genres_list'].apply(
                    lambda x: 1 if genre in x else 0
                )
            
            # Language encoding
            all_languages = set()
            for languages in df['languages_list']:
                all_languages.update(languages)
            
            for language in all_languages:
                df[f'lang_{language.lower()}'] = df['languages_list'].apply(
                    lambda x: 1 if language in x else 0
                )
            
            # Content type encoding
            df['is_movie'] = (df['content_type'] == 'movie').astype(int)
            df['is_tv'] = (df['content_type'] == 'tv').astype(int)
            df['is_anime'] = (df['content_type'] == 'anime').astype(int)
            
            logger.info(f"Processed content data: {len(df)} items with {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing content data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def preprocess_interactions_data(interactions_data):
        """Advanced interactions data preprocessing"""
        try:
            df = pd.DataFrame(interactions_data)
            if df.empty:
                return df
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Advanced interaction weights
            interaction_weights = {
                'view': 1.0,
                'like': 2.5,
                'favorite': 4.0,
                'watchlist': 3.0,
                'search': 0.5,
                'share': 2.0,
                'comment': 3.5,
                'rate': 2.0
            }
            
            df['base_weight'] = df['interaction_type'].map(interaction_weights).fillna(1.0)
            
            # Time-based weighting (recent interactions have higher weight)
            max_timestamp = df['timestamp'].max()
            df['days_ago'] = (max_timestamp - df['timestamp']).dt.days
            df['recency_weight'] = np.exp(-df['days_ago'] / 60.0)  # 60-day half-life
            
            # Rating-based weighting
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df['rating_weight'] = df['rating'].fillna(7.0) / 10.0
            
            # Calculate final weight
            df['final_weight'] = (
                df['base_weight'] * 
                df['recency_weight'] * 
                df['rating_weight']
            )
            
            # User interaction frequency
            user_interaction_counts = df.groupby('user_id').size()
            df['user_activity_level'] = df['user_id'].map(user_interaction_counts)
            df['user_activity_norm'] = df['user_activity_level'] / df['user_activity_level'].max()
            
            # Content popularity from interactions
            content_interaction_counts = df.groupby('content_id').size()
            df['content_popularity'] = df['content_id'].map(content_interaction_counts)
            
            # Seasonal patterns
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            
            # Interaction sequences (for session-based recommendations)
            df = df.sort_values(['user_id', 'timestamp'])
            df['session_id'] = (
                df.groupby('user_id')['timestamp']
                .diff()
                .dt.total_seconds()
                .fillna(0)
                .gt(3600)  # 1 hour gap = new session
                .groupby(df['user_id'])
                .cumsum()
            )
            
            logger.info(f"Processed interactions data: {len(df)} interactions")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing interactions data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def preprocess_users_data(users_data):
        """Process user data for personalization"""
        try:
            df = pd.DataFrame(users_data)
            if df.empty:
                return df
            
            # Parse JSON fields
            df['preferred_genres_list'] = df['preferred_genres'].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x else []
            )
            df['preferred_languages_list'] = df['preferred_languages'].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x else []
            )
            
            # Convert dates
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['last_active'] = pd.to_datetime(df['last_active'], errors='coerce')
            
            # User metrics
            df['account_age_days'] = (datetime.now() - df['created_at']).dt.days
            df['last_active_days'] = (datetime.now() - df['last_active']).dt.days
            
            # Activity level classification
            df['activity_level'] = pd.cut(
                df['last_active_days'],
                bins=[-1, 7, 30, 90, float('inf')],
                labels=['very_active', 'active', 'moderate', 'inactive']
            )
            
            logger.info(f"Processed users data: {len(df)} users")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing users data: {e}")
            return pd.DataFrame()

# Advanced Recommendation Algorithms
class AdvancedCollaborativeFiltering:
    """State-of-the-art collaborative filtering with multiple algorithms"""
    
    def __init__(self):
        self.models = {}
        self.user_means = {}
        self.item_means = {}
        self.global_mean = 0
        self.trained = False
        
    def fit(self, interactions_df, content_df):
        """Train multiple collaborative filtering models"""
        try:
            if interactions_df.empty:
                return
            
            # Prepare rating matrix
            rating_data = interactions_df.groupby(['user_id', 'content_id']).agg({
                'rating': 'mean',
                'final_weight': 'sum'
            }).reset_index()
            
            # Fill missing ratings with weighted average
            rating_data['rating'] = rating_data['rating'].fillna(
                rating_data['final_weight'] * 7.0  # Assume neutral-positive rating
            )
            
            # Create Surprise dataset
            reader = Reader(rating_scale=(1, 10))
            dataset = Dataset.load_from_df(rating_data[['user_id', 'content_id', 'rating']], reader)
            trainset = dataset.build_full_trainset()
            
            # Train multiple models
            algorithms = {
                'svd': SVD(n_factors=200, n_epochs=30, lr_all=0.007, reg_all=0.02),
                'svd_pp': SVD(n_factors=150, n_epochs=25, lr_all=0.008, reg_all=0.015),
                'nmf': SurpriseNMF(n_factors=100, n_epochs=50),
                'knn_user': KNNWithMeans(k=50, sim_options={'name': 'cosine', 'user_based': True}),
                'knn_item': KNNWithMeans(k=40, sim_options={'name': 'cosine', 'user_based': False}),
                'baseline': BaselineOnly()
            }
            
            for name, algo in algorithms.items():
                try:
                    algo.fit(trainset)
                    self.models[name] = algo
                    logger.info(f"Trained {name} model successfully")
                except Exception as e:
                    logger.warning(f"Failed to train {name}: {e}")
            
            # Calculate global statistics
            self.global_mean = rating_data['rating'].mean()
            self.user_means = rating_data.groupby('user_id')['rating'].mean().to_dict()
            self.item_means = rating_data.groupby('content_id')['rating'].mean().to_dict()
            
            self.trained = True
            logger.info(f"Collaborative filtering trained with {len(rating_data)} ratings")
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering: {e}")
    
    def predict_rating(self, user_id, content_id):
        """Predict rating using ensemble of models"""
        if not self.trained or not self.models:
            return self.global_mean
        
        predictions = []
        weights = {'svd': 0.3, 'svd_pp': 0.25, 'nmf': 0.2, 'knn_user': 0.1, 'knn_item': 0.1, 'baseline': 0.05}
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(user_id, content_id)
                predictions.append(pred.est * weights.get(model_name, 0.1))
            except:
                continue
        
        if predictions:
            return sum(predictions) / sum(weights[name] for name in self.models.keys() if name in weights)
        else:
            return self.user_means.get(user_id, self.global_mean)
    
    def get_user_recommendations(self, user_id, content_df, interactions_df, n_recommendations=50):
        """Get collaborative filtering recommendations"""
        try:
            if not self.trained:
                return []
            
            # Get user's interaction history
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]['content_id'].unique()
            
            # Get all content not interacted with
            all_content_ids = content_df['id'].unique()
            candidate_content = [cid for cid in all_content_ids if cid not in user_interactions]
            
            if not candidate_content:
                return []
            
            # Predict ratings for all candidates
            predictions = []
            for content_id in candidate_content:
                predicted_rating = self.predict_rating(user_id, content_id)
                predictions.append({
                    'content_id': content_id,
                    'score': predicted_rating / 10.0,  # Normalize to 0-1
                    'reason': 'Users with similar preferences also liked this'
                })
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x['score'], reverse=True)
            return predictions[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {e}")
            return []

class AdvancedContentBasedFiltering:
    """Advanced content-based filtering with multiple similarity metrics"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        self.count_vectorizer = CountVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = None
        self.count_matrix = None
        self.content_features_matrix = None
        self.similarity_matrices = {}
        
    def fit(self, content_df):
        """Train content-based models"""
        try:
            if content_df.empty:
                return
            
            # TF-IDF on combined text features
            text_features = content_df['combined_features'].fillna('')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
            
            # Count vectorizer for genre/language features
            genre_lang_features = content_df['genre_text'] + ' ' + content_df['language_text']
            self.count_matrix = self.count_vectorizer.fit_transform(genre_lang_features.fillna(''))
            
            # Numerical features matrix
            feature_cols = [col for col in content_df.columns if col.endswith('_norm')]
            if feature_cols:
                numerical_features = content_df[feature_cols].fillna(0).values
                
                # Combine all features
                self.content_features_matrix = sparse.hstack([
                    self.tfidf_matrix,
                    self.count_matrix,
                    sparse.csr_matrix(numerical_features)
                ])
            else:
                self.content_features_matrix = sparse.hstack([
                    self.tfidf_matrix,
                    self.count_matrix
                ])
            
            # Pre-compute similarity matrices
            self.similarity_matrices = {
                'cosine': cosine_similarity(self.content_features_matrix),
                'linear': linear_kernel(self.content_features_matrix)
            }
            
            logger.info(f"Content-based model trained with {content_df.shape[0]} items")
            
        except Exception as e:
            logger.error(f"Error training content-based model: {e}")
    
    def get_content_similarities(self, content_id, content_df, similarity_type='cosine'):
        """Get content similarities using specified metric"""
        try:
            if content_id not in content_df['id'].values:
                return []
            
            # Get content index
            content_idx = content_df[content_df['id'] == content_id].index[0]
            
            # Get similarity scores
            similarity_matrix = self.similarity_matrices.get(similarity_type, self.similarity_matrices['cosine'])
            sim_scores = similarity_matrix[content_idx]
            
            # Create recommendations
            recommendations = []
            for idx, score in enumerate(sim_scores):
                if idx != content_idx and score > 0.1:  # Minimum similarity threshold
                    other_content_id = content_df.iloc[idx]['id']
                    recommendations.append({
                        'content_id': other_content_id,
                        'score': float(score),
                        'reason': 'Similar content and themes'
                    })
            
            # Sort by similarity score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting content similarities: {e}")
            return []

class SemanticSimilarityEngine:
    """Advanced semantic similarity using sentence transformers"""
    
    def __init__(self):
        self.model = None
        self.embeddings = None
        self.content_ids = None
        
    def initialize(self):
        """Initialize sentence transformer model"""
        try:
            # Try multiple models in order of preference
            models_to_try = [
                'all-MiniLM-L6-v2',
                'all-MiniLM-L12-v2',
                'paraphrase-MiniLM-L6-v2'
            ]
            
            for model_name in models_to_try:
                try:
                    self.model = SentenceTransformer(model_name)
                    logger.info(f"Loaded semantic model: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            if self.model is None:
                logger.warning("No semantic model could be loaded")
                
        except Exception as e:
            logger.error(f"Error initializing semantic model: {e}")
    
    def fit(self, content_df):
        """Generate embeddings for content"""
        try:
            if self.model is None:
                return
            
            # Prepare text for embedding
            texts = []
            content_ids = []
            
            for _, content in content_df.iterrows():
                text = f"{content['title']} {content['overview']} {content['genre_text']}"
                texts.append(text)
                content_ids.append(content['id'])
            
            # Generate embeddings
            self.embeddings = self.model.encode(texts, show_progress_bar=False)
            self.content_ids = content_ids
            
            logger.info(f"Generated semantic embeddings for {len(texts)} content items")
            
        except Exception as e:
            logger.error(f"Error generating semantic embeddings: {e}")
    
    def get_semantic_similarities(self, content_id, n_recommendations=50):
        """Get semantically similar content"""
        try:
            if self.embeddings is None or content_id not in self.content_ids:
                return []
            
            # Find content index
            content_idx = self.content_ids.index(content_id)
            
            # Calculate cosine similarities
            similarities = cosine_similarity([self.embeddings[content_idx]], self.embeddings)[0]
            
            # Create recommendations
            recommendations = []
            for idx, score in enumerate(similarities):
                if idx != content_idx and score > 0.3:  # Minimum similarity threshold
                    other_content_id = self.content_ids[idx]
                    recommendations.append({
                        'content_id': other_content_id,
                        'score': float(score),
                        'reason': 'Semantically similar content'
                    })
            
            # Sort by similarity score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting semantic similarities: {e}")
            return []

class TrendingAnalysisEngine:
    """Advanced trending analysis with multiple signals"""
    
    @staticmethod
    def calculate_comprehensive_trending_score(content_df, interactions_df, time_windows=None):
        """Calculate trending scores using multiple time windows and signals"""
        try:
            if time_windows is None:
                time_windows = [6, 12, 24, 48, 168]  # hours
            
            current_time = datetime.now()
            trending_scores = {}
            
            for _, content in content_df.iterrows():
                content_id = content['id']
                base_score = 0.0
                
                # Base quality and popularity
                base_score += (content['rating'] / 10.0) * 0.2
                base_score += min(content['popularity'] / 100.0, 1.0) * 0.15
                base_score += min(np.log1p(content['vote_count']) / 10.0, 1.0) * 0.1
                
                # Multi-window interaction analysis
                for window_hours in time_windows:
                    window_start = current_time - timedelta(hours=window_hours)
                    window_interactions = interactions_df[
                        (interactions_df['content_id'] == content_id) &
                        (pd.to_datetime(interactions_df['timestamp']) >= window_start)
                    ]
                    
                    if not window_interactions.empty:
                        # Interaction velocity (interactions per hour)
                        velocity = len(window_interactions) / window_hours
                        base_score += min(velocity * 0.1, 0.2)
                        
                        # Unique users in window
                        unique_users = window_interactions['user_id'].nunique()
                        base_score += min(unique_users / 10.0, 0.15) * (0.5 / len(time_windows))
                        
                        # Weighted interactions
                        total_weight = window_interactions['final_weight'].sum()
                        base_score += min(total_weight / 20.0, 0.1) * (0.3 / len(time_windows))
                
                # Recency boost for new releases
                if content['is_new_release']:
                    base_score *= 1.5
                
                # Quality boost for critics choice
                if content['is_critics_choice']:
                    base_score *= 1.2
                
                # Content type adjustments
                if content['content_type'] == 'movie':
                    base_score *= 1.1
                elif content['content_type'] == 'anime':
                    base_score *= 1.05
                
                trending_scores[content_id] = base_score
            
            return trending_scores
            
        except Exception as e:
            logger.error(f"Error calculating trending scores: {e}")
            return {}
    
    @staticmethod
    def get_regional_trending(content_df, interactions_df, language=None, region=None):
        """Get regional trending with cultural preferences"""
        try:
            filtered_content = content_df.copy()
            
            # Language filtering
            if language:
                language_variants = {
                    'hindi': ['hindi', 'hi', 'bollywood'],
                    'telugu': ['telugu', 'te', 'tollywood'],
                    'tamil': ['tamil', 'ta', 'kollywood'],
                    'kannada': ['kannada', 'kn', 'sandalwood'],
                    'malayalam': ['malayalam', 'ml', 'mollywood'],
                    'english': ['english', 'en', 'hollywood'],
                    'japanese': ['japanese', 'ja', 'anime'],
                    'korean': ['korean', 'ko', 'kdrama']
                }
                
                target_languages = language_variants.get(language.lower(), [language])
                
                def matches_language(lang_list):
                    if not lang_list:
                        return False
                    return any(
                        any(target in lang.lower() for target in target_languages)
                        for lang in lang_list
                    )
                
                filtered_content = filtered_content[
                    filtered_content['languages_list'].apply(matches_language)
                ]
            
            # Calculate trending scores for filtered content
            trending_scores = TrendingAnalysisEngine.calculate_comprehensive_trending_score(
                filtered_content, interactions_df
            )
            
            # Apply regional preferences
            regional_boost = {
                'India': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam'],
                'Japan': ['japanese'],
                'Korea': ['korean'],
                'USA': ['english']
            }
            
            if region and region in regional_boost:
                preferred_languages = regional_boost[region]
                for content_id, score in trending_scores.items():
                    content_row = filtered_content[filtered_content['id'] == content_id]
                    if not content_row.empty:
                        content_languages = content_row.iloc[0]['languages_list']
                        if any(lang in content_languages for lang in preferred_languages):
                            trending_scores[content_id] *= 1.3
            
            return trending_scores
            
        except Exception as e:
            logger.error(f"Error getting regional trending: {e}")
            return {}

class UserProfilingEngine:
    """Advanced user profiling and behavior analysis"""
    
    @staticmethod
    def build_comprehensive_user_profile(user_id, interactions_df, content_df, users_df=None):
        """Build comprehensive user profile from all available data"""
        try:
            profile = {
                'user_id': user_id,
                'preferences': {},
                'behavior_patterns': {},
                'recommendations_context': {}
            }
            
            # Get user's interactions
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            if user_interactions.empty:
                return profile
            
            # Get interacted content
            content_ids = user_interactions['content_id'].unique()
            user_content = content_df[content_df['id'].isin(content_ids)]
            
            # Genre preferences with weights
            genre_weights = defaultdict(float)
            for _, interaction in user_interactions.iterrows():
                content = content_df[content_df['id'] == interaction['content_id']]
                if not content.empty:
                    genres = content.iloc[0]['genres_list']
                    weight = interaction['final_weight']
                    for genre in genres:
                        genre_weights[genre] += weight
            
            # Normalize genre preferences
            total_weight = sum(genre_weights.values())
            if total_weight > 0:
                profile['preferences']['genres'] = {
                    genre: weight / total_weight 
                    for genre, weight in genre_weights.items()
                }
            
            # Language preferences
            language_weights = defaultdict(float)
            for _, interaction in user_interactions.iterrows():
                content = content_df[content_df['id'] == interaction['content_id']]
                if not content.empty:
                    languages = content.iloc[0]['languages_list']
                    weight = interaction['final_weight']
                    for language in languages:
                        language_weights[language] += weight
            
            total_lang_weight = sum(language_weights.values())
            if total_lang_weight > 0:
                profile['preferences']['languages'] = {
                    lang: weight / total_lang_weight 
                    for lang, weight in language_weights.items()
                }
            
            # Content type preferences
            content_type_weights = defaultdict(float)
            for _, interaction in user_interactions.iterrows():
                content = content_df[content_df['id'] == interaction['content_id']]
                if not content.empty:
                    content_type = content.iloc[0]['content_type']
                    weight = interaction['final_weight']
                    content_type_weights[content_type] += weight
            
            total_ct_weight = sum(content_type_weights.values())
            if total_ct_weight > 0:
                profile['preferences']['content_types'] = {
                    ct: weight / total_ct_weight 
                    for ct, weight in content_type_weights.items()
                }
            
            # Quality preferences
            ratings = user_interactions['rating'].dropna()
            if not ratings.empty:
                profile['preferences']['avg_rating_given'] = ratings.mean()
                profile['preferences']['rating_std'] = ratings.std()
                profile['preferences']['prefers_high_quality'] = ratings.mean() > 7.5
            
            # Temporal behavior patterns
            timestamps = pd.to_datetime(user_interactions['timestamp'])
            if not timestamps.empty:
                profile['behavior_patterns']['peak_hours'] = timestamps.dt.hour.mode().tolist()
                profile['behavior_patterns']['active_days'] = timestamps.dt.dayofweek.mode().tolist()
                profile['behavior_patterns']['activity_frequency'] = len(user_interactions) / max(1, (timestamps.max() - timestamps.min()).days)
            
            # Interaction patterns
            interaction_types = user_interactions['interaction_type'].value_counts()
            profile['behavior_patterns']['interaction_distribution'] = interaction_types.to_dict()
            
            # Exploration vs exploitation
            unique_genres = len(set(genre for content in user_content['genres_list'] for genre in content))
            total_content = len(user_content)
            profile['behavior_patterns']['diversity_score'] = unique_genres / max(1, total_content)
            
            # User stated preferences (if available)
            if users_df is not None:
                user_data = users_df[users_df['id'] == user_id]
                if not user_data.empty:
                    user_row = user_data.iloc[0]
                    profile['preferences']['stated_genres'] = user_row.get('preferred_genres_list', [])
                    profile['preferences']['stated_languages'] = user_row.get('preferred_languages_list', [])
                    profile['behavior_patterns']['account_age_days'] = user_row.get('account_age_days', 0)
                    profile['behavior_patterns']['activity_level'] = user_row.get('activity_level', 'unknown')
            
            # Recent activity context
            recent_interactions = user_interactions[
                pd.to_datetime(user_interactions['timestamp']) >= 
                (datetime.now() - timedelta(days=7))
            ]
            
            if not recent_interactions.empty:
                recent_content_ids = recent_interactions['content_id'].unique()
                recent_content = content_df[content_df['id'].isin(recent_content_ids)]
                
                # Recent genre trends
                recent_genres = []
                for _, content in recent_content.iterrows():
                    recent_genres.extend(content['genres_list'])
                
                if recent_genres:
                    recent_genre_counts = Counter(recent_genres)
                    profile['recommendations_context']['recent_genre_focus'] = recent_genre_counts.most_common(3)
                
                # Recent content types
                recent_content_types = recent_content['content_type'].value_counts()
                profile['recommendations_context']['recent_content_type_focus'] = recent_content_types.to_dict()
            
            return profile
            
        except Exception as e:
            logger.error(f"Error building user profile: {e}")
            return {'user_id': user_id, 'preferences': {}, 'behavior_patterns': {}, 'recommendations_context': {}}

class HybridRecommendationEngine:
    """Advanced hybrid recommendation engine combining all approaches"""
    
    def __init__(self):
        self.collaborative_engine = AdvancedCollaborativeFiltering()
        self.content_engine = AdvancedContentBasedFiltering()
        self.semantic_engine = SemanticSimilarityEngine()
        self.weights = {
            'collaborative': 0.35,
            'content': 0.25,
            'semantic': 0.20,
            'trending': 0.10,
            'quality': 0.10
        }
        
    def fit(self, content_df, interactions_df, users_df):
        """Train all recommendation engines"""
        try:
            logger.info("Training hybrid recommendation engine...")
            
            # Train individual engines
            self.collaborative_engine.fit(interactions_df, content_df)
            self.content_engine.fit(content_df)
            
            # Initialize and train semantic engine
            self.semantic_engine.initialize()
            self.semantic_engine.fit(content_df)
            
            logger.info("Hybrid recommendation engine training completed")
            
        except Exception as e:
            logger.error(f"Error training hybrid engine: {e}")
    
    def get_personalized_recommendations(self, user_id, content_df, interactions_df, users_df, 
                                       user_profile=None, n_recommendations=50):
        """Get comprehensive personalized recommendations"""
        try:
            # Build/use user profile
            if user_profile is None:
                user_profile = UserProfilingEngine.build_comprehensive_user_profile(
                    user_id, interactions_df, content_df, users_df
                )
            
            # Get recommendations from each engine
            all_recommendations = defaultdict(float)
            recommendation_sources = defaultdict(list)
            
            # 1. Collaborative filtering recommendations
            collab_recs = self.collaborative_engine.get_user_recommendations(
                user_id, content_df, interactions_df, n_recommendations * 2
            )
            
            for rec in collab_recs:
                content_id = rec['content_id']
                score = rec['score'] * self.weights['collaborative']
                all_recommendations[content_id] += score
                recommendation_sources[content_id].append('collaborative')
            
            # 2. Content-based recommendations (based on user's top genres)
            user_genres = user_profile.get('preferences', {}).get('genres', {})
            top_genres = sorted(user_genres.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for genre, weight in top_genres:
                genre_content = content_df[
                    content_df['genres_list'].apply(lambda x: genre in x)
                ]
                
                for _, content in genre_content.head(20).iterrows():
                    content_id = content['id']
                    score = (content['quality_score'] * weight * self.weights['content'])
                    all_recommendations[content_id] += score
                    recommendation_sources[content_id].append('content_genre')
            
            # 3. Semantic recommendations (based on recent interactions)
            recent_interactions = interactions_df[
                (interactions_df['user_id'] == user_id) &
                (pd.to_datetime(interactions_df['timestamp']) >= 
                 (datetime.now() - timedelta(days=14)))
            ]
            
            for _, interaction in recent_interactions.head(5).iterrows():
                semantic_recs = self.semantic_engine.get_semantic_similarities(
                    interaction['content_id'], n_recommendations=20
                )
                
                for rec in semantic_recs:
                    content_id = rec['content_id']
                    score = rec['score'] * self.weights['semantic'] * 0.2  # Distribute among recent items
                    all_recommendations[content_id] += score
                    recommendation_sources[content_id].append('semantic')
            
            # 4. Quality and trending boost
            trending_scores = TrendingAnalysisEngine.calculate_comprehensive_trending_score(
                content_df, interactions_df
            )
            
            for content_id in all_recommendations.keys():
                content_row = content_df[content_df['id'] == content_id]
                if not content_row.empty:
                    content_data = content_row.iloc[0]
                    
                    # Quality boost
                    quality_boost = content_data['quality_score'] * self.weights['quality']
                    all_recommendations[content_id] += quality_boost
                    
                    # Trending boost
                    trending_boost = trending_scores.get(content_id, 0) * self.weights['trending']
                    all_recommendations[content_id] += trending_boost
            
            # 5. Apply user preference filters and boosts
            for content_id in list(all_recommendations.keys()):
                content_row = content_df[content_df['id'] == content_id]
                if not content_row.empty:
                    content_data = content_row.iloc[0]
                    
                    # Language preference boost
                    content_languages = content_data['languages_list']
                    user_languages = user_profile.get('preferences', {}).get('languages', {})
                    for lang in content_languages:
                        if lang in user_languages:
                            all_recommendations[content_id] *= (1 + user_languages[lang] * 0.3)
                    
                    # Content type preference boost
                    content_type = content_data['content_type']
                    user_content_types = user_profile.get('preferences', {}).get('content_types', {})
                    if content_type in user_content_types:
                        all_recommendations[content_id] *= (1 + user_content_types[content_type] * 0.2)
                    
                    # Quality preference alignment
                    if user_profile.get('preferences', {}).get('prefers_high_quality', False):
                        if content_data['rating'] >= 8.0:
                            all_recommendations[content_id] *= 1.2
                        elif content_data['rating'] < 6.0:
                            all_recommendations[content_id] *= 0.8
            
            # 6. Diversity and novelty adjustments
            user_interacted_content = interactions_df[interactions_df['user_id'] == user_id]['content_id'].unique()
            
            # Remove already interacted content
            for content_id in user_interacted_content:
                all_recommendations.pop(content_id, None)
            
            # Sort by final score
            sorted_recommendations = sorted(
                all_recommendations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Format recommendations
            recommendations = []
            for content_id, score in sorted_recommendations[:n_recommendations]:
                sources = recommendation_sources[content_id]
                reason = self._generate_recommendation_reason(sources, user_profile)
                
                recommendations.append({
                    'content_id': content_id,
                    'score': float(score),
                    'reason': reason,
                    'sources': sources
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return []
    
    def _generate_recommendation_reason(self, sources, user_profile):
        """Generate human-readable recommendation reason"""
        if 'collaborative' in sources and 'content_genre' in sources:
            return "Recommended based on your taste and similar users' preferences"
        elif 'collaborative' in sources:
            return "Users with similar preferences also enjoyed this"
        elif 'content_genre' in sources:
            return "Matches your favorite genres and interests"
        elif 'semantic' in sources:
            return "Similar to content you recently enjoyed"
        else:
            return "Trending content that matches your profile"

# Main ML Service
class MLRecommendationService:
    """Main ML recommendation service with all engines"""
    
    def __init__(self):
        self.hybrid_engine = HybridRecommendationEngine()
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_update = None
        
    def update_models(self):
        """Update all ML models with latest data"""
        global model_store, models_initialized
        
        try:
            with models_lock:
                logger.info("Starting comprehensive model update...")
                start_time = time.time()
                
                # Fetch all data
                content_data, interactions_data, users_data = AdvancedDataProcessor.fetch_comprehensive_data()
                
                # Preprocess all data
                content_df = AdvancedDataProcessor.preprocess_content_data(content_data)
                interactions_df = AdvancedDataProcessor.preprocess_interactions_data(interactions_data)
                users_df = AdvancedDataProcessor.preprocess_users_data(users_data)
                
                if content_df.empty:
                    logger.warning("No content data available for training")
                    return False
                
                # Store in model store
                model_store.content_df = content_df
                model_store.interactions_df = interactions_df
                model_store.users_df = users_df
                
                # Create metadata maps
                model_store.content_metadata = {
                    row['id']: row.to_dict() 
                    for _, row in content_df.iterrows()
                }
                
                if not users_df.empty:
                    model_store.user_metadata = {
                        row['id']: row.to_dict() 
                        for _, row in users_df.iterrows()
                    }
                
                # Train hybrid engine
                self.hybrid_engine.fit(content_df, interactions_df, users_df)
                
                # Calculate and store trending scores
                trending_scores = TrendingAnalysisEngine.calculate_comprehensive_trending_score(
                    content_df, interactions_df
                )
                model_store.trending_weights = trending_scores
                
                # Update metadata
                self.last_update = datetime.now()
                model_store.last_update = self.last_update
                model_store.update_count += 1
                models_initialized = True
                
                # Clear cache
                self.cache.clear()
                
                update_time = time.time() - start_time
                logger.info(f"Model update completed in {update_time:.2f}s. "
                           f"Content: {len(content_df)}, Interactions: {len(interactions_df)}, Users: {len(users_df)}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating models: {e}")
            return False
    
    def _get_cache_key(self, prefix, **kwargs):
        """Generate cache key from parameters"""
        key_parts = [prefix]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        return "_".join(key_parts)
    
    def _get_cached_result(self, cache_key):
        """Get cached result if valid"""
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                result['cached'] = True
                return result
        return None
    
    def _cache_result(self, cache_key, result):
        """Cache result with timestamp"""
        result_copy = result.copy()
        result_copy['cached'] = False
        self.cache[cache_key] = (result_copy, datetime.now())
    
    def get_trending_recommendations(self, limit=20, content_type='all', region=None, language=None):
        """Get advanced trending recommendations"""
        try:
            cache_key = self._get_cache_key('trending', limit=limit, content_type=content_type, 
                                          region=region, language=language)
            cached = self._get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            interactions_df = model_store.interactions_df
            
            # Filter by content type
            if content_type != 'all':
                content_df = content_df[content_df['content_type'] == content_type]
            
            # Calculate trending scores
            if region or language:
                trending_scores = TrendingAnalysisEngine.get_regional_trending(
                    content_df, interactions_df, language=language, region=region
                )
            else:
                trending_scores = model_store.trending_weights or {}
            
            # Sort by trending score
            trending_items = sorted(trending_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for content_id, score in trending_items[:limit]:
                if content_id in model_store.content_metadata:
                    recommendations.append({
                        'content_id': content_id,
                        'score': float(score),
                        'reason': 'Currently trending in your region' if (region or language) else 'Currently trending globally'
                    })
            
            result = {
                'recommendations': recommendations,
                'strategy': 'advanced_trending_analysis',
                'cached': False
            }
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_personalized_recommendations(self, user_data, limit=20):
        """Get comprehensive personalized recommendations"""
        try:
            user_id = user_data.get('user_id')
            cache_key = self._get_cache_key('personalized', user_id=user_id, limit=limit)
            cached = self._get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            interactions_df = model_store.interactions_df
            users_df = model_store.users_df
            
            # Build user profile
            user_profile = UserProfilingEngine.build_comprehensive_user_profile(
                user_id, interactions_df, content_df, users_df
            )
            
            # Get hybrid recommendations
            recommendations = self.hybrid_engine.get_personalized_recommendations(
                user_id, content_df, interactions_df, users_df, user_profile, limit * 2
            )
            
            # Apply final filters and ranking
            final_recommendations = self._apply_personalization_filters(
                recommendations, user_data, user_profile, limit
            )
            
            result = {
                'recommendations': final_recommendations,
                'strategy': 'advanced_hybrid_personalized',
                'user_profile_summary': {
                    'top_genres': list(user_profile.get('preferences', {}).get('genres', {}).keys())[:3],
                    'preferred_languages': list(user_profile.get('preferences', {}).get('languages', {}).keys()),
                    'activity_level': user_profile.get('behavior_patterns', {}).get('activity_level', 'unknown')
                },
                'cached': False
            }
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def _apply_personalization_filters(self, recommendations, user_data, user_profile, limit):
        """Apply final personalization filters"""
        try:
            # Get user preferences from request data
            preferred_genres = user_data.get('preferred_genres', [])
            preferred_languages = user_data.get('preferred_languages', [])
            
            # Score adjustments
            for rec in recommendations:
                content_id = rec['content_id']
                content_data = model_store.content_metadata.get(content_id, {})
                
                # Genre preference boost
                content_genres = content_data.get('genres_list', [])
                genre_matches = len(set(preferred_genres) & set(content_genres))
                if genre_matches > 0:
                    rec['score'] *= (1.0 + 0.15 * genre_matches)
                
                # Language preference boost
                content_languages = content_data.get('languages_list', [])
                lang_matches = len(set(preferred_languages) & set(content_languages))
                if lang_matches > 0:
                    rec['score'] *= (1.0 + 0.25 * lang_matches)
                
                # Diversity penalty for overrepresented genres
                top_user_genres = list(user_profile.get('preferences', {}).get('genres', {}).keys())[:2]
                if len(set(content_genres) & set(top_user_genres)) == len(content_genres):
                    rec['score'] *= 0.9  # Small penalty for too similar content
            
            # Re-sort and limit
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error applying personalization filters: {e}")
            return recommendations[:limit]
    
    def get_similar_recommendations(self, content_id, limit=20):
        """Get comprehensive similar content recommendations"""
        try:
            cache_key = self._get_cache_key('similar', content_id=content_id, limit=limit)
            cached = self._get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            all_recommendations = defaultdict(float)
            
            # 1. Content-based similarities
            content_similarities = self.hybrid_engine.content_engine.get_content_similarities(
                content_id, content_df, 'cosine'
            )
            for rec in content_similarities[:30]:
                all_recommendations[rec['content_id']] += rec['score'] * 0.4
            
            # 2. Semantic similarities
            semantic_similarities = self.hybrid_engine.semantic_engine.get_semantic_similarities(
                content_id, 30
            )
            for rec in semantic_similarities:
                all_recommendations[rec['content_id']] += rec['score'] * 0.35
            
            # 3. Collaborative similarities (users who liked this also liked)
            content_users = model_store.interactions_df[
                model_store.interactions_df['content_id'] == content_id
            ]['user_id'].unique()
            
            if len(content_users) > 0:
                # Find content liked by these users
                similar_user_content = model_store.interactions_df[
                    (model_store.interactions_df['user_id'].isin(content_users)) &
                    (model_store.interactions_df['content_id'] != content_id) &
                    (model_store.interactions_df['final_weight'] > 1.5)
                ]
                
                content_popularity = similar_user_content.groupby('content_id')['final_weight'].sum()
                for other_content_id, weight in content_popularity.head(20).items():
                    normalized_weight = min(weight / 10.0, 1.0)
                    all_recommendations[other_content_id] += normalized_weight * 0.25
            
            # Sort and format
            sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for other_content_id, score in sorted_recs[:limit]:
                recommendations.append({
                    'content_id': other_content_id,
                    'score': float(score),
                    'reason': 'Similar content based on multiple similarity factors'
                })
            
            result = {
                'recommendations': recommendations,
                'strategy': 'multi_factor_similarity',
                'cached': False
            }
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting similar recommendations: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_genre_recommendations(self, genre, limit=20, content_type='movie', region=None):
        """Get advanced genre-based recommendations"""
        try:
            cache_key = self._get_cache_key('genre', genre=genre, limit=limit, 
                                          content_type=content_type, region=region)
            cached = self._get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            
            # Filter by content type and genre
            filtered_df = content_df[content_df['content_type'] == content_type]
            
            def has_genre(genres_list):
                if not genres_list:
                    return False
                return any(genre.lower() in g.lower() for g in genres_list)
            
            genre_content = filtered_df[filtered_df['genres_list'].apply(has_genre)]
            
            # Advanced scoring
            recommendations = []
            for _, content in genre_content.iterrows():
                score = 0.0
                
                # Base quality score
                score += content['quality_score'] * 0.4
                
                # Popularity score (normalized)
                if filtered_df['popularity'].max() > 0:
                    score += (content['popularity'] / filtered_df['popularity'].max()) * 0.2
                
                # Rating score (normalized)
                score += (content['rating'] / 10.0) * 0.2
                
                # Recency bonus
                if content['is_new_release']:
                    score += 0.1
                
                # Critics choice bonus
                if content['is_critics_choice']:
                    score += 0.05
                
                # Trending bonus
                if content['is_trending']:
                    score += 0.05
                
                # Regional adjustment
                if region:
                    content_languages = content['languages_list']
                    regional_languages = {
                        'india': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam'],
                        'japan': ['japanese'],
                        'korea': ['korean']
                    }
                    if region.lower() in regional_languages:
                        preferred_langs = regional_languages[region.lower()]
                        if any(lang in content_languages for lang in preferred_langs):
                            score *= 1.2
                
                recommendations.append({
                    'content_id': content['id'],
                    'score': float(score),
                    'reason': f'Top-rated {genre} {content_type} with high quality score'
                })
            
            # Sort by score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            result = {
                'recommendations': recommendations[:limit],
                'strategy': 'advanced_genre_filtering',
                'cached': False
            }
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting genre recommendations: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_regional_recommendations(self, language, limit=20, content_type='movie'):
        """Get advanced regional/language recommendations"""
        try:
            cache_key = self._get_cache_key('regional', language=language, limit=limit, content_type=content_type)
            cached = self._get_cached_result(cache_key)
            if cached:
                return cached
            
            # Use trending with regional/language filter
            result = self.get_trending_recommendations(limit, content_type, language=language)
            result['strategy'] = 'regional_trending_optimized'
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting regional recommendations: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_anime_recommendations(self, limit=20, genre=None):
        """Get specialized anime recommendations"""
        try:
            cache_key = self._get_cache_key('anime', limit=limit, genre=genre)
            cached = self._get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            
            # Filter anime content
            anime_df = content_df[content_df['content_type'] == 'anime']
            
            if genre:
                # Anime genre filtering with specialized logic
                anime_genre_keywords = {
                    'shonen': ['action', 'adventure', 'shounen', 'fighting', 'martial arts'],
                    'shojo': ['romance', 'drama', 'shoujo', 'slice of life', 'school'],
                    'seinen': ['thriller', 'psychological', 'seinen', 'mature', 'dark'],
                    'josei': ['romance', 'drama', 'josei', 'adult', 'realistic'],
                    'isekai': ['fantasy', 'adventure', 'isekai', 'parallel world'],
                    'mecha': ['mecha', 'robot', 'sci-fi', 'action']
                }
                
                keywords = anime_genre_keywords.get(genre.lower(), [genre.lower()])
                
                def has_anime_genre(genres_list):
                    if not genres_list:
                        return False
                    genre_text = ' '.join(genres_list).lower()
                    return any(keyword in genre_text for keyword in keywords)
                
                anime_df = anime_df[anime_df['genres_list'].apply(has_anime_genre)]
            
            # Anime-specific scoring
            recommendations = []
            for _, content in anime_df.iterrows():
                score = 0.0
                
                # Rating score (anime tends to have higher ratings)
                score += (content['rating'] / 10.0) * 0.4
                
                # Popularity score
                if anime_df['popularity'].max() > 0:
                    score += (content['popularity'] / anime_df['popularity'].max()) * 0.3
                
                # Recency bonus (recent anime get boost)
                if content['content_age_years'] <= 2:
                    score += 0.2
                
                # Quality indicators for anime
                if content['vote_count'] > 1000:  # Well-rated anime
                    score += 0.1
                
                recommendations.append({
                    'content_id': content['id'],
                    'score': float(score),
                    'reason': f'Top {genre or "anime"} recommendation with high community rating'
                })
            
            # Sort by score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            result = {
                'recommendations': recommendations[:limit],
                'strategy': 'anime_specialized_scoring',
                'cached': False
            }
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting anime recommendations: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_new_releases(self, limit=20, content_type='movie', language=None):
        """Get advanced new releases recommendations"""
        try:
            cache_key = self._get_cache_key('new_releases', limit=limit, content_type=content_type, language=language)
            cached = self._get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            
            # Filter new releases
            new_releases = content_df[
                (content_df['is_new_release'] == True) | 
                (content_df['content_age_days'] <= 90)  # Last 3 months
            ]
            
            # Filter by content type
            new_releases = new_releases[new_releases['content_type'] == content_type]
            
            # Apply language filter if specified
            if language:
                language_variants = {
                    'hindi': ['hindi', 'hi'],
                    'telugu': ['telugu', 'te'],
                    'tamil': ['tamil', 'ta'],
                    'kannada': ['kannada', 'kn'],
                    'malayalam': ['malayalam', 'ml'],
                    'english': ['english', 'en'],
                    'japanese': ['japanese', 'ja'],
                    'korean': ['korean', 'ko']
                }
                
                target_languages = language_variants.get(language.lower(), [language])
                
                def has_language(lang_list):
                    if not lang_list:
                        return False
                    return any(lang.lower() in target_languages for lang in lang_list)
                
                new_releases = new_releases[new_releases['languages_list'].apply(has_language)]
            
            # Score new releases
            recommendations = []
            for _, content in new_releases.iterrows():
                score = 0.0
                
                # Recency score (more recent = higher score)
                days_old = content['content_age_days']
                recency_score = max(0, (90 - days_old) / 90.0)  # Normalize to 0-1
                score += recency_score * 0.4
                
                # Quality score
                score += content['quality_score'] * 0.3
                
                # Rating score
                score += (content['rating'] / 10.0) * 0.2
                
                # Trending bonus
                if content['is_trending']:
                    score += 0.1
                
                recommendations.append({
                    'content_id': content['id'],
                    'score': float(score),
                    'reason': f'Recent {content_type} release with high quality'
                })
            
            # Sort by score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            result = {
                'recommendations': recommendations[:limit],
                'strategy': 'new_releases_quality_filtered',
                'cached': False
            }
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting new releases: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_critics_choice(self, limit=20, content_type='movie'):
        """Get advanced critics choice recommendations"""
        try:
            cache_key = self._get_cache_key('critics_choice', limit=limit, content_type=content_type)
            cached = self._get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            
            # Filter critics choice and high-quality content
            critics_choice = content_df[
                ((content_df['is_critics_choice'] == True) | 
                 (content_df['rating'] >= 8.0)) &
                (content_df['content_type'] == content_type) &
                (content_df['vote_count'] >= 100)  # Minimum vote threshold
            ]
            
            # Advanced scoring for critics choice
            recommendations = []
            for _, content in critics_choice.iterrows():
                score = 0.0
                
                # Rating score (primary factor)
                score += (content['rating'] / 10.0) * 0.5
                
                # Vote count score (credibility)
                max_votes = critics_choice['vote_count'].max()
                if max_votes > 0:
                    score += (content['vote_count'] / max_votes) * 0.2
                
                # Quality score
                score += content['quality_score'] * 0.2
                
                # Critics choice bonus
                if content['is_critics_choice']:
                    score += 0.1
                
                recommendations.append({
                    'content_id': content['id'],
                    'score': float(score),
                    'reason': 'Highly rated by critics and audiences with exceptional quality'
                })
            
            # Sort by score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            result = {
                'recommendations': recommendations[:limit],
                'strategy': 'critics_choice_quality_weighted',
                'cached': False
            }
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting critics choice: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}

# Initialize ML service
ml_service = MLRecommendationService()

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    health_status = {
        'status': 'healthy' if models_initialized else 'initializing',
        'timestamp': datetime.now().isoformat(),
        'models_initialized': models_initialized,
        'last_update': model_store.last_update.isoformat() if model_store.last_update else None,
        'update_count': model_store.update_count,
        'cache_size': len(ml_service.cache),
        'data_status': {
            'content_count': len(model_store.content_df) if model_store.content_df is not None else 0,
            'interactions_count': len(model_store.interactions_df) if model_store.interactions_df is not None else 0,
            'users_count': len(model_store.users_df) if model_store.users_df is not None else 0
        },
        'engine_status': {
            'collaborative_trained': model_store.is_initialized() and ml_service.hybrid_engine.collaborative_engine.trained,
            'content_based_ready': model_store.content_df is not None and not model_store.content_df.empty,
            'semantic_ready': ml_service.hybrid_engine.semantic_engine.embeddings is not None
        }
    }
    
    return jsonify(health_status), 200

@app.route('/api/update-models', methods=['POST'])
def update_models():
    """Force comprehensive model update"""
    try:
        success = ml_service.update_models()
        
        return jsonify({
            'status': 'success' if success else 'error',
            'message': 'Models updated successfully' if success else 'Model update failed',
            'timestamp': datetime.now().isoformat(),
            'models_initialized': models_initialized
        }), 200 if success else 500
        
    except Exception as e:
        logger.error(f"Model update API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_comprehensive_stats():
    """Get comprehensive ML service statistics"""
    try:
        stats = {
            'service_status': 'operational' if models_initialized else 'initializing',
            'timestamp': datetime.now().isoformat(),
            'models_initialized': models_initialized,
            'last_update': model_store.last_update.isoformat() if model_store.last_update else None,
            'update_count': model_store.update_count,
            'cache_statistics': {
                'cache_size': len(ml_service.cache),
                'cache_ttl_seconds': ml_service.cache_ttl,
                'cache_hit_potential': len(ml_service.cache) > 0
            },
            'data_statistics': {
                'total_content': len(model_store.content_df) if model_store.content_df is not None else 0,
                'total_interactions': len(model_store.interactions_df) if model_store.interactions_df is not None else 0,
                'unique_users': len(model_store.users_df) if model_store.users_df is not None else 0,
                'content_types': {},
                'interaction_types': {}
            },
            'model_performance': {
                'collaborative_filtering': {
                    'trained': ml_service.hybrid_engine.collaborative_engine.trained,
                    'models_count': len(ml_service.hybrid_engine.collaborative_engine.models)
                },
                'content_based': {
                    'ready': model_store.content_df is not None and not model_store.content_df.empty,
                    'similarity_matrices': len(ml_service.hybrid_engine.content_engine.similarity_matrices)
                },
                'semantic_similarity': {
                    'ready': ml_service.hybrid_engine.semantic_engine.embeddings is not None,
                    'model_loaded': ml_service.hybrid_engine.semantic_engine.model is not None
                }
            },
            'recommendation_coverage': {
                'trending_ready': model_store.trending_weights is not None,
                'personalization_ready': model_store.users_df is not None and not model_store.users_df.empty,
                'similarity_ready': ml_service.hybrid_engine.content_engine.similarity_matrices != {}
            }
        }
        
        # Add detailed data statistics
        if model_store.content_df is not None and not model_store.content_df.empty:
            stats['data_statistics']['content_types'] = model_store.content_df['content_type'].value_counts().to_dict()
        
        if model_store.interactions_df is not None and not model_store.interactions_df.empty:
            stats['data_statistics']['interaction_types'] = model_store.interactions_df['interaction_type'].value_counts().to_dict()
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({'error': str(e)}), 500

# Recommendation API Endpoints
@app.route('/api/trending', methods=['GET'])
def get_trending():
    """Get advanced trending recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        content_type = request.args.get('content_type', 'all')
        region = request.args.get('region')
        language = request.args.get('language')
        
        result = ml_service.get_trending_recommendations(limit, content_type, region, language)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Trending API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_personalized_recommendations():
    """Get comprehensive personalized recommendations"""
    try:
        user_data = request.get_json() or {}
        limit = int(request.args.get('limit', 20))
        
        result = ml_service.get_personalized_recommendations(user_data, limit)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Personalized recommendations API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/similar/<int:content_id>', methods=['GET'])
def get_similar_recommendations(content_id):
    """Get comprehensive similar content recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        
        result = ml_service.get_similar_recommendations(content_id, limit)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Similar recommendations API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/genre/<genre>', methods=['GET'])
def get_genre_recommendations(genre):
    """Get advanced genre-based recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        content_type = request.args.get('content_type', 'movie')
        region = request.args.get('region')
        
        result = ml_service.get_genre_recommendations(genre, limit, content_type, region)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Genre recommendations API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/regional/<language>', methods=['GET'])
def get_regional_recommendations(language):
    """Get advanced regional/language recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        content_type = request.args.get('content_type', 'movie')
        
        result = ml_service.get_regional_recommendations(language, limit, content_type)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Regional recommendations API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/anime', methods=['GET'])
def get_anime_recommendations():
    """Get specialized anime recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        genre = request.args.get('genre')
        
        result = ml_service.get_anime_recommendations(limit, genre)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Anime recommendations API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/new-releases', methods=['GET'])
def get_new_releases():
    """Get advanced new releases recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        content_type = request.args.get('content_type', 'movie')
        language = request.args.get('language')
        
        result = ml_service.get_new_releases(limit, content_type, language)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"New releases API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/critics-choice', methods=['GET'])
def get_critics_choice():
    """Get advanced critics choice recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        content_type = request.args.get('content_type', 'movie')
        
        result = ml_service.get_critics_choice(limit, content_type)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Critics choice API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

# User Profile API
@app.route('/api/user-profile/<int:user_id>', methods=['GET'])
def get_user_profile(user_id):
    """Get comprehensive user profile"""
    try:
        if not model_store.is_initialized():
            return jsonify({'error': 'Models not initialized'}), 503
        
        user_profile = UserProfilingEngine.build_comprehensive_user_profile(
            user_id, model_store.interactions_df, model_store.content_df, model_store.users_df
        )
        
        return jsonify(user_profile), 200
        
    except Exception as e:
        logger.error(f"User profile API error: {e}")
        return jsonify({'error': str(e)}), 500

# Cache Management
@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear recommendation cache"""
    try:
        ml_service.cache.clear()
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics"""
    try:
        return jsonify({
            'cache_size': len(ml_service.cache),
            'cache_ttl_seconds': ml_service.cache_ttl,
            'cached_keys': list(ml_service.cache.keys()) if len(ml_service.cache) < 50 else f"{len(ml_service.cache)} keys (too many to list)"
        }), 200
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize models on startup
def initialize_models_async():
    """Initialize models asynchronously"""
    try:
        logger.info("Starting ML service initialization...")
        success = ml_service.update_models()
        if success:
            logger.info("ML service initialization completed successfully")
        else:
            logger.warning("ML service initialization completed with warnings")
    except Exception as e:
        logger.error(f"ML service initialization failed: {e}")

# Run initialization in background
executor.submit(initialize_models_async)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting ML Recommendation Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)