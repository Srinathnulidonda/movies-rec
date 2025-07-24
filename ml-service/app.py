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
import asyncio
import aiohttp
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
from functools import lru_cache, wraps
import requests
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any, Union

# Advanced ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, rbf_kernel
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Advanced Collaborative Filtering
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import ItemItemRecommender, CosineRecommender
from implicit.lmf import LogisticMatrixFactorization
from lightfm import LightFM
from lightfm.data import Dataset as LightFMDataset

# Similarity and Indexing
import faiss
from annoy import AnnoyIndex

# Scientific Computing
from scipy import sparse
from scipy.stats import pearsonr, spearmanr, zscore
from scipy.spatial.distance import cosine, euclidean, manhattan, jaccard
from scipy.special import expit
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Advanced NLP
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Graph Analysis
import networkx as nx

# Performance
from numba import jit, prange
import joblib

# Caching
from diskcache import Cache
from cachetools import TTLCache, LRUCache

# Utilities
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
executor = ThreadPoolExecutor(max_workers=6)

# Advanced caching
memory_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes TTL
disk_cache = Cache('/tmp/ml_cache', size_limit=int(1e9))  # 1GB disk cache

# Ultra-Advanced Model Store
class UltraAdvancedModelStore:
    def __init__(self):
        # Core Data
        self.content_df = None
        self.interactions_df = None
        self.users_df = None
        
        # Advanced Feature Matrices
        self.content_tfidf_matrix = None
        self.content_embeddings = None
        self.user_item_matrix = None
        self.item_features_matrix = None
        self.user_features_matrix = None
        
        # Multiple Recommendation Models
        self.collaborative_models = {}
        self.content_similarity_indices = {}
        self.semantic_indices = {}
        
        # Advanced Analytics
        self.user_clusters = {}
        self.content_clusters = {}
        self.temporal_patterns = {}
        self.behavioral_models = {}
        
        # Real-time Components
        self.streaming_buffer = deque(maxlen=1000)
        self.real_time_weights = {}
        self.trending_signals = {}
        
        # Performance Indices
        self.faiss_index = None
        self.annoy_indices = {}
        
        # Metadata and Mappings
        self.content_metadata = {}
        self.user_metadata = {}
        self.genre_embeddings = {}
        self.language_embeddings = {}
        
        # Model Performance Tracking
        self.model_metrics = {}
        self.recommendation_quality = {}
        
        # Update tracking
        self.last_update = None
        self.update_count = 0
        self.partial_updates = 0
        
    def is_initialized(self):
        return (self.content_df is not None and 
                len(self.content_df) > 0 and
                len(self.collaborative_models) > 0)
    
    def get_cache_stats(self):
        return {
            'memory_cache_size': len(memory_cache),
            'disk_cache_size': len(disk_cache) if disk_cache else 0,
            'streaming_buffer_size': len(self.streaming_buffer),
            'faiss_index_ready': self.faiss_index is not None,
            'collaborative_models': list(self.collaborative_models.keys())
        }

# Global model store
model_store = UltraAdvancedModelStore()

# Ultra-Advanced Data Processor
class UltraAdvancedDataProcessor:
    """Ultra-advanced data processing with real-time capabilities"""
    
    @staticmethod
    async def fetch_data_async():
        """Asynchronous data fetching for better performance"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Create async tasks for all endpoints
            endpoints = [
                (f"{BACKEND_URL}/api/admin/content/all", 'content'),
                (f"{BACKEND_URL}/api/admin/interactions/all", 'interactions'),
                (f"{BACKEND_URL}/api/admin/users/all", 'users')
            ]
            
            for url, data_type in endpoints:
                tasks.append(UltraAdvancedDataProcessor._fetch_endpoint(session, url, data_type))
            
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            content_data, interactions_data, users_data = [], [], []
            for result in results:
                if isinstance(result, dict):
                    if result['type'] == 'content':
                        content_data = result['data']
                    elif result['type'] == 'interactions':
                        interactions_data = result['data']
                    elif result['type'] == 'users':
                        users_data = result['data']
            
            return content_data, interactions_data, users_data
    
    @staticmethod
    async def _fetch_endpoint(session, url, data_type):
        """Fetch single endpoint asynchronously"""
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return {'type': data_type, 'data': data.get(data_type, [])}
                else:
                    logger.warning(f"Failed to fetch {data_type}: HTTP {response.status}")
                    return {'type': data_type, 'data': []}
        except Exception as e:
            logger.error(f"Error fetching {data_type}: {e}")
            return {'type': data_type, 'data': []}
    
    @staticmethod
    def fetch_comprehensive_data():
        """Fetch data with async support and fallback"""
        try:
            # Try async approach first
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            content_data, interactions_data, users_data = loop.run_until_complete(
                UltraAdvancedDataProcessor.fetch_data_async()
            )
            loop.close()
            
            if content_data or interactions_data:
                logger.info(f"Async fetch successful: {len(content_data)} content, {len(interactions_data)} interactions, {len(users_data)} users")
                return content_data, interactions_data, users_data
        except Exception as e:
            logger.warning(f"Async fetch failed, falling back to sync: {e}")
        
        # Fallback to synchronous approach
        max_retries = 3
        for attempt in range(max_retries):
            try:
                content_response = requests.get(f"{BACKEND_URL}/api/admin/content/all", timeout=30)
                interactions_response = requests.get(f"{BACKEND_URL}/api/admin/interactions/all", timeout=30)
                users_response = requests.get(f"{BACKEND_URL}/api/admin/users/all", timeout=30)
                
                content_data = content_response.json().get('content', []) if content_response.status_code == 200 else []
                interactions_data = interactions_response.json().get('interactions', []) if interactions_response.status_code == 200 else []
                users_data = users_response.json().get('users', []) if users_response.status_code == 200 else []
                
                if content_data or interactions_data:
                    logger.info(f"Sync fetch successful: {len(content_data)} content, {len(interactions_data)} interactions, {len(users_data)} users")
                    return content_data, interactions_data, users_data
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"All fetch attempts failed: {e}")
                    break
                time.sleep(2 ** attempt)
        
        # Final fallback to comprehensive sample data
        return UltraAdvancedDataProcessor.create_ultra_realistic_sample_data()
    
    @staticmethod
    def create_ultra_realistic_sample_data():
        """Create ultra-realistic sample data with advanced patterns"""
        logger.info("Creating ultra-realistic sample data with advanced behavioral patterns")
        
        # Advanced content generation
        content_data = []
        
        # Realistic genre combinations with weights
        genre_combinations = [
            (['Action', 'Adventure'], 0.15),
            (['Comedy', 'Romance'], 0.12),
            (['Drama', 'Thriller'], 0.10),
            (['Horror', 'Mystery'], 0.08),
            (['Sci-Fi', 'Action'], 0.09),
            (['Animation', 'Family'], 0.08),
            (['Crime', 'Drama'], 0.07),
            (['Fantasy', 'Adventure'], 0.06),
            (['Romance', 'Drama'], 0.08),
            (['Thriller', 'Crime'], 0.06),
            (['Comedy', 'Family'], 0.05),
            (['Documentary'], 0.03),
            (['Musical', 'Romance'], 0.02),
            (['War', 'Drama'], 0.01)
        ]
        
        languages = ['english', 'hindi', 'telugu', 'tamil', 'kannada', 'malayalam', 'japanese', 'korean', 'spanish', 'french', 'german', 'italian']
        content_types = ['movie', 'tv', 'anime']
        
        # Create 2000 diverse content items
        for i in range(1, 2001):
            # Select genre combination based on weights
            genres, _ = zip(*genre_combinations)
            weights = [w for _, w in genre_combinations]
            selected_genres = np.random.choice(len(genres), p=np.array(weights)/sum(weights))
            genre_list = list(genres[selected_genres])
            
            # Content type selection with realistic distribution
            content_type = np.random.choice(content_types, p=[0.55, 0.30, 0.15])
            
            # Language selection with realistic regional distribution
            if content_type == 'anime':
                language = 'japanese' if np.random.random() < 0.9 else np.random.choice(['english', 'korean'])
            else:
                language_probs = [0.35, 0.12, 0.08, 0.08, 0.04, 0.04, 0.15, 0.05, 0.03, 0.03, 0.02, 0.01]
                language = np.random.choice(languages, p=language_probs)
            
            # Advanced rating generation with genre bias
            base_rating = 6.5
            if 'Drama' in genre_list:
                base_rating += 0.8
            if 'Action' in genre_list:
                base_rating += 0.3
            if 'Comedy' in genre_list:
                base_rating += 0.1
            if content_type == 'anime':
                base_rating += 0.6
            if 'Documentary' in genre_list:
                base_rating += 0.9
            
            # Add noise and clamp
            rating = max(1.0, min(10.0, np.random.normal(base_rating, 1.1)))
            
            # Popularity based on multiple factors
            popularity_base = np.random.lognormal(3.5, 1.0)
            if content_type == 'anime' and language == 'japanese':
                popularity_base *= 1.3
            if 'Action' in genre_list or 'Adventure' in genre_list:
                popularity_base *= 1.2
            if rating > 8.5:
                popularity_base *= 1.4
            
            # Release date with seasonal patterns
            if content_type == 'anime':
                # Anime tends to be more recent
                days_ago = np.random.exponential(300)
            else:
                days_ago = np.random.exponential(800)
            
            release_date = datetime.now() - timedelta(days=int(days_ago))
            
            # Determine special flags
            is_new_release = days_ago <= 90
            is_trending = (days_ago <= 30 and popularity_base > 80) or np.random.random() < 0.03
            is_critics_choice = (rating >= 8.7 and np.random.random() < 0.4) or np.random.random() < 0.08
            
            # Generate realistic runtime
            if content_type == 'movie':
                runtime = max(80, int(np.random.normal(115, 25)))
            elif content_type == 'anime':
                runtime = max(20, int(np.random.normal(24, 5)))
            else:  # TV
                runtime = max(20, int(np.random.normal(45, 15)))
            
            # Vote count based on popularity and age
            base_votes = max(10, int(np.random.lognormal(6, 1.5)))
            if popularity_base > 50:
                base_votes *= 2
            if days_ago > 365:
                base_votes = int(base_votes * (1 + days_ago / 1000))
            
            content_data.append({
                'id': i,
                'tmdb_id': i * 10 + np.random.randint(1, 10),
                'mal_id': i if content_type == 'anime' else None,
                'title': f"{genre_list[0]} {content_type.title()} - {i}",
                'original_title': f"Original {genre_list[0]} {i}",
                'content_type': content_type,
                'genres': json.dumps(genre_list),
                'languages': json.dumps([language]),
                'rating': round(rating, 1),
                'popularity': round(popularity_base, 2),
                'release_date': release_date.strftime('%Y-%m-%d'),
                'overview': UltraAdvancedDataProcessor._generate_realistic_overview(genre_list, content_type),
                'runtime': runtime,
                'vote_count': base_votes,
                'is_trending': is_trending,
                'is_new_release': is_new_release,
                'is_critics_choice': is_critics_choice,
                'poster_path': f'/poster_{i}.jpg',
                'backdrop_path': f'/backdrop_{i}.jpg',
                'youtube_trailer_id': f'trailer_{i}',
                'created_at': release_date.isoformat(),
                'updated_at': datetime.now().isoformat()
            })
        
        # Advanced user generation with realistic profiles
        users_data = []
        for i in range(1, 201):  # 200 diverse users
            # Create realistic user personas
            user_type = np.random.choice(['casual', 'enthusiast', 'binge_watcher', 'critic', 'anime_fan'], 
                                       p=[0.4, 0.25, 0.15, 0.1, 0.1])
            
            if user_type == 'anime_fan':
                preferred_genres = np.random.choice(
                    ['Animation', 'Action', 'Adventure', 'Fantasy', 'Sci-Fi', 'Romance'],
                    size=np.random.randint(3, 6), replace=False
                ).tolist()
                preferred_languages = ['japanese', 'english']
            elif user_type == 'critic':
                preferred_genres = np.random.choice(
                    ['Drama', 'Thriller', 'Crime', 'Documentary', 'War', 'Biography'],
                    size=np.random.randint(2, 4), replace=False
                ).tolist()
                preferred_languages = np.random.choice(languages, size=np.random.randint(1, 3), replace=False).tolist()
            else:
                preferred_genres = np.random.choice(
                    ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Animation'],
                    size=np.random.randint(2, 5), replace=False
                ).tolist()
                preferred_languages = np.random.choice(languages, size=np.random.randint(1, 3), replace=False).tolist()
            
            # User location influences language preferences
            location = np.random.choice(['India', 'USA', 'Japan', 'UK', 'Canada', 'Germany', 'France', 'South Korea'], 
                                      p=[0.3, 0.25, 0.1, 0.1, 0.08, 0.05, 0.05, 0.07])
            
            if location == 'India' and 'english' not in preferred_languages:
                preferred_languages.extend(['hindi', 'telugu', 'tamil'])
            elif location == 'Japan' and 'japanese' not in preferred_languages:
                preferred_languages.append('japanese')
            
            users_data.append({
                'id': i,
                'username': f'user_{i}',
                'email': f'user_{i}@example.com',
                'preferred_languages': json.dumps(list(set(preferred_languages))),
                'preferred_genres': json.dumps(preferred_genres),
                'location': location,
                'user_type': user_type,
                'created_at': (datetime.now() - timedelta(days=np.random.randint(1, 730))).isoformat(),
                'last_active': (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat(),
                'is_admin': i <= 10  # First 10 users are admins
            })
        
        # Ultra-realistic interaction generation with behavioral patterns
        interactions_data = []
        interaction_id = 1
        
        for user_id in range(1, 201):
            user_data = users_data[user_id - 1]
            user_type = user_data['user_type']
            user_preferences = json.loads(user_data['preferred_genres'])
            user_languages = json.loads(user_data['preferred_languages'])
            
            # Interaction count based on user type
            if user_type == 'binge_watcher':
                num_interactions = np.random.poisson(80)
            elif user_type == 'enthusiast':
                num_interactions = np.random.poisson(60)
            elif user_type == 'anime_fan':
                num_interactions = np.random.poisson(45)
            elif user_type == 'critic':
                num_interactions = np.random.poisson(35)
            else:  # casual
                num_interactions = np.random.poisson(25)
            
            # Generate interactions with realistic temporal patterns
            user_sessions = UltraAdvancedDataProcessor._generate_user_sessions(num_interactions)
            
            for session_interactions in user_sessions:
                for interaction_data in session_interactions:
                    # Select content based on user preferences
                    if np.random.random() < 0.75:  # 75% preference-based
                        matching_content = [
                            c for c in content_data 
                            if (any(genre in json.loads(c['genres']) for genre in user_preferences) or
                                any(lang in json.loads(c['languages']) for lang in user_languages))
                        ]
                        if matching_content:
                            content = np.random.choice(matching_content)
                        else:
                            content = np.random.choice(content_data)
                    else:  # 25% exploration
                        content = np.random.choice(content_data)
                    
                    interaction_type = interaction_data['type']
                    timestamp = interaction_data['timestamp']
                    
                    # Generate realistic ratings based on user type and content
                    rating = None
                    if interaction_type in ['like', 'favorite']:
                        if user_type == 'critic':
                            rating = max(7, min(10, int(np.random.normal(8.2, 1.0))))
                        else:
                            rating = max(6, min(10, int(np.random.normal(7.8, 1.2))))
                    elif interaction_type == 'view' and np.random.random() < 0.4:
                        if user_type == 'critic':
                            rating = max(4, min(10, int(np.random.normal(7.0, 1.5))))
                        else:
                            rating = max(5, min(10, int(np.random.normal(7.2, 1.3))))
                    
                    interactions_data.append({
                        'id': interaction_id,
                        'user_id': user_id,
                        'content_id': content['id'],
                        'interaction_type': interaction_type,
                        'rating': rating,
                        'timestamp': timestamp.isoformat()
                    })
                    interaction_id += 1
        
        logger.info(f"Generated ultra-realistic data: {len(content_data)} content, {len(interactions_data)} interactions, {len(users_data)} users")
        return content_data, interactions_data, users_data
    
    @staticmethod
    def _generate_realistic_overview(genres, content_type):
        """Generate realistic content overviews"""
        templates = {
            'Action': [
                "An adrenaline-pumping {content_type} featuring intense action sequences and compelling characters.",
                "High-octane {content_type} with spectacular fight scenes and edge-of-your-seat moments.",
                "Explosive {content_type} that delivers non-stop action and thrilling adventures."
            ],
            'Drama': [
                "A deeply emotional {content_type} exploring complex human relationships and profound themes.",
                "Powerful dramatic {content_type} that examines the depths of human experience.",
                "Moving {content_type} with outstanding performances and thought-provoking storytelling."
            ],
            'Comedy': [
                "Hilarious {content_type} guaranteed to keep you laughing with clever wit and humor.",
                "Side-splitting {content_type} featuring memorable characters and comedic situations.",
                "Feel-good {content_type} with perfect timing and brilliant comedic performances."
            ],
            'Horror': [
                "Spine-chilling {content_type} that will keep you on the edge of your seat.",
                "Terrifying {content_type} with psychological depth and atmospheric tension.",
                "Haunting {content_type} that masterfully blends horror with compelling storytelling."
            ]
        }
        
        primary_genre = genres[0] if genres else 'Drama'
        template = np.random.choice(templates.get(primary_genre, templates['Drama']))
        
        additional_elements = [
            "Features stunning cinematography and exceptional direction.",
            "Showcases breakthrough performances from a talented cast.",
            "Combines entertainment with meaningful social commentary.",
            "Delivers an unforgettable viewing experience.",
            "Represents the pinnacle of modern filmmaking."
        ]
        
        overview = template.format(content_type=content_type)
        if np.random.random() < 0.7:
            overview += " " + np.random.choice(additional_elements)
        
        return overview
    
    @staticmethod
    def _generate_user_sessions(total_interactions):
        """Generate realistic user sessions with temporal patterns"""
        sessions = []
        remaining_interactions = total_interactions
        
        while remaining_interactions > 0:
            # Session size (realistic binge patterns)
            session_size = min(
                remaining_interactions,
                max(1, int(np.random.lognormal(1.5, 0.8)))
            )
            
            # Session start time (realistic viewing patterns)
            base_time = datetime.now() - timedelta(days=np.random.exponential(60))
            
            # Prefer evening hours for viewing
            hour = max(0, min(23, int(np.random.normal(20, 3))))
            session_start = base_time.replace(hour=hour, minute=np.random.randint(0, 60))
            
            session_interactions = []
            for i in range(session_size):
                # Interaction types with realistic probabilities
                if i == 0:  # First interaction in session is usually a view
                    interaction_type = 'view'
                else:
                    interaction_type = np.random.choice(
                        ['view', 'like', 'favorite', 'watchlist', 'search'],
                        p=[0.6, 0.2, 0.05, 0.1, 0.05]
                    )
                
                # Time between interactions in session (minutes)
                if i == 0:
                    interaction_time = session_start
                else:
                    minutes_gap = max(1, int(np.random.exponential(15)))
                    interaction_time = session_interactions[-1]['timestamp'] + timedelta(minutes=minutes_gap)
                
                session_interactions.append({
                    'type': interaction_type,
                    'timestamp': interaction_time
                })
            
            sessions.append(session_interactions)
            remaining_interactions -= session_size
        
        return sessions
    
    @staticmethod
    def ultra_preprocess_content_data(content_data):
        """Ultra-advanced content preprocessing with deep feature engineering"""
        try:
            df = pd.DataFrame(content_data)
            if df.empty:
                return df
            
            # Basic preprocessing
            df['genres_list'] = df['genres'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else [])
            df['languages_list'] = df['languages'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else [])
            
            # Advanced text processing
            df['genre_text'] = df['genres_list'].apply(lambda x: ' '.join(x))
            df['language_text'] = df['languages_list'].apply(lambda x: ' '.join(x))
            
            # Deep text analysis
            df['overview_clean'] = df['overview'].fillna('').apply(
                lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower())
            )
            
            # Sentiment and readability analysis
            df['overview_sentiment'] = df['overview'].fillna('').apply(
                lambda x: TextBlob(str(x)).sentiment.polarity
            )
            df['overview_subjectivity'] = df['overview'].fillna('').apply(
                lambda x: TextBlob(str(x)).sentiment.subjectivity
            )
            df['overview_length'] = df['overview'].fillna('').apply(len)
            df['overview_words'] = df['overview'].fillna('').apply(lambda x: len(str(x).split()))
            
            # Advanced feature combinations
            df['combined_features'] = (
                df['title'].fillna('') + ' ' +
                df['overview_clean'] + ' ' +
                df['genre_text'] + ' ' +
                df['language_text'] + ' ' +
                df['content_type'].fillna('')
            )
            
            # Genre diversity and popularity
            df['genre_count'] = df['genres_list'].apply(len)
            df['language_count'] = df['languages_list'].apply(len)
            
            # Temporal features
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['release_year'] = df['release_date'].dt.year
            df['release_month'] = df['release_date'].dt.month
            df['release_day_of_year'] = df['release_date'].dt.dayofyear
            df['release_quarter'] = df['release_date'].dt.quarter
            
            current_date = datetime.now()
            df['content_age_days'] = (current_date - df['release_date']).dt.days
            df['content_age_years'] = df['content_age_days'] / 365.25
            
            # Popularity and quality metrics
            numerical_cols = ['rating', 'popularity', 'runtime', 'vote_count']
            for col in numerical_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())
            
            # Advanced quality scoring
            df['vote_weight'] = np.log1p(df['vote_count'])
            df['weighted_rating'] = (df['rating'] * df['vote_weight']) / (df['vote_weight'] + 10)
            
            df['quality_score'] = (
                (df['weighted_rating'] / 10.0) * 0.4 +
                (np.log1p(df['vote_count']) / np.log1p(df['vote_count'].max())) * 0.25 +
                (df['popularity'] / df['popularity'].max()) * 0.2 +
                ((df['overview_sentiment'] + 1) / 2) * 0.1 +
                (1 - df['overview_subjectivity']) * 0.05
            )
            
            # Recency and trending factors
            df['recency_score'] = np.exp(-df['content_age_days'] / 365.0)
            df['trend_momentum'] = df['is_trending'].astype(int) * df['recency_score']
            
            # Genre and language encoding with advanced techniques
            all_genres = set()
            for genres in df['genres_list']:
                all_genres.update(genres)
            
            all_languages = set()
            for languages in df['languages_list']:
                all_languages.update(languages)
            
            # One-hot encoding for genres
            for genre in all_genres:
                df[f'genre_{genre.lower().replace(" ", "_").replace("-", "_")}'] = df['genres_list'].apply(
                    lambda x: 1 if genre in x else 0
                )
            
            # One-hot encoding for languages
            for language in all_languages:
                df[f'lang_{language.lower()}'] = df['languages_list'].apply(
                    lambda x: 1 if language in x else 0
                )
            
            # Content type encoding
            content_type_dummies = pd.get_dummies(df['content_type'], prefix='type')
            df = pd.concat([df, content_type_dummies], axis=1)
            
            # Seasonal and temporal patterns
            df['is_summer_release'] = df['release_month'].isin([6, 7, 8]).astype(int)
            df['is_holiday_release'] = df['release_month'].isin([11, 12, 1]).astype(int)
            df['is_weekend_prime'] = df['release_day_of_year'].apply(
                lambda x: 1 if x % 7 in [5, 6] else 0
            )
            
            # Normalize numerical features
            scaler = RobustScaler()
            numerical_features = [col for col in df.columns if col.endswith('_norm')]
            
            for col in numerical_cols:
                if col in df.columns and len(df) > 1:
                    df[f'{col}_norm'] = scaler.fit_transform(df[[col]].fillna(0))
            
            logger.info(f"Ultra-processed content data: {len(df)} items with {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error in ultra preprocessing content data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def ultra_preprocess_interactions_data(interactions_data):
        """Ultra-advanced interactions preprocessing with behavioral analysis"""
        try:
            df = pd.DataFrame(interactions_data)
            if df.empty:
                return df
            
            # Temporal processing
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Advanced interaction weights with context
            base_weights = {
                'view': 1.0,
                'like': 2.5,
                'favorite': 4.0,
                'watchlist': 3.0,
                'search': 0.5,
                'share': 2.0,
                'comment': 3.5,
                'rate': 2.0,
                'download': 3.0,
                'follow': 1.5
            }
            
            df['base_weight'] = df['interaction_type'].map(base_weights).fillna(1.0)
            
            # Multi-level time decay
            max_timestamp = df['timestamp'].max()
            df['days_ago'] = (max_timestamp - df['timestamp']).dt.days
            df['hours_ago'] = (max_timestamp - df['timestamp']).dt.total_seconds() / 3600
            
            # Multiple recency weights for different purposes
            df['short_term_weight'] = np.exp(-df['hours_ago'] / 168)  # 1 week half-life
            df['medium_term_weight'] = np.exp(-df['days_ago'] / 30)   # 1 month half-life
            df['long_term_weight'] = np.exp(-df['days_ago'] / 90)     # 3 month half-life
            
            # Rating-based adjustments
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df['rating_weight'] = df['rating'].fillna(7.0) / 10.0
            df['rating_confidence'] = df['rating'].notna().astype(float)
            
            # User behavior patterns
            user_stats = df.groupby('user_id').agg({
                'timestamp': ['count', 'min', 'max'],
                'interaction_type': lambda x: x.nunique(),
                'base_weight': 'sum'
            }).round(3)
            
            user_stats.columns = ['total_interactions', 'first_interaction', 'last_interaction', 'interaction_variety', 'total_weight']
            user_stats['user_tenure_days'] = (user_stats['last_interaction'] - user_stats['first_interaction']).dt.days
            user_stats['interaction_frequency'] = user_stats['total_interactions'] / (user_stats['user_tenure_days'] + 1)
            
            # Merge user stats back
            df = df.merge(user_stats[['interaction_frequency', 'interaction_variety']], left_on='user_id', right_index=True, how='left')
            
            # Content popularity from interactions
            content_stats = df.groupby('content_id').agg({
                'user_id': 'nunique',
                'base_weight': 'sum',
                'rating': 'mean'
            }).round(3)
            content_stats.columns = ['unique_users', 'total_engagement', 'avg_user_rating']
            
            df = df.merge(content_stats, left_on='content_id', right_index=True, how='left')
            
            # Session identification
            df = df.sort_values(['user_id', 'timestamp'])
            df['time_gap'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
            df['new_session'] = (df['time_gap'] > 3600).astype(int)  # 1 hour gap = new session
            df['session_id'] = df.groupby('user_id')['new_session'].cumsum()
            
            # Session-level features
            session_stats = df.groupby(['user_id', 'session_id']).agg({
                'interaction_type': 'count',
                'base_weight': 'sum',
                'timestamp': ['min', 'max']
            })
            session_stats.columns = ['session_length', 'session_weight', 'session_start', 'session_end']
            session_stats['session_duration_minutes'] = (session_stats['session_end'] - session_stats['session_start']).dt.total_seconds() / 60
            
            df = df.merge(session_stats[['session_length', 'session_duration_minutes']], 
                         left_on=['user_id', 'session_id'], right_index=True, how='left')
            
            # Final composite weights
            df['final_weight'] = (
                df['base_weight'] * 
                df['short_term_weight'] * 
                df['rating_weight'] * 
                (1 + np.log1p(df['session_length']) * 0.1)
            )
            
            # Temporal patterns
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 23)).astype(int)
            df['is_prime_time'] = ((df['hour'] >= 19) & (df['hour'] <= 22)).astype(int)
            
            logger.info(f"Ultra-processed interactions data: {len(df)} interactions with advanced behavioral features")
            return df
            
        except Exception as e:
            logger.error(f"Error in ultra preprocessing interactions data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def ultra_preprocess_users_data(users_data):
        """Ultra-advanced user preprocessing with personality profiling"""
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
            
            # Temporal features
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['last_active'] = pd.to_datetime(df['last_active'], errors='coerce')
            
            current_time = datetime.now()
            df['account_age_days'] = (current_time - df['created_at']).dt.days
            df['last_active_days'] = (current_time - df['last_active']).dt.days
            df['last_active_hours'] = (current_time - df['last_active']).dt.total_seconds() / 3600
            
            # User engagement classification
            df['activity_level'] = pd.cut(
                df['last_active_days'],
                bins=[-1, 1, 7, 30, 90, float('inf')],
                labels=['very_active', 'active', 'moderate', 'low', 'inactive']
            )
            
            # Preference diversity
            df['genre_diversity'] = df['preferred_genres_list'].apply(len)
            df['language_diversity'] = df['preferred_languages_list'].apply(len)
            
            # User type encoding if available
            if 'user_type' in df.columns:
                user_type_dummies = pd.get_dummies(df['user_type'], prefix='type')
                df = pd.concat([df, user_type_dummies], axis=1)
            
            # Location-based features
            if 'location' in df.columns:
                location_dummies = pd.get_dummies(df['location'], prefix='location')
                df = pd.concat([df, location_dummies], axis=1)
            
            # Preference encoding
            all_genres = set()
            for genres in df['preferred_genres_list']:
                all_genres.update(genres)
            
            for genre in all_genres:
                df[f'prefers_{genre.lower().replace(" ", "_")}'] = df['preferred_genres_list'].apply(
                    lambda x: 1 if genre in x else 0
                )
            
            all_languages = set()
            for languages in df['preferred_languages_list']:
                all_languages.update(languages)
            
            for language in all_languages:
                df[f'speaks_{language.lower()}'] = df['preferred_languages_list'].apply(
                    lambda x: 1 if language in x else 0
                )
            
            logger.info(f"Ultra-processed users data: {len(df)} users with personality profiling")
            return df
            
        except Exception as e:
            logger.error(f"Error in ultra preprocessing users data: {e}")
            return pd.DataFrame()

# Ultra-Advanced Collaborative Filtering
class UltraAdvancedCollaborativeFiltering:
    """State-of-the-art collaborative filtering with multiple advanced algorithms"""
    
    def __init__(self):
        self.models = {}
        self.user_item_matrix = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.user_factors = None
        self.item_factors = None
        self.trained = False
        
        # Initialize multiple models
        self.models = {
            'als': AlternatingLeastSquares(factors=150, iterations=30, regularization=0.01, random_state=42),
            'bpr': BayesianPersonalizedRanking(factors=100, iterations=50, regularization=0.01, random_state=42),
            'lmf': LogisticMatrixFactorization(factors=80, iterations=30, regularization=0.01, random_state=42),
            'item_knn': ItemItemRecommender(K=50),
            'cosine': CosineRecommender(K=40)
        }
        
    def fit(self, interactions_df, content_df):
        """Train ensemble of collaborative filtering models"""
        try:
            if interactions_df.empty:
                return
            
            # Prepare rating matrix with advanced aggregation
            rating_data = interactions_df.groupby(['user_id', 'content_id']).agg({
                'final_weight': 'sum',
                'rating': 'mean',
                'interaction_type': lambda x: x.value_counts().index[0]  # Most frequent interaction
            }).reset_index()
            
            # Use weighted score as implicit feedback
            rating_data['implicit_score'] = rating_data['final_weight'] * rating_data['rating'].fillna(5.0)
            
            # Create mappings
            unique_users = rating_data['user_id'].unique()
            unique_items = rating_data['content_id'].unique()
            
            self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
            self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
            self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
            self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
            
            # Create sparse matrix
            rows = [self.user_mapping[user_id] for user_id in rating_data['user_id']]
            cols = [self.item_mapping[item_id] for item_id in rating_data['content_id']]
            data = rating_data['implicit_score'].values
            
            self.user_item_matrix = sparse.csr_matrix(
                (data, (rows, cols)), 
                shape=(len(unique_users), len(unique_items))
            )
            
            # Train all models
            successful_models = []
            for name, model in self.models.items():
                try:
                    if name in ['als', 'bpr', 'lmf']:
                        model.fit(self.user_item_matrix)
                        # Store factors for advanced recommendations
                        if hasattr(model, 'user_factors'):
                            self.user_factors = model.user_factors
                        if hasattr(model, 'item_factors'):
                            self.item_factors = model.item_factors
                    elif name in ['item_knn', 'cosine']:
                        model.fit(self.user_item_matrix.T)  # Item-based needs transposed matrix
                    
                    successful_models.append(name)
                    logger.info(f"Successfully trained {name} model")
                    
                except Exception as e:
                    logger.warning(f"Failed to train {name} model: {e}")
                    del self.models[name]
            
            self.trained = len(successful_models) > 0
            logger.info(f"Collaborative filtering trained with {len(successful_models)} models and {len(rating_data)} interactions")
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering: {e}")
    
    def predict_rating(self, user_id, content_id):
        """Ensemble prediction from multiple models"""
        if not self.trained or user_id not in self.user_mapping or content_id not in self.item_mapping:
            return 5.0
        
        predictions = []
        weights = {'als': 0.4, 'bpr': 0.3, 'lmf': 0.2, 'item_knn': 0.05, 'cosine': 0.05}
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[content_id]
        
        for model_name, model in self.models.items():
            try:
                if model_name in ['als', 'bpr', 'lmf']:
                    # Get prediction from matrix factorization models
                    if hasattr(model, 'predict'):
                        pred = model.predict(user_idx, item_idx)
                    else:
                        # Manual prediction using factors
                        pred = np.dot(model.user_factors[user_idx], model.item_factors[item_idx])
                    
                    predictions.append(pred * weights.get(model_name, 0.1))
                    
            except Exception as e:
                continue
        
        if predictions:
            # Normalize ensemble prediction to 1-10 scale
            ensemble_pred = sum(predictions)
            return max(1.0, min(10.0, ensemble_pred * 2 + 5))
        else:
            return 5.0
    
    def get_user_recommendations(self, user_id, content_df, interactions_df, n_recommendations=50):
        """Get ensemble collaborative filtering recommendations"""
        try:
            if not self.trained or user_id not in self.user_mapping:
                return []
            
            user_idx = self.user_mapping[user_id]
            all_recommendations = defaultdict(float)
            model_weights = {'als': 0.4, 'bpr': 0.3, 'lmf': 0.2, 'item_knn': 0.05, 'cosine': 0.05}
            
            # Get recommendations from each model
            for model_name, model in self.models.items():
                try:
                    if model_name in ['als', 'bpr', 'lmf']:
                        recs = model.recommend(
                            user_idx, 
                            self.user_item_matrix[user_idx], 
                            N=n_recommendations * 2,
                            filter_already_liked_items=True
                        )
                        
                        for item_idx, score in recs:
                            content_id = self.reverse_item_mapping[item_idx]
                            weighted_score = score * model_weights.get(model_name, 0.1)
                            all_recommendations[content_id] += weighted_score
                            
                except Exception as e:
                    logger.warning(f"Error getting recommendations from {model_name}: {e}")
                    continue
            
            # Sort and format recommendations
            sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for content_id, score in sorted_recs[:n_recommendations]:
                recommendations.append({
                    'content_id': content_id,
                    'score': float(score),
                    'reason': 'Advanced ensemble collaborative filtering based on users with similar taste'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {e}")
            return []
    
    def get_similar_items(self, content_id, content_df, n_recommendations=20):
        """Get similar items using multiple collaborative approaches"""
        try:
            if not self.trained or content_id not in self.item_mapping:
                return []
            
            item_idx = self.item_mapping[content_id]
            all_recommendations = defaultdict(float)
            
            # Use item-based models
            item_models = ['item_knn', 'cosine']
            for model_name in item_models:
                if model_name in self.models:
                    try:
                        model = self.models[model_name]
                        similar_items = model.similar_items(item_idx, N=n_recommendations * 2)
                        
                        for other_item_idx, score in similar_items:
                            other_content_id = self.reverse_item_mapping[other_item_idx]
                            if other_content_id != content_id:
                                all_recommendations[other_content_id] += score * 0.5
                                
                    except Exception as e:
                        continue
            
            # Use matrix factorization item similarities
            if self.item_factors is not None:
                try:
                    item_vector = self.item_factors[item_idx]
                    similarities = np.dot(self.item_factors, item_vector)
                    
                    top_indices = np.argsort(similarities)[::-1][1:n_recommendations*2+1]
                    for idx in top_indices:
                        other_content_id = self.reverse_item_mapping[idx]
                        score = similarities[idx]
                        if score > 0.1:  # Minimum similarity threshold
                            all_recommendations[other_content_id] += score * 0.5
                            
                except Exception as e:
                    logger.warning(f"Error computing factor-based similarities: {e}")
            
            # Sort and format
            sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for content_id_rec, score in sorted_recs[:n_recommendations]:
                recommendations.append({
                    'content_id': content_id_rec,
                    'score': float(score),
                    'reason': 'Advanced collaborative similarity analysis'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting similar items: {e}")
            return []

# Ultra-Advanced Content-Based Filtering
class UltraAdvancedContentBasedFiltering:
    """Ultra-advanced content-based filtering with multiple similarity approaches"""
    
    def __init__(self):
        # Multiple vectorizers for different aspects
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=15000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True,
            analyzer='word'
        )
        
        self.char_vectorizer = TfidfVectorizer(
            max_features=5000,
            analyzer='char_wb',
            ngram_range=(2, 4),
            min_df=2
        )
        
        # Feature matrices
        self.tfidf_matrix = None
        self.char_matrix = None
        self.numerical_matrix = None
        self.combined_matrix = None
        
        # Similarity indices
        self.similarity_matrices = {}
        self.faiss_index = None
        self.content_embeddings = None
        
    def fit(self, content_df):
        """Train ultra-advanced content-based models"""
        try:
            if content_df.empty:
                return
            
            # Text feature extraction
            text_features = content_df['combined_features'].fillna('')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
            
            # Character-level features for language similarity
            title_overview = (content_df['title'].fillna('') + ' ' + content_df['overview'].fillna('')).str[:500]
            self.char_matrix = self.char_vectorizer.fit_transform(title_overview)
            
            # Numerical features matrix
            numerical_cols = [col for col in content_df.columns if col.endswith('_norm') or 
                            col.startswith('genre_') or col.startswith('lang_') or col.startswith('type_')]
            
            if numerical_cols:
                self.numerical_matrix = sparse.csr_matrix(content_df[numerical_cols].fillna(0).values)
            else:
                self.numerical_matrix = sparse.csr_matrix((len(content_df), 1))
            
            # Combine all features
            self.combined_matrix = sparse.hstack([
                self.tfidf_matrix,
                self.char_matrix,
                self.numerical_matrix
            ])
            
            # Pre-compute similarity matrices
            logger.info("Computing similarity matrices...")
            
            # Cosine similarity (most common)
            self.similarity_matrices['cosine'] = cosine_similarity(self.combined_matrix)
            
            # Linear kernel (faster alternative)
            self.similarity_matrices['linear'] = linear_kernel(self.combined_matrix)
            
            # Initialize FAISS for fast similarity search
            try:
                if self.combined_matrix.shape[1] > 0:
                    dense_matrix = self.combined_matrix.toarray().astype('float32')
                    
                    # L2 normalize for cosine similarity
                    norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    dense_matrix = dense_matrix / norms
                    
                    # Build FAISS index
                    dimension = dense_matrix.shape[1]
                    self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                    self.faiss_index.add(dense_matrix)
                    
                    self.content_embeddings = dense_matrix
                    logger.info(f"FAISS index built with {len(dense_matrix)} items and {dimension} dimensions")
                    
            except Exception as e:
                logger.warning(f"Failed to build FAISS index: {e}")
            
            logger.info(f"Ultra-advanced content model trained with {content_df.shape[0]} items")
            
        except Exception as e:
            logger.error(f"Error training ultra-advanced content model: {e}")
    
    def get_content_similarities(self, content_id, content_df, similarity_type='cosine', n_recommendations=50):
        """Get content similarities using specified method"""
        try:
            if content_id not in content_df['id'].values:
                return []
            
            content_idx = content_df[content_df['id'] == content_id].index[0]
            
            # Use FAISS for fast similarity search if available
            if self.faiss_index is not None and similarity_type == 'cosine':
                query_vector = self.content_embeddings[content_idx:content_idx+1]
                similarities, indices = self.faiss_index.search(query_vector, n_recommendations + 1)
                
                recommendations = []
                for i, (score, idx) in enumerate(zip(similarities[0], indices[0])):
                    if idx != content_idx and score > 0.1:  # Exclude self and low similarity
                        other_content_id = content_df.iloc[idx]['id']
                        recommendations.append({
                            'content_id': other_content_id,
                            'score': float(score),
                            'reason': 'Ultra-advanced content similarity with semantic understanding'
                        })
                
                return recommendations[:n_recommendations]
            
            # Fallback to pre-computed similarity matrices
            elif similarity_type in self.similarity_matrices:
                similarity_matrix = self.similarity_matrices[similarity_type]
                sim_scores = similarity_matrix[content_idx]
                
                recommendations = []
                for idx, score in enumerate(sim_scores):
                    if idx != content_idx and score > 0.1:
                        other_content_id = content_df.iloc[idx]['id']
                        recommendations.append({
                            'content_id': other_content_id,
                            'score': float(score),
                            'reason': f'Content similarity using {similarity_type} metric'
                        })
                
                recommendations.sort(key=lambda x: x['score'], reverse=True)
                return recommendations[:n_recommendations]
            
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting content similarities: {e}")
            return []

# Ultra-Advanced Semantic Similarity Engine
class UltraAdvancedSemanticEngine:
    """Ultra-advanced semantic similarity with multiple transformer models"""
    
    def __init__(self):
        self.models = {}
        self.embeddings = {}
        self.content_ids = None
        self.faiss_indices = {}
        
    def initialize(self):
        """Initialize multiple sentence transformer models"""
        model_configs = [
            ('primary', 'all-MiniLM-L6-v2'),
            ('multilingual', 'paraphrase-multilingual-MiniLM-L12-v2'),
            ('large', 'all-MiniLM-L12-v2')
        ]
        
        for name, model_name in model_configs:
            try:
                self.models[name] = SentenceTransformer(model_name)
                logger.info(f"Loaded semantic model: {name} ({model_name})")
            except Exception as e:
                logger.warning(f"Failed to load {name} model: {e}")
                continue
        
        if not self.models:
            logger.warning("No semantic models could be loaded")
    
    def fit(self, content_df):
        """Generate embeddings using multiple models"""
        try:
            if not self.models:
                return
            
            # Prepare texts
            texts = []
            content_ids = []
            
            for _, content in content_df.iterrows():
                # Create rich text representation
                text_parts = [
                    content.get('title', ''),
                    content.get('overview', ''),
                    content.get('genre_text', ''),
                    content.get('language_text', ''),
                    content.get('content_type', '')
                ]
                
                text = ' '.join(filter(None, text_parts))
                texts.append(text)
                content_ids.append(content['id'])
            
            self.content_ids = content_ids
            
            # Generate embeddings with each model
            for model_name, model in self.models.items():
                try:
                    logger.info(f"Generating embeddings with {model_name} model...")
                    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
                    self.embeddings[model_name] = embeddings
                    
                    # Build FAISS index for fast search
                    dimension = embeddings.shape[1]
                    faiss_index = faiss.IndexFlatIP(dimension)
                    faiss_index.add(embeddings.astype('float32'))
                    self.faiss_indices[model_name] = faiss_index
                    
                    logger.info(f"Generated {len(embeddings)} embeddings with {model_name} ({dimension}D)")
                    
                except Exception as e:
                    logger.error(f"Error generating embeddings with {model_name}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error in semantic model fitting: {e}")
    
    def get_semantic_similarities(self, content_id, n_recommendations=50, model_preference='primary'):
        """Get semantically similar content using ensemble approach"""
        try:
            if not self.embeddings or content_id not in self.content_ids:
                return []
            
            content_idx = self.content_ids.index(content_id)
            all_recommendations = defaultdict(float)
            
            # Model weights for ensemble
            model_weights = {
                'primary': 0.5,
                'multilingual': 0.3,
                'large': 0.2
            }
            
            # Get recommendations from each available model
            for model_name, embeddings in self.embeddings.items():
                if model_name in self.faiss_indices:
                    try:
                        query_vector = embeddings[content_idx:content_idx+1]
                        similarities, indices = self.faiss_indices[model_name].search(
                            query_vector.astype('float32'), 
                            n_recommendations * 2
                        )
                        
                        weight = model_weights.get(model_name, 0.1)
                        for score, idx in zip(similarities[0], indices[0]):
                            if idx != content_idx and score > 0.3:  # Semantic similarity threshold
                                other_content_id = self.content_ids[idx]
                                all_recommendations[other_content_id] += float(score) * weight
                                
                    except Exception as e:
                        logger.warning(f"Error with {model_name} semantic search: {e}")
                        continue
            
            # Sort by ensemble score
            sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for content_id_rec, score in sorted_recs[:n_recommendations]:
                recommendations.append({
                    'content_id': content_id_rec,
                    'score': float(score),
                    'reason': 'Advanced semantic similarity using transformer ensemble'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting semantic similarities: {e}")
            return []

# Ultra-Advanced Trending Analysis Engine
class UltraAdvancedTrendingEngine:
    """Ultra-advanced trending analysis with real-time signals"""
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_velocity_score(interaction_counts, time_weights):
        """JIT-compiled velocity calculation for performance"""
        return np.sum(interaction_counts * time_weights)
    
    @staticmethod
    def calculate_ultra_trending_score(content_df, interactions_df, real_time_signals=None):
        """Calculate trending scores using advanced multi-signal analysis"""
        try:
            current_time = datetime.now()
            trending_scores = {}
            
            # Multiple time windows for trend detection
            time_windows = [1, 3, 6, 12, 24, 48, 168]  # hours
            window_weights = [0.3, 0.25, 0.2, 0.1, 0.08, 0.05, 0.02]
            
            # Real-time signal integration
            real_time_boost = real_time_signals or {}
            
            for _, content in content_df.iterrows():
                content_id = content['id']
                score = 0.0
                
                # Base quality and popularity foundation
                base_score = (
                    (content['quality_score'] * 0.25) +
                    (min(content['popularity'] / 100.0, 1.0) * 0.2) +
                    (content['rating'] / 10.0 * 0.15) +
                    (min(np.log1p(content['vote_count']) / 10.0, 1.0) * 0.1)
                )
                
                # Multi-window velocity analysis
                content_interactions = interactions_df[interactions_df['content_id'] == content_id]
                
                if not content_interactions.empty:
                    for window_hours, weight in zip(time_windows, window_weights):
                        window_start = current_time - timedelta(hours=window_hours)
                        window_interactions = content_interactions[
                            pd.to_datetime(content_interactions['timestamp']) >= window_start
                        ]
                        
                        if not window_interactions.empty:
                            # Advanced velocity metrics
                            interaction_velocity = len(window_interactions) / window_hours
                            unique_user_velocity = window_interactions['user_id'].nunique() / window_hours
                            weight_velocity = window_interactions['final_weight'].sum() / window_hours
                            
                            # Engagement quality
                            avg_session_length = window_interactions['session_length'].mean()
                            engagement_quality = min(avg_session_length / 5.0, 2.0)
                            
                            # Combined velocity score
                            velocity_score = (
                                interaction_velocity * 0.4 +
                                unique_user_velocity * 0.4 +
                                weight_velocity * 0.2
                            ) * engagement_quality
                            
                            score += velocity_score * weight
                
                # Content-specific boosters
                age_penalty = max(0, 1 - (content['content_age_days'] / 365.0))
                score *= (1 + age_penalty * 0.3)
                
                # Special flags
                if content['is_new_release']:
                    score *= 1.4
                if content['is_critics_choice']:
                    score *= 1.2
                if content['is_trending']:  # Historical trending
                    score *= 1.1
                
                # Content type adjustments
                type_multipliers = {'movie': 1.1, 'anime': 1.05, 'tv': 1.0}
                score *= type_multipliers.get(content['content_type'], 1.0)
                
                # Real-time signal integration
                rt_boost = real_time_boost.get(content_id, 0)
                score += rt_boost * 0.2
                
                # Seasonal and temporal adjustments
                current_month = current_time.month
                if content['release_month'] == current_month:
                    score *= 1.05  # Anniversary boost
                
                trending_scores[content_id] = base_score + score
            
            return trending_scores
            
        except Exception as e:
            logger.error(f"Error calculating ultra trending scores: {e}")
            return {}
    
    @staticmethod
    def get_regional_trending(content_df, interactions_df, language=None, region=None, cultural_signals=None):
        """Get regional trending with advanced cultural intelligence"""
        try:
            filtered_content = content_df.copy()
            
            # Advanced language matching
            if language:
                language_variants = {
                    'hindi': ['hindi', 'hi', 'bollywood', 'bhojpuri', 'urdu'],
                    'telugu': ['telugu', 'te', 'tollywood'],
                    'tamil': ['tamil', 'ta', 'kollywood'],
                    'kannada': ['kannada', 'kn', 'sandalwood'],
                    'malayalam': ['malayalam', 'ml', 'mollywood'],
                    'english': ['english', 'en', 'hollywood'],
                    'japanese': ['japanese', 'ja', 'anime', 'jpop'],
                    'korean': ['korean', 'ko', 'kdrama', 'kpop'],
                    'spanish': ['spanish', 'es', 'latino', 'telenovela'],
                    'french': ['french', 'fr', 'francophone'],
                    'german': ['german', 'de', 'deutsch'],
                    'italian': ['italian', 'it', 'cinema']
                }
                
                target_languages = language_variants.get(language.lower(), [language])
                
                def advanced_language_match(lang_list):
                    if not lang_list:
                        return False
                    
                    # Direct match
                    direct_match = any(
                        any(target.lower() in lang.lower() for target in target_languages)
                        for lang in lang_list
                    )
                    
                    # Cultural context match (e.g., Bollywood content in Hindi context)
                    cultural_match = False
                    if language.lower() == 'hindi':
                        cultural_match = any('bollywood' in str(lang).lower() for lang in lang_list)
                    elif language.lower() == 'japanese':
                        cultural_match = any('anime' in str(lang).lower() for lang in lang_list)
                    
                    return direct_match or cultural_match
                
                filtered_content = filtered_content[
                    filtered_content['languages_list'].apply(advanced_language_match)
                ]
            
            # Regional cultural preferences
            regional_preferences = {
                'India': {
                    'preferred_languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam'],
                    'genre_boost': {'Drama': 1.2, 'Romance': 1.1, 'Action': 1.1},
                    'content_type_boost': {'movie': 1.1}
                },
                'Japan': {
                    'preferred_languages': ['japanese'],
                    'genre_boost': {'Animation': 1.3, 'Fantasy': 1.2, 'Sci-Fi': 1.1},
                    'content_type_boost': {'anime': 1.4}
                },
                'South Korea': {
                    'preferred_languages': ['korean'],
                    'genre_boost': {'Drama': 1.3, 'Romance': 1.2, 'Thriller': 1.1},
                    'content_type_boost': {'tv': 1.2}
                },
                'USA': {
                    'preferred_languages': ['english'],
                    'genre_boost': {'Action': 1.2, 'Comedy': 1.1, 'Sci-Fi': 1.1},
                    'content_type_boost': {'movie': 1.1}
                }
            }
            
            # Calculate base trending scores
            trending_scores = UltraAdvancedTrendingEngine.calculate_ultra_trending_score(
                filtered_content, interactions_df
            )
            
            # Apply regional preferences
            if region and region in regional_preferences:
                prefs = regional_preferences[region]
                
                for content_id, score in trending_scores.items():
                    content_row = filtered_content[filtered_content['id'] == content_id]
                    if not content_row.empty:
                        content_data = content_row.iloc[0]
                        
                        # Language preference boost
                        content_languages = content_data['languages_list']
                        if any(lang in content_languages for lang in prefs['preferred_languages']):
                            score *= 1.3
                        
                        # Genre preference boost
                        content_genres = content_data['genres_list']
                        for genre in content_genres:
                            if genre in prefs['genre_boost']:
                                score *= prefs['genre_boost'][genre]
                        
                        # Content type boost
                        content_type = content_data['content_type']
                        if content_type in prefs['content_type_boost']:
                            score *= prefs['content_type_boost'][content_type]
                        
                        trending_scores[content_id] = score
            
            # Cultural signals integration
            if cultural_signals:
                for content_id, cultural_score in cultural_signals.items():
                    if content_id in trending_scores:
                        trending_scores[content_id] += cultural_score * 0.15
            
            return trending_scores
            
        except Exception as e:
            logger.error(f"Error getting regional trending: {e}")
            return {}

# Ultra-Advanced User Profiling Engine
class UltraAdvancedUserProfiler:
    """Ultra-advanced user profiling with behavioral AI"""
    
    @staticmethod
    def build_ultra_user_profile(user_id, interactions_df, content_df, users_df=None):
        """Build comprehensive user profile with AI-driven behavioral analysis"""
        try:
            profile = {
                'user_id': user_id,
                'personality_traits': {},
                'preferences': {},
                'behavioral_patterns': {},
                'contextual_factors': {},
                'recommendation_strategy': {}
            }
            
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            if user_interactions.empty:
                return profile
            
            # Advanced preference learning
            profile['preferences'] = UltraAdvancedUserProfiler._analyze_preferences(
                user_interactions, content_df
            )
            
            # Behavioral pattern recognition
            profile['behavioral_patterns'] = UltraAdvancedUserProfiler._analyze_behavior_patterns(
                user_interactions
            )
            
            # Personality trait inference
            profile['personality_traits'] = UltraAdvancedUserProfiler._infer_personality_traits(
                user_interactions, content_df
            )
            
            # Contextual factor analysis
            profile['contextual_factors'] = UltraAdvancedUserProfiler._analyze_contextual_factors(
                user_interactions
            )
            
            # Recommendation strategy optimization
            profile['recommendation_strategy'] = UltraAdvancedUserProfiler._optimize_recommendation_strategy(
                profile
            )
            
            # Integration with stated preferences
            if users_df is not None:
                user_data = users_df[users_df['id'] == user_id]
                if not user_data.empty:
                    stated_prefs = UltraAdvancedUserProfiler._integrate_stated_preferences(
                        user_data.iloc[0], profile
                    )
                    profile.update(stated_prefs)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error building ultra user profile: {e}")
            return {'user_id': user_id, 'preferences': {}, 'behavioral_patterns': {}}
    
    @staticmethod
    def _analyze_preferences(user_interactions, content_df):
        """Advanced preference analysis with temporal weighting"""
        preferences = {}
        
        # Get interacted content
        content_ids = user_interactions['content_id'].unique()
        user_content = content_df[content_df['id'].isin(content_ids)]
        
        # Genre preferences with temporal decay and interaction weighting
        genre_scores = defaultdict(float)
        total_weight = 0
        
        for _, interaction in user_interactions.iterrows():
            content = content_df[content_df['id'] == interaction['content_id']]
            if not content.empty:
                content_data = content.iloc[0]
                interaction_weight = interaction['final_weight']
                
                for genre in content_data['genres_list']:
                    genre_scores[genre] += interaction_weight
                    total_weight += interaction_weight
        
        if total_weight > 0:
            preferences['genres'] = {
                genre: score / total_weight 
                for genre, score in genre_scores.items()
            }
        
        # Language preferences
        language_scores = defaultdict(float)
        for _, interaction in user_interactions.iterrows():
            content = content_df[content_df['id'] == interaction['content_id']]
            if not content.empty:
                content_data = content.iloc[0]
                interaction_weight = interaction['final_weight']
                
                for language in content_data['languages_list']:
                    language_scores[language] += interaction_weight
        
        if total_weight > 0:
            preferences['languages'] = {
                lang: score / total_weight 
                for lang, score in language_scores.items()
            }
        
        # Quality preferences
        ratings = user_interactions['rating'].dropna()
        if not ratings.empty:
            preferences['quality_profile'] = {
                'avg_rating_given': float(ratings.mean()),
                'rating_std': float(ratings.std()),
                'quality_sensitivity': 'high' if ratings.std() > 1.5 else 'moderate' if ratings.std() > 0.8 else 'low',
                'prefers_high_quality': ratings.mean() > 7.5
            }
        
        # Content type preferences
        content_type_scores = defaultdict(float)
        for _, interaction in user_interactions.iterrows():
            content = content_df[content_df['id'] == interaction['content_id']]
            if not content.empty:
                content_type = content.iloc[0]['content_type']
                content_type_scores[content_type] += interaction['final_weight']
        
        if total_weight > 0:
            preferences['content_types'] = {
                ct: score / total_weight 
                for ct, score in content_type_scores.items()
            }
        
        return preferences
    
    @staticmethod
    def _analyze_behavior_patterns(user_interactions):
        """Advanced behavioral pattern analysis"""
        patterns = {}
        
        # Temporal patterns
        timestamps = pd.to_datetime(user_interactions['timestamp'])
        if not timestamps.empty:
            patterns['temporal'] = {
                'peak_hours': timestamps.dt.hour.mode().tolist(),
                'active_days': timestamps.dt.dayofweek.mode().tolist(),
                'activity_distribution': timestamps.dt.hour.value_counts().to_dict(),
                'session_patterns': UltraAdvancedUserProfiler._analyze_session_patterns(user_interactions)
            }
        
        # Interaction diversity
        interaction_types = user_interactions['interaction_type'].value_counts()
        patterns['interaction_diversity'] = {
            'dominant_interaction': interaction_types.index[0] if not interaction_types.empty else None,
            'interaction_distribution': interaction_types.to_dict(),
            'engagement_depth': UltraAdvancedUserProfiler._calculate_engagement_depth(user_interactions)
        }
        
        # Content exploration vs exploitation
        unique_content = user_interactions['content_id'].nunique()
        total_interactions = len(user_interactions)
        patterns['exploration_ratio'] = unique_content / max(total_interactions, 1)
        
        # Binge watching detection
        patterns['binge_behavior'] = UltraAdvancedUserProfiler._detect_binge_behavior(user_interactions)
        
        return patterns
    
    @staticmethod
    def _infer_personality_traits(user_interactions, content_df):
        """Infer personality traits from viewing behavior"""
        traits = {}
        
        # Get content details for analysis
        content_ids = user_interactions['content_id'].unique()
        user_content = content_df[content_df['id'].isin(content_ids)]
        
        if user_content.empty:
            return traits
        
        # Openness to experience
        genre_diversity = user_content['genres_list'].apply(len).mean()
        unique_genres = len(set(genre for content in user_content['genres_list'] for genre in content))
        traits['openness'] = min(1.0, (unique_genres + genre_diversity) / 15.0)
        
        # Conscientiousness (rating behavior, consistent viewing)
        rating_completeness = user_interactions['rating'].notna().mean()
        viewing_consistency = UltraAdvancedUserProfiler._calculate_viewing_consistency(user_interactions)
        traits['conscientiousness'] = (rating_completeness + viewing_consistency) / 2.0
        
        # Extroversion (social content, popular content preference)
        social_content_ratio = (user_content['content_type'] == 'tv').mean()
        popular_content_ratio = (user_content['popularity'] > user_content['popularity'].median()).mean()
        traits['extroversion'] = (social_content_ratio + popular_content_ratio) / 2.0
        
        # Agreeableness (family content, positive ratings)
        family_content_ratio = user_content['genres_list'].apply(
            lambda x: any(genre in ['Family', 'Comedy', 'Romance'] for genre in x)
        ).mean()
        positive_rating_ratio = (user_interactions['rating'] >= 7).sum() / max(user_interactions['rating'].notna().sum(), 1)
        traits['agreeableness'] = (family_content_ratio + positive_rating_ratio) / 2.0
        
        # Neuroticism (thriller/horror preference, rating volatility)
        intense_content_ratio = user_content['genres_list'].apply(
            lambda x: any(genre in ['Horror', 'Thriller', 'Mystery'] for genre in x)
        ).mean()
        rating_volatility = user_interactions['rating'].std() / 10.0 if user_interactions['rating'].notna().sum() > 1 else 0
        traits['neuroticism'] = (intense_content_ratio + rating_volatility) / 2.0
        
        return traits
    
    @staticmethod
    def _analyze_contextual_factors(user_interactions):
        """Analyze contextual factors affecting recommendations"""
        factors = {}
        
        # Recency bias
        recent_interactions = user_interactions[
            pd.to_datetime(user_interactions['timestamp']) >= 
            (datetime.now() - timedelta(days=7))
        ]
        
        if not recent_interactions.empty:
            factors['recent_focus'] = {
                'interaction_types': recent_interactions['interaction_type'].value_counts().to_dict(),
                'activity_level': len(recent_interactions),
                'engagement_trend': UltraAdvancedUserProfiler._calculate_engagement_trend(user_interactions)
            }
        
        # Device/time context inference
        factors['usage_context'] = UltraAdvancedUserProfiler._infer_usage_context(user_interactions)
        
        return factors
    
    @staticmethod
    def _optimize_recommendation_strategy(profile):
        """Optimize recommendation strategy based on user profile"""
        strategy = {}
        
        traits = profile.get('personality_traits', {})
        patterns = profile.get('behavioral_patterns', {})
        
        # Determine primary recommendation approach
        if traits.get('openness', 0.5) > 0.7:
            strategy['primary_approach'] = 'exploration'
            strategy['diversity_weight'] = 0.4
        elif patterns.get('exploration_ratio', 0.5) > 0.6:
            strategy['primary_approach'] = 'balanced'
            strategy['diversity_weight'] = 0.3
        else:
            strategy['primary_approach'] = 'exploitation'
            strategy['diversity_weight'] = 0.2
        
        # Algorithm weighting
        if traits.get('conscientiousness', 0.5) > 0.6:
            strategy['algorithm_weights'] = {
                'collaborative': 0.4,
                'content': 0.3,
                'semantic': 0.2,
                'trending': 0.1
            }
        else:
            strategy['algorithm_weights'] = {
                'collaborative': 0.3,
                'content': 0.2,
                'semantic': 0.2,
                'trending': 0.3
            }
        
        # Personalization intensity
        interaction_count = patterns.get('interaction_diversity', {}).get('engagement_depth', 0)
        if interaction_count > 50:
            strategy['personalization_intensity'] = 'high'
        elif interaction_count > 20:
            strategy['personalization_intensity'] = 'medium'
        else:
            strategy['personalization_intensity'] = 'low'
        
        return strategy
    
    @staticmethod
    def _analyze_session_patterns(user_interactions):
        """Analyze user session patterns"""
        if 'session_id' not in user_interactions.columns:
            return {}
        
        session_stats = user_interactions.groupby('session_id').agg({
            'interaction_type': 'count',
            'final_weight': 'sum',
            'timestamp': ['min', 'max']
        })
        
        session_stats.columns = ['session_length', 'session_weight', 'session_start', 'session_end']
        session_stats['session_duration'] = (session_stats['session_end'] - session_stats['session_start']).dt.total_seconds() / 60
        
        return {
            'avg_session_length': float(session_stats['session_length'].mean()),
            'avg_session_duration': float(session_stats['session_duration'].mean()),
            'session_intensity': float(session_stats['session_weight'].mean()),
            'binge_sessions': int((session_stats['session_length'] > 5).sum())
        }
    
    @staticmethod
    def _calculate_engagement_depth(user_interactions):
        """Calculate user engagement depth score"""
        interaction_weights = {
            'view': 1, 'like': 2, 'favorite': 4, 'watchlist': 3, 
            'search': 0.5, 'share': 2.5, 'comment': 3.5, 'rate': 2
        }
        
        weighted_interactions = user_interactions['interaction_type'].map(interaction_weights).sum()
        total_interactions = len(user_interactions)
        
        return weighted_interactions / max(total_interactions, 1)
    
    @staticmethod
    def _detect_binge_behavior(user_interactions):
        """Detect binge watching patterns"""
        if 'session_length' not in user_interactions.columns:
            return {'is_binge_watcher': False, 'binge_score': 0.0}
        
        long_sessions = user_interactions['session_length'] > 3
        binge_score = long_sessions.mean() if not long_sessions.empty else 0.0
        
        return {
            'is_binge_watcher': binge_score > 0.3,
            'binge_score': float(binge_score),
            'max_session_length': int(user_interactions['session_length'].max())
        }
    
    @staticmethod
    def _calculate_viewing_consistency(user_interactions):
        """Calculate viewing consistency score"""
        if len(user_interactions) < 2:
            return 0.0
        
        # Time gaps between interactions
        timestamps = pd.to_datetime(user_interactions['timestamp']).sort_values()
        time_gaps = timestamps.diff().dt.total_seconds().dropna()
        
        if len(time_gaps) == 0:
            return 0.0
        
        # Coefficient of variation (lower = more consistent)
        cv = time_gaps.std() / time_gaps.mean() if time_gaps.mean() > 0 else float('inf')
        consistency_score = max(0.0, 1.0 - min(cv / 10.0, 1.0))
        
        return consistency_score
    
    @staticmethod
    def _calculate_engagement_trend(user_interactions):
        """Calculate recent engagement trend"""
        if len(user_interactions) < 5:
            return 'stable'
        
        # Sort by timestamp
        sorted_interactions = user_interactions.sort_values('timestamp')
        
        # Split into early and recent halves
        mid_point = len(sorted_interactions) // 2
        early_engagement = sorted_interactions.iloc[:mid_point]['final_weight'].mean()
        recent_engagement = sorted_interactions.iloc[mid_point:]['final_weight'].mean()
        
        ratio = recent_engagement / max(early_engagement, 0.1)
        
        if ratio > 1.2:
            return 'increasing'
        elif ratio < 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    @staticmethod
    def _infer_usage_context(user_interactions):
        """Infer usage context from temporal patterns"""
        timestamps = pd.to_datetime(user_interactions['timestamp'])
        
        if timestamps.empty:
            return {}
        
        # Hour distribution
        hour_dist = timestamps.dt.hour.value_counts().sort_index()
        
        # Classify viewing context
        evening_ratio = hour_dist[18:23].sum() / len(timestamps)
        weekend_ratio = timestamps[timestamps.dt.dayofweek >= 5].shape[0] / len(timestamps)
        
        context = {}
        
        if evening_ratio > 0.6:
            context['primary_time'] = 'evening'
        elif hour_dist[12:17].sum() / len(timestamps) > 0.4:
            context['primary_time'] = 'afternoon'
        else:
            context['primary_time'] = 'mixed'
        
        if weekend_ratio > 0.4:
            context['usage_pattern'] = 'weekend_focused'
        else:
            context['usage_pattern'] = 'weekday_regular'
        
        return context
    
    @staticmethod
    def _integrate_stated_preferences(user_data, profile):
        """Integrate stated user preferences with inferred preferences"""
        integration = {}
        
        # Compare stated vs inferred preferences
        stated_genres = user_data.get('preferred_genres_list', [])
        inferred_genres = list(profile.get('preferences', {}).get('genres', {}).keys())
        
        # Calculate preference alignment
        if stated_genres and inferred_genres:
            alignment = len(set(stated_genres) & set(inferred_genres)) / len(set(stated_genres) | set(inferred_genres))
            integration['preference_alignment'] = alignment
            integration['recommendation_confidence'] = 'high' if alignment > 0.6 else 'medium' if alignment > 0.3 else 'low'
        
        # Preference evolution tracking
        integration['stated_preferences'] = {
            'genres': stated_genres,
            'languages': user_data.get('preferred_languages_list', [])
        }
        
        return {'preference_integration': integration}

# Ultra-Advanced Hybrid Recommendation Engine
class UltraAdvancedHybridEngine:
    """Ultra-advanced hybrid recommendation engine with AI orchestration"""
    
    def __init__(self):
        self.collaborative_engine = UltraAdvancedCollaborativeFiltering()
        self.content_engine = UltraAdvancedContentBasedFiltering()
        self.semantic_engine = UltraAdvancedSemanticEngine()
        
        # Dynamic weighting system
        self.base_weights = {
            'collaborative': 0.35,
            'content': 0.25,
            'semantic': 0.20,
            'trending': 0.12,
            'quality': 0.08
        }
        
        # Performance tracking
        self.algorithm_performance = defaultdict(lambda: {'accuracy': 0.5, 'diversity': 0.5, 'novelty': 0.5})
        
    def fit(self, content_df, interactions_df, users_df):
        """Train all engines with performance monitoring"""
        try:
            logger.info("Training ultra-advanced hybrid recommendation engine...")
            
            # Train individual engines
            self.collaborative_engine.fit(interactions_df, content_df)
            self.content_engine.fit(content_df)
            
            # Initialize and train semantic engine
            self.semantic_engine.initialize()
            self.semantic_engine.fit(content_df)
            
            logger.info("Ultra-advanced hybrid engine training completed")
            
        except Exception as e:
            logger.error(f"Error training ultra-advanced hybrid engine: {e}")
    
    def get_ultra_personalized_recommendations(self, user_id, content_df, interactions_df, users_df, 
                                             user_profile=None, n_recommendations=50, context=None):
        """Get ultra-personalized recommendations with AI orchestration"""
        try:
            # Build or use user profile
            if user_profile is None:
                user_profile = UltraAdvancedUserProfiler.build_ultra_user_profile(
                    user_id, interactions_df, content_df, users_df
                )
            
            # Dynamic weight adjustment based on user profile
            adjusted_weights = self._adjust_weights_for_user(user_profile)
            
            # Context-aware adjustments
            if context:
                adjusted_weights = self._apply_contextual_adjustments(adjusted_weights, context)
            
            # Get recommendations from each engine
            all_recommendations = defaultdict(float)
            recommendation_sources = defaultdict(list)
            
            # 1. Collaborative filtering with ensemble
            collab_recs = self.collaborative_engine.get_user_recommendations(
                user_id, content_df, interactions_df, n_recommendations * 3
            )
            
            for rec in collab_recs:
                content_id = rec['content_id']
                score = rec['score'] * adjusted_weights['collaborative']
                all_recommendations[content_id] += score
                recommendation_sources[content_id].append('collaborative')
            
            # 2. Content-based recommendations with user preference focus
            user_genres = user_profile.get('preferences', {}).get('genres', {})
            top_genres = sorted(user_genres.items(), key=lambda x: x[1], reverse=True)[:4]
            
            for genre, preference_strength in top_genres:
                genre_content = content_df[
                    content_df['genres_list'].apply(lambda x: genre in x)
                ]
                
                for _, content in genre_content.head(20).iterrows():
                    content_id = content['id']
                    # Weight by preference strength and content quality
                    score = (content['quality_score'] * preference_strength * 
                           adjusted_weights['content'] * 0.25)  # Distribute among genres
                    all_recommendations[content_id] += score
                    recommendation_sources[content_id].append('content_genre')
            
            # 3. Semantic recommendations based on recent high-engagement content
            recent_high_engagement = interactions_df[
                (interactions_df['user_id'] == user_id) &
                (interactions_df['final_weight'] > 2.0) &
                (pd.to_datetime(interactions_df['timestamp']) >= 
                 (datetime.now() - timedelta(days=21)))
            ]
            
            for _, interaction in recent_high_engagement.head(5).iterrows():
                semantic_recs = self.semantic_engine.get_semantic_similarities(
                    interaction['content_id'], n_recommendations=15
                )
                
                for rec in semantic_recs:
                    content_id = rec['content_id']
                    # Weight by original interaction strength
                    score = (rec['score'] * adjusted_weights['semantic'] * 
                           interaction['final_weight'] / 10.0)
                    all_recommendations[content_id] += score
                    recommendation_sources[content_id].append('semantic')
            
            # 4. Trending boost with personalization
            trending_scores = UltraAdvancedTrendingEngine.calculate_ultra_trending_score(
                content_df, interactions_df
            )
            
            # Apply user preference filters to trending content
            user_languages = user_profile.get('preferences', {}).get('languages', {})
            user_content_types = user_profile.get('preferences', {}).get('content_types', {})
            
            for content_id, trending_score in trending_scores.items():
                content_row = content_df[content_df['id'] == content_id]
                if not content_row.empty:
                    content_data = content_row.iloc[0]
                    
                    # Apply user preference multipliers
                    multiplier = 1.0
                    
                    # Language preference
                    content_languages = content_data['languages_list']
                    for lang in content_languages:
                        if lang in user_languages:
                            multiplier *= (1 + user_languages[lang] * 0.3)
                    
                    # Content type preference
                    content_type = content_data['content_type']
                    if content_type in user_content_types:
                        multiplier *= (1 + user_content_types[content_type] * 0.2)
                    
                    # Genre preference
                    content_genres = content_data['genres_list']
                    for genre in content_genres:
                        if genre in user_genres:
                            multiplier *= (1 + user_genres[genre] * 0.2)
                    
                    score = trending_score * adjusted_weights['trending'] * multiplier
                    all_recommendations[content_id] += score
                    recommendation_sources[content_id].append('trending')
            
            # 5. Quality and diversity optimization
            quality_boost = self._apply_quality_boost(all_recommendations, content_df, user_profile)
            diversity_adjustment = self._apply_diversity_optimization(
                all_recommendations, content_df, user_profile, recommendation_sources
            )
            
            # Combine all adjustments
            for content_id in all_recommendations.keys():
                all_recommendations[content_id] += quality_boost.get(content_id, 0)
                all_recommendations[content_id] *= diversity_adjustment.get(content_id, 1.0)
            
            # 6. Remove already interacted content
            user_interacted_content = interactions_df[interactions_df['user_id'] == user_id]['content_id'].unique()
            for content_id in user_interacted_content:
                all_recommendations.pop(content_id, None)
            
            # 7. Apply final ranking and selection
            final_recommendations = self._apply_final_ranking(
                all_recommendations, recommendation_sources, content_df, user_profile
            )
            
            return final_recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting ultra-personalized recommendations: {e}")
            return []
    
    def _adjust_weights_for_user(self, user_profile):
        """Dynamically adjust algorithm weights based on user profile"""
        weights = self.base_weights.copy()
        
        # Adjust based on personality traits
        traits = user_profile.get('personality_traits', {})
        
        # High openness users get more semantic and content diversity
        if traits.get('openness', 0.5) > 0.7:
            weights['semantic'] += 0.1
            weights['content'] += 0.05
            weights['collaborative'] -= 0.1
            weights['trending'] -= 0.05
        
        # High conscientiousness users get more collaborative (social proof)
        if traits.get('conscientiousness', 0.5) > 0.7:
            weights['collaborative'] += 0.1
            weights['trending'] -= 0.05
            weights['semantic'] -= 0.05
        
        # Extroverted users get more trending content
        if traits.get('extroversion', 0.5) > 0.7:
            weights['trending'] += 0.1
            weights['collaborative'] += 0.05
            weights['content'] -= 0.1
            weights['semantic'] -= 0.05
        
        # Adjust based on recommendation strategy
        strategy = user_profile.get('recommendation_strategy', {})
        if 'algorithm_weights' in strategy:
            strategy_weights = strategy['algorithm_weights']
            # Blend with personality-based adjustments
            for alg, weight in strategy_weights.items():
                if alg in weights:
                    weights[alg] = (weights[alg] + weight) / 2
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _apply_contextual_adjustments(self, weights, context):
        """Apply contextual adjustments to algorithm weights"""
        adjusted_weights = weights.copy()
        
        # Time-based adjustments
        if context.get('time_of_day') == 'evening':
            adjusted_weights['trending'] += 0.05
            adjusted_weights['collaborative'] += 0.05
            adjusted_weights['content'] -= 0.05
            adjusted_weights['semantic'] -= 0.05
        
        # Device context
        if context.get('device_type') == 'mobile':
            adjusted_weights['trending'] += 0.1
            adjusted_weights['collaborative'] -= 0.05
            adjusted_weights['content'] -= 0.05
        
        # Seasonal adjustments
        current_month = datetime.now().month
        if current_month in [11, 12, 1]:  # Holiday season
            adjusted_weights['trending'] += 0.1
            adjusted_weights['quality'] += 0.05
            adjusted_weights['content'] -= 0.1
            adjusted_weights['semantic'] -= 0.05
        
        # Normalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _apply_quality_boost(self, recommendations, content_df, user_profile):
        """Apply intelligent quality boost based on user preferences"""
        quality_boosts = {}
        
        quality_preference = user_profile.get('preferences', {}).get('quality_profile', {})
        prefers_high_quality = quality_preference.get('prefers_high_quality', False)
        quality_sensitivity = quality_preference.get('quality_sensitivity', 'moderate')
        
        boost_multiplier = {
            'high': 0.3,
            'moderate': 0.2,
            'low': 0.1
        }.get(quality_sensitivity, 0.2)
        
        for content_id in recommendations.keys():
            content_row = content_df[content_df['id'] == content_id]
            if not content_row.empty:
                content_data = content_row.iloc[0]
                
                # Quality boost calculation
                quality_score = content_data['quality_score']
                rating = content_data['rating']
                vote_count = content_data['vote_count']
                
                if prefers_high_quality and rating >= 8.0:
                    boost = quality_score * boost_multiplier
                    # Additional boost for well-reviewed content
                    if vote_count > 1000:
                        boost *= 1.2
                    quality_boosts[content_id] = boost
                elif not prefers_high_quality and quality_sensitivity == 'low':
                    # For users who don't prioritize quality, avoid very low quality
                    if rating < 5.0:
                        quality_boosts[content_id] = -0.1
                    else:
                        quality_boosts[content_id] = 0.0
        
        return quality_boosts
    
    def _apply_diversity_optimization(self, recommendations, content_df, user_profile, sources):
        """Apply diversity optimization to prevent filter bubbles"""
        diversity_adjustments = {}
        
        # Get user's diversity preference
        exploration_ratio = user_profile.get('behavioral_patterns', {}).get('exploration_ratio', 0.5)
        openness = user_profile.get('personality_traits', {}).get('openness', 0.5)
        
        # Calculate diversity target
        diversity_target = (exploration_ratio + openness) / 2.0
        
        # Track genre and language distribution in recommendations
        genre_counts = defaultdict(int)
        language_counts = defaultdict(int)
        content_type_counts = defaultdict(int)
        
        for content_id in recommendations.keys():
            content_row = content_df[content_df['id'] == content_id]
            if not content_row.empty:
                content_data = content_row.iloc[0]
                
                for genre in content_data['genres_list']:
                    genre_counts[genre] += 1
                
                for language in content_data['languages_list']:
                    language_counts[language] += 1
                
                content_type_counts[content_data['content_type']] += 1
        
        # Apply diversity penalties/boosts
        for content_id in recommendations.keys():
            content_row = content_df[content_df['id'] == content_id]
            if not content_row.empty:
                content_data = content_row.iloc[0]
                
                diversity_factor = 1.0
                
                # Genre diversity
                for genre in content_data['genres_list']:
                    if genre_counts[genre] > len(recommendations) * 0.3:  # Over-represented
                        diversity_factor *= (1 - diversity_target * 0.2)
                
                # Language diversity
                for language in content_data['languages_list']:
                    if language_counts[language] > len(recommendations) * 0.7:  # Over-represented
                        diversity_factor *= (1 - diversity_target * 0.1)
                
                # Content type diversity
                content_type = content_data['content_type']
                if content_type_counts[content_type] > len(recommendations) * 0.8:
                    diversity_factor *= (1 - diversity_target * 0.15)
                
                # Boost for under-represented content from multiple sources
                if len(sources[content_id]) > 2:  # Multiple algorithms agree
                    diversity_factor *= 1.1
                
                diversity_adjustments[content_id] = diversity_factor
        
        return diversity_adjustments
    
    def _apply_final_ranking(self, recommendations, sources, content_df, user_profile):
        """Apply final intelligent ranking with multiple criteria"""
        
        # Convert to list for ranking
        recommendation_items = []
        for content_id, score in recommendations.items():
            content_row = content_df[content_df['id'] == content_id]
            if not content_row.empty:
                content_data = content_row.iloc[0]
                
                # Calculate confidence score based on multiple factors
                confidence_factors = {
                    'algorithm_agreement': len(sources[content_id]) / 4.0,  # Max 4 sources
                    'content_quality': content_data['quality_score'],
                    'popularity_signal': min(content_data['popularity'] / 100.0, 1.0),
                    'recency_relevance': content_data.get('recency_score', 0.5)
                }
                
                confidence_score = sum(confidence_factors.values()) / len(confidence_factors)
                
                # Final ranking score
                final_score = score * (0.8 + confidence_score * 0.2)
                
                recommendation_items.append({
                    'content_id': content_id,
                    'score': float(final_score),
                    'confidence': float(confidence_score),
                    'sources': sources[content_id],
                    'reason': self._generate_explanation(sources[content_id], user_profile, content_data)
                })
        
        # Sort by final score
        recommendation_items.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendation_items
    
    def _generate_explanation(self, sources, user_profile, content_data):
        """Generate human-readable recommendation explanation"""
        primary_source = Counter(sources).most_common(1)[0][0] if sources else 'unknown'
        
        explanations = {
            'collaborative': "Users with similar taste also loved this",
            'content_genre': f"Matches your interest in {', '.join(content_data['genres_list'][:2])}",
            'semantic': "Similar themes to content you recently enjoyed",
            'trending': "Currently popular and trending",
            'quality': "High-quality content with excellent ratings"
        }
        
        base_explanation = explanations.get(primary_source, "Recommended for you")
        
        # Add personality-based context
        traits = user_profile.get('personality_traits', {})
        if traits.get('openness', 0.5) > 0.7 and primary_source == 'semantic':
            base_explanation += " - perfect for exploring new themes"
        elif traits.get('conscientiousness', 0.5) > 0.7 and primary_source == 'collaborative':
            base_explanation += " - trusted by similar viewers"
        elif traits.get('extroversion', 0.5) > 0.7 and primary_source == 'trending':
            base_explanation += " - join the conversation"
        
        return base_explanation

# Main Ultra-Advanced ML Service
class UltraAdvancedMLService:
    """Main ML service orchestrator with ultra-advanced capabilities"""
    
    def __init__(self):
        self.hybrid_engine = UltraAdvancedHybridEngine()
        self.cache = TTLCache(maxsize=2000, ttl=300)  # 5 minutes TTL
        self.real_time_signals = defaultdict(float)
        self.last_update = None
        self.performance_monitor = defaultdict(list)
        
    def update_models(self):
        """Update all ML models with comprehensive monitoring"""
        global model_store, models_initialized
        
        try:
            with models_lock:
                logger.info("Starting ultra-advanced model update...")
                start_time = time.time()
                
                # Fetch all data
                content_data, interactions_data, users_data = UltraAdvancedDataProcessor.fetch_comprehensive_data()
                
                # Ultra preprocess all data
                content_df = UltraAdvancedDataProcessor.ultra_preprocess_content_data(content_data)
                interactions_df = UltraAdvancedDataProcessor.ultra_preprocess_interactions_data(interactions_data)
                users_df = UltraAdvancedDataProcessor.ultra_preprocess_users_data(users_data)
                
                if content_df.empty:
                    logger.warning("No content data available for training")
                    return False
                
                # Store in model store
                model_store.content_df = content_df
                model_store.interactions_df = interactions_df
                model_store.users_df = users_df
                
                # Create comprehensive metadata
                model_store.content_metadata = {
                    row['id']: row.to_dict() 
                    for _, row in content_df.iterrows()
                }
                
                if not users_df.empty:
                    model_store.user_metadata = {
                        row['id']: row.to_dict() 
                        for _, row in users_df.iterrows()
                    }
                
                # Train ultra-advanced hybrid engine
                self.hybrid_engine.fit(content_df, interactions_df, users_df)
                
                # Calculate trending scores with real-time signals
                trending_scores = UltraAdvancedTrendingEngine.calculate_ultra_trending_score(
                    content_df, interactions_df, self.real_time_signals
                )
                model_store.trending_weights = trending_scores
                
                # Update metadata
                self.last_update = datetime.now()
                model_store.last_update = self.last_update
                model_store.update_count += 1
                models_initialized = True
                
                # Clear cache after update
                self.cache.clear()
                
                update_time = time.time() - start_time
                logger.info(f"Ultra-advanced model update completed in {update_time:.2f}s. "
                           f"Content: {len(content_df)}, Interactions: {len(interactions_df)}, Users: {len(users_df)}")
                
                # Log performance metrics
                self.performance_monitor['update_time'].append(update_time)
                self.performance_monitor['data_size'].append(len(content_df))
                
                return True
                
        except Exception as e:
            logger.error(f"Error in ultra-advanced model update: {e}")
            return False
    
    def get_cache_key(self, prefix, **kwargs):
        """Generate intelligent cache key"""
        key_parts = [prefix]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{hash(str(v)) % 1000000}")
        return "_".join(key_parts)
    
    def get_cached_result(self, cache_key):
        """Get cached result with hit tracking"""
        if cache_key in self.cache:
            result = self.cache[cache_key].copy()
            result['cached'] = True
            return result
        return None
    
    def cache_result(self, cache_key, result):
        """Cache result with metadata"""
        result_copy = result.copy()
        result_copy['cached'] = False
        result_copy['cache_timestamp'] = datetime.now().isoformat()
        self.cache[cache_key] = result_copy
    
    def get_ultra_trending_recommendations(self, limit=20, content_type='all', region=None, language=None, context=None):
        """Get ultra-advanced trending recommendations"""
        try:
            cache_key = self.get_cache_key('ultra_trending', limit=limit, content_type=content_type, 
                                         region=region, language=language, context=str(context))
            cached = self.get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            interactions_df = model_store.interactions_df
            
            # Filter by content type
            if content_type != 'all':
                content_df = content_df[content_df['content_type'] == content_type]
            
            # Calculate trending scores with cultural intelligence
            if region or language:
                trending_scores = UltraAdvancedTrendingEngine.get_regional_trending(
                    content_df, interactions_df, language=language, region=region
                )
            else:
                trending_scores = model_store.trending_weights or {}
            
            # Apply contextual adjustments
            if context:
                trending_scores = self._apply_trending_context_adjustments(trending_scores, context, content_df)
            
            # Sort and format
            trending_items = sorted(trending_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for content_id, score in trending_items[:limit]:
                if content_id in model_store.content_metadata:
                    content_data = model_store.content_metadata[content_id]
                    
                    recommendations.append({
                        'content_id': content_id,
                        'score': float(score),
                        'reason': self._generate_trending_reason(content_data, region, language, context),
                        'trending_velocity': self._calculate_trending_velocity(content_id, interactions_df),
                        'cultural_relevance': self._calculate_cultural_relevance(content_data, region, language)
                    })
            
            result = {
                'recommendations': recommendations,
                'strategy': 'ultra_advanced_trending_analysis',
                'region_focus': region,
                'language_focus': language,
                'context_applied': context is not None,
                'cached': False
            }
            
            self.cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting ultra trending recommendations: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_ultra_personalized_recommendations(self, user_data, limit=20, context=None):
        """Get ultra-personalized recommendations with AI orchestration"""
        try:
            user_id = user_data.get('user_id')
            cache_key = self.get_cache_key('ultra_personalized', user_id=user_id, limit=limit, context=str(context))
            cached = self.get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            interactions_df = model_store.interactions_df
            users_df = model_store.users_df
            
            # Build ultra user profile
            user_profile = UltraAdvancedUserProfiler.build_ultra_user_profile(
                user_id, interactions_df, content_df, users_df
            )
            
            # Get ultra-personalized recommendations
            recommendations = self.hybrid_engine.get_ultra_personalized_recommendations(
                user_id, content_df, interactions_df, users_df, user_profile, limit * 2, context
            )
            
            # Apply final personalization filters
            final_recommendations = self._apply_ultra_personalization_filters(
                recommendations, user_data, user_profile, limit, context
            )
            
            result = {
                'recommendations': final_recommendations,
                'strategy': 'ultra_advanced_hybrid_personalized',
                'user_profile_insights': {
                    'personality_type': self._classify_user_personality(user_profile),
                    'recommendation_readiness': self._assess_recommendation_readiness(user_profile),
                    'diversity_preference': user_profile.get('behavioral_patterns', {}).get('exploration_ratio', 0.5),
                    'quality_sensitivity': user_profile.get('preferences', {}).get('quality_profile', {}).get('quality_sensitivity', 'moderate')
                },
                'personalization_confidence': self._calculate_personalization_confidence(user_profile, len(recommendations)),
                'cached': False
            }
            
            self.cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting ultra-personalized recommendations: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_ultra_similar_recommendations(self, content_id, limit=20, context=None):
        """Get ultra-advanced similar content recommendations"""
        try:
            cache_key = self.get_cache_key('ultra_similar', content_id=content_id, limit=limit, context=str(context))
            cached = self.get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            all_recommendations = defaultdict(float)
            similarity_explanations = {}
            
            # Multi-algorithm similarity ensemble
            algorithms = [
                ('content_cosine', 0.35),
                ('content_linear', 0.25),
                ('semantic_primary', 0.25),
                ('collaborative', 0.15)
            ]
            
            for algo_name, weight in algorithms:
                try:
                    if algo_name.startswith('content_'):
                        similarity_type = algo_name.split('_')[1]
                        similarities = self.hybrid_engine.content_engine.get_content_similarities(
                            content_id, content_df, similarity_type, limit * 2
                        )
                    elif algo_name.startswith('semantic_'):
                        similarities = self.hybrid_engine.semantic_engine.get_semantic_similarities(
                            content_id, limit * 2
                        )
                    elif algo_name == 'collaborative':
                        similarities = self.hybrid_engine.collaborative_engine.get_similar_items(
                            content_id, content_df, limit * 2
                        )
                    else:
                        continue
                    
                    for sim in similarities:
                        sim_content_id = sim['content_id']
                        score = sim['score'] * weight
                        all_recommendations[sim_content_id] += score
                        
                        if sim_content_id not in similarity_explanations:
                            similarity_explanations[sim_content_id] = []
                        similarity_explanations[sim_content_id].append(algo_name)
                        
                except Exception as e:
                    logger.warning(f"Error with {algo_name} similarity: {e}")
                    continue
            
            # Apply context-aware adjustments
            if context:
                all_recommendations = self._apply_similarity_context_adjustments(
                    all_recommendations, content_id, context, content_df
                )
            
            # Sort and format
            sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for sim_content_id, score in sorted_recs[:limit]:
                algorithms_used = similarity_explanations.get(sim_content_id, [])
                consensus_strength = len(algorithms_used) / len(algorithms)
                
                recommendations.append({
                    'content_id': sim_content_id,
                    'score': float(score),
                    'consensus_strength': float(consensus_strength),
                    'algorithms_used': algorithms_used,
                    'reason': self._generate_similarity_reason(algorithms_used, consensus_strength)
                })
            
            result = {
                'recommendations': recommendations,
                'strategy': 'ultra_multi_algorithm_similarity',
                'base_content_id': content_id,
                'algorithms_applied': len(algorithms),
                'context_adjusted': context is not None,
                'cached': False
            }
            
            self.cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting ultra similar recommendations: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_ultra_genre_recommendations(self, genre, limit=20, content_type='movie', region=None, context=None):
        """Get ultra-advanced genre recommendations"""
        try:
            cache_key = self.get_cache_key('ultra_genre', genre=genre, limit=limit, 
                                         content_type=content_type, region=region, context=str(context))
            cached = self.get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            
            # Advanced genre filtering with fuzzy matching
            filtered_df = content_df[content_df['content_type'] == content_type]
            
            def advanced_genre_match(genres_list):
                if not genres_list:
                    return False
                
                # Direct match
                direct_match = any(genre.lower() in g.lower() for g in genres_list)
                
                # Fuzzy genre matching
                genre_synonyms = {
                    'action': ['adventure', 'thriller'],
                    'drama': ['biography', 'history'],
                    'comedy': ['family', 'romantic comedy'],
                    'horror': ['thriller', 'mystery'],
                    'sci-fi': ['fantasy', 'adventure'],
                    'romance': ['drama', 'comedy']
                }
                
                fuzzy_match = False
                if genre.lower() in genre_synonyms:
                    related_genres = genre_synonyms[genre.lower()]
                    fuzzy_match = any(
                        any(related.lower() in g.lower() for g in genres_list)
                        for related in related_genres
                    )
                
                return direct_match or fuzzy_match
            
            genre_content = filtered_df[filtered_df['genres_list'].apply(advanced_genre_match)]
            
            # Ultra-advanced scoring
            recommendations = []
            for _, content in genre_content.iterrows():
                score = 0.0
                
                # Multi-factor quality scoring
                score += content['quality_score'] * 0.35
                score += (content['rating'] / 10.0) * 0.25
                score += min(content['popularity'] / 100.0, 1.0) * 0.2
                score += min(np.log1p(content['vote_count']) / 10.0, 1.0) * 0.1
                score += content.get('recency_score', 0.5) * 0.1
                
                # Genre specificity bonus
                exact_match = any(genre.lower() == g.lower() for g in content['genres_list'])
                if exact_match:
                    score *= 1.2
                
                # Special flags
                if content['is_new_release']:
                    score *= 1.15
                if content['is_critics_choice']:
                    score *= 1.1
                if content['is_trending']:
                    score *= 1.05
                
                # Regional adjustment
                if region:
                    regional_boost = self._calculate_regional_relevance(content, region)
                    score *= (1 + regional_boost * 0.3)
                
                # Context adjustments
                if context:
                    context_adjustment = self._apply_genre_context_adjustments(content, context, genre)
                    score *= context_adjustment
                
                recommendations.append({
                    'content_id': content['id'],
                    'score': float(score),
                    'genre_match_strength': 1.0 if exact_match else 0.7,
                    'quality_indicators': {
                        'rating': content['rating'],
                        'vote_count': content['vote_count'],
                        'quality_score': content['quality_score']
                    },
                    'reason': f'Top-quality {genre} {content_type} with excellent ratings and user engagement'
                })
            
            # Sort and limit
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            result = {
                'recommendations': recommendations[:limit],
                'strategy': 'ultra_advanced_genre_filtering',
                'genre_focus': genre,
                'content_type': content_type,
                'total_matches': len(recommendations),
                'region_optimized': region is not None,
                'context_applied': context is not None,
                'cached': False
            }
            
            self.cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting ultra genre recommendations: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_ultra_regional_recommendations(self, language, limit=20, content_type='movie', context=None):
        """Get ultra-advanced regional recommendations"""
        try:
            # Use trending with advanced regional optimization
            result = self.get_ultra_trending_recommendations(
                limit=limit, 
                content_type=content_type, 
                language=language,
                context=context
            )
            result['strategy'] = 'ultra_regional_trending_optimized'
            return result
            
        except Exception as e:
            logger.error(f"Error getting ultra regional recommendations: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_ultra_anime_recommendations(self, limit=20, genre=None, context=None):
        """Get ultra-specialized anime recommendations"""
        try:
            cache_key = self.get_cache_key('ultra_anime', limit=limit, genre=genre, context=str(context))
            cached = self.get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            
            # Filter anime content
            anime_df = content_df[content_df['content_type'] == 'anime']
            
            if genre:
                # Ultra-advanced anime genre filtering
                anime_genre_intelligence = {
                    'shonen': {
                        'keywords': ['action', 'adventure', 'shounen', 'fighting', 'martial arts', 'battle'],
                        'demographic': 'young_male',
                        'themes': ['friendship', 'power', 'competition']
                    },
                    'shojo': {
                        'keywords': ['romance', 'drama', 'shoujo', 'slice of life', 'school', 'magical girl'],
                        'demographic': 'young_female', 
                        'themes': ['love', 'relationships', 'emotions']
                    },
                    'seinen': {
                        'keywords': ['thriller', 'psychological', 'seinen', 'mature', 'dark', 'complex'],
                        'demographic': 'adult_male',
                        'themes': ['psychology', 'philosophy', 'realism']
                    },
                    'josei': {
                        'keywords': ['romance', 'drama', 'josei', 'adult', 'realistic', 'workplace'],
                        'demographic': 'adult_female',
                        'themes': ['mature_relationships', 'career', 'life']
                    },
                    'isekai': {
                        'keywords': ['fantasy', 'adventure', 'isekai', 'parallel world', 'reincarnation'],
                        'demographic': 'mixed',
                        'themes': ['escapism', 'power_fantasy', 'world_building']
                    },
                    'mecha': {
                        'keywords': ['mecha', 'robot', 'sci-fi', 'action', 'pilot'],
                        'demographic': 'mixed',
                        'themes': ['technology', 'war', 'humanity']
                    }
                }
                
                genre_info = anime_genre_intelligence.get(genre.lower(), {})
                keywords = genre_info.get('keywords', [genre.lower()])
                
                def ultra_anime_genre_match(genres_list):
                    if not genres_list:
                        return False
                    genre_text = ' '.join(genres_list).lower()
                    return any(keyword in genre_text for keyword in keywords)
                
                anime_df = anime_df[anime_df['genres_list'].apply(ultra_anime_genre_match)]
            
            # Ultra-specialized anime scoring
            recommendations = []
            for _, content in anime_df.iterrows():
                score = 0.0
                
                # Anime-specific quality factors
                score += (content['rating'] / 10.0) * 0.4  # Rating is crucial for anime
                
                # Popularity with anime context
                if anime_df['popularity'].max() > 0:
                    score += (content['popularity'] / anime_df['popularity'].max()) * 0.25
                
                # Recency bonus (anime community values current seasons)
                if content['content_age_years'] <= 2:
                    score += 0.2
                elif content['content_age_years'] <= 5:
                    score += 0.1
                
                # Community engagement (vote count very important for anime)
                if content['vote_count'] > 5000:
                    score += 0.15
                elif content['vote_count'] > 1000:
                    score += 0.1
                
                # Anime-specific quality indicators
                if content['rating'] >= 9.0:  # Masterpiece tier
                    score += 0.2
                elif content['rating'] >= 8.5:  # Excellent tier
                    score += 0.1
                
                # Context adjustments for anime
                if context:
                    score *= self._apply_anime_context_adjustments(content, context, genre)
                
                recommendations.append({
                    'content_id': content['id'],
                    'score': float(score),
                    'anime_tier': self._classify_anime_tier(content),
                    'community_rating': content['rating'],
                    'popularity_rank': self._calculate_anime_popularity_rank(content, anime_df),
                    'reason': f'Top-tier {genre or "anime"} with exceptional community ratings and engagement'
                })
            
            # Sort by score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            result = {
                'recommendations': recommendations[:limit],
                'strategy': 'ultra_anime_specialized_intelligence',
                'anime_genre_focus': genre,
                'total_anime_analyzed': len(anime_df),
                'genre_intelligence_applied': genre is not None,
                'context_optimized': context is not None,
                'cached': False
            }
            
            self.cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting ultra anime recommendations: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_ultra_new_releases(self, limit=20, content_type='movie', language=None, context=None):
        """Get ultra-advanced new releases recommendations"""
        try:
            cache_key = self.get_cache_key('ultra_new_releases', limit=limit, content_type=content_type, 
                                         language=language, context=str(context))
            cached = self.get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            
            # Ultra-advanced new release filtering
            current_date = datetime.now()
            
            # Multiple recency tiers
            recency_tiers = {
                'brand_new': 30,      # Last 30 days
                'very_recent': 60,    # Last 60 days  
                'recent': 120,        # Last 120 days
                'somewhat_recent': 180 # Last 180 days
            }
            
            new_releases = content_df[
                (content_df['is_new_release'] == True) | 
                (content_df['content_age_days'] <= recency_tiers['somewhat_recent'])
            ]
            
            # Filter by content type
            new_releases = new_releases[new_releases['content_type'] == content_type]
            
            # Ultra-advanced language filtering
            if language:
                def ultra_language_match(lang_list):
                    if not lang_list:
                        return False
                    
                    language_variants = {
                        'hindi': ['hindi', 'hi', 'bollywood', 'bhojpuri'],
                        'telugu': ['telugu', 'te', 'tollywood'],
                        'tamil': ['tamil', 'ta', 'kollywood'],
                        'kannada': ['kannada', 'kn', 'sandalwood'],
                        'malayalam': ['malayalam', 'ml', 'mollywood'],
                        'english': ['english', 'en', 'hollywood'],
                        'japanese': ['japanese', 'ja', 'anime'],
                        'korean': ['korean', 'ko', 'kdrama', 'kpop']
                    }
                    
                    target_languages = language_variants.get(language.lower(), [language])
                    return any(
                        any(target.lower() in lang.lower() for target in target_languages)
                        for lang in lang_list
                    )
                
                new_releases = new_releases[new_releases['languages_list'].apply(ultra_language_match)]
            
            # Ultra-advanced scoring for new releases
            recommendations = []
            for _, content in new_releases.iterrows():
                score = 0.0
                
                # Recency tier scoring
                days_old = content['content_age_days']
                if days_old <= recency_tiers['brand_new']:
                    recency_score = 1.0
                elif days_old <= recency_tiers['very_recent']:
                    recency_score = 0.8
                elif days_old <= recency_tiers['recent']:
                    recency_score = 0.6
                else:
                    recency_score = 0.4
                
                score += recency_score * 0.4
                
                # Quality and potential
                score += content['quality_score'] * 0.25
                score += (content['rating'] / 10.0) * 0.2
                
                # Early adoption indicators
                if content['vote_count'] > 100:  # Has some reviews
                    score += 0.1
                if content['popularity'] > 20:  # Getting attention
                    score += 0.05
                
                # Trending new release bonus
                if content['is_trending'] and days_old <= 30:
                    score += 0.15
                
                # Critics early recognition
                if content['is_critics_choice'] and days_old <= 60:
                    score += 0.1
                
                # Context adjustments
                if context:
                    score *= self._apply_new_release_context_adjustments(content, context, language)
                
                # Calculate release momentum
                momentum_score = self._calculate_release_momentum(content, model_store.interactions_df)
                score += momentum_score * 0.1
                
                recommendations.append({
                    'content_id': content['id'],
                    'score': float(score),
                    'recency_tier': self._classify_recency_tier(days_old, recency_tiers),
                    'release_momentum': float(momentum_score),
                    'early_indicators': {
                        'rating': content['rating'],
                        'vote_count': content['vote_count'],
                        'popularity': content['popularity']
                    },
                    'reason': f'Fresh {content_type} release with high potential and early positive signals'
                })
            
            # Sort by score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            result = {
                'recommendations': recommendations[:limit],
                'strategy': 'ultra_new_releases_momentum_analysis',
                'recency_focus': True,
                'language_optimized': language is not None,
                'momentum_analyzed': True,
                'context_applied': context is not None,
                'cached': False
            }
            
            self.cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting ultra new releases: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    def get_ultra_critics_choice(self, limit=20, content_type='movie', context=None):
        """Get ultra-advanced critics choice recommendations"""
        try:
            cache_key = self.get_cache_key('ultra_critics_choice', limit=limit, content_type=content_type, 
                                         context=str(context))
            cached = self.get_cached_result(cache_key)
            if cached:
                return cached
            
            if not model_store.is_initialized():
                return {'recommendations': [], 'strategy': 'fallback', 'cached': False}
            
            content_df = model_store.content_df
            
            # Ultra-advanced critics choice filtering
            critics_choice = content_df[
                ((content_df['is_critics_choice'] == True) | 
                 (content_df['rating'] >= 8.0) |
                 (content_df['quality_score'] >= 0.8)) &
                (content_df['content_type'] == content_type) &
                (content_df['vote_count'] >= 50)  # Minimum credibility threshold
            ]
            
            # Ultra-advanced scoring for critics choice
            recommendations = []
            for _, content in critics_choice.iterrows():
                score = 0.0
                
                # Primary quality indicators
                score += (content['rating'] / 10.0) * 0.4
                score += content['quality_score'] * 0.3
                
                # Credibility and consensus
                credibility_score = min(np.log1p(content['vote_count']) / 10.0, 1.0)
                score += credibility_score * 0.15
                
                # Critical acclaim indicators
                if content['is_critics_choice']:
                    score += 0.1
                
                # Exceptional quality bonuses
                if content['rating'] >= 9.0:
                    score += 0.15  # Masterpiece tier
                elif content['rating'] >= 8.5:
                    score += 0.1   # Excellent tier
                
                # Balanced appeal (not just niche)
                if content['popularity'] > 10:  # Has broader appeal
                    score += 0.05
                
                # Context adjustments
                if context:
                    score *= self._apply_critics_choice_context_adjustments(content, context)
                
                # Calculate critical consensus strength
                consensus_strength = self._calculate_critical_consensus(content)
                score += consensus_strength * 0.1
                
                recommendations.append({
                    'content_id': content['id'],
                    'score': float(score),
                    'quality_tier': self._classify_quality_tier(content),
                    'critical_consensus': float(consensus_strength),
                    'credibility_score': float(credibility_score),
                    'critical_indicators': {
                        'rating': content['rating'],
                        'vote_count': content['vote_count'],
                        'quality_score': content['quality_score'],
                        'is_critics_choice': content['is_critics_choice']
                    },
                    'reason': 'Critically acclaimed masterpiece with exceptional ratings and widespread recognition'
                })
            
            # Sort by score
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            result = {
                'recommendations': recommendations[:limit],
                'strategy': 'ultra_critics_choice_consensus_analysis',
                'quality_focus': 'critical_excellence',
                'consensus_analyzed': True,
                'credibility_weighted': True,
                'context_applied': context is not None,
                'cached': False
            }
            
            self.cache_result(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting ultra critics choice: {e}")
            return {'recommendations': [], 'strategy': 'error', 'cached': False}
    
    # Helper methods for context adjustments and analysis
    def _apply_trending_context_adjustments(self, trending_scores, context, content_df):
        """Apply context-aware adjustments to trending scores"""
        adjusted_scores = trending_scores.copy()
        
        for content_id, score in trending_scores.items():
            content_row = content_df[content_df['id'] == content_id]
            if not content_row.empty:
                content_data = content_row.iloc[0]
                adjustment = 1.0
                
                # Time context
                if context.get('time_preference') == 'evening':
                    if any(genre in ['Drama', 'Thriller', 'Horror'] for genre in content_data['genres_list']):
                        adjustment *= 1.1
                elif context.get('time_preference') == 'weekend':
                    if any(genre in ['Action', 'Comedy', 'Adventure'] for genre in content_data['genres_list']):
                        adjustment *= 1.1
                
                # Device context
                if context.get('device') == 'mobile':
                    if content_data['runtime'] and content_data['runtime'] < 120:  # Shorter content for mobile
                        adjustment *= 1.1
                
                adjusted_scores[content_id] = score * adjustment
        
        return adjusted_scores
    
    def _apply_ultra_personalization_filters(self, recommendations, user_data, user_profile, limit, context):
        """Apply ultra-advanced personalization filters"""
        # User preferences from request
        preferred_genres = user_data.get('preferred_genres', [])
        preferred_languages = user_data.get('preferred_languages', [])
        
        # Enhanced scoring
        for rec in recommendations:
            content_id = rec['content_id']
            content_data = model_store.content_metadata.get(content_id, {})
            
            # Multi-layered preference matching
            preference_boost = 1.0
            
            # Genre preference (from both stated and inferred)
            content_genres = content_data.get('genres_list', [])
            stated_genre_matches = len(set(preferred_genres) & set(content_genres))
            inferred_genres = user_profile.get('preferences', {}).get('genres', {})
            
            if stated_genre_matches > 0:
                preference_boost *= (1.0 + 0.2 * stated_genre_matches)
            
            for genre in content_genres:
                if genre in inferred_genres:
                    preference_boost *= (1.0 + 0.15 * inferred_genres[genre])
            
            # Language preference
            content_languages = content_data.get('languages_list', [])
            lang_matches = len(set(preferred_languages) & set(content_languages))
            if lang_matches > 0:
                preference_boost *= (1.0 + 0.3 * lang_matches)
            
            # Personality-based adjustments
            traits = user_profile.get('personality_traits', {})
            if traits.get('openness', 0.5) > 0.7:
                # Boost diverse and unique content
                unique_genres = len(content_genres)
                if unique_genres > 2:
                    preference_boost *= 1.1
            
            if traits.get('conscientiousness', 0.5) > 0.7:
                # Boost high-quality, well-reviewed content
                if content_data.get('rating', 0) >= 8.0:
                    preference_boost *= 1.1
            
            # Context-based adjustments
            if context:
                if context.get('mood') == 'relaxed' and 'Comedy' in content_genres:
                    preference_boost *= 1.2
                elif context.get('mood') == 'intense' and any(g in content_genres for g in ['Thriller', 'Action', 'Horror']):
                    preference_boost *= 1.2
            
            rec['score'] *= preference_boost
        
        # Re-sort and apply diversity
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply diversity constraints
        final_recs = []
        used_genres = set()
        used_languages = set()
        
        diversity_target = user_profile.get('behavioral_patterns', {}).get('exploration_ratio', 0.5)
        max_same_genre = max(2, int(limit * (1 - diversity_target)))
        
        for rec in recommendations:
            if len(final_recs) >= limit:
                break
            
            content_data = model_store.content_metadata.get(rec['content_id'], {})
            content_genres = set(content_data.get('genres_list', []))
            content_languages = set(content_data.get('languages_list', []))
            
            # Check diversity constraints
            genre_overlap = len(content_genres & used_genres)
            lang_overlap = len(content_languages & used_languages)
            
            if (genre_overlap < max_same_genre or len(final_recs) < limit // 2):
                final_recs.append(rec)
                used_genres.update(content_genres)
                used_languages.update(content_languages)
        
        return final_recs
    
    def _generate_trending_reason(self, content_data, region, language, context):
        """Generate contextual trending reason"""
        base_reason = "Currently trending"
        
        if region:
            base_reason += f" in {region}"
        if language:
            base_reason += f" among {language} content"
        
        # Add context
        if context:
            if context.get('time_preference') == 'evening':
                base_reason += " for evening viewing"
            elif context.get('device') == 'mobile':
                base_reason += " for mobile viewing"
        
        return base_reason
    
    def _calculate_trending_velocity(self, content_id, interactions_df):
        """Calculate trending velocity score"""
        current_time = datetime.now()
        recent_interactions = interactions_df[
            (interactions_df['content_id'] == content_id) &
            (pd.to_datetime(interactions_df['timestamp']) >= current_time - timedelta(hours=24))
        ]
        
        if recent_interactions.empty:
            return 0.0
        
        # Velocity = interactions per hour
        velocity = len(recent_interactions) / 24.0
        return min(velocity / 10.0, 1.0)  # Normalize to 0-1
    
    def _calculate_cultural_relevance(self, content_data, region, language):
        """Calculate cultural relevance score"""
        relevance = 0.5  # Base relevance
        
        if language:
            content_languages = content_data.get('languages_list', [])
            if any(language.lower() in lang.lower() for lang in content_languages):
                relevance += 0.3
        
        if region:
            # Regional content type preferences
            regional_preferences = {
                'India': ['movie'],
                'Japan': ['anime'],
                'South Korea': ['tv'],
                'USA': ['movie', 'tv']
            }
            
            preferred_types = regional_preferences.get(region, [])
            if content_data.get('content_type') in preferred_types:
                relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _classify_user_personality(self, user_profile):
        """Classify user personality type"""
        traits = user_profile.get('personality_traits', {})
        
        if not traits:
            return 'unknown'
        
        # Simple personality classification
        openness = traits.get('openness', 0.5)
        conscientiousness = traits.get('conscientiousness', 0.5)
        extroversion = traits.get('extroversion', 0.5)
        
        if openness > 0.7 and extroversion > 0.6:
            return 'explorer'
        elif conscientiousness > 0.7 and openness < 0.4:
            return 'traditionalist'
        elif extroversion > 0.7:
            return 'social'
        elif openness > 0.7:
            return 'curious'
        else:
            return 'balanced'
    
    def _assess_recommendation_readiness(self, user_profile):
        """Assess how ready the user profile is for personalization"""
        behavior = user_profile.get('behavioral_patterns', {})
        preferences = user_profile.get('preferences', {})
        
        # Factors contributing to readiness
        factors = []
        
        # Interaction diversity
        interaction_diversity = behavior.get('interaction_diversity', {})
        if interaction_diversity.get('engagement_depth', 0) > 2:
            factors.append('high_engagement')
        
        # Preference clarity
        if len(preferences.get('genres', {})) >= 3:
            factors.append('clear_preferences')
        
        # Activity consistency
        if behavior.get('temporal', {}).get('session_patterns', {}).get('avg_session_length', 0) > 3:
            factors.append('consistent_usage')
        
        readiness_score = len(factors) / 3.0
        
        if readiness_score >= 0.8:
            return 'high'
        elif readiness_score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_personalization_confidence(self, user_profile, recommendation_count):
        """Calculate confidence in personalization quality"""
        base_confidence = 0.5
        
        # Data richness
        preferences = user_profile.get('preferences', {})
        if len(preferences.get('genres', {})) > 0:
            base_confidence += 0.2
        
        # Behavioral insights
        behavior = user_profile.get('behavioral_patterns', {})
        if behavior.get('exploration_ratio', 0) > 0:
            base_confidence += 0.1
        
        # Personality insights
        traits = user_profile.get('personality_traits', {})
        if len(traits) > 0:
            base_confidence += 0.1
        
        # Recommendation quantity
        if recommendation_count >= 20:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    # Additional helper methods for similarity, genre, anime, etc.
    def _apply_similarity_context_adjustments(self, recommendations, base_content_id, context, content_df):
        """Apply context adjustments to similarity recommendations"""
        # Get base content info for context
        base_content = content_df[content_df['id'] == base_content_id]
        if base_content.empty:
            return recommendations
        
        base_data = base_content.iloc[0]
        
        for content_id, score in recommendations.items():
            content_row = content_df[content_df['id'] == content_id]
            if not content_row.empty:
                content_data = content_row.iloc[0]
                adjustment = 1.0
                
                # If seeking variety, boost different content types
                if context and context.get('variety_seeking'):
                    if content_data['content_type'] != base_data['content_type']:
                        adjustment *= 1.2
                
                # If seeking similar quality level
                if context and context.get('quality_matching'):
                    rating_diff = abs(content_data['rating'] - base_data['rating'])
                    if rating_diff <= 1.0:  # Similar quality
                        adjustment *= 1.1
                
                recommendations[content_id] = score * adjustment
        
        return recommendations
    
    def _generate_similarity_reason(self, algorithms_used, consensus_strength):
        """Generate explanation for similarity recommendations"""
        if consensus_strength >= 0.8:
            return "Strong consensus across multiple similarity algorithms - highly recommended"
        elif consensus_strength >= 0.6:
            return "Multiple algorithms agree this content is similar to your selection"
        elif 'semantic' in algorithms_used:
            return "Similar themes and storytelling elements"
        elif 'collaborative' in algorithms_used:
            return "Other users with similar taste also enjoyed this"
        else:
            return "Similar content characteristics and features"
    
    def _calculate_regional_relevance(self, content, region):
        """Calculate regional relevance boost"""
        if not region:
            return 0.0
        
        # Language-region mapping
        region_languages = {
            'India': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam'],
            'Japan': ['japanese'],
            'South Korea': ['korean'],
            'USA': ['english'],
            'UK': ['english'],
            'Canada': ['english', 'french']
        }
        
        preferred_languages = region_languages.get(region, [])
        content_languages = content.get('languages_list', [])
        
        if any(lang in content_languages for lang in preferred_languages):
            return 0.3
        
        return 0.0
    
    def _apply_genre_context_adjustments(self, content, context, genre):
        """Apply context adjustments for genre recommendations"""
        adjustment = 1.0
        
        if not context:
            return adjustment
        
        content_genres = content.get('genres_list', [])
        
        # Mood-based adjustments
        mood = context.get('mood')
        if mood == 'energetic' and genre.lower() in ['action', 'adventure']:
            adjustment *= 1.2
        elif mood == 'relaxed' and genre.lower() in ['comedy', 'romance']:
            adjustment *= 1.2
        elif mood == 'thoughtful' and genre.lower() in ['drama', 'documentary']:
            adjustment *= 1.2
        
        # Time-based adjustments
        time_pref = context.get('time_preference')
        if time_pref == 'evening' and genre.lower() in ['thriller', 'horror']:
            adjustment *= 1.1
        elif time_pref == 'weekend' and genre.lower() in ['action', 'comedy']:
            adjustment *= 1.1
        
        return adjustment
    
    def _apply_anime_context_adjustments(self, content, context, genre):
        """Apply anime-specific context adjustments"""
        adjustment = 1.0
        
        if not context:
            return adjustment
        
        # Anime viewing context
        if context.get('binge_intention') and content.get('content_type') == 'anime':
            adjustment *= 1.2  # Anime is great for binging
        
        # Season preference
        if context.get('season_preference') == 'current':
            if content.get('content_age_years', 10) <= 1:
                adjustment *= 1.3
        
        return adjustment
    
    def _classify_anime_tier(self, content):
        """Classify anime into quality tiers"""
        rating = content.get('rating', 0)
        vote_count = content.get('vote_count', 0)
        
        if rating >= 9.0 and vote_count > 5000:
            return 'legendary'
        elif rating >= 8.5 and vote_count > 2000:
            return 'masterpiece'
        elif rating >= 8.0 and vote_count > 1000:
            return 'excellent'
        elif rating >= 7.5:
            return 'good'
        else:
            return 'average'
    
    def _calculate_anime_popularity_rank(self, content, anime_df):
        """Calculate anime popularity rank within dataset"""
        if anime_df.empty:
            return 0.5
        
        popularity = content.get('popularity', 0)
        rank = (anime_df['popularity'] < popularity).sum() / len(anime_df)
        return rank
    
    def _apply_new_release_context_adjustments(self, content, context, language):
        """Apply context adjustments for new releases"""
        adjustment = 1.0
        
        if not context:
            return adjustment
        
        # Early adopter preference
        if context.get('early_adopter') and content.get('content_age_days', 365) <= 30:
            adjustment *= 1.3
        
        # Quality risk tolerance
        if context.get('quality_risk_tolerance') == 'low':
            if content.get('vote_count', 0) < 50:  # Too few reviews
                adjustment *= 0.7
        
        return adjustment
    
    def _classify_recency_tier(self, days_old, recency_tiers):
        """Classify content into recency tiers"""
        for tier_name, max_days in recency_tiers.items():
            if days_old <= max_days:
                return tier_name
        return 'old'
    
    def _calculate_release_momentum(self, content, interactions_df):
        """Calculate momentum score for new releases"""
        content_id = content.get('id')
        if not content_id:
            return 0.0
        
        # Get interactions in first 30 days after release
        release_date = content.get('release_date')
        if not release_date:
            return 0.0
        
        try:
            release_datetime = pd.to_datetime(release_date)
            momentum_window = release_datetime + timedelta(days=30)
            
            momentum_interactions = interactions_df[
                (interactions_df['content_id'] == content_id) &
                (pd.to_datetime(interactions_df['timestamp']) >= release_datetime) &
                (pd.to_datetime(interactions_df['timestamp']) <= momentum_window)
            ]
            
            if momentum_interactions.empty:
                return 0.0
            
            # Calculate momentum based on interaction velocity and engagement
            momentum = len(momentum_interactions) / 30.0  # Interactions per day
            engagement_quality = momentum_interactions['final_weight'].mean()
            
            return min((momentum * engagement_quality) / 10.0, 1.0)
            
        except:
            return 0.0
    
    def _apply_critics_choice_context_adjustments(self, content, context):
        """Apply context adjustments for critics choice"""
        adjustment = 1.0
        
        if not context:
            return adjustment
        
        # Preference for award winners
        if context.get('award_preference') and content.get('is_critics_choice'):
            adjustment *= 1.2
        
        # Quality over popularity preference
        if context.get('quality_over_popularity'):
            rating = content.get('rating', 0)
            popularity = content.get('popularity', 0)
            if rating > 8.0 and popularity < 50:  # High quality, not mainstream
                adjustment *= 1.3
        
        return adjustment
    
    def _classify_quality_tier(self, content):
        """Classify content into quality tiers"""
        rating = content.get('rating', 0)
        vote_count = content.get('vote_count', 0)
        quality_score = content.get('quality_score', 0)
        
        if rating >= 9.0 and vote_count > 1000 and quality_score > 0.9:
            return 'masterpiece'
        elif rating >= 8.5 and vote_count > 500 and quality_score > 0.8:
            return 'excellent'
        elif rating >= 8.0 and vote_count > 200 and quality_score > 0.7:
            return 'very_good'
        elif rating >= 7.5 and quality_score > 0.6:
            return 'good'
        else:
            return 'average'
    
    def _calculate_critical_consensus(self, content):
        """Calculate critical consensus strength"""
        rating = content.get('rating', 0)
        vote_count = content.get('vote_count', 0)
        is_critics_choice = content.get('is_critics_choice', False)
        
        consensus = 0.0
        
        # High rating consensus
        if rating >= 8.5:
            consensus += 0.4
        elif rating >= 8.0:
            consensus += 0.3
        elif rating >= 7.5:
            consensus += 0.2
        
        # Vote count indicates consensus breadth
        if vote_count > 2000:
            consensus += 0.3
        elif vote_count > 1000:
            consensus += 0.2
        elif vote_count > 500:
            consensus += 0.1
        
        # Official critics choice
        if is_critics_choice:
            consensus += 0.3
        
        return min(consensus, 1.0)

# Initialize ultra-advanced ML service
ml_service = UltraAdvancedMLService()

# API Routes with ultra-advanced features
@app.route('/api/health', methods=['GET'])
def health_check():
    """Ultra-comprehensive health check"""
    health_status = {
        'status': 'operational' if models_initialized else 'initializing',
        'timestamp': datetime.now().isoformat(),
        'models_initialized': models_initialized,
        'last_update': model_store.last_update.isoformat() if model_store.last_update else None,
        'update_count': model_store.update_count,
        'cache_statistics': model_store.get_cache_stats(),
        'data_status': {
            'content_count': len(model_store.content_df) if model_store.content_df is not None else 0,
            'interactions_count': len(model_store.interactions_df) if model_store.interactions_df is not None else 0,
            'users_count': len(model_store.users_df) if model_store.users_df is not None else 0
        },
        'engine_status': {
            'collaborative_models': len(ml_service.hybrid_engine.collaborative_engine.models),
            'content_similarity_ready': ml_service.hybrid_engine.content_engine.similarity_matrices != {},
            'semantic_models': len(ml_service.hybrid_engine.semantic_engine.models),
            'faiss_indices': len(ml_service.hybrid_engine.semantic_engine.faiss_indices)
        },
        'performance_metrics': {
            'avg_update_time': np.mean(ml_service.performance_monitor['update_time']) if ml_service.performance_monitor['update_time'] else 0,
            'cache_hit_rate': len(ml_service.cache) / max(len(ml_service.cache) + 1, 1),
            'real_time_signals': len(ml_service.real_time_signals)
        }
    }
    
    return jsonify(health_status), 200

@app.route('/api/update-models', methods=['POST'])
def update_models():
    """Force ultra-advanced model update"""
    try:
        success = ml_service.update_models()
        
        return jsonify({
            'status': 'success' if success else 'error',
            'message': 'Ultra-advanced models updated successfully' if success else 'Model update failed',
            'timestamp': datetime.now().isoformat(),
            'models_initialized': models_initialized,
            'performance_impact': {
                'update_time': ml_service.performance_monitor['update_time'][-1] if ml_service.performance_monitor['update_time'] else None,
                'cache_cleared': True
            }
        }), 200 if success else 500
        
    except Exception as e:
        logger.error(f"Ultra model update API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_ultra_comprehensive_stats():
    """Get ultra-comprehensive ML service statistics"""
    try:
        stats = {
            'service_status': 'operational' if models_initialized else 'initializing',
            'timestamp': datetime.now().isoformat(),
            'version': 'ultra_advanced_v2.0',
            'models_initialized': models_initialized,
            'last_update': model_store.last_update.isoformat() if model_store.last_update else None,
            'update_count': model_store.update_count,
            'cache_statistics': {
                'memory_cache_size': len(ml_service.cache),
                'cache_ttl_seconds': ml_service.cache.ttl,
                'real_time_signals': len(ml_service.real_time_signals),
                'performance_history': len(ml_service.performance_monitor)
            },
            'data_statistics': {
                'total_content': len(model_store.content_df) if model_store.content_df is not None else 0,
                'total_interactions': len(model_store.interactions_df) if model_store.interactions_df is not None else 0,
                'unique_users': len(model_store.users_df) if model_store.users_df is not None else 0,
                'content_types': {},
                'interaction_types': {},
                'language_distribution': {},
                'genre_distribution': {}
            },
            'model_performance': {
                'collaborative_filtering': {
                    'models_active': len(ml_service.hybrid_engine.collaborative_engine.models),
                    'matrix_size': ml_service.hybrid_engine.collaborative_engine.user_item_matrix.shape if ml_service.hybrid_engine.collaborative_engine.user_item_matrix is not None else [0, 0],
                    'trained': ml_service.hybrid_engine.collaborative_engine.trained
                },
                'content_based': {
                    'similarity_matrices': len(ml_service.hybrid_engine.content_engine.similarity_matrices),
                    'faiss_ready': ml_service.hybrid_engine.content_engine.faiss_index is not None,
                    'feature_dimensions': ml_service.hybrid_engine.content_engine.combined_matrix.shape[1] if ml_service.hybrid_engine.content_engine.combined_matrix is not None else 0
                },
                'semantic_similarity': {
                    'models_loaded': len(ml_service.hybrid_engine.semantic_engine.models),
                    'embeddings_ready': len(ml_service.hybrid_engine.semantic_engine.embeddings),
                    'faiss_indices': len(ml_service.hybrid_engine.semantic_engine.faiss_indices)
                }
            },
            'algorithm_weights': ml_service.hybrid_engine.base_weights,
            'recommendation_coverage': {
                'trending_ready': model_store.trending_weights is not None,
                'personalization_ready': model_store.users_df is not None and not model_store.users_df.empty,
                'similarity_ready': len(ml_service.hybrid_engine.content_engine.similarity_matrices) > 0,
                'semantic_ready': len(ml_service.hybrid_engine.semantic_engine.embeddings) > 0
            }
        }
        
        # Add detailed data statistics
        if model_store.content_df is not None and not model_store.content_df.empty:
            stats['data_statistics']['content_types'] = model_store.content_df['content_type'].value_counts().to_dict()
            
            # Language distribution
            all_languages = []
            for lang_list in model_store.content_df['languages_list']:
                all_languages.extend(lang_list)
            stats['data_statistics']['language_distribution'] = dict(Counter(all_languages).most_common(10))
            
            # Genre distribution
            all_genres = []
            for genre_list in model_store.content_df['genres_list']:
                all_genres.extend(genre_list)
            stats['data_statistics']['genre_distribution'] = dict(Counter(all_genres).most_common(15))
        
        if model_store.interactions_df is not None and not model_store.interactions_df.empty:
            stats['data_statistics']['interaction_types'] = model_store.interactions_df['interaction_type'].value_counts().to_dict()
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Ultra stats API error: {e}")
        return jsonify({'error': str(e)}), 500

# Ultra-Advanced Recommendation API Endpoints
@app.route('/api/trending', methods=['GET'])
def get_trending():
    """Get ultra-advanced trending recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        content_type = request.args.get('content_type', 'all')
        region = request.args.get('region')
        language = request.args.get('language')
        
        # Extract context from request headers
        context = {
            'time_preference': request.headers.get('X-Time-Preference'),
            'device': request.headers.get('X-Device-Type'),
            'user_agent': request.headers.get('User-Agent')
        }
        
        result = ml_service.get_ultra_trending_recommendations(limit, content_type, region, language, context)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Ultra trending API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_personalized_recommendations():
    """Get ultra-personalized recommendations"""
    try:
        user_data = request.get_json() or {}
        limit = int(request.args.get('limit', 20))
        
        # Extract context
        context = {
            'time_preference': request.headers.get('X-Time-Preference'),
            'device': request.headers.get('X-Device-Type'),
            'mood': user_data.get('mood'),
            'session_type': user_data.get('session_type')
        }
        
        result = ml_service.get_ultra_personalized_recommendations(user_data, limit, context)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Ultra personalized recommendations API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/similar/<int:content_id>', methods=['GET'])
def get_similar_recommendations(content_id):
    """Get ultra-advanced similar content recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        
        context = {
            'variety_seeking': request.args.get('variety_seeking') == 'true',
            'quality_matching': request.args.get('quality_matching') == 'true'
        }
        
        result = ml_service.get_ultra_similar_recommendations(content_id, limit, context)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Ultra similar recommendations API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/genre/<genre>', methods=['GET'])
def get_genre_recommendations(genre):
    """Get ultra-advanced genre-based recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        content_type = request.args.get('content_type', 'movie')
        region = request.args.get('region')
        
        context = {
            'mood': request.args.get('mood'),
            'time_preference': request.args.get('time_preference')
        }
        
        result = ml_service.get_ultra_genre_recommendations(genre, limit, content_type, region, context)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Ultra genre recommendations API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/regional/<language>', methods=['GET'])
def get_regional_recommendations(language):
    """Get ultra-advanced regional/language recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        content_type = request.args.get('content_type', 'movie')
        
        context = {
            'cultural_preference': request.args.get('cultural_preference'),
            'content_discovery': request.args.get('content_discovery') == 'true'
        }
        
        result = ml_service.get_ultra_regional_recommendations(language, limit, content_type, context)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Ultra regional recommendations API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/anime', methods=['GET'])
def get_anime_recommendations():
    """Get ultra-specialized anime recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        genre = request.args.get('genre')
        
        context = {
            'binge_intention': request.args.get('binge_intention') == 'true',
            'season_preference': request.args.get('season_preference'),
            'dub_preference': request.args.get('dub_preference')
        }
        
        result = ml_service.get_ultra_anime_recommendations(limit, genre, context)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Ultra anime recommendations API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/new-releases', methods=['GET'])
def get_new_releases():
    """Get ultra-advanced new releases recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        content_type = request.args.get('content_type', 'movie')
        language = request.args.get('language')
        
        context = {
            'early_adopter': request.args.get('early_adopter') == 'true',
            'quality_risk_tolerance': request.args.get('quality_risk_tolerance', 'medium')
        }
        
        result = ml_service.get_ultra_new_releases(limit, content_type, language, context)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Ultra new releases API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

@app.route('/api/critics-choice', methods=['GET'])
def get_critics_choice():
    """Get ultra-advanced critics choice recommendations"""
    try:
        limit = int(request.args.get('limit', 20))
        content_type = request.args.get('content_type', 'movie')
        
        context = {
            'award_preference': request.args.get('award_preference') == 'true',
            'quality_over_popularity': request.args.get('quality_over_popularity') == 'true'
        }
        
        result = ml_service.get_ultra_critics_choice(limit, content_type, context)
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Ultra critics choice API error: {e}")
        return jsonify({'recommendations': [], 'strategy': 'error', 'cached': False}), 500

# Ultra User Profile API
@app.route('/api/user-profile/<int:user_id>', methods=['GET'])
def get_ultra_user_profile(user_id):
    """Get ultra-comprehensive user profile"""
    try:
        if not model_store.is_initialized():
            return jsonify({'error': 'Models not initialized'}), 503
        
        user_profile = UltraAdvancedUserProfiler.build_ultra_user_profile(
            user_id, model_store.interactions_df, model_store.content_df, model_store.users_df
        )
        
        # Add profile insights
        profile_insights = {
            'personality_classification': ml_service._classify_user_personality(user_profile),
            'recommendation_readiness': ml_service._assess_recommendation_readiness(user_profile),
            'personalization_confidence': ml_service._calculate_personalization_confidence(user_profile, 50)
        }
        
        user_profile['profile_insights'] = profile_insights
        
        return jsonify(user_profile), 200
        
    except Exception as e:
        logger.error(f"Ultra user profile API error: {e}")
        return jsonify({'error': str(e)}), 500

# Real-time Signal Integration
@app.route('/api/real-time-signal', methods=['POST'])
def update_real_time_signal():
    """Update real-time trending signals"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        signal_strength = data.get('signal_strength', 1.0)
        signal_type = data.get('signal_type', 'interaction')
        
        if content_id:
            # Update real-time signals with decay
            current_signal = ml_service.real_time_signals.get(content_id, 0.0)
            ml_service.real_time_signals[content_id] = current_signal + signal_strength
            
            # Apply decay to prevent infinite accumulation
            if len(ml_service.real_time_signals) > 1000:
                for cid in list(ml_service.real_time_signals.keys()):
                    ml_service.real_time_signals[cid] *= 0.95
                    if ml_service.real_time_signals[cid] < 0.1:
                        del ml_service.real_time_signals[cid]
        
        return jsonify({'status': 'success', 'signals_count': len(ml_service.real_time_signals)}), 200
        
    except Exception as e:
        logger.error(f"Real-time signal error: {e}")
        return jsonify({'error': str(e)}), 500

# Cache Management
@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear all caches"""
    try:
        ml_service.cache.clear()
        memory_cache.clear()
        if disk_cache:
            disk_cache.clear()
        
        return jsonify({
            'status': 'success',
            'message': 'All caches cleared successfully',
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get comprehensive cache statistics"""
    try:
        return jsonify({
            'memory_cache': {
                'size': len(ml_service.cache),
                'maxsize': ml_service.cache.maxsize,
                'ttl': ml_service.cache.ttl,
                'hits': getattr(ml_service.cache, 'hits', 0),
                'misses': getattr(ml_service.cache, 'misses', 0)
            },
            'global_memory_cache': {
                'size': len(memory_cache),
                'maxsize': memory_cache.maxsize,
                'ttl': memory_cache.ttl
            },
            'disk_cache': {
                'size': len(disk_cache) if disk_cache else 0,
                'available': disk_cache is not None
            },
            'real_time_signals': len(ml_service.real_time_signals)
        }), 200
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return jsonify({'error': str(e)}), 500

# Performance Monitoring
@app.route('/api/performance', methods=['GET'])
def get_performance_metrics():
    """Get performance metrics"""
    try:
        return jsonify({
            'update_times': ml_service.performance_monitor['update_time'][-10:],  # Last 10
            'data_sizes': ml_service.performance_monitor['data_size'][-10:],
            'cache_efficiency': len(ml_service.cache) / max(len(ml_service.cache) + 1, 1),
            'models_status': {
                'collaborative': ml_service.hybrid_engine.collaborative_engine.trained,
                'content_based': len(ml_service.hybrid_engine.content_engine.similarity_matrices) > 0,
                'semantic': len(ml_service.hybrid_engine.semantic_engine.embeddings) > 0
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize models on startup
def initialize_ultra_models_async():
    """Initialize ultra-advanced models asynchronously"""
    try:
        logger.info("Starting ultra-advanced ML service initialization...")
        start_time = time.time()
        success = ml_service.update_models()
        initialization_time = time.time() - start_time
        
        if success:
            logger.info(f"Ultra-advanced ML service initialization completed successfully in {initialization_time:.2f}s")
        else:
            logger.warning(f"Ultra-advanced ML service initialization completed with warnings in {initialization_time:.2f}s")
            
    except Exception as e:
        logger.error(f"Ultra-advanced ML service initialization failed: {e}")

# Run initialization in background
if __name__ == '__main__':
    executor.submit(initialize_ultra_models_async)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Ultra-Advanced ML Recommendation Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)