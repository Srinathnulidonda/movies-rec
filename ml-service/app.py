from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import pickle
import joblib
import os
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import requests
import asyncio
import concurrent.futures
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
import threading
import time
import nltk
import re
import math
import heapq
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("CatBoost not available, will skip CatBoost models")

try:
    from surprise import Dataset, Reader, SVD, NMF as SurpriseNMF, SlopeOne, CoClustering
    from surprise.model_selection import train_test_split as surprise_train_test_split
    from surprise import accuracy
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Scikit-surprise not available, will skip collaborative filtering models")

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'ml-service-secret-key')

# Database configuration (connect to same database as main backend)
if os.environ.get('DATABASE_URL'):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_recommendations.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model storage paths
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize NLTK if available
if NLTK_AVAILABLE:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except:
        logger.warning("Failed to download NLTK data")
        NLTK_AVAILABLE = False

# Database Models (same as main backend)
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
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    content_type = db.Column(db.String(20), nullable=False)
    genres = db.Column(db.Text)
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
    ott_platforms = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AnonymousInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)
    ip_address = db.Column(db.String(45))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Advanced Text Processing
class AdvancedTextProcessor:
    def __init__(self):
        if NLTK_AVAILABLE:
            try:
                self.stemmer = PorterStemmer()
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stemmer = None
                self.lemmatizer = None
                self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        else:
            self.stemmer = None
            self.lemmatizer = None
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        if NLTK_AVAILABLE and self.stemmer:
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
        else:
            tokens = text.split()
        
        # Remove stop words and apply lemmatization
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                if self.lemmatizer:
                    try:
                        lemmatized = self.lemmatizer.lemmatize(token)
                        processed_tokens.append(lemmatized)
                    except:
                        processed_tokens.append(token)
                else:
                    processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def extract_keywords(self, text, max_keywords=10):
        """Extract keywords from text using TF-IDF"""
        if not text:
            return []
        
        try:
            vectorizer = TfidfVectorizer(max_features=max_keywords, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, score in keyword_scores if score > 0]
        except:
            return text.split()[:max_keywords]

# Custom Matrix Factorization
class CustomMatrixFactorization:
    def __init__(self, n_factors=50, learning_rate=0.01, regularization=0.02, epochs=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        self.user_map = {}
        self.item_map = {}
    
    def fit(self, user_item_matrix):
        """Train the matrix factorization model"""
        # Convert to numpy array if needed
        if hasattr(user_item_matrix, 'toarray'):
            ratings_matrix = user_item_matrix.toarray()
        else:
            ratings_matrix = np.array(user_item_matrix)
        
        n_users, n_items = ratings_matrix.shape
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = np.mean(ratings_matrix[ratings_matrix > 0])
        
        # Get non-zero indices
        user_indices, item_indices = np.nonzero(ratings_matrix)
        
        # Training loop with adaptive learning rate
        initial_lr = self.learning_rate
        
        for epoch in range(self.epochs):
            # Shuffle training data
            indices = np.random.permutation(len(user_indices))
            
            epoch_error = 0
            for idx in indices:
                user_id = user_indices[idx]
                item_id = item_indices[idx]
                rating = ratings_matrix[user_id, item_id]
                
                # Compute prediction
                prediction = self.global_bias + self.user_bias[user_id] + self.item_bias[item_id]
                prediction += np.dot(self.user_factors[user_id], self.item_factors[item_id])
                
                # Compute error
                error = rating - prediction
                epoch_error += error ** 2
                
                # Update biases
                user_bias_old = self.user_bias[user_id]
                self.user_bias[user_id] += self.learning_rate * (error - self.regularization * self.user_bias[user_id])
                self.item_bias[item_id] += self.learning_rate * (error - self.regularization * self.item_bias[item_id])
                
                # Update factors
                user_factors_old = self.user_factors[user_id].copy()
                self.user_factors[user_id] += self.learning_rate * (error * self.item_factors[item_id] - self.regularization * self.user_factors[user_id])
                self.item_factors[item_id] += self.learning_rate * (error * user_factors_old - self.regularization * self.item_factors[item_id])
            
            # Adaptive learning rate
            if epoch > 10:
                self.learning_rate = initial_lr * (0.95 ** (epoch - 10))
            
            if epoch % 20 == 0:
                rmse = np.sqrt(epoch_error / len(user_indices))
                logger.info(f"Epoch {epoch}, RMSE: {rmse:.4f}, LR: {self.learning_rate:.6f}")
    
    def predict(self, user_id, item_id):
        """Predict rating for a user-item pair"""
        if user_id >= len(self.user_factors) or item_id >= len(self.item_factors):
            return self.global_bias
        
        prediction = self.global_bias + self.user_bias[user_id] + self.item_bias[item_id]
        prediction += np.dot(self.user_factors[user_id], self.item_factors[item_id])
        
        return max(0, min(5, prediction))  # Clamp between 0 and 5

# Advanced Recommendation Engine
class AdvancedRecommendationEngine:
    def __init__(self):
        self.content_features = None
        self.user_features = None
        self.tfidf_vectorizer = None
        self.content_similarity_matrix = None
        self.custom_mf_model = None
        self.surprise_models = {}
        self.ensemble_models = {}
        self.text_processor = AdvancedTextProcessor()
        self.scaler = StandardScaler()
        self.models_loaded = False
        self.last_training = None
        
        # Model weights for ensemble
        self.model_weights = {
            'content_based': 0.25,
            'collaborative_mf': 0.30,
            'collaborative_surprise': 0.20 if SURPRISE_AVAILABLE else 0.0,
            'lightgbm': 0.15,
            'xgboost': 0.10,
            'catboost': 0.05 if CATBOOST_AVAILABLE else 0.0
        }
        
        # Normalize weights if some models are not available
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
        
        # Initialize and load models
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize ML models and load if available"""
        try:
            self.load_models()
            if not self.models_loaded:
                logger.info("No pre-trained models found. Will train on first request.")
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
    
    def prepare_content_features(self):
        """Extract and prepare comprehensive content features"""
        try:
            contents = Content.query.all()
            if not contents:
                return None
            
            content_data = []
            for content in contents:
                try:
                    genres = json.loads(content.genres or '[]')
                    languages = json.loads(content.languages or '[]')
                except json.JSONDecodeError:
                    genres = []
                    languages = []
                
                # Process text features
                text_content = f"{content.title or ''} {content.overview or ''}"
                cleaned_text = self.text_processor.clean_text(text_content)
                keywords = self.text_processor.extract_keywords(content.overview or '', max_keywords=5)
                
                content_data.append({
                    'content_id': content.id,
                    'title': content.title or '',
                    'overview': content.overview or '',
                    'cleaned_text': cleaned_text,
                    'keywords': ' '.join(keywords),
                    'genres': genres,
                    'languages': languages,
                    'content_type': content.content_type,
                    'rating': content.rating or 0,
                    'popularity': content.popularity or 0,
                    'runtime': content.runtime or 0,
                    'vote_count': content.vote_count or 0,
                    'release_year': content.release_date.year if content.release_date else 2000,
                    'decade': (content.release_date.year // 10) * 10 if content.release_date else 2000
                })
            
            df = pd.DataFrame(content_data)
            
            # Create comprehensive text features
            df['combined_text'] = df['title'] + ' ' + df['overview'] + ' ' + df['keywords'] + ' ' + df['genres'].apply(lambda x: ' '.join(x))
            
            # Enhanced TF-IDF with n-grams
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,  # Reduced for deployment
                stop_words='english',
                ngram_range=(1, 2),  # Reduced n-gram range
                min_df=2,
                max_df=0.8,
                sublinear_tf=True
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['combined_text'])
            
            # Calculate enhanced content similarity
            self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Prepare numerical features with feature engineering
            numerical_features = df[['rating', 'popularity', 'runtime', 'vote_count', 'release_year', 'decade']].fillna(0)
            
            # Add derived features
            numerical_features['rating_popularity_ratio'] = numerical_features['rating'] / (numerical_features['popularity'] + 1)
            numerical_features['votes_per_year'] = numerical_features['vote_count'] / (2024 - numerical_features['release_year'] + 1)
            numerical_features['is_recent'] = (numerical_features['release_year'] >= 2020).astype(int)
            numerical_features['is_classic'] = (numerical_features['release_year'] <= 1990).astype(int)
            numerical_features['runtime_category'] = pd.cut(numerical_features['runtime'], bins=[0, 90, 120, 180, float('inf')], labels=[0, 1, 2, 3]).astype(int)
            
            # Encode categorical features with advanced techniques
            genre_features = self._encode_multi_label_features(df['genres'].tolist(), 'genres')
            language_features = self._encode_multi_label_features(df['languages'].tolist(), 'languages')
            
            # Content type encoding
            type_features = pd.get_dummies(df['content_type'])
            
            # Combine all features
            self.content_features = np.hstack([
                tfidf_matrix.toarray(),
                self.scaler.fit_transform(numerical_features),
                genre_features,
                language_features,
                type_features.values
            ])
            
            # Store mappings
            self.content_id_to_index = {content_id: idx for idx, content_id in enumerate(df['content_id'])}
            self.index_to_content_id = {idx: content_id for content_id, idx in self.content_id_to_index.items()}
            
            logger.info(f"Prepared enhanced features for {len(contents)} content items with {self.content_features.shape[1]} features")
            return df
            
        except Exception as e:
            logger.error(f"Content feature preparation error: {e}")
            return None
    
    def _encode_multi_label_features(self, multi_label_list, feature_type=''):
        """Enhanced multi-label encoding with TF-IDF weighting"""
        all_labels = set()
        for labels in multi_label_list:
            all_labels.update(labels)
        
        all_labels = sorted(list(all_labels))
        
        # Create TF-IDF weighted features for better representation
        label_counts = Counter()
        for labels in multi_label_list:
            for label in labels:
                label_counts[label] += 1
        
        total_docs = len(multi_label_list)
        encoded_features = np.zeros((len(multi_label_list), len(all_labels)))
        
        for i, labels in enumerate(multi_label_list):
            for label in labels:
                if label in all_labels:
                    label_idx = all_labels.index(label)
                    # TF-IDF weighting: more weight to rare labels
                    tf = 1 / len(labels) if len(labels) > 0 else 0  # Term frequency
                    idf = math.log(total_docs / (label_counts[label] + 1))  # Inverse document frequency
                    encoded_features[i, label_idx] = tf * idf
        
        return encoded_features
    
    def prepare_user_features(self):
        """Prepare comprehensive user interaction features"""
        try:
            users = User.query.all()
            interactions = UserInteraction.query.all()
            
            if not users or not interactions:
                return None
            
            # Create enhanced user-item matrix with temporal weights
            user_item_matrix = {}
            user_profiles = {}
            
            # Calculate temporal weights (more recent interactions have higher weight)
            current_time = datetime.utcnow()
            
            for interaction in interactions:
                user_id = interaction.user_id
                content_id = interaction.content_id
                
                if user_id not in user_item_matrix:
                    user_item_matrix[user_id] = {}
                    user_profiles[user_id] = {
                        'genre_preferences': Counter(),
                        'language_preferences': Counter(),
                        'type_preferences': Counter(),
                        'decade_preferences': Counter(),
                        'interaction_counts': Counter(),
                        'avg_rating': 0,
                        'total_interactions': 0,
                        'recency_factor': 0
                    }
                
                # Base weight from interaction type
                base_weight = {
                    'view': 1.0,
                    'like': 2.5,
                    'favorite': 4.0,
                    'watchlist': 3.0,
                    'search': 0.5,
                    'rating': 3.5
                }.get(interaction.interaction_type, 1.0)
                
                # Temporal decay (interactions lose weight over time)
                time_diff = (current_time - interaction.timestamp).days
                temporal_weight = math.exp(-time_diff / 365)  # Decay over 1 year
                
                # Rating boost
                rating_weight = (interaction.rating / 5.0) if interaction.rating else 1.0
                
                final_weight = base_weight * temporal_weight * rating_weight
                
                user_item_matrix[user_id][content_id] = final_weight
                
                # Update user profile
                content = Content.query.get(content_id)
                if content:
                    profile = user_profiles[user_id]
                    
                    # Genre preferences
                    try:
                        genres = json.loads(content.genres or '[]')
                        for genre in genres:
                            profile['genre_preferences'][genre] += final_weight
                    except:
                        pass
                    
                    # Language preferences
                    try:
                        languages = json.loads(content.languages or '[]')
                        for language in languages:
                            profile['language_preferences'][language] += final_weight
                    except:
                        pass
                    
                    # Content type preferences
                    profile['type_preferences'][content.content_type] += final_weight
                    
                    # Decade preferences
                    if content.release_date:
                        decade = (content.release_date.year // 10) * 10
                        profile['decade_preferences'][decade] += final_weight
                    
                    # Interaction statistics
                    profile['interaction_counts'][interaction.interaction_type] += 1
                    profile['total_interactions'] += 1
                    profile['recency_factor'] += temporal_weight
                    
                    # Average rating calculation
                    if interaction.rating:
                        current_avg = profile['avg_rating']
                        current_count = profile['interaction_counts']['rating']
                        if current_count > 0:
                            profile['avg_rating'] = ((current_avg * (current_count - 1)) + interaction.rating) / current_count
            
            self.user_item_matrix = user_item_matrix
            self.user_profiles = user_profiles
            
            logger.info(f"Prepared enhanced user features for {len(users)} users")
            return True
            
        except Exception as e:
            logger.error(f"User feature preparation error: {e}")
            return None
    
    def train_collaborative_filtering_models(self):
        """Train multiple collaborative filtering models"""
        try:
            if not hasattr(self, 'user_item_matrix'):
                return False
            
            success = False
            
            # Train Surprise models if available
            if SURPRISE_AVAILABLE:
                try:
                    # Prepare data for Surprise library
                    ratings_data = []
                    for user_id, items in self.user_item_matrix.items():
                        for item_id, rating in items.items():
                            ratings_data.append((user_id, item_id, rating))
                    
                    if not ratings_data:
                        return False
                    
                    # Create Surprise dataset
                    df_ratings = pd.DataFrame(ratings_data, columns=['user_id', 'item_id', 'rating'])
                    reader = Reader(rating_scale=(0, 5))
                    data = Dataset.load_from_df(df_ratings, reader)
                    
                    # Train limited Surprise models for deployment
                    algorithms = {
                        'svd': SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02),  # Reduced complexity
                        'nmf': SurpriseNMF(n_factors=30, n_epochs=20)  # Reduced complexity
                    }
                    
                    trainset, testset = surprise_train_test_split(data, test_size=0.2)
                    
                    for name, algorithm in algorithms.items():
                        try:
                            algorithm.fit(trainset)
                            predictions = algorithm.test(testset)
                            rmse = accuracy.rmse(predictions, verbose=False)
                            logger.info(f"Trained {name} model with RMSE: {rmse:.4f}")
                            self.surprise_models[name] = algorithm
                            success = True
                        except Exception as e:
                            logger.error(f"Failed to train {name}: {e}")
                except Exception as e:
                    logger.error(f"Surprise training failed: {e}")
            
            # Train custom matrix factorization
            try:
                all_users = sorted(set(self.user_item_matrix.keys()))
                all_items = sorted(set(item for items in self.user_item_matrix.values() for item in items))
                
                # Create user-item mapping
                user_to_idx = {user: idx for idx, user in enumerate(all_users)}
                item_to_idx = {item: idx for idx, item in enumerate(all_items)}
                
                # Create matrix
                matrix = np.zeros((len(all_users), len(all_items)))
                for user_id, items in self.user_item_matrix.items():
                    user_idx = user_to_idx[user_id]
                    for item_id, rating in items.items():
                        item_idx = item_to_idx[item_id]
                        matrix[user_idx, item_idx] = rating
                
                # Train custom model with reduced complexity
                self.custom_mf_model = CustomMatrixFactorization(n_factors=50, epochs=100)  # Reduced complexity
                self.custom_mf_model.fit(matrix)
                
                # Store mappings
                self.user_to_idx = user_to_idx
                self.item_to_idx = item_to_idx
                self.idx_to_user = {idx: user for user, idx in user_to_idx.items()}
                self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
                
                success = True
            except Exception as e:
                logger.error(f"Custom matrix factorization training failed: {e}")
            
            if success:
                logger.info("Successfully trained collaborative filtering models")
            
            return success
            
        except Exception as e:
            logger.error(f"Collaborative filtering training error: {e}")
            return False
    
    def train_ensemble_models(self):
        """Train ensemble models (LightGBM, XGBoost, CatBoost)"""
        try:
            if not hasattr(self, 'user_item_matrix') or self.content_features is None:
                return False
            
            # Prepare training data
            X = []
            y = []
            
            for user_id, items in self.user_item_matrix.items():
                user_features = self._get_enhanced_user_features(user_id)
                
                for content_id, rating in items.items():
                    if content_id in self.content_id_to_index:
                        content_idx = self.content_id_to_index[content_id]
                        content_features = self.content_features[content_idx]
                        
                        # Enhanced feature combination
                        combined_features = np.concatenate([
                            user_features,
                            content_features,
                            self._get_interaction_features(user_id, content_id)
                        ])
                        
                        X.append(combined_features)
                        y.append(rating)
            
            if not X:
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            success = False
            
            # Train LightGBM
            try:
                lgb_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42
                }
                
                lgb_train = lgb.Dataset(X_train, label=y_train)
                lgb_valid = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
                
                self.ensemble_models['lightgbm'] = lgb.train(
                    lgb_params,
                    lgb_train,
                    num_boost_round=200,  # Reduced for deployment
                    valid_sets=[lgb_valid],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                )
                
                lgb_pred = self.ensemble_models['lightgbm'].predict(X_test)
                lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
                logger.info(f"LightGBM trained with RMSE: {lgb_rmse:.4f}")
                success = True
                
            except Exception as e:
                logger.error(f"LightGBM training failed: {e}")
            
            # Train XGBoost
            try:
                self.ensemble_models['xgboost'] = xgb.XGBRegressor(
                    n_estimators=200,  # Reduced for deployment
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbosity=0,
                    early_stopping_rounds=30
                )
                
                self.ensemble_models['xgboost'].fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False
                )
                
                xgb_pred = self.ensemble_models['xgboost'].predict(X_test)
                xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
                logger.info(f"XGBoost trained with RMSE: {xgb_rmse:.4f}")
                success = True
                
            except Exception as e:
                logger.error(f"XGBoost training failed: {e}")
            
            # Train CatBoost if available
            if CATBOOST_AVAILABLE:
                try:
                    self.ensemble_models['catboost'] = cb.CatBoostRegressor(
                        iterations=200,  # Reduced for deployment
                        learning_rate=0.05,
                        depth=6,
                        random_state=42,
                        verbose=0,
                        early_stopping_rounds=30
                    )
                    
                    self.ensemble_models['catboost'].fit(
                        X_train, y_train,
                        eval_set=(X_test, y_test),
                        verbose=False
                    )
                    
                    cb_pred = self.ensemble_models['catboost'].predict(X_test)
                    cb_rmse = np.sqrt(mean_squared_error(y_test, cb_pred))
                    logger.info(f"CatBoost trained with RMSE: {cb_rmse:.4f}")
                    success = True
                    
                except Exception as e:
                    logger.error(f"CatBoost training failed: {e}")
            
            if success:
                logger.info("Ensemble models training completed")
            
            return success
            
        except Exception as e:
            logger.error(f"Ensemble models training error: {e}")
            return False
    
    def _get_enhanced_user_features(self, user_id):
        """Get enhanced user features including preferences and behavior patterns"""
        if user_id not in self.user_profiles:
            return np.zeros(40)  # Reduced feature size for deployment
        
        profile = self.user_profiles[user_id]
        features = []
        
        # Top genre preferences (normalized)
        top_genres = profile['genre_preferences'].most_common(5)  # Reduced
        genre_features = [score for _, score in top_genres] + [0] * (5 - len(top_genres))
        total_genre_score = sum(genre_features) or 1
        genre_features = [score / total_genre_score for score in genre_features]
        
        # Top language preferences (normalized)
        top_languages = profile['language_preferences'].most_common(3)  # Reduced
        language_features = [score for _, score in top_languages] + [0] * (3 - len(top_languages))
        total_language_score = sum(language_features) or 1
        language_features = [score / total_language_score for score in language_features]
        
        # Content type preferences (normalized)
        type_preferences = [
            profile['type_preferences'].get('movie', 0),
            profile['type_preferences'].get('tv', 0),
            profile['type_preferences'].get('anime', 0)
        ]
        total_type_score = sum(type_preferences) or 1
        type_features = [score / total_type_score for score in type_preferences]
        
        # Decade preferences (top 3)
        top_decades = profile['decade_preferences'].most_common(3)  # Reduced
        decade_features = [score for _, score in top_decades] + [0] * (3 - len(top_decades))
        total_decade_score = sum(decade_features) or 1
        decade_features = [score / total_decade_score for score in decade_features]
        
        # Behavioral features
        behavioral_features = [
            profile['avg_rating'] / 5.0,  # Normalized average rating
            min(profile['total_interactions'] / 100.0, 1.0),  # Normalized activity level
            profile['recency_factor'] / profile['total_interactions'] if profile['total_interactions'] > 0 else 0,  # Recency factor
            len(profile['genre_preferences']) / 20.0,  # Genre diversity
            len(profile['language_preferences']) / 10.0,  # Language diversity
            profile['interaction_counts']['like'] / max(profile['total_interactions'], 1),  # Like ratio
            profile['interaction_counts']['favorite'] / max(profile['total_interactions'], 1),  # Favorite ratio
        ]
        
        # User account features
        user = User.query.get(user_id)
        account_features = [0, 0, 0]  # Default values
        if user:
            days_since_registration = (datetime.utcnow() - user.created_at).days
            days_since_last_active = (datetime.utcnow() - user.last_active).days
            account_features = [
                min(days_since_registration / 365.0, 1.0),  # Account age (normalized to 1 year)
                max(0, 1 - days_since_last_active / 30.0),  # Recent activity (0-1, decay over 30 days)
                1 if user.is_admin else 0  # Admin flag
            ]
        
        # Combine all features
        features = (genre_features + language_features + type_features + 
                   decade_features + behavioral_features + account_features)
        
        return np.array(features)
    
    def _get_interaction_features(self, user_id, content_id):
        """Get interaction-specific features"""
        features = []
        
        # Content popularity features
        content = Content.query.get(content_id)
        if content:
            features.extend([
                content.popularity / 100.0 if content.popularity else 0,
                content.vote_count / 10000.0 if content.vote_count else 0,
                content.rating / 10.0 if content.rating else 0,
                content.runtime / 200.0 if content.runtime else 0,
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # User-content genre match
        if user_id in self.user_profiles and content and content.genres:
            try:
                user_genres = set(self.user_profiles[user_id]['genre_preferences'].keys())
                content_genres = set(json.loads(content.genres))
                genre_match = len(user_genres.intersection(content_genres)) / max(len(content_genres), 1)
                features.append(genre_match)
            except:
                features.append(0)
        else:
            features.append(0)
        
        # Temporal features
        if content and content.release_date:
            current_year = datetime.now().year
            content_age = current_year - content.release_date.year
            features.extend([
                max(0, 1 - content_age / 50.0),  # Recency score (decay over 50 years)
                1 if content_age <= 2 else 0,  # Is new release
                1 if content_age >= 30 else 0,  # Is classic
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def train_all_models(self):
        """Train all ML models"""
        try:
            logger.info("Starting comprehensive model training...")
            
            # Prepare features
            content_df = self.prepare_content_features()
            if content_df is None:
                logger.error("Failed to prepare content features")
                return False
            
            user_features = self.prepare_user_features()
            if user_features is None:
                logger.error("Failed to prepare user features")
                return False
            
            # Train models
            cf_success = self.train_collaborative_filtering_models()
            ensemble_success = self.train_ensemble_models()
            
            if cf_success or ensemble_success:
                self.models_loaded = True
                self.last_training = datetime.utcnow()
                self.save_models()
                logger.info("Comprehensive model training completed successfully")
                return True
            else:
                logger.error("All model training failed")
                return False
                
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return False
    
    def get_content_based_recommendations(self, user_id, num_recommendations=20):
        """Get enhanced content-based recommendations"""
        try:
            if self.content_similarity_matrix is None:
                return []
            
            # Get user's interaction history
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            if not user_interactions:
                return []
            
            # Calculate user preference profile based on interaction history
            content_scores = defaultdict(float)
            interaction_weights = defaultdict(float)
            
            for interaction in user_interactions:
                content_id = interaction.content_id
                if content_id not in self.content_id_to_index:
                    continue
                
                content_idx = self.content_id_to_index[content_id]
                
                # Enhanced weight calculation
                base_weight = {
                    'view': 1.0,
                    'like': 3.0,
                    'favorite': 5.0,
                    'watchlist': 4.0,
                    'search': 0.5
                }.get(interaction.interaction_type, 1.0)
                
                # Rating boost
                rating_weight = (interaction.rating / 5.0) if interaction.rating else 1.0
                
                # Temporal decay
                time_diff = (datetime.utcnow() - interaction.timestamp).days
                temporal_weight = math.exp(-time_diff / 180)  # 6-month decay
                
                final_weight = base_weight * rating_weight * temporal_weight
                interaction_weights[content_id] = final_weight
                
                # Add similar content scores
                similar_scores = self.content_similarity_matrix[content_idx]
                for similar_idx, similarity in enumerate(similar_scores):
                    if similarity > 0.1:  # Threshold for relevance
                        similar_content_id = self.index_to_content_id[similar_idx]
                        # Apply diminishing returns
                        decay_factor = 1.0 / (1.0 + interaction_weights.get(similar_content_id, 0) * 0.1)
                        content_scores[similar_content_id] += similarity * final_weight * decay_factor
            
            # Remove already interacted content
            interacted_content = set(interaction.content_id for interaction in user_interactions)
            for content_id in interacted_content:
                content_scores.pop(content_id, None)
            
            # Sort and return top recommendations
            sorted_recommendations = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Content-based recommendation error: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id, num_recommendations=20):
        """Get collaborative filtering recommendations using ensemble of models"""
        try:
            recommendations = defaultdict(float)
            
            # Custom matrix factorization predictions
            if self.custom_mf_model and user_id in self.user_to_idx:
                user_idx = self.user_to_idx[user_id]
                user_interacted = set(self.user_item_matrix.get(user_id, {}).keys())
                
                for item_id, item_idx in self.item_to_idx.items():
                    if item_id not in user_interacted:
                        predicted_rating = self.custom_mf_model.predict(user_idx, item_idx)
                        recommendations[item_id] += predicted_rating * 0.6  # Increased weight for custom model
            
            # Surprise models predictions
            if SURPRISE_AVAILABLE:
                for model_name, model in self.surprise_models.items():
                    if model:
                        try:
                            # Get all content not interacted by user
                            all_content = Content.query.all()
                            user_interacted = set(interaction.content_id for interaction in 
                                                UserInteraction.query.filter_by(user_id=user_id).all())
                            
                            for content in all_content:
                                if content.id not in user_interacted:
                                    prediction = model.predict(user_id, content.id)
                                    weight = 0.2  # Reduced weight for Surprise models
                                    recommendations[content.id] += prediction.est * weight
                        except Exception as e:
                            logger.warning(f"Prediction failed for {model_name}: {e}")
            
            # Sort and return top recommendations
            sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            return sorted_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Collaborative filtering error: {e}")
            return []
    
    def get_ensemble_recommendations(self, user_id, num_recommendations=20):
        """Get ensemble model recommendations"""
        try:
            if not self.ensemble_models or self.content_features is None:
                return []
            
            user_features = self._get_enhanced_user_features(user_id)
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_content = set(interaction.content_id for interaction in user_interactions)
            
            predictions = defaultdict(list)
            
            # Get predictions from each ensemble model
            for content_id, content_idx in self.content_id_to_index.items():
                if content_id not in interacted_content:
                    content_features = self.content_features[content_idx]
                    interaction_features = self._get_interaction_features(user_id, content_id)
                    combined_features = np.concatenate([user_features, content_features, interaction_features]).reshape(1, -1)
                    
                    # LightGBM prediction
                    if 'lightgbm' in self.ensemble_models:
                        try:
                            lgb_pred = self.ensemble_models['lightgbm'].predict(combined_features)[0]
                            predictions[content_id].append(('lightgbm', lgb_pred))
                        except:
                            pass
                    
                    # XGBoost prediction
                    if 'xgboost' in self.ensemble_models:
                        try:
                            xgb_pred = self.ensemble_models['xgboost'].predict(combined_features)[0]
                            predictions[content_id].append(('xgboost', xgb_pred))
                        except:
                            pass
                    
                    # CatBoost prediction
                    if 'catboost' in self.ensemble_models:
                        try:
                            cb_pred = self.ensemble_models['catboost'].predict(combined_features)[0]
                            predictions[content_id].append(('catboost', cb_pred))
                        except:
                            pass
            
            # Ensemble predictions with weighted voting
            final_scores = {}
            for content_id, model_predictions in predictions.items():
                if model_predictions:
                    weighted_score = 0
                    total_weight = 0
                    
                    for model_name, score in model_predictions:
                        weight = {'lightgbm': 0.4, 'xgboost': 0.35, 'catboost': 0.25}.get(model_name, 0.33)
                        weighted_score += score * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        final_scores[content_id] = weighted_score / total_weight
            
            # Sort and return top recommendations
            sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Ensemble recommendation error: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id, num_recommendations=20):
        """Get hybrid recommendations combining all approaches"""
        try:
            # Get recommendations from different algorithms
            content_based = self.get_content_based_recommendations(user_id, num_recommendations * 2)
            collaborative = self.get_collaborative_recommendations(user_id, num_recommendations * 2)
            ensemble = self.get_ensemble_recommendations(user_id, num_recommendations * 2)
            
            # Combine scores with weights
            all_scores = defaultdict(list)
            
            # Collect scores from all methods
            for content_id, score in content_based:
                all_scores[content_id].append(('content', score))
            
            for content_id, score in collaborative:
                all_scores[content_id].append(('collaborative', score))
            
            for content_id, score in ensemble:
                all_scores[content_id].append(('ensemble', score))
            
            # Hybrid scoring
            final_scores = {}
            
            for content_id, scores in all_scores.items():
                if len(scores) >= 1:  # At least one algorithm must recommend
                    # Weighted average
                    weighted_score = 0
                    total_weight = 0
                    
                    for method, score in scores:
                        weight = self.model_weights.get(method, 0.33)
                        weighted_score += score * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        # Apply confidence boost for multiple algorithm agreement
                        confidence_boost = min(len(scores) / 2.0, 1.2)
                        final_scores[content_id] = (weighted_score / total_weight) * confidence_boost
            
            # Sort and prepare result
            sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            
            result = []
            for content_id, score in sorted_recommendations[:num_recommendations]:
                reason = self._generate_recommendation_reason(user_id, content_id, all_scores[content_id])
                result.append({
                    'content_id': content_id,
                    'score': float(score),
                    'reason': reason,
                    'confidence': min(len(all_scores[content_id]) / 2.0, 1.0)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid recommendation error: {e}")
            return []
    
    def _generate_recommendation_reason(self, user_id, content_id, algorithm_scores):
        """Generate explanation for recommendation"""
        try:
            content = Content.query.get(content_id)
            if not content:
                return "Recommended for you"
            
            user_profile = self.user_profiles.get(user_id, {})
            reasons = []
            
            # Genre-based reason
            if 'genre_preferences' in user_profile:
                try:
                    content_genres = set(json.loads(content.genres or '[]'))
                    user_top_genres = set([genre for genre, _ in user_profile['genre_preferences'].most_common(3)])
                    common_genres = content_genres.intersection(user_top_genres)
                    
                    if common_genres:
                        reasons.append(f"You enjoy {', '.join(list(common_genres)[:2])}")
                except:
                    pass
            
            # Algorithm-based reason
            algorithm_types = [algo for algo, _ in algorithm_scores]
            if 'collaborative' in algorithm_types:
                reasons.append("Users with similar taste enjoyed this")
            elif 'content' in algorithm_types:
                reasons.append("Similar to content you've liked")
            elif 'ensemble' in algorithm_types:
                reasons.append("Highly recommended by our AI models")
            
            # Quality-based reason
            if content.rating and content.rating >= 8.0:
                reasons.append("Highly rated content")
            elif content.popularity and content.popularity >= 50:
                reasons.append("Trending now")
            
            # Combine reasons
            if reasons:
                return "; ".join(reasons[:2])  # Limit to 2 main reasons
            else:
                return "Discovered just for you"
                
        except Exception as e:
            logger.error(f"Reason generation error: {e}")
            return "Recommended for you"
    
    def save_models(self):
        """Save trained models to disk with compression"""
        try:
            model_data = {
                'content_features': self.content_features,
                'content_similarity_matrix': self.content_similarity_matrix,
                'content_id_to_index': getattr(self, 'content_id_to_index', {}),
                'index_to_content_id': getattr(self, 'index_to_content_id', {}),
                'user_item_matrix': getattr(self, 'user_item_matrix', {}),
                'user_profiles': getattr(self, 'user_profiles', {}),
                'user_to_idx': getattr(self, 'user_to_idx', {}),
                'item_to_idx': getattr(self, 'item_to_idx', {}),
                'idx_to_user': getattr(self, 'idx_to_user', {}),
                'idx_to_item': getattr(self, 'idx_to_item', {}),
                'model_weights': self.model_weights,
                'last_training': self.last_training.isoformat() if self.last_training else None
            }
            
            # Save core data with compression
            joblib.dump(model_data, os.path.join(MODEL_DIR, 'model_data.pkl'), compress=3)
            
            # Save TF-IDF vectorizer
            if self.tfidf_vectorizer:
                joblib.dump(self.tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), compress=3)
            
            # Save custom matrix factorization model
            if self.custom_mf_model:
                joblib.dump(self.custom_mf_model, os.path.join(MODEL_DIR, 'custom_mf_model.pkl'), compress=3)
            
            # Save Surprise models
            if SURPRISE_AVAILABLE:
                for name, model in self.surprise_models.items():
                    if model:
                        joblib.dump(model, os.path.join(MODEL_DIR, f'surprise_{name}_model.pkl'), compress=3)
            
            # Save ensemble models
            for name, model in self.ensemble_models.items():
                if model:
                    if name == 'lightgbm':
                        model.save_model(os.path.join(MODEL_DIR, f'{name}_model.txt'))
                    else:
                        joblib.dump(model, os.path.join(MODEL_DIR, f'{name}_model.pkl'), compress=3)
            
            # Save scaler
            if hasattr(self, 'scaler'):
                joblib.dump(self.scaler, os.path.join(MODEL_DIR, 'scaler.pkl'), compress=3)
            
            logger.info("Models saved successfully with compression")
            
        except Exception as e:
            logger.error(f"Model saving error: {e}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            model_data_path = os.path.join(MODEL_DIR, 'model_data.pkl')
            if not os.path.exists(model_data_path):
                return False
            
            # Load core data
            model_data = joblib.load(model_data_path)
            self.content_features = model_data.get('content_features')
            self.content_similarity_matrix = model_data.get('content_similarity_matrix')
            self.content_id_to_index = model_data.get('content_id_to_index', {})
            self.index_to_content_id = model_data.get('index_to_content_id', {})
            self.user_item_matrix = model_data.get('user_item_matrix', {})
            self.user_profiles = model_data.get('user_profiles', {})
            self.user_to_idx = model_data.get('user_to_idx', {})
            self.item_to_idx = model_data.get('item_to_idx', {})
            self.idx_to_user = model_data.get('idx_to_user', {})
            self.idx_to_item = model_data.get('idx_to_item', {})
            self.model_weights = model_data.get('model_weights', self.model_weights)
            
            last_training_str = model_data.get('last_training')
            if last_training_str:
                self.last_training = datetime.fromisoformat(last_training_str)
            
            # Load TF-IDF vectorizer
            tfidf_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
            if os.path.exists(tfidf_path):
                self.tfidf_vectorizer = joblib.load(tfidf_path)
            
            # Load custom matrix factorization model
            mf_path = os.path.join(MODEL_DIR, 'custom_mf_model.pkl')
            if os.path.exists(mf_path):
                self.custom_mf_model = joblib.load(mf_path)
            
            # Load Surprise models
            if SURPRISE_AVAILABLE:
                surprise_models = ['svd', 'nmf']
                for name in surprise_models:
                    model_path = os.path.join(MODEL_DIR, f'surprise_{name}_model.pkl')
                    if os.path.exists(model_path):
                        self.surprise_models[name] = joblib.load(model_path)
            
            # Load ensemble models
            ensemble_models = ['xgboost']
            if CATBOOST_AVAILABLE:
                ensemble_models.append('catboost')
                
            for name in ensemble_models:
                model_path = os.path.join(MODEL_DIR, f'{name}_model.pkl')
                if os.path.exists(model_path):
                    self.ensemble_models[name] = joblib.load(model_path)
            
            # Load LightGBM model
            lgb_path = os.path.join(MODEL_DIR, 'lightgbm_model.txt')
            if os.path.exists(lgb_path):
                self.ensemble_models['lightgbm'] = lgb.Booster(model_file=lgb_path)
            
            # Load scaler
            scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            self.models_loaded = True
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return False
    
    def needs_retraining(self):
        """Check if models need retraining"""
        if not self.last_training:
            return True
        
        # Retrain if last training was more than 24 hours ago
        time_since_training = datetime.utcnow() - self.last_training
        return time_since_training > timedelta(hours=24)

# Initialize recommendation engine
recommendation_engine = AdvancedRecommendationEngine()

# API Routes
@app.route('/api/recommendations', methods=['POST'])
def get_personalized_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        num_recommendations = data.get('num_recommendations', 20)
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        # Check if models need training/retraining
        if not recommendation_engine.models_loaded or recommendation_engine.needs_retraining():
            logger.info("Training/retraining models...")
            success = recommendation_engine.train_all_models()
            if not success:
                return jsonify({'error': 'Model training failed'}), 500
        
        # Get hybrid recommendations
        recommendations = recommendation_engine.get_hybrid_recommendations(user_id, num_recommendations)
        
        available_algorithms = ['enhanced_content_based', 'custom_collaborative_filtering']
        if SURPRISE_AVAILABLE:
            available_algorithms.append('surprise_collaborative_filtering')
        available_algorithms.extend(['lightgbm_ensemble', 'xgboost_ensemble'])
        if CATBOOST_AVAILABLE:
            available_algorithms.append('catboost_ensemble')
        available_algorithms.append('advanced_hybrid_ml')
        
        return jsonify({
            'recommendations': recommendations,
            'algorithm': 'advanced_hybrid_ml',
            'model_last_trained': recommendation_engine.last_training.isoformat() if recommendation_engine.last_training else None,
            'available_algorithms': available_algorithms,
            'models_active': {
                'surprise_available': SURPRISE_AVAILABLE,
                'catboost_available': CATBOOST_AVAILABLE,
                'nltk_available': NLTK_AVAILABLE
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Personalized recommendation error: {e}")
        return jsonify({'error': 'Failed to generate recommendations'}), 500

@app.route('/api/recommendations/content-based', methods=['POST'])
def get_content_based_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        num_recommendations = data.get('num_recommendations', 20)
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        if not recommendation_engine.models_loaded:
            recommendation_engine.train_all_models()
        
        recommendations = recommendation_engine.get_content_based_recommendations(user_id, num_recommendations)
        
        result = []
        for content_id, score in recommendations:
            result.append({
                'content_id': content_id,
                'score': float(score),
                'reason': 'Similar to your interests',
                'algorithm': 'enhanced_content_based'
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Content-based recommendation error: {e}")
        return jsonify({'error': 'Failed to generate content-based recommendations'}), 500

@app.route('/api/recommendations/collaborative', methods=['POST'])
def get_collaborative_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        num_recommendations = data.get('num_recommendations', 20)
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        if not recommendation_engine.models_loaded:
            recommendation_engine.train_all_models()
        
        recommendations = recommendation_engine.get_collaborative_recommendations(user_id, num_recommendations)
        
        result = []
        for content_id, score in recommendations:
            result.append({
                'content_id': content_id,
                'score': float(score),
                'reason': 'Users with similar taste enjoyed this',
                'algorithm': 'ensemble_collaborative_filtering'
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Collaborative recommendation error: {e}")
        return jsonify({'error': 'Failed to generate collaborative recommendations'}), 500

@app.route('/api/train-models', methods=['POST'])
def train_models():
    try:
        # Optional parameters
        data = request.get_json() or {}
        force_retrain = data.get('force_retrain', False)
        
        if force_retrain or recommendation_engine.needs_retraining():
            success = recommendation_engine.train_all_models()
            
            if success:
                models_trained = {
                    'collaborative_filtering': len(recommendation_engine.surprise_models) + 1,
                    'ensemble_models': len(recommendation_engine.ensemble_models),
                    'content_based': 1 if recommendation_engine.content_similarity_matrix is not None else 0
                }
                
                return jsonify({
                    'message': 'Models trained successfully',
                    'last_training': recommendation_engine.last_training.isoformat(),
                    'models_trained': models_trained,
                    'library_status': {
                        'surprise_available': SURPRISE_AVAILABLE,
                        'catboost_available': CATBOOST_AVAILABLE,
                        'nltk_available': NLTK_AVAILABLE
                    }
                }), 200
            else:
                return jsonify({'error': 'Model training failed'}), 500
        else:
            return jsonify({
                'message': 'Models are up to date',
                'last_training': recommendation_engine.last_training.isoformat() if recommendation_engine.last_training else None
            }), 200
            
    except Exception as e:
        logger.error(f"Model training error: {e}")
        return jsonify({'error': 'Model training failed'}), 500

@app.route('/api/model-status', methods=['GET'])
def get_model_status():
    try:
        available_algorithms = ['enhanced_content_based', 'custom_collaborative_filtering']
        if SURPRISE_AVAILABLE:
            available_algorithms.append('surprise_collaborative_filtering')
        available_algorithms.extend(['lightgbm_ensemble', 'xgboost_ensemble'])
        if CATBOOST_AVAILABLE:
            available_algorithms.append('catboost_ensemble')
        available_algorithms.append('advanced_hybrid_ml')
        
        return jsonify({
            'models_loaded': recommendation_engine.models_loaded,
            'last_training': recommendation_engine.last_training.isoformat() if recommendation_engine.last_training else None,
            'available_algorithms': available_algorithms,
            'library_status': {
                'surprise_available': SURPRISE_AVAILABLE,
                'catboost_available': CATBOOST_AVAILABLE,
                'nltk_available': NLTK_AVAILABLE
            },
            'model_details': {
                'content_features_shape': recommendation_engine.content_features.shape if recommendation_engine.content_features is not None else None,
                'content_similarity_available': recommendation_engine.content_similarity_matrix is not None,
                'collaborative_models': list(recommendation_engine.surprise_models.keys()) + ['custom_matrix_factorization'],
                'ensemble_models': list(recommendation_engine.ensemble_models.keys()),
                'user_count': len(recommendation_engine.user_item_matrix) if hasattr(recommendation_engine, 'user_item_matrix') else 0,
                'content_count': len(recommendation_engine.content_id_to_index) if hasattr(recommendation_engine, 'content_id_to_index') else 0
            },
            'text_processing': {
                'tfidf_vectorizer_available': recommendation_engine.tfidf_vectorizer is not None,
                'advanced_text_processing': NLTK_AVAILABLE,
                'nlp_features': ['stemming', 'lemmatization', 'stopword_removal', 'keyword_extraction'] if NLTK_AVAILABLE else ['basic_text_processing']
            },
            'feature_engineering': {
                'multi_label_encoding': True,
                'temporal_weighting': True,
                'interaction_features': True,
                'reduced_complexity_for_deployment': True
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Model status error: {e}")
        return jsonify({'error': 'Failed to get model status'}), 500

@app.route('/api/user-profile/<int:user_id>', methods=['GET'])
def get_user_profile(user_id):
    try:
        if user_id not in recommendation_engine.user_profiles:
            return jsonify({'error': 'User profile not found'}), 404
        
        profile = recommendation_engine.user_profiles[user_id]
        
        # Format the profile for response
        formatted_profile = {
            'user_id': user_id,
            'top_genres': [
                {'genre': genre, 'score': score} 
                for genre, score in profile['genre_preferences'].most_common(10)
            ],
            'top_languages': [
                {'language': lang, 'score': score} 
                for lang, score in profile['language_preferences'].most_common(5)
            ],
            'content_type_preferences': dict(profile['type_preferences']),
            'decade_preferences': [
                {'decade': decade, 'score': score} 
                for decade, score in profile['decade_preferences'].most_common(5)
            ],
            'interaction_statistics': dict(profile['interaction_counts']),
            'average_rating': profile['avg_rating'],
            'total_interactions': profile['total_interactions'],
            'activity_recency': profile['recency_factor'] / max(profile['total_interactions'], 1)
        }
        
        return jsonify(formatted_profile), 200
        
    except Exception as e:
        logger.error(f"User profile error: {e}")
        return jsonify({'error': 'Failed to get user profile'}), 500

# Background model training scheduler (simplified for deployment)
def background_model_training():
    """Background task to retrain models periodically"""
    while True:
        try:
            if recommendation_engine.needs_retraining():
                logger.info("Starting background model training...")
                recommendation_engine.train_all_models()
            
            # Sleep for 12 hours (increased for deployment)
            time.sleep(12 * 60 * 60)
            
        except Exception as e:
            logger.error(f"Background training error: {e}")
            time.sleep(2 * 60 * 60)  # Wait 2 hours before retry

# Start background training thread
training_thread = threading.Thread(target=background_model_training, daemon=True)
training_thread.start()

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    available_algorithms = []
    if recommendation_engine.content_similarity_matrix is not None:
        available_algorithms.append('Enhanced Content-Based Filtering')
    available_algorithms.append('Custom Matrix Factorization')
    if SURPRISE_AVAILABLE:
        available_algorithms.append('Scikit-Surprise Collaborative Filtering')
    available_algorithms.extend(['LightGBM Ensemble', 'XGBoost Ensemble'])
    if CATBOOST_AVAILABLE:
        available_algorithms.append('CatBoost Ensemble')
    available_algorithms.append('Advanced Hybrid Recommendation')
    
    features = [
        'Robust Error Handling',
        'Fallback Mechanisms',
        'Reduced Complexity for Deployment',
        'Multi-Label Feature Engineering',
        'Temporal Interaction Weighting',
        'Explainable Recommendations'
    ]
    
    if NLTK_AVAILABLE:
        features.append('Advanced Text Processing with NLP')
    else:
        features.append('Basic Text Processing')
        
    return jsonify({
        'status': 'healthy',
        'service': 'advanced-ml-recommendation-service',
        'models_loaded': recommendation_engine.models_loaded,
        'algorithms_available': available_algorithms,
        'features': features,
        'library_status': {
            'surprise_available': SURPRISE_AVAILABLE,
            'catboost_available': CATBOOST_AVAILABLE,
            'nltk_available': NLTK_AVAILABLE
        },
        'timestamp': datetime.utcnow().isoformat()
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)