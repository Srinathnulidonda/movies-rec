from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import requests
import asyncio
import concurrent.futures
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import joblib
import threading
import time

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

# Database Models (matching main backend)
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

# Advanced Recommendation Engine
class AdvancedRecommendationEngine:
    def __init__(self):
        self.content_features = None
        self.user_features = None
        self.tfidf_vectorizer = None
        self.content_similarity_matrix = None
        self.svd_model = None
        self.neural_model = None
        self.genre_encoder = None
        self.language_encoder = None
        self.models_loaded = False
        self.last_training = None
        
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
        """Extract and prepare content features for ML models"""
        try:
            contents = Content.query.all()
            if not contents:
                return None
            
            content_data = []
            for content in contents:
                genres = json.loads(content.genres or '[]')
                languages = json.loads(content.languages or '[]')
                
                content_data.append({
                    'content_id': content.id,
                    'title': content.title or '',
                    'overview': content.overview or '',
                    'genres': genres,
                    'languages': languages,
                    'content_type': content.content_type,
                    'rating': content.rating or 0,
                    'popularity': content.popularity or 0,
                    'runtime': content.runtime or 0,
                    'release_year': content.release_date.year if content.release_date else 2000
                })
            
            df = pd.DataFrame(content_data)
            
            # Create text features for TF-IDF
            df['text_features'] = df['title'] + ' ' + df['overview'] + ' ' + df['genres'].apply(lambda x: ' '.join(x))
            
            # TF-IDF for text features
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['text_features'])
            
            # Calculate content similarity matrix
            self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Prepare numerical features
            numerical_features = df[['rating', 'popularity', 'runtime', 'release_year']].fillna(0)
            
            # Encode categorical features
            genre_features = self._encode_multi_label_features(df['genres'].tolist())
            language_features = self._encode_multi_label_features(df['languages'].tolist())
            type_features = pd.get_dummies(df['content_type'])
            
            # Combine all features
            self.content_features = np.hstack([
                tfidf_matrix.toarray(),
                numerical_features.values,
                genre_features,
                language_features,
                type_features.values
            ])
            
            # Store content ID mapping
            self.content_id_to_index = {content_id: idx for idx, content_id in enumerate(df['content_id'])}
            self.index_to_content_id = {idx: content_id for content_id, idx in self.content_id_to_index.items()}
            
            logger.info(f"Prepared features for {len(contents)} content items")
            return df
            
        except Exception as e:
            logger.error(f"Content feature preparation error: {e}")
            return None
    
    def _encode_multi_label_features(self, multi_label_list):
        """Encode multi-label features (genres, languages) into binary matrix"""
        all_labels = set()
        for labels in multi_label_list:
            all_labels.update(labels)
        
        all_labels = sorted(list(all_labels))
        encoded_features = np.zeros((len(multi_label_list), len(all_labels)))
        
        for i, labels in enumerate(multi_label_list):
            for label in labels:
                if label in all_labels:
                    encoded_features[i, all_labels.index(label)] = 1
        
        return encoded_features
    
    def prepare_user_features(self):
        """Prepare user interaction features"""
        try:
            users = User.query.all()
            interactions = UserInteraction.query.all()
            
            if not users or not interactions:
                return None
            
            # Create user-item matrix
            user_item_matrix = {}
            user_genre_preferences = {}
            user_language_preferences = {}
            
            for interaction in interactions:
                user_id = interaction.user_id
                content_id = interaction.content_id
                
                if user_id not in user_item_matrix:
                    user_item_matrix[user_id] = {}
                
                # Weight different interaction types
                weight = {
                    'view': 1.0,
                    'like': 2.0,
                    'favorite': 3.0,
                    'watchlist': 2.5,
                    'search': 0.5
                }.get(interaction.interaction_type, 1.0)
                
                if interaction.rating:
                    weight *= (interaction.rating / 5.0)  # Normalize rating to 0-2 range
                
                user_item_matrix[user_id][content_id] = weight
                
                # Extract content preferences
                content = Content.query.get(content_id)
                if content:
                    if user_id not in user_genre_preferences:
                        user_genre_preferences[user_id] = Counter()
                        user_language_preferences[user_id] = Counter()
                    
                    genres = json.loads(content.genres or '[]')
                    languages = json.loads(content.languages or '[]')
                    
                    for genre in genres:
                        user_genre_preferences[user_id][genre] += weight
                    
                    for language in languages:
                        user_language_preferences[user_id][language] += weight
            
            self.user_item_matrix = user_item_matrix
            self.user_genre_preferences = user_genre_preferences
            self.user_language_preferences = user_language_preferences
            
            logger.info(f"Prepared user features for {len(users)} users")
            return True
            
        except Exception as e:
            logger.error(f"User feature preparation error: {e}")
            return None
    
    def train_svd_model(self):
        """Train SVD model for collaborative filtering"""
        try:
            if not hasattr(self, 'user_item_matrix'):
                return False
            
            # Prepare data for Surprise
            ratings_data = []
            for user_id, items in self.user_item_matrix.items():
                for item_id, rating in items.items():
                    ratings_data.append([user_id, item_id, rating])
            
            if not ratings_data:
                return False
            
            df = pd.DataFrame(ratings_data, columns=['user_id', 'item_id', 'rating'])
            
            # Create Surprise dataset
            reader = Reader(rating_scale=(0, 5))
            dataset = Dataset.load_from_df(df, reader)
            
            # Train SVD model
            self.svd_model = SVD(n_factors=50, random_state=42)
            trainset = dataset.build_full_trainset()
            self.svd_model.fit(trainset)
            
            logger.info("SVD model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"SVD training error: {e}")
            return False
    
    def train_neural_model(self):
        """Train neural network for deep learning recommendations"""
        try:
            if not hasattr(self, 'user_item_matrix') or self.content_features is None:
                return False
            
            # Prepare training data
            X_user = []
            X_content = []
            y = []
            
            for user_id, items in self.user_item_matrix.items():
                for content_id, rating in items.items():
                    if content_id in self.content_id_to_index:
                        # User features (simplified)
                        user_features = self._get_user_features(user_id)
                        content_idx = self.content_id_to_index[content_id]
                        content_features = self.content_features[content_idx]
                        
                        X_user.append(user_features)
                        X_content.append(content_features)
                        y.append(rating)
            
            if not X_user:
                return False
            
            X_user = np.array(X_user)
            X_content = np.array(X_content)
            y = np.array(y)
            
            # Build neural network
            user_input = Input(shape=(X_user.shape[1],), name='user_input')
            content_input = Input(shape=(X_content.shape[1],), name='content_input')
            
            # User branch
            user_dense = Dense(64, activation='relu')(user_input)
            user_dense = Dropout(0.3)(user_dense)
            user_dense = Dense(32, activation='relu')(user_dense)
            
            # Content branch
            content_dense = Dense(128, activation='relu')(content_input)
            content_dense = Dropout(0.3)(content_dense)
            content_dense = Dense(64, activation='relu')(content_dense)
            
            # Combine branches
            combined = Concatenate()([user_dense, content_dense])
            combined = Dense(32, activation='relu')(combined)
            combined = Dropout(0.3)(combined)
            output = Dense(1, activation='sigmoid')(combined)
            
            self.neural_model = Model(inputs=[user_input, content_input], outputs=output)
            self.neural_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train model
            self.neural_model.fit(
                [X_user, X_content], y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            logger.info("Neural network model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Neural network training error: {e}")
            return False
    
    def _get_user_features(self, user_id):
        """Get user features for neural network"""
        user = User.query.get(user_id)
        if not user:
            return np.zeros(10)  # Default features
        
        # Basic user features
        features = []
        
        # Preferred genres (top 5)
        if user_id in self.user_genre_preferences:
            top_genres = self.user_genre_preferences[user_id].most_common(5)
            genre_scores = [score for _, score in top_genres] + [0] * (5 - len(top_genres))
        else:
            genre_scores = [0] * 5
        
        # Preferred languages (top 3)
        if user_id in self.user_language_preferences:
            top_languages = self.user_language_preferences[user_id].most_common(3)
            language_scores = [score for _, score in top_languages] + [0] * (3 - len(top_languages))
        else:
            language_scores = [0] * 3
        
        # User activity level
        interaction_count = UserInteraction.query.filter_by(user_id=user_id).count()
        activity_level = min(interaction_count / 100.0, 1.0)  # Normalize to 0-1
        
        # Days since registration
        days_since_registration = (datetime.utcnow() - user.created_at).days
        registration_age = min(days_since_registration / 365.0, 1.0)  # Normalize to 0-1
        
        features = genre_scores + language_scores + [activity_level, registration_age]
        return np.array(features)
    
    def train_all_models(self):
        """Train all ML models"""
        try:
            logger.info("Starting model training...")
            
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
            svd_success = self.train_svd_model()
            neural_success = self.train_neural_model()
            
            if svd_success or neural_success:
                self.models_loaded = True
                self.last_training = datetime.utcnow()
                self.save_models()
                logger.info("Model training completed successfully")
                return True
            else:
                logger.error("All model training failed")
                return False
                
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return False
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            model_data = {
                'content_features': self.content_features,
                'content_similarity_matrix': self.content_similarity_matrix,
                'content_id_to_index': getattr(self, 'content_id_to_index', {}),
                'index_to_content_id': getattr(self, 'index_to_content_id', {}),
                'user_item_matrix': getattr(self, 'user_item_matrix', {}),
                'user_genre_preferences': getattr(self, 'user_genre_preferences', {}),
                'user_language_preferences': getattr(self, 'user_language_preferences', {}),
                'last_training': self.last_training.isoformat() if self.last_training else None
            }
            
            # Save core data
            joblib.dump(model_data, os.path.join(MODEL_DIR, 'model_data.pkl'))
            
            # Save TF-IDF vectorizer
            if self.tfidf_vectorizer:
                joblib.dump(self.tfidf_vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
            
            # Save SVD model
            if self.svd_model:
                joblib.dump(self.svd_model, os.path.join(MODEL_DIR, 'svd_model.pkl'))
            
            # Save neural network model
            if self.neural_model:
                self.neural_model.save(os.path.join(MODEL_DIR, 'neural_model.h5'))
            
            logger.info("Models saved successfully")
            
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
            self.user_genre_preferences = model_data.get('user_genre_preferences', {})
            self.user_language_preferences = model_data.get('user_language_preferences', {})
            
            last_training_str = model_data.get('last_training')
            if last_training_str:
                self.last_training = datetime.fromisoformat(last_training_str)
            
            # Load TF-IDF vectorizer
            tfidf_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
            if os.path.exists(tfidf_path):
                self.tfidf_vectorizer = joblib.load(tfidf_path)
            
            # Load SVD model
            svd_path = os.path.join(MODEL_DIR, 'svd_model.pkl')
            if os.path.exists(svd_path):
                self.svd_model = joblib.load(svd_path)
            
            # Load neural network model
            neural_path = os.path.join(MODEL_DIR, 'neural_model.h5')
            if os.path.exists(neural_path):
                self.neural_model = tf.keras.models.load_model(neural_path)
            
            self.models_loaded = True
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return False
    
    def get_content_based_recommendations(self, user_id, num_recommendations=20):
        """Get content-based recommendations"""
        try:
            if self.content_similarity_matrix is None:
                return []
            
            # Get user's interaction history
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            if not user_interactions:
                return []
            
            # Calculate user preference profile
            content_scores = defaultdict(float)
            
            for interaction in user_interactions:
                content_id = interaction.content_id
                if content_id not in self.content_id_to_index:
                    continue
                
                content_idx = self.content_id_to_index[content_id]
                
                # Weight by interaction type and rating
                weight = {
                    'view': 1.0,
                    'like': 2.0,
                    'favorite': 3.0,
                    'watchlist': 2.5
                }.get(interaction.interaction_type, 1.0)
                
                if interaction.rating:
                    weight *= (interaction.rating / 5.0)
                
                # Add similar content scores
                similar_scores = self.content_similarity_matrix[content_idx]
                for similar_idx, similarity in enumerate(similar_scores):
                    similar_content_id = self.index_to_content_id[similar_idx]
                    content_scores[similar_content_id] += similarity * weight
            
            # Remove already interacted content
            interacted_content = {interaction.content_id for interaction in user_interactions}
            for content_id in interacted_content:
                content_scores.pop(content_id, None)
            
            # Sort by score and return top recommendations
            recommendations = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
            return [(content_id, score) for content_id, score in recommendations[:num_recommendations]]
            
        except Exception as e:
            logger.error(f"Content-based recommendation error: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id, num_recommendations=20):
        """Get collaborative filtering recommendations using SVD"""
        try:
            if not self.svd_model or user_id not in self.user_item_matrix:
                return []
            
            # Get all content IDs
            all_content_ids = set()
            for items in self.user_item_matrix.values():
                all_content_ids.update(items.keys())
            
            # Get user's interacted content
            user_interacted = set(self.user_item_matrix.get(user_id, {}).keys())
            
            # Predict ratings for uninteracted content
            predictions = []
            for content_id in all_content_ids:
                if content_id not in user_interacted:
                    predicted_rating = self.svd_model.predict(user_id, content_id).est
                    predictions.append((content_id, predicted_rating))
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Collaborative filtering error: {e}")
            return []
    
    def get_neural_recommendations(self, user_id, num_recommendations=20):
        """Get neural network-based recommendations"""
        try:
            if not self.neural_model or self.content_features is None:
                return []
            
            # Get user features
            user_features = self._get_user_features(user_id)
            
            # Get user's interacted content
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_content = {interaction.content_id for interaction in user_interactions}
            
            # Predict for all content
            predictions = []
            for content_id, content_idx in self.content_id_to_index.items():
                if content_id not in interacted_content:
                    content_features = self.content_features[content_idx]
                    
                    # Predict using neural network
                    user_input = np.array([user_features])
                    content_input = np.array([content_features])
                    
                    predicted_score = self.neural_model.predict([user_input, content_input])[0][0]
                    predictions.append((content_id, float(predicted_score)))
            
            # Sort by predicted score
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Neural recommendation error: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id, num_recommendations=20):
        """Get hybrid recommendations combining multiple approaches"""
        try:
            # Get recommendations from different algorithms
            content_based = self.get_content_based_recommendations(user_id, num_recommendations * 2)
            collaborative = self.get_collaborative_recommendations(user_id, num_recommendations * 2)
            neural = self.get_neural_recommendations(user_id, num_recommendations * 2)
            
            # Combine scores with weights
            combined_scores = defaultdict(float)
            weights = {'content': 0.3, 'collaborative': 0.4, 'neural': 0.3}
            
            # Add content-based scores
            for content_id, score in content_based:
                combined_scores[content_id] += score * weights['content']
            
            # Add collaborative filtering scores
            for content_id, score in collaborative:
                combined_scores[content_id] += score * weights['collaborative']
            
            # Add neural network scores
            for content_id, score in neural:
                combined_scores[content_id] += score * weights['neural']
            
            # Sort and return top recommendations
            recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Add recommendation reasons
            result = []
            for content_id, score in recommendations[:num_recommendations]:
                reason = self._get_recommendation_reason(user_id, content_id)
                result.append({
                    'content_id': content_id,
                    'score': score,
                    'reason': reason
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid recommendation error: {e}")
            return []
    
    def _get_recommendation_reason(self, user_id, content_id):
        """Generate explanation for recommendation"""
        try:
            content = Content.query.get(content_id)
            if not content:
                return "Recommended for you"
            
            # Get user's preferred genres
            if user_id in self.user_genre_preferences:
                user_top_genres = set([genre for genre, _ in self.user_genre_preferences[user_id].most_common(3)])
                content_genres = set(json.loads(content.genres or '[]'))
                
                common_genres = user_top_genres.intersection(content_genres)
                if common_genres:
                    return f"Because you like {', '.join(list(common_genres)[:2])}"
            
            # Check if highly rated
            if content.rating and content.rating >= 8.0:
                return "Highly rated content"
            
            # Check if popular
            if content.popularity and content.popularity >= 50:
                return "Trending now"
            
            return "Recommended for you"
            
        except Exception as e:
            logger.error(f"Recommendation reason error: {e}")
            return "Recommended for you"
    
    def get_diversity_adjusted_recommendations(self, recommendations, diversity_factor=0.3):
        """Adjust recommendations for diversity"""
        try:
            if not recommendations:
                return recommendations
            
            # Get content details for diversity calculation
            content_details = {}
            for rec in recommendations:
                content = Content.query.get(rec['content_id'])
                if content:
                    content_details[rec['content_id']] = {
                        'genres': json.loads(content.genres or '[]'),
                        'content_type': content.content_type,
                        'languages': json.loads(content.languages or '[]')
                    }
            
            # Select diverse recommendations
            final_recommendations = []
            selected_genres = set()
            selected_types = set()
            
            for rec in recommendations:
                content_id = rec['content_id']
                if content_id in content_details:
                    content_info = content_details[content_id]
                    
                    # Calculate diversity bonus
                    genre_diversity = len(set(content_info['genres']) - selected_genres) > 0
                    type_diversity = content_info['content_type'] not in selected_types
                    
                    if genre_diversity or type_diversity or len(final_recommendations) < 5:
                        final_recommendations.append(rec)
                        selected_genres.update(content_info['genres'])
                        selected_types.add(content_info['content_type'])
                    
                    if len(final_recommendations) >= len(recommendations):
                        break
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Diversity adjustment error: {e}")
            return recommendations

# Initialize recommendation engine
recommendation_engine = AdvancedRecommendationEngine()

# API Routes
@app.route('/api/recommendations', methods=['POST'])
def get_personalized_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        # Check if models need training/retraining
        if not recommendation_engine.models_loaded or recommendation_engine._needs_retraining():
            logger.info("Training/retraining models...")
            success = recommendation_engine.train_all_models()
            if not success:
                return jsonify({'error': 'Model training failed'}), 500
        
        # Get hybrid recommendations
        recommendations = recommendation_engine.get_hybrid_recommendations(user_id, 30)
        
        # Apply diversity adjustment
        diverse_recommendations = recommendation_engine.get_diversity_adjusted_recommendations(recommendations)
        
        return jsonify({
            'recommendations': diverse_recommendations,
            'algorithm': 'hybrid_ml',
            'model_last_trained': recommendation_engine.last_training.isoformat() if recommendation_engine.last_training else None
        }), 200
        
    except Exception as e:
        logger.error(f"Personalized recommendation error: {e}")
        return jsonify({'error': 'Failed to generate recommendations'}), 500

@app.route('/api/recommendations/content-based', methods=['POST'])
def get_content_based_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        if not recommendation_engine.models_loaded:
            recommendation_engine.train_all_models()
        
        recommendations = recommendation_engine.get_content_based_recommendations(user_id, 20)
        
        result = []
        for content_id, score in recommendations:
            result.append({
                'content_id': content_id,
                'score': score,
                'reason': 'Similar to your interests'
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
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        if not recommendation_engine.models_loaded:
            recommendation_engine.train_all_models()
        
        recommendations = recommendation_engine.get_collaborative_recommendations(user_id, 20)
        
        result = []
        for content_id, score in recommendations:
            result.append({
                'content_id': content_id,
                'score': score,
                'reason': 'Users like you also enjoyed this'
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Collaborative recommendation error: {e}")
        return jsonify({'error': 'Failed to generate collaborative recommendations'}), 500

@app.route('/api/recommendations/neural', methods=['POST'])
def get_neural_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        if not recommendation_engine.models_loaded:
            recommendation_engine.train_all_models()
        
        recommendations = recommendation_engine.get_neural_recommendations(user_id, 20)
        
        result = []
        for content_id, score in recommendations:
            result.append({
                'content_id': content_id,
                'score': score,
                'reason': 'AI-powered recommendation'
            })
        
        return jsonify({'recommendations': result}), 200
        
    except Exception as e:
        logger.error(f"Neural recommendation error: {e}")
        return jsonify({'error': 'Failed to generate neural recommendations'}), 500

@app.route('/api/train-models', methods=['POST'])
def train_models():
    try:
        success = recommendation_engine.train_all_models()
        
        if success:
            return jsonify({
                'message': 'Models trained successfully',
                'last_training': recommendation_engine.last_training.isoformat()
            }), 200
        else:
            return jsonify({'error': 'Model training failed'}), 500
            
    except Exception as e:
        logger.error(f"Model training error: {e}")
        return jsonify({'error': 'Model training failed'}), 500

@app.route('/api/model-status', methods=['GET'])
def get_model_status():
    try:
        return jsonify({
            'models_loaded': recommendation_engine.models_loaded,
            'last_training': recommendation_engine.last_training.isoformat() if recommendation_engine.last_training else None,
            'available_algorithms': ['content_based', 'collaborative', 'neural', 'hybrid'],
            'content_features_shape': recommendation_engine.content_features.shape if recommendation_engine.content_features is not None else None,
            'user_count': len(recommendation_engine.user_item_matrix) if hasattr(recommendation_engine, 'user_item_matrix') else 0
        }), 200
        
    except Exception as e:
        logger.error(f"Model status error: {e}")
        return jsonify({'error': 'Failed to get model status'}), 500

# Helper method for recommendation engine
def _needs_retraining(self):
    """Check if models need retraining"""
    if not self.last_training:
        return True
    
    # Retrain if last training was more than 24 hours ago
    time_since_training = datetime.utcnow() - self.last_training
    return time_since_training > timedelta(hours=24)

# Add method to recommendation engine
recommendation_engine._needs_retraining = _needs_retraining.__get__(recommendation_engine, AdvancedRecommendationEngine)

# Background model training scheduler
def background_model_training():
    """Background task to retrain models periodically"""
    while True:
        try:
            if recommendation_engine._needs_retraining():
                logger.info("Starting background model training...")
                recommendation_engine.train_all_models()
            
            # Sleep for 6 hours
            time.sleep(6 * 60 * 60)
            
        except Exception as e:
            logger.error(f"Background training error: {e}")
            time.sleep(60 * 60)  # Wait 1 hour before retry

# Start background training thread
training_thread = threading.Thread(target=background_model_training, daemon=True)
training_thread.start()

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'ml-recommendation-service',
        'models_loaded': recommendation_engine.models_loaded,
        'timestamp': datetime.utcnow().isoformat()
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)