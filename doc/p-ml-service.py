# ml-service/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import implicit
from scipy.sparse import csr_matrix, coo_matrix
import networkx as nx
from collections import defaultdict, Counter
import requests
import json
import pickle
import os
from datetime import datetime, timedelta
import redis
from threading import Thread
import time
import hashlib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'https://backend-app-970m.onrender.com')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
MODEL_CACHE_DURATION = 3600  # 1 hour

# Initialize Redis for caching
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()  # Test connection
except:
    redis_client = None
    print("Redis not available, caching disabled")

# Genre mapping from backend
GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western", 10759: "Action & Adventure",
    10762: "Kids", 10763: "News", 10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap",
    10767: "Talk", 10768: "War & Politics"
}

class CustomCollaborativeFiltering:
    """Custom implementation of collaborative filtering algorithms"""
    
    def __init__(self, n_factors=50, learning_rate=0.01, regularization=0.01, epochs=20):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        
    def fit(self, ratings_df):
        """Fit the collaborative filtering model"""
        # Create user and item mappings
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['item_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = ratings_df['rating'].mean()
        
        # Convert to matrices
        user_indices = [self.user_to_idx[u] for u in ratings_df['user_id']]
        item_indices = [self.item_to_idx[i] for i in ratings_df['item_id']]
        ratings = ratings_df['rating'].values
        
        # Training loop
        for epoch in range(self.epochs):
            for user_idx, item_idx, rating in zip(user_indices, item_indices, ratings):
                # Predict rating
                prediction = (self.global_bias + 
                            self.user_bias[user_idx] + 
                            self.item_bias[item_idx] +
                            np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
                
                # Calculate error
                error = rating - prediction
                
                # Update biases
                self.user_bias[user_idx] += self.learning_rate * (error - self.regularization * self.user_bias[user_idx])
                self.item_bias[item_idx] += self.learning_rate * (error - self.regularization * self.item_bias[item_idx])
                
                # Update factors
                user_factors_copy = self.user_factors[user_idx].copy()
                self.user_factors[user_idx] += self.learning_rate * (error * self.item_factors[item_idx] - 
                                                                   self.regularization * self.user_factors[user_idx])
                self.item_factors[item_idx] += self.learning_rate * (error * user_factors_copy - 
                                                                   self.regularization * self.item_factors[item_idx])
            
            # Decay learning rate
            self.learning_rate *= 0.99
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_bias
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        prediction = (self.global_bias + 
                     self.user_bias[user_idx] + 
                     self.item_bias[item_idx] +
                     np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        
        return prediction
    
    def recommend(self, user_id, n_recommendations=10, exclude_known=True):
        """Get recommendations for a user"""
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Calculate predictions for all items
        predictions = []
        for item_idx in range(len(self.idx_to_item)):
            item_id = self.idx_to_item[item_idx]
            prediction = (self.global_bias + 
                         self.user_bias[user_idx] + 
                         self.item_bias[item_idx] +
                         np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
            predictions.append((item_id, prediction))
        
        # Sort by prediction score
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]

class AdvancedRecommendationEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.vectorizers = {}
        self.user_profiles = {}
        self.content_features = {}
        self.interaction_matrix = None
        self.content_similarity_matrix = None
        self.user_similarity_matrix = None
        self.graph = None
        self.trending_scores = {}
        self.seasonal_patterns = {}
        self.last_update = None
        self.collaborative_model = None
        
    def fetch_data_from_backend(self):
        """Fetch all necessary data from backend"""
        try:
            # For demo purposes, create realistic sample data
            # In production, replace with actual API calls to your backend
            
            self.users_data = self._generate_sample_users_data()
            self.content_data = self._generate_sample_content_data()
            self.interactions_data = self._generate_sample_interactions_data()
            
            print(f"Loaded {len(self.users_data)} users, {len(self.content_data)} content items, {len(self.interactions_data)} interactions")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def _generate_sample_users_data(self):
        """Generate sample users data with realistic patterns"""
        np.random.seed(42)  # For reproducibility
        users = []
        
        for user_id in range(1, 1001):
            # Create diverse user types
            user_type = np.random.choice(['casual', 'enthusiast', 'binge_watcher', 'explorer'], p=[0.4, 0.3, 0.2, 0.1])
            
            preferences = {
                'user_type': user_type,
                'genre_weights': {}
            }
            
            # Assign genre preferences based on user type
            if user_type == 'casual':
                preferred_genres = np.random.choice([28, 35, 18], size=2, replace=False)
            elif user_type == 'enthusiast':
                preferred_genres = np.random.choice(list(GENRE_MAP.keys()), size=4, replace=False)
            elif user_type == 'binge_watcher':
                preferred_genres = np.random.choice([18, 10759, 80], size=3, replace=False)
            else:  # explorer
                preferred_genres = np.random.choice(list(GENRE_MAP.keys()), size=6, replace=False)
            
            for genre in preferred_genres:
                preferences['genre_weights'][str(genre)] = np.random.uniform(0.5, 1.0)
            
            users.append({
                'user_id': user_id,
                'preferences': preferences,
                'created_at': pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            })
        
        return pd.DataFrame(users)
    
    def _generate_sample_content_data(self):
        """Generate sample content data with realistic distributions"""
        np.random.seed(42)
        content_items = []
        
        content_types = ['movie', 'tv', 'anime']
        type_probs = [0.5, 0.35, 0.15]
        
        for content_id in range(1, 5001):
            content_type = np.random.choice(content_types, p=type_probs)
            
            # Genre assignment based on content type
            if content_type == 'anime':
                possible_genres = [16, 35, 18, 28, 10759]  # Animation, Comedy, Drama, Action, Action & Adventure
            elif content_type == 'tv':
                possible_genres = [18, 35, 80, 10759, 9648]  # Drama, Comedy, Crime, Action & Adventure, Mystery
            else:  # movie
                possible_genres = list(GENRE_MAP.keys())
            
            num_genres = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            genres = np.random.choice(possible_genres, size=min(num_genres, len(possible_genres)), replace=False).tolist()
            
            # Realistic rating distribution
            if content_type == 'anime':
                rating = np.random.normal(7.5, 1.2)
            else:
                rating = np.random.normal(6.8, 1.5)
            
            rating = np.clip(rating, 1.0, 10.0)
            
            # Release date with realistic distribution
            if content_type == 'anime':
                year = np.random.choice(range(2010, 2025), p=np.linspace(0.1, 0.3, 15))
            else:
                year = np.random.choice(range(2000, 2025), p=np.concatenate([
                    np.linspace(0.01, 0.02, 10),  # 2000-2009
                    np.linspace(0.03, 0.05, 10),  # 2010-2019
                    np.linspace(0.06, 0.08, 5)    # 2020-2024
                ]))
            
            content_items.append({
                'content_id': content_id,
                'title': f'{content_type.title()} {content_id}',
                'genres': genres,
                'overview': f'An engaging {content_type} with {", ".join([GENRE_MAP.get(g, str(g)) for g in genres])} elements.',
                'rating': round(rating, 1),
                'popularity': np.random.exponential(20),
                'content_type': content_type,
                'release_date': pd.Timestamp(f'{year}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}'),
                'runtime': np.random.randint(60, 180) if content_type == 'movie' else np.random.randint(20, 60)
            })
        
        return pd.DataFrame(content_items)
    
    def _generate_sample_interactions_data(self):
        """Generate realistic user interaction patterns"""
        np.random.seed(42)
        interactions = []
        
        interaction_types = ['view', 'like', 'favorite', 'wishlist']
        
        for user_id in range(1, 1001):
            user_data = self.users_data[self.users_data['user_id'] == user_id].iloc[0]
            user_type = user_data['preferences']['user_type']
            
            # Different interaction patterns based on user type
            if user_type == 'casual':
                num_interactions = np.random.randint(5, 25)
            elif user_type == 'enthusiast':
                num_interactions = np.random.randint(50, 150)
            elif user_type == 'binge_watcher':
                num_interactions = np.random.randint(100, 300)
            else:  # explorer
                num_interactions = np.random.randint(75, 200)
            
            # Select content based on user preferences
            preferred_genres = [int(g) for g in user_data['preferences']['genre_weights'].keys()]
            
            for _ in range(num_interactions):
                # Higher probability of interacting with preferred genres
                if np.random.random() < 0.7 and preferred_genres:
                    # Select content with preferred genres
                    genre_filter = self.content_data[
                        self.content_data['genres'].apply(
                            lambda x: any(g in preferred_genres for g in x)
                        )
                    ]
                    if not genre_filter.empty:
                        content_id = np.random.choice(genre_filter['content_id'].values)
                    else:
                        content_id = np.random.choice(self.content_data['content_id'].values)
                else:
                    content_id = np.random.choice(self.content_data['content_id'].values)
                
                # Interaction type probabilities
                if user_type == 'casual':
                    interaction_type = np.random.choice(interaction_types, p=[0.6, 0.2, 0.1, 0.1])
                elif user_type == 'enthusiast':
                    interaction_type = np.random.choice(interaction_types, p=[0.4, 0.3, 0.2, 0.1])
                elif user_type == 'binge_watcher':
                    interaction_type = np.random.choice(interaction_types, p=[0.7, 0.15, 0.1, 0.05])
                else:  # explorer
                    interaction_type = np.random.choice(interaction_types, p=[0.5, 0.25, 0.15, 0.1])
                
                # Rating based on interaction type and content rating
                content_rating = self.content_data[self.content_data['content_id'] == content_id]['rating'].iloc[0]
                
                if interaction_type == 'favorite':
                    rating = max(4, min(5, int(content_rating * 0.8 + np.random.normal(0, 0.5))))
                elif interaction_type == 'like':
                    rating = max(3, min(5, int(content_rating * 0.7 + np.random.normal(0, 0.7))))
                elif interaction_type == 'wishlist':
                    rating = max(3, min(5, int(content_rating * 0.6 + np.random.normal(0, 0.8))))
                else:  # view
                    rating = max(1, min(5, int(content_rating * 0.5 + np.random.normal(0, 1.0))))
                
                # Timestamp
                days_ago = np.random.exponential(30)  # More recent interactions are more likely
                timestamp = datetime.now() - timedelta(days=min(days_ago, 365))
                
                interactions.append({
                    'user_id': user_id,
                    'content_id': content_id,
                    'interaction_type': interaction_type,
                    'rating': rating,
                    'timestamp': timestamp
                })
        
        return pd.DataFrame(interactions)
    
    def build_user_profiles(self):
        """Build comprehensive user profiles with advanced analytics"""
        print("Building advanced user profiles...")
        
        for user_id in self.users_data['user_id'].unique():
            user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
            user_data = self.users_data[self.users_data['user_id'] == user_id].iloc[0]
            
            profile = {
                'user_id': user_id,
                'user_type': user_data['preferences']['user_type'],
                'total_interactions': len(user_interactions),
                'interaction_diversity': len(user_interactions['interaction_type'].unique()),
                'avg_rating': user_interactions['rating'].mean() if not user_interactions.empty else 0,
                'rating_variance': user_interactions['rating'].var() if not user_interactions.empty else 0,
                'genre_preferences': {},
                'content_type_preferences': {},
                'temporal_patterns': {},
                'recency_bias': 0,
                'exploration_vs_exploitation': 0,
                'quality_sensitivity': 0,
                'popularity_bias': 0
            }
            
            if not user_interactions.empty:
                # Analyze genre preferences with decay for older interactions
                for _, interaction in user_interactions.iterrows():
                    content = self.content_data[self.content_data['content_id'] == interaction['content_id']]
                    if not content.empty:
                        content_row = content.iloc[0]
                        
                        # Time decay factor
                        days_ago = (datetime.now() - interaction['timestamp']).days
                        time_decay = np.exp(-days_ago / 90)  # 90-day half-life
                        
                        # Weight calculation
                        weight = self._get_interaction_weight(interaction['interaction_type'], interaction['rating']) * time_decay
                        
                        # Genre preferences
                        for genre in content_row['genres']:
                            genre_name = GENRE_MAP.get(genre, str(genre))
                            profile['genre_preferences'][genre_name] = profile['genre_preferences'].get(genre_name, 0) + weight
                        
                        # Content type preferences
                        content_type = content_row['content_type']
                        profile['content_type_preferences'][content_type] = profile['content_type_preferences'].get(content_type, 0) + weight
                
                # Normalize preferences
                total_genre_weight = sum(profile['genre_preferences'].values())
                if total_genre_weight > 0:
                    profile['genre_preferences'] = {k: v/total_genre_weight for k, v in profile['genre_preferences'].items()}
                
                total_content_weight = sum(profile['content_type_preferences'].values())
                if total_content_weight > 0:
                    profile['content_type_preferences'] = {k: v/total_content_weight for k, v in profile['content_type_preferences'].items()}
                
                # Advanced metrics
                profile['temporal_patterns'] = self._analyze_temporal_patterns(user_interactions)
                profile['recency_bias'] = self._calculate_recency_bias(user_interactions)
                profile['exploration_vs_exploitation'] = self._calculate_exploration_score(user_interactions)
                profile['quality_sensitivity'] = self._calculate_quality_sensitivity(user_interactions)
                profile['popularity_bias'] = self._calculate_popularity_bias(user_interactions)
            
            self.user_profiles[user_id] = profile
    
    def _get_interaction_weight(self, interaction_type, rating):
        """Convert interaction type and rating to weight with sophisticated scaling"""
        base_weights = {'view': 1, 'like': 3, 'favorite': 5, 'wishlist': 2}
        base_weight = base_weights.get(interaction_type, 1)
        
        # Non-linear rating scaling
        rating_weight = (rating / 5.0) ** 1.5 if rating else 1.0
        
        return base_weight * rating_weight
    
    def _analyze_temporal_patterns(self, user_interactions):
        """Advanced temporal pattern analysis"""
        if user_interactions.empty:
            return {}
        
        timestamps = pd.to_datetime(user_interactions['timestamp'])
        
        # Hour preferences
        hour_counts = timestamps.dt.hour.value_counts()
        preferred_hours = hour_counts.nlargest(3).index.tolist()
        
        # Day of week preferences
        dow_counts = timestamps.dt.dayofweek.value_counts()
        preferred_days = dow_counts.nlargest(2).index.tolist()
        
        # Activity intensity over time
        daily_activity = timestamps.dt.date.value_counts().sort_index()
        activity_trend = np.polyfit(range(len(daily_activity)), daily_activity.values, 1)[0] if len(daily_activity) > 1 else 0
        
        return {
            'preferred_hours': preferred_hours,
            'preferred_days': preferred_days,
            'activity_trend': float(activity_trend),
            'session_length': self._calculate_session_metrics(timestamps),
            'weekend_vs_weekday': self._calculate_weekend_preference(timestamps)
        }
    
    def _calculate_session_metrics(self, timestamps):
        """Calculate user session patterns"""
        if len(timestamps) < 2:
            return {'avg_session_length': 1, 'sessions_per_day': 1}
        
        # Sort timestamps
        sorted_timestamps = timestamps.sort_values()
        
        # Identify sessions (gap > 2 hours = new session)
        time_diffs = sorted_timestamps.diff()
        session_breaks = time_diffs > pd.Timedelta(hours=2)
        
        sessions = []
        current_session = [sorted_timestamps.iloc[0]]
        
        for i, (timestamp, is_break) in enumerate(zip(sorted_timestamps.iloc[1:], session_breaks.iloc[1:]), 1):
            if is_break:
                sessions.append(current_session)
                current_session = [timestamp]
            else:
                current_session.append(timestamp)
        
        sessions.append(current_session)
        
        avg_session_length = np.mean([len(session) for session in sessions])
        total_days = (sorted_timestamps.max() - sorted_timestamps.min()).days or 1
        sessions_per_day = len(sessions) / total_days
        
        return {
            'avg_session_length': avg_session_length,
            'sessions_per_day': sessions_per_day
        }
    
    def _calculate_weekend_preference(self, timestamps):
        """Calculate weekend vs weekday preference"""
        weekday_count = sum(timestamps.dt.dayofweek < 5)
        weekend_count = sum(timestamps.dt.dayofweek >= 5)
        
        total = weekday_count + weekend_count
        if total == 0:
            return 0.5
        
        return weekend_count / total
    
    def _calculate_recency_bias(self, user_interactions):
        """Calculate how much user prefers recent content"""
        if user_interactions.empty:
            return 0.5
        
        recent_threshold = datetime.now() - timedelta(days=30)
        recent_interactions = user_interactions[user_interactions['timestamp'] > recent_threshold]
        
        return len(recent_interactions) / len(user_interactions)
    
    def _calculate_exploration_score(self, user_interactions):
        """Calculate user's exploration vs exploitation tendency"""
        if user_interactions.empty:
            return 0.5
        
        content_ids = user_interactions['content_id'].unique()
        total_interactions = len(user_interactions)
        unique_content = len(content_ids)
        
        # Higher score = more exploration
        exploration_score = unique_content / total_interactions
        
        # Consider genre diversity
        content_genres = []
        for content_id in content_ids:
            content = self.content_data[self.content_data['content_id'] == content_id]
            if not content.empty:
                content_genres.extend(content.iloc[0]['genres'])
        
        unique_genres = len(set(content_genres))
        max_possible_genres = len(GENRE_MAP)
        genre_diversity = unique_genres / max_possible_genres
        
        return (exploration_score + genre_diversity) / 2
    
    def _calculate_quality_sensitivity(self, user_interactions):
        """Calculate how sensitive user is to content quality"""
        if user_interactions.empty:
            return 0.5
        
        # Correlation between content rating and user rating
        user_ratings = []
        content_ratings = []
        
        for _, interaction in user_interactions.iterrows():
            content = self.content_data[self.content_data['content_id'] == interaction['content_id']]
            if not content.empty and interaction['rating'] is not None:
                user_ratings.append(interaction['rating'])
                content_ratings.append(content.iloc[0]['rating'])
        
        if len(user_ratings) < 2:
            return 0.5
        
        correlation = np.corrcoef(user_ratings, content_ratings)[0, 1]
        return (correlation + 1) / 2  # Normalize to 0-1
    
    def _calculate_popularity_bias(self, user_interactions):
        """Calculate user's bias toward popular content"""
        if user_interactions.empty:
            return 0.5
        
        popularity_scores = []
        for _, interaction in user_interactions.iterrows():
            content = self.content_data[self.content_data['content_id'] == interaction['content_id']]
            if not content.empty:
                popularity_scores.append(content.iloc[0]['popularity'])
        
        if not popularity_scores:
            return 0.5
        
        user_avg_popularity = np.mean(popularity_scores)
        global_avg_popularity = self.content_data['popularity'].mean()
        
        # Normalize relative to global average
        return min(1.0, user_avg_popularity / global_avg_popularity) if global_avg_popularity > 0 else 0.5
    
    def build_content_features(self):
        """Build sophisticated content feature matrix"""
        print("Building advanced content features...")
        
        # Text features using TF-IDF with advanced parameters
        text_features = self.content_data['title'] + ' ' + self.content_data['overview'].fillna('')
        
        self.vectorizers['tfidf'] = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        tfidf_matrix = self.vectorizers['tfidf'].fit_transform(text_features)
        
        # Genre features with TF-IDF weighting
        all_genres = set()
        for genres in self.content_data['genres']:
            all_genres.update([GENRE_MAP.get(g, str(g)) for g in genres])
        
        genre_features = np.zeros((len(self.content_data), len(all_genres)))
        genre_list = list(all_genres)
        
        for i, genres in enumerate(self.content_data['genres']):
            for genre in genres:
                genre_name = GENRE_MAP.get(genre, str(genre))
                if genre_name in genre_list:
                    # Weight by inverse frequency of genre
                    genre_frequency = sum(1 for g_list in self.content_data['genres'] if genre in g_list)
                    weight = np.log(len(self.content_data) / genre_frequency) if genre_frequency > 0 else 1
                    genre_features[i, genre_list.index(genre_name)] = weight
        
        # Numerical features with sophisticated scaling
        numerical_features = self.content_data[['rating', 'popularity', 'runtime']].fillna(0)
        
        # Apply different scaling strategies
        self.scalers['rating'] = MinMaxScaler()
        self.scalers['popularity'] = StandardScaler()
        self.scalers['runtime'] = StandardScaler()
        
        scaled_rating = self.scalers['rating'].fit_transform(numerical_features[['rating']])
        scaled_popularity = self.scalers['popularity'].fit_transform(numerical_features[['popularity']])
        scaled_runtime = self.scalers['runtime'].fit_transform(numerical_features[['runtime']])
        
        scaled_numerical = np.hstack([scaled_rating, scaled_popularity, scaled_runtime])
        
        # Content type features
        content_types = pd.get_dummies(self.content_data['content_type'])
        
        # Temporal features
        release_years = pd.to_datetime(self.content_data['release_date']).dt.year.fillna(2000)
        current_year = datetime.now().year
        
        # Multiple temporal representations
        year_recency = ((current_year - release_years) / (current_year - 1900)).values.reshape(-1, 1)
        decade_features = pd.get_dummies((release_years // 10) * 10)
        
        # Quality indicators
        rating_percentile = self.content_data['rating'].rank(pct=True).values.reshape(-1, 1)
        popularity_percentile = self.content_data['popularity'].rank(pct=True).values.reshape(-1, 1)
        
        # Combine all features
        from scipy.sparse import hstack, csr_matrix
        
        content_features = hstack([
            tfidf_matrix,
            csr_matrix(genre_features),
            csr_matrix(scaled_numerical),
            csr_matrix(content_types.values),
            csr_matrix(year_recency),
            csr_matrix(decade_features.values),
            csr_matrix(rating_percentile),
            csr_matrix(popularity_percentile)
        ])
        
        self.content_features['matrix'] = content_features
        self.content_features['genre_list'] = genre_list
        
        # Calculate content similarity matrix using multiple metrics
        print("Computing content similarity matrix...")
        self.content_similarity_matrix = cosine_similarity(content_features)
        
        # Add penalty for different content types
        for i in range(len(self.content_data)):
            for j in range(len(self.content_data)):
                if self.content_data.iloc[i]['content_type'] != self.content_data.iloc[j]['content_type']:
                    self.content_similarity_matrix[i, j] *= 0.7  # Reduce similarity across types
    
    def build_interaction_matrix(self):
        """Build sophisticated user-item interaction matrix"""
        print("Building advanced interaction matrix...")
        
        # Create weighted interaction matrix with temporal decay
        interaction_weights = self.interactions_data.copy()
        
        # Calculate weights with temporal decay
        current_time = datetime.now()
        interaction_weights['days_ago'] = (current_time - pd.to_datetime(interaction_weights['timestamp'])).dt.days
        interaction_weights['time_decay'] = np.exp(-interaction_weights['days_ago'] / 60)  # 60-day half-life
        
        interaction_weights['weight'] = interaction_weights.apply(
            lambda x: self._get_interaction_weight(x['interaction_type'], x['rating']) * x['time_decay'], 
            axis=1
        )
        
        # Aggregate weights for multiple interactions
        aggregated = interaction_weights.groupby(['user_id', 'content_id'])['weight'].sum().reset_index()
        
        # Create matrix
        user_ids = sorted(aggregated['user_id'].unique())
        content_ids = sorted(aggregated['content_id'].unique())
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        content_to_idx = {content_id: idx for idx, content_id in enumerate(content_ids)}
        
        matrix = np.zeros((len(user_ids), len(content_ids)))
        
        for _, row in aggregated.iterrows():
            user_idx = user_to_idx[row['user_id']]
            content_idx = content_to_idx[row['content_id']]
            matrix[user_idx, content_idx] = row['weight']
        
        self.interaction_matrix = {
            'matrix': csr_matrix(matrix),
            'user_to_idx': user_to_idx,
            'content_to_idx': content_to_idx,
            'idx_to_user': {idx: user_id for user_id, idx in user_to_idx.items()},
            'idx_to_content': {idx: content_id for content_id, idx in content_to_idx.items()}
        }
        
        # Build user similarity matrix
        self.user_similarity_matrix = cosine_similarity(matrix)
    
    def train_collaborative_filtering(self):
        """Train custom collaborative filtering models"""
        print("Training collaborative filtering models...")
        
        # Prepare data for custom collaborative filtering
        ratings_data = []
        for _, row in self.interactions_data.iterrows():
            weight = self._get_interaction_weight(row['interaction_type'], row['rating'])
            ratings_data.append({
                'user_id': row['user_id'],
                'item_id': row['content_id'],
                'rating': weight
            })
        
        ratings_df = pd.DataFrame(ratings_data)
        
        # Train custom collaborative filtering model
        self.collaborative_model = CustomCollaborativeFiltering(
            n_factors=64,
            learning_rate=0.01,
            regularization=0.01,
            epochs=30
        )
        
        self.collaborative_model.fit(ratings_df)
        
        # Train SVD using scikit-learn
        if self.interaction_matrix is not None:
            self.models['svd'] = TruncatedSVD(n_components=50, random_state=42)
            self.models['svd'].fit(self.interaction_matrix['matrix'])
            
            # Train NMF
            self.models['nmf'] = NMF(n_components=30, random_state=42, max_iter=200)
            self.models['nmf'].fit(self.interaction_matrix['matrix'])
    
    def train_matrix_factorization(self):
        """Train matrix factorization using Implicit library"""
        print("Training matrix factorization models...")
        
        if self.interaction_matrix is None:
            return
        
        try:
            # Train Alternating Least Squares
            self.models['als'] = implicit.als.AlternatingLeastSquares(
                factors=64,
                regularization=0.01,
                iterations=30,
                alpha=1.0
            )
            
            # Implicit expects item-user matrix (transposed)
            item_user_matrix = self.interaction_matrix['matrix'].T.astype(np.float32)
            self.models['als'].fit(item_user_matrix)
            
            # Train Bayesian Personalized Ranking
            self.models['bpr'] = implicit.bpr.BayesianPersonalizedRanking(
                factors=64,
                learning_rate=0.01,
                regularization=0.01,
                iterations=100
            )
            
            self.models['bpr'].fit(item_user_matrix)
            
        except Exception as e:
            print(f"Error training matrix factorization models: {e}")
    
    def train_ensemble_models(self):
        """Train ensemble models for advanced recommendations"""
        print("Training ensemble models...")
        
        # Prepare features for ensemble
        features = []
        targets = []
        
        for _, row in self.interactions_data.iterrows():
            user_id = row['user_id']
            content_id = row['content_id']
            
            if user_id in self.user_profiles:
                # User features
                profile = self.user_profiles[user_id]
                user_feat = [
                    profile['total_interactions'],
                    profile['interaction_diversity'],
                    profile['avg_rating'],
                    profile['recency_bias'],
                    profile['exploration_vs_exploitation'],
                    profile['quality_sensitivity'],
                    profile['popularity_bias']
                ]
                
                # Content features
                content_row = self.content_data[self.content_data['content_id'] == content_id]
                if not content_row.empty:
                    content_row = content_row.iloc[0]
                    content_feat = [
                        content_row['rating'],
                        content_row['popularity'],
                        content_row['runtime'],
                        len(content_row['genres']),
                        (datetime.now() - content_row['release_date']).days / 365.25  # Age in years
                    ]
                    
                    # User-content interaction features
                    genre_match = len(set(content_row['genres']) & 
                                    set([k for k, v in profile['genre_preferences'].items() 
                                         if k in GENRE_MAP.values() and v > 0.1])) / max(1, len(content_row['genres']))
                    
                    content_type_pref = profile['content_type_preferences'].get(content_row['content_type'], 0)
                    
                    interaction_feat = [genre_match, content_type_pref]
                    
                    combined_features = user_feat + content_feat + interaction_feat
                    features.append(combined_features)
                    targets.append(self._get_interaction_weight(row['interaction_type'], row['rating']))
        
        if features:
            X = np.array(features)
            y = np.array(targets)
            
            # Train multiple ensemble models
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
            self.models['random_forest'].fit(X, y)
            
            self.models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
            self.models['gradient_boosting'].fit(X, y)
            
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )
            self.models['lightgbm'].fit(X, y)
            
            # Neural network with advanced architecture
            self.models['neural_network'] = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                random_state=42,
                max_iter=300
            )
            self.models['neural_network'].fit(X, y)
    
    def build_graph_network(self):
        """Build sophisticated user-content graph network"""
        print("Building advanced graph network...")
        
        self.graph = nx.Graph()
        
        # Add user nodes with attributes
        for user_id in self.users_data['user_id'].unique():
            profile = self.user_profiles.get(user_id, {})
            self.graph.add_node(f"user_{user_id}", 
                              type='user',
                              user_type=profile.get('user_type', 'casual'),
                              total_interactions=profile.get('total_interactions', 0))
        
        # Add content nodes with attributes
        for _, content in self.content_data.iterrows():
            self.graph.add_node(f"content_{content['content_id']}", 
                              type='content',
                              content_type=content['content_type'],
                              rating=content['rating'],
                              popularity=content['popularity'])
        
        # Add user-content edges with sophisticated weights
        for _, row in self.interactions_data.iterrows():
            # Time decay
            days_ago = (datetime.now() - row['timestamp']).days
            time_decay = np.exp(-days_ago / 90)
            
            weight = self._get_interaction_weight(row['interaction_type'], row['rating']) * time_decay
            
            self.graph.add_edge(
                f"user_{row['user_id']}", 
                f"content_{row['content_id']}", 
                weight=weight,
                interaction_type=row['interaction_type']
            )
        
        # Add content-content edges based on enhanced similarity
        print("Adding content similarity edges...")
        for i, content_id1 in enumerate(self.content_data['content_id']):
            for j, content_id2 in enumerate(self.content_data['content_id']):
                if i < j and self.content_similarity_matrix is not None:
                    similarity = self.content_similarity_matrix[i, j]
                    if similarity > 0.6:  # High similarity threshold
                        self.graph.add_edge(
                            f"content_{content_id1}",
                            f"content_{content_id2}",
                            weight=similarity,
                            type='similarity'
                        )
        
        # Add user-user edges based on similar preferences
        print("Adding user similarity edges...")
        for user_id1 in list(self.user_profiles.keys())[:500]:  # Limit for performance
            for user_id2 in list(self.user_profiles.keys())[:500]:
                if user_id1 < user_id2:
                    similarity = self._calculate_user_similarity_advanced(user_id1, user_id2)
                    if similarity > 0.3:
                        self.graph.add_edge(
                            f"user_{user_id1}",
                            f"user_{user_id2}",
                            weight=similarity,
                            type='user_similarity'
                        )
    
    def _calculate_user_similarity_advanced(self, user_id1, user_id2):
        """Calculate advanced user similarity"""
        profile1 = self.user_profiles.get(user_id1, {})
        profile2 = self.user_profiles.get(user_id2, {})
        
        if not profile1 or not profile2:
            return 0
        
        # Genre preference similarity
        genres1 = profile1.get('genre_preferences', {})
        genres2 = profile2.get('genre_preferences', {})
        
        all_genres = set(genres1.keys()) | set(genres2.keys())
        if not all_genres:
            return 0
        
        vec1 = [genres1.get(genre, 0) for genre in all_genres]
        vec2 = [genres2.get(genre, 0) for genre in all_genres]
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        genre_similarity = dot_product / (magnitude1 * magnitude2)
        
        # Behavioral similarity
        behavior_similarity = 1 - abs(profile1.get('exploration_vs_exploitation', 0.5) - 
                                    profile2.get('exploration_vs_exploitation', 0.5))
        
        # Combined similarity
        return (genre_similarity + behavior_similarity) / 2
    
    def analyze_trending_patterns(self):
        """Advanced trending pattern analysis"""
        print("Analyzing trending patterns...")
        
        # Time-weighted trending analysis
        current_time = datetime.now()
        trending_weights = {}
        
        for timeframe, weight in [(1, 3.0), (7, 2.0), (30, 1.0)]:  # 1 day, 1 week, 1 month
            recent_interactions = self.interactions_data[
                self.interactions_data['timestamp'] > current_time - timedelta(days=timeframe)
            ]
            
            if not recent_interactions.empty:
                trending_scores = recent_interactions.groupby('content_id').agg({
                    'user_id': 'nunique',  # Unique users
                    'interaction_type': 'count',  # Total interactions
                    'rating': 'mean'  # Average rating
                }).fillna(0)
                
                # Calculate momentum (rate of growth)
                for content_id in trending_scores.index:
                    user_count = trending_scores.loc[content_id, 'user_id']
                    interaction_count = trending_scores.loc[content_id, 'interaction_type']
                    avg_rating = trending_scores.loc[content_id, 'rating']
                    
                    # Multi-factor trending score
                    score = (user_count * 0.4 + interaction_count * 0.3 + avg_rating * 0.3) * weight
                    
                    if content_id not in trending_weights:
                        trending_weights[content_id] = 0
                    trending_weights[content_id] += score
        
        # Normalize trending scores
        if trending_weights:
            max_score = max(trending_weights.values())
            self.trending_scores = {k: v/max_score for k, v in trending_weights.items()}
        
        # Seasonal patterns with advanced analysis
        self._analyze_seasonal_patterns()
    
    def _analyze_seasonal_patterns(self):
        """Analyze seasonal viewing patterns"""
        interactions_with_season = self.interactions_data.copy()
        interactions_with_season['month'] = pd.to_datetime(interactions_with_season['timestamp']).dt.month
        interactions_with_season['season'] = interactions_with_season['month'].apply(self._get_season)
        interactions_with_season['year'] = pd.to_datetime(interactions_with_season['timestamp']).dt.year
        
        # Content preferences by season
        seasonal_preferences = {}
        
        for season in ['winter', 'spring', 'summer', 'autumn']:
            season_data = interactions_with_season[interactions_with_season['season'] == season]
            
            if not season_data.empty:
                # Analyze genre preferences by season
                genre_scores = defaultdict(float)
                
                for _, interaction in season_data.iterrows():
                    content = self.content_data[self.content_data['content_id'] == interaction['content_id']]
                    if not content.empty:
                        content_row = content.iloc[0]
                        weight = self._get_interaction_weight(interaction['interaction_type'], interaction['rating'])
                        
                        for genre in content_row['genres']:
                            genre_name = GENRE_MAP.get(genre, str(genre))
                            genre_scores[genre_name] += weight
                
                # Content type preferences
                content_type_scores = season_data.groupby('content_id').size()
                seasonal_content = content_type_scores.nlargest(50).index.tolist()
                
                seasonal_preferences[season] = {
                    'content_ids': seasonal_content,
                    'genre_preferences': dict(genre_scores)
                }
        
        self.seasonal_patterns = seasonal_preferences
    
    def _get_season(self, month):
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def get_recommendations(self, user_id, num_recommendations=20):
        """Generate comprehensive recommendations using all advanced algorithms"""
        print(f"Generating recommendations for user {user_id}...")
        
        recommendations = {}
        
        # 1. Content-based recommendations
        recommendations['content_based'] = self._get_content_based_recommendations(user_id, num_recommendations // 6)
        
        # 2. Collaborative filtering recommendations
        recommendations['collaborative'] = self._get_collaborative_recommendations(user_id, num_recommendations // 6)
        
        # 3. Matrix factorization recommendations
        recommendations['matrix_factorization'] = self._get_matrix_factorization_recommendations(user_id, num_recommendations // 6)
        
        # 4. Graph-based recommendations
        recommendations['graph_based'] = self._get_graph_recommendations(user_id, num_recommendations // 6)
        
        # 5. Trending recommendations
        recommendations['trending'] = self._get_trending_recommendations(user_id, num_recommendations // 6)
        
        # 6. Seasonal recommendations
        recommendations['seasonal'] = self._get_seasonal_recommendations(user_id, num_recommendations // 6)
        
        # 7. Ensemble hybrid recommendations
        recommendations['ensemble'] = self._get_ensemble_recommendations(user_id, num_recommendations // 3)
        
        # Combine and rank all recommendations
        final_recommendations = self._combine_recommendations_advanced(user_id, recommendations, num_recommendations)
        
        return final_recommendations
    
    def _get_content_based_recommendations(self, user_id, num_recs):
        """Advanced content-based filtering recommendations"""
        if user_id not in self.user_profiles or self.content_features.get('matrix') is None:
            return []
        
        user_profile = self.user_profiles[user_id]
        user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
        
        if user_interactions.empty:
            return []
        
        # Create sophisticated user preference vector
        user_vector = np.zeros(self.content_features['matrix'].shape[1])
        total_weight = 0
        
        for _, interaction in user_interactions.iterrows():
            content_idx = self.content_data[self.content_data['content_id'] == interaction['content_id']].index
            if not content_idx.empty:
                content_idx = content_idx[0]
                
                # Time decay
                days_ago = (datetime.now() - interaction['timestamp']).days
                time_decay = np.exp(-days_ago / 90)
                
                weight = self._get_interaction_weight(interaction['interaction_type'], interaction['rating']) * time_decay
                user_vector += self.content_features['matrix'][content_idx].toarray()[0] * weight
                total_weight += weight
        
        if total_weight > 0:
            user_vector = user_vector / total_weight
        
        # Calculate similarities with user profile boost
        similarities = cosine_similarity([user_vector], self.content_features['matrix'])[0]
        
        # Apply user-specific boosts
        for i, content_row in self.content_data.iterrows():
            # Boost based on user preferences
            genre_boost = 0
            for genre in content_row['genres']:
                genre_name = GENRE_MAP.get(genre, str(genre))
                genre_boost += user_profile['genre_preferences'].get(genre_name, 0)
            
            content_type_boost = user_profile['content_type_preferences'].get(content_row['content_type'], 0)
            
            # Quality sensitivity boost
            if user_profile['quality_sensitivity'] > 0.7 and content_row['rating'] > 7:
                similarities[i] *= 1.2
            
            # Popularity bias adjustment
            popularity_percentile = self.content_data['popularity'].rank(pct=True).iloc[i]
            if user_profile['popularity_bias'] > 0.7:
                similarities[i] *= (1 + popularity_percentile * 0.3)
            
            similarities[i] *= (1 + genre_boost * 0.5 + content_type_boost * 0.3)
        
        # Get top recommendations
        content_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        interacted_content = set(user_interactions['content_id'])
        
        for idx in content_indices:
            if len(recommendations) >= num_recs:
                break
            
            content_id = self.content_data.iloc[idx]['content_id']
            if content_id not in interacted_content:
                recommendations.append({
                    'content_id': content_id,
                    'score': float(similarities[idx]),
                    'reason': 'content_based'
                })
        
        return recommendations
    
    def _get_collaborative_recommendations(self, user_id, num_recs):
        """Advanced collaborative filtering recommendations"""
        recommendations = []
        
        if self.collaborative_model is None:
            return recommendations
        
        try:
            # Get recommendations from custom collaborative filtering
            cf_recommendations = self.collaborative_model.recommend(user_id, num_recs * 2)
            
            user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
            interacted_content = set(user_interactions['content_id'])
            
            for content_id, score in cf_recommendations:
                if len(recommendations) >= num_recs:
                    break
                
                if content_id not in interacted_content:
                    recommendations.append({
                        'content_id': content_id,
                        'score': float(score),
                        'reason': 'collaborative'
                    })
        except Exception as e:
            print(f"Error in collaborative filtering: {e}")
        
        return recommendations
    
    def _get_matrix_factorization_recommendations(self, user_id, num_recs):
        """Advanced matrix factorization recommendations"""
        recommendations = []
        
        if 'als' not in self.models or self.interaction_matrix is None:
            return recommendations
        
        user_to_idx = self.interaction_matrix['user_to_idx']
        idx_to_content = self.interaction_matrix['idx_to_content']
        
        if user_id not in user_to_idx:
            return recommendations
        
        user_idx = user_to_idx[user_id]
        
        try:
            # Get recommendations from ALS model
            item_ids, scores = self.models['als'].recommend(
                user_idx, 
                self.interaction_matrix['matrix'][user_idx],
                N=num_recs * 2,
                filter_already_liked_items=True
            )
            
            for item_idx, score in zip(item_ids, scores):
                if len(recommendations) >= num_recs:
                    break
                
                if item_idx in idx_to_content:
                    content_id = idx_to_content[item_idx]
                    recommendations.append({
                        'content_id': content_id,
                        'score': float(score),
                        'reason': 'matrix_factorization'
                    })
        except Exception as e:
            print(f"Error in matrix factorization: {e}")
        
        return recommendations
    
    def _get_graph_recommendations(self, user_id, num_recs):
        """Advanced graph-based recommendations"""
        recommendations = []
        
        if self.graph is None:
            return recommendations
        
        user_node = f"user_{user_id}"
        if user_node not in self.graph:
            return recommendations
        
        try:
            # Use multiple graph algorithms
            
            # 1. Personalized PageRank
            pagerank_scores = nx.pagerank(self.graph, personalization={user_node: 1})
            
            # 2. Random walk with restart
            # Implement custom random walk for diversity
            random_walk_scores = self._random_walk_with_restart(user_node, restart_prob=0.15, steps=1000)
            
            # Combine scores
            combined_scores = {}
            for node in pagerank_scores:
                if node.startswith('content_'):
                    content_id = int(node.split('_')[1])
                    combined_score = (pagerank_scores[node] * 0.7 + 
                                    random_walk_scores.get(node, 0) * 0.3)
                    combined_scores[content_id] = combined_score
            
            # Filter and sort
            user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
            interacted_content = set(user_interactions['content_id'])
            
            sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            for content_id, score in sorted_scores:
                if len(recommendations) >= num_recs:
                    break
                
                if content_id not in interacted_content:
                    recommendations.append({
                        'content_id': content_id,
                        'score': float(score),
                        'reason': 'graph_based'
                    })
        except Exception as e:
            print(f"Error in graph recommendations: {e}")
        
        return recommendations
    
    def _random_walk_with_restart(self, start_node, restart_prob=0.15, steps=1000):
        """Implement random walk with restart for graph recommendations"""
        if start_node not in self.graph:
            return {}
        
        visit_counts = defaultdict(int)
        current_node = start_node
        
        for _ in range(steps):
            visit_counts[current_node] += 1
            
            if np.random.random() < restart_prob:
                current_node = start_node
            else:
                neighbors = list(self.graph.neighbors(current_node))
                if neighbors:
                    # Weight by edge weights
                    weights = [self.graph[current_node][neighbor].get('weight', 1) for neighbor in neighbors]
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    current_node = np.random.choice(neighbors, p=weights)
                else:
                    current_node = start_node
        
        # Normalize visit counts
        total_visits = sum(visit_counts.values())
        return {node: count/total_visits for node, count in visit_counts.items()}
    
    def _get_trending_recommendations(self, user_id, num_recs):
        """Advanced trending content recommendations"""
        recommendations = []
        
        if not self.trending_scores:
            return recommendations
        
        user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
        interacted_content = set(user_interactions['content_id'])
        
        # Filter by user preferences if available
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            
            # Boost trending content that matches user preferences
            boosted_scores = {}
            for content_id, score in self.trending_scores.items():
                content = self.content_data[self.content_data['content_id'] == content_id]
                if not content.empty:
                    content_row = content.iloc[0]
                    
                    # Genre preference boost
                    genre_boost = 0
                    for genre in content_row['genres']:
                        genre_name = GENRE_MAP.get(genre, str(genre))
                        genre_boost += profile['genre_preferences'].get(genre_name, 0)
                    
                    # Content type boost
                    content_type_boost = profile['content_type_preferences'].get(content_row['content_type'], 0)
                    
                    boosted_score = score * (1 + genre_boost * 0.5 + content_type_boost * 0.3)
                    boosted_scores[content_id] = boosted_score
            
            trending_items = sorted(boosted_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            trending_items = sorted(self.trending_scores.items(), key=lambda x: x[1], reverse=True)
        
        for content_id, score in trending_items:
            if len(recommendations) >= num_recs:
                break
            
            if content_id not in interacted_content:
                recommendations.append({
                    'content_id': content_id,
                    'score': float(score),
                    'reason': 'trending'
                })
        
        return recommendations
    
    def _get_seasonal_recommendations(self, user_id, num_recs):
        """Advanced seasonal content recommendations"""
        recommendations = []
        
        if not self.seasonal_patterns:
            return recommendations
        
        current_season = self._get_season(datetime.now().month)
        seasonal_data = self.seasonal_patterns.get(current_season, {})
        
        if not seasonal_data:
            return recommendations
        
        user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
        interacted_content = set(user_interactions['content_id'])
        
        seasonal_content = seasonal_data.get('content_ids', [])
        seasonal_genres = seasonal_data.get('genre_preferences', {})
        
        # Score seasonal content
        content_scores = []
        
        for content_id in seasonal_content:
            if content_id not in interacted_content:
                content = self.content_data[self.content_data['content_id'] == content_id]
                if not content.empty:
                    content_row = content.iloc[0]
                    
                    # Base seasonal score
                    score = 1.0
                    
                    # Boost by seasonal genre preferences
                    for genre in content_row['genres']:
                        genre_name = GENRE_MAP.get(genre, str(genre))
                        score += seasonal_genres.get(genre_name, 0) * 0.1
                    
                    # User preference boost if available
                    if user_id in self.user_profiles:
                        profile = self.user_profiles[user_id]
                        for genre in content_row['genres']:
                            genre_name = GENRE_MAP.get(genre, str(genre))
                            score += profile['genre_preferences'].get(genre_name, 0) * 0.2
                    
                    content_scores.append((content_id, score))
        
        # Sort by score
        content_scores.sort(key=lambda x: x[1], reverse=True)
        
        for content_id, score in content_scores[:num_recs]:
            recommendations.append({
                'content_id': content_id,
                'score': float(score),
                'reason': 'seasonal'
            })
        
        return recommendations
    
    def _get_ensemble_recommendations(self, user_id, num_recs):
        """Advanced ensemble model recommendations"""
        recommendations = []
        
        if 'random_forest' not in self.models or user_id not in self.user_profiles:
            return recommendations
        
        user_profile = self.user_profiles[user_id]
        user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
        interacted_content = set(user_interactions['content_id'])
        
        # Prepare user features
        user_feat = [
            user_profile['total_interactions'],
            user_profile['interaction_diversity'],
            user_profile['avg_rating'],
            user_profile['recency_bias'],
            user_profile['exploration_vs_exploitation'],
            user_profile['quality_sensitivity'],
            user_profile['popularity_bias']
        ]
        
        # Predict for all uninteracted content
        content_scores = []
        for _, content_row in self.content_data.iterrows():
            content_id = content_row['content_id']
            
            if content_id not in interacted_content:
                content_feat = [
                    content_row['rating'],
                    content_row['popularity'],
                    content_row['runtime'],
                    len(content_row['genres']),
                    (datetime.now() - content_row['release_date']).days / 365.25
                ]
                
                # Enhanced interaction features
                genre_match = len(set(content_row['genres']) & 
                                set([k for k, v in user_profile['genre_preferences'].items() 
                                     if k in [GENRE_MAP.get(g) for g in content_row['genres']] and v > 0.1])) / max(1, len(content_row['genres']))
                
                content_type_pref = user_profile['content_type_preferences'].get(content_row['content_type'], 0)
                
                interaction_feat = [genre_match, content_type_pref]
                
                combined_features = np.array([user_feat + content_feat + interaction_feat])
                
                # Get predictions from all ensemble models
                scores = []
                model_weights = {
                    'random_forest': 0.3,
                    'gradient_boosting': 0.25,
                    'lightgbm': 0.25,
                    'neural_network': 0.2
                }
                
                weighted_score = 0
                total_weight = 0
                
                for model_name, weight in model_weights.items():
                    if model_name in self.models:
                        try:
                            score = self.models[model_name].predict(combined_features)[0]
                            weighted_score += score * weight
                            total_weight += weight
                        except Exception as e:
                            print(f"Error with {model_name}: {e}")
                
                if total_weight > 0:
                    final_score = weighted_score / total_weight
                    content_scores.append((content_id, final_score))
        
        # Sort by score
        content_scores.sort(key=lambda x: x[1], reverse=True)
        
        for content_id, score in content_scores[:num_recs]:
            recommendations.append({
                'content_id': content_id,
                'score': float(score),
                'reason': 'ensemble'
            })
        
        return recommendations
    
    def _combine_recommendations_advanced(self, user_id, recommendations_dict, num_final):
        """Advanced recommendation combination with sophisticated ranking"""
        all_recommendations = {}
        
        # Collect all recommendations
        for reason, recs in recommendations_dict.items():
            for rec in recs:
                content_id = rec['content_id']
                score = rec['score']
                
                if content_id not in all_recommendations:
                    all_recommendations[content_id] = {
                        'scores': {},
                        'reasons': [],
                        'final_score': 0,
                        'diversity_score': 0,
                        'novelty_score': 0
                    }
                
                all_recommendations[content_id]['scores'][reason] = score
                all_recommendations[content_id]['reasons'].append(reason)
        
        # Advanced scoring with multiple factors
        user_profile = self.user_profiles.get(user_id, {})
        
        # Dynamic weights based on user profile
        base_weights = {
            'content_based': 0.2,
            'collaborative': 0.2,
            'matrix_factorization': 0.15,
            'graph_based': 0.15,
            'trending': 0.1,
            'seasonal': 0.05,
            'ensemble': 0.15
        }
        
        # Adjust weights based on user characteristics
        if user_profile.get('exploration_vs_exploitation', 0.5) > 0.7:
            base_weights['trending'] *= 1.5
            base_weights['seasonal'] *= 1.3
        
        if user_profile.get('quality_sensitivity', 0.5) > 0.7:
            base_weights['content_based'] *= 1.3
            base_weights['ensemble'] *= 1.2
        
        # Calculate final scores
        for content_id, data in all_recommendations.items():
            final_score = 0
            total_weight = 0
            
            # Base score from algorithms
            for reason, score in data['scores'].items():
                weight = base_weights.get(reason, 0.1)
                final_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                final_score /= total_weight
            
            # Diversity bonus (multiple algorithms)
            diversity_bonus = len(data['reasons']) * 0.05
            final_score += diversity_bonus
            data['diversity_score'] = diversity_bonus
            
            # Novelty bonus (content the user hasn't seen similar to)
            novelty_bonus = self._calculate_novelty_bonus(user_id, content_id)
            final_score += novelty_bonus
            data['novelty_score'] = novelty_bonus
            
            # User-specific boosts
            content_boost = self._calculate_user_content_boost(user_id, content_id)
            final_score *= content_boost
            
            data['final_score'] = final_score
        
        # Sort by final score
        sorted_recommendations = sorted(
            all_recommendations.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )
        
        # Apply diversity filtering to avoid too similar recommendations
        final_recs = self._apply_diversity_filtering(sorted_recommendations, num_final)
        
        return final_recs
    
    def _calculate_novelty_bonus(self, user_id, content_id):
        """Calculate novelty bonus for unexplored content"""
        user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
        
        if user_interactions.empty:
            return 0.1
        
        # Get content info
        content = self.content_data[self.content_data['content_id'] == content_id]
        if content.empty:
            return 0
        
        content_row = content.iloc[0]
        content_genres = set(content_row['genres'])
        content_type = content_row['content_type']
        
        # Check if user has interacted with similar content
        similar_interactions = 0
        total_interactions = len(user_interactions)
        
        for _, interaction in user_interactions.iterrows():
            interacted_content = self.content_data[self.content_data['content_id'] == interaction['content_id']]
            if not interacted_content.empty:
                interacted_row = interacted_content.iloc[0]
                
                # Check genre overlap
                genre_overlap = len(content_genres & set(interacted_row['genres'])) / len(content_genres | set(interacted_row['genres']))
                
                # Check content type
                same_type = content_type == interacted_row['content_type']
                
                if genre_overlap > 0.5 or same_type:
                    similar_interactions += 1
        
        # Higher novelty bonus for less explored content
        novelty_ratio = 1 - (similar_interactions / total_interactions)
        return novelty_ratio * 0.15
    
    def _calculate_user_content_boost(self, user_id, content_id):
        """Calculate user-specific content boost"""
        if user_id not in self.user_profiles:
            return 1.0
        
        profile = self.user_profiles[user_id]
        content = self.content_data[self.content_data['content_id'] == content_id]
        
        if content.empty:
            return 1.0
        
        content_row = content.iloc[0]
        boost = 1.0
        
        # Quality boost
        if profile.get('quality_sensitivity', 0.5) > 0.6 and content_row['rating'] > 8:
            boost *= 1.2
        
        # Popularity boost
        popularity_percentile = self.content_data['popularity'].rank(pct=True).loc[content.index[0]]
        if profile.get('popularity_bias', 0.5) > 0.6:
            boost *= (1 + popularity_percentile * 0.3)
        
        # Recency boost for new content
        if profile.get('recency_bias', 0.5) > 0.6:
            content_age_days = (datetime.now() - content_row['release_date']).days
            if content_age_days < 365:  # Less than a year old
                boost *= 1.15
        
        return boost
    
    def _apply_diversity_filtering(self, sorted_recommendations, num_final):
        """Apply diversity filtering to recommendations"""
        final_recs = []
        selected_content_ids = []
        
        for content_id, data in sorted_recommendations:
            if len(final_recs) >= num_final:
                break
            
            # Check diversity with already selected content
            should_include = True
            
            if len(selected_content_ids) > 0:
                content = self.content_data[self.content_data['content_id'] == content_id]
                if not content.empty:
                    content_row = content.iloc[0]
                    
                    # Check similarity with already selected content
                    for selected_id in selected_content_ids[-5:]:  # Check last 5 selections
                        selected_content = self.content_data[self.content_data['content_id'] == selected_id]
                        if not selected_content.empty:
                            selected_row = selected_content.iloc[0]
                            
                            # Genre overlap
                            genre_overlap = len(set(content_row['genres']) & set(selected_row['genres'])) / len(set(content_row['genres']) | set(selected_row['genres']))
                            
                            # Same content type and high genre overlap = too similar
                            if (content_row['content_type'] == selected_row['content_type'] and 
                                genre_overlap > 0.8):
                                should_include = False
                                break
            
            if should_include:
                content_info = self.content_data[self.content_data['content_id'] == content_id]
                if not content_info.empty:
                    content_info = content_info.iloc[0]
                    
                    final_recs.append({
                        'content_id': content_id,
                        'title': content_info['title'],
                        'content_type': content_info['content_type'],
                        'genres': [GENRE_MAP.get(g, str(g)) for g in content_info['genres']],
                        'rating': float(content_info['rating']),
                        'popularity': float(content_info['popularity']),
                        'release_date': content_info['release_date'].isoformat(),
                        'final_score': float(data['final_score']),
                        'recommendation_reasons': data['reasons'],
                        'algorithm_scores': {k: float(v) for k, v in data['scores'].items()},
                        'diversity_score': float(data['diversity_score']),
                        'novelty_score': float(data['novelty_score'])
                    })
                    
                    selected_content_ids.append(content_id)
        
        return final_recs
    
    def update_models(self):
        """Update all models with latest data"""
        print("Updating advanced recommendation models...")
        
        self.last_update = datetime.now()
        
        # Fetch latest data
        if not self.fetch_data_from_backend():
            print("Failed to fetch data from backend")
            return False
        
        try:
            # Build all components
            self.build_user_profiles()
            self.build_content_features()
            self.build_interaction_matrix()
            self.train_collaborative_filtering()
            self.train_matrix_factorization()
            self.train_ensemble_models()
            self.build_graph_network()
            self.analyze_trending_patterns()
            
            print("All models updated successfully")
            return True
        except Exception as e:
            print(f"Error updating models: {e}")
            return False
    
    def get_cache_key(self, user_id, params=None):
        """Generate cache key for user recommendations"""
        base_key = f"recommendations:user:{user_id}"
        if params:
            param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
            base_key += f":{param_hash}"
        return base_key
    
    def cache_recommendations(self, user_id, recommendations, params=None):
        """Cache recommendations in Redis"""
        if redis_client:
            try:
                cache_key = self.get_cache_key(user_id, params)
                redis_client.setex(
                    cache_key, 
                    MODEL_CACHE_DURATION, 
                    json.dumps(recommendations, default=str)
                )
            except Exception as e:
                print(f"Error caching recommendations: {e}")
    
    def get_cached_recommendations(self, user_id, params=None):
        """Get cached recommendations from Redis"""
        if redis_client:
            try:
                cache_key = self.get_cache_key(user_id, params)
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                print(f"Error retrieving cached recommendations: {e}")
        return None

# Initialize the recommendation engine
recommendation_engine = AdvancedRecommendationEngine()

def initialize_models():
    """Initialize models in background"""
    try:
        print("Starting model initialization...")
        success = recommendation_engine.update_models()
        if success:
            print("Advanced recommendation engine initialized successfully")
        else:
            print("Failed to initialize recommendation engine")
    except Exception as e:
        print(f"Error initializing models: {e}")

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(recommendation_engine.models),
        'users_count': len(recommendation_engine.user_profiles),
        'content_count': len(recommendation_engine.content_data) if hasattr(recommendation_engine, 'content_data') else 0,
        'last_update': recommendation_engine.last_update.isoformat() if recommendation_engine.last_update else None,
        'redis_available': redis_client is not None
    })

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Main recommendation endpoint with advanced features"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        num_recommendations = data.get('num_recommendations', 20)
        include_reasons = data.get('include_reasons', True)
        cache_enabled = data.get('cache_enabled', True)
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Check cache first
        cache_params = {'num_recommendations': num_recommendations} if cache_enabled else None
        cached_recs = recommendation_engine.get_cached_recommendations(user_id, cache_params) if cache_enabled else None
        
        if cached_recs:
            return jsonify({
                'recommendations': cached_recs[:num_recommendations],
                'cached': True,
                'timestamp': datetime.now().isoformat(),
                'user_profile': recommendation_engine.user_profiles.get(user_id, {}) if include_reasons else None
            })
        
        # Generate new recommendations
        recommendations = recommendation_engine.get_recommendations(user_id, num_recommendations)
        
        # Cache recommendations
        if cache_enabled:
            recommendation_engine.cache_recommendations(user_id, recommendations, cache_params)
        
        response = {
            'recommendations': recommendations,
            'cached': False,
            'timestamp': datetime.now().isoformat(),
            'total_recommendations': len(recommendations),
            'algorithms_used': list(set([reason for rec in recommendations for reason in rec.get('recommendation_reasons', [])]))
        }
        
        if include_reasons:
            response['user_profile'] = recommendation_engine.user_profiles.get(user_id, {})
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/update-models', methods=['POST'])
def update_models():
    """Update recommendation models"""
    try:
        # Run model update in background
        thread = Thread(target=recommendation_engine.update_models)
        thread.start()
        
        return jsonify({
            'status': 'Model update started',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user-profile/<int:user_id>', methods=['GET'])
def get_user_profile(user_id):
    """Get detailed user profile and preferences"""
    try:
        if user_id in recommendation_engine.user_profiles:
            profile = recommendation_engine.user_profiles[user_id]
            
            # Add interaction statistics
            user_interactions = recommendation_engine.interactions_data[
                recommendation_engine.interactions_data['user_id'] == user_id
            ]
            
            stats = {
                'total_interactions': len(user_interactions),
                'interaction_types': user_interactions['interaction_type'].value_counts().to_dict(),
                'average_rating': float(user_interactions['rating'].mean()) if not user_interactions.empty else 0,
                'most_recent_interaction': user_interactions['timestamp'].max().isoformat() if not user_interactions.empty else None,
                'content_type_distribution': {}
            }
            
            # Content type distribution
            for _, interaction in user_interactions.iterrows():
                content = recommendation_engine.content_data[
                    recommendation_engine.content_data['content_id'] == interaction['content_id']
                ]
                if not content.empty:
                    content_type = content.iloc[0]['content_type']
                    stats['content_type_distribution'][content_type] = stats['content_type_distribution'].get(content_type, 0) + 1
            
            return jsonify({
                'user_profile': profile,
                'interaction_statistics': stats,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'User profile not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/similar-content/<int:content_id>', methods=['GET'])
def get_similar_content(content_id):
    """Get similar content recommendations"""
    try:
        num_similar = request.args.get('num_similar', 10, type=int)
        
        if recommendation_engine.content_similarity_matrix is None:
            return jsonify({'error': 'Content similarity matrix not built'}), 500
        
        content_idx = None
        for idx, row in recommendation_engine.content_data.iterrows():
            if row['content_id'] == content_id:
                content_idx = idx
                break
        
        if content_idx is None:
            return jsonify({'error': 'Content not found'}), 404
        
        # Get similarity scores
        similarities = recommendation_engine.content_similarity_matrix[content_idx]
        similar_indices = np.argsort(similarities)[::-1][1:num_similar+1]  # Exclude self
        
        similar_content = []
        for idx in similar_indices:
            content_row = recommendation_engine.content_data.iloc[idx]
            similar_content.append({
                'content_id': int(content_row['content_id']),
                'title': content_row['title'],
                'content_type': content_row['content_type'],
                'genres': [GENRE_MAP.get(g, str(g)) for g in content_row['genres']],
                'similarity_score': float(similarities[idx]),
                'rating': float(content_row['rating']),
                'popularity': float(content_row['popularity']),
                'release_date': content_row['release_date'].isoformat()
            })
        
        # Get base content info
        base_content = recommendation_engine.content_data.iloc[content_idx]
        
        return jsonify({
            'similar_content': similar_content,
            'base_content': {
                'content_id': int(base_content['content_id']),
                'title': base_content['title'],
                'content_type': base_content['content_type'],
                'genres': [GENRE_MAP.get(g, str(g)) for g in base_content['genres']],
                'rating': float(base_content['rating'])
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trending', methods=['GET'])
def get_trending():
    """Get trending content with advanced analytics"""
    try:
        num_trending = request.args.get('num_trending', 20, type=int)
        content_type = request.args.get('content_type', None)
        time_window = request.args.get('time_window', 'week')  # day, week, month
        
        if not recommendation_engine.trending_scores:
            return jsonify({'error': 'Trending data not available'}), 500
        
        # Filter by time window
        if time_window == 'day':
            cutoff_date = datetime.now() - timedelta(days=1)
        elif time_window == 'month':
            cutoff_date = datetime.now() - timedelta(days=30)
        else:  # week
            cutoff_date = datetime.now() - timedelta(days=7)
        
        # Get recent interactions for trending calculation
        recent_interactions = recommendation_engine.interactions_data[
            recommendation_engine.interactions_data['timestamp'] > cutoff_date
        ]
        
        # Recalculate trending scores for the specific time window
        if not recent_interactions.empty:
            trending_scores = recent_interactions.groupby('content_id').agg({
                'user_id': 'nunique',
                'interaction_type': 'count',
                'rating': 'mean'
            }).fillna(0)
            
            trending_scores['trending_score'] = (
                trending_scores['user_id'] * 0.4 +
                trending_scores['interaction_type'] * 0.3 +
                trending_scores['rating'] * 0.3
            )
            
            trending_items = trending_scores['trending_score'].to_dict()
        else:
            trending_items = recommendation_engine.trending_scores
        
        # Sort by trending score
        sorted_trending = sorted(trending_items.items(), key=lambda x: x[1], reverse=True)
        
        trending_content = []
        for content_id, score in sorted_trending:
            if len(trending_content) >= num_trending:
                break
                
            content_row = recommendation_engine.content_data[
                recommendation_engine.content_data['content_id'] == content_id
            ]
            
            if not content_row.empty:
                content_row = content_row.iloc[0]
                
                # Filter by content type if specified
                if content_type and content_row['content_type'] != content_type:
                    continue
                
                trending_content.append({
                    'content_id': int(content_id),
                    'title': content_row['title'],
                    'content_type': content_row['content_type'],
                    'genres': [GENRE_MAP.get(g, str(g)) for g in content_row['genres']],
                    'trending_score': float(score),
                    'rating': float(content_row['rating']),
                    'popularity': float(content_row['popularity']),
                    'release_date': content_row['release_date'].isoformat()
                })
        
        return jsonify({
            'trending_content': trending_content,
            'time_window': time_window,
            'content_type_filter': content_type,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-stats', methods=['GET'])
def get_model_stats():
    """Get comprehensive model statistics and performance metrics"""
    try:
        # Basic model info
        stats = {
            'models_loaded': list(recommendation_engine.models.keys()),
            'total_users': len(recommendation_engine.user_profiles),
            'total_content': len(recommendation_engine.content_data) if hasattr(recommendation_engine, 'content_data') else 0,
            'total_interactions': len(recommendation_engine.interactions_data) if hasattr(recommendation_engine, 'interactions_data') else 0,
            'last_update': recommendation_engine.last_update.isoformat() if recommendation_engine.last_update else None,
            'cache_available': redis_client is not None
        }
        
        # Matrix information
        if recommendation_engine.content_similarity_matrix is not None:
            stats['content_similarity_matrix_shape'] = recommendation_engine.content_similarity_matrix.shape
        
        if recommendation_engine.interaction_matrix:
            stats['interaction_matrix_shape'] = recommendation_engine.interaction_matrix['matrix'].shape
        
        # Graph information
        if recommendation_engine.graph:
            stats['graph_nodes'] = recommendation_engine.graph.number_of_nodes()
            stats['graph_edges'] = recommendation_engine.graph.number_of_edges()
        
        # Trending and seasonal data
        stats['trending_items_count'] = len(recommendation_engine.trending_scores)
        stats['seasonal_patterns'] = {k: len(v.get('content_ids', [])) for k, v in recommendation_engine.seasonal_patterns.items()}
        
        # User type distribution
        if hasattr(recommendation_engine, 'users_data'):
            user_types = recommendation_engine.users_data['preferences'].apply(lambda x: x.get('user_type', 'unknown')).value_counts().to_dict()
            stats['user_type_distribution'] = user_types
        
        # Content type distribution
        if hasattr(recommendation_engine, 'content_data'):
            content_types = recommendation_engine.content_data['content_type'].value_counts().to_dict()
            stats['content_type_distribution'] = content_types
        
        # Interaction type distribution
        if hasattr(recommendation_engine, 'interactions_data'):
            interaction_types = recommendation_engine.interactions_data['interaction_type'].value_counts().to_dict()
            stats['interaction_type_distribution'] = interaction_types
        
        return jsonify({
            'model_statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommendations/batch', methods=['POST'])
def get_batch_recommendations():
    """Get recommendations for multiple users"""
    try:
        data = request.get_json()
        user_ids = data.get('user_ids', [])
        num_recommendations = data.get('num_recommendations', 10)
        
        if not user_ids:
            return jsonify({'error': 'user_ids list is required'}), 400
        
        batch_results = {}
        
        for user_id in user_ids:
            try:
                recommendations = recommendation_engine.get_recommendations(user_id, num_recommendations)
                batch_results[user_id] = {
                    'recommendations': recommendations,
                    'success': True
                }
            except Exception as e:
                batch_results[user_id] = {
                    'error': str(e),
                    'success': False
                }
        
        return jsonify({
            'batch_recommendations': batch_results,
            'timestamp': datetime.now().isoformat(),
            'total_users': len(user_ids),
            'successful_recommendations': sum(1 for r in batch_results.values() if r['success'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize models when the app starts
if __name__ == '__main__':
    # Start model initialization in background
    init_thread = Thread(target=initialize_models)
    init_thread.start()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5001)
