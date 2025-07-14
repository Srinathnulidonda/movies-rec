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
from surprise import Dataset, Reader, SVD, NMF as SurpriseNMF, KNNWithMeans, BaselineOnly
from surprise.model_selection import train_test_split
import implicit
from scipy.sparse import csr_matrix
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
except:
    redis_client = None

# Genre mapping from backend
GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western", 10759: "Action & Adventure",
    10762: "Kids", 10763: "News", 10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap",
    10767: "Talk", 10768: "War & Politics"
}

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
        
    def fetch_data_from_backend(self):
        """Fetch all necessary data from backend"""
        try:
            # Fetch users data
            users_response = requests.get(f"{BACKEND_URL}/api/admin/analytics", timeout=10)
            
            # Since we don't have direct API endpoints, we'll simulate the data structure
            # In a real implementation, you'd create specific API endpoints for ML data
            
            # For now, let's create sample data structure that matches your backend
            self.users_data = self._generate_sample_users_data()
            self.content_data = self._generate_sample_content_data()
            self.interactions_data = self._generate_sample_interactions_data()
            
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def _generate_sample_users_data(self):
        """Generate sample users data - replace with actual API call"""
        return pd.DataFrame({
            'user_id': range(1, 1001),
            'preferences': [{'genre_weights': {str(g): np.random.random() for g in [28, 35, 18, 27, 878]}} for _ in range(1000)],
            'created_at': pd.date_range('2023-01-01', periods=1000, freq='H')
        })
    
    def _generate_sample_content_data(self):
        """Generate sample content data - replace with actual API call"""
        content_types = ['movie', 'tv', 'anime']
        return pd.DataFrame({
            'content_id': range(1, 5001),
            'title': [f'Content {i}' for i in range(1, 5001)],
            'genres': [[np.random.choice(list(GENRE_MAP.keys())) for _ in range(np.random.randint(1, 4))] for _ in range(5000)],
            'overview': [f'This is an amazing {np.random.choice(content_types)} with great storyline.' for _ in range(5000)],
            'rating': np.random.uniform(1, 10, 5000),
            'popularity': np.random.uniform(0, 100, 5000),
            'content_type': [np.random.choice(content_types) for _ in range(5000)],
            'release_date': pd.date_range('2000-01-01', periods=5000, freq='D'),
            'runtime': np.random.randint(60, 180, 5000)
        })
    
    def _generate_sample_interactions_data(self):
        """Generate sample interactions data - replace with actual API call"""
        interaction_types = ['view', 'like', 'favorite', 'wishlist']
        return pd.DataFrame({
            'user_id': np.random.randint(1, 1001, 50000),
            'content_id': np.random.randint(1, 5001, 50000),
            'interaction_type': [np.random.choice(interaction_types) for _ in range(50000)],
            'rating': np.random.randint(1, 6, 50000),
            'timestamp': pd.date_range('2023-01-01', periods=50000, freq='T')
        })
    
    def build_user_profiles(self):
        """Build comprehensive user profiles"""
        print("Building user profiles...")
        
        for user_id in self.users_data['user_id'].unique():
            user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
            user_data = self.users_data[self.users_data['user_id'] == user_id].iloc[0]
            
            profile = {
                'user_id': user_id,
                'total_interactions': len(user_interactions),
                'interaction_diversity': len(user_interactions['interaction_type'].unique()),
                'avg_rating': user_interactions['rating'].mean(),
                'rating_variance': user_interactions['rating'].var(),
                'genre_preferences': {},
                'content_type_preferences': {},
                'temporal_patterns': {},
                'recency_bias': 0,
                'exploration_vs_exploitation': 0
            }
            
            # Analyze genre preferences
            for _, interaction in user_interactions.iterrows():
                content = self.content_data[self.content_data['content_id'] == interaction['content_id']]
                if not content.empty:
                    content_row = content.iloc[0]
                    for genre in content_row['genres']:
                        genre_name = GENRE_MAP.get(genre, str(genre))
                        weight = self._get_interaction_weight(interaction['interaction_type'], interaction['rating'])
                        profile['genre_preferences'][genre_name] = profile['genre_preferences'].get(genre_name, 0) + weight
                    
                    # Content type preferences
                    content_type = content_row['content_type']
                    weight = self._get_interaction_weight(interaction['interaction_type'], interaction['rating'])
                    profile['content_type_preferences'][content_type] = profile['content_type_preferences'].get(content_type, 0) + weight
            
            # Normalize preferences
            total_genre_weight = sum(profile['genre_preferences'].values())
            if total_genre_weight > 0:
                profile['genre_preferences'] = {k: v/total_genre_weight for k, v in profile['genre_preferences'].items()}
            
            total_content_weight = sum(profile['content_type_preferences'].values())
            if total_content_weight > 0:
                profile['content_type_preferences'] = {k: v/total_content_weight for k, v in profile['content_type_preferences'].items()}
            
            # Temporal patterns
            profile['temporal_patterns'] = self._analyze_temporal_patterns(user_interactions)
            
            # Recency bias
            profile['recency_bias'] = self._calculate_recency_bias(user_interactions)
            
            # Exploration vs exploitation
            profile['exploration_vs_exploitation'] = self._calculate_exploration_score(user_interactions)
            
            self.user_profiles[user_id] = profile
    
    def _get_interaction_weight(self, interaction_type, rating):
        """Convert interaction type and rating to weight"""
        base_weights = {'view': 1, 'like': 3, 'favorite': 5, 'wishlist': 2}
        base_weight = base_weights.get(interaction_type, 1)
        rating_weight = rating / 5.0 if rating else 1.0
        return base_weight * rating_weight
    
    def _analyze_temporal_patterns(self, user_interactions):
        """Analyze user's temporal viewing patterns"""
        if user_interactions.empty:
            return {}
        
        timestamps = pd.to_datetime(user_interactions['timestamp'])
        
        return {
            'preferred_hours': timestamps.dt.hour.mode().values[0] if not timestamps.empty else 20,
            'preferred_days': timestamps.dt.dayofweek.mode().values[0] if not timestamps.empty else 5,
            'activity_level': len(timestamps) / (timestamps.max() - timestamps.min()).days if len(timestamps) > 1 else 1,
            'binge_tendency': self._calculate_binge_tendency(timestamps)
        }
    
    def _calculate_binge_tendency(self, timestamps):
        """Calculate user's tendency to binge watch"""
        if len(timestamps) < 2:
            return 0
        
        time_diffs = timestamps.diff().dropna()
        short_intervals = (time_diffs < pd.Timedelta(hours=2)).sum()
        return short_intervals / len(time_diffs)
    
    def _calculate_recency_bias(self, user_interactions):
        """Calculate how much user prefers recent content"""
        if user_interactions.empty:
            return 0.5
        
        recent_interactions = user_interactions[user_interactions['timestamp'] > datetime.now() - timedelta(days=30)]
        return len(recent_interactions) / len(user_interactions)
    
    def _calculate_exploration_score(self, user_interactions):
        """Calculate user's exploration vs exploitation tendency"""
        if user_interactions.empty:
            return 0.5
        
        content_ids = user_interactions['content_id'].unique()
        repeated_content = user_interactions['content_id'].value_counts()
        exploration_score = len(content_ids) / len(user_interactions)
        return exploration_score
    
    def build_content_features(self):
        """Build comprehensive content feature matrix"""
        print("Building content features...")
        
        # Text features using TF-IDF
        text_features = self.content_data['title'] + ' ' + self.content_data['overview'].fillna('')
        
        self.vectorizers['tfidf'] = TfidfVectorizer(
            max_features=5000, 
            stop_words='english', 
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        tfidf_matrix = self.vectorizers['tfidf'].fit_transform(text_features)
        
        # Genre features (one-hot encoding)
        all_genres = set()
        for genres in self.content_data['genres']:
            all_genres.update([GENRE_MAP.get(g, str(g)) for g in genres])
        
        genre_features = np.zeros((len(self.content_data), len(all_genres)))
        genre_list = list(all_genres)
        
        for i, genres in enumerate(self.content_data['genres']):
            for genre in genres:
                genre_name = GENRE_MAP.get(genre, str(genre))
                if genre_name in genre_list:
                    genre_features[i, genre_list.index(genre_name)] = 1
        
        # Numerical features
        numerical_features = self.content_data[['rating', 'popularity', 'runtime']].fillna(0)
        self.scalers['numerical'] = StandardScaler()
        scaled_numerical = self.scalers['numerical'].fit_transform(numerical_features)
        
        # Content type features
        content_types = pd.get_dummies(self.content_data['content_type'])
        
        # Release year features
        release_years = pd.to_datetime(self.content_data['release_date']).dt.year.fillna(2000)
        current_year = datetime.now().year
        year_features = ((current_year - release_years) / (current_year - 1900)).values.reshape(-1, 1)
        
        # Combine all features
        from scipy.sparse import hstack, csr_matrix
        
        content_features = hstack([
            tfidf_matrix,
            csr_matrix(genre_features),
            csr_matrix(scaled_numerical),
            csr_matrix(content_types.values),
            csr_matrix(year_features)
        ])
        
        self.content_features['matrix'] = content_features
        self.content_features['genre_list'] = genre_list
        
        # Calculate content similarity matrix
        self.content_similarity_matrix = cosine_similarity(content_features)
    
    def build_interaction_matrix(self):
        """Build user-item interaction matrix"""
        print("Building interaction matrix...")
        
        # Create weighted interaction matrix
        interaction_weights = self.interactions_data.copy()
        interaction_weights['weight'] = interaction_weights.apply(
            lambda x: self._get_interaction_weight(x['interaction_type'], x['rating']), axis=1
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
    
    def train_collaborative_filtering(self):
        """Train collaborative filtering models using Surprise library"""
        print("Training collaborative filtering models...")
        
        # Prepare data for Surprise
        ratings_data = []
        for _, row in self.interactions_data.iterrows():
            weight = self._get_interaction_weight(row['interaction_type'], row['rating'])
            ratings_data.append((row['user_id'], row['content_id'], weight))
        
        # Create Surprise dataset
        reader = Reader(rating_scale=(0, 5))
        dataset = Dataset.load_from_df(pd.DataFrame(ratings_data, columns=['user_id', 'content_id', 'rating']), reader)
        
        # Train multiple models
        self.models['svd'] = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        self.models['nmf'] = SurpriseNMF(n_factors=50, n_epochs=20)
        self.models['knn'] = KNNWithMeans(k=40, sim_options={'name': 'cosine', 'user_based': True})
        
        trainset = dataset.build_full_trainset()
        
        for name, model in self.models.items():
            if name in ['svd', 'nmf', 'knn']:
                model.fit(trainset)
    
    def train_matrix_factorization(self):
        """Train matrix factorization using Implicit library"""
        print("Training matrix factorization...")
        
        if self.interaction_matrix is None:
            return
        
        # Train Alternating Least Squares
        self.models['als'] = implicit.als.AlternatingLeastSquares(
            factors=64, 
            regularization=0.01, 
            iterations=20
        )
        
        # Implicit expects item-user matrix
        item_user_matrix = self.interaction_matrix['matrix'].T
        self.models['als'].fit(item_user_matrix)
        
        # Train Bayesian Personalized Ranking
        self.models['bpr'] = implicit.bpr.BayesianPersonalizedRanking(
            factors=64,
            learning_rate=0.01,
            regularization=0.01,
            iterations=100
        )
        
        self.models['bpr'].fit(item_user_matrix)
    
    def train_ensemble_models(self):
        """Train ensemble models for advanced recommendations"""
        print("Training ensemble models...")
        
        # Prepare features for ensemble
        user_features = []
        item_features = []
        ratings = []
        
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
                    profile['exploration_vs_exploitation']
                ]
                
                # Content features
                content_row = self.content_data[self.content_data['content_id'] == content_id]
                if not content_row.empty:
                    content_row = content_row.iloc[0]
                    content_feat = [
                        content_row['rating'],
                        content_row['popularity'],
                        content_row['runtime'],
                        len(content_row['genres'])
                    ]
                    
                    combined_features = user_feat + content_feat
                    user_features.append(combined_features)
                    ratings.append(self._get_interaction_weight(row['interaction_type'], row['rating']))
        
        if user_features:
            X = np.array(user_features)
            y = np.array(ratings)
            
            # Train multiple ensemble models
            self.models['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['random_forest'].fit(X, y)
            
            self.models['gradient_boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.models['gradient_boosting'].fit(X, y)
            
            self.models['lightgbm'] = lgb.LGBMRegressor(n_estimators=100, random_state=42)
            self.models['lightgbm'].fit(X, y)
            
            # Neural network
            self.models['neural_network'] = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                random_state=42,
                max_iter=200
            )
            self.models['neural_network'].fit(X, y)
    
    def build_graph_network(self):
        """Build user-content graph for graph-based recommendations"""
        print("Building graph network...")
        
        self.graph = nx.Graph()
        
        # Add user nodes
        for user_id in self.users_data['user_id'].unique():
            self.graph.add_node(f"user_{user_id}", type='user')
        
        # Add content nodes
        for content_id in self.content_data['content_id'].unique():
            self.graph.add_node(f"content_{content_id}", type='content')
        
        # Add edges based on interactions
        for _, row in self.interactions_data.iterrows():
            weight = self._get_interaction_weight(row['interaction_type'], row['rating'])
            self.graph.add_edge(
                f"user_{row['user_id']}", 
                f"content_{row['content_id']}", 
                weight=weight
            )
        
        # Add content-content edges based on similarity
        for i, content_id1 in enumerate(self.content_data['content_id']):
            for j, content_id2 in enumerate(self.content_data['content_id']):
                if i < j and self.content_similarity_matrix is not None:
                    similarity = self.content_similarity_matrix[i, j]
                    if similarity > 0.7:  # Only add high similarity edges
                        self.graph.add_edge(
                            f"content_{content_id1}",
                            f"content_{content_id2}",
                            weight=similarity
                        )
    
    def analyze_trending_patterns(self):
        """Analyze trending patterns and seasonal behaviors"""
        print("Analyzing trending patterns...")
        
        # Calculate trending scores based on recent interactions
        recent_interactions = self.interactions_data[
            self.interactions_data['timestamp'] > datetime.now() - timedelta(days=7)
        ]
        
        trending_scores = recent_interactions.groupby('content_id').agg({
            'user_id': 'nunique',  # Number of unique users
            'interaction_type': 'count',  # Total interactions
            'rating': 'mean'  # Average rating
        }).fillna(0)
        
        # Normalize trending scores
        trending_scores['trending_score'] = (
            trending_scores['user_id'] * 0.4 +
            trending_scores['interaction_type'] * 0.3 +
            trending_scores['rating'] * 0.3
        )
        
        self.trending_scores = trending_scores['trending_score'].to_dict()
        
        # Seasonal patterns
        interactions_with_season = self.interactions_data.copy()
        interactions_with_season['month'] = pd.to_datetime(interactions_with_season['timestamp']).dt.month
        interactions_with_season['season'] = interactions_with_season['month'].apply(self._get_season)
        
        seasonal_preferences = interactions_with_season.groupby(['season', 'content_id']).size().reset_index(name='count')
        self.seasonal_patterns = seasonal_preferences.groupby('season')['content_id'].apply(list).to_dict()
    
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
        """Generate comprehensive recommendations for a user"""
        recommendations = {}
        
        # 1. Content-based recommendations
        recommendations['content_based'] = self._get_content_based_recommendations(user_id, num_recommendations // 4)
        
        # 2. Collaborative filtering recommendations
        recommendations['collaborative'] = self._get_collaborative_recommendations(user_id, num_recommendations // 4)
        
        # 3. Matrix factorization recommendations
        recommendations['matrix_factorization'] = self._get_matrix_factorization_recommendations(user_id, num_recommendations // 4)
        
        # 4. Graph-based recommendations
        recommendations['graph_based'] = self._get_graph_recommendations(user_id, num_recommendations // 4)
        
        # 5. Trending recommendations
        recommendations['trending'] = self._get_trending_recommendations(user_id, num_recommendations // 5)
        
        # 6. Seasonal recommendations
        recommendations['seasonal'] = self._get_seasonal_recommendations(user_id, num_recommendations // 5)
        
        # 7. Ensemble hybrid recommendations
        recommendations['ensemble'] = self._get_ensemble_recommendations(user_id, num_recommendations // 4)
        
        # Combine and rank all recommendations
        final_recommendations = self._combine_recommendations(user_id, recommendations, num_recommendations)
        
        return final_recommendations
    
    def _get_content_based_recommendations(self, user_id, num_recs):
        """Content-based filtering recommendations"""
        if user_id not in self.user_profiles or self.content_features.get('matrix') is None:
            return []
        
        user_profile = self.user_profiles[user_id]
        user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
        
        if user_interactions.empty:
            return []
        
        # Create user preference vector
        user_vector = np.zeros(self.content_features['matrix'].shape[1])
        
        for _, interaction in user_interactions.iterrows():
            content_idx = self.content_data[self.content_data['content_id'] == interaction['content_id']].index
            if not content_idx.empty:
                content_idx = content_idx[0]
                weight = self._get_interaction_weight(interaction['interaction_type'], interaction['rating'])
                user_vector += self.content_features['matrix'][content_idx].toarray()[0] * weight
        
        user_vector = user_vector / np.linalg.norm(user_vector) if np.linalg.norm(user_vector) > 0 else user_vector
        
        # Calculate similarities
        similarities = cosine_similarity([user_vector], self.content_features['matrix'])[0]
        
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
                    'score': similarities[idx],
                    'reason': 'content_based'
                })
        
        return recommendations
    
    def _get_collaborative_recommendations(self, user_id, num_recs):
        """Collaborative filtering recommendations"""
        recommendations = []
        
        if 'svd' not in self.models:
            return recommendations
        
        # Get all content IDs
        all_content_ids = self.content_data['content_id'].unique()
        user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
        interacted_content = set(user_interactions['content_id'])
        
        # Predict ratings for uninteracted content
        predictions = []
        for content_id in all_content_ids:
            if content_id not in interacted_content:
                pred = self.models['svd'].predict(user_id, content_id)
                predictions.append((content_id, pred.est))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        for content_id, score in predictions[:num_recs]:
            recommendations.append({
                'content_id': content_id,
                'score': score,
                'reason': 'collaborative'
            })
        
        return recommendations
    
    def _get_matrix_factorization_recommendations(self, user_id, num_recs):
        """Matrix factorization recommendations using Implicit"""
        recommendations = []
        
        if 'als' not in self.models or self.interaction_matrix is None:
            return recommendations
        
        user_to_idx = self.interaction_matrix['user_to_idx']
        idx_to_content = self.interaction_matrix['idx_to_content']
        
        if user_id not in user_to_idx:
            return recommendations
        
        user_idx = user_to_idx[user_id]
        
        # Get recommendations from ALS model
        try:
            item_ids, scores = self.models['als'].recommend(
                user_idx, 
                self.interaction_matrix['matrix'][user_idx],
                N=num_recs,
                filter_already_liked_items=True
            )
            
            for item_idx, score in zip(item_ids, scores):
                if item_idx in idx_to_content:
                    content_id = idx_to_content[item_idx]
                    recommendations.append({
                        'content_id': content_id,
                        'score': float(score),
                        'reason': 'matrix_factorization'
                    })
        except:
            pass
        
        return recommendations
    
    def _get_graph_recommendations(self, user_id, num_recs):
        """Graph-based recommendations using network analysis"""
        recommendations = []
        
        if self.graph is None:
            return recommendations
        
        user_node = f"user_{user_id}"
        if user_node not in self.graph:
            return recommendations
        
        # Use personalized PageRank
        try:
            pagerank_scores = nx.pagerank(self.graph, personalization={user_node: 1})
            
            # Filter content nodes and sort by score
            content_scores = []
            user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
            interacted_content = set(user_interactions['content_id'])
            
            for node, score in pagerank_scores.items():
                if node.startswith('content_'):
                    content_id = int(node.split('_')[1])
                    if content_id not in interacted_content:
                        content_scores.append((content_id, score))
            
            content_scores.sort(key=lambda x: x[1], reverse=True)
            
            for content_id, score in content_scores[:num_recs]:
                recommendations.append({
                    'content_id': content_id,
                    'score': score,
                    'reason': 'graph_based'
                })
        except:
            pass
        
        return recommendations
    
    def _get_trending_recommendations(self, user_id, num_recs):
        """Trending content recommendations"""
        recommendations = []
        
        if not self.trending_scores:
            return recommendations
        
        user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
        interacted_content = set(user_interactions['content_id'])
        
        # Sort by trending score
        trending_items = sorted(self.trending_scores.items(), key=lambda x: x[1], reverse=True)
        
        for content_id, score in trending_items:
            if len(recommendations) >= num_recs:
                break
            
            if content_id not in interacted_content:
                recommendations.append({
                    'content_id': content_id,
                    'score': score,
                    'reason': 'trending'
                })
        
        return recommendations
    
    def _get_seasonal_recommendations(self, user_id, num_recs):
        """Seasonal content recommendations"""
        recommendations = []
        
        if not self.seasonal_patterns:
            return recommendations
        
        current_season = self._get_season(datetime.now().month)
        seasonal_content = self.seasonal_patterns.get(current_season, [])
        
        user_interactions = self.interactions_data[self.interactions_data['user_id'] == user_id]
        interacted_content = set(user_interactions['content_id'])
        
        for content_id in seasonal_content:
            if len(recommendations) >= num_recs:
                break
            
            if content_id not in interacted_content:
                recommendations.append({
                    'content_id': content_id,
                    'score': 1.0,
                    'reason': 'seasonal'
                })
        
        return recommendations
    
    def _get_ensemble_recommendations(self, user_id, num_recs):
        """Ensemble model recommendations"""
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
            user_profile['exploration_vs_exploitation']
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
                    len(content_row['genres'])
                ]
                
                combined_features = np.array([user_feat + content_feat])
                
                # Average predictions from ensemble models
                scores = []
                for model_name in ['random_forest', 'gradient_boosting', 'lightgbm', 'neural_network']:
                    if model_name in self.models:
                        try:
                            score = self.models[model_name].predict(combined_features)[0]
                            scores.append(score)
                        except:
                            pass
                
                if scores:
                    avg_score = np.mean(scores)
                    content_scores.append((content_id, avg_score))
        
        # Sort by score
        content_scores.sort(key=lambda x: x[1], reverse=True)
        
        for content_id, score in content_scores[:num_recs]:
            recommendations.append({
                'content_id': content_id,
                'score': score,
                'reason': 'ensemble'
            })
        
        return recommendations
    
    def _combine_recommendations(self, user_id, recommendations_dict, num_final):
        """Combine and rank all recommendations using advanced ensemble techniques"""
        all_recommendations = {}
        
        # Collect all recommendations with their scores and reasons
        for reason, recs in recommendations_dict.items():
            for rec in recs:
                content_id = rec['content_id']
                score = rec['score']
                
                if content_id not in all_recommendations:
                    all_recommendations[content_id] = {
                        'scores': {},
                        'reasons': [],
                        'final_score': 0
                    }
                
                all_recommendations[content_id]['scores'][reason] = score
                all_recommendations[content_id]['reasons'].append(reason)
        
        # Calculate final scores using weighted combination
        weights = {
            'content_based': 0.2,
            'collaborative': 0.2,
            'matrix_factorization': 0.15,
            'graph_based': 0.15,
            'trending': 0.1,
            'seasonal': 0.05,
            'ensemble': 0.15
        }
        
        for content_id, data in all_recommendations.items():
            final_score = 0
            total_weight = 0
            
            for reason, score in data['scores'].items():
                weight = weights.get(reason, 0.1)
                final_score += score * weight
                total_weight += weight
            
            # Normalize score
            if total_weight > 0:
                final_score /= total_weight
            
            # Boost score if recommended by multiple algorithms
            diversity_bonus = len(data['reasons']) * 0.1
            final_score += diversity_bonus
            
            # Apply user-specific boosts
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                
                # Boost trending content for users with high recency bias
                if 'trending' in data['reasons'] and profile['recency_bias'] > 0.7:
                    final_score *= 1.2
                
                # Boost diverse content for exploratory users
                if profile['exploration_vs_exploitation'] > 0.7:
                    final_score *= 1.1
            
            data['final_score'] = final_score
        
        # Sort by final score and return top recommendations
        sorted_recommendations = sorted(
            all_recommendations.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )
        
        final_recs = []
        for content_id, data in sorted_recommendations[:num_final]:
            content_info = self.content_data[self.content_data['content_id'] == content_id]
            if not content_info.empty:
                content_info = content_info.iloc[0]
                
                final_recs.append({
                    'content_id': content_id,
                    'title': content_info['title'],
                    'content_type': content_info['content_type'],
                    'genres': [GENRE_MAP.get(g, str(g)) for g in content_info['genres']],
                    'rating': content_info['rating'],
                    'popularity': content_info['popularity'],
                    'final_score': data['final_score'],
                    'recommendation_reasons': data['reasons'],
                    'algorithm_scores': data['scores']
                })
        
        return final_recs
    
    def update_models(self):
        """Update all models with latest data"""
        print("Updating recommendation models...")
        
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
            
            print("Models updated successfully")
            return True
        except Exception as e:
            print(f"Error updating models: {e}")
            return False
    
    def get_cache_key(self, user_id):
        """Generate cache key for user recommendations"""
        return f"recommendations:user:{user_id}"
    
    def cache_recommendations(self, user_id, recommendations):
        """Cache recommendations in Redis"""
        if redis_client:
            try:
                cache_key = self.get_cache_key(user_id)
                redis_client.setex(
                    cache_key, 
                    MODEL_CACHE_DURATION, 
                    json.dumps(recommendations, default=str)
                )
            except:
                pass
    
    def get_cached_recommendations(self, user_id):
        """Get cached recommendations from Redis"""
        if redis_client:
            try:
                cache_key = self.get_cache_key(user_id)
                cached = redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except:
                pass
        return None

# Initialize the recommendation engine
recommendation_engine = AdvancedRecommendationEngine()

def initialize_models():
    """Initialize models in background"""
    try:
        success = recommendation_engine.update_models()
        if success:
            print("Recommendation engine initialized successfully")
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
        'last_update': recommendation_engine.last_update.isoformat() if recommendation_engine.last_update else None
    })

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Main recommendation endpoint"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        num_recommendations = data.get('num_recommendations', 20)
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Check cache first
        cached_recs = recommendation_engine.get_cached_recommendations(user_id)
        if cached_recs:
            return jsonify({
                'recommendations': cached_recs[:num_recommendations],
                'cached': True,
                'timestamp': datetime.now().isoformat()
            })
        
        # Generate new recommendations
        recommendations = recommendation_engine.get_recommendations(user_id, num_recommendations)
        
        # Cache recommendations
        recommendation_engine.cache_recommendations(user_id, recommendations)
        
        return jsonify({
            'recommendations': recommendations,
            'cached': False,
            'timestamp': datetime.now().isoformat(),
            'total_recommendations': len(recommendations)
        })
        
    except Exception as e:
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
    """Get user profile and preferences"""
    try:
        if user_id in recommendation_engine.user_profiles:
            profile = recommendation_engine.user_profiles[user_id]
            return jsonify({
                'user_profile': profile,
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
                'content_id': content_row['content_id'],
                'title': content_row['title'],
                'content_type': content_row['content_type'],
                'genres': [GENRE_MAP.get(g, str(g)) for g in content_row['genres']],
                'similarity_score': similarities[idx],
                'rating': content_row['rating']
            })
        
        return jsonify({
            'similar_content': similar_content,
            'base_content_id': content_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trending', methods=['GET'])
def get_trending():
    """Get trending content"""
    try:
        num_trending = request.args.get('num_trending', 20, type=int)
        
        if not recommendation_engine.trending_scores:
            return jsonify({'error': 'Trending data not available'}), 500
        
        # Sort by trending score
        trending_items = sorted(
            recommendation_engine.trending_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_trending]
        
        trending_content = []
        for content_id, score in trending_items:
            content_row = recommendation_engine.content_data[
                recommendation_engine.content_data['content_id'] == content_id
            ]
            
            if not content_row.empty:
                content_row = content_row.iloc[0]
                trending_content.append({
                    'content_id': content_id,
                    'title': content_row['title'],
                    'content_type': content_row['content_type'],
                    'genres': [GENRE_MAP.get(g, str(g)) for g in content_row['genres']],
                    'trending_score': score,
                    'rating': content_row['rating'],
                    'popularity': content_row['popularity']
                })
        
        return jsonify({
            'trending_content': trending_content,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-stats', methods=['GET'])
def get_model_stats():
    """Get model statistics and performance metrics"""
    try:
        stats = {
            'models_loaded': list(recommendation_engine.models.keys()),
            'total_users': len(recommendation_engine.user_profiles),
            'total_content': len(recommendation_engine.content_data) if hasattr(recommendation_engine, 'content_data') else 0,
            'total_interactions': len(recommendation_engine.interactions_data) if hasattr(recommendation_engine, 'interactions_data') else 0,
            'content_similarity_matrix_shape': recommendation_engine.content_similarity_matrix.shape if recommendation_engine.content_similarity_matrix is not None else None,
            'interaction_matrix_shape': recommendation_engine.interaction_matrix['matrix'].shape if recommendation_engine.interaction_matrix else None,
            'graph_nodes': recommendation_engine.graph.number_of_nodes() if recommendation_engine.graph else 0,
            'graph_edges': recommendation_engine.graph.number_of_edges() if recommendation_engine.graph else 0,
            'trending_items_count': len(recommendation_engine.trending_scores),
            'last_update': recommendation_engine.last_update.isoformat() if recommendation_engine.last_update else None,
            'cache_available': redis_client is not None
        }
        
        return jsonify({
            'model_statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize models on startup
@app.before_first_request
def startup():
    """Initialize models on first request"""
    thread = Thread(target=initialize_models)
    thread.start()

if __name__ == '__main__':
    # Initialize models
    initialize_models()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5001)