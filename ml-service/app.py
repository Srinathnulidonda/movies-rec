from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
import sqlite3
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import threading
import time
import requests
from functools import lru_cache
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class AdvancedRecommendationEngine:
    def __init__(self):
        self.user_item_matrix = None
        self.content_features = None
        self.user_profiles = {}
        self.item_profiles = {}
        self.models = {}
        self.scaler = StandardScaler()
        self.content_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.genre_weights = {}
        self.popularity_scores = {}
        self.seasonal_weights = {}
        self.user_clusters = {}
        self.item_clusters = {}
        self.last_update = datetime.now()
        self.update_lock = threading.Lock()
        
    def load_data_from_backend(self):
        """Load user interactions and content from backend database"""
        try:
            backend_url = os.getenv('BACKEND_URL', 'https://backend-app-970m.onrender.com')
            
            # Fetch user interactions
            interactions_response = requests.get(f'{backend_url}/api/ml/interactions', timeout=30)
            content_response = requests.get(f'{backend_url}/api/ml/content', timeout=30)
            
            if interactions_response.status_code == 200 and content_response.status_code == 200:
                interactions_data = interactions_response.json()
                content_data = content_response.json()
                
                self.interactions_df = pd.DataFrame(interactions_data)
                self.content_df = pd.DataFrame(content_data)
                
                if not self.interactions_df.empty:
                    self.interactions_df['rating'] = self.interactions_df['rating'].fillna(3.5)
                    self.interactions_df['weight'] = self.interactions_df['interaction_type'].map({
                        'favorite': 5.0, 'like': 4.0, 'view': 3.0, 'wishlist': 4.5
                    }).fillna(3.0)
                
                return True
            else:
                self._create_dummy_data()
                return False
                
        except Exception as e:
            print(f"Error loading data: {e}")
            self._create_dummy_data()
            return False
    
    def _create_dummy_data(self):
        """Create dummy data for testing"""
        np.random.seed(42)
        
        # Generate dummy interactions
        users = list(range(1, 51))
        items = list(range(1, 201))
        
        interactions = []
        for user in users:
            n_interactions = np.random.randint(5, 20)
            user_items = np.random.choice(items, n_interactions, replace=False)
            
            for item in user_items:
                interactions.append({
                    'user_id': user,
                    'content_id': item,
                    'rating': np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5]),
                    'interaction_type': np.random.choice(['view', 'like', 'favorite'], p=[0.5, 0.3, 0.2]),
                    'weight': np.random.uniform(3.0, 5.0)
                })
        
        self.interactions_df = pd.DataFrame(interactions)
        
        # Generate dummy content
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller', 'Animation']
        content = []
        
        for item in items:
            content.append({
                'id': item,
                'title': f'Movie {item}',
                'genres': [np.random.choice(genres), np.random.choice(genres)],
                'overview': f'This is the overview for movie {item}',
                'rating': np.random.uniform(3.0, 9.0),
                'popularity': np.random.uniform(0.1, 100.0),
                'release_date': f'2020-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}',
                'runtime': np.random.randint(80, 180),
                'language': np.random.choice(['en', 'hi', 'te', 'ta'])
            })
        
        self.content_df = pd.DataFrame(content)
    
    def build_user_item_matrix(self):
        """Build user-item interaction matrix"""
        if self.interactions_df.empty:
            return
            
        pivot_data = self.interactions_df.pivot_table(
            index='user_id', 
            columns='content_id', 
            values='weight', 
            fill_value=0
        )
        
        self.user_item_matrix = csr_matrix(pivot_data.values)
        self.user_ids = pivot_data.index.tolist()
        self.item_ids = pivot_data.columns.tolist()
        
    def build_content_features(self):
        """Build content-based features"""
        if self.content_df.empty:
            return
            
        # Text features
        content_text = self.content_df['title'].fillna('') + ' ' + self.content_df['overview'].fillna('')
        text_features = self.content_vectorizer.fit_transform(content_text)
        
        # Genre features
        all_genres = set()
        for genres in self.content_df['genres']:
            if isinstance(genres, list):
                all_genres.update(genres)
        
        genre_features = np.zeros((len(self.content_df), len(all_genres)))
        genre_list = list(all_genres)
        
        for i, genres in enumerate(self.content_df['genres']):
            if isinstance(genres, list):
                for genre in genres:
                    if genre in genre_list:
                        genre_features[i, genre_list.index(genre)] = 1
        
        # Numerical features
        numerical_features = self.content_df[['rating', 'popularity', 'runtime']].fillna(0).values
        numerical_features = self.scaler.fit_transform(numerical_features)
        
        # Combine features
        self.content_features = np.hstack([
            text_features.toarray(),
            genre_features,
            numerical_features
        ])
        
    def train_matrix_factorization(self):
        """Train matrix factorization models"""
        if self.user_item_matrix is None:
            return
            
        # NMF Model
        nmf_model = NMF(n_components=50, init='random', random_state=42, max_iter=200)
        self.models['nmf'] = nmf_model.fit(self.user_item_matrix)
        
        # SVD Model
        svd_model = TruncatedSVD(n_components=50, random_state=42)
        self.models['svd'] = svd_model.fit(self.user_item_matrix)
        
    def train_clustering_models(self):
        """Train user and item clustering models"""
        if self.user_item_matrix is None:
            return
            
        # User clustering
        user_kmeans = KMeans(n_clusters=min(10, len(self.user_ids)), random_state=42)
        user_clusters = user_kmeans.fit_predict(self.user_item_matrix)
        
        for i, user_id in enumerate(self.user_ids):
            self.user_clusters[user_id] = user_clusters[i]
        
        # Item clustering
        if self.content_features is not None:
            item_kmeans = KMeans(n_clusters=min(20, len(self.item_ids)), random_state=42)
            item_clusters = item_kmeans.fit_predict(self.content_features)
            
            for i, item_id in enumerate(self.item_ids):
                if i < len(item_clusters):
                    self.item_clusters[item_id] = item_clusters[i]
        
        self.models['user_kmeans'] = user_kmeans
        self.models['item_kmeans'] = item_kmeans if self.content_features is not None else None
        
    def build_user_profiles(self):
        """Build comprehensive user profiles"""
        if self.interactions_df.empty:
            return
            
        for user_id in self.interactions_df['user_id'].unique():
            user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
            
            # Genre preferences
            genre_prefs = defaultdict(float)
            for _, interaction in user_interactions.iterrows():
                content_id = interaction['content_id']
                weight = interaction['weight']
                
                content_row = self.content_df[self.content_df['id'] == content_id]
                if not content_row.empty:
                    genres = content_row.iloc[0]['genres']
                    if isinstance(genres, list):
                        for genre in genres:
                            genre_prefs[genre] += weight
            
            # Normalize genre preferences
            total_weight = sum(genre_prefs.values())
            if total_weight > 0:
                genre_prefs = {k: v/total_weight for k, v in genre_prefs.items()}
            
            # Rating patterns
            ratings = user_interactions['rating'].values
            avg_rating = np.mean(ratings)
            rating_std = np.std(ratings)
            
            # Temporal patterns
            interaction_types = user_interactions['interaction_type'].value_counts().to_dict()
            
            self.user_profiles[user_id] = {
                'genre_preferences': dict(genre_prefs),
                'avg_rating': avg_rating,
                'rating_std': rating_std,
                'interaction_patterns': interaction_types,
                'total_interactions': len(user_interactions),
                'cluster': self.user_clusters.get(user_id, 0)
            }
    
    def calculate_popularity_scores(self):
        """Calculate time-decayed popularity scores"""
        if self.content_df.empty:
            return
            
        current_date = datetime.now()
        
        for _, content in self.content_df.iterrows():
            content_id = content['id']
            
            # Base popularity
            base_popularity = content.get('popularity', 0)
            
            # Interaction-based popularity
            content_interactions = self.interactions_df[
                self.interactions_df['content_id'] == content_id
            ]
            
            interaction_score = len(content_interactions) * 0.1
            
            # Time decay for release date
            try:
                release_date = datetime.strptime(content['release_date'], '%Y-%m-%d')
                days_since_release = (current_date - release_date).days
                time_decay = max(0.1, 1 - (days_since_release / 365) * 0.1)
            except:
                time_decay = 0.5
            
            self.popularity_scores[content_id] = (base_popularity + interaction_score) * time_decay
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """Advanced collaborative filtering recommendations"""
        if self.user_item_matrix is None or user_id not in self.user_ids:
            return []
            
        user_idx = self.user_ids.index(user_id)
        user_vector = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Matrix factorization recommendations
        nmf_recs = []
        svd_recs = []
        
        if 'nmf' in self.models:
            nmf_user_features = self.models['nmf'].transform(self.user_item_matrix[user_idx])
            nmf_predictions = np.dot(nmf_user_features, self.models['nmf'].components_)
            nmf_recs = self._get_top_items(nmf_predictions.flatten(), user_vector, n_recommendations)
        
        if 'svd' in self.models:
            svd_user_features = self.models['svd'].transform(self.user_item_matrix[user_idx])
            svd_item_features = self.models['svd'].components_
            svd_predictions = np.dot(svd_user_features, svd_item_features)
            svd_recs = self._get_top_items(svd_predictions.flatten(), user_vector, n_recommendations)
        
        # User-based collaborative filtering
        user_similarities = []
        for i, other_user_id in enumerate(self.user_ids):
            if i != user_idx:
                other_vector = self.user_item_matrix[i].toarray().flatten()
                similarity = 1 - cosine(user_vector, other_vector)
                user_similarities.append((i, similarity))
        
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get recommendations from similar users
        similar_user_recs = []
        for similar_user_idx, similarity in user_similarities[:10]:
            similar_user_vector = self.user_item_matrix[similar_user_idx].toarray().flatten()
            for item_idx, rating in enumerate(similar_user_vector):
                if rating > 0 and user_vector[item_idx] == 0:
                    similar_user_recs.append((self.item_ids[item_idx], rating * similarity))
        
        # Aggregate similar user recommendations
        similar_user_scores = defaultdict(float)
        for item_id, score in similar_user_recs:
            similar_user_scores[item_id] += score
        
        similar_user_final = sorted(similar_user_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Combine all recommendations
        all_recommendations = {}
        
        # Weight different methods
        for item_id, score in nmf_recs[:20]:
            all_recommendations[item_id] = all_recommendations.get(item_id, 0) + score * 0.4
            
        for item_id, score in svd_recs[:20]:
            all_recommendations[item_id] = all_recommendations.get(item_id, 0) + score * 0.4
            
        for item_id, score in similar_user_final[:20]:
            all_recommendations[item_id] = all_recommendations.get(item_id, 0) + score * 0.2
        
        # Sort and return top recommendations
        final_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in final_recs[:n_recommendations]]
    
    def get_content_based_recommendations(self, user_id, n_recommendations=10):
        """Advanced content-based recommendations"""
        if user_id not in self.user_profiles or self.content_features is None:
            return []
            
        user_profile = self.user_profiles[user_id]
        
        # Build user preference vector
        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
        liked_items = user_interactions[user_interactions['weight'] >= 4]['content_id'].tolist()
        
        if not liked_items:
            return []
        
        # Calculate average feature vector for liked items
        liked_features = []
        for item_id in liked_items:
            item_idx = self.content_df[self.content_df['id'] == item_id].index
            if len(item_idx) > 0:
                liked_features.append(self.content_features[item_idx[0]])
        
        if not liked_features:
            return []
        
        user_preference_vector = np.mean(liked_features, axis=0)
        
        # Calculate similarities with all items
        similarities = []
        for i, item_id in enumerate(self.content_df['id']):
            if item_id not in liked_items:
                item_features = self.content_features[i]
                similarity = 1 - cosine(user_preference_vector, item_features)
                
                # Boost with genre preferences
                item_genres = self.content_df.iloc[i]['genres']
                genre_boost = 0
                if isinstance(item_genres, list):
                    for genre in item_genres:
                        genre_boost += user_profile['genre_preferences'].get(genre, 0)
                
                final_score = similarity * 0.7 + genre_boost * 0.3
                similarities.append((item_id, final_score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in similarities[:n_recommendations]]
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=10):
        """Hybrid recommendations combining multiple approaches"""
        collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations)
        content_recs = self.get_content_based_recommendations(user_id, n_recommendations)
        
        # Popularity-based recommendations
        popularity_recs = sorted(self.popularity_scores.items(), key=lambda x: x[1], reverse=True)
        popularity_recs = [item_id for item_id, _ in popularity_recs[:n_recommendations]]
        
        # Cluster-based recommendations
        cluster_recs = []
        if user_id in self.user_clusters:
            user_cluster = self.user_clusters[user_id]
            cluster_users = [uid for uid, cluster in self.user_clusters.items() if cluster == user_cluster]
            
            cluster_items = defaultdict(float)
            for cluster_user in cluster_users:
                if cluster_user != user_id:
                    user_interactions = self.interactions_df[
                        self.interactions_df['user_id'] == cluster_user
                    ]
                    for _, interaction in user_interactions.iterrows():
                        if interaction['weight'] >= 4:
                            cluster_items[interaction['content_id']] += interaction['weight']
            
            cluster_recs = [item_id for item_id, _ in sorted(cluster_items.items(), key=lambda x: x[1], reverse=True)]
        
        # Combine all recommendations with weights
        final_scores = defaultdict(float)
        
        # Weight collaborative filtering
        for i, item_id in enumerate(collab_recs):
            final_scores[item_id] += (len(collab_recs) - i) * 0.4
        
        # Weight content-based
        for i, item_id in enumerate(content_recs):
            final_scores[item_id] += (len(content_recs) - i) * 0.3
        
        # Weight popularity
        for i, item_id in enumerate(popularity_recs):
            final_scores[item_id] += (len(popularity_recs) - i) * 0.2
        
        # Weight cluster-based
        for i, item_id in enumerate(cluster_recs[:n_recommendations]):
            final_scores[item_id] += (n_recommendations - i) * 0.1
        
        # Sort and return final recommendations
        final_recs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in final_recs[:n_recommendations]]
    
    def _get_top_items(self, predictions, user_vector, n_items):
        """Get top items from predictions, excluding already rated items"""
        item_scores = []
        for i, score in enumerate(predictions):
            if i < len(self.item_ids) and user_vector[i] == 0:
                item_scores.append((self.item_ids[i], score))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_items]
    
    def train_all_models(self):
        """Train all recommendation models"""
        print("Loading data...")
        self.load_data_from_backend()
        
        print("Building matrices...")
        self.build_user_item_matrix()
        self.build_content_features()
        
        print("Training models...")
        self.train_matrix_factorization()
        self.train_clustering_models()
        
        print("Building profiles...")
        self.build_user_profiles()
        self.calculate_popularity_scores()
        
        self.last_update = datetime.now()
        print("Training completed!")
    
    def should_retrain(self):
        """Check if models should be retrained"""
        return (datetime.now() - self.last_update).seconds > 3600
    
    def get_model_stats(self):
        """Get model statistics"""
        return {
            'last_update': self.last_update.isoformat(),
            'num_users': len(self.user_ids) if self.user_ids else 0,
            'num_items': len(self.item_ids) if self.item_ids else 0,
            'num_interactions': len(self.interactions_df) if hasattr(self, 'interactions_df') else 0,
            'models_trained': list(self.models.keys()),
            'user_clusters': len(set(self.user_clusters.values())) if self.user_clusters else 0,
            'item_clusters': len(set(self.item_clusters.values())) if self.item_clusters else 0
        }

# Initialize the recommendation engine
recommender = AdvancedRecommendationEngine()

# Background training thread
def background_training():
    while True:
        try:
            if recommender.should_retrain():
                with recommender.update_lock:
                    recommender.train_all_models()
            time.sleep(300)  # Check every 5 minutes
        except Exception as e:
            print(f"Background training error: {e}")
            time.sleep(600)  # Wait 10 minutes on error

# Start background training
training_thread = threading.Thread(target=background_training, daemon=True)
training_thread.start()

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'ML Recommendation Service',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        n_recommendations = data.get('n_recommendations', 10)
        method = data.get('method', 'hybrid')
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Ensure models are trained
        if not recommender.models or recommender.should_retrain():
            with recommender.update_lock:
                recommender.train_all_models()
        
        # Get recommendations based on method
        if method == 'collaborative':
            recommendations = recommender.get_collaborative_recommendations(user_id, n_recommendations)
        elif method == 'content':
            recommendations = recommender.get_content_based_recommendations(user_id, n_recommendations)
        else:  # hybrid
            recommendations = recommender.get_hybrid_recommendations(user_id, n_recommendations)
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'method': method,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/similar_items/<int:item_id>', methods=['GET'])
def get_similar_items(item_id):
    try:
        n_similar = request.args.get('n_similar', 10, type=int)
        
        if recommender.content_features is None:
            return jsonify({'error': 'Content features not available'}), 400
        
        # Find item index
        item_row = recommender.content_df[recommender.content_df['id'] == item_id]
        if item_row.empty:
            return jsonify({'error': 'Item not found'}), 404
        
        item_idx = item_row.index[0]
        item_features = recommender.content_features[item_idx]
        
        # Calculate similarities
        similarities = []
        for i, other_item_id in enumerate(recommender.content_df['id']):
            if other_item_id != item_id:
                other_features = recommender.content_features[i]
                similarity = 1 - cosine(item_features, other_features)
                similarities.append((other_item_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_items = [item_id for item_id, _ in similarities[:n_similar]]
        
        return jsonify({
            'item_id': item_id,
            'similar_items': similar_items,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user_profile/<int:user_id>', methods=['GET'])
def get_user_profile(user_id):
    try:
        if user_id not in recommender.user_profiles:
            return jsonify({'error': 'User profile not found'}), 404
        
        profile = recommender.user_profiles[user_id].copy()
        profile['user_id'] = user_id
        profile['timestamp'] = datetime.now().isoformat()
        
        return jsonify(profile)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trending', methods=['GET'])
def get_trending():
    try:
        n_items = request.args.get('n_items', 20, type=int)
        
        # Get trending items based on recent interactions and popularity
        trending_items = sorted(
            recommender.popularity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_items]
        
        return jsonify({
            'trending_items': [item_id for item_id, _ in trending_items],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        return jsonify(recommender.get_model_stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_models():
    try:
        # Force retrain models
        with recommender.update_lock:
            recommender.train_all_models()
        
        return jsonify({
            'status': 'success',
            'message': 'Models retrained successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_recommend', methods=['POST'])
def batch_recommend():
    try:
        data = request.get_json()
        user_ids = data.get('user_ids', [])
        n_recommendations = data.get('n_recommendations', 10)
        method = data.get('method', 'hybrid')
        
        if not user_ids:
            return jsonify({'error': 'user_ids list is required'}), 400
        
        # Ensure models are trained
        if not recommender.models or recommender.should_retrain():
            with recommender.update_lock:
                recommender.train_all_models()
        
        batch_results = {}
        
        for user_id in user_ids:
            try:
                if method == 'collaborative':
                    recommendations = recommender.get_collaborative_recommendations(user_id, n_recommendations)
                elif method == 'content':
                    recommendations = recommender.get_content_based_recommendations(user_id, n_recommendations)
                else:  # hybrid
                    recommendations = recommender.get_hybrid_recommendations(user_id, n_recommendations)
                
                batch_results[user_id] = recommendations
                
            except Exception as e:
                batch_results[user_id] = {'error': str(e)}
        
        return jsonify({
            'results': batch_results,
            'method': method,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize models on startup
@app.before_first_request
def initialize_models():
    try:
        recommender.train_all_models()
    except Exception as e:
        print(f"Error initializing models: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)