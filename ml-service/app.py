from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
import json
from datetime import datetime, timedelta
import os
from functools import wraps
import logging
from threading import Thread
import time
import pickle
import sqlite3
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv('BACKEND_URL', 'https://backend-app-970m.onrender.com')
MODEL_UPDATE_INTERVAL = 3600

class AdvancedRecommendationML:
    def __init__(self):
        self.content_vectors = None
        self.user_profiles = {}
        self.item_factors = None
        self.user_factors = None
        self.content_clusters = None
        self.genre_weights = {}
        self.popularity_scores = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.scaler = StandardScaler()
        self.nmf_model = NMF(n_components=50, random_state=42)
        self.kmeans_model = KMeans(n_clusters=10, random_state=42)
        self.last_update = datetime.now()
        
    def fetch_data_from_backend(self):
        """Fetch data from main backend"""
        try:
            response = requests.get(f"{BACKEND_URL}/api/ml/data", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Backend API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch data from backend: {e}")
            return self.load_cached_data()
    
    def load_cached_data(self):
        """Load cached data from local storage"""
        try:
            with open('cached_data.json', 'r') as f:
                return json.load(f)
        except:
            return {'users': [], 'content': [], 'interactions': []}
    
    def save_cached_data(self, data):
        """Save data to local cache"""
        try:
            with open('cached_data.json', 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save cached data: {e}")
    
    def preprocess_content(self, content_data):
        """Preprocess content for feature extraction"""
        processed = []
        for item in content_data:
            features = []
            if item.get('title'):
                features.append(item['title'])
            if item.get('overview'):
                features.append(item['overview'])
            if item.get('genres'):
                genre_names = [str(g) for g in item['genres']]
                features.extend(genre_names)
            processed.append(' '.join(features))
        return processed
    
    def build_content_features(self, content_data):
        """Build content feature matrix"""
        if not content_data:
            return None
        
        text_features = self.preprocess_content(content_data)
        content_vectors = self.tfidf_vectorizer.fit_transform(text_features)
        
        additional_features = []
        for item in content_data:
            features = [
                item.get('rating', 0) / 10.0,
                item.get('popularity', 0) / 100.0,
                len(item.get('genres', [])) / 10.0,
                item.get('runtime', 0) / 200.0 if item.get('runtime') else 0
            ]
            additional_features.append(features)
        
        additional_features = self.scaler.fit_transform(additional_features)
        
        from scipy.sparse import hstack
        self.content_vectors = hstack([content_vectors, additional_features])
        
        return self.content_vectors
    
    def build_user_item_matrix(self, users_data, interactions_data):
        """Build user-item interaction matrix"""
        user_ids = [u['id'] for u in users_data]
        content_ids = list(set([i['content_id'] for i in interactions_data]))
        
        user_item_matrix = np.zeros((len(user_ids), len(content_ids)))
        
        user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
        content_id_map = {cid: idx for idx, cid in enumerate(content_ids)}
        
        for interaction in interactions_data:
            if interaction['user_id'] in user_id_map and interaction['content_id'] in content_id_map:
                user_idx = user_id_map[interaction['user_id']]
                content_idx = content_id_map[interaction['content_id']]
                
                rating = interaction.get('rating', 0)
                if interaction['interaction_type'] == 'favorite':
                    rating = max(rating, 5)
                elif interaction['interaction_type'] == 'like':
                    rating = max(rating, 4)
                elif interaction['interaction_type'] == 'view':
                    rating = max(rating, 3)
                
                user_item_matrix[user_idx, content_idx] = rating
        
        return user_item_matrix, user_id_map, content_id_map
    
    def matrix_factorization(self, user_item_matrix):
        """Perform matrix factorization using NMF"""
        try:
            self.user_factors = self.nmf_model.fit_transform(user_item_matrix)
            self.item_factors = self.nmf_model.components_
            return True
        except Exception as e:
            logger.error(f"Matrix factorization failed: {e}")
            return False
    
    def cluster_content(self, content_vectors):
        """Cluster content using K-means"""
        try:
            if content_vectors is not None:
                self.content_clusters = self.kmeans_model.fit_predict(content_vectors.toarray())
                return True
        except Exception as e:
            logger.error(f"Content clustering failed: {e}")
        return False
    
    def calculate_genre_preferences(self, user_interactions, content_data):
        """Calculate user genre preferences"""
        user_genre_scores = defaultdict(lambda: defaultdict(float))
        
        content_dict = {item['id']: item for item in content_data}
        
        for interaction in user_interactions:
            user_id = interaction['user_id']
            content_id = interaction['content_id']
            
            if content_id in content_dict:
                content = content_dict[content_id]
                genres = content.get('genres', [])
                
                score = 1.0
                if interaction['interaction_type'] == 'favorite':
                    score = 2.0
                elif interaction['interaction_type'] == 'like':
                    score = 1.5
                elif interaction.get('rating'):
                    score = interaction['rating'] / 5.0
                
                for genre in genres:
                    user_genre_scores[user_id][genre] += score
        
        return dict(user_genre_scores)
    
    def train_models(self):
        """Train all ML models"""
        data = self.fetch_data_from_backend()
        if not data:
            logger.warning("No data available for training")
            return False
        
        self.save_cached_data(data)
        
        users_data = data.get('users', [])
        content_data = data.get('content', [])
        interactions_data = data.get('interactions', [])
        
        if not users_data or not content_data or not interactions_data:
            logger.warning("Insufficient data for training")
            return False
        
        logger.info(f"Training with {len(users_data)} users, {len(content_data)} content items, {len(interactions_data)} interactions")
        
        # Build content features
        content_vectors = self.build_content_features(content_data)
        if content_vectors is not None:
            self.cluster_content(content_vectors)
        
        # Build user-item matrix
        user_item_matrix, user_id_map, content_id_map = self.build_user_item_matrix(users_data, interactions_data)
        
        # Matrix factorization
        self.matrix_factorization(user_item_matrix)
        
        # Calculate genre preferences
        self.user_genre_preferences = self.calculate_genre_preferences(interactions_data, content_data)
        
        # Store mappings
        self.user_id_map = user_id_map
        self.content_id_map = content_id_map
        self.content_data = content_data
        
        # Calculate popularity scores
        content_popularity = defaultdict(int)
        for interaction in interactions_data:
            weight = 1
            if interaction['interaction_type'] == 'favorite':
                weight = 3
            elif interaction['interaction_type'] == 'like':
                weight = 2
            content_popularity[interaction['content_id']] += weight
        
        self.popularity_scores = dict(content_popularity)
        
        self.last_update = datetime.now()
        logger.info("Model training completed successfully")
        return True
    
    def get_content_based_recommendations(self, user_id, n_recommendations=10):
        """Content-based recommendations"""
        if self.content_vectors is None:
            return []
        
        user_interactions = [i for i in self.interactions_data if i['user_id'] == user_id]
        if not user_interactions:
            return self.get_popularity_based_recommendations(n_recommendations)
        
        user_content_ids = [i['content_id'] for i in user_interactions]
        user_content_indices = [self.content_id_map.get(cid) for cid in user_content_ids if cid in self.content_id_map]
        
        if not user_content_indices:
            return []
        
        user_profile = np.mean(self.content_vectors[user_content_indices], axis=0)
        content_similarities = cosine_similarity(user_profile, self.content_vectors).flatten()
        
        similar_indices = np.argsort(content_similarities)[::-1]
        
        recommendations = []
        for idx in similar_indices:
            content_id = list(self.content_id_map.keys())[list(self.content_id_map.values()).index(idx)]
            if content_id not in user_content_ids and len(recommendations) < n_recommendations:
                content_item = next((c for c in self.content_data if c['id'] == content_id), None)
                if content_item:
                    recommendations.append({
                        'content_id': content_id,
                        'score': float(content_similarities[idx]),
                        'method': 'content_based',
                        'content': content_item
                    })
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """Collaborative filtering recommendations"""
        if self.user_factors is None or self.item_factors is None:
            return []
        
        if user_id not in self.user_id_map:
            return []
        
        user_idx = self.user_id_map[user_id]
        user_vector = self.user_factors[user_idx]
        
        scores = np.dot(user_vector, self.item_factors)
        
        user_interactions = [i for i in self.interactions_data if i['user_id'] == user_id]
        user_content_ids = [i['content_id'] for i in user_interactions]
        
        recommendations = []
        for content_idx, score in enumerate(scores):
            content_id = list(self.content_id_map.keys())[list(self.content_id_map.values()).index(content_idx)]
            if content_id not in user_content_ids:
                content_item = next((c for c in self.content_data if c['id'] == content_id), None)
                if content_item:
                    recommendations.append({
                        'content_id': content_id,
                        'score': float(score),
                        'method': 'collaborative',
                        'content': content_item
                    })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:n_recommendations]
    
    def get_genre_based_recommendations(self, user_id, n_recommendations=10):
        """Genre-based recommendations"""
        if user_id not in self.user_genre_preferences:
            return []
        
        user_genres = self.user_genre_preferences[user_id]
        
        recommendations = []
        for content in self.content_data:
            if content['genres']:
                score = 0
                for genre in content['genres']:
                    score += user_genres.get(genre, 0)
                
                if score > 0:
                    recommendations.append({
                        'content_id': content['id'],
                        'score': score,
                        'method': 'genre_based',
                        'content': content
                    })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:n_recommendations]
    
    def get_popularity_based_recommendations(self, n_recommendations=10):
        """Popularity-based recommendations"""
        popularity_recs = []
        for content in self.content_data:
            popularity = self.popularity_scores.get(content['id'], 0)
            popularity_recs.append({
                'content_id': content['id'],
                'score': popularity,
                'method': 'popularity',
                'content': content
            })
        
        return sorted(popularity_recs, key=lambda x: x['score'], reverse=True)[:n_recommendations]
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=10):
        """Hybrid recommendations combining multiple methods"""
        content_recs = self.get_content_based_recommendations(user_id, n_recommendations//2)
        collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations//2)
        genre_recs = self.get_genre_based_recommendations(user_id, n_recommendations//2)
        
        # Combine and weight recommendations
        all_recs = {}
        
        for rec in content_recs:
            all_recs[rec['content_id']] = rec
            all_recs[rec['content_id']]['combined_score'] = rec['score'] * 0.4
        
        for rec in collab_recs:
            if rec['content_id'] in all_recs:
                all_recs[rec['content_id']]['combined_score'] += rec['score'] * 0.4
            else:
                all_recs[rec['content_id']] = rec
                all_recs[rec['content_id']]['combined_score'] = rec['score'] * 0.4
        
        for rec in genre_recs:
            if rec['content_id'] in all_recs:
                all_recs[rec['content_id']]['combined_score'] += rec['score'] * 0.2
            else:
                all_recs[rec['content_id']] = rec
                all_recs[rec['content_id']]['combined_score'] = rec['score'] * 0.2
        
        # Sort by combined score
        final_recs = sorted(all_recs.values(), key=lambda x: x['combined_score'], reverse=True)
        
        return final_recs[:n_recommendations]

ml_engine = AdvancedRecommendationML()

def model_update_worker():
    """Background worker to update models periodically"""
    while True:
        try:
            time.sleep(MODEL_UPDATE_INTERVAL)
            logger.info("Starting scheduled model update")
            ml_engine.train_models()
            logger.info("Scheduled model update completed")
        except Exception as e:
            logger.error(f"Error in model update worker: {e}")

def requires_trained_model(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(ml_engine, 'content_data') or not ml_engine.content_data:
            return jsonify({'error': 'Models not trained yet', 'recommendations': []}), 503
        return f(*args, **kwargs)
    return decorated_function

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'last_model_update': ml_engine.last_update.isoformat(),
        'models_trained': hasattr(ml_engine, 'content_data') and bool(ml_engine.content_data)
    })

@app.route('/train', methods=['POST'])
def train_models():
    """Manually trigger model training"""
    try:
        success = ml_engine.train_models()
        if success:
            return jsonify({'status': 'success', 'message': 'Models trained successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Model training failed'}), 500
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/recommend', methods=['POST'])
@requires_trained_model
def get_recommendations():
    """Get personalized recommendations"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        n_recommendations = data.get('n_recommendations', 10)
        method = data.get('method', 'hybrid')
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        if method == 'content':
            recommendations = ml_engine.get_content_based_recommendations(user_id, n_recommendations)
        elif method == 'collaborative':
            recommendations = ml_engine.get_collaborative_recommendations(user_id, n_recommendations)
        elif method == 'genre':
            recommendations = ml_engine.get_genre_based_recommendations(user_id, n_recommendations)
        elif method == 'popularity':
            recommendations = ml_engine.get_popularity_based_recommendations(n_recommendations)
        else:
            recommendations = ml_engine.get_hybrid_recommendations(user_id, n_recommendations)
        
        return jsonify({
            'recommendations': recommendations,
            'user_id': user_id,
            'method': method,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({'error': str(e), 'recommendations': []}), 500

@app.route('/similar/<int:content_id>')
@requires_trained_model
def get_similar_content(content_id):
    """Get similar content recommendations"""
    try:
        if ml_engine.content_vectors is None:
            return jsonify({'similar_content': []}), 503
        
        if content_id not in ml_engine.content_id_map:
            return jsonify({'similar_content': []}), 404
        
        content_idx = ml_engine.content_id_map[content_id]
        content_vector = ml_engine.content_vectors[content_idx]
        
        similarities = cosine_similarity(content_vector, ml_engine.content_vectors).flatten()
        similar_indices = np.argsort(similarities)[::-1][1:11]  # Exclude self
        
        similar_content = []
        for idx in similar_indices:
            similar_content_id = list(ml_engine.content_id_map.keys())[list(ml_engine.content_id_map.values()).index(idx)]
            content_item = next((c for c in ml_engine.content_data if c['id'] == similar_content_id), None)
            if content_item:
                similar_content.append({
                    'content_id': similar_content_id,
                    'similarity_score': float(similarities[idx]),
                    'content': content_item
                })
        
        return jsonify({'similar_content': similar_content})
    
    except Exception as e:
        logger.error(f"Similar content error: {e}")
        return jsonify({'error': str(e), 'similar_content': []}), 500

@app.route('/user-profile/<int:user_id>')
@requires_trained_model
def get_user_profile(user_id):
    """Get user preference profile"""
    try:
        profile = {
            'user_id': user_id,
            'genre_preferences': ml_engine.user_genre_preferences.get(user_id, {}),
            'has_interactions': user_id in ml_engine.user_id_map,
            'recommendation_methods': ['hybrid', 'content', 'collaborative', 'genre', 'popularity']
        }
        
        return jsonify(profile)
    
    except Exception as e:
        logger.error(f"User profile error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/trending')
@requires_trained_model
def get_trending():
    """Get trending content based on recent interactions"""
    try:
        trending = ml_engine.get_popularity_based_recommendations(20)
        return jsonify({'trending': trending})
    
    except Exception as e:
        logger.error(f"Trending error: {e}")
        return jsonify({'error': str(e), 'trending': []}), 500

@app.route('/stats')
@requires_trained_model
def get_stats():
    """Get ML service statistics"""
    try:
        stats = {
            'total_users': len(ml_engine.user_id_map) if hasattr(ml_engine, 'user_id_map') else 0,
            'total_content': len(ml_engine.content_data) if hasattr(ml_engine, 'content_data') else 0,
            'total_interactions': len(ml_engine.interactions_data) if hasattr(ml_engine, 'interactions_data') else 0,
            'model_features': {
                'content_vectors': ml_engine.content_vectors.shape if ml_engine.content_vectors is not None else None,
                'user_factors': ml_engine.user_factors.shape if ml_engine.user_factors is not None else None,
                'item_factors': ml_engine.item_factors.shape if ml_engine.item_factors is not None else None,
                'content_clusters': len(set(ml_engine.content_clusters)) if ml_engine.content_clusters is not None else 0
            },
            'last_update': ml_engine.last_update.isoformat()
        }
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting ML Recommendation Service")
    
    # Initial model training
    logger.info("Performing initial model training")
    ml_engine.train_models()
    
    # Start background update worker
    update_thread = Thread(target=model_update_worker, daemon=True)
    update_thread.start()
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), debug=False)