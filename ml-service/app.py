#ml-service/app.py
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
from datetime import datetime
import redis
import json

app = Flask(__name__)
CORS(app)

# Configuration
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
MAIN_BACKEND_URL = os.environ.get('MAIN_BACKEND_URL', 'http://localhost:5000')
ML_SERVICE_PORT = int(os.environ.get('ML_SERVICE_PORT', 5001))

# Initialize Redis for caching
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except:
    redis_client = None

# Global ML Models
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
content_features = None
similarity_matrix = None
user_clusters = None
svd_model = None

class MLRecommendationEngine:
    def __init__(self):
        self.models = {}
        self.feature_cache = {}
    
    def fetch_content_data(self):
        """Fetch content data from main backend"""
        try:
            response = requests.get(f"{MAIN_BACKEND_URL}/api/ml/content-data")
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def fetch_user_interactions(self):
        """Fetch user interaction data from main backend"""
        try:
            response = requests.get(f"{MAIN_BACKEND_URL}/api/ml/user-interactions")
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def build_content_features(self, content_data):
        """Build content feature matrix"""
        global content_features, similarity_matrix
        
        # Prepare text features
        text_features = []
        for content in content_data:
            features = f"{' '.join(content.get('genres', []))} {content.get('overview', '')} {content.get('language', '')}"
            text_features.append(features)
        
        # TF-IDF vectorization
        content_features = tfidf_vectorizer.fit_transform(text_features)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(content_features)
        
        # Cache the model
        if redis_client:
            redis_client.setex('content_features_shape', 3600, json.dumps(content_features.shape))
            redis_client.setex('similarity_updated', 3600, datetime.now().isoformat())
    
    def build_user_model(self, user_data, content_data):
        """Build collaborative filtering model"""
        global svd_model, user_clusters
        
        # Create user-item matrix
        user_ids = list(set([u['user_id'] for u in user_data]))
        content_ids = list(set([u['content_id'] for u in user_data]))
        
        if not user_ids or not content_ids:
            return
        
        # Build matrix
        user_item_matrix = pd.DataFrame(0, index=user_ids, columns=content_ids)
        
        for interaction in user_data:
            user_id = interaction['user_id']
            content_id = interaction['content_id']
            
            if interaction['interaction_type'] == 'rated':
                user_item_matrix.loc[user_id, content_id] = interaction.get('value', 3)
            elif interaction['interaction_type'] == 'favorite':
                user_item_matrix.loc[user_id, content_id] = 5
            elif interaction['interaction_type'] == 'watched':
                user_item_matrix.loc[user_id, content_id] = 3
        
        # SVD for dimensionality reduction
        svd_model = TruncatedSVD(n_components=min(50, len(user_ids)-1))
        user_vectors = svd_model.fit_transform(user_item_matrix.fillna(0))
        
        # User clustering
        if len(user_ids) > 5:
            n_clusters = min(5, len(user_ids)//2)
            user_clusters = KMeans(n_clusters=n_clusters, random_state=42)
            user_clusters.fit(user_vectors)
        
        # Cache models
        if redis_client:
            redis_client.setex('user_model_updated', 3600, datetime.now().isoformat())
    
    def get_content_recommendations(self, content_id, n_recommendations=10):
        """Content-based recommendations"""
        if similarity_matrix is None:
            return []
        
        try:
            content_data = self.fetch_content_data()
            if not content_data:
                return []
            
            # Find content index
            content_index = next((i for i, c in enumerate(content_data) if c['id'] == content_id), None)
            if content_index is None:
                return []
            
            # Get similarity scores
            sim_scores = list(enumerate(similarity_matrix[content_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Return top recommendations
            recommendations = []
            for i, score in sim_scores[1:n_recommendations+1]:
                if i < len(content_data):
                    recommendations.append({
                        'content_id': content_data[i]['id'],
                        'score': float(score),
                        'reason': 'Similar content'
                    })
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Content recommendation error: {e}")
            return []
    
    def get_user_recommendations(self, user_id, n_recommendations=10):
        """Collaborative filtering recommendations"""
        if svd_model is None:
            return []
        
        try:
            user_data = self.fetch_user_interactions()
            content_data = self.fetch_content_data()
            
            if not user_data or not content_data:
                return []
            
            # Get user's interactions
            user_interactions = [u for u in user_data if u['user_id'] == user_id]
            if not user_interactions:
                return []
            
            # Find similar users
            user_ids = list(set([u['user_id'] for u in user_data]))
            if user_id not in user_ids:
                return []
            
            # Get recommendations based on similar users
            similar_users = self.find_similar_users(user_id, user_data)
            recommendations = []
            
            for similar_user in similar_users[:5]:
                similar_user_interactions = [u for u in user_data if u['user_id'] == similar_user['user_id']]
                
                for interaction in similar_user_interactions:
                    if interaction['interaction_type'] in ['favorite', 'rated'] and interaction.get('value', 0) >= 4:
                        # Check if user hasn't interacted with this content
                        if not any(u['content_id'] == interaction['content_id'] for u in user_interactions):
                            recommendations.append({
                                'content_id': interaction['content_id'],
                                'score': similar_user['similarity'] * interaction.get('value', 3),
                                'reason': f'Users with similar taste liked this'
                            })
            
            # Sort and return top recommendations
            recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logging.error(f"User recommendation error: {e}")
            return []
    
    def find_similar_users(self, user_id, user_data):
        """Find users with similar preferences"""
        # Simple similarity based on common interactions
        target_user_content = set([u['content_id'] for u in user_data if u['user_id'] == user_id])
        
        user_similarities = []
        for other_user_id in set([u['user_id'] for u in user_data]):
            if other_user_id == user_id:
                continue
            
            other_user_content = set([u['content_id'] for u in user_data if u['user_id'] == other_user_id])
            
            # Jaccard similarity
            intersection = len(target_user_content.intersection(other_user_content))
            union = len(target_user_content.union(other_user_content))
            
            if union > 0:
                similarity = intersection / union
                user_similarities.append({
                    'user_id': other_user_id,
                    'similarity': similarity
                })
        
        return sorted(user_similarities, key=lambda x: x['similarity'], reverse=True)
    
    def get_trending_predictions(self, content_data):
        """Predict trending content"""
        if not content_data:
            return []
        
        # Simple trending based on popularity and recency
        trending_scores = []
        for content in content_data:
            # Combine popularity, rating, and recency
            popularity = content.get('popularity', 0)
            rating = content.get('rating', 0)
            
            # Recency score (newer content gets higher score)
            try:
                release_date = datetime.fromisoformat(content.get('release_date', '1900-01-01'))
                days_since_release = (datetime.now() - release_date).days
                recency_score = max(0, 1 - days_since_release / 365)  # Decay over a year
            except:
                recency_score = 0
            
            trending_score = (popularity * 0.4) + (rating * 0.4) + (recency_score * 0.2)
            
            trending_scores.append({
                'content_id': content['id'],
                'score': trending_score,
                'reason': 'Trending prediction'
            })
        
        return sorted(trending_scores, key=lambda x: x['score'], reverse=True)[:20]

# Initialize ML engine
ml_engine = MLRecommendationEngine()

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "service": "ml-service"})

@app.route('/train', methods=['POST'])
def train_models():
    """Train/update ML models"""
    try:
        # Fetch data
        content_data = ml_engine.fetch_content_data()
        user_data = ml_engine.fetch_user_interactions()
        
        if not content_data:
            return jsonify({"error": "No content data available"}), 400
        
        # Build models
        ml_engine.build_content_features(content_data)
        
        if user_data:
            ml_engine.build_user_model(user_data, content_data)
        
        return jsonify({
            "message": "Models trained successfully",
            "content_count": len(content_data),
            "user_interactions": len(user_data) if user_data else 0,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Training error: {e}")
        return jsonify({"error": "Training failed"}), 500

@app.route('/recommendations/content/<int:content_id>')
def content_recommendations(content_id):
    """Get content-based recommendations"""
    try:
        n_recommendations = min(int(request.args.get('limit', 10)), 50)
        recommendations = ml_engine.get_content_recommendations(content_id, n_recommendations)
        
        return jsonify({
            "recommendations": recommendations,
            "content_id": content_id,
            "method": "content-based"
        })
        
    except Exception as e:
        logging.error(f"Content recommendation error: {e}")
        return jsonify({"error": "Recommendation failed"}), 500

@app.route('/recommendations/user/<int:user_id>')
def user_recommendations(user_id):
    """Get collaborative filtering recommendations"""
    try:
        n_recommendations = min(int(request.args.get('limit', 10)), 50)
        recommendations = ml_engine.get_user_recommendations(user_id, n_recommendations)
        
        return jsonify({
            "recommendations": recommendations,
            "user_id": user_id,
            "method": "collaborative-filtering"
        })
        
    except Exception as e:
        logging.error(f"User recommendation error: {e}")
        return jsonify({"error": "Recommendation failed"}), 500

@app.route('/recommendations/hybrid/<int:user_id>')
def hybrid_recommendations(user_id):
    """Get hybrid recommendations"""
    try:
        n_recommendations = min(int(request.args.get('limit', 20)), 50)
        
        # Get user's recent interactions
        user_data = ml_engine.fetch_user_interactions()
        user_interactions = [u for u in user_data if u['user_id'] == user_id] if user_data else []
        
        # Get content-based recommendations from user's favorites
        content_recs = []
        for interaction in user_interactions[-5:]:  # Last 5 interactions
            if interaction['interaction_type'] in ['favorite', 'rated']:
                content_recs.extend(ml_engine.get_content_recommendations(interaction['content_id'], 5))
        
        # Get collaborative recommendations
        collab_recs = ml_engine.get_user_recommendations(user_id, 10)
        
        # Combine and score
        combined_recs = {}
        for rec in content_recs:
            combined_recs[rec['content_id']] = combined_recs.get(rec['content_id'], 0) + rec['score'] * 0.6
        
        for rec in collab_recs:
            combined_recs[rec['content_id']] = combined_recs.get(rec['content_id'], 0) + rec['score'] * 0.4
        
        # Sort and format
        hybrid_recommendations = [
            {
                'content_id': content_id,
                'score': score,
                'reason': 'Hybrid recommendation'
            }
            for content_id, score in sorted(combined_recs.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return jsonify({
            "recommendations": hybrid_recommendations[:n_recommendations],
            "user_id": user_id,
            "method": "hybrid"
        })
        
    except Exception as e:
        logging.error(f"Hybrid recommendation error: {e}")
        return jsonify({"error": "Recommendation failed"}), 500

@app.route('/trending')
def trending_predictions():
    """Get trending content predictions"""
    try:
        content_data = ml_engine.fetch_content_data()
        trending = ml_engine.get_trending_predictions(content_data)
        
        return jsonify({
            "trending": trending,
            "method": "ml-prediction"
        })
        
    except Exception as e:
        logging.error(f"Trending prediction error: {e}")
        return jsonify({"error": "Trending prediction failed"}), 500

@app.route('/similarity/<int:content_id>')
def content_similarity(content_id):
    """Get content similarity scores"""
    try:
        similar_content = ml_engine.get_content_recommendations(content_id, 20)
        
        return jsonify({
            "similar_content": similar_content,
            "content_id": content_id
        })
        
    except Exception as e:
        logging.error(f"Similarity error: {e}")
        return jsonify({"error": "Similarity calculation failed"}), 500

@app.route('/models/status')
def model_status():
    """Get ML models status"""
    try:
        status = {
            "content_features": content_features is not None,
            "similarity_matrix": similarity_matrix is not None,
            "svd_model": svd_model is not None,
            "user_clusters": user_clusters is not None,
            "vectorizer_vocab_size": len(tfidf_vectorizer.vocabulary_) if hasattr(tfidf_vectorizer, 'vocabulary_') else 0
        }
        
        if redis_client:
            status["cache_status"] = "connected"
            status["last_training"] = redis_client.get('user_model_updated')
        else:
            status["cache_status"] = "disconnected"
        
        return jsonify(status)
        
    except Exception as e:
        logging.error(f"Status error: {e}")
        return jsonify({"error": "Status check failed"}), 500

if __name__ == '__main__':
    # Initial training
    try:
        content_data = ml_engine.fetch_content_data()
        if content_data:
            ml_engine.build_content_features(content_data)
        
        user_data = ml_engine.fetch_user_interactions()
        if user_data and content_data:
            ml_engine.build_user_model(user_data, content_data)
            
        print("Initial ML models trained successfully")
    except Exception as e:
        print(f"Initial training failed: {e}")
    
    app.run(host='0.0.0.0', port=ML_SERVICE_PORT, debug=os.environ.get('FLASK_ENV') == 'development')