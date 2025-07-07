# ml-service/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import requests
from datetime import datetime
import asyncio
import aiohttp
import json
from functools import lru_cache
import redis
from threading import Thread
import time

app = Flask(__name__)
CORS(app)

# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'https://backend-app-970m.onrender.com')
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')

# Initialize Redis for caching
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
except:
    redis_client = None

class AdvancedRecommendationEngine:
    def __init__(self):
        self.content_features = None
        self.user_profiles = {}
        self.similarity_matrix = None
        self.svd_model = None
        self.scaler = MinMaxScaler()
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.genre_weights = {
            'Action': 1.2, 'Adventure': 1.1, 'Comedy': 1.3, 'Drama': 1.0,
            'Horror': 1.4, 'Romance': 1.1, 'Sci-Fi': 1.2, 'Thriller': 1.3
        }
        
    def load_data(self):
        """Load and preprocess data from backend"""
        try:
            # Get content data
            response = requests.get(f"{BACKEND_URL}/api/content/all", timeout=10)
            if response.status_code == 200:
                content_data = response.json()
                self.content_df = pd.DataFrame(content_data)
                
            # Get user interactions
            response = requests.get(f"{BACKEND_URL}/api/interactions/all", timeout=10)
            if response.status_code == 200:
                interactions_data = response.json()
                self.interactions_df = pd.DataFrame(interactions_data)
                
            return True
        except:
            # Use fallback data if backend unavailable
            self.content_df = pd.DataFrame()
            self.interactions_df = pd.DataFrame()
            return False
    
    def build_content_features(self):
        """Build advanced content feature matrix"""
        if self.content_df.empty:
            return
            
        features = []
        for _, content in self.content_df.iterrows():
            # Combine text features
            text_features = f"{content.get('title', '')} {content.get('overview', '')}"
            
            # Add genre information with weights
            if content.get('genres'):
                genres = content['genres'] if isinstance(content['genres'], list) else []
                weighted_genres = []
                for genre in genres:
                    genre_name = str(genre)
                    weight = self.genre_weights.get(genre_name, 1.0)
                    weighted_genres.extend([genre_name] * int(weight * 3))
                text_features += " " + " ".join(weighted_genres)
            
            # Add popularity and rating boost
            popularity = content.get('popularity', 0)
            rating = content.get('rating', 0)
            if popularity > 50:
                text_features += " popular trending"
            if rating > 7:
                text_features += " highly_rated excellent"
                
            features.append(text_features)
        
        # Create TF-IDF matrix
        self.content_features = self.tfidf.fit_transform(features)
        self.similarity_matrix = cosine_similarity(self.content_features)
        
        # Apply SVD for dimensionality reduction
        self.svd_model = TruncatedSVD(n_components=100, random_state=42)
        self.reduced_features = self.svd_model.fit_transform(self.content_features)
        
    def build_user_profiles(self):
        """Build user preference profiles"""
        if self.interactions_df.empty:
            return
            
        for user_id in self.interactions_df['user_id'].unique():
            user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
            
            # Weight interactions by type and recency
            weights = {
                'favorite': 3.0, 'like': 2.0, 'view': 1.0,
                'wishlist': 2.5, 'share': 1.5
            }
            
            profile = {
                'preferences': {},
                'genre_scores': {},
                'avg_rating': 0,
                'interaction_count': len(user_interactions)
            }
            
            total_weight = 0
            for _, interaction in user_interactions.iterrows():
                weight = weights.get(interaction.get('interaction_type', 'view'), 1.0)
                
                # Add recency boost
                days_ago = (datetime.now() - pd.to_datetime(interaction.get('created_at', datetime.now()))).days
                recency_boost = max(0.1, 1 - (days_ago / 365))
                final_weight = weight * recency_boost
                
                content_id = interaction.get('content_id')
                if content_id in self.content_df.index:
                    content = self.content_df.loc[content_id]
                    
                    # Update genre preferences
                    if content.get('genres'):
                        for genre in content['genres']:
                            genre_name = str(genre)
                            profile['genre_scores'][genre_name] = profile['genre_scores'].get(genre_name, 0) + final_weight
                
                total_weight += final_weight
            
            # Normalize scores
            if total_weight > 0:
                for genre in profile['genre_scores']:
                    profile['genre_scores'][genre] /= total_weight
            
            self.user_profiles[user_id] = profile
    
    @lru_cache(maxsize=1000)
    def get_collaborative_recommendations(self, user_id, limit=20):
        """Advanced collaborative filtering"""
        if user_id not in self.user_profiles:
            return []
            
        user_profile = self.user_profiles[user_id]
        user_interactions = set(self.interactions_df[self.interactions_df['user_id'] == user_id]['content_id'])
        
        # Find similar users
        similar_users = []
        for other_user_id, other_profile in self.user_profiles.items():
            if other_user_id == user_id:
                continue
                
            # Calculate similarity based on genre preferences
            similarity = self.calculate_profile_similarity(user_profile, other_profile)
            if similarity > 0.3:
                similar_users.append((other_user_id, similarity))
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)
        
        # Get recommendations from similar users
        recommendations = {}
        for similar_user_id, similarity in similar_users[:10]:
            similar_interactions = self.interactions_df[
                (self.interactions_df['user_id'] == similar_user_id) &
                (self.interactions_df['interaction_type'].isin(['favorite', 'like']))
            ]
            
            for _, interaction in similar_interactions.iterrows():
                content_id = interaction['content_id']
                if content_id not in user_interactions:
                    score = similarity * 2.0  # Base score from similarity
                    
                    # Add interaction type weight
                    if interaction['interaction_type'] == 'favorite':
                        score *= 1.5
                    
                    recommendations[content_id] = recommendations.get(content_id, 0) + score
        
        # Sort and return top recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [content_id for content_id, _ in sorted_recs[:limit]]
    
    def calculate_profile_similarity(self, profile1, profile2):
        """Calculate similarity between user profiles"""
        genres1 = set(profile1['genre_scores'].keys())
        genres2 = set(profile2['genre_scores'].keys())
        common_genres = genres1 & genres2
        
        if not common_genres:
            return 0
        
        # Calculate cosine similarity on genre preferences
        vector1 = [profile1['genre_scores'].get(genre, 0) for genre in common_genres]
        vector2 = [profile2['genre_scores'].get(genre, 0) for genre in common_genres]
        
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = sum(a * a for a in vector1) ** 0.5
        magnitude2 = sum(b * b for b in vector2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def get_content_based_recommendations(self, user_id, limit=20):
        """Enhanced content-based recommendations"""
        if user_id not in self.user_profiles or self.similarity_matrix is None:
            return []
        
        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
        liked_content = user_interactions[user_interactions['interaction_type'].isin(['favorite', 'like'])]
        
        if liked_content.empty:
            return []
        
        # Get content indices
        content_indices = []
        for content_id in liked_content['content_id']:
            if content_id in self.content_df.index:
                content_indices.append(self.content_df.index.get_loc(content_id))
        
        if not content_indices:
            return []
        
        # Calculate average similarity
        avg_similarity = np.mean(self.similarity_matrix[content_indices], axis=0)
        
        # Get top similar content
        similar_indices = np.argsort(avg_similarity)[::-1]
        
        # Filter out already interacted content
        user_content_ids = set(user_interactions['content_id'])
        recommendations = []
        
        for idx in similar_indices:
            if len(recommendations) >= limit:
                break
            content_id = self.content_df.iloc[idx]['id']
            if content_id not in user_content_ids:
                recommendations.append(content_id)
        
        return recommendations
    
    def get_hybrid_recommendations(self, user_id, limit=20):
        """Hybrid recommendation combining multiple approaches"""
        # Get recommendations from different methods
        collab_recs = self.get_collaborative_recommendations(user_id, limit)
        content_recs = self.get_content_based_recommendations(user_id, limit)
        
        # Combine with weights
        combined_scores = {}
        
        # Collaborative filtering (weight: 0.6)
        for i, content_id in enumerate(collab_recs):
            score = (len(collab_recs) - i) / len(collab_recs) * 0.6
            combined_scores[content_id] = combined_scores.get(content_id, 0) + score
        
        # Content-based (weight: 0.4)
        for i, content_id in enumerate(content_recs):
            score = (len(content_recs) - i) / len(content_recs) * 0.4
            combined_scores[content_id] = combined_scores.get(content_id, 0) + score
        
        # Add diversity bonus
        for content_id in combined_scores:
            if content_id in self.content_df.index:
                content = self.content_df.loc[content_id]
                # Boost less popular but high-rated content
                if content.get('popularity', 0) < 20 and content.get('rating', 0) > 7:
                    combined_scores[content_id] *= 1.1
        
        # Sort and return
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [content_id for content_id, _ in sorted_recs[:limit]]

# Initialize recommendation engine
recommender = AdvancedRecommendationEngine()

def update_models():
    """Update ML models with fresh data"""
    try:
        print("Updating ML models...")
        if recommender.load_data():
            recommender.build_content_features()
            recommender.build_user_profiles()
            print("Models updated successfully")
        else:
            print("Failed to load data from backend")
    except Exception as e:
        print(f"Error updating models: {e}")

def cache_get(key):
    """Get from cache"""
    if redis_client:
        try:
            value = redis_client.get(key)
            return json.loads(value) if value else None
        except:
            pass
    return None

def cache_set(key, value, expire=3600):
    """Set cache with expiration"""
    if redis_client:
        try:
            redis_client.setex(key, expire, json.dumps(value))
        except:
            pass

# API Routes
@app.route('/recommend', methods=['POST'])
def recommend():
    """Get personalized recommendations"""
    data = request.get_json()
    user_id = data.get('user_id')
    limit = data.get('limit', 20)
    
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
    
    # Check cache first
    cache_key = f"recommendations:{user_id}:{limit}"
    cached_result = cache_get(cache_key)
    if cached_result:
        return jsonify(cached_result)
    
    try:
        # Get hybrid recommendations
        recommendations = recommender.get_hybrid_recommendations(user_id, limit)
        
        # Add content details
        detailed_recs = []
        for content_id in recommendations:
            if content_id in recommender.content_df.index:
                content = recommender.content_df.loc[content_id]
                detailed_recs.append({
                    'id': content_id,
                    'title': content.get('title', 'Unknown'),
                    'overview': content.get('overview', ''),
                    'rating': content.get('rating', 0),
                    'poster_path': content.get('poster_path', ''),
                    'genres': content.get('genres', [])
                })
        
        result = {
            'recommendations': detailed_recs,
            'algorithm': 'hybrid',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache result
        cache_set(cache_key, result, 1800)  # 30 minutes
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/similar/<int:content_id>')
def get_similar_content(content_id):
    """Get similar content"""
    limit = request.args.get('limit', 10, type=int)
    
    cache_key = f"similar:{content_id}:{limit}"
    cached_result = cache_get(cache_key)
    if cached_result:
        return jsonify(cached_result)
    
    try:
        if content_id not in recommender.content_df.index:
            return jsonify({'error': 'Content not found'}), 404
        
        content_idx = recommender.content_df.index.get_loc(content_id)
        similarities = recommender.similarity_matrix[content_idx]
        similar_indices = np.argsort(similarities)[::-1][1:limit+1]
        
        similar_content = []
        for idx in similar_indices:
            content = recommender.content_df.iloc[idx]
            similar_content.append({
                'id': content['id'],
                'title': content.get('title', 'Unknown'),
                'similarity_score': float(similarities[idx]),
                'poster_path': content.get('poster_path', ''),
                'rating': content.get('rating', 0)
            })
        
        result = {'similar_content': similar_content}
        cache_set(cache_key, result, 3600)  # 1 hour
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trending')
def get_trending():
    """Get trending content based on ML analysis"""
    limit = request.args.get('limit', 20, type=int)
    
    cache_key = f"trending:{limit}"
    cached_result = cache_get(cache_key)
    if cached_result:
        return jsonify(cached_result)
    
    try:
        # Calculate trending score based on recent interactions and popularity
        trending_scores = {}
        
        # Recent interactions (last 7 days)
        recent_date = datetime.now() - pd.Timedelta(days=7)
        recent_interactions = recommender.interactions_df[
            pd.to_datetime(recommender.interactions_df['created_at']) > recent_date
        ]
        
        # Calculate interaction velocity
        for content_id in recent_interactions['content_id'].unique():
            content_interactions = recent_interactions[recent_interactions['content_id'] == content_id]
            
            # Weight by interaction type
            weights = {'favorite': 3, 'like': 2, 'view': 1, 'share': 2.5}
            total_score = sum(weights.get(interaction['interaction_type'], 1) 
                            for _, interaction in content_interactions.iterrows())
            
            # Add popularity boost
            if content_id in recommender.content_df.index:
                content = recommender.content_df.loc[content_id]
                popularity_boost = min(content.get('popularity', 0) / 100, 1.0)
                total_score *= (1 + popularity_boost)
            
            trending_scores[content_id] = total_score
        
        # Sort by trending score
        sorted_trending = sorted(trending_scores.items(), key=lambda x: x[1], reverse=True)
        
        trending_content = []
        for content_id, score in sorted_trending[:limit]:
            if content_id in recommender.content_df.index:
                content = recommender.content_df.loc[content_id]
                trending_content.append({
                    'id': content_id,
                    'title': content.get('title', 'Unknown'),
                    'trending_score': float(score),
                    'poster_path': content.get('poster_path', ''),
                    'rating': content.get('rating', 0)
                })
        
        result = {'trending': trending_content}
        cache_set(cache_key, result, 1800)  # 30 minutes
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': recommender.content_features is not None,
        'cache_available': redis_client is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/update-models', methods=['POST'])
def update_models_endpoint():
    """Manually trigger model update"""
    thread = Thread(target=update_models)
    thread.start()
    return jsonify({'status': 'Model update started'})

# Initialize models on startup
@app.before_first_request
def initialize():
    update_models()

if __name__ == '__main__':
    # Start background model update
    update_models()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5001)