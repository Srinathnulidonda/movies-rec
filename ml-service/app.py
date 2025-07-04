#ml-service/app.py
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import sqlite3
import json
import logging
from datetime import datetime, timedelta
import requests
from collections import defaultdict
import pickle
import os
from threading import Thread
import time
import hashlib
from typing import Dict, List, Optional, Tuple
import redis
from celery import Celery
import implicit
import lightgbm as lgb
import optuna
from sentence_transformers import SentenceTransformer
import faiss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
class Config:
    DATABASE_URL = os.getenv('DATABASE_URL', '../recommendations.db')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379')
    ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', './models')
    CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))  # 1 hour
    MAX_RECOMMENDATIONS = int(os.getenv('MAX_RECOMMENDATIONS', 50))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1000))

config = Config()

# Initialize Redis for caching
try:
    redis_client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")
    redis_client = None

# Initialize Celery for background tasks
celery_app = Celery('ml_service', broker=config.CELERY_BROKER_URL)

# Global ML models storage
class ModelStore:
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.sentence_transformer = None
        self.faiss_index = None
        self.content_embeddings = {}
        
    def initialize_models(self):
        """Initialize ML models"""
        try:
            # Load sentence transformer for semantic similarity
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded successfully")
            
            # Initialize FAISS for fast similarity search
            self.faiss_index = faiss.IndexFlatIP(384)  # Dimension for all-MiniLM-L6-v2
            logger.info("FAISS index initialized")
            
        except Exception as e:
            logger.error(f"Model initialization error: {e}")

model_store = ModelStore()

class AdvancedRecommendationEngine:
    def __init__(self):
        self.als_model = None
        self.content_features = {}
        self.user_embeddings = {}
        self.content_embeddings = {}
        self.lightgbm_model = None
        self.feature_cache = {}
        
    def get_cache_key(self, key_type: str, **kwargs) -> str:
        """Generate cache key for Redis"""
        key_data = f"{key_type}:{':'.join(f'{k}={v}' for k, v in sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached result from Redis"""
        if not redis_client:
            return None
        
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        return None
    
    def set_cached_result(self, cache_key: str, data: Dict, ttl: int = None):
        """Set cached result in Redis"""
        if not redis_client:
            return
        
        try:
            redis_client.setex(
                cache_key,
                ttl or config.CACHE_TTL,
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from database with caching"""
        cache_key = self.get_cache_key('data_load')
        cached_data = self.get_cached_result(cache_key)
        
        if cached_data:
            content_df = pd.DataFrame(cached_data['content'])
            interactions_df = pd.DataFrame(cached_data['interactions'])
            return content_df, interactions_df
        
        try:
            conn = sqlite3.connect(config.DATABASE_URL)
            
            # Load content with aggregated ratings
            content_query = '''
                SELECT c.*, 
                       COALESCE(AVG(ui.rating), 0) as avg_rating,
                       COALESCE(COUNT(ui.rating), 0) as rating_count,
                       COALESCE(SUM(CASE WHEN ui.interaction_type = 'watchlist' THEN 1 ELSE 0 END), 0) as watchlist_count
                FROM content c
                LEFT JOIN user_interactions ui ON c.id = ui.content_id
                GROUP BY c.id
                ORDER BY c.id
            '''
            content_df = pd.read_sql_query(content_query, conn)
            
            # Load user interactions
            interactions_query = '''
                SELECT ui.*, c.genre_ids, c.content_type, c.vote_average
                FROM user_interactions ui
                JOIN content c ON ui.content_id = c.id
                ORDER BY ui.created_at DESC
            '''
            interactions_df = pd.read_sql_query(interactions_query, conn)
            
            conn.close()
            
            # Cache the data
            cache_data = {
                'content': content_df.to_dict('records'),
                'interactions': interactions_df.to_dict('records')
            }
            self.set_cached_result(cache_key, cache_data, ttl=1800)  # 30 minutes
            
            return content_df, interactions_df
            
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def build_content_embeddings(self, content_df: pd.DataFrame) -> Dict:
        """Build content embeddings using sentence transformers"""
        if model_store.sentence_transformer is None:
            logger.warning("Sentence transformer not available")
            return {}
        
        cache_key = self.get_cache_key('content_embeddings', version='v1')
        cached_embeddings = self.get_cached_result(cache_key)
        
        if cached_embeddings:
            return cached_embeddings
        
        try:
            embeddings = {}
            
            for idx, row in content_df.iterrows():
                content_id = row['id']
                
                # Combine text features
                text_features = []
                if pd.notna(row.get('title')):
                    text_features.append(row['title'])
                if pd.notna(row.get('overview')):
                    text_features.append(row['overview'])
                
                # Add genre information
                genres = json.loads(row.get('genre_ids', '[]'))
                if genres:
                    text_features.append(' '.join(genres))
                
                combined_text = ' '.join(text_features)
                
                if combined_text.strip():
                    embedding = model_store.sentence_transformer.encode(combined_text)
                    embeddings[content_id] = embedding.tolist()
            
            # Cache embeddings
            self.set_cached_result(cache_key, embeddings, ttl=86400)  # 24 hours
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Content embeddings error: {e}")
            return {}
    
    def train_als_model(self, interactions_df: pd.DataFrame):
        """Train ALS model for collaborative filtering"""
        if interactions_df.empty:
            return None
        
        try:
            # Prepare data for ALS
            user_ids = interactions_df['user_id'].unique()
            content_ids = interactions_df['content_id'].unique()
            
            # Create mappings
            user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
            content_to_idx = {content_id: idx for idx, content_id in enumerate(content_ids)}
            
            # Create interaction matrix
            n_users = len(user_ids)
            n_items = len(content_ids)
            
            interaction_matrix = np.zeros((n_users, n_items))
            
            for _, row in interactions_df.iterrows():
                user_idx = user_to_idx[row['user_id']]
                content_idx = content_to_idx[row['content_id']]
                
                # Weight different interaction types
                weight = 1.0
                if row['interaction_type'] == 'rating':
                    weight = row['rating'] / 5.0
                elif row['interaction_type'] == 'watchlist':
                    weight = 0.8
                elif row['interaction_type'] == 'favorite':
                    weight = 1.2
                
                interaction_matrix[user_idx, content_idx] = weight
            
            # Train ALS model
            als_model = implicit.als.AlternatingLeastSquares(
                factors=50,
                regularization=0.1,
                iterations=20,
                calculate_training_loss=True
            )
            
            # Convert to sparse matrix
            from scipy.sparse import csr_matrix
            sparse_matrix = csr_matrix(interaction_matrix)
            
            als_model.fit(sparse_matrix)
            
            # Store model and mappings
            self.als_model = als_model
            self.user_to_idx = user_to_idx
            self.content_to_idx = content_to_idx
            self.idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
            self.idx_to_content = {idx: content_id for content_id, idx in content_to_idx.items()}
            
            logger.info(f"ALS model trained with {n_users} users and {n_items} items")
            
            return als_model
            
        except Exception as e:
            logger.error(f"ALS training error: {e}")
            return None
    
    def collaborative_filtering(self, user_id: int, limit: int = 20) -> List[Dict]:
        """Collaborative filtering using ALS"""
        if self.als_model is None:
            return []
        
        cache_key = self.get_cache_key('collab_filter', user_id=user_id, limit=limit)
        cached_result = self.get_cached_result(cache_key)
        
        if cached_result:
            return cached_result
        
        try:
            if user_id not in self.user_to_idx:
                # Handle new user with popularity-based recommendations
                return self.popularity_based_recommendations(limit)
            
            user_idx = self.user_to_idx[user_id]
            
            # Get recommendations from ALS
            recommended_items, scores = self.als_model.recommend(
                user_idx,
                sparse_matrix=None,
                N=limit,
                filter_already_liked_items=True
            )
            
            # Get content details
            content_df, _ = self.load_data()
            recommendations = []
            
            for item_idx, score in zip(recommended_items, scores):
                content_id = self.idx_to_content[item_idx]
                content_row = content_df[content_df['id'] == content_id]
                
                if not content_row.empty:
                    content_data = content_row.iloc[0].to_dict()
                    content_data['recommendation_score'] = float(score)
                    content_data['algorithm'] = 'collaborative_filtering'
                    recommendations.append(content_data)
            
            # Cache results
            self.set_cached_result(cache_key, recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Collaborative filtering error: {e}")
            return []
    
    def content_based_recommendations(self, user_id: int, limit: int = 20) -> List[Dict]:
        """Content-based recommendations using embeddings"""
        cache_key = self.get_cache_key('content_based', user_id=user_id, limit=limit)
        cached_result = self.get_cached_result(cache_key)
        
        if cached_result:
            return cached_result
        
        try:
            content_df, interactions_df = self.load_data()
            content_embeddings = self.build_content_embeddings(content_df)
            
            if not content_embeddings:
                return []
            
            # Get user's interaction history
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            if user_interactions.empty:
                return self.popularity_based_recommendations(limit)
            
            # Build user profile from interactions
            user_profile = np.zeros(384)  # Dimension for all-MiniLM-L6-v2
            total_weight = 0
            
            for _, interaction in user_interactions.iterrows():
                content_id = interaction['content_id']
                
                if content_id in content_embeddings:
                    # Weight based on interaction type and rating
                    weight = 1.0
                    if interaction['interaction_type'] == 'rating':
                        weight = interaction['rating'] / 5.0
                    elif interaction['interaction_type'] == 'watchlist':
                        weight = 0.7
                    elif interaction['interaction_type'] == 'favorite':
                        weight = 1.0
                    
                    embedding = np.array(content_embeddings[content_id])
                    user_profile += weight * embedding
                    total_weight += weight
            
            if total_weight > 0:
                user_profile /= total_weight
            
            # Find similar content
            similarities = {}
            interacted_content = set(user_interactions['content_id'])
            
            for content_id, embedding in content_embeddings.items():
                if content_id not in interacted_content:
                    similarity = cosine_similarity([user_profile], [embedding])[0][0]
                    similarities[content_id] = similarity
            
            # Get top recommendations
            top_content = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:limit]
            
            recommendations = []
            for content_id, similarity in top_content:
                content_row = content_df[content_df['id'] == content_id]
                
                if not content_row.empty:
                    content_data = content_row.iloc[0].to_dict()
                    content_data['recommendation_score'] = float(similarity)
                    content_data['algorithm'] = 'content_based'
                    recommendations.append(content_data)
            
            # Cache results
            self.set_cached_result(cache_key, recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Content-based recommendations error: {e}")
            return []
    
    def popularity_based_recommendations(self, limit: int = 20) -> List[Dict]:
        """Popularity-based recommendations for new users"""
        cache_key = self.get_cache_key('popularity', limit=limit)
        cached_result = self.get_cached_result(cache_key)
        
        if cached_result:
            return cached_result
        
        try:
            content_df, _ = self.load_data()
            
            # Calculate popularity score
            content_df['popularity_score'] = (
                content_df['avg_rating'] * 0.4 +
                np.log1p(content_df['rating_count']) * 0.3 +
                np.log1p(content_df['watchlist_count']) * 0.2 +
                content_df['vote_average'] / 10.0 * 0.1
            )
            
            # Get top popular content
            top_content = content_df.nlargest(limit, 'popularity_score')
            
            recommendations = []
            for _, row in top_content.iterrows():
                content_data = row.to_dict()
                content_data['recommendation_score'] = float(row['popularity_score'])
                content_data['algorithm'] = 'popularity'
                recommendations.append(content_data)
            
            # Cache results
            self.set_cached_result(cache_key, recommendations, ttl=7200)  # 2 hours
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Popularity recommendations error: {e}")
            return []
    
    def hybrid_recommendations(self, user_id: int, limit: int = 20) -> List[Dict]:
        """Hybrid recommendations combining multiple approaches"""
        cache_key = self.get_cache_key('hybrid', user_id=user_id, limit=limit)
        cached_result = self.get_cached_result(cache_key)
        
        if cached_result:
            return cached_result
        
        try:
            # Get recommendations from different algorithms
            collab_recs = self.collaborative_filtering(user_id, limit)
            content_recs = self.content_based_recommendations(user_id, limit)
            
            # Determine weights based on user activity
            _, interactions_df = self.load_data()
            user_interaction_count = len(interactions_df[interactions_df['user_id'] == user_id])
            
            if user_interaction_count > 20:
                collab_weight = 0.6
                content_weight = 0.4
            elif user_interaction_count > 5:
                collab_weight = 0.4
                content_weight = 0.6
            else:
                # New user - rely more on content-based
                collab_weight = 0.2
                content_weight = 0.8
            
            # Combine recommendations
            combined_recs = {}
            
            # Add collaborative filtering recommendations
            for rec in collab_recs:
                content_id = rec['id']
                rec['recommendation_score'] = rec['recommendation_score'] * collab_weight
                combined_recs[content_id] = rec
            
            # Add content-based recommendations
            for rec in content_recs:
                content_id = rec['id']
                if content_id in combined_recs:
                    # Combine scores
                    combined_recs[content_id]['recommendation_score'] += rec['recommendation_score'] * content_weight
                    combined_recs[content_id]['algorithm'] = 'hybrid'
                else:
                    rec['recommendation_score'] = rec['recommendation_score'] * content_weight
                    rec['algorithm'] = 'hybrid'
                    combined_recs[content_id] = rec
            
            # Sort by combined score
            final_recs = sorted(combined_recs.values(), key=lambda x: x['recommendation_score'], reverse=True)
            
            # Apply diversity filter
            diverse_recs = self.apply_diversity_filter(final_recs[:limit*2], user_id)
            
            result = diverse_recs[:limit]
            
            # Cache results
            self.set_cached_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid recommendations error: {e}")
            return self.popularity_based_recommendations(limit)
    
    def apply_diversity_filter(self, recommendations: List[Dict], user_id: int) -> List[Dict]:
        """Apply diversity filter to recommendations"""
        if not recommendations:
            return recommendations
        
        diverse_recs = []
        selected_genres = set()
        selected_types = set()
        
        # Sort by score first
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        for rec in recommendations:
            genres = json.loads(rec.get('genre_ids', '[]'))
            content_type = rec.get('content_type', '')
            
            # Calculate diversity bonus
            genre_diversity = len(set(genres) - selected_genres) / max(len(genres), 1)
            type_diversity = 1.0 if content_type not in selected_types else 0.5
            
            diversity_bonus = (genre_diversity + type_diversity) / 2 * 0.1
            
            # Apply diversity bonus
            rec['recommendation_score'] += diversity_bonus
            
            diverse_recs.append(rec)
            selected_genres.update(genres)
            selected_types.add(content_type)
        
        return sorted(diverse_recs, key=lambda x: x['recommendation_score'], reverse=True)
    
    def update_models(self):
        """Update all ML models with latest data"""
        try:
            logger.info("Starting model update...")
            
            # Load latest data
            content_df, interactions_df = self.load_data()
            
            # Train ALS model
            self.train_als_model(interactions_df)
            
            # Clear caches
            if redis_client:
                redis_client.flushdb()
            
            logger.info("Model update completed successfully")
            
        except Exception as e:
            logger.error(f"Model update error: {e}")

# Initialize recommendation engine
rec_engine = AdvancedRecommendationEngine()

# Background task for model updates
@celery_app.task
def update_models_task():
    """Background task to update models"""
    rec_engine.update_models()

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'redis_connected': redis_client is not None and redis_client.ping() if redis_client else False,
        'models_loaded': rec_engine.als_model is not None
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get recommendations for a user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        limit = min(data.get('limit', 20), config.MAX_RECOMMENDATIONS)
        algorithm = data.get('algorithm', 'hybrid')
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Get recommendations based on algorithm
        if algorithm == 'collaborative':
            recommendations = rec_engine.collaborative_filtering(user_id, limit)
        elif algorithm == 'content_based':
            recommendations = rec_engine.content_based_recommendations(user_id, limit)
        elif algorithm == 'popularity':
            recommendations = rec_engine.popularity_based_recommendations(limit)
        else:  # hybrid
            recommendations = rec_engine.hybrid_recommendations(user_id, limit)
        
        return jsonify({
            'recommendations': recommendations,
            'algorithm': algorithm,
            'user_id': user_id,
            'count': len(recommendations),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/similar', methods=['POST'])
def similar_content():
    """Get similar content to a given item"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        limit = min(data.get('limit', 10), 20)
        
        if not content_id:
            return jsonify({'error': 'Content ID is required'}), 400
        
        cache_key = rec_engine.get_cache_key('similar', content_id=content_id, limit=limit)
        cached_result = rec_engine.get_cached_result(cache_key)
        
        if cached_result:
            return jsonify(cached_result)
        
        # Get content embeddings
        content_df, _ = rec_engine.load_data()
        content_embeddings = rec_engine.build_content_embeddings(content_df)
        
        if content_id not in content_embeddings:
            return jsonify({'error': 'Content not found'}), 404
        
        target_embedding = content_embeddings[content_id]
        similarities = {}
        
        for cid, embedding in content_embeddings.items():
            if cid != content_id:
                similarity = cosine_similarity([target_embedding], [embedding])[0][0]
                similarities[cid] = similarity
        
        # Get top similar content
        top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        similar_content = []
        for cid, similarity in top_similar:
            content_row = content_df[content_df['id'] == cid]
            if not content_row.empty:
                content_data = content_row.iloc[0].to_dict()
                content_data['similarity_score'] = float(similarity)
                similar_content.append(content_data)
        
        result = {
            'similar': similar_content,
            'content_id': content_id,
            'count': len(similar_content),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache result
        rec_engine.set_cached_result(cache_key, result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Similar content error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/update_models', methods=['POST'])
def trigger_model_update():
    """Trigger model update"""
    try:
        # Schedule background task
        update_models_task.delay()
        
        return jsonify({
            'message': 'Model update scheduled',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Model update trigger error: {e}")
        return jsonify({'error': 'Failed to schedule model update'}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get service metrics"""
    try:
        # Get basic metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'models_loaded': rec_engine.als_model is not None,
            'cache_connected': redis_client is not None
        }
        
        # Add cache statistics if Redis is available
        if redis_client:
            try:
                info = redis_client.info()
                metrics['cache_stats'] = {
                    'used_memory': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'total_commands_processed': info.get('total_commands_processed')
                }
            except Exception as e:
                logger.warning(f"Failed to get cache stats: {e}")
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return jsonify({'error': 'Failed to get metrics'}), 500

# Initialize models on startup
@app.before_first_request
def initialize_service():
    """Initialize service on first request"""
    model_store.initialize_models()
    rec_engine.update_models()

if __name__ == '__main__':
    # Initialize models
    model_store.initialize_models()
    
    # Start the service
    logger.info("Starting Enhanced ML Recommendation Service...")
    app.run(debug=False, host='0.0.0.0', port=5001)