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
import psutil
import gc
from scipy.sparse import csr_matrix


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
    CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))
    MAX_RECOMMENDATIONS = int(os.getenv('MAX_RECOMMENDATIONS', 50))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1000))
    USE_HEAVY_MODELS = os.getenv('USE_HEAVY_MODELS', 'false').lower() == 'true'  # Add this
    MEMORY_LIMIT = int(os.getenv('MEMORY_LIMIT', 400))  # MB

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
        self._transformer_loaded = False
        
    def initialize_models(self):
        """Initialize only essential models"""
        try:
            if config.USE_HEAVY_MODELS:
                logger.info("Heavy models enabled via environment variable")
            else:
                logger.info("Heavy models disabled - using lightweight alternatives")
            
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
    
    def get_sentence_transformer(self):
        """Lazy load sentence transformer only when needed and allowed"""
        if not config.USE_HEAVY_MODELS:
            logger.info("Heavy models disabled - skipping sentence transformer")
            return None
            
        if not self._transformer_loaded:
            try:
                logger.info("Loading sentence transformer on demand...")
                # Use a smaller model for memory efficiency
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                self._transformer_loaded = True
                logger.info("Sentence transformer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
                return None
        return self.sentence_transformer

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
        # Check memory before starting
        try:
            memory_before = get_memory_usage()
            if memory_before > config.MEMORY_LIMIT * 0.8:  # 80% of limit
                logger.warning(f"Memory usage high ({memory_before:.1f}MB) - using TF-IDF fallback")
                return self._build_tfidf_embeddings(content_df)
        except Exception as e:
            logger.warning(f"Memory check failed: {e}, proceeding with fallback")
            return self._build_tfidf_embeddings(content_df)
        
        # Use lazy loading
        transformer = model_store.get_sentence_transformer()
        if transformer is None:
            logger.warning("Sentence transformer not available - using fallback")
            return self._build_tfidf_embeddings(content_df)
        
        cache_key = self.get_cache_key('content_embeddings', version='v1')
        cached_embeddings = self.get_cached_result(cache_key)
        
        if cached_embeddings:
            return cached_embeddings
        
        try:
            embeddings = {}
            
            # Process in smaller batches to manage memory
            batch_size = 10  # Smaller batch size
            content_items = list(content_df.iterrows())
            
            for i in range(0, len(content_items), batch_size):
                batch = content_items[i:i+batch_size]
                
                for idx, row in batch:
                    content_id = row['id']
                    
                    # Combine text features
                    text_features = []
                    if pd.notna(row.get('title')):
                        text_features.append(row['title'])
                    if pd.notna(row.get('overview')):
                        # Truncate overview to reduce memory usage
                        overview = row['overview'][:200] if len(row['overview']) > 200 else row['overview']
                        text_features.append(overview)
                    
                    # Add genre information
                    try:
                        genre_ids = row.get('genre_ids', '[]')
                        if isinstance(genre_ids, str):
                            genres = json.loads(genre_ids)
                        else:
                            genres = genre_ids if isinstance(genre_ids, list) else []
                        
                        if genres:
                            text_features.append(' '.join(str(g) for g in genres))
                    except Exception as e:
                        logger.debug(f"Genre parsing error for content {content_id}: {e}")
                    
                    combined_text = ' '.join(text_features)
                    
                    if combined_text.strip():
                        try:
                            embedding = transformer.encode(combined_text)
                            embeddings[content_id] = embedding.tolist()
                        except Exception as e:
                            logger.warning(f"Embedding error for content {content_id}: {e}")
                
                # Log progress and check memory
                if i % 50 == 0:
                    logger.info(f"Processed {i}/{len(content_items)} content items")
                    try:
                        current_memory = get_memory_usage()
                        if current_memory > config.MEMORY_LIMIT * 0.9:
                            logger.warning(f"Memory usage critical ({current_memory:.1f}MB), stopping embedding process")
                            break
                    except:
                        pass
            
            # Cache embeddings
            if embeddings:
                self.set_cached_result(cache_key, embeddings, ttl=86400)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Content embeddings error: {e}")
            # Fallback to TF-IDF
            return self._build_tfidf_embeddings(content_df)
        
    def _build_tfidf_embeddings(self, content_df: pd.DataFrame) -> Dict:
        """Fallback method using TF-IDF for embeddings (lighter on memory)"""
        try:
            logger.info("Using TF-IDF fallback for embeddings")
            
            texts = []
            content_ids = []
            
            for idx, row in content_df.iterrows():
                content_id = row['id']
                
                # Combine text features
                text_features = []
                if pd.notna(row.get('title')):
                    text_features.append(str(row['title']))
                if pd.notna(row.get('overview')):
                    # Truncate overview
                    overview = str(row['overview'])
                    overview = overview[:200] if len(overview) > 200 else overview
                    text_features.append(overview)
                
                # Add genre information
                try:
                    genre_ids = row.get('genre_ids', '[]')
                    if isinstance(genre_ids, str):
                        genres = json.loads(genre_ids)
                    else:
                        genres = genre_ids if isinstance(genre_ids, list) else []
                    
                    if genres:
                        text_features.append(' '.join(str(g) for g in genres))
                except Exception as e:
                    logger.debug(f"Genre parsing error for content {content_id}: {e}")
                
                combined_text = ' '.join(text_features)
                
                if combined_text.strip():
                    texts.append(combined_text)
                    content_ids.append(content_id)
            
            if not texts:
                logger.warning("No text data available for TF-IDF")
                return {}
            
            # Use TF-IDF with limited features
            vectorizer = TfidfVectorizer(
                max_features=1000,  # Limit features
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Convert to embeddings dictionary
            embeddings = {}
            for i, content_id in enumerate(content_ids):
                embeddings[content_id] = tfidf_matrix[i].toarray()[0].tolist()
            
            logger.info(f"TF-IDF embeddings created for {len(embeddings)} items")
            return embeddings
            
        except Exception as e:
            logger.error(f"TF-IDF embeddings error: {e}")
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
            sparse_matrix = csr_matrix(interaction_matrix)
            
            als_model.fit(sparse_matrix)
            
            # Store model and mappings
            self.als_model = als_model
            self.user_to_idx = user_to_idx
            self.content_to_idx = content_to_idx
            self.idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
            self.idx_to_content = {idx: content_id for content_id, idx in content_to_idx.items()}
            self.interaction_matrix = sparse_matrix  # Store the matrix
            
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
            
            # Use the stored interaction matrix
            if not hasattr(self, 'interaction_matrix') or self.interaction_matrix is None:
                return self.popularity_based_recommendations(limit)
            
            # Get recommendations from ALS
            recommended_items, scores = self.als_model.recommend(
                user_idx,
                self.interaction_matrix,
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
            
            # Get embedding dimension from first embedding
            sample_embedding = next(iter(content_embeddings.values()))
            embedding_dim = len(sample_embedding)
            
            # Build user profile from interactions
            user_profile = np.zeros(embedding_dim)
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
        """Hybrid recommendations with fallback for missing models"""
        try:
            # Try to get cached results first
            cache_key = self.get_cache_key('hybrid', user_id=user_id, limit=limit)
            cached_result = self.get_cached_result(cache_key)
            
            if cached_result:
                return cached_result
            
            # Check if models are available
            if not model_store._transformer_loaded and self.als_model is None:
                logger.info("Models not loaded, using popularity-based recommendations")
                return self.popularity_based_recommendations(limit)
            
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
        recommendations.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)
        
        for rec in recommendations:
            try:
                genre_ids = rec.get('genre_ids', '[]')
                if isinstance(genre_ids, str):
                    genres = json.loads(genre_ids)
                else:
                    genres = genre_ids if isinstance(genre_ids, list) else []
            except Exception as e:
                logger.debug(f"Genre parsing error in diversity filter: {e}")
                genres = []
                
            content_type = rec.get('content_type', '')
            
            # Calculate diversity bonus
            if genres:
                genre_diversity = len(set(str(g) for g in genres) - selected_genres) / len(genres)
            else:
                genre_diversity = 0
                
            type_diversity = 1.0 if content_type not in selected_types else 0.5
            
            diversity_bonus = (genre_diversity + type_diversity) / 2 * 0.1
            
            # Apply diversity bonus
            current_score = rec.get('recommendation_score', 0)
            rec['recommendation_score'] = current_score + diversity_bonus
            
            diverse_recs.append(rec)
            selected_genres.update(str(g) for g in genres)
            selected_types.add(content_type)
        
        return sorted(diverse_recs, key=lambda x: x.get('recommendation_score', 0), reverse=True)
    
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
model_store = ModelStore()
rec_engine = AdvancedRecommendationEngine()
# Initialize models at module level for production deployment
try:
    model_store.initialize_models()  # This now only does basic setup
    logger.info("Basic model store initialized")
except Exception as e:
    logger.error(f"Model initialization failed: {e}")

# Background task for model updates
@celery_app.task
def update_models_task():
    """Background task to update models"""
    rec_engine.update_models()

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    redis_connected = False
    if redis_client:
        try:
            redis_client.ping()
            redis_connected = True
        except:
            redis_connected = False
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'redis_connected': redis_connected,
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
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
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
        if content_df.empty:
            return jsonify({'error': 'No content data available'}), 404
            
        content_embeddings = rec_engine.build_content_embeddings(content_df)
        
        if not content_embeddings:
            return jsonify({'error': 'Content embeddings not available'}), 500
        
        if content_id not in content_embeddings:
            return jsonify({'error': 'Content not found'}), 404
        
        target_embedding = content_embeddings[content_id]
        similarities = {}
        
        for cid, embedding in content_embeddings.items():
            if cid != content_id:
                try:
                    similarity = cosine_similarity([target_embedding], [embedding])[0][0]
                    similarities[cid] = similarity
                except Exception as e:
                    logger.debug(f"Similarity calculation error for content {cid}: {e}")
                    continue
        
        # Get top similar content
        top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        similar_content_list = []
        for cid, similarity in top_similar:
            content_row = content_df[content_df['id'] == cid]
            if not content_row.empty:
                content_data = content_row.iloc[0].to_dict()
                content_data['similarity_score'] = float(similarity)
                similar_content_list.append(content_data)
        
        result = {
            'similar': similar_content_list,
            'content_id': content_id,
            'count': len(similar_content_list),
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

# Add a separate initialization endpoint:
@app.route('/initialize', methods=['POST'])
def initialize_models():
    """Initialize heavy models on demand"""
    try:
        # This will trigger model loading
        model_store.get_sentence_transformer()
        rec_engine.update_models()
        
        return jsonify({
            'message': 'Models initialized successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Model initialization error: {e}")
        return jsonify({'error': 'Model initialization failed'}), 500

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

def get_memory_usage():
    """Get current memory usage"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except Exception as e:
        logger.warning(f"Memory usage check failed: {e}")
        return 0

@app.route('/memory', methods=['GET'])
def memory_status():
    """Check memory usage"""
    try:
        memory_mb = get_memory_usage()
        return jsonify({
            'memory_usage_mb': memory_mb,
            'memory_limit_mb': config.MEMORY_LIMIT,
            'memory_usage_percent': (memory_mb / config.MEMORY_LIMIT) * 100,
            'models_loaded': {
                'sentence_transformer': model_store._transformer_loaded,
                'als_model': rec_engine.als_model is not None
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 6. Add garbage collection helper:
def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()
    logger.info(f"Memory usage after cleanup: {get_memory_usage():.1f}MB")


def create_app():
    """Application factory pattern"""
    app_instance = Flask(__name__)
    
    # Initialize models during app creation
    model_store.initialize_models()
    
    # Load data and train models
    content_df, interactions_df = rec_engine.load_data()
    if not interactions_df.empty:
        rec_engine.train_als_model(interactions_df)
    
    return app_instance


if __name__ == '__main__':
    try:
        # Initialize models
        logger.info("Initializing model store...")
        model_store.initialize_models()
        
        # Load data and train models
        logger.info("Loading data and training models...")
        content_df, interactions_df = rec_engine.load_data()
        if not interactions_df.empty:
            rec_engine.train_als_model(interactions_df)
        else:
            logger.warning("No interaction data available for training")
        
        # Start the service
        logger.info("Starting Enhanced ML Recommendation Service...")
        app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        import traceback
        traceback.print_exc()
        raise