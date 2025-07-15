## Powerful ML Service (ml-service/app.py)
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import implicit
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k
import joblib
import sqlite3
import logging
import asyncio
from datetime import datetime, timedelta
import redis
from collections import defaultdict
import os
import json
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation ML Service",
    description="Advanced ML-powered recommendation system",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///movie_rec.db')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
MODEL_UPDATE_INTERVAL = int(os.getenv('MODEL_UPDATE_INTERVAL', '3600'))  # 1 hour

# Initialize Redis (optional)
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
    logger.info("Redis connected successfully")
except Exception as e:
    redis_client = None
    logger.warning(f"Redis not available: {e}")

# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: int
    limit: int = 20
    content_types: Optional[List[str]] = None
    genres: Optional[List[str]] = None
    exclude_seen: bool = True

class RecommendationResponse(BaseModel):
    recommendations: List[int]
    scores: List[float]
    algorithm_used: str
    model_version: str
    explanation: Dict[str, Any]

class UserInteractionRequest(BaseModel):
    user_id: int
    content_id: int
    interaction_type: str
    rating: Optional[int] = None
    implicit_feedback: Optional[float] = None

class ContentSimilarityRequest(BaseModel):
    content_id: int
    limit: int = 10
    similarity_threshold: float = 0.1

# Advanced ML Models Container
class AdvancedRecommenderSystem:
    def __init__(self):
        self.models = {}
        self.data_processors = {}
        self.feature_extractors = {}
        self.model_metadata = {}
        self.is_trained = False
        self.last_update = None
        
        # Initialize components
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all ML models"""
        # LightFM for hybrid recommendations
        self.models['lightfm'] = LightFM(
            loss='warp',
            learning_rate=0.05,
            item_alpha=1e-6,
            user_alpha=1e-6,
            max_sampled=10
        )
        
        # Implicit ALS for collaborative filtering
        self.models['implicit_als'] = implicit.als.AlternatingLeastSquares(
            factors=64,
            regularization=0.01,
            iterations=15,
            alpha=40.0
        )
        
        # BPR for ranking
        self.models['implicit_bpr'] = implicit.bpr.BayesianPersonalizedRanking(
            factors=64,
            learning_rate=0.01,
            regularization=0.01,
            iterations=100
        )
        
        # Content-based models
        self.models['content_tfidf'] = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # SVD for dimensionality reduction
        self.models['svd'] = TruncatedSVD(
            n_components=100,
            random_state=42
        )
        
        # KMeans for user clustering
        self.models['user_clustering'] = KMeans(
            n_clusters=20,
            random_state=42,
            n_init=10
        )
        
        # Nearest Neighbors for similarity
        self.models['knn'] = NearestNeighbors(
            n_neighbors=50,
            algorithm='auto',
            metric='cosine'
        )
        
        # Feature scaler
        self.data_processors['scaler'] = StandardScaler()
        
        logger.info("ML models initialized successfully")
    
    def load_data_from_db(self):
        """Load and preprocess data from database"""
        try:
            # Connect to database
            conn = sqlite3.connect(DATABASE_URL.replace('sqlite:///', ''))
            
            # Load content data
            content_query = """
            SELECT id, title, overview, genres, language, rating, 
                   popularity, content_type, cast, crew, keywords,
                   release_date, runtime
            FROM content
            """
            self.content_df = pd.read_sql_query(content_query, conn)
            
            # Load user interactions
            interactions_query = """
            SELECT ui.user_id, ui.content_id, ui.interaction_type, 
                   ui.rating, ui.created_at,
                   c.genres, c.language, c.content_type
            FROM user_interaction ui
            JOIN content c ON ui.content_id = c.id
            WHERE ui.created_at >= date('now', '-365 days')
            """
            self.interactions_df = pd.read_sql_query(interactions_query, conn)
            
            # Load user data
            users_query = """
            SELECT id, preferences, demographics, created_at
            FROM user
            """
            self.users_df = pd.read_sql_query(users_query, conn)
            
            conn.close()
            
            # Preprocess data
            self._preprocess_data()
            
            logger.info(f"Data loaded: {len(self.content_df)} content items, "
                       f"{len(self.interactions_df)} interactions, "
                       f"{len(self.users_df)} users")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            return False
    
    def _preprocess_data(self):
        """Preprocess loaded data"""
        # Parse JSON fields
        def safe_json_loads(x):
            if pd.isna(x) or x == '':
                return []
            try:
                return json.loads(x) if isinstance(x, str) else x
            except:
                return []
        
        self.content_df['genres'] = self.content_df['genres'].apply(safe_json_loads)
        self.content_df['cast'] = self.content_df['cast'].apply(safe_json_loads)
        self.content_df['crew'] = self.content_df['crew'].apply(safe_json_loads)
        self.content_df['keywords'] = self.content_df['keywords'].apply(safe_json_loads)
        
        self.users_df['preferences'] = self.users_df['preferences'].apply(safe_json_loads)
        self.users_df['demographics'] = self.users_df['demographics'].apply(safe_json_loads)
        
        # Fill missing values
        self.content_df['overview'] = self.content_df['overview'].fillna('')
        self.content_df['rating'] = self.content_df['rating'].fillna(0)
        self.content_df['popularity'] = self.content_df['popularity'].fillna(0)
        
        # Create content features
        self._create_content_features()
        
        # Create user-item matrix
        self._create_user_item_matrix()
        
        # Process user features
        self._create_user_features()
    
    def _create_content_features(self):
        """Create comprehensive content features"""
        features = []
        
        for _, content in self.content_df.iterrows():
            feature_parts = []
            
            # Title and overview
            if content['title']:
                feature_parts.append(content['title'])
            if content['overview']:
                feature_parts.append(content['overview'])
            
            # Genres (weighted more heavily)
            if content['genres']:
                genre_names = [str(g) for g in content['genres']]
                feature_parts.extend(genre_names * 3)
            
            # Cast (top 5)
            if content['cast']:
                cast_names = [person.get('name', '') for person in content['cast'][:5]]
                feature_parts.extend(cast_names)
            
            # Directors (weighted heavily)
            if content['crew']:
                directors = [person.get('name', '') for person in content['crew'] 
                           if person.get('job') == 'Director']
                feature_parts.extend(directors * 2)
            
            # Keywords
            if content['keywords']:
                keyword_names = [kw.get('name', '') for kw in content['keywords']]
                feature_parts.extend(keyword_names)
            
            # Language and content type
            if content['language']:
                feature_parts.append(content['language'])
            if content['content_type']:
                feature_parts.append(content['content_type'])
            
            features.append(' '.join(filter(None, feature_parts)))
        
        # Create TF-IDF matrix
        self.content_features_matrix = self.models['content_tfidf'].fit_transform(features)
        
        # Reduce dimensionality
        self.content_features_reduced = self.models['svd'].fit_transform(
            self.content_features_matrix.toarray()
        )
        
        # Calculate content similarity matrix
        self.content_similarity = cosine_similarity(self.content_features_matrix)
        
        logger.info(f"Content features created: {self.content_features_matrix.shape}")
    
    def _create_user_item_matrix(self):
        """Create user-item interaction matrix"""
        # Create rating matrix
        rating_matrix = self.interactions_df.pivot_table(
            index='user_id',
            columns='content_id',
            values='rating',
            fill_value=0
        )
        
        # Create implicit feedback matrix
        implicit_matrix = self.interactions_df.groupby(['user_id', 'content_id']).agg({
            'interaction_type': lambda x: self._calculate_implicit_score(x.tolist())
        }).reset_index()
        
        implicit_pivot = implicit_matrix.pivot_table(
            index='user_id',
            columns='content_id',
            values='interaction_type',
            fill_value=0
        )
        
        # Ensure matrices have same shape
        all_users = sorted(set(rating_matrix.index) | set(implicit_pivot.index))
        all_items = sorted(set(rating_matrix.columns) | set(implicit_pivot.columns))
        
        self.rating_matrix = rating_matrix.reindex(
            index=all_users, columns=all_items, fill_value=0
        )
        self.implicit_matrix = implicit_pivot.reindex(
            index=all_users, columns=all_items, fill_value=0
        )
        
        # Convert to sparse matrices
        self.rating_sparse = csr_matrix(self.rating_matrix.values)
        self.implicit_sparse = csr_matrix(self.implicit_matrix.values)
        
        # Store user and item mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(all_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(all_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        logger.info(f"User-item matrices created: {self.rating_matrix.shape}")
    
    def _calculate_implicit_score(self, interactions):
        """Calculate implicit feedback score"""
        scores = {
            'view': 1.0,
            'like': 2.0,
            'favorite': 3.0,
            'wishlist': 1.5,
            'rating': 2.5
        }
        
        total_score = 0
        for interaction in interactions:
            total_score += scores.get(interaction, 0.5)
        
        return min(total_score, 10.0)  # Cap at 10
    
    def _create_user_features(self):
        """Create user feature matrix"""
        user_features = []
        
        for _, user in self.users_df.iterrows():
            features = {}
            
            # Demographic features
            demographics = user.get('demographics', {})
            features['age_group'] = demographics.get('age_group', 'unknown')
            features['gender'] = demographics.get('gender', 'unknown')
            features['location'] = demographics.get('location', 'unknown')
            
            # Preference features
            preferences = user.get('preferences', {})
            features['favorite_genres'] = preferences.get('favorite_genres', [])
            features['content_types'] = preferences.get('content_types', [])
            
            user_features.append(features)
        
        self.user_features_df = pd.DataFrame(user_features)
        
        # One-hot encode categorical features
        categorical_features = ['age_group', 'gender', 'location']
        self.user_features_encoded = pd.get_dummies(
            self.user_features_df[categorical_features],
            prefix=categorical_features
        )
        
        # Scale features
        self.user_features_scaled = self.data_processors['scaler'].fit_transform(
            self.user_features_encoded.fillna(0)
        )
        
        # Cluster users
        self.user_clusters = self.models['user_clustering'].fit_predict(
            self.user_features_scaled
        )
        
        logger.info(f"User features created: {self.user_features_scaled.shape}")
    
    def train_all_models(self):
        """Train all recommendation models"""
        try:
            logger.info("Starting model training...")
            
            # Train LightFM
            self._train_lightfm()
            
            # Train Implicit models
            self._train_implicit_models()
            
            # Train content-based models
            self._train_content_models()
            
            # Update model metadata
            self.model_metadata = {
                'last_trained': datetime.now().isoformat(),
                'data_size': {
                    'users': len(self.users_df),
                    'items': len(self.content_df),
                    'interactions': len(self.interactions_df)
                },
                'model_performance': self._evaluate_models()
            }
            
            self.is_trained = True
            self.last_update = datetime.now()
            
            # Save models
            self._save_models()
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def _train_lightfm(self):
        """Train LightFM hybrid model"""
        # Prepare data for LightFM
        dataset = Dataset()
        
        # Fit the dataset
        user_ids = list(self.user_to_idx.keys())
        item_ids = list(self.item_to_idx.keys())
        
        dataset.fit(user_ids, item_ids)
        
        # Build interactions matrix
        interactions = []
        for _, row in self.interactions_df.iterrows():
            if row['rating'] and row['rating'] >= 3:  # Only positive interactions
                interactions.append((row['user_id'], row['content_id'], row['rating']))
        
        (interactions_matrix, weights) = dataset.build_interactions(interactions)
        
        # Train model
        self.models['lightfm'].fit(
            interactions_matrix,
            sample_weight=weights,
            epochs=20,
            num_threads=4,
            verbose=False
        )
        
        self.lightfm_dataset = dataset
        self.lightfm_interactions = interactions_matrix
        
        logger.info("LightFM model trained")
    
    def _train_implicit_models(self):
        """Train implicit feedback models"""
        # Transpose for implicit library (items x users)
        implicit_matrix_T = self.implicit_sparse.T.tocsr()
        
        # Train ALS
        self.models['implicit_als'].fit(implicit_matrix_T)
        
        # Train BPR
        self.models['implicit_bpr'].fit(implicit_matrix_T)
        
        logger.info("Implicit models trained")
    
    def _train_content_models(self):
        """Train content-based models"""
        # KNN on content features
        self.models['knn'].fit(self.content_features_reduced)
        
        logger.info("Content-based models trained")
    
    def _evaluate_models(self):
        """Evaluate model performance"""
        try:
            performance = {}
            
            # Split data for evaluation
            train_interactions = self.interactions_df.sample(frac=0.8, random_state=42)
            test_interactions = self.interactions_df.drop(train_interactions.index)
            
            # LightFM evaluation
            if hasattr(self, 'lightfm_interactions'):
                test_precision = precision_at_k(
                    self.models['lightfm'],
                    self.lightfm_interactions,
                    k=10
                ).mean()
                performance['lightfm_precision_at_10'] = float(test_precision)
            
            # Content-based evaluation (using similarity)
            content_similarities = []
            for i in range(min(100, len(self.content_similarity))):
                similarities = self.content_similarity[i]
                top_similar = np.argsort(similarities)[-11:-1]  # Top 10 excluding self
                content_similarities.append(np.mean(similarities[top_similar]))
            
            performance['content_avg_similarity'] = float(np.mean(content_similarities))
            
            return performance
            
        except Exception as e:
            logger.warning(f"Error evaluating models: {e}")
            return {}
    
    def _save_models(self):
        """Save trained models"""
        try:
            model_dir = "saved_models"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save LightFM
            if hasattr(self, 'models') and 'lightfm' in self.models:
                joblib.dump(self.models['lightfm'], f"{model_dir}/lightfm_model.pkl")
            
            # Save implicit models
            for model_name in ['implicit_als', 'implicit_bpr']:
                if model_name in self.models:
                    joblib.dump(self.models[model_name], f"{model_dir}/{model_name}.pkl")
            
            # Save preprocessors
            joblib.dump(self.models['content_tfidf'], f"{model_dir}/content_tfidf.pkl")
            joblib.dump(self.models['svd'], f"{model_dir}/svd.pkl")
            joblib.dump(self.data_processors['scaler'], f"{model_dir}/scaler.pkl")
            
            # Save matrices and mappings
            np.save(f"{model_dir}/content_similarity.npy", self.content_similarity)
            joblib.dump(self.user_to_idx, f"{model_dir}/user_to_idx.pkl")
            joblib.dump(self.item_to_idx, f"{model_dir}/item_to_idx.pkl")
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def get_hybrid_recommendations(self, user_id: int, limit: int = 20) -> Dict[str, Any]:
        """Get recommendations using hybrid approach"""
        try:
            recommendations = {}
            
            # LightFM recommendations
            lightfm_recs = self._get_lightfm_recommendations(user_id, limit)
            recommendations['lightfm'] = lightfm_recs
            
            # Collaborative filtering recommendations
            cf_recs = self._get_collaborative_recommendations(user_id, limit)
            recommendations['collaborative'] = cf_recs
            
            # Content-based recommendations
            content_recs = self._get_content_recommendations(user_id, limit)
            recommendations['content_based'] = content_recs
            
            # Popularity-based recommendations
            popularity_recs = self._get_popularity_recommendations(user_id, limit)
            recommendations['popularity'] = popularity_recs
            
            # Combine recommendations with weights
            final_recs = self._combine_recommendations(recommendations, {
                'lightfm': 0.3,
                'collaborative': 0.25,
                'content_based': 0.25,
                'popularity': 0.2
            })
            
            return {
                'recommendations': final_recs[:limit],
                'individual_algorithms': recommendations,
                'algorithm_weights': {
                    'lightfm': 0.3,
                    'collaborative': 0.25,
                    'content_based': 0.25,
                    'popularity': 0.2
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting hybrid recommendations: {e}")
            return {'recommendations': [], 'error': str(e)}
    
    def _get_lightfm_recommendations(self, user_id: int, limit: int) -> List[int]:
        """Get LightFM recommendations"""
        try:
            if user_id not in self.user_to_idx:
                return []
            
            user_idx = self.user_to_idx[user_id]
            n_items = len(self.item_to_idx)
            
            scores = self.models['lightfm'].predict(
                user_idx,
                np.arange(n_items)
            )
            
            # Get top recommendations
            top_items = np.argsort(scores)[::-1][:limit * 2]  # Get more for filtering
            
            # Convert back to content IDs and filter seen items
            recommendations = []
            seen_items = self._get_user_seen_items(user_id)
            
            for item_idx in top_items:
                content_id = self.idx_to_item[item_idx]
                if content_id not in seen_items:
                    recommendations.append(content_id)
                if len(recommendations) >= limit:
                    break
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error in LightFM recommendations: {e}")
            return []
    
    def _get_collaborative_recommendations(self, user_id: int, limit: int) -> List[int]:
        """Get collaborative filtering recommendations"""
        try:
            if user_id not in self.user_to_idx:
                return []
            
            user_idx = self.user_to_idx[user_id]
            
            # Use implicit ALS
            recommendations, scores = self.models['implicit_als'].recommend(
                user_idx,
                self.implicit_sparse[user_idx],
                N=limit * 2,
                filter_already_liked_items=True
            )
            
            # Convert to content IDs
            content_ids = [self.idx_to_item[idx] for idx in recommendations]
            return content_ids[:limit]
            
        except Exception as e:
            logger.warning(f"Error in collaborative recommendations: {e}")
            return []
    
    def _get_content_recommendations(self, user_id: int, limit: int) -> List[int]:
        """Get content-based recommendations"""
        try:
            # Get user's favorite content
            user_interactions = self.interactions_df[
                (self.interactions_df['user_id'] == user_id) &
                (self.interactions_df['interaction_type'].isin(['favorite', 'like']))
            ]
            
            if user_interactions.empty:
                return []
            
            # Get content indices
            favorite_content_ids = user_interactions['content_id'].tolist()
            content_indices = []
            
            for content_id in favorite_content_ids:
                try:
                    idx = self.content_df[self.content_df['id'] == content_id].index[0]
                    content_indices.append(idx)
                except:
                    continue
            
            if not content_indices:
                return []
            
            # Calculate average similarity
            similarities = self.content_similarity[content_indices].mean(axis=0)
            top_indices = np.argsort(similarities)[::-1]
            
            # Get recommendations
            recommendations = []
            seen_items = self._get_user_seen_items(user_id)
            
            for idx in top_indices:
                content_id = self.content_df.iloc[idx]['id']
                if content_id not in seen_items and content_id not in favorite_content_ids:
                    recommendations.append(content_id)
                if len(recommendations) >= limit:
                    break
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error in content-based recommendations: {e}")
            return []
    
    def _get_popularity_recommendations(self, user_id: int, limit: int) -> List[int]:
        """Get popularity-based recommendations"""
        try:
            # Get user preferences
            user_data = self.users_df[self.users_df['id'] == user_id]
            if user_data.empty:
                # Default popular content
                popular_content = self.content_df.nlargest(limit * 2, 'popularity')
            else:
                # Filter by user preferences
                user_prefs = user_data.iloc[0].get('preferences', {})
                preferred_genres = user_prefs.get('favorite_genres', [])
                
                if preferred_genres:
                    # Filter by preferred genres
                    filtered_content = self.content_df[
                        self.content_df['genres'].apply(
                            lambda x: any(genre in x for genre in preferred_genres)
                        )
                    ]
                    popular_content = filtered_content.nlargest(limit * 2, 'popularity')
                else:
                    popular_content = self.content_df.nlargest(limit * 2, 'popularity')
            
            # Filter seen items
            seen_items = self._get_user_seen_items(user_id)
            recommendations = []
            
            for _, content in popular_content.iterrows():
                if content['id'] not in seen_items:
                    recommendations.append(content['id'])
                if len(recommendations) >= limit:
                    break
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error in popularity recommendations: {e}")
            return []
    
    def _get_user_seen_items(self, user_id: int) -> set:
        """Get items user has already seen"""
        user_interactions = self.interactions_df[
            self.interactions_df['user_id'] == user_id
        ]
        return set(user_interactions['content_id'].tolist())
    
    def _combine_recommendations(self, recommendations: Dict, weights: Dict) -> List[int]:
        """Combine recommendations from multiple algorithms"""
        combined_scores = defaultdict(float)
        
        for algorithm, recs in recommendations.items():
            weight = weights.get(algorithm, 0.25)
            for i, content_id in enumerate(recs):
                # Higher weight for higher ranking
                score = weight * (1.0 / (i + 1))
                combined_scores[content_id] += score
        
        # Sort by combined score
        sorted_recs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [content_id for content_id, _ in sorted_recs]

# Global model instance
recommender_system = AdvancedRecommenderSystem()

# Background task for model training
async def train_models_background():
    """Background task to train models"""
    logger.info("Starting background model training")
    
    # Load data
    if recommender_system.load_data_from_db():
        # Train models
        success = recommender_system.train_all_models()
        if success:
            logger.info("Background model training completed successfully")
        else:
            logger.error("Background model training failed")
    else:
        logger.error("Failed to load data for model training")

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting ML service...")
    
    # Try to load existing models
    try:
        # Load data and train models in background
        asyncio.create_task(train_models_background())
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.get("/")
async def root():
    return {
        "service": "Movie Recommendation ML Service",
        "version": "2.0.0",
        "status": "running",
        "models_trained": recommender_system.is_trained,
        "last_update": recommender_system.last_update.isoformat() if recommender_system.last_update else None
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations for a user"""
    try:
        if not recommender_system.is_trained:
            # Return popular recommendations as fallback
            popular_recs = recommender_system._get_popularity_recommendations(
                request.user_id, request.limit
            )
            return RecommendationResponse(
                recommendations=popular_recs,
                scores=[1.0] * len(popular_recs),
                algorithm_used="popularity_fallback",
                model_version="1.0",
                explanation={
                    "reason": "Models not trained yet, using popularity-based recommendations",
                    "total_algorithms": 1
                }
            )
        
        # Get hybrid recommendations
        result = recommender_system.get_hybrid_recommendations(
            request.user_id, request.limit
        )
        
        recommendations = result.get('recommendations', [])
        
        return RecommendationResponse(
            recommendations=recommendations,
            scores=[1.0] * len(recommendations),  # Normalized scores
            algorithm_used="hybrid",
            model_version="2.0",
            explanation={
                "algorithms_used": list(result.get('algorithm_weights', {}).keys()),
                "total_recommendations": len(recommendations),
                "personalization_score": min(len(recommendations) / request.limit, 1.0)
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similar")
async def get_similar_content(request: ContentSimilarityRequest):
    """Get similar content based on content ID"""
    try:
        if not recommender_system.is_trained:
            raise HTTPException(status_code=503, detail="Models not trained yet")
        
        # Find content index
        content_row = recommender_system.content_df[
            recommender_system.content_df['id'] == request.content_id
        ]
        
        if content_row.empty:
            raise HTTPException(status_code=404, detail="Content not found")
        
        content_idx = content_row.index[0]
        
        # Get similarities
        similarities = recommender_system.content_similarity[content_idx]
        
        # Filter by threshold and get top similar
        similar_indices = np.where(similarities >= request.similarity_threshold)[0]
        similar_scores = similarities[similar_indices]
        
        # Sort by similarity
        sorted_idx = np.argsort(similar_scores)[::-1][1:request.limit+1]  # Exclude self
        
        similar_content_ids = []
        scores = []
        
        for idx in sorted_idx:
            actual_idx = similar_indices[idx]
            content_id = recommender_system.content_df.iloc[actual_idx]['id']
            score = similar_scores[idx]
            
            similar_content_ids.append(content_id)
            scores.append(float(score))
        
        return {
            "similar_content": similar_content_ids,
            "similarity_scores": scores,
            "base_content_id": request.content_id,
            "algorithm": "content_based_cosine_similarity"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similar content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_interaction")
async def update_user_interaction(request: UserInteractionRequest):
    """Update user interaction for real-time learning"""
    try:
        # Store interaction in cache for batch processing
        if redis_client:
            interaction_data = {
                "user_id": request.user_id,
                "content_id": request.content_id,
                "interaction_type": request.interaction_type,
                "rating": request.rating,
                "timestamp": datetime.now().isoformat()
            }
            
            redis_client.lpush(
                "new_interactions",
                json.dumps(interaction_data)
            )
            
            # Trim list to keep only recent interactions
            redis_client.ltrim("new_interactions", 0, 9999)
        
        return {"status": "success", "message": "Interaction recorded"}
        
    except Exception as e:
        logger.error(f"Error updating interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    try:
        background_tasks.add_task(train_models_background)
        return {"status": "success", "message": "Model retraining started"}
        
    except Exception as e:
        logger.error(f"Error starting model retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_trained": recommender_system.is_trained,
        "last_update": recommender_system.last_update.isoformat() if recommender_system.last_update else None,
        "model_metadata": recommender_system.model_metadata,
        "redis_available": redis_client is not None
    }

@app.get("/stats")
async def get_service_stats():
    """Get service statistics"""
    try:
        stats = {
            "models_trained": recommender_system.is_trained,
            "last_update": recommender_system.last_update.isoformat() if recommender_system.last_update else None,
            "model_metadata": recommender_system.model_metadata
        }
        
        if recommender_system.is_trained:
            stats.update({
                "data_stats": {
                    "total_users": len(recommender_system.users_df),
                    "total_content": len(recommender_system.content_df),
                    "total_interactions": len(recommender_system.interactions_df),
                    "content_similarity_matrix_shape": recommender_system.content_similarity.shape,
                    "user_clusters": len(set(recommender_system.user_clusters))
                }
            })
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)