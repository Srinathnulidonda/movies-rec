# backend/personalized/recommendation_engine.py


"""
CineBrain Hybrid Recommendation Engine
Multi-algorithm recommendation system with real-time adaptation
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import logging
import random
import math
from typing import List, Dict, Any, Tuple, Optional, Set
from sqlalchemy import func, desc, and_, or_
import lightgbm as lgb
from annoy import AnnoyIndex

from .profile_analyzer import UserProfileAnalyzer, CinematicDNAAnalyzer
from .utils import (
    VectorOperations, ContentEmbedding, CacheManager, 
    LanguagePriorityManager, DataProcessor, calculate_content_quality_score,
    safe_json_loads, PRIORITY_LANGUAGES
)
from .feedback import FeedbackProcessor, OnlineLearner
from .metrics import PerformanceTracker, RecommendationMetrics
from services.algorithms import UltraPowerfulSimilarityEngine

logger = logging.getLogger(__name__)

class CollaborativeFilteringEngine:
    """Matrix factorization based collaborative filtering"""
    
    def __init__(self, n_factors: int = 50, regularization: float = 0.01):
        self.n_factors = n_factors
        self.regularization = regularization
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}
        self.global_mean = 0
        self.is_fitted = False
        
        # Use NMF for non-negative matrix factorization
        self.nmf_model = NMF(
            n_components=n_factors,
            init='nndsvd',
            regularization=regularization,
            max_iter=200,
            random_state=42
        )
        
        # Alternative SVD model
        self.svd_model = TruncatedSVD(
            n_components=n_factors,
            n_iter=100,
            random_state=42
        )
    
    def fit(self, interactions_df: pd.DataFrame):
        """Fit collaborative filtering model on user-item interactions"""
        try:
            if interactions_df.empty:
                logger.warning("Empty interactions dataframe for CF fitting")
                return
            
            # Create user-item matrix
            self.user_map = {u: i for i, u in enumerate(interactions_df['user_id'].unique())}
            self.item_map = {i: j for j, i in enumerate(interactions_df['content_id'].unique())}
            
            # Create sparse matrix
            row = [self.user_map[u] for u in interactions_df['user_id']]
            col = [self.item_map[i] for i in interactions_df['content_id']]
            
            # Use ratings if available, otherwise implicit feedback
            if 'rating' in interactions_df.columns:
                data = interactions_df['rating'].fillna(5.0).values
                self.global_mean = data.mean()
            else:
                # Implicit feedback: use interaction counts/types
                interaction_weights = {
                    'favorite': 5.0,
                    'watchlist': 4.0,
                    'view': 3.0,
                    'rating': 3.0,
                    'search': 2.0,
                    'click': 1.0
                }
                data = [interaction_weights.get(t, 1.0) 
                       for t in interactions_df.get('interaction_type', ['view']*len(interactions_df))]
                self.global_mean = np.mean(data)
            
            # Normalize ratings
            data_normalized = data - self.global_mean
            
            # Create sparse matrix
            n_users = len(self.user_map)
            n_items = len(self.item_map)
            interaction_matrix = csr_matrix(
                (data_normalized, (row, col)), 
                shape=(n_users, n_items)
            )
            
            # Fit models
            if np.all(data >= 0):
                # Use NMF for non-negative data
                self.user_factors = self.nmf_model.fit_transform(interaction_matrix)
                self.item_factors = self.nmf_model.components_.T
            else:
                # Use SVD for general data
                self.user_factors = self.svd_model.fit_transform(interaction_matrix)
                self.item_factors = self.svd_model.components_.T
            
            self.is_fitted = True
            logger.info(f"Fitted collaborative filtering with {n_users} users and {n_items} items")
            
        except Exception as e:
            logger.error(f"Error fitting collaborative filtering: {e}")
            self.is_fitted = False
    
    def predict_user_preferences(self, user_id: int, 
                               candidate_items: List[int],
                               n_recommendations: int = 20) -> List[Tuple[int, float]]:
        """Predict user preferences for candidate items"""
        if not self.is_fitted or user_id not in self.user_map:
            return []
        
        try:
            user_idx = self.user_map[user_id]
            user_vec = self.user_factors[user_idx]
            
            predictions = []
            for item_id in candidate_items:
                if item_id in self.item_map:
                    item_idx = self.item_map[item_id]
                    item_vec = self.item_factors[item_idx]
                    
                    # Predict rating
                    prediction = np.dot(user_vec, item_vec) + self.global_mean
                    predictions.append((item_id, prediction))
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            return predictions[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error predicting user preferences: {e}")
            return []
    
    def find_similar_users(self, user_id: int, n_neighbors: int = 10) -> List[Tuple[int, float]]:
        """Find similar users based on latent factors"""
        if not self.is_fitted or user_id not in self.user_map:
            return []
        
        try:
            user_idx = self.user_map[user_id]
            user_vec = self.user_factors[user_idx].reshape(1, -1)
            
            # Compute similarities with all users
            similarities = cosine_similarity(user_vec, self.user_factors)[0]
            
            # Get top similar users (excluding self)
            similar_users = []
            reverse_user_map = {v: k for k, v in self.user_map.items()}
            
            for idx, sim in enumerate(similarities):
                if idx != user_idx:
                    similar_users.append((reverse_user_map[idx], sim))
            
            similar_users.sort(key=lambda x: x[1], reverse=True)
            
            return similar_users[:n_neighbors]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return []

class ContentBasedEngine:
    """Enhanced content-based filtering with embeddings"""
    
    def __init__(self, embedding_dim: int = 100):
        self.embedding_dim = embedding_dim
        self.content_embedder = ContentEmbedding(max_features=5000)
        self.content_index = None
        self.content_map = {}
        self.is_fitted = False
        
        # Annoy index for fast similarity search
        self.annoy_index = None
    
    def fit(self, content_list: List[Any]):
        """Fit content embeddings and build search index"""
        try:
            if not content_list:
                logger.warning("Empty content list for content-based fitting")
                return
            
            # Generate embeddings
            embeddings = self.content_embedder.fit_transform(content_list)
            
            # Build content mapping
            self.content_map = {content.id: idx for idx, content in enumerate(content_list)}
            
            # Build Annoy index for fast similarity search
            self.annoy_index = AnnoyIndex(embeddings.shape[1], 'angular')
            
            for idx, content in enumerate(content_list):
                self.annoy_index.add_item(idx, embeddings[idx])
            
            self.annoy_index.build(50)  # 50 trees
            self.is_fitted = True
            
            logger.info(f"Built content index with {len(content_list)} items")
            
        except Exception as e:
            logger.error(f"Error fitting content-based engine: {e}")
            self.is_fitted = False
    
    def find_similar_content(self, content_id: int, 
                           n_similar: int = 20,
                           exclude_ids: Set[int] = None) -> List[Tuple[int, float]]:
        """Find similar content using embeddings"""
        if not self.is_fitted or content_id not in self.content_map:
            return []
        
        try:
            content_idx = self.content_map[content_id]
            exclude_ids = exclude_ids or set()
            
            # Search for more items to account for filtering
            similar_indices, distances = self.annoy_index.get_nns_by_item(
                content_idx, 
                n_similar * 2, 
                include_distances=True
            )
            
            # Convert to content IDs and similarities
            reverse_map = {v: k for k, v in self.content_map.items()}
            similar_content = []
            
            for idx, distance in zip(similar_indices, distances):
                if idx != content_idx:
                    similar_id = reverse_map[idx]
                    if similar_id not in exclude_ids:
                        # Convert distance to similarity (1 - normalized_distance)
                        similarity = 1 - (distance / 2)  # Angular distance is in [0, 2]
                        similar_content.append((similar_id, similarity))
            
            return similar_content[:n_similar]
            
        except Exception as e:
            logger.error(f"Error finding similar content: {e}")
            return []
    
    def get_content_recommendations(self, user_profile: Dict[str, Any],
                                  candidate_content: List[Any],
                                  n_recommendations: int = 20) -> List[Tuple[Any, float]]:
        """Get content-based recommendations for user"""
        try:
            # Extract user preferences
            genre_prefs = user_profile.get('genre_preferences', {}).get('genre_scores', {})
            lang_prefs = user_profile.get('language_preferences', {}).get('preferred_languages', [])
            quality_threshold = user_profile.get('quality_preferences', {}).get('quality_threshold', 7.0)
            
            recommendations = []
            
            for content in candidate_content:
                score = 0.0
                
                # Genre matching
                if content.genres:
                    content_genres = safe_json_loads(content.genres, [])
                    genre_match = sum(genre_prefs.get(g, 0) for g in content_genres)
                    score += genre_match * 0.4
                
                # Language matching with Telugu priority
                if content.languages:
                    lang_score = LanguagePriorityManager.calculate_language_score(
                        content, lang_prefs
                    )
                    score += lang_score * 0.3
                
                # Quality score
                if content.rating and content.rating >= quality_threshold:
                    quality_score = (content.rating - quality_threshold) / (10 - quality_threshold)
                    score += quality_score * 0.2
                
                # Recency bonus for new content
                if content.release_date:
                    days_old = (datetime.now().date() - content.release_date).days
                    if days_old < 30:
                        recency_bonus = (30 - days_old) / 30 * 0.1
                        score += recency_bonus
                
                recommendations.append((content, score))
            
            # Sort by score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting content recommendations: {e}")
            return []

class DeepRankingModel:
    """LightGBM-based ranking model for final recommendation scoring"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.is_fitted = False
        
        # Model parameters optimized for ranking
        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10, 20],
            'learning_rate': 0.05,
            'num_leaves': 127,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
    
    def prepare_features(self, user_profile: Dict[str, Any],
                        content_list: List[Any],
                        cf_scores: Dict[int, float] = None,
                        cb_scores: Dict[int, float] = None) -> pd.DataFrame:
        """Prepare feature matrix for ranking"""
        features_list = []
        
        for content in content_list:
            features = {}
            
            # Content features
            features['content_id'] = content.id
            features['rating'] = content.rating or 0
            features['vote_count'] = content.vote_count or 0
            features['popularity'] = content.popularity or 0
            features['quality_score'] = calculate_content_quality_score(content)
            
            # Release date features
            if content.release_date:
                features['days_since_release'] = (datetime.now().date() - content.release_date).days
                features['is_new_release'] = features['days_since_release'] < 30
            else:
                features['days_since_release'] = 365
                features['is_new_release'] = 0
            
            # Content type features
            features['is_movie'] = int(content.content_type == 'movie')
            features['is_tv'] = int(content.content_type == 'tv')
            features['is_anime'] = int(content.content_type == 'anime')
            
            # Language features
            languages = safe_json_loads(content.languages, [])
            features['has_telugu'] = int(any('telugu' in l.lower() for l in languages))
            features['has_english'] = int(any('english' in l.lower() for l in languages))
            features['num_languages'] = len(languages)
            
            # Genre features
            genres = safe_json_loads(content.genres, [])
            features['num_genres'] = len(genres)
            
            # User preference alignment
            if user_profile:
                # Genre preference score
                genre_prefs = user_profile.get('genre_preferences', {}).get('genre_scores', {})
                features['genre_pref_score'] = sum(genre_prefs.get(g, 0) for g in genres)
                
                # Language preference score
                lang_prefs = user_profile.get('language_preferences', {}).get('preferred_languages', [])
                features['lang_pref_score'] = LanguagePriorityManager.calculate_language_score(
                    content, lang_prefs
                )
                
                # Quality alignment
                quality_threshold = user_profile.get('quality_preferences', {}).get('quality_threshold', 7.0)
                features['meets_quality'] = int(content.rating >= quality_threshold if content.rating else 0)
            
            # Algorithm scores
            features['cf_score'] = cf_scores.get(content.id, 0) if cf_scores else 0
            features['cb_score'] = cb_scores.get(content.id, 0) if cb_scores else 0
            
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        self.feature_columns = [col for col in df.columns if col != 'content_id']
        
        return df
    
    def fit(self, training_data: pd.DataFrame, 
            relevance_labels: np.ndarray,
            group_sizes: List[int] = None):
        """Train the ranking model"""
        try:
            X = training_data[self.feature_columns]
            y = relevance_labels
            
            # Create dataset
            train_data = lgb.Dataset(X, label=y, group=group_sizes)
            
            # Train model
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            self.is_fitted = True
            logger.info("Trained deep ranking model")
            
        except Exception as e:
            logger.error(f"Error training ranking model: {e}")
            self.is_fitted = False
    
    def predict_scores(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores"""
        if not self.is_fitted:
            # Return simple quality scores as fallback
            return features_df['quality_score'].values if 'quality_score' in features_df else np.zeros(len(features_df))
        
        try:
            X = features_df[self.feature_columns]
            scores = self.model.predict(X, num_iteration=self.model.best_iteration)
            return scores
            
        except Exception as e:
            logger.error(f"Error predicting scores: {e}")
            return features_df['quality_score'].values if 'quality_score' in features_df else np.zeros(len(features_df))

class CineBrainRecommendationEngine:
    """
    Main hybrid recommendation engine orchestrating all algorithms
    """
    
    def __init__(self, app=None, db=None, models=None, cache=None):
        self.app = app
        self.db = db
        self.models = models
        self.cache = CacheManager(cache)
        
        # Initialize components
        self.profile_analyzer = UserProfileAnalyzer(db, models, self.cache)
        self.cinematic_dna = CinematicDNAAnalyzer()
        self.cf_engine = CollaborativeFilteringEngine()
        self.cb_engine = ContentBasedEngine()
        self.ranking_model = DeepRankingModel()
        self.similarity_engine = UltraPowerfulSimilarityEngine()
        
        # Feedback and learning
        self.feedback_processor = FeedbackProcessor()
        self.online_learner = OnlineLearner()
        
        # Performance tracking
        self.tracker = PerformanceTracker()
        
        # Algorithm weights (adjustable)
        self.algorithm_weights = {
            'collaborative': 0.35,
            'content_based': 0.25,
            'popularity': 0.15,
            'cinematic_dna': 0.15,
            'feedback_adjustment': 0.10
        }
        
        # Initialize models if possible
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models with available data"""
        try:
            # Load interaction data
            if self.models and 'UserInteraction' in self.models:
                interactions = self.models['UserInteraction'].query.limit(10000).all()
                
                if interactions:
                    # Prepare interaction dataframe
                    interactions_data = []
                    for interaction in interactions:
                        interactions_data.append({
                            'user_id': interaction.user_id,
                            'content_id': interaction.content_id,
                            'interaction_type': interaction.interaction_type,
                            'rating': interaction.rating,
                            'timestamp': interaction.timestamp
                        })
                    
                    interactions_df = pd.DataFrame(interactions_data)
                    
                    # Fit collaborative filtering
                    self.cf_engine.fit(interactions_df)
                    
                    logger.info("Initialized collaborative filtering model")
            
            # Initialize content-based model
            if self.models and 'Content' in self.models:
                content_list = self.models['Content'].query.filter(
                    self.models['Content'].title.isnot(None)
                ).limit(5000).all()
                
                if content_list:
                    self.cb_engine.fit(content_list)
                    logger.info("Initialized content-based model")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def generate_personalized_recommendations(self, user_id: int,
                                            categories: List[str] = None,
                                            limit: int = 20,
                                            language_priority: bool = True,
                                            include_reasons: bool = False,
                                            diversity_factor: float = 0.3) -> Dict[str, Any]:
        """Generate personalized recommendations using hybrid approach"""
        try:
            # Build or retrieve user profile
            user_profile = self.profile_analyzer.build_comprehensive_user_profile(user_id)
            
            if not user_profile or user_profile.get('cold_start'):
                return self._get_cold_start_recommendations(user_id, limit)
            
            # Get recommendation categories
            if not categories:
                categories = self._get_default_categories(user_profile)
            
            all_recommendations = {}
            
            for category in categories:
                try:
                    recommendations = self.generate_category_recommendations(
                        user_id=user_id,
                        category=category,
                        limit=limit,
                        user_profile=user_profile,
                        language_priority=language_priority,
                        include_reasons=include_reasons
                    )
                    
                    if recommendations:
                        # Apply diversity injection
                        if diversity_factor > 0:
                            recommendations = self._inject_diversity(
                                recommendations, 
                                diversity_factor,
                                user_profile
                            )
                        
                        all_recommendations[category] = recommendations
                        
                except Exception as e:
                    logger.error(f"Error generating {category} recommendations: {e}")
                    continue
            
            # Build response
            response = {
                'user_id': user_id,
                'recommendations': all_recommendations,
                'profile_insights': self._generate_profile_insights(user_profile),
                'recommendation_metadata': {
                    'algorithm_version': '2.0.0',
                    'profile_completeness': user_profile.get('profile_completeness', 0),
                    'confidence_score': user_profile.get('confidence_score', 0),
                    'user_segment': user_profile.get('user_segment', 'unknown'),
                    'language_priority_applied': language_priority,
                    'diversity_factor': diversity_factor,
                    'generated_at': datetime.utcnow().isoformat()
                }
            }
            
            # Track metrics
            self.tracker.log_recommendation_served(
                user_id=user_id,
                categories=list(all_recommendations.keys()),
                recommendation_count=sum(len(recs) for recs in all_recommendations.values())
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating personalized recommendations: {e}")
            return self._get_fallback_recommendations(user_id, limit)
    
    def generate_category_recommendations(self, user_id: int,
                                        category: str,
                                        limit: int = 20,
                                        user_profile: Dict[str, Any] = None,
                                        language_priority: bool = True,
                                        include_reasons: bool = False) -> List[Dict[str, Any]]:
        """Generate recommendations for a specific category"""
        
        # Category to method mapping
        category_methods = {
            'cinebrain_for_you': self._generate_for_you,
            'because_you_watched': self._generate_because_you_watched,
            'trending_for_you': self._generate_trending_for_you,
            'new_releases_for_you': self._generate_new_releases_for_you,
            'your_language_priority': self._generate_language_priority,
            'cinematic_dna_matches': self._generate_cinematic_dna_matches,
            'hidden_gems': self._generate_hidden_gems,
            'critics_choice_personalized': self._generate_critics_choice_personalized,
            'similar_users_like': self._generate_similar_users_like,
            'genre_deep_dive': self._generate_genre_deep_dive
        }
        
        generator = category_methods.get(category)
        if not generator:
            logger.warning(f"Unknown category: {category}")
            return []
        
        # Get user profile if not provided
        if not user_profile:
            user_profile = self.profile_analyzer.build_comprehensive_user_profile(user_id)
        
        # Generate recommendations
        recommendations = generator(
            user_id=user_id,
            user_profile=user_profile,
            limit=limit,
            language_priority=language_priority,
            include_reasons=include_reasons
        )
        
        return recommendations
    
    def _generate_for_you(self, user_id: int, user_profile: Dict[str, Any],
                         limit: int, language_priority: bool,
                         include_reasons: bool) -> List[Dict[str, Any]]:
        """Generate main 'For You' recommendations using hybrid approach"""
        
        # Get candidate content pool
        candidate_content = self._get_candidate_pool(user_id, limit * 10)
        
        if not candidate_content:
            return []
        
        # Get scores from different algorithms
        scores = defaultdict(lambda: defaultdict(float))
        
        # 1. Collaborative Filtering scores
        if self.cf_engine.is_fitted:
            cf_predictions = self.cf_engine.predict_user_preferences(
                user_id,
                [c.id for c in candidate_content],
                limit * 2
            )
            
            for content_id, score in cf_predictions:
                scores[content_id]['collaborative'] = score
        
        # 2. Content-Based scores
        if self.cb_engine.is_fitted:
            cb_recommendations = self.cb_engine.get_content_recommendations(
                user_profile,
                candidate_content,
                limit * 2
            )
            
            for content, score in cb_recommendations:
                scores[content.id]['content_based'] = score
        
        # 3. Cinematic DNA scores
        cinematic_dna = user_profile.get('cinematic_dna', {})
        if cinematic_dna:
            dna_matches = self.cinematic_dna.find_cinematic_matches(
                cinematic_dna,
                candidate_content,
                limit * 2
            )
            
            for content, score, reason in dna_matches:
                scores[content.id]['cinematic_dna'] = score
                scores[content.id]['dna_reason'] = reason
        
        # 4. Popularity scores
        for content in candidate_content:
            pop_score = self._calculate_popularity_score(content)
            scores[content.id]['popularity'] = pop_score
        
        # 5. Feedback adjustment
        feedback_adjustments = self.online_learner.get_user_adjustments(user_id)
        for content_id, adjustment in feedback_adjustments.items():
            if content_id in scores:
                scores[content_id]['feedback'] = adjustment
        
        # Combine scores using weighted hybrid
        final_scores = []
        content_map = {c.id: c for c in candidate_content}
        
        for content_id, score_dict in scores.items():
            content = content_map.get(content_id)
            if not content:
                continue
            
            # Calculate weighted score
            final_score = 0
            for algo, weight in self.algorithm_weights.items():
                final_score += score_dict.get(algo, 0) * weight
            
            # Apply language priority boost
            if language_priority:
                lang_boost = LanguagePriorityManager.calculate_language_score(
                    content,
                    user_profile.get('language_preferences', {}).get('preferred_languages', [])
                )
                final_score *= (1 + lang_boost * 0.2)
            
            final_scores.append({
                'content': content,
                'score': final_score,
                'components': dict(score_dict),
                'reason': self._generate_recommendation_reason(content, score_dict, user_profile)
            })
        
        # Sort by final score
        final_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Format recommendations
        recommendations = []
        for item in final_scores[:limit]:
            rec = self._format_recommendation(
                item['content'],
                item['score'],
                item['reason'] if include_reasons else None,
                'hybrid_personalized'
            )
            
            if include_reasons:
                rec['score_breakdown'] = item['components']
            
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_because_you_watched(self, user_id: int, user_profile: Dict[str, Any],
                                    limit: int, language_priority: bool,
                                    include_reasons: bool) -> List[Dict[str, Any]]:
        """Generate 'Because You Watched' recommendations"""
        
        # Get recent viewed content
        recent_views = self.models['UserInteraction'].query.filter(
            and_(
                self.models['UserInteraction'].user_id == user_id,
                self.models['UserInteraction'].interaction_type.in_(['view', 'favorite', 'rating']),
                self.models['UserInteraction'].timestamp > datetime.utcnow() - timedelta(days=30)
            )
        ).order_by(desc(self.models['UserInteraction'].timestamp)).limit(5).all()
        
        if not recent_views:
            return []
        
        recommendations = []
        seen_ids = set()
        
        for interaction in recent_views:
            base_content = self.models['Content'].query.get(interaction.content_id)
            if not base_content:
                continue
            
            # Use ultra-similarity engine
            similar_results = self.similarity_engine.find_ultra_similar_content(
                base_content,
                self._get_candidate_pool(user_id, 100),
                limit=10,
                min_similarity=0.6,
                strict_mode=True
            )
            
            for result in similar_results:
                content = result['content']
                if content.id not in seen_ids:
                    seen_ids.add(content.id)
                    
                    reason = f"Because you watched {base_content.title}"
                    if include_reasons:
                        reason = result.get('match_explanation', {}).get('recommendation_note', reason)
                    
                    rec = self._format_recommendation(
                        content,
                        result['similarity_score'],
                        reason,
                        'content_similarity'
                    )
                    
                    rec['based_on'] = {
                        'content_id': base_content.id,
                        'title': base_content.title,
                        'interaction_type': interaction.interaction_type
                    }
                    
                    recommendations.append(rec)
                    
                    if len(recommendations) >= limit:
                        break
            
            if len(recommendations) >= limit:
                break
        
        return recommendations[:limit]
    
    def _generate_trending_for_you(self, user_id: int, user_profile: Dict[str, Any],
                                 limit: int, language_priority: bool,
                                 include_reasons: bool) -> List[Dict[str, Any]]:
        """Generate personalized trending content"""
        
        # Get trending content
        trending_content = self.models['Content'].query.filter(
            or_(
                self.models['Content'].is_trending == True,
                self.models['Content'].popularity > 50
            )
        ).order_by(desc(self.models['Content'].popularity)).limit(limit * 3).all()
        
        # Filter and score based on user preferences
        scored_content = []
        
        for content in trending_content:
            # Calculate personalization score
            score = self._calculate_personalization_score(content, user_profile)
            
            # Boost for language preference
            if language_priority:
                lang_score = LanguagePriorityManager.calculate_language_score(
                    content,
                    user_profile.get('language_preferences', {}).get('preferred_languages', [])
                )
                score *= (1 + lang_score * 0.3)
            
            reason = "Trending now"
            if include_reasons:
                genres = safe_json_loads(content.genres, [])
                matching_genres = [g for g in genres 
                                 if g in user_profile.get('genre_preferences', {}).get('top_genres', [])]
                if matching_genres:
                    reason = f"Trending in {', '.join(matching_genres[:2])}"
            
            scored_content.append({
                'content': content,
                'score': score,
                'reason': reason
            })
        
        # Sort by score and format
        scored_content.sort(key=lambda x: x['score'], reverse=True)
        
        recommendations = []
        for item in scored_content[:limit]:
            rec = self._format_recommendation(
                item['content'],
                item['score'],
                item['reason'],
                'trending_personalized'
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _calculate_personalization_score(self, content: Any, 
                                       user_profile: Dict[str, Any]) -> float:
        """Calculate how well content matches user preferences"""
        score = 0.0
        
        # Genre match
        if content.genres:
            genres = safe_json_loads(content.genres, [])
            genre_prefs = user_profile.get('genre_preferences', {}).get('genre_scores', {})
            genre_match = sum(genre_prefs.get(g, 0) for g in genres)
            score += genre_match * 0.4
        
        # Language match
        if content.languages:
            languages = safe_json_loads(content.languages, [])
            lang_prefs = user_profile.get('language_preferences', {}).get('preferred_languages', [])
            if any(lang in languages for lang in lang_prefs):
                score += 0.3
        
        # Quality match
        quality_threshold = user_profile.get('quality_preferences', {}).get('quality_threshold', 7.0)
        if content.rating and content.rating >= quality_threshold:
            score += 0.2
        
        # Content type match
        content_type_prefs = user_profile.get('content_type_preferences', {}).get('type_scores', {})
        score += content_type_prefs.get(content.content_type, 0) * 0.1
        
        return min(score, 1.0)
    
    def _format_recommendation(self, content: Any, score: float,
                             reason: str = None, source: str = None) -> Dict[str, Any]:
        """Format content as recommendation object"""
        return {
            'id': content.id,
            'title': content.title,
            'content_type': content.content_type,
            'genres': safe_json_loads(content.genres, []),
            'languages': safe_json_loads(content.languages, []),
            'rating': content.rating,
            'release_date': content.release_date.isoformat() if content.release_date else None,
            'poster_path': self._format_poster_path(content.poster_path),
            'overview': content.overview[:200] + '...' if content.overview else '',
            'recommendation_score': round(score, 4),
            'recommendation_reason': reason,
            'source_algorithm': source,
            'is_new_release': content.is_new_release if hasattr(content, 'is_new_release') else False,
            'is_trending': content.is_trending if hasattr(content, 'is_trending') else False,
            'youtube_trailer_id': content.youtube_trailer_id if hasattr(content, 'youtube_trailer_id') else None
        }
    
    def _format_poster_path(self, poster_path: str) -> str:
        """Format poster path to full URL"""
        if not poster_path:
            return None
        
        if poster_path.startswith('http'):
            return poster_path
        
        return f"https://image.tmdb.org/t/p/w300{poster_path}"
    
    def _get_candidate_pool(self, user_id: int, pool_size: int = 1000) -> List[Any]:
        """Get candidate content pool for recommendations"""
        
        # Get user's already interacted content
        interacted = self.models['UserInteraction'].query.filter_by(
            user_id=user_id
        ).with_entities(self.models['UserInteraction'].content_id).all()
        
        interacted_ids = [i[0] for i in interacted]
        
        # Get diverse content pool
        query = self.models['Content'].query
        
        if interacted_ids:
            query = query.filter(~self.models['Content'].id.in_(interacted_ids))
        
        # Quality filter
        query = query.filter(
            and_(
                self.models['Content'].title.isnot(None),
                self.models['Content'].rating >= 5.0
            )
        )
        
        # Get mix of popular and recent content
        popular_content = query.order_by(
            desc(self.models['Content'].popularity)
        ).limit(pool_size // 2).all()
        
        recent_content = query.order_by(
            desc(self.models['Content'].release_date)
        ).limit(pool_size // 2).all()
        
        # Combine and deduplicate
        content_pool = list({c.id: c for c in popular_content + recent_content}.values())
        
        return content_pool[:pool_size]
    
    # Additional category generators would follow...
    # [Implementations for other categories like hidden_gems, critics_choice_personalized, etc.]
    
    def _inject_diversity(self, recommendations: List[Dict[str, Any]],
                         diversity_factor: float,
                         user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Inject diversity into recommendations"""
        
        if len(recommendations) < 5 or diversity_factor == 0:
            return recommendations
        
        # Track seen attributes
        seen_genres = set()
        seen_languages = set()
        seen_types = set()
        
        diverse_recs = []
        held_back = []
        
        for rec in recommendations:
            genres = set(rec.get('genres', []))
            languages = set(rec.get('languages', []))
            content_type = rec.get('content_type')
            
            # Calculate novelty
            genre_novelty = len(genres - seen_genres) / max(len(genres), 1)
            lang_novelty = len(languages - seen_languages) / max(len(languages), 1)
            type_novelty = 1.0 if content_type not in seen_types else 0.0
            
            overall_novelty = (genre_novelty + lang_novelty + type_novelty) / 3
            
            # Accept if novel enough or in top positions
            if overall_novelty > diversity_factor or len(diverse_recs) < 3:
                diverse_recs.append(rec)
                seen_genres.update(genres)
                seen_languages.update(languages)
                seen_types.add(content_type)
            else:
                held_back.append(rec)
        
        # Add held back items if we need more
        while len(diverse_recs) < len(recommendations) and held_back:
            diverse_recs.append(held_back.pop(0))
        
        return diverse_recs
    
    def update_user_preferences_realtime(self, user_id: int, 
                                       interaction_data: Dict[str, Any]):
        """Update user preferences in real-time based on interactions"""
        try:
            # Process through feedback processor
            self.feedback_processor.process_feedback(
                user_id=user_id,
                content_id=interaction_data.get('content_id'),
                feedback_type=interaction_data.get('interaction_type'),
                feedback_value=interaction_data.get('rating'),
                context=interaction_data.get('metadata', {})
            )
            
            # Update online learner
            self.online_learner.update_user_model(user_id, interaction_data)
            
            # Invalidate cached profile
            if self.cache:
                cache_key = self.cache.get_user_profile_cache_key(user_id)
                self.cache.delete(cache_key)
            
            logger.info(f"Updated realtime preferences for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating realtime preferences: {e}")
    
    def get_user_recommendation_metrics(self, user_id: int) -> Dict[str, Any]:
        """Get recommendation performance metrics for user"""
        return self.tracker.get_user_metrics(user_id)
    
    def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        return {
            'cf_engine_fitted': self.cf_engine.is_fitted,
            'cb_engine_fitted': self.cb_engine.is_fitted,
            'ranking_model_fitted': self.ranking_model.is_fitted,
            'total_users': len(self.cf_engine.user_map) if self.cf_engine.user_map else 0,
            'total_items': len(self.cf_engine.item_map) if self.cf_engine.item_map else 0,
            'last_update': datetime.utcnow().isoformat()
        }
    
    def get_ultra_similar_content(self, base_content_id: int,
                                limit: int = 15,
                                strict_mode: bool = True,
                                min_similarity: float = 0.5,
                                include_explanations: bool = True) -> List[Dict[str, Any]]:
        """Get ultra-similar content using advanced similarity engine"""
        
        # Get base content
        base_content = self.models['Content'].query.get(base_content_id)
        if not base_content:
            return []
        
        # Get content pool
        content_pool = self._get_candidate_pool(0, 500)  # Use 0 for no user filtering
        
        # Find similar content
        similar_results = self.similarity_engine.find_ultra_similar_content(
            base_content,
            content_pool,
            limit=limit,
            min_similarity=min_similarity,
            strict_mode=strict_mode
        )
        
        # Format results
        recommendations = []
        for result in similar_results:
            rec = self._format_recommendation(
                result['content'],
                result['similarity_score'],
                result.get('match_type'),
                'ultra_similarity'
            )
            
            if include_explanations:
                rec['similarity_details'] = result.get('detail_scores', {})
                rec['match_explanation'] = result.get('match_type')
                rec['confidence'] = result.get('confidence')
            
            recommendations.append(rec)
        
        return recommendations
    
    # Fallback methods
    def _get_cold_start_recommendations(self, user_id: int, limit: int) -> Dict[str, Any]:
        """Get recommendations for new users"""
        
        # Get popular content by category
        popular_movies = self.models['Content'].query.filter(
            self.models['Content'].content_type == 'movie'
        ).order_by(desc(self.models['Content'].popularity)).limit(limit).all()
        
        popular_tv = self.models['Content'].query.filter(
            self.models['Content'].content_type == 'tv'
        ).order_by(desc(self.models['Content'].popularity)).limit(limit).all()
        
        # Telugu content (priority for CineBrain)
        telugu_content = self.models['Content'].query.filter(
            self.models['Content'].languages.contains('"Telugu"')
        ).order_by(desc(self.models['Content'].rating)).limit(limit).all()
        
        return {
            'user_id': user_id,
            'recommendations': {
                'popular_movies': [self._format_recommendation(c, 1.0, "Popular movie", "popularity") 
                                 for c in popular_movies],
                'popular_tv_shows': [self._format_recommendation(c, 1.0, "Popular TV show", "popularity") 
                                   for c in popular_tv],
                'telugu_favorites': [self._format_recommendation(c, 1.0, "Top Telugu content", "language") 
                                   for c in telugu_content]
            },
            'profile_insights': {
                'status': 'new_user',
                'message': 'Start watching and rating content to get personalized recommendations!'
            },
            'recommendation_metadata': {
                'type': 'cold_start',
                'algorithm_version': '2.0.0'
            }
        }
    
    def _get_fallback_recommendations(self, user_id: int, limit: int) -> Dict[str, Any]:
        """Fallback recommendations when main algorithms fail"""
        
        trending = self.models['Content'].query.filter(
            self.models['Content'].is_trending == True
        ).order_by(desc(self.models['Content'].popularity)).limit(limit).all()
        
        return {
            'user_id': user_id,
            'recommendations': {
                'trending_now': [self._format_recommendation(c, 1.0, "Trending now", "fallback") 
                               for c in trending]
            },
            'error': 'Using fallback recommendations',
            'recommendation_metadata': {
                'type': 'fallback',
                'algorithm_version': '2.0.0'
            }
        }
    
    def _get_default_categories(self, user_profile: Dict[str, Any]) -> List[str]:
        """Get default recommendation categories based on user profile"""
        
        if user_profile.get('cold_start'):
            return ['popular_movies', 'popular_tv_shows', 'telugu_favorites']
        
        engagement_level = user_profile.get('engagement_metrics', {}).get('engagement_level', 'low')
        
        if engagement_level == 'high':
            return [
                'cinebrain_for_you',
                'because_you_watched',
                'hidden_gems',
                'cinematic_dna_matches',
                'genre_deep_dive'
            ]
        elif engagement_level == 'medium':
            return [
                'cinebrain_for_you',
                'trending_for_you',
                'new_releases_for_you',
                'your_language_priority'
            ]
        else:
            return [
                'trending_for_you',
                'popular_movies',
                'your_language_priority'
            ]
    
    def _generate_profile_insights(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about user profile"""
        
        return {
            'profile_strength': 'strong' if user_profile.get('confidence_score', 0) > 0.7 else 'building',
            'cinematic_sophistication': user_profile.get('cinematic_dna', {}).get('cinematic_sophistication', 0.5),
            'dominant_themes': list(user_profile.get('cinematic_dna', {}).get('dominant_themes', {}).keys())[:3],
            'preferred_languages': user_profile.get('language_preferences', {}).get('preferred_languages', [])[:3],
            'taste_complexity': 'eclectic' if user_profile.get('genre_preferences', {}).get('genre_diversity', 0) > 5 else 'focused',
            'recommendation_accuracy': user_profile.get('confidence_score', 0) * 100
        }
    
    def _calculate_popularity_score(self, content: Any) -> float:
        """Calculate normalized popularity score"""
        
        score = 0.0
        
        if content.popularity:
            score += min(content.popularity / 100, 1.0) * 0.4
        
        if content.vote_count:
            score += min(content.vote_count / 1000, 1.0) * 0.3
        
        if content.rating:
            score += (content.rating / 10) * 0.3
        
        return score
    
    def _generate_recommendation_reason(self, content: Any,
                                      score_components: Dict[str, float],
                                      user_profile: Dict[str, Any]) -> str:
        """Generate human-readable recommendation reason"""
        
        # Find strongest signal
        top_component = max(score_components.items(), key=lambda x: x[1])[0]
        
        if top_component == 'cinematic_dna' and 'dna_reason' in score_components:
            return score_components['dna_reason']
        elif top_component == 'collaborative':
            return "Users with similar taste loved this"
        elif top_component == 'content_based':
            return "Matches your content preferences"
        elif top_component == 'popularity':
            return "Popular and highly rated"
        else:
            return "Recommended for you"
    
    # Placeholder methods for additional categories
    def _generate_new_releases_for_you(self, **kwargs):
        """Generate personalized new releases"""
        # Implementation would follow pattern of other generators
        return []
    
    def _generate_language_priority(self, **kwargs):
        """Generate language-based recommendations"""
        # Implementation would follow pattern of other generators
        return []
    
    def _generate_cinematic_dna_matches(self, **kwargs):
        """Generate pure cinematic DNA matches"""
        # Implementation would follow pattern of other generators
        return []
    
    def _generate_hidden_gems(self, **kwargs):
        """Generate hidden gem recommendations"""
        # Implementation would follow pattern of other generators
        return []
    
    def _generate_critics_choice_personalized(self, **kwargs):
        """Generate personalized critics' choice"""
        # Implementation would follow pattern of other generators
        return []
    
    def _generate_similar_users_like(self, **kwargs):
        """Generate recommendations based on similar users"""
        # Implementation would follow pattern of other generators
        return []
    
    def _generate_genre_deep_dive(self, **kwargs):
        """Generate deep genre exploration recommendations"""
        # Implementation would follow pattern of other generators
        return []