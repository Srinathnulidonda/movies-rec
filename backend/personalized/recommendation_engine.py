# backend/personalized/recommendation_engine.py`

"""
CineBrain Hybrid Recommendation Engine
Modern AI-powered recommendation system with multi-algorithm fusion
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import networkx as nx
import json
import logging
import random
from typing import List, Dict, Any, Tuple, Optional, Set
from heapq import heappush, heappop

from .utils import (
    TeluguPriorityManager,
    EmbeddingManager,
    SimilarityEngine,
    CacheManager,
    safe_json_loads,
    normalize_scores,
    calculate_diversity_score
)
from .profile_analyzer import UserProfileAnalyzer

logger = logging.getLogger(__name__)

class CollaborativeEngine:
    """
    Matrix factorization-based collaborative filtering
    """
    
    def __init__(self, n_factors: int = 50, max_iter: int = 200):
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}
        self.is_fitted = False
        
        # NMF for matrix factorization
        self.nmf = NMF(
            n_components=n_factors,
            max_iter=max_iter,
            init='nndsvd',
            random_state=42
        )
    
    def fit(self, interaction_matrix: pd.DataFrame):
        """
        Fit collaborative filtering model
        
        Args:
            interaction_matrix: User-item interaction matrix
        """
        try:
            if interaction_matrix.empty:
                logger.warning("Empty interaction matrix for collaborative filtering")
                return
            
            # Create user and item mappings
            self.user_map = {uid: idx for idx, uid in enumerate(interaction_matrix.index)}
            self.item_map = {iid: idx for idx, iid in enumerate(interaction_matrix.columns)}
            
            # Convert to numpy array
            matrix = interaction_matrix.values
            
            # Apply NMF
            self.user_factors = self.nmf.fit_transform(matrix)
            self.item_factors = self.nmf.components_.T
            
            self.is_fitted = True
            logger.info(f"Fitted collaborative model with {len(self.user_map)} users and {len(self.item_map)} items")
            
        except Exception as e:
            logger.error(f"Error fitting collaborative model: {e}")
            self.is_fitted = False
    
    def predict_scores(self, user_id: int, item_ids: List[int]) -> Dict[int, float]:
        """
        Predict scores for user-item pairs
        
        Args:
            user_id: User ID
            item_ids: List of item IDs
        
        Returns:
            Dict[int, float]: Item scores
        """
        if not self.is_fitted or user_id not in self.user_map:
            return {item_id: 0.0 for item_id in item_ids}
        
        try:
            user_idx = self.user_map[user_id]
            user_vec = self.user_factors[user_idx]
            
            scores = {}
            for item_id in item_ids:
                if item_id in self.item_map:
                    item_idx = self.item_map[item_id]
                    item_vec = self.item_factors[item_idx]
                    score = np.dot(user_vec, item_vec)
                    scores[item_id] = float(score)
                else:
                    scores[item_id] = 0.0
            
            return scores
            
        except Exception as e:
            logger.error(f"Error predicting collaborative scores: {e}")
            return {item_id: 0.0 for item_id in item_ids}

class ContentBasedEngine:
    """
    Content-based filtering using embeddings and features
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.feature_weights = {
            'embedding': 0.4,
            'genre': 0.25,
            'language': 0.2,
            'quality': 0.15
        }
    
    def compute_scores(self, user_profile: Dict[str, Any], 
                      content_list: List[Any]) -> Dict[int, float]:
        """
        Compute content-based scores for items
        
        Args:
            user_profile: User profile
            content_list: List of content items
        
        Returns:
            Dict[int, float]: Content scores
        """
        scores = {}
        user_id = user_profile.get('user_id', 0)
        
        # Get user embedding similarity scores
        embedding_scores = self.embedding_manager.compute_user_content_similarity(
            user_id, content_list
        )
        
        # Get user preferences
        genre_prefs = user_profile.get('implicit_preferences', {}).get('genre_preferences', {}).get('counts', {})
        lang_prefs = user_profile.get('explicit_preferences', {}).get('preferred_languages', [])
        quality_threshold = user_profile.get('recommendation_context', {}).get('quality_preference', 7.0)
        
        for idx, content in enumerate(content_list):
            score_components = {}
            
            # Embedding similarity
            score_components['embedding'] = float(embedding_scores[idx])
            
            # Genre matching
            if content.genres:
                genres = safe_json_loads(content.genres)
                genre_score = sum(genre_prefs.get(g, 0) for g in genres) / max(sum(genre_prefs.values()), 1)
                score_components['genre'] = min(genre_score, 1.0)
            else:
                score_components['genre'] = 0.0
            
            # Language matching with Telugu priority
            if content.languages:
                languages = safe_json_loads(content.languages)
                lang_score = TeluguPriorityManager.calculate_language_score(languages, lang_prefs)
                score_components['language'] = lang_score
            else:
                score_components['language'] = 0.0
            
            # Quality score
            if content.rating and content.rating >= quality_threshold:
                quality_score = (content.rating - quality_threshold) / (10 - quality_threshold)
                score_components['quality'] = quality_score
            else:
                score_components['quality'] = 0.0
            
            # Weighted combination
            final_score = sum(
                score_components[key] * self.feature_weights[key] 
                for key in self.feature_weights
            )
            
            scores[content.id] = final_score
        
        return scores

class GraphBasedEngine:
    """
    Graph-based recommendations using content and user relationships
    """
    
    def __init__(self):
        self.content_graph = nx.Graph()
        self.user_graph = nx.Graph()
        self.is_built = False
    
    def build_content_graph(self, content_list: List[Any], similarity_engine: SimilarityEngine):
        """
        Build content similarity graph
        
        Args:
            content_list: List of content items
            similarity_engine: Similarity computation engine
        """
        try:
            logger.info(f"Building content graph with {len(content_list)} items")
            
            # Add nodes
            for content in content_list:
                self.content_graph.add_node(
                    content.id,
                    title=content.title,
                    genres=safe_json_loads(content.genres or '[]'),
                    languages=safe_json_loads(content.languages or '[]'),
                    rating=content.rating or 0
                )
            
            # Add edges based on similarity
            # Sample pairs to avoid O(n²) complexity for large datasets
            max_pairs = min(10000, len(content_list) * 20)
            sampled_pairs = []
            
            for i, content1 in enumerate(content_list):
                # For each content, find similar items
                similarities = []
                
                for j, content2 in enumerate(content_list):
                    if i >= j:
                        continue
                    
                    sim_scores = similarity_engine.compute_content_similarity(content1, content2)
                    avg_sim = np.mean(list(sim_scores.values()))
                    
                    if avg_sim > 0.5:  # Threshold for edge creation
                        similarities.append((j, avg_sim))
                
                # Keep top-k similar items
                similarities.sort(key=lambda x: x[1], reverse=True)
                for j, sim in similarities[:5]:  # Top 5 similar items
                    sampled_pairs.append((i, j, sim))
                
                if len(sampled_pairs) >= max_pairs:
                    break
            
            # Add edges
            for i, j, weight in sampled_pairs:
                self.content_graph.add_edge(
                    content_list[i].id,
                    content_list[j].id,
                    weight=weight
                )
            
            self.is_built = True
            logger.info(f"Content graph built with {self.content_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error building content graph: {e}")
            self.is_built = False
    
    def get_graph_recommendations(self, seed_content_ids: List[int], 
                                 limit: int = 20) -> List[Tuple[int, float]]:
        """
        Get recommendations using graph propagation
        
        Args:
            seed_content_ids: Starting content IDs
            limit: Number of recommendations
        
        Returns:
            List[Tuple[int, float]]: Recommended content IDs with scores
        """
        if not self.is_built or not seed_content_ids:
            return []
        
        try:
            # Personalized PageRank from seed nodes
            personalization = {
                node: 1.0 if node in seed_content_ids else 0.0
                for node in self.content_graph.nodes()
            }
            
            pagerank_scores = nx.pagerank(
                self.content_graph,
                personalization=personalization,
                alpha=0.85,
                max_iter=100
            )
            
            # Filter out seed nodes and sort
            recommendations = [
                (node, score) 
                for node, score in pagerank_scores.items() 
                if node not in seed_content_ids
            ]
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in graph recommendations: {e}")
            return []

class ClusteringEngine:
    """
    User and content clustering for personalized recommendations
    """
    
    def __init__(self, profile_analyzer: UserProfileAnalyzer):
        self.profile_analyzer = profile_analyzer
        self.user_clusters = {}
        self.content_clusters = {}
    
    def get_cluster_recommendations(self, user_id: int, 
                                  content_pool: List[Any],
                                  limit: int = 20) -> List[Tuple[Any, float]]:
        """
        Get recommendations based on user clustering
        
        Args:
            user_id: User ID
            content_pool: Available content
            limit: Number of recommendations
        
        Returns:
            List[Tuple[Any, float]]: Content with scores
        """
        try:
            # Get user's cluster preferences
            user_profile = self.profile_analyzer.build_user_profile(user_id)
            user_clusters = user_profile.get('preference_clusters', {})
            
            if not user_clusters or not user_clusters.get('primary_cluster'):
                return []
            
            primary_cluster = user_clusters['primary_cluster']
            
            # Score content based on cluster alignment
            scores = []
            for content in content_pool:
                # Simple scoring based on content features matching cluster
                # In production, this would use more sophisticated cluster matching
                score = random.uniform(0.4, 0.8)  # Placeholder
                scores.append((content, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:limit]
            
        except Exception as e:
            logger.error(f"Error in cluster recommendations: {e}")
            return []

class HybridRecommendationEngine:
    """
    Main recommendation engine orchestrating multiple algorithms
    """
    
    def __init__(self, db=None, models=None, cache_manager=None):
        self.db = db
        self.models = models or {}
        self.cache_manager = cache_manager or CacheManager()
        
        # Initialize components
        self.profile_analyzer = UserProfileAnalyzer(db, models, cache_manager=cache_manager)
        self.embedding_manager = EmbeddingManager(cache_manager=cache_manager)
        self.similarity_engine = SimilarityEngine(self.embedding_manager, cache_manager)
        
        # Initialize recommendation engines
        self.collaborative_engine = CollaborativeEngine()
        self.content_based_engine = ContentBasedEngine(self.embedding_manager)
        self.graph_engine = GraphBasedEngine()
        self.clustering_engine = ClusteringEngine(self.profile_analyzer)
        
        # Algorithm weights (dynamic per user segment)
        self.default_weights = {
            'collaborative': 0.25,
            'content_based': 0.30,
            'graph_based': 0.20,
            'clustering': 0.15,
            'popularity': 0.10
        }
        
        # Exploration parameters
        self.exploration_rate = 0.15
        self.diversity_weight = 0.2
        
        # Performance tracking
        self.recommendation_history = defaultdict(list)
        
        # Initialize models if possible
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models with available data"""
        try:
            if not self.models or not self.db:
                logger.warning("Cannot initialize models: missing database or models")
                return
            
            # Load content for embeddings
            Content = self.models.get('Content')
            if Content:
                content_list = Content.query.filter(
                    Content.title.isnot(None)
                ).limit(5000).all()
                
                if content_list:
                    # Fit content embeddings
                    self.embedding_manager.fit_content_embeddings(content_list)
                    
                    # Build content graph
                    self.graph_engine.build_content_graph(
                        content_list[:1000],  # Limit for performance
                        self.similarity_engine
                    )
                    
                    logger.info("Initialized content embeddings and graph")
            
            # Load interaction matrix for collaborative filtering
            UserInteraction = self.models.get('UserInteraction')
            if UserInteraction:
                # Build interaction matrix
                interactions = UserInteraction.query.limit(10000).all()
                
                if len(interactions) > 100:
                    interaction_df = self._build_interaction_matrix(interactions)
                    if not interaction_df.empty:
                        self.collaborative_engine.fit(interaction_df)
                        logger.info("Initialized collaborative filtering model")
                    
        except Exception as e:
            logger.error(f"Error initializing recommendation models: {e}")
    
    def _build_interaction_matrix(self, interactions: List[Any]) -> pd.DataFrame:
        """Build user-item interaction matrix"""
        try:
            # Create sparse matrix representation
            user_item_data = defaultdict(lambda: defaultdict(float))
            
            for interaction in interactions:
                user_id = interaction.user_id
                content_id = interaction.content_id
                
                # Weight by interaction type
                weight_map = {
                    'favorite': 5.0,
                    'rating': 4.0,
                    'like': 3.0,
                    'view': 2.0,
                    'search': 1.0
                }
                
                weight = weight_map.get(interaction.interaction_type, 1.0)
                
                # Add rating if available
                if interaction.rating:
                    weight *= (interaction.rating / 10.0)
                
                user_item_data[user_id][content_id] = max(
                    user_item_data[user_id][content_id],
                    weight
                )
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(user_item_data, orient='index')
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error building interaction matrix: {e}")
            return pd.DataFrame()
    
    def generate_recommendations(self, user_id: int, limit: int = 20,
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate personalized recommendations using hybrid approach
        
        Args:
            user_id: User ID
            limit: Number of recommendations
            context: Additional context (device, time, etc.)
        
        Returns:
            Dict containing recommendations and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Build/retrieve user profile
            user_profile = self.profile_analyzer.build_user_profile(user_id)
            
            # Get candidate content pool
            candidates = self._get_candidate_pool(user_id, user_profile, limit * 20)
            
            if not candidates:
                return self._get_fallback_recommendations(user_id, limit)
            
            # Get algorithm weights based on user segment
            weights = self._get_personalized_weights(user_profile)
            
            # Collect scores from different algorithms
            all_scores = defaultdict(lambda: defaultdict(float))
            
            # 1. Collaborative filtering scores
            if self.collaborative_engine.is_fitted and weights['collaborative'] > 0:
                cf_scores = self.collaborative_engine.predict_scores(
                    user_id,
                    [c.id for c in candidates]
                )
                for content_id, score in cf_scores.items():
                    all_scores[content_id]['collaborative'] = score
            
            # 2. Content-based scores
            if weights['content_based'] > 0:
                cb_scores = self.content_based_engine.compute_scores(
                    user_profile,
                    candidates
                )
                for content_id, score in cb_scores.items():
                    all_scores[content_id]['content_based'] = score
            
            # 3. Graph-based scores
            if self.graph_engine.is_built and weights['graph_based'] > 0:
                # Use user's recent interactions as seeds
                seed_ids = self._get_user_seed_content(user_id)
                if seed_ids:
                    graph_recs = self.graph_engine.get_graph_recommendations(
                        seed_ids,
                        limit * 2
                    )
                    for content_id, score in graph_recs:
                        all_scores[content_id]['graph_based'] = score
            
            # 4. Clustering scores
            if weights['clustering'] > 0:
                cluster_recs = self.clustering_engine.get_cluster_recommendations(
                    user_id,
                    candidates,
                    limit * 2
                )
                for content, score in cluster_recs:
                    all_scores[content.id]['clustering'] = score
            
            # 5. Popularity scores
            if weights['popularity'] > 0:
                for content in candidates:
                    pop_score = self._calculate_popularity_score(content)
                    all_scores[content.id]['popularity'] = pop_score
            
            # Combine scores using weighted average
            final_scores = self._combine_scores(all_scores, weights, candidates)
            
            # Apply context adjustments
            if context:
                final_scores = self._apply_context_adjustments(final_scores, context, user_profile)
            
            # Rank and select top items
            ranked_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Apply diversity injection
            diverse_items = self._inject_diversity(ranked_items, candidates, limit)
            
            # Apply exploration
            final_items = self._apply_exploration(diverse_items, candidates, limit)
            
            # Format response
            recommendations = self._format_recommendations(final_items, user_profile)
            
            # Track performance
            self._track_recommendation_event(user_id, recommendations, context)
            
            # Calculate metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                'status': 'success',
                'user_id': user_id,
                'recommendations': recommendations,
                'metadata': {
                    'algorithm_weights': weights,
                    'total_candidates': len(candidates),
                    'user_segment': user_profile.get('user_segment', 'unknown'),
                    'processing_time_ms': int(processing_time * 1000),
                    'diversity_score': calculate_diversity_score([item['content'] for item in recommendations]),
                    'timestamp': datetime.utcnow().isoformat(),
                    'profile_confidence': user_profile.get('confidence_score', 0),
                    'telugu_content_ratio': self._calculate_telugu_ratio(recommendations)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return self._get_error_response(user_id, str(e))
    
    def _get_candidate_pool(self, user_id: int, user_profile: Dict[str, Any],
                          pool_size: int = 500) -> List[Any]:
        """Get candidate content pool for recommendations"""
        try:
            Content = self.models.get('Content')
            UserInteraction = self.models.get('UserInteraction')
            
            if not Content:
                return []
            
            # Get user's already interacted content
            interacted_ids = set()
            if UserInteraction:
                interactions = UserInteraction.query.filter_by(
                    user_id=user_id
                ).with_entities(UserInteraction.content_id).all()
                interacted_ids = {i[0] for i in interactions}
            
            # Build query
            query = Content.query.filter(
                Content.id.notin_(interacted_ids),
                Content.title.isnot(None)
            )
            
            # Quality filter based on user preference
            quality_threshold = user_profile.get('recommendation_context', {}).get('quality_preference', 6.0)
            query = query.filter(Content.rating >= quality_threshold)
            
            # Get diverse content pool
            # 1. Recent releases
            recent_date = datetime.utcnow().date() - timedelta(days=90)
            recent_content = query.filter(
                Content.release_date >= recent_date
            ).limit(pool_size // 3).all()
            
            # 2. Popular content
            popular_content = query.order_by(
                Content.popularity.desc()
            ).limit(pool_size // 3).all()
            
            # 3. High-rated content
            rated_content = query.order_by(
                Content.rating.desc()
            ).limit(pool_size // 3).all()
            
            # Combine and deduplicate
            all_content = list({c.id: c for c in recent_content + popular_content + rated_content}.values())
            
            # Apply Telugu-first sorting
            user_languages = user_profile.get('explicit_preferences', {}).get('preferred_languages', [])
            sorted_content = TeluguPriorityManager.sort_by_language_priority(
                all_content,
                user_languages
            )
            
            return sorted_content[:pool_size]
            
        except Exception as e:
            logger.error(f"Error getting candidate pool: {e}")
            return []
    
    def _get_personalized_weights(self, user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Get algorithm weights based on user profile"""
        user_segment = user_profile.get('user_segment', 'new_user')
        confidence = user_profile.get('confidence_score', 0)
        
        weights = self.default_weights.copy()
        
        if user_segment == 'new_user':
            # New users - rely more on popularity and content
            weights['collaborative'] = 0.05
            weights['content_based'] = 0.25
            weights['graph_based'] = 0.10
            weights['clustering'] = 0.10
            weights['popularity'] = 0.50
        
        elif user_segment == 'regular_user':
            # Regular users - balanced approach
            weights['collaborative'] = 0.20
            weights['content_based'] = 0.30
            weights['graph_based'] = 0.20
            weights['clustering'] = 0.15
            weights['popularity'] = 0.15
        
        elif user_segment in ['active_user', 'power_user']:
            # Active users - personalized algorithms
            weights['collaborative'] = 0.35
            weights['content_based'] = 0.25
            weights['graph_based'] = 0.25
            weights['clustering'] = 0.10
            weights['popularity'] = 0.05
        
        # Adjust based on confidence
        if confidence < 0.3:
            # Low confidence - increase popularity weight
            popularity_boost = (0.3 - confidence) * 0.5
            weights['popularity'] = min(weights['popularity'] + popularity_boost, 0.6)
            
            # Normalize other weights
            remaining = 1.0 - weights['popularity']
            other_sum = sum(v for k, v in weights.items() if k != 'popularity')
            
            if other_sum > 0:
                for k in weights:
                    if k != 'popularity':
                        weights[k] = (weights[k] / other_sum) * remaining
        
        return weights
    
    def _combine_scores(self, all_scores: Dict[int, Dict[str, float]],
                       weights: Dict[str, float],
                       candidates: List[Any]) -> Dict[int, float]:
        """Combine scores from different algorithms"""
        combined_scores = {}
        content_map = {c.id: c for c in candidates}
        
        for content_id, algo_scores in all_scores.items():
            if content_id not in content_map:
                continue
            
            # Weighted average of algorithm scores
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for algo, score in algo_scores.items():
                if algo in weights and score > 0:
                    # Normalize score to 0-1 range
                    normalized_score = min(max(score, 0), 1)
                    weighted_sum += normalized_score * weights[algo]
                    weight_sum += weights[algo]
            
            if weight_sum > 0:
                combined_scores[content_id] = weighted_sum / weight_sum
            else:
                combined_scores[content_id] = 0.0
            
            # Apply language boost
            content = content_map[content_id]
            if content.languages:
                languages = safe_json_loads(content.languages)
                lang_score = TeluguPriorityManager.calculate_language_score(languages)
                combined_scores[content_id] *= (1 + lang_score * 0.2)  # Up to 20% boost
        
        return combined_scores
    
    def _apply_context_adjustments(self, scores: Dict[int, float],
                                 context: Dict[str, Any],
                                 user_profile: Dict[str, Any]) -> Dict[int, float]:
        """Apply contextual adjustments to scores"""
        adjusted_scores = scores.copy()
        Content = self.models.get('Content')
        
        if not Content:
            return adjusted_scores
        
        # Time of day adjustments
        current_hour = datetime.utcnow().hour
        
        for content_id, score in scores.items():
            content = Content.query.get(content_id)
            if not content:
                continue
            
            adjustment_factor = 1.0
            
            # Device-based adjustments
            if context.get('device') == 'mobile':
                # Prefer shorter content on mobile
                if content.runtime and content.runtime < 90:
                    adjustment_factor *= 1.1
                elif content.runtime and content.runtime > 150:
                    adjustment_factor *= 0.9
            
            # Time-based adjustments
            if 22 <= current_hour or current_hour < 2:
                # Late night - prefer certain genres
                if content.genres:
                    genres = safe_json_loads(content.genres)
                    if any(g in ['Horror', 'Thriller', 'Mystery'] for g in genres):
                        adjustment_factor *= 1.2
            elif 6 <= current_hour < 9:
                # Morning - prefer lighter content
                if content.genres:
                    genres = safe_json_loads(content.genres)
                    if any(g in ['Comedy', 'Animation', 'Family'] for g in genres):
                        adjustment_factor *= 1.2
            
            # Weekend adjustments
            if datetime.utcnow().weekday() >= 5:  # Saturday or Sunday
                # Prefer movies and longer content
                if content.content_type == 'movie':
                    adjustment_factor *= 1.1
            
            adjusted_scores[content_id] = score * adjustment_factor
        
        return adjusted_scores
    
    def _inject_diversity(self, ranked_items: List[Tuple[int, float]],
                         candidates: List[Any],
                         limit: int) -> List[Tuple[int, float]]:
        """Inject diversity into recommendations"""
        if len(ranked_items) <= 5:
            return ranked_items[:limit]
        
        content_map = {c.id: c for c in candidates}
        diverse_items = []
        
        # Track seen attributes
        seen_genres = set()
        seen_types = set()
        seen_languages = set()
        
        # Always include top items
        for i, (content_id, score) in enumerate(ranked_items[:3]):
            diverse_items.append((content_id, score))
            content = content_map.get(content_id)
            if content:
                if content.genres:
                    seen_genres.update(safe_json_loads(content.genres))
                seen_types.add(content.content_type)
                if content.languages:
                    seen_languages.update(safe_json_loads(content.languages))
        
        # Add diverse items
        for content_id, score in ranked_items[3:]:
            if len(diverse_items) >= limit:
                break
            
            content = content_map.get(content_id)
            if not content:
                continue
            
            # Calculate novelty
            genres = set(safe_json_loads(content.genres or '[]'))
            languages = set(safe_json_loads(content.languages or '[]'))
            
            genre_novelty = len(genres - seen_genres) / max(len(genres), 1) if genres else 0
            lang_novelty = len(languages - seen_languages) / max(len(languages), 1) if languages else 0
            type_novelty = 1.0 if content.content_type not in seen_types else 0.0
            
            overall_novelty = (genre_novelty + lang_novelty + type_novelty) / 3
            
            # Accept if novel enough or score is very high
            if overall_novelty > self.diversity_weight or score > 0.85:
                diverse_items.append((content_id, score))
                seen_genres.update(genres)
                seen_types.add(content.content_type)
                seen_languages.update(languages)
        
        return diverse_items
    
    def _apply_exploration(self, items: List[Tuple[int, float]],
                          candidates: List[Any],
                          limit: int) -> List[Tuple[int, float]]:
        """Apply exploration strategy"""
        if random.random() > self.exploration_rate:
            return items[:limit]
        
        # Reserve slots for exploration
        exploit_count = int(limit * (1 - self.exploration_rate))
        explore_count = limit - exploit_count
        
        # Keep top items
        final_items = items[:exploit_count]
        
        # Add random exploration items
        content_map = {c.id: c for c in candidates}
        already_selected = {item[0] for item in final_items}
        
        exploration_pool = [
            c for c in candidates 
            if c.id not in already_selected
        ]
        
        if exploration_pool:
            # Weighted random selection favoring quality
            weights = []
            for content in exploration_pool:
                weight = 1.0
                if content.rating:
                    weight *= (content.rating / 10.0)
                if content.languages:
                    languages = safe_json_loads(content.languages)
                    lang_score = TeluguPriorityManager.calculate_language_score(languages)
                    weight *= (1 + lang_score)
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(exploration_pool)] * len(exploration_pool)
            
            # Sample exploration items
            explore_items = np.random.choice(
                exploration_pool,
                size=min(explore_count, len(exploration_pool)),
                replace=False,
                p=weights
            )
            
            for content in explore_items:
                final_items.append((content.id, random.uniform(0.3, 0.5)))
        
        return final_items
    
    def _format_recommendations(self, items: List[Tuple[int, float]],
                              user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format recommendations for response"""
        Content = self.models.get('Content')
        if not Content:
            return []
        
        formatted = []
        
        for rank, (content_id, score) in enumerate(items):
            content = Content.query.get(content_id)
            if not content:
                continue
            
            # Determine recommendation reason
            reason = self._generate_recommendation_reason(content, user_profile)
            
            formatted_item = {
                'rank': rank + 1,
                'content': {
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': safe_json_loads(content.genres or '[]'),
                    'languages': safe_json_loads(content.languages or '[]'),
                    'rating': content.rating,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'overview': content.overview[:200] + '...' if content.overview else '',
                    'runtime': content.runtime,
                    'popularity': content.popularity,
                    'vote_count': content.vote_count,
                    'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None,
                    'backdrop_path': f"https://image.tmdb.org/t/p/w1280{content.backdrop_path}" if content.backdrop_path else None,
                    'is_trending': getattr(content, 'is_trending', False),
                    'is_new_release': getattr(content, 'is_new_release', False),
                    'youtube_trailer_id': getattr(content, 'youtube_trailer_id', None)
                },
                'recommendation_score': round(score, 4),
                'reason': reason,
                'match_percentage': int(score * 100),
                'personalization_type': self._get_personalization_type(content, user_profile)
            }
            
            formatted.append(formatted_item)
        
        return formatted
    
    def _generate_recommendation_reason(self, content: Any,
                                      user_profile: Dict[str, Any]) -> str:
        """Generate human-readable recommendation reason"""
        
        # Check for Telugu content
        if content.languages:
            languages = safe_json_loads(content.languages)
            if any('telugu' in lang.lower() for lang in languages):
                return "Top Telugu content matching your taste"
        
        # Check genre match
        user_genres = user_profile.get('implicit_preferences', {}).get('genre_preferences', {}).get('top_genres', [])
        if content.genres and user_genres:
            content_genres = safe_json_loads(content.genres)
            matching_genres = set(content_genres) & set(user_genres)
            if matching_genres:
                return f"Because you enjoy {', '.join(list(matching_genres)[:2])}"
        
        # Check if new release
        if hasattr(content, 'is_new_release') and content.is_new_release:
            return "New release you might enjoy"
        
        # Check rating
        if content.rating and content.rating >= 8:
            return "Highly rated by viewers like you"
        
        # Default reason
        return "Recommended for you"
    
    def _get_personalization_type(self, content: Any,
                                 user_profile: Dict[str, Any]) -> str:
        """Determine personalization type for content"""
        
        # Check profile confidence
        confidence = user_profile.get('confidence_score', 0)
        
        if confidence < 0.3:
            return 'popularity_based'
        elif confidence < 0.6:
            return 'hybrid'
        else:
            return 'personalized'
    
    def _calculate_popularity_score(self, content: Any) -> float:
        """Calculate normalized popularity score"""
        score = 0.0
        
        if content.popularity:
            score += min(content.popularity / 100, 1.0) * 0.4
        
        if content.vote_count:
            score += min(content.vote_count / 1000, 1.0) * 0.3
        
        if content.rating:
            score += (content.rating / 10.0) * 0.3
        
        return score
    
    def _get_user_seed_content(self, user_id: int, limit: int = 10) -> List[int]:
        """Get user's recent interactions as seed content"""
        UserInteraction = self.models.get('UserInteraction')
        if not UserInteraction:
            return []
        
        recent_interactions = UserInteraction.query.filter_by(
            user_id=user_id
        ).order_by(
            UserInteraction.timestamp.desc()
        ).limit(limit).all()
        
        return [i.content_id for i in recent_interactions]
    
    def _calculate_telugu_ratio(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate ratio of Telugu content in recommendations"""
        if not recommendations:
            return 0.0
        
        telugu_count = 0
        
        for rec in recommendations:
            languages = rec['content']['languages']
            if any('telugu' in lang.lower() for lang in languages):
                telugu_count += 1
        
        return telugu_count / len(recommendations)
    
    def _track_recommendation_event(self, user_id: int,
                                   recommendations: List[Dict[str, Any]],
                                   context: Dict[str, Any]):
        """Track recommendation generation event"""
        try:
            event = {
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat(),
                'recommendation_count': len(recommendations),
                'content_ids': [r['content']['id'] for r in recommendations],
                'context': context
            }
            
            # Store in history (limited size)
            self.recommendation_history[user_id].append(event)
            
            # Keep only recent history
            if len(self.recommendation_history[user_id]) > 100:
                self.recommendation_history[user_id] = self.recommendation_history[user_id][-50:]
                
        except Exception as e:
            logger.warning(f"Failed to track recommendation event: {e}")
    
    def _get_fallback_recommendations(self, user_id: int, limit: int) -> Dict[str, Any]:
        """Get fallback recommendations when main algorithm fails"""
        try:
            Content = self.models.get('Content')
            if not Content:
                return self._get_error_response(user_id, "Content model not available")
            
            # Get popular Telugu content
            telugu_content = Content.query.filter(
                Content.languages.contains('Telugu'),
                Content.rating >= 7.0
            ).order_by(
                Content.popularity.desc()
            ).limit(limit // 2).all()
            
            # Get popular general content
            popular_content = Content.query.filter(
                Content.rating >= 7.5
            ).order_by(
                Content.popularity.desc()
            ).limit(limit // 2).all()
            
            # Combine and format
            all_content = list({c.id: c for c in telugu_content + popular_content}.values())
            
            recommendations = []
            for rank, content in enumerate(all_content[:limit]):
                recommendations.append({
                    'rank': rank + 1,
                    'content': {
                        'id': content.id,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': safe_json_loads(content.genres or '[]'),
                        'languages': safe_json_loads(content.languages or '[]'),
                        'rating': content.rating,
                        'poster_path': f"https://image.tmdb.org/t/p/w500{content.poster_path}" if content.poster_path else None
                    },
                    'recommendation_score': 0.5,
                    'reason': "Popular content you might enjoy",
                    'match_percentage': 50,
                    'personalization_type': 'fallback'
                })
            
            return {
                'status': 'fallback',
                'user_id': user_id,
                'recommendations': recommendations,
                'metadata': {
                    'message': 'Using fallback recommendations',
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in fallback recommendations: {e}")
            return self._get_error_response(user_id, str(e))
    
    def _get_error_response(self, user_id: int, error: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'status': 'error',
            'user_id': user_id,
            'recommendations': [],
            'error': error,
            'metadata': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }
    
    def get_similar_content(self, content_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get similar content recommendations"""
        try:
            Content = self.models.get('Content')
            if not Content:
                return []
            
            base_content = Content.query.get(content_id)
            if not base_content:
                return []
            
            # Get candidates (excluding the base content)
            candidates = Content.query.filter(
                Content.id != content_id,
                Content.title.isnot(None)
            ).limit(200).all()
            
            # Calculate similarities
            similarities = []
            
            for candidate in candidates:
                sim_scores = self.similarity_engine.compute_content_similarity(
                    base_content,
                    candidate
                )
                avg_similarity = np.mean(list(sim_scores.values()))
                similarities.append({
                    'content': candidate,
                    'similarity': avg_similarity,
                    'details': sim_scores
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Format results
            results = []
            for item in similarities[:limit]:
                results.append({
                    'content': {
                        'id': item['content'].id,
                        'title': item['content'].title,
                        'content_type': item['content'].content_type,
                        'genres': safe_json_loads(item['content'].genres or '[]'),
                        'languages': safe_json_loads(item['content'].languages or '[]'),
                        'rating': item['content'].rating,
                        'poster_path': f"https://image.tmdb.org/t/p/w500{item['content'].poster_path}" if item['content'].poster_path else None
                    },
                    'similarity_score': round(item['similarity'], 4),
                    'similarity_details': {
                        k: round(v, 3) for k, v in item['details'].items()
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting similar content: {e}")
            return []
    
    def update_user_feedback(self, user_id: int, content_id: int,
                           feedback_type: str, feedback_value: Any = None):
        """Process user feedback to improve recommendations"""
        try:
            # Update user profile in real-time
            interaction_data = {
                'content_id': content_id,
                'interaction_type': feedback_type,
                'rating': feedback_value if feedback_type == 'rating' else None,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Update profile analyzer
            self.profile_analyzer.update_profile_realtime(user_id, interaction_data)
            
            # Update user embedding
            self.embedding_manager.update_user_embedding(user_id, interaction_data)
            
            logger.info(f"Updated user {user_id} feedback for content {content_id}")
            
        except Exception as e:
            logger.error(f"Error processing user feedback: {e}")