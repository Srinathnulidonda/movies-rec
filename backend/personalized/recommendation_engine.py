# backend/personalized/recommendation_engine.py

"""
CineBrain Modern Recommendation Engine
====================================

Advanced hybrid recommendation engine that combines multiple algorithms
to generate personalized feeds similar to TikTok, YouTube, or Instagram.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from surprise import Dataset, Reader, SVD, NMF as SurpriseNMF, KNNBasic
from surprise.model_selection import train_test_split
import networkx as nx
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import random
import math

from .utils import (
    EmbeddingManager,
    SimilarityCalculator, 
    CacheManager,
    PerformanceOptimizer,
    LANGUAGE_PRIORITY,
    PRIORITY_LANGUAGES,
    decay_weight,
    normalize_vector,
    create_cache_key
)

from .profile_analyzer import (
    UserProfileAnalyzer,
    UserPreferenceProfile,
    CinematicDNAAnalyzer
)

logger = logging.getLogger(__name__)

@dataclass
class RecommendationItem:
    """Structured recommendation item with metadata"""
    content_id: int
    title: str
    content_type: str
    genres: List[str]
    languages: List[str]
    rating: float
    release_date: Optional[str]
    poster_path: Optional[str]
    overview: str
    youtube_trailer_id: Optional[str]
    recommendation_score: float
    recommendation_reasons: List[str]
    algorithm_source: str
    confidence_level: str
    personalization_factors: Dict[str, float]

@dataclass
class RecommendationResponse:
    """Complete recommendation response with metadata"""
    user_id: int
    recommendations: List[RecommendationItem]
    total_count: int
    algorithm_breakdown: Dict[str, int]
    personalization_strength: float
    freshness_score: float
    diversity_score: float
    generated_at: datetime
    cache_duration: int
    next_refresh: datetime

class ContentBasedRecommender:
    """
    Advanced content-based recommendation using TF-IDF and feature similarity
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """Initialize content-based recommender"""
        self.embedding_manager = embedding_manager
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        self.content_features_cache = {}
        
        logger.info("CineBrain ContentBasedRecommender initialized")
    
    def generate_recommendations(self, user_profile: UserPreferenceProfile,
                               content_pool: List[Dict], limit: int = 20) -> List[Tuple[Dict, float]]:
        """
        Generate content-based recommendations
        
        Args:
            user_profile: User preference profile
            content_pool: Available content to recommend from
            limit: Maximum number of recommendations
            
        Returns:
            List of (content, score) tuples
        """
        try:
            recommendations = []
            
            # Extract user preference vectors
            user_genre_prefs = user_profile.genre_preferences
            user_lang_prefs = user_profile.language_preferences
            
            for content in content_pool:
                try:
                    # Calculate content-based similarity score
                    score = self._calculate_content_similarity(content, user_profile)
                    
                    if score > 0.3:  # Minimum threshold
                        recommendations.append((content, score))
                        
                except Exception as e:
                    logger.warning(f"Error processing content {content.get('id')}: {e}")
                    continue
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error generating content-based recommendations: {e}")
            return []
    
    def _calculate_content_similarity(self, content: Dict, user_profile: UserPreferenceProfile) -> float:
        """Calculate similarity between content and user preferences"""
        score = 0.0
        
        # Genre similarity (40% weight)
        content_genres = set(content.get('genres', []))
        user_genres = set(user_profile.genre_preferences.keys())
        
        if content_genres and user_genres:
            genre_overlap = len(content_genres & user_genres)
            max_genres = max(len(content_genres), len(user_genres))
            genre_score = genre_overlap / max_genres
            
            # Weight by user preference strength
            weighted_genre_score = 0
            for genre in content_genres:
                if genre in user_profile.genre_preferences:
                    weighted_genre_score += user_profile.genre_preferences[genre]
            
            score += (genre_score * 0.6 + weighted_genre_score * 0.4) * 0.4
        
        # Language similarity with Telugu priority (30% weight)
        content_languages = set(content.get('languages', []))
        user_languages = set(user_profile.language_preferences.keys())
        
        if content_languages and user_languages:
            lang_score = 0
            for lang in content_languages:
                if lang in user_profile.language_preferences:
                    lang_weight = LANGUAGE_PRIORITY.get(lang.lower(), 0.5)
                    lang_score += user_profile.language_preferences[lang] * lang_weight
            
            score += min(lang_score, 1.0) * 0.3
        
        # Quality match (20% weight)
        content_rating = content.get('rating', 0)
        if content_rating >= user_profile.quality_threshold:
            quality_score = min(content_rating / 10.0, 1.0)
            score += quality_score * 0.2
        
        # Content type preference (10% weight)
        content_type = content.get('content_type', '')
        if content_type in user_profile.content_type_preferences:
            type_score = user_profile.content_type_preferences[content_type]
            score += type_score * 0.1
        
        return min(score, 1.0)

class CollaborativeFilteringRecommender:
    """
    Advanced collaborative filtering using matrix factorization and user similarity
    """
    
    def __init__(self, similarity_calculator: SimilarityCalculator):
        """Initialize collaborative filtering recommender"""
        self.similarity_calculator = similarity_calculator
        self.svd_model = SVD(n_factors=50, random_state=42)
        self.nmf_model = SurpriseNMF(n_factors=30, random_state=42)
        self.user_knn = KNNBasic(k=20, sim_options={'name': 'cosine', 'user_based': True})
        self.is_trained = False
        
        logger.info("CineBrain CollaborativeFilteringRecommender initialized")
    
    def train_models(self, interaction_data: List[Dict]):
        """Train collaborative filtering models on interaction data"""
        try:
            if len(interaction_data) < 10:
                logger.warning("Insufficient data for collaborative filtering training")
                return
            
            # Prepare data for Surprise library
            df = pd.DataFrame(interaction_data)
            df = df[df['rating'].notna()]  # Only use rated interactions
            
            if len(df) < 10:
                logger.warning("Insufficient rating data for collaborative filtering")
                return
            
            # Create Surprise dataset
            reader = Reader(rating_scale=(1, 10))
            dataset = Dataset.load_from_df(df[['user_id', 'content_id', 'rating']], reader)
            
            # Train models
            trainset = dataset.build_full_trainset()
            
            self.svd_model.fit(trainset)
            self.nmf_model.fit(trainset)
            self.user_knn.fit(trainset)
            
            self.is_trained = True
            logger.info(f"Trained collaborative filtering models on {len(df)} ratings")
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering models: {e}")
    
    def generate_recommendations(self, user_id: int, user_profile: UserPreferenceProfile,
                               content_pool: List[Dict], limit: int = 20) -> List[Tuple[Dict, float]]:
        """Generate collaborative filtering recommendations"""
        try:
            if not self.is_trained:
                logger.warning("Collaborative filtering models not trained")
                return []
            
            recommendations = []
            
            for content in content_pool:
                try:
                    content_id = content.get('id')
                    
                    # Get predictions from different models
                    svd_pred = self.svd_model.predict(user_id, content_id)
                    nmf_pred = self.nmf_model.predict(user_id, content_id)
                    knn_pred = self.user_knn.predict(user_id, content_id)
                    
                    # Ensemble the predictions
                    ensemble_score = (
                        svd_pred.est * 0.4 +
                        nmf_pred.est * 0.3 +
                        knn_pred.est * 0.3
                    ) / 10.0  # Normalize to 0-1
                    
                    # Apply confidence weighting
                    confidence = min(
                        svd_pred.details.get('was_impossible', False),
                        nmf_pred.details.get('was_impossible', False),
                        knn_pred.details.get('was_impossible', False)
                    )
                    
                    if not confidence and ensemble_score > 0.5:
                        recommendations.append((content, ensemble_score))
                        
                except Exception as e:
                    logger.warning(f"Error predicting for content {content.get('id')}: {e}")
                    continue
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error generating collaborative filtering recommendations: {e}")
            return []

class GraphBasedRecommender:
    """
    Graph-based recommendation using content and user relationship networks
    """
    
    def __init__(self):
        """Initialize graph-based recommender"""
        self.content_graph = nx.Graph()
        self.user_graph = nx.Graph()
        self.bipartite_graph = nx.Graph()
        
        logger.info("CineBrain GraphBasedRecommender initialized")
    
    def build_content_graph(self, content_data: List[Dict], interaction_data: List[Dict]):
        """Build content similarity graph"""
        try:
            # Add content nodes
            for content in content_data:
                self.content_graph.add_node(
                    content['id'],
                    **{k: v for k, v in content.items() if k != 'id'}
                )
            
            # Add edges based on co-interactions
            user_content_map = defaultdict(set)
            for interaction in interaction_data:
                user_content_map[interaction['user_id']].add(interaction['content_id'])
            
            # Calculate content co-occurrence
            content_cooccurrence = defaultdict(int)
            for user_id, content_ids in user_content_map.items():
                content_list = list(content_ids)
                for i in range(len(content_list)):
                    for j in range(i + 1, len(content_list)):
                        content1, content2 = content_list[i], content_list[j]
                        content_cooccurrence[(content1, content2)] += 1
            
            # Add edges with weights
            for (content1, content2), weight in content_cooccurrence.items():
                if weight >= 2:  # Minimum co-occurrence threshold
                    self.content_graph.add_edge(content1, content2, weight=weight)
            
            logger.info(f"Built content graph with {self.content_graph.number_of_nodes()} nodes and {self.content_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error building content graph: {e}")
    
    def generate_recommendations(self, user_id: int, user_interactions: List[int],
                               limit: int = 20) -> List[Tuple[int, float]]:
        """Generate graph-based recommendations using random walks"""
        try:
            if not self.content_graph.nodes():
                return []
            
            # Get user's interacted content that exists in graph
            seed_nodes = [cid for cid in user_interactions if cid in self.content_graph.nodes()]
            
            if not seed_nodes:
                return []
            
            # Perform personalized PageRank
            personalization = {node: 1.0 if node in seed_nodes else 0.0 
                             for node in self.content_graph.nodes()}
            
            pagerank_scores = nx.pagerank(
                self.content_graph,
                personalization=personalization,
                alpha=0.85,
                max_iter=100
            )
            
            # Filter out already interacted content and sort
            recommendations = [
                (content_id, score) for content_id, score in pagerank_scores.items()
                if content_id not in user_interactions
            ]
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error generating graph-based recommendations: {e}")
            return []

class ClusteringBasedRecommender:
    """
    Clustering-based recommendation using user and content clusters
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """Initialize clustering-based recommender"""
        self.embedding_manager = embedding_manager
        self.user_clusters = None
        self.content_clusters = None
        self.n_user_clusters = 10
        self.n_content_clusters = 20
        
        logger.info("CineBrain ClusteringBasedRecommender initialized")
    
    def train_clusters(self, user_embeddings: Dict[int, np.ndarray], 
                      content_embeddings: Dict[int, np.ndarray]):
        """Train user and content clusters"""
        try:
            # Train user clusters
            if len(user_embeddings) >= self.n_user_clusters:
                user_ids = list(user_embeddings.keys())
                user_vectors = np.array([user_embeddings[uid] for uid in user_ids])
                
                kmeans_users = KMeans(n_clusters=self.n_user_clusters, random_state=42)
                user_cluster_labels = kmeans_users.fit_predict(user_vectors)
                
                self.user_clusters = {
                    user_ids[i]: label for i, label in enumerate(user_cluster_labels)
                }
            
            # Train content clusters
            if len(content_embeddings) >= self.n_content_clusters:
                content_ids = list(content_embeddings.keys())
                content_vectors = np.array([content_embeddings[cid] for cid in content_ids])
                
                kmeans_content = KMeans(n_clusters=self.n_content_clusters, random_state=42)
                content_cluster_labels = kmeans_content.fit_predict(content_vectors)
                
                self.content_clusters = {
                    content_ids[i]: label for i, label in enumerate(content_cluster_labels)
                }
            
            logger.info(f"Trained clusters: {len(self.user_clusters or {})} user clusters, {len(self.content_clusters or {})} content clusters")
            
        except Exception as e:
            logger.error(f"Error training clusters: {e}")
    
    def generate_recommendations(self, user_id: int, content_pool: List[Dict],
                               limit: int = 20) -> List[Tuple[Dict, float]]:
        """Generate cluster-based recommendations"""
        try:
            if not self.user_clusters or not self.content_clusters:
                return []
            
            if user_id not in self.user_clusters:
                return []
            
            user_cluster = self.user_clusters[user_id]
            
            # Find similar users in same cluster
            similar_users = [uid for uid, cluster in self.user_clusters.items() 
                           if cluster == user_cluster and uid != user_id]
            
            # Get content preferences from similar users' clusters
            recommendations = []
            
            for content in content_pool:
                content_id = content.get('id')
                if content_id in self.content_clusters:
                    content_cluster = self.content_clusters[content_id]
                    
                    # Calculate cluster-based score
                    cluster_score = self._calculate_cluster_affinity(
                        user_cluster, content_cluster, similar_users
                    )
                    
                    if cluster_score > 0.3:
                        recommendations.append((content, cluster_score))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error generating cluster-based recommendations: {e}")
            return []
    
    def _calculate_cluster_affinity(self, user_cluster: int, content_cluster: int,
                                  similar_users: List[int]) -> float:
        """Calculate affinity between user cluster and content cluster"""
        # This is a simplified version - in practice, you'd analyze
        # interaction patterns between user clusters and content clusters
        base_score = 0.5
        
        # Boost if similar users have interacted with this content cluster
        if similar_users:
            # This would require tracking cluster interaction patterns
            base_score += 0.2
        
        return min(base_score, 1.0)

class HybridRecommendationEngine:
    """
    Hybrid recommendation engine that combines multiple algorithms
    """
    
    def __init__(self, embedding_manager: EmbeddingManager, 
                 similarity_calculator: SimilarityCalculator):
        """Initialize hybrid recommendation engine"""
        self.embedding_manager = embedding_manager
        self.similarity_calculator = similarity_calculator
        
        # Initialize individual recommenders
        self.content_based = ContentBasedRecommender(embedding_manager)
        self.collaborative = CollaborativeFilteringRecommender(similarity_calculator)
        self.graph_based = GraphBasedRecommender()
        self.clustering_based = ClusteringBasedRecommender(embedding_manager)
        
        # Algorithm weights
        self.algorithm_weights = {
            'content_based': 0.3,
            'collaborative': 0.25,
            'graph_based': 0.25,
            'clustering': 0.2
        }
        
        # Performance tracking
        self.performance_optimizer = PerformanceOptimizer()
        
        logger.info("CineBrain HybridRecommendationEngine initialized")
    
    @PerformanceOptimizer.time_function('generate_hybrid_recommendations')
    def generate_recommendations(self, user_id: int, user_profile: UserPreferenceProfile,
                               content_pool: List[Dict], interaction_data: List[Dict],
                               limit: int = 20) -> List[RecommendationItem]:
        """
        Generate hybrid recommendations by combining multiple algorithms
        
        Args:
            user_id: User identifier
            user_profile: User preference profile
            content_pool: Available content to recommend
            interaction_data: Historical interaction data
            limit: Maximum number of recommendations
            
        Returns:
            List of RecommendationItem objects
        """
        try:
            all_recommendations = defaultdict(list)
            algorithm_scores = defaultdict(dict)
            
            # Generate recommendations from each algorithm
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                
                # Content-based recommendations
                futures['content_based'] = executor.submit(
                    self.content_based.generate_recommendations,
                    user_profile, content_pool, limit * 2
                )
                
                # Collaborative filtering recommendations
                futures['collaborative'] = executor.submit(
                    self.collaborative.generate_recommendations,
                    user_id, user_profile, content_pool, limit * 2
                )
                
                # Graph-based recommendations
                user_interactions = [i['content_id'] for i in interaction_data if i['user_id'] == user_id]
                futures['graph_based'] = executor.submit(
                    self.graph_based.generate_recommendations,
                    user_id, user_interactions, limit * 2
                )
                
                # Clustering-based recommendations
                futures['clustering'] = executor.submit(
                    self.clustering_based.generate_recommendations,
                    user_id, content_pool, limit * 2
                )
                
                # Collect results
                for algorithm, future in futures.items():
                    try:
                        recommendations = future.result(timeout=10)
                        all_recommendations[algorithm] = recommendations
                        
                        # Store individual algorithm scores
                        for content, score in recommendations:
                            content_id = content.get('id') if isinstance(content, dict) else content
                            algorithm_scores[content_id][algorithm] = score
                            
                    except Exception as e:
                        logger.warning(f"Error getting {algorithm} recommendations: {e}")
                        all_recommendations[algorithm] = []
            
            # Combine recommendations using hybrid scoring
            final_recommendations = self._combine_recommendations(
                all_recommendations, algorithm_scores, content_pool, user_profile
            )
            
            # Apply diversity and freshness filters
            final_recommendations = self._apply_diversity_filter(final_recommendations)
            final_recommendations = self._apply_freshness_boost(final_recommendations)
            
            # Convert to RecommendationItem objects
            recommendation_items = self._convert_to_recommendation_items(
                final_recommendations[:limit], algorithm_scores, user_profile
            )
            
            logger.info(f"Generated {len(recommendation_items)} hybrid recommendations for user {user_id}")
            return recommendation_items
            
        except Exception as e:
            logger.error(f"Error generating hybrid recommendations: {e}")
            return []
    
    def _combine_recommendations(self, all_recommendations: Dict[str, List],
                               algorithm_scores: Dict[int, Dict[str, float]],
                               content_pool: List[Dict],
                               user_profile: UserPreferenceProfile) -> List[Tuple[Dict, float]]:
        """Combine recommendations from different algorithms using weighted scoring"""
        try:
            content_map = {c['id']: c for c in content_pool}
            combined_scores = defaultdict(float)
            content_algorithms = defaultdict(set)
            
            # Calculate weighted hybrid scores
            for content_id, algo_scores in algorithm_scores.items():
                total_score = 0
                total_weight = 0
                
                for algorithm, score in algo_scores.items():
                    weight = self.algorithm_weights.get(algorithm, 0.25)
                    total_score += score * weight
                    total_weight += weight
                    content_algorithms[content_id].add(algorithm)
                
                if total_weight > 0:
                    # Normalize by actual weight used
                    normalized_score = total_score / total_weight
                    
                    # Boost score if multiple algorithms agree
                    algorithm_agreement_boost = len(content_algorithms[content_id]) / len(self.algorithm_weights)
                    final_score = normalized_score * (1 + algorithm_agreement_boost * 0.2)
                    
                    combined_scores[content_id] = min(final_score, 1.0)
            
            # Convert to list of (content, score) tuples
            recommendations = []
            for content_id, score in combined_scores.items():
                if content_id in content_map:
                    recommendations.append((content_map[content_id], score))
            
            # Sort by combined score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations
            
        except Exception as e:
            logger.error(f"Error combining recommendations: {e}")
            return []
    
    def _apply_diversity_filter(self, recommendations: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """Apply diversity filter to avoid over-concentration in genres/languages"""
        try:
            if len(recommendations) <= 10:
                return recommendations
            
            diverse_recommendations = []
            seen_genres = set()
            seen_languages = set()
            genre_counts = defaultdict(int)
            language_counts = defaultdict(int)
            
            for content, score in recommendations:
                content_genres = set(content.get('genres', []))
                content_languages = set(content.get('languages', []))
                
                # Check if adding this content increases diversity
                new_genres = len(content_genres - seen_genres)
                new_languages = len(content_languages - seen_languages)
                
                # Calculate diversity bonus
                diversity_bonus = (new_genres * 0.1) + (new_languages * 0.05)
                
                # Check for over-representation
                max_genre_count = max(genre_counts.values()) if genre_counts else 0
                max_lang_count = max(language_counts.values()) if language_counts else 0
                
                # Penalize if genre/language is over-represented
                over_representation_penalty = 0
                if max_genre_count > 3:  # More than 3 items from same genre
                    common_genres = content_genres & seen_genres
                    if common_genres:
                        over_representation_penalty += 0.1
                
                if max_lang_count > 5:  # More than 5 items in same language
                    common_languages = content_languages & seen_languages
                    if common_languages:
                        over_representation_penalty += 0.05
                
                # Apply diversity adjustments
                adjusted_score = score + diversity_bonus - over_representation_penalty
                diverse_recommendations.append((content, max(adjusted_score, 0)))
                
                # Update tracking
                seen_genres.update(content_genres)
                seen_languages.update(content_languages)
                for genre in content_genres:
                    genre_counts[genre] += 1
                for language in content_languages:
                    language_counts[language] += 1
            
            # Re-sort by adjusted scores
            diverse_recommendations.sort(key=lambda x: x[1], reverse=True)
            return diverse_recommendations
            
        except Exception as e:
            logger.error(f"Error applying diversity filter: {e}")
            return recommendations
    
    def _apply_freshness_boost(self, recommendations: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """Apply freshness boost to recent content"""
        try:
            current_year = datetime.now().year
            boosted_recommendations = []
            
            for content, score in recommendations:
                release_date = content.get('release_date')
                freshness_boost = 0
                
                if release_date:
                    try:
                        if isinstance(release_date, str):
                            release_year = datetime.fromisoformat(release_date).year
                        else:
                            release_year = release_date.year
                        
                        # Boost recent content
                        years_old = current_year - release_year
                        if years_old <= 1:
                            freshness_boost = 0.15  # 15% boost for very recent
                        elif years_old <= 3:
                            freshness_boost = 0.1   # 10% boost for recent
                        elif years_old <= 5:
                            freshness_boost = 0.05  # 5% boost for somewhat recent
                        
                    except Exception:
                        pass
                
                boosted_score = min(score + freshness_boost, 1.0)
                boosted_recommendations.append((content, boosted_score))
            
            # Re-sort by boosted scores
            boosted_recommendations.sort(key=lambda x: x[1], reverse=True)
            return boosted_recommendations
            
        except Exception as e:
            logger.error(f"Error applying freshness boost: {e}")
            return recommendations
    
    def _convert_to_recommendation_items(self, recommendations: List[Tuple[Dict, float]],
                                       algorithm_scores: Dict[int, Dict[str, float]],
                                       user_profile: UserPreferenceProfile) -> List[RecommendationItem]:
        """Convert recommendations to structured RecommendationItem objects"""
        recommendation_items = []
        
        for i, (content, score) in enumerate(recommendations):
            try:
                content_id = content['id']
                
                # Generate recommendation reasons
                reasons = self._generate_recommendation_reasons(
                    content, user_profile, algorithm_scores.get(content_id, {})
                )
                
                # Determine algorithm source and confidence
                content_algo_scores = algorithm_scores.get(content_id, {})
                primary_algorithm = max(content_algo_scores.items(), key=lambda x: x[1])[0] if content_algo_scores else 'hybrid'
                
                confidence_level = self._determine_confidence_level(score, len(content_algo_scores))
                
                # Calculate personalization factors
                personalization_factors = self._calculate_personalization_factors(content, user_profile)
                
                recommendation_item = RecommendationItem(
                    content_id=content_id,
                    title=content.get('title', ''),
                    content_type=content.get('content_type', ''),
                    genres=content.get('genres', []),
                    languages=content.get('languages', []),
                    rating=content.get('rating', 0),
                    release_date=content.get('release_date'),
                    poster_path=self._format_poster_path(content.get('poster_path')),
                    overview=content.get('overview', '')[:200] + '...' if content.get('overview') else '',
                    youtube_trailer_id=content.get('youtube_trailer_id'),
                    recommendation_score=round(score, 4),
                    recommendation_reasons=reasons,
                    algorithm_source=primary_algorithm,
                    confidence_level=confidence_level,
                    personalization_factors=personalization_factors
                )
                
                recommendation_items.append(recommendation_item)
                
            except Exception as e:
                logger.warning(f"Error converting recommendation item: {e}")
                continue
        
        return recommendation_items
    
    def _generate_recommendation_reasons(self, content: Dict, user_profile: UserPreferenceProfile,
                                       algo_scores: Dict[str, float]) -> List[str]:
        """Generate human-readable recommendation reasons"""
        reasons = []
        
        # Genre-based reasons
        content_genres = set(content.get('genres', []))
        user_top_genres = set(list(user_profile.genre_preferences.keys())[:3])
        
        matching_genres = content_genres & user_top_genres
        if matching_genres:
            reasons.append(f"Matches your interest in {', '.join(list(matching_genres)[:2])}")
        
        # Language-based reasons
        content_languages = set(content.get('languages', []))
        user_top_languages = set(list(user_profile.language_preferences.keys())[:2])
        
        matching_languages = content_languages & user_top_languages
        if matching_languages:
            if 'Telugu' in matching_languages or 'telugu' in [l.lower() for l in matching_languages]:
                reasons.append("Perfect for Telugu cinema lovers")
            else:
                reasons.append(f"In your preferred language: {', '.join(matching_languages)}")
        
        # Quality-based reasons
        content_rating = content.get('rating', 0)
        if content_rating >= user_profile.quality_threshold and content_rating >= 8.0:
            reasons.append("Highly rated and acclaimed")
        
        # Algorithm-based reasons
        if algo_scores:
            dominant_algo = max(algo_scores.items(), key=lambda x: x[1])[0]
            if dominant_algo == 'collaborative' and algo_scores[dominant_algo] > 0.7:
                reasons.append("Loved by users with similar taste")
            elif dominant_algo == 'content_based' and algo_scores[dominant_algo] > 0.7:
                reasons.append("Similar to content you've enjoyed")
        
        # Freshness reasons
        if content.get('release_date'):
            try:
                release_date = datetime.fromisoformat(content['release_date']) if isinstance(content['release_date'], str) else content['release_date']
                days_old = (datetime.now() - release_date).days
                if days_old <= 30:
                    reasons.append("Fresh and trending")
                elif days_old <= 90:
                    reasons.append("Recent release")
            except:
                pass
        
        return reasons[:3]  # Limit to top 3 reasons
    
    def _determine_confidence_level(self, score: float, num_algorithms: int) -> str:
        """Determine confidence level based on score and algorithm agreement"""
        if score >= 0.8 and num_algorithms >= 3:
            return 'very_high'
        elif score >= 0.7 and num_algorithms >= 2:
            return 'high'
        elif score >= 0.6:
            return 'medium'
        elif score >= 0.4:
            return 'moderate'
        else:
            return 'low'
    
    def _calculate_personalization_factors(self, content: Dict, user_profile: UserPreferenceProfile) -> Dict[str, float]:
        """Calculate specific personalization factors"""
        factors = {}
        
        # Genre match factor
        content_genres = set(content.get('genres', []))
        user_genres = set(user_profile.genre_preferences.keys())
        if content_genres and user_genres:
            factors['genre_match'] = len(content_genres & user_genres) / len(content_genres | user_genres)
        
        # Language match factor
        content_languages = set(content.get('languages', []))
        user_languages = set(user_profile.language_preferences.keys())
        if content_languages and user_languages:
            factors['language_match'] = len(content_languages & user_languages) / len(content_languages | user_languages)
        
        # Quality alignment factor
        content_rating = content.get('rating', 0)
        if content_rating > 0:
            factors['quality_alignment'] = min(content_rating / user_profile.quality_threshold, 1.0)
        
        # Sophistication match factor
        factors['sophistication_match'] = user_profile.sophistication_score
        
        return factors
    
    def _format_poster_path(self, poster_path: Optional[str]) -> Optional[str]:
        """Format poster path for API response"""
        if not poster_path:
            return None
        
        if poster_path.startswith('http'):
            return poster_path
        else:
            return f"https://image.tmdb.org/t/p/w300{poster_path}"

class ModernPersonalizationEngine:
    """
    Main personalization engine that orchestrates the entire recommendation process
    """
    
    def __init__(self, db, models: Dict, profile_analyzer: UserProfileAnalyzer,
                 similarity_calculator: SimilarityCalculator, cache_manager: CacheManager):
        """Initialize modern personalization engine"""
        self.db = db
        self.models = models
        self.profile_analyzer = profile_analyzer
        self.similarity_calculator = similarity_calculator
        self.cache_manager = cache_manager
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(cache=cache_manager.cache)
        
        # Initialize hybrid recommendation engine
        self.hybrid_engine = HybridRecommendationEngine(
            self.embedding_manager, 
            similarity_calculator
        )
        
        # Performance optimization
        self.performance_optimizer = PerformanceOptimizer()
        
        logger.info("CineBrain ModernPersonalizationEngine initialized")
    
    @PerformanceOptimizer.time_function('generate_personalized_feed')
    def generate_personalized_feed(self, user_id: int, limit: int = 50,
                                 categories: Optional[List[str]] = None) -> RecommendationResponse:
        """
        Generate personalized content feed like modern social media platforms
        
        Args:
            user_id: User identifier
            limit: Maximum number of recommendations
            categories: Specific recommendation categories
            
        Returns:
            RecommendationResponse with personalized recommendations
        """
        try:
            # Check cache first
            cached_recommendations = self.cache_manager.get_user_recommendations(
                user_id, 'personalized_feed'
            )
            
            if cached_recommendations:
                logger.info(f"Returning cached recommendations for user {user_id}")
                return self._convert_cached_to_response(cached_recommendations, user_id)
            
            # Build user profile
            user_profile = self.profile_analyzer.build_comprehensive_user_profile(user_id)
            if not user_profile:
                return self._generate_cold_start_recommendations(user_id, limit)
            
            # Get content pool
            content_pool = self._get_content_pool(user_profile)
            
            # Get interaction data for collaborative filtering
            interaction_data = self._get_interaction_data()
            
            # Train models if needed
            self._ensure_models_trained(interaction_data)
            
            # Generate recommendations
            recommendations = self.hybrid_engine.generate_recommendations(
                user_id, user_profile, content_pool, interaction_data, limit
            )
            
            # Calculate response metadata
            algorithm_breakdown = self._calculate_algorithm_breakdown(recommendations)
            personalization_strength = self._calculate_personalization_strength(user_profile)
            freshness_score = self._calculate_freshness_score(recommendations)
            diversity_score = self._calculate_diversity_score(recommendations)
            
            # Create response
            response = RecommendationResponse(
                user_id=user_id,
                recommendations=recommendations,
                total_count=len(recommendations),
                algorithm_breakdown=algorithm_breakdown,
                personalization_strength=personalization_strength,
                freshness_score=freshness_score,
                diversity_score=diversity_score,
                generated_at=datetime.utcnow(),
                cache_duration=3600,  # 1 hour
                next_refresh=datetime.utcnow() + timedelta(hours=1)
            )
            
            # Cache the recommendations
            self._cache_recommendations(user_id, response)
            
            logger.info(f"Generated {len(recommendations)} personalized recommendations for user {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating personalized feed for user {user_id}: {e}")
            return self._generate_fallback_recommendations(user_id, limit)
    
    def _get_content_pool(self, user_profile: UserPreferenceProfile) -> List[Dict]:
        """Get relevant content pool for recommendations"""
        try:
            Content = self.models['Content']
            
            # Build query with user preferences
            query = self.db.session.query(Content)
            
            # Filter by quality threshold
            query = query.filter(Content.rating >= max(user_profile.quality_threshold - 1, 5.0))
            
            # Prefer Telugu content
            preferred_languages = list(user_profile.language_preferences.keys())
            if 'Telugu' in preferred_languages or 'telugu' in [l.lower() for l in preferred_languages]:
                # Boost Telugu content in the pool
                telugu_content = query.filter(Content.languages.contains('Telugu')).limit(300).all()
                other_content = query.filter(~Content.languages.contains('Telugu')).limit(200).all()
                all_content = telugu_content + other_content
            else:
                all_content = query.limit(500).all()
            
            # Convert to dictionary format
            content_pool = []
            for content in all_content:
                content_dict = {
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating or 0,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'poster_path': content.poster_path,
                    'overview': content.overview or '',
                    'youtube_trailer_id': content.youtube_trailer_id,
                    'popularity': content.popularity or 0
                }
                content_pool.append(content_dict)
            
            logger.info(f"Retrieved content pool of {len(content_pool)} items")
            return content_pool
            
        except Exception as e:
            logger.error(f"Error getting content pool: {e}")
            return []
    
    def _ensure_models_trained(self, interaction_data: List[Dict]):
        """Ensure recommendation models are trained"""
        try:
            # Train collaborative filtering models
            self.hybrid_engine.collaborative.train_models(interaction_data)
            
            # Train clustering models
            user_embeddings = {}
            content_embeddings = {}
            
            # This would typically be done periodically, not on every request
            # For demo purposes, we'll skip the heavy computation
            
        except Exception as e:
            logger.warning(f"Error training models: {e}")
    
    def _cache_recommendations(self, user_id: int, response: RecommendationResponse):
        """Cache recommendations for future requests"""
        try:
            cache_data = {
                'recommendations': [
                    {
                        'content_id': rec.content_id,
                        'title': rec.title,
                        'content_type': rec.content_type,
                        'genres': rec.genres,
                        'languages': rec.languages,
                        'rating': rec.rating,
                        'release_date': rec.release_date,
                        'poster_path': rec.poster_path,
                        'overview': rec.overview,
                        'youtube_trailer_id': rec.youtube_trailer_id,
                        'recommendation_score': rec.recommendation_score,
                        'recommendation_reasons': rec.recommendation_reasons,
                        'algorithm_source': rec.algorithm_source,
                        'confidence_level': rec.confidence_level,
                        'personalization_factors': rec.personalization_factors
                    }
                    for rec in response.recommendations
                ],
                'metadata': {
                    'algorithm_breakdown': response.algorithm_breakdown,
                    'personalization_strength': response.personalization_strength,
                    'freshness_score': response.freshness_score,
                    'diversity_score': response.diversity_score,
                    'generated_at': response.generated_at.isoformat()
                }
            }
            
            self.cache_manager.set_user_recommendations(
                user_id, cache_data, 'personalized_feed', timeout=3600
            )
            
        except Exception as e:
            logger.warning(f"Error caching recommendations: {e}")

class RecommendationOrchestrator:
    """
    High-level orchestrator for all recommendation operations
    """
    
    def __init__(self, personalization_engine: ModernPersonalizationEngine):
        """Initialize recommendation orchestrator"""
        self.personalization_engine = personalization_engine
        self.performance_optimizer = PerformanceOptimizer()
        
        logger.info("CineBrain RecommendationOrchestrator initialized")
    
    async def get_recommendations_async(self, user_id: int, request_params: Dict) -> Dict[str, Any]:
        """Get recommendations asynchronously for better performance"""
        try:
            loop = asyncio.get_event_loop()
            
            # Run recommendation generation in thread pool
            recommendations = await loop.run_in_executor(
                None,
                self.personalization_engine.generate_personalized_feed,
                user_id,
                request_params.get('limit', 50)
            )
            
            return self._format_api_response(recommendations)
            
        except Exception as e:
            logger.error(f"Error getting async recommendations: {e}")
            return {'error': 'Failed to generate recommendations'}
    
    def _format_api_response(self, response: RecommendationResponse) -> Dict[str, Any]:
        """Format recommendation response for API"""
        return {
            'success': True,
            'user_id': response.user_id,
            'recommendations': [
                {
                    'id': rec.content_id,
                    'title': rec.title,
                    'content_type': rec.content_type,
                    'genres': rec.genres,
                    'languages': rec.languages,
                    'rating': rec.rating,
                    'release_date': rec.release_date,
                    'poster_path': rec.poster_path,
                    'overview': rec.overview,
                    'youtube_trailer_id': rec.youtube_trailer_id,
                    'score': rec.recommendation_score,
                    'reasons': rec.recommendation_reasons,
                    'source': rec.algorithm_source,
                    'confidence': rec.confidence_level,
                    'personalization': rec.personalization_factors
                }
                for rec in response.recommendations
            ],
            'metadata': {
                'total_count': response.total_count,
                'algorithm_breakdown': response.algorithm_breakdown,
                'personalization_strength': round(response.personalization_strength, 3),
                'freshness_score': round(response.freshness_score, 3),
                'diversity_score': round(response.diversity_score, 3),
                'generated_at': response.generated_at.isoformat(),
                'next_refresh': response.next_refresh.isoformat()
            }
        }
