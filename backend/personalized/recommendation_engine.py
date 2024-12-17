# backend/personalized/recommendation_engine.py

"""
CineBrain Advanced Recommendation Engine
Production-grade AI-powered recommendation system with neural collaborative filtering,
cinematic DNA matching, and real-time adaptive personalization
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
import json
import logging
import math
import random
import time
import threading
from typing import List, Dict, Any, Tuple, Optional, Set
from contextlib import contextmanager
from sqlalchemy import func, desc, and_, or_
from sqlalchemy.exc import OperationalError, DisconnectionError
import traceback

# Import from existing algorithms
from services.algorithms import (
    ContentBasedFiltering,
    CollaborativeFiltering, 
    HybridRecommendationEngine,
    PopularityRanking,
    LanguagePriorityFilter,
    UltraPowerfulSimilarityEngine,
    AdvancedAlgorithms,
    PRIORITY_LANGUAGES,
    LANGUAGE_WEIGHTS
)

from .profile_analyzer import AdvancedProfileAnalyzer

logger = logging.getLogger(__name__)

# Advanced recommendation configuration
RECOMMENDATION_CONFIG = {
    'telugu_priority_boost': 1.3,
    'cultural_authenticity_weight': 0.25,
    'quality_threshold_weight': 0.2,
    'discovery_exploration_rate': 0.15,
    'real_time_adaptation_rate': 0.1,
    'neural_embedding_weight': 0.3,
    'collaborative_weight': 0.25,
    'content_based_weight': 0.2,
    'popularity_weight': 0.15,
    'serendipity_injection_rate': 0.1
}

class NeuralCollaborativeFiltering:
    """
    Advanced neural collaborative filtering with embedding-based similarity
    """
    
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.interaction_matrix = None
        self.svd_model = TruncatedSVD(n_components=embedding_dim)
        
    def train_embeddings(self, interactions: List[Any], content_list: List[Any]):
        """Train user and item embeddings from interactions"""
        try:
            # Build interaction matrix
            user_ids = list(set([i.user_id for i in interactions]))
            content_ids = list(set([i.content_id for i in interactions]))
            
            user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
            content_to_idx = {content_id: idx for idx, content_id in enumerate(content_ids)}
            
            # Create sparse interaction matrix
            rows, cols, data = [], [], []
            for interaction in interactions:
                if interaction.user_id in user_to_idx and interaction.content_id in content_to_idx:
                    rows.append(user_to_idx[interaction.user_id])
                    cols.append(content_to_idx[interaction.content_id])
                    
                    # Weight interactions by type and rating
                    weight = self._calculate_interaction_weight(interaction)
                    data.append(weight)
            
            self.interaction_matrix = csr_matrix(
                (data, (rows, cols)), 
                shape=(len(user_ids), len(content_ids))
            )
            
            # Train SVD embeddings
            if self.interaction_matrix.nnz > 0:
                user_factors = self.svd_model.fit_transform(self.interaction_matrix)
                item_factors = self.svd_model.transform(self.interaction_matrix.T)
                
                # Store embeddings
                for idx, user_id in enumerate(user_ids):
                    self.user_embeddings[user_id] = user_factors[idx]
                
                for idx, content_id in enumerate(content_ids):
                    self.item_embeddings[content_id] = item_factors[idx]
            
            logger.info(f"✅ Trained neural embeddings for {len(user_ids)} users and {len(content_ids)} items")
            
        except Exception as e:
            logger.error(f"Error training neural embeddings: {e}")
    
    def _calculate_interaction_weight(self, interaction: Any) -> float:
        """Calculate weight for interaction based on type and rating"""
        base_weights = {
            'favorite': 5.0,
            'watchlist': 3.0,
            'rating': 2.0,
            'view': 1.5,
            'like': 1.0,
            'search': 0.5
        }
        
        weight = base_weights.get(interaction.interaction_type, 1.0)
        
        # Boost by rating if available
        if hasattr(interaction, 'rating') and interaction.rating:
            weight *= (interaction.rating / 10.0) * 2
        
        # Recency boost
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        recency_factor = math.exp(-days_ago / 30)  # Exponential decay over 30 days
        weight *= (0.5 + 0.5 * recency_factor)
        
        return weight
    
    def get_user_recommendations(self, user_id: int, content_pool: List[Any], 
                               limit: int = 50) -> List[Tuple[Any, float]]:
        """Generate recommendations using neural collaborative filtering"""
        try:
            if user_id not in self.user_embeddings:
                return []
            
            user_embedding = self.user_embeddings[user_id]
            recommendations = []
            
            for content in content_pool:
                if content.id in self.item_embeddings:
                    item_embedding = self.item_embeddings[content.id]
                    similarity = cosine_similarity([user_embedding], [item_embedding])[0][0]
                    recommendations.append((content, similarity))
            
            # Sort by similarity and return top recommendations
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error getting neural collaborative recommendations: {e}")
            return []

class CulturalAwarenessEngine:
    """
    Advanced cultural awareness and regional content prioritization
    """
    
    def __init__(self):
        self.cultural_patterns = {
            'telugu_cinema_markers': {
                'tollywood_directors': ['rajamouli', 'trivikram', 'koratala', 'sukumar', 'vamshi'],
                'telugu_actors': ['mahesh', 'allu', 'ram', 'ntr', 'prabhas', 'vijay'],
                'regional_themes': ['village', 'family', 'tradition', 'honor', 'revenge'],
                'cultural_elements': ['festival', 'ritual', 'custom', 'heritage', 'telugu']
            },
            'pan_indian_elements': {
                'universal_themes': ['love', 'friendship', 'success', 'journey', 'dreams'],
                'cross_cultural_appeal': ['action', 'romance', 'comedy', 'thriller', 'drama']
            }
        }
        
        self.regional_weights = {
            'telugu': 1.0,
            'hindi': 0.85,
            'tamil': 0.8,
            'malayalam': 0.75,
            'kannada': 0.7,
            'english': 0.9,
            'international': 0.6
        }
    
    def calculate_cultural_relevance(self, content: Any, user_cultural_profile: Dict[str, Any]) -> float:
        """Calculate cultural relevance score for content"""
        try:
            relevance_score = 0.0
            
            # Primary language match
            content_languages = self._extract_content_languages(content)
            user_primary_culture = user_cultural_profile.get('primary_culture', 'telugu')
            
            if user_primary_culture in content_languages:
                relevance_score += 0.4
            
            # Regional weight application
            for lang in content_languages:
                lang_weight = self.regional_weights.get(lang, 0.5)
                relevance_score += lang_weight * 0.2
            
            # Cultural authenticity bonus
            if self._has_cultural_authenticity(content, user_primary_culture):
                relevance_score += 0.25
            
            # Cross-cultural bridge score
            bridge_score = user_cultural_profile.get('bridge_score', 0.5)
            if bridge_score > 0.7:  # User is culturally adventurous
                relevance_score += 0.15
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating cultural relevance: {e}")
            return 0.5
    
    def _extract_content_languages(self, content: Any) -> List[str]:
        """Extract and normalize content languages"""
        if not content.languages:
            return []
        
        try:
            languages = json.loads(content.languages)
            return [lang.lower().strip() for lang in languages]
        except:
            return []
    
    def _has_cultural_authenticity(self, content: Any, culture: str) -> bool:
        """Check if content has cultural authenticity markers"""
        if culture != 'telugu':
            return False
        
        content_text = f"{content.title} {content.overview or ''}".lower()
        
        # Check for Telugu cinema markers
        markers = self.cultural_patterns['telugu_cinema_markers']
        
        for category, keywords in markers.items():
            if any(keyword in content_text for keyword in keywords):
                return True
        
        return False

class AdaptivePersonalizationEngine:
    """
    Real-time adaptive personalization with learning capabilities
    """
    
    def __init__(self, cache=None):
        self.cache = cache
        self.user_adaptation_history = defaultdict(list)
        self.recommendation_feedback = defaultdict(list)
        self.learning_rate = 0.1
        
        # Adaptation strategies
        self.adaptation_strategies = {
            'preference_drift': self._handle_preference_drift,
            'quality_evolution': self._handle_quality_evolution,
            'discovery_expansion': self._handle_discovery_expansion,
            'cultural_exploration': self._handle_cultural_exploration
        }
    
    def adapt_recommendations(self, user_id: int, base_recommendations: List[Tuple[Any, float]], 
                            user_profile: Dict[str, Any], 
                            recent_feedback: List[Dict[str, Any]]) -> List[Tuple[Any, float]]:
        """Adapt recommendations based on recent user behavior and feedback"""
        try:
            if not recent_feedback:
                return base_recommendations
            
            # Analyze recent feedback patterns
            feedback_analysis = self._analyze_feedback_patterns(recent_feedback)
            
            # Apply adaptation strategies
            adapted_recommendations = base_recommendations.copy()
            
            for strategy_name, adaptation_func in self.adaptation_strategies.items():
                if self._should_apply_strategy(strategy_name, feedback_analysis):
                    adapted_recommendations = adaptation_func(
                        adapted_recommendations, user_profile, feedback_analysis
                    )
            
            # Apply real-time learning
            adapted_recommendations = self._apply_real_time_learning(
                user_id, adapted_recommendations, feedback_analysis
            )
            
            # Record adaptation for future learning
            self._record_adaptation(user_id, feedback_analysis, adapted_recommendations)
            
            return adapted_recommendations
            
        except Exception as e:
            logger.error(f"Error in adaptive personalization: {e}")
            return base_recommendations
    
    def _analyze_feedback_patterns(self, feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in user feedback"""
        patterns = {
            'positive_feedback_ratio': 0.0,
            'genre_preferences_shift': {},
            'quality_expectation_change': 0.0,
            'discovery_appetite': 0.0,
            'cultural_exploration_trend': 0.0
        }
        
        if not feedback:
            return patterns
        
        positive_count = len([f for f in feedback if f.get('type') in ['like', 'favorite', 'high_rating']])
        patterns['positive_feedback_ratio'] = positive_count / len(feedback)
        
        # Analyze genre preferences from recent feedback
        genre_feedback = defaultdict(list)
        for f in feedback:
            content_genres = f.get('content_genres', [])
            feedback_type = f.get('type', 'neutral')
            for genre in content_genres:
                genre_feedback[genre].append(1 if feedback_type in ['like', 'favorite'] else 0)
        
        for genre, scores in genre_feedback.items():
            patterns['genre_preferences_shift'][genre] = np.mean(scores) if scores else 0.5
        
        return patterns
    
    def _should_apply_strategy(self, strategy_name: str, feedback_analysis: Dict[str, Any]) -> bool:
        """Determine if adaptation strategy should be applied"""
        thresholds = {
            'preference_drift': feedback_analysis.get('positive_feedback_ratio', 1.0) < 0.6,
            'quality_evolution': abs(feedback_analysis.get('quality_expectation_change', 0)) > 0.3,
            'discovery_expansion': feedback_analysis.get('discovery_appetite', 0) > 0.7,
            'cultural_exploration': feedback_analysis.get('cultural_exploration_trend', 0) > 0.6
        }
        
        return thresholds.get(strategy_name, False)
    
    def _handle_preference_drift(self, recommendations: List[Tuple[Any, float]], 
                               user_profile: Dict[str, Any], 
                               feedback_analysis: Dict[str, Any]) -> List[Tuple[Any, float]]:
        """Handle detected preference drift"""
        # Boost recommendations based on positive feedback genres
        genre_shifts = feedback_analysis.get('genre_preferences_shift', {})
        
        adjusted_recommendations = []
        for content, score in recommendations:
            try:
                content_genres = json.loads(content.genres or '[]')
                genre_boost = 0.0
                
                for genre in content_genres:
                    if genre in genre_shifts:
                        genre_boost += genre_shifts[genre] * 0.2
                
                adjusted_score = score + genre_boost
                adjusted_recommendations.append((content, min(adjusted_score, 1.0)))
            except:
                adjusted_recommendations.append((content, score))
        
        return adjusted_recommendations
    
    def _record_adaptation(self, user_id: int, feedback_analysis: Dict[str, Any], 
                         adapted_recommendations: List[Tuple[Any, float]]):
        """Record adaptation for future learning"""
        adaptation_record = {
            'timestamp': datetime.utcnow(),
            'feedback_analysis': feedback_analysis,
            'adaptation_applied': True,
            'recommendation_count': len(adapted_recommendations)
        }
        
        self.user_adaptation_history[user_id].append(adaptation_record)
        
        # Keep only recent history (last 50 adaptations)
        if len(self.user_adaptation_history[user_id]) > 50:
            self.user_adaptation_history[user_id] = self.user_adaptation_history[user_id][-50:]

class CineBrainRecommendationEngine:
    """
    Production-grade CineBrain Recommendation Engine
    Combines neural collaborative filtering, cultural awareness, and adaptive personalization
    """
    
    def __init__(self, db, models, cache=None):
        self.db = db
        self.models = models
        self.cache = cache
        
        # Initialize core components
        self.profile_analyzer = AdvancedProfileAnalyzer(db, models, cache)
        self.neural_cf = NeuralCollaborativeFiltering(embedding_dim=128)
        self.cultural_engine = CulturalAwarenessEngine()
        self.adaptive_engine = AdaptivePersonalizationEngine(cache)
        
        # Legacy algorithm integration
        self.content_based = ContentBasedFiltering()
        self.collaborative = CollaborativeFiltering()
        self.hybrid_engine = HybridRecommendationEngine()
        self.similarity_engine = UltraPowerfulSimilarityEngine()
        
        # Performance tracking
        self.performance_metrics = {
            'recommendations_generated': 0,
            'avg_generation_time': 0.0,
            'cache_hit_rate': 0.0,
            'user_satisfaction_estimate': 0.0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize neural models
        self._initialize_neural_models()
    
    def _initialize_neural_models(self):
        """Initialize and train neural collaborative filtering models"""
        try:
            # Get recent interactions for training
            cutoff_date = datetime.utcnow() - timedelta(days=90)  # Train on last 90 days
            
            with self.safe_db_operation():
                recent_interactions = self.models['UserInteraction'].query.filter(
                    self.models['UserInteraction'].timestamp >= cutoff_date
                ).all()
                
                content_ids = list(set([i.content_id for i in recent_interactions]))
                content_list = []
                
                if content_ids:
                    content_list = self.models['Content'].query.filter(
                        self.models['Content'].id.in_(content_ids)
                    ).all()
                
                # Train neural collaborative filtering
                self.neural_cf.train_embeddings(recent_interactions, content_list)
                
                logger.info(f"✅ Neural models initialized with {len(recent_interactions)} interactions")
                
        except Exception as e:
            logger.error(f"❌ Error initializing neural models: {e}")
    
    @contextmanager
    def safe_db_operation(self):
        """Safe database operation context manager"""
        try:
            yield
        except (OperationalError, DisconnectionError) as e:
            logger.error(f"Database connection error in recommendation engine: {e}")
            try:
                self.db.session.rollback()
                self.db.session.close()
            except:
                pass
            raise
        except Exception as e:
            logger.error(f"Database operation error: {e}")
            try:
                self.db.session.rollback()
            except:
                pass
            raise
    
    def generate_personalized_recommendations(self, user_id: int, 
                                            recommendation_type: str = 'for_you',
                                            limit: int = 50,
                                            filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate advanced personalized recommendations
        
        Args:
            user_id: Target user ID
            recommendation_type: Type of recommendations ('for_you', 'discover', 'trending', etc.)
            limit: Maximum number of recommendations
            filters: Additional filters (genre, language, etc.)
            
        Returns:
            Dict containing recommendations and metadata
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Check cache first
                cache_key = f"cinebrain:advanced_recs:{user_id}:{recommendation_type}:{limit}"
                if self.cache:
                    cached_result = self.cache.get(cache_key)
                    if cached_result:
                        logger.info(f"🚀 Cache hit for user {user_id} recommendations")
                        self.performance_metrics['cache_hit_rate'] += 1
                        return cached_result
                
                # Build comprehensive user profile
                user_profile = self.profile_analyzer.build_comprehensive_user_profile(user_id)
                
                if not user_profile:
                    logger.warning(f"⚠️ No profile available for user {user_id}, using cold start")
                    return self._generate_cold_start_recommendations(user_id, recommendation_type, limit)
                
                # Get content pool
                content_pool = self._get_content_pool(user_id, user_profile, filters)
                
                # Generate recommendations based on type
                recommendations = self._generate_typed_recommendations(
                    user_id, user_profile, content_pool, recommendation_type, limit
                )
                
                # Apply adaptive personalization
                recent_feedback = self._get_recent_feedback(user_id)
                if recent_feedback:
                    recommendations = self.adaptive_engine.adapt_recommendations(
                        user_id, recommendations, user_profile, recent_feedback
                    )
                
                # Apply final enhancements
                enhanced_recommendations = self._apply_final_enhancements(
                    recommendations, user_profile, recommendation_type
                )
                
                # Format response
                formatted_result = self._format_recommendation_response(
                    user_id, enhanced_recommendations, user_profile, recommendation_type, start_time
                )
                
                # Cache result
                if self.cache:
                    cache_timeout = 3600 if recommendation_type == 'for_you' else 1800
                    self.cache.set(cache_key, formatted_result, timeout=cache_timeout)
                
                # Update performance metrics
                self.performance_metrics['recommendations_generated'] += 1
                self.performance_metrics['avg_generation_time'] = (
                    self.performance_metrics['avg_generation_time'] * 0.9 + 
                    (time.time() - start_time) * 0.1
                )
                
                logger.info(f"✅ Generated {len(enhanced_recommendations)} {recommendation_type} recommendations for user {user_id} in {time.time() - start_time:.2f}s")
                
                return formatted_result
                
        except Exception as e:
            logger.error(f"❌ Error generating recommendations for user {user_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._generate_fallback_recommendations(user_id, recommendation_type, limit)
    
    def _generate_typed_recommendations(self, user_id: int, user_profile: Dict[str, Any],
                                      content_pool: List[Any], recommendation_type: str,
                                      limit: int) -> List[Tuple[Any, float]]:
        """Generate recommendations based on specific type"""
        
        generators = {
            'for_you': self._generate_for_you_recommendations,
            'discover': self._generate_discover_recommendations,
            'trending_for_you': self._generate_trending_personalized,
            'your_language': self._generate_language_specific,
            'because_you_watched': self._generate_because_you_watched,
            'hidden_gems': self._generate_hidden_gems,
            'quality_picks': self._generate_quality_picks,
            'telugu_specials': self._generate_telugu_specials
        }
        
        generator = generators.get(recommendation_type, self._generate_for_you_recommendations)
        return generator(user_id, user_profile, content_pool, limit)
    
    def _generate_for_you_recommendations(self, user_id: int, user_profile: Dict[str, Any],
                                        content_pool: List[Any], limit: int) -> List[Tuple[Any, float]]:
        """Generate comprehensive 'For You' recommendations"""
        try:
            recommendations = []
            
            # 1. Neural Collaborative Filtering (30%)
            neural_recs = self.neural_cf.get_user_recommendations(user_id, content_pool, limit)
            for content, score in neural_recs[:int(limit * 0.3)]:
                recommendations.append((content, score * RECOMMENDATION_CONFIG['neural_embedding_weight']))
            
            # 2. Cinematic DNA Matching (25%)
            cinematic_dna = user_profile.get('cinematic_dna', {})
            dna_matches = self._get_cinematic_dna_matches(cinematic_dna, content_pool, int(limit * 0.25))
            recommendations.extend(dna_matches)
            
            # 3. Cultural Awareness (20%)
            cultural_profile = user_profile.get('cultural_profile', {})
            cultural_recs = self._get_culturally_relevant_content(cultural_profile, content_pool, int(limit * 0.2))
            recommendations.extend(cultural_recs)
            
            # 4. Quality-based Recommendations (15%)
            quality_standards = user_profile.get('quality_standards', {})
            quality_recs = self._get_quality_recommendations(quality_standards, content_pool, int(limit * 0.15))
            recommendations.extend(quality_recs)
            
            # 5. Serendipity Injection (10%)
            serendipity_recs = self._inject_serendipity(user_profile, content_pool, int(limit * 0.1))
            recommendations.extend(serendipity_recs)
            
            # Remove duplicates and rank
            unique_recommendations = self._remove_duplicates_and_rank(recommendations)
            
            return unique_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error generating for_you recommendations: {e}")
            return []
    
    def _generate_discover_recommendations(self, user_id: int, user_profile: Dict[str, Any],
                                         content_pool: List[Any], limit: int) -> List[Tuple[Any, float]]:
        """Generate discovery recommendations outside user's comfort zone"""
        try:
            # Get user's typical preferences
            typical_genres = set(user_profile.get('cinematic_dna', {}).get('dominant_genres', []))
            typical_languages = set(user_profile.get('cultural_profile', {}).get('language_distribution', {}).keys())
            
            discovery_content = []
            
            # Filter for discovery content (different from typical preferences)
            for content in content_pool:
                try:
                    content_genres = set(json.loads(content.genres or '[]'))
                    content_languages = set(self._extract_content_languages(content))
                    
                    # Calculate novelty score
                    genre_novelty = len(content_genres - typical_genres) / max(len(content_genres), 1)
                    language_novelty = len(content_languages - typical_languages) / max(len(content_languages), 1)
                    
                    novelty_score = (genre_novelty + language_novelty) / 2
                    
                    # Only include if sufficiently novel and high quality
                    if novelty_score > 0.3 and content.rating and content.rating >= 7.0:
                        # Boost Telugu content even in discovery
                        telugu_boost = 1.2 if any('telugu' in lang.lower() for lang in content_languages) else 1.0
                        
                        final_score = novelty_score * (content.rating / 10) * telugu_boost
                        discovery_content.append((content, final_score))
                        
                except Exception as e:
                    continue
            
            # Sort by discovery score
            discovery_content.sort(key=lambda x: x[1], reverse=True)
            
            return discovery_content[:limit]
            
        except Exception as e:
            logger.error(f"Error generating discover recommendations: {e}")
            return []
    
    def _generate_telugu_specials(self, user_id: int, user_profile: Dict[str, Any],
                                content_pool: List[Any], limit: int) -> List[Tuple[Any, float]]:
        """Generate special Telugu content recommendations"""
        try:
            telugu_content = []
            
            for content in content_pool:
                if self._is_telugu_content(content):
                    # Calculate Telugu-specific score
                    telugu_score = 0.0
                    
                    # Base quality score
                    if content.rating:
                        telugu_score += (content.rating / 10) * 0.4
                    
                    # Popularity in Telugu cinema
                    if content.popularity:
                        telugu_score += min(content.popularity / 100, 1.0) * 0.3
                    
                    # Cultural authenticity
                    authenticity_score = self.cultural_engine._has_cultural_authenticity(content, 'telugu')
                    telugu_score += (1.0 if authenticity_score else 0.5) * 0.3
                    
                    telugu_content.append((content, telugu_score))
            
            # Sort by Telugu-specific score
            telugu_content.sort(key=lambda x: x[1], reverse=True)
            
            return telugu_content[:limit]
            
        except Exception as e:
            logger.error(f"Error generating Telugu specials: {e}")
            return []
    
    def _get_content_pool(self, user_id: int, user_profile: Dict[str, Any], 
                         filters: Dict[str, Any] = None) -> List[Any]:
        """Get filtered content pool for recommendations"""
        try:
            with self.safe_db_operation():
                # Get user's interacted content to exclude
                user_interactions = self.models['UserInteraction'].query.filter_by(
                    user_id=user_id
                ).all()
                
                interacted_content_ids = set([i.content_id for i in user_interactions])
                
                # Build query
                query = self.models['Content'].query.filter(
                    ~self.models['Content'].id.in_(interacted_content_ids)
                ).filter(
                    self.models['Content'].rating.isnot(None)
                ).filter(
                    self.models['Content'].rating >= user_profile.get('quality_standards', {}).get('minimum_rating', 6.0)
                )
                
                # Apply filters if provided
                if filters:
                    if 'genres' in filters:
                        # This would need proper genre filtering
                        pass
                    
                    if 'languages' in filters:
                        for language in filters['languages']:
                            query = query.filter(self.models['Content'].languages.contains(f'"{language}"'))
                    
                    if 'content_type' in filters:
                        query = query.filter(self.models['Content'].content_type == filters['content_type'])
                
                # Order by popularity and limit for performance
                content_pool = query.order_by(desc(self.models['Content'].popularity)).limit(2000).all()
                
                logger.info(f"Retrieved content pool of {len(content_pool)} items for user {user_id}")
                return content_pool
                
        except Exception as e:
            logger.error(f"Error getting content pool: {e}")
            return []
    
    def _get_recent_feedback(self, user_id: int) -> List[Dict[str, Any]]:
        """Get recent user feedback for adaptive personalization"""
        try:
            # This would integrate with a feedback tracking system
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent feedback: {e}")
            return []
    
    def _format_recommendation_response(self, user_id: int, recommendations: List[Tuple[Any, float]],
                                      user_profile: Dict[str, Any], recommendation_type: str,
                                      start_time: float) -> Dict[str, Any]:
        """Format recommendations for API response"""
        try:
            formatted_recommendations = []
            
            for rank, (content, score) in enumerate(recommendations, 1):
                # Extract content languages
                content_languages = self._extract_content_languages(content)
                
                # Determine primary language
                primary_language = 'telugu'
                for lang in PRIORITY_LANGUAGES:
                    if lang in content_languages:
                        primary_language = lang
                        break
                
                # Generate recommendation reason
                reason = self._generate_recommendation_reason(
                    content, user_profile, recommendation_type, score
                )
                
                formatted_rec = {
                    'rank': rank,
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': content_languages,
                    'primary_language': primary_language,
                    'rating': content.rating,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'poster_path': self._format_poster_path(content.poster_path),
                    'overview': content.overview[:200] + '...' if content.overview and len(content.overview) > 200 else content.overview,
                    'recommendation_score': round(score, 4),
                    'recommendation_reason': reason,
                    'cultural_relevance': self.cultural_engine.calculate_cultural_relevance(
                        content, user_profile.get('cultural_profile', {})
                    ),
                    'confidence': self._calculate_recommendation_confidence(score, user_profile),
                    'youtube_trailer_id': content.youtube_trailer_id
                }
                
                formatted_recommendations.append(formatted_rec)
            
            # Generate response metadata
            response = {
                'success': True,
                'user_id': user_id,
                'recommendation_type': recommendation_type,
                'recommendations': formatted_recommendations,
                'total_count': len(formatted_recommendations),
                'user_insights': {
                    'telugu_cinema_affinity': user_profile.get('telugu_cinema_affinity', 0.8),
                    'personalization_strength': user_profile.get('personalization_strength', 0.6),
                    'recommendation_confidence': user_profile.get('recommendation_confidence', 0.7),
                    'cultural_profile': user_profile.get('cultural_profile', {}).get('primary_culture', 'telugu')
                },
                'algorithm_metadata': {
                    'engine_version': '3.0_neural_cultural',
                    'techniques_used': [
                        'neural_collaborative_filtering',
                        'cinematic_dna_analysis',
                        'cultural_awareness_engine',
                        'adaptive_personalization',
                        'telugu_cinema_prioritization'
                    ],
                    'processing_time': round(time.time() - start_time, 3),
                    'content_pool_size': len(self._get_content_pool(user_id, user_profile)),
                    'personalization_level': 'advanced',
                    'cache_used': False
                },
                'next_refresh': (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting recommendation response: {e}")
            return {
                'success': False,
                'error': 'Failed to format recommendations',
                'user_id': user_id,
                'recommendation_type': recommendation_type
            }
    
    def _generate_recommendation_reason(self, content: Any, user_profile: Dict[str, Any],
                                      recommendation_type: str, score: float) -> str:
        """Generate human-readable recommendation reason"""
        try:
            reasons = []
            
            # Check for cinematic DNA match
            cinematic_dna = user_profile.get('cinematic_dna', {})
            if cinematic_dna.get('telugu_cinema_connection', {}).get('affinity_score', 0) > 0.7:
                if self._is_telugu_content(content):
                    reasons.append("matches your Telugu cinema preferences")
            
            # Check for quality alignment
            if content.rating and content.rating >= user_profile.get('quality_standards', {}).get('minimum_rating', 8.0):
                reasons.append("meets your high quality standards")
            
            # Check for genre preferences
            content_genres = set(json.loads(content.genres or '[]'))
            user_top_genres = set(user_profile.get('cinematic_dna', {}).get('dominant_genres', []))
            
            if content_genres & user_top_genres:
                common_genres = list(content_genres & user_top_genres)
                if len(common_genres) == 1:
                    reasons.append(f"you enjoy {common_genres[0].lower()} content")
                else:
                    reasons.append(f"combines {' and '.join(common_genres[:2]).lower()} elements you love")
            
            # Type-specific reasons
            if recommendation_type == 'discover':
                reasons.append("expands your cinematic horizons")
            elif recommendation_type == 'trending_for_you':
                reasons.append("trending content tailored to your taste")
            elif recommendation_type == 'hidden_gems':
                reasons.append("high-quality content you might have missed")
            
            # Default reason
            if not reasons:
                reasons.append("recommended based on your viewing patterns")
            
            # Construct final reason
            if len(reasons) == 1:
                return f"CineBrain suggests this because it {reasons[0]}"
            else:
                return f"CineBrain recommends this as it {reasons[0]} and {reasons[1]}"
                
        except Exception as e:
            logger.error(f"Error generating recommendation reason: {e}")
            return "Recommended by CineBrain's advanced AI"
    
    def _is_telugu_content(self, content: Any) -> bool:
        """Check if content is Telugu"""
        if not content.languages:
            return False
        try:
            languages = json.loads(content.languages)
            return any('telugu' in lang.lower() or 'te' in lang.lower() for lang in languages)
        except:
            return False
    
    def _extract_content_languages(self, content: Any) -> List[str]:
        """Extract content languages"""
        if not content.languages:
            return []
        try:
            languages = json.loads(content.languages)
            return [lang.strip().title() for lang in languages]
        except:
            return []
    
    def _format_poster_path(self, poster_path: str) -> str:
        """Format poster path for display"""
        if not poster_path:
            return None
        
        if poster_path.startswith('http'):
            return poster_path
        else:
            return f"https://image.tmdb.org/t/p/w300{poster_path}"
    
    def _generate_cold_start_recommendations(self, user_id: int, recommendation_type: str, 
                                           limit: int) -> Dict[str, Any]:
        """Generate recommendations for new users"""
        try:
            # Get popular Telugu content as default
            with self.safe_db_operation():
                popular_telugu = self.models['Content'].query.filter(
                    self.models['Content'].languages.contains('"Telugu"')
                ).order_by(desc(self.models['Content'].rating)).limit(limit).all()
                
                recommendations = []
                for rank, content in enumerate(popular_telugu, 1):
                    recommendations.append({
                        'rank': rank,
                        'id': content.id,
                        'slug': content.slug,
                        'title': content.title,
                        'content_type': content.content_type,
                        'rating': content.rating,
                        'poster_path': self._format_poster_path(content.poster_path),
                        'recommendation_reason': 'Popular Telugu content to get you started on CineBrain'
                    })
                
                return {
                    'success': True,
                    'user_id': user_id,
                    'recommendation_type': recommendation_type,
                    'recommendations': recommendations,
                    'cold_start': True,
                    'message': 'Start interacting with content to get personalized recommendations!'
                }
                
        except Exception as e:
            logger.error(f"Error generating cold start recommendations: {e}")
            return self._generate_fallback_recommendations(user_id, recommendation_type, limit)
    
    def _generate_fallback_recommendations(self, user_id: int, recommendation_type: str, 
                                         limit: int) -> Dict[str, Any]:
        """Generate fallback recommendations for error cases"""
        return {
            'success': False,
            'user_id': user_id,
            'recommendation_type': recommendation_type,
            'error': 'Unable to generate personalized recommendations',
            'recommendations': [],
            'fallback': True
        }
    
    # Additional helper methods for comprehensive recommendation generation
    def _get_cinematic_dna_matches(self, cinematic_dna: Dict[str, Any], 
                                 content_pool: List[Any], limit: int) -> List[Tuple[Any, float]]:
        """Get content matching cinematic DNA"""
        # This would use the existing cinematic DNA analysis
        return []
    
    def _get_culturally_relevant_content(self, cultural_profile: Dict[str, Any],
                                       content_pool: List[Any], limit: int) -> List[Tuple[Any, float]]:
        """Get culturally relevant content"""
        relevant_content = []
        
        for content in content_pool:
            relevance_score = self.cultural_engine.calculate_cultural_relevance(content, cultural_profile)
            if relevance_score > 0.5:
                relevant_content.append((content, relevance_score))
        
        relevant_content.sort(key=lambda x: x[1], reverse=True)
        return relevant_content[:limit]
    
    def _get_quality_recommendations(self, quality_standards: Dict[str, Any],
                                   content_pool: List[Any], limit: int) -> List[Tuple[Any, float]]:
        """Get high-quality recommendations"""
        quality_content = []
        min_rating = quality_standards.get('minimum_rating', 8.0)
        
        for content in content_pool:
            if content.rating and content.rating >= min_rating:
                quality_score = content.rating / 10
                quality_content.append((content, quality_score))
        
        quality_content.sort(key=lambda x: x[1], reverse=True)
        return quality_content[:limit]
    
    def _inject_serendipity(self, user_profile: Dict[str, Any],
                          content_pool: List[Any], limit: int) -> List[Tuple[Any, float]]:
        """Inject serendipitous recommendations"""
        # Select random high-quality content for serendipity
        high_quality_content = [c for c in content_pool if c.rating and c.rating >= 7.5]
        
        if len(high_quality_content) > limit:
            serendipity_content = random.sample(high_quality_content, limit)
        else:
            serendipity_content = high_quality_content
        
        return [(content, 0.7) for content in serendipity_content]
    
    def _remove_duplicates_and_rank(self, recommendations: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """Remove duplicates and rank recommendations"""
        seen_ids = set()
        unique_recommendations = []
        
        for content, score in recommendations:
            if content.id not in seen_ids:
                seen_ids.add(content.id)
                unique_recommendations.append((content, score))
        
        # Sort by score
        unique_recommendations.sort(key=lambda x: x[1], reverse=True)
        return unique_recommendations
    
    def _calculate_recommendation_confidence(self, score: float, user_profile: Dict[str, Any]) -> str:
        """Calculate confidence level for recommendation"""
        base_confidence = user_profile.get('recommendation_confidence', 0.5)
        combined_confidence = (score + base_confidence) / 2
        
        if combined_confidence > 0.8:
            return 'very_high'
        elif combined_confidence > 0.6:
            return 'high'
        elif combined_confidence > 0.4:
            return 'medium'
        else:
            return 'low'
    
    # Placeholder methods for additional recommendation types
    def _generate_trending_personalized(self, user_id: int, user_profile: Dict[str, Any],
                                      content_pool: List[Any], limit: int) -> List[Tuple[Any, float]]:
        """Generate personalized trending recommendations"""
        return []
    
    def _generate_language_specific(self, user_id: int, user_profile: Dict[str, Any],
                                  content_pool: List[Any], limit: int) -> List[Tuple[Any, float]]:
        """Generate language-specific recommendations"""
        return []
    
    def _generate_because_you_watched(self, user_id: int, user_profile: Dict[str, Any],
                                    content_pool: List[Any], limit: int) -> List[Tuple[Any, float]]:
        """Generate 'because you watched' recommendations"""
        return []
    
    def _generate_hidden_gems(self, user_id: int, user_profile: Dict[str, Any],
                            content_pool: List[Any], limit: int) -> List[Tuple[Any, float]]:
        """Generate hidden gem recommendations"""
        return []
    
    def _generate_quality_picks(self, user_id: int, user_profile: Dict[str, Any],
                              content_pool: List[Any], limit: int) -> List[Tuple[Any, float]]:
        """Generate quality pick recommendations"""
        return []