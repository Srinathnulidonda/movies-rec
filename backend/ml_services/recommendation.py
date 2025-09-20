import numpy as np
from datetime import datetime, timedelta
import json
import logging
import hashlib
from collections import defaultdict, Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from .algorithm import (
    AdvancedCollaborativeFiltering, AdvancedContentBasedFiltering, 
    AdvancedMatrixFactorization, HybridRecommendationSystem,
    RealtimePersonalizationEngine
)

logger = logging.getLogger(__name__)

class AdvancedRecommendationEngine:
    def __init__(self, db, models, config=None):
        self.db = db
        self.models = models
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
        self.config = config or {
            'cache_ttl': 3600,
            'max_cache_size': 10000,
            'async_processing': True,
            'enable_explanation': True,
            'enable_realtime_learning': True,
            'performance_tracking': True,
            'min_confidence_threshold': 0.1,
            'diversity_enforcement': True,
            'novelty_boost': True,
            'cold_start_strategy': 'advanced'
        }
        
        # Initialize algorithm components
        self.collaborative_filtering = AdvancedCollaborativeFiltering(db, models)
        self.content_filtering = AdvancedContentBasedFiltering(db, models)
        self.matrix_factorization = AdvancedMatrixFactorization(db, models)
        self.hybrid_system = HybridRecommendationSystem(db, models)
        
        if self.config['enable_realtime_learning']:
            self.realtime_engine = RealtimePersonalizationEngine(db, models)
        else:
            self.realtime_engine = None
        
        # Caching and performance tracking
        self.recommendation_cache = {}
        self.cache_timestamps = {}
        self.performance_metrics = defaultdict(list)
        self.user_profiles_cache = {}
        
        # Thread safety
        self._cache_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        
        # Background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks for cache cleanup and model updates"""
        def background_worker():
            while True:
                try:
                    self._cleanup_expired_cache()
                    if self.realtime_engine:
                        self.realtime_engine.cleanup_old_sessions()
                    time.sleep(300)  # Run every 5 minutes
                except Exception as e:
                    logger.error(f"Background task error: {e}")
                    time.sleep(60)
        
        if self.config['async_processing']:
            background_thread = threading.Thread(target=background_worker, daemon=True)
            background_thread.start()
    
    def get_personalized_recommendations(self, user_id, limit=20, content_type='all', 
                                       strategy='adaptive', context=None):
        """Main entry point for personalized recommendations"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(user_id, limit, content_type, strategy, context)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result:
                self._record_performance_metric('cache_hit', time.time() - start_time)
                return cached_result
            
            # Get user profile strength
            profile_strength = self._assess_user_profile_strength(user_id)
            
            # Choose optimal strategy based on user data
            if strategy == 'adaptive':
                strategy = self._choose_optimal_strategy(user_id, profile_strength, context)
            
            # Generate recommendations
            recommendations = self._generate_recommendations_by_strategy(
                user_id, limit, content_type, strategy, context
            )
            
            # Apply post-processing filters
            recommendations = self._apply_post_processing(
                recommendations, user_id, limit, context
            )
            
            # Generate explanations if enabled
            if self.config['enable_explanation']:
                recommendations = self._add_explanations(recommendations, user_id)
            
            # Cache the result
            self._store_in_cache(cache_key, recommendations)
            
            # Record performance metrics
            self._record_performance_metric('recommendation_generation', time.time() - start_time)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            # Fallback to simple popular content
            return self._get_fallback_recommendations(user_id, limit, content_type)
    
    def get_advanced_recommendations(self, user_id, limit=20, include_explanations=True,
                                   diversity_factor=0.3, novelty_factor=0.2, context=None):
        """Advanced recommendations with comprehensive ML features"""
        start_time = time.time()
        
        try:
            # Get hybrid recommendations
            recommendations = self.hybrid_system.get_hybrid_recommendations(
                user_id, limit * 2, strategy='adaptive'
            )
            
            # Apply real-time adjustments if enabled
            if self.realtime_engine:
                recommendations = self.realtime_engine.get_realtime_adjusted_recommendations(
                    user_id, recommendations, limit * 2
                )
            
            # Advanced post-processing
            recommendations = self._apply_advanced_post_processing(
                recommendations, user_id, limit, diversity_factor, novelty_factor, context
            )
            
            # Add comprehensive explanations
            if include_explanations:
                recommendations = self._add_advanced_explanations(recommendations, user_id, context)
            
            # Calculate confidence scores
            recommendations = self._calculate_confidence_scores(recommendations, user_id)
            
            # Add novelty and diversity metrics
            recommendations = self._add_quality_metrics(recommendations, user_id)
            
            self._record_performance_metric('advanced_recommendation_generation', time.time() - start_time)
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error generating advanced recommendations for user {user_id}: {e}")
            # Fallback to basic recommendations
            return self.get_personalized_recommendations(user_id, limit, 'all', 'hybrid', context)
    
    def _choose_optimal_strategy(self, user_id, profile_strength, context):
        """Choose optimal recommendation strategy based on user profile and context"""
        interaction_count = self.get_user_interaction_count(user_id)
        
        # Context-aware strategy selection
        if context:
            if context.get('device') == 'mobile':
                # Mobile users prefer faster, simpler recommendations
                return 'content_based' if interaction_count < 10 else 'collaborative'
            
            if context.get('time_of_day') == 'evening':
                # Evening users might want different content
                return 'hybrid'
            
            if context.get('location_type') == 'home':
                # Home viewing might prefer longer content
                return 'matrix_factorization' if interaction_count > 20 else 'hybrid'
        
        # Profile strength-based strategy
        if profile_strength == 'new_user':
            return 'content_based'
        elif profile_strength == 'weak':
            return 'hybrid'
        elif profile_strength == 'moderate':
            return 'collaborative'
        elif profile_strength in ['strong', 'very_strong']:
            return 'matrix_factorization'
        else:
            return 'hybrid'
    
    def _generate_recommendations_by_strategy(self, user_id, limit, content_type, strategy, context):
        """Generate recommendations using specified strategy"""
        if strategy == 'collaborative':
            return self._get_collaborative_recommendations(user_id, limit, content_type)
        elif strategy == 'content_based':
            return self._get_content_based_recommendations(user_id, limit, content_type)
        elif strategy == 'matrix_factorization':
            return self._get_matrix_factorization_recommendations(user_id, limit, content_type)
        elif strategy == 'hybrid':
            return self._get_hybrid_recommendations(user_id, limit, content_type)
        else:
            return self._get_hybrid_recommendations(user_id, limit, content_type)
    
    def _get_collaborative_recommendations(self, user_id, limit, content_type):
        """Get collaborative filtering recommendations"""
        try:
            raw_recommendations = self.collaborative_filtering.user_based_recommendations_advanced(
                user_id, limit * 2
            )
            
            recommendations = []
            for content_id, score in raw_recommendations:
                content = self.Content.query.get(content_id)
                if content and (content_type == 'all' or content.content_type == content_type):
                    recommendations.append({
                        'content': content,
                        'score': score,
                        'algorithm': 'collaborative_filtering',
                        'reason': 'Based on users with similar preferences',
                        'confidence': min(score / 5.0, 1.0)
                    })
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Collaborative filtering error: {e}")
            return []
    
    def _get_content_based_recommendations(self, user_id, limit, content_type):
        """Get content-based filtering recommendations"""
        try:
            raw_recommendations = self.content_filtering.get_content_recommendations_advanced(
                user_id, limit * 2
            )
            
            recommendations = []
            for content_id, score in raw_recommendations:
                content = self.Content.query.get(content_id)
                if content and (content_type == 'all' or content.content_type == content_type):
                    recommendations.append({
                        'content': content,
                        'score': score,
                        'algorithm': 'content_based_filtering',
                        'reason': 'Based on content you previously enjoyed',
                        'confidence': min(score, 1.0)
                    })
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Content-based filtering error: {e}")
            return []
    
    def _get_matrix_factorization_recommendations(self, user_id, limit, content_type):
        """Get matrix factorization recommendations"""
        try:
            raw_recommendations = self.matrix_factorization.get_recommendations_advanced(
                user_id, limit * 2
            )
            
            recommendations = []
            for content_id, score in raw_recommendations:
                content = self.Content.query.get(content_id)
                if content and (content_type == 'all' or content.content_type == content_type):
                    # Normalize MF scores
                    normalized_score = max(0, min(1, (score + 5) / 10))
                    
                    recommendations.append({
                        'content': content,
                        'score': normalized_score,
                        'algorithm': 'matrix_factorization',
                        'reason': 'Based on latent preference patterns',
                        'confidence': min(abs(score) / 5.0, 1.0)
                    })
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Matrix factorization error: {e}")
            return []
    
    def _get_hybrid_recommendations(self, user_id, limit, content_type):
        """Get hybrid recommendations"""
        try:
            recommendations = self.hybrid_system.get_hybrid_recommendations(
                user_id, limit * 2, strategy='adaptive'
            )
            
            # Filter by content type
            if content_type != 'all':
                recommendations = [
                    rec for rec in recommendations 
                    if rec['content'].content_type == content_type
                ]
            
            # Format for consistency
            for rec in recommendations:
                rec['algorithm'] = 'hybrid_system'
                rec['reason'] = 'Based on multiple advanced algorithms'
                if 'confidence' not in rec:
                    rec['confidence'] = min(rec['score'], 1.0)
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Hybrid recommendations error: {e}")
            return []
    
    def _apply_post_processing(self, recommendations, user_id, limit, context):
        """Apply post-processing filters and enhancements"""
        if not recommendations:
            return recommendations
        
        # Remove duplicates
        seen_content_ids = set()
        unique_recommendations = []
        
        for rec in recommendations:
            content_id = rec['content'].id
            if content_id not in seen_content_ids:
                seen_content_ids.add(content_id)
                unique_recommendations.append(rec)
        
        recommendations = unique_recommendations
        
        # Apply diversity if enabled
        if self.config['diversity_enforcement']:
            recommendations = self._apply_diversity_filter(recommendations, 0.2, limit * 2)
        
        # Apply novelty boost if enabled
        if self.config['novelty_boost']:
            recommendations = self._apply_novelty_boost(recommendations, user_id)
        
        # Context-aware adjustments
        if context:
            recommendations = self._apply_context_adjustments(recommendations, context)
        
        # Filter by confidence threshold
        recommendations = [
            rec for rec in recommendations 
            if rec.get('confidence', 0) >= self.config['min_confidence_threshold']
        ]
        
        return recommendations
    
    def _apply_advanced_post_processing(self, recommendations, user_id, limit, 
                                      diversity_factor, novelty_factor, context):
        """Apply advanced post-processing with custom factors"""
        if not recommendations:
            return recommendations
        
        # Advanced diversity algorithm
        if diversity_factor > 0:
            recommendations = self._apply_advanced_diversity(recommendations, diversity_factor, limit)
        
        # Advanced novelty algorithm
        if novelty_factor > 0:
            recommendations = self._apply_advanced_novelty(recommendations, user_id, novelty_factor)
        
        # Serendipity injection
        recommendations = self._inject_serendipity(recommendations, user_id, 0.1)
        
        # Quality assurance
        recommendations = self._apply_quality_assurance(recommendations, user_id)
        
        return recommendations
    
    def _apply_diversity_filter(self, recommendations, diversity_factor, limit):
        """Apply diversity filter to recommendations"""
        if not recommendations or len(recommendations) <= 1:
            return recommendations
        
        final_recommendations = []
        remaining = recommendations[:]
        
        # Start with the highest scored item
        final_recommendations.append(remaining.pop(0))
        
        while len(final_recommendations) < limit and remaining:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for idx, rec in enumerate(remaining):
                # Calculate diversity score
                diversity_score = self._calculate_recommendation_diversity(
                    rec, final_recommendations
                )
                
                # Combine relevance and diversity
                combined_score = ((1 - diversity_factor) * rec['score'] + 
                                diversity_factor * diversity_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = rec
                    best_idx = idx
            
            if best_candidate:
                final_recommendations.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break
        
        return final_recommendations
    
    def _apply_advanced_diversity(self, recommendations, diversity_factor, limit):
        """Apply advanced diversity algorithm with multiple diversity measures"""
        if not recommendations:
            return recommendations
        
        final_recommendations = []
        remaining = recommendations[:]
        
        # Start with top recommendation
        final_recommendations.append(remaining.pop(0))
        
        while len(final_recommendations) < limit and remaining:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for idx, rec in enumerate(remaining):
                # Multiple diversity measures
                genre_diversity = self._calculate_genre_diversity(rec, final_recommendations)
                language_diversity = self._calculate_language_diversity(rec, final_recommendations)
                type_diversity = self._calculate_type_diversity(rec, final_recommendations)
                temporal_diversity = self._calculate_temporal_diversity(rec, final_recommendations)
                
                # Weighted diversity score
                diversity_score = (0.4 * genre_diversity + 
                                 0.25 * language_diversity + 
                                 0.2 * type_diversity + 
                                 0.15 * temporal_diversity)
                
                # Combined score
                combined_score = ((1 - diversity_factor) * rec['score'] + 
                                diversity_factor * diversity_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = rec
                    best_idx = idx
            
            if best_candidate:
                final_recommendations.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break
        
        return final_recommendations
    
    def _calculate_genre_diversity(self, candidate_rec, selected_recs):
        """Calculate genre diversity score"""
        try:
            candidate_genres = set(json.loads(candidate_rec['content'].genres or '[]'))
            
            if not selected_recs:
                return 1.0
            
            diversity_scores = []
            for selected_rec in selected_recs:
                try:
                    selected_genres = set(json.loads(selected_rec['content'].genres or '[]'))
                    overlap = len(candidate_genres & selected_genres)
                    union = len(candidate_genres | selected_genres)
                    diversity = 1.0 - (overlap / max(union, 1))
                    diversity_scores.append(diversity)
                except:
                    diversity_scores.append(0.5)
            
            return np.mean(diversity_scores)
        except:
            return 0.5
    
    def _calculate_language_diversity(self, candidate_rec, selected_recs):
        """Calculate language diversity score"""
        try:
            candidate_languages = set(json.loads(candidate_rec['content'].languages or '[]'))
            
            if not selected_recs:
                return 1.0
            
            diversity_scores = []
            for selected_rec in selected_recs:
                try:
                    selected_languages = set(json.loads(selected_rec['content'].languages or '[]'))
                    if not candidate_languages & selected_languages:
                        diversity_scores.append(1.0)
                    else:
                        overlap = len(candidate_languages & selected_languages)
                        diversity = 1.0 - (overlap / max(len(candidate_languages), 1))
                        diversity_scores.append(diversity)
                except:
                    diversity_scores.append(0.5)
            
            return np.mean(diversity_scores)
        except:
            return 0.5
    
    def _calculate_type_diversity(self, candidate_rec, selected_recs):
        """Calculate content type diversity score"""
        candidate_type = candidate_rec['content'].content_type
        
        if not selected_recs:
            return 1.0
        
        selected_types = [rec['content'].content_type for rec in selected_recs]
        type_diversity = 1.0 - (selected_types.count(candidate_type) / len(selected_types))
        
        return type_diversity
    
    def _calculate_temporal_diversity(self, candidate_rec, selected_recs):
        """Calculate temporal (release year) diversity score"""
        try:
            candidate_year = candidate_rec['content'].release_date.year if candidate_rec['content'].release_date else None
            
            if not candidate_year or not selected_recs:
                return 1.0
            
            selected_years = []
            for rec in selected_recs:
                if rec['content'].release_date:
                    selected_years.append(rec['content'].release_date.year)
            
            if not selected_years:
                return 1.0
            
            # Calculate average year difference
            year_differences = [abs(candidate_year - year) for year in selected_years]
            avg_difference = np.mean(year_differences)
            
            # Normalize to 0-1 scale (20 years = max diversity)
            diversity_score = min(avg_difference / 20.0, 1.0)
            
            return diversity_score
        except:
            return 0.5
    
    def _apply_novelty_boost(self, recommendations, user_id):
        """Apply novelty boost to recommendations"""
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not user_interactions:
            return recommendations
        
        # Analyze user's historical preferences
        user_preferences = self._analyze_user_preferences(user_interactions)
        
        # Boost novelty scores
        for rec in recommendations:
            novelty_score = self._calculate_novelty_score(rec['content'], user_preferences)
            rec['novelty_score'] = novelty_score
            
            # Apply novelty boost
            novelty_boost = novelty_score * 0.1  # 10% maximum boost
            rec['score'] = rec['score'] * (1 + novelty_boost)
        
        return recommendations
    
    def _apply_advanced_novelty(self, recommendations, user_id, novelty_factor):
        """Apply advanced novelty algorithm"""
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not user_interactions:
            return recommendations
        
        user_preferences = self._analyze_user_preferences(user_interactions)
        
        for rec in recommendations:
            # Multiple novelty measures
            genre_novelty = self._calculate_genre_novelty(rec['content'], user_preferences)
            language_novelty = self._calculate_language_novelty(rec['content'], user_preferences)
            type_novelty = self._calculate_type_novelty(rec['content'], user_preferences)
            temporal_novelty = self._calculate_temporal_novelty(rec['content'], user_preferences)
            
            # Combined novelty score
            novelty_score = (0.4 * genre_novelty + 
                           0.25 * language_novelty + 
                           0.2 * type_novelty + 
                           0.15 * temporal_novelty)
            
            rec['novelty_score'] = novelty_score
            
            # Apply novelty factor
            rec['score'] = ((1 - novelty_factor) * rec['score'] + 
                          novelty_factor * novelty_score)
        
        return recommendations
    
    def _inject_serendipity(self, recommendations, user_id, serendipity_factor):
        """Inject serendipitous recommendations"""
        if serendipity_factor <= 0 or not recommendations:
            return recommendations
        
        # Number of serendipitous items to inject
        num_serendipitous = max(1, int(len(recommendations) * serendipity_factor))
        
        # Get serendipitous candidates
        serendipitous_candidates = self._get_serendipitous_candidates(user_id, num_serendipitous * 3)
        
        # Replace some recommendations with serendipitous ones
        if serendipitous_candidates:
            # Replace lowest scoring recommendations
            recommendations = recommendations[:-num_serendipitous]
            
            for candidate in serendipitous_candidates[:num_serendipitous]:
                recommendations.append({
                    'content': candidate,
                    'score': 0.5,  # Medium score for serendipity
                    'algorithm': 'serendipity_injection',
                    'reason': 'Something different you might enjoy',
                    'confidence': 0.4,
                    'serendipitous': True
                })
        
        return recommendations
    
    def _get_serendipitous_candidates(self, user_id, limit):
        """Get candidates for serendipitous recommendations"""
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        user_seen_content = set(interaction.content_id for interaction in user_interactions)
        
        # Get user's typical preferences
        user_preferences = self._analyze_user_preferences(user_interactions)
        
        # Find content that's different from user's typical preferences
        candidates = []
        
        # Get highly rated content from different genres/languages
        for content in self.Content.query.filter(
            ~self.Content.id.in_(user_seen_content),
            self.Content.rating >= 7.0
        ).limit(limit * 5).all():
            
            # Check if it's sufficiently different
            if self._is_serendipitous(content, user_preferences):
                candidates.append(content)
                
                if len(candidates) >= limit:
                    break
        
        return candidates
    
    def _is_serendipitous(self, content, user_preferences):
        """Check if content is serendipitous for the user"""
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            content_languages = set(json.loads(content.languages or '[]'))
            
            # Check genre novelty
            common_genres = content_genres & set(user_preferences['top_genres'])
            genre_novelty = len(common_genres) / max(len(content_genres), 1) < 0.3
            
            # Check language novelty
            common_languages = content_languages & set(user_preferences['top_languages'])
            language_novelty = len(common_languages) == 0
            
            # Check type novelty
            type_novelty = content.content_type not in user_preferences['top_types'][:2]
            
            # Content is serendipitous if it's novel in at least 2 dimensions
            novelty_count = sum([genre_novelty, language_novelty, type_novelty])
            
            return novelty_count >= 2
        except:
            return False
    
    def _apply_quality_assurance(self, recommendations, user_id):
        """Apply quality assurance checks"""
        quality_assured = []
        
        for rec in recommendations:
            content = rec['content']
            
            # Quality checks
            quality_score = 0.0
            quality_factors = []
            
            # Rating quality
            if content.rating and content.rating >= 6.0:
                quality_factors.append(min(content.rating / 10.0, 1.0))
            
            # Vote count (popularity indicator)
            if content.vote_count and content.vote_count >= 10:
                vote_score = min(content.vote_count / 1000.0, 1.0)
                quality_factors.append(vote_score)
            
            # Recency factor (newer content gets slight boost)
            if content.release_date:
                years_old = (datetime.now().year - content.release_date.year)
                recency_score = max(0, 1 - years_old / 20.0)
                quality_factors.append(recency_score * 0.5)
            
            # Calculate overall quality score
            if quality_factors:
                quality_score = np.mean(quality_factors)
            
            # Only include if quality meets minimum threshold
            if quality_score >= 0.3:  # Minimum quality threshold
                rec['quality_score'] = quality_score
                quality_assured.append(rec)
        
        return quality_assured
    
    def _add_explanations(self, recommendations, user_id):
        """Add explanations to recommendations"""
        for rec in recommendations:
            explanation = self._generate_explanation(rec, user_id)
            rec['explanation'] = explanation
        
        return recommendations
    
    def _add_advanced_explanations(self, recommendations, user_id, context):
        """Add comprehensive explanations with context"""
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        user_preferences = self._analyze_user_preferences(user_interactions)
        
        for rec in recommendations:
            explanation = self._generate_advanced_explanation(rec, user_preferences, context)
            rec['explanation'] = explanation
            rec['explanation_confidence'] = self._calculate_explanation_confidence(explanation)
        
        return recommendations
    
    def _generate_explanation(self, recommendation, user_id):
        """Generate explanation for a recommendation"""
        content = recommendation['content']
        algorithm = recommendation.get('algorithm', 'unknown')
        
        explanations = []
        
        # Algorithm-based explanation
        if algorithm == 'collaborative_filtering':
            explanations.append("Users with similar taste also enjoyed this")
        elif algorithm == 'content_based_filtering':
            explanations.append("Similar to content you previously liked")
        elif algorithm == 'matrix_factorization':
            explanations.append("Predicted to match your preferences")
        elif algorithm == 'hybrid_system':
            explanations.append("Recommended by our advanced AI system")
        
        # Content-specific explanations
        try:
            genres = json.loads(content.genres or '[]')
            if genres:
                explanations.append(f"Features {', '.join(genres[:2])}")
        except:
            pass
        
        # Quality indicators
        if content.rating and content.rating >= 8.0:
            explanations.append("Highly rated by critics and users")
        
        if getattr(content, 'is_trending', False):
            explanations.append("Currently trending")
        
        return "; ".join(explanations[:3]) if explanations else "Recommended for you"
    
    def _generate_advanced_explanation(self, recommendation, user_preferences, context):
        """Generate advanced explanation with detailed reasoning"""
        content = recommendation['content']
        explanations = []
        
        # Preference matching
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            matching_genres = content_genres & set(user_preferences['top_genres'][:3])
            
            if matching_genres:
                explanations.append(f"Matches your interest in {', '.join(list(matching_genres)[:2])}")
        except:
            pass
        
        # Similar content explanation
        if recommendation.get('algorithm') == 'content_based_filtering':
            similar_content = self._find_similar_content_explanation(content, user_preferences)
            if similar_content:
                explanations.append(f"Similar to {similar_content}")
        
        # Collaborative explanation
        if recommendation.get('algorithm') == 'collaborative_filtering':
            explanations.append("People with similar preferences highly rated this")
        
        # Quality explanation
        quality_reasons = []
        if content.rating and content.rating >= 8.0:
            quality_reasons.append(f"highly rated ({content.rating:.1f}/10)")
        
        if content.vote_count and content.vote_count >= 1000:
            quality_reasons.append("popular among viewers")
        
        if quality_reasons:
            explanations.append(f"This content is {' and '.join(quality_reasons)}")
        
        # Novelty explanation
        if recommendation.get('novelty_score', 0) > 0.7:
            explanations.append("Something new for you to explore")
        
        # Context-based explanation
        if context:
            if context.get('time_of_day') == 'evening':
                explanations.append("Perfect for evening viewing")
            
            if context.get('device') == 'mobile':
                explanations.append("Great for mobile viewing")
        
        return "; ".join(explanations[:4]) if explanations else "Recommended based on your preferences"
    
    def _find_similar_content_explanation(self, content, user_preferences):
        """Find similar content that user has interacted with for explanation"""
        # This is a simplified version - in practice, you'd want to track which content
        # the recommendation is most similar to from the user's history
        if user_preferences['recent_content']:
            return user_preferences['recent_content'][0].get('title', 'content you enjoyed')
        return None
    
    def _calculate_confidence_scores(self, recommendations, user_id):
        """Calculate confidence scores for recommendations"""
        user_interaction_count = self.get_user_interaction_count(user_id)
        
        for rec in recommendations:
            confidence_factors = []
            
            # Algorithm confidence
            algorithm = rec.get('algorithm', 'unknown')
            if algorithm == 'collaborative_filtering':
                confidence_factors.append(min(user_interaction_count / 50.0, 1.0))
            elif algorithm == 'content_based_filtering':
                confidence_factors.append(0.8)  # Generally reliable
            elif algorithm == 'matrix_factorization':
                confidence_factors.append(min(user_interaction_count / 30.0, 0.9))
            elif algorithm == 'hybrid_system':
                confidence_factors.append(0.85)  # High confidence in hybrid
            
            # Score-based confidence
            score_confidence = min(rec['score'], 1.0)
            confidence_factors.append(score_confidence)
            
            # Content quality confidence
            content = rec['content']
            if content.rating and content.vote_count:
                quality_confidence = min((content.rating / 10.0) * (content.vote_count / 100.0), 1.0)
                confidence_factors.append(quality_confidence)
            
            # Calculate overall confidence
            rec['confidence'] = np.mean(confidence_factors) if confidence_factors else 0.5
        
        return recommendations
    
    def _add_quality_metrics(self, recommendations, user_id):
        """Add novelty and diversity metrics to recommendations"""
        for i, rec in enumerate(recommendations):
            # Novelty score (already calculated if novelty was applied)
            if 'novelty_score' not in rec:
                rec['novelty_score'] = self._calculate_simple_novelty(rec['content'], user_id)
            
            # Diversity contribution (how much this recommendation adds to diversity)
            if i > 0:
                diversity_contribution = self._calculate_recommendation_diversity(
                    rec, recommendations[:i]
                )
                rec['diversity_contribution'] = diversity_contribution
            else:
                rec['diversity_contribution'] = 1.0
            
            # Popularity percentile
            if rec['content'].popularity:
                max_popularity = self.db.session.query(
                    self.db.func.max(self.Content.popularity)
                ).scalar() or 1
                rec['popularity_percentile'] = rec['content'].popularity / max_popularity
            else:
                rec['popularity_percentile'] = 0.0
        
        return recommendations
    
    def _calculate_simple_novelty(self, content, user_id):
        """Calculate simple novelty score for content"""
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not user_interactions:
            return 1.0
        
        # Simple novelty based on genre overlap
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            user_genres = set()
            
            for interaction in user_interactions:
                interacted_content = self.Content.query.get(interaction.content_id)
                if interacted_content:
                    try:
                        genres = set(json.loads(interacted_content.genres or '[]'))
                        user_genres.update(genres)
                    except:
                        pass
            
            if user_genres:
                overlap = len(content_genres & user_genres)
                novelty = 1.0 - (overlap / max(len(content_genres), 1))
                return novelty
        except:
            pass
        
        return 0.5
    
    def _calculate_recommendation_diversity(self, candidate_rec, selected_recs):
        """Calculate how diverse a recommendation is compared to selected ones"""
        if not selected_recs:
            return 1.0
        
        diversity_scores = []
        
        for selected_rec in selected_recs:
            similarity = self._calculate_simple_content_similarity(
                candidate_rec['content'], selected_rec['content']
            )
            diversity = 1.0 - similarity
            diversity_scores.append(diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 1.0
    
    def _calculate_simple_content_similarity(self, content1, content2):
        """Calculate simple similarity between two content items"""
        similarity_factors = []
        
        # Genre similarity
        try:
            genres1 = set(json.loads(content1.genres or '[]'))
            genres2 = set(json.loads(content2.genres or '[]'))
            
            if genres1 or genres2:
                genre_overlap = len(genres1 & genres2) / max(len(genres1 | genres2), 1)
                similarity_factors.append(genre_overlap)
        except:
            pass
        
        # Content type similarity
        type_similarity = 1.0 if content1.content_type == content2.content_type else 0.0
        similarity_factors.append(type_similarity)
        
        # Language similarity
        try:
            langs1 = set(json.loads(content1.languages or '[]'))
            langs2 = set(json.loads(content2.languages or '[]'))
            
            if langs1 or langs2:
                lang_overlap = len(langs1 & langs2) / max(len(langs1 | langs2), 1)
                similarity_factors.append(lang_overlap)
        except:
            pass
        
        return np.mean(similarity_factors) if similarity_factors else 0.0
    
    def _analyze_user_preferences(self, user_interactions):
        """Analyze user preferences from interactions"""
        preferences = {
            'top_genres': [],
            'top_languages': [],
            'top_types': [],
            'avg_rating': 0.0,
            'recent_content': []
        }
        
        genre_counts = Counter()
        language_counts = Counter()
        type_counts = Counter()
        ratings = []
        
        # Analyze recent interactions (last 10)
        recent_interactions = sorted(user_interactions, key=lambda x: x.timestamp, reverse=True)[:10]
        
        for interaction in user_interactions:
            content = self.Content.query.get(interaction.content_id)
            if not content:
                continue
            
            weight = self._get_interaction_weight(interaction)
            
            # Genre analysis
            try:
                genres = json.loads(content.genres or '[]')
                for genre in genres:
                    genre_counts[genre] += weight
            except:
                pass
            
            # Language analysis
            try:
                languages = json.loads(content.languages or '[]')
                for language in languages:
                    language_counts[language] += weight
            except:
                pass
            
            # Content type analysis
            type_counts[content.content_type] += weight
            
            # Rating analysis
            if interaction.rating:
                ratings.append(interaction.rating)
            
            # Recent content for explanations
            if interaction in recent_interactions:
                preferences['recent_content'].append({
                    'title': content.title,
                    'genres': json.loads(content.genres or '[]') if content.genres else []
                })
        
        preferences['top_genres'] = [genre for genre, _ in genre_counts.most_common(5)]
        preferences['top_languages'] = [lang for lang, _ in language_counts.most_common(3)]
        preferences['top_types'] = [ctype for ctype, _ in type_counts.most_common(3)]
        
        if ratings:
            preferences['avg_rating'] = np.mean(ratings)
        
        return preferences
    
    def _get_interaction_weight(self, interaction):
        """Get weight for an interaction"""
        weights = {
            'view': 1.0,
            'like': 2.5,
            'favorite': 4.0,
            'watchlist': 3.0,
            'rating': interaction.rating * 0.4 if interaction.rating else 2.0,
            'search': 0.5,
            'share': 2.0
        }
        
        base_weight = weights.get(interaction.interaction_type, 1.0)
        
        # Apply temporal decay
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        temporal_factor = 0.95 ** (days_ago / 30.0)
        
        return base_weight * temporal_factor
    
    def _get_fallback_recommendations(self, user_id, limit, content_type):
        """Get fallback recommendations when main algorithms fail"""
        try:
            # Get popular content as fallback
            query = self.Content.query.order_by(
                self.Content.popularity.desc(),
                self.Content.rating.desc()
            )
            
            if content_type != 'all':
                query = query.filter_by(content_type=content_type)
            
            popular_content = query.limit(limit).all()
            
            fallback_recommendations = []
            for content in popular_content:
                fallback_recommendations.append({
                    'content': content,
                    'score': 0.5,
                    'algorithm': 'fallback_popular',
                    'reason': 'Popular content',
                    'confidence': 0.3
                })
            
            return fallback_recommendations
            
        except Exception as e:
            logger.error(f"Fallback recommendations error: {e}")
            return []
    
    # Caching and performance methods
    def _generate_cache_key(self, user_id, limit, content_type, strategy, context):
        """Generate cache key for recommendations"""
        context_str = json.dumps(context, sort_keys=True) if context else ""
        key_string = f"{user_id}:{limit}:{content_type}:{strategy}:{context_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key):
        """Get recommendations from cache"""
        with self._cache_lock:
            if cache_key in self.recommendation_cache:
                timestamp = self.cache_timestamps.get(cache_key, 0)
                if time.time() - timestamp < self.config['cache_ttl']:
                    return self.recommendation_cache[cache_key]
                else:
                    # Expired cache entry
                    del self.recommendation_cache[cache_key]
                    del self.cache_timestamps[cache_key]
        
        return None
    
    def _store_in_cache(self, cache_key, recommendations):
        """Store recommendations in cache"""
        with self._cache_lock:
            # Implement LRU cache with size limit
            if len(self.recommendation_cache) >= self.config['max_cache_size']:
                # Remove oldest entry
                oldest_key = min(self.cache_timestamps.items(), key=lambda x: x[1])[0]
                del self.recommendation_cache[oldest_key]
                del self.cache_timestamps[oldest_key]
            
            self.recommendation_cache[cache_key] = recommendations
            self.cache_timestamps[cache_key] = time.time()
    
    def _cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        with self._cache_lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self.cache_timestamps.items()
                if current_time - timestamp > self.config['cache_ttl']
            ]
            
            for key in expired_keys:
                del self.recommendation_cache[key]
                del self.cache_timestamps[key]
    
    def _record_performance_metric(self, metric_name, value):
        """Record performance metric"""
        if not self.config['performance_tracking']:
            return
        
        with self._metrics_lock:
            self.performance_metrics[metric_name].append(value)
            
            # Keep only last 1000 measurements
            if len(self.performance_metrics[metric_name]) > 1000:
                self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-1000:]
    
    # Public utility methods
    def get_user_interaction_count(self, user_id):
        """Get total interaction count for user"""
        return self.UserInteraction.query.filter_by(user_id=user_id).count()
    
    def _assess_user_profile_strength(self, user_id):
        """Assess user profile strength"""
        interaction_count = self.get_user_interaction_count(user_id)
        
        if interaction_count == 0:
            return 'new_user'
        elif interaction_count < 5:
            return 'weak'
        elif interaction_count < 20:
            return 'moderate'
        elif interaction_count < 50:
            return 'strong'
        else:
            return 'very_strong'
    
    def get_user_profile_strength(self, user_id):
        """Get user profile strength (public method)"""
        return self._assess_user_profile_strength(user_id)
    
    def update_user_profile(self, user_id):
        """Update user profile cache"""
        if user_id in self.user_profiles_cache:
            del self.user_profiles_cache[user_id]
        
        # Clear related caches
        with self._cache_lock:
            keys_to_remove = [
                key for key in self.recommendation_cache.keys()
                if key.startswith(f"{user_id}:")
            ]
            
            for key in keys_to_remove:
                del self.recommendation_cache[key]
                del self.cache_timestamps[key]
    
    def record_recommendation_feedback(self, user_id, content_id, feedback_type, 
                                     recommendation_id, feedback_value=1.0):
        """Record feedback on recommendations"""
        try:
            # Store feedback in database
            feedback_interaction = self.UserInteraction(
                user_id=user_id,
                content_id=content_id,
                interaction_type='recommendation_feedback',
                interaction_metadata={
                    'feedback_type': feedback_type,
                    'recommendation_id': recommendation_id,
                    'feedback_value': feedback_value,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            self.db.session.add(feedback_interaction)
            self.db.session.commit()
            
            # Update real-time learning if enabled
            if self.realtime_engine:
                self.realtime_engine.update_realtime_profile(user_id, {
                    'content_id': content_id,
                    'interaction_type': f'feedback_{feedback_type}',
                    'context': {
                        'feedback_value': feedback_value,
                        'recommendation_id': recommendation_id
                    }
                })
            
            # Clear user's cache
            self.update_user_profile(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording recommendation feedback: {e}")
            return False
    
    def get_recommendation_metrics(self, user_id):
        """Get recommendation quality metrics for user"""
        try:
            profile_strength = self._assess_user_profile_strength(user_id)
            interaction_count = self.get_user_interaction_count(user_id)
            
            # Calculate basic metrics
            metrics = {
                'user_profile_strength': profile_strength,
                'total_interactions': interaction_count,
                'recommendation_accuracy': 0.0,
                'diversity_score': 0.0,
                'novelty_score': 0.0,
                'coverage_score': 0.0
            }
            
            # Get recent recommendations for evaluation
            try:
                test_recommendations = self.get_personalized_recommendations(
                    user_id, limit=20, strategy='hybrid'
                )
                
                if test_recommendations:
                    # Calculate diversity
                    diversity_scores = []
                    for i in range(1, len(test_recommendations)):
                        diversity = self._calculate_recommendation_diversity(
                            test_recommendations[i], test_recommendations[:i]
                        )
                        diversity_scores.append(diversity)
                    
                    metrics['diversity_score'] = np.mean(diversity_scores) if diversity_scores else 0.0
                    
                    # Calculate average novelty
                    novelty_scores = [
                        rec.get('novelty_score', 0.5) for rec in test_recommendations
                    ]
                    metrics['novelty_score'] = np.mean(novelty_scores)
                    
                    # Calculate coverage (percentage of total content recommended)
                    total_content = self.Content.query.count()
                    unique_content = len(set(rec['content'].id for rec in test_recommendations))
                    metrics['coverage_score'] = unique_content / max(total_content, 1)
                    
                    # Calculate accuracy estimate based on confidence scores
                    confidence_scores = [
                        rec.get('confidence', 0.5) for rec in test_recommendations
                    ]
                    metrics['recommendation_accuracy'] = np.mean(confidence_scores)
                
            except Exception as e:
                logger.warning(f"Error calculating advanced metrics: {e}")
            
            # Add performance metrics if available
            if self.config['performance_tracking']:
                with self._metrics_lock:
                    if 'recommendation_generation' in self.performance_metrics:
                        avg_response_time = np.mean(self.performance_metrics['recommendation_generation'][-100:])
                        metrics['avg_response_time'] = avg_response_time
                    
                    cache_hits = len(self.performance_metrics.get('cache_hit', []))
                    total_requests = (cache_hits + 
                                    len(self.performance_metrics.get('recommendation_generation', [])))
                    
                    if total_requests > 0:
                        metrics['cache_hit_rate'] = cache_hits / total_requests
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating recommendation metrics: {e}")
            return {
                'user_profile_strength': 'unknown',
                'total_interactions': 0,
                'error': str(e)
            }

# Create alias for backward compatibility
RecommendationEngine = AdvancedRecommendationEngine