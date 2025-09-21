#backend/ml_services/recommendation.py
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import random
from collections import defaultdict, Counter
from .algorithm import (
    AdvancedUserProfiler, NeuralCollaborativeFiltering, 
    AdvancedContentAnalyzer, HybridRecommendationEngine, EvaluationMetrics
)

logger = logging.getLogger(__name__)

class PersonalizedRecommendationEngine:
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
        # Initialize advanced components
        self.user_profiler = AdvancedUserProfiler(db, models)
        self.neural_cf = NeuralCollaborativeFiltering(db, models)
        self.content_analyzer = AdvancedContentAnalyzer(db, models)
        self.hybrid_engine = HybridRecommendationEngine(db, models)
        
        # Advanced tracking
        self.interaction_weights = {
            'search': 1.0,
            'view': 2.0,
            'like': 4.0,
            'favorite': 5.0,
            'watchlist': 4.5,
            'rating': 0.0,  # Calculated dynamically
            'share': 3.0,
            'download': 4.0,
            'click': 0.5
        }
        
        # Content type specific weights
        self.content_type_weights = {
            'movie': 1.0,
            'tv': 1.0,
            'anime': 1.0
        }
        
        # Train neural models periodically
        self.last_model_update = None
        self.model_update_frequency = timedelta(hours=6)
        
    def get_ultra_personalized_recommendations(self, user_id, content_type='all', limit=20, 
                                             strategy='advanced_hybrid', diversity_factor=0.3):
        """
        Get ultra-personalized recommendations with 100% accuracy targeting
        """
        try:
            # Update neural models if needed
            self._update_models_if_needed()
            
            # Build comprehensive user profile
            user_profile = self.user_profiler.build_comprehensive_user_profile(user_id)
            
            # Check user interaction count for strategy selection
            interaction_count = self.get_user_interaction_count(user_id)
            
            if interaction_count < 3:
                # Cold start - use advanced profile-based recommendations
                return self._get_cold_start_ultra_recommendations(user_id, content_type, limit)
            elif interaction_count < 10:
                # Moderate data - use hybrid with exploration boost
                return self._get_moderate_data_recommendations(user_id, content_type, limit, user_profile)
            else:
                # Rich data - use full advanced hybrid system
                return self._get_rich_data_recommendations(user_id, content_type, limit, user_profile, diversity_factor)
                
        except Exception as e:
            logger.error(f"Error in ultra personalized recommendations: {e}")
            return self._get_fallback_recommendations(user_id, content_type, limit)
    
    def _get_rich_data_recommendations(self, user_id, content_type, limit, user_profile, diversity_factor):
        """Get recommendations for users with rich interaction history"""
        try:
            # Multi-stage recommendation pipeline
            
            # Stage 1: Get base recommendations from hybrid engine
            base_recommendations = self.hybrid_engine.get_personalized_recommendations(
                user_id, content_type, limit * 3
            )
            
            # Stage 2: Apply advanced filtering and scoring
            enhanced_recommendations = []
            
            for rec in base_recommendations:
                content = rec['content']
                base_score = rec['score']
                
                # Calculate advanced personal score
                personal_score = self._calculate_ultra_personal_score(
                    user_id, content.id, user_profile, base_score
                )
                
                # Calculate novelty and diversity scores
                novelty_score = self._calculate_novelty_score(user_id, content.id, user_profile)
                diversity_score = self._calculate_diversity_contribution(content.id, enhanced_recommendations)
                
                # Calculate temporal relevance
                temporal_score = self._calculate_temporal_relevance(content, user_profile)
                
                # Calculate quality-preference match
                quality_match = self._calculate_quality_preference_match(content, user_profile)
                
                # Combine all scores with advanced weighting
                final_score = (
                    personal_score * 0.40 +
                    novelty_score * 0.20 +
                    diversity_score * diversity_factor +
                    temporal_score * 0.15 +
                    quality_match * 0.15 +
                    base_score * 0.10
                )
                
                enhanced_recommendations.append({
                    'content': content,
                    'score': final_score,
                    'personal_score': personal_score,
                    'novelty_score': novelty_score,
                    'diversity_score': diversity_score,
                    'temporal_score': temporal_score,
                    'quality_match': quality_match,
                    'explanation': self._generate_ultra_explanation(user_id, content.id, user_profile),
                    'confidence': self._calculate_confidence_score(personal_score, novelty_score, quality_match),
                    'algorithm_breakdown': rec.get('algorithm_breakdown', {}),
                    'recommendation_reasons': self._get_detailed_reasons(user_id, content.id, user_profile)
                })
            
            # Stage 3: Apply diversity optimization
            if diversity_factor > 0:
                enhanced_recommendations = self._optimize_diversity_advanced(
                    enhanced_recommendations, diversity_factor, limit
                )
            
            # Stage 4: Final ranking and selection
            enhanced_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return enhanced_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in rich data recommendations: {e}")
            return []
    
    def _get_moderate_data_recommendations(self, user_id, content_type, limit, user_profile):
        """Get recommendations for users with moderate interaction history"""
        try:
            recommendations = defaultdict(float)
            algorithm_contributions = defaultdict(lambda: defaultdict(float))
            
            # 1. Content-based recommendations (higher weight)
            content_recs = self._get_enhanced_content_based_recommendations(
                user_id, user_profile, limit * 2
            )
            for content_id, score in content_recs:
                weight = 0.35
                recommendations[content_id] += score * weight
                algorithm_contributions[content_id]['content_based'] = score * weight
            
            # 2. Profile matching recommendations
            profile_recs = self._get_profile_matching_recommendations(
                user_id, user_profile, limit * 2
            )
            for content_id, score in profile_recs:
                weight = 0.30
                recommendations[content_id] += score * weight
                algorithm_contributions[content_id]['profile_matching'] = score * weight
            
            # 3. Popularity with preference boost
            popularity_recs = self._get_popularity_with_preference_boost(
                user_id, user_profile, limit * 2
            )
            for content_id, score in popularity_recs:
                weight = 0.25
                recommendations[content_id] += score * weight
                algorithm_contributions[content_id]['popularity_boosted'] = score * weight
            
            # 4. Exploration recommendations
            exploration_recs = self._get_exploration_recommendations(
                user_id, user_profile, limit
            )
            for content_id, score in exploration_recs:
                weight = 0.10
                recommendations[content_id] += score * weight
                algorithm_contributions[content_id]['exploration'] = score * weight
            
            # Filter by content type
            if content_type != 'all':
                filtered_recommendations = {}
                for content_id, score in recommendations.items():
                    content = self.Content.query.get(content_id)
                    if content and content.content_type == content_type:
                        filtered_recommendations[content_id] = score
                recommendations = filtered_recommendations
            
            # Build detailed response
            detailed_recommendations = []
            sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            
            for content_id, final_score in sorted_recommendations[:limit]:
                content = self.Content.query.get(content_id)
                if content:
                    detailed_recommendations.append({
                        'content': content,
                        'score': final_score,
                        'algorithm_breakdown': dict(algorithm_contributions[content_id]),
                        'explanation': self._generate_moderate_explanation(user_id, content_id, user_profile),
                        'confidence': min(final_score / max(recommendations.values()), 1.0),
                        'recommendation_reasons': self._get_moderate_reasons(user_id, content_id, user_profile)
                    })
            
            return detailed_recommendations
            
        except Exception as e:
            logger.error(f"Error in moderate data recommendations: {e}")
            return []
    
    def _get_cold_start_ultra_recommendations(self, user_id, content_type, limit):
        """Get recommendations for new users with minimal data"""
        try:
            user = self.User.query.get(user_id)
            recommendations = []
            
            # Use user preferences if available
            preferred_genres = []
            preferred_languages = []
            
            if user:
                try:
                    preferred_genres = json.loads(user.preferred_genres or '[]')
                    preferred_languages = json.loads(user.preferred_languages or '[]')
                except:
                    pass
            
            # Strategy 1: Use explicit preferences
            if preferred_genres or preferred_languages:
                preference_recs = self._get_preference_based_recommendations(
                    preferred_genres, preferred_languages, content_type, limit
                )
                recommendations.extend(preference_recs)
            
            # Strategy 2: High-quality popular content
            popular_quality_recs = self._get_popular_quality_recommendations(content_type, limit)
            recommendations.extend(popular_quality_recs)
            
            # Strategy 3: Trending content with regional preference
            trending_recs = self._get_trending_with_regional_preference(content_type, limit)
            recommendations.extend(trending_recs)
            
            # Strategy 4: Diverse genre sampling
            diverse_recs = self._get_diverse_genre_sampling(content_type, limit)
            recommendations.extend(diverse_recs)
            
            # Remove duplicates and rank
            seen_content = set()
            unique_recommendations = []
            
            for rec in recommendations:
                content_id = rec['content'].id
                if content_id not in seen_content:
                    seen_content.add(content_id)
                    unique_recommendations.append(rec)
            
            # Sort by score and return top recommendations
            unique_recommendations.sort(key=lambda x: x['score'], reverse=True)
            return unique_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in cold start recommendations: {e}")
            return []
    
    def _calculate_ultra_personal_score(self, user_id, content_id, user_profile, base_score):
        """Calculate ultra-personalized score"""
        try:
            content = self.Content.query.get(content_id)
            if not content:
                return base_score
            
            personal_score = base_score
            
            # Genre affinity scoring
            try:
                content_genres = set(json.loads(content.genres or '[]'))
                genre_score = 0.0
                total_genre_weight = sum(user_profile['genre_preferences'].values())
                
                for genre in content_genres:
                    if genre in user_profile['genre_preferences']:
                        genre_weight = user_profile['genre_preferences'][genre]
                        genre_score += genre_weight / max(total_genre_weight, 1.0)
                
                personal_score += genre_score * 2.0
                
            except:
                pass
            
            # Language affinity scoring
            try:
                content_languages = set(json.loads(content.languages or '[]'))
                language_score = 0.0
                total_lang_weight = sum(user_profile['language_preferences'].values())
                
                for language in content_languages:
                    if language in user_profile['language_preferences']:
                        lang_weight = user_profile['language_preferences'][language]
                        language_score += lang_weight / max(total_lang_weight, 1.0)
                
                personal_score += language_score * 1.5
                
            except:
                pass
            
            # Content type preference
            if content.content_type in user_profile['content_type_preferences']:
                type_preference = user_profile['content_type_preferences'][content.content_type]
                total_type_weight = sum(user_profile['content_type_preferences'].values())
                normalized_type_preference = type_preference / max(total_type_weight, 1.0)
                personal_score += normalized_type_preference * 1.0
            
            # Quality preference matching
            if content.rating and user_profile['quality_preference'] > 0:
                expected_quality = user_profile['quality_preference']
                quality_diff = abs(content.rating - expected_quality) / 10.0
                quality_match = 1.0 - quality_diff
                personal_score += quality_match * 0.5
            
            # Temporal preference (decade matching)
            if content.release_date:
                content_decade = (content.release_date.year // 10) * 10
                if content_decade in user_profile['preferred_decades']:
                    decade_preference = user_profile['preferred_decades'][content_decade]
                    total_decade_weight = sum(user_profile['preferred_decades'].values())
                    normalized_decade_preference = decade_preference / max(total_decade_weight, 1.0)
                    personal_score += normalized_decade_preference * 0.3
            
            # Runtime preference matching
            if content.runtime:
                runtime_category = self._categorize_runtime(content.runtime)
                if runtime_category in user_profile['runtime_preferences']:
                    runtime_preference = user_profile['runtime_preferences'][runtime_category]
                    total_runtime_weight = sum(user_profile['runtime_preferences'].values())
                    normalized_runtime_preference = runtime_preference / max(total_runtime_weight, 1.0)
                    personal_score += normalized_runtime_preference * 0.2
            
            return personal_score
            
        except Exception as e:
            logger.warning(f"Error calculating ultra personal score: {e}")
            return base_score
    
    def _calculate_novelty_score(self, user_id, content_id, user_profile):
        """Calculate novelty score for exploration"""
        try:
            content = self.Content.query.get(content_id)
            if not content:
                return 0.0
            
            novelty_score = 0.0
            
            # Genre novelty
            try:
                content_genres = set(json.loads(content.genres or '[]'))
                user_genres = set(user_profile['genre_preferences'].keys())
                
                new_genres = content_genres - user_genres
                genre_novelty = len(new_genres) / max(len(content_genres), 1)
                novelty_score += genre_novelty * 0.4
                
            except:
                pass
            
            # Language novelty
            try:
                content_languages = set(json.loads(content.languages or '[]'))
                user_languages = set(user_profile['language_preferences'].keys())
                
                new_languages = content_languages - user_languages
                language_novelty = len(new_languages) / max(len(content_languages), 1)
                novelty_score += language_novelty * 0.3
                
            except:
                pass
            
            # Content type novelty
            if content.content_type not in user_profile['content_type_preferences']:
                novelty_score += 0.3
            
            return min(novelty_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating novelty score: {e}")
            return 0.0
    
    def _calculate_diversity_contribution(self, content_id, existing_recommendations):
        """Calculate how much this content contributes to diversity"""
        try:
            if not existing_recommendations:
                return 1.0
            
            content = self.Content.query.get(content_id)
            if not content:
                return 0.0
            
            try:
                content_genres = set(json.loads(content.genres or '[]'))
            except:
                content_genres = set()
            
            # Calculate genre overlap with existing recommendations
            existing_genres = set()
            for rec in existing_recommendations:
                try:
                    rec_genres = set(json.loads(rec['content'].genres or '[]'))
                    existing_genres.update(rec_genres)
                except:
                    pass
            
            if not existing_genres:
                return 1.0
            
            # Calculate diversity contribution
            new_genres = content_genres - existing_genres
            diversity_contribution = len(new_genres) / max(len(content_genres), 1)
            
            return diversity_contribution
            
        except Exception as e:
            logger.warning(f"Error calculating diversity contribution: {e}")
            return 0.5
    
    def _calculate_temporal_relevance(self, content, user_profile):
        """Calculate temporal relevance based on user patterns"""
        try:
            relevance_score = 0.5  # Base relevance
            
            # Recency preference
            if content.release_date:
                years_old = (datetime.now().date() - content.release_date).days / 365.25
                
                # Check if user prefers recent content
                recent_preference = 0.0
                for decade, weight in user_profile['preferred_decades'].items():
                    if decade >= 2010:  # Recent decades
                        recent_preference += weight
                
                total_decade_weight = sum(user_profile['preferred_decades'].values())
                normalized_recent_preference = recent_preference / max(total_decade_weight, 1.0)
                
                if normalized_recent_preference > 0.5:  # User prefers recent content
                    relevance_score += max(0, (1.0 - years_old / 10.0)) * 0.5
                else:  # User is open to older content
                    relevance_score += 0.3
            
            # Trending boost
            if hasattr(content, 'is_trending') and content.is_trending:
                relevance_score += 0.2
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating temporal relevance: {e}")
            return 0.5
    
    def _calculate_quality_preference_match(self, content, user_profile):
        """Calculate how well content quality matches user preferences"""
        try:
            if not content.rating or user_profile['quality_preference'] <= 0:
                return 0.5
            
            expected_quality = user_profile['quality_preference']
            quality_diff = abs(content.rating - expected_quality) / 10.0
            quality_match = 1.0 - quality_diff
            
            # Boost for high-quality content if user appreciates quality
            if expected_quality > 7.0 and content.rating >= 8.0:
                quality_match += 0.2
            
            return max(0.0, min(1.0, quality_match))
            
        except Exception as e:
            logger.warning(f"Error calculating quality preference match: {e}")
            return 0.5
    
    def _generate_ultra_explanation(self, user_id, content_id, user_profile):
        """Generate detailed explanation for recommendation"""
        try:
            content = self.Content.query.get(content_id)
            if not content:
                return "Recommended for you"
            
            explanations = []
            
            # Genre-based explanation
            try:
                content_genres = set(json.loads(content.genres or '[]'))
                top_user_genres = sorted(
                    user_profile['genre_preferences'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                
                matching_genres = content_genres & set([g[0] for g in top_user_genres])
                if matching_genres:
                    explanations.append(f"Perfect match for your love of {', '.join(list(matching_genres)[:2])}")
            except:
                pass
            
            # Quality explanation
            if content.rating and content.rating >= 8.5:
                explanations.append("Exceptional quality that matches your standards")
            elif content.rating and content.rating >= 7.5:
                explanations.append("High-quality content you'll appreciate")
            
            # Popularity explanation
            if content.popularity and content.popularity > 1000:
                explanations.append("Loved by users with similar tastes")
            
            # Novelty explanation
            novelty_score = self._calculate_novelty_score(user_id, content_id, user_profile)
            if novelty_score > 0.7:
                explanations.append("Something new to expand your horizons")
            elif novelty_score > 0.4:
                explanations.append("A fresh take on genres you enjoy")
            
            # Temporal explanation
            if content.release_date and (datetime.now().year - content.release_date.year) <= 2:
                explanations.append("Recent release that's trending")
            
            if not explanations:
                return "Carefully selected based on your unique preferences"
            
            return "; ".join(explanations[:3])
            
        except Exception as e:
            logger.warning(f"Error generating ultra explanation: {e}")
            return "Personalized recommendation"
    
    def _get_detailed_reasons(self, user_id, content_id, user_profile):
        """Get detailed reasons for recommendation"""
        try:
            content = self.Content.query.get(content_id)
            if not content:
                return []
            
            reasons = []
            
            # Analyze user's similar content interactions
            similar_content = self._find_similar_watched_content(user_id, content_id)
            if similar_content:
                reasons.append({
                    'type': 'similar_content',
                    'explanation': f"Similar to {similar_content[0].title} which you enjoyed",
                    'confidence': 0.8
                })
            
            # Analyze genre preferences
            try:
                content_genres = set(json.loads(content.genres or '[]'))
                top_genres = sorted(
                    user_profile['genre_preferences'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                for genre, weight in top_genres:
                    if genre in content_genres:
                        reasons.append({
                            'type': 'genre_preference',
                            'explanation': f"You consistently enjoy {genre} content",
                            'confidence': weight
                        })
                        break
            except:
                pass
            
            # Quality reasoning
            if content.rating and content.rating >= 8.0:
                reasons.append({
                    'type': 'quality_assurance',
                    'explanation': f"High rating of {content.rating}/10 ensures quality",
                    'confidence': content.rating / 10.0
                })
            
            return reasons[:5]
            
        except Exception as e:
            logger.warning(f"Error getting detailed reasons: {e}")
            return []
    
    def _find_similar_watched_content(self, user_id, content_id):
        """Find similar content that user has watched"""
        try:
            user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            similar_content = []
            
            for interaction in user_interactions[:20]:  # Limit for performance
                if interaction.interaction_type in ['like', 'favorite', 'rating']:
                    if interaction.rating and interaction.rating >= 4.0:
                        similarity = self.content_analyzer.calculate_advanced_similarity(
                            content_id, interaction.content_id
                        )
                        if similarity > 0.6:
                            content = self.Content.query.get(interaction.content_id)
                            if content:
                                similar_content.append(content)
            
            return similar_content[:3]
            
        except Exception as e:
            logger.warning(f"Error finding similar watched content: {e}")
            return []
    
    def _update_models_if_needed(self):
        """Update neural models if needed"""
        try:
            current_time = datetime.utcnow()
            
            if (self.last_model_update is None or 
                current_time - self.last_model_update > self.model_update_frequency):
                
                logger.info("Updating neural collaborative filtering model...")
                self.neural_cf.train_neural_model()
                self.last_model_update = current_time
                logger.info("Neural model updated successfully")
                
        except Exception as e:
            logger.warning(f"Failed to update neural models: {e}")
    
    def get_user_interaction_count(self, user_id):
        """Get total interaction count for user"""
        return self.UserInteraction.query.filter_by(user_id=user_id).count()
    
    def record_interaction_with_advanced_tracking(self, user_id, content_id, interaction_type, 
                                                metadata=None, rating=None):
        """Record interaction with advanced tracking"""
        try:
            # Calculate interaction weight
            weight = self.interaction_weights.get(interaction_type, 1.0)
            if rating:
                weight = max(weight, rating * 0.8)
            
            # Create interaction record
            interaction = self.UserInteraction(
                user_id=user_id,
                content_id=content_id,
                interaction_type=interaction_type,
                rating=rating,
                interaction_metadata=metadata or {}
            )
            
            self.db.session.add(interaction)
            self.db.session.commit()
            
            # Update user profile cache
            if user_id in self.user_profiler.user_profiles:
                del self.user_profiler.user_profiles[user_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
            self.db.session.rollback()
            return False
    
    def _categorize_runtime(self, runtime):
        """Categorize content by runtime"""
        if runtime < 90:
            return 'short'
        elif runtime < 150:
            return 'medium'
        else:
            return 'long'
    
    def _get_fallback_recommendations(self, user_id, content_type, limit):
        """Get fallback recommendations when main algorithms fail"""
        try:
            query = self.Content.query
            
            if content_type != 'all':
                query = query.filter_by(content_type=content_type)
            
            popular_content = query.filter(
                self.Content.rating >= 7.0,
                self.Content.popularity > 100
            ).order_by(
                self.Content.rating.desc(),
                self.Content.popularity.desc()
            ).limit(limit).all()
            
            recommendations = []
            for content in popular_content:
                recommendations.append({
                    'content': content,
                    'score': (content.rating or 0) / 10.0,
                    'explanation': 'Popular high-quality content',
                    'confidence': 0.5,
                    'algorithm_breakdown': {'fallback': 1.0}
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in fallback recommendations: {e}")
            return []