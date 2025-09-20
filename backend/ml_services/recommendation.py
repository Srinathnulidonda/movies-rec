#backend/ml_services/recommendation.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import random
import math
from collections import defaultdict, Counter, deque
from .algorithm import (
    AdvancedCollaborativeFiltering, DeepContentAnalyzer, NeuralRecommendationEngine,
    AdvancedDiversityOptimizer, BehavioralPatternAnalyzer
)

logger = logging.getLogger(__name__)

class AdvancedRecommendationEngine:
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
        # Initialize advanced components
        self.collaborative_filtering = AdvancedCollaborativeFiltering(db, models)
        self.content_analyzer = DeepContentAnalyzer(db, models)
        self.neural_engine = NeuralRecommendationEngine(db, models)
        self.diversity_optimizer = AdvancedDiversityOptimizer(db, models)
        self.behavior_analyzer = BehavioralPatternAnalyzer(db, models)
        
        # User profiles and caches
        self.user_profiles = {}
        self.user_behavior_patterns = {}
        self.content_similarity_cache = {}
        self.recommendation_cache = {}
        
        # Advanced parameters
        self.algorithm_weights = {
            'collaborative_advanced': 0.30,
            'content_deep': 0.25,
            'neural_embedding': 0.25,
            'behavioral_pattern': 0.20
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'minimum_rating': 5.5,
            'minimum_vote_count': 10,
            'recency_bonus_days': 365
        }
        
    def initialize_user_profile(self, user_id, initial_preferences=None):
        """Initialize comprehensive user profile"""
        try:
            user = self.User.query.get(user_id)
            if not user:
                return False
            
            profile = {
                'user_id': user_id,
                'initialized_at': datetime.utcnow(),
                'preferences': initial_preferences or {},
                'behavioral_patterns': self.behavior_analyzer.analyze_user_behavioral_patterns(user_id),
                'interaction_count': 0,
                'quality_preference': 7.0,
                'diversity_preference': 0.5,
                'novelty_preference': 0.3,
                'last_updated': datetime.utcnow()
            }
            
            self.user_profiles[user_id] = profile
            return True
            
        except Exception as e:
            logger.error(f"Error initializing user profile for {user_id}: {e}")
            return False
    
    def process_real_time_interaction(self, user_id, content_id, interaction_type, rating=None, metadata=None):
        """Process interaction in real-time and update user profile"""
        try:
            # Update user profile
            if user_id not in self.user_profiles:
                self.initialize_user_profile(user_id)
            
            profile = self.user_profiles[user_id]
            profile['interaction_count'] += 1
            profile['last_updated'] = datetime.utcnow()
            
            # Analyze interaction for profile updates
            self._update_profile_from_interaction(user_id, content_id, interaction_type, rating, metadata)
            
            # Clear relevant caches
            self._clear_user_caches(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing real-time interaction: {e}")
            return False
    
    def _update_profile_from_interaction(self, user_id, content_id, interaction_type, rating, metadata):
        """Update user profile based on new interaction"""
        profile = self.user_profiles[user_id]
        content = self.Content.query.get(content_id)
        
        if not content:
            return
        
        # Update quality preferences
        if rating and rating > 0:
            current_quality = profile.get('quality_preference', 7.0)
            profile['quality_preference'] = (current_quality * 0.9) + (rating * 0.1)
        elif content.rating:
            if interaction_type in ['favorite', 'watchlist', 'like']:
                current_quality = profile.get('quality_preference', 7.0)
                profile['quality_preference'] = (current_quality * 0.95) + (content.rating * 0.05)
        
        # Update content type preferences
        if 'content_type_preferences' not in profile:
            profile['content_type_preferences'] = defaultdict(float)
        
        interaction_weight = self._get_profile_update_weight(interaction_type, metadata)
        profile['content_type_preferences'][content.content_type] += interaction_weight
        
        # Update genre preferences
        try:
            genres = json.loads(content.genres or '[]')
            if 'genre_preferences' not in profile:
                profile['genre_preferences'] = defaultdict(float)
            
            for genre in genres:
                profile['genre_preferences'][genre] += interaction_weight
        except:
            pass
        
        # Update language preferences
        try:
            languages = json.loads(content.languages or '[]')
            if 'language_preferences' not in profile:
                profile['language_preferences'] = defaultdict(float)
            
            for language in languages:
                profile['language_preferences'][language] += interaction_weight
        except:
            pass
    
    def _get_profile_update_weight(self, interaction_type, metadata):
        """Get weight for profile updates"""
        base_weights = {
            'favorite': 5.0, 'watchlist': 4.0, 'like': 3.0, 'rating': 2.5,
            'view': 1.0, 'search_click': 0.5, 'share': 2.0
        }
        
        weight = base_weights.get(interaction_type, 1.0)
        
        # Adjust based on metadata
        if metadata:
            engagement_score = metadata.get('engagement_score', 1.0)
            view_percentage = metadata.get('view_percentage', 50) / 100.0
            weight *= engagement_score * (0.5 + 0.5 * view_percentage)
        
        return weight
    
    def get_ultra_personalized_recommendations(self, user_id, limit=20, content_type='all', 
                                             diversity_factor=0.4, novelty_factor=0.3,
                                             include_explanations=True, time_context='any',
                                             mood_context='neutral'):
        """Get ultra-personalized recommendations using all advanced techniques"""
        try:
            # Ensure user profile exists
            if user_id not in self.user_profiles:
                self.initialize_user_profile(user_id)
            
            # Get base recommendations from multiple algorithms
            recommendations = self._generate_multi_algorithm_recommendations(
                user_id, limit * 3, content_type
            )
            
            if not recommendations:
                return self._get_fallback_recommendations(user_id, limit, content_type)
            
            # Apply contextual filtering
            recommendations = self._apply_contextual_filtering(
                recommendations, time_context, mood_context
            )
            
            # Apply novelty filtering
            if novelty_factor > 0:
                recommendations = self._apply_novelty_filtering(
                    user_id, recommendations, novelty_factor
                )
            
            # Apply diversity optimization
            if diversity_factor > 0:
                rec_tuples = [(rec['content_id'], rec['score']) for rec in recommendations]
                diverse_tuples = self.diversity_optimizer.optimize_recommendations_for_diversity(
                    rec_tuples, diversity_factor, limit
                )
                diverse_content_ids = [content_id for content_id, _ in diverse_tuples]
                recommendations = [rec for rec in recommendations if rec['content_id'] in diverse_content_ids]
            
            # Sort by final score and limit
            recommendations.sort(key=lambda x: x['final_score'], reverse=True)
            recommendations = recommendations[:limit]
            
            # Generate comprehensive results
            results = []
            for rec in recommendations:
                content = self.Content.query.get(rec['content_id'])
                if content:
                    result = {
                        'content': content,
                        'personalization_score': rec['final_score'],
                        'confidence_score': rec.get('confidence', 0.8),
                        'novelty_score': rec.get('novelty_score', 0.5),
                        'diversity_contribution': rec.get('diversity_contribution', 0.0),
                        'explanation': self._generate_detailed_explanation(user_id, content.id, rec) if include_explanations else '',
                        'matching_factors': rec.get('matching_factors', []),
                        'predicted_rating': self._predict_user_rating(user_id, content.id),
                        'recommendation_strength': self._calculate_recommendation_strength(rec),
                        'algorithm_breakdown': rec.get('algorithm_contributions', {}),
                        'behavioral_match': rec.get('behavioral_match', 0.5),
                        'temporal_relevance': rec.get('temporal_relevance', 0.5)
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ultra-personalized recommendations: {e}")
            return []
    
    def _generate_multi_algorithm_recommendations(self, user_id, limit, content_type):
        """Generate recommendations using multiple algorithms"""
        all_recommendations = {}
        
        # 1. Advanced Collaborative Filtering
        try:
            collab_recs = self.collaborative_filtering.get_user_based_recommendations_advanced(
                user_id, limit
            )
            for content_id, score in collab_recs:
                if content_id not in all_recommendations:
                    all_recommendations[content_id] = {
                        'content_id': content_id,
                        'scores': {},
                        'algorithm_contributions': {}
                    }
                all_recommendations[content_id]['scores']['collaborative'] = score
                all_recommendations[content_id]['algorithm_contributions']['collaborative'] = score * self.algorithm_weights['collaborative_advanced']
        except Exception as e:
            logger.warning(f"Collaborative filtering failed: {e}")
        
        # 2. Deep Content Analysis
        try:
            content_recs = self._get_content_based_recommendations_advanced(user_id, limit)
            for content_id, score in content_recs:
                if content_id not in all_recommendations:
                    all_recommendations[content_id] = {
                        'content_id': content_id,
                        'scores': {},
                        'algorithm_contributions': {}
                    }
                all_recommendations[content_id]['scores']['content_deep'] = score
                all_recommendations[content_id]['algorithm_contributions']['content_deep'] = score * self.algorithm_weights['content_deep']
        except Exception as e:
            logger.warning(f"Deep content analysis failed: {e}")
        
        # 3. Neural Embedding Recommendations
        try:
            neural_recs = self._get_neural_recommendations(user_id, limit)
            for content_id, score in neural_recs:
                if content_id not in all_recommendations:
                    all_recommendations[content_id] = {
                        'content_id': content_id,
                        'scores': {},
                        'algorithm_contributions': {}
                    }
                all_recommendations[content_id]['scores']['neural'] = score
                all_recommendations[content_id]['algorithm_contributions']['neural'] = score * self.algorithm_weights['neural_embedding']
        except Exception as e:
            logger.warning(f"Neural recommendations failed: {e}")
        
        # 4. Behavioral Pattern Matching
        try:
            behavioral_recs = self._get_behavioral_pattern_recommendations(user_id, limit)
            for content_id, score in behavioral_recs:
                if content_id not in all_recommendations:
                    all_recommendations[content_id] = {
                        'content_id': content_id,
                        'scores': {},
                        'algorithm_contributions': {}
                    }
                all_recommendations[content_id]['scores']['behavioral'] = score
                all_recommendations[content_id]['algorithm_contributions']['behavioral'] = score * self.algorithm_weights['behavioral_pattern']
        except Exception as e:
            logger.warning(f"Behavioral pattern matching failed: {e}")
        
        # Combine scores and calculate final scores
        final_recommendations = []
        for content_id, rec_data in all_recommendations.items():
            # Calculate weighted final score
            final_score = sum(rec_data['algorithm_contributions'].values())
            
            # Apply quality filtering
            content = self.Content.query.get(content_id)
            if content and self._passes_quality_threshold(content):
                # Apply personalization boost
                personalization_boost = self._calculate_personalization_boost(user_id, content)
                final_score *= (1.0 + personalization_boost)
                
                rec_data['final_score'] = final_score
                rec_data['confidence'] = self._calculate_confidence_score(rec_data)
                rec_data['matching_factors'] = self._identify_matching_factors(user_id, content)
                
                final_recommendations.append(rec_data)
        
        return final_recommendations
    
    def _get_content_based_recommendations_advanced(self, user_id, limit):
        """Get advanced content-based recommendations"""
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not user_interactions:
            return []
        
        # Get user's preferred content
        preferred_content = []
        for interaction in user_interactions:
            if interaction.content_id and interaction.interaction_type in ['favorite', 'like', 'watchlist', 'rating']:
                weight = self._get_interaction_weight_for_content_analysis(interaction)
                if weight >= 2.0:
                    preferred_content.append((interaction.content_id, weight))
        
        if not preferred_content:
            return []
        
        # Find similar content
        recommendations = defaultdict(float)
        user_seen_content = set(interaction.content_id for interaction in user_interactions if interaction.content_id)
        
        all_content = self.Content.query.limit(5000).all()  # Limit for performance
        
        for content in all_content:
            if content.id in user_seen_content:
                continue
            
            content_score = 0.0
            total_weight = 0.0
            
            for preferred_content_id, user_weight in preferred_content:
                similarity = self.content_analyzer.calculate_advanced_content_similarity(
                    content.id, preferred_content_id
                )
                content_score += similarity * user_weight
                total_weight += user_weight
            
            if total_weight > 0:
                recommendations[content.id] = content_score / total_weight
        
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:limit]
    
    def _get_neural_recommendations(self, user_id, limit):
        """Get neural network-based recommendations"""
        user_seen_content = set()
        for interaction in self.UserInteraction.query.filter_by(user_id=user_id).all():
            if interaction.content_id:
                user_seen_content.add(interaction.content_id)
        
        recommendations = []
        all_content = self.Content.query.limit(3000).all()  # Limit for performance
        
        for content in all_content:
            if content.id not in user_seen_content:
                affinity_score = self.neural_engine.predict_user_item_affinity(user_id, content.id)
                if affinity_score > 0.1:  # Threshold for relevance
                    recommendations.append((content.id, affinity_score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]
    
    def _get_behavioral_pattern_recommendations(self, user_id, limit):
        """Get recommendations based on behavioral patterns"""
        if user_id not in self.user_behavior_patterns:
            self.user_behavior_patterns[user_id] = self.behavior_analyzer.analyze_user_behavioral_patterns(user_id)
        
        patterns = self.user_behavior_patterns[user_id]
        
        # Get content based on behavioral patterns
        recommendations = []
        
        # Content type preferences
        preferred_types = [ptype for ptype, _ in patterns['content_preferences']['preferred_types']]
        preferred_genres = [genre for genre, _ in patterns['content_preferences']['preferred_genres']]
        
        if preferred_types and preferred_genres:
            for content_type in preferred_types[:2]:
                for genre in preferred_genres[:3]:
                    content_items = self.Content.query.filter(
                        self.Content.content_type == content_type,
                        self.Content.genres.contains(genre)
                    ).order_by(self.Content.rating.desc()).limit(10).all()
                    
                    for content in content_items:
                        # Calculate behavioral match score
                        match_score = self._calculate_behavioral_match_score(user_id, content, patterns)
                        recommendations.append((content.id, match_score))
        
        # Remove duplicates and sort
        seen_ids = set()
        unique_recommendations = []
        for content_id, score in recommendations:
            if content_id not in seen_ids:
                seen_ids.add(content_id)
                unique_recommendations.append((content_id, score))
        
        unique_recommendations.sort(key=lambda x: x[1], reverse=True)
        return unique_recommendations[:limit]
    
    def _calculate_behavioral_match_score(self, user_id, content, patterns):
        """Calculate how well content matches user's behavioral patterns"""
        score = 0.0
        
        # Quality preference match
        quality_pref = patterns['quality_preferences']['avg_preferred_quality']
        if content.rating:
            quality_diff = abs(content.rating - quality_pref)
            quality_score = max(0, 1 - quality_diff / 5.0)
            score += quality_score * 0.3
        
        # Content type preference match
        preferred_types = dict(patterns['content_preferences']['preferred_types'])
        if content.content_type in preferred_types:
            type_score = preferred_types[content.content_type] / max(sum(preferred_types.values()), 1)
            score += type_score * 0.25
        
        # Genre preference match
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            preferred_genres = dict(patterns['content_preferences']['preferred_genres'])
            
            genre_score = 0.0
            for genre in content_genres:
                if genre in preferred_genres:
                    genre_score += preferred_genres[genre] / max(sum(preferred_genres.values()), 1)
            
            score += min(genre_score, 1.0) * 0.25
        except:
            pass
        
        # Temporal relevance
        if content.release_date:
            years_old = (datetime.now().date() - content.release_date).days / 365
            if patterns['discovery_patterns']['openness_to_new'] > 0.7:
                # User likes new content
                temporal_score = max(0, 1 - years_old / 10)
            else:
                # User is okay with older content
                temporal_score = 0.7
            score += temporal_score * 0.2
        
        return score
    
    def _passes_quality_threshold(self, content):
        """Check if content passes quality thresholds"""
        if content.rating and content.rating < self.quality_thresholds['minimum_rating']:
            return False
        
        if content.vote_count and content.vote_count < self.quality_thresholds['minimum_vote_count']:
            return False
        
        return True
    
    def _calculate_personalization_boost(self, user_id, content):
        """Calculate personalization boost for content"""
        profile = self.user_profiles.get(user_id, {})
        boost = 0.0
        
        # Language preference boost
        try:
            content_languages = set(json.loads(content.languages or '[]'))
            if 'language_preferences' in profile:
                for language in content_languages:
                    if language in profile['language_preferences']:
                        boost += 0.1
        except:
            pass
        
        # Genre preference boost
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            if 'genre_preferences' in profile:
                for genre in content_genres:
                    if genre in profile['genre_preferences']:
                        boost += 0.05
        except:
            pass
        
        # Content type preference boost
        if 'content_type_preferences' in profile:
            if content.content_type in profile['content_type_preferences']:
                boost += 0.15
        
        # Recency boost
        if content.release_date:
            days_old = (datetime.now().date() - content.release_date).days
            if days_old <= self.quality_thresholds['recency_bonus_days']:
                boost += 0.1 * (1 - days_old / self.quality_thresholds['recency_bonus_days'])
        
        return min(boost, 0.5)  # Cap boost at 50%
    
    def _calculate_confidence_score(self, rec_data):
        """Calculate confidence score for recommendation"""
        scores = rec_data['scores']
        
        if not scores:
            return 0.0
        
        # Number of algorithms that contributed
        algorithm_count = len(scores)
        algorithm_bonus = min(algorithm_count / 4.0, 1.0)
        
        # Score consistency
        score_values = list(scores.values())
        if len(score_values) > 1:
            score_std = np.std(score_values)
            consistency_score = max(0, 1 - score_std)
        else:
            consistency_score = 0.5
        
        # Final score magnitude
        final_score = rec_data.get('final_score', 0)
        magnitude_score = min(final_score / 5.0, 1.0)
        
        confidence = (algorithm_bonus * 0.4 + consistency_score * 0.3 + magnitude_score * 0.3)
        return confidence
    
    def _identify_matching_factors(self, user_id, content):
        """Identify why content matches user preferences"""
        factors = []
        profile = self.user_profiles.get(user_id, {})
        
        # Check genre matches
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            if 'genre_preferences' in profile:
                matching_genres = content_genres & set(profile['genre_preferences'].keys())
                if matching_genres:
                    factors.append(f"Genres: {', '.join(list(matching_genres)[:3])}")
        except:
            pass
        
        # Check language matches
        try:
            content_languages = set(json.loads(content.languages or '[]'))
            if 'language_preferences' in profile:
                matching_languages = content_languages & set(profile['language_preferences'].keys())
                if matching_languages:
                    factors.append(f"Languages: {', '.join(matching_languages)}")
        except:
            pass
        
        # Check content type
        if 'content_type_preferences' in profile:
            if content.content_type in profile['content_type_preferences']:
                factors.append(f"Content type: {content.content_type}")
        
        # Check quality match
        if content.rating and content.rating >= profile.get('quality_preference', 7.0):
            factors.append(f"High quality (Rating: {content.rating})")
        
        return factors
    
    def get_contextual_recommendations(self, user_id, context, limit=15):
        """Get recommendations based on viewing context"""
        try:
            viewing_time = context.get('viewing_time', 'evening')
            available_time = context.get('available_time', 120)
            companions = context.get('viewing_companions', 'alone')
            device = context.get('viewing_device', 'tv')
            mood = context.get('mood', 'neutral')
            occasion = context.get('occasion', 'regular')
            
            # Get base recommendations
            base_recommendations = self.get_ultra_personalized_recommendations(
                user_id, limit * 2, include_explanations=False
            )
            
            contextual_recommendations = []
            
            for rec in base_recommendations:
                content = rec['content']
                context_score = self._calculate_context_match_score(content, context)
                
                if context_score > 0.3:  # Minimum context relevance
                    rec['context_match_score'] = context_score
                    rec['context_explanation'] = self._generate_context_explanation(content, context)
                    rec['suitability_factors'] = self._get_context_suitability_factors(content, context)
                    contextual_recommendations.append(rec)
            
            # Sort by context-adjusted score
            contextual_recommendations.sort(
                key=lambda x: x['personalization_score'] * (0.7 + 0.3 * x['context_match_score']),
                reverse=True
            )
            
            return contextual_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in contextual recommendations: {e}")
            return []
    
    def _calculate_context_match_score(self, content, context):
        """Calculate how well content matches viewing context"""
        score = 0.0
        
        available_time = context.get('available_time', 120)
        viewing_companions = context.get('viewing_companions', 'alone')
        mood = context.get('mood', 'neutral')
        
        # Runtime suitability
        if content.runtime:
            if content.runtime <= available_time:
                score += 1.0
            elif content.runtime <= available_time * 1.2:
                score += 0.7
            else:
                score += 0.3
        else:
            score += 0.5  # Neutral for unknown runtime
        
        # Companion suitability
        try:
            genres = set(json.loads(content.genres or '[]'))
            
            if viewing_companions == 'family':
                family_friendly_genres = {'Family', 'Animation', 'Adventure', 'Comedy'}
                if genres & family_friendly_genres:
                    score += 0.8
                elif 'Horror' in genres or 'Adult' in genres:
                    score -= 0.5
                else:
                    score += 0.4
            elif viewing_companions == 'friends':
                social_genres = {'Comedy', 'Action', 'Adventure', 'Thriller'}
                if genres & social_genres:
                    score += 0.6
                else:
                    score += 0.3
            else:  # alone
                score += 0.5  # All content suitable for solo viewing
        except:
            score += 0.3
        
        # Mood matching
        try:
            genres = set(json.loads(content.genres or '[]'))
            
            mood_genre_map = {
                'happy': {'Comedy', 'Adventure', 'Family', 'Musical'},
                'sad': {'Drama', 'Romance', 'Biography'},
                'excited': {'Action', 'Thriller', 'Adventure', 'Science Fiction'},
                'relaxed': {'Comedy', 'Romance', 'Documentary', 'Drama'},
                'neutral': set()  # All genres acceptable
            }
            
            if mood in mood_genre_map:
                suitable_genres = mood_genre_map[mood]
                if mood == 'neutral' or genres & suitable_genres:
                    score += 0.6
                else:
                    score += 0.2
        except:
            score += 0.3
        
        return min(score / 3.0, 1.0)  # Normalize to 0-1
    
    def predict_user_rating(self, user_id, content_id):
        """Predict what rating a user would give to content"""
        try:
            # Use neural embedding prediction
            base_prediction = self.neural_engine.predict_user_item_affinity(user_id, content_id)
            
            # Adjust based on user's rating patterns
            user_interactions = self.UserInteraction.query.filter_by(
                user_id=user_id, interaction_type='rating'
            ).all()
            
            if user_interactions:
                user_ratings = [interaction.rating for interaction in user_interactions if interaction.rating]
                if user_ratings:
                    avg_user_rating = sum(user_ratings) / len(user_ratings)
                    rating_std = np.std(user_ratings)
                    
                    # Adjust prediction based on user's rating tendencies
                    adjusted_prediction = base_prediction * 10.0  # Scale to 0-10
                    
                    # Apply user bias
                    global_avg = 7.0  # Assume global average
                    user_bias = avg_user_rating - global_avg
                    adjusted_prediction += user_bias
                    
                    return max(1.0, min(10.0, adjusted_prediction))
            
            return base_prediction * 10.0
            
        except Exception as e:
            logger.error(f"Error predicting user rating: {e}")
            return 7.0
    
    def get_user_behavior_insights(self, user_id):
        """Get comprehensive user behavior insights"""
        try:
            if user_id not in self.user_behavior_patterns:
                self.user_behavior_patterns[user_id] = self.behavior_analyzer.analyze_user_behavioral_patterns(user_id)
            
            patterns = self.user_behavior_patterns[user_id]
            
            insights = {
                'viewing_patterns': patterns['temporal'],
                'content_preferences': patterns['content_preferences'],
                'engagement_level': patterns['interaction_patterns']['engagement_level'],
                'discovery_openness': patterns['discovery_patterns']['openness_to_new'],
                'quality_sensitivity': patterns['quality_preferences']['is_quality_sensitive'],
                'recommendation_accuracy': self._calculate_recommendation_accuracy(user_id),
                'profile_completeness': self._calculate_profile_completeness(user_id),
                'personalization_strength': self.get_user_profile_strength(user_id)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting user behavior insights: {e}")
            return {}
    
    def get_smart_recommendations(self, user_id, current_context, limit=12):
        """Get smart recommendations based on current user state and patterns"""
        try:
            # Analyze current context
            current_hour = current_context.get('hour', 20)
            is_weekend = current_context.get('is_weekend', False)
            
            # Get user patterns
            if user_id not in self.user_behavior_patterns:
                self.user_behavior_patterns[user_id] = self.behavior_analyzer.analyze_user_behavioral_patterns(user_id)
            
            patterns = self.user_behavior_patterns[user_id]
            
            # Determine optimal content for current time
            peak_hours = patterns['temporal']['peak_hours']
            is_peak_time = current_hour in peak_hours
            
            # Get base recommendations
            base_recs = self.get_ultra_personalized_recommendations(
                user_id, limit * 2, include_explanations=False
            )
            
            smart_recommendations = []
            
            for rec in base_recs:
                content = rec['content']
                
                # Calculate timing relevance
                timing_score = self._calculate_timing_relevance(content, current_hour, is_weekend)
                
                # Calculate pattern match
                pattern_score = self._calculate_pattern_match(content, patterns)
                
                # Smart score combines personalization with context
                smart_score = (
                    rec['personalization_score'] * 0.6 +
                    timing_score * 0.25 +
                    pattern_score * 0.15
                )
                
                smart_recommendations.append({
                    'content': content,
                    'smart_score': smart_score,
                    'timing_relevance': timing_score,
                    'pattern_match': pattern_score,
                    'reason': self._generate_smart_recommendation_reason(
                        content, timing_score, pattern_score, is_peak_time, is_weekend
                    )
                })
            
            smart_recommendations.sort(key=lambda x: x['smart_score'], reverse=True)
            return smart_recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in smart recommendations: {e}")
            return []
    
    def _calculate_timing_relevance(self, content, current_hour, is_weekend):
        """Calculate how relevant content is for current time"""
        score = 0.5  # Base score
        
        # Content type timing preferences
        if content.content_type == 'movie':
            if current_hour >= 19 or is_weekend:  # Evening or weekend
                score += 0.3
        elif content.content_type == 'tv':
            if 18 <= current_hour <= 23:  # Prime time
                score += 0.4
        elif content.content_type == 'anime':
            if current_hour >= 20 or is_weekend:  # Evening or weekend
                score += 0.2
        
        # Runtime consideration
        if content.runtime:
            if current_hour >= 22:  # Late night - prefer shorter content
                if content.runtime <= 90:
                    score += 0.2
                else:
                    score -= 0.1
            elif current_hour <= 12 and is_weekend:  # Weekend morning - any length okay
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_pattern_match(self, content, patterns):
        """Calculate how well content matches user patterns"""
        score = 0.0
        
        # Content type pattern match
        preferred_types = dict(patterns['content_preferences']['preferred_types'])
        total_type_weight = sum(preferred_types.values())
        
        if content.content_type in preferred_types and total_type_weight > 0:
            score += (preferred_types[content.content_type] / total_type_weight) * 0.4
        
        # Genre pattern match
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            preferred_genres = dict(patterns['content_preferences']['preferred_genres'])
            total_genre_weight = sum(preferred_genres.values())
            
            if total_genre_weight > 0:
                genre_score = 0.0
                for genre in content_genres:
                    if genre in preferred_genres:
                        genre_score += preferred_genres[genre] / total_genre_weight
                score += min(genre_score, 1.0) * 0.4
        except:
            pass
        
        # Quality pattern match
        quality_pref = patterns['quality_preferences']['avg_preferred_quality']
        if content.rating:
            quality_diff = abs(content.rating - quality_pref)
            quality_score = max(0, 1 - quality_diff / 5.0)
            score += quality_score * 0.2
        
        return score
    
    def _generate_smart_recommendation_reason(self, content, timing_score, pattern_score, is_peak_time, is_weekend):
        """Generate reason for smart recommendation"""
        reasons = []
        
        if timing_score > 0.7:
            if is_weekend:
                reasons.append("Perfect for weekend viewing")
            elif is_peak_time:
                reasons.append("Great for your usual viewing time")
            else:
                reasons.append("Good timing for this content")
        
        if pattern_score > 0.6:
            reasons.append("Matches your viewing patterns")
        
        if content.content_type == 'movie' and content.runtime and content.runtime <= 120:
            reasons.append("Perfect length for tonight")
        
        if not reasons:
            reasons.append("Recommended based on your preferences")
        
        return "; ".join(reasons[:2])
    
    def update_search_behavior(self, user_id, query, metadata):
        """Update user profile based on search behavior"""
        try:
            if user_id not in self.user_profiles:
                self.initialize_user_profile(user_id)
            
            profile = self.user_profiles[user_id]
            
            # Initialize search patterns if not exists
            if 'search_patterns' not in profile:
                profile['search_patterns'] = {
                    'frequent_terms': defaultdict(int),
                    'search_times': [],
                    'clicked_positions': []
                }
            
            # Update search terms
            query_terms = query.lower().split()
            for term in query_terms:
                if len(term) > 2:  # Ignore very short terms
                    profile['search_patterns']['frequent_terms'][term] += 1
            
            # Update search timing
            profile['search_patterns']['search_times'].append(datetime.utcnow().hour)
            
            # Update clicked positions
            if metadata.get('clicked_position', -1) >= 0:
                profile['search_patterns']['clicked_positions'].append(metadata['clicked_position'])
            
            # Maintain only recent data (last 100 entries)
            if len(profile['search_patterns']['search_times']) > 100:
                profile['search_patterns']['search_times'] = profile['search_patterns']['search_times'][-100:]
            
            if len(profile['search_patterns']['clicked_positions']) > 100:
                profile['search_patterns']['clicked_positions'] = profile['search_patterns']['clicked_positions'][-100:]
            
        except Exception as e:
            logger.error(f"Error updating search behavior: {e}")
    
    def process_advanced_feedback(self, user_id, feedback_data):
        """Process advanced user feedback to improve recommendations"""
        try:
            if user_id not in self.user_profiles:
                self.initialize_user_profile(user_id)
            
            profile = self.user_profiles[user_id]
            
            # Initialize feedback tracking
            if 'feedback_history' not in profile:
                profile['feedback_history'] = []
            
            # Store feedback
            feedback_entry = {
                'content_id': feedback_data['content_id'],
                'feedback_type': feedback_data['feedback_type'],
                'reasons': feedback_data.get('feedback_reasons', []),
                'quality_rating': feedback_data.get('recommendation_quality', 'good'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            profile['feedback_history'].append(feedback_entry)
            
            # Maintain only recent feedback (last 50 entries)
            if len(profile['feedback_history']) > 50:
                profile['feedback_history'] = profile['feedback_history'][-50:]
            
            # Update algorithm weights based on feedback
            self._adjust_algorithm_weights_based_on_feedback(user_id, feedback_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing advanced feedback: {e}")
            return False
    
    def _adjust_algorithm_weights_based_on_feedback(self, user_id, feedback_data):
        """Adjust algorithm weights based on user feedback"""
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        
        # Initialize personalized weights if not exists
        if 'personalized_algorithm_weights' not in profile:
            profile['personalized_algorithm_weights'] = self.algorithm_weights.copy()
        
        feedback_type = feedback_data['feedback_type']
        content_id = feedback_data['content_id']
        
        # Determine which algorithm likely recommended this content
        # This is simplified - in practice, we'd track this during recommendation generation
        
        if feedback_type in ['loved', 'liked']:
            # Positive feedback - slightly increase weights
            adjustment = 0.02
        elif feedback_type in ['disliked', 'hated']:
            # Negative feedback - slightly decrease weights
            adjustment = -0.02
        else:
            adjustment = 0.0
        
        # Apply adjustment to all algorithms (simplified approach)
        for algorithm in profile['personalized_algorithm_weights']:
            profile['personalized_algorithm_weights'][algorithm] += adjustment
            
        # Normalize weights
        total_weight = sum(profile['personalized_algorithm_weights'].values())
        if total_weight > 0:
            for algorithm in profile['personalized_algorithm_weights']:
                profile['personalized_algorithm_weights'][algorithm] /= total_weight
    
    def get_comprehensive_user_analysis(self, user_id):
        """Get comprehensive analysis of user behavior and preferences"""
        try:
            analysis = {
                'user_id': user_id,
                'profile_summary': self.user_profiles.get(user_id, {}),
                'behavioral_patterns': self.get_user_behavior_insights(user_id),
                'recommendation_performance': self._get_recommendation_performance_metrics(user_id),
                'content_exploration': self._analyze_content_exploration(user_id),
                'engagement_trends': self._analyze_engagement_trends(user_id),
                'personalization_opportunities': self._identify_personalization_opportunities(user_id)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive user analysis: {e}")
            return {}
    
    def _get_recommendation_performance_metrics(self, user_id):
        """Get metrics on recommendation performance for user"""
        try:
            # Get recent interactions
            recent_interactions = self.UserInteraction.query.filter(
                self.UserInteraction.user_id == user_id,
                self.UserInteraction.timestamp >= datetime.utcnow() - timedelta(days=30)
            ).all()
            
            # Calculate metrics
            total_recommendations_clicked = len([
                i for i in recent_interactions 
                if i.interaction_type in ['view', 'like', 'favorite', 'watchlist']
            ])
            
            positive_interactions = len([
                i for i in recent_interactions 
                if i.interaction_type in ['like', 'favorite', 'watchlist']
            ])
            
            click_through_rate = positive_interactions / max(total_recommendations_clicked, 1)
            
            return {
                'total_interactions_30d': len(recent_interactions),
                'positive_interactions_30d': positive_interactions,
                'click_through_rate': click_through_rate,
                'engagement_score': min(click_through_rate * 2, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating recommendation performance: {e}")
            return {}
    
    def _analyze_content_exploration(self, user_id):
        """Analyze how much user explores different content"""
        try:
            interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            
            content_types = set()
            genres = set()
            languages = set()
            
            for interaction in interactions:
                if interaction.content_id:
                    content = self.Content.query.get(interaction.content_id)
                    if content:
                        content_types.add(content.content_type)
                        
                        try:
                            content_genres = json.loads(content.genres or '[]')
                            genres.update(content_genres)
                        except:
                            pass
                        
                        try:
                            content_languages = json.loads(content.languages or '[]')
                            languages.update(content_languages)
                        except:
                            pass
            
            return {
                'content_types_explored': len(content_types),
                'genres_explored': len(genres),
                'languages_explored': len(languages),
                'exploration_breadth': (len(content_types) + len(genres) + len(languages)) / 3.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content exploration: {e}")
            return {}
    
    def _analyze_engagement_trends(self, user_id):
        """Analyze user engagement trends over time"""
        try:
            # Get interactions from last 90 days
            interactions = self.UserInteraction.query.filter(
                self.UserInteraction.user_id == user_id,
                self.UserInteraction.timestamp >= datetime.utcnow() - timedelta(days=90)
            ).order_by(self.UserInteraction.timestamp).all()
            
            if not interactions:
                return {'trend': 'insufficient_data'}
            
            # Group by weeks
            weekly_counts = defaultdict(int)
            for interaction in interactions:
                week = interaction.timestamp.strftime('%Y-%W')
                weekly_counts[week] += 1
            
            # Calculate trend
            weeks = sorted(weekly_counts.keys())
            if len(weeks) >= 4:
                early_weeks = weeks[:len(weeks)//2]
                late_weeks = weeks[len(weeks)//2:]
                
                early_avg = sum(weekly_counts[week] for week in early_weeks) / len(early_weeks)
                late_avg = sum(weekly_counts[week] for week in late_weeks) / len(late_weeks)
                
                if late_avg > early_avg * 1.2:
                    trend = 'increasing'
                elif late_avg < early_avg * 0.8:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'insufficient_data'
            
            return {
                'trend': trend,
                'weekly_activity': dict(weekly_counts),
                'total_interactions_90d': len(interactions)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing engagement trends: {e}")
            return {}
    
    def _identify_personalization_opportunities(self, user_id):
        """Identify opportunities to improve personalization"""
        opportunities = []
        
        try:
            profile = self.user_profiles.get(user_id, {})
            
            # Check interaction count
            interaction_count = profile.get('interaction_count', 0)
            if interaction_count < 10:
                opportunities.append({
                    'type': 'low_interaction_count',
                    'description': 'More interactions needed for better personalization',
                    'priority': 'high'
                })
            
            # Check genre diversity
            if 'genre_preferences' in profile:
                genre_count = len(profile['genre_preferences'])
                if genre_count < 3:
                    opportunities.append({
                        'type': 'low_genre_diversity',
                        'description': 'Try exploring different genres for better recommendations',
                        'priority': 'medium'
                    })
            
            # Check rating behavior
            user_ratings = self.UserInteraction.query.filter_by(
                user_id=user_id, interaction_type='rating'
            ).count()
            
            if user_ratings < 5:
                opportunities.append({
                    'type': 'insufficient_ratings',
                    'description': 'Rating content helps improve recommendation accuracy',
                    'priority': 'medium'
                })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying personalization opportunities: {e}")
            return []
    
    # Helper methods for other functionalities
    def get_user_interaction_count(self, user_id):
        """Get total interaction count for user"""
        return self.UserInteraction.query.filter_by(user_id=user_id).count()
    
    def get_user_profile_strength(self, user_id):
        """Get user profile strength assessment"""
        interaction_count = self.get_user_interaction_count(user_id)
        
        if interaction_count == 0:
            return 'new_user'
        elif interaction_count < 5:
            return 'weak'
        elif interaction_count < 15:
            return 'developing'
        elif interaction_count < 50:
            return 'strong'
        else:
            return 'very_strong'
    
    def _predict_user_rating(self, user_id, content_id):
        """Internal method to predict user rating"""
        return self.predict_user_rating(user_id, content_id)
    
    def _calculate_recommendation_strength(self, rec_data):
        """Calculate overall recommendation strength"""
        final_score = rec_data.get('final_score', 0)
        confidence = rec_data.get('confidence', 0)
        
        strength_score = (final_score * 0.7 + confidence * 0.3)
        
        if strength_score >= 0.8:
            return 'very_strong'
        elif strength_score >= 0.6:
            return 'strong'
        elif strength_score >= 0.4:
            return 'moderate'
        else:
            return 'weak'
    
    def _generate_detailed_explanation(self, user_id, content_id, rec_data):
        """Generate detailed explanation for recommendation"""
        explanations = []
        
        # Algorithm contributions
        contributions = rec_data.get('algorithm_contributions', {})
        top_algorithm = max(contributions.items(), key=lambda x: x[1]) if contributions else None
        
        if top_algorithm:
            algo_name, score = top_algorithm
            if algo_name == 'collaborative':
                explanations.append("Users with similar tastes also enjoyed this")
            elif algo_name == 'content_deep':
                explanations.append("Matches the style of content you've enjoyed")
            elif algo_name == 'neural':
                explanations.append("AI analysis suggests you'll love this")
            elif algo_name == 'behavioral':
                explanations.append("Fits your viewing patterns perfectly")
        
        # Matching factors
        matching_factors = rec_data.get('matching_factors', [])
        if matching_factors:
            explanations.extend(matching_factors[:2])
        
        # Quality indicator
        content = self.Content.query.get(content_id)
        if content and content.rating and content.rating >= 8.0:
            explanations.append("Highly rated by critics and users")
        
        return "; ".join(explanations[:3]) if explanations else "Recommended based on your preferences"
    
    def _get_interaction_weight_for_content_analysis(self, interaction):
        """Get interaction weight for content analysis"""
        weights = {
            'favorite': 5.0, 'watchlist': 4.0, 'like': 3.0,
            'rating': interaction.rating if interaction.rating else 2.5,
            'view': 1.0, 'search_click': 0.5
        }
        
        base_weight = weights.get(interaction.interaction_type, 1.0)
        
        # Time decay
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        time_decay = math.exp(-days_ago / 45.0)
        
        return base_weight * time_decay
    
    def _apply_contextual_filtering(self, recommendations, time_context, mood_context):
        """Apply contextual filtering to recommendations"""
        if time_context == 'any' and mood_context == 'neutral':
            return recommendations
        
        filtered_recommendations = []
        
        for rec in recommendations:
            content = self.Content.query.get(rec['content_id'])
            if content:
                context_score = self._calculate_simple_context_score(content, time_context, mood_context)
                if context_score > 0.3:
                    rec['context_score'] = context_score
                    filtered_recommendations.append(rec)
        
        return filtered_recommendations
    
    def _calculate_simple_context_score(self, content, time_context, mood_context):
        """Calculate simple context score"""
        score = 0.5  # Base score
        
        try:
            genres = set(json.loads(content.genres or '[]'))
            
            # Time context
            if time_context == 'morning':
                if content.content_type in ['tv', 'anime'] or 'Comedy' in genres:
                    score += 0.3
            elif time_context == 'evening':
                if content.content_type == 'movie' or 'Drama' in genres:
                    score += 0.3
            
            # Mood context
            mood_genre_map = {
                'happy': {'Comedy', 'Adventure', 'Family'},
                'sad': {'Drama', 'Romance'},
                'excited': {'Action', 'Thriller', 'Adventure'},
                'relaxed': {'Comedy', 'Romance', 'Documentary'}
            }
            
            if mood_context in mood_genre_map:
                if genres & mood_genre_map[mood_context]:
                    score += 0.4
        except:
            pass
        
        return min(score, 1.0)
    
    def _apply_novelty_filtering(self, user_id, recommendations, novelty_factor):
        """Apply novelty filtering to recommendations"""
        # Simple novelty implementation
        user_genres = set()
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        for interaction in user_interactions:
            if interaction.content_id:
                content = self.Content.query.get(interaction.content_id)
                if content:
                    try:
                        genres = json.loads(content.genres or '[]')
                        user_genres.update(genres)
                    except:
                        pass
        
        for rec in recommendations:
            content = self.Content.query.get(rec['content_id'])
            if content:
                try:
                    content_genres = set(json.loads(content.genres or '[]'))
                    novelty_score = 1.0 - (len(content_genres & user_genres) / max(len(content_genres), 1))
                    rec['novelty_score'] = novelty_score
                    
                    # Adjust final score based on novelty
                    rec['final_score'] = (
                        rec['final_score'] * (1.0 - novelty_factor) +
                        novelty_score * novelty_factor
                    )
                except:
                    rec['novelty_score'] = 0.5
        
        return recommendations
    
    def _get_fallback_recommendations(self, user_id, limit, content_type):
        """Get fallback recommendations when main algorithms fail"""
        try:
            # Get popular content filtered by quality
            query = self.Content.query.filter(
                self.Content.rating >= 7.0,
                self.Content.vote_count >= 100
            )
            
            if content_type != 'all':
                query = query.filter(self.Content.content_type == content_type)
            
            popular_content = query.order_by(
                self.Content.popularity.desc()
            ).limit(limit).all()
            
            results = []
            for content in popular_content:
                results.append({
                    'content': content,
                    'personalization_score': 0.5,
                    'confidence_score': 0.3,
                    'novelty_score': 0.5,
                    'diversity_contribution': 0.0,
                    'explanation': 'Popular high-quality content',
                    'matching_factors': ['High quality', 'Popular choice'],
                    'predicted_rating': 7.0,
                    'recommendation_strength': 'moderate',
                    'algorithm_breakdown': {'fallback': 1.0},
                    'behavioral_match': 0.3,
                    'temporal_relevance': 0.5
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fallback recommendations: {e}")
            return []
    
    def _clear_user_caches(self, user_id):
        """Clear user-specific caches"""
        # Clear collaborative filtering cache
        if hasattr(self.collaborative_filtering, 'user_similarity_cache'):
            self.collaborative_filtering.user_similarity_cache.pop(user_id, None)
        
        # Clear neural embeddings cache
        if hasattr(self.neural_engine, 'user_embeddings'):
            self.neural_engine.user_embeddings.pop(user_id, None)
        
        # Clear behavior patterns cache
        self.user_behavior_patterns.pop(user_id, None)
    
    def _calculate_recommendation_accuracy(self, user_id):
        """Calculate recommendation accuracy for user"""
        try:
            # Get user's positive interactions in last 30 days
            recent_interactions = self.UserInteraction.query.filter(
                self.UserInteraction.user_id == user_id,
                self.UserInteraction.timestamp >= datetime.utcnow() - timedelta(days=30),
                self.UserInteraction.interaction_type.in_(['like', 'favorite', 'watchlist', 'rating'])
            ).all()
            
            if not recent_interactions:
                return 0.0
            
            # Simple accuracy calculation based on positive interactions
            positive_interactions = len([
                i for i in recent_interactions 
                if i.interaction_type in ['like', 'favorite', 'watchlist'] or 
                   (i.interaction_type == 'rating' and i.rating and i.rating >= 7.0)
            ])
            
            return min(positive_interactions / len(recent_interactions), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating recommendation accuracy: {e}")
            return 0.0
    
    def _calculate_profile_completeness(self, user_id):
        """Calculate how complete the user profile is"""
        profile = self.user_profiles.get(user_id, {})
        
        completeness_factors = 0
        total_factors = 6
        
        if profile.get('interaction_count', 0) > 5:
            completeness_factors += 1
        
        if 'genre_preferences' in profile and len(profile['genre_preferences']) > 3:
            completeness_factors += 1
        
        if 'language_preferences' in profile and len(profile['language_preferences']) > 1:
            completeness_factors += 1
        
        if 'content_type_preferences' in profile and len(profile['content_type_preferences']) > 1:
            completeness_factors += 1
        
        # Check if user has rated content
        rating_count = self.UserInteraction.query.filter_by(
            user_id=user_id, interaction_type='rating'
        ).count()
        if rating_count > 3:
            completeness_factors += 1
        
        # Check behavioral patterns
        if user_id in self.user_behavior_patterns:
            completeness_factors += 1
        
        return completeness_factors / total_factors