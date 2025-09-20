import numpy as np
from datetime import datetime, timedelta
import json
import logging
import random
from collections import defaultdict, Counter
from .algorithm import (
    EnhancedCollaborativeFiltering, AdvancedContentBasedFiltering, 
    PrecisionMatrixFactorization, BehavioralPatternAnalyzer, RealTimePersonalizationEngine
)

logger = logging.getLogger(__name__)

class PersonalizedRecommendationEngine:
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
        self.collaborative_filtering = EnhancedCollaborativeFiltering(db, models)
        self.content_based_filtering = AdvancedContentBasedFiltering(db, models)
        self.matrix_factorization = PrecisionMatrixFactorization(db, models)
        self.behavior_analyzer = BehavioralPatternAnalyzer(db, models)
        self.realtime_engine = RealTimePersonalizationEngine(db, models)
        
        self.algorithm_weights = {
            'collaborative': 0.35,
            'content_based': 0.30,
            'matrix_factorization': 0.25,
            'behavioral': 0.10
        }
        
        self.user_profiles_cache = {}
        self.content_features_cache = {}
        
    def initialize_user_profile(self, user_id):
        user = self.User.query.get(user_id)
        if not user:
            return False
        
        try:
            preferred_languages = json.loads(user.preferred_languages or '[]')
            preferred_genres = json.loads(user.preferred_genres or '[]')
        except:
            preferred_languages = ['english', 'telugu']
            preferred_genres = []
        
        initial_interactions = []
        
        if preferred_genres:
            for genre in preferred_genres[:3]:
                popular_content = self.Content.query.filter(
                    self.Content.genres.contains(genre)
                ).order_by(self.Content.rating.desc()).limit(2).all()
                
                for content in popular_content:
                    interaction = self.UserInteraction(
                        user_id=user_id,
                        content_id=content.id,
                        interaction_type='preference_based',
                        interaction_metadata={'source': 'initial_preference'}
                    )
                    initial_interactions.append(interaction)
        
        if preferred_languages:
            for language in preferred_languages[:2]:
                lang_content = self.Content.query.filter(
                    self.Content.languages.contains(language)
                ).order_by(self.Content.popularity.desc()).limit(2).all()
                
                for content in lang_content:
                    interaction = self.UserInteraction(
                        user_id=user_id,
                        content_id=content.id,
                        interaction_type='preference_based',
                        interaction_metadata={'source': 'initial_language_preference'}
                    )
                    initial_interactions.append(interaction)
        
        for interaction in initial_interactions:
            self.db.session.add(interaction)
        
        try:
            self.db.session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize user profile: {e}")
            self.db.session.rollback()
            return False
    
    def get_ultra_personalized_recommendations(self, user_id, limit=30, content_type='all', 
                                             strategy='intelligent_hybrid', include_explanations=True):
        
        user_interaction_count = self.get_user_interaction_count(user_id)
        
        if user_interaction_count < 3:
            return self._get_enhanced_cold_start_recommendations(user_id, limit, content_type)
        
        all_recommendations = defaultdict(lambda: {
            'scores': [],
            'algorithms': [],
            'explanations': [],
            'confidence_factors': []
        })
        
        try:
            collab_recs = self.collaborative_filtering.get_precision_user_recommendations(user_id, limit * 2)
            for content_id, score, metadata in collab_recs:
                all_recommendations[content_id]['scores'].append(score * self.algorithm_weights['collaborative'])
                all_recommendations[content_id]['algorithms'].append('collaborative')
                all_recommendations[content_id]['explanations'].append(
                    f"Users with similar preferences enjoyed this ({len(metadata.get('sources', []))} similar users)"
                )
                all_recommendations[content_id]['confidence_factors'].append(metadata.get('profile_match', 0.5))
        except Exception as e:
            logger.warning(f"Collaborative filtering failed: {e}")
        
        try:
            content_recs = self.content_based_filtering.get_precision_content_recommendations(user_id, limit * 2)
            for content_id, score, metadata in content_recs:
                all_recommendations[content_id]['scores'].append(score * self.algorithm_weights['content_based'])
                all_recommendations[content_id]['algorithms'].append('content_based')
                
                match_details = metadata.get('content_match_details', {})
                explanation_parts = []
                if 'matching_genres' in match_details:
                    explanation_parts.append(f"Matches your interest in {', '.join(match_details['matching_genres'][:2])}")
                if 'matching_languages' in match_details:
                    explanation_parts.append(f"In your preferred languages")
                
                explanation = '; '.join(explanation_parts) if explanation_parts else "Matches your content preferences"
                all_recommendations[content_id]['explanations'].append(explanation)
                all_recommendations[content_id]['confidence_factors'].append(metadata.get('base_similarity', 0.5))
        except Exception as e:
            logger.warning(f"Content-based filtering failed: {e}")
        
        try:
            mf_recs = self.matrix_factorization.get_precision_recommendations(user_id, limit * 2)
            for content_id, score, metadata in mf_recs:
                all_recommendations[content_id]['scores'].append(score * self.algorithm_weights['matrix_factorization'])
                all_recommendations[content_id]['algorithms'].append('matrix_factorization')
                all_recommendations[content_id]['explanations'].append(
                    f"Based on your latent preferences (confidence: {metadata.get('confidence', 0.5):.2f})"
                )
                all_recommendations[content_id]['confidence_factors'].append(metadata.get('confidence', 0.5))
        except Exception as e:
            logger.warning(f"Matrix factorization failed: {e}")
        
        try:
            behavior_patterns = self.behavior_analyzer.analyze_user_behavior_patterns(user_id)
            behavioral_boost = self._calculate_behavioral_boost(behavior_patterns)
            
            for content_id in all_recommendations:
                content = self.Content.query.get(content_id)
                if content:
                    boost = self._get_content_behavioral_boost(content, behavior_patterns, behavioral_boost)
                    if boost > 0:
                        all_recommendations[content_id]['scores'].append(boost * self.algorithm_weights['behavioral'])
                        all_recommendations[content_id]['algorithms'].append('behavioral')
                        all_recommendations[content_id]['explanations'].append(
                            "Matches your viewing patterns and preferences"
                        )
                        all_recommendations[content_id]['confidence_factors'].append(0.7)
        except Exception as e:
            logger.warning(f"Behavioral analysis failed: {e}")
        
        final_recommendations = []
        
        for content_id, data in all_recommendations.items():
            content = self.Content.query.get(content_id)
            if not content:
                continue
            
            if content_type != 'all' and content.content_type != content_type:
                continue
            
            if not data['scores']:
                continue
            
            combined_score = sum(data['scores'])
            avg_confidence = sum(data['confidence_factors']) / len(data['confidence_factors'])
            
            personalization_score = combined_score * avg_confidence
            
            quality_boost = self._calculate_quality_boost(content)
            final_score = personalization_score * quality_boost
            
            match_reason = self._generate_comprehensive_explanation(data['explanations'], data['algorithms'])
            
            behavioral_match = self._calculate_behavioral_match_score(content_id, user_id)
            preference_alignment = self._calculate_preference_alignment(content, user_id)
            
            final_recommendations.append({
                'content': content,
                'personalization_score': round(final_score, 4),
                'match_reason': match_reason,
                'confidence_level': round(avg_confidence, 3),
                'behavioral_match': round(behavioral_match, 3),
                'preference_alignment': round(preference_alignment, 3),
                'algorithm_contributions': dict(zip(data['algorithms'], data['scores'])),
                'quality_boost': round(quality_boost, 3)
            })
        
        final_recommendations.sort(key=lambda x: x['personalization_score'], reverse=True)
        
        diversified_recommendations = self._apply_intelligent_diversification(
            final_recommendations, user_id, limit
        )
        
        return diversified_recommendations
    
    def get_behavior_driven_recommendations(self, user_id, limit=25, behavior_focus='comprehensive', temporal_weight=0.7):
        behavior_patterns = self.behavior_analyzer.analyze_user_behavior_patterns(user_id)
        
        if not behavior_patterns:
            return self._get_enhanced_cold_start_recommendations(user_id, limit)
        
        user_interactions = set(
            interaction.content_id for interaction in 
            self.UserInteraction.query.filter_by(user_id=user_id).all()
        )
        
        candidate_content = self.Content.query.limit(3000).all()
        
        recommendations = []
        
        for content in candidate_content:
            if content.id not in user_interactions:
                behavior_score = self._calculate_comprehensive_behavior_score(
                    content, behavior_patterns, behavior_focus, temporal_weight
                )
                
                if behavior_score > 0.2:
                    temporal_relevance = self._calculate_temporal_relevance(content, behavior_patterns)
                    interaction_affinity = self._calculate_interaction_affinity(content, user_id)
                    
                    final_score = behavior_score * (1 + temporal_relevance + interaction_affinity)
                    
                    behavior_pattern = self._identify_behavior_pattern(content, behavior_patterns)
                    
                    recommendations.append({
                        'content': content,
                        'behavior_score': round(final_score, 4),
                        'behavior_pattern': behavior_pattern,
                        'temporal_relevance': round(temporal_relevance, 3),
                        'interaction_affinity': round(interaction_affinity, 3)
                    })
        
        recommendations.sort(key=lambda x: x['behavior_score'], reverse=True)
        return recommendations[:limit]
    
    def _calculate_comprehensive_behavior_score(self, content, behavior_patterns, focus, temporal_weight):
        score = 0.0
        
        if 'content_type_preferences' in behavior_patterns:
            type_pref = behavior_patterns['content_type_preferences'].get(content.content_type, 0)
            score += type_pref * 0.3
        
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            if 'evolving_preferences' in behavior_patterns:
                for genre in content_genres:
                    if genre in behavior_patterns['evolving_preferences']:
                        if behavior_patterns['evolving_preferences'][genre] == 'increasing':
                            score += 0.4
                        elif behavior_patterns['evolving_preferences'][genre] == 'decreasing':
                            score -= 0.2
        except:
            pass
        
        if content.release_date and 'peak_activity_hours' in behavior_patterns:
            current_hour = datetime.now().hour
            if current_hour in behavior_patterns['peak_activity_hours']:
                score += 0.2 * temporal_weight
        
        if content.rating:
            if content.rating >= 8.0:
                score += 0.3
            elif content.rating >= 7.0:
                score += 0.2
        
        if focus == 'exploration' and behavior_patterns.get('discovery_style') == 'exploration':
            if content.is_new_release or content.popularity < 100:
                score += 0.4
        elif focus == 'focused' and behavior_patterns.get('discovery_style') == 'search_based':
            if content.vote_count and content.vote_count > 1000:
                score += 0.3
        
        return min(score, 2.0)
    
    def _calculate_behavioral_match_score(self, content_id, user_id):
        content = self.Content.query.get(content_id)
        if not content:
            return 0.0
        
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not user_interactions:
            return 0.5
        
        similar_content_interactions = []
        
        for interaction in user_interactions:
            other_content = self.Content.query.get(interaction.content_id)
            if other_content and other_content.content_type == content.content_type:
                try:
                    other_genres = set(json.loads(other_content.genres or '[]'))
                    content_genres = set(json.loads(content.genres or '[]'))
                    
                    if other_genres & content_genres:
                        weight = self._calculate_interaction_weight(interaction)
                        similar_content_interactions.append(weight)
                except:
                    pass
        
        if not similar_content_interactions:
            return 0.3
        
        avg_interaction_weight = sum(similar_content_interactions) / len(similar_content_interactions)
        return min(avg_interaction_weight / 5.0, 1.0)
    
    def _calculate_preference_alignment(self, content, user_id):
        user = self.User.query.get(user_id)
        if not user:
            return 0.5
        
        try:
            user_preferred_languages = set(json.loads(user.preferred_languages or '[]'))
            user_preferred_genres = set(json.loads(user.preferred_genres or '[]'))
        except:
            user_preferred_languages = set()
            user_preferred_genres = set()
        
        alignment_score = 0.0
        
        try:
            content_languages = set(json.loads(content.languages or '[]'))
            if user_preferred_languages and content_languages:
                lang_overlap = len(user_preferred_languages & content_languages) / len(content_languages)
                alignment_score += lang_overlap * 0.4
        except:
            pass
        
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            if user_preferred_genres and content_genres:
                genre_overlap = len(user_preferred_genres & content_genres) / len(content_genres)
                alignment_score += genre_overlap * 0.6
        except:
            pass
        
        return min(alignment_score, 1.0)
    
    def _get_enhanced_cold_start_recommendations(self, user_id, limit, content_type='all'):
        user = self.User.query.get(user_id)
        if not user:
            return []
        
        try:
            preferred_languages = json.loads(user.preferred_languages or '[]')
            preferred_genres = json.loads(user.preferred_genres or '[]')
        except:
            preferred_languages = ['english', 'telugu']
            preferred_genres = []
        
        recommendations = []
        
        base_query = self.Content.query
        if content_type != 'all':
            base_query = base_query.filter_by(content_type=content_type)
        
        if preferred_languages:
            for language in preferred_languages[:3]:
                lang_content = base_query.filter(
                    self.Content.languages.contains(language)
                ).order_by(
                    self.Content.rating.desc(),
                    self.Content.popularity.desc()
                ).limit(8).all()
                
                for content in lang_content:
                    score = self._calculate_cold_start_score(content, preferred_languages, preferred_genres)
                    recommendations.append({
                        'content': content,
                        'personalization_score': score,
                        'match_reason': f"Popular {language} content with high ratings",
                        'confidence_level': 0.6,
                        'behavioral_match': 0.5,
                        'preference_alignment': 0.8
                    })
        
        if preferred_genres:
            for genre in preferred_genres[:3]:
                genre_content = base_query.filter(
                    self.Content.genres.contains(genre)
                ).order_by(
                    self.Content.rating.desc(),
                    self.Content.popularity.desc()
                ).limit(6).all()
                
                for content in genre_content:
                    score = self._calculate_cold_start_score(content, preferred_languages, preferred_genres)
                    recommendations.append({
                        'content': content,
                        'personalization_score': score,
                        'match_reason': f"Highly rated {genre} content",
                        'confidence_level': 0.7,
                        'behavioral_match': 0.5,
                        'preference_alignment': 0.9
                    })
        
        trending_content = base_query.filter_by(is_trending=True).order_by(
            self.Content.rating.desc()
        ).limit(8).all()
        
        for content in trending_content:
            score = self._calculate_cold_start_score(content, preferred_languages, preferred_genres)
            recommendations.append({
                'content': content,
                'personalization_score': score,
                'match_reason': "Currently trending with high ratings",
                'confidence_level': 0.5,
                'behavioral_match': 0.4,
                'preference_alignment': 0.6
            })
        
        seen_ids = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec['content'].id not in seen_ids:
                seen_ids.add(rec['content'].id)
                unique_recommendations.append(rec)
        
        unique_recommendations.sort(key=lambda x: x['personalization_score'], reverse=True)
        return unique_recommendations[:limit]
    
    def _calculate_cold_start_score(self, content, preferred_languages, preferred_genres):
        score = 0.0
        
        if content.rating:
            score += (content.rating / 10.0) * 0.4
        
        if content.popularity:
            score += min(content.popularity / 1000.0, 1.0) * 0.2
        
        try:
            content_languages = set(json.loads(content.languages or '[]'))
            lang_match = len(set(preferred_languages) & content_languages) / max(len(content_languages), 1)
            score += lang_match * 0.3
        except:
            pass
        
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            genre_match = len(set(preferred_genres) & content_genres) / max(len(content_genres), 1)
            score += genre_match * 0.1
        except:
            pass
        
        return score
    
    def _calculate_quality_boost(self, content):
        boost = 1.0
        
        if content.rating:
            if content.rating >= 9.0:
                boost *= 1.3
            elif content.rating >= 8.0:
                boost *= 1.2
            elif content.rating >= 7.5:
                boost *= 1.1
        
        if content.is_critics_choice:
            boost *= 1.15
        
        if content.vote_count and content.vote_count >= 5000:
            boost *= 1.1
        
        if content.is_trending:
            boost *= 1.05
        
        return boost
    
    def _generate_comprehensive_explanation(self, explanations, algorithms):
        if not explanations:
            return "Personalized recommendation based on your preferences"
        
        unique_explanations = list(dict.fromkeys(explanations))
        
        if len(unique_explanations) == 1:
            return unique_explanations[0]
        
        primary_explanation = unique_explanations[0]
        
        algorithm_diversity = len(set(algorithms))
        if algorithm_diversity >= 3:
            return f"{primary_explanation} (Multi-algorithm analysis)"
        else:
            return primary_explanation
    
    def _apply_intelligent_diversification(self, recommendations, user_id, limit):
        if len(recommendations) <= limit:
            return recommendations
        
        user_profile = self.get_comprehensive_user_profile(user_id)
        
        final_recommendations = []
        selected_genres = set()
        selected_types = set()
        
        for rec in recommendations:
            content = rec['content']
            
            try:
                content_genres = set(json.loads(content.genres or '[]'))
            except:
                content_genres = set()
            
            diversity_penalty = 0.0
            
            if content_genres & selected_genres:
                overlap_ratio = len(content_genres & selected_genres) / max(len(content_genres), 1)
                diversity_penalty = overlap_ratio * 0.3
            
            if content.content_type in selected_types:
                diversity_penalty += 0.2
            
            adjusted_score = rec['personalization_score'] * (1 - diversity_penalty)
            rec['personalization_score'] = adjusted_score
            
            final_recommendations.append(rec)
            selected_genres.update(content_genres)
            selected_types.add(content.content_type)
            
            if len(final_recommendations) >= limit:
                break
        
        return final_recommendations
    
    def _calculate_temporal_relevance(self, content, behavior_patterns):
        relevance = 0.0
        
        if content.release_date:
            days_since_release = (datetime.now().date() - content.release_date).days
            
            if days_since_release <= 30:
                relevance += 0.3
            elif days_since_release <= 90:
                relevance += 0.2
            elif days_since_release <= 365:
                relevance += 0.1
        
        current_hour = datetime.now().hour
        if 'peak_activity_hours' in behavior_patterns:
            if current_hour in behavior_patterns['peak_activity_hours']:
                relevance += 0.2
        
        return relevance
    
    def _calculate_interaction_affinity(self, content, user_id):
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not user_interactions:
            return 0.0
        
        similar_content_count = 0
        total_interactions = len(user_interactions)
        
        for interaction in user_interactions:
            other_content = self.Content.query.get(interaction.content_id)
            if other_content and other_content.content_type == content.content_type:
                try:
                    other_genres = set(json.loads(other_content.genres or '[]'))
                    content_genres = set(json.loads(content.genres or '[]'))
                    
                    if other_genres & content_genres:
                        similar_content_count += 1
                except:
                    pass
        
        return similar_content_count / total_interactions if total_interactions > 0 else 0.0
    
    def _identify_behavior_pattern(self, content, behavior_patterns):
        patterns = []
        
        if 'content_type_preferences' in behavior_patterns:
            type_pref = behavior_patterns['content_type_preferences'].get(content.content_type, 0)
            if type_pref > 0.5:
                patterns.append(f"Strong {content.content_type} preference")
        
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            if 'evolving_preferences' in behavior_patterns:
                for genre in content_genres:
                    if genre in behavior_patterns['evolving_preferences']:
                        trend = behavior_patterns['evolving_preferences'][genre]
                        patterns.append(f"{genre} interest {trend}")
        except:
            pass
        
        if 'discovery_style' in behavior_patterns:
            patterns.append(f"{behavior_patterns['discovery_style']} discovery style")
        
        return '; '.join(patterns) if patterns else "General preference match"
    
    def process_real_time_interaction(self, user_id, content_id, interaction_type, rating=None, metadata=None):
        return self.realtime_engine.process_real_time_interaction(
            user_id, content_id, interaction_type, rating, metadata
        )
    
    def record_detailed_feedback(self, user_id, content_id, feedback_type, feedback_value=1.0, feedback_context=None):
        try:
            feedback_interaction = self.UserInteraction(
                user_id=user_id,
                content_id=content_id,
                interaction_type='detailed_feedback',
                interaction_metadata={
                    'feedback_type': feedback_type,
                    'feedback_value': feedback_value,
                    'feedback_context': feedback_context or {},
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            self.db.session.add(feedback_interaction)
            self.db.session.commit()
            
            self.update_user_behavior_profile(user_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to record detailed feedback: {e}")
            self.db.session.rollback()
            return False
    
    def update_user_behavior_profile(self, user_id):
        if user_id in self.user_profiles_cache:
            del self.user_profiles_cache[user_id]
        
        if hasattr(self.collaborative_filtering, 'user_behavior_profiles'):
            self.collaborative_filtering.user_behavior_profiles.pop(user_id, None)
        
        if hasattr(self.content_based_filtering, 'user_content_profiles'):
            self.content_based_filtering.user_content_profiles.pop(user_id, None)
    
    def get_comprehensive_user_profile(self, user_id):
        if user_id in self.user_profiles_cache:
            return self.user_profiles_cache[user_id]
        
        user = self.User.query.get(user_id)
        if not user:
            return {}
        
        interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        profile = {
            'user_id': user_id,
            'total_interactions': len(interactions),
            'interaction_types': Counter(),
            'genre_preferences': Counter(),
            'language_preferences': Counter(),
            'content_type_distribution': Counter(),
            'rating_statistics': {},
            'temporal_patterns': {},
            'profile_strength': 'new_user',
            'last_updated': datetime.utcnow().isoformat()
        }
        
        ratings = []
        for interaction in interactions:
            profile['interaction_types'][interaction.interaction_type] += 1
            
            if interaction.rating:
                ratings.append(interaction.rating)
            
            content = self.Content.query.get(interaction.content_id)
            if content:
                try:
                    genres = json.loads(content.genres or '[]')
                    for genre in genres:
                        weight = self._calculate_interaction_weight(interaction)
                        profile['genre_preferences'][genre] += weight
                except:
                    pass
                
                try:
                    languages = json.loads(content.languages or '[]')
                    for language in languages:
                        weight = self._calculate_interaction_weight(interaction)
                        profile['language_preferences'][language] += weight
                except:
                    pass
                
                profile['content_type_distribution'][content.content_type] += 1
        
        if ratings:
            profile['rating_statistics'] = {
                'average': round(sum(ratings) / len(ratings), 2),
                'count': len(ratings),
                'min': min(ratings),
                'max': max(ratings)
            }
        
        total_interactions = len(interactions)
        if total_interactions >= 100:
            profile['profile_strength'] = 'very_strong'
        elif total_interactions >= 50:
            profile['profile_strength'] = 'strong'
        elif total_interactions >= 20:
            profile['profile_strength'] = 'moderate'
        elif total_interactions >= 5:
            profile['profile_strength'] = 'weak'
        
        behavior_patterns = self.behavior_analyzer.analyze_user_behavior_patterns(user_id)
        profile['behavior_analysis'] = behavior_patterns
        
        self.user_profiles_cache[user_id] = profile
        return profile
    
    def analyze_user_behavior_patterns(self, user_id):
        return self.behavior_analyzer.analyze_user_behavior_patterns(user_id)
    
    def get_user_analytics(self, user_id):
        profile = self.get_comprehensive_user_profile(user_id)
        
        analytics = {
            'profile_completeness': self._calculate_profile_completeness(profile),
            'engagement_score': self._calculate_engagement_score(user_id),
            'preference_clarity': self._calculate_preference_clarity(profile),
            'discovery_tendency': self._calculate_discovery_tendency(user_id),
            'recommendation_receptiveness': self._calculate_recommendation_receptiveness(user_id)
        }
        
        return analytics
    
    def _calculate_profile_completeness(self, profile):
        completeness = 0.0
        
        if profile['total_interactions'] > 0:
            completeness += 0.3
        
        if profile['rating_statistics']:
            completeness += 0.2
        
        if len(profile['genre_preferences']) >= 3:
            completeness += 0.2
        
        if len(profile['content_type_distribution']) >= 2:
            completeness += 0.15
        
        if profile['total_interactions'] >= 20:
            completeness += 0.15
        
        return round(completeness, 2)
    
    def _calculate_engagement_score(self, user_id):
        interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not interactions:
            return 0.0
        
        total_weight = sum(self._calculate_interaction_weight(interaction) for interaction in interactions)
        avg_weight = total_weight / len(interactions)
        
        return min(avg_weight / 3.0, 1.0)
    
    def _calculate_preference_clarity(self, profile):
        if not profile['genre_preferences']:
            return 0.0
        
        total_genre_weight = sum(profile['genre_preferences'].values())
        top_3_weight = sum([weight for _, weight in profile['genre_preferences'].most_common(3)])
        
        clarity = top_3_weight / total_genre_weight if total_genre_weight > 0 else 0.0
        return round(clarity, 2)
    
    def _calculate_discovery_tendency(self, user_id):
        interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not interactions:
            return 0.5
        
        search_count = sum(1 for i in interactions if i.interaction_type == 'search')
        exploration_score = search_count / len(interactions) if interactions else 0
        
        return round(exploration_score, 2)
    
    def _calculate_recommendation_receptiveness(self, user_id):
        feedback_interactions = self.UserInteraction.query.filter_by(
            user_id=user_id,
            interaction_type='detailed_feedback'
        ).all()
        
        if not feedback_interactions:
            return 0.5
        
        positive_feedback = 0
        for interaction in feedback_interactions:
            if interaction.interaction_metadata:
                feedback_type = interaction.interaction_metadata.get('feedback_type', '')
                if feedback_type in ['like', 'love', 'interested']:
                    positive_feedback += 1
        
        receptiveness = positive_feedback / len(feedback_interactions) if feedback_interactions else 0.5
        return round(receptiveness, 2)
    
    def _calculate_behavioral_boost(self, behavior_patterns):
        boost_factors = {}
        
        if 'content_type_preferences' in behavior_patterns:
            for content_type, preference in behavior_patterns['content_type_preferences'].items():
                boost_factors[f"type_{content_type}"] = preference
        
        if 'evolving_preferences' in behavior_patterns:
            for genre, trend in behavior_patterns['evolving_preferences'].items():
                if trend == 'increasing':
                    boost_factors[f"genre_{genre}"] = 0.3
                elif trend == 'decreasing':
                    boost_factors[f"genre_{genre}"] = -0.2
        
        return boost_factors
    
    def _get_content_behavioral_boost(self, content, behavior_patterns, behavioral_boost):
        boost = 0.0
        
        type_key = f"type_{content.content_type}"
        if type_key in behavioral_boost:
            boost += behavioral_boost[type_key] * 0.5
        
        try:
            content_genres = json.loads(content.genres or '[]')
            for genre in content_genres:
                genre_key = f"genre_{genre}"
                if genre_key in behavioral_boost:
                    boost += behavioral_boost[genre_key] * 0.3
        except:
            pass
        
        return boost
    
    def _calculate_interaction_weight(self, interaction):
        base_weights = {
            'view': 1.0,
            'like': 3.0,
            'favorite': 5.0,
            'watchlist': 4.0,
            'rating': (interaction.rating or 3.0) * 0.8,
            'search': 0.8,
            'detailed_feedback': 2.0,
            'preference_based': 1.5
        }
        
        weight = base_weights.get(interaction.interaction_type, 1.0)
        
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        time_decay = math.exp(-days_ago / 60.0)
        
        return weight * time_decay
    
    def get_user_interaction_count(self, user_id):
        return self.UserInteraction.query.filter_by(user_id=user_id).count()

import math