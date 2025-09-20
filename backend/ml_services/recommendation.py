import numpy as np
from datetime import datetime, timedelta
import json
import logging
import random
from collections import defaultdict, Counter
from .algorithm import (
    CollaborativeFiltering, ContentBasedFiltering, MatrixFactorization,
    DeepLearningRecommender, DiversityOptimizer, NoveltyDetector,
    ColdStartHandler, RealtimeLearning, EvaluationMetrics
)

logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
        self.collaborative_filtering = CollaborativeFiltering(db, models)
        self.content_based_filtering = ContentBasedFiltering(db, models)
        self.matrix_factorization = MatrixFactorization(db, models)
        self.deep_learning = DeepLearningRecommender(db, models)
        self.diversity_optimizer = DiversityOptimizer(db, models)
        self.novelty_detector = NoveltyDetector(db, models)
        self.cold_start_handler = ColdStartHandler(db, models)
        self.realtime_learning = RealtimeLearning(db, models)
        self.evaluation_metrics = EvaluationMetrics(db, models)
        
        self.algorithm_weights = {
            'collaborative_user': 0.25,
            'collaborative_item': 0.20,
            'content_based': 0.20,
            'matrix_factorization': 0.15,
            'deep_learning': 0.20
        }
        
        self.user_profiles = {}
        
    def get_personalized_recommendations(self, user_id, limit=20, content_type='all', strategy='hybrid'):
        try:
            user_interaction_count = self.get_user_interaction_count(user_id)
            
            if user_interaction_count < 5:
                return self._get_cold_start_recommendations(user_id, limit, content_type)
            
            if strategy == 'collaborative':
                return self._get_collaborative_recommendations(user_id, limit, content_type)
            elif strategy == 'content_based':
                return self._get_content_based_recommendations(user_id, limit, content_type)
            elif strategy == 'matrix_factorization':
                return self._get_matrix_factorization_recommendations(user_id, limit, content_type)
            elif strategy == 'deep_learning':
                return self._get_deep_learning_recommendations(user_id, limit, content_type)
            else:
                return self._get_hybrid_recommendations(user_id, limit, content_type)
                
        except Exception as e:
            logger.error(f"Error in personalized recommendations: {e}")
            return []
    
    def get_advanced_recommendations(self, user_id, limit=20, include_explanations=True, diversity_factor=0.3):
        try:
            user_interaction_count = self.get_user_interaction_count(user_id)
            
            if user_interaction_count < 3:
                base_recommendations = self._get_cold_start_recommendations(user_id, limit * 2)
            else:
                base_recommendations = self._get_hybrid_recommendations(user_id, limit * 2)
            
            if not base_recommendations:
                return []
            
            enhanced_recommendations = []
            
            for content_id, base_score in base_recommendations:
                content = self.Content.query.get(content_id)
                if not content:
                    continue
                
                novelty_score = self.novelty_detector.calculate_novelty_score(user_id, content_id)
                
                feedback_adjusted_score = self.realtime_learning.get_feedback_adjusted_score(
                    user_id, content_id, base_score
                )
                
                final_score = (0.7 * feedback_adjusted_score + 0.3 * novelty_score)
                
                explanation = self._generate_explanation(user_id, content_id, include_explanations)
                
                algorithm_mix = self._get_algorithm_contribution(user_id, content_id)
                
                enhanced_recommendations.append({
                    'content': content,
                    'ml_score': final_score,
                    'explanation': explanation,
                    'algorithm_mix': algorithm_mix,
                    'confidence': min(final_score / max(base_score, 0.1), 1.0),
                    'novelty_score': novelty_score,
                    'diversity_contribution': 0.0
                })
            
            enhanced_recommendations.sort(key=lambda x: x['ml_score'], reverse=True)
            
            if diversity_factor > 0:
                rec_tuples = [(rec['content'].id, rec['ml_score']) for rec in enhanced_recommendations]
                diverse_tuples = self.diversity_optimizer.optimize_diversity(
                    rec_tuples, diversity_factor, limit
                )
                
                diverse_content_ids = [content_id for content_id, _ in diverse_tuples]
                final_recommendations = []
                
                for rec in enhanced_recommendations:
                    if rec['content'].id in diverse_content_ids:
                        rec['diversity_contribution'] = diversity_factor
                        final_recommendations.append(rec)
                        if len(final_recommendations) >= limit:
                            break
                
                return final_recommendations
            else:
                return enhanced_recommendations[:limit]
                
        except Exception as e:
            logger.error(f"Error in advanced recommendations: {e}")
            return []
    
    def _get_hybrid_recommendations(self, user_id, limit, content_type='all'):
        all_recommendations = defaultdict(float)
        algorithm_contributions = defaultdict(lambda: defaultdict(float))
        
        try:
            collab_user_recs = self.collaborative_filtering.user_based_recommendations(user_id, limit * 2)
            for content_id, score in collab_user_recs:
                weighted_score = score * self.algorithm_weights['collaborative_user']
                all_recommendations[content_id] += weighted_score
                algorithm_contributions[content_id]['collaborative_user'] = weighted_score
        except Exception as e:
            logger.warning(f"Collaborative user filtering failed: {e}")
        
        try:
            collab_item_recs = self.collaborative_filtering.item_based_recommendations(user_id, limit * 2)
            for content_id, score in collab_item_recs:
                weighted_score = score * self.algorithm_weights['collaborative_item']
                all_recommendations[content_id] += weighted_score
                algorithm_contributions[content_id]['collaborative_item'] = weighted_score
        except Exception as e:
            logger.warning(f"Collaborative item filtering failed: {e}")
        
        try:
            content_recs = self.content_based_filtering.get_content_recommendations(user_id, limit * 2)
            for content_id, score in content_recs:
                weighted_score = score * self.algorithm_weights['content_based']
                all_recommendations[content_id] += weighted_score
                algorithm_contributions[content_id]['content_based'] = weighted_score
        except Exception as e:
            logger.warning(f"Content-based filtering failed: {e}")
        
        try:
            mf_recs = self.matrix_factorization.get_recommendations(user_id, limit * 2)
            for content_id, score in mf_recs:
                weighted_score = score * self.algorithm_weights['matrix_factorization']
                all_recommendations[content_id] += weighted_score
                algorithm_contributions[content_id]['matrix_factorization'] = weighted_score
        except Exception as e:
            logger.warning(f"Matrix factorization failed: {e}")
        
        try:
            deep_recs = self.deep_learning.get_deep_recommendations(user_id, limit * 2)
            for content_id, score in deep_recs:
                weighted_score = score * self.algorithm_weights['deep_learning']
                all_recommendations[content_id] += weighted_score
                algorithm_contributions[content_id]['deep_learning'] = weighted_score
        except Exception as e:
            logger.warning(f"Deep learning recommendations failed: {e}")
        
        if content_type != 'all':
            filtered_recommendations = []
            for content_id, score in all_recommendations.items():
                content = self.Content.query.get(content_id)
                if content and content.content_type == content_type:
                    filtered_recommendations.append((content_id, score))
            all_recommendations = dict(filtered_recommendations)
        
        sorted_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for content_id, score in sorted_recommendations[:limit]:
            content = self.Content.query.get(content_id)
            if content:
                result.append({
                    'content': content,
                    'score': score,
                    'reason': self._generate_recommendation_reason(user_id, content_id),
                    'algorithm': 'hybrid',
                    'confidence': min(score, 1.0)
                })
        
        return result
    
    def _get_collaborative_recommendations(self, user_id, limit, content_type='all'):
        user_recs = self.collaborative_filtering.user_based_recommendations(user_id, limit // 2)
        item_recs = self.collaborative_filtering.item_based_recommendations(user_id, limit // 2)
        
        combined_recs = defaultdict(float)
        for content_id, score in user_recs:
            combined_recs[content_id] += score * 0.6
        
        for content_id, score in item_recs:
            combined_recs[content_id] += score * 0.4
        
        if content_type != 'all':
            filtered_recs = []
            for content_id, score in combined_recs.items():
                content = self.Content.query.get(content_id)
                if content and content.content_type == content_type:
                    filtered_recs.append((content_id, score))
            combined_recs = dict(filtered_recs)
        
        sorted_recs = sorted(combined_recs.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for content_id, score in sorted_recs[:limit]:
            content = self.Content.query.get(content_id)
            if content:
                result.append({
                    'content': content,
                    'score': score,
                    'reason': 'Based on users with similar preferences',
                    'algorithm': 'collaborative_filtering',
                    'confidence': min(score / 5.0, 1.0)
                })
        
        return result
    
    def _get_content_based_recommendations(self, user_id, limit, content_type='all'):
        recommendations = self.content_based_filtering.get_content_recommendations(user_id, limit * 2)
        
        if content_type != 'all':
            filtered_recs = []
            for content_id, score in recommendations:
                content = self.Content.query.get(content_id)
                if content and content.content_type == content_type:
                    filtered_recs.append((content_id, score))
            recommendations = filtered_recs
        
        result = []
        for content_id, score in recommendations[:limit]:
            content = self.Content.query.get(content_id)
            if content:
                result.append({
                    'content': content,
                    'score': score,
                    'reason': 'Based on content you enjoyed',
                    'algorithm': 'content_based',
                    'confidence': min(score, 1.0)
                })
        
        return result
    
    def _get_matrix_factorization_recommendations(self, user_id, limit, content_type='all'):
        recommendations = self.matrix_factorization.get_recommendations(user_id, limit * 2)
        
        if content_type != 'all':
            filtered_recs = []
            for content_id, score in recommendations:
                content = self.Content.query.get(content_id)
                if content and content.content_type == content_type:
                    filtered_recs.append((content_id, score))
            recommendations = filtered_recs
        
        result = []
        for content_id, score in recommendations[:limit]:
            content = self.Content.query.get(content_id)
            if content:
                result.append({
                    'content': content,
                    'score': score,
                    'reason': 'Based on latent preference patterns',
                    'algorithm': 'matrix_factorization',
                    'confidence': min(abs(score) / 5.0, 1.0)
                })
        
        return result
    
    def _get_deep_learning_recommendations(self, user_id, limit, content_type='all'):
        recommendations = self.deep_learning.get_deep_recommendations(user_id, limit * 2)
        
        if content_type != 'all':
            filtered_recs = []
            for content_id, score in recommendations:
                content = self.Content.query.get(content_id)
                if content and content.content_type == content_type:
                    filtered_recs.append((content_id, score))
            recommendations = filtered_recs
        
        result = []
        for content_id, score in recommendations[:limit]:
            content = self.Content.query.get(content_id)
            if content:
                result.append({
                    'content': content,
                    'score': score,
                    'reason': 'Based on deep learning analysis',
                    'algorithm': 'deep_learning',
                    'confidence': min(score, 1.0)
                })
        
        return result
    
    def _get_cold_start_recommendations(self, user_id, limit, content_type='all'):
        recommendations = self.cold_start_handler.get_cold_start_recommendations(user_id, limit * 2)
        
        if content_type != 'all':
            filtered_recs = []
            for content_id, score in recommendations:
                content = self.Content.query.get(content_id)
                if content and content.content_type == content_type:
                    filtered_recs.append((content_id, score))
            recommendations = filtered_recs
        
        result = []
        for content_id, score in recommendations[:limit]:
            content = self.Content.query.get(content_id)
            if content:
                result.append({
                    'content': content,
                    'score': score,
                    'reason': 'Popular content matching your preferences',
                    'algorithm': 'cold_start',
                    'confidence': 0.5
                })
        
        return result
    
    def _generate_recommendation_reason(self, user_id, content_id):
        content = self.Content.query.get(content_id)
        if not content:
            return "Recommended for you"
        
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not user_interactions:
            return "Popular content that matches your preferences"
        
        user_genres = Counter()
        user_languages = Counter()
        user_types = Counter()
        
        for interaction in user_interactions:
            interacted_content = self.Content.query.get(interaction.content_id)
            if interacted_content:
                try:
                    genres = json.loads(interacted_content.genres or '[]')
                    for genre in genres:
                        user_genres[genre] += 1
                except:
                    pass
                
                try:
                    languages = json.loads(interacted_content.languages or '[]')
                    for language in languages:
                        user_languages[language] += 1
                except:
                    pass
                
                user_types[interacted_content.content_type] += 1
        
        reasons = []
        
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            top_user_genres = set([genre for genre, _ in user_genres.most_common(3)])
            
            matching_genres = content_genres & top_user_genres
            if matching_genres:
                reasons.append(f"You enjoyed {', '.join(list(matching_genres)[:2])} content")
        except:
            pass
        
        try:
            content_languages = set(json.loads(content.languages or '[]'))
            top_user_languages = set([lang for lang, _ in user_languages.most_common(2)])
            
            matching_languages = content_languages & top_user_languages
            if matching_languages:
                reasons.append(f"Matches your language preferences")
        except:
            pass
        
        if content.content_type in user_types:
            reasons.append(f"You watch {content.content_type} content")
        
        if content.rating and content.rating >= 8.0:
            reasons.append("Highly rated content")
        
        if not reasons:
            return "Recommended based on your viewing history"
        
        return "; ".join(reasons[:2])
    
    def _generate_explanation(self, user_id, content_id, include_explanations):
        if not include_explanations:
            return "Personalized recommendation"
        
        content = self.Content.query.get(content_id)
        if not content:
            return "Recommended for you"
        
        explanations = []
        
        similar_content = self._find_similar_watched_content(user_id, content_id)
        if similar_content:
            explanations.append(f"Similar to {similar_content[:1][0].title}")
        
        user_profile = self._get_user_profile_summary(user_id)
        
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            if content_genres & set(user_profile.get('top_genres', [])):
                matching_genre = list(content_genres & set(user_profile.get('top_genres', [])))[0]
                explanations.append(f"Matches your interest in {matching_genre}")
        except:
            pass
        
        if content.rating and content.rating >= 8.5:
            explanations.append("Highly rated by critics and users")
        
        novelty_score = self.novelty_detector.calculate_novelty_score(user_id, content_id)
        if novelty_score > 0.7:
            explanations.append("Something new for you to explore")
        
        if not explanations:
            return "Recommended based on your preferences"
        
        return "; ".join(explanations[:3])
    
    def _get_algorithm_contribution(self, user_id, content_id):
        contributions = {}
        
        try:
            user_sim = self.collaborative_filtering.calculate_user_similarity(user_id)
            if user_sim:
                contributions['collaborative_user'] = min(max(user_sim.values()) * 0.25, 0.25)
        except:
            contributions['collaborative_user'] = 0.0
        
        try:
            user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            if user_interactions:
                for interaction in user_interactions[:5]:
                    item_sim = self.collaborative_filtering.calculate_item_similarity(interaction.content_id)
                    if content_id in item_sim:
                        contributions['collaborative_item'] = max(
                            contributions.get('collaborative_item', 0.0),
                            item_sim[content_id] * 0.20
                        )
                        break
        except:
            contributions['collaborative_item'] = 0.0
        
        try:
            content_similarity = self.content_based_filtering.calculate_content_similarity
            user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            max_content_sim = 0.0
            
            for interaction in user_interactions[:10]:
                sim = content_similarity(content_id, interaction.content_id)
                max_content_sim = max(max_content_sim, sim)
            
            contributions['content_based'] = max_content_sim * 0.20
        except:
            contributions['content_based'] = 0.0
        
        try:
            deep_sim = self.deep_learning.calculate_deep_similarity(user_id, content_id)
            contributions['deep_learning'] = deep_sim * 0.20
        except:
            contributions['deep_learning'] = 0.0
        
        contributions['matrix_factorization'] = 0.15
        
        return contributions
    
    def _find_similar_watched_content(self, user_id, content_id):
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        similar_content = []
        for interaction in user_interactions:
            try:
                similarity = self.content_based_filtering.calculate_content_similarity(
                    content_id, interaction.content_id
                )
                if similarity > 0.5:
                    content = self.Content.query.get(interaction.content_id)
                    if content:
                        similar_content.append(content)
            except:
                continue
        
        return similar_content[:3]
    
    def _get_user_profile_summary(self, user_id):
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        profile = {
            'top_genres': [],
            'top_languages': [],
            'preferred_types': [],
            'avg_rating': 0.0,
            'interaction_count': len(user_interactions)
        }
        
        if not user_interactions:
            self.user_profiles[user_id] = profile
            return profile
        
        genre_counts = Counter()
        language_counts = Counter()
        type_counts = Counter()
        ratings = []
        
        for interaction in user_interactions:
            content = self.Content.query.get(interaction.content_id)
            if content:
                try:
                    genres = json.loads(content.genres or '[]')
                    for genre in genres:
                        weight = self._get_interaction_weight_profile(interaction)
                        genre_counts[genre] += weight
                except:
                    pass
                
                try:
                    languages = json.loads(content.languages or '[]')
                    for language in languages:
                        weight = self._get_interaction_weight_profile(interaction)
                        language_counts[language] += weight
                except:
                    pass
                
                weight = self._get_interaction_weight_profile(interaction)
                type_counts[content.content_type] += weight
                
                if interaction.rating:
                    ratings.append(interaction.rating)
        
        profile['top_genres'] = [genre for genre, _ in genre_counts.most_common(5)]
        profile['top_languages'] = [lang for lang, _ in language_counts.most_common(3)]
        profile['preferred_types'] = [ctype for ctype, _ in type_counts.most_common(3)]
        profile['avg_rating'] = sum(ratings) / len(ratings) if ratings else 0.0
        
        self.user_profiles[user_id] = profile
        return profile
    
    def _get_interaction_weight_profile(self, interaction):
        weights = {
            'view': 1.0,
            'like': 2.0,
            'favorite': 3.0,
            'watchlist': 2.5,
            'rating': interaction.rating * 0.5 if interaction.rating else 2.0,
            'search': 0.5
        }
        
        return weights.get(interaction.interaction_type, 1.0)
    
    def update_user_profile(self, user_id):
        if user_id in self.user_profiles:
            del self.user_profiles[user_id]
        
        self.collaborative_filtering.user_similarity_cache.pop(user_id, None)
        
        self._get_user_profile_summary(user_id)
    
    def get_user_interaction_count(self, user_id):
        return self.UserInteraction.query.filter_by(user_id=user_id).count()
    
    def get_user_profile_strength(self, user_id):
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
    
    def record_recommendation_feedback(self, user_id, content_id, feedback_type, recommendation_id, feedback_value=1.0):
        return self.realtime_learning.update_recommendations_with_feedback(
            user_id, content_id, feedback_type, feedback_value
        )
    
    def get_recommendation_metrics(self, user_id):
        try:
            recent_recommendations = self._get_hybrid_recommendations(user_id, 20)
            
            if not recent_recommendations:
                return {
                    'precision_at_10': 0.0,
                    'ndcg_at_10': 0.0,
                    'diversity_score': 0.0,
                    'user_profile_strength': self.get_user_profile_strength(user_id)
                }
            
            rec_tuples = [(rec['content'].id, rec['score']) for rec in recent_recommendations]
            
            precision = self.evaluation_metrics.calculate_precision_at_k(user_id, rec_tuples, 10)
            ndcg = self.evaluation_metrics.calculate_ndcg_at_k(user_id, rec_tuples, 10)
            diversity = self.evaluation_metrics.calculate_diversity_metric(rec_tuples)
            
            return {
                'precision_at_10': round(precision, 3),
                'ndcg_at_10': round(ndcg, 3),
                'diversity_score': round(diversity, 3),
                'user_profile_strength': self.get_user_profile_strength(user_id),
                'total_interactions': self.get_user_interaction_count(user_id)
            }
            
        except Exception as e:
            logger.error(f"Error calculating recommendation metrics: {e}")
            return {
                'precision_at_10': 0.0,
                'ndcg_at_10': 0.0,
                'diversity_score': 0.0,
                'user_profile_strength': self.get_user_profile_strength(user_id),
                'error': str(e)
            }