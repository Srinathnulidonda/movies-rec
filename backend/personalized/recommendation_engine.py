# backend/personalized/recommendation_engine.py

"""
CineBrain Advanced Personalized Recommendation Engine
AI-powered recommendation system with adaptive algorithms and real-time learning

This module provides:
- Hybrid recommendation strategies with cultural prioritization
- Real-time preference adaptation
- Telugu-first recommendation prioritization
- Multi-algorithm mixing with confidence weighting
- Contextual recommendation generation
- Advanced similarity matching
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import logging
import math
import random
from typing import List, Dict, Any, Tuple, Optional, Set
from sqlalchemy import func, desc, and_, or_

# Import from existing algorithms
from services.algorithms import (
    ContentBasedFiltering,
    CollaborativeFiltering,
    HybridRecommendationEngine,
    UltraPowerfulSimilarityEngine,
    PopularityRanking,
    LanguagePriorityFilter,
    RecommendationOrchestrator,
    AdvancedAlgorithms,
    LANGUAGE_WEIGHTS,
    PRIORITY_LANGUAGES
)

logger = logging.getLogger(__name__)

class AdaptiveAlgorithmMixer:
    """
    Adaptive Algorithm Mixer that intelligently combines multiple recommendation strategies
    based on user profile, context, and real-time feedback
    """
    
    def __init__(self):
        self.algorithm_weights = {
            'cold_start': {
                'popularity': 0.4,
                'language_priority': 0.3,
                'cultural_affinity': 0.2,
                'trending': 0.1
            },
            'content_based_primary': {
                'content_based': 0.5,
                'popularity': 0.2,
                'language_priority': 0.2,
                'cultural_affinity': 0.1
            },
            'content_collaborative_mix': {
                'content_based': 0.35,
                'collaborative': 0.25,
                'popularity': 0.15,
                'language_priority': 0.15,
                'similarity_engine': 0.1
            },
            'advanced_hybrid': {
                'content_based': 0.25,
                'collaborative': 0.25,
                'similarity_engine': 0.2,
                'language_priority': 0.15,
                'popularity': 0.1,
                'serendipity': 0.05
            }
        }
        
        self.telugu_cultural_boost = 1.5
        self.indian_cultural_boost = 1.2
        self.recency_decay_days = 30
        
    def mix_algorithms(self, strategy: str, user_profile: Dict[str, Any], 
                      content_pool: List[Any], algorithms: Dict[str, Any],
                      limit: int = 50) -> List[Tuple[Any, float, str]]:
        """
        Mix multiple algorithms based on strategy and user profile
        
        Args:
            strategy: Recommendation strategy to use
            user_profile: Comprehensive user profile
            content_pool: Available content to recommend from
            algorithms: Dictionary of initialized algorithm instances
            limit: Number of recommendations to generate
            
        Returns:
            List of (content, score, explanation) tuples
        """
        try:
            weights = self.algorithm_weights.get(strategy, self.algorithm_weights['content_based_primary'])
            
            # Generate recommendations from each algorithm
            algorithm_results = {}
            
            # Content-based recommendations
            if 'content_based' in weights:
                algorithm_results['content_based'] = self._get_content_based_recommendations(
                    user_profile, content_pool, algorithms, limit * 2
                )
            
            # Collaborative filtering
            if 'collaborative' in weights:
                algorithm_results['collaborative'] = self._get_collaborative_recommendations(
                    user_profile, content_pool, algorithms, limit * 2
                )
            
            # Ultra similarity engine
            if 'similarity_engine' in weights:
                algorithm_results['similarity_engine'] = self._get_similarity_recommendations(
                    user_profile, content_pool, algorithms, limit * 2
                )
            
            # Popularity-based
            if 'popularity' in weights:
                algorithm_results['popularity'] = self._get_popularity_recommendations(
                    content_pool, algorithms, limit * 2
                )
            
            # Language priority
            if 'language_priority' in weights:
                algorithm_results['language_priority'] = self._get_language_priority_recommendations(
                    user_profile, content_pool, algorithms, limit * 2
                )
            
            # Cultural affinity
            if 'cultural_affinity' in weights:
                algorithm_results['cultural_affinity'] = self._get_cultural_recommendations(
                    user_profile, content_pool, limit * 2
                )
            
            # Trending content
            if 'trending' in weights:
                algorithm_results['trending'] = self._get_trending_recommendations(
                    content_pool, algorithms, limit * 2
                )
            
            # Serendipity (discovery)
            if 'serendipity' in weights:
                algorithm_results['serendipity'] = self._get_serendipity_recommendations(
                    user_profile, content_pool, algorithms, limit
                )
            
            # Combine results using weighted scoring
            combined_results = self._combine_algorithm_results(
                algorithm_results, weights, user_profile, limit
            )
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error mixing algorithms: {e}")
            return []
    
    def _get_content_based_recommendations(self, user_profile: Dict[str, Any], 
                                         content_pool: List[Any], algorithms: Dict[str, Any],
                                         limit: int) -> List[Tuple[Any, float, str]]:
        """Get content-based recommendations"""
        try:
            content_based = algorithms.get('content_based') or ContentBasedFiltering()
            
            # Create preference profile for content-based algorithm
            cb_profile = {
                'preferred_genres': self._extract_preferred_genres(user_profile),
                'preferred_languages': self._extract_preferred_languages(user_profile),
                'avg_rating': self._get_average_rating_preference(user_profile),
                'user_id': user_profile['user_id']
            }
            
            recommendations = content_based.get_recommendations(cb_profile, content_pool, limit)
            
            # Add explanations
            results = []
            for content, score in recommendations:
                explanation = f"Matches your genre preferences and viewing history"
                results.append((content, score, explanation))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting content-based recommendations: {e}")
            return []
    
    def _get_collaborative_recommendations(self, user_profile: Dict[str, Any],
                                         content_pool: List[Any], algorithms: Dict[str, Any],
                                         limit: int) -> List[Tuple[Any, float, str]]:
        """Get collaborative filtering recommendations"""
        try:
            collaborative = algorithms.get('collaborative') or CollaborativeFiltering()
            
            # This would need user ratings data - placeholder for now
            user_ratings = {}  # Would be populated from database
            
            recommendations = collaborative.user_based_cf(
                user_profile['user_id'], user_ratings, content_pool
            )
            
            results = []
            for content, score in recommendations[:limit]:
                explanation = f"Recommended by users with similar taste"
                results.append((content, score, explanation))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {e}")
            return []
    
    def _get_similarity_recommendations(self, user_profile: Dict[str, Any],
                                      content_pool: List[Any], algorithms: Dict[str, Any],
                                      limit: int) -> List[Tuple[Any, float, str]]:
        """Get ultra-similarity based recommendations"""
        try:
            similarity_engine = algorithms.get('similarity_engine') or UltraPowerfulSimilarityEngine()
            
            # Get user's recent high-rated content
            recent_favorites = self._get_recent_favorites(user_profile)
            
            if not recent_favorites:
                return []
            
            all_results = []
            for base_content_id in recent_favorites[:3]:  # Use top 3 favorites
                base_content = next((c for c in content_pool if c.id == base_content_id), None)
                if base_content:
                    similar_results = similarity_engine.find_ultra_similar_content(
                        base_content, content_pool, limit=15, min_similarity=0.6
                    )
                    
                    for result in similar_results:
                        content = result['content']
                        score = result['similarity_score']
                        explanation = f"Similar to {base_content.title} - {result['match_type']}"
                        all_results.append((content, score, explanation))
            
            # Remove duplicates and sort by score
            seen_ids = set()
            unique_results = []
            for content, score, explanation in sorted(all_results, key=lambda x: x[1], reverse=True):
                if content.id not in seen_ids:
                    seen_ids.add(content.id)
                    unique_results.append((content, score, explanation))
                    if len(unique_results) >= limit:
                        break
            
            return unique_results
            
        except Exception as e:
            logger.error(f"Error getting similarity recommendations: {e}")
            return []
    
    def _get_popularity_recommendations(self, content_pool: List[Any],
                                      algorithms: Dict[str, Any], limit: int) -> List[Tuple[Any, float, str]]:
        """Get popularity-based recommendations"""
        try:
            popularity_scores = PopularityRanking.calculate_popularity_scores(content_pool)
            
            results = []
            for content, score in popularity_scores[:limit]:
                explanation = f"Popular content trending now"
                results.append((content, score, explanation))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting popularity recommendations: {e}")
            return []
    
    def _get_language_priority_recommendations(self, user_profile: Dict[str, Any],
                                             content_pool: List[Any], algorithms: Dict[str, Any],
                                             limit: int) -> List[Tuple[Any, float, str]]:
        """Get language priority recommendations"""
        try:
            preferred_languages = self._extract_preferred_languages(user_profile)
            
            language_scores = LanguagePriorityFilter.apply_language_scores(
                content_pool, preferred_languages
            )
            
            results = []
            for content, score in language_scores[:limit]:
                primary_lang = self._get_primary_language(content)
                explanation = f"In your preferred language: {primary_lang}"
                results.append((content, score, explanation))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting language priority recommendations: {e}")
            return []
    
    def _get_cultural_recommendations(self, user_profile: Dict[str, Any],
                                    content_pool: List[Any], limit: int) -> List[Tuple[Any, float, str]]:
        """Get culturally aligned recommendations"""
        try:
            cinematic_dna = user_profile.get('cinematic_dna', {})
            telugu_affinity = cinematic_dna.get('telugu_cultural_affinity', 0.5)
            indian_affinity = cinematic_dna.get('indian_cultural_affinity', 0.5)
            
            cultural_scores = []
            
            for content in content_pool:
                score = 0.0
                explanation = ""
                
                languages = json.loads(content.languages or '[]')
                
                # Telugu content bonus
                if any('telugu' in lang.lower() for lang in languages):
                    score += telugu_affinity * self.telugu_cultural_boost
                    explanation = "Telugu content matching your cultural preferences"
                
                # Other Indian languages
                elif any(lang.lower() in ['hindi', 'tamil', 'malayalam', 'kannada'] for lang in languages):
                    score += indian_affinity * self.indian_cultural_boost
                    explanation = "Indian content matching your cultural preferences"
                
                if score > 0:
                    cultural_scores.append((content, score, explanation))
            
            cultural_scores.sort(key=lambda x: x[1], reverse=True)
            return cultural_scores[:limit]
            
        except Exception as e:
            logger.error(f"Error getting cultural recommendations: {e}")
            return []
    
    def _get_trending_recommendations(self, content_pool: List[Any],
                                    algorithms: Dict[str, Any], limit: int) -> List[Tuple[Any, float, str]]:
        """Get trending content recommendations"""
        try:
            trending_scores = []
            
            for content in content_pool:
                if content.is_trending:
                    score = PopularityRanking.calculate_trending_score(content)
                    trending_scores.append((content, score, "Currently trending content"))
            
            trending_scores.sort(key=lambda x: x[1], reverse=True)
            return trending_scores[:limit]
            
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    def _get_serendipity_recommendations(self, user_profile: Dict[str, Any],
                                       content_pool: List[Any], algorithms: Dict[str, Any],
                                       limit: int) -> List[Tuple[Any, float, str]]:
        """Get serendipitous (discovery) recommendations"""
        try:
            user_genres = set(self._extract_preferred_genres(user_profile))
            serendipity_scores = []
            
            for content in content_pool:
                if content.genres:
                    try:
                        content_genres = set(json.loads(content.genres))
                        
                        # Score content outside usual preferences but high quality
                        if not content_genres.intersection(user_genres):
                            if content.rating and content.rating >= 7.5:
                                serendipity_score = AdvancedAlgorithms.calculate_serendipity_score(
                                    content, {
                                        'preferred_genres': list(user_genres),
                                        'preferred_languages': self._extract_preferred_languages(user_profile)
                                    }
                                )
                                explanation = f"Hidden gem outside your usual preferences"
                                serendipity_scores.append((content, serendipity_score, explanation))
                    except:
                        pass
            
            serendipity_scores.sort(key=lambda x: x[1], reverse=True)
            return serendipity_scores[:limit]
            
        except Exception as e:
            logger.error(f"Error getting serendipity recommendations: {e}")
            return []
    
    def _combine_algorithm_results(self, algorithm_results: Dict[str, List[Tuple[Any, float, str]]],
                                 weights: Dict[str, float], user_profile: Dict[str, Any],
                                 limit: int) -> List[Tuple[Any, float, str]]:
        """Combine results from multiple algorithms using weighted scoring"""
        try:
            combined_scores = defaultdict(lambda: {'score': 0.0, 'explanations': [], 'content': None})
            
            # Apply weights to each algorithm's results
            for algorithm, results in algorithm_results.items():
                weight = weights.get(algorithm, 0.0)
                
                for content, score, explanation in results:
                    content_id = content.id
                    combined_scores[content_id]['score'] += score * weight
                    combined_scores[content_id]['explanations'].append(f"{algorithm}: {explanation}")
                    combined_scores[content_id]['content'] = content
            
            # Apply cultural and language bonuses
            for content_id, data in combined_scores.items():
                content = data['content']
                bonus = self._calculate_cultural_bonus(content, user_profile)
                data['score'] *= (1 + bonus)
            
            # Convert to list and sort
            final_results = []
            for content_id, data in combined_scores.items():
                content = data['content']
                score = data['score']
                
                # Create combined explanation
                primary_explanation = data['explanations'][0] if data['explanations'] else "Personalized for you"
                if len(data['explanations']) > 1:
                    primary_explanation += f" (+{len(data['explanations'])-1} more factors)"
                
                final_results.append((content, score, primary_explanation))
            
            # Sort by score and apply diversity
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            # Apply diversity injection
            diverse_results = self._apply_diversity_injection(final_results, limit)
            
            return diverse_results[:limit]
            
        except Exception as e:
            logger.error(f"Error combining algorithm results: {e}")
            return []
    
    def _calculate_cultural_bonus(self, content: Any, user_profile: Dict[str, Any]) -> float:
        """Calculate cultural alignment bonus"""
        cinematic_dna = user_profile.get('cinematic_dna', {})
        languages = json.loads(content.languages or '[]')
        
        bonus = 0.0
        
        # Telugu bonus
        if any('telugu' in lang.lower() for lang in languages):
            telugu_affinity = cinematic_dna.get('telugu_cultural_affinity', 0.5)
            bonus += telugu_affinity * 0.3
        
        # Indian content bonus
        indian_languages = ['hindi', 'tamil', 'malayalam', 'kannada']
        if any(any(lang_name in lang.lower() for lang_name in indian_languages) for lang in languages):
            indian_affinity = cinematic_dna.get('indian_cultural_affinity', 0.5)
            bonus += indian_affinity * 0.2
        
        return bonus
    
    def _apply_diversity_injection(self, recommendations: List[Tuple[Any, float, str]],
                                 limit: int) -> List[Tuple[Any, float, str]]:
        """Apply diversity to avoid repetitive recommendations"""
        if len(recommendations) <= limit:
            return recommendations
        
        diverse_recs = []
        seen_genres = set()
        seen_languages = set()
        genre_counts = defaultdict(int)
        
        # Always include top 30% without modification
        guaranteed_count = max(1, int(limit * 0.3))
        diverse_recs.extend(recommendations[:guaranteed_count])
        
        # Update tracking
        for content, _, _ in diverse_recs:
            try:
                genres = set(json.loads(content.genres or '[]'))
                languages = set(json.loads(content.languages or '[]'))
                seen_genres.update(genres)
                seen_languages.update(languages)
                for genre in genres:
                    genre_counts[genre] += 1
            except:
                pass
        
        # Add remaining with diversity consideration
        for content, score, explanation in recommendations[guaranteed_count:]:
            if len(diverse_recs) >= limit:
                break
            
            try:
                genres = set(json.loads(content.genres or '[]'))
                languages = set(json.loads(content.languages or '[]'))
                
                # Calculate diversity value
                new_genres = len(genres - seen_genres)
                new_languages = len(languages - seen_languages)
                
                # Check if genres are over-represented
                over_represented = any(genre_counts[g] >= limit * 0.4 for g in genres)
                
                # Include if adds diversity or has very high score
                if (new_genres > 0 or new_languages > 0) or score > 0.8 or not over_represented:
                    diverse_recs.append((content, score, explanation))
                    seen_genres.update(genres)
                    seen_languages.update(languages)
                    for genre in genres:
                        genre_counts[genre] += 1
            except:
                # Include if we can't parse metadata
                diverse_recs.append((content, score, explanation))
        
        return diverse_recs
    
    # Helper methods
    def _extract_preferred_genres(self, user_profile: Dict[str, Any]) -> List[str]:
        """Extract preferred genres from user profile"""
        cinematic_dna = user_profile.get('cinematic_dna', {})
        genre_sophistication = cinematic_dna.get('genre_sophistication', {})
        
        # Get genres with high sophistication scores
        preferred = []
        for genre, data in genre_sophistication.items():
            if isinstance(data, dict) and data.get('sophistication_score', 0) > 0.6:
                preferred.append(genre)
        
        return preferred[:5]  # Top 5 genres
    
    def _extract_preferred_languages(self, user_profile: Dict[str, Any]) -> List[str]:
        """Extract preferred languages from user profile"""
        cinematic_dna = user_profile.get('cinematic_dna', {})
        cultural_alignment = cinematic_dna.get('cultural_alignment', {})
        
        languages = ['Telugu']  # Always start with Telugu
        
        # Add based on cultural affinity
        if cultural_alignment.get('indian_mainstream', 0) > 0.5:
            languages.extend(['Hindi', 'Tamil'])
        
        if cultural_alignment.get('global_blockbuster', 0) > 0.5:
            languages.append('English')
        
        return languages
    
    def _get_average_rating_preference(self, user_profile: Dict[str, Any]) -> float:
        """Get user's average rating preference"""
        behavior_profile = user_profile.get('behavior_profile', {})
        rating_behavior = behavior_profile.get('rating_behavior', {})
        
        return rating_behavior.get('average_rating', 7.0)
    
    def _get_recent_favorites(self, user_profile: Dict[str, Any]) -> List[int]:
        """Get user's recent favorite content IDs"""
        # This would typically query the database for recent favorites
        # For now, return empty list - would be implemented with actual DB queries
        return []
    
    def _get_primary_language(self, content: Any) -> str:
        """Get primary language of content"""
        try:
            languages = json.loads(content.languages or '[]')
            return languages[0] if languages else 'Unknown'
        except:
            return 'Unknown'

class ContentPersonalizer:
    """
    Content Personalizer that adapts content presentation and filtering
    based on individual user preferences
    """
    
    def __init__(self):
        self.personalization_factors = {
            'cultural_alignment': 0.3,
            'quality_preference': 0.25,
            'recency_preference': 0.2,
            'genre_affinity': 0.15,
            'language_priority': 0.1
        }
    
    def personalize_content_list(self, content_list: List[Any], user_profile: Dict[str, Any],
                               context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Personalize content list for individual user
        
        Args:
            content_list: List of content items to personalize
            user_profile: User's comprehensive profile
            context: Additional context (time, device, etc.)
            
        Returns:
            Personalized and formatted content list
        """
        try:
            personalized_items = []
            
            for content in content_list:
                personalized_item = self._personalize_single_content(content, user_profile, context)
                if personalized_item:
                    personalized_items.append(personalized_item)
            
            # Sort by personalization score
            personalized_items.sort(key=lambda x: x['personalization_score'], reverse=True)
            
            return personalized_items
            
        except Exception as e:
            logger.error(f"Error personalizing content list: {e}")
            return self._format_content_list_fallback(content_list)
    
    def _personalize_single_content(self, content: Any, user_profile: Dict[str, Any],
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Personalize a single content item"""
        try:
            # Calculate personalization score
            personalization_score = self._calculate_personalization_score(content, user_profile)
            
            # Generate personalized explanation
            explanation = self._generate_personalized_explanation(content, user_profile)
            
            # Determine recommendation strength
            rec_strength = self._determine_recommendation_strength(personalization_score)
            
            # Format content with personalization
            formatted_content = {
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'rating': content.rating,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'poster_path': self._format_poster_path(content.poster_path),
                'overview': content.overview[:200] + '...' if content.overview else '',
                'youtube_trailer_id': content.youtube_trailer_id,
                'is_trending': content.is_trending,
                'is_new_release': content.is_new_release,
                
                # Personalization fields
                'personalization_score': round(personalization_score, 3),
                'recommendation_strength': rec_strength,
                'personalized_explanation': explanation,
                'cultural_match': self._assess_cultural_match(content, user_profile),
                'quality_match': self._assess_quality_match(content, user_profile),
                'language_priority_rank': self._get_language_priority_rank(content),
                'why_recommended': self._generate_why_recommended(content, user_profile),
                
                # User-specific flags
                'highly_recommended': personalization_score > 0.8,
                'cultural_favorite': self._is_cultural_favorite(content, user_profile),
                'hidden_gem': self._is_hidden_gem(content, user_profile),
                'perfect_match': personalization_score > 0.9
            }
            
            return formatted_content
            
        except Exception as e:
            logger.error(f"Error personalizing content {content.id}: {e}")
            return None
    
    def _calculate_personalization_score(self, content: Any, user_profile: Dict[str, Any]) -> float:
        """Calculate overall personalization score for content"""
        try:
            score = 0.0
            
            # Cultural alignment factor
            cultural_score = self._calculate_cultural_alignment_score(content, user_profile)
            score += cultural_score * self.personalization_factors['cultural_alignment']
            
            # Quality preference factor
            quality_score = self._calculate_quality_preference_score(content, user_profile)
            score += quality_score * self.personalization_factors['quality_preference']
            
            # Recency preference factor
            recency_score = self._calculate_recency_preference_score(content, user_profile)
            score += recency_score * self.personalization_factors['recency_preference']
            
            # Genre affinity factor
            genre_score = self._calculate_genre_affinity_score(content, user_profile)
            score += genre_score * self.personalization_factors['genre_affinity']
            
            # Language priority factor
            language_score = self._calculate_language_priority_score(content, user_profile)
            score += language_score * self.personalization_factors['language_priority']
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating personalization score: {e}")
            return 0.5
    
    def _calculate_cultural_alignment_score(self, content: Any, user_profile: Dict[str, Any]) -> float:
        """Calculate cultural alignment score"""
        cinematic_dna = user_profile.get('cinematic_dna', {})
        cultural_alignment = cinematic_dna.get('cultural_alignment', {})
        
        languages = json.loads(content.languages or '[]')
        score = 0.0
        
        # Telugu content alignment
        if any('telugu' in lang.lower() for lang in languages):
            telugu_traditional = cultural_alignment.get('telugu_traditional', 0.5)
            score += telugu_traditional * 1.0
        
        # Indian content alignment
        indian_languages = ['hindi', 'tamil', 'malayalam', 'kannada']
        if any(any(lang_name in lang.lower() for lang_name in indian_languages) for lang in languages):
            indian_mainstream = cultural_alignment.get('indian_mainstream', 0.5)
            score += indian_mainstream * 0.8
        
        # Western content alignment
        if any('english' in lang.lower() for lang in languages):
            western_commercial = cultural_alignment.get('western_commercial', 0.5)
            global_blockbuster = cultural_alignment.get('global_blockbuster', 0.5)
            score += max(western_commercial, global_blockbuster) * 0.6
        
        return min(score, 1.0)
    
    def _calculate_quality_preference_score(self, content: Any, user_profile: Dict[str, Any]) -> float:
        """Calculate quality preference alignment score"""
        behavior_profile = user_profile.get('behavior_profile', {})
        rating_behavior = behavior_profile.get('rating_behavior', {})
        
        if not rating_behavior.get('has_ratings', False) or not content.rating:
            return 0.7  # Neutral score
        
        user_avg_rating = rating_behavior.get('average_rating', 7.0)
        content_rating = content.rating
        
        # Calculate how well content rating aligns with user preferences
        rating_diff = abs(user_avg_rating - content_rating)
        
        if rating_diff <= 0.5:
            return 1.0
        elif rating_diff <= 1.0:
            return 0.8
        elif rating_diff <= 1.5:
            return 0.6
        elif rating_diff <= 2.0:
            return 0.4
        else:
            return 0.2
    
    def _calculate_recency_preference_score(self, content: Any, user_profile: Dict[str, Any]) -> float:
        """Calculate recency preference score"""
        if not content.release_date:
            return 0.5
        
        behavior_profile = user_profile.get('behavior_profile', {})
        
        # Calculate content age
        current_year = datetime.now().year
        content_year = content.release_date.year
        age_years = current_year - content_year
        
        # User recency preference from behavior
        # This would be calculated from user's viewing patterns
        # For now, use a general preference
        
        if age_years <= 1:
            return 1.0  # Very recent
        elif age_years <= 3:
            return 0.8  # Recent
        elif age_years <= 5:
            return 0.6  # Moderately recent
        elif age_years <= 10:
            return 0.4  # Older but not ancient
        else:
            return 0.2  # Classic content
    
    def _calculate_genre_affinity_score(self, content: Any, user_profile: Dict[str, Any]) -> float:
        """Calculate genre affinity score"""
        if not content.genres:
            return 0.5
        
        try:
            content_genres = set(json.loads(content.genres))
            cinematic_dna = user_profile.get('cinematic_dna', {})
            genre_sophistication = cinematic_dna.get('genre_sophistication', {})
            
            total_affinity = 0.0
            genre_count = 0
            
            for genre in content_genres:
                if genre in genre_sophistication:
                    genre_data = genre_sophistication[genre]
                    if isinstance(genre_data, dict):
                        affinity = genre_data.get('sophistication_score', 0.5)
                        total_affinity += affinity
                        genre_count += 1
            
            if genre_count > 0:
                return total_affinity / genre_count
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating genre affinity: {e}")
            return 0.5
    
    def _calculate_language_priority_score(self, content: Any, user_profile: Dict[str, Any]) -> float:
        """Calculate language priority score"""
        try:
            languages = json.loads(content.languages or '[]')
            if not languages:
                return 0.3
            
            # Check against priority order
            for i, priority_lang in enumerate(PRIORITY_LANGUAGES):
                for content_lang in languages:
                    if priority_lang in content_lang.lower():
                        # Higher score for higher priority languages
                        return 1.0 - (i * 0.15)
            
            return 0.2  # Low score for non-priority languages
            
        except Exception as e:
            logger.error(f"Error calculating language priority score: {e}")
            return 0.5
    
    def _generate_personalized_explanation(self, content: Any, user_profile: Dict[str, Any]) -> str:
        """Generate personalized explanation for why content is recommended"""
        explanations = []
        
        # Cultural alignment
        if self._is_cultural_favorite(content, user_profile):
            explanations.append("matches your cultural preferences")
        
        # Genre alignment
        genres = json.loads(content.genres or '[]')
        preferred_genres = self._get_user_preferred_genres(user_profile)
        matching_genres = set(genres) & set(preferred_genres)
        if matching_genres:
            explanations.append(f"you enjoy {list(matching_genres)[0]} content")
        
        # Quality alignment
        behavior_profile = user_profile.get('behavior_profile', {})
        rating_behavior = behavior_profile.get('rating_behavior', {})
        if rating_behavior.get('has_ratings', False) and content.rating:
            user_avg = rating_behavior.get('average_rating', 7.0)
            if abs(content.rating - user_avg) <= 1.0:
                explanations.append("aligns with your quality standards")
        
        # Language preference
        languages = json.loads(content.languages or '[]')
        if any('telugu' in lang.lower() for lang in languages):
            explanations.append("in your preferred Telugu language")
        
        if explanations:
            return f"Recommended because it {' and '.join(explanations[:2])}"
        else:
            return "Personalized recommendation based on your viewing history"
    
    def _determine_recommendation_strength(self, score: float) -> str:
        """Determine recommendation strength based on score"""
        if score >= 0.9:
            return 'perfect_match'
        elif score >= 0.8:
            return 'highly_recommended'
        elif score >= 0.7:
            return 'recommended'
        elif score >= 0.6:
            return 'might_like'
        else:
            return 'worth_exploring'
    
    def _assess_cultural_match(self, content: Any, user_profile: Dict[str, Any]) -> str:
        """Assess cultural match level"""
        cinematic_dna = user_profile.get('cinematic_dna', {})
        telugu_affinity = cinematic_dna.get('telugu_cultural_affinity', 0.5)
        indian_affinity = cinematic_dna.get('indian_cultural_affinity', 0.5)
        
        languages = json.loads(content.languages or '[]')
        
        if any('telugu' in lang.lower() for lang in languages):
            if telugu_affinity > 0.8:
                return 'perfect_cultural_match'
            elif telugu_affinity > 0.6:
                return 'strong_cultural_match'
            else:
                return 'moderate_cultural_match'
        
        indian_languages = ['hindi', 'tamil', 'malayalam', 'kannada']
        if any(any(lang_name in lang.lower() for lang_name in indian_languages) for lang in languages):
            if indian_affinity > 0.7:
                return 'strong_cultural_match'
            else:
                return 'moderate_cultural_match'
        
        return 'neutral_cultural_match'
    
    def _assess_quality_match(self, content: Any, user_profile: Dict[str, Any]) -> str:
        """Assess quality match level"""
        if not content.rating:
            return 'unrated'
        
        behavior_profile = user_profile.get('behavior_profile', {})
        rating_behavior = behavior_profile.get('rating_behavior', {})
        
        if not rating_behavior.get('has_ratings', False):
            return 'unknown_preference'
        
        user_avg = rating_behavior.get('average_rating', 7.0)
        diff = abs(content.rating - user_avg)
        
        if diff <= 0.5:
            return 'perfect_quality_match'
        elif diff <= 1.0:
            return 'good_quality_match'
        elif diff <= 1.5:
            return 'acceptable_quality_match'
        else:
            return 'quality_mismatch'
    
    def _get_language_priority_rank(self, content: Any) -> int:
        """Get language priority rank for content"""
        try:
            languages = json.loads(content.languages or '[]')
            
            for i, priority_lang in enumerate(PRIORITY_LANGUAGES):
                for content_lang in languages:
                    if priority_lang in content_lang.lower():
                        return i + 1
            
            return 999  # Very low priority
            
        except Exception as e:
            return 999
    
    def _generate_why_recommended(self, content: Any, user_profile: Dict[str, Any]) -> List[str]:
        """Generate list of reasons why content is recommended"""
        reasons = []
        
        # Cultural reasons
        if self._is_cultural_favorite(content, user_profile):
            languages = json.loads(content.languages or '[]')
            if any('telugu' in lang.lower() for lang in languages):
                reasons.append("Telugu content preference")
            else:
                reasons.append("Cultural content preference")
        
        # Genre reasons
        genres = json.loads(content.genres or '[]')
        preferred_genres = self._get_user_preferred_genres(user_profile)
        matching_genres = set(genres) & set(preferred_genres)
        if matching_genres:
            reasons.append(f"You enjoy {list(matching_genres)[0]} genre")
        
        # Quality reasons
        if content.rating and content.rating >= 8.0:
            reasons.append("High-quality content")
        
        # Trending reasons
        if content.is_trending:
            reasons.append("Currently trending")
        
        # New release reasons
        if content.is_new_release:
            reasons.append("Recent release")
        
        return reasons[:3]  # Return top 3 reasons
    
    def _is_cultural_favorite(self, content: Any, user_profile: Dict[str, Any]) -> bool:
        """Check if content aligns with user's cultural preferences"""
        cinematic_dna = user_profile.get('cinematic_dna', {})
        telugu_affinity = cinematic_dna.get('telugu_cultural_affinity', 0.5)
        
        languages = json.loads(content.languages or '[]')
        
        # Strong Telugu affinity and Telugu content
        if telugu_affinity > 0.7 and any('telugu' in lang.lower() for lang in languages):
            return True
        
        return False
    
    def _is_hidden_gem(self, content: Any, user_profile: Dict[str, Any]) -> bool:
        """Check if content is a hidden gem for the user"""
        # High rating but not very popular
        if content.rating and content.rating >= 8.0:
            if not content.popularity or content.popularity < 20:
                return True
        
        return False
    
    def _get_user_preferred_genres(self, user_profile: Dict[str, Any]) -> List[str]:
        """Get user's preferred genres"""
        cinematic_dna = user_profile.get('cinematic_dna', {})
        genre_sophistication = cinematic_dna.get('genre_sophistication', {})
        
        preferred = []
        for genre, data in genre_sophistication.items():
            if isinstance(data, dict) and data.get('sophistication_score', 0) > 0.6:
                preferred.append(genre)
        
        return preferred
    
    def _format_poster_path(self, poster_path: str) -> str:
        """Format poster path for display"""
        if not poster_path:
            return None
        
        if poster_path.startswith('http'):
            return poster_path
        else:
            return f"https://image.tmdb.org/t/p/w300{poster_path}"
    
    def _format_content_list_fallback(self, content_list: List[Any]) -> List[Dict[str, Any]]:
        """Fallback content formatting"""
        formatted = []
        for content in content_list:
            formatted.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'rating': content.rating,
                'poster_path': self._format_poster_path(content.poster_path),
                'personalization_score': 0.5,
                'recommendation_strength': 'standard'
            })
        return formatted

class PersonalizedRecommendationEngine:
    """
    Main Personalized Recommendation Engine orchestrating all components
    """
    
    def __init__(self, db, models, services, profile_analyzer, cache=None):
        self.db = db
        self.models = models
        self.services = services
        self.profile_analyzer = profile_analyzer
        self.cache = cache
        
        # Initialize components
        self.algorithm_mixer = AdaptiveAlgorithmMixer()
        self.content_personalizer = ContentPersonalizer()
        
        # Initialize existing algorithms
        self.algorithms = {
            'content_based': ContentBasedFiltering(),
            'collaborative': CollaborativeFiltering(),
            'hybrid': HybridRecommendationEngine(),
            'similarity_engine': UltraPowerfulSimilarityEngine(),
            'orchestrator': RecommendationOrchestrator()
        }
        
        # Real-time learning settings
        self.enable_real_time = False
        self.feedback_buffer = []
        self.regional_config = {}
        
    def configure_regional_priorities(self, config: Dict[str, Any]):
        """Configure regional and cultural priorities"""
        self.regional_config = config
        logger.info(f"Configured regional priorities: {config}")
    
    def enable_real_time_learning(self):
        """Enable real-time learning from user feedback"""
        self.enable_real_time = True
        logger.info("Real-time learning enabled")
    
    def get_personalized_recommendations(self, user_id: int, recommendation_type: str = 'for_you',
                                       limit: int = 50, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get personalized recommendations for a user
        
        Args:
            user_id: User ID
            recommendation_type: Type of recommendations (for_you, discover, mix, etc.)
            limit: Number of recommendations
            context: Additional context (device, time, etc.)
            
        Returns:
            Personalized recommendations with metadata
        """
        try:
            # Build/get user profile
            user_profile = self.profile_analyzer.build_comprehensive_profile(user_id)
            
            if not user_profile:
                return self._get_cold_start_recommendations(user_id, limit)
            
            # Get content pool
            content_pool = self._get_content_pool(user_id, user_profile, context)
            
            # Determine recommendation strategy
            strategy = user_profile.get('recommendations_strategy', 'content_based_primary')
            
            # Generate recommendations using algorithm mixer
            raw_recommendations = self.algorithm_mixer.mix_algorithms(
                strategy, user_profile, content_pool, self.algorithms, limit * 2
            )
            
            # Apply content personalization
            personalized_content = []
            for content, score, explanation in raw_recommendations:
                personalized_item = self.content_personalizer._personalize_single_content(
                    content, user_profile, context
                )
                if personalized_item:
                    personalized_item['algorithm_explanation'] = explanation
                    personalized_item['combined_score'] = score
                    personalized_content.append(personalized_item)
            
            # Sort by personalization score
            personalized_content.sort(key=lambda x: x['personalization_score'], reverse=True)
            
            # Apply final filtering based on recommendation type
            final_recommendations = self._apply_type_specific_filtering(
                personalized_content, recommendation_type, user_profile, limit
            )
            
            # Generate metadata
            metadata = self._generate_recommendation_metadata(
                user_profile, recommendation_type, strategy, len(final_recommendations)
            )
            
            return {
                'success': True,
                'user_id': user_id,
                'recommendation_type': recommendation_type,
                'recommendations': final_recommendations,
                'metadata': metadata,
                'profile_insights': self._generate_profile_insights(user_profile),
                'next_recommendations_available_at': (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating personalized recommendations for user {user_id}: {e}")
            return self._get_error_fallback_recommendations(user_id, limit)
    
    def get_discovery_recommendations(self, user_id: int, limit: int = 30) -> Dict[str, Any]:
        """Get discovery recommendations to help user explore new content"""
        try:
            user_profile = self.profile_analyzer.build_comprehensive_profile(user_id)
            
            if not user_profile:
                return self._get_cold_start_recommendations(user_id, limit)
            
            # Get diverse content pool for discovery
            content_pool = self._get_discovery_content_pool(user_profile)
            
            # Use serendipity-focused algorithm mixing
            discovery_strategy = 'advanced_hybrid'  # More diverse mixing
            
            raw_recommendations = self.algorithm_mixer.mix_algorithms(
                discovery_strategy, user_profile, content_pool, self.algorithms, limit * 2
            )
            
            # Apply discovery-specific personalization
            discovery_recommendations = []
            for content, score, explanation in raw_recommendations:
                personalized_item = self.content_personalizer._personalize_single_content(
                    content, user_profile
                )
                if personalized_item:
                    # Add discovery-specific fields
                    personalized_item['discovery_reason'] = self._generate_discovery_reason(content, user_profile)
                    personalized_item['exploration_factor'] = self._calculate_exploration_factor(content, user_profile)
                    discovery_recommendations.append(personalized_item)
            
            # Sort by exploration potential
            discovery_recommendations.sort(
                key=lambda x: x['exploration_factor'] * x['personalization_score'], 
                reverse=True
            )
            
            return {
                'success': True,
                'user_id': user_id,
                'recommendation_type': 'discovery',
                'recommendations': discovery_recommendations[:limit],
                'discovery_insights': {
                    'exploration_areas': self._identify_exploration_areas(user_profile),
                    'comfort_zone_expansion': self._assess_comfort_zone_expansion(user_profile),
                    'discovery_readiness': user_profile.get('behavior_profile', {}).get('content_exploration', {}).get('exploration_tendency', 'medium')
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating discovery recommendations for user {user_id}: {e}")
            return self._get_error_fallback_recommendations(user_id, limit)
    
    def get_mixed_recommendations(self, user_id: int, limit: int = 40) -> Dict[str, Any]:
        """Get mixed recommendations combining different strategies"""
        try:
            user_profile = self.profile_analyzer.build_comprehensive_profile(user_id)
            
            if not user_profile:
                return self._get_cold_start_recommendations(user_id, limit)
            
            # Get multiple recommendation types
            safe_recs = self.get_personalized_recommendations(user_id, 'for_you', limit // 2)
            discovery_recs = self.get_discovery_recommendations(user_id, limit // 4)
            trending_recs = self._get_trending_for_user(user_id, limit // 4)
            
            # Combine and interleave
            mixed_recommendations = self._interleave_recommendations([
                safe_recs.get('recommendations', []),
                discovery_recs.get('recommendations', []),
                trending_recs
            ])
            
            return {
                'success': True,
                'user_id': user_id,
                'recommendation_type': 'mixed',
                'recommendations': mixed_recommendations[:limit],
                'mix_composition': {
                    'personalized': len(safe_recs.get('recommendations', [])),
                    'discovery': len(discovery_recs.get('recommendations', [])),
                    'trending': len(trending_recs)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating mixed recommendations for user {user_id}: {e}")
            return self._get_error_fallback_recommendations(user_id, limit)
    
    def process_user_feedback(self, user_id: int, feedback_data: Dict[str, Any]) -> bool:
        """Process user feedback for real-time learning"""
        try:
            if not self.enable_real_time:
                return False
            
            # Add to feedback buffer
            feedback_entry = {
                'user_id': user_id,
                'timestamp': datetime.utcnow(),
                'content_id': feedback_data.get('content_id'),
                'feedback_type': feedback_data.get('feedback_type'),  # like, dislike, watch, skip
                'context': feedback_data.get('context', {}),
                'recommendation_type': feedback_data.get('recommendation_type'),
                'position_in_list': feedback_data.get('position')
            }
            
            self.feedback_buffer.append(feedback_entry)
            
            # Update profile analyzer with real-time feedback
            self.profile_analyzer.update_profile_realtime(user_id, feedback_data)
            
            # Process feedback if buffer is full
            if len(self.feedback_buffer) > 100:
                self._process_feedback_batch()
            
            logger.info(f"Processed feedback for user {user_id}: {feedback_data.get('feedback_type')}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing user feedback: {e}")
            return False
    
    def _get_content_pool(self, user_id: int, user_profile: Dict[str, Any], 
                         context: Dict[str, Any] = None) -> List[Any]:
        """Get appropriate content pool for recommendations"""
        try:
            # Get user's interaction history
            interacted_content_ids = self._get_user_interacted_content_ids(user_id)
            
            # Base query
            query = self.models['Content'].query.filter(
                ~self.models['Content'].id.in_(interacted_content_ids) if interacted_content_ids else True
            )
            
            # Apply quality filter based on user preferences
            behavior_profile = user_profile.get('behavior_profile', {})
            rating_behavior = behavior_profile.get('rating_behavior', {})
            
            if rating_behavior.get('has_ratings', False):
                avg_rating = rating_behavior.get('average_rating', 7.0)
                min_rating = max(avg_rating - 2.0, 4.0)  # Don't go below 4.0
                query = query.filter(self.models['Content'].rating >= min_rating)
            else:
                query = query.filter(self.models['Content'].rating >= 6.0)  # General quality filter
            
            # Apply recency filter if user prefers recent content
            content_pool = query.order_by(desc(self.models['Content'].rating)).limit(1000).all()
            
            return content_pool
            
        except Exception as e:
            logger.error(f"Error getting content pool: {e}")
            return []
    
    def _get_discovery_content_pool(self, user_profile: Dict[str, Any]) -> List[Any]:
        """Get content pool optimized for discovery"""
        try:
            # Get user's interaction history to find unexplored areas
            user_id = user_profile['user_id']
            interacted_content_ids = self._get_user_interacted_content_ids(user_id)
            
            # Get user's genre preferences to find gaps
            preferred_genres = set(self.content_personalizer._get_user_preferred_genres(user_profile))
            
            # Find content in unexplored genres
            query = self.models['Content'].query.filter(
                ~self.models['Content'].id.in_(interacted_content_ids) if interacted_content_ids else True,
                self.models['Content'].rating >= 7.0  # High quality for discovery
            )
            
            all_content = query.limit(800).all()
            
            # Filter for discovery potential
            discovery_content = []
            for content in all_content:
                if content.genres:
                    try:
                        content_genres = set(json.loads(content.genres))
                        # Include if it has genres user hasn't explored much
                        if not content_genres.issubset(preferred_genres):
                            discovery_content.append(content)
                    except:
                        pass
            
            return discovery_content[:500]  # Limit for performance
            
        except Exception as e:
            logger.error(f"Error getting discovery content pool: {e}")
            return []
    
    def _apply_type_specific_filtering(self, recommendations: List[Dict[str, Any]], 
                                     rec_type: str, user_profile: Dict[str, Any],
                                     limit: int) -> List[Dict[str, Any]]:
        """Apply filtering specific to recommendation type"""
        if rec_type == 'for_you':
            # For You: High confidence, culturally aligned
            filtered = [r for r in recommendations if r['personalization_score'] > 0.6]
        
        elif rec_type == 'discover':
            # Discovery: High exploration factor
            filtered = [r for r in recommendations if r.get('exploration_factor', 0) > 0.5]
        
        elif rec_type == 'trending':
            # Trending: Recent and popular
            filtered = [r for r in recommendations if r.get('is_trending', False) or r.get('is_new_release', False)]
        
        else:
            # Default filtering
            filtered = recommendations
        
        return filtered[:limit]
    
    def _generate_recommendation_metadata(self, user_profile: Dict[str, Any], 
                                        rec_type: str, strategy: str, count: int) -> Dict[str, Any]:
        """Generate metadata for recommendations"""
        return {
            'recommendation_engine_version': '3.0.0',
            'strategy_used': strategy,
            'profile_confidence': user_profile.get('profile_confidence', 0.5),
            'personalization_readiness': user_profile.get('personalization_readiness', 'medium'),
            'content_history_size': user_profile.get('content_history_size', 0),
            'recommendations_count': count,
            'telugu_priority_applied': True,
            'cultural_alignment_applied': True,
            'real_time_learning_enabled': self.enable_real_time,
            'generated_at': datetime.utcnow().isoformat(),
            'cache_duration': 1800,  # 30 minutes
            'algorithms_used': list(self.algorithms.keys()),
            'recommendation_freshness': 'real_time' if self.enable_real_time else 'periodic'
        }
    
    def _generate_profile_insights(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about user profile for frontend"""
        cinematic_dna = user_profile.get('cinematic_dna', {})
        behavior_profile = user_profile.get('behavior_profile', {})
        
        return {
            'profile_strength': user_profile.get('personalization_readiness', 'medium'),
            'cinematic_sophistication': cinematic_dna.get('cinematic_sophistication_score', 0.5),
            'cultural_affinity': {
                'telugu': cinematic_dna.get('telugu_cultural_affinity', 0.5),
                'indian': cinematic_dna.get('indian_cultural_affinity', 0.5),
                'global': cinematic_dna.get('global_cinema_exposure', 0.5)
            },
            'exploration_tendency': behavior_profile.get('content_exploration', {}).get('exploration_tendency', 'medium'),
            'quality_preference': behavior_profile.get('rating_behavior', {}).get('rating_tendency', 'balanced'),
            'recommendation_accuracy': min(user_profile.get('profile_confidence', 0.5) * 100, 95),
            'next_profile_update': user_profile.get('next_update_due', datetime.utcnow().isoformat())
        }
    
    def _get_cold_start_recommendations(self, user_id: int, limit: int) -> Dict[str, Any]:
        """Get recommendations for new users"""
        try:
            # Use orchestrator for popular content with Telugu priority
            orchestrator = self.algorithms['orchestrator']
            
            # Get trending content with language priority
            content_pool = self.models['Content'].query.filter(
                self.models['Content'].rating >= 7.0
            ).order_by(desc(self.models['Content'].popularity)).limit(200).all()
            
            trending_categories = orchestrator.get_trending_with_algorithms(
                content_pool, limit=limit, apply_language_priority=True
            )
            
            # Format for cold start response
            cold_start_recs = []
            
            # Prioritize Telugu content
            for category, recs in trending_categories.items():
                if 'telugu' in category.lower() or category in ['trending_movies', 'critics_choice']:
                    cold_start_recs.extend(recs[:10])
            
            return {
                'success': True,
                'user_id': user_id,
                'recommendation_type': 'cold_start',
                'recommendations': cold_start_recs[:limit],
                'message': 'Welcome to CineBrain! These recommendations are based on popular Telugu and regional content. Interact with content to get personalized recommendations.',
                'next_step': 'Rate some content or add to favorites to improve recommendations'
            }
            
        except Exception as e:
            logger.error(f"Error generating cold start recommendations: {e}")
            return self._get_error_fallback_recommendations(user_id, limit)
    
    def _get_error_fallback_recommendations(self, user_id: int, limit: int) -> Dict[str, Any]:
        """Fallback recommendations for error cases"""
        try:
            # Simple popular content
            popular_content = self.models['Content'].query.filter(
                self.models['Content'].rating >= 7.0
            ).order_by(desc(self.models['Content'].popularity)).limit(limit).all()
            
            formatted_content = []
            for content in popular_content:
                formatted_content.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'poster_path': self.content_personalizer._format_poster_path(content.poster_path),
                    'personalization_score': 0.5,
                    'recommendation_strength': 'popular'
                })
            
            return {
                'success': True,
                'user_id': user_id,
                'recommendation_type': 'fallback',
                'recommendations': formatted_content,
                'message': 'Showing popular content due to temporary service limitation'
            }
            
        except Exception as e:
            logger.error(f"Error in fallback recommendations: {e}")
            return {
                'success': False,
                'user_id': user_id,
                'recommendations': [],
                'error': 'Unable to generate recommendations at this time'
            }
    
    # Helper methods
    def _get_user_interacted_content_ids(self, user_id: int) -> List[int]:
        """Get IDs of content user has interacted with"""
        try:
            interactions = self.models['UserInteraction'].query.filter_by(user_id=user_id).all()
            return [i.content_id for i in interactions]
        except:
            return []
    
    def _generate_discovery_reason(self, content: Any, user_profile: Dict[str, Any]) -> str:
        """Generate reason for discovery recommendation"""
        preferred_genres = set(self.content_personalizer._get_user_preferred_genres(user_profile))
        content_genres = set(json.loads(content.genres or '[]'))
        
        new_genres = content_genres - preferred_genres
        if new_genres:
            return f"Explore {list(new_genres)[0]} genre"
        else:
            return "Highly rated content worth exploring"
    
    def _calculate_exploration_factor(self, content: Any, user_profile: Dict[str, Any]) -> float:
        """Calculate how much this content expands user's horizons"""
        preferred_genres = set(self.content_personalizer._get_user_preferred_genres(user_profile))
        content_genres = set(json.loads(content.genres or '[]'))
        
        # Higher score for content outside user's comfort zone but still quality
        novelty = len(content_genres - preferred_genres) / max(len(content_genres), 1)
        quality = (content.rating / 10) if content.rating else 0.7
        
        return novelty * 0.6 + quality * 0.4
    
    def _identify_exploration_areas(self, user_profile: Dict[str, Any]) -> List[str]:
        """Identify areas user could explore"""
        preferred_genres = set(self.content_personalizer._get_user_preferred_genres(user_profile))
        all_genres = ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller', 'Horror', 'Sci-Fi', 'Fantasy']
        
        unexplored = [g for g in all_genres if g not in preferred_genres]
        return unexplored[:3]
    
    def _assess_comfort_zone_expansion(self, user_profile: Dict[str, Any]) -> str:
        """Assess user's readiness for comfort zone expansion"""
        behavior_profile = user_profile.get('behavior_profile', {})
        exploration_tendency = behavior_profile.get('content_exploration', {}).get('exploration_tendency', 'medium')
        
        if exploration_tendency == 'high':
            return 'ready_for_adventure'
        elif exploration_tendency == 'medium':
            return 'gradual_expansion'
        else:
            return 'comfort_zone_focus'
    
    def _get_trending_for_user(self, user_id: int, limit: int) -> List[Dict[str, Any]]:
        """Get trending content personalized for user"""
        try:
            trending_content = self.models['Content'].query.filter(
                self.models['Content'].is_trending == True
            ).order_by(desc(self.models['Content'].popularity)).limit(limit).all()
            
            return [self.content_personalizer._personalize_single_content(
                content, {'user_id': user_id}
            ) for content in trending_content]
            
        except:
            return []
    
    def _interleave_recommendations(self, rec_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Interleave multiple recommendation lists"""
        interleaved = []
        max_len = max(len(lst) for lst in rec_lists if lst)
        
        for i in range(max_len):
            for rec_list in rec_lists:
                if i < len(rec_list):
                    # Avoid duplicates
                    content_id = rec_list[i]['id']
                    if not any(r['id'] == content_id for r in interleaved):
                        interleaved.append(rec_list[i])
        
        return interleaved
    
    def _process_feedback_batch(self):
        """Process batch of feedback for algorithm improvement"""
        try:
            # This would implement batch learning from feedback
            # For now, just clear the buffer
            logger.info(f"Processing {len(self.feedback_buffer)} feedback entries")
            self.feedback_buffer = []
        except Exception as e:
            logger.error(f"Error processing feedback batch: {e}")