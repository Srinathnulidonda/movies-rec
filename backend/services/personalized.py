# backend/services/personalized.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import logging
import math
import re
from typing import List, Dict, Any, Tuple, Optional
import hashlib
from sqlalchemy import func, desc, and_, or_
from sqlalchemy.exc import OperationalError, DisconnectionError
from contextlib import contextmanager
import traceback
import psycopg2
from services.algorithms import (
    ContentBasedFiltering, 
    CollaborativeFiltering, 
    HybridRecommendationEngine,
    LanguagePriorityFilter,
    PopularityRanking,
    UltraPowerfulSimilarityEngine,
    LANGUAGE_WEIGHTS,
    PRIORITY_LANGUAGES
)

logger = logging.getLogger(__name__)

class UserProfileAnalyzer:
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.Review = models['Review']
    
    def build_comprehensive_user_profile(self, user_id: int) -> Dict[str, Any]:
        try:
            user = self.User.query.get(user_id)
            if not user:
                return {}
            
            interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            
            profile = {
                'user_id': user_id,
                'username': user.username,
                'registration_date': user.created_at,
                'last_active': user.last_active,
                'explicit_preferences': self._analyze_explicit_preferences(user),
                'implicit_preferences': self._analyze_implicit_preferences(interactions),
                'viewing_patterns': self._analyze_viewing_patterns(interactions),
                'rating_patterns': self._analyze_rating_patterns(interactions),
                'search_patterns': self._analyze_search_patterns(interactions),
                'genre_preferences': self._analyze_genre_preferences(interactions),
                'language_preferences': self._analyze_language_preferences(interactions),
                'content_type_preferences': self._analyze_content_type_preferences(interactions),
                'mood_preferences': self._analyze_mood_preferences(interactions),
                'temporal_preferences': self._analyze_temporal_preferences(interactions),
                'quality_threshold': self._calculate_quality_threshold(interactions),
                'engagement_score': self._calculate_engagement_score(interactions),
                'exploration_tendency': self._calculate_exploration_tendency(interactions),
                'loyalty_score': self._calculate_loyalty_score(interactions),
                'recommendation_history': self._get_recommendation_history(user_id),
                'feedback_patterns': self._analyze_feedback_patterns(user_id),
                'recent_activity': self._get_recent_activity(interactions),
                'current_interests': self._infer_current_interests(interactions),
                'profile_completeness': 0.0,
                'confidence_score': 0.0,
                'last_updated': datetime.utcnow()
            }
            
            profile['profile_completeness'] = self._calculate_profile_completeness(profile)
            profile['confidence_score'] = self._calculate_confidence_score(profile, interactions)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error building user profile for user {user_id}: {e}")
            return {}
    
    def _analyze_explicit_preferences(self, user: Any) -> Dict[str, Any]:
        return {
            'preferred_languages': json.loads(user.preferred_languages or '[]'),
            'preferred_genres': json.loads(user.preferred_genres or '[]'),
            'location': user.location,
            'profile_set': bool(user.preferred_languages or user.preferred_genres)
        }
    
    def _analyze_implicit_preferences(self, interactions: List[Any]) -> Dict[str, Any]:
        if not interactions:
            return {}
        
        interaction_counts = Counter([i.interaction_type for i in interactions])
        recent_interactions = [i for i in interactions 
                             if i.timestamp > datetime.utcnow() - timedelta(days=30)]
        
        return {
            'total_interactions': len(interactions),
            'interaction_distribution': dict(interaction_counts),
            'recent_activity_count': len(recent_interactions),
            'average_interactions_per_day': len(interactions) / max(
                (datetime.utcnow() - min(i.timestamp for i in interactions)).days, 1
            ) if interactions else 0,
            'most_common_interaction': interaction_counts.most_common(1)[0][0] if interaction_counts else None
        }
    
    def _analyze_viewing_patterns(self, interactions: List[Any]) -> Dict[str, Any]:
        view_interactions = [i for i in interactions if i.interaction_type in ['view', 'watch', 'play']]
        
        if not view_interactions:
            return {}
        
        hour_distribution = Counter([i.timestamp.hour for i in view_interactions])
        day_distribution = Counter([i.timestamp.weekday() for i in view_interactions])
        
        content_ids = [i.content_id for i in view_interactions]
        contents = self.Content.query.filter(self.Content.id.in_(content_ids)).all()
        
        runtime_preferences = []
        for content in contents:
            if content.runtime:
                runtime_preferences.append(content.runtime)
        
        return {
            'total_views': len(view_interactions),
            'peak_viewing_hours': [hour for hour, _ in hour_distribution.most_common(3)],
            'preferred_days': [day for day, _ in day_distribution.most_common(3)],
            'average_content_runtime': np.mean(runtime_preferences) if runtime_preferences else 0,
            'runtime_preference_std': np.std(runtime_preferences) if runtime_preferences else 0,
            'binge_tendency': self._calculate_binge_tendency(view_interactions),
            'viewing_consistency': self._calculate_viewing_consistency(view_interactions)
        }
    
    def _analyze_rating_patterns(self, interactions: List[Any]) -> Dict[str, Any]:
        rating_interactions = [i for i in interactions if i.rating is not None]
        
        if not rating_interactions:
            return {}
        
        ratings = [i.rating for i in rating_interactions]
        
        return {
            'total_ratings': len(rating_interactions),
            'average_rating': np.mean(ratings),
            'rating_std': np.std(ratings),
            'rating_distribution': dict(Counter([round(r) for r in ratings])),
            'rating_tendency': 'harsh' if np.mean(ratings) < 6 else 'generous' if np.mean(ratings) > 8 else 'balanced',
            'rating_consistency': 1 - (np.std(ratings) / 10),
            'high_rating_threshold': np.percentile(ratings, 75),
            'low_rating_threshold': np.percentile(ratings, 25)
        }
    
    def _analyze_search_patterns(self, interactions: List[Any]) -> Dict[str, Any]:
        search_interactions = [i for i in interactions if i.interaction_type == 'search']
        
        if not search_interactions:
            return {}
        
        search_terms = []
        for interaction in search_interactions:
            if interaction.interaction_metadata:
                metadata = json.loads(interaction.interaction_metadata)
                if 'search_query' in metadata:
                    search_terms.append(metadata['search_query'].lower())
        
        return {
            'total_searches': len(search_interactions),
            'unique_search_terms': len(set(search_terms)),
            'most_searched_terms': [term for term, _ in Counter(search_terms).most_common(5)],
            'search_frequency': len(search_interactions) / max(
                (datetime.utcnow() - min(i.timestamp for i in search_interactions)).days, 1
            ),
            'search_diversity': len(set(search_terms)) / max(len(search_terms), 1)
        }
    
    def _analyze_genre_preferences(self, interactions: List[Any]) -> Dict[str, Any]:
        content_ids = [i.content_id for i in interactions]
        contents = self.Content.query.filter(self.Content.id.in_(content_ids)).all()
        
        genre_scores = defaultdict(float)
        interaction_weights = {
            'favorite': 3.0,
            'watchlist': 2.0,
            'view': 1.5,
            'search': 1.0,
            'rating': 1.0
        }
        
        for interaction in interactions:
            content = next((c for c in contents if c.id == interaction.content_id), None)
            if not content or not content.genres:
                continue
            
            weight = interaction_weights.get(interaction.interaction_type, 1.0)
            
            days_ago = (datetime.utcnow() - interaction.timestamp).days
            recency_weight = math.exp(-days_ago / 30)
            
            rating_weight = 1.0
            if interaction.rating:
                rating_weight = (interaction.rating / 10) * 2
            
            final_weight = weight * recency_weight * rating_weight
            
            try:
                genres = json.loads(content.genres)
                for genre in genres:
                    genre_scores[genre] += final_weight
            except:
                continue
        
        if genre_scores:
            max_score = max(genre_scores.values())
            normalized_scores = {genre: score/max_score for genre, score in genre_scores.items()}
        else:
            normalized_scores = {}
        
        return {
            'genre_scores': dict(normalized_scores),
            'top_genres': [genre for genre, _ in Counter(normalized_scores).most_common(5)],
            'genre_diversity': len(normalized_scores),
            'dominant_genre': max(normalized_scores.items(), key=lambda x: x[1])[0] if normalized_scores else None
        }
    
    def _analyze_language_preferences(self, interactions: List[Any]) -> Dict[str, Any]:
        content_ids = [i.content_id for i in interactions]
        contents = self.Content.query.filter(self.Content.id.in_(content_ids)).all()
        
        language_scores = defaultdict(float)
        interaction_weights = {
            'favorite': 3.0,
            'watchlist': 2.0,
            'view': 1.5,
            'search': 1.0
        }
        
        for interaction in interactions:
            content = next((c for c in contents if c.id == interaction.content_id), None)
            if not content or not content.languages:
                continue
            
            weight = interaction_weights.get(interaction.interaction_type, 1.0)
            
            try:
                languages = json.loads(content.languages)
                for language in languages:
                    lang_lower = language.lower()
                    priority_weight = LANGUAGE_WEIGHTS.get(lang_lower, 0.5)
                    language_scores[language] += weight * priority_weight
            except:
                continue
        
        if language_scores:
            max_score = max(language_scores.values())
            normalized_scores = {lang: score/max_score for lang, score in language_scores.items()}
            sorted_languages = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            normalized_scores = {}
            sorted_languages = []
        
        return {
            'language_scores': dict(normalized_scores),
            'preferred_languages': [lang for lang, _ in sorted_languages[:3]],
            'primary_language': sorted_languages[0][0] if sorted_languages else None,
            'language_diversity': len(normalized_scores),
            'telugu_preference': normalized_scores.get('Telugu', 0) > 0.7
        }
    
    def _analyze_content_type_preferences(self, interactions: List[Any]) -> Dict[str, Any]:
        content_ids = [i.content_id for i in interactions]
        contents = self.Content.query.filter(self.Content.id.in_(content_ids)).all()
        
        type_scores = defaultdict(float)
        interaction_weights = {
            'favorite': 3.0,
            'watchlist': 2.0,
            'view': 1.5,
            'search': 1.0
        }
        
        for interaction in interactions:
            content = next((c for c in contents if c.id == interaction.content_id), None)
            if not content:
                continue
            
            weight = interaction_weights.get(interaction.interaction_type, 1.0)
            type_scores[content.content_type] += weight
        
        if type_scores:
            total_score = sum(type_scores.values())
            normalized_scores = {ctype: score/total_score for ctype, score in type_scores.items()}
        else:
            normalized_scores = {}
        
        return {
            'content_type_scores': dict(normalized_scores),
            'preferred_content_type': max(normalized_scores.items(), key=lambda x: x[1])[0] if normalized_scores else None,
            'content_type_diversity': len(normalized_scores),
            'movie_preference': normalized_scores.get('movie', 0),
            'tv_preference': normalized_scores.get('tv', 0),
            'anime_preference': normalized_scores.get('anime', 0)
        }
    
    def _analyze_mood_preferences(self, interactions: List[Any]) -> Dict[str, Any]:
        content_ids = [i.content_id for i in interactions]
        contents = self.Content.query.filter(self.Content.id.in_(content_ids)).all()
        
        mood_indicators = {
            'action': ['action', 'adventure', 'thriller', 'crime'],
            'drama': ['drama', 'biography', 'history'],
            'comedy': ['comedy', 'family'],
            'romance': ['romance'],
            'horror': ['horror', 'mystery'],
            'sci-fi': ['science fiction', 'fantasy'],
            'documentary': ['documentary']
        }
        
        mood_scores = defaultdict(float)
        
        for interaction in interactions:
            content = next((c for c in contents if c.id == interaction.content_id), None)
            if not content or not content.genres:
                continue
            
            weight = 2.0 if interaction.interaction_type in ['favorite', 'watchlist'] else 1.0
            
            try:
                genres = [g.lower() for g in json.loads(content.genres)]
                for mood, indicators in mood_indicators.items():
                    if any(indicator in genres for indicator in indicators):
                        mood_scores[mood] += weight
            except:
                continue
        
        return {
            'mood_scores': dict(mood_scores),
            'dominant_mood': max(mood_scores.items(), key=lambda x: x[1])[0] if mood_scores else None,
            'mood_diversity': len(mood_scores)
        }
    
    def _analyze_temporal_preferences(self, interactions: List[Any]) -> Dict[str, Any]:
        if not interactions:
            return {}
        
        hours = [i.timestamp.hour for i in interactions]
        hour_dist = Counter(hours)
        
        weekdays = [i.timestamp.weekday() for i in interactions]
        weekday_dist = Counter(weekdays)
        
        sessions = self._identify_viewing_sessions(interactions)
        
        return {
            'peak_hours': [h for h, _ in hour_dist.most_common(3)],
            'preferred_weekdays': [d for d, _ in weekday_dist.most_common(3)],
            'average_session_length': np.mean([s['duration'] for s in sessions]) if sessions else 0,
            'typical_viewing_time': 'evening' if 18 <= np.mean(hours) <= 22 else 'morning' if np.mean(hours) < 12 else 'afternoon',
            'weekend_vs_weekday': {
                'weekend': sum(1 for d in weekdays if d >= 5) / len(weekdays),
                'weekday': sum(1 for d in weekdays if d < 5) / len(weekdays)
            }
        }
    
    def _calculate_quality_threshold(self, interactions: List[Any]) -> float:
        rated_interactions = [i for i in interactions if i.rating is not None]
        
        if not rated_interactions:
            return 7.0
        
        ratings = [i.rating for i in rated_interactions]
        threshold = np.percentile(ratings, 25)
        
        avg_rating = np.mean(ratings)
        if avg_rating < 6:
            threshold = max(threshold - 0.5, 1.0)
        elif avg_rating > 8:
            threshold = min(threshold + 0.5, 10.0)
        
        return float(threshold)
    
    def _calculate_engagement_score(self, interactions: List[Any]) -> float:
        if not interactions:
            return 0.0
        
        total_interactions = len(interactions)
        unique_content = len(set(i.content_id for i in interactions))
        rating_count = len([i for i in interactions if i.rating is not None])
        recent_activity = len([i for i in interactions 
                              if i.timestamp > datetime.utcnow() - timedelta(days=7)])
        
        interaction_score = min(total_interactions / 100, 1.0)
        diversity_score = unique_content / max(total_interactions, 1)
        rating_score = rating_count / max(total_interactions, 1)
        recency_score = recent_activity / 10
        
        engagement = (
            interaction_score * 0.3 +
            diversity_score * 0.2 +
            rating_score * 0.2 +
            recency_score * 0.3
        )
        
        return min(engagement, 1.0)
    
    def _calculate_exploration_tendency(self, interactions: List[Any]) -> float:
        if len(interactions) < 10:
            return 0.5
        
        content_ids = [i.content_id for i in interactions]
        contents = self.Content.query.filter(self.Content.id.in_(content_ids)).all()
        
        all_genres = []
        for content in contents:
            if content.genres:
                try:
                    all_genres.extend(json.loads(content.genres))
                except:
                    continue
        
        unique_genres = len(set(all_genres))
        total_genre_interactions = len(all_genres)
        
        exploration_score = unique_genres / max(total_genre_interactions, 1)
        
        return min(exploration_score * 2, 1.0)
    
    def _calculate_loyalty_score(self, interactions: List[Any]) -> float:
        if not interactions:
            return 0.0
        
        timestamps = [i.timestamp for i in interactions]
        first_interaction = min(timestamps)
        last_interaction = max(timestamps)
        total_span = (last_interaction - first_interaction).days
        
        if total_span == 0:
            return 0.5
        
        active_days = len(set(t.date() for t in timestamps))
        consistency = active_days / max(total_span, 1)
        
        daily_activity = [0] * (total_span + 1)
        for ts in timestamps:
            day_index = (ts - first_interaction).days
            daily_activity[day_index] = 1
        
        loyalty = min(consistency * 2, 1.0)
        
        return loyalty
    
    def _get_recommendation_history(self, user_id: int) -> Dict[str, Any]:
        return {
            'total_recommendations_served': 0,
            'recommendations_clicked': 0,
            'recommendations_rated': 0,
            'average_recommendation_rating': 0.0,
            'click_through_rate': 0.0,
            'recommendation_accuracy': 0.0
        }
    
    def _analyze_feedback_patterns(self, user_id: int) -> Dict[str, Any]:
        return {
            'feedback_frequency': 0.0,
            'positive_feedback_rate': 0.0,
            'negative_feedback_rate': 0.0,
            'feedback_consistency': 0.0
        }
    
    def _get_recent_activity(self, interactions: List[Any]) -> Dict[str, Any]:
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_interactions = [i for i in interactions if i.timestamp > recent_cutoff]
        
        if not recent_interactions:
            return {}
        
        recent_content_ids = [i.content_id for i in recent_interactions]
        recent_contents = self.Content.query.filter(self.Content.id.in_(recent_content_ids)).all()
        
        return {
            'recent_interaction_count': len(recent_interactions),
            'recent_content_types': list(set(c.content_type for c in recent_contents)),
            'recent_genres': self._extract_recent_genres(recent_contents),
            'recent_languages': self._extract_recent_languages(recent_contents),
            'trending_interest': self._identify_trending_interest(recent_interactions, recent_contents)
        }
    
    def _infer_current_interests(self, interactions: List[Any]) -> Dict[str, Any]:
        recent_cutoff = datetime.utcnow() - timedelta(days=14)
        recent_interactions = [i for i in interactions if i.timestamp > recent_cutoff]
        
        if not recent_interactions:
            return {}
        
        interaction_types = Counter([i.interaction_type for i in recent_interactions])
        
        content_ids = [i.content_id for i in recent_interactions]
        contents = self.Content.query.filter(self.Content.id.in_(content_ids)).all()
        
        current_genres = []
        current_languages = []
        for content in contents:
            if content.genres:
                try:
                    current_genres.extend(json.loads(content.genres))
                except:
                    pass
            if content.languages:
                try:
                    current_languages.extend(json.loads(content.languages))
                except:
                    pass
        
        return {
            'current_genre_interest': [g for g, _ in Counter(current_genres).most_common(3)],
            'current_language_interest': [l for l, _ in Counter(current_languages).most_common(2)],
            'current_activity_pattern': dict(interaction_types),
            'interest_intensity': len(recent_interactions) / 14,
            'interest_focus': 'broad' if len(set(current_genres)) > 5 else 'focused'
        }
    
    def _calculate_profile_completeness(self, profile: Dict[str, Any]) -> float:
        completeness_factors = []
        
        if profile['explicit_preferences']['profile_set']:
            completeness_factors.append(0.2)
        
        if profile['implicit_preferences']['total_interactions'] > 10:
            completeness_factors.append(0.3)
        
        if profile.get('rating_patterns', {}).get('total_ratings', 0) > 5:
            completeness_factors.append(0.2)
        
        if len(profile.get('genre_preferences', {}).get('genre_scores', {})) > 3:
            completeness_factors.append(0.15)
        
        if profile.get('language_preferences', {}).get('primary_language'):
            completeness_factors.append(0.15)
        
        return sum(completeness_factors)
    
    def _calculate_confidence_score(self, profile: Dict[str, Any], interactions: List[Any]) -> float:
        if not interactions:
            return 0.0
        
        confidence_factors = []
        
        interaction_count = len(interactions)
        if interaction_count > 50:
            confidence_factors.append(0.4)
        elif interaction_count > 20:
            confidence_factors.append(0.3)
        elif interaction_count > 10:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        recent_interactions = [i for i in interactions 
                              if i.timestamp > datetime.utcnow() - timedelta(days=30)]
        if len(recent_interactions) > 5:
            confidence_factors.append(0.3)
        
        rating_patterns = profile.get('rating_patterns', {})
        if rating_patterns.get('rating_consistency', 0) > 0.7:
            confidence_factors.append(0.2)
        
        if profile['profile_completeness'] > 0.7:
            confidence_factors.append(0.1)
        
        return sum(confidence_factors)
    
    def _calculate_binge_tendency(self, view_interactions: List[Any]) -> float:
        if len(view_interactions) < 5:
            return 0.0
        
        daily_views = defaultdict(int)
        for interaction in view_interactions:
            date_key = interaction.timestamp.date()
            daily_views[date_key] += 1
        
        binge_days = sum(1 for count in daily_views.values() if count >= 3)
        total_active_days = len(daily_views)
        
        return binge_days / total_active_days if total_active_days > 0 else 0.0
    
    def _calculate_viewing_consistency(self, view_interactions: List[Any]) -> float:
        if len(view_interactions) < 7:
            return 0.0
        
        daily_views = defaultdict(int)
        for interaction in view_interactions:
            date_key = interaction.timestamp.date()
            daily_views[date_key] += 1
        
        view_counts = list(daily_views.values())
        if len(view_counts) < 2:
            return 0.0
        
        consistency = 1 - (np.std(view_counts) / (np.mean(view_counts) + 1))
        return max(consistency, 0.0)
    
    def _identify_viewing_sessions(self, interactions: List[Any]) -> List[Dict[str, Any]]:
        if not interactions:
            return []
        
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        
        sessions = []
        current_session = []
        
        for i, interaction in enumerate(sorted_interactions):
            if i == 0:
                current_session = [interaction]
            else:
                time_gap = (interaction.timestamp - sorted_interactions[i-1].timestamp).total_seconds() / 3600
                if time_gap > 2:
                    if current_session:
                        sessions.append({
                            'start_time': current_session[0].timestamp,
                            'end_time': current_session[-1].timestamp,
                            'duration': (current_session[-1].timestamp - current_session[0].timestamp).total_seconds() / 3600,
                            'interaction_count': len(current_session)
                        })
                    current_session = [interaction]
                else:
                    current_session.append(interaction)
        
        if current_session:
            sessions.append({
                'start_time': current_session[0].timestamp,
                'end_time': current_session[-1].timestamp,
                'duration': (current_session[-1].timestamp - current_session[0].timestamp).total_seconds() / 3600,
                'interaction_count': len(current_session)
            })
        
        return sessions
    
    def _extract_recent_genres(self, contents: List[Any]) -> List[str]:
        genres = []
        for content in contents:
            if content.genres:
                try:
                    genres.extend(json.loads(content.genres))
                except:
                    pass
        return [g for g, _ in Counter(genres).most_common(5)]
    
    def _extract_recent_languages(self, contents: List[Any]) -> List[str]:
        languages = []
        for content in contents:
            if content.languages:
                try:
                    languages.extend(json.loads(content.languages))
                except:
                    pass
        return [l for l, _ in Counter(languages).most_common(3)]
    
    def _identify_trending_interest(self, interactions: List[Any], contents: List[Any]) -> str:
        if not interactions:
            return "none"
        
        interaction_types = Counter([i.interaction_type for i in interactions])
        
        if interaction_types.get('search', 0) > 3:
            return "exploring"
        elif interaction_types.get('favorite', 0) > 2:
            return "favoriting"
        elif interaction_types.get('watchlist', 0) > 2:
            return "curating"
        elif interaction_types.get('view', 0) > 5:
            return "binge_watching"
        else:
            return "casual_browsing"


class NetflixLevelRecommendationEngine:
    def __init__(self, db, models, cache=None):
        self.db = db
        self.models = models
        self.cache = cache
        
        self.user_profiler = UserProfileAnalyzer(db, models)
        self.content_based = ContentBasedFiltering()
        self.collaborative = CollaborativeFiltering()
        self.hybrid = HybridRecommendationEngine()
        self.similarity_engine = UltraPowerfulSimilarityEngine()
        
        self.engines = {
            'collaborative_filtering': self._collaborative_filtering_recommendations,
            'content_based': self._content_based_recommendations,
            'hybrid_matrix_factorization': self._hybrid_matrix_factorization,
            'sequence_aware': self._sequence_aware_recommendations,
            'context_aware': self._context_aware_recommendations,
            'language_priority': self._language_priority_recommendations,
            'mood_based': self._mood_based_recommendations,
            'similarity_based': self._similarity_based_recommendations
        }

    @contextmanager
    def safe_db_operation(self):
        try:
            yield
        except (OperationalError, DisconnectionError, psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            logger.error(f"Database connection error: {e}")
            try:
                self.db.session.rollback()
            except:
                pass
            try:
                self.db.session.close()
            except:
                pass
            raise
        except Exception as e:
            logger.error(f"Database operation error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            try:
                self.db.session.rollback()
            except:
                pass
            raise

    def get_personalized_recommendations(self, user_id: int, limit: int = 50, 
                                        categories: List[str] = None) -> Dict[str, Any]:
        try:
            try:
                with self.safe_db_operation():
                    user_profile = self.user_profiler.build_comprehensive_user_profile(user_id)
            except Exception as e:
                logger.error(f"Error building user profile for user {user_id}: {e}")
                return self._get_cold_start_recommendations(user_id, limit)
            
            if not user_profile or user_profile.get('confidence_score', 0) < 0.1:
                logger.info(f"Low confidence profile for user {user_id}, using cold start")
                return self._get_cold_start_recommendations(user_id, limit)
            
            all_recommendations = {}
            
            if not categories:
                categories = [
                    'for_you',
                    'because_you_watched',
                    'trending_for_you',
                    'new_releases_for_you',
                    'your_language',
                    'your_genres',
                    'hidden_gems',
                    'continue_watching',
                    'quick_picks',
                    'critically_acclaimed'
                ]
            
            successful_categories = 0
            for category in categories:
                try:
                    recommendations = self._generate_category_recommendations(
                        user_profile, category, limit
                    )
                    if recommendations:
                        all_recommendations[category] = recommendations
                        successful_categories += 1
                        logger.info(f"Successfully generated {len(recommendations)} recommendations for {category}")
                    else:
                        logger.warning(f"No recommendations generated for category: {category}")
                except Exception as e:
                    logger.error(f"Error generating {category} recommendations: {e}")
                    continue
            
            if successful_categories == 0:
                logger.warning(f"No successful recommendation categories for user {user_id}")
                return self._get_fallback_recommendations(user_id, limit)
            
            response = {
                'user_id': user_id,
                'recommendations': all_recommendations,
                'profile_insights': self._generate_profile_insights(user_profile),
                'recommendation_metadata': {
                    'profile_completeness': user_profile.get('profile_completeness', 0),
                    'confidence_score': user_profile.get('confidence_score', 0),
                    'algorithms_used': list(self.engines.keys()),
                    'total_categories': len(all_recommendations),
                    'successful_categories': successful_categories,
                    'generated_at': datetime.utcnow().isoformat(),
                    'cache_duration': 3600,
                    'next_update': (datetime.utcnow() + timedelta(hours=1)).isoformat()
                }
            }
            
            if self.cache:
                try:
                    cache_key = f"personalized_recs:{user_id}:{hash(str(categories))}"
                    self.cache.set(cache_key, response, timeout=3600)
                except Exception as e:
                    logger.warning(f"Failed to cache recommendations: {e}")
            
            return response
            
        except Exception as e:
            logger.error(f"Critical error generating personalized recommendations for user {user_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_fallback_recommendations(user_id, limit)
    
    def _generate_category_recommendations(self, user_profile: Dict[str, Any], 
                                         category: str, limit: int) -> List[Dict[str, Any]]:
        
        category_generators = {
            'for_you': self._generate_for_you_recommendations,
            'because_you_watched': self._generate_because_you_watched,
            'trending_for_you': self._generate_trending_for_you,
            'new_releases_for_you': self._generate_new_releases_for_you,
            'your_language': self._generate_your_language_recommendations,
            'your_genres': self._generate_your_genre_recommendations,
            'hidden_gems': self._generate_hidden_gems,
            'continue_watching': self._generate_continue_watching,
            'quick_picks': self._generate_quick_picks,
            'critically_acclaimed': self._generate_critically_acclaimed
        }
        
        generator = category_generators.get(category)
        if not generator:
            return []
        
        try:
            with self.safe_db_operation():
                return generator(user_profile, limit)
        except Exception as e:
            logger.error(f"Error generating {category} recommendations: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _generate_for_you_recommendations(self, user_profile: Dict[str, Any], 
                                        limit: int) -> List[Dict[str, Any]]:
        user_id = user_profile['user_id']
        
        interacted_content_ids = self._get_user_interacted_content(user_id)
        content_pool = self._get_filtered_content_pool(
            exclude_ids=interacted_content_ids,
            user_profile=user_profile
        )
        
        recommendations = []
        
        hybrid_recs = self._hybrid_matrix_factorization(user_profile, content_pool, limit)
        for rec in hybrid_recs[:int(limit * 0.4)]:
            rec['source'] = 'hybrid_collaborative'
            rec['weight'] = 0.4
            recommendations.append(rec)
        
        similarity_recs = self._similarity_based_recommendations(user_profile, content_pool, limit)
        for rec in similarity_recs[:int(limit * 0.3)]:
            rec['source'] = 'similarity_based'
            rec['weight'] = 0.3
            recommendations.append(rec)
        
        language_recs = self._language_priority_recommendations(user_profile, content_pool, limit)
        for rec in language_recs[:int(limit * 0.2)]:
            rec['source'] = 'language_priority'
            rec['weight'] = 0.2
            recommendations.append(rec)
        
        serendipity_recs = self._generate_serendipity_recommendations(user_profile, content_pool, limit)
        for rec in serendipity_recs[:int(limit * 0.1)]:
            rec['source'] = 'serendipity'
            rec['weight'] = 0.1
            recommendations.append(rec)
        
        unique_recommendations = self._remove_duplicates_and_rerank(recommendations)
        
        return unique_recommendations[:limit]
    
    def _generate_because_you_watched(self, user_profile: Dict[str, Any], 
                                     limit: int) -> List[Dict[str, Any]]:
        user_id = user_profile['user_id']
        
        try:
            recent_interactions = self.models['UserInteraction'].query.filter(
                and_(
                    self.models['UserInteraction'].user_id == user_id,
                    self.models['UserInteraction'].interaction_type.in_(['view', 'favorite', 'watchlist']),
                    self.models['UserInteraction'].timestamp > datetime.utcnow() - timedelta(days=30)
                )
            ).order_by(desc(self.models['UserInteraction'].timestamp)).limit(10).all()
            
            if not recent_interactions:
                logger.info(f"No recent interactions found for user {user_id}")
                return []
            
            recommendations = []
            
            try:
                content_pool = self._get_base_content_pool()
            except Exception as e:
                logger.error(f"Error getting content pool: {e}")
                return []
            
            for interaction in recent_interactions[:3]:
                try:
                    base_content = self.models['Content'].query.get(interaction.content_id)
                    if not base_content:
                        logger.warning(f"Content not found for ID: {interaction.content_id}")
                        continue
                    
                    similar_content = self.similarity_engine.find_ultra_similar_content(
                        base_content,
                        content_pool,
                        limit=10,
                        min_similarity=0.6,
                        strict_mode=True
                    )
                    
                    for similar in similar_content:
                        content = similar['content']
                        recommendations.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'languages': json.loads(content.languages or '[]'),
                            'rating': content.rating,
                            'poster_path': self._format_poster_path(content.poster_path),
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'similarity_score': similar['similarity_score'],
                            'match_explanation': similar.get('match_explanation', {}),
                            'because_of': {
                                'title': base_content.title,
                                'content_id': base_content.id,
                                'interaction_type': interaction.interaction_type
                            },
                            'source': 'similarity_engine',
                            'confidence': similar.get('confidence', 'medium')
                        })
                
                except Exception as e:
                    logger.error(f"Error processing interaction {interaction.id}: {e}")
                    continue
            
            unique_recs = self._remove_duplicates_and_rerank(recommendations)
            return unique_recs[:limit]
            
        except Exception as e:
            logger.error(f"Error in _generate_because_you_watched: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _generate_trending_for_you(self, user_profile: Dict[str, Any], 
                                  limit: int) -> List[Dict[str, Any]]:
        trending_content = self.models['Content'].query.filter(
            self.models['Content'].is_trending == True
        ).order_by(desc(self.models['Content'].popularity)).limit(100).all()
        
        personalized_trending = []
        
        for content in trending_content:
            score = self._calculate_personalization_score(content, user_profile)
            if score > 0.3:
                personalized_trending.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'poster_path': self._format_poster_path(content.poster_path),
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'personalization_score': score,
                    'trending_rank': trending_content.index(content) + 1,
                    'source': 'personalized_trending',
                    'is_trending': True
                })
        
        personalized_trending.sort(key=lambda x: x['personalization_score'], reverse=True)
        return personalized_trending[:limit]
    
    def _generate_new_releases_for_you(self, user_profile: Dict[str, Any], 
                                      limit: int) -> List[Dict[str, Any]]:
        cutoff_date = datetime.utcnow().date() - timedelta(days=60)
        new_releases = self.models['Content'].query.filter(
            self.models['Content'].release_date >= cutoff_date
        ).order_by(desc(self.models['Content'].release_date)).limit(100).all()
        
        personalized_releases = []
        
        for content in new_releases:
            score = self._calculate_personalization_score(content, user_profile)
            if score > 0.3:
                days_since_release = (datetime.utcnow().date() - content.release_date).days
                freshness_bonus = max(1 - (days_since_release / 60), 0) * 0.2
                
                personalized_releases.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'poster_path': self._format_poster_path(content.poster_path),
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'personalization_score': score + freshness_bonus,
                    'days_since_release': days_since_release,
                    'source': 'personalized_new_releases',
                    'is_new_release': True
                })
        
        personalized_releases.sort(key=lambda x: x['personalization_score'], reverse=True)
        return personalized_releases[:limit]
    
    def _generate_your_language_recommendations(self, user_profile: Dict[str, Any], 
                                              limit: int) -> List[Dict[str, Any]]:
        language_prefs = user_profile.get('language_preferences', {})
        preferred_languages = language_prefs.get('preferred_languages', [])
        
        if not preferred_languages:
            preferred_languages = ['Telugu', 'English']
        
        recommendations = []
        
        for language in preferred_languages[:2]:
            lang_content = self.models['Content'].query.filter(
                self.models['Content'].languages.contains(f'"{language}"')
            ).order_by(desc(self.models['Content'].rating)).limit(25).all()
            
            for content in lang_content:
                score = self._calculate_personalization_score(content, user_profile)
                language_bonus = 0.3 if language == preferred_languages[0] else 0.15
                
                recommendations.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'poster_path': self._format_poster_path(content.poster_path),
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'personalization_score': score + language_bonus,
                    'primary_language': language,
                    'source': 'language_specific'
                })
        
        recommendations.sort(key=lambda x: x['personalization_score'], reverse=True)
        return recommendations[:limit]
    
    def _generate_your_genre_recommendations(self, user_profile: Dict[str, Any], 
                                           limit: int) -> List[Dict[str, Any]]:
        genre_prefs = user_profile.get('genre_preferences', {})
        top_genres = genre_prefs.get('top_genres', [])
        
        if not top_genres:
            return []
        
        recommendations = []
        
        for genre in top_genres[:3]:
            genre_content = self.models['Content'].query.filter(
                self.models['Content'].genres.contains(f'"{genre}"')
            ).order_by(desc(self.models['Content'].rating)).limit(20).all()
            
            for content in genre_content:
                score = self._calculate_personalization_score(content, user_profile)
                genre_bonus = 0.2
                
                recommendations.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'poster_path': self._format_poster_path(content.poster_path),
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'personalization_score': score + genre_bonus,
                    'primary_genre': genre,
                    'source': 'genre_specific'
                })
        
        recommendations.sort(key=lambda x: x['personalization_score'], reverse=True)
        return recommendations[:limit]
    
    def _generate_hidden_gems(self, user_profile: Dict[str, Any], 
                             limit: int) -> List[Dict[str, Any]]:
        hidden_gems = self.models['Content'].query.filter(
            and_(
                self.models['Content'].rating >= 7.5,
                self.models['Content'].vote_count >= 100,
                self.models['Content'].popularity < 50
            )
        ).order_by(desc(self.models['Content'].rating)).limit(50).all()
        
        recommendations = []
        
        for content in hidden_gems:
            score = self._calculate_personalization_score(content, user_profile)
            hidden_gem_bonus = 0.15
            
            if score > 0.2:
                recommendations.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'poster_path': self._format_poster_path(content.poster_path),
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'personalization_score': score + hidden_gem_bonus,
                    'source': 'hidden_gems',
                    'is_hidden_gem': True,
                    'gem_reason': 'High rating with lower popularity'
                })
        
        recommendations.sort(key=lambda x: x['personalization_score'], reverse=True)
        return recommendations[:limit]
    
    def _generate_continue_watching(self, user_profile: Dict[str, Any], 
                                   limit: int) -> List[Dict[str, Any]]:
        user_id = user_profile['user_id']
        
        tv_interactions = self.models['UserInteraction'].query.join(
            self.models['Content']
        ).filter(
            and_(
                self.models['UserInteraction'].user_id == user_id,
                self.models['UserInteraction'].interaction_type == 'view',
                self.models['Content'].content_type.in_(['tv', 'anime'])
            )
        ).all()
        
        recommendations = []
        content_pool = self._get_base_content_pool()
        
        for interaction in tv_interactions[-10:]:
            base_content = self.models['Content'].query.get(interaction.content_id)
            if not base_content:
                continue
            
            similar_series = self.similarity_engine.find_ultra_similar_content(
                base_content,
                [c for c in content_pool if c.content_type == base_content.content_type],
                limit=5,
                min_similarity=0.5
            )
            
            for similar in similar_series:
                content = similar['content']
                recommendations.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'poster_path': self._format_poster_path(content.poster_path),
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'similarity_score': similar['similarity_score'],
                    'continue_reason': f"Similar to {base_content.title}",
                    'source': 'continue_watching'
                })
        
        unique_recs = self._remove_duplicates_and_rerank(recommendations)
        return unique_recs[:limit]
    
    def _generate_quick_picks(self, user_profile: Dict[str, Any], 
                             limit: int) -> List[Dict[str, Any]]:
        quick_content = self.models['Content'].query.filter(
            or_(
                and_(
                    self.models['Content'].content_type == 'movie',
                    self.models['Content'].runtime <= 90
                ),
                self.models['Content'].content_type.in_(['tv', 'anime'])
            )
        ).order_by(desc(self.models['Content'].rating)).limit(50).all()
        
        recommendations = []
        
        for content in quick_content:
            score = self._calculate_personalization_score(content, user_profile)
            quick_bonus = 0.1
            
            if score > 0.3:
                recommendations.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'runtime': content.runtime,
                    'poster_path': self._format_poster_path(content.poster_path),
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'personalization_score': score + quick_bonus,
                    'source': 'quick_picks',
                    'quick_reason': 'Short and engaging'
                })
        
        recommendations.sort(key=lambda x: x['personalization_score'], reverse=True)
        return recommendations[:limit]
    
    def _generate_critically_acclaimed(self, user_profile: Dict[str, Any], 
                                      limit: int) -> List[Dict[str, Any]]:
        acclaimed_content = self.models['Content'].query.filter(
            and_(
                self.models['Content'].rating >= 8.0,
                self.models['Content'].vote_count >= 500
            )
        ).order_by(desc(self.models['Content'].rating)).limit(50).all()
        
        recommendations = []
        
        for content in acclaimed_content:
            score = self._calculate_personalization_score(content, user_profile)
            acclaim_bonus = (content.rating - 8.0) * 0.05
            
            if score > 0.2:
                recommendations.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'vote_count': content.vote_count,
                    'poster_path': self._format_poster_path(content.poster_path),
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'personalization_score': score + acclaim_bonus,
                    'source': 'critically_acclaimed',
                    'acclaim_reason': f'Highly rated ({content.rating}/10)'
                })
        
        recommendations.sort(key=lambda x: x['personalization_score'], reverse=True)
        return recommendations[:limit]
    
    def _calculate_personalization_score(self, content: Any, user_profile: Dict[str, Any]) -> float:
        score = 0.0
        
        genre_prefs = user_profile.get('genre_preferences', {})
        if genre_prefs.get('genre_scores') and content.genres:
            try:
                content_genres = set(json.loads(content.genres))
                for genre in content_genres:
                    score += genre_prefs['genre_scores'].get(genre, 0) * 0.3
            except:
                pass
        
        lang_prefs = user_profile.get('language_preferences', {})
        if lang_prefs.get('preferred_languages') and content.languages:
            try:
                content_languages = set(json.loads(content.languages))
                user_languages = set(lang_prefs['preferred_languages'])
                if content_languages & user_languages:
                    score += 0.25
            except:
                pass
        
        content_type_prefs = user_profile.get('content_type_preferences', {})
        if content_type_prefs.get('content_type_scores'):
            type_score = content_type_prefs['content_type_scores'].get(content.content_type, 0)
            score += type_score * 0.2
        
        quality_threshold = user_profile.get('quality_threshold', 7.0)
        if content.rating and content.rating >= quality_threshold:
            score += 0.15
        
        rating_patterns = user_profile.get('rating_patterns', {})
        if rating_patterns.get('average_rating') and content.rating:
            rating_diff = abs(rating_patterns['average_rating'] - content.rating)
            if rating_diff < 1.5:
                score += 0.1
        
        return min(score, 1.0)
    
    def _get_cold_start_recommendations(self, user_id: int, limit: int) -> Dict[str, Any]:
        popular_movies = self.models['Content'].query.filter(
            self.models['Content'].content_type == 'movie'
        ).order_by(desc(self.models['Content'].popularity)).limit(15).all()
        
        popular_tv = self.models['Content'].query.filter(
            self.models['Content'].content_type == 'tv'
        ).order_by(desc(self.models['Content'].popularity)).limit(15).all()
        
        popular_anime = self.models['Content'].query.filter(
            self.models['Content'].content_type == 'anime'
        ).order_by(desc(self.models['Content'].popularity)).limit(10).all()
        
        telugu_content = self.models['Content'].query.filter(
            self.models['Content'].languages.contains('"Telugu"')
        ).order_by(desc(self.models['Content'].rating)).limit(10).all()
        
        cold_start_recs = {
            'popular_movies': self._format_content_list(popular_movies),
            'popular_tv_shows': self._format_content_list(popular_tv),
            'popular_anime': self._format_content_list(popular_anime),
            'telugu_favorites': self._format_content_list(telugu_content)
        }
        
        return {
            'user_id': user_id,
            'recommendations': cold_start_recs,
            'profile_insights': {
                'status': 'new_user',
                'message': 'Start interacting with content to get personalized recommendations!'
            },
            'recommendation_metadata': {
                'type': 'cold_start',
                'profile_completeness': 0.0,
                'confidence_score': 0.0,
                'generated_at': datetime.utcnow().isoformat()
            }
        }
    
    def _get_fallback_recommendations(self, user_id: int, limit: int) -> Dict[str, Any]:
        try:
            trending = self.models['Content'].query.filter(
                self.models['Content'].is_trending == True
            ).order_by(desc(self.models['Content'].popularity)).limit(limit).all()
            
            return {
                'user_id': user_id,
                'recommendations': {
                    'trending_now': self._format_content_list(trending)
                },
                'profile_insights': {
                    'status': 'fallback',
                    'message': 'Showing trending content'
                },
                'recommendation_metadata': {
                    'type': 'fallback',
                    'generated_at': datetime.utcnow().isoformat()
                }
            }
        except:
            return {
                'user_id': user_id,
                'recommendations': {},
                'error': 'Unable to generate recommendations'
            }
    
    def _format_content_list(self, content_list: List[Any]) -> List[Dict[str, Any]]:
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
                'overview': content.overview[:150] + '...' if content.overview else ''
            })
        return formatted
    
    def _format_poster_path(self, poster_path: str) -> str:
        if not poster_path:
            return None
        
        if poster_path.startswith('http'):
            return poster_path
        else:
            return f"https://image.tmdb.org/t/p/w300{poster_path}"
    
    def _generate_profile_insights(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        insights = {
            'profile_strength': 'strong' if user_profile.get('confidence_score', 0) > 0.7 else 'building',
            'primary_interests': [],
            'viewing_style': 'casual',
            'recommendation_tip': ''
        }
        
        genre_prefs = user_profile.get('genre_preferences', {})
        if genre_prefs.get('top_genres'):
            insights['primary_interests'] = genre_prefs['top_genres'][:3]
        
        engagement = user_profile.get('engagement_score', 0)
        if engagement > 0.8:
            insights['viewing_style'] = 'enthusiast'
        elif engagement > 0.5:
            insights['viewing_style'] = 'regular'
        else:
            insights['viewing_style'] = 'casual'
        
        completeness = user_profile.get('profile_completeness', 0)
        if completeness < 0.5:
            insights['recommendation_tip'] = 'Rate more content to improve recommendations'
        elif completeness < 0.8:
            insights['recommendation_tip'] = 'Add content to your watchlist for better suggestions'
        else:
            insights['recommendation_tip'] = 'Your recommendations are highly personalized!'
        
        return insights
    
    def update_user_preferences_realtime(self, user_id: int, interaction_data: Dict[str, Any]):
        try:
            if self.cache:
                cache_keys = [
                    f"personalized_recs:{user_id}:*",
                    f"user_profile:{user_id}"
                ]
                for key in cache_keys:
                    try:
                        self.cache.delete(key)
                    except:
                        pass
            
            try:
                with self.safe_db_operation():
                    interaction = self.models['UserInteraction'](
                        user_id=user_id,
                        content_id=interaction_data.get('content_id'),
                        interaction_type=interaction_data.get('interaction_type'),
                        rating=interaction_data.get('rating'),
                        interaction_metadata=json.dumps(interaction_data.get('metadata', {}))
                    )
                    
                    self.db.session.add(interaction)
                    self.db.session.commit()
                    
                    logger.info(f"Updated preferences for user {user_id} with interaction {interaction_data.get('interaction_type')}")
                    return True
                    
            except Exception as e:
                logger.error(f"Error recording interaction: {e}")
                self.db.session.rollback()
                return False
                
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False
    
    def _collaborative_filtering_recommendations(self, user_profile: Dict[str, Any], 
                                               content_pool: List[Any], limit: int) -> List[Dict[str, Any]]:
        return []
    
    def _content_based_recommendations(self, user_profile: Dict[str, Any], 
                                     content_pool: List[Any], limit: int) -> List[Dict[str, Any]]:
        return []
    
    def _hybrid_matrix_factorization(self, user_profile: Dict[str, Any], 
                                   content_pool: List[Any], limit: int) -> List[Dict[str, Any]]:
        return []
    
    def _sequence_aware_recommendations(self, user_profile: Dict[str, Any], 
                                      content_pool: List[Any], limit: int) -> List[Dict[str, Any]]:
        return []
    
    def _context_aware_recommendations(self, user_profile: Dict[str, Any], 
                                     content_pool: List[Any], limit: int) -> List[Dict[str, Any]]:
        return []
    
    def _language_priority_recommendations(self, user_profile: Dict[str, Any], 
                                         content_pool: List[Any], limit: int) -> List[Dict[str, Any]]:
        return []
    
    def _mood_based_recommendations(self, user_profile: Dict[str, Any], 
                                  content_pool: List[Any], limit: int) -> List[Dict[str, Any]]:
        return []
    
    def _similarity_based_recommendations(self, user_profile: Dict[str, Any], 
                                        content_pool: List[Any], limit: int) -> List[Dict[str, Any]]:
        return []
    
    def _generate_serendipity_recommendations(self, user_profile: Dict[str, Any], 
                                            content_pool: List[Any], limit: int) -> List[Dict[str, Any]]:
        return []
    
    def _get_user_interacted_content(self, user_id: int) -> List[int]:
        interactions = self.models['UserInteraction'].query.filter_by(user_id=user_id).all()
        return [i.content_id for i in interactions]
    
    def _get_filtered_content_pool(self, exclude_ids: List[int] = None, 
                                  user_profile: Dict[str, Any] = None) -> List[Any]:
        query = self.models['Content'].query
        
        if exclude_ids:
            query = query.filter(~self.models['Content'].id.in_(exclude_ids))
        
        return query.order_by(desc(self.models['Content'].rating)).limit(1000).all()
    
    def _get_base_content_pool(self) -> List[Any]:
        try:
            return self.models['Content'].query.filter(
                self.models['Content'].rating.isnot(None)
            ).order_by(desc(self.models['Content'].rating)).limit(1000).all()
        except Exception as e:
            logger.error(f"Error getting base content pool: {e}")
            try:
                return self.models['Content'].query.limit(500).all()
            except Exception as e2:
                logger.error(f"Error with fallback content pool query: {e2}")
                return []
    
    def _remove_duplicates_and_rerank(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_ids = set()
        unique_recs = []
        
        for rec in recommendations:
            if rec['id'] not in seen_ids:
                seen_ids.add(rec['id'])
                unique_recs.append(rec)
        
        unique_recs.sort(key=lambda x: x.get('personalization_score', x.get('similarity_score', 0)), reverse=True)
        
        return unique_recs


recommendation_engine = None

def init_personalized(app, db, models, services, cache=None):
    global recommendation_engine
    
    try:
        recommendation_engine = NetflixLevelRecommendationEngine(db, models, cache)
        logger.info("Netflix-level personalized recommendation system initialized successfully")
        return recommendation_engine
    except Exception as e:
        logger.error(f"Failed to initialize personalized recommendation system: {e}")
        return None

def get_recommendation_engine():
    return recommendation_engine