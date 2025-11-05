# backend/personalized/profile_analyzer.py

"""
CineBrain User Profile Analyzer
Real-time user profiling with continuous learning and cinematic DNA analysis
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import math

from .utils import (
    TeluguPriorityManager,
    EmbeddingManager,
    SimilarityEngine,
    CacheManager,
    safe_json_loads,
    normalize_scores
)

logger = logging.getLogger(__name__)

class UserProfileAnalyzer:
    """
    Advanced user profile analyzer with real-time learning capabilities
    """
    
    def __init__(self, db=None, models=None, embedding_manager=None, 
                 similarity_engine=None, cache_manager=None):
        self.db = db
        self.models = models or {}
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.similarity_engine = similarity_engine or SimilarityEngine()
        self.cache_manager = cache_manager or CacheManager()
        
        # User clustering for personalization
        self.user_clusterer = KMeans(n_clusters=8, random_state=42)
        self.scaler = StandardScaler()
        
        # Profile update parameters
        self.min_interactions_for_profile = 5
        self.profile_decay_rate = 0.95  # Daily decay for temporal relevance
        self.learning_rate = 0.1
        
        # Cinematic DNA components
        self.cinematic_themes = {
            'action_adventure': ['action', 'adventure', 'thriller', 'chase', 'fight'],
            'romance_drama': ['romance', 'love', 'drama', 'emotional', 'relationship'],
            'comedy_entertainment': ['comedy', 'funny', 'humor', 'entertainment', 'lighthearted'],
            'mystery_crime': ['mystery', 'crime', 'detective', 'investigation', 'thriller'],
            'fantasy_scifi': ['fantasy', 'science fiction', 'sci-fi', 'supernatural', 'magic'],
            'family_kids': ['family', 'kids', 'children', 'animation', 'adventure'],
            'horror_suspense': ['horror', 'scary', 'suspense', 'supernatural', 'thriller'],
            'biographical_historical': ['biography', 'history', 'historical', 'true story', 'based on']
        }
        
        # Telugu cinema specific patterns
        self.telugu_patterns = {
            'mass_commercial': ['mass', 'commercial', 'entertainment', 'blockbuster'],
            'family_sentiment': ['family', 'sentiment', 'emotional', 'relationships'],
            'action_heroism': ['action', 'hero', 'heroism', 'fight', 'justice'],
            'romantic_musical': ['romance', 'music', 'songs', 'dance', 'love'],
            'social_message': ['social', 'message', 'awareness', 'issue', 'society']
        }
    
    def build_user_profile(self, user_id: int, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Build comprehensive user profile with real-time updates
        
        Args:
            user_id: User ID
            force_refresh: Force profile rebuild
        
        Returns:
            Dict[str, Any]: Comprehensive user profile
        """
        # Check cache first
        if not force_refresh:
            cache_key = self.cache_manager.get_user_cache_key(user_id, "profile")
            cached_profile = self.cache_manager.get(cache_key)
            if cached_profile:
                logger.debug(f"Retrieved cached profile for user {user_id}")
                return cached_profile
        
        try:
            # Get user data
            user = self._get_user(user_id)
            if not user:
                return self._create_cold_start_profile(user_id)
            
            # Get user interactions
            interactions = self._get_user_interactions(user_id)
            if len(interactions) < self.min_interactions_for_profile:
                return self._create_minimal_profile(user, interactions)
            
            # Build comprehensive profile
            profile = {
                'user_id': user_id,
                'last_updated': datetime.utcnow().isoformat(),
                'profile_version': '3.0',
                
                # Core preferences
                'explicit_preferences': self._extract_explicit_preferences(user),
                'implicit_preferences': self._extract_implicit_preferences(interactions),
                
                # Advanced analysis
                'cinematic_dna': self._analyze_cinematic_dna(interactions),
                'behavioral_patterns': self._analyze_behavioral_patterns(interactions),
                'temporal_patterns': self._analyze_temporal_patterns(interactions),
                'engagement_metrics': self._calculate_engagement_metrics(interactions),
                
                # Personalization features
                'user_embedding': self.embedding_manager.get_user_embedding(user_id).tolist(),
                'preference_clusters': self._identify_preference_clusters(interactions),
                'recommendation_context': self._build_recommendation_context(interactions),
                
                # Quality metrics
                'profile_completeness': 0.0,
                'confidence_score': 0.0,
                'diversity_score': 0.0
            }
            
            # Calculate derived metrics
            profile['profile_completeness'] = self._calculate_profile_completeness(profile)
            profile['confidence_score'] = self._calculate_confidence_score(profile, interactions)
            profile['diversity_score'] = self._calculate_preference_diversity(interactions)
            
            # User segmentation
            profile['user_segment'] = self._determine_user_segment(profile)
            
            # Cache the profile
            cache_key = self.cache_manager.get_user_cache_key(user_id, "profile")
            self.cache_manager.set(cache_key, profile, ttl=3600)  # Cache for 1 hour
            
            logger.info(f"Built profile for user {user_id} with {len(interactions)} interactions")
            return profile
            
        except Exception as e:
            logger.error(f"Error building profile for user {user_id}: {e}")
            return self._create_error_profile(user_id)
    
    def update_profile_realtime(self, user_id: int, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user profile in real-time based on new interaction
        
        Args:
            user_id: User ID
            interaction_data: New interaction data
        
        Returns:
            Dict[str, Any]: Updated profile sections
        """
        try:
            # Update user embedding first
            self.embedding_manager.update_user_embedding(user_id, interaction_data, self.learning_rate)
            
            # Get current profile
            current_profile = self.build_user_profile(user_id)
            
            # Extract interaction features
            content_id = interaction_data.get('content_id')
            interaction_type = interaction_data.get('interaction_type', 'view')
            rating = interaction_data.get('rating')
            
            # Update specific profile sections
            updates = {
                'last_interaction': datetime.utcnow().isoformat(),
                'interaction_type': interaction_type,
                'updated_sections': []
            }
            
            # Update implicit preferences
            if interaction_type in ['favorite', 'like', 'rating']:
                implicit_update = self._update_implicit_preferences(
                    current_profile.get('implicit_preferences', {}),
                    interaction_data
                )
                updates['implicit_preferences'] = implicit_update
                updates['updated_sections'].append('implicit_preferences')
            
            # Update behavioral patterns
            behavioral_update = self._update_behavioral_patterns(
                current_profile.get('behavioral_patterns', {}),
                interaction_data
            )
            updates['behavioral_patterns'] = behavioral_update
            updates['updated_sections'].append('behavioral_patterns')
            
            # Update engagement metrics
            engagement_update = self._update_engagement_metrics(
                current_profile.get('engagement_metrics', {}),
                interaction_data
            )
            updates['engagement_metrics'] = engagement_update
            updates['updated_sections'].append('engagement_metrics')
            
            # Invalidate cached profile to force refresh on next request
            cache_key = self.cache_manager.get_user_cache_key(user_id, "profile")
            self.cache_manager.delete(cache_key)
            
            logger.debug(f"Updated profile for user {user_id} with {interaction_type} interaction")
            return updates
            
        except Exception as e:
            logger.error(f"Error updating profile for user {user_id}: {e}")
            return {'error': str(e)}
    
    def _get_user(self, user_id: int):
        """Get user from database"""
        if not self.models.get('User') or not self.db:
            return None
        return self.models['User'].query.get(user_id)
    
    def _get_user_interactions(self, user_id: int) -> List[Any]:
        """Get user interactions from database"""
        if not self.models.get('UserInteraction') or not self.db:
            return []
        
        # Get recent interactions (last 6 months for relevance)
        cutoff_date = datetime.utcnow() - timedelta(days=180)
        
        interactions = self.models['UserInteraction'].query.filter(
            self.models['UserInteraction'].user_id == user_id,
            self.models['UserInteraction'].timestamp >= cutoff_date
        ).order_by(self.models['UserInteraction'].timestamp.desc()).all()
        
        return interactions
    
    def _extract_explicit_preferences(self, user) -> Dict[str, Any]:
        """Extract user's explicit preferences from profile"""
        try:
            return {
                'preferred_languages': safe_json_loads(getattr(user, 'preferred_languages', '[]')),
                'preferred_genres': safe_json_loads(getattr(user, 'preferred_genres', '[]')),
                'location': getattr(user, 'location', ''),
                'created_at': getattr(user, 'created_at', datetime.utcnow()).isoformat()
            }
        except Exception as e:
            logger.warning(f"Error extracting explicit preferences: {e}")
            return {
                'preferred_languages': ['telugu', 'english'],  # Default Telugu-first
                'preferred_genres': [],
                'location': '',
                'created_at': datetime.utcnow().isoformat()
            }
    
    def _extract_implicit_preferences(self, interactions: List[Any]) -> Dict[str, Any]:
        """Extract implicit preferences from user interactions"""
        if not interactions:
            return {
                'total_interactions': 0,
                'genre_preferences': {},
                'language_preferences': {},
                'content_type_preferences': {},
                'rating_patterns': {},
                'interaction_types': {}
            }
        
        # Collect interaction data
        genres = []
        languages = []
        content_types = []
        ratings = []
        interaction_types = Counter()
        
        # Weight recent interactions more heavily
        now = datetime.utcnow()
        
        for interaction in interactions:
            # Calculate recency weight
            days_ago = (now - interaction.timestamp).days
            recency_weight = math.exp(-days_ago / 30)  # Exponential decay over 30 days
            
            interaction_types[interaction.interaction_type] += recency_weight
            
            # Get content for this interaction
            content = self._get_content_for_interaction(interaction)
            if not content:
                continue
            
            # Extract content features with recency weighting
            if content.genres:
                content_genres = safe_json_loads(content.genres)
                for genre in content_genres:
                    genres.extend([genre] * int(recency_weight * 10))
            
            if content.languages:
                content_languages = safe_json_loads(content.languages)
                for language in content_languages:
                    languages.extend([language] * int(recency_weight * 10))
            
            content_types.extend([content.content_type] * int(recency_weight * 10))
            
            if interaction.rating:
                ratings.append({
                    'rating': interaction.rating,
                    'weight': recency_weight,
                    'timestamp': interaction.timestamp.isoformat()
                })
        
        # Calculate preference scores
        genre_counter = Counter(genres)
        language_counter = Counter(languages)
        type_counter = Counter(content_types)
        
        return {
            'total_interactions': len(interactions),
            'genre_preferences': {
                'counts': dict(genre_counter.most_common(10)),
                'top_genres': [g for g, _ in genre_counter.most_common(5)],
                'diversity': len(set(genres)) / max(len(genres), 1)
            },
            'language_preferences': {
                'counts': dict(language_counter.most_common(5)),
                'top_languages': [l for l, _ in language_counter.most_common(3)],
                'telugu_preference': language_counter.get('Telugu', 0) / max(len(languages), 1)
            },
            'content_type_preferences': {
                'counts': dict(type_counter),
                'primary_type': type_counter.most_common(1)[0][0] if type_counter else 'movie'
            },
            'rating_patterns': {
                'average_rating': np.mean([r['rating'] for r in ratings]) if ratings else 0,
                'rating_std': np.std([r['rating'] for r in ratings]) if ratings else 0,
                'total_ratings': len(ratings),
                'rating_distribution': Counter([r['rating'] for r in ratings])
            },
            'interaction_types': dict(interaction_types)
        }
    
    def _analyze_cinematic_dna(self, interactions: List[Any]) -> Dict[str, Any]:
        """Analyze user's cinematic DNA based on content consumption"""
        if not interactions:
            return {'themes': {}, 'telugu_patterns': {}, 'sophistication': 0.0}
        
        # Collect content overviews and metadata
        all_text = []
        telugu_content_count = 0
        total_content = 0
        
        for interaction in interactions:
            content = self._get_content_for_interaction(interaction)
            if not content:
                continue
            
            total_content += 1
            
            # Collect text for theme analysis
            text_parts = []
            if content.title:
                text_parts.append(content.title)
            if content.overview:
                text_parts.append(content.overview)
            
            all_text.append(' '.join(text_parts).lower())
            
            # Check for Telugu content
            if content.languages:
                languages = safe_json_loads(content.languages)
                if any('telugu' in lang.lower() for lang in languages):
                    telugu_content_count += 1
        
        # Analyze themes
        theme_scores = {}
        for theme, keywords in self.cinematic_themes.items():
            score = 0
            for text in all_text:
                for keyword in keywords:
                    if keyword in text:
                        score += 1
            theme_scores[theme] = score / max(len(all_text), 1)
        
        # Analyze Telugu-specific patterns
        telugu_pattern_scores = {}
        for pattern, keywords in self.telugu_patterns.items():
            score = 0
            for text in all_text:
                for keyword in keywords:
                    if keyword in text:
                        score += 1
            telugu_pattern_scores[pattern] = score / max(len(all_text), 1)
        
        # Calculate sophistication level
        sophistication = self._calculate_cinematic_sophistication(interactions)
        
        return {
            'themes': theme_scores,
            'telugu_patterns': telugu_pattern_scores,
            'sophistication': sophistication,
            'telugu_content_ratio': telugu_content_count / max(total_content, 1),
            'dominant_theme': max(theme_scores.items(), key=lambda x: x[1])[0] if theme_scores else None
        }
    
    def _analyze_behavioral_patterns(self, interactions: List[Any]) -> Dict[str, Any]:
        """Analyze user's behavioral patterns"""
        if not interactions:
            return {}
        
        # Time-based patterns
        hour_distribution = Counter()
        day_distribution = Counter()
        
        # Interaction patterns
        session_lengths = []
        current_session = []
        session_gap_threshold = timedelta(hours=2)
        
        interactions_sorted = sorted(interactions, key=lambda x: x.timestamp)
        
        for i, interaction in enumerate(interactions_sorted):
            hour_distribution[interaction.timestamp.hour] += 1
            day_distribution[interaction.timestamp.weekday()] += 1
            
            # Session analysis
            if (current_session and 
                interaction.timestamp - current_session[-1].timestamp > session_gap_threshold):
                # End current session
                if len(current_session) > 1:
                    session_duration = (current_session[-1].timestamp - current_session[0].timestamp).seconds / 60
                    session_lengths.append(session_duration)
                current_session = [interaction]
            else:
                current_session.append(interaction)
        
        # Add final session
        if len(current_session) > 1:
            session_duration = (current_session[-1].timestamp - current_session[0].timestamp).seconds / 60
            session_lengths.append(session_duration)
        
        return {
            'peak_hours': [h for h, _ in hour_distribution.most_common(3)],
            'active_days': [d for d, _ in day_distribution.most_common(3)],
            'session_stats': {
                'average_session_length': np.mean(session_lengths) if session_lengths else 0,
                'total_sessions': len(session_lengths),
                'longest_session': max(session_lengths) if session_lengths else 0
            },
            'activity_consistency': np.std(list(hour_distribution.values())) if hour_distribution else 0,
            'weekend_ratio': (day_distribution[5] + day_distribution[6]) / max(sum(day_distribution.values()), 1)
        }
    
    def _analyze_temporal_patterns(self, interactions: List[Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in user behavior"""
        if not interactions:
            return {}
        
        # Group interactions by week to see trends
        weekly_activity = defaultdict(int)
        monthly_activity = defaultdict(int)
        
        for interaction in interactions:
            # Week-based grouping
            week_key = interaction.timestamp.strftime('%Y-W%U')
            weekly_activity[week_key] += 1
            
            # Month-based grouping
            month_key = interaction.timestamp.strftime('%Y-%m')
            monthly_activity[month_key] += 1
        
        # Calculate trends
        weekly_counts = list(weekly_activity.values())
        monthly_counts = list(monthly_activity.values())
        
        return {
            'weekly_trend': self._calculate_trend(weekly_counts),
            'monthly_trend': self._calculate_trend(monthly_counts),
            'activity_stability': 1 - np.std(weekly_counts) / max(np.mean(weekly_counts), 1) if weekly_counts else 0,
            'recent_activity_change': self._calculate_recent_activity_change(interactions)
        }
    
    def _calculate_engagement_metrics(self, interactions: List[Any]) -> Dict[str, Any]:
        """Calculate user engagement metrics"""
        if not interactions:
            return {
                'engagement_score': 0.0,
                'interaction_diversity': 0.0,
                'rating_engagement': 0.0,
                'discovery_rate': 0.0
            }
        
        # Interaction type diversity
        interaction_types = Counter(i.interaction_type for i in interactions)
        type_diversity = len(interaction_types) / len(interactions) if interactions else 0
        
        # Rating engagement
        ratings = [i for i in interactions if i.rating is not None]
        rating_engagement = len(ratings) / len(interactions) if interactions else 0
        
        # Discovery rate (unique content consumed)
        unique_content = len(set(i.content_id for i in interactions))
        discovery_rate = unique_content / len(interactions) if interactions else 0
        
        # Overall engagement score
        engagement_score = (
            min(len(interactions) / 100, 1.0) * 0.3 +  # Activity volume
            type_diversity * 0.3 +  # Interaction diversity
            rating_engagement * 0.2 +  # Rating participation
            discovery_rate * 0.2  # Content discovery
        )
        
        return {
            'engagement_score': engagement_score,
            'interaction_diversity': type_diversity,
            'rating_engagement': rating_engagement,
            'discovery_rate': discovery_rate,
            'total_interactions': len(interactions),
            'unique_content': unique_content
        }
    
    def _identify_preference_clusters(self, interactions: List[Any]) -> Dict[str, Any]:
        """Identify user preference clusters using ML"""
        if len(interactions) < 10:
            return {'clusters': [], 'primary_cluster': None}
        
        try:
            # Create feature vectors for clustering
            features = []
            
            for interaction in interactions:
                content = self._get_content_for_interaction(interaction)
                if not content:
                    continue
                
                # Create feature vector
                feature_vector = []
                
                # Genre features (one-hot encoded)
                genres = safe_json_loads(content.genres or '[]')
                common_genres = ['Action', 'Drama', 'Comedy', 'Romance', 'Thriller', 'Horror', 'Sci-Fi', 'Animation']
                for genre in common_genres:
                    feature_vector.append(1 if genre in genres else 0)
                
                # Content type features
                feature_vector.append(1 if content.content_type == 'movie' else 0)
                feature_vector.append(1 if content.content_type == 'tv' else 0)
                feature_vector.append(1 if content.content_type == 'anime' else 0)
                
                # Rating feature
                feature_vector.append(content.rating / 10.0 if content.rating else 0.5)
                
                # Language features
                languages = safe_json_loads(content.languages or '[]')
                feature_vector.append(1 if any('telugu' in lang.lower() for lang in languages) else 0)
                feature_vector.append(1 if any('english' in lang.lower() for lang in languages) else 0)
                
                features.append(feature_vector)
            
            if len(features) < 5:
                return {'clusters': [], 'primary_cluster': None}
            
            # Normalize features
            features_array = np.array(features)
            features_normalized = self.scaler.fit_transform(features_array)
            
            # Perform clustering
            n_clusters = min(3, len(features) // 3)  # Adaptive cluster count
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(features_normalized)
            
            # Analyze clusters
            cluster_analysis = {}
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_interactions = [interactions[i] for i in cluster_indices if i < len(interactions)]
                
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_interactions),
                    'dominant_features': self._analyze_cluster_features(cluster_interactions)
                }
            
            # Find primary cluster
            primary_cluster = max(cluster_analysis.items(), key=lambda x: x[1]['size'])[0]
            
            return {
                'clusters': cluster_analysis,
                'primary_cluster': primary_cluster,
                'cluster_count': n_clusters
            }
            
        except Exception as e:
            logger.warning(f"Error in preference clustering: {e}")
            return {'clusters': [], 'primary_cluster': None}
    
    def _build_recommendation_context(self, interactions: List[Any]) -> Dict[str, Any]:
        """Build context for recommendations"""
        if not interactions:
            return {'context_type': 'cold_start'}
        
        recent_interactions = [i for i in interactions 
                             if i.timestamp > datetime.utcnow() - timedelta(days=7)]
        
        context = {
            'context_type': 'active_user',
            'recent_activity_level': len(recent_interactions),
            'primary_content_type': self._get_primary_content_type(interactions),
            'discovery_mode': len(set(i.content_id for i in recent_interactions)) / max(len(recent_interactions), 1),
            'quality_preference': self._calculate_quality_preference(interactions),
            'language_context': self._get_language_context(interactions)
        }
        
        return context
    
    # Helper methods
    def _get_content_for_interaction(self, interaction):
        """Get content object for interaction"""
        if not self.models.get('Content'):
            return None
        return self.models['Content'].query.get(interaction.content_id)
    
    def _calculate_cinematic_sophistication(self, interactions: List[Any]) -> float:
        """Calculate user's cinematic sophistication level"""
        if not interactions:
            return 0.0
        
        sophistication_factors = []
        
        for interaction in interactions:
            content = self._get_content_for_interaction(interaction)
            if not content:
                continue
            
            # Rating factor (higher ratings suggest discerning taste)
            if content.rating:
                rating_factor = min(content.rating / 10.0, 1.0)
                sophistication_factors.append(rating_factor)
            
            # Vote count factor (less mainstream = more sophisticated)
            if content.vote_count:
                # Inverse relationship with vote count
                vote_factor = 1.0 / (1.0 + content.vote_count / 1000.0)
                sophistication_factors.append(vote_factor)
        
        return np.mean(sophistication_factors) if sophistication_factors else 0.5
    
    def _calculate_trend(self, values: List[int]) -> str:
        """Calculate trend direction from time series values"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_recent_activity_change(self, interactions: List[Any]) -> float:
        """Calculate change in recent activity levels"""
        if len(interactions) < 10:
            return 0.0
        
        # Split interactions into recent and older
        midpoint = len(interactions) // 2
        recent = interactions[:midpoint]
        older = interactions[midpoint:]
        
        recent_rate = len(recent) / 30  # Assuming recent = last 30 days
        older_rate = len(older) / 30
        
        if older_rate == 0:
            return 1.0 if recent_rate > 0 else 0.0
        
        return (recent_rate - older_rate) / older_rate
    
    def _calculate_profile_completeness(self, profile: Dict[str, Any]) -> float:
        """Calculate profile completeness score"""
        completeness_factors = []
        
        # Explicit preferences
        explicit = profile.get('explicit_preferences', {})
        if explicit.get('preferred_languages'):
            completeness_factors.append(0.2)
        if explicit.get('preferred_genres'):
            completeness_factors.append(0.2)
        
        # Implicit preferences
        implicit = profile.get('implicit_preferences', {})
        if implicit.get('total_interactions', 0) > 10:
            completeness_factors.append(0.3)
        
        # Cinematic DNA
        dna = profile.get('cinematic_dna', {})
        if dna.get('themes'):
            completeness_factors.append(0.2)
        
        # Behavioral patterns
        behavioral = profile.get('behavioral_patterns', {})
        if behavioral.get('session_stats'):
            completeness_factors.append(0.1)
        
        return sum(completeness_factors)
    
    def _calculate_confidence_score(self, profile: Dict[str, Any], interactions: List[Any]) -> float:
        """Calculate confidence in profile accuracy"""
        confidence_factors = []
        
        # Interaction volume
        interaction_count = len(interactions)
        volume_confidence = min(interaction_count / 50, 1.0)
        confidence_factors.append(volume_confidence * 0.4)
        
        # Interaction diversity
        interaction_types = set(i.interaction_type for i in interactions)
        diversity_confidence = len(interaction_types) / 5  # Assuming 5 main types
        confidence_factors.append(min(diversity_confidence, 1.0) * 0.3)
        
        # Temporal consistency
        if len(interactions) > 5:
            recent_interactions = len([i for i in interactions 
                                     if i.timestamp > datetime.utcnow() - timedelta(days=30)])
            temporal_confidence = min(recent_interactions / 10, 1.0)
            confidence_factors.append(temporal_confidence * 0.3)
        
        return sum(confidence_factors)
    
    def _calculate_preference_diversity(self, interactions: List[Any]) -> float:
        """Calculate diversity of user preferences"""
        if not interactions:
            return 0.0
        
        # Collect genres and content types
        all_genres = []
        all_types = []
        
        for interaction in interactions:
            content = self._get_content_for_interaction(interaction)
            if not content:
                continue
            
            if content.genres:
                genres = safe_json_loads(content.genres)
                all_genres.extend(genres)
            
            all_types.append(content.content_type)
        
        # Calculate diversity metrics
        genre_diversity = len(set(all_genres)) / max(len(all_genres), 1)
        type_diversity = len(set(all_types)) / max(len(all_types), 1)
        
        return (genre_diversity + type_diversity) / 2
    
    def _determine_user_segment(self, profile: Dict[str, Any]) -> str:
        """Determine user segment based on profile"""
        engagement = profile.get('engagement_metrics', {}).get('engagement_score', 0)
        confidence = profile.get('confidence_score', 0)
        interactions = profile.get('implicit_preferences', {}).get('total_interactions', 0)
        
        if confidence > 0.8 and engagement > 0.7:
            return 'power_user'
        elif confidence > 0.6 and interactions > 20:
            return 'active_user'
        elif interactions > 5:
            return 'regular_user'
        else:
            return 'new_user'
    
    def _create_cold_start_profile(self, user_id: int) -> Dict[str, Any]:
        """Create profile for new users"""
        return {
            'user_id': user_id,
            'profile_type': 'cold_start',
            'last_updated': datetime.utcnow().isoformat(),
            'explicit_preferences': {
                'preferred_languages': ['telugu', 'english'],  # Default Telugu-first
                'preferred_genres': [],
                'location': ''
            },
            'implicit_preferences': {
                'total_interactions': 0
            },
            'cinematic_dna': {
                'themes': {},
                'sophistication': 0.5
            },
            'user_embedding': np.zeros(128).tolist(),
            'profile_completeness': 0.1,
            'confidence_score': 0.0,
            'user_segment': 'new_user',
            'recommendation_context': {
                'context_type': 'cold_start',
                'language_context': 'telugu_first'
            }
        }
    
    def _create_minimal_profile(self, user, interactions: List[Any]) -> Dict[str, Any]:
        """Create minimal profile for users with few interactions"""
        explicit_prefs = self._extract_explicit_preferences(user)
        
        return {
            'user_id': user.id,
            'profile_type': 'minimal',
            'last_updated': datetime.utcnow().isoformat(),
            'explicit_preferences': explicit_prefs,
            'implicit_preferences': self._extract_implicit_preferences(interactions),
            'profile_completeness': 0.3,
            'confidence_score': 0.2,
            'user_segment': 'new_user'
        }
    
    def _create_error_profile(self, user_id: int) -> Dict[str, Any]:
        """Create error fallback profile"""
        return {
            'user_id': user_id,
            'profile_type': 'error',
            'error': 'Failed to build profile',
            'last_updated': datetime.utcnow().isoformat(),
            'user_segment': 'unknown'
        }
    
    # Real-time update methods
    def _update_implicit_preferences(self, current_prefs: Dict[str, Any], 
                                   interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update implicit preferences with new interaction"""
        # This would contain incremental update logic
        # For now, return current preferences with interaction count increment
        updated = current_prefs.copy()
        updated['total_interactions'] = updated.get('total_interactions', 0) + 1
        return updated
    
    def _update_behavioral_patterns(self, current_patterns: Dict[str, Any],
                                  interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update behavioral patterns with new interaction"""
        updated = current_patterns.copy()
        # Add incremental pattern updates here
        return updated
    
    def _update_engagement_metrics(self, current_metrics: Dict[str, Any],
                                 interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update engagement metrics with new interaction"""
        updated = current_metrics.copy()
        # Add incremental engagement updates here
        return updated
    
    def _get_primary_content_type(self, interactions: List[Any]) -> str:
        """Get user's primary content type preference"""
        type_counter = Counter()
        for interaction in interactions:
            content = self._get_content_for_interaction(interaction)
            if content:
                type_counter[content.content_type] += 1
        
        return type_counter.most_common(1)[0][0] if type_counter else 'movie'
    
    def _calculate_quality_preference(self, interactions: List[Any]) -> float:
        """Calculate user's quality preference threshold"""
        ratings = []
        for interaction in interactions:
            content = self._get_content_for_interaction(interaction)
            if content and content.rating:
                ratings.append(content.rating)
        
        return np.mean(ratings) if ratings else 7.0
    
    def _get_language_context(self, interactions: List[Any]) -> str:
        """Get user's language context"""
        telugu_count = 0
        total_count = 0
        
        for interaction in interactions:
            content = self._get_content_for_interaction(interaction)
            if content and content.languages:
                total_count += 1
                languages = safe_json_loads(content.languages)
                if any('telugu' in lang.lower() for lang in languages):
                    telugu_count += 1
        
        if total_count == 0:
            return 'telugu_first'
        
        telugu_ratio = telugu_count / total_count
        
        if telugu_ratio > 0.7:
            return 'telugu_primary'
        elif telugu_ratio > 0.3:
            return 'telugu_mixed'
        else:
            return 'international'
    
    def _analyze_cluster_features(self, cluster_interactions: List[Any]) -> Dict[str, Any]:
        """Analyze features of a preference cluster"""
        # This would analyze the dominant features of a cluster
        # For now, return basic analysis
        return {
            'size': len(cluster_interactions),
            'avg_rating': 7.0  # Placeholder
        }