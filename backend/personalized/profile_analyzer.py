# backend/personalized/profile_analyzer.py

"""
CineBrain User Profile Analyzer
==============================

Continuously learns and updates user preference models through real-time
interaction monitoring and sophisticated behavioral analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import logging
import networkx as nx
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from .utils import (
    EmbeddingManager, 
    SimilarityCalculator, 
    CacheManager,
    PerformanceOptimizer,
    LANGUAGE_PRIORITY,
    PRIORITY_LANGUAGES,
    decay_weight,
    normalize_vector
)

logger = logging.getLogger(__name__)

@dataclass
class UserInteractionEvent:
    """Structured representation of a user interaction event"""
    user_id: int
    content_id: int
    interaction_type: str
    timestamp: datetime
    rating: Optional[float] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict] = None
    context: Optional[Dict] = None

@dataclass 
class UserPreferenceProfile:
    """Comprehensive user preference profile"""
    user_id: int
    embedding: np.ndarray
    genre_preferences: Dict[str, float]
    language_preferences: Dict[str, float]
    content_type_preferences: Dict[str, float]
    quality_threshold: float
    sophistication_score: float
    engagement_level: str
    last_updated: datetime
    confidence_score: float
    temporal_patterns: Dict[str, Any]
    cinematic_dna: Dict[str, Any]

class CinematicDNAAnalyzer:
    """
    Advanced cinematic DNA analyzer that identifies deep narrative and stylistic preferences
    """
    
    def __init__(self):
        """Initialize cinematic DNA analyzer with thematic patterns"""
        
        # Define cinematic themes with Telugu cinema emphasis
        self.cinematic_themes = {
            'heroic_journey': {
                'keywords': ['hero', 'journey', 'destiny', 'courage', 'sacrifice', 'honor'],
                'telugu_emphasis': ['veera', 'yodha', 'dharma', 'karma'],
                'weight': 1.0
            },
            'family_values': {
                'keywords': ['family', 'tradition', 'values', 'legacy', 'generation', 'respect'],
                'telugu_emphasis': ['kutumbam', 'parampara', 'aadarsha'],
                'weight': 0.9
            },
            'romance_emotion': {
                'keywords': ['love', 'romance', 'passion', 'emotion', 'heart', 'relationship'],
                'telugu_emphasis': ['prema', 'ishq', 'mohabbat'],
                'weight': 0.8
            },
            'action_spectacle': {
                'keywords': ['action', 'fight', 'battle', 'war', 'combat', 'intense'],
                'telugu_emphasis': ['yuddham', 'samaaram', 'shakti'],
                'weight': 0.85
            },
            'social_message': {
                'keywords': ['social', 'message', 'society', 'change', 'awareness', 'justice'],
                'telugu_emphasis': ['nyayam', 'samaj', 'badlaav'],
                'weight': 0.7
            },
            'comedy_entertainment': {
                'keywords': ['comedy', 'humor', 'entertainment', 'fun', 'laughter', 'joy'],
                'telugu_emphasis': ['vinoda', 'hasya', 'maja'],
                'weight': 0.6
            }
        }
        
        # Director signature styles
        self.director_signatures = {
            'S.S. Rajamouli': {
                'style': ['epic_scale', 'visual_spectacle', 'emotional_core', 'cultural_pride'],
                'themes': ['heroism', 'tradition', 'loyalty', 'sacrifice'],
                'weight': 1.0
            },
            'Christopher Nolan': {
                'style': ['complex_narrative', 'time_manipulation', 'cerebral', 'practical_effects'],
                'themes': ['reality_perception', 'memory', 'sacrifice', 'obsession'],
                'weight': 0.9
            },
            'Trivikram Srinivas': {
                'style': ['dialogue_driven', 'family_drama', 'contemporary', 'witty'],
                'themes': ['family_values', 'relationships', 'humor', 'wisdom'],
                'weight': 0.95
            },
            'Koratala Siva': {
                'style': ['mass_appeal', 'social_message', 'emotional', 'commercial'],
                'themes': ['social_change', 'leadership', 'responsibility', 'justice'],
                'weight': 0.9
            }
        }
        
        logger.info("CineBrain CinematicDNAAnalyzer initialized with Telugu cinema emphasis")
    
    def analyze_user_cinematic_dna(self, user_interactions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze user's cinematic DNA based on interaction history
        
        Args:
            user_interactions: List of user interactions with content metadata
            
        Returns:
            Dict containing cinematic DNA profile
        """
        try:
            if not user_interactions:
                return self._get_default_dna_profile()
            
            theme_scores = defaultdict(float)
            style_scores = defaultdict(float)
            director_affinity = defaultdict(float)
            
            total_weight = 0
            
            for interaction in user_interactions:
                # Get interaction weight
                weight = self._get_interaction_weight(interaction)
                total_weight += weight
                
                # Analyze themes from content
                content_themes = self._extract_content_themes(interaction)
                for theme, score in content_themes.items():
                    theme_scores[theme] += score * weight
                
                # Analyze director affinity
                director = interaction.get('director', '')
                if director in self.director_signatures:
                    director_weight = self.director_signatures[director]['weight']
                    director_affinity[director] += weight * director_weight
                
                # Analyze cinematic style
                content_style = self._analyze_cinematic_style(interaction)
                for style, score in content_style.items():
                    style_scores[style] += score * weight
            
            # Normalize scores
            if total_weight > 0:
                theme_scores = {k: v/total_weight for k, v in theme_scores.items()}
                style_scores = {k: v/total_weight for k, v in style_scores.items()}
                director_affinity = {k: v/total_weight for k, v in director_affinity.items()}
            
            # Calculate sophistication and preferences
            sophistication = self._calculate_cinematic_sophistication(user_interactions)
            narrative_complexity = self._calculate_narrative_complexity_preference(user_interactions)
            
            return {
                'dominant_themes': dict(sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)[:5]),
                'preferred_styles': dict(sorted(style_scores.items(), key=lambda x: x[1], reverse=True)[:5]),
                'director_affinity': dict(sorted(director_affinity.items(), key=lambda x: x[1], reverse=True)[:3]),
                'cinematic_sophistication': sophistication,
                'narrative_complexity_preference': narrative_complexity,
                'telugu_cinema_affinity': self._calculate_telugu_affinity(user_interactions),
                'quality_over_popularity': self._calculate_quality_preference(user_interactions),
                'genre_depth': self._calculate_genre_depth(user_interactions),
                'last_analyzed': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cinematic DNA: {e}")
            return self._get_default_dna_profile()
    
    def _extract_content_themes(self, interaction: Dict) -> Dict[str, float]:
        """Extract thematic elements from content"""
        themes = {}
        
        # Analyze overview/plot
        overview = interaction.get('overview', '').lower()
        title = interaction.get('title', '').lower()
        genres = interaction.get('genres', [])
        
        combined_text = f"{title} {overview}"
        
        for theme_name, theme_data in self.cinematic_themes.items():
            score = 0.0
            
            # Check keywords
            for keyword in theme_data['keywords']:
                if keyword in combined_text:
                    score += 0.5
            
            # Check Telugu-specific terms (higher weight)
            for telugu_term in theme_data.get('telugu_emphasis', []):
                if telugu_term in combined_text:
                    score += 1.0
            
            # Genre-based scoring
            if theme_name == 'action_spectacle' and any(g in ['Action', 'Adventure', 'Thriller'] for g in genres):
                score += 0.7
            elif theme_name == 'romance_emotion' and 'Romance' in genres:
                score += 0.8
            elif theme_name == 'family_values' and any(g in ['Family', 'Drama'] for g in genres):
                score += 0.6
            
            if score > 0:
                themes[theme_name] = min(score * theme_data['weight'], 1.0)
        
        return themes
    
    def _calculate_telugu_affinity(self, interactions: List[Dict]) -> float:
        """Calculate user's affinity for Telugu cinema"""
        telugu_interactions = 0
        total_interactions = len(interactions)
        
        for interaction in interactions:
            languages = interaction.get('languages', [])
            if any('telugu' in lang.lower() for lang in languages):
                telugu_interactions += 1
        
        return telugu_interactions / total_interactions if total_interactions > 0 else 0.0

class RealTimeProfileUpdater:
    """
    Real-time profile updater that continuously learns from user interactions
    """
    
    def __init__(self, embedding_manager: EmbeddingManager, cache_manager: CacheManager):
        """
        Initialize real-time profile updater
        
        Args:
            embedding_manager: EmbeddingManager instance
            cache_manager: CacheManager instance
        """
        self.embedding_manager = embedding_manager
        self.cache_manager = cache_manager
        self.performance_optimizer = PerformanceOptimizer()
        
        # Learning parameters
        self.learning_rates = {
            'favorite': 0.3,
            'watchlist': 0.2,
            'rating': 0.25,
            'view': 0.1,
            'search': 0.05,
            'like': 0.15
        }
        
        # Interaction queue for batch processing
        self.interaction_queue = []
        self.batch_size = 10
        
        logger.info("CineBrain RealTimeProfileUpdater initialized")
    
    @PerformanceOptimizer.time_function('update_user_profile')
    def update_user_profile(self, user_id: int, interaction_event: UserInteractionEvent) -> bool:
        """
        Update user profile in real-time based on new interaction
        
        Args:
            user_id: User identifier
            interaction_event: New interaction event
            
        Returns:
            bool: Success status
        """
        try:
            # Add to interaction queue
            self.interaction_queue.append(interaction_event)
            
            # Calculate learning rate
            learning_rate = self._calculate_adaptive_learning_rate(user_id, interaction_event)
            
            # Update user embedding
            interaction_data = self._convert_event_to_interaction_data(interaction_event)
            updated_embedding = self.embedding_manager.update_user_embedding_realtime(
                user_id, interaction_data
            )
            
            # Invalidate relevant caches
            self.cache_manager.invalidate_user_cache(user_id)
            
            # Process batch if queue is full
            if len(self.interaction_queue) >= self.batch_size:
                self._process_interaction_batch()
            
            logger.info(f"Updated profile for user {user_id} with learning rate {learning_rate}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating user profile for user {user_id}: {e}")
            return False
    
    def _calculate_adaptive_learning_rate(self, user_id: int, 
                                        interaction_event: UserInteractionEvent) -> float:
        """Calculate adaptive learning rate based on context"""
        base_rate = self.learning_rates.get(interaction_event.interaction_type, 0.1)
        
        # Adjust based on interaction recency
        time_since_interaction = datetime.utcnow() - interaction_event.timestamp
        recency_factor = max(0.5, 1.0 - time_since_interaction.total_seconds() / 3600)  # Decay over 1 hour
        
        # Adjust based on user engagement level
        engagement_factor = 1.0  # Could be calculated based on user's historical activity
        
        # Adjust based on content quality
        quality_factor = 1.0
        if interaction_event.rating:
            quality_factor = min(interaction_event.rating / 10.0, 1.0)
        
        return base_rate * recency_factor * engagement_factor * quality_factor
    
    def _convert_event_to_interaction_data(self, event: UserInteractionEvent) -> Dict:
        """Convert interaction event to data format for embedding update"""
        return {
            'content_id': event.content_id,
            'interaction_type': event.interaction_type,
            'rating': event.rating,
            'timestamp': event.timestamp,
            'metadata': event.metadata or {},
            'context': event.context or {}
        }

class UserProfileAnalyzer:
    """
    Main user profile analyzer that orchestrates all profile analysis components
    """
    
    def __init__(self, db, models: Dict, embedding_manager: EmbeddingManager, 
                 cache_manager: CacheManager):
        """
        Initialize user profile analyzer
        
        Args:
            db: Database instance
            models: Dictionary of database models
            embedding_manager: EmbeddingManager instance
            cache_manager: CacheManager instance
        """
        self.db = db
        self.models = models
        self.embedding_manager = embedding_manager
        self.cache_manager = cache_manager
        
        # Initialize sub-components
        self.cinematic_dna_analyzer = CinematicDNAAnalyzer()
        self.realtime_updater = RealTimeProfileUpdater(embedding_manager, cache_manager)
        self.performance_optimizer = PerformanceOptimizer()
        
        # User profile cache
        self.user_profiles = {}
        
        logger.info("CineBrain UserProfileAnalyzer initialized")
    
    @PerformanceOptimizer.time_function('build_comprehensive_profile')
    def build_comprehensive_user_profile(self, user_id: int, 
                                       force_refresh: bool = False) -> Optional[UserPreferenceProfile]:
        """
        Build comprehensive user preference profile
        
        Args:
            user_id: User identifier
            force_refresh: Force profile regeneration
            
        Returns:
            UserPreferenceProfile or None
        """
        try:
            # Check cache first (unless force refresh)
            if not force_refresh and user_id in self.user_profiles:
                cached_profile = self.user_profiles[user_id]
                # Check if profile is recent (within 1 hour)
                if (datetime.utcnow() - cached_profile.last_updated).total_seconds() < 3600:
                    return cached_profile
            
            # Get user interactions
            user_interactions = self._get_user_interactions(user_id)
            if not user_interactions:
                return self._create_cold_start_profile(user_id)
            
            # Convert interactions to structured format
            interaction_data = self._prepare_interaction_data(user_interactions)
            
            # Create user embedding
            user_embedding = self.embedding_manager.create_user_embedding(user_id, interaction_data)
            
            # Analyze preferences
            genre_preferences = self._analyze_genre_preferences(interaction_data)
            language_preferences = self._analyze_language_preferences(interaction_data)
            content_type_preferences = self._analyze_content_type_preferences(interaction_data)
            
            # Calculate quality and sophistication metrics
            quality_threshold = self._calculate_quality_threshold(interaction_data)
            sophistication_score = self._calculate_sophistication_score(interaction_data)
            
            # Determine engagement level
            engagement_level = self._determine_engagement_level(interaction_data)
            
            # Calculate confidence score
            confidence_score = self._calculate_profile_confidence(interaction_data)
            
            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(interaction_data)
            
            # Analyze cinematic DNA
            cinematic_dna = self.cinematic_dna_analyzer.analyze_user_cinematic_dna(interaction_data)
            
            # Create comprehensive profile
            profile = UserPreferenceProfile(
                user_id=user_id,
                embedding=user_embedding,
                genre_preferences=genre_preferences,
                language_preferences=language_preferences,
                content_type_preferences=content_type_preferences,
                quality_threshold=quality_threshold,
                sophistication_score=sophistication_score,
                engagement_level=engagement_level,
                last_updated=datetime.utcnow(),
                confidence_score=confidence_score,
                temporal_patterns=temporal_patterns,
                cinematic_dna=cinematic_dna
            )
            
            # Cache the profile
            self.user_profiles[user_id] = profile
            
            logger.info(f"Built comprehensive profile for user {user_id} with confidence {confidence_score}")
            return profile
            
        except Exception as e:
            logger.error(f"Error building user profile for user {user_id}: {e}")
            return None
    
    def update_profile_realtime(self, user_id: int, interaction_event: UserInteractionEvent) -> bool:
        """
        Update user profile in real-time
        
        Args:
            user_id: User identifier
            interaction_event: New interaction event
            
        Returns:
            bool: Success status
        """
        try:
            # Update through real-time updater
            success = self.realtime_updater.update_user_profile(user_id, interaction_event)
            
            # Invalidate cached profile to force refresh on next access
            if user_id in self.user_profiles:
                del self.user_profiles[user_id]
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating profile in real-time for user {user_id}: {e}")
            return False
    
    def _get_user_interactions(self, user_id: int) -> List[Any]:
        """Get user interactions from database"""
        try:
            UserInteraction = self.models['UserInteraction']
            Content = self.models['Content']
            
            # Get interactions with content details
            interactions = self.db.session.query(UserInteraction, Content).join(
                Content, UserInteraction.content_id == Content.id
            ).filter(UserInteraction.user_id == user_id).order_by(
                UserInteraction.timestamp.desc()
            ).limit(500).all()  # Limit to recent 500 interactions
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting user interactions for user {user_id}: {e}")
            return []
    
    def _prepare_interaction_data(self, interactions: List[Tuple]) -> List[Dict]:
        """Prepare interaction data for analysis"""
        interaction_data = []
        
        for interaction, content in interactions:
            try:
                data = {
                    'content_id': content.id,
                    'interaction_type': interaction.interaction_type,
                    'rating': interaction.rating,
                    'timestamp': interaction.timestamp,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'overview': content.overview or '',
                    'director': getattr(content, 'director', ''),
                    'release_year': content.release_date.year if content.release_date else None,
                    'rating_score': content.rating or 0,
                    'popularity': content.popularity or 0
                }
                interaction_data.append(data)
                
            except Exception as e:
                logger.warning(f"Error preparing interaction data: {e}")
                continue
        
        return interaction_data
    
    def _analyze_genre_preferences(self, interaction_data: List[Dict]) -> Dict[str, float]:
        """Analyze user's genre preferences with weighted scoring"""
        genre_scores = defaultdict(float)
        total_weight = 0
        
        for interaction in interaction_data:
            # Calculate weight based on interaction type and recency
            weight = self._get_interaction_weight(interaction)
            total_weight += weight
            
            # Add genre scores
            for genre in interaction.get('genres', []):
                genre_scores[genre] += weight
        
        # Normalize scores
        if total_weight > 0:
            genre_scores = {genre: score/total_weight for genre, score in genre_scores.items()}
        
        # Sort and return top genres
        sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_genres[:10])  # Top 10 genres
    
    def _analyze_language_preferences(self, interaction_data: List[Dict]) -> Dict[str, float]:
        """Analyze user's language preferences with Telugu priority"""
        language_scores = defaultdict(float)
        total_weight = 0
        
        for interaction in interaction_data:
            weight = self._get_interaction_weight(interaction)
            total_weight += weight
            
            # Add language scores with priority weighting
            for language in interaction.get('languages', []):
                lang_priority = LANGUAGE_PRIORITY.get(language.lower(), 0.5)
                language_scores[language] += weight * lang_priority
        
        # Normalize scores
        if total_weight > 0:
            language_scores = {lang: score/total_weight for lang, score in language_scores.items()}
        
        # Sort by preference
        sorted_languages = sorted(language_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_languages[:5])  # Top 5 languages
    
    def _get_interaction_weight(self, interaction: Dict) -> float:
        """Calculate weight for an interaction based on type and recency"""
        # Base weights by interaction type
        type_weights = {
            'favorite': 3.0,
            'watchlist': 2.5,
            'rating': 2.0,
            'view': 1.5,
            'search': 1.0,
            'like': 1.2
        }
        
        base_weight = type_weights.get(interaction.get('interaction_type'), 1.0)
        
        # Apply recency decay
        timestamp = interaction.get('timestamp', datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        days_ago = (datetime.utcnow() - timestamp).days
        recency_weight = decay_weight(days_ago)
        
        # Apply rating boost
        rating_weight = 1.0
        if interaction.get('rating'):
            rating_weight = min(interaction['rating'] / 8.0, 1.5)  # Boost for high ratings
        
        return base_weight * recency_weight * rating_weight
    
    def _create_cold_start_profile(self, user_id: int) -> UserPreferenceProfile:
        """Create a default profile for new users"""
        return UserPreferenceProfile(
            user_id=user_id,
            embedding=np.zeros(self.embedding_manager.embedding_dim),
            genre_preferences={'Action': 0.3, 'Drama': 0.3, 'Comedy': 0.2, 'Romance': 0.2},
            language_preferences={'Telugu': 0.4, 'English': 0.3, 'Hindi': 0.3},
            content_type_preferences={'movie': 0.6, 'tv': 0.3, 'anime': 0.1},
            quality_threshold=7.0,
            sophistication_score=0.5,
            engagement_level='new_user',
            last_updated=datetime.utcnow(),
            confidence_score=0.1,
            temporal_patterns={},
            cinematic_dna={}
        )
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the profile analyzer"""
        return self.performance_optimizer.get_performance_stats()