# backend/personalized/profile_analyzer.py 

"""
CineBrain Advanced Profile Analyzer
Comprehensive user behavior analysis with Cinematic DNA extraction

This module provides deep understanding of user preferences through:
- Behavioral pattern analysis
- Cinematic DNA extraction and modeling
- Preference embedding generation
- Real-time adaptation to user feedback
- Cultural and linguistic preference modeling
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import re
from typing import List, Dict, Any, Tuple, Optional, Set
import hashlib
from sqlalchemy import func, desc, and_, or_
import networkx as nx
from scipy.stats import entropy
from scipy.spatial.distance import cosine, euclidean
import pytz

logger = logging.getLogger(__name__)

# Regional and cultural configuration
TELUGU_CULTURAL_MARKERS = {
    'themes': ['family_values', 'tradition', 'heroism', 'justice', 'cultural_pride'],
    'styles': ['grand_scale', 'emotional_depth', 'music_centric', 'visual_spectacle'],
    'directors': ['S.S. Rajamouli', 'Trivikram Srinivas', 'Koratala Siva', 'Sukumar'],
    'keywords': ['tollywood', 'telugu', 'andhra', 'telangana', 'hyderabad', 'vijayawada']
}

CINEMATIC_DNA_DIMENSIONS = {
    'narrative_complexity': {
        'linear': 0.2,
        'non_linear': 0.8,
        'multi_layered': 1.0,
        'experimental': 0.9
    },
    'emotional_tone': {
        'light_hearted': 0.3,
        'balanced': 0.5,
        'intense': 0.8,
        'dark': 0.9,
        'uplifting': 0.4
    },
    'production_scale': {
        'intimate': 0.3,
        'medium': 0.5,
        'large': 0.8,
        'epic': 1.0
    },
    'cultural_authenticity': {
        'universal': 0.4,
        'regional': 0.8,
        'local': 1.0,
        'traditional': 0.9
    }
}

class CinematicDNAAnalyzer:
    """
    Advanced Cinematic DNA Analysis System
    Extracts deep cinematic preferences and creates DNA profiles
    """
    
    def __init__(self):
        self.narrative_patterns = {
            'hero_journey': {
                'keywords': ['journey', 'quest', 'adventure', 'transformation', 'destiny'],
                'weight': 1.0,
                'cultural_variants': {
                    'telugu': ['veera', 'yodhude', 'parakrami', 'dharma'],
                    'indian': ['dharma', 'karma', 'moksha', 'samskaras']
                }
            },
            'revenge_arc': {
                'keywords': ['revenge', 'vengeance', 'justice', 'betrayal', 'retribution'],
                'weight': 0.9,
                'cultural_variants': {
                    'telugu': ['badla', 'pratikaramu', 'nyayam'],
                    'indian': ['badla', 'insaaf', 'nyaya']
                }
            },
            'family_saga': {
                'keywords': ['family', 'generations', 'legacy', 'tradition', 'values'],
                'weight': 0.8,
                'cultural_variants': {
                    'telugu': ['kutumbam', 'vamsam', 'parampara'],
                    'indian': ['parivaar', 'khandaan', 'sanskriti']
                }
            },
            'love_story': {
                'keywords': ['love', 'romance', 'relationship', 'marriage', 'couple'],
                'weight': 0.7,
                'cultural_variants': {
                    'telugu': ['prema', 'sneham', 'kalyanam'],
                    'indian': ['pyaar', 'mohabbat', 'ishq']
                }
            },
            'social_commentary': {
                'keywords': ['society', 'reform', 'change', 'awareness', 'system'],
                'weight': 0.85,
                'cultural_variants': {
                    'telugu': ['samajam', 'sudhar', 'jagratha'],
                    'indian': ['samaj', 'sudhar', 'chetna']
                }
            }
        }
        
        self.directorial_signatures = {
            'S.S. Rajamouli': {
                'style_markers': ['epic_scale', 'visual_spectacle', 'emotional_core', 'mythological'],
                'themes': ['heroism', 'sacrifice', 'duty', 'honor'],
                'technical_excellence': 1.0,
                'cultural_authenticity': 1.0
            },
            'Christopher Nolan': {
                'style_markers': ['complex_narrative', 'time_manipulation', 'practical_effects'],
                'themes': ['reality', 'memory', 'sacrifice', 'obsession'],
                'technical_excellence': 1.0,
                'cultural_authenticity': 0.5
            },
            'Denis Villeneuve': {
                'style_markers': ['atmospheric', 'philosophical', 'visual_poetry'],
                'themes': ['humanity', 'communication', 'existence'],
                'technical_excellence': 0.9,
                'cultural_authenticity': 0.6
            }
        }
        
        self.cinematic_styles = {
            'commercial_blockbuster': {
                'indicators': ['action', 'mass', 'entertainment', 'commercial', 'star'],
                'telugu_weight': 1.5,
                'global_weight': 1.0
            },
            'art_cinema': {
                'indicators': ['artistic', 'parallel', 'independent', 'festival', 'experimental'],
                'telugu_weight': 0.8,
                'global_weight': 1.2
            },
            'family_entertainer': {
                'indicators': ['family', 'entertainment', 'values', 'comedy', 'drama'],
                'telugu_weight': 1.3,
                'global_weight': 0.9
            }
        }
        
        self.embedding_engine = TruncatedSVD(n_components=100, random_state=42)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        
    def extract_cinematic_dna(self, content_list: List[Any]) -> Dict[str, Any]:
        """
        Extract comprehensive cinematic DNA from user's content interaction history
        
        Args:
            content_list: List of content items user has interacted with
            
        Returns:
            Comprehensive cinematic DNA profile
        """
        if not content_list:
            return self._get_default_dna_profile()
        
        dna_profile = {
            'narrative_preferences': {},
            'style_affinities': {},
            'director_preferences': {},
            'genre_sophistication': {},
            'cultural_alignment': {},
            'production_scale_preference': 'medium',
            'narrative_complexity_tolerance': 0.5,
            'emotional_range_preference': {},
            'temporal_patterns': {},
            'quality_expectations': {},
            'cinematic_sophistication_score': 0.0,
            'telugu_cultural_affinity': 0.0,
            'indian_cultural_affinity': 0.0,
            'global_cinema_exposure': 0.0
        }
        
        # Analyze narrative patterns
        dna_profile['narrative_preferences'] = self._analyze_narrative_patterns(content_list)
        
        # Analyze directorial style preferences
        dna_profile['director_preferences'] = self._analyze_director_preferences(content_list)
        
        # Analyze genre sophistication
        dna_profile['genre_sophistication'] = self._analyze_genre_sophistication(content_list)
        
        # Analyze cultural alignment
        dna_profile['cultural_alignment'] = self._analyze_cultural_alignment(content_list)
        
        # Analyze production preferences
        dna_profile['production_scale_preference'] = self._analyze_production_scale(content_list)
        
        # Analyze emotional preferences
        dna_profile['emotional_range_preference'] = self._analyze_emotional_preferences(content_list)
        
        # Analyze temporal viewing patterns
        dna_profile['temporal_patterns'] = self._analyze_temporal_patterns(content_list)
        
        # Calculate sophistication score
        dna_profile['cinematic_sophistication_score'] = self._calculate_sophistication_score(content_list)
        
        # Calculate cultural affinities
        dna_profile['telugu_cultural_affinity'] = self._calculate_telugu_affinity(content_list)
        dna_profile['indian_cultural_affinity'] = self._calculate_indian_affinity(content_list)
        dna_profile['global_cinema_exposure'] = self._calculate_global_exposure(content_list)
        
        return dna_profile
    
    def _analyze_narrative_patterns(self, content_list: List[Any]) -> Dict[str, float]:
        """Analyze user's preference for different narrative patterns"""
        pattern_scores = defaultdict(float)
        
        for content in content_list:
            content_text = f"{content.title} {content.overview or ''}"
            
            # Extract narrative patterns
            for pattern_name, pattern_data in self.narrative_patterns.items():
                score = self._calculate_pattern_score(content_text, pattern_data, content)
                pattern_scores[pattern_name] += score
        
        # Normalize scores
        if pattern_scores:
            max_score = max(pattern_scores.values())
            if max_score > 0:
                pattern_scores = {k: v/max_score for k, v in pattern_scores.items()}
        
        return dict(pattern_scores)
    
    def _analyze_director_preferences(self, content_list: List[Any]) -> Dict[str, float]:
        """Analyze user's affinity for specific directorial styles"""
        director_scores = defaultdict(float)
        
        for content in content_list:
            content_text = f"{content.title} {content.overview or ''}"
            
            # Check for directorial style markers
            for director, signature in self.directorial_signatures.items():
                style_match = 0
                for marker in signature['style_markers']:
                    if any(keyword in content_text.lower() for keyword in marker.split('_')):
                        style_match += 1
                
                if style_match > 0:
                    director_scores[director] += (style_match / len(signature['style_markers'])) * signature['technical_excellence']
        
        # Normalize
        if director_scores:
            total_score = sum(director_scores.values())
            if total_score > 0:
                director_scores = {k: v/total_score for k, v in director_scores.items()}
        
        return dict(director_scores)
    
    def _analyze_genre_sophistication(self, content_list: List[Any]) -> Dict[str, float]:
        """Analyze sophistication level across different genres"""
        genre_ratings = defaultdict(list)
        genre_counts = Counter()
        
        for content in content_list:
            if content.genres:
                try:
                    genres = json.loads(content.genres)
                    for genre in genres:
                        genre_counts[genre] += 1
                        if content.rating:
                            genre_ratings[genre].append(content.rating)
                except:
                    pass
        
        sophistication = {}
        for genre, ratings in genre_ratings.items():
            if ratings:
                avg_rating = np.mean(ratings)
                rating_std = np.std(ratings)
                count = genre_counts[genre]
                
                # Sophistication = high average rating + consistency + exploration
                sophistication[genre] = {
                    'average_rating': avg_rating,
                    'consistency': 1 - (rating_std / 10) if rating_std > 0 else 1.0,
                    'exploration_depth': min(count / 10, 1.0),
                    'sophistication_score': (avg_rating / 10) * 0.5 + (1 - rating_std / 10) * 0.3 + min(count / 10, 1.0) * 0.2
                }
        
        return sophistication
    
    def _analyze_cultural_alignment(self, content_list: List[Any]) -> Dict[str, float]:
        """Analyze user's cultural content preferences"""
        cultural_scores = {
            'telugu_traditional': 0.0,
            'indian_mainstream': 0.0,
            'indian_regional': 0.0,
            'western_commercial': 0.0,
            'international_arthouse': 0.0,
            'global_blockbuster': 0.0
        }
        
        total_content = len(content_list)
        
        for content in content_list:
            languages = json.loads(content.languages or '[]')
            content_text = f"{content.title} {content.overview or ''}".lower()
            
            # Telugu content analysis
            if any('telugu' in lang.lower() for lang in languages):
                telugu_markers = sum(1 for marker in TELUGU_CULTURAL_MARKERS['keywords'] 
                                   if marker in content_text)
                cultural_scores['telugu_traditional'] += (1 + telugu_markers * 0.2)
            
            # Indian content analysis
            indian_languages = ['hindi', 'tamil', 'malayalam', 'kannada', 'bengali', 'marathi']
            if any(any(lang_name in lang.lower() for lang_name in indian_languages) for lang in languages):
                cultural_scores['indian_mainstream'] += 1
            
            # Western content
            if any('english' in lang.lower() for lang in languages):
                if content.rating and content.rating >= 7.0:
                    cultural_scores['western_commercial'] += 1
                else:
                    cultural_scores['global_blockbuster'] += 1
        
        # Normalize by total content
        if total_content > 0:
            cultural_scores = {k: v/total_content for k, v in cultural_scores.items()}
        
        return cultural_scores
    
    def _calculate_sophistication_score(self, content_list: List[Any]) -> float:
        """Calculate overall cinematic sophistication score"""
        factors = []
        
        # Rating distribution analysis
        ratings = [c.rating for c in content_list if c.rating]
        if ratings:
            avg_rating = np.mean(ratings)
            factors.append(min(avg_rating / 10, 1.0))
        
        # Genre diversity
        all_genres = []
        for content in content_list:
            if content.genres:
                try:
                    all_genres.extend(json.loads(content.genres))
                except:
                    pass
        
        if all_genres:
            unique_genres = len(set(all_genres))
            genre_diversity = min(unique_genres / 15, 1.0)  # Max 15 genres for full score
            factors.append(genre_diversity)
        
        # Content type diversity
        content_types = [c.content_type for c in content_list]
        type_diversity = len(set(content_types)) / 3  # movie, tv, anime
        factors.append(min(type_diversity, 1.0))
        
        # Language diversity
        all_languages = []
        for content in content_list:
            if content.languages:
                try:
                    all_languages.extend(json.loads(content.languages))
                except:
                    pass
        
        if all_languages:
            lang_diversity = min(len(set(all_languages)) / 5, 1.0)
            factors.append(lang_diversity)
        
        # Temporal spread (preference for different eras)
        years = [c.release_date.year for c in content_list if c.release_date]
        if years:
            year_spread = (max(years) - min(years)) / 50  # 50 years for full score
            factors.append(min(year_spread, 1.0))
        
        return np.mean(factors) if factors else 0.5
    
    def _calculate_telugu_affinity(self, content_list: List[Any]) -> float:
        """Calculate affinity for Telugu cinema"""
        telugu_content = 0
        telugu_quality = []
        total_content = len(content_list)
        
        for content in content_list:
            languages = json.loads(content.languages or '[]')
            if any('telugu' in lang.lower() for lang in languages):
                telugu_content += 1
                if content.rating:
                    telugu_quality.append(content.rating)
        
        if total_content == 0:
            return 0.0
        
        # Base affinity from percentage of Telugu content
        base_affinity = telugu_content / total_content
        
        # Quality bonus
        quality_bonus = 0
        if telugu_quality:
            avg_quality = np.mean(telugu_quality)
            if avg_quality >= 7.0:
                quality_bonus = 0.2
        
        return min(base_affinity + quality_bonus, 1.0)
    
    def _calculate_indian_affinity(self, content_list: List[Any]) -> float:
        """Calculate affinity for Indian cinema overall"""
        indian_languages = ['telugu', 'hindi', 'tamil', 'malayalam', 'kannada', 'bengali', 'marathi']
        indian_content = 0
        total_content = len(content_list)
        
        for content in content_list:
            languages = json.loads(content.languages or '[]')
            if any(any(lang_name in lang.lower() for lang_name in indian_languages) 
                   for lang in languages):
                indian_content += 1
        
        return indian_content / max(total_content, 1)
    
    def _calculate_global_exposure(self, content_list: List[Any]) -> float:
        """Calculate exposure to global cinema"""
        global_languages = ['english', 'french', 'spanish', 'japanese', 'korean', 'german']
        global_content = 0
        total_content = len(content_list)
        
        for content in content_list:
            languages = json.loads(content.languages or '[]')
            if any(any(lang_name in lang.lower() for lang_name in global_languages) 
                   for lang in languages):
                global_content += 1
        
        return global_content / max(total_content, 1)
    
    def _get_default_dna_profile(self) -> Dict[str, Any]:
        """Return default DNA profile for new users"""
        return {
            'narrative_preferences': {
                'hero_journey': 0.5,
                'love_story': 0.6,
                'family_saga': 0.7
            },
            'style_affinities': {},
            'director_preferences': {},
            'genre_sophistication': {},
            'cultural_alignment': {
                'telugu_traditional': 0.8,
                'indian_mainstream': 0.6,
                'global_blockbuster': 0.4
            },
            'production_scale_preference': 'large',
            'cinematic_sophistication_score': 0.5,
            'telugu_cultural_affinity': 0.8,
            'indian_cultural_affinity': 0.7,
            'global_cinema_exposure': 0.3
        }
    
    def _calculate_pattern_score(self, content_text: str, pattern_data: Dict, content: Any) -> float:
        """Calculate how well content matches a narrative pattern"""
        text_lower = content_text.lower()
        base_score = 0
        
        # Check base keywords
        for keyword in pattern_data['keywords']:
            if keyword in text_lower:
                base_score += 1
        
        # Check cultural variants if applicable
        languages = json.loads(content.languages or '[]')
        for lang in languages:
            lang_lower = lang.lower()
            if 'telugu' in lang_lower and 'telugu' in pattern_data.get('cultural_variants', {}):
                telugu_keywords = pattern_data['cultural_variants']['telugu']
                for keyword in telugu_keywords:
                    if keyword in text_lower:
                        base_score += 1.5  # Higher weight for cultural alignment
            elif any(indian_lang in lang_lower for indian_lang in ['hindi', 'tamil', 'malayalam']) and 'indian' in pattern_data.get('cultural_variants', {}):
                indian_keywords = pattern_data['cultural_variants']['indian']
                for keyword in indian_keywords:
                    if keyword in text_lower:
                        base_score += 1.2
        
        # Normalize by number of total keywords
        total_keywords = len(pattern_data['keywords'])
        cultural_keywords = sum(len(v) for v in pattern_data.get('cultural_variants', {}).values())
        
        normalized_score = base_score / max(total_keywords + cultural_keywords, 1)
        return min(normalized_score * pattern_data['weight'], 1.0)

class UserBehaviorAnalyzer:
    """
    Advanced User Behavior Analysis Engine
    Understands user interaction patterns, preferences, and behavioral signals
    """
    
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.interaction_weights = {
            'favorite': 3.0,
            'watchlist': 2.5,
            'rating': 2.0,
            'view': 1.5,
            'like': 1.2,
            'search': 1.0,
            'click': 0.8
        }
        
        self.time_decay_factor = 0.95  # Decay factor per day
        self.min_interactions_for_stable_profile = 15
        
    def analyze_user_behavior(self, user_id: int) -> Dict[str, Any]:
        """
        Comprehensive analysis of user behavior patterns
        
        Args:
            user_id: User ID to analyze
            
        Returns:
            Detailed behavioral analysis
        """
        try:
            # Get user interactions
            interactions = self.models['UserInteraction'].query.filter_by(
                user_id=user_id
            ).order_by(self.models['UserInteraction'].timestamp.desc()).all()
            
            if not interactions:
                return self._get_default_behavior_profile()
            
            behavior_profile = {
                'engagement_patterns': self._analyze_engagement_patterns(interactions),
                'temporal_behavior': self._analyze_temporal_behavior(interactions),
                'content_exploration': self._analyze_exploration_behavior(interactions),
                'rating_behavior': self._analyze_rating_behavior(interactions),
                'session_patterns': self._analyze_session_patterns(interactions),
                'preference_stability': self._analyze_preference_stability(interactions),
                'discovery_openness': self._analyze_discovery_openness(interactions),
                'quality_sensitivity': self._analyze_quality_sensitivity(interactions),
                'cultural_preference_strength': self._analyze_cultural_preferences(interactions),
                'behavior_confidence': self._calculate_behavior_confidence(interactions),
                'recommendation_receptiveness': self._analyze_recommendation_receptiveness(interactions)
            }
            
            return behavior_profile
            
        except Exception as e:
            logger.error(f"Error analyzing user behavior for user {user_id}: {e}")
            return self._get_default_behavior_profile()
    
    def _analyze_engagement_patterns(self, interactions: List[Any]) -> Dict[str, Any]:
        """Analyze how user engages with different types of content"""
        engagement_by_type = defaultdict(lambda: defaultdict(int))
        engagement_by_genre = defaultdict(lambda: defaultdict(int))
        
        for interaction in interactions:
            content = self.models['Content'].query.get(interaction.content_id)
            if content:
                engagement_by_type[content.content_type][interaction.interaction_type] += 1
                
                if content.genres:
                    try:
                        genres = json.loads(content.genres)
                        for genre in genres:
                            engagement_by_genre[genre][interaction.interaction_type] += 1
                    except:
                        pass
        
        # Calculate engagement scores
        engagement_scores = {}
        for content_type, interactions_dict in engagement_by_type.items():
            total_score = sum(count * self.interaction_weights.get(interaction_type, 1.0) 
                            for interaction_type, count in interactions_dict.items())
            engagement_scores[content_type] = total_score
        
        # Find dominant engagement pattern
        dominant_type = max(engagement_scores.keys(), key=lambda k: engagement_scores[k]) if engagement_scores else 'movie'
        
        return {
            'engagement_by_type': dict(engagement_by_type),
            'engagement_scores': engagement_scores,
            'dominant_content_type': dominant_type,
            'total_weighted_engagement': sum(engagement_scores.values()),
            'engagement_diversity': len(engagement_scores) / 3  # Assuming 3 main types
        }
    
    def _analyze_temporal_behavior(self, interactions: List[Any]) -> Dict[str, Any]:
        """Analyze when and how frequently user interacts"""
        if not interactions:
            return {}
        
        # Time analysis
        hours = [interaction.timestamp.hour for interaction in interactions]
        weekdays = [interaction.timestamp.weekday() for interaction in interactions]
        
        # Session analysis
        sessions = self._identify_sessions(interactions)
        
        # Activity frequency
        first_interaction = min(interaction.timestamp for interaction in interactions)
        last_interaction = max(interaction.timestamp for interaction in interactions)
        total_days = max((last_interaction - first_interaction).days, 1)
        
        return {
            'peak_hours': [h for h, _ in Counter(hours).most_common(3)],
            'preferred_days': [d for d, _ in Counter(weekdays).most_common(3)],
            'average_session_length': np.mean([s['duration_minutes'] for s in sessions]) if sessions else 0,
            'sessions_per_day': len(sessions) / total_days,
            'activity_consistency': self._calculate_activity_consistency(interactions),
            'binge_tendency': self._calculate_binge_tendency(interactions),
            'total_sessions': len(sessions)
        }
    
    def _analyze_exploration_behavior(self, interactions: List[Any]) -> Dict[str, Any]:
        """Analyze user's willingness to explore new content"""
        content_ids = [i.content_id for i in interactions]
        unique_content = len(set(content_ids))
        total_interactions = len(interactions)
        
        # Genre exploration
        all_genres = []
        for interaction in interactions:
            content = self.models['Content'].query.get(interaction.content_id)
            if content and content.genres:
                try:
                    all_genres.extend(json.loads(content.genres))
                except:
                    pass
        
        unique_genres = len(set(all_genres))
        
        # Language exploration
        all_languages = []
        for interaction in interactions:
            content = self.models['Content'].query.get(interaction.content_id)
            if content and content.languages:
                try:
                    all_languages.extend(json.loads(content.languages))
                except:
                    pass
        
        unique_languages = len(set(all_languages))
        
        # Calculate exploration scores
        content_exploration = unique_content / total_interactions
        genre_exploration = min(unique_genres / 15, 1.0)  # Normalized to max 15 genres
        language_exploration = min(unique_languages / 5, 1.0)  # Normalized to max 5 languages
        
        return {
            'content_diversity_ratio': content_exploration,
            'genre_exploration_score': genre_exploration,
            'language_exploration_score': language_exploration,
            'overall_exploration_score': (content_exploration + genre_exploration + language_exploration) / 3,
            'unique_content_count': unique_content,
            'unique_genres_count': unique_genres,
            'unique_languages_count': unique_languages,
            'exploration_tendency': 'high' if genre_exploration > 0.7 else 'medium' if genre_exploration > 0.4 else 'low'
        }
    
    def _analyze_rating_behavior(self, interactions: List[Any]) -> Dict[str, Any]:
        """Analyze user's rating patterns and tendencies"""
        rating_interactions = [i for i in interactions if i.rating is not None]
        
        if not rating_interactions:
            return {'has_ratings': False}
        
        ratings = [i.rating for i in rating_interactions]
        
        return {
            'has_ratings': True,
            'total_ratings': len(ratings),
            'average_rating': np.mean(ratings),
            'rating_variance': np.var(ratings),
            'rating_range': max(ratings) - min(ratings),
            'rating_distribution': dict(Counter([round(r) for r in ratings])),
            'rating_tendency': self._classify_rating_tendency(ratings),
            'rating_consistency': 1 - (np.std(ratings) / 10),
            'harsh_rating_threshold': np.percentile(ratings, 25),
            'generous_rating_threshold': np.percentile(ratings, 75)
        }
    
    def _classify_rating_tendency(self, ratings: List[float]) -> str:
        """Classify user's rating tendency"""
        avg_rating = np.mean(ratings)
        std_rating = np.std(ratings)
        
        if avg_rating >= 8.0:
            return 'generous' if std_rating < 1.5 else 'positive_but_discriminating'
        elif avg_rating <= 6.0:
            return 'harsh' if std_rating < 1.5 else 'critical_but_fair'
        else:
            return 'balanced' if std_rating < 2.0 else 'highly_variable'
    
    def _calculate_behavior_confidence(self, interactions: List[Any]) -> float:
        """Calculate confidence level in behavioral analysis"""
        factors = []
        
        # Interaction volume
        interaction_count = len(interactions)
        volume_score = min(interaction_count / 50, 1.0)  # 50 interactions for full confidence
        factors.append(volume_score)
        
        # Interaction diversity
        interaction_types = set(i.interaction_type for i in interactions)
        diversity_score = len(interaction_types) / 6  # Assuming 6 main interaction types
        factors.append(min(diversity_score, 1.0))
        
        # Temporal spread
        if len(interactions) > 1:
            timestamps = [i.timestamp for i in interactions]
            span_days = (max(timestamps) - min(timestamps)).days
            temporal_score = min(span_days / 30, 1.0)  # 30 days for full confidence
            factors.append(temporal_score)
        
        # Rating availability
        rating_interactions = [i for i in interactions if i.rating is not None]
        rating_score = min(len(rating_interactions) / 10, 1.0)  # 10 ratings for full confidence
        factors.append(rating_score)
        
        return np.mean(factors) if factors else 0.0
    
    def _get_default_behavior_profile(self) -> Dict[str, Any]:
        """Default behavior profile for new users"""
        return {
            'engagement_patterns': {
                'dominant_content_type': 'movie',
                'engagement_diversity': 0.3
            },
            'temporal_behavior': {
                'peak_hours': [19, 20, 21],  # Evening
                'preferred_days': [5, 6],    # Weekend
                'binge_tendency': 0.3
            },
            'content_exploration': {
                'overall_exploration_score': 0.5,
                'exploration_tendency': 'medium'
            },
            'rating_behavior': {
                'has_ratings': False
            },
            'behavior_confidence': 0.1,
            'recommendation_receptiveness': 0.8  # New users are typically open
        }

class PreferenceEmbeddingEngine:
    """
    Advanced Preference Embedding Engine
    Creates dense vector representations of user preferences for similarity matching
    """
    
    def __init__(self, embedding_size: int = 128):
        self.embedding_size = embedding_size
        self.genre_embeddings = {}
        self.language_embeddings = {}
        self.director_embeddings = {}
        self.user_embeddings_cache = {}
        
        # Initialize embedding spaces
        self._initialize_embedding_spaces()
        
    def _initialize_embedding_spaces(self):
        """Initialize embedding spaces for different content attributes"""
        # Genre embedding space
        genres = [
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
            'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War'
        ]
        
        for i, genre in enumerate(genres):
            # Create embeddings with some semantic relationships
            embedding = np.random.randn(self.embedding_size) * 0.1
            
            # Add genre-specific patterns
            if genre in ['Action', 'Adventure', 'Thriller']:
                embedding[:20] += 0.5  # High-energy cluster
            elif genre in ['Drama', 'Biography', 'History']:
                embedding[20:40] += 0.5  # Serious content cluster
            elif genre in ['Comedy', 'Family', 'Animation']:
                embedding[40:60] += 0.5  # Light content cluster
            elif genre in ['Horror', 'Mystery', 'Crime']:
                embedding[60:80] += 0.5  # Dark content cluster
            
            self.genre_embeddings[genre] = embedding
        
        # Language embeddings with cultural clustering
        languages = ['Telugu', 'English', 'Hindi', 'Tamil', 'Malayalam', 'Kannada', 'Japanese', 'Korean']
        for i, language in enumerate(languages):
            embedding = np.random.randn(self.embedding_size) * 0.1
            
            # Cultural clusters
            if language in ['Telugu', 'Hindi', 'Tamil', 'Malayalam', 'Kannada']:
                embedding[:30] += 0.7  # Indian languages cluster
                if language == 'Telugu':
                    embedding[30:40] += 0.8  # Telugu specific
            elif language == 'English':
                embedding[40:70] += 0.6  # Western content
            elif language in ['Japanese', 'Korean']:
                embedding[70:100] += 0.6  # East Asian content
            
            self.language_embeddings[language] = embedding
    
    def create_user_embedding(self, user_id: int, cinematic_dna: Dict[str, Any], 
                            behavior_profile: Dict[str, Any], content_history: List[Any]) -> np.ndarray:
        """
        Create comprehensive user preference embedding
        
        Args:
            user_id: User identifier
            cinematic_dna: User's cinematic DNA profile
            behavior_profile: User's behavior analysis
            content_history: User's content interaction history
            
        Returns:
            Dense vector representation of user preferences
        """
        try:
            # Initialize embedding
            user_embedding = np.zeros(self.embedding_size)
            
            # 1. Genre preferences (25% of embedding)
            genre_component = self._encode_genre_preferences(content_history)
            user_embedding[:32] = genre_component[:32]
            
            # 2. Language preferences (20% of embedding)
            language_component = self._encode_language_preferences(content_history, cinematic_dna)
            user_embedding[32:58] = language_component[:26]
            
            # 3. Cinematic DNA (25% of embedding)
            dna_component = self._encode_cinematic_dna(cinematic_dna)
            user_embedding[58:90] = dna_component[:32]
            
            # 4. Behavioral patterns (20% of embedding)
            behavior_component = self._encode_behavior_patterns(behavior_profile)
            user_embedding[90:116] = behavior_component[:26]
            
            # 5. Quality and recency preferences (10% of embedding)
            quality_component = self._encode_quality_preferences(content_history)
            user_embedding[116:] = quality_component[:12]
            
            # Normalize embedding
            norm = np.linalg.norm(user_embedding)
            if norm > 0:
                user_embedding = user_embedding / norm
            
            # Cache the embedding
            self.user_embeddings_cache[user_id] = {
                'embedding': user_embedding,
                'created_at': datetime.utcnow(),
                'content_count': len(content_history)
            }
            
            return user_embedding
            
        except Exception as e:
            logger.error(f"Error creating user embedding for user {user_id}: {e}")
            return np.random.randn(self.embedding_size) * 0.1
    
    def _encode_genre_preferences(self, content_history: List[Any]) -> np.ndarray:
        """Encode genre preferences into vector space"""
        genre_weights = defaultdict(float)
        total_weight = 0
        
        for content in content_history:
            if content.genres:
                try:
                    genres = json.loads(content.genres)
                    weight = self._get_content_weight(content)
                    for genre in genres:
                        genre_weights[genre] += weight
                        total_weight += weight
                except:
                    pass
        
        # Normalize weights
        if total_weight > 0:
            genre_weights = {k: v/total_weight for k, v in genre_weights.items()}
        
        # Create weighted average of genre embeddings
        genre_embedding = np.zeros(self.embedding_size)
        for genre, weight in genre_weights.items():
            if genre in self.genre_embeddings:
                genre_embedding += self.genre_embeddings[genre] * weight
        
        return genre_embedding
    
    def _encode_language_preferences(self, content_history: List[Any], 
                                   cinematic_dna: Dict[str, Any]) -> np.ndarray:
        """Encode language preferences with cultural affinity"""
        language_weights = defaultdict(float)
        total_weight = 0
        
        for content in content_history:
            if content.languages:
                try:
                    languages = json.loads(content.languages)
                    weight = self._get_content_weight(content)
                    for language in languages:
                        language_weights[language] += weight
                        total_weight += weight
                except:
                    pass
        
        # Apply cultural affinity boost
        telugu_affinity = cinematic_dna.get('telugu_cultural_affinity', 0)
        indian_affinity = cinematic_dna.get('indian_cultural_affinity', 0)
        
        if 'Telugu' in language_weights:
            language_weights['Telugu'] *= (1 + telugu_affinity)
        
        indian_languages = ['Hindi', 'Tamil', 'Malayalam', 'Kannada']
        for lang in indian_languages:
            if lang in language_weights:
                language_weights[lang] *= (1 + indian_affinity * 0.5)
        
        # Normalize
        total_weight = sum(language_weights.values())
        if total_weight > 0:
            language_weights = {k: v/total_weight for k, v in language_weights.items()}
        
        # Create weighted average
        language_embedding = np.zeros(self.embedding_size)
        for language, weight in language_weights.items():
            if language in self.language_embeddings:
                language_embedding += self.language_embeddings[language] * weight
        
        return language_embedding
    
    def _encode_cinematic_dna(self, cinematic_dna: Dict[str, Any]) -> np.ndarray:
        """Encode cinematic DNA characteristics"""
        dna_embedding = np.zeros(self.embedding_size)
        
        # Narrative preferences
        narrative_prefs = cinematic_dna.get('narrative_preferences', {})
        for i, (pattern, score) in enumerate(narrative_prefs.items()):
            if i < 8:  # Use first 8 dimensions for narrative patterns
                dna_embedding[i] = score
        
        # Style preferences
        style_prefs = cinematic_dna.get('style_affinities', {})
        for i, (style, score) in enumerate(style_prefs.items()):
            if i < 8:  # Use next 8 dimensions for style
                dna_embedding[8 + i] = score
        
        # Sophistication and cultural alignment
        dna_embedding[16] = cinematic_dna.get('cinematic_sophistication_score', 0.5)
        dna_embedding[17] = cinematic_dna.get('telugu_cultural_affinity', 0.5)
        dna_embedding[18] = cinematic_dna.get('indian_cultural_affinity', 0.5)
        dna_embedding[19] = cinematic_dna.get('global_cinema_exposure', 0.5)
        
        return dna_embedding
    
    def _encode_behavior_patterns(self, behavior_profile: Dict[str, Any]) -> np.ndarray:
        """Encode behavioral patterns"""
        behavior_embedding = np.zeros(self.embedding_size)
        
        # Engagement patterns
        engagement = behavior_profile.get('engagement_patterns', {})
        behavior_embedding[0] = engagement.get('engagement_diversity', 0.5)
        
        # Exploration behavior
        exploration = behavior_profile.get('content_exploration', {})
        behavior_embedding[1] = exploration.get('overall_exploration_score', 0.5)
        behavior_embedding[2] = exploration.get('genre_exploration_score', 0.5)
        behavior_embedding[3] = exploration.get('language_exploration_score', 0.5)
        
        # Temporal behavior
        temporal = behavior_profile.get('temporal_behavior', {})
        behavior_embedding[4] = temporal.get('binge_tendency', 0.3)
        behavior_embedding[5] = temporal.get('activity_consistency', 0.5)
        
        # Rating behavior
        rating = behavior_profile.get('rating_behavior', {})
        if rating.get('has_ratings', False):
            behavior_embedding[6] = rating.get('rating_consistency', 0.5)
            avg_rating = rating.get('average_rating', 7.0)
            behavior_embedding[7] = avg_rating / 10  # Normalize to [0,1]
        
        return behavior_embedding
    
    def _encode_quality_preferences(self, content_history: List[Any]) -> np.ndarray:
        """Encode quality and recency preferences"""
        quality_embedding = np.zeros(self.embedding_size)
        
        if not content_history:
            return quality_embedding
        
        # Average rating preference
        ratings = [c.rating for c in content_history if c.rating]
        if ratings:
            quality_embedding[0] = np.mean(ratings) / 10
        
        # Recency preference (preference for newer content)
        current_year = datetime.now().year
        recent_content = sum(1 for c in content_history 
                           if c.release_date and c.release_date.year >= current_year - 2)
        quality_embedding[1] = recent_content / len(content_history)
        
        # Popularity preference
        popularities = [c.popularity for c in content_history if c.popularity]
        if popularities:
            quality_embedding[2] = np.mean(popularities) / 100  # Normalize
        
        return quality_embedding
    
    def _get_content_weight(self, content: Any) -> float:
        """Calculate weight for content based on rating and recency"""
        base_weight = 1.0
        
        # Rating boost
        if content.rating:
            rating_boost = (content.rating / 10) * 0.5
            base_weight += rating_boost
        
        # Recency boost
        if content.release_date:
            days_old = (datetime.now().date() - content.release_date).days
            recency_boost = math.exp(-days_old / 365) * 0.3  # Exponential decay over a year
            base_weight += recency_boost
        
        return base_weight
    
    def calculate_user_similarity(self, user_id1: int, user_id2: int) -> float:
        """Calculate similarity between two users based on their embeddings"""
        if user_id1 in self.user_embeddings_cache and user_id2 in self.user_embeddings_cache:
            embedding1 = self.user_embeddings_cache[user_id1]['embedding']
            embedding2 = self.user_embeddings_cache[user_id2]['embedding']
            
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return max(0, similarity)  # Ensure non-negative
        
        return 0.0

class ProfileAnalyzer:
    """
    Main Profile Analyzer orchestrating all analysis components
    """
    
    def __init__(self, db, models, services, cache=None):
        self.db = db
        self.models = models
        self.services = services
        self.cache = cache
        
        # Initialize analysis engines
        self.cinematic_dna_analyzer = CinematicDNAAnalyzer()
        self.behavior_analyzer = UserBehaviorAnalyzer(db, models)
        self.embedding_engine = PreferenceEmbeddingEngine()
        
        # Cache settings
        self.profile_cache_duration = 3600  # 1 hour
        self.embedding_cache_duration = 7200  # 2 hours
    
    def build_comprehensive_profile(self, user_id: int) -> Dict[str, Any]:
        """
        Build a comprehensive user profile combining all analysis components
        
        Args:
            user_id: User ID to analyze
            
        Returns:
            Complete user profile with all analysis components
        """
        cache_key = f"cinebrain:profile:{user_id}"
        
        # Check cache first
        if self.cache:
            try:
                cached_profile = self.cache.get(cache_key)
                if cached_profile:
                    return json.loads(cached_profile)
            except:
                pass
        
        try:
            # Get user's content interaction history
            content_history = self._get_user_content_history(user_id)
            
            # Build profile components
            cinematic_dna = self.cinematic_dna_analyzer.extract_cinematic_dna(content_history)
            behavior_profile = self.behavior_analyzer.analyze_user_behavior(user_id)
            
            # Create user embedding
            user_embedding = self.embedding_engine.create_user_embedding(
                user_id, cinematic_dna, behavior_profile, content_history
            )
            
            # Compile comprehensive profile
            comprehensive_profile = {
                'user_id': user_id,
                'profile_version': '3.0',
                'cinematic_dna': cinematic_dna,
                'behavior_profile': behavior_profile,
                'user_embedding': user_embedding.tolist(),  # Convert to list for JSON serialization
                'content_history_size': len(content_history),
                'profile_confidence': self._calculate_profile_confidence(
                    cinematic_dna, behavior_profile, content_history
                ),
                'personalization_readiness': self._assess_personalization_readiness(
                    cinematic_dna, behavior_profile, content_history
                ),
                'recommendations_strategy': self._determine_recommendation_strategy(
                    cinematic_dna, behavior_profile, content_history
                ),
                'profile_created_at': datetime.utcnow().isoformat(),
                'next_update_due': (datetime.utcnow() + timedelta(hours=24)).isoformat()
            }
            
            # Cache the profile
            if self.cache:
                try:
                    self.cache.set(
                        cache_key, 
                        json.dumps(comprehensive_profile, default=str),
                        timeout=self.profile_cache_duration
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache profile for user {user_id}: {e}")
            
            return comprehensive_profile
            
        except Exception as e:
            logger.error(f"Error building comprehensive profile for user {user_id}: {e}")
            return self._get_default_comprehensive_profile(user_id)
    
    def _get_user_content_history(self, user_id: int) -> List[Any]:
        """Get user's content interaction history"""
        try:
            # Get user interactions
            interactions = self.models['UserInteraction'].query.filter_by(
                user_id=user_id
            ).order_by(self.models['UserInteraction'].timestamp.desc()).all()
            
            # Get content objects
            content_ids = list(set(i.content_id for i in interactions))
            content_history = self.models['Content'].query.filter(
                self.models['Content'].id.in_(content_ids)
            ).all()
            
            return content_history
            
        except Exception as e:
            logger.error(f"Error getting content history for user {user_id}: {e}")
            return []
    
    def _calculate_profile_confidence(self, cinematic_dna: Dict[str, Any],
                                    behavior_profile: Dict[str, Any],
                                    content_history: List[Any]) -> float:
        """Calculate confidence level in the profile analysis"""
        factors = []
        
        # Content volume factor
        content_count = len(content_history)
        volume_factor = min(content_count / 30, 1.0)  # 30 items for full confidence
        factors.append(volume_factor)
        
        # Behavior confidence
        behavior_confidence = behavior_profile.get('behavior_confidence', 0.0)
        factors.append(behavior_confidence)
        
        # Cinematic DNA sophistication
        sophistication = cinematic_dna.get('cinematic_sophistication_score', 0.5)
        factors.append(sophistication)
        
        # Cultural alignment confidence
        cultural_scores = cinematic_dna.get('cultural_alignment', {})
        cultural_confidence = max(cultural_scores.values()) if cultural_scores else 0.5
        factors.append(cultural_confidence)
        
        return np.mean(factors)
    
    def _assess_personalization_readiness(self, cinematic_dna: Dict[str, Any],
                                         behavior_profile: Dict[str, Any],
                                         content_history: List[Any]) -> str:
        """Assess how ready the user is for personalized recommendations"""
        profile_confidence = self._calculate_profile_confidence(
            cinematic_dna, behavior_profile, content_history
        )
        
        content_count = len(content_history)
        has_ratings = behavior_profile.get('rating_behavior', {}).get('has_ratings', False)
        
        if profile_confidence >= 0.8 and content_count >= 20:
            return 'high'
        elif profile_confidence >= 0.6 and content_count >= 10:
            return 'medium'
        elif profile_confidence >= 0.4 and content_count >= 5:
            return 'low'
        else:
            return 'cold_start'
    
    def _determine_recommendation_strategy(self, cinematic_dna: Dict[str, Any],
                                         behavior_profile: Dict[str, Any],
                                         content_history: List[Any]) -> str:
        """Determine the best recommendation strategy for this user"""
        readiness = self._assess_personalization_readiness(
            cinematic_dna, behavior_profile, content_history
        )
        
        if readiness == 'high':
            # Use advanced hybrid approach
            return 'advanced_hybrid'
        elif readiness == 'medium':
            # Use content-based with some collaborative
            return 'content_collaborative_mix'
        elif readiness == 'low':
            # Use content-based primarily
            return 'content_based_primary'
        else:
            # Use popularity and cultural preferences
            return 'popularity_cultural'
    
    def _get_default_comprehensive_profile(self, user_id: int) -> Dict[str, Any]:
        """Return default profile for error cases"""
        return {
            'user_id': user_id,
            'profile_version': '3.0',
            'cinematic_dna': self.cinematic_dna_analyzer._get_default_dna_profile(),
            'behavior_profile': self.behavior_analyzer._get_default_behavior_profile(),
            'user_embedding': np.random.randn(128).tolist(),
            'content_history_size': 0,
            'profile_confidence': 0.1,
            'personalization_readiness': 'cold_start',
            'recommendations_strategy': 'popularity_cultural',
            'profile_created_at': datetime.utcnow().isoformat(),
            'next_update_due': (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }
    
    def update_profile_realtime(self, user_id: int, interaction_data: Dict[str, Any]) -> bool:
        """Update user profile in real-time based on new interaction"""
        try:
            # Invalidate cache
            cache_key = f"cinebrain:profile:{user_id}"
            if self.cache:
                self.cache.delete(cache_key)
            
            # For now, just invalidate cache - full rebuild will happen on next request
            # In production, you might want incremental updates here
            
            logger.info(f"Profile cache invalidated for user {user_id} after interaction: {interaction_data.get('interaction_type')}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating profile for user {user_id}: {e}")
            return False
    
    def get_user_similarity_candidates(self, user_id: int, limit: int = 50) -> List[int]:
        """Get users similar to the given user for collaborative filtering"""
        try:
            # Get user embedding
            if user_id not in self.embedding_engine.user_embeddings_cache:
                # Build profile to generate embedding
                self.build_comprehensive_profile(user_id)
            
            if user_id not in self.embedding_engine.user_embeddings_cache:
                return []
            
            user_embedding = self.embedding_engine.user_embeddings_cache[user_id]['embedding']
            
            # Calculate similarities with other users
            similarities = []
            for other_user_id, data in self.embedding_engine.user_embeddings_cache.items():
                if other_user_id != user_id:
                    other_embedding = data['embedding']
                    similarity = np.dot(user_embedding, other_embedding)
                    similarities.append((other_user_id, similarity))
            
            # Sort by similarity and return top candidates
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [user_id for user_id, _ in similarities[:limit]]
            
        except Exception as e:
            logger.error(f"Error getting similarity candidates for user {user_id}: {e}")
            return []