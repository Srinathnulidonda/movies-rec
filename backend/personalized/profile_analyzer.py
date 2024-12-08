# backend/personalized/profile_analyzer.py

"""
CineBrain User Profile Analyzer
Advanced user profiling with cinematic DNA analysis and behavioral modeling
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import logging
import math
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy import func, desc, and_

from .utils import (
    VectorOperations, DataProcessor, LanguagePriorityManager,
    safe_json_loads, LANGUAGE_WEIGHTS, PRIORITY_LANGUAGES
)

logger = logging.getLogger(__name__)

class CinematicDNAAnalyzer:
    """
    Analyze user's cinematic DNA - deep understanding of cinematic preferences
    """
    
    def __init__(self):
        # Advanced cinematic theme detection
        self.cinematic_themes = {
            'hero_journey': {
                'keywords': ['hero', 'journey', 'overcome', 'struggle', 'destiny', 'chosen', 'epic'],
                'weight': 1.0,
                'cultural_weight': {'telugu': 1.3, 'hindi': 1.2}  # Higher in Indian cinema
            },
            'justice_revenge': {
                'keywords': ['justice', 'revenge', 'vengeance', 'betrayal', 'redemption', 'vigilante'],
                'weight': 0.95,
                'cultural_weight': {'telugu': 1.4, 'tamil': 1.3}  # Strong in South Indian cinema
            },
            'family_honor': {
                'keywords': ['family', 'honor', 'tradition', 'legacy', 'generation', 'respect'],
                'weight': 0.9,
                'cultural_weight': {'telugu': 1.5, 'hindi': 1.4, 'tamil': 1.3}
            },
            'love_sacrifice': {
                'keywords': ['love', 'sacrifice', 'devotion', 'romantic', 'passion', 'heartbreak'],
                'weight': 0.8,
                'cultural_weight': {'hindi': 1.2, 'telugu': 1.1}
            },
            'power_corruption': {
                'keywords': ['power', 'corruption', 'politics', 'conspiracy', 'manipulation'],
                'weight': 0.85,
                'cultural_weight': {'english': 1.2, 'telugu': 1.1}
            },
            'supernatural_mythology': {
                'keywords': ['supernatural', 'mythology', 'gods', 'divine', 'spiritual', 'mystical'],
                'weight': 0.9,
                'cultural_weight': {'telugu': 1.6, 'hindi': 1.4, 'tamil': 1.3}  # Strong in Indian content
            },
            'comedy_entertainment': {
                'keywords': ['comedy', 'humor', 'funny', 'entertainment', 'lighthearted', 'fun'],
                'weight': 0.7,
                'cultural_weight': {'telugu': 1.2, 'hindi': 1.1}
            },
            'action_adventure': {
                'keywords': ['action', 'adventure', 'thrill', 'excitement', 'chase', 'fight'],
                'weight': 0.85,
                'cultural_weight': {'english': 1.2, 'telugu': 1.3}
            }
        }
        
        # Cinematic style preferences
        self.cinematic_styles = {
            'commercial_masala': {
                'indicators': ['commercial', 'masala', 'entertainment', 'mass', 'blockbuster'],
                'runtime_range': (120, 180),
                'weight': 1.0,
                'cultural_weight': {'telugu': 1.5, 'hindi': 1.4, 'tamil': 1.3}
            },
            'art_house': {
                'indicators': ['art', 'artistic', 'indie', 'independent', 'festival'],
                'rating_threshold': 7.5,
                'weight': 0.9,
                'cultural_weight': {'english': 1.3, 'telugu': 0.8}
            },
            'epic_spectacle': {
                'indicators': ['epic', 'spectacular', 'grand', 'scale', 'visual'],
                'runtime_min': 150,
                'budget_indicator': 'high',
                'weight': 1.0,
                'cultural_weight': {'telugu': 1.6, 'hindi': 1.4}  # S.S. Rajamouli influence
            },
            'intimate_drama': {
                'indicators': ['intimate', 'personal', 'character', 'emotional', 'human'],
                'genres': ['Drama', 'Biography'],
                'weight': 0.8
            },
            'high_concept': {
                'indicators': ['concept', 'innovative', 'unique', 'experimental', 'creative'],
                'weight': 0.85,
                'cultural_weight': {'english': 1.2}
            }
        }
        
        # Director style signatures
        self.director_signatures = {
            'S.S. Rajamouli': {
                'style': ['epic_scale', 'visual_spectacle', 'emotional_core', 'mythology'],
                'themes': ['heroism', 'tradition', 'loyalty', 'justice', 'family'],
                'weight': 1.0,
                'cultural_impact': 'high'
            },
            'Christopher Nolan': {
                'style': ['complex_narrative', 'time_manipulation', 'cerebral', 'practical'],
                'themes': ['reality', 'memory', 'sacrifice', 'obsession'],
                'weight': 0.95,
                'cultural_impact': 'medium'
            },
            'Puri Jagannadh': {
                'style': ['commercial', 'stylized', 'dialogues', 'characterization'],
                'themes': ['heroism', 'justice', 'style', 'attitude'],
                'weight': 0.9,
                'cultural_impact': 'high'
            },
            'Denis Villeneuve': {
                'style': ['atmospheric', 'philosophical', 'visual_poetry', 'slow_burn'],
                'themes': ['humanity', 'communication', 'existence', 'future'],
                'weight': 0.9,
                'cultural_impact': 'medium'
            }
        }
    
    def analyze_cinematic_dna(self, content_list: List[Any], 
                            user_languages: List[str] = None) -> Dict[str, Any]:
        """Analyze user's cinematic DNA from their content consumption"""
        if not content_list:
            return self._get_default_dna_profile()
        
        dna_profile = {
            'dominant_themes': {},
            'preferred_styles': {},
            'director_affinity': {},
            'cultural_preferences': {},
            'narrative_complexity': 0.0,
            'scale_preference': 'medium',
            'quality_threshold': 7.0,
            'emotional_tone_preference': {},
            'cinematic_sophistication': 0.0,
            'commercial_vs_art': 0.5,  # 0 = pure art, 1 = pure commercial
            'cultural_identity_score': 0.0
        }
        
        # Extract themes with cultural weighting
        theme_scores = self._extract_weighted_themes(content_list, user_languages)
        dna_profile['dominant_themes'] = self._normalize_scores(theme_scores)
        
        # Analyze cinematic styles
        style_scores = self._analyze_cinematic_styles(content_list, user_languages)
        dna_profile['preferred_styles'] = self._normalize_scores(style_scores)
        
        # Calculate sophistication and preferences
        dna_profile['cinematic_sophistication'] = self._calculate_sophistication(content_list)
        dna_profile['commercial_vs_art'] = self._calculate_commercial_art_ratio(content_list)
        dna_profile['narrative_complexity'] = self._analyze_narrative_complexity(content_list)
        dna_profile['quality_threshold'] = self._calculate_quality_threshold(content_list)
        dna_profile['cultural_identity_score'] = self._calculate_cultural_identity(content_list, user_languages)
        
        # Emotional tone analysis
        dna_profile['emotional_tone_preference'] = self._analyze_emotional_preferences(content_list)
        
        return dna_profile
    
    def _extract_weighted_themes(self, content_list: List[Any], 
                                user_languages: List[str] = None) -> Dict[str, float]:
        """Extract themes with cultural and language weighting"""
        theme_scores = defaultdict(float)
        
        for content in content_list:
            content_text = f"{content.title} {content.overview or ''}"
            content_languages = safe_json_loads(content.languages, [])
            
            for theme_name, theme_data in self.cinematic_themes.items():
                base_score = 0.0
                keywords = theme_data['keywords']
                base_weight = theme_data['weight']
                
                # Keyword matching
                for keyword in keywords:
                    if keyword in content_text.lower():
                        base_score += 1.0
                
                if base_score > 0:
                    # Apply cultural weighting
                    cultural_multiplier = 1.0
                    cultural_weights = theme_data.get('cultural_weight', {})
                    
                    for lang in content_languages:
                        lang_code = lang.lower()[:2]  # Get language code
                        if lang_code in cultural_weights:
                            cultural_multiplier = max(cultural_multiplier, cultural_weights[lang_code])
                    
                    # Apply user language preference bonus
                    if user_languages:
                        for user_lang in user_languages:
                            if any(user_lang.lower() in cl.lower() for cl in content_languages):
                                cultural_multiplier *= 1.2
                                break
                    
                    final_score = (base_score / len(keywords)) * base_weight * cultural_multiplier
                    theme_scores[theme_name] += final_score
        
        return dict(theme_scores)
    
    def _analyze_cinematic_styles(self, content_list: List[Any], 
                                 user_languages: List[str] = None) -> Dict[str, float]:
        """Analyze preferred cinematic styles"""
        style_scores = defaultdict(float)
        
        for content in content_list:
            content_text = f"{content.title} {content.overview or ''}".lower()
            content_languages = safe_json_loads(content.languages, [])
            content_genres = safe_json_loads(content.genres, [])
            
            for style_name, style_data in self.cinematic_styles.items():
                score = 0.0
                
                # Indicator matching
                if 'indicators' in style_data:
                    for indicator in style_data['indicators']:
                        if indicator in content_text:
                            score += 0.3
                
                # Runtime check
                if 'runtime_range' in style_data and content.runtime:
                    min_runtime, max_runtime = style_data['runtime_range']
                    if min_runtime <= content.runtime <= max_runtime:
                        score += 0.4
                elif 'runtime_min' in style_data and content.runtime:
                    if content.runtime >= style_data['runtime_min']:
                        score += 0.4
                
                # Genre check
                if 'genres' in style_data:
                    for genre in style_data['genres']:
                        if genre in content_genres:
                            score += 0.3
                
                # Rating threshold check
                if 'rating_threshold' in style_data and content.rating:
                    if content.rating >= style_data['rating_threshold']:
                        score += 0.3
                
                if score > 0:
                    # Apply cultural weighting
                    cultural_multiplier = 1.0
                    cultural_weights = style_data.get('cultural_weight', {})
                    
                    for lang in content_languages:
                        lang_key = lang.lower()
                        if lang_key in cultural_weights:
                            cultural_multiplier = max(cultural_multiplier, cultural_weights[lang_key])
                    
                    final_score = min(score, 1.0) * style_data['weight'] * cultural_multiplier
                    style_scores[style_name] += final_score
        
        return dict(style_scores)
    
    def _calculate_sophistication(self, content_list: List[Any]) -> float:
        """Calculate user's cinematic sophistication level"""
        if not content_list:
            return 0.5
        
        factors = []
        
        # Average rating factor
        ratings = [c.rating for c in content_list if c.rating]
        if ratings:
            avg_rating = np.mean(ratings)
            rating_factor = min(avg_rating / 10.0, 1.0)
            factors.append(rating_factor)
        
        # Genre diversity factor
        all_genres = []
        for content in content_list:
            genres = safe_json_loads(content.genres, [])
            all_genres.extend(genres)
        
        if all_genres:
            unique_genres = len(set(all_genres))
            total_genres = len(all_genres)
            diversity_factor = min(unique_genres / total_genres, 1.0)
            factors.append(diversity_factor)
        
        # Content type diversity
        content_types = [c.content_type for c in content_list]
        type_diversity = len(set(content_types)) / len(content_types) if content_types else 0
        factors.append(type_diversity)
        
        # Language diversity factor
        all_languages = []
        for content in content_list:
            languages = safe_json_loads(content.languages, [])
            all_languages.extend(languages)
        
        if all_languages:
            unique_languages = len(set(all_languages))
            lang_diversity = min(unique_languages / 5.0, 1.0)  # Normalize to max 5 languages
            factors.append(lang_diversity)
        
        # Runtime preference (longer content = higher sophistication)
        runtimes = [c.runtime for c in content_list if c.runtime]
        if runtimes:
            avg_runtime = np.mean(runtimes)
            runtime_factor = min(avg_runtime / 180.0, 1.0)  # Normalize to 3 hours
            factors.append(runtime_factor)
        
        return np.mean(factors) if factors else 0.5
    
    def _calculate_commercial_art_ratio(self, content_list: List[Any]) -> float:
        """Calculate preference for commercial vs art house content"""
        commercial_indicators = ['action', 'comedy', 'thriller', 'adventure', 'blockbuster']
        art_indicators = ['drama', 'documentary', 'biography', 'art', 'indie']
        
        commercial_score = 0.0
        art_score = 0.0
        
        for content in content_list:
            text = f"{content.title} {content.overview or ''}".lower()
            genres = safe_json_loads(content.genres, [])
            
            # Check commercial indicators
            for indicator in commercial_indicators:
                if indicator in text or any(indicator.title() in g for g in genres):
                    commercial_score += 1.0
            
            # Check art house indicators
            for indicator in art_indicators:
                if indicator in text or any(indicator.title() in g for g in genres):
                    art_score += 1.0
            
            # Rating-based scoring (very high ratings suggest art house appeal)
            if content.rating:
                if content.rating >= 8.5 and (content.vote_count or 0) > 1000:
                    art_score += 0.5
                elif content.rating >= 7.0 and (content.popularity or 0) > 50:
                    commercial_score += 0.5
        
        total_score = commercial_score + art_score
        if total_score == 0:
            return 0.5  # Neutral
        
        return commercial_score / total_score
    
    def _analyze_narrative_complexity(self, content_list: List[Any]) -> float:
        """Analyze preference for narrative complexity"""
        complexity_indicators = [
            'complex', 'intricate', 'layered', 'non-linear', 'puzzle', 
            'mystery', 'psychological', 'philosophical', 'cerebral'
        ]
        
        complexity_score = 0.0
        total_content = len(content_list)
        
        for content in content_list:
            text = f"{content.title} {content.overview or ''}".lower()
            
            # Check complexity indicators
            for indicator in complexity_indicators:
                if indicator in text:
                    complexity_score += 1.0
            
            # Runtime factor (longer movies often more complex)
            if content.runtime and content.runtime > 150:
                complexity_score += 0.5
            
            # Genre factor
            complex_genres = ['Mystery', 'Thriller', 'Science Fiction', 'Drama']
            genres = safe_json_loads(content.genres, [])
            if any(g in complex_genres for g in genres):
                complexity_score += 0.3
        
        return min(complexity_score / total_content, 1.0) if total_content > 0 else 0.0
    
    def _calculate_quality_threshold(self, content_list: List[Any]) -> float:
        """Calculate user's quality threshold"""
        ratings = [c.rating for c in content_list if c.rating]
        
        if not ratings:
            return 7.0  # Default threshold
        
        # Use 25th percentile as quality threshold
        threshold = np.percentile(ratings, 25)
        
        # Adjust based on average rating
        avg_rating = np.mean(ratings)
        if avg_rating < 6:
            threshold = max(threshold - 0.5, 1.0)
        elif avg_rating > 8:
            threshold = min(threshold + 0.5, 10.0)
        
        return float(threshold)
    
    def _calculate_cultural_identity(self, content_list: List[Any], 
                                   user_languages: List[str] = None) -> float:
        """Calculate cultural identity score (preference for local/regional content)"""
        indian_languages = ['telugu', 'hindi', 'tamil', 'malayalam', 'kannada', 'marathi', 'bengali']
        indian_content_count = 0
        total_content = len(content_list)
        
        for content in content_list:
            languages = safe_json_loads(content.languages, [])
            for lang in languages:
                if any(indian_lang in lang.lower() for indian_lang in indian_languages):
                    indian_content_count += 1
                    break
        
        base_score = indian_content_count / total_content if total_content > 0 else 0
        
        # Bonus for Telugu content (CineBrain's focus)
        telugu_content_count = 0
        for content in content_list:
            languages = safe_json_loads(content.languages, [])
            if any('telugu' in lang.lower() for lang in languages):
                telugu_content_count += 1
        
        telugu_bonus = (telugu_content_count / total_content) * 0.5 if total_content > 0 else 0
        
        return min(base_score + telugu_bonus, 1.0)
    
    def _analyze_emotional_preferences(self, content_list: List[Any]) -> Dict[str, float]:
        """Analyze emotional tone preferences"""
        emotional_indicators = {
            'uplifting': ['inspire', 'hope', 'triumph', 'overcome', 'success', 'achieve'],
            'dark': ['dark', 'grim', 'tragic', 'death', 'violence', 'brutal'],
            'romantic': ['love', 'romance', 'romantic', 'passion', 'relationship'],
            'humorous': ['comedy', 'funny', 'humor', 'laugh', 'entertaining'],
            'intense': ['intense', 'thriller', 'suspense', 'gripping', 'tension'],
            'emotional': ['emotional', 'touching', 'heartfelt', 'moving', 'tearjerker']
        }
        
        emotion_scores = defaultdict(float)
        
        for content in content_list:
            text = f"{content.title} {content.overview or ''}".lower()
            
            for emotion, indicators in emotional_indicators.items():
                for indicator in indicators:
                    if indicator in text:
                        emotion_scores[emotion] += 1.0
        
        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}
        
        return dict(emotion_scores)
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return {}
        
        max_score = max(scores.values())
        if max_score == 0:
            return scores
        
        return {k: v/max_score for k, v in scores.items()}
    
    def _get_default_dna_profile(self) -> Dict[str, Any]:
        """Get default DNA profile for new users"""
        return {
            'dominant_themes': {},
            'preferred_styles': {},
            'director_affinity': {},
            'cultural_preferences': {},
            'narrative_complexity': 0.5,
            'scale_preference': 'medium',
            'quality_threshold': 7.0,
            'emotional_tone_preference': {},
            'cinematic_sophistication': 0.5,
            'commercial_vs_art': 0.6,  # Slightly commercial by default
            'cultural_identity_score': 0.8  # Assume preference for local content
        }

class UserProfileAnalyzer:
    """
    Comprehensive user profile analysis with multi-dimensional behavioral modeling
    """
    
    def __init__(self, db, models, cache_manager=None):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.Review = models.get('Review')
        
        self.cinematic_dna = CinematicDNAAnalyzer()
        self.cache_manager = cache_manager
        self.scaler = StandardScaler()
        
        # User segmentation clusters
        self.user_clusters = None
        self.cluster_model = KMeans(n_clusters=5, random_state=42)
    
    def build_comprehensive_user_profile(self, user_id: int, 
                                       force_refresh: bool = False) -> Dict[str, Any]:
        """Build comprehensive user profile with caching"""
        
        # Check cache first
        if self.cache_manager and not force_refresh:
            cache_key = self.cache_manager.get_user_profile_cache_key(user_id)
            cached_profile = self.cache_manager.get(cache_key)
            if cached_profile:
                logger.info(f"Retrieved cached profile for user {user_id}")
                return cached_profile
        
        try:
            user = self.User.query.get(user_id)
            if not user:
                return {}
            
            # Get all user interactions
            interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not interactions:
                return self._build_cold_start_profile(user)
            
            # Get content for important interactions
            important_interactions = [i for i in interactions 
                                    if i.interaction_type in ['favorite', 'watchlist', 'rating']]
            
            content_ids = [i.content_id for i in important_interactions]
            important_content = self.Content.query.filter(
                self.Content.id.in_(content_ids)
            ).all() if content_ids else []
            
            # Build comprehensive profile
            profile = {
                'user_id': user_id,
                'username': user.username,
                'registration_date': user.created_at,
                'last_active': user.last_active,
                
                # Core preference analysis
                'explicit_preferences': self._analyze_explicit_preferences(user),
                'implicit_preferences': self._analyze_implicit_preferences(interactions),
                'cinematic_dna': self.cinematic_dna.analyze_cinematic_dna(
                    important_content, 
                    self._get_user_languages(user)
                ),
                
                # Behavioral analysis
                'viewing_patterns': self._analyze_viewing_patterns(interactions),
                'rating_patterns': self._analyze_rating_patterns(interactions),
                'search_patterns': self._analyze_search_patterns(interactions),
                'temporal_patterns': self._analyze_temporal_patterns(interactions),
                
                # Content preferences
                'genre_preferences': self._analyze_genre_preferences(interactions, important_content),
                'language_preferences': self._analyze_language_preferences(interactions, important_content),
                'content_type_preferences': self._analyze_content_type_preferences(interactions, important_content),
                'quality_preferences': self._analyze_quality_preferences(interactions, important_content),
                
                # Advanced metrics
                'engagement_metrics': self._calculate_engagement_metrics(interactions),
                'diversity_metrics': self._calculate_diversity_metrics(interactions, important_content),
                'loyalty_metrics': self._calculate_loyalty_metrics(interactions),
                'exploration_metrics': self._calculate_exploration_metrics(interactions, important_content),
                
                # Predictive features
                'next_content_prediction': self._predict_next_content_type(interactions),
                'recommendation_receptivity': self._calculate_recommendation_receptivity(interactions),
                'churn_risk': self._calculate_churn_risk(interactions),
                
                # Profile metadata
                'profile_completeness': 0.0,
                'confidence_score': 0.0,
                'user_segment': 'unknown',
                'last_updated': datetime.utcnow()
            }
            
            # Calculate derived metrics
            profile['profile_completeness'] = self._calculate_profile_completeness(profile)
            profile['confidence_score'] = self._calculate_confidence_score(profile, interactions)
            profile['user_segment'] = self._determine_user_segment(profile)
            
            # Cache the profile
            if self.cache_manager:
                cache_key = self.cache_manager.get_user_profile_cache_key(user_id)
                self.cache_manager.set(cache_key, profile, ttl=3600)  # Cache for 1 hour
            
            logger.info(f"Built comprehensive profile for user {user_id} with {len(interactions)} interactions")
            return profile
            
        except Exception as e:
            logger.error(f"Error building user profile for user {user_id}: {e}")
            return {}
    
    def _get_user_languages(self, user: Any) -> List[str]:
        """Get user's preferred languages"""
        explicit_langs = safe_json_loads(user.preferred_languages, [])
        if explicit_langs:
            return explicit_langs
        
        # Default to Telugu-first if no explicit preference
        return ['Telugu', 'English']
    
    def _build_cold_start_profile(self, user: Any) -> Dict[str, Any]:
        """Build basic profile for new users"""
        return {
            'user_id': user.id,
            'username': user.username,
            'registration_date': user.created_at,
            'explicit_preferences': self._analyze_explicit_preferences(user),
            'cold_start': True,
            'profile_completeness': 0.1,
            'confidence_score': 0.0,
            'user_segment': 'new_user',
            'cinematic_dna': self.cinematic_dna._get_default_dna_profile(),
            'language_preferences': {
                'preferred_languages': self._get_user_languages(user),
                'telugu_preference': True
            },
            'last_updated': datetime.utcnow()
        }
    
    # Continue with other analysis methods...
    # [The rest of the methods would follow the same pattern as the original code
    # but with enhanced analytics and proper modular structure]
    
    def _analyze_genre_preferences(self, interactions: List[Any], 
                                 content_list: List[Any]) -> Dict[str, Any]:
        """Enhanced genre preference analysis with cultural weighting"""
        content_map = {c.id: c for c in content_list}
        
        # Weighted scoring based on interaction type and recency
        interaction_weights = {
            'favorite': 3.0,
            'watchlist': 2.0,
            'rating': 2.5,
            'view': 1.5,
            'search': 1.0
        }
        
        genre_scores = defaultdict(float)
        genre_contexts = defaultdict(list)
        
        for interaction in interactions:
            content = content_map.get(interaction.content_id)
            if not content or not content.genres:
                continue
            
            weight = interaction_weights.get(interaction.interaction_type, 1.0)
            
            # Recency weighting
            days_ago = (datetime.utcnow() - interaction.timestamp).days
            recency_weight = math.exp(-days_ago / 30)  # Exponential decay over 30 days
            
            # Rating weighting
            rating_weight = 1.0
            if interaction.rating:
                rating_weight = (interaction.rating / 10) * 2  # Higher rated content gets more weight
            
            # Language cultural bonus
            content_languages = safe_json_loads(content.languages, [])
            cultural_bonus = 1.0
            if any('telugu' in lang.lower() for lang in content_languages):
                cultural_bonus = 1.3  # Telugu content gets priority
            elif any('hindi' in lang.lower() for lang in content_languages):
                cultural_bonus = 1.2
            
            final_weight = weight * recency_weight * rating_weight * cultural_bonus
            
            try:
                genres = safe_json_loads(content.genres, [])
                for genre in genres:
                    genre_scores[genre] += final_weight
                    genre_contexts[genre].append({
                        'content_title': content.title,
                        'interaction_type': interaction.interaction_type,
                        'rating': interaction.rating,
                        'timestamp': interaction.timestamp
                    })
            except:
                continue
        
        # Normalize scores
        if genre_scores:
            max_score = max(genre_scores.values())
            normalized_scores = {genre: score/max_score for genre, score in genre_scores.items()}
        else:
            normalized_scores = {}
        
        # Calculate genre diversity and evolution
        genre_evolution = self._analyze_genre_evolution(interactions, content_map)
        
        return {
            'genre_scores': dict(normalized_scores),
            'top_genres': [genre for genre, _ in Counter(normalized_scores).most_common(5)],
            'genre_diversity': len(normalized_scores),
            'genre_evolution': genre_evolution,
            'dominant_genre': max(normalized_scores.items(), key=lambda x: x[1])[0] if normalized_scores else None,
            'emerging_interests': self._identify_emerging_genre_interests(genre_contexts),
            'cultural_genre_preference': self._analyze_cultural_genre_preferences(genre_scores, content_map)
        }
    
    def _analyze_genre_evolution(self, interactions: List[Any], 
                               content_map: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze how user's genre preferences evolve over time"""
        if len(interactions) < 10:
            return {'insufficient_data': True}
        
        # Split interactions into time periods
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        mid_point = len(sorted_interactions) // 2
        
        early_interactions = sorted_interactions[:mid_point]
        recent_interactions = sorted_interactions[mid_point:]
        
        early_genres = defaultdict(int)
        recent_genres = defaultdict(int)
        
        for interaction in early_interactions:
            content = content_map.get(interaction.content_id)
            if content and content.genres:
                genres = safe_json_loads(content.genres, [])
                for genre in genres:
                    early_genres[genre] += 1
        
        for interaction in recent_interactions:
            content = content_map.get(interaction.content_id)
            if content and content.genres:
                genres = safe_json_loads(content.genres, [])
                for genre in genres:
                    recent_genres[genre] += 1
        
        # Calculate shifts
        all_genres = set(early_genres.keys()) | set(recent_genres.keys())
        genre_shifts = {}
        
        for genre in all_genres:
            early_count = early_genres.get(genre, 0)
            recent_count = recent_genres.get(genre, 0)
            
            early_ratio = early_count / len(early_interactions) if early_interactions else 0
            recent_ratio = recent_count / len(recent_interactions) if recent_interactions else 0
            
            shift = recent_ratio - early_ratio
            genre_shifts[genre] = shift
        
        # Identify trends
        growing_interests = {g: s for g, s in genre_shifts.items() if s > 0.1}
        declining_interests = {g: s for g, s in genre_shifts.items() if s < -0.1}
        
        return {
            'growing_interests': growing_interests,
            'declining_interests': declining_interests,
            'stability_score': 1 - np.std(list(genre_shifts.values())) if genre_shifts else 0,
            'exploration_trend': 'increasing' if len(recent_genres) > len(early_genres) else 'decreasing'
        }
    
    # [Additional methods would continue in the same pattern...]
    
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Generate user embedding vector for similarity computations"""
        profile = self.build_comprehensive_user_profile(user_id)
        
        if not profile or profile.get('cold_start'):
            return np.zeros(50)  # Default embedding size
        
        # Extract numerical features for embedding
        features = []
        
        # Genre preferences (top 10 genres)
        genre_prefs = profile.get('genre_preferences', {}).get('genre_scores', {})
        top_genres = ['Action', 'Drama', 'Comedy', 'Romance', 'Thriller', 
                     'Adventure', 'Crime', 'Fantasy', 'Horror', 'Mystery']
        
        for genre in top_genres:
            features.append(genre_prefs.get(genre, 0.0))
        
        # Language preferences
        lang_prefs = profile.get('language_preferences', {}).get('language_scores', {})
        for lang in PRIORITY_LANGUAGES:
            features.append(lang_prefs.get(lang.title(), 0.0))
        
        # Cinematic DNA features
        dna = profile.get('cinematic_dna', {})
        features.extend([
            dna.get('cinematic_sophistication', 0.5),
            dna.get('commercial_vs_art', 0.5),
            dna.get('narrative_complexity', 0.5),
            dna.get('cultural_identity_score', 0.5)
        ])
        
        # Behavioral features
        engagement = profile.get('engagement_metrics', {})
        features.extend([
            engagement.get('total_interactions', 0) / 100.0,  # Normalize
            engagement.get('recent_activity_ratio', 0),
            engagement.get('consistency_score', 0),
            profile.get('confidence_score', 0)
        ])
        
        # Content type preferences
        content_prefs = profile.get('content_type_preferences', {}).get('type_scores', {})
        features.extend([
            content_prefs.get('movie', 0),
            content_prefs.get('tv', 0),
            content_prefs.get('anime', 0)
        ])
        
        # Quality and rating patterns
        quality = profile.get('quality_preferences', {})
        features.extend([
            quality.get('quality_threshold', 7.0) / 10.0,
            quality.get('rating_strictness', 0.5),
            quality.get('popular_vs_niche', 0.5)
        ])
        
        # Pad or trim to exactly 50 features
        while len(features) < 50:
            features.append(0.0)
        features = features[:50]
        
        return np.array(features, dtype=np.float32)