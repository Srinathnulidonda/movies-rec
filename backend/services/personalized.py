#backend/services/personalized.py
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

class CinematicDNAAnalyzer:
    """
    Sophisticated Cinematic DNA Analysis Engine for CineBrain
    Analyzes deep patterns, themes, and cinematic styles beyond simple genre matching
    """
    
    def __init__(self):
        # Cinematic themes mapping
        self.cinematic_themes = {
            'justice_revenge': {
                'keywords': ['justice', 'revenge', 'vengeance', 'betrayal', 'redemption', 'vigilante', 'corrupt', 'law'],
                'weight': 1.0
            },
            'heist_crime': {
                'keywords': ['heist', 'robbery', 'theft', 'criminal', 'gang', 'mafia', 'organized crime', 'con'],
                'weight': 0.9
            },
            'underdog_triumph': {
                'keywords': ['underdog', 'overcome', 'triumph', 'struggle', 'dreams', 'poverty', 'rise', 'success'],
                'weight': 0.8
            },
            'epic_mythology': {
                'keywords': ['epic', 'legend', 'mythology', 'ancient', 'gods', 'prophecy', 'destiny', 'kingdom'],
                'weight': 0.9
            },
            'complex_morality': {
                'keywords': ['moral', 'ethical', 'choice', 'consequence', 'gray area', 'complex', 'dilemma'],
                'weight': 0.85
            },
            'family_bonds': {
                'keywords': ['family', 'father', 'mother', 'brother', 'sister', 'legacy', 'generation', 'honor'],
                'weight': 0.7
            },
            'survival_thriller': {
                'keywords': ['survival', 'danger', 'trapped', 'escape', 'life or death', 'rescue', 'catastrophe'],
                'weight': 0.8
            },
            'love_sacrifice': {
                'keywords': ['love', 'sacrifice', 'devotion', 'loss', 'heartbreak', 'romantic', 'passion'],
                'weight': 0.6
            }
        }
        
        # Cinematic styles mapping
        self.cinematic_styles = {
            'grand_scale_epic': {
                'indicators': ['epic', 'grand', 'spectacular', 'massive', 'scale', 'cinematic', 'visual'],
                'runtime_min': 120,
                'budget_indicator': 'high',
                'weight': 1.0
            },
            'character_driven_drama': {
                'indicators': ['character', 'intimate', 'personal', 'emotional', 'psychological', 'study'],
                'genres': ['Drama', 'Biography'],
                'weight': 0.9
            },
            'non_linear_narrative': {
                'indicators': ['complex', 'puzzle', 'twist', 'non-linear', 'timeline', 'mystery', 'revelation'],
                'directors': ['Christopher Nolan', 'Denis Villeneuve', 'Rian Johnson'],
                'weight': 0.95
            },
            'high_octane_action': {
                'indicators': ['action', 'intense', 'adrenaline', 'explosive', 'chase', 'fight', 'combat'],
                'genres': ['Action', 'Thriller'],
                'weight': 0.8
            },
            'critical_acclaim': {
                'rating_threshold': 8.0,
                'vote_threshold': 1000,
                'weight': 0.7
            },
            'visual_masterpiece': {
                'indicators': ['visual', 'cinematography', 'stunning', 'beautiful', 'artistic', 'masterpiece'],
                'weight': 0.8
            }
        }
        
        # Director signatures (CineBrain's curated list)
        self.director_signatures = {
            'Christopher Nolan': {
                'style': ['complex_narrative', 'time_manipulation', 'cerebral', 'practical_effects'],
                'themes': ['reality_perception', 'memory', 'sacrifice', 'obsession']
            },
            'Denis Villeneuve': {
                'style': ['atmospheric', 'philosophical', 'visual_poetry', 'slow_burn'],
                'themes': ['humanity', 'communication', 'existence', 'future']
            },
            'S.S. Rajamouli': {
                'style': ['epic_scale', 'visual_spectacle', 'emotional_core', 'cultural_pride'],
                'themes': ['heroism', 'tradition', 'loyalty', 'justice']
            },
            'Quentin Tarantino': {
                'style': ['non_linear', 'dialogue_heavy', 'violence_stylized', 'pop_culture'],
                'themes': ['revenge', 'redemption', 'justice', 'morality']
            }
        }
    
    def analyze_cinematic_dna(self, content_list: List[Any]) -> Dict[str, Any]:
        """
        Analyze the cinematic DNA of user's favorite/watchlist content
        Returns deep insights about user's taste preferences
        """
        if not content_list:
            return {}
        
        dna_profile = {
            'dominant_themes': {},
            'preferred_styles': {},
            'director_affinity': {},
            'narrative_complexity': 0.0,
            'scale_preference': 'medium',
            'quality_threshold': 7.0,
            'cultural_preferences': {},
            'emotional_tone': {},
            'cinematic_sophistication': 0.0
        }
        
        theme_scores = defaultdict(float)
        style_scores = defaultdict(float)
        director_scores = defaultdict(float)
        
        for content in content_list:
            # Analyze themes from overview and title
            content_text = f"{content.title} {content.overview or ''}"
            themes = self._extract_themes(content_text)
            
            for theme, score in themes.items():
                theme_scores[theme] += score
            
            # Analyze cinematic style
            styles = self._analyze_cinematic_style(content)
            for style, score in styles.items():
                style_scores[style] += score
            
            # Director analysis (if available through cast/crew)
            # This would be enhanced when cast/crew data is available
            
            # Quality assessment
            if content.rating:
                dna_profile['quality_threshold'] = max(
                    dna_profile['quality_threshold'],
                    content.rating * 0.8  # Weighted average
                )
        
        # Normalize and rank
        if theme_scores:
            max_theme_score = max(theme_scores.values())
            dna_profile['dominant_themes'] = {
                theme: score/max_theme_score 
                for theme, score in sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        if style_scores:
            max_style_score = max(style_scores.values())
            dna_profile['preferred_styles'] = {
                style: score/max_style_score 
                for style, score in sorted(style_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        # Calculate sophistication score
        dna_profile['cinematic_sophistication'] = self._calculate_sophistication(content_list)
        
        return dna_profile
    
    def _extract_themes(self, text: str) -> Dict[str, float]:
        """Extract cinematic themes from content text"""
        text_lower = text.lower()
        themes = {}
        
        for theme_name, theme_data in self.cinematic_themes.items():
            score = 0.0
            keywords = theme_data['keywords']
            weight = theme_data['weight']
            
            for keyword in keywords:
                if keyword in text_lower:
                    score += weight
            
            if score > 0:
                themes[theme_name] = score / len(keywords)  # Normalize by keyword count
        
        return themes
    
    def _analyze_cinematic_style(self, content: Any) -> Dict[str, float]:
        """Analyze cinematic style of content"""
        styles = {}
        content_text = f"{content.title} {content.overview or ''}".lower()
        
        for style_name, style_data in self.cinematic_styles.items():
            score = 0.0
            
            # Check for style indicators in text
            if 'indicators' in style_data:
                for indicator in style_data['indicators']:
                    if indicator in content_text:
                        score += 0.3
            
            # Check runtime criteria
            if 'runtime_min' in style_data and content.runtime:
                if content.runtime >= style_data['runtime_min']:
                    score += 0.4
            
            # Check genre alignment
            if 'genres' in style_data and content.genres:
                try:
                    content_genres = json.loads(content.genres)
                    for genre in style_data['genres']:
                        if genre in content_genres:
                            score += 0.3
                except:
                    pass
            
            # Check critical acclaim
            if style_name == 'critical_acclaim':
                if content.rating and content.vote_count:
                    if (content.rating >= style_data['rating_threshold'] and 
                        content.vote_count >= style_data['vote_threshold']):
                        score = 1.0
            
            if score > 0:
                styles[style_name] = min(score, 1.0) * style_data['weight']
        
        return styles
    
    def _calculate_sophistication(self, content_list: List[Any]) -> float:
        """Calculate user's cinematic sophistication level"""
        if not content_list:
            return 0.5
        
        sophistication_factors = []
        
        # Average rating preference
        ratings = [c.rating for c in content_list if c.rating]
        if ratings:
            avg_rating = np.mean(ratings)
            sophistication_factors.append(min(avg_rating / 10, 1.0))
        
        # Genre diversity
        all_genres = []
        for content in content_list:
            if content.genres:
                try:
                    all_genres.extend(json.loads(content.genres))
                except:
                    pass
        
        if all_genres:
            genre_diversity = len(set(all_genres)) / len(all_genres)
            sophistication_factors.append(genre_diversity)
        
        # Content type diversity
        content_types = [c.content_type for c in content_list]
        type_diversity = len(set(content_types)) / len(content_types) if content_types else 0
        sophistication_factors.append(type_diversity)
        
        return np.mean(sophistication_factors) if sophistication_factors else 0.5
    
    def find_cinematic_matches(self, dna_profile: Dict[str, Any], content_pool: List[Any], 
                             limit: int = 20) -> List[Tuple[Any, float, str]]:
        """
        Find content that matches the user's cinematic DNA
        Returns list of (content, match_score, reason)
        """
        matches = []
        
        for content in content_pool:
            match_score = 0.0
            reasons = []
            
            # Theme matching
            content_themes = self._extract_themes(f"{content.title} {content.overview or ''}")
            for theme, user_score in dna_profile.get('dominant_themes', {}).items():
                if theme in content_themes:
                    theme_match = content_themes[theme] * user_score
                    match_score += theme_match * 0.4
                    if theme_match > 0.3:
                        reasons.append(f"shares {theme.replace('_', ' ')} themes")
            
            # Style matching
            content_styles = self._analyze_cinematic_style(content)
            for style, user_score in dna_profile.get('preferred_styles', {}).items():
                if style in content_styles:
                    style_match = content_styles[style] * user_score
                    match_score += style_match * 0.3
                    if style_match > 0.3:
                        reasons.append(f"matches {style.replace('_', ' ')} style")
            
            # Quality threshold
            if content.rating and content.rating >= dna_profile.get('quality_threshold', 7.0):
                match_score += 0.2
                reasons.append("meets quality standards")
            
            # Sophistication matching
            content_sophistication = self._calculate_sophistication([content])
            user_sophistication = dna_profile.get('cinematic_sophistication', 0.5)
            if abs(content_sophistication - user_sophistication) < 0.3:
                match_score += 0.1
            
            if match_score > 0.3 and reasons:
                reason_text = f"CineBrain recommends because it {', '.join(reasons[:2])}"
                matches.append((content, match_score, reason_text))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]

class UserProfileAnalyzer:
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.Review = models['Review']
        self.cinematic_dna = CinematicDNAAnalyzer()
    
    def build_comprehensive_user_profile(self, user_id: int) -> Dict[str, Any]:
        try:
            user = self.User.query.get(user_id)
            if not user:
                return {}
            
            interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            
            # Get user's favorites and watchlist for cinematic DNA analysis
            favorite_interactions = [i for i in interactions if i.interaction_type == 'favorite']
            watchlist_interactions = [i for i in interactions if i.interaction_type == 'watchlist']
            
            # Get actual content for DNA analysis
            all_important_content_ids = [i.content_id for i in favorite_interactions + watchlist_interactions]
            important_content = self.Content.query.filter(
                self.Content.id.in_(all_important_content_ids)
            ).all() if all_important_content_ids else []
            
            # Analyze cinematic DNA
            cinematic_dna_profile = self.cinematic_dna.analyze_cinematic_dna(important_content)
            
            profile = {
                'user_id': user_id,
                'username': user.username,
                'registration_date': user.created_at,
                'last_active': user.last_active,
                'explicit_preferences': self._analyze_explicit_preferences(user),
                'implicit_preferences': self._analyze_implicit_preferences(interactions),
                'cinematic_dna': cinematic_dna_profile,  # NEW: Deep cinematic analysis
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

class CineBrainRecommendationEngine:
    """
    CineBrain's Advanced Personalized Content Recommendation Engine
    Focuses on deep cinematic DNA analysis and strict language prioritization
    """
    
    def __init__(self, db, models, cache=None):
        self.db = db
        self.models = models
        self.cache = cache
        
        self.user_profiler = UserProfileAnalyzer(db, models)
        self.content_based = ContentBasedFiltering()
        self.collaborative = CollaborativeFiltering()
        self.hybrid = HybridRecommendationEngine()
        self.similarity_engine = UltraPowerfulSimilarityEngine()
        self.cinematic_dna = CinematicDNAAnalyzer()
        
        # Feedback learning system
        self.feedback_log = []

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
        """
        CineBrain's Advanced Personalized Recommendation System
        Analyzes cinematic DNA and applies strict language prioritization
        """
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
                    'cinebrain_for_you',
                    'because_you_watched',
                    'trending_for_you',
                    'new_releases_for_you',
                    'your_language_priority',
                    'cinematic_dna_matches',
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
                'profile_insights': self._generate_cinebrain_insights(user_profile),
                'recommendation_metadata': {
                    'platform': 'cinebrain',
                    'engine_version': 'advanced_cinematic_dna_v2.0',
                    'profile_completeness': user_profile.get('profile_completeness', 0),
                    'confidence_score': user_profile.get('confidence_score', 0),
                    'cinematic_sophistication': user_profile.get('cinematic_dna', {}).get('cinematic_sophistication', 0.5),
                    'language_priority_applied': True,
                    'algorithms_used': ['cinematic_dna', 'language_priority', 'hybrid_collaborative'],
                    'total_categories': len(all_recommendations),
                    'successful_categories': successful_categories,
                    'generated_at': datetime.utcnow().isoformat(),
                    'cache_duration': 3600,
                    'next_update': (datetime.utcnow() + timedelta(hours=1)).isoformat()
                }
            }
            
            if self.cache:
                try:
                    cache_key = f"cinebrain_recs:{user_id}:{hash(str(categories))}"
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
            'cinebrain_for_you': self._generate_cinebrain_for_you,
            'because_you_watched': self._generate_because_you_watched_with_reasons,
            'trending_for_you': self._generate_trending_for_you,
            'new_releases_for_you': self._generate_new_releases_for_you,
            'your_language_priority': self._generate_language_priority_recommendations,
            'cinematic_dna_matches': self._generate_cinematic_dna_matches,
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
    
    def _generate_cinebrain_for_you(self, user_profile: Dict[str, Any], 
                                   limit: int) -> List[Dict[str, Any]]:
        """
        CineBrain's main 'For You' feed with strict language prioritization
        and cinematic DNA analysis
        """
        user_id = user_profile['user_id']
        
        # Get user's interacted content
        interacted_content_ids = self._get_user_interacted_content(user_id)
        
        # Get content pool with language priority filtering
        explicit_languages = user_profile.get('explicit_preferences', {}).get('preferred_languages', [])
        implicit_languages = user_profile.get('language_preferences', {}).get('preferred_languages', [])
        
        # Combine and prioritize languages
        all_languages = explicit_languages + implicit_languages
        priority_languages = list(dict.fromkeys(all_languages))  # Remove duplicates while preserving order
        
        if not priority_languages:
            priority_languages = ['Telugu', 'English']  # CineBrain default
        
        # Get content pool with strict language prioritization
        language_grouped_content = self._get_language_grouped_content_pool(
            exclude_ids=interacted_content_ids,
            priority_languages=priority_languages
        )
        
        recommendations = []
        cinematic_dna = user_profile.get('cinematic_dna', {})
        
        # Primary language group (highest priority)
        primary_lang = priority_languages[0] if priority_languages else 'Telugu'
        primary_content = language_grouped_content.get(primary_lang, [])
        
        if primary_content:
            # Apply cinematic DNA matching to primary language content
            dna_matches = self.cinematic_dna.find_cinematic_matches(
                cinematic_dna, primary_content, limit=int(limit * 0.6)
            )
            
            for content, score, reason in dna_matches:
                recommendations.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'poster_path': self._format_poster_path(content.poster_path),
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'cinebrain_score': round(score, 3),
                    'recommendation_reason': reason,
                    'language_priority': 1,
                    'primary_language': primary_lang,
                    'source': 'cinematic_dna_primary_language',
                    'youtube_trailer_id': content.youtube_trailer_id
                })
        
        # Secondary languages (fill remaining slots)
        remaining_slots = limit - len(recommendations)
        if remaining_slots > 0 and len(priority_languages) > 1:
            for i, lang in enumerate(priority_languages[1:], 2):
                lang_content = language_grouped_content.get(lang, [])
                if not lang_content:
                    continue
                
                lang_limit = min(remaining_slots // (len(priority_languages) - 1), 10)
                lang_matches = self.cinematic_dna.find_cinematic_matches(
                    cinematic_dna, lang_content, limit=lang_limit
                )
                
                for content, score, reason in lang_matches:
                    recommendations.append({
                        'id': content.id,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'languages': json.loads(content.languages or '[]'),
                        'rating': content.rating,
                        'poster_path': self._format_poster_path(content.poster_path),
                        'overview': content.overview[:150] + '...' if content.overview else '',
                        'cinebrain_score': round(score * 0.9, 3),  # Slight penalty for secondary languages
                        'recommendation_reason': reason,
                        'language_priority': i,
                        'primary_language': lang,
                        'source': f'cinematic_dna_secondary_language_{i}',
                        'youtube_trailer_id': content.youtube_trailer_id
                    })
                    
                    if len(recommendations) >= limit:
                        break
                
                if len(recommendations) >= limit:
                    break
        
        return recommendations[:limit]
    
    def _generate_because_you_watched_with_reasons(self, user_profile: Dict[str, Any], 
                                                  limit: int) -> List[Dict[str, Any]]:
        """
        Generate 'Because you watched X' recommendations with detailed reasoning
        """
        user_id = user_profile['user_id']
        
        try:
            # Get user's favorites and recent watchlist items
            recent_interactions = self.models['UserInteraction'].query.filter(
                and_(
                    self.models['UserInteraction'].user_id == user_id,
                    self.models['UserInteraction'].interaction_type.in_(['favorite', 'watchlist', 'view']),
                    self.models['UserInteraction'].timestamp > datetime.utcnow() - timedelta(days=30)
                )
            ).order_by(desc(self.models['UserInteraction'].timestamp)).limit(10).all()
            
            if not recent_interactions:
                return []
            
            recommendations = []
            content_pool = self._get_base_content_pool()
            
            # Get user's explicit language preferences for prioritization
            explicit_languages = user_profile.get('explicit_preferences', {}).get('preferred_languages', [])
            
            for interaction in recent_interactions[:3]:  # Focus on top 3 recent items
                try:
                    base_content = self.models['Content'].query.get(interaction.content_id)
                    if not base_content:
                        continue
                    
                    # Find similar content using cinematic DNA
                    base_dna = self.cinematic_dna.analyze_cinematic_dna([base_content])
                    similar_content = self.cinematic_dna.find_cinematic_matches(
                        base_dna, content_pool, limit=15
                    )
                    
                    for similar_item, similarity_score, dna_reason in similar_content:
                        content = similar_item
                        
                        # Generate detailed reason
                        detailed_reason = self._generate_detailed_recommendation_reason(
                            base_content, content, user_profile
                        )
                        
                        # Language priority bonus
                        content_languages = json.loads(content.languages or '[]')
                        language_bonus = 0.0
                        for lang in explicit_languages:
                            if any(lang.lower() in cl.lower() for cl in content_languages):
                                language_bonus = 0.2
                                break
                        
                        final_score = similarity_score + language_bonus
                        
                        recommendations.append({
                            'id': content.id,
                            'title': content.title,
                            'content_type': content.content_type,
                            'genres': json.loads(content.genres or '[]'),
                            'languages': content_languages,
                            'rating': content.rating,
                            'poster_path': self._format_poster_path(content.poster_path),
                            'overview': content.overview[:150] + '...' if content.overview else '',
                            'cinebrain_score': round(final_score, 3),
                            'recommendation_reason': detailed_reason,
                            'because_of': {
                                'title': base_content.title,
                                'content_id': base_content.id,
                                'interaction_type': interaction.interaction_type
                            },
                            'source': 'cinematic_similarity_analysis',
                            'confidence': 'high' if similarity_score > 0.7 else 'medium',
                            'youtube_trailer_id': content.youtube_trailer_id
                        })
                
                except Exception as e:
                    logger.error(f"Error processing interaction {interaction.id}: {e}")
                    continue
            
            # Remove duplicates and sort by score
            unique_recs = self._remove_duplicates_and_rerank(recommendations)
            return unique_recs[:limit]
            
        except Exception as e:
            logger.error(f"Error in _generate_because_you_watched_with_reasons: {e}")
            return []
    
    def _generate_language_priority_recommendations(self, user_profile: Dict[str, Any], 
                                                   limit: int) -> List[Dict[str, Any]]:
        """
        Generate recommendations with strict language priority
        """
        explicit_languages = user_profile.get('explicit_preferences', {}).get('preferred_languages', [])
        implicit_languages = user_profile.get('language_preferences', {}).get('preferred_languages', [])
        
        # Combine with explicit taking precedence
        priority_languages = explicit_languages + [l for l in implicit_languages if l not in explicit_languages]
        
        if not priority_languages:
            priority_languages = ['Telugu', 'English', 'Hindi']
        
        recommendations = []
        
        for i, language in enumerate(priority_languages):
            # Get content in this language
            lang_content = self.models['Content'].query.filter(
                self.models['Content'].languages.contains(f'"{language}"')
            ).order_by(desc(self.models['Content'].rating)).limit(20).all()
            
            lang_limit = min(limit // len(priority_languages), 15)
            
            for content in lang_content[:lang_limit]:
                personalization_score = self._calculate_personalization_score(content, user_profile)
                language_priority_bonus = 1.0 - (i * 0.1)  # Decreasing bonus for lower priority
                
                final_score = personalization_score * language_priority_bonus
                
                recommendations.append({
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'poster_path': self._format_poster_path(content.poster_path),
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'cinebrain_score': round(final_score, 3),
                    'recommendation_reason': f"Top-rated {language} content matching your taste profile",
                    'language_priority': i + 1,
                    'primary_language': language,
                    'source': 'strict_language_priority',
                    'youtube_trailer_id': content.youtube_trailer_id
                })
        
        recommendations.sort(key=lambda x: x['cinebrain_score'], reverse=True)
        return recommendations[:limit]
    
    def _generate_cinematic_dna_matches(self, user_profile: Dict[str, Any], 
                                       limit: int) -> List[Dict[str, Any]]:
        """
        Generate recommendations based purely on cinematic DNA analysis
        """
        user_id = user_profile['user_id']
        cinematic_dna = user_profile.get('cinematic_dna', {})
        
        if not cinematic_dna or not cinematic_dna.get('dominant_themes'):
            return []
        
        # Get content pool
        interacted_content_ids = self._get_user_interacted_content(user_id)
        content_pool = self._get_filtered_content_pool(exclude_ids=interacted_content_ids)
        
        # Find matches using cinematic DNA
        dna_matches = self.cinematic_dna.find_cinematic_matches(
            cinematic_dna, content_pool, limit=limit
        )
        
        recommendations = []
        for content, score, reason in dna_matches:
            recommendations.append({
                'id': content.id,
                'title': content.title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'rating': content.rating,
                'poster_path': self._format_poster_path(content.poster_path),
                'overview': content.overview[:150] + '...' if content.overview else '',
                'cinebrain_score': round(score, 3),
                'recommendation_reason': reason,
                'cinematic_match_type': 'dna_analysis',
                'source': 'pure_cinematic_dna',
                'confidence': 'very_high' if score > 0.8 else 'high',
                'youtube_trailer_id': content.youtube_trailer_id
            })
        
        return recommendations
    
    def _generate_detailed_recommendation_reason(self, base_content: Any, 
                                               recommended_content: Any, 
                                               user_profile: Dict[str, Any]) -> str:
        """
        Generate detailed, specific reasons for recommendations
        """
        reasons = []
        
        # Analyze common themes
        base_themes = self.cinematic_dna._extract_themes(f"{base_content.title} {base_content.overview or ''}")
        rec_themes = self.cinematic_dna._extract_themes(f"{recommended_content.title} {recommended_content.overview or ''}")
        
        common_themes = set(base_themes.keys()) & set(rec_themes.keys())
        if common_themes:
            theme_names = [theme.replace('_', ' ') for theme in list(common_themes)[:2]]
            reasons.append(f"shares {' and '.join(theme_names)} themes with {base_content.title}")
        
        # Analyze genres
        try:
            base_genres = set(json.loads(base_content.genres or '[]'))
            rec_genres = set(json.loads(recommended_content.genres or '[]'))
            common_genres = base_genres & rec_genres
            
            if common_genres and not common_themes:  # Only mention if themes weren't already covered
                genre_list = list(common_genres)[:2]
                reasons.append(f"similar {'/'.join(genre_list).lower()} style")
        except:
            pass
        
        # Quality and rating similarity
        if base_content.rating and recommended_content.rating:
            rating_diff = abs(base_content.rating - recommended_content.rating)
            if rating_diff < 1.0:
                reasons.append("maintains similar quality standards")
        
        # Cinematic style analysis
        base_styles = self.cinematic_dna._analyze_cinematic_style(base_content)
        rec_styles = self.cinematic_dna._analyze_cinematic_style(recommended_content)
        
        common_styles = set(base_styles.keys()) & set(rec_styles.keys())
        if common_styles:
            style_names = [style.replace('_', ' ') for style in list(common_styles)[:1]]
            reasons.append(f"employs {style_names[0]} approach")
        
        # Default reason if nothing specific found
        if not reasons:
            reasons.append("appeals to your sophisticated taste profile")
        
        # Construct final reason
        base_reason = f"Because you appreciated {base_content.title}"
        detail_reason = ", ".join(reasons[:2])  # Limit to 2 reasons for readability
        
        return f"{base_reason}, CineBrain suggests this as it {detail_reason}"
    
    def integrate_user_feedback(self, user_id: int, feedback_data: Dict[str, Any]):
        """
        Continuous Learning Loop - Integrate user feedback to improve recommendations
        """
        try:
            feedback_entry = {
                'user_id': user_id,
                'timestamp': datetime.utcnow(),
                'feedback_type': feedback_data.get('type'),  # 'like', 'dislike', 'watch', 'skip'
                'content_id': feedback_data.get('content_id'),
                'recommendation_reason': feedback_data.get('reason'),
                'category': feedback_data.get('category'),
                'user_rating': feedback_data.get('rating'),
                'metadata': feedback_data.get('metadata', {})
            }
            
            self.feedback_log.append(feedback_entry)
            
            # Update user preferences in real-time
            self.update_user_preferences_realtime(user_id, {
                'content_id': feedback_data.get('content_id'),
                'interaction_type': 'feedback',
                'rating': feedback_data.get('rating'),
                'metadata': {
                    'feedback_type': feedback_data.get('type'),
                    'recommendation_category': feedback_data.get('category'),
                    'feedback_context': feedback_data.get('metadata', {})
                }
            })
            
            # Analyze feedback patterns for algorithm improvement
            if len(self.feedback_log) > 100:
                self._analyze_feedback_patterns()
            
            logger.info(f"Integrated feedback for user {user_id}: {feedback_data.get('type')} on content {feedback_data.get('content_id')}")
            
        except Exception as e:
            logger.error(f"Error integrating user feedback: {e}")
    
    def _analyze_feedback_patterns(self):
        """
        Analyze feedback patterns to improve recommendation algorithms
        """
        try:
            recent_feedback = [f for f in self.feedback_log if 
                             f['timestamp'] > datetime.utcnow() - timedelta(days=7)]
            
            if not recent_feedback:
                return
            
            # Analyze category performance
            category_performance = defaultdict(list)
            for feedback in recent_feedback:
                category = feedback.get('category', 'unknown')
                feedback_type = feedback.get('feedback_type')
                
                if feedback_type in ['like', 'watch']:
                    category_performance[category].append(1)
                elif feedback_type in ['dislike', 'skip']:
                    category_performance[category].append(0)
            
            # Log performance insights
            for category, scores in category_performance.items():
                if len(scores) >= 5:  # Minimum feedback for meaningful analysis
                    success_rate = np.mean(scores)
                    logger.info(f"CineBrain category '{category}' success rate: {success_rate:.2%}")
                    
                    if success_rate < 0.4:
                        logger.warning(f"Category '{category}' needs algorithm improvement")
            
        except Exception as e:
            logger.error(f"Error analyzing feedback patterns: {e}")
    
    def _get_language_grouped_content_pool(self, exclude_ids: List[int] = None, 
                                          priority_languages: List[str] = None) -> Dict[str, List[Any]]:
        """
        Get content pool grouped by language with priority ordering
        """
        try:
            query = self.models['Content'].query
            
            if exclude_ids:
                query = query.filter(~self.models['Content'].id.in_(exclude_ids))
            
            # Get all content
            all_content = query.filter(
                and_(
                    self.models['Content'].title.isnot(None),
                    self.models['Content'].content_type.isnot(None),
                    self.models['Content'].rating >= 6.0  # Quality filter
                )
            ).order_by(desc(self.models['Content'].rating)).limit(1000).all()
            
            # Group by language
            language_groups = defaultdict(list)
            
            for content in all_content:
                if not content.languages:
                    continue
                
                try:
                    content_languages = json.loads(content.languages)
                    for language in content_languages:
                        # Normalize language name
                        lang_normalized = language.strip().title()
                        language_groups[lang_normalized].append(content)
                except:
                    continue
            
            # Prioritize based on user preferences
            if priority_languages:
                prioritized_groups = {}
                for lang in priority_languages:
                    lang_normalized = lang.strip().title()
                    if lang_normalized in language_groups:
                        prioritized_groups[lang_normalized] = language_groups[lang_normalized]
                
                # Add remaining languages
                for lang, content_list in language_groups.items():
                    if lang not in prioritized_groups:
                        prioritized_groups[lang] = content_list
                
                return prioritized_groups
            
            return dict(language_groups)
            
        except Exception as e:
            logger.error(f"Error getting language grouped content pool: {e}")
            return {}
    
    def _generate_cinebrain_insights(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate CineBrain-specific user insights
        """
        cinematic_dna = user_profile.get('cinematic_dna', {})
        
        insights = {
            'profile_strength': 'strong' if user_profile.get('confidence_score', 0) > 0.7 else 'building',
            'cinematic_sophistication': cinematic_dna.get('cinematic_sophistication', 0.5),
            'dominant_themes': list(cinematic_dna.get('dominant_themes', {}).keys())[:3],
            'preferred_styles': list(cinematic_dna.get('preferred_styles', {}).keys())[:3],
            'language_priority': user_profile.get('language_preferences', {}).get('preferred_languages', [])[:3],
            'taste_profile': self._classify_taste_profile(user_profile),
            'recommendation_accuracy': min(user_profile.get('confidence_score', 0) * 100, 95),
            'cinebrain_tier': 'cinephile' if cinematic_dna.get('cinematic_sophistication', 0) > 0.7 else 'enthusiast',
            'improvement_tip': self._get_personalized_tip(user_profile)
        }
        
        return insights
    
    def _classify_taste_profile(self, user_profile: Dict[str, Any]) -> str:
        """
        Classify user's taste profile based on their preferences and behavior
        """
        cinematic_dna = user_profile.get('cinematic_dna', {})
        sophistication = cinematic_dna.get('cinematic_sophistication', 0.5)
        themes = cinematic_dna.get('dominant_themes', {})
        
        if sophistication > 0.8:
            return 'cinephile'
        elif 'complex_morality' in themes or 'epic_mythology' in themes:
            return 'narrative_connoisseur'
        elif 'justice_revenge' in themes or 'heist_crime' in themes:
            return 'action_aficionado'
        elif user_profile.get('exploration_tendency', 0) > 0.7:
            return 'genre_explorer'
        else:
            return 'mainstream_enthusiast'
    
    def _get_personalized_tip(self, user_profile: Dict[str, Any]) -> str:
        """
        Get personalized tip for improving recommendation accuracy
        """
        completeness = user_profile.get('profile_completeness', 0)
        
        if completeness < 0.3:
            return "Add more content to your favorites and watchlist for better CineBrain recommendations"
        elif completeness < 0.6:
            return "Rate more content to help CineBrain understand your taste preferences"
        elif completeness < 0.8:
            return "Explore different genres to help CineBrain discover hidden gems for you"
        else:
            return "Your CineBrain profile is optimized for maximum personalization!"
    
    # [Previous methods remain the same - update_user_preferences_realtime, etc.]
    
    def update_user_preferences_realtime(self, user_id: int, interaction_data: Dict[str, Any]):
        try:
            if self.cache:
                cache_keys = [
                    f"cinebrain_recs:{user_id}:*",
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
                'message': 'Start interacting with content to get CineBrain\'s personalized recommendations!'
            },
            'recommendation_metadata': {
                'platform': 'cinebrain',
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
                    'message': 'Showing CineBrain trending content'
                },
                'recommendation_metadata': {
                    'platform': 'cinebrain',
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
    
    # [Include all other existing methods with minimal changes...]
    
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
    
    def _get_user_interacted_content(self, user_id: int) -> List[int]:
        interactions = self.models['UserInteraction'].query.filter_by(user_id=user_id).all()
        return [i.content_id for i in interactions]
    
    def _get_filtered_content_pool(self, exclude_ids: List[int] = None, 
                                  user_profile: Dict[str, Any] = None) -> List[Any]:
        try:
            query = self.models['Content'].query
            
            if exclude_ids:
                query = query.filter(~self.models['Content'].id.in_(exclude_ids))
            
            query = query.filter(
                and_(
                    self.models['Content'].title.isnot(None),
                    self.models['Content'].content_type.isnot(None)
                )
            )
            
            return query.order_by(desc(self.models['Content'].rating)).limit(500).all()
            
        except Exception as e:
            logger.error(f"Error getting filtered content pool: {e}")
            return []
    
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
        
        unique_recs.sort(key=lambda x: x.get('cinebrain_score', x.get('similarity_score', 0)), reverse=True)
        
        return unique_recs
    
    # Additional methods from original implementation would go here...
    def _generate_trending_for_you(self, user_profile: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        # Implementation similar to original but with CineBrain branding
        pass
    
    def _generate_new_releases_for_you(self, user_profile: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        # Implementation similar to original but with CineBrain branding
        pass
    
    def _generate_hidden_gems(self, user_profile: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        # Implementation similar to original but with CineBrain branding
        pass
    
    def _generate_continue_watching(self, user_profile: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        # Implementation similar to original but with CineBrain branding
        pass
    
    def _generate_quick_picks(self, user_profile: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        # Implementation similar to original but with CineBrain branding
        pass
    
    def _generate_critically_acclaimed(self, user_profile: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        # Implementation similar to original but with CineBrain branding
        pass

# Updated initialization function
recommendation_engine = None

def init_personalized(app, db, models, services, cache=None):
    global recommendation_engine
    
    try:
        recommendation_engine = CineBrainRecommendationEngine(db, models, cache)
        logger.info("CineBrain's Advanced Personalized Recommendation System initialized successfully")
        return recommendation_engine
    except Exception as e:
        logger.error(f"Failed to initialize CineBrain recommendation system: {e}")
        return None

def get_recommendation_engine():
    return recommendation_engine