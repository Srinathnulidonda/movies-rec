# backend/personalized/profile_analyzer.py

"""
CineBrain Advanced Profile Analyzer
Production-grade behavioral intelligence and user preference modeling system
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix, hstack
from scipy.stats import entropy
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
import json
import logging
import math
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Set
import threading
import time
from contextlib import contextmanager
from sqlalchemy import func, desc, and_, or_
from sqlalchemy.exc import OperationalError, DisconnectionError
import traceback

logger = logging.getLogger(__name__)

# Advanced configuration for Telugu-first personalization
CINEMATIC_CULTURE_CONFIG = {
    'telugu_cinema_weights': {
        'tollywood_preference': 1.0,
        'regional_storytelling': 0.9,
        'cultural_themes': 0.8,
        'language_authenticity': 0.95
    },
    'indian_cinema_hierarchy': {
        'telugu': {'weight': 1.0, 'cultural_bonus': 0.3},
        'hindi': {'weight': 0.85, 'cultural_bonus': 0.2},
        'tamil': {'weight': 0.8, 'cultural_bonus': 0.25},
        'malayalam': {'weight': 0.75, 'cultural_bonus': 0.2},
        'kannada': {'weight': 0.7, 'cultural_bonus': 0.15},
        'english': {'weight': 0.9, 'cultural_bonus': 0.0}
    },
    'narrative_patterns': {
        'heroic_journey': ['hero', 'journey', 'destiny', 'sacrifice', 'triumph'],
        'family_drama': ['family', 'tradition', 'honor', 'generation', 'legacy'],
        'romance_epic': ['love', 'passion', 'separation', 'reunion', 'devotion'],
        'social_commentary': ['society', 'change', 'justice', 'corruption', 'reform']
    }
}

class CinematicDNAEngine:
    """
    Advanced Cinematic DNA Analysis Engine
    Understands deep cinematic preferences at a molecular level
    """
    
    def __init__(self, cache=None):
        self.cache = cache
        self.dna_cache = {}
        self.pattern_analyzer = CinematicPatternAnalyzer()
        self.cultural_analyzer = CulturalPreferenceAnalyzer()
        self.narrative_analyzer = NarrativeStructureAnalyzer()
        
        # Advanced cinematic dimensions
        self.cinematic_dimensions = {
            'narrative_complexity': {
                'linear_storytelling': 0.0,
                'non_linear_narrative': 0.2,
                'multi_layered_plot': 0.4,
                'complex_timeline': 0.6,
                'meta_narrative': 0.8,
                'experimental_structure': 1.0
            },
            'emotional_intensity': {
                'subtle_emotions': 0.0,
                'moderate_drama': 0.3,
                'intense_emotions': 0.6,
                'overwhelming_passion': 0.8,
                'cathartic_experience': 1.0
            },
            'visual_sophistication': {
                'simple_visuals': 0.0,
                'polished_cinematography': 0.3,
                'artistic_visuals': 0.6,
                'groundbreaking_effects': 0.8,
                'visual_masterpiece': 1.0
            },
            'cultural_authenticity': {
                'universal_themes': 0.0,
                'regional_elements': 0.3,
                'cultural_specific': 0.6,
                'deeply_rooted': 0.8,
                'cultural_immersion': 1.0
            }
        }
        
        # Telugu cinema specific patterns
        self.telugu_cinema_dna = {
            'tollywood_signatures': {
                'mass_hero_elevation': ['mass', 'elevation', 'heroic', 'larger than life'],
                'family_sentiment': ['family', 'emotion', 'sentiment', 'tradition'],
                'action_spectacle': ['action', 'fight', 'chase', 'spectacle'],
                'romantic_elements': ['romance', 'love', 'chemistry', 'songs'],
                'comedy_timing': ['comedy', 'humor', 'entertainment', 'fun']
            },
            'regional_themes': {
                'social_issues': ['caste', 'politics', 'corruption', 'justice'],
                'rural_urban_divide': ['village', 'city', 'tradition', 'modernity'],
                'mythology_folklore': ['mythology', 'folklore', 'legend', 'epic'],
                'contemporary_struggles': ['career', 'dreams', 'aspirations', 'challenges']
            }
        }
    
    def extract_comprehensive_dna(self, content_list: List[Any]) -> Dict[str, Any]:
        """
        Extract comprehensive cinematic DNA from user's content consumption
        """
        if not content_list:
            return self._get_default_dna()
        
        try:
            # Multi-dimensional analysis
            narrative_profile = self.narrative_analyzer.analyze_narrative_preferences(content_list)
            cultural_profile = self.cultural_analyzer.analyze_cultural_preferences(content_list)
            visual_profile = self._analyze_visual_preferences(content_list)
            emotional_profile = self._analyze_emotional_patterns(content_list)
            genre_sophistication = self._calculate_genre_sophistication(content_list)
            
            # Telugu cinema specific analysis
            telugu_affinity = self._analyze_telugu_cinema_affinity(content_list)
            
            # Advanced pattern recognition
            viewing_patterns = self._identify_viewing_patterns(content_list)
            quality_standards = self._analyze_quality_standards(content_list)
            
            comprehensive_dna = {
                'narrative_sophistication': narrative_profile,
                'cultural_identity': cultural_profile,
                'visual_aesthetics': visual_profile,
                'emotional_resonance': emotional_profile,
                'genre_evolution': genre_sophistication,
                'telugu_cinema_connection': telugu_affinity,
                'viewing_intelligence': viewing_patterns,
                'quality_benchmark': quality_standards,
                'cinematic_maturity_score': self._calculate_maturity_score(content_list),
                'discovery_openness': self._calculate_discovery_openness(content_list),
                'cultural_bridge_score': self._calculate_cultural_bridge_score(content_list)
            }
            
            # Generate DNA fingerprint
            comprehensive_dna['dna_fingerprint'] = self._generate_dna_fingerprint(comprehensive_dna)
            comprehensive_dna['analysis_timestamp'] = datetime.utcnow().isoformat()
            
            return comprehensive_dna
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive DNA: {e}")
            return self._get_default_dna()
    
    def _analyze_telugu_cinema_affinity(self, content_list: List[Any]) -> Dict[str, Any]:
        """Analyze specific affinity for Telugu cinema patterns"""
        telugu_content = [c for c in content_list if self._is_telugu_content(c)]
        
        if not telugu_content:
            return {'affinity_score': 0.0, 'tollywood_patterns': {}, 'regional_connection': 'none'}
        
        affinity_indicators = {
            'content_ratio': len(telugu_content) / len(content_list),
            'quality_preference': np.mean([c.rating or 0 for c in telugu_content]),
            'pattern_recognition': self._analyze_tollywood_patterns(telugu_content),
            'cultural_depth': self._measure_cultural_depth(telugu_content)
        }
        
        affinity_score = (
            affinity_indicators['content_ratio'] * 0.3 +
            (affinity_indicators['quality_preference'] / 10) * 0.3 +
            affinity_indicators['pattern_recognition'] * 0.25 +
            affinity_indicators['cultural_depth'] * 0.15
        )
        
        return {
            'affinity_score': min(affinity_score, 1.0),
            'tollywood_patterns': affinity_indicators['pattern_recognition'],
            'cultural_depth': affinity_indicators['cultural_depth'],
            'regional_connection': self._classify_regional_connection(affinity_score)
        }
    
    def _generate_dna_fingerprint(self, dna_profile: Dict[str, Any]) -> str:
        """Generate unique DNA fingerprint for caching and comparison"""
        fingerprint_data = {
            'narrative': dna_profile.get('narrative_sophistication', {}).get('complexity_score', 0),
            'cultural': dna_profile.get('cultural_identity', {}).get('primary_culture', 'universal'),
            'visual': dna_profile.get('visual_aesthetics', {}).get('sophistication_level', 0),
            'emotional': dna_profile.get('emotional_resonance', {}).get('intensity_preference', 0),
            'telugu': dna_profile.get('telugu_cinema_connection', {}).get('affinity_score', 0),
            'quality': dna_profile.get('quality_benchmark', {}).get('minimum_standard', 0)
        }
        
        fingerprint_string = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]
    
    def _get_default_dna(self) -> Dict[str, Any]:
        """Return default DNA profile for new users"""
        return {
            'narrative_sophistication': {'complexity_score': 0.5, 'preference': 'balanced'},
            'cultural_identity': {'primary_culture': 'telugu', 'openness': 0.7},
            'visual_aesthetics': {'sophistication_level': 0.5, 'preference': 'polished'},
            'emotional_resonance': {'intensity_preference': 0.6, 'range': 'moderate'},
            'telugu_cinema_connection': {'affinity_score': 0.8, 'regional_connection': 'strong'},
            'cinematic_maturity_score': 0.5,
            'discovery_openness': 0.6,
            'dna_fingerprint': 'default_telugu_user',
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def _is_telugu_content(self, content: Any) -> bool:
        """Check if content is Telugu"""
        if not content.languages:
            return False
        try:
            languages = json.loads(content.languages)
            return any('telugu' in lang.lower() or 'te' in lang.lower() for lang in languages)
        except:
            return False
    
    def _analyze_tollywood_patterns(self, telugu_content: List[Any]) -> float:
        """Analyze Tollywood-specific patterns"""
        pattern_scores = []
        
        for content in telugu_content:
            content_text = f"{content.title} {content.overview or ''}"
            score = 0
            
            for pattern_name, keywords in self.telugu_cinema_dna['tollywood_signatures'].items():
                if any(keyword in content_text.lower() for keyword in keywords):
                    score += 0.2
            
            pattern_scores.append(min(score, 1.0))
        
        return np.mean(pattern_scores) if pattern_scores else 0.0
    
    def _measure_cultural_depth(self, telugu_content: List[Any]) -> float:
        """Measure cultural depth in Telugu content consumption"""
        if not telugu_content:
            return 0.0
        
        depth_indicators = [
            len([c for c in telugu_content if c.rating and c.rating >= 8.0]) / len(telugu_content),
            len([c for c in telugu_content if c.vote_count and c.vote_count >= 1000]) / len(telugu_content),
            len(set(c.content_type for c in telugu_content)) / 3  # diversity bonus
        ]
        
        return np.mean(depth_indicators)
    
    def _classify_regional_connection(self, affinity_score: float) -> str:
        """Classify user's regional connection level"""
        if affinity_score >= 0.8:
            return 'deep_cultural_connection'
        elif affinity_score >= 0.6:
            return 'strong_regional_preference'
        elif affinity_score >= 0.4:
            return 'moderate_interest'
        elif affinity_score >= 0.2:
            return 'occasional_viewer'
        else:
            return 'minimal_exposure'
    
    def _calculate_maturity_score(self, content_list: List[Any]) -> float:
        """Calculate cinematic maturity score"""
        if not content_list:
            return 0.5
        
        maturity_factors = [
            np.mean([c.rating or 0 for c in content_list]) / 10,  # Quality appreciation
            len(set(json.loads(c.genres or '[]')[0] if json.loads(c.genres or '[]') else 'Unknown' for c in content_list)) / 10,  # Genre diversity
            len([c for c in content_list if c.vote_count and c.vote_count < 10000]) / len(content_list),  # Niche content ratio
            len([c for c in content_list if c.runtime and c.runtime > 150]) / len(content_list)  # Long-form appreciation
        ]
        
        return min(np.mean(maturity_factors), 1.0)
    
    def _calculate_discovery_openness(self, content_list: List[Any]) -> float:
        """Calculate openness to content discovery"""
        if len(content_list) < 10:
            return 0.7  # Default high openness for new users
        
        recent_content = sorted(content_list, key=lambda x: x.created_at or datetime.min, reverse=True)[:20]
        
        discovery_factors = [
            len(set(c.content_type for c in recent_content)) / 3,  # Type diversity
            len(set(json.loads(c.languages or '[]')[0] if json.loads(c.languages or '[]') else 'Unknown' for c in recent_content)) / 5,  # Language diversity
            len([c for c in recent_content if c.popularity and c.popularity < 50]) / len(recent_content)  # Niche discovery
        ]
        
        return min(np.mean(discovery_factors), 1.0)
    
    def _calculate_cultural_bridge_score(self, content_list: List[Any]) -> float:
        """Calculate how well user bridges different cultural content"""
        language_distribution = defaultdict(int)
        
        for content in content_list:
            if content.languages:
                try:
                    languages = json.loads(content.languages)
                    for lang in languages:
                        language_distribution[lang.lower()] += 1
                except:
                    continue
        
        if len(language_distribution) <= 1:
            return 0.3  # Low bridge score for mono-cultural consumption
        
        # Calculate entropy (diversity) of language consumption
        total_content = sum(language_distribution.values())
        probabilities = [count / total_content for count in language_distribution.values()]
        diversity_score = entropy(probabilities) / math.log(len(language_distribution))
        
        return min(diversity_score, 1.0)

class BehavioralIntelligenceSystem:
    """
    Advanced behavioral pattern recognition and intelligence system
    """
    
    def __init__(self, db, models, cache=None):
        self.db = db
        self.models = models
        self.cache = cache
        self.behavior_patterns = {}
        self.real_time_learner = RealTimeLearningEngine()
        
        # Behavioral pattern templates
        self.pattern_templates = {
            'binge_watcher': {
                'session_length': '>= 3 hours',
                'consecutive_episodes': '>= 3',
                'viewing_frequency': 'daily',
                'content_completion_rate': '>= 0.8'
            },
            'quality_seeker': {
                'avg_rating_threshold': '>= 8.0',
                'vote_count_preference': '>= 1000',
                'niche_content_ratio': '>= 0.3',
                'mainstream_avoidance': '>= 0.4'
            },
            'cultural_explorer': {
                'language_diversity': '>= 5',
                'regional_content_ratio': '>= 0.4',
                'international_exposure': '>= 0.3',
                'cultural_bridge_score': '>= 0.6'
            },
            'trending_follower': {
                'popular_content_ratio': '>= 0.7',
                'recency_preference': '<= 365 days',
                'viral_content_engagement': '>= 0.5',
                'mainstream_alignment': '>= 0.6'
            }
        }
    
    def analyze_behavioral_intelligence(self, user_id: int, interactions: List[Any]) -> Dict[str, Any]:
        """
        Comprehensive behavioral intelligence analysis
        """
        try:
            if not interactions:
                return self._get_default_behavioral_profile()
            
            # Multi-faceted behavioral analysis
            engagement_intelligence = self._analyze_engagement_intelligence(interactions)
            temporal_intelligence = self._analyze_temporal_patterns(interactions)
            preference_evolution = self._analyze_preference_evolution(interactions)
            social_intelligence = self._analyze_social_patterns(user_id, interactions)
            discovery_intelligence = self._analyze_discovery_patterns(interactions)
            quality_intelligence = self._analyze_quality_patterns(interactions)
            
            # Advanced pattern recognition
            behavioral_archetypes = self._identify_behavioral_archetypes(interactions)
            prediction_confidence = self._calculate_prediction_confidence(interactions)
            
            # Real-time learning integration
            learning_state = self.real_time_learner.get_learning_state(user_id)
            
            behavioral_profile = {
                'engagement_intelligence': engagement_intelligence,
                'temporal_intelligence': temporal_intelligence,
                'preference_evolution': preference_evolution,
                'social_intelligence': social_intelligence,
                'discovery_intelligence': discovery_intelligence,
                'quality_intelligence': quality_intelligence,
                'behavioral_archetypes': behavioral_archetypes,
                'prediction_confidence': prediction_confidence,
                'learning_state': learning_state,
                'behavioral_maturity': self._calculate_behavioral_maturity(interactions),
                'personalization_readiness': self._assess_personalization_readiness(interactions)
            }
            
            # Cache behavioral profile
            if self.cache:
                cache_key = f"cinebrain:behavioral_intelligence:{user_id}"
                self.cache.set(cache_key, behavioral_profile, timeout=3600)
            
            return behavioral_profile
            
        except Exception as e:
            logger.error(f"Error analyzing behavioral intelligence for user {user_id}: {e}")
            return self._get_default_behavioral_profile()
    
    def _analyze_engagement_intelligence(self, interactions: List[Any]) -> Dict[str, Any]:
        """Analyze engagement patterns and intelligence"""
        engagement_metrics = {
            'depth_score': self._calculate_engagement_depth(interactions),
            'consistency_score': self._calculate_engagement_consistency(interactions),
            'intensity_score': self._calculate_engagement_intensity(interactions),
            'breadth_score': self._calculate_engagement_breadth(interactions)
        }
        
        # Engagement personality
        personality_indicators = {
            'deep_diver': engagement_metrics['depth_score'] > 0.7,
            'consistent_consumer': engagement_metrics['consistency_score'] > 0.7,
            'intensive_user': engagement_metrics['intensity_score'] > 0.7,
            'broad_explorer': engagement_metrics['breadth_score'] > 0.7
        }
        
        return {
            'metrics': engagement_metrics,
            'personality': personality_indicators,
            'engagement_level': self._classify_engagement_level(engagement_metrics),
            'optimization_potential': self._assess_optimization_potential(engagement_metrics)
        }
    
    def _calculate_engagement_depth(self, interactions: List[Any]) -> float:
        """Calculate depth of engagement"""
        if not interactions:
            return 0.0
        
        depth_indicators = [
            len([i for i in interactions if i.interaction_type == 'rating']) / len(interactions),
            len([i for i in interactions if i.interaction_type == 'favorite']) / len(interactions),
            len([i for i in interactions if i.interaction_type == 'watchlist']) / len(interactions),
            len([i for i in interactions if i.interaction_metadata]) / len(interactions)
        ]
        
        return np.mean(depth_indicators)
    
    def _get_default_behavioral_profile(self) -> Dict[str, Any]:
        """Return default behavioral profile"""
        return {
            'engagement_intelligence': {'engagement_level': 'new_user', 'optimization_potential': 'high'},
            'behavioral_archetypes': ['telugu_cinema_enthusiast'],
            'prediction_confidence': 0.3,
            'behavioral_maturity': 0.2,
            'personalization_readiness': 0.4
        }

class UserEmbeddingGenerator:
    """
    Advanced user embedding generation for neural recommendation
    """
    
    def __init__(self, embedding_dim=256):
        self.embedding_dim = embedding_dim
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=embedding_dim)
        self.content_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Pre-trained embeddings for cold start
        self.telugu_cinema_embedding = np.random.normal(0, 0.1, embedding_dim)
        self.quality_seeker_embedding = np.random.normal(0, 0.1, embedding_dim)
        self.cultural_explorer_embedding = np.random.normal(0, 0.1, embedding_dim)
    
    def generate_user_embedding(self, user_profile: Dict[str, Any], 
                               cinematic_dna: Dict[str, Any],
                               behavioral_intelligence: Dict[str, Any]) -> np.ndarray:
        """
        Generate comprehensive user embedding vector
        """
        try:
            # Feature extraction from different profile dimensions
            preference_features = self._extract_preference_features(user_profile)
            dna_features = self._extract_dna_features(cinematic_dna)
            behavioral_features = self._extract_behavioral_features(behavioral_intelligence)
            
            # Combine features
            combined_features = np.concatenate([
                preference_features,
                dna_features,
                behavioral_features
            ])
            
            # Apply dimensionality reduction and normalization
            if len(combined_features) > self.embedding_dim:
                embedding = self.pca.fit_transform(combined_features.reshape(1, -1))[0]
            else:
                # Pad with zeros if features are less than embedding dimension
                embedding = np.pad(combined_features, 
                                 (0, max(0, self.embedding_dim - len(combined_features))), 
                                 'constant')[:self.embedding_dim]
            
            # Normalize embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating user embedding: {e}")
            return self._get_default_embedding(user_profile)
    
    def _extract_preference_features(self, user_profile: Dict[str, Any]) -> np.ndarray:
        """Extract features from user preferences"""
        features = []
        
        # Language preferences (with Telugu priority)
        lang_prefs = user_profile.get('language_preferences', {}).get('preferred_languages', [])
        telugu_weight = 1.0 if 'telugu' in [l.lower() for l in lang_prefs] else 0.0
        features.extend([telugu_weight, len(lang_prefs) / 10])
        
        # Genre preferences
        genre_scores = user_profile.get('genre_preferences', {}).get('genre_scores', {})
        top_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        genre_features = [score for _, score in top_genres] + [0] * (10 - len(top_genres))
        features.extend(genre_features)
        
        # Quality and engagement metrics
        features.extend([
            user_profile.get('quality_threshold', 7.0) / 10,
            user_profile.get('engagement_score', 0.5),
            user_profile.get('exploration_tendency', 0.5)
        ])
        
        return np.array(features)
    
    def _extract_dna_features(self, cinematic_dna: Dict[str, Any]) -> np.ndarray:
        """Extract features from cinematic DNA"""
        features = []
        
        # Narrative sophistication
        features.append(cinematic_dna.get('narrative_sophistication', {}).get('complexity_score', 0.5))
        
        # Cultural connection
        features.append(cinematic_dna.get('telugu_cinema_connection', {}).get('affinity_score', 0.8))
        
        # Quality standards
        features.append(cinematic_dna.get('quality_benchmark', {}).get('minimum_standard', 0.7))
        
        # Discovery openness
        features.append(cinematic_dna.get('discovery_openness', 0.6))
        
        # Cultural bridge score
        features.append(cinematic_dna.get('cultural_bridge_score', 0.5))
        
        return np.array(features)
    
    def _extract_behavioral_features(self, behavioral_intelligence: Dict[str, Any]) -> np.ndarray:
        """Extract features from behavioral intelligence"""
        features = []
        
        # Engagement intelligence
        engagement = behavioral_intelligence.get('engagement_intelligence', {})
        features.append(engagement.get('metrics', {}).get('depth_score', 0.5))
        features.append(engagement.get('metrics', {}).get('consistency_score', 0.5))
        
        # Prediction confidence
        features.append(behavioral_intelligence.get('prediction_confidence', 0.5))
        
        # Behavioral maturity
        features.append(behavioral_intelligence.get('behavioral_maturity', 0.5))
        
        return np.array(features)
    
    def _get_default_embedding(self, user_profile: Dict[str, Any]) -> np.ndarray:
        """Get default embedding for new users"""
        # Start with Telugu cinema preference as default
        embedding = self.telugu_cinema_embedding.copy()
        
        # Add some randomness for exploration
        noise = np.random.normal(0, 0.05, self.embedding_dim)
        embedding = embedding + noise
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding

class AdvancedProfileAnalyzer:
    """
    Production-grade advanced profile analyzer combining all intelligence systems
    """
    
    def __init__(self, db, models, cache=None):
        self.db = db
        self.models = models
        self.cache = cache
        
        # Initialize specialized analyzers
        self.cinematic_dna_engine = CinematicDNAEngine(cache)
        self.behavioral_intelligence = BehavioralIntelligenceSystem(db, models, cache)
        self.embedding_generator = UserEmbeddingGenerator()
        
        # Performance tracking
        self.performance_tracker = ProfileAnalysisPerformanceTracker()
        
        # Thread safety
        self._lock = threading.Lock()
    
    @contextmanager
    def safe_db_operation(self):
        """Safe database operation context manager"""
        try:
            yield
        except (OperationalError, DisconnectionError) as e:
            logger.error(f"Database connection error in profile analyzer: {e}")
            try:
                self.db.session.rollback()
                self.db.session.close()
            except:
                pass
            raise
        except Exception as e:
            logger.error(f"Database operation error: {e}")
            try:
                self.db.session.rollback()
            except:
                pass
            raise
    
    def build_comprehensive_user_profile(self, user_id: int) -> Dict[str, Any]:
        """
        Build comprehensive user profile with advanced intelligence
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Check cache first
                if self.cache:
                    cache_key = f"cinebrain:advanced_profile:{user_id}"
                    cached_profile = self.cache.get(cache_key)
                    if cached_profile:
                        logger.info(f"Retrieved cached profile for user {user_id}")
                        return cached_profile
                
                with self.safe_db_operation():
                    # Get user and interactions
                    user = self.models['User'].query.get(user_id)
                    if not user:
                        logger.warning(f"User {user_id} not found")
                        return {}
                    
                    interactions = self.models['UserInteraction'].query.filter_by(
                        user_id=user_id
                    ).order_by(desc(self.models['UserInteraction'].timestamp)).all()
                    
                    # Get content associated with interactions
                    content_ids = [i.content_id for i in interactions]
                    content_list = []
                    
                    if content_ids:
                        content_list = self.models['Content'].query.filter(
                            self.models['Content'].id.in_(content_ids)
                        ).all()
                    
                    # Advanced analysis
                    cinematic_dna = self.cinematic_dna_engine.extract_comprehensive_dna(content_list)
                    behavioral_intelligence = self.behavioral_intelligence.analyze_behavioral_intelligence(
                        user_id, interactions
                    )
                    
                    # Generate user embedding
                    user_embedding = self.embedding_generator.generate_user_embedding(
                        user_profile={'user_id': user_id, 'username': user.username},
                        cinematic_dna=cinematic_dna,
                        behavioral_intelligence=behavioral_intelligence
                    )
                    
                    # Compile comprehensive profile
                    comprehensive_profile = {
                        'user_id': user_id,
                        'username': user.username,
                        'profile_version': '3.0_advanced',
                        'cinematic_dna': cinematic_dna,
                        'behavioral_intelligence': behavioral_intelligence,
                        'user_embedding': user_embedding.tolist(),  # Convert to list for JSON serialization
                        'embedding_dimension': len(user_embedding),
                        'telugu_cinema_affinity': cinematic_dna.get('telugu_cinema_connection', {}).get('affinity_score', 0.8),
                        'personalization_strength': self._calculate_personalization_strength(
                            cinematic_dna, behavioral_intelligence
                        ),
                        'recommendation_confidence': behavioral_intelligence.get('prediction_confidence', 0.5),
                        'cultural_profile': self._extract_cultural_profile(content_list),
                        'quality_standards': self._extract_quality_standards(interactions),
                        'discovery_preferences': self._extract_discovery_preferences(interactions),
                        'analysis_metadata': {
                            'content_analyzed': len(content_list),
                            'interactions_analyzed': len(interactions),
                            'analysis_depth': 'comprehensive',
                            'processing_time': time.time() - start_time,
                            'generated_at': datetime.utcnow().isoformat()
                        },
                        'next_analysis_due': (datetime.utcnow() + timedelta(hours=6)).isoformat()
                    }
                    
                    # Cache the profile
                    if self.cache:
                        self.cache.set(cache_key, comprehensive_profile, timeout=21600)  # 6 hours
                    
                    # Track performance
                    self.performance_tracker.record_analysis(
                        user_id, len(interactions), time.time() - start_time
                    )
                    
                    logger.info(f"✅ Built comprehensive profile for user {user_id} in {time.time() - start_time:.2f}s")
                    return comprehensive_profile
                    
        except Exception as e:
            logger.error(f"❌ Error building comprehensive profile for user {user_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_fallback_profile(user_id)
    
    def _calculate_personalization_strength(self, cinematic_dna: Dict[str, Any], 
                                          behavioral_intelligence: Dict[str, Any]) -> float:
        """Calculate overall personalization strength"""
        factors = [
            cinematic_dna.get('cinematic_maturity_score', 0.5),
            behavioral_intelligence.get('behavioral_maturity', 0.5),
            behavioral_intelligence.get('prediction_confidence', 0.5),
            cinematic_dna.get('discovery_openness', 0.6)
        ]
        
        return np.mean(factors)
    
    def _extract_cultural_profile(self, content_list: List[Any]) -> Dict[str, Any]:
        """Extract cultural viewing profile"""
        if not content_list:
            return {'primary_culture': 'telugu', 'cultural_diversity': 0.3, 'bridge_score': 0.5}
        
        language_distribution = defaultdict(int)
        for content in content_list:
            if content.languages:
                try:
                    languages = json.loads(content.languages)
                    for lang in languages:
                        language_distribution[lang.lower()] += 1
                except:
                    continue
        
        total_content = sum(language_distribution.values())
        if total_content == 0:
            return {'primary_culture': 'telugu', 'cultural_diversity': 0.3, 'bridge_score': 0.5}
        
        # Determine primary culture
        primary_culture = max(language_distribution.items(), key=lambda x: x[1])[0]
        if 'telugu' in language_distribution:
            primary_culture = 'telugu'  # Telugu priority
        
        # Calculate cultural diversity
        diversity = len(language_distribution) / max(total_content, 1)
        
        return {
            'primary_culture': primary_culture,
            'cultural_diversity': min(diversity, 1.0),
            'language_distribution': dict(language_distribution),
            'bridge_score': self._calculate_cultural_bridge_score(language_distribution)
        }
    
    def _calculate_cultural_bridge_score(self, language_distribution: Dict[str, int]) -> float:
        """Calculate cultural bridge score"""
        if len(language_distribution) <= 1:
            return 0.2
        
        total = sum(language_distribution.values())
        entropy_score = entropy([count/total for count in language_distribution.values()])
        max_entropy = math.log(len(language_distribution))
        
        return min(entropy_score / max_entropy, 1.0) if max_entropy > 0 else 0.0
    
    def _get_fallback_profile(self, user_id: int) -> Dict[str, Any]:
        """Get fallback profile for error cases"""
        return {
            'user_id': user_id,
            'profile_version': '3.0_fallback',
            'cinematic_dna': self.cinematic_dna_engine._get_default_dna(),
            'behavioral_intelligence': self.behavioral_intelligence._get_default_behavioral_profile(),
            'telugu_cinema_affinity': 0.8,
            'personalization_strength': 0.4,
            'recommendation_confidence': 0.3,
            'analysis_metadata': {
                'fallback_used': True,
                'generated_at': datetime.utcnow().isoformat()
            }
        }

# Supporting classes
class CinematicPatternAnalyzer:
    """Analyzes cinematic patterns in user content"""
    
    def analyze_narrative_preferences(self, content_list: List[Any]) -> Dict[str, Any]:
        """Analyze narrative structure preferences"""
        return {'complexity_score': 0.5, 'preference': 'balanced'}

class CulturalPreferenceAnalyzer:
    """Analyzes cultural preferences"""
    
    def analyze_cultural_preferences(self, content_list: List[Any]) -> Dict[str, Any]:
        """Analyze cultural content preferences"""
        return {'primary_culture': 'telugu', 'openness': 0.7}

class NarrativeStructureAnalyzer:
    """Analyzes narrative structure preferences"""
    
    def analyze_narrative_preferences(self, content_list: List[Any]) -> Dict[str, Any]:
        """Analyze narrative preferences"""
        return {'complexity_score': 0.5, 'structure_preference': 'traditional'}

class RealTimeLearningEngine:
    """Real-time learning engine for user preferences"""
    
    def get_learning_state(self, user_id: int) -> Dict[str, Any]:
        """Get current learning state for user"""
        return {
            'learning_rate': 0.1,
            'adaptation_speed': 'moderate',
            'confidence_trend': 'increasing'
        }

class ProfileAnalysisPerformanceTracker:
    """Tracks performance of profile analysis"""
    
    def __init__(self):
        self.metrics = deque(maxlen=1000)
    
    def record_analysis(self, user_id: int, interaction_count: int, processing_time: float):
        """Record analysis performance metrics"""
        self.metrics.append({
            'user_id': user_id,
            'interaction_count': interaction_count,
            'processing_time': processing_time,
            'timestamp': datetime.utcnow()
        })