# backend/services/personalized.py
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import json
import logging
import jwt
import math
import random
import heapq
from functools import wraps
import hashlib
from sqlalchemy import func, and_, or_, desc, text, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import joinedload, relationship
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
from typing import Dict, List, Tuple, Optional, Any, Set
import re
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Create personalized blueprint
personalized_bp = Blueprint('personalized', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Global variables - will be initialized by main app
db = None
cache = None
User = None
Content = None
UserInteraction = None
AnonymousInteraction = None
ContentPerson = None
Person = None
Review = None
UserPreference = None
RecommendationFeedback = None
UserSession = None
app = None
services = None

def init_personalized(flask_app, database, models, app_services, app_cache):
    """Initialize personalized module with app context and models"""
    global db, cache, User, Content, UserInteraction, AnonymousInteraction
    global ContentPerson, Person, Review, UserPreference, RecommendationFeedback
    global UserSession, app, services
    
    app = flask_app
    db = database
    cache = app_cache
    services = app_services
    
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    AnonymousInteraction = models['AnonymousInteraction']
    ContentPerson = models.get('ContentPerson')
    Person = models.get('Person')
    Review = models.get('Review')
    
    # Create additional models if they don't exist
    if 'UserPreference' not in models:
        UserPreference = create_user_preference_model(db)
        models['UserPreference'] = UserPreference
    else:
        UserPreference = models['UserPreference']
    
    if 'RecommendationFeedback' not in models:
        RecommendationFeedback = create_recommendation_feedback_model(db)
        models['RecommendationFeedback'] = RecommendationFeedback
    else:
        RecommendationFeedback = models['RecommendationFeedback']
    
    if 'UserSession' not in models:
        UserSession = create_user_session_model(db)
        models['UserSession'] = UserSession
    else:
        UserSession = models['UserSession']
    
    # Create tables if they don't exist
    with flask_app.app_context():
        db.create_all()

def create_user_preference_model(db):
    """Create UserPreference model for storing detailed user preferences"""
    
    class UserPreference(db.Model):
        __tablename__ = 'user_preferences'
        
        id = db.Column(db.Integer, primary_key=True)
        user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
        
        # Preference data stored as JSON
        genre_preferences = db.Column(db.Text)  # JSON: {"action": 0.8, "drama": 0.6, ...}
        language_preferences = db.Column(db.Text)  # JSON: {"english": 0.7, "telugu": 0.9, ...}
        content_type_preferences = db.Column(db.Text)  # JSON: {"movie": 0.6, "tv": 0.3, ...}
        quality_preferences = db.Column(db.Text)  # JSON: {"min_rating": 6.5, "avg_rating": 7.8, ...}
        runtime_preferences = db.Column(db.Text)  # JSON: {"min": 90, "max": 180, "avg": 120}
        
        # Behavioral patterns
        viewing_patterns = db.Column(db.Text)  # JSON: viewing time patterns
        search_patterns = db.Column(db.Text)  # JSON: search behavior analysis
        sequence_patterns = db.Column(db.Text)  # JSON: sequential viewing patterns
        
        # Advanced preferences
        cast_crew_preferences = db.Column(db.Text)  # JSON: favorite actors/directors
        franchise_preferences = db.Column(db.Text)  # JSON: franchise affinities
        mood_preferences = db.Column(db.Text)  # JSON: mood-based preferences
        
        # Recommendation metadata
        exploration_tendency = db.Column(db.Float, default=0.5)
        diversity_preference = db.Column(db.Float, default=0.5)
        recency_bias = db.Column(db.Float, default=0.5)
        
        # Profile strength and confidence
        profile_strength = db.Column(db.Float, default=0.0)
        confidence_score = db.Column(db.Float, default=0.0)
        
        # Timestamps
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        last_calculated = db.Column(db.DateTime, default=datetime.utcnow)
        
        # User mode tracking
        current_mode = db.Column(db.String(50), default='discovery_mode')  # discovery, comfort, binge, selective
        mode_history = db.Column(db.Text)  # JSON: history of mode changes
        
    return UserPreference

def create_recommendation_feedback_model(db):
    """Create RecommendationFeedback model for tracking recommendation performance"""
    
    class RecommendationFeedback(db.Model):
        __tablename__ = 'recommendation_feedback'
        
        id = db.Column(db.Integer, primary_key=True)
        user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
        content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
        
        # Recommendation metadata
        recommendation_score = db.Column(db.Float)
        recommendation_method = db.Column(db.String(100))  # Which algorithm generated this
        recommendation_reason = db.Column(db.Text)
        recommendation_rank = db.Column(db.Integer)  # Position in recommendation list
        
        # User feedback
        feedback_type = db.Column(db.String(50))  # clicked, ignored, liked, disliked, watched
        user_rating = db.Column(db.Float)
        watch_duration = db.Column(db.Integer)  # In seconds
        completion_rate = db.Column(db.Float)  # Percentage of content watched
        
        # Context
        device_type = db.Column(db.String(50))
        time_of_day = db.Column(db.Integer)  # Hour of day
        day_of_week = db.Column(db.Integer)
        user_mode = db.Column(db.String(50))  # User mode at time of recommendation
        
        # Performance metrics
        was_successful = db.Column(db.Boolean, default=False)
        engagement_score = db.Column(db.Float)
        
        # Timestamps
        recommended_at = db.Column(db.DateTime, default=datetime.utcnow)
        feedback_at = db.Column(db.DateTime)
        
        __table_args__ = (
            db.Index('idx_user_feedback', 'user_id', 'feedback_at'),
            db.Index('idx_method_performance', 'recommendation_method', 'was_successful'),
        )
        
    return RecommendationFeedback

def create_user_session_model(db):
    """Create UserSession model for tracking user sessions"""
    
    class UserSession(db.Model):
        __tablename__ = 'user_sessions'
        
        id = db.Column(db.Integer, primary_key=True)
        user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
        session_id = db.Column(db.String(100), unique=True, nullable=False)
        
        # Session data
        start_time = db.Column(db.DateTime, default=datetime.utcnow)
        end_time = db.Column(db.DateTime)
        duration = db.Column(db.Integer)  # In seconds
        
        # Session activity
        interactions_count = db.Column(db.Integer, default=0)
        content_viewed = db.Column(db.Text)  # JSON: list of content IDs
        genres_explored = db.Column(db.Text)  # JSON: genres viewed in session
        
        # Session characteristics
        session_type = db.Column(db.String(50))  # browsing, watching, searching
        user_mode = db.Column(db.String(50))  # discovery, comfort, binge, selective
        
        # Device and context
        device_type = db.Column(db.String(50))
        ip_address = db.Column(db.String(45))
        user_agent = db.Column(db.Text)
        
        __table_args__ = (
            db.Index('idx_user_sessions', 'user_id', 'start_time'),
        )
        
    return UserSession

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.secret_key, algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
                
            # Update last active
            current_user.last_active = datetime.utcnow()
            db.session.commit()
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return jsonify({'error': 'Authentication failed'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

class EnhancedUserProfiler:
    """Enhanced user profiling with real-time adaptive learning"""
    
    def __init__(self):
        # Interaction weights with refined granularity
        self.interaction_weights = {
            'rating': {'weight': 1.0, 'confidence': 0.95},
            'favorite': {'weight': 0.95, 'confidence': 0.9},
            'watchlist': {'weight': 0.8, 'confidence': 0.8},
            'like': {'weight': 0.7, 'confidence': 0.75},
            'view': {'weight': 0.5, 'confidence': 0.6},
            'search': {'weight': 0.3, 'confidence': 0.4},
            'click': {'weight': 0.2, 'confidence': 0.3}
        }
        
        # Temporal decay factors
        self.temporal_factors = {
            'immediate': 1.0,    # Last 7 days
            'recent': 0.9,       # Last 30 days
            'moderate': 0.7,     # Last 90 days
            'old': 0.5,          # Last 180 days
            'ancient': 0.3       # Older than 180 days
        }
        
        # User modes
        self.user_modes = {
            'discovery_mode': {'exploration': 0.8, 'familiar': 0.2},
            'comfort_mode': {'exploration': 0.2, 'familiar': 0.8},
            'binge_mode': {'exploration': 0.3, 'familiar': 0.7},
            'selective_mode': {'exploration': 0.4, 'familiar': 0.6}
        }
    
    def build_comprehensive_profile(self, user_id: int, force_update: bool = False) -> Dict[str, Any]:
        """Build ultra-comprehensive user profile with advanced analytics"""
        try:
            # Check if we have a stored preference
            user_pref = UserPreference.query.filter_by(user_id=user_id).first()
            
            # Check if we need to update
            if user_pref and not force_update:
                time_since_update = datetime.utcnow() - user_pref.last_calculated
                if time_since_update.total_seconds() < 1800:  # 30 minutes
                    return self._load_stored_profile(user_pref)
            
            # Build new profile
            profile = self._build_new_profile(user_id)
            
            # Store/update in database
            self._store_profile(user_id, profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error building comprehensive profile for user {user_id}: {e}")
            return self._get_default_profile(user_id)
    
    def _build_new_profile(self, user_id: int) -> Dict[str, Any]:
        """Build a new comprehensive profile from scratch"""
        
        # Get all user interactions
        interactions = db.session.query(UserInteraction, Content).join(
            Content, UserInteraction.content_id == Content.id
        ).filter(UserInteraction.user_id == user_id).order_by(
            UserInteraction.timestamp.desc()
        ).all()
        
        if not interactions:
            return self._get_default_profile(user_id)
        
        profile = {
            'user_id': user_id,
            'interaction_count': len(interactions),
            'profile_strength': 0.0,
            
            # Core preferences
            'genre_preferences': defaultdict(float),
            'language_preferences': defaultdict(float),
            'content_type_preferences': defaultdict(float),
            'quality_preferences': {},
            'runtime_preferences': {},
            'release_period_preferences': defaultdict(float),
            
            # Advanced behavioral patterns
            'viewing_patterns': {},
            'search_behavior': {},
            'rating_patterns': {},
            'sequence_patterns': [],
            'seasonal_patterns': defaultdict(list),
            'contextual_patterns': defaultdict(dict),
            
            # Preference evolution
            'preference_evolution': defaultdict(list),
            'trending_interests': [],
            'declining_interests': [],
            
            # Social and collaborative signals
            'similarity_clusters': [],
            'influence_factors': {},
            
            # Predictive features
            'next_likely_genres': [],
            'mood_indicators': {},
            'exploration_tendency': 0.0,
            'binge_patterns': {},
            
            # Content-specific insights
            'cast_crew_preferences': defaultdict(float),
            'director_preferences': defaultdict(float),
            'franchise_preferences': defaultdict(float),
            
            # User mode
            'current_mode': 'discovery_mode',
            'mode_history': [],
            
            # Affinity scores
            'affinity_scores': {},
            
            # Metadata
            'last_updated': datetime.utcnow().isoformat(),
            'confidence_score': 0.0
        }
        
        # Process interactions
        self._extract_core_preferences(profile, interactions)
        self._extract_temporal_patterns(profile, interactions)
        self._extract_sequential_patterns(profile, interactions)
        self._extract_search_patterns(profile, user_id)
        self._extract_social_signals(profile, user_id)
        self._extract_content_creator_preferences(profile, interactions)
        self._calculate_prediction_features(profile, interactions)
        self._calculate_affinity_scores(profile, interactions)
        self._detect_user_mode(profile, interactions)
        self._calculate_confidence_scores(profile, interactions)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        profile = self._serialize_profile(profile)
        
        return profile
    
    def _extract_core_preferences(self, profile: Dict, interactions: List[Tuple]) -> None:
        """Extract core user preferences with confidence weighting"""
        total_weight = 0
        quality_ratings = []
        runtimes = []
        
        for interaction, content in interactions:
            # Calculate temporal weight
            days_ago = (datetime.utcnow() - interaction.timestamp).days
            temporal_weight = self._get_temporal_weight(days_ago)
            
            # Get interaction weight and confidence
            interaction_data = self.interaction_weights.get(
                interaction.interaction_type, 
                {'weight': 0.1, 'confidence': 0.2}
            )
            base_weight = interaction_data['weight'] * temporal_weight
            
            # Apply rating boost
            rating_boost = 1.0
            if interaction.rating:
                rating_boost = (interaction.rating / 5.0) * 1.5
                quality_ratings.append(interaction.rating)
            
            final_weight = base_weight * rating_boost
            total_weight += final_weight
            
            # Extract genres
            try:
                genres = json.loads(content.genres or '[]')
                for i, genre in enumerate(genres[:5]):
                    importance = 1.0 / (i + 1)
                    profile['genre_preferences'][genre.lower()] += final_weight * importance
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Extract languages
            try:
                languages = json.loads(content.languages or '[]')
                for lang in languages:
                    profile['language_preferences'][lang.lower()] += final_weight
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Content type preferences
            profile['content_type_preferences'][content.content_type] += final_weight
            
            # Runtime preferences
            if content.runtime:
                runtimes.append((content.runtime, final_weight))
            
            # Release period preferences
            if content.release_date:
                year = content.release_date.year
                decade = f"{(year // 10) * 10}s"
                profile['release_period_preferences'][decade] += final_weight
        
        # Normalize preferences
        if total_weight > 0:
            for key in ['genre_preferences', 'language_preferences', 'content_type_preferences']:
                for item in profile[key]:
                    profile[key][item] /= total_weight
        
        # Calculate quality preferences
        if quality_ratings:
            profile['quality_preferences'] = {
                'average_rating': np.mean(quality_ratings),
                'rating_std': np.std(quality_ratings),
                'high_quality_bias': len([r for r in quality_ratings if r >= 4]) / len(quality_ratings),
                'rating_distribution': dict(Counter(quality_ratings))
            }
        
        # Calculate runtime preferences
        if runtimes:
            weighted_runtimes = [runtime for runtime, weight in runtimes 
                               for _ in range(int(weight * 10))]
            if weighted_runtimes:
                profile['runtime_preferences'] = {
                    'preferred_range': [
                        np.percentile(weighted_runtimes, 25), 
                        np.percentile(weighted_runtimes, 75)
                    ],
                    'average': np.mean(weighted_runtimes),
                    'tolerance': np.std(weighted_runtimes)
                }
    
    def _extract_temporal_patterns(self, profile: Dict, interactions: List[Tuple]) -> None:
        """Extract advanced temporal viewing patterns"""
        hourly_patterns = defaultdict(list)
        daily_patterns = defaultdict(list)
        weekly_patterns = defaultdict(int)
        monthly_patterns = defaultdict(int)
        
        for interaction, content in interactions:
            timestamp = interaction.timestamp
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            week_of_year = timestamp.isocalendar()[1]
            month = timestamp.month
            
            # Hourly patterns with content type
            hourly_patterns[hour].append(content.content_type)
            
            # Daily patterns with genres
            try:
                genres = json.loads(content.genres or '[]')
                if genres:
                    daily_patterns[day_of_week].extend(genres)
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Weekly and monthly activity
            weekly_patterns[week_of_year] += 1
            monthly_patterns[month] += 1
        
        # Analyze patterns
        profile['viewing_patterns'] = {
            'peak_hours': self._find_peak_periods(hourly_patterns),
            'preferred_days': self._find_preferred_days(daily_patterns),
            'activity_rhythm': {
                'weekly_consistency': np.std(list(weekly_patterns.values())) if weekly_patterns else 0,
                'seasonal_preference': dict(monthly_patterns)
            },
            'content_timing': self._analyze_content_timing(hourly_patterns, daily_patterns)
        }
    
    def _extract_sequential_patterns(self, profile: Dict, interactions: List[Tuple]) -> None:
        """Extract sequence-aware behavioral patterns"""
        sequences = []
        current_sequence = []
        
        for i, (interaction, content) in enumerate(interactions):
            try:
                genres = json.loads(content.genres or '[]')
                primary_genre = genres[0].lower() if genres else 'unknown'
                
                item = {
                    'content_type': content.content_type,
                    'primary_genre': primary_genre,
                    'rating': content.rating or 0,
                    'interaction_type': interaction.interaction_type,
                    'timestamp': interaction.timestamp.isoformat()
                }
                
                current_sequence.append(item)
                
                # Detect sequence breaks
                if i > 0:
                    time_gap = (interaction.timestamp - interactions[i-1][0].timestamp).total_seconds()
                    if time_gap > 86400 or len(current_sequence) > 10:
                        if len(current_sequence) >= 3:
                            sequences.append(current_sequence.copy())
                        current_sequence = [item]
            
            except (json.JSONDecodeError, TypeError, IndexError):
                continue
        
        # Add final sequence
        if len(current_sequence) >= 3:
            sequences.append(current_sequence)
        
        # Analyze sequences for patterns
        profile['sequence_patterns'] = self._analyze_sequences(sequences)
    
    def _extract_search_patterns(self, profile: Dict, user_id: int) -> None:
        """Extract and analyze search behavior patterns"""
        try:
            search_interactions = UserInteraction.query.filter_by(
                user_id=user_id,
                interaction_type='search'
            ).all()
            
            if not search_interactions:
                profile['search_behavior'] = {'patterns': [], 'keywords': [], 'intent_analysis': {}}
                return
            
            search_queries = []
            search_timestamps = []
            
            for interaction in search_interactions:
                try:
                    metadata = json.loads(interaction.interaction_metadata or '{}')
                    query = metadata.get('search_query', '')
                    if query:
                        search_queries.append(query.lower())
                        search_timestamps.append(interaction.timestamp)
                except (json.JSONDecodeError, TypeError):
                    continue
            
            if search_queries:
                profile['search_behavior'] = {
                    'total_searches': len(search_queries),
                    'unique_queries': len(set(search_queries)),
                    'query_diversity': len(set(search_queries)) / len(search_queries) if search_queries else 0,
                    'common_keywords': self._extract_search_keywords(search_queries),
                    'search_intent': self._analyze_search_intent(search_queries),
                    'temporal_search_patterns': self._analyze_search_timing(search_timestamps)
                }
            
        except Exception as e:
            logger.error(f"Error extracting search patterns: {e}")
            profile['search_behavior'] = {}
    
    def _extract_social_signals(self, profile: Dict, user_id: int) -> None:
        """Extract social and collaborative signals"""
        try:
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            user_content_ids = set([i.content_id for i in user_interactions])
            
            if len(user_content_ids) < 5:
                profile['similarity_clusters'] = []
                return
            
            # Find users with similar preferences
            similar_users = db.session.query(
                UserInteraction.user_id, 
                func.count().label('common_content')
            ).filter(
                UserInteraction.content_id.in_(user_content_ids),
                UserInteraction.user_id != user_id
            ).group_by(UserInteraction.user_id).having(func.count() >= 3).all()
            
            similarity_scores = []
            for other_user_id, common_count in similar_users[:50]:  # Limit to top 50
                other_interactions = UserInteraction.query.filter_by(user_id=other_user_id).all()
                other_content_ids = set([i.content_id for i in other_interactions])
                
                intersection = len(user_content_ids & other_content_ids)
                union = len(user_content_ids | other_content_ids)
                
                if union > 0:
                    jaccard_similarity = intersection / union
                    if jaccard_similarity > 0.1:
                        similarity_scores.append({
                            'user_id': other_user_id,
                            'similarity': jaccard_similarity,
                            'common_content': intersection
                        })
            
            # Sort by similarity and keep top 10
            similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
            profile['similarity_clusters'] = similarity_scores[:10]
            
        except Exception as e:
            logger.error(f"Error extracting social signals: {e}")
            profile['similarity_clusters'] = []
    
    def _extract_content_creator_preferences(self, profile: Dict, interactions: List[Tuple]) -> None:
        """Extract preferences for cast, crew, directors"""
        try:
            content_ids = [content.id for _, content in interactions]
            
            if not content_ids or not ContentPerson:
                return
            
            # Get cast and crew information
            cast_crew = db.session.query(ContentPerson, Person).join(
                Person, ContentPerson.person_id == Person.id
            ).filter(ContentPerson.content_id.in_(content_ids)).all()
            
            director_counts = defaultdict(float)
            actor_counts = defaultdict(float)
            
            for content_person, person in cast_crew:
                weight = 1.0
                
                # Find the interaction weight for this content
                for interaction, content in interactions:
                    if content.id == content_person.content_id:
                        interaction_data = self.interaction_weights.get(
                            interaction.interaction_type, 
                            {'weight': 0.1}
                        )
                        weight = interaction_data['weight']
                        if interaction.rating:
                            weight *= (interaction.rating / 5.0)
                        break
                
                if content_person.role_type == 'crew' and content_person.job == 'Director':
                    director_counts[person.name] += weight
                elif content_person.role_type == 'cast':
                    actor_counts[person.name] += weight
            
            # Keep top preferences
            profile['director_preferences'] = dict(
                sorted(director_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            )
            profile['cast_crew_preferences'] = dict(
                sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)[:30]
            )
            
        except Exception as e:
            logger.error(f"Error extracting content creator preferences: {e}")
    
    def _calculate_prediction_features(self, profile: Dict, interactions: List[Tuple]) -> None:
        """Calculate features for predicting future preferences"""
        if len(interactions) < 5:
            return
        
        # Analyze preference evolution
        time_windows = [7, 30, 90, 180]  # days
        current_time = datetime.utcnow()
        
        preference_evolution = {}
        
        for window in time_windows:
            window_start = current_time - timedelta(days=window)
            window_interactions = [(i, c) for i, c in interactions if i.timestamp >= window_start]
            
            if window_interactions:
                window_genres = defaultdict(int)
                for interaction, content in window_interactions:
                    try:
                        genres = json.loads(content.genres or '[]')
                        for genre in genres:
                            window_genres[genre.lower()] += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                preference_evolution[f'{window}_days'] = dict(window_genres)
        
        profile['preference_evolution'] = preference_evolution
        
        # Calculate exploration tendency
        all_genres = set()
        for interaction, content in interactions:
            try:
                genres = json.loads(content.genres or '[]')
                all_genres.update([g.lower() for g in genres])
            except (json.JSONDecodeError, TypeError):
                pass
        
        profile['exploration_tendency'] = len(all_genres) / len(interactions) if interactions else 0
        
        # Predict trending interests
        if len(time_windows) >= 2:
            recent_genres = preference_evolution.get('30_days', {})
            older_genres = preference_evolution.get('90_days', {})
            
            trending = []
            declining = []
            
            for genre, recent_count in recent_genres.items():
                older_count = older_genres.get(genre, 0)
                if recent_count > older_count * 1.5:
                    trending.append(genre)
                elif recent_count < older_count * 0.5:
                    declining.append(genre)
            
            profile['trending_interests'] = trending
            profile['declining_interests'] = declining
    
    def _calculate_affinity_scores(self, profile: Dict, interactions: List[Tuple]) -> None:
        """Calculate real-time affinity scores for different content attributes"""
        affinity_scores = {
            'genre_affinity': defaultdict(float),
            'actor_affinity': defaultdict(float),
            'director_affinity': defaultdict(float),
            'decade_affinity': defaultdict(float),
            'language_affinity': defaultdict(float),
            'quality_tier_affinity': defaultdict(float)
        }
        
        for idx, (interaction, content) in enumerate(interactions[:100]):  # Recent 100
            # Exponential decay for recency
            recency_weight = math.exp(-idx * 0.02)
            
            # Base weight from interaction type
            base_weight = self.interaction_weights.get(
                interaction.interaction_type, {'weight': 0.1}
            )['weight']
            
            # Rating multiplier
            rating_mult = (interaction.rating / 5.0) if interaction.rating else 0.7
            
            final_weight = base_weight * recency_weight * rating_mult
            
            # Update affinities
            try:
                # Genre affinity
                genres = json.loads(content.genres or '[]')
                for genre in genres:
                    affinity_scores['genre_affinity'][genre.lower()] += final_weight
                
                # Language affinity
                languages = json.loads(content.languages or '[]')
                for lang in languages:
                    affinity_scores['language_affinity'][lang.lower()] += final_weight
                
                # Decade affinity
                if content.release_date:
                    decade = f"{(content.release_date.year // 10) * 10}s"
                    affinity_scores['decade_affinity'][decade] += final_weight
                
                # Quality tier affinity
                if content.rating:
                    if content.rating >= 8:
                        tier = 'exceptional'
                    elif content.rating >= 7:
                        tier = 'great'
                    elif content.rating >= 6:
                        tier = 'good'
                    else:
                        tier = 'average'
                    affinity_scores['quality_tier_affinity'][tier] += final_weight
                    
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Normalize scores
        for category in affinity_scores:
            total = sum(affinity_scores[category].values())
            if total > 0:
                for key in affinity_scores[category]:
                    affinity_scores[category][key] /= total
        
        profile['affinity_scores'] = dict(affinity_scores)
    
    def _detect_user_mode(self, profile: Dict, interactions: List[Tuple]) -> None:
        """Detect current user behavior mode"""
        if not interactions:
            profile['current_mode'] = 'discovery_mode'
            return
        
        # Analyze last 5 interactions
        recent = interactions[:5]
        
        # Check for exploration
        unique_genres = set()
        for interaction, content in recent:
            try:
                genres = json.loads(content.genres or '[]')
                unique_genres.update(genres)
            except (json.JSONDecodeError, TypeError):
                pass
        
        if len(unique_genres) > 7:
            profile['current_mode'] = 'discovery_mode'
            return
        
        # Check for binge pattern
        if len(recent) >= 3:
            time_gaps = []
            for i in range(1, len(recent)):
                gap = (recent[i-1][0].timestamp - recent[i][0].timestamp).seconds / 60
                time_gaps.append(gap)
            
            if time_gaps and np.mean(time_gaps) < 30:  # Less than 30 min between views
                profile['current_mode'] = 'binge_mode'
                return
        
        # Check for selective pattern
        ratings = [i.rating for i, _ in recent if i.rating]
        if len(recent) < 3 and ratings and np.mean(ratings) >= 4:
            profile['current_mode'] = 'selective_mode'
            return
        
        profile['current_mode'] = 'comfort_mode'
    
    def _calculate_confidence_scores(self, profile: Dict, interactions: List[Tuple]) -> None:
        """Calculate confidence scores for profile reliability"""
        interaction_count = len(interactions)
        
        # Base confidence from interaction count
        base_confidence = min(interaction_count / 50.0, 1.0)
        
        # Adjust for interaction diversity
        interaction_types = set([i.interaction_type for i, _ in interactions])
        diversity_bonus = len(interaction_types) / 6.0
        
        # Adjust for rating data availability
        rating_interactions = [i for i, _ in interactions if i.rating is not None]
        rating_bonus = len(rating_interactions) / max(interaction_count, 1) * 0.3
        
        # Adjust for temporal spread
        if interactions:
            time_span = (interactions[0][0].timestamp - interactions[-1][0].timestamp).days
            temporal_bonus = min(time_span / 90.0, 1.0) * 0.2
        else:
            temporal_bonus = 0
        
        confidence = min(base_confidence + diversity_bonus + rating_bonus + temporal_bonus, 1.0)
        profile['confidence_score'] = round(confidence, 3)
        profile['profile_strength'] = round(confidence * 100, 1)
    
    def _get_temporal_weight(self, days_ago: int) -> float:
        """Get temporal weight based on how many days ago the interaction occurred"""
        if days_ago <= 7:
            return self.temporal_factors['immediate']
        elif days_ago <= 30:
            return self.temporal_factors['recent']
        elif days_ago <= 90:
            return self.temporal_factors['moderate']
        elif days_ago <= 180:
            return self.temporal_factors['old']
        else:
            return self.temporal_factors['ancient']
    
    def _find_peak_periods(self, hourly_patterns: Dict) -> List[int]:
        """Find peak viewing hours"""
        hour_counts = {hour: len(content_types) for hour, content_types in hourly_patterns.items()}
        if not hour_counts:
            return []
        
        max_count = max(hour_counts.values())
        threshold = max_count * 0.7
        
        return [hour for hour, count in hour_counts.items() if count >= threshold]
    
    def _find_preferred_days(self, daily_patterns: Dict) -> List[int]:
        """Find preferred viewing days"""
        day_counts = {day: len(genres) for day, genres in daily_patterns.items()}
        if not day_counts:
            return []
        
        max_count = max(day_counts.values())
        threshold = max_count * 0.6
        
        return [day for day, count in day_counts.items() if count >= threshold]
    
    def _analyze_content_timing(self, hourly_patterns: Dict, daily_patterns: Dict) -> Dict:
        """Analyze when different content types are preferred"""
        timing_analysis = {}
        
        # Analyze content type by hour
        hour_content_types = defaultdict(lambda: defaultdict(int))
        for hour, content_types in hourly_patterns.items():
            for content_type in content_types:
                hour_content_types[hour][content_type] += 1
        
        timing_analysis['hourly_preferences'] = dict(hour_content_types)
        
        return timing_analysis
    
    def _analyze_sequences(self, sequences: List[List[Dict]]) -> Dict:
        """Analyze sequential patterns in viewing behavior"""
        if not sequences:
            return {}
        
        # Find common transitions
        transitions = defaultdict(int)
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current = f"{sequence[i]['content_type']}_{sequence[i]['primary_genre']}"
                next_item = f"{sequence[i+1]['content_type']}_{sequence[i+1]['primary_genre']}"
                transitions[f"{current}->{next_item}"] += 1
        
        # Find binge patterns
        binge_patterns = []
        for sequence in sequences:
            if len(sequence) >= 3:
                content_types = [item['content_type'] for item in sequence]
                if len(set(content_types)) == 1:
                    binge_patterns.append(content_types[0])
        
        return {
            'common_transitions': dict(sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]),
            'binge_patterns': dict(Counter(binge_patterns)),
            'average_sequence_length': np.mean([len(seq) for seq in sequences]) if sequences else 0,
            'total_sequences': len(sequences)
        }
    
    def _extract_search_keywords(self, search_queries: List[str]) -> List[str]:
        """Extract common keywords from search queries"""
        all_words = []
        for query in search_queries:
            words = re.findall(r'\b\w+\b', query.lower())
            all_words.extend([word for word in words if len(word) > 2])
        
        word_counts = Counter(all_words)
        return [word for word, count in word_counts.most_common(20)]
    
    def _analyze_search_intent(self, search_queries: List[str]) -> Dict:
        """Analyze search intent patterns"""
        intent_patterns = {
            'specific_titles': 0,
            'genre_searches': 0,
            'actor_searches': 0,
            'year_searches': 0,
            'language_searches': 0
        }
        
        genre_keywords = ['action', 'comedy', 'drama', 'horror', 'romance', 'thriller', 'sci-fi', 'fantasy']
        language_keywords = ['hindi', 'telugu', 'tamil', 'english', 'malayalam', 'kannada']
        
        for query in search_queries:
            query_lower = query.lower()
            
            if any(genre in query_lower for genre in genre_keywords):
                intent_patterns['genre_searches'] += 1
            
            if any(lang in query_lower for lang in language_keywords):
                intent_patterns['language_searches'] += 1
            
            if re.search(r'\b(19|20)\d{2}\b', query):
                intent_patterns['year_searches'] += 1
            
            # Default to specific title search
            if not any([
                any(genre in query_lower for genre in genre_keywords),
                any(lang in query_lower for lang in language_keywords),
                re.search(r'\b(19|20)\d{2}\b', query)
            ]):
                intent_patterns['specific_titles'] += 1
        
        return intent_patterns
    
    def _analyze_search_timing(self, search_timestamps: List[datetime]) -> Dict:
        """Analyze when user searches for content"""
        if not search_timestamps:
            return {}
        
        hours = [ts.hour for ts in search_timestamps]
        days = [ts.weekday() for ts in search_timestamps]
        
        return {
            'peak_search_hours': [hour for hour, count in Counter(hours).most_common(3)],
            'preferred_search_days': [day for day, count in Counter(days).most_common(3)]
        }
    
    def _serialize_profile(self, profile: Dict) -> Dict:
        """Convert defaultdicts to regular dicts for JSON serialization"""
        serialized = {}
        for key, value in profile.items():
            if isinstance(value, defaultdict):
                serialized[key] = dict(value)
            elif isinstance(value, dict):
                serialized[key] = {k: dict(v) if isinstance(v, defaultdict) else v 
                                 for k, v in value.items()}
            else:
                serialized[key] = value
        return serialized
    
    def _get_default_profile(self, user_id: int) -> Dict[str, Any]:
        """Return default profile for new users"""
        return {
            'user_id': user_id,
            'interaction_count': 0,
            'profile_strength': 0.0,
            'genre_preferences': {'action': 0.2, 'drama': 0.2, 'comedy': 0.15, 'thriller': 0.15, 'romance': 0.1, 'sci-fi': 0.1, 'horror': 0.08},
            'language_preferences': {'english': 0.4, 'telugu': 0.3, 'hindi': 0.2, 'tamil': 0.1},
            'content_type_preferences': {'movie': 0.5, 'tv': 0.3, 'anime': 0.2},
            'quality_preferences': {'average_rating': 7.0, 'high_quality_bias': 0.6},
            'runtime_preferences': {'preferred_range': [90, 150], 'average': 120},
            'release_period_preferences': {'2020s': 0.4, '2010s': 0.3, '2000s': 0.2, '1990s': 0.1},
            'viewing_patterns': {},
            'search_behavior': {},
            'sequence_patterns': {},
            'similarity_clusters': [],
            'cast_crew_preferences': {},
            'director_preferences': {},
            'exploration_tendency': 0.5,
            'current_mode': 'discovery_mode',
            'affinity_scores': {},
            'confidence_score': 0.0,
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def _load_stored_profile(self, user_pref: UserPreference) -> Dict[str, Any]:
        """Load stored profile from database"""
        profile = {
            'user_id': user_pref.user_id,
            'profile_strength': user_pref.profile_strength,
            'confidence_score': user_pref.confidence_score,
            'exploration_tendency': user_pref.exploration_tendency,
            'current_mode': user_pref.current_mode,
            'last_updated': user_pref.updated_at.isoformat()
        }
        
        # Load JSON fields
        json_fields = [
            'genre_preferences', 'language_preferences', 'content_type_preferences',
            'quality_preferences', 'runtime_preferences', 'viewing_patterns',
            'search_patterns', 'sequence_patterns', 'cast_crew_preferences',
            'franchise_preferences', 'mood_preferences'
        ]
        
        for field in json_fields:
            try:
                value = getattr(user_pref, field)
                if value:
                    profile[field] = json.loads(value)
                else:
                    profile[field] = {}
            except (json.JSONDecodeError, TypeError, AttributeError):
                profile[field] = {}
        
        return profile
    
    def _store_profile(self, user_id: int, profile: Dict) -> None:
        """Store or update user profile in database"""
        try:
            user_pref = UserPreference.query.filter_by(user_id=user_id).first()
            
            if not user_pref:
                user_pref = UserPreference(user_id=user_id)
                db.session.add(user_pref)
            
            # Update fields
            user_pref.profile_strength = profile.get('profile_strength', 0.0)
            user_pref.confidence_score = profile.get('confidence_score', 0.0)
            user_pref.exploration_tendency = profile.get('exploration_tendency', 0.5)
            user_pref.current_mode = profile.get('current_mode', 'discovery_mode')
            user_pref.last_calculated = datetime.utcnow()
            
            # Store JSON fields
            json_fields = {
                'genre_preferences': profile.get('genre_preferences', {}),
                'language_preferences': profile.get('language_preferences', {}),
                'content_type_preferences': profile.get('content_type_preferences', {}),
                'quality_preferences': profile.get('quality_preferences', {}),
                'runtime_preferences': profile.get('runtime_preferences', {}),
                'viewing_patterns': profile.get('viewing_patterns', {}),
                'search_patterns': profile.get('search_behavior', {}),
                'sequence_patterns': profile.get('sequence_patterns', {}),
                'cast_crew_preferences': profile.get('cast_crew_preferences', {}),
                'franchise_preferences': profile.get('franchise_preferences', {}),
                'mood_preferences': profile.get('mood_indicators', {})
            }
            
            for field, value in json_fields.items():
                setattr(user_pref, field, json.dumps(value))
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error storing user profile: {e}")
            db.session.rollback()

class NeuralCollaborativeFiltering:
    """Advanced neural collaborative filtering with deep embeddings"""
    
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.user_mapping = {}
        self.item_mapping = {}
        self.model_trained = False
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for neural collaborative filtering"""
        try:
            interactions = UserInteraction.query.filter(
                UserInteraction.interaction_type.in_(['rating', 'favorite', 'like', 'view', 'watchlist'])
            ).all()
            
            if not interactions:
                return np.array([]), np.array([]), np.array([])
            
            users = list(set([i.user_id for i in interactions]))
            items = list(set([i.content_id for i in interactions]))
            
            self.user_mapping = {user_id: idx for idx, user_id in enumerate(users)}
            self.item_mapping = {item_id: idx for idx, item_id in enumerate(items)}
            
            user_ids = []
            item_ids = []
            ratings = []
            
            for interaction in interactions:
                user_idx = self.user_mapping[interaction.user_id]
                item_idx = self.item_mapping[interaction.content_id]
                
                # Convert interaction to implicit rating
                if interaction.rating:
                    rating = interaction.rating
                elif interaction.interaction_type == 'favorite':
                    rating = 5.0
                elif interaction.interaction_type == 'like':
                    rating = 4.0
                elif interaction.interaction_type == 'watchlist':
                    rating = 3.5
                elif interaction.interaction_type == 'view':
                    rating = 3.0
                else:
                    rating = 2.5
                
                user_ids.append(user_idx)
                item_ids.append(item_idx)
                ratings.append(rating)
            
            return np.array(user_ids), np.array(item_ids), np.array(ratings)
            
        except Exception as e:
            logger.error(f"Error preparing NCF data: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def train_embeddings(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray):
        """Train neural collaborative filtering embeddings"""
        try:
            if len(user_ids) == 0:
                return False
            
            n_users = len(self.user_mapping)
            n_items = len(self.item_mapping)
            
            # Create rating matrix
            rating_matrix = np.zeros((n_users, n_items))
            for user_idx, item_idx, rating in zip(user_ids, item_ids, ratings):
                rating_matrix[user_idx, item_idx] = rating
            
            # Use SVD for matrix factorization
            svd = TruncatedSVD(n_components=min(self.embedding_dim, min(n_users, n_items) - 1))
            
            # Handle sparse matrix
            mask = rating_matrix > 0
            filled_matrix = rating_matrix.copy()
            if np.any(mask):
                mean_rating = np.mean(rating_matrix[mask])
                filled_matrix[~mask] = mean_rating
            else:
                filled_matrix[~mask] = 3.0
            
            # Fit SVD
            user_factors = svd.fit_transform(filled_matrix)
            item_factors = svd.components_.T
            
            # Store embeddings
            for user_id, user_idx in self.user_mapping.items():
                self.user_embeddings[user_id] = user_factors[user_idx]
            
            for item_id, item_idx in self.item_mapping.items():
                self.item_embeddings[item_id] = item_factors[item_idx]
            
            self.model_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error training NCF embeddings: {e}")
            return False
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 20) -> List[Dict]:
        """Get recommendations using neural collaborative filtering"""
        try:
            if not self.model_trained or user_id not in self.user_embeddings:
                return []
            
            user_embedding = self.user_embeddings[user_id]
            
            # Calculate scores for all items
            item_scores = []
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_items = set([i.content_id for i in user_interactions])
            
            for item_id, item_embedding in self.item_embeddings.items():
                if item_id not in interacted_items:
                    # Calculate cosine similarity
                    score = np.dot(user_embedding, item_embedding) / (
                        np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding) + 1e-8
                    )
                    item_scores.append((item_id, float(score)))
            
            # Sort by score and get top recommendations
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for item_id, score in item_scores[:n_recommendations]:
                content = Content.query.get(item_id)
                if content:
                    recommendations.append({
                        'content_id': item_id,
                        'score': score,
                        'method': 'neural_collaborative',
                        'content': content
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting NCF recommendations: {e}")
            return []

class AdvancedContentEmbeddings:
    """Advanced content embeddings with semantic understanding"""
    
    def __init__(self):
        self.content_embeddings = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.content_features = None
        self.content_mapping = {}
        
    def build_content_embeddings(self):
        """Build comprehensive content embeddings"""
        try:
            contents = Content.query.all()
            if not contents:
                return False
            
            content_descriptions = []
            content_metadata = []
            content_ids = []
            
            for content in contents:
                # Text description
                description_parts = []
                
                if content.title:
                    description_parts.append(content.title.lower())
                if content.original_title and content.original_title != content.title:
                    description_parts.append(content.original_title.lower())
                if content.overview:
                    description_parts.append(content.overview.lower())
                
                # Add genres
                try:
                    genres = json.loads(content.genres or '[]')
                    description_parts.extend([g.lower() for g in genres])
                except (json.JSONDecodeError, TypeError):
                    genres = []
                
                # Add languages
                try:
                    languages = json.loads(content.languages or '[]')
                    description_parts.extend([l.lower() for l in languages])
                except (json.JSONDecodeError, TypeError):
                    languages = []
                
                # Add content type
                description_parts.append(content.content_type)
                
                # Create metadata features
                metadata = {
                    'rating': content.rating or 0,
                    'popularity': content.popularity or 0,
                    'runtime': content.runtime or 120,
                    'release_year': content.release_date.year if content.release_date else 2020,
                    'vote_count': content.vote_count or 0,
                    'is_trending': int(content.is_trending or False),
                    'is_new_release': int(content.is_new_release or False),
                    'genre_count': len(genres),
                    'language_count': len(languages)
                }
                
                content_descriptions.append(' '.join(description_parts))
                content_metadata.append(metadata)
                content_ids.append(content.id)
            
            # Create TF-IDF features
            tfidf_features = self.tfidf_vectorizer.fit_transform(content_descriptions)
            
            # Normalize metadata features
            metadata_df = pd.DataFrame(content_metadata)
            scaler = StandardScaler()
            metadata_features = scaler.fit_transform(metadata_df)
            
            # Combine features
            combined_features = np.hstack([tfidf_features.toarray(), metadata_features])
            
            # Store embeddings
            for i, content_id in enumerate(content_ids):
                self.content_embeddings[content_id] = combined_features[i]
            
            self.content_mapping = {content_id: idx for idx, content_id in enumerate(content_ids)}
            self.content_features = combined_features
            
            return True
            
        except Exception as e:
            logger.error(f"Error building content embeddings: {e}")
            return False
    
    def get_content_similarities(self, content_id: int, n_similar: int = 20) -> List[Tuple[int, float]]:
        """Get similar content using advanced embeddings"""
        try:
            if content_id not in self.content_embeddings:
                return []
            
            content_embedding = self.content_embeddings[content_id]
            similarities = []
            
            for other_id, other_embedding in self.content_embeddings.items():
                if other_id != content_id:
                    # Calculate weighted similarity
                    cosine_sim = cosine_similarity([content_embedding], [other_embedding])[0][0]
                    
                    # Add content-specific bonuses
                    content = Content.query.get(content_id)
                    other_content = Content.query.get(other_id)
                    
                    if content and other_content:
                        bonus = 0
                        
                        # Same content type bonus
                        if content.content_type == other_content.content_type:
                            bonus += 0.1
                        
                        # Rating similarity bonus
                        if content.rating and other_content.rating:
                            rating_diff = abs(content.rating - other_content.rating)
                            if rating_diff <= 1.0:
                                bonus += 0.05
                        
                        # Release year proximity bonus
                        if content.release_date and other_content.release_date:
                            year_diff = abs(content.release_date.year - other_content.release_date.year)
                            if year_diff <= 3:
                                bonus += 0.03
                        
                        final_similarity = cosine_sim + bonus
                        similarities.append((other_id, float(final_similarity)))
            
            # Sort by similarity and return top N
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:n_similar]
            
        except Exception as e:
            logger.error(f"Error calculating content similarities: {e}")
            return []

class SequenceAwareRecommender:
    """Advanced sequence-aware recommendation engine"""
    
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.transition_matrices = defaultdict(lambda: defaultdict(float))
        
    def build_sequence_models(self, user_id: int):
        """Build advanced sequence models for user"""
        try:
            interactions = UserInteraction.query.filter_by(user_id=user_id).order_by(
                UserInteraction.timestamp
            ).all()
            
            if len(interactions) < 3:
                return
            
            # Get content for interactions
            content_ids = [i.content_id for i in interactions]
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_map = {c.id: c for c in contents}
            
            # Build transition models
            for i in range(len(interactions) - 1):
                current_content = content_map.get(interactions[i].content_id)
                next_content = content_map.get(interactions[i + 1].content_id)
                
                if current_content and next_content:
                    # Genre transitions
                    try:
                        current_genres = json.loads(current_content.genres or '[]')
                        next_genres = json.loads(next_content.genres or '[]')
                        
                        if current_genres and next_genres:
                            current_genre = current_genres[0].lower()
                            next_genre = next_genres[0].lower()
                            self.transition_matrices['genre'][f"{current_genre}->{next_genre}"] += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
                    
                    # Content type transitions
                    self.transition_matrices['content_type'][
                        f"{current_content.content_type}->{next_content.content_type}"
                    ] += 1
                    
                    # Quality transitions
                    if current_content.rating and next_content.rating:
                        current_tier = self._get_rating_tier(current_content.rating)
                        next_tier = self._get_rating_tier(next_content.rating)
                        self.transition_matrices['quality'][f"{current_tier}->{next_tier}"] += 1
            
        except Exception as e:
            logger.error(f"Error building sequence models: {e}")
    
    def get_sequence_predictions(self, user_id: int, recent_interactions: List[int], 
                               n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations based on sequence patterns"""
        try:
            self.build_sequence_models(user_id)
            
            if not recent_interactions:
                recent = UserInteraction.query.filter_by(user_id=user_id).order_by(
                    UserInteraction.timestamp.desc()
                ).limit(3).all()
                recent_interactions = [i.content_id for i in recent]
            
            if not recent_interactions:
                return []
            
            # Get recent content details
            recent_contents = Content.query.filter(Content.id.in_(recent_interactions)).all()
            
            predictions = defaultdict(float)
            
            for content in recent_contents:
                # Predict based on transitions
                try:
                    genres = json.loads(content.genres or '[]')
                    if genres:
                        primary_genre = genres[0].lower()
                        
                        # Genre-based predictions
                        for transition, count in self.transition_matrices['genre'].items():
                            if transition.startswith(f"{primary_genre}->"):
                                next_genre = transition.split('->')[1]
                                predictions[f"genre_{next_genre}"] += count * 0.3
                    
                    # Content type predictions
                    for transition, count in self.transition_matrices['content_type'].items():
                        if transition.startswith(f"{content.content_type}->"):
                            next_type = transition.split('->')[1]
                            predictions[f"type_{next_type}"] += count * 0.2
                    
                except (json.JSONDecodeError, TypeError):
                    continue
            
            # Convert predictions to recommendations
            recommendations = []
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_content = set([i.content_id for i in user_interactions])
            
            for prediction_key, score in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                if len(recommendations) >= n_recommendations:
                    break
                
                # Find matching content
                prediction_type, value = prediction_key.split('_', 1)
                
                query = Content.query.filter(~Content.id.in_(interacted_content))
                
                if prediction_type == 'genre':
                    query = query.filter(Content.genres.contains(value.title()))
                elif prediction_type == 'type':
                    query = query.filter(Content.content_type == value)
                
                matching_content = query.order_by(Content.popularity.desc()).limit(2).all()
                
                for content in matching_content:
                    if content.id not in [r['content_id'] for r in recommendations]:
                        recommendations.append({
                            'content_id': content.id,
                            'score': score,
                            'method': 'sequence_aware',
                            'content': content
                        })
                        break
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting sequence predictions: {e}")
            return []
    
    def _get_rating_tier(self, rating: float) -> str:
        """Get rating tier for content"""
        if rating >= 7.0:
            return 'high'
        elif rating >= 5.0:
            return 'medium'
        else:
            return 'low'

class HyperPersonalizedEngine:
    """Ultra-advanced personalized recommendation engine"""
    
    def __init__(self):
        self.profiler = EnhancedUserProfiler()
        self.ncf_engine = NeuralCollaborativeFiltering()
        self.content_embeddings = AdvancedContentEmbeddings()
        self.sequence_engine = SequenceAwareRecommender()
        
        # Algorithm weights based on user mode
        self.mode_weights = {
            'discovery_mode': {
                'exploration_boost': 0.35,
                'content_embeddings': 0.25,
                'neural_collaborative': 0.20,
                'sequence_aware': 0.10,
                'profile_based': 0.10
            },
            'comfort_mode': {
                'profile_based': 0.35,
                'sequence_aware': 0.25,
                'neural_collaborative': 0.20,
                'content_embeddings': 0.15,
                'exploration_boost': 0.05
            },
            'binge_mode': {
                'sequence_aware': 0.40,
                'content_embeddings': 0.25,
                'neural_collaborative': 0.20,
                'profile_based': 0.10,
                'exploration_boost': 0.05
            },
            'selective_mode': {
                'neural_collaborative': 0.35,
                'profile_based': 0.30,
                'content_embeddings': 0.20,
                'sequence_aware': 0.10,
                'exploration_boost': 0.05
            }
        }
    
    def get_hyper_personalized_recommendations(self, user_id: int, 
                                             content_type: Optional[str] = None,
                                             context: Optional[Dict] = None, 
                                             n_recommendations: int = 20) -> Dict:
        """Get ultra-personalized recommendations with maximum accuracy"""
        try:
            # Build comprehensive user profile
            user_profile = self.profiler.build_comprehensive_profile(user_id)
            user_mode = user_profile.get('current_mode', 'discovery_mode')
            confidence_level = user_profile.get('confidence_score', 0.0)
            
            # Initialize recommendation sources
            all_recommendations = []
            algorithm_performance = {}
            
            # Get weights for current user mode
            weights = self.mode_weights.get(user_mode, self.mode_weights['comfort_mode'])
            
            # 1. Neural Collaborative Filtering
            try:
                if not self.ncf_engine.model_trained:
                    user_ids, item_ids, ratings = self.ncf_engine.prepare_data()
                    if len(user_ids) > 0:
                        self.ncf_engine.train_embeddings(user_ids, item_ids, ratings)
                
                ncf_recs = self.ncf_engine.get_user_recommendations(user_id, n_recommendations * 2)
                all_recommendations.extend(ncf_recs)
                algorithm_performance['neural_collaborative'] = len(ncf_recs)
            except Exception as e:
                logger.warning(f"NCF failed: {e}")
                algorithm_performance['neural_collaborative'] = 0
            
            # 2. Content Embeddings
            try:
                if not self.content_embeddings.content_embeddings:
                    self.content_embeddings.build_content_embeddings()
                
                content_recs = self._get_content_embedding_recommendations(
                    user_id, user_profile, n_recommendations * 2
                )
                all_recommendations.extend(content_recs)
                algorithm_performance['content_embeddings'] = len(content_recs)
            except Exception as e:
                logger.warning(f"Content embeddings failed: {e}")
                algorithm_performance['content_embeddings'] = 0
            
            # 3. Sequence-Aware Recommendations
            try:
                sequence_recs = self.sequence_engine.get_sequence_predictions(
                    user_id, None, n_recommendations
                )
                all_recommendations.extend(sequence_recs)
                algorithm_performance['sequence_aware'] = len(sequence_recs)
            except Exception as e:
                logger.warning(f"Sequence-aware failed: {e}")
                algorithm_performance['sequence_aware'] = 0
            
            # 4. Profile-Based Recommendations
            try:
                profile_recs = self._get_profile_based_recommendations(
                    user_profile, content_type, n_recommendations
                )
                all_recommendations.extend(profile_recs)
                algorithm_performance['profile_based'] = len(profile_recs)
            except Exception as e:
                logger.warning(f"Profile-based failed: {e}")
                algorithm_performance['profile_based'] = 0
            
            # 5. Exploration Recommendations
            try:
                exploration_recs = self._get_exploration_recommendations(
                    user_profile, content_type, n_recommendations // 4
                )
                all_recommendations.extend(exploration_recs)
                algorithm_performance['exploration_boost'] = len(exploration_recs)
            except Exception as e:
                logger.warning(f"Exploration failed: {e}")
                algorithm_performance['exploration_boost'] = 0
            
            # 6. Combine and rank with advanced fusion
            final_recommendations = self._advanced_recommendation_fusion(
                all_recommendations, user_profile, weights
            )
            
            # 7. Apply contextual filtering
            if context:
                final_recommendations = self._apply_contextual_filtering(
                    final_recommendations, context, user_profile
                )
            
            # 8. Apply diversity
            final_recommendations = self._apply_diversity(
                final_recommendations, user_profile
            )
            
            # 9. Filter by content type if specified
            if content_type:
                final_recommendations = [
                    rec for rec in final_recommendations 
                    if rec['content'].content_type == content_type
                ]
            
            # 10. Format for API response
            formatted_recommendations = self._format_recommendations(
                final_recommendations[:n_recommendations], user_profile
            )
            
            # Track recommendation feedback
            self._track_recommendations(user_id, formatted_recommendations, user_mode)
            
            return {
                'recommendations': formatted_recommendations,
                'user_profile_insights': {
                    'confidence_level': confidence_level,
                    'profile_strength': user_profile.get('profile_strength', 0),
                    'user_mode': user_mode,
                    'top_genres': list(user_profile.get('genre_preferences', {}).keys())[:5],
                    'preferred_languages': list(user_profile.get('language_preferences', {}).keys())[:3],
                    'exploration_tendency': user_profile.get('exploration_tendency', 0.5),
                    'trending_interests': user_profile.get('trending_interests', [])
                },
                'recommendation_metadata': {
                    'total_recommendations': len(formatted_recommendations),
                    'algorithm_performance': algorithm_performance,
                    'personalization_strength': min(confidence_level * 100, 100),
                    'recommendation_accuracy_estimate': self._estimate_accuracy(
                        user_profile, algorithm_performance
                    ),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in hyper-personalized recommendations: {e}")
            return {
                'recommendations': [],
                'user_profile_insights': {},
                'recommendation_metadata': {'error': str(e)}
            }
    
    def _get_content_embedding_recommendations(self, user_id: int, user_profile: Dict, 
                                             n_recommendations: int) -> List[Dict]:
        """Get recommendations using content embeddings"""
        try:
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not interactions:
                return []
            
            # Get content embeddings for user's interacted content
            content_ids = [i.content_id for i in interactions]
            user_content_embeddings = []
            interaction_weights = []
            
            for interaction in interactions:
                if interaction.content_id in self.content_embeddings.content_embeddings:
                    embedding = self.content_embeddings.content_embeddings[interaction.content_id]
                    user_content_embeddings.append(embedding)
                    
                    # Weight by interaction type and rating
                    weight = self.profiler.interaction_weights.get(
                        interaction.interaction_type, {'weight': 0.1}
                    )['weight']
                    
                    if interaction.rating:
                        weight *= (interaction.rating / 5.0)
                    
                    interaction_weights.append(weight)
            
            if not user_content_embeddings:
                return []
            
            # Create weighted user profile embedding
            user_content_embeddings = np.array(user_content_embeddings)
            interaction_weights = np.array(interaction_weights)
            interaction_weights = interaction_weights / np.sum(interaction_weights)
            
            user_profile_vector = np.average(user_content_embeddings, axis=0, weights=interaction_weights)
            
            # Find similar content
            recommendations = []
            interacted_content = set(content_ids)
            
            similarities = []
            for content_id, content_embedding in self.content_embeddings.content_embeddings.items():
                if content_id not in interacted_content:
                    similarity = cosine_similarity([user_profile_vector], [content_embedding])[0][0]
                    similarities.append((content_id, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            for content_id, similarity in similarities[:n_recommendations]:
                content = Content.query.get(content_id)
                if content:
                    recommendations.append({
                        'content_id': content_id,
                        'score': float(similarity),
                        'method': 'content_embeddings',
                        'content': content
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in content embedding recommendations: {e}")
            return []
    
    def _get_profile_based_recommendations(self, user_profile: Dict, 
                                         content_type: Optional[str], 
                                         n_recommendations: int) -> List[Dict]:
        """Get recommendations based on user profile"""
        try:
            recommendations = []
            
            # Get preferences
            genre_prefs = user_profile.get('genre_preferences', {})
            lang_prefs = user_profile.get('language_preferences', {})
            quality_prefs = user_profile.get('quality_preferences', {})
            
            # Build query
            query = Content.query
            
            if content_type:
                query = query.filter(Content.content_type == content_type)
            
            # Apply quality filter
            min_rating = quality_prefs.get('average_rating', 6.0) - 1.0
            if min_rating > 0:
                query = query.filter(Content.rating >= min_rating)
            
            # Get candidates
            candidates = query.order_by(Content.popularity.desc()).limit(n_recommendations * 5).all()
            
            # Score candidates
            scored_candidates = []
            
            for content in candidates:
                score = self._calculate_profile_match_score(content, user_profile)
                if score > 0.3:
                    scored_candidates.append({
                        'content_id': content.id,
                        'score': score,
                        'method': 'profile_based',
                        'content': content
                    })
            
            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
            return scored_candidates[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in profile-based recommendations: {e}")
            return []
    
    def _get_exploration_recommendations(self, user_profile: Dict, 
                                       content_type: Optional[str], 
                                       n_recommendations: int) -> List[Dict]:
        """Get exploration recommendations"""
        try:
            exploration_tendency = user_profile.get('exploration_tendency', 0.5)
            
            if exploration_tendency < 0.3:
                return []
            
            # Get user's interaction history
            user_id = user_profile['user_id']
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_genres = set()
            interacted_content_ids = set([i.content_id for i in interactions])
            
            # Extract interacted genres
            content_ids = [i.content_id for i in interactions]
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            
            for content in contents:
                try:
                    genres = json.loads(content.genres or '[]')
                    interacted_genres.update([g.lower() for g in genres])
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Find unexplored content
            query = Content.query.filter(~Content.id.in_(interacted_content_ids))
            
            if content_type:
                query = query.filter(Content.content_type == content_type)
            
            # Get high-quality unexplored content
            exploration_candidates = query.filter(
                Content.rating >= 7.0,
                Content.vote_count >= 100
            ).order_by(Content.popularity.desc()).limit(n_recommendations * 3).all()
            
            recommendations = []
            
            for content in exploration_candidates:
                exploration_score = self._calculate_exploration_score(
                    content, interacted_genres
                )
                
                if exploration_score > 0.5:
                    recommendations.append({
                        'content_id': content.id,
                        'score': exploration_score,
                        'method': 'exploration_boost',
                        'content': content
                    })
            
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in exploration recommendations: {e}")
            return []
    
    def _calculate_profile_match_score(self, content: Content, user_profile: Dict) -> float:
        """Calculate how well content matches user profile"""
        try:
            score = 0.0
            
            # Genre matching
            try:
                content_genres = json.loads(content.genres or '[]')
                user_genres = user_profile.get('genre_preferences', {})
                
                for genre in content_genres:
                    genre_pref = user_genres.get(genre.lower(), 0)
                    score += genre_pref * 0.4
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Language matching
            try:
                content_languages = json.loads(content.languages or '[]')
                user_languages = user_profile.get('language_preferences', {})
                
                for lang in content_languages:
                    lang_pref = user_languages.get(lang.lower(), 0)
                    score += lang_pref * 0.3
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Content type matching
            content_type_prefs = user_profile.get('content_type_preferences', {})
            content_type_pref = content_type_prefs.get(content.content_type, 0)
            score += content_type_pref * 0.2
            
            # Quality matching
            quality_prefs = user_profile.get('quality_preferences', {})
            if content.rating and quality_prefs.get('average_rating'):
                quality_diff = abs(content.rating - quality_prefs['average_rating'])
                quality_score = max(0, 1 - (quality_diff / 5.0))
                score += quality_score * 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating profile match score: {e}")
            return 0.0
    
    def _calculate_exploration_score(self, content: Content, interacted_genres: Set[str]) -> float:
        """Calculate exploration score for content"""
        try:
            score = 0.0
            
            # Genre novelty
            try:
                content_genres = json.loads(content.genres or '[]')
                new_genres = [g.lower() for g in content_genres if g.lower() not in interacted_genres]
                genre_novelty = len(new_genres) / max(len(content_genres), 1) if content_genres else 0
                score += genre_novelty * 0.5
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Quality bonus
            if content.rating and content.rating >= 8.0:
                score += 0.3
            elif content.rating and content.rating >= 7.5:
                score += 0.2
            
            # Popularity consideration
            if content.popularity and content.popularity >= 100:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating exploration score: {e}")
            return 0.0
    
    def _advanced_recommendation_fusion(self, all_recommendations: List[Dict], 
                                      user_profile: Dict, weights: Dict) -> List[Dict]:
        """Advanced fusion of recommendations from multiple algorithms"""
        try:
            # Group by content_id
            content_scores = defaultdict(lambda: {'score': 0, 'sources': [], 'content': None})
            
            for rec in all_recommendations:
                content_id = rec['content_id']
                method = rec.get('method', 'unknown')
                score = rec['score']
                
                # Apply method weight
                weighted_score = score * weights.get(method, 0.1)
                
                content_scores[content_id]['score'] += weighted_score
                content_scores[content_id]['sources'].append(method)
                content_scores[content_id]['content'] = rec['content']
            
            # Normalize and apply bonuses
            fused_recommendations = []
            
            for content_id, data in content_scores.items():
                # Multi-method consensus bonus
                if len(data['sources']) >= 3:
                    data['score'] *= 1.15
                elif len(data['sources']) >= 2:
                    data['score'] *= 1.08
                
                # Add content-specific bonuses
                content = data['content']
                if content:
                    # Fresh content bonus
                    if content.is_new_release:
                        data['score'] *= 1.05
                    
                    # Trending bonus
                    if content.is_trending:
                        data['score'] *= 1.03
                    
                    # High quality bonus
                    if content.rating and content.rating >= 8.0:
                        data['score'] *= 1.02
                
                fused_recommendations.append({
                    'content_id': content_id,
                    'score': data['score'],
                    'content': data['content'],
                    'sources': data['sources']
                })
            
            # Sort by score
            fused_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return fused_recommendations
            
        except Exception as e:
            logger.error(f"Error in recommendation fusion: {e}")
            return []
    
    def _apply_contextual_filtering(self, recommendations: List[Dict], 
                                  context: Dict, user_profile: Dict) -> List[Dict]:
        """Apply contextual filtering to recommendations"""
        try:
            current_hour = datetime.utcnow().hour
            current_day = datetime.utcnow().weekday()
            
            # Get user's temporal patterns
            viewing_patterns = user_profile.get('viewing_patterns', {})
            peak_hours = viewing_patterns.get('peak_hours', [])
            preferred_days = viewing_patterns.get('preferred_days', [])
            
            for rec in recommendations:
                content = rec['content']
                contextual_boost = 0.0
                
                # Time-based adjustments
                if current_hour in peak_hours:
                    contextual_boost += 0.05
                
                if current_day in preferred_days:
                    contextual_boost += 0.03
                
                # Weekend/weekday patterns
                is_weekend = current_day >= 5
                if is_weekend and content.runtime and content.runtime >= 120:
                    contextual_boost += 0.02  # Longer content for weekends
                elif not is_weekend and content.runtime and content.runtime <= 100:
                    contextual_boost += 0.02  # Shorter content for weekdays
                
                # Apply boost
                rec['score'] += rec['score'] * contextual_boost
            
            # Re-sort
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error applying contextual filtering: {e}")
            return recommendations
    
    def _apply_diversity(self, recommendations: List[Dict], user_profile: Dict) -> List[Dict]:
        """Apply diversity to recommendations"""
        try:
            diversity_preference = user_profile.get('exploration_tendency', 0.5)
            
            if diversity_preference < 0.3:
                return recommendations
            
            # Ensure diversity
            seen_genres = set()
            seen_content_types = set()
            diverse_recommendations = []
            other_recommendations = []
            
            for rec in recommendations:
                content = rec['content']
                
                try:
                    genres = json.loads(content.genres or '[]')
                    primary_genre = genres[0].lower() if genres else 'unknown'
                    
                    is_diverse = (
                        primary_genre not in seen_genres or
                        content.content_type not in seen_content_types
                    )
                    
                    if is_diverse and len(diverse_recommendations) < len(recommendations) * 0.7:
                        seen_genres.add(primary_genre)
                        seen_content_types.add(content.content_type)
                        diverse_recommendations.append(rec)
                    else:
                        other_recommendations.append(rec)
                
                except (json.JSONDecodeError, TypeError):
                    other_recommendations.append(rec)
            
            # Combine diverse and other recommendations
            final_recommendations = diverse_recommendations + other_recommendations
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error applying diversity: {e}")
            return recommendations
    
    def _format_recommendations(self, recommendations: List[Dict], user_profile: Dict) -> List[Dict]:
        """Format recommendations for API response"""
        try:
            formatted_recs = []
            
            for rec in recommendations:
                content = rec['content']
                
                # Ensure slug exists
                if not content.slug:
                    try:
                        content.ensure_slug()
                        db.session.commit()
                    except Exception:
                        content.slug = f"content-{content.id}"
                
                # Prepare URLs
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                poster_url = None
                if content.poster_path:
                    if content.poster_path.startswith('http'):
                        poster_url = content.poster_path
                    else:
                        poster_url = f"https://image.tmdb.org/t/p/w300{content.poster_path}"
                
                # Generate recommendation reason
                reason = self._generate_recommendation_reason(rec, user_profile)
                
                # Predict user rating
                predicted_rating = self._predict_user_rating(content, user_profile)
                
                formatted_rec = {
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'runtime': content.runtime,
                    'poster_path': poster_url,
                    'overview': content.overview[:200] + '...' if content.overview and len(content.overview) > 200 else content.overview,
                    'youtube_trailer': youtube_url,
                    'recommendation_score': round(rec['score'], 4),
                    'recommendation_reason': reason,
                    'predicted_rating': predicted_rating,
                    'methods_used': rec.get('sources', ['hybrid']),
                    'is_trending': content.is_trending,
                    'is_new_release': content.is_new_release
                }
                
                formatted_recs.append(formatted_rec)
            
            return formatted_recs
            
        except Exception as e:
            logger.error(f"Error formatting recommendations: {e}")
            return []
    
    def _generate_recommendation_reason(self, recommendation: Dict, user_profile: Dict) -> str:
        """Generate personalized recommendation reason"""
        try:
            content = recommendation['content']
            methods = recommendation.get('sources', [])
            score = recommendation.get('score', 0)
            
            reasons = []
            
            # Method-based reasons
            if 'neural_collaborative' in methods:
                reasons.append("users like you loved this")
            
            if 'sequence_aware' in methods:
                reasons.append("perfectly follows your viewing pattern")
            
            if 'content_embeddings' in methods:
                reasons.append("matches your content preferences")
            
            if 'exploration_boost' in methods:
                reasons.append("something new to discover")
            
            # Content-specific reasons
            try:
                content_genres = json.loads(content.genres or '[]')
                user_genres = user_profile.get('genre_preferences', {})
                
                for genre in content_genres[:2]:
                    if user_genres.get(genre.lower(), 0) > 0.3:
                        reasons.append(f"features {genre.lower()}")
                        break
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Quality reasons
            if content.rating and content.rating >= 8.0:
                reasons.append("highly rated")
            
            # Trending/new
            if content.is_trending:
                reasons.append("trending now")
            if content.is_new_release:
                reasons.append("just released")
            
            # Combine reasons
            if reasons:
                if score > 0.8:
                    prefix = "Perfect match"
                elif score > 0.6:
                    prefix = "Great choice"
                else:
                    prefix = "Recommended"
                
                return f"{prefix}: {', '.join(reasons[:2])}"
            else:
                return "Recommended based on your preferences"
                
        except Exception as e:
            logger.error(f"Error generating recommendation reason: {e}")
            return "Recommended for you"
    
    def _predict_user_rating(self, content: Content, user_profile: Dict) -> float:
        """Predict how user would rate this content"""
        try:
            predicted_rating = 3.0  # Base rating
            
            # Get affinity scores
            affinity_scores = user_profile.get('affinity_scores', {})
            
            # Genre match
            try:
                genres = json.loads(content.genres or '[]')
                genre_affinities = affinity_scores.get('genre_affinity', {})
                
                if genres and genre_affinities:
                    max_affinity = max([genre_affinities.get(g.lower(), 0) for g in genres])
                    predicted_rating += max_affinity * 2
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Quality alignment
            quality_prefs = user_profile.get('quality_preferences', {})
            user_avg = quality_prefs.get('average_rating', 7.0)
            
            if content.rating:
                if content.rating >= user_avg:
                    predicted_rating += 0.5
                if content.rating >= 8:
                    predicted_rating += 0.3
            
            # Language match
            try:
                languages = json.loads(content.languages or '[]')
                lang_affinities = affinity_scores.get('language_affinity', {})
                
                if languages and lang_affinities:
                    max_lang_affinity = max([lang_affinities.get(l.lower(), 0) for l in languages])
                    predicted_rating += max_lang_affinity
            except (json.JSONDecodeError, TypeError):
                pass
            
            return min(round(predicted_rating, 1), 5.0)
            
        except Exception as e:
            logger.error(f"Error predicting user rating: {e}")
            return 3.5
    
    def _track_recommendations(self, user_id: int, recommendations: List[Dict], user_mode: str) -> None:
        """Track recommendations for performance analysis"""
        try:
            for idx, rec in enumerate(recommendations[:10]):  # Track top 10
                feedback = RecommendationFeedback(
                    user_id=user_id,
                    content_id=rec['id'],
                    recommendation_score=rec['recommendation_score'],
                    recommendation_method=','.join(rec['methods_used']),
                    recommendation_reason=rec['recommendation_reason'],
                    recommendation_rank=idx + 1,
                    user_mode=user_mode,
                    time_of_day=datetime.utcnow().hour,
                    day_of_week=datetime.utcnow().weekday()
                )
                db.session.add(feedback)
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error tracking recommendations: {e}")
            db.session.rollback()
    
    def _estimate_accuracy(self, user_profile: Dict, algorithm_performance: Dict) -> float:
        """Estimate recommendation accuracy"""
        try:
            confidence = user_profile.get('confidence_score', 0)
            interaction_count = user_profile.get('interaction_count', 0)
            
            # Base accuracy from profile strength
            base_accuracy = min(confidence * 80, 80)
            
            # Interaction bonus
            interaction_bonus = min(interaction_count / 100.0 * 10, 10)
            
            # Algorithm diversity bonus
            active_algorithms = sum(1 for count in algorithm_performance.values() if count > 0)
            algorithm_bonus = min(active_algorithms * 2, 8)
            
            estimated_accuracy = base_accuracy + interaction_bonus + algorithm_bonus
            
            return min(round(estimated_accuracy, 1), 95.0)
            
        except Exception as e:
            logger.error(f"Error estimating accuracy: {e}")
            return 75.0

# Initialize the hyper-personalized engine
hyper_engine = HyperPersonalizedEngine()

# API Routes - These will be imported by users.py
def get_personalized_recommendations_for_user(user_id: int, content_type: Optional[str] = None, 
                                             limit: int = 20) -> Dict:
    """Get personalized recommendations for a user (to be called from users.py)"""
    context = {
        'time': datetime.utcnow().hour,
        'day': datetime.utcnow().weekday()
    }
    
    return hyper_engine.get_hyper_personalized_recommendations(
        user_id, content_type, context, limit
    )

def update_user_profile(user_id: int, force_update: bool = False) -> Dict:
    """Update user profile (to be called from users.py)"""
    return hyper_engine.profiler.build_comprehensive_profile(user_id, force_update)

def record_recommendation_feedback(user_id: int, content_id: int, feedback_type: str, 
                                  rating: Optional[float] = None) -> bool:
    """Record feedback for a recommendation"""
    try:
        feedback = RecommendationFeedback.query.filter_by(
            user_id=user_id,
            content_id=content_id
        ).order_by(RecommendationFeedback.recommended_at.desc()).first()
        
        if feedback:
            feedback.feedback_type = feedback_type
            feedback.user_rating = rating
            feedback.feedback_at = datetime.utcnow()
            feedback.was_successful = feedback_type in ['liked', 'watched', 'favorited']
            
            if feedback_type == 'watched' and rating:
                feedback.engagement_score = rating / 5.0
            
            db.session.commit()
            return True
            
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        db.session.rollback()
    
    return False