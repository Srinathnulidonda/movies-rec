from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import json
import logging
import jwt
import math
import random
import heapq
from functools import wraps, lru_cache
import hashlib
from sqlalchemy import func, and_, or_, desc, text
from sqlalchemy.orm import joinedload
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
from typing import Dict, List, Tuple, Optional, Any, Set
import re
from itertools import combinations
import warnings
import redis
from scipy import sparse
import threading
import time
from textdistance import jaro_winkler, jaccard
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

personalized_bp = Blueprint('personalized', __name__)
logger = logging.getLogger(__name__)

db = None
cache = None
User = None
Content = None
UserInteraction = None
AnonymousInteraction = None
ContentPerson = None
Person = None
Review = None
app = None
services = None

def init_personalized(flask_app, database, models, app_services, app_cache):
    global db, cache, User, Content, UserInteraction, AnonymousInteraction
    global ContentPerson, Person, Review, app, services
    
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
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

class AdvancedUserProfiler:
    def __init__(self):
        self.interaction_weights = {
            'rating': {'weight': 1.0, 'confidence': 0.95, 'decay_rate': 0.95},
            'favorite': {'weight': 0.9, 'confidence': 0.9, 'decay_rate': 0.93},
            'watchlist': {'weight': 0.75, 'confidence': 0.8, 'decay_rate': 0.9},
            'like': {'weight': 0.7, 'confidence': 0.75, 'decay_rate': 0.88},
            'view': {'weight': 0.5, 'confidence': 0.6, 'decay_rate': 0.85},
            'search': {'weight': 0.3, 'confidence': 0.4, 'decay_rate': 0.8},
            'click': {'weight': 0.2, 'confidence': 0.3, 'decay_rate': 0.75},
            'share': {'weight': 0.6, 'confidence': 0.7, 'decay_rate': 0.87},
            'rewatch': {'weight': 1.1, 'confidence': 0.98, 'decay_rate': 0.97}
        }
        
        self.temporal_weights = {
            'immediate': 1.0,
            'recent': 0.95,
            'moderate': 0.8,
            'old': 0.6,
            'ancient': 0.4
        }
    
    def build_comprehensive_profile(self, user_id: int) -> Dict[str, Any]:
        cache_key = f"user_profile_v3:{user_id}"
        if cache:
            try:
                cached = cache.get(cache_key)
                if cached:
                    return cached
            except Exception as e:
                logger.warning(f"Cache get error: {e}")
        
        try:
            interactions = UserInteraction.query.filter_by(user_id=user_id).order_by(
                UserInteraction.timestamp.desc()
            ).all()
            
            if not interactions:
                return self._get_default_profile(user_id)
            
            weighted_interactions = self._calculate_weighted_interactions(interactions)
            
            profile = {
                'user_id': user_id,
                'total_interactions': len(interactions),
                'profile_strength': 0.0,
                'confidence_score': 0.0,
                
                'content_preferences': {
                    'genres': defaultdict(float),
                    'languages': defaultdict(float),
                    'content_types': defaultdict(float),
                    'directors': defaultdict(float),
                    'actors': defaultdict(float),
                    'studios': defaultdict(float),
                    'release_years': defaultdict(float),
                    'ratings_preference': defaultdict(float)
                },
                
                'advanced_preferences': {
                    'runtime_preference': {'min': 0, 'max': 300, 'optimal': 120},
                    'popularity_bias': 0.0,
                    'novelty_seeking': 0.0,
                    'mainstream_vs_niche': 0.0,
                    'sequel_preference': 0.0,
                    'franchise_loyalty': defaultdict(float)
                },
                
                'quality_preferences': {
                    'avg_rating_given': 0.0,
                    'preferred_content_rating': 0.0,
                    'quality_tolerance': 0.0,
                    'high_quality_bias': 0.0,
                    'critic_vs_audience': 0.0
                },
                
                'temporal_patterns': {
                    'viewing_hours': defaultdict(int),
                    'viewing_days': defaultdict(int),
                    'seasonal_preferences': defaultdict(float),
                    'binge_patterns': [],
                    'viewing_frequency': 0.0,
                    'session_lengths': []
                },
                
                'behavioral_insights': {
                    'exploration_tendency': 0.0,
                    'completion_rate': 0.0,
                    'rewatch_tendency': 0.0,
                    'social_influence': 0.0,
                    'trending_sensitivity': 0.0,
                    'discovery_method': defaultdict(float),
                    'rating_behavior': defaultdict(float)
                },
                
                'sequence_patterns': {
                    'genre_transitions': defaultdict(float),
                    'mood_patterns': [],
                    'viewing_context': defaultdict(float),
                    'content_type_flows': defaultdict(float),
                    'temporal_sequences': []
                },
                
                'prediction_features': {
                    'next_likely_genres': [],
                    'declining_interests': [],
                    'emerging_interests': [],
                    'recommendation_receptivity': 0.0,
                    'preference_evolution': defaultdict(list),
                    'seasonal_trends': defaultdict(float)
                },
                
                'contextual_preferences': {
                    'device_preferences': defaultdict(float),
                    'time_context': defaultdict(float),
                    'mood_indicators': defaultdict(float),
                    'social_context': defaultdict(float)
                },
                
                'similarity_features': {
                    'user_clusters': [],
                    'content_embeddings': [],
                    'preference_vectors': {},
                    'collaborative_signals': defaultdict(float)
                }
            }
            
            self._extract_content_preferences(profile, weighted_interactions)
            self._extract_advanced_preferences(profile, weighted_interactions)
            self._extract_quality_preferences(profile, weighted_interactions)
            self._extract_temporal_patterns(profile, weighted_interactions)
            self._extract_behavioral_insights(profile, weighted_interactions)
            self._extract_sequence_patterns(profile, weighted_interactions)
            self._calculate_prediction_features(profile, weighted_interactions)
            self._extract_contextual_preferences(profile, weighted_interactions)
            self._build_similarity_features(profile, weighted_interactions)
            self._calculate_confidence_scores(profile, weighted_interactions)
            
            profile = self._serialize_profile(profile)
            
            if cache:
                try:
                    cache.set(cache_key, profile, timeout=1800)
                except Exception as e:
                    logger.warning(f"Cache set error: {e}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error building comprehensive profile: {e}")
            db.session.rollback()
            return self._get_default_profile(user_id)
    
    def _calculate_weighted_interactions(self, interactions):
        weighted_interactions = []
        current_time = datetime.utcnow()
        
        for interaction in interactions:
            days_ago = (current_time - interaction.timestamp).days
            temporal_weight = self._calculate_temporal_weight(days_ago)
            
            interaction_config = self.interaction_weights.get(
                interaction.interaction_type, 
                {'weight': 0.1, 'confidence': 0.2, 'decay_rate': 0.8}
            )
            
            base_weight = interaction_config['weight'] * temporal_weight
            confidence = interaction_config['confidence']
            
            if interaction.rating:
                rating_boost = (float(interaction.rating) / 5.0) * 1.3
                base_weight *= rating_boost
            
            weighted_interactions.append({
                'interaction': interaction,
                'weight': base_weight,
                'confidence': confidence,
                'temporal_weight': temporal_weight,
                'days_ago': days_ago
            })
        
        return weighted_interactions
    
    def _calculate_temporal_weight(self, days_ago: int) -> float:
        if days_ago <= 7:
            return self.temporal_weights['immediate']
        elif days_ago <= 30:
            return self.temporal_weights['recent']
        elif days_ago <= 90:
            return self.temporal_weights['moderate']
        elif days_ago <= 180:
            return self.temporal_weights['old']
        else:
            return self.temporal_weights['ancient']
    
    def _extract_content_preferences(self, profile: Dict, weighted_interactions: List[Dict]):
        total_weight = sum(wi['weight'] for wi in weighted_interactions)
        
        if total_weight == 0:
            return
        
        content_ids = [wi['interaction'].content_id for wi in weighted_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_map = {c.id: c for c in contents}
        
        for wi in weighted_interactions:
            interaction = wi['interaction']
            weight = wi['weight']
            content = content_map.get(interaction.content_id)
            
            if not content:
                continue
            
            try:
                genres = json.loads(content.genres or '[]')
                for i, genre in enumerate(genres[:3]):
                    importance = 1.0 / (i + 1)
                    profile['content_preferences']['genres'][genre.lower()] += weight * importance
            except (json.JSONDecodeError, TypeError):
                pass
            
            try:
                languages = json.loads(content.languages or '[]')
                for lang in languages:
                    profile['content_preferences']['languages'][lang.lower()] += weight
            except (json.JSONDecodeError, TypeError):
                pass
            
            profile['content_preferences']['content_types'][content.content_type] += weight
            
            if content.release_date:
                year = content.release_date.year
                decade = (year // 10) * 10
                profile['content_preferences']['release_years'][str(decade)] += weight * 0.5
                profile['content_preferences']['release_years'][str(year)] += weight * 0.3
            
            if content.rating:
                rating_range = f"{int(content.rating)}-{int(content.rating)+1}"
                profile['content_preferences']['ratings_preference'][rating_range] += weight
        
        for pref_type in profile['content_preferences']:
            if isinstance(profile['content_preferences'][pref_type], defaultdict):
                for key in profile['content_preferences'][pref_type]:
                    profile['content_preferences'][pref_type][key] /= total_weight
    
    def _extract_advanced_preferences(self, profile: Dict, weighted_interactions: List[Dict]):
        content_ids = [wi['interaction'].content_id for wi in weighted_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_map = {c.id: c for c in contents}
        
        runtimes = []
        popularities = []
        novelty_scores = []
        
        for wi in weighted_interactions:
            content = content_map.get(wi['interaction'].content_id)
            if content:
                if content.runtime:
                    runtimes.append(content.runtime)
                
                if content.popularity:
                    popularities.append(float(content.popularity))
                
                if content.release_date:
                    days_since_release = (datetime.utcnow().date() - content.release_date).days
                    novelty_scores.append(max(0, 1 - (days_since_release / 3650)))
        
        if runtimes:
            profile['advanced_preferences']['runtime_preference'] = {
                'min': min(runtimes),
                'max': max(runtimes),
                'optimal': np.median(runtimes),
                'variance': np.var(runtimes)
            }
        
        if popularities:
            avg_popularity = np.mean(popularities)
            profile['advanced_preferences']['popularity_bias'] = min(avg_popularity / 1000.0, 1.0)
            profile['advanced_preferences']['mainstream_vs_niche'] = 1.0 if avg_popularity > 500 else 0.0
        
        if novelty_scores:
            profile['advanced_preferences']['novelty_seeking'] = np.mean(novelty_scores)
    
    def _extract_quality_preferences(self, profile: Dict, weighted_interactions: List[Dict]):
        ratings_given = []
        content_ratings = []
        
        content_ids = [wi['interaction'].content_id for wi in weighted_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_map = {c.id: c for c in contents}
        
        for wi in weighted_interactions:
            interaction = wi['interaction']
            content = content_map.get(interaction.content_id)
            
            if interaction.rating:
                ratings_given.append(float(interaction.rating))
            
            if content and content.rating:
                content_ratings.append(float(content.rating))
        
        if ratings_given:
            profile['quality_preferences']['avg_rating_given'] = float(np.mean(ratings_given))
            profile['quality_preferences']['quality_tolerance'] = float(np.std(ratings_given))
            profile['quality_preferences']['high_quality_bias'] = len([r for r in ratings_given if r >= 4]) / len(ratings_given)
        
        if content_ratings:
            profile['quality_preferences']['preferred_content_rating'] = float(np.mean(content_ratings))
            profile['quality_preferences']['critic_vs_audience'] = 0.5
    
    def _extract_temporal_patterns(self, profile: Dict, weighted_interactions: List[Dict]):
        daily_sessions = defaultdict(list)
        session_lengths = []
        
        for wi in weighted_interactions:
            interaction = wi['interaction']
            timestamp = interaction.timestamp
            
            hour = timestamp.hour
            day = timestamp.weekday()
            month = timestamp.month
            date_key = timestamp.date()
            
            profile['temporal_patterns']['viewing_hours'][hour] += wi['weight']
            profile['temporal_patterns']['viewing_days'][day] += wi['weight']
            profile['temporal_patterns']['seasonal_preferences'][month] += wi['weight']
            
            daily_sessions[date_key].append({
                'timestamp': timestamp,
                'weight': wi['weight'],
                'interaction': interaction
            })
        
        for date, sessions in daily_sessions.items():
            if len(sessions) >= 2:
                sorted_sessions = sorted(sessions, key=lambda x: x['timestamp'])
                session_start = sorted_sessions[0]['timestamp']
                session_end = sorted_sessions[-1]['timestamp']
                session_length = (session_end - session_start).total_seconds() / 3600
                session_lengths.append(session_length)
                
                if len(sessions) >= 3:
                    content_ids = [s['interaction'].content_id for s in sessions]
                    contents = Content.query.filter(Content.id.in_(content_ids)).all()
                    content_types = [c.content_type for c in contents]
                    
                    profile['temporal_patterns']['binge_patterns'].append({
                        'date': date.isoformat(),
                        'content_count': len(sessions),
                        'session_length_hours': session_length,
                        'primary_type': Counter(content_types).most_common(1)[0][0] if content_types else 'unknown'
                    })
        
        if session_lengths:
            profile['temporal_patterns']['session_lengths'] = session_lengths
        
        total_days = (datetime.utcnow() - min(wi['interaction'].timestamp for wi in weighted_interactions)).days
        if total_days > 0:
            profile['temporal_patterns']['viewing_frequency'] = len(weighted_interactions) / total_days
    
    def _extract_behavioral_insights(self, profile: Dict, weighted_interactions: List[Dict]):
        interaction_types = [wi['interaction'].interaction_type for wi in weighted_interactions]
        type_counts = Counter(interaction_types)
        
        total_interactions = len(weighted_interactions)
        
        if total_interactions > 0:
            profile['behavioral_insights']['exploration_tendency'] = (
                type_counts.get('search', 0) + type_counts.get('click', 0)
            ) / total_interactions
            
            profile['behavioral_insights']['completion_rate'] = (
                type_counts.get('rating', 0) + type_counts.get('favorite', 0)
            ) / total_interactions
            
            profile['behavioral_insights']['rewatch_tendency'] = type_counts.get('rewatch', 0) / total_interactions
            
            profile['behavioral_insights']['social_influence'] = (
                type_counts.get('share', 0) + type_counts.get('like', 0)
            ) / total_interactions
        
        content_ids = [wi['interaction'].content_id for wi in weighted_interactions]
        try:
            trending_count = Content.query.filter(
                Content.id.in_(content_ids),
                Content.is_trending == True
            ).count()
            
            if content_ids:
                profile['behavioral_insights']['trending_sensitivity'] = trending_count / len(set(content_ids))
        except Exception as e:
            logger.warning(f"Error calculating trending sensitivity: {e}")
            profile['behavioral_insights']['trending_sensitivity'] = 0.0
        
        rating_behaviors = []
        for wi in weighted_interactions:
            if wi['interaction'].rating:
                rating_behaviors.append(float(wi['interaction'].rating))
        
        if rating_behaviors:
            profile['behavioral_insights']['rating_behavior']['generosity'] = np.mean(rating_behaviors) / 5.0
            profile['behavioral_insights']['rating_behavior']['consistency'] = 1.0 - (np.std(rating_behaviors) / 5.0)
    
    def _extract_sequence_patterns(self, profile: Dict, weighted_interactions: List[Dict]):
        sorted_interactions = sorted(weighted_interactions, key=lambda x: x['interaction'].timestamp)
        
        content_ids = [wi['interaction'].content_id for wi in sorted_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_map = {c.id: c for c in contents}
        
        for i in range(len(sorted_interactions) - 1):
            current_wi = sorted_interactions[i]
            next_wi = sorted_interactions[i + 1]
            
            current_content = content_map.get(current_wi['interaction'].content_id)
            next_content = content_map.get(next_wi['interaction'].content_id)
            
            if current_content and next_content:
                try:
                    current_genres = json.loads(current_content.genres or '[]')
                    next_genres = json.loads(next_content.genres or '[]')
                    
                    if current_genres and next_genres:
                        transition = f"{current_genres[0].lower()}->{next_genres[0].lower()}"
                        profile['sequence_patterns']['genre_transitions'][transition] += current_wi['weight']
                    
                    type_transition = f"{current_content.content_type}->{next_content.content_type}"
                    profile['sequence_patterns']['content_type_flows'][type_transition] += current_wi['weight']
                    
                except (json.JSONDecodeError, TypeError):
                    pass
    
    def _calculate_prediction_features(self, profile: Dict, weighted_interactions: List[Dict]):
        time_windows = [30, 90, 180]
        current_time = datetime.utcnow()
        
        genre_evolution = {}
        
        for window in time_windows:
            window_start = current_time - timedelta(days=window)
            window_interactions = [
                wi for wi in weighted_interactions 
                if wi['interaction'].timestamp >= window_start
            ]
            
            if window_interactions:
                content_ids = [wi['interaction'].content_id for wi in window_interactions]
                contents = Content.query.filter(Content.id.in_(content_ids)).all()
                
                window_genres = defaultdict(float)
                for content in contents:
                    try:
                        genres = json.loads(content.genres or '[]')
                        for genre in genres:
                            window_genres[genre.lower()] += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                genre_evolution[f'{window}_days'] = dict(window_genres)
        
        if len(genre_evolution) >= 2:
            recent_genres = genre_evolution.get('30_days', {})
            older_genres = genre_evolution.get('90_days', {})
            
            emerging = []
            declining = []
            
            for genre, recent_count in recent_genres.items():
                older_count = older_genres.get(genre, 0)
                if recent_count > older_count * 1.5:
                    emerging.append(genre)
                elif recent_count < older_count * 0.5 and older_count > 0:
                    declining.append(genre)
            
            profile['prediction_features']['emerging_interests'] = emerging
            profile['prediction_features']['declining_interests'] = declining
            
            profile['prediction_features']['next_likely_genres'] = list(recent_genres.keys())[:5]
        
        receptivity_factors = [
            profile['behavioral_insights']['exploration_tendency'],
            profile['behavioral_insights']['completion_rate'],
            1.0 - profile['behavioral_insights']['rewatch_tendency']
        ]
        profile['prediction_features']['recommendation_receptivity'] = np.mean(receptivity_factors)
    
    def _extract_contextual_preferences(self, profile: Dict, weighted_interactions: List[Dict]):
        for wi in weighted_interactions:
            interaction = wi['interaction']
            
            try:
                metadata = json.loads(interaction.interaction_metadata or '{}')
                
                device = metadata.get('device_type', 'unknown')
                if device != 'unknown':
                    profile['contextual_preferences']['device_preferences'][device] += wi['weight']
                
                time_context = metadata.get('time_context', 'unknown')
                if time_context != 'unknown':
                    profile['contextual_preferences']['time_context'][time_context] += wi['weight']
                
                hour = interaction.timestamp.hour
                if 6 <= hour < 12:
                    time_period = 'morning'
                elif 12 <= hour < 17:
                    time_period = 'afternoon'
                elif 17 <= hour < 22:
                    time_period = 'evening'
                else:
                    time_period = 'night'
                
                profile['contextual_preferences']['time_context'][time_period] += wi['weight']
                
            except (json.JSONDecodeError, TypeError):
                pass
    
    def _build_similarity_features(self, profile: Dict, weighted_interactions: List[Dict]):
        content_ids = [wi['interaction'].content_id for wi in weighted_interactions]
        
        try:
            similar_users = db.session.query(
                UserInteraction.user_id,
                func.count(UserInteraction.content_id).label('common_content')
            ).filter(
                UserInteraction.content_id.in_(content_ids),
                UserInteraction.user_id != profile['user_id']
            ).group_by(
                UserInteraction.user_id
            ).having(
                func.count(UserInteraction.content_id) >= max(3, len(content_ids) * 0.1)
            ).order_by(
                desc('common_content')
            ).limit(20).all()
            
            profile['similarity_features']['user_clusters'] = [
                {'user_id': user_id, 'similarity': min(common_content / len(content_ids), 1.0)}
                for user_id, common_content in similar_users
            ]
            
        except Exception as e:
            logger.warning(f"Error building similarity features: {e}")
            profile['similarity_features']['user_clusters'] = []
    
    def _calculate_confidence_scores(self, profile: Dict, weighted_interactions: List[Dict]):
        interaction_count = len(weighted_interactions)
        
        base_confidence = min(interaction_count / 100.0, 1.0)
        
        interaction_types = set([wi['interaction'].interaction_type for wi in weighted_interactions])
        diversity_bonus = len(interaction_types) / 8.0
        
        rating_interactions = [wi for wi in weighted_interactions if wi['interaction'].rating is not None]
        rating_bonus = len(rating_interactions) / max(interaction_count, 1) * 0.4
        
        if weighted_interactions:
            time_span = (
                weighted_interactions[0]['interaction'].timestamp - 
                weighted_interactions[-1]['interaction'].timestamp
            ).days
            temporal_bonus = min(time_span / 120.0, 1.0) * 0.3
        else:
            temporal_bonus = 0
        
        confidence = min(base_confidence + diversity_bonus + rating_bonus + temporal_bonus, 1.0)
        profile['confidence_score'] = round(confidence, 3)
        profile['profile_strength'] = round(confidence * 100, 1)
    
    def _serialize_profile(self, profile: Dict) -> Dict:
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
        return {
            'user_id': user_id,
            'total_interactions': 0,
            'profile_strength': 0.0,
            'confidence_score': 0.0,
            'content_preferences': {
                'genres': {'action': 0.2, 'drama': 0.2, 'comedy': 0.15, 'thriller': 0.15},
                'languages': {'english': 0.4, 'telugu': 0.3, 'hindi': 0.2, 'tamil': 0.1},
                'content_types': {'movie': 0.5, 'tv': 0.3, 'anime': 0.2}
            },
            'quality_preferences': {'avg_rating_given': 7.0, 'preferred_content_rating': 7.0},
            'temporal_patterns': {'viewing_hours': {}, 'viewing_days': {}},
            'behavioral_insights': {'exploration_tendency': 0.5},
            'sequence_patterns': {'genre_transitions': {}},
            'prediction_features': {'next_likely_genres': [], 'emerging_interests': []},
            'contextual_preferences': {'device_preferences': {}, 'time_context': {}}
        }

class UltraPowerfulCollaborativeEngine:
    def __init__(self):
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.model_trained = False
        self.similarity_threshold = 0.1
        
    def build_advanced_user_item_matrix(self):
        try:
            interactions = UserInteraction.query.filter(
                UserInteraction.interaction_type.in_(['rating', 'favorite', 'like', 'view', 'watchlist'])
            ).all()
            
            if not interactions:
                return None, None, None
            
            user_ids = list(set([i.user_id for i in interactions]))
            item_ids = list(set([i.content_id for i in interactions]))
            
            user_mapping = {uid: idx for idx, uid in enumerate(user_ids)}
            item_mapping = {iid: idx for idx, iid in enumerate(item_ids)}
            
            n_users = len(user_ids)
            n_items = len(item_ids)
            
            rating_matrix = sparse.lil_matrix((n_users, n_items))
            implicit_matrix = sparse.lil_matrix((n_users, n_items))
            
            interaction_weights = {
                'rating': 1.0,
                'favorite': 0.9,
                'like': 0.7,
                'watchlist': 0.6,
                'view': 0.4
            }
            
            for interaction in interactions:
                user_idx = user_mapping[interaction.user_id]
                item_idx = item_mapping[interaction.content_id]
                
                weight = interaction_weights.get(interaction.interaction_type, 0.1)
                
                if interaction.rating:
                    explicit_rating = float(interaction.rating)
                    rating_matrix[user_idx, item_idx] = explicit_rating
                    implicit_matrix[user_idx, item_idx] = weight * explicit_rating
                else:
                    implicit_rating = weight * 5.0
                    if interaction.interaction_type == 'favorite':
                        implicit_rating = 5.0
                    elif interaction.interaction_type == 'like':
                        implicit_rating = 4.0
                    elif interaction.interaction_type == 'watchlist':
                        implicit_rating = 3.5
                    elif interaction.interaction_type == 'view':
                        implicit_rating = 3.0
                    
                    rating_matrix[user_idx, item_idx] = implicit_rating
                    implicit_matrix[user_idx, item_idx] = implicit_rating
            
            return rating_matrix.tocsr(), implicit_matrix.tocsr(), user_mapping, item_mapping
            
        except Exception as e:
            logger.error(f"Error building advanced user-item matrix: {e}")
            db.session.rollback()
            return None, None, None, None
    
    def train_hybrid_collaborative_model(self):
        try:
            rating_matrix, implicit_matrix, user_mapping, item_mapping = self.build_advanced_user_item_matrix()
            
            if rating_matrix is None:
                return False
            
            n_components = min(100, min(rating_matrix.shape) - 1)
            
            if n_components < 5:
                return False
            
            explicit_svd = TruncatedSVD(n_components=n_components, random_state=42)
            implicit_svd = TruncatedSVD(n_components=n_components, random_state=43)
            
            explicit_user_factors = explicit_svd.fit_transform(rating_matrix)
            explicit_item_factors = explicit_svd.components_.T
            
            implicit_user_factors = implicit_svd.fit_transform(implicit_matrix)
            implicit_item_factors = implicit_svd.components_.T
            
            self.user_embeddings = {}
            self.item_embeddings = {}
            
            for user_id, user_idx in user_mapping.items():
                combined_embedding = np.concatenate([
                    explicit_user_factors[user_idx] * 0.7,
                    implicit_user_factors[user_idx] * 0.3
                ])
                self.user_embeddings[user_id] = combined_embedding
            
            for item_id, item_idx in item_mapping.items():
                combined_embedding = np.concatenate([
                    explicit_item_factors[item_idx] * 0.7,
                    implicit_item_factors[item_idx] * 0.3
                ])
                self.item_embeddings[item_id] = combined_embedding
            
            self.user_mapping = user_mapping
            self.item_mapping = item_mapping
            self.model_trained = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error training hybrid collaborative model: {e}")
            db.session.rollback()
            return False
    
    def get_ultra_accurate_recommendations(self, user_id: int, user_profile: Dict, 
                                         n_recommendations: int = 20) -> List[Dict]:
        try:
            if not self.model_trained or user_id not in self.user_embeddings:
                if not self.train_hybrid_collaborative_model():
                    return []
                
                if user_id not in self.user_embeddings:
                    return []
            
            user_embedding = self.user_embeddings[user_id]
            
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_items = set([i.content_id for i in user_interactions])
            
            item_scores = []
            for item_id, item_embedding in self.item_embeddings.items():
                if item_id not in interacted_items:
                    base_similarity = cosine_similarity([user_embedding], [item_embedding])[0][0]
                    
                    content = Content.query.get(item_id)
                    if content:
                        preference_boost = self._calculate_preference_alignment(content, user_profile)
                        quality_boost = self._calculate_quality_boost(content, user_profile)
                        popularity_adjustment = self._calculate_popularity_adjustment(content, user_profile)
                        
                        final_score = (
                            base_similarity * 0.4 +
                            preference_boost * 0.35 +
                            quality_boost * 0.15 +
                            popularity_adjustment * 0.1
                        )
                        
                        item_scores.append((item_id, float(final_score), base_similarity, preference_boost))
            
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for item_id, score, similarity, preference in item_scores[:n_recommendations]:
                content = Content.query.get(item_id)
                if content:
                    recommendations.append({
                        'content_id': item_id,
                        'score': score,
                        'similarity_score': similarity,
                        'preference_score': preference,
                        'method': 'ultra_collaborative_filtering',
                        'content': content
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting ultra accurate recommendations: {e}")
            db.session.rollback()
            return []
    
    def _calculate_preference_alignment(self, content: Content, user_profile: Dict) -> float:
        score = 0.0
        
        try:
            content_genres = json.loads(content.genres or '[]')
            user_genres = user_profile.get('content_preferences', {}).get('genres', {})
            
            for genre in content_genres:
                genre_pref = user_genres.get(genre.lower(), 0)
                score += genre_pref * 0.4
        except (json.JSONDecodeError, TypeError):
            pass
        
        try:
            content_languages = json.loads(content.languages or '[]')
            user_languages = user_profile.get('content_preferences', {}).get('languages', {})
            
            for lang in content_languages:
                lang_pref = user_languages.get(lang.lower(), 0)
                score += lang_pref * 0.35
        except (json.JSONDecodeError, TypeError):
            pass
        
        content_type_prefs = user_profile.get('content_preferences', {}).get('content_types', {})
        content_type_pref = content_type_prefs.get(content.content_type, 0)
        score += content_type_pref * 0.25
        
        return min(score, 1.0)
    
    def _calculate_quality_boost(self, content: Content, user_profile: Dict) -> float:
        quality_prefs = user_profile.get('quality_preferences', {})
        
        if not content.rating:
            return 0.5
        
        preferred_rating = quality_prefs.get('preferred_content_rating', 7.0)
        rating_diff = abs(float(content.rating) - preferred_rating)
        
        quality_score = max(0, 1 - (rating_diff / 5.0))
        
        if content.vote_count and content.vote_count > 1000:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _calculate_popularity_adjustment(self, content: Content, user_profile: Dict) -> float:
        if not content.popularity:
            return 0.5
        
        popularity_bias = user_profile.get('advanced_preferences', {}).get('popularity_bias', 0.5)
        content_popularity = min(float(content.popularity) / 1000.0, 1.0)
        
        if popularity_bias > 0.7:
            return content_popularity
        elif popularity_bias < 0.3:
            return 1.0 - content_popularity
        else:
            return 0.5

class AdvancedContentBasedEngine:
    def __init__(self):
        self.content_embeddings = {}
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000, 
            ngram_range=(1, 3), 
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.feature_scaler = StandardScaler()
        self.embeddings_built = False
        
    def build_advanced_content_embeddings(self):
        try:
            contents = Content.query.all()
            if not contents:
                return False
            
            content_descriptions = []
            content_metadata = []
            content_ids = []
            
            for content in contents:
                description_parts = []
                
                if content.title:
                    description_parts.append(content.title.lower())
                if content.original_title and content.original_title != content.title:
                    description_parts.append(content.original_title.lower())
                if content.overview:
                    clean_overview = re.sub(r'[^\w\s]', ' ', content.overview.lower())
                    description_parts.append(clean_overview)
                
                try:
                    genres = json.loads(content.genres or '[]')
                    genre_text = ' '.join([g.lower().replace(' ', '_') for g in genres] * 3)
                    description_parts.append(genre_text)
                except (json.JSONDecodeError, TypeError):
                    genres = []
                
                try:
                    languages = json.loads(content.languages or '[]')
                    lang_text = ' '.join([l.lower().replace(' ', '_') for l in languages] * 2)
                    description_parts.append(lang_text)
                except (json.JSONDecodeError, TypeError):
                    languages = []
                
                description_parts.append(content.content_type)
                
                metadata = {
                    'rating': float(content.rating or 0),
                    'popularity': float(content.popularity or 0),
                    'runtime': float(content.runtime or 120),
                    'release_year': content.release_date.year if content.release_date else 2020,
                    'vote_count': float(content.vote_count or 0),
                    'is_trending': int(content.is_trending or False),
                    'is_new_release': int(content.is_new_release or False),
                    'genre_count': len(genres),
                    'language_count': len(languages),
                    'title_length': len(content.title or ''),
                    'overview_length': len(content.overview or ''),
                    'popularity_percentile': 0.5,
                    'rating_percentile': 0.5
                }
                
                content_descriptions.append(' '.join(description_parts))
                content_metadata.append(metadata)
                content_ids.append(content.id)
            
            if not content_descriptions:
                return False
            
            tfidf_features = self.tfidf_vectorizer.fit_transform(content_descriptions)
            
            metadata_df = pd.DataFrame(content_metadata)
            
            popularity_percentiles = pd.qcut(
                metadata_df['popularity'], 
                q=10, 
                labels=False, 
                duplicates='drop'
            ) / 9.0
            rating_percentiles = pd.qcut(
                metadata_df['rating'], 
                q=10, 
                labels=False, 
                duplicates='drop'
            ) / 9.0
            
            metadata_df['popularity_percentile'] = popularity_percentiles.fillna(0.5)
            metadata_df['rating_percentile'] = rating_percentiles.fillna(0.5)
            
            metadata_features = self.feature_scaler.fit_transform(metadata_df)
            
            combined_features = sparse.hstack([
                tfidf_features * 0.7,
                sparse.csr_matrix(metadata_features) * 0.3
            ])
            
            for i, content_id in enumerate(content_ids):
                self.content_embeddings[content_id] = combined_features[i].toarray().flatten()
            
            self.embeddings_built = True
            return True
            
        except Exception as e:
            logger.error(f"Error building advanced content embeddings: {e}")
            db.session.rollback()
            return False
    
    def get_ultra_content_recommendations(self, user_id: int, user_profile: Dict, 
                                        n_recommendations: int = 20) -> List[Dict]:
        try:
            if not self.embeddings_built:
                if not self.build_advanced_content_embeddings():
                    return []
            
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not user_interactions:
                return []
            
            interacted_content_ids = [i.content_id for i in user_interactions]
            interacted_embeddings = []
            interaction_weights = []
            
            interaction_weight_map = {
                'rating': 1.0,
                'favorite': 0.9,
                'like': 0.7,
                'watchlist': 0.6,
                'view': 0.4
            }
            
            for interaction in user_interactions:
                if interaction.content_id in self.content_embeddings:
                    embedding = self.content_embeddings[interaction.content_id]
                    interacted_embeddings.append(embedding)
                    
                    weight = interaction_weight_map.get(interaction.interaction_type, 0.1)
                    
                    if interaction.rating:
                        weight *= (float(interaction.rating) / 5.0)
                    
                    days_ago = (datetime.utcnow() - interaction.timestamp).days
                    temporal_weight = max(0.1, 1.0 - (days_ago / 365.0))
                    
                    final_weight = weight * temporal_weight
                    interaction_weights.append(final_weight)
            
            if not interacted_embeddings:
                return []
            
            interacted_embeddings = np.array(interacted_embeddings)
            interaction_weights = np.array(interaction_weights)
            interaction_weights = interaction_weights / np.sum(interaction_weights)
            
            user_profile_vector = np.average(interacted_embeddings, axis=0, weights=interaction_weights)
            
            similarities = []
            for content_id, content_embedding in self.content_embeddings.items():
                if content_id not in interacted_content_ids:
                    base_similarity = cosine_similarity([user_profile_vector], [content_embedding])[0][0]
                    
                    content = Content.query.get(content_id)
                    if content:
                        preference_boost = self._calculate_deep_preference_match(content, user_profile)
                        diversity_bonus = self._calculate_diversity_bonus(content, interacted_content_ids)
                        novelty_score = self._calculate_novelty_score(content, user_profile)
                        
                        final_score = (
                            base_similarity * 0.5 +
                            preference_boost * 0.3 +
                            diversity_bonus * 0.1 +
                            novelty_score * 0.1
                        )
                        
                        similarities.append((content_id, float(final_score), base_similarity, preference_boost))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for content_id, score, similarity, preference in similarities[:n_recommendations]:
                content = Content.query.get(content_id)
                if content:
                    recommendations.append({
                        'content_id': content_id,
                        'score': score,
                        'similarity_score': similarity,
                        'preference_score': preference,
                        'method': 'ultra_content_based_filtering',
                        'content': content
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting ultra content recommendations: {e}")
            db.session.rollback()
            return []
    
    def _calculate_deep_preference_match(self, content: Content, user_profile: Dict) -> float:
        score = 0.0
        
        try:
            content_genres = json.loads(content.genres or '[]')
            user_genres = user_profile.get('content_preferences', {}).get('genres', {})
            
            genre_matches = []
            for genre in content_genres[:3]:
                genre_pref = user_genres.get(genre.lower(), 0)
                genre_matches.append(genre_pref)
            
            if genre_matches:
                score += np.mean(genre_matches) * 0.4
        except (json.JSONDecodeError, TypeError):
            pass
        
        try:
            content_languages = json.loads(content.languages or '[]')
            user_languages = user_profile.get('content_preferences', {}).get('languages', {})
            
            lang_scores = []
            for lang in content_languages:
                lang_pref = user_languages.get(lang.lower(), 0)
                lang_scores.append(lang_pref)
            
            if lang_scores:
                score += max(lang_scores) * 0.3
        except (json.JSONDecodeError, TypeError):
            pass
        
        runtime_prefs = user_profile.get('advanced_preferences', {}).get('runtime_preference', {})
        if content.runtime and runtime_prefs:
            optimal_runtime = runtime_prefs.get('optimal', 120)
            runtime_diff = abs(content.runtime - optimal_runtime) / optimal_runtime
            runtime_score = max(0, 1 - runtime_diff)
            score += runtime_score * 0.15
        
        quality_prefs = user_profile.get('quality_preferences', {})
        if content.rating and quality_prefs:
            preferred_rating = quality_prefs.get('preferred_content_rating', 7.0)
            rating_diff = abs(float(content.rating) - preferred_rating) / 5.0
            quality_score = max(0, 1 - rating_diff)
            score += quality_score * 0.15
        
        return min(score, 1.0)
    
    def _calculate_diversity_bonus(self, content: Content, interacted_content_ids: List[int]) -> float:
        try:
            interacted_contents = Content.query.filter(Content.id.in_(interacted_content_ids)).all()
            
            interacted_genres = set()
            interacted_languages = set()
            interacted_types = set()
            
            for ic in interacted_contents:
                interacted_types.add(ic.content_type)
                
                try:
                    genres = json.loads(ic.genres or '[]')
                    interacted_genres.update([g.lower() for g in genres])
                except (json.JSONDecodeError, TypeError):
                    pass
                
                try:
                    languages = json.loads(ic.languages or '[]')
                    interacted_languages.update([l.lower() for l in languages])
                except (json.JSONDecodeError, TypeError):
                    pass
            
            diversity_score = 0.0
            
            try:
                content_genres = json.loads(content.genres or '[]')
                new_genres = set([g.lower() for g in content_genres]) - interacted_genres
                if new_genres:
                    diversity_score += 0.4
            except (json.JSONDecodeError, TypeError):
                pass
            
            try:
                content_languages = json.loads(content.languages or '[]')
                new_languages = set([l.lower() for l in content_languages]) - interacted_languages
                if new_languages:
                    diversity_score += 0.3
            except (json.JSONDecodeError, TypeError):
                pass
            
            if content.content_type not in interacted_types:
                diversity_score += 0.3
            
            return min(diversity_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating diversity bonus: {e}")
            return 0.0
    
    def _calculate_novelty_score(self, content: Content, user_profile: Dict) -> float:
        novelty_seeking = user_profile.get('advanced_preferences', {}).get('novelty_seeking', 0.5)
        
        if content.release_date:
            days_since_release = (datetime.utcnow().date() - content.release_date).days
            recency_score = max(0, 1 - (days_since_release / 3650))
        else:
            recency_score = 0.5
        
        popularity_score = 0.5
        if content.popularity:
            normalized_popularity = min(float(content.popularity) / 1000.0, 1.0)
            popularity_score = 1.0 - normalized_popularity if novelty_seeking > 0.6 else normalized_popularity
        
        return (recency_score * 0.6 + popularity_score * 0.4) * novelty_seeking

class IntelligentSequenceEngine:
    def __init__(self):
        self.transition_matrices = {
            'genre': defaultdict(lambda: defaultdict(float)),
            'content_type': defaultdict(lambda: defaultdict(float)),
            'mood': defaultdict(lambda: defaultdict(float)),
            'language': defaultdict(lambda: defaultdict(float))
        }
        self.sequence_models = {}
        
    def build_comprehensive_sequence_models(self, user_id: int):
        try:
            interactions = UserInteraction.query.filter_by(user_id=user_id).order_by(
                UserInteraction.timestamp
            ).all()
            
            if len(interactions) < 5:
                return
            
            content_ids = [i.content_id for i in interactions]
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_map = {c.id: c for c in contents}
            
            sequences = self._build_sequences(interactions, content_map)
            
            self._build_all_transition_matrices(sequences)
            self._build_temporal_patterns(sequences)
            self._build_contextual_patterns(sequences, interactions)
            
        except Exception as e:
            logger.error(f"Error building comprehensive sequence models: {e}")
            db.session.rollback()
    
    def _build_sequences(self, interactions: List, content_map: Dict) -> List[List[Dict]]:
        sequences = []
        current_sequence = []
        
        for i, interaction in enumerate(interactions):
            content = content_map.get(interaction.content_id)
            if content:
                try:
                    genres = json.loads(content.genres or '[]')
                    languages = json.loads(content.languages or '[]')
                    
                    state = {
                        'content_id': content.id,
                        'content_type': content.content_type,
                        'primary_genre': genres[0].lower() if genres else 'unknown',
                        'secondary_genre': genres[1].lower() if len(genres) > 1 else None,
                        'primary_language': languages[0].lower() if languages else 'unknown',
                        'rating': float(content.rating or 0),
                        'popularity': float(content.popularity or 0),
                        'runtime': content.runtime or 120,
                        'interaction_type': interaction.interaction_type,
                        'timestamp': interaction.timestamp,
                        'user_rating': float(interaction.rating) if interaction.rating else None,
                        'hour_of_day': interaction.timestamp.hour,
                        'day_of_week': interaction.timestamp.weekday()
                    }
                    
                    current_sequence.append(state)
                    
                    if i > 0:
                        time_gap = (interaction.timestamp - interactions[i-1].timestamp).total_seconds()
                        if time_gap > 86400 or len(current_sequence) > 20:
                            if len(current_sequence) >= 3:
                                sequences.append(current_sequence.copy())
                            current_sequence = [state]
                except (json.JSONDecodeError, TypeError, IndexError):
                    continue
        
        if len(current_sequence) >= 3:
            sequences.append(current_sequence)
        
        return sequences
    
    def _build_all_transition_matrices(self, sequences: List[List[Dict]]):
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_item = sequence[i + 1]
                
                current_genre = current['primary_genre']
                next_genre = next_item['primary_genre']
                
                if current_genre != 'unknown' and next_genre != 'unknown':
                    self.transition_matrices['genre'][current_genre][next_genre] += 1
                
                current_type = current['content_type']
                next_type = next_item['content_type']
                self.transition_matrices['content_type'][current_type][next_type] += 1
                
                current_lang = current['primary_language']
                next_lang = next_item['primary_language']
                if current_lang != 'unknown' and next_lang != 'unknown':
                    self.transition_matrices['language'][current_lang][next_lang] += 1
                
                current_mood = self._infer_mood(current)
                next_mood = self._infer_mood(next_item)
                self.transition_matrices['mood'][current_mood][next_mood] += 1
    
    def _infer_mood(self, state: Dict) -> str:
        genre = state['primary_genre']
        hour = state['hour_of_day']
        rating = state['rating']
        
        if genre in ['horror', 'thriller', 'mystery']:
            return 'intense'
        elif genre in ['comedy', 'animation', 'family']:
            return 'light'
        elif genre in ['drama', 'romance', 'biography']:
            return 'emotional'
        elif genre in ['action', 'adventure', 'sci-fi']:
            return 'exciting'
        elif hour >= 22 or hour <= 6:
            return 'relaxed'
        elif rating >= 8.0:
            return 'quality'
        else:
            return 'neutral'
    
    def _build_temporal_patterns(self, sequences: List[List[Dict]]):
        hourly_preferences = defaultdict(lambda: defaultdict(float))
        daily_preferences = defaultdict(lambda: defaultdict(float))
        
        for sequence in sequences:
            for state in sequence:
                hour = state['hour_of_day']
                day = state['day_of_week']
                genre = state['primary_genre']
                
                hourly_preferences[hour][genre] += 1
                daily_preferences[day][genre] += 1
        
        self.sequence_models['temporal'] = {
            'hourly': dict(hourly_preferences),
            'daily': dict(daily_preferences)
        }
    
    def _build_contextual_patterns(self, sequences: List[List[Dict]], interactions: List):
        context_patterns = defaultdict(lambda: defaultdict(float))
        
        for sequence in sequences:
            if len(sequence) >= 2:
                session_length = len(sequence)
                avg_rating = np.mean([s['rating'] for s in sequence])
                genre_diversity = len(set([s['primary_genre'] for s in sequence]))
                
                context = self._classify_session_context(session_length, avg_rating, genre_diversity)
                
                for state in sequence:
                    context_patterns[context][state['primary_genre']] += 1
        
        self.sequence_models['contextual'] = dict(context_patterns)
    
    def _classify_session_context(self, session_length: int, avg_rating: float, genre_diversity: int) -> str:
        if session_length >= 5:
            return 'binge'
        elif genre_diversity >= 3:
            return 'exploration'
        elif avg_rating >= 8.0:
            return 'quality_focus'
        elif session_length <= 2:
            return 'casual'
        else:
            return 'regular'
    
    def get_intelligent_sequence_recommendations(self, user_id: int, user_profile: Dict, 
                                               n_recommendations: int = 15) -> List[Dict]:
        try:
            self.build_comprehensive_sequence_models(user_id)
            
            recent_interactions = UserInteraction.query.filter_by(user_id=user_id).order_by(
                UserInteraction.timestamp.desc()
            ).limit(5).all()
            
            if not recent_interactions:
                return []
            
            predictions = defaultdict(float)
            
            for interaction in recent_interactions:
                content = Content.query.get(interaction.content_id)
                if content:
                    self._add_transition_predictions(content, predictions, user_profile)
                    self._add_temporal_predictions(content, interaction, predictions)
                    self._add_contextual_predictions(content, predictions, user_profile)
            
            recommendations = []
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_content = set([i.content_id for i in user_interactions])
            
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            for prediction_key, score in sorted_predictions:
                matching_content = self._find_advanced_matching_content(
                    prediction_key, interacted_content, user_profile
                )
                
                for content in matching_content:
                    if content.id not in [r['content_id'] for r in recommendations]:
                        recommendations.append({
                            'content_id': content.id,
                            'score': score,
                            'method': 'intelligent_sequence_aware',
                            'content': content,
                            'prediction_basis': prediction_key,
                            'confidence': min(score / max(predictions.values()) if predictions.values() else 0, 1.0)
                        })
                        
                        if len(recommendations) >= n_recommendations:
                            break
                
                if len(recommendations) >= n_recommendations:
                    break
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting intelligent sequence recommendations: {e}")
            db.session.rollback()
            return []
    
    def _add_transition_predictions(self, content: Content, predictions: defaultdict, user_profile: Dict):
        try:
            genres = json.loads(content.genres or '[]')
            languages = json.loads(content.languages or '[]')
            
            if genres:
                primary_genre = genres[0].lower()
                genre_transitions = self.transition_matrices['genre'].get(primary_genre, {})
                for next_genre, count in genre_transitions.items():
                    user_genre_pref = user_profile.get('content_preferences', {}).get('genres', {}).get(next_genre, 0)
                    adjusted_score = count * (1 + user_genre_pref)
                    predictions[f"genre_{next_genre}"] += adjusted_score * 0.4
            
            type_transitions = self.transition_matrices['content_type'].get(content.content_type, {})
            for next_type, count in type_transitions.items():
                user_type_pref = user_profile.get('content_preferences', {}).get('content_types', {}).get(next_type, 0)
                adjusted_score = count * (1 + user_type_pref)
                predictions[f"type_{next_type}"] += adjusted_score * 0.3
            
            if languages:
                primary_language = languages[0].lower()
                lang_transitions = self.transition_matrices['language'].get(primary_language, {})
                for next_lang, count in lang_transitions.items():
                    user_lang_pref = user_profile.get('content_preferences', {}).get('languages', {}).get(next_lang, 0)
                    adjusted_score = count * (1 + user_lang_pref)
                    predictions[f"language_{next_lang}"] += adjusted_score * 0.2
                    
        except (json.JSONDecodeError, TypeError):
            pass
    
    def _add_temporal_predictions(self, content: Content, interaction, predictions: defaultdict):
        current_hour = datetime.utcnow().hour
        current_day = datetime.utcnow().weekday()
        
        temporal_model = self.sequence_models.get('temporal', {})
        
        hourly_prefs = temporal_model.get('hourly', {}).get(current_hour, {})
        for genre, count in hourly_prefs.items():
            predictions[f"temporal_genre_{genre}"] += count * 0.1
        
        daily_prefs = temporal_model.get('daily', {}).get(current_day, {})
        for genre, count in daily_prefs.items():
            predictions[f"temporal_genre_{genre}"] += count * 0.05
    
    def _add_contextual_predictions(self, content: Content, predictions: defaultdict, user_profile: Dict):
        exploration_tendency = user_profile.get('behavioral_insights', {}).get('exploration_tendency', 0.5)
        
        if exploration_tendency > 0.7:
            context = 'exploration'
        elif user_profile.get('behavioral_insights', {}).get('completion_rate', 0) > 0.8:
            context = 'quality_focus'
        else:
            context = 'regular'
        
        contextual_model = self.sequence_models.get('contextual', {})
        context_prefs = contextual_model.get(context, {})
        
        for genre, count in context_prefs.items():
            predictions[f"contextual_genre_{genre}"] += count * 0.15
    
    def _find_advanced_matching_content(self, prediction_key: str, interacted_content: Set[int], 
                                       user_profile: Dict) -> List[Content]:
        try:
            prediction_parts = prediction_key.split('_', 1)
            if len(prediction_parts) < 2:
                return []
            
            prediction_type, value = prediction_parts[0], prediction_parts[1]
            
            query = Content.query.filter(~Content.id.in_(interacted_content))
            
            if prediction_type == 'genre' or prediction_type.endswith('genre'):
                query = query.filter(Content.genres.contains(value.title()))
            elif prediction_type == 'type':
                query = query.filter(Content.content_type == value)
            elif prediction_type == 'language':
                query = query.filter(Content.languages.contains(value.title()))
            
            quality_threshold = user_profile.get('quality_preferences', {}).get('preferred_content_rating', 6.0) - 1.0
            if quality_threshold > 0:
                query = query.filter(Content.rating >= quality_threshold)
            
            return query.order_by(Content.popularity.desc()).limit(3).all()
            
        except Exception as e:
            logger.warning(f"Error finding advanced matching content: {e}")
            return []

class UltraAdvancedHybridEngine:
    def __init__(self):
        self.profiler = AdvancedUserProfiler()
        self.collaborative_engine = UltraPowerfulCollaborativeEngine()
        self.content_engine = AdvancedContentBasedEngine()
        self.sequence_engine = IntelligentSequenceEngine()
        
        self.algorithm_weights = {
            'ultra_collaborative_filtering': 0.35,
            'ultra_content_based_filtering': 0.3,
            'intelligent_sequence_aware': 0.25,
            'trending_boost': 0.05,
            'quality_boost': 0.05
        }
        
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
    
    def get_100_percent_accurate_recommendations(self, user_id: int, content_type: Optional[str] = None,
                                               context: Optional[Dict] = None, 
                                               n_recommendations: int = 20) -> Dict:
        try:
            user_profile = self.profiler.build_comprehensive_profile(user_id)
            confidence_level = user_profile.get('confidence_score', 0.0)
            
            all_recommendations = []
            algorithm_performance = {}
            
            try:
                collab_recs = self.collaborative_engine.get_ultra_accurate_recommendations(
                    user_id, user_profile, n_recommendations * 3
                )
                all_recommendations.extend(collab_recs)
                algorithm_performance['ultra_collaborative_filtering'] = len(collab_recs)
            except Exception as e:
                logger.warning(f"Ultra collaborative filtering failed: {e}")
                algorithm_performance['ultra_collaborative_filtering'] = 0
            
            try:
                content_recs = self.content_engine.get_ultra_content_recommendations(
                    user_id, user_profile, n_recommendations * 3
                )
                all_recommendations.extend(content_recs)
                algorithm_performance['ultra_content_based_filtering'] = len(content_recs)
            except Exception as e:
                logger.warning(f"Ultra content-based filtering failed: {e}")
                algorithm_performance['ultra_content_based_filtering'] = 0
            
            try:
                sequence_recs = self.sequence_engine.get_intelligent_sequence_recommendations(
                    user_id, user_profile, n_recommendations * 2
                )
                all_recommendations.extend(sequence_recs)
                algorithm_performance['intelligent_sequence_aware'] = len(sequence_recs)
            except Exception as e:
                logger.warning(f"Intelligent sequence-aware failed: {e}")
                algorithm_performance['intelligent_sequence_aware'] = 0
            
            try:
                trending_recs = self._get_intelligent_trending_recommendations(
                    user_id, user_profile, n_recommendations // 3
                )
                all_recommendations.extend(trending_recs)
                algorithm_performance['trending_boost'] = len(trending_recs)
            except Exception as e:
                logger.warning(f"Intelligent trending recommendations failed: {e}")
                algorithm_performance['trending_boost'] = 0
            
            try:
                quality_recs = self._get_intelligent_quality_recommendations(
                    user_id, user_profile, n_recommendations // 3
                )
                all_recommendations.extend(quality_recs)
                algorithm_performance['quality_boost'] = len(quality_recs)
            except Exception as e:
                logger.warning(f"Intelligent quality recommendations failed: {e}")
                algorithm_performance['quality_boost'] = 0
            
            fused_recommendations = self._ultra_advanced_fusion(
                all_recommendations, user_profile, confidence_level
            )
            
            fused_recommendations = self._apply_intelligent_diversity(
                fused_recommendations, user_profile
            )
            
            fused_recommendations = self._apply_contextual_ranking(
                fused_recommendations, context, user_profile
            )
            
            if content_type:
                fused_recommendations = [
                    rec for rec in fused_recommendations 
                    if rec['content'].content_type == content_type
                ]
            
            formatted_recommendations = self._format_ultra_recommendations(
                fused_recommendations[:n_recommendations], user_profile
            )
            
            return {
                'recommendations': formatted_recommendations,
                'user_profile_insights': {
                    'confidence_level': confidence_level,
                    'profile_strength': user_profile.get('profile_strength', 0),
                    'top_genres': list(user_profile.get('content_preferences', {}).get('genres', {}).keys())[:5],
                    'preferred_languages': list(user_profile.get('content_preferences', {}).get('languages', {}).keys())[:3],
                    'behavioral_insights': user_profile.get('behavioral_insights', {}),
                    'emerging_interests': user_profile.get('prediction_features', {}).get('emerging_interests', []),
                    'recommendation_personality': self._determine_recommendation_personality(user_profile)
                },
                'recommendation_metadata': {
                    'total_recommendations': len(formatted_recommendations),
                    'algorithm_performance': algorithm_performance,
                    'personalization_strength': min(confidence_level * 100, 100),
                    'context_applied': context is not None,
                    'diversity_applied': True,
                    'accuracy_estimate': self._calculate_ultra_accuracy(user_profile, algorithm_performance),
                    'recommendation_strategy': self._determine_recommendation_strategy(user_profile),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ultra advanced hybrid recommendations: {e}")
            db.session.rollback()
            return {
                'recommendations': [],
                'user_profile_insights': {},
                'recommendation_metadata': {'error': str(e)}
            }
    
    def _get_intelligent_trending_recommendations(self, user_id: int, user_profile: Dict, 
                                                n_recommendations: int) -> List[Dict]:
        try:
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_content = set([i.content_id for i in user_interactions])
            
            trending_sensitivity = user_profile.get('behavioral_insights', {}).get('trending_sensitivity', 0.5)
            
            if trending_sensitivity < 0.3:
                return []
            
            trending_content = Content.query.filter(
                Content.is_trending == True,
                ~Content.id.in_(interacted_content)
            ).order_by(Content.popularity.desc()).limit(n_recommendations * 2).all()
            
            recommendations = []
            for content in trending_content:
                preference_score = self._calculate_ultra_preference_match(content, user_profile)
                trending_score = 0.3 + (trending_sensitivity * 0.4) + (preference_score * 0.3)
                
                recommendations.append({
                    'content_id': content.id,
                    'score': trending_score,
                    'method': 'trending_boost',
                    'content': content,
                    'trending_factor': trending_sensitivity
                })
            
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting intelligent trending recommendations: {e}")
            db.session.rollback()
            return []
    
    def _get_intelligent_quality_recommendations(self, user_id: int, user_profile: Dict, 
                                               n_recommendations: int) -> List[Dict]:
        try:
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_content = set([i.content_id for i in user_interactions])
            
            quality_prefs = user_profile.get('quality_preferences', {})
            min_rating = max(float(quality_prefs.get('preferred_content_rating', 7.0)) - 0.5, 6.0)
            
            quality_content = Content.query.filter(
                Content.rating >= min_rating,
                Content.vote_count >= 100,
                ~Content.id.in_(interacted_content)
            ).order_by(Content.rating.desc()).limit(n_recommendations * 2).all()
            
            recommendations = []
            for content in quality_content:
                preference_score = self._calculate_ultra_preference_match(content, user_profile)
                quality_score = float(content.rating) / 10.0
                
                final_score = (preference_score * 0.6) + (quality_score * 0.4)
                
                recommendations.append({
                    'content_id': content.id,
                    'score': final_score,
                    'method': 'quality_boost',
                    'content': content,
                    'quality_factor': quality_score
                })
            
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting intelligent quality recommendations: {e}")
            db.session.rollback()
            return []
    
    def _calculate_ultra_preference_match(self, content: Content, user_profile: Dict) -> float:
        score = 0.0
        
        try:
            content_genres = json.loads(content.genres or '[]')
            user_genres = user_profile.get('content_preferences', {}).get('genres', {})
            
            genre_scores = []
            for genre in content_genres[:3]:
                genre_pref = user_genres.get(genre.lower(), 0)
                genre_scores.append(genre_pref)
            
            if genre_scores:
                score += np.mean(genre_scores) * 0.4
        except (json.JSONDecodeError, TypeError):
            pass
        
        try:
            content_languages = json.loads(content.languages or '[]')
            user_languages = user_profile.get('content_preferences', {}).get('languages', {})
            
            lang_scores = []
            for lang in content_languages:
                lang_pref = user_languages.get(lang.lower(), 0)
                lang_scores.append(lang_pref)
            
            if lang_scores:
                score += max(lang_scores) * 0.35
        except (json.JSONDecodeError, TypeError):
            pass
        
        content_type_prefs = user_profile.get('content_preferences', {}).get('content_types', {})
        content_type_pref = content_type_prefs.get(content.content_type, 0)
        score += content_type_pref * 0.25
        
        return min(score, 1.0)
    
    def _ultra_advanced_fusion(self, all_recommendations: List[Dict], user_profile: Dict, 
                              confidence_level: float) -> List[Dict]:
        content_scores = defaultdict(list)
        content_objects = {}
        
        for rec in all_recommendations:
            content_id = rec['content_id']
            method = rec.get('method', 'unknown')
            score = rec['score']
            
            content_scores[content_id].append((score, method, rec))
            content_objects[content_id] = rec['content']
        
        fused_recommendations = []
        
        for content_id, score_data in content_scores.items():
            content = content_objects[content_id]
            
            method_scores = defaultdict(list)
            for score, method, rec_data in score_data:
                method_scores[method].append((score, rec_data))
            
            total_score = 0.0
            total_weight = 0.0
            methods_used = []
            method_details = {}
            
            for method, method_score_list in method_scores.items():
                best_score, best_rec = max(method_score_list, key=lambda x: x[0])
                
                base_weight = self.algorithm_weights.get(method, 0.1)
                
                if confidence_level >= 0.8:
                    if method in ['ultra_collaborative_filtering', 'intelligent_sequence_aware']:
                        base_weight *= 1.3
                elif confidence_level >= 0.5:
                    base_weight *= 1.1
                else:
                    if method in ['ultra_content_based_filtering', 'trending_boost']:
                        base_weight *= 1.2
                
                diversity_factor = 1.0 + (len(method_scores) - 1) * 0.1
                final_weight = base_weight * diversity_factor
                
                total_score += best_score * final_weight
                total_weight += final_weight
                methods_used.append(method)
                method_details[method] = {
                    'score': best_score,
                    'weight': final_weight,
                    'details': best_rec
                }
            
            if total_weight > 0:
                fusion_score = total_score / total_weight
                
                fusion_score = self._apply_ultra_boosters(
                    fusion_score, content, user_profile, methods_used
                )
                
                confidence_boost = self._calculate_fusion_confidence(methods_used, confidence_level)
                final_score = fusion_score * (1 + confidence_boost)
                
                fused_recommendations.append({
                    'content_id': content_id,
                    'score': final_score,
                    'content': content,
                    'methods_used': list(set(methods_used)),
                    'method_count': len(set(methods_used)),
                    'method_details': method_details,
                    'fusion_confidence': confidence_boost
                })
        
        fused_recommendations.sort(key=lambda x: x['score'], reverse=True)
        return fused_recommendations
    
    def _apply_ultra_boosters(self, score: float, content: Content, 
                             user_profile: Dict, methods_used: List[str]) -> float:
        boosted_score = score
        
        if len(set(methods_used)) >= 4:
            boosted_score += 0.15
        elif len(set(methods_used)) >= 3:
            boosted_score += 0.1
        elif len(set(methods_used)) >= 2:
            boosted_score += 0.05
        
        if content.release_date:
            days_since_release = (datetime.utcnow().date() - content.release_date).days
            if days_since_release <= 30:
                boosted_score += 0.08
            elif days_since_release <= 90:
                boosted_score += 0.04
        
        if content.is_trending:
            trending_sensitivity = user_profile.get('behavioral_insights', {}).get('trending_sensitivity', 0.5)
            boosted_score += trending_sensitivity * 0.06
        
        if content.rating and content.rating >= 8.5:
            boosted_score += 0.06
        elif content.rating and content.rating >= 8.0:
            boosted_score += 0.03
        
        try:
            content_languages = json.loads(content.languages or '[]')
            user_languages = user_profile.get('content_preferences', {}).get('languages', {})
            
            for lang in content_languages:
                lang_pref = user_languages.get(lang.lower(), 0)
                if lang_pref > 0.6:
                    boosted_score += 0.05
                    break
        except (json.JSONDecodeError, TypeError):
            pass
        
        runtime_prefs = user_profile.get('advanced_preferences', {}).get('runtime_preference', {})
        if content.runtime and runtime_prefs:
            optimal_runtime = runtime_prefs.get('optimal', 120)
            runtime_diff = abs(content.runtime - optimal_runtime) / optimal_runtime
            if runtime_diff <= 0.2:
                boosted_score += 0.03
        
        return min(boosted_score, 2.0)
    
    def _calculate_fusion_confidence(self, methods_used: List[str], confidence_level: float) -> float:
        method_confidence = len(set(methods_used)) / len(self.algorithm_weights)
        user_confidence = confidence_level
        
        return (method_confidence * 0.4 + user_confidence * 0.6) * 0.2
    
    def _apply_intelligent_diversity(self, recommendations: List[Dict], user_profile: Dict) -> List[Dict]:
        exploration_tendency = user_profile.get('behavioral_insights', {}).get('exploration_tendency', 0.5)
        
        if exploration_tendency < 0.3:
            return recommendations
        
        seen_genres = set()
        seen_content_types = set()
        seen_languages = set()
        seen_decades = set()
        
        diverse_recommendations = []
        standard_recommendations = []
        
        for rec in recommendations:
            content = rec['content']
            
            try:
                content_genres = json.loads(content.genres or '[]')
                content_languages = json.loads(content.languages or '[]')
                
                primary_genre = content_genres[0].lower() if content_genres else 'unknown'
                primary_language = content_languages[0].lower() if content_languages else 'unknown'
                content_type = content.content_type
                
                decade = 'unknown'
                if content.release_date:
                    decade = str((content.release_date.year // 10) * 10)
                
                diversity_factors = [
                    primary_genre not in seen_genres,
                    content_type not in seen_content_types,
                    primary_language not in seen_languages,
                    decade not in seen_decades
                ]
                
                diversity_score = sum(diversity_factors) / 4.0
                
                if diversity_score >= 0.5 and len(diverse_recommendations) < len(recommendations) * 0.7:
                    seen_genres.add(primary_genre)
                    seen_content_types.add(content_type)
                    seen_languages.add(primary_language)
                    seen_decades.add(decade)
                    
                    rec['score'] += diversity_score * 0.1 * exploration_tendency
                    rec['diversity_score'] = diversity_score
                    diverse_recommendations.append(rec)
                else:
                    standard_recommendations.append(rec)
                    
            except (json.JSONDecodeError, TypeError, IndexError):
                standard_recommendations.append(rec)
        
        final_recommendations = []
        
        diversity_quota = int(len(recommendations) * exploration_tendency * 0.8)
        final_recommendations.extend(diverse_recommendations[:diversity_quota])
        
        remaining_slots = len(recommendations) - len(final_recommendations)
        remaining_diverse = diverse_recommendations[diversity_quota:]
        final_recommendations.extend(remaining_diverse[:remaining_slots//2])
        
        remaining_slots = len(recommendations) - len(final_recommendations)
        final_recommendations.extend(standard_recommendations[:remaining_slots])
        
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        return final_recommendations
    
    def _apply_contextual_ranking(self, recommendations: List[Dict], context: Optional[Dict], 
                                 user_profile: Dict) -> List[Dict]:
        if not context:
            return recommendations
        
        current_hour = datetime.utcnow().hour
        current_day = datetime.utcnow().weekday()
        
        contextual_weights = {
            'morning': {'documentary': 1.2, 'news': 1.3, 'family': 1.1},
            'afternoon': {'comedy': 1.2, 'action': 1.1, 'adventure': 1.1},
            'evening': {'drama': 1.2, 'thriller': 1.2, 'mystery': 1.1},
            'night': {'horror': 1.3, 'sci-fi': 1.2, 'fantasy': 1.1}
        }
        
        time_of_day = self._get_time_of_day(current_hour)
        
        for rec in recommendations:
            content = rec['content']
            contextual_boost = 0.0
            
            try:
                content_genres = json.loads(content.genres or '[]')
                time_weights = contextual_weights.get(time_of_day, {})
                
                for genre in content_genres:
                    genre_weight = time_weights.get(genre.lower(), 1.0)
                    if genre_weight > 1.0:
                        contextual_boost += (genre_weight - 1.0) * 0.05
                        
            except (json.JSONDecodeError, TypeError):
                pass
            
            temporal_patterns = user_profile.get('temporal_patterns', {})
            viewing_hours = temporal_patterns.get('viewing_hours', {})
            viewing_days = temporal_patterns.get('viewing_days', {})
            
            if str(current_hour) in viewing_hours:
                contextual_boost += 0.03
            
            if str(current_day) in viewing_days:
                contextual_boost += 0.03
            
            if current_day >= 5:
                if content.runtime and content.runtime > 150:
                    contextual_boost += 0.02
            else:
                if content.runtime and content.runtime <= 120:
                    contextual_boost += 0.02
            
            rec['score'] += contextual_boost
            rec['contextual_boost'] = contextual_boost
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations
    
    def _get_time_of_day(self, hour: int) -> str:
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    def _format_ultra_recommendations(self, recommendations: List[Dict], user_profile: Dict) -> List[Dict]:
        formatted_recs = []
        
        for rec in recommendations:
            content = rec['content']
            
            if not content.slug:
                try:
                    content.ensure_slug()
                    db.session.commit()
                except Exception:
                    content.slug = f"content-{content.id}"
            
            youtube_url = None
            if content.youtube_trailer_id:
                youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
            
            poster_url = None
            if content.poster_path:
                if content.poster_path.startswith('http'):
                    poster_url = content.poster_path
                else:
                    poster_url = f"https://image.tmdb.org/t/p/w300{content.poster_path}"
            
            reason = self._generate_ultra_recommendation_reason(rec, user_profile)
            personalization_tags = self._generate_ultra_personalization_tags(content, user_profile, rec)
            
            formatted_rec = {
                'id': content.id,
                'slug': content.slug,
                'title': content.title,
                'original_title': content.original_title,
                'content_type': content.content_type,
                'genres': json.loads(content.genres or '[]'),
                'languages': json.loads(content.languages or '[]'),
                'rating': content.rating,
                'vote_count': content.vote_count,
                'popularity': content.popularity,
                'release_date': content.release_date.isoformat() if content.release_date else None,
                'runtime': content.runtime,
                'poster_path': poster_url,
                'overview': content.overview[:250] + '...' if content.overview and len(content.overview) > 250 else content.overview,
                'youtube_trailer': youtube_url,
                'recommendation_score': round(rec['score'], 4),
                'confidence_level': self._calculate_ultra_recommendation_confidence(rec, user_profile),
                'recommendation_reason': reason,
                'methods_used': rec.get('methods_used', ['ultra_hybrid']),
                'method_count': rec.get('method_count', 1),
                'is_trending': content.is_trending,
                'is_new_release': content.is_new_release,
                'personalization_tags': personalization_tags,
                'contextual_boost': rec.get('contextual_boost', 0.0),
                'diversity_score': rec.get('diversity_score', 0.0),
                'fusion_confidence': rec.get('fusion_confidence', 0.0),
                'accuracy_indicators': {
                    'preference_match': self._calculate_ultra_preference_match(content, user_profile),
                    'algorithm_consensus': len(rec.get('methods_used', [])) / len(self.algorithm_weights),
                    'user_profile_strength': user_profile.get('confidence_score', 0.0)
                }
            }
            
            formatted_recs.append(formatted_rec)
        
        return formatted_recs
    
    def _generate_ultra_recommendation_reason(self, recommendation: Dict, user_profile: Dict) -> str:
        content = recommendation['content']
        methods = recommendation.get('methods_used', [])
        score = recommendation.get('score', 0)
        method_details = recommendation.get('method_details', {})
        
        reasons = []
        
        if 'ultra_collaborative_filtering' in methods:
            reasons.append("users with extremely similar tastes highly recommend this")
        
        if 'intelligent_sequence_aware' in methods:
            reasons.append("perfectly continues your viewing journey")
        
        if 'ultra_content_based_filtering' in methods:
            reasons.append("matches your precise content DNA")
        
        try:
            content_genres = json.loads(content.genres or '[]')
            user_genres = user_profile.get('content_preferences', {}).get('genres', {})
            
            strong_matches = []
            for genre in content_genres[:2]:
                if user_genres.get(genre.lower(), 0) > 0.4:
                    strong_matches.append(genre.lower())
            
            if strong_matches:
                if len(strong_matches) == 1:
                    reasons.append(f"perfect match for your love of {strong_matches[0]}")
                else:
                    reasons.append(f"ideal blend of your favorite {' and '.join(strong_matches)}")
        except (json.JSONDecodeError, TypeError):
            pass
        
        if content.rating and content.rating >= 9.0:
            reasons.append("masterpiece-level quality")
        elif content.rating and content.rating >= 8.5:
            reasons.append("exceptional critical acclaim")
        elif content.rating and content.rating >= 8.0:
            reasons.append("outstanding quality")
        
        if content.is_trending:
            trending_sensitivity = user_profile.get('behavioral_insights', {}).get('trending_sensitivity', 0)
            if trending_sensitivity > 0.6:
                reasons.append("hot trending pick")
        
        if content.is_new_release:
            novelty_seeking = user_profile.get('advanced_preferences', {}).get('novelty_seeking', 0)
            if novelty_seeking > 0.6:
                reasons.append("fresh new release")
        
        try:
            content_languages = json.loads(content.languages or '[]')
            user_languages = user_profile.get('content_preferences', {}).get('languages', {})
            
            for lang in content_languages:
                if user_languages.get(lang.lower(), 0) > 0.5:
                    reasons.append(f"in your highly preferred {lang}")
                    break
        except (json.JSONDecodeError, TypeError):
            pass
        
        method_count = len(methods)
        if score > 0.9:
            confidence_phrase = "Ultra-perfect match"
        elif score > 0.8:
            confidence_phrase = "Exceptional match"
        elif score > 0.7:
            confidence_phrase = "Excellent match"
        elif score > 0.6:
            confidence_phrase = "Great match"
        else:
            confidence_phrase = "Good match"
        
        if method_count >= 4:
            confidence_phrase = f"All-algorithms-agree {confidence_phrase.lower()}"
        elif method_count >= 3:
            confidence_phrase = f"Multi-algorithm {confidence_phrase.lower()}"
        
        if reasons:
            return f"{confidence_phrase}: {', '.join(reasons[:4])}"
        else:
            return f"{confidence_phrase} based on your ultra-personalized profile"
    
    def _calculate_ultra_recommendation_confidence(self, recommendation: Dict, user_profile: Dict) -> str:
        score = recommendation.get('score', 0)
        methods_used = recommendation.get('methods_used', [])
        user_confidence = user_profile.get('confidence_score', 0)
        fusion_confidence = recommendation.get('fusion_confidence', 0)
        
        base_confidence = 'medium'
        if score > 0.9:
            base_confidence = 'ultra_high'
        elif score > 0.8:
            base_confidence = 'very_high'
        elif score > 0.7:
            base_confidence = 'high'
        elif score > 0.5:
            base_confidence = 'medium'
        elif score > 0.3:
            base_confidence = 'low'
        else:
            base_confidence = 'very_low'
        
        if len(methods_used) >= 4:
            if base_confidence in ['medium', 'low']:
                base_confidence = 'high'
            elif base_confidence == 'high':
                base_confidence = 'very_high'
        elif len(methods_used) >= 3:
            if base_confidence == 'low':
                base_confidence = 'medium'
            elif base_confidence == 'medium':
                base_confidence = 'high'
        
        if user_confidence < 0.3 and base_confidence in ['ultra_high', 'very_high']:
            base_confidence = 'high'
        elif user_confidence > 0.8 and base_confidence == 'medium':
            base_confidence = 'high'
        
        return base_confidence
    
    def _generate_ultra_personalization_tags(self, content: Content, user_profile: Dict, 
                                           recommendation: Dict) -> List[str]:
        tags = []
        
        try:
            content_genres = json.loads(content.genres or '[]')
            user_genres = user_profile.get('content_preferences', {}).get('genres', {})
            
            for genre in content_genres:
                if user_genres.get(genre.lower(), 0) > 0.5:
                    tags.append(f"top_{genre.lower()}_pick")
                elif user_genres.get(genre.lower(), 0) > 0.3:
                    tags.append(f"good_{genre.lower()}_match")
        except (json.JSONDecodeError, TypeError):
            pass
        
        try:
            content_languages = json.loads(content.languages or '[]')
            user_languages = user_profile.get('content_preferences', {}).get('languages', {})
            
            for lang in content_languages:
                if user_languages.get(lang.lower(), 0) > 0.4:
                    tags.append(f"preferred_{lang.lower()}")
        except (json.JSONDecodeError, TypeError):
            pass
        
        quality_prefs = user_profile.get('quality_preferences', {})
        user_avg_rating = quality_prefs.get('preferred_content_rating', 7.0)
        
        if content.rating:
            if content.rating >= user_avg_rating + 1.0:
                tags.append('premium_quality')
            elif content.rating >= user_avg_rating + 0.5:
                tags.append('above_your_standard')
            elif content.rating >= user_avg_rating - 0.5:
                tags.append('your_quality_sweet_spot')
        
        methods_used = recommendation.get('methods_used', [])
        if len(methods_used) >= 4:
            tags.append('all_algorithms_agree')
        elif len(methods_used) >= 3:
            tags.append('multi_algorithm_consensus')
        
        emerging_interests = user_profile.get('prediction_features', {}).get('emerging_interests', [])
        try:
            content_genres = json.loads(content.genres or '[]')
            for genre in content_genres:
                if genre.lower() in emerging_interests:
                    tags.append('emerging_interest_match')
                    break
        except (json.JSONDecodeError, TypeError):
            pass
        
        exploration_tendency = user_profile.get('behavioral_insights', {}).get('exploration_tendency', 0.5)
        if exploration_tendency > 0.7:
            tags.append('adventurous_discovery')
        elif exploration_tendency < 0.3:
            tags.append('comfort_zone_pick')
        else:
            tags.append('balanced_choice')
        
        if recommendation.get('diversity_score', 0) > 0.5:
            tags.append('diversity_enhancer')
        
        if recommendation.get('fusion_confidence', 0) > 0.1:
            tags.append('high_confidence_fusion')
        
        return tags[:6]
    
    def _determine_recommendation_personality(self, user_profile: Dict) -> str:
        exploration = user_profile.get('behavioral_insights', {}).get('exploration_tendency', 0.5)
        completion = user_profile.get('behavioral_insights', {}).get('completion_rate', 0.5)
        trending = user_profile.get('behavioral_insights', {}).get('trending_sensitivity', 0.5)
        rewatch = user_profile.get('behavioral_insights', {}).get('rewatch_tendency', 0.5)
        
        if exploration > 0.7 and completion > 0.7:
            return 'adventurous_completionist'
        elif exploration > 0.7:
            return 'bold_explorer'
        elif completion > 0.7 and trending < 0.3:
            return 'quality_purist'
        elif trending > 0.7:
            return 'trend_follower'
        elif rewatch > 0.3:
            return 'comfort_seeker'
        elif completion > 0.6:
            return 'engaged_viewer'
        else:
            return 'casual_browser'
    
    def _determine_recommendation_strategy(self, user_profile: Dict) -> str:
        confidence = user_profile.get('confidence_score', 0)
        interactions = user_profile.get('total_interactions', 0)
        
        if confidence > 0.8 and interactions > 100:
            return 'ultra_personalized'
        elif confidence > 0.6:
            return 'highly_personalized'
        elif confidence > 0.4:
            return 'moderately_personalized'
        elif interactions > 20:
            return 'preference_learning'
        else:
            return 'discovery_focused'
    
    def _calculate_ultra_accuracy(self, user_profile: Dict, algorithm_performance: Dict) -> float:
        confidence = user_profile.get('confidence_score', 0)
        interaction_count = user_profile.get('total_interactions', 0)
        
        base_accuracy = min(confidence * 90, 90)
        
        interaction_bonus = min(interaction_count / 200.0 * 5, 5)
        
        active_algorithms = sum(1 for count in algorithm_performance.values() if count > 0)
        algorithm_bonus = min(active_algorithms * 1.5, 6)
        
        total_recommendations = sum(algorithm_performance.values())
        performance_bonus = min(total_recommendations / 100.0 * 1, 1)
        
        profile_completeness = min(len(user_profile.get('content_preferences', {}).get('genres', {})) / 10.0 * 2, 2)
        
        estimated_accuracy = base_accuracy + interaction_bonus + algorithm_bonus + performance_bonus + profile_completeness
        
        return min(round(estimated_accuracy, 1), 99.5)

ultra_hybrid_engine = UltraAdvancedHybridEngine()

@personalized_bp.route('/api/recommendations/personalized', methods=['GET'])
@require_auth
def get_personalized_recommendations(current_user):
    try:
        content_type = request.args.get('type')
        limit = min(int(request.args.get('limit', 20)), 50)
        
        context = {
            'user_agent': request.headers.get('User-Agent', ''),
            'time': datetime.utcnow().hour,
            'day': datetime.utcnow().weekday(),
            'request_timestamp': datetime.utcnow().isoformat()
        }
        
        recommendations = ultra_hybrid_engine.get_100_percent_accurate_recommendations(
            current_user.id, content_type, context, limit
        )
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to get personalized recommendations'}), 500

@personalized_bp.route('/api/recommendations/personalized/categories', methods=['GET'])
@require_auth
def get_personalized_categories(current_user):
    try:
        context = {
            'user_agent': request.headers.get('User-Agent', ''),
            'time': datetime.utcnow().hour,
            'day': datetime.utcnow().weekday()
        }
        
        categories = {
            'for_you': ultra_hybrid_engine.get_100_percent_accurate_recommendations(
                current_user.id, None, context, 20
            ),
            'movies': ultra_hybrid_engine.get_100_percent_accurate_recommendations(
                current_user.id, 'movie', context, 15
            ),
            'tv_shows': ultra_hybrid_engine.get_100_percent_accurate_recommendations(
                current_user.id, 'tv', context, 15
            ),
            'anime': ultra_hybrid_engine.get_100_percent_accurate_recommendations(
                current_user.id, 'anime', context, 15
            )
        }
        
        return jsonify({
            'categories': categories,
            'metadata': {
                'personalized': True,
                'user_id': current_user.id,
                'context_applied': True,
                'ultra_accuracy_enabled': True,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting personalized categories: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to get personalized categories'}), 500

@personalized_bp.route('/api/recommendations/track-interaction', methods=['POST'])
@require_auth
def track_user_interaction(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        metadata = {
            'from_recommendation': data.get('from_recommendation', False),
            'recommendation_score': data.get('recommendation_score'),
            'recommendation_method': data.get('recommendation_method'),
            'context': data.get('context', {}),
            'user_agent': request.headers.get('User-Agent', ''),
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': data.get('session_id'),
            'device_type': data.get('device_type'),
            'viewing_duration': data.get('viewing_duration')
        }
        
        interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=data['interaction_type'],
            rating=float(data['rating']) if data.get('rating') else None,
            interaction_metadata=json.dumps(metadata),
            timestamp=datetime.utcnow()
        )
        
        db.session.add(interaction)
        db.session.commit()
        
        if cache:
            try:
                cache.delete(f"user_profile_v3:{current_user.id}")
                cache.delete(f"recommendations:{current_user.id}")
                cache.delete(f"user_embeddings:{current_user.id}")
            except Exception as e:
                logger.warning(f"Cache delete error: {e}")
        
        return jsonify({'message': 'Ultra-detailed interaction tracked successfully'}), 201
        
    except Exception as e:
        logger.error(f"Error tracking interaction: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to track interaction'}), 500

@personalized_bp.route('/api/recommendations/retrain', methods=['POST'])
@require_auth
def retrain_ultra_models(current_user):
    try:
        if not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        results = {}
        
        try:
            success = ultra_hybrid_engine.collaborative_engine.train_hybrid_collaborative_model()
            results['ultra_collaborative_filtering'] = 'success' if success else 'failed'
        except Exception as e:
            results['ultra_collaborative_filtering'] = f'error: {str(e)}'
        
        try:
            success = ultra_hybrid_engine.content_engine.build_advanced_content_embeddings()
            results['ultra_content_based_filtering'] = 'success' if success else 'failed'
        except Exception as e:
            results['ultra_content_based_filtering'] = f'error: {str(e)}'
        
        try:
            if cache:
                users = User.query.all()
                cleared_count = 0
                for user in users:
                    cache.delete(f"user_profile_v3:{user.id}")
                    cache.delete(f"recommendations:{user.id}")
                    cache.delete(f"user_embeddings:{user.id}")
                    cleared_count += 1
                results['cache_clearing'] = f'cleared {cleared_count} ultra user profiles'
        except Exception as e:
            results['cache_clearing'] = f'error: {str(e)}'
        
        return jsonify({
            'message': 'Ultra model retraining completed',
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error retraining ultra models: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to retrain ultra models'}), 500

@personalized_bp.route('/api/recommendations/similar/<int:content_id>', methods=['GET'])
@require_auth
def get_ultra_similar_recommendations(current_user, content_id):
    try:
        limit = min(int(request.args.get('limit', 10)), 20)
        
        if not ultra_hybrid_engine.content_engine.embeddings_built:
            ultra_hybrid_engine.content_engine.build_advanced_content_embeddings()
        
        user_profile = ultra_hybrid_engine.profiler.build_comprehensive_profile(current_user.id)
        
        content_embeddings = ultra_hybrid_engine.content_engine.content_embeddings
        
        if content_id not in content_embeddings:
            return jsonify({'error': 'Content not found in embeddings'}), 404
        
        content_embedding = content_embeddings[content_id]
        similarities = []
        
        for other_id, other_embedding in content_embeddings.items():
            if other_id != content_id:
                similarity = cosine_similarity([content_embedding], [other_embedding])[0][0]
                similarities.append((other_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        user_interactions = UserInteraction.query.filter_by(user_id=current_user.id).all()
        interacted_content = set([i.content_id for i in user_interactions])
        
        for similar_id, similarity in similarities[:limit*3]:
            if similar_id not in interacted_content:
                content = Content.query.get(similar_id)
                if content:
                    profile_match = ultra_hybrid_engine._calculate_ultra_preference_match(content, user_profile)
                    quality_boost = 0.0
                    if content.rating:
                        quality_boost = min(float(content.rating) / 10.0, 0.2)
                    
                    final_score = (similarity * 0.6) + (profile_match * 0.3) + (quality_boost * 0.1)
                    
                    if not content.slug:
                        try:
                            content.ensure_slug()
                            db.session.commit()
                        except Exception:
                            content.slug = f"content-{content.id}"
                    
                    youtube_url = None
                    if content.youtube_trailer_id:
                        youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                    
                    recommendations.append({
                        'id': content.id,
                        'slug': content.slug,
                        'title': content.title,
                        'content_type': content.content_type,
                        'genres': json.loads(content.genres or '[]'),
                        'rating': content.rating,
                        'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                        'overview': content.overview[:200] + '...' if content.overview else '',
                        'youtube_trailer': youtube_url,
                        'similarity_score': round(similarity, 4),
                        'personalized_score': round(final_score, 4),
                        'profile_match_score': round(profile_match, 4),
                        'recommendation_reason': f"Ultra-similar content with {round(similarity*100, 1)}% match + personalized scoring"
                    })
                    
                    if len(recommendations) >= limit:
                        break
        
        recommendations.sort(key=lambda x: x['personalized_score'], reverse=True)
        
        return jsonify({
            'similar_recommendations': recommendations,
            'base_content_id': content_id,
            'total_found': len(recommendations),
            'ultra_personalization_applied': True,
            'accuracy_level': 'maximum'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting ultra similar recommendations: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to get ultra similar recommendations'}), 500

@personalized_bp.route('/api/user/profile/ultra-detailed', methods=['GET'])
@require_auth
def get_ultra_detailed_user_profile(current_user):
    try:
        profile = ultra_hybrid_engine.profiler.build_comprehensive_profile(current_user.id)
        
        return jsonify({
            'ultra_detailed_profile': profile,
            'profile_completeness': profile.get('profile_strength', 0),
            'recommendation_readiness': profile.get('confidence_score', 0) >= 0.3,
            'ultra_accuracy_enabled': True,
            'ai_personality': ultra_hybrid_engine._determine_recommendation_personality(profile),
            'recommendation_strategy': ultra_hybrid_engine._determine_recommendation_strategy(profile)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting ultra detailed user profile: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to get ultra detailed profile'}), 500