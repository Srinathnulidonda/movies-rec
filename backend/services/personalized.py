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
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated_function

class UserInteractionTracker:
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
    
    def track_interaction(self, user_id: int, content_id: int, interaction_type: str, 
                         rating: Optional[float] = None, metadata: Optional[Dict] = None):
        try:
            existing_interaction = UserInteraction.query.filter_by(
                user_id=user_id,
                content_id=content_id,
                interaction_type=interaction_type
            ).first()
            
            if existing_interaction and interaction_type in ['rating', 'view', 'rewatch']:
                existing_interaction.rating = rating
                existing_interaction.timestamp = datetime.utcnow()
                if metadata:
                    existing_interaction.interaction_metadata = json.dumps(metadata)
            else:
                interaction = UserInteraction(
                    user_id=user_id,
                    content_id=content_id,
                    interaction_type=interaction_type,
                    rating=rating,
                    interaction_metadata=json.dumps(metadata or {}),
                    timestamp=datetime.utcnow()
                )
                db.session.add(interaction)
            
            db.session.commit()
            
            if cache:
                cache.delete(f"user_profile:{user_id}")
                cache.delete(f"recommendations:{user_id}")
                cache.delete(f"user_embeddings:{user_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error tracking interaction: {e}")
            db.session.rollback()
            return False
    
    def get_weighted_interactions(self, user_id: int) -> List[Dict]:
        interactions = UserInteraction.query.filter_by(user_id=user_id).order_by(
            UserInteraction.timestamp.desc()
        ).all()
        
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
                rating_boost = (interaction.rating / 5.0) * 1.3
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

class AdvancedUserProfiler:
    def __init__(self):
        self.tracker = UserInteractionTracker()
        self.genre_vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
        self.content_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        
    def build_comprehensive_profile(self, user_id: int) -> Dict[str, Any]:
        cache_key = f"user_profile_v2:{user_id}"
        if cache:
            cached = cache.get(cache_key)
            if cached:
                return cached
        
        weighted_interactions = self.tracker.get_weighted_interactions(user_id)
        
        if not weighted_interactions:
            return self._get_default_profile(user_id)
        
        profile = {
            'user_id': user_id,
            'total_interactions': len(weighted_interactions),
            'profile_strength': 0.0,
            'confidence_score': 0.0,
            
            'content_preferences': {
                'genres': defaultdict(float),
                'languages': defaultdict(float),
                'content_types': defaultdict(float),
                'directors': defaultdict(float),
                'actors': defaultdict(float),
                'studios': defaultdict(float)
            },
            
            'quality_preferences': {
                'avg_rating_given': 0.0,
                'preferred_content_rating': 0.0,
                'quality_tolerance': 0.0,
                'high_quality_bias': 0.0
            },
            
            'temporal_patterns': {
                'viewing_hours': defaultdict(int),
                'viewing_days': defaultdict(int),
                'binge_patterns': [],
                'seasonal_preferences': defaultdict(float)
            },
            
            'behavioral_insights': {
                'exploration_tendency': 0.0,
                'completion_rate': 0.0,
                'rewatch_tendency': 0.0,
                'social_influence': 0.0,
                'trending_sensitivity': 0.0
            },
            
            'sequence_patterns': {
                'genre_transitions': defaultdict(float),
                'mood_patterns': [],
                'viewing_context': defaultdict(float)
            },
            
            'prediction_features': {
                'next_likely_genres': [],
                'declining_interests': [],
                'emerging_interests': [],
                'recommendation_receptivity': 0.0
            },
            
            'contextual_preferences': {
                'device_preferences': defaultdict(float),
                'time_context': defaultdict(float),
                'mood_indicators': defaultdict(float)
            }
        }
        
        self._extract_content_preferences(profile, weighted_interactions)
        self._extract_quality_preferences(profile, weighted_interactions)
        self._extract_temporal_patterns(profile, weighted_interactions)
        self._extract_behavioral_insights(profile, weighted_interactions)
        self._extract_sequence_patterns(profile, weighted_interactions)
        self._calculate_prediction_features(profile, weighted_interactions)
        self._extract_contextual_preferences(profile, weighted_interactions)
        self._calculate_confidence_scores(profile, weighted_interactions)
        
        profile = self._serialize_profile(profile)
        
        if cache:
            cache.set(cache_key, profile, timeout=1800)
        
        return profile
    
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
        
        for pref_type in ['genres', 'languages', 'content_types']:
            for key in profile['content_preferences'][pref_type]:
                profile['content_preferences'][pref_type][key] /= total_weight
    
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
                ratings_given.append(interaction.rating)
            
            if content and content.rating:
                content_ratings.append(content.rating)
        
        if ratings_given:
            profile['quality_preferences']['avg_rating_given'] = np.mean(ratings_given)
            profile['quality_preferences']['quality_tolerance'] = np.std(ratings_given)
            profile['quality_preferences']['high_quality_bias'] = len([r for r in ratings_given if r >= 4]) / len(ratings_given)
        
        if content_ratings:
            profile['quality_preferences']['preferred_content_rating'] = np.mean(content_ratings)
    
    def _extract_temporal_patterns(self, profile: Dict, weighted_interactions: List[Dict]):
        for wi in weighted_interactions:
            interaction = wi['interaction']
            timestamp = interaction.timestamp
            
            hour = timestamp.hour
            day = timestamp.weekday()
            month = timestamp.month
            
            profile['temporal_patterns']['viewing_hours'][hour] += wi['weight']
            profile['temporal_patterns']['viewing_days'][day] += wi['weight']
            profile['temporal_patterns']['seasonal_preferences'][month] += wi['weight']
        
        self._detect_binge_patterns(profile, weighted_interactions)
    
    def _detect_binge_patterns(self, profile: Dict, weighted_interactions: List[Dict]):
        content_ids = [wi['interaction'].content_id for wi in weighted_interactions]
        contents = Content.query.filter(Content.id.in_(content_ids)).all()
        content_map = {c.id: c for c in contents}
        
        daily_sessions = defaultdict(list)
        
        for wi in weighted_interactions:
            interaction = wi['interaction']
            date_key = interaction.timestamp.date()
            content = content_map.get(interaction.content_id)
            
            if content:
                daily_sessions[date_key].append({
                    'content': content,
                    'timestamp': interaction.timestamp,
                    'type': interaction.interaction_type
                })
        
        binge_sessions = []
        for date, sessions in daily_sessions.items():
            if len(sessions) >= 3:
                content_types = [s['content'].content_type for s in sessions]
                if len(set(content_types)) <= 2:
                    binge_sessions.append({
                        'date': date.isoformat(),
                        'content_count': len(sessions),
                        'primary_type': Counter(content_types).most_common(1)[0][0]
                    })
        
        profile['temporal_patterns']['binge_patterns'] = binge_sessions[-10:]
    
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
        trending_count = Content.query.filter(
            Content.id.in_(content_ids),
            Content.is_trending == True
        ).count()
        
        if content_ids:
            profile['behavioral_insights']['trending_sensitivity'] = trending_count / len(set(content_ids))
    
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
                        profile['sequence_patterns']['genre_transitions'][transition] += 1
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
                elif recent_count < older_count * 0.5:
                    declining.append(genre)
            
            profile['prediction_features']['emerging_interests'] = emerging
            profile['prediction_features']['declining_interests'] = declining
            
            profile['prediction_features']['next_likely_genres'] = list(recent_genres.keys())[:5]
    
    def _extract_contextual_preferences(self, profile: Dict, weighted_interactions: List[Dict]):
        for wi in weighted_interactions:
            interaction = wi['interaction']
            
            try:
                metadata = json.loads(interaction.interaction_metadata or '{}')
                
                device = metadata.get('device', 'unknown')
                if device != 'unknown':
                    profile['contextual_preferences']['device_preferences'][device] += wi['weight']
                
                time_context = metadata.get('time_context', 'unknown')
                if time_context != 'unknown':
                    profile['contextual_preferences']['time_context'][time_context] += wi['weight']
                
            except (json.JSONDecodeError, TypeError):
                pass
    
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

class CollaborativeFilteringEngine:
    def __init__(self):
        self.user_similarity_cache = {}
        self.item_similarity_cache = {}
        self.model_updated = False
        
    def build_user_item_matrix(self):
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
            
            for interaction in interactions:
                user_idx = user_mapping[interaction.user_id]
                item_idx = item_mapping[interaction.content_id]
                
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
                
                rating_matrix[user_idx, item_idx] = rating
            
            return rating_matrix.tocsr(), user_mapping, item_mapping
            
        except Exception as e:
            logger.error(f"Error building user-item matrix: {e}")
            return None, None, None
    
    def train_collaborative_model(self):
        try:
            rating_matrix, user_mapping, item_mapping = self.build_user_item_matrix()
            
            if rating_matrix is None:
                return False
            
            n_components = min(50, min(rating_matrix.shape) - 1)
            
            if n_components < 2:
                return False
            
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            user_factors = svd.fit_transform(rating_matrix)
            item_factors = svd.components_.T
            
            self.user_embeddings = {}
            self.item_embeddings = {}
            
            for user_id, user_idx in user_mapping.items():
                self.user_embeddings[user_id] = user_factors[user_idx]
            
            for item_id, item_idx in item_mapping.items():
                self.item_embeddings[item_id] = item_factors[item_idx]
            
            self.user_mapping = user_mapping
            self.item_mapping = item_mapping
            self.model_updated = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error training collaborative model: {e}")
            return False
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 20) -> List[Dict]:
        try:
            if not self.model_updated or user_id not in self.user_embeddings:
                if not self.train_collaborative_model():
                    return []
                
                if user_id not in self.user_embeddings:
                    return []
            
            user_embedding = self.user_embeddings[user_id]
            
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_items = set([i.content_id for i in user_interactions])
            
            item_scores = []
            for item_id, item_embedding in self.item_embeddings.items():
                if item_id not in interacted_items:
                    score = np.dot(user_embedding, item_embedding) / (
                        np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding)
                    )
                    item_scores.append((item_id, float(score)))
            
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for item_id, score in item_scores[:n_recommendations]:
                content = Content.query.get(item_id)
                if content:
                    recommendations.append({
                        'content_id': item_id,
                        'score': score,
                        'method': 'collaborative_filtering',
                        'content': content
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {e}")
            return []

class ContentBasedFilteringEngine:
    def __init__(self):
        self.content_embeddings = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 3), stop_words='english')
        self.feature_scaler = StandardScaler()
        self.embeddings_built = False
        
    def build_content_embeddings(self):
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
                    description_parts.append(content.overview.lower())
                
                try:
                    genres = json.loads(content.genres or '[]')
                    description_parts.extend([g.lower() for g in genres])
                except (json.JSONDecodeError, TypeError):
                    genres = []
                
                try:
                    languages = json.loads(content.languages or '[]')
                    description_parts.extend([l.lower() for l in languages])
                except (json.JSONDecodeError, TypeError):
                    languages = []
                
                description_parts.append(content.content_type)
                
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
            
            if not content_descriptions:
                return False
            
            tfidf_features = self.tfidf_vectorizer.fit_transform(content_descriptions)
            
            metadata_df = pd.DataFrame(content_metadata)
            metadata_features = self.feature_scaler.fit_transform(metadata_df)
            
            combined_features = sparse.hstack([tfidf_features, sparse.csr_matrix(metadata_features)])
            
            for i, content_id in enumerate(content_ids):
                self.content_embeddings[content_id] = combined_features[i].toarray().flatten()
            
            self.embeddings_built = True
            return True
            
        except Exception as e:
            logger.error(f"Error building content embeddings: {e}")
            return False
    
    def get_content_based_recommendations(self, user_id: int, n_recommendations: int = 20) -> List[Dict]:
        try:
            if not self.embeddings_built:
                if not self.build_content_embeddings():
                    return []
            
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not user_interactions:
                return []
            
            interacted_content_ids = [i.content_id for i in user_interactions]
            interacted_embeddings = []
            interaction_weights = []
            
            tracker = UserInteractionTracker()
            
            for interaction in user_interactions:
                if interaction.content_id in self.content_embeddings:
                    embedding = self.content_embeddings[interaction.content_id]
                    interacted_embeddings.append(embedding)
                    
                    weight_config = tracker.interaction_weights.get(
                        interaction.interaction_type, 
                        {'weight': 0.1}
                    )
                    weight = weight_config['weight']
                    
                    if interaction.rating:
                        weight *= (interaction.rating / 5.0)
                    
                    interaction_weights.append(weight)
            
            if not interacted_embeddings:
                return []
            
            interacted_embeddings = np.array(interacted_embeddings)
            interaction_weights = np.array(interaction_weights)
            interaction_weights = interaction_weights / np.sum(interaction_weights)
            
            user_profile_vector = np.average(interacted_embeddings, axis=0, weights=interaction_weights)
            
            similarities = []
            for content_id, content_embedding in self.content_embeddings.items():
                if content_id not in interacted_content_ids:
                    similarity = cosine_similarity([user_profile_vector], [content_embedding])[0][0]
                    similarities.append((content_id, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for content_id, similarity in similarities[:n_recommendations]:
                content = Content.query.get(content_id)
                if content:
                    recommendations.append({
                        'content_id': content_id,
                        'score': float(similarity),
                        'method': 'content_based_filtering',
                        'content': content
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting content-based recommendations: {e}")
            return []

class SequenceAwareEngine:
    def __init__(self):
        self.transition_matrices = defaultdict(lambda: defaultdict(float))
        self.sequence_models = {}
        
    def build_sequence_models(self, user_id: int):
        try:
            interactions = UserInteraction.query.filter_by(user_id=user_id).order_by(
                UserInteraction.timestamp
            ).all()
            
            if len(interactions) < 3:
                return
            
            content_ids = [i.content_id for i in interactions]
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_map = {c.id: c for c in contents}
            
            sequences = []
            current_sequence = []
            
            for i, interaction in enumerate(interactions):
                content = content_map.get(interaction.content_id)
                if content:
                    try:
                        genres = json.loads(content.genres or '[]')
                        primary_genre = genres[0].lower() if genres else 'unknown'
                        
                        state = {
                            'content_id': content.id,
                            'content_type': content.content_type,
                            'primary_genre': primary_genre,
                            'rating': content.rating or 0,
                            'interaction_type': interaction.interaction_type,
                            'timestamp': interaction.timestamp
                        }
                        
                        current_sequence.append(state)
                        
                        if i > 0:
                            time_gap = (interaction.timestamp - interactions[i-1].timestamp).total_seconds()
                            if time_gap > 86400 or len(current_sequence) > 15:
                                if len(current_sequence) >= 3:
                                    sequences.append(current_sequence.copy())
                                current_sequence = [state]
                    except (json.JSONDecodeError, TypeError, IndexError):
                        continue
            
            if len(current_sequence) >= 3:
                sequences.append(current_sequence)
            
            self._build_transition_matrices(sequences)
            
        except Exception as e:
            logger.error(f"Error building sequence models: {e}")
    
    def _build_transition_matrices(self, sequences: List[List[Dict]]):
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current = sequence[i]
                next_item = sequence[i + 1]
                
                current_genre = current['primary_genre']
                next_genre = next_item['primary_genre']
                
                if current_genre != 'unknown' and next_genre != 'unknown':
                    self.transition_matrices['genre'][f"{current_genre}->{next_genre}"] += 1
                
                current_type = current['content_type']
                next_type = next_item['content_type']
                
                self.transition_matrices['content_type'][f"{current_type}->{next_type}"] += 1
    
    def get_sequence_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict]:
        try:
            self.build_sequence_models(user_id)
            
            recent_interactions = UserInteraction.query.filter_by(user_id=user_id).order_by(
                UserInteraction.timestamp.desc()
            ).limit(3).all()
            
            if not recent_interactions:
                return []
            
            predictions = defaultdict(float)
            
            for interaction in recent_interactions:
                content = Content.query.get(interaction.content_id)
                if content:
                    try:
                        genres = json.loads(content.genres or '[]')
                        primary_genre = genres[0].lower() if genres else 'unknown'
                        
                        if primary_genre != 'unknown':
                            for transition, count in self.transition_matrices['genre'].items():
                                if transition.startswith(f"{primary_genre}->"):
                                    next_genre = transition.split('->')[1]
                                    predictions[f"genre_{next_genre}"] += count * 0.4
                        
                        content_type = content.content_type
                        for transition, count in self.transition_matrices['content_type'].items():
                            if transition.startswith(f"{content_type}->"):
                                next_type = transition.split('->')[1]
                                predictions[f"type_{next_type}"] += count * 0.3
                                
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            recommendations = []
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_content = set([i.content_id for i in user_interactions])
            
            for prediction_key, score in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                matching_content = self._find_matching_content(prediction_key, interacted_content)
                
                for content in matching_content[:2]:
                    if content.id not in [r['content_id'] for r in recommendations]:
                        recommendations.append({
                            'content_id': content.id,
                            'score': score,
                            'method': 'sequence_aware',
                            'content': content,
                            'prediction_basis': prediction_key
                        })
                        
                        if len(recommendations) >= n_recommendations:
                            break
                
                if len(recommendations) >= n_recommendations:
                    break
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting sequence recommendations: {e}")
            return []
    
    def _find_matching_content(self, prediction_key: str, interacted_content: Set[int]) -> List[Content]:
        try:
            prediction_type, value = prediction_key.split('_', 1)
            
            query = Content.query.filter(~Content.id.in_(interacted_content))
            
            if prediction_type == 'genre':
                query = query.filter(Content.genres.contains(value.title()))
            elif prediction_type == 'type':
                query = query.filter(Content.content_type == value)
            
            return query.order_by(Content.popularity.desc()).limit(5).all()
            
        except Exception:
            return []

class ContextAwareRankingEngine:
    def __init__(self):
        self.context_weights = {
            'time_of_day': {
                'morning': {'documentary': 1.2, 'news': 1.3, 'educational': 1.2},
                'afternoon': {'comedy': 1.2, 'action': 1.1, 'adventure': 1.1},
                'evening': {'drama': 1.2, 'thriller': 1.2, 'mystery': 1.1},
                'night': {'horror': 1.3, 'sci-fi': 1.2, 'fantasy': 1.1}
            },
            'day_of_week': {
                'weekday': {'short': 1.2, 'educational': 1.1},
                'weekend': {'long': 1.2, 'binge': 1.3, 'family': 1.2}
            },
            'device': {
                'mobile': {'short': 1.3, 'vertical': 1.2},
                'tablet': {'medium': 1.2, 'portable': 1.1},
                'desktop': {'long': 1.2, 'detailed': 1.1},
                'tv': {'cinematic': 1.3, 'high_quality': 1.2}
            }
        }
        
    def apply_contextual_ranking(self, recommendations: List[Dict], context: Dict, 
                               user_profile: Dict) -> List[Dict]:
        try:
            current_hour = datetime.utcnow().hour
            current_day = datetime.utcnow().weekday()
            
            time_of_day = self._get_time_of_day(current_hour)
            day_type = 'weekend' if current_day >= 5 else 'weekday'
            device_type = self._detect_device_type(context.get('user_agent', ''))
            
            for rec in recommendations:
                content = rec['content']
                contextual_boost = 0.0
                
                contextual_boost += self._apply_time_context(content, time_of_day)
                contextual_boost += self._apply_day_context(content, day_type)
                contextual_boost += self._apply_device_context(content, device_type)
                contextual_boost += self._apply_user_pattern_context(content, user_profile, current_hour, current_day)
                
                rec['score'] += contextual_boost
                rec['contextual_boost'] = contextual_boost
            
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations
            
        except Exception as e:
            logger.error(f"Error applying contextual ranking: {e}")
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
    
    def _detect_device_type(self, user_agent: str) -> str:
        user_agent = user_agent.lower()
        if 'mobile' in user_agent or 'android' in user_agent:
            return 'mobile'
        elif 'tablet' in user_agent or 'ipad' in user_agent:
            return 'tablet'
        elif 'tv' in user_agent or 'smart' in user_agent:
            return 'tv'
        else:
            return 'desktop'
    
    def _apply_time_context(self, content: Content, time_of_day: str) -> float:
        boost = 0.0
        
        try:
            genres = json.loads(content.genres or '[]')
            time_weights = self.context_weights['time_of_day'].get(time_of_day, {})
            
            for genre in genres:
                genre_weight = time_weights.get(genre.lower(), 1.0)
                if genre_weight > 1.0:
                    boost += (genre_weight - 1.0) * 0.1
                    
        except (json.JSONDecodeError, TypeError):
            pass
        
        return boost
    
    def _apply_day_context(self, content: Content, day_type: str) -> float:
        boost = 0.0
        
        day_weights = self.context_weights['day_of_week'].get(day_type, {})
        
        if content.runtime:
            if day_type == 'weekend' and content.runtime > 150:
                boost += 0.05
            elif day_type == 'weekday' and content.runtime <= 120:
                boost += 0.05
        
        return boost
    
    def _apply_device_context(self, content: Content, device_type: str) -> float:
        boost = 0.0
        
        device_weights = self.context_weights['device'].get(device_type, {})
        
        if device_type == 'mobile' and content.runtime and content.runtime <= 90:
            boost += 0.05
        elif device_type == 'tv' and content.rating and content.rating >= 8.0:
            boost += 0.05
        
        return boost
    
    def _apply_user_pattern_context(self, content: Content, user_profile: Dict, 
                                  current_hour: int, current_day: int) -> float:
        boost = 0.0
        
        temporal_patterns = user_profile.get('temporal_patterns', {})
        viewing_hours = temporal_patterns.get('viewing_hours', {})
        viewing_days = temporal_patterns.get('viewing_days', {})
        
        if str(current_hour) in viewing_hours:
            boost += 0.03
        
        if str(current_day) in viewing_days:
            boost += 0.03
        
        behavioral_insights = user_profile.get('behavioral_insights', {})
        
        if content.is_trending and behavioral_insights.get('trending_sensitivity', 0) > 0.5:
            boost += 0.04
        
        return boost

class HybridRecommendationEngine:
    def __init__(self):
        self.profiler = AdvancedUserProfiler()
        self.collaborative_engine = CollaborativeFilteringEngine()
        self.content_engine = ContentBasedFilteringEngine()
        self.sequence_engine = SequenceAwareEngine()
        self.context_engine = ContextAwareRankingEngine()
        
        self.algorithm_weights = {
            'collaborative_filtering': 0.3,
            'content_based_filtering': 0.25,
            'sequence_aware': 0.2,
            'trending_boost': 0.1,
            'quality_boost': 0.1,
            'diversity_boost': 0.05
        }
        
    def get_hybrid_recommendations(self, user_id: int, content_type: Optional[str] = None,
                                 context: Optional[Dict] = None, 
                                 n_recommendations: int = 20) -> Dict:
        try:
            user_profile = self.profiler.build_comprehensive_profile(user_id)
            confidence_level = user_profile.get('confidence_score', 0.0)
            
            all_recommendations = []
            algorithm_performance = {}
            
            try:
                collab_recs = self.collaborative_engine.get_user_recommendations(user_id, n_recommendations * 2)
                all_recommendations.extend(collab_recs)
                algorithm_performance['collaborative_filtering'] = len(collab_recs)
            except Exception as e:
                logger.warning(f"Collaborative filtering failed: {e}")
                algorithm_performance['collaborative_filtering'] = 0
            
            try:
                content_recs = self.content_engine.get_content_based_recommendations(user_id, n_recommendations * 2)
                all_recommendations.extend(content_recs)
                algorithm_performance['content_based_filtering'] = len(content_recs)
            except Exception as e:
                logger.warning(f"Content-based filtering failed: {e}")
                algorithm_performance['content_based_filtering'] = 0
            
            try:
                sequence_recs = self.sequence_engine.get_sequence_recommendations(user_id, n_recommendations)
                all_recommendations.extend(sequence_recs)
                algorithm_performance['sequence_aware'] = len(sequence_recs)
            except Exception as e:
                logger.warning(f"Sequence-aware failed: {e}")
                algorithm_performance['sequence_aware'] = 0
            
            try:
                trending_recs = self._get_trending_recommendations(user_id, user_profile, n_recommendations // 4)
                all_recommendations.extend(trending_recs)
                algorithm_performance['trending_boost'] = len(trending_recs)
            except Exception as e:
                logger.warning(f"Trending recommendations failed: {e}")
                algorithm_performance['trending_boost'] = 0
            
            try:
                quality_recs = self._get_quality_recommendations(user_id, user_profile, n_recommendations // 4)
                all_recommendations.extend(quality_recs)
                algorithm_performance['quality_boost'] = len(quality_recs)
            except Exception as e:
                logger.warning(f"Quality recommendations failed: {e}")
                algorithm_performance['quality_boost'] = 0
            
            fused_recommendations = self._advanced_fusion(all_recommendations, user_profile, confidence_level)
            
            if context:
                fused_recommendations = self.context_engine.apply_contextual_ranking(
                    fused_recommendations, context, user_profile
                )
            
            fused_recommendations = self._apply_diversity(fused_recommendations, user_profile)
            
            if content_type:
                fused_recommendations = [
                    rec for rec in fused_recommendations 
                    if rec['content'].content_type == content_type
                ]
            
            formatted_recommendations = self._format_recommendations(
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
                    'emerging_interests': user_profile.get('prediction_features', {}).get('emerging_interests', [])
                },
                'recommendation_metadata': {
                    'total_recommendations': len(formatted_recommendations),
                    'algorithm_performance': algorithm_performance,
                    'personalization_strength': min(confidence_level * 100, 100),
                    'context_applied': context is not None,
                    'diversity_applied': True,
                    'accuracy_estimate': self._estimate_accuracy(user_profile, algorithm_performance),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return {
                'recommendations': [],
                'user_profile_insights': {},
                'recommendation_metadata': {'error': str(e)}
            }
    
    def _get_trending_recommendations(self, user_id: int, user_profile: Dict, 
                                    n_recommendations: int) -> List[Dict]:
        user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        interacted_content = set([i.content_id for i in user_interactions])
        
        trending_content = Content.query.filter(
            Content.is_trending == True,
            ~Content.id.in_(interacted_content)
        ).order_by(Content.popularity.desc()).limit(n_recommendations * 2).all()
        
        recommendations = []
        for content in trending_content:
            score = self._calculate_profile_match_score(content, user_profile) + 0.2
            recommendations.append({
                'content_id': content.id,
                'score': score,
                'method': 'trending_boost',
                'content': content
            })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def _get_quality_recommendations(self, user_id: int, user_profile: Dict, 
                                   n_recommendations: int) -> List[Dict]:
        user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
        interacted_content = set([i.content_id for i in user_interactions])
        
        quality_prefs = user_profile.get('quality_preferences', {})
        min_rating = quality_prefs.get('preferred_content_rating', 7.0) - 0.5
        
        quality_content = Content.query.filter(
            Content.rating >= min_rating,
            Content.vote_count >= 100,
            ~Content.id.in_(interacted_content)
        ).order_by(Content.rating.desc()).limit(n_recommendations * 2).all()
        
        recommendations = []
        for content in quality_content:
            score = self._calculate_profile_match_score(content, user_profile) + 0.15
            recommendations.append({
                'content_id': content.id,
                'score': score,
                'method': 'quality_boost',
                'content': content
            })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def _calculate_profile_match_score(self, content: Content, user_profile: Dict) -> float:
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
                score += lang_pref * 0.3
        except (json.JSONDecodeError, TypeError):
            pass
        
        content_type_prefs = user_profile.get('content_preferences', {}).get('content_types', {})
        content_type_pref = content_type_prefs.get(content.content_type, 0)
        score += content_type_pref * 0.2
        
        quality_prefs = user_profile.get('quality_preferences', {})
        if content.rating and quality_prefs.get('preferred_content_rating'):
            quality_diff = abs(content.rating - quality_prefs['preferred_content_rating'])
            quality_score = max(0, 1 - (quality_diff / 5.0))
            score += quality_score * 0.1
        
        return min(score, 1.0)
    
    def _advanced_fusion(self, all_recommendations: List[Dict], user_profile: Dict, 
                        confidence_level: float) -> List[Dict]:
        content_scores = defaultdict(list)
        content_objects = {}
        
        for rec in all_recommendations:
            content_id = rec['content_id']
            method = rec.get('method', 'unknown')
            score = rec['score']
            
            content_scores[content_id].append((score, method))
            content_objects[content_id] = rec['content']
        
        fused_recommendations = []
        
        for content_id, scores in content_scores.items():
            content = content_objects[content_id]
            
            total_score = 0.0
            total_weight = 0.0
            methods_used = []
            
            method_scores = defaultdict(list)
            for score, method in scores:
                method_scores[method].append(score)
                methods_used.append(method)
            
            for method, method_score_list in method_scores.items():
                best_score = max(method_score_list)
                
                base_weight = self.algorithm_weights.get(method, 0.1)
                
                if confidence_level >= 0.8:
                    if method in ['collaborative_filtering', 'sequence_aware']:
                        base_weight *= 1.2
                elif confidence_level >= 0.5:
                    base_weight *= 1.0
                else:
                    if method in ['content_based_filtering', 'trending_boost']:
                        base_weight *= 1.1
                
                total_score += best_score * base_weight
                total_weight += base_weight
            
            if total_weight > 0:
                fusion_score = total_score / total_weight
                
                fusion_score = self._apply_post_fusion_boosters(
                    fusion_score, content, user_profile, methods_used
                )
                
                fused_recommendations.append({
                    'content_id': content_id,
                    'score': fusion_score,
                    'content': content,
                    'methods_used': list(set(methods_used)),
                    'method_count': len(set(methods_used))
                })
        
        fused_recommendations.sort(key=lambda x: x['score'], reverse=True)
        return fused_recommendations
    
    def _apply_post_fusion_boosters(self, score: float, content: Content, 
                                  user_profile: Dict, methods_used: List[str]) -> float:
        boosted_score = score
        
        if len(set(methods_used)) >= 3:
            boosted_score += 0.1
        elif len(set(methods_used)) >= 2:
            boosted_score += 0.05
        
        if content.release_date:
            days_since_release = (datetime.utcnow().date() - content.release_date).days
            if days_since_release <= 30:
                boosted_score += 0.05
            elif days_since_release <= 90:
                boosted_score += 0.02
        
        if content.is_trending:
            boosted_score += 0.03
        
        if content.rating and content.rating >= 8.5:
            boosted_score += 0.04
        elif content.rating and content.rating >= 8.0:
            boosted_score += 0.02
        
        try:
            content_languages = json.loads(content.languages or '[]')
            user_languages = user_profile.get('content_preferences', {}).get('languages', {})
            
            for lang in content_languages:
                lang_pref = user_languages.get(lang.lower(), 0)
                if lang_pref > 0.5:
                    boosted_score += 0.03
                    break
        except (json.JSONDecodeError, TypeError):
            pass
        
        return min(boosted_score, 2.0)
    
    def _apply_diversity(self, recommendations: List[Dict], user_profile: Dict) -> List[Dict]:
        exploration_tendency = user_profile.get('behavioral_insights', {}).get('exploration_tendency', 0.5)
        
        if exploration_tendency < 0.3:
            return recommendations
        
        seen_genres = set()
        seen_content_types = set()
        seen_languages = set()
        
        diverse_recommendations = []
        other_recommendations = []
        
        for rec in recommendations:
            content = rec['content']
            
            try:
                content_genres = json.loads(content.genres or '[]')
                content_languages = json.loads(content.languages or '[]')
                
                primary_genre = content_genres[0].lower() if content_genres else 'unknown'
                primary_language = content_languages[0].lower() if content_languages else 'unknown'
                content_type = content.content_type
                
                is_diverse = (
                    primary_genre not in seen_genres or
                    content_type not in seen_content_types or
                    primary_language not in seen_languages
                )
                
                if is_diverse and len(diverse_recommendations) < len(recommendations) * 0.7:
                    seen_genres.add(primary_genre)
                    seen_content_types.add(content_type)
                    seen_languages.add(primary_language)
                    
                    rec['score'] += 0.05
                    diverse_recommendations.append(rec)
                else:
                    other_recommendations.append(rec)
                    
            except (json.JSONDecodeError, TypeError, IndexError):
                other_recommendations.append(rec)
        
        final_recommendations = []
        
        diversity_count = int(len(recommendations) * exploration_tendency)
        final_recommendations.extend(diverse_recommendations[:diversity_count])
        
        remaining_slots = len(recommendations) - len(final_recommendations)
        
        remaining_diverse = diverse_recommendations[diversity_count:]
        final_recommendations.extend(remaining_diverse[:remaining_slots//2])
        
        remaining_slots = len(recommendations) - len(final_recommendations)
        final_recommendations.extend(other_recommendations[:remaining_slots])
        
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        return final_recommendations
    
    def _format_recommendations(self, recommendations: List[Dict], user_profile: Dict) -> List[Dict]:
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
            
            reason = self._generate_recommendation_reason(rec, user_profile)
            
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
                'overview': content.overview[:200] + '...' if content.overview and len(content.overview) > 200 else content.overview,
                'youtube_trailer': youtube_url,
                'recommendation_score': round(rec['score'], 4),
                'confidence_level': self._calculate_recommendation_confidence(rec, user_profile),
                'recommendation_reason': reason,
                'methods_used': rec.get('methods_used', ['hybrid']),
                'is_trending': content.is_trending,
                'is_new_release': content.is_new_release,
                'personalization_tags': self._generate_personalization_tags(content, user_profile),
                'contextual_boost': rec.get('contextual_boost', 0.0)
            }
            
            formatted_recs.append(formatted_rec)
        
        return formatted_recs
    
    def _generate_recommendation_reason(self, recommendation: Dict, user_profile: Dict) -> str:
        content = recommendation['content']
        methods = recommendation.get('methods_used', [])
        score = recommendation.get('score', 0)
        
        reasons = []
        
        if 'collaborative_filtering' in methods:
            reasons.append("users with similar tastes highly recommend this")
        
        if 'sequence_aware' in methods:
            reasons.append("perfectly follows your viewing pattern")
        
        if 'content_based_filtering' in methods:
            reasons.append("matches your content preferences precisely")
        
        try:
            content_genres = json.loads(content.genres or '[]')
            user_genres = user_profile.get('content_preferences', {}).get('genres', {})
            
            matching_genres = []
            for genre in content_genres[:2]:
                if user_genres.get(genre.lower(), 0) > 0.3:
                    matching_genres.append(genre.lower())
            
            if matching_genres:
                if len(matching_genres) == 1:
                    reasons.append(f"you love {matching_genres[0]} content")
                else:
                    reasons.append(f"combines your favorite genres: {' and '.join(matching_genres)}")
        except (json.JSONDecodeError, TypeError):
            pass
        
        if content.rating and content.rating >= 8.5:
            reasons.append("exceptional quality (highly rated)")
        elif content.rating and content.rating >= 8.0:
            reasons.append("excellent quality")
        
        if content.is_trending:
            reasons.append("trending now")
        
        if content.is_new_release:
            reasons.append("fresh release")
        
        try:
            content_languages = json.loads(content.languages or '[]')
            user_languages = user_profile.get('content_preferences', {}).get('languages', {})
            
            for lang in content_languages:
                if user_languages.get(lang.lower(), 0) > 0.4:
                    reasons.append(f"in your preferred {lang} language")
                    break
        except (json.JSONDecodeError, TypeError):
            pass
        
        if score > 0.8:
            confidence_phrase = "Perfect match"
        elif score > 0.6:
            confidence_phrase = "Great match"
        elif score > 0.4:
            confidence_phrase = "Good match"
        else:
            confidence_phrase = "Interesting option"
        
        if reasons:
            return f"{confidence_phrase}: {', '.join(reasons[:3])}"
        else:
            return f"{confidence_phrase} based on your unique preferences"
    
    def _calculate_recommendation_confidence(self, recommendation: Dict, user_profile: Dict) -> str:
        score = recommendation.get('score', 0)
        methods_used = recommendation.get('methods_used', [])
        user_confidence = user_profile.get('confidence_score', 0)
        
        if score > 0.8:
            base_confidence = 'very_high'
        elif score > 0.6:
            base_confidence = 'high'
        elif score > 0.4:
            base_confidence = 'medium'
        elif score > 0.2:
            base_confidence = 'low'
        else:
            base_confidence = 'very_low'
        
        if len(methods_used) >= 3:
            if base_confidence == 'medium':
                base_confidence = 'high'
            elif base_confidence == 'low':
                base_confidence = 'medium'
        
        if user_confidence < 0.3 and base_confidence in ['very_high', 'high']:
            base_confidence = 'medium'
        
        return base_confidence
    
    def _generate_personalization_tags(self, content: Content, user_profile: Dict) -> List[str]:
        tags = []
        
        try:
            content_genres = json.loads(content.genres or '[]')
            user_genres = user_profile.get('content_preferences', {}).get('genres', {})
            
            for genre in content_genres:
                if user_genres.get(genre.lower(), 0) > 0.4:
                    tags.append(f"favorite_{genre.lower()}")
        except (json.JSONDecodeError, TypeError):
            pass
        
        try:
            content_languages = json.loads(content.languages or '[]')
            user_languages = user_profile.get('content_preferences', {}).get('languages', {})
            
            for lang in content_languages:
                if user_languages.get(lang.lower(), 0) > 0.3:
                    tags.append(f"preferred_{lang.lower()}")
        except (json.JSONDecodeError, TypeError):
            pass
        
        quality_prefs = user_profile.get('quality_preferences', {})
        user_avg_rating = quality_prefs.get('preferred_content_rating', 7.0)
        
        if content.rating:
            if content.rating >= user_avg_rating + 0.5:
                tags.append('higher_quality_than_usual')
            elif content.rating >= user_avg_rating - 0.5:
                tags.append('your_quality_range')
        
        trending_interests = user_profile.get('prediction_features', {}).get('emerging_interests', [])
        try:
            content_genres = json.loads(content.genres or '[]')
            for genre in content_genres:
                if genre.lower() in trending_interests:
                    tags.append('emerging_interest')
                    break
        except (json.JSONDecodeError, TypeError):
            pass
        
        exploration_tendency = user_profile.get('behavioral_insights', {}).get('exploration_tendency', 0.5)
        if exploration_tendency > 0.7:
            tags.append('adventurous_choice')
        elif exploration_tendency < 0.3:
            tags.append('safe_choice')
        
        return tags[:5]
    
    def _estimate_accuracy(self, user_profile: Dict, algorithm_performance: Dict) -> float:
        confidence = user_profile.get('confidence_score', 0)
        interaction_count = user_profile.get('total_interactions', 0)
        
        base_accuracy = min(confidence * 85, 85)
        
        interaction_bonus = min(interaction_count / 150.0 * 10, 10)
        
        active_algorithms = sum(1 for count in algorithm_performance.values() if count > 0)
        algorithm_bonus = min(active_algorithms * 2, 8)
        
        total_recommendations = sum(algorithm_performance.values())
        performance_bonus = min(total_recommendations / 80.0 * 2, 2)
        
        estimated_accuracy = base_accuracy + interaction_bonus + algorithm_bonus + performance_bonus
        
        return min(round(estimated_accuracy, 1), 97.0)

hybrid_engine = HybridRecommendationEngine()

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
        
        recommendations = hybrid_engine.get_hybrid_recommendations(
            current_user.id, content_type, context, limit
        )
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}")
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
            'for_you': hybrid_engine.get_hybrid_recommendations(
                current_user.id, None, context, 20
            ),
            'movies': hybrid_engine.get_hybrid_recommendations(
                current_user.id, 'movie', context, 15
            ),
            'tv_shows': hybrid_engine.get_hybrid_recommendations(
                current_user.id, 'tv', context, 15
            ),
            'anime': hybrid_engine.get_hybrid_recommendations(
                current_user.id, 'anime', context, 15
            )
        }
        
        return jsonify({
            'categories': categories,
            'metadata': {
                'personalized': True,
                'user_id': current_user.id,
                'context_applied': True,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting personalized categories: {e}")
        return jsonify({'error': 'Failed to get personalized categories'}), 500

@personalized_bp.route('/api/recommendations/track-interaction', methods=['POST'])
@require_auth
def track_user_interaction(current_user):
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'interaction_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        tracker = UserInteractionTracker()
        
        metadata = {
            'from_recommendation': data.get('from_recommendation', False),
            'recommendation_score': data.get('recommendation_score'),
            'recommendation_method': data.get('recommendation_method'),
            'context': data.get('context', {}),
            'user_agent': request.headers.get('User-Agent', ''),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        success = tracker.track_interaction(
            current_user.id,
            data['content_id'],
            data['interaction_type'],
            data.get('rating'),
            metadata
        )
        
        if success:
            return jsonify({'message': 'Interaction tracked successfully'}), 201
        else:
            return jsonify({'error': 'Failed to track interaction'}), 500
            
    except Exception as e:
        logger.error(f"Error tracking interaction: {e}")
        return jsonify({'error': 'Failed to track interaction'}), 500

@personalized_bp.route('/api/recommendations/retrain', methods=['POST'])
@require_auth
def retrain_models(current_user):
    try:
        if not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        results = {}
        
        try:
            success = hybrid_engine.collaborative_engine.train_collaborative_model()
            results['collaborative_filtering'] = 'success' if success else 'failed'
        except Exception as e:
            results['collaborative_filtering'] = f'error: {str(e)}'
        
        try:
            success = hybrid_engine.content_engine.build_content_embeddings()
            results['content_based_filtering'] = 'success' if success else 'failed'
        except Exception as e:
            results['content_based_filtering'] = f'error: {str(e)}'
        
        try:
            if cache:
                users = User.query.all()
                cleared_count = 0
                for user in users:
                    cache.delete(f"user_profile_v2:{user.id}")
                    cache.delete(f"recommendations:{user.id}")
                    cache.delete(f"user_embeddings:{user.id}")
                    cleared_count += 1
                results['cache_clearing'] = f'cleared {cleared_count} user profiles'
        except Exception as e:
            results['cache_clearing'] = f'error: {str(e)}'
        
        return jsonify({
            'message': 'Model retraining completed',
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        return jsonify({'error': 'Failed to retrain models'}), 500

@personalized_bp.route('/api/recommendations/similar/<int:content_id>', methods=['GET'])
@require_auth
def get_similar_recommendations(current_user, content_id):
    try:
        limit = min(int(request.args.get('limit', 10)), 20)
        
        if not hybrid_engine.content_engine.embeddings_built:
            hybrid_engine.content_engine.build_content_embeddings()
        
        user_profile = hybrid_engine.profiler.build_comprehensive_profile(current_user.id)
        
        content_embeddings = hybrid_engine.content_engine.content_embeddings
        
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
        
        for similar_id, similarity in similarities[:limit*2]:
            if similar_id not in interacted_content:
                content = Content.query.get(similar_id)
                if content:
                    profile_match = hybrid_engine._calculate_profile_match_score(content, user_profile)
                    final_score = (similarity * 0.7) + (profile_match * 0.3)
                    
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
                        'overview': content.overview[:150] + '...' if content.overview else '',
                        'youtube_trailer': youtube_url,
                        'similarity_score': round(similarity, 3),
                        'personalized_score': round(final_score, 3),
                        'recommendation_reason': f"Similar content with {round(similarity*100, 1)}% match"
                    })
                    
                    if len(recommendations) >= limit:
                        break
        
        recommendations.sort(key=lambda x: x['personalized_score'], reverse=True)
        
        return jsonify({
            'similar_recommendations': recommendations,
            'base_content_id': content_id,
            'total_found': len(recommendations),
            'personalization_applied': True
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting similar recommendations: {e}")
        return jsonify({'error': 'Failed to get similar recommendations'}), 500

@personalized_bp.route('/api/user/profile/detailed', methods=['GET'])
@require_auth
def get_detailed_user_profile(current_user):
    try:
        profile = hybrid_engine.profiler.build_comprehensive_profile(current_user.id)
        
        return jsonify({
            'detailed_profile': profile,
            'profile_completeness': profile.get('profile_strength', 0),
            'recommendation_readiness': profile.get('confidence_score', 0) >= 0.3
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting detailed user profile: {e}")
        return jsonify({'error': 'Failed to get detailed profile'}), 500