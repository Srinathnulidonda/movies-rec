# backend/services/personalized.py
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import json
import logging
import jwt
import math
import random
from functools import wraps
import hashlib
from sqlalchemy import func, and_, or_, desc, text
from sqlalchemy.orm import joinedload
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
from typing import Dict, List, Tuple, Optional, Any

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
app = None
services = None

def init_personalized(flask_app, database, models, app_services, app_cache):
    """Initialize personalized module with app context and models"""
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

class UserProfileAnalyzer:
    """Advanced user profiling and preference extraction"""
    
    def __init__(self):
        self.genre_weights = {
            'primary': 1.0,
            'secondary': 0.7,
            'tertiary': 0.4
        }
        
        self.interaction_weights = {
            'rating': 1.0,
            'favorite': 0.9,
            'watchlist': 0.7,
            'like': 0.6,
            'view': 0.3,
            'search': 0.2
        }
        
        self.temporal_decay = 0.95  # Decay factor for older interactions
        
    def build_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Build comprehensive user profile from all interactions"""
        try:
            cache_key = f"user_profile:{user_id}"
            cached_profile = cache.get(cache_key) if cache else None
            
            if cached_profile:
                return cached_profile
            
            # Get all user interactions
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not interactions:
                return self._get_default_profile()
            
            # Get content for interactions
            content_ids = [i.content_id for i in interactions]
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_map = {c.id: c for c in contents}
            
            profile = {
                'user_id': user_id,
                'genres': defaultdict(float),
                'languages': defaultdict(float),
                'content_types': defaultdict(float),
                'cast_crew': defaultdict(float),
                'temporal_patterns': defaultdict(list),
                'quality_preference': 0.0,
                'popularity_bias': 0.0,
                'diversity_score': 0.0,
                'interaction_count': len(interactions),
                'preferred_runtime': {'min': 0, 'max': 300, 'avg': 120},
                'release_date_preference': defaultdict(float),
                'rating_patterns': defaultdict(list),
                'sequence_patterns': [],
                'contextual_preferences': defaultdict(dict)
            }
            
            total_weight = 0
            quality_sum = 0
            popularity_sum = 0
            runtimes = []
            release_years = []
            interaction_sequence = []
            
            # Process each interaction
            for interaction in sorted(interactions, key=lambda x: x.timestamp):
                content = content_map.get(interaction.content_id)
                if not content:
                    continue
                
                # Calculate temporal weight (more recent = higher weight)
                days_ago = (datetime.utcnow() - interaction.timestamp).days
                temporal_weight = self.temporal_decay ** (days_ago / 30)
                
                # Get interaction weight
                interaction_weight = self.interaction_weights.get(interaction.interaction_type, 0.3)
                
                # Apply rating boost if available
                rating_boost = 1.0
                if interaction.rating:
                    rating_boost = min(interaction.rating / 5.0, 1.0)
                
                final_weight = temporal_weight * interaction_weight * rating_boost
                total_weight += final_weight
                
                # Extract genres
                try:
                    genres = json.loads(content.genres or '[]')
                    for i, genre in enumerate(genres[:3]):  # Top 3 genres
                        weight_multiplier = self.genre_weights['primary'] if i == 0 else \
                                          self.genre_weights['secondary'] if i == 1 else \
                                          self.genre_weights['tertiary']
                        profile['genres'][genre.lower()] += final_weight * weight_multiplier
                except (json.JSONDecodeError, TypeError):
                    pass
                
                # Extract languages
                try:
                    languages = json.loads(content.languages or '[]')
                    for lang in languages:
                        profile['languages'][lang.lower()] += final_weight
                except (json.JSONDecodeError, TypeError):
                    pass
                
                # Content type preference
                profile['content_types'][content.content_type] += final_weight
                
                # Quality and popularity analysis
                if content.rating:
                    quality_sum += content.rating * final_weight
                if content.popularity:
                    popularity_sum += content.popularity * final_weight
                
                # Runtime preferences
                if content.runtime:
                    runtimes.append(content.runtime)
                
                # Release date preferences
                if content.release_date:
                    year = content.release_date.year
                    decade = f"{(year // 10) * 10}s"
                    profile['release_date_preference'][decade] += final_weight
                    release_years.append(year)
                
                # Rating patterns
                if interaction.rating:
                    profile['rating_patterns'][content.content_type].append(interaction.rating)
                
                # Sequence tracking
                interaction_sequence.append({
                    'content_id': content.id,
                    'content_type': content.content_type,
                    'genres': genres if 'genres' in locals() else [],
                    'timestamp': interaction.timestamp.isoformat(),
                    'interaction_type': interaction.interaction_type
                })
                
                # Temporal patterns
                hour = interaction.timestamp.hour
                day_of_week = interaction.timestamp.weekday()
                profile['temporal_patterns']['hours'].append(hour)
                profile['temporal_patterns']['days'].append(day_of_week)
            
            # Normalize weights
            if total_weight > 0:
                for genre in profile['genres']:
                    profile['genres'][genre] /= total_weight
                for lang in profile['languages']:
                    profile['languages'][lang] /= total_weight
                for ctype in profile['content_types']:
                    profile['content_types'][ctype] /= total_weight
                
                profile['quality_preference'] = quality_sum / total_weight if quality_sum > 0 else 5.0
                profile['popularity_bias'] = popularity_sum / total_weight if popularity_sum > 0 else 100.0
            
            # Runtime analysis
            if runtimes:
                profile['preferred_runtime'] = {
                    'min': min(runtimes),
                    'max': max(runtimes),
                    'avg': sum(runtimes) / len(runtimes),
                    'std': np.std(runtimes) if len(runtimes) > 1 else 0
                }
            
            # Diversity calculation
            profile['diversity_score'] = self._calculate_diversity_score(profile['genres'])
            
            # Sequence patterns (last 20 interactions)
            profile['sequence_patterns'] = interaction_sequence[-20:]
            
            # Contextual preferences
            profile['contextual_preferences'] = self._extract_contextual_preferences(interactions, content_map)
            
            # Convert defaultdict to regular dict for JSON serialization
            profile['genres'] = dict(profile['genres'])
            profile['languages'] = dict(profile['languages'])
            profile['content_types'] = dict(profile['content_types'])
            profile['cast_crew'] = dict(profile['cast_crew'])
            profile['release_date_preference'] = dict(profile['release_date_preference'])
            profile['rating_patterns'] = dict(profile['rating_patterns'])
            profile['contextual_preferences'] = dict(profile['contextual_preferences'])
            
            # Cache the profile
            if cache:
                cache.set(cache_key, profile, timeout=3600)  # Cache for 1 hour
            
            return profile
            
        except Exception as e:
            logger.error(f"Error building user profile for user {user_id}: {e}")
            return self._get_default_profile()
    
    def _get_default_profile(self) -> Dict[str, Any]:
        """Return default profile for new users"""
        return {
            'user_id': None,
            'genres': {'action': 0.3, 'drama': 0.2, 'comedy': 0.2, 'thriller': 0.15, 'romance': 0.15},
            'languages': {'english': 0.4, 'telugu': 0.3, 'hindi': 0.2, 'tamil': 0.1},
            'content_types': {'movie': 0.6, 'tv': 0.3, 'anime': 0.1},
            'cast_crew': {},
            'temporal_patterns': defaultdict(list),
            'quality_preference': 7.0,
            'popularity_bias': 100.0,
            'diversity_score': 0.5,
            'interaction_count': 0,
            'preferred_runtime': {'min': 90, 'max': 180, 'avg': 120},
            'release_date_preference': {'2020s': 0.4, '2010s': 0.3, '2000s': 0.2, '1990s': 0.1},
            'rating_patterns': {},
            'sequence_patterns': [],
            'contextual_preferences': {}
        }
    
    def _calculate_diversity_score(self, genres: Dict[str, float]) -> float:
        """Calculate user's genre diversity preference"""
        if not genres:
            return 0.5
        
        # Calculate entropy of genre distribution
        total = sum(genres.values())
        if total == 0:
            return 0.5
        
        entropy = 0
        for weight in genres.values():
            if weight > 0:
                p = weight / total
                entropy -= p * math.log2(p)
        
        # Normalize entropy (max entropy for uniform distribution)
        max_entropy = math.log2(len(genres)) if len(genres) > 1 else 1
        return entropy / max_entropy if max_entropy > 0 else 0.5
    
    def _extract_contextual_preferences(self, interactions, content_map) -> Dict[str, Any]:
        """Extract contextual preferences like time-based patterns"""
        context = {
            'weekend_preferences': defaultdict(float),
            'weekday_preferences': defaultdict(float),
            'time_of_day_preferences': defaultdict(float),
            'binge_watching_tendency': 0.0
        }
        
        weekend_interactions = []
        weekday_interactions = []
        
        for interaction in interactions:
            content = content_map.get(interaction.content_id)
            if not content:
                continue
            
            day_of_week = interaction.timestamp.weekday()
            hour = interaction.timestamp.hour
            
            # Weekend (Friday evening, Saturday, Sunday)
            is_weekend = day_of_week >= 5 or (day_of_week == 4 and hour >= 18)
            
            try:
                genres = json.loads(content.genres or '[]')
                primary_genre = genres[0].lower() if genres else 'unknown'
                
                if is_weekend:
                    weekend_interactions.append(interaction)
                    context['weekend_preferences'][primary_genre] += 1
                else:
                    weekday_interactions.append(interaction)
                    context['weekday_preferences'][primary_genre] += 1
                
                # Time of day preferences
                if 6 <= hour < 12:
                    time_period = 'morning'
                elif 12 <= hour < 18:
                    time_period = 'afternoon'
                elif 18 <= hour < 22:
                    time_period = 'evening'
                else:
                    time_period = 'night'
                
                context['time_of_day_preferences'][f"{time_period}_{primary_genre}"] += 1
                
            except (json.JSONDecodeError, TypeError, IndexError):
                pass
        
        # Calculate binge watching tendency
        context['binge_watching_tendency'] = self._calculate_binge_tendency(interactions)
        
        return context
    
    def _calculate_binge_tendency(self, interactions) -> float:
        """Calculate user's tendency to binge watch"""
        if len(interactions) < 3:
            return 0.0
        
        # Group interactions by day
        daily_interactions = defaultdict(list)
        for interaction in interactions:
            date_key = interaction.timestamp.date()
            daily_interactions[date_key].append(interaction)
        
        binge_days = 0
        total_days = len(daily_interactions)
        
        for date, day_interactions in daily_interactions.items():
            if len(day_interactions) >= 3:  # 3+ interactions in a day suggests binge behavior
                binge_days += 1
        
        return binge_days / total_days if total_days > 0 else 0.0

class CollaborativeFilteringEngine:
    """Advanced collaborative filtering with matrix factorization"""
    
    def __init__(self):
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def build_rating_matrix(self) -> Tuple[np.ndarray, Dict, Dict]:
        """Build user-item rating matrix from interactions"""
        try:
            # Get all rating interactions
            interactions = UserInteraction.query.filter(
                UserInteraction.interaction_type.in_(['rating', 'favorite', 'like'])
            ).all()
            
            if not interactions:
                return np.array([]), {}, {}
            
            # Create mappings
            users = list(set([i.user_id for i in interactions]))
            items = list(set([i.content_id for i in interactions]))
            
            self.user_mapping = {user_id: idx for idx, user_id in enumerate(users)}
            self.item_mapping = {item_id: idx for idx, item_id in enumerate(items)}
            self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
            self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
            
            # Build matrix
            matrix = np.zeros((len(users), len(items)))
            
            for interaction in interactions:
                user_idx = self.user_mapping[interaction.user_id]
                item_idx = self.item_mapping[interaction.content_id]
                
                # Convert interaction to rating
                if interaction.interaction_type == 'rating' and interaction.rating:
                    rating = interaction.rating
                elif interaction.interaction_type == 'favorite':
                    rating = 5.0
                elif interaction.interaction_type == 'like':
                    rating = 4.0
                else:
                    rating = 3.0
                
                matrix[user_idx, item_idx] = rating
            
            return matrix, self.user_mapping, self.item_mapping
            
        except Exception as e:
            logger.error(f"Error building rating matrix: {e}")
            return np.array([]), {}, {}
    
    def train_matrix_factorization(self, n_factors=50, n_epochs=100) -> bool:
        """Train matrix factorization model"""
        try:
            rating_matrix, user_map, item_map = self.build_rating_matrix()
            
            if rating_matrix.size == 0:
                return False
            
            # Use TruncatedSVD for matrix factorization
            svd = TruncatedSVD(n_components=min(n_factors, min(rating_matrix.shape) - 1))
            
            # Handle sparse matrix
            mask = rating_matrix > 0
            filled_matrix = rating_matrix.copy()
            filled_matrix[~mask] = np.mean(rating_matrix[mask]) if np.any(mask) else 3.0
            
            # Fit the model
            user_factors = svd.fit_transform(filled_matrix)
            item_factors = svd.components_.T
            
            self.user_factors = user_factors
            self.item_factors = item_factors
            self.model = svd
            
            return True
            
        except Exception as e:
            logger.error(f"Error training matrix factorization: {e}")
            return False
    
    def get_collaborative_recommendations(self, user_id: int, n_recommendations=20) -> List[Dict]:
        """Get collaborative filtering recommendations"""
        try:
            if self.user_factors is None or user_id not in self.user_mapping:
                return []
            
            user_idx = self.user_mapping[user_id]
            user_vector = self.user_factors[user_idx]
            
            # Calculate item scores
            scores = np.dot(user_vector, self.item_factors.T)
            
            # Get user's already interacted items
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_items = set([i.content_id for i in user_interactions])
            
            # Get top recommendations
            recommendations = []
            sorted_indices = np.argsort(scores)[::-1]
            
            for idx in sorted_indices:
                if len(recommendations) >= n_recommendations:
                    break
                
                item_id = self.reverse_item_mapping.get(idx)
                if item_id and item_id not in interacted_items:
                    content = Content.query.get(item_id)
                    if content:
                        recommendations.append({
                            'content_id': item_id,
                            'score': float(scores[idx]),
                            'method': 'collaborative_filtering',
                            'content': content
                        })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {e}")
            return []

class ContentBasedFilteringEngine:
    """Advanced content-based filtering with TF-IDF and embeddings"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.content_features = None
        self.content_mapping = {}
        
    def build_content_features(self):
        """Build content feature matrix"""
        try:
            contents = Content.query.all()
            
            if not contents:
                return False
            
            # Create content descriptions
            content_descriptions = []
            content_ids = []
            
            for content in contents:
                description_parts = []
                
                # Add title and overview
                if content.title:
                    description_parts.append(content.title.lower())
                if content.overview:
                    description_parts.append(content.overview.lower())
                
                # Add genres
                try:
                    genres = json.loads(content.genres or '[]')
                    description_parts.extend([g.lower() for g in genres])
                except (json.JSONDecodeError, TypeError):
                    pass
                
                # Add languages
                try:
                    languages = json.loads(content.languages or '[]')
                    description_parts.extend([l.lower() for l in languages])
                except (json.JSONDecodeError, TypeError):
                    pass
                
                # Add content type
                description_parts.append(content.content_type)
                
                # Join all parts
                full_description = ' '.join(description_parts)
                content_descriptions.append(full_description)
                content_ids.append(content.id)
            
            # Create TF-IDF matrix
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.content_features = self.tfidf_vectorizer.fit_transform(content_descriptions)
            self.content_mapping = {content_id: idx for idx, content_id in enumerate(content_ids)}
            
            return True
            
        except Exception as e:
            logger.error(f"Error building content features: {e}")
            return False
    
    def get_content_based_recommendations(self, user_profile: Dict, n_recommendations=20) -> List[Dict]:
        """Get content-based recommendations using user profile"""
        try:
            if self.content_features is None:
                if not self.build_content_features():
                    return []
            
            # Build user preference vector
            user_preferences = []
            for genre, weight in user_profile.get('genres', {}).items():
                user_preferences.extend([genre.lower()] * int(weight * 10))
            
            for lang, weight in user_profile.get('languages', {}).items():
                user_preferences.extend([lang.lower()] * int(weight * 10))
            
            if not user_preferences:
                return []
            
            # Create user preference description
            user_description = ' '.join(user_preferences)
            
            # Transform user preferences
            user_vector = self.tfidf_vectorizer.transform([user_description])
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(user_vector, self.content_features)[0]
            
            # Get user's interacted content
            user_id = user_profile.get('user_id')
            interacted_content = set()
            if user_id:
                interactions = UserInteraction.query.filter_by(user_id=user_id).all()
                interacted_content = set([i.content_id for i in interactions])
            
            # Get top recommendations
            recommendations = []
            sorted_indices = np.argsort(similarity_scores)[::-1]
            
            reverse_mapping = {idx: content_id for content_id, idx in self.content_mapping.items()}
            
            for idx in sorted_indices:
                if len(recommendations) >= n_recommendations:
                    break
                
                content_id = reverse_mapping.get(idx)
                if content_id and content_id not in interacted_content:
                    content = Content.query.get(content_id)
                    if content and similarity_scores[idx] > 0.1:  # Minimum similarity threshold
                        recommendations.append({
                            'content_id': content_id,
                            'score': float(similarity_scores[idx]),
                            'method': 'content_based_filtering',
                            'content': content
                        })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting content-based recommendations: {e}")
            return []

class HybridRecommendationEngine:
    """Hybrid recommendation engine combining multiple approaches"""
    
    def __init__(self):
        self.collaborative_engine = CollaborativeFilteringEngine()
        self.content_engine = ContentBasedFilteringEngine()
        self.profile_analyzer = UserProfileAnalyzer()
        
        # Weights for different recommendation methods
        self.method_weights = {
            'collaborative_filtering': 0.4,
            'content_based_filtering': 0.3,
            'popularity_based': 0.15,
            'trending_based': 0.1,
            'diversity_boost': 0.05
        }
    
    def get_hybrid_recommendations(self, user_id: int, content_type: Optional[str] = None, 
                                 n_recommendations: int = 20) -> List[Dict]:
        """Get hybrid recommendations combining multiple methods"""
        try:
            # Build user profile
            user_profile = self.profile_analyzer.build_user_profile(user_id)
            
            all_recommendations = []
            
            # 1. Collaborative Filtering
            try:
                if not self.collaborative_engine.model:
                    self.collaborative_engine.train_matrix_factorization()
                
                collab_recs = self.collaborative_engine.get_collaborative_recommendations(
                    user_id, n_recommendations * 2
                )
                all_recommendations.extend(collab_recs)
            except Exception as e:
                logger.warning(f"Collaborative filtering failed: {e}")
            
            # 2. Content-Based Filtering
            try:
                content_recs = self.content_engine.get_content_based_recommendations(
                    user_profile, n_recommendations * 2
                )
                all_recommendations.extend(content_recs)
            except Exception as e:
                logger.warning(f"Content-based filtering failed: {e}")
            
            # 3. Popularity-Based Recommendations
            try:
                popularity_recs = self._get_popularity_recommendations(
                    user_profile, content_type, n_recommendations
                )
                all_recommendations.extend(popularity_recs)
            except Exception as e:
                logger.warning(f"Popularity-based recommendations failed: {e}")
            
            # 4. Trending Recommendations
            try:
                trending_recs = self._get_trending_recommendations(
                    user_profile, content_type, n_recommendations // 2
                )
                all_recommendations.extend(trending_recs)
            except Exception as e:
                logger.warning(f"Trending recommendations failed: {e}")
            
            # 5. Combine and rank all recommendations
            final_recommendations = self._combine_and_rank_recommendations(
                all_recommendations, user_profile, n_recommendations
            )
            
            # 6. Apply diversity and freshness
            final_recommendations = self._apply_diversity_and_freshness(
                final_recommendations, user_profile
            )
            
            # 7. Filter by content type if specified
            if content_type:
                final_recommendations = [
                    rec for rec in final_recommendations 
                    if rec['content'].content_type == content_type
                ]
            
            return final_recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting hybrid recommendations: {e}")
            return []
    
    def _get_popularity_recommendations(self, user_profile: Dict, content_type: Optional[str], 
                                      n_recommendations: int) -> List[Dict]:
        """Get popularity-based recommendations"""
        try:
            query = Content.query.filter(Content.popularity.isnot(None))
            
            if content_type:
                query = query.filter(Content.content_type == content_type)
            
            # Apply language preferences
            preferred_languages = list(user_profile.get('languages', {}).keys())
            if preferred_languages:
                language_filters = []
                for lang in preferred_languages[:3]:  # Top 3 preferred languages
                    language_filters.append(Content.languages.contains(lang))
                if language_filters:
                    query = query.filter(or_(*language_filters))
            
            popular_content = query.order_by(Content.popularity.desc()).limit(n_recommendations * 2).all()
            
            recommendations = []
            for content in popular_content:
                recommendations.append({
                    'content_id': content.id,
                    'score': content.popularity / 1000.0,  # Normalize popularity score
                    'method': 'popularity_based',
                    'content': content
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting popularity recommendations: {e}")
            return []
    
    def _get_trending_recommendations(self, user_profile: Dict, content_type: Optional[str], 
                                    n_recommendations: int) -> List[Dict]:
        """Get trending recommendations"""
        try:
            query = Content.query.filter(Content.is_trending == True)
            
            if content_type:
                query = query.filter(Content.content_type == content_type)
            
            trending_content = query.order_by(Content.created_at.desc()).limit(n_recommendations).all()
            
            recommendations = []
            for content in trending_content:
                # Calculate trending score based on recency and popularity
                days_since_added = (datetime.utcnow() - content.created_at).days
                trending_score = (content.popularity or 100) / (1 + days_since_added * 0.1)
                
                recommendations.append({
                    'content_id': content.id,
                    'score': trending_score / 1000.0,
                    'method': 'trending_based',
                    'content': content
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
            return []
    
    def _combine_and_rank_recommendations(self, all_recommendations: List[Dict], 
                                        user_profile: Dict, n_recommendations: int) -> List[Dict]:
        """Combine and rank recommendations from different methods"""
        try:
            # Group recommendations by content_id
            content_scores = defaultdict(list)
            content_objects = {}
            
            for rec in all_recommendations:
                content_id = rec['content_id']
                content_scores[content_id].append((rec['score'], rec['method']))
                content_objects[content_id] = rec['content']
            
            # Calculate final scores
            final_recommendations = []
            
            for content_id, scores in content_scores.items():
                content = content_objects[content_id]
                
                # Calculate weighted average score
                total_score = 0
                total_weight = 0
                
                method_scores = defaultdict(float)
                for score, method in scores:
                    method_scores[method] = max(method_scores[method], score)
                
                for method, score in method_scores.items():
                    weight = self.method_weights.get(method, 0.1)
                    total_score += score * weight
                    total_weight += weight
                
                if total_weight > 0:
                    final_score = total_score / total_weight
                    
                    # Apply user profile boosters
                    final_score = self._apply_profile_boosters(final_score, content, user_profile)
                    
                    final_recommendations.append({
                        'content_id': content_id,
                        'score': final_score,
                        'content': content,
                        'methods_used': list(method_scores.keys())
                    })
            
            # Sort by score
            final_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error combining recommendations: {e}")
            return []
    
    def _apply_profile_boosters(self, score: float, content: Content, user_profile: Dict) -> float:
        """Apply user profile-based score boosters"""
        try:
            boosted_score = score
            
            # Genre preference booster
            try:
                content_genres = json.loads(content.genres or '[]')
                user_genres = user_profile.get('genres', {})
                
                for genre in content_genres:
                    genre_preference = user_genres.get(genre.lower(), 0)
                    boosted_score += genre_preference * 0.2
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Language preference booster
            try:
                content_languages = json.loads(content.languages or '[]')
                user_languages = user_profile.get('languages', {})
                
                for lang in content_languages:
                    lang_preference = user_languages.get(lang.lower(), 0)
                    boosted_score += lang_preference * 0.15
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Quality preference booster
            if content.rating and user_profile.get('quality_preference', 5.0):
                quality_match = 1.0 - abs(content.rating - user_profile['quality_preference']) / 10.0
                boosted_score += quality_match * 0.1
            
            # Content type preference booster
            content_type_pref = user_profile.get('content_types', {}).get(content.content_type, 0)
            boosted_score += content_type_pref * 0.1
            
            # Runtime preference booster
            if content.runtime and user_profile.get('preferred_runtime'):
                preferred_runtime = user_profile['preferred_runtime']
                runtime_match = 1.0 - abs(content.runtime - preferred_runtime.get('avg', 120)) / 180.0
                boosted_score += max(0, runtime_match) * 0.05
            
            return boosted_score
            
        except Exception as e:
            logger.error(f"Error applying profile boosters: {e}")
            return score
    
    def _apply_diversity_and_freshness(self, recommendations: List[Dict], 
                                     user_profile: Dict) -> List[Dict]:
        """Apply diversity and freshness to recommendations"""
        try:
            if not recommendations:
                return recommendations
            
            diversity_score = user_profile.get('diversity_score', 0.5)
            
            # If user likes diversity, promote diverse content
            if diversity_score > 0.6:
                seen_genres = set()
                diverse_recommendations = []
                other_recommendations = []
                
                for rec in recommendations:
                    try:
                        content_genres = json.loads(rec['content'].genres or '[]')
                        primary_genre = content_genres[0].lower() if content_genres else 'unknown'
                        
                        if primary_genre not in seen_genres:
                            seen_genres.add(primary_genre)
                            rec['score'] += 0.1  # Diversity boost
                            diverse_recommendations.append(rec)
                        else:
                            other_recommendations.append(rec)
                    except (json.JSONDecodeError, TypeError, IndexError):
                        other_recommendations.append(rec)
                
                # Interleave diverse and other recommendations
                final_recommendations = []
                for i in range(max(len(diverse_recommendations), len(other_recommendations))):
                    if i < len(diverse_recommendations):
                        final_recommendations.append(diverse_recommendations[i])
                    if i < len(other_recommendations):
                        final_recommendations.append(other_recommendations[i])
                
                recommendations = final_recommendations
            
            # Apply freshness boost for new releases
            for rec in recommendations:
                if rec['content'].release_date:
                    days_since_release = (datetime.utcnow().date() - rec['content'].release_date).days
                    if days_since_release <= 30:  # Released in last 30 days
                        rec['score'] += 0.05  # Freshness boost
            
            # Re-sort after applying boosts
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error applying diversity and freshness: {e}")
            return recommendations

class SequentialRecommendationEngine:
    """Sequential recommendation engine for session-based recommendations"""
    
    def __init__(self):
        self.sequence_length = 10
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        
    def build_transition_matrix(self, user_id: int):
        """Build transition matrix from user's interaction sequence"""
        try:
            interactions = UserInteraction.query.filter_by(user_id=user_id).order_by(
                UserInteraction.timestamp
            ).all()
            
            if len(interactions) < 2:
                return
            
            # Get content for interactions
            content_ids = [i.content_id for i in interactions]
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_map = {c.id: c for c in contents}
            
            # Build sequences
            sequences = []
            current_sequence = []
            
            for i, interaction in enumerate(interactions):
                content = content_map.get(interaction.content_id)
                if not content:
                    continue
                
                # Create item representation (genre + content_type)
                try:
                    genres = json.loads(content.genres or '[]')
                    primary_genre = genres[0].lower() if genres else 'unknown'
                    item_repr = f"{content.content_type}_{primary_genre}"
                    
                    current_sequence.append(item_repr)
                    
                    # If sequence is too long or there's a time gap, start new sequence
                    if len(current_sequence) > self.sequence_length:
                        sequences.append(current_sequence)
                        current_sequence = current_sequence[-3:]  # Keep last 3 items
                    
                    # Check time gap
                    if i > 0:
                        time_gap = (interaction.timestamp - interactions[i-1].timestamp).total_seconds()
                        if time_gap > 86400:  # More than 1 day gap
                            if len(current_sequence) > 1:
                                sequences.append(current_sequence)
                            current_sequence = [item_repr]
                
                except (json.JSONDecodeError, TypeError, IndexError):
                    continue
            
            if len(current_sequence) > 1:
                sequences.append(current_sequence)
            
            # Build transition matrix
            for sequence in sequences:
                for i in range(len(sequence) - 1):
                    current_item = sequence[i]
                    next_item = sequence[i + 1]
                    self.transition_matrix[current_item][next_item] += 1
            
            # Normalize transition probabilities
            for current_item in self.transition_matrix:
                total = sum(self.transition_matrix[current_item].values())
                if total > 0:
                    for next_item in self.transition_matrix[current_item]:
                        self.transition_matrix[current_item][next_item] /= total
            
        except Exception as e:
            logger.error(f"Error building transition matrix: {e}")
    
    def get_sequential_recommendations(self, user_id: int, last_interactions: List[int], 
                                     n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations based on sequential patterns"""
        try:
            # Build transition matrix for user
            self.build_transition_matrix(user_id)
            
            if not self.transition_matrix:
                return []
            
            # Get last interacted content
            if not last_interactions:
                recent_interactions = UserInteraction.query.filter_by(user_id=user_id).order_by(
                    UserInteraction.timestamp.desc()
                ).limit(3).all()
                last_interactions = [i.content_id for i in recent_interactions]
            
            if not last_interactions:
                return []
            
            # Get content representations for last interactions
            last_contents = Content.query.filter(Content.id.in_(last_interactions)).all()
            last_representations = []
            
            for content in last_contents:
                try:
                    genres = json.loads(content.genres or '[]')
                    primary_genre = genres[0].lower() if genres else 'unknown'
                    item_repr = f"{content.content_type}_{primary_genre}"
                    last_representations.append(item_repr)
                except (json.JSONDecodeError, TypeError, IndexError):
                    continue
            
            if not last_representations:
                return []
            
            # Get next item predictions
            next_item_scores = defaultdict(float)
            
            for last_repr in last_representations:
                if last_repr in self.transition_matrix:
                    for next_repr, prob in self.transition_matrix[last_repr].items():
                        next_item_scores[next_repr] += prob
            
            # Convert back to content recommendations
            recommendations = []
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_content = set([i.content_id for i in user_interactions])
            
            for next_repr, score in sorted(next_item_scores.items(), key=lambda x: x[1], reverse=True):
                if len(recommendations) >= n_recommendations:
                    break
                
                # Parse representation
                try:
                    content_type, genre = next_repr.split('_', 1)
                    
                    # Find content matching this representation
                    matching_content = Content.query.filter(
                        Content.content_type == content_type,
                        Content.genres.contains(genre.title()),
                        ~Content.id.in_(interacted_content)
                    ).order_by(Content.rating.desc()).limit(5).all()
                    
                    for content in matching_content:
                        if content.id not in [r['content_id'] for r in recommendations]:
                            recommendations.append({
                                'content_id': content.id,
                                'score': score,
                                'method': 'sequential_pattern',
                                'content': content,
                                'pattern': next_repr
                            })
                            break
                
                except ValueError:
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting sequential recommendations: {e}")
            return []

class PersonalizedRecommendationService:
    """Main personalized recommendation service"""
    
    def __init__(self):
        self.hybrid_engine = HybridRecommendationEngine()
        self.sequential_engine = SequentialRecommendationEngine()
        self.profile_analyzer = UserProfileAnalyzer()
        
    def get_personalized_recommendations(self, user_id: int, content_type: Optional[str] = None,
                                       context: Optional[Dict] = None, n_recommendations: int = 20) -> Dict:
        """Get comprehensive personalized recommendations"""
        try:
            # Build user profile
            user_profile = self.profile_analyzer.build_user_profile(user_id)
            
            # Get hybrid recommendations (main algorithm)
            hybrid_recs = self.hybrid_engine.get_hybrid_recommendations(
                user_id, content_type, n_recommendations
            )
            
            # Get sequential recommendations
            sequential_recs = self.sequential_engine.get_sequential_recommendations(
                user_id, None, n_recommendations // 2
            )
            
            # Apply contextual filtering if context provided
            if context:
                hybrid_recs = self._apply_contextual_filtering(hybrid_recs, context, user_profile)
                sequential_recs = self._apply_contextual_filtering(sequential_recs, context, user_profile)
            
            # Combine and deduplicate
            all_recs = hybrid_recs + sequential_recs
            seen_content_ids = set()
            final_recs = []
            
            for rec in all_recs:
                if rec['content_id'] not in seen_content_ids:
                    seen_content_ids.add(rec['content_id'])
                    final_recs.append(rec)
                
                if len(final_recs) >= n_recommendations:
                    break
            
            # Format for API response
            formatted_recs = []
            for rec in final_recs:
                content = rec['content']
                
                # Ensure slug exists
                if not content.slug:
                    try:
                        content.ensure_slug()
                        db.session.commit()
                    except Exception:
                        content.slug = f"content-{content.id}"
                
                youtube_url = None
                if content.youtube_trailer_id:
                    youtube_url = f"https://www.youtube.com/watch?v={content.youtube_trailer_id}"
                
                formatted_recs.append({
                    'id': content.id,
                    'slug': content.slug,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': json.loads(content.genres or '[]'),
                    'languages': json.loads(content.languages or '[]'),
                    'rating': content.rating,
                    'poster_path': f"https://image.tmdb.org/t/p/w300{content.poster_path}" if content.poster_path and not content.poster_path.startswith('http') else content.poster_path,
                    'overview': content.overview[:150] + '...' if content.overview else '',
                    'youtube_trailer': youtube_url,
                    'recommendation_score': round(rec.get('score', 0), 3),
                    'recommendation_reason': self._generate_recommendation_reason(rec, user_profile),
                    'methods_used': rec.get('methods_used', [rec.get('method', 'hybrid')])
                })
            
            return {
                'recommendations': formatted_recs,
                'user_profile_summary': {
                    'favorite_genres': list(user_profile.get('genres', {}).keys())[:5],
                    'preferred_languages': list(user_profile.get('languages', {}).keys())[:3],
                    'content_type_preferences': user_profile.get('content_types', {}),
                    'quality_preference': user_profile.get('quality_preference', 7.0),
                    'diversity_score': user_profile.get('diversity_score', 0.5),
                    'interaction_count': user_profile.get('interaction_count', 0)
                },
                'metadata': {
                    'total_recommendations': len(formatted_recs),
                    'algorithms_used': ['hybrid_filtering', 'sequential_patterns', 'profile_based'],
                    'personalization_strength': min(user_profile.get('interaction_count', 0) / 50.0, 1.0),
                    'context_applied': context is not None,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return {
                'recommendations': [],
                'user_profile_summary': {},
                'metadata': {'error': str(e)}
            }
    
    def _apply_contextual_filtering(self, recommendations: List[Dict], 
                                  context: Dict, user_profile: Dict) -> List[Dict]:
        """Apply contextual filtering based on time, device, etc."""
        try:
            current_hour = datetime.utcnow().hour
            current_day = datetime.utcnow().weekday()
            
            # Weekend/weekday preferences
            is_weekend = current_day >= 5
            contextual_prefs = user_profile.get('contextual_preferences', {})
            
            for rec in recommendations:
                content = rec['content']
                
                # Time-based adjustments
                if 22 <= current_hour or current_hour <= 6:  # Night time
                    # Boost shorter content during night
                    if content.runtime and content.runtime <= 120:
                        rec['score'] += 0.05
                
                # Weekend/weekday adjustments
                try:
                    genres = json.loads(content.genres or '[]')
                    primary_genre = genres[0].lower() if genres else 'unknown'
                    
                    if is_weekend:
                        weekend_prefs = contextual_prefs.get('weekend_preferences', {})
                        if primary_genre in weekend_prefs:
                            rec['score'] += weekend_prefs[primary_genre] * 0.1
                    else:
                        weekday_prefs = contextual_prefs.get('weekday_preferences', {})
                        if primary_genre in weekday_prefs:
                            rec['score'] += weekday_prefs[primary_genre] * 0.1
                
                except (json.JSONDecodeError, TypeError, IndexError):
                    pass
            
            # Re-sort after contextual adjustments
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error applying contextual filtering: {e}")
            return recommendations
    
    def _generate_recommendation_reason(self, recommendation: Dict, user_profile: Dict) -> str:
        """Generate human-readable recommendation reason"""
        try:
            content = recommendation['content']
            methods = recommendation.get('methods_used', [recommendation.get('method', 'hybrid')])
            
            reasons = []
            
            # Genre-based reason
            try:
                content_genres = json.loads(content.genres or '[]')
                user_genres = user_profile.get('genres', {})
                
                for genre in content_genres[:2]:
                    if user_genres.get(genre.lower(), 0) > 0.3:
                        reasons.append(f"you enjoy {genre.lower()} content")
                        break
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Language-based reason
            try:
                content_languages = json.loads(content.languages or '[]')
                user_languages = user_profile.get('languages', {})
                
                for lang in content_languages:
                    if user_languages.get(lang.lower(), 0) > 0.3:
                        reasons.append(f"matches your {lang} preference")
                        break
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Method-based reason
            if 'collaborative_filtering' in methods:
                reasons.append("users with similar tastes loved this")
            elif 'sequential_pattern' in methods:
                reasons.append("follows your viewing pattern")
            elif 'trending_based' in methods:
                reasons.append("trending now")
            
            # Quality-based reason
            if content.rating and content.rating >= 8.0:
                reasons.append("highly rated")
            
            if reasons:
                return "Recommended because " + " and ".join(reasons[:2])
            else:
                return "Recommended based on your preferences"
                
        except Exception as e:
            logger.error(f"Error generating recommendation reason: {e}")
            return "Recommended for you"

# Initialize the recommendation service
recommendation_service = PersonalizedRecommendationService()

# API Routes
@personalized_bp.route('/api/recommendations/personalized', methods=['GET'])
@require_auth
def get_personalized_recommendations(current_user):
    """Get personalized recommendations for authenticated user"""
    try:
        content_type = request.args.get('type')  # movie, tv, anime
        limit = min(int(request.args.get('limit', 20)), 50)
        
        # Context information
        context = {
            'device': request.headers.get('User-Agent', ''),
            'time': datetime.utcnow().hour,
            'day': datetime.utcnow().weekday()
        }
        
        recommendations = recommendation_service.get_personalized_recommendations(
            current_user.id, content_type, context, limit
        )
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}")
        return jsonify({'error': 'Failed to get personalized recommendations'}), 500

@personalized_bp.route('/api/recommendations/personalized/categories', methods=['GET'])
@require_auth
def get_personalized_categories(current_user):
    """Get personalized recommendations grouped by categories"""
    try:
        categories = {
            'for_you': recommendation_service.get_personalized_recommendations(
                current_user.id, None, None, 15
            )['recommendations'],
            'movies': recommendation_service.get_personalized_recommendations(
                current_user.id, 'movie', None, 10
            )['recommendations'],
            'tv_shows': recommendation_service.get_personalized_recommendations(
                current_user.id, 'tv', None, 10
            )['recommendations'],
            'anime': recommendation_service.get_personalized_recommendations(
                current_user.id, 'anime', None, 10
            )['recommendations']
        }
        
        return jsonify({
            'categories': categories,
            'metadata': {
                'personalized': True,
                'user_id': current_user.id,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting personalized categories: {e}")
        return jsonify({'error': 'Failed to get personalized categories'}), 500

@personalized_bp.route('/api/user/profile', methods=['GET'])
@require_auth
def get_user_profile(current_user):
    """Get detailed user profile and preferences"""
    try:
        profile = recommendation_service.profile_analyzer.build_user_profile(current_user.id)
        
        return jsonify({
            'profile': profile,
            'recommendations_available': profile['interaction_count'] > 0
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return jsonify({'error': 'Failed to get user profile'}), 500

@personalized_bp.route('/api/recommendations/feedback', methods=['POST'])
@require_auth
def record_recommendation_feedback(current_user):
    """Record user feedback on recommendations for learning"""
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'feedback_type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Record feedback as interaction
        feedback_interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=f"feedback_{data['feedback_type']}",  # feedback_like, feedback_dislike, etc.
            interaction_metadata=json.dumps({
                'recommendation_score': data.get('recommendation_score'),
                'recommendation_reason': data.get('recommendation_reason'),
                'user_comment': data.get('comment', '')
            })
        )
        
        db.session.add(feedback_interaction)
        db.session.commit()
        
        # Clear user profile cache to incorporate new feedback
        if cache:
            cache.delete(f"user_profile:{current_user.id}")
        
        return jsonify({'message': 'Feedback recorded successfully'}), 201
        
    except Exception as e:
        logger.error(f"Error recording recommendation feedback: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record feedback'}), 500

@personalized_bp.route('/api/recommendations/retrain', methods=['POST'])
@require_auth
def retrain_models(current_user):
    """Retrain recommendation models (admin only)"""
    try:
        if not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        # Retrain collaborative filtering model
        success = recommendation_service.hybrid_engine.collaborative_engine.train_matrix_factorization()
        
        # Rebuild content features
        content_success = recommendation_service.hybrid_engine.content_engine.build_content_features()
        
        # Clear all recommendation caches
        if cache:
            # Clear user profile caches
            users = User.query.all()
            for user in users:
                cache.delete(f"user_profile:{user.id}")
        
        return jsonify({
            'message': 'Models retrained successfully',
            'collaborative_filtering': success,
            'content_based': content_success
        }), 200
        
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        return jsonify({'error': 'Failed to retrain models'}), 500

@personalized_bp.route('/api/recommendations/similar-users', methods=['GET'])
@require_auth
def get_similar_users(current_user):
    """Get users with similar preferences (for debugging/analytics)"""
    try:
        if not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        user_profile = recommendation_service.profile_analyzer.build_user_profile(current_user.id)
        
        # Find users with similar genre preferences
        all_users = User.query.all()
        similar_users = []
        
        for user in all_users:
            if user.id == current_user.id:
                continue
            
            other_profile = recommendation_service.profile_analyzer.build_user_profile(user.id)
            
            # Calculate genre similarity
            user_genres = user_profile.get('genres', {})
            other_genres = other_profile.get('genres', {})
            
            if user_genres and other_genres:
                # Calculate cosine similarity
                common_genres = set(user_genres.keys()) & set(other_genres.keys())
                if common_genres:
                    similarity = sum(user_genres[g] * other_genres[g] for g in common_genres)
                    
                    if similarity > 0.3:  # Threshold for similarity
                        similar_users.append({
                            'user_id': user.id,
                            'username': user.username,
                            'similarity_score': round(similarity, 3),
                            'common_genres': list(common_genres),
                            'interaction_count': other_profile.get('interaction_count', 0)
                        })
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return jsonify({
            'similar_users': similar_users[:10],
            'total_found': len(similar_users)
        }), 200
        
    except Exception as e:
        logger.error(f"Error finding similar users: {e}")
        return jsonify({'error': 'Failed to find similar users'}), 500