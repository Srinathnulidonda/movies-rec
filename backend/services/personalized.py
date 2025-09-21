# backend/services/personalized.py (Enhanced Version with Advanced ML)
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
from sqlalchemy import func, and_, or_, desc, text
from sqlalchemy.orm import joinedload
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

class AdvancedUserProfiler:
    """Advanced user profiling with deep behavioral analysis"""
    
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
        
        # Advanced feature extractors
        self.genre_vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        self.content_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
        self.search_vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
        
    def build_comprehensive_profile(self, user_id: int) -> Dict[str, Any]:
        """Build ultra-comprehensive user profile with advanced analytics"""
        try:
            cache_key = f"advanced_profile:{user_id}"
            if cache:
                cached = cache.get(cache_key)
                if cached:
                    return cached
            
            # Get all user interactions with content
            interactions = db.session.query(UserInteraction, Content).join(
                Content, UserInteraction.content_id == Content.id
            ).filter(UserInteraction.user_id == user_id).all()
            
            if not interactions:
                return self._get_enhanced_default_profile(user_id)
            
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
                
                # Metadata
                'last_updated': datetime.utcnow().isoformat(),
                'confidence_score': 0.0
            }
            
            # Process interactions chronologically
            sorted_interactions = sorted(interactions, key=lambda x: x[0].timestamp)
            
            # Extract behavioral signals
            self._extract_core_preferences(profile, sorted_interactions)
            self._extract_temporal_patterns(profile, sorted_interactions)
            self._extract_sequential_patterns(profile, sorted_interactions)
            self._extract_search_patterns(profile, user_id)
            self._extract_social_signals(profile, user_id)
            self._extract_content_creator_preferences(profile, sorted_interactions)
            self._calculate_prediction_features(profile, sorted_interactions)
            self._calculate_confidence_scores(profile, sorted_interactions)
            
            # Convert defaultdicts to regular dicts for JSON serialization
            profile = self._serialize_profile(profile)
            
            # Cache the profile
            if cache:
                cache.set(cache_key, profile, timeout=1800)  # 30 minutes
            
            return profile
            
        except Exception as e:
            logger.error(f"Error building comprehensive profile for user {user_id}: {e}")
            return self._get_enhanced_default_profile(user_id)
    
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
            interaction_data = self.interaction_weights.get(interaction.interaction_type, 
                                                          {'weight': 0.1, 'confidence': 0.2})
            base_weight = interaction_data['weight'] * temporal_weight
            
            # Apply rating boost
            rating_boost = 1.0
            if interaction.rating:
                rating_boost = (interaction.rating / 5.0) * 1.5  # Boost for high ratings
                quality_ratings.append(interaction.rating)
            
            final_weight = base_weight * rating_boost
            total_weight += final_weight
            
            # Extract genres with weighted importance
            try:
                genres = json.loads(content.genres or '[]')
                for i, genre in enumerate(genres[:5]):  # Top 5 genres
                    importance = 1.0 / (i + 1)  # Diminishing importance
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
            weighted_runtimes = [runtime for runtime, weight in runtimes for _ in range(int(weight * 10))]
            profile['runtime_preferences'] = {
                'preferred_range': [np.percentile(weighted_runtimes, 25), np.percentile(weighted_runtimes, 75)],
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
            # Create content representation
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
                
                # Detect sequence breaks (more than 1 day gap)
                if i > 0:
                    time_gap = (interaction.timestamp - interactions[i-1][0].timestamp).total_seconds()
                    if time_gap > 86400 or len(current_sequence) > 10:  # 1 day or max length
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
            # Get search interactions
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
                # Analyze search patterns
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
            # Find users with similar preferences
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            user_content_ids = set([i.content_id for i in user_interactions])
            
            if len(user_content_ids) < 5:
                profile['similarity_clusters'] = []
                return
            
            # Find users who interacted with similar content
            similar_users = db.session.query(UserInteraction.user_id, func.count().label('common_content')).filter(
                UserInteraction.content_id.in_(user_content_ids),
                UserInteraction.user_id != user_id
            ).group_by(UserInteraction.user_id).having(func.count() >= 3).all()
            
            similarity_scores = []
            for other_user_id, common_count in similar_users:
                # Calculate Jaccard similarity
                other_interactions = UserInteraction.query.filter_by(user_id=other_user_id).all()
                other_content_ids = set([i.content_id for i in other_interactions])
                
                intersection = len(user_content_ids & other_content_ids)
                union = len(user_content_ids | other_content_ids)
                
                if union > 0:
                    jaccard_similarity = intersection / union
                    if jaccard_similarity > 0.1:  # Minimum similarity threshold
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
                        interaction_data = self.interaction_weights.get(interaction.interaction_type, 
                                                                      {'weight': 0.1})
                        weight = interaction_data['weight']
                        if interaction.rating:
                            weight *= (interaction.rating / 5.0)
                        break
                
                if content_person.role_type == 'crew' and content_person.job == 'Director':
                    director_counts[person.name] += weight
                elif content_person.role_type == 'cast':
                    actor_counts[person.name] += weight
            
            # Keep top preferences
            profile['director_preferences'] = dict(sorted(director_counts.items(), 
                                                        key=lambda x: x[1], reverse=True)[:20])
            profile['cast_crew_preferences'] = dict(sorted(actor_counts.items(), 
                                                         key=lambda x: x[1], reverse=True)[:30])
            
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
        
        # Predict trending interests (genres gaining momentum)
        if len(time_windows) >= 2:
            recent_genres = preference_evolution.get('30_days', {})
            older_genres = preference_evolution.get('90_days', {})
            
            trending = []
            for genre, recent_count in recent_genres.items():
                older_count = older_genres.get(genre, 0)
                if recent_count > older_count * 1.5:  # 50% increase
                    trending.append(genre)
            
            profile['trending_interests'] = trending
    
    def _calculate_confidence_scores(self, profile: Dict, interactions: List[Tuple]) -> None:
        """Calculate confidence scores for profile reliability"""
        interaction_count = len(interactions)
        
        # Base confidence from interaction count
        base_confidence = min(interaction_count / 50.0, 1.0)  # Max confidence at 50 interactions
        
        # Adjust for interaction diversity
        interaction_types = set([i.interaction_type for i, _ in interactions])
        diversity_bonus = len(interaction_types) / 6.0  # Assuming 6 main interaction types
        
        # Adjust for rating data availability
        rating_interactions = [i for i, _ in interactions if i.rating is not None]
        rating_bonus = len(rating_interactions) / max(interaction_count, 1) * 0.3
        
        # Adjust for temporal spread
        if interactions:
            time_span = (interactions[-1][0].timestamp - interactions[0][0].timestamp).days
            temporal_bonus = min(time_span / 90.0, 1.0) * 0.2  # Max bonus for 90+ days
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
        threshold = max_count * 0.7  # Top 70% threshold
        
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
                if len(set(content_types)) == 1:  # Same content type
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
            # Simple tokenization and filtering
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
            
            # Check for specific patterns
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
    
    def _get_enhanced_default_profile(self, user_id: int) -> Dict[str, Any]:
        """Return enhanced default profile for new users"""
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
            'confidence_score': 0.0,
            'last_updated': datetime.utcnow().isoformat()
        }

class NeuralCollaborativeFiltering:
    """Advanced neural collaborative filtering with deep embeddings"""
    
    def __init__(self, embedding_dim=64, hidden_layers=[128, 64, 32]):
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.user_mapping = {}
        self.item_mapping = {}
        self.model_trained = False
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for neural collaborative filtering"""
        try:
            # Get all interactions with ratings or implicit feedback
            interactions = UserInteraction.query.filter(
                UserInteraction.interaction_type.in_(['rating', 'favorite', 'like', 'view', 'watchlist'])
            ).all()
            
            if not interactions:
                return np.array([]), np.array([]), np.array([])
            
            # Create user and item mappings
            users = list(set([i.user_id for i in interactions]))
            items = list(set([i.content_id for i in interactions]))
            
            self.user_mapping = {user_id: idx for idx, user_id in enumerate(users)}
            self.item_mapping = {item_id: idx for idx, item_id in enumerate(items)}
            
            # Prepare training data
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
        """Train neural collaborative filtering embeddings using matrix factorization approximation"""
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
            from sklearn.decomposition import TruncatedSVD
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
                        np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding)
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
                        'method': 'neural_collaborative_filtering',
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
        self.genre_embeddings = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.content_features = None
        self.content_mapping = {}
        
    def build_content_embeddings(self):
        """Build comprehensive content embeddings"""
        try:
            contents = Content.query.all()
            if not contents:
                return False
            
            # Create comprehensive content descriptions
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
        self.state_representations = {}
        
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
            
            # Create enhanced state representations
            states = []
            for interaction in interactions:
                content = content_map.get(interaction.content_id)
                if content:
                    state = self._create_state_representation(content, interaction)
                    states.append(state)
            
            # Build transition models with different granularities
            self._build_genre_transitions(states)
            self._build_content_type_transitions(states)
            self._build_quality_transitions(states)
            self._build_temporal_transitions(states)
            
        except Exception as e:
            logger.error(f"Error building sequence models: {e}")
    
    def _create_state_representation(self, content: Content, interaction: UserInteraction) -> Dict:
        """Create comprehensive state representation"""
        try:
            genres = json.loads(content.genres or '[]')
            languages = json.loads(content.languages or '[]')
            
            return {
                'content_id': content.id,
                'content_type': content.content_type,
                'primary_genre': genres[0].lower() if genres else 'unknown',
                'secondary_genre': genres[1].lower() if len(genres) > 1 else None,
                'language': languages[0].lower() if languages else 'unknown',
                'rating_tier': self._get_rating_tier(content.rating),
                'popularity_tier': self._get_popularity_tier(content.popularity),
                'runtime_tier': self._get_runtime_tier(content.runtime),
                'release_decade': self._get_release_decade(content.release_date),
                'interaction_type': interaction.interaction_type,
                'user_rating': interaction.rating,
                'timestamp': interaction.timestamp
            }
        except Exception:
            return {}
    
    def _build_genre_transitions(self, states: List[Dict]):
        """Build genre-based transition matrix"""
        for i in range(len(states) - 1):
            current_genre = states[i]['primary_genre']
            next_genre = states[i + 1]['primary_genre']
            
            if current_genre != 'unknown' and next_genre != 'unknown':
                self.transition_matrices['genre'][f"{current_genre}->{next_genre}"] += 1
    
    def _build_content_type_transitions(self, states: List[Dict]):
        """Build content type transition matrix"""
        for i in range(len(states) - 1):
            current_type = states[i]['content_type']
            next_type = states[i + 1]['content_type']
            
            self.transition_matrices['content_type'][f"{current_type}->{next_type}"] += 1
    
    def _build_quality_transitions(self, states: List[Dict]):
        """Build quality-based transition matrix"""
        for i in range(len(states) - 1):
            current_quality = states[i]['rating_tier']
            next_quality = states[i + 1]['rating_tier']
            
            if current_quality and next_quality:
                self.transition_matrices['quality'][f"{current_quality}->{next_quality}"] += 1
    
    def _build_temporal_transitions(self, states: List[Dict]):
        """Build temporal pattern transitions"""
        for i in range(len(states) - 1):
            current_time = states[i]['timestamp']
            next_time = states[i + 1]['timestamp']
            
            time_gap = (next_time - current_time).total_seconds() / 3600  # hours
            
            if time_gap <= 2:  # Immediate viewing
                pattern = 'immediate'
            elif time_gap <= 24:  # Same day
                pattern = 'same_day'
            elif time_gap <= 168:  # Same week
                pattern = 'same_week'
            else:
                pattern = 'delayed'
            
            current_repr = f"{states[i]['content_type']}_{states[i]['primary_genre']}"
            next_repr = f"{states[i + 1]['content_type']}_{states[i + 1]['primary_genre']}"
            
            self.transition_matrices[f'temporal_{pattern}'][f"{current_repr}->{next_repr}"] += 1
    
    def get_sequence_predictions(self, user_id: int, recent_interactions: List[int], 
                               n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations based on sequence patterns"""
        try:
            self.build_sequence_models(user_id)
            
            if not recent_interactions:
                # Get recent interactions
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
                # Predict based on different transition models
                self._add_genre_predictions(content, predictions)
                self._add_content_type_predictions(content, predictions)
                self._add_quality_predictions(content, predictions)
            
            # Convert predictions to recommendations
            recommendations = []
            user_interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_content = set([i.content_id for i in user_interactions])
            
            for prediction_key, score in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                if len(recommendations) >= n_recommendations:
                    break
                
                # Parse prediction key and find matching content
                matching_content = self._find_matching_content(prediction_key, interacted_content)
                
                for content in matching_content[:2]:  # Max 2 per prediction
                    if content.id not in [r['content_id'] for r in recommendations]:
                        recommendations.append({
                            'content_id': content.id,
                            'score': score,
                            'method': 'sequence_aware',
                            'content': content,
                            'prediction_basis': prediction_key
                        })
                        break
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting sequence predictions: {e}")
            return []
    
    def _add_genre_predictions(self, content: Content, predictions: Dict):
        """Add genre-based predictions"""
        try:
            genres = json.loads(content.genres or '[]')
            primary_genre = genres[0].lower() if genres else 'unknown'
            
            if primary_genre != 'unknown':
                for transition, count in self.transition_matrices['genre'].items():
                    if transition.startswith(f"{primary_genre}->"):
                        next_genre = transition.split('->')[1]
                        predictions[f"genre_{next_genre}"] += count * 0.3
        except (json.JSONDecodeError, TypeError):
            pass
    
    def _add_content_type_predictions(self, content: Content, predictions: Dict):
        """Add content type predictions"""
        content_type = content.content_type
        
        for transition, count in self.transition_matrices['content_type'].items():
            if transition.startswith(f"{content_type}->"):
                next_type = transition.split('->')[1]
                predictions[f"type_{next_type}"] += count * 0.2
    
    def _add_quality_predictions(self, content: Content, predictions: Dict):
        """Add quality-based predictions"""
        quality_tier = self._get_rating_tier(content.rating)
        
        if quality_tier:
            for transition, count in self.transition_matrices['quality'].items():
                if transition.startswith(f"{quality_tier}->"):
                    next_quality = transition.split('->')[1]
                    predictions[f"quality_{next_quality}"] += count * 0.1
    
    def _find_matching_content(self, prediction_key: str, interacted_content: Set[int]) -> List[Content]:
        """Find content matching prediction criteria"""
        try:
            prediction_type, value = prediction_key.split('_', 1)
            
            query = Content.query.filter(~Content.id.in_(interacted_content))
            
            if prediction_type == 'genre':
                query = query.filter(Content.genres.contains(value.title()))
            elif prediction_type == 'type':
                query = query.filter(Content.content_type == value)
            elif prediction_type == 'quality':
                if value == 'high':
                    query = query.filter(Content.rating >= 7.0)
                elif value == 'medium':
                    query = query.filter(and_(Content.rating >= 5.0, Content.rating < 7.0))
                elif value == 'low':
                    query = query.filter(Content.rating < 5.0)
            
            return query.order_by(Content.popularity.desc()).limit(5).all()
            
        except Exception:
            return []
    
    def _get_rating_tier(self, rating: float) -> Optional[str]:
        """Get rating tier for content"""
        if not rating:
            return None
        
        if rating >= 7.0:
            return 'high'
        elif rating >= 5.0:
            return 'medium'
        else:
            return 'low'
    
    def _get_popularity_tier(self, popularity: float) -> Optional[str]:
        """Get popularity tier for content"""
        if not popularity:
            return None
        
        if popularity >= 100:
            return 'high'
        elif popularity >= 50:
            return 'medium'
        else:
            return 'low'
    
    def _get_runtime_tier(self, runtime: int) -> Optional[str]:
        """Get runtime tier for content"""
        if not runtime:
            return None
        
        if runtime <= 90:
            return 'short'
        elif runtime <= 150:
            return 'medium'
        else:
            return 'long'
    
    def _get_release_decade(self, release_date) -> Optional[str]:
        """Get release decade"""
        if not release_date:
            return None
        
        decade = (release_date.year // 10) * 10
        return f"{decade}s"

class HyperPersonalizedEngine:
    """Ultra-advanced personalized recommendation engine"""
    
    def __init__(self):
        self.profiler = AdvancedUserProfiler()
        self.ncf_engine = NeuralCollaborativeFiltering()
        self.content_embeddings = AdvancedContentEmbeddings()
        self.sequence_engine = SequenceAwareRecommender()
        
        # Advanced weighting system
        self.algorithm_weights = {
            'neural_collaborative': 0.35,
            'content_embeddings': 0.25,
            'sequence_aware': 0.20,
            'profile_based': 0.15,
            'exploration_boost': 0.05
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
    def get_hyper_personalized_recommendations(self, user_id: int, content_type: Optional[str] = None,
                                             context: Optional[Dict] = None, 
                                             n_recommendations: int = 20) -> Dict:
        """Get ultra-personalized recommendations with maximum accuracy"""
        try:
            # Build comprehensive user profile
            user_profile = self.profiler.build_comprehensive_profile(user_id)
            confidence_level = user_profile.get('confidence_score', 0.0)
            
            # Initialize recommendation sources
            all_recommendations = []
            algorithm_performance = {}
            
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
            
            # 2. Advanced Content Embeddings
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
                all_recommendations, user_profile, confidence_level
            )
            
            # 7. Apply contextual and temporal filters
            if context:
                final_recommendations = self._apply_advanced_contextual_filtering(
                    final_recommendations, context, user_profile
                )
            
            # 8. Apply diversity and serendipity
            final_recommendations = self._apply_diversity_and_serendipity(
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
            
            return {
                'recommendations': formatted_recommendations,
                'user_profile_insights': {
                    'confidence_level': confidence_level,
                    'profile_strength': user_profile.get('profile_strength', 0),
                    'top_genres': list(user_profile.get('genre_preferences', {}).keys())[:5],
                    'preferred_languages': list(user_profile.get('language_preferences', {}).keys())[:3],
                    'viewing_patterns': user_profile.get('viewing_patterns', {}),
                    'exploration_tendency': user_profile.get('exploration_tendency', 0.5),
                    'trending_interests': user_profile.get('trending_interests', [])
                },
                'recommendation_metadata': {
                    'total_recommendations': len(formatted_recommendations),
                    'algorithm_performance': algorithm_performance,
                    'personalization_strength': min(confidence_level * 100, 100),
                    'context_applied': context is not None,
                    'diversity_applied': True,
                    'serendipity_applied': True,
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
        """Get recommendations using advanced content embeddings"""
        try:
            # Get user's interaction history
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
            
            # Normalize weights
            interaction_weights = interaction_weights / np.sum(interaction_weights)
            
            # Create user profile vector
            user_profile_vector = np.average(user_content_embeddings, axis=0, weights=interaction_weights)
            
            # Find similar content
            recommendations = []
            interacted_content = set(content_ids)
            
            similarities = []
            for content_id, content_embedding in self.content_embeddings.content_embeddings.items():
                if content_id not in interacted_content:
                    similarity = cosine_similarity([user_profile_vector], [content_embedding])[0][0]
                    similarities.append((content_id, similarity))
            
            # Sort by similarity and get top recommendations
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
    
    def _get_profile_based_recommendations(self, user_profile: Dict, content_type: Optional[str], 
                                         n_recommendations: int) -> List[Dict]:
        """Get recommendations based on detailed user profile"""
        try:
            recommendations = []
            
            # Get preferred genres and languages
            preferred_genres = user_profile.get('genre_preferences', {})
            preferred_languages = user_profile.get('language_preferences', {})
            quality_prefs = user_profile.get('quality_preferences', {})
            runtime_prefs = user_profile.get('runtime_preferences', {})
            
            # Build query based on preferences
            query = Content.query
            
            if content_type:
                query = query.filter(Content.content_type == content_type)
            
            # Apply quality filters
            min_rating = quality_prefs.get('average_rating', 6.0) - 1.0
            if min_rating > 0:
                query = query.filter(Content.rating >= min_rating)
            
            # Apply runtime filters
            if runtime_prefs.get('preferred_range'):
                min_runtime, max_runtime = runtime_prefs['preferred_range']
                query = query.filter(
                    and_(Content.runtime >= min_runtime - 20, Content.runtime <= max_runtime + 20)
                )
            
            # Get candidates
            candidates = query.order_by(Content.popularity.desc()).limit(n_recommendations * 5).all()
            
            # Score candidates based on profile
            scored_candidates = []
            
            for content in candidates:
                score = self._calculate_profile_match_score(content, user_profile)
                if score > 0.3:  # Minimum threshold
                    scored_candidates.append({
                        'content_id': content.id,
                        'score': score,
                        'method': 'profile_based',
                        'content': content
                    })
            
            # Sort by score and return top N
            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
            return scored_candidates[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in profile-based recommendations: {e}")
            return []
    
    def _get_exploration_recommendations(self, user_profile: Dict, content_type: Optional[str], 
                                       n_recommendations: int) -> List[Dict]:
        """Get exploration recommendations for content discovery"""
        try:
            exploration_tendency = user_profile.get('exploration_tendency', 0.5)
            
            if exploration_tendency < 0.3:
                return []  # User doesn't like exploration
            
            # Get user's interaction history
            user_id = user_profile['user_id']
            interactions = UserInteraction.query.filter_by(user_id=user_id).all()
            interacted_genres = set()
            interacted_languages = set()
            interacted_content_ids = set([i.content_id for i in interactions])
            
            # Extract interacted genres and languages
            content_ids = [i.content_id for i in interactions]
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            
            for content in contents:
                try:
                    genres = json.loads(content.genres or '[]')
                    languages = json.loads(content.languages or '[]')
                    interacted_genres.update([g.lower() for g in genres])
                    interacted_languages.update([l.lower() for l in languages])
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Find content with unexplored characteristics
            query = Content.query.filter(~Content.id.in_(interacted_content_ids))
            
            if content_type:
                query = query.filter(Content.content_type == content_type)
            
            # Get high-quality content from unexplored areas
            exploration_candidates = query.filter(
                Content.rating >= 7.0,  # High quality only for exploration
                Content.vote_count >= 100  # Ensure reliability
            ).order_by(Content.popularity.desc()).limit(n_recommendations * 3).all()
            
            recommendations = []
            
            for content in exploration_candidates:
                exploration_score = self._calculate_exploration_score(
                    content, interacted_genres, interacted_languages
                )
                
                if exploration_score > 0.5:
                    recommendations.append({
                        'content_id': content.id,
                        'score': exploration_score,
                        'method': 'exploration_boost',
                        'content': content
                    })
            
            # Sort by exploration score
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
    
    def _calculate_exploration_score(self, content: Content, interacted_genres: Set[str], 
                                   interacted_languages: Set[str]) -> float:
        """Calculate exploration score for content discovery"""
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
            
            # Language novelty
            try:
                content_languages = json.loads(content.languages or '[]')
                new_languages = [l.lower() for l in content_languages if l.lower() not in interacted_languages]
                lang_novelty = len(new_languages) / max(len(content_languages), 1) if content_languages else 0
                score += lang_novelty * 0.3
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Quality bonus (encourage high-quality exploration)
            if content.rating and content.rating >= 8.0:
                score += 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating exploration score: {e}")
            return 0.0
    
    def _advanced_recommendation_fusion(self, all_recommendations: List[Dict], 
                                      user_profile: Dict, confidence_level: float) -> List[Dict]:
        """Advanced fusion of recommendations from multiple algorithms"""
        try:
            # Group recommendations by content_id
            content_scores = defaultdict(list)
            content_objects = {}
            
            for rec in all_recommendations:
                content_id = rec['content_id']
                method = rec.get('method', 'unknown')
                score = rec['score']
                
                content_scores[content_id].append((score, method))
                content_objects[content_id] = rec['content']
            
            # Calculate fusion scores
            fused_recommendations = []
            
            for content_id, scores in content_scores.items():
                content = content_objects[content_id]
                
                # Calculate weighted fusion score
                total_score = 0.0
                total_weight = 0.0
                methods_used = []
                
                # Group scores by method
                method_scores = defaultdict(list)
                for score, method in scores:
                    method_scores[method].append(score)
                    methods_used.append(method)
                
                # Calculate method-wise scores
                for method, method_score_list in method_scores.items():
                    # Use max score for each method
                    best_score = max(method_score_list)
                    
                    # Get method weight (adjust based on confidence)
                    base_weight = self.algorithm_weights.get(method, 0.1)
                    
                    # Adjust weight based on user profile confidence
                    if confidence_level >= self.confidence_thresholds['high']:
                        # High confidence: prefer sophisticated methods
                        if method in ['neural_collaborative', 'sequence_aware']:
                            base_weight *= 1.2
                    elif confidence_level >= self.confidence_thresholds['medium']:
                        # Medium confidence: balanced approach
                        base_weight *= 1.0
                    else:
                        # Low confidence: prefer simple methods
                        if method in ['profile_based', 'content_embeddings']:
                            base_weight *= 1.1
                    
                    total_score += best_score * base_weight
                    total_weight += base_weight
                
                if total_weight > 0:
                    fusion_score = total_score / total_weight
                    
                    # Apply post-fusion boosters
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
            
            # Sort by fusion score
            fused_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return fused_recommendations
            
        except Exception as e:
            logger.error(f"Error in recommendation fusion: {e}")
            return []
    
    def _apply_post_fusion_boosters(self, score: float, content: Content, 
                                  user_profile: Dict, methods_used: List[str]) -> float:
        """Apply post-fusion score boosters"""
        try:
            boosted_score = score
            
            # Multi-method consensus bonus
            if len(set(methods_used)) >= 3:
                boosted_score += 0.1
            elif len(set(methods_used)) >= 2:
                boosted_score += 0.05
            
            # Fresh content bonus
            if content.release_date:
                days_since_release = (datetime.utcnow().date() - content.release_date).days
                if days_since_release <= 30:
                    boosted_score += 0.05
                elif days_since_release <= 90:
                    boosted_score += 0.02
            
            # Trending content bonus
            if content.is_trending:
                boosted_score += 0.03
            
            # High quality bonus
            if content.rating and content.rating >= 8.5:
                boosted_score += 0.04
            elif content.rating and content.rating >= 8.0:
                boosted_score += 0.02
            
            # Language preference bonus
            try:
                content_languages = json.loads(content.languages or '[]')
                user_languages = user_profile.get('language_preferences', {})
                
                for lang in content_languages:
                    lang_pref = user_languages.get(lang.lower(), 0)
                    if lang_pref > 0.5:  # Strong language preference
                        boosted_score += 0.03
                        break
            except (json.JSONDecodeError, TypeError):
                pass
            
            return min(boosted_score, 2.0)  # Cap the boost
            
        except Exception as e:
            logger.error(f"Error applying post-fusion boosters: {e}")
            return score
    
    def _apply_advanced_contextual_filtering(self, recommendations: List[Dict], 
                                           context: Dict, user_profile: Dict) -> List[Dict]:
        """Apply advanced contextual filtering"""
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
                
                # Device-based adjustments (if available in context)
                device = context.get('device', '').lower()
                if 'mobile' in device and content.runtime and content.runtime <= 120:
                    contextual_boost += 0.02  # Shorter content for mobile
                
                # Weekend/weekday patterns
                is_weekend = current_day >= 5
                contextual_prefs = user_profile.get('contextual_patterns', {})
                
                if is_weekend:
                    weekend_prefs = contextual_prefs.get('weekend_preferences', {})
                    # Apply weekend preferences
                elif current_day < 5:  # Weekday
                    weekday_prefs = contextual_prefs.get('weekday_preferences', {})
                    # Apply weekday preferences
                
                # Late night adjustments
                if 22 <= current_hour or current_hour <= 6:
                    if content.runtime and content.runtime <= 90:
                        contextual_boost += 0.03  # Shorter content for late night
                
                # Apply boost
                rec['score'] += contextual_boost
            
            # Re-sort after contextual adjustments
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error applying contextual filtering: {e}")
            return recommendations
    
    def _apply_diversity_and_serendipity(self, recommendations: List[Dict], 
                                       user_profile: Dict) -> List[Dict]:
        """Apply diversity and serendipity to recommendations"""
        try:
            diversity_preference = user_profile.get('exploration_tendency', 0.5)
            
            if diversity_preference < 0.3:
                return recommendations  # User prefers familiar content
            
            # Ensure genre diversity
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
                    
                    # Check for diversity
                    is_diverse = (
                        primary_genre not in seen_genres or
                        content_type not in seen_content_types or
                        primary_language not in seen_languages
                    )
                    
                    if is_diverse and len(diverse_recommendations) < len(recommendations) * 0.7:
                        seen_genres.add(primary_genre)
                        seen_content_types.add(content_type)
                        seen_languages.add(primary_language)
                        
                        # Apply diversity boost
                        rec['score'] += 0.05
                        diverse_recommendations.append(rec)
                    else:
                        other_recommendations.append(rec)
                
                except (json.JSONDecodeError, TypeError, IndexError):
                    other_recommendations.append(rec)
            
            # Interleave diverse and other recommendations
            final_recommendations = []
            
            # Add diverse recommendations first (weighted by diversity preference)
            diversity_count = int(len(recommendations) * diversity_preference)
            final_recommendations.extend(diverse_recommendations[:diversity_count])
            
            # Fill remaining slots
            remaining_slots = len(recommendations) - len(final_recommendations)
            
            # Add remaining diverse recommendations
            remaining_diverse = diverse_recommendations[diversity_count:]
            final_recommendations.extend(remaining_diverse[:remaining_slots//2])
            
            # Add other recommendations
            remaining_slots = len(recommendations) - len(final_recommendations)
            final_recommendations.extend(other_recommendations[:remaining_slots])
            
            # Re-sort by score
            final_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error applying diversity and serendipity: {e}")
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
                
                # Prepare media URLs
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
                reason = self._generate_advanced_recommendation_reason(rec, user_profile)
                
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
                    'personalization_tags': self._generate_personalization_tags(content, user_profile)
                }
                
                formatted_recs.append(formatted_rec)
            
            return formatted_recs
            
        except Exception as e:
            logger.error(f"Error formatting recommendations: {e}")
            return []
    
    def _generate_advanced_recommendation_reason(self, recommendation: Dict, user_profile: Dict) -> str:
        """Generate detailed recommendation reason"""
        try:
            content = recommendation['content']
            methods = recommendation.get('methods_used', [])
            score = recommendation.get('score', 0)
            
            reasons = []
            
            # Method-based reasons
            if 'neural_collaborative' in methods:
                reasons.append("users with similar tastes highly recommend this")
            
            if 'sequence_aware' in methods:
                reasons.append("perfectly follows your viewing pattern")
            
            if 'content_embeddings' in methods:
                reasons.append("matches your content preferences precisely")
            
            # Content-specific reasons
            try:
                content_genres = json.loads(content.genres or '[]')
                user_genres = user_profile.get('genre_preferences', {})
                
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
            
            # Quality reasons
            if content.rating and content.rating >= 8.5:
                reasons.append("exceptional quality (highly rated)")
            elif content.rating and content.rating >= 8.0:
                reasons.append("excellent quality")
            
            # Trending reasons
            if content.is_trending:
                reasons.append("trending now")
            
            # New release reasons
            if content.is_new_release:
                reasons.append("fresh release")
            
            # Language match
            try:
                content_languages = json.loads(content.languages or '[]')
                user_languages = user_profile.get('language_preferences', {})
                
                for lang in content_languages:
                    if user_languages.get(lang.lower(), 0) > 0.4:
                        reasons.append(f"in your preferred {lang} language")
                        break
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Confidence-based reasoning
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
                
        except Exception as e:
            logger.error(f"Error generating recommendation reason: {e}")
            return "Recommended for you"
    
    def _calculate_recommendation_confidence(self, recommendation: Dict, user_profile: Dict) -> str:
        """Calculate confidence level for recommendation"""
        try:
            score = recommendation.get('score', 0)
            methods_used = recommendation.get('methods_used', [])
            user_confidence = user_profile.get('confidence_score', 0)
            
            # Base confidence from score
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
            
            # Adjust based on method consensus
            if len(methods_used) >= 3:
                if base_confidence == 'medium':
                    base_confidence = 'high'
                elif base_confidence == 'low':
                    base_confidence = 'medium'
            
            # Adjust based on user profile confidence
            if user_confidence < 0.3 and base_confidence in ['very_high', 'high']:
                base_confidence = 'medium'  # Lower confidence for new users
            
            return base_confidence
            
        except Exception as e:
            logger.error(f"Error calculating recommendation confidence: {e}")
            return 'medium'
    
    def _generate_personalization_tags(self, content: Content, user_profile: Dict) -> List[str]:
        """Generate personalization tags for recommendation"""
        try:
            tags = []
            
            # Genre tags
            try:
                content_genres = json.loads(content.genres or '[]')
                user_genres = user_profile.get('genre_preferences', {})
                
                for genre in content_genres:
                    if user_genres.get(genre.lower(), 0) > 0.4:
                        tags.append(f"favorite_{genre.lower()}")
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Language tags
            try:
                content_languages = json.loads(content.languages or '[]')
                user_languages = user_profile.get('language_preferences', {})
                
                for lang in content_languages:
                    if user_languages.get(lang.lower(), 0) > 0.3:
                        tags.append(f"preferred_{lang.lower()}")
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Quality tags
            quality_prefs = user_profile.get('quality_preferences', {})
            user_avg_rating = quality_prefs.get('average_rating', 7.0)
            
            if content.rating:
                if content.rating >= user_avg_rating + 0.5:
                    tags.append('higher_quality_than_usual')
                elif content.rating >= user_avg_rating - 0.5:
                    tags.append('your_quality_range')
            
            # Runtime tags
            runtime_prefs = user_profile.get('runtime_preferences', {})
            if content.runtime and runtime_prefs.get('preferred_range'):
                min_runtime, max_runtime = runtime_prefs['preferred_range']
                if min_runtime <= content.runtime <= max_runtime:
                    tags.append('ideal_length')
            
            # Trending tags
            trending_interests = user_profile.get('trending_interests', [])
            try:
                content_genres = json.loads(content.genres or '[]')
                for genre in content_genres:
                    if genre.lower() in trending_interests:
                        tags.append('trending_interest')
                        break
            except (json.JSONDecodeError, TypeError):
                pass
            
            # Exploration tags
            exploration_tendency = user_profile.get('exploration_tendency', 0.5)
            if exploration_tendency > 0.7:
                tags.append('adventurous_choice')
            elif exploration_tendency < 0.3:
                tags.append('safe_choice')
            
            return tags[:5]  # Limit to 5 tags
            
        except Exception as e:
            logger.error(f"Error generating personalization tags: {e}")
            return []
    
    def _estimate_accuracy(self, user_profile: Dict, algorithm_performance: Dict) -> float:
        """Estimate recommendation accuracy percentage"""
        try:
            confidence = user_profile.get('confidence_score', 0)
            interaction_count = user_profile.get('interaction_count', 0)
            
            # Base accuracy from user profile strength
            base_accuracy = min(confidence * 80, 80)  # Max 80% from profile
            
            # Bonus for interaction count
            interaction_bonus = min(interaction_count / 100.0 * 10, 10)  # Max 10% bonus
            
            # Bonus for algorithm diversity
            active_algorithms = sum(1 for count in algorithm_performance.values() if count > 0)
            algorithm_bonus = min(active_algorithms * 2, 8)  # Max 8% bonus
            
            # Bonus for strong algorithm performance
            total_recommendations = sum(algorithm_performance.values())
            performance_bonus = min(total_recommendations / 50.0 * 2, 2)  # Max 2% bonus
            
            estimated_accuracy = base_accuracy + interaction_bonus + algorithm_bonus + performance_bonus
            
            return min(round(estimated_accuracy, 1), 95.0)  # Cap at 95%
            
        except Exception as e:
            logger.error(f"Error estimating accuracy: {e}")
            return 75.0  # Default estimate

# Initialize the hyper-personalized engine
hyper_engine = HyperPersonalizedEngine()

# API Routes
@personalized_bp.route('/api/recommendations/hyper-personalized', methods=['GET'])
@require_auth
def get_hyper_personalized_recommendations(current_user):
    """Get ultra-personalized recommendations with maximum accuracy"""
    try:
        content_type = request.args.get('type')  # movie, tv, anime
        limit = min(int(request.args.get('limit', 20)), 50)
        
        # Enhanced context information
        context = {
            'device': request.headers.get('User-Agent', ''),
            'time': datetime.utcnow().hour,
            'day': datetime.utcnow().weekday(),
            'user_agent': request.headers.get('User-Agent', ''),
            'request_timestamp': datetime.utcnow().isoformat()
        }
        
        recommendations = hyper_engine.get_hyper_personalized_recommendations(
            current_user.id, content_type, context, limit
        )
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        logger.error(f"Error getting hyper-personalized recommendations: {e}")
        return jsonify({'error': 'Failed to get personalized recommendations'}), 500

@personalized_bp.route('/api/recommendations/hyper-personalized/categories', methods=['GET'])
@require_auth
def get_hyper_personalized_categories(current_user):
    """Get hyper-personalized recommendations grouped by categories"""
    try:
        context = {
            'device': request.headers.get('User-Agent', ''),
            'time': datetime.utcnow().hour,
            'day': datetime.utcnow().weekday()
        }
        
        categories = {
            'for_you': hyper_engine.get_hyper_personalized_recommendations(
                current_user.id, None, context, 15
            ),
            'movies': hyper_engine.get_hyper_personalized_recommendations(
                current_user.id, 'movie', context, 12
            ),
            'tv_shows': hyper_engine.get_hyper_personalized_recommendations(
                current_user.id, 'tv', context, 12
            ),
            'anime': hyper_engine.get_hyper_personalized_recommendations(
                current_user.id, 'anime', context, 12
            )
        }
        
        return jsonify({
            'categories': categories,
            'metadata': {
                'hyper_personalized': True,
                'user_id': current_user.id,
                'context_applied': True,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting hyper-personalized categories: {e}")
        return jsonify({'error': 'Failed to get personalized categories'}), 500

@personalized_bp.route('/api/user/profile/advanced', methods=['GET'])
@require_auth
def get_advanced_user_profile(current_user):
    """Get ultra-detailed user profile with advanced insights"""
    try:
        profile = hyper_engine.profiler.build_comprehensive_profile(current_user.id)
        
        return jsonify({
            'advanced_profile': profile,
            'profile_completeness': profile.get('profile_strength', 0),
            'recommendation_readiness': profile.get('confidence_score', 0) >= 0.3
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting advanced user profile: {e}")
        return jsonify({'error': 'Failed to get advanced profile'}), 500

@personalized_bp.route('/api/recommendations/similar-advanced/<int:content_id>', methods=['GET'])
@require_auth
def get_advanced_similar_recommendations(current_user, content_id):
    """Get advanced similar recommendations using content embeddings"""
    try:
        limit = min(int(request.args.get('limit', 10)), 20)
        
        # Build content embeddings if not already built
        if not hyper_engine.content_embeddings.content_embeddings:
            hyper_engine.content_embeddings.build_content_embeddings()
        
        # Get similar content using advanced embeddings
        similar_content = hyper_engine.content_embeddings.get_content_similarities(
            content_id, limit * 2
        )
        
        # Get user profile for personalization
        user_profile = hyper_engine.profiler.build_comprehensive_profile(current_user.id)
        
        # Format recommendations with personalization
        recommendations = []
        for similar_id, similarity in similar_content:
            content = Content.query.get(similar_id)
            if content:
                # Calculate personalized score
                profile_match = hyper_engine._calculate_profile_match_score(content, user_profile)
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
                    'recommendation_reason': f"Similar content with {round(similarity*100, 1)}% match to your preferences"
                })
        
        # Sort by personalized score
        recommendations.sort(key=lambda x: x['personalized_score'], reverse=True)
        
        return jsonify({
            'similar_recommendations': recommendations[:limit],
            'base_content_id': content_id,
            'total_found': len(recommendations),
            'personalization_applied': True
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting advanced similar recommendations: {e}")
        return jsonify({'error': 'Failed to get similar recommendations'}), 500

@personalized_bp.route('/api/recommendations/retrain-models', methods=['POST'])
@require_auth
def retrain_advanced_models(current_user):
    """Retrain all advanced recommendation models"""
    try:
        if not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        results = {}
        
        # Retrain Neural Collaborative Filtering
        try:
            user_ids, item_ids, ratings = hyper_engine.ncf_engine.prepare_data()
            if len(user_ids) > 0:
                success = hyper_engine.ncf_engine.train_embeddings(user_ids, item_ids, ratings)
                results['ncf_training'] = 'success' if success else 'failed'
            else:
                results['ncf_training'] = 'no_data'
        except Exception as e:
            results['ncf_training'] = f'error: {str(e)}'
        
        # Rebuild Content Embeddings
        try:
            success = hyper_engine.content_embeddings.build_content_embeddings()
            results['content_embeddings'] = 'success' if success else 'failed'
        except Exception as e:
            results['content_embeddings'] = f'error: {str(e)}'
        
        # Clear all user profile caches
        try:
            if cache:
                # Clear user profile caches
                users = User.query.all()
                cleared_count = 0
                for user in users:
                    cache.delete(f"advanced_profile:{user.id}")
                    cache.delete(f"user_profile:{user.id}")
                    cleared_count += 1
                results['cache_clearing'] = f'cleared {cleared_count} user profiles'
        except Exception as e:
            results['cache_clearing'] = f'error: {str(e)}'
        
        return jsonify({
            'message': 'Advanced model retraining completed',
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error retraining advanced models: {e}")
        return jsonify({'error': 'Failed to retrain models'}), 500

@personalized_bp.route('/api/recommendations/feedback-advanced', methods=['POST'])
@require_auth
def record_advanced_feedback(current_user):
    """Record advanced feedback for recommendation learning"""
    try:
        data = request.get_json()
        
        required_fields = ['content_id', 'feedback_type', 'recommendation_score']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Enhanced feedback metadata
        feedback_metadata = {
            'recommendation_score': data['recommendation_score'],
            'recommendation_methods': data.get('recommendation_methods', []),
            'user_rating': data.get('user_rating'),
            'feedback_type': data['feedback_type'],
            'context': {
                'time': datetime.utcnow().hour,
                'day': datetime.utcnow().weekday(),
                'device': request.headers.get('User-Agent', '')
            },
            'user_comment': data.get('comment', ''),
            'interaction_timestamp': datetime.utcnow().isoformat()
        }
        
        # Record feedback as interaction
        feedback_interaction = UserInteraction(
            user_id=current_user.id,
            content_id=data['content_id'],
            interaction_type=f"advanced_feedback_{data['feedback_type']}",
            rating=data.get('user_rating'),
            interaction_metadata=json.dumps(feedback_metadata)
        )
        
        db.session.add(feedback_interaction)
        db.session.commit()
        
        # Clear user profile cache to incorporate new feedback
        if cache:
            cache.delete(f"advanced_profile:{current_user.id}")
            cache.delete(f"user_profile:{current_user.id}")
        
        return jsonify({
            'message': 'Advanced feedback recorded successfully',
            'feedback_id': feedback_interaction.id
        }), 201
        
    except Exception as e:
        logger.error(f"Error recording advanced feedback: {e}")
        db.session.rollback()
        return jsonify({'error': 'Failed to record feedback'}), 500

@personalized_bp.route('/api/recommendations/accuracy-report', methods=['GET'])
@require_auth
def get_accuracy_report(current_user):
    """Get detailed accuracy report for user's recommendations"""
    try:
        # Get user profile
        user_profile = hyper_engine.profiler.build_comprehensive_profile(current_user.id)
        
        # Analyze feedback history
        feedback_interactions = UserInteraction.query.filter(
            UserInteraction.user_id == current_user.id,
            UserInteraction.interaction_type.like('advanced_feedback_%')
        ).all()
        
        accuracy_metrics = {
            'total_feedback': len(feedback_interactions),
            'positive_feedback': 0,
            'negative_feedback': 0,
            'accuracy_by_method': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'accuracy_by_content_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'overall_accuracy': 0.0,
            'improvement_trend': [],
            'recommendation_quality': user_profile.get('confidence_score', 0) * 100
        }
        
        for feedback in feedback_interactions:
            try:
                metadata = json.loads(feedback.interaction_metadata or '{}')
                feedback_type = metadata.get('feedback_type', 'unknown')
                methods = metadata.get('recommendation_methods', ['unknown'])
                
                # Count positive/negative feedback
                if feedback_type in ['like', 'love', 'perfect']:
                    accuracy_metrics['positive_feedback'] += 1
                    is_correct = True
                elif feedback_type in ['dislike', 'hate', 'terrible']:
                    accuracy_metrics['negative_feedback'] += 1
                    is_correct = False
                else:
                    continue  # Skip neutral feedback
                
                # Accuracy by method
                for method in methods:
                    accuracy_metrics['accuracy_by_method'][method]['total'] += 1
                    if is_correct:
                        accuracy_metrics['accuracy_by_method'][method]['correct'] += 1
                
                # Accuracy by content type
                content = Content.query.get(feedback.content_id)
                if content:
                    content_type = content.content_type
                    accuracy_metrics['accuracy_by_content_type'][content_type]['total'] += 1
                    if is_correct:
                        accuracy_metrics['accuracy_by_content_type'][content_type]['correct'] += 1
                
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Calculate overall accuracy
        if accuracy_metrics['total_feedback'] > 0:
            accuracy_metrics['overall_accuracy'] = round(
                (accuracy_metrics['positive_feedback'] / accuracy_metrics['total_feedback']) * 100, 1
            )
        
        # Calculate method accuracies
        method_accuracies = {}
        for method, stats in accuracy_metrics['accuracy_by_method'].items():
            if stats['total'] > 0:
                method_accuracies[method] = round((stats['correct'] / stats['total']) * 100, 1)
        
        # Calculate content type accuracies
        content_type_accuracies = {}
        for content_type, stats in accuracy_metrics['accuracy_by_content_type'].items():
            if stats['total'] > 0:
                content_type_accuracies[content_type] = round((stats['correct'] / stats['total']) * 100, 1)
        
        return jsonify({
            'accuracy_report': {
                'overall_accuracy': accuracy_metrics['overall_accuracy'],
                'total_feedback': accuracy_metrics['total_feedback'],
                'positive_feedback': accuracy_metrics['positive_feedback'],
                'negative_feedback': accuracy_metrics['negative_feedback'],
                'recommendation_quality_score': accuracy_metrics['recommendation_quality'],
                'accuracy_by_method': method_accuracies,
                'accuracy_by_content_type': content_type_accuracies,
                'profile_strength': user_profile.get('profile_strength', 0),
                'learning_status': 'excellent' if accuracy_metrics['overall_accuracy'] > 80 else
                                 'good' if accuracy_metrics['overall_accuracy'] > 60 else
                                 'improving' if accuracy_metrics['overall_accuracy'] > 40 else
                                 'learning'
            },
            'recommendations': {
                'provide_more_feedback': accuracy_metrics['total_feedback'] < 10,
                'try_different_genres': user_profile.get('exploration_tendency', 0.5) < 0.3,
                'system_confidence': user_profile.get('confidence_score', 0)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting accuracy report: {e}")
        return jsonify({'error': 'Failed to get accuracy report'}), 500