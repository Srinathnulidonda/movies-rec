# ml-services/app.py
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text, func, and_, or_

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization

import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import redis
import networkx as nx
import joblib
from textblob import TextBlob
import nltk
from numba import jit, cuda
import psutil

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'ml-service-secret-key')

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://movies_rec_panf_user:BO5X3d2QihK7GG9hxgtBiCtni8NTbbIi@dpg-d2q7gamr433s73e0hcm0-a/movies_rec_panf')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app, origins=['*'])
db = SQLAlchemy(app)

REDIS_URL = os.environ.get('REDIS_URL', 'redis://red-d2qlbuje5dus73c71qog:xp7inVzgblGCbo9I4taSGLdKUg0xY91I@red-d2qlbuje5dus73c71qog:6379')
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False
    redis_client = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_UPDATE_INTERVAL = 3600
SIMILARITY_THRESHOLD = 0.1
DIVERSITY_WEIGHT = 0.3
NOVELTY_WEIGHT = 0.2
POPULARITY_WEIGHT = 0.1
ACCURACY_WEIGHT = 0.4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class Content(db.Model):
    __tablename__ = 'content'
    
    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(150), unique=True, nullable=False, index=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    imdb_id = db.Column(db.String(20))
    mal_id = db.Column(db.Integer)
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    content_type = db.Column(db.String(20), nullable=False)
    genres = db.Column(db.Text)
    anime_genres = db.Column(db.Text)
    languages = db.Column(db.Text)
    release_date = db.Column(db.Date)
    runtime = db.Column(db.Integer)
    rating = db.Column(db.Float)
    vote_count = db.Column(db.Integer)
    popularity = db.Column(db.Float)
    overview = db.Column(db.Text)
    poster_path = db.Column(db.String(255))
    backdrop_path = db.Column(db.String(255))
    trailer_url = db.Column(db.String(255))
    youtube_trailer_id = db.Column(db.String(255))
    is_trending = db.Column(db.Boolean, default=False)
    is_new_release = db.Column(db.Boolean, default=False)
    is_critics_choice = db.Column(db.Boolean, default=False)
    critics_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    preferred_languages = db.Column(db.Text)
    preferred_genres = db.Column(db.Text)
    location = db.Column(db.String(100))
    avatar_url = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)
    rating = db.Column(db.Float)
    interaction_metadata = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class AnonymousInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    interaction_type = db.Column(db.String(20), nullable=False)
    ip_address = db.Column(db.String(45))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class CacheManager:
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.default_timeout = 1800
    
    def get(self, key):
        if not self.redis_client:
            return None
        try:
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except:
            return None
    
    def set(self, key, value, timeout=None):
        if not self.redis_client:
            return False
        try:
            timeout = timeout or self.default_timeout
            return self.redis_client.setex(key, timeout, json.dumps(value))
        except:
            return False
    
    def delete(self, key):
        if not self.redis_client:
            return False
        try:
            return self.redis_client.delete(key)
        except:
            return False

cache_manager = CacheManager(redis_client)

class AdvancedUserBehaviorAnalyzer:
    def __init__(self):
        self.interaction_weights = {
            'view': 1.0,
            'search': 0.8,
            'like': 2.5,
            'favorite': 4.0,
            'watchlist': 3.5,
            'rating': 0.0,
            'similar_view': 0.6
        }
        
        self.time_decay_factor = 0.95
        self.genre_importance = 0.35
        self.language_importance = 0.3
        self.content_type_importance = 0.25
        self.quality_importance = 0.1
        
    def extract_comprehensive_user_features(self, user_data: Dict) -> Dict:
        try:
            user_id = user_data.get('user_id')
            interactions = user_data.get('interactions', [])
            preferred_languages = user_data.get('preferred_languages', [])
            preferred_genres = user_data.get('preferred_genres', [])
            
            if not interactions:
                return self._get_cold_start_features(user_id, preferred_languages, preferred_genres)
            
            content_ids = [interaction['content_id'] for interaction in interactions]
            content_data = self._fetch_content_data(content_ids)
            
            now = datetime.utcnow()
            interaction_scores = defaultdict(float)
            genre_affinity = defaultdict(float)
            language_affinity = defaultdict(float)
            content_type_scores = defaultdict(float)
            rating_scores = defaultdict(list)
            temporal_patterns = defaultdict(list)
            quality_preferences = []
            
            for interaction in interactions:
                content_id = interaction['content_id']
                interaction_type = interaction['interaction_type']
                timestamp = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
                rating = interaction.get('rating')
                
                days_ago = (now - timestamp).days
                time_weight = self.time_decay_factor ** days_ago
                
                base_score = self.interaction_weights.get(interaction_type, 0.5)
                weighted_score = base_score * time_weight
                
                interaction_scores[content_id] += weighted_score
                
                if rating and rating > 0:
                    rating_weight = (rating / 5.0) * 2.0
                    interaction_scores[content_id] += rating_weight * time_weight
                    rating_scores[content_id].append(rating)
                
                temporal_patterns[interaction_type].append(timestamp)
                
                content_info = content_data.get(content_id)
                if content_info:
                    try:
                        genres = json.loads(content_info.get('genres', '[]'))
                        for genre in genres:
                            genre_affinity[genre.lower()] += weighted_score
                    except:
                        pass
                    
                    try:
                        languages = json.loads(content_info.get('languages', '[]'))
                        for lang in languages:
                            language_affinity[lang.lower()] += weighted_score
                    except:
                        pass
                    
                    content_type = content_info.get('content_type', 'movie')
                    content_type_scores[content_type] += weighted_score
                    
                    content_rating = content_info.get('rating', 0)
                    if content_rating > 0:
                        quality_preferences.append(content_rating)
            
            features = {
                'user_id': user_id,
                'total_interactions': len(interactions),
                'interaction_scores': dict(interaction_scores),
                'genre_affinity': dict(genre_affinity),
                'language_affinity': dict(language_affinity),
                'content_type_preference': self._normalize_scores(content_type_scores),
                'avg_rating': self._calculate_avg_rating(rating_scores),
                'rating_variance': self._calculate_rating_variance(rating_scores),
                'quality_preference': np.mean(quality_preferences) if quality_preferences else 7.0,
                'interaction_diversity': self._calculate_interaction_diversity(interactions),
                'temporal_consistency': self._calculate_temporal_consistency(temporal_patterns),
                'preferred_languages': preferred_languages,
                'preferred_genres': preferred_genres,
                'recency_bias': self._calculate_recency_bias(interactions),
                'exploration_tendency': self._calculate_exploration_tendency(interactions),
                'rating_strictness': self._calculate_rating_strictness(rating_scores),
                'activity_level': self._calculate_activity_level(interactions),
                'content_discovery_pattern': self._calculate_discovery_pattern(interactions),
                'genre_diversity': len(set(genre_affinity.keys())),
                'language_diversity': len(set(language_affinity.keys())),
                'content_type_diversity': len(set(content_type_scores.keys())),
                'seasonal_patterns': self._calculate_seasonal_patterns(temporal_patterns),
                'preferred_content_qualities': self._analyze_quality_preferences(quality_preferences),
                'interaction_intensity': self._calculate_interaction_intensity(interactions)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting user features: {e}")
            return self._get_default_features(user_id)
    
    def _fetch_content_data(self, content_ids: List[int]) -> Dict:
        try:
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            return {
                content.id: {
                    'genres': content.genres,
                    'languages': content.languages,
                    'content_type': content.content_type,
                    'rating': content.rating,
                    'popularity': content.popularity,
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'runtime': content.runtime,
                    'overview': content.overview
                }
                for content in contents
            }
        except Exception as e:
            logger.error(f"Error fetching content data: {e}")
            return {}
    
    def _normalize_scores(self, scores_dict: defaultdict) -> Dict:
        total = sum(scores_dict.values())
        if total == 0:
            return {'movie': 0.4, 'tv': 0.4, 'anime': 0.2}
        return {k: v/total for k, v in scores_dict.items()}
    
    def _get_cold_start_features(self, user_id: int, preferred_languages: List, preferred_genres: List) -> Dict:
        return {
            'user_id': user_id,
            'total_interactions': 0,
            'interaction_scores': {},
            'genre_affinity': {genre.lower(): 1.0 for genre in preferred_genres},
            'language_affinity': {lang.lower(): 1.0 for lang in preferred_languages},
            'content_type_preference': {'movie': 0.4, 'tv': 0.4, 'anime': 0.2},
            'avg_rating': 4.0,
            'rating_variance': 0.5,
            'quality_preference': 7.0,
            'interaction_diversity': 0.0,
            'temporal_consistency': 0.0,
            'preferred_languages': preferred_languages,
            'preferred_genres': preferred_genres,
            'recency_bias': 0.8,
            'exploration_tendency': 0.6,
            'rating_strictness': 0.5,
            'activity_level': 'new',
            'content_discovery_pattern': 'preference_based',
            'genre_diversity': len(preferred_genres),
            'language_diversity': len(preferred_languages),
            'content_type_diversity': 3,
            'seasonal_patterns': {},
            'preferred_content_qualities': 'medium_high',
            'interaction_intensity': 'low'
        }
    
    def _calculate_avg_rating(self, rating_scores: Dict) -> float:
        all_ratings = []
        for ratings in rating_scores.values():
            all_ratings.extend(ratings)
        return np.mean(all_ratings) if all_ratings else 4.0
    
    def _calculate_rating_variance(self, rating_scores: Dict) -> float:
        all_ratings = []
        for ratings in rating_scores.values():
            all_ratings.extend(ratings)
        return np.var(all_ratings) if len(all_ratings) > 1 else 0.5
    
    def _calculate_interaction_diversity(self, interactions: List) -> float:
        interaction_types = [i['interaction_type'] for i in interactions]
        unique_types = set(interaction_types)
        return len(unique_types) / len(self.interaction_weights)
    
    def _calculate_temporal_consistency(self, temporal_patterns: Dict) -> float:
        if not temporal_patterns:
            return 0.0
        
        consistency_scores = []
        for interaction_type, timestamps in temporal_patterns.items():
            if len(timestamps) > 1:
                intervals = []
                sorted_timestamps = sorted(timestamps)
                for i in range(1, len(sorted_timestamps)):
                    interval = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds()
                    intervals.append(interval)
                
                if intervals:
                    consistency = 1.0 / (1.0 + np.std(intervals) / np.mean(intervals))
                    consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_recency_bias(self, interactions: List) -> float:
        if not interactions:
            return 0.8
        
        now = datetime.utcnow()
        recent_interactions = 0
        for interaction in interactions:
            timestamp = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
            days_ago = (now - timestamp).days
            if days_ago <= 30:
                recent_interactions += 1
        
        return recent_interactions / len(interactions)
    
    def _calculate_exploration_tendency(self, interactions: List) -> float:
        if len(interactions) < 5:
            return 0.6
        
        unique_content = set(i['content_id'] for i in interactions)
        return len(unique_content) / len(interactions)
    
    def _calculate_rating_strictness(self, rating_scores: Dict) -> float:
        all_ratings = []
        for ratings in rating_scores.values():
            all_ratings.extend(ratings)
        
        if not all_ratings:
            return 0.5
        
        avg_rating = np.mean(all_ratings)
        return 1.0 - (avg_rating - 1.0) / 4.0
    
    def _calculate_activity_level(self, interactions: List) -> str:
        interaction_count = len(interactions)
        if interaction_count < 5:
            return 'low'
        elif interaction_count < 20:
            return 'medium'
        elif interaction_count < 50:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_discovery_pattern(self, interactions: List) -> str:
        search_count = sum(1 for i in interactions if i['interaction_type'] == 'search')
        total_interactions = len(interactions)
        
        if total_interactions == 0:
            return 'preference_based'
        
        search_ratio = search_count / total_interactions
        if search_ratio > 0.6:
            return 'search_driven'
        elif search_ratio > 0.3:
            return 'mixed'
        else:
            return 'recommendation_driven'
    
    def _calculate_seasonal_patterns(self, temporal_patterns: Dict) -> Dict:
        patterns = {}
        for interaction_type, timestamps in temporal_patterns.items():
            months = [ts.month for ts in timestamps]
            if months:
                month_counts = Counter(months)
                patterns[interaction_type] = dict(month_counts)
        return patterns
    
    def _analyze_quality_preferences(self, quality_scores: List) -> str:
        if not quality_scores:
            return 'medium'
        
        avg_quality = np.mean(quality_scores)
        if avg_quality >= 8.0:
            return 'high'
        elif avg_quality >= 6.5:
            return 'medium_high'
        elif avg_quality >= 5.0:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_interaction_intensity(self, interactions: List) -> str:
        if not interactions:
            return 'low'
        
        now = datetime.utcnow()
        recent_interactions = 0
        for interaction in interactions:
            timestamp = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
            days_ago = (now - timestamp).days
            if days_ago <= 7:
                recent_interactions += 1
        
        weekly_rate = recent_interactions
        if weekly_rate >= 20:
            return 'very_high'
        elif weekly_rate >= 10:
            return 'high'
        elif weekly_rate >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _get_default_features(self, user_id: int) -> Dict:
        return {
            'user_id': user_id,
            'total_interactions': 0,
            'interaction_scores': {},
            'genre_affinity': {},
            'language_affinity': {},
            'content_type_preference': {'movie': 0.4, 'tv': 0.4, 'anime': 0.2},
            'avg_rating': 4.0,
            'rating_variance': 0.5,
            'quality_preference': 7.0,
            'interaction_diversity': 0.0,
            'temporal_consistency': 0.0,
            'preferred_languages': [],
            'preferred_genres': [],
            'recency_bias': 0.8,
            'exploration_tendency': 0.6,
            'rating_strictness': 0.5,
            'activity_level': 'new',
            'content_discovery_pattern': 'preference_based',
            'genre_diversity': 0,
            'language_diversity': 0,
            'content_type_diversity': 0,
            'seasonal_patterns': {},
            'preferred_content_qualities': 'medium',
            'interaction_intensity': 'low'
        }

class RealTimeContentFetcher:
    def __init__(self):
        self.content_cache = {}
        self.cache_timeout = 1800
        
    def fetch_content_for_recommendations(self, limit: int = 1000, content_types: List[str] = None) -> List[Dict]:
        try:
            cache_key = f"content_fetch:{limit}:{':'.join(content_types or [])}"
            cached_content = cache_manager.get(cache_key)
            if cached_content:
                return cached_content
            
            query = Content.query
            
            if content_types:
                query = query.filter(Content.content_type.in_(content_types))
            
            contents = query.order_by(Content.popularity.desc(), Content.rating.desc()).limit(limit).all()
            
            content_data = []
            for content in contents:
                try:
                    genres = json.loads(content.genres) if content.genres else []
                except:
                    genres = []
                
                try:
                    languages = json.loads(content.languages) if content.languages else []
                except:
                    languages = []
                
                try:
                    anime_genres = json.loads(content.anime_genres) if content.anime_genres else []
                except:
                    anime_genres = []
                
                content_item = {
                    'id': content.id,
                    'title': content.title,
                    'original_title': content.original_title,
                    'content_type': content.content_type,
                    'genres': genres,
                    'anime_genres': anime_genres,
                    'languages': languages,
                    'rating': float(content.rating or 0),
                    'vote_count': int(content.vote_count or 0),
                    'popularity': float(content.popularity or 0),
                    'runtime': int(content.runtime or 0),
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'overview': content.overview or '',
                    'is_trending': bool(content.is_trending),
                    'is_new_release': bool(content.is_new_release),
                    'is_critics_choice': bool(content.is_critics_choice),
                    'critics_score': float(content.critics_score or 0),
                    'poster_path': content.poster_path,
                    'youtube_trailer_id': content.youtube_trailer_id
                }
                content_data.append(content_item)
            
            cache_manager.set(cache_key, content_data, self.cache_timeout)
            return content_data
            
        except Exception as e:
            logger.error(f"Error fetching content: {e}")
            return []
    
    def fetch_content_by_ids(self, content_ids: List[int]) -> Dict[int, Dict]:
        try:
            contents = Content.query.filter(Content.id.in_(content_ids)).all()
            content_dict = {}
            
            for content in contents:
                try:
                    genres = json.loads(content.genres) if content.genres else []
                except:
                    genres = []
                
                try:
                    languages = json.loads(content.languages) if content.languages else []
                except:
                    languages = []
                
                content_dict[content.id] = {
                    'id': content.id,
                    'title': content.title,
                    'content_type': content.content_type,
                    'genres': genres,
                    'languages': languages,
                    'rating': float(content.rating or 0),
                    'popularity': float(content.popularity or 0),
                    'release_date': content.release_date.isoformat() if content.release_date else None,
                    'overview': content.overview or '',
                    'poster_path': content.poster_path,
                    'youtube_trailer_id': content.youtube_trailer_id
                }
            
            return content_dict
            
        except Exception as e:
            logger.error(f"Error fetching content by IDs: {e}")
            return {}

class AdvancedContentAnalyzer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.scaler = StandardScaler()
        self.content_features_cache = {}
        
    def extract_comprehensive_content_features(self, content_data: List[Dict]) -> pd.DataFrame:
        try:
            features = []
            
            for content in content_data:
                content_id = content.get('id')
                
                feature_dict = {
                    'content_id': content_id,
                    'title': content.get('title', ''),
                    'content_type': content.get('content_type', 'movie'),
                    'rating': float(content.get('rating', 0.0)),
                    'vote_count': int(content.get('vote_count', 0)),
                    'popularity': float(content.get('popularity', 0.0)),
                    'runtime': int(content.get('runtime', 0)),
                    'release_year': self._extract_release_year(content.get('release_date')),
                    'critics_score': float(content.get('critics_score', 0.0)),
                }
                
                genres = content.get('genres', [])
                genre_features = self._encode_genres(genres)
                feature_dict.update(genre_features)
                
                languages = content.get('languages', [])
                language_features = self._encode_languages(languages)
                feature_dict.update(language_features)
                
                anime_genres = content.get('anime_genres', [])
                anime_genre_features = self._encode_anime_genres(anime_genres)
                feature_dict.update(anime_genre_features)
                
                overview = content.get('overview', '')
                text_features = self._extract_text_features(overview)
                feature_dict.update(text_features)
                
                feature_dict.update({
                    'is_trending': int(content.get('is_trending', False)),
                    'is_new_release': int(content.get('is_new_release', False)),
                    'is_critics_choice': int(content.get('is_critics_choice', False)),
                    'age_score': self._calculate_age_score(feature_dict['release_year']),
                    'popularity_score': self._normalize_popularity(feature_dict['popularity']),
                    'quality_score': self._calculate_quality_score(
                        feature_dict['rating'], 
                        feature_dict['vote_count']
                    ),
                    'runtime_score': self._normalize_runtime(feature_dict['runtime']),
                    'recency_score': self._calculate_recency_score(feature_dict['release_year']),
                    'genre_count': len(genres),
                    'language_count': len(languages),
                    'has_trailer': 1 if content.get('youtube_trailer_id') else 0
                })
                
                features.append(feature_dict)
            
            return pd.DataFrame(features)
            
        except Exception as e:
            logger.error(f"Error extracting content features: {e}")
            return pd.DataFrame()
    
    def _extract_release_year(self, release_date: str) -> int:
        if not release_date:
            return datetime.now().year
        try:
            return int(release_date[:4])
        except:
            return datetime.now().year
    
    def _encode_genres(self, genres: List[str]) -> Dict:
        common_genres = [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
            'Romance', 'Science Fiction', 'Thriller', 'War', 'Western', 'Biography'
        ]
        
        genre_dict = {}
        for genre in common_genres:
            genre_key = f'genre_{genre.lower().replace(" ", "_")}'
            genre_dict[genre_key] = 1 if genre in genres else 0
        
        return genre_dict
    
    def _encode_languages(self, languages: List[str]) -> Dict:
        common_languages = ['english', 'telugu', 'hindi', 'tamil', 'malayalam', 'kannada', 'japanese', 'korean', 'spanish', 'french']
        
        language_dict = {}
        for lang in common_languages:
            lang_key = f'lang_{lang}'
            language_dict[lang_key] = 1 if any(lang in l.lower() for l in languages) else 0
        
        return language_dict
    
    def _encode_anime_genres(self, anime_genres: List[str]) -> Dict:
        anime_genre_list = ['Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports']
        
        anime_dict = {}
        for genre in anime_genre_list:
            genre_key = f'anime_{genre.lower().replace(" ", "_").replace("-", "_")}'
            anime_dict[genre_key] = 1 if genre in anime_genres else 0
        
        return anime_dict
    
    def _extract_text_features(self, text: str) -> Dict:
        if not text:
            return {
                'text_length': 0,
                'word_count': 0,
                'sentiment_score': 0.0,
                'readability_score': 0.0
            }
        
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
        except:
            sentiment = 0.0
        
        words = text.split()
        readability = len([w for w in words if len(w) > 6]) / len(words) if words else 0
        
        return {
            'text_length': len(text),
            'word_count': len(words),
            'sentiment_score': sentiment,
            'readability_score': readability
        }
    
    def _calculate_age_score(self, release_year: int) -> float:
        current_year = datetime.now().year
        age = current_year - release_year
        return max(0.0, 1.0 - (age / 50.0))
    
    def _normalize_popularity(self, popularity: float) -> float:
        return min(1.0, popularity / 1000.0)
    
    def _calculate_quality_score(self, rating: float, vote_count: int) -> float:
        if vote_count == 0:
            return rating / 10.0
        
        confidence = min(1.0, vote_count / 1000.0)
        return (rating / 10.0) * (0.5 + 0.5 * confidence)
    
    def _normalize_runtime(self, runtime: int) -> float:
        if runtime <= 0:
            return 0.5
        
        if runtime <= 60:
            return 0.3
        elif runtime <= 120:
            return 1.0
        elif runtime <= 180:
            return 0.8
        else:
            return 0.6
    
    def _calculate_recency_score(self, release_year: int) -> float:
        current_year = datetime.now().year
        years_old = current_year - release_year
        
        if years_old <= 1:
            return 1.0
        elif years_old <= 3:
            return 0.9
        elif years_old <= 5:
            return 0.7
        elif years_old <= 10:
            return 0.5
        else:
            return 0.3

class HybridPersonalizedRecommendationEngine:
    def __init__(self):
        self.content_based_weight = 0.3
        self.collaborative_weight = 0.25
        self.popularity_weight = 0.2
        self.diversity_weight = 0.15
        self.novelty_weight = 0.1
        
        self.content_fetcher = RealTimeContentFetcher()
        self.content_analyzer = AdvancedContentAnalyzer()
        
    def generate_personalized_recommendations(self, user_features: Dict, num_recommendations: int = 20) -> List[Dict]:
        try:
            user_id = user_features['user_id']
            activity_level = user_features['activity_level']
            
            if activity_level == 'new' or user_features['total_interactions'] < 3:
                return self._cold_start_recommendations(user_features, num_recommendations)
            
            content_data = self.content_fetcher.fetch_content_for_recommendations(limit=2000)
            if not content_data:
                return []
            
            content_df = self.content_analyzer.extract_comprehensive_content_features(content_data)
            if content_df.empty:
                return []
            
            interacted_content_ids = set(user_features['interaction_scores'].keys())
            available_content = content_df[~content_df['content_id'].isin(interacted_content_ids)]
            
            if available_content.empty:
                return []
            
            content_scores = self._calculate_content_based_scores(user_features, available_content)
            collaborative_scores = self._calculate_collaborative_scores(user_features, available_content)
            popularity_scores = self._calculate_popularity_scores(available_content)
            diversity_scores = self._calculate_diversity_scores(user_features, available_content)
            novelty_scores = self._calculate_novelty_scores(user_features, available_content)
            
            final_scores = {}
            for content_id in available_content['content_id']:
                final_score = (
                    content_scores.get(content_id, 0) * self.content_based_weight +
                    collaborative_scores.get(content_id, 0) * self.collaborative_weight +
                    popularity_scores.get(content_id, 0) * self.popularity_weight +
                    diversity_scores.get(content_id, 0) * self.diversity_weight +
                    novelty_scores.get(content_id, 0) * self.novelty_weight
                )
                final_scores[content_id] = final_score
            
            sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            content_dict = {row['content_id']: row for _, row in available_content.iterrows()}
            
            for i, (content_id, score) in enumerate(sorted_recommendations[:num_recommendations]):
                content_info = content_dict.get(content_id)
                if content_info is not None:
                    rec = {
                        'content_id': int(content_id),
                        'score': float(score),
                        'rank': i + 1,
                        'reason': self._generate_recommendation_reason(content_info, user_features),
                        'confidence': self._calculate_confidence(score, user_features),
                        'content_type': content_info['content_type'],
                        'title': content_info['title'],
                        'rating': content_info['rating']
                    }
                    recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating personalized recommendations: {e}")
            return []
    
    def _cold_start_recommendations(self, user_features: Dict, num_recommendations: int) -> List[Dict]:
        try:
            preferred_genres = [g.lower() for g in user_features.get('preferred_genres', [])]
            preferred_languages = [l.lower() for l in user_features.get('preferred_languages', [])]
            
            content_data = self.content_fetcher.fetch_content_for_recommendations(limit=1000)
            if not content_data:
                return []
            
            content_df = self.content_analyzer.extract_comprehensive_content_features(content_data)
            if content_df.empty:
                return []
            
            recommendations = []
            
            for _, content in content_df.iterrows():
                score = 0.0
                
                if preferred_genres:
                    genre_match = 0
                    for genre in preferred_genres:
                        genre_col = f'genre_{genre.replace(" ", "_")}'
                        if genre_col in content and content[genre_col] == 1:
                            genre_match += 1
                    if genre_match > 0:
                        score += (genre_match / len(preferred_genres)) * 0.4
                
                if preferred_languages:
                    lang_match = 0
                    for lang in preferred_languages:
                        lang_col = f'lang_{lang}'
                        if lang_col in content and content[lang_col] == 1:
                            lang_match += 1
                    if lang_match > 0:
                        score += (lang_match / len(preferred_languages)) * 0.3
                
                score += content.get('quality_score', 0) * 0.2
                score += content.get('popularity_score', 0) * 0.1
                
                if score > 0:
                    recommendations.append({
                        'content_id': int(content['content_id']),
                        'score': float(score),
                        'reason': 'Based on your preferences',
                        'confidence': 0.7,
                        'content_type': content['content_type'],
                        'title': content['title'],
                        'rating': content['rating']
                    })
            
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            for i, rec in enumerate(recommendations[:num_recommendations]):
                rec['rank'] = i + 1
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error in cold start recommendations: {e}")
            return []
    
    def _calculate_content_based_scores(self, user_features: Dict, content_df: pd.DataFrame) -> Dict[int, float]:
        try:
            scores = {}
            genre_affinity = user_features.get('genre_affinity', {})
            language_affinity = user_features.get('language_affinity', {})
            content_type_preference = user_features.get('content_type_preference', {})
            quality_preference = user_features.get('quality_preference', 7.0)
            
            for _, content in content_df.iterrows():
                content_id = content['content_id']
                score = 0.0
                
                for genre, affinity in genre_affinity.items():
                    genre_col = f'genre_{genre.replace(" ", "_")}'
                    if genre_col in content and content[genre_col] == 1:
                        score += affinity * 0.4
                
                for lang, affinity in language_affinity.items():
                    lang_col = f'lang_{lang}'
                    if lang_col in content and content[lang_col] == 1:
                        score += affinity * 0.3
                
                content_type = content.get('content_type', 'movie')
                type_preference = content_type_preference.get(content_type, 0.1)
                score += type_preference * 0.2
                
                content_quality = content.get('rating', 0)
                if content_quality > 0:
                    quality_diff = abs(content_quality - quality_preference)
                    quality_score = max(0, 1.0 - (quality_diff / 10.0))
                    score += quality_score * 0.1
                
                scores[content_id] = score
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating content-based scores: {e}")
            return {}
    
    def _calculate_collaborative_scores(self, user_features: Dict, content_df: pd.DataFrame) -> Dict[int, float]:
        try:
            scores = {}
            interaction_scores = user_features.get('interaction_scores', {})
            
            if not interaction_scores:
                return scores
            
            interacted_content_ids = list(interaction_scores.keys())
            interacted_content = self.content_fetcher.fetch_content_by_ids(interacted_content_ids)
            
            for _, content in content_df.iterrows():
                content_id = content['content_id']
                similarity_score = 0.0
                total_weight = 0.0
                
                for inter_id, inter_score in interaction_scores.items():
                    inter_content = interacted_content.get(inter_id)
                    if inter_content:
                        similarity = self._calculate_content_similarity(content, inter_content)
                        similarity_score += similarity * inter_score
                        total_weight += inter_score
                
                if total_weight > 0:
                    scores[content_id] = similarity_score / total_weight
                else:
                    scores[content_id] = 0.0
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating collaborative scores: {e}")
            return {}
    
    def _calculate_content_similarity(self, content1: pd.Series, content2: Dict) -> float:
        try:
            similarity = 0.0
            
            content1_genres = set()
            content2_genres = set(content2.get('genres', []))
            
            for col in content1.index:
                if col.startswith('genre_') and content1[col] == 1:
                    genre_name = col.replace('genre_', '').replace('_', ' ').title()
                    content1_genres.add(genre_name)
            
            if content1_genres and content2_genres:
                genre_intersection = len(content1_genres.intersection(content2_genres))
                genre_union = len(content1_genres.union(content2_genres))
                genre_sim = genre_intersection / genre_union if genre_union > 0 else 0
                similarity += genre_sim * 0.4
            
            content1_languages = set()
            content2_languages = set(content2.get('languages', []))
            
            for col in content1.index:
                if col.startswith('lang_') and content1[col] == 1:
                    lang_name = col.replace('lang_', '')
                    content1_languages.add(lang_name)
            
            if content1_languages and content2_languages:
                lang_intersection = len(content1_languages.intersection(content2_languages))
                lang_union = len(content1_languages.union(content2_languages))
                lang_sim = lang_intersection / lang_union if lang_union > 0 else 0
                similarity += lang_sim * 0.3
            
            type_sim = 1.0 if content1.get('content_type') == content2.get('content_type') else 0.0
            similarity += type_sim * 0.2
            
            rating1 = content1.get('rating', 0)
            rating2 = content2.get('rating', 0)
            if rating1 > 0 and rating2 > 0:
                rating_sim = 1.0 - abs(rating1 - rating2) / 10.0
                similarity += rating_sim * 0.1
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating content similarity: {e}")
            return 0.0
    
    def _calculate_popularity_scores(self, content_df: pd.DataFrame) -> Dict[int, float]:
        try:
            scores = {}
            max_popularity = content_df['popularity_score'].max() if 'popularity_score' in content_df.columns else 1
            
            for _, content in content_df.iterrows():
                content_id = content['content_id']
                popularity = content.get('popularity_score', 0)
                scores[content_id] = popularity / max_popularity if max_popularity > 0 else 0
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating popularity scores: {e}")
            return {}
    
    def _calculate_diversity_scores(self, user_features: Dict, content_df: pd.DataFrame) -> Dict[int, float]:
        try:
            scores = {}
            user_genres = set(user_features.get('genre_affinity', {}).keys())
            user_languages = set(user_features.get('language_affinity', {}).keys())
            user_types = set(user_features.get('content_type_preference', {}).keys())
            
            for _, content in content_df.iterrows():
                content_id = content['content_id']
                diversity_score = 0.0
                
                content_genres = set()
                for col in content.index:
                    if col.startswith('genre_') and content[col] == 1:
                        genre_name = col.replace('genre_', '').replace('_', ' ')
                        content_genres.add(genre_name)
                
                content_languages = set()
                for col in content.index:
                    if col.startswith('lang_') and content[col] == 1:
                        lang_name = col.replace('lang_', '')
                        content_languages.add(lang_name)
                
                content_type = content.get('content_type', 'movie')
                
                if user_genres:
                    genre_diversity = len(content_genres - user_genres) / len(content_genres) if content_genres else 0
                    diversity_score += genre_diversity * 0.5
                
                if user_languages:
                    lang_diversity = len(content_languages - user_languages) / len(content_languages) if content_languages else 0
                    diversity_score += lang_diversity * 0.3
                
                if content_type not in user_types:
                    diversity_score += 0.2
                
                scores[content_id] = diversity_score
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating diversity scores: {e}")
            return {}
    
    def _calculate_novelty_scores(self, user_features: Dict, content_df: pd.DataFrame) -> Dict[int, float]:
        try:
            scores = {}
            
            for _, content in content_df.iterrows():
                content_id = content['content_id']
                novelty_score = 0.0
                
                novelty_score += content.get('recency_score', 0) * 0.4
                novelty_score += content.get('is_new_release', 0) * 0.3
                novelty_score += (1.0 - content.get('popularity_score', 0)) * 0.2
                novelty_score += content.get('is_critics_choice', 0) * 0.1
                
                scores[content_id] = novelty_score
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating novelty scores: {e}")
            return {}
    
    def _generate_recommendation_reason(self, content_info: pd.Series, user_features: Dict) -> str:
        reasons = []
        
        user_genres = user_features.get('genre_affinity', {})
        if user_genres:
            for genre in user_genres.keys():
                genre_col = f'genre_{genre.replace(" ", "_")}'
                if genre_col in content_info and content_info[genre_col] == 1:
                    reasons.append(f"matches your interest in {genre}")
                    break
        
        user_languages = user_features.get('language_affinity', {})
        if user_languages:
            for lang in user_languages.keys():
                lang_col = f'lang_{lang}'
                if lang_col in content_info and content_info[lang_col] == 1:
                    reasons.append(f"in your preferred language")
                    break
        
        if content_info.get('rating', 0) >= 8.0:
            reasons.append("highly rated")
        
        if content_info.get('is_trending', 0):
            reasons.append("currently trending")
        
        if content_info.get('is_critics_choice', 0):
            reasons.append("critics' choice")
        
        if not reasons:
            reasons.append("recommended for you")
        
        return ", ".join(reasons[:2])
    
    def _calculate_confidence(self, score: float, user_features: Dict) -> float:
        base_confidence = min(1.0, score * 2)
        interaction_boost = min(0.3, user_features['total_interactions'] / 50.0)
        return min(1.0, base_confidence + interaction_boost)

class MLPersonalizedRecommendationService:
    def __init__(self):
        self.behavior_analyzer = AdvancedUserBehaviorAnalyzer()
        self.recommendation_engine = HybridPersonalizedRecommendationEngine()
        self.content_fetcher = RealTimeContentFetcher()
        
    def get_personalized_recommendations(self, user_data: Dict, num_recommendations: int = 20) -> Dict:
        try:
            cache_key = f"personalized_v2:{user_data['user_id']}:{num_recommendations}"
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            user_features = self.behavior_analyzer.extract_comprehensive_user_features(user_data)
            
            recommendations = self.recommendation_engine.generate_personalized_recommendations(
                user_features, num_recommendations
            )
            
            metadata = {
                'algorithm': 'advanced_hybrid_personalized',
                'user_activity_level': user_features['activity_level'],
                'total_interactions': user_features['total_interactions'],
                'content_discovery_pattern': user_features['content_discovery_pattern'],
                'genre_diversity': user_features['genre_diversity'],
                'language_diversity': user_features['language_diversity'],
                'recommendation_confidence': np.mean([rec.get('confidence', 0) for rec in recommendations]),
                'personalization_score': self._calculate_personalization_score(user_features),
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': '4.0',
                'features_analyzed': len(user_features),
                'real_time_content': True,
                'database_connected': True
            }
            
            result = {
                'recommendations': recommendations,
                'metadata': metadata
            }
            
            cache_manager.set(cache_key, result, timeout=900)
            return result
            
        except Exception as e:
            logger.error(f"Personalized recommendation error: {e}")
            return {
                'recommendations': [],
                'metadata': {
                    'algorithm': 'error',
                    'error': str(e),
                    'fallback_applied': True
                }
            }
    
    def get_similar_content_recommendations(self, content_id: int, num_recommendations: int = 10) -> Dict:
        try:
            cache_key = f"similar_v2:{content_id}:{num_recommendations}"
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            base_content = Content.query.get(content_id)
            if not base_content:
                return {'recommendations': [], 'error': 'Content not found'}
            
            try:
                base_genres = json.loads(base_content.genres) if base_content.genres else []
            except:
                base_genres = []
            
            try:
                base_languages = json.loads(base_content.languages) if base_content.languages else []
            except:
                base_languages = []
            
            similar_content = Content.query.filter(
                Content.id != content_id,
                Content.content_type == base_content.content_type
            ).limit(500).all()
            
            recommendations = []
            
            for content in similar_content:
                try:
                    content_genres = json.loads(content.genres) if content.genres else []
                except:
                    content_genres = []
                
                try:
                    content_languages = json.loads(content.languages) if content.languages else []
                except:
                    content_languages = []
                
                similarity_score = 0.0
                
                if base_genres and content_genres:
                    genre_intersection = len(set(base_genres).intersection(set(content_genres)))
                    genre_union = len(set(base_genres).union(set(content_genres)))
                    genre_sim = genre_intersection / genre_union if genre_union > 0 else 0
                    similarity_score += genre_sim * 0.5
                
                if base_languages and content_languages:
                    lang_intersection = len(set(base_languages).intersection(set(content_languages)))
                    lang_union = len(set(base_languages).union(set(content_languages)))
                    lang_sim = lang_intersection / lang_union if lang_union > 0 else 0
                    similarity_score += lang_sim * 0.3
                
                if base_content.rating and content.rating:
                    rating_sim = 1.0 - abs(base_content.rating - content.rating) / 10.0
                    similarity_score += rating_sim * 0.2
                
                if similarity_score > 0.3:
                    recommendations.append({
                        'content_id': content.id,
                        'score': float(similarity_score),
                        'reason': 'content_similarity',
                        'title': content.title,
                        'content_type': content.content_type,
                        'rating': content.rating
                    })
            
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            for i, rec in enumerate(recommendations[:num_recommendations]):
                rec['rank'] = i + 1
            
            result = {
                'recommendations': recommendations[:num_recommendations],
                'metadata': {
                    'algorithm': 'advanced_content_similarity',
                    'base_content_id': content_id,
                    'similarity_threshold': 0.3,
                    'total_analyzed': len(similar_content)
                }
            }
            
            cache_manager.set(cache_key, result, timeout=1800)
            return result
            
        except Exception as e:
            logger.error(f"Similar content error: {e}")
            return {'recommendations': [], 'error': str(e)}
    
    def get_trending_recommendations(self, region: str = 'IN', num_recommendations: int = 20) -> Dict:
        try:
            cache_key = f"trending_v2:{region}:{num_recommendations}"
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            trending_content = Content.query.filter(
                or_(
                    Content.is_trending == True,
                    Content.is_new_release == True,
                    Content.popularity > 50.0
                )
            ).order_by(
                Content.popularity.desc(),
                Content.rating.desc()
            ).limit(num_recommendations * 2).all()
            
            recommendations = []
            
            for i, content in enumerate(trending_content[:num_recommendations]):
                trending_score = 0.0
                
                if content.is_trending:
                    trending_score += 0.4
                if content.is_new_release:
                    trending_score += 0.3
                if content.popularity:
                    trending_score += min(0.3, content.popularity / 100.0)
                
                recommendations.append({
                    'content_id': content.id,
                    'score': float(trending_score),
                    'rank': i + 1,
                    'reason': 'trending_algorithm',
                    'title': content.title,
                    'content_type': content.content_type,
                    'rating': content.rating,
                    'is_trending': content.is_trending,
                    'is_new_release': content.is_new_release
                })
            
            result = {
                'recommendations': recommendations,
                'metadata': {
                    'algorithm': 'weighted_trending_score',
                    'region': region,
                    'weights': {
                        'trending_flag': 0.4,
                        'new_release_flag': 0.3,
                        'popularity': 0.3
                    }
                }
            }
            
            cache_manager.set(cache_key, result, timeout=900)
            return result
            
        except Exception as e:
            logger.error(f"Trending recommendations error: {e}")
            return {'recommendations': [], 'error': str(e)}
    
    def _calculate_personalization_score(self, user_features: Dict) -> float:
        score = 0.0
        
        score += min(0.3, user_features['total_interactions'] / 50.0)
        score += user_features['interaction_diversity'] * 0.2
        score += user_features['genre_diversity'] / 20.0 * 0.2
        score += user_features['language_diversity'] / 10.0 * 0.1
        score += (1.0 - user_features['rating_variance']) * 0.2
        
        return min(1.0, score)

ml_service = MLPersonalizedRecommendationService()

@app.route('/health', methods=['GET'])
def health_check():
    try:
        db.session.execute(text('SELECT 1'))
        db_status = 'connected'
    except:
        db_status = 'disconnected'
    
    system_info = {
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total // (1024**3),
        'memory_available': psutil.virtual_memory().available // (1024**3),
    }
    
    content_count = Content.query.count() if db_status == 'connected' else 0
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '4.0',
        'device': str(DEVICE),
        'database_status': db_status,
        'redis_available': REDIS_AVAILABLE,
        'content_count': content_count,
        'system_info': system_info,
        'features': {
            'real_time_personalization': True,
            'advanced_user_behavior_analysis': True,
            'hybrid_recommendation_engine': True,
            'content_similarity_matching': True,
            'diversity_optimization': True,
            'cold_start_handling': True,
            'trending_analysis': True,
            'multi_language_support': True,
            'multi_content_type_support': True
        },
        'libraries': {
            'pytorch': torch.__version__,
            'sklearn': sklearn.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__
        }
    })

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    try:
        user_data = request.json
        
        if not user_data or 'user_id' not in user_data:
            return jsonify({'error': 'Invalid user data provided'}), 400
        
        limit = request.args.get('limit', 20, type=int)
        limit = min(limit, 50)
        
        result = ml_service.get_personalized_recommendations(user_data, limit)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Recommendations API error: {e}")
        return jsonify({
            'error': 'Failed to generate recommendations',
            'recommendations': [],
            'metadata': {'error': str(e)}
        }), 500

@app.route('/api/similar/<int:content_id>', methods=['GET'])
def get_similar(content_id):
    try:
        limit = request.args.get('limit', 10, type=int)
        limit = min(limit, 20)
        
        result = ml_service.get_similar_content_recommendations(content_id, limit)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Similar content API error: {e}")
        return jsonify({
            'error': 'Failed to get similar content',
            'recommendations': []
        }), 500

@app.route('/api/trending', methods=['GET'])
def get_trending():
    try:
        region = request.args.get('region', 'IN')
        limit = request.args.get('limit', 20, type=int)
        limit = min(limit, 50)
        
        result = ml_service.get_trending_recommendations(region, limit)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Trending API error: {e}")
        return jsonify({
            'error': 'Failed to get trending content',
            'recommendations': []
        }), 500

@app.route('/api/analytics/user-profile', methods=['POST'])
def analyze_user_profile():
    try:
        user_data = request.json
        
        if not user_data or 'user_id' not in user_data:
            return jsonify({'error': 'Invalid user data provided'}), 400
        
        behavior_analyzer = AdvancedUserBehaviorAnalyzer()
        user_features = behavior_analyzer.extract_comprehensive_user_features(user_data)
        
        safe_features = user_features.copy()
        safe_features.pop('interaction_scores', None)
        
        return jsonify({
            'user_profile': safe_features,
            'insights': {
                'activity_level': user_features['activity_level'],
                'discovery_pattern': user_features['content_discovery_pattern'],
                'interaction_intensity': user_features['interaction_intensity'],
                'personalization_potential': ml_service._calculate_personalization_score(user_features),
                'preferences': {
                    'genres': dict(list(user_features.get('genre_affinity', {}).items())[:5]),
                    'languages': dict(list(user_features.get('language_affinity', {}).items())[:3]),
                    'content_types': user_features.get('content_type_preference', {})
                },
                'behavior_patterns': {
                    'exploration_tendency': user_features['exploration_tendency'],
                    'recency_bias': user_features['recency_bias'],
                    'quality_preference': user_features['quality_preference'],
                    'rating_behavior': {
                        'average': user_features['avg_rating'],
                        'strictness': user_features['rating_strictness']
                    }
                }
            }
        })
        
    except Exception as e:
        logger.error(f"User profile analysis error: {e}")
        return jsonify({'error': 'Failed to analyze user profile'}), 500

@app.route('/api/content/stats', methods=['GET'])
def content_stats():
    try:
        total_content = Content.query.count()
        movies_count = Content.query.filter_by(content_type='movie').count()
        tv_count = Content.query.filter_by(content_type='tv').count()
        anime_count = Content.query.filter_by(content_type='anime').count()
        trending_count = Content.query.filter_by(is_trending=True).count()
        
        return jsonify({
            'total_content': total_content,
            'breakdown': {
                'movies': movies_count,
                'tv_shows': tv_count,
                'anime': anime_count
            },
            'trending_content': trending_count,
            'database_connected': True,
            'last_updated': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Content stats error: {e}")
        return jsonify({
            'error': 'Failed to get content statistics',
            'database_connected': False
        }), 500

if __name__ == '__main__':
    print("=== Advanced Real-Time ML Recommendation Service ===")
    print(f"PyTorch Device: {DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Redis Available: {REDIS_AVAILABLE}")
    print(f"Database: Connected to PostgreSQL")
    print("=== Real-Time Personalization Active ===")
    
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)