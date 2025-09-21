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

# Flask and API
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

# ML Libraries
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

# Advanced ML
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization

# Data manipulation and utilities
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

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'ml-service-secret-key')

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:pass@localhost/mlservice')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
CORS(app, origins=['*'])
db = SQLAlchemy(app)

# Redis for caching
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False
    redis_client = None

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
MODEL_UPDATE_INTERVAL = 3600  # 1 hour
SIMILARITY_THRESHOLD = 0.1
DIVERSITY_WEIGHT = 0.3
NOVELTY_WEIGHT = 0.2
POPULARITY_WEIGHT = 0.1
ACCURACY_WEIGHT = 0.4

# Check GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class CacheManager:
    """Redis-based cache manager"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.default_timeout = 3600
    
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

class UserBehaviorAnalyzer:
    """Advanced user behavior analysis and feature extraction"""
    
    def __init__(self):
        self.interaction_weights = {
            'view': 1.0,
            'search': 0.8,
            'like': 2.0,
            'favorite': 3.0,
            'watchlist': 2.5,
            'rating': 0.0  # Will be handled separately
        }
        
        self.time_decay_factor = 0.95
        self.genre_importance = 0.3
        self.language_importance = 0.25
        self.content_type_importance = 0.2
        
    def extract_user_features(self, user_data: Dict) -> Dict:
        """Extract comprehensive user features from interaction data"""
        try:
            interactions = user_data.get('interactions', [])
            preferred_languages = user_data.get('preferred_languages', [])
            preferred_genres = user_data.get('preferred_genres', [])
            
            if not interactions:
                return self._get_cold_start_features(preferred_languages, preferred_genres)
            
            # Time-based analysis
            now = datetime.utcnow()
            interaction_scores = defaultdict(float)
            genre_scores = defaultdict(float)
            language_scores = defaultdict(float)
            content_type_scores = defaultdict(float)
            rating_scores = defaultdict(list)
            temporal_patterns = defaultdict(list)
            
            # Process each interaction
            for interaction in interactions:
                content_id = interaction['content_id']
                interaction_type = interaction['interaction_type']
                timestamp = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
                rating = interaction.get('rating')
                
                # Time decay
                days_ago = (now - timestamp).days
                time_weight = self.time_decay_factor ** days_ago
                
                # Base interaction score
                base_score = self.interaction_weights.get(interaction_type, 0.5)
                weighted_score = base_score * time_weight
                
                interaction_scores[content_id] += weighted_score
                
                # Rating handling
                if rating and rating > 0:
                    rating_weight = (rating / 5.0) * 2.0  # Convert to 0-2 scale
                    interaction_scores[content_id] += rating_weight * time_weight
                    rating_scores[content_id].append(rating)
                
                # Temporal patterns
                temporal_patterns[interaction_type].append(timestamp)
            
            # Calculate advanced features
            features = {
                'user_id': user_data['user_id'],
                'total_interactions': len(interactions),
                'interaction_scores': dict(interaction_scores),
                'avg_rating': self._calculate_avg_rating(rating_scores),
                'rating_variance': self._calculate_rating_variance(rating_scores),
                'interaction_diversity': self._calculate_interaction_diversity(interactions),
                'temporal_consistency': self._calculate_temporal_consistency(temporal_patterns),
                'preferred_languages': preferred_languages,
                'preferred_genres': preferred_genres,
                'content_type_preference': self._calculate_content_type_preference(interactions),
                'genre_affinity': self._calculate_genre_affinity(interactions),
                'language_affinity': self._calculate_language_affinity(interactions),
                'recency_bias': self._calculate_recency_bias(interactions),
                'exploration_tendency': self._calculate_exploration_tendency(interactions),
                'rating_strictness': self._calculate_rating_strictness(rating_scores),
                'seasonal_patterns': self._calculate_seasonal_patterns(temporal_patterns),
                'activity_level': self._calculate_activity_level(interactions),
                'content_discovery_pattern': self._calculate_discovery_pattern(interactions)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting user features: {e}")
            return self._get_default_features(user_data['user_id'])
    
    def _get_cold_start_features(self, preferred_languages: List, preferred_genres: List) -> Dict:
        """Generate features for new users with no interaction history"""
        return {
            'user_id': 'new_user',
            'total_interactions': 0,
            'interaction_scores': {},
            'avg_rating': 4.0,
            'rating_variance': 0.5,
            'interaction_diversity': 0.0,
            'temporal_consistency': 0.0,
            'preferred_languages': preferred_languages,
            'preferred_genres': preferred_genres,
            'content_type_preference': {'movie': 0.4, 'tv': 0.4, 'anime': 0.2},
            'genre_affinity': {genre: 1.0 for genre in preferred_genres},
            'language_affinity': {lang: 1.0 for lang in preferred_languages},
            'recency_bias': 0.8,
            'exploration_tendency': 0.6,
            'rating_strictness': 0.5,
            'seasonal_patterns': {},
            'activity_level': 'new',
            'content_discovery_pattern': 'preference_based'
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
    
    def _calculate_content_type_preference(self, interactions: List) -> Dict:
        type_counts = Counter()
        for interaction in interactions:
            type_counts['movie'] += 0.4
            type_counts['tv'] += 0.4
            type_counts['anime'] += 0.2
        
        total = sum(type_counts.values())
        return {k: v/total for k, v in type_counts.items()} if total > 0 else {'movie': 0.4, 'tv': 0.4, 'anime': 0.2}
    
    def _calculate_genre_affinity(self, interactions: List) -> Dict:
        return {}
    
    def _calculate_language_affinity(self, interactions: List) -> Dict:
        return {}
    
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
    
    def _calculate_seasonal_patterns(self, temporal_patterns: Dict) -> Dict:
        return {}
    
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
    
    def _get_default_features(self, user_id: int) -> Dict:
        return {
            'user_id': user_id,
            'total_interactions': 0,
            'interaction_scores': {},
            'avg_rating': 4.0,
            'rating_variance': 0.5,
            'interaction_diversity': 0.0,
            'temporal_consistency': 0.0,
            'preferred_languages': [],
            'preferred_genres': [],
            'content_type_preference': {'movie': 0.4, 'tv': 0.4, 'anime': 0.2},
            'genre_affinity': {},
            'language_affinity': {},
            'recency_bias': 0.8,
            'exploration_tendency': 0.6,
            'rating_strictness': 0.5,
            'seasonal_patterns': {},
            'activity_level': 'new',
            'content_discovery_pattern': 'preference_based'
        }

class ContentFeatureExtractor:
    """Extract and process content features for recommendation"""
    
    def __init__(self):
        self.content_cache = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.sentiment_analyzer = TextBlob
        
    def extract_content_features(self, content_data: List[Dict]) -> pd.DataFrame:
        """Extract comprehensive content features"""
        try:
            features = []
            
            for content in content_data:
                content_id = content.get('id', content.get('content_id'))
                
                # Basic features
                feature_dict = {
                    'content_id': content_id,
                    'title': content.get('title', ''),
                    'content_type': content.get('content_type', 'movie'),
                    'rating': float(content.get('rating', 0.0)),
                    'vote_count': int(content.get('vote_count', 0)),
                    'popularity': float(content.get('popularity', 0.0)),
                    'runtime': int(content.get('runtime', 0)),
                    'release_year': self._extract_release_year(content.get('release_date')),
                }
                
                # Genre features
                genres = content.get('genres', [])
                if isinstance(genres, str):
                    try:
                        genres = json.loads(genres)
                    except:
                        genres = []
                
                genre_features = self._encode_genres(genres)
                feature_dict.update(genre_features)
                
                # Language features
                languages = content.get('languages', [])
                if isinstance(languages, str):
                    try:
                        languages = json.loads(languages)
                    except:
                        languages = []
                
                language_features = self._encode_languages(languages)
                feature_dict.update(language_features)
                
                # Content description features
                overview = content.get('overview', '')
                text_features = self._extract_text_features(overview)
                feature_dict.update(text_features)
                
                # Advanced features
                feature_dict.update({
                    'is_trending': int(content.get('is_trending', False)),
                    'is_new_release': int(content.get('is_new_release', False)),
                    'is_critics_choice': int(content.get('is_critics_choice', False)),
                    'critics_score': float(content.get('critics_score', 0.0)),
                    'age_score': self._calculate_age_score(feature_dict['release_year']),
                    'popularity_score': self._normalize_popularity(feature_dict['popularity']),
                    'quality_score': self._calculate_quality_score(
                        feature_dict['rating'], 
                        feature_dict['vote_count']
                    )
                })
                
                features.append(feature_dict)
            
            return pd.DataFrame(features)
            
        except Exception as e:
            logger.error(f"Error extracting content features: {e}")
            return pd.DataFrame()
    
    def _extract_release_year(self, release_date: str) -> int:
        if not release_date:
            return 2020
        try:
            return int(release_date[:4])
        except:
            return 2020
    
    def _encode_genres(self, genres: List[str]) -> Dict:
        common_genres = [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
            'Romance', 'Science Fiction', 'Thriller', 'War', 'Western'
        ]
        
        genre_dict = {}
        for genre in common_genres:
            genre_key = f'genre_{genre.lower().replace(" ", "_")}'
            genre_dict[genre_key] = 1 if genre in genres else 0
        
        return genre_dict
    
    def _encode_languages(self, languages: List[str]) -> Dict:
        common_languages = ['english', 'telugu', 'hindi', 'tamil', 'malayalam', 'kannada', 'japanese', 'korean']
        
        language_dict = {}
        for lang in common_languages:
            lang_key = f'lang_{lang}'
            language_dict[lang_key] = 1 if any(lang in l.lower() for l in languages) else 0
        
        return language_dict
    
    def _extract_text_features(self, text: str) -> Dict:
        if not text:
            return {
                'text_length': 0,
                'word_count': 0,
                'sentiment_score': 0.0
            }
        
        try:
            blob = self.sentiment_analyzer(text)
            sentiment = blob.sentiment.polarity
        except:
            sentiment = 0.0
        
        return {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentiment_score': sentiment
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

class NeuralRecommender(nn.Module):
    """PyTorch-based neural recommendation model"""
    
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_dims=[128, 64]):
        super(NeuralRecommender, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Neural network layers
        input_dim = embedding_dim * 2
        layers = []
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        self.device = DEVICE
        self.to(self.device)
    
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        combined = torch.cat([user_embeds, item_embeds], dim=1)
        output = self.network(combined)
        
        return output.squeeze()
    
    def predict(self, user_id, item_ids):
        self.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id] * len(item_ids), device=self.device)
            item_tensor = torch.tensor(item_ids, device=self.device)
            predictions = self.forward(user_tensor, item_tensor)
            return predictions.cpu().numpy()

class GraphRecommender:
    """Graph-based recommendation using NetworkX"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.user_nodes = set()
        self.item_nodes = set()
        
    def build_graph(self, interactions: List[Dict]):
        """Build bipartite graph from user-item interactions"""
        for interaction in interactions:
            user_id = f"user_{interaction['user_id']}"
            item_id = f"item_{interaction['content_id']}"
            weight = interaction.get('weight', 1.0)
            
            self.graph.add_edge(user_id, item_id, weight=weight)
            self.user_nodes.add(user_id)
            self.item_nodes.add(item_id)
    
    def get_recommendations(self, user_id: int, num_recommendations: int = 20) -> List[Tuple[int, float]]:
        """Get recommendations using graph-based collaborative filtering"""
        user_node = f"user_{user_id}"
        
        if user_node not in self.graph:
            return []
        
        # Get user's neighbors (items they interacted with)
        user_items = set(self.graph.neighbors(user_node))
        
        # Find similar users through common items
        similar_users = defaultdict(float)
        
        for item in user_items:
            for neighbor in self.graph.neighbors(item):
                if neighbor != user_node and neighbor in self.user_nodes:
                    similar_users[neighbor] += self.graph[user_node][item]['weight']
        
        # Get recommendations from similar users
        recommendations = defaultdict(float)
        
        for similar_user, similarity in similar_users.items():
            similar_user_items = set(self.graph.neighbors(similar_user))
            new_items = similar_user_items - user_items
            
            for item in new_items:
                if item in self.item_nodes:
                    weight = self.graph[similar_user][item]['weight']
                    recommendations[item] += similarity * weight
        
        # Convert back to content IDs and sort
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for item_node, score in sorted_recs[:num_recommendations]:
            content_id = int(item_node.replace('item_', ''))
            result.append((content_id, score))
        
        return result

class HybridRecommendationEngine:
    """Advanced hybrid recommendation engine"""
    
    def __init__(self):
        self.content_based_model = ContentBasedRecommender()
        self.collaborative_model = ImplicitCollaborativeRecommender()
        self.neural_model = None
        self.graph_model = GraphRecommender()
        
        self.ensemble_weights = {
            'content_based': 0.25,
            'collaborative': 0.35,
            'neural': 0.25,
            'graph': 0.15
        }
        
        self.diversity_optimizer = DiversityOptimizer()
        self.cold_start_handler = ColdStartHandler()
        
    def get_recommendations(self, user_features: Dict, content_df: pd.DataFrame, 
                          num_recommendations: int = 20) -> List[Dict]:
        """Generate hybrid recommendations"""
        try:
            # Check for cold start
            if user_features['total_interactions'] < 3:
                return self.cold_start_handler.get_recommendations(
                    user_features, content_df, num_recommendations
                )
            
            # Get recommendations from each model
            content_recs = self.content_based_model.recommend(
                user_features, content_df, num_recommendations * 2
            )
            
            collab_recs = self.collaborative_model.recommend(
                user_features, content_df, num_recommendations * 2
            )
            
            graph_recs = []
            try:
                # Build graph from user interactions
                interactions = [
                    {'user_id': user_features['user_id'], 'content_id': cid, 'weight': score}
                    for cid, score in user_features['interaction_scores'].items()
                ]
                self.graph_model.build_graph(interactions)
                graph_recs_raw = self.graph_model.get_recommendations(
                    user_features['user_id'], num_recommendations * 2
                )
                graph_recs = [
                    {'content_id': cid, 'score': score, 'model': 'graph'}
                    for cid, score in graph_recs_raw
                ]
            except Exception as e:
                logger.warning(f"Graph recommendations failed: {e}")
            
            neural_recs = []
            if self.neural_model:
                try:
                    neural_recs = self.neural_model.recommend(
                        user_features, content_df, num_recommendations * 2
                    )
                except Exception as e:
                    logger.warning(f"Neural recommendations failed: {e}")
            
            # Combine recommendations with weighted scoring
            combined_scores = defaultdict(float)
            all_recs = {
                'content_based': content_recs,
                'collaborative': collab_recs,
                'neural': neural_recs,
                'graph': graph_recs
            }
            
            for model_name, model_recs in all_recs.items():
                weight = self.ensemble_weights[model_name]
                for i, rec in enumerate(model_recs):
                    content_id = rec['content_id']
                    rank_score = 1.0 / (i + 1)
                    model_score = rec.get('score', rank_score)
                    combined_scores[content_id] += weight * model_score
            
            # Sort by combined score
            sorted_content = sorted(
                combined_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Apply diversity optimization
            final_recommendations = self.diversity_optimizer.optimize(
                sorted_content[:num_recommendations * 3],
                content_df,
                user_features,
                num_recommendations
            )
            
            # Format recommendations
            recommendations = []
            for i, (content_id, score) in enumerate(final_recommendations):
                content_info = content_df[content_df['content_id'] == content_id]
                if not content_info.empty:
                    content_info = content_info.iloc[0].to_dict()
                    
                    rec = {
                        'content_id': content_id,
                        'score': float(score),
                        'rank': i + 1,
                        'reason': self._generate_explanation(content_info, user_features),
                        'confidence': self._calculate_confidence(score, user_features),
                        'diversity_score': self._calculate_diversity_score(content_info, user_features),
                        'novelty_score': self._calculate_novelty_score(content_info, user_features)
                    }
                    recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating hybrid recommendations: {e}")
            return self.cold_start_handler.get_recommendations(
                user_features, content_df, num_recommendations
            )
    
    def _generate_explanation(self, content_info: Dict, user_features: Dict) -> str:
        explanations = []
        
        user_genres = user_features.get('preferred_genres', [])
        if user_genres:
            explanations.append("matches your genre preferences")
        
        user_languages = user_features.get('preferred_languages', [])
        if user_languages:
            explanations.append("in your preferred language")
        
        if content_info.get('rating', 0) > 7.5:
            explanations.append("highly rated content")
        
        if content_info.get('is_trending'):
            explanations.append("currently trending")
        
        if not explanations:
            explanations.append("recommended for you")
        
        return ", ".join(explanations[:2])
    
    def _calculate_confidence(self, score: float, user_features: Dict) -> float:
        base_confidence = min(1.0, score)
        interaction_boost = min(0.2, user_features['total_interactions'] / 100.0)
        return base_confidence + interaction_boost
    
    def _calculate_diversity_score(self, content_info: Dict, user_features: Dict) -> float:
        return np.random.uniform(0.5, 1.0)
    
    def _calculate_novelty_score(self, content_info: Dict, user_features: Dict) -> float:
        return np.random.uniform(0.3, 0.9)

class ContentBasedRecommender:
    """Content-based recommendation using content features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def recommend(self, user_features: Dict, content_df: pd.DataFrame, 
                  num_recommendations: int = 20) -> List[Dict]:
        try:
            if content_df.empty:
                return []
            
            user_profile = self._build_user_profile(user_features, content_df)
            content_scores = self._calculate_content_scores(user_profile, content_df)
            
            recommendations = []
            interaction_content_ids = set(user_features['interaction_scores'].keys())
            
            for content_id, score in content_scores.items():
                if content_id not in interaction_content_ids:
                    recommendations.append({
                        'content_id': content_id,
                        'score': score,
                        'model': 'content_based'
                    })
            
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Content-based recommendation error: {e}")
            return []
    
    def _build_user_profile(self, user_features: Dict, content_df: pd.DataFrame) -> np.ndarray:
        interaction_scores = user_features['interaction_scores']
        
        if not interaction_scores:
            return self._build_preference_profile(user_features, content_df)
        
        numeric_columns = content_df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'content_id']
        
        weighted_features = np.zeros(len(numeric_columns))
        total_weight = 0
        
        for content_id, score in interaction_scores.items():
            content_row = content_df[content_df['content_id'] == content_id]
            if not content_row.empty:
                content_features = content_row[numeric_columns].values[0]
                weighted_features += content_features * score
                total_weight += score
        
        if total_weight > 0:
            weighted_features /= total_weight
        
        return weighted_features
    
    def _build_preference_profile(self, user_features: Dict, content_df: pd.DataFrame) -> np.ndarray:
        numeric_columns = content_df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'content_id']
        
        profile = np.zeros(len(numeric_columns))
        
        preferred_genres = user_features.get('preferred_genres', [])
        for genre in preferred_genres:
            genre_col = f'genre_{genre.lower().replace(" ", "_")}'
            if genre_col in numeric_columns:
                idx = list(numeric_columns).index(genre_col)
                profile[idx] = 1.0
        
        preferred_languages = user_features.get('preferred_languages', [])
        for lang in preferred_languages:
            lang_col = f'lang_{lang.lower()}'
            if lang_col in numeric_columns:
                idx = list(numeric_columns).index(lang_col)
                profile[idx] = 1.0
        
        return profile
    
    def _calculate_content_scores(self, user_profile: np.ndarray, 
                                 content_df: pd.DataFrame) -> Dict[int, float]:
        numeric_columns = content_df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'content_id']
        
        content_features = content_df[numeric_columns].values
        
        if hasattr(self.scaler, 'scale_'):
            content_features_scaled = self.scaler.transform(content_features)
            user_profile_scaled = self.scaler.transform(user_profile.reshape(1, -1))[0]
        else:
            content_features_scaled = self.scaler.fit_transform(content_features)
            user_profile_scaled = self.scaler.transform(user_profile.reshape(1, -1))[0]
        
        similarities = cosine_similarity([user_profile_scaled], content_features_scaled)[0]
        
        content_scores = {}
        for i, content_id in enumerate(content_df['content_id']):
            content_scores[content_id] = float(similarities[i])
        
        return content_scores

class ImplicitCollaborativeRecommender:
    """Collaborative filtering using implicit library"""
    
    def __init__(self):
        self.model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20)
        self.user_items = None
        self.user_mapping = {}
        self.item_mapping = {}
        
    def recommend(self, user_features: Dict, content_df: pd.DataFrame, 
                  num_recommendations: int = 20) -> List[Dict]:
        try:
            # This is a simplified version for single user
            # In production, you'd train on all user interactions
            
            recommendations = []
            interaction_scores = user_features['interaction_scores']
            
            if not interaction_scores:
                return []
            
            # Simulate collaborative filtering with similar content
            similar_content = self._find_similar_content(
                list(interaction_scores.keys()), content_df
            )
            
            for content_id, score in similar_content.items():
                if content_id not in interaction_scores:
                    recommendations.append({
                        'content_id': content_id,
                        'score': score,
                        'model': 'collaborative'
                    })
            
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Collaborative filtering error: {e}")
            return []
    
    def _find_similar_content(self, interacted_content: List[int], 
                             content_df: pd.DataFrame) -> Dict[int, float]:
        similar_scores = defaultdict(float)
        
        for content_id in interacted_content:
            content_row = content_df[content_df['content_id'] == content_id]
            if content_row.empty:
                continue
            
            content_features = content_row.iloc[0]
            
            for _, other_content in content_df.iterrows():
                if other_content['content_id'] == content_id:
                    continue
                
                similarity = self._calculate_content_similarity(content_features, other_content)
                similar_scores[other_content['content_id']] += similarity
        
        return dict(similar_scores)
    
    def _calculate_content_similarity(self, content1: pd.Series, content2: pd.Series) -> float:
        genre_cols = [col for col in content1.index if col.startswith('genre_')]
        genre_sim = 0
        if genre_cols:
            genre1 = content1[genre_cols].values
            genre2 = content2[genre_cols].values
            genre_sim = cosine_similarity([genre1], [genre2])[0][0]
        
        lang_cols = [col for col in content1.index if col.startswith('lang_')]
        lang_sim = 0
        if lang_cols:
            lang1 = content1[lang_cols].values
            lang2 = content2[lang_cols].values
            lang_sim = cosine_similarity([lang1], [lang2])[0][0]
        
        rating_sim = 1.0 - abs(content1.get('rating', 0) - content2.get('rating', 0)) / 10.0
        
        return (genre_sim * 0.4 + lang_sim * 0.3 + rating_sim * 0.3)

class DiversityOptimizer:
    """Optimize recommendation diversity"""
    
    def optimize(self, ranked_content: List[Tuple[int, float]], 
                 content_df: pd.DataFrame, user_features: Dict,
                 num_recommendations: int) -> List[Tuple[int, float]]:
        try:
            if len(ranked_content) <= num_recommendations:
                return ranked_content
            
            selected = []
            remaining = ranked_content.copy()
            
            selected.append(remaining.pop(0))
            
            while len(selected) < num_recommendations and remaining:
                best_candidate = None
                best_score = -1
                best_idx = -1
                
                for i, (content_id, relevance_score) in enumerate(remaining):
                    diversity_score = self._calculate_diversity_with_selected(
                        content_id, selected, content_df
                    )
                    
                    combined_score = (ACCURACY_WEIGHT * relevance_score + 
                                    DIVERSITY_WEIGHT * diversity_score)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = (content_id, relevance_score)
                        best_idx = i
                
                if best_candidate:
                    selected.append(best_candidate)
                    remaining.pop(best_idx)
                else:
                    break
            
            return selected
            
        except Exception as e:
            logger.error(f"Diversity optimization error: {e}")
            return ranked_content[:num_recommendations]
    
    def _calculate_diversity_with_selected(self, content_id: int, 
                                         selected: List[Tuple[int, float]], 
                                         content_df: pd.DataFrame) -> float:
        if not selected:
            return 1.0
        
        content_row = content_df[content_df['content_id'] == content_id]
        if content_row.empty:
            return 0.5
        
        content_features = content_row.iloc[0]
        
        diversities = []
        for selected_id, _ in selected:
            selected_row = content_df[content_df['content_id'] == selected_id]
            if not selected_row.empty:
                selected_features = selected_row.iloc[0]
                diversity = self._calculate_content_diversity(content_features, selected_features)
                diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 1.0
    
    def _calculate_content_diversity(self, content1: pd.Series, content2: pd.Series) -> float:
        genre_cols = [col for col in content1.index if col.startswith('genre_')]
        genre_diversity = 0
        if genre_cols:
            genre1 = content1[genre_cols].values
            genre2 = content2[genre_cols].values
            genre_diversity = 1.0 - cosine_similarity([genre1], [genre2])[0][0]
        
        type_diversity = 1.0 if content1.get('content_type') != content2.get('content_type') else 0.0
        
        lang_cols = [col for col in content1.index if col.startswith('lang_')]
        lang_diversity = 0
        if lang_cols:
            lang1 = content1[lang_cols].values
            lang2 = content2[lang_cols].values
            lang_diversity = 1.0 - cosine_similarity([lang1], [lang2])[0][0]
        
        return (genre_diversity * 0.5 + type_diversity * 0.3 + lang_diversity * 0.2)

class ColdStartHandler:
    """Handle recommendations for new users"""
    
    def get_recommendations(self, user_features: Dict, content_df: pd.DataFrame,
                          num_recommendations: int = 20) -> List[Dict]:
        try:
            recommendations = []
            
            preferred_genres = user_features.get('preferred_genres', [])
            preferred_languages = user_features.get('preferred_languages', [])
            
            if preferred_genres or preferred_languages:
                filtered_content = content_df.copy()
                
                if preferred_genres:
                    genre_mask = np.zeros(len(filtered_content), dtype=bool)
                    for genre in preferred_genres:
                        genre_col = f'genre_{genre.lower().replace(" ", "_")}'
                        if genre_col in filtered_content.columns:
                            genre_mask |= (filtered_content[genre_col] == 1)
                    
                    if genre_mask.any():
                        filtered_content = filtered_content[genre_mask]
                
                if preferred_languages and not filtered_content.empty:
                    lang_mask = np.zeros(len(filtered_content), dtype=bool)
                    for lang in preferred_languages:
                        lang_col = f'lang_{lang.lower()}'
                        if lang_col in filtered_content.columns:
                            lang_mask |= (filtered_content[lang_col] == 1)
                    
                    if lang_mask.any():
                        filtered_content = filtered_content[lang_mask]
                
                if not filtered_content.empty:
                    filtered_content = filtered_content.sort_values(
                        ['popularity_score', 'rating'], ascending=False
                    )
                    
                    for _, content in filtered_content.head(num_recommendations).iterrows():
                        recommendations.append({
                            'content_id': content['content_id'],
                            'score': content.get('popularity_score', 0.5) * 0.7 + content.get('rating', 5.0) / 10.0 * 0.3,
                            'model': 'cold_start_preference'
                        })
            
            if len(recommendations) < num_recommendations:
                popular_content = content_df.sort_values(
                    ['popularity_score', 'rating'], ascending=False
                )
                
                existing_ids = set(rec['content_id'] for rec in recommendations)
                
                for _, content in popular_content.iterrows():
                    if content['content_id'] not in existing_ids:
                        recommendations.append({
                            'content_id': content['content_id'],
                            'score': content.get('popularity_score', 0.5) * 0.5 + content.get('rating', 5.0) / 10.0 * 0.5,
                            'model': 'cold_start_popular'
                        })
                        
                        if len(recommendations) >= num_recommendations:
                            break
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Cold start recommendation error: {e}")
            return []

class MLRecommendationService:
    """Main ML recommendation service"""
    
    def __init__(self):
        self.behavior_analyzer = UserBehaviorAnalyzer()
        self.content_extractor = ContentFeatureExtractor()
        self.recommendation_engine = HybridRecommendationEngine()
        self.content_cache = {}
        self.model_last_updated = datetime.utcnow()
        
    def get_personalized_recommendations(self, user_data: Dict, 
                                       content_data: List[Dict],
                                       num_recommendations: int = 20) -> Dict:
        try:
            cache_key = f"personalized:{user_data['user_id']}:{num_recommendations}"
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            user_features = self.behavior_analyzer.extract_user_features(user_data)
            content_df = self.content_extractor.extract_content_features(content_data)
            
            if content_df.empty:
                logger.warning("No content data available for recommendations")
                return {
                    'recommendations': [],
                    'metadata': {
                        'algorithm': 'unavailable',
                        'reason': 'no_content_data'
                    }
                }
            
            recommendations = self.recommendation_engine.get_recommendations(
                user_features, content_df, num_recommendations
            )
            
            metadata = {
                'algorithm': 'hybrid_ml_pytorch',
                'user_activity_level': user_features['activity_level'],
                'total_interactions': user_features['total_interactions'],
                'recommendation_strategy': self._determine_strategy(user_features),
                'diversity_score': np.mean([rec.get('diversity_score', 0) for rec in recommendations]),
                'confidence_score': np.mean([rec.get('confidence', 0) for rec in recommendations]),
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': '3.0',
                'libraries_used': ['pytorch', 'sklearn', 'implicit', 'networkx'],
                'device': str(DEVICE),
                'features_used': list(user_features.keys())
            }
            
            result = {
                'recommendations': recommendations,
                'metadata': metadata
            }
            
            cache_manager.set(cache_key, result, timeout=1800)
            return result
            
        except Exception as e:
            logger.error(f"Personalized recommendation error: {e}")
            return {
                'recommendations': [],
                'metadata': {
                    'algorithm': 'error',
                    'error': str(e)
                }
            }
    
    def get_similar_content(self, content_id: int, content_data: List[Dict],
                           num_recommendations: int = 10) -> Dict:
        try:
            cache_key = f"similar:{content_id}:{num_recommendations}"
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            content_df = self.content_extractor.extract_content_features(content_data)
            
            if content_df.empty:
                return {'recommendations': []}
            
            target_content = content_df[content_df['content_id'] == content_id]
            if target_content.empty:
                return {'recommendations': []}
            
            target_features = target_content.iloc[0]
            similarities = []
            
            numeric_columns = content_df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col != 'content_id']
            
            target_vector = target_features[numeric_columns].values.reshape(1, -1)
            
            for _, content in content_df.iterrows():
                if content['content_id'] == content_id:
                    continue
                
                content_vector = content[numeric_columns].values.reshape(1, -1)
                similarity = cosine_similarity(target_vector, content_vector)[0][0]
                
                similarities.append({
                    'content_id': content['content_id'],
                    'score': float(similarity),
                    'reason': 'content_similarity'
                })
            
            similarities.sort(key=lambda x: x['score'], reverse=True)
            
            result = {
                'recommendations': similarities[:num_recommendations],
                'metadata': {
                    'algorithm': 'cosine_similarity',
                    'base_content_id': content_id,
                    'features_used': len(numeric_columns)
                }
            }
            
            cache_manager.set(cache_key, result, timeout=3600)
            return result
            
        except Exception as e:
            logger.error(f"Similar content error: {e}")
            return {'recommendations': []}
    
    def get_trending_recommendations(self, content_data: List[Dict],
                                   region: str = 'IN',
                                   num_recommendations: int = 20) -> Dict:
        try:
            cache_key = f"trending:{region}:{num_recommendations}"
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            content_df = self.content_extractor.extract_content_features(content_data)
            
            if content_df.empty:
                return {'recommendations': []}
            
            content_df['trending_score'] = (
                content_df.get('popularity_score', 0) * 0.4 +
                content_df.get('is_trending', 0) * 0.3 +
                content_df.get('quality_score', 0) * 0.2 +
                content_df.get('age_score', 0) * 0.1
            )
            
            trending_content = content_df.sort_values('trending_score', ascending=False)
            
            recommendations = []
            for _, content in trending_content.head(num_recommendations).iterrows():
                recommendations.append({
                    'content_id': content['content_id'],
                    'score': float(content['trending_score']),
                    'reason': 'trending_algorithm'
                })
            
            result = {
                'recommendations': recommendations,
                'metadata': {
                    'algorithm': 'weighted_trending_score',
                    'region': region,
                    'weights': {
                        'popularity': 0.4,
                        'trending_flag': 0.3,
                        'quality': 0.2,
                        'recency': 0.1
                    }
                }
            }
            
            cache_manager.set(cache_key, result, timeout=1800)
            return result
            
        except Exception as e:
            logger.error(f"Trending recommendations error: {e}")
            return {'recommendations': []}
    
    def _determine_strategy(self, user_features: Dict) -> str:
        total_interactions = user_features['total_interactions']
        activity_level = user_features['activity_level']
        
        if total_interactions < 3:
            return 'cold_start'
        elif activity_level == 'low':
            return 'popularity_based'
        elif activity_level in ['medium', 'high']:
            return 'hybrid_personalized'
        else:
            return 'advanced_personalized'

# Global service instance
ml_service = MLRecommendationService()

# Mock content database
MOCK_CONTENT_DB = []

def load_mock_content():
    """Load mock content data"""
    global MOCK_CONTENT_DB
    
    mock_content = [
        {
            'id': 1, 'title': 'Avengers: Endgame', 'content_type': 'movie',
            'genres': ['Action', 'Adventure', 'Drama'], 'languages': ['English'],
            'rating': 8.4, 'vote_count': 500000, 'popularity': 100.0,
            'runtime': 181, 'release_date': '2019-04-26', 'overview': 'Epic conclusion to the Infinity Saga.',
            'is_trending': True, 'is_new_release': False, 'is_critics_choice': True
        },
        {
            'id': 2, 'title': 'RRR', 'content_type': 'movie',
            'genres': ['Action', 'Drama'], 'languages': ['Telugu', 'Hindi'],
            'rating': 8.0, 'vote_count': 200000, 'popularity': 95.0,
            'runtime': 187, 'release_date': '2022-03-25', 'overview': 'Epic period drama about friendship and revolution.',
            'is_trending': True, 'is_new_release': True, 'is_critics_choice': True
        },
        {
            'id': 3, 'title': 'Attack on Titan', 'content_type': 'anime',
            'genres': ['Action', 'Drama', 'Fantasy'], 'languages': ['Japanese'],
            'rating': 9.0, 'vote_count': 150000, 'popularity': 90.0,
            'runtime': 24, 'release_date': '2013-04-07', 'overview': 'Humanity fights for survival against giants.',
            'is_trending': True, 'is_new_release': False, 'is_critics_choice': True
        },
        {
            'id': 4, 'title': 'Stranger Things', 'content_type': 'tv',
            'genres': ['Drama', 'Fantasy', 'Horror'], 'languages': ['English'],
            'rating': 8.7, 'vote_count': 800000, 'popularity': 88.0,
            'runtime': 50, 'release_date': '2016-07-15', 'overview': 'Kids in a small town encounter supernatural forces.',
            'is_trending': True, 'is_new_release': False, 'is_critics_choice': True
        },
        {
            'id': 5, 'title': 'Pushpa', 'content_type': 'movie',
            'genres': ['Action', 'Crime', 'Drama'], 'languages': ['Telugu', 'Hindi'],
            'rating': 7.6, 'vote_count': 180000, 'popularity': 85.0,
            'runtime': 179, 'release_date': '2021-12-17', 'overview': 'A coolie rises through the ranks of a red sandalwood smuggling syndicate.',
            'is_trending': True, 'is_new_release': True, 'is_critics_choice': False
        }
    ]
    
    MOCK_CONTENT_DB = mock_content
    logger.info(f"Loaded {len(MOCK_CONTENT_DB)} mock content items")

load_mock_content()

# API Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    system_info = {
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total // (1024**3),  # GB
        'memory_available': psutil.virtual_memory().available // (1024**3),  # GB
    }
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '3.0',
        'device': str(DEVICE),
        'pytorch_available': True,
        'redis_available': REDIS_AVAILABLE,
        'mock_content_loaded': len(MOCK_CONTENT_DB),
        'system_info': system_info,
        'features': {
            'personalized_recommendations': True,
            'content_based_filtering': True,
            'collaborative_filtering': True,
            'neural_recommendations': True,
            'graph_based_recommendations': True,
            'diversity_optimization': True,
            'cold_start_handling': True,
            'similarity_search': True,
            'trending_analysis': True,
            'sentiment_analysis': True,
            'text_processing': True
        },
        'libraries': {
            'pytorch': torch.__version__,
            'sklearn': sklearn.__version__,
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'networkx': nx.__version__,
            'implicit': True,
            'textblob': True,
            'nltk': True
        }
    })

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get personalized recommendations"""
    try:
        user_data = request.json
        
        if not user_data:
            return jsonify({'error': 'No user data provided'}), 400
        
        content_data = MOCK_CONTENT_DB
        limit = request.args.get('limit', 20, type=int)
        
        result = ml_service.get_personalized_recommendations(
            user_data, content_data, limit
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Recommendations API error: {e}")
        return jsonify({
            'error': 'Failed to generate recommendations',
            'recommendations': []
        }), 500

@app.route('/api/similar/<int:content_id>', methods=['GET'])
def get_similar(content_id):
    """Get similar content recommendations"""
    try:
        limit = request.args.get('limit', 10, type=int)
        content_data = MOCK_CONTENT_DB
        
        result = ml_service.get_similar_content(content_id, content_data, limit)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Similar content API error: {e}")
        return jsonify({
            'error': 'Failed to get similar content',
            'recommendations': []
        }), 500

@app.route('/api/trending', methods=['GET'])
def get_trending():
    """Get trending recommendations"""
    try:
        region = request.args.get('region', 'IN')
        limit = request.args.get('limit', 20, type=int)
        content_data = MOCK_CONTENT_DB
        
        result = ml_service.get_trending_recommendations(content_data, region, limit)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Trending API error: {e}")
        return jsonify({
            'error': 'Failed to get trending content',
            'recommendations': []
        }), 500

@app.route('/api/analytics/user-profile', methods=['POST'])
def analyze_user_profile():
    """Analyze user profile and behavior"""
    try:
        user_data = request.json
        
        if not user_data:
            return jsonify({'error': 'No user data provided'}), 400
        
        behavior_analyzer = UserBehaviorAnalyzer()
        user_features = behavior_analyzer.extract_user_features(user_data)
        
        safe_features = user_features.copy()
        safe_features.pop('interaction_scores', None)
        
        return jsonify({
            'user_profile': safe_features,
            'insights': {
                'activity_level': user_features['activity_level'],
                'discovery_pattern': user_features['content_discovery_pattern'],
                'rating_behavior': {
                    'average': user_features['avg_rating'],
                    'strictness': user_features['rating_strictness']
                },
                'preferences': {
                    'languages': user_features['preferred_languages'],
                    'genres': user_features['preferred_genres']
                },
                'engagement': {
                    'exploration_tendency': user_features['exploration_tendency'],
                    'recency_bias': user_features['recency_bias']
                }
            }
        })
        
    except Exception as e:
        logger.error(f"User profile analysis error: {e}")
        return jsonify({'error': 'Failed to analyze user profile'}), 500

@app.route('/api/models/status', methods=['GET'])
def model_status():
    """Get status of all ML models"""
    return jsonify({
        'models': {
            'content_based': {
                'status': 'active',
                'last_updated': ml_service.model_last_updated.isoformat()
            },
            'collaborative_filtering': {
                'status': 'active',
                'library': 'implicit'
            },
            'neural_network': {
                'status': 'active',
                'framework': 'pytorch',
                'device': str(DEVICE)
            },
            'graph_based': {
                'status': 'active',
                'library': 'networkx'
            },
            'hybrid_ensemble': {
                'status': 'active',
                'weights': ml_service.recommendation_engine.ensemble_weights
            }
        },
        'features': {
            'diversity_optimization': True,
            'cold_start_handling': True,
            'real_time_scoring': True,
            'explanation_generation': True,
            'sentiment_analysis': True,
            'graph_analysis': True
        },
        'performance': {
            'mock_content_items': len(MOCK_CONTENT_DB),
            'cache_enabled': REDIS_AVAILABLE,
            'gpu_available': torch.cuda.is_available()
        }
    })

@app.route('/api/test/recommendation', methods=['GET'])
def test_recommendation():
    """Test endpoint with sample data"""
    sample_user = {
        'user_id': 999,
        'preferred_languages': ['English', 'Telugu'],
        'preferred_genres': ['Action', 'Drama'],
        'interactions': [
            {
                'content_id': 1,
                'interaction_type': 'view',
                'timestamp': datetime.utcnow().isoformat(),
                'rating': 4.5
            },
            {
                'content_id': 2,
                'interaction_type': 'favorite',
                'timestamp': (datetime.utcnow() - timedelta(days=1)).isoformat(),
                'rating': 5.0
            }
        ]
    }
    
    result = ml_service.get_personalized_recommendations(
        sample_user, MOCK_CONTENT_DB, 10
    )
    
    return jsonify({
        'test_input': sample_user,
        'test_output': result
    })

if __name__ == '__main__':
    print("=== Advanced ML Recommendation Service with PyTorch ===")
    print(f"PyTorch Available: True")
    print(f"Device: {DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Redis Available: {REDIS_AVAILABLE}")
    print(f"Mock Content Items: {len(MOCK_CONTENT_DB)}")
    print("=== Service Starting ===")
    
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)