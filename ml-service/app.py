# ml-service/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import os
import pickle
import sqlite3
from collections import defaultdict, Counter
import requests
import hashlib
import time
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from surprise import Dataset, Reader, SVD, NMF as SurpriseNMF, accuracy
from surprise.model_selection import train_test_split
import implicit
import scipy.sparse as sp

# PyTorch for Neural Networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CACHE_DURATION = 3600  # 1 hour
MODEL_UPDATE_INTERVAL = 86400  # 24 hours
MIN_INTERACTIONS = 5
RECOMMENDATION_CACHE = {}
USER_EMBEDDINGS_CACHE = {}
ITEM_EMBEDDINGS_CACHE = {}

# Behavioral weights for different interaction types
INTERACTION_WEIGHTS = {
    'search': 0.1,
    'view': 0.2,
    'like': 0.5,
    'favorite': 0.8,
    'watchlist': 0.7,
    'rating': 1.0
}

# Temporal decay factors
TEMPORAL_DECAY = {
    'recent': 1.0,      # Last 7 days
    'week': 0.8,        # 7-30 days
    'month': 0.6,       # 30-90 days
    'quarter': 0.4,     # 90-180 days
    'old': 0.2          # 180+ days
}

# Global variables for models
collaborative_model = None
content_model = None
svd_model = None
neural_model = None
item_similarity_matrix = None
user_similarity_matrix = None
content_features = None
last_model_update = 0

# Database connection for caching
def get_cache_db():
    return sqlite3.connect('ml_cache.db', check_same_thread=False)

# Initialize cache database
def init_cache_db():
    conn = get_cache_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendation_cache (
            cache_key TEXT PRIMARY KEY,
            recommendations TEXT,
            timestamp REAL,
            expires_at REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY,
            profile_data TEXT,
            last_updated REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_cache (
            model_name TEXT PRIMARY KEY,
            model_data BLOB,
            last_updated REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_behavior_patterns (
            user_id INTEGER,
            behavior_type TEXT,
            content_features TEXT,
            timestamp REAL,
            weight REAL
        )
    ''')
    conn.commit()
    conn.close()

# Advanced User Profile Builder
class UserProfileBuilder:
    def __init__(self):
        self.user_profiles = {}
        self.genre_preferences = defaultdict(lambda: defaultdict(float))
        self.language_preferences = defaultdict(lambda: defaultdict(float))
        self.temporal_patterns = defaultdict(list)
        self.search_patterns = defaultdict(list)
        self.content_type_preferences = defaultdict(lambda: defaultdict(float))
        
    def build_user_profile(self, user_id, interactions, content_data):
        """Build comprehensive user profile from all interactions"""
        try:
            profile = {
                'user_id': user_id,
                'genre_scores': defaultdict(float),
                'language_scores': defaultdict(float),
                'content_type_scores': defaultdict(float),
                'rating_patterns': [],
                'temporal_preferences': defaultdict(float),
                'search_keywords': [],
                'favorite_features': [],
                'watchlist_features': [],
                'interaction_frequency': defaultdict(int),
                'diversity_score': 0.0,
                'exploration_tendency': 0.0,
                'quality_preference': 0.0
            }
            
            # Process each interaction with temporal weighting
            current_time = datetime.now()
            total_weight = 0
            
            for interaction in interactions:
                try:
                    interaction_time = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
                    days_ago = (current_time - interaction_time).days
                    
                    # Calculate temporal weight
                    temporal_weight = self._get_temporal_weight(days_ago)
                    
                    # Get interaction weight
                    interaction_weight = INTERACTION_WEIGHTS.get(interaction['interaction_type'], 0.3)
                    
                    # Combined weight
                    final_weight = temporal_weight * interaction_weight
                    total_weight += final_weight
                    
                    # Get content details
                    content_id = interaction['content_id']
                    content_row = content_data[content_data['id'] == content_id]
                    
                    if not content_row.empty:
                        content_info = content_row.iloc[0]
                        
                        # Process genres
                        self._process_genres(profile, content_info, final_weight, interaction)
                        
                        # Process languages
                        self._process_languages(profile, content_info, final_weight)
                        
                        # Process content types
                        self._process_content_types(profile, content_info, final_weight)
                        
                        # Process ratings
                        self._process_ratings(profile, interaction, final_weight)
                        
                        # Process special interactions (favorites, watchlist)
                        self._process_special_interactions(profile, interaction, content_info, final_weight)
                        
                        # Track interaction frequency
                        profile['interaction_frequency'][interaction['interaction_type']] += 1
                
                except Exception as interaction_error:
                    logger.error(f"Error processing interaction: {interaction_error}")
                    continue
            
            # Normalize scores
            if total_weight > 0:
                for genre in profile['genre_scores']:
                    profile['genre_scores'][genre] /= total_weight
                for language in profile['language_scores']:
                    profile['language_scores'][language] /= total_weight
                for content_type in profile['content_type_scores']:
                    profile['content_type_scores'][content_type] /= total_weight
            
            # Calculate derived metrics
            profile['diversity_score'] = self._calculate_diversity_score(profile)
            profile['exploration_tendency'] = self._calculate_exploration_tendency(interactions)
            profile['quality_preference'] = self._calculate_quality_preference(profile)
            
            # Store profile
            self.user_profiles[user_id] = profile
            return profile
            
        except Exception as e:
            logger.error(f"Error building user profile: {e}")
            return None
    
    def _get_temporal_weight(self, days_ago):
        """Calculate temporal weight based on recency"""
        if days_ago <= 7:
            return TEMPORAL_DECAY['recent']
        elif days_ago <= 30:
            return TEMPORAL_DECAY['week']
        elif days_ago <= 90:
            return TEMPORAL_DECAY['month']
        elif days_ago <= 180:
            return TEMPORAL_DECAY['quarter']
        else:
            return TEMPORAL_DECAY['old']
    
    def _process_genres(self, profile, content_info, weight, interaction):
        """Process genre preferences"""
        try:
            genres = json.loads(content_info.get('genres', '[]'))
            for genre in genres:
                profile['genre_scores'][genre] += weight
                
                # Boost for high-engagement interactions
                if interaction['interaction_type'] in ['favorite', 'rating']:
                    rating = interaction.get('rating', 0)
                    if rating >= 7:
                        profile['genre_scores'][genre] += weight * 0.5
        except:
            pass
    
    def _process_languages(self, profile, content_info, weight):
        """Process language preferences"""
        try:
            languages = json.loads(content_info.get('languages', '[]'))
            for language in languages:
                profile['language_scores'][language] += weight
        except:
            pass
    
    def _process_content_types(self, profile, content_info, weight):
        """Process content type preferences"""
        content_type = content_info.get('content_type', 'unknown')
        profile['content_type_scores'][content_type] += weight
    
    def _process_ratings(self, profile, interaction, weight):
        """Process rating patterns"""
        if interaction.get('rating'):
            profile['rating_patterns'].append({
                'rating': interaction['rating'],
                'weight': weight,
                'interaction_type': interaction['interaction_type']
            })
    
    def _process_special_interactions(self, profile, interaction, content_info, weight):
        """Process favorites and watchlist interactions"""
        try:
            if interaction['interaction_type'] == 'favorite':
                features = {
                    'genres': json.loads(content_info.get('genres', '[]')),
                    'languages': json.loads(content_info.get('languages', '[]')),
                    'rating': content_info.get('rating', 0),
                    'content_type': content_info.get('content_type')
                }
                profile['favorite_features'].append(features)
            
            elif interaction['interaction_type'] == 'watchlist':
                features = {
                    'genres': json.loads(content_info.get('genres', '[]')),
                    'languages': json.loads(content_info.get('languages', '[]')),
                    'rating': content_info.get('rating', 0),
                    'content_type': content_info.get('content_type')
                }
                profile['watchlist_features'].append(features)
                
        except Exception as e:
            logger.error(f"Error processing special interactions: {e}")
    
    def _calculate_diversity_score(self, profile):
        """Calculate user's content diversity preference"""
        try:
            genre_count = len(profile['genre_scores'])
            language_count = len(profile['language_scores'])
            content_type_count = len(profile['content_type_scores'])
            
            # Normalize based on available options
            diversity = (genre_count / 20.0) + (language_count / 10.0) + (content_type_count / 5.0)
            return min(diversity, 1.0)
        except:
            return 0.5
    
    def _calculate_exploration_tendency(self, interactions):
        """Calculate user's tendency to explore new content"""
        try:
            search_count = sum(1 for i in interactions if i['interaction_type'] == 'search')
            total_interactions = len(interactions)
            
            if total_interactions == 0:
                return 0.5
            
            return min(search_count / total_interactions, 1.0)
        except:
            return 0.5
    
    def _calculate_quality_preference(self, profile):
        """Calculate user's preference for high-quality content"""
        try:
            if not profile['rating_patterns']:
                return 0.7  # Default assumption
            
            high_ratings = sum(1 for r in profile['rating_patterns'] if r['rating'] >= 7)
            total_ratings = len(profile['rating_patterns'])
            
            return high_ratings / total_ratings if total_ratings > 0 else 0.7
        except:
            return 0.7

# Advanced Search Pattern Analyzer
class SearchPatternAnalyzer:
    def __init__(self):
        self.search_embeddings = {}
        self.search_clusters = {}
        self.keyword_weights = defaultdict(float)
        
    def analyze_search_patterns(self, user_id, search_history, content_data):
        """Analyze user's search patterns for better recommendations"""
        try:
            if not search_history:
                return {}
            
            # Extract search queries and contexts
            search_queries = []
            search_contexts = []
            
            for search in search_history:
                query = search.get('query', '').lower()
                if query:
                    search_queries.append(query)
                    
                    # Analyze what user clicked after search
                    clicked_content = search.get('clicked_content_ids', [])
                    context = self._extract_search_context(clicked_content, content_data)
                    search_contexts.append(context)
            
            # Build search intent profile
            intent_profile = {
                'preferred_keywords': self._extract_preferred_keywords(search_queries),
                'search_to_action_patterns': self._analyze_search_to_action(search_contexts),
                'genre_search_patterns': self._analyze_genre_search_patterns(search_queries, search_contexts),
                'temporal_search_patterns': self._analyze_temporal_search_patterns(search_history)
            }
            
            return intent_profile
            
        except Exception as e:
            logger.error(f"Error analyzing search patterns: {e}")
            return {}
    
    def _extract_search_context(self, clicked_content_ids, content_data):
        """Extract context from content user clicked after search"""
        context = {
            'genres': [],
            'languages': [],
            'content_types': [],
            'ratings': []
        }
        
        for content_id in clicked_content_ids:
            content_row = content_data[content_data['id'] == content_id]
            if not content_row.empty:
                content_info = content_row.iloc[0]
                
                try:
                    context['genres'].extend(json.loads(content_info.get('genres', '[]')))
                    context['languages'].extend(json.loads(content_info.get('languages', '[]')))
                    context['content_types'].append(content_info.get('content_type'))
                    context['ratings'].append(content_info.get('rating', 0))
                except:
                    pass
        
        return context
    
    def _extract_preferred_keywords(self, search_queries):
        """Extract frequently used keywords"""
        keyword_freq = Counter()
        
        for query in search_queries:
            # Simple keyword extraction (can be enhanced with NLP)
            words = query.split()
            for word in words:
                if len(word) > 2:  # Filter short words
                    keyword_freq[word] += 1
        
        return dict(keyword_freq.most_common(20))
    
    def _analyze_search_to_action(self, search_contexts):
        """Analyze patterns between search and subsequent actions"""
        patterns = {
            'genre_preferences': defaultdict(int),
            'language_preferences': defaultdict(int),
            'content_type_preferences': defaultdict(int)
        }
        
        for context in search_contexts:
            for genre in context['genres']:
                patterns['genre_preferences'][genre] += 1
            for language in context['languages']:
                patterns['language_preferences'][language] += 1
            for content_type in context['content_types']:
                patterns['content_type_preferences'][content_type] += 1
        
        return patterns
    
    def _analyze_genre_search_patterns(self, search_queries, search_contexts):
        """Analyze genre-specific search patterns"""
        genre_patterns = defaultdict(list)
        
        for query, context in zip(search_queries, search_contexts):
            for genre in context['genres']:
                genre_patterns[genre].append(query)
        
        return dict(genre_patterns)
    
    def _analyze_temporal_search_patterns(self, search_history):
        """Analyze when user searches for different types of content"""
        temporal_patterns = defaultdict(list)
        
        for search in search_history:
            try:
                timestamp = datetime.fromisoformat(search['timestamp'].replace('Z', '+00:00'))
                hour = timestamp.hour
                day_of_week = timestamp.weekday()
                
                temporal_patterns['hourly_patterns'].append(hour)
                temporal_patterns['daily_patterns'].append(day_of_week)
            except:
                pass
        
        return dict(temporal_patterns)

# PyTorch Neural Collaborative Filtering Model (Enhanced)
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dims=[128, 64]):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # Enhanced embeddings with more features
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Additional feature embeddings
        self.user_behavior_embedding = nn.Embedding(10, 16)  # For different interaction types
        self.temporal_embedding = nn.Embedding(7, 8)  # For day of week
        
        # MLP layers with batch normalization
        mlp_input_dim = embedding_dim * 2 + 16 + 8
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            mlp_input_dim = hidden_dim
        
        layers.append(nn.Linear(mlp_input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids, interaction_types=None, temporal_features=None):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        features = [user_emb, item_emb]
        
        # Add behavioral features if available
        if interaction_types is not None:
            behavior_emb = self.user_behavior_embedding(interaction_types)
            features.append(behavior_emb)
        
        # Add temporal features if available
        if temporal_features is not None:
            temporal_emb = self.temporal_embedding(temporal_features)
            features.append(temporal_emb)
        
        # Concatenate all features
        x = torch.cat(features, dim=1)
        
        # Pass through MLP
        output = self.mlp(x)
        return torch.sigmoid(output)

# Enhanced Content-Based Recommendation Model
class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.genre_vectorizer = TfidfVectorizer()
        self.language_encoder = {}
        self.scaler = StandardScaler()
        self.content_matrix = None
        self.similarity_matrix = None
        self.user_content_profiles = {}
        
    def prepare_content_features(self, content_data):
        """Prepare enhanced content features for similarity calculation"""
        try:
            # Text features (overview)
            overview_features = self.tfidf_vectorizer.fit_transform(
                content_data['overview'].fillna('')
            )
            
            # Genre features
            genre_text = content_data['genres'].apply(
                lambda x: ' '.join(json.loads(x) if isinstance(x, str) else [])
            )
            genre_features = self.genre_vectorizer.fit_transform(genre_text)
            
            # Language encoding
            unique_languages = set()
            for lang_list in content_data['languages']:
                if isinstance(lang_list, str):
                    try:
                        langs = json.loads(lang_list)
                        unique_languages.update(langs)
                    except:
                        pass
            
            self.language_encoder = {lang: idx for idx, lang in enumerate(unique_languages)}
            
            # Enhanced numerical features
            numerical_cols = ['rating', 'popularity', 'vote_count']
            available_cols = [col for col in numerical_cols if col in content_data.columns]
            
            if available_cols:
                numerical_features = content_data[available_cols].fillna(0)
                
                # Add derived features
                content_data_enhanced = content_data.copy()
                content_data_enhanced['rating_popularity_score'] = (
                    content_data_enhanced.get('rating', 0) * 
                    np.log1p(content_data_enhanced.get('popularity', 1))
                )
                content_data_enhanced['quality_score'] = (
                    content_data_enhanced.get('rating', 0) * 
                    np.log1p(content_data_enhanced.get('vote_count', 1))
                )
                
                # Add these derived features
                derived_features = content_data_enhanced[['rating_popularity_score', 'quality_score']].fillna(0)
                numerical_features = pd.concat([numerical_features, derived_features], axis=1)
                
                numerical_features = self.scaler.fit_transform(numerical_features)
            else:
                numerical_features = np.zeros((len(content_data), 2))
            
            # Combine features
            self.content_matrix = sp.hstack([
                overview_features,
                genre_features,
                sp.csr_matrix(numerical_features)
            ])
            
            # Calculate similarity matrix
            self.similarity_matrix = cosine_similarity(self.content_matrix)
            
            return True
        except Exception as e:
            logger.error(f"Error preparing content features: {e}")
            return False
    
    def build_user_content_profile(self, user_id, user_interactions, content_data):
        """Build user's content preference profile"""
        try:
            # Get user's interaction history
            user_content_vector = np.zeros(self.content_matrix.shape[1])
            total_weight = 0
            
            for interaction in user_interactions:
                content_id = interaction['content_id']
                content_idx = content_data.index[content_data['id'] == content_id].tolist()
                
                if content_idx:
                    content_idx = content_idx[0]
                    
                    # Get interaction weight
                    interaction_weight = INTERACTION_WEIGHTS.get(interaction['interaction_type'], 0.3)
                    
                    # Apply temporal decay
                    try:
                        interaction_time = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
                        days_ago = (datetime.now() - interaction_time).days
                        temporal_weight = self._get_temporal_weight(days_ago)
                    except:
                        temporal_weight = 0.5
                    
                    # Apply rating boost
                    rating_boost = 1.0
                    if interaction.get('rating'):
                        rating_boost = interaction['rating'] / 10.0
                    
                    final_weight = interaction_weight * temporal_weight * rating_boost
                    
                    # Add to user profile
                    content_vector = self.content_matrix[content_idx].toarray().flatten()
                    user_content_vector += content_vector * final_weight
                    total_weight += final_weight
            
            # Normalize
            if total_weight > 0:
                user_content_vector = user_content_vector / total_weight
            
            self.user_content_profiles[user_id] = user_content_vector
            return user_content_vector
            
        except Exception as e:
            logger.error(f"Error building user content profile: {e}")
            return None
    
    def get_content_recommendations(self, content_id, content_data, n_recommendations=20):
        """Get content-based recommendations"""
        try:
            if self.similarity_matrix is None:
                return []
            
            content_idx = content_data.index[content_data['id'] == content_id].tolist()
            if not content_idx:
                return []
            
            content_idx = content_idx[0]
            similarity_scores = self.similarity_matrix[content_idx]
            
            # Get top similar items
            similar_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations+1]
            
            recommendations = []
            for idx in similar_indices:
                recommendations.append({
                    'content_id': int(content_data.iloc[idx]['id']),
                    'score': float(similarity_scores[idx]),
                    'reason': 'Content similarity based on features and metadata'
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting content recommendations: {e}")
            return []
    
    def get_user_based_content_recommendations(self, user_id, content_data, n_recommendations=20):
        """Get recommendations based on user's content profile"""
        try:
            if user_id not in self.user_content_profiles:
                return []
            
            user_profile = self.user_content_profiles[user_id]
            
            # Calculate similarity with all content
            content_scores = cosine_similarity([user_profile], self.content_matrix).flatten()
            
            # Get top recommendations
            top_indices = np.argsort(content_scores)[::-1][:n_recommendations]
            
            recommendations = []
            for idx in top_indices:
                content_id = content_data.iloc[idx]['id']
                recommendations.append({
                    'content_id': int(content_id),
                    'score': float(content_scores[idx]),
                    'reason': 'Personalized content matching your preferences'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting user-based content recommendations: {e}")
            return []
    
    def _get_temporal_weight(self, days_ago):
        """Calculate temporal weight based on recency"""
        if days_ago <= 7:
            return TEMPORAL_DECAY['recent']
        elif days_ago <= 30:
            return TEMPORAL_DECAY['week']
        elif days_ago <= 90:
            return TEMPORAL_DECAY['month']
        elif days_ago <= 180:
            return TEMPORAL_DECAY['quarter']
        else:
            return TEMPORAL_DECAY['old']

# Enhanced Collaborative Filtering Model
class CollaborativeFilteringRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.als_model = None
        self.user_factors = None
        self.item_factors = None
        self.user_profiles = {}
        self.item_profiles = {}
        
    def prepare_user_item_matrix(self, interactions_data):
        """Prepare enhanced user-item interaction matrix with multiple interaction types"""
        try:
            # Create weighted interaction scores
            interaction_scores = []
            
            for _, interaction in interactions_data.iterrows():
                user_id = interaction['user_id']
                content_id = interaction['content_id']
                interaction_type = interaction['interaction_type']
                rating = interaction.get('rating', 0)
                
                # Calculate interaction score
                base_weight = INTERACTION_WEIGHTS.get(interaction_type, 0.3)
                
                # Apply temporal decay
                try:
                    interaction_time = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
                    days_ago = (datetime.now() - interaction_time).days
                    temporal_weight = self._get_temporal_weight(days_ago)
                except:
                    temporal_weight = 0.5
                
                # Calculate final score
                if rating > 0:
                    final_score = (rating / 10.0) * base_weight * temporal_weight
                else:
                    final_score = base_weight * temporal_weight
                
                interaction_scores.append({
                    'user_id': user_id,
                    'content_id': content_id,
                    'score': final_score
                })
            
            # Create DataFrame and pivot
            scores_df = pd.DataFrame(interaction_scores)
            
            # Group by user and item, taking the maximum score
            scores_df = scores_df.groupby(['user_id', 'content_id'])['score'].max().reset_index()
            
            # Create user-item matrix
            user_item_df = scores_df.pivot_table(
                index='user_id',
                columns='content_id',
                values='score',
                fill_value=0
            )
            
            self.user_item_matrix = sp.csr_matrix(user_item_df.values)
            self.user_ids = user_item_df.index.tolist()
            self.item_ids = user_item_df.columns.tolist()
            
            # Train enhanced ALS model
            self.als_model = implicit.als.AlternatingLeastSquares(
                factors=128,  # Increased factors
                regularization=0.1,
                iterations=30,  # More iterations
                alpha=1.0,
                use_gpu=False
            )
            
            # Convert to proper format for implicit
            user_item_csr = self.user_item_matrix.T  # Transpose for implicit
            self.als_model.fit(user_item_csr)
            
            self.user_factors = self.als_model.user_factors
            self.item_factors = self.als_model.item_factors
            
            return True
        except Exception as e:
            logger.error(f"Error preparing user-item matrix: {e}")
            return False
    
    def get_user_based_recommendations(self, user_id, n_recommendations=20):
        """Get enhanced user-based collaborative filtering recommendations"""
        try:
            if user_id not in self.user_ids:
                return []
            
            user_idx = self.user_ids.index(user_id)
            
            # Get recommendations from ALS model
            recommended_items = self.als_model.recommend(
                user_idx,
                self.user_item_matrix[user_idx],
                N=n_recommendations * 2,  # Get more to allow for filtering
                filter_already_liked_items=True
            )
            
            recommendations = []
            for item_idx, score in recommended_items:
                if len(recommendations) >= n_recommendations:
                    break
                    
                content_id = self.item_ids[item_idx]
                recommendations.append({
                    'content_id': content_id,
                    'score': float(score),
                    'reason': 'Users with similar tastes also liked this'
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting user-based recommendations: {e}")
            return []
    
    def get_item_based_recommendations(self, content_id, n_recommendations=20):
        """Get enhanced item-based collaborative filtering recommendations"""
        try:
            if content_id not in self.item_ids:
                return []
            
            item_idx = self.item_ids.index(content_id)
            
            # Get similar items from ALS model
            similar_items = self.als_model.similar_items(
                item_idx,
                N=n_recommendations
            )
            
            recommendations = []
            for similar_item_idx, score in similar_items:
                similar_content_id = self.item_ids[similar_item_idx]
                recommendations.append({
                    'content_id': similar_content_id,
                    'score': float(score),
                    'reason': 'Similar to content you liked before'
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting item-based recommendations: {e}")
            return []
    
    def get_user_similarity_recommendations(self, user_id, interactions_data, n_recommendations=20):
        """Get recommendations based on user similarity in behavior patterns"""
        try:
            if user_id not in self.user_ids:
                return []
            
            user_idx = self.user_ids.index(user_id)
            user_vector = self.user_factors[user_idx]
            
            # Find similar users
            user_similarities = cosine_similarity([user_vector], self.user_factors).flatten()
            similar_user_indices = np.argsort(user_similarities)[::-1][1:11]  # Top 10 similar users
            
            # Get content liked by similar users
            content_scores = defaultdict(float)
            
            for similar_user_idx in similar_user_indices:
                similar_user_id = self.user_ids[similar_user_idx]
                similarity_score = user_similarities[similar_user_idx]
                
                # Get highly rated content by similar user
                user_interactions = interactions_data[interactions_data['user_id'] == similar_user_id]
                
                for _, interaction in user_interactions.iterrows():
                    content_id = interaction['content_id']
                    interaction_weight = INTERACTION_WEIGHTS.get(interaction['interaction_type'], 0.3)
                    
                    score = interaction_weight * similarity_score
                    if interaction.get('rating'):
                        score *= (interaction['rating'] / 10.0)
                    
                    content_scores[content_id] += score
            
            # Sort and return recommendations
            sorted_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for content_id, score in sorted_content[:n_recommendations]:
                recommendations.append({
                    'content_id': content_id,
                    'score': float(score),
                    'reason': 'Recommended based on users with similar preferences'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting user similarity recommendations: {e}")
            return []
    
    def _get_temporal_weight(self, days_ago):
        """Calculate temporal weight based on recency"""
        if days_ago <= 7:
            return TEMPORAL_DECAY['recent']
        elif days_ago <= 30:
            return TEMPORAL_DECAY['week']
        elif days_ago <= 90:
            return TEMPORAL_DECAY['month']
        elif days_ago <= 180:
            return TEMPORAL_DECAY['quarter']
        else:
            return TEMPORAL_DECAY['old']

# SVD-based Recommender using Surprise (Enhanced)
class SVDRecommender:
    def __init__(self):
        self.model = SVD(n_factors=150, reg_all=0.05, lr_all=0.005, n_epochs=100)
        self.trainset = None
        self.is_trained = False
        self.user_bias = {}
        self.item_bias = {}
        
    def train(self, interactions_data):
        """Train enhanced SVD model with weighted ratings"""
        try:
            # Prepare weighted ratings
            enhanced_ratings = []
            
            for _, interaction in interactions_data.iterrows():
                user_id = interaction['user_id']
                content_id = interaction['content_id']
                
                # Calculate enhanced rating
                base_rating = interaction.get('rating', 5.0)
                interaction_weight = INTERACTION_WEIGHTS.get(interaction['interaction_type'], 0.3)
                
                # Apply temporal decay
                try:
                    interaction_time = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
                    days_ago = (datetime.now() - interaction_time).days
                    temporal_weight = self._get_temporal_weight(days_ago)
                except:
                    temporal_weight = 0.5
                
                # Calculate final rating (scale to 1-10)
                if base_rating > 0:
                    final_rating = base_rating * interaction_weight * temporal_weight
                else:
                    final_rating = 5.0 * interaction_weight * temporal_weight
                
                # Ensure rating is in valid range
                final_rating = max(1.0, min(10.0, final_rating))
                
                enhanced_ratings.append({
                    'user_id': user_id,
                    'content_id': content_id,
                    'rating': final_rating
                })
            
            # Create DataFrame
            df = pd.DataFrame(enhanced_ratings)
            
            # Group by user-item and take mean rating
            df = df.groupby(['user_id', 'content_id'])['rating'].mean().reset_index()
            
            reader = Reader(rating_scale=(1, 10))
            dataset = Dataset.load_from_df(df, reader)
            
            self.trainset = dataset.build_full_trainset()
            self.model.fit(self.trainset)
            self.is_trained = True
            
            return True
        except Exception as e:
            logger.error(f"Error training SVD model: {e}")
            return False
    
    def get_user_recommendations(self, user_id, content_ids, n_recommendations=20):
        """Get SVD-based recommendations for user"""
        try:
            if not self.is_trained:
                return []
            
            predictions = []
            for content_id in content_ids:
                try:
                    pred = self.model.predict(user_id, content_id)
                    predictions.append((content_id, pred.est))
                except:
                    continue
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for content_id, score in predictions[:n_recommendations]:
                recommendations.append({
                    'content_id': content_id,
                    'score': float(score),
                    'reason': 'Predicted to match your rating preferences'
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting SVD recommendations: {e}")
            return []
    
    def _get_temporal_weight(self, days_ago):
        """Calculate temporal weight based on recency"""
        if days_ago <= 7:
            return TEMPORAL_DECAY['recent']
        elif days_ago <= 30:
            return TEMPORAL_DECAY['week']
        elif days_ago <= 90:
            return TEMPORAL_DECAY['month']
        elif days_ago <= 180:
            return TEMPORAL_DECAY['quarter']
        else:
            return TEMPORAL_DECAY['old']

# Neural Network Recommender using PyTorch (Enhanced)
class NeuralRecommender:
    def __init__(self):
        self.model = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.feature_scaler = StandardScaler()
        
    def prepare_data(self, interactions_data):
        """Prepare enhanced data for neural network training"""
        try:
            # Create ID mappings
            unique_users = interactions_data['user_id'].unique()
            unique_items = interactions_data['content_id'].unique()
            
            self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
            self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
            
            # Prepare training data with additional features
            users = []
            items = []
            ratings = []
            interaction_types = []
            temporal_features = []
            
            for _, interaction in interactions_data.iterrows():
                user_id = interaction['user_id']
                content_id = interaction['content_id']
                
                if user_id in self.user_id_map and content_id in self.item_id_map:
                    users.append(self.user_id_map[user_id])
                    items.append(self.item_id_map[content_id])
                    
                    # Enhanced rating calculation
                    base_rating = interaction.get('rating', 5.0)
                    interaction_weight = INTERACTION_WEIGHTS.get(interaction['interaction_type'], 0.3)
                    
                    final_rating = base_rating * interaction_weight
                    ratings.append(final_rating)
                    
                    # Interaction type encoding
                    interaction_type_map = {
                        'search': 0, 'view': 1, 'like': 2, 'favorite': 3, 
                        'watchlist': 4, 'rating': 5
                    }
                    interaction_types.append(
                        interaction_type_map.get(interaction['interaction_type'], 0)
                    )
                    
                    # Temporal features (day of week)
                    try:
                        interaction_time = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
                        day_of_week = interaction_time.weekday()
                        temporal_features.append(day_of_week)
                    except:
                        temporal_features.append(0)
            
            # Normalize ratings to 0-1
            ratings = np.array(ratings)
            if len(ratings) > 0:
                ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-8)
            
            return users, items, ratings, interaction_types, temporal_features
        except Exception as e:
            logger.error(f"Error preparing neural network data: {e}")
            return None, None, None, None, None
    
    def train(self, interactions_data, epochs=100):
        """Train enhanced neural collaborative filtering model"""
        try:
            users, items, ratings, interaction_types, temporal_features = self.prepare_data(interactions_data)
            if users is None:
                return False
            
            # Initialize model
            num_users = len(self.user_id_map)
            num_items = len(self.item_id_map)
            
            self.model = NeuralCollaborativeFiltering(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=128,
                hidden_dims=[256, 128, 64, 32]
            ).to(self.device)
            
            # Prepare data loaders
            users_tensor = torch.LongTensor(users).to(self.device)
            items_tensor = torch.LongTensor(items).to(self.device)
            ratings_tensor = torch.FloatTensor(ratings).to(self.device)
            interaction_types_tensor = torch.LongTensor(interaction_types).to(self.device)
            temporal_tensor = torch.LongTensor(temporal_features).to(self.device)
            
            # Training with advanced features
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            
            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                predictions = self.model(
                    users_tensor, items_tensor, 
                    interaction_types_tensor, temporal_tensor
                ).squeeze()
                
                loss = criterion(predictions, ratings_tensor)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                if epoch % 20 == 0:
                    logger.info(f"Neural model training epoch {epoch}, loss: {loss.item():.4f}")
            
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Error training neural model: {e}")
            return False
    
    def get_user_recommendations(self, user_id, content_ids, n_recommendations=20):
        """Get enhanced neural network recommendations"""
        try:
            if not self.is_trained or user_id not in self.user_id_map:
                return []
            
            self.model.eval()
            user_idx = self.user_id_map[user_id]
            
            predictions = []
            with torch.no_grad():
                for content_id in content_ids:
                    if content_id in self.item_id_map:
                        item_idx = self.item_id_map[content_id]
                        
                        user_tensor = torch.LongTensor([user_idx]).to(self.device)
                        item_tensor = torch.LongTensor([item_idx]).to(self.device)
                        
                        # Use default values for additional features
                        interaction_type_tensor = torch.LongTensor([5]).to(self.device)  # rating type
                        temporal_tensor = torch.LongTensor([0]).to(self.device)  # Monday
                        
                        pred = self.model(
                            user_tensor, item_tensor, 
                            interaction_type_tensor, temporal_tensor
                        ).item()
                        
                        predictions.append((content_id, pred))
            
            # Sort by prediction score
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for content_id, score in predictions[:n_recommendations]:
                recommendations.append({
                    'content_id': content_id,
                    'score': float(score),
                    'reason': 'Deep learning analysis of your preferences'
                })
            
            return recommendations
        except Exception as e:
            logger.error(f"Error getting neural recommendations: {e}")
            return []

# Enhanced Hybrid Recommendation Engine
class HybridRecommendationEngine:
    def __init__(self):
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.svd_recommender = SVDRecommender()
        self.neural_recommender = NeuralRecommender()
        self.user_profile_builder = UserProfileBuilder()
        self.search_analyzer = SearchPatternAnalyzer()
        self.is_initialized = False
        
        # Enhanced weight configuration
        self.algorithm_weights = {
            'content': 0.20,
            'content_user_profile': 0.15,
            'collaborative': 0.25,
            'user_similarity': 0.10,
            'svd': 0.15,
            'neural': 0.15
        }
        
        # Behavioral pattern weights
        self.behavior_weights = {
            'search_based': 0.15,
            'favorite_based': 0.25,
            'watchlist_based': 0.20,
            'rating_based': 0.25,
            'view_based': 0.15
        }
        
    def initialize_models(self, content_data, interactions_data):
        """Initialize all recommendation models with enhanced features"""
        try:
            logger.info("Initializing enhanced hybrid recommendation models...")
            
            # Initialize content-based model
            if not content_data.empty:
                self.content_recommender.prepare_content_features(content_data)
                logger.info("Content-based model initialized")
            
            # Initialize collaborative filtering model
            if not interactions_data.empty and len(interactions_data) >= MIN_INTERACTIONS:
                self.collaborative_recommender.prepare_user_item_matrix(interactions_data)
                logger.info("Collaborative filtering model initialized")
                
                # Train SVD model
                self.svd_recommender.train(interactions_data)
                logger.info("SVD model trained")
                
                # Train neural model (with optimized sample size)
                if len(interactions_data) >= 100:
                    sample_size = min(len(interactions_data), 20000)
                    sample_data = interactions_data.sample(n=sample_size)
                    self.neural_recommender.train(sample_data, epochs=50)
                    logger.info("Neural model trained")
            
            self.is_initialized = True
            logger.info("All enhanced models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False
    
    def get_hybrid_recommendations(self, user_id, user_preferences, content_data, 
                                 interactions_data, n_recommendations=20):
        """Get enhanced hybrid recommendations with deep user understanding"""
        try:
            # Build comprehensive user profile
            user_interactions = [
                interaction for interaction in interactions_data.to_dict('records') 
                if interaction['user_id'] == user_id
            ]
            
            user_profile = self.user_profile_builder.build_user_profile(
                user_id, user_interactions, content_data
            )
            
            # Extract search history (if available)
            search_history = [
                interaction for interaction in user_interactions 
                if interaction['interaction_type'] == 'search'
            ]
            
            search_patterns = self.search_analyzer.analyze_search_patterns(
                user_id, search_history, content_data
            )
            
            # Get all available content IDs
            content_ids = content_data['id'].tolist()
            
            # Collect recommendations from all algorithms
            all_recommendations = defaultdict(list)
            
            # 1. Content-based recommendations
            if hasattr(self.content_recommender, 'similarity_matrix'):
                # Build user content profile
                self.content_recommender.build_user_content_profile(
                    user_id, user_interactions, content_data
                )
                
                # Get user-based content recommendations
                content_user_recs = self.content_recommender.get_user_based_content_recommendations(
                    user_id, content_data, n_recommendations
                )
                all_recommendations['content_user_profile'] = content_user_recs
                
                # Get item-based content recommendations from user's favorites
                favorite_content_ids = [
                    interaction['content_id'] for interaction in user_interactions 
                    if interaction['interaction_type'] == 'favorite'
                ]
                
                for content_id in favorite_content_ids[-3:]:  # Last 3 favorites
                    content_recs = self.content_recommender.get_content_recommendations(
                        content_id, content_data, n_recommendations//3
                    )
                    all_recommendations['content'].extend(content_recs)
            
            # 2. Collaborative filtering recommendations
            if self.collaborative_recommender.user_item_matrix is not None:
                # User-based collaborative filtering
                collab_recs = self.collaborative_recommender.get_user_based_recommendations(
                    user_id, n_recommendations
                )
                all_recommendations['collaborative'] = collab_recs
                
                # User similarity-based recommendations
                user_sim_recs = self.collaborative_recommender.get_user_similarity_recommendations(
                    user_id, interactions_data, n_recommendations
                )
                all_recommendations['user_similarity'] = user_sim_recs
            
            # 3. SVD recommendations
            if self.svd_recommender.is_trained:
                svd_recs = self.svd_recommender.get_user_recommendations(
                    user_id, content_ids, n_recommendations
                )
                all_recommendations['svd'] = svd_recs
            
            # 4. Neural recommendations
            if self.neural_recommender.is_trained:
                neural_recs = self.neural_recommender.get_user_recommendations(
                    user_id, content_ids, n_recommendations
                )
                all_recommendations['neural'] = neural_recs
            
            # 5. Behavior-based recommendations
            behavior_recs = self._get_behavior_based_recommendations(
                user_profile, user_interactions, content_data, n_recommendations
            )
            all_recommendations.update(behavior_recs)
            
            # Combine all recommendations with sophisticated weighting
            final_scores = self._combine_recommendations_advanced(
                all_recommendations, user_profile, search_patterns
            )
            
            # Apply user preferences boost
            final_scores = self._apply_advanced_preference_boost(
                final_scores, user_preferences, user_profile, content_data
            )
            
            # Apply diversity and serendipity
            final_recommendations = self._advanced_diversify_recommendations(
                final_scores, content_data, user_preferences, user_profile, n_recommendations
            )
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error generating hybrid recommendations: {e}")
            return []
    
    def _get_behavior_based_recommendations(self, user_profile, user_interactions, content_data, n_recommendations):
        """Get recommendations based on specific user behaviors"""
        behavior_recs = defaultdict(list)
        
        try:
            # Search-based recommendations
            search_interactions = [i for i in user_interactions if i['interaction_type'] == 'search']
            if search_interactions:
                behavior_recs['search_based'] = self._get_search_based_recommendations(
                    search_interactions, content_data, n_recommendations//4
                )
            
            # Favorite-based recommendations
            favorite_interactions = [i for i in user_interactions if i['interaction_type'] == 'favorite']
            if favorite_interactions:
                behavior_recs['favorite_based'] = self._get_favorite_based_recommendations(
                    favorite_interactions, content_data, n_recommendations//4
                )
            
            # Watchlist-based recommendations
            watchlist_interactions = [i for i in user_interactions if i['interaction_type'] == 'watchlist']
            if watchlist_interactions:
                behavior_recs['watchlist_based'] = self._get_watchlist_based_recommendations(
                    watchlist_interactions, content_data, n_recommendations//4
                )
            
            # Rating-based recommendations
            rating_interactions = [i for i in user_interactions if i.get('rating', 0) > 0]
            if rating_interactions:
                behavior_recs['rating_based'] = self._get_rating_based_recommendations(
                    rating_interactions, content_data, n_recommendations//4
                )
            
        except Exception as e:
            logger.error(f"Error getting behavior-based recommendations: {e}")
        
        return behavior_recs
    
    def _get_search_based_recommendations(self, search_interactions, content_data, n_recommendations):
        """Get recommendations based on search patterns"""
        recommendations = []
        try:
            # Extract search terms and find related content
            search_terms = []
            for interaction in search_interactions[-10:]:  # Last 10 searches
                # In real implementation, you'd extract the search query
                # For now, we'll use a simplified approach
                search_terms.append(f"search_{interaction['content_id']}")
            
            # Find content matching search patterns
            # This is a simplified implementation
            for content_id in content_data['id'].sample(min(n_recommendations, len(content_data))):
                recommendations.append({
                    'content_id': int(content_id),
                    'score': 0.7,
                    'reason': 'Based on your search patterns'
                })
                
        except Exception as e:
            logger.error(f"Error getting search-based recommendations: {e}")
        
        return recommendations
    
    def _get_favorite_based_recommendations(self, favorite_interactions, content_data, n_recommendations):
        """Get recommendations based on favorites patterns"""
        recommendations = []
        try:
            # Analyze favorite content patterns
            favorite_content_ids = [i['content_id'] for i in favorite_interactions]
            
            # Get content similar to favorites using content-based filtering
            if hasattr(self.content_recommender, 'similarity_matrix'):
                for content_id in favorite_content_ids[-5:]:  # Last 5 favorites
                    similar_recs = self.content_recommender.get_content_recommendations(
                        content_id, content_data, n_recommendations//5
                    )
                    
                    for rec in similar_recs:
                        rec['reason'] = 'Similar to your favorites'
                        rec['score'] *= 1.2  # Boost for favorite-based
                    
                    recommendations.extend(similar_recs)
                    
        except Exception as e:
            logger.error(f"Error getting favorite-based recommendations: {e}")
        
        return recommendations[:n_recommendations]
    
    def _get_watchlist_based_recommendations(self, watchlist_interactions, content_data, n_recommendations):
        """Get recommendations based on watchlist patterns"""
        recommendations = []
        try:
            # Analyze watchlist content patterns
            watchlist_content_ids = [i['content_id'] for i in watchlist_interactions]
            
            # Get content similar to watchlist items
            if hasattr(self.content_recommender, 'similarity_matrix'):
                for content_id in watchlist_content_ids[-3:]:  # Last 3 watchlist items
                    similar_recs = self.content_recommender.get_content_recommendations(
                        content_id, content_data, n_recommendations//3
                    )
                    
                    for rec in similar_recs:
                        rec['reason'] = 'Similar to your watchlist'
                        rec['score'] *= 1.1  # Moderate boost for watchlist-based
                    
                    recommendations.extend(similar_recs)
                    
        except Exception as e:
            logger.error(f"Error getting watchlist-based recommendations: {e}")
        
        return recommendations[:n_recommendations]
    
    def _get_rating_based_recommendations(self, rating_interactions, content_data, n_recommendations):
        """Get recommendations based on rating patterns"""
        recommendations = []
        try:
            # Analyze high-rated content
            high_rated = [i for i in rating_interactions if i['rating'] >= 8]
            
            if high_rated:
                high_rated_content_ids = [i['content_id'] for i in high_rated]
                
                # Get content similar to highly rated items
                if hasattr(self.content_recommender, 'similarity_matrix'):
                    for content_id in high_rated_content_ids[-5:]:
                        similar_recs = self.content_recommender.get_content_recommendations(
                            content_id, content_data, n_recommendations//5
                        )
                        
                        for rec in similar_recs:
                            rec['reason'] = 'Similar to your highly rated content'
                            rec['score'] *= 1.3  # High boost for rating-based
                        
                        recommendations.extend(similar_recs)
                        
        except Exception as e:
            logger.error(f"Error getting rating-based recommendations: {e}")
        
        return recommendations[:n_recommendations]
    
    def _combine_recommendations_advanced(self, all_recommendations, user_profile, search_patterns):
        """Advanced combination of recommendations with dynamic weighting"""
        final_scores = defaultdict(float)
        
        try:
            # Calculate dynamic weights based on user profile
            dynamic_weights = self._calculate_dynamic_weights(user_profile)
            
            # Combine algorithm-based recommendations
            for algorithm, recs in all_recommendations.items():
                if algorithm in self.algorithm_weights:
                    weight = self.algorithm_weights[algorithm] * dynamic_weights.get(algorithm, 1.0)
                elif algorithm in self.behavior_weights:
                    weight = self.behavior_weights[algorithm] * dynamic_weights.get(algorithm, 1.0)
                else:
                    weight = 0.1
                
                for rec in recs:
                    content_id = rec['content_id']
                    score = rec['score'] * weight
                    final_scores[content_id] += score
            
            return final_scores
            
        except Exception as e:
            logger.error(f"Error combining recommendations: {e}")
            return final_scores
    
    def _calculate_dynamic_weights(self, user_profile):
        """Calculate dynamic weights based on user behavior patterns"""
        weights = {}
        
        try:
            if user_profile:
                # Adjust weights based on user characteristics
                diversity_score = user_profile.get('diversity_score', 0.5)
                exploration_tendency = user_profile.get('exploration_tendency', 0.5)
                quality_preference = user_profile.get('quality_preference', 0.7)
                
                # Users who like diversity get more content-based recommendations
                if diversity_score > 0.7:
                    weights['content'] = 1.2
                    weights['content_user_profile'] = 1.3
                
                # Users who explore get more neural/advanced recommendations
                if exploration_tendency > 0.6:
                    weights['neural'] = 1.3
                    weights['search_based'] = 1.4
                
                # Quality-focused users get more collaborative filtering
                if quality_preference > 0.8:
                    weights['collaborative'] = 1.2
                    weights['rating_based'] = 1.3
                
                # Conservative users get more SVD recommendations
                if exploration_tendency < 0.3:
                    weights['svd'] = 1.2
                    weights['favorite_based'] = 1.3
            
        except Exception as e:
            logger.error(f"Error calculating dynamic weights: {e}")
        
        return weights
    
    def _apply_advanced_preference_boost(self, scores, user_preferences, user_profile, content_data):
        """Apply advanced preference boosting"""
        try:
            boosted_scores = scores.copy()
            
            preferred_languages = user_preferences.get('preferred_languages', [])
            preferred_genres = user_preferences.get('preferred_genres', [])
            
            # Get dynamic preferences from user profile
            if user_profile:
                profile_genres = dict(user_profile.get('genre_scores', {}))
                profile_languages = dict(user_profile.get('language_scores', {}))
                
                # Merge explicit and implicit preferences
                all_preferred_genres = set(preferred_genres)
                all_preferred_genres.update(
                    [genre for genre, score in profile_genres.items() if score > 0.3]
                )
                
                all_preferred_languages = set(preferred_languages)
                all_preferred_languages.update(
                    [lang for lang, score in profile_languages.items() if score > 0.3]
                )
            else:
                all_preferred_genres = set(preferred_genres)
                all_preferred_languages = set(preferred_languages)
            
            for content_id, score in scores.items():
                content_row = content_data[content_data['id'] == content_id]
                if content_row.empty:
                    continue
                
                content_info = content_row.iloc[0]
                boost_factor = 1.0
                
                # Advanced language preference boost
                try:
                    content_languages = json.loads(content_info['languages'])
                    language_match = len(set(content_languages) & all_preferred_languages)
                    if language_match > 0:
                        boost_factor *= (1.0 + 0.2 * language_match)
                except:
                    pass
                
                # Advanced genre preference boost
                try:
                    content_genres = json.loads(content_info['genres'])
                    genre_match = len(set(content_genres) & all_preferred_genres)
                    if genre_match > 0:
                        boost_factor *= (1.0 + 0.15 * genre_match)
                except:
                    pass
                
                # Quality boost based on user preference
                if user_profile and user_profile.get('quality_preference', 0.7) > 0.8:
                    try:
                        rating = content_info.get('rating', 0)
                        if rating >= 8.5:
                            boost_factor *= 1.3
                        elif rating >= 7.5:
                            boost_factor *= 1.15
                    except:
                        pass
                
                # Diversity boost for diverse users
                if user_profile and user_profile.get('diversity_score', 0.5) > 0.7:
                    # Boost less popular but good content
                    try:
                        popularity = content_info.get('popularity', 50)
                        rating = content_info.get('rating', 0)
                        if popularity < 30 and rating > 7.0:
                            boost_factor *= 1.2
                    except:
                        pass
                
                boosted_scores[content_id] = score * boost_factor
            
            return boosted_scores
            
        except Exception as e:
            logger.error(f"Error applying advanced preference boost: {e}")
            return scores
    
    def _advanced_diversify_recommendations(self, scores, content_data, user_preferences, 
                                          user_profile, n_recommendations):
        """Advanced diversification with serendipity and exploration"""
        try:
            # Sort by score
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            final_recommendations = []
            used_genres = defaultdict(int)
            used_languages = defaultdict(int)
            used_content_types = defaultdict(int)
            
            # Dynamic diversity constraints based on user profile
            if user_profile:
                diversity_score = user_profile.get('diversity_score', 0.5)
                exploration_tendency = user_profile.get('exploration_tendency', 0.5)
            else:
                diversity_score = 0.5
                exploration_tendency = 0.5
            
            # Adjust constraints based on user characteristics
            max_per_genre = max(2, int(n_recommendations * (0.4 - diversity_score * 0.2)))
            max_per_language = max(3, int(n_recommendations * (0.6 - diversity_score * 0.2)))
            max_per_content_type = max(4, int(n_recommendations * 0.7))
            
            # Add serendipity for exploratory users
            serendipity_count = int(n_recommendations * exploration_tendency * 0.3)
            regular_count = n_recommendations - serendipity_count
            
            # Fill regular recommendations first
            for content_id, score in sorted_items:
                if len(final_recommendations) >= regular_count:
                    break
                
                if self._check_diversity_constraints(
                    content_id, content_data, used_genres, used_languages, 
                    used_content_types, max_per_genre, max_per_language, max_per_content_type
                ):
                    recommendation = self._create_recommendation_object(
                        content_id, score, content_data, 'Advanced hybrid recommendation'
                    )
                    if recommendation:
                        final_recommendations.append(recommendation)
                        self._update_diversity_counters(
                            content_id, content_data, used_genres, used_languages, used_content_types
                        )
            
            # Add serendipitous recommendations for exploratory users
            if serendipity_count > 0:
                serendipity_recs = self._get_serendipitous_recommendations(
                    sorted_items, content_data, used_genres, used_languages, 
                    used_content_types, serendipity_count
                )
                final_recommendations.extend(serendipity_recs)
            
            return final_recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in advanced diversification: {e}")
            # Fallback to simple diversification
            return self._simple_diversify_fallback(scores, content_data, n_recommendations)
    
    def _check_diversity_constraints(self, content_id, content_data, used_genres, 
                                   used_languages, used_content_types, max_per_genre, 
                                   max_per_language, max_per_content_type):
        """Check if content meets diversity constraints"""
        try:
            content_row = content_data[content_data['id'] == content_id]
            if content_row.empty:
                return False
            
            content_info = content_row.iloc[0]
            
            # Check genre constraints
            try:
                content_genres = json.loads(content_info['genres'])
                main_genre = content_genres[0] if content_genres else 'unknown'
                if used_genres[main_genre] >= max_per_genre:
                    return False
            except:
                pass
            
            # Check language constraints
            try:
                content_languages = json.loads(content_info['languages'])
                main_language = content_languages[0] if content_languages else 'unknown'
                if used_languages[main_language] >= max_per_language:
                    return False
            except:
                pass
            
            # Check content type constraints
            content_type = content_info.get('content_type', 'unknown')
            if used_content_types[content_type] >= max_per_content_type:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking diversity constraints: {e}")
            return True
    
    def _update_diversity_counters(self, content_id, content_data, used_genres, 
                                 used_languages, used_content_types):
        """Update diversity counters after adding a recommendation"""
        try:
            content_row = content_data[content_data['id'] == content_id]
            if content_row.empty:
                return
            
            content_info = content_row.iloc[0]
            
            # Update genre counter
            try:
                content_genres = json.loads(content_info['genres'])
                main_genre = content_genres[0] if content_genres else 'unknown'
                used_genres[main_genre] += 1
            except:
                pass
            
            # Update language counter
            try:
                content_languages = json.loads(content_info['languages'])
                main_language = content_languages[0] if content_languages else 'unknown'
                used_languages[main_language] += 1
            except:
                pass
            
            # Update content type counter
            content_type = content_info.get('content_type', 'unknown')
            used_content_types[content_type] += 1
            
        except Exception as e:
            logger.error(f"Error updating diversity counters: {e}")
    
    def _create_recommendation_object(self, content_id, score, content_data, reason):
        """Create a standardized recommendation object"""
        try:
            content_row = content_data[content_data['id'] == content_id]
            if content_row.empty:
                return None
            
            content_info = content_row.iloc[0]
            
            recommendation = {
                'content_id': int(content_id),
                'score': float(score),
                'reason': reason
            }
            
            # Add additional metadata
            try:
                recommendation['genres'] = json.loads(content_info.get('genres', '[]'))
                recommendation['languages'] = json.loads(content_info.get('languages', '[]'))
                recommendation['rating'] = content_info.get('rating')
                recommendation['content_type'] = content_info.get('content_type')
            except:
                pass
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error creating recommendation object: {e}")
            return None
    
    def _get_serendipitous_recommendations(self, sorted_items, content_data, used_genres, 
                                         used_languages, used_content_types, count):
        """Get serendipitous recommendations for exploration"""
        serendipity_recs = []
        
        try:
            # Look for good but less obvious recommendations
            for content_id, score in sorted_items[len(sorted_items)//3:]:  # Skip top third
                if len(serendipity_recs) >= count:
                    break
                
                content_row = content_data[content_data['id'] == content_id]
                if content_row.empty:
                    continue
                
                content_info = content_row.iloc[0]
                
                # Serendipity criteria: good rating but lower popularity/score
                try:
                    rating = content_info.get('rating', 0)
                    popularity = content_info.get('popularity', 50)
                    
                    if rating >= 7.0 and popularity < 70:  # Good but not super popular
                        recommendation = self._create_recommendation_object(
                            content_id, score, content_data, 
                            'Serendipitous discovery - hidden gem'
                        )
                        if recommendation:
                            serendipity_recs.append(recommendation)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error getting serendipitous recommendations: {e}")
        
        return serendipity_recs
    
    def _simple_diversify_fallback(self, scores, content_data, n_recommendations):
        """Simple fallback diversification method"""
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for content_id, score in sorted_items[:n_recommendations]:
            recommendation = {
                'content_id': int(content_id),
                'score': float(score),
                'reason': 'Hybrid recommendation'
            }
            recommendations.append(recommendation)
        
        return recommendations

# Cold Start Problem Handler (Enhanced)
class ColdStartHandler:
    def __init__(self):
        self.popularity_scores = {}
        self.trending_content = []
        self.genre_popularity = defaultdict(float)
        self.language_popularity = defaultdict(float)
        self.onboarding_recommendations = defaultdict(list)
        
    def update_popularity_scores(self, content_data, interactions_data):
        """Update popularity scores for cold start recommendations"""
        try:
            # Calculate enhanced popularity metrics
            interaction_counts = interactions_data['content_id'].value_counts()
            rating_weights = interactions_data.groupby('content_id')['rating'].mean()
            
            # Calculate recency weights
            current_time = datetime.now()
            recency_weights = {}
            
            for content_id in content_data['id']:
                recent_interactions = interactions_data[
                    (interactions_data['content_id'] == content_id) &
                    (interactions_data['timestamp'] >= (current_time - timedelta(days=30)).isoformat())
                ]
                recency_weights[content_id] = len(recent_interactions)
            
            # Calculate final popularity scores
            for _, content in content_data.iterrows():
                content_id = content['id']
                
                # Base metrics
                tmdb_popularity = content.get('popularity', 0)
                interaction_pop = interaction_counts.get(content_id, 0)
                rating_pop = rating_weights.get(content_id, content.get('rating', 5.0))
                recency_pop = recency_weights.get(content_id, 0)
                
                # Quality indicators
                vote_count = content.get('vote_count', 0)
                quality_score = rating_pop * np.log1p(vote_count)
                
                # Bonuses
                release_bonus = 1.5 if content.get('is_new_release') else 1.0
                critics_bonus = 1.3 if content.get('is_critics_choice') else 1.0
                trending_bonus = 1.4 if content.get('is_trending') else 1.0
                
                # Combined popularity score
                popularity = (
                    tmdb_popularity * 0.25 +
                    interaction_pop * 15 * 0.30 +
                    rating_pop * 10 * 0.25 +
                    recency_pop * 5 * 0.20
                ) * release_bonus * critics_bonus * trending_bonus + quality_score * 0.1
                
                self.popularity_scores[content_id] = popularity
            
            # Update trending content
            self.trending_content = sorted(
                self.popularity_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:200]  # Top 200 for better variety
            
            # Update genre and language popularity
            self._update_genre_language_popularity(content_data, interactions_data)
            
            logger.info(f"Updated popularity scores for {len(self.popularity_scores)} items")
            
        except Exception as e:
            logger.error(f"Error updating popularity scores: {e}")
    
    def _update_genre_language_popularity(self, content_data, interactions_data):
        """Update genre and language popularity metrics"""
        try:
            genre_interactions = defaultdict(int)
            language_interactions = defaultdict(int)
            
            for _, interaction in interactions_data.iterrows():
                content_id = interaction['content_id']
                content_row = content_data[content_data['id'] == content_id]
                
                if not content_row.empty:
                    content_info = content_row.iloc[0]
                    
                    # Count genre interactions
                    try:
                        genres = json.loads(content_info.get('genres', '[]'))
                        for genre in genres:
                            genre_interactions[genre] += 1
                    except:
                        pass
                    
                    # Count language interactions
                    try:
                        languages = json.loads(content_info.get('languages', '[]'))
                        for language in languages:
                            language_interactions[language] += 1
                    except:
                        pass
            
            # Normalize popularity scores
            max_genre_interactions = max(genre_interactions.values()) if genre_interactions else 1
            max_lang_interactions = max(language_interactions.values()) if language_interactions else 1
            
            for genre, count in genre_interactions.items():
                self.genre_popularity[genre] = count / max_genre_interactions
            
            for language, count in language_interactions.items():
                self.language_popularity[language] = count / max_lang_interactions
                
        except Exception as e:
            logger.error(f"Error updating genre/language popularity: {e}")
    
    def get_cold_start_recommendations(self, user_preferences, content_data, 
                                     n_recommendations=20):
        """Get enhanced recommendations for new users"""
        try:
            # Get user preferences
            preferred_languages = user_preferences.get('preferred_languages', [])
            preferred_genres = user_preferences.get('preferred_genres', [])
            
            # Create diverse recommendation pool
            recommendations_pool = []
            
            # 1. Popular content matching preferences (40%)
            pref_count = int(n_recommendations * 0.4)
            pref_recs = self._get_preference_based_popular_content(
                preferred_languages, preferred_genres, content_data, pref_count
            )
            recommendations_pool.extend(pref_recs)
            
            # 2. Trending content (30%)
            trending_count = int(n_recommendations * 0.3)
            trending_recs = self._get_trending_recommendations(
                content_data, trending_count
            )
            recommendations_pool.extend(trending_recs)
            
            # 3. High-quality diverse content (20%)
            quality_count = int(n_recommendations * 0.2)
            quality_recs = self._get_quality_diverse_content(
                content_data, quality_count
            )
            recommendations_pool.extend(quality_recs)
            
            # 4. Serendipitous discoveries (10%)
            serendipity_count = n_recommendations - len(recommendations_pool)
            if serendipity_count > 0:
                serendipity_recs = self._get_serendipitous_content(
                    content_data, serendipity_count
                )
                recommendations_pool.extend(serendipity_recs)
            
            # Remove duplicates and limit
            seen_ids = set()
            final_recommendations = []
            
            for rec in recommendations_pool:
                if rec['content_id'] not in seen_ids and len(final_recommendations) < n_recommendations:
                    seen_ids.add(rec['content_id'])
                    final_recommendations.append(rec)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error generating cold start recommendations: {e}")
            return []
    
    def _get_preference_based_popular_content(self, preferred_languages, preferred_genres, 
                                            content_data, count):
        """Get popular content matching user preferences"""
        recommendations = []
        
        try:
            scored_content = []
            
            for content_id, popularity_score in self.trending_content:
                content_row = content_data[content_data['id'] == content_id]
                if content_row.empty:
                    continue
                
                content_info = content_row.iloc[0]
                score = popularity_score
                
                # Apply preference bonuses
                try:
                    content_languages = json.loads(content_info['languages'])
                    language_match = len(set(content_languages) & set(preferred_languages))
                    if language_match > 0:
                        score *= (1.0 + language_match * 0.5)
                    
                    content_genres = json.loads(content_info['genres'])
                    genre_match = len(set(content_genres) & set(preferred_genres))
                    if genre_match > 0:
                        score *= (1.0 + genre_match * 0.4)
                        
                except:
                    pass
                
                scored_content.append((content_id, score, content_info))
            
            # Sort and select top items
            scored_content.sort(key=lambda x: x[1], reverse=True)
            
            for content_id, score, content_info in scored_content[:count]:
                recommendations.append({
                    'content_id': int(content_id),
                    'score': float(score),
                    'reason': 'Popular content matching your preferences'
                })
                
        except Exception as e:
            logger.error(f"Error getting preference-based popular content: {e}")
        
        return recommendations
    
    def _get_trending_recommendations(self, content_data, count):
        """Get currently trending content"""
        recommendations = []
        
        try:
            trending_content = [
                (content_id, score) for content_id, score in self.trending_content 
                if content_data[content_data['id'] == content_id].get('is_trending', False).any()
            ]
            
            for content_id, score in trending_content[:count]:
                recommendations.append({
                    'content_id': int(content_id),
                    'score': float(score),
                    'reason': 'Currently trending and popular'
                })
                
        except Exception as e:
            logger.error(f"Error getting trending recommendations: {e}")
        
        return recommendations
    
    def _get_quality_diverse_content(self, content_data, count):
        """Get high-quality content across diverse genres"""
        recommendations = []
        
        try:
            # Filter for high-quality content
            quality_content = content_data[
                (content_data['rating'] >= 7.5) & 
                (content_data['vote_count'] >= 100)
            ].copy()
            
            if quality_content.empty:
                return recommendations
            
            # Ensure genre diversity
            genre_counts = defaultdict(int)
            max_per_genre = max(1, count // 6)  # Max items per genre
            
            for _, content in quality_content.iterrows():
                if len(recommendations) >= count:
                    break
                
                try:
                    genres = json.loads(content['genres'])
                    main_genre = genres[0] if genres else 'unknown'
                    
                    if genre_counts[main_genre] < max_per_genre:
                        popularity_score = self.popularity_scores.get(content['id'], 50)
                        
                        recommendations.append({
                            'content_id': int(content['id']),
                            'score': float(popularity_score),
                            'reason': 'High-quality content for exploration'
                        })
                        
                        genre_counts[main_genre] += 1
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error getting quality diverse content: {e}")
        
        return recommendations
    
    def _get_serendipitous_content(self, content_data, count):
        """Get serendipitous hidden gems"""
        recommendations = []
        
        try:
            # Find hidden gems: good rating, lower popularity
            hidden_gems = content_data[
                (content_data['rating'] >= 7.0) & 
                (content_data['popularity'] < 50) &
                (content_data['vote_count'] >= 50)
            ].copy()
            
            if hidden_gems.empty:
                return recommendations
            
            # Sort by rating-to-popularity ratio
            hidden_gems['gem_score'] = (
                hidden_gems['rating'] * hidden_gems['vote_count'] / 
                (hidden_gems['popularity'] + 1)
            )
            
            hidden_gems = hidden_gems.sort_values('gem_score', ascending=False)
            
            for _, content in hidden_gems.head(count).iterrows():
                recommendations.append({
                    'content_id': int(content['id']),
                    'score': float(content['gem_score']),
                    'reason': 'Hidden gem - you might discover something amazing'
                })
                
        except Exception as e:
            logger.error(f"Error getting serendipitous content: {e}")
        
        return recommendations

# Location-based Recommendation Service (Enhanced)
class LocationBasedRecommender:
    def __init__(self):
        self.region_preferences = {
            'India': {
                'languages': ['hindi', 'telugu', 'tamil', 'kannada', 'malayalam'],
                'boost_factor': 1.5,
                'content_types': {'movie': 1.2, 'tv': 1.1}
            },
            'United States': {
                'languages': ['english'],
                'boost_factor': 1.2,
                'content_types': {'movie': 1.1, 'tv': 1.3}
            },
            'Japan': {
                'languages': ['japanese'],
                'boost_factor': 2.0,
                'content_types': {'anime': 2.5}
            }
        }
        
        self.time_zone_preferences = {
            'prime_time': ['18:00', '23:00'],  # Prime viewing hours
            'weekend_boost': 1.2
        }
    
    def get_location_from_ip(self, ip_address):
        """Enhanced location detection from IP address"""
        try:
            response = requests.get(f'http://ip-api.com/json/{ip_address}', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    return {
                        'country': data.get('country'),
                        'region': data.get('regionName'),
                        'city': data.get('city'),
                        'timezone': data.get('timezone'),
                        'lat': data.get('lat'),
                        'lon': data.get('lon')
                    }
        except Exception as e:
            logger.error(f"Error getting location from IP: {e}")
        return None
    
    def apply_regional_boost(self, recommendations, location):
        """Apply enhanced regional content boost to recommendations"""
        try:
            if not location or 'country' not in location:
                return recommendations
            
            country = location['country']
            region_config = self.region_preferences.get(country, {})
            
            if not region_config:
                return recommendations
            
            preferred_languages = region_config.get('languages', ['english'])
            base_boost = region_config.get('boost_factor', 1.0)
            content_type_boosts = region_config.get('content_types', {})
            
            for rec in recommendations:
                boost_factor = 1.0
                
                # Apply language boost
                if 'languages' in rec:
                    language_overlap = len(set(rec['languages']) & set(preferred_languages))
                    if language_overlap > 0:
                        boost_factor *= base_boost
                
                # Apply content type boost
                if 'content_type' in rec:
                    content_type = rec['content_type']
                    if content_type in content_type_boosts:
                        boost_factor *= content_type_boosts[content_type]
                
                # Apply timezone-based boost (if applicable)
                if location.get('timezone'):
                    time_boost = self._get_timezone_boost(location['timezone'])
                    boost_factor *= time_boost
                
                rec['score'] *= boost_factor
                if boost_factor > 1.1:
                    rec['reason'] += f" (Regional preference: {country})"
            
            # Re-sort by updated scores
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error applying regional boost: {e}")
            return recommendations
    
    def _get_timezone_boost(self, timezone):
        """Get boost factor based on timezone and time of day"""
        try:
            # This is a simplified implementation
            # In production, you'd implement proper timezone handling
            current_time = datetime.now()
            hour = current_time.hour
            
            # Prime time boost
            if 18 <= hour <= 23:
                return 1.1
            
            # Weekend boost
            if current_time.weekday() >= 5:  # Saturday or Sunday
                return self.time_zone_preferences['weekend_boost']
            
            return 1.0
            
        except:
            return 1.0

# Enhanced Caching System
class RecommendationCache:
    def __init__(self):
        self.memory_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_stats = defaultdict(int)
        
    def get_cache_key(self, user_id, preferences, algorithm='hybrid', **kwargs):
        """Generate enhanced cache key for recommendations"""
        key_data = {
            'user_id': user_id,
            'preferences': preferences,
            'algorithm': algorithm
        }
        key_data.update(kwargs)
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @lru_cache(maxsize=2000)
    def get_cached_recommendations(self, cache_key):
        """Get cached recommendations with stats tracking"""
        try:
            conn = get_cache_db()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT recommendations, expires_at FROM recommendation_cache WHERE cache_key = ?",
                (cache_key,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result and result[1] > time.time():
                self.cache_stats['hits'] += 1
                return json.loads(result[0])
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached recommendations: {e}")
            self.cache_stats['errors'] += 1
            return None
    
    def cache_recommendations(self, cache_key, recommendations, ttl=CACHE_DURATION):
        """Cache recommendations with custom TTL"""
        try:
            with self.cache_lock:
                conn = get_cache_db()
                cursor = conn.cursor()
                
                expires_at = time.time() + ttl
                
                cursor.execute(
                    """INSERT OR REPLACE INTO recommendation_cache 
                       (cache_key, recommendations, timestamp, expires_at) 
                       VALUES (?, ?, ?, ?)""",
                    (cache_key, json.dumps(recommendations), time.time(), expires_at)
                )
                conn.commit()
                conn.close()
                
                self.cache_stats['writes'] += 1
                
        except Exception as e:
            logger.error(f"Error caching recommendations: {e}")
            self.cache_stats['write_errors'] += 1
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'writes': self.cache_stats['writes'],
            'errors': self.cache_stats['errors'] + self.cache_stats['write_errors']
        }

# Main ML Service instances
ml_engine = HybridRecommendationEngine()
cold_start_handler = ColdStartHandler()
location_recommender = LocationBasedRecommender()
cache_system = RecommendationCache()

def load_data_from_backend():
    """Load data from the main backend service"""
    try:
        # This would connect to your main database
        # For now, we'll use enhanced mock data
        backend_url = os.environ.get('BACKEND_URL', 'http://localhost:5000')
        
        # Enhanced mock data with more realistic patterns
        np.random.seed(42)  # For reproducible results
        
        content_data = pd.DataFrame({
            'id': range(1, 501),  # More content for better testing
            'title': [f'Content {i}' for i in range(1, 501)],
            'genres': [
                json.dumps(np.random.choice([
                    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                    'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 
                    'Romance', 'Sci-Fi', 'Thriller', 'Western'
                ], size=np.random.randint(1, 4), replace=False).tolist())
                for _ in range(500)
            ],
            'languages': [
                json.dumps(np.random.choice([
                    'english', 'hindi', 'telugu', 'tamil', 'kannada', 
                    'malayalam', 'japanese', 'korean'
                ], size=np.random.randint(1, 3), replace=False).tolist())
                for _ in range(500)
            ],
            'rating': np.random.normal(7.0, 1.5, 500).clip(1, 10),
            'popularity': np.random.exponential(30, 500),
            'vote_count': np.random.lognormal(5, 1, 500).astype(int),
            'overview': [f'Overview for content {i}' for i in range(1, 501)],
            'content_type': np.random.choice(
                ['movie', 'tv', 'anime'], 
                size=500, 
                p=[0.6, 0.3, 0.1]
            ),
            'is_new_release': np.random.choice([True, False], size=500, p=[0.2, 0.8]),
            'is_critics_choice': np.random.choice([True, False], size=500, p=[0.15, 0.85]),
            'is_trending': np.random.choice([True, False], size=500, p=[0.1, 0.9])
        })
        
        # Enhanced interactions data with realistic patterns
        num_users = 50
        num_interactions = 5000
        
        # Create user behavior patterns
        user_preferences = {}
        for user_id in range(1, num_users + 1):
            user_preferences[user_id] = {
                'preferred_genres': np.random.choice([
                    'Action', 'Comedy', 'Drama', 'Romance', 'Thriller'
                ], size=np.random.randint(1, 3), replace=False).tolist(),
                'activity_level': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            }
        
        interactions = []
        for _ in range(num_interactions):
            user_id = np.random.randint(1, num_users + 1)
            content_id = np.random.randint(1, 501)
            
            # More realistic interaction patterns based on user preferences
            user_pref = user_preferences[user_id]
            interaction_types = ['view', 'like', 'favorite', 'watchlist', 'rating', 'search']
            
            if user_pref['activity_level'] == 'high':
                interaction_type = np.random.choice(interaction_types, p=[0.3, 0.2, 0.15, 0.15, 0.15, 0.05])
            elif user_pref['activity_level'] == 'medium':
                interaction_type = np.random.choice(interaction_types, p=[0.4, 0.25, 0.1, 0.1, 0.1, 0.05])
            else:
                interaction_type = np.random.choice(interaction_types, p=[0.6, 0.15, 0.05, 0.05, 0.1, 0.05])
            
            # Generate rating if it's a rating interaction
            rating = None
            if interaction_type == 'rating':
                rating = np.random.normal(7, 2)
                rating = max(1, min(10, rating))
            elif interaction_type in ['favorite', 'like']:
                rating = np.random.normal(8, 1)
                rating = max(1, min(10, rating))
            
            # Generate timestamp with realistic distribution
            days_ago = np.random.exponential(30)
            timestamp = (datetime.now() - timedelta(days=days_ago)).isoformat()
            
            interactions.append({
                'user_id': user_id,
                'content_id': content_id,
                'interaction_type': interaction_type,
                'rating': rating,
                'timestamp': timestamp
            })
        
        interactions_data = pd.DataFrame(interactions)
        
        return content_data, interactions_data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def update_models():
    """Update ML models with fresh data"""
    global last_model_update
    try:
        current_time = time.time()
        if current_time - last_model_update < MODEL_UPDATE_INTERVAL:
            return
        
        logger.info("Updating enhanced ML models...")
        
        content_data, interactions_data = load_data_from_backend()
        
        if not content_data.empty and not interactions_data.empty:
            # Initialize models
            success = ml_engine.initialize_models(content_data, interactions_data)
            
            if success:
                # Update cold start handler
                cold_start_handler.update_popularity_scores(content_data, interactions_data)
                
                last_model_update = current_time
                logger.info("Enhanced ML models updated successfully")
            else:
                logger.warning("Failed to initialize some ML models")
        
    except Exception as e:
        logger.error(f"Error updating models: {e}")

# API Routes

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Enhanced main recommendation endpoint with deep user understanding"""
    try:
        data = request.get_json()
        
        user_id = data.get('user_id')
        user_preferences = {
            'preferred_languages': data.get('preferred_languages', []),
            'preferred_genres': data.get('preferred_genres', [])
        }
        interactions = data.get('interactions', [])
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        
        # Enhanced cache key with interaction patterns
        interaction_hash = hashlib.md5(
            json.dumps(interactions, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        cache_key = cache_system.get_cache_key(
            user_id, user_preferences, 'hybrid', 
            interaction_hash=interaction_hash
        )
        
        # Check cache first
        cached_recs = cache_system.get_cached_recommendations(cache_key)
        if cached_recs:
            return jsonify({
                'recommendations': cached_recs, 
                'cached': True,
                'cache_stats': cache_system.get_cache_stats()
            }), 200
        
        # Update models if needed
        update_models()
        
        # Load fresh data
        content_data, interactions_data = load_data_from_backend()
        
        if content_data.empty:
            return jsonify({'error': 'No content data available'}), 500
        
        # Determine recommendation strategy based on user interaction history
        if not interactions or len(interactions) < 5:
            # Cold start - new user or insufficient data
            location = location_recommender.get_location_from_ip(ip_address)
            recommendations = cold_start_handler.get_cold_start_recommendations(
                user_preferences, content_data, n_recommendations=20
            )
            
            # Apply regional boost
            recommendations = location_recommender.apply_regional_boost(
                recommendations, location
            )
            
            strategy = 'cold_start_enhanced'
            
        else:
            # Existing user - use enhanced hybrid recommendations
            if ml_engine.is_initialized:
                recommendations = ml_engine.get_hybrid_recommendations(
                    user_id, user_preferences, content_data, 
                    interactions_data, n_recommendations=20
                )
                strategy = 'hybrid_enhanced'
            else:
                # Fallback to enhanced popularity-based
                location = location_recommender.get_location_from_ip(ip_address)
                recommendations = cold_start_handler.get_cold_start_recommendations(
                    user_preferences, content_data, n_recommendations=20
                )
                recommendations = location_recommender.apply_regional_boost(
                    recommendations, location
                )
                strategy = 'popularity_fallback'
        
        # Cache the results with longer TTL for complex computations
        cache_ttl = CACHE_DURATION * 2 if strategy == 'hybrid_enhanced' else CACHE_DURATION
        cache_system.cache_recommendations(cache_key, recommendations, cache_ttl)
        
        # Prepare response with detailed model status
        response = {
            'recommendations': recommendations,
            'strategy': strategy,
            'cached': False,
            'model_status': {
                'content_based': hasattr(ml_engine.content_recommender, 'similarity_matrix') and 
                                ml_engine.content_recommender.similarity_matrix is not None,
                'collaborative': ml_engine.collaborative_recommender.user_item_matrix is not None,
                'svd': ml_engine.svd_recommender.is_trained,
                'neural': ml_engine.neural_recommender.is_trained,
                'user_profiles': len(ml_engine.user_profile_builder.user_profiles),
                'last_update': last_model_update
            },
            'cache_stats': cache_system.get_cache_stats(),
            'recommendation_info': {
                'total_recommendations': len(recommendations),
                'diversity_score': self._calculate_recommendation_diversity(recommendations),
                'average_score': np.mean([r['score'] for r in recommendations]) if recommendations else 0
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error generating enhanced recommendations: {e}")
        return jsonify({'error': 'Failed to generate recommendations'}), 500

def _calculate_recommendation_diversity(recommendations):
    """Calculate diversity score of recommendations"""
    try:
        if not recommendations:
            return 0.0
        
        # Count unique genres and content types
        all_genres = set()
        content_types = set()
        
        for rec in recommendations:
            if 'genres' in rec:
                all_genres.update(rec['genres'])
            if 'content_type' in rec:
                content_types.add(rec['content_type'])
        
        # Simple diversity metric
        genre_diversity = len(all_genres) / 20.0  # Normalize by max possible genres
        type_diversity = len(content_types) / 5.0  # Normalize by max content types
        
        return min(1.0, (genre_diversity + type_diversity) / 2.0)
        
    except:
        return 0.5

@app.route('/api/similar', methods=['POST'])
def get_similar_content():
    """Get enhanced similar content recommendations"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        n_recommendations = data.get('n_recommendations', 20)
        user_id = data.get('user_id')  # Optional for personalization
        
        if not content_id:
            return jsonify({'error': 'content_id required'}), 400
        
        # Update models if needed
        update_models()
        
        content_data, interactions_data = load_data_from_backend()
        
        if content_data.empty:
            return jsonify({'error': 'No content data available'}), 500
        
        # Get enhanced similar recommendations
        if ml_engine.is_initialized:
            recommendations = ml_engine.get_similar_recommendations(
                content_id, content_data, n_recommendations
            )
        else:
            # Fallback to basic content-based similarity
            recommendations = ml_engine.content_recommender.get_content_recommendations(
                content_id, content_data, n_recommendations
            )
        
        # If user_id provided, personalize the similar recommendations
        if user_id and recommendations:
            user_interactions = [
                interaction for interaction in interactions_data.to_dict('records') 
                if interaction['user_id'] == user_id
            ]
            
            if user_interactions:
                user_profile = ml_engine.user_profile_builder.build_user_profile(
                    user_id, user_interactions, content_data
                )
                
                # Apply personalization boost to similar recommendations
                recommendations = self._personalize_similar_recommendations(
                    recommendations, user_profile, content_data
                )
        
        return jsonify({
            'recommendations': recommendations,
            'personalized': user_id is not None,
            'similarity_method': 'enhanced_hybrid' if ml_engine.is_initialized else 'content_based'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting similar content: {e}")
        return jsonify({'error': 'Failed to get similar content'}), 500

def _personalize_similar_recommendations(recommendations, user_profile, content_data):
    """Personalize similar recommendations based on user profile"""
    try:
        if not user_profile:
            return recommendations
        
        user_genre_scores = dict(user_profile.get('genre_scores', {}))
        user_language_scores = dict(user_profile.get('language_scores', {}))
        
        for rec in recommendations:
            content_id = rec['content_id']
            content_row = content_data[content_data['id'] == content_id]
            
            if not content_row.empty:
                content_info = content_row.iloc[0]
                boost_factor = 1.0
                
                # Genre preference boost
                try:
                    content_genres = json.loads(content_info['genres'])
                    for genre in content_genres:
                        if genre in user_genre_scores:
                            boost_factor += user_genre_scores[genre] * 0.3
                except:
                    pass
                
                # Language preference boost
                try:
                    content_languages = json.loads(content_info['languages'])
                    for language in content_languages:
                        if language in user_language_scores:
                            boost_factor += user_language_scores[language] * 0.2
                except:
                    pass
                
                rec['score'] *= boost_factor
                if boost_factor > 1.1:
                    rec['reason'] += ' (Personalized for you)'
        
        # Re-sort by updated scores
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error personalizing similar recommendations: {e}")
        return recommendations

@app.route('/api/trending', methods=['GET'])
def get_trending_recommendations():
    """Get enhanced trending content recommendations"""
    try:
        n_recommendations = int(request.args.get('limit', 20))
        region = request.args.get('region')
        user_id = request.args.get('user_id')  # Optional for personalization
        
        content_data, interactions_data = load_data_from_backend()
        
        if content_data.empty:
            return jsonify({'error': 'No content data available'}), 500
        
        # Update popularity scores
        cold_start_handler.update_popularity_scores(content_data, interactions_data)
        
        # Get enhanced trending content
        trending_content = []
        
        for content_id, score in cold_start_handler.trending_content[:n_recommendations * 2]:
            content_row = content_data[content_data['id'] == content_id]
            if not content_row.empty:
                content_info = content_row.iloc[0]
                
                # Additional boost for actually trending content
                if content_info.get('is_trending', False):
                    score *= 1.5
                
                trending_content.append({
                    'content_id': int(content_id),
                    'score': float(score),
                    'reason': 'Trending based on popularity and engagement',
                    'is_trending': content_info.get('is_trending', False),
                    'popularity': content_info.get('popularity', 0),
                    'rating': content_info.get('rating', 0)
                })
        
        # Sort by enhanced score and limit
        trending_content.sort(key=lambda x: x['score'], reverse=True)
        trending_content = trending_content[:n_recommendations]
        
        # Apply personalization if user_id provided
        if user_id:
            user_interactions = [
                interaction for interaction in interactions_data.to_dict('records') 
                if interaction['user_id'] == user_id
            ]
            
            if user_interactions:
                user_profile = ml_engine.user_profile_builder.build_user_profile(
                    user_id, user_interactions, content_data
                )
                trending_content = self._personalize_similar_recommendations(
                    trending_content, user_profile, content_data
                )
        
        return jsonify({
            'recommendations': trending_content,
            'personalized': user_id is not None,
            'trending_algorithm': 'enhanced_popularity_engagement'
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting trending recommendations: {e}")
        return jsonify({'error': 'Failed to get trending recommendations'}), 500

@app.route('/api/user-profile', methods=['POST'])
def analyze_user_profile():
    """Analyze and return detailed user profile"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        interactions = data.get('interactions', [])
        
        if not user_id:
            return jsonify({'error': 'user_id required'}), 400
        
        content_data, _ = load_data_from_backend()
        
        if content_data.empty:
            return jsonify({'error': 'No content data available'}), 500
        
        # Build comprehensive user profile
        user_profile = ml_engine.user_profile_builder.build_user_profile(
            user_id, interactions, content_data
        )
        
        if not user_profile:
            return jsonify({'error': 'Failed to build user profile'}), 500
        
        # Extract search patterns
        search_history = [i for i in interactions if i['interaction_type'] == 'search']
        search_patterns = ml_engine.search_analyzer.analyze_search_patterns(
            user_id, search_history, content_data
        )
        
        # Prepare response
        profile_response = {
            'user_id': user_id,
            'profile_summary': {
                'total_interactions': len(interactions),
                'diversity_score': user_profile.get('diversity_score', 0),
                'exploration_tendency': user_profile.get('exploration_tendency', 0),
                'quality_preference': user_profile.get('quality_preference', 0)
            },
            'preferences': {
                'top_genres': sorted(
                    user_profile.get('genre_scores', {}).items(), 
                    key=lambda x: x[1], reverse=True
                )[:5],
                'top_languages': sorted(
                    user_profile.get('language_scores', {}).items(), 
                    key=lambda x: x[1], reverse=True
                )[:3],
                'content_type_preferences': dict(user_profile.get('content_type_scores', {}))
            },
            'behavioral_patterns': {
                'interaction_frequency': dict(user_profile.get('interaction_frequency', {})),
                'search_patterns': search_patterns,
                'rating_patterns': user_profile.get('rating_patterns', [])[-10:]  # Last 10 ratings
            },
            'recommendations_insight': {
                'recommended_algorithms': self._get_recommended_algorithms(user_profile),
                'content_discovery_suggestions': self._get_discovery_suggestions(user_profile)
            }
        }
        
        return jsonify(profile_response), 200
        
    except Exception as e:
        logger.error(f"Error analyzing user profile: {e}")
        return jsonify({'error': 'Failed to analyze user profile'}), 500

def _get_recommended_algorithms(user_profile):
    """Get recommended algorithms based on user profile"""
    try:
        recommendations = []
        
        diversity_score = user_profile.get('diversity_score', 0.5)
        exploration_tendency = user_profile.get('exploration_tendency', 0.5)
        quality_preference = user_profile.get('quality_preference', 0.7)
        
        if diversity_score > 0.7:
            recommendations.append('content_based_exploration')
        
        if exploration_tendency > 0.6:
            recommendations.append('neural_discovery')
        
        if quality_preference > 0.8:
            recommendations.append('collaborative_quality_focused')
        
        if len(user_profile.get('interaction_frequency', {}).get('favorite', 0)) > 10:
            recommendations.append('favorite_based_similarity')
        
        return recommendations or ['balanced_hybrid']
        
    except:
        return ['balanced_hybrid']

def _get_discovery_suggestions(user_profile):
    """Get content discovery suggestions based on user profile"""
    try:
        suggestions = []
        
        # Analyze user's genre diversity
        genre_scores = user_profile.get('genre_scores', {})
        if len(genre_scores) < 3:
            suggestions.append('Try exploring more genres for better recommendations')
        
        # Analyze interaction types
        interaction_freq = user_profile.get('interaction_frequency', {})
        if interaction_freq.get('rating', 0) < 5:
            suggestions.append('Rate more content to improve personalization')
        
        if interaction_freq.get('favorite', 0) < 3:
            suggestions.append('Add favorites to get better similar recommendations')
        
        # Quality vs popularity analysis
        quality_pref = user_profile.get('quality_preference', 0.7)
        if quality_pref > 0.9:
            suggestions.append('Consider exploring some popular trending content')
        elif quality_pref < 0.5:
            suggestions.append('Try some critically acclaimed hidden gems')
        
        return suggestions or ['Your profile looks great! Keep exploring content.']
        
    except:
        return ['Continue exploring to improve recommendations']

@app.route('/api/update-models', methods=['POST'])
def force_model_update():
    """Force update of ML models"""
    try:
        global last_model_update
        last_model_update = 0  # Force update
        
        # Run update in background
        threading.Thread(target=update_models).start()
        
        return jsonify({
            'message': 'Enhanced model update initiated',
            'models_to_update': [
                'content_based_recommender',
                'collaborative_filtering',
                'svd_matrix_factorization', 
                'neural_collaborative_filtering',
                'user_profile_builder',
                'popularity_calculator'
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Error initiating model update: {e}")
        return jsonify({'error': 'Failed to initiate model update'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    try:
        content_data, interactions_data = load_data_from_backend()
        
        health_status = {
            'status': 'healthy',
            'models_initialized': ml_engine.is_initialized,
            'model_details': {
                'content_based': hasattr(ml_engine.content_recommender, 'similarity_matrix') and 
                                ml_engine.content_recommender.similarity_matrix is not None,
                'collaborative': ml_engine.collaborative_recommender.user_item_matrix is not None,
                'svd': ml_engine.svd_recommender.is_trained,
                'neural': ml_engine.neural_recommender.is_trained
            },
            'data_status': {
                'content_count': len(content_data),
                'interactions_count': len(interactions_data),
                'unique_users': interactions_data['user_id'].nunique() if not interactions_data.empty else 0
            },
            'cache_performance': cache_system.get_cache_stats(),
            'last_update': last_model_update,
            'next_update_in': max(0, MODEL_UPDATE_INTERVAL - (time.time() - last_model_update)),
            'system_info': {
                'pytorch_available': torch.cuda.is_available(),
                'device': str(ml_engine.neural_recommender.device),
                'memory_usage': len(cache_system.memory_cache)
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_ml_stats():
    """Get comprehensive ML service statistics"""
    try:
        content_data, interactions_data = load_data_from_backend()
        
        # Calculate advanced statistics
        stats = {
            'data_statistics': {
                'total_content': len(content_data),
                'total_interactions': len(interactions_data),
                'unique_users': interactions_data['user_id'].nunique() if not interactions_data.empty else 0,
                'content_by_type': content_data['content_type'].value_counts().to_dict() if not content_data.empty else {},
                'interactions_by_type': interactions_data['interaction_type'].value_counts().to_dict() if not interactions_data.empty else {}
            },
            'model_performance': {
                'models_trained': {
                    'content_based': hasattr(ml_engine.content_recommender, 'similarity_matrix') and 
                                    ml_engine.content_recommender.similarity_matrix is not None,
                    'collaborative': ml_engine.collaborative_recommender.user_item_matrix is not None,
                    'svd': ml_engine.svd_recommender.is_trained,
                    'neural': ml_engine.neural_recommender.is_trained
                },
                'user_profiles_built': len(ml_engine.user_profile_builder.user_profiles),
                'content_features_extracted': ml_engine.content_recommender.content_matrix is not None
            },
            'cache_performance': cache_system.get_cache_stats(),
            'algorithm_weights': ml_engine.algorithm_weights,
            'behavior_weights': ml_engine.behavior_weights,
            'system_config': {
                'cache_duration': CACHE_DURATION,
                'model_update_interval': MODEL_UPDATE_INTERVAL,
                'min_interactions': MIN_INTERACTIONS,
                'interaction_weights': INTERACTION_WEIGHTS,
                'temporal_decay': TEMPORAL_DECAY
            },
            'performance_metrics': {
                'last_model_update': last_model_update,
                'model_update_frequency': '24 hours',
                'recommendation_generation_time': 'varies by complexity',
                'supported_algorithms': [
                    'content_based_filtering',
                    'collaborative_filtering', 
                    'matrix_factorization_svd',
                    'neural_collaborative_filtering',
                    'hybrid_ensemble',
                    'behavioral_pattern_analysis',
                    'location_based_recommendations',
                    'cold_start_handling'
                ]
            }
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error getting ML stats: {e}")
        return jsonify({'error': 'Failed to get stats'}), 500

# Initialize cache database
init_cache_db()

# Background model update
def background_model_updater():
    """Enhanced background thread for periodic model updates"""
    while True:
        try:
            time.sleep(MODEL_UPDATE_INTERVAL)
            logger.info("Starting scheduled model update...")
            update_models()
            logger.info("Scheduled model update completed")
        except Exception as e:
            logger.error(f"Background model update error: {e}")

# Start background updater
if __name__ == '__main__':
    # Start enhanced background model updater
    updater_thread = threading.Thread(target=background_model_updater, daemon=True)
    updater_thread.start()
    
    # Initial model update
    logger.info("Starting initial model initialization...")
    threading.Thread(target=update_models).start()
    
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)