# backend/services/personalized.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import nltk
from textblob import TextBlob
import spacy
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import joblib
import os
import networkx as nx
from itertools import combinations
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

logger = logging.getLogger(__name__)

app = None
db = None
cache = None
models = {}
services = {}

User = None
Content = None
UserInteraction = None
UserBehaviorProfile = None
ContentFeatures = None
RecommendationModel = None
UserSession = None
SearchHistory = None

@dataclass
class RecommendationResult:
    content_id: int
    score: float
    reason: str
    category: str
    confidence: float
    source_algorithms: List[str]
    diversity_score: float
    novelty_score: float

class UltraAdvancedPersonalizationEngine:
    
    def __init__(self, app, db, models, services, cache):
        self.app = app
        self.db = db
        self.models = models
        self.services = services
        self.cache = cache
        
        self.content_vectorizer = TfidfVectorizer(
            max_features=20000,
            stop_words='english',
            ngram_range=(1, 4),
            min_df=1,
            max_df=0.9,
            sublinear_tf=True
        )
        
        self.user_item_matrix = None
        self.content_features_matrix = None
        self.user_profiles = {}
        self.content_embeddings = {}
        self.genre_clusters = None
        self.content_graph = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        
        self.models_ensemble = {
            'svd': None,
            'nmf': None,
            'neural_cf': None,
            'graph_based': None,
            'deep_learning': None,
            'contextual_bandit': None,
            'sequence_model': None,
            'ensemble_meta': None,
            'transformer': None,
            'factorization_machine': None
        }
        
        self.algorithm_weights = {
            'collaborative_svd': 0.15,
            'collaborative_nmf': 0.12,
            'content_based': 0.13,
            'hybrid_neural': 0.15,
            'graph_based': 0.10,
            'sequence_aware': 0.08,
            'contextual': 0.07,
            'clustering_based': 0.06,
            'popularity_boost': 0.05,
            'novelty_boost': 0.04,
            'diversity_boost': 0.03,
            'temporal_boost': 0.02
        }
        
        self.user_segments = {}
        self.content_clusters = {}
        self.trending_patterns = {}
        
        self._initialize_all_models()
        
        self.update_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
        self.update_thread.start()
    
    def _initialize_all_models(self):
        try:
            model_path = 'models/ultra_personalization/'
            os.makedirs(model_path, exist_ok=True)
            
            self._build_comprehensive_features()
            self._build_user_item_matrices()
            self._train_matrix_factorization_models()
            self._train_neural_collaborative_filtering()
            self._build_content_knowledge_graph()
            self._train_sequence_models()
            self._train_contextual_bandits()
            self._train_ensemble_meta_learner()
            self._build_user_segments()
            self._compute_similarity_matrices()
            
            logger.info("Ultra-advanced personalization models initialized")
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
    
    def _build_comprehensive_features(self):
        try:
            contents = self.models['Content'].query.all()
            
            text_features = []
            content_metadata = []
            content_graph_data = []
            
            for content in contents:
                text = f"{content.title or ''} {content.original_title or ''} {content.overview or ''}"
                
                try:
                    genres = json.loads(content.genres or '[]')
                    anime_genres = json.loads(content.anime_genres or '[]') if hasattr(content, 'anime_genres') else []
                    languages = json.loads(content.languages or '[]')
                    text += " " + " ".join(genres + anime_genres + languages)
                except:
                    genres = []
                    anime_genres = []
                    languages = []
                
                storyline_features = self._advanced_storyline_analysis(content.overview or "")
                
                text_features.append(text)
                
                content_metadata.append({
                    'id': content.id,
                    'content_type': content.content_type,
                    'genres': genres,
                    'anime_genres': anime_genres,
                    'languages': languages,
                    'rating': content.rating or 0,
                    'popularity': content.popularity or 0,
                    'release_year': content.release_date.year if content.release_date else 2020,
                    'runtime': content.runtime or 120,
                    'vote_count': content.vote_count or 0,
                    'storyline_sentiment': storyline_features.get('sentiment', 0),
                    'storyline_complexity': storyline_features.get('complexity', 0.5),
                    'theme_vector': storyline_features.get('themes', [0] * 10),
                    'emotion_vector': storyline_features.get('emotions', [0] * 8)
                })
                
                content_graph_data.append({
                    'id': content.id,
                    'genres': genres,
                    'languages': languages,
                    'content_type': content.content_type
                })
            
            if text_features:
                self.content_features_matrix = self.content_vectorizer.fit_transform(text_features)
                
                numerical_features = []
                for meta in content_metadata:
                    features = [
                        meta['rating'],
                        meta['popularity'],
                        meta['release_year'] / 2024.0,
                        meta['runtime'] / 300.0,
                        np.log1p(meta['vote_count']),
                        meta['storyline_sentiment'],
                        meta['storyline_complexity'],
                        len(meta['genres']) / 10.0,
                        len(meta['languages']) / 5.0
                    ]
                    features.extend(meta['theme_vector'])
                    features.extend(meta['emotion_vector'])
                    numerical_features.append(features)
                
                scaler = StandardScaler()
                numerical_features_scaled = scaler.fit_transform(numerical_features)
                
                self.content_features_matrix = sp.hstack([
                    self.content_features_matrix,
                    sp.csr_matrix(numerical_features_scaled)
                ])
                
                self.content_id_to_index = {meta['id']: i for i, meta in enumerate(content_metadata)}
                self.index_to_content_id = {i: meta['id'] for i, meta in enumerate(content_metadata)}
                self.content_metadata = content_metadata
                self.content_graph_data = content_graph_data
            
            logger.info(f"Comprehensive features built for {len(text_features)} items")
            
        except Exception as e:
            logger.error(f"Feature building error: {e}")
    
    def _advanced_storyline_analysis(self, text):
        try:
            if not text:
                return {'sentiment': 0, 'complexity': 0.5, 'themes': [0]*10, 'emotions': [0]*8}
            
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            
            complexity = len(set(text.split())) / max(len(text.split()), 1)
            
            themes = [0] * 10
            emotions = [0] * 8
            
            theme_keywords = {
                0: ['love', 'romance', 'relationship', 'heart'],
                1: ['action', 'fight', 'battle', 'war', 'combat'],
                2: ['mystery', 'secret', 'hidden', 'unknown'],
                3: ['adventure', 'journey', 'travel', 'explore'],
                4: ['family', 'father', 'mother', 'children'],
                5: ['friendship', 'friend', 'buddy', 'companion'],
                6: ['crime', 'criminal', 'police', 'detective'],
                7: ['fantasy', 'magic', 'supernatural', 'mystical'],
                8: ['horror', 'scary', 'fear', 'terror'],
                9: ['comedy', 'funny', 'humor', 'laugh']
            }
            
            emotion_keywords = {
                0: ['happy', 'joy', 'excited', 'cheerful'],
                1: ['sad', 'depressed', 'melancholy', 'sorrow'],
                2: ['angry', 'rage', 'furious', 'mad'],
                3: ['fear', 'afraid', 'scared', 'terrified'],
                4: ['surprise', 'shocked', 'amazed', 'astonished'],
                5: ['disgust', 'revolted', 'repulsed', 'sick'],
                6: ['anticipation', 'expect', 'hope', 'await'],
                7: ['trust', 'faith', 'believe', 'confident']
            }
            
            text_lower = text.lower()
            
            for theme_id, keywords in theme_keywords.items():
                themes[theme_id] = sum(text_lower.count(keyword) for keyword in keywords) / len(text.split())
            
            for emotion_id, keywords in emotion_keywords.items():
                emotions[emotion_id] = sum(text_lower.count(keyword) for keyword in keywords) / len(text.split())
            
            if nlp:
                doc = nlp(text[:1000])
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                for _, label in entities:
                    if label == 'PERSON':
                        themes[4] += 0.1
                    elif label in ['GPE', 'LOC']:
                        themes[3] += 0.1
            
            return {
                'sentiment': sentiment,
                'complexity': complexity,
                'themes': themes,
                'emotions': emotions
            }
            
        except Exception as e:
            logger.error(f"Storyline analysis error: {e}")
            return {'sentiment': 0, 'complexity': 0.5, 'themes': [0]*10, 'emotions': [0]*8}
    
    def _build_user_item_matrices(self):
        try:
            interactions = self.models['UserInteraction'].query.all()
            
            if not interactions:
                return
            
            user_ids = list(set([inter.user_id for inter in interactions]))
            content_ids = list(set([inter.content_id for inter in interactions]))
            
            self.user_id_to_index = {uid: i for i, uid in enumerate(user_ids)}
            self.content_id_to_matrix_index = {cid: i for i, cid in enumerate(content_ids)}
            
            explicit_matrix = np.zeros((len(user_ids), len(content_ids)))
            implicit_matrix = np.zeros((len(user_ids), len(content_ids)))
            confidence_matrix = np.ones((len(user_ids), len(content_ids)))
            
            interaction_weights = {
                'rating': 1.0,
                'view': 0.5,
                'favorite': 1.5,
                'watchlist': 1.2,
                'search': 0.2,
                'share': 0.8,
                'like': 0.7,
                'comment': 0.6,
                'download': 1.0,
                'complete': 1.3
            }
            
            for inter in interactions:
                user_idx = self.user_id_to_index[inter.user_id]
                content_idx = self.content_id_to_matrix_index[inter.content_id]
                
                if inter.rating:
                    explicit_matrix[user_idx, content_idx] = inter.rating
                    confidence_matrix[user_idx, content_idx] = 2.0
                
                weight = interaction_weights.get(inter.interaction_type, 0.3)
                implicit_matrix[user_idx, content_idx] += weight
                confidence_matrix[user_idx, content_idx] += weight
            
            implicit_matrix = np.clip(implicit_matrix, 0, 5)
            
            self.explicit_matrix = explicit_matrix
            self.implicit_matrix = implicit_matrix
            self.confidence_matrix = confidence_matrix
            self.user_item_matrix = np.where(explicit_matrix > 0, explicit_matrix, implicit_matrix)
            
            logger.info(f"User-item matrices built: {self.user_item_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Matrix building error: {e}")
    
    def _train_matrix_factorization_models(self):
        try:
            if self.user_item_matrix is None or self.user_item_matrix.size == 0:
                return
            
            n_components = min(100, min(self.user_item_matrix.shape) - 1)
            
            self.models_ensemble['svd'] = TruncatedSVD(
                n_components=n_components,
                algorithm='randomized',
                n_iter=10,
                random_state=42
            )
            
            self.models_ensemble['nmf'] = NMF(
                n_components=n_components,
                init='random',
                random_state=42,
                max_iter=1000,
                alpha=0.1
            )
            
            user_factors_svd = self.models_ensemble['svd'].fit_transform(self.user_item_matrix)
            item_factors_svd = self.models_ensemble['svd'].components_.T
            
            user_factors_nmf = self.models_ensemble['nmf'].fit_transform(self.user_item_matrix)
            item_factors_nmf = self.models_ensemble['nmf'].components_.T
            
            self.user_factors_svd = user_factors_svd
            self.item_factors_svd = item_factors_svd
            self.user_factors_nmf = user_factors_nmf
            self.item_factors_nmf = item_factors_nmf
            
            logger.info(f"Matrix factorization models trained with {n_components} components")
            
        except Exception as e:
            logger.error(f"Matrix factorization training error: {e}")
    
    def _train_neural_collaborative_filtering(self):
        try:
            if self.user_item_matrix is None:
                return
            
            X_train = []
            y_train = []
            
            for user_idx, user_id in enumerate(self.user_id_to_index.keys()):
                for content_idx, content_id in enumerate(self.content_id_to_matrix_index.keys()):
                    rating = self.user_item_matrix[self.user_id_to_index[user_id], 
                                                 self.content_id_to_matrix_index[content_id]]
                    if rating > 0:
                        user_features = self._get_user_feature_vector(user_id)
                        content_features = self._get_content_feature_vector(content_id)
                        
                        feature_vector = np.concatenate([user_features, content_features])
                        X_train.append(feature_vector)
                        y_train.append(rating)
            
            if len(X_train) > 100:
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                self.models_ensemble['neural_cf'] = MLPRegressor(
                    hidden_layer_sizes=(256, 128, 64, 32),
                    activation='relu',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=500,
                    random_state=42
                )
                
                self.models_ensemble['neural_cf'].fit(X_train, y_train)
                
                logger.info("Neural collaborative filtering model trained")
            
        except Exception as e:
            logger.error(f"Neural CF training error: {e}")
    
    def _build_content_knowledge_graph(self):
        try:
            self.content_graph = nx.Graph()
            
            for content_data in self.content_graph_data:
                self.content_graph.add_node(
                    content_data['id'],
                    content_type=content_data['content_type'],
                    genres=content_data['genres'],
                    languages=content_data['languages']
                )
            
            for i, content1 in enumerate(self.content_graph_data):
                for j, content2 in enumerate(self.content_graph_data[i+1:], i+1):
                    similarity = self._calculate_content_similarity(content1, content2)
                    if similarity > 0.3:
                        self.content_graph.add_edge(content1['id'], content2['id'], weight=similarity)
            
            if len(self.content_graph.nodes()) > 0:
                self.pagerank_scores = nx.pagerank(self.content_graph, weight='weight')
                self.centrality_scores = nx.degree_centrality(self.content_graph)
            
            logger.info(f"Content knowledge graph built with {len(self.content_graph.nodes())} nodes")
            
        except Exception as e:
            logger.error(f"Knowledge graph building error: {e}")
    
    def _calculate_content_similarity(self, content1, content2):
        try:
            similarity = 0.0
            
            common_genres = set(content1['genres']) & set(content2['genres'])
            similarity += len(common_genres) * 0.4
            
            common_languages = set(content1['languages']) & set(content2['languages'])
            similarity += len(common_languages) * 0.3
            
            if content1['content_type'] == content2['content_type']:
                similarity += 0.3
            
            return min(1.0, similarity)
            
        except Exception as e:
            return 0.0
    
    def _train_sequence_models(self):
        try:
            users_sequences = {}
            
            for user_id in self.user_id_to_index.keys():
                interactions = self.models['UserInteraction'].query.filter_by(
                    user_id=user_id
                ).order_by(self.models['UserInteraction'].timestamp).all()
                
                if len(interactions) >= 3:
                    sequence = []
                    for inter in interactions:
                        if inter.content_id in self.content_id_to_matrix_index:
                            sequence.append(inter.content_id)
                    
                    if len(sequence) >= 3:
                        users_sequences[user_id] = sequence
            
            self.user_sequences = users_sequences
            
            sequence_patterns = defaultdict(list)
            
            for user_id, sequence in users_sequences.items():
                for i in range(len(sequence) - 2):
                    pattern = tuple(sequence[i:i+2])
                    next_item = sequence[i+2]
                    sequence_patterns[pattern].append(next_item)
            
            self.sequence_patterns = dict(sequence_patterns)
            
            logger.info(f"Sequence models trained with {len(users_sequences)} user sequences")
            
        except Exception as e:
            logger.error(f"Sequence training error: {e}")
    
    def _train_contextual_bandits(self):
        try:
            self.contextual_rewards = defaultdict(list)
            
            for user_id in self.user_id_to_index.keys():
                interactions = self.models['UserInteraction'].query.filter_by(user_id=user_id).all()
                
                for inter in interactions:
                    context = {
                        'hour': inter.timestamp.hour if inter.timestamp else 12,
                        'day': inter.timestamp.weekday() if inter.timestamp else 1,
                        'user_id': user_id,
                        'content_id': inter.content_id
                    }
                    
                    reward = 1.0
                    if inter.rating:
                        reward = inter.rating / 5.0
                    elif inter.interaction_type == 'favorite':
                        reward = 1.0
                    elif inter.interaction_type == 'watchlist':
                        reward = 0.8
                    elif inter.interaction_type == 'view':
                        reward = 0.6
                    else:
                        reward = 0.3
                    
                    self.contextual_rewards[user_id].append((context, reward))
            
            logger.info("Contextual bandits trained")
            
        except Exception as e:
            logger.error(f"Contextual bandits training error: {e}")
    
    def _train_ensemble_meta_learner(self):
        try:
            meta_features = []
            meta_targets = []
            
            for user_id in list(self.user_id_to_index.keys())[:100]:
                for content_id in list(self.content_id_to_matrix_index.keys())[:200]:
                    actual_rating = self.user_item_matrix[
                        self.user_id_to_index[user_id],
                        self.content_id_to_matrix_index[content_id]
                    ]
                    
                    if actual_rating > 0:
                        predictions = []
                        
                        svd_pred = self._predict_svd(user_id, content_id)
                        nmf_pred = self._predict_nmf(user_id, content_id)
                        content_pred = self._predict_content_based(user_id, content_id)
                        graph_pred = self._predict_graph_based(user_id, content_id)
                        
                        predictions.extend([svd_pred, nmf_pred, content_pred, graph_pred])
                        
                        user_features = self._get_user_meta_features(user_id)
                        content_features = self._get_content_meta_features(content_id)
                        
                        meta_feature = predictions + user_features + content_features
                        meta_features.append(meta_feature)
                        meta_targets.append(actual_rating)
            
            if len(meta_features) > 50:
                meta_features = np.array(meta_features)
                meta_targets = np.array(meta_targets)
                
                self.models_ensemble['ensemble_meta'] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                
                self.models_ensemble['ensemble_meta'].fit(meta_features, meta_targets)
                
                logger.info("Ensemble meta-learner trained")
            
        except Exception as e:
            logger.error(f"Meta-learner training error: {e}")
    
    def _build_user_segments(self):
        try:
            user_features = []
            user_ids = []
            
            for user_id in self.user_id_to_index.keys():
                profile = self._get_comprehensive_user_profile(user_id)
                
                features = [
                    profile.get('avg_rating', 3.5),
                    profile.get('total_interactions', 0),
                    profile.get('diversity_score', 0.5),
                    len(profile.get('genre_preferences', {})),
                    len(profile.get('language_preferences', {})),
                    profile.get('activity_level_numeric', 0.5),
                    profile.get('exploration_tendency', 0.5),
                    profile.get('popular_affinity', 0.5)
                ]
                
                user_features.append(features)
                user_ids.append(user_id)
            
            if len(user_features) >= 5:
                user_features = np.array(user_features)
                
                n_clusters = min(10, len(user_features) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(user_features)
                
                for user_id, cluster in zip(user_ids, clusters):
                    self.user_segments[user_id] = cluster
                
                logger.info(f"User segmentation completed with {n_clusters} segments")
            
        except Exception as e:
            logger.error(f"User segmentation error: {e}")
    
    def _compute_similarity_matrices(self):
        try:
            if self.user_item_matrix is not None:
                self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
                self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            
            if self.content_features_matrix is not None:
                self.content_content_similarity = cosine_similarity(self.content_features_matrix)
            
            logger.info("Similarity matrices computed")
            
        except Exception as e:
            logger.error(f"Similarity computation error: {e}")
    
    def _get_user_feature_vector(self, user_id):
        try:
            profile = self._get_comprehensive_user_profile(user_id)
            
            features = [
                profile.get('avg_rating', 3.5) / 5.0,
                min(1.0, profile.get('total_interactions', 0) / 100.0),
                profile.get('diversity_score', 0.5),
                profile.get('activity_level_numeric', 0.5),
                profile.get('exploration_tendency', 0.5),
                profile.get('popular_affinity', 0.5)
            ]
            
            genre_prefs = profile.get('genre_preferences', {})
            top_genres = ['Action', 'Drama', 'Comedy', 'Romance', 'Thriller', 'Horror', 'Animation', 'Documentary']
            for genre in top_genres:
                features.append(genre_prefs.get(genre, 0.0))
            
            return np.array(features)
            
        except Exception as e:
            return np.array([0.5] * 14)
    
    def _get_content_feature_vector(self, content_id):
        try:
            if content_id not in self.content_id_to_index:
                return np.array([0.5] * 10)
            
            idx = self.content_id_to_index[content_id]
            meta = self.content_metadata[idx]
            
            features = [
                meta['rating'] / 10.0,
                min(1.0, meta['popularity'] / 1000.0),
                meta['release_year'] / 2024.0,
                meta['runtime'] / 300.0,
                meta['storyline_sentiment'],
                meta['storyline_complexity'],
                len(meta['genres']) / 10.0,
                len(meta['languages']) / 5.0,
                1.0 if meta['content_type'] == 'movie' else 0.0,
                1.0 if meta['content_type'] == 'anime' else 0.0
            ]
            
            return np.array(features)
            
        except Exception as e:
            return np.array([0.5] * 10)
    
    def _get_comprehensive_user_profile(self, user_id):
        try:
            cache_key = f"ultra_profile:{user_id}"
            cached = self.cache.get(cache_key) if self.cache else None
            
            if cached:
                return cached
            
            interactions = self.models['UserInteraction'].query.filter_by(user_id=user_id).all()
            
            if not interactions:
                return self._get_default_profile()
            
            genre_preferences = defaultdict(float)
            language_preferences = defaultdict(float)
            content_type_preferences = defaultdict(float)
            time_preferences = defaultdict(int)
            rating_patterns = []
            interaction_patterns = defaultdict(int)
            
            total_score = 0
            
            for inter in interactions:
                content = self.models['Content'].query.get(inter.content_id)
                if not content:
                    continue
                
                weights = {
                    'rating': inter.rating if inter.rating else 3.0,
                    'view': 3.0,
                    'favorite': 5.0,
                    'watchlist': 4.0,
                    'search': 1.0,
                    'share': 2.5,
                    'like': 3.5,
                    'comment': 2.0,
                    'download': 4.5,
                    'complete': 5.0
                }
                
                weight = weights.get(inter.interaction_type, 1.0)
                if inter.rating:
                    weight = inter.rating * 1.5
                    rating_patterns.append(inter.rating)
                
                total_score += weight
                interaction_patterns[inter.interaction_type] += 1
                
                try:
                    genres = json.loads(content.genres or '[]')
                    for genre in genres:
                        genre_preferences[genre] += weight
                except:
                    pass
                
                try:
                    languages = json.loads(content.languages or '[]')
                    for lang in languages:
                        language_preferences[lang] += weight
                except:
                    pass
                
                content_type_preferences[content.content_type] += weight
                
                hour = inter.timestamp.hour if inter.timestamp else 12
                time_preferences[hour] += 1
            
            if total_score > 0:
                for genre in genre_preferences:
                    genre_preferences[genre] /= total_score
                for lang in language_preferences:
                    language_preferences[lang] /= total_score
                for ctype in content_type_preferences:
                    content_type_preferences[ctype] /= total_score
            
            activity_levels = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'very_high': 1.0}
            activity_level = 'low' if len(interactions) < 5 else 'medium' if len(interactions) < 20 else 'high' if len(interactions) < 50 else 'very_high'
            
            exploration_tendency = len(genre_preferences) / max(len(interactions), 1)
            popular_affinity = sum(1 for inter in interactions if self._is_popular_content(inter.content_id)) / max(len(interactions), 1)
            
            profile = {
                'user_id': user_id,
                'genre_preferences': dict(genre_preferences),
                'language_preferences': dict(language_preferences),
                'content_type_preferences': dict(content_type_preferences),
                'time_preferences': dict(time_preferences),
                'interaction_patterns': dict(interaction_patterns),
                'avg_rating': np.mean(rating_patterns) if rating_patterns else 3.5,
                'rating_variance': np.var(rating_patterns) if rating_patterns else 1.0,
                'total_interactions': len(interactions),
                'diversity_score': len(genre_preferences) / max(len(interactions), 1),
                'activity_level': activity_level,
                'activity_level_numeric': activity_levels[activity_level],
                'exploration_tendency': exploration_tendency,
                'popular_affinity': popular_affinity,
                'preferred_time': max(time_preferences, key=time_preferences.get) if time_preferences else 20,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            if self.cache:
                self.cache.set(cache_key, profile, timeout=1800)
            
            return profile
            
        except Exception as e:
            logger.error(f"Profile building error: {e}")
            return self._get_default_profile()
    
    def _get_default_profile(self):
        return {
            'genre_preferences': {'Action': 0.2, 'Drama': 0.2, 'Comedy': 0.2},
            'language_preferences': {'english': 0.4, 'telugu': 0.3, 'hindi': 0.3},
            'content_type_preferences': {'movie': 0.5, 'tv': 0.3, 'anime': 0.2},
            'time_preferences': {20: 5, 21: 4, 19: 3},
            'interaction_patterns': {},
            'avg_rating': 3.5,
            'rating_variance': 1.0,
            'total_interactions': 0,
            'diversity_score': 0.5,
            'activity_level': 'new',
            'activity_level_numeric': 0.1,
            'exploration_tendency': 0.5,
            'popular_affinity': 0.5,
            'preferred_time': 20,
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def _is_popular_content(self, content_id):
        try:
            content = self.models['Content'].query.get(content_id)
            return content and (content.popularity or 0) > 100
        except:
            return False
    
    def _get_user_meta_features(self, user_id):
        profile = self._get_comprehensive_user_profile(user_id)
        return [
            profile['activity_level_numeric'],
            profile['diversity_score'],
            profile['exploration_tendency'],
            profile['popular_affinity']
        ]
    
    def _get_content_meta_features(self, content_id):
        try:
            content = self.models['Content'].query.get(content_id)
            if not content:
                return [0.5, 0.5, 0.5, 0.5]
            
            return [
                (content.rating or 5.0) / 10.0,
                min(1.0, (content.popularity or 0) / 1000.0),
                1.0 if content.is_trending else 0.0,
                1.0 if content.is_new_release else 0.0
            ]
        except:
            return [0.5, 0.5, 0.5, 0.5]
    
    def _predict_svd(self, user_id, content_id):
        try:
            if (user_id not in self.user_id_to_index or 
                content_id not in self.content_id_to_matrix_index or
                self.models_ensemble['svd'] is None):
                return 3.5
            
            user_idx = self.user_id_to_index[user_id]
            content_idx = self.content_id_to_matrix_index[content_id]
            
            user_vector = self.user_factors_svd[user_idx]
            item_vector = self.item_factors_svd[content_idx]
            
            prediction = np.dot(user_vector, item_vector)
            return max(0.5, min(5.0, prediction))
            
        except Exception as e:
            return 3.5
    
    def _predict_nmf(self, user_id, content_id):
        try:
            if (user_id not in self.user_id_to_index or 
                content_id not in self.content_id_to_matrix_index or
                self.models_ensemble['nmf'] is None):
                return 3.5
            
            user_idx = self.user_id_to_index[user_id]
            content_idx = self.content_id_to_matrix_index[content_id]
            
            user_vector = self.user_factors_nmf[user_idx]
            item_vector = self.item_factors_nmf[content_idx]
            
            prediction = np.dot(user_vector, item_vector)
            return max(0.5, min(5.0, prediction))
            
        except Exception as e:
            return 3.5
    
    def _predict_content_based(self, user_id, content_id):
        try:
            if (content_id not in self.content_id_to_index or
                not hasattr(self, 'content_content_similarity')):
                return 3.5
            
            interactions = self.models['UserInteraction'].query.filter_by(user_id=user_id).all()
            
            if not interactions:
                return 3.5
            
            content_idx = self.content_id_to_index[content_id]
            weighted_sum = 0.0
            similarity_sum = 0.0
            
            for inter in interactions:
                if inter.content_id in self.content_id_to_index:
                    other_idx = self.content_id_to_index[inter.content_id]
                    similarity = self.content_content_similarity[content_idx, other_idx]
                    
                    rating = inter.rating if inter.rating else 3.0
                    if inter.interaction_type == 'favorite':
                        rating = 5.0
                    elif inter.interaction_type == 'watchlist':
                        rating = 4.0
                    
                    weighted_sum += similarity * rating
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                prediction = weighted_sum / similarity_sum
                return max(0.5, min(5.0, prediction))
            
            return 3.5
            
        except Exception as e:
            return 3.5
    
    def _predict_graph_based(self, user_id, content_id):
        try:
            if (not hasattr(self, 'content_graph') or 
                content_id not in self.content_graph.nodes()):
                return 3.5
            
            pagerank_score = self.pagerank_scores.get(content_id, 0.001)
            centrality_score = self.centrality_scores.get(content_id, 0.001)
            
            neighbors = list(self.content_graph.neighbors(content_id))
            
            user_interactions = self.models['UserInteraction'].query.filter_by(user_id=user_id).all()
            interacted_content_ids = {inter.content_id for inter in user_interactions}
            
            neighbor_score = 0.0
            if neighbors:
                relevant_neighbors = [n for n in neighbors if n in interacted_content_ids]
                neighbor_score = len(relevant_neighbors) / len(neighbors)
            
            prediction = (pagerank_score * 1000 + centrality_score + neighbor_score) * 2.5
            return max(0.5, min(5.0, prediction))
            
        except Exception as e:
            return 3.5
    
    def _predict_neural_cf(self, user_id, content_id):
        try:
            if self.models_ensemble['neural_cf'] is None:
                return 3.5
            
            user_features = self._get_user_feature_vector(user_id)
            content_features = self._get_content_feature_vector(content_id)
            
            feature_vector = np.concatenate([user_features, content_features]).reshape(1, -1)
            prediction = self.models_ensemble['neural_cf'].predict(feature_vector)[0]
            
            return max(0.5, min(5.0, prediction))
            
        except Exception as e:
            return 3.5
    
    def _predict_sequence_aware(self, user_id, content_id):
        try:
            if user_id not in self.user_sequences:
                return 3.5
            
            user_sequence = self.user_sequences[user_id]
            
            if len(user_sequence) < 2:
                return 3.5
            
            recent_pattern = tuple(user_sequence[-2:])
            
            if recent_pattern in self.sequence_patterns:
                predictions = self.sequence_patterns[recent_pattern]
                if content_id in predictions:
                    frequency = predictions.count(content_id) / len(predictions)
                    return 3.5 + frequency * 1.5
            
            return 3.5
            
        except Exception as e:
            return 3.5
    
    def _predict_contextual_bandit(self, user_id, content_id, context=None):
        try:
            if user_id not in self.contextual_rewards:
                return 3.5
            
            current_hour = context.get('time', 20) if context else 20
            current_day = context.get('day', 1) if context else 1
            
            user_rewards = self.contextual_rewards[user_id]
            
            relevant_rewards = []
            for ctx, reward in user_rewards:
                if abs(ctx['hour'] - current_hour) <= 2:
                    relevant_rewards.append(reward)
            
            if relevant_rewards:
                return np.mean(relevant_rewards) * 5.0
            
            return 3.5
            
        except Exception as e:
            return 3.5
    
    def _predict_ensemble_meta(self, user_id, content_id, context=None):
        try:
            if self.models_ensemble['ensemble_meta'] is None:
                return 3.5
            
            predictions = []
            
            svd_pred = self._predict_svd(user_id, content_id)
            nmf_pred = self._predict_nmf(user_id, content_id)
            content_pred = self._predict_content_based(user_id, content_id)
            graph_pred = self._predict_graph_based(user_id, content_id)
            neural_pred = self._predict_neural_cf(user_id, content_id)
            
            predictions.extend([svd_pred, nmf_pred, content_pred, graph_pred, neural_pred])
            
            user_features = self._get_user_meta_features(user_id)
            content_features = self._get_content_meta_features(content_id)
            
            meta_feature = np.array(predictions + user_features + content_features).reshape(1, -1)
            
            ensemble_prediction = self.models_ensemble['ensemble_meta'].predict(meta_feature)[0]
            
            return max(0.5, min(5.0, ensemble_prediction))
            
        except Exception as e:
            return 3.5
    
    def get_ultra_personalized_recommendations(self, user_id, limit=20, context=None):
        try:
            user_profile = self._get_comprehensive_user_profile(user_id)
            
            all_content_ids = [content.id for content in self.models['Content'].query.all()]
            
            user_interactions = self.models['UserInteraction'].query.filter_by(user_id=user_id).all()
            interacted_content_ids = {inter.content_id for inter in user_interactions}
            
            candidate_content_ids = [cid for cid in all_content_ids if cid not in interacted_content_ids]
            
            if len(candidate_content_ids) > 1000:
                candidate_content_ids = np.random.choice(candidate_content_ids, 1000, replace=False).tolist()
            
            recommendations = []
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_content = {
                    executor.submit(self._score_content_for_user, user_id, content_id, context): content_id
                    for content_id in candidate_content_ids
                }
                
                for future in as_completed(future_to_content):
                    content_id = future_to_content[future]
                    try:
                        score_result = future.result()
                        if score_result:
                            recommendations.append(score_result)
                    except Exception as e:
                        logger.warning(f"Content scoring error for {content_id}: {e}")
            
            recommendations.sort(key=lambda x: x.score, reverse=True)
            
            recommendations = self._apply_advanced_ranking(recommendations, user_profile, context)
            
            recommendations = self._ensure_ultra_diversity(recommendations, user_profile)
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Ultra personalized recommendations error: {e}")
            return []
    
    def _score_content_for_user(self, user_id, content_id, context=None):
        try:
            algorithm_scores = {}
            
            algorithm_scores['svd'] = self._predict_svd(user_id, content_id)
            algorithm_scores['nmf'] = self._predict_nmf(user_id, content_id)
            algorithm_scores['content_based'] = self._predict_content_based(user_id, content_id)
            algorithm_scores['graph_based'] = self._predict_graph_based(user_id, content_id)
            algorithm_scores['neural_cf'] = self._predict_neural_cf(user_id, content_id)
            algorithm_scores['sequence_aware'] = self._predict_sequence_aware(user_id, content_id)
            algorithm_scores['contextual'] = self._predict_contextual_bandit(user_id, content_id, context)
            algorithm_scores['ensemble'] = self._predict_ensemble_meta(user_id, content_id, context)
            
            final_score = 0.0
            confidence_scores = []
            
            weights = {
                'svd': 0.15,
                'nmf': 0.12,
                'content_based': 0.15,
                'graph_based': 0.10,
                'neural_cf': 0.18,
                'sequence_aware': 0.08,
                'contextual': 0.07,
                'ensemble': 0.15
            }
            
            for algo, score in algorithm_scores.items():
                weight = weights.get(algo, 0.1)
                final_score += score * weight
                confidence_scores.append(abs(score - 3.5))
            
            confidence = 1.0 - (np.std(confidence_scores) / 2.5) if confidence_scores else 0.5
            
            novelty_score = self._calculate_novelty_score(user_id, content_id)
            diversity_score = self._calculate_diversity_score(user_id, content_id)
            
            final_score = (final_score * 0.7 + novelty_score * 0.15 + diversity_score * 0.15)
            
            reason_components = []
            top_algorithms = sorted(algorithm_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            for algo, score in top_algorithms:
                if score > 3.5:
                    reason_components.append(f"{algo}({score:.1f})")
            
            reason = "Multi-algorithm recommendation: " + ", ".join(reason_components) if reason_components else "Exploratory recommendation"
            
            return RecommendationResult(
                content_id=content_id,
                score=final_score,
                reason=reason,
                category="ultra_hybrid",
                confidence=confidence,
                source_algorithms=list(algorithm_scores.keys()),
                diversity_score=diversity_score,
                novelty_score=novelty_score
            )
            
        except Exception as e:
            logger.error(f"Content scoring error: {e}")
            return None
    
    def _calculate_novelty_score(self, user_id, content_id):
        try:
            content = self.models['Content'].query.get(content_id)
            if not content:
                return 0.5
            
            user_profile = self._get_comprehensive_user_profile(user_id)
            
            try:
                content_genres = set(json.loads(content.genres or '[]'))
                user_genres = set(user_profile['genre_preferences'].keys())
                
                genre_overlap = len(content_genres & user_genres) / max(len(content_genres), 1)
                novelty = 1.0 - genre_overlap
            except:
                novelty = 0.5
            
            if content.release_date:
                days_old = (datetime.now().date() - content.release_date).days
                recency_boost = max(0, (365 - days_old) / 365) if days_old < 365 else 0
                novelty += recency_boost * 0.3
            
            return min(1.0, novelty)
            
        except Exception as e:
            return 0.5
    
    def _calculate_diversity_score(self, user_id, content_id):
        try:
            content = self.models['Content'].query.get(content_id)
            if not content:
                return 0.5
            
            user_interactions = self.models['UserInteraction'].query.filter_by(user_id=user_id).all()
            
            if not user_interactions:
                return 1.0
            
            interacted_types = set()
            interacted_genres = set()
            
            for inter in user_interactions:
                inter_content = self.models['Content'].query.get(inter.content_id)
                if inter_content:
                    interacted_types.add(inter_content.content_type)
                    try:
                        genres = json.loads(inter_content.genres or '[]')
                        interacted_genres.update(genres)
                    except:
                        pass
            
            diversity_score = 0.0
            
            if content.content_type not in interacted_types:
                diversity_score += 0.4
            
            try:
                content_genres = set(json.loads(content.genres or '[]'))
                new_genres = content_genres - interacted_genres
                diversity_score += (len(new_genres) / max(len(content_genres), 1)) * 0.6
            except:
                diversity_score += 0.3
            
            return min(1.0, diversity_score)
            
        except Exception as e:
            return 0.5
    
    def _apply_advanced_ranking(self, recommendations, user_profile, context=None):
        try:
            current_hour = datetime.now().hour
            current_day = datetime.now().weekday()
            
            for rec in recommendations:
                content = self.models['Content'].query.get(rec.content_id)
                if not content:
                    continue
                
                time_boost = 1.0
                if context and 'preferred_time' in user_profile:
                    preferred_hour = user_profile['preferred_time']
                    time_diff = abs(current_hour - preferred_hour)
                    time_boost = max(0.7, 1.0 - (time_diff / 12.0))
                
                quality_boost = 1.0
                if content.rating and content.rating > user_profile.get('avg_rating', 3.5):
                    quality_boost = 1.0 + min(0.3, (content.rating - user_profile['avg_rating']) / 5.0)
                
                popularity_adjustment = 1.0
                user_popular_affinity = user_profile.get('popular_affinity', 0.5)
                content_popularity = min(1.0, (content.popularity or 0) / 1000.0)
                
                if user_popular_affinity > 0.7 and content_popularity > 0.5:
                    popularity_adjustment = 1.1
                elif user_popular_affinity < 0.3 and content_popularity < 0.3:
                    popularity_adjustment = 1.1
                
                exploration_boost = 1.0
                if user_profile.get('exploration_tendency', 0.5) > 0.7:
                    exploration_boost = 1.0 + rec.novelty_score * 0.2
                
                rec.score *= time_boost * quality_boost * popularity_adjustment * exploration_boost
            
            recommendations.sort(key=lambda x: x.score, reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Advanced ranking error: {e}")
            return recommendations
    
    def _ensure_ultra_diversity(self, recommendations, user_profile):
        try:
            if not recommendations:
                return recommendations
            
            exploration_tendency = user_profile.get('exploration_tendency', 0.5)
            
            if exploration_tendency > 0.7:
                target_ratio = {'movie': 0.3, 'tv': 0.3, 'anime': 0.4}
            elif exploration_tendency < 0.3:
                user_type_prefs = user_profile.get('content_type_preferences', {})
                target_ratio = {k: v for k, v in user_type_prefs.items()}
                if not target_ratio:
                    target_ratio = {'movie': 0.4, 'tv': 0.35, 'anime': 0.25}
            else:
                target_ratio = {'movie': 0.4, 'tv': 0.35, 'anime': 0.25}
            
            grouped = defaultdict(list)
            for rec in recommendations:
                try:
                    content = self.models['Content'].query.get(rec.content_id)
                    if content:
                        grouped[content.content_type].append(rec)
                except:
                    grouped['unknown'].append(rec)
            
            total_needed = len(recommendations)
            diversified = []
            
            for content_type, ratio in target_ratio.items():
                needed = int(total_needed * ratio)
                available = grouped.get(content_type, [])
                available.sort(key=lambda x: x.score, reverse=True)
                diversified.extend(available[:needed])
            
            used_ids = {rec.content_id for rec in diversified}
            remaining = [rec for rec in recommendations if rec.content_id not in used_ids]
            remaining.sort(key=lambda x: x.score, reverse=True)
            
            slots_left = total_needed - len(diversified)
            diversified.extend(remaining[:slots_left])
            
            diversified.sort(key=lambda x: x.score, reverse=True)
            
            return diversified[:total_needed]
            
        except Exception as e:
            logger.error(f"Ultra diversity ensuring error: {e}")
            return recommendations
    
    def update_user_interaction(self, user_id, content_id, interaction_type, rating=None):
        try:
            cache_keys = [
                f"ultra_profile:{user_id}",
                f"user_profile:{user_id}",
                f"recommendations:{user_id}",
                f"personalized:{user_id}"
            ]
            
            if self.cache:
                for key in cache_keys:
                    self.cache.delete(key)
            
            self._schedule_model_update()
            
        except Exception as e:
            logger.error(f"Interaction update error: {e}")
    
    def _schedule_model_update(self):
        self.needs_update = True
    
    def _continuous_learning_loop(self):
        while True:
            try:
                time.sleep(1800)
                
                if hasattr(self, 'needs_update') and self.needs_update:
                    logger.info("Starting continuous learning update...")
                    
                    self._build_user_item_matrices()
                    self._train_matrix_factorization_models()
                    self._build_user_segments()
                    self._compute_similarity_matrices()
                    
                    self.needs_update = False
                    logger.info("Continuous learning update completed")
                
            except Exception as e:
                logger.error(f"Continuous learning error: {e}")
                time.sleep(300)

personalization_engine = None

def init_personalized(flask_app, database, model_dict, service_dict, cache_instance):
    global app, db, cache, models, services, personalization_engine
    global User, Content, UserInteraction, UserBehaviorProfile, ContentFeatures, RecommendationModel, UserSession, SearchHistory
    
    app = flask_app
    db = database
    cache = cache_instance
    models = model_dict
    services = service_dict
    
    User = models['User']
    Content = models['Content']
    UserInteraction = models['UserInteraction']
    
    _create_personalization_tables()
    
    personalization_engine = UltraAdvancedPersonalizationEngine(
        app, db, models, services, cache
    )
    
    logger.info("Ultra-advanced personalized recommendation service initialized")

def _create_personalization_tables():
    global UserBehaviorProfile, ContentFeatures, RecommendationModel, UserSession, SearchHistory
    
    try:
        class UserBehaviorProfile(db.Model):
            __tablename__ = 'user_behavior_profiles'
            
            id = db.Column(db.Integer, primary_key=True)
            user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
            
            genre_preferences = db.Column(db.JSON)
            language_preferences = db.Column(db.JSON)
            content_type_preferences = db.Column(db.JSON)
            time_preferences = db.Column(db.JSON)
            
            avg_rating = db.Column(db.Float, default=3.5)
            rating_variance = db.Column(db.Float, default=1.0)
            diversity_score = db.Column(db.Float, default=0.5)
            activity_level = db.Column(db.String(20), default='medium')
            
            preferred_viewing_time = db.Column(db.Integer)
            session_duration_avg = db.Column(db.Float)
            
            exploration_tendency = db.Column(db.Float, default=0.5)
            popular_content_affinity = db.Column(db.Float, default=0.5)
            niche_content_affinity = db.Column(db.Float, default=0.5)
            
            created_at = db.Column(db.DateTime, default=datetime.utcnow)
            updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        class ContentFeatures(db.Model):
            __tablename__ = 'content_features'
            
            id = db.Column(db.Integer, primary_key=True)
            content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False, unique=True)
            
            sentiment_polarity = db.Column(db.Float)
            sentiment_subjectivity = db.Column(db.Float)
            complexity_score = db.Column(db.Float)
            themes = db.Column(db.JSON)
            keywords = db.Column(db.JSON)
            
            genre_embedding = db.Column(db.JSON)
            content_embedding = db.Column(db.JSON)
            popularity_tier = db.Column(db.String(20))
            quality_score = db.Column(db.Float)
            
            avg_user_rating = db.Column(db.Float)
            total_interactions = db.Column(db.Integer, default=0)
            favorite_ratio = db.Column(db.Float, default=0.0)
            completion_rate = db.Column(db.Float, default=0.0)
            
            created_at = db.Column(db.DateTime, default=datetime.utcnow)
            updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        class RecommendationModel(db.Model):
            __tablename__ = 'recommendation_models'
            
            id = db.Column(db.Integer, primary_key=True)
            model_name = db.Column(db.String(100), nullable=False)
            model_version = db.Column(db.String(50), nullable=False)
            model_data = db.Column(db.LargeBinary)
            performance_metrics = db.Column(db.JSON)
            
            is_active = db.Column(db.Boolean, default=False)
            created_at = db.Column(db.DateTime, default=datetime.utcnow)
        
        class UserSession(db.Model):
            __tablename__ = 'user_sessions'
            
            id = db.Column(db.Integer, primary_key=True)
            user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
            session_id = db.Column(db.String(100), nullable=False)
            
            start_time = db.Column(db.DateTime, default=datetime.utcnow)
            end_time = db.Column(db.DateTime)
            duration_minutes = db.Column(db.Float)
            
            pages_viewed = db.Column(db.Integer, default=0)
            content_viewed = db.Column(db.JSON)
            searches_performed = db.Column(db.Integer, default=0)
            interactions_count = db.Column(db.Integer, default=0)
            
            device_type = db.Column(db.String(50))
            location_info = db.Column(db.JSON)
            referrer = db.Column(db.String(255))
        
        class SearchHistory(db.Model):
            __tablename__ = 'search_history'
            
            id = db.Column(db.Integer, primary_key=True)
            user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
            session_id = db.Column(db.String(100))
            
            query = db.Column(db.String(500), nullable=False)
            query_processed = db.Column(db.String(500))
            results_count = db.Column(db.Integer)
            
            intent = db.Column(db.String(50))
            entities_extracted = db.Column(db.JSON)
            sentiment = db.Column(db.Float)
            
            clicked_content_ids = db.Column(db.JSON)
            time_to_click = db.Column(db.Float)
            result_satisfaction = db.Column(db.Float)
            
            timestamp = db.Column(db.DateTime, default=datetime.utcnow)
        
        models['UserBehaviorProfile'] = UserBehaviorProfile
        models['ContentFeatures'] = ContentFeatures
        models['RecommendationModel'] = RecommendationModel
        models['UserSession'] = UserSession
        models['SearchHistory'] = SearchHistory
        
        db.create_all()
        
        logger.info("Ultra-personalization tables created successfully")
        
    except Exception as e:
        logger.error(f"Error creating personalization tables: {e}")

def get_personalization_engine():
    return personalization_engine