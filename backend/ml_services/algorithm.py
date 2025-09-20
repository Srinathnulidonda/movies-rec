import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial.distance import jaccard, hamming
from scipy.stats import pearsonr, spearmanr
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
import logging
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

logger = logging.getLogger(__name__)

class AdvancedCollaborativeFiltering:
    def __init__(self, db, models, config=None):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
        self.config = config or {
            'min_interactions': 3,
            'similarity_threshold': 0.1,
            'neighborhood_size': 50,
            'regularization': 0.01,
            'learning_rate': 0.01,
            'epochs': 100,
            'use_bias': True,
            'use_temporal_decay': True,
            'temporal_decay_factor': 0.95
        }
        
        self.user_similarity_cache = {}
        self.item_similarity_cache = {}
        self.user_item_matrix = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        self.global_mean = 0.0
        self.user_biases = {}
        self.item_biases = {}
        self.last_matrix_update = None
        self._lock = threading.Lock()
        
    def _get_interaction_weight(self, interaction):
        """Advanced interaction weighting with temporal decay and interaction type preferences"""
        base_weights = {
            'view': 1.0,
            'like': 3.0,
            'favorite': 5.0,
            'watchlist': 4.0,
            'rating': 0.0,  # Will be handled separately
            'search': 0.5,
            'share': 2.0,
            'download': 3.5,
            'complete_watch': 4.5
        }
        
        base_weight = base_weights.get(interaction.interaction_type, 1.0)
        
        # Handle ratings specially
        if interaction.interaction_type == 'rating' and interaction.rating:
            base_weight = max(0.1, interaction.rating / 2.0)  # Scale 1-10 to 0.1-5.0
        
        # Apply temporal decay
        if self.config['use_temporal_decay']:
            days_ago = (datetime.utcnow() - interaction.timestamp).days
            temporal_factor = self.config['temporal_decay_factor'] ** (days_ago / 30.0)
            base_weight *= temporal_factor
        
        # Apply user activity normalization
        user_interaction_count = self.UserInteraction.query.filter_by(user_id=interaction.user_id).count()
        activity_factor = math.log(user_interaction_count + 1) / math.log(100)  # Normalize heavy users
        base_weight *= min(1.0, activity_factor)
        
        return base_weight
    
    def build_user_item_matrix(self, force_rebuild=False):
        """Build optimized sparse user-item matrix with advanced preprocessing"""
        with self._lock:
            if (self.user_item_matrix is not None and 
                self.last_matrix_update and 
                not force_rebuild and
                (datetime.utcnow() - self.last_matrix_update).seconds < 3600):
                return
            
            logger.info("Building advanced user-item matrix...")
            
            # Get all interactions
            interactions = self.UserInteraction.query.all()
            
            if not interactions:
                logger.warning("No interactions found for matrix building")
                return
            
            # Build mappings
            user_ids = list(set(interaction.user_id for interaction in interactions))
            item_ids = list(set(interaction.content_id for interaction in interactions))
            
            self.user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
            self.item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
            self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
            self.idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}
            
            n_users, n_items = len(user_ids), len(item_ids)
            
            # Use sparse matrix for memory efficiency
            matrix = lil_matrix((n_users, n_items))
            
            # Fill matrix with weighted interactions
            for interaction in interactions:
                user_idx = self.user_to_idx[interaction.user_id]
                item_idx = self.item_to_idx[interaction.content_id]
                weight = self._get_interaction_weight(interaction)
                
                # Accumulate weights for multiple interactions
                matrix[user_idx, item_idx] += weight
            
            # Convert to CSR for efficient operations
            self.user_item_matrix = matrix.tocsr()
            
            # Calculate global statistics
            self.global_mean = self.user_item_matrix.data.mean()
            
            # Calculate user and item biases
            if self.config['use_bias']:
                self._calculate_biases()
            
            self.last_matrix_update = datetime.utcnow()
            logger.info(f"Matrix built: {n_users} users, {n_items} items, density: {self.user_item_matrix.nnz / (n_users * n_items):.4f}")
    
    def _calculate_biases(self):
        """Calculate user and item biases for better prediction accuracy"""
        self.user_biases = {}
        self.item_biases = {}
        
        # User biases (deviation from global mean)
        for user_id, user_idx in self.user_to_idx.items():
            user_ratings = self.user_item_matrix[user_idx].data
            if len(user_ratings) > 0:
                self.user_biases[user_id] = user_ratings.mean() - self.global_mean
            else:
                self.user_biases[user_id] = 0.0
        
        # Item biases (deviation from global mean)
        for item_id, item_idx in self.item_to_idx.items():
            item_ratings = self.user_item_matrix[:, item_idx].data
            if len(item_ratings) > 0:
                self.item_biases[item_id] = item_ratings.mean() - self.global_mean
            else:
                self.item_biases[item_id] = 0.0
    
    def calculate_user_similarity_advanced(self, user_id, similarity_metric='cosine'):
        """Advanced user similarity calculation with multiple metrics"""
        if user_id not in self.user_to_idx:
            return {}
        
        cache_key = f"{user_id}_{similarity_metric}"
        if cache_key in self.user_similarity_cache:
            return self.user_similarity_cache[cache_key]
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_item_matrix[user_idx]
        
        similarities = {}
        
        if similarity_metric == 'cosine':
            # Cosine similarity
            sim_matrix = cosine_similarity(user_vector, self.user_item_matrix)
            similarities = {
                self.idx_to_user[idx]: sim_matrix[0, idx]
                for idx in range(len(self.idx_to_user))
                if idx != user_idx and sim_matrix[0, idx] > self.config['similarity_threshold']
            }
        
        elif similarity_metric == 'pearson':
            # Pearson correlation
            user_data = user_vector.toarray().flatten()
            for other_user_id, other_idx in self.user_to_idx.items():
                if other_user_id != user_id:
                    other_data = self.user_item_matrix[other_idx].toarray().flatten()
                    
                    # Find common items
                    common_mask = (user_data > 0) & (other_data > 0)
                    if common_mask.sum() >= self.config['min_interactions']:
                        corr, _ = pearsonr(user_data[common_mask], other_data[common_mask])
                        if not np.isnan(corr) and corr > self.config['similarity_threshold']:
                            similarities[other_user_id] = corr
        
        elif similarity_metric == 'jaccard':
            # Jaccard similarity for binary interactions
            user_items = set(user_vector.nonzero()[1])
            for other_user_id, other_idx in self.user_to_idx.items():
                if other_user_id != user_id:
                    other_items = set(self.user_item_matrix[other_idx].nonzero()[1])
                    
                    if len(user_items) > 0 and len(other_items) > 0:
                        intersection = len(user_items & other_items)
                        union = len(user_items | other_items)
                        jaccard_sim = intersection / union if union > 0 else 0.0
                        
                        if jaccard_sim > self.config['similarity_threshold']:
                            similarities[other_user_id] = jaccard_sim
        
        # Sort by similarity and keep top neighbors
        similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:self.config['neighborhood_size']])
        
        self.user_similarity_cache[cache_key] = similarities
        return similarities
    
    def user_based_recommendations_advanced(self, user_id, limit=20, diversity_factor=0.2):
        """Advanced user-based collaborative filtering with diversity"""
        self.build_user_item_matrix()
        
        if user_id not in self.user_to_idx:
            return []
        
        # Get similar users using multiple similarity metrics
        cosine_similarities = self.calculate_user_similarity_advanced(user_id, 'cosine')
        pearson_similarities = self.calculate_user_similarity_advanced(user_id, 'pearson')
        
        # Combine similarities with weights
        combined_similarities = defaultdict(float)
        for user, sim in cosine_similarities.items():
            combined_similarities[user] += sim * 0.6
        for user, sim in pearson_similarities.items():
            combined_similarities[user] += sim * 0.4
        
        # Get user's interactions
        user_idx = self.user_to_idx[user_id]
        user_interactions = set(self.user_item_matrix[user_idx].nonzero()[1])
        
        # Calculate recommendation scores
        recommendations = defaultdict(float)
        total_similarity = defaultdict(float)
        
        for similar_user_id, similarity in combined_similarities.items():
            if similarity <= 0:
                continue
            
            similar_user_idx = self.user_to_idx[similar_user_id]
            similar_user_items = self.user_item_matrix[similar_user_idx].nonzero()[1]
            
            for item_idx in similar_user_items:
                if item_idx not in user_interactions:
                    item_id = self.idx_to_item[item_idx]
                    rating = self.user_item_matrix[similar_user_idx, item_idx]
                    
                    # Apply bias correction
                    if self.config['use_bias']:
                        predicted_rating = (self.global_mean + 
                                          self.user_biases.get(user_id, 0) + 
                                          self.item_biases.get(item_id, 0))
                        bias_corrected_rating = rating - predicted_rating
                    else:
                        bias_corrected_rating = rating - self.global_mean
                    
                    recommendations[item_id] += similarity * bias_corrected_rating
                    total_similarity[item_id] += abs(similarity)
        
        # Normalize recommendations
        for item_id in recommendations:
            if total_similarity[item_id] > 0:
                recommendations[item_id] /= total_similarity[item_id]
                
                # Add bias back
                if self.config['use_bias']:
                    recommendations[item_id] += (self.global_mean + 
                                               self.user_biases.get(user_id, 0) + 
                                               self.item_biases.get(item_id, 0))
        
        # Sort and return top recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Apply diversity if requested
        if diversity_factor > 0:
            sorted_recommendations = self._apply_diversity_filter(sorted_recommendations, diversity_factor, limit)
        
        return sorted_recommendations[:limit]
    
    def _apply_diversity_filter(self, recommendations, diversity_factor, limit):
        """Apply diversity filtering to recommendations"""
        if not recommendations:
            return recommendations
        
        final_recommendations = []
        remaining_recommendations = recommendations[:]
        
        # Add the top recommendation first
        final_recommendations.append(remaining_recommendations.pop(0))
        
        while len(final_recommendations) < limit and remaining_recommendations:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for idx, (item_id, score) in enumerate(remaining_recommendations):
                # Calculate diversity score
                diversity_score = self._calculate_item_diversity(item_id, [rec[0] for rec in final_recommendations])
                
                # Combine relevance and diversity
                combined_score = (1 - diversity_factor) * score + diversity_factor * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = (item_id, score)
                    best_idx = idx
            
            if best_candidate:
                final_recommendations.append(best_candidate)
                remaining_recommendations.pop(best_idx)
            else:
                break
        
        return final_recommendations
    
    def _calculate_item_diversity(self, item_id, selected_items):
        """Calculate diversity score for an item against selected items"""
        if not selected_items:
            return 1.0
        
        item = self.Content.query.get(item_id)
        if not item:
            return 0.0
        
        try:
            item_genres = set(json.loads(item.genres or '[]'))
            item_languages = set(json.loads(item.languages or '[]'))
        except:
            return 0.5
        
        diversity_scores = []
        
        for selected_id in selected_items:
            selected_item = self.Content.query.get(selected_id)
            if selected_item:
                try:
                    selected_genres = set(json.loads(selected_item.genres or '[]'))
                    selected_languages = set(json.loads(selected_item.languages or '[]'))
                    
                    # Genre diversity
                    genre_overlap = len(item_genres & selected_genres) / max(len(item_genres | selected_genres), 1)
                    genre_diversity = 1 - genre_overlap
                    
                    # Language diversity
                    lang_overlap = len(item_languages & selected_languages) / max(len(item_languages | selected_languages), 1)
                    lang_diversity = 1 - lang_overlap
                    
                    # Content type diversity
                    type_diversity = 0.0 if item.content_type == selected_item.content_type else 1.0
                    
                    # Combined diversity
                    combined_diversity = (0.5 * genre_diversity + 0.3 * lang_diversity + 0.2 * type_diversity)
                    diversity_scores.append(combined_diversity)
                except:
                    diversity_scores.append(0.5)
        
        return np.mean(diversity_scores) if diversity_scores else 1.0

class AdvancedContentBasedFiltering:
    def __init__(self, db, models, config=None):
        self.db = db
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
        self.config = config or {
            'text_weight': 0.3,
            'genre_weight': 0.25,
            'language_weight': 0.15,
            'metadata_weight': 0.15,
            'popularity_weight': 0.1,
            'temporal_weight': 0.05,
            'tfidf_max_features': 10000,
            'min_df': 2,
            'max_df': 0.8
        }
        
        self.content_features_cache = {}
        self.tfidf_vectorizer = None
        self.content_tfidf_matrix = None
        self.feature_weights = None
        self._lock = threading.Lock()
        
    def _initialize_tfidf(self):
        """Initialize and fit TF-IDF vectorizer on all content"""
        if self.tfidf_vectorizer is not None:
            return
        
        with self._lock:
            logger.info("Initializing advanced TF-IDF vectorizer...")
            
            contents = self.Content.query.all()
            documents = []
            
            for content in contents:
                # Combine all text features
                text_features = []
                
                if content.overview:
                    text_features.append(content.overview)
                
                if content.title:
                    # Add title multiple times for higher weight
                    text_features.extend([content.title] * 3)
                
                try:
                    genres = json.loads(content.genres or '[]')
                    text_features.extend(genres)
                except:
                    pass
                
                try:
                    languages = json.loads(content.languages or '[]')
                    text_features.extend(languages)
                except:
                    pass
                
                # Add content type
                text_features.append(content.content_type)
                
                documents.append(' '.join(text_features))
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config['tfidf_max_features'],
                min_df=self.config['min_df'],
                max_df=self.config['max_df'],
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True,
                strip_accents='unicode'
            )
            
            self.content_tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            logger.info(f"TF-IDF initialized: {self.content_tfidf_matrix.shape}")
    
    def extract_advanced_features(self, content_id):
        """Extract comprehensive content features"""
        if content_id in self.content_features_cache:
            return self.content_features_cache[content_id]
        
        content = self.Content.query.get(content_id)
        if not content:
            return None
        
        features = {
            'genres': set(),
            'languages': set(),
            'content_type': content.content_type,
            'rating': content.rating or 0,
            'popularity': content.popularity or 0,
            'runtime': content.runtime or 0,
            'release_year': 0,
            'vote_count': content.vote_count or 0,
            'is_trending': getattr(content, 'is_trending', False),
            'is_new_release': getattr(content, 'is_new_release', False),
            'overview': content.overview or '',
            'title': content.title or ''
        }
        
        # Parse JSON fields safely
        try:
            features['genres'] = set(json.loads(content.genres or '[]'))
        except:
            pass
        
        try:
            features['languages'] = set(json.loads(content.languages or '[]'))
        except:
            pass
        
        if content.release_date:
            features['release_year'] = content.release_date.year
            # Calculate age factor
            current_year = datetime.now().year
            age = current_year - features['release_year']
            features['age_factor'] = max(0, 1 - age / 50.0)  # Normalize age
        else:
            features['age_factor'] = 0.5
        
        # Calculate popularity percentile
        if content.popularity:
            max_popularity = self.db.session.query(self.db.func.max(self.Content.popularity)).scalar() or 1
            features['popularity_percentile'] = content.popularity / max_popularity
        else:
            features['popularity_percentile'] = 0.0
        
        # Calculate rating quality score
        if content.rating and content.vote_count:
            # Weighted rating based on vote count
            min_votes = 10
            features['weighted_rating'] = ((content.vote_count / (content.vote_count + min_votes)) * content.rating + 
                                         (min_votes / (content.vote_count + min_votes)) * 6.0)
        else:
            features['weighted_rating'] = 0.0
        
        self.content_features_cache[content_id] = features
        return features
    
    def calculate_content_similarity_advanced(self, content_id1, content_id2):
        """Advanced multi-feature content similarity"""
        features1 = self.extract_advanced_features(content_id1)
        features2 = self.extract_advanced_features(content_id2)
        
        if not features1 or not features2:
            return 0.0
        
        similarity_components = []
        
        # 1. Genre similarity (Jaccard + weighted)
        if features1['genres'] or features2['genres']:
            genre_jaccard = len(features1['genres'] & features2['genres']) / max(len(features1['genres'] | features2['genres']), 1)
            similarity_components.append(('genre', genre_jaccard, self.config['genre_weight']))
        
        # 2. Language similarity
        if features1['languages'] or features2['languages']:
            lang_jaccard = len(features1['languages'] & features2['languages']) / max(len(features1['languages'] | features2['languages']), 1)
            similarity_components.append(('language', lang_jaccard, self.config['language_weight']))
        
        # 3. Content type similarity
        type_sim = 1.0 if features1['content_type'] == features2['content_type'] else 0.0
        similarity_components.append(('type', type_sim, 0.1))
        
        # 4. Rating similarity
        rating_diff = abs(features1['weighted_rating'] - features2['weighted_rating'])
        rating_sim = max(0, 1 - rating_diff / 10.0)
        similarity_components.append(('rating', rating_sim, 0.1))
        
        # 5. Temporal similarity
        year_diff = abs(features1['release_year'] - features2['release_year'])
        year_sim = max(0, 1 - year_diff / 20.0)
        similarity_components.append(('temporal', year_sim, self.config['temporal_weight']))
        
        # 6. Popularity similarity
        pop_diff = abs(features1['popularity_percentile'] - features2['popularity_percentile'])
        pop_sim = max(0, 1 - pop_diff)
        similarity_components.append(('popularity', pop_sim, self.config['popularity_weight']))
        
        # 7. Text similarity (TF-IDF)
        if features1['overview'] and features2['overview']:
            self._initialize_tfidf()
            try:
                # Get content indices
                content1_idx = None
                content2_idx = None
                
                all_contents = self.Content.query.all()
                for idx, content in enumerate(all_contents):
                    if content.id == content_id1:
                        content1_idx = idx
                    elif content.id == content_id2:
                        content2_idx = idx
                
                if content1_idx is not None and content2_idx is not None:
                    text_sim = cosine_similarity(
                        self.content_tfidf_matrix[content1_idx:content1_idx+1],
                        self.content_tfidf_matrix[content2_idx:content2_idx+1]
                    )[0][0]
                    similarity_components.append(('text', text_sim, self.config['text_weight']))
            except:
                # Fallback to simple text comparison
                text1_words = set(features1['overview'].lower().split())
                text2_words = set(features2['overview'].lower().split())
                text_sim = len(text1_words & text2_words) / max(len(text1_words | text2_words), 1)
                similarity_components.append(('text', text_sim, self.config['text_weight']))
        
        # Calculate weighted similarity
        total_weight = sum(weight for _, _, weight in similarity_components)
        if total_weight == 0:
            return 0.0
        
        weighted_similarity = sum(sim * weight for _, sim, weight in similarity_components) / total_weight
        
        return min(1.0, max(0.0, weighted_similarity))
    
    def get_content_recommendations_advanced(self, user_id, limit=20, diversity_factor=0.3):
        """Advanced content-based recommendations with user profile learning"""
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not user_interactions:
            return []
        
        # Build user profile from interactions
        user_profile = self._build_user_profile(user_interactions)
        
        # Get candidate items (not interacted with)
        user_seen_content = set(interaction.content_id for interaction in user_interactions)
        all_content = self.Content.query.filter(~self.Content.id.in_(user_seen_content)).all()
        
        recommendations = []
        
        # Score each candidate item
        for content in all_content:
            score = self._calculate_content_score(content, user_profile)
            if score > 0:
                recommendations.append((content.id, score))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity if requested
        if diversity_factor > 0:
            recommendations = self._apply_content_diversity(recommendations, diversity_factor, limit)
        
        return recommendations[:limit]
    
    def _build_user_profile(self, user_interactions):
        """Build comprehensive user profile from interactions"""
        profile = {
            'genres': Counter(),
            'languages': Counter(),
            'content_types': Counter(),
            'avg_rating': 0.0,
            'rating_variance': 0.0,
            'preferred_year_range': [0, 0],
            'popularity_preference': 0.0,
            'text_preferences': [],
            'interaction_weights': Counter()
        }
        
        ratings = []
        years = []
        popularities = []
        
        for interaction in user_interactions:
            content = self.Content.query.get(interaction.content_id)
            if not content:
                continue
            
            weight = self._get_interaction_weight(interaction)
            profile['interaction_weights'][interaction.interaction_type] += weight
            
            # Genre preferences
            try:
                genres = json.loads(content.genres or '[]')
                for genre in genres:
                    profile['genres'][genre] += weight
            except:
                pass
            
            # Language preferences
            try:
                languages = json.loads(content.languages or '[]')
                for language in languages:
                    profile['languages'][language] += weight
            except:
                pass
            
            # Content type preferences
            profile['content_types'][content.content_type] += weight
            
            # Rating analysis
            if interaction.rating:
                ratings.extend([interaction.rating] * int(weight))
            elif content.rating:
                ratings.extend([content.rating] * int(weight))
            
            # Year preferences
            if content.release_date:
                years.extend([content.release_date.year] * int(weight))
            
            # Popularity preferences
            if content.popularity:
                popularities.extend([content.popularity] * int(weight))
            
            # Text preferences (extract keywords from liked content)
            if weight >= 2.0 and content.overview:  # Only from positive interactions
                profile['text_preferences'].append(content.overview)
        
        # Calculate statistics
        if ratings:
            profile['avg_rating'] = np.mean(ratings)
            profile['rating_variance'] = np.var(ratings)
        
        if years:
            profile['preferred_year_range'] = [np.percentile(years, 25), np.percentile(years, 75)]
        
        if popularities:
            profile['popularity_preference'] = np.mean(popularities)
        
        return profile
    
    def _calculate_content_score(self, content, user_profile):
        """Calculate content score based on user profile"""
        score_components = []
        
        # Genre matching
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            genre_score = 0.0
            total_genre_weight = sum(user_profile['genres'].values())
            
            if total_genre_weight > 0:
                for genre in content_genres:
                    genre_score += user_profile['genres'].get(genre, 0) / total_genre_weight
                
                genre_score /= max(len(content_genres), 1)  # Normalize by number of genres
            
            score_components.append(('genre', genre_score, 0.3))
        except:
            score_components.append(('genre', 0.0, 0.3))
        
        # Language matching
        try:
            content_languages = set(json.loads(content.languages or '[]'))
            language_score = 0.0
            total_lang_weight = sum(user_profile['languages'].values())
            
            if total_lang_weight > 0:
                for language in content_languages:
                    language_score += user_profile['languages'].get(language, 0) / total_lang_weight
                
                language_score /= max(len(content_languages), 1)
            
            score_components.append(('language', language_score, 0.2))
        except:
            score_components.append(('language', 0.0, 0.2))
        
        # Content type matching
        type_score = 0.0
        total_type_weight = sum(user_profile['content_types'].values())
        if total_type_weight > 0:
            type_score = user_profile['content_types'].get(content.content_type, 0) / total_type_weight
        
        score_components.append(('type', type_score, 0.15))
        
        # Rating compatibility
        rating_score = 0.0
        if content.rating and user_profile['avg_rating'] > 0:
            rating_diff = abs(content.rating - user_profile['avg_rating'])
            rating_score = max(0, 1 - rating_diff / 5.0)  # Normalize to 0-1
        
        score_components.append(('rating', rating_score, 0.15))
        
        # Year preference
        year_score = 0.0
        if content.release_date and user_profile['preferred_year_range'][1] > 0:
            year = content.release_date.year
            min_year, max_year = user_profile['preferred_year_range']
            
            if min_year <= year <= max_year:
                year_score = 1.0
            else:
                # Gradual decline outside preferred range
                distance = min(abs(year - min_year), abs(year - max_year))
                year_score = max(0, 1 - distance / 20.0)
        
        score_components.append(('year', year_score, 0.1))
        
        # Popularity matching
        popularity_score = 0.0
        if content.popularity and user_profile['popularity_preference'] > 0:
            pop_ratio = content.popularity / user_profile['popularity_preference']
            popularity_score = 1.0 / (1.0 + abs(1.0 - pop_ratio))  # Closer to 1.0 is better
        
        score_components.append(('popularity', popularity_score, 0.1))
        
        # Calculate weighted score
        total_weight = sum(weight for _, _, weight in score_components)
        final_score = sum(score * weight for _, score, weight in score_components) / total_weight
        
        return final_score
    
    def _get_interaction_weight(self, interaction):
        """Get interaction weight for content-based filtering"""
        weights = {
            'view': 1.0,
            'like': 2.5,
            'favorite': 4.0,
            'watchlist': 3.0,
            'rating': interaction.rating * 0.4 if interaction.rating else 2.0,
            'search': 0.5,
            'share': 2.0
        }
        
        base_weight = weights.get(interaction.interaction_type, 1.0)
        
        # Apply temporal decay
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        temporal_factor = 0.95 ** (days_ago / 30.0)
        
        return base_weight * temporal_factor
    
    def _apply_content_diversity(self, recommendations, diversity_factor, limit):
        """Apply diversity to content-based recommendations"""
        if not recommendations or len(recommendations) <= limit:
            return recommendations
        
        final_recommendations = []
        remaining = recommendations[:]
        
        # Add top recommendation
        final_recommendations.append(remaining.pop(0))
        
        while len(final_recommendations) < limit and remaining:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for idx, (content_id, score) in enumerate(remaining):
                # Calculate diversity
                diversity = self._calculate_content_diversity(
                    content_id, 
                    [rec[0] for rec in final_recommendations]
                )
                
                # Combine relevance and diversity
                combined_score = (1 - diversity_factor) * score + diversity_factor * diversity
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = (content_id, score)
                    best_idx = idx
            
            if best_candidate:
                final_recommendations.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break
        
        return final_recommendations
    
    def _calculate_content_diversity(self, content_id, selected_content_ids):
        """Calculate diversity score for content against selected content"""
        if not selected_content_ids:
            return 1.0
        
        content = self.Content.query.get(content_id)
        if not content:
            return 0.0
        
        diversity_scores = []
        
        for selected_id in selected_content_ids:
            similarity = self.calculate_content_similarity_advanced(content_id, selected_id)
            diversity = 1.0 - similarity
            diversity_scores.append(diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 1.0

class AdvancedMatrixFactorization:
    def __init__(self, db, models, config=None):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
        self.config = config or {
            'n_factors': 100,
            'learning_rate': 0.005,
            'regularization': 0.02,
            'epochs': 200,
            'early_stopping_patience': 10,
            'min_improvement': 0.001,
            'use_bias': True,
            'use_temporal': True,
            'batch_size': 1000
        }
        
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_item = {}
        self.training_history = []
        self.is_trained = False
        self._lock = threading.Lock()
    
    def prepare_training_data(self):
        """Prepare and optimize training data"""
        interactions = self.UserInteraction.query.all()
        
        if not interactions:
            return None, None, None, None
        
        # Build user-item rating matrix
        user_item_pairs = []
        ratings = []
        timestamps = []
        
        for interaction in interactions:
            weight = self._get_interaction_weight(interaction)
            user_item_pairs.append((interaction.user_id, interaction.content_id))
            ratings.append(weight)
            timestamps.append(interaction.timestamp)
        
        # Create mappings
        unique_users = list(set(pair[0] for pair in user_item_pairs))
        unique_items = list(set(pair[1] for pair in user_item_pairs))
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Convert to indices
        user_indices = [self.user_to_idx[pair[0]] for pair in user_item_pairs]
        item_indices = [self.item_to_idx[pair[1]] for pair in user_item_pairs]
        
        return np.array(user_indices), np.array(item_indices), np.array(ratings), timestamps
    
    def train_advanced(self):
        """Advanced matrix factorization training with regularization and early stopping"""
        with self._lock:
            logger.info("Starting advanced matrix factorization training...")
            
            user_indices, item_indices, ratings, timestamps = self.prepare_training_data()
            
            if user_indices is None:
                logger.warning("No training data available")
                return False
            
            n_users = len(self.user_to_idx)
            n_items = len(self.item_to_idx)
            n_factors = self.config['n_factors']
            
            # Initialize factors with small random values
            self.user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
            self.item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
            
            if self.config['use_bias']:
                self.user_biases = np.zeros(n_users)
                self.item_biases = np.zeros(n_items)
                self.global_bias = np.mean(ratings)
            
            # Training loop with early stopping
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config['epochs']):
                epoch_loss = 0.0
                
                # Shuffle training data
                indices = np.random.permutation(len(user_indices))
                
                # Mini-batch training
                for start_idx in range(0, len(indices), self.config['batch_size']):
                    end_idx = min(start_idx + self.config['batch_size'], len(indices))
                    batch_indices = indices[start_idx:end_idx]
                    
                    batch_users = user_indices[batch_indices]
                    batch_items = item_indices[batch_indices]
                    batch_ratings = ratings[batch_indices]
                    
                    # Calculate predictions and errors
                    predictions = self._predict_batch(batch_users, batch_items)
                    errors = batch_ratings - predictions
                    
                    # Update factors using gradient descent
                    for i, (user_idx, item_idx, error) in enumerate(zip(batch_users, batch_items, errors)):
                        # Store current factors for update
                        user_factors_old = self.user_factors[user_idx].copy()
                        
                        # Update factors
                        self.user_factors[user_idx] += self.config['learning_rate'] * (
                            error * self.item_factors[item_idx] - 
                            self.config['regularization'] * self.user_factors[user_idx]
                        )
                        
                        self.item_factors[item_idx] += self.config['learning_rate'] * (
                            error * user_factors_old - 
                            self.config['regularization'] * self.item_factors[item_idx]
                        )
                        
                        # Update biases
                        if self.config['use_bias']:
                            self.user_biases[user_idx] += self.config['learning_rate'] * (
                                error - self.config['regularization'] * self.user_biases[user_idx]
                            )
                            
                            self.item_biases[item_idx] += self.config['learning_rate'] * (
                                error - self.config['regularization'] * self.item_biases[item_idx]
                            )
                    
                    epoch_loss += np.sum(errors ** 2)
                
                # Calculate total loss with regularization
                reg_loss = (self.config['regularization'] * 
                           (np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)))
                
                if self.config['use_bias']:
                    reg_loss += self.config['regularization'] * (
                        np.sum(self.user_biases ** 2) + np.sum(self.item_biases ** 2)
                    )
                
                total_loss = epoch_loss + reg_loss
                self.training_history.append(total_loss)
                
                # Early stopping check
                if total_loss < best_loss - self.config['min_improvement']:
                    best_loss = total_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")
            
            self.is_trained = True
            logger.info("Matrix factorization training completed")
            return True
    
    def _predict_batch(self, user_indices, item_indices):
        """Predict ratings for a batch of user-item pairs"""
        predictions = np.zeros(len(user_indices))
        
        for i, (user_idx, item_idx) in enumerate(zip(user_indices, item_indices)):
            prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            
            if self.config['use_bias']:
                prediction += (self.global_bias + 
                             self.user_biases[user_idx] + 
                             self.item_biases[item_idx])
            
            predictions[i] = prediction
        
        return predictions
    
    def predict_rating(self, user_id, item_id):
        """Predict rating for a specific user-item pair"""
        if not self.is_trained:
            self.train_advanced()
        
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_bias if self.config['use_bias'] else 0.0
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]
        
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        if self.config['use_bias']:
            prediction += (self.global_bias + 
                         self.user_biases[user_idx] + 
                         self.item_biases[item_idx])
        
        return prediction
    
    def get_recommendations_advanced(self, user_id, limit=20, exclude_seen=True):
        """Get advanced matrix factorization recommendations"""
        if not self.is_trained:
            self.train_advanced()
        
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get user's seen items if excluding
        seen_items = set()
        if exclude_seen:
            user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            seen_items = set(interaction.content_id for interaction in user_interactions)
        
        recommendations = []
        
        for item_id, item_idx in self.item_to_idx.items():
            if exclude_seen and item_id in seen_items:
                continue
            
            predicted_rating = self.predict_rating(user_id, item_id)
            recommendations.append((item_id, predicted_rating))
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:limit]
    
    def _get_interaction_weight(self, interaction):
        """Convert interaction to rating weight"""
        weights = {
            'view': 2.0,
            'like': 4.0,
            'favorite': 5.0,
            'watchlist': 3.5,
            'rating': interaction.rating if interaction.rating else 3.0,
            'search': 1.0,
            'share': 3.0
        }
        
        base_weight = weights.get(interaction.interaction_type, 2.0)
        
        # Apply temporal decay
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        temporal_factor = 0.95 ** (days_ago / 30.0)
        
        return base_weight * temporal_factor

class HybridRecommendationSystem:
    def __init__(self, db, models, config=None):
        self.db = db
        self.models = models
        
        self.config = config or {
            'collaborative_weight': 0.4,
            'content_weight': 0.35,
            'matrix_factorization_weight': 0.25,
            'min_interactions_for_cf': 5,
            'diversity_factor': 0.2,
            'novelty_factor': 0.1,
            'popularity_boost': 0.05
        }
        
        # Initialize individual systems
        self.collaborative_filtering = AdvancedCollaborativeFiltering(db, models)
        self.content_filtering = AdvancedContentBasedFiltering(db, models)
        self.matrix_factorization = AdvancedMatrixFactorization(db, models)
        
    def get_hybrid_recommendations(self, user_id, limit=20, strategy='adaptive'):
        """Get hybrid recommendations using multiple algorithms"""
        # Determine user's interaction count for adaptive weighting
        user_interaction_count = self.models['UserInteraction'].query.filter_by(user_id=user_id).count()
        
        # Adaptive strategy based on user data availability
        if strategy == 'adaptive':
            if user_interaction_count < self.config['min_interactions_for_cf']:
                # New user: rely more on content-based and popularity
                weights = {
                    'collaborative': 0.1,
                    'content': 0.5,
                    'matrix_factorization': 0.2,
                    'popularity': 0.2
                }
            elif user_interaction_count < 20:
                # Moderate user: balanced approach
                weights = {
                    'collaborative': 0.3,
                    'content': 0.4,
                    'matrix_factorization': 0.25,
                    'popularity': 0.05
                }
            else:
                # Power user: rely more on collaborative filtering
                weights = {
                    'collaborative': 0.45,
                    'content': 0.25,
                    'matrix_factorization': 0.25,
                    'popularity': 0.05
                }
        else:
            # Use default weights
            weights = {
                'collaborative': self.config['collaborative_weight'],
                'content': self.config['content_weight'],
                'matrix_factorization': self.config['matrix_factorization_weight'],
                'popularity': 0.0
            }
        
        # Get recommendations from each system
        recommendations_dict = defaultdict(float)
        algorithm_contributions = defaultdict(lambda: defaultdict(float))
        
        # Collaborative Filtering
        if weights['collaborative'] > 0:
            try:
                cf_recs = self.collaborative_filtering.user_based_recommendations_advanced(
                    user_id, limit * 3
                )
                for content_id, score in cf_recs:
                    weighted_score = score * weights['collaborative']
                    recommendations_dict[content_id] += weighted_score
                    algorithm_contributions[content_id]['collaborative'] = weighted_score
            except Exception as e:
                logger.warning(f"Collaborative filtering failed: {e}")
        
        # Content-Based Filtering
        if weights['content'] > 0:
            try:
                cb_recs = self.content_filtering.get_content_recommendations_advanced(
                    user_id, limit * 3
                )
                for content_id, score in cb_recs:
                    weighted_score = score * weights['content']
                    recommendations_dict[content_id] += weighted_score
                    algorithm_contributions[content_id]['content'] = weighted_score
            except Exception as e:
                logger.warning(f"Content-based filtering failed: {e}")
        
        # Matrix Factorization
        if weights['matrix_factorization'] > 0:
            try:
                mf_recs = self.matrix_factorization.get_recommendations_advanced(
                    user_id, limit * 3
                )
                for content_id, score in mf_recs:
                    # Normalize MF scores to 0-1 range
                    normalized_score = (score + 5) / 10.0  # Assuming scores range from -5 to 5
                    weighted_score = normalized_score * weights['matrix_factorization']
                    recommendations_dict[content_id] += weighted_score
                    algorithm_contributions[content_id]['matrix_factorization'] = weighted_score
            except Exception as e:
                logger.warning(f"Matrix factorization failed: {e}")
        
        # Add popularity boost for trending/new content
        if weights.get('popularity', 0) > 0:
            try:
                popular_content = self.models['Content'].query.filter(
                    self.models['Content'].is_trending == True
                ).limit(limit).all()
                
                for content in popular_content:
                    if content.id not in recommendations_dict:
                        popularity_score = (content.popularity or 0) / 1000.0  # Normalize
                        weighted_score = popularity_score * weights['popularity']
                        recommendations_dict[content.id] += weighted_score
                        algorithm_contributions[content.id]['popularity'] = weighted_score
            except Exception as e:
                logger.warning(f"Popularity boost failed: {e}")
        
        # Sort recommendations
        sorted_recommendations = sorted(
            recommendations_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Apply diversity and novelty filters
        if self.config['diversity_factor'] > 0 or self.config['novelty_factor'] > 0:
            sorted_recommendations = self._apply_advanced_filters(
                sorted_recommendations, 
                user_id, 
                limit
            )
        
        # Format output with algorithm contributions
        final_recommendations = []
        for content_id, score in sorted_recommendations[:limit]:
            content = self.models['Content'].query.get(content_id)
            if content:
                final_recommendations.append({
                    'content': content,
                    'score': score,
                    'algorithm_contributions': dict(algorithm_contributions[content_id])
                })
        
        return final_recommendations
    
    def _apply_advanced_filters(self, recommendations, user_id, limit):
        """Apply diversity and novelty filters to recommendations"""
        if not recommendations:
            return recommendations
        
        final_recommendations = []
        remaining = recommendations[:]
        
        # Add top recommendation first
        final_recommendations.append(remaining.pop(0))
        
        while len(final_recommendations) < limit and remaining:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for idx, (content_id, score) in enumerate(remaining):
                # Calculate diversity score
                diversity_score = self._calculate_diversity_score(
                    content_id, 
                    [rec[0] for rec in final_recommendations]
                )
                
                # Calculate novelty score
                novelty_score = self._calculate_novelty_score(user_id, content_id)
                
                # Combine scores
                combined_score = (
                    (1 - self.config['diversity_factor'] - self.config['novelty_factor']) * score +
                    self.config['diversity_factor'] * diversity_score +
                    self.config['novelty_factor'] * novelty_score
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = (content_id, score)
                    best_idx = idx
            
            if best_candidate:
                final_recommendations.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break
        
        return final_recommendations
    
    def _calculate_diversity_score(self, content_id, selected_content_ids):
        """Calculate diversity score for hybrid system"""
        if not selected_content_ids:
            return 1.0
        
        # Use content-based similarity for diversity calculation
        diversity_scores = []
        
        for selected_id in selected_content_ids:
            similarity = self.content_filtering.calculate_content_similarity_advanced(
                content_id, selected_id
            )
            diversity = 1.0 - similarity
            diversity_scores.append(diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 1.0
    
    def _calculate_novelty_score(self, user_id, content_id):
        """Calculate novelty score based on user's interaction history"""
        content = self.models['Content'].query.get(content_id)
        if not content:
            return 0.0
        
        user_interactions = self.models['UserInteraction'].query.filter_by(user_id=user_id).all()
        
        if not user_interactions:
            return 1.0  # Everything is novel for new users
        
        # Analyze user's historical preferences
        user_genres = Counter()
        user_languages = Counter()
        user_types = Counter()
        
        for interaction in user_interactions:
            interacted_content = self.models['Content'].query.get(interaction.content_id)
            if interacted_content:
                try:
                    genres = json.loads(interacted_content.genres or '[]')
                    for genre in genres:
                        user_genres[genre] += 1
                except:
                    pass
                
                try:
                    languages = json.loads(interacted_content.languages or '[]')
                    for language in languages:
                        user_languages[language] += 1
                except:
                    pass
                
                user_types[interacted_content.content_type] += 1
        
        # Calculate novelty factors
        novelty_factors = []
        
        # Genre novelty
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            if content_genres:
                genre_familiarity = sum(user_genres.get(genre, 0) for genre in content_genres)
                genre_novelty = 1.0 / (1.0 + genre_familiarity / len(content_genres))
                novelty_factors.append(genre_novelty * 0.5)
        except:
            novelty_factors.append(0.25)
        
        # Language novelty
        try:
            content_languages = set(json.loads(content.languages or '[]'))
            if content_languages:
                lang_familiarity = sum(user_languages.get(lang, 0) for lang in content_languages)
                lang_novelty = 1.0 / (1.0 + lang_familiarity / len(content_languages))
                novelty_factors.append(lang_novelty * 0.3)
        except:
            novelty_factors.append(0.15)
        
        # Content type novelty
        type_familiarity = user_types.get(content.content_type, 0)
        type_novelty = 1.0 / (1.0 + type_familiarity)
        novelty_factors.append(type_novelty * 0.2)
        
        return sum(novelty_factors)

class RealtimePersonalizationEngine:
    def __init__(self, db, models, config=None):
        self.db = db
        self.models = models
        
        self.config = config or {
            'learning_rate': 0.1,
            'decay_factor': 0.95,
            'session_weight': 1.5,
            'recent_interactions_window': 24,  # hours
            'feedback_integration_rate': 0.2
        }
        
        self.session_profiles = {}  # Store temporary session-based profiles
        self.realtime_adjustments = defaultdict(float)
        self._lock = threading.Lock()
    
    def update_realtime_profile(self, user_id, interaction_data):
        """Update user profile in real-time based on new interactions"""
        with self._lock:
            current_time = datetime.utcnow()
            
            # Get or create session profile
            if user_id not in self.session_profiles:
                self.session_profiles[user_id] = {
                    'session_start': current_time,
                    'interactions': [],
                    'preferences': defaultdict(float),
                    'last_update': current_time
                }
            
            session_profile = self.session_profiles[user_id]
            
            # Add new interaction
            session_profile['interactions'].append({
                'content_id': interaction_data['content_id'],
                'interaction_type': interaction_data['interaction_type'],
                'timestamp': current_time,
                'context': interaction_data.get('context', {})
            })
            
            # Update preferences based on content
            self._update_session_preferences(session_profile, interaction_data)
            
            session_profile['last_update'] = current_time
    
    def _update_session_preferences(self, session_profile, interaction_data):
        """Update session preferences based on interaction"""
        content = self.models['Content'].query.get(interaction_data['content_id'])
        if not content:
            return
        
        # Weight based on interaction type
        interaction_weights = {
            'view': 1.0,
            'like': 2.0,
            'favorite': 3.0,
            'share': 2.5,
            'search': 0.5
        }
        
        weight = interaction_weights.get(interaction_data['interaction_type'], 1.0)
        weight *= self.config['session_weight']
        
        # Update genre preferences
        try:
            genres = json.loads(content.genres or '[]')
            for genre in genres:
                session_profile['preferences'][f"genre_{genre}"] += weight
        except:
            pass
        
        # Update language preferences
        try:
            languages = json.loads(content.languages or '[]')
            for language in languages:
                session_profile['preferences'][f"language_{language}"] += weight
        except:
            pass
        
        # Update content type preferences
        session_profile['preferences'][f"type_{content.content_type}"] += weight
    
    def get_realtime_adjusted_recommendations(self, user_id, base_recommendations, limit=20):
        """Adjust base recommendations with real-time learning"""
        if user_id not in self.session_profiles:
            return base_recommendations[:limit]
        
        session_profile = self.session_profiles[user_id]
        
        # Apply real-time adjustments
        adjusted_recommendations = []
        
        for rec in base_recommendations:
            content = rec['content']
            base_score = rec['score']
            
            # Calculate real-time adjustment
            adjustment = self._calculate_realtime_adjustment(content, session_profile)
            
            # Apply adjustment
            adjusted_score = base_score * (1 + adjustment * self.config['learning_rate'])
            
            adjusted_recommendations.append({
                'content': content,
                'score': adjusted_score,
                'base_score': base_score,
                'realtime_adjustment': adjustment,
                'algorithm_contributions': rec.get('algorithm_contributions', {})
            })
        
        # Sort by adjusted score
        adjusted_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return adjusted_recommendations[:limit]
    
    def _calculate_realtime_adjustment(self, content, session_profile):
        """Calculate real-time adjustment factor for content"""
        adjustment = 0.0
        total_session_weight = sum(session_profile['preferences'].values())
        
        if total_session_weight == 0:
            return 0.0
        
        # Genre adjustment
        try:
            genres = json.loads(content.genres or '[]')
            for genre in genres:
                pref_key = f"genre_{genre}"
                if pref_key in session_profile['preferences']:
                    genre_weight = session_profile['preferences'][pref_key] / total_session_weight
                    adjustment += genre_weight * 0.5
        except:
            pass
        
        # Language adjustment
        try:
            languages = json.loads(content.languages or '[]')
            for language in languages:
                pref_key = f"language_{language}"
                if pref_key in session_profile['preferences']:
                    lang_weight = session_profile['preferences'][pref_key] / total_session_weight
                    adjustment += lang_weight * 0.3
        except:
            pass
        
        # Content type adjustment
        type_key = f"type_{content.content_type}"
        if type_key in session_profile['preferences']:
            type_weight = session_profile['preferences'][type_key] / total_session_weight
            adjustment += type_weight * 0.2
        
        return adjustment
    
    def cleanup_old_sessions(self, max_age_hours=24):
        """Clean up old session profiles"""
        with self._lock:
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(hours=max_age_hours)
            
            expired_sessions = [
                user_id for user_id, profile in self.session_profiles.items()
                if profile['last_update'] < cutoff_time
            ]
            
            for user_id in expired_sessions:
                del self.session_profiles[user_id]