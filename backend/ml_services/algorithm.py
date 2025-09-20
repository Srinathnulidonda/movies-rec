#backend/ml_services/algorithm.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jaccard, cityblock
from scipy.stats import pearsonr, spearmanr
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
import logging
import pickle
from textblob import TextBlob
import re

logger = logging.getLogger(__name__)

class AdvancedCollaborativeFiltering:
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.user_similarity_cache = {}
        self.item_similarity_cache = {}
        self.user_clusters = {}
        self.interaction_weights = {
            'search_click': 0.5,
            'view': 1.0,
            'like': 3.0,
            'favorite': 5.0,
            'watchlist': 4.0,
            'rating': 'dynamic',  # Based on actual rating
            'share': 2.5,
            'comment': 2.0,
            'search_query': 0.3
        }
        
    def get_enhanced_user_item_matrix(self, min_interactions=3):
        """Creates enhanced user-item matrix with sophisticated weighting"""
        interactions = self.UserInteraction.query.all()
        
        user_item_dict = defaultdict(lambda: defaultdict(float))
        user_interaction_counts = defaultdict(int)
        
        for interaction in interactions:
            if interaction.content_id is None:  # Skip search queries
                continue
                
            weight = self._calculate_enhanced_interaction_weight(interaction)
            user_item_dict[interaction.user_id][interaction.content_id] += weight
            user_interaction_counts[interaction.user_id] += 1
        
        # Filter out users with insufficient interactions
        filtered_users = {
            user_id: items for user_id, items in user_item_dict.items()
            if user_interaction_counts[user_id] >= min_interactions
        }
        
        if not filtered_users:
            return None, None, None
        
        users = list(filtered_users.keys())
        all_items = set()
        for items in filtered_users.values():
            all_items.update(items.keys())
        items = list(all_items)
        
        matrix = np.zeros((len(users), len(items)))
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        item_to_idx = {item: idx for idx, item in enumerate(items)}
        
        for user_id, items_dict in filtered_users.items():
            user_idx = user_to_idx[user_id]
            for item_id, weight in items_dict.items():
                item_idx = item_to_idx[item_id]
                matrix[user_idx, item_idx] = weight
        
        return matrix, user_to_idx, item_to_idx
    
    def _calculate_enhanced_interaction_weight(self, interaction):
        """Calculate sophisticated interaction weights with temporal and contextual factors"""
        base_weight = self.interaction_weights.get(interaction.interaction_type, 1.0)
        
        # Handle rating-based weight
        if interaction.interaction_type == 'rating' and interaction.rating:
            base_weight = interaction.rating / 2.0  # Scale 0-10 to 0-5
        elif base_weight == 'dynamic':
            base_weight = 2.0
        
        # Temporal decay
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        temporal_weight = math.exp(-days_ago / 45.0)  # 45-day half-life
        
        # Context-based adjustments
        context_weight = 1.0
        if hasattr(interaction, 'interaction_metadata') and interaction.interaction_metadata:
            metadata = interaction.interaction_metadata
            
            # View percentage adjustment for view interactions
            if interaction.interaction_type == 'view':
                view_percentage = metadata.get('view_percentage', 50)
                context_weight *= (view_percentage / 100.0) * 1.5 + 0.5
            
            # Engagement score adjustment
            engagement_score = metadata.get('engagement_score', 1.0)
            context_weight *= engagement_score
            
            # Session duration bonus
            session_duration = metadata.get('session_duration', 0)
            if session_duration > 300:  # 5+ minutes
                context_weight *= 1.2
        
        # Recency boost for recent interactions
        if days_ago <= 7:
            recency_boost = 1.3
        elif days_ago <= 30:
            recency_boost = 1.1
        else:
            recency_boost = 1.0
        
        final_weight = base_weight * temporal_weight * context_weight * recency_boost
        return max(final_weight, 0.1)  # Minimum weight
    
    def calculate_advanced_user_similarity(self, user_id, similarity_method='hybrid'):
        """Calculate user similarity using multiple methods"""
        cache_key = f"{user_id}_{similarity_method}"
        if cache_key in self.user_similarity_cache:
            return self.user_similarity_cache[cache_key]
        
        matrix, user_to_idx, item_to_idx = self.get_enhanced_user_item_matrix()
        if matrix is None or user_id not in user_to_idx:
            return {}
        
        user_idx = user_to_idx[user_id]
        user_vector = matrix[user_idx]
        
        similarities = {}
        
        if similarity_method in ['cosine', 'hybrid']:
            cosine_sims = cosine_similarity([user_vector], matrix)[0]
            similarities['cosine'] = cosine_sims
        
        if similarity_method in ['pearson', 'hybrid']:
            pearson_sims = []
            for other_idx in range(len(matrix)):
                if other_idx != user_idx:
                    # Only calculate for users with common items
                    common_items = (user_vector > 0) & (matrix[other_idx] > 0)
                    if np.sum(common_items) > 2:
                        corr, _ = pearsonr(user_vector[common_items], matrix[other_idx][common_items])
                        pearson_sims.append(corr if not np.isnan(corr) else 0)
                    else:
                        pearson_sims.append(0)
                else:
                    pearson_sims.append(1.0)
            similarities['pearson'] = np.array(pearson_sims)
        
        if similarity_method in ['jaccard', 'hybrid']:
            jaccard_sims = []
            for other_idx in range(len(matrix)):
                if other_idx != user_idx:
                    user_binary = (user_vector > 0).astype(int)
                    other_binary = (matrix[other_idx] > 0).astype(int)
                    intersection = np.sum(user_binary & other_binary)
                    union = np.sum(user_binary | other_binary)
                    jaccard_sim = intersection / union if union > 0 else 0
                    jaccard_sims.append(jaccard_sim)
                else:
                    jaccard_sims.append(1.0)
            similarities['jaccard'] = np.array(jaccard_sims)
        
        # Combine similarities for hybrid approach
        if similarity_method == 'hybrid':
            combined_sim = (
                0.4 * similarities['cosine'] +
                0.35 * similarities.get('pearson', similarities['cosine']) +
                0.25 * similarities['jaccard']
            )
        else:
            combined_sim = similarities[similarity_method]
        
        result = {}
        for other_user_id, other_idx in user_to_idx.items():
            if other_user_id != user_id:
                result[other_user_id] = combined_sim[other_idx]
        
        self.user_similarity_cache[cache_key] = result
        return result
    
    def get_user_based_recommendations_advanced(self, user_id, limit=20, min_similarity=0.1):
        """Advanced user-based collaborative filtering with clustering"""
        similarities = self.calculate_advanced_user_similarity(user_id, 'hybrid')
        
        if not similarities:
            return []
        
        # Get top similar users
        similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        similar_users = [(uid, sim) for uid, sim in similar_users if sim >= min_similarity][:100]
        
        # Get user's interaction history
        user_interactions = {}
        for interaction in self.UserInteraction.query.filter_by(user_id=user_id).all():
            if interaction.content_id:
                weight = self._calculate_enhanced_interaction_weight(interaction)
                user_interactions[interaction.content_id] = max(
                    user_interactions.get(interaction.content_id, 0), weight
                )
        
        # Generate recommendations from similar users
        recommendations = defaultdict(float)
        recommendation_sources = defaultdict(list)
        
        for similar_user_id, similarity_score in similar_users:
            similar_user_interactions = self.UserInteraction.query.filter_by(
                user_id=similar_user_id
            ).all()
            
            for interaction in similar_user_interactions:
                if (interaction.content_id and 
                    interaction.content_id not in user_interactions):
                    
                    weight = self._calculate_enhanced_interaction_weight(interaction)
                    score = similarity_score * weight
                    recommendations[interaction.content_id] += score
                    recommendation_sources[interaction.content_id].append({
                        'user_id': similar_user_id,
                        'similarity': similarity_score,
                        'interaction_weight': weight
                    })
        
        # Apply content quality filtering
        filtered_recommendations = []
        for content_id, score in recommendations.items():
            content = self.Content.query.get(content_id)
            if content and content.rating and content.rating >= 6.0:  # Quality threshold
                filtered_recommendations.append((content_id, score))
        
        sorted_recommendations = sorted(
            filtered_recommendations, 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        return sorted_recommendations

class DeepContentAnalyzer:
    def __init__(self, db, models):
        self.db = db
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000, 
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2
        )
        self.content_embeddings = {}
        self.genre_embeddings = {}
        self.language_embeddings = {}
        
    def extract_deep_content_features(self, content_id):
        """Extract comprehensive content features with deep analysis"""
        if content_id in self.content_embeddings:
            return self.content_embeddings[content_id]
        
        content = self.Content.query.get(content_id)
        if not content:
            return None
        
        features = {}
        
        # Basic metadata features
        features['content_type'] = content.content_type
        features['rating'] = content.rating or 0
        features['popularity'] = content.popularity or 0
        features['vote_count'] = content.vote_count or 0
        features['runtime'] = content.runtime or 0
        
        # Release date features
        if content.release_date:
            features['release_year'] = content.release_date.year
            features['release_month'] = content.release_date.month
            features['years_since_release'] = datetime.now().year - content.release_date.year
            features['is_recent'] = features['years_since_release'] <= 3
            features['is_classic'] = features['years_since_release'] >= 20
        else:
            features.update({
                'release_year': 0, 'release_month': 0, 
                'years_since_release': 0, 'is_recent': False, 'is_classic': False
            })
        
        # Genre analysis
        try:
            genres = json.loads(content.genres or '[]')
            features['genres'] = set(genres)
            features['genre_count'] = len(genres)
            features['is_multi_genre'] = len(genres) > 3
            
            # Genre categories
            action_genres = {'Action', 'Thriller', 'Adventure', 'Crime'}
            drama_genres = {'Drama', 'Romance', 'Biography', 'History'}
            comedy_genres = {'Comedy', 'Family', 'Animation'}
            horror_genres = {'Horror', 'Mystery', 'Suspense'}
            
            features['is_action'] = bool(set(genres) & action_genres)
            features['is_drama'] = bool(set(genres) & drama_genres)
            features['is_comedy'] = bool(set(genres) & comedy_genres)
            features['is_horror'] = bool(set(genres) & horror_genres)
            
        except:
            features.update({
                'genres': set(), 'genre_count': 0, 'is_multi_genre': False,
                'is_action': False, 'is_drama': False, 'is_comedy': False, 'is_horror': False
            })
        
        # Language analysis
        try:
            languages = json.loads(content.languages or '[]')
            features['languages'] = set(languages)
            features['language_count'] = len(languages)
            features['is_multilingual'] = len(languages) > 1
            
            # Regional language detection
            indian_languages = {'hi', 'te', 'ta', 'ml', 'kn', 'hindi', 'telugu', 'tamil', 'malayalam', 'kannada'}
            features['is_indian_content'] = bool(set(languages) & indian_languages)
            features['is_english'] = 'en' in languages or 'english' in languages
            features['is_telugu'] = 'te' in languages or 'telugu' in languages
            
        except:
            features.update({
                'languages': set(), 'language_count': 0, 'is_multilingual': False,
                'is_indian_content': False, 'is_english': False, 'is_telugu': False
            })
        
        # Overview analysis
        if content.overview:
            blob = TextBlob(content.overview)
            features['overview_sentiment'] = blob.sentiment.polarity
            features['overview_subjectivity'] = blob.sentiment.subjectivity
            features['overview_length'] = len(content.overview)
            features['overview_word_count'] = len(content.overview.split())
            
            # Extract keywords
            keywords = self._extract_keywords(content.overview)
            features['overview_keywords'] = keywords
        else:
            features.update({
                'overview_sentiment': 0, 'overview_subjectivity': 0,
                'overview_length': 0, 'overview_word_count': 0,
                'overview_keywords': set()
            })
        
        # Popularity analysis
        if content.popularity:
            features['popularity_tier'] = self._get_popularity_tier(content.popularity)
        else:
            features['popularity_tier'] = 'unknown'
        
        # Quality indicators
        features['quality_score'] = self._calculate_quality_score(content)
        features['is_high_quality'] = features['quality_score'] >= 7.5
        features['is_critically_acclaimed'] = (content.rating or 0) >= 8.0 and (content.vote_count or 0) >= 1000
        
        self.content_embeddings[content_id] = features
        return features
    
    def _extract_keywords(self, text):
        """Extract meaningful keywords from text"""
        if not text:
            return set()
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stop_words = {'that', 'with', 'have', 'this', 'will', 'they', 'from', 'been', 'more', 'when', 'where', 'what', 'after', 'before'}
        keywords = [word for word in words if word not in stop_words]
        return set(keywords[:10])  # Top 10 keywords
    
    def _get_popularity_tier(self, popularity):
        """Categorize content by popularity"""
        if popularity >= 100:
            return 'viral'
        elif popularity >= 50:
            return 'very_popular'
        elif popularity >= 20:
            return 'popular'
        elif popularity >= 5:
            return 'moderate'
        else:
            return 'niche'
    
    def _calculate_quality_score(self, content):
        """Calculate overall quality score"""
        score = 0
        
        if content.rating:
            score += content.rating * 0.6
        
        if content.vote_count:
            vote_bonus = min(content.vote_count / 1000, 2.0)  # Max 2 points for votes
            score += vote_bonus
        
        if content.popularity:
            popularity_bonus = min(content.popularity / 50, 1.0)  # Max 1 point for popularity
            score += popularity_bonus
        
        return min(score, 10.0)
    
    def calculate_advanced_content_similarity(self, content_id1, content_id2):
        """Calculate sophisticated content similarity"""
        features1 = self.extract_deep_content_features(content_id1)
        features2 = self.extract_deep_content_features(content_id2)
        
        if not features1 or not features2:
            return 0.0
        
        similarity_components = []
        
        # Genre similarity (weighted by importance)
        genre_jaccard = len(features1['genres'] & features2['genres']) / max(
            len(features1['genres'] | features2['genres']), 1
        )
        similarity_components.append(('genre', genre_jaccard, 0.25))
        
        # Language similarity
        language_jaccard = len(features1['languages'] & features2['languages']) / max(
            len(features1['languages'] | features2['languages']), 1
        )
        similarity_components.append(('language', language_jaccard, 0.15))
        
        # Content type similarity
        type_sim = 1.0 if features1['content_type'] == features2['content_type'] else 0.0
        similarity_components.append(('type', type_sim, 0.2))
        
        # Quality similarity
        quality_diff = abs(features1['quality_score'] - features2['quality_score'])
        quality_sim = max(0, 1 - quality_diff / 10.0)
        similarity_components.append(('quality', quality_sim, 0.1))
        
        # Temporal similarity
        year_diff = abs(features1['release_year'] - features2['release_year'])
        temporal_sim = max(0, 1 - year_diff / 20.0)
        similarity_components.append(('temporal', temporal_sim, 0.1))
        
        # Overview keyword similarity
        keyword_jaccard = len(features1['overview_keywords'] & features2['overview_keywords']) / max(
            len(features1['overview_keywords'] | features2['overview_keywords']), 1
        )
        similarity_components.append(('keywords', keyword_jaccard, 0.1))
        
        # Sentiment similarity
        sentiment_diff = abs(features1['overview_sentiment'] - features2['overview_sentiment'])
        sentiment_sim = max(0, 1 - sentiment_diff)
        similarity_components.append(('sentiment', sentiment_sim, 0.05))
        
        # Runtime similarity (for movies/shows)
        if features1['runtime'] > 0 and features2['runtime'] > 0:
            runtime_diff = abs(features1['runtime'] - features2['runtime'])
            runtime_sim = max(0, 1 - runtime_diff / 180.0)  # 3-hour max difference
            similarity_components.append(('runtime', runtime_sim, 0.05))
        
        # Calculate weighted similarity
        total_similarity = sum(sim * weight for _, sim, weight in similarity_components)
        
        return total_similarity

class NeuralRecommendationEngine:
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.embedding_dim = 128
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.trained_models = {}
        
    def create_neural_user_embedding(self, user_id):
        """Create sophisticated neural user embedding"""
        if user_id in self.user_embeddings:
            return self.user_embeddings[user_id]
        
        user = self.User.query.get(user_id)
        interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not interactions:
            # Cold start embedding based on preferences
            embedding = self._create_cold_start_embedding(user)
        else:
            embedding = self._create_interaction_based_embedding(user_id, interactions)
        
        self.user_embeddings[user_id] = embedding
        return embedding
    
    def _create_cold_start_embedding(self, user):
        """Create embedding for users with no interactions"""
        embedding = np.random.normal(0, 0.01, self.embedding_dim)
        
        try:
            preferred_genres = json.loads(user.preferred_genres or '[]')
            preferred_languages = json.loads(user.preferred_languages or '[]')
            
            # Encode genre preferences
            genre_map = {
                'Action': 0, 'Comedy': 1, 'Drama': 2, 'Horror': 3, 'Romance': 4,
                'Thriller': 5, 'Adventure': 6, 'Animation': 7, 'Crime': 8, 'Documentary': 9
            }
            
            for genre in preferred_genres:
                if genre in genre_map and genre_map[genre] < 20:
                    embedding[genre_map[genre]] = 1.0
            
            # Encode language preferences
            language_map = {
                'english': 20, 'hindi': 21, 'telugu': 22, 'tamil': 23,
                'malayalam': 24, 'kannada': 25, 'japanese': 26, 'korean': 27
            }
            
            for language in preferred_languages:
                if language in language_map and language_map[language] < 30:
                    embedding[language_map[language]] = 1.0
                    
        except:
            pass
        
        return embedding
    
    def _create_interaction_based_embedding(self, user_id, interactions):
        """Create embedding based on user interactions"""
        embedding = np.zeros(self.embedding_dim)
        
        # Aggregate content features
        genre_weights = defaultdict(float)
        language_weights = defaultdict(float)
        type_weights = defaultdict(float)
        quality_scores = []
        sentiment_scores = []
        
        total_weight = 0
        
        for interaction in interactions:
            if not interaction.content_id:
                continue
                
            content = self.Content.query.get(interaction.content_id)
            if not content:
                continue
            
            weight = self._calculate_interaction_weight(interaction)
            total_weight += weight
            
            # Process genres
            try:
                genres = json.loads(content.genres or '[]')
                for genre in genres:
                    genre_weights[genre] += weight
            except:
                pass
            
            # Process languages
            try:
                languages = json.loads(content.languages or '[]')
                for language in languages:
                    language_weights[language] += weight
            except:
                pass
            
            # Process content type
            type_weights[content.content_type] += weight
            
            # Process quality
            if content.rating:
                quality_scores.append(content.rating * weight)
        
        if total_weight == 0:
            return np.random.normal(0, 0.01, self.embedding_dim)
        
        # Normalize weights
        for genre in genre_weights:
            genre_weights[genre] /= total_weight
        for language in language_weights:
            language_weights[language] /= total_weight
        for content_type in type_weights:
            type_weights[content_type] /= total_weight
        
        # Encode into embedding
        # Genres (0-29)
        genre_list = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
                     'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
                     'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War',
                     'Western', 'Sport', 'Musical', 'Film-Noir', 'Short', 'News', 'Reality-TV',
                     'Talk-Show', 'Game-Show', 'Adult', 'Indie', 'Experimental']
        
        for i, genre in enumerate(genre_list):
            if i < 30 and genre in genre_weights:
                embedding[i] = genre_weights[genre]
        
        # Languages (30-39)
        language_list = ['english', 'hindi', 'telugu', 'tamil', 'malayalam', 
                        'kannada', 'japanese', 'korean', 'french', 'spanish']
        
        for i, language in enumerate(language_list):
            if 30 + i < 40 and language in language_weights:
                embedding[30 + i] = language_weights[language]
        
        # Content types (40-42)
        type_list = ['movie', 'tv', 'anime']
        for i, content_type in enumerate(type_list):
            if 40 + i < 43 and content_type in type_weights:
                embedding[40 + i] = type_weights[content_type]
        
        # Quality preference (43)
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            embedding[43] = avg_quality / 10.0
        
        # Interaction diversity (44)
        unique_types = len(set(interaction.interaction_type for interaction in interactions))
        embedding[44] = min(unique_types / 5.0, 1.0)
        
        # Recent activity (45)
        recent_interactions = [
            interaction for interaction in interactions
            if (datetime.utcnow() - interaction.timestamp).days <= 7
        ]
        embedding[45] = min(len(recent_interactions) / 10.0, 1.0)
        
        # Viewing patterns (46-50)
        # Time-based patterns could be encoded here
        
        return embedding
    
    def _calculate_interaction_weight(self, interaction):
        """Calculate weight for interaction in neural embedding"""
        weights = {
            'view': 1.0, 'like': 3.0, 'favorite': 5.0, 'watchlist': 4.0,
            'rating': interaction.rating / 2.0 if interaction.rating else 2.0,
            'search_click': 0.5, 'share': 2.5
        }
        
        base_weight = weights.get(interaction.interaction_type, 1.0)
        
        # Time decay
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        time_decay = math.exp(-days_ago / 30.0)
        
        return base_weight * time_decay
    
    def predict_user_item_affinity(self, user_id, content_id):
        """Predict user affinity for content using neural approach"""
        user_embedding = self.create_neural_user_embedding(user_id)
        
        # Create item embedding (simplified for now)
        content = self.Content.query.get(content_id)
        if not content:
            return 0.0
        
        item_embedding = np.zeros(self.embedding_dim)
        
        # Encode content features into item embedding
        try:
            genres = json.loads(content.genres or '[]')
            genre_list = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 
                         'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
                         'Music', 'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War',
                         'Western', 'Sport', 'Musical', 'Film-Noir', 'Short', 'News', 'Reality-TV',
                         'Talk-Show', 'Game-Show', 'Adult', 'Indie', 'Experimental']
            
            for i, genre in enumerate(genre_list):
                if i < 30 and genre in genres:
                    item_embedding[i] = 1.0
        except:
            pass
        
        try:
            languages = json.loads(content.languages or '[]')
            language_list = ['english', 'hindi', 'telugu', 'tamil', 'malayalam', 
                            'kannada', 'japanese', 'korean', 'french', 'spanish']
            
            for i, language in enumerate(language_list):
                if 30 + i < 40 and language in languages:
                    item_embedding[30 + i] = 1.0
        except:
            pass
        
        # Content type
        type_map = {'movie': 40, 'tv': 41, 'anime': 42}
        if content.content_type in type_map:
            item_embedding[type_map[content.content_type]] = 1.0
        
        # Quality
        if content.rating:
            item_embedding[43] = content.rating / 10.0
        
        # Calculate similarity
        similarity = cosine_similarity(
            user_embedding.reshape(1, -1),
            item_embedding.reshape(1, -1)
        )[0][0]
        
        return max(0, similarity)

class AdvancedDiversityOptimizer:
    def __init__(self, db, models):
        self.db = db
        self.Content = models['Content']
        
    def optimize_recommendations_for_diversity(self, recommendations, diversity_factor=0.4, limit=20):
        """Advanced diversity optimization using multiple criteria"""
        if not recommendations or len(recommendations) <= limit:
            return recommendations
        
        final_recommendations = []
        remaining_recommendations = recommendations[:]
        
        # Start with the highest-scored item
        final_recommendations.append(remaining_recommendations.pop(0))
        
        while len(final_recommendations) < limit and remaining_recommendations:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for idx, (content_id, original_score) in enumerate(remaining_recommendations):
                # Calculate diversity score
                diversity_score = self._calculate_comprehensive_diversity(
                    [rec[0] for rec in final_recommendations], content_id
                )
                
                # Combine original score with diversity
                combined_score = (1 - diversity_factor) * original_score + diversity_factor * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = (content_id, original_score)
                    best_idx = idx
            
            if best_candidate:
                final_recommendations.append(best_candidate)
                remaining_recommendations.pop(best_idx)
            else:
                break
        
        return final_recommendations
    
    def _calculate_comprehensive_diversity(self, existing_content_ids, new_content_id):
        """Calculate comprehensive diversity score"""
        if not existing_content_ids:
            return 1.0
        
        new_content = self.Content.query.get(new_content_id)
        if not new_content:
            return 0.0
        
        diversity_factors = []
        
        for existing_id in existing_content_ids:
            existing_content = self.Content.query.get(existing_id)
            if not existing_content:
                continue
            
            # Content type diversity
            type_diversity = 0.0 if new_content.content_type == existing_content.content_type else 1.0
            diversity_factors.append(type_diversity * 0.3)
            
            # Genre diversity
            try:
                new_genres = set(json.loads(new_content.genres or '[]'))
                existing_genres = set(json.loads(existing_content.genres or '[]'))
                genre_overlap = len(new_genres & existing_genres) / max(len(new_genres | existing_genres), 1)
                genre_diversity = 1 - genre_overlap
                diversity_factors.append(genre_diversity * 0.25)
            except:
                diversity_factors.append(0.0)
            
            # Language diversity
            try:
                new_languages = set(json.loads(new_content.languages or '[]'))
                existing_languages = set(json.loads(existing_content.languages or '[]'))
                if new_languages & existing_languages:
                    language_diversity = 0.3
                else:
                    language_diversity = 1.0
                diversity_factors.append(language_diversity * 0.2)
            except:
                diversity_factors.append(0.0)
            
            # Release year diversity
            if new_content.release_date and existing_content.release_date:
                year_diff = abs(new_content.release_date.year - existing_content.release_date.year)
                year_diversity = min(year_diff / 20.0, 1.0)
                diversity_factors.append(year_diversity * 0.15)
            
            # Quality tier diversity
            new_quality_tier = self._get_quality_tier(new_content.rating or 0)
            existing_quality_tier = self._get_quality_tier(existing_content.rating or 0)
            quality_diversity = 0.0 if new_quality_tier == existing_quality_tier else 0.5
            diversity_factors.append(quality_diversity * 0.1)
        
        return sum(diversity_factors) / len(diversity_factors) if diversity_factors else 1.0
    
    def _get_quality_tier(self, rating):
        """Get quality tier for rating"""
        if rating >= 8.5:
            return 'excellent'
        elif rating >= 7.5:
            return 'very_good'
        elif rating >= 6.5:
            return 'good'
        elif rating >= 5.5:
            return 'average'
        else:
            return 'below_average'

class BehavioralPatternAnalyzer:
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
    def analyze_user_behavioral_patterns(self, user_id):
        """Comprehensive analysis of user behavioral patterns"""
        interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not interactions:
            return self._get_default_patterns()
        
        patterns = {}
        
        # Temporal patterns
        patterns['temporal'] = self._analyze_temporal_patterns(interactions)
        
        # Content preferences
        patterns['content_preferences'] = self._analyze_content_preferences(interactions)
        
        # Interaction patterns
        patterns['interaction_patterns'] = self._analyze_interaction_patterns(interactions)
        
        # Discovery patterns
        patterns['discovery_patterns'] = self._analyze_discovery_patterns(interactions)
        
        # Quality preferences
        patterns['quality_preferences'] = self._analyze_quality_preferences(interactions)
        
        # Engagement patterns
        patterns['engagement_patterns'] = self._analyze_engagement_patterns(interactions)
        
        return patterns
    
    def _analyze_temporal_patterns(self, interactions):
        """Analyze when user is most active"""
        hourly_activity = defaultdict(int)
        daily_activity = defaultdict(int)
        
        for interaction in interactions:
            hour = interaction.timestamp.hour
            day = interaction.timestamp.weekday()
            
            hourly_activity[hour] += 1
            daily_activity[day] += 1
        
        peak_hours = sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_days = sorted(daily_activity.items(), key=lambda x: x[1], reverse=True)[:2]
        
        return {
            'peak_hours': [hour for hour, _ in peak_hours],
            'peak_days': [day for day, _ in peak_days],
            'is_weekend_user': sum(daily_activity[5], daily_activity[6]) > sum(daily_activity[i] for i in range(5)),
            'activity_distribution': dict(hourly_activity)
        }
    
    def _analyze_content_preferences(self, interactions):
        """Analyze content type and genre preferences"""
        type_counts = defaultdict(int)
        genre_counts = defaultdict(int)
        language_counts = defaultdict(int)
        
        for interaction in interactions:
            if not interaction.content_id:
                continue
                
            content = self.Content.query.get(interaction.content_id)
            if not content:
                continue
            
            weight = self._get_preference_weight(interaction)
            
            type_counts[content.content_type] += weight
            
            try:
                genres = json.loads(content.genres or '[]')
                for genre in genres:
                    genre_counts[genre] += weight
            except:
                pass
            
            try:
                languages = json.loads(content.languages or '[]')
                for language in languages:
                    language_counts[language] += weight
            except:
                pass
        
        return {
            'preferred_types': sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'preferred_genres': sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'preferred_languages': sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'diversity_score': len(genre_counts) / max(sum(type_counts.values()), 1)
        }
    
    def _get_preference_weight(self, interaction):
        """Get weight for preference analysis"""
        weights = {
            'favorite': 5.0, 'watchlist': 4.0, 'like': 3.0,
            'rating': interaction.rating if interaction.rating else 2.0,
            'view': 1.0, 'search_click': 0.5
        }
        return weights.get(interaction.interaction_type, 1.0)
    
    def _get_default_patterns(self):
        """Default patterns for new users"""
        return {
            'temporal': {'peak_hours': [20, 21, 22], 'peak_days': [5, 6], 'is_weekend_user': True},
            'content_preferences': {'preferred_types': [], 'preferred_genres': [], 'preferred_languages': []},
            'interaction_patterns': {'engagement_level': 'new_user'},
            'discovery_patterns': {'openness_to_new': 1.0},
            'quality_preferences': {'min_quality': 6.0},
            'engagement_patterns': {'depth': 'shallow'}
        }
    
    def _analyze_interaction_patterns(self, interactions):
        """Analyze how user interacts with content"""
        interaction_types = defaultdict(int)
        
        for interaction in interactions:
            interaction_types[interaction.interaction_type] += 1
        
        total_interactions = len(interactions)
        
        # Calculate engagement level
        high_engagement_types = {'favorite', 'rating', 'watchlist'}
        high_engagement_count = sum(interaction_types[t] for t in high_engagement_types)
        engagement_ratio = high_engagement_count / max(total_interactions, 1)
        
        if engagement_ratio > 0.5:
            engagement_level = 'very_high'
        elif engagement_ratio > 0.3:
            engagement_level = 'high'
        elif engagement_ratio > 0.15:
            engagement_level = 'moderate'
        else:
            engagement_level = 'low'
        
        return {
            'engagement_level': engagement_level,
            'interaction_distribution': dict(interaction_types),
            'total_interactions': total_interactions,
            'avg_interactions_per_day': total_interactions / max((datetime.utcnow() - min(i.timestamp for i in interactions)).days, 1)
        }
    
    def _analyze_discovery_patterns(self, interactions):
        """Analyze user's openness to discovering new content"""
        # This would analyze how often user explores different genres, types, etc.
        content_variety = set()
        
        for interaction in interactions:
            if interaction.content_id:
                content = self.Content.query.get(interaction.content_id)
                if content:
                    content_variety.add(content.content_type)
                    try:
                        genres = json.loads(content.genres or '[]')
                        content_variety.update(genres)
                    except:
                        pass
        
        openness_score = min(len(content_variety) / 20.0, 1.0)  # Normalized to 0-1
        
        return {
            'openness_to_new': openness_score,
            'content_variety_score': len(content_variety),
            'exploration_tendency': 'high' if openness_score > 0.7 else 'moderate' if openness_score > 0.4 else 'low'
        }
    
    def _analyze_quality_preferences(self, interactions):
        """Analyze user's quality preferences"""
        ratings_engaged = []
        
        for interaction in interactions:
            if interaction.content_id:
                content = self.Content.query.get(interaction.content_id)
                if content and content.rating:
                    weight = self._get_preference_weight(interaction)
                    if weight >= 2.0:  # Only consider meaningful interactions
                        ratings_engaged.append(content.rating)
        
        if ratings_engaged:
            avg_quality = sum(ratings_engaged) / len(ratings_engaged)
            min_quality = min(ratings_engaged)
            quality_std = np.std(ratings_engaged)
        else:
            avg_quality = 7.0
            min_quality = 6.0
            quality_std = 1.0
        
        return {
            'avg_preferred_quality': avg_quality,
            'min_quality_threshold': min_quality,
            'quality_tolerance': quality_std,
            'is_quality_sensitive': quality_std < 1.5
        }
    
    def _analyze_engagement_patterns(self, interactions):
        """Analyze depth and style of user engagement"""
        view_interactions = [i for i in interactions if i.interaction_type == 'view']
        rating_interactions = [i for i in interactions if i.interaction_type == 'rating']
        
        engagement_depth = 'deep' if len(rating_interactions) > len(view_interactions) * 0.3 else 'shallow'
        
        return {
            'depth': engagement_depth,
            'rates_frequently': len(rating_interactions) > 10,
            'uses_watchlist': any(i.interaction_type == 'watchlist' for i in interactions),
            'social_engagement': any(i.interaction_type in ['share', 'comment'] for i in interactions)
        }