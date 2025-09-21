#backend/ml_services/algorithm.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jaccard, correlation
from scipy.stats import pearsonr
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
import re
from textblob import TextBlob

logger = logging.getLogger(__name__)

class AdvancedUserProfiler:
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.user_profiles = {}
        
    def build_comprehensive_user_profile(self, user_id):
        """Build detailed user profile from all interactions"""
        interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        profile = {
            'user_id': user_id,
            'genre_preferences': defaultdict(float),
            'language_preferences': defaultdict(float),
            'content_type_preferences': defaultdict(float),
            'rating_patterns': defaultdict(list),
            'temporal_patterns': defaultdict(int),
            'search_patterns': defaultdict(int),
            'viewing_behavior': {
                'quick_views': 0,
                'extended_views': 0,
                'repeated_views': 0,
                'completion_rate': 0.0
            },
            'quality_preference': 0.0,
            'popularity_preference': 0.0,
            'novelty_preference': 0.0,
            'diversity_score': 0.0,
            'interaction_weights': defaultdict(float),
            'preferred_decades': defaultdict(float),
            'runtime_preferences': defaultdict(float),
            'sentiment_analysis': defaultdict(float)
        }
        
        if not interactions:
            return profile
            
        # Analyze each interaction with advanced weighting
        for interaction in interactions:
            content = self.Content.query.get(interaction.content_id)
            if not content:
                continue
                
            # Calculate interaction weight with time decay and type importance
            weight = self._calculate_advanced_interaction_weight(interaction)
            interaction_type = interaction.interaction_type
            
            # Update content type preferences
            profile['content_type_preferences'][content.content_type] += weight
            
            # Analyze genres with sentiment
            if content.genres:
                try:
                    genres = json.loads(content.genres)
                    for genre in genres:
                        profile['genre_preferences'][genre] += weight
                        
                        # Add sentiment analysis for rating-based interactions
                        if interaction.rating:
                            sentiment_weight = (interaction.rating - 5.0) / 5.0  # Normalize to -1 to 1
                            profile['sentiment_analysis'][genre] += sentiment_weight * weight
                except:
                    pass
            
            # Analyze languages
            if content.languages:
                try:
                    languages = json.loads(content.languages)
                    for language in languages:
                        profile['language_preferences'][language] += weight
                except:
                    pass
            
            # Analyze temporal patterns
            hour = interaction.timestamp.hour
            profile['temporal_patterns'][hour] += 1
            
            # Analyze search patterns
            if interaction_type == 'search':
                if hasattr(interaction, 'interaction_metadata') and interaction.interaction_metadata:
                    search_query = interaction.interaction_metadata.get('query', '').lower()
                    words = re.findall(r'\w+', search_query)
                    for word in words:
                        if len(word) > 3:
                            profile['search_patterns'][word] += weight
            
            # Analyze viewing behavior
            if interaction_type == 'view':
                view_duration = getattr(interaction, 'view_duration', None)
                if view_duration:
                    if view_duration < 300:  # Less than 5 minutes
                        profile['viewing_behavior']['quick_views'] += 1
                    else:
                        profile['viewing_behavior']['extended_views'] += 1
            
            # Rating patterns
            if interaction.rating:
                profile['rating_patterns'][content.content_type].append(interaction.rating)
                
            # Quality preference (based on content rating)
            if content.rating:
                if interaction_type in ['like', 'favorite', 'watchlist']:
                    profile['quality_preference'] += content.rating * weight * 0.1
                    
            # Popularity preference
            if content.popularity:
                if interaction_type in ['like', 'favorite']:
                    profile['popularity_preference'] += math.log(content.popularity + 1) * weight * 0.1
                    
            # Decade preferences
            if content.release_date:
                decade = (content.release_date.year // 10) * 10
                profile['preferred_decades'][decade] += weight
                
            # Runtime preferences
            if content.runtime:
                runtime_category = self._categorize_runtime(content.runtime)
                profile['runtime_preferences'][runtime_category] += weight
                
            profile['interaction_weights'][interaction_type] += weight
        
        # Calculate derived metrics
        profile = self._calculate_derived_metrics(profile, interactions)
        
        # Cache the profile
        self.user_profiles[user_id] = profile
        return profile
    
    def _calculate_advanced_interaction_weight(self, interaction):
        """Calculate sophisticated interaction weight"""
        base_weights = {
            'search': 0.5,
            'view': 1.0,
            'like': 2.5,
            'favorite': 4.0,
            'watchlist': 3.0,
            'rating': 0.0,  # Will be calculated based on actual rating
            'share': 2.0,
            'download': 3.5
        }
        
        base_weight = base_weights.get(interaction.interaction_type, 1.0)
        
        # Rating-based weight
        if interaction.rating:
            rating_weight = interaction.rating * 0.8
            base_weight = max(base_weight, rating_weight)
        
        # Time decay (recent interactions are more important)
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        time_decay = math.exp(-days_ago / 60.0)  # 60-day half-life
        
        # Frequency boost (repeated interactions on same content)
        frequency_boost = 1.0
        if hasattr(interaction, 'frequency'):
            frequency_boost = 1 + math.log(interaction.frequency + 1) * 0.3
        
        return base_weight * time_decay * frequency_boost
    
    def _categorize_runtime(self, runtime):
        """Categorize content by runtime"""
        if runtime < 90:
            return 'short'
        elif runtime < 150:
            return 'medium'
        else:
            return 'long'
    
    def _calculate_derived_metrics(self, profile, interactions):
        """Calculate advanced derived metrics"""
        if not interactions:
            return profile
        
        # Calculate average ratings by content type
        for content_type, ratings in profile['rating_patterns'].items():
            if ratings:
                profile['rating_patterns'][content_type] = {
                    'average': np.mean(ratings),
                    'std': np.std(ratings),
                    'count': len(ratings),
                    'min': min(ratings),
                    'max': max(ratings)
                }
        
        # Calculate diversity score
        genre_count = len(profile['genre_preferences'])
        language_count = len(profile['language_preferences'])
        type_count = len(profile['content_type_preferences'])
        
        profile['diversity_score'] = (genre_count * 0.5 + language_count * 0.3 + type_count * 0.2) / 10.0
        
        # Normalize preferences
        total_genre_weight = sum(profile['genre_preferences'].values())
        if total_genre_weight > 0:
            for genre in profile['genre_preferences']:
                profile['genre_preferences'][genre] /= total_genre_weight
        
        total_lang_weight = sum(profile['language_preferences'].values())
        if total_lang_weight > 0:
            for lang in profile['language_preferences']:
                profile['language_preferences'][lang] /= total_lang_weight
        
        # Calculate completion rate
        total_views = profile['viewing_behavior']['quick_views'] + profile['viewing_behavior']['extended_views']
        if total_views > 0:
            profile['viewing_behavior']['completion_rate'] = profile['viewing_behavior']['extended_views'] / total_views
        
        return profile

class NeuralCollaborativeFiltering:
    def __init__(self, db, models, embedding_dim=64, hidden_dims=[128, 64, 32]):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.model = None
        
    def prepare_neural_data(self):
        """Prepare data for neural collaborative filtering"""
        interactions = self.UserInteraction.query.all()
        
        data = []
        for interaction in interactions:
            weight = self._calculate_interaction_weight(interaction)
            data.append({
                'user_id': interaction.user_id,
                'item_id': interaction.content_id,
                'rating': weight,
                'timestamp': interaction.timestamp
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return None, None, None
        
        # Create mappings
        unique_users = df['user_id'].unique()
        unique_items = df['item_id'].unique()
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        # Create features
        user_features = []
        item_features = []
        ratings = []
        
        for _, row in df.iterrows():
            user_idx = user_to_idx[row['user_id']]
            item_idx = item_to_idx[row['item_id']]
            
            # User features (embedding indices + metadata)
            user_feat = [user_idx]
            user = self.User.query.get(row['user_id'])
            if user:
                # Add user metadata features
                user_feat.extend([
                    len(json.loads(user.preferred_genres or '[]')),
                    len(json.loads(user.preferred_languages or '[]')),
                    (datetime.utcnow() - user.created_at).days
                ])
            else:
                user_feat.extend([0, 0, 0])
            
            # Item features
            item_feat = [item_idx]
            content = self.Content.query.get(row['item_id'])
            if content:
                item_feat.extend([
                    content.rating or 0,
                    content.popularity or 0,
                    content.vote_count or 0,
                    len(json.loads(content.genres or '[]')),
                    content.runtime or 0
                ])
            else:
                item_feat.extend([0, 0, 0, 0, 0])
            
            user_features.append(user_feat)
            item_features.append(item_feat)
            ratings.append(row['rating'])
        
        return np.array(user_features), np.array(item_features), np.array(ratings), user_to_idx, item_to_idx
    
    def train_neural_model(self):
        """Train neural collaborative filtering model"""
        user_features, item_features, ratings, user_to_idx, item_to_idx = self.prepare_neural_data()
        
        if user_features is None:
            return False
        
        # Combine user and item features
        combined_features = np.concatenate([user_features, item_features], axis=1)
        
        # Train neural network
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_dims,
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        self.model.fit(combined_features, ratings)
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        
        return True
    
    def get_neural_recommendations(self, user_id, limit=20):
        """Get recommendations using neural collaborative filtering"""
        if not self.model or user_id not in self.user_to_idx:
            return []
        
        user_interactions = set(
            interaction.content_id for interaction in 
            self.UserInteraction.query.filter_by(user_id=user_id).all()
        )
        
        user = self.User.query.get(user_id)
        user_idx = self.user_to_idx[user_id]
        
        recommendations = []
        
        for item_id, item_idx in self.item_to_idx.items():
            if item_id in user_interactions:
                continue
            
            # Prepare features
            user_feat = [user_idx]
            if user:
                user_feat.extend([
                    len(json.loads(user.preferred_genres or '[]')),
                    len(json.loads(user.preferred_languages or '[]')),
                    (datetime.utcnow() - user.created_at).days
                ])
            else:
                user_feat.extend([0, 0, 0])
            
            content = self.Content.query.get(item_id)
            item_feat = [item_idx]
            if content:
                item_feat.extend([
                    content.rating or 0,
                    content.popularity or 0,
                    content.vote_count or 0,
                    len(json.loads(content.genres or '[]')),
                    content.runtime or 0
                ])
            else:
                item_feat.extend([0, 0, 0, 0, 0])
            
            combined_feat = np.array([user_feat + item_feat])
            predicted_rating = self.model.predict(combined_feat)[0]
            
            recommendations.append((item_id, predicted_rating))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]
    
    def _calculate_interaction_weight(self, interaction):
        """Calculate interaction weight for neural model"""
        weights = {
            'search': 1.0,
            'view': 2.0,
            'like': 4.0,
            'favorite': 5.0,
            'watchlist': 4.5,
            'rating': interaction.rating if interaction.rating else 3.0,
            'share': 3.5
        }
        
        base_weight = weights.get(interaction.interaction_type, 1.0)
        
        # Time decay
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        time_decay = math.exp(-days_ago / 30.0)
        
        return base_weight * time_decay

class AdvancedContentAnalyzer:
    def __init__(self, db, models):
        self.db = db
        self.Content = models['Content']
        self.content_vectors = {}
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def extract_advanced_content_features(self, content_id):
        """Extract comprehensive content features"""
        content = self.Content.query.get(content_id)
        if not content:
            return None
        
        if content_id in self.content_vectors:
            return self.content_vectors[content_id]
        
        features = {
            'basic': self._extract_basic_features(content),
            'textual': self._extract_textual_features(content),
            'temporal': self._extract_temporal_features(content),
            'quality': self._extract_quality_features(content),
            'popularity': self._extract_popularity_features(content),
            'categorical': self._extract_categorical_features(content)
        }
        
        # Combine all features into a single vector
        feature_vector = self._combine_features(features)
        
        self.content_vectors[content_id] = feature_vector
        return feature_vector
    
    def _extract_basic_features(self, content):
        """Extract basic content features"""
        features = {}
        
        # Genres (one-hot encoded)
        all_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
                     'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
                     'Romance', 'Science Fiction', 'Thriller', 'War', 'Western']
        
        content_genres = json.loads(content.genres or '[]')
        for genre in all_genres:
            features[f'genre_{genre}'] = 1.0 if genre in content_genres else 0.0
        
        # Languages (one-hot encoded)
        all_languages = ['en', 'hi', 'te', 'ta', 'ml', 'kn', 'ja', 'ko', 'fr', 'es', 'de']
        content_languages = json.loads(content.languages or '[]')
        for lang in all_languages:
            features[f'lang_{lang}'] = 1.0 if lang in content_languages else 0.0
        
        # Content type
        features['is_movie'] = 1.0 if content.content_type == 'movie' else 0.0
        features['is_tv'] = 1.0 if content.content_type == 'tv' else 0.0
        features['is_anime'] = 1.0 if content.content_type == 'anime' else 0.0
        
        return features
    
    def _extract_textual_features(self, content):
        """Extract features from text fields"""
        features = {}
        
        # Overview sentiment and complexity
        if content.overview:
            blob = TextBlob(content.overview)
            features['overview_sentiment'] = blob.sentiment.polarity
            features['overview_subjectivity'] = blob.sentiment.subjectivity
            features['overview_length'] = len(content.overview)
            features['overview_word_count'] = len(content.overview.split())
        else:
            features.update({
                'overview_sentiment': 0.0,
                'overview_subjectivity': 0.0,
                'overview_length': 0.0,
                'overview_word_count': 0.0
            })
        
        return features
    
    def _extract_temporal_features(self, content):
        """Extract temporal features"""
        features = {}
        
        if content.release_date:
            release_year = content.release_date.year
            current_year = datetime.now().year
            
            features['release_year'] = release_year
            features['age_years'] = current_year - release_year
            features['decade'] = (release_year // 10) * 10
            features['is_recent'] = 1.0 if (current_year - release_year) <= 3 else 0.0
            features['is_classic'] = 1.0 if (current_year - release_year) >= 20 else 0.0
        else:
            features.update({
                'release_year': 2000,
                'age_years': 20,
                'decade': 2000,
                'is_recent': 0.0,
                'is_classic': 0.0
            })
        
        # Runtime features
        if content.runtime:
            features['runtime'] = content.runtime
            features['runtime_normalized'] = min(content.runtime / 180.0, 1.0)
            features['is_short'] = 1.0 if content.runtime < 90 else 0.0
            features['is_long'] = 1.0 if content.runtime > 150 else 0.0
        else:
            features.update({
                'runtime': 120,
                'runtime_normalized': 0.67,
                'is_short': 0.0,
                'is_long': 0.0
            })
        
        return features
    
    def _extract_quality_features(self, content):
        """Extract quality-related features"""
        features = {}
        
        features['rating'] = content.rating or 0.0
        features['rating_normalized'] = (content.rating or 0.0) / 10.0
        features['vote_count'] = content.vote_count or 0
        features['vote_count_log'] = math.log(content.vote_count + 1) if content.vote_count else 0.0
        
        # Quality indicators
        features['is_high_rated'] = 1.0 if (content.rating or 0) >= 8.0 else 0.0
        features['is_well_reviewed'] = 1.0 if (content.vote_count or 0) >= 1000 else 0.0
        features['is_trending'] = 1.0 if content.is_trending else 0.0
        features['is_critics_choice'] = 1.0 if content.is_critics_choice else 0.0
        
        return features
    
    def _extract_popularity_features(self, content):
        """Extract popularity features"""
        features = {}
        
        features['popularity'] = content.popularity or 0.0
        features['popularity_log'] = math.log(content.popularity + 1) if content.popularity else 0.0
        features['popularity_normalized'] = min((content.popularity or 0.0) / 1000.0, 1.0)
        
        return features
    
    def _extract_categorical_features(self, content):
        """Extract categorical features"""
        features = {}
        
        # Content flags
        features['is_new_release'] = 1.0 if content.is_new_release else 0.0
        
        # Anime-specific features
        if content.content_type == 'anime':
            anime_genres = json.loads(content.anime_genres or '[]')
            features['has_action_anime'] = 1.0 if 'Action' in anime_genres else 0.0
            features['has_romance_anime'] = 1.0 if 'Romance' in anime_genres else 0.0
            features['has_comedy_anime'] = 1.0 if 'Comedy' in anime_genres else 0.0
        else:
            features.update({
                'has_action_anime': 0.0,
                'has_romance_anime': 0.0,
                'has_comedy_anime': 0.0
            })
        
        return features
    
    def _combine_features(self, features_dict):
        """Combine all features into a single vector"""
        combined = {}
        for category, features in features_dict.items():
            for key, value in features.items():
                combined[f"{category}_{key}"] = value
        
        return combined
    
    def calculate_advanced_similarity(self, content_id1, content_id2):
        """Calculate advanced similarity between content"""
        features1 = self.extract_advanced_content_features(content_id1)
        features2 = self.extract_advanced_content_features(content_id2)
        
        if not features1 or not features2:
            return 0.0
        
        # Ensure both feature vectors have same keys
        all_keys = set(features1.keys()) | set(features2.keys())
        
        vector1 = np.array([features1.get(key, 0.0) for key in sorted(all_keys)])
        vector2 = np.array([features2.get(key, 0.0) for key in sorted(all_keys)])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
        
        return max(0.0, similarity)

class HybridRecommendationEngine:
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
        # Initialize components
        self.user_profiler = AdvancedUserProfiler(db, models)
        self.neural_cf = NeuralCollaborativeFiltering(db, models)
        self.content_analyzer = AdvancedContentAnalyzer(db, models)
        
        # Advanced algorithm weights
        self.algorithm_weights = {
            'neural_collaborative': 0.30,
            'advanced_content_based': 0.25,
            'user_profile_matching': 0.20,
            'popularity_boosted': 0.15,
            'novelty_exploration': 0.10
        }
        
    def get_personalized_recommendations(self, user_id, content_type='all', limit=20):
        """Get highly personalized recommendations"""
        try:
            # Build comprehensive user profile
            user_profile = self.user_profiler.build_comprehensive_user_profile(user_id)
            
            # Get recommendations from multiple algorithms
            recommendations = defaultdict(float)
            algorithm_scores = defaultdict(lambda: defaultdict(float))
            
            # 1. Neural Collaborative Filtering
            neural_recs = self._get_neural_recommendations(user_id, limit * 3)
            for content_id, score in neural_recs:
                weight = self.algorithm_weights['neural_collaborative']
                recommendations[content_id] += score * weight
                algorithm_scores[content_id]['neural_collaborative'] = score * weight
            
            # 2. Advanced Content-Based
            content_recs = self._get_advanced_content_recommendations(user_id, user_profile, limit * 3)
            for content_id, score in content_recs:
                weight = self.algorithm_weights['advanced_content_based']
                recommendations[content_id] += score * weight
                algorithm_scores[content_id]['advanced_content_based'] = score * weight
            
            # 3. User Profile Matching
            profile_recs = self._get_profile_based_recommendations(user_id, user_profile, limit * 3)
            for content_id, score in profile_recs:
                weight = self.algorithm_weights['user_profile_matching']
                recommendations[content_id] += score * weight
                algorithm_scores[content_id]['user_profile_matching'] = score * weight
            
            # 4. Popularity Boosted
            popularity_recs = self._get_popularity_boosted_recommendations(user_id, user_profile, limit * 2)
            for content_id, score in popularity_recs:
                weight = self.algorithm_weights['popularity_boosted']
                recommendations[content_id] += score * weight
                algorithm_scores[content_id]['popularity_boosted'] = score * weight
            
            # 5. Novelty Exploration
            novelty_recs = self._get_novelty_recommendations(user_id, user_profile, limit)
            for content_id, score in novelty_recs:
                weight = self.algorithm_weights['novelty_exploration']
                recommendations[content_id] += score * weight
                algorithm_scores[content_id]['novelty_exploration'] = score * weight
            
            # Filter by content type if specified
            if content_type != 'all':
                filtered_recommendations = {}
                for content_id, score in recommendations.items():
                    content = self.Content.query.get(content_id)
                    if content and content.content_type == content_type:
                        filtered_recommendations[content_id] = score
                recommendations = filtered_recommendations
            
            # Sort and return top recommendations
            sorted_recommendations = sorted(
                recommendations.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            # Build detailed response
            detailed_recommendations = []
            for content_id, final_score in sorted_recommendations:
                content = self.Content.query.get(content_id)
                if content:
                    detailed_recommendations.append({
                        'content': content,
                        'score': final_score,
                        'algorithm_breakdown': dict(algorithm_scores[content_id]),
                        'explanation': self._generate_explanation(user_id, content_id, user_profile),
                        'confidence': min(final_score / max(recommendations.values()), 1.0)
                    })
            
            return detailed_recommendations
            
        except Exception as e:
            logger.error(f"Error in personalized recommendations: {e}")
            return []
    
    def _get_neural_recommendations(self, user_id, limit):
        """Get neural collaborative filtering recommendations"""
        try:
            if not self.neural_cf.model:
                self.neural_cf.train_neural_model()
            
            return self.neural_cf.get_neural_recommendations(user_id, limit)
        except Exception as e:
            logger.warning(f"Neural recommendations failed: {e}")
            return []
    
    def _get_advanced_content_recommendations(self, user_id, user_profile, limit):
        """Get advanced content-based recommendations"""
        try:
            user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
            
            if not user_interactions:
                return []
            
            # Get user's preferred content
            liked_content = []
            for interaction in user_interactions:
                weight = self.user_profiler._calculate_advanced_interaction_weight(interaction)
                if weight >= 2.0:  # Only consider significant interactions
                    liked_content.append((interaction.content_id, weight))
            
            if not liked_content:
                return []
            
            # Find similar content
            recommendations = defaultdict(float)
            user_seen_content = set(interaction.content_id for interaction in user_interactions)
            
            # Get all potential content
            all_content = self.Content.query.limit(5000).all()
            
            for content in all_content:
                if content.id in user_seen_content:
                    continue
                
                content_score = 0.0
                total_weight = 0.0
                
                for liked_content_id, user_weight in liked_content:
                    similarity = self.content_analyzer.calculate_advanced_similarity(
                        content.id, liked_content_id
                    )
                    content_score += similarity * user_weight
                    total_weight += user_weight
                
                if total_weight > 0:
                    normalized_score = content_score / total_weight
                    
                    # Apply user preference boosts
                    preference_boost = self._calculate_preference_boost(content, user_profile)
                    final_score = normalized_score * preference_boost
                    
                    recommendations[content.id] = final_score
            
            return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:limit]
            
        except Exception as e:
            logger.warning(f"Advanced content recommendations failed: {e}")
            return []
    
    def _get_profile_based_recommendations(self, user_id, user_profile, limit):
        """Get recommendations based on detailed user profile"""
        try:
            recommendations = defaultdict(float)
            user_interactions = set(
                interaction.content_id for interaction in 
                self.UserInteraction.query.filter_by(user_id=user_id).all()
            )
            
            # Query content based on user preferences
            query = self.Content.query
            
            # Get content that matches user's top preferences
            top_genres = sorted(
                user_profile['genre_preferences'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            top_languages = sorted(
                user_profile['language_preferences'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            # Sample content from database
            all_content = query.limit(3000).all()
            
            for content in all_content:
                if content.id in user_interactions:
                    continue
                
                score = self._calculate_profile_match_score(content, user_profile)
                
                if score > 0.1:  # Only consider reasonable matches
                    recommendations[content.id] = score
            
            return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:limit]
            
        except Exception as e:
            logger.warning(f"Profile-based recommendations failed: {e}")
            return []
    
    def _get_popularity_boosted_recommendations(self, user_id, user_profile, limit):
        """Get popularity-based recommendations with user preference boost"""
        try:
            user_interactions = set(
                interaction.content_id for interaction in 
                self.UserInteraction.query.filter_by(user_id=user_id).all()
            )
            
            # Get popular content
            popular_content = self.Content.query.filter(
                self.Content.popularity > 100
            ).order_by(
                self.Content.popularity.desc()
            ).limit(limit * 3).all()
            
            recommendations = []
            
            for content in popular_content:
                if content.id in user_interactions:
                    continue
                
                # Base popularity score
                popularity_score = math.log(content.popularity + 1) / 10.0
                
                # Apply user preference boost
                preference_boost = self._calculate_preference_boost(content, user_profile)
                
                final_score = popularity_score * preference_boost
                recommendations.append((content.id, final_score))
            
            return sorted(recommendations, key=lambda x: x[1], reverse=True)[:limit]
            
        except Exception as e:
            logger.warning(f"Popularity-boosted recommendations failed: {e}")
            return []
    
    def _get_novelty_recommendations(self, user_id, user_profile, limit):
        """Get novel recommendations for exploration"""
        try:
            user_interactions = set(
                interaction.content_id for interaction in 
                self.UserInteraction.query.filter_by(user_id=user_id).all()
            )
            
            # Get diverse content that user hasn't explored
            recommendations = []
            
            # Find content from unexplored genres
            user_genres = set(user_profile['genre_preferences'].keys())
            all_content = self.Content.query.limit(2000).all()
            
            for content in all_content:
                if content.id in user_interactions:
                    continue
                
                try:
                    content_genres = set(json.loads(content.genres or '[]'))
                    
                    # Calculate novelty score
                    novelty_score = len(content_genres - user_genres) / max(len(content_genres), 1)
                    
                    # Ensure quality threshold
                    if (content.rating or 0) >= 6.0 and novelty_score > 0.3:
                        # Boost score based on content quality
                        quality_boost = (content.rating or 0) / 10.0
                        final_score = novelty_score * quality_boost
                        
                        recommendations.append((content.id, final_score))
                        
                except:
                    continue
            
            return sorted(recommendations, key=lambda x: x[1], reverse=True)[:limit]
            
        except Exception as e:
            logger.warning(f"Novelty recommendations failed: {e}")
            return []
    
    def _calculate_preference_boost(self, content, user_profile):
        """Calculate preference boost based on user profile"""
        boost = 1.0
        
        try:
            # Genre preference boost
            content_genres = set(json.loads(content.genres or '[]'))
            for genre in content_genres:
                if genre in user_profile['genre_preferences']:
                    boost += user_profile['genre_preferences'][genre] * 0.5
            
            # Language preference boost
            content_languages = set(json.loads(content.languages or '[]'))
            for language in content_languages:
                if language in user_profile['language_preferences']:
                    boost += user_profile['language_preferences'][language] * 0.3
            
            # Content type preference boost
            if content.content_type in user_profile['content_type_preferences']:
                boost += user_profile['content_type_preferences'][content.content_type] * 0.2
            
            # Quality preference boost
            if content.rating and user_profile['quality_preference'] > 0:
                quality_match = abs(content.rating - (user_profile['quality_preference'] * 10)) / 10.0
                boost += (1 - quality_match) * 0.2
                
        except Exception as e:
            logger.warning(f"Error calculating preference boost: {e}")
        
        return boost
    
    def _calculate_profile_match_score(self, content, user_profile):
        """Calculate how well content matches user profile"""
        score = 0.0
        
        try:
            # Genre matching
            content_genres = set(json.loads(content.genres or '[]'))
            for genre in content_genres:
                if genre in user_profile['genre_preferences']:
                    score += user_profile['genre_preferences'][genre] * 0.4
            
            # Language matching
            content_languages = set(json.loads(content.languages or '[]'))
            for language in content_languages:
                if language in user_profile['language_preferences']:
                    score += user_profile['language_preferences'][language] * 0.3
            
            # Content type matching
            if content.content_type in user_profile['content_type_preferences']:
                score += user_profile['content_type_preferences'][content.content_type] * 0.2
            
            # Quality matching
            if content.rating:
                expected_quality = user_profile['quality_preference']
                if expected_quality > 0:
                    quality_score = 1 - abs(content.rating - expected_quality) / 10.0
                    score += max(0, quality_score) * 0.1
                    
        except Exception as e:
            logger.warning(f"Error calculating profile match score: {e}")
        
        return max(0.0, score)
    
    def _generate_explanation(self, user_id, content_id, user_profile):
        """Generate explanation for recommendation"""
        try:
            content = self.Content.query.get(content_id)
            if not content:
                return "Recommended for you"
            
            explanations = []
            
            # Genre-based explanation
            try:
                content_genres = set(json.loads(content.genres or '[]'))
                top_user_genres = sorted(
                    user_profile['genre_preferences'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                
                matching_genres = content_genres & set([g[0] for g in top_user_genres])
                if matching_genres:
                    explanations.append(f"You enjoy {', '.join(list(matching_genres)[:2])}")
            except:
                pass
            
            # Content type explanation
            if content.content_type in user_profile['content_type_preferences']:
                preference_strength = user_profile['content_type_preferences'][content.content_type]
                if preference_strength > 0.3:
                    explanations.append(f"You frequently watch {content.content_type}s")
            
            # Quality explanation
            if content.rating and content.rating >= 8.0:
                explanations.append("Highly rated content")
            
            # Popularity explanation
            if content.popularity and content.popularity > 500:
                explanations.append("Popular among users with similar tastes")
            
            if not explanations:
                return "Recommended based on your viewing history"
            
            return "; ".join(explanations[:3])
            
        except Exception as e:
            logger.warning(f"Error generating explanation: {e}")
            return "Personalized recommendation"

class EvaluationMetrics:
    @staticmethod
    def precision_at_k(user_interactions, recommendations, k=10):
        """Calculate precision at k"""
        if not recommendations or k <= 0:
            return 0.0
        
        top_k = [rec[0] if isinstance(rec, tuple) else rec for rec in recommendations[:k]]
        relevant_items = set(interaction.content_id for interaction in user_interactions 
                           if interaction.interaction_type in ['like', 'favorite', 'rating'] 
                           and (not interaction.rating or interaction.rating >= 4.0))
        
        relevant_in_top_k = len(set(top_k) & relevant_items)
        return relevant_in_top_k / k
    
    @staticmethod
    def diversity_score(contents):
        """Calculate diversity score of content list"""
        if len(contents) <= 1:
            return 0.0
        
        genre_sets = []
        for content in contents:
            try:
                genres = set(json.loads(content.genres or '[]'))
                genre_sets.append(genres)
            except:
                genre_sets.append(set())
        
        total_pairs = len(genre_sets) * (len(genre_sets) - 1) / 2
        diverse_pairs = 0
        
        for i in range(len(genre_sets)):
            for j in range(i + 1, len(genre_sets)):
                overlap = len(genre_sets[i] & genre_sets[j])
                union = len(genre_sets[i] | genre_sets[j])
                if union > 0 and overlap / union < 0.5:
                    diverse_pairs += 1
        
        return diverse_pairs / total_pairs if total_pairs > 0 else 0.0
    
    @staticmethod
    def coverage_score(recommended_items, total_items):
        """Calculate catalog coverage"""
        if total_items <= 0:
            return 0.0
        return len(set(recommended_items)) / total_items