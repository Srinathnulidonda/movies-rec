import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jaccard
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class EnhancedCollaborativeFiltering:
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.user_similarity_cache = {}
        self.item_similarity_cache = {}
        self.user_behavior_profiles = {}
        
    def get_enhanced_user_item_matrix(self, interaction_types=None, temporal_weight=True):
        if interaction_types is None:
            interaction_types = ['view', 'like', 'favorite', 'rating', 'watchlist', 'search']
        
        interactions = self.UserInteraction.query.filter(
            self.UserInteraction.interaction_type.in_(interaction_types)
        ).all()
        
        user_item_dict = defaultdict(dict)
        for interaction in interactions:
            weight = self._calculate_enhanced_interaction_weight(interaction, temporal_weight)
            if interaction.content_id in user_item_dict[interaction.user_id]:
                user_item_dict[interaction.user_id][interaction.content_id] = max(
                    user_item_dict[interaction.user_id][interaction.content_id], weight
                )
            else:
                user_item_dict[interaction.user_id][interaction.content_id] = weight
        
        users = list(user_item_dict.keys())
        items = list(set().union(*[items.keys() for items in user_item_dict.values()]))
        
        matrix = np.zeros((len(users), len(items)))
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        item_to_idx = {item: idx for idx, item in enumerate(items)}
        
        for user_id, items_dict in user_item_dict.items():
            for item_id, weight in items_dict.items():
                matrix[user_to_idx[user_id]][item_to_idx[item_id]] = weight
        
        return matrix, user_to_idx, item_to_idx
    
    def _calculate_enhanced_interaction_weight(self, interaction, temporal_weight=True):
        base_weights = {
            'view': 1.0,
            'like': 3.0,
            'favorite': 5.0,
            'watchlist': 4.0,
            'rating': (interaction.rating or 3.0) * 0.8,
            'search': 0.8,
            'share': 2.5,
            'download': 3.5
        }
        
        base_weight = base_weights.get(interaction.interaction_type, 1.0)
        
        if interaction.interaction_metadata:
            metadata = interaction.interaction_metadata
            
            if interaction.interaction_type == 'view':
                duration = metadata.get('view_duration', 0)
                completion = metadata.get('completion_percentage', 0)
                if duration > 300:
                    base_weight *= 1.5
                if completion > 80:
                    base_weight *= 1.8
                elif completion > 50:
                    base_weight *= 1.4
            
            if interaction.interaction_type == 'search':
                if metadata.get('clicked_result', False):
                    base_weight *= 2.0
        
        if temporal_weight:
            days_ago = (datetime.utcnow() - interaction.timestamp).days
            if days_ago <= 7:
                time_multiplier = 1.5
            elif days_ago <= 30:
                time_multiplier = 1.2
            elif days_ago <= 90:
                time_multiplier = 1.0
            else:
                time_multiplier = math.exp(-days_ago / 180.0)
            
            base_weight *= time_multiplier
        
        return min(base_weight, 10.0)
    
    def calculate_advanced_user_similarity(self, user_id, similarity_threshold=0.1):
        cache_key = f"{user_id}_advanced"
        if cache_key in self.user_similarity_cache:
            return self.user_similarity_cache[cache_key]
        
        matrix, user_to_idx, item_to_idx = self.get_enhanced_user_item_matrix()
        
        if user_id not in user_to_idx:
            return {}
        
        user_idx = user_to_idx[user_id]
        user_vector = matrix[user_idx].reshape(1, -1)
        
        similarities = cosine_similarity(user_vector, matrix)[0]
        
        user_profile = self._get_user_behavior_profile(user_id)
        
        result = {}
        for other_user_id, other_idx in user_to_idx.items():
            if other_user_id != user_id:
                cosine_sim = similarities[other_idx]
                if cosine_sim > similarity_threshold:
                    other_profile = self._get_user_behavior_profile(other_user_id)
                    profile_similarity = self._calculate_profile_similarity(user_profile, other_profile)
                    
                    combined_similarity = 0.7 * cosine_sim + 0.3 * profile_similarity
                    result[other_user_id] = combined_similarity
        
        self.user_similarity_cache[cache_key] = result
        return result
    
    def _get_user_behavior_profile(self, user_id):
        if user_id in self.user_behavior_profiles:
            return self.user_behavior_profiles[user_id]
        
        interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        profile = {
            'genre_preferences': Counter(),
            'language_preferences': Counter(),
            'content_type_preferences': Counter(),
            'rating_patterns': [],
            'interaction_frequency': defaultdict(int),
            'temporal_patterns': defaultdict(int),
            'avg_rating': 0.0
        }
        
        for interaction in interactions:
            content = self.Content.query.get(interaction.content_id)
            if content:
                try:
                    genres = json.loads(content.genres or '[]')
                    for genre in genres:
                        weight = self._calculate_enhanced_interaction_weight(interaction, False)
                        profile['genre_preferences'][genre] += weight
                except:
                    pass
                
                try:
                    languages = json.loads(content.languages or '[]')
                    for language in languages:
                        weight = self._calculate_enhanced_interaction_weight(interaction, False)
                        profile['language_preferences'][language] += weight
                except:
                    pass
                
                weight = self._calculate_enhanced_interaction_weight(interaction, False)
                profile['content_type_preferences'][content.content_type] += weight
                
                if interaction.rating:
                    profile['rating_patterns'].append(interaction.rating)
                
                profile['interaction_frequency'][interaction.interaction_type] += 1
                
                hour = interaction.timestamp.hour
                profile['temporal_patterns'][hour] += 1
        
        if profile['rating_patterns']:
            profile['avg_rating'] = sum(profile['rating_patterns']) / len(profile['rating_patterns'])
        
        self.user_behavior_profiles[user_id] = profile
        return profile
    
    def _calculate_profile_similarity(self, profile1, profile2):
        similarities = []
        
        genres1 = set(profile1['genre_preferences'].keys())
        genres2 = set(profile2['genre_preferences'].keys())
        if genres1 and genres2:
            genre_overlap = len(genres1 & genres2) / len(genres1 | genres2)
            similarities.append(genre_overlap * 0.4)
        
        langs1 = set(profile1['language_preferences'].keys())
        langs2 = set(profile2['language_preferences'].keys())
        if langs1 and langs2:
            lang_overlap = len(langs1 & langs2) / len(langs1 | langs2)
            similarities.append(lang_overlap * 0.3)
        
        types1 = set(profile1['content_type_preferences'].keys())
        types2 = set(profile2['content_type_preferences'].keys())
        if types1 and types2:
            type_overlap = len(types1 & types2) / len(types1 | types2)
            similarities.append(type_overlap * 0.2)
        
        if profile1['avg_rating'] > 0 and profile2['avg_rating'] > 0:
            rating_similarity = 1 - abs(profile1['avg_rating'] - profile2['avg_rating']) / 10.0
            similarities.append(rating_similarity * 0.1)
        
        return sum(similarities) if similarities else 0.0
    
    def get_precision_user_recommendations(self, user_id, limit=30):
        similarities = self.calculate_advanced_user_similarity(user_id)
        
        if not similarities:
            return []
        
        similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:100]
        
        user_interactions = set(
            interaction.content_id for interaction in 
            self.UserInteraction.query.filter_by(user_id=user_id).all()
        )
        
        recommendations = defaultdict(float)
        recommendation_sources = defaultdict(list)
        
        for similar_user_id, similarity_score in similar_users:
            if similarity_score < 0.2:
                continue
            
            similar_user_interactions = self.UserInteraction.query.filter_by(
                user_id=similar_user_id
            ).all()
            
            for interaction in similar_user_interactions:
                if interaction.content_id not in user_interactions:
                    weight = self._calculate_enhanced_interaction_weight(interaction)
                    score = similarity_score * weight
                    recommendations[interaction.content_id] += score
                    recommendation_sources[interaction.content_id].append({
                        'similar_user': similar_user_id,
                        'similarity': similarity_score,
                        'interaction_type': interaction.interaction_type,
                        'weight': weight
                    })
        
        user_profile = self._get_user_behavior_profile(user_id)
        
        enhanced_recommendations = []
        for content_id, base_score in recommendations.items():
            content = self.Content.query.get(content_id)
            if content:
                profile_match = self._calculate_content_profile_match(content, user_profile)
                final_score = base_score * (1 + profile_match)
                
                enhanced_recommendations.append((content_id, final_score, {
                    'base_score': base_score,
                    'profile_match': profile_match,
                    'sources': recommendation_sources[content_id][:3]
                }))
        
        enhanced_recommendations.sort(key=lambda x: x[1], reverse=True)
        return enhanced_recommendations[:limit]
    
    def _calculate_content_profile_match(self, content, user_profile):
        match_score = 0.0
        
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            user_top_genres = set([genre for genre, _ in user_profile['genre_preferences'].most_common(5)])
            if content_genres and user_top_genres:
                genre_match = len(content_genres & user_top_genres) / len(content_genres)
                match_score += genre_match * 0.4
        except:
            pass
        
        try:
            content_languages = set(json.loads(content.languages or '[]'))
            user_top_languages = set([lang for lang, _ in user_profile['language_preferences'].most_common(3)])
            if content_languages and user_top_languages:
                lang_match = len(content_languages & user_top_languages) / len(content_languages)
                match_score += lang_match * 0.3
        except:
            pass
        
        if content.content_type in user_profile['content_type_preferences']:
            type_weight = user_profile['content_type_preferences'][content.content_type]
            total_type_weight = sum(user_profile['content_type_preferences'].values())
            if total_type_weight > 0:
                type_match = type_weight / total_type_weight
                match_score += type_match * 0.3
        
        return match_score

class AdvancedContentBasedFiltering:
    def __init__(self, db, models):
        self.db = db
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.content_features_cache = {}
        self.user_content_profiles = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
        
    def build_comprehensive_content_features(self, content_id):
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
            'overview': content.overview or '',
            'vote_count': content.vote_count or 0,
            'is_trending': content.is_trending or False,
            'is_new_release': content.is_new_release or False,
            'is_critics_choice': content.is_critics_choice or False
        }
        
        try:
            genres = json.loads(content.genres or '[]')
            features['genres'] = set(genres)
        except:
            pass
        
        try:
            languages = json.loads(content.languages or '[]')
            features['languages'] = set(languages)
        except:
            pass
        
        if content.release_date:
            features['release_year'] = content.release_date.year
        
        features['decade'] = features['release_year'] // 10 * 10 if features['release_year'] > 0 else 0
        
        if features['runtime'] > 0:
            if features['runtime'] < 90:
                features['duration_category'] = 'short'
            elif features['runtime'] < 150:
                features['duration_category'] = 'medium'
            else:
                features['duration_category'] = 'long'
        else:
            features['duration_category'] = 'unknown'
        
        if features['rating'] >= 8.0:
            features['quality_tier'] = 'excellent'
        elif features['rating'] >= 7.0:
            features['quality_tier'] = 'good'
        elif features['rating'] >= 6.0:
            features['quality_tier'] = 'average'
        else:
            features['quality_tier'] = 'below_average'
        
        self.content_features_cache[content_id] = features
        return features
    
    def build_user_content_profile(self, user_id):
        if user_id in self.user_content_profiles:
            return self.user_content_profiles[user_id]
        
        interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        profile = {
            'preferred_genres': Counter(),
            'preferred_languages': Counter(),
            'content_type_preferences': Counter(),
            'quality_preferences': Counter(),
            'duration_preferences': Counter(),
            'decade_preferences': Counter(),
            'rating_distribution': [],
            'interaction_patterns': defaultdict(float),
            'content_features_liked': defaultdict(float)
        }
        
        total_weight = 0
        
        for interaction in interactions:
            content_features = self.build_comprehensive_content_features(interaction.content_id)
            if not content_features:
                continue
            
            weight = self._calculate_interaction_weight(interaction)
            total_weight += weight
            
            for genre in content_features['genres']:
                profile['preferred_genres'][genre] += weight
            
            for language in content_features['languages']:
                profile['preferred_languages'][language] += weight
            
            profile['content_type_preferences'][content_features['content_type']] += weight
            profile['quality_preferences'][content_features['quality_tier']] += weight
            profile['duration_preferences'][content_features['duration_category']] += weight
            
            if content_features['decade'] > 0:
                profile['decade_preferences'][content_features['decade']] += weight
            
            if interaction.rating:
                profile['rating_distribution'].append(interaction.rating)
            
            profile['interaction_patterns'][interaction.interaction_type] += weight
            
            for feature_key, feature_value in content_features.items():
                if isinstance(feature_value, (int, float, bool)):
                    profile['content_features_liked'][feature_key] += weight * float(feature_value)
        
        if total_weight > 0:
            for key in ['preferred_genres', 'preferred_languages', 'content_type_preferences', 
                       'quality_preferences', 'duration_preferences', 'decade_preferences']:
                for item in profile[key]:
                    profile[key][item] /= total_weight
        
        profile['avg_rating'] = sum(profile['rating_distribution']) / len(profile['rating_distribution']) if profile['rating_distribution'] else 0
        profile['rating_variance'] = np.var(profile['rating_distribution']) if len(profile['rating_distribution']) > 1 else 0
        
        self.user_content_profiles[user_id] = profile
        return profile
    
    def calculate_advanced_content_similarity(self, user_id, content_id):
        user_profile = self.build_user_content_profile(user_id)
        content_features = self.build_comprehensive_content_features(content_id)
        
        if not user_profile or not content_features:
            return 0.0
        
        similarity_components = []
        
        genre_score = 0.0
        for genre in content_features['genres']:
            if genre in user_profile['preferred_genres']:
                genre_score += user_profile['preferred_genres'][genre]
        
        if content_features['genres']:
            genre_score /= len(content_features['genres'])
        similarity_components.append(genre_score * 0.35)
        
        language_score = 0.0
        for language in content_features['languages']:
            if language in user_profile['preferred_languages']:
                language_score += user_profile['preferred_languages'][language]
        
        if content_features['languages']:
            language_score /= len(content_features['languages'])
        similarity_components.append(language_score * 0.25)
        
        content_type_score = user_profile['content_type_preferences'].get(content_features['content_type'], 0)
        similarity_components.append(content_type_score * 0.15)
        
        quality_score = user_profile['quality_preferences'].get(content_features['quality_tier'], 0)
        similarity_components.append(quality_score * 0.1)
        
        duration_score = user_profile['duration_preferences'].get(content_features['duration_category'], 0)
        similarity_components.append(duration_score * 0.05)
        
        decade_score = user_profile['decade_preferences'].get(content_features['decade'], 0)
        similarity_components.append(decade_score * 0.05)
        
        if user_profile['avg_rating'] > 0 and content_features['rating'] > 0:
            rating_diff = abs(user_profile['avg_rating'] - content_features['rating'])
            rating_score = max(0, 1 - rating_diff / 5.0)
            similarity_components.append(rating_score * 0.05)
        
        return sum(similarity_components)
    
    def get_precision_content_recommendations(self, user_id, limit=30):
        user_interactions = set(
            interaction.content_id for interaction in 
            self.UserInteraction.query.filter_by(user_id=user_id).all()
        )
        
        if not user_interactions:
            return []
        
        user_profile = self.build_user_content_profile(user_id)
        
        candidate_content = self.Content.query.limit(5000).all()
        
        recommendations = []
        
        for content in candidate_content:
            if content.id not in user_interactions:
                similarity_score = self.calculate_advanced_content_similarity(user_id, content.id)
                
                if similarity_score > 0.1:
                    boost_factors = self._calculate_boost_factors(content, user_profile)
                    final_score = similarity_score * boost_factors
                    
                    recommendations.append((content.id, final_score, {
                        'base_similarity': similarity_score,
                        'boost_factors': boost_factors,
                        'content_match_details': self._get_match_details(content, user_profile)
                    }))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]
    
    def _calculate_boost_factors(self, content, user_profile):
        boost = 1.0
        
        if content.is_trending:
            boost *= 1.1
        
        if content.is_critics_choice:
            boost *= 1.15
        
        if content.rating and content.rating >= 8.0:
            boost *= 1.2
        
        if content.vote_count and content.vote_count >= 1000:
            boost *= 1.05
        
        if content.release_date:
            days_since_release = (datetime.now().date() - content.release_date).days
            if days_since_release <= 90:
                boost *= 1.1
        
        return boost
    
    def _get_match_details(self, content, user_profile):
        details = {}
        
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            user_top_genres = set([genre for genre, _ in user_profile['preferred_genres'].most_common(3)])
            matching_genres = content_genres & user_top_genres
            if matching_genres:
                details['matching_genres'] = list(matching_genres)
        except:
            pass
        
        try:
            content_languages = set(json.loads(content.languages or '[]'))
            user_top_languages = set([lang for lang, _ in user_profile['preferred_languages'].most_common(2)])
            matching_languages = content_languages & user_top_languages
            if matching_languages:
                details['matching_languages'] = list(matching_languages)
        except:
            pass
        
        if content.content_type in user_profile['content_type_preferences']:
            details['content_type_preference'] = user_profile['content_type_preferences'][content.content_type]
        
        return details
    
    def _calculate_interaction_weight(self, interaction):
        weights = {
            'view': 1.0,
            'like': 3.0,
            'favorite': 5.0,
            'watchlist': 4.0,
            'rating': (interaction.rating or 3.0) * 0.8,
            'search': 0.8
        }
        
        base_weight = weights.get(interaction.interaction_type, 1.0)
        
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        time_decay = math.exp(-days_ago / 60.0)
        
        return base_weight * time_decay

class PrecisionMatrixFactorization:
    def __init__(self, db, models, n_factors=100, learning_rate=0.005, regularization=0.02, epochs=150):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_bias = None
        self.trained = False
        
    def prepare_enhanced_data(self):
        interactions = self.UserInteraction.query.all()
        
        ratings = []
        for interaction in interactions:
            weight = self._calculate_enhanced_weight(interaction)
            ratings.append({
                'user_id': interaction.user_id,
                'item_id': interaction.content_id,
                'rating': weight,
                'timestamp': interaction.timestamp
            })
        
        df = pd.DataFrame(ratings)
        
        if df.empty:
            return None, None, None
        
        recent_cutoff = datetime.utcnow() - timedelta(days=365)
        df = df[df['timestamp'] >= recent_cutoff]
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(df['user_id'].unique())}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(df['item_id'].unique())}
        
        n_users = len(user_to_idx)
        n_items = len(item_to_idx)
        
        rating_matrix = np.zeros((n_users, n_items))
        
        for _, row in df.iterrows():
            user_idx = user_to_idx[row['user_id']]
            item_idx = item_to_idx[row['item_id']]
            
            if rating_matrix[user_idx, item_idx] == 0:
                rating_matrix[user_idx, item_idx] = row['rating']
            else:
                rating_matrix[user_idx, item_idx] = max(rating_matrix[user_idx, item_idx], row['rating'])
        
        return rating_matrix, user_to_idx, item_to_idx
    
    def _calculate_enhanced_weight(self, interaction):
        base_weights = {
            'view': 2.0,
            'like': 4.0,
            'favorite': 6.0,
            'watchlist': 5.0,
            'rating': (interaction.rating or 3.0) * 1.2,
            'search': 1.5
        }
        
        weight = base_weights.get(interaction.interaction_type, 2.0)
        
        if interaction.interaction_metadata:
            metadata = interaction.interaction_metadata
            if interaction.interaction_type == 'view':
                completion = metadata.get('completion_percentage', 0)
                if completion > 80:
                    weight *= 1.5
                elif completion > 50:
                    weight *= 1.2
        
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        if days_ago <= 30:
            time_boost = 1.3
        elif days_ago <= 90:
            time_boost = 1.1
        else:
            time_boost = math.exp(-days_ago / 180.0)
        
        return min(weight * time_boost, 10.0)
    
    def train_advanced_model(self):
        rating_matrix, user_to_idx, item_to_idx = self.prepare_enhanced_data()
        
        if rating_matrix is None:
            return False
        
        n_users, n_items = rating_matrix.shape
        
        self.user_factors = np.random.normal(0, 0.05, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.05, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        nonzero_ratings = rating_matrix[rating_matrix > 0]
        self.global_bias = np.mean(nonzero_ratings) if len(nonzero_ratings) > 0 else 3.0
        
        for epoch in range(self.epochs):
            total_error = 0
            num_ratings = 0
            
            for i in range(n_users):
                for j in range(n_items):
                    if rating_matrix[i, j] > 0:
                        prediction = self.predict_rating(i, j)
                        error = rating_matrix[i, j] - prediction
                        total_error += error ** 2
                        num_ratings += 1
                        
                        user_factor_old = self.user_factors[i].copy()
                        
                        self.user_factors[i] += self.learning_rate * (
                            error * self.item_factors[j] - self.regularization * self.user_factors[i]
                        )
                        self.item_factors[j] += self.learning_rate * (
                            error * user_factor_old - self.regularization * self.item_factors[j]
                        )
                        
                        self.user_biases[i] += self.learning_rate * (
                            error - self.regularization * self.user_biases[i]
                        )
                        self.item_biases[j] += self.learning_rate * (
                            error - self.regularization * self.item_biases[j]
                        )
            
            if epoch % 10 == 0 and num_ratings > 0:
                rmse = np.sqrt(total_error / num_ratings)
                logger.info(f"Epoch {epoch}, RMSE: {rmse}")
        
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.trained = True
        return True
    
    def predict_rating(self, user_idx, item_idx):
        prediction = self.global_bias + self.user_biases[user_idx] + self.item_biases[item_idx]
        prediction += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return prediction
    
    def get_precision_recommendations(self, user_id, limit=30):
        if not self.trained:
            if not self.train_advanced_model():
                return []
        
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        user_interactions = set(
            interaction.content_id for interaction in 
            self.UserInteraction.query.filter_by(user_id=user_id).all()
        )
        
        recommendations = []
        
        for item_id, item_idx in self.item_to_idx.items():
            if item_id not in user_interactions:
                predicted_rating = self.predict_rating(user_idx, item_idx)
                
                confidence = self._calculate_confidence(user_idx, item_idx)
                adjusted_score = predicted_rating * confidence
                
                recommendations.append((item_id, adjusted_score, {
                    'predicted_rating': predicted_rating,
                    'confidence': confidence
                }))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]
    
    def _calculate_confidence(self, user_idx, item_idx):
        user_factor_norm = np.linalg.norm(self.user_factors[user_idx])
        item_factor_norm = np.linalg.norm(self.item_factors[item_idx])
        
        confidence = min(user_factor_norm * item_factor_norm / 10.0, 1.0)
        return max(confidence, 0.1)

class BehavioralPatternAnalyzer:
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
    def analyze_user_behavior_patterns(self, user_id):
        interactions = self.UserInteraction.query.filter_by(user_id=user_id).order_by(
            self.UserInteraction.timestamp.asc()
        ).all()
        
        if not interactions:
            return {}
        
        patterns = {
            'interaction_sequence': [],
            'temporal_patterns': defaultdict(int),
            'content_progression': [],
            'preference_evolution': defaultdict(list),
            'engagement_intensity': defaultdict(float),
            'discovery_patterns': defaultdict(int)
        }
        
        for i, interaction in enumerate(interactions):
            patterns['interaction_sequence'].append({
                'type': interaction.interaction_type,
                'content_id': interaction.content_id,
                'timestamp': interaction.timestamp,
                'rating': interaction.rating
            })
            
            hour = interaction.timestamp.hour
            day_of_week = interaction.timestamp.weekday()
            patterns['temporal_patterns'][f"hour_{hour}"] += 1
            patterns['temporal_patterns'][f"day_{day_of_week}"] += 1
            
            content = self.Content.query.get(interaction.content_id)
            if content:
                patterns['content_progression'].append({
                    'content_type': content.content_type,
                    'rating': content.rating,
                    'release_year': content.release_date.year if content.release_date else None,
                    'genres': json.loads(content.genres or '[]')
                })
                
                try:
                    genres = json.loads(content.genres or '[]')
                    for genre in genres:
                        patterns['preference_evolution'][genre].append({
                            'timestamp': interaction.timestamp,
                            'interaction_type': interaction.interaction_type,
                            'user_rating': interaction.rating
                        })
                except:
                    pass
                
                weight = self._calculate_engagement_weight(interaction)
                patterns['engagement_intensity'][content.content_type] += weight
                
                if interaction.interaction_type == 'search':
                    patterns['discovery_patterns']['search_based'] += 1
                elif i == 0 or interactions[i-1].content_id != interaction.content_id:
                    patterns['discovery_patterns']['exploration'] += 1
        
        return self._analyze_patterns(patterns)
    
    def _calculate_engagement_weight(self, interaction):
        weights = {
            'view': 1.0,
            'like': 2.5,
            'favorite': 4.0,
            'watchlist': 3.0,
            'rating': (interaction.rating or 3.0) * 0.8,
            'search': 0.5
        }
        
        return weights.get(interaction.interaction_type, 1.0)
    
    def _analyze_patterns(self, patterns):
        analysis = {}
        
        if patterns['temporal_patterns']:
            peak_hours = sorted(
                [(hour, count) for hour, count in patterns['temporal_patterns'].items() if hour.startswith('hour_')],
                key=lambda x: x[1], reverse=True
            )[:3]
            analysis['peak_activity_hours'] = [int(hour.split('_')[1]) for hour, _ in peak_hours]
            
            peak_days = sorted(
                [(day, count) for day, count in patterns['temporal_patterns'].items() if day.startswith('day_')],
                key=lambda x: x[1], reverse=True
            )[:2]
            analysis['preferred_days'] = [int(day.split('_')[1]) for day, _ in peak_days]
        
        if patterns['engagement_intensity']:
            total_engagement = sum(patterns['engagement_intensity'].values())
            analysis['content_type_preferences'] = {
                content_type: round(intensity / total_engagement, 3)
                for content_type, intensity in patterns['engagement_intensity'].items()
            }
        
        if patterns['preference_evolution']:
            analysis['evolving_preferences'] = {}
            for genre, evolution in patterns['preference_evolution'].items():
                if len(evolution) >= 3:
                    recent_interactions = evolution[-5:]
                    early_interactions = evolution[:5]
                    
                    recent_avg = sum(1 for i in recent_interactions if i['interaction_type'] in ['like', 'favorite', 'watchlist']) / len(recent_interactions)
                    early_avg = sum(1 for i in early_interactions if i['interaction_type'] in ['like', 'favorite', 'watchlist']) / len(early_interactions)
                    
                    trend = recent_avg - early_avg
                    if abs(trend) > 0.2:
                        analysis['evolving_preferences'][genre] = 'increasing' if trend > 0 else 'decreasing'
        
        if patterns['discovery_patterns']:
            total_discovery = sum(patterns['discovery_patterns'].values())
            analysis['discovery_style'] = max(patterns['discovery_patterns'].items(), key=lambda x: x[1])[0]
        
        analysis['interaction_diversity'] = len(set(i['type'] for i in patterns['interaction_sequence']))
        analysis['content_diversity'] = len(set(c['content_type'] for c in patterns['content_progression']))
        
        return analysis

class RealTimePersonalizationEngine:
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.real_time_weights = defaultdict(lambda: defaultdict(float))
        self.session_interactions = defaultdict(list)
        
    def process_real_time_interaction(self, user_id, content_id, interaction_type, rating=None, metadata=None):
        current_time = datetime.utcnow()
        
        self.session_interactions[user_id].append({
            'content_id': content_id,
            'interaction_type': interaction_type,
            'rating': rating,
            'metadata': metadata or {},
            'timestamp': current_time
        })
        
        recent_interactions = [
            i for i in self.session_interactions[user_id]
            if (current_time - i['timestamp']).total_seconds() < 3600
        ]
        self.session_interactions[user_id] = recent_interactions
        
        self._update_real_time_weights(user_id, content_id, interaction_type, rating, metadata)
        
        return self._get_immediate_recommendations(user_id)
    
    def _update_real_time_weights(self, user_id, content_id, interaction_type, rating, metadata):
        content = self.Content.query.get(content_id)
        if not content:
            return
        
        base_weight = self._calculate_immediate_weight(interaction_type, rating, metadata)
        
        try:
            genres = json.loads(content.genres or '[]')
            for genre in genres:
                self.real_time_weights[user_id][f"genre_{genre}"] += base_weight * 0.3
        except:
            pass
        
        try:
            languages = json.loads(content.languages or '[]')
            for language in languages:
                self.real_time_weights[user_id][f"language_{language}"] += base_weight * 0.2
        except:
            pass
        
        self.real_time_weights[user_id][f"type_{content.content_type}"] += base_weight * 0.25
        
        if content.release_date:
            decade = content.release_date.year // 10 * 10
            self.real_time_weights[user_id][f"decade_{decade}"] += base_weight * 0.1
        
        if content.rating:
            if content.rating >= 8.0:
                self.real_time_weights[user_id]["high_quality"] += base_weight * 0.15
            elif content.rating >= 7.0:
                self.real_time_weights[user_id]["good_quality"] += base_weight * 0.1
    
    def _calculate_immediate_weight(self, interaction_type, rating, metadata):
        weights = {
            'view': 1.0,
            'like': 3.0,
            'favorite': 5.0,
            'watchlist': 4.0,
            'rating': (rating or 3.0) * 1.0,
            'search': 0.8,
            'share': 2.5
        }
        
        base_weight = weights.get(interaction_type, 1.0)
        
        if metadata and interaction_type == 'view':
            completion = metadata.get('completion_percentage', 0)
            if completion > 80:
                base_weight *= 2.0
            elif completion > 50:
                base_weight *= 1.5
        
        return base_weight
    
    def _get_immediate_recommendations(self, user_id, limit=10):
        if user_id not in self.real_time_weights:
            return []
        
        user_weights = self.real_time_weights[user_id]
        
        recent_content_ids = set(
            i['content_id'] for i in self.session_interactions[user_id]
        )
        
        all_interactions = set(
            interaction.content_id for interaction in 
            self.UserInteraction.query.filter_by(user_id=user_id).all()
        )
        
        candidate_content = self.Content.query.limit(1000).all()
        
        recommendations = []
        
        for content in candidate_content:
            if content.id not in all_interactions and content.id not in recent_content_ids:
                score = self._calculate_real_time_score(content, user_weights)
                if score > 0.5:
                    recommendations.append((content.id, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]
    
    def _calculate_real_time_score(self, content, user_weights):
        score = 0.0
        
        try:
            genres = json.loads(content.genres or '[]')
            for genre in genres:
                weight_key = f"genre_{genre}"
                if weight_key in user_weights:
                    score += user_weights[weight_key] * 0.3
        except:
            pass
        
        try:
            languages = json.loads(content.languages or '[]')
            for language in languages:
                weight_key = f"language_{language}"
                if weight_key in user_weights:
                    score += user_weights[weight_key] * 0.2
        except:
            pass
        
        type_key = f"type_{content.content_type}"
        if type_key in user_weights:
            score += user_weights[type_key] * 0.25
        
        if content.release_date:
            decade = content.release_date.year // 10 * 10
            decade_key = f"decade_{decade}"
            if decade_key in user_weights:
                score += user_weights[decade_key] * 0.1
        
        if content.rating and content.rating >= 8.0 and "high_quality" in user_weights:
            score += user_weights["high_quality"] * 0.15
        elif content.rating and content.rating >= 7.0 and "good_quality" in user_weights:
            score += user_weights["good_quality"] * 0.1
        
        return score
    
    def get_session_based_recommendations(self, user_id, limit=15):
        session_data = self.session_interactions.get(user_id, [])
        
        if not session_data:
            return []
        
        session_patterns = self._analyze_session_patterns(session_data)
        
        return self._generate_session_recommendations(user_id, session_patterns, limit)
    
    def _analyze_session_patterns(self, session_data):
        patterns = {
            'dominant_genres': Counter(),
            'content_types': Counter(),
            'engagement_level': 0.0,
            'exploration_vs_focused': 'focused'
        }
        
        total_weight = 0
        unique_content_types = set()
        
        for interaction in session_data:
            weight = self._calculate_immediate_weight(
                interaction['interaction_type'], 
                interaction['rating'], 
                interaction['metadata']
            )
            total_weight += weight
            
            content = self.Content.query.get(interaction['content_id'])
            if content:
                try:
                    genres = json.loads(content.genres or '[]')
                    for genre in genres:
                        patterns['dominant_genres'][genre] += weight
                except:
                    pass
                
                patterns['content_types'][content.content_type] += weight
                unique_content_types.add(content.content_type)
        
        patterns['engagement_level'] = total_weight / max(len(session_data), 1)
        
        if len(unique_content_types) > 2:
            patterns['exploration_vs_focused'] = 'exploration'
        
        return patterns
    
    def _generate_session_recommendations(self, user_id, patterns, limit):
        recommendations = []
        
        top_genres = [genre for genre, _ in patterns['dominant_genres'].most_common(3)]
        preferred_types = [ctype for ctype, _ in patterns['content_types'].most_common(2)]
        
        user_interactions = set(
            interaction.content_id for interaction in 
            self.UserInteraction.query.filter_by(user_id=user_id).all()
        )
        
        session_content = set(
            i['content_id'] for i in self.session_interactions.get(user_id, [])
        )
        
        candidate_content = self.Content.query.limit(2000).all()
        
        for content in candidate_content:
            if content.id not in user_interactions and content.id not in session_content:
                score = 0.0
                
                try:
                    content_genres = set(json.loads(content.genres or '[]'))
                    matching_genres = content_genres & set(top_genres)
                    if matching_genres:
                        score += len(matching_genres) / len(content_genres) * 0.6
                except:
                    pass
                
                if content.content_type in preferred_types:
                    type_preference = patterns['content_types'][content.content_type]
                    total_type_preference = sum(patterns['content_types'].values())
                    score += (type_preference / total_type_preference) * 0.4
                
                if patterns['engagement_level'] > 3.0 and content.rating and content.rating >= 8.0:
                    score *= 1.2
                
                if score > 0.3:
                    recommendations.append((content.id, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]