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

class CollaborativeFiltering:
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.user_similarity_cache = {}
        self.item_similarity_cache = {}
        
    def get_user_item_matrix(self, interaction_types=None):
        if interaction_types is None:
            interaction_types = ['view', 'like', 'favorite', 'rating', 'watchlist']
        
        interactions = self.UserInteraction.query.filter(
            self.UserInteraction.interaction_type.in_(interaction_types)
        ).all()
        
        user_item_dict = defaultdict(dict)
        for interaction in interactions:
            weight = self._get_interaction_weight(interaction)
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
    
    def _get_interaction_weight(self, interaction):
        weights = {
            'view': 1.0,
            'like': 2.0,
            'favorite': 3.0,
            'watchlist': 2.5,
            'rating': interaction.rating * 0.5 if interaction.rating else 2.0,
            'search': 0.5
        }
        
        base_weight = weights.get(interaction.interaction_type, 1.0)
        
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        time_decay = math.exp(-days_ago / 30.0)
        
        return base_weight * time_decay
    
    def calculate_user_similarity(self, user_id, target_users=None):
        if user_id in self.user_similarity_cache:
            return self.user_similarity_cache[user_id]
        
        matrix, user_to_idx, item_to_idx = self.get_user_item_matrix()
        
        if user_id not in user_to_idx:
            return {}
        
        user_idx = user_to_idx[user_id]
        user_vector = matrix[user_idx].reshape(1, -1)
        
        similarities = cosine_similarity(user_vector, matrix)[0]
        
        result = {}
        for other_user_id, other_idx in user_to_idx.items():
            if other_user_id != user_id:
                result[other_user_id] = similarities[other_idx]
        
        self.user_similarity_cache[user_id] = result
        return result
    
    def calculate_item_similarity(self, item_id, target_items=None):
        if item_id in self.item_similarity_cache:
            return self.item_similarity_cache[item_id]
        
        matrix, user_to_idx, item_to_idx = self.get_user_item_matrix()
        
        if item_id not in item_to_idx:
            return {}
        
        item_idx = item_to_idx[item_id]
        item_vector = matrix[:, item_idx].reshape(-1, 1)
        
        similarities = cosine_similarity(item_vector.T, matrix.T)[0]
        
        result = {}
        for other_item_id, other_idx in item_to_idx.items():
            if other_item_id != item_id:
                result[other_item_id] = similarities[other_idx]
        
        self.item_similarity_cache[item_id] = result
        return result
    
    def user_based_recommendations(self, user_id, limit=20):
        similarities = self.calculate_user_similarity(user_id)
        
        if not similarities:
            return []
        
        similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:50]
        
        user_interactions = set(
            interaction.content_id for interaction in 
            self.UserInteraction.query.filter_by(user_id=user_id).all()
        )
        
        recommendations = defaultdict(float)
        
        for similar_user_id, similarity_score in similar_users:
            if similarity_score < 0.1:
                continue
            
            similar_user_interactions = self.UserInteraction.query.filter_by(
                user_id=similar_user_id
            ).all()
            
            for interaction in similar_user_interactions:
                if interaction.content_id not in user_interactions:
                    weight = self._get_interaction_weight(interaction)
                    recommendations[interaction.content_id] += similarity_score * weight
        
        sorted_recommendations = sorted(
            recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        return [(content_id, score) for content_id, score in sorted_recommendations]
    
    def item_based_recommendations(self, user_id, limit=20):
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not user_interactions:
            return []
        
        recommendations = defaultdict(float)
        
        for interaction in user_interactions:
            similarities = self.calculate_item_similarity(interaction.content_id)
            interaction_weight = self._get_interaction_weight(interaction)
            
            for similar_item_id, similarity_score in similarities.items():
                if similarity_score > 0.1:
                    recommendations[similar_item_id] += similarity_score * interaction_weight
        
        user_seen_items = set(interaction.content_id for interaction in user_interactions)
        
        filtered_recommendations = [
            (item_id, score) for item_id, score in recommendations.items()
            if item_id not in user_seen_items
        ]
        
        sorted_recommendations = sorted(
            filtered_recommendations, 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        return sorted_recommendations

class ContentBasedFiltering:
    def __init__(self, db, models):
        self.db = db
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.content_features_cache = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
    def extract_content_features(self, content_id):
        if content_id in self.content_features_cache:
            return self.content_features_cache[content_id]
        
        content = self.Content.query.get(content_id)
        if not content:
            return None
        
        features = {}
        
        try:
            genres = json.loads(content.genres or '[]')
            features['genres'] = set(genres)
        except:
            features['genres'] = set()
        
        try:
            languages = json.loads(content.languages or '[]')
            features['languages'] = set(languages)
        except:
            features['languages'] = set()
        
        features['content_type'] = content.content_type
        features['rating'] = content.rating or 0
        features['popularity'] = content.popularity or 0
        features['runtime'] = content.runtime or 0
        
        if content.release_date:
            features['release_year'] = content.release_date.year
        else:
            features['release_year'] = 0
        
        features['overview'] = content.overview or ''
        
        self.content_features_cache[content_id] = features
        return features
    
    def calculate_content_similarity(self, content_id1, content_id2):
        features1 = self.extract_content_features(content_id1)
        features2 = self.extract_content_features(content_id2)
        
        if not features1 or not features2:
            return 0.0
        
        similarity_scores = []
        
        genre_similarity = len(features1['genres'] & features2['genres']) / max(
            len(features1['genres'] | features2['genres']), 1
        )
        similarity_scores.append(genre_similarity * 0.3)
        
        language_similarity = len(features1['languages'] & features2['languages']) / max(
            len(features1['languages'] | features2['languages']), 1
        )
        similarity_scores.append(language_similarity * 0.2)
        
        content_type_similarity = 1.0 if features1['content_type'] == features2['content_type'] else 0.0
        similarity_scores.append(content_type_similarity * 0.15)
        
        rating_diff = abs(features1['rating'] - features2['rating'])
        rating_similarity = max(0, 1 - rating_diff / 10.0)
        similarity_scores.append(rating_similarity * 0.15)
        
        year_diff = abs(features1['release_year'] - features2['release_year'])
        year_similarity = max(0, 1 - year_diff / 20.0)
        similarity_scores.append(year_similarity * 0.1)
        
        if features1['overview'] and features2['overview']:
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([
                    features1['overview'], features2['overview']
                ])
                text_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                similarity_scores.append(text_similarity * 0.1)
            except:
                similarity_scores.append(0.0)
        
        return sum(similarity_scores)
    
    def get_content_recommendations(self, user_id, limit=20):
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not user_interactions:
            return []
        
        liked_content = []
        for interaction in user_interactions:
            weight = self._get_interaction_weight(interaction)
            if weight >= 2.0:
                liked_content.append((interaction.content_id, weight))
        
        if not liked_content:
            return []
        
        all_content = self.Content.query.all()
        user_seen_content = set(interaction.content_id for interaction in user_interactions)
        
        recommendations = defaultdict(float)
        
        for content in all_content:
            if content.id in user_seen_content:
                continue
            
            content_score = 0.0
            total_weight = 0.0
            
            for liked_content_id, user_weight in liked_content:
                similarity = self.calculate_content_similarity(content.id, liked_content_id)
                content_score += similarity * user_weight
                total_weight += user_weight
            
            if total_weight > 0:
                recommendations[content.id] = content_score / total_weight
        
        sorted_recommendations = sorted(
            recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        return sorted_recommendations
    
    def _get_interaction_weight(self, interaction):
        weights = {
            'view': 1.0,
            'like': 2.0,
            'favorite': 3.0,
            'watchlist': 2.5,
            'rating': interaction.rating * 0.5 if interaction.rating else 2.0,
            'search': 0.5
        }
        
        base_weight = weights.get(interaction.interaction_type, 1.0)
        
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        time_decay = math.exp(-days_ago / 30.0)
        
        return base_weight * time_decay

class MatrixFactorization:
    def __init__(self, db, models, n_factors=50, learning_rate=0.01, regularization=0.01, epochs=100):
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
        
    def prepare_data(self):
        interactions = self.UserInteraction.query.all()
        
        ratings = []
        for interaction in interactions:
            weight = self._get_interaction_weight(interaction)
            ratings.append({
                'user_id': interaction.user_id,
                'item_id': interaction.content_id,
                'rating': weight
            })
        
        df = pd.DataFrame(ratings)
        
        if df.empty:
            return None, None, None
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(df['user_id'].unique())}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(df['item_id'].unique())}
        
        n_users = len(user_to_idx)
        n_items = len(item_to_idx)
        
        rating_matrix = np.zeros((n_users, n_items))
        
        for _, row in df.iterrows():
            user_idx = user_to_idx[row['user_id']]
            item_idx = item_to_idx[row['item_id']]
            rating_matrix[user_idx, item_idx] = row['rating']
        
        return rating_matrix, user_to_idx, item_to_idx
    
    def train(self):
        rating_matrix, user_to_idx, item_to_idx = self.prepare_data()
        
        if rating_matrix is None:
            return False
        
        n_users, n_items = rating_matrix.shape
        
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_bias = np.mean(rating_matrix[rating_matrix > 0])
        
        for epoch in range(self.epochs):
            for i in range(n_users):
                for j in range(n_items):
                    if rating_matrix[i, j] > 0:
                        prediction = self.predict_rating(i, j)
                        error = rating_matrix[i, j] - prediction
                        
                        user_factor = self.user_factors[i].copy()
                        
                        self.user_factors[i] += self.learning_rate * (
                            error * self.item_factors[j] - self.regularization * self.user_factors[i]
                        )
                        self.item_factors[j] += self.learning_rate * (
                            error * user_factor - self.regularization * self.item_factors[j]
                        )
                        
                        self.user_biases[i] += self.learning_rate * (
                            error - self.regularization * self.user_biases[i]
                        )
                        self.item_biases[j] += self.learning_rate * (
                            error - self.regularization * self.item_biases[j]
                        )
        
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        return True
    
    def predict_rating(self, user_idx, item_idx):
        prediction = self.global_bias + self.user_biases[user_idx] + self.item_biases[item_idx]
        prediction += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return prediction
    
    def get_recommendations(self, user_id, limit=20):
        if self.user_factors is None:
            if not self.train():
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
                recommendations.append((item_id, predicted_rating))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]
    
    def _get_interaction_weight(self, interaction):
        weights = {
            'view': 1.0,
            'like': 2.0,
            'favorite': 3.0,
            'watchlist': 2.5,
            'rating': interaction.rating * 0.5 if interaction.rating else 2.0,
            'search': 0.5
        }
        
        base_weight = weights.get(interaction.interaction_type, 1.0)
        
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        time_decay = math.exp(-days_ago / 30.0)
        
        return base_weight * time_decay

class DeepLearningRecommender:
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        self.embedding_dim = 50
        self.user_embeddings = {}
        self.item_embeddings = {}
        
    def create_user_embedding(self, user_id):
        user = self.User.query.get(user_id)
        if not user:
            return np.zeros(self.embedding_dim)
        
        interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not interactions:
            try:
                preferences = json.loads(user.preferred_genres or '[]')
                languages = json.loads(user.preferred_languages or '[]')
            except:
                preferences = []
                languages = []
            
            embedding = np.random.normal(0, 0.1, self.embedding_dim)
            return embedding
        
        genre_counts = Counter()
        language_counts = Counter()
        type_counts = Counter()
        rating_sum = 0
        rating_count = 0
        
        for interaction in interactions:
            content = self.Content.query.get(interaction.content_id)
            if content:
                try:
                    genres = json.loads(content.genres or '[]')
                    for genre in genres:
                        genre_counts[genre] += self._get_interaction_weight(interaction)
                except:
                    pass
                
                try:
                    languages = json.loads(content.languages or '[]')
                    for language in languages:
                        language_counts[language] += self._get_interaction_weight(interaction)
                except:
                    pass
                
                type_counts[content.content_type] += self._get_interaction_weight(interaction)
                
                if interaction.rating:
                    rating_sum += interaction.rating
                    rating_count += 1
        
        embedding = np.zeros(self.embedding_dim)
        
        top_genres = [genre for genre, _ in genre_counts.most_common(10)]
        for i, genre in enumerate(top_genres):
            if i < 20:
                embedding[i] = genre_counts[genre] / max(sum(genre_counts.values()), 1)
        
        top_languages = [lang for lang, _ in language_counts.most_common(5)]
        for i, lang in enumerate(top_languages):
            if 20 + i < 30:
                embedding[20 + i] = language_counts[lang] / max(sum(language_counts.values()), 1)
        
        if 'movie' in type_counts:
            embedding[30] = type_counts['movie'] / max(sum(type_counts.values()), 1)
        if 'tv' in type_counts:
            embedding[31] = type_counts['tv'] / max(sum(type_counts.values()), 1)
        if 'anime' in type_counts:
            embedding[32] = type_counts['anime'] / max(sum(type_counts.values()), 1)
        
        if rating_count > 0:
            embedding[33] = rating_sum / rating_count / 10.0
        
        embedding[34] = len(interactions) / 100.0
        
        recent_interactions = [
            interaction for interaction in interactions
            if (datetime.utcnow() - interaction.timestamp).days <= 7
        ]
        embedding[35] = len(recent_interactions) / max(len(interactions), 1)
        
        return embedding
    
    def create_item_embedding(self, content_id):
        content = self.Content.query.get(content_id)
        if not content:
            return np.zeros(self.embedding_dim)
        
        embedding = np.zeros(self.embedding_dim)
        
        try:
            genres = json.loads(content.genres or '[]')
            genre_vector = np.zeros(20)
            common_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Thriller', 
                           'Adventure', 'Animation', 'Crime', 'Documentary', 'Fantasy',
                           'Science Fiction', 'Mystery', 'Family', 'War', 'Western',
                           'Music', 'History', 'Biography', 'Sport']
            
            for i, genre in enumerate(common_genres):
                if genre in genres:
                    genre_vector[i] = 1.0
            
            embedding[:20] = genre_vector
        except:
            pass
        
        try:
            languages = json.loads(content.languages or '[]')
            language_vector = np.zeros(10)
            common_languages = ['en', 'hi', 'te', 'ta', 'ml', 'kn', 'ja', 'ko', 'fr', 'es']
            
            for i, lang in enumerate(common_languages):
                if lang in languages:
                    language_vector[i] = 1.0
            
            embedding[20:30] = language_vector
        except:
            pass
        
        type_vector = np.zeros(3)
        if content.content_type == 'movie':
            type_vector[0] = 1.0
        elif content.content_type == 'tv':
            type_vector[1] = 1.0
        elif content.content_type == 'anime':
            type_vector[2] = 1.0
        
        embedding[30:33] = type_vector
        
        embedding[33] = (content.rating or 0) / 10.0
        embedding[34] = min((content.popularity or 0) / 1000.0, 1.0)
        embedding[35] = min((content.vote_count or 0) / 10000.0, 1.0)
        
        if content.release_date:
            years_since_release = (datetime.now().year - content.release_date.year)
            embedding[36] = max(0, 1 - years_since_release / 50.0)
        
        if content.runtime:
            embedding[37] = min(content.runtime / 300.0, 1.0)
        
        return embedding
    
    def calculate_deep_similarity(self, user_id, content_id):
        user_embedding = self.create_user_embedding(user_id)
        item_embedding = self.create_item_embedding(content_id)
        
        similarity = cosine_similarity(
            user_embedding.reshape(1, -1), 
            item_embedding.reshape(1, -1)
        )[0][0]
        
        return max(0, similarity)
    
    def get_deep_recommendations(self, user_id, limit=20):
        user_interactions = set(
            interaction.content_id for interaction in 
            self.UserInteraction.query.filter_by(user_id=user_id).all()
        )
        
        all_content = self.Content.query.limit(5000).all()
        
        recommendations = []
        
        for content in all_content:
            if content.id not in user_interactions:
                similarity = self.calculate_deep_similarity(user_id, content.id)
                recommendations.append((content.id, similarity))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:limit]
    
    def _get_interaction_weight(self, interaction):
        weights = {
            'view': 1.0,
            'like': 2.0,
            'favorite': 3.0,
            'watchlist': 2.5,
            'rating': interaction.rating * 0.5 if interaction.rating else 2.0,
            'search': 0.5
        }
        
        base_weight = weights.get(interaction.interaction_type, 1.0)
        
        days_ago = (datetime.utcnow() - interaction.timestamp).days
        time_decay = math.exp(-days_ago / 30.0)
        
        return base_weight * time_decay

class DiversityOptimizer:
    def __init__(self, db, models):
        self.db = db
        self.Content = models['Content']
        
    def calculate_diversity_score(self, content_list):
        if len(content_list) <= 1:
            return 0.0
        
        total_pairs = len(content_list) * (len(content_list) - 1) / 2
        dissimilar_pairs = 0
        
        for i in range(len(content_list)):
            for j in range(i + 1, len(content_list)):
                content1 = self.Content.query.get(content_list[i])
                content2 = self.Content.query.get(content_list[j])
                
                if content1 and content2:
                    if self._are_dissimilar(content1, content2):
                        dissimilar_pairs += 1
        
        return dissimilar_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def _are_dissimilar(self, content1, content2):
        if content1.content_type != content2.content_type:
            return True
        
        try:
            genres1 = set(json.loads(content1.genres or '[]'))
            genres2 = set(json.loads(content2.genres or '[]'))
            genre_overlap = len(genres1 & genres2) / max(len(genres1 | genres2), 1)
            if genre_overlap < 0.3:
                return True
        except:
            pass
        
        try:
            languages1 = set(json.loads(content1.languages or '[]'))
            languages2 = set(json.loads(content2.languages or '[]'))
            if not languages1 & languages2:
                return True
        except:
            pass
        
        if content1.release_date and content2.release_date:
            year_diff = abs(content1.release_date.year - content2.release_date.year)
            if year_diff > 10:
                return True
        
        return False
    
    def optimize_diversity(self, recommendations, diversity_factor=0.3, limit=20):
        if not recommendations or len(recommendations) <= limit:
            return recommendations
        
        final_recommendations = []
        remaining_recommendations = recommendations[:]
        
        final_recommendations.append(remaining_recommendations.pop(0))
        
        while len(final_recommendations) < limit and remaining_recommendations:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for idx, (content_id, original_score) in enumerate(remaining_recommendations):
                current_diversity = self.calculate_diversity_score(
                    [rec[0] for rec in final_recommendations] + [content_id]
                )
                
                combined_score = (1 - diversity_factor) * original_score + diversity_factor * current_diversity
                
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

class NoveltyDetector:
    def __init__(self, db, models):
        self.db = db
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
    def calculate_novelty_score(self, user_id, content_id):
        content = self.Content.query.get(content_id)
        if not content:
            return 0.0
        
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        
        if not user_interactions:
            return 1.0
        
        novelty_factors = []
        
        try:
            content_genres = set(json.loads(content.genres or '[]'))
            user_genres = set()
            
            for interaction in user_interactions:
                interacted_content = self.Content.query.get(interaction.content_id)
                if interacted_content:
                    try:
                        genres = set(json.loads(interacted_content.genres or '[]'))
                        user_genres.update(genres)
                    except:
                        pass
            
            if user_genres:
                genre_novelty = 1 - len(content_genres & user_genres) / max(len(content_genres), 1)
                novelty_factors.append(genre_novelty * 0.4)
            else:
                novelty_factors.append(0.4)
        except:
            novelty_factors.append(0.0)
        
        user_content_types = set()
        for interaction in user_interactions:
            interacted_content = self.Content.query.get(interaction.content_id)
            if interacted_content:
                user_content_types.add(interacted_content.content_type)
        
        if content.content_type not in user_content_types:
            novelty_factors.append(0.3)
        else:
            novelty_factors.append(0.0)
        
        try:
            content_languages = set(json.loads(content.languages or '[]'))
            user_languages = set()
            
            for interaction in user_interactions:
                interacted_content = self.Content.query.get(interaction.content_id)
                if interacted_content:
                    try:
                        languages = set(json.loads(interacted_content.languages or '[]'))
                        user_languages.update(languages)
                    except:
                        pass
            
            if user_languages:
                language_novelty = 1 - len(content_languages & user_languages) / max(len(content_languages), 1)
                novelty_factors.append(language_novelty * 0.3)
            else:
                novelty_factors.append(0.3)
        except:
            novelty_factors.append(0.0)
        
        return min(sum(novelty_factors), 1.0)

class ColdStartHandler:
    def __init__(self, db, models):
        self.db = db
        self.User = models['User']
        self.Content = models['Content']
        self.UserInteraction = models['UserInteraction']
        
    def get_cold_start_recommendations(self, user_id, limit=20):
        user = self.User.query.get(user_id)
        if not user:
            return self._get_popular_content(limit)
        
        try:
            preferred_genres = json.loads(user.preferred_genres or '[]')
            preferred_languages = json.loads(user.preferred_languages or '[]')
        except:
            preferred_genres = []
            preferred_languages = []
        
        if not preferred_genres and not preferred_languages:
            return self._get_popular_content(limit)
        
        recommendations = []
        
        if preferred_genres:
            for genre in preferred_genres:
                genre_content = self.Content.query.filter(
                    self.Content.genres.contains(genre)
                ).order_by(
                    self.Content.rating.desc(),
                    self.Content.popularity.desc()
                ).limit(5).all()
                
                for content in genre_content:
                    recommendations.append((content.id, content.rating or 0))
        
        if preferred_languages:
            for language in preferred_languages:
                lang_content = self.Content.query.filter(
                    self.Content.languages.contains(language)
                ).order_by(
                    self.Content.rating.desc(),
                    self.Content.popularity.desc()
                ).limit(5).all()
                
                for content in lang_content:
                    recommendations.append((content.id, content.rating or 0))
        
        if not recommendations:
            return self._get_popular_content(limit)
        
        seen_ids = set()
        unique_recommendations = []
        for content_id, score in recommendations:
            if content_id not in seen_ids:
                seen_ids.add(content_id)
                unique_recommendations.append((content_id, score))
        
        unique_recommendations.sort(key=lambda x: x[1], reverse=True)
        return unique_recommendations[:limit]
    
    def _get_popular_content(self, limit):
        popular_content = self.Content.query.order_by(
            self.Content.popularity.desc(),
            self.Content.rating.desc()
        ).limit(limit).all()
        
        return [(content.id, content.popularity or 0) for content in popular_content]

class RealtimeLearning:
    def __init__(self, db, models):
        self.db = db
        self.UserInteraction = models['UserInteraction']
        self.learning_rate = 0.1
        self.feedback_weights = {
            'click': 0.1,
            'view_time': 0.2,
            'like': 0.5,
            'dislike': -0.5,
            'share': 0.3,
            'add_to_watchlist': 0.4,
            'remove_from_watchlist': -0.3
        }
        
    def update_recommendations_with_feedback(self, user_id, content_id, feedback_type, feedback_value=1.0):
        feedback_weight = self.feedback_weights.get(feedback_type, 0.0)
        
        interaction = self.UserInteraction.query.filter_by(
            user_id=user_id,
            content_id=content_id,
            interaction_type='recommendation_feedback'
        ).first()
        
        if interaction:
            current_metadata = interaction.interaction_metadata or {}
            current_score = current_metadata.get('cumulative_score', 0.0)
            new_score = current_score + self.learning_rate * feedback_weight * feedback_value
            
            current_metadata['cumulative_score'] = new_score
            current_metadata['last_updated'] = datetime.utcnow().isoformat()
            current_metadata['feedback_count'] = current_metadata.get('feedback_count', 0) + 1
            
            interaction.interaction_metadata = current_metadata
        else:
            new_interaction = self.UserInteraction(
                user_id=user_id,
                content_id=content_id,
                interaction_type='recommendation_feedback',
                interaction_metadata={
                    'cumulative_score': self.learning_rate * feedback_weight * feedback_value,
                    'last_updated': datetime.utcnow().isoformat(),
                    'feedback_count': 1,
                    'initial_feedback': feedback_type
                }
            )
            self.db.session.add(new_interaction)
        
        self.db.session.commit()
        
        return True
    
    def get_feedback_adjusted_score(self, user_id, content_id, base_score):
        feedback_interaction = self.UserInteraction.query.filter_by(
            user_id=user_id,
            content_id=content_id,
            interaction_type='recommendation_feedback'
        ).first()
        
        if feedback_interaction and feedback_interaction.interaction_metadata:
            feedback_score = feedback_interaction.interaction_metadata.get('cumulative_score', 0.0)
            return base_score + feedback_score
        
        return base_score

class EvaluationMetrics:
    def __init__(self, db, models):
        self.db = db
        self.UserInteraction = models['UserInteraction']
        self.Content = models['Content']
        
    def calculate_precision_at_k(self, user_id, recommendations, k=10):
        if not recommendations or k <= 0:
            return 0.0
        
        top_k_recommendations = [rec[0] for rec in recommendations[:k]]
        
        relevant_interactions = self.UserInteraction.query.filter(
            self.UserInteraction.user_id == user_id,
            self.UserInteraction.content_id.in_(top_k_recommendations),
            self.UserInteraction.interaction_type.in_(['like', 'favorite', 'rating'])
        ).all()
        
        relevant_count = len([
            interaction for interaction in relevant_interactions
            if interaction.interaction_type in ['like', 'favorite'] or 
               (interaction.rating and interaction.rating >= 4.0)
        ])
        
        return relevant_count / k
    
    def calculate_ndcg_at_k(self, user_id, recommendations, k=10):
        if not recommendations or k <= 0:
            return 0.0
        
        top_k_recommendations = [rec[0] for rec in recommendations[:k]]
        
        dcg = 0.0
        for i, content_id in enumerate(top_k_recommendations):
            relevance = self._get_relevance_score(user_id, content_id)
            if relevance > 0:
                dcg += relevance / math.log2(i + 2)
        
        user_interactions = self.UserInteraction.query.filter_by(user_id=user_id).all()
        ideal_relevances = []
        
        for interaction in user_interactions:
            relevance = self._get_relevance_score(user_id, interaction.content_id)
            if relevance > 0:
                ideal_relevances.append(relevance)
        
        ideal_relevances.sort(reverse=True)
        
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances[:k]):
            idcg += relevance / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _get_relevance_score(self, user_id, content_id):
        interactions = self.UserInteraction.query.filter_by(
            user_id=user_id,
            content_id=content_id
        ).all()
        
        max_relevance = 0.0
        
        for interaction in interactions:
            if interaction.interaction_type == 'favorite':
                max_relevance = max(max_relevance, 5.0)
            elif interaction.interaction_type == 'like':
                max_relevance = max(max_relevance, 4.0)
            elif interaction.interaction_type == 'watchlist':
                max_relevance = max(max_relevance, 3.0)
            elif interaction.interaction_type == 'rating' and interaction.rating:
                max_relevance = max(max_relevance, interaction.rating)
            elif interaction.interaction_type == 'view':
                max_relevance = max(max_relevance, 2.0)
        
        return max_relevance
    
    def calculate_coverage(self, all_recommendations, total_items):
        if total_items <= 0:
            return 0.0
        
        unique_items = set()
        for rec_list in all_recommendations:
            for content_id, _ in rec_list:
                unique_items.add(content_id)
        
        return len(unique_items) / total_items
    
    def calculate_diversity_metric(self, recommendations):
        if len(recommendations) <= 1:
            return 0.0
        
        total_pairs = 0
        diverse_pairs = 0
        
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                content1 = self.Content.query.get(recommendations[i][0])
                content2 = self.Content.query.get(recommendations[j][0])
                
                if content1 and content2:
                    total_pairs += 1
                    if self._are_diverse(content1, content2):
                        diverse_pairs += 1
        
        return diverse_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def _are_diverse(self, content1, content2):
        if content1.content_type != content2.content_type:
            return True
        
        try:
            genres1 = set(json.loads(content1.genres or '[]'))
            genres2 = set(json.loads(content2.genres or '[]'))
            
            overlap = len(genres1 & genres2) / max(len(genres1 | genres2), 1)
            if overlap < 0.5:
                return True
        except:
            pass
        
        return False