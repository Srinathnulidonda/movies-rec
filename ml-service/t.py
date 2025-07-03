# ml-service/app.py
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.cluster import KMeans
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import os
import pickle
import threading
import time
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
DB_PATH = '../backend/recommendations.db'
MODEL_CACHE_PATH = 'models/'
RETRAIN_INTERVAL = 3600  # 1 hour

# Create model cache directory
os.makedirs(MODEL_CACHE_PATH, exist_ok=True)

class AdvancedRecommendationEngine:
    def __init__(self):
        self.models = {}
        self.user_item_matrix = None
        self.content_features = None
        self.user_profiles = {}
        self.item_profiles = {}
        self.last_update = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.content_similarity = None
        self.user_clusters = None
        self.item_clusters = None
        
    def load_data(self):
        """Load and preprocess data from database"""
        try:
            conn = sqlite3.connect(DB_PATH)
            
            # Load user interactions
            interactions_df = pd.read_sql_query('''
                SELECT ui.user_id, ui.content_id, ui.rating, ui.interaction_type,
                       c.title, c.genre_ids, c.content_type, c.overview,
                       c.vote_average, c.popularity, c.release_date
                FROM user_interactions ui
                JOIN content c ON ui.content_id = c.id
                WHERE ui.rating IS NOT NULL
            ''', conn)
            
            # Load content data
            content_df = pd.read_sql_query('''
                SELECT id, title, genre_ids, content_type, overview,
                       vote_average, popularity, release_date, runtime
                FROM content
            ''', conn)
            
            # Load viewing sessions
            sessions_df = pd.read_sql_query('''
                SELECT user_id, content_id, watch_duration, completion_rate,
                       session_start, device_type
                FROM viewing_sessions
            ''', conn)
            
            conn.close()
            
            return interactions_df, content_df, sessions_df
            
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def preprocess_data(self, interactions_df, content_df, sessions_df):
        """Advanced data preprocessing"""
        # Handle missing values
        interactions_df['rating'] = interactions_df['rating'].fillna(interactions_df['rating'].mean())
        content_df['overview'] = content_df['overview'].fillna('')
        content_df['genre_ids'] = content_df['genre_ids'].fillna('[]')
        
        # Create user-item rating matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id', columns='content_id', values='rating', fill_value=0
        )
        
        # Extract content features
        content_features = []
        for _, row in content_df.iterrows():
            genres = json.loads(row['genre_ids']) if row['genre_ids'] != '[]' else []
            feature_text = f"{row['overview']} {' '.join(map(str, genres))} {row['content_type']}"
            content_features.append(feature_text)
        
        # Create content similarity matrix
        if content_features:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_features)
            self.content_similarity = cosine_similarity(tfidf_matrix)
        
        # Create user profiles based on preferences
        self.create_user_profiles(interactions_df, content_df)
        
        # Create item profiles
        self.create_item_profiles(content_df, interactions_df)
        
        # Perform clustering
        self.perform_clustering(interactions_df, content_df)
        
        return interactions_df, content_df
    
    def create_user_profiles(self, interactions_df, content_df):
        """Create detailed user preference profiles"""
        for user_id in interactions_df['user_id'].unique():
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            # Genre preferences
            genre_scores = defaultdict(float)
            content_type_scores = defaultdict(float)
            
            for _, interaction in user_interactions.iterrows():
                content_info = content_df[content_df['id'] == interaction['content_id']]
                if not content_info.empty:
                    genres = json.loads(content_info.iloc[0]['genre_ids'])
                    weight = interaction['rating'] / 10.0
                    
                    for genre in genres:
                        genre_scores[genre] += weight
                    content_type_scores[content_info.iloc[0]['content_type']] += weight
            
            # Temporal preferences
            recent_interactions = user_interactions.sort_values('created_at', ascending=False).head(10)
            recent_weight = 1.5  # Boost recent preferences
            
            self.user_profiles[user_id] = {
                'genre_preferences': dict(genre_scores),
                'content_type_preferences': dict(content_type_scores),
                'avg_rating': user_interactions['rating'].mean(),
                'rating_variance': user_interactions['rating'].var(),
                'interaction_count': len(user_interactions),
                'recent_boost': recent_weight
            }
    
    def create_item_profiles(self, content_df, interactions_df):
        """Create item profiles with aggregated user feedback"""
        for content_id in content_df['id'].unique():
            content_interactions = interactions_df[interactions_df['content_id'] == content_id]
            content_info = content_df[content_df['id'] == content_id].iloc[0]
            
            self.item_profiles[content_id] = {
                'avg_rating': content_interactions['rating'].mean() if not content_interactions.empty else content_info['vote_average'],
                'rating_count': len(content_interactions),
                'genres': json.loads(content_info['genre_ids']),
                'content_type': content_info['content_type'],
                'popularity': content_info['popularity'],
                'release_year': content_info['release_date'][:4] if content_info['release_date'] else None
            }
    
    def perform_clustering(self, interactions_df, content_df):
        """Perform user and item clustering"""
        try:
            # User clustering based on rating patterns
            if len(self.user_item_matrix) > 5:
                user_features = self.user_item_matrix.values
                n_clusters = min(5, len(self.user_item_matrix) // 2)
                self.user_clusters = KMeans(n_clusters=n_clusters, random_state=42).fit(user_features)
            
            # Item clustering based on content features
            if len(content_df) > 10:
                item_features = []
                for _, row in content_df.iterrows():
                    genres = json.loads(row['genre_ids']) if row['genre_ids'] != '[]' else []
                    feature_vector = [
                        row['vote_average'] or 0,
                        row['popularity'] or 0,
                        len(genres),
                        1 if row['content_type'] == 'movie' else 0,
                        1 if row['content_type'] == 'tv' else 0,
                        1 if row['content_type'] == 'anime' else 0
                    ]
                    item_features.append(feature_vector)
                
                if item_features:
                    n_clusters = min(8, len(item_features) // 3)
                    self.item_clusters = KMeans(n_clusters=n_clusters, random_state=42).fit(item_features)
        
        except Exception as e:
            logger.error(f"Clustering error: {e}")
    
    def train_models(self):
        """Train multiple recommendation models"""
        try:
            interactions_df, content_df, sessions_df = self.load_data()
            
            if interactions_df.empty:
                logger.warning("No interaction data available")
                return
            
            interactions_df, content_df = self.preprocess_data(interactions_df, content_df, sessions_df)
            
            # Convert to sparse matrix for implicit models
            sparse_matrix = csr_matrix(self.user_item_matrix.values)
            
            # Train ALS model
            self.models['als'] = AlternatingLeastSquares(factors=64, regularization=0.1, iterations=20)
            self.models['als'].fit(sparse_matrix.T)
            
            # Train BPR model
            self.models['bpr'] = BayesianPersonalizedRanking(factors=64, regularization=0.01, iterations=100)
            self.models['bpr'].fit(sparse_matrix.T)
            
            # Train NMF model
            self.models['nmf'] = NMF(n_components=32, random_state=42)
            self.models['nmf'].fit(self.user_item_matrix.fillna(0))
            
            # Train SVD model
            self.models['svd'] = TruncatedSVD(n_components=32, random_state=42)
            self.models['svd'].fit(self.user_item_matrix.fillna(0))
            
            self.last_update = datetime.now()
            logger.info("Models trained successfully")
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
    
    def get_collaborative_recommendations(self, user_id, limit=20):
        """Advanced collaborative filtering using multiple algorithms"""
        try:
            if user_id not in self.user_item_matrix.index:
                return self.get_popular_recommendations(limit)
            
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            recommendations = []
            
            # ALS recommendations
            if 'als' in self.models:
                als_recs = self.models['als'].recommend(user_idx, self.user_item_matrix.iloc[user_idx].values, N=limit)
                for item_idx, score in als_recs:
                    content_id = self.user_item_matrix.columns[item_idx]
                    recommendations.append({
                        'content_id': content_id,
                        'score': float(score),
                        'algorithm': 'als'
                    })
            
            # BPR recommendations
            if 'bpr' in self.models:
                bpr_recs = self.models['bpr'].recommend(user_idx, self.user_item_matrix.iloc[user_idx].values, N=limit)
                for item_idx, score in bpr_recs:
                    content_id = self.user_item_matrix.columns[item_idx]
                    recommendations.append({
                        'content_id': content_id,
                        'score': float(score),
                        'algorithm': 'bpr'
                    })
            
            # Ensemble scoring
            content_scores = defaultdict(list)
            for rec in recommendations:
                content_scores[rec['content_id']].append(rec['score'])
            
            # Average ensemble scores
            final_recommendations = []
            for content_id, scores in content_scores.items():
                avg_score = np.mean(scores)
                final_recommendations.append({
                    'content_id': content_id,
                    'score': avg_score,
                    'algorithm': 'ensemble_collaborative'
                })
            
            return sorted(final_recommendations, key=lambda x: x['score'], reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Collaborative filtering error: {e}")
            return self.get_popular_recommendations(limit)
    
    def get_content_based_recommendations(self, user_id, limit=20):
        """Advanced content-based filtering"""
        try:
            if user_id not in self.user_profiles:
                return self.get_popular_recommendations(limit)
            
            user_profile = self.user_profiles[user_id]
            recommendations = []
            
            # Score items based on user preferences
            for content_id, item_profile in self.item_profiles.items():
                if content_id in self.user_item_matrix.columns and self.user_item_matrix.loc[user_id, content_id] > 0:
                    continue  # Skip already rated items
                
                score = 0
                
                # Genre matching
                for genre in item_profile['genres']:
                    if genre in user_profile['genre_preferences']:
                        score += user_profile['genre_preferences'][genre] * 0.4
                
                # Content type matching
                content_type = item_profile['content_type']
                if content_type in user_profile['content_type_preferences']:
                    score += user_profile['content_type_preferences'][content_type] * 0.3
                
                # Quality score
                score += (item_profile['avg_rating'] / 10.0) * 0.3
                
                recommendations.append({
                    'content_id': content_id,
                    'score': score,
                    'algorithm': 'content_based'
                })
            
            return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Content-based filtering error: {e}")
            return self.get_popular_recommendations(limit)
    
    def get_hybrid_recommendations(self, user_id, limit=20):
        """Hybrid recommendation combining multiple approaches"""
        try:
            # Get recommendations from different algorithms
            collab_recs = self.get_collaborative_recommendations(user_id, limit)
            content_recs = self.get_content_based_recommendations(user_id, limit)
            popularity_recs = self.get_popular_recommendations(limit)
            
            # Combine with weighted scores
            all_recommendations = {}
            
            # Collaborative filtering (40% weight)
            for rec in collab_recs:
                content_id = rec['content_id']
                if content_id not in all_recommendations:
                    all_recommendations[content_id] = {'score': 0, 'algorithms': []}
                all_recommendations[content_id]['score'] += rec['score'] * 0.4
                all_recommendations[content_id]['algorithms'].append('collaborative')
            
            # Content-based filtering (35% weight)
            for rec in content_recs:
                content_id = rec['content_id']
                if content_id not in all_recommendations:
                    all_recommendations[content_id] = {'score': 0, 'algorithms': []}
                all_recommendations[content_id]['score'] += rec['score'] * 0.35
                all_recommendations[content_id]['algorithms'].append('content_based')
            
            # Popularity boost (25% weight)
            for rec in popularity_recs:
                content_id = rec['content_id']
                if content_id not in all_recommendations:
                    all_recommendations[content_id] = {'score': 0, 'algorithms': []}
                all_recommendations[content_id]['score'] += rec['score'] * 0.25
                all_recommendations[content_id]['algorithms'].append('popularity')
            
            # User cluster boost
            if self.user_clusters and user_id in self.user_item_matrix.index:
                user_idx = self.user_item_matrix.index.get_loc(user_id)
                user_cluster = self.user_clusters.predict([self.user_item_matrix.iloc[user_idx].values])[0]
                
                # Boost items liked by users in the same cluster
                cluster_boost = self.get_cluster_based_boost(user_cluster, user_id)
                for content_id, boost in cluster_boost.items():
                    if content_id in all_recommendations:
                        all_recommendations[content_id]['score'] += boost * 0.1
            
            # Convert to list and sort
            final_recommendations = []
            for content_id, data in all_recommendations.items():
                final_recommendations.append({
                    'content_id': content_id,
                    'score': data['score'],
                    'algorithm': 'hybrid',
                    'algorithms_used': data['algorithms']
                })
            
            return sorted(final_recommendations, key=lambda x: x['score'], reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Hybrid recommendation error: {e}")
            return self.get_popular_recommendations(limit)
    
    def get_cluster_based_boost(self, user_cluster, user_id):
        """Get recommendations based on user cluster preferences"""
        cluster_preferences = defaultdict(float)
        
        try:
            # Find users in the same cluster
            cluster_users = []
            for idx, user_id_check in enumerate(self.user_item_matrix.index):
                if user_id_check != user_id:
                    user_cluster_pred = self.user_clusters.predict([self.user_item_matrix.iloc[idx].values])[0]
                    if user_cluster_pred == user_cluster:
                        cluster_users.append(user_id_check)
            
            # Get preferences of cluster users
            for cluster_user in cluster_users:
                user_ratings = self.user_item_matrix.loc[cluster_user]
                high_rated = user_ratings[user_ratings >= 7]
                for content_id, rating in high_rated.items():
                    cluster_preferences[content_id] += rating / 10.0
            
        except Exception as e:
            logger.error(f"Cluster boost error: {e}")
        
        return cluster_preferences
    
    def get_popular_recommendations(self, limit=20):
        """Fallback popular recommendations"""
        try:
            conn = sqlite3.connect(DB_PATH)
            popular_content = pd.read_sql_query('''
                SELECT id, title, vote_average, popularity,
                       (vote_average * 0.6 + popularity * 0.4) as score
                FROM content
                WHERE vote_average >= 6.0
                ORDER BY score DESC
                LIMIT ?
            ''', conn, params=(limit,))
            conn.close()
            
            recommendations = []
            for _, row in popular_content.iterrows():
                recommendations.append({
                    'content_id': row['id'],
                    'score': row['score'],
                    'algorithm': 'popularity'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Popular recommendations error: {e}")
            return []
    
    def get_similar_content(self, content_id, limit=10):
        """Get similar content using content similarity"""
        try:
            conn = sqlite3.connect(DB_PATH)
            content_df = pd.read_sql_query('SELECT * FROM content', conn)
            conn.close()
            
            if content_id not in content_df['id'].values:
                return []
            
            content_idx = content_df[content_df['id'] == content_id].index[0]
            
            if self.content_similarity is not None:
                similarities = self.content_similarity[content_idx]
                similar_indices = similarities.argsort()[::-1][1:limit+1]  # Exclude self
                
                recommendations = []
                for idx in similar_indices:
                    similar_content_id = content_df.iloc[idx]['id']
                    similarity_score = similarities[idx]
                    
                    recommendations.append({
                        'content_id': similar_content_id,
                        'score': float(similarity_score),
                        'algorithm': 'content_similarity'
                    })
                
                return recommendations
            
            return []
            
        except Exception as e:
            logger.error(f"Similar content error: {e}")
            return []
    
    def update_user_preferences(self, user_id, interaction_data):
        """Real-time user preference updates"""
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    'genre_preferences': defaultdict(float),
                    'content_type_preferences': defaultdict(float),
                    'avg_rating': 5.0,
                    'rating_variance': 1.0,
                    'interaction_count': 0,
                    'recent_boost': 1.0
                }
            
            # Update based on new interaction
            for interaction in interaction_data:
                content_id = interaction['content_id']
                rating = interaction.get('rating', 5)
                
                # Get content info
                conn = sqlite3.connect(DB_PATH)
                content_info = pd.read_sql_query(
                    'SELECT genre_ids, content_type FROM content WHERE id = ?',
                    conn, params=(content_id,)
                )
                conn.close()
                
                if not content_info.empty:
                    genres = json.loads(content_info.iloc[0]['genre_ids'])
                    content_type = content_info.iloc[0]['content_type']
                    
                    # Update preferences with decay
                    decay_factor = 0.9
                    weight = rating / 10.0
                    
                    for genre in genres:
                        self.user_profiles[user_id]['genre_preferences'][genre] = (
                            self.user_profiles[user_id]['genre_preferences'][genre] * decay_factor +
                            weight * (1 - decay_factor)
                        )
                    
                    self.user_profiles[user_id]['content_type_preferences'][content_type] = (
                        self.user_profiles[user_id]['content_type_preferences'][content_type] * decay_factor +
                        weight * (1 - decay_factor)
                    )
            
            logger.info(f"Updated preferences for user {user_id}")
            
        except Exception as e:
            logger.error(f"User preference update error: {e}")

# Global recommendation engine instance
rec_engine = AdvancedRecommendationEngine()

# Background training thread
def background_training():
    """Background thread for periodic model retraining"""
    while True:
        try:
            rec_engine.train_models()
            time.sleep(RETRAIN_INTERVAL)
        except Exception as e:
            logger.error(f"Background training error: {e}")
            time.sleep(300)  # Wait 5 minutes on error

# Start background training
training_thread = threading.Thread(target=background_training, daemon=True)
training_thread.start()

# API Endpoints
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        limit = min(data.get('limit', 20), 50)
        algorithm = data.get('algorithm', 'hybrid_advanced')
        
        if not user_id:
            return jsonify({'error': 'user_id required'}), 400
        
        # Get recommendations based on algorithm
        if algorithm == 'collaborative':
            recommendations = rec_engine.get_collaborative_recommendations(user_id, limit)
        elif algorithm == 'content_based':
            recommendations = rec_engine.get_content_based_recommendations(user_id, limit)
        elif algorithm == 'hybrid_advanced':
            recommendations = rec_engine.get_hybrid_recommendations(user_id, limit)
        else:
            recommendations = rec_engine.get_hybrid_recommendations(user_id, limit)
        
        # Enrich with content details
        enriched_recommendations = []
        if recommendations:
            content_ids = [rec['content_id'] for rec in recommendations]
            conn = sqlite3.connect(DB_PATH)
            content_details = pd.read_sql_query(
                f'SELECT * FROM content WHERE id IN ({",".join(map(str, content_ids))})',
                conn
            )
            conn.close()
            
            for rec in recommendations:
                content_detail = content_details[content_details['id'] == rec['content_id']]
                if not content_detail.empty:
                    enriched_rec = rec.copy()
                    enriched_rec.update(content_detail.iloc[0].to_dict())
                    enriched_recommendations.append(enriched_rec)
        
        return jsonify({
            'recommendations': enriched_recommendations,
            'algorithm': algorithm,
            'total_count': len(enriched_recommendations)
        })
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({'error': 'Recommendation service error'}), 500

@app.route('/similar', methods=['POST'])
def similar_content():
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        limit = min(data.get('limit', 10), 20)
        
        if not content_id:
            return jsonify({'error': 'content_id required'}), 400
        
        similar = rec_engine.get_similar_content(content_id, limit)
        
        # Enrich with content details
        enriched_similar = []
        if similar:
            content_ids = [rec['content_id'] for rec in similar]
            conn = sqlite3.connect(DB_PATH)
            content_details = pd.read_sql_query(
                f'SELECT * FROM content WHERE id IN ({",".join(map(str, content_ids))})',
                conn
            )
            conn.close()
            
            for rec in similar:
                content_detail = content_details[content_details['id'] == rec['content_id']]
                if not content_detail.empty:
                    enriched_rec = rec.copy()
                    enriched_rec.update(content_detail.iloc[0].to_dict())
                    enriched_similar.append(enriched_rec)
        
        return jsonify({
            'similar': enriched_similar,
            'total_count': len(enriched_similar)
        })
        
    except Exception as e:
        logger.error(f"Similar content error: {e}")
        return jsonify({'error': 'Similar content service error'}), 500

@app.route('/update_preferences', methods=['POST'])
def update_preferences():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        interaction_data = data.get('data', [])
        
        if not user_id:
            return jsonify({'error': 'user_id required'}), 400
        
        rec_engine.update_user_preferences(user_id, interaction_data)
        
        return jsonify({'message': 'Preferences updated successfully'})
        
    except Exception as e:
        logger.error(f"Update preferences error: {e}")
        return jsonify({'error': 'Update preferences service error'}), 500

@app.route('/retrain', methods=['POST'])
def retrain_models():
    try:
        rec_engine.train_models()
        return jsonify({'message': 'Models retrained successfully'})
    except Exception as e:
        logger.error(f"Retrain error: {e}")
        return jsonify({'error': 'Retrain service error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'last_update': rec_engine.last_update.isoformat() if rec_engine.last_update else None,
        'models_loaded': len(rec_engine.models),
        'users_profiled': len(rec_engine.user_profiles)
    })

if __name__ == '__main__':
    # Initial training
    rec_engine.train_models()
    app.run(debug=True, host='0.0.0.0', port=5001)