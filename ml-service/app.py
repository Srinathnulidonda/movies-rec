# ml-service/app.py
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import sqlite3
from datetime import datetime, timedelta
import pickle
import os
from threading import Thread
import time
import requests
from collections import defaultdict
import json

app = Flask(__name__)

# Configuration
DATABASE_PATH = os.getenv('DATABASE_PATH', './movie_rec.db')
MODEL_PATH = './models/'
RETRAIN_INTERVAL = 24 * 60 * 60  # 24 hours

# Ensure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

class AdvancedRecommendationEngine:
    def __init__(self):
        self.user_item_matrix = None
        self.item_features = None
        self.user_clusters = None
        self.content_similarity = None
        self.svd_model = None
        self.kmeans_model = None
        self.tfidf_vectorizer = None
        self.is_trained = False
        
    def load_data(self):
        """Load data from SQLite database"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            
            # Check if tables exist
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if 'user_interaction' not in tables:
                print("user_interaction table not found, creating empty dataframe")
                interactions_df = pd.DataFrame(columns=['user_id', 'content_id', 'interaction_type', 'rating', 'created_at'])
            else:
                # Load user interactions
                interactions_df = pd.read_sql_query("""
                    SELECT user_id, content_id, interaction_type, rating, created_at
                    FROM user_interaction
                """, conn)
            
            if 'content' not in tables:
                print("content table not found, creating empty dataframe")
                content_df = pd.DataFrame(columns=['id', 'title', 'overview', 'genres', 'language', 'rating', 'popularity', 'content_type'])
            else:
                # Load content data
                content_df = pd.read_sql_query("""
                    SELECT id, title, overview, genres, language, rating, popularity, content_type
                    FROM content
                """, conn)
            
            if 'user' not in tables:
                print("user table not found, creating empty dataframe")
                users_df = pd.DataFrame(columns=['id', 'preferences', 'created_at'])
            else:
                # Load user data
                users_df = pd.read_sql_query("""
                    SELECT id, preferences, created_at
                    FROM user
                """, conn)
            
            conn.close()
            
            return interactions_df, content_df, users_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Return empty dataframes on error
            interactions_df = pd.DataFrame(columns=['user_id', 'content_id', 'interaction_type', 'rating', 'created_at'])
            content_df = pd.DataFrame(columns=['id', 'title', 'overview', 'genres', 'language', 'rating', 'popularity', 'content_type'])
            users_df = pd.DataFrame(columns=['id', 'preferences', 'created_at'])
            return interactions_df, content_df, users_df
                
    def preprocess_data(self, interactions_df, content_df, users_df):
        """Preprocess data for ML models"""
        # Create user-item rating matrix
        user_item_ratings = interactions_df.pivot_table(
            index='user_id', 
            columns='content_id', 
            values='rating', 
            fill_value=0
        )
        
        # Create implicit feedback matrix (views, likes, favorites)
        interaction_weights = {'view': 1, 'like': 3, 'favorite': 5, 'wishlist': 2}
        
        interactions_df['weight'] = interactions_df['interaction_type'].map(interaction_weights)
        interactions_df['weight'] = interactions_df['weight'].fillna(1)
        
        user_item_implicit = interactions_df.groupby(['user_id', 'content_id'])['weight'].sum().reset_index()
        user_item_implicit = user_item_implicit.pivot_table(
            index='user_id', 
            columns='content_id', 
            values='weight', 
            fill_value=0
        )
        
        # Combine explicit and implicit feedback
        self.user_item_matrix = user_item_ratings.add(user_item_implicit, fill_value=0)
        
        # Content features
        content_features = []
        for _, content in content_df.iterrows():
            features = f"{content['title']} {content['overview'] or ''}"
            if content['genres']:
                try:
                    genres = json.loads(content['genres']) if isinstance(content['genres'], str) else content['genres']
                    features += " " + " ".join(map(str, genres))
                except:
                    pass
            features += f" {content['language']} {content['content_type']}"
            content_features.append(features)
        
        self.item_features = content_features
        
        return user_item_ratings, user_item_implicit, content_df
    
    def train_matrix_factorization(self):
        """Train SVD model for collaborative filtering"""
        if self.user_item_matrix is None or self.user_item_matrix.shape[0] < 2:
            return
        
        # Apply SVD
        n_components = min(50, min(self.user_item_matrix.shape) - 1)
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Fit the model
        user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        item_factors = self.svd_model.components_.T
        
        # Store factors
        self.user_factors = user_factors
        self.item_factors = item_factors
        
        print(f"SVD Model trained with {n_components} components")
    
    def train_content_similarity(self):
        """Train content-based similarity model"""
        if not self.item_features:
            return
        
        # TF-IDF vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.item_features)
        self.content_similarity = cosine_similarity(tfidf_matrix)
        
        print(f"Content similarity matrix built: {self.content_similarity.shape}")
    
    def train_user_clustering(self):
        """Train user clustering model"""
        if self.user_item_matrix is None or self.user_item_matrix.shape[0] < 5:
            return
        
        # K-means clustering on user preferences
        n_clusters = min(10, max(2, self.user_item_matrix.shape[0] // 10))
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Fit clustering model
        self.user_clusters = self.kmeans_model.fit_predict(self.user_item_matrix)
        
        print(f"User clustering completed with {n_clusters} clusters")
    
    def train_all_models(self):
        """Train all ML models"""
        print("Starting model training...")
        
        # Load and preprocess data
        interactions_df, content_df, users_df = self.load_data()
        
        if interactions_df.empty or content_df.empty:
            print("No data available for training")
            return
        
        self.preprocess_data(interactions_df, content_df, users_df)
        
        # Train models
        self.train_matrix_factorization()
        self.train_content_similarity()
        self.train_user_clustering()
        
        self.is_trained = True
        
        # Save models
        self.save_models()
        
        print("Model training completed successfully")
    
    def save_models(self):
        """Save trained models"""
        try:
            models_data = {
                'user_item_matrix': self.user_item_matrix,
                'content_similarity': self.content_similarity,
                'user_factors': getattr(self, 'user_factors', None),
                'item_factors': getattr(self, 'item_factors', None),
                'user_clusters': self.user_clusters,
                'item_features': self.item_features,
                'is_trained': self.is_trained
            }
            
            with open(f"{MODEL_PATH}recommendation_models.pkl", 'wb') as f:
                pickle.dump(models_data, f)
            
            # Save vectorizer separately
            if self.tfidf_vectorizer:
                with open(f"{MODEL_PATH}tfidf_vectorizer.pkl", 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f)
            
            # Save SVD model
            if self.svd_model:
                with open(f"{MODEL_PATH}svd_model.pkl", 'wb') as f:
                    pickle.dump(self.svd_model, f)
            
            # Save KMeans model
            if self.kmeans_model:
                with open(f"{MODEL_PATH}kmeans_model.pkl", 'wb') as f:
                    pickle.dump(self.kmeans_model, f)
            
            print("Models saved successfully")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self):
        """Load trained models"""
        try:
            with open(f"{MODEL_PATH}recommendation_models.pkl", 'rb') as f:
                models_data = pickle.load(f)
            
            self.user_item_matrix = models_data.get('user_item_matrix')
            self.content_similarity = models_data.get('content_similarity')
            self.user_factors = models_data.get('user_factors')
            self.item_factors = models_data.get('item_factors')
            self.user_clusters = models_data.get('user_clusters')
            self.item_features = models_data.get('item_features')
            self.is_trained = models_data.get('is_trained', False)
            
            # Load vectorizer
            if os.path.exists(f"{MODEL_PATH}tfidf_vectorizer.pkl"):
                with open(f"{MODEL_PATH}tfidf_vectorizer.pkl", 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
            
            # Load SVD model
            if os.path.exists(f"{MODEL_PATH}svd_model.pkl"):
                with open(f"{MODEL_PATH}svd_model.pkl", 'rb') as f:
                    self.svd_model = pickle.load(f)
            
            # Load KMeans model
            if os.path.exists(f"{MODEL_PATH}kmeans_model.pkl"):
                with open(f"{MODEL_PATH}kmeans_model.pkl", 'rb') as f:
                    self.kmeans_model = pickle.load(f)
            
            print("Models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_matrix_factorization_recommendations(self, user_id, top_k=10):
        """Get recommendations using matrix factorization"""
        if not self.is_trained or self.svd_model is None:
            return []
        
        try:
            # Get user index
            if user_id not in self.user_item_matrix.index:
                return []
            
            user_idx = list(self.user_item_matrix.index).index(user_id)
            
            # Get user factors
            user_vector = self.user_factors[user_idx]
            
            # Calculate scores for all items
            scores = np.dot(user_vector, self.item_factors.T)
            
            # Get top recommendations
            item_indices = np.argsort(scores)[::-1]
            
            # Filter out already interacted items
            user_items = set(self.user_item_matrix.columns[self.user_item_matrix.iloc[user_idx] > 0])
            
            recommendations = []
            for item_idx in item_indices:
                if len(recommendations) >= top_k:
                    break
                
                item_id = self.user_item_matrix.columns[item_idx]
                if item_id not in user_items:
                    recommendations.append({
                        'content_id': int(item_id),
                        'score': float(scores[item_idx]),
                        'method': 'matrix_factorization'
                    })
            
            return recommendations
        except Exception as e:
            print(f"Error in matrix factorization recommendations: {e}")
            return []
    
    def get_content_based_recommendations(self, user_id, top_k=10):
        """Get content-based recommendations"""
        if not self.is_trained or self.content_similarity is None:
            return []
        
        try:
            # Get user's interaction history
            conn = sqlite3.connect(DATABASE_PATH)
            user_interactions = pd.read_sql_query("""
                SELECT content_id, interaction_type, rating
                FROM user_interaction
                WHERE user_id = ? AND interaction_type IN ('favorite', 'like')
                ORDER BY created_at DESC
                LIMIT 10
            """, conn, params=(user_id,))
            conn.close()
            
            if user_interactions.empty:
                return []
            
            # Get content indices
            content_indices = []
            for content_id in user_interactions['content_id']:
                try:
                    # Map content_id to similarity matrix index
                    conn = sqlite3.connect(DATABASE_PATH)
                    content_data = pd.read_sql_query("""
                        SELECT id FROM content ORDER BY id
                    """, conn)
                    conn.close()
                    
                    if content_id in content_data['id'].values:
                        idx = content_data[content_data['id'] == content_id].index[0]
                        if idx < len(self.content_similarity):
                            content_indices.append(idx)
                except:
                    continue
            
            if not content_indices:
                return []
            
            # Calculate average similarity
            avg_similarity = np.mean(self.content_similarity[content_indices], axis=0)
            
            # Get top similar content
            similar_indices = np.argsort(avg_similarity)[::-1]
            
            recommendations = []
            user_content_ids = set(user_interactions['content_id'])
            
            conn = sqlite3.connect(DATABASE_PATH)
            content_data = pd.read_sql_query("""
                SELECT id FROM content ORDER BY id
            """, conn)
            conn.close()
            
            for idx in similar_indices:
                if len(recommendations) >= top_k:
                    break
                
                if idx < len(content_data):
                    content_id = content_data.iloc[idx]['id']
                    if content_id not in user_content_ids:
                        recommendations.append({
                            'content_id': int(content_id),
                            'score': float(avg_similarity[idx]),
                            'method': 'content_based'
                        })
            
            return recommendations
        except Exception as e:
            print(f"Error in content-based recommendations: {e}")
            return []
    
    def get_cluster_based_recommendations(self, user_id, top_k=10):
        """Get recommendations based on user clustering"""
        if not self.is_trained or self.user_clusters is None:
            return []
        
        try:
            # Get user cluster
            if user_id not in self.user_item_matrix.index:
                return []
            
            user_idx = list(self.user_item_matrix.index).index(user_id)
            user_cluster = self.user_clusters[user_idx]
            
            # Get other users in the same cluster
            cluster_users = [
                self.user_item_matrix.index[i] for i, cluster in enumerate(self.user_clusters)
                if cluster == user_cluster and i != user_idx
            ]
            
            if not cluster_users:
                return []
            
            # Get popular items in the cluster
            cluster_preferences = self.user_item_matrix.loc[cluster_users].mean(axis=0)
            user_items = set(self.user_item_matrix.columns[self.user_item_matrix.iloc[user_idx] > 0])
            
            # Sort by cluster popularity
            sorted_items = cluster_preferences.sort_values(ascending=False)
            
            recommendations = []
            for item_id, score in sorted_items.items():
                if len(recommendations) >= top_k:
                    break
                
                if item_id not in user_items and score > 0:
                    recommendations.append({
                        'content_id': int(item_id),
                        'score': float(score),
                        'method': 'cluster_based'
                    })
            
            return recommendations
        except Exception as e:
            print(f"Error in cluster-based recommendations: {e}")
            return []
    
    def get_trending_recommendations(self, top_k=10):
        """Get trending recommendations"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            trending_content = pd.read_sql_query("""
                SELECT c.id, c.title, c.popularity, COUNT(ui.id) as interaction_count
                FROM content c
                LEFT JOIN user_interaction ui ON c.id = ui.content_id
                WHERE ui.created_at >= datetime('now', '-7 days') OR ui.created_at IS NULL
                GROUP BY c.id
                ORDER BY c.popularity DESC, interaction_count DESC
                LIMIT ?
            """, conn, params=(top_k,))
            conn.close()
            
            recommendations = []
            for _, row in trending_content.iterrows():
                recommendations.append({
                    'content_id': int(row['id']),
                    'score': float(row['popularity']),
                    'method': 'trending'
                })
            
            return recommendations
        except Exception as e:
            print(f"Error in trending recommendations: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id, top_k=20):
        """Get hybrid recommendations combining multiple methods"""
        # Get recommendations from different methods
        mf_recs = self.get_matrix_factorization_recommendations(user_id, top_k//2)
        content_recs = self.get_content_based_recommendations(user_id, top_k//2)
        cluster_recs = self.get_cluster_based_recommendations(user_id, top_k//3)
        trending_recs = self.get_trending_recommendations(top_k//4)
        
        # Combine and weight recommendations
        all_recs = defaultdict(float)
        method_weights = {
            'matrix_factorization': 0.4,
            'content_based': 0.3,
            'cluster_based': 0.2,
            'trending': 0.1
        }
        
        for recs in [mf_recs, content_recs, cluster_recs, trending_recs]:
            for rec in recs:
                content_id = rec['content_id']
                score = rec['score']
                method = rec['method']
                weight = method_weights.get(method, 0.1)
                all_recs[content_id] += score * weight
        
        # Sort by combined score
        sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
        
        # Format recommendations
        recommendations = []
        for content_id, score in sorted_recs[:top_k]:
            recommendations.append({
                'content_id': content_id,
                'score': score,
                'method': 'hybrid'
            })
        
        return recommendations

# Initialize ML engine
ml_engine = AdvancedRecommendationEngine()

# Background training scheduler
def schedule_retraining():
    """Schedule periodic model retraining"""
    while True:
        time.sleep(RETRAIN_INTERVAL)
        try:
            print("Starting scheduled model retraining...")
            ml_engine.train_all_models()
            print("Scheduled retraining completed")
        except Exception as e:
            print(f"Error in scheduled retraining: {e}")

# API Routes
@app.route('/recommend', methods=['POST'])
def recommend():
    """Get ML-powered recommendations"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        top_k = data.get('top_k', 20)
        method = data.get('method', 'hybrid')
        
        if not ml_engine.is_trained:
            # Return trending recommendations as fallback
            recommendations = ml_engine.get_trending_recommendations(top_k)
        else:
            if method == 'hybrid':
                recommendations = ml_engine.get_hybrid_recommendations(user_id, top_k)
            elif method == 'matrix_factorization':
                recommendations = ml_engine.get_matrix_factorization_recommendations(user_id, top_k)
            elif method == 'content_based':
                recommendations = ml_engine.get_content_based_recommendations(user_id, top_k)
            elif method == 'cluster_based':
                recommendations = ml_engine.get_cluster_based_recommendations(user_id, top_k)
            elif method == 'trending':
                recommendations = ml_engine.get_trending_recommendations(top_k)
            else:
                recommendations = ml_engine.get_hybrid_recommendations(user_id, top_k)
        
        return jsonify({
            'recommendations': recommendations,
            'method': method,
            'count': len(recommendations)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_models():
    """Trigger model training"""
    try:
        # Start training in background
        thread = Thread(target=ml_engine.train_all_models)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'training_started'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-status')
def model_status():
    """Get model training status"""
    return jsonify({
        'is_trained': ml_engine.is_trained,
        'has_user_matrix': ml_engine.user_item_matrix is not None,
        'has_content_similarity': ml_engine.content_similarity is not None,
        'has_user_clusters': ml_engine.user_clusters is not None,
        'matrix_shape': ml_engine.user_item_matrix.shape if ml_engine.user_item_matrix is not None else None
    })

@app.route('/similarity', methods=['POST'])
def get_similarity():
    """Get content similarity scores"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        top_k = data.get('top_k', 10)
        
        if not ml_engine.is_trained or ml_engine.content_similarity is None:
            return jsonify({'error': 'Models not trained'}), 400
        
        # Get content index
        conn = sqlite3.connect(DATABASE_PATH)
        content_data = pd.read_sql_query("""
            SELECT id FROM content ORDER BY id
        """, conn)
        conn.close()
        
        if content_id not in content_data['id'].values:
            return jsonify({'error': 'Content not found'}), 404
        
        content_idx = content_data[content_data['id'] == content_id].index[0]
        
        if content_idx >= len(ml_engine.content_similarity):
            return jsonify({'error': 'Content index out of range'}), 400
        
        # Get similarity scores
        similarity_scores = ml_engine.content_similarity[content_idx]
        similar_indices = np.argsort(similarity_scores)[::-1][1:top_k+1]  # Exclude self
        
        similar_content = []
        for idx in similar_indices:
            if idx < len(content_data):
                similar_content.append({
                    'content_id': int(content_data.iloc[idx]['id']),
                    'similarity_score': float(similarity_scores[idx])
                })
        
        return jsonify({
            'similar_content': similar_content,
            'base_content_id': content_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user-profile', methods=['POST'])
def get_user_profile():
    """Get user preference profile"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        # Get user interactions
        conn = sqlite3.connect(DATABASE_PATH)
        
        interactions = pd.read_sql_query("""
            SELECT ui.content_id, ui.interaction_type, ui.rating, c.genres, c.language, c.content_type
            FROM user_interaction ui
            JOIN content c ON ui.content_id = c.id
            WHERE ui.user_id = ?
        """, conn, params=(user_id,))
        
        conn.close()
        
        if interactions.empty:
            return jsonify({'profile': {}, 'message': 'No interactions found'})
        
        # Analyze preferences
        genre_preferences = defaultdict(int)
        language_preferences = defaultdict(int)
        type_preferences = defaultdict(int)
        
        for _, row in interactions.iterrows():
            weight = 1
            if row['interaction_type'] == 'favorite':
                weight = 5
            elif row['interaction_type'] == 'like':
                weight = 3
            elif row['interaction_type'] == 'wishlist':
                weight = 2
            
            # Genre preferences
            if row['genres']:
                try:
                    genres = json.loads(row['genres']) if isinstance(row['genres'], str) else row['genres']
                    for genre in genres:
                        genre_preferences[str(genre)] += weight
                except:
                    pass
            
            # Language preferences
            if row['language']:
                language_preferences[row['language']] += weight
            
            # Content type preferences
            if row['content_type']:
                type_preferences[row['content_type']] += weight
        
        # Get user cluster if available
        user_cluster = None
        if ml_engine.is_trained and ml_engine.user_clusters is not None:
            if user_id in ml_engine.user_item_matrix.index:
                user_idx = list(ml_engine.user_item_matrix.index).index(user_id)
                user_cluster = int(ml_engine.user_clusters[user_idx])
        
        profile = {
            'user_id': user_id,
            'total_interactions': len(interactions),
            'genre_preferences': dict(genre_preferences),
            'language_preferences': dict(language_preferences),
            'type_preferences': dict(type_preferences),
            'user_cluster': user_cluster,
            'favorite_genres': sorted(genre_preferences.items(), key=lambda x: x[1], reverse=True)[:5],
            'favorite_languages': sorted(language_preferences.items(), key=lambda x: x[1], reverse=True)[:3],
            'favorite_types': sorted(type_preferences.items(), key=lambda x: x[1], reverse=True)
        }
        
        return jsonify({'profile': profile})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': ml_engine.is_trained
    })

if __name__ == '__main__':
    # Initialize ML service
    print("Initializing ML service...")
    
    # Try to load existing models
    if not ml_engine.load_models():
        print("No existing models found, starting initial training...")
        # Start initial training in background
        thread = Thread(target=ml_engine.train_all_models)
        thread.daemon = True
        thread.start()
    
    # Start background retraining scheduler
    retraining_thread = Thread(target=schedule_retraining)
    retraining_thread.daemon = True
    retraining_thread.start()
    
    print("ML service initialized")
    
    # Get port from environment variable for deployment
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)