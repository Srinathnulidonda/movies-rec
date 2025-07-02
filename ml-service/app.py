from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pickle
import sqlite3
import logging
from datetime import datetime
import joblib
from typing import List, Dict, Tuple
import requests
import json

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ml-service-secret-key'

# Configuration
DATABASE_PATH = '../movie_app.db'  # Path to main app database
MODEL_PATH = 'models/'
TMDB_API_KEY = '1cf86635f20bb2aff8e70940e7c3ddd5'

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieRecommendationEngine:
    def __init__(self):
        self.content_vectorizer = None
        self.content_similarity_matrix = None
        self.collaborative_model = None
        self.user_movie_matrix = None
        self.movie_features = None
        self.scaler = StandardScaler()
        self.loaded = False
        
    def load_data_from_db(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data from SQLite database"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            
            # Load movies
            movies_query = """
            SELECT id, tmdb_id, title, overview, genre_ids, vote_average, 
                   vote_count, popularity, content_type, release_date
            FROM movie
            """
            movies_df = pd.read_sql_query(movies_query, conn)
            
            # Load user interactions
            interactions_query = """
            SELECT wh.user_id, wh.movie_id, wh.rating, wh.watched_at,
                   f.user_id as fav_user_id, f.movie_id as fav_movie_id,
                   w.user_id as watch_user_id, w.movie_id as watch_movie_id
            FROM watch_history wh
            LEFT JOIN favorite f ON wh.user_id = f.user_id AND wh.movie_id = f.movie_id
            LEFT JOIN watchlist w ON wh.user_id = w.user_id AND wh.movie_id = w.movie_id
            """
            interactions_df = pd.read_sql_query(interactions_query, conn)
            
            # Load users
            users_query = "SELECT id, username, created_at FROM user"
            users_df = pd.read_sql_query(users_query, conn)
            
            conn.close()
            return movies_df, interactions_df, users_df
            
        except Exception as e:
            logger.error(f"Database loading error: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def preprocess_movies(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess movie data for recommendation algorithms"""
        try:
            # Handle missing values
            movies_df['overview'] = movies_df['overview'].fillna('')
            movies_df['genre_ids'] = movies_df['genre_ids'].fillna('')
            movies_df['vote_average'] = movies_df['vote_average'].fillna(0)
            movies_df['popularity'] = movies_df['popularity'].fillna(0)
            
            # Process genres
            movies_df['genres_list'] = movies_df['genre_ids'].apply(
                lambda x: x.split(',') if x else []
            )
            
            # Create content features
            movies_df['content_features'] = (
                movies_df['title'] + ' ' + 
                movies_df['overview'] + ' ' + 
                movies_df['genre_ids'].str.replace(',', ' ') + ' ' +
                movies_df['content_type']
            )
            
            # Normalize numerical features
            numerical_features = ['vote_average', 'vote_count', 'popularity']
            movies_df[numerical_features] = self.scaler.fit_transform(
                movies_df[numerical_features]
            )
            
            return movies_df
            
        except Exception as e:
            logger.error(f"Movie preprocessing error: {e}")
            return movies_df
    
    def build_content_based_model(self, movies_df: pd.DataFrame):
        """Build content-based recommendation model"""
        try:
            # Create TF-IDF vectorizer for content features
            self.content_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            # Fit and transform content features
            content_matrix = self.content_vectorizer.fit_transform(
                movies_df['content_features']
            )
            
            # Calculate cosine similarity
            self.content_similarity_matrix = cosine_similarity(content_matrix)
            
            logger.info("Content-based model built successfully")
            
        except Exception as e:
            logger.error(f"Content-based model building error: {e}")
    
    def build_collaborative_model(self, interactions_df: pd.DataFrame):
        """Build collaborative filtering model"""
        try:
            if interactions_df.empty:
                logger.warning("No interaction data available for collaborative filtering")
                return
            
            # Create user-movie matrix
            # Assign implicit ratings based on interactions
            interactions_df['implicit_rating'] = 1.0  # Base rating for watching
            interactions_df.loc[interactions_df['fav_user_id'].notna(), 'implicit_rating'] = 2.0  # Higher for favorites
            interactions_df.loc[interactions_df['rating'].notna(), 'implicit_rating'] = interactions_df['rating']
            
            # Create user-movie matrix
            self.user_movie_matrix = interactions_df.pivot_table(
                index='user_id',
                columns='movie_id',
                values='implicit_rating',
                fill_value=0
            )
            
            # Use Matrix Factorization (SVD) for collaborative filtering
            if self.user_movie_matrix.shape[0] > 10 and self.user_movie_matrix.shape[1] > 10:
                n_components = min(50, min(self.user_movie_matrix.shape) - 1)
                self.collaborative_model = TruncatedSVD(
                    n_components=n_components,
                    random_state=42
                )
                self.collaborative_model.fit(self.user_movie_matrix)
                
                logger.info("Collaborative filtering model built successfully")
            else:
                logger.warning("Insufficient data for collaborative filtering")
                
        except Exception as e:
            logger.error(f"Collaborative model building error: {e}")
    
    def build_hybrid_model(self, movies_df: pd.DataFrame):
        """Build hybrid recommendation model combining multiple approaches"""
        try:
            # Extract movie features for hybrid approach
            feature_columns = ['vote_average', 'vote_count', 'popularity']
            self.movie_features = movies_df[['id'] + feature_columns].set_index('id')
            
            # Build KNN model for similar movies
            self.knn_model = NearestNeighbors(
                n_neighbors=20,
                metric='cosine',
                algorithm='brute'
            )
            
            # Combine content and numerical features
            if hasattr(self, 'content_vectorizer') and self.content_vectorizer:
                content_features = self.content_vectorizer.transform(movies_df['content_features'])
                numerical_features = movies_df[feature_columns].values
                
                # Combine features (weighted combination)
                combined_features = np.hstack([
                    content_features.toarray() * 0.7,  # Content weight
                    numerical_features * 0.3  # Numerical weight
                ])
                
                self.knn_model.fit(combined_features)
                self.combined_features = combined_features
                
                logger.info("Hybrid model built successfully")
                
        except Exception as e:
            logger.error(f"Hybrid model building error: {e}")
    
    def train_models(self):
        """Train all recommendation models"""
        try:
            logger.info("Starting model training...")
            
            # Load data
            movies_df, interactions_df, users_df = self.load_data_from_db()
            
            if movies_df.empty:
                logger.error("No movie data available for training")
                return False
            
            # Preprocess data
            movies_df = self.preprocess_movies(movies_df)
            self.movies_df = movies_df  # Store for recommendations
            
            # Build models
            self.build_content_based_model(movies_df)
            self.build_collaborative_model(interactions_df)
            self.build_hybrid_model(movies_df)
            
            # Save models
            self.save_models()
            self.loaded = True
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return False
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            import os
            os.makedirs(MODEL_PATH, exist_ok=True)
            
            # Save models
            if self.content_vectorizer:
                joblib.dump(self.content_vectorizer, f'{MODEL_PATH}/content_vectorizer.pkl')
            
            if self.content_similarity_matrix is not None:
                np.save(f'{MODEL_PATH}/content_similarity_matrix.npy', self.content_similarity_matrix)
            
            if self.collaborative_model:
                joblib.dump(self.collaborative_model, f'{MODEL_PATH}/collaborative_model.pkl')
            
            if self.user_movie_matrix is not None:
                self.user_movie_matrix.to_pickle(f'{MODEL_PATH}/user_movie_matrix.pkl')
            
            if hasattr(self, 'knn_model'):
                joblib.dump(self.knn_model, f'{MODEL_PATH}/knn_model.pkl')
            
            if hasattr(self, 'combined_features'):
                np.save(f'{MODEL_PATH}/combined_features.npy', self.combined_features)
            
            joblib.dump(self.scaler, f'{MODEL_PATH}/scaler.pkl')
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Model saving error: {e}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            import os
            
            if not os.path.exists(MODEL_PATH):
                logger.warning("Model directory doesn't exist, training new models...")
                return self.train_models()
            
            # Load models
            vectorizer_path = f'{MODEL_PATH}/content_vectorizer.pkl'
            if os.path.exists(vectorizer_path):
                self.content_vectorizer = joblib.load(vectorizer_path)
            
            similarity_path = f'{MODEL_PATH}/content_similarity_matrix.npy'
            if os.path.exists(similarity_path):
                self.content_similarity_matrix = np.load(similarity_path)
            
            collaborative_path = f'{MODEL_PATH}/collaborative_model.pkl'
            if os.path.exists(collaborative_path):
                self.collaborative_model = joblib.load(collaborative_path)
            
            user_matrix_path = f'{MODEL_PATH}/user_movie_matrix.pkl'
            if os.path.exists(user_matrix_path):
                self.user_movie_matrix = pd.read_pickle(user_matrix_path)
            
            knn_path = f'{MODEL_PATH}/knn_model.pkl'
            if os.path.exists(knn_path):
                self.knn_model = joblib.load(knn_path)
            
            features_path = f'{MODEL_PATH}/combined_features.npy'
            if os.path.exists(features_path):
                self.combined_features = np.load(features_path)
            
            scaler_path = f'{MODEL_PATH}/scaler.pkl'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Load movie data
            movies_df, _, _ = self.load_data_from_db()
            self.movies_df = self.preprocess_movies(movies_df)
            
            self.loaded = True
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            return self.train_models()
    
    def get_content_based_recommendations(self, movie_ids: List[int], n_recommendations: int = 10) -> List[int]:
        """Get content-based recommendations"""
        try:
            if self.content_similarity_matrix is None or self.movies_df.empty:
                return []
            
            # Get movie indices
            movie_indices = []
            for movie_id in movie_ids:
                idx = self.movies_df[self.movies_df['id'] == movie_id].index
                if len(idx) > 0:
                    movie_indices.append(idx[0])
            
            if not movie_indices:
                return []
            
            # Calculate average similarity scores
            similarity_scores = np.mean([
                self.content_similarity_matrix[idx] for idx in movie_indices
            ], axis=0)
            
            # Get top similar movies
            similar_indices = similarity_scores.argsort()[::-1]
            
            # Filter out input movies and get recommendations
            recommendations = []
            for idx in similar_indices:
                movie_id = self.movies_df.iloc[idx]['id']
                if movie_id not in movie_ids:
                    recommendations.append(movie_id)
                    if len(recommendations) >= n_recommendations:
                        break
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Content-based recommendation error: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Get collaborative filtering recommendations"""
        try:
            if (self.collaborative_model is None or 
                self.user_movie_matrix is None or 
                user_id not in self.user_movie_matrix.index):
                return []
            
            # Get user's rating vector
            user_ratings = self.user_movie_matrix.loc[user_id].values.reshape(1, -1)
            
            # Transform using SVD
            user_factors = self.collaborative_model.transform(user_ratings)
            
            # Reconstruct ratings for all movies
            reconstructed_ratings = self.collaborative_model.inverse_transform(user_factors)
            
            # Get movie IDs and their predicted ratings
            movie_ids = self.user_movie_matrix.columns.tolist()
            predicted_ratings = reconstructed_ratings[0]
            
            # Create recommendations (exclude already rated movies)
            user_rated_movies = self.user_movie_matrix.loc[user_id]
            rated_movie_indices = user_rated_movies[user_rated_movies > 0].index.tolist()
            
            recommendations = []
            for i, (movie_id, rating) in enumerate(zip(movie_ids, predicted_ratings)):
                if movie_id not in rated_movie_indices:
                    recommendations.append((movie_id, rating))
            
            # Sort by predicted rating and return top N
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return [movie_id for movie_id, _ in recommendations[:n_recommendations]]
            
        except Exception as e:
            logger.error(f"Collaborative recommendation error: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id: int, user_preferences: Dict, n_recommendations: int = 10) -> List[int]:
        """Get hybrid recommendations combining multiple approaches"""
        try:
            recommendations = {}
            
            # Content-based recommendations
            watched_movies = user_preferences.get('watched_movies', [])
            favorite_movies = user_preferences.get('favorite_movies', [])
            
            if watched_movies or favorite_movies:
                content_recs = self.get_content_based_recommendations(
                    watched_movies + favorite_movies, 
                    n_recommendations * 2
                )
                for i, movie_id in enumerate(content_recs):
                    score = (len(content_recs) - i) / len(content_recs)
                    recommendations[movie_id] = recommendations.get(movie_id, 0) + score * 0.4
            
            # Collaborative filtering recommendations
            collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
            for i, movie_id in enumerate(collab_recs):
                score = (len(collab_recs) - i) / len(collab_recs)
                recommendations[movie_id] = recommendations.get(movie_id, 0) + score * 0.3
            
            # Popularity-based recommendations (for cold start)
            if not recommendations:
                popular_movies = self.movies_df.nlargest(n_recommendations, 'popularity')['id'].tolist()
                for i, movie_id in enumerate(popular_movies):
                    score = (len(popular_movies) - i) / len(popular_movies)
                    recommendations[movie_id] = score * 0.3
            
            # KNN-based recommendations
            if hasattr(self, 'knn_model') and (watched_movies or favorite_movies):
                try:
                    # Get features for user's preferred movies
                    preference_movies = watched_movies + favorite_movies
                    if preference_movies:
                        movie_indices = []
                        for movie_id in preference_movies[:5]:  # Limit to avoid noise
                            idx = self.movies_df[self.movies_df['id'] == movie_id].index
                            if len(idx) > 0:
                                movie_indices.append(idx[0])
                        
                        if movie_indices:
                            # Average features of preferred movies
                            avg_features = np.mean([
                                self.combined_features[idx] for idx in movie_indices
                            ], axis=0).reshape(1, -1)
                            
                            # Find similar movies
                            distances, indices = self.knn_model.kneighbors(avg_features)
                            
                            for i, idx in enumerate(indices[0]):
                                movie_id = self.movies_df.iloc[idx]['id']
                                if movie_id not in preference_movies:
                                    score = 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity
                                    recommendations[movie_id] = recommendations.get(movie_id, 0) + score * 0.3
                
                except Exception as e:
                    logger.error(f"KNN recommendation error: {e}")
            
            # Sort and return top recommendations
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            return [movie_id for movie_id, _ in sorted_recs[:n_recommendations]]
            
        except Exception as e:
            logger.error(f"Hybrid recommendation error: {e}")
            return []
    
    def get_trending_recommendations(self, content_type: str = None, n_recommendations: int = 10) -> List[int]:
        """Get trending/popular recommendations"""
        try:
            if self.movies_df.empty:
                return []
            
            movies_filtered = self.movies_df
            if content_type:
                movies_filtered = movies_filtered[movies_filtered['content_type'] == content_type]
            
            # Sort by popularity and vote average
            trending_movies = movies_filtered.nlargest(n_recommendations, 'popularity')
            return trending_movies['id'].tolist()
            
        except Exception as e:
            logger.error(f"Trending recommendation error: {e}")
            return []

# Initialize recommendation engine
recommendation_engine = MovieRecommendationEngine()

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': recommendation_engine.loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/train', methods=['POST'])
def train_models():
    """Endpoint to retrain models"""
    try:
        success = recommendation_engine.train_models()
        return jsonify({
            'success': success,
            'message': 'Model training completed' if success else 'Model training failed'
        })
    except Exception as e:
        logger.error(f"Training endpoint error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Main recommendation endpoint"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        preferences = data.get('preferences', {})
        n_recommendations = data.get('n_recommendations', 10)
        recommendation_type = data.get('type', 'hybrid')  # hybrid, content, collaborative, trending
        
        if not recommendation_engine.loaded:
            logger.warning("Models not loaded, attempting to load...")
            if not recommendation_engine.load_models():
                return jsonify({
                    'recommendations': [],
                    'message': 'Models not available'
                }), 503
        
        recommendations = []
        
        if recommendation_type == 'hybrid':
            recommendations = recommendation_engine.get_hybrid_recommendations(
                user_id, preferences, n_recommendations
            )
        elif recommendation_type == 'content':
            watched_movies = preferences.get('watched_movies', [])
            favorite_movies = preferences.get('favorite_movies', [])
            recommendations = recommendation_engine.get_content_based_recommendations(
                watched_movies + favorite_movies, n_recommendations
            )
        elif recommendation_type == 'collaborative':
            recommendations = recommendation_engine.get_collaborative_recommendations(
                user_id, n_recommendations
            )
        elif recommendation_type == 'trending':
            content_type = data.get('content_type')
            recommendations = recommendation_engine.get_trending_recommendations(
                content_type, n_recommendations
            )
        
        # Fallback to trending if no recommendations
        if not recommendations:
            recommendations = recommendation_engine.get_trending_recommendations(
                n_recommendations=n_recommendations
            )
        
        return jsonify({
            'recommendations': recommendations,
            'count': len(recommendations),
            'type': recommendation_type
        })
        
    except Exception as e:
        logger.error(f"Recommendation endpoint error: {e}")
        return jsonify({
            'recommendations': [],
            'message': 'Recommendation generation failed'
        }), 500

@app.route('/similar/<int:movie_id>', methods=['GET'])
def get_similar_movies(movie_id):
    """Get movies similar to a specific movie"""
    try:
        n_recommendations = request.args.get('n', 10, type=int)
        
        if not recommendation_engine.loaded:
            if not recommendation_engine.load_models():
                return jsonify({'similar_movies': []}), 503
        
        similar_movies = recommendation_engine.get_content_based_recommendations(
            [movie_id], n_recommendations
        )
        
        return jsonify({
            'similar_movies': similar_movies,
            'count': len(similar_movies)
        })
        
    except Exception as e:
        logger.error(f"Similar movies error: {e}")
        return jsonify({'similar_movies': []}), 500

@app.route('/stats', methods=['GET'])
def get_model_stats():
    """Get model statistics and information"""
    try:
        stats = {
            'models_loaded': recommendation_engine.loaded,
            'has_content_model': recommendation_engine.content_vectorizer is not None,
            'has_collaborative_model': recommendation_engine.collaborative_model is not None,
            'has_hybrid_model': hasattr(recommendation_engine, 'knn_model'),
        }
        
        if recommendation_engine.loaded and hasattr(recommendation_engine, 'movies_df'):
            stats.update({
                'total_movies': len(recommendation_engine.movies_df),
                'content_types': recommendation_engine.movies_df['content_type'].value_counts().to_dict()
            })
        
        if recommendation_engine.user_movie_matrix is not None:
            stats.update({
                'total_users': len(recommendation_engine.user_movie_matrix.index),
                'total_interactions': recommendation_engine.user_movie_matrix.sum().sum()
            })
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

# Initialize models on startup
@app.before_first_request
def initialize_models():
    try:
        logger.info("Initializing ML models...")
        recommendation_engine.load_models()
    except Exception as e:
        logger.error(f"Model initialization error: {e}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)