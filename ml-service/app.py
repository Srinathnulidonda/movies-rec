from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import pickle
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])

# Configuration
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
MODEL_PATH = 'recommendation_model.pkl'

class MovieRecommendationModel:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.svd = TruncatedSVD(n_components=100, random_state=42)
        self.movie_features = None
        self.movie_ids = None
        self.movies_df = None
        self.is_trained = False
        
    def fetch_movie_data(self, num_pages=20):
        """Fetch popular movies from TMDB for training"""
        movies = []
        
        try:
            for page in range(1, num_pages + 1):
                response = requests.get(
                    'https://api.themoviedb.org/3/movie/popular',
                    params={
                        'api_key': TMDB_API_KEY,
                        'page': page,
                        'language': 'en-US'
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for movie in data.get('results', []):
                        # Get detailed movie info
                        detail_response = requests.get(
                            f'https://api.themoviedb.org/3/movie/{movie["id"]}',
                            params={'api_key': TMDB_API_KEY}
                        )
                        
                        if detail_response.status_code == 200:
                            detail = detail_response.json()
                            
                            movies.append({
                                'id': movie['id'],
                                'title': movie['title'],
                                'overview': movie.get('overview', ''),
                                'genres': ' '.join([g['name'] for g in detail.get('genres', [])]),
                                'vote_average': movie.get('vote_average', 0),
                                'popularity': movie.get('popularity', 0),
                                'release_year': movie.get('release_date', '')[:4] if movie.get('release_date') else '',
                                'runtime': detail.get('runtime', 0),
                                'budget': detail.get('budget', 0),
                                'revenue': detail.get('revenue', 0)
                            })
                else:
                    logger.error(f"Error fetching movies page {page}: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error fetching movie data: {e}")
            
        return movies
    
    def train_model(self):
        """Train the recommendation model"""
        try:
            logger.info("Fetching movie data for training...")
            movies_data = self.fetch_movie_data()
            
            if not movies_data:
                logger.error("No movie data available for training")
                return False
            
            # Create DataFrame
            self.movies_df = pd.DataFrame(movies_data)
            logger.info(f"Training on {len(self.movies_df)} movies")
            
            # Create feature text combining overview, genres, and other features
            self.movies_df['features'] = (
                self.movies_df['overview'].fillna('') + ' ' +
                self.movies_df['genres'].fillna('') + ' ' +
                self.movies_df['release_year'].astype(str) + ' ' +
                (self.movies_df['vote_average'] >= 7).astype(str)  # High rating indicator
            )
            
            # Create TF-IDF features
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['features'])
            
            # Apply SVD for dimensionality reduction
            self.movie_features = self.svd.fit_transform(tfidf_matrix)
            self.movie_ids = self.movies_df['id'].values
            
            self.is_trained = True
            logger.info("Model training completed successfully")
            
            # Save model
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'svd': self.svd,
                'movie_features': self.movie_features,
                'movie_ids': self.movie_ids,
                'movies_df': self.movies_df,
                'is_trained': self.is_trained
            }
            
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.tfidf_vectorizer = model_data['tfidf_vectorizer']
                self.svd = model_data['svd']
                self.movie_features = model_data['movie_features']
                self.movie_ids = model_data['movie_ids']
                self.movies_df = model_data['movies_df']
                self.is_trained = model_data['is_trained']
                
                logger.info("Model loaded successfully")
                return True
            else:
                logger.info("No saved model found, training new model...")
                return self.train_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return self.train_model()
    
    def get_recommendations(self, user_movie_ids, num_recommendations=12):
        """Get movie recommendations based on user's movie preferences"""
        if not self.is_trained:
            logger.error("Model not trained")
            return []
        
        try:
            # Find user movies in our dataset
            user_indices = []
            for movie_id in user_movie_ids:
                indices = np.where(self.movie_ids == movie_id)[0]
                if len(indices) > 0:
                    user_indices.append(indices[0])
            
            if not user_indices:
                # Return popular movies if no matches found
                return self.get_popular_recommendations(num_recommendations)
            
            # Calculate user profile as average of liked movies
            user_profile = np.mean(self.movie_features[user_indices], axis=0).reshape(1, -1)
            
            # Calculate similarities with all movies
            similarities = cosine_similarity(user_profile, self.movie_features).flatten()
            
            # Get movie indices sorted by similarity (excluding user's movies)
            movie_indices = np.argsort(similarities)[::-1]
            
            # Filter out movies user has already seen
            recommendations = []
            for idx in movie_indices:
                movie_id = int(self.movie_ids[idx])
                if movie_id not in user_movie_ids:
                    recommendations.append(movie_id)
                    if len(recommendations) >= num_recommendations:
                        break
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self.get_popular_recommendations(num_recommendations)
    
    def get_popular_recommendations(self, num_recommendations=12):
        """Get popular movie recommendations as fallback"""
        if not self.is_trained or self.movies_df is None:
            return []
        
        try:
            # Sort by popularity and rating
            popular_movies = self.movies_df.nlargest(num_recommendations, ['popularity', 'vote_average'])
            return popular_movies['id'].tolist()
            
        except Exception as e:
            logger.error(f"Error getting popular recommendations: {e}")
            return []
    
    def get_similar_movies(self, movie_id, num_recommendations=6):
        """Get movies similar to a specific movie"""
        if not self.is_trained:
            return []
        
        try:
            # Find movie index
            movie_indices = np.where(self.movie_ids == movie_id)[0]
            if len(movie_indices) == 0:
                return []
            
            movie_idx = movie_indices[0]
            movie_features = self.movie_features[movie_idx].reshape(1, -1)
            
            # Calculate similarities
            similarities = cosine_similarity(movie_features, self.movie_features).flatten()
            
            # Get most similar movies (excluding the movie itself)
            similar_indices = np.argsort(similarities)[::-1][1:num_recommendations+1]
            
            return [int(self.movie_ids[idx]) for idx in similar_indices]
            
        except Exception as e:
            logger.error(f"Error finding similar movies: {e}")
            return []

# Initialize model
model = MovieRecommendationModel()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_trained': model.is_trained,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint to trigger model training"""
    try:
        success = model.train_model()
        
        if success:
            return jsonify({
                'message': 'Model trained successfully',
                'num_movies': len(model.movies_df) if model.movies_df is not None else 0
            })
        else:
            return jsonify({'error': 'Model training failed'}), 500
            
    except Exception as e:
        logger.error(f"Training endpoint error: {e}")
        return jsonify({'error': 'Training failed'}), 500

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get personalized movie recommendations"""
    try:
        data = request.get_json()
        user_movie_ids = data.get('user_movies', [])
        num_recommendations = data.get('num_recommendations', 12)
        
        if not model.is_trained:
            # Try to load model first
            if not model.load_model():
                return jsonify({'error': 'Model not available'}), 503
        
        recommendations = model.get_recommendations(user_movie_ids, num_recommendations)
        
        return jsonify({
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({'error': 'Failed to generate recommendations'}), 500

@app.route('/similar/<int:movie_id>', methods=['GET'])
def get_similar_movies(movie_id):
    """Get movies similar to a specific movie"""
    try:
        num_recommendations = int(request.args.get('num_recommendations', 6))
        
        if not model.is_trained:
            if not model.load_model():
                return jsonify({'error': 'Model not available'}), 503
        
        similar_movies = model.get_similar_movies(movie_id, num_recommendations)
        
        return jsonify({
            'similar_movies': similar_movies,
            'count': len(similar_movies)
        })
        
    except Exception as e:
        logger.error(f"Similar movies error: {e}")
        return jsonify({'error': 'Failed to find similar movies'}), 500

@app.route('/popular', methods=['GET'])
def get_popular():
    """Get popular movie recommendations"""
    try:
        num_recommendations = int(request.args.get('num_recommendations', 12))
        
        if not model.is_trained:
            if not model.load_model():
                return jsonify({'error': 'Model not available'}), 503
        
        popular_movies = model.get_popular_recommendations(num_recommendations)
        
        return jsonify({
            'popular_movies': popular_movies,
            'count': len(popular_movies)
        })
        
    except Exception as e:
        logger.error(f"Popular movies error: {e}")
        return jsonify({'error': 'Failed to get popular movies'}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the current model"""
    try:
        info = {
            'is_trained': model.is_trained,
            'num_movies': len(model.movies_df) if model.movies_df is not None else 0,
            'model_exists': os.path.exists(MODEL_PATH)
        }
        
        if model.is_trained and model.movies_df is not None:
            info.update({
                'avg_rating': float(model.movies_df['vote_average'].mean()),
                'genres_available': len(set(' '.join(model.movies_df['genres']).split())),
                'year_range': f"{model.movies_df['release_year'].min()}-{model.movies_df['release_year'].max()}"
            })
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': 'Failed to get model info'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load or train model on startup
    logger.info("Starting ML service...")
    model.load_model()
    
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)