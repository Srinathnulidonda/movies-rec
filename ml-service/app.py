from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)
CORS(app)

TMDB_API_KEY = os.getenv('TMDB_API_KEY', 'your_tmdb_api_key')

# Simple recommendation model
class MovieRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.movies_data = []
        self.similarity_matrix = None
        
    def fetch_popular_movies(self):
        """Fetch popular movies to build recommendation base"""
        movies = []
        for page in range(1, 6):  # Get 100 movies (5 pages)
            response = requests.get(f'https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&page={page}')
            if response.status_code == 200:
                data = response.json()
                for movie in data.get('results', []):
                    movies.append({
                        'id': movie.get('id'),
                        'title': movie.get('title'),
                        'overview': movie.get('overview', ''),
                        'genres': movie.get('genre_ids', []),
                        'poster': f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else None
                    })
        return movies
    
    def train(self):
        """Simple content-based filtering using movie overviews"""
        self.movies_data = self.fetch_popular_movies()
        overviews = [movie['overview'] for movie in self.movies_data]
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(overviews)
        
        # Calculate cosine similarity
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return True
    
    def recommend(self, favorites=[], wishlist=[], num_recommendations=10):
        """Generate recommendations based on user preferences"""
        if not self.similarity_matrix.any():
            self.train()
        
        if not favorites and not wishlist:
            # Return popular movies if no preferences
            return self.movies_data[:num_recommendations]
        
        # Find similar movies based on favorites and wishlist
        user_movies = favorites + wishlist
        recommended_indices = set()
        
        for movie_id in user_movies:
            # Find movie index in our data
            movie_idx = None
            for i, movie in enumerate(self.movies_data):
                if movie['id'] == movie_id:
                    movie_idx = i
                    break
            
            if movie_idx is not None:
                # Get similarity scores
                sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                
                # Add top similar movies
                for idx, score in sim_scores[1:6]:  # Skip the movie itself
                    if idx not in recommended_indices and self.movies_data[idx]['id'] not in user_movies:
                        recommended_indices.add(idx)
        
        # Convert indices to movie objects
        recommendations = [self.movies_data[idx] for idx in list(recommended_indices)[:num_recommendations]]
        
        # Fill with popular movies if not enough recommendations
        if len(recommendations) < num_recommendations:
            for movie in self.movies_data:
                if movie['id'] not in user_movies and movie not in recommendations:
                    recommendations.append(movie)
                    if len(recommendations) >= num_recommendations:
                        break
        
        return recommendations[:num_recommendations]

# Initialize recommender
recommender = MovieRecommender()

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        favorites = data.get('favorites', [])
        wishlist = data.get('wishlist', [])
        
        recommendations = recommender.recommend(favorites, wishlist)
        
        return jsonify({
            'recommendations': recommendations,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint to retrain the model"""
    try:
        recommender.train()
        return jsonify({'status': 'model trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Train model on startup
    recommender.train()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))