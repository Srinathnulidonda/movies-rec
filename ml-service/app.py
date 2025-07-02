from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pickle
import os
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configuration
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')

# In-memory storage for demo (use Redis/database in production)
movie_features = {}
user_profiles = {}
similarity_matrix = None
movie_index_map = {}

class MovieRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.svd = TruncatedSVD(n_components=100)
        self.movies_df = None
        self.content_matrix = None
        
    def load_movie_data(self):
        """Load and prepare movie data from TMDB"""
        try:
            # Get popular movies for initial dataset
            movies_data = []
            for page in range(1, 6):  # Get 5 pages of popular movies
                url = f"https://api.themoviedb.org/3/movie/popular"
                params = {'api_key': TMDB_API_KEY, 'page': page}
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    for movie in data.get('results', []):
                        # Get detailed info for each movie
                        detail_url = f"https://api.themoviedb.org/3/movie/{movie['id']}"
                        detail_params = {'api_key': TMDB_API_KEY, 'append_to_response': 'credits,keywords'}
                        detail_response = requests.get(detail_url, params=detail_params)
                        
                        if detail_response.status_code == 200:
                            detailed_movie = detail_response.json()
                            
                            # Extract features
                            genres = ' '.join([g['name'] for g in detailed_movie.get('genres', [])])
                            keywords = ' '.join([k['name'] for k in detailed_movie.get('keywords', {}).get('keywords', [])])
                            cast = ' '.join([c['name'] for c in detailed_movie.get('credits', {}).get('cast', [])[:5]])
                            director = ''
                            crew = detailed_movie.get('credits', {}).get('crew', [])
                            for member in crew:
                                if member.get('job') == 'Director':
                                    director = member.get('name', '')
                                    break
                            
                            movies_data.append({
                                'id': movie['id'],
                                'title': movie['title'],
                                'overview': movie.get('overview', ''),
                                'genres': genres,
                                'keywords': keywords,
                                'cast': cast,
                                'director': director,
                                'rating': movie.get('vote_average', 0),
                                'popularity': movie.get('popularity', 0),
                                'release_date': movie.get('release_date', ''),
                                'poster_path': movie.get('poster_path', ''),
                                'combined_features': f"{movie.get('overview', '')} {genres} {keywords} {cast} {director}"
                            })
            
            self.movies_df = pd.DataFrame(movies_data)
            logging.info(f"Loaded {len(self.movies_df)} movies")
            
            # Create content-based features
            if not self.movies_df.empty:
                self.content_matrix = self.tfidf.fit_transform(self.movies_df['combined_features'].fillna(''))
                logging.info("Content matrix created successfully")
                
        except Exception as e:
            logging.error(f"Error loading movie data: {e}")
            
    def get_content_recommendations(self, movie_ids, n_recommendations=20):
        """Get content-based recommendations"""
        if self.movies_df is None or self.content_matrix is None:
            return []
            
        try:
            # Find movies in our dataset
            user_movies = self.movies_df[self.movies_df['id'].isin(movie_ids)]
            if user_movies.empty:
                return self.get_popular_movies(n_recommendations)
            
            # Calculate average feature vector for user's movies
            user_indices = user_movies.index.tolist()
            user_profile = self.content_matrix[user_indices].mean(axis=0)
            
            # Calculate similarity with all movies
            similarities = cosine_similarity(user_profile, self.content_matrix).flatten()
            
            # Get top recommendations (excluding already watched)
            movie_scores = list(enumerate(similarities))
            movie_scores = [score for score in movie_scores if score[0] not in user_indices]
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for idx, score in movie_scores[:n_recommendations]:
                movie = self.movies_df.iloc[idx]
                recommendations.append({
                    'id': int(movie['id']),
                    'title': movie['title'],
                    'overview': movie['overview'],
                    'poster_path': movie['poster_path'],
                    'vote_average': movie['rating'],
                    'release_date': movie['release_date'],
                    'score': float(score)
                })
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error in content recommendations: {e}")
            return self.get_popular_movies(n_recommendations)
    
    def get_popular_movies(self, n_recommendations=20):
        """Fallback to popular movies"""
        if self.movies_df is None:
            return []
            
        popular = self.movies_df.nlargest(n_recommendations, 'popularity')
        recommendations = []
        
        for _, movie in popular.iterrows():
            recommendations.append({
                'id': int(movie['id']),
                'title': movie['title'],
                'overview': movie['overview'],
                'poster_path': movie['poster_path'],
                'vote_average': movie['rating'],
                'release_date': movie['release_date'],
                'score': movie['popularity'] / 1000  # Normalize score
            })
        
        return recommendations
    
    def get_hybrid_recommendations(self, watch_history, favorites, n_recommendations=20):
        """Combine content-based and popularity-based recommendations"""
        try:
            # Weight favorites higher than watch history
            all_movies = list(set(watch_history + favorites * 2))  # Favorites get double weight
            
            if not all_movies:
                return self.get_popular_movies(n_recommendations)
            
            content_recs = self.get_content_recommendations(all_movies, n_recommendations * 2)
            
            # Boost scores for movies similar to favorites
            if favorites:
                favorite_boost = {}
                favorite_recs = self.get_content_recommendations(favorites, n_recommendations)
                for rec in favorite_recs:
                    favorite_boost[rec['id']] = rec['score'] * 0.3  # 30% boost
                
                for rec in content_recs:
                    if rec['id'] in favorite_boost:
                        rec['score'] += favorite_boost[rec['id']]
            
            # Re-sort by score and return top N
            content_recs.sort(key=lambda x: x['score'], reverse=True)
            return content_recs[:n_recommendations]
            
        except Exception as e:
            logging.error(f"Error in hybrid recommendations: {e}")
            return self.get_popular_movies(n_recommendations)

# Initialize recommender
recommender = MovieRecommender()

@app.before_first_request
def initialize():
    """Initialize the ML service"""
    logging.info("Initializing ML service...")
    recommender.load_movie_data()
    logging.info("ML service initialized successfully")

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        watch_history = data.get('watch_history', [])
        favorites = data.get('favorites', [])
        
        # Get recommendations
        recommendations = recommender.get_hybrid_recommendations(watch_history, favorites)
        
        # Store user profile for future improvements
        user_profiles[user_id] = {
            'watch_history': watch_history,
            'favorites': favorites,
            'last_updated': datetime.now().isoformat()
        }
        
        return jsonify({
            'results': recommendations,
            'total_results': len(recommendations),
            'user_id': user_id
        })
        
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}")
        return jsonify({'error': 'Failed to generate recommendations'}), 500

@app.route('/api/similar/<int:movie_id>', methods=['GET'])
def get_similar_movies(movie_id):
    """Get movies similar to a specific movie"""
    try:
        similar_movies = recommender.get_content_recommendations([movie_id], 10)
        return jsonify({
            'results': similar_movies,
            'total_results': len(similar_movies)
        })
    except Exception as e:
        logging.error(f"Error getting similar movies: {e}")
        return jsonify({'error': 'Failed to get similar movies'}), 500

@app.route('/api/popular', methods=['GET'])
def get_popular():
    """Get popular movies"""
    try:
        popular_movies = recommender.get_popular_movies(20)
        return jsonify({
            'results': popular_movies,
            'total_results': len(popular_movies)
        })
    except Exception as e:
        logging.error(f"Error getting popular movies: {e}")
        return jsonify({'error': 'Failed to get popular movies'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'movies_loaded': len(recommender.movies_df) if recommender.movies_df is not None else 0,
        'users_profiled': len(user_profiles)
    })

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host='0.0.0.0', port=5001)