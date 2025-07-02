#ml-services/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import requests
import pickle
import os
import json
from datetime import datetime
import logging
from scipy.sparse import hstack
from collections import defaultdict
from sklearn.preprocessing import RobustScaler

app = Flask(__name__)
CORS(app)

# Configuration
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', 'your-tmdb-api-key')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRecommendationEngine:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.movie_features = None
        self.cluster_model = None
        self.user_profiles = defaultdict(lambda: None)
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            with open('movie_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.movies_df = model_data['movies_df']
                self.tfidf_matrix = model_data['tfidf_matrix']
                self.svd_model = model_data['svd_model']
                self.movie_features = model_data['movie_features']
                self.cluster_model = model_data.get('cluster_model')
                self.user_profiles = model_data.get('user_profiles', defaultdict(lambda: None))
            logger.info(f"Loaded model with {len(self.movies_df)} movies")
        except FileNotFoundError:
            logger.info("Creating new model...")
            self.create_model()
    
    def create_model(self):
        """Create recommendation model using TMDB data"""
        movies_data = []
        for page in range(1, 51):  # Increased to 50 pages for more diverse data
            try:
                url = f'{TMDB_BASE_URL}/movie/popular'
                params = {'api_key': TMDB_API_KEY, 'page': page}
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for movie in data.get('results', []):
                    movie_details = self.get_movie_details(movie['id'])
                    if movie_details:
                        movies_data.append(movie_details)
                
            except requests.RequestException as e:
                logger.error(f"Error fetching page {page}: {e}")
                continue
        
        self.movies_df = pd.DataFrame(movies_data)
        
        if len(self.movies_df) > 0:
            self.create_features()
            self.cluster_movies()
            self.save_model()
        else:
            self.create_fallback_model()
    
    def get_movie_details(self, movie_id):
        """Get detailed movie information from TMDB"""
        try:
            url = f'{TMDB_BASE_URL}/movie/{movie_id}'
            params = {
                'api_key': TMDB_API_KEY,
                'append_to_response': 'credits,keywords,similar'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            genres = ' '.join([g['name'] for g in data.get('genres', [])])
            cast = ' '.join([c['name'] for c in data.get('credits', {}).get('cast', [])[:7]])
            director = ''
            for crew in data.get('credits', {}).get('crew', []):
                if crew['job'] == 'Director':
                    director = crew['name']
                    break
            keywords = ' '.join([k['name'] for k in data.get('keywords', {}).get('keywords', [])])
            
            return {
                'id': data['id'],
                'title': data.get('title', ''),
                'overview': data.get('overview', ''),
                'genres': genres,
                'cast': cast,
                'director': director,
                'keywords': keywords,
                'vote_average': data.get('vote_average', 0),
                'popularity': data.get('popularity', 0),
                'release_date': data.get('release_date', ''),
                'runtime': data.get('runtime', 0),
                'vote_count': data.get('vote_count', 0)
            }
        except requests.RequestException as e:
            logger.error(f"Error fetching movie {movie_id}: {e}")
            return None
    
    def create_features(self):
        """Create enhanced features for recommendations with better text processing"""
        # Fill NaN values first
        self.movies_df['overview'] = self.movies_df['overview'].fillna('')
        self.movies_df['genres'] = self.movies_df['genres'].fillna('')
        self.movies_df['cast'] = self.movies_df['cast'].fillna('')
        self.movies_df['director'] = self.movies_df['director'].fillna('')
        self.movies_df['keywords'] = self.movies_df['keywords'].fillna('')
        
        # Create weighted combined features
        self.movies_df['combined_features'] = (
            self.movies_df['overview'] + ' ' +
            (self.movies_df['genres'] + ' ') * 3 +  # Weight genres more heavily
            (self.movies_df['cast'] + ' ') * 2 +    # Weight cast moderately
            (self.movies_df['director'] + ' ') * 2 + # Weight director moderately
            self.movies_df['keywords']
        )
        
        # Enhanced TF-IDF with better parameters
        tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['combined_features'])
        
        # Normalize numerical features properly
        numerical_features = self.movies_df[['vote_average', 'popularity', 'vote_count', 'runtime']].fillna(0)
        
        # Use robust scaling for better handling of outliers
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        numerical_features_scaled = scaler.fit_transform(numerical_features)
        
        # Combine features
        self.movie_features = hstack([self.tfidf_matrix, numerical_features_scaled])
        
        # Apply SVD with better parameters
        self.svd_model = TruncatedSVD(n_components=300, random_state=42)
        self.movie_features = self.svd_model.fit_transform(self.movie_features)

    
    def cluster_movies(self):
        """Cluster movies for better recommendation diversity"""
        if self.movie_features is not None:
            self.cluster_model = KMeans(n_clusters=20, random_state=42)
            self.movies_df['cluster'] = self.cluster_model.fit_predict(self.movie_features)
    
    def create_fallback_model(self):
        """Create a fallback model with sample data"""
        sample_movies = [
            {
                'id': 550, 'title': 'Fight Club', 'overview': 'A insomniac office worker...',
                'genres': 'Drama', 'cast': 'Brad Pitt Edward Norton', 'director': 'David Fincher',
                'keywords': 'underground fight club', 'vote_average': 8.4, 'popularity': 50.0,
                'release_date': '1999-10-15', 'runtime': 139, 'vote_count': 10000
            },
            # ... (other sample movies as in original)
        ]
        self.movies_df = pd.DataFrame(sample_movies)
        self.create_features()
        self.cluster_movies()
        self.save_model()
    
    def save_model(self):
        """Save the trained model"""
        model_data = {
            'movies_df': self.movies_df,
            'tfidf_matrix': self.tfidf_matrix,
            'svd_model': self.svd_model,
            'movie_features': self.movie_features,
            'cluster_model': self.cluster_model,
            'user_profiles': self.user_profiles
        }
        with open('movie_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved with {len(self.movies_df)} movies")
    
    def build_user_profile(self, user_id, watch_history, favorites, wishlist):
        """Build enhanced user profile with time decay and genre preferences"""
        if self.movie_features is None:
            return None
            
        user_features = np.zeros(self.movie_features.shape[1])
        genre_preferences = defaultdict(float)
        total_weight = 0
        
        # Time decay factor (more recent interactions matter more)
        from datetime import datetime, timedelta
        current_time = datetime.now()
        
        # Process watch history with rating and time decay
        for item in watch_history:
            if len(item) >= 3:
                movie_id, movie_type, rating = item[:3]
                # If timestamp is available, use it for decay
                time_weight = 1.0  # Default weight if no timestamp
                
                idx = self.movies_df[self.movies_df['id'] == movie_id].index
                if len(idx) > 0:
                    movie_idx = idx[0]
                    # Rating weight (1-5 scale normalized)
                    rating_weight = (rating / 5.0) if rating else 0.7
                    # Combined weight
                    final_weight = 0.4 * rating_weight * time_weight
                    
                    user_features += self.movie_features[movie_idx] * final_weight
                    total_weight += final_weight
                    
                    # Track genre preferences
                    movie_genres = self.movies_df.iloc[movie_idx]['genres']
                    for genre in movie_genres.split():
                        genre_preferences[genre] += rating_weight
        
        # Process favorites (higher weight)
        for item in favorites:
            movie_id, movie_type = item[:2]
            idx = self.movies_df[self.movies_df['id'] == movie_id].index
            if len(idx) > 0:
                movie_idx = idx[0]
                weight = 0.6  # High weight for favorites
                user_features += self.movie_features[movie_idx] * weight
                total_weight += weight
                
                # Strong genre preference for favorites
                movie_genres = self.movies_df.iloc[movie_idx]['genres']
                for genre in movie_genres.split():
                    genre_preferences[genre] += 1.2
        
        # Process wishlist (moderate weight, indicates interest)
        for item in wishlist:
            movie_id, movie_type = item[:2]
            idx = self.movies_df[self.movies_df['id'] == movie_id].index
            if len(idx) > 0:
                movie_idx = idx[0]
                weight = 0.3  # Moderate weight for wishlist
                user_features += self.movie_features[movie_idx] * weight
                total_weight += weight
                
                # Moderate genre preference for wishlist
                movie_genres = self.movies_df.iloc[movie_idx]['genres']
                for genre in movie_genres.split():
                    genre_preferences[genre] += 0.8
        
        if total_weight > 0:
            user_features /= total_weight
            
            # Store user profile with genre preferences
            self.user_profiles[user_id] = {
                'features': user_features,
                'genre_preferences': dict(genre_preferences),
                'last_updated': datetime.now().isoformat()
            }
            
            self.save_model()
            return user_features
        
        return None

    
    def get_collaborative_recommendations(self, user_id, watch_history, n_recommendations=10):
        """Get collaborative filtering recommendations"""
        if not watch_history:
            return self.get_popular_recommendations(n_recommendations)
        
        highly_rated = [item for item in watch_history if len(item) > 2 and item[2] >= 4]
        if not highly_rated:
            return self.get_popular_recommendations(n_recommendations)
        
        movie_ids = [item[0] for item in highly_rated]
        return self.get_content_recommendations(movie_ids, n_recommendations)
    
    def get_popular_recommendations(self, n_recommendations=10):
        """Get popular movie recommendations with diversity"""
        if self.movies_df is None or len(self.movies_df) == 0:
            return []
        
        # Sort by weighted score of popularity and rating
        self.movies_df['weighted_score'] = (
            0.6 * (self.movies_df['popularity'] / self.movies_df['popularity'].max()) +
            0.4 * (self.movies_df['vote_average'] / self.movies_df['vote_average'].max())
        )
        
        cluster_counts = defaultdict(int)
        max_per_cluster = n_recommendations // 5
        recommendations = []
        
        popular_movies = self.movies_df.sort_values('weighted_score', ascending=False)
        for _, movie in popular_movies.iterrows():
            cluster = movie['cluster']
            if cluster_counts[cluster] < max_per_cluster and len(recommendations) < n_recommendations:
                recommendations.append({
                    'id': int(movie['id']),
                    'title': movie['title'],
                    'overview': movie['overview'],
                    'vote_average': movie['vote_average'],
                    'popularity': movie['popularity'],
                    'similarity_score': 0.0,
                    'cluster': int(cluster)
                })
                cluster_counts[cluster] += 1
        
        return recommendations
    
    def get_hybrid_recommendations(self, user_id, watch_history, favorites, wishlist, n_recommendations=10):
        """Get hybrid recommendations with enhanced personalization and diversity"""
        recommendations = []
        watched_movie_ids = set([item[0] for item in watch_history])
        favorite_movie_ids = set([item[0] for item in favorites])
        wishlist_movie_ids = set([item[0] for item in wishlist])
        excluded_ids = watched_movie_ids.union(favorite_movie_ids).union(wishlist_movie_ids)
        
        # Build or get user profile
        user_profile_data = self.user_profiles.get(user_id)
        if user_profile_data is None or watch_history or favorites or wishlist:
            user_features = self.build_user_profile(user_id, watch_history, favorites, wishlist)
            user_profile_data = self.user_profiles.get(user_id)
        else:
            user_features = user_profile_data.get('features')
        
        if user_features is not None and user_profile_data is not None:
            # Get similarities
            similarities = cosine_similarity([user_features], self.movie_features)[0]
            genre_preferences = user_profile_data.get('genre_preferences', {})
            
            # Create scored recommendations with genre bonus
            scored_movies = []
            for idx, similarity in enumerate(similarities):
                movie_data = self.movies_df.iloc[idx]
                if movie_data['id'] not in excluded_ids:
                    # Calculate genre bonus
                    genre_bonus = 0
                    movie_genres = movie_data['genres'].split()
                    for genre in movie_genres:
                        genre_bonus += genre_preferences.get(genre, 0)
                    
                    # Normalize genre bonus
                    genre_bonus = min(genre_bonus / 10.0, 0.3)  # Cap at 0.3
                    
                    # Quality bonus based on ratings and popularity
                    quality_score = (
                        0.3 * (movie_data['vote_average'] / 10.0) +
                        0.2 * min(movie_data['popularity'] / 100.0, 1.0) +
                        0.1 * min(movie_data['vote_count'] / 1000.0, 1.0)
                    )
                    
                    final_score = similarity + genre_bonus + quality_score
                    
                    scored_movies.append({
                        'idx': idx,
                        'score': final_score,
                        'similarity': similarity,
                        'genre_bonus': genre_bonus,
                        'quality_score': quality_score,
                        'cluster': movie_data['cluster']
                    })
            
            # Sort by final score
            scored_movies.sort(key=lambda x: x['score'], reverse=True)
            
            # Ensure diversity across clusters and genres
            cluster_counts = defaultdict(int)
            genre_counts = defaultdict(int)
            max_per_cluster = max(2, n_recommendations // 4)
            max_per_genre = max(3, n_recommendations // 3)
            
            for scored_movie in scored_movies:
                if len(recommendations) >= n_recommendations:
                    break
                    
                idx = scored_movie['idx']
                cluster = scored_movie['cluster']
                movie_data = self.movies_df.iloc[idx]
                
                # Check diversity constraints
                primary_genre = movie_data['genres'].split()[0] if movie_data['genres'] else 'Unknown'
                
                if (cluster_counts[cluster] < max_per_cluster and 
                    genre_counts[primary_genre] < max_per_genre):
                    
                    recommendations.append({
                        'id': int(movie_data['id']),
                        'title': movie_data['title'],
                        'overview': movie_data['overview'],
                        'vote_average': movie_data['vote_average'],
                        'popularity': movie_data['popularity'],
                        'genres': movie_data['genres'],
                        'similarity_score': float(scored_movie['similarity']),
                        'genre_bonus': float(scored_movie['genre_bonus']),
                        'quality_score': float(scored_movie['quality_score']),
                        'final_score': float(scored_movie['score']),
                        'source': 'hybrid_personalized',
                        'cluster': int(cluster)
                    })
                    
                    cluster_counts[cluster] += 1
                    genre_counts[primary_genre] += 1
        
        # Fill remaining slots with diverse popular recommendations
        if len(recommendations) < n_recommendations:
            popular_recs = self.get_popular_recommendations(n_recommendations - len(recommendations))
            for rec in popular_recs:
                if rec['id'] not in [r['id'] for r in recommendations]:
                    rec['source'] = 'popular_fallback'
                    recommendations.append(rec)
        
        return recommendations[:n_recommendations]

# Initialize the recommendation engine
recommendation_engine = AdvancedRecommendationEngine()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'Advanced ML Recommendation Service'})

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        watch_history = data.get('watch_history', [])
        favorites = data.get('favorites', [])
        wishlist = data.get('wishlist', [])  # Added wishlist support
        n_recommendations = data.get('n_recommendations', 10)
        
        recommendations = recommendation_engine.get_hybrid_recommendations(
            user_id, watch_history, favorites, wishlist, n_recommendations
        )
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'total_count': len(recommendations),
            'generated_at': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/content-recommendations', methods=['POST'])
def get_content_recommendations():
    try:
        data = request.get_json()
        movie_ids = data.get('movie_ids', [])
        n_recommendations = data.get('n_recommendations', 10)
        
        recommendations = recommendation_engine.get_content_recommendations(
            movie_ids, n_recommendations
        )
        
        return jsonify({
            'recommendations': recommendations,
            'total_count': len(recommendations),
            'generated_at': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Content recommendation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/popular-recommendations', methods=['GET'])
def get_popular_recommendations():
    try:
        n_recommendations = int(request.args.get('n_recommendations', 10))
        
        recommendations = recommendation_engine.get_popular_recommendations(n_recommendations)
        
        return jsonify({
            'recommendations': recommendations,
            'total_count': len(recommendations),
            'generated_at': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Popular recommendation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    try:
        recommendation_engine.create_model()
        return jsonify({'message': 'Model retrained successfully'})
    except Exception as e:
        logger.error(f"Model retraining error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    try:
        info = {
            'total_movies': len(recommendation_engine.movies_df) if recommendation_engine.movies_df is not None else 0,
            'feature_dimensions': recommendation_engine.movie_features.shape if recommendation_engine.movie_features is not None else None,
            'cluster_count': recommendation_engine.cluster_model.n_clusters if recommendation_engine.cluster_model is not None else 0,
            'model_type': 'Advanced Hybrid (Content-based + Collaborative + Clustering)',
            'last_updated': datetime.now().isoformat()
        }
        return jsonify(info)
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': str(e)}), 500
    
# 5. Add new endpoint for genre-based recommendations
@app.route('/genre-recommendations', methods=['POST'])
def get_genre_recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        preferred_genres = data.get('genres', [])
        n_recommendations = data.get('n_recommendations', 10)
        
        if not preferred_genres:
            return jsonify({'error': 'At least one genre required'}), 400
        
        # Filter movies by genres
        genre_mask = recommendation_engine.movies_df['genres'].str.contains(
            '|'.join(preferred_genres), case=False, na=False
        )
        filtered_movies = recommendation_engine.movies_df[genre_mask]
        
        if len(filtered_movies) == 0:
            return jsonify({'recommendations': [], 'message': 'No movies found for specified genres'})
        
        # Sort by weighted score
        filtered_movies = filtered_movies.copy()
        filtered_movies['weighted_score'] = (
            0.4 * (filtered_movies['vote_average'] / filtered_movies['vote_average'].max()) +
            0.3 * (filtered_movies['popularity'] / filtered_movies['popularity'].max()) +
            0.3 * (filtered_movies['vote_count'] / filtered_movies['vote_count'].max())
        )
        
        top_movies = filtered_movies.nlargest(n_recommendations, 'weighted_score')
        
        recommendations = []
        for _, movie in top_movies.iterrows():
            recommendations.append({
                'id': int(movie['id']),
                'title': movie['title'],
                'overview': movie['overview'],
                'vote_average': movie['vote_average'],
                'popularity': movie['popularity'],
                'genres': movie['genres'],
                'weighted_score': movie['weighted_score'],
                'source': 'genre_based'
            })
        
        return jsonify({
            'recommendations': recommendations,
            'total_count': len(recommendations),
            'genres_requested': preferred_genres,
            'generated_at': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Genre recommendation error: {e}")
        return jsonify({'error': str(e)}), 500

# 6. Add user profile endpoint
@app.route('/user-profile/<user_id>', methods=['GET'])
def get_user_profile(user_id):
    try:
        profile = recommendation_engine.user_profiles.get(user_id)
        if profile:
            return jsonify({
                'user_id': user_id,
                'genre_preferences': profile.get('genre_preferences', {}),
                'last_updated': profile.get('last_updated'),
                'profile_exists': True
            })
        else:
            return jsonify({
                'user_id': user_id,
                'profile_exists': False,
                'message': 'User profile not found. Interact with movies to build profile.'
            })
    except Exception as e:
        logger.error(f"User profile error: {e}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)