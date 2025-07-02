from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import sqlite3
import pickle
import os
from datetime import datetime, timedelta
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import logging
from collections import defaultdict, Counter
import threading
import time

app = Flask(__name__)
CORS(app)

# Configuration
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '1cf86635f20bb2aff8e70940e7c3ddd5')
TMDB_BASE_URL = 'https://api.themoviedb.org/3'

# Global variables for models and data
content_similarity_matrix = None
tfidf_vectorizer = None
movie_features_df = None
user_item_matrix = None
svd_model = None
scaler = None
movie_metadata = {}
genre_profiles = {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieRecommendationEngine:
    def __init__(self):
        self.content_similarity_matrix = None
        self.tfidf_vectorizer = None
        self.movie_features_df = None
        self.user_item_matrix = None
        self.svd_model = None
        self.scaler = None
        self.movie_metadata = {}
        self.genre_profiles = {}
        self.popularity_scores = {}
        self.last_update = None
        
    def initialize_models(self):
        """Initialize and train recommendation models"""
        try:
            logger.info("Initializing ML models...")
            
            # Load movie data from TMDB
            self.load_movie_data()
            
            # Build content-based model
            self.build_content_model()
            
            # Build collaborative filtering model
            self.build_collaborative_model()
            
            # Calculate popularity scores
            self.calculate_popularity_scores()
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def load_movie_data(self):
        """Load movie data from TMDB API"""
        logger.info("Loading movie data from TMDB...")
        
        movies_data = []
        tv_data = []
        
        # Fetch popular movies and TV shows
        for page in range(1, 11):  # Get first 10 pages
            try:
                # Movies
                movie_url = f'{TMDB_BASE_URL}/movie/popular'
                movie_params = {'api_key': TMDB_API_KEY, 'page': page}
                movie_response = requests.get(movie_url, params=movie_params)
                
                if movie_response.status_code == 200:
                    movies_data.extend(movie_response.json().get('results', []))
                
                # TV Shows
                tv_url = f'{TMDB_BASE_URL}/tv/popular'
                tv_params = {'api_key': TMDB_API_KEY, 'page': page}
                tv_response = requests.get(tv_url, params=tv_params)
                
                if tv_response.status_code == 200:
                    tv_data.extend(tv_response.json().get('results', []))
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                continue
        
        # Process and store movie metadata
        all_content = []
        
        for movie in movies_data:
            movie['media_type'] = 'movie'
            movie['content_id'] = f"movie_{movie['id']}"
            all_content.append(movie)
            self.movie_metadata[movie['content_id']] = movie
        
        for show in tv_data:
            show['media_type'] = 'tv'
            show['content_id'] = f"tv_{show['id']}"
            # Normalize field names
            show['title'] = show.get('name', show.get('title', ''))
            show['release_date'] = show.get('first_air_date', show.get('release_date', ''))
            all_content.append(show)
            self.movie_metadata[show['content_id']] = show
        
        logger.info(f"Loaded {len(all_content)} items from TMDB")
        return all_content
    
    def build_content_model(self):
        """Build content-based recommendation model"""
        logger.info("Building content-based model...")
        
        try:
            # Prepare features dataframe
            features_data = []
            
            for content_id, metadata in self.movie_metadata.items():
                # Get genre names
                genre_names = [str(g) for g in metadata.get('genre_ids', [])]
                genres_str = ' '.join(genre_names)
                
                # Create content string for TF-IDF
                overview = metadata.get('overview', '')
                title = metadata.get('title', '')
                
                content_text = f"{title} {overview} {genres_str}"
                
                features_data.append({
                    'content_id': content_id,
                    'content_text': content_text,
                    'popularity': metadata.get('popularity', 0),
                    'vote_average': metadata.get('vote_average', 0),
                    'vote_count': metadata.get('vote_count', 0),
                    'genres': genre_names,
                    'media_type': metadata.get('media_type', 'movie')
                })
            
            self.movie_features_df = pd.DataFrame(features_data)
            
            # Build TF-IDF matrix
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.movie_features_df['content_text'].fillna('')
            )
            
            # Calculate similarity matrix
            self.content_similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
            
            logger.info("Content-based model built successfully")
            
        except Exception as e:
            logger.error(f"Error building content model: {e}")
    
    def build_collaborative_model(self):
        """Build collaborative filtering model using matrix factorization"""
        logger.info("Building collaborative filtering model...")
        
        try:
            # Create synthetic user-item interaction data
            # In a real scenario, this would come from actual user data
            user_interactions = self.generate_synthetic_interactions()
            
            # Create user-item matrix
            self.user_item_matrix = pd.pivot_table(
                user_interactions,
                index='user_id',
                columns='content_id',
                values='rating',
                fill_value=0
            )
            
            # Apply SVD for matrix factorization
            self.svd_model = TruncatedSVD(n_components=50, random_state=42)
            user_factors = self.svd_model.fit_transform(self.user_item_matrix)
            
            # Store user factors for later use
            self.user_factors = user_factors
            self.item_factors = self.svd_model.components_.T
            
            logger.info("Collaborative filtering model built successfully")
            
        except Exception as e:
            logger.error(f"Error building collaborative model: {e}")
    
    def generate_synthetic_interactions(self):
        """Generate synthetic user interactions for demonstration"""
        interactions = []
        
        # Generate interactions based on content popularity and genres
        content_ids = list(self.movie_metadata.keys())
        
        for user_id in range(1, 1001):  # 1000 synthetic users
            # Each user interacts with 10-50 items
            n_interactions = np.random.randint(10, 51)
            
            # Select items based on popularity (popular items more likely)
            popularities = [self.movie_metadata[cid].get('popularity', 0) for cid in content_ids]
            popularities = np.array(popularities)
            popularities = popularities / popularities.sum()  # normalize
            
            selected_items = np.random.choice(
                content_ids, 
                size=n_interactions, 
                replace=False, 
                p=popularities
            )
            
            for content_id in selected_items:
                # Generate rating based on item quality and random user preference
                base_rating = self.movie_metadata[content_id].get('vote_average', 5) / 2  # Scale to 1-5
                user_bias = np.random.normal(0, 1)  # User-specific bias
                rating = max(1, min(5, base_rating + user_bias))
                
                interactions.append({
                    'user_id': user_id,
                    'content_id': content_id,
                    'rating': rating
                })
        
        return pd.DataFrame(interactions)
    
    def calculate_popularity_scores(self):
        """Calculate popularity scores for trending recommendations"""
        current_time = datetime.now()
        
        for content_id, metadata in self.movie_metadata.items():
            # Base popularity from TMDB
            base_popularity = metadata.get('popularity', 0)
            
            # Boost recent releases
            release_date = metadata.get('release_date', '')
            release_boost = 1.0
            
            if release_date:
                try:
                    release_dt = datetime.strptime(release_date, '%Y-%m-%d')
                    days_since_release = (current_time - release_dt).days
                    
                    if days_since_release < 365:  # Recent releases get boost
                        release_boost = 1.5 - (days_since_release / 365) * 0.5
                except:
                    pass
            
            # Quality boost
            vote_average = metadata.get('vote_average', 0)
            vote_count = metadata.get('vote_count', 0)
            
            quality_score = (vote_average / 10) * min(1, vote_count / 100)
            
            final_score = base_popularity * release_boost * (1 + quality_score)
            self.popularity_scores[content_id] = final_score
    
    def get_content_based_recommendations(self, user_preferences, n_recommendations=20):
        """Get content-based recommendations"""
        try:
            if self.content_similarity_matrix is None:
                return []
            
            # Get user's preferred content based on watch history and favorites
            preferred_content_ids = []
            
            # Process watch history
            for item in user_preferences.get('watch_history', []):
                movie_id, movie_type = item[0], item[1]
                rating = item[2] if len(item) > 2 else None
                
                if rating and rating >= 4:  # Only consider highly rated items
                    content_id = f"{movie_type}_{movie_id}"
                    if content_id in self.movie_metadata:
                        preferred_content_ids.append(content_id)
            
            # Process favorites
            for item in user_preferences.get('favorites', []):
                movie_id, movie_type = item[0], item[1]
                content_id = f"{movie_type}_{movie_id}"
                if content_id in self.movie_metadata:
                    preferred_content_ids.append(content_id)
            
            if not preferred_content_ids:
                # Return popular content if no preferences
                return self.get_popular_recommendations(n_recommendations)
            
            # Get similarity scores
            content_indices = {cid: idx for idx, cid in enumerate(self.movie_features_df['content_id'])}
            
            similarity_scores = []
            for content_id in preferred_content_ids:
                if content_id in content_indices:
                    idx = content_indices[content_id]
                    scores = list(enumerate(self.content_similarity_matrix[idx]))
                    similarity_scores.extend(scores)
            
            # Aggregate and sort scores
            score_dict = defaultdict(float)
            for idx, score in similarity_scores:
                score_dict[idx] += score
            
            # Sort by similarity
            sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Get recommendations
            recommendations = []
            seen_content_ids = set(preferred_content_ids)
            
            for idx, score in sorted_scores:
                if len(recommendations) >= n_recommendations:
                    break
                
                content_id = self.movie_features_df.iloc[idx]['content_id']
                
                if content_id not in seen_content_ids:
                    movie_data = self.movie_metadata.get(content_id)
                    if movie_data:
                        movie_data['recommendation_score'] = float(score)
                        movie_data['recommendation_reason'] = 'Content similarity'
                        recommendations.append(movie_data)
                        seen_content_ids.add(content_id)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=20):
        """Get collaborative filtering recommendations"""
        try:
            if self.svd_model is None or self.user_item_matrix is None:
                return []
            
            # For new users, return popular items
            if user_id not in self.user_item_matrix.index:
                return self.get_popular_recommendations(n_recommendations)
            
            # Get user factors
            user_idx = list(self.user_item_matrix.index).index(user_id)
            user_vector = self.user_factors[user_idx]
            
            # Calculate predicted ratings for all items
            predicted_ratings = np.dot(user_vector, self.item_factors.T)
            
            # Get items user hasn't interacted with
            user_ratings = self.user_item_matrix.loc[user_id]
            unrated_items = user_ratings[user_ratings == 0].index
            
            # Get predictions for unrated items
            item_indices = {item: idx for idx, item in enumerate(self.user_item_matrix.columns)}
            predictions = []
            
            for item in unrated_items:
                if item in item_indices and item in self.movie_metadata:
                    item_idx = item_indices[item]
                    predicted_rating = predicted_ratings[item_idx]
                    predictions.append((item, predicted_rating))
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Convert to movie data
            recommendations = []
            for content_id, rating in predictions[:n_recommendations]:
                movie_data = self.movie_metadata.get(content_id)
                if movie_data:
                    movie_data['recommendation_score'] = float(rating)
                    movie_data['recommendation_reason'] = 'Collaborative filtering'
                    recommendations.append(movie_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id, user_preferences, n_recommendations=20):
        """Get hybrid recommendations combining content and collaborative filtering"""
        try:
            # Get recommendations from both methods
            content_recs = self.get_content_based_recommendations(user_preferences, n_recommendations)
            collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations)
            
            # Combine and weight the recommendations
            combined_scores = defaultdict(float)
            all_recommendations = {}
            
            # Weight content-based recommendations (60%)
            for rec in content_recs:
                content_id = rec['content_id']
                combined_scores[content_id] += rec.get('recommendation_score', 0) * 0.6
                all_recommendations[content_id] = rec
            
            # Weight collaborative recommendations (40%)
            for rec in collab_recs:
                content_id = rec['content_id']
                combined_scores[content_id] += rec.get('recommendation_score', 0) * 0.4
                if content_id not in all_recommendations:
                    all_recommendations[content_id] = rec
            
            # Add popularity boost
            for content_id in combined_scores:
                popularity_score = self.popularity_scores.get(content_id, 0)
                combined_scores[content_id] += popularity_score * 0.1
            
            # Sort by combined score
            sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Prepare final recommendations
            final_recommendations = []
            for content_id, score in sorted_recs[:n_recommendations]:
                rec = all_recommendations[content_id].copy()
                rec['recommendation_score'] = float(score)
                rec['recommendation_reason'] = 'Hybrid (content + collaborative)'
                final_recommendations.append(rec)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return self.get_popular_recommendations(n_recommendations)
    
    def get_popular_recommendations(self, n_recommendations=20):
        """Get popular content as fallback"""
        try:
            # Sort by popularity scores
            sorted_content = sorted(
                self.popularity_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            recommendations = []
            for content_id, score in sorted_content[:n_recommendations]:
                movie_data = self.movie_metadata.get(content_id)
                if movie_data:
                    movie_data['recommendation_score'] = float(score)
                    movie_data['recommendation_reason'] = 'Popular content'
                    recommendations.append(movie_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in popular recommendations: {e}")
            return []

# Initialize the recommendation engine
rec_engine = MovieRecommendationEngine()

def initialize_models_async():
    """Initialize models in background thread"""
    rec_engine.initialize_models()

# Start model initialization in background
threading.Thread(target=initialize_models_async, daemon=True).start()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': rec_engine.content_similarity_matrix is not None,
        'last_update': rec_engine.last_update
    })

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Main recommendation endpoint"""
    try:
        data = request.get_json()
        
        user_id = data.get('user_id')
        recommendation_type = data.get('recommendation_type', 'hybrid')
        n_recommendations = data.get('n_recommendations', 20)
        
        # User preference data
        user_preferences = {
            'watch_history': data.get('watch_history', []),
            'favorites': data.get('favorites', []),
            'wishlist': data.get('wishlist', [])
        }
        
        # Check if models are loaded
        if rec_engine.content_similarity_matrix is None:
            return jsonify({
                'recommendations': [],
                'message': 'Models are still loading, please try again in a moment',
                'status': 'loading'
            })
        
        # Get recommendations based on type
        if recommendation_type == 'content':
            recommendations = rec_engine.get_content_based_recommendations(
                user_preferences, n_recommendations
            )
        elif recommendation_type == 'collaborative':
            recommendations = rec_engine.get_collaborative_recommendations(
                user_id, n_recommendations
            )
        else:  # hybrid
            recommendations = rec_engine.get_hybrid_recommendations(
                user_id, user_preferences, n_recommendations
            )
        
        # Add diversity to recommendations
        diverse_recommendations = add_diversity(recommendations)
        
        return jsonify({
            'recommendations': diverse_recommendations,
            'total_count': len(diverse_recommendations),
            'recommendation_type': recommendation_type,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {e}")
        return jsonify({
            'error': 'Failed to generate recommendations',
            'recommendations': [],
            'status': 'error'
        }), 500

def add_diversity(recommendations):
    """Add diversity to recommendations by ensuring genre variety"""
    if not recommendations:
        return recommendations
    
    # Group by genres
    genre_groups = defaultdict(list)
    for rec in recommendations:
        genres = rec.get('genre_ids', [])
        if genres:
            primary_genre = genres[0]  # Use first genre as primary
            genre_groups[primary_genre].append(rec)
        else:
            genre_groups['unknown'].append(rec)
    
    # Diversify selection
    diverse_recs = []
    max_per_genre = max(2, len(recommendations) // len(genre_groups))
    
    for genre, recs in genre_groups.items():
        diverse_recs.extend(recs[:max_per_genre])
    
    # Fill remaining slots with highest scored items
    remaining_slots = len(recommendations) - len(diverse_recs)
    if remaining_slots > 0:
        all_remaining = [r for r in recommendations if r not in diverse_recs]
        all_remaining.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)
        diverse_recs.extend(all_remaining[:remaining_slots])
    
    return diverse_recs[:len(recommendations)]

@app.route('/trending', methods=['GET'])
def get_trending():
    """Get trending content"""
    try:
        n_items = request.args.get('n_items', 20, type=int)
        media_type = request.args.get('media_type', 'all')
        
        # Filter by media type if specified
        if media_type != 'all':
            filtered_content = {
                k: v for k, v in rec_engine.movie_metadata.items()
                if v.get('media_type') == media_type
            }
        else:
            filtered_content = rec_engine.movie_metadata
        
        # Sort by popularity scores
        trending = []
        for content_id, score in sorted(
            rec_engine.popularity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if content_id in filtered_content:
                movie_data = filtered_content[content_id].copy()
                movie_data['trending_score'] = float(score)
                trending.append(movie_data)
            
            if len(trending) >= n_items:
                break
        
        return jsonify({
            'trending': trending,
            'total_count': len(trending),
            'media_type': media_type
        })
        
    except Exception as e:
        logger.error(f"Error in trending endpoint: {e}")
        return jsonify({'error': 'Failed to get trending content'}), 500

@app.route('/similar/<content_id>', methods=['GET'])
def get_similar_content(content_id):
    """Get similar content to a specific item"""
    try:
        n_similar = request.args.get('n_similar', 10, type=int)
        
        if rec_engine.content_similarity_matrix is None:
            return jsonify({'error': 'Models not loaded yet'}), 503
        
        # Find content index
        content_indices = {
            cid: idx for idx, cid in enumerate(rec_engine.movie_features_df['content_id'])
        }
        
        if content_id not in content_indices:
            return jsonify({'error': 'Content not found'}), 404
        
        idx = content_indices[content_id]
        
        # Get similarity scores
        sim_scores = list(enumerate(rec_engine.content_similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get similar items (excluding the item itself)
        similar_items = []
        for i, (item_idx, score) in enumerate(sim_scores[1:n_similar+1]):
            similar_content_id = rec_engine.movie_features_df.iloc[item_idx]['content_id']
            movie_data = rec_engine.movie_metadata.get(similar_content_id)
            
            if movie_data:
                movie_data['similarity_score'] = float(score)
                similar_items.append(movie_data)
        
        return jsonify({
            'similar_items': similar_items,
            'total_count': len(similar_items),
            'reference_content_id': content_id
        })
        
    except Exception as e:
        logger.error(f"Error in similar content endpoint: {e}")
        return jsonify({'error': 'Failed to get similar content'}), 500

@app.route('/update-models', methods=['POST'])
def update_models():
    """Manually trigger model update"""
    try:
        # Run model update in background
        threading.Thread(target=rec_engine.initialize_models, daemon=True).start()
        
        return jsonify({
            'message': 'Model update started',
            'status': 'updating'
        })
        
    except Exception as e:
        logger.error(f"Error updating models: {e}")
        return jsonify({'error': 'Failed to start model update'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)