#ml-service/app.py
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
JIKAN_API_URL = 'https://api.jikan.moe/v4'

# Global variables for models and data
content_similarity_matrix = None
tfidf_vectorizer = None
movie_features_df = None
user_item_matrix = None
svd_model = None
scaler = None
movie_metadata = {}
genre_profiles = {}
user_preferences = {}

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
        self.anime_metadata = {}
        
    def initialize_models(self):
        """Initialize and train recommendation models"""
        try:
            logger.info("Initializing ML models...")
            
            # Load movie and anime data
            self.load_movie_data()
            self.load_anime_data()
            
            # Build content-based model
            self.build_content_model()
            
            # Build collaborative filtering model
            self.build_collaborative_model()
            
            # Calculate popularity scores
            self.calculate_popularity_scores()
            
            # Load user preferences
            self.load_user_preferences()
            
            logger.info("ML models initialized successfully")
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def load_movie_data(self):
        """Load movie and TV data from TMDB API"""
        logger.info("Loading movie and TV data from TMDB...")
        
        movies_data = []
        tv_data = []
        
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
                logger.error(f"Error fetching TMDB page {page}: {e}")
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
            show['title'] = show.get('name', show.get('title', ''))
            show['release_date'] = show.get('first_air_date', show.get('release_date', ''))
            all_content.append(show)
            self.movie_metadata[show['content_id']] = show
        
        logger.info(f"Loaded {len(all_content)} items from TMDB")
        return all_content
    
    def load_anime_data(self):
        """Load anime data from Jikan API"""
        logger.info("Loading anime data from Jikan...")
        
        anime_data = []
        
        for page in range(1, 6):  # Get first 5 pages
            try:
                anime_url = f'{JIKAN_API_URL}/top/anime'
                anime_params = {'type': 'tv', 'filter': 'bypopularity', 'page': page, 'limit': 25}
                anime_response = requests.get(anime_url, params=anime_params)
                
                if anime_response.status_code == 200:
                    anime_data.extend(anime_response.json().get('data', []))
                
                time.sleep(1)  # Rate limiting for Jikan API
                
            except Exception as e:
                logger.error(f"Error fetching Jikan page {page}: {e}")
                continue
        
        for anime in anime_data:
            anime['media_type'] = 'anime'
            anime['content_id'] = f"anime_{anime['mal_id']}"
            anime['title'] = anime.get('title', '')
            anime['release_date'] = anime.get('aired', {}).get('from', '')
            anime['genre_ids'] = [g['mal_id'] for g in anime.get('genres', [])]
            anime['genres'] = [g['name'] for g in anime.get('genres', [])]
            anime['overview'] = anime.get('synopsis', '')
            self.movie_metadata[anime['content_id']] = anime
            self.anime_metadata[anime['content_id']] = anime
        
        logger.info(f"Loaded {len(anime_data)} anime items from Jikan")
    
    def load_user_preferences(self):
        """Load user preferences from database"""
        try:
            conn = sqlite3.connect('movie_app.db')
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, preferred_genres, preferred_languages, content_types FROM user_preferences')
            prefs = cursor.fetchall()
            conn.close()
            
            for user_id, genres, languages, content_types in prefs:
                self.user_preferences[user_id] = {
                    'preferred_genres': json.loads(genres) if genres else [],
                    'preferred_languages': json.loads(languages) if languages else [],
                    'content_types': json.loads(content_types) if content_types else []
                }
            
            logger.info(f"Loaded preferences for {len(self.user_preferences)} users")
            
        except Exception as e:
            logger.error(f"Error loading user preferences: {e}")
    
    def build_content_model(self):
        """Build content-based recommendation model"""
        logger.info("Building content-based model...")
        
        try:
            features_data = []
            
            for content_id, metadata in self.movie_metadata.items():
                genre_names = metadata.get('genres', []) or [str(g) for g in metadata.get('genre_ids', [])]
                genres_str = ' '.join(genre_names)
                
                overview = metadata.get('overview', '')
                title = metadata.get('title', '')
                additional_features = []
                
                # Anime-specific features
                if metadata['media_type'] == 'anime':
                    additional_features.extend([
                        metadata.get('status', ''),
                        metadata.get('source', ''),
                        ' '.join([s['name'] for s in metadata.get('studios', [])]),
                        ' '.join([t['name'] for t in metadata.get('themes', [])]),
                        ' '.join([d['name'] for d in metadata.get('demographics', [])])
                    ])
                
                content_text = f"{title} {overview} {genres_str} {' '.join(additional_features)}"
                
                features_data.append({
                    'content_id': content_id,
                    'content_text': content_text,
                    'popularity': metadata.get('popularity', 0),
                    'vote_average': metadata.get('vote_average', 0) or metadata.get('score', 0),
                    'vote_count': metadata.get('vote_count', 0) or metadata.get('scored_by', 0),
                    'genres': genre_names,
                    'media_type': metadata['media_type']
                })
            
            self.movie_features_df = pd.DataFrame(features_data)
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.movie_features_df['content_text'].fillna('')
            )
            
            self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
            
            logger.info("Content-based model built successfully")
            
        except Exception as e:
            logger.error(f"Error building content model: {e}")
    
    def build_collaborative_model(self):
        """Build collaborative filtering model using matrix factorization"""
        logger.info("Building collaborative filtering model...")
        
        try:
            user_interactions = self.load_user_interactions()
            
            if user_interactions.empty:
                user_interactions = self.generate_synthetic_interactions()
            
            self.user_item_matrix = pd.pivot_table(
                user_interactions,
                index='user_id',
                columns='content_id',
                values='rating',
                fill_value=0
            )
            
            self.svd_model = TruncatedSVD(n_components=100, random_state=42)
            user_factors = self.svd_model.fit_transform(self.user_item_matrix)
            
            self.user_factors = user_factors
            self.item_factors = self.svd_model.components_.T
            
            logger.info("Collaborative filtering model built successfully")
            
        except Exception as e:
            logger.error(f"Error building collaborative model: {e}")
    
    def load_user_interactions(self):
        """Load user interactions from database"""
        try:
            conn = sqlite3.connect('movie_app.db')
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, movie_id, movie_type, rating FROM watch_history WHERE rating IS NOT NULL')
            interactions = cursor.fetchall()
            conn.close()
            
            interaction_data = []
            for user_id, movie_id, movie_type, rating in interactions:
                content_id = f"{movie_type}_{movie_id}"
                if content_id in self.movie_metadata:
                    interaction_data.append({
                        'user_id': user_id,
                        'content_id': content_id,
                        'rating': rating / 2  # Scale 1-10 to 1-5
                    })
            
            return pd.DataFrame(interaction_data)
            
        except Exception as e:
            logger.error(f"Error loading user interactions: {e}")
            return pd.DataFrame()
    
    def generate_synthetic_interactions(self):
        """Generate synthetic user interactions"""
        interactions = []
        content_ids = list(self.movie_metadata.keys())
        
        for user_id in range(1, 1001):
            n_interactions = np.random.randint(20, 100)
            popularities = [self.movie_metadata[cid].get('popularity', 0) for cid in content_ids]
            popularities = np.array(popularities)
            popularities = popularities / (popularities.sum() or 1)
            
            selected_items = np.random.choice(
                content_ids, 
                size=n_interactions, 
                replace=False, 
                p=popularities
            )
            
            for content_id in selected_items:
                base_rating = self.movie_metadata[content_id].get('vote_average', 5) or self.movie_metadata[content_id].get('score', 5)
                base_rating = base_rating / 2
                user_bias = np.random.normal(0, 0.5)
                rating = max(1, min(5, base_rating + user_bias))
                
                interactions.append({
                    'user_id': user_id,
                    'content_id': content_id,
                    'rating': rating
                })
        
        return pd.DataFrame(interactions)
    
    def calculate_popularity_scores(self):
        """Calculate popularity scores for content"""
        current_time = datetime.now()
        
        for content_id, metadata in self.movie_metadata.items():
            base_popularity = metadata.get('popularity', 0) or metadata.get('members', 0) / 1000
            release_date = metadata.get('release_date', '') or metadata.get('aired', {}).get('from', '')
            release_boost = 1.0
            
            if release_date:
                try:
                    release_dt = datetime.strptime(release_date[:10], '%Y-%m-%d')
                    days_since_release = (current_time - release_dt).days
                    if days_since_release < 730:  # Boost for content within 2 years
                        release_boost = 2.0 - (days_since_release / 730) * 1.0
                except:
                    pass
            
            vote_average = metadata.get('vote_average', 0) or metadata.get('score', 0)
            vote_count = metadata.get('vote_count', 0) or metadata.get('scored_by', 0)
            quality_score = (vote_average / 10) * min(1, vote_count / 1000)
            
            # Anime-specific popularity boost
            anime_boost = 1.2 if metadata['media_type'] == 'anime' else 1.0
            
            final_score = base_popularity * release_boost * (1 + quality_score) * anime_boost
            self.popularity_scores[content_id] = final_score
    
    def get_content_based_recommendations(self, user_preferences, n_recommendations=20):
        """Get content-based recommendations"""
        try:
            if self.content_similarity_matrix is None:
                return self.get_popular_recommendations(n_recommendations)
            
            preferred_content_ids = []
            user_genres = user_preferences.get('preferred_genres', [])
            
            # Process watch history and favorites
            for item in user_preferences.get('watch_history', []):
                movie_id, movie_type, rating = item[0], item[1], item[2]
                if rating and rating >= 7:  # Consider highly rated items (7/10 or higher)
                    content_id = f"{movie_type}_{movie_id}"
                    if content_id in self.movie_metadata:
                        preferred_content_ids.append(content_id)
            
            for item in user_preferences.get('favorites', []):
                movie_id, movie_type = item[0], item[1]
                content_id = f"{movie_type}_{movie_id}"
                if content_id in self.movie_metadata:
                    preferred_content_ids.append(content_id)
            
            if not preferred_content_ids:
                return self.get_popular_recommendations(n_recommendations, user_genres)
            
            content_indices = {cid: idx for idx, cid in enumerate(self.movie_features_df['content_id'])}
            similarity_scores = []
            
            for content_id in preferred_content_ids:
                if content_id in content_indices:
                    idx = content_indices[content_id]
                    scores = list(enumerate(self.content_similarity_matrix[idx]))
                    similarity_scores.extend(scores)
            
            score_dict = defaultdict(float)
            for idx, score in similarity_scores:
                score_dict[idx] += score
            
            # Boost scores based on user preferences
            for idx in score_dict:
                content_id = self.movie_features_df.iloc[idx]['content_id']
                content_genres = self.movie_features_df.iloc[idx]['genres']
                genre_overlap = len(set(content_genres) & set(user_genres))
                score_dict[idx] *= (1 + 0.2 * genre_overlap)
            
            sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            seen_content_ids = set(preferred_content_ids)
            
            for idx, score in sorted_scores:
                if len(recommendations) >= n_recommendations:
                    break
                
                content_id = self.movie_features_df.iloc[idx]['content_id']
                if content_id not in seen_content_ids:
                    movie_data = self.movie_metadata.get(content_id).copy()
                    movie_data['recommendation_score'] = float(score)
                    movie_data['recommendation_reason'] = 'Content similarity'
                    recommendations.append(movie_data)
                    seen_content_ids.add(content_id)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {e}")
            return self.get_popular_recommendations(n_recommendations)
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=20):
        """Get collaborative filtering recommendations"""
        try:
            if self.svd_model is None or self.user_item_matrix is None:
                return self.get_popular_recommendations(n_recommendations)
            
            if user_id not in self.user_item_matrix.index:
                return self.get_popular_recommendations(n_recommendations)
            
            user_idx = list(self.user_item_matrix.index).index(user_id)
            user_vector = self.user_factors[user_idx]
            
            predicted_ratings = np.dot(user_vector, self.item_factors.T)
            
            user_ratings = self.user_item_matrix.loc[user_id]
            unrated_items = user_ratings[user_ratings == 0].index
            
            item_indices = {item: idx for idx, item in enumerate(self.user_item_matrix.columns)}
            predictions = []
            
            for item in unrated_items:
                if item in item_indices and item in self.movie_metadata:
                    item_idx = item_indices[item]
                    predicted_rating = predicted_ratings[item_idx]
                    predictions.append((item, predicted_rating))
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for content_id, rating in predictions[:n_recommendations]:
                movie_data = self.movie_metadata.get(content_id).copy()
                movie_data['recommendation_score'] = float(rating)
                movie_data['recommendation_reason'] = 'Collaborative filtering'
                recommendations.append(movie_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return self.get_popular_recommendations(n_recommendations)
    
    def get_hybrid_recommendations(self, user_id, user_preferences, n_recommendations=20):
        """Get hybrid recommendations combining content and collaborative filtering"""
        try:
            content_recs = self.get_content_based_recommendations(user_preferences, n_recommendations)
            collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations)
            
            combined_scores = defaultdict(float)
            all_recommendations = {}
            
            # Weight content-based (50%), collaborative (30%), and popularity (20%)
            for rec in content_recs:
                content_id = rec['content_id']
                combined_scores[content_id] += rec.get('recommendation_score', 0) * 0.5
                all_recommendations[content_id] = rec
            
            for rec in collab_recs:
                content_id = rec['content_id']
                combined_scores[content_id] += rec.get('recommendation_score', 0) * 0.3
                if content_id not in all_recommendations:
                    all_recommendations[content_id] = rec
            
            # Add popularity boost and user preference boost
            user_genres = user_preferences.get('preferred_genres', [])
            user_content_types = user_preferences.get('content_types', [])
            
            for content_id in combined_scores:
                popularity_score = self.popularity_scores.get(content_id, 0)
                combined_scores[content_id] += popularity_score * 0.2
                
                # Boost based on preferred genres and content types
                content = self.movie_metadata.get(content_id)
                if content:
                    genre_overlap = len(set(content.get('genres', []) or content.get('genre_ids', [])) & set(user_genres))
                    combined_scores[content_id] *= (1 + 0.15 * genre_overlap)
                    if content['media_type'] in user_content_types:
                        combined_scores[content_id] *= 1.1
            
            sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            final_recommendations = []
            for content_id, score in sorted_recs[:n_recommendations]:
                rec = all_recommendations[content_id].copy()
                rec['recommendation_score'] = float(score)
                rec['recommendation_reason'] = 'Hybrid (content + collaborative + popularity)'
                final_recommendations.append(rec)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return self.get_popular_recommendations(n_recommendations)
    
    def get_popular_recommendations(self, n_recommendations=20, preferred_genres=None):
        """Get popular content with genre filtering"""
        try:
            sorted_content = sorted(
                self.popularity_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            recommendations = []
            for content_id, score in sorted_content:
                movie_data = self.movie_metadata.get(content_id)
                if movie_data:
                    if preferred_genres:
                        content_genres = movie_data.get('genres', []) or movie_data.get('genre_ids', [])
                        if not any(g in preferred_genres for g in content_genres):
                            continue
                    movie_data = movie_data.copy()
                    movie_data['recommendation_score'] = float(score)
                    movie_data['recommendation_reason'] = 'Popular content'
                    recommendations.append(movie_data)
                    if len(recommendations) >= n_recommendations:
                        break
            
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
        
        user_preferences = {
            'watch_history': data.get('watch_history', []),
            'favorites': data.get('favorites', []),
            'wishlist': data.get('wishlist', []),
            'preferred_genres': rec_engine.user_preferences.get(user_id, {}).get('preferred_genres', []),
            'content_types': rec_engine.user_preferences.get(user_id, {}).get('content_types', ['movie', 'tv', 'anime'])
        }
        
        if rec_engine.content_similarity_matrix is None:
            return jsonify({
                'recommendations': [],
                'message': 'Models are still loading, please try again in a moment',
                'status': 'loading'
            })
        
        if recommendation_type == 'content':
            recommendations = rec_engine.get_content_based_recommendations(
                user_preferences, n_recommendations
            )
        elif recommendation_type == 'collaborative':
            recommendations = rec_engine.get_collaborative_recommendations(
                user_id, n_recommendations
            )
        else:
            recommendations = rec_engine.get_hybrid_recommendations(
                user_id, user_preferences, n_recommendations
            )
        
        diverse_recommendations = add_diversity(recommendations, user_preferences)
        
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

def add_diversity(recommendations, user_preferences):
    """Add diversity to recommendations by ensuring genre and media type variety"""
    if not recommendations:
        return recommendations
    
    genre_groups = defaultdict(list)
    media_type_groups = defaultdict(list)
    
    for rec in recommendations:
        genres = rec.get('genres', []) or rec.get('genre_ids', [])
        primary_genre = genres[0] if genres else 'unknown'
        media_type = rec.get('media_type', 'movie')
        
        genre_groups[primary_genre].append(rec)
        media_type_groups[media_type].append(rec)
    
    # Ensure balanced genre and media type distribution
    max_per_genre = max(3, len(recommendations) // (len(genre_groups) or 1))
    max_per_media_type = max(3, len(recommendations) // (len(media_type_groups) or 1))
    
    diverse_recs = []
    genre_counts = Counter()
    media_type_counts = Counter()
    
    # Prioritize user-preferred genres and content types
    preferred_genres = user_preferences.get('preferred_genres', [])
    preferred_content_types = user_preferences.get('content_types', ['movie', 'tv', 'anime'])
    
    for genre, recs in sorted(genre_groups.items(), key=lambda x: -len(set(x[0]) & set(preferred_genres))):
        selected = sorted(recs, key=lambda x: x.get('recommendation_score', 0), reverse=True)[:max_per_genre]
        for rec in selected:
            if genre_counts[genre] < max_per_genre and media_type_counts[rec['media_type']] < max_per_media_type:
                diverse_recs.append(rec)
                genre_counts[genre] += 1
                media_type_counts[rec['media_type']] += 1
    
    # Fill remaining slots
    remaining_slots = len(recommendations) - len(diverse_recs)
    if remaining_slots > 0:
        all_remaining = [r for r in recommendations if r not in diverse_recs]
        all_remaining.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)
        
        for rec in all_remaining:
            if len(diverse_recs) >= len(recommendations):
                break
            media_type = rec['media_type']
            genres = rec.get('genres', []) or rec.get('genre_ids', [])
            primary_genre = genres[0] if genres else 'unknown'
            
            if media_type_counts[media_type] < max_per_media_type and genre_counts[primary_genre] < max_per_genre:
                diverse_recs.append(rec)
                genre_counts[primary_genre] += 1
                media_type_counts[media_type] += 1
    
    # Boost recommendations matching preferred content types
    for rec in diverse_recs:
        if rec['media_type'] in preferred_content_types:
            rec['recommendation_score'] = rec.get('recommendation_score', 0) * 1.1
    
    return diverse_recs[:len(recommendations)]

@app.route('/trending', methods=['GET'])
def get_trending():
    """Get trending content"""
    try:
        n_items = request.args.get('n_items', 20, type=int)
        media_type = request.args.get('media_type', 'all')
        
        filtered_content = {
            k: v for k, v in rec_engine.movie_metadata.items()
            if media_type == 'all' or v.get('media_type') == media_type
        }
        
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
        logger.error(f"Error in trending endpoint déclenche: {e}")
        return jsonify({'error': 'Failed to get trending content'}), 500

@app.route('/similar/<content_id>', methods=['GET'])
def get_similar_content(content_id):
    """Get similar content to a specific item"""
    try:
        n_similar = request.args.get('n_similar', 10, type=int)
        
        if rec_engine.content_similarity_matrix is None:
            return jsonify({'error': 'Models not loaded yet'}), 503
        
        content_indices = {
            cid: idx for idx, cid in enumerate(rec_engine.movie_features_df['content_id'])
        }
        
        if content_id not in content_indices:
            return jsonify({'error': 'Content not found'}), 404
        
        idx = content_indices[content_id]
        sim_scores = list(enumerate(rec_engine.content_similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        similar_items = []
        for i, (item_idx, score) in enumerate(sim_scores[1:n_similar+1]):
            similar_content_id = rec_engine.movie_features_df.iloc[item_idx]['content_id']
            movie_data = rec_engine.movie_metadata.get(similar_content_id)
            
            if movie_data:
                movie_data = movie_data.copy()
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