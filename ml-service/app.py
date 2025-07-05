#ml-service/app.py
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import joblib
import threading
import time
from collections import defaultdict
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://localhost/moviedb')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# ML Models storage
models = {
    'user_similarity': None,
    'content_similarity': None,
    'svd_model': None,
    'tfidf_vectorizer': None,
    'content_vectors': None,
    'user_profiles': None,
    'last_trained': None
}

# Cache for recommendations
recommendation_cache = {}
cache_lock = threading.Lock()

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DATABASE_URL)

def load_user_data():
    """Load user interaction data"""
    conn = get_db_connection()
    
    try:
        # Load user interactions
        user_df = pd.read_sql('''
            SELECT ui.user_id, ui.content_id, ui.rating, ui.interaction_type,
                   c.title, c.genres, c.overview, c.type, c.region
            FROM user_interactions ui
            JOIN content c ON ui.content_id = c.id
            WHERE ui.rating IS NOT NULL
        ''', conn)
        
        # Load content data
        content_df = pd.read_sql('''
            SELECT id, title, overview, genres, type, region, rating, vote_count
            FROM content
        ''', conn)
        
        return user_df, content_df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()
    finally:
        conn.close()

def build_user_item_matrix(user_df):
    """Build user-item interaction matrix"""
    try:
        # Create pivot table
        user_item_matrix = user_df.pivot_table(
            index='user_id', 
            columns='content_id', 
            values='rating', 
            fill_value=0
        )
        
        return user_item_matrix
        
    except Exception as e:
        logger.error(f"Error building user-item matrix: {e}")
        return pd.DataFrame()

def compute_user_similarity(user_item_matrix):
    """Compute user similarity matrix"""
    try:
        # Use cosine similarity
        user_similarity = cosine_similarity(user_item_matrix)
        
        # Convert to DataFrame for easier handling
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_item_matrix.index,
            columns=user_item_matrix.index
        )
        
        return user_similarity_df
        
    except Exception as e:
        logger.error(f"Error computing user similarity: {e}")
        return pd.DataFrame()

def build_content_vectors(content_df):
    """Build content feature vectors"""
    try:
        # Prepare text features
        content_text = []
        for _, row in content_df.iterrows():
            text_features = []
            
            # Add title
            if pd.notna(row['title']):
                text_features.append(row['title'])
            
            # Add overview
            if pd.notna(row['overview']):
                text_features.append(row['overview'])
            
            # Add genres
            if pd.notna(row['genres']) and row['genres']:
                if isinstance(row['genres'], str):
                    genres = json.loads(row['genres'])
                else:
                    genres = row['genres']
                
                if isinstance(genres, list):
                    genre_names = [g.get('name', '') for g in genres if isinstance(g, dict)]
                    text_features.extend(genre_names)
            
            # Add type and region
            if pd.notna(row['type']):
                text_features.append(row['type'])
            if pd.notna(row['region']):
                text_features.append(row['region'])
            
            content_text.append(' '.join(text_features))
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        content_vectors = vectorizer.fit_transform(content_text)
        
        return vectorizer, content_vectors
        
    except Exception as e:
        logger.error(f"Error building content vectors: {e}")
        return None, None

def train_svd_model(user_item_matrix):
    """Train SVD model for matrix factorization"""
    try:
        # Fill NaN values with 0 and convert to dense matrix
        matrix = user_item_matrix.fillna(0).values
        
        # Standardize the data
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(matrix)
        
        # Train SVD model
        svd = TruncatedSVD(n_components=50, random_state=42)
        svd.fit(matrix_scaled)
        
        return svd, scaler
        
    except Exception as e:
        logger.error(f"Error training SVD model: {e}")
        return None, None

def build_user_profiles(user_df):
    """Build user preference profiles"""
    try:
        user_profiles = {}
        
        for user_id in user_df['user_id'].unique():
            user_data = user_df[user_df['user_id'] == user_id]
            
            profile = {
                'favorite_genres': defaultdict(float),
                'preferred_types': defaultdict(float),
                'preferred_regions': defaultdict(float),
                'avg_rating': user_data['rating'].mean(),
                'total_interactions': len(user_data)
            }
            
            # Calculate genre preferences
            for _, row in user_data.iterrows():
                if pd.notna(row['genres']) and row['genres']:
                    try:
                        if isinstance(row['genres'], str):
                            genres = json.loads(row['genres'])
                        else:
                            genres = row['genres']
                        
                        if isinstance(genres, list):
                            for genre in genres:
                                if isinstance(genre, dict) and 'name' in genre:
                                    weight = row['rating'] / 5.0
                                    profile['favorite_genres'][genre['name']] += weight
                    except:
                        pass
                
                # Type preferences
                if pd.notna(row['type']):
                    weight = row['rating'] / 5.0
                    profile['preferred_types'][row['type']] += weight
                
                # Region preferences
                if pd.notna(row['region']):
                    weight = row['rating'] / 5.0
                    profile['preferred_regions'][row['region']] += weight
            
            user_profiles[user_id] = profile
        
        return user_profiles
        
    except Exception as e:
        logger.error(f"Error building user profiles: {e}")
        return {}

def get_collaborative_recommendations(user_id, user_similarity_df, user_item_matrix, n_recommendations=20):
    """Get collaborative filtering recommendations"""
    try:
        if user_id not in user_similarity_df.index:
            return []
        
        # Get similar users
        similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False)[1:11]
        
        # Get items rated by similar users
        user_items = user_item_matrix.loc[user_id]
        unrated_items = user_items[user_items == 0].index
        
        recommendations = defaultdict(float)
        
        for similar_user_id, similarity in similar_users.items():
            if similarity > 0.1:  # Threshold for similarity
                similar_user_items = user_item_matrix.loc[similar_user_id]
                
                for item_id in unrated_items:
                    if similar_user_items[item_id] > 0:
                        recommendations[item_id] += similarity * similar_user_items[item_id]
        
        # Sort and return top recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [{'content_id': int(item_id), 'score': float(score)} for item_id, score in sorted_recs[:n_recommendations]]
        
    except Exception as e:
        logger.error(f"Error in collaborative recommendations: {e}")
        return []

def get_content_based_recommendations(user_id, content_vectors, user_profiles, content_df, n_recommendations=20):
    """Get content-based recommendations"""
    try:
        if user_id not in user_profiles:
            return []
        
        profile = user_profiles[user_id]
        recommendations = []
        
        # Score content based on user preferences
        for idx, row in content_df.iterrows():
            score = 0
            
            # Genre matching
            if pd.notna(row['genres']) and row['genres']:
                try:
                    if isinstance(row['genres'], str):
                        genres = json.loads(row['genres'])
                    else:
                        genres = row['genres']
                    
                    if isinstance(genres, list):
                        for genre in genres:
                            if isinstance(genre, dict) and 'name' in genre:
                                genre_name = genre['name']
                                if genre_name in profile['favorite_genres']:
                                    score += profile['favorite_genres'][genre_name]
                except:
                    pass
            
            # Type matching
            if pd.notna(row['type']) and row['type'] in profile['preferred_types']:
                score += profile['preferred_types'][row['type']]
            
            # Region matching
            if pd.notna(row['region']) and row['region'] in profile['preferred_regions']:
                score += profile['preferred_regions'][row['region']]
            
            # Rating boost
            if pd.notna(row['rating']) and row['rating'] > 7:
                score += 0.5
            
            if score > 0:
                recommendations.append({
                    'content_id': int(row['id']),
                    'score': float(score)
                })
        
        # Sort and return top recommendations
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:n_recommendations]
        
    except Exception as e:
        logger.error(f"Error in content-based recommendations: {e}")
        return []

def get_hybrid_recommendations(user_id, n_recommendations=20):
    """Get hybrid recommendations combining multiple approaches"""
    try:
        collaborative_recs = get_collaborative_recommendations(
            user_id, models['user_similarity'], models['user_item_matrix']
        )
        
        content_recs = get_content_based_recommendations(
            user_id, models['content_vectors'], models['user_profiles'], models['content_df']
        )
        
        # Combine recommendations with weights
        hybrid_scores = defaultdict(float)
        
        # Weight collaborative filtering
        for rec in collaborative_recs:
            hybrid_scores[rec['content_id']] += 0.6 * rec['score']
        
        # Weight content-based filtering
        for rec in content_recs:
            hybrid_scores[rec['content_id']] += 0.4 * rec['score']
        
        # Sort and return
        sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return [{'content_id': int(item_id), 'score': float(score)} for item_id, score in sorted_recs[:n_recommendations]]
        
    except Exception as e:
        logger.error(f"Error in hybrid recommendations: {e}")
        return []

def train_models():
    """Train all ML models"""
    try:
        logger.info("Starting model training...")
        
        # Load data
        user_df, content_df = load_user_data()
        
        if user_df.empty or content_df.empty:
            logger.warning("No data available for training")
            return
        
        # Build user-item matrix
        user_item_matrix = build_user_item_matrix(user_df)
        
        # Compute user similarity
        user_similarity = compute_user_similarity(user_item_matrix)
        
        # Build content vectors
        vectorizer, content_vectors = build_content_vectors(content_df)
        
        # Train SVD model
        svd_model, scaler = train_svd_model(user_item_matrix)
        
        # Build user profiles
        user_profiles = build_user_profiles(user_df)
        
        # Update global models
        models.update({
            'user_similarity': user_similarity,
            'content_similarity': cosine_similarity(content_vectors) if content_vectors is not None else None,
            'svd_model': svd_model,
            'tfidf_vectorizer': vectorizer,
            'content_vectors': content_vectors,
            'user_profiles': user_profiles,
            'user_item_matrix': user_item_matrix,
            'content_df': content_df,
            'last_trained': datetime.now()
        })
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error training models: {e}")

def should_retrain():
    """Check if models should be retrained"""
    if models['last_trained'] is None:
        return True
    
    # Retrain every 24 hours
    time_since_training = datetime.now() - models['last_trained']
    return time_since_training > timedelta(hours=24)

# Background training thread
def background_training():
    """Background thread for periodic model training"""
    while True:
        try:
            if should_retrain():
                train_models()
            time.sleep(3600)  # Check every hour
        except Exception as e:
            logger.error(f"Background training error: {e}")
            time.sleep(3600)

# Start background training
training_thread = threading.Thread(target=background_training, daemon=True)
training_thread.start()

# API Routes
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    """Get personalized recommendations for a user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Check cache first
        cache_key = f"user_{user_id}"
        with cache_lock:
            if cache_key in recommendation_cache:
                cached_data = recommendation_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < timedelta(hours=1):
                    return jsonify(cached_data['recommendations']), 200
        
        # Generate recommendations
        if models['user_similarity'] is None:
            train_models()
        
        if models['user_similarity'] is None:
            return jsonify({'recommendations': []}), 200
        
        # Get hybrid recommendations
        recommendations = get_hybrid_recommendations(user_id)
        
        # Cache the results
        with cache_lock:
            recommendation_cache[cache_key] = {
                'recommendations': {'recommendations': recommendations},
                'timestamp': datetime.now()
            }
        
        return jsonify({'recommendations': recommendations}), 200
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return jsonify({'error': 'Failed to generate recommendations'}), 500

@app.route('/similar-content', methods=['POST'])
def get_similar_content():
    """Get content similar to a given content item"""
    try:
        data = request.get_json()
        content_id = data.get('content_id')
        n_similar = data.get('n_similar', 10)
        
        if not content_id:
            return jsonify({'error': 'Content ID is required'}), 400
        
        if models['content_similarity'] is None:
            train_models()
        
        if models['content_similarity'] is None:
            return jsonify({'similar_content': []}), 200
        
        # Find similar content using content similarity matrix
        content_df = models['content_df']
        content_idx = content_df[content_df['id'] == content_id].index
        
        if len(content_idx) == 0:
            return jsonify({'similar_content': []}), 200
        
        idx = content_idx[0]
        similarity_scores = models['content_similarity'][idx]
        
        # Get top similar content
        similar_indices = similarity_scores.argsort()[-n_similar-1:-1][::-1]
        
        similar_content = []
        for i in similar_indices:
            if i < len(content_df):
                similar_content.append({
                    'content_id': int(content_df.iloc[i]['id']),
                    'similarity_score': float(similarity_scores[i])
                })
        
        return jsonify({'similar_content': similar_content}), 200
        
    except Exception as e:
        logger.error(f"Similar content error: {e}")
        return jsonify({'error': 'Failed to find similar content'}), 500

@app.route('/train', methods=['POST'])
def trigger_training():
    """Manually trigger model training"""
    try:
        train_models()
        return jsonify({'message': 'Model training completed'}), 200
    except Exception as e:
        logger.error(f"Training trigger error: {e}")
        return jsonify({'error': 'Training failed'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models['last_trained'] is not None,
        'last_trained': models['last_trained'].isoformat() if models['last_trained'] else None
    }), 200

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get ML service statistics"""
    try:
        stats = {
            'models_loaded': models['last_trained'] is not None,
            'last_trained': models['last_trained'].isoformat() if models['last_trained'] else None,
            'cache_size': len(recommendation_cache),
            'user_similarity_shape': models['user_similarity'].shape if models['user_similarity'] is not None else None,
            'content_vectors_shape': models['content_vectors'].shape if models['content_vectors'] is not None else None,
            'user_profiles_count': len(models['user_profiles']) if models['user_profiles'] else 0
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': 'Failed to get stats'}), 500

# Initialize models on startup
@app.before_first_request
def initialize_models():
    """Initialize ML models on startup"""
    try:
        train_models()
    except Exception as e:
        logger.error(f"Model initialization error: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)